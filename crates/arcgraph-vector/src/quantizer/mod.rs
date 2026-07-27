//! Quantizer state machine + per-encoding codec primitives.
//!
//! Per ADR-035 D-4, every (tenant, index) arena carries a
//! [`QuantizerState`] tag that drives encode / decode dispatch:
//!
//! - [`QuantizerState::None`] — raw f32 storage; no codebook.
//! - [`QuantizerState::Sq8`] — per-dim `(scale, bias)` codebook
//!   trained on a representative sample (1 % of collection or
//!   1 M vectors, whichever smaller; per ADR-035 §3.3).
//! - [`QuantizerState::Binary`] — sign function (no training).
//! - [`QuantizerState::RaBitQ`] — ADR-209 centroid + rotation
//!   codebook for standalone RaBitQ payloads.
//!
//! ## Slice E.1 surface (this module dir)
//!
//! - [`sq8`][mod@sq8]: [`Sq8Trainer`] (per-dim min/max scan) +
//!   [`Sq8Codebook`] (encode / decode).
//! - [`binary`][mod@binary]: [`binary_encode`] (sign function
//!   pack) + [`binary_decode`] (sign restore for testing /
//!   reference).
//! - [`rabitq`][mod@rabitq]: [`RaBitQTrainer`] + [`RaBitQCodebook`]
//!   standalone codec (no index wiring in slice 1).
//! - [`dispatch`][mod@dispatch]: [`auto_quantizer_for_collection`]
//!   (the ADR-035 D-4 + Q3 ratification 10 M auto-quantize
//!   threshold).
//!
//! Training itself runs on the Tokio background pool per ADR-002.
//! E.1 ships only the standalone primitives — no index, arena, or
//! rescore wiring (that lives in E.2 / E.3 / F.1).
//!
//! ## OQ-V4 re-encode contract (per ADR-035 §5.2 step 6)
//!
//! Vectors live in EITHER a raw-f32 staging arena OR a quantized
//! arena, gated by a per-vector **published flag**. The re-encode
//! pass (training-completion background pass) walks staging,
//! computes quantized representations, atomic-stores the published
//! flag, then removes from staging. Readers consult the flag to
//! decide which arena to read; there is no moment when a vector is
//! in neither, and the flag boundary survives crash / snapshot /
//! replay because it lives in the arena snapshot itself.
//!
//! E.1's role in this contract: provide the trainer that produces
//! the codebook, and the codec that the re-encode pass invokes
//! per-vector. The actual staging-vs-quantized arena split lives
//! in Slice F.1 (per-tenant arena routing); the published-flag
//! pattern is wired in Slice G.4 (staging arena commit). E.1 keeps
//! the codec purely standalone — `encode(&[f32]) -> Vec<u8>` is
//! free of arena state by design, so the re-encode pass can call
//! it per-vector without holding any arena lock.

use serde::{Deserialize, Serialize};

use crate::error::VectorIndexError;

pub mod binary;
pub mod dispatch;
pub mod rabitq;
pub mod sq8;

pub use binary::{binary_decode, binary_encode, binary_encode_aligned};
pub use dispatch::auto_quantizer_for_collection;
pub use rabitq::{
    RABITQ_FASTSCAN_QUERY_BITS, RaBitQCodebook, RaBitQFastScanQuery, RaBitQParams, RaBitQPayload,
    RaBitQQuery, RaBitQTrainer, estimate_ip_unit, estimate_l2_sq,
};
pub use sq8::{Sq8Codebook, Sq8Trainer};

/// Per-dimension SQ8 quantizer parameters.
///
/// At Slice A this was a placeholder; Slice E.1 populates it via
/// [`Sq8Trainer`] (see [`mod@sq8`]). The shape is preserved across
/// the Slice A → E.1 transition so the foundation API is stable.
///
/// The dimension is `scale.len()` (== `bias.len()` by
/// constructor invariant); the struct does not carry a separate
/// `dim` field because that would require keeping it in sync
/// with the vector lengths after deserialization.
///
/// ## Encoding convention (per ADR-035 §3.3, refined by #116)
///
/// Encoded as **i8 in `-128..=127`** (kernel-native). The
/// trainer fits `scale = (max - min) / 255.0` and `bias = min`;
/// the codec applies a centering shift inline so the output
/// lands directly in the range simsimd's signed integer SIMD
/// kernels read:
/// - encode: `q[d] = (round((x[d] - bias[d]) / scale[d]) - 128.0)`
///   `.clamp(-128, 127) as i8`
/// - decode: `x[d] ≈ (q[d] as f32 + 128.0) * scale[d] + bias[d]`
///
/// Earlier ADR-035 §3.3 wording specified `u8 in 0..=255`; #116
/// supersedes for the codec / arena boundary so the kernel byte
/// width matches the storage byte width with no per-read
/// translation. The `(scale, bias)` codebook itself is unchanged
/// — production code that consumed `Sq8Params` directly
/// (snapshot serializer, catalog) keeps the same shape.
///
/// Stored alongside the arena in the ARCV snapshot
/// (`arena-{tenant}-{index}-{lsn}.snap`) per
/// `docs/design/vector-storage-layout.md` §10.3.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sq8Params {
    /// Per-dimension scale: encode as
    /// `q[i] = round((x[i] - bias[i]) / scale[i])` clamped to
    /// `0..=255`. Length equals `bias.len()` by constructor
    /// invariant.
    pub scale: Vec<f32>,

    /// Per-dimension bias (the per-dim min observed during
    /// training). Length equals `scale.len()` by constructor
    /// invariant.
    pub bias: Vec<f32>,
}

impl Sq8Params {
    /// Construct from explicit per-dim scale + bias.
    ///
    /// Returns [`VectorIndexError::DimensionMismatch`] if any of
    /// the following defensive checks fail (issue #104):
    ///
    /// - `scale.len() != bias.len()` — shape mismatch; the error
    ///   surfaces `expected = scale.len()`, `got = bias.len()`.
    /// - `scale.is_empty()` — degenerate zero-dim codebook;
    ///   reported as `expected = 0, got = 0`.
    /// - any `scale[i]` is non-finite (NaN, +∞, −∞) or zero —
    ///   would divide-by-zero on encode (per ADR-035 §9.3 the
    ///   trainer collapses constant dimensions to a sentinel
    ///   `scale = 1.0` instead of the literal zero; this
    ///   constructor refuses to accept the raw zero defensively
    ///   in case the trainer ever skips its fallback).
    /// - any `bias[i]` is non-finite — would propagate NaN
    ///   through the decode pipeline.
    ///
    /// All non-`DimensionMismatch` invalidity (NaN scale, zero
    /// scale, NaN bias) is reported as
    /// `DimensionMismatch { expected: dim, got: dim }` — the sole
    /// `VectorIndexError` variant available at the codec surface
    /// for "this codebook is unfit for use." The trainer is the
    /// production producer of `Sq8Params` and never emits these
    /// cases by construction (see [`Sq8Trainer::train`]); a
    /// match-equal error surface signals "the constructor's
    /// defense-in-depth caught a corrupted or hand-built codebook."
    ///
    /// # Errors
    ///
    /// - [`VectorIndexError::DimensionMismatch`] for any of the
    ///   conditions enumerated above.
    pub fn try_new(scale: Vec<f32>, bias: Vec<f32>) -> Result<Self, VectorIndexError> {
        if scale.len() != bias.len() {
            return Err(VectorIndexError::DimensionMismatch {
                expected: scale.len(),
                got: bias.len(),
            });
        }
        let dim = scale.len();
        if dim == 0 {
            return Err(VectorIndexError::DimensionMismatch {
                expected: 0,
                got: 0,
            });
        }
        if !scale.iter().all(|s| s.is_finite() && *s != 0.0) {
            return Err(VectorIndexError::DimensionMismatch {
                expected: dim,
                got: dim,
            });
        }
        if !bias.iter().all(|b| b.is_finite()) {
            return Err(VectorIndexError::DimensionMismatch {
                expected: dim,
                got: dim,
            });
        }
        Ok(Self { scale, bias })
    }

    /// Vector dimension this codebook was trained for. Derived
    /// from `scale.len()` (which equals `bias.len()` by
    /// constructor invariant).
    #[inline]
    #[must_use]
    pub fn dim(&self) -> usize {
        self.scale.len()
    }
}

/// Per-arena quantizer state. Stored on the arena's catalog
/// record; consulted at every encode / decode / search dispatch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum QuantizerState {
    /// Raw f32 — no quantization, no rescore needed.
    None,

    /// Trained SQ8 codebook (per-dim scale + bias). Encoded
    /// vectors are `i8` per dimension (-128..=127 per #116
    /// closure of the §3.3 codec/kernel byte-width split);
    /// decode is `x[i] = (q[i] as f32 + 128.0) * scale[i] + bias[i]`.
    Sq8 { params: Sq8Params },

    /// Binary (sign function): bit `i` = `(f32_vec[i] > 0)`.
    /// No training, no codebook. Hamming distance is the natural
    /// metric.
    Binary,

    /// RaBitQ centroid + explicit row-major orthogonal rotation.
    /// Slice 1 exposes the standalone codec only; index-side
    /// surfaces reject this variant until ADR-209 slice 2.
    RaBitQ { params: RaBitQParams },
}

impl QuantizerState {
    /// Whether this quantizer is lossy enough that rescore (with
    /// full-precision rescoring) is the recommended default
    /// configuration.
    ///
    /// Per ADR-035 D-4 + AC-1/AC-2, both SQ8 and Binary search
    /// paths default to `rescore_factor = 5×` to recover
    /// recall@10 ≥ 0.95. Operators may override via
    /// `rescore_factor = 1` (bypasses rescore), accepting the
    /// SQ8-alone recall floor of ≥ 0.92 vs the SQ8 + rescore
    /// default of ≥ 0.95. The predicate is therefore "the
    /// recommended default suggests rescore", not "rescore is
    /// strictly required."
    #[inline]
    #[must_use]
    pub const fn default_recommends_rescore(&self) -> bool {
        !matches!(self, Self::None)
    }
}

impl Default for QuantizerState {
    /// The arena default before quantizer training has fired.
    /// At index creation, `count(N) < 10 M` collections stay at
    /// `None` per the auto-quantize threshold rule (ADR-035 D-4).
    #[inline]
    fn default() -> Self {
        Self::None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sq8_params_roundtrip_with_matching_lengths() {
        let p = Sq8Params::try_new(vec![1.0, 2.0, 3.0], vec![0.1, 0.2, 0.3]).unwrap();
        assert_eq!(p.dim(), 3);
        assert_eq!(p.scale.len(), 3);
        assert_eq!(p.bias.len(), 3);
    }

    #[test]
    fn sq8_params_rejects_dimension_mismatch_in_scale() {
        let err = Sq8Params::try_new(vec![1.0, 2.0], vec![0.1, 0.2, 0.3]).unwrap_err();
        assert!(
            matches!(
                err,
                VectorIndexError::DimensionMismatch {
                    expected: 2,
                    got: 3
                }
            ),
            "got: {err:?}"
        );
    }

    #[test]
    fn sq8_params_rejects_dimension_mismatch_in_bias() {
        let err = Sq8Params::try_new(vec![1.0, 2.0, 3.0], vec![0.1, 0.2]).unwrap_err();
        assert!(
            matches!(
                err,
                VectorIndexError::DimensionMismatch {
                    expected: 3,
                    got: 2
                }
            ),
            "got: {err:?}"
        );
    }

    #[test]
    fn sq8_params_rejects_empty() {
        let err = Sq8Params::try_new(vec![], vec![]).unwrap_err();
        assert!(
            matches!(
                err,
                VectorIndexError::DimensionMismatch {
                    expected: 0,
                    got: 0
                }
            ),
            "got: {err:?}"
        );
    }

    #[test]
    fn sq8_params_rejects_zero_scale_dim() {
        // Constant dimension trainer-fallback case: max == min →
        // scale[i] = 0, would divide-by-zero on encode. Per S-2
        // fold-in this surfaces as DimensionMismatch{N, N}.
        let err = Sq8Params::try_new(vec![1.0, 0.0, 1.0], vec![0.0, 0.0, 0.0]).unwrap_err();
        assert!(
            matches!(
                err,
                VectorIndexError::DimensionMismatch {
                    expected: 3,
                    got: 3
                }
            ),
            "got: {err:?}"
        );
    }

    #[test]
    fn sq8_params_rejects_nan_scale() {
        let err = Sq8Params::try_new(vec![1.0, f32::NAN, 1.0], vec![0.0, 0.0, 0.0]).unwrap_err();
        assert!(
            matches!(err, VectorIndexError::DimensionMismatch { .. }),
            "got: {err:?}"
        );
    }

    #[test]
    fn sq8_params_rejects_infinite_scale() {
        assert!(Sq8Params::try_new(vec![1.0, f32::INFINITY, 1.0], vec![0.0, 0.0, 0.0]).is_err());
        assert!(
            Sq8Params::try_new(vec![1.0, f32::NEG_INFINITY, 1.0], vec![0.0, 0.0, 0.0]).is_err()
        );
    }

    #[test]
    fn sq8_params_rejects_infinite_bias() {
        assert!(Sq8Params::try_new(vec![1.0, 1.0, 1.0], vec![0.0, f32::INFINITY, 0.0]).is_err());
    }

    #[test]
    fn sq8_params_rejects_nan_bias() {
        assert!(Sq8Params::try_new(vec![1.0, 1.0, 1.0], vec![0.0, f32::NAN, 0.0]).is_err());
    }

    #[test]
    fn quantizer_state_none_does_not_recommend_rescore() {
        assert!(!QuantizerState::None.default_recommends_rescore());
    }

    #[test]
    fn quantizer_state_sq8_recommends_rescore() {
        let p = Sq8Params::try_new(vec![1.0, 1.0], vec![0.0, 0.0]).unwrap();
        assert!(QuantizerState::Sq8 { params: p }.default_recommends_rescore());
    }

    #[test]
    fn quantizer_state_binary_recommends_rescore() {
        assert!(QuantizerState::Binary.default_recommends_rescore());
    }

    #[test]
    fn quantizer_state_rabitq_recommends_rescore() {
        let params = RaBitQParams::try_new(2, vec![0.0, 0.0], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        assert!(QuantizerState::RaBitQ { params }.default_recommends_rescore());
    }

    #[test]
    fn quantizer_state_default_is_none() {
        assert_eq!(QuantizerState::default(), QuantizerState::None);
    }
}
