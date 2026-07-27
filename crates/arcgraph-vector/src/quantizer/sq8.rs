//! SQ8 (8-bit scalar quantization) trainer + codebook.
//!
//! Per ADR-035 §3.3 + D-4, SQ8 is the v1.0 default for
//! collections with `count(N) ≥ 10 M`. The training pipeline:
//!
//! 1. Sample `N_train = min(0.01 × N, 1 M)` vectors from the
//!    collection (reservoir sample is the caller's responsibility;
//!    [`Sq8Trainer::train`] consumes whatever sample is presented).
//! 2. For each dimension `d`: compute `min_d`, `max_d`.
//! 3. Per-dim params:
//!    - `scale_d = (max_d - min_d) / 255.0`
//!    - `bias_d  = min_d`
//! 4. Encode: `q[d] = (round((x[d] - bias[d]) / scale[d]) - 128.0).clamp(-128, 127) as i8`
//! 5. Decode: `x[d] ≈ (q[d] as f32 + 128.0) * scale[d] + bias[d]`
//!
//! Recall@10 vs raw f32: ≥ 0.99 for typical embedding
//! distributions (per Faiss benchmarks). Recall@10 ship-blocking
//! bound at SQ8 + `rescore_factor = 5×` is ≥ 0.95 (ADR-035 AC-1a);
//! this module provides the codec; the rescore wiring is Slice
//! E.2 / E.3.
//!
//! ## Byte-width convention (per #116 closure / Slice F.1)
//!
//! Encoded values are emitted as `i8` directly so that the
//! [`crate::distance::L2Sq8`] / [`crate::distance::IpSq8`] /
//! [`crate::distance::CosineSq8`] kernels — which reinterpret
//! bytes as `i8` for simsimd's signed integer SIMD path — read
//! the codec's output verbatim. The shift `(round(...) - 128)`
//! is applied inline at encode time; decode applies the inverse
//! `(q + 128)`. L2 distance is translation-invariant so primary
//! recall is unchanged vs the historical u8-then-shift pipeline.
//! Earlier ADR-035 §3.3 wording specified `u8 in 0..=255`; #116
//! supersedes for the codec / arena boundary. The Sq8Params
//! `(scale, bias)` codebook itself is unchanged: production code
//! that consumed `Sq8Params` directly (snapshot serializer,
//! catalog) keeps the same shape.
//!
//! ## Constant-dimension fallback (per ADR-035 §9.3)
//!
//! When `min_d == max_d` for a single dimension, the naïve
//! `scale_d = 0` triggers divide-by-zero on encode. The trainer
//! collapses such dimensions to a sentinel `(scale_d = 1.0,
//! bias_d = min_d)`: the dimension never varies in the sample,
//! so any input value at that dim flows through the same
//! `(round(...) - 128).clamp(-128, 127)` encode + `(q + 128) *
//! scale + bias` decode pipeline as the non-constant dims (see
//! [`Sq8Codebook::encode`] / [`Sq8Codebook::decode`] for the
//! authoritative formulas). With `scale = 1.0` and `bias =
//! min_d`, the input `x = min_d` round-trips exactly through the
//! sentinel value `q = -128`. For inputs that drift outside
//! `min_d ± 255`, the clamp introduces a bounded saturation
//! error — acceptable because the trainer signals via
//! [`tracing::warn`] that the dimension was constant in the
//! sample, and an operator-driven `REINDEX` may refresh the
//! codebook on a richer sample.
//!
//! This is a per-dimension fallback. ADR-035 §9.3 also describes
//! a wholesale fallback to [`super::QuantizerState::None`] when
//! ALL training vectors are degenerate (constant or NaN/Inf);
//! that fallback lives at the call site (the trainer-trigger
//! routine in Slice F.1 / G.4) where it can flip the catalog
//! state. [`Sq8Trainer::train`] errors on wholly-degenerate input
//! (`samples.is_empty()` or non-finite samples) so the call site
//! can map the error to the §9.3 fallback path.
//!
//! ## OQ-V4 re-encode contract (per ADR-035 §5.2 step 6)
//!
//! [`Sq8Codebook::encode`] is **arena-state-free**: it consumes
//! one f32 vector and produces one i8 vector with no shared
//! state, no locks, no I/O. The OQ-V4 re-encode pass
//! (training-completion background pass) calls it per-vector and
//! atomic-stores the published flag after the per-vector write
//! lands in the quantized arena. The codec's freedom-from-state
//! is what lets the re-encode pass run without serialization
//! against concurrent inserts — see `super::mod` docs.

use tracing::warn;

use crate::error::VectorIndexError;

use super::Sq8Params;

/// SQ8 quantization range: 256 levels per dim. The training
/// formula `scale = (max - min) / 255.0` defines the level
/// spacing; the encode then shifts by -128 to land in the
/// `-128..=127` (i8) range that simsimd's signed integer SIMD
/// kernels read directly (per #116 closure / Slice F.1).
const SQ8_QUANT_LEVELS: f32 = 255.0;

/// Trainer for [`Sq8Codebook`]. Stateless surface — `train` is a
/// pure function of the sample slice. Callers (Slice F.1 / G.4
/// re-encode trigger) prepare the reservoir sample and invoke
/// `Sq8Trainer::train`; on success they install the codebook into
/// the arena's [`super::QuantizerState`].
#[derive(Debug, Default, Clone, Copy)]
pub struct Sq8Trainer;

impl Sq8Trainer {
    /// Train an [`Sq8Codebook`] from `samples`. Each sample MUST
    /// have the same dimension as the others; the dim is inferred
    /// from `samples[0].len()`.
    ///
    /// # Errors
    ///
    /// - [`VectorIndexError::DimensionMismatch`] with `expected: 0,
    ///   got: 0` when `samples` is empty (no training data).
    /// - [`VectorIndexError::DimensionMismatch`] with `expected:
    ///   samples[0].len(), got: samples[i].len()` when any sample
    ///   has a different dim than the first.
    /// - [`VectorIndexError::DimensionMismatch`] with `expected: 0,
    ///   got: 0` when `samples[0]` is empty (zero-dim codebook).
    /// - [`VectorIndexError::DimensionMismatch`] with `expected:
    ///   dim, got: dim` when any sample contains a non-finite
    ///   value (NaN or ±∞). The trainer rejects rather than
    ///   silently masking; the call site applies the §9.3
    ///   wholesale-fallback to [`super::QuantizerState::None`].
    ///
    /// # Behavior on constant dimensions
    ///
    /// Per the module-level docs (ADR-035 §9.3), a dimension whose
    /// per-sample min equals its max is collapsed to
    /// `(scale = 1.0, bias = min)` and a `tracing::warn!` is
    /// emitted. The training succeeds; the resulting codebook is
    /// valid (passes [`Sq8Params::try_new`]).
    pub fn train(&self, samples: &[&[f32]]) -> Result<Sq8Codebook, VectorIndexError> {
        if samples.is_empty() {
            return Err(VectorIndexError::DimensionMismatch {
                expected: 0,
                got: 0,
            });
        }

        let dim = samples[0].len();
        if dim == 0 {
            return Err(VectorIndexError::DimensionMismatch {
                expected: 0,
                got: 0,
            });
        }

        // Validate uniform dim and finite values up front; on
        // failure we surface a single DimensionMismatch with the
        // first offending sample's dim (uniformity case) or
        // `dim, dim` (finiteness case, per S-2 fold-in).
        for sample in samples.iter() {
            if sample.len() != dim {
                return Err(VectorIndexError::DimensionMismatch {
                    expected: dim,
                    got: sample.len(),
                });
            }
            if !sample.iter().all(|x| x.is_finite()) {
                return Err(VectorIndexError::DimensionMismatch {
                    expected: dim,
                    got: dim,
                });
            }
        }

        // Per-dim min/max scan. We seed with samples[0] (so the
        // first iteration of the inner loop is a no-op rather than
        // a needless `±INFINITY` comparison).
        let mut mins = samples[0].to_vec();
        let mut maxs = samples[0].to_vec();
        for sample in &samples[1..] {
            for d in 0..dim {
                let v = sample[d];
                if v < mins[d] {
                    mins[d] = v;
                }
                if v > maxs[d] {
                    maxs[d] = v;
                }
            }
        }

        // Per-dim scale/bias with constant-dim fallback (§9.3).
        let mut scale = Vec::with_capacity(dim);
        let mut bias = Vec::with_capacity(dim);
        let mut constant_dims: usize = 0;
        for d in 0..dim {
            let lo = mins[d];
            let hi = maxs[d];
            if hi == lo {
                // Constant dim fallback per ADR-035 §9.3.
                scale.push(1.0);
                bias.push(lo);
                constant_dims += 1;
            } else {
                let s = (hi - lo) / SQ8_QUANT_LEVELS;
                // (hi - lo) > 0 and finite (we checked all samples
                // are finite); s > 0 and finite by construction.
                scale.push(s);
                bias.push(lo);
            }
        }

        if constant_dims > 0 {
            warn!(
                arcgraph.vector.sq8_trainer_constant_dims = constant_dims,
                arcgraph.vector.sq8_trainer_dim = dim,
                "SQ8 trainer collapsed {constant_dims}/{dim} constant dims to (scale=1.0, bias=min) per ADR-035 §9.3"
            );
        }

        // try_new is the codebook validity gate; the trainer's
        // construction satisfies it by construction (matching
        // lengths, non-zero finite scale, finite bias). The
        // explicit Result-propagation here is defense in depth:
        // a future trainer change that breaks an invariant
        // surfaces here rather than at the first encode call.
        let params = Sq8Params::try_new(scale, bias)?;
        Ok(Sq8Codebook { params })
    }
}

/// SQ8 codebook: trained per-dim `(scale, bias)` pair plus the
/// per-vector `encode` / `decode` operations.
///
/// Wraps [`Sq8Params`] (the serialized form stored alongside the
/// arena snapshot) and exposes the codec methods. Callers that
/// only need the params (e.g., snapshot writers) consume
/// `codebook.params` directly.
#[derive(Debug, Clone, PartialEq)]
pub struct Sq8Codebook {
    params: Sq8Params,
}

impl Sq8Codebook {
    /// Construct from already-trained [`Sq8Params`] (e.g., loaded
    /// from a snapshot). Validity checks are deferred to
    /// [`Sq8Params::try_new`]; this constructor accepts any
    /// `Sq8Params` because by that point the params have already
    /// been validated either by the trainer or by snapshot
    /// deserialization.
    #[must_use]
    pub const fn from_params(params: Sq8Params) -> Self {
        Self { params }
    }

    /// Borrow the underlying parameters for snapshot serialization
    /// or arena catalog updates.
    #[inline]
    #[must_use]
    pub const fn params(&self) -> &Sq8Params {
        &self.params
    }

    /// Consume the codebook, yielding the parameters (e.g., to
    /// install into [`super::QuantizerState::Sq8`]).
    #[inline]
    #[must_use]
    pub fn into_params(self) -> Sq8Params {
        self.params
    }

    /// Codebook dimension. Encoded / decoded vectors must match.
    #[inline]
    #[must_use]
    pub fn dim(&self) -> usize {
        self.params.dim()
    }

    /// Encode an `f32` vector into an `i8` quantized vector.
    ///
    /// `vec.len()` must equal `self.dim()`. Each element is mapped
    /// via the per-dim affine quantizer plus a centering shift so
    /// the output lands directly in the `i8` range that the
    /// simsimd-backed [`crate::distance::L2Sq8`] kernel reads:
    ///
    /// ```text
    /// q[d] = (round((x[d] - bias[d]) / scale[d]) - 128.0)
    ///            .clamp(-128.0, 127.0) as i8
    /// ```
    ///
    /// The clamp guards against samples that drift outside the
    /// `[min, max]` envelope observed during training (and against
    /// rounding spillover at the boundary). Per #116 closure /
    /// Slice F.1, the codec emits `i8` directly so the per-tenant
    /// arena (Slice F.1) and the kernel (Slice B) share one byte
    /// width with no per-read translation.
    ///
    /// # Errors
    ///
    /// - [`VectorIndexError::DimensionMismatch`] with the codebook
    ///   dim as `expected` and the input length as `got`.
    pub fn encode(&self, vec: &[f32]) -> Result<Vec<i8>, VectorIndexError> {
        if vec.len() != self.dim() {
            return Err(VectorIndexError::DimensionMismatch {
                expected: self.dim(),
                got: vec.len(),
            });
        }
        let mut out = Vec::with_capacity(self.dim());
        for (d, &x) in vec.iter().enumerate() {
            // Quotient lands in [0.0, 255.0] for in-envelope inputs;
            // shifting by -128 centers the range on i8's signed
            // representation. NaN propagation via `q_f` is not
            // possible because the codebook constructor rejects
            // non-finite scale/bias and the input contract for
            // encode is "finite f32"; if a caller violates that
            // contract, NaN clamps to the lower bound (-128) per
            // f32::clamp's NaN handling.
            let q_f = ((x - self.params.bias[d]) / self.params.scale[d]).round() - 128.0;
            let q_clamped = q_f.clamp(-128.0, SQ8_QUANT_LEVELS - 128.0);
            out.push(q_clamped as i8);
        }
        Ok(out)
    }

    /// Decode an `i8` quantized vector back to `f32`.
    ///
    /// `q.len()` must equal `self.dim()`. Each element is mapped
    /// via the inverse affine, undoing the centering shift applied
    /// at encode time:
    ///
    /// ```text
    /// x[d] ≈ (q[d] as f32 + 128.0) * scale[d] + bias[d]
    /// ```
    ///
    /// The decode is exact for inputs that round-tripped without
    /// clamp saturation; values that hit the `-128` or `127`
    /// boundary on encode lose the out-of-range residual.
    ///
    /// # Errors
    ///
    /// - [`VectorIndexError::DimensionMismatch`] with the codebook
    ///   dim as `expected` and the input length as `got`.
    pub fn decode(&self, q: &[i8]) -> Result<Vec<f32>, VectorIndexError> {
        if q.len() != self.dim() {
            return Err(VectorIndexError::DimensionMismatch {
                expected: self.dim(),
                got: q.len(),
            });
        }
        let mut out = Vec::with_capacity(self.dim());
        for (d, &qd) in q.iter().enumerate() {
            out.push((f32::from(qd) + 128.0) * self.params.scale[d] + self.params.bias[d]);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift64 PRNG for reproducible test data.
    /// Avoids pulling in `rand` as a dev-dep for unit tests.
    struct Xs64(u64);

    impl Xs64 {
        const fn new(seed: u64) -> Self {
            // xorshift64 requires non-zero seed; mix in a
            // constant so callers may pass 0.
            Self(seed ^ 0x9E37_79B9_7F4A_7C15)
        }

        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }

        /// Uniform f32 in `[lo, hi)`.
        fn next_uniform(&mut self, lo: f32, hi: f32) -> f32 {
            // Use the high 24 bits to fill an f32 mantissa so
            // every f32 in the interval is reachable.
            let bits = (self.next_u64() >> 40) as u32;
            let unit = (bits as f32) / ((1u32 << 24) as f32);
            lo + (hi - lo) * unit
        }
    }

    #[test]
    fn xs64_uniform_is_within_bounds() {
        let mut rng = Xs64::new(42);
        for _ in 0..1000 {
            let v = rng.next_uniform(-1.0, 1.0);
            assert!((-1.0..1.0).contains(&v), "got: {v}");
        }
    }

    #[test]
    fn trainer_rejects_empty_samples() {
        let err = Sq8Trainer.train(&[]).unwrap_err();
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
    fn trainer_rejects_zero_dim_sample() {
        let empty: &[f32] = &[];
        let err = Sq8Trainer.train(&[empty]).unwrap_err();
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
    fn trainer_rejects_non_uniform_dim() {
        let s0: &[f32] = &[0.0, 1.0, 2.0];
        let s1: &[f32] = &[0.0, 1.0];
        let err = Sq8Trainer.train(&[s0, s1]).unwrap_err();
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
    fn trainer_rejects_non_finite_input() {
        let s0: &[f32] = &[0.0, 1.0, f32::NAN];
        let err = Sq8Trainer.train(&[s0]).unwrap_err();
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

        let s1: &[f32] = &[0.0, f32::INFINITY, 2.0];
        let err = Sq8Trainer.train(&[s1]).unwrap_err();
        assert!(matches!(err, VectorIndexError::DimensionMismatch { .. }));
    }

    #[test]
    fn trainer_collapses_constant_dim_to_sentinel() {
        // Dim 1 is constant (always 0.5); dims 0 and 2 vary.
        let s0: &[f32] = &[0.0, 0.5, -1.0];
        let s1: &[f32] = &[1.0, 0.5, 1.0];
        let s2: &[f32] = &[0.5, 0.5, 0.0];
        let cb = Sq8Trainer.train(&[s0, s1, s2]).unwrap();
        let p = cb.params();
        assert_eq!(p.dim(), 3);
        // Constant dim → sentinel (scale=1.0, bias=0.5).
        assert_eq!(p.scale[1], 1.0);
        assert_eq!(p.bias[1], 0.5);
        // Varying dims have non-trivial scale.
        assert!(p.scale[0] > 0.0 && p.scale[0] < 1.0);
        assert!(p.scale[2] > 0.0 && p.scale[2] < 1.0);
    }

    #[test]
    fn trainer_one_sample_is_all_constant_dims() {
        // A single sample produces all-constant dims; trainer
        // succeeds via the §9.3 per-dim fallback (scale=1.0,
        // bias=value). All emit the warn but the codebook is
        // valid.
        let s: &[f32] = &[1.0, 2.0, 3.0];
        let cb = Sq8Trainer.train(&[s]).unwrap();
        for d in 0..3 {
            assert_eq!(cb.params().scale[d], 1.0);
        }
        assert_eq!(cb.params().bias, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn encode_rejects_wrong_dim() {
        let cb = Sq8Trainer
            .train(&[&[0.0, 1.0, 2.0][..], &[1.0, 2.0, 3.0][..]])
            .unwrap();
        let err = cb.encode(&[0.0, 1.0]).unwrap_err();
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
    fn decode_rejects_wrong_dim() {
        let cb = Sq8Trainer
            .train(&[&[0.0, 1.0, 2.0][..], &[1.0, 2.0, 3.0][..]])
            .unwrap();
        let err = cb.decode(&[0_i8, 0_i8]).unwrap_err();
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
    fn round_trip_sample_corner_min_max() {
        // Train on the corners; encode + decode the corners; the
        // boundary values should round-trip exactly to within the
        // 1-bin quantization step (≤ scale/2).
        let s0: &[f32] = &[-1.0, -1.0, -1.0];
        let s1: &[f32] = &[1.0, 1.0, 1.0];
        let cb = Sq8Trainer.train(&[s0, s1]).unwrap();
        // scale = (1 - (-1))/255 = 2/255 ≈ 0.00784
        for &x in &[-1.0_f32, 1.0_f32] {
            let v: Vec<f32> = vec![x; 3];
            let q = cb.encode(&v).unwrap();
            let d = cb.decode(&q).unwrap();
            for d_v in &d {
                let err = (d_v - x).abs();
                // 1-bin tolerance.
                let bin = (1.0 - (-1.0)) / 255.0;
                assert!(err <= bin, "x={x} decoded={d_v} err={err} bin={bin}");
            }
        }
    }

    #[test]
    fn round_trip_clamp_saturates_outside_training_envelope() {
        // Train on [0, 1]; encode 2.0 (above max); decode should
        // yield 1.0 (clamped to max via i8=127 → 255 unsigned →
        // 1.0) — bounded saturation, not divergence.
        let s0: &[f32] = &[0.0];
        let s1: &[f32] = &[1.0];
        let cb = Sq8Trainer.train(&[s0, s1]).unwrap();
        let q = cb.encode(&[2.0]).unwrap();
        // i8 max = 127, the centered representation of u8 = 255.
        assert_eq!(q[0], 127_i8);
        let d = cb.decode(&q).unwrap();
        // Decode at the clamp boundary: (127 + 128) * scale + bias
        //   = 255 * (1/255) + 0 = 1.0
        assert!((d[0] - 1.0).abs() < 1e-6, "got: {}", d[0]);
    }

    #[test]
    fn round_trip_clamp_saturates_below_training_envelope() {
        // Train on [0, 1]; encode -1.0 (below min); the i8 lower
        // bound (-128) corresponds to u8 = 0 and decodes back to
        // bias = 0.0.
        let s0: &[f32] = &[0.0];
        let s1: &[f32] = &[1.0];
        let cb = Sq8Trainer.train(&[s0, s1]).unwrap();
        let q = cb.encode(&[-1.0]).unwrap();
        // i8 min = -128, the centered representation of u8 = 0.
        assert_eq!(q[0], -128_i8);
        let d = cb.decode(&q).unwrap();
        // Decode at the clamp boundary: (-128 + 128) * scale + bias
        //   = 0 * (1/255) + 0 = 0.0
        assert!(d[0].abs() < 1e-6, "got: {}", d[0]);
    }

    #[test]
    fn round_trip_uniform_768_dim_within_one_percent() {
        // Smaller version of the integration test; full 10K
        // sample with the 1% per-dim error bound lives in
        // tests/quantizer.rs.
        const DIM: usize = 768;
        const N: usize = 1024;
        let mut rng = Xs64::new(7);
        let storage: Vec<Vec<f32>> = (0..N)
            .map(|_| (0..DIM).map(|_| rng.next_uniform(-1.0, 1.0)).collect())
            .collect();
        let samples: Vec<&[f32]> = storage.iter().map(Vec::as_slice).collect();
        let cb = Sq8Trainer.train(&samples).unwrap();

        // Per-dim quantization step is (max - min) / 255. For a
        // ~uniform sample on [-1, 1] this is roughly 2/255 ≈ 0.78%
        // of the [-1, 1] range, with at most a 1-bin rounding
        // error (≤ scale/2). The 1% absolute bound is satisfied
        // by construction.
        let probe = &storage[0];
        let q = cb.encode(probe).unwrap();
        let d = cb.decode(&q).unwrap();
        for i in 0..DIM {
            let err_abs = (probe[i] - d[i]).abs();
            let bin = cb.params().scale[i];
            // 1-bin tolerance — this is the maximal round-trip
            // error for any input within the training envelope.
            assert!(
                err_abs <= bin + 1e-6,
                "dim={i} probe={} decoded={} err={} bin={}",
                probe[i],
                d[i],
                err_abs,
                bin
            );
        }
    }

    #[test]
    fn from_params_then_encode_matches_trainer() {
        let cb1 = Sq8Trainer
            .train(&[&[0.0, 1.0][..], &[1.0, 2.0][..]])
            .unwrap();
        let cb2 = Sq8Codebook::from_params(cb1.params().clone());
        let probe = vec![0.5, 1.5];
        assert_eq!(cb1.encode(&probe).unwrap(), cb2.encode(&probe).unwrap());
    }

    #[test]
    fn into_params_roundtrips() {
        let cb = Sq8Trainer
            .train(&[&[0.0, 1.0][..], &[1.0, 2.0][..]])
            .unwrap();
        let p = cb.into_params();
        assert_eq!(p.dim(), 2);
    }
}
