//! Auto-quantize threshold dispatch.
//!
//! Per ADR-035 D-4 + Q3 ratification (owner-ratified 2026-04-25 in
//! ADR-035 §3.3 / §5.3 / §6.3 / §9.3), v1.0 collections auto-select
//! their target encoding by size:
//!
//! - `count(N) < 10 M` → raw F32 (no quantization training).
//! - `count(N) ≥ 10 M` → SQ8 (4× memory reduction; recall@10
//!   ≥ 0.99 typical / ≥ 0.95 ship-blocking with the default
//!   `rescore_factor = 5×` per AC-1a).
//! - operator override permits Binary at any size (32× memory
//!   reduction; recall@10 ≥ 0.95 with rescore per AC-2). The
//!   binary case is **not** auto-selected — operators opt in via
//!   `QUANTIZER = BINARY` at DDL.
//!
//! The 10 M threshold is informed by ADR-035 §4.1.2 memory math:
//! at 768-dim × 10 M × f32 = ~30 GB raw vectors — comfortable on
//! the 100–200 GB RAM budget at v1.0. Above 10 M, the 4× memory
//! reduction materially extends the in-RAM HNSW regime.
//!
//! ## Why `Option<Encoding>` and not `QuantizerState`
//!
//! [`super::QuantizerState`] is the **live** state of an arena: it
//! holds either `None` (raw F32) or a **trained** codebook
//! (`Sq8 { params }`). Per the OQ-V4 contract (ADR-035 §5.2 step
//! 6), at index creation the live state is always
//! [`super::QuantizerState::None`] regardless of the auto-quantize
//! signal — vectors land in the raw-F32 staging arena until the
//! trainer fires (`vectors_count >= 100_000`), and only then does
//! the state transition to `Sq8 { params: <trained> }`.
//!
//! This helper expresses the **configuration hint** — the target
//! encoding the trainer should produce when it eventually fires.
//! [`Encoding`] is the right type for that hint: orthogonal to
//! "is the codebook trained yet". Returning `Option<Encoding>`
//! makes the staging-vs-trained distinction explicit at the call
//! site (ADR-035 §5.2 step 6 / Slice F.1 arena routing): the
//! caller stores the hint on the arena's catalog record, then
//! consults [`super::QuantizerState`] for the live state. See the
//! tests in `tests/quantizer.rs` for the boundary case at
//! `N == 10_000_000`.

use crate::Encoding;

/// The auto-quantize threshold per ADR-035 D-4: 10 M vectors. The
/// constant is exposed so call sites in the catalog / arena layer
/// (Slice F.1) can express their threshold logic in the same
/// unit.
pub const AUTO_QUANTIZE_THRESHOLD_VECTORS: usize = 10_000_000;

/// Auto-select the target encoding for a collection of
/// `n_vectors` per ADR-035 D-4 + Q3 ratification.
///
/// - Returns `None` for `n_vectors < 10_000_000` — the collection
///   stays at raw F32 (no training scheduled).
/// - Returns `Some(Encoding::Sq8)` for `n_vectors >= 10_000_000` —
///   the trainer is scheduled to fire (per OQ-V4) and produce a
///   trained [`super::Sq8Params`] codebook for the arena.
///
/// Binary is **not** auto-selected; operators opt in explicitly
/// at DDL. F16 is a v1.1 halfvec compatibility option (not
/// auto-selected at v1.0).
///
/// The function is `const`-qualified so the catalog-layer
/// configuration code can pre-compute the encoding at compile
/// time when the collection size is known statically (rare in
/// practice, but defensible because the threshold logic has no
/// runtime dependency).
#[inline]
#[must_use]
pub const fn auto_quantizer_for_collection(n_vectors: usize) -> Option<Encoding> {
    if n_vectors >= AUTO_QUANTIZE_THRESHOLD_VECTORS {
        Some(Encoding::Sq8)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn under_threshold_yields_none() {
        assert_eq!(auto_quantizer_for_collection(0), None);
        assert_eq!(auto_quantizer_for_collection(1), None);
        assert_eq!(auto_quantizer_for_collection(100_000), None);
        assert_eq!(auto_quantizer_for_collection(1_000_000), None);
        // The boundary case lives in the integration test
        // `auto_quantizer_threshold_at_10m` per the slice spec.
        assert_eq!(
            auto_quantizer_for_collection(AUTO_QUANTIZE_THRESHOLD_VECTORS - 1),
            None
        );
    }

    #[test]
    fn at_or_above_threshold_yields_sq8() {
        assert_eq!(
            auto_quantizer_for_collection(AUTO_QUANTIZE_THRESHOLD_VECTORS),
            Some(Encoding::Sq8)
        );
        assert_eq!(
            auto_quantizer_for_collection(AUTO_QUANTIZE_THRESHOLD_VECTORS + 1),
            Some(Encoding::Sq8)
        );
        assert_eq!(
            auto_quantizer_for_collection(100_000_000),
            Some(Encoding::Sq8)
        );
        assert_eq!(
            auto_quantizer_for_collection(usize::MAX),
            Some(Encoding::Sq8)
        );
    }

    #[test]
    fn threshold_is_10_million() {
        assert_eq!(AUTO_QUANTIZE_THRESHOLD_VECTORS, 10_000_000);
    }

    #[test]
    fn binary_is_never_auto_selected() {
        for n in [0_usize, 1, 100, 10_000, 10_000_000, 100_000_000, usize::MAX] {
            assert_ne!(auto_quantizer_for_collection(n), Some(Encoding::Binary));
        }
    }

    #[test]
    fn f16_is_never_auto_selected_at_v1() {
        // F16 is a v1.1 halfvec compatibility option per ADR-035
        // §3.2; v1.0 auto-select returns SQ8 above the threshold,
        // never F16.
        for n in [0_usize, 1, 10_000_000, usize::MAX] {
            assert_ne!(auto_quantizer_for_collection(n), Some(Encoding::F16));
        }
    }

    #[test]
    fn rabitq_is_never_auto_selected_at_v1() {
        for n in [0_usize, 1, 10_000_000, 100_000_000, usize::MAX] {
            assert_ne!(auto_quantizer_for_collection(n), Some(Encoding::RaBitQ));
        }
    }
}
