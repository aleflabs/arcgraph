//! Vector encoding, distance metric, and index-type tags.
//!
//! These three enums fully describe the **shape** of a vector
//! search at v1.0:
//!
//! - [`Encoding`] — how vectors are stored on disk and in the
//!   arena. Per ADR-035 D-4: F32 default for small collections,
//!   SQ8 default at ≥10 M, binary opt-in for memory-constrained
//!   workloads, RaBitQ opt-in for ADR-209 compressed nav payloads,
//!   F16 for halfvec parity (pgvector-class).
//! - [`Metric`] — distance / similarity function the kernel
//!   computes. Per ADR-035 D-2: L2 / IP / Cosine universally;
//!   Hamming exclusively for binary.
//! - [`IndexType`] — which index algorithm the catalog selected.
//!   Per ADR-035 D-1: HNSW for hot ≤ ~50 M collections, DiskANN
//!   for 50 M – 1 B.

use serde::{Deserialize, Serialize};

/// On-disk + in-arena vector encoding. Drives both storage size
/// and the [`crate::DistanceKernel`] dispatch.
///
/// Per ADR-035 D-4 the v1.0 default is auto-selected by collection
/// size:
///
/// - `count(N) < 10 M` → [`Encoding::F32`] (raw, no quantization).
/// - `count(N) ≥ 10 M` → [`Encoding::Sq8`] (4× memory reduction,
///   99 %+ recall preserved).
/// - operator override permits [`Encoding::Binary`] at any size
///   (32× memory reduction, recall@10 ≥ 0.95 with rescore).
/// - [`Encoding::RaBitQ`] is ADR-209's trained 1-bit rotated
///   quantizer; slice 1 exposes the standalone codec only.
///
/// [`Encoding::F16`] is a v1.1 halfvec compatibility option; v1.0
/// does not auto-select it but the kernel surface ships at v1.0
/// to keep the encoding axis closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Encoding {
    /// 32-bit IEEE-754 float per dimension. Default for
    /// `count(N) < 10 M` collections; rescore source for SQ8 +
    /// binary indexes.
    F32,

    /// 16-bit IEEE-754 half-precision float per dimension. v1.1
    /// halfvec compatibility (pgvector parity). Kernel surface
    /// ships at v1.0 (Slice B) for closed enum dispatch.
    F16,

    /// 8-bit scalar-quantized integer per dimension; per-dim
    /// `(scale, bias)` codebook stored alongside the arena. v1.0
    /// default for `count(N) ≥ 10 M`. See [`crate::Sq8Params`].
    Sq8,

    /// 1-bit-per-dimension sign quantization. Packed at 128-byte
    /// alignment (per ADR-035 S-1 fold-in). Hamming distance is
    /// the natural metric.
    Binary,

    /// RaBitQ trained 1-bit rotated quantization: packed sign bits
    /// plus `f_o` and `n_o` f32 factors. The serde name is fixed
    /// to `rabitq`; `rename_all = "snake_case"` would otherwise
    /// emit the ugly and permanent wire spelling `ra_bit_q`.
    #[serde(rename = "rabitq")]
    RaBitQ,
}

impl Encoding {
    /// Bytes per vector at the given dimension. Used by the arena
    /// allocator to size each slot.
    ///
    /// - F32: `dim * 4`
    /// - F16: `dim * 2`
    /// - SQ8: `dim`
    /// - Binary: `(dim + 7) / 8` rounded up to 128-byte alignment
    ///   per ADR-035 §5.2 / §S-1 (only enforced by the arena
    ///   layout — this helper returns the unaligned packed size).
    /// - RaBitQ: `(dim + 7) / 8 + 8` for sign bits plus `f_o`
    ///   and `n_o`.
    #[inline]
    #[must_use]
    pub const fn bytes_per_vector_unaligned(self, dim: usize) -> usize {
        match self {
            Self::F32 => dim * 4,
            Self::F16 => dim * 2,
            Self::Sq8 => dim,
            Self::Binary => dim.div_ceil(8),
            Self::RaBitQ => dim.div_ceil(8) + 8,
        }
    }

    /// Bytes per vector with cache-line alignment applied.
    ///
    /// Binary and RaBitQ vectors are padded to the next multiple
    /// of 64 bytes (one cache line) per ADR-035 §S-1; this guarantees
    /// the per-vector slot starts on a cache-line boundary so
    /// the SIMD Hamming kernel does not straddle lines on
    /// consecutive comparisons. Non-binary encodings have
    /// natural alignment from their element size (F32 → 4-byte
    /// aligned, F16 → 2-byte, SQ8 → 1-byte) and do not need
    /// padding at the per-vector level — the arena allocator
    /// applies its own larger alignment for the section start
    /// per `docs/design/vector-storage-layout.md` §3.
    ///
    /// Consumers (Slice F.1 per-tenant arena routing, Slice G.2
    /// snapshot flush) call this helper to size each vector slot.
    #[inline]
    #[must_use]
    pub const fn bytes_per_vector_aligned(self, dim: usize) -> usize {
        let unaligned = self.bytes_per_vector_unaligned(dim);
        match self {
            Self::Binary | Self::RaBitQ => unaligned.next_multiple_of(64),
            _ => unaligned,
        }
    }

    /// Whether this encoding requires a trained codebook before
    /// vectors can be encoded (`Sq8` and `RaBitQ` do; `F32`,
    /// `F16`, `Binary` do not — binary is the sign function, no
    /// training).
    #[inline]
    #[must_use]
    pub const fn requires_training(self) -> bool {
        matches!(self, Self::Sq8 | Self::RaBitQ)
    }
}

/// Distance / similarity metric the kernel computes. v1.0 ships
/// L2 + IP + Cosine across F32/F16/SQ8, plus Hamming for Binary.
///
/// Cosine is computed as IP after L2-normalization OR via the
/// `simsimd::SpatialSimilarity::cosine` kernel directly (Slice B).
/// At the trait surface, callers select the metric; the kernel
/// chooses the dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Metric {
    /// Squared Euclidean distance: `Σᵢ (aᵢ − bᵢ)²`. Lower is
    /// closer.
    L2,
    /// Inner product: `Σᵢ aᵢ · bᵢ`. **Higher** is closer (the
    /// trait surface returns "closeness as f32"; callers must
    /// honor the ordering convention per metric).
    Ip,
    /// Cosine similarity: `IP(a, b) / (‖a‖₂ · ‖b‖₂)`. Higher is
    /// closer.
    Cosine,
    /// Hamming distance on packed binary vectors: `popcount(a XOR b)`.
    /// Lower is closer. Defined only for [`Encoding::Binary`].
    Hamming,
}

impl Metric {
    /// Whether this metric is **valid** for the given encoding.
    /// Hamming applies only to Binary; the other three work on
    /// any non-binary encoding.
    ///
    /// Note: this check is the encoding/metric compatibility
    /// gate at the type level. Individual backends may impose
    /// **additional** restrictions at construction time. In
    /// particular, [`crate::diskann::DiskAnnGraph::new`] also
    /// rejects [`Metric::Ip`] per issue #109 defensive (a),
    /// pending the v1.1 sign-aware α-prune comparator. HNSW
    /// has no such restriction.
    #[inline]
    #[must_use]
    pub const fn is_valid_for(self, encoding: Encoding) -> bool {
        match (self, encoding) {
            (Self::Hamming, Encoding::Binary) => true,
            (Self::Hamming, _) => false,
            (_, Encoding::Binary) => false,
            _ => true,
        }
    }
}

/// Vector index algorithm. Selected at `DEFINE INDEX` DDL
/// per ADR-035 D-1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexType {
    /// Hierarchical Navigable Small World (Malkov & Yashunin
    /// TPAMI 2018). v1.0 default for hot collections ≤ ~50 M
    /// vectors per tenant.
    Hnsw,
    /// DiskANN / Vamana (Subramanya et al. NeurIPS 2019). v1.0
    /// default for 50 M – 1 B.
    DiskAnn,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_per_vector_f32_768() {
        assert_eq!(Encoding::F32.bytes_per_vector_unaligned(768), 3072);
    }

    #[test]
    fn bytes_per_vector_f16_768() {
        assert_eq!(Encoding::F16.bytes_per_vector_unaligned(768), 1536);
    }

    #[test]
    fn bytes_per_vector_sq8_768() {
        assert_eq!(Encoding::Sq8.bytes_per_vector_unaligned(768), 768);
    }

    #[test]
    fn bytes_per_vector_binary_768() {
        // 768 bits / 8 = 96 bytes (aligned to 128 in arena layout
        // per ADR-035 S-1 fold-in; this helper returns unaligned).
        assert_eq!(Encoding::Binary.bytes_per_vector_unaligned(768), 96);
    }

    #[test]
    fn binary_packs_odd_dimension_with_ceiling() {
        // 7-bit vector → 1 byte.
        assert_eq!(Encoding::Binary.bytes_per_vector_unaligned(7), 1);
        // 9-bit vector → 2 bytes.
        assert_eq!(Encoding::Binary.bytes_per_vector_unaligned(9), 2);
    }

    // ─── bytes_per_vector_aligned (S-1) ───────────────────────

    #[test]
    fn binary_dim_768_aligns_to_128() {
        // 768 / 8 = 96 bytes unaligned; next multiple of 64 = 128.
        // ADR-035 §S-1 fold-in: binary vectors live on cache-line
        // boundaries.
        assert_eq!(Encoding::Binary.bytes_per_vector_aligned(768), 128);
    }

    #[test]
    fn binary_dim_64_already_cache_aligned() {
        // 64 / 8 = 8 bytes unaligned; next multiple of 64 = 64.
        assert_eq!(Encoding::Binary.bytes_per_vector_aligned(64), 64);
    }

    #[test]
    fn binary_dim_513_aligns_to_128() {
        // 513 / 8 ceil = 65 bytes; next multiple of 64 = 128.
        assert_eq!(Encoding::Binary.bytes_per_vector_aligned(513), 128);
    }

    #[test]
    fn f32_dim_768_no_padding() {
        let dim = 768;
        assert_eq!(
            Encoding::F32.bytes_per_vector_aligned(dim),
            Encoding::F32.bytes_per_vector_unaligned(dim)
        );
    }

    #[test]
    fn sq8_dim_768_no_padding() {
        let dim = 768;
        assert_eq!(
            Encoding::Sq8.bytes_per_vector_aligned(dim),
            Encoding::Sq8.bytes_per_vector_unaligned(dim)
        );
    }

    #[test]
    fn f16_dim_768_no_padding() {
        let dim = 768;
        assert_eq!(
            Encoding::F16.bytes_per_vector_aligned(dim),
            Encoding::F16.bytes_per_vector_unaligned(dim)
        );
    }

    #[test]
    fn requires_training_only_for_sq8() {
        assert!(!Encoding::F32.requires_training());
        assert!(!Encoding::F16.requires_training());
        assert!(Encoding::Sq8.requires_training());
        assert!(!Encoding::Binary.requires_training());
    }

    #[test]
    fn metric_validity_matrix() {
        // L2/IP/Cosine valid on non-binary encodings.
        for m in [Metric::L2, Metric::Ip, Metric::Cosine] {
            assert!(m.is_valid_for(Encoding::F32));
            assert!(m.is_valid_for(Encoding::F16));
            assert!(m.is_valid_for(Encoding::Sq8));
            assert!(!m.is_valid_for(Encoding::Binary));
        }
        // Hamming valid only on Binary.
        assert!(Metric::Hamming.is_valid_for(Encoding::Binary));
        for e in [Encoding::F32, Encoding::F16, Encoding::Sq8] {
            assert!(!Metric::Hamming.is_valid_for(e));
        }
    }
}
