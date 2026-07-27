//! Distance kernels — simsimd-backed implementations.
//!
//! Per ADR-035 D-2 the v1.0 distance substrate is the
//! [`simsimd`](https://crates.io/crates/simsimd) crate (Apache
//! 2.0). Runtime AVX-512 / AVX-2 / NEON / SVE / SVE2 dispatch is
//! handled by simsimd's CPU-feature detection; this module ships
//! only the type-safe wrappers and the byte-slice plumbing.
//!
//! ## Trait surface (Slice A scaffold)
//!
//! [`DistanceKernel`] is byte-oriented so the same dispatch
//! surface handles f32 (4 bytes/dim), f16 (2 bytes/dim), SQ8
//! (1 byte/dim), and binary (1 bit/dim, packed). Slice A
//! shipped the trait with [`unimplemented!`] defaults; Slice B
//! adds the ten concrete kernels per ADR-035 §D-2:
//!
//! - L2 / IP / Cosine on F32 — [`L2F32`], [`IpF32`], [`CosineF32`].
//! - L2 / IP / Cosine on F16 — [`L2F16`], [`IpF16`], [`CosineF16`].
//! - L2 / IP / Cosine on SQ8 — [`L2Sq8`], [`IpSq8`], [`CosineSq8`].
//! - Hamming on Binary       — [`HammingBinary`].
//!
//! ## Byte-slice contract
//!
//! Callers pass `&[u8]` slices that the kernel reinterprets via
//! [`bytemuck::cast_slice`] to the appropriate typed view. The
//! arena allocator (Slice F.1) guarantees pointer alignment for
//! the encoding it stores; hot-path callers do not pay an
//! alignment-check cost in release. In debug builds the kernels
//! `debug_assert!` the slice is correctly aligned and sized.
//! RaBitQ is the one stateful kernel here: aligned payloads do
//! not self-describe `dim` (for example 128 B covers many nearby
//! dimensions), so [`L2RaBitQSym`] carries the dimension.
//!
//! ## Return-value convention
//!
//! Per [`Metric`] semantics:
//! - `L2` / `Hamming` → lower is closer.
//! - `Ip` / `Cosine` → higher is closer (the trait surface still
//!   returns `f32` distance; callers reverse the comparison).
//!
//! For Cosine specifically, simsimd returns the **cosine distance**
//! (`1 - cos(θ)` ∈ `[0, 2]`); callers that want similarity
//! directly should subtract from `1.0`. The kernel surface
//! reports the raw simsimd value to keep the trait neutral.

use simsimd::{BinarySimilarity, SpatialSimilarity};

use crate::{Encoding, Metric};

/// SIMD-backed distance kernel.
///
/// Implementors are stateless. Per ADR-035 D-2 the v1.0 dispatch
/// is via runtime CPU detection inside `simsimd`; the trait
/// surface is encoding- and metric-tagged so callers can refuse
/// unsupported pairs at dispatch time.
pub trait DistanceKernel: Send + Sync {
    /// Compute the distance / similarity between two encoded
    /// vector byte slices. Slice lengths must equal
    /// `encoding().bytes_per_vector_unaligned(dim)` for the
    /// arena's `dim`; impls `debug_assert!` length and alignment.
    ///
    /// Return value follows the [`Metric`] convention:
    /// - L2 / Hamming → lower is closer.
    /// - IP / Cosine → see module docs.
    fn distance(&self, a: &[u8], b: &[u8]) -> f32;

    /// Which metric this kernel computes.
    fn metric(&self) -> Metric;

    /// Which encoding this kernel consumes.
    fn encoding(&self) -> Encoding;
}

// ─── helpers ──────────────────────────────────────────────────────

/// Cast a `&[u8]` slice to a typed view via `bytemuck`. Panics
/// in debug if alignment or length is wrong; in release, relies
/// on `bytemuck::cast_slice`'s safety checks (which themselves
/// panic on misalignment but inline well).
#[inline]
fn cast_view<T: bytemuck::Pod>(bytes: &[u8]) -> &[T] {
    debug_assert!(
        bytes.as_ptr() as usize % std::mem::align_of::<T>() == 0,
        "DistanceKernel input not aligned for {:?}",
        std::any::type_name::<T>()
    );
    debug_assert!(
        bytes.len() % std::mem::size_of::<T>() == 0,
        "DistanceKernel input length not a multiple of {:?} size",
        std::any::type_name::<T>()
    );
    bytemuck::cast_slice(bytes)
}

/// Funnel the simsimd `Option<Distance>` result down to `f32`
/// and panic on dispatch failure (which only occurs on length
/// mismatch — caught by `debug_assert!` above).
#[inline]
fn unwrap_distance(name: &'static str, d: Option<simsimd::Distance>) -> f32 {
    match d {
        Some(v) => v as f32,
        None => panic!("simsimd kernel {name} returned None — input length mismatch"),
    }
}

// Compile-time assertion: simsimd::f16 must be 2 bytes wide
// (single IEEE-754 half-precision) for the byte-slice cast in
// `cast_view_f16` to be sound. If a future simsimd release
// changes the layout this `const _` will refuse to compile.
const _: () = assert!(
    std::mem::size_of::<simsimd::f16>() == 2,
    "simsimd::f16 is expected to be 2 bytes wide"
);
const _: () = assert!(
    std::mem::align_of::<simsimd::f16>() <= 2,
    "simsimd::f16 alignment must be ≤ 2 bytes"
);

/// Reinterpret a `&[u8]` byte slice as a `&[simsimd::f16]` view.
///
/// simsimd's `f16` is not `bytemuck::Pod` (no impl is exposed by
/// the upstream crate), so the cast goes through `unsafe`. The
/// upstream type IS a simple 2-byte half-precision value (verified
/// by the `const _` size + alignment assertions above), so the
/// reinterpretation is sound when the byte slice meets the
/// alignment + length contract that the arena allocator
/// guarantees.
#[allow(unsafe_code)]
#[inline]
fn cast_view_f16(bytes: &[u8]) -> &[simsimd::f16] {
    debug_assert!(
        bytes.as_ptr() as usize % std::mem::align_of::<simsimd::f16>() == 0,
        "DistanceKernel f16 input not aligned for simsimd::f16"
    );
    debug_assert!(
        bytes.len() % std::mem::size_of::<simsimd::f16>() == 0,
        "DistanceKernel f16 input length not a multiple of f16 size"
    );
    // SAFETY:
    // - `simsimd::f16` has size = 2 bytes (compile-time assertion
    //   above), so the slice length / element count math is sound.
    // - alignment of `simsimd::f16` is ≤ 2 bytes (compile-time
    //   assertion above); `debug_assert!` above checks the slice
    //   is aligned, and the arena allocator (Slice F.1 contract)
    //   guarantees alignment in release.
    // - `simsimd::f16` is the upstream half-precision wrapper and
    //   has no padding bytes / niches that would make any 2-byte
    //   bit pattern an invalid value (a `f16` represents NaN /
    //   ±Inf / subnormal / normal — every bit pattern is a valid
    //   IEEE-754 half).
    // - the resulting slice's lifetime is tied to the input
    //   slice's lifetime (same `'_`), so no use-after-free.
    unsafe {
        std::slice::from_raw_parts(
            bytes.as_ptr().cast::<simsimd::f16>(),
            bytes.len() / std::mem::size_of::<simsimd::f16>(),
        )
    }
}

// ─── F32 kernels ──────────────────────────────────────────────────

/// L2 (squared Euclidean) on F32.
#[derive(Debug, Clone, Copy, Default)]
pub struct L2F32;

impl DistanceKernel for L2F32 {
    #[inline]
    fn distance(&self, a: &[u8], b: &[u8]) -> f32 {
        let av: &[f32] = cast_view(a);
        let bv: &[f32] = cast_view(b);
        debug_assert_eq!(av.len(), bv.len(), "kernel length mismatch");
        unwrap_distance("L2F32", f32::sqeuclidean(av, bv))
    }

    #[inline]
    fn metric(&self) -> Metric {
        Metric::L2
    }

    #[inline]
    fn encoding(&self) -> Encoding {
        Encoding::F32
    }
}

/// Inner product on F32.
#[derive(Debug, Clone, Copy, Default)]
pub struct IpF32;

impl DistanceKernel for IpF32 {
    #[inline]
    fn distance(&self, a: &[u8], b: &[u8]) -> f32 {
        let av: &[f32] = cast_view(a);
        let bv: &[f32] = cast_view(b);
        debug_assert_eq!(av.len(), bv.len(), "kernel length mismatch");
        unwrap_distance("IpF32", f32::dot(av, bv))
    }

    #[inline]
    fn metric(&self) -> Metric {
        Metric::Ip
    }

    #[inline]
    fn encoding(&self) -> Encoding {
        Encoding::F32
    }
}

/// Cosine distance on F32 (returns `1 - cos(θ)`).
#[derive(Debug, Clone, Copy, Default)]
pub struct CosineF32;

impl DistanceKernel for CosineF32 {
    #[inline]
    fn distance(&self, a: &[u8], b: &[u8]) -> f32 {
        let av: &[f32] = cast_view(a);
        let bv: &[f32] = cast_view(b);
        debug_assert_eq!(av.len(), bv.len(), "kernel length mismatch");
        unwrap_distance("CosineF32", f32::cosine(av, bv))
    }

    #[inline]
    fn metric(&self) -> Metric {
        Metric::Cosine
    }

    #[inline]
    fn encoding(&self) -> Encoding {
        Encoding::F32
    }
}

// ─── F16 kernels ──────────────────────────────────────────────────
//
// simsimd exports its own `f16` newtype that is `Pod` (verified
// by the bytemuck `cast_slice` call below — compilation gates
// the assertion). The arena layout stores f16 vectors as
// 2-byte LE half-precision values per ADR-035 §5.2.

/// L2 (squared Euclidean) on F16.
#[derive(Debug, Clone, Copy, Default)]
pub struct L2F16;

impl DistanceKernel for L2F16 {
    #[inline]
    fn distance(&self, a: &[u8], b: &[u8]) -> f32 {
        let av: &[simsimd::f16] = cast_view_f16(a);
        let bv: &[simsimd::f16] = cast_view_f16(b);
        debug_assert_eq!(av.len(), bv.len(), "kernel length mismatch");
        unwrap_distance("L2F16", simsimd::f16::sqeuclidean(av, bv))
    }

    #[inline]
    fn metric(&self) -> Metric {
        Metric::L2
    }

    #[inline]
    fn encoding(&self) -> Encoding {
        Encoding::F16
    }
}

/// Inner product on F16.
#[derive(Debug, Clone, Copy, Default)]
pub struct IpF16;

impl DistanceKernel for IpF16 {
    #[inline]
    fn distance(&self, a: &[u8], b: &[u8]) -> f32 {
        let av: &[simsimd::f16] = cast_view_f16(a);
        let bv: &[simsimd::f16] = cast_view_f16(b);
        debug_assert_eq!(av.len(), bv.len(), "kernel length mismatch");
        unwrap_distance("IpF16", simsimd::f16::dot(av, bv))
    }

    #[inline]
    fn metric(&self) -> Metric {
        Metric::Ip
    }

    #[inline]
    fn encoding(&self) -> Encoding {
        Encoding::F16
    }
}

/// Cosine distance on F16.
#[derive(Debug, Clone, Copy, Default)]
pub struct CosineF16;

impl DistanceKernel for CosineF16 {
    #[inline]
    fn distance(&self, a: &[u8], b: &[u8]) -> f32 {
        let av: &[simsimd::f16] = cast_view_f16(a);
        let bv: &[simsimd::f16] = cast_view_f16(b);
        debug_assert_eq!(av.len(), bv.len(), "kernel length mismatch");
        unwrap_distance("CosineF16", simsimd::f16::cosine(av, bv))
    }

    #[inline]
    fn metric(&self) -> Metric {
        Metric::Cosine
    }

    #[inline]
    fn encoding(&self) -> Encoding {
        Encoding::F16
    }
}

// ─── SQ8 kernels ──────────────────────────────────────────────────
//
// SQ8 vectors are stored as i8 per dimension (per-dim min/max
// codebook in `Sq8Params` is consulted at decode time). The
// distance kernel computes raw integer distance; rescore against
// f32 (Slice E.2/E.3) recovers the precision lost at quantization.

/// L2 (squared Euclidean) on SQ8 (i8).
#[derive(Debug, Clone, Copy, Default)]
pub struct L2Sq8;

impl DistanceKernel for L2Sq8 {
    #[inline]
    fn distance(&self, a: &[u8], b: &[u8]) -> f32 {
        let av: &[i8] = cast_view(a);
        let bv: &[i8] = cast_view(b);
        debug_assert_eq!(av.len(), bv.len(), "kernel length mismatch");
        unwrap_distance("L2Sq8", i8::sqeuclidean(av, bv))
    }

    #[inline]
    fn metric(&self) -> Metric {
        Metric::L2
    }

    #[inline]
    fn encoding(&self) -> Encoding {
        Encoding::Sq8
    }
}

/// Inner product on SQ8 (i8).
#[derive(Debug, Clone, Copy, Default)]
pub struct IpSq8;

impl DistanceKernel for IpSq8 {
    #[inline]
    fn distance(&self, a: &[u8], b: &[u8]) -> f32 {
        let av: &[i8] = cast_view(a);
        let bv: &[i8] = cast_view(b);
        debug_assert_eq!(av.len(), bv.len(), "kernel length mismatch");
        unwrap_distance("IpSq8", i8::dot(av, bv))
    }

    #[inline]
    fn metric(&self) -> Metric {
        Metric::Ip
    }

    #[inline]
    fn encoding(&self) -> Encoding {
        Encoding::Sq8
    }
}

/// Cosine distance on SQ8 (i8).
#[derive(Debug, Clone, Copy, Default)]
pub struct CosineSq8;

impl DistanceKernel for CosineSq8 {
    #[inline]
    fn distance(&self, a: &[u8], b: &[u8]) -> f32 {
        let av: &[i8] = cast_view(a);
        let bv: &[i8] = cast_view(b);
        debug_assert_eq!(av.len(), bv.len(), "kernel length mismatch");
        unwrap_distance("CosineSq8", i8::cosine(av, bv))
    }

    #[inline]
    fn metric(&self) -> Metric {
        Metric::Cosine
    }

    #[inline]
    fn encoding(&self) -> Encoding {
        Encoding::Sq8
    }
}

// ─── Binary kernel ────────────────────────────────────────────────

/// Hamming distance on packed binary vectors (`u8` slices,
/// 8 bits per byte). Per ADR-035 §S-1, the arena layout pads
/// binary vectors to 128-byte alignment; this kernel itself
/// only requires byte alignment, but the arena's stricter
/// guarantee is preserved upstream.
#[derive(Debug, Clone, Copy, Default)]
pub struct HammingBinary;

impl DistanceKernel for HammingBinary {
    #[inline]
    fn distance(&self, a: &[u8], b: &[u8]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "binary vectors length mismatch");
        unwrap_distance("HammingBinary", u8::hamming(a, b))
    }

    #[inline]
    fn metric(&self) -> Metric {
        Metric::Hamming
    }

    #[inline]
    fn encoding(&self) -> Encoding {
        Encoding::Binary
    }
}

// ─── RaBitQ symmetric build kernel ───────────────────────────────

/// Symmetric L2 estimator over two RaBitQ payloads.
///
/// This kernel exists for Vamana build, whose hot path needs node-pair
/// distances (`payload ↔ payload`). Slice-1's asymmetric theorem covers a
/// prepared full-precision query; build has no such query. For payload `a`,
/// RaBitQ stores signs `s_a = sign(y_a)`, `f_a = <xbar_a, obar_a>`, and
/// residual norm `n_a`. Orthogonality gives
/// `<xbar_a,xbar_b> = <s_a,s_b>/D = (D - 2H)/D`, where `H` is XOR-popcount
/// over the sign codes. The symmetric build kernel intentionally uses this
/// scaled-codeword inner product directly:
///
/// `est<xbar_a,xbar_b> = (D - 2H) / D`
///
/// and `est d2 = max(0, n_a^2 + n_b^2 - 2*n_a*n_b*est_ip)`. If either
/// residual norm is zero, the degenerate side is the centroid and the formula
/// returns the other residual norm squared without a special case.
///
/// Honesty: the symmetric kernel is the decoded-codeword distance, not Gao &
/// Long Thm 3.2. Its angle signal is monotone in the residual angle with
/// `E[est_ip] = 1 - 2theta/pi`; the theorem's `1/f` de-bias belongs only to
/// the asymmetric serve estimator, where the query is full-precision and
/// independent of the data code. The clamp remains a guard for estimator and
/// rounding noise, but self-distance is exact via `H = 0`. Budget at 768d:
/// about 12 XOR+POPCNT words plus scalar combine, while fetching 128 B instead
/// of SQ8's 768 B.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L2RaBitQSym {
    dim: usize,
}

impl L2RaBitQSym {
    #[inline]
    #[must_use]
    pub const fn new(dim: usize) -> Self {
        Self { dim }
    }

    #[inline]
    #[must_use]
    pub const fn dim(self) -> usize {
        self.dim
    }
}

impl DistanceKernel for L2RaBitQSym {
    #[inline]
    fn distance(&self, a: &[u8], b: &[u8]) -> f32 {
        let code_bytes = self.dim.div_ceil(8);
        let unaligned = code_bytes + 8;
        debug_assert!(a.len() >= unaligned, "short RaBitQ payload");
        debug_assert_eq!(a.len(), b.len(), "RaBitQ vectors length mismatch");
        let h = hamming_prefix(&a[..code_bytes], &b[..code_bytes]);
        let f_a = f32::from_le_bytes(a[code_bytes..code_bytes + 4].try_into().unwrap());
        let n_a = f32::from_le_bytes(a[code_bytes + 4..code_bytes + 8].try_into().unwrap());
        let f_b = f32::from_le_bytes(b[code_bytes..code_bytes + 4].try_into().unwrap());
        let n_b = f32::from_le_bytes(b[code_bytes + 4..code_bytes + 8].try_into().unwrap());
        // `f` is still part of the stable RaBitQ payload layout for the
        // asymmetric serve estimator. The symmetric build kernel deliberately
        // does not apply the serve-side `1/(f_a*f_b)` de-bias.
        debug_assert!(f_a.is_finite());
        debug_assert!(f_b.is_finite());
        let ip = (self.dim as f64 - 2.0 * h as f64) / self.dim as f64;
        let n_a = f64::from(n_a);
        let n_b = f64::from(n_b);
        let est = n_a * n_a + n_b * n_b - 2.0 * n_a * n_b * ip;
        est.max(0.0) as f32
    }

    #[inline]
    fn metric(&self) -> Metric {
        Metric::L2
    }

    #[inline]
    fn encoding(&self) -> Encoding {
        Encoding::RaBitQ
    }
}

#[inline]
fn hamming_prefix(a: &[u8], b: &[u8]) -> u32 {
    debug_assert_eq!(a.len(), b.len(), "RaBitQ code length mismatch");
    let mut h = 0u32;
    let a_words = a.chunks_exact(8);
    let b_words = b.chunks_exact(8);
    let a_rem = a_words.remainder();
    let b_rem = b_words.remainder();
    for (wa, wb) in a_words.zip(b_words) {
        let wa = u64::from_le_bytes(wa.try_into().unwrap());
        let wb = u64::from_le_bytes(wb.try_into().unwrap());
        h += (wa ^ wb).count_ones();
    }
    for (&ba, &bb) in a_rem.iter().zip(b_rem) {
        h += (ba ^ bb).count_ones();
    }
    h
}

#[cfg(test)]
mod tests {
    //! Module-level smoke tests — verifies metric / encoding
    //! tagging and Send + Sync. Reference-value correctness
    //! tests live in `tests/distance.rs` (Slice B integration
    //! tests).

    use super::*;
    use crate::quantizer::{RaBitQCodebook, RaBitQParams, RaBitQTrainer};

    #[test]
    fn f32_kernels_report_their_metric_and_encoding() {
        assert_eq!(L2F32.metric(), Metric::L2);
        assert_eq!(L2F32.encoding(), Encoding::F32);
        assert_eq!(IpF32.metric(), Metric::Ip);
        assert_eq!(IpF32.encoding(), Encoding::F32);
        assert_eq!(CosineF32.metric(), Metric::Cosine);
        assert_eq!(CosineF32.encoding(), Encoding::F32);
    }

    #[test]
    fn f16_kernels_report_their_metric_and_encoding() {
        assert_eq!(L2F16.metric(), Metric::L2);
        assert_eq!(L2F16.encoding(), Encoding::F16);
        assert_eq!(IpF16.metric(), Metric::Ip);
        assert_eq!(IpF16.encoding(), Encoding::F16);
        assert_eq!(CosineF16.metric(), Metric::Cosine);
        assert_eq!(CosineF16.encoding(), Encoding::F16);
    }

    #[test]
    fn sq8_kernels_report_their_metric_and_encoding() {
        assert_eq!(L2Sq8.metric(), Metric::L2);
        assert_eq!(L2Sq8.encoding(), Encoding::Sq8);
        assert_eq!(IpSq8.metric(), Metric::Ip);
        assert_eq!(IpSq8.encoding(), Encoding::Sq8);
        assert_eq!(CosineSq8.metric(), Metric::Cosine);
        assert_eq!(CosineSq8.encoding(), Encoding::Sq8);
    }

    #[test]
    fn binary_kernel_reports_hamming_on_binary() {
        assert_eq!(HammingBinary.metric(), Metric::Hamming);
        assert_eq!(HammingBinary.encoding(), Encoding::Binary);
    }

    #[test]
    fn rabitq_kernel_reports_l2_on_rabitq() {
        let k = L2RaBitQSym::new(768);
        assert_eq!(k.dim(), 768);
        assert_eq!(k.metric(), Metric::L2);
        assert_eq!(k.encoding(), Encoding::RaBitQ);
    }

    #[test]
    fn rabitq_symmetric_kernel_clamps_self_distance_to_zero() {
        let cb = identity_rabitq(8);
        let payload = cb
            .encode_aligned(&[1.0, -2.0, 0.5, 3.0, -1.0, 0.25, 0.75, -0.5])
            .unwrap();
        assert_eq!(L2RaBitQSym::new(8).distance(&payload, &payload), 0.0);
    }

    #[test]
    fn rabitq_symmetric_kernel_handles_degenerate_centroid_side() {
        let cb = identity_rabitq(8);
        let centroid = vec![0.0; 8];
        let other = vec![1.0, -2.0, 0.5, 3.0, -1.0, 0.25, 0.75, -0.5];
        let p_centroid = cb.encode_aligned(&centroid).unwrap();
        let p_other = cb.encode_aligned(&other).unwrap();
        let expected = other.iter().map(|x| x * x).sum::<f32>();
        let got = L2RaBitQSym::new(8).distance(&p_centroid, &p_other);
        assert!((got - expected).abs() < 1e-5, "{got} != {expected}");
    }

    #[test]
    fn rabitq_symmetric_kernel_preserves_correlated_pair_ranking() {
        let dim = 128;
        let corpus = cluster_corpus(0xA209_0001, 16, 24, dim, 0.02);
        let refs: Vec<&[f32]> = corpus.iter().map(Vec::as_slice).collect();
        let cb = RaBitQTrainer.train(&refs, 0x7580_0002).unwrap();
        let payloads: Vec<Vec<u8>> = corpus
            .iter()
            .map(|v| cb.encode_aligned(v).unwrap())
            .collect();
        let kernel = L2RaBitQSym::new(dim);

        let mut pairs = Vec::new();
        for cluster in 0..16 {
            let base = cluster * 24;
            for j in 1..13 {
                pairs.push((base, base + j));
            }
        }
        for cluster in 0..16 {
            let next = ((cluster + 7) % 16) * 24;
            let base = cluster * 24;
            for j in 0..6 {
                pairs.push((base + j, next + j));
            }
        }

        let true_d2: Vec<f64> = pairs
            .iter()
            .map(|&(a, b)| l2_sq_f32(&corpus[a], &corpus[b]) as f64)
            .collect();
        let sym_d2: Vec<f64> = pairs
            .iter()
            .map(|&(a, b)| f64::from(kernel.distance(&payloads[a], &payloads[b])))
            .collect();

        let rho = spearman(&true_d2, &sym_d2);
        let same_zero_rate = sym_d2[..192].iter().filter(|&&d| d == 0.0).count() as f64 / 192.0;
        eprintln!(
            "W1_RABITQ_SYM_RANK_FIDELITY rho={rho:.4} pin=0.70 same_cluster_zero_rate={same_zero_rate:.4} pin<=0.05"
        );
        assert!(
            rho >= 0.70,
            "W1 RaBitQ symmetric rank-fidelity rho={rho:.4} < 0.70"
        );
        assert!(
            same_zero_rate <= 0.05,
            "W1 RaBitQ same-cluster zero-tie rate {same_zero_rate:.4} > 0.05"
        );
    }

    fn identity_rabitq(dim: usize) -> RaBitQCodebook {
        let mut rotation = vec![0.0; dim * dim];
        for d in 0..dim {
            rotation[d * dim + d] = 1.0;
        }
        RaBitQCodebook::from_params(RaBitQParams::try_new(dim, vec![0.0; dim], rotation).unwrap())
    }

    fn cluster_corpus(
        seed: u32,
        clusters: usize,
        per: usize,
        dim: usize,
        sigma: f32,
    ) -> Vec<Vec<f32>> {
        let mut rng = Xs32::new(seed);
        let centers: Vec<Vec<f32>> = (0..clusters)
            .map(|_| (0..dim).map(|_| rng.signed()).collect())
            .collect();
        let mut out = Vec::with_capacity(clusters * per);
        for c in &centers {
            for _ in 0..per {
                out.push(c.iter().map(|&cc| cc + rng.gauss() * sigma).collect());
            }
        }
        out
    }

    fn l2_sq_f32(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| {
                let d = x - y;
                d * d
            })
            .sum()
    }

    fn spearman(a: &[f64], b: &[f64]) -> f64 {
        debug_assert_eq!(a.len(), b.len());
        let ar = ranks(a);
        let br = ranks(b);
        pearson(&ar, &br)
    }

    fn ranks(values: &[f64]) -> Vec<f64> {
        let mut order: Vec<usize> = (0..values.len()).collect();
        order.sort_by(|&i, &j| values[i].total_cmp(&values[j]).then(i.cmp(&j)));
        let mut ranks = vec![0.0; values.len()];
        let mut i = 0;
        while i < order.len() {
            let mut j = i + 1;
            while j < order.len() && values[order[i]] == values[order[j]] {
                j += 1;
            }
            let rank = (i + j - 1) as f64 / 2.0;
            for k in i..j {
                ranks[order[k]] = rank;
            }
            i = j;
        }
        ranks
    }

    fn pearson(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len() as f64;
        let mean_a = a.iter().sum::<f64>() / n;
        let mean_b = b.iter().sum::<f64>() / n;
        let mut num = 0.0;
        let mut den_a = 0.0;
        let mut den_b = 0.0;
        for (&x, &y) in a.iter().zip(b) {
            let dx = x - mean_a;
            let dy = y - mean_b;
            num += dx * dy;
            den_a += dx * dx;
            den_b += dy * dy;
        }
        num / (den_a.sqrt() * den_b.sqrt())
    }

    struct Xs32(u32);
    impl Xs32 {
        fn new(seed: u32) -> Self {
            Self(if seed == 0 { 0xDEAD_BEEF } else { seed })
        }

        fn next_u32(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            self.0 = x;
            x
        }

        fn signed(&mut self) -> f32 {
            (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
        }

        fn gauss(&mut self) -> f32 {
            let u1 = (self.next_u32() as f32 / u32::MAX as f32).max(1e-10);
            let u2 = self.next_u32() as f32 / u32::MAX as f32;
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
        }
    }

    #[test]
    fn kernels_are_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<L2F32>();
        assert_send_sync::<IpF32>();
        assert_send_sync::<CosineF32>();
        assert_send_sync::<L2F16>();
        assert_send_sync::<IpF16>();
        assert_send_sync::<CosineF16>();
        assert_send_sync::<L2Sq8>();
        assert_send_sync::<IpSq8>();
        assert_send_sync::<CosineSq8>();
        assert_send_sync::<HammingBinary>();
        assert_send_sync::<L2RaBitQSym>();
    }
}
