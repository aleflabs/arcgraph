//! M4-64b — SIMD RRF (Reciprocal-Rank Fusion) score-vector helper.
//!
//! Per ADR-038 amendment-02 §M4.f the RrfFusion scope is "vectorize the
//! per-rank-list `1 / (k + rank)` accumulation. AVX2/NEON has no native
//! f32 reciprocal-divide that's strictly correct, so use
//! `_mm256_div_ps` (or NEON's `vdivq_f32`) on packed f32, then sum into
//! the candidate-score map."
//!
//! # Latency budget (PD-5)
//!
//! Per ADR-038 amendment-03 §Structural-1 the slice-level acceptance is
//! **≥1.5× speedup vs scalar**. Concrete per-arch budget:
//!
//! - **AVX2**: ~14 cycle latency per `_mm256_div_ps` (Ice Lake; per
//!   Intel Optimization Reference §15.5.2); 8 × f32 lanes per register
//!   ⇒ effective ~2 cycles/element. Scalar `divss` is also ~14 cycles
//!   per scalar div, so the speedup at 8 lanes/iter is ≥ 4× over a
//!   tight scalar loop.
//! - **NEON**: ~6 cycle latency per `vdivq_f32` (Cortex-A78 + Apple
//!   M-series; per ARM SOG §3.18 + Apple Silicon dispatch
//!   characterization); 4 × f32 lanes per register ⇒ effective ~1.5
//!   cycles/element. Speedup ≥ 3× over scalar `f32 / f32`.
//! - **Scalar**: baseline; one f32 div per iter.
//!
//! # Why f32 (not f64)?
//!
//! RRF score precision is bounded by the rank-list cardinality (typical
//! K ≤ 1000), which fits comfortably in f32's ~7-decimal-digit precision.
//! The downstream HashMap aggregation widens to f64 for the per-node
//! sum (preserving precision across many lists); the per-list score
//! generation runs at f32 to maximize SIMD lane density.
//!
//! # Production precision
//!
//! This remains a benchmarkable SIMD utility. Public hybrid fusion uses
//! [`crate::executor::fusion::rrf_fuse`], whose exact `f64` arithmetic is
//! shared with `graph.search`. The scalar fallback here preserves the
//! SAME bytewise-equivalent f32 values per the
//! `m4_64b_scalar_vs_vector_equivalence_proptest`.

use super::SimdBackend;

/// Compute a vector of RRF contributions: `result[i] = 1.0 / (k + (i +
/// 1))` for `i` in `0..rank_count`. The `+1` reflects the 1-indexed
/// rank convention per Cormack SIGIR 2009.
///
/// # Backend selection
///
/// Same dispatch as [`super::filter::simd_filter_i64_cmp`].
///
/// # Bytewise equivalence
///
/// All backends produce bytewise-identical f32 vectors for the same
/// input. The proptest pins this invariant: scalar and SIMD outputs
/// MUST agree byte-for-byte. The `vdivq_f32` / `_mm256_div_ps`
/// intrinsics are specified as IEEE-754 division, the same standard
/// the scalar `f32 / f32` operator follows; lane width does not affect
/// precision.
#[must_use]
pub fn simd_rrf_scores(k: u32, rank_count: usize) -> Vec<f32> {
    if rank_count == 0 {
        return Vec::new();
    }
    match SimdBackend::detect() {
        #[cfg(target_arch = "x86_64")]
        SimdBackend::X86Avx2 => {
            // SAFETY: AVX2 gated by `is_x86_feature_detected!`; the
            // inner `#[target_feature]` re-asserts. The bulk loop
            // bounds the index by `rank_count`; `_mm256_storeu_ps`
            // tolerates unaligned stores.
            unsafe { x86_avx2::rrf_scores(k, rank_count) }
        }
        #[cfg(target_arch = "aarch64")]
        SimdBackend::AArch64Neon => {
            // SAFETY: NEON gated; same bulk-loop bounds + unaligned-
            // store invariants.
            unsafe { aarch64_neon::rrf_scores(k, rank_count) }
        }
        _ => scalar::rrf_scores(k, rank_count),
    }
}

/// Scalar fallback. Always available.
pub mod scalar {
    /// Baseline scalar implementation. Used as the equivalence
    /// reference in the proptest + the bench's scalar arm.
    #[must_use]
    pub fn rrf_scores(k: u32, rank_count: usize) -> Vec<f32> {
        let k_f = k as f32;
        let mut out = Vec::with_capacity(rank_count);
        for rank0 in 0..rank_count {
            let rank_1based = (rank0 + 1) as f32;
            out.push(1.0_f32 / (k_f + rank_1based));
        }
        out
    }
}

#[cfg(target_arch = "x86_64")]
pub mod x86_avx2 {
    //! AVX2 backend. 8 × f32 lanes per 256-bit `__m256`. One register
    //! per iteration ⇒ 8 ranks / iter.

    use core::arch::x86_64::*;

    const LANES_PER_ITER: usize = 8;

    /// # Safety
    ///
    /// Caller MUST guarantee AVX2 is available (gated upstream via
    /// `is_x86_feature_detected!`).
    #[target_feature(enable = "avx2")]
    pub unsafe fn rrf_scores(k: u32, rank_count: usize) -> Vec<f32> {
        let mut out: Vec<f32> = vec![0.0; rank_count];
        let out_ptr = out.as_mut_ptr();
        let bulk_end = rank_count - (rank_count % LANES_PER_ITER);

        let k_f = k as f32;
        let k_v = _mm256_set1_ps(k_f);
        let one_v = _mm256_set1_ps(1.0_f32);

        let mut i = 0;
        while i < bulk_end {
            let ranks = _mm256_setr_ps(
                (i + 1) as f32,
                (i + 2) as f32,
                (i + 3) as f32,
                (i + 4) as f32,
                (i + 5) as f32,
                (i + 6) as f32,
                (i + 7) as f32,
                (i + 8) as f32,
            );
            let denom = _mm256_add_ps(k_v, ranks);
            let result = _mm256_div_ps(one_v, denom);
            // SAFETY: `_mm256_storeu_ps` tolerates unaligned stores;
            // out_ptr.add(i) lies within the pre-allocated `rank_count`
            // f32s.
            unsafe { _mm256_storeu_ps(out_ptr.add(i), result) };
            i += LANES_PER_ITER;
        }

        // Scalar tail.
        while i < rank_count {
            let denom = k_f + (i + 1) as f32;
            // SAFETY: i < rank_count = out's length.
            unsafe { out_ptr.add(i).write(1.0_f32 / denom) };
            i += 1;
        }
        out
    }
}

#[cfg(target_arch = "aarch64")]
pub mod aarch64_neon {
    //! NEON backend. 4 × f32 lanes per 128-bit `float32x4_t`. One
    //! register per iteration ⇒ 4 ranks / iter.

    use core::arch::aarch64::*;

    const LANES_PER_ITER: usize = 4;

    /// # Safety
    ///
    /// Caller MUST guarantee NEON is available (gated upstream via
    /// `std::arch::is_aarch64_feature_detected!`).
    #[target_feature(enable = "neon")]
    pub unsafe fn rrf_scores(k: u32, rank_count: usize) -> Vec<f32> {
        let mut out: Vec<f32> = vec![0.0; rank_count];
        let out_ptr = out.as_mut_ptr();
        let bulk_end = rank_count - (rank_count % LANES_PER_ITER);

        let k_f = k as f32;
        let k_v = vdupq_n_f32(k_f);
        let one_v = vdupq_n_f32(1.0_f32);

        let mut i = 0;
        while i < bulk_end {
            let r = [
                (i + 1) as f32,
                (i + 2) as f32,
                (i + 3) as f32,
                (i + 4) as f32,
            ];
            // SAFETY: unaligned load tolerated; `r` is a stack array of
            // 4 f32s, `vld1q_f32` reads exactly 4 contiguous f32s.
            let ranks = unsafe { vld1q_f32(r.as_ptr()) };
            let denom = vaddq_f32(k_v, ranks);
            let result = vdivq_f32(one_v, denom);
            // SAFETY: `vst1q_f32` tolerates unaligned stores; out_ptr.add(i)
            // lies within the pre-allocated `rank_count` f32s.
            unsafe { vst1q_f32(out_ptr.add(i), result) };
            i += LANES_PER_ITER;
        }

        // Scalar tail.
        while i < rank_count {
            let denom = k_f + (i + 1) as f32;
            // SAFETY: i < rank_count.
            unsafe { out_ptr.add(i).write(1.0_f32 / denom) };
            i += 1;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_scalar_rrf_known_values() {
        // Cormack SIGIR 2009 default k=60. rank=1 → 1/61 ≈ 0.0163934.
        let v = scalar::rrf_scores(60, 3);
        assert_eq!(v.len(), 3);
        assert!((v[0] - (1.0_f32 / 61.0)).abs() < 1e-7);
        assert!((v[1] - (1.0_f32 / 62.0)).abs() < 1e-7);
        assert!((v[2] - (1.0_f32 / 63.0)).abs() < 1e-7);
    }

    #[test]
    fn unit_scalar_rrf_zero_rank_count_is_empty() {
        let v = scalar::rrf_scores(60, 0);
        assert!(v.is_empty());
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn unit_x86_avx2_rrf_matches_scalar_bytewise() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("skipped: x86_64 AVX2 unavailable on this host");
            return;
        }
        for &(k, n) in &[(60_u32, 1), (60, 7), (60, 8), (60, 100), (1, 1024)] {
            let scalar = scalar::rrf_scores(k, n);
            // SAFETY: AVX2 gated.
            let simd = unsafe { x86_avx2::rrf_scores(k, n) };
            assert_eq!(
                scalar.len(),
                simd.len(),
                "AVX2 vs scalar length mismatch for (k={}, n={})",
                k,
                n
            );
            for i in 0..scalar.len() {
                // Bytewise equality via to_bits() — IEEE-754 division
                // is deterministic.
                assert_eq!(
                    scalar[i].to_bits(),
                    simd[i].to_bits(),
                    "AVX2 vs scalar bytewise mismatch at i={} (k={}, n={})",
                    i,
                    k,
                    n
                );
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn unit_aarch64_neon_rrf_matches_scalar_bytewise() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("skipped: AArch64 NEON unavailable on this host");
            return;
        }
        for &(k, n) in &[(60_u32, 1), (60, 7), (60, 8), (60, 100), (1, 1024)] {
            let scalar = scalar::rrf_scores(k, n);
            // SAFETY: NEON gated.
            let simd = unsafe { aarch64_neon::rrf_scores(k, n) };
            assert_eq!(
                scalar.len(),
                simd.len(),
                "NEON vs scalar length mismatch for (k={}, n={})",
                k,
                n
            );
            for i in 0..scalar.len() {
                assert_eq!(
                    scalar[i].to_bits(),
                    simd[i].to_bits(),
                    "NEON vs scalar bytewise mismatch at i={} (k={}, n={})",
                    i,
                    k,
                    n
                );
            }
        }
    }

    #[test]
    fn unit_dispatch_rrf_matches_scalar_baseline() {
        for &(k, n) in &[(60_u32, 5), (60, 100), (1, 33)] {
            let scalar = scalar::rrf_scores(k, n);
            let dispatch = simd_rrf_scores(k, n);
            assert_eq!(scalar.len(), dispatch.len());
            for i in 0..scalar.len() {
                assert_eq!(scalar[i].to_bits(), dispatch[i].to_bits());
            }
        }
    }

    #[test]
    fn unit_simd_path_handles_short_rank_counts() {
        // Boundary: rank_count < LANES_PER_ITER triggers scalar tail.
        for n in 0..=3_usize {
            let scalar = scalar::rrf_scores(60, n);
            let dispatch = simd_rrf_scores(60, n);
            assert_eq!(scalar.len(), dispatch.len());
            for i in 0..scalar.len() {
                assert_eq!(
                    scalar[i].to_bits(),
                    dispatch[i].to_bits(),
                    "boundary n={} bytewise mismatch at i={}",
                    n,
                    i
                );
            }
        }
    }
}
