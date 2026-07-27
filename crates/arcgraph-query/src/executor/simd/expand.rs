//! M4-64b — SIMD NodeId membership helper for ExpandOp's neighbor scan.
//!
//! Per ADR-038 amendment-02 §M4.f the ExpandOp scope is "vectorize the
//! linear neighbor-id scan against the per-source adjacency list. SIMD-
//! compare neighbor candidate IDs against a packed source-set; emit
//! matches."
//!
//! # Latency budget (PD-5)
//!
//! Per ADR-038 amendment-03 §Structural-1 the slice-level acceptance is
//! **≥1.5× speedup vs scalar**. Concrete per-arch budget at the
//! `simd_neighbor_match_mask` helper:
//!
//! - **AVX2**: 1 cycle per 4-lane `_mm256_cmpeq_epi64` (Ice Lake / Skylake-X
//!   per Intel Optimization Reference §15.5.1); the helper unrolls one
//!   register per iter for 4 candidate-ids per iter.
//! - **NEON**: ~1 cycle per 2-lane `vceqq_u64` (Cortex-A78 per ARM SOG
//!   §3.18); 2 candidate-ids per iter.
//! - **Scalar**: O(N × K) for N candidates and K targets.
//!
//! The expected speedup is **≥1.5×** over scalar at N≥1024, K∈\[1,4\].
//! Larger K (membership against many targets) sub-linear-degrades the
//! win because the broadcast-and-OR loop runs K times per candidate.
//!
//! # Operator wiring
//!
//! [`crate::executor::ops::expand::ExpandOp::with_dst_allow_set`]
//! consumes this helper after the substrate's `expand` returns the
//! per-source adjacency list. The substrate already filters by
//! rel-type and direction; the dst-allow-set is the orthogonal filter
//! the planner pushes down from a downstream `WHERE b.id IN [...]`
//! predicate (forward pin: pushdown wiring lands at M4-72; the helper
//! itself is exercised today via the unit + bench).

use super::SimdBackend;

/// Vectorized membership check: for each `candidate`, return `true` iff
/// it equals any element in `targets`.
///
/// Returns a `Vec<bool>` of length `candidates.len()` where `result[i]`
/// is the membership truth for `candidates[i]`.
///
/// # Empty `targets` semantics
///
/// `targets.is_empty()` returns `false` for every candidate. The
/// caller's wrapper code in [`crate::executor::ops::expand::ExpandOp`]
/// short-circuits on empty allow-sets BEFORE calling this helper (for
/// the "no-filter" case the substrate-returned edges pass through
/// unconditionally) — this helper preserves the strict semantic
/// regardless.
///
/// # Backend selection
///
/// Same dispatch shape as
/// [`super::filter::simd_filter_i64_cmp`]: runtime feature detection
/// picks AVX2 / NEON / scalar. The scalar fallback is a tight nested
/// loop (`for c in candidates { for t in targets { ... } }`).
///
/// # Why u64 (not NodeId)?
///
/// Decoupling from the `NodeId` newtype keeps the helper callable from
/// any module that needs raw u64 membership (the bench, future
/// observability hooks). Callers convert via `node_id.raw()`.
#[must_use]
pub fn simd_neighbor_match_mask(candidates: &[u64], targets: &[u64]) -> Vec<bool> {
    if targets.is_empty() {
        return vec![false; candidates.len()];
    }
    match SimdBackend::detect() {
        #[cfg(target_arch = "x86_64")]
        SimdBackend::X86Avx2 => {
            // SAFETY: AVX2 gated by `is_x86_feature_detected!`; the
            // inner `#[target_feature]` re-asserts at the function-
            // attribute level. The candidates slice's length bounds
            // the bulk loop; targets are loaded element-by-element via
            // broadcast. Unaligned loads via `_mm256_loadu_si256`.
            unsafe { x86_avx2::neighbor_match_mask(candidates, targets) }
        }
        #[cfg(target_arch = "aarch64")]
        SimdBackend::AArch64Neon => {
            // SAFETY: NEON gated above; same length / unaligned-load
            // invariants as the AVX2 path.
            unsafe { aarch64_neon::neighbor_match_mask(candidates, targets) }
        }
        _ => scalar::neighbor_match_mask(candidates, targets),
    }
}

/// Scalar fallback. Always available regardless of arch / runtime
/// detection.
pub mod scalar {
    /// Scalar baseline: O(N × K) nested loop with early-exit on first
    /// match. Used as the equivalence reference in the proptest +
    /// bench's scalar arm.
    #[must_use]
    pub fn neighbor_match_mask(candidates: &[u64], targets: &[u64]) -> Vec<bool> {
        let mut out = Vec::with_capacity(candidates.len());
        for &c in candidates {
            let mut hit = false;
            for &t in targets {
                if c == t {
                    hit = true;
                    break;
                }
            }
            out.push(hit);
        }
        out
    }
}

#[cfg(target_arch = "x86_64")]
pub mod x86_avx2 {
    //! AVX2 backend. 4 × u64 lanes per 256-bit `__m256i`; one register
    //! unrolled per iteration ⇒ 4 candidates / iter. The targets-loop
    //! broadcasts each target across all 4 lanes via
    //! `_mm256_set1_epi64x`, compares against the candidates with
    //! `_mm256_cmpeq_epi64`, and OR-accumulates the comparison register
    //! into the running mask.

    use core::arch::x86_64::*;

    const LANES_PER_ITER: usize = 4;

    /// # Safety
    ///
    /// Caller MUST guarantee AVX2 is available (gated upstream via
    /// `is_x86_feature_detected!`); `#[target_feature]` re-asserts the
    /// precondition.
    #[target_feature(enable = "avx2")]
    pub unsafe fn neighbor_match_mask(candidates: &[u64], targets: &[u64]) -> Vec<bool> {
        let len = candidates.len();
        let mut out: Vec<bool> = vec![false; len];
        let out_ptr = out.as_mut_ptr();
        let bulk_end = len - (len % LANES_PER_ITER);

        let mut i = 0;
        while i < bulk_end {
            // SAFETY: unaligned load tolerated; bulk-loop invariant
            // `i + LANES_PER_ITER <= bulk_end <= len` ensures bounds.
            let cands = unsafe { _mm256_loadu_si256(candidates.as_ptr().add(i).cast::<__m256i>()) };

            let mut acc = _mm256_setzero_si256();
            for &t in targets {
                let target_v = _mm256_set1_epi64x(t as i64);
                let cmp = _mm256_cmpeq_epi64(cands, target_v);
                acc = _mm256_or_si256(acc, cmp);
            }

            // SAFETY: lanes_to_mask4 is `#[target_feature(enable =
            // "avx2")]`; AVX2 gated upstream.
            let lanes = unsafe { lanes_to_mask4(acc) };
            for (lane, &mask_bit) in lanes.iter().enumerate() {
                // SAFETY: out_ptr valid for `len` writes; `i + lane <
                // bulk_end <= len`.
                unsafe { out_ptr.add(i + lane).write(mask_bit) };
            }
            i += LANES_PER_ITER;
        }

        // Scalar tail.
        while i < len {
            let c = candidates[i];
            let mut hit = false;
            for &t in targets {
                if c == t {
                    hit = true;
                    break;
                }
            }
            // SAFETY: i < len = out's length.
            unsafe { out_ptr.add(i).write(hit) };
            i += 1;
        }
        out
    }

    /// Extract 4 boolean lanes from a 256-bit comparison register
    /// (0xFF…FF per true lane, 0x00…00 per false lane).
    ///
    /// # Safety
    ///
    /// Caller MUST guarantee AVX2 (gated by `#[target_feature]`).
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn lanes_to_mask4(cmp: __m256i) -> [bool; 4] {
        let mut buf = [0_i64; 4];
        // SAFETY: unaligned-store-tolerant; `buf` is 32 bytes = one
        // 256-bit register.
        unsafe { _mm256_storeu_si256(buf.as_mut_ptr().cast::<__m256i>(), cmp) };
        [buf[0] != 0, buf[1] != 0, buf[2] != 0, buf[3] != 0]
    }
}

#[cfg(target_arch = "aarch64")]
pub mod aarch64_neon {
    //! NEON backend. 2 × u64 lanes per 128-bit `uint64x2_t`; one
    //! register unrolled per iteration ⇒ 2 candidates / iter. The
    //! targets-loop broadcasts each target via `vdupq_n_u64`, compares
    //! via `vceqq_u64`, and OR-accumulates via `vorrq_u64`.

    use core::arch::aarch64::*;

    const LANES_PER_ITER: usize = 2;

    /// # Safety
    ///
    /// Caller MUST guarantee NEON is available (gated upstream via
    /// `std::arch::is_aarch64_feature_detected!`).
    #[target_feature(enable = "neon")]
    pub unsafe fn neighbor_match_mask(candidates: &[u64], targets: &[u64]) -> Vec<bool> {
        let len = candidates.len();
        let mut out: Vec<bool> = vec![false; len];
        let out_ptr = out.as_mut_ptr();
        let bulk_end = len - (len % LANES_PER_ITER);

        let mut i = 0;
        while i < bulk_end {
            // SAFETY: unaligned load tolerated by `vld1q_u64`; bulk-loop
            // bounds same as AVX2.
            let cands = unsafe { vld1q_u64(candidates.as_ptr().add(i)) };

            let mut acc = vdupq_n_u64(0);
            for &t in targets {
                let target_v = vdupq_n_u64(t);
                let cmp = vceqq_u64(cands, target_v);
                acc = vorrq_u64(acc, cmp);
            }

            // SAFETY: lanes_to_mask2 is `#[target_feature(enable =
            // "neon")]`; NEON gated upstream.
            let lanes = unsafe { lanes_to_mask2(acc) };
            for (lane, &mask_bit) in lanes.iter().enumerate() {
                // SAFETY: out_ptr valid; index bounded.
                unsafe { out_ptr.add(i + lane).write(mask_bit) };
            }
            i += LANES_PER_ITER;
        }

        // Scalar tail.
        while i < len {
            let c = candidates[i];
            let mut hit = false;
            for &t in targets {
                if c == t {
                    hit = true;
                    break;
                }
            }
            // SAFETY: i < len.
            unsafe { out_ptr.add(i).write(hit) };
            i += 1;
        }
        out
    }

    /// Extract 2 boolean lanes from a 128-bit comparison register.
    ///
    /// # Safety
    ///
    /// Caller MUST guarantee NEON (gated by `#[target_feature]`).
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn lanes_to_mask2(cmp: uint64x2_t) -> [bool; 2] {
        [vgetq_lane_u64(cmp, 0) != 0, vgetq_lane_u64(cmp, 1) != 0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----------------------------------------------------------------
    // Scalar fallback baseline
    // ----------------------------------------------------------------

    #[test]
    fn unit_scalar_membership_returns_per_candidate_truth() {
        let candidates = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let targets = vec![3, 5, 7];
        let mask = scalar::neighbor_match_mask(&candidates, &targets);
        assert_eq!(
            mask,
            vec![false, false, true, false, true, false, true, false]
        );
    }

    #[test]
    fn unit_scalar_empty_targets_returns_all_false() {
        let candidates = vec![1, 2, 3];
        let mask = scalar::neighbor_match_mask(&candidates, &[]);
        assert_eq!(mask, vec![false, false, false]);
    }

    #[test]
    fn unit_dispatch_membership_handles_empty_candidates() {
        let mask = simd_neighbor_match_mask(&[], &[1, 2, 3]);
        assert!(mask.is_empty());
    }

    // ----------------------------------------------------------------
    // Per-arch parity
    // ----------------------------------------------------------------

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn unit_x86_avx2_membership_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("skipped: x86_64 AVX2 unavailable on this host");
            return;
        }
        let candidates: Vec<u64> = (0..200).collect();
        for targets in [
            vec![5_u64],
            vec![5, 17, 99, 199],
            vec![0, 200, 400], // partial overlap (only 0 hits)
            vec![],
        ] {
            let scalar = scalar::neighbor_match_mask(&candidates, &targets);
            // SAFETY: AVX2 gated.
            let simd = unsafe { x86_avx2::neighbor_match_mask(&candidates, &targets) };
            assert_eq!(simd, scalar, "AVX2 vs scalar for targets={:?}", targets);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn unit_aarch64_neon_membership_matches_scalar() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("skipped: AArch64 NEON unavailable on this host");
            return;
        }
        let candidates: Vec<u64> = (0..200).collect();
        for targets in [vec![5_u64], vec![5, 17, 99, 199], vec![0, 200, 400], vec![]] {
            let scalar = scalar::neighbor_match_mask(&candidates, &targets);
            // SAFETY: NEON gated.
            let simd = unsafe { aarch64_neon::neighbor_match_mask(&candidates, &targets) };
            assert_eq!(simd, scalar, "NEON vs scalar for targets={:?}", targets);
        }
    }

    #[test]
    fn unit_dispatch_membership_matches_scalar_baseline() {
        let candidates: Vec<u64> = (0..32).collect();
        let targets: Vec<u64> = vec![0, 5, 10, 15, 20, 25, 30, 100];
        let scalar = scalar::neighbor_match_mask(&candidates, &targets);
        let dispatch = simd_neighbor_match_mask(&candidates, &targets);
        assert_eq!(scalar, dispatch);
    }

    #[test]
    fn unit_simd_path_handles_lengths_below_lanes_per_iter() {
        // Boundary: tiny candidates (< LANES_PER_ITER) should run
        // exclusively through the scalar tail loop.
        for len in 0..=3 {
            let candidates: Vec<u64> = (0..len as u64).collect();
            let targets = vec![0_u64, 1, 2];
            let s = scalar::neighbor_match_mask(&candidates, &targets);
            let d = simd_neighbor_match_mask(&candidates, &targets);
            assert_eq!(s, d, "boundary length {} dispatch != scalar", len);
        }
    }
}
