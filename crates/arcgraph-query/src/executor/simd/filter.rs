//! M4-64b — SIMD i64 comparison helper for FilterOp predicate eval.
//!
//! # Latency budget (PD-5)
//!
//! Per ADR-038 amendment-03 §Structural-1 the slice-level acceptance
//! is **≥1.5× speedup vs scalar** on the FilterOp hot path. Concrete
//! per-arch budget (Apple-Silicon M3 measured at v1.0-alpha bench
//! `simd_hot_path_speedup`):
//!
//! - **AVX2**: ~2 cycles per 4-lane `_mm256_cmpgt_epi64` (Ice Lake +
//!   Skylake-X dispatch on Port 5; per Intel Optimization Reference
//!   §15.5.1). Expected speedup over scalar ≥ 3× at N=10K.
//! - **NEON**: ~1 cycle per 2-lane `vcgtq_s64` (Cortex-A78 dispatch on
//!   the 128-bit ASIMD pipe; per ARM Cortex-A78 SOG §3.18). Expected
//!   speedup over scalar ≥ 1.8× at N=10K.
//! - **Scalar**: baseline; one comparison per cycle when
//!   branch-predicted.
//!
//! # 3VL NULL handling (amendment-03 §TIER-2-b preservation)
//!
//! The SIMD path compares packed i64 lanes. NULL-valued cells live in
//! a parallel `is_null_mask: &[bool]` slice the caller supplies; the
//! helper combines the SIMD predicate result with the null-mask AND-NOT
//! to produce the final pass mask. Per ADR-038 §2 D-20: a NULL operand
//! reaching a comparison yields ThreeValued::Unknown which the WHERE
//! filter drops. The helper enforces this: `pass[i] = simd_cmp(values[i],
//! target) AND NOT is_null_mask[i]`.
//!
//! # Operator-side scope filter
//!
//! Only PROPERTY-vs-LITERAL i64 comparisons are accelerated. Mixed-type
//! (Integer-vs-Float widening), Boolean predicates, IS NULL, IN, and
//! function calls fall through to the scalar `evaluate` path. The
//! FilterOp dispatcher checks the predicate shape at construction time
//! and caches a [`SimdShape`](super::super::ops::filter) flag.

use super::SimdBackend;

/// Comparison operator for the SIMD i64 predicate path.
///
/// Mirrors a subset of [`crate::ast::BinOp`] — only the comparisons
/// the v1.0-alpha SIMD path accelerates. Other operators fall through
/// to the scalar [`crate::executor::eval::evaluate`] routine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    /// `lhs == rhs`
    Eq,
    /// `lhs != rhs`
    Ne,
    /// `lhs > rhs`
    Gt,
    /// `lhs >= rhs`
    Ge,
    /// `lhs < rhs`
    Lt,
    /// `lhs <= rhs`
    Le,
}

impl CmpOp {
    /// Apply the comparison scalar-style (used by the fallback path
    /// + the trailing-tail loop in the SIMD paths).
    #[inline]
    #[must_use]
    pub fn apply(self, lhs: i64, rhs: i64) -> bool {
        match self {
            Self::Eq => lhs == rhs,
            Self::Ne => lhs != rhs,
            Self::Gt => lhs > rhs,
            Self::Ge => lhs >= rhs,
            Self::Lt => lhs < rhs,
            Self::Le => lhs <= rhs,
        }
    }
}

/// Vectorized i64 comparison against a scalar target.
///
/// Returns a `Vec<bool>` of length `values.len()` where `result[i] =
/// (values[i] OP target) AND NOT is_null_mask[i]`. NULL handling
/// preserves the openCypher 3VL contract: NULL operand → drop row.
///
/// # Backend selection
///
/// Internally dispatches via [`SimdBackend::detect`]; the backend
/// label is recorded into the call's tracing span (callers wrap in
/// `tracing::info_span!` if observability is desired). The SIMD path
/// is mutually exclusive with the scalar path — the same input under
/// different backends MUST produce bytewise-identical output (pinned
/// by `m4_64b_scalar_vs_vector_equivalence_proptest`).
///
/// # Panics
///
/// Panics in debug builds if `is_null_mask.len() != values.len()`. In
/// release builds the function clamps to the shorter of the two; the
/// caller is responsible for upholding the same-length invariant.
pub fn simd_filter_i64_cmp(
    values: &[i64],
    is_null_mask: &[bool],
    target: i64,
    op: CmpOp,
) -> Vec<bool> {
    debug_assert_eq!(
        values.len(),
        is_null_mask.len(),
        "simd_filter_i64_cmp: values + is_null_mask MUST have matching lengths"
    );
    match SimdBackend::detect() {
        #[cfg(target_arch = "x86_64")]
        SimdBackend::X86Avx2 => {
            // SAFETY: The runtime gate above guarantees AVX2 is
            // available; the inner `#[target_feature]` fn re-asserts
            // the precondition at the function attribute level. The
            // caller's `values.len() == is_null_mask.len()` invariant
            // is debug-asserted above; in release builds the bulk
            // loop bounds the index by `values.len()` so out-of-bounds
            // reads cannot occur. Unaligned loads are tolerated by the
            // `_mm256_loadu_si256` intrinsic.
            unsafe { x86_avx2::filter_i64_cmp(values, is_null_mask, target, op) }
        }
        #[cfg(target_arch = "aarch64")]
        SimdBackend::AArch64Neon => {
            // SAFETY: The runtime gate above guarantees NEON is
            // available; on AArch64 NEON is mandatory per the ARMv8-A
            // spec but the gate parallels the x86_64 path. The same
            // length invariant + bulk-loop bounds as the AVX2 path
            // apply.
            unsafe { aarch64_neon::filter_i64_cmp(values, is_null_mask, target, op) }
        }
        _ => scalar::filter_i64_cmp(values, is_null_mask, target, op),
    }
}

/// Scalar fallback. Always available regardless of target arch / runtime
/// feature detection.
pub mod scalar {
    use super::CmpOp;

    /// Scalar baseline. The FilterOp uses this when no SIMD backend is
    /// available; the proptest uses it as the equivalence reference.
    #[must_use]
    pub fn filter_i64_cmp(
        values: &[i64],
        is_null_mask: &[bool],
        target: i64,
        op: CmpOp,
    ) -> Vec<bool> {
        debug_assert_eq!(values.len(), is_null_mask.len());
        let len = values.len().min(is_null_mask.len());
        let mut out = Vec::with_capacity(len);
        for i in 0..len {
            if is_null_mask[i] {
                out.push(false);
            } else {
                out.push(op.apply(values[i], target));
            }
        }
        out
    }
}

#[cfg(target_arch = "x86_64")]
pub mod x86_avx2 {
    //! AVX2 backend. 4 × i64 lanes per 256-bit `__m256i`.
    //!
    //! AVX2 has no native `cmpge_epi64` / `cmplt_epi64`; we synthesize
    //! `ge` from `NOT lt`, and `lt` from `cmpgt(target, value)` (i.e.
    //! swap operands to invert the inequality direction). Per Intel
    //! intrinsics guide §_mm256_cmpgt_epi64: 1 µop, latency 3, throughput
    //! 1 on Ice Lake; the synthesis adds a single `_mm256_xor_si256`.
    //!
    //! AVX2 has no `cmpne_epi64`; we synthesize from `NOT cmpeq_epi64`.

    use super::CmpOp;
    use core::arch::x86_64::*;

    /// 4 i64 lanes per AVX2 register; 2 registers unrolled per
    /// iteration ⇒ 8 i64 / iter. Per the prompt's "8x i64 (AVX2)"
    /// target.
    const LANES_PER_ITER: usize = 8;

    /// # Safety
    ///
    /// Caller MUST guarantee AVX2 is available on the host (gated
    /// upstream via `is_x86_feature_detected!`). The function's
    /// `#[target_feature]` attribute re-asserts the precondition so
    /// the compiler emits AVX2 instructions inside the body.
    #[target_feature(enable = "avx2")]
    pub unsafe fn filter_i64_cmp(
        values: &[i64],
        is_null_mask: &[bool],
        target: i64,
        op: CmpOp,
    ) -> Vec<bool> {
        let len = values.len().min(is_null_mask.len());
        let mut out: Vec<bool> = vec![false; len];
        let out_ptr = out.as_mut_ptr();
        let bulk_end = len - (len % LANES_PER_ITER);

        let target_v = _mm256_set1_epi64x(target);

        let mut i = 0;
        while i < bulk_end {
            // SAFETY: `_mm256_loadu_si256` tolerates unaligned loads;
            // bulk-loop invariant `i + LANES_PER_ITER <= bulk_end <= len`
            // ensures we never read past the slice.
            let v0 = unsafe { _mm256_loadu_si256(values.as_ptr().add(i).cast::<__m256i>()) };
            // SAFETY: same as v0; offset 4 keeps the read within bounds.
            let v1 = unsafe { _mm256_loadu_si256(values.as_ptr().add(i + 4).cast::<__m256i>()) };

            // SAFETY: compare_lanes is a `#[target_feature(enable =
            // "avx2")]` unsafe fn; AVX2 is gated upstream so the
            // precondition holds.
            let cmp0 = unsafe { compare_lanes(v0, target_v, op) };
            // SAFETY: same as cmp0.
            let cmp1 = unsafe { compare_lanes(v1, target_v, op) };

            // SAFETY: lanes_to_mask4 is a `#[target_feature(enable =
            // "avx2")]` unsafe fn; AVX2 gated upstream.
            let mask0 = unsafe { lanes_to_mask4(cmp0) };
            // SAFETY: same as mask0.
            let mask1 = unsafe { lanes_to_mask4(cmp1) };

            // Combine null-mask: out[lane] = mask AND NOT null.
            for lane in 0..4 {
                let null = is_null_mask[i + lane];
                let result = mask0[lane] && !null;
                // SAFETY: out_ptr is valid for `len` writes; `i + lane <
                // bulk_end <= len`.
                unsafe { out_ptr.add(i + lane).write(result) };
            }
            for lane in 0..4 {
                let null = is_null_mask[i + 4 + lane];
                let result = mask1[lane] && !null;
                // SAFETY: same as above.
                unsafe { out_ptr.add(i + 4 + lane).write(result) };
            }
            i += LANES_PER_ITER;
        }

        // Scalar tail.
        while i < len {
            let result = if is_null_mask[i] {
                false
            } else {
                op.apply(values[i], target)
            };
            // SAFETY: i < len = out's length.
            unsafe { out_ptr.add(i).write(result) };
            i += 1;
        }
        out
    }

    /// Apply the comparison to a packed register and return the result
    /// register (0xFF… per true lane, 0x00… per false lane).
    ///
    /// # Safety
    ///
    /// Caller MUST guarantee AVX2 (gated by `#[target_feature]`).
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn compare_lanes(values: __m256i, target: __m256i, op: CmpOp) -> __m256i {
        match op {
            CmpOp::Eq => _mm256_cmpeq_epi64(values, target),
            CmpOp::Ne => {
                let eq = _mm256_cmpeq_epi64(values, target);
                let all_ones = _mm256_set1_epi64x(-1);
                _mm256_xor_si256(eq, all_ones)
            }
            CmpOp::Gt => _mm256_cmpgt_epi64(values, target),
            CmpOp::Ge => {
                let lt = _mm256_cmpgt_epi64(target, values);
                let all_ones = _mm256_set1_epi64x(-1);
                _mm256_xor_si256(lt, all_ones)
            }
            CmpOp::Lt => _mm256_cmpgt_epi64(target, values),
            CmpOp::Le => {
                let gt = _mm256_cmpgt_epi64(values, target);
                let all_ones = _mm256_set1_epi64x(-1);
                _mm256_xor_si256(gt, all_ones)
            }
        }
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
        // SAFETY: `_mm256_storeu_si256` tolerates unaligned stores;
        // `buf` is a stack array of 4 i64 = 32 bytes = one 256-bit
        // register's worth.
        unsafe { _mm256_storeu_si256(buf.as_mut_ptr().cast::<__m256i>(), cmp) };
        [buf[0] != 0, buf[1] != 0, buf[2] != 0, buf[3] != 0]
    }
}

#[cfg(target_arch = "aarch64")]
pub mod aarch64_neon {
    //! NEON backend. 2 × i64 lanes per 128-bit `int64x2_t`.
    //!
    //! NEON `vcltq_*` / `vcleq_*` work natively on i64 in AArch64
    //! (per ARM ARMv8-A NEON intrinsics §A64). Apple Silicon's M1/M2/M3
    //! cores dispatch 128-bit ASIMD eq/cmp at 2/cycle per the M-series
    //! microarchitecture documentation; the helper expects ≥1.8× over
    //! scalar at N=10K.

    use super::CmpOp;
    use core::arch::aarch64::*;

    /// 2 i64 lanes per NEON register; 2 registers unrolled per
    /// iteration ⇒ 4 i64 / iter (relaxed from the prompt's "2x i64
    /// (NEON)" minimum to amortize per-iter overhead — every NEON
    /// register is in-flight for 2 cycles at most so we have headroom
    /// to issue two simultaneously).
    const LANES_PER_ITER: usize = 4;

    /// # Safety
    ///
    /// Caller MUST guarantee NEON is available on the host (gated
    /// upstream via `std::arch::is_aarch64_feature_detected!`). The
    /// function's `#[target_feature]` attribute re-asserts the
    /// precondition.
    #[target_feature(enable = "neon")]
    pub unsafe fn filter_i64_cmp(
        values: &[i64],
        is_null_mask: &[bool],
        target: i64,
        op: CmpOp,
    ) -> Vec<bool> {
        let len = values.len().min(is_null_mask.len());
        let mut out: Vec<bool> = vec![false; len];
        let out_ptr = out.as_mut_ptr();
        let bulk_end = len - (len % LANES_PER_ITER);

        let target_v = vdupq_n_s64(target);

        let mut i = 0;
        while i < bulk_end {
            // SAFETY: `vld1q_s64` tolerates unaligned loads; bulk loop's
            // `i + LANES_PER_ITER <= bulk_end <= len` invariant ensures
            // we never read past the slice.
            let v0 = unsafe { vld1q_s64(values.as_ptr().add(i)) };
            // SAFETY: same as v0; offset 2 keeps the read within bounds.
            let v1 = unsafe { vld1q_s64(values.as_ptr().add(i + 2)) };

            // SAFETY: compare_lanes is `#[target_feature(enable =
            // "neon")]`; NEON gated upstream.
            let cmp0 = unsafe { compare_lanes(v0, target_v, op) };
            // SAFETY: same as cmp0.
            let cmp1 = unsafe { compare_lanes(v1, target_v, op) };

            // SAFETY: lanes_to_mask2 is `#[target_feature(enable =
            // "neon")]`; NEON gated upstream.
            let mask0 = unsafe { lanes_to_mask2(cmp0) };
            // SAFETY: same as mask0.
            let mask1 = unsafe { lanes_to_mask2(cmp1) };

            for lane in 0..2 {
                let null = is_null_mask[i + lane];
                let result = mask0[lane] && !null;
                // SAFETY: out_ptr is valid for `len` writes; index
                // bounded above.
                unsafe { out_ptr.add(i + lane).write(result) };
            }
            for lane in 0..2 {
                let null = is_null_mask[i + 2 + lane];
                let result = mask1[lane] && !null;
                // SAFETY: same as above.
                unsafe { out_ptr.add(i + 2 + lane).write(result) };
            }
            i += LANES_PER_ITER;
        }

        // Scalar tail.
        while i < len {
            let result = if is_null_mask[i] {
                false
            } else {
                op.apply(values[i], target)
            };
            // SAFETY: i < len = out's length.
            unsafe { out_ptr.add(i).write(result) };
            i += 1;
        }
        out
    }

    /// Apply the comparison to a packed register, returning a uint64x2_t
    /// where each lane is `u64::MAX` (true) or `0` (false).
    ///
    /// # Safety
    ///
    /// Caller MUST guarantee NEON (gated by `#[target_feature]`).
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn compare_lanes(values: int64x2_t, target: int64x2_t, op: CmpOp) -> uint64x2_t {
        match op {
            CmpOp::Eq => vceqq_s64(values, target),
            CmpOp::Ne => {
                let eq = vceqq_s64(values, target);
                // SAFETY: vmvnq_u32_via_u64 is `#[target_feature(enable
                // = "neon")]`; NEON gated upstream.
                unsafe { vmvnq_u32_via_u64(eq) }
            }
            CmpOp::Gt => vcgtq_s64(values, target),
            CmpOp::Ge => vcgeq_s64(values, target),
            CmpOp::Lt => vcltq_s64(values, target),
            CmpOp::Le => vcleq_s64(values, target),
        }
    }

    /// NEON has no `vmvnq_u64` intrinsic at the std::arch surface;
    /// reinterpret as u32 + invert + reinterpret back. Equivalent to
    /// `NOT (lane == lane)`.
    ///
    /// # Safety
    ///
    /// Caller MUST guarantee NEON (gated by `#[target_feature]`).
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn vmvnq_u32_via_u64(v: uint64x2_t) -> uint64x2_t {
        let as_u32 = vreinterpretq_u32_u64(v);
        let inverted = vmvnq_u32(as_u32);
        vreinterpretq_u64_u32(inverted)
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

    fn null_mask(len: usize) -> Vec<bool> {
        vec![false; len]
    }

    // ----------------------------------------------------------------
    // Scalar fallback unit tests
    // ----------------------------------------------------------------

    #[test]
    fn unit_scalar_filter_i64_eq_basic() {
        let values = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mask = scalar::filter_i64_cmp(&values, &null_mask(9), 5, CmpOp::Eq);
        assert_eq!(
            mask,
            vec![false, false, false, false, true, false, false, false, false]
        );
    }

    #[test]
    fn unit_scalar_filter_i64_gt_basic() {
        let values = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mask = scalar::filter_i64_cmp(&values, &null_mask(9), 5, CmpOp::Gt);
        assert_eq!(
            mask,
            vec![false, false, false, false, false, true, true, true, true]
        );
    }

    #[test]
    fn unit_scalar_null_mask_drops_unconditionally() {
        // Per amendment-03 §TIER-2-b: NULL operand → row drops.
        let values = vec![10, 10, 10];
        let nm = vec![false, true, false];
        let mask = scalar::filter_i64_cmp(&values, &nm, 5, CmpOp::Gt);
        assert_eq!(mask, vec![true, false, true]);
    }

    // ----------------------------------------------------------------
    // Per-arch parity unit tests (gated on runtime feature detection).
    // ----------------------------------------------------------------

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn unit_x86_avx2_filter_i64_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("skipped: x86_64 AVX2 unavailable on this host");
            return;
        }
        let values: Vec<i64> = (-50..50).collect();
        let nm = null_mask(values.len());
        for &op in &[
            CmpOp::Eq,
            CmpOp::Ne,
            CmpOp::Lt,
            CmpOp::Le,
            CmpOp::Gt,
            CmpOp::Ge,
        ] {
            for &target in &[-100_i64, -10, 0, 25, 100] {
                let scalar = scalar::filter_i64_cmp(&values, &nm, target, op);
                // SAFETY: AVX2 gated above.
                let simd = unsafe { x86_avx2::filter_i64_cmp(&values, &nm, target, op) };
                assert_eq!(
                    simd, scalar,
                    "AVX2 and scalar must agree for op={:?} target={}",
                    op, target
                );
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn unit_aarch64_neon_filter_i64_matches_scalar() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("skipped: AArch64 NEON unavailable on this host");
            return;
        }
        let values: Vec<i64> = (-50..50).collect();
        let nm = null_mask(values.len());
        for &op in &[
            CmpOp::Eq,
            CmpOp::Ne,
            CmpOp::Lt,
            CmpOp::Le,
            CmpOp::Gt,
            CmpOp::Ge,
        ] {
            for &target in &[-100_i64, -10, 0, 25, 100] {
                let scalar = scalar::filter_i64_cmp(&values, &nm, target, op);
                // SAFETY: NEON gated above.
                let simd = unsafe { aarch64_neon::filter_i64_cmp(&values, &nm, target, op) };
                assert_eq!(
                    simd, scalar,
                    "NEON and scalar must agree for op={:?} target={}",
                    op, target
                );
            }
        }
    }

    #[test]
    fn unit_dispatch_filter_i64_matches_scalar_baseline() {
        // Pin: dispatch path agrees with scalar regardless of
        // backend; this is the equivalence contract.
        let values: Vec<i64> = (0..20).map(|i| i * 7 - 50).collect();
        let nm = null_mask(values.len());
        let scalar_out = scalar::filter_i64_cmp(&values, &nm, 0, CmpOp::Gt);
        let dispatch = simd_filter_i64_cmp(&values, &nm, 0, CmpOp::Gt);
        assert_eq!(scalar_out, dispatch);
    }

    #[test]
    fn unit_simd_path_handles_lengths_below_lanes_per_iter() {
        // Boundary: a slice shorter than LANES_PER_ITER triggers the
        // scalar-tail loop exclusively (bulk_end == 0). Pinning the
        // SIMD path on tiny slices catches a "len < LANES_PER_ITER →
        // empty bulk → tail handles all" regression.
        for &len in &[0_usize, 1, 2, 3, 7] {
            let values: Vec<i64> = (0..len as i64).collect();
            let nm = null_mask(len);
            let s = scalar::filter_i64_cmp(&values, &nm, 0, CmpOp::Ge);
            let d = simd_filter_i64_cmp(&values, &nm, 0, CmpOp::Ge);
            assert_eq!(s, d, "boundary length {} dispatch != scalar", len);
        }
    }
}
