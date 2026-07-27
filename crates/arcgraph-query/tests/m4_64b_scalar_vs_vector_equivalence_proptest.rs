//! W13α / M4-64b — scalar-vs-vector equivalence proptest per
//! ADR-038 amendment-02 §M4.f + amendment-03 §Structural-1.
//!
//! # Invariant
//!
//! For every random input `(values, mask, target, op, k, rank_count)`
//! the SIMD-dispatched output MUST agree bytewise with the scalar
//! reference output across all three operator helpers:
//!
//! 1. [`simd_filter_i64_cmp`] — `Vec<bool>` equality.
//! 2. [`simd_neighbor_match_mask`] — `Vec<bool>` equality.
//! 3. [`simd_rrf_scores`] — `Vec<f32>` bitwise equality
//!    (`to_bits()`).
//!
//! # Hardening
//!
//! Per the spawn prompt's empirical-gauntlet step 6, this proptest
//! runs at `PROPTEST_CASES=10000` to expose rare-tail divergences
//! (length boundaries, sign-extension edge cases for AVX2 i64 cmp,
//! IEEE-754 division corner cases for f32 RRF). The default
//! `PROPTEST_CASES` (256) is the CI everyday cadence; the 10K-case
//! gauntlet runs at PR-prep time.
//!
//! # Why bytewise (not approximate-equal) on the f32 path
//!
//! IEEE-754 division is deterministic; identical (k, rank) inputs
//! through `_mm256_div_ps` / `vdivq_f32` produce the SAME bit pattern
//! as scalar `1.0_f32 / x`. Approximate-equal would mask a bug that
//! shifts the lane-width-vs-precision contract.

use arcgraph_query::executor::simd::expand::{scalar as expand_scalar, simd_neighbor_match_mask};
use arcgraph_query::executor::simd::filter::{CmpOp, scalar as filter_scalar, simd_filter_i64_cmp};
use arcgraph_query::executor::simd::rrf::{scalar as rrf_scalar, simd_rrf_scores};
use proptest::prelude::*;

/// Map a u8 to one of the 6 SIMD-supported comparison operators.
fn cmp_op_from_u8(b: u8) -> CmpOp {
    match b % 6 {
        0 => CmpOp::Eq,
        1 => CmpOp::Ne,
        2 => CmpOp::Lt,
        3 => CmpOp::Le,
        4 => CmpOp::Gt,
        _ => CmpOp::Ge,
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        // The default is 256; CI tunes to 10000 via PROPTEST_CASES env.
        cases: 256,
        ..ProptestConfig::default()
    })]

    /// FilterOp i64 SIMD path: scalar and SIMD outputs MUST match.
    #[test]
    fn filter_simd_vs_scalar_equivalence(
        values in prop::collection::vec(any::<i64>(), 0..256),
        nulls in prop::collection::vec(any::<bool>(), 0..256),
        target in any::<i64>(),
        op_byte in any::<u8>(),
    ) {
        // Truncate nulls to the same length as values (paired vectors).
        let len = values.len();
        let null_mask: Vec<bool> = nulls.into_iter().take(len).chain(std::iter::repeat(false)).take(len).collect();
        prop_assert_eq!(null_mask.len(), values.len());
        let op = cmp_op_from_u8(op_byte);
        let scalar = filter_scalar::filter_i64_cmp(&values, &null_mask, target, op);
        let simd = simd_filter_i64_cmp(&values, &null_mask, target, op);
        prop_assert_eq!(scalar, simd, "FilterOp SIMD vs scalar diverged");
    }

    /// ExpandOp NodeId membership: scalar and SIMD outputs MUST match.
    #[test]
    fn expand_simd_vs_scalar_equivalence(
        candidates in prop::collection::vec(any::<u64>(), 0..256),
        targets in prop::collection::vec(any::<u64>(), 0..16),
    ) {
        let scalar = expand_scalar::neighbor_match_mask(&candidates, &targets);
        let simd = simd_neighbor_match_mask(&candidates, &targets);
        prop_assert_eq!(scalar, simd, "ExpandOp SIMD vs scalar diverged");
    }

    /// RrfFusion f32 scores: scalar and SIMD outputs MUST match
    /// bytewise (IEEE-754 division is deterministic).
    #[test]
    fn rrf_simd_vs_scalar_bytewise_equivalence(
        k in 1_u32..=10_000,
        rank_count in 0_usize..512,
    ) {
        let scalar = rrf_scalar::rrf_scores(k, rank_count);
        let simd = simd_rrf_scores(k, rank_count);
        prop_assert_eq!(scalar.len(), simd.len(), "RRF length mismatch");
        for i in 0..scalar.len() {
            prop_assert_eq!(
                scalar[i].to_bits(),
                simd[i].to_bits(),
                "RRF SIMD vs scalar bytewise diverged at i={}", i
            );
        }
    }

    /// Cross-helper joint invariant: a single random bundle
    /// exercises all three SIMD paths in one shrink-friendly case.
    /// The shrinker on a counterexample narrows the responsible
    /// helper at minimum cost.
    #[test]
    fn joint_simd_vs_scalar_equivalence_compound(
        f_values in prop::collection::vec(any::<i64>(), 0..128),
        f_nulls in prop::collection::vec(any::<bool>(), 0..128),
        f_target in any::<i64>(),
        f_op_byte in any::<u8>(),
        e_candidates in prop::collection::vec(any::<u64>(), 0..128),
        e_targets in prop::collection::vec(any::<u64>(), 0..8),
        r_k in 1_u32..=1_000,
        r_rank_count in 0_usize..256,
    ) {
        // FilterOp.
        let f_len = f_values.len();
        let f_nm: Vec<bool> = f_nulls
            .into_iter()
            .take(f_len)
            .chain(std::iter::repeat(false))
            .take(f_len)
            .collect();
        let f_op = cmp_op_from_u8(f_op_byte);
        let f_scalar = filter_scalar::filter_i64_cmp(&f_values, &f_nm, f_target, f_op);
        let f_simd = simd_filter_i64_cmp(&f_values, &f_nm, f_target, f_op);
        prop_assert_eq!(f_scalar, f_simd, "compound: FilterOp diverged");

        // ExpandOp.
        let e_scalar = expand_scalar::neighbor_match_mask(&e_candidates, &e_targets);
        let e_simd = simd_neighbor_match_mask(&e_candidates, &e_targets);
        prop_assert_eq!(e_scalar, e_simd, "compound: ExpandOp diverged");

        // RrfFusion.
        let r_scalar = rrf_scalar::rrf_scores(r_k, r_rank_count);
        let r_simd = simd_rrf_scores(r_k, r_rank_count);
        prop_assert_eq!(r_scalar.len(), r_simd.len());
        for i in 0..r_scalar.len() {
            prop_assert_eq!(
                r_scalar[i].to_bits(),
                r_simd[i].to_bits(),
                "compound: RrfFusion diverged at i={}", i
            );
        }
    }
}
