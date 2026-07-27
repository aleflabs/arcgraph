//! M4-42 (M4-04b) selectivity proptest — bounds-correctness invariant.
//!
//! Property: **for every random catalog stats configuration and every
//! estimator method, the result is a finite f64 in `[0.0, 1.0]`.**
//!
//! Selectivity is a probability — the M4-05 cost-based planner uses
//! it as a filter-passthrough fraction, so the unit-interval bound is
//! load-bearing for cost monotonicity. Inf / NaN / negative values
//! would silently corrupt cost estimates and produce non-monotonic
//! plan choices. The proptest pins the invariant across 256 random
//! `(label_card, total_node, rel_type_card, total_rel, list_size)`
//! tuples.
//!
//! # Why a proptest (not just unit tests)
//!
//! The unit tests in `src/semantic/selectivity.rs` cover known shapes
//! (Some, None, Some(0), small list_size). The proptest covers the
//! tail: very large totals (`u64::MAX` / 2 to avoid overflow on the
//! `as f64` cast), very small fractions, list-size > total, and the
//! `card > total` race-condition shape (label cardinality larger than
//! total — possible during a concurrent commit between the two atomic
//! reads in the production catalog).
//!
//! # ADR provenance
//! - ADR-038 §2 D-27 — M4-42 selectivity estimator contract.
//! - ADR-038 amendment-03 §M4-42 row — the proptest pin.

use arcgraph_core::{LabelId, TypeId};
use arcgraph_query::semantic::{BindingId, SelectivityEstimator, StubCatalogProvider};
use proptest::prelude::*;

/// Build a stub catalog from raw ingredients. `None` for any of the
/// `Option<u64>` arguments simulates the M4-41 "no stats collected"
/// sentinel.
fn build_cat(
    label_card: Option<u64>,
    total_node: Option<u64>,
    rel_type_card: Option<u64>,
    total_rel: Option<u64>,
) -> StubCatalogProvider {
    let mut c = StubCatalogProvider::new();
    if let Some(n) = total_node {
        c = c.with_total_node_count(n);
    }
    if let Some(n) = total_rel {
        c = c.with_total_rel_count(n);
    }
    if let Some(n) = label_card {
        c = c.with_label_cardinality(LabelId::new(1), n);
    }
    if let Some(n) = rel_type_card {
        c = c.with_rel_type_cardinality(TypeId::new(1), n);
    }
    c
}

/// Assert the bounds invariant: every estimator returns a finite f64
/// in `[0.0, 1.0]`, never negative, never > 1.0, never NaN, never Inf.
fn assert_bounds(s: f64, label: &str) {
    assert!(s.is_finite(), "{label}: non-finite ({s})");
    assert!(!s.is_nan(), "{label}: NaN");
    assert!(s >= 0.0, "{label}: negative ({s})");
    assert!(s <= 1.0, "{label}: > 1 ({s})");
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 256,
        ..ProptestConfig::default()
    })]

    /// For any random `(label_card, total_node, rel_type_card,
    /// total_rel, list_size)` tuple, every estimator method returns a
    /// finite f64 in [0, 1]. Pins the cost-monotonicity invariant the
    /// M4-05 cost-based planner will lean on.
    ///
    /// `u64::MAX / 2` upper bound on the cardinalities defends
    /// against the `as f64` cast overflowing — `f64` has 53 bits of
    /// mantissa, so values near u64::MAX round, but values up to
    /// 2^62 are exactly representable as f64 modulo trailing-zero
    /// rounding.
    #[test]
    fn estimators_always_in_unit_interval(
        label_card in proptest::option::of(0u64..u64::MAX / 2),
        total_node in proptest::option::of(0u64..u64::MAX / 2),
        rel_type_card in proptest::option::of(0u64..u64::MAX / 2),
        total_rel in proptest::option::of(0u64..u64::MAX / 2),
        list_size in 0usize..1_000_000,
    ) {
        let cat = build_cat(label_card, total_node, rel_type_card, total_rel);
        let est = SelectivityEstimator::new(&cat);
        let v = BindingId::new(0);
        let l = LabelId::new(1);
        let t = TypeId::new(1);

        assert_bounds(est.estimate_eq(v, Some(l)), "estimate_eq");
        assert_bounds(est.estimate_eq(v, None), "estimate_eq (no label)");
        assert_bounds(est.estimate_lt(v, Some(l)), "estimate_lt");
        assert_bounds(est.estimate_lt(v, None), "estimate_lt (no label)");
        assert_bounds(est.estimate_in(v, Some(l), list_size), "estimate_in");
        assert_bounds(est.estimate_in(v, None, list_size), "estimate_in (no label)");
        assert_bounds(est.estimate_label(l), "estimate_label");
        assert_bounds(est.estimate_rel_type(t), "estimate_rel_type");
    }
}
