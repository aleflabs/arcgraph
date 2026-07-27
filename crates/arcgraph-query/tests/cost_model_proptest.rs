//! Proptest for the M4-51 cost model.
//!
//! # Invariants pinned
//!
//! 1. **Cost monotonicity in input cardinality** (
//!    [`cost_monotonic_in_scan_cardinality`]). For a plan rooted at a
//!    single label scan, increasing the catalog's reported
//!    `label_cardinality` MUST produce a cost estimate that is greater
//!    than or equal to the smaller cardinality's cost. The cost model
//!    depends on this for plan ordering: M4-52's join-ordering DP
//!    picks the lowest-cost plan; if cost were non-monotonic in scan
//!    cardinality, the DP could pick provably worse plans.
//!
//! 2. **`cost_join` monotonicity in EACH input cardinality**
//!    ([`cost_join_monotonic_in_each_input`]). Wave 9d M4-52b
//!    Group A retro MED-2 (W9a CR-A-4 escalation): with M4-51
//!    (PR #232) and M4-52 (PR #242) both consuming `cost_join`,
//!    the underlying primitive is on its third use-site (M4-91
//!    EXPLAIN now consumes transitively via `enumerate_join_order`
//!    per W9d M4-52b CRIT-1 closure). The plan-level cost-optimality
//!    oracle in `m4_52_join_enumeration_proptest.rs` is non-vacuous,
//!    but it operates at the DP-chooses-best layer; a `cost_join`
//!    formula regression that flipped a sign in a way preserving
//!    relative DP orderings on the proptest's sampled cases would
//!    slip through. This pin tests `cost_join` itself: holding
//!    either side fixed, increasing the other cardinality must
//!    NEVER decrease total cost.
//!
//! 3. **`cost_left_outer_join` lower bound** (
//!    [`cost_left_outer_join_card_at_least_left`]). Outer-join
//!    semantics constrain output cardinality to ≥ left input
//!    (Cypher 9 §6.5 OPTIONAL MATCH; every left row matches at least
//!    once with NULL fills). Co-packed with MED-2 per W9a CR-A-4.
//!
//! # Why these are the right oracles
//!
//! Cost monotonicity is the **necessary, weaker form** of
//! plan-ordering correctness. The scan-cardinality slice pins the
//! load-bearing catalog-driven case end-to-end; the `cost_join`
//! per-input slice pins the binary-operator case at the boundary
//! M4-52 consumes inside its DP loop.
//!
//! # Proptest cases
//!
//! - `cost_monotonic_in_scan_cardinality` — 256 random `(small,
//!   large)` cardinality pairs with `small ≤ large`. For each pair,
//!   build a Scan-Filter-Project plan, swap the catalog cardinality,
//!   assert `cost(small) ≤ cost(large)`.
//! - `cost_join_monotonic_in_each_input` — sample triples `(l, r,
//!   delta)` with `delta > 0`. Assert
//!   `cost_join(l, r) <= cost_join(l + delta, r)` AND
//!   `cost_join(l, r) <= cost_join(l, r + delta)` (monotonic in BOTH
//!   inputs, holding the other fixed).
//! - `cost_left_outer_join_card_at_least_left` — sample `(l, r)`
//!   pairs. Assert `output_card(cost_left_outer_join(l, r)) >= l` for
//!   both Cartesian and SharedBindings join conditions.
//!
//! # ADR provenance
//! - ADR-038 amendment-02 §M4.e — M4-51 contract; "monotonicity in
//!   input cardinality" is the documented proptest baseline.
//! - PR #172 (M4-42) review N1 + the
//!   `tests/selectivity_proptest.rs` precedent — same loud-on-dev /
//!   safe-on-prod invariant pinning discipline.
//! - W9a Group A retro CR-A-4 (`group-A-final.md` §4) and W9b Group A
//!   retro MED-2 — `cost_join` per-input monotonicity gap.

use proptest::prelude::*;

use arcgraph_core::{LabelId, Lsn};
use arcgraph_query::ast::{BinOp, Literal};
use arcgraph_query::error::Span;
use arcgraph_query::logical_plan::{
    JoinAlgorithm, JoinCondition, LogicalEmpty, LogicalFilter, LogicalJoin, LogicalLeftOuterJoin,
    LogicalPlan, LogicalProject, LogicalScan,
};
use arcgraph_query::planner::cost::estimate_costs;
use arcgraph_query::planner::cost::operator::{cost_join, cost_left_outer_join};
use arcgraph_query::planner::cost::{Cardinality, Cost};
use arcgraph_query::semantic::StubCatalogProvider;
use arcgraph_query::semantic::bound_ast::{BindingId, BoundExpression};

fn span() -> Span {
    Span::point(1, 1)
}

fn build_plan(label: LabelId) -> LogicalPlan {
    let scan = LogicalPlan::Scan(LogicalScan {
        label: Some(label),
        var: BindingId::new(0),
        read_lsn: Lsn::MAX,
        span: span(),
    });
    let predicate = BoundExpression::BinaryOp {
        op: BinOp::Eq,
        lhs: Box::new(BoundExpression::VariableRef {
            name: "v".into(),
            binding_id: BindingId::new(0),
            span: span(),
            type_info: None,
        }),
        rhs: Box::new(BoundExpression::Literal {
            value: Literal::Integer(42),
            span: span(),
            type_info: None,
        }),
        span: span(),
        type_info: None,
    };
    let filter = LogicalPlan::Filter(LogicalFilter {
        input: Box::new(scan),
        predicate,
        span: span(),
    });
    LogicalPlan::Project(LogicalProject {
        input: Box::new(filter),
        items: Vec::new(),
        span: span(),
    })
}

fn cost_for_cardinality(label: LabelId, cardinality: u64, total: u64) -> f64 {
    let cat = StubCatalogProvider::new()
        .with_total_node_count(total)
        .with_label_cardinality(label, cardinality);
    estimate_costs(build_plan(label), &cat).total_cost().total()
}

proptest! {
    /// **Monotonicity in scan cardinality.** Same plan, same catalog
    /// total, varying label_cardinality from `small` to `large`
    /// (`small ≤ large`) — the cost estimate must be monotonically
    /// non-decreasing.
    ///
    /// Bounds chosen so the catalog totals stay valid (`total ≥
    /// large`) and the f64 arithmetic stays in the precise range
    /// (cardinalities ≤ 1M).
    #[test]
    fn cost_monotonic_in_scan_cardinality(
        small in 0_u64..=500_000,
        delta in 0_u64..=500_000,
    ) {
        let large = small.saturating_add(delta).min(1_000_000);
        let total = large.saturating_add(1).max(1);
        let label = LabelId::new(1);

        let cost_small = cost_for_cardinality(label, small, total);
        let cost_large = cost_for_cardinality(label, large, total);

        prop_assert!(
            cost_small.is_finite() && cost_small >= 0.0,
            "cost(small={small}) must be finite and non-negative; got {cost_small}"
        );
        prop_assert!(
            cost_large.is_finite() && cost_large >= 0.0,
            "cost(large={large}) must be finite and non-negative; got {cost_large}"
        );
        prop_assert!(
            cost_small <= cost_large,
            "monotonicity violated: small={small} cost={cost_small} > large={large} cost={cost_large}"
        );
    }
}

// ---------------------------------------------------------------------
// W9d M4-52b MED-2 (W9a CR-A-4 escalation): `cost_join` per-input
// monotonicity + `cost_left_outer_join` left-bound proptest.
// ---------------------------------------------------------------------

fn empty_plan() -> Box<LogicalPlan> {
    Box::new(LogicalPlan::Empty(LogicalEmpty { span: span() }))
}

fn join_node(on: Vec<BindingId>) -> LogicalJoin {
    LogicalJoin {
        left: empty_plan(),
        right: empty_plan(),
        on: JoinCondition::SharedBindings(on),
        algorithm: JoinAlgorithm::Auto,
        span: span(),
    }
}

fn left_outer_join_node(on: Vec<BindingId>) -> LogicalLeftOuterJoin {
    LogicalLeftOuterJoin {
        left: empty_plan(),
        right: empty_plan(),
        on: JoinCondition::SharedBindings(on),
        span: span(),
    }
}

/// Returns `(local_cost, output_card)` for an equi-join on the named
/// shared binding given left/right input cardinalities.
fn equi_join_cost(l: f64, r: f64) -> (Cost, Cardinality) {
    cost_join(
        &join_node(vec![BindingId::new(0)]),
        Cardinality::new(l),
        Cardinality::new(r),
    )
}

proptest! {
    /// **`cost_join` monotonicity in EACH input** (W9d MED-2 / W9a CR-A-4).
    ///
    /// Holding right fixed, increasing left MUST NOT decrease total
    /// (local + output) cost-relevant value. Symmetric: holding left
    /// fixed, increasing right MUST NOT decrease.
    ///
    /// We assert against `local_cost.total()` (the operator's local
    /// cost contribution) since `cost_join`'s formula
    /// `(l + r) * HASH_JOIN_COST_PER_ROW` is symmetric and strictly
    /// monotonic per-input. The output cardinality formula
    /// `(l*r)/max(l,r) = min(l,r)` is monotonic in EACH input
    /// (raising `l` from 5 to 10 with `r=20` raises output from 5 to
    /// 10; lowering or holding `l` cannot decrease output beyond
    /// pre-existing `r` cap). Both checks pinned below.
    ///
    /// Phase 4.2 controlled-mutation (verified at slice-build time):
    /// inverting the formula
    /// `(l+r) * HASH_JOIN_COST_PER_ROW` → `(l-r).abs() * HASH_JOIN_COST_PER_ROW`
    /// causes this proptest to FAIL at minimal input
    /// (`l=0, r=0, delta=1` → small=0, larger=delta * 2 vs delta * 0).
    /// See commit body Phase 4.2 section.
    #[test]
    fn cost_join_monotonic_in_each_input(
        l in 0_u64..=500_000,
        r in 0_u64..=500_000,
        delta in 1_u64..=10_000,
    ) {
        let l = l as f64;
        let r = r as f64;
        let delta = delta as f64;

        // Holding right fixed; increasing left MUST NOT decrease cost.
        let (small_cost, small_card) = equi_join_cost(l, r);
        let (larger_l_cost, larger_l_card) = equi_join_cost(l + delta, r);
        prop_assert!(
            larger_l_cost.total() >= small_cost.total() - 1e-9,
            "cost_join not monotonic in LEFT input: \
             l={l} r={r} delta={delta} small_cost={} larger_l_cost={}",
            small_cost.total(),
            larger_l_cost.total(),
        );
        prop_assert!(
            larger_l_card.rows() >= small_card.rows() - 1e-9,
            "cost_join output_card not monotonic in LEFT input: \
             l={l} r={r} delta={delta} small_card={} larger_l_card={}",
            small_card.rows(),
            larger_l_card.rows(),
        );

        // Symmetric: holding left fixed; increasing right MUST NOT
        // decrease cost.
        let (small_cost_r, small_card_r) = equi_join_cost(l, r);
        let (larger_r_cost, larger_r_card) = equi_join_cost(l, r + delta);
        prop_assert!(
            larger_r_cost.total() >= small_cost_r.total() - 1e-9,
            "cost_join not monotonic in RIGHT input: \
             l={l} r={r} delta={delta} small_cost={} larger_r_cost={}",
            small_cost_r.total(),
            larger_r_cost.total(),
        );
        prop_assert!(
            larger_r_card.rows() >= small_card_r.rows() - 1e-9,
            "cost_join output_card not monotonic in RIGHT input: \
             l={l} r={r} delta={delta} small_card={} larger_r_card={}",
            small_card_r.rows(),
            larger_r_card.rows(),
        );
    }

    /// **`cost_left_outer_join` output ≥ left** (W9a CR-A-4 spec
    /// co-pack). Cypher 9 §6.5 OPTIONAL MATCH semantics: every left
    /// row matches at least once (with NULL fills if no right row
    /// matches). Pins the `.max(l)` clamp at
    /// `cost::operator::cost_left_outer_join`.
    ///
    /// Sampled across both join-condition shapes (Cartesian + shared-
    /// bindings) to pin both branches of the inner-estimate match.
    #[test]
    fn cost_left_outer_join_card_at_least_left(
        l in 0_u64..=100_000,
        r in 0_u64..=100_000,
        cartesian in proptest::bool::ANY,
    ) {
        let l = l as f64;
        let r = r as f64;
        let on = if cartesian { Vec::new() } else { vec![BindingId::new(0)] };
        let (_, card) = cost_left_outer_join(
            &left_outer_join_node(on),
            Cardinality::new(l),
            Cardinality::new(r),
        );
        prop_assert!(
            card.rows() >= l - 1e-9,
            "cost_left_outer_join output_card must be ≥ left input \
             (Cypher 9 §6.5 NULL-fill invariant): \
             l={l} r={r} cartesian={cartesian} card={}",
            card.rows(),
        );
    }
}
