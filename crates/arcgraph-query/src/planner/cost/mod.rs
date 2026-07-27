//! Cost model + per-operator cost functions for the M4-51 planner.
//!
//! # Slice scope
//!
//! M4-51 (M4-05a) ships:
//! - [`Cost`] / [`Cardinality`] / [`CostNode`] / [`CostedPlan`] — the
//!   cost-annotation data types.
//! - [`composition`] — single-helper-module compositional selectivity
//!   rules (AND / OR / NOT / n-ary folds).
//! - [`operator`] — per-operator cost functions (one per
//!   [`crate::logical_plan::LogicalPlan`] variant).
//! - [`predicate`] — predicate-walker that composes
//!   [`crate::semantic::SelectivityEstimator`] outputs through
//!   [`composition`] to produce a filter's combined selectivity.
//! - [`estimate_costs`] — the public entry point that walks a
//!   [`crate::logical_plan::LogicalPlan`] + [`crate::semantic::CatalogProvider`]
//!   and returns a [`CostedPlan`].
//!
//! # Budget (Prime Directive 5)
//!
//! Per ADR-036 §D-25 (multi-step pipeline budget; "Plan parse + cost"
//! row), the M4-05 plan-build budget is **5 ms** end-to-end. The
//! per-operator cost functions here are O(1) plus O(predicate-tree-size)
//! for filters; an entire plan walk is O(plan-nodes × predicates).
//! At v1.0 plan sizes (≤ 50 nodes per query, ≤ 10 predicates per
//! filter) the walk completes in microseconds — well inside budget.
//! [`crate::semantic::CatalogProvider::snapshot`] is called ONCE per
//! walk, not per-predicate.
//!
//! # Cost units
//!
//! Costs are unitless `f64` values relative to a notional "scan one
//! tuple" baseline. The intent is **plan-relative ordering**: two
//! candidate plans for the same query can be compared by total cost
//! to pick the cheaper one. Absolute cost values DO NOT correspond
//! to milliseconds, microseconds, or any other physical unit — only
//! the v1.1 sketch-aware refinement (M4-04c) plus the M4-71 row-count
//! observer feedback loop will tighten the constants enough to be
//! milliseconds-meaningful.
//!
//! Per-operator cost-function rustdoc cites the back-of-envelope
//! breakdown for each constant; see [`operator`].
//!
//! # Output cardinality estimation
//!
//! Every cost-node carries a [`Cardinality`] (estimated row count
//! flowing OUT of the operator). Downstream operators (e.g.,
//! [`crate::logical_plan::LogicalFilter`] over a
//! [`crate::logical_plan::LogicalScan`]) consume the upstream
//! cardinality as their input row-count for their own cost
//! computation — this is the chain that makes cost monotonic in
//! input size (the proptest-pinned invariant in
//! `tests/cost_model_proptest.rs`).
//!
//! # Stats=None graceful degradation
//!
//! When the catalog snapshot is empty (fresh tenant, cold restart),
//! per-predicate selectivity falls back to
//! [`crate::semantic::DEFAULT_LABEL_SELECTIVITY`] / friends per
//! [`crate::semantic::SelectivityEstimator`]. The cost-model walker
//! does NOT short-circuit on empty stats; it produces a finite,
//! non-NaN cost even for a fresh tenant — the cost may be inaccurate
//! relative to the eventual M4-71-feedback-loop-converged numbers,
//! but the planner's plan ordering remains coherent.
//!
//! # ADR provenance
//! - ADR-038 amendment-02 §M4.e — M4-51 cost model + per-operator
//!   cost-functions slice scope.
//! - ADR-036 §D-25 — 5 ms plan-build budget (M4-05 plan parse +
//!   cost row).
//! - ADR-038 §2 D-27 — selectivity estimators (M4-42 input).
//! - ADR-038 §2 D-25 — catalog stats schema (M4-41 input).
//! - PR #172 (M4-42) review Finding 4 — single-helper-module
//!   selectivity composition.

pub(crate) const COST_HINT_HIGH: &str = "COST_HINT 'high'";

pub mod composition;
pub mod operator;
pub mod predicate;
mod walker;

use crate::logical_plan::LogicalPlan;
use crate::semantic::CatalogProvider;

pub use walker::estimate_costs;
pub(crate) use walker::estimate_costs_with_frozen;
pub(crate) use walker::walker_card_for;

/// A unitless cost estimate (`f64`).
///
/// Costs are plan-relative: two candidate plans for the same query
/// can be compared by total cost, but absolute values do NOT
/// correspond to milliseconds. See module docs §"Cost units".
///
/// # Invariants
///
/// - `total >= 0.0`. The cost-model walker never produces negative
///   costs; the constructor saturates to zero defensively.
/// - `total` is finite. NaN / Inf inputs are rejected (saturate
///   to zero) — defense-in-depth against a future formula refinement.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Cost {
    total: f64,
}

impl Cost {
    /// Construct a cost from a finite, non-negative `f64`. NaN, Inf,
    /// and negative values saturate to `0.0` per the defense-in-depth
    /// discipline (see module docs §"Stats=None graceful degradation").
    #[inline]
    #[must_use]
    pub fn new(total: f64) -> Self {
        let total = if total.is_finite() && total >= 0.0 {
            total
        } else {
            0.0
        };
        Self { total }
    }

    /// The zero cost. Useful for accumulator-style folds.
    #[inline]
    #[must_use]
    pub fn zero() -> Self {
        Self { total: 0.0 }
    }

    /// Read the underlying total.
    #[inline]
    #[must_use]
    pub fn total(self) -> f64 {
        self.total
    }

    /// Add another cost into this one, saturating to a finite,
    /// non-negative `f64` on overflow / NaN. Provided as a method
    /// for chaining ergonomics; equivalent to the [`std::ops::Add`]
    /// impl below. Named `plus` (not `add`) to avoid the
    /// `clippy::should_implement_trait` collision with
    /// [`std::ops::Add::add`].
    #[inline]
    #[must_use]
    pub fn plus(self, other: Cost) -> Cost {
        Cost::new(self.total + other.total)
    }
}

impl std::ops::Add for Cost {
    type Output = Cost;
    #[inline]
    fn add(self, rhs: Cost) -> Cost {
        Cost::plus(self, rhs)
    }
}

/// An estimated row cardinality flowing OUT of an operator.
///
/// Stored as a `f64` so non-integer estimates (e.g., a filter applied
/// to a 1000-row scan with selectivity 0.001 → 1.0 row) are
/// representable without rounding noise. The cost-model walker
/// rounds-up to integer row counts only at execution time.
///
/// # Invariants
///
/// - `rows >= 0.0`. Negative values saturate to zero.
/// - `rows` is finite.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Cardinality {
    rows: f64,
}

impl Cardinality {
    /// Construct a cardinality. NaN / Inf / negative saturate to
    /// zero; positive Inf saturates to [`f64::MAX`] for safety.
    #[inline]
    #[must_use]
    pub fn new(rows: f64) -> Self {
        let rows = if rows.is_nan() || rows < 0.0 {
            0.0
        } else if rows == f64::INFINITY {
            f64::MAX
        } else {
            rows
        };
        Self { rows }
    }

    /// The zero cardinality (no rows).
    #[inline]
    #[must_use]
    pub fn zero() -> Self {
        Self { rows: 0.0 }
    }

    /// Read the underlying row count.
    #[inline]
    #[must_use]
    pub fn rows(self) -> f64 {
        self.rows
    }
}

/// Cost annotation for a single [`LogicalPlan`] node.
///
/// Carries:
/// - **`local_cost`** — the per-operator cost contribution (excludes
///   the cost of children).
/// - **`subtree_cost`** — `local_cost` plus the sum of children's
///   subtree costs. The root `CostedNode::subtree_cost` is the
///   total plan cost for plan-comparison purposes (M4-52
///   consumer).
/// - **`output_card`** — estimated row cardinality flowing OUT of
///   this operator (consumed by parent nodes' input-cardinality
///   slot).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CostNode {
    /// Per-operator cost (excludes children).
    pub local_cost: Cost,
    /// Subtree-cumulative cost (`local_cost + Σ children_subtree_cost`).
    pub subtree_cost: Cost,
    /// Estimated output cardinality.
    pub output_card: Cardinality,
}

impl CostNode {
    /// Construct a leaf cost node (no children to accumulate). The
    /// `subtree_cost` equals `local_cost`.
    #[inline]
    #[must_use]
    pub fn leaf(local_cost: Cost, output_card: Cardinality) -> Self {
        Self {
            local_cost,
            subtree_cost: local_cost,
            output_card,
        }
    }

    /// Construct a unary-input cost node — combines local cost with a
    /// single child's subtree cost.
    #[inline]
    #[must_use]
    pub fn unary(local_cost: Cost, output_card: Cardinality, child: Cost) -> Self {
        Self {
            local_cost,
            subtree_cost: local_cost.plus(child),
            output_card,
        }
    }

    /// Construct an n-ary-input cost node — combines local cost with
    /// the sum of children's subtree costs.
    #[inline]
    #[must_use]
    pub fn n_ary(local_cost: Cost, output_card: Cardinality, children: &[Cost]) -> Self {
        let children_total = children.iter().copied().fold(Cost::zero(), Cost::plus);
        Self {
            local_cost,
            subtree_cost: local_cost.plus(children_total),
            output_card,
        }
    }
}

/// Parallel-tree cost annotation for a [`LogicalPlan`].
///
/// Each [`CostedTree`] node carries a [`CostNode`] plus child cost
/// trees in the same order as the underlying [`LogicalPlan`]'s
/// children. Walk this tree in lockstep with the plan tree to
/// produce EXPLAIN output (M4-91) or to feed M4-52's plan
/// enumeration.
#[derive(Debug, Clone)]
pub struct CostedTree {
    /// Cost annotation for this node.
    pub cost: CostNode,
    /// Child cost subtrees, in plan-tree order.
    pub children: Vec<CostedTree>,
}

impl CostedTree {
    /// Construct a leaf cost tree (no children).
    #[inline]
    #[must_use]
    pub fn leaf(cost: CostNode) -> Self {
        Self {
            cost,
            children: Vec::new(),
        }
    }

    /// Total subtree cost at this node — equivalent to
    /// `self.cost.subtree_cost`. Provided as a method for symmetry
    /// with downstream M4-52 plan-enumeration code.
    #[inline]
    #[must_use]
    pub fn total_cost(&self) -> Cost {
        self.cost.subtree_cost
    }

    /// Estimated output cardinality at this node.
    #[inline]
    #[must_use]
    pub fn output_card(&self) -> Cardinality {
        self.cost.output_card
    }
}

/// A cost-annotated logical plan — the public output of M4-51.
///
/// Wraps a [`LogicalPlan`] (consumed read-only — M4-51 does NOT
/// modify the plan tree) plus a parallel [`CostedTree`] of cost
/// annotations.
///
/// # Consumers
/// - **M4-52** (M4-05b — forward) plan enumeration: compares
///   alternative plan shapes by [`CostedPlan::total_cost`].
/// - **M4-91** (M4-08+ — forward) EXPLAIN: walks plan + costs in
///   lockstep to render annotated plan output.
/// - **M4-71** (M4-07+ — forward) row-count observer: compares
///   estimated output cardinalities to runtime-observed cardinalities;
///   feedback updates `CatalogStats`.
#[derive(Debug, Clone)]
pub struct CostedPlan {
    plan: LogicalPlan,
    costs: CostedTree,
    diagnostics: Vec<String>,
}

impl CostedPlan {
    /// Bundle a plan with its cost tree. The tree's pre-order walk
    /// MUST match the plan's pre-order walk; the M4-51 walker
    /// guarantees this.
    #[must_use]
    pub fn new(plan: LogicalPlan, costs: CostedTree) -> Self {
        Self {
            plan,
            costs,
            diagnostics: Vec::new(),
        }
    }

    /// Bundle a plan with cost diagnostics surfaced to EXPLAIN.
    #[must_use]
    pub fn with_diagnostics(
        plan: LogicalPlan,
        costs: CostedTree,
        diagnostics: Vec<String>,
    ) -> Self {
        Self {
            plan,
            costs,
            diagnostics,
        }
    }

    /// The wrapped logical plan.
    #[must_use]
    pub fn plan(&self) -> &LogicalPlan {
        &self.plan
    }

    /// Root cost annotation for the entire plan.
    #[must_use]
    pub fn root_cost(&self) -> CostNode {
        self.costs.cost
    }

    /// The full cost-tree.
    #[must_use]
    pub fn costs(&self) -> &CostedTree {
        &self.costs
    }

    /// Cost-walker diagnostics, including ADR-025 §5 supernode
    /// firewall degradation notes.
    #[must_use]
    pub fn diagnostics(&self) -> &[String] {
        &self.diagnostics
    }

    /// Total subtree cost — the value M4-52 uses to compare candidate
    /// plans.
    #[must_use]
    pub fn total_cost(&self) -> Cost {
        self.costs.total_cost()
    }

    /// Estimated output cardinality of the root operator.
    #[must_use]
    pub fn output_card(&self) -> Cardinality {
        self.costs.output_card()
    }

    /// Decompose into the plan + cost tree. Useful for callers that
    /// need ownership of both halves (e.g., the EXPLAIN renderer).
    #[must_use]
    pub fn into_parts(self) -> (LogicalPlan, CostedTree) {
        (self.plan, self.costs)
    }
}

/// Helper: capture a plan-time snapshot from any
/// [`CatalogProvider`] for use by the M4-51 cost walker.
///
/// Pulled out as a small function so the M4-52 forward consumer can
/// share the same snapshot-capture site. v1.0 simply delegates to
/// [`CatalogProvider::snapshot`]; the function exists as a stable
/// hook the M4-52 enumeration walker can reuse.
#[must_use]
pub(crate) fn capture_snapshot(catalog: &dyn CatalogProvider) -> crate::semantic::CatalogSnapshot {
    catalog.snapshot()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cost_new_saturates_nan_inf_negative_to_zero() {
        assert_eq!(Cost::new(f64::NAN).total(), 0.0);
        assert_eq!(Cost::new(f64::INFINITY).total(), 0.0);
        assert_eq!(Cost::new(-7.5).total(), 0.0);
        assert_eq!(Cost::new(0.0).total(), 0.0);
        assert_eq!(Cost::new(42.0).total(), 42.0);
    }

    #[test]
    fn cost_add_is_associative_and_zero_is_identity() {
        let a = Cost::new(1.0);
        let b = Cost::new(2.0);
        let c = Cost::new(3.0);
        // Associativity: (a + b) + c == a + (b + c).
        assert_eq!(((a + b) + c).total(), (a + (b + c)).total());
        // Identity.
        assert_eq!((a + Cost::zero()).total(), a.total());
    }

    #[test]
    fn cardinality_saturates_negative_and_nan() {
        assert_eq!(Cardinality::new(-1.0).rows(), 0.0);
        assert_eq!(Cardinality::new(f64::NAN).rows(), 0.0);
        // Inf saturates to f64::MAX (we don't want a planner to
        // produce Inf-cost downstream).
        assert_eq!(Cardinality::new(f64::INFINITY).rows(), f64::MAX);
        assert_eq!(Cardinality::new(7.5).rows(), 7.5);
    }

    #[test]
    fn cost_node_leaf_subtree_equals_local() {
        let n = CostNode::leaf(Cost::new(5.0), Cardinality::new(10.0));
        assert_eq!(n.local_cost.total(), 5.0);
        assert_eq!(n.subtree_cost.total(), 5.0);
        assert_eq!(n.output_card.rows(), 10.0);
    }

    #[test]
    fn cost_node_unary_sums_local_plus_child() {
        let n = CostNode::unary(Cost::new(2.0), Cardinality::new(3.0), Cost::new(7.0));
        assert_eq!(n.local_cost.total(), 2.0);
        assert_eq!(n.subtree_cost.total(), 9.0);
    }

    #[test]
    fn cost_node_n_ary_sums_local_plus_all_children() {
        let n = CostNode::n_ary(
            Cost::new(1.0),
            Cardinality::new(0.0),
            &[Cost::new(2.0), Cost::new(3.0), Cost::new(4.0)],
        );
        assert_eq!(n.subtree_cost.total(), 10.0);
    }
}
