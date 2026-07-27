//! I-13 — EXPLAIN ↔ live-planner cost-equivalence under fault injection.
//!
//! Closes GitHub issue #269.
//!
//! # The invariant (I-13)
//!
//! For any read query `Q` and any catalog state `C` (including
//! fault-degraded states such as stats-unavailable, mid-rebuild,
//! cross-key inconsistent), the cost number rendered by `EXPLAIN Q`
//! against `C` MUST equal the cost the live planner computes for the
//! plan it selects when executing `Q` against `C`.
//!
//! This invariant guards against regressions that would remove the
//! `enumerate_join_order` wiring from `crate::explain::plan_tree_for`
//! (PR #258) or otherwise diverge the EXPLAIN and live cost-source
//! paths.
//!
//! # How equivalence is achieved structurally
//!
//! Both paths consume:
//!
//! 1. A SINGLE `catalog.snapshot()` capture per call. EXPLAIN /
//!    `explain_with_cache` / `profile` use `FrozenCatalog` internally;
//!    the live `execute` path uses the same `FrozenCatalog` via
//!    `plan_for_execute` (issue #261 closure, PR #267).
//! 2. The same `enumerate_join_order_with_frozen` DP enumerator (the
//!    PR #258 wiring this test pins as load-bearing).
//! 3. The same `estimate_costs_with_frozen` per-operator cost walker.
//!
//! Under fault states the cost walker degrades gracefully via the
//! `DEFAULT_*_SELECTIVITY` constants per
//! `crate::planner::cost::operator` rustdoc; both paths see the same
//! degradation because they walk the same snapshot through the same
//! code. Determinism of the cost walker over `(plan, snapshot)`
//! completes the invariant.
//!
//! # Test ↔ production-path equivalence (PR #342 R1 §M-1 closure)
//!
//! `cost_via_live_planner` routes through
//! `QueryEngine::plan_and_cost_for_execute_for_test`, which is the
//! `#[doc(hidden)] pub` test-seam twin of
//! `QueryEngine::plan_for_execute`. Both methods share ONE
//! implementation (`plan_for_execute_optionally_costed`), so the
//! "live planner" comparand in this test is the LITERAL production-
//! path cost number — not a public-API reapproximation that captures
//! its own snapshots. Future refactors that change how production
//! captures its `FrozenCatalog` or wires the DP enumerator
//! automatically apply to this test's comparand too.
//!
//! # Test class layout
//!
//! - **Static fault states** (`cost_equivalence_under_*`): each test
//!   builds a catalog in a specific fault state, runs both paths, and
//!   asserts equal cost + equal plan shape.
//! - **Phase 4.3 reverse-test pin** (multi-join skewed-cardinality
//!   chain): asserts that EXPLAIN's reported cost matches the live
//!   planner's cost on a query where the DP must re-order. Removing
//!   the `enumerate_join_order_with_frozen` call from
//!   `plan_tree_for` would flip EXPLAIN's reported cost to the
//!   un-enumerated (input-order) plan's cost, diverging from the
//!   live planner's optimized-plan cost — this test would fail.
//! - **Proptest** (`prop_cost_equivalence_under_random_fault_state`):
//!   100 iterations of randomized fault states.
//!
//! # Bounded-context posture
//!
//! This test lives in `arcgraph-query` (the EXPLAIN ↔ live planner
//! comparison), NOT in `arcgraph-storage` where the K-3 storage slice
//! shipped (PR #264). Wiring a query-crate test inside a storage-crate
//! test crate would violate the bounded-context discipline.
//!
//! # ADR cites
//!
//! - ADR-038 §2 D-19 — EXPLAIN/PROFILE clause contract.
//! - ADR-038 amendment-02 §M4.e — M4-05 cost-based planner + DP
//!   join ordering.
//! - ADR-038 amendment-06 §D-25.1 — cross-key snapshot mechanism
//!   (the two-marker SeqLock protocol underlying M4-04e; PR #220
//!   producer; M4-51 + M4-52 + M4-91 consumers).

use arcgraph_core::{LabelId, TypeId};
use arcgraph_query::explain::{PlanTree, PlanTreeOp, QueryEngine, explain};
use arcgraph_query::logical_plan::LogicalPlan;
use arcgraph_query::parse_multi;
use arcgraph_query::planner::cost::Cost;
use arcgraph_query::semantic::{CatalogProvider, StubCatalogProvider};

use proptest::prelude::*;

// ---------------------------------------------------------------------
// Helpers: drive both paths through the same catalog
// ---------------------------------------------------------------------

/// Drive EXPLAIN over `query` against `cat` and return the rendered
/// plan tree's root cost.
fn cost_via_explain<C: CatalogProvider>(query: &str, cat: &C) -> (f64, PlanTree) {
    let pt = explain(query, cat).expect("EXPLAIN must not error on fault-free pipeline input");
    (pt.estimated_cost.total(), pt)
}

/// Drive the production live-planner pipeline (the same one
/// `QueryEngine::execute` → `plan_for_execute` uses) over `query`
/// against `cat` and return the optimized plan + its cost.
///
/// Routes through `QueryEngine::plan_and_cost_for_execute_for_test`,
/// which is the `#[doc(hidden)] pub` test-seam twin of
/// `QueryEngine::plan_for_execute` — both methods share ONE
/// implementation (`plan_for_execute_optionally_costed`), so any
/// future refactor that changes how production captures its
/// `FrozenCatalog` or wires the DP enumerator automatically applies
/// to this helper too. The cost is walked via the SAME
/// `estimate_costs_with_frozen` against the SAME externally-captured
/// `FrozenCatalog` that EXPLAIN uses, so the comparand in
/// `assert_cost_equivalence` is the literal production-path cost
/// number — not a public-API reapproximation that takes its own
/// snapshots (closes PR #342 R1 §M-1).
fn cost_via_live_planner<C: CatalogProvider>(query: &str, cat: &C) -> (f64, LogicalPlan) {
    let engine = QueryEngine::new(cat);
    let (optimized, cost) = engine
        .plan_and_cost_for_execute_for_test(query)
        .expect("plan_and_cost_for_execute must not error on fault-free pipeline input");
    (cost, optimized)
}

/// Shape extractor: pre-order list of operator names. Used for plan-
/// shape equivalence assertions (the cost equivalence test pins not
/// just the number but also that both paths picked the same plan).
fn shape(pt: &PlanTree) -> Vec<&'static str> {
    let mut out = Vec::new();
    walk(pt, &mut out);
    out
}

fn walk(pt: &PlanTree, out: &mut Vec<&'static str>) {
    out.push(pt.op.name());
    for c in &pt.children {
        walk(c, out);
    }
}

/// Assert cost-equivalence + plan-shape-equivalence between EXPLAIN
/// and the live planner over the same catalog. Bit-equal `f64`
/// comparison is the load-bearing oracle: both paths apply identical
/// arithmetic in identical order on identical inputs, so the result
/// is deterministic. A tiny epsilon (1e-12 relative) absorbs any
/// hypothetical IEEE-754 reassociation noise without weakening the
/// invariant.
#[track_caller]
fn assert_cost_equivalence<C: CatalogProvider>(query: &str, cat: &C, scenario: &str) {
    let (cost_explain, pt) = cost_via_explain(query, cat);
    let (cost_live, plan_live) = cost_via_live_planner(query, cat);

    let shape_explain = shape(&pt);
    let shape_live = shape_from_logical(&plan_live);
    assert_eq!(
        shape_explain, shape_live,
        "[{scenario}] EXPLAIN and live-planner picked DIFFERENT plan shapes — \
         I-13 cost-equivalence is meaningless across different plans. \
         EXPLAIN shape: {shape_explain:?}\n  live shape:    {shape_live:?}",
    );

    let abs_diff = (cost_explain - cost_live).abs();
    let rel_tol = 1e-12_f64 * cost_explain.abs().max(1.0);
    assert!(
        abs_diff <= rel_tol,
        "[{scenario}] I-13 VIOLATION: EXPLAIN cost ({cost_explain}) != \
         live-planner cost ({cost_live}); diff={abs_diff}, tol={rel_tol}. \
         Plan shape (same on both sides): {shape_explain:?}. \
         Likely cause: `enumerate_join_order_with_frozen` wiring has been \
         removed from `crate::explain::plan_tree_for` (PR #258 regression), \
         OR the cost walker's snapshot capture has drifted across the two \
         paths (issue #261 regression).",
    );
}

/// Project a `LogicalPlan` tree shape into a flat pre-order operator-
/// name list. Mirrors [`shape`] for the `PlanTree` side.
fn shape_from_logical(plan: &LogicalPlan) -> Vec<&'static str> {
    let mut out = Vec::new();
    walk_logical(plan, &mut out);
    out
}

fn walk_logical(plan: &LogicalPlan, out: &mut Vec<&'static str>) {
    let (name, children) = match plan {
        LogicalPlan::Scan(_) => ("Scan", Vec::new()),
        LogicalPlan::PropertyIndexScan(_) => ("PropertyIndexScan", Vec::new()),
        LogicalPlan::CountStore(_) => ("CountStore", Vec::new()),
        LogicalPlan::Empty(_) => ("Empty", Vec::new()),
        LogicalPlan::Expand(_) => ("Expand", Vec::new()),
        LogicalPlan::RankByHybrid(_) => ("RankByHybrid", Vec::new()),
        LogicalPlan::VectorNear(_) => ("VectorNear", Vec::new()),
        LogicalPlan::TextMatch(_) => ("TextMatch", Vec::new()),
        LogicalPlan::CreateNode(_) => ("CreateNode", Vec::new()),
        LogicalPlan::CreateVectorIndex(_) => ("CreateVectorIndex", Vec::new()),
        LogicalPlan::CreatePropertyIndex(_) => ("CreatePropertyIndex", Vec::new()),
        LogicalPlan::CreateRel(c) => ("CreateRel", vec![&*c.source_plan, &*c.target_plan]),
        LogicalPlan::Delete(d) => ("Delete", vec![&*d.input]),
        LogicalPlan::Set(s) => ("Set", vec![&*s.input]),
        LogicalPlan::Remove(r) => ("Remove", vec![&*r.input]),
        LogicalPlan::Filter(f) => ("Filter", vec![&*f.input]),
        LogicalPlan::Project(p) => ("Project", vec![&*p.input]),
        LogicalPlan::Limit(l) => ("Limit", vec![&*l.input]),
        LogicalPlan::Skip(s) => ("Skip", vec![&*s.input]),
        LogicalPlan::DynamicLimit(l) => ("DynamicLimit", vec![&*l.input]),
        LogicalPlan::Sort(s) => ("Sort", vec![&*s.input]),
        LogicalPlan::Distinct(d) => ("Distinct", vec![&*d.input]),
        LogicalPlan::Unwind(u) => ("Unwind", vec![&*u.input]),
        LogicalPlan::ProcedureCall(p) => ("Unwind", vec![&*p.input]),
        LogicalPlan::Aggregate(a) => ("Aggregate", vec![&*a.input]),
        LogicalPlan::CommunityLookup(c) => ("CommunityLookup", vec![&*c.input]),
        LogicalPlan::NamedPath(n) => ("NamedPath", vec![&*n.input]),
        LogicalPlan::Join(j) => ("Join", vec![&*j.left, &*j.right]),
        LogicalPlan::LeftOuterJoin(j) => ("LeftOuterJoin", vec![&*j.left, &*j.right]),
        LogicalPlan::Fusion(f) => {
            let kids: Vec<&LogicalPlan> = f.inputs.iter().map(|b| &**b).collect();
            ("Fusion", kids)
        }
        LogicalPlan::Union(u) => {
            let kids: Vec<&LogicalPlan> = u.arms.iter().collect();
            ("Union", kids)
        }
        LogicalPlan::Merge(m) => ("Merge", vec![&*m.match_branch, &*m.create_branch]),
        // ADR-192 (#623): CALL{} walks driving input + subquery body; the
        // seed is a leaf.
        LogicalPlan::Call(c) => ("Call", vec![&*c.input, &*c.body]),
        LogicalPlan::CorrelationSeed(_) => ("CorrelationSeed", Vec::new()),
    };
    out.push(name);
    for c in children {
        walk_logical(c, out);
    }
}

// ---------------------------------------------------------------------
// Catalog fixtures: each represents a discrete fault state
// ---------------------------------------------------------------------

/// Common label / rel-type / property set used across all fault-state
/// fixtures so the same query text exercises every test.
fn base_catalog() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["Person", "Comment", "Forum", "Place"])
        .with_rel_types(["KNOWS", "LIKES", "IS_LOCATED_IN"])
        .with_properties(["name", "city", "id", "age"])
}

/// Full stats present — the canonical baseline. Every label and
/// rel-type carries a cardinality; totals are populated.
fn cat_full_stats() -> StubCatalogProvider {
    base_catalog()
        .with_total_node_count(111_000)
        .with_total_rel_count(5_000_000)
        .with_label_cardinality(LabelId::new(1), 10_000) // Person
        .with_label_cardinality(LabelId::new(2), 100_000) // Comment
        .with_label_cardinality(LabelId::new(3), 10_000) // Forum
        .with_label_cardinality(LabelId::new(4), 1_000) // Place
        .with_rel_type_cardinality(TypeId::new(1), 3_000_000) // KNOWS
        .with_rel_type_cardinality(TypeId::new(2), 1_250_000) // LIKES
        .with_rel_type_cardinality(TypeId::new(3), 750_000) // IS_LOCATED_IN
}

/// Stats UNAVAILABLE — fresh-tenant / cold-start state. Every counter
/// returns `None`; the cost walker falls back to
/// `DEFAULT_*_SELECTIVITY` constants per
/// `crate::planner::cost::operator` rustdoc.
fn cat_stats_unavailable() -> StubCatalogProvider {
    base_catalog()
}

/// Partial label cardinalities — totals + rel-type cardinalities are
/// known but per-label values are missing. Simulates a mid-rebuild
/// state where only the global aggregates have been observed.
fn cat_partial_label_cards() -> StubCatalogProvider {
    base_catalog()
        .with_total_node_count(111_000)
        .with_total_rel_count(5_000_000)
        .with_rel_type_cardinality(TypeId::new(1), 3_000_000)
        .with_rel_type_cardinality(TypeId::new(2), 1_250_000)
        .with_rel_type_cardinality(TypeId::new(3), 750_000)
}

/// Partial rel-type cardinalities — labels + totals are known, per-
/// rel-type values are missing.
fn cat_partial_rel_type_cards() -> StubCatalogProvider {
    base_catalog()
        .with_total_node_count(111_000)
        .with_total_rel_count(5_000_000)
        .with_label_cardinality(LabelId::new(1), 10_000)
        .with_label_cardinality(LabelId::new(2), 100_000)
        .with_label_cardinality(LabelId::new(3), 10_000)
        .with_label_cardinality(LabelId::new(4), 1_000)
}

/// Zero totals + zero per-key cardinalities — the "observed-then-
/// fully-deleted" sentinel state. Distinct from "stats unavailable":
/// `total_nodes() == Some(0)` (commits observed; current count is 0)
/// rather than `None` (no commits observed yet).
fn cat_zero_observed() -> StubCatalogProvider {
    base_catalog()
        .with_total_node_count(0)
        .with_total_rel_count(0)
        .with_label_cardinality(LabelId::new(1), 0)
        .with_label_cardinality(LabelId::new(2), 0)
        .with_label_cardinality(LabelId::new(3), 0)
        .with_label_cardinality(LabelId::new(4), 0)
        .with_rel_type_cardinality(TypeId::new(1), 0)
        .with_rel_type_cardinality(TypeId::new(2), 0)
        .with_rel_type_cardinality(TypeId::new(3), 0)
}

/// Cross-key inconsistency — `sum(label_cards) > total_nodes`. The
/// cost walker MUST tolerate this defensively: per the
/// `CatalogSnapshot` docs §"Cross-key consistency" producers SHOULD
/// honor `sum(label_cards) ≤ total_nodes`, but a fault-recovery
/// window or a manual catalog edit COULD violate it. The invariant
/// here is "EXPLAIN and live agree", not "the cost is sensible";
/// graceful degradation to a finite, non-NaN cost is the contract
/// (per `crate::planner::cost::Cost::new` rustdoc).
fn cat_cross_key_inconsistent() -> StubCatalogProvider {
    base_catalog()
        .with_total_node_count(1_000) // SMALLER than sum below
        .with_total_rel_count(2_000)
        .with_label_cardinality(LabelId::new(1), 10_000) // 10× total
        .with_label_cardinality(LabelId::new(2), 100_000)
        .with_label_cardinality(LabelId::new(3), 10_000)
        .with_label_cardinality(LabelId::new(4), 1_000)
}

/// Skewed cardinalities — 5 orders of magnitude between smallest and
/// largest label. The DP MUST re-order joins to put the smallest leaf
/// leftmost; EXPLAIN's reported cost MUST be the post-DP cost
/// (matching what the live planner picks). This is the Phase 4.3
/// reverse-test surrogate: under skewed cardinalities the wired
/// EXPLAIN cost differs from the un-wired (input-order) cost.
fn cat_skewed_for_phase_4_3_pin() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["Small", "Medium", "Large"])
        .with_rel_types(["R1", "R2"])
        .with_total_node_count(1_001_010)
        .with_total_rel_count(2_000_000)
        .with_label_cardinality(LabelId::new(1), 10) // Small
        .with_label_cardinality(LabelId::new(2), 1_000) // Medium
        .with_label_cardinality(LabelId::new(3), 1_000_000) // Large
}

// ---------------------------------------------------------------------
// The 3-leaf inner-join query: the cost-equivalence workhorse.
// ---------------------------------------------------------------------

const JOIN_QUERY: &str = "MATCH (p:Person)-[:KNOWS]->(c:Comment)-[:IS_LOCATED_IN]->(pl:Place) \
                          RETURN p, c, pl";

const FILTER_QUERY: &str = "MATCH (p:Person) WHERE p.age > 30 RETURN p.name";

const SIMPLE_SCAN_QUERY: &str = "MATCH (p:Person) RETURN p";

const SKEWED_JOIN_QUERY: &str = "MATCH (c:Large)-[:R1]->(b:Medium)-[:R2]->(a:Small) RETURN a, b, c";

// ---------------------------------------------------------------------
// Static fault-state pins
// ---------------------------------------------------------------------

#[test]
fn cost_equivalence_full_stats_simple_scan() {
    assert_cost_equivalence(
        SIMPLE_SCAN_QUERY,
        &cat_full_stats(),
        "full_stats/simple_scan",
    );
}

#[test]
fn cost_equivalence_full_stats_filter() {
    assert_cost_equivalence(FILTER_QUERY, &cat_full_stats(), "full_stats/filter");
}

#[test]
fn cost_equivalence_full_stats_three_leaf_join() {
    // Multi-join query: the DP re-orders; cost-equivalence pins that
    // EXPLAIN's reported cost matches the live planner's picked-plan
    // cost (the Phase 4.3 reverse-test surrogate; see also
    // `cost_equivalence_skewed_chain_pins_dp_wiring` below for the
    // strong-divergence variant).
    assert_cost_equivalence(JOIN_QUERY, &cat_full_stats(), "full_stats/three_leaf_join");
}

#[test]
fn cost_equivalence_stats_unavailable_simple_scan() {
    assert_cost_equivalence(
        SIMPLE_SCAN_QUERY,
        &cat_stats_unavailable(),
        "stats_unavailable/simple_scan",
    );
}

#[test]
fn cost_equivalence_stats_unavailable_three_leaf_join() {
    assert_cost_equivalence(
        JOIN_QUERY,
        &cat_stats_unavailable(),
        "stats_unavailable/three_leaf_join",
    );
}

#[test]
fn cost_equivalence_partial_label_cards_three_leaf_join() {
    // `total_node_count` is present but per-label `label_card()` returns
    // `None` for every label. The cost walker falls back to
    // `DEFAULT_LABEL_SELECTIVITY * total_node_count` per the operator
    // module rustdoc.
    assert_cost_equivalence(
        JOIN_QUERY,
        &cat_partial_label_cards(),
        "partial_label_cards/three_leaf_join",
    );
}

#[test]
fn cost_equivalence_partial_rel_type_cards_three_leaf_join() {
    // Mirror of the partial_label_cards case for the rel-type axis.
    assert_cost_equivalence(
        JOIN_QUERY,
        &cat_partial_rel_type_cards(),
        "partial_rel_type_cards/three_leaf_join",
    );
}

#[test]
fn cost_equivalence_zero_observed_three_leaf_join() {
    // `Some(0)` is the "observed-then-fully-deleted" sentinel — distinct
    // from `None`. Cost walker returns finite, non-NaN costs (per the
    // `Cost::new` saturation invariant); EXPLAIN and live agree.
    assert_cost_equivalence(
        JOIN_QUERY,
        &cat_zero_observed(),
        "zero_observed/three_leaf_join",
    );
}

#[test]
fn cost_equivalence_cross_key_inconsistent_three_leaf_join() {
    // Defensive case: producer violated `sum(label_cards) ≤
    // total_nodes`. The cost walker tolerates the inconsistency; both
    // paths reach the same (potentially-implausible) cost number; the
    // invariant pinned here is "EXPLAIN agrees with live planner", NOT
    // "the cost is realistic".
    assert_cost_equivalence(
        JOIN_QUERY,
        &cat_cross_key_inconsistent(),
        "cross_key_inconsistent/three_leaf_join",
    );
}

#[test]
fn cost_equivalence_explain_wrapper_idempotent() {
    // Bare query vs `EXPLAIN <query>` wrapper produce the same cost
    // (per the EXPLAIN wrapper-stripping discipline; mirror of the
    // `explain_keyword_prefix_strips_the_wrapper` pin in
    // `tests/m4_91_explain_integration.rs`). The live planner side
    // is invariant under the wrapper because it always sees the inner
    // statement after `strip_explain_or_profile`.
    let cat = cat_full_stats();
    let (cost_bare, _) = cost_via_explain(JOIN_QUERY, &cat);
    let (cost_wrapped, _) = cost_via_explain(&format!("EXPLAIN {JOIN_QUERY}"), &cat);
    assert_eq!(
        cost_bare, cost_wrapped,
        "EXPLAIN wrapper must be a control-bit only — cost values diverged",
    );
}

// ---------------------------------------------------------------------
// Phase 4.3 reverse-test surrogate: skewed-chain DP-wiring pin
// ---------------------------------------------------------------------

/// Phase 4.3 reverse-test surrogate. Under skewed cardinalities the DP
/// must re-order joins; EXPLAIN's reported cost MUST equal the cost
/// the live planner uses for its optimized plan.
///
/// # Reverse-test cycle (verified at slice-build time)
///
/// 1. **State 1** (wiring on, baseline) — EXPLAIN cost == live cost
///    on the skewed chain (both reflect the post-DP cost of the
///    Small-leftmost plan).
/// 2. **State 2** (wiring removed — comment out the
///    `enumerate_join_order_with_frozen` call in
///    `crate::explain::plan_tree_for`) — EXPLAIN cost FLIPS to the
///    un-enumerated plan's cost (Large-leftmost, ~12.37 M) while the
///    live planner still optimizes (Small-leftmost, ~11.37 M). The
///    `abs_diff > rel_tol` arm of `assert_cost_equivalence` triggers
///    and this test FAILS, surfacing the regression.
/// 3. **State 3** (wiring restored) — back to State 1.
///
/// This is the load-bearing reverse-test pin for the I-13 invariant:
/// it is a NORMAL (non-`#[ignore]`'d) test that runs in every CI
/// build, so the wiring presence is verified on every commit.
#[test]
fn cost_equivalence_skewed_chain_pins_dp_wiring() {
    assert_cost_equivalence(
        SKEWED_JOIN_QUERY,
        &cat_skewed_for_phase_4_3_pin(),
        "phase_4_3_reverse_test/skewed_chain",
    );

    // Strong-shape pin: in addition to cost-equivalence, the optimized
    // plan's leftmost Scan leaf MUST be the smallest (Small=10). If
    // the wiring is removed, the leftmost leaf flips to Large=1M,
    // independently catching the regression even if a future cost-
    // model refinement makes the numeric difference smaller than the
    // 1e-12 relative tolerance.
    let cat = cat_skewed_for_phase_4_3_pin();
    let (_, pt) = cost_via_explain(SKEWED_JOIN_QUERY, &cat);
    let leftmost = leftmost_scan(&pt).expect("skewed-chain plan has a Scan leaf");
    assert!(
        (leftmost.estimated_card.rows() - 10.0).abs() < 1e-9,
        "DP must put smallest-leaf (Small=10) leftmost; got {} rows. \
         Likely cause: `enumerate_join_order_with_frozen` wiring removed \
         from `plan_tree_for`. Tree:\n{pt}",
        leftmost.estimated_card.rows(),
    );
}

fn leftmost_scan(pt: &PlanTree) -> Option<&PlanTree> {
    let mut node = pt;
    loop {
        if node.op == PlanTreeOp::Scan {
            return Some(node);
        }
        node = node.children.first()?;
    }
}

// ---------------------------------------------------------------------
// Cross-call internal-consistency pin: drift between two EXPLAINs
// ---------------------------------------------------------------------

/// Two successive `explain` calls on the SAME immutable catalog
/// produce the SAME cost. Pins determinism + the cost walker's
/// idempotence over `(query, catalog)`.
///
/// Distinct from cross-call equivalence under MUTATION (the I-13
/// invariant explicitly does NOT promise stability across catalog
/// mutations; it promises EACH call sees an internally-consistent
/// snapshot).
#[test]
fn cost_equivalence_is_deterministic_across_repeat_explains() {
    let cat = cat_full_stats();
    let (cost_a, pt_a) = cost_via_explain(JOIN_QUERY, &cat);
    let (cost_b, pt_b) = cost_via_explain(JOIN_QUERY, &cat);
    assert_eq!(
        cost_a, cost_b,
        "repeat EXPLAIN diverged on immutable catalog"
    );
    assert_eq!(
        shape(&pt_a),
        shape(&pt_b),
        "repeat EXPLAIN plan shape drifted"
    );
}

/// EXPLAIN cost reflects the catalog state at call time; mutating the
/// catalog between calls produces a different (but still internally-
/// consistent) cost on the second call.
///
/// This pins the "internal consistency per call" half of the I-13
/// invariant: each call observes a single snapshot and produces a
/// cost that's consistent with the live planner's plan choice on
/// THAT same snapshot. Calls with different snapshots may diverge —
/// the invariant promises consistency WITHIN each call, not across.
#[test]
fn cost_equivalence_each_call_internally_consistent_under_drift() {
    // Baseline state.
    let cat_a = cat_full_stats();
    let (cost_explain_a, _) = cost_via_explain(JOIN_QUERY, &cat_a);
    let (cost_live_a, _) = cost_via_live_planner(JOIN_QUERY, &cat_a);
    let abs_diff_a = (cost_explain_a - cost_live_a).abs();
    let tol_a = 1e-12_f64 * cost_explain_a.abs().max(1.0);
    assert!(
        abs_diff_a <= tol_a,
        "baseline call lost internal consistency: explain={cost_explain_a} live={cost_live_a}",
    );

    // Mutated state: scale every cardinality 10×. EXPLAIN + live both
    // see this NEW state; each call is internally consistent against
    // the new snapshot. The new costs differ from the baseline costs
    // — that is correct behavior under drift, NOT a violation of
    // I-13.
    let cat_b = base_catalog()
        .with_total_node_count(1_110_000)
        .with_total_rel_count(50_000_000)
        .with_label_cardinality(LabelId::new(1), 100_000)
        .with_label_cardinality(LabelId::new(2), 1_000_000)
        .with_label_cardinality(LabelId::new(3), 100_000)
        .with_label_cardinality(LabelId::new(4), 10_000)
        .with_rel_type_cardinality(TypeId::new(1), 30_000_000)
        .with_rel_type_cardinality(TypeId::new(2), 12_500_000)
        .with_rel_type_cardinality(TypeId::new(3), 7_500_000);
    let (cost_explain_b, _) = cost_via_explain(JOIN_QUERY, &cat_b);
    let (cost_live_b, _) = cost_via_live_planner(JOIN_QUERY, &cat_b);
    let abs_diff_b = (cost_explain_b - cost_live_b).abs();
    let tol_b = 1e-12_f64 * cost_explain_b.abs().max(1.0);
    assert!(
        abs_diff_b <= tol_b,
        "post-drift call lost internal consistency: \
         explain={cost_explain_b} live={cost_live_b}",
    );

    // Cost moved (mutation actually flowed through to cost output);
    // this is a non-vacuity check on the test itself — if the costs
    // were unchanged the drift case would be trivially passing
    // without exercising the invariant.
    assert!(
        cost_explain_b > cost_explain_a * 1.5,
        "drift did not flow through to cost: a={cost_explain_a} b={cost_explain_b} \
         — the test's non-vacuity oracle did not fire. Fix the test, not the code.",
    );
}

// ---------------------------------------------------------------------
// Proptest: randomized fault states (≥ 100 iterations)
// ---------------------------------------------------------------------

/// Compact randomized fault state. Each field independently models a
/// catalog state in fault-injection terms; the proptest body builds
/// the corresponding `StubCatalogProvider` and asserts the I-13
/// invariant.
#[derive(Debug, Clone)]
struct FaultState {
    total_nodes: Option<u64>,
    total_rels: Option<u64>,
    /// One entry per base-catalog label (Person, Comment, Forum, Place);
    /// `Some` populates that label's cardinality, `None` leaves it
    /// missing (fall-back path).
    label_cards: [Option<u64>; 4],
    /// Mirror of `label_cards` for rel-types (KNOWS, LIKES,
    /// IS_LOCATED_IN).
    rel_type_cards: [Option<u64>; 3],
    /// commits_observed watermark — drives the M4-53 plan cache;
    /// included for forward-compatibility with cached-EXPLAIN
    /// proptests, ignored by the cache-less path used here.
    commits_observed: u64,
}

prop_compose! {
    fn arb_fault_state()(
        total_nodes in proptest::option::of(0u64..2_000_000),
        total_rels in proptest::option::of(0u64..10_000_000),
        l0 in proptest::option::of(0u64..1_000_000),
        l1 in proptest::option::of(0u64..1_000_000),
        l2 in proptest::option::of(0u64..1_000_000),
        l3 in proptest::option::of(0u64..1_000_000),
        r0 in proptest::option::of(0u64..5_000_000),
        r1 in proptest::option::of(0u64..5_000_000),
        r2 in proptest::option::of(0u64..5_000_000),
        commits_observed in 0u64..1_000,
    ) -> FaultState {
        FaultState {
            total_nodes,
            total_rels,
            label_cards: [l0, l1, l2, l3],
            rel_type_cards: [r0, r1, r2],
            commits_observed,
        }
    }
}

fn build_cat_from_fault(state: &FaultState) -> StubCatalogProvider {
    let mut cat = base_catalog();
    if let Some(n) = state.total_nodes {
        cat = cat.with_total_node_count(n);
    }
    if let Some(n) = state.total_rels {
        cat = cat.with_total_rel_count(n);
    }
    for (i, val) in state.label_cards.iter().enumerate() {
        if let Some(c) = val {
            cat = cat.with_label_cardinality(LabelId::new((i + 1) as u32), *c);
        }
    }
    for (i, val) in state.rel_type_cards.iter().enumerate() {
        if let Some(c) = val {
            cat = cat.with_rel_type_cardinality(TypeId::new((i + 1) as u32), *c);
        }
    }
    cat.with_commits_observed_count(state.commits_observed)
}

proptest! {
    // 100 cases satisfies the spawn-prompt §Task 3 "100+ iterations"
    // ask. Per-iteration wall-clock is dominated by parse + cost-walk
    // (≪ 1 ms); 100 iterations finishes well inside the per-test
    // budget without dominating overall CI runtime.
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// I-13 invariant under randomized fault states.
    ///
    /// Asserts cost-equivalence + plan-shape-equivalence on the
    /// 3-leaf join query for every generated catalog state. The
    /// `prop_assert_*` macros surface the offending state in the
    /// proptest output on failure.
    #[test]
    fn prop_cost_equivalence_under_random_fault_state(state in arb_fault_state()) {
        let cat = build_cat_from_fault(&state);
        let (cost_explain, pt) = cost_via_explain(JOIN_QUERY, &cat);
        let (cost_live, plan_live) = cost_via_live_planner(JOIN_QUERY, &cat);

        let shape_explain = shape(&pt);
        let shape_live = shape_from_logical(&plan_live);
        prop_assert_eq!(
            shape_explain.clone(), shape_live.clone(),
            "shape divergence under fault {:?}", state,
        );

        let abs_diff = (cost_explain - cost_live).abs();
        let rel_tol = 1e-12_f64 * cost_explain.abs().max(1.0);
        prop_assert!(
            abs_diff <= rel_tol,
            "cost divergence under fault {:?}: explain={} live={} diff={} tol={}",
            state, cost_explain, cost_live, abs_diff, rel_tol,
        );

        // Cost is finite + non-NaN regardless of fault. Defense-in-
        // depth: the `Cost::new` constructor saturates pathological
        // inputs to zero, so this is a sanity check on that
        // saturation under randomized inputs (per `cost::mod`
        // rustdoc §"Stats=None graceful degradation").
        prop_assert!(
            cost_explain.is_finite(),
            "EXPLAIN cost not finite under fault {:?}: {}",
            state, cost_explain,
        );
        prop_assert!(
            cost_explain >= 0.0,
            "EXPLAIN cost negative under fault {:?}: {}",
            state, cost_explain,
        );
    }

    /// Sanity check: even on randomized fault states, EXPLAIN never
    /// errors at the planner-only path. Parse / bind / typecheck /
    /// cross-substrate validate run on identical inputs every
    /// iteration (catalog SHAPE — labels + rel-types + properties —
    /// is invariant; only cardinality numbers vary). Cardinality
    /// numbers do not affect those passes; they affect only the
    /// cost walker.
    #[test]
    fn prop_explain_does_not_error_on_random_fault_state(state in arb_fault_state()) {
        let cat = build_cat_from_fault(&state);
        let result = explain(JOIN_QUERY, &cat);
        prop_assert!(
            result.is_ok(),
            "EXPLAIN errored on fault {:?}: {:?}",
            state, result,
        );
    }
}

// ---------------------------------------------------------------------
// Multi-statement variant (per M4-83): parse_multi smoke check
// ---------------------------------------------------------------------

/// Confirms the I-13 helpers compose against `parse_multi` for
/// multi-statement input. v1.0 multi-statement input has the same
/// per-statement cost-equivalence contract as the single-statement
/// path (per ADR-038 §5.4.1 closure / M4-83). Smoke pin only — the
/// load-bearing per-statement coverage is in the single-statement
/// tests above.
#[test]
fn cost_equivalence_multi_statement_smoke() {
    let multi = format!("{JOIN_QUERY}; {FILTER_QUERY};");
    let cat = cat_full_stats();
    let stmts = parse_multi(&multi).expect("parse_multi");
    assert_eq!(
        stmts.len(),
        2,
        "expected 2 parsed statements, got {}",
        stmts.len()
    );
    // Per-statement cost-equivalence: each statement individually
    // honors I-13. The multi-statement executor's per-statement
    // planning routes through the same `enumerate_join_order_with_frozen`
    // + `estimate_costs_with_frozen` pair as the single-statement
    // path (per `crate::explain::QueryEngine::execute_multi_with_query_id_and_deadline`
    // implementation).
    assert_cost_equivalence(JOIN_QUERY, &cat, "multi_smoke/stmt_1");
    assert_cost_equivalence(FILTER_QUERY, &cat, "multi_smoke/stmt_2");
}

// ---------------------------------------------------------------------
// Cost saturation pins
// ---------------------------------------------------------------------

/// Per the `Cost::new` rustdoc, NaN / Inf / negative `f64` inputs to
/// the cost constructor saturate to 0. Under any fault state the
/// reported cost MUST be finite, non-NaN, and non-negative. The
/// proptest above asserts this stochastically; this test pins the
/// invariant deterministically on the union of all named fault
/// fixtures.
#[test]
fn cost_under_every_fault_fixture_is_finite_and_nonneg() {
    let fixtures: &[(&str, StubCatalogProvider)] = &[
        ("full_stats", cat_full_stats()),
        ("stats_unavailable", cat_stats_unavailable()),
        ("partial_label_cards", cat_partial_label_cards()),
        ("partial_rel_type_cards", cat_partial_rel_type_cards()),
        ("zero_observed", cat_zero_observed()),
        ("cross_key_inconsistent", cat_cross_key_inconsistent()),
    ];
    for (name, cat) in fixtures {
        let (cost_explain, _) = cost_via_explain(JOIN_QUERY, cat);
        let (cost_live, _) = cost_via_live_planner(JOIN_QUERY, cat);
        assert!(cost_explain.is_finite(), "[{name}] EXPLAIN cost not finite");
        assert!(cost_live.is_finite(), "[{name}] live cost not finite");
        assert!(cost_explain >= 0.0, "[{name}] EXPLAIN cost negative");
        assert!(cost_live >= 0.0, "[{name}] live cost negative");
        // Bonus: the `Cost` type's defensive constructor saturates
        // NaN inputs, so this is also an implicit check on the cost
        // walker's arithmetic chain.
        let _ = Cost::new(cost_explain);
        let _ = Cost::new(cost_live);
    }
}
