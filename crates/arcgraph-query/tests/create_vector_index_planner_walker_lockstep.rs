//! Regression guard — CREATE VECTOR INDEX planner-walker LOCKSTEP
//! (the #872 rebase onto #863).
//!
//! ## Why this test exists
//!
//! #863 (`fix(query): chain every CREATE item`) made
//! `LogicalPlan::CreateNode` / `CreateRel` NON-leaves — each gained an
//! optional `input` child for the left-deep multi-pattern-CREATE chain —
//! and updated the THREE cost-tree-lockstep walkers together:
//!   - `planner::cost::walker` (builds the `CostedTree`),
//!   - `explain::PlanTree::from_costed_plan` (walks plan + cost trees in
//!     lockstep, indexing `child_at(costs, N)`),
//!   - `observer::row_count::plan_children` (the PROFILE row-count walker;
//!     `zip`s plan-children against the cost tree).
//!
//! #863's R1 caught a REACHABLE `EXPLAIN`/`PROFILE` server panic when
//! these walkers were NOT in lockstep: `from_costed_plan` indexed a child
//! slot the cost walker had not produced (`child index 0 … only 0
//! children`). The lockstep is therefore correctness-load-bearing.
//!
//! #872 adds `LogicalPlan::CreateVectorIndex` as a **LEAF** (metadata-
//! only DDL — no `input` child, nothing to recurse). The rebase had to
//! KEEP #863's input-child recursion for CreateNode/CreateRel AND add
//! CreateVectorIndex as a leaf in all three walkers consistently. This
//! test pins that lockstep for the new leaf: the cost walker, the
//! `PlanTree` walker (the #863 panic site), and the row-count walker MUST
//! all agree that a CreateVectorIndex plan has ZERO children.
//!
//! ## Note on the EXPLAIN / PROFILE keyword surface
//!
//! `EXPLAIN`/`PROFILE` wrap a `ReadQuery` (`Statement::Explain(ReadQuery)`
//! / `Statement::Profile(ReadQuery)`), whereas CREATE VECTOR INDEX is a
//! top-level `Statement::IndexDdl`. So `EXPLAIN CREATE VECTOR INDEX …` is
//! a clean PARSE ERROR — NOT a reachable plan-tree surface, and NOT a
//! panic (contrast `EXPLAIN CREATE (:T)`, where CREATE-node is a
//! read-query *clause* and does plan). The lockstep walkers are therefore
//! exercised by driving the exact planner path the `explain()` /
//! `profile()` APIs use internally — `estimate_costs`
//! → `PlanTree::from_costed_plan` → `walk_plan_and_costs` — directly on
//! the lowered CreateVectorIndex plan. That IS the precise code path (and
//! panic site) #863 fixed; the leaf-arms added here are what keep it safe.

use arcgraph_query::logical_plan::{LogicalPlan, LogicalPlanLoweringVisitor};
use arcgraph_query::observer::{OperatorKind, walk_plan_and_costs};
use arcgraph_query::planner::cost::estimate_costs;
use arcgraph_query::semantic::{
    BindingVisitor, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};
use arcgraph_query::{PlanTree, PlanTreeOp, explain, parse};

/// A literal-form CREATE VECTOR INDEX (no `$param`s → binds + lowers
/// without a parameter bag). Mirrors the #872 e2e literal form.
const CVI_QUERY: &str = "CREATE VECTOR INDEX myidx FOR (n:Doc) ON n.vec";

/// parse → bind → type-check → cross-substrate → lower a DDL statement:
/// the front-end half of the `explain()` / execute pipeline, minus the
/// `issue_832` helper's Read-only guard (CREATE VECTOR INDEX is a
/// top-level `Statement::IndexDdl`, accept-and-register per ADR-200).
/// Returns the lowered `LogicalPlan::CreateVectorIndex`.
fn lower_ddl(query: &str, cat: &StubCatalogProvider) -> LogicalPlan {
    let stmt = parse(query).expect("parse OK");
    let mut bound = BindingVisitor::bind(&stmt, query, cat).expect("bind OK");
    TypeCheckVisitor::check(&mut bound, cat).expect("type-check OK (CVI is accept-and-register)");
    CrossSubstrateValidator::validate(&bound, cat).expect("cross-substrate OK");
    LogicalPlanLoweringVisitor::lower(&bound).expect("lower OK")
}

/// DFS count of a given operator kind across the whole [`PlanTree`].
fn count_plan_tree_op(tree: &PlanTree, op: PlanTreeOp) -> usize {
    let here = usize::from(tree.op == op);
    here + tree
        .children
        .iter()
        .map(|child| count_plan_tree_op(child, op))
        .sum::<usize>()
}

#[test]
fn create_vector_index_lowers_to_a_leaf_plan() {
    let cat = StubCatalogProvider::new();
    let plan = lower_ddl(CVI_QUERY, &cat);
    assert!(
        matches!(plan, LogicalPlan::CreateVectorIndex(_)),
        "CREATE VECTOR INDEX must lower to LogicalPlan::CreateVectorIndex, got {plan:?}"
    );
}

#[test]
fn cost_walker_emits_a_zero_child_leaf_for_create_vector_index() {
    // The cost walker (`planner::cost::walker`) MUST emit a 0-child
    // CostedTree leaf for CreateVectorIndex. #863's panic class is a
    // cost-tree / plan-tree child-count DISAGREEMENT; this pins the
    // cost-tree side at zero children (the lockstep contract for a leaf).
    let cat = StubCatalogProvider::new();
    let plan = lower_ddl(CVI_QUERY, &cat);
    let costed = estimate_costs(plan, &cat);
    assert!(
        costed.costs().children.is_empty(),
        "CreateVectorIndex cost tree must be a leaf (0 children), got {}: {:#?}",
        costed.costs().children.len(),
        costed.costs()
    );
}

#[test]
fn plan_tree_from_costed_plan_does_not_panic_for_create_vector_index() {
    // THE #863 panic site. `PlanTree::from_costed_plan` walks the plan +
    // cost trees in lockstep and indexes `child_at(costs, N)`. If the
    // CreateVectorIndex arm in the plan-tree walker disagreed with the
    // cost walker on child count, this PANICS (the exact `EXPLAIN`/
    // `PROFILE` server-crash class). A leaf (0 children) in both walkers
    // → no panic + a single CreateVectorIndex op.
    let cat = StubCatalogProvider::new();
    let plan = lower_ddl(CVI_QUERY, &cat);
    let costed = estimate_costs(plan, &cat);
    let tree = PlanTree::from_costed_plan(&costed); // must NOT panic
    assert_eq!(
        count_plan_tree_op(&tree, PlanTreeOp::CreateVectorIndex),
        1,
        "plan tree must show exactly 1 CreateVectorIndex op: {tree:#?}"
    );
    assert!(
        tree.children.is_empty(),
        "the CreateVectorIndex PlanTree node must be a leaf (0 children): {tree:#?}"
    );
}

#[test]
fn profile_row_count_walk_attributes_the_create_vector_index_leaf() {
    // The PROFILE row-count walker (`observer::walk_plan_and_costs`
    // → `plan_children`) `zip`s plan-children against the cost tree. A
    // leaf CreateVectorIndex must yield exactly ONE entry — lockstep with
    // the cost + plan-tree walkers. An arity disagreement would drop or
    // mis-attribute it (the #863 MED sister class). This is the PROFILE
    // variant of the no-panic guard.
    let cat = StubCatalogProvider::new();
    let plan = lower_ddl(CVI_QUERY, &cat);
    let costed = estimate_costs(plan, &cat);
    let entries = walk_plan_and_costs(costed.plan(), costed.costs());
    let cvi_entries = entries
        .iter()
        .filter(|e| e.op_kind == OperatorKind::CreateVectorIndex)
        .count();
    assert_eq!(
        cvi_entries, 1,
        "the PROFILE row-count walk must emit exactly one CreateVectorIndex entry: {entries:#?}"
    );
}

#[test]
fn explain_api_on_create_vector_index_does_not_panic() {
    // Faithful "EXPLAIN of CREATE VECTOR INDEX must not panic" guard
    // through the PUBLIC `explain()` API with the literal keyword form.
    // EXPLAIN wraps a `ReadQuery`; CREATE VECTOR INDEX is a top-level
    // `Statement::IndexDdl` — so this is a clean parse error (`Err`),
    // never a panic. Reaching the assert at all proves no panic.
    let cat = StubCatalogProvider::new();
    let result = explain(
        "EXPLAIN CREATE VECTOR INDEX myidx FOR (n:Doc) ON n.vec",
        &cat,
    );
    assert!(
        result.is_err(),
        "EXPLAIN over a top-level IndexDdl statement is a clean parse error \
         (EXPLAIN wraps ReadQuery, CVI is IndexDdl), got Ok: {result:?}"
    );
}
