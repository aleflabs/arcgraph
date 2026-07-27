//! Issue #832 — CRITICAL silent data loss on multi-pattern CREATE.
//!
//! `CREATE (:T {n:1}),(:T {n:2}),(:T {n:3})` (multiple comma-separated
//! node patterns in ONE clause) persisted only the LAST node, silently
//! — no error. The decisive customer-reported (CZ) signature over a
//! live `arcgraph serve --bolt --data` with the neo4j driver was:
//!
//! ```text
//! MATCH (t:T) RETURN count(t)      → 1     (want 3)
//! MATCH (t:T) RETURN collect(t.n)  → [3]   (want {1,2,3})
//! ```
//!
//! Real-customer impact: any bulk import that emits multi-node CREATEs
//! could silently lose records.
//!
//! ## Root cause (bisected — query LOWERING, NOT storage)
//!
//! `lower_create` accumulated items with `current = Some(op)`, which
//! OVERWROTE the accumulator each iteration — the lowered plan kept
//! ONLY the last item. The other patterns never reached the executor
//! or storage at all (EXPLAIN showed a single `CreateNode`). Storage
//! is exonerated: it persisted the one node it was asked to, correctly.
//!
//! The fix threads each CREATE item as the next item's `input`, so the
//! lowering builds a left-deep chain where EVERY item executes.
//!
//! ## RED-on-revert
//!
//! Revert any of {`LogicalCreateNode::input` wiring in `lower_create`,
//! `CreateNodeOp::with_input` streaming, the pipeline `with_input`
//! build} and `count_t` collapses back to 1 / `collect_t` to `[3]`.
//! These assertions are the bug-specific oracle.

use std::collections::BTreeSet;

use arcgraph_core::{LabelId, Lsn, PartitionId, TenantId};
use arcgraph_query::executor::substrate::StubExecutorSubstrate;
use arcgraph_query::executor::{ExecutionContext, value::Value};
use arcgraph_query::logical_plan::{LogicalPlan, LogicalPlanLoweringVisitor};
use arcgraph_query::observer::{OperatorKind, walk_plan_and_costs};
use arcgraph_query::planner::cost::estimate_costs;
use arcgraph_query::semantic::{
    BindingVisitor, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};
use arcgraph_query::{
    ExecutorSubstrate, PlanTree, PlanTreeOp, Statement, executor::Pipeline, explain, parse,
};

/// LabelId the StubExecutorSubstrate assigns to the FIRST interned
/// label name (its `next_label` allocator starts at 1024). Pre-binding
/// the catalog to the same id closes the catalog↔substrate id
/// divergence so a MATCH-lowered Scan reads the CREATE-d nodes.
const STUB_FIRST_LABEL_ID: u32 = 1024;

fn lower_with(query: &str, cat: &StubCatalogProvider) -> LogicalPlan {
    let stmt = parse(query).expect("parse OK");
    let inner = match stmt {
        Statement::Read(_) => stmt,
        other => panic!("expected Read statement, got {other:?}"),
    };
    let mut bound = BindingVisitor::bind(&inner, query, cat).expect("bind OK");
    TypeCheckVisitor::check(&mut bound, cat).expect("type-check OK");
    CrossSubstrateValidator::validate(&bound, cat).expect("cross-substrate OK");
    LogicalPlanLoweringVisitor::lower(&bound).expect("lower OK")
}

fn lower(query: &str) -> LogicalPlan {
    lower_with(query, &StubCatalogProvider::new())
}

fn execute(
    plan: &LogicalPlan,
    substrate: &StubExecutorSubstrate,
    ctx: &ExecutionContext,
) -> Vec<Vec<Value>> {
    let mut op = Pipeline::build(plan).expect("pipeline build OK");
    let mut out: Vec<Vec<Value>> = Vec::new();
    loop {
        let b = op.next_batch(ctx, substrate).expect("batch OK");
        if b.is_empty() {
            break;
        }
        for i in 0..b.row_count() {
            out.push(b.row(i).to_vec());
        }
    }
    out
}

/// Number of create ops on the chain (follows `input`).
fn create_chain_depth(plan: &LogicalPlan) -> usize {
    match plan {
        LogicalPlan::CreateNode(c) => 1 + c.input.as_deref().map(create_chain_depth).unwrap_or(0),
        LogicalPlan::CreateRel(c) => 1 + c.input.as_deref().map(create_chain_depth).unwrap_or(0),
        LogicalPlan::Project(p) => create_chain_depth(&p.input),
        LogicalPlan::Filter(f) => create_chain_depth(&f.input),
        _ => 0,
    }
}

fn int_of(v: &Value) -> i64 {
    match v {
        Value::Integer(n) => *n,
        Value::Node(node) => match node.properties.get("n") {
            Some(Value::Integer(n)) => *n,
            other => panic!("node.n is not an integer: {other:?}"),
        },
        other => panic!("expected Integer, got {other:?}"),
    }
}

// =====================================================================
// PRIMARY — the exact CZ signature: count=3, collect={1,2,3} not [3].
// =====================================================================

#[test]
fn multi_pattern_create_persists_all_three_nodes() {
    let s = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);

    // The bug's exact query. All-anonymous, single clause, 3 patterns.
    let plan = lower("CREATE (:T {n: 1}),(:T {n: 2}),(:T {n: 3})");

    // Plan-shape oracle: a 3-item chain, NOT a single (last) CreateNode.
    assert_eq!(
        create_chain_depth(&plan),
        3,
        "multi-pattern CREATE MUST lower to a 3-item chain (every \
         pattern executes); the bug lowered to a single CreateNode: \
         {plan:#?}"
    );

    let _ = execute(&plan, &s, &ctx);

    // count(t) oracle: 3, not 1.
    let nodes = s
        .scan_nodes(TenantId::DEFAULT, None, Lsn::MAX)
        .expect("scan_nodes OK");
    assert_eq!(
        nodes.len(),
        3,
        "MATCH (t:T) RETURN count(t) MUST be 3 — the bug returned 1"
    );

    // collect(t.n) oracle: the SET {1,2,3}, not [3].
    let collected: BTreeSet<i64> = nodes.iter().map(int_of_node).collect();
    assert_eq!(
        collected,
        BTreeSet::from([1, 2, 3]),
        "collect(t.n) MUST be {{1,2,3}} — the bug returned [3] (only \
         the LAST pattern's value)"
    );
}

/// Read `n` off a scanned node.
fn int_of_node(bn: &arcgraph_query::executor::BoundNode) -> i64 {
    match bn.node.properties.get("n") {
        Some(Value::Integer(n)) => *n,
        other => panic!("node.n is not an integer: {other:?}"),
    }
}

#[test]
fn multi_pattern_create_round_trips_through_match() {
    // Customer-path fidelity: CREATE the 3 nodes, then read them back
    // through a real MATCH (t:T) RETURN t.n — mirrors the CZ probe.
    let s = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let create = lower("CREATE (:T {n: 1}),(:T {n: 2}),(:T {n: 3})");
    let _ = execute(&create, &s, &ctx);

    let cat = StubCatalogProvider::new().with_label_id("T", LabelId::new(STUB_FIRST_LABEL_ID));
    let ctx2 = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let match_rows = execute(&lower_with("MATCH (t:T) RETURN t.n", &cat), &s, &ctx2);

    assert_eq!(
        match_rows.len(),
        3,
        "MATCH (t:T) returns 3 rows (count(t)=3)"
    );
    let vals: BTreeSet<i64> = match_rows.iter().map(|r| int_of(&r[0])).collect();
    assert_eq!(
        vals,
        BTreeSet::from([1, 2, 3]),
        "MATCH (t:T) RETURN t.n yields {{1,2,3}} (collect=[3] was the bug)"
    );
}

// =====================================================================
// Determinism oracle — exact {creation-order node-id → n} mapping.
// =====================================================================

#[test]
fn multi_pattern_create_is_deterministic_left_to_right() {
    let s = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let _ = execute(
        &lower("CREATE (:T {n: 10}),(:T {n: 20}),(:T {n: 30})"),
        &s,
        &ctx,
    );

    let mut nodes = s
        .scan_nodes(TenantId::DEFAULT, None, Lsn::MAX)
        .expect("scan_nodes OK");
    // Allocation is monotonic in create order; sorting by node-id
    // recovers the left-to-right CREATE order.
    nodes.sort_by_key(|bn| bn.node.id);
    let seq: Vec<i64> = nodes.iter().map(int_of_node).collect();
    assert_eq!(
        seq,
        vec![10, 20, 30],
        "nodes persist in left-to-right CREATE order with the EXACT \
         per-pattern property — no collapse, no reorder, no dup"
    );
}

// =====================================================================
// Bound RETURN — every chained binding is in scope downstream.
// =====================================================================

#[test]
fn multi_pattern_create_bound_return_carries_all_bindings() {
    let s = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let plan = lower("CREATE (a:T {n: 1}),(b:T {n: 2}),(c:T {n: 3}) RETURN a, b, c");
    let rows = execute(&plan, &s, &ctx);

    assert_eq!(rows.len(), 1, "CREATE …,…,… RETURN a,b,c emits ONE row");
    assert_eq!(
        rows[0].len(),
        3,
        "the row binds all THREE created nodes (a,b,c) — the bug \
         dropped a,b from the plan's schema entirely"
    );
    let vals: BTreeSet<i64> = rows[0].iter().map(int_of).collect();
    assert_eq!(vals, BTreeSet::from([1, 2, 3]));

    // The three bound node-ids are DISTINCT (no slot-reuse / collapse).
    let ids: BTreeSet<_> = rows[0]
        .iter()
        .map(|v| match v {
            Value::Node(n) => n.id,
            other => panic!("expected Node, got {other:?}"),
        })
        .collect();
    assert_eq!(ids.len(), 3, "three DISTINCT node-ids");
}

// =====================================================================
// SISTER PATTERN (same root cause) — multi-PATH CREATE.
// `CREATE (a)-[:R]->(b),(c)-[:R]->(d)` collapsed to the LAST path.
// =====================================================================

#[test]
fn multi_path_create_persists_both_paths() {
    let s = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let plan = lower("CREATE (:A {n: 1})-[:R]->(:B {n: 2}),(:A {n: 3})-[:R]->(:B {n: 4})");

    // Two CreateRel ops on the chain (both paths lowered).
    assert_eq!(
        create_chain_depth(&plan),
        2,
        "multi-path CREATE MUST lower to a 2-path chain: {plan:#?}"
    );

    let _ = execute(&plan, &s, &ctx);

    // All FOUR endpoint nodes persist (the bug kept only the last path).
    let nodes = s
        .scan_nodes(TenantId::DEFAULT, None, Lsn::MAX)
        .expect("scan_nodes OK");
    assert_eq!(
        nodes.len(),
        4,
        "both paths' endpoints persist (4 nodes) — the bug persisted 2"
    );
    let vals: BTreeSet<i64> = nodes.iter().map(int_of_node).collect();
    assert_eq!(
        vals,
        BTreeSet::from([1, 2, 3, 4]),
        "all four endpoint property values present"
    );
}

// =====================================================================
// Single-pattern CREATE is UNCHANGED (no over-correction / regression).
// =====================================================================

#[test]
fn single_pattern_create_still_persists_exactly_one() {
    let s = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let plan = lower("CREATE (:T {n: 99})");
    assert_eq!(create_chain_depth(&plan), 1, "single CREATE = depth 1");
    let _ = execute(&plan, &s, &ctx);
    let nodes = s
        .scan_nodes(TenantId::DEFAULT, None, Lsn::MAX)
        .expect("scan_nodes OK");
    assert_eq!(nodes.len(), 1, "single-pattern CREATE persists exactly 1");
    assert_eq!(int_of_node(&nodes[0]), 99);
}

// =====================================================================
// EXPLAIN / PROFILE WALKER LOCKSTEP (#863 R1: HIGH + MED sister).
//
// The #832 execute-path fix added `LogicalCreateNode::input` /
// `LogicalCreateRel::input`, but two SIBLING walkers were not updated in
// lockstep with the EXPLAIN plan-tree walker:
//   - HIGH: the cost walker (`planner/cost/walker.rs`) still treated
//     CreateNode as a 0-child leaf and CreateRel as exactly 2 children,
//     so a multi-pattern CREATE produced a CostedTree whose arity
//     disagreed with the plan-tree walker. `PlanTree::from_costed_plan`
//     indexes `child_at(costs, 0)` (CreateNode) / `child_at(costs, 2)`
//     (CreateRel) — both PANIC on the short cost tree. This is a
//     REACHABLE server-crash on the public `explain()` API:
//        EXPLAIN CREATE (:T{n:1}),(:T{n:2})         → panic (idx 0 / 0 kids)
//        EXPLAIN CREATE (a)-[:R]->(b),(c)-[:R]->(d) → panic (idx 2 / 2 kids)
//   - MED: the PROFILE row-count walker
//     (`observer/row_count.rs::plan_children`) had the identical
//     leaf/2-arity assumption — its `zip` against the cost tree stops
//     short, silently dropping the chained creates' per-op rows.
//
// RED-on-revert: revert the walker.rs lockstep and the `explain()` tests
// PANIC inside `child_at`; revert row_count's `plan_children` and
// `profile_row_count_walk_attributes_every_chained_create` under-counts
// (1 instead of 3). Verified by temporary revert — see the #863 fix-up
// completion report.
// =====================================================================

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
fn explain_multi_pattern_create_does_not_panic_and_shows_all_creates() {
    let cat = StubCatalogProvider::new();
    // The reviewer's exact repro form. Before this fix-up the cost
    // walker emitted a 0-child leaf for the chained CreateNode and
    // `PlanTree::from_costed_plan` PANICKED in `child_at(costs, 0)`.
    let tree = explain("EXPLAIN CREATE (:T {n: 1}),(:T {n: 2}),(:T {n: 3})", &cat)
        .expect("EXPLAIN of a 3-pattern CREATE plans without error (was a reachable panic)");

    // STRUCTURE oracle (not merely "no panic"): the plan tree shows all
    // THREE chained CreateNode ops — the honest N-op output the PR body
    // claims. The bug showed exactly ONE.
    assert_eq!(
        count_plan_tree_op(&tree, PlanTreeOp::CreateNode),
        3,
        "EXPLAIN must show 3 chained CreateNode ops, not 1: {tree:#?}"
    );
}

#[test]
fn explain_multi_path_create_does_not_panic_and_shows_both_rels() {
    let cat = StubCatalogProvider::new();
    // Multi-PATH sister: before the fix-up the cost walker emitted only
    // [source, target] for CreateRel, so `child_at(costs, 2)` (the chain
    // `input`) PANICKED with "child index 2 … only 2 children".
    let tree = explain("EXPLAIN CREATE (a)-[:R]->(b),(c)-[:R]->(d)", &cat)
        .expect("EXPLAIN of a 2-path CREATE plans without error (was a reachable panic)");

    // STRUCTURE oracle: both chained CreateRel ops + all four endpoint
    // CreateNode producers appear (the bug collapsed to one path).
    assert_eq!(
        count_plan_tree_op(&tree, PlanTreeOp::CreateRel),
        2,
        "EXPLAIN must show 2 chained CreateRel ops (both paths): {tree:#?}"
    );
    assert_eq!(
        count_plan_tree_op(&tree, PlanTreeOp::CreateNode),
        4,
        "EXPLAIN must show all 4 endpoint CreateNode producers: {tree:#?}"
    );
}

#[test]
fn profile_row_count_walk_attributes_every_chained_create() {
    // MED sister: the PROFILE row-count walker
    // (`observer::walk_plan_and_costs` → `plan_children`) must recurse
    // the CREATE-item chain in lockstep with the cost tree, else the
    // chained creates' per-op rows are silently dropped (the `zip`
    // stops short). RED-on-revert of `row_count.rs::plan_children`: this
    // walk yields 1 CreateNode entry instead of 3.
    let cat = StubCatalogProvider::new();
    let plan = lower("CREATE (:T {n: 1}),(:T {n: 2}),(:T {n: 3})");
    let costed = estimate_costs(plan, &cat);
    let entries = walk_plan_and_costs(costed.plan(), costed.costs());

    let create_entries = entries
        .iter()
        .filter(|e| e.op_kind == OperatorKind::CreateNode)
        .count();
    assert_eq!(
        create_entries, 3,
        "the PROFILE row-count walk must emit one entry per chained \
         CreateNode (3) — the un-lockstepped walker emitted 1: {entries:#?}"
    );
}
