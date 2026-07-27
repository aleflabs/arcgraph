//! ADR-151 W26-θ Phase 5 — MERGE node-shape create-branch smoke test.
//!
//! `MERGE (n:User {id: 42})` on an empty store: the match-branch
//! probes (returns 0 rows); the create-branch fires (allocates a new
//! NodeId via the Stub's `create_node`). The test asserts the Stub
//! observes the new node in subsequent `scan_nodes`.

use arcgraph_core::{Lsn, PartitionId, TenantId};

use arcgraph_query::executor::substrate::{ExecutorSubstrate, StubExecutorSubstrate};
use arcgraph_query::executor::{ExecutionContext, Pipeline};
use arcgraph_query::logical_plan::{LogicalPlan, LogicalPlanLoweringVisitor};
use arcgraph_query::parse;
use arcgraph_query::semantic::{
    BindingVisitor, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};

fn lower(query: &str) -> LogicalPlan {
    let stmt = parse(query).expect("parse OK");
    let cat = StubCatalogProvider::new();
    let mut bound = BindingVisitor::bind(&stmt, query, &cat).expect("bind OK");
    TypeCheckVisitor::check(&mut bound, &cat).expect("type-check OK");
    CrossSubstrateValidator::validate(&bound, &cat).expect("cross-substrate OK");
    LogicalPlanLoweringVisitor::lower(&bound).expect("lower OK")
}

fn drain(plan: &LogicalPlan, stub: &StubExecutorSubstrate, ctx: &ExecutionContext) {
    let mut op = Pipeline::build(plan).expect("pipeline build OK");
    loop {
        let b = op.next_batch(ctx, stub).expect("batch OK");
        if b.is_empty() {
            break;
        }
    }
}

#[test]
fn merge_node_parses_and_lowers_to_logical_merge() {
    let plan = lower("MERGE (n:User {id: 42})");
    assert!(
        matches!(&plan, LogicalPlan::Merge(_)),
        "expected LogicalPlan::Merge at top of plan; got {plan:?}"
    );
}

#[test]
fn merge_node_round_trip_display() {
    let stmt = parse("MERGE (n:User {id: 42})").expect("parse OK");
    let printed = format!("{stmt}");
    let reparsed = parse(&printed).expect("reparse OK");
    assert_eq!(stmt, reparsed, "Display round-trip failed: `{printed}`");
}

#[test]
fn merge_node_create_branch_fires_on_empty_store() {
    // ADR-151 §D-7: match-branch probes an empty store → returns 0
    // rows → create-branch fires → Stub allocates a new node.
    let tenant = TenantId::DEFAULT;
    let plan = lower("MERGE (n:User {id: 42})");
    let s = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);
    drain(&plan, &s, &ctx);
    let nodes = s.scan_nodes(tenant, None, Lsn::MAX).expect("scan_nodes OK");
    assert_eq!(
        nodes.len(),
        1,
        "create-branch should have created exactly 1 node on empty store"
    );
}

#[test]
fn merge_node_anonymous_create_branch_fires_on_empty_store() {
    // `MERGE (:User)` — anonymous binding shape; still works via
    // create-branch on empty store.
    let tenant = TenantId::DEFAULT;
    let plan = lower("MERGE (:User)");
    let s = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);
    drain(&plan, &s, &ctx);
    let nodes = s.scan_nodes(tenant, None, Lsn::MAX).expect("scan_nodes OK");
    assert_eq!(
        nodes.len(),
        1,
        "anonymous MERGE create-branch should still fire on empty store"
    );
}
