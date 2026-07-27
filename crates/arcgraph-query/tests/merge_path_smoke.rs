//! ADR-151 W26-θ Phase 5 — MERGE path-shape create-branch smoke test.
//!
//! `MERGE (a:User {id:1})-[r:FOLLOWS]->(b:User {id:2})` on an empty
//! store: the match-branch's Scan + Expand + Scan chain probes
//! (returns 0 rows); the create-branch (CreateRel wrapping CreateNode
//! source + CreateNode target) fires → source node, target node, and
//! rel are all created atomically within the executor's composition.

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
fn merge_path_parses_and_lowers() {
    let plan = lower("MERGE (a:User {id: 1})-[r:FOLLOWS]->(b:User {id: 2})");
    assert!(
        matches!(&plan, LogicalPlan::Merge(_)),
        "expected LogicalPlan::Merge at top: {plan:?}"
    );
}

#[test]
fn merge_path_round_trip_display() {
    let stmt = parse("MERGE (a:User {id: 1})-[r:FOLLOWS]->(b:User {id: 2})").expect("parse OK");
    let printed = format!("{stmt}");
    let reparsed = parse(&printed).expect("reparse OK");
    assert_eq!(stmt, reparsed, "Display round-trip failed: `{printed}`");
}

#[test]
fn merge_path_create_branch_fires_on_empty_store() {
    let tenant = TenantId::DEFAULT;
    let plan = lower("MERGE (a:User {id: 1})-[r:FOLLOWS]->(b:User {id: 2})");
    let s = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);
    drain(&plan, &s, &ctx);
    // The CreateRel chain creates 2 nodes + 1 rel.
    let nodes = s.scan_nodes(tenant, None, Lsn::MAX).expect("scan_nodes OK");
    assert_eq!(
        nodes.len(),
        2,
        "path create-branch should have created 2 nodes (source + target)"
    );
}

#[test]
fn merge_path_no_action_smoke() {
    // Path-shape MERGE without action clauses — just exercise the
    // create-branch path through CreateRel.
    let tenant = TenantId::DEFAULT;
    let plan = lower("MERGE (a:User)-[r:KNOWS]->(b:User)");
    let s = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);
    drain(&plan, &s, &ctx);
    let nodes = s.scan_nodes(tenant, None, Lsn::MAX).expect("scan_nodes OK");
    // The Phase 5 v1.0-α narrowing per ADR-151 §D-6 (match-branch
    // scans with label=None) means MERGE-on-empty-store always
    // creates; the Path-shape Stub-side scan returns 0 rows on empty
    // store, so the create-branch fires + makes 2 nodes.
    assert_eq!(nodes.len(), 2, "path create-branch created source + target");
}
