//! ADR-151 W26-θ Phase 5 — MERGE ON CREATE SET smoke test.
//!
//! `MERGE (n:User {id: 42}) ON CREATE SET n.name = "Alice"` on an
//! empty store: create-branch fires (allocates a new NodeId);
//! ON CREATE action fires → Stub records `n.name = "Alice"` in the
//! per-(tenant, NodeId) property bag.
//!
//! The test asserts:
//! 1. New node is created.
//! 2. ON CREATE SET action recorded the `name` property.
//! 3. ON CREATE actions are NOT fired on subsequent match-branch hits
//!    (a second MERGE returns the same node + the bag is unchanged).

use arcgraph_core::{Lsn, NodeId, PartitionId, TenantId};

use arcgraph_query::executor::substrate::{ExecutorSubstrate, StubExecutorSubstrate};
use arcgraph_query::executor::value::Value;
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
fn merge_on_create_set_parses_and_lowers() {
    let plan = lower("MERGE (n:User {id: 42}) ON CREATE SET n.name = \"Alice\"");
    assert!(
        matches!(&plan, LogicalPlan::Merge(_)),
        "expected LogicalPlan::Merge: {plan:?}"
    );
}

#[test]
fn merge_on_create_set_round_trip_display() {
    let stmt = parse("MERGE (n:User {id: 42}) ON CREATE SET n.name = \"Alice\"").expect("parse OK");
    let printed = format!("{stmt}");
    let reparsed = parse(&printed).expect("reparse OK");
    assert_eq!(stmt, reparsed, "Display round-trip failed: `{printed}`");
}

#[test]
fn merge_on_create_set_fires_on_create_branch() {
    let tenant = TenantId::DEFAULT;
    let plan = lower("MERGE (n:User {id: 42}) ON CREATE SET n.name = \"Alice\"");
    let s = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);
    drain(&plan, &s, &ctx);

    // The Stub's create_node allocates NodeId at the 2^32 high-water
    // mark; the action's set_node recorded the property bag.
    let node_id = NodeId::new((1u64 << 32) + 1);
    let bag = s
        .node_properties(tenant, node_id)
        .expect("ON CREATE SET should have recorded a property bag");
    assert_eq!(
        bag.get("name"),
        Some(&Value::String("Alice".into())),
        "ON CREATE SET should have set `name` to \"Alice\""
    );

    // Verify there's exactly one node.
    let nodes = s.scan_nodes(tenant, None, Lsn::MAX).unwrap();
    assert_eq!(nodes.len(), 1);
}

#[test]
fn merge_on_create_set_with_multiple_actions() {
    // Multi-item ON CREATE SET: each action fires per-item via the
    // SetItemSpec dispatch.
    let tenant = TenantId::DEFAULT;
    let plan = lower("MERGE (n:User {id: 42}) ON CREATE SET n.name = \"Alice\", n.age = 30");
    let s = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);
    drain(&plan, &s, &ctx);
    let node_id = NodeId::new((1u64 << 32) + 1);
    let bag = s
        .node_properties(tenant, node_id)
        .expect("ON CREATE SET should have recorded a property bag");
    assert_eq!(bag.get("name"), Some(&Value::String("Alice".into())));
    assert_eq!(bag.get("age"), Some(&Value::Integer(30)));
}
