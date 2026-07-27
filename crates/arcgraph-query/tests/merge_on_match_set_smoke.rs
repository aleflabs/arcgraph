//! ADR-151 W26-θ Phase 5 — MERGE ON MATCH SET smoke test.
//!
//! Pre-bake a `:User` node; then `MERGE (n:User) ON MATCH SET
//! n.last_seen = "today"` — the match-branch's Scan finds the
//! pre-baked node → match-branch fires → ON MATCH action fires →
//! Stub records `n.last_seen = "today"` in the pre-baked node's
//! property bag. No new node is created.
//!
//! ADR-152-amendment-01 update: the match-branch now ENFORCES the
//! pattern label (§D-2), so the pre-baked node's `LabelId` is aligned
//! with the catalog's `User` id (`USER_LABEL`) for the match to fire.

use arcgraph_core::{LabelId, Lsn, NodeId, PartitionId, TenantId};

use arcgraph_query::executor::substrate::{ExecutorSubstrate, StubExecutorSubstrate};
use arcgraph_query::executor::value::{NodeView, Value};
use arcgraph_query::executor::{ExecutionContext, Pipeline};
use arcgraph_query::logical_plan::{LogicalPlan, LogicalPlanLoweringVisitor};
use arcgraph_query::parse;
use arcgraph_query::semantic::{
    BindingVisitor, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};

/// Interned `LabelId` for `User` — see `merge_node_match_branch_smoke`.
const USER_LABEL: u32 = 1024;

fn lower(query: &str) -> LogicalPlan {
    let stmt = parse(query).expect("parse OK");
    // ADR-152-amendment-01 §D-1 — `User` interned so the MERGE
    // match-branch lowers to `Scan{label: Some(USER_LABEL)}`.
    let cat = StubCatalogProvider::new().with_label_id("User", LabelId::new(USER_LABEL));
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
fn merge_on_match_set_round_trip_display() {
    let stmt = parse("MERGE (n:User) ON MATCH SET n.last_seen = \"today\"").expect("parse OK");
    let printed = format!("{stmt}");
    let reparsed = parse(&printed).expect("reparse OK");
    assert_eq!(stmt, reparsed, "Display round-trip failed: `{printed}`");
}

#[test]
fn merge_on_match_set_fires_on_match_branch() {
    let tenant = TenantId::DEFAULT;
    let pre_id = NodeId::new(1);
    let pre = NodeView::new(pre_id, Some(LabelId::new(USER_LABEL)));
    let s = StubExecutorSubstrate::new().with_node(tenant, pre);
    let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);
    let plan = lower("MERGE (n:User) ON MATCH SET n.last_seen = \"today\"");
    drain(&plan, &s, &ctx);

    // Pre-baked node id=1 should have the on_match-set bag.
    let bag = s
        .node_properties(tenant, pre_id)
        .expect("ON MATCH SET should have recorded a property bag on the pre-baked node");
    assert_eq!(
        bag.get("last_seen"),
        Some(&Value::String("today".into())),
        "ON MATCH SET should have set `last_seen` to \"today\""
    );

    // No new node was created.
    let nodes = s.scan_nodes(tenant, None, Lsn::MAX).unwrap();
    assert_eq!(nodes.len(), 1, "match-branch fired; no duplicate created");
}

#[test]
fn merge_on_match_does_not_fire_on_create_branch() {
    // On an EMPTY store, the create-branch fires; ON MATCH actions
    // are NOT fired. The new node's property bag should be empty (no
    // last_seen entry).
    let tenant = TenantId::DEFAULT;
    let plan = lower("MERGE (n:User) ON MATCH SET n.last_seen = \"today\"");
    let s = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);
    drain(&plan, &s, &ctx);
    let node_id = NodeId::new((1u64 << 32) + 1);
    // ON MATCH was not fired → no property bag was set on the new
    // node (or it's set but doesn't contain last_seen).
    let bag = s.node_properties(tenant, node_id);
    assert!(
        bag.as_ref().and_then(|b| b.get("last_seen")).is_none(),
        "ON MATCH should NOT have fired on the create-branch row; bag={bag:?}"
    );
}

#[test]
fn merge_with_both_on_create_and_on_match_fires_correct_branch() {
    // Test the full shape: MERGE + ON CREATE + ON MATCH. On empty
    // store, ON CREATE fires; ON MATCH does not.
    let tenant = TenantId::DEFAULT;
    let plan = lower("MERGE (n:User) ON CREATE SET n.name = \"new\" ON MATCH SET n.name = \"old\"");
    let s = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);
    drain(&plan, &s, &ctx);
    let node_id = NodeId::new((1u64 << 32) + 1);
    let bag = s
        .node_properties(tenant, node_id)
        .expect("on_create fired; bag recorded");
    assert_eq!(
        bag.get("name"),
        Some(&Value::String("new".into())),
        "ON CREATE branch — `name` should be \"new\""
    );
}
