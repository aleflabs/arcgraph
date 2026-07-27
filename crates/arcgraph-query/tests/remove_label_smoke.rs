//! ADR-150 W26-θ Phase 4 — REMOVE n:Label end-to-end smoke test.
//!
//! Mirrors `set_label_smoke.rs` — production label mutation is
//! forward-pinned to v1.1 per ADR-150 §D-9; the Stub substrate
//! exercises the in-memory sidecar.

use arcgraph_core::{NodeId, PartitionId, TenantId};

use arcgraph_query::executor::substrate::StubExecutorSubstrate;
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
fn remove_label_parses_and_lowers() {
    let _ = lower("CREATE (n:User) REMOVE n:VIP");
}

#[test]
fn remove_label_round_trip_display() {
    let stmt = parse("CREATE (n:User) REMOVE n:VIP").expect("parse OK");
    let printed = format!("{stmt}");
    let reparsed = parse(&printed).expect("reparse OK");
    assert_eq!(stmt, reparsed, "Display round-trip failed: `{printed}`");
}

#[test]
fn remove_label_clears_sidecar_entry() {
    let tenant = TenantId::DEFAULT;
    let s = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);

    // Pre-populate the sidecar with two labels via direct substrate
    // call (we can't do CREATE → SET label → REMOVE label as a
    // single statement in Phase 4 because we don't yet ship CREATE +
    // SET multi-clause via the planner pre-MATCH-link; we use direct
    // substrate ops to populate state).
    use arcgraph_query::ExecutorSubstrate;
    use arcgraph_query::executor::substrate::SetNodeMutation;
    let node = NodeId::new((1u64 << 32) + 1);
    s.set_node(
        tenant,
        node,
        &SetNodeMutation::LabelAdd(vec!["VIP".into(), "Premium".into()]),
        &arcgraph_query::executor::ExecutionContext::new(tenant, arcgraph_core::PartitionId::ZERO),
    )
    .expect("set_node label-add OK");
    assert_eq!(
        s.additional_labels(tenant, node),
        vec!["VIP".to_string(), "Premium".to_string()]
    );

    // Now run a CREATE (m) REMOVE m:VIP — exercises the planner +
    // executor wire-through against a DIFFERENT node. Orthogonal to
    // our pre-populated state; serves as the end-to-end pin.
    drain(&lower("CREATE (m:User) REMOVE m:VIP"), &s, &ctx);
    // The CREATE'd second node (id (1<<32)+2) had no labels in the
    // sidecar, so REMOVE is a no-op there.
    let other_labels = s.additional_labels(tenant, NodeId::new((1u64 << 32) + 2));
    assert!(other_labels.is_empty());

    // Now exercise direct remove on the pre-populated node.
    use arcgraph_query::executor::substrate::RemoveNodeMutation;
    s.remove_node(
        tenant,
        node,
        &RemoveNodeMutation::LabelRemove(vec!["VIP".into()]),
        &arcgraph_query::executor::ExecutionContext::new(tenant, arcgraph_core::PartitionId::ZERO),
    )
    .expect("remove_node label-remove OK");
    let after = s.additional_labels(tenant, node);
    assert_eq!(
        after,
        vec!["Premium".to_string()],
        "VIP removed, Premium retained"
    );
}
