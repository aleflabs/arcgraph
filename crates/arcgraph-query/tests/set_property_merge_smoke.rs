//! ADR-150 W26-θ Phase 4 — SET property MERGE (`n += {props}`)
//! end-to-end smoke test.
//!
//! PropertyMerge additively overlays the new entries — entries NOT in
//! the new map are PRESERVED.

use arcgraph_core::{NodeId, PartitionId, TenantId};

use arcgraph_query::executor::substrate::StubExecutorSubstrate;
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
fn set_property_merge_parses_and_lowers() {
    let _ = lower("CREATE (n:User) SET n += {age: 25}");
}

#[test]
fn set_property_merge_round_trip_display() {
    let stmt = parse("CREATE (n:User) SET n += {age: 25}").expect("parse OK");
    let printed = format!("{stmt}");
    let reparsed = parse(&printed).expect("reparse OK");
    assert_eq!(stmt, reparsed, "Display round-trip failed: `{printed}`");
}

#[test]
fn set_property_merge_preserves_existing_entries() {
    let tenant = TenantId::DEFAULT;
    let s = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);

    // First write `name = "Alice"`.
    drain(&lower("CREATE (n:User) SET n.name = \"Alice\""), &s, &ctx);
    let bag_pre = s
        .node_properties(tenant, NodeId::new((1u64 << 32) + 1))
        .expect("pre-bag exists");
    assert_eq!(bag_pre.get("name"), Some(&Value::String("Alice".into())));

    // Now MERGE adds `age = 25` without clearing `name`.
    use arcgraph_query::ExecutorSubstrate;
    use arcgraph_query::executor::ExecutionContext;
    use arcgraph_query::executor::substrate::SetNodeMutation;
    let __ctx = ExecutionContext::new(tenant, arcgraph_core::PartitionId::ZERO);
    s.set_node(
        tenant,
        NodeId::new((1u64 << 32) + 1),
        &SetNodeMutation::PropertyMerge(vec![("age".into(), Value::Integer(25))]),
        &__ctx,
    )
    .expect("set_node merge OK");

    let bag_post = s
        .node_properties(tenant, NodeId::new((1u64 << 32) + 1))
        .expect("bag exists");
    assert_eq!(
        bag_post.get("name"),
        Some(&Value::String("Alice".into())),
        "PropertyMerge preserves prior entries"
    );
    assert_eq!(bag_post.get("age"), Some(&Value::Integer(25)));
}
