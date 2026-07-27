//! ADR-150 W26-θ Phase 4 — SET property REPLACE (`n = {props}`)
//! end-to-end smoke test.
//!
//! PropertyReplace overwrites the full property bag — entries NOT in
//! the new map are CLEARED.

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
fn set_property_replace_parses_and_lowers() {
    let _ = lower("CREATE (n:User) SET n = {name: \"Bob\", age: 30}");
}

#[test]
fn set_property_replace_round_trip_display() {
    let stmt = parse("CREATE (n:User) SET n = {name: \"Bob\", age: 30}").expect("parse OK");
    let printed = format!("{stmt}");
    let reparsed = parse(&printed).expect("reparse OK");
    assert_eq!(stmt, reparsed, "Display round-trip failed: `{printed}`");
}

#[test]
fn set_property_replace_overwrites_full_bag() {
    let tenant = TenantId::DEFAULT;
    // First write a property with assign, then replace with a fresh
    // bag — observe that the replace clears the prior assign's entry.
    let s = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);

    drain(&lower("CREATE (n:User) SET n.old = 1"), &s, &ctx);
    // After the first execution, NodeId((1<<32)+1) has `old = 1`.
    let bag_before = s
        .node_properties(tenant, NodeId::new((1u64 << 32) + 1))
        .expect("pre-bag exists");
    assert_eq!(bag_before.get("old"), Some(&Value::Integer(1)));

    // Run another query that reaches the same NodeId via direct
    // substrate call. We bypass the executor for clarity: invoke
    // set_node directly with PropertyReplace to validate the
    // semantic.
    use arcgraph_query::ExecutorSubstrate;
    use arcgraph_query::executor::ExecutionContext;
    use arcgraph_query::executor::substrate::SetNodeMutation;
    let __ctx = ExecutionContext::new(tenant, arcgraph_core::PartitionId::ZERO);
    s.set_node(
        tenant,
        NodeId::new((1u64 << 32) + 1),
        &SetNodeMutation::PropertyReplace(vec![
            ("name".into(), Value::String("Bob".into())),
            ("age".into(), Value::Integer(30)),
        ]),
        &__ctx,
    )
    .expect("set_node replace OK");

    let bag_after = s
        .node_properties(tenant, NodeId::new((1u64 << 32) + 1))
        .expect("bag exists");
    assert_eq!(bag_after.get("name"), Some(&Value::String("Bob".into())));
    assert_eq!(bag_after.get("age"), Some(&Value::Integer(30)));
    assert_eq!(
        bag_after.get("old"),
        None,
        "PropertyReplace clears prior entries"
    );
}
