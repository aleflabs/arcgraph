//! ADR-150 W26-θ Phase 4 — REMOVE n.prop end-to-end smoke test.

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
fn remove_property_parses_and_lowers() {
    let plan = lower("CREATE (n:User) REMOVE n.age");
    assert!(has_remove(&plan), "expected LogicalPlan::Remove: {plan:?}");
}

#[test]
fn remove_property_round_trip_display() {
    let stmt = parse("CREATE (n:User) REMOVE n.age").expect("parse OK");
    let printed = format!("{stmt}");
    let reparsed = parse(&printed).expect("reparse OK");
    assert_eq!(stmt, reparsed, "Display round-trip failed: `{printed}`");
}

#[test]
fn remove_property_clears_entry_in_stub_bag() {
    use arcgraph_query::ExecutorSubstrate;
    use arcgraph_query::executor::substrate::{RemoveNodeMutation, SetNodeMutation};
    let tenant = TenantId::DEFAULT;
    let s = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);

    // Run CREATE → REMOVE n.age through the planner. The CREATE
    // allocates NodeId((1<<32)+1); the REMOVE runs against that
    // same NodeId. We then directly populate the bag for that
    // NodeId (post-CREATE/post-REMOVE — these direct ops execute
    // AFTER the planner-driven REMOVE) and verify direct REMOVE
    // clears the entry.
    drain(&lower("CREATE (n:User) REMOVE n.age"), &s, &ctx);

    // After the planner-driven REMOVE, the bag is empty (or the
    // entry didn't exist). Direct populate + direct REMOVE
    // exercises the Stub's bookkeeping shape.
    let node = NodeId::new((1u64 << 32) + 1);
    let __ctx =
        arcgraph_query::executor::ExecutionContext::new(tenant, arcgraph_core::PartitionId::ZERO);
    s.set_node(
        tenant,
        node,
        &SetNodeMutation::PropertyAssign {
            name: "age".into(),
            value: Value::Integer(25),
        },
        &__ctx,
    )
    .expect("set_node OK");
    s.set_node(
        tenant,
        node,
        &SetNodeMutation::PropertyAssign {
            name: "name".into(),
            value: Value::String("Alice".into()),
        },
        &__ctx,
    )
    .expect("set_node OK");

    let bag_pre = s.node_properties(tenant, node).expect("pre-bag exists");
    assert_eq!(bag_pre.get("age"), Some(&Value::Integer(25)));

    s.remove_node(
        tenant,
        node,
        &RemoveNodeMutation::Property("age".into()),
        &__ctx,
    )
    .expect("remove_node OK");
    let bag_post = s.node_properties(tenant, node).expect("bag exists");
    assert_eq!(bag_post.get("age"), None, "REMOVE cleared the entry");
    assert_eq!(bag_post.get("name"), Some(&Value::String("Alice".into())));
}

fn has_remove(p: &LogicalPlan) -> bool {
    match p {
        LogicalPlan::Remove(_) => true,
        LogicalPlan::Filter(f) => has_remove(&f.input),
        LogicalPlan::Project(pr) => has_remove(&pr.input),
        LogicalPlan::Limit(l) => has_remove(&l.input),
        LogicalPlan::Skip(s) => has_remove(&s.input),
        LogicalPlan::Sort(s) => has_remove(&s.input),
        LogicalPlan::Distinct(d) => has_remove(&d.input),
        LogicalPlan::Unwind(u) => has_remove(&u.input),
        LogicalPlan::ProcedureCall(p) => has_remove(&p.input),
        LogicalPlan::Aggregate(a) => has_remove(&a.input),
        LogicalPlan::CommunityLookup(c) => has_remove(&c.input),
        LogicalPlan::NamedPath(np) => has_remove(&np.input),
        LogicalPlan::DynamicLimit(l) => has_remove(&l.input),
        LogicalPlan::Join(j) => has_remove(&j.left) || has_remove(&j.right),
        LogicalPlan::LeftOuterJoin(j) => has_remove(&j.left) || has_remove(&j.right),
        LogicalPlan::Fusion(f) => f.inputs.iter().any(|inp| has_remove(inp)),
        LogicalPlan::CreateRel(c) => has_remove(&c.source_plan) || has_remove(&c.target_plan),
        LogicalPlan::Delete(d) => has_remove(&d.input),
        LogicalPlan::Set(s) => has_remove(&s.input),
        _ => false,
    }
}
