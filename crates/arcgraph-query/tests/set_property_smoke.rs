//! ADR-150 W26-θ Phase 4 — SET property assign / merge / replace
//! end-to-end smoke tests.
//!
//! Walks Parse → Bind → TypeCheck → CrossSubstrate → Lower → Pipeline
//! → execute against `StubExecutorSubstrate`. The Stub records the
//! post-SET property bag via its sidecar; assertions read the bag
//! back.

use arcgraph_core::{NodeId, PartitionId, TenantId};

use arcgraph_query::executor::substrate::StubExecutorSubstrate;
use arcgraph_query::executor::value::{NodeView, Value};
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

fn has_set(p: &LogicalPlan) -> bool {
    match p {
        LogicalPlan::Set(_) => true,
        LogicalPlan::Filter(f) => has_set(&f.input),
        LogicalPlan::Project(pr) => has_set(&pr.input),
        LogicalPlan::Limit(l) => has_set(&l.input),
        LogicalPlan::Skip(s) => has_set(&s.input),
        LogicalPlan::Join(j) => has_set(&j.left) || has_set(&j.right),
        LogicalPlan::LeftOuterJoin(j) => has_set(&j.left) || has_set(&j.right),
        LogicalPlan::Aggregate(a) => has_set(&a.input),
        LogicalPlan::Sort(s) => has_set(&s.input),
        LogicalPlan::Distinct(d) => has_set(&d.input),
        LogicalPlan::Unwind(u) => has_set(&u.input),
        LogicalPlan::ProcedureCall(p) => has_set(&p.input),
        LogicalPlan::NamedPath(np) => has_set(&np.input),
        LogicalPlan::DynamicLimit(l) => has_set(&l.input),
        LogicalPlan::CommunityLookup(c) => has_set(&c.input),
        LogicalPlan::Fusion(f) => f.inputs.iter().any(|inp| has_set(inp)),
        LogicalPlan::CreateRel(c) => has_set(&c.source_plan) || has_set(&c.target_plan),
        LogicalPlan::Delete(d) => has_set(&d.input),
        LogicalPlan::Remove(r) => has_set(&r.input),
        _ => false,
    }
}

#[test]
fn set_property_assign_parses_through_planner() {
    let plan = lower("CREATE (n:User) SET n.name = \"Alice\"");
    assert!(
        has_set(&plan),
        "expected LogicalPlan::Set in plan: {plan:?}"
    );
}

#[test]
fn set_property_assign_round_trip_display() {
    // Display roundtrips the per-key property write.
    let stmt = parse("CREATE (n:User) SET n.name = \"Alice\"").expect("parse OK");
    let printed = format!("{stmt}");
    let reparsed = parse(&printed).expect("reparse OK");
    assert_eq!(stmt, reparsed, "Display round-trip failed: `{printed}`");
}

#[test]
fn set_property_assign_routes_through_stub_substrate() {
    // CREATE introduces one node binding; SET writes a property on
    // that binding. We pre-bake the resulting NodeId in the Stub so
    // node_properties() can read it back.
    let tenant = TenantId::DEFAULT;
    let plan = lower("CREATE (n:User) SET n.name = \"Alice\"");
    let s = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);
    drain(&plan, &s, &ctx);

    // The CREATE allocated NodeId(2^32) (Stub allocator's high-water
    // mark). The Stub's set_node recorded the property bag.
    let bag = s
        .node_properties(tenant, NodeId::new((1u64 << 32) + 1))
        .expect("SET recorded a property bag");
    let v = bag.get("name").expect("name key present");
    assert_eq!(v, &Value::String("Alice".into()));
}

#[test]
fn set_property_assign_integer_value_works() {
    let tenant = TenantId::DEFAULT;
    let plan = lower("CREATE (n:User) SET n.age = 42");
    let s = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);
    drain(&plan, &s, &ctx);
    let bag = s
        .node_properties(tenant, NodeId::new((1u64 << 32) + 1))
        .expect("SET recorded a property bag");
    let v = bag.get("age").expect("age key present");
    assert_eq!(v, &Value::Integer(42));
}

#[test]
fn set_property_assign_match_then_set_routes_through_substrate() {
    // MATCH (n) SET n.name = "Bob" — the openCypher v9 §6 shape; no
    // label filter so the StubCatalogProvider doesn't need
    // to recognize a label name.
    let tenant = TenantId::DEFAULT;
    let plan = lower("MATCH (n) SET n.name = \"Bob\"");
    let pre = NodeView::new(NodeId::new(1), None);
    let s = StubExecutorSubstrate::new().with_node(tenant, pre);
    let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);
    drain(&plan, &s, &ctx);
    let bag = s
        .node_properties(tenant, NodeId::new(1))
        .expect("SET recorded a property bag");
    assert_eq!(bag.get("name"), Some(&Value::String("Bob".into())));
}

#[test]
fn set_multi_item_assign_routes_per_item() {
    // SET n.a = 1, n.b = 2 — two items, two substrate calls.
    let tenant = TenantId::DEFAULT;
    let plan = lower("CREATE (n:User) SET n.a = 1, n.b = 2");
    let s = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);
    drain(&plan, &s, &ctx);
    let bag = s
        .node_properties(tenant, NodeId::new((1u64 << 32) + 1))
        .expect("SET recorded a property bag");
    assert_eq!(bag.get("a"), Some(&Value::Integer(1)));
    assert_eq!(bag.get("b"), Some(&Value::Integer(2)));
}

#[test]
fn set_property_rejects_non_literal_value_at_type_check() {
    // Phase 4 inherits Phase 1's literal-only narrowing per ADR-150
    // §D-4. `SET n.x = $p` is rejected at type-check.
    let stmt = parse("CREATE (n:User) SET n.x = $p").expect("parse OK");
    let cat = StubCatalogProvider::new();
    let mut bound = BindingVisitor::bind(&stmt, "CREATE (n:User) SET n.x = $p", &cat)
        .expect("bind OK; binding pass is permissive on parameters");
    let r = TypeCheckVisitor::check(&mut bound, &cat);
    assert!(
        r.is_err(),
        "expected SET-property non-literal to surface at type-check"
    );
}
