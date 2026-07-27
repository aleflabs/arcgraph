//! ADR-150 W26-θ Phase 4 — SET label-add end-to-end smoke test.
//!
//! Label mutations on the Stub substrate track an additional-labels
//! sidecar (per-(tenant, NodeId) Vec<String>); the production
//! substrate forward-pins label mutation to v1.1 per ADR-150 §D-9
//! (the storage `update_node` primitive preserves `label_id`
//! immutably per `crud.rs:3754` "PR #170 reviewer Finding 4"). The
//! Stub-side coverage exercises the AST → bind → typecheck → lower →
//! executor path end-to-end at v1.0-α.

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
fn set_label_add_parses_and_lowers() {
    let plan = lower("CREATE (n:User) SET n:VIP");
    assert!(
        matches!(
            find_set_first(&plan),
            Some(arcgraph_query::logical_plan::LogicalSetMutation::LabelAdd(
                _
            ))
        ),
        "expected LogicalSetMutation::LabelAdd in plan: {plan:?}"
    );
}

#[test]
fn set_label_add_round_trip_display() {
    let stmt = parse("CREATE (n:User) SET n:VIP").expect("parse OK");
    let printed = format!("{stmt}");
    let reparsed = parse(&printed).expect("reparse OK");
    assert_eq!(stmt, reparsed, "Display round-trip failed: `{printed}`");
}

#[test]
fn set_multi_label_add_round_trip_display() {
    let stmt = parse("CREATE (n:User) SET n:VIP:Premium").expect("parse OK");
    let printed = format!("{stmt}");
    let reparsed = parse(&printed).expect("reparse OK");
    assert_eq!(stmt, reparsed, "Display round-trip failed: `{printed}`");
}

#[test]
fn set_label_add_routes_through_stub_substrate_sidecar() {
    let tenant = TenantId::DEFAULT;
    let plan = lower("CREATE (n:User) SET n:VIP");
    let s = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);
    drain(&plan, &s, &ctx);
    let labels = s.additional_labels(tenant, NodeId::new((1u64 << 32) + 1));
    assert_eq!(labels, vec!["VIP".to_string()]);
}

#[test]
fn set_multi_label_add_routes_through_stub_substrate_sidecar() {
    let tenant = TenantId::DEFAULT;
    let plan = lower("CREATE (n:User) SET n:VIP:Premium");
    let s = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);
    drain(&plan, &s, &ctx);
    let labels = s.additional_labels(tenant, NodeId::new((1u64 << 32) + 1));
    assert_eq!(labels, vec!["VIP".to_string(), "Premium".to_string()]);
}

fn find_set_first(p: &LogicalPlan) -> Option<arcgraph_query::logical_plan::LogicalSetMutation> {
    match p {
        LogicalPlan::Set(s) => s.items.first().map(|item| item.mutation.clone()),
        LogicalPlan::Filter(f) => find_set_first(&f.input),
        LogicalPlan::Project(pr) => find_set_first(&pr.input),
        LogicalPlan::Limit(l) => find_set_first(&l.input),
        LogicalPlan::Skip(s) => find_set_first(&s.input),
        LogicalPlan::Sort(s) => find_set_first(&s.input),
        LogicalPlan::Distinct(d) => find_set_first(&d.input),
        LogicalPlan::Unwind(u) => find_set_first(&u.input),
        LogicalPlan::ProcedureCall(p) => find_set_first(&p.input),
        LogicalPlan::Aggregate(a) => find_set_first(&a.input),
        LogicalPlan::CommunityLookup(c) => find_set_first(&c.input),
        LogicalPlan::NamedPath(np) => find_set_first(&np.input),
        LogicalPlan::DynamicLimit(l) => find_set_first(&l.input),
        LogicalPlan::Join(j) => find_set_first(&j.left).or_else(|| find_set_first(&j.right)),
        LogicalPlan::LeftOuterJoin(j) => {
            find_set_first(&j.left).or_else(|| find_set_first(&j.right))
        }
        LogicalPlan::Fusion(f) => f.inputs.iter().find_map(|inp| find_set_first(inp)),
        LogicalPlan::CreateRel(c) => {
            find_set_first(&c.source_plan).or_else(|| find_set_first(&c.target_plan))
        }
        LogicalPlan::Delete(d) => find_set_first(&d.input),
        LogicalPlan::Remove(r) => find_set_first(&r.input),
        _ => None,
    }
}
