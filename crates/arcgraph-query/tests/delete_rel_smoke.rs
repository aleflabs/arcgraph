//! ADR-149 W26-θ Phase 3 — DELETE rel end-to-end smoke test.
//!
//! Pins the DELETE-rel sub-shape: `CREATE (a)-[r:KNOWS]->(b) DELETE r`.
//! The DELETE clause resolves `r` against the prior CREATE-rel's
//! introduced binding (the CreateRelOp produces ONE row carrying the
//! new rel binding); the DeleteOp consumes that row + dispatches to
//! `substrate.delete_rel`. The Stub's tombstoned_rels set filters the
//! rel from subsequent `expand` calls.
//!
//! At Phase 3 the rel-DELETE shape DOES NOT trigger any DETACH walk
//! (rels have no attached children at the storage layer at v1.0-α
//! per ADR-149 §D-7). The `detach` flag is grammar-admissible on a
//! rel-only DELETE but is semantically a no-op at the substrate
//! layer — the executor consumes it for the dispatch but the
//! `delete_rel` substrate method does not branch on it.

use arcgraph_core::{NodeId, PartitionId, RelId, TenantId, TypeId};

use arcgraph_query::ExecutorSubstrate;
use arcgraph_query::executor::substrate::StubExecutorSubstrate;
use arcgraph_query::executor::value::RelView;
use arcgraph_query::executor::{ExecutionContext, value::NodeView};
use arcgraph_query::logical_plan::{LogicalPlan, LogicalPlanLoweringVisitor};
use arcgraph_query::semantic::{
    BindingVisitor, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};
use arcgraph_query::{Statement, executor::Pipeline, parse};

fn lower(query: &str) -> LogicalPlan {
    let stmt = parse(query).expect("parse OK");
    let inner = match stmt {
        Statement::Read(_) => stmt,
        other => panic!("expected Read statement, got {other:?}"),
    };
    let cat = StubCatalogProvider::new();
    let mut bound = BindingVisitor::bind(&inner, query, &cat).expect("bind OK");
    TypeCheckVisitor::check(&mut bound, &cat).expect("type-check OK");
    CrossSubstrateValidator::validate(&bound, &cat).expect("cross-substrate OK");
    LogicalPlanLoweringVisitor::lower(&bound).expect("lower OK")
}

#[test]
fn delete_rel_parses_through_planner() {
    let plan = lower("CREATE (a:User)-[r:KNOWS]->(b:User) DELETE r");
    assert!(has_delete(&plan), "expected Delete in plan: {plan:?}");
}

#[test]
fn delete_rel_round_trip_display() {
    let original = "CREATE (a:User)-[r:KNOWS]->(b:User) DELETE r";
    let parsed = parse(original).expect("parse OK");
    let printed = format!("{parsed}");
    let re_parsed = parse(&printed).expect("re-parse OK");
    assert_eq!(parsed, re_parsed, "Display round-trips");
}

#[test]
fn delete_rel_pre_existing_rel_tombstones_via_stub() {
    // Pre-bake nodes + a rel via `with_node` + `with_edge`, then
    // call `delete_rel` directly; subsequent `expand` returns 0.
    let tenant = TenantId::DEFAULT;
    let lbl = arcgraph_core::LabelId::new(1024);
    let n1 = NodeView::new(NodeId::new(1), Some(lbl));
    let n2 = NodeView::new(NodeId::new(2), Some(lbl));
    let rel = RelView::new(RelId::new(100), n1.id, n2.id, Some(TypeId::new(1024)));
    let s = StubExecutorSubstrate::new()
        .with_node(tenant, n1.clone())
        .with_node(tenant, n2.clone())
        .with_edge(tenant, rel.clone());
    // Verify the rel is visible.
    let pre_expand = s
        .expand(
            tenant,
            n1.id,
            None,
            arcgraph_query::logical_plan::Direction::LeftToRight,
            arcgraph_core::Lsn::MAX,
        )
        .expect("expand OK");
    assert_eq!(pre_expand.len(), 1, "pre-bake rel is visible");
    // Tombstone.
    s.delete_rel(
        tenant,
        rel.id,
        &arcgraph_query::executor::ExecutionContext::new(tenant, arcgraph_core::PartitionId::ZERO),
    )
    .expect("delete_rel OK");
    // Rel no longer visible.
    let post_expand = s
        .expand(
            tenant,
            n1.id,
            None,
            arcgraph_query::logical_plan::Direction::LeftToRight,
            arcgraph_core::Lsn::MAX,
        )
        .expect("expand OK");
    assert_eq!(
        post_expand.len(),
        0,
        "rel filtered post-delete: {post_expand:?}"
    );
}

#[test]
fn delete_rel_executes_end_to_end_against_stub() {
    // End-to-end: `CREATE ... DELETE r` tombstones the just-created rel.
    let plan = lower("CREATE (a:User)-[r:KNOWS]->(b:User) DELETE r");
    let s = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let mut op = Pipeline::build(&plan).expect("pipeline build OK");
    loop {
        let b = op.next_batch(&ctx, &s).expect("batch OK");
        if b.is_empty() {
            break;
        }
    }
    // Both nodes were created and survive; the rel was created then
    // tombstoned, so `expand` from a → ... yields 0.
    let nodes = s
        .scan_nodes(TenantId::DEFAULT, None, arcgraph_core::Lsn::MAX)
        .expect("scan_nodes OK");
    assert_eq!(
        nodes.len(),
        2,
        "both endpoints persist post-DELETE-rel: {nodes:?}"
    );
    // Expand from a (the CREATE-introduced source) yields 0 because
    // the rel is tombstoned.
    let a_id = nodes
        .iter()
        .map(|b| b.node.id)
        .min()
        .expect("at least one node");
    let edges = s
        .expand(
            TenantId::DEFAULT,
            a_id,
            None,
            arcgraph_query::logical_plan::Direction::LeftToRight,
            arcgraph_core::Lsn::MAX,
        )
        .expect("expand OK");
    assert_eq!(edges.len(), 0, "rel was tombstoned: {edges:?}");
}

fn has_delete(plan: &LogicalPlan) -> bool {
    match plan {
        LogicalPlan::Delete(_) => true,
        LogicalPlan::Filter(f) => has_delete(&f.input),
        LogicalPlan::Project(p) => has_delete(&p.input),
        LogicalPlan::Limit(l) => has_delete(&l.input),
        LogicalPlan::Skip(s) => has_delete(&s.input),
        LogicalPlan::DynamicLimit(d) => has_delete(&d.input),
        LogicalPlan::Sort(s) => has_delete(&s.input),
        LogicalPlan::Distinct(d) => has_delete(&d.input),
        LogicalPlan::Unwind(u) => has_delete(&u.input),
        LogicalPlan::ProcedureCall(p) => has_delete(&p.input),
        LogicalPlan::Aggregate(a) => has_delete(&a.input),
        LogicalPlan::CommunityLookup(c) => has_delete(&c.input),
        LogicalPlan::NamedPath(np) => has_delete(&np.input),
        LogicalPlan::Join(j) => has_delete(&j.left) || has_delete(&j.right),
        LogicalPlan::LeftOuterJoin(j) => has_delete(&j.left) || has_delete(&j.right),
        LogicalPlan::Fusion(f) => f.inputs.iter().any(|inp| has_delete(inp)),
        LogicalPlan::Union(u) => u.arms.iter().any(has_delete),
        LogicalPlan::CreateRel(c) => has_delete(&c.source_plan) || has_delete(&c.target_plan),
        LogicalPlan::Set(s) => has_delete(&s.input),
        LogicalPlan::Remove(r) => has_delete(&r.input),
        LogicalPlan::Merge(m) => has_delete(&m.match_branch) || has_delete(&m.create_branch),
        LogicalPlan::Scan(_)
        | LogicalPlan::PropertyIndexScan(_)
        | LogicalPlan::CountStore(_)
        | LogicalPlan::Expand(_)
        | LogicalPlan::Empty(_)
        | LogicalPlan::RankByHybrid(_)
        | LogicalPlan::VectorNear(_)
        | LogicalPlan::TextMatch(_)
        | LogicalPlan::CreateNode(_)
        | LogicalPlan::CreateVectorIndex(_)
        | LogicalPlan::CreatePropertyIndex(_)
        | LogicalPlan::Call(_)
        | LogicalPlan::CorrelationSeed(_) => false,
    }
}
