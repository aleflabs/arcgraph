//! ADR-149 W26-θ Phase 3 — DELETE node end-to-end smoke test.
//!
//! Walks the full query-side pipeline:
//!
//! 1. Parse `CREATE (n:User) DELETE n` to a `Statement`.
//! 2. Bind via `BindingVisitor::bind` against a `StubCatalogProvider`.
//! 3. Type-check via `TypeCheckVisitor::check`.
//! 4. Cross-substrate validate via `CrossSubstrateValidator::validate`.
//! 5. Lower to a `LogicalPlan` via `LogicalPlanLoweringVisitor::lower`.
//! 6. Build a `Pipeline` and execute against a `StubExecutorSubstrate`,
//!    asserting the node is tombstoned (post-execute `scan_nodes`
//!    sees zero rows).
//!
//! Note: The single-statement form `CREATE (n:User) DELETE n` is
//! ADMITTED by the parser per the W26-θ Phase 3 framework — the
//! DELETE clause resolves `n` against the prior CREATE's
//! introduced binding (a CREATE clause produces ONE binding row,
//! the DELETE consumes it). MATCH→DELETE (`MATCH (n:User) DELETE n`)
//! is the more typical openCypher v9 §6 shape and is the canonical
//! pin in `tests/delete_*_smoke.rs`.

use arcgraph_core::{PartitionId, TenantId};

use arcgraph_query::ExecutorSubstrate;
use arcgraph_query::executor::substrate::StubExecutorSubstrate;
use arcgraph_query::executor::{ExecutionContext, value::NodeView};
use arcgraph_query::logical_plan::{LogicalPlan, LogicalPlanLoweringVisitor};
use arcgraph_query::semantic::{
    BindingVisitor, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};
use arcgraph_query::{Statement, executor::Pipeline, parse};

/// Walk Parse → Bind → TypeCheck → CrossSubstrate → Lower for a single
/// query string + a fresh `StubCatalogProvider`. Returns the lowered
/// plan (asserts success at every stage).
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
fn delete_node_parses_through_planner() {
    // ADR-149 Phase 3 happy path: CREATE-then-DELETE-node.
    let plan = lower("CREATE (n:User) DELETE n");
    assert!(
        has_delete(&plan),
        "expected LogicalPlan::Delete in plan: {plan:?}"
    );
}

#[test]
fn detach_delete_parses_through_planner() {
    let plan = lower("CREATE (n:User) DETACH DELETE n");
    assert!(
        has_delete(&plan),
        "expected LogicalPlan::Delete in plan: {plan:?}"
    );
    // DETACH flag is preserved.
    assert!(plan_detach_flag(&plan), "expected detach=true in plan");
}

#[test]
fn delete_node_lowers_with_detach_false() {
    let plan = lower("CREATE (n:User) DELETE n");
    assert!(
        !plan_detach_flag(&plan),
        "expected detach=false for bare DELETE"
    );
}

#[test]
fn delete_node_executes_against_stub_substrate_tombstones_node() {
    // End-to-end: CREATE then DELETE leaves the substrate with zero
    // CREATE-introduced nodes (tombstones filter the just-created
    // entry).
    let plan = lower("CREATE (n:User) DELETE n");
    let s = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let mut op = Pipeline::build(&plan).expect("pipeline build OK");
    // Consume the pipeline; the DELETE operator pulls the
    // CREATE-produced row, dispatches to delete_node, then settles
    // into EOS.
    loop {
        let b = op.next_batch(&ctx, &s).expect("batch OK");
        if b.is_empty() {
            break;
        }
    }
    // Substrate now has 0 nodes (the CREATE was undone by the
    // DELETE).
    let nodes = s
        .scan_nodes(TenantId::DEFAULT, None, arcgraph_core::Lsn::MAX)
        .expect("scan_nodes OK");
    assert_eq!(
        nodes.len(),
        0,
        "scan_nodes observes 0 nodes post-DELETE: {nodes:?}"
    );
}

#[test]
fn detach_delete_round_trip_display_preserves_detach() {
    let original = "CREATE (n:User) DETACH DELETE n";
    let parsed = parse(original).expect("parse OK");
    let printed = format!("{parsed}");
    let re_parsed = parse(&printed).expect("re-parse OK");
    assert_eq!(parsed, re_parsed, "Display round-trips");
    assert!(
        printed.contains("DETACH DELETE"),
        "Display preserves DETACH prefix: {printed}"
    );
}

#[test]
fn delete_node_grammar_admits_multi_item() {
    // Multi-item DELETE: parse-only smoke. The semantics + executor
    // dispatch are pinned by other tests; the grammar should admit
    // `DELETE n, m`.
    let stmt = parse("CREATE (n:User), (m:Admin) DELETE n, m").expect("multi-item DELETE parses");
    let cat = StubCatalogProvider::new();
    let _ = BindingVisitor::bind(&stmt, "...", &cat).expect("bind");
}

#[test]
fn delete_node_rejects_undeclared_variable_at_binding() {
    // ADR-149 §D-3: DELETE items RESOLVE against prior bindings.
    // An undeclared variable surfaces BindingError::UndeclaredVariable.
    let stmt = parse("DELETE x").expect("parse OK (grammar admits)");
    let cat = StubCatalogProvider::new();
    let result = BindingVisitor::bind(&stmt, "DELETE x", &cat);
    assert!(
        result.is_err(),
        "expected bind error for undeclared `x` in DELETE: {result:?}"
    );
}

#[test]
fn delete_node_pre_existing_node_with_match_tombstones() {
    // Pre-bake a node via `with_node`, then run
    // `MATCH (n:User) DELETE n` — the substrate's tombstone set
    // filters the node from subsequent scans.
    //
    // We don't go through the planner for this test (MATCH-side
    // lowering for non-CREATE patterns requires a full
    // CatalogProvider setup); we exercise the Stub's delete_node
    // directly to pin the tombstone-filter contract.
    let tenant = TenantId::DEFAULT;
    let label = arcgraph_core::LabelId::new(1024);
    let pre = NodeView::new(arcgraph_core::NodeId::new(7), Some(label));
    let s = StubExecutorSubstrate::new().with_node(tenant, pre.clone());
    // Verify the node is initially visible.
    let pre_nodes = s
        .scan_nodes(tenant, None, arcgraph_core::Lsn::MAX)
        .expect("scan_nodes OK");
    assert_eq!(pre_nodes.len(), 1, "pre-bake node initially visible");
    // Tombstone via the Stub's delete_node.
    s.delete_node(
        tenant,
        pre.id,
        false,
        &arcgraph_query::executor::ExecutionContext::new(tenant, arcgraph_core::PartitionId::ZERO),
    )
    .expect("delete_node OK");
    // The node is no longer visible.
    let post_nodes = s
        .scan_nodes(tenant, None, arcgraph_core::Lsn::MAX)
        .expect("scan_nodes OK");
    assert_eq!(
        post_nodes.len(),
        0,
        "pre-bake node filtered post-delete: {post_nodes:?}"
    );
}

/// Recursively search a LogicalPlan tree for a Delete variant.
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
        // ADR-150 W26-θ Phase 4: Set / Remove walk the input sub-plan.
        LogicalPlan::Set(s) => has_delete(&s.input),
        LogicalPlan::Remove(r) => has_delete(&r.input),
        // ADR-151 W26-θ Phase 5: Merge walks both branches.
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

/// Return the `detach` flag of the first LogicalPlan::Delete found in
/// the tree.
fn plan_detach_flag(plan: &LogicalPlan) -> bool {
    match plan {
        LogicalPlan::Delete(d) => d.detach,
        LogicalPlan::Filter(f) => plan_detach_flag(&f.input),
        LogicalPlan::Project(p) => plan_detach_flag(&p.input),
        LogicalPlan::Limit(l) => plan_detach_flag(&l.input),
        LogicalPlan::Skip(s) => plan_detach_flag(&s.input),
        LogicalPlan::DynamicLimit(d) => plan_detach_flag(&d.input),
        LogicalPlan::Sort(s) => plan_detach_flag(&s.input),
        LogicalPlan::Distinct(d) => plan_detach_flag(&d.input),
        LogicalPlan::Unwind(u) => plan_detach_flag(&u.input),
        LogicalPlan::ProcedureCall(p) => plan_detach_flag(&p.input),
        LogicalPlan::Aggregate(a) => plan_detach_flag(&a.input),
        LogicalPlan::CommunityLookup(c) => plan_detach_flag(&c.input),
        LogicalPlan::NamedPath(np) => plan_detach_flag(&np.input),
        _ => false,
    }
}
