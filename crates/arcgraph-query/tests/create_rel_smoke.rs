//! ADR-148 W26-θ Phase 2 — CREATE rel end-to-end smoke test.
//!
//! Walks the full query-side pipeline:
//!
//! 1. Parse `CREATE (a:User)-[r:KNOWS {since: 2024}]->(b:User) RETURN r`
//!    to a `Statement`.
//! 2. Bind via `BindingVisitor::bind` against a `StubCatalogProvider`.
//! 3. Type-check via `TypeCheckVisitor::check`.
//! 4. Cross-substrate validate via `CrossSubstrateValidator::validate`.
//! 5. Lower to a `LogicalPlan` via `LogicalPlanLoweringVisitor::lower`
//!    (emits a tree carrying a `LogicalPlan::CreateRel` over two
//!    `LogicalPlan::CreateNode` sub-plans).
//! 6. Build a `Pipeline` and execute against a `StubExecutorSubstrate`,
//!    asserting exactly one row is emitted and the new rel-id is
//!    bound to `r`.
//!
//! Forward-pins (parallel to ADR-147 Phase 1):
//! - Phase 2 properties are stored empty at this layer; the test does
//!   NOT round-trip
//!   property values through MATCH-by-property.
//! - Both endpoints are inline-CREATE node specs at Phase 2; MATCH-bound
//!   resolution forward-pinned to Phase 5.

use arcgraph_core::{PartitionId, TenantId};

use arcgraph_query::ExecutorSubstrate;
use arcgraph_query::executor::substrate::StubExecutorSubstrate;
use arcgraph_query::executor::{ExecutionContext, value::Value};
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
fn create_rel_left_to_right_parses_through_planner() {
    let plan = lower("CREATE (a:User)-[r:KNOWS]->(b:User) RETURN r");
    assert!(
        find_create_rel(&plan),
        "expected CreateRel in plan: {plan:?}"
    );
}

#[test]
fn create_rel_right_to_left_parses_through_planner() {
    let plan = lower("CREATE (a:User)<-[r:FOLLOWED]-(b:User) RETURN r");
    assert!(
        find_create_rel(&plan),
        "expected CreateRel in plan: {plan:?}"
    );
}

#[test]
fn create_rel_anonymous_rel_var_parses() {
    // Anonymous rel: `(a)-[:R]->(b)` — no rel-var binding.
    let plan = lower("CREATE (a:User)-[:KNOWS]->(b:User)");
    assert!(
        find_create_rel(&plan),
        "expected CreateRel in plan: {plan:?}"
    );
}

#[test]
fn create_rel_with_literal_properties_parses() {
    let plan = lower("CREATE (a:User)-[r:KNOWS {since: 2024}]->(b:User) RETURN r");
    assert!(find_create_rel(&plan), "expected CreateRel: {plan:?}");
}

#[test]
fn create_rel_executes_against_stub_substrate_emits_one_row() {
    let plan = lower("CREATE (a:User)-[r:KNOWS]->(b:User) RETURN r");
    let s = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let mut op = Pipeline::build(&plan).expect("pipeline build OK");
    let b1 = op.next_batch(&ctx, &s).expect("first batch OK");
    assert_eq!(b1.row_count(), 1, "exactly one row from CREATE-rel");
    let b2 = op.next_batch(&ctx, &s).expect("second batch OK");
    assert!(b2.is_empty(), "second batch is EOS");
}

#[test]
fn create_rel_with_return_emits_rel_value_in_row() {
    let plan = lower("CREATE (a:User)-[r:KNOWS]->(b:User) RETURN r");
    let s = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let mut op = Pipeline::build(&plan).expect("pipeline build OK");
    let b1 = op.next_batch(&ctx, &s).expect("first batch OK");
    assert_eq!(b1.row_count(), 1);
    // The row's first cell is a Value::Relationship carrying the new id.
    let r = b1.row(0);
    assert!(!r.is_empty(), "RETURN r row carries the binding cell");
    assert!(
        matches!(r[0], Value::Relationship(_)),
        "RETURN r row's bound cell is a Relationship value: {:?}",
        r[0]
    );
}

#[test]
fn create_rel_anonymous_emits_zero_column_row() {
    // openCypher v9 § 6 — "1 relationship created" semantic for
    // anonymous CREATE-rels. The row is a zero-column tuple.
    let plan = lower("CREATE (a:User)-[:KNOWS]->(b:User)");
    let s = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let mut op = Pipeline::build(&plan).expect("pipeline build OK");
    let b1 = op.next_batch(&ctx, &s).expect("first batch OK");
    assert_eq!(b1.row_count(), 1, "anonymous CREATE-rel still emits 1 row");
    let r = b1.row(0);
    assert!(r.is_empty(), "anonymous CREATE-rel row is 0-column");
}

#[test]
fn create_rel_creates_two_nodes_and_one_edge_observable_via_substrate() {
    // End-to-end on the stub: a CREATE-path lowers to
    // CreateNode(source) + CreateNode(target) + CreateRel. The stub
    // unions the CREATE-d nodes into `scan_nodes` and the CREATE-d
    // edges into `expand`.
    let plan = lower("CREATE (a:User)-[r:KNOWS]->(b:User) RETURN r");
    let s = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let mut op = Pipeline::build(&plan).expect("pipeline build OK");
    let _ = op.next_batch(&ctx, &s).expect("first batch OK");
    // scan_nodes finds the 2 CREATE-d nodes.
    let nodes = s
        .scan_nodes(TenantId::DEFAULT, None, arcgraph_core::Lsn::MAX)
        .expect("scan_nodes OK");
    assert_eq!(nodes.len(), 2, "substrate observed exactly 2 CREATEs");
    // expand from the source NodeId finds the new rel.
    let source_id = nodes[0].node.id;
    let edges = s
        .expand(
            TenantId::DEFAULT,
            source_id,
            None,
            arcgraph_query::logical_plan::Direction::LeftToRight,
            arcgraph_core::Lsn::MAX,
        )
        .expect("expand OK");
    assert_eq!(edges.len(), 1, "expand observed exactly 1 CREATE-rel");
}

#[test]
fn create_rel_admits_parameter_property_value_at_type_check() {
    // T4-rel (ADR-147-amendment-03, D-1) — FLIP: parameter-typed rel
    // property values are now ADMITTED at type-check (the CREATE-path
    // bags flow through the same evaluable gate as node bags). Was
    // `create_rel_rejects_parameter_property_value_at_type_check`.
    let stmt = parse("CREATE (a:User)-[r:KNOWS {since: $year}]->(b:User) RETURN r").expect("parse");
    let cat = StubCatalogProvider::new();
    let mut bound = BindingVisitor::bind(&stmt, "...", &cat).expect("bind");
    let result = TypeCheckVisitor::check(&mut bound, &cat);
    assert!(
        result.is_ok(),
        "type-check must ADMIT a parameter-typed CREATE-rel property (amendment-03); got {:?}",
        result.err()
    );
}

#[test]
fn create_rel_still_rejects_function_call_property_value_at_type_check() {
    // T13-rel — a FunctionCall rel property value stays REJECTED at
    // type-check (determinism / unbounded-materialization fence).
    let stmt =
        parse("CREATE (a:User)-[r:KNOWS {since: timestamp()}]->(b:User) RETURN r").expect("parse");
    let cat = StubCatalogProvider::new();
    let mut bound = BindingVisitor::bind(&stmt, "...", &cat).expect("bind");
    let result = TypeCheckVisitor::check(&mut bound, &cat);
    assert!(
        result.is_err(),
        "function-call CREATE-rel property must stay rejected at type-check"
    );
    let errs = result.unwrap_err();
    assert!(
        errs.iter()
            .any(|e| format!("{e:?}").contains("CreatePropertyValueNotLiteral")),
        "expected CreatePropertyValueNotLiteral on rel bag; got {errs:?}"
    );
}

#[test]
fn create_rel_admits_each_literal_kind_on_rel_bag() {
    // Each literal kind: Integer / Float / String / Bool / Null.
    let queries = [
        r#"CREATE (a:User)-[r:KNOWS {since: 2024}]->(b:User) RETURN r"#,
        r#"CREATE (a:User)-[r:KNOWS {weight: 0.5}]->(b:User) RETURN r"#,
        r#"CREATE (a:User)-[r:KNOWS {kind: "close"}]->(b:User) RETURN r"#,
        r#"CREATE (a:User)-[r:KNOWS {active: TRUE}]->(b:User) RETURN r"#,
        r#"CREATE (a:User)-[r:KNOWS {note: NULL}]->(b:User) RETURN r"#,
    ];
    for q in queries {
        let stmt = parse(q).unwrap_or_else(|e| panic!("parse {q:?} OK: {e:?}"));
        let cat = StubCatalogProvider::new();
        let mut bound = BindingVisitor::bind(&stmt, q, &cat).expect("bind");
        TypeCheckVisitor::check(&mut bound, &cat)
            .unwrap_or_else(|e| panic!("type-check {q:?}: {e:?}"));
    }
}

#[test]
fn create_rel_round_trip_display_re_parses() {
    let original = "CREATE (a:User)-[r:KNOWS {since: 2024}]->(b:User) RETURN r";
    let parsed = parse(original).expect("parse OK");
    let printed = format!("{parsed}");
    let re_parsed = parse(&printed).expect("re-parse OK");
    assert_eq!(parsed, re_parsed, "Display round-trips for CREATE-path");
}

#[test]
fn create_rel_grammar_rejects_phase2_unsupported_shapes() {
    // ADR-148 §D-1 Phase 2 narrowing: undirected + label-less rel +
    // variable-length CREATE all reject at parse time.
    let rejected = [
        (
            "CREATE (a:User)-[r:KNOWS]-(b:User)",
            "undirected rel (Phase 4 forward-pin)",
        ),
        (
            "CREATE (a:User)-[r]->(b:User)",
            "label-less rel (mandatory at Phase 2)",
        ),
        (
            "CREATE (a:User)-[r:KNOWS*1..3]->(b:User)",
            "variable-length CREATE (v1.2 forward-pin)",
        ),
    ];
    for (q, reason) in rejected {
        let r = parse(q);
        assert!(
            r.is_err(),
            "{reason}: expected parse rejection for {q:?}, got {r:?}"
        );
    }
}

#[test]
fn create_rel_multi_item_clause_admits_path_then_node() {
    // ADR-148 §D-1 — the CREATE-clause admits `,`-separated items
    // where each is a `create_path | create_node`. The grammar's
    // longest-match ordering puts `create_path` first.
    let stmt =
        parse("CREATE (a:Foo)-[r:LINK]->(b:Bar), (c:Standalone)").expect("multi-item parses");
    let cat = StubCatalogProvider::new();
    let _ = BindingVisitor::bind(&stmt, "...", &cat).expect("bind");
}

/// Recursively search a LogicalPlan tree for a CreateRel variant.
fn find_create_rel(plan: &LogicalPlan) -> bool {
    match plan {
        LogicalPlan::CreateRel(_) => true,
        LogicalPlan::CreateNode(_) => false,
        LogicalPlan::CreateVectorIndex(_) => false,
        LogicalPlan::CreatePropertyIndex(_) => false,
        LogicalPlan::Filter(f) => find_create_rel(&f.input),
        LogicalPlan::Project(p) => find_create_rel(&p.input),
        LogicalPlan::Limit(l) => find_create_rel(&l.input),
        LogicalPlan::Skip(s) => find_create_rel(&s.input),
        LogicalPlan::DynamicLimit(d) => find_create_rel(&d.input),
        LogicalPlan::Sort(s) => find_create_rel(&s.input),
        LogicalPlan::Distinct(d) => find_create_rel(&d.input),
        LogicalPlan::Unwind(u) => find_create_rel(&u.input),
        LogicalPlan::ProcedureCall(p) => find_create_rel(&p.input),
        LogicalPlan::Aggregate(a) => find_create_rel(&a.input),
        LogicalPlan::CommunityLookup(c) => find_create_rel(&c.input),
        LogicalPlan::NamedPath(np) => find_create_rel(&np.input),
        LogicalPlan::Join(j) => find_create_rel(&j.left) || find_create_rel(&j.right),
        LogicalPlan::LeftOuterJoin(j) => find_create_rel(&j.left) || find_create_rel(&j.right),
        LogicalPlan::Fusion(f) => f.inputs.iter().any(|inp| find_create_rel(inp)),
        LogicalPlan::Union(u) => u.arms.iter().any(find_create_rel),
        // ADR-149 W26-θ Phase 3: Delete walks the input sub-plan.
        LogicalPlan::Delete(d) => find_create_rel(&d.input),
        // ADR-150 W26-θ Phase 4: Set / Remove walk the input sub-plan.
        LogicalPlan::Set(s) => find_create_rel(&s.input),
        LogicalPlan::Remove(r) => find_create_rel(&r.input),
        // ADR-151 W26-θ Phase 5: Merge walks both branches.
        LogicalPlan::Merge(m) => {
            find_create_rel(&m.match_branch) || find_create_rel(&m.create_branch)
        }
        LogicalPlan::Scan(_)
        | LogicalPlan::PropertyIndexScan(_)
        | LogicalPlan::CountStore(_)
        | LogicalPlan::Expand(_)
        | LogicalPlan::Empty(_)
        | LogicalPlan::RankByHybrid(_)
        | LogicalPlan::VectorNear(_)
        | LogicalPlan::TextMatch(_)
        | LogicalPlan::Call(_)
        | LogicalPlan::CorrelationSeed(_) => false,
    }
}
