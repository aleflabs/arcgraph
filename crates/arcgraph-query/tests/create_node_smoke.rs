//! ADR-147 W26-θ Phase 1 — CREATE node end-to-end smoke test.
//!
//! Walks the full query-side pipeline:
//!
//! 1. Parse `CREATE (n:User {id: 42}) RETURN n` to a `Statement`.
//! 2. Bind via `BindingVisitor::bind` against a `StubCatalogProvider`.
//! 3. Type-check via `TypeCheckVisitor::check`.
//! 4. Cross-substrate validate via `CrossSubstrateValidator::validate`.
//! 5. Lower to a `LogicalPlan` via `LogicalPlanLoweringVisitor::lower`.
//! 6. Build a `Pipeline` and execute against a `StubExecutorSubstrate`,
//!    asserting exactly one row is emitted and the new node-id is
//!    bound to `n`.
//!
//! Forward-pins:
//! - Phase 1 properties are stored empty at this layer; the test does
//!   NOT round-trip
//!   property values through MATCH-by-property. The round-trip via
//!   `scan_nodes` (by id + label) is asserted in `create_node_proptest.rs`
//!   and the executor's unit tests in `executor/ops/create_node.rs`.

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
fn create_node_with_var_and_label_parses_through_planner() {
    // Phase 1 happy path: CREATE with var binding + label + properties.
    let plan = lower("CREATE (n:User {id: 42}) RETURN n");
    // The top of the plan is the RETURN's Project; the CREATE node
    // operator is its leaf input (after lowering's left-deep chain).
    // We don't pin the exact tree shape (the lowering is free to
    // re-shape via the cost walker) — just assert that a CreateNode
    // variant exists somewhere in the tree.
    let has_create = find_create_node(&plan);
    assert!(has_create, "expected CreateNode in plan: {plan:?}");
}

#[test]
fn create_node_anonymous_parses_through_planner() {
    let plan = lower("CREATE (:User)");
    let has_create = find_create_node(&plan);
    assert!(has_create, "expected CreateNode in plan: {plan:?}");
}

#[test]
fn create_node_no_label_parses_through_planner() {
    let plan = lower("CREATE (n) RETURN n");
    let has_create = find_create_node(&plan);
    assert!(has_create, "expected CreateNode in plan: {plan:?}");
}

#[test]
fn create_node_multiple_items_lowers_to_chain() {
    // issue #832 (silent multi-pattern data loss). BEFORE the fix this
    // test only asserted "a CreateNode is present" — a WEAK oracle that
    // PASSED even though `lower_create` discarded all but the last item
    // (the bug). The strengthened oracle pins the actual chain depth AND
    // executes end-to-end to assert BOTH nodes persist.
    let plan = lower("CREATE (a:Foo), (b:Bar)");
    assert_eq!(
        create_chain_depth(&plan),
        2,
        "CREATE (a),(b) MUST lower to a 2-item left-deep chain (every \
         item executes); got {plan:#?}"
    );

    // End-to-end: both nodes MUST persist (the load-bearing oracle).
    let s = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let mut op = Pipeline::build(&plan).expect("pipeline build OK");
    loop {
        let b = op.next_batch(&ctx, &s).expect("batch OK");
        if b.is_empty() {
            break;
        }
    }
    let nodes = s
        .scan_nodes(TenantId::DEFAULT, None, arcgraph_core::Lsn::MAX)
        .expect("scan_nodes OK");
    assert_eq!(
        nodes.len(),
        2,
        "CREATE (a),(b) MUST persist BOTH nodes (issue #832 — the bug \
         persisted only the LAST)"
    );
}

/// Depth of a CREATE-item chain — follows the `input` thread through
/// `CreateNode` / `CreateRel`, descending through wrapping clauses
/// (e.g. a RETURN `Project`). Returns the number of create ops on the
/// chain (issue #832 oracle).
fn create_chain_depth(plan: &LogicalPlan) -> usize {
    match plan {
        LogicalPlan::CreateNode(c) => 1 + c.input.as_deref().map(create_chain_depth).unwrap_or(0),
        LogicalPlan::CreateRel(c) => 1 + c.input.as_deref().map(create_chain_depth).unwrap_or(0),
        LogicalPlan::Project(p) => create_chain_depth(&p.input),
        LogicalPlan::Filter(f) => create_chain_depth(&f.input),
        _ => 0,
    }
}

#[test]
fn create_node_executes_against_stub_substrate_emits_one_row() {
    // End-to-end through the executor.
    let plan = lower("CREATE (n:User) RETURN n");
    let s = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);

    // We pull the row directly out of the pipeline — the
    // `QueryEngine::execute` surface adds budget + cancellation +
    // RAII-guard layers around the same call, but for the smoke we
    // bind directly to the operator dispatch.
    let mut op = Pipeline::build(&plan).expect("pipeline build OK");
    let b1 = op.next_batch(&ctx, &s).expect("first batch OK");
    assert_eq!(b1.row_count(), 1, "exactly one row from CREATE");
    let b2 = op.next_batch(&ctx, &s).expect("second batch OK");
    assert!(b2.is_empty(), "second batch is EOS");

    // Assert the substrate observed the create — `scan_nodes`
    // returns the new node-id.
    let nodes = s
        .scan_nodes(TenantId::DEFAULT, None, arcgraph_core::Lsn::MAX)
        .expect("scan_nodes OK");
    assert_eq!(nodes.len(), 1, "substrate observed exactly one CREATE");
}

#[test]
fn create_node_with_return_emits_node_value_in_row() {
    let plan = lower("CREATE (n:User) RETURN n");
    let s = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let mut op = Pipeline::build(&plan).expect("pipeline build OK");
    let b1 = op.next_batch(&ctx, &s).expect("first batch OK");
    let rows = b1.row_count();
    assert_eq!(rows, 1);
    // The row's first cell is a Value::Node carrying the new id.
    let r = b1.row(0);
    assert!(!r.is_empty(), "RETURN n row carries the binding cell");
    // The cell should be a Node (the materialized NodeView from
    // CreateNodeOp); a future amendment may expose the
    // direct-substring read. The non-null assertion is the load-
    // bearing pin for v1.0-α.
    assert!(
        !matches!(r[0], Value::Null),
        "RETURN n row's bound cell is non-null"
    );
}

#[test]
fn create_node_anonymous_emits_zero_column_row() {
    // openCypher v9 § 6 — "1 node created" semantic for anonymous
    // CREATEs. The row is a zero-column tuple.
    let plan = lower("CREATE (:User)");
    let s = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let mut op = Pipeline::build(&plan).expect("pipeline build OK");
    let b1 = op.next_batch(&ctx, &s).expect("first batch OK");
    assert_eq!(b1.row_count(), 1, "anonymous CREATE still emits 1 row");
    let r = b1.row(0);
    assert!(r.is_empty(), "anonymous CREATE row is 0-column");
}

#[test]
fn create_node_admits_parameter_property_value_at_type_check() {
    // T4 (ADR-147-amendment-03, D-1) — FLIP of the Phase 1 rejection.
    // Parameter-typed CREATE property values are now ADMITTED at
    // type-check (the live CreateSpineOp executor `evaluate`s them). Was
    // `create_node_rejects_parameter_property_value_at_type_check` (RED
    // on revert: the old code pushed `CreatePropertyValueNotLiteral`).
    let stmt = parse("CREATE (n:User {id: $p}) RETURN n").expect("parse OK");
    let cat = StubCatalogProvider::new();
    let mut bound = BindingVisitor::bind(&stmt, "...", &cat).expect("bind OK");
    let result = TypeCheckVisitor::check(&mut bound, &cat);
    assert!(
        result.is_ok(),
        "type-check must ADMIT a parameter-typed CREATE property (amendment-03); got {:?}",
        result.err()
    );
}

#[test]
fn create_node_still_rejects_map_property_value_at_type_check() {
    // T9-adjacent (ADR-147-amendment-03, D-1) — a MAP LITERAL property
    // value stays REJECTED at type-check (openCypher forbids map property
    // values; ADR-191 D-11). The amendment lifted params/exprs, NOT maps.
    // (`{a:1}` no inner space — the compound-atomic prop grammar suppresses
    // implicit whitespace; a pre-existing grammar constraint, out of D-1
    // scope.)
    let stmt = parse("CREATE (n:User {m: {a:1}}) RETURN n").expect("parse OK");
    let cat = StubCatalogProvider::new();
    let mut bound = BindingVisitor::bind(&stmt, "...", &cat).expect("bind OK");
    let result = TypeCheckVisitor::check(&mut bound, &cat);
    assert!(
        result.is_err(),
        "map-literal CREATE property must stay rejected at type-check"
    );
    let errs = result.unwrap_err();
    assert!(
        errs.iter()
            .any(|e| format!("{e:?}").contains("CreatePropertyValueNotLiteral")),
        "expected CreatePropertyValueNotLiteral on a map property; got {errs:?}"
    );
}

#[test]
fn create_node_still_rejects_function_call_property_value_at_type_check() {
    // T13 (ADR-147-amendment-03, D-1) — a FunctionCall property value
    // (here `timestamp()`) stays REJECTED at type-check (determinism +
    // unbounded-materialization fence). Deferred to a later amendment.
    let stmt = parse("CREATE (n:User {t: timestamp()}) RETURN n").expect("parse OK");
    let cat = StubCatalogProvider::new();
    let mut bound = BindingVisitor::bind(&stmt, "...", &cat).expect("bind OK");
    let result = TypeCheckVisitor::check(&mut bound, &cat);
    assert!(
        result.is_err(),
        "function-call CREATE property must stay rejected at type-check"
    );
    let errs = result.unwrap_err();
    assert!(
        errs.iter()
            .any(|e| format!("{e:?}").contains("CreatePropertyValueNotLiteral")),
        "expected CreatePropertyValueNotLiteral on a function-call property; got {errs:?}"
    );
}

#[test]
fn create_node_still_rejects_nested_function_in_list_at_type_check() {
    // T11 (ADR-147-amendment-03, D-1) — the recursion closes the nesting
    // bypass: a `FunctionCall` hidden inside a list literal element
    // (`[randomUUID()]`) makes the WHOLE list inadmissible at type-check
    // (never reaches the executor / never OOMs).
    let stmt = parse("CREATE (n:User {x: [randomUUID()]}) RETURN n").expect("parse OK");
    let cat = StubCatalogProvider::new();
    let mut bound = BindingVisitor::bind(&stmt, "...", &cat).expect("bind OK");
    let result = TypeCheckVisitor::check(&mut bound, &cat);
    assert!(
        result.is_err(),
        "a function call nested in a list element must be rejected at type-check (T11)"
    );
    let errs = result.unwrap_err();
    assert!(
        errs.iter()
            .any(|e| format!("{e:?}").contains("CreatePropertyValueNotLiteral")),
        "expected CreatePropertyValueNotLiteral for [randomUUID()]; got {errs:?}"
    );
}

#[test]
fn create_node_admits_arithmetic_expression_property_value_at_type_check() {
    // T-expr (ADR-147-amendment-03, D-1) — a bounded arithmetic
    // expression over params (`$a+1`) is ADMITTED at type-check. (No
    // spaces around `+`: the compound-atomic prop grammar suppresses
    // implicit whitespace; the whitespaced form `$a + 1` is a pre-existing
    // parse limitation out of D-1 scope.)
    let stmt = parse("CREATE (n:User {id: $a+1}) RETURN n").expect("parse OK");
    let cat = StubCatalogProvider::new();
    let mut bound = BindingVisitor::bind(&stmt, "...", &cat).expect("bind OK");
    let result = TypeCheckVisitor::check(&mut bound, &cat);
    assert!(
        result.is_ok(),
        "bounded arithmetic CREATE property must be admitted (amendment-03); got {:?}",
        result.err()
    );
}

#[test]
fn create_node_admits_literal_property_values_at_type_check() {
    // Each literal kind: Integer / Float / String / Bool / Null.
    let queries = [
        r#"CREATE (n:User {id: 42}) RETURN n"#,
        r#"CREATE (n:User {weight: 3.14}) RETURN n"#,
        r#"CREATE (n:User {name: "alice"}) RETURN n"#,
        r#"CREATE (n:User {flag: TRUE}) RETURN n"#,
        r#"CREATE (n:User {nothing: NULL}) RETURN n"#,
    ];
    for q in queries {
        let stmt = parse(q).unwrap_or_else(|e| panic!("parse {q:?} OK: {e:?}"));
        let cat = StubCatalogProvider::new();
        let mut bound = BindingVisitor::bind(&stmt, q, &cat).expect("bind OK");
        TypeCheckVisitor::check(&mut bound, &cat).unwrap_or_else(|e| {
            panic!("type-check {q:?} must accept literal property values: {e:?}")
        });
    }
}

#[test]
fn create_node_grammar_admits_multi_item_clause() {
    // ADR-147 §D-1 — grammar admits CREATE (a), (b), ... ; lowering
    // produces a CreateNode per item.
    let stmt = parse("CREATE (a:Foo), (b:Bar), (c:Baz) RETURN a, b, c").expect("parse OK");
    let cat = StubCatalogProvider::new();
    let _ = BindingVisitor::bind(&stmt, "...", &cat).expect("bind OK");
}

#[test]
fn create_node_round_trip_display_re_parses() {
    // ADR-147 §D-2 + grammar_proptest discipline: Display round-trips
    // through the parser.
    let original = "CREATE (n:User {id: 42}) RETURN n";
    let parsed = parse(original).expect("parse OK");
    let printed = format!("{parsed}");
    let re_parsed = parse(&printed).expect("re-parse OK");
    assert_eq!(parsed, re_parsed, "Display round-trips");
}

#[test]
fn create_node_grammar_rejects_phase1_unsupported_shapes() {
    // ADR-147 §D-1 Phase 1 narrowing: multi-label + bare-CREATE +
    // no-parens MUST be rejected at parse time (syntactic level —
    // the grammar Phase 1 admits only `(var?:Label? {props})`).
    //
    // ADR-148 W26-θ Phase 2 update: CREATE-rel `(a)-[:R]->(b)` now
    // PARSES (Phase 2 lit). The remaining Phase 1 narrowings (multi-
    // label / no-parens / bare keyword) still reject.
    let rejected = [
        ("CREATE (n:User:Admin)", "multi-label (v1.1 forward-pin)"),
        ("CREATE n:User RETURN n", "no-parens"),
        ("CREATE", "bare keyword"),
    ];
    for (q, reason) in rejected {
        let r = parse(q);
        assert!(
            r.is_err(),
            "{reason}: expected parse rejection for {q:?}, got {r:?}"
        );
    }
}

/// Recursively search a LogicalPlan tree for a CreateNode variant.
fn find_create_node(plan: &LogicalPlan) -> bool {
    match plan {
        LogicalPlan::CreateNode(_) => true,
        LogicalPlan::Filter(f) => find_create_node(&f.input),
        LogicalPlan::Project(p) => find_create_node(&p.input),
        LogicalPlan::Limit(l) => find_create_node(&l.input),
        LogicalPlan::Skip(s) => find_create_node(&s.input),
        LogicalPlan::DynamicLimit(d) => find_create_node(&d.input),
        LogicalPlan::Sort(s) => find_create_node(&s.input),
        LogicalPlan::Distinct(d) => find_create_node(&d.input),
        LogicalPlan::Unwind(u) => find_create_node(&u.input),
        LogicalPlan::ProcedureCall(p) => find_create_node(&p.input),
        LogicalPlan::Aggregate(a) => find_create_node(&a.input),
        LogicalPlan::CommunityLookup(c) => find_create_node(&c.input),
        LogicalPlan::NamedPath(np) => find_create_node(&np.input),
        LogicalPlan::Join(j) => find_create_node(&j.left) || find_create_node(&j.right),
        LogicalPlan::LeftOuterJoin(j) => find_create_node(&j.left) || find_create_node(&j.right),
        LogicalPlan::Fusion(f) => f.inputs.iter().any(|inp| find_create_node(inp)),
        LogicalPlan::Union(u) => u.arms.iter().any(find_create_node),
        // ADR-148 W26-θ Phase 2: CreateRel carries source + target
        // sub-plans (each typically a CreateNode at Phase 2).
        LogicalPlan::CreateRel(c) => {
            find_create_node(&c.source_plan) || find_create_node(&c.target_plan)
        }
        // ADR-149 W26-θ Phase 3: Delete walks its input sub-plan.
        LogicalPlan::Delete(d) => find_create_node(&d.input),
        // ADR-150 W26-θ Phase 4: Set / Remove walk their input
        // sub-plan.
        LogicalPlan::Set(s) => find_create_node(&s.input),
        LogicalPlan::Remove(r) => find_create_node(&r.input),
        // ADR-151 W26-θ Phase 5: Merge walks both match + create
        // sub-plans (the create-branch is a CreateNode for Node-shape
        // / CreateRel-wrapping-CreateNodes for Path-shape).
        LogicalPlan::Merge(m) => {
            find_create_node(&m.match_branch) || find_create_node(&m.create_branch)
        }
        LogicalPlan::Scan(_)
        | LogicalPlan::PropertyIndexScan(_)
        | LogicalPlan::CountStore(_)
        | LogicalPlan::Expand(_)
        | LogicalPlan::Empty(_)
        | LogicalPlan::RankByHybrid(_)
        | LogicalPlan::VectorNear(_)
        | LogicalPlan::TextMatch(_)
        | LogicalPlan::CreateVectorIndex(_)
        | LogicalPlan::CreatePropertyIndex(_)
        | LogicalPlan::Call(_)
        | LogicalPlan::CorrelationSeed(_) => false,
    }
}
