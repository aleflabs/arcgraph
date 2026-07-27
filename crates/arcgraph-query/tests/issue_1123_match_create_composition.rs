//! Issue #1123 — MATCH→CREATE composition and prebound CREATE endpoints.

use arcgraph_core::{LabelId, Lsn, PartitionId, TenantId, TypeId};
use arcgraph_query::ExecutorSubstrate;
use arcgraph_query::executor::substrate::StubExecutorSubstrate;
use arcgraph_query::executor::{ExecutionContext, Pipeline, value::Value};
use arcgraph_query::logical_plan::{LogicalPlan, LogicalPlanLoweringVisitor};
use arcgraph_query::semantic::{
    BindingVisitor, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};
use arcgraph_query::{QueryEngine, Statement, parse};

const PERSON: u32 = 1024;
const TAG: u32 = 1025;
const A_LABEL: u32 = 1024;
const B_LABEL: u32 = 1025;
const X_LABEL: u32 = 1026;
const Y_LABEL: u32 = 1027;
const REL: u32 = 1024;

fn cat() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_label_id("Person", LabelId::new(PERSON))
        .with_label_id("Tag", LabelId::new(TAG))
        .with_label_id("A", LabelId::new(A_LABEL))
        .with_label_id("B", LabelId::new(B_LABEL))
        .with_label_id("X", LabelId::new(X_LABEL))
        .with_label_id("Y", LabelId::new(Y_LABEL))
        .with_label_id("Label", LabelId::new(TAG))
        .with_rel_type_id("KNOWS", TypeId::new(REL))
        .with_rel_type_id("R", TypeId::new(REL))
}

fn lower_with(query: &str, cat: &StubCatalogProvider) -> LogicalPlan {
    let stmt = parse(query).expect("parse OK");
    let inner = match stmt {
        Statement::Read(_) => stmt,
        other => panic!("expected Read statement, got {other:?}"),
    };
    let mut bound = BindingVisitor::bind(&inner, query, cat).expect("bind OK");
    TypeCheckVisitor::check(&mut bound, cat).expect("type-check OK");
    CrossSubstrateValidator::validate(&bound, cat).expect("cross-substrate OK");
    LogicalPlanLoweringVisitor::lower(&bound).expect("lower OK")
}

fn execute_query(
    query: &str,
    substrate: &StubExecutorSubstrate,
    cat: &StubCatalogProvider,
) -> Vec<Vec<Value>> {
    let plan = lower_with(query, cat);
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let mut op = Pipeline::build(&plan).expect("pipeline build OK");
    let mut out = Vec::new();
    loop {
        let batch = op.next_batch(&ctx, substrate).expect("batch OK");
        if batch.is_empty() {
            break;
        }
        for i in 0..batch.row_count() {
            out.push(batch.row(i).to_vec());
        }
    }
    out
}

fn string_cell(v: &Value) -> &str {
    match v {
        Value::String(s) => s,
        other => panic!("expected String, got {other:?}"),
    }
}

fn node_label(v: &Value) -> Option<&str> {
    match v {
        Value::Node(n) => n.label_name.as_deref(),
        other => panic!("expected Node, got {other:?}"),
    }
}

fn seed_people(substrate: &StubExecutorSubstrate, cat: &StubCatalogProvider) {
    let rows = execute_query(
        r#"CREATE (:Person {name: "Alice"}),(:Person {name: "Bob"})"#,
        substrate,
        cat,
    );
    assert_eq!(rows.len(), 1);
}

#[test]
fn repro_match_match_create_uses_prebound_endpoints() {
    let substrate = StubExecutorSubstrate::new();
    let cat = cat();
    seed_people(&substrate, &cat);

    let rows = execute_query(
        r#"MATCH (p1:Person {name: "Alice"}) MATCH (p2:Person {name: "Bob"})
           CREATE (p1)-[:KNOWS]->(p2) RETURN p1.name, p2.name"#,
        &substrate,
        &cat,
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(string_cell(&rows[0][0]), "Alice");
    assert_eq!(string_cell(&rows[0][1]), "Bob");

    let edge_rows = execute_query(
        r#"MATCH (a)-[r:KNOWS]->(b) RETURN a.name, b.name"#,
        &substrate,
        &cat,
    );
    assert_eq!(edge_rows.len(), 1, "exactly one KNOWS edge was created");
    assert_eq!(string_cell(&edge_rows[0][0]), "Alice");
    assert_eq!(string_cell(&edge_rows[0][1]), "Bob");
    let nodes = substrate
        .scan_nodes(TenantId::DEFAULT, None, Lsn::MAX)
        .expect("scan_nodes OK");
    assert_eq!(nodes.len(), 2, "prebound endpoints must not create nodes");
}

#[test]
fn product_match_rows_create_one_edge_per_row() {
    let substrate = StubExecutorSubstrate::new();
    let cat = cat();
    execute_query(
        r#"CREATE (:A {name: "A1"}),(:A {name: "A2"}),(:B {name: "B"})"#,
        &substrate,
        &cat,
    );
    let rows = execute_query(
        r#"MATCH (a:A) MATCH (b:B) CREATE (a)-[:R]->(b) RETURN a.name, b.name"#,
        &substrate,
        &cat,
    );
    assert_eq!(rows.len(), 2);
    let edge_rows = execute_query(
        r#"MATCH (a)-[r:R]->(b) RETURN a.name, b.name"#,
        &substrate,
        &cat,
    );
    assert_eq!(edge_rows.len(), 2);
    assert!(edge_rows.iter().all(|r| string_cell(&r[1]) == "B"));
}

#[test]
fn fresh_create_after_match_runs_per_input_row() {
    let substrate = StubExecutorSubstrate::new();
    let cat = cat();
    seed_people(&substrate, &cat);
    let rows = execute_query(
        r#"MATCH (a:Person) CREATE (b:Tag) RETURN b"#,
        &substrate,
        &cat,
    );
    assert_eq!(rows.len(), 2);
    let tags = execute_query(r#"MATCH (t:Tag) RETURN t"#, &substrate, &cat);
    assert_eq!(
        tags.len(),
        2,
        "CREATE after MATCH is per-row, not single-shot"
    );
}

#[test]
fn with_carried_prebound_endpoints_create_edges() {
    let substrate = StubExecutorSubstrate::new();
    let cat = cat();
    seed_people(&substrate, &cat);
    let rows = execute_query(
        r#"MATCH (p1:Person {name: "Alice"}) MATCH (p2:Person {name: "Bob"})
           WITH p1, p2 CREATE (p1)-[:KNOWS]->(p2) RETURN p1.name, p2.name"#,
        &substrate,
        &cat,
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(string_cell(&rows[0][0]), "Alice");
    assert_eq!(string_cell(&rows[0][1]), "Bob");
}

#[test]
fn comma_pattern_match_variant_creates_edge() {
    let substrate = StubExecutorSubstrate::new();
    let cat = cat();
    seed_people(&substrate, &cat);
    let rows = execute_query(
        r#"MATCH (a:Person),(b:Person) WHERE a.name = "Alice" AND b.name = "Bob"
           CREATE (a)-[:R]->(b) RETURN a.name, b.name"#,
        &substrate,
        &cat,
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(string_cell(&rows[0][0]), "Alice");
    assert_eq!(string_cell(&rows[0][1]), "Bob");
}

#[test]
fn prebound_with_label_in_create_is_duplicate_binding_error() {
    let query = r#"MATCH (p1:Person {name: "Alice"}) CREATE (p1:Label)"#;
    let parsed = parse(query).expect("parse OK");
    let errors = BindingVisitor::bind(&parsed, query, &cat()).expect_err("bind must fail");
    let rendered = errors
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(rendered.contains("duplicate binding `p1`"), "{rendered}");
    assert!(
        rendered.contains("already bound — cannot re-declare with labels/properties in CREATE"),
        "{rendered}"
    );
}

#[test]
fn standalone_both_new_create_rel_unchanged() {
    let substrate = StubExecutorSubstrate::new();
    let cat = cat();
    let rows = execute_query(r#"CREATE (a)-[:R]->(b)"#, &substrate, &cat);
    assert_eq!(rows.len(), 1);
    let nodes = substrate
        .scan_nodes(TenantId::DEFAULT, None, Lsn::MAX)
        .expect("scan_nodes OK");
    assert_eq!(nodes.len(), 2);
    let edge_rows = execute_query(r#"MATCH (a)-[r:R]->(b) RETURN r"#, &substrate, &cat);
    assert_eq!(edge_rows.len(), 1);
}

#[test]
fn mixed_prebound_source_and_fresh_target_runs_per_row() {
    let substrate = StubExecutorSubstrate::new();
    let cat = cat();
    seed_people(&substrate, &cat);
    let rows = execute_query(
        r#"MATCH (p1:Person) CREATE (p1)-[:R]->(new:Tag) RETURN p1.name"#,
        &substrate,
        &cat,
    );
    assert_eq!(rows.len(), 2);
    let tags = execute_query(r#"MATCH (t:Tag) RETURN t"#, &substrate, &cat);
    assert_eq!(tags.len(), 2);
    let edges = execute_query(
        r#"MATCH (p:Person)-[r:R]->(t:Tag) RETURN p.name, t"#,
        &substrate,
        &cat,
    );
    assert_eq!(edges.len(), 2);
}

#[test]
fn return_after_create_keeps_prebound_variables_projectable() {
    let substrate = StubExecutorSubstrate::new();
    let cat = cat();
    seed_people(&substrate, &cat);
    let rows = execute_query(
        r#"MATCH (p1:Person {name: "Alice"}) MATCH (p2:Person {name: "Bob"})
           CREATE (p1)-[:KNOWS]->(p2) RETURN p1, p2"#,
        &substrate,
        &cat,
    );
    assert_eq!(rows.len(), 1);
    assert!(matches!(rows[0][0], Value::Node(_)));
    assert!(matches!(rows[0][1], Value::Node(_)));
}

#[test]
fn tck_match5_setup_builds_create_spine_without_debug_stack_overflow() {
    let substrate = StubExecutorSubstrate::new();
    let cat = StubCatalogProvider::new()
        .with_labels(["A", "B", "C", "D"])
        .with_rel_types(["LIKES"])
        .with_properties(["name"]);
    let n = 15;
    let nodes = (0..n)
        .map(|i| format!("(n{i}:A {{name: 'n{i}'}})"))
        .collect::<Vec<_>>()
        .join(", ");
    let rels = (1..n)
        .map(|i| format!("(n0)-[:LIKES]->(n{i})"))
        .collect::<Vec<_>>()
        .join(", ");
    let query = format!("CREATE {nodes}\nCREATE {rels}");
    // PR #1142 CI red: the debug TCK harness overflowed its stack in
    // clauses/match/Match5.feature while building this setup's left-deep
    // MATCH->CREATE-seeded write spine.
    let rows = execute_query(&query, &substrate, &cat);
    assert_eq!(rows.len(), 1);
    let edges = execute_query(r#"MATCH (a)-[r]->(b) RETURN r"#, &substrate, &cat);
    assert_eq!(edges.len(), 14);
}

#[test]
fn deep_create_spine_pull_path_uses_bounded_stack() {
    let substrate = StubExecutorSubstrate::new();
    let cat = StubCatalogProvider::new()
        .with_labels(["A", "B", "C", "D"])
        .with_rel_types(["LIKES"])
        .with_properties(["name"]);
    let n = std::env::var("ARCGRAPH_1123_CREATE_SPINE_N")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(600);
    let nodes = (0..n)
        .map(|i| format!("(n{i}:A {{name: 'n{i}'}})"))
        .collect::<Vec<_>>()
        .join(", ");
    let rels = (1..n)
        .map(|i| format!("(n0)-[:LIKES]->(n{i})"))
        .collect::<Vec<_>>()
        .join(", ");
    let query = format!("CREATE {nodes}\nCREATE {rels}");

    // PR #1142 R2 measured debug pull-stack abort at n=287: nested
    // CreateNodeOp/CreateRelOp input.next_batch frames recursed one
    // frame per CREATE item. The composite CreateSpineOp keeps pull
    // frames O(1) in chain depth.
    let engine = QueryEngine::new(&cat);
    let result = engine
        .execute(&query, &substrate)
        .expect("deep CREATE spine must build and pull in debug");
    assert_eq!(result.rows().len(), 1);

    let edges = execute_query(r#"MATCH (a)-[r]->(b) RETURN r"#, &substrate, &cat);
    assert_eq!(edges.len(), n - 1);
}

#[test]
fn create_spine_threads_match_and_fresh_endpoint_bindings() {
    let substrate = StubExecutorSubstrate::new();
    let cat = cat();
    seed_people(&substrate, &cat);

    let rows = execute_query(
        r#"MATCH (p:Person) CREATE (a:X)-[:R]->(b:Y) RETURN p, a, b"#,
        &substrate,
        &cat,
    );
    assert_eq!(rows.len(), 2);
    for row in &rows {
        assert_eq!(row.len(), 3);
        assert!(matches!(row[0], Value::Node(_)));
        assert_eq!(node_label(&row[1]), Some("X"));
        assert_eq!(node_label(&row[2]), Some("Y"));
    }

    let edges = execute_query(r#"MATCH (a)-[r:R]->(b) RETURN r"#, &substrate, &cat);
    assert_eq!(edges.len(), 2);
}
