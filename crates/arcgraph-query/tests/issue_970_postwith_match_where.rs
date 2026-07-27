//! #970 - a post-WITH MATCH WHERE may reference carried pipeline bindings.
//!
//! These full-pipeline regressions pin the compute-then-filter idiom:
//! `MATCH ... WITH avg(...) AS a MATCH ... WHERE m.sal > a`.

use arcgraph_core::LabelId;
use arcgraph_query::QueryEngine;
use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::executor::value::Value;
use arcgraph_query::explain;
use arcgraph_query::explain::{PlanTree, PlanTreeOp};
use arcgraph_query::semantic::StubCatalogProvider;

const STUB_FIRST_LABEL_ID: u32 = 1024;

fn catalog() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_label_id("E", LabelId::new(STUB_FIRST_LABEL_ID))
        .with_properties(["sal"])
}

fn run_after_fixture(query: &str) -> Vec<Vec<Value>> {
    let catalog = catalog();
    let substrate = StubExecutorSubstrate::new();
    let engine = QueryEngine::new(&catalog);
    for sal in [40, 60, 80, 50, 70, 100] {
        engine
            .execute(&format!("CREATE (:E {{sal: {sal}}}) RETURN 1"), &substrate)
            .expect("create fixture");
    }
    engine.execute(query, &substrate).expect("execute").rows
}

fn one_cell(query: &str) -> Value {
    let rows = run_after_fixture(query);
    assert_eq!(rows.len(), 1, "expected one row for `{query}`: {rows:?}");
    assert_eq!(
        rows[0].len(),
        1,
        "expected one column for `{query}`: {rows:?}"
    );
    rows[0][0].clone()
}

fn filter_directly_over_scan(pt: &PlanTree) -> bool {
    (pt.op == PlanTreeOp::Filter && pt.children.iter().any(|c| c.op == PlanTreeOp::Scan))
        || pt.children.iter().any(filter_directly_over_scan)
}

#[test]
fn avg_carried_binding_is_visible_to_post_with_match_where() {
    assert_eq!(
        one_cell("MATCH (n:E) WITH avg(n.sal) AS a MATCH (m:E) WHERE m.sal > a RETURN count(m)"),
        Value::Integer(3)
    );
}

#[test]
fn literal_carried_binding_is_visible_to_post_with_match_where() {
    assert_eq!(
        one_cell("WITH 50 AS a MATCH (m:E) WHERE m.sal > a RETURN count(m)"),
        Value::Integer(4)
    );
}

#[test]
fn explicit_with_workaround_stays_green() {
    assert_eq!(
        one_cell(
            "MATCH (n:E) WITH avg(n.sal) AS a MATCH (m:E) WITH m, a WHERE m.sal > a RETURN count(m)"
        ),
        Value::Integer(3)
    );
}

#[test]
fn return_count_and_carried_avg_control_stays_green() {
    let rows = run_after_fixture("MATCH (n:E) WITH avg(n.sal) AS a MATCH (m:E) RETURN count(m), a");
    assert_eq!(rows.len(), 1, "expected one row: {rows:?}");
    assert_eq!(rows[0][0], Value::Integer(6));
    match rows[0][1] {
        Value::Float(f) => assert!((f - (400.0 / 6.0)).abs() < 1e-9, "avg was {f}"),
        ref other => panic!("expected float avg, got {other:?}"),
    }
}

#[test]
fn first_match_where_still_pushes_filter_to_scan() {
    assert_eq!(
        one_cell("MATCH (n:E) WHERE n.sal > 50 RETURN count(n)"),
        Value::Integer(4)
    );

    let pt = explain(
        "EXPLAIN MATCH (n:E) WHERE n.sal > 50 RETURN count(n)",
        &catalog(),
    )
    .expect("explain");
    assert!(
        filter_directly_over_scan(&pt),
        "first MATCH WHERE must retain Filter->Scan pushdown: {pt:?}"
    );
}

#[test]
fn mixed_pattern_and_carried_predicate_filters_correctly() {
    assert_eq!(
        one_cell(
            "MATCH (n:E) WITH avg(n.sal) AS a MATCH (m:E) WHERE m.sal > a AND m.sal < 100 RETURN count(m)"
        ),
        Value::Integer(2)
    );
}

#[test]
fn carried_binding_not_referenced_keeps_match_where_pushdown() {
    assert_eq!(
        one_cell("WITH 50 AS a MATCH (m:E) WHERE m.sal > 30 RETURN count(m)"),
        Value::Integer(6)
    );

    let pt = explain(
        "EXPLAIN WITH 50 AS a MATCH (m:E) WHERE m.sal > 30 RETURN count(m)",
        &catalog(),
    )
    .expect("explain");
    assert!(
        filter_directly_over_scan(&pt),
        "pattern-only post-WITH MATCH WHERE must retain Filter->Scan pushdown: {pt:?}"
    );
}
