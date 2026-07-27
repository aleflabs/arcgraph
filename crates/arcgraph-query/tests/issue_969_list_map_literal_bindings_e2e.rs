//! #969 — list/map literal elements must evaluate against the current row.
//!
//! These are full-pipeline `QueryEngine::execute` regressions for the
//! silent-wrong bug where `[n.a]` / `{a: n.a}` evaluated row-dependent
//! elements against an empty binding context and returned `null`, while the
//! same property reference worked as a standalone column, map projection,
//! or aggregate argument.

use arcgraph_core::LabelId;
use arcgraph_query::QueryEngine;
use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::executor::value::Value;
use arcgraph_query::semantic::StubCatalogProvider;
use std::collections::BTreeMap;

const STUB_FIRST_LABEL_ID: u32 = 1024;

fn catalog() -> StubCatalogProvider {
    StubCatalogProvider::new().with_label_id("T", LabelId::new(STUB_FIRST_LABEL_ID))
}

fn run_after_fixture(query: &str) -> Vec<Vec<Value>> {
    let catalog = catalog();
    let substrate = StubExecutorSubstrate::new();
    let engine = QueryEngine::new(&catalog);
    engine
        .execute(
            r#"CREATE (n:T {a: 1, b: 2, name: "hi"}) RETURN n"#,
            &substrate,
        )
        .expect("create fixture");
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

fn vmap(entries: &[(&str, Value)]) -> Value {
    Value::Map(
        entries
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect::<BTreeMap<_, _>>(),
    )
}

#[test]
fn property_refs_inside_list_literal_use_current_row() {
    assert_eq!(
        one_cell("MATCH (n:T) RETURN [n.a, n.b]"),
        Value::List(vec![Value::Integer(1), Value::Integer(2)])
    );
    assert_eq!(
        one_cell("MATCH (n:T) RETURN [n.a, 99]"),
        Value::List(vec![Value::Integer(1), Value::Integer(99)])
    );
    assert_eq!(
        one_cell("MATCH (n:T) RETURN [99, n.a]"),
        Value::List(vec![Value::Integer(99), Value::Integer(1)])
    );
}

#[test]
fn property_refs_inside_map_literal_use_current_row() {
    assert_eq!(
        one_cell("MATCH (n:T) RETURN {name: n.name, age: n.a}"),
        vmap(&[
            ("age", Value::Integer(1)),
            ("name", Value::String("hi".into())),
        ])
    );
}

#[test]
fn nested_list_and_map_literals_recurse_with_current_row() {
    assert_eq!(
        one_cell("MATCH (n:T) RETURN [[n.a], n.b]"),
        Value::List(vec![
            Value::List(vec![Value::Integer(1)]),
            Value::Integer(2),
        ])
    );
    assert_eq!(
        one_cell("MATCH (n:T) RETURN {k: [n.a]}"),
        vmap(&[("k", Value::List(vec![Value::Integer(1)]))])
    );
    assert_eq!(
        one_cell("MATCH (n:T) RETURN [{a: n.a}]"),
        Value::List(vec![vmap(&[("a", Value::Integer(1))])])
    );
}

#[test]
fn existing_projection_controls_still_work() {
    assert_eq!(
        run_after_fixture("MATCH (n:T) RETURN n.a, n.b"),
        vec![vec![Value::Integer(1), Value::Integer(2)]]
    );
    assert_eq!(
        one_cell("MATCH (n:T) RETURN collect(n.a)"),
        Value::List(vec![Value::Integer(1)])
    );
    assert_eq!(
        one_cell("MATCH (n:T) RETURN n{.a}"),
        vmap(&[("a", Value::Integer(1))])
    );
}
