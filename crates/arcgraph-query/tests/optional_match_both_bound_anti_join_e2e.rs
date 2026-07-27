//! #996 — OPTIONAL MATCH with both endpoints already bound.
//!
//! Regression fixture for the TriadicSelection1 anti-join shape:
//! `(a)-[:KNOWS]->(b)-[:KNOWS]->(c)` followed by
//! `OPTIONAL MATCH (a)-[r:KNOWS]->(c)`. The optional pattern is an
//! existence check over the specific bound `(a,c)` pair. It must
//! correlate on both endpoints, emit the matching relationship when it
//! exists, and emit exactly one NULL `r` row when it does not.

use arcgraph_core::{LabelId, NodeId, RelId, TenantId, TypeId};
use arcgraph_query::QueryEngine;
use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::executor::value::{NodeView, RelView, Value};
use arcgraph_query::semantic::StubCatalogProvider;

fn cat() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["A"])
        .with_rel_types(["KNOWS"])
        .with_properties(["name"])
}

fn triadic_substrate() -> StubExecutorSubstrate {
    StubExecutorSubstrate::new()
        .with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(1), Some(LabelId::new(1)))
                .with_property("name", Value::String("a".into())),
        )
        .with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(2), None).with_property("name", Value::String("b".into())),
        )
        .with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(3), None).with_property("name", Value::String("c1".into())),
        )
        .with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(4), None).with_property("name", Value::String("c2".into())),
        )
        .with_edge(
            TenantId::DEFAULT,
            RelView::new(
                RelId::new(10),
                NodeId::new(1),
                NodeId::new(2),
                Some(TypeId::new(1)),
            ),
        )
        .with_edge(
            TenantId::DEFAULT,
            RelView::new(
                RelId::new(11),
                NodeId::new(2),
                NodeId::new(3),
                Some(TypeId::new(1)),
            ),
        )
        .with_edge(
            TenantId::DEFAULT,
            RelView::new(
                RelId::new(12),
                NodeId::new(2),
                NodeId::new(4),
                Some(TypeId::new(1)),
            ),
        )
        .with_edge(
            TenantId::DEFAULT,
            RelView::new(
                RelId::new(13),
                NodeId::new(1),
                NodeId::new(3),
                Some(TypeId::new(1)),
            ),
        )
}

fn execute(query: &str) -> Vec<Vec<Value>> {
    QueryEngine::new(&cat())
        .execute(query, &triadic_substrate())
        .unwrap_or_else(|e| panic!("execute failed for `{query}`: {e:?}"))
        .rows()
        .to_vec()
}

fn string_col(rows: &[Vec<Value>]) -> Vec<String> {
    rows.iter()
        .map(|row| match row.first() {
            Some(Value::String(s)) => s.clone(),
            other => panic!("expected String in first column, got {other:?}"),
        })
        .collect()
}

fn name_and_is_null(rows: &[Vec<Value>]) -> Vec<(String, bool)> {
    rows.iter()
        .map(|row| match row.as_slice() {
            [Value::String(name), Value::Boolean(is_null)] => (name.clone(), *is_null),
            other => panic!("expected (String, Boolean), got {other:?}"),
        })
        .collect()
}

#[test]
fn anti_join_null_fill_keeps_only_missing_direct_edge() {
    let rows = execute(
        "MATCH (a:A)-[:KNOWS]->(b)-[:KNOWS]->(c) \
         OPTIONAL MATCH (a)-[r:KNOWS]->(c) \
         WITH c WHERE r IS NULL \
         RETURN c.name ORDER BY c.name",
    );
    assert_eq!(string_col(&rows), vec!["c2".to_string()]);
}

#[test]
fn diagnostic_correlates_both_endpoints_and_null_fills_once() {
    let rows = execute(
        "MATCH (a:A)-[:KNOWS]->(b)-[:KNOWS]->(c) \
         OPTIONAL MATCH (a)-[r:KNOWS]->(c) \
         RETURN c.name, r IS NULL ORDER BY c.name",
    );
    assert_eq!(
        name_and_is_null(&rows),
        vec![("c1".to_string(), false), ("c2".to_string(), true)]
    );
}

#[test]
fn positive_filter_keeps_only_existing_direct_edge() {
    let rows = execute(
        "MATCH (a:A)-[:KNOWS]->(b)-[:KNOWS]->(c) \
         OPTIONAL MATCH (a)-[r]->(c) \
         WITH c WHERE r IS NOT NULL \
         RETURN c.name ORDER BY c.name",
    );
    assert_eq!(string_col(&rows), vec!["c1".to_string()]);
}

#[test]
fn existing_both_bound_edge_returns_relationship_not_null_without_duplication() {
    let rows = execute(
        "MATCH (a:A)-[:KNOWS]->(b)-[:KNOWS]->(c) \
         WITH a, c WHERE c.name = 'c1' \
         OPTIONAL MATCH (a)-[r:KNOWS]->(c) \
         RETURN c.name, r IS NULL",
    );
    assert_eq!(name_and_is_null(&rows), vec![("c1".to_string(), false)]);
}

#[test]
fn normal_new_target_optional_expand_still_fans_out() {
    let rows = execute(
        "MATCH (a:A) \
         OPTIONAL MATCH (a)-[r:KNOWS]->(x) \
         RETURN x.name ORDER BY x.name",
    );
    assert_eq!(string_col(&rows), vec!["b".to_string(), "c1".to_string()]);
}
