//! #949 — anonymous relationship inline property predicates must filter.
//!
//! `MATCH (a)-[:R {w: 1}]->(x)` is valid openCypher. The relationship is
//! anonymous to the user, but lowering still needs an internal binding so
//! the existing property-filter predicate can inspect the expanded rel.

use arcgraph_core::{LabelId, NodeId, RelId, TenantId, TypeId};
use arcgraph_query::QueryEngine;
use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::executor::value::{NodeView, RelView, Value};
use arcgraph_query::semantic::StubCatalogProvider;

const NODE_LABEL: u32 = 1;
const R_TYPE: u32 = 1;

fn catalog() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["N"])
        .with_rel_types(["R"])
        .with_properties(["id", "w", "k"])
}

fn node(id: u64) -> NodeView {
    NodeView::new(NodeId::new(id), Some(LabelId::new(NODE_LABEL)))
        .with_property("id", Value::String(format!("n{id}")))
}

fn rel(id: u64, from: u64, to: u64, w: i64, k: &str) -> RelView {
    RelView::new(
        RelId::new(id),
        NodeId::new(from),
        NodeId::new(to),
        Some(TypeId::new(R_TYPE)),
    )
    .with_property("w", Value::Integer(w))
    .with_property("k", Value::String(k.to_string()))
}

fn graph() -> StubExecutorSubstrate {
    StubExecutorSubstrate::new()
        .with_node(TenantId::DEFAULT, node(1))
        .with_node(TenantId::DEFAULT, node(2))
        .with_node(TenantId::DEFAULT, node(3))
        .with_node(TenantId::DEFAULT, node(4))
        .with_edge(TenantId::DEFAULT, rel(10, 1, 2, 1, "x"))
        .with_edge(TenantId::DEFAULT, rel(11, 1, 3, 2, "y"))
        .with_edge(TenantId::DEFAULT, rel(12, 1, 4, 1, "z"))
}

fn ids(cypher: &str) -> Vec<String> {
    let result = QueryEngine::new(&catalog())
        .execute(cypher, &graph())
        .unwrap_or_else(|e| panic!("execute must not error for `{cypher}`: {e:?}"));
    let mut out: Vec<String> = result
        .rows
        .into_iter()
        .map(|row| {
            assert_eq!(row.len(), 1, "expected one projected column for `{cypher}`");
            match &row[0] {
                Value::String(s) => s.clone(),
                other => panic!("expected string id for `{cypher}`, got {other:?}"),
            }
        })
        .collect();
    out.sort();
    out
}

#[test]
fn anonymous_rel_inline_property_w_1_filters_to_matching_targets() {
    assert_eq!(
        ids("MATCH (a {id: 'n1'})-[:R {w: 1}]->(x) RETURN x.id"),
        vec!["n2".to_string(), "n4".to_string()],
        "anonymous rel inline property w=1 must not be silently dropped",
    );
}

#[test]
fn anonymous_rel_inline_property_w_2_filters_to_matching_target() {
    assert_eq!(
        ids("MATCH (a {id: 'n1'})-[:R {w: 2}]->(x) RETURN x.id"),
        vec!["n3".to_string()],
        "anonymous rel inline property w=2 must select only the w=2 edge",
    );
}

#[test]
fn named_rel_inline_property_still_filters() {
    assert_eq!(
        ids("MATCH (a {id: 'n1'})-[r:R {w: 1}]->(x) RETURN x.id"),
        vec!["n2".to_string(), "n4".to_string()],
        "named rel inline properties must keep their existing behavior",
    );
}

#[test]
fn anonymous_rel_without_properties_still_matches_all_r_edges() {
    assert_eq!(
        ids("MATCH (a {id: 'n1'})-[:R]->(x) RETURN x.id"),
        vec!["n2".to_string(), "n3".to_string(), "n4".to_string()],
        "anonymous rel without inline properties must not get a spurious filter",
    );
}

#[test]
fn anonymous_rel_multi_property_filter_requires_all_properties() {
    assert_eq!(
        ids("MATCH (a {id: 'n1'})-[:R {w: 1, k: 'x'}]->(x) RETURN x.id"),
        vec!["n2".to_string()],
        "anonymous rel multi-property inline predicate must be an AND filter",
    );
}
