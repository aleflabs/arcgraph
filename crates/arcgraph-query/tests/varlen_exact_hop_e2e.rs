//! #939: bare openCypher variable length `*N` means exactly N hops.

use arcgraph_core::{NodeId, RelId, TenantId, TypeId};
use arcgraph_query::QueryEngine;
use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::executor::value::{NodeView, RelView, Value};
use arcgraph_query::semantic::StubCatalogProvider;

const TYPE_R: u32 = 1;

fn catalog() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_rel_types(["R"])
        .with_properties(["id"])
}

fn chain_a_to_e() -> StubExecutorSubstrate {
    let mut s = StubExecutorSubstrate::new();
    for (id, name) in [(1, "a"), (2, "b"), (3, "c"), (4, "d"), (5, "e")] {
        s = s.with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(id), None).with_property("id", Value::String(name.into())),
        );
    }
    for (rel, from, to) in [(10, 1, 2), (11, 2, 3), (12, 3, 4), (13, 4, 5)] {
        s = s.with_edge(
            TenantId::DEFAULT,
            RelView::new(
                RelId::new(rel),
                NodeId::new(from),
                NodeId::new(to),
                Some(TypeId::new(TYPE_R)),
            ),
        );
    }
    s
}

fn ids(query: &str) -> Vec<String> {
    let s = chain_a_to_e();
    let cat = catalog();
    let rows = QueryEngine::new(&cat)
        .execute(query, &s)
        .expect("execute")
        .rows()
        .to_vec();
    let mut ids: Vec<String> = rows
        .iter()
        .map(|row| match row.first().expect("non-empty row") {
            Value::String(id) => id.clone(),
            other => panic!("expected String x.id, got {other:?}"),
        })
        .collect();
    ids.sort();
    ids
}

#[test]
fn bare_star_n_is_exact_hop_count() {
    assert_eq!(ids("MATCH (a {id:'a'})-[:R*1]->(x) RETURN x.id"), vec!["b"]);
    assert_eq!(ids("MATCH (a {id:'a'})-[:R*2]->(x) RETURN x.id"), vec!["c"]);
    assert_eq!(ids("MATCH (a {id:'a'})-[:R*3]->(x) RETURN x.id"), vec!["d"]);
}

#[test]
fn explicit_open_and_bounded_ranges_still_work() {
    assert_eq!(
        ids("MATCH (a {id:'a'})-[:R*1..]->(x) RETURN x.id"),
        vec!["b", "c", "d", "e"]
    );
    assert_eq!(
        ids("MATCH (a {id:'a'})-[:R*2..3]->(x) RETURN x.id"),
        vec!["c", "d"]
    );
}
