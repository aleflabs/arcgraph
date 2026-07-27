//! **#796 / ADR-038 amendment-12** — permissive dynamic-schema binding:
//! an unknown label / rel-type ⇒ EMPTY match (openCypher "unknown ⇒ empty"),
//! NOT a `-32005` `UnknownLabel`/`UnknownRelType` error, END-TO-END.
//!
//! # ADR-133 §D-4 "Query" active-verification gate
//!
//! Every assertion drives a REAL ArcQL query through the FULL pipeline
//! (`QueryEngine::execute`) against a POPULATED substrate, and asserts the
//! returned rows == a hand-computed oracle — the discriminating pin is that
//! an unknown label returns EMPTY (matches NOTHING), NOT all nodes (the wrong
//! "drop the constraint" semantics) and NOT an error (the pre-amendment
//! `-32005`). Known labels/rel-types are unaffected.

use arcgraph_core::{LabelId, NodeId, RelId, TenantId, TypeId};
use arcgraph_query::QueryEngine;
use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::executor::value::{NodeView, RelView, Value};
use arcgraph_query::semantic::StubCatalogProvider;

const PERSON: u32 = 1;
const KNOWS: u32 = 1;

fn person(id: u64, name: &str) -> NodeView {
    NodeView::new(NodeId::new(id), Some(LabelId::new(PERSON)))
        .with_property("name", Value::String(name.to_string()))
}

/// A 3-person graph (Alice, Bob, Carol) with Alice-[:KNOWS]->Bob.
fn graph() -> StubExecutorSubstrate {
    StubExecutorSubstrate::new()
        .with_node(TenantId::DEFAULT, person(1, "Alice"))
        .with_node(TenantId::DEFAULT, person(2, "Bob"))
        .with_node(TenantId::DEFAULT, person(3, "Carol"))
        .with_edge(
            TenantId::DEFAULT,
            RelView::new(
                RelId::new(1),
                NodeId::new(1),
                NodeId::new(2),
                Some(TypeId::new(KNOWS)),
            ),
        )
}

/// Catalog knows `Person`/`KNOWS`/`name` but NOT `Ghost`/`GHOSTREL`.
fn catalog() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["Person"])
        .with_rel_types(["KNOWS"])
        .with_properties(["name"])
}

fn names(cypher: &str) -> Vec<String> {
    let rows = QueryEngine::new(&catalog())
        .execute(cypher, &graph())
        .expect("execute must not error (permissive binding)")
        .rows;
    let mut out: Vec<String> = rows
        .into_iter()
        .map(|r| match &r[0] {
            Value::String(s) => s.clone(),
            other => panic!("expected String, got {other:?}"),
        })
        .collect();
    out.sort();
    out
}

// =====================================================================
// PART A — KNOWN label/rel-type unaffected (active verification: rows ==
// hand-computed oracle).
// =====================================================================

#[test]
fn known_label_returns_matching_nodes() {
    // Oracle: all 3 Person nodes.
    assert_eq!(
        names("MATCH (a:Person) RETURN a.name"),
        vec!["Alice".to_string(), "Bob".to_string(), "Carol".to_string()]
    );
}

#[test]
fn known_rel_type_traverses() {
    // Oracle: Alice-[:KNOWS]->Bob ⇒ {Bob}.
    assert_eq!(
        names("MATCH (a:Person)-[:KNOWS]->(b) RETURN b.name"),
        vec!["Bob".to_string()]
    );
}

// =====================================================================
// PART B — UNKNOWN label/rel-type ⇒ EMPTY (the correctness pin). These
// FAIL (rows == all-nodes) under the wrong "drop the constraint" fix, and
// ERROR under the pre-amendment strict binding.
// =====================================================================

#[test]
fn unknown_label_is_empty_not_all_nodes() {
    // Oracle: EMPTY — no node carries label `Ghost`. (NOT the 3 Person nodes.)
    assert_eq!(names("MATCH (a:Ghost) RETURN a.name"), Vec::<String>::new());
}

#[test]
fn unknown_rel_type_is_empty() {
    // Oracle: EMPTY — no relationship carries type `GHOSTREL`.
    assert_eq!(
        names("MATCH (a:Person)-[:GHOSTREL]->(b) RETURN b.name"),
        Vec::<String>::new()
    );
}

#[test]
fn unknown_label_does_not_error() {
    // The pre-amendment behaviour raised `-32005`; permissive binding must
    // return Ok (empty), never Err.
    let r = QueryEngine::new(&catalog()).execute("MATCH (a:Ghost) RETURN a", &graph());
    assert!(r.is_ok(), "unknown label must NOT error (#796): {r:?}");
    assert_eq!(r.unwrap().rows.len(), 0);
}

// =====================================================================
// PART C — #796 verbatim: OPTIONAL MATCH over an unknown label forwards
// null then the dependent MATCH yields empty — no error. (TCK With1 [5]
// "Forwarding null".)
// =====================================================================

#[test]
fn cz796_optional_match_unknown_label_forwards_empty() {
    let r = QueryEngine::new(&catalog()).execute(
        "OPTIONAL MATCH (a:Ghost)\nWITH a\nMATCH (a)-->(b)\nRETURN *",
        &graph(),
    );
    assert!(
        r.is_ok(),
        "#796 OPTIONAL MATCH over unknown label must not error: {r:?}"
    );
    assert_eq!(r.unwrap().rows.len(), 0, "empty result, no -32005");
}

// =====================================================================
// PART D — cold-start: EMPTY database (no nodes) + unknown label. The
// #796 cold-start/empty-db repro — must return empty, not -32005.
// =====================================================================

#[test]
fn cz796_cold_start_empty_db_unknown_label() {
    let empty = StubExecutorSubstrate::new();
    let r = QueryEngine::new(&StubCatalogProvider::new()).execute(
        "MATCH (a:Account) WHERE a.balance > 50000 RETURN a.id",
        &empty,
    );
    assert!(
        r.is_ok(),
        "cold-start unknown label must not error (#796): {r:?}"
    );
    assert_eq!(r.unwrap().rows.len(), 0);
}
