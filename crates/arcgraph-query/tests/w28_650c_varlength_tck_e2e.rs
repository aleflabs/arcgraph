//! ADR-133 active end-to-end verification for #650-C openCypher v9 §3
//! variable-length path execution (ADR-186).
//!
//! Golden row-multiset oracles mirroring the vendored openCypher TCK
//! `clauses/match/Match5.feature` — "Match variable length patterns
//! over given graphs scenarios". The TCK graph (that file's
//! `Background`) is a 4-level binary `:LIKES` tree rooted at `n0:A`;
//! we reproduce it as a `StubExecutorSubstrate` fixture with
//! deterministic NodeId / RelId assignment, then drive the FULL
//! `QueryEngine` pipeline (parse → bind → type-check → lower →
//! enumerate → execute) and assert the exact result-multiset with a
//! strong `==` oracle.
//!
//! Each test cites the TCK scenario it pins. The TCK form uses two
//! `MATCH` clauses (`MATCH (a:A) MATCH (a)-[...]->(c)`); we use the
//! semantically-identical single-clause form `MATCH (a:A)-[...]->(c)`
//! (same graph, same result-multiset) — noted per `feedback_cite_
//! correctness_not_just_resolution.md`.
//!
//! Cite-correctness verified against the real file at HEAD:
//! - [1]  `*`     (unbounded)      → Match5.feature:69
//! - [6]  `*0..2` (zero-bounded)   → Match5.feature:159
//! - [7]  `*1..2` (bounded)        → Match5.feature:177
//!
//! The cap-exceeded honesty pin (RC-1) is NOT a TCK scenario — it is
//! ArcGraph's explicit `*N..` depth cap (ADR-186 RC-1): a traversal
//! that would extend past the cap ERRORS rather than silently
//! truncating.

use arcgraph_core::{LabelId, NodeId, RelId, TenantId, TypeId};
use arcgraph_query::QueryEngine;
use arcgraph_query::executor::value::{NodeView, RelView};
use arcgraph_query::executor::{StubExecutorSubstrate, Value};
use arcgraph_query::semantic::StubCatalogProvider;

/// The Match5.feature `Background` graph: a 4-level binary `:LIKES`
/// tree rooted at `n0:A`. Labels A/B/C/D = LabelId 1/2/3/4; LIKES =
/// TypeId 1. NodeId/RelId assigned deterministically.
fn tck_binary_tree() -> StubExecutorSubstrate {
    // (NodeId, name, LabelId) — A=1, B=2, C=3, D=4.
    let nodes: &[(u64, &str, u32)] = &[
        (1, "n0", 1),
        (2, "n00", 2),
        (3, "n01", 2),
        (4, "n000", 3),
        (5, "n001", 3),
        (6, "n010", 3),
        (7, "n011", 3),
        (8, "n0000", 4),
        (9, "n0001", 4),
        (10, "n0010", 4),
        (11, "n0011", 4),
        (12, "n0100", 4),
        (13, "n0101", 4),
        (14, "n0110", 4),
        (15, "n0111", 4),
    ];
    // (RelId, src, dst) — the LIKES edges, in the TCK's written order.
    let edges: &[(u64, u64, u64)] = &[
        (100, 1, 2),
        (101, 1, 3),
        (102, 2, 4),
        (103, 2, 5),
        (104, 3, 6),
        (105, 3, 7),
        (106, 4, 8),
        (107, 4, 9),
        (108, 5, 10),
        (109, 5, 11),
        (110, 6, 12),
        (111, 6, 13),
        (112, 7, 14),
        (113, 7, 15),
    ];
    let mut s = StubExecutorSubstrate::new();
    for &(id, name, label) in nodes {
        s = s.with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(id), Some(LabelId::new(label)))
                .with_property("name", Value::String(name.to_owned())),
        );
    }
    for &(rel, src, dst) in edges {
        s = s.with_edge(
            TenantId::DEFAULT,
            RelView::new(
                RelId::new(rel),
                NodeId::new(src),
                NodeId::new(dst),
                Some(TypeId::new(1)),
            ),
        );
    }
    s
}

fn tck_catalog() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["A", "B", "C", "D"])
        .with_rel_types(["LIKES"])
        .with_properties(["name"])
}

/// Sorted `c.name` string column of the result set (the strong `==`
/// oracle key — order-independent multiset comparison).
fn names(rows: &[Vec<Value>]) -> Vec<String> {
    let mut v: Vec<String> = rows
        .iter()
        .map(|r| match r.first().expect("non-empty row") {
            Value::String(s) => s.clone(),
            other => panic!("expected String c.name, got {other:?}"),
        })
        .collect();
    v.sort();
    v
}

#[test]
fn tck_match5_scenario_7_bounded_1_2() {
    // Match5.feature [7] "Handling upper and lower bounded variable
    // length match 2": (a:A)-[:LIKES*1..2]->(c) ⇒ the 1-hop children
    // {n00,n01} + the 2-hop grandchildren {n000,n001,n010,n011}.
    let s = tck_binary_tree();
    let cat = tck_catalog();
    let rows = QueryEngine::new(&cat)
        .execute("MATCH (a:A)-[:LIKES*1..2]->(c) RETURN c.name", &s)
        .expect("execute *1..2");
    assert_eq!(
        names(rows.rows()),
        vec!["n00", "n000", "n001", "n01", "n010", "n011"],
        "Match5.feature [7] *1..2 golden multiset"
    );
}

#[test]
fn tck_match5_scenario_6_zero_bounded_0_2() {
    // Match5.feature [6] "Handling upper and lower bounded variable
    // length match 1": (a:A)-[:LIKES*0..2]->(c) ⇒ the [7] set PLUS the
    // depth-0 identity path (the start node n0 itself).
    let s = tck_binary_tree();
    let cat = tck_catalog();
    let rows = QueryEngine::new(&cat)
        .execute("MATCH (a:A)-[:LIKES*0..2]->(c) RETURN c.name", &s)
        .expect("execute *0..2");
    assert_eq!(
        names(rows.rows()),
        vec!["n0", "n00", "n000", "n001", "n01", "n010", "n011"],
        "Match5.feature [6] *0..2 golden multiset (includes the *0 start)"
    );
}

#[test]
fn tck_match5_scenario_1_unbounded_within_cap() {
    // Match5.feature [1] "Handling unbounded variable length match":
    // (a:A)-[:LIKES*]->(c). The tree depth is 3 (≤ the depth cap of
    // 5), so the unbounded traversal returns COMPLETE results — all 14
    // descendants — and does NOT error. (A dead-end frontier reached
    // strictly before the cap is complete, not truncated: RC-1.)
    let s = tck_binary_tree();
    let cat = tck_catalog();
    let rows = QueryEngine::new(&cat)
        .execute("MATCH (a:A)-[:LIKES*]->(c) RETURN c.name", &s)
        .expect("execute * (unbounded, within cap)");
    assert_eq!(rows.len(), 14, "all 14 descendants enumerated");
    assert_eq!(
        names(rows.rows()),
        vec![
            "n00", "n000", "n0000", "n0001", "n001", "n0010", "n0011", "n01", "n010", "n0100",
            "n0101", "n011", "n0110", "n0111",
        ],
        "Match5.feature [1] unbounded golden multiset"
    );
}

#[test]
fn varlength_depth_cap_exceeded_errors_not_truncates() {
    // ADR-186 RC-1 honesty pin (NOT a TCK scenario): an unbounded `*`
    // traversal whose paths would extend PAST the depth cap (5)
    // surfaces a STRUCTURED error rather than a silently-truncated
    // (wrong) result set. Fixture: a 6-edge `:LIKES` chain (node 1 =
    // A; the start reaches a node still carrying an outgoing edge at
    // depth 5).
    let mut s = StubExecutorSubstrate::new().with_node(
        TenantId::DEFAULT,
        NodeView::new(NodeId::new(1), Some(LabelId::new(1)))
            .with_property("name", Value::String("c1".to_owned())),
    );
    for i in 2..=7u64 {
        s = s.with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(i), Some(LabelId::new(2)))
                .with_property("name", Value::String(format!("c{i}"))),
        );
    }
    for i in 1..=6u64 {
        s = s.with_edge(
            TenantId::DEFAULT,
            RelView::new(
                RelId::new(i * 10),
                NodeId::new(i),
                NodeId::new(i + 1),
                Some(TypeId::new(1)),
            ),
        );
    }
    let cat = tck_catalog();
    let r = QueryEngine::new(&cat).execute("MATCH (a:A)-[:LIKES*]->(c) RETURN c.name", &s);
    assert!(
        r.is_err(),
        "unbounded traversal past the depth cap MUST error, not truncate; got {r:?}"
    );
}
