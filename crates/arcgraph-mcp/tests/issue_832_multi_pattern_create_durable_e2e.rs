//! Issue #832 — multi-pattern CREATE silent data loss, DURABLE path.
//!
//! The bug was reported by CZ over a live `arcgraph serve --bolt --data`
//! (durable storage) with the neo4j driver 6.2.0 — a
//! `langchain_neo4j.Neo4jGraph.add_graph_documents` KG build emits
//! multi-node CREATEs and silently lost all but the last node.
//!
//! This suite drives the EXACT customer signature through the MCP
//! `graph.raw_query` boundary over the production-shaped durable backend
//! (`CrudStore` + `PrimaryIndex` + `crud::commit` WAL path — the same
//! engine `serve --bolt --data` runs, with `InMemoryPageIo` so the test
//! is hermetic). The `writes.nodes_created` summary is the durable
//! oracle: the bug reported `1`, the fix reports `3`.
//!
//! Root cause (bisected) is in `arcgraph-query` LOWERING
//! (`lower_create` discarded all but the last CREATE item); storage is
//! exonerated. This durable test proves the fix end-to-end on the path
//! the bug was actually reported on.
//!
//! RED-on-revert: revert the `lower_create` chain wiring and
//! `nodes_created` collapses to 1 / the MATCH row_count to 1.

#![allow(clippy::unwrap_used)]

mod raw_query_write_common;
use raw_query_write_common::{assert_writes, fresh_dispatcher, parse_body, raw_query};

#[test]
fn multi_pattern_create_persists_all_three_durable() {
    let d = fresh_dispatcher();

    // The exact CZ query: 3 comma-separated node patterns in ONE clause.
    let create = parse_body(&raw_query(&d, "CREATE (:T {n: 1}),(:T {n: 2}),(:T {n: 3})"));
    // Durable WriteSummary oracle: nodes_created MUST be 3 (the bug
    // reported 1). Properties set at CREATE time do NOT tick
    // properties_set (that counter is for the SET clause).
    assert_eq!(
        create["writes"]["nodes_created"], 3,
        "durable writes.nodes_created MUST be 3 — the #832 bug reported \
         1 (silent data loss): {create:?}"
    );

    // Read-back through a real MATCH against durable storage: count=3.
    let read = parse_body(&raw_query(&d, "MATCH (t:T) RETURN t.n"));
    assert_eq!(
        read["row_count"], 3,
        "MATCH (t:T) RETURN t.n MUST see 3 durably-persisted nodes \
         (count(t)=3; the bug returned 1): {read:?}"
    );

    // collect(t.n) == {1,2,3}, not [3]. Extract the projected values.
    let rows = read["rows"].as_array().expect("rows array");
    let mut vals: Vec<i64> = rows
        .iter()
        .map(|r| {
            // Each row is an array of projected cells; t.n is cell 0.
            r.as_array()
                .and_then(|cells| cells.first())
                .and_then(serde_json::Value::as_i64)
                .unwrap_or_else(|| panic!("row cell t.n is not an integer: {r:?}"))
        })
        .collect();
    vals.sort_unstable();
    assert_eq!(
        vals,
        vec![1, 2, 3],
        "collect(t.n) MUST be {{1,2,3}} durably — the bug yielded [3]"
    );
}

#[test]
fn multi_pattern_create_no_props_clean_write_summary_durable() {
    // No-property variant pins the FULL durable WriteSummary tuple:
    // 3 nodes created, every other counter zero.
    let d = fresh_dispatcher();
    let create = parse_body(&raw_query(&d, "CREATE (:T),(:T),(:T)"));
    assert_writes(&create, (3, 0, 0, 0, 0, 0, 0, 0));

    let read = parse_body(&raw_query(&d, "MATCH (t:T) RETURN t"));
    assert_eq!(read["row_count"], 3, "3 durable nodes: {read:?}");
}

#[test]
fn multi_path_create_persists_both_paths_durable() {
    // Sister pattern (same root cause): two comma-separated PATHS.
    // 4 endpoint nodes + 2 rels durably; the bug kept only the last
    // path (2 nodes, 1 rel).
    let d = fresh_dispatcher();
    let create = parse_body(&raw_query(
        &d,
        "CREATE (:A {n: 1})-[:R]->(:B {n: 2}),(:A {n: 3})-[:R]->(:B {n: 4})",
    ));
    assert_eq!(
        create["writes"]["nodes_created"], 4,
        "durable nodes_created MUST be 4 (both paths): {create:?}"
    );
    assert_eq!(
        create["writes"]["rels_created"], 2,
        "durable rels_created MUST be 2 (both paths): {create:?}"
    );

    // Both A endpoints observable durably.
    let read = parse_body(&raw_query(&d, "MATCH (a:A) RETURN a"));
    assert_eq!(read["row_count"], 2, "2 durable :A nodes: {read:?}");
}

#[test]
fn multi_pattern_create_bound_return_durable() {
    // CREATE (a),(b),(c) RETURN a,b,c — every chained binding in scope.
    let d = fresh_dispatcher();
    let create = parse_body(&raw_query(
        &d,
        "CREATE (a:T {n: 1}),(b:T {n: 2}),(c:T {n: 3}) RETURN a, b, c",
    ));
    assert_eq!(
        create["writes"]["nodes_created"], 3,
        "durable nodes_created MUST be 3: {create:?}"
    );
    assert_eq!(
        create["row_count"], 1,
        "CREATE …,…,… RETURN a,b,c emits ONE row binding all three: \
         {create:?}"
    );
}

#[test]
fn single_pattern_create_still_one_durable() {
    // Guard against over-correction: a single-pattern CREATE persists
    // exactly one node durably (unchanged behavior).
    let d = fresh_dispatcher();
    let create = parse_body(&raw_query(&d, "CREATE (:T {n: 99})"));
    assert_writes(&create, (1, 0, 0, 0, 0, 0, 0, 0));
    let read = parse_body(&raw_query(&d, "MATCH (t:T) RETURN t"));
    assert_eq!(read["row_count"], 1, "exactly one durable node: {read:?}");
}
