//! W27-β ADR-153 — `graph.raw_query` CREATE node end-to-end integration.
//!
//! Closes audit-2026-05-27 finding "No write-op exposed via MCP
//! `graph.raw_query`" for the Phase 1 / ADR-147 surface. The 4 W26-θ
//! Phase 1 e2e tests at `mcp_create_node_e2e.rs` exercised the
//! executor → substrate path via `Pipeline::build` directly; the test
//! file below routes the same shape through the JSON-RPC dispatcher +
//! `raw_query_tool` boundary so the wire-shape pin is load-bearing.
//!
//! Per ADR-153 §D-1 the tool admits any ArcQL clause that parses;
//! §D-2 ratifies the `writes: WriteSummary` envelope shape;
//! §D-3 establishes the read-write composition contract
//! (`CREATE (n) ... RETURN n` returns BOTH the writes counters and the
//! RETURN rows in a single response).

#![allow(clippy::unwrap_used)]

mod raw_query_write_common;
use raw_query_write_common::{assert_writes, fresh_dispatcher, parse_body, raw_query};

#[test]
fn create_node_returns_one_row_and_writes_nodes_created_one() {
    // The canonical ADR-153 §D-3 read-write composition demo:
    // `CREATE (n:User) RETURN n` returns ONE RETURN row + a writes
    // summary with nodes_created = 1.
    let d = fresh_dispatcher();
    let resp = raw_query(&d, "CREATE (n:User) RETURN n");
    let body = parse_body(&resp);
    assert_eq!(body["row_count"], 1, "CREATE...RETURN emits 1 row: {body}");
    assert_writes(&body, (1, 0, 0, 0, 0, 0, 0, 0));
}

#[test]
fn create_node_anonymous_emits_one_row_and_writes_one() {
    // The anonymous CREATE (`CREATE (:User)`) still emits one row per
    // ADR-147 §D-7 (the openCypher "1 node created" signal) and ticks
    // writes.nodes_created exactly once.
    let d = fresh_dispatcher();
    let resp = raw_query(&d, "CREATE (:User)");
    let body = parse_body(&resp);
    // CreateNodeOp emits one row (anonymous = 0-column tuple per
    // ADR-147 §D-7 + the create_node.rs test pin).
    assert_eq!(body["row_count"], 1);
    assert_writes(&body, (1, 0, 0, 0, 0, 0, 0, 0));
}

#[test]
fn create_node_with_literal_properties_persists_via_raw_query_surface() {
    // ADR-147 §D-4 admits literal property bags. Post-ADR-152 the
    // substrate PERSISTS the bag (`PropertyData::Blob` via the
    // BlobStore chain — the round-trip is asserted in
    // `mcp_property_persistence_e2e`); the v1.2 strict-schema property
    // TYPING remains forward-pinned to issue #356. This test asserts the
    // write reaches the substrate: row_count + nodes_created = 1.
    let d = fresh_dispatcher();
    let resp = raw_query(
        &d,
        r#"CREATE (n:User {id: 42, name: "alice", flag: TRUE}) RETURN n"#,
    );
    let body = parse_body(&resp);
    assert_eq!(body["row_count"], 1);
    assert_writes(&body, (1, 0, 0, 0, 0, 0, 0, 0));
}

#[test]
fn three_creates_sum_to_nodes_created_three_when_three_statements_run() {
    // ADR-153 v1.0-α posture: each `graph.raw_query` invocation is its
    // OWN transaction (one statement, one tx, one envelope). Multi-
    // statement batching is forward-pinned per ADR-153 §"Forward-
    // deferred". Three independent calls SHOULD show nodes_created = 1
    // each time, not 3 in a single response.
    let d = fresh_dispatcher();
    for label in &["User", "User", "Article"] {
        let q = format!("CREATE (n:{label}) RETURN n");
        let body = parse_body(&raw_query(&d, &q));
        assert_eq!(body["row_count"], 1, "each CREATE → 1 row");
        assert_writes(&body, (1, 0, 0, 0, 0, 0, 0, 0));
    }
}

#[test]
fn pure_read_after_create_observes_zero_writes_in_summary() {
    // ADR-153 §D-2 zero-counter pin for read traffic.
    // Sequence: CREATE x3 → MATCH x1. The MATCH's response carries
    // zero writes; the CREATE responses each carried one.
    let d = fresh_dispatcher();
    for _ in 0..3 {
        let _ = parse_body(&raw_query(&d, "CREATE (n:User) RETURN n"));
    }
    let body = parse_body(&raw_query(&d, "MATCH (n:User) RETURN n"));
    assert_eq!(body["row_count"], 3, "MATCH observes the three CREATEs");
    assert_writes(&body, (0, 0, 0, 0, 0, 0, 0, 0));
}
