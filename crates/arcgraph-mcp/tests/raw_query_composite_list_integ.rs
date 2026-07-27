//! ADR-152-amendment-02 (W28) — `graph.raw_query` composite `List`-literal
//! end-to-end integration + the **ADR-133 active end-to-end verification**
//! (Query class) for this slice.
//!
//! Routes `CREATE (n:User {tags: [...]})` then `MATCH (n:User) RETURN
//! n.tags` through the production `Dispatcher::dispatch("graph.raw_query")`
//! (JSON-RPC boundary + per-tenant guard + scope check + the real
//! CrudExecutorSubstrate over the in-memory `CrudStore` JSON-blob path,
//! `InMemoryPageIo` — in-memory / non-durable page IO, NOT WAL-to-disk)
//! and asserts the returned JSON list `==` the oracle. The list literals
//! use the **spaced** form
//! (`["a", "b", "c"]`) so the test exercises the §D-5 grammar whitespace
//! fix end-to-end (the no-space form would have passed even at the
//! pre-amendment grammar).
//!
//! `Map` rejection is pinned through the same surface (honest forward-pin
//! per amendment-02 §D-2).

#![allow(clippy::unwrap_used)]

mod raw_query_write_common;
use raw_query_write_common::{fresh_dispatcher, parse_body, raw_query};
use serde_json::json;

#[test]
fn create_list_then_match_returns_list_via_raw_query() {
    // Active-verification core oracle (spaced form → exercises §D-5).
    let d = fresh_dispatcher();
    let create = parse_body(&raw_query(
        &d,
        r#"CREATE (n:User {tags: ["a", "b", "c"]}) RETURN n"#,
    ));
    assert_eq!(
        create["row_count"], 1,
        "CREATE … RETURN emits 1 row: {create}"
    );

    let body = parse_body(&raw_query(&d, "MATCH (n:User) RETURN n.tags"));
    assert_eq!(
        body["row_count"], 1,
        "MATCH observes the created node: {body}"
    );
    assert_eq!(
        body["rows"][0][0],
        json!(["a", "b", "c"]),
        "n.tags round-trips through the production graph.raw_query surface as the exact list: {body}"
    );
}

#[test]
fn nested_list_round_trips_via_raw_query() {
    let d = fresh_dispatcher();
    let _ = parse_body(&raw_query(
        &d,
        "CREATE (n:User {matrix: [[1, 2], [3]]}) RETURN n",
    ));
    let body = parse_body(&raw_query(&d, "MATCH (n:User) RETURN n.matrix"));
    assert_eq!(
        body["rows"][0][0],
        json!([[1, 2], [3]]),
        "nested list round-trips through the production surface: {body}"
    );
}

#[test]
fn heterogeneous_list_round_trips_via_raw_query() {
    let d = fresh_dispatcher();
    let _ = parse_body(&raw_query(
        &d,
        r#"CREATE (n:User {mixed: [1, "x", true]}) RETURN n"#,
    ));
    let body = parse_body(&raw_query(&d, "MATCH (n:User) RETURN n.mixed"));
    assert_eq!(
        body["rows"][0][0],
        json!([1, "x", true]),
        "heterogeneous list round-trips through the production surface: {body}"
    );
}

#[test]
fn empty_list_round_trips_via_raw_query() {
    let d = fresh_dispatcher();
    let _ = parse_body(&raw_query(&d, "CREATE (n:User {tags: []}) RETURN n"));
    let body = parse_body(&raw_query(&d, "MATCH (n:User) RETURN n.tags"));
    assert_eq!(
        body["rows"][0][0],
        json!([]),
        "empty-list value round-trips as an empty JSON array (not null): {body}"
    );
}

#[test]
fn map_property_value_rejected_via_raw_query() {
    // Honest forward-pin (amendment-02 §D-2): a `Map` property value
    // surfaces an error envelope through the production surface (it is
    // NOT silently coerced). The no-space form is used because the
    // map-in-property-bag whitespace quirk is out of this List slice
    // (map_literal whitespace deferred per §D-5) — `{k:1}` parses, then
    // rejects at the executor.
    let d = fresh_dispatcher();

    // Seed a valid User first so the `User` label is interned — the
    // dispatcher's binder hard-errors on an unknown label, so an
    // all-rejected workload would never reach the executor's Map gate.
    let seed = parse_body(&raw_query(&d, "CREATE (n:User {id: 1}) RETURN n"));
    assert_eq!(seed["row_count"], 1, "seed User created: {seed}");

    let resp = raw_query(&d, "CREATE (n:User {m: {k:1}}) RETURN n");
    assert!(
        !resp["error"].is_null(),
        "Map-literal property value must surface an error envelope (deferred per §D-2); resp={resp:?}"
    );

    // The rejected Map-CREATE persisted no node — only the seed remains
    // (ADR-031 commit-or-rollback discipline).
    let after = parse_body(&raw_query(&d, "MATCH (n:User) RETURN n"));
    assert_eq!(
        after["row_count"], 1,
        "a rejected Map-literal CREATE persists no node; only the seed User remains: {after}"
    );
}
