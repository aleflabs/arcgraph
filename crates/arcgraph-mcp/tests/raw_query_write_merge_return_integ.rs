//! ADR-151-amendment-01 — RETURN-after-MERGE (node-shape) VALUE oracles
//! through the production `graph.raw_query` dispatcher.
//!
//! This is the ADR-133 §D-4 Query-class active-verification surface for
//! the amendment: it routes real `MERGE … RETURN n.<prop>` queries
//! through `Dispatcher::dispatch("graph.raw_query")` (the same
//! production path the MCP surface exposes) and asserts the **actual
//! returned cell value** (exact `==` vs an oracle), not a count.
//!
//! The pre-amendment MERGE surface was terminal (emitted 0 rows;
//! RETURN-after-MERGE was forward-pinned per ADR-151 §D-9). The
//! amendment lifts the **node-shape NAMED** case to v1.0-α: the single-
//! statement projection reads the MERGE's own in-memory binding row (no
//! substrate re-read), so it needs no statement-scoped batch tx (that
//! stays pinned for *multi-statement* read-your-writes). Path-shape +
//! anonymous merges stay terminal.
//!
//! RC-2 (the correctness-critical case): `ON CREATE SET` / `ON MATCH
//! SET` mutations are mirrored onto the emitted `NodeView` (single
//! source of truth with `set.rs`), so `MERGE … ON MATCH SET n.x = 2
//! RETURN n.x` returns `2` — not the stale/Null pre-SET bag.

#![allow(clippy::unwrap_used)]

mod raw_query_write_common;
use raw_query_write_common::{assert_writes, fresh_dispatcher, parse_body, raw_query};
use serde_json::json;

/// Extract the single returned cell `rows[0][0]` from a parsed body.
/// The body's `rows` are JSON arrays of cells per the
/// `value_to_json` wire bridge (`Integer → number`, `List → array`,
/// `Null → null`).
fn first_cell(body: &serde_json::Value) -> &serde_json::Value {
    &body["rows"][0][0]
}

#[test]
fn merge_create_branch_returns_property_value() {
    // ORACLE: `MERGE (n:User {id: 42}) RETURN n.id` → 42 on a fresh
    // tenant (create branch: User not interned → match miss → create →
    // the created NodeView carries the literal bag {id:42}).
    let d = fresh_dispatcher();
    let body = parse_body(&raw_query(&d, "MERGE (n:User {id: 42}) RETURN n.id"));
    assert_eq!(
        body["row_count"], 1,
        "RETURN-after-MERGE emits exactly one row: {body}"
    );
    assert_eq!(
        first_cell(&body),
        &json!(42),
        "create-branch n.id == 42 (exact): {body}"
    );
    assert_writes(&body, (1, 0, 0, 0, 0, 0, 0, 0));
}

#[test]
fn merge_match_branch_returns_same_value_no_second_create() {
    // ORACLE: a second identical MERGE hits the match branch → still
    // returns 42, and `nodes_created = 0` (idempotent — no duplicate).
    // Exercises the production property-bag match (ADR-152) + label
    // enforcement (ADR-152-amendment-01).
    let d = fresh_dispatcher();
    let first = parse_body(&raw_query(&d, "MERGE (n:User {id: 42}) RETURN n.id"));
    assert_eq!(first_cell(&first), &json!(42), "create-branch: {first}");
    assert_writes(&first, (1, 0, 0, 0, 0, 0, 0, 0));

    let second = parse_body(&raw_query(&d, "MERGE (n:User {id: 42}) RETURN n.id"));
    assert_eq!(
        second["row_count"], 1,
        "match branch emits the matched binding row: {second}"
    );
    assert_eq!(
        first_cell(&second),
        &json!(42),
        "match-branch n.id == 42 (exact): {second}"
    );
    assert_writes(&second, (0, 0, 0, 0, 0, 0, 0, 0));
}

#[test]
fn merge_on_match_set_returns_post_set_value() {
    // RC-2 — the make-or-break. `MERGE (n:User {id:1}) ON MATCH SET
    // n.x = 2 RETURN n.x`:
    //   First run  → create branch (on_match is dead) → n.x is Null.
    //   Second run → match branch → on_match fires → n.x mirrored to 2.
    let d = fresh_dispatcher();
    let first = parse_body(&raw_query(
        &d,
        "MERGE (n:User {id: 1}) ON MATCH SET n.x = 2 RETURN n.x",
    ));
    assert_eq!(first["row_count"], 1, "create branch emits a row: {first}");
    assert_eq!(
        first_cell(&first),
        &json!(null),
        "create branch: on_match dead → n.x is Null: {first}"
    );
    assert_writes(&first, (1, 0, 0, 0, 0, 0, 0, 0));

    let second = parse_body(&raw_query(
        &d,
        "MERGE (n:User {id: 1}) ON MATCH SET n.x = 2 RETURN n.x",
    ));
    assert_eq!(second["row_count"], 1, "match branch emits a row: {second}");
    assert_eq!(
        first_cell(&second),
        &json!(2),
        "RC-2 post-SET state: ON MATCH SET n.x = 2 → RETURN n.x == 2: {second}"
    );
    // Match branch (no create) + exactly one property set.
    assert_writes(&second, (0, 0, 0, 0, 1, 0, 0, 0));
}

#[test]
fn merge_on_create_set_returns_post_set_value() {
    // RC-2 — the create-fires-on_create path. `MERGE (n:User {id:5}) ON
    // CREATE SET n.y = 9 RETURN n.y` on a fresh tenant → create branch →
    // on_create fires → n.y mirrored to 9 in the emitted row.
    let d = fresh_dispatcher();
    let body = parse_body(&raw_query(
        &d,
        "MERGE (n:User {id: 5}) ON CREATE SET n.y = 9 RETURN n.y",
    ));
    assert_eq!(body["row_count"], 1, "create branch emits a row: {body}");
    assert_eq!(
        first_cell(&body),
        &json!(9),
        "RC-2 post-SET state: ON CREATE SET n.y = 9 → RETURN n.y == 9: {body}"
    );
    // Create branch + exactly one property set.
    assert_writes(&body, (1, 0, 0, 0, 1, 0, 0, 0));
}

#[test]
fn merge_create_branch_returns_list_property_exactly() {
    // ORACLE: `MERGE (n:User {tags: ["a","b"]}) RETURN n.tags` → the
    // exact list (composite List-of-scalars property per
    // ADR-152-amendment-02, round-tripped through the create-branch
    // NodeView bag).
    let d = fresh_dispatcher();
    let body = parse_body(&raw_query(
        &d,
        r#"MERGE (n:User {tags: ["a", "b"]}) RETURN n.tags"#,
    ));
    assert_eq!(body["row_count"], 1, "create branch emits a row: {body}");
    assert_eq!(
        first_cell(&body),
        &json!(["a", "b"]),
        "list property round-trips exactly: {body}"
    );
    assert_writes(&body, (1, 0, 0, 0, 0, 0, 0, 0));
}

#[test]
fn merge_anonymous_node_stays_terminal() {
    // RC-3 pin: an ANONYMOUS node merge `MERGE (:User)` has no binding
    // to project → terminal → row_count = 0 (contrast the named case →
    // 1). The create-branch side-effect still fires.
    let d = fresh_dispatcher();
    let body = parse_body(&raw_query(&d, "MERGE (:User)"));
    assert_eq!(
        body["row_count"], 0,
        "anonymous MERGE stays terminal (no binding to emit): {body}"
    );
    assert_writes(&body, (1, 0, 0, 0, 0, 0, 0, 0));
}

#[test]
fn merge_path_shape_stays_terminal() {
    // RC-3 pin: a PATH-shape merge → `output_binding` is None (the match
    // `[source, rel, target]` vs create `[rel]` schemas are
    // un-unionable) → terminal → row_count = 0. The create-branch still
    // mints source + target + rel.
    let d = fresh_dispatcher();
    let body = parse_body(&raw_query(&d, "MERGE (a:User)-[r:FOLLOWS]->(b:User)"));
    assert_eq!(
        body["row_count"], 0,
        "path-shape MERGE stays terminal: {body}"
    );
    // Create branch mints 2 nodes + 1 rel.
    assert_writes(&body, (2, 0, 1, 0, 0, 0, 0, 0));
}
