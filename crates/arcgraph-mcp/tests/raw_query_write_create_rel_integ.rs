//! W27-β ADR-153 — `graph.raw_query` CREATE relationship integration.
//!
//! Closes audit-2026-05-27 finding for the Phase 2 / ADR-148 surface.
//! Counterpart to `raw_query_write_create_node_integ.rs` for the
//! `CREATE (a)-[r:TYPE]->(b)` clause family. Each CREATE-rel
//! statement creates TWO nodes + ONE rel; the writes summary
//! aggregates accordingly per ADR-153 §D-2 + ADR-148 §"Counting
//! semantics".

#![allow(clippy::unwrap_used)]

mod raw_query_write_common;
use raw_query_write_common::{assert_writes, fresh_dispatcher, parse_body, raw_query};

#[test]
fn create_relationship_creates_two_nodes_and_one_rel_in_one_statement() {
    // ADR-148 §"Counting semantics": a single CREATE-rel statement
    // creates two nodes + one rel because the source + target patterns
    // are CREATE-shaped (NOT MATCH-shaped). The writes summary surfaces
    // nodes_created=2, rels_created=1.
    let d = fresh_dispatcher();
    let resp = raw_query(&d, "CREATE (a:User)-[r:FOLLOWS]->(b:User) RETURN r");
    let body = parse_body(&resp);
    // CreateRelOp emits one row binding the rel to `r`.
    assert_eq!(body["row_count"], 1, "CREATE-rel...RETURN emits 1 row");
    assert_writes(&body, (2, 0, 1, 0, 0, 0, 0, 0));
}

#[test]
fn create_relationship_with_literal_properties_through_raw_query() {
    // ADR-148 §D-4 property posture inherits ADR-147 §D-4: literal
    // bags on rel + endpoints. Substrate stores PropertyData::Empty at
    // this layer. The writes summary still surfaces
    // 2 nodes + 1 rel created (property typing is forward-deferred but
    // the CREATE call boundaries are intact).
    let d = fresh_dispatcher();
    let resp = raw_query(
        &d,
        r#"CREATE (a:User {id: 1})-[r:KNOWS {since: 2024}]->(b:User {id: 2}) RETURN r"#,
    );
    let body = parse_body(&resp);
    assert_eq!(body["row_count"], 1);
    assert_writes(&body, (2, 0, 1, 0, 0, 0, 0, 0));
}

#[test]
fn create_rel_anonymous_target_still_creates_both_endpoints() {
    // The target pattern omits its variable binding; CreateRelOp still
    // creates the target node (and the rel still ticks rels_created).
    // Pin: anonymous endpoints don't escape counting.
    let d = fresh_dispatcher();
    let resp = raw_query(&d, "CREATE (a:User)-[r:FOLLOWS]->(:User) RETURN r");
    let body = parse_body(&resp);
    assert_eq!(body["row_count"], 1);
    assert_writes(&body, (2, 0, 1, 0, 0, 0, 0, 0));
}

#[test]
fn three_rel_creations_aggregate_per_statement_not_across_envelopes() {
    // ADR-153 v1.0-α: one statement = one tx = one envelope. Three
    // independent CREATE-rel statements each report (2, 0, 1, ...).
    let d = fresh_dispatcher();
    for _ in 0..3 {
        let body = parse_body(&raw_query(
            &d,
            "CREATE (a:User)-[r:FOLLOWS]->(b:User) RETURN r",
        ));
        assert_writes(&body, (2, 0, 1, 0, 0, 0, 0, 0));
    }
    // After 3 CREATE-rel statements, the per-tenant store holds 6
    // nodes + 3 rels — verify by reading. (Read-side observes the
    // accumulated state but writes summary stays zero.)
    let body = parse_body(&raw_query(&d, "MATCH (n:User) RETURN n"));
    assert_eq!(body["row_count"], 6, "6 User nodes across 3 statements");
    assert_writes(&body, (0, 0, 0, 0, 0, 0, 0, 0));
}
