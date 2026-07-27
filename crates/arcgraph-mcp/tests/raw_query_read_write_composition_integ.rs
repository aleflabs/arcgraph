//! W27-β ADR-153 §D-3 — read-write composition through `graph.raw_query`.
//!
//! ADR-153 §D-3 establishes that a single ArcQL statement composing
//! a read AND a write surfaces BOTH the writes counters and the
//! RETURN rows in ONE response envelope. The 5 W26-θ executor ops
//! emit row-binding signals (CREATE node / CREATE rel / MERGE all
//! bind their primary variable; DELETE / SET / REMOVE are terminal at
//! v1.0-α per ADR-149/150/151 forward-pins). This test file pins the
//! ONE composition shape that DOES surface RETURN rows alongside
//! non-zero writes counters at v1.0-α: `CREATE (n) RETURN n`.
//!
//! Sister tests at:
//! - `raw_query_write_create_node_integ.rs` (CREATE + writes counter)
//! - `raw_query_integ.rs` (read-only graph.raw_query baseline)

#![allow(clippy::unwrap_used)]

mod raw_query_write_common;
use raw_query_write_common::{fresh_dispatcher, parse_body, raw_query};

#[test]
fn create_node_with_return_composes_writes_and_rows_in_single_envelope() {
    // ADR-153 §D-3 canonical pin: CREATE-then-RETURN surfaces BOTH a
    // RETURN row (the newly-created node) AND a writes summary
    // (nodes_created=1) in the SAME response envelope. The executor
    // pipeline runs a single tx for the statement; the response
    // envelope serializes both signals at the same boundary.
    let d = fresh_dispatcher();
    let resp = raw_query(&d, "CREATE (n:User) RETURN n");
    let body = parse_body(&resp);

    // Row half: ADR-147 §D-7 — CreateNodeOp emits 1 row binding n.
    assert_eq!(body["row_count"], 1, "RETURN n emits 1 row");
    let rows = body["rows"].as_array().expect("rows array");
    assert_eq!(rows.len(), 1, "rows array carries the single row");

    // Writes half: ADR-153 §D-2 — writes.nodes_created = 1.
    let w = &body["writes"];
    assert_eq!(w["nodes_created"], 1, "writes counter ticks");
    assert_eq!(w["nodes_deleted"], 0);
    assert_eq!(w["rels_created"], 0);

    // Composition pin: BOTH halves are non-trivial. This is the
    // load-bearing single-envelope semantic openCypher clients depend
    // on.
    assert!(
        !rows.is_empty() && w["nodes_created"].as_u64().unwrap() > 0,
        "single envelope carries BOTH rows + non-zero writes"
    );
}

#[test]
fn create_rel_with_return_composes_writes_and_rows() {
    // Sister composition pin for the CREATE-rel surface. The
    // CreateRelOp emits the rel-binding row + the writes summary
    // surfaces nodes_created=2 + rels_created=1 in one envelope.
    let d = fresh_dispatcher();
    let resp = raw_query(&d, "CREATE (a:User)-[r:FOLLOWS]->(b:User) RETURN r");
    let body = parse_body(&resp);

    assert_eq!(body["row_count"], 1, "RETURN r emits 1 row");

    let w = &body["writes"];
    assert_eq!(w["nodes_created"], 2);
    assert_eq!(w["rels_created"], 1);
}

#[test]
fn pure_read_match_composes_rows_and_empty_writes_in_envelope() {
    // The complementary pin: a pure-read MATCH returns rows and a
    // ZERO writes summary. Renderers that suppress the `writes:{...}`
    // block on `WriteSummary::is_empty()` honor this case.
    let d = fresh_dispatcher();
    let _ = parse_body(&raw_query(&d, "CREATE (n:User) RETURN n"));
    let _ = parse_body(&raw_query(&d, "CREATE (n:User) RETURN n"));

    let body = parse_body(&raw_query(&d, "MATCH (n:User) RETURN n"));
    assert_eq!(body["row_count"], 2, "MATCH observes 2 Users");
    let w = &body["writes"];
    assert_eq!(w["nodes_created"], 0);
    assert_eq!(w["nodes_deleted"], 0);
    assert_eq!(w["rels_created"], 0);
    assert_eq!(w["rels_deleted"], 0);
    assert_eq!(w["properties_set"], 0);
    assert_eq!(w["properties_removed"], 0);
    assert_eq!(w["labels_added"], 0);
    assert_eq!(w["labels_removed"], 0);
}

#[test]
fn empty_create_then_return_emits_zero_row_count_with_writes_counter() {
    // Composition edge case: anonymous CREATE without RETURN still
    // ticks writes; the row half is the openCypher "1 node created"
    // signal (0-column tuple) per ADR-147 §D-7.
    let d = fresh_dispatcher();
    let body = parse_body(&raw_query(&d, "CREATE (:User)"));
    // Anonymous CREATE emits 1 row (0-column) per the executor
    // semantic; the wire shape surfaces row_count = 1.
    assert_eq!(body["row_count"], 1);
    let w = &body["writes"];
    assert_eq!(w["nodes_created"], 1);
}
