//! W27-β ADR-153 — `graph.raw_query` DELETE / DETACH DELETE integration.
//!
//! Closes audit-2026-05-27 finding for the Phase 3 / ADR-149 surface.
//! Each DELETE clause is wired through the MCP boundary; the writes
//! summary surfaces `nodes_deleted` / `rels_deleted` per ADR-153 §D-2
//! + ADR-149 §"Counting semantics".

#![allow(clippy::unwrap_used)]

mod raw_query_write_common;
use raw_query_write_common::{assert_writes, fresh_dispatcher, parse_body, raw_query};

#[test]
fn create_then_match_then_delete_round_trips_through_raw_query() {
    // Round-trip CREATE → MATCH → DELETE → MATCH. Each phase's
    // writes summary reflects the statement's effects:
    //   - CREATE: nodes_created=1, others zero.
    //   - MATCH: zero writes (pure-read).
    //   - DELETE: nodes_deleted=1 (Phase 3 emits 0 rows).
    //   - MATCH-after-delete: 0 rows, zero writes.
    let d = fresh_dispatcher();
    let create = parse_body(&raw_query(&d, "CREATE (n:User) RETURN n"));
    assert_writes(&create, (1, 0, 0, 0, 0, 0, 0, 0));

    let match_pre = parse_body(&raw_query(&d, "MATCH (n:User) RETURN n"));
    assert_eq!(match_pre["row_count"], 1, "MATCH sees the CREATE");
    assert_writes(&match_pre, (0, 0, 0, 0, 0, 0, 0, 0));

    let delete = parse_body(&raw_query(&d, "MATCH (n:User) DELETE n"));
    // ADR-149 §D-9: Phase 3 DELETE is TERMINAL; emits 0 rows.
    assert_eq!(delete["row_count"], 0, "DELETE emits 0 rows at Phase 3");
    assert_writes(&delete, (0, 1, 0, 0, 0, 0, 0, 0));

    let match_post = parse_body(&raw_query(&d, "MATCH (n:User) RETURN n"));
    assert_eq!(match_post["row_count"], 0, "post-DELETE MATCH sees 0 rows");
    assert_writes(&match_post, (0, 0, 0, 0, 0, 0, 0, 0));
}

#[test]
fn detach_delete_v1_0_alpha_surface_cascade_rels_not_counted() {
    // ADR-153 §D-2 v1.0-α posture pin: DETACH DELETE cascades through
    // the substrate's INTERNAL `crud::delete_rel_with_store` calls,
    // NOT through the `ExecutorSubstrate::delete_rel` trait call. The
    // CountingSubstrate observes only trait-level calls, so the
    // cascade rels do NOT tick `rels_deleted` at v1.0-α.
    //
    // Forward-pinned to v1.1: route substrate cascades through
    // `self.delete_rel(..., &arcgraph_query::executor::ExecutionContext::new(TenantId::DEFAULT, arcgraph_core::PartitionId::ZERO))` so the counter surfaces openCypher v9
    // "cascade counts as deletion" semantics. ADR-153 §"Forward-
    // deferred" names the amendment hook. Sister test
    // `delete_relationship_only_does_not_touch_nodes_count` pins the
    // explicit-DELETE-rel counting path which works correctly
    // end-to-end.
    let d = fresh_dispatcher();
    // Build a node with one outbound rel: (a:User)-[:FOLLOWS]->(b:User).
    let _ = parse_body(&raw_query(
        &d,
        "CREATE (a:User)-[r:FOLLOWS]->(b:User) RETURN r",
    ));
    // DETACH DELETE a — the cascade tombstones the FOLLOWS rel + the a
    // node. The b node remains. WriteSummary surfaces nodes_deleted=1;
    // rels_deleted=0 per the v1.0-α cascade-internal-not-counted rule
    // codified in ADR-153 §D-2.
    let detach = parse_body(&raw_query(
        &d,
        "MATCH (a:User)-[r:FOLLOWS]->(b:User) DETACH DELETE a",
    ));
    assert_writes(&detach, (0, 1, 0, 0, 0, 0, 0, 0));
    let post = parse_body(&raw_query(&d, "MATCH (n:User) RETURN n"));
    assert_eq!(post["row_count"], 1, "only b remains");
}

#[test]
fn delete_rel_and_node_in_single_clause_ticks_both_counters() {
    // `DELETE r, n` after a MATCH that bound both: the DeleteOp
    // dispatches per item, so the trait-level `delete_rel(...)` +
    // `delete_node(...)` both fire → CountingSubstrate observes
    // BOTH counters tick. This is the v1.0-α-supported pattern for
    // surfacing accurate rel + node counts together.
    let d = fresh_dispatcher();
    let _ = parse_body(&raw_query(
        &d,
        "CREATE (a:User)-[r:FOLLOWS]->(b:User) RETURN r",
    ));
    let del = parse_body(&raw_query(
        &d,
        "MATCH (a:User)-[r:FOLLOWS]->(b:User) DELETE r, a",
    ));
    assert_writes(&del, (0, 1, 0, 1, 0, 0, 0, 0));
    let post = parse_body(&raw_query(&d, "MATCH (n:User) RETURN n"));
    assert_eq!(post["row_count"], 1, "only b remains");
}

#[test]
fn delete_relationship_only_does_not_touch_nodes_count() {
    // Delete the relationship but keep both endpoints. The writes
    // summary surfaces rels_deleted=1, nodes_deleted=0.
    let d = fresh_dispatcher();
    let _ = parse_body(&raw_query(
        &d,
        "CREATE (a:User)-[r:FOLLOWS]->(b:User) RETURN r",
    ));
    let del = parse_body(&raw_query(
        &d,
        "MATCH (a:User)-[r:FOLLOWS]->(b:User) DELETE r",
    ));
    assert_writes(&del, (0, 0, 0, 1, 0, 0, 0, 0));
    // Both endpoints still observable.
    let post = parse_body(&raw_query(&d, "MATCH (n:User) RETURN n"));
    assert_eq!(post["row_count"], 2, "both endpoints kept");
}
