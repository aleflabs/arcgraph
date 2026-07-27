//! W27-β ADR-153 — `graph.raw_query` MERGE integration.
//!
//! Closes audit-2026-05-27 finding for the Phase 5 / ADR-151 surface.
//! MERGE is match-or-create: when the match pattern misses, the create
//! branch fires; when it hits, the on_match actions fire. The writes
//! summary reflects WHICH branch executed.

#![allow(clippy::unwrap_used)]

mod raw_query_write_common;
use raw_query_write_common::{assert_writes, fresh_dispatcher, parse_body, raw_query};

// v1.0-α posture (ADR-151-amendment-01): a node-shape NAMED MERGE
// (`MERGE (n:Label …)`) EMITS its matched/created binding row — the
// RETURN-after-MERGE node-shape case is lifted to v1.0-α (the amendment
// reconciles the shipped code with ADR-151 §D-7's own row-emitting
// pseudocode + lifts the §D-9 forward-pin row). A bare named
// `MERGE (n:User)` therefore surfaces row_count = 1 (CREATE-aligned:
// bare `CREATE (:User)` already returns its trigger row — see
// `raw_query_write_create_node_integ.rs`). Path-shape + anonymous
// merges stay terminal; multi-statement read-your-writes (the genuine
// batch-tx surface) stays pinned per ADR-151 §D-9. The strict-
// openCypher "no-RETURN ⇒ 0 rows" posture is a pre-existing, project-
// wide write-op behavior (it affects all 5 ops, not just MERGE) and is
// out of scope here. The writes summary remains the branch-outcome
// signal; the emitted row enables `MERGE … RETURN n` (see
// `raw_query_write_merge_return_integ.rs` for the value-oracle suite).

#[test]
fn first_merge_misses_match_and_takes_create_branch() {
    // Empty graph + MERGE → match misses → create branch fires →
    // nodes_created=1. ADR-151-amendment-01: a node-shape named MERGE
    // emits its created binding row, so row_count=1 (0→1 vs the
    // pre-amendment terminal posture; CREATE-aligned).
    let d = fresh_dispatcher();
    let resp = raw_query(&d, "MERGE (n:User)");
    let body = parse_body(&resp);
    assert_eq!(
        body["row_count"], 1,
        "node-shape named MERGE emits its binding row (ADR-151-amendment-01)"
    );
    assert_writes(&body, (1, 0, 0, 0, 0, 0, 0, 0));
}

#[test]
fn second_merge_hits_match_and_does_not_create() {
    // Run MERGE twice. First creates; second matches. The second
    // MERGE's writes summary surfaces ALL ZEROS (match-branch is
    // pure-read at v1.0-α). Use a MATCH afterward to verify the
    // node exists.
    let d = fresh_dispatcher();
    let first = parse_body(&raw_query(&d, "MERGE (n:User)"));
    assert_writes(&first, (1, 0, 0, 0, 0, 0, 0, 0));
    let second = parse_body(&raw_query(&d, "MERGE (n:User)"));
    // Match hit → no CREATE → counter stays zero.
    assert_writes(&second, (0, 0, 0, 0, 0, 0, 0, 0));
    // Sanity: MATCH after the two MERGEs returns exactly 1 node.
    let match_post = parse_body(&raw_query(&d, "MATCH (n:User) RETURN n"));
    assert_eq!(match_post["row_count"], 1, "exactly 1 User node");
}

#[test]
fn merge_on_create_set_fires_only_on_first_run() {
    // ADR-151 §D-5 ON CREATE SET: action fires only when the create
    // branch is taken. First run: nodes_created=1, properties_set=1.
    // Second run: ALL ZEROS (no new node, on_create skipped, on_match
    // not specified).
    let d = fresh_dispatcher();
    let first = parse_body(&raw_query(
        &d,
        "MERGE (n:User) ON CREATE SET n.fresh = TRUE",
    ));
    assert_writes(&first, (1, 0, 0, 0, 1, 0, 0, 0));
    let second = parse_body(&raw_query(
        &d,
        "MERGE (n:User) ON CREATE SET n.fresh = TRUE",
    ));
    assert_writes(&second, (0, 0, 0, 0, 0, 0, 0, 0));
}

#[test]
fn merge_on_match_set_fires_only_on_second_run() {
    // ADR-151 §D-5 ON MATCH SET: action fires only when the match
    // branch is taken. First run: nodes_created=1, properties_set=0
    // (on_match skipped). Second run: nodes_created=0, properties_set=1
    // (on_match fires).
    let d = fresh_dispatcher();
    let first = parse_body(&raw_query(
        &d,
        "MERGE (n:User) ON MATCH SET n.visited = TRUE",
    ));
    assert_writes(&first, (1, 0, 0, 0, 0, 0, 0, 0));
    let second = parse_body(&raw_query(
        &d,
        "MERGE (n:User) ON MATCH SET n.visited = TRUE",
    ));
    assert_writes(&second, (0, 0, 0, 0, 1, 0, 0, 0));
}
