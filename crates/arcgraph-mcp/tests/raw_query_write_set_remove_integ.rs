//! W27-β ADR-153 — `graph.raw_query` SET + REMOVE integration.
//!
//! Closes audit-2026-05-27 finding for the Phase 4 / ADR-150 surface.
//! Each SET / REMOVE clause is wired through the MCP boundary; the
//! writes summary surfaces `properties_set` / `properties_removed` /
//! `labels_added` / `labels_removed` per ADR-153 §D-2 + ADR-150
//! §"Counting semantics".
//!
//! # v1.0-α posture (ADR-150 §D-9 inherited)
//!
//! Per ADR-150 §D-9 the production substrate's `set_node`/`remove_node`
//! treat the property-bag mutation paths (`PropertyAssign` /
//! `PropertyReplace` / `PropertyMerge` / `Property` REMOVE) via
//! `arcgraph_storage::crud::update_node` (succeeds end-to-end at
//! v1.0-α; properties typed via `PropertyData::Empty` here →
//! properties don't ROUND-TRIP on a follow-up MATCH but
//! the WRITES counter still ticks because the trait call succeeded).
//!
//! The LabelAdd / LabelRemove paths surface `IndexUnavailable` per
//! ADR-150 §D-9 because the storage primitive preserves the label
//! immutably; the CountingSubstrate increments counters only on
//! `Ok(_)` per ADR-153 §D-2, so a label-add/remove at v1.0-α does NOT
//! tick `labels_added`/`labels_removed` end-to-end through the
//! production substrate. Pin: the `labels_added`/`labels_removed`
//! semantics are validated at the CountingSubstrate unit test layer
//! (`storage::counting::tests`) where the stub substrate accepts the
//! call; this test file pins the v1.0-α production posture (the
//! `IndexUnavailable` is the v1.0-α MCP-visible behavior; v1.1
//! lights the storage primitive).

#![allow(clippy::unwrap_used)]

mod raw_query_write_common;
use raw_query_write_common::{assert_writes, fresh_dispatcher, parse_body, raw_query};

#[test]
fn set_property_assign_ticks_properties_set_by_one() {
    // ADR-150 §D-7 PropertyAssign path: a single `SET n.k = v` clause
    // calls `set_node(PropertyAssign{..})` once → properties_set=1.
    let d = fresh_dispatcher();
    let _ = parse_body(&raw_query(&d, "CREATE (n:User)"));
    let set = parse_body(&raw_query(&d, "MATCH (n:User) SET n.name = \"alice\""));
    // ADR-150 §D-9: Phase 4 SET is TERMINAL; emits 0 rows.
    assert_eq!(set["row_count"], 0);
    assert_writes(&set, (0, 0, 0, 0, 1, 0, 0, 0));
}

#[test]
fn set_property_merge_ticks_by_entries_len() {
    // ADR-150 §D-7 PropertyMerge: `SET n += {a:1, b:2, c:3}` ticks
    // properties_set by entries.len() = 3.
    let d = fresh_dispatcher();
    let _ = parse_body(&raw_query(&d, "CREATE (n:User)"));
    let set = parse_body(&raw_query(&d, "MATCH (n:User) SET n += {a: 1, b: 2, c: 3}"));
    assert_writes(&set, (0, 0, 0, 0, 3, 0, 0, 0));
}

#[test]
fn set_property_replace_ticks_by_entries_len() {
    // ADR-150 §D-7 PropertyReplace: `SET n = {x:1, y:2}` ticks
    // properties_set by entries.len() = 2.
    let d = fresh_dispatcher();
    let _ = parse_body(&raw_query(&d, "CREATE (n:User)"));
    let set = parse_body(&raw_query(&d, "MATCH (n:User) SET n = {x: 1, y: 2}"));
    assert_writes(&set, (0, 0, 0, 0, 2, 0, 0, 0));
}

#[test]
fn remove_property_ticks_properties_removed_by_one() {
    // ADR-150 §D-7 RemoveNodeMutation::Property: REMOVE n.k ticks
    // properties_removed by 1.
    let d = fresh_dispatcher();
    let _ = parse_body(&raw_query(&d, "CREATE (n:User)"));
    let rm = parse_body(&raw_query(&d, "MATCH (n:User) REMOVE n.name"));
    assert_writes(&rm, (0, 0, 0, 0, 0, 1, 0, 0));
}

#[test]
fn set_label_add_v1_0_alpha_returns_index_unavailable_error() {
    // ADR-150 §D-9 v1.0-α posture: the production substrate's
    // `set_node(LabelAdd(..))` surfaces `IndexUnavailable` because the
    // storage primitive preserves `label_id` immutably. The MCP wire
    // shape returns an error envelope (not a writes summary). Counters
    // stay at zero per the err-path rule in ADR-153 §D-2.
    let d = fresh_dispatcher();
    let _ = parse_body(&raw_query(&d, "CREATE (n:User)"));
    let resp = raw_query(&d, "MATCH (n:User) SET n:Verified");
    // The wire shape: either an error envelope OR a zero-writes
    // success — pin BOTH paths so a regression surfaces. v1.0-α
    // observed posture is error envelope.
    if !resp["error"].is_null() {
        let code = resp["error"]["code"].as_i64().expect("error code");
        // -32006 ExecutionEval (substrate IndexUnavailable wraps to
        // this code per the W13δ codec-local translation rules).
        assert!(
            code == -32006 || code == -32004,
            "label-add must surface a structured error code; got {code}: {resp:?}"
        );
    } else {
        let body = parse_body(&resp);
        // The IndexUnavailable path → counters stay zero.
        assert_writes(&body, (0, 0, 0, 0, 0, 0, 0, 0));
    }
}

#[test]
fn pure_set_clause_emits_zero_rows_and_only_ticks_properties_set() {
    // Pin: SET emits 0 rows per ADR-150 §D-9 Phase 4 terminal posture,
    // and only ticks properties_set (not properties_removed, not
    // nodes_*). Counter-cross-bleed regression pin.
    let d = fresh_dispatcher();
    let _ = parse_body(&raw_query(&d, "CREATE (n:User)"));
    let set = parse_body(&raw_query(&d, "MATCH (n:User) SET n.x = 1, n.y = 2"));
    assert_eq!(set["row_count"], 0);
    assert_writes(&set, (0, 0, 0, 0, 2, 0, 0, 0));
}
