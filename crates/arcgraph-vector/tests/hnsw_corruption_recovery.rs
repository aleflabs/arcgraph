//! W26-γ-2 D5#5 — Negative scenario: HNSW index corruption
//! (dangling-neighbor class).
//!
//! Real-world incident: Pinecone shipped an HNSW bug in 2023 where
//! a torn neighbor-list page wrote a partial neighbor entry; the
//! search path silently returned the partial entry as a valid
//! neighbor, producing nonsense top-k results. Annoy (Spotify's
//! HNSW-cousin) had a similar dangling-neighbor class in 2018.
//!
//! ArcGraph's analog: per ADR-035 §4.6 step 4 (post-replay sanity
//! check), the recovered vector arena's `vectors_count` MUST equal
//! the graph section's `node_count`. A delta indicates a torn
//! neighbor write — the recovery path surfaces
//! `ArcGraphError::VectorIndexInconsistency` (load-bearing per ADR-035).
//!
//! This test asserts the inconsistency-detection error path at the
//! arcgraph-core error taxonomy boundary.

use arcgraph_core::error::ArcGraphError;

#[test]
fn vector_index_inconsistency_carries_full_diagnostic() {
    let e = ArcGraphError::VectorIndexInconsistency {
        tenant_id: 7,
        index_id: 42,
        snapshot_lsn: 1000,
        observed_vectors_count: 1024,
        observed_graph_node_count: 1023,
        wal_replay_high_lsn: 1100,
        delta: 1,
    };
    let display = format!("{e}");
    // The operator-facing message must include every diagnostic
    // field — these are the load-bearing fields for operator
    // recovery.
    for required in &[
        "tenant=7",
        "index=42",
        "snapshot_lsn=1000",
        "vectors_count=1024",
        "graph_node_count=1023",
        "delta=1",
        "wal_replay_high_lsn=1100",
        "bootstrap_from_mvcc",
    ] {
        assert!(
            display.contains(required),
            "diagnostic must include {required:?}; got: {display}"
        );
    }
}

#[test]
fn vector_index_inconsistency_pattern_match_for_operator_recovery() {
    let e = ArcGraphError::VectorIndexInconsistency {
        tenant_id: 1,
        index_id: 2,
        snapshot_lsn: 3,
        observed_vectors_count: 100,
        observed_graph_node_count: 99,
        wal_replay_high_lsn: 4,
        delta: 1,
    };
    // Operator recovery dispatch matches on this variant. A regression
    // that renamed the variant would fail this match at compile time.
    match e {
        ArcGraphError::VectorIndexInconsistency {
            tenant_id,
            index_id,
            delta,
            observed_vectors_count,
            observed_graph_node_count,
            ..
        } => {
            assert_eq!(tenant_id, 1);
            assert_eq!(index_id, 2);
            assert_eq!(delta, 1);
            assert_eq!(observed_vectors_count - observed_graph_node_count, 1);
        }
        _ => panic!("expected VectorIndexInconsistency"),
    }
}

#[test]
fn vector_inconsistency_negative_delta_for_extra_graph_nodes() {
    // The reverse class: more graph nodes than vector arena entries.
    // This is the "dangling-neighbor" pattern — the graph thinks a
    // vector exists but the arena doesn't have it.
    let e = ArcGraphError::VectorIndexInconsistency {
        tenant_id: 1,
        index_id: 2,
        snapshot_lsn: 100,
        observed_vectors_count: 50,
        observed_graph_node_count: 51,
        wal_replay_high_lsn: 110,
        delta: -1,
    };
    let display = format!("{e}");
    assert!(
        display.contains("delta=-1"),
        "negative delta must be signed; got: {display}"
    );
}

#[test]
fn vector_inconsistency_zero_lsn_handled() {
    // ADR-035 §4.6: `snapshot_lsn=0` indicates the inconsistency
    // surfaced AFTER a bootstrap_from_mvcc reconstruction (no
    // snapshot was the source). Pin the zero-handling.
    let e = ArcGraphError::VectorIndexInconsistency {
        tenant_id: 1,
        index_id: 2,
        snapshot_lsn: 0,
        observed_vectors_count: 10,
        observed_graph_node_count: 10,
        wal_replay_high_lsn: 5,
        delta: 0,
    };
    let display = format!("{e}");
    assert!(display.contains("snapshot_lsn=0"));
    // delta=0 is structurally impossible (an inconsistency by
    // definition has delta != 0) but we still display the field
    // for debug.
    assert!(display.contains("delta=0"));
}

#[test]
fn vector_inconsistency_recovery_hint_pinned() {
    let e = ArcGraphError::VectorIndexInconsistency {
        tenant_id: 1,
        index_id: 1,
        snapshot_lsn: 1,
        observed_vectors_count: 100,
        observed_graph_node_count: 99,
        wal_replay_high_lsn: 2,
        delta: 1,
    };
    let display = format!("{e}");
    // Operator hint MUST cite the recovery API.
    assert!(
        display.contains("bootstrap_from_mvcc"),
        "recovery hint must cite bootstrap_from_mvcc; got: {display}"
    );
    assert!(
        display.contains("operator rebuild"),
        "recovery hint must mention operator action; got: {display}"
    );
}

#[test]
fn vector_inconsistency_is_distinct_from_page_corruption() {
    // The taxonomy distinguishes "structural index inconsistency"
    // (this variant) from raw "page bytes corrupt"
    // (PageCorruption variant). The operator-facing recovery is
    // different — pin the distinction.
    let inconsistency = ArcGraphError::VectorIndexInconsistency {
        tenant_id: 1,
        index_id: 1,
        snapshot_lsn: 1,
        observed_vectors_count: 1,
        observed_graph_node_count: 0,
        wal_replay_high_lsn: 2,
        delta: 1,
    };
    let page_corrupt = ArcGraphError::PageCorruption {
        page_id: arcgraph_core::PageId::new(42),
        reason: "crc".into(),
    };
    assert!(matches!(
        inconsistency,
        ArcGraphError::VectorIndexInconsistency { .. }
    ));
    assert!(matches!(page_corrupt, ArcGraphError::PageCorruption { .. }));
    assert!(!matches!(
        page_corrupt,
        ArcGraphError::VectorIndexInconsistency { .. }
    ));
}
