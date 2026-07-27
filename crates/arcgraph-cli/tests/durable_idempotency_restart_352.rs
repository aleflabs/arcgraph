//! #352 Part 2 (ADR-199) — the `graph.ingest` `external_id → internal_id`
//! idempotency binding must SURVIVE a durable `--data` restart.
//!
//! # The bug (#352)
//!
//! Before this fix the binding lived ONLY in a process-side in-memory map
//! (`StorageBackend::idempotency`). `external_id` never reached
//! `arcgraph-storage` — `create_node` persists only the property bag — so the
//! map was the sole record of the binding anywhere, and it was lost on every
//! process bounce. An ingest issued BEFORE a restart therefore returned
//! `Inserted` with a NEW id AFTER the restart (a duplicate-mint), instead of
//! `Idempotent { internal_id }` with the original id — a correctness bug at
//! v1.1's availability bar.
//!
//! # The fix (ADR-199 v6 CommitBundle fold)
//!
//! The binding now rides INSIDE the owning commit's `CommitBundle` (bundle
//! format v6, the new `idempotency_bindings` section) — durable + atomic with
//! the node/rel write that allocated the internal id. On restart,
//! `recover_from_wal` rebuilds the storage-resident
//! `arcgraph_storage::IdempotencyStore` from those sections (wired into the
//! durable bootstrap exactly like the InternTable, #776), so a re-ingest
//! resolves idempotently to the ORIGINAL id.
//!
//! # This is the ADR-133 active verification for #352 Part 2.
//!
//! The test drives the production `bootstrap_storage_backend(--data)` surface
//! end-to-end through the real `StorageIngestProvider`: ingest a node by
//! `external_id` (capture the `Inserted` id), restart (drop the guard so the
//! WAL writer drains + fsyncs + joins), re-bootstrap the SAME dir, and assert
//! the re-ingest returns `Idempotent` with the **exact same** id. The strong
//! oracle is the real id equality across the restart, NOT a proxy.

use std::collections::BTreeMap;

use arcgraph_cli::bootstrap::{BootstrapMode, bootstrap_storage_backend};
use arcgraph_core::TenantId;
use arcgraph_mcp::storage::{StorageBackend, StorageIngestProvider, StorageRawQueryExecutor};
use arcgraph_mcp::tools::ingest::{
    IngestBatch, IngestProvider, IngestRecordOutcome, IngestSummary, NodeIngest, RelIngest,
};
use arcgraph_mcp::tools::raw_query::{RawQueryExecutor, RawQueryRows};
use arcgraph_query::CancellationToken;
use tempfile::TempDir;

/// One node carrying an `external_id`.
fn node_batch(external_id: &str, label: &str) -> IngestBatch {
    IngestBatch {
        nodes: vec![NodeIngest {
            external_id: Some(external_id.into()),
            label: label.into(),
            properties: BTreeMap::new(),
        }],
        relationships: Vec::new(),
        acl_grants: vec![],
    }
}

/// Ingest `batch` through the production provider over `backend`, panicking on
/// transport error so a failure surfaces with the diagnostic attached.
fn ingest(backend: &StorageBackend, batch: IngestBatch) -> Vec<IngestRecordOutcome> {
    ingest_summary(backend, batch).records
}

fn ingest_summary(backend: &StorageBackend, batch: IngestBatch) -> IngestSummary {
    let provider = StorageIngestProvider::new(backend.clone());
    provider
        .ingest(TenantId::DEFAULT, batch)
        .expect("ingest call returns Ok")
}

fn run_query(backend: &StorageBackend, query: &str) -> RawQueryRows {
    let exec = StorageRawQueryExecutor::new(backend.clone());
    let cancel = CancellationToken::new();
    exec.execute(TenantId::DEFAULT, query, 1000, &cancel)
        .unwrap_or_else(|e| panic!("query {query:?} failed: {e:?}"))
}

fn run_count(backend: &StorageBackend, query: &str) -> u64 {
    let rows = run_query(backend, query);
    let first = rows
        .rows
        .first()
        .unwrap_or_else(|| panic!("query {query:?} returned no rows"));
    let val = first
        .as_array()
        .and_then(|a| a.first())
        .unwrap_or_else(|| panic!("query {query:?} row 0 has no col 0: {first:?}"));
    val.as_u64()
        .unwrap_or_else(|| panic!("query {query:?} col 0 is not a u64: {val:?}"))
}

/// Extract the single record's internal id, asserting it is an `Inserted`.
fn expect_inserted(records: &[IngestRecordOutcome]) -> u64 {
    match records.first().expect("one record") {
        IngestRecordOutcome::Inserted { internal_id, .. } => *internal_id,
        other => panic!("expected Inserted, got {other:?}"),
    }
}

fn node_with_v(external_id: &str, v: i64) -> IngestBatch {
    IngestBatch {
        nodes: vec![NodeIngest {
            external_id: Some(external_id.into()),
            label: "P".into(),
            properties: BTreeMap::from([("v".into(), serde_json::Value::Number(v.into()))]),
        }],
        relationships: Vec::new(),
        acl_grants: vec![],
    }
}

/// Extract the single record's internal id, asserting it is an `Idempotent`.
fn expect_idempotent(records: &[IngestRecordOutcome]) -> u64 {
    match records.first().expect("one record") {
        IngestRecordOutcome::Idempotent { internal_id, .. } => *internal_id,
        other => panic!("expected Idempotent, got {other:?}"),
    }
}

// ─────────────────────────────────────────────────────────────────────
// #352 — the headline acceptance: Inserted{id} → restart → Idempotent{same id}.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn idempotency_binding_survives_durable_restart_352() {
    let tmp = TempDir::new().expect("tempdir");
    let data_dir = tmp.path().join("db");

    // ── Session 1: bootstrap durable, ingest node "alice" by external_id.
    let alice_id = {
        let (backend, guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
            data_dir: data_dir.clone(),
        })
        .expect("durable bootstrap (session 1)");
        assert!(guard.is_durable(), "Durable mode must own a WAL writer");

        let recs = ingest(&backend, node_batch("alice", "Person"));
        let id = expect_inserted(&recs);

        // In-session sanity: a same-process re-ingest is already idempotent.
        let again = ingest(&backend, node_batch("alice", "Person"));
        assert_eq!(
            expect_idempotent(&again),
            id,
            "in-session re-ingest must resolve to the same id BEFORE restart",
        );

        // Drop the guard → the WalWriter drains + fsyncs + joins (graceful
        // "process restart"). The v6 CommitBundle carrying alice's node AND
        // its idempotency binding is durable past this point.
        drop(guard);
        id
    };

    // ── Session 2: re-bootstrap the SAME dir → WAL recovery on startup
    //    rebuilds the IdempotencyStore from the v6 idempotency_bindings.
    let (backend2, _guard2) = bootstrap_storage_backend(&BootstrapMode::Durable {
        data_dir: data_dir.clone(),
    })
    .expect("durable bootstrap (session 2 — recover)");

    // #352 ORACLE — the re-ingest of "alice" returns Idempotent with the
    // EXACT SAME id allocated before the restart. Pre-fix this returned
    // `Inserted` with a NEW (duplicate) id because the in-memory map was lost.
    let recovered = ingest(&backend2, node_batch("alice", "Person"));
    let recovered_id = expect_idempotent(&recovered);
    assert_eq!(
        recovered_id, alice_id,
        "#352: a node ingested BEFORE restart MUST resolve Idempotent to its \
         ORIGINAL id ({alice_id}) after restart, not mint a duplicate (got {recovered_id})",
    );
}

#[test]
fn deleted_external_id_release_survives_durable_restart_1010() {
    let tmp = TempDir::new().expect("tempdir");
    let data_dir = tmp.path().join("db");

    {
        let (backend, guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
            data_dir: data_dir.clone(),
        })
        .expect("durable bootstrap (session 1)");

        let first = ingest_summary(&backend, node_with_v("x", 1));
        assert_eq!(first.inserted_count, 1);
        assert_eq!(first.failed_count, 0);
        let _first_id = expect_inserted(&first.records);
        assert_eq!(run_count(&backend, "MATCH (n) RETURN count(n)"), 1);

        run_query(&backend, "MATCH (n) DELETE n");
        assert_eq!(run_count(&backend, "MATCH (n) RETURN count(n)"), 0);

        let same = ingest_summary(&backend, node_with_v("x", 1));
        assert_eq!(same.inserted_count, 1);
        assert_eq!(same.failed_count, 0);
        let _second_id = expect_inserted(&same.records);
        assert_eq!(run_count(&backend, "MATCH (n) RETURN count(n)"), 1);

        run_query(&backend, "MATCH (n) DELETE n");
        assert_eq!(run_count(&backend, "MATCH (n) RETURN count(n)"), 0);
        drop(guard);
    }

    let (backend2, _guard2) = bootstrap_storage_backend(&BootstrapMode::Durable {
        data_dir: data_dir.clone(),
    })
    .expect("durable bootstrap (session 2 — recover)");

    let changed = ingest_summary(&backend2, node_with_v("x", 9));
    assert_eq!(changed.inserted_count, 1);
    assert_eq!(changed.failed_count, 0);
    let _third_id = expect_inserted(&changed.records);
    assert_eq!(run_count(&backend2, "MATCH (n) RETURN count(n)"), 1);
}

// ─────────────────────────────────────────────────────────────────────
// #352 — an EDGE can resolve a node endpoint committed by a PRIOR process.
// This is the second failure mode the in-memory map caused: a post-restart
// edge to a pre-restart node failed "rel endpoints unresolved" though the
// node existed durably.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn edge_resolves_prior_process_node_external_id_352() {
    let tmp = TempDir::new().expect("tempdir");
    let data_dir = tmp.path().join("db");

    // ── Session 1: ingest node "hub" by external_id, then restart.
    let hub_id = {
        let (backend, guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
            data_dir: data_dir.clone(),
        })
        .expect("durable bootstrap (session 1)");
        let recs = ingest(&backend, node_batch("hub", "Account"));
        let id = expect_inserted(&recs);
        drop(guard);
        id
    };

    // ── Session 2: recover, then ingest a NEW node "spoke" + an edge
    //    hub → spoke, referencing "hub" by external_id. The edge endpoint
    //    "hub" must resolve to the PRIOR process's id via the recovered
    //    IdempotencyStore.
    let (backend2, _guard2) = bootstrap_storage_backend(&BootstrapMode::Durable {
        data_dir: data_dir.clone(),
    })
    .expect("durable bootstrap (session 2 — recover)");

    let batch = IngestBatch {
        nodes: vec![NodeIngest {
            external_id: Some("spoke".into()),
            label: "Account".into(),
            properties: BTreeMap::new(),
        }],
        relationships: vec![RelIngest {
            external_id: None,
            from_external_id: "hub".into(),
            to_external_id: "spoke".into(),
            rel_type: "SENT".into(),
            properties: BTreeMap::new(),
        }],
        acl_grants: vec![],
    };
    let provider = StorageIngestProvider::new(backend2.clone());
    let summary = provider
        .ingest(TenantId::DEFAULT, batch)
        .expect("ingest call returns Ok");

    // The edge must NOT fail "rel endpoints unresolved" — "hub" resolves to
    // the recovered id. inserted_count = 2 (spoke node + the edge).
    assert_eq!(
        summary.failed_count, 0,
        "#352: edge to a prior-process node must NOT fail unresolved; records={:?}",
        summary.records,
    );
    assert_eq!(
        summary.inserted_count, 2,
        "#352: spoke node + hub→spoke edge both created",
    );

    // And re-ingesting "hub" still resolves to the prior-process id (the
    // binding is durable, not a fresh mint).
    let re_hub = ingest(&backend2, node_batch("hub", "Account"));
    assert_eq!(
        expect_idempotent(&re_hub),
        hub_id,
        "#352: hub still resolves Idempotent to its original prior-process id",
    );
}
