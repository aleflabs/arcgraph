//! #1221 (ADR-218) — the document-level read ACL applied through the live
//! `graph.ingest` write-through MUST SURVIVE a bare durable `--data`
//! restart without an auxiliary seed path.
//!
//! # The bug (#1221)
//!
//! Before this fix the `PermissionIndex` (the enforcement plane
//! `graph.search` reads, ADR-212 §D-2(b)) lived ONLY in a process-side
//! in-memory map. A bare `serve --data` restart came up DENY-ALL — every
//! principal resolved to 0 visible docs, even though the graph data
//! survived WAL replay (#849). For a permissions product, silently losing
//! every ACL on restart is a GA-blocking durability defect.
//!
//! # The fix (ADR-218 v8 CommitBundle fold)
//!
//! `apply_doc_acl` / `revoke_doc` now durify each op into the WAL's v8
//! `acl_grants` section (a dedicated single-op commit). On restart,
//! `recover_from_wal` re-drives the ops into a fresh `PermissionIndex` —
//! the SAME `Arc` the served router adopts (`build_durable_backend` wires
//! it into both the replay target and the router) — so enforcement is
//! intact before serving.
//!
//! # This is the ADR-133 active verification for #1221 — the END-TO-END
//! restart oracle through the production `bootstrap_storage_backend(--data)`
//! surface (the storage-crate seam is pinned in isolation by
//! `crates/arcgraph-storage/tests/acl_wal_replay_1221.rs`).
//!
//! The test drives the REAL `StorageIngestProvider` (which routes ACL
//! grants through `apply_live_acl_grants` → `PermissionIndex::apply_doc_acl`
//! → the durable `CrudAclWalSink`), restarts (drop the guard so the WAL
//! drains + fsyncs), re-bootstraps the SAME dir, and asserts the grantee
//! still sees the doc and a non-grantee denies — WITHOUT re-ingesting.
//! The strong oracle is the real `effective(principal).is_visible(node)`
//! resolution across the restart, NOT a proxy.

use std::collections::BTreeMap;

use arcgraph_cli::bootstrap::{BootstrapMode, bootstrap_storage_backend};
use arcgraph_core::{NodeId, PartitionId, TenantId};
use arcgraph_mcp::storage::{StorageBackend, StorageIngestProvider};
use arcgraph_mcp::tools::ingest::{
    AclGrant, IngestBatch, IngestProvider, IngestRecordOutcome, NodeIngest,
};
use tempfile::TempDir;

/// One node carrying an `external_id`, plus a read-ACL grant for that doc.
fn node_with_acl(external_id: &str, principals: &[&str]) -> IngestBatch {
    IngestBatch {
        nodes: vec![NodeIngest {
            external_id: Some(external_id.into()),
            label: "Doc".into(),
            properties: BTreeMap::new(),
        }],
        relationships: Vec::new(),
        acl_grants: vec![AclGrant {
            external_id: external_id.into(),
            read_principals: Some(principals.iter().map(|s| (*s).to_owned()).collect()),
        }],
    }
}

fn ingest(backend: &StorageBackend, batch: IngestBatch) -> Vec<IngestRecordOutcome> {
    let provider = StorageIngestProvider::new(backend.clone());
    provider
        .ingest(TenantId::DEFAULT, batch)
        .expect("ingest call returns Ok")
        .records
}

fn committed_id(records: &[IngestRecordOutcome]) -> u64 {
    match records.first().expect("one node record") {
        IngestRecordOutcome::Inserted { internal_id, .. }
        | IngestRecordOutcome::Idempotent { internal_id, .. } => *internal_id,
        other => panic!("expected a committed node, got {other:?}"),
    }
}

/// `effective(principal).is_visible(node)` through the served router's
/// live `PermissionIndex` — the exact oracle production enforcement reads.
fn can_read(backend: &StorageBackend, principal: &str, node: u64) -> bool {
    let handle = backend
        .router()
        .route(TenantId::DEFAULT, PartitionId::ZERO)
        .expect("route DEFAULT tenant");
    handle
        .permissions()
        .effective(principal)
        .is_visible(NodeId::new(node))
}

// ─────────────────────────────────────────────────────────────────────
// THE ORACLE (Director-required): a live-ingested ACL survives a bare
// `--data` restart; grantee sees the doc, non-grantee denies-all.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn live_acl_grant_survives_durable_restart_1221() {
    let tmp = TempDir::new().expect("tempdir");
    let data_dir = tmp.path().join("db");

    // ── Session 1: bootstrap durable, ingest "doc:1" granting alice
    //    (live write-through → durable v8 acl_grants commit). ──
    let doc1 = {
        let (backend, guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
            data_dir: data_dir.clone(),
        })
        .expect("durable bootstrap (session 1)");
        assert!(guard.is_durable(), "Durable mode must own a WAL writer");

        let recs = ingest(&backend, node_with_acl("doc:1", &["alice"]));
        let id = committed_id(&recs);

        // In-session sanity BEFORE restart: alice sees doc:1, bob does not.
        assert!(
            can_read(&backend, "alice", id),
            "in-session: alice (grantee) must see doc:1 before restart"
        );
        assert!(
            !can_read(&backend, "bob", id),
            "in-session: bob (non-grantee) must NOT see doc:1"
        );

        // Drop the guard → the WalWriter drains + fsyncs + joins (graceful
        // "process restart"). The v8 CommitBundle carrying the acl_grant is
        // durable past this point.
        drop(guard);
        id
    };

    // ── Session 2: re-bootstrap the SAME dir → WAL recovery rebuilds the
    //    PermissionIndex from the v8 acl_grants section. NO re-ingest, NO
    //    auxiliary seed path. ──
    let (backend2, _guard2) = bootstrap_storage_backend(&BootstrapMode::Durable {
        data_dir: data_dir.clone(),
    })
    .expect("durable bootstrap (session 2 — recover)");

    // #1221 ORACLE — the grant survived the bounce.
    assert!(
        can_read(&backend2, "alice", doc1),
        "#1221: alice (grantee) MUST still see doc:1 after a bare restart \
         (pre-fix: deny-all — the ACL was lost)"
    );
    // Non-grantee still denies (no widen on recovery).
    assert!(
        !can_read(&backend2, "bob", doc1),
        "#1221: bob (non-grantee) MUST still deny after restart (no widen)"
    );
    // A doc that was never granted is UNCLASSIFIED for everyone.
    assert!(!can_read(&backend2, "alice", doc1 + 9999));
}

// ─────────────────────────────────────────────────────────────────────
// Revoke durability: a grant THEN a re-narrowed grant (revoking a
// principal) survives restart with the narrowed set enforced.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn live_acl_revoke_survives_durable_restart_1221() {
    let tmp = TempDir::new().expect("tempdir");
    let data_dir = tmp.path().join("db");

    let doc = {
        let (backend, guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
            data_dir: data_dir.clone(),
        })
        .expect("durable bootstrap (session 1)");

        // Grant alice + bob.
        let recs = ingest(&backend, node_with_acl("doc:2", &["alice", "bob"]));
        let id = committed_id(&recs);
        assert!(can_read(&backend, "bob", id), "bob granted initially");

        // Re-ingest the SAME doc with a NARROWED grant (alice only) — the
        // live write-through re-applies the ACL, revoking bob. This is the
        // idempotent re-ingest path (same external_id), which still routes
        // the acl_grant through apply_doc_acl → durify.
        let recs2 = ingest(&backend, node_with_acl("doc:2", &["alice"]));
        assert_eq!(committed_id(&recs2), id, "same doc id (idempotent node)");
        assert!(
            !can_read(&backend, "bob", id),
            "bob revoked by the narrowed re-grant, in-session"
        );

        drop(guard);
        id
    };

    // Restart and verify the NARROWED set is what survives.
    let (backend2, _guard2) = bootstrap_storage_backend(&BootstrapMode::Durable {
        data_dir: data_dir.clone(),
    })
    .expect("durable bootstrap (session 2 — recover)");

    assert!(
        can_read(&backend2, "alice", doc),
        "#1221: alice keeps doc:2 after restart"
    );
    assert!(
        !can_read(&backend2, "bob", doc),
        "#1221: bob's revocation (narrowed re-grant) survived the restart — \
         last-writer-wins per doc across the v8 acl_grants commits"
    );
}
