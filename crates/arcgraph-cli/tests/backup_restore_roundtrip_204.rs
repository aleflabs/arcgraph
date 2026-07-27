//! ADR-204 D-5 — the §5.4 PROVEN-restore gate.
//!
//! End-to-end round-trip over the REAL durable stack (mirrors
//! `durable_bootstrap_restart.rs`, the ADR-183 active verification):
//!
//! 1. bootstrap durable at dir A → commit nodes + a relationship
//!    through the production CRUD write path (Strict tier — fsync
//!    before ack) → capture the reference oracle (byte-level record
//!    fields + the production-substrate scan view) → graceful drop;
//! 2. `backup_create(A, B)` (the ADR-204 D-1/D-2 verb — exclusive
//!    LOCK + allowlist copy + checksummed manifest);
//! 3. `backup_restore(B, C)` (verify → refuse → place — D-4) into a
//!    FRESH dir C;
//! 4. bootstrap durable at **C** → the standard K-3-proven boot
//!    recovery replays the restored WAL → **exact equality** with
//!    the reference (record fields byte-identical; substrate scan
//!    count + (id, label) multiset identical) → and the restored
//!    store is LIVE (a post-restore commit succeeds + reads back).
//!
//! The negatives (tamper → loud verify failure with target untouched;
//! refuse-overwrite; locked-source; future-manifest-version) are
//! pinned as unit tests in `arcgraph_cli::ops::backup::tests` — this
//! file is the positive gate §5.4 flips on ("wired ≠ semantics-
//! shipped": the gate IS this test being green).
//!
//! Backup non-destructiveness is also pinned: after the backup, dir A
//! itself still boots + serves the same data (a backup must never
//! damage its source).

use std::sync::Arc;

use arcgraph_cli::bootstrap::{BootstrapMode, bootstrap_storage_backend};
use arcgraph_cli::ops::backup::{backup_create, backup_restore};
use arcgraph_core::{LabelId, NodeId, PartitionId, TenantId, TypeId};
use arcgraph_mcp::storage::{CrudExecutorSubstrate, StorageBackend};
use arcgraph_query::executor::substrate::ExecutorSubstrate;
use arcgraph_storage::crud::{
    CrudStore, PropertyData, commit, create_node, create_rel, read_node_with_store,
    read_rel_with_store,
};
use tempfile::TempDir;

fn crud_for(backend: &StorageBackend, tenant: TenantId) -> Arc<CrudStore> {
    backend
        .router()
        .route(tenant, PartitionId::ZERO)
        .expect("route tenant")
        .crud()
        .clone()
}

fn substrate_for(backend: &StorageBackend) -> CrudExecutorSubstrate {
    CrudExecutorSubstrate::new(
        Arc::clone(backend.router()),
        Arc::clone(backend.txn_manager()),
        Arc::clone(backend.intern_table()),
    )
}

fn commit_node(
    backend: &StorageBackend,
    crud: &Arc<CrudStore>,
    tenant: TenantId,
    label: u32,
    a: u32,
    b: u32,
) -> NodeId {
    let mut tx = backend.txn_manager().begin(tenant);
    let id = create_node(
        crud,
        &mut tx,
        tenant,
        LabelId::new(label),
        &PropertyData::InlineU32Pair(a, b),
    )
    .expect("create_node");
    commit(tx, crud).expect("commit node");
    id
}

/// The production-substrate scan view: sorted (node_id, label_id)
/// pairs — the query-path oracle (exactly what a served `MATCH (n)`
/// scans).
fn scan_view(backend: &StorageBackend, tenant: TenantId) -> Vec<(u64, u32)> {
    let substrate = substrate_for(backend);
    let mut view: Vec<(u64, u32)> = substrate
        .scan_nodes(tenant, None, arcgraph_core::Lsn::MAX)
        .expect("scan_nodes")
        .into_iter()
        .map(|n| (n.node.id.raw(), n.node.label.map_or(0, |l| l.raw())))
        .collect();
    view.sort_unstable();
    view
}

#[test]
fn backup_restore_round_trip_is_exactly_equal_and_live() {
    let tmp = TempDir::new().expect("tempdir");
    let dir_a = tmp.path().join("a"); // original store
    let dir_b = tmp.path().join("b"); // backup artifact
    let dir_c = tmp.path().join("c"); // restored store

    // ── Session 1 (dir A): ingest the reference dataset.
    let (src_id, dst_id, rel_id, reference_view) = {
        let (backend, guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
            data_dir: dir_a.clone(),
        })
        .expect("durable bootstrap (A)");
        assert!(guard.is_durable(), "Durable mode must own a WAL writer");
        let crud = crud_for(&backend, TenantId::DEFAULT);

        let src_id = commit_node(&backend, &crud, TenantId::DEFAULT, 7, 111, 222);
        let dst_id = commit_node(&backend, &crud, TenantId::DEFAULT, 8, 333, 444);
        // A third node so the count oracle is > the rel endpoints.
        let _extra = commit_node(&backend, &crud, TenantId::DEFAULT, 9, 555, 666);

        let mut tx = backend.txn_manager().begin(TenantId::DEFAULT);
        let rel_id = create_rel(
            &crud,
            &mut tx,
            TenantId::DEFAULT,
            src_id,
            dst_id,
            TypeId::new(5),
            &PropertyData::Empty,
        )
        .expect("create_rel");
        commit(tx, &crud).expect("commit rel");

        let reference_view = scan_view(&backend, TenantId::DEFAULT);
        assert_eq!(reference_view.len(), 3, "reference dataset is 3 nodes");

        (src_id, dst_id, rel_id, reference_view)
        // drop → WalWriter drains + fsyncs + joins; LOCK released.
    };

    // ── Backup A → B (cold; takes A's LOCK exclusively).
    let manifest = backup_create(&dir_a, &dir_b).expect("backup create");
    assert!(
        !manifest.files.is_empty(),
        "manifest must list the allowlisted files (wal segments at minimum)"
    );
    assert!(
        manifest.files.iter().any(|f| f.path.starts_with("wal/")),
        "the WAL — the source of truth — must be in the backup; got {:?}",
        manifest.files,
    );

    // ── Restore B → C (verify → refuse → place).
    let restored = backup_restore(&dir_b, &dir_c).expect("backup restore");
    assert_eq!(
        restored.files.len(),
        manifest.files.len(),
        "restore places exactly the manifest's file set"
    );

    // ── Session 2 (dir C — the RESTORED store): boot recovery replays.
    {
        let (backend_c, _guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
            data_dir: dir_c.clone(),
        })
        .expect("durable bootstrap (C — restored)");
        assert!(
            backend_c.router().tenants().contains(&TenantId::DEFAULT),
            "DEFAULT tenant present after restore + recovery"
        );

        let crud_c = crud_for(&backend_c, TenantId::DEFAULT);
        let tx = backend_c.txn_manager().begin(TenantId::DEFAULT);

        // Byte-level record oracle (the ADR-183 R1 form).
        let node = read_node_with_store(&crud_c, &tx, src_id)
            .expect("read node")
            .expect("node MUST survive backup→restore (ADR-204 D-5)");
        assert_eq!(node.label_id, 7);
        assert_eq!(node.inline_u32a, 111);
        assert_eq!(node.inline_u32b, 222);
        let rel = read_rel_with_store(&crud_c, &tx, rel_id)
            .expect("read rel")
            .expect("rel MUST survive backup→restore (ADR-204 D-5)");
        assert_eq!(rel.src_id, src_id.raw());
        assert_eq!(rel.dst_id, dst_id.raw());
        drop(tx);

        // Query-path oracle: the production substrate's scan view is
        // EXACTLY the reference (count + (id, label) multiset).
        let restored_view = scan_view(&backend_c, TenantId::DEFAULT);
        assert_eq!(
            restored_view, reference_view,
            "restored scan view must equal the pre-backup reference exactly"
        );

        // LIVE store: a post-restore commit succeeds and reads back —
        // restore yields a working database, not a read-only fossil.
        let new_id = commit_node(&backend_c, &crud_c, TenantId::DEFAULT, 10, 777, 888);
        let tx2 = backend_c.txn_manager().begin(TenantId::DEFAULT);
        let fresh = read_node_with_store(&crud_c, &tx2, new_id)
            .expect("read fresh")
            .expect("post-restore commit must read back");
        assert_eq!(fresh.label_id, 10, "restored store accepts new commits");
    }

    // ── Non-destructiveness: dir A still boots and serves the SAME
    //    reference data (backup never damages its source).
    {
        let (backend_a2, _guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
            data_dir: dir_a.clone(),
        })
        .expect("durable re-bootstrap (A — post-backup)");
        let view_a = scan_view(&backend_a2, TenantId::DEFAULT);
        assert_eq!(
            view_a, reference_view,
            "the backup source must be untouched by backup_create"
        );
    }
}
