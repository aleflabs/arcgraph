//! #1379 (MUST-CON-04) — storage-backed oracle: DELETE revokes the
//! deleted node's doc-ACL so it stops leaking, and the revoke is DURABLE
//! across a bare restart.
//!
//! # The bug this pins
//!
//! `crud::delete_node_with_store` tombstones the MVCC record but does NOT
//! touch the tenant's `PermissionIndex`. So after a delete the node's
//! `doc_class` mapping SURVIVES: `is_visible(id, P)` stays `true` for
//! every principal `P` it was granted to, and — because the BM25 / vector
//! substrates tombstone lazily on a SEPARATE seam — the deleted node keeps
//! leaking to its principals AND stays retrievable. A live data-leak.
//!
//! # THE ORACLE (Director-required, storage-backed, RED-on-revert)
//!
//! [`delete_revokes_acl_and_purges_bm25`] seeds a node against a REAL
//! `CrudStore` (dual-write primary index) + grants a doc-ACL visible to
//! principal `alice` + indexes it in a REAL `Bm25Service` → asserts
//! `is_visible == true` AND BM25 returns it AND the MVCC record reads live
//! → deletes via [`delete_node_with_store_and_revoke`] (+ the BM25
//! `delete_document` the served path drives) → asserts (a)
//! `is_visible == false`, (b) BM25 no longer returns it, (c) the MVCC
//! record reads not-live (tombstoned). NOT a stub — every hit comes off a
//! real seed→index→delete→search corpus.
//!
//! RED-on-revert is proven VERBATIM by
//! [`revert_delete_without_revoke_still_leaks`]: it runs the identical
//! seed then deletes via the OLD [`delete_node_with_store`] (no revoke) —
//! and asserts the node STILL leaks (`is_visible == true`). Removing the
//! `revoke_doc` call from the fix collapses the two tests into "both
//! leak"; the passing oracle above is what fails on that revert.
//!
//! # Durability ([`revoke_survives_restart`])
//!
//! Delete-with-revoke through the durable write-through (a real
//! `CrudAclWalSink`) → fsync/"crash" → re-open a FRESH `PermissionIndex`
//! off the SAME WAL dir with NO re-seed → the revoke SURVIVED: the node
//! stays invisible post-restart (the WAL `Revoke` replayed).

use std::collections::BTreeSet;
use std::sync::Arc;

use arcgraph_bm25::{Bm25Service, IndexId as Bm25IndexId};
use arcgraph_core::{LabelId, Lsn, NodeId, TenantId};
use arcgraph_storage::crud::{
    CrudAclWalSink, CrudStore, PropertyData, commit, create_node, delete_node_with_store,
    delete_node_with_store_and_revoke, read_node_with_store,
};
use arcgraph_storage::mutation_log::Bm25IndexStoreHandle;
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::permissions::PermissionIndex;
use arcgraph_storage::primary_index::PrimaryIndex;
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{
    PageStoreTarget, PrimaryPageStoreHandle, WalConfig, WalWriter, recover_from_wal,
};
use tempfile::TempDir;

const TENANT: TenantId = TenantId::DEFAULT;

fn grants(items: &[&str]) -> BTreeSet<String> {
    items.iter().map(|s| (*s).to_owned()).collect()
}

fn test_wal_config(dir: &std::path::Path) -> WalConfig {
    WalConfig {
        dir: dir.to_path_buf(),
        segment_size_bytes: 64 * 1024 * 1024,
        group_commit_window: std::time::Duration::from_millis(2),
        group_commit_max_batch: 4,
        metrics_sink: None,
        encryption: None,
        inflight_budget_bytes: None,
    }
}

/// A real dual-write `CrudStore` (primary index attached), plus the
/// pieces the delete path needs. Ephemeral (no WAL) — used by the two
/// leak oracles; the durability test wires the WAL separately.
struct Stack {
    mgr: Arc<TxnManager>,
    store: Arc<CrudStore>,
    permissions: Arc<PermissionIndex>,
}

fn build_stack() -> Stack {
    let mgr = Arc::new(TxnManager::new());
    let alloc = Arc::new(PageAllocator::new());
    let primary =
        Arc::new(PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&alloc), None).expect("primary"));
    let store = Arc::new(CrudStore::new_with_index(
        None,
        Arc::clone(&primary),
        Arc::clone(&alloc),
    ));
    Stack {
        mgr,
        store,
        permissions: Arc::new(PermissionIndex::new()),
    }
}

/// Seed one node in MVCC + primary index, return its id.
fn seed_node(stack: &Stack, label: u32) -> NodeId {
    let mut tx = stack.mgr.begin(TENANT);
    let id = create_node(
        &stack.store,
        &mut tx,
        TENANT,
        LabelId::new(label),
        &PropertyData::InlineU32Pair(1, 2),
    )
    .expect("create_node");
    commit(tx, &stack.store).expect("commit seed");
    id
}

/// Build a real per-tenant BM25 handle + upsert `(node, body)` at
/// `commit_lsn`, then commit-pending so the doc is queryable.
fn seed_bm25(
    svc: &Arc<Bm25Service>,
    node: NodeId,
    body: &str,
) -> Arc<arcgraph_bm25::Bm25IndexHandle> {
    let bm = svc
        .handle(TENANT, Bm25IndexId::DEFAULT_BM25)
        .expect("bm25 handle");
    bm.upsert_document(node, body, Lsn::new(10))
        .expect("bm25 upsert");
    let store: Arc<dyn Bm25IndexStoreHandle> = svc.clone();
    store.commit_pending(TENANT).expect("bm25 commit_pending");
    bm
}

/// `true` iff a live MVCC record still reads back for `id` (the
/// `read_node`-liveness signal the `search_filtered` belt uses).
fn mvcc_live(stack: &Stack, id: NodeId) -> bool {
    let tx = stack.mgr.begin(TENANT);
    matches!(read_node_with_store(&stack.store, &tx, id), Ok(Some(_)))
}

// ─────────────────────────────────────────────────────────────────────
// THE ORACLE — delete-with-revoke stops the leak (is_visible + BM25).
// RED-on-revert twin below (`revert_delete_without_revoke_still_leaks`).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn delete_revokes_acl_and_purges_bm25() {
    let stack = build_stack();
    let tmp = TempDir::new().expect("tempdir");
    let svc = Bm25Service::new(tmp.path().join("bm25"));

    // ── Seed: node in MVCC, ACL grant → alice, BM25 doc. ──
    let id = seed_node(&stack, 7);
    stack.permissions.apply_doc_acl(id, grants(&["alice"]));
    let bm = seed_bm25(&svc, id, "secret keyword payload");

    // ── Pre-delete: alice SEES it, BM25 RETURNS it, MVCC is live. ──
    assert!(
        stack.permissions.effective("alice").is_visible(id),
        "pre-delete: alice must see the granted node"
    );
    let bm_pre = bm
        .search("secret", 10, Lsn::new(u64::MAX - 1))
        .expect("bm25 search pre-delete");
    assert!(
        bm_pre.iter().any(|(n, _)| *n == id),
        "pre-delete: BM25 must return the seeded node; got {bm_pre:?}"
    );
    assert!(
        mvcc_live(&stack, id),
        "pre-delete: MVCC record must be live"
    );

    // ── DELETE via the fixed path (revokes the doc-ACL). Also drive the
    //    BM25 `delete_document` the served substrate's
    //    `mark_bm25_node_deleted` seam drives, so BM25 stops matching. ──
    let mut del = stack.mgr.begin(TENANT);
    delete_node_with_store_and_revoke(&stack.store, &mut del, id, &stack.permissions)
        .expect("delete-with-revoke");
    commit(del, &stack.store).expect("commit delete");
    bm.delete_document(id, Lsn::MAX).expect("bm25 delete");
    let store: Arc<dyn Bm25IndexStoreHandle> = svc.clone();
    store.commit_pending(TENANT).expect("bm25 commit_pending");

    // ── Post-delete: the leak is CLOSED on all three planes. ──
    assert!(
        !stack.permissions.effective("alice").is_visible(id),
        "#1379: after delete-with-revoke, alice must NOT see the node \
         (is_visible must be false — the doc-ACL was revoked)"
    );
    let bm_post = bm
        .search("secret", 10, Lsn::new(u64::MAX - 1))
        .expect("bm25 search post-delete");
    assert!(
        !bm_post.iter().any(|(n, _)| *n == id),
        "#1379: after delete, BM25 must NOT return the node; got {bm_post:?}"
    );
    assert!(
        !mvcc_live(&stack, id),
        "#1379: after delete, the MVCC record must read not-live (tombstoned) \
         — the liveness signal the search_filtered belt uses"
    );
}

// ─────────────────────────────────────────────────────────────────────
// RED-on-revert (verbatim): the OLD `delete_node_with_store` (no revoke)
// leaves the node VISIBLE — the exact #1379 leak. This is what the oracle
// above catches; removing the `revoke_doc` from the fix makes this the
// observed behavior of the fixed path too, i.e. the oracle goes RED.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn revert_delete_without_revoke_still_leaks() {
    let stack = build_stack();

    let id = seed_node(&stack, 7);
    stack.permissions.apply_doc_acl(id, grants(&["alice"]));
    assert!(
        stack.permissions.effective("alice").is_visible(id),
        "pre-delete: alice must see the granted node"
    );

    // Delete WITHOUT the revoke (the pre-#1379 behavior).
    let mut del = stack.mgr.begin(TENANT);
    delete_node_with_store(&stack.store, &mut del, id).expect("delete (no revoke)");
    commit(del, &stack.store).expect("commit delete");

    // THE LEAK: the MVCC record is tombstoned, yet the doc-ACL SURVIVES,
    // so alice STILL sees the deleted node. This is the vulnerability
    // `delete_node_with_store_and_revoke` closes; if the fix's revoke were
    // removed, the oracle test would observe THIS (is_visible == true) and
    // fail.
    assert!(
        !mvcc_live(&stack, id),
        "delete tombstones the MVCC record (that part always worked)"
    );
    assert!(
        stack.permissions.effective("alice").is_visible(id),
        "#1379 leak reproduced: delete WITHOUT revoke leaves the doc-ACL \
         intact → alice STILL sees the deleted node (is_visible == true)"
    );
}

// ─────────────────────────────────────────────────────────────────────
// Durability — the revoke survives a bare restart via the WAL `Revoke`.
// ─────────────────────────────────────────────────────────────────────

fn recover_index(wal_dir: &std::path::Path, index: Arc<PermissionIndex>) {
    let mgr = Arc::new(TxnManager::new());
    let alloc = Arc::new(PageAllocator::new());
    let primary =
        Arc::new(PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&alloc), None).expect("primary"));
    let primary_handle: Arc<dyn PrimaryPageStoreHandle> =
        Arc::clone(primary.page_store()) as Arc<dyn PrimaryPageStoreHandle>;
    let target = PageStoreTarget::primary_only(primary_handle).with_permission_index(index);
    recover_from_wal(wal_dir, mgr, target, None).expect("recover_from_wal");
}

#[test]
fn revoke_survives_restart() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();

    // ── Process 1: grant → alice, then delete-with-revoke through the
    //    DURABLE write-through (a real CrudAclWalSink), then "crash". ──
    let id = NodeId::new(42);
    {
        let mgr = Arc::new(TxnManager::new());
        let alloc = Arc::new(PageAllocator::new());
        let writer = WalWriter::spawn(test_wal_config(&wal_dir)).unwrap();
        mgr.attach_wal(writer.handle());
        // Mirror the acl_wal_replay_1221 test-harness symmetry: build a
        // PrimaryIndex on the write manager so its root-page alloc advances
        // the write-side LSN in lockstep with the recovery-side manager
        // (which also builds one). See that test's rationale note.
        let primary =
            Arc::new(PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&alloc), None).unwrap());
        let mut s = CrudStore::new();
        s.attach_wal(writer.handle());
        s.attach_primary_index(Arc::clone(&primary));
        let store = Arc::new(s);

        let index = PermissionIndex::new();
        index.set_wal_sink(Arc::new(CrudAclWalSink::new(
            Arc::clone(&mgr),
            Arc::clone(&store),
            TENANT,
        )));

        // Grant then revoke via the DURABLE write-through (each op rides
        // its own single-op v8 commit). `revoke_doc` durifies a `Revoke`.
        index.apply_doc_acl(id, grants(&["alice"]));
        assert!(
            index.effective("alice").is_visible(id),
            "in-process pre-revoke: alice sees the node"
        );
        index.revoke_doc(id);
        assert!(
            !index.effective("alice").is_visible(id),
            "in-process post-revoke: alice no longer sees the node"
        );

        writer.shutdown().unwrap();
    }

    // ── Process 2 (bare restart): FRESH index off the SAME WAL, NO
    //    re-seed. Replay must re-drive apply THEN revoke ⇒ still invisible. ──
    let recovered = Arc::new(PermissionIndex::new());
    recover_index(&wal_dir, Arc::clone(&recovered));

    assert!(
        !recovered.effective("alice").is_visible(id),
        "#1379 durability: after restart, the delete's doc-ACL revoke must \
         have SURVIVED (WAL Revoke replayed) — alice must still NOT see the \
         deleted node"
    );
}
