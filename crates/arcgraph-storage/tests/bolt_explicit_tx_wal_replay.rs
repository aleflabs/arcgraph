//! ADR-197 (#802) R1 finding #2 — explicit-tx (`OwnedTxn`) COMMIT WAL
//! round-trip.
//!
//! Proves the Bolt explicit-transaction COMMIT path
//! (`OwnedTxn::into_inner` → [`arcgraph_storage::crud::commit`]) appends
//! a real `CommitBundle` to the WAL whose staged record + primary-index
//! pages SURVIVE a [`recover_from_wal`] round-trip — i.e. a managed-tx
//! write is durable + recoverable, NOT merely MVCC-version-store
//! visible.
//!
//! This is the WAL leg of R1 finding #1's regression coverage; the
//! primary-index + CDC legs are proven in `arcgraph-mcp`'s
//! `storage::bolt::txn_fault_injection` production-substrate suite
//! (`commit_persists_into_primary_index_and_cdc`). With the pre-fix
//! MVCC-only commit (`OwnedTxn::commit`) the `CommitBundle` carries NO
//! staged record page, so the recovered store would NOT find the node —
//! the same silent half-commit the primary-index oracle catches. Here we
//! drive the FIXED path (`crud::commit(owned.into_inner(), &store)` — the
//! exact call `CrudExecutorSubstrate::commit_held_txn` makes) and assert
//! durability.

use std::sync::Arc;

use arcgraph_core::{LabelId, TenantId};
use arcgraph_storage::crud::{
    CrudStore, PropertyData, commit, create_node, crud_allocator_seed_handle, read_node_with_store,
};
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::{PrimaryIndex, PrimaryKey, RecordKind};
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{
    AllocatorSeedHandle, BlobStoreHandle, PageStoreTarget, PrimaryPageStoreHandle,
    RecordPageStoreHandle, WalConfig, WalWriter, recover_from_wal,
};
use tempfile::TempDir;

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

/// Build a durable CRUD stack (WalWriter + WAL-aware TxnManager +
/// PrimaryIndex + CrudStore). Mirrors `wal_replay_round_trip.rs`'s
/// `build_stack` — the production wiring pattern.
fn build_stack(
    wal_dir: &std::path::Path,
) -> (
    WalWriter,
    Arc<TxnManager>,
    Arc<PrimaryIndex>,
    Arc<CrudStore>,
) {
    let writer = WalWriter::spawn(test_wal_config(wal_dir)).unwrap();
    let handle = writer.handle();
    let mgr = Arc::new(TxnManager::with_wal(handle.clone()));
    let alloc = Arc::new(PageAllocator::new());
    let primary = Arc::new(
        PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&alloc), Some(handle.clone())).unwrap(),
    );
    let store = Arc::new(CrudStore::new_with_index(
        Some(handle.clone()),
        Arc::clone(&primary),
        Arc::clone(&alloc),
    ));
    (writer, mgr, primary, store)
}

/// Recover a fresh stack from `wal_dir` — equivalent to a process
/// restart post-crash. Wires the replay target to the primary + record +
/// blob stores + allocator-seed handle (mirrors `wal_replay_round_trip.rs`'s
/// `recover_stack`).
fn recover_stack(
    wal_dir: &std::path::Path,
) -> (
    WalWriter,
    Arc<TxnManager>,
    Arc<PrimaryIndex>,
    Arc<CrudStore>,
) {
    let writer = WalWriter::spawn(test_wal_config(wal_dir)).unwrap();
    let handle = writer.handle();
    let mgr = Arc::new(TxnManager::with_wal(handle.clone()));
    let alloc = Arc::new(PageAllocator::new());
    let primary = Arc::new(
        PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&alloc), Some(handle.clone())).unwrap(),
    );
    let store = Arc::new(CrudStore::new_with_index(
        Some(handle.clone()),
        Arc::clone(&primary),
        Arc::clone(&alloc),
    ));
    let primary_handle: Arc<dyn PrimaryPageStoreHandle> =
        Arc::clone(primary.page_store()) as Arc<dyn PrimaryPageStoreHandle>;
    let records_handle: Arc<dyn RecordPageStoreHandle> = Arc::clone(
        store
            .records()
            .expect("CrudStore constructed via new_with_index exposes record store"),
    ) as Arc<dyn RecordPageStoreHandle>;
    let blob_handle: Arc<dyn BlobStoreHandle> =
        Arc::clone(store.blob_store()) as Arc<dyn BlobStoreHandle>;
    let allocator_seed: Arc<dyn AllocatorSeedHandle> =
        crud_allocator_seed_handle(Arc::clone(&store), Arc::clone(&alloc));
    let target = PageStoreTarget::primary_only(primary_handle)
        .with_record_store(records_handle)
        .with_blob_store(blob_handle)
        .with_allocator_seed(allocator_seed);
    let _report = recover_from_wal(wal_dir, Arc::clone(&mgr), target, None).unwrap();
    (writer, mgr, primary, store)
}

#[test]
fn explicit_tx_commit_survives_wal_replay() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();

    // Pre-crash: stage a CREATE into a HELD `OwnedTxn` (the Bolt
    // explicit-tx handle), then COMMIT through the SAME path the
    // finding-#1 fix routes the Bolt COMMIT through —
    // `crud::commit(owned.into_inner(), &store)` (the exact call
    // `CrudExecutorSubstrate::commit_held_txn` makes). This is the
    // explicit-tx analogue of the auto-commit `create_node` + `commit`.
    let (writer, mgr, primary, store) = build_stack(&wal_dir);
    let mut owned = mgr.begin_owned(TenantId::DEFAULT);
    let id = create_node(
        &store,
        owned.txn_mut(),
        TenantId::DEFAULT,
        LabelId::new(42),
        &PropertyData::InlineU32Pair(7, 9),
    )
    .unwrap();
    // NOTE: the load-bearing line. Swapping this for the MVCC-only
    // `owned.commit()` makes the post-replay `read_node_with_store`
    // return `None` (the `CommitBundle` would carry no staged record
    // page) → the `.expect(...)` below panics — the WAL-leg expression
    // of R1 finding #1.
    let commit_lsn = commit(owned.into_inner(), &store).unwrap();
    assert!(
        commit_lsn.raw() > 0,
        "explicit-tx commit must return a real CommitBundle LSN"
    );
    // Pre-replay sanity: the dual-write fired (primary index resolves it).
    assert!(
        primary
            .lookup(PrimaryKey::new(
                TenantId::DEFAULT,
                RecordKind::Node,
                id.raw()
            ))
            .unwrap()
            .is_some(),
        "explicit-tx commit must dual-write the primary index pre-replay"
    );

    // Shutdown (flush the WAL) + drop the stack — a process restart.
    writer.shutdown().unwrap();
    drop(store);
    drop(primary);
    drop(mgr);

    // Post-crash: fresh stack + recovery from the WAL.
    let (writer2, mgr2, primary2, store2) = recover_stack(&wal_dir);
    let tx2 = mgr2.begin(TenantId::DEFAULT);
    let rec = read_node_with_store(&store2, &tx2, id)
        .unwrap()
        .expect("explicit-tx-committed node must be recoverable from the WAL");
    assert_eq!(rec.label_id, 42);
    assert_eq!(rec.inline_u32a, 7);
    assert_eq!(rec.inline_u32b, 9);
    // And the recovered primary index resolves it too (the staged
    // primary-index page replayed).
    assert!(
        primary2
            .lookup(PrimaryKey::new(
                TenantId::DEFAULT,
                RecordKind::Node,
                id.raw()
            ))
            .unwrap()
            .is_some(),
        "recovered primary index must resolve the explicit-tx-committed node"
    );
    writer2.shutdown().unwrap();
}
