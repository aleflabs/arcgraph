//! M3 consistency-neutral differential: the production v8 and v9 commit
//! paths expose identical MVCC, snapshot, and read-your-own-writes behavior.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use arcgraph_core::{LabelId, TenantId, TypeId};
use arcgraph_storage::DeltaPageStore;
use arcgraph_storage::crud::{
    CrudStore, PropertyData, commit, create_node, create_rel, crud_allocator_seed_handle,
    delete_node_with_store, delete_rel_with_store, node_mvcc_key, read_node, read_rel, update_node,
};
use arcgraph_storage::io::{InMemoryPageIo, PageIo};
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::page_store::{
    BufferedRecordPageStore, PerTenantBufferPool, PerTenantBufferPoolConfig,
};
use arcgraph_storage::primary_index::{PrimaryIndex, PrimaryPageStore};
use arcgraph_storage::record_store::RecordPageStore;
use arcgraph_storage::redo::DirtyPageTable;
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::segment::{SegmentHeader, segment_filename};
use arcgraph_storage::wal::{
    BUNDLE_FORMAT_V8, BUNDLE_FORMAT_V9, PageStoreTarget, ReplayConfig, ReplayExecutor, WalConfig,
    WalRecoveryReader, WalWriter,
};
use bytes::Bytes;
use tempfile::tempdir;

fn config(dir: PathBuf) -> WalConfig {
    WalConfig {
        dir,
        segment_size_bytes: 64 * 1024 * 1024,
        group_commit_window: Duration::from_millis(1),
        group_commit_max_batch: 16,
        metrics_sink: None,
        encryption: None,
        inflight_budget_bytes: None,
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Observations {
    create_ryow: (u32, u32),
    old_snapshot_after_update: (u32, u32),
    update_ryow: (u32, u32),
    current_after_update: (u32, u32),
    delete_ryow_none: bool,
    current_after_delete_none: bool,
    old_snapshot_after_delete: (u32, u32),
    rel_create_ryow: bool,
    rel_delete_ryow_none: bool,
    current_rel_after_delete_none: bool,
    commit_clock_strictly_increases: bool,
}

fn run(format_version: u16) -> Observations {
    let dir = tempdir().unwrap();
    std::fs::write(
        dir.path().join(segment_filename(0)),
        SegmentHeader { format_version }.encode(),
    )
    .unwrap();
    let manager = Arc::new(TxnManager::new());
    let allocator = Arc::new(PageAllocator::new());
    let primary =
        Arc::new(PrimaryIndex::new(Arc::clone(&manager), Arc::clone(&allocator), None).unwrap());
    let mut store = CrudStore::new_with_index(None, Arc::clone(&primary), allocator);
    let writer =
        WalWriter::spawn_from(config(dir.path().to_path_buf()), manager.current_lsn()).unwrap();
    let wal = writer.handle();
    manager.attach_wal(wal.clone());
    primary.attach_wal(wal.clone());
    store.attach_wal(wal);

    let mut create_tx = manager.begin(TenantId::DEFAULT);
    let node = create_node(
        &store,
        &mut create_tx,
        TenantId::DEFAULT,
        LabelId::new(1),
        &PropertyData::InlineU32Pair(1, 2),
    )
    .unwrap();
    let create_ryow_record = read_node(&create_tx, node).unwrap().unwrap();
    let create_ryow = (
        create_ryow_record.inline_u32a,
        create_ryow_record.inline_u32b,
    );
    let create_lsn = commit(create_tx, &store).unwrap();
    let old_snapshot = manager.begin(TenantId::DEFAULT);

    let mut update_tx = manager.begin(TenantId::DEFAULT);
    update_node(
        &store,
        &mut update_tx,
        node,
        &PropertyData::InlineU32Pair(7, 8),
    )
    .unwrap();
    let update_ryow_record = read_node(&update_tx, node).unwrap().unwrap();
    let update_ryow = (
        update_ryow_record.inline_u32a,
        update_ryow_record.inline_u32b,
    );
    let update_lsn = commit(update_tx, &store).unwrap();
    let old_after_update = read_node(&old_snapshot, node).unwrap().unwrap();
    let current_after_update_tx = manager.begin(TenantId::DEFAULT);
    let current_after_update_record = read_node(&current_after_update_tx, node).unwrap().unwrap();
    current_after_update_tx.abort();

    let mut delete_tx = manager.begin(TenantId::DEFAULT);
    delete_node_with_store(&store, &mut delete_tx, node).unwrap();
    let delete_ryow_none = read_node(&delete_tx, node).unwrap().is_none();
    let delete_lsn = commit(delete_tx, &store).unwrap();
    let current_after_delete = manager.begin(TenantId::DEFAULT);
    let current_after_delete_none = read_node(&current_after_delete, node).unwrap().is_none();
    current_after_delete.abort();
    let old_after_delete = read_node(&old_snapshot, node).unwrap().unwrap();
    old_snapshot.abort();

    let mut endpoints_tx = manager.begin(TenantId::DEFAULT);
    let src = create_node(
        &store,
        &mut endpoints_tx,
        TenantId::DEFAULT,
        LabelId::new(2),
        &PropertyData::Empty,
    )
    .unwrap();
    let dst = create_node(
        &store,
        &mut endpoints_tx,
        TenantId::DEFAULT,
        LabelId::new(2),
        &PropertyData::Empty,
    )
    .unwrap();
    commit(endpoints_tx, &store).unwrap();
    let mut rel_tx = manager.begin(TenantId::DEFAULT);
    let rel = create_rel(
        &store,
        &mut rel_tx,
        TenantId::DEFAULT,
        src,
        dst,
        TypeId::new(3),
        &PropertyData::Empty,
    )
    .unwrap();
    let rel_create_ryow = read_rel(&rel_tx, rel).unwrap().is_some();
    commit(rel_tx, &store).unwrap();
    let mut rel_delete_tx = manager.begin(TenantId::DEFAULT);
    delete_rel_with_store(&store, &mut rel_delete_tx, rel).unwrap();
    let rel_delete_ryow_none = read_rel(&rel_delete_tx, rel).unwrap().is_none();
    commit(rel_delete_tx, &store).unwrap();
    let current_rel = manager.begin(TenantId::DEFAULT);
    let current_rel_after_delete_none = read_rel(&current_rel, rel).unwrap().is_none();
    current_rel.abort();
    writer.shutdown().unwrap();

    Observations {
        create_ryow,
        old_snapshot_after_update: (old_after_update.inline_u32a, old_after_update.inline_u32b),
        update_ryow,
        current_after_update: (
            current_after_update_record.inline_u32a,
            current_after_update_record.inline_u32b,
        ),
        delete_ryow_none,
        current_after_delete_none,
        old_snapshot_after_delete: (old_after_delete.inline_u32a, old_after_delete.inline_u32b),
        rel_create_ryow,
        rel_delete_ryow_none,
        current_rel_after_delete_none,
        commit_clock_strictly_increases: create_lsn < update_lsn && update_lsn < delete_lsn,
    }
}

#[test]
fn production_v9_mvcc_commit_and_ryow_match_v8() {
    let v8 = run(BUNDLE_FORMAT_V8);
    let v9 = run(BUNDLE_FORMAT_V9);
    assert_eq!(v9, v8);
    assert_eq!(v9.create_ryow, (1, 2));
    assert_eq!(v9.old_snapshot_after_update, (1, 2));
    assert_eq!(v9.update_ryow, (7, 8));
    assert_eq!(v9.current_after_update, (7, 8));
    assert!(v9.delete_ryow_none);
    assert!(v9.current_after_delete_none);
    assert_eq!(v9.old_snapshot_after_delete, (1, 2));
    assert!(v9.rel_create_ryow);
    assert!(v9.rel_delete_ryow_none);
    assert!(v9.current_rel_after_delete_none);
    assert!(v9.commit_clock_strictly_increases);
}

#[derive(Debug, PartialEq, Eq)]
struct RecoveredCanonical {
    mvcc: Vec<(u64, Option<Bytes>)>,
    primary_pages: Vec<(u64, Vec<u8>)>,
    record_pages: Vec<(u64, Vec<Option<Vec<u8>>>)>,
    allocator_advances: Vec<(u8, u64)>,
}

fn buffered_recovery_store() -> Arc<BufferedRecordPageStore> {
    let io: Arc<dyn PageIo> = Arc::new(InMemoryPageIo::new());
    let pools = Arc::new(PerTenantBufferPool::with_config(
        io,
        PerTenantBufferPoolConfig {
            frames_per_tenant: 32,
            write_fraction: 0.0,
        },
    ));
    Arc::new(BufferedRecordPageStore::with_cache_cap(pools, 64))
}

fn crash_and_recover_update_delete(format_version: u16) -> RecoveredCanonical {
    let dir = tempdir().unwrap();
    std::fs::write(
        dir.path().join(segment_filename(0)),
        SegmentHeader { format_version }.encode(),
    )
    .unwrap();
    let manager = Arc::new(TxnManager::new());
    let allocator = Arc::new(PageAllocator::new());
    let primary =
        Arc::new(PrimaryIndex::new(Arc::clone(&manager), Arc::clone(&allocator), None).unwrap());
    let mut store = CrudStore::new_with_index(None, Arc::clone(&primary), allocator);
    let writer =
        WalWriter::spawn_from(config(dir.path().to_path_buf()), manager.current_lsn()).unwrap();
    let wal = writer.handle();
    manager.attach_wal(wal.clone());
    primary.attach_wal(wal.clone());
    store.attach_wal(wal);

    let mut create = manager.begin(TenantId::DEFAULT);
    let updated = create_node(
        &store,
        &mut create,
        TenantId::DEFAULT,
        LabelId::new(7),
        &PropertyData::InlineU32Pair(1, 2),
    )
    .unwrap();
    let deleted = create_node(
        &store,
        &mut create,
        TenantId::DEFAULT,
        LabelId::new(8),
        &PropertyData::InlineU32Pair(3, 4),
    )
    .unwrap();
    commit(create, &store).unwrap();
    let mut mutate = manager.begin(TenantId::DEFAULT);
    update_node(
        &store,
        &mut mutate,
        updated,
        &PropertyData::InlineU32Pair(70, 80),
    )
    .unwrap();
    delete_node_with_store(&store, &mut mutate, deleted).unwrap();
    commit(mutate, &store).unwrap();
    writer.shutdown().unwrap();
    drop(store);
    drop(primary);
    drop(manager);

    let recovered_txn = Arc::new(TxnManager::new());
    let recovered_primary = Arc::new(PrimaryPageStore::new());
    let recovered_allocator = Arc::new(PageAllocator::new());
    let recovered_crud = Arc::new(CrudStore::new());
    let seed = crud_allocator_seed_handle(
        Arc::clone(&recovered_crud),
        Arc::clone(&recovered_allocator),
    );
    let (target, raw_records, buffered_records) = if format_version == BUNDLE_FORMAT_V8 {
        let records = Arc::new(RecordPageStore::new());
        (
            PageStoreTarget::primary_only(Arc::clone(&recovered_primary) as _)
                .with_record_store(Arc::clone(&records) as _)
                .with_allocator_seed(seed),
            Some(records),
            None,
        )
    } else {
        let props = buffered_recovery_store();
        let records = buffered_recovery_store();
        (
            PageStoreTarget::primary_only(Arc::clone(&recovered_primary) as _)
                .with_delta_stores(
                    props,
                    Arc::clone(&records) as Arc<dyn DeltaPageStore>,
                    Arc::new(DirtyPageTable::new()),
                )
                .with_allocator_seed(seed),
            None,
            Some(records),
        )
    };
    ReplayExecutor::new(
        ReplayConfig::with_wal_dir(dir.path()),
        Arc::clone(&recovered_txn),
        target,
    )
    .run(WalRecoveryReader::open(dir.path()).unwrap())
    .unwrap();

    let reader = recovered_txn.begin(TenantId::DEFAULT);
    let mvcc: Vec<_> = [updated.raw(), deleted.raw()]
        .into_iter()
        .map(|id| {
            let canonical = reader
                .read(node_mvcc_key(arcgraph_core::NodeId::new(id)))
                .map(|bytes| {
                    Bytes::copy_from_slice(&bytes[..arcgraph_core::record::NodeRecord::SIZE - 8])
                });
            (id, canonical)
        })
        .collect();
    reader.abort();
    let mut primary_pages: Vec<_> = recovered_primary
        .iter_pages()
        .into_iter()
        .map(|(id, latch)| (id.raw(), latch.read().as_ref().to_vec()))
        .collect();
    primary_pages.sort_by_key(|entry| entry.0);

    let page_ids: Vec<_> = if let Some(records) = &raw_records {
        records.iter_pages().into_iter().map(|(id, _)| id).collect()
    } else {
        // The identical workload allocates one direct record page. Its id is
        // format-neutral; discover it from the v9 tracked-page namespace by
        // probing the bounded allocator range used by this two-node history.
        (1..=16)
            .map(arcgraph_core::PageId::new)
            .filter(|id| {
                buffered_records
                    .as_ref()
                    .unwrap()
                    .read_page_for_redo(TenantId::DEFAULT, *id)
                    .unwrap()
                    .is_some()
            })
            .collect()
    };
    let mut record_pages = Vec::new();
    for id in page_ids {
        let bytes = if let Some(records) = &raw_records {
            records.latch(id).unwrap().read().clone()
        } else {
            buffered_records
                .as_ref()
                .unwrap()
                .read_page_for_redo(TenantId::DEFAULT, id)
                .unwrap()
                .unwrap()
                .clone()
        };
        let view = arcgraph_storage::records::SlottedPageRef::open(bytes.as_ref()).unwrap();
        let slots = (0..view.slot_count())
            .map(|slot| {
                view.read_node(arcgraph_storage::records::SlotId(slot))
                    .unwrap()
                    .map(|record| {
                        record.to_bytes()[..arcgraph_core::record::NodeRecord::SIZE - 8].to_vec()
                    })
            })
            .collect();
        record_pages.push((id.raw(), slots));
    }
    record_pages.sort_by_key(|entry| entry.0);
    let mut allocator_advances: Vec<_> = recovered_crud
        .snapshot_allocator_advances()
        .into_iter()
        .map(|advance| (advance.kind.as_byte(), advance.new_high_water))
        .collect();
    allocator_advances.sort_unstable();
    RecoveredCanonical {
        mvcc,
        primary_pages,
        record_pages,
        allocator_advances,
    }
}

#[test]
fn identical_production_v8_v9_workload_crash_recovers_byte_equal_update_delete() {
    let v8 = crash_and_recover_update_delete(BUNDLE_FORMAT_V8);
    let v9 = crash_and_recover_update_delete(BUNDLE_FORMAT_V9);
    assert_eq!(v9.mvcc, v8.mvcc, "MVCC canonical bytes differ");
    assert_eq!(v9.primary_pages, v8.primary_pages, "primary owner differs");
    assert_eq!(v9.record_pages, v8.record_pages, "record owner differs");
    assert_eq!(
        v9.allocator_advances, v8.allocator_advances,
        "allocator owner differs"
    );
    assert!(
        v9.mvcc[0].1.is_some(),
        "durable update vanished on recovery"
    );
    assert!(
        v9.mvcc[1].1.is_none(),
        "durable delete resurrected on recovery"
    );
}
