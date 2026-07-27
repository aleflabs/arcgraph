//! M3 phase-5 physical recovery gates: lifecycle, sub-LSN idempotence,
//! surviving-torn-page failure, and the Director-ruling store-5 rejection.

use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex};

use arcgraph_core::record::{NodeRecord, PAGE_SIZE, PageType, RelRecord};
use arcgraph_core::{ArcGraphError, LabelId, Lsn, NodeId, PageId, RelId, TenantId, TypeId};
use arcgraph_storage::blob::BlobStore;
use arcgraph_storage::crud::{CrudStore, read_rel_with_store, rel_mvcc_key};
use arcgraph_storage::idempotency::IdempotencyStore;
use arcgraph_storage::io::PageBuf;
use arcgraph_storage::io::{InMemoryPageIo, PageIo};
use arcgraph_storage::m3_migration::{
    M3_PROPS_STORE_FILE, M3_RECORD_STORE_FILE, bootstrap_primary_from_v9_base,
    load_v9_physical_base, m3_record_store_path,
};
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::page_store::{
    BufferedRecordPageStore, PerTenantBufferPool, PerTenantBufferPoolConfig, TenantFilePageIo,
    TenantPageIo,
};
use arcgraph_storage::permissions::PermissionIndex;
use arcgraph_storage::primary_index::{PrimaryIndex, PrimaryKey, PrimaryPageStore, RecordKind};
use arcgraph_storage::records::{SlotId, SlottedPage, SlottedPageRef};
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{
    AclGrantEntry, AclGrantOp, AllocatorAdvance, AllocatorKind, AllocatorSeedHandle,
    BUNDLE_FORMAT_V9, DeltaOp, DeltaOpKind, IdempotencyBindingEntry, IdempotencyBindingOp,
    PageStoreTarget, ReplayConfig, ReplayExecutor, STORE_BLOB_OVERFLOW, STORE_RECORD,
    SegmentHeader, WalRecord, WalRecordType, WalRecoveryReader, encode_commit_bundle_v9,
    segment_filename,
};
use arcgraph_storage::{
    DeltaPageStore, DirtyPageKey, DirtyPageTable, RecoveryDeltaOutcome, apply_recovery_delta,
};
use bytes::Bytes;
use tempfile::tempdir;

#[derive(Default)]
struct MemoryDeltaStore {
    pages: Mutex<HashMap<PageId, Box<PageBuf>>>,
}

impl DeltaPageStore for MemoryDeltaStore {
    fn read_page_for_redo(
        &self,
        _tenant: TenantId,
        page_id: PageId,
    ) -> arcgraph_core::Result<Option<Box<PageBuf>>> {
        Ok(self.pages.lock().unwrap().get(&page_id).cloned())
    }

    fn install_page_from_redo(
        &self,
        _tenant: TenantId,
        page_id: PageId,
        page: Box<PageBuf>,
    ) -> arcgraph_core::Result<()> {
        self.pages.lock().unwrap().insert(page_id, page);
        Ok(())
    }
}

fn page_alloc(page_no: u64, op_lsn: u64) -> DeltaOp {
    let mut payload = Vec::with_capacity(9);
    payload.push(PageType::Node.as_byte());
    payload.extend_from_slice(&1u64.to_le_bytes());
    DeltaOp::new(
        DeltaOpKind::PageAlloc,
        STORE_RECORD,
        TenantId::DEFAULT,
        page_no,
        0,
        Lsn::new(op_lsn),
        Bytes::from(payload),
    )
    .unwrap()
}

fn put_node(page_no: u64, op_lsn: u64, node: u64) -> DeltaOp {
    let record = NodeRecord::new(NodeId::new(node), LabelId::new(3), Lsn::new(op_lsn));
    DeltaOp::new(
        DeltaOpKind::PutRecord,
        STORE_RECORD,
        TenantId::DEFAULT,
        page_no,
        0,
        Lsn::new(op_lsn),
        Bytes::copy_from_slice(&record.to_bytes()),
    )
    .unwrap()
}

#[test]
fn primary_bootstrap_reads_redone_record_cache_not_stale_home_file() {
    let dir = tempdir().unwrap();
    let page_id = PageId::new(0);
    let mut stale = Box::new([0u8; PAGE_SIZE]);
    let mut stale_page = SlottedPage::init(
        stale.as_mut(),
        arcgraph_core::PageHeader::new(page_id, PageType::Node, TenantId::DEFAULT),
    )
    .unwrap();
    stale_page
        .put_node_at(
            SlotId(0),
            &NodeRecord::new(NodeId::new(1), LabelId::new(1), Lsn::new(5)),
        )
        .unwrap();
    std::fs::write(dir.path().join(M3_RECORD_STORE_FILE), stale.as_ref()).unwrap();

    let io: Arc<dyn PageIo> = Arc::new(InMemoryPageIo::new());
    let pools = Arc::new(PerTenantBufferPool::with_config(
        io,
        PerTenantBufferPoolConfig {
            frames_per_tenant: 4,
            write_fraction: 0.0,
        },
    ));
    let records = BufferedRecordPageStore::with_cache_cap(pools, 4);
    let mut redone = Box::new([0u8; PAGE_SIZE]);
    let mut redone_page = SlottedPage::init(
        redone.as_mut(),
        arcgraph_core::PageHeader::new(page_id, PageType::Node, TenantId::DEFAULT),
    )
    .unwrap();
    redone_page
        .put_node_at(
            SlotId(0),
            &NodeRecord::new(NodeId::new(2), LabelId::new(2), Lsn::new(10)),
        )
        .unwrap();
    records
        .install_page_from_redo(TenantId::DEFAULT, page_id, redone)
        .unwrap();

    let txn = Arc::new(TxnManager::new());
    let allocator = Arc::new(PageAllocator::new());
    let primary = PrimaryIndex::new(txn, allocator, None).unwrap();
    bootstrap_primary_from_v9_base(&records, &primary, Lsn::new(10)).unwrap();
    assert!(
        primary
            .lookup(PrimaryKey::new(TenantId::DEFAULT, RecordKind::Node, 1))
            .unwrap()
            .is_none(),
        "stale home record must not be re-indexed after redo"
    );
    assert_eq!(
        primary
            .lookup(PrimaryKey::new(TenantId::DEFAULT, RecordKind::Node, 2))
            .unwrap()
            .unwrap()
            .slot,
        SlotId(0)
    );
}

#[test]
fn v9_base_load_and_primary_fast_path_hide_expired_relationship() {
    let dir = tempdir().unwrap();
    let page_id = PageId::new(0);
    let rel_id = RelId::new(77);
    let mut image = Box::new([0u8; PAGE_SIZE]);
    let mut page = SlottedPage::init(
        image.as_mut(),
        arcgraph_core::PageHeader::new(page_id, PageType::Rel, TenantId::DEFAULT),
    )
    .unwrap();
    let mut expired = RelRecord::new(
        rel_id,
        TypeId::new(4),
        NodeId::new(1),
        NodeId::new(2),
        Lsn::new(5),
    );
    expired.expired_lsn = 9;
    page.put_rel_at(SlotId(0), &expired).unwrap();
    let record_path = m3_record_store_path(dir.path(), TenantId::DEFAULT);
    std::fs::create_dir_all(record_path.parent().unwrap()).unwrap();
    std::fs::write(&record_path, image.as_ref()).unwrap();
    std::fs::write(dir.path().join(M3_PROPS_STORE_FILE), []).unwrap();

    let io: Arc<dyn TenantPageIo> =
        Arc::new(TenantFilePageIo::new(dir.path(), M3_RECORD_STORE_FILE));
    let pools = Arc::new(PerTenantBufferPool::with_tenant_io(
        io,
        PerTenantBufferPoolConfig {
            frames_per_tenant: 4,
            write_fraction: 0.0,
        },
    ));
    let records = Arc::new(BufferedRecordPageStore::with_cache_cap(pools, 4));
    let txn = Arc::new(TxnManager::new());
    let blob = Arc::new(BlobStore::new());
    load_v9_physical_base(dir.path(), Lsn::new(10), &txn, &records, &blob).unwrap();

    let allocator = Arc::new(PageAllocator::new());
    let primary =
        Arc::new(PrimaryIndex::new(Arc::clone(&txn), Arc::clone(&allocator), None).unwrap());
    let key = PrimaryKey::new(TenantId::DEFAULT, RecordKind::Rel, rel_id.raw());
    primary
        .upsert(
            key,
            arcgraph_storage::primary_index::PageSlot::new(page_id, SlotId(0)),
        )
        .unwrap();
    bootstrap_primary_from_v9_base(&records, &primary, Lsn::new(10)).unwrap();
    assert!(
        primary.lookup(key).unwrap().is_none(),
        "recovery reconciliation must remove an expired retained owner-2 entry"
    );

    let store = CrudStore::new_with_existing_buffered_page_store(
        Some(Arc::clone(&primary)),
        None,
        allocator,
        Arc::clone(&records),
        blob,
    );
    let reader = txn.begin(TenantId::DEFAULT);
    assert!(
        reader.read(rel_mvcc_key(rel_id)).is_none(),
        "base load must rebuild the finite expired_lsn as an MVCC tombstone"
    );

    // Re-introduce a stale accelerator entry to isolate the served fast-path
    // guard from the reconciliation assertion above.
    primary
        .upsert(
            key,
            arcgraph_storage::primary_index::PageSlot::new(page_id, SlotId(0)),
        )
        .unwrap();
    assert!(
        read_rel_with_store(&store, &reader, rel_id)
            .unwrap()
            .is_none(),
        "a stale primary hit must not bypass the relationship's upper MVCC bound"
    );
    reader.abort();
}

#[test]
fn missing_formatted_live_and_double_replay_are_total() {
    let props = MemoryDeltaStore::default();
    let records = MemoryDeltaStore::default();
    let dpt = DirtyPageTable::new();
    let alloc = page_alloc(7, 10);
    let put = put_node(7, 11, 99);

    assert_eq!(
        apply_recovery_delta(&props, &records, &dpt, &alloc, Lsn::new(11)).unwrap(),
        RecoveryDeltaOutcome::Formatted
    );
    assert_eq!(
        apply_recovery_delta(&props, &records, &dpt, &put, Lsn::new(11)).unwrap(),
        RecoveryDeltaOutcome::Applied
    );
    let before = dpt.snapshot();
    assert_eq!(before.len(), 1);
    assert_eq!(before[0].rec_lsn, Lsn::new(10));
    assert_eq!(before[0].dirty_gen, 2);

    assert_eq!(
        apply_recovery_delta(&props, &records, &dpt, &alloc, Lsn::new(11)).unwrap(),
        RecoveryDeltaOutcome::Idempotent
    );
    assert_eq!(
        apply_recovery_delta(&props, &records, &dpt, &put, Lsn::new(11)).unwrap(),
        RecoveryDeltaOutcome::Idempotent
    );
    assert_eq!(
        dpt.snapshot(),
        before,
        "idempotent replay must not re-dirty"
    );

    let page = records
        .read_page_for_redo(TenantId::DEFAULT, PageId::new(7))
        .unwrap()
        .unwrap();
    let view = SlottedPageRef::open(page.as_ref()).unwrap();
    assert_eq!(view.page_lsn(), Lsn::new(11));
    assert_eq!(view.read_node(SlotId(0)).unwrap().unwrap().id, 99);
}

#[test]
fn data_before_page_alloc_is_corruption() {
    let error = apply_recovery_delta(
        &MemoryDeltaStore::default(),
        &MemoryDeltaStore::default(),
        &DirtyPageTable::new(),
        &put_node(8, 20, 1),
        Lsn::new(20),
    )
    .unwrap_err();
    assert!(matches!(error, ArcGraphError::WalCorruption { .. }));
    assert!(error.to_string().contains("no preceding PageAlloc"));
}

#[test]
fn torn_page_surviving_dwb_is_corruption_not_a_page_lsn_comparison() {
    let records = MemoryDeltaStore::default();
    records
        .pages
        .lock()
        .unwrap()
        .insert(PageId::new(9), Box::new([0xA5; PAGE_SIZE]));
    let error = apply_recovery_delta(
        &MemoryDeltaStore::default(),
        &records,
        &DirtyPageTable::new(),
        &put_node(9, 30, 1),
        Lsn::new(30),
    )
    .unwrap_err();
    assert!(matches!(error, ArcGraphError::WalCorruption { .. }));
    assert!(error.to_string().contains("survived DWB restore"));
}

#[test]
fn store_five_delta_is_wal_corruption_on_replay() {
    // Struct literal deliberately bypasses `DeltaOp::new`: this models a
    // hostile/corrupt frame reaching the replay boundary and pins the
    // Director ruling independently from producer validation.
    let op = DeltaOp {
        kind: DeltaOpKind::PutPropBlock,
        store_id: STORE_BLOB_OVERFLOW,
        tenant_id: TenantId::DEFAULT,
        page_no: 5,
        slot: 0,
        op_lsn: Lsn::new(40),
        payload: Bytes::from_static(b"forbidden-overflow-delta"),
    };
    let error = apply_recovery_delta(
        &MemoryDeltaStore::default(),
        &MemoryDeltaStore::default(),
        &DirtyPageTable::new(),
        &op,
        Lsn::new(40),
    )
    .unwrap_err();
    assert!(matches!(error, ArcGraphError::WalCorruption { .. }));
    assert!(error.to_string().contains("blob.overflow store_id 5"));
}

#[test]
fn dpt_identity_keeps_tenant_store_and_page() {
    let props = MemoryDeltaStore::default();
    let records = MemoryDeltaStore::default();
    let dpt = DirtyPageTable::new();
    apply_recovery_delta(&props, &records, &dpt, &page_alloc(12, 50), Lsn::new(50)).unwrap();
    assert_eq!(
        dpt.snapshot()[0].key,
        DirtyPageKey {
            tenant_id: TenantId::DEFAULT,
            store_id: STORE_RECORD,
            page_no: 12,
        }
    );
}

fn v9_bundle(page_no: u64, base_lsn: u64) -> Vec<u8> {
    let deltas = vec![
        page_alloc(page_no, base_lsn),
        put_node(page_no, base_lsn + 1, page_no),
    ];
    encode_commit_bundle_v9(
        Lsn::new(base_lsn + 1),
        TenantId::DEFAULT,
        &HashMap::new(),
        &[],
        &deltas,
        &[],
        &[],
        &[],
        &[],
        &[],
    )
    .unwrap()
}

fn write_v9_segment(dir: &std::path::Path, payloads: &[Vec<u8>]) {
    let mut bytes = SegmentHeader {
        format_version: BUNDLE_FORMAT_V9,
    }
    .encode()
    .to_vec();
    for (index, payload) in payloads.iter().enumerate() {
        WalRecord {
            record_type: WalRecordType::CommitBundle,
            txn_id: index as u64 + 1,
            lsn: Lsn::new(index as u64 + 1),
            timestamp_ms: 0,
            tenant_id: TenantId::DEFAULT,
            payload: payload.clone(),
        }
        .encode(&mut bytes)
        .unwrap();
    }
    std::fs::write(dir.join(segment_filename(0)), bytes).unwrap();
}

#[test]
fn executor_sorts_v9_by_range_base_and_tolerates_gap_and_duplicate() {
    let dir = tempdir().unwrap();
    let low = v9_bundle(2, 2);
    let high = v9_bundle(10, 10);
    // Disk order is deliberately high, low, duplicate-low. Recovery order
    // must be low, duplicate-low, high; the 4..9 LSN gap is legal.
    write_v9_segment(dir.path(), &[high, low.clone(), low]);

    let props = Arc::new(MemoryDeltaStore::default());
    let records = Arc::new(MemoryDeltaStore::default());
    let dpt = Arc::new(DirtyPageTable::new());
    let primary = Arc::new(arcgraph_storage::primary_index::PrimaryPageStore::new());
    let target = PageStoreTarget::primary_only(primary).with_delta_stores(
        props,
        Arc::clone(&records) as Arc<dyn DeltaPageStore>,
        Arc::clone(&dpt),
    );
    let txn = Arc::new(TxnManager::new());
    let mut executor = ReplayExecutor::new(
        ReplayConfig::with_wal_dir(dir.path()),
        Arc::clone(&txn),
        target,
    );
    let applied = executor
        .run(WalRecoveryReader::open(dir.path()).unwrap())
        .unwrap();
    assert_eq!(applied, Lsn::new(11));
    assert!(
        records
            .read_page_for_redo(TenantId::DEFAULT, PageId::new(2))
            .unwrap()
            .is_some()
    );
    assert!(
        records
            .read_page_for_redo(TenantId::DEFAULT, PageId::new(10))
            .unwrap()
            .is_some()
    );
    assert_eq!(dpt.len(), 2);
    assert!(executor.metrics().snapshot().bundles_skipped_idempotent >= 1);
    let record_page = records
        .read_page_for_redo(TenantId::DEFAULT, PageId::new(2))
        .unwrap()
        .unwrap();
    let record_bytes = SlottedPageRef::open(record_page.as_ref())
        .unwrap()
        .read_node(SlotId(0))
        .unwrap()
        .unwrap()
        .to_bytes();
    let reader = txn.begin(TenantId::DEFAULT);
    assert_eq!(
        reader
            .read(arcgraph_storage::crud::node_mvcc_key(NodeId::new(2)))
            .unwrap(),
        Bytes::copy_from_slice(&record_bytes),
        "recovered MVCC bytes must be reconstructed from PutRecord exactly"
    );
    reader.abort();
}

#[test]
fn incremental_checkpoint_replays_physical_rec_lsn_without_rewinding_logical_state() {
    let dir = tempdir().unwrap();
    let deltas = vec![page_alloc(33, 10), put_node(33, 11, 330)];
    let old_mvcc = HashMap::from([(42, Some(Bytes::from_static(b"old-wal-value")))]);
    let payload = encode_commit_bundle_v9(
        Lsn::new(11),
        TenantId::DEFAULT,
        &old_mvcc,
        &[],
        &deltas,
        &[],
        &[],
        &[],
        &[],
        &[],
    )
    .unwrap();
    write_v9_segment(dir.path(), &[payload]);

    let props = Arc::new(MemoryDeltaStore::default());
    let records = Arc::new(MemoryDeltaStore::default());
    let dpt = Arc::new(DirtyPageTable::new());
    let target = PageStoreTarget::primary_only(Arc::new(PrimaryPageStore::new()))
        .with_delta_stores(props, Arc::clone(&records) as Arc<dyn DeltaPageStore>, dpt);
    let txn = Arc::new(TxnManager::new());
    txn.apply_replay_mvcc_write(
        Lsn::new(20),
        TenantId::DEFAULT,
        42,
        Some(Bytes::from_static(b"checkpoint-value")),
    );
    txn.seed_after_replay(Lsn::new(20));

    let mut executor =
        ReplayExecutor::new(ReplayConfig::with_wal_dir(dir.path()), txn.clone(), target)
            .with_incremental_checkpoint(Lsn::new(20), Lsn::new(10));
    assert_eq!(
        executor
            .run(WalRecoveryReader::open(dir.path()).unwrap())
            .unwrap(),
        Lsn::new(20)
    );
    assert!(
        records
            .read_page_for_redo(TenantId::DEFAULT, PageId::new(33))
            .unwrap()
            .is_some(),
        "physical redo at/below checkpoint must start from recLSN"
    );
    let reader = txn.begin(TenantId::DEFAULT);
    assert_eq!(
        reader.read(42).unwrap(),
        Bytes::from_static(b"checkpoint-value")
    );
    let recovered = records
        .read_page_for_redo(TenantId::DEFAULT, PageId::new(33))
        .unwrap()
        .unwrap();
    let recovered_node = SlottedPageRef::open(recovered.as_ref())
        .unwrap()
        .read_node(SlotId(0))
        .unwrap()
        .unwrap();
    assert_eq!(
        reader
            .read(arcgraph_storage::crud::node_mvcc_key(NodeId::new(330)))
            .unwrap(),
        Bytes::copy_from_slice(&recovered_node.to_bytes()),
        "a pre-frontier physical redo must rebuild the MVCC row omitted from incremental metadata"
    );
    reader.abort();
}

#[test]
fn executor_rejects_overlapping_nonduplicate_v9_ranges() {
    let dir = tempdir().unwrap();
    // [3,4] and [4,5] overlap at sub-LSN 4 but are not duplicates.
    write_v9_segment(dir.path(), &[v9_bundle(3, 3), v9_bundle(4, 4)]);
    let props = Arc::new(MemoryDeltaStore::default());
    let records = Arc::new(MemoryDeltaStore::default());
    let dpt = Arc::new(DirtyPageTable::new());
    let primary = Arc::new(arcgraph_storage::primary_index::PrimaryPageStore::new());
    let target = PageStoreTarget::primary_only(primary).with_delta_stores(props, records, dpt);
    let mut executor = ReplayExecutor::new(
        ReplayConfig::with_wal_dir(dir.path()),
        Arc::new(TxnManager::new()),
        target,
    );
    let error = executor
        .run(WalRecoveryReader::open(dir.path()).unwrap())
        .unwrap_err();
    assert!(matches!(error, ArcGraphError::WalCorruption { .. }));
    assert!(error.to_string().contains("redo ranges overlap"));
}

#[derive(Default)]
struct RecordingAllocatorSeed {
    advances: Mutex<Vec<AllocatorAdvance>>,
}

impl AllocatorSeedHandle for RecordingAllocatorSeed {
    fn seed_from_advance(&self, advance: AllocatorAdvance) {
        self.advances.lock().unwrap().push(advance);
    }
}

#[test]
fn executor_reconstructs_retained_v8_logical_sections_from_v9() {
    let dir = tempdir().unwrap();
    let allocator_advances = vec![AllocatorAdvance {
        tenant: TenantId::DEFAULT,
        kind: AllocatorKind::Node,
        new_high_water: 44,
    }];
    let idempotency_bindings = vec![IdempotencyBindingEntry {
        op: IdempotencyBindingOp::Install,
        tenant: TenantId::DEFAULT,
        kind: 0,
        internal_id: 44,
        external_id: "node:44".to_owned(),
    }];
    let acl_grants = vec![
        AclGrantEntry {
            op: AclGrantOp::Apply,
            tenant: TenantId::DEFAULT,
            doc: NodeId::new(44),
            grants: BTreeSet::from(["principal:reader".to_owned()]),
        },
        AclGrantEntry {
            op: AclGrantOp::Revoke,
            tenant: TenantId::DEFAULT,
            doc: NodeId::new(44),
            grants: BTreeSet::new(),
        },
    ];
    let payload = encode_commit_bundle_v9(
        Lsn::new(70),
        TenantId::DEFAULT,
        &HashMap::new(),
        &[],
        &[],
        &[],
        &[],
        &allocator_advances,
        &idempotency_bindings,
        &acl_grants,
    )
    .unwrap();
    write_v9_segment(dir.path(), &[payload]);

    let seed = Arc::new(RecordingAllocatorSeed::default());
    let seed_handle: Arc<dyn AllocatorSeedHandle> = seed.clone();
    let idempotency = Arc::new(IdempotencyStore::new());
    let permissions = Arc::new(PermissionIndex::new());
    let target = PageStoreTarget::primary_only(Arc::new(PrimaryPageStore::new()))
        .with_allocator_seed(seed_handle)
        .with_idempotency_store(Arc::clone(&idempotency))
        .with_permission_index(Arc::clone(&permissions));
    let mut executor = ReplayExecutor::new(
        ReplayConfig::default_with_temp_spill(),
        Arc::new(TxnManager::new()),
        target,
    );
    let high = executor
        .run(WalRecoveryReader::open(dir.path()).unwrap())
        .unwrap();

    assert_eq!(high, Lsn::new(70));
    assert_eq!(*seed.advances.lock().unwrap(), allocator_advances);
    assert_eq!(
        idempotency
            .get(TenantId::DEFAULT, 0, "node:44")
            .unwrap()
            .internal_id,
        44
    );
    assert_eq!(
        permissions.doc_grant_count(),
        0,
        "v8 ACL append order must make the trailing revoke win"
    );
    let metrics = executor.metrics().snapshot();
    assert_eq!(metrics.allocator_advances_applied, 1);
    assert_eq!(metrics.idempotency_bindings_recovered, 1);
    assert_eq!(metrics.acl_grants_recovered, 2);
}
