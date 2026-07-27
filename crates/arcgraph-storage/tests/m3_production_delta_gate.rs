//! Production commit-path cutover gates for a migrated v9 generation.

use std::path::PathBuf;
use std::sync::Arc;
#[cfg(debug_assertions)]
use std::sync::Barrier;
use std::time::Duration;

use arcgraph_core::{DurabilityTier, LabelId, PageId, TenantDurabilityLookup, TenantId, TypeId};
use arcgraph_index::{PropertyValue, SecondaryIndex, SecondaryKey};
use arcgraph_storage::crud::{
    CrudStore, INLINE_U32A_PROPERTY_KEY, PropertyData, commit, create_node, create_rel,
    node_mvcc_key, read_node_with_store, read_rel_with_store, update_node, update_rel,
};
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::{PageSlot, PrimaryIndex, PrimaryKey, RecordKind};
use arcgraph_storage::records::SlotId;
use arcgraph_storage::redo::DirtyPageTable;
use arcgraph_storage::secondary_handle::SecondaryIndexHandle;
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::segment::{SegmentHeader, list_segments, segment_filename};
use arcgraph_storage::wal::{
    BUNDLE_FORMAT_V8, BUNDLE_FORMAT_V9, BundlePageKind, DeltaOpKind, STORE_PROPS, STORE_RECORD,
    WalConfig, WalRecord, WalRecordType, WalWriter, decode_commit_bundle_v9,
};
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

fn records(dir: &std::path::Path) -> Vec<WalRecord> {
    let mut records = Vec::new();
    for segment in list_segments(dir).unwrap() {
        let bytes = std::fs::read(dir.join(segment_filename(segment))).unwrap();
        let header = SegmentHeader::decode(&bytes[..SegmentHeader::SIZE]).unwrap();
        assert_eq!(header.format_version, BUNDLE_FORMAT_V9);
        let mut cursor = SegmentHeader::SIZE;
        while cursor < bytes.len() {
            let (record, consumed) = WalRecord::decode(&bytes[cursor..]).unwrap();
            records.push(record);
            cursor += consumed;
        }
    }
    records
}

#[derive(Debug)]
struct Periodic;

impl TenantDurabilityLookup for Periodic {
    fn durability_tier(&self, _tenant: TenantId) -> DurabilityTier {
        DurabilityTier::Periodic { rpo_ms: 60_000 }
    }
}

#[test]
fn migrated_generation_production_commit_emits_record_and_prop_deltas() {
    let dir = tempdir().unwrap();
    std::fs::write(
        dir.path().join(segment_filename(0)),
        SegmentHeader {
            format_version: BUNDLE_FORMAT_V9,
        }
        .encode(),
    )
    .unwrap();

    // Bootstrap the empty primary index before attaching the migrated WAL;
    // migration normally restores this state from its final checkpoint.
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
    store.attach_wal(wal.clone());

    let mut node_ids = Vec::new();
    for fill in [0xA5, 0xB6] {
        let mut tx = manager.begin(TenantId::DEFAULT);
        let node_id = create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(7),
            &PropertyData::Blob(vec![fill; 96]),
        )
        .unwrap();
        commit(tx, &store).unwrap();
        node_ids.push(node_id);
    }
    writer.shutdown().unwrap();

    // A v9 builder must not mutate the record page before WAL durability.
    // Disconnecting the writer forces the Phase-2 failure after the update
    // intent was built; direct page bytes remain byte-identical without a
    // record-page preimage rollback.
    let seed_slot = primary
        .lookup(PrimaryKey::new(
            TenantId::DEFAULT,
            RecordKind::Node,
            node_ids[0].raw(),
        ))
        .unwrap()
        .unwrap();
    let record_store = store.records().unwrap();
    let before = {
        let latch = record_store.latch(seed_slot.page).unwrap();
        let guard = latch.read();
        guard.clone()
    };
    let mut failed_update = manager.begin(TenantId::DEFAULT);
    update_node(
        &store,
        &mut failed_update,
        node_ids[0],
        &PropertyData::InlineU32Pair(999, 999),
    )
    .unwrap();
    assert!(commit(failed_update, &store).is_err());
    let after = {
        let latch = record_store.latch(seed_slot.page).unwrap();
        let guard = latch.read();
        guard.clone()
    };
    assert_eq!(before.as_ref(), after.as_ref());

    // A failed create releases its slot reservation. After writer restart the
    // next commit reclaims slot 2 rather than leaking it permanently.
    let mut failed_create = manager.begin(TenantId::DEFAULT);
    let failed_id = create_node(
        &store,
        &mut failed_create,
        TenantId::DEFAULT,
        LabelId::new(7),
        &PropertyData::Empty,
    )
    .unwrap();
    assert!(commit(failed_create, &store).is_err());
    assert!(
        primary
            .lookup(PrimaryKey::new(
                TenantId::DEFAULT,
                RecordKind::Node,
                failed_id.raw(),
            ))
            .unwrap()
            .is_none()
    );

    let writer2 =
        WalWriter::spawn_from(config(dir.path().to_path_buf()), manager.current_lsn()).unwrap();
    let wal2 = writer2.handle();
    manager.attach_wal(wal2.clone());
    primary.attach_wal(wal2.clone());
    store.attach_wal(wal2);
    let mut retry = manager.begin(TenantId::DEFAULT);
    let retry_id = create_node(
        &store,
        &mut retry,
        TenantId::DEFAULT,
        LabelId::new(7),
        &PropertyData::Empty,
    )
    .unwrap();
    commit(retry, &store).unwrap();
    let retry_slot = primary
        .lookup(PrimaryKey::new(
            TenantId::DEFAULT,
            RecordKind::Node,
            retry_id.raw(),
        ))
        .unwrap()
        .unwrap();
    assert_eq!(
        retry_slot.slot.raw(),
        2,
        "failed reservation must be reused"
    );
    writer2.shutdown().unwrap();

    let commits: Vec<_> = records(dir.path())
        .into_iter()
        .filter(|record| record.record_type == WalRecordType::CommitBundle)
        .collect();
    assert_eq!(commits.len(), 3);
    let bundles: Vec<_> = commits
        .iter()
        .map(|record| decode_commit_bundle_v9(&record.payload, TenantId::DEFAULT).unwrap())
        .collect();
    for (record, bundle) in commits.iter().zip(&bundles) {
        assert_eq!(record.lsn, bundle.commit_lsn);
    }
    assert_eq!(wal.last_durable_lsn(), bundles[1].commit_lsn);
    for bundle in &bundles[..2] {
        assert!(
            bundle
                .deltas
                .iter()
                .any(|op| { op.kind == DeltaOpKind::PutRecord && op.store_id == STORE_RECORD })
        );
        assert!(
            bundle
                .deltas
                .iter()
                .any(|op| { op.kind == DeltaOpKind::PutPropBlock && op.store_id == STORE_PROPS })
        );
        assert!(
            bundle
                .deltas
                .windows(2)
                .all(|pair| pair[0].op_lsn.raw() + 1 == pair[1].op_lsn.raw())
        );
        assert!(bundle.staged_pages.iter().all(|page| matches!(
            page.kind,
            BundlePageKind::PrimaryIndex | BundlePageKind::SecondaryIndex | BundlePageKind::Blob
        )));
    }
    for (node_id, bundle) in node_ids.iter().zip(&bundles[..2]) {
        assert!(
            !bundle.mvcc_writes.contains_key(&node_mvcc_key(*node_id)),
            "PutRecord must be the single durable copy of record/MVCC bytes"
        );
        let delta = bundle
            .deltas
            .iter()
            .find(|op| {
                op.kind == DeltaOpKind::PutRecord
                    && u64::from_le_bytes(op.payload[..8].try_into().unwrap()) == node_id.raw()
            })
            .unwrap();
        let reader = manager.begin(TenantId::DEFAULT);
        assert_eq!(
            reader.read(node_mvcc_key(*node_id)).unwrap(),
            delta.payload,
            "live MVCC version must be byte-identical to PutRecord"
        );
        reader.abort();
    }
    assert_eq!(
        bundles[0]
            .deltas
            .iter()
            .filter(|op| op.kind == DeltaOpKind::PageAlloc)
            .count(),
        2,
        "first commit allocates one record and one props page"
    );
    assert_eq!(
        bundles[1]
            .deltas
            .iter()
            .filter(|op| op.kind == DeltaOpKind::PageAlloc)
            .count(),
        0,
        "pooled pages must log only newly appended slots"
    );
    assert_eq!(
        bundles[0].redo_range().end().raw() + 1,
        bundles[1].redo_range().base().raw(),
        "production ranges tile the global redo clock"
    );
}

#[test]
fn batched_node_wal_amplification_is_at_most_280_bytes_per_node() {
    const NODES: usize = 1_000;
    let dir = tempdir().unwrap();
    std::fs::write(
        dir.path().join(segment_filename(0)),
        SegmentHeader {
            format_version: BUNDLE_FORMAT_V9,
        }
        .encode(),
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

    let mut tx = manager.begin(TenantId::DEFAULT);
    for index in 0..NODES {
        create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(7),
            &PropertyData::Blob(
                format!("id={index};name=user_{index};age={}", index % 90).into_bytes(),
            ),
        )
        .unwrap();
    }
    commit(tx, &store).unwrap();
    writer.shutdown().unwrap();

    let wal_bytes = std::fs::metadata(dir.path().join(segment_filename(0)))
        .unwrap()
        .len()
        - SegmentHeader::SIZE as u64;
    let bytes_per_node = wal_bytes as f64 / NODES as f64;
    eprintln!("m3 batch WAL: {wal_bytes} B / {NODES} = {bytes_per_node:.3} B/node");
    assert!(
        bytes_per_node <= 280.0,
        "headline M3 batch WAL ceiling exceeded: {bytes_per_node:.3} B/node"
    );

    let decoded: Vec<_> = records(dir.path())
        .into_iter()
        .filter(|record| record.record_type == WalRecordType::CommitBundle)
        .map(|record| decode_commit_bundle_v9(&record.payload, record.tenant_id).unwrap())
        .collect();
    let delta_count: usize = decoded.iter().map(|bundle| bundle.deltas.len()).sum();
    assert!(
        delta_count >= NODES,
        "the headline gate must observe physiological deltas, not merely a small total"
    );
    assert!(
        decoded
            .iter()
            .all(|bundle| bundle.staged_pages.iter().all(|page| {
                !matches!(
                    page.kind,
                    BundlePageKind::Record | BundlePageKind::PropSlotted
                )
            })),
        "v9 record/property effects regressed to retained full-page images"
    );

    // Identical production workload through the v8 producer. The absolute
    // 280-B ceiling alone is dishonest because this workload's v8 images also
    // fit below it; require a measured v9 advantage and explicit delta share.
    let v8_dir = tempdir().unwrap();
    std::fs::write(
        v8_dir.path().join(segment_filename(0)),
        SegmentHeader {
            format_version: BUNDLE_FORMAT_V8,
        }
        .encode(),
    )
    .unwrap();
    let v8_manager = Arc::new(TxnManager::new());
    let v8_allocator = Arc::new(PageAllocator::new());
    let v8_primary = Arc::new(
        PrimaryIndex::new(Arc::clone(&v8_manager), Arc::clone(&v8_allocator), None).unwrap(),
    );
    let mut v8_store = CrudStore::new_with_index(None, Arc::clone(&v8_primary), v8_allocator);
    let v8_writer = WalWriter::spawn_from(
        config(v8_dir.path().to_path_buf()),
        v8_manager.current_lsn(),
    )
    .unwrap();
    let v8_wal = v8_writer.handle();
    v8_manager.attach_wal(v8_wal.clone());
    v8_primary.attach_wal(v8_wal.clone());
    v8_store.attach_wal(v8_wal);
    let mut v8_tx = v8_manager.begin(TenantId::DEFAULT);
    for index in 0..NODES {
        create_node(
            &v8_store,
            &mut v8_tx,
            TenantId::DEFAULT,
            LabelId::new(7),
            &PropertyData::Blob(
                format!("id={index};name=user_{index};age={}", index % 90).into_bytes(),
            ),
        )
        .unwrap();
    }
    commit(v8_tx, &v8_store).unwrap();
    v8_writer.shutdown().unwrap();
    let v8_wal_bytes = std::fs::metadata(v8_dir.path().join(segment_filename(0)))
        .unwrap()
        .len()
        - SegmentHeader::SIZE as u64;
    assert!(
        wal_bytes * 100 <= v8_wal_bytes * 95,
        "v9 must beat the identical v8 workload by at least 5%: v9={wal_bytes}, v8={v8_wal_bytes}"
    );
}

#[test]
fn periodic_v9_create_and_update_read_from_mvcc_until_multi_page_apply() {
    let dir = tempdir().unwrap();
    std::fs::write(
        dir.path().join(segment_filename(0)),
        SegmentHeader {
            format_version: BUNDLE_FORMAT_V9,
        }
        .encode(),
    )
    .unwrap();
    let mut manager_inner = TxnManager::new();
    manager_inner.set_durability_lookup(Arc::new(Periodic));
    let manager = Arc::new(manager_inner);
    let allocator = Arc::new(PageAllocator::new());
    let primary =
        Arc::new(PrimaryIndex::new(Arc::clone(&manager), Arc::clone(&allocator), None).unwrap());
    let mut store = CrudStore::new_with_index(None, Arc::clone(&primary), allocator);
    let mut wal_config = config(dir.path().to_path_buf());
    wal_config.group_commit_window = Duration::from_secs(60);
    let writer = WalWriter::spawn_from(wal_config, manager.current_lsn()).unwrap();
    let wal = writer.handle();
    manager.attach_wal(wal.clone());
    primary.attach_wal(wal.clone());
    store.attach_wal(wal.clone());
    let dpt = Arc::new(DirtyPageTable::new());
    store.attach_m3_dirty_page_table(Arc::clone(&dpt));

    // One public CRUD commit allocates both a node page and a relationship
    // page. Neither page exists physically before the Periodic WAL record is
    // fsynced, but both committed MVCC versions are already visible.
    let mut tx = manager.begin(TenantId::DEFAULT);
    let node_id = create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(9),
        &PropertyData::InlineU32Pair(11, 12),
    )
    .unwrap();
    let rel_id = create_rel(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        node_id,
        node_id,
        TypeId::new(4),
        &PropertyData::InlineU32Pair(21, 22),
    )
    .unwrap();
    let create_lsn = commit(tx, &store).unwrap();
    let node_slot = primary
        .lookup(PrimaryKey::new(
            TenantId::DEFAULT,
            RecordKind::Node,
            node_id.raw(),
        ))
        .unwrap()
        .unwrap();
    let rel_slot = primary
        .lookup(PrimaryKey::new(
            TenantId::DEFAULT,
            RecordKind::Rel,
            rel_id.raw(),
        ))
        .unwrap()
        .unwrap();
    assert_ne!(
        node_slot.page, rel_slot.page,
        "the production commit must cover multiple record pages"
    );
    assert!(store.records().unwrap().latch(node_slot.page).is_err());
    assert!(store.records().unwrap().latch(rel_slot.page).is_err());
    assert!(wal.last_durable_lsn().raw() < create_lsn.raw());
    assert!(
        dpt.is_empty(),
        "un-durable reservations must not enter the DPT"
    );

    let reader = manager.begin(TenantId::DEFAULT);
    let node = read_node_with_store(&store, &reader, node_id)
        .expect("deferred create must not surface MissingPage")
        .expect("committed node create must be visible before fsync");
    assert_eq!((node.inline_u32a, node.inline_u32b), (11, 12));
    let rel = read_rel_with_store(&store, &reader, rel_id)
        .expect("deferred create must not surface MissingPage")
        .expect("committed rel create must be visible before fsync");
    assert_eq!((rel.inline_u32a, rel.inline_u32b), (21, 22));
    reader.abort();

    wal.flush().unwrap();
    assert_eq!(store.drain_deferred_v9_applies().unwrap(), 1);
    assert_eq!(dpt.len(), 2, "both durable record pages must enter the DPT");
    let reader = manager.begin(TenantId::DEFAULT);
    let node = read_node_with_store(&store, &reader, node_id)
        .unwrap()
        .unwrap();
    assert_eq!((node.inline_u32a, node.inline_u32b), (11, 12));
    let rel = read_rel_with_store(&store, &reader, rel_id)
        .unwrap()
        .unwrap();
    assert_eq!((rel.inline_u32a, rel.inline_u32b), (21, 22));
    reader.abort();

    // Update both records in one second multi-page commit. The physical pages
    // intentionally retain the old values until fsync; reads must not let
    // those snapshot-visible old records beat the newer MVCC versions.
    let mut tx = manager.begin(TenantId::DEFAULT);
    update_node(
        &store,
        &mut tx,
        node_id,
        &PropertyData::InlineU32Pair(111, 112),
    )
    .unwrap();
    update_rel(
        &store,
        &mut tx,
        rel_id,
        &PropertyData::InlineU32Pair(121, 122),
    )
    .unwrap();
    let update_lsn = commit(tx, &store).unwrap();
    assert!(wal.last_durable_lsn().raw() < update_lsn.raw());

    let reader = manager.begin(TenantId::DEFAULT);
    let node = read_node_with_store(&store, &reader, node_id)
        .unwrap()
        .unwrap();
    assert_eq!((node.inline_u32a, node.inline_u32b), (111, 112));
    let rel = read_rel_with_store(&store, &reader, rel_id)
        .unwrap()
        .unwrap();
    assert_eq!((rel.inline_u32a, rel.inline_u32b), (121, 122));
    reader.abort();

    let records = store.records().unwrap();
    let node_latch = records.latch(node_slot.page).unwrap();
    let node_guard = node_latch.read();
    let node_page = arcgraph_storage::records::SlottedPageRef::open(node_guard.as_ref()).unwrap();
    let stale_node = node_page.read_node(node_slot.slot).unwrap().unwrap();
    assert_eq!((stale_node.inline_u32a, stale_node.inline_u32b), (11, 12));
    drop(node_guard);
    let rel_latch = records.latch(rel_slot.page).unwrap();
    let rel_guard = rel_latch.read();
    let rel_page = arcgraph_storage::records::SlottedPageRef::open(rel_guard.as_ref()).unwrap();
    let stale_rel = rel_page.read_rel(rel_slot.slot).unwrap().unwrap();
    assert_eq!((stale_rel.inline_u32a, stale_rel.inline_u32b), (21, 22));
    drop(rel_guard);

    wal.flush().unwrap();
    assert_eq!(store.drain_deferred_v9_applies().unwrap(), 1);
    let reader = manager.begin(TenantId::DEFAULT);
    let node = read_node_with_store(&store, &reader, node_id)
        .unwrap()
        .unwrap();
    assert_eq!((node.inline_u32a, node.inline_u32b), (111, 112));
    let rel = read_rel_with_store(&store, &reader, rel_id)
        .unwrap()
        .unwrap();
    assert_eq!((rel.inline_u32a, rel.inline_u32b), (121, 122));
    reader.abort();
    writer.shutdown().unwrap();
}

#[test]
fn v9_exact_durability_proofs_drain_on_deferred_crud_and_standalone_index_paths() {
    let dir = tempdir().unwrap();
    std::fs::write(
        dir.path().join(segment_filename(0)),
        SegmentHeader {
            format_version: BUNDLE_FORMAT_V9,
        }
        .encode(),
    )
    .unwrap();
    let mut manager_inner = TxnManager::new();
    manager_inner.set_durability_lookup(Arc::new(Periodic));
    let manager = Arc::new(manager_inner);
    let allocator = Arc::new(PageAllocator::new());
    let primary =
        Arc::new(PrimaryIndex::new(Arc::clone(&manager), Arc::clone(&allocator), None).unwrap());
    let mut store = CrudStore::new_with_index(None, Arc::clone(&primary), allocator);
    let mut wal_config = config(dir.path().to_path_buf());
    wal_config.group_commit_window = Duration::from_secs(60);
    let writer = WalWriter::spawn_from(wal_config, manager.current_lsn()).unwrap();
    let wal = writer.handle();
    manager.attach_wal(wal.clone());
    primary.attach_wal(wal.clone());
    store.attach_wal(wal.clone());

    // The public CRUD path is Periodic, so its exact proof is retained until
    // the deferred physical apply drains after fsync.
    let mut tx = manager.begin(TenantId::DEFAULT);
    create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(17),
        &PropertyData::InlineU32Pair(1, 2),
    )
    .unwrap();
    let crud_lsn = commit(tx, &store).unwrap();
    wal.flush().unwrap();
    assert_eq!(store.drain_deferred_v9_applies().unwrap(), 1);
    let crud_proof_leaked = wal.take_exact_durable(crud_lsn);

    // A public standalone primary-index write reaches the second v9 commit
    // path. SYSTEM durability is deliberately Strict, but it uses the same
    // exact-proof accounting and must consume its proof before returning.
    primary
        .insert(
            PrimaryKey::new(TenantId::DEFAULT, RecordKind::Node, 1_000_000),
            PageSlot::new(PageId::new(9_999), SlotId(7)),
        )
        .unwrap();
    let standalone_lsn = manager.current_lsn();
    let standalone_proof_leaked = wal.take_exact_durable(standalone_lsn);

    assert_eq!(
        (crud_proof_leaked, standalone_proof_leaked),
        (false, false),
        "every successful consumer path must leave the exact-durability proof map empty"
    );
    writer.shutdown().unwrap();
}

fn periodic_secondary_stack() -> (
    tempfile::TempDir,
    WalWriter,
    Arc<TxnManager>,
    Arc<SecondaryIndex>,
    CrudStore,
    arcgraph_storage::wal::WalHandle,
) {
    let dir = tempdir().unwrap();
    std::fs::write(
        dir.path().join(segment_filename(0)),
        SegmentHeader {
            format_version: BUNDLE_FORMAT_V9,
        }
        .encode(),
    )
    .unwrap();
    let mut manager_inner = TxnManager::new();
    manager_inner.set_durability_lookup(Arc::new(Periodic));
    let manager = Arc::new(manager_inner);
    let allocator = Arc::new(PageAllocator::new());
    // Bootstrap both index roots before WAL attachment, matching migrated
    // recovery where the roots already exist at generation cutover.
    let primary =
        Arc::new(PrimaryIndex::new(Arc::clone(&manager), Arc::clone(&allocator), None).unwrap());
    let secondary =
        Arc::new(SecondaryIndex::new(Arc::clone(&manager), Arc::clone(&allocator), None).unwrap());
    let secondary_handle: Arc<dyn SecondaryIndexHandle> = Arc::clone(&secondary) as _;
    let mut store = CrudStore::new_with_indices(
        None,
        Arc::clone(&primary),
        Some(secondary_handle),
        allocator,
    );
    let mut wal_config = config(dir.path().to_path_buf());
    wal_config.group_commit_window = Duration::from_secs(60);
    let writer = WalWriter::spawn_from(wal_config, manager.current_lsn()).unwrap();
    let wal = writer.handle();
    manager.attach_wal(wal.clone());
    primary.attach_wal(wal.clone());
    store.attach_wal(wal.clone());
    (dir, writer, manager, secondary, store, wal)
}

fn secondary_key(label: LabelId, value: u32) -> SecondaryKey {
    SecondaryKey::new(
        TenantId::DEFAULT,
        label,
        INLINE_U32A_PROPERTY_KEY,
        PropertyValue::U32(value),
    )
}

fn update_inline_a(manager: &TxnManager, store: &CrudStore, id: arcgraph_core::NodeId, value: u32) {
    let mut tx = manager.begin(TenantId::DEFAULT);
    update_node(store, &mut tx, id, &PropertyData::InlineU32Pair(value, 0)).unwrap();
    commit(tx, store).unwrap();
}

#[cfg(debug_assertions)]
fn advance_secondary_horizon(manager: &TxnManager, store: &CrudStore, label: u32) {
    let mut tick = manager.begin(TenantId::DEFAULT);
    create_node(
        store,
        &mut tick,
        TenantId::DEFAULT,
        LabelId::new(label),
        &PropertyData::InlineU32Pair(909, 0),
    )
    .unwrap();
    commit(tick, store).unwrap();
}

#[cfg(debug_assertions)]
fn pause_next_deferred_apply(
    store: &Arc<CrudStore>,
) -> (
    Arc<Barrier>,
    std::thread::JoinHandle<arcgraph_core::Result<usize>>,
) {
    let entered = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    store.__test_gate_next_deferred_v9_apply(Arc::clone(&entered), Arc::clone(&release));
    let drain_store = Arc::clone(store);
    let drainer = std::thread::spawn(move || drain_store.drain_deferred_v9_applies());
    entered.wait();
    (release, drainer)
}

#[test]
fn periodic_deferred_a_b_a_keeps_live_secondary_entry_after_drain() {
    const A: u32 = 101;
    const B: u32 = 202;
    let (_dir, writer, manager, secondary, store, wal) = periodic_secondary_stack();
    let label = LabelId::new(23);

    let mut tx = manager.begin(TenantId::DEFAULT);
    let id = create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        label,
        &PropertyData::InlineU32Pair(A, 0),
    )
    .unwrap();
    commit(tx, &store).unwrap();
    wal.flush().unwrap();
    assert_eq!(store.drain_deferred_v9_applies().unwrap(), 1);

    update_inline_a(&manager, &store, id, B);
    assert!(store.deferred_v9_boundary().is_some());

    // Advance the secondary-removal horizon with an unrelated public CRUD
    // commit while B's record slot remains physically deferred. This removes
    // the old A entry before the final A insert and keeps this gate scoped to
    // pending-slot pre-image selection.
    let mut tick = manager.begin(TenantId::DEFAULT);
    create_node(
        &store,
        &mut tick,
        TenantId::DEFAULT,
        LabelId::new(24),
        &PropertyData::InlineU32Pair(909, 0),
    )
    .unwrap();
    commit(tick, &store).unwrap();
    update_inline_a(&manager, &store, id, A);
    wal.flush().unwrap();
    assert_eq!(store.drain_deferred_v9_applies().unwrap(), 3);

    let live_hits = secondary.lookup(secondary_key(label, A)).unwrap();
    assert!(
        live_hits.contains(&id),
        "A→B→A must retain the live A secondary entry after deferred apply"
    );
    writer.shutdown().unwrap();
}

#[test]
fn periodic_deferred_a_b_c_removes_superseded_secondary_entry_after_drain() {
    const A: u32 = 303;
    const B: u32 = 404;
    const C: u32 = 505;
    let (_dir, writer, manager, secondary, store, wal) = periodic_secondary_stack();
    let label = LabelId::new(29);

    let mut tx = manager.begin(TenantId::DEFAULT);
    let id = create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        label,
        &PropertyData::InlineU32Pair(A, 0),
    )
    .unwrap();
    commit(tx, &store).unwrap();
    wal.flush().unwrap();
    assert_eq!(store.drain_deferred_v9_applies().unwrap(), 1);

    update_inline_a(&manager, &store, id, B);
    assert!(store.deferred_v9_boundary().is_some());
    update_inline_a(&manager, &store, id, C);
    wal.flush().unwrap();
    assert_eq!(store.drain_deferred_v9_applies().unwrap(), 2);

    // Advance the snapshot horizon once more so C's deferred removal of B is
    // eligible, while keeping the public Periodic CRUD path throughout.
    update_inline_a(&manager, &store, id, C);
    wal.flush().unwrap();
    assert_eq!(store.drain_deferred_v9_applies().unwrap(), 1);

    let live_hits = secondary.lookup(secondary_key(label, C)).unwrap();
    assert!(live_hits.contains(&id), "the live C entry must be present");
    let superseded_hits = secondary.lookup(secondary_key(label, B)).unwrap();
    assert!(
        !superseded_hits.contains(&id),
        "A→B→C must remove the superseded B secondary entry after deferred apply"
    );
    writer.shutdown().unwrap();
}

#[cfg(debug_assertions)]
#[test]
fn deterministic_barrier_deferred_a_b_a_keeps_live_and_removes_stale_secondary_entry() {
    const A: u32 = 611;
    const B: u32 = 622;
    let (_dir, writer, manager, secondary, store, wal) = periodic_secondary_stack();
    let store = Arc::new(store);
    let label = LabelId::new(61);

    let mut tx = manager.begin(TenantId::DEFAULT);
    let id = create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        label,
        &PropertyData::InlineU32Pair(A, 0),
    )
    .unwrap();
    commit(tx, &store).unwrap();
    wal.flush().unwrap();
    assert_eq!(store.drain_deferred_v9_applies().unwrap(), 1);

    update_inline_a(&manager, &store, id, B);
    assert!(store.deferred_v9_boundary().is_some());
    advance_secondary_horizon(&manager, &store, 62);
    assert!(
        !secondary
            .lookup(secondary_key(label, A))
            .unwrap()
            .contains(&id),
        "the horizon tick must remove old A before the final A insert"
    );

    wal.flush().unwrap();
    let (release, drainer) = pause_next_deferred_apply(&store);
    update_inline_a(&manager, &store, id, A);
    release.wait();
    drainer.join().unwrap().unwrap();

    // Clear the final commit's deferred B removal without touching the
    // target node, then materialize every remaining Periodic record delta.
    advance_secondary_horizon(&manager, &store, 63);
    wal.flush().unwrap();
    store.drain_deferred_v9_applies().unwrap();

    assert!(
        secondary
            .lookup(secondary_key(label, A))
            .unwrap()
            .contains(&id),
        "A→B→A must retain the live A entry across the forced deferred-apply window"
    );
    assert!(
        !secondary
            .lookup(secondary_key(label, B))
            .unwrap()
            .contains(&id),
        "A→B→A must remove the stale B entry after the horizon clears"
    );
    writer.shutdown().unwrap();
}

#[cfg(debug_assertions)]
#[test]
fn deterministic_barrier_deferred_a_b_c_keeps_live_and_removes_stale_secondary_entry() {
    const A: u32 = 711;
    const B: u32 = 722;
    const C: u32 = 733;
    let (_dir, writer, manager, secondary, store, wal) = periodic_secondary_stack();
    let store = Arc::new(store);
    let label = LabelId::new(71);

    let mut tx = manager.begin(TenantId::DEFAULT);
    let id = create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        label,
        &PropertyData::InlineU32Pair(A, 0),
    )
    .unwrap();
    commit(tx, &store).unwrap();
    wal.flush().unwrap();
    assert_eq!(store.drain_deferred_v9_applies().unwrap(), 1);

    update_inline_a(&manager, &store, id, B);
    assert!(store.deferred_v9_boundary().is_some());
    advance_secondary_horizon(&manager, &store, 72);

    wal.flush().unwrap();
    let (release, drainer) = pause_next_deferred_apply(&store);
    update_inline_a(&manager, &store, id, C);
    release.wait();
    drainer.join().unwrap().unwrap();

    advance_secondary_horizon(&manager, &store, 73);
    wal.flush().unwrap();
    store.drain_deferred_v9_applies().unwrap();

    assert!(
        secondary
            .lookup(secondary_key(label, C))
            .unwrap()
            .contains(&id),
        "A→B→C must retain the live C entry across the forced deferred-apply window"
    );
    assert!(
        !secondary
            .lookup(secondary_key(label, B))
            .unwrap()
            .contains(&id),
        "A→B→C must remove the stale B entry after the horizon clears"
    );
    writer.shutdown().unwrap();
}
