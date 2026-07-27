//! M3 phase-4 gate: v9 incremental metadata is established only after the
//! DWB/home pass and structurally excludes the legacy owner 1/3/4 walks.

use std::sync::Arc;
use std::sync::Barrier;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Duration;

use arcgraph_core::record::{NodeRecord, PAGE_SIZE, PageHeader, PageType};
use arcgraph_core::{LabelId, Lsn, NodeId, PageId, TenantId};
use arcgraph_storage::blob::BlobStore;
#[cfg(debug_assertions)]
use arcgraph_storage::blob::{BlobBoundConfig, BlobSpill};
use arcgraph_storage::buffer::BufferPool;
#[cfg(debug_assertions)]
use arcgraph_storage::checkpoint::CheckpointError;
use arcgraph_storage::checkpoint::{
    CheckpointSnapshot, DoublewriteArea, incremental_checkpoint, incremental_metadata_path,
    read_incremental_metadata, read_latest_sidecar,
};
use arcgraph_storage::crud::DeferredV9Boundary;
use arcgraph_storage::crud::{CrudStore, crud_allocator_seed_handle, node_mvcc_key};
use arcgraph_storage::idempotency::IdempotencyStore;
use arcgraph_storage::intern::InternTable;
use arcgraph_storage::io::{InMemoryPageIo, PageIo};
use arcgraph_storage::m3_migration::{
    M3_PROPS_STORE_FILE, M3_RECORD_STORE_FILE, load_v9_physical_base, m3_record_store_path,
};
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::page_store::{
    BufferedRecordPageStore, PerTenantBufferPool, PerTenantBufferPoolConfig, RecordPageBackend,
    TenantFilePageIo, TenantPageIo,
};
use arcgraph_storage::permissions::PermissionIndex;
use arcgraph_storage::primary_index::PrimaryPageStore;
use arcgraph_storage::record_store::RecordPageStore;
use arcgraph_storage::records::{SlotId, SlottedPage, SlottedPageRef};
use arcgraph_storage::redo::{DirtyPageKey, DirtyPageTable};
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{
    BUNDLE_FORMAT_V9, DeltaOp, DeltaOpKind, PageStoreTarget, ReplayConfig, ReplayExecutor,
    STORE_PROPS, STORE_RECORD, SegmentHeader, WalRecord, WalRecordType, WalRecoveryReader,
    encode_commit_bundle_v9, reclaim_segments_below, segment_filename,
};
use arcgraph_storage::{DOUBLEWRITE_FILE, WriteBehindCheckpointer};
use bytes::Bytes;
use tempfile::tempdir;

fn buffered_store() -> Arc<BufferedRecordPageStore> {
    let io: Arc<dyn PageIo> = Arc::new(InMemoryPageIo::new());
    let pools = Arc::new(PerTenantBufferPool::with_config(
        io,
        PerTenantBufferPoolConfig {
            frames_per_tenant: 16,
            write_fraction: 0.0,
        },
    ));
    Arc::new(BufferedRecordPageStore::with_cache_cap(pools, 32))
}

fn tenant_buffered_store(path: &std::path::Path) -> Arc<BufferedRecordPageStore> {
    let io: Arc<dyn TenantPageIo> = Arc::new(TenantFilePageIo::new(path, M3_RECORD_STORE_FILE));
    let pools = Arc::new(PerTenantBufferPool::with_tenant_io(
        io,
        PerTenantBufferPoolConfig {
            frames_per_tenant: 16,
            write_fraction: 0.0,
        },
    ));
    Arc::new(BufferedRecordPageStore::with_cache_cap(pools, 32))
}

fn install_record_page(store: &BufferedRecordPageStore, page_id: PageId) {
    store
        .install_fresh(page_id, PageType::Node, TenantId::DEFAULT)
        .unwrap();
    let pinned = store.latch_pinned(page_id).unwrap();
    let mut guard = pinned.latch().write();
    SlottedPage::open(guard.as_mut())
        .unwrap()
        .put_node_at(
            SlotId(0),
            &NodeRecord::new(NodeId::new(7), LabelId::new(1), Lsn::new(7)),
        )
        .unwrap();
}

fn install_prop_page(store: &BufferedRecordPageStore, page_id: PageId) {
    let mut bytes = [0u8; PAGE_SIZE];
    let mut page = SlottedPage::init(
        &mut bytes,
        PageHeader::new(page_id, PageType::PropSlotted, TenantId::DEFAULT),
    )
    .unwrap();
    page.put_bag_at(SlotId(0), b"p").unwrap();
    RecordPageBackend::install_or_replace(store, page_id, Box::new(bytes)).unwrap();
}

fn write_v9_bundle_segment(
    data_dir: &std::path::Path,
    segment_no: u64,
    txn_id: u64,
    payload: Vec<u8>,
) {
    let mut segment = SegmentHeader {
        format_version: BUNDLE_FORMAT_V9,
    }
    .encode()
    .to_vec();
    WalRecord {
        record_type: WalRecordType::CommitBundle,
        txn_id,
        lsn: Lsn::new(txn_id),
        timestamp_ms: 0,
        tenant_id: TenantId::DEFAULT,
        payload,
    }
    .encode(&mut segment)
    .unwrap();
    std::fs::write(data_dir.join(segment_filename(segment_no)), segment).unwrap();
}

struct Owners {
    txn: Arc<TxnManager>,
    primary: Arc<PrimaryPageStore>,
    record: Arc<RecordPageStore>,
    blob: Arc<BlobStore>,
    allocator: Arc<PageAllocator>,
    crud: Arc<CrudStore>,
    intern: Arc<InternTable>,
    idempotency: Arc<IdempotencyStore>,
    permissions: Arc<PermissionIndex>,
}

impl Owners {
    fn new() -> Self {
        let allocator = Arc::new(PageAllocator::new());
        let record = Arc::new(RecordPageStore::new());
        let blob = Arc::new(BlobStore::new());
        let crud = Arc::new(CrudStore::new_with_existing_page_stores(
            None,
            None,
            Arc::clone(&allocator),
            Arc::clone(&record),
            Arc::clone(&blob),
        ));
        Self {
            txn: Arc::new(TxnManager::new()),
            primary: Arc::new(PrimaryPageStore::new()),
            record,
            blob,
            allocator,
            crud,
            intern: Arc::new(InternTable::new()),
            idempotency: Arc::new(IdempotencyStore::new()),
            permissions: Arc::new(PermissionIndex::new()),
        }
    }

    #[cfg(debug_assertions)]
    fn with_blob(blob: Arc<BlobStore>) -> Self {
        let allocator = Arc::new(PageAllocator::new());
        let record = Arc::new(RecordPageStore::new());
        let crud = Arc::new(CrudStore::new_with_existing_page_stores(
            None,
            None,
            Arc::clone(&allocator),
            Arc::clone(&record),
            Arc::clone(&blob),
        ));
        Self {
            txn: Arc::new(TxnManager::new()),
            primary: Arc::new(PrimaryPageStore::new()),
            record,
            blob,
            allocator,
            crud,
            intern: Arc::new(InternTable::new()),
            idempotency: Arc::new(IdempotencyStore::new()),
            permissions: Arc::new(PermissionIndex::new()),
        }
    }
}

#[cfg(debug_assertions)]
#[test]
fn spill_backfill_read_does_not_hold_checkpoint_freeze() {
    let dir = tempdir().unwrap();
    let spill = Arc::new(BlobSpill::open(dir.path()).unwrap());
    let blob = Arc::new(BlobStore::with_bound(
        Arc::clone(&spill),
        BlobBoundConfig::from_cap_bytes((PAGE_SIZE * 2) as u64),
    ));
    for i in 0..6u8 {
        blob.put(TenantId::DEFAULT, &[i; 32]).unwrap();
    }
    blob.for_each_resident_overflow_page(|_, _, _| Ok::<(), ()>(()))
        .unwrap();
    blob.force_drain_for_test().unwrap();
    assert!(
        blob.evicted_count() > 0,
        "premise: checkpoint must backfill at least one spill image"
    );

    let owners = Arc::new(Owners::with_blob(blob));
    owners.txn.seed_after_replay(Lsn::new(10));
    let checkpointer = WriteBehindCheckpointer::new(
        Arc::new(DirtyPageTable::new()),
        buffered_store(),
        buffered_store(),
    )
    .with_doublewrite_area(Arc::new(DoublewriteArea::new(dir.path())));
    let entered = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    spill.__test_gate_next_read(Arc::clone(&entered), Arc::clone(&release));

    let producer_owners = Arc::clone(&owners);
    let data_dir = dir.path().to_path_buf();
    let producer = std::thread::spawn(move || {
        let seed = crud_allocator_seed_handle(
            Arc::clone(&producer_owners.crud),
            Arc::clone(&producer_owners.allocator),
        );
        let snapshot = CheckpointSnapshot {
            txn: &producer_owners.txn,
            primary_pages: &producer_owners.primary,
            record_pages: &producer_owners.record,
            blob: &producer_owners.blob,
            allocator_seed: seed.as_ref(),
            intern: &producer_owners.intern,
            idempotency: &producer_owners.idempotency,
            permissions: &producer_owners.permissions,
            permissions_tenant: TenantId::DEFAULT,
        };
        incremental_checkpoint(
            &data_dir,
            &BufferPool::new(8, Arc::new(InMemoryPageIo::new())),
            &snapshot,
            &checkpointer,
            || (Vec::new(), None),
            Ok,
        )
    });

    entered.wait();
    let reader_txn = Arc::clone(&owners.txn);
    let (acquired_tx, acquired_rx) = mpsc::channel();
    let reader = std::thread::spawn(move || {
        let _commit = reader_txn.__test_commit_read_guard();
        acquired_tx.send(()).unwrap();
    });
    let acquired_outside_freeze = acquired_rx.recv_timeout(Duration::from_secs(1)).is_ok();
    release.wait();
    producer.join().unwrap().unwrap();
    reader.join().unwrap();
    assert!(
        acquired_outside_freeze,
        "a synchronous spill read held checkpoint_freeze and stopped foreground commits"
    );
}

#[cfg(debug_assertions)]
#[test]
fn large_owner_five_scan_is_constant_memory_outside_freeze_and_reclaims_wal() {
    const OVERFLOW_PAGES: u64 = 2_048;
    let dir = tempdir().unwrap();
    let spill = Arc::new(BlobSpill::open(dir.path()).unwrap());
    let blob = Arc::new(BlobStore::with_bound(
        Arc::clone(&spill),
        BlobBoundConfig::from_cap_bytes((PAGE_SIZE * 2) as u64),
    ));
    for page in 0..OVERFLOW_PAGES {
        blob.put(TenantId::DEFAULT, &page.to_le_bytes()).unwrap();
    }
    blob.for_each_resident_overflow_page(|_, _, _| Ok::<(), ()>(()))
        .unwrap();
    blob.force_drain_for_test().unwrap();
    assert!(
        blob.evicted_count() >= OVERFLOW_PAGES - 2,
        "premise: the owner must be overwhelmingly spill-resident"
    );

    // One closed, decodable v9 segment plus one active segment. Successful
    // establishment must permit the real reclaimer to delete the closed one.
    let empty_bundle = encode_commit_bundle_v9(
        Lsn::new(1),
        TenantId::DEFAULT,
        &std::collections::HashMap::new(),
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
    )
    .unwrap();
    let mut closed = SegmentHeader {
        format_version: BUNDLE_FORMAT_V9,
    }
    .encode()
    .to_vec();
    WalRecord {
        record_type: WalRecordType::CommitBundle,
        txn_id: 1,
        lsn: Lsn::new(1),
        timestamp_ms: 0,
        tenant_id: TenantId::DEFAULT,
        payload: empty_bundle,
    }
    .encode(&mut closed)
    .unwrap();
    std::fs::write(dir.path().join(segment_filename(0)), closed).unwrap();
    std::fs::write(
        dir.path().join(segment_filename(1)),
        SegmentHeader {
            format_version: BUNDLE_FORMAT_V9,
        }
        .encode(),
    )
    .unwrap();

    let owners = Arc::new(Owners::with_blob(blob));
    owners.txn.seed_after_replay(Lsn::new(10));
    let checkpointer = WriteBehindCheckpointer::new(
        Arc::new(DirtyPageTable::new()),
        buffered_store(),
        buffered_store(),
    )
    .with_doublewrite_area(Arc::new(DoublewriteArea::new(dir.path())));
    let entered = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    spill.__test_gate_next_capture_scan(Arc::clone(&entered), Arc::clone(&release));

    let producer_owners = Arc::clone(&owners);
    let data_dir = dir.path().to_path_buf();
    let producer = std::thread::spawn(move || {
        let seed = crud_allocator_seed_handle(
            Arc::clone(&producer_owners.crud),
            Arc::clone(&producer_owners.allocator),
        );
        let snapshot = CheckpointSnapshot {
            txn: &producer_owners.txn,
            primary_pages: &producer_owners.primary,
            record_pages: &producer_owners.record,
            blob: &producer_owners.blob,
            allocator_seed: seed.as_ref(),
            intern: &producer_owners.intern,
            idempotency: &producer_owners.idempotency,
            permissions: &producer_owners.permissions,
            permissions_tenant: TenantId::DEFAULT,
        };
        incremental_checkpoint(
            &data_dir,
            &BufferPool::new(8, Arc::new(InMemoryPageIo::new())),
            &snapshot,
            &checkpointer,
            || (Vec::new(), None),
            Ok,
        )
    });

    entered.wait();
    let reader_txn = Arc::clone(&owners.txn);
    let (acquired_tx, acquired_rx) = mpsc::channel();
    let reader = std::thread::spawn(move || {
        let _commit = reader_txn.__test_commit_read_guard();
        acquired_tx.send(()).unwrap();
    });
    let acquired_outside_freeze = acquired_rx.recv_timeout(Duration::from_secs(1)).is_ok();
    release.wait();
    reader.join().unwrap();
    let report = producer.join().unwrap().unwrap();
    assert!(
        acquired_outside_freeze,
        "the O(N) owner-5 cursor scan held the global commit freeze"
    );
    assert_eq!(report.metadata.counts.blob_pages, OVERFLOW_PAGES);
    assert!(
        report.metadata.overflow_peak_resident <= PAGE_SIZE + 64,
        "owner-5 caller-owned peak {} exceeded one page plus fixed token",
        report.metadata.overflow_peak_resident
    );
    let reclaimed = reclaim_segments_below(dir.path(), report.checkpoint_lsn).unwrap();
    assert_eq!(reclaimed.deleted_segments, vec![0]);
    assert!(!dir.path().join(segment_filename(0)).exists());
    assert!(dir.path().join(segment_filename(1)).exists());
}

#[cfg(debug_assertions)]
#[test]
fn owner_five_resident_to_spill_epoch_handoff_captures_once_and_reclaims_wal() {
    const OVERFLOW_PAGES: u64 = 3;
    let dir = tempdir().unwrap();
    let spill = Arc::new(BlobSpill::open(dir.path()).unwrap());
    let blob = Arc::new(BlobStore::with_bound(
        Arc::clone(&spill),
        BlobBoundConfig::from_cap_bytes((PAGE_SIZE * 4) as u64),
    ));
    for page in 0..OVERFLOW_PAGES {
        blob.put(TenantId::DEFAULT, &page.to_le_bytes()).unwrap();
    }
    assert_eq!(blob.logical_page_count(), OVERFLOW_PAGES as usize);
    assert_eq!(blob.resident_page_count(), OVERFLOW_PAGES as usize);
    blob.for_each_resident_overflow_page(|_, _, _| Ok::<(), ()>(()))
        .unwrap();

    let empty_bundle = encode_commit_bundle_v9(
        Lsn::new(1),
        TenantId::DEFAULT,
        &std::collections::HashMap::new(),
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
    )
    .unwrap();
    write_v9_bundle_segment(dir.path(), 0, 1, empty_bundle);
    std::fs::write(
        dir.path().join(segment_filename(1)),
        SegmentHeader {
            format_version: BUNDLE_FORMAT_V9,
        }
        .encode(),
    )
    .unwrap();

    let evict_entered = Arc::new(Barrier::new(2));
    let evict_release = Arc::new(Barrier::new(2));
    spill.__test_gate_next_evict_epoch_sample(
        Arc::clone(&evict_entered),
        Arc::clone(&evict_release),
    );
    let evict_blob = Arc::clone(&blob);
    let evictor = std::thread::spawn(move || evict_blob.force_drain_for_test());
    evict_entered.wait();

    let capture_entered = Arc::new(Barrier::new(2));
    let capture_release = Arc::new(Barrier::new(2));
    spill.__test_gate_after_next_resident_capture(
        Arc::clone(&capture_entered),
        Arc::clone(&capture_release),
    );
    let owners = Arc::new(Owners::with_blob(Arc::clone(&blob)));
    owners.txn.seed_after_replay(Lsn::new(10));
    let checkpointer = WriteBehindCheckpointer::new(
        Arc::new(DirtyPageTable::new()),
        buffered_store(),
        buffered_store(),
    )
    .with_doublewrite_area(Arc::new(DoublewriteArea::new(dir.path())));
    let checkpoint_owners = Arc::clone(&owners);
    let data_dir = dir.path().to_path_buf();
    let checkpoint = std::thread::spawn(move || {
        let seed = crud_allocator_seed_handle(
            Arc::clone(&checkpoint_owners.crud),
            Arc::clone(&checkpoint_owners.allocator),
        );
        let snapshot = CheckpointSnapshot {
            txn: &checkpoint_owners.txn,
            primary_pages: &checkpoint_owners.primary,
            record_pages: &checkpoint_owners.record,
            blob: &checkpoint_owners.blob,
            allocator_seed: seed.as_ref(),
            intern: &checkpoint_owners.intern,
            idempotency: &checkpoint_owners.idempotency,
            permissions: &checkpoint_owners.permissions,
            permissions_tenant: TenantId::DEFAULT,
        };
        incremental_checkpoint(
            &data_dir,
            &BufferPool::new(8, Arc::new(InMemoryPageIo::new())),
            &snapshot,
            &checkpointer,
            || (Vec::new(), None),
            Ok,
        )
    });

    // The checkpoint has stamped and emitted every resident image while the
    // evictor still holds its stale pre-capture epoch sample. Publish that
    // stale sample, perform the atomic fetch-max handoff, remove the resident,
    // and only then let the spill pass classify the published offset.
    capture_entered.wait();
    evict_release.wait();
    evictor.join().unwrap().unwrap();
    capture_release.wait();

    let report = checkpoint
        .join()
        .unwrap()
        .expect("resident-to-spill capture must establish without CountSkew");
    assert_eq!(report.metadata.counts.blob_pages, OVERFLOW_PAGES);
    assert_eq!(blob.logical_page_count(), OVERFLOW_PAGES as usize);
    assert!(blob.evicted_count() > 0, "the raced page must reach spill");
    let selected = read_latest_sidecar(dir.path()).unwrap().unwrap();
    assert_eq!(selected.checkpoint_lsn, report.checkpoint_lsn);
    let reclaimed = reclaim_segments_below(dir.path(), report.checkpoint_lsn).unwrap();
    assert_eq!(reclaimed.deleted_segments, vec![0]);
    assert!(!dir.path().join(segment_filename(0)).exists());
    assert!(dir.path().join(segment_filename(1)).exists());
}

#[cfg(debug_assertions)]
#[test]
fn owner_five_publish_evict_capture_churn_has_zero_count_skew_establishment_failures() {
    const DETERMINISTIC_ROUNDS: usize = 7;
    const OVERFLOW_PAGES: u64 = 3;
    let tenant = TenantId::new(1_468);
    let mut peak_spill_bytes = 0;

    for round in 1..=DETERMINISTIC_ROUNDS {
        let dir = tempdir().unwrap();
        let spill = Arc::new(BlobSpill::open(dir.path()).unwrap());
        let blob = Arc::new(BlobStore::with_bound(
            Arc::clone(&spill),
            BlobBoundConfig::from_cap_bytes((PAGE_SIZE * 4) as u64),
        ));
        for page in 0..OVERFLOW_PAGES {
            blob.put(tenant, &page.to_le_bytes()).unwrap();
        }
        blob.for_each_resident_overflow_page(|_, _, _| Ok::<(), ()>(()))
            .unwrap();

        // Force the exact resident-capture → stale eviction sample → spill
        // publication handoff. The old free-running loop depended on scheduler
        // luck and could append until the CI disk filled; these one-shot gates
        // cover the same assertion path with bounded work every round.
        let evict_sampled = Arc::new(Barrier::new(2));
        let release_evict_sample = Arc::new(Barrier::new(2));
        spill.__test_gate_next_evict_epoch_sample(
            Arc::clone(&evict_sampled),
            Arc::clone(&release_evict_sample),
        );
        let epoch_published = Arc::new(Barrier::new(2));
        let release_epoch_publish = Arc::new(Barrier::new(2));
        spill.__test_gate_after_next_evict_epoch_publish(
            Arc::clone(&epoch_published),
            Arc::clone(&release_epoch_publish),
        );
        let evict_blob = Arc::clone(&blob);
        let evictor = std::thread::spawn(move || evict_blob.force_drain_for_test());
        evict_sampled.wait();

        let resident_captured = Arc::new(Barrier::new(2));
        let release_spill_scan = Arc::new(Barrier::new(2));
        spill.__test_gate_after_next_resident_capture(
            Arc::clone(&resident_captured),
            Arc::clone(&release_spill_scan),
        );
        let owners = Arc::new(Owners::with_blob(Arc::clone(&blob)));
        owners.txn.seed_after_replay(Lsn::new(10));
        let checkpointer = WriteBehindCheckpointer::new(
            Arc::new(DirtyPageTable::new()),
            buffered_store(),
            buffered_store(),
        )
        .with_doublewrite_area(Arc::new(DoublewriteArea::new(dir.path())));
        let checkpoint_owners = Arc::clone(&owners);
        let data_dir = dir.path().to_path_buf();
        let checkpoint = std::thread::spawn(move || {
            let seed = crud_allocator_seed_handle(
                Arc::clone(&checkpoint_owners.crud),
                Arc::clone(&checkpoint_owners.allocator),
            );
            let snapshot = CheckpointSnapshot {
                txn: &checkpoint_owners.txn,
                primary_pages: &checkpoint_owners.primary,
                record_pages: &checkpoint_owners.record,
                blob: &checkpoint_owners.blob,
                allocator_seed: seed.as_ref(),
                intern: &checkpoint_owners.intern,
                idempotency: &checkpoint_owners.idempotency,
                permissions: &checkpoint_owners.permissions,
                permissions_tenant: tenant,
            };
            incremental_checkpoint(
                &data_dir,
                &BufferPool::new(8, Arc::new(InMemoryPageIo::new())),
                &snapshot,
                &checkpointer,
                || (Vec::new(), None),
                Ok,
            )
        });

        resident_captured.wait();
        release_evict_sample.wait();
        epoch_published.wait();
        release_spill_scan.wait();
        let checkpoint_result = checkpoint.join().unwrap();
        release_epoch_publish.wait();
        evictor
            .join()
            .unwrap()
            .expect("deterministic eviction must complete");

        let report = match checkpoint_result {
            Ok(report) => report,
            Err(CheckpointError::CountSkew {
                owner: "v9_blob_overflow",
                header,
                streamed,
            }) => panic!(
                "deterministic round {round} observed owner-5 CountSkew: \
                 header={header}, streamed={streamed}"
            ),
            Err(error) => {
                panic!("unexpected checkpoint establishment failure in round {round}: {error}")
            }
        };
        assert_eq!(report.metadata.counts.blob_pages, OVERFLOW_PAGES);
        assert_eq!(blob.logical_page_count(), OVERFLOW_PAGES as usize);
        assert!(blob.evicted_count() > 0, "round {round} must evict");
        let spill_len = std::fs::metadata(spill.path()).unwrap().len();
        peak_spill_bytes = peak_spill_bytes.max(spill_len);
        assert!(
            spill_len <= OVERFLOW_PAGES * PAGE_SIZE as u64,
            "round {round} spill grew beyond its bounded page set: \
            bytes={spill_len}, pages={OVERFLOW_PAGES}"
        );
    }
    eprintln!(
        "FIX1468_BOUNDED_HAMMER peak_spill_bytes={peak_spill_bytes} \
         cap_bytes={} rounds={DETERMINISTIC_ROUNDS}",
        OVERFLOW_PAGES * PAGE_SIZE as u64
    );
}

#[test]
fn v9_metadata_is_streamed_owner_subset_and_store_five_page_images() {
    let dir = tempdir().unwrap();
    let owners = Owners::new();
    owners.txn.seed_after_replay(Lsn::new(20));
    const IMAGE_PAGES: u64 = 64;
    for page in 0..IMAGE_PAGES {
        owners
            .primary
            .install_fresh(PageId::new(91 + page), PageType::IndexLeaf)
            .unwrap();
        owners.blob.put(TenantId::DEFAULT, b"overflow").unwrap();
    }

    let props = buffered_store();
    let records = buffered_store();
    install_prop_page(&props, PageId::new(1));
    install_record_page(&records, PageId::new(2));
    let dpt = Arc::new(DirtyPageTable::new());
    dpt.mark_dirty(
        DirtyPageKey {
            tenant_id: TenantId::DEFAULT,
            store_id: STORE_PROPS,
            page_no: 1,
        },
        Lsn::new(10),
    );
    dpt.mark_dirty(
        DirtyPageKey {
            tenant_id: TenantId::DEFAULT,
            store_id: STORE_RECORD,
            page_no: 2,
        },
        Lsn::new(11),
    );
    let checkpointer = WriteBehindCheckpointer::new(dpt.clone(), props, records)
        .with_doublewrite_area(Arc::new(DoublewriteArea::new(dir.path())));

    let seed = crud_allocator_seed_handle(Arc::clone(&owners.crud), Arc::clone(&owners.allocator));
    let snapshot = CheckpointSnapshot {
        txn: &owners.txn,
        primary_pages: &owners.primary,
        record_pages: &owners.record,
        blob: &owners.blob,
        allocator_seed: seed.as_ref(),
        intern: &owners.intern,
        idempotency: &owners.idempotency,
        permissions: &owners.permissions,
        permissions_tenant: TenantId::DEFAULT,
    };
    let pool = BufferPool::new(8, Arc::new(InMemoryPageIo::new()));
    let report = incremental_checkpoint(
        dir.path(),
        &pool,
        &snapshot,
        &checkpointer,
        || (Vec::new(), None),
        Ok,
    )
    .unwrap();

    assert_eq!(report.checkpoint_lsn, Lsn::new(20));
    assert_eq!(report.redo_lsn, Lsn::new(20));
    assert_eq!(report.metadata.counts.mvcc_records, 0);
    assert_eq!(report.metadata.counts.record_pages, 0);
    assert_eq!(report.metadata.counts.primary_pages, IMAGE_PAGES);
    assert_eq!(report.metadata.counts.blob_pages, IMAGE_PAGES);
    assert!(report.metadata.max_in_flight <= PAGE_SIZE);
    assert!(
        report.metadata.body_len > report.metadata.max_in_flight as u64 * 100,
        "streaming working set must stay O(page) while total metadata grows"
    );
    assert!(dpt.is_empty());
    assert!(dir.path().join(DOUBLEWRITE_FILE).exists());

    let sidecar = read_latest_sidecar(dir.path()).unwrap().unwrap();
    assert!(sidecar.incremental_metadata);
    assert!(!sidecar.full_state_snapshot);

    let bytes = std::fs::read(incremental_metadata_path(
        dir.path(),
        Lsn::new(20),
        report.metadata.generation,
    ))
    .unwrap();
    assert_eq!(&bytes[0..4], b"AGCM");
    assert_eq!(u16::from_le_bytes(bytes[4..6].try_into().unwrap()), 9);
    assert_eq!(u64::from_le_bytes(bytes[8..16].try_into().unwrap()), 20);
    assert_eq!(u64::from_le_bytes(bytes[16..24].try_into().unwrap()), 20);
    assert_eq!(u64::from_le_bytes(bytes[32..40].try_into().unwrap()), 0);

    // Fixed section order is the structural RED-on-revert: owner tags 1
    // (MVCC) and 3 (record pages) have no slots in the v9 encoder. Owner 2 is
    // primary-only at M3; secondary SMO/page-LSN checkpointing is M4-deferred
    // while secondary mutations retain full images in v9 CommitBundles. Tag 4
    // is retained solely for the Director-ruling store-5 overflow page image.
    let mut pos = 40;
    assert_eq!(bytes[pos], 2);
    let primary = u64::from_le_bytes(bytes[pos + 1..pos + 9].try_into().unwrap()) as usize;
    pos += 9 + primary * (8 + PAGE_SIZE);
    assert_eq!(bytes[pos], 4);
    let overflow = u64::from_le_bytes(bytes[pos + 1..pos + 9].try_into().unwrap()) as usize;
    pos += 9 + overflow * (16 + PAGE_SIZE);
    for expected_tag in 5u8..=8 {
        assert_eq!(bytes[pos], expected_tag);
        let count = u64::from_le_bytes(bytes[pos + 1..pos + 9].try_into().unwrap());
        assert_eq!(count, 0);
        pos += 9;
    }
    assert_eq!(pos + 4, bytes.len(), "only the CRC footer follows owner 8");
    assert_eq!(
        u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()),
        crc32c::crc32c(&bytes[..pos])
    );

    // Phase-5 streaming reader restores only the retained owners and returns
    // the exact redo/DPT anchor without reading the whole metadata into RAM.
    let recovered = Owners::new();
    let recovered_seed = crud_allocator_seed_handle(
        Arc::clone(&recovered.crud),
        Arc::clone(&recovered.allocator),
    );
    let recovered_snapshot = CheckpointSnapshot {
        txn: &recovered.txn,
        primary_pages: &recovered.primary,
        record_pages: &recovered.record,
        blob: &recovered.blob,
        allocator_seed: recovered_seed.as_ref(),
        intern: &recovered.intern,
        idempotency: &recovered.idempotency,
        permissions: &recovered.permissions,
        permissions_tenant: TenantId::DEFAULT,
    };
    let restored = read_incremental_metadata(
        dir.path(),
        &recovered_snapshot,
        Lsn::new(20),
        report.metadata.generation,
    )
    .expect("streaming v9 metadata restore");
    assert_eq!(restored.redo_lsn, Lsn::new(20));
    assert!(restored.dpt.is_empty());
    assert_eq!(recovered.primary.len(), IMAGE_PAGES as usize);
    assert_eq!(recovered.blob.logical_page_count(), IMAGE_PAGES as usize);
    assert_eq!(restored.counts.mvcc_records, 0);
    assert_eq!(restored.counts.record_pages, 0);
}

#[test]
fn v9_establishment_refuses_missing_doublewrite_area() {
    let dir = tempdir().unwrap();
    let owners = Owners::new();
    let props = buffered_store();
    let records = buffered_store();
    let dpt = Arc::new(DirtyPageTable::new());
    let checkpointer = WriteBehindCheckpointer::new(dpt, props, records);
    let seed = crud_allocator_seed_handle(Arc::clone(&owners.crud), Arc::clone(&owners.allocator));
    let snapshot = CheckpointSnapshot {
        txn: &owners.txn,
        primary_pages: &owners.primary,
        record_pages: &owners.record,
        blob: &owners.blob,
        allocator_seed: seed.as_ref(),
        intern: &owners.intern,
        idempotency: &owners.idempotency,
        permissions: &owners.permissions,
        permissions_tenant: TenantId::DEFAULT,
    };
    let pool = BufferPool::new(8, Arc::new(InMemoryPageIo::new()));
    let error = incremental_checkpoint(
        dir.path(),
        &pool,
        &snapshot,
        &checkpointer,
        || (Vec::new(), None),
        Ok,
    )
    .unwrap_err();
    assert!(error.to_string().contains("requires a DoublewriteArea"));
    assert!(read_latest_sidecar(dir.path()).unwrap().is_none());
}

#[test]
fn deferred_periodic_boundary_clamps_logical_and_physical_recovery_floors() {
    let dir = tempdir().unwrap();
    let owners = Owners::new();
    owners.txn.seed_after_replay(Lsn::new(20));
    let props = buffered_store();
    let records = buffered_store();
    let dpt = Arc::new(DirtyPageTable::new());
    let checkpointer = WriteBehindCheckpointer::new(dpt, props, records)
        .with_doublewrite_area(Arc::new(DoublewriteArea::new(dir.path())));
    let seed = crud_allocator_seed_handle(Arc::clone(&owners.crud), Arc::clone(&owners.allocator));
    let snapshot = CheckpointSnapshot {
        txn: &owners.txn,
        primary_pages: &owners.primary,
        record_pages: &owners.record,
        blob: &owners.blob,
        allocator_seed: seed.as_ref(),
        intern: &owners.intern,
        idempotency: &owners.idempotency,
        permissions: &owners.permissions,
        permissions_tenant: TenantId::DEFAULT,
    };
    let pool = BufferPool::new(8, Arc::new(InMemoryPageIo::new()));
    let rendezvous = Arc::new(Barrier::new(2));
    let producer_edge = Arc::clone(&rendezvous);
    let injector = std::thread::spawn(move || {
        rendezvous.wait();
        DeferredV9Boundary {
            commit_lsn: Lsn::new(15),
            redo_lsn: Lsn::new(13),
        }
    });
    let report = incremental_checkpoint(
        dir.path(),
        &pool,
        &snapshot,
        &checkpointer,
        move || {
            producer_edge.wait();
            (Vec::new(), Some(injector.join().unwrap()))
        },
        Ok,
    )
    .unwrap();

    assert_eq!(report.checkpoint_lsn, Lsn::new(14));
    assert_eq!(report.redo_lsn, Lsn::new(13));
    let restored = read_incremental_metadata(
        dir.path(),
        &snapshot,
        Lsn::new(14),
        report.metadata.generation,
    )
    .unwrap();
    assert_eq!(restored.redo_lsn, Lsn::new(13));
}

#[test]
fn owner_two_checkpoint_establishes_under_paced_concurrent_installs() {
    let dir = tempdir().unwrap();
    let orphan_tmp = dir.path().join("CHECKPOINT.v9.0000000000000028.tmp.7.9");
    std::fs::write(&orphan_tmp, b"aborted owner-2 capture").unwrap();
    let owners = Arc::new(Owners::new());
    owners.txn.seed_after_replay(Lsn::new(40));
    for page in 0..4_096u64 {
        owners
            .primary
            .install_fresh(PageId::new(10_000 + page), PageType::IndexLeaf)
            .unwrap();
    }
    let props = buffered_store();
    let records = buffered_store();
    let checkpointer =
        WriteBehindCheckpointer::new(Arc::new(DirtyPageTable::new()), props, records)
            .with_doublewrite_area(Arc::new(DoublewriteArea::new(dir.path())));
    let running = Arc::new(AtomicBool::new(true));
    let mutator_owners = Arc::clone(&owners);
    let mutator_running = Arc::clone(&running);
    let mutator = std::thread::spawn(move || {
        let mut page = 100_000u64;
        while mutator_running.load(Ordering::Acquire) {
            let _commit = mutator_owners.txn.__test_commit_read_guard();
            mutator_owners
                .primary
                .install_fresh(PageId::new(page), PageType::IndexLeaf)
                .unwrap();
            page += 1;
            std::thread::yield_now();
        }
    });

    let producer_owners = Arc::clone(&owners);
    let data_dir = dir.path().to_path_buf();
    let (done_tx, done_rx) = mpsc::channel();
    let producer = std::thread::spawn(move || {
        let seed = crud_allocator_seed_handle(
            Arc::clone(&producer_owners.crud),
            Arc::clone(&producer_owners.allocator),
        );
        let snapshot = CheckpointSnapshot {
            txn: &producer_owners.txn,
            primary_pages: &producer_owners.primary,
            record_pages: &producer_owners.record,
            blob: &producer_owners.blob,
            allocator_seed: seed.as_ref(),
            intern: &producer_owners.intern,
            idempotency: &producer_owners.idempotency,
            permissions: &producer_owners.permissions,
            permissions_tenant: TenantId::DEFAULT,
        };
        let pool = BufferPool::new(8, Arc::new(InMemoryPageIo::new()));
        let result = incremental_checkpoint(
            &data_dir,
            &pool,
            &snapshot,
            &checkpointer,
            || (Vec::new(), None),
            Ok,
        );
        done_tx.send(result).unwrap();
    });

    let report = done_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("checkpoint must establish under sustained owner-2 installs")
        .expect("owner-2 capture must not abort with CountSkew");
    running.store(false, Ordering::Release);
    mutator.join().unwrap();
    producer.join().unwrap();
    assert_eq!(report.checkpoint_lsn, Lsn::new(40));
    assert!(report.metadata.counts.primary_pages >= 4_096);
    assert!(
        !orphan_tmp.exists(),
        "producer startup must sweep orphan temps"
    );
}

#[test]
fn failed_commit_overcapture_window_cannot_persist_phantom_primary_page() {
    let dir = tempdir().unwrap();
    let owners = Arc::new(Owners::new());
    owners.txn.seed_after_replay(Lsn::new(50));
    for page in 0..2_048u64 {
        owners
            .primary
            .install_fresh(PageId::new(20_000 + page), PageType::IndexLeaf)
            .unwrap();
    }
    let props = buffered_store();
    let records = buffered_store();
    let checkpointer =
        WriteBehindCheckpointer::new(Arc::new(DirtyPageTable::new()), props, records)
            .with_doublewrite_area(Arc::new(DoublewriteArea::new(dir.path())));
    let running = Arc::new(AtomicBool::new(true));
    let rollback_owners = Arc::clone(&owners);
    let rollback_running = Arc::clone(&running);
    let phantom = PageId::new(999_999);
    let rollback = std::thread::spawn(move || {
        while rollback_running.load(Ordering::Acquire) {
            let _commit = rollback_owners.txn.__test_commit_read_guard();
            rollback_owners
                .primary
                .install_fresh(phantom, PageType::IndexLeaf)
                .unwrap();
            std::thread::yield_now();
            assert!(rollback_owners.primary.remove_page(phantom).is_some());
        }
    });

    let seed = crud_allocator_seed_handle(Arc::clone(&owners.crud), Arc::clone(&owners.allocator));
    let snapshot = CheckpointSnapshot {
        txn: &owners.txn,
        primary_pages: &owners.primary,
        record_pages: &owners.record,
        blob: &owners.blob,
        allocator_seed: seed.as_ref(),
        intern: &owners.intern,
        idempotency: &owners.idempotency,
        permissions: &owners.permissions,
        permissions_tenant: TenantId::DEFAULT,
    };
    let pool = BufferPool::new(8, Arc::new(InMemoryPageIo::new()));
    let report = incremental_checkpoint(
        dir.path(),
        &pool,
        &snapshot,
        &checkpointer,
        || (Vec::new(), None),
        Ok,
    )
    .unwrap();
    running.store(false, Ordering::Release);
    rollback.join().unwrap();

    let recovered = Owners::new();
    let recovered_seed = crud_allocator_seed_handle(
        Arc::clone(&recovered.crud),
        Arc::clone(&recovered.allocator),
    );
    let recovered_snapshot = CheckpointSnapshot {
        txn: &recovered.txn,
        primary_pages: &recovered.primary,
        record_pages: &recovered.record,
        blob: &recovered.blob,
        allocator_seed: recovered_seed.as_ref(),
        intern: &recovered.intern,
        idempotency: &recovered.idempotency,
        permissions: &recovered.permissions,
        permissions_tenant: TenantId::DEFAULT,
    };
    read_incremental_metadata(
        dir.path(),
        &recovered_snapshot,
        Lsn::new(50),
        report.metadata.generation,
    )
    .unwrap();
    assert!(
        recovered.primary.latch(phantom).is_err(),
        "crash recovery must not serve a primary page rolled back before WAL durability"
    );
}

#[test]
fn exactly_one_incremental_metadata_generation_survives_repeated_establishment() {
    let dir = tempdir().unwrap();
    let owners = Owners::new();
    let props = buffered_store();
    let records = buffered_store();
    let checkpointer =
        WriteBehindCheckpointer::new(Arc::new(DirtyPageTable::new()), props, records)
            .with_doublewrite_area(Arc::new(DoublewriteArea::new(dir.path())));
    let seed = crud_allocator_seed_handle(Arc::clone(&owners.crud), Arc::clone(&owners.allocator));
    let snapshot = CheckpointSnapshot {
        txn: &owners.txn,
        primary_pages: &owners.primary,
        record_pages: &owners.record,
        blob: &owners.blob,
        allocator_seed: seed.as_ref(),
        intern: &owners.intern,
        idempotency: &owners.idempotency,
        permissions: &owners.permissions,
        permissions_tenant: TenantId::DEFAULT,
    };
    let pool = BufferPool::new(8, Arc::new(InMemoryPageIo::new()));

    for frontier in [10, 20, 30] {
        owners.txn.seed_after_replay(Lsn::new(frontier));
        incremental_checkpoint(
            dir.path(),
            &pool,
            &snapshot,
            &checkpointer,
            || (Vec::new(), None),
            Ok,
        )
        .unwrap();
    }

    let live: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|name| name.starts_with("CHECKPOINT.v9.") && name.ends_with(".meta"))
        .collect();
    let sidecar = read_latest_sidecar(dir.path()).unwrap().unwrap();
    assert_eq!(sidecar.checkpoint_lsn, Lsn::new(30));
    assert_eq!(sidecar.metadata_generation, 3);
    assert_eq!(
        live,
        vec![format!(
            "CHECKPOINT.v9.000000000000001e.{:016x}.meta",
            sidecar.metadata_generation
        )]
    );
}

#[test]
fn repeated_frontier_failed_establishment_preserves_selected_metadata_generation() {
    const FRONTIER: Lsn = Lsn::new(50);
    let dir = tempdir().unwrap();
    let owners = Owners::new();
    owners.txn.seed_after_replay(FRONTIER);
    let checkpointer = WriteBehindCheckpointer::new(
        Arc::new(DirtyPageTable::new()),
        buffered_store(),
        buffered_store(),
    )
    .with_doublewrite_area(Arc::new(DoublewriteArea::new(dir.path())));
    let seed = crud_allocator_seed_handle(Arc::clone(&owners.crud), Arc::clone(&owners.allocator));
    let snapshot = CheckpointSnapshot {
        txn: &owners.txn,
        primary_pages: &owners.primary,
        record_pages: &owners.record,
        blob: &owners.blob,
        allocator_seed: seed.as_ref(),
        intern: &owners.intern,
        idempotency: &owners.idempotency,
        permissions: &owners.permissions,
        permissions_tenant: TenantId::DEFAULT,
    };
    let pool = BufferPool::new(8, Arc::new(InMemoryPageIo::new()));

    let first = incremental_checkpoint(
        dir.path(),
        &pool,
        &snapshot,
        &checkpointer,
        || (Vec::new(), None),
        Ok,
    )
    .unwrap();
    assert_eq!(first.checkpoint_lsn, FRONTIER);
    let selected = read_latest_sidecar(dir.path()).unwrap().unwrap();
    let selected_path = incremental_metadata_path(
        dir.path(),
        selected.checkpoint_lsn,
        selected.metadata_generation,
    );
    let selected_bytes = std::fs::read(&selected_path).unwrap();
    assert_eq!(first.metadata.generation, selected.metadata_generation);

    // Model a Periodic owner-2/store-5 install that is visible to the second
    // outside-frontier capture but whose WAL has not reached the final fsync.
    // The second attempt deliberately keeps the same logical frontier.
    let phantom_primary = PageId::new(900_050);
    let mut barrier_calls = 0u8;
    let error = incremental_checkpoint(
        dir.path(),
        &pool,
        &snapshot,
        &checkpointer,
        || {
            owners
                .primary
                .install_fresh(phantom_primary, PageType::IndexLeaf)
                .unwrap();
            owners
                .blob
                .put(TenantId::DEFAULT, b"unfsynced-periodic-store-5")
                .unwrap();
            (Vec::new(), None)
        },
        |horizon| {
            barrier_calls += 1;
            if barrier_calls == 1 {
                Ok(horizon)
            } else {
                Err(arcgraph_storage::checkpoint::CheckpointError::Io(
                    std::io::Error::other("injected final WAL fsync failure"),
                ))
            }
        },
    )
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("injected final WAL fsync failure")
    );

    // Non-vacuity: the aborted replacement metadata really overcaptured both
    // owners, but it is an unselected immutable generation.
    let replacement_generation = selected.metadata_generation + 1;
    let replacement_owners = Owners::new();
    let replacement_seed = crud_allocator_seed_handle(
        Arc::clone(&replacement_owners.crud),
        Arc::clone(&replacement_owners.allocator),
    );
    let replacement_snapshot = CheckpointSnapshot {
        txn: &replacement_owners.txn,
        primary_pages: &replacement_owners.primary,
        record_pages: &replacement_owners.record,
        blob: &replacement_owners.blob,
        allocator_seed: replacement_seed.as_ref(),
        intern: &replacement_owners.intern,
        idempotency: &replacement_owners.idempotency,
        permissions: &replacement_owners.permissions,
        permissions_tenant: TenantId::DEFAULT,
    };
    let replacement = read_incremental_metadata(
        dir.path(),
        &replacement_snapshot,
        FRONTIER,
        replacement_generation,
    )
    .unwrap();
    assert_eq!(replacement.counts.primary_pages, 1);
    assert_eq!(replacement.counts.blob_pages, 1);
    assert!(replacement_owners.primary.latch(phantom_primary).is_ok());
    assert_eq!(replacement_owners.blob.logical_page_count(), 1);

    // Crash/restart still selects the first sidecar and therefore the exact
    // old bytes. Neither unfsynced effect may become durable state.
    let after_failure = read_latest_sidecar(dir.path()).unwrap().unwrap();
    assert_eq!(after_failure, selected);
    assert_eq!(std::fs::read(&selected_path).unwrap(), selected_bytes);
    let recovered = Owners::new();
    let recovered_seed = crud_allocator_seed_handle(
        Arc::clone(&recovered.crud),
        Arc::clone(&recovered.allocator),
    );
    let recovered_snapshot = CheckpointSnapshot {
        txn: &recovered.txn,
        primary_pages: &recovered.primary,
        record_pages: &recovered.record,
        blob: &recovered.blob,
        allocator_seed: recovered_seed.as_ref(),
        intern: &recovered.intern,
        idempotency: &recovered.idempotency,
        permissions: &recovered.permissions,
        permissions_tenant: TenantId::DEFAULT,
    };
    let restored = read_incremental_metadata(
        dir.path(),
        &recovered_snapshot,
        after_failure.checkpoint_lsn,
        after_failure.metadata_generation,
    )
    .unwrap();
    assert_eq!(restored.counts.primary_pages, 0);
    assert_eq!(restored.counts.blob_pages, 0);
    assert!(recovered.primary.latch(phantom_primary).is_err());
    assert_eq!(recovered.blob.logical_page_count(), 0);
}

#[test]
fn real_store_recovers_same_page_number_for_two_tenants_after_incremental_checkpoint() {
    const T1: TenantId = TenantId::new(101);
    const T2: TenantId = TenantId::new(202);
    let dir = tempdir().unwrap();
    std::fs::write(dir.path().join(M3_PROPS_STORE_FILE), []).unwrap();

    let mut segment = SegmentHeader {
        format_version: BUNDLE_FORMAT_V9,
    }
    .encode()
    .to_vec();
    for (txn_id, tenant, base_lsn, node_id) in [(1, T1, 1, 11), (2, T2, 3, 22)] {
        let mut alloc_payload = Vec::with_capacity(9);
        alloc_payload.push(PageType::Node.as_byte());
        alloc_payload.extend_from_slice(&1u64.to_le_bytes());
        let node = NodeRecord::new(
            NodeId::new(node_id),
            LabelId::new(node_id as u32),
            Lsn::new(base_lsn + 1),
        );
        let deltas = vec![
            DeltaOp::new(
                DeltaOpKind::PageAlloc,
                STORE_RECORD,
                tenant,
                1,
                0,
                Lsn::new(base_lsn),
                Bytes::from(alloc_payload),
            )
            .unwrap(),
            DeltaOp::new(
                DeltaOpKind::PutRecord,
                STORE_RECORD,
                tenant,
                1,
                0,
                Lsn::new(base_lsn + 1),
                Bytes::copy_from_slice(&node.to_bytes()),
            )
            .unwrap(),
        ];
        let payload = encode_commit_bundle_v9(
            Lsn::new(base_lsn + 1),
            tenant,
            &std::collections::HashMap::new(),
            &[],
            &deltas,
            &[],
            &[],
            &[],
            &[],
            &[],
        )
        .unwrap();
        WalRecord {
            record_type: WalRecordType::CommitBundle,
            txn_id,
            lsn: Lsn::new(base_lsn + 1),
            timestamp_ms: 0,
            tenant_id: tenant,
            payload,
        }
        .encode(&mut segment)
        .unwrap();
    }
    std::fs::write(dir.path().join(segment_filename(0)), segment).unwrap();

    let owners = Owners::new();
    let props = buffered_store();
    let records = tenant_buffered_store(dir.path());
    let dpt = Arc::new(DirtyPageTable::new());
    let primary_handle: Arc<dyn arcgraph_storage::wal::PrimaryPageStoreHandle> =
        owners.primary.clone();
    let target = PageStoreTarget::primary_only(primary_handle).with_delta_stores(
        Arc::clone(&props) as Arc<dyn arcgraph_storage::DeltaPageStore>,
        Arc::clone(&records) as Arc<dyn arcgraph_storage::DeltaPageStore>,
        Arc::clone(&dpt),
    );
    let mut replay = ReplayExecutor::new(
        ReplayConfig::with_wal_dir(dir.path()),
        Arc::clone(&owners.txn),
        target,
    );
    assert_eq!(
        replay
            .run(WalRecoveryReader::open(dir.path()).unwrap())
            .unwrap(),
        Lsn::new(4)
    );

    let records_flush: Arc<dyn arcgraph_storage::PageFlushTarget> = records.clone();
    let checkpointer = WriteBehindCheckpointer::new(dpt, props, records_flush)
        .with_doublewrite_area(Arc::new(DoublewriteArea::new(dir.path())));
    let seed = crud_allocator_seed_handle(Arc::clone(&owners.crud), Arc::clone(&owners.allocator));
    let snapshot = CheckpointSnapshot {
        txn: &owners.txn,
        primary_pages: &owners.primary,
        record_pages: &owners.record,
        blob: &owners.blob,
        allocator_seed: seed.as_ref(),
        intern: &owners.intern,
        idempotency: &owners.idempotency,
        permissions: &owners.permissions,
        permissions_tenant: TenantId::DEFAULT,
    };
    let report = incremental_checkpoint(
        dir.path(),
        &BufferPool::new(8, Arc::new(InMemoryPageIo::new())),
        &snapshot,
        &checkpointer,
        || (Vec::new(), None),
        Ok,
    )
    .unwrap();
    assert!(m3_record_store_path(dir.path(), T1).is_file());
    assert!(m3_record_store_path(dir.path(), T2).is_file());
    drop(replay);
    drop(records);

    let recovered_records = tenant_buffered_store(dir.path());
    let recovered = Owners::new();
    let base = load_v9_physical_base(
        dir.path(),
        report.checkpoint_lsn,
        &recovered.txn,
        &recovered_records,
        &recovered.blob,
    )
    .unwrap();
    assert_eq!(base.record_pages, 2);
    assert_eq!(base.nodes, 2);
    for (tenant, expected_node) in [(T1, 11), (T2, 22)] {
        let bytes = recovered_records
            .copy_page_pinned_for_tenant(tenant, PageId::new(1))
            .unwrap()
            .unwrap();
        let page = SlottedPageRef::open(bytes.as_ref()).unwrap();
        assert_eq!(TenantId::new(page.header().tenant_id), tenant);
        assert_eq!(
            page.read_node(SlotId(0)).unwrap().unwrap().id,
            expected_node
        );
    }
}

#[test]
fn below_redo_retained_update_cannot_resurrect_checkpointed_delete() {
    let dir = tempdir().unwrap();
    std::fs::write(dir.path().join(M3_PROPS_STORE_FILE), []).unwrap();
    let node_id = NodeId::new(700);
    let page_id = PageId::new(7);
    let key = node_mvcc_key(node_id);

    // Logical order is UPDATE@10 -> DELETE@20, while permanent disk arrival
    // order is DELETE in segment 0 -> UPDATE in active segment 1. Recovery
    // must sort by redo range, not segment order. Reclamation can delete the
    // closed delete segment after its tombstone reaches the home page, while
    // segment-retention slack keeps the below-redo update in active segment 1.
    let mut alloc_payload = Vec::with_capacity(9);
    alloc_payload.push(PageType::Node.as_byte());
    alloc_payload.extend_from_slice(&1u64.to_le_bytes());
    let updated = NodeRecord::new(node_id, LabelId::new(70), Lsn::new(10));
    let update = encode_commit_bundle_v9(
        Lsn::new(10),
        TenantId::DEFAULT,
        &std::collections::HashMap::new(),
        &[],
        &[
            DeltaOp::new(
                DeltaOpKind::PageAlloc,
                STORE_RECORD,
                TenantId::DEFAULT,
                page_id.raw(),
                0,
                Lsn::new(9),
                Bytes::from(alloc_payload),
            )
            .unwrap(),
            DeltaOp::new(
                DeltaOpKind::PutRecord,
                STORE_RECORD,
                TenantId::DEFAULT,
                page_id.raw(),
                0,
                Lsn::new(10),
                Bytes::copy_from_slice(&updated.to_bytes()),
            )
            .unwrap(),
        ],
        &[],
        &[],
        &[],
        &[],
        &[],
    )
    .unwrap();
    let delete = encode_commit_bundle_v9(
        Lsn::new(20),
        TenantId::DEFAULT,
        &std::collections::HashMap::from([(key, None)]),
        &[],
        &[DeltaOp::new(
            DeltaOpKind::TombstoneRecord,
            STORE_RECORD,
            TenantId::DEFAULT,
            page_id.raw(),
            0,
            Lsn::new(20),
            Bytes::new(),
        )
        .unwrap()],
        &[],
        &[],
        &[],
        &[],
        &[],
    )
    .unwrap();
    write_v9_bundle_segment(dir.path(), 0, 20, delete);
    write_v9_bundle_segment(dir.path(), 1, 10, update);

    let owners = Owners::new();
    let props = buffered_store();
    let records = tenant_buffered_store(dir.path());
    let dpt = Arc::new(DirtyPageTable::new());
    let target = PageStoreTarget::primary_only(owners.primary.clone()).with_delta_stores(
        Arc::clone(&props) as Arc<dyn arcgraph_storage::DeltaPageStore>,
        Arc::clone(&records) as Arc<dyn arcgraph_storage::DeltaPageStore>,
        Arc::clone(&dpt),
    );
    let mut replay = ReplayExecutor::new(
        ReplayConfig::with_wal_dir(dir.path()),
        Arc::clone(&owners.txn),
        target,
    );
    assert_eq!(
        replay
            .run(WalRecoveryReader::open(dir.path()).unwrap())
            .unwrap(),
        Lsn::new(20)
    );
    assert!(
        owners
            .txn
            .read_at(TenantId::DEFAULT, key, Lsn::new(20))
            .is_none(),
        "premise: the durable delete wins before checkpoint"
    );

    let records_flush: Arc<dyn arcgraph_storage::PageFlushTarget> = records.clone();
    let checkpointer = WriteBehindCheckpointer::new(Arc::clone(&dpt), props, records_flush)
        .with_doublewrite_area(Arc::new(DoublewriteArea::new(dir.path())));
    let seed = crud_allocator_seed_handle(Arc::clone(&owners.crud), Arc::clone(&owners.allocator));
    let snapshot = CheckpointSnapshot {
        txn: &owners.txn,
        primary_pages: &owners.primary,
        record_pages: &owners.record,
        blob: &owners.blob,
        allocator_seed: seed.as_ref(),
        intern: &owners.intern,
        idempotency: &owners.idempotency,
        permissions: &owners.permissions,
        permissions_tenant: TenantId::DEFAULT,
    };
    let checkpoint = incremental_checkpoint(
        dir.path(),
        &BufferPool::new(8, Arc::new(InMemoryPageIo::new())),
        &snapshot,
        &checkpointer,
        || (Vec::new(), None),
        Ok,
    )
    .unwrap();
    assert_eq!(checkpoint.checkpoint_lsn, Lsn::new(20));
    assert_eq!(checkpoint.redo_lsn, Lsn::new(20));
    let home = records
        .copy_page_pinned_for_tenant(TenantId::DEFAULT, page_id)
        .unwrap()
        .unwrap();
    assert!(
        SlottedPageRef::open(home.as_ref())
            .unwrap()
            .read_node(SlotId(0))
            .unwrap()
            .is_none(),
        "checkpoint home base must contain the durable tombstone"
    );

    let reclaimed = reclaim_segments_below(dir.path(), checkpoint.checkpoint_lsn).unwrap();
    assert_eq!(reclaimed.deleted_segments, vec![0]);
    assert!(!dir.path().join(segment_filename(0)).exists());
    assert!(dir.path().join(segment_filename(1)).exists());
    drop(replay);
    drop(checkpointer);
    drop(records);

    // Restart through the established metadata + real tenant home store,
    // then replay the retained active segment. Its UPDATE range ends at 10,
    // wholly below redo_floor=20, so neither physical nor MVCC state may run.
    let recovered = Owners::new();
    let recovered_seed = crud_allocator_seed_handle(
        Arc::clone(&recovered.crud),
        Arc::clone(&recovered.allocator),
    );
    let recovered_snapshot = CheckpointSnapshot {
        txn: &recovered.txn,
        primary_pages: &recovered.primary,
        record_pages: &recovered.record,
        blob: &recovered.blob,
        allocator_seed: recovered_seed.as_ref(),
        intern: &recovered.intern,
        idempotency: &recovered.idempotency,
        permissions: &recovered.permissions,
        permissions_tenant: TenantId::DEFAULT,
    };
    let restored = read_incremental_metadata(
        dir.path(),
        &recovered_snapshot,
        checkpoint.checkpoint_lsn,
        checkpoint.metadata.generation,
    )
    .unwrap();
    assert_eq!(restored.redo_lsn, checkpoint.redo_lsn);
    let recovered_records = tenant_buffered_store(dir.path());
    let base = load_v9_physical_base(
        dir.path(),
        checkpoint.checkpoint_lsn,
        &recovered.txn,
        &recovered_records,
        &recovered.blob,
    )
    .unwrap();
    assert_eq!(base.record_pages, 1);
    assert_eq!(base.nodes, 0);
    assert!(
        recovered
            .txn
            .read_at(TenantId::DEFAULT, key, checkpoint.checkpoint_lsn)
            .is_none(),
        "checkpoint base must start deleted"
    );
    let mut recovered_replay = ReplayExecutor::new(
        ReplayConfig::with_wal_dir(dir.path()),
        Arc::clone(&recovered.txn),
        PageStoreTarget::primary_only(recovered.primary.clone()).with_delta_stores(
            buffered_store() as Arc<dyn arcgraph_storage::DeltaPageStore>,
            Arc::clone(&recovered_records) as Arc<dyn arcgraph_storage::DeltaPageStore>,
            Arc::new(DirtyPageTable::new()),
        ),
    )
    .with_incremental_checkpoint(checkpoint.checkpoint_lsn, checkpoint.redo_lsn);
    recovered_replay
        .run(WalRecoveryReader::open(dir.path()).unwrap())
        .unwrap();
    assert!(
        recovered
            .txn
            .read_at(TenantId::DEFAULT, key, checkpoint.checkpoint_lsn)
            .is_none(),
        "a retained bundle wholly below redo_floor resurrected the durable delete"
    );
}
