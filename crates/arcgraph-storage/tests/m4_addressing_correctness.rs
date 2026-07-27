use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Barrier, mpsc};
use std::thread;
use std::time::Duration;

use arcgraph_core::record::{NodeRecord, PAGE_SIZE, PageHeader, PageType, RelRecord, node_flags};
use arcgraph_core::{
    DurabilityTier, LabelId, Lsn, NodeId, PageId, RelId, TenantDurabilityLookup, TenantId, TypeId,
};
use arcgraph_storage::address::{AddressError, MAX_ID};
use arcgraph_storage::addressed_store::{
    AddressedRecordStore, AddressedStoreError, address_read_disposition,
};
use arcgraph_storage::crud::{
    CrudStore, PropertyData, REL_TAG_BIT, commit, create_node, create_rel, delete_node_with_store,
    read_node_with_address, read_node_with_store, read_rel_with_address, read_rel_with_store,
    update_node, update_rel,
};
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::{PrimaryIndex, RecordKind};
use arcgraph_storage::records::{
    NODE_CAPACITY, PageError, REL_CAPACITY, SLOT_AREA_START, SLOT_SIZE, SlotId, SlottedPage,
};
use arcgraph_storage::transaction::{Transaction, TxnManager};
use arcgraph_storage::wal::segment::{SegmentHeader, segment_filename};
use arcgraph_storage::wal::{
    BUNDLE_FORMAT_V9, WalConfig, WalRecordType, WalRecoveryReader, WalWriter,
    encode_commit_bundle_v9,
};

#[test]
fn address_boundary_vectors() {
    let tenants = [TenantId::new(41), TenantId::new(73)];
    assert!(tenants.iter().all(|tenant| *tenant != TenantId::DEFAULT));

    let node_cap = u64::from(NODE_CAPACITY);
    let rel_cap = u64::from(REL_CAPACITY);
    let node_vectors = [
        (0, Err(AddressError::ReservedSentinel)),
        (1, Ok((0, 1))),
        (node_cap - 1, Ok((0, NODE_CAPACITY - 1))),
        (node_cap, Ok((1, 0))),
        (node_cap + 1, Ok((1, 1))),
        (MAX_ID - 1, Ok((77_507_328_040_796_435, 41))),
        (MAX_ID, Ok((77_507_328_040_796_435, 42))),
        (MAX_ID + 1, Err(AddressError::OutOfRange)),
    ];
    let rel_vectors = [
        (0, Err(AddressError::ReservedSentinel)),
        (1, Ok((0, 1))),
        (rel_cap - 1, Ok((0, REL_CAPACITY - 1))),
        (rel_cap, Ok((1, 0))),
        (rel_cap + 1, Ok((1, 1))),
        (MAX_ID - 1, Ok((113_868_790_578_454_022, 24))),
        (MAX_ID, Ok((113_868_790_578_454_022, 25))),
        (MAX_ID + 1, Err(AddressError::OutOfRange)),
    ];

    for tenant in tenants {
        for (id, expected) in node_vectors {
            assert_eq!(
                RecordKind::Node.address(id),
                expected,
                "node tenant={tenant:?} id={id}"
            );
        }
        for (id, expected) in rel_vectors {
            assert_eq!(
                RecordKind::Rel.address(id),
                expected,
                "rel tenant={tenant:?} id={id}"
            );
        }

        for kind in [RecordKind::Node, RecordKind::Rel] {
            assert_eq!(
                kind.address(REL_TAG_BIT | 1),
                Err(AddressError::OutOfRange),
                "tagged id accepted for tenant={tenant:?} kind={kind:?}"
            );
        }
    }
}

fn node_record(id: u64) -> NodeRecord {
    NodeRecord::new(NodeId::new(id), LabelId::new(id as u32), Lsn::new(id + 100))
}

fn rel_record(id: u64) -> RelRecord {
    RelRecord::new(
        RelId::new(id),
        TypeId::new(id as u32),
        NodeId::new(id + 1),
        NodeId::new(id + 2),
        Lsn::new(id + 100),
    )
}

fn assert_node_write_at_slot(tenant: TenantId) {
    let header = PageHeader::new(PageId::new(101), PageType::Node, tenant);
    let mut sparse_bytes = [0u8; PAGE_SIZE];
    {
        let mut sparse = SlottedPage::init(&mut sparse_bytes, header).unwrap();
        let at_zero = node_record(1);
        let past_high_water = node_record(3);
        let at_capacity = node_record(u64::from(NODE_CAPACITY));
        sparse.write_node_at_slot(SlotId(0), &at_zero).unwrap();
        sparse
            .write_node_at_slot(SlotId(2), &past_high_water)
            .unwrap();
        sparse
            .write_node_at_slot(SlotId(NODE_CAPACITY - 1), &at_capacity)
            .unwrap();

        assert_eq!(sparse.slot_count(), NODE_CAPACITY);
        assert_eq!(
            sparse.read_node(SlotId(0)).unwrap().unwrap().to_bytes(),
            at_zero.to_bytes()
        );
        assert_eq!(
            sparse.read_node(SlotId(2)).unwrap().unwrap().to_bytes(),
            past_high_water.to_bytes()
        );
        assert_eq!(
            sparse
                .read_node(SlotId(NODE_CAPACITY - 1))
                .unwrap()
                .unwrap()
                .to_bytes(),
            at_capacity.to_bytes()
        );
        assert_eq!(sparse.read_node(SlotId(1)).unwrap(), None);
        assert_eq!(sparse.read_node(SlotId(NODE_CAPACITY - 2)).unwrap(), None);
    }

    let records: Vec<_> = (1..=u64::from(NODE_CAPACITY)).map(node_record).collect();
    let mut dense_bytes = [0u8; PAGE_SIZE];
    let mut addressed_bytes = [0u8; PAGE_SIZE];
    {
        let mut dense = SlottedPage::init(&mut dense_bytes, header).unwrap();
        for record in &records {
            dense.insert_node(record).unwrap();
        }
        let mut addressed = SlottedPage::init(&mut addressed_bytes, header).unwrap();
        for (slot, record) in records.iter().enumerate().rev() {
            addressed
                .write_node_at_slot(SlotId(slot as u16), record)
                .unwrap();
        }
    }
    assert_eq!(addressed_bytes, dense_bytes, "node tenant={tenant:?}");

    assert_unified_node_empty_encoding(tenant, header);
}

fn assert_unified_node_empty_encoding(tenant: TenantId, header: PageHeader) {
    let mut gap_bytes = [0u8; PAGE_SIZE];
    {
        let mut gap = SlottedPage::init(&mut gap_bytes, header).unwrap();
        gap.write_node_at_slot(SlotId(1), &node_record(2)).unwrap();
        assert_eq!(gap.read_node(SlotId(0)).unwrap(), None);
    }
    let mut tombstone_bytes = [0u8; PAGE_SIZE];
    {
        let mut tombstone = SlottedPage::init(&mut tombstone_bytes, header).unwrap();
        tombstone.insert_node(&node_record(1)).unwrap();
        tombstone.tombstone(SlotId(0)).unwrap();
        assert_eq!(tombstone.read_node(SlotId(0)).unwrap(), None);
    }
    assert_eq!(
        &gap_bytes[SLOT_AREA_START..SLOT_AREA_START + SLOT_SIZE],
        &tombstone_bytes[SLOT_AREA_START..SLOT_AREA_START + SLOT_SIZE],
        "node gap and tombstone encodings diverged for tenant={tenant:?}"
    );
}

fn assert_rel_write_at_slot(tenant: TenantId) {
    let header = PageHeader::new(PageId::new(202), PageType::Rel, tenant);
    let mut sparse_bytes = [0u8; PAGE_SIZE];
    {
        let mut sparse = SlottedPage::init(&mut sparse_bytes, header).unwrap();
        let at_zero = rel_record(1);
        let past_high_water = rel_record(3);
        let at_capacity = rel_record(u64::from(REL_CAPACITY));
        sparse.write_rel_at_slot(SlotId(0), &at_zero).unwrap();
        sparse
            .write_rel_at_slot(SlotId(2), &past_high_water)
            .unwrap();
        sparse
            .write_rel_at_slot(SlotId(REL_CAPACITY - 1), &at_capacity)
            .unwrap();

        assert_eq!(sparse.slot_count(), REL_CAPACITY);
        assert_eq!(
            sparse.read_rel(SlotId(0)).unwrap().unwrap().to_bytes(),
            at_zero.to_bytes()
        );
        assert_eq!(
            sparse.read_rel(SlotId(2)).unwrap().unwrap().to_bytes(),
            past_high_water.to_bytes()
        );
        assert_eq!(
            sparse
                .read_rel(SlotId(REL_CAPACITY - 1))
                .unwrap()
                .unwrap()
                .to_bytes(),
            at_capacity.to_bytes()
        );
        assert_eq!(sparse.read_rel(SlotId(1)).unwrap(), None);
        assert_eq!(sparse.read_rel(SlotId(REL_CAPACITY - 2)).unwrap(), None);
    }

    let records: Vec<_> = (1..=u64::from(REL_CAPACITY)).map(rel_record).collect();
    let mut dense_bytes = [0u8; PAGE_SIZE];
    let mut addressed_bytes = [0u8; PAGE_SIZE];
    {
        let mut dense = SlottedPage::init(&mut dense_bytes, header).unwrap();
        for record in &records {
            dense.insert_rel(record).unwrap();
        }
        let mut addressed = SlottedPage::init(&mut addressed_bytes, header).unwrap();
        for (slot, record) in records.iter().enumerate().rev() {
            addressed
                .write_rel_at_slot(SlotId(slot as u16), record)
                .unwrap();
        }
    }
    assert_eq!(addressed_bytes, dense_bytes, "rel tenant={tenant:?}");

    let mut gap_bytes = [0u8; PAGE_SIZE];
    {
        let mut gap = SlottedPage::init(&mut gap_bytes, header).unwrap();
        gap.write_rel_at_slot(SlotId(1), &rel_record(2)).unwrap();
        assert_eq!(gap.read_rel(SlotId(0)).unwrap(), None);
    }
    let mut tombstone_bytes = [0u8; PAGE_SIZE];
    {
        let mut tombstone = SlottedPage::init(&mut tombstone_bytes, header).unwrap();
        tombstone.insert_rel(&rel_record(1)).unwrap();
        tombstone.tombstone(SlotId(0)).unwrap();
        assert_eq!(tombstone.read_rel(SlotId(0)).unwrap(), None);
    }
    assert_eq!(
        &gap_bytes[SLOT_AREA_START..SLOT_AREA_START + SLOT_SIZE],
        &tombstone_bytes[SLOT_AREA_START..SLOT_AREA_START + SLOT_SIZE],
        "rel gap and tombstone encodings diverged for tenant={tenant:?}"
    );
}

#[test]
fn write_at_slot_out_of_order_and_gaps() {
    for tenant in [TenantId::new(41), TenantId::new(73)] {
        assert_ne!(tenant, TenantId::DEFAULT);
        assert_node_write_at_slot(tenant);
        assert_rel_write_at_slot(tenant);
    }
}

#[test]
fn existing_node_page_trailing_gap_is_notfound() {
    let store = AddressedRecordStore::new();
    for tenant in [TenantId::new(41), TenantId::new(73)] {
        assert_ne!(tenant, TenantId::DEFAULT);
        store.write_node(tenant, &node_record(1)).unwrap();
        assert_eq!(store.page_count(tenant, RecordKind::Node), 1);
        assert_eq!(store.read_node(tenant, NodeId::new(2)).unwrap(), None);
    }
}

#[test]
fn existing_rel_page_trailing_gap_is_notfound() {
    let store = AddressedRecordStore::new();
    for tenant in [TenantId::new(41), TenantId::new(73)] {
        assert_ne!(tenant, TenantId::DEFAULT);
        store.write_rel(tenant, &rel_record(1)).unwrap();
        assert_eq!(store.page_count(tenant, RecordKind::Rel), 1);
        assert_eq!(store.read_rel(tenant, RelId::new(2)).unwrap(), None);
    }
}

#[test]
fn address_read_error_taxonomy_guard() {
    let page_errors = vec![
        PageError::Full { needed: 9, free: 8 },
        PageError::SlotOutOfRange { slot: 2, count: 1 },
        PageError::SlotTombstoned(3),
        PageError::WrongPageType {
            got: PageType::Rel.as_byte(),
            expected: PageType::Node.as_byte(),
        },
        PageError::Format("forged slot offset".to_owned()),
        PageError::RecordDecode("bad record version".to_owned()),
        PageError::ChecksumMismatch {
            stored: 0x1111_2222,
            computed: 0x3333_4444,
        },
        PageError::StalePublish {
            slot: 4,
            current: 200,
            incoming: 150,
        },
    ];
    for error in page_errors {
        let expected_discriminant = std::mem::discriminant(&error);
        let is_gap = matches!(error, PageError::SlotOutOfRange { .. });
        let disposition =
            address_read_disposition::<NodeRecord>(Err(AddressedStoreError::Page(error)));
        if is_gap {
            assert!(matches!(disposition, Ok(None)));
        } else {
            match disposition {
                Err(AddressedStoreError::Page(propagated)) => {
                    assert_eq!(std::mem::discriminant(&propagated), expected_discriminant);
                }
                other => panic!("non-gap PageError was masked as {other:?}"),
            }
        }
    }

    let tenant_error = AddressedStoreError::TenantMismatch {
        page_id: PageId::new(7),
        got: TenantId::new(73),
        expected: TenantId::new(41),
    };
    assert!(matches!(
        address_read_disposition::<NodeRecord>(Err(tenant_error)),
        Err(AddressedStoreError::TenantMismatch {
            page_id,
            got,
            expected,
        }) if page_id == PageId::new(7)
            && got == TenantId::new(73)
            && expected == TenantId::new(41)
    ));

    let type_error = AddressedStoreError::PageTypeMismatch {
        page_id: PageId::new(8),
        got: PageType::Rel.as_byte(),
        expected: PageType::Node.as_byte(),
    };
    assert!(matches!(
        address_read_disposition::<NodeRecord>(Err(type_error)),
        Err(AddressedStoreError::PageTypeMismatch {
            page_id,
            got,
            expected,
        }) if page_id == PageId::new(8)
            && got == PageType::Rel.as_byte()
            && expected == PageType::Node.as_byte()
    ));
}

fn assert_node_read_paths_identical(store: &CrudStore, tx: &Transaction<'_>, id: NodeId) {
    let primary = read_node_with_store(store, tx, id).unwrap();
    let addressed = read_node_with_address(store, tx, id).unwrap();
    assert_eq!(
        primary.map(|record| record.to_bytes()),
        addressed.map(|record| record.to_bytes()),
        "node bytes diverged tenant={:?} id={}",
        tx.tenant(),
        id.raw(),
    );
    assert_eq!(
        primary.map(|record| record.is_visible_at(tx.snapshot())),
        addressed.map(|record| record.is_visible_at(tx.snapshot())),
        "node visibility diverged tenant={:?} id={} snapshot={:?}",
        tx.tenant(),
        id.raw(),
        tx.snapshot(),
    );
}

fn assert_rel_read_paths_identical(store: &CrudStore, tx: &Transaction<'_>, id: RelId) {
    let primary = read_rel_with_store(store, tx, id).unwrap();
    let addressed = read_rel_with_address(store, tx, id).unwrap();
    assert_eq!(
        primary.map(|record| record.to_bytes()),
        addressed.map(|record| record.to_bytes()),
        "rel bytes diverged tenant={:?} id={}",
        tx.tenant(),
        id.raw(),
    );
    assert_eq!(
        primary.map(|record| record.is_visible_at(tx.snapshot())),
        addressed.map(|record| record.is_visible_at(tx.snapshot())),
        "rel visibility diverged tenant={:?} id={} snapshot={:?}",
        tx.tenant(),
        id.raw(),
        tx.snapshot(),
    );
}

#[derive(Debug)]
struct Periodic;

impl TenantDurabilityLookup for Periodic {
    fn durability_tier(&self, _tenant: TenantId) -> DurabilityTier {
        DurabilityTier::Periodic { rpo_ms: 60_000 }
    }
}

fn periodic_wal_config(dir: PathBuf) -> WalConfig {
    WalConfig {
        dir,
        segment_size_bytes: 64 * 1024 * 1024,
        group_commit_window: Duration::from_secs(60),
        group_commit_max_batch: 16,
        metrics_sink: None,
        encryption: None,
        inflight_budget_bytes: None,
    }
}

#[test]
fn addressing_only_mvcc_byte_identical() {
    for tenant in [TenantId::new(41), TenantId::new(73)] {
        assert_ne!(tenant, TenantId::DEFAULT);
        // The production v9 record store currently owns one global physical
        // page-id namespace. Use an independent stack per tenant so this
        // differential tests tenant separation without manufacturing a
        // cross-tenant page-id collision outside the addressing invariant.
        let dir = tempfile::tempdir().unwrap();
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
        let primary = Arc::new(
            PrimaryIndex::new(Arc::clone(&manager), Arc::clone(&allocator), None).unwrap(),
        );
        let addressed = Arc::new(AddressedRecordStore::new());
        let mut store = CrudStore::new_with_index(None, Arc::clone(&primary), allocator)
            .with_addressed_record_store(Arc::clone(&addressed));
        let writer = WalWriter::spawn_from(
            periodic_wal_config(dir.path().to_path_buf()),
            manager.current_lsn(),
        )
        .unwrap();
        let wal = writer.handle();
        manager.attach_wal(wal.clone());
        primary.attach_wal(wal.clone());
        store.attach_wal(wal.clone());

        let mut create_tx = manager.begin(tenant);
        let first = create_node(
            &store,
            &mut create_tx,
            tenant,
            LabelId::new(11),
            &PropertyData::InlineU32Pair(1, 2),
        )
        .unwrap();
        let deleted = create_node(
            &store,
            &mut create_tx,
            tenant,
            LabelId::new(12),
            &PropertyData::InlineU32Pair(3, 4),
        )
        .unwrap();
        commit(create_tx, &store).unwrap();

        let mut rel_tx = manager.begin(tenant);
        let rel = create_rel(
            &store,
            &mut rel_tx,
            tenant,
            first,
            deleted,
            TypeId::new(21),
            &PropertyData::InlineU32Pair(5, 6),
        )
        .unwrap();
        commit(rel_tx, &store).unwrap();

        let before_change = manager.begin(tenant);
        let mut change_tx = manager.begin(tenant);
        update_node(
            &store,
            &mut change_tx,
            first,
            &PropertyData::InlineU32Pair(101, 102),
        )
        .unwrap();
        update_rel(
            &store,
            &mut change_tx,
            rel,
            &PropertyData::InlineU32Pair(105, 106),
        )
        .unwrap();
        delete_node_with_store(&store, &mut change_tx, deleted).unwrap();
        let changed_lsn = commit(change_tx, &store).unwrap();
        let current = manager.begin(tenant);

        let pending_boundary = store
            .deferred_v9_boundary()
            .expect("Periodic v9 physical install must remain in flight");
        assert!(
            wal.last_durable_lsn().raw() < changed_lsn.raw(),
            "tenant={tenant:?}: differential must run before install LSN is durable"
        );

        for tx in [&before_change, &current] {
            assert_node_read_paths_identical(&store, tx, first);
            assert_node_read_paths_identical(&store, tx, deleted);
            assert_rel_read_paths_identical(&store, tx, rel);
        }

        let physical_node = addressed.read_node(tenant, first).unwrap().unwrap();
        assert_eq!(physical_node.created_lsn, changed_lsn.raw());
        let physical_rel = addressed.read_rel(tenant, rel).unwrap().unwrap();
        assert_eq!(physical_rel.created_lsn, changed_lsn.raw());
        assert_eq!(addressed.read_node(tenant, deleted).unwrap(), None);
        assert!(addressed.page_count(tenant, RecordKind::Node) > 0);
        assert!(addressed.page_count(tenant, RecordKind::Rel) > 0);

        // Make the install frontier observable instead of comparing the
        // naturally identical addressed and MVCC bytes. Both injected
        // records remain live and snapshot-visible, but their payloads are
        // not authoritative while their created LSN is at or beyond the
        // oldest Periodic-v9 install still in flight. Removing only the
        // pending-frontier predicate returns these divergent physical bytes.
        let mut pending_node_disagreement = physical_node;
        pending_node_disagreement.inline_u32a =
            pending_node_disagreement.inline_u32a.wrapping_add(1);
        assert!(pending_node_disagreement.created_lsn >= pending_boundary.commit_lsn.raw());
        assert!(pending_node_disagreement.is_visible_at(current.snapshot()));
        assert_ne!(
            pending_node_disagreement.to_bytes(),
            physical_node.to_bytes()
        );
        addressed
            .write_node(tenant, &pending_node_disagreement)
            .unwrap();
        assert_node_read_paths_identical(&store, &current, first);
        addressed.write_node(tenant, &physical_node).unwrap();

        let mut pending_rel_disagreement = physical_rel;
        pending_rel_disagreement.inline_u32a = pending_rel_disagreement.inline_u32a.wrapping_add(1);
        assert!(pending_rel_disagreement.created_lsn >= pending_boundary.commit_lsn.raw());
        assert!(pending_rel_disagreement.is_visible_at(current.snapshot()));
        assert_ne!(pending_rel_disagreement.to_bytes(), physical_rel.to_bytes());
        addressed
            .write_rel(tenant, &pending_rel_disagreement)
            .unwrap();
        assert_rel_read_paths_identical(&store, &current, rel);
        addressed.write_rel(tenant, &physical_rel).unwrap();

        wal.flush().unwrap();
        assert_eq!(
            store.drain_deferred_v9_applies().unwrap(),
            3,
            "tenant={tenant:?}: create, rel-create, and change installs must drain"
        );
        assert_node_read_paths_identical(&store, &current, first);
        assert_node_read_paths_identical(&store, &current, deleted);
        assert_rel_read_paths_identical(&store, &current, rel);

        // The alternate is an accelerator, never the authority. Model a
        // valid physical tombstone-flag disagreement after the install queue
        // drains: both selectors must reject that physical verdict and return
        // the byte-identical authoritative MVCC version. Reverting only
        // `read_node_with_address` to a raw created-LSN comparison returns
        // these flagged bytes and makes this differential fail.
        let mut flagged_tombstone = physical_node;
        flagged_tombstone.flags |= node_flags::DELETED;
        addressed.write_node(tenant, &flagged_tombstone).unwrap();
        assert!(!flagged_tombstone.is_visible_at(current.snapshot()));
        assert_node_read_paths_identical(&store, &current, first);
        addressed.write_node(tenant, &physical_node).unwrap();
        writer.shutdown().unwrap();
    }
}

/// ADR-230 amendment-04 P1-a: several post-fsync publishers are in flight at
/// once, including two inverted publishes to the same authoritative slot and
/// an adjacent-slot publish. The older same-slot publisher is deliberately
/// released after the newer one; removing the page-level monotonic guard lets
/// it overwrite LSN 200 with LSN 150 and makes this gate RED.
#[test]
fn addressed_publish_lsn_monotone_concurrent_pipelined() {
    for tenant in [TenantId::new(41), TenantId::new(73)] {
        assert_ne!(tenant, TenantId::DEFAULT);
        let wal_dir = tempfile::tempdir().unwrap();
        std::fs::write(
            wal_dir.path().join(segment_filename(0)),
            SegmentHeader {
                format_version: BUNDLE_FORMAT_V9,
            }
            .encode(),
        )
        .unwrap();
        let writer = WalWriter::spawn(WalConfig {
            dir: wal_dir.path().to_path_buf(),
            segment_size_bytes: 64 * 1024 * 1024,
            group_commit_window: Duration::from_millis(10),
            group_commit_max_batch: 16,
            metrics_sink: None,
            encryption: None,

            inflight_budget_bytes: None,
        })
        .unwrap();
        let wal = writer.handle();
        let metrics = writer.fire_metrics();

        let store = Arc::new(AddressedRecordStore::new());
        let barrier = Arc::new(Barrier::new(3));
        let (newer_done_tx, newer_done_rx) = mpsc::channel();

        let newer_store = Arc::clone(&store);
        let newer_barrier = Arc::clone(&barrier);
        let newer_wal = wal.clone();
        let newer = thread::spawn(move || {
            newer_barrier.wait();
            let commit_lsn = Lsn::new(200);
            let payload = encode_commit_bundle_v9(
                commit_lsn,
                tenant,
                &HashMap::new(),
                &[],
                &[],
                &[],
                &[],
                &[],
                &[],
                &[],
            )
            .unwrap();
            newer_wal
                .append_at(
                    commit_lsn,
                    WalRecordType::CommitBundle,
                    200,
                    0,
                    tenant,
                    payload,
                )
                .unwrap();
            let record = NodeRecord::new(NodeId::new(1), LabelId::new(7), Lsn::new(200));
            newer_store.write_node(tenant, &record).unwrap();
            newer_done_tx.send(()).unwrap();
        });

        let older_store = Arc::clone(&store);
        let older_barrier = Arc::clone(&barrier);
        let older_wal = wal.clone();
        let older = thread::spawn(move || {
            older_barrier.wait();
            let commit_lsn = Lsn::new(150);
            let payload = encode_commit_bundle_v9(
                commit_lsn,
                tenant,
                &HashMap::new(),
                &[],
                &[],
                &[],
                &[],
                &[],
                &[],
                &[],
            )
            .unwrap();
            older_wal
                .append_at(
                    commit_lsn,
                    WalRecordType::CommitBundle,
                    150,
                    0,
                    tenant,
                    payload,
                )
                .unwrap();
            newer_done_rx.recv().unwrap();
            let record = NodeRecord::new(NodeId::new(1), LabelId::new(7), Lsn::new(150));
            older_store.write_node(tenant, &record)
        });

        let adjacent_store = Arc::clone(&store);
        let adjacent_barrier = Arc::clone(&barrier);
        let adjacent_wal = wal.clone();
        let adjacent = thread::spawn(move || {
            adjacent_barrier.wait();
            let commit_lsn = Lsn::new(220);
            let payload = encode_commit_bundle_v9(
                commit_lsn,
                tenant,
                &HashMap::new(),
                &[],
                &[],
                &[],
                &[],
                &[],
                &[],
                &[],
            )
            .unwrap();
            adjacent_wal
                .append_at(
                    commit_lsn,
                    WalRecordType::CommitBundle,
                    220,
                    0,
                    tenant,
                    payload,
                )
                .unwrap();
            let record = NodeRecord::new(NodeId::new(2), LabelId::new(8), Lsn::new(220));
            adjacent_store.write_node(tenant, &record).unwrap();
        });

        newer.join().unwrap();
        adjacent.join().unwrap();
        assert!(matches!(
            older.join().unwrap(),
            Err(AddressedStoreError::Page(PageError::StalePublish {
                current: 200,
                incoming: 150,
                ..
            }))
        ));
        assert_eq!(
            store
                .read_node(tenant, NodeId::new(1))
                .unwrap()
                .unwrap()
                .created_lsn,
            200
        );
        assert_eq!(
            store
                .read_node(tenant, NodeId::new(2))
                .unwrap()
                .unwrap()
                .created_lsn,
            220
        );

        let newest_rel = RelRecord::new(
            RelId::new(1),
            TypeId::new(9),
            NodeId::new(1),
            NodeId::new(2),
            Lsn::new(240),
        );
        store.write_rel(tenant, &newest_rel).unwrap();
        let stale_rel = RelRecord::new(
            RelId::new(1),
            TypeId::new(9),
            NodeId::new(1),
            NodeId::new(2),
            Lsn::new(180),
        );
        assert!(matches!(
            store.write_rel(tenant, &stale_rel),
            Err(AddressedStoreError::Page(PageError::StalePublish {
                current: 240,
                incoming: 180,
                ..
            }))
        ));

        assert!(
            store
                .tombstone_node_at_lsn(tenant, NodeId::new(1), Lsn::new(250))
                .unwrap()
        );
        assert!(matches!(
            store.write_node(
                tenant,
                &NodeRecord::new(NodeId::new(1), LabelId::new(10), Lsn::new(300))
            ),
            Err(AddressedStoreError::PermanentTombstone {
                id: 1,
                tombstone_lsn: 250,
                ..
            })
        ));
        assert_eq!(store.read_node(tenant, NodeId::new(1)).unwrap(), None);

        assert!(
            store
                .tombstone_rel_at_lsn(tenant, RelId::new(1), Lsn::new(260))
                .unwrap()
        );
        assert!(matches!(
            store.write_rel(
                tenant,
                &RelRecord::new(
                    RelId::new(1),
                    TypeId::new(9),
                    NodeId::new(1),
                    NodeId::new(2),
                    Lsn::new(310),
                )
            ),
            Err(AddressedStoreError::PermanentTombstone {
                id: 1,
                tombstone_lsn: 260,
                ..
            })
        ));
        assert_eq!(store.read_rel(tenant, RelId::new(1)).unwrap(), None);
        writer.shutdown().unwrap();
        assert_eq!(metrics.total_records_fired(), 3);
        assert!(
            metrics.total_fires() < metrics.total_records_fired(),
            "three concurrent commits were serialized one-fsync-per-commit"
        );
        let records = WalRecoveryReader::open(wal_dir.path())
            .unwrap()
            .collect::<arcgraph_core::Result<Vec<_>>>()
            .unwrap();
        assert_eq!(records.len(), 3);
        assert!(
            records
                .iter()
                .all(|record| record.record_type == WalRecordType::CommitBundle)
        );
    }
}
