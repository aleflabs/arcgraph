use std::collections::HashMap;
use std::sync::{Arc, Barrier};
use std::thread;

use arcgraph_core::record::{NodeRecord, PageHeader};
use arcgraph_core::{ArcGraphError, LabelId, Lsn, NodeId, PAGE_SIZE, PageId, PageType, TenantId};
use arcgraph_storage::address::MAX_ID;
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::checkpoint::{
    DoublewriteArea, ExtentDirectoryDoublewriteHome, WriteBehindCheckpointer,
};
use arcgraph_storage::extent::{
    DIR_PAGE_TAG, DIRECTORY_ENTRIES_PER_PAGE, DIRECTORY_ENTRY_BYTES, DIRECTORY_HEAD_BYTES,
    DirectoryHeadPageIo, EXTENT_BYTES, ExtentAllocation, ExtentApplyOutcome, ExtentDataPageStore,
    ExtentDirectory, PairedAffinityAllocator, directory_page_no, directory_page_offset,
};
use arcgraph_storage::io::{PageIo, PosixPageIo};
use arcgraph_storage::primary_index::{PrimaryPageStore, RecordKind};
use arcgraph_storage::records::{NODE_CAPACITY, SlotId, SlottedPageRef};
use arcgraph_storage::redo::{DeltaPageStore, DirtyPageKey, DirtyPageTable, apply_recovery_delta};
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{
    BUNDLE_FORMAT_V9, DeltaIntent, DeltaOp, DeltaOpKind, PageStoreTarget, ReplayConfig,
    ReplayExecutor, STORE_PROPS, STORE_RECORD, STORE_TEL, SegmentHeader, WalRecord, WalRecordType,
    WalRecoveryReader, decode_commit_bundle_v9, encode_commit_bundle_v9, segment_filename,
};
use bytes::Bytes;

#[test]
fn extent_directory_identity_and_tag_boundary() {
    let tenants = [TenantId::new(41), TenantId::new(73)];
    assert!(tenants.iter().all(|tenant| *tenant != TenantId::DEFAULT));

    for kind in [RecordKind::Node, RecordKind::Rel] {
        let (max_page_no, _) = kind.address(MAX_ID).unwrap();
        assert!(max_page_no < DIR_PAGE_TAG, "{kind:?} aliases directory tag");
        assert_ne!(max_page_no, directory_page_no(0).unwrap());
        assert_ne!(max_page_no, directory_page_no(1).unwrap());
    }

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("nodes.store");
    let physical = Arc::new(PosixPageIo::create(path).unwrap());
    for k in [0_u64, 1, 17] {
        let mut bytes = [0_u8; PAGE_SIZE];
        bytes[..8].copy_from_slice(&k.to_le_bytes());
        physical.write_page(PageId::new(k), &bytes).unwrap();
    }
    physical.flush().unwrap();

    let head_io = Arc::new(DirectoryHeadPageIo::new(physical));
    let pool = BufferPool::new(8, head_io);
    for tenant in tenants {
        for k in [0_u64, 1, 17] {
            let page_no = directory_page_no(k).unwrap();
            assert_eq!(page_no, DIR_PAGE_TAG | k);
            assert_eq!(
                directory_page_offset(page_no).unwrap(),
                k * PAGE_SIZE as u64
            );
            let page = pool.pin_read(PageId::new(page_no)).unwrap();
            assert_eq!(&page.as_bytes()[..8], &k.to_le_bytes(), "tenant={tenant:?}");
        }
    }
}

fn extent_bundle(tenant: TenantId, allocation: ExtentAllocation, lsn: Lsn) -> Vec<u8> {
    let intent = DeltaIntent::extent_alloc(STORE_RECORD, tenant, allocation)
        .assign(lsn, lsn)
        .unwrap();
    encode_commit_bundle_v9(
        lsn,
        tenant,
        &HashMap::new(),
        &[],
        &[intent],
        &[],
        &[],
        &[],
        &[],
        &[],
    )
    .unwrap()
}

fn write_real_wal(dir: &std::path::Path, tenant: TenantId, payload: Vec<u8>, lsn: Lsn) {
    let mut bytes = SegmentHeader {
        format_version: BUNDLE_FORMAT_V9,
    }
    .encode()
    .to_vec();
    WalRecord {
        record_type: WalRecordType::CommitBundle,
        txn_id: tenant.raw(),
        lsn,
        timestamp_ms: 0,
        tenant_id: tenant,
        payload,
    }
    .encode(&mut bytes)
    .unwrap();
    std::fs::write(dir.join(segment_filename(0)), bytes).unwrap();
}

#[test]
fn extent_alloc_delta_page_lsn_idempotent_conflict_corrupts() {
    for (ordinal, tenant) in [TenantId::new(41), TenantId::new(73)]
        .into_iter()
        .enumerate()
    {
        assert_ne!(tenant, TenantId::DEFAULT);
        let fixture = tempfile::tempdir().unwrap();
        let wal_dir = fixture.path().join("wal");
        std::fs::create_dir(&wal_dir).unwrap();
        let store_path = fixture.path().join("nodes.store");
        let physical = Arc::new(PosixPageIo::create(store_path).unwrap());
        let directory = Arc::new(ExtentDirectory::new(
            tenant,
            STORE_RECORD,
            physical.clone(),
            8,
        ));
        let dpt = Arc::new(DirtyPageTable::new());
        let allocation = ExtentAllocation {
            logical_extent: 7,
            physical_offset: DIRECTORY_HEAD_BYTES,
            pairing: 99,
        };
        let lsn = Lsn::new(100 + ordinal as u64);
        write_real_wal(
            &wal_dir,
            tenant,
            extent_bundle(tenant, allocation, lsn),
            lsn,
        );

        let records: Vec<_> = WalRecoveryReader::open(&wal_dir)
            .unwrap()
            .collect::<arcgraph_core::Result<Vec<_>>>()
            .unwrap();
        assert_eq!(records.len(), 1);
        let decoded = decode_commit_bundle_v9(&records[0].payload, records[0].tenant_id).unwrap();
        assert_eq!(decoded.deltas.len(), 1);
        assert_eq!(
            directory
                .apply_extent_alloc(&decoded.deltas[0], &dpt)
                .unwrap(),
            ExtentApplyOutcome::Applied
        );
        assert_eq!(
            directory
                .apply_extent_alloc(&decoded.deltas[0], &dpt)
                .unwrap(),
            ExtentApplyOutcome::Idempotent
        );
        assert_eq!(
            directory.mapping(allocation.logical_extent).unwrap(),
            Some(allocation)
        );
        assert_eq!(dpt.len(), 1);

        let dwb = Arc::new(DoublewriteArea::new(fixture.path()));
        let inert = Arc::new(ExtentDirectory::new(
            tenant,
            STORE_RECORD + 100,
            Arc::new(arcgraph_storage::io::InMemoryPageIo::new()),
            2,
        ));
        let checkpointer = WriteBehindCheckpointer::new(dpt.clone(), inert.clone(), inert)
            .with_directory_target(STORE_RECORD, directory.clone())
            .with_doublewrite_area(dwb.clone());
        let report = checkpointer.flush_pass_with_doublewrite(lsn).unwrap();
        assert_eq!(report.flushed_pages, 1);
        assert!(dpt.is_empty());

        let tagged_page = allocation.directory_page_no().unwrap();
        let physical_page =
            PageId::new(directory_page_offset(tagged_page).unwrap() / PAGE_SIZE as u64);
        physical
            .write_page(physical_page, &[0_u8; PAGE_SIZE])
            .unwrap();
        physical.flush().unwrap();
        let mut restore_home =
            ExtentDirectoryDoublewriteHome::new().with_directory(directory.clone());
        let restore = dwb.restore(&mut restore_home).unwrap();
        assert_eq!(restore.valid_slots, 1);
        assert_eq!(restore.restored_pages, 1);
        let reopened = ExtentDirectory::new(tenant, STORE_RECORD, physical.clone(), 2);
        assert_eq!(
            reopened.mapping(allocation.logical_extent).unwrap(),
            Some(allocation)
        );

        for conflict in [
            ExtentAllocation {
                physical_offset: allocation.physical_offset + EXTENT_BYTES,
                ..allocation
            },
            ExtentAllocation {
                pairing: allocation.pairing + 1,
                ..allocation
            },
        ] {
            let conflict_op = DeltaIntent::extent_alloc(STORE_RECORD, tenant, conflict)
                .assign(Lsn::new(lsn.raw() + 1), Lsn::new(lsn.raw() + 1))
                .unwrap();
            let error = directory
                .apply_extent_alloc(&conflict_op, &dpt)
                .unwrap_err();
            assert!(matches!(error, ArcGraphError::WalCorruption { .. }));
        }
        assert_eq!(
            directory.mapping(allocation.logical_extent).unwrap(),
            Some(allocation)
        );

        // The generation is part of the durable entry identity even though
        // v1 ExtentAlloc payloads always install generation 1. A same-shape
        // mapping with a hostile persisted generation must not be accepted as
        // an idempotent replay.
        let mut head = [0_u8; PAGE_SIZE];
        physical.read_page(physical_page, &mut head).unwrap();
        let slot = (allocation.logical_extent % DIRECTORY_ENTRIES_PER_PAGE) as usize;
        let entry_offset = PageHeader::SIZE + slot * DIRECTORY_ENTRY_BYTES;
        head[entry_offset + 20..entry_offset + 24].copy_from_slice(&2_u32.to_le_bytes());
        let header_bytes: &[u8; PageHeader::SIZE] = head[..PageHeader::SIZE].try_into().unwrap();
        let mut header = PageHeader::from_bytes(header_bytes).unwrap();
        header.checksum = crc32c::crc32c(&head[PageHeader::SIZE..]);
        head[..PageHeader::SIZE].copy_from_slice(&header.to_bytes());
        physical.write_page(physical_page, &head).unwrap();
        physical.flush().unwrap();
        let generation_conflict = ExtentDirectory::new(tenant, STORE_RECORD, physical.clone(), 2);
        let error = generation_conflict
            .mapping(allocation.logical_extent)
            .unwrap_err();
        assert!(matches!(error, ArcGraphError::WalCorruption { .. }));
        assert!(error.to_string().contains("generation 2"));
        let error = generation_conflict
            .apply_extent_alloc(&decoded.deltas[0], &dpt)
            .unwrap_err();
        assert!(matches!(error, ArcGraphError::WalCorruption { .. }));
        assert!(error.to_string().contains("unique dense physical ledger"));
    }

    let tenant = TenantId::new(41);
    let allocation = ExtentAllocation {
        logical_extent: 3,
        physical_offset: DIRECTORY_HEAD_BYTES + 500 * EXTENT_BYTES,
        pairing: 8,
    };
    let extent = DeltaIntent::extent_alloc(STORE_RECORD, tenant, allocation)
        .assign(Lsn::new(200), Lsn::new(201))
        .unwrap();
    let page_alloc = DeltaOp::new(
        DeltaOpKind::PageAlloc,
        STORE_RECORD,
        tenant,
        allocation.logical_extent * 256,
        0,
        Lsn::new(201),
        {
            let mut payload = vec![PageType::Node.as_byte()];
            payload.extend_from_slice(&1_u64.to_le_bytes());
            Bytes::from(payload)
        },
    )
    .unwrap();
    let good = encode_commit_bundle_v9(
        Lsn::new(201),
        tenant,
        &HashMap::new(),
        &[],
        &[extent.clone(), page_alloc.clone()],
        &[],
        &[],
        &[],
        &[],
        &[],
    )
    .unwrap();
    let tail_start = 16 + extent.encoded_len() + page_alloc.encoded_len();
    let mut out_of_order = Vec::new();
    out_of_order.extend_from_slice(&201_u64.to_le_bytes());
    out_of_order.extend_from_slice(&0_u32.to_le_bytes());
    out_of_order.extend_from_slice(&2_u32.to_le_bytes());
    let early_page = DeltaOp {
        op_lsn: Lsn::new(200),
        ..page_alloc
    };
    let late_extent = DeltaOp {
        op_lsn: Lsn::new(201),
        ..extent
    };
    early_page.encode_into(&mut out_of_order).unwrap();
    late_extent.encode_into(&mut out_of_order).unwrap();
    out_of_order.extend_from_slice(&good[tail_start..]);
    let error = decode_commit_bundle_v9(&out_of_order, tenant).unwrap_err();
    assert!(matches!(error, ArcGraphError::WalCorruption { .. }));
    assert!(error.to_string().contains("ExtentAlloc"));

    // Frame the hostile payload through the real WAL so this lower-bound
    // check cannot be satisfied solely by the intent constructor.
    let fixture = tempfile::tempdir().unwrap();
    let wal_dir = fixture.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();
    let valid = ExtentAllocation {
        logical_extent: 4,
        physical_offset: DIRECTORY_HEAD_BYTES + EXTENT_BYTES,
        pairing: 4,
    };
    let lsn = Lsn::new(220);
    let mut hostile = extent_bundle(tenant, valid, lsn);
    let physical_offset = 16 + DeltaOp::FIXED_PREFIX_LEN + 8;
    hostile[physical_offset..physical_offset + 8]
        .copy_from_slice(&(DIRECTORY_HEAD_BYTES - PAGE_SIZE as u64).to_le_bytes());
    write_real_wal(&wal_dir, tenant, hostile, lsn);
    let records = WalRecoveryReader::open(&wal_dir)
        .unwrap()
        .collect::<arcgraph_core::Result<Vec<_>>>()
        .unwrap();
    let error = decode_commit_bundle_v9(&records[0].payload, tenant).unwrap_err();
    assert!(matches!(error, ArcGraphError::WalCorruption { .. }));
    assert!(error.to_string().contains("reserved directory head"));
}

#[test]
fn apply_extent_alloc_rejects_non_dense_physical_offset() {
    for tenant in [TenantId::new(41), TenantId::new(73)] {
        let fixture = tempfile::tempdir().unwrap();
        let wal_dir = fixture.path().join("wal");
        std::fs::create_dir(&wal_dir).unwrap();
        let allocation = ExtentAllocation {
            logical_extent: 17,
            physical_offset: DIRECTORY_HEAD_BYTES + EXTENT_BYTES,
            pairing: 17,
        };
        let lsn = Lsn::new(250);
        write_real_wal(
            &wal_dir,
            tenant,
            extent_bundle(tenant, allocation, lsn),
            lsn,
        );
        let records = WalRecoveryReader::open(&wal_dir)
            .unwrap()
            .collect::<arcgraph_core::Result<Vec<_>>>()
            .unwrap();
        let decoded = decode_commit_bundle_v9(&records[0].payload, tenant).unwrap();
        let directory = ExtentDirectory::new(
            tenant,
            STORE_RECORD,
            Arc::new(PosixPageIo::create(fixture.path().join("nodes.store")).unwrap()),
            4,
        );
        let error = directory
            .apply_extent_alloc(&decoded.deltas[0], &DirtyPageTable::new())
            .expect_err("the first installed extent must start at the directory head");
        assert!(matches!(error, ArcGraphError::WalCorruption { .. }));
        assert!(error.to_string().contains("not dense next offset"));
        assert_eq!(directory.mapping(allocation.logical_extent).unwrap(), None);
    }
}

fn crash_trace_bundle(
    tenant: TenantId,
    allocation: ExtentAllocation,
    node: &NodeRecord,
) -> Vec<u8> {
    let page_no = allocation.logical_extent * 256;
    let extent = DeltaIntent::extent_alloc(STORE_RECORD, tenant, allocation)
        .assign(Lsn::new(300), Lsn::new(302))
        .unwrap();
    let page = DeltaIntent::page_alloc(STORE_RECORD, tenant, page_no, PageType::Node, 1)
        .assign(Lsn::new(301), Lsn::new(302))
        .unwrap();
    let put = DeltaOp::new(
        DeltaOpKind::PutRecord,
        STORE_RECORD,
        tenant,
        page_no,
        1,
        Lsn::new(302),
        Bytes::copy_from_slice(&node.to_bytes()),
    )
    .unwrap();
    encode_commit_bundle_v9(
        Lsn::new(302),
        tenant,
        &HashMap::new(),
        &[],
        &[extent, page, put],
        &[],
        &[],
        &[],
        &[],
        &[],
    )
    .unwrap()
}

#[test]
fn extent_directory_recovers_before_page_ops() {
    for (ordinal, tenant) in [TenantId::new(41), TenantId::new(73)]
        .into_iter()
        .enumerate()
    {
        assert_ne!(tenant, TenantId::DEFAULT);
        let fixture = tempfile::tempdir().unwrap();
        let wal_dir = fixture.path().join("wal");
        std::fs::create_dir(&wal_dir).unwrap();
        let store_path = fixture.path().join("nodes.store");
        let logical_extent = 2 + ordinal as u64;
        let id = logical_extent * 256 * u64::from(NODE_CAPACITY) + 1;
        let node = NodeRecord::new(NodeId::new(id), LabelId::new(7), Lsn::new(302));
        let allocation = ExtentAllocation {
            logical_extent,
            physical_offset: DIRECTORY_HEAD_BYTES,
            pairing: 55,
        };
        write_real_wal(
            &wal_dir,
            tenant,
            crash_trace_bundle(tenant, allocation, &node),
            Lsn::new(302),
        );

        // Deterministic crash injection: construct the committed cache state,
        // then drop every cache/DPT owner without invoking the checkpointer.
        {
            let physical = Arc::new(PosixPageIo::create(&store_path).unwrap());
            let directory = Arc::new(ExtentDirectory::new(tenant, STORE_RECORD, physical, 8));
            let data = Arc::new(ExtentDataPageStore::new(directory.clone(), 8));
            let dpt = DirtyPageTable::new();
            let decoded =
                decode_commit_bundle_v9(&crash_trace_bundle(tenant, allocation, &node), tenant)
                    .unwrap();
            directory
                .apply_extent_alloc(&decoded.deltas[0], &dpt)
                .unwrap();
            for op in &decoded.deltas[1..] {
                apply_recovery_delta(data.as_ref(), data.as_ref(), &dpt, op, Lsn::new(302))
                    .unwrap();
            }
            assert_eq!(directory.mapping(logical_extent).unwrap(), Some(allocation));
            assert_eq!(dpt.len(), 2);
        }

        let physical = Arc::new(PosixPageIo::open(&store_path).unwrap());
        let recovered_directory = Arc::new(ExtentDirectory::new(tenant, STORE_RECORD, physical, 8));
        assert_eq!(
            recovered_directory.mapping(logical_extent).unwrap(),
            None,
            "fault injection accidentally checkpointed directory state"
        );
        let recovered_data = Arc::new(ExtentDataPageStore::new(recovered_directory.clone(), 8));
        let recovery_dpt = Arc::new(DirtyPageTable::new());
        let target = PageStoreTarget::primary_only(Arc::new(PrimaryPageStore::new()))
            .with_delta_stores(
                recovered_data.clone(),
                recovered_data.clone(),
                recovery_dpt.clone(),
            )
            .with_extent_directory(recovered_directory.clone());
        let mut replay = ReplayExecutor::new(
            ReplayConfig::with_wal_dir(&wal_dir),
            Arc::new(TxnManager::new()),
            target,
        );
        assert_eq!(
            replay
                .run(WalRecoveryReader::open(&wal_dir).unwrap())
                .unwrap(),
            Lsn::new(302)
        );
        assert_eq!(
            recovered_directory.mapping(logical_extent).unwrap(),
            Some(allocation)
        );
        let (page_no, slot) = RecordKind::Node.address(id).unwrap();
        assert_eq!(page_no, logical_extent * 256);
        assert_eq!(
            recovered_directory.resolve_data_page(page_no).unwrap(),
            allocation.physical_offset
        );
        let page = recovered_data
            .read_page_for_redo(tenant, PageId::new(page_no))
            .unwrap()
            .unwrap();
        let page = SlottedPageRef::open(page.as_ref()).unwrap();
        assert_eq!(
            page.read_node(SlotId(slot)).unwrap().unwrap().to_bytes(),
            node.to_bytes()
        );
        assert_eq!(recovery_dpt.len(), 2);
    }
}

fn marker_page(
    tenant: TenantId,
    page_no: u64,
    page_type: PageType,
    marker: u64,
    lsn: Lsn,
) -> Box<[u8; PAGE_SIZE]> {
    let mut bytes = Box::new([0_u8; PAGE_SIZE]);
    bytes[PageHeader::SIZE..PageHeader::SIZE + 8].copy_from_slice(&marker.to_le_bytes());
    let mut header = PageHeader::new(PageId::new(page_no), page_type, tenant);
    header.lsn = lsn.raw();
    header.checksum = crc32c::crc32c(&bytes[PageHeader::SIZE..]);
    bytes[..PageHeader::SIZE].copy_from_slice(&header.to_bytes());
    bytes
}

fn read_marker(store: &ExtentDataPageStore, tenant: TenantId, page_no: u64) -> u64 {
    let bytes = store
        .read_page_for_redo(tenant, PageId::new(page_no))
        .unwrap()
        .unwrap();
    u64::from_le_bytes(
        bytes[PageHeader::SIZE..PageHeader::SIZE + 8]
            .try_into()
            .unwrap(),
    )
}

#[test]
fn affinity_pairing_correctness_neutral_under_race() {
    for tenant in [TenantId::new(41), TenantId::new(73)] {
        assert_ne!(tenant, TenantId::DEFAULT);
        let fixture = tempfile::tempdir().unwrap();
        let props_path = fixture.path().join("props.store");
        let tel_path = fixture.path().join("tel.store");
        let props_physical = Arc::new(PosixPageIo::create(&props_path).unwrap());
        let tel_physical = Arc::new(PosixPageIo::create(&tel_path).unwrap());
        let props_directory =
            Arc::new(ExtentDirectory::new(tenant, STORE_PROPS, props_physical, 4));
        let tel_directory = Arc::new(ExtentDirectory::new(tenant, STORE_TEL, tel_physical, 4));
        let props_data = Arc::new(ExtentDataPageStore::new(props_directory.clone(), 8));
        let tel_data = Arc::new(ExtentDataPageStore::new(tel_directory.clone(), 8));
        let dpt = Arc::new(DirtyPageTable::new());
        let pairer = Arc::new(
            PairedAffinityAllocator::new_recovered(
                props_data.clone(),
                tel_data.clone(),
                dpt.clone(),
            )
            .unwrap(),
        );
        let record_page = 9 * 256 + 7;
        let barrier = Arc::new(Barrier::new(2));

        let mut joins = Vec::new();
        for writer in 0_u64..2 {
            let pairer = pairer.clone();
            let props_data = props_data.clone();
            let tel_data = tel_data.clone();
            let dpt = dpt.clone();
            let barrier = barrier.clone();
            joins.push(thread::spawn(move || {
                barrier.wait();
                let base_lsn = Lsn::new(500 + writer * 10);
                let placement = pairer.place(record_page, base_lsn).unwrap();
                let extent_allocs = placement.extent_allocs.len();
                let commit_lsn = placement.last_op_lsn();
                let placement = placement.install_committed(commit_lsn).unwrap();
                let marker = 0xA11F_0000 + writer;
                let pages = [
                    (STORE_PROPS, placement.property_page, PageType::PropSlotted),
                    (STORE_TEL, placement.out_tel_page, PageType::Tel),
                    (STORE_TEL, placement.in_tel_page, PageType::Tel),
                ];
                for (index, (store_id, page_no, page_type)) in pages.into_iter().enumerate() {
                    let op_lsn = Lsn::new(base_lsn.raw() + 2 + index as u64);
                    let page = marker_page(tenant, page_no, page_type, marker, op_lsn);
                    let store = if store_id == STORE_PROPS {
                        props_data.as_ref()
                    } else {
                        tel_data.as_ref()
                    };
                    store
                        .install_page_from_redo(tenant, PageId::new(page_no), page)
                        .unwrap();
                    dpt.mark_dirty(
                        DirtyPageKey {
                            tenant_id: tenant,
                            store_id,
                            page_no,
                        },
                        op_lsn,
                    );
                }
                let mut node = NodeRecord::new(
                    NodeId::new(1000 + writer),
                    LabelId::new(3),
                    Lsn::new(base_lsn.raw() + 4),
                );
                node.property_ref = placement.property_page;
                node.out_tel_ref = placement.out_tel_page;
                node.in_tel_ref = placement.in_tel_page;
                (node, marker, extent_allocs)
            }));
        }
        let results: Vec<_> = joins.into_iter().map(|join| join.join().unwrap()).collect();
        assert!(matches!(
            results.iter().map(|result| result.2).sum::<usize>(),
            2 | 4
        ));
        for (node, marker, _) in &results {
            assert_eq!(node.property_ref / 256, record_page / 256);
            assert_eq!(node.out_tel_ref / 256, record_page / 256);
            assert_eq!(node.in_tel_ref / 256, record_page / 256);
            assert_eq!(read_marker(&props_data, tenant, node.property_ref), *marker);
            assert_eq!(read_marker(&tel_data, tenant, node.out_tel_ref), *marker);
            assert_eq!(read_marker(&tel_data, tenant, node.in_tel_ref), *marker);
        }

        let checkpointer =
            WriteBehindCheckpointer::new(dpt.clone(), props_data.clone(), props_data.clone())
                .with_data_target(STORE_TEL, tel_data.clone())
                .with_directory_target(STORE_PROPS, props_directory.clone())
                .with_directory_target(STORE_TEL, tel_directory.clone())
                .with_doublewrite_area(Arc::new(DoublewriteArea::new(fixture.path())));
        let report = checkpointer
            .flush_pass_with_doublewrite(Lsn::new(600))
            .unwrap();
        assert_eq!(report.flushed_pages, 8);
        assert!(dpt.is_empty());

        let reopened_props_io = Arc::new(PosixPageIo::open(&props_path).unwrap());
        let reopened_tel_io = Arc::new(PosixPageIo::open(&tel_path).unwrap());
        let reopened_props_dir = Arc::new(ExtentDirectory::new(
            tenant,
            STORE_PROPS,
            reopened_props_io.clone(),
            4,
        ));
        let reopened_tel_dir = Arc::new(ExtentDirectory::new(
            tenant,
            STORE_TEL,
            reopened_tel_io.clone(),
            4,
        ));
        let reopened_props = ExtentDataPageStore::new(reopened_props_dir.clone(), 8);
        let reopened_tel = ExtentDataPageStore::new(reopened_tel_dir.clone(), 8);
        for (node, marker, _) in &results {
            assert_eq!(
                read_marker(&reopened_props, tenant, node.property_ref),
                *marker
            );
            assert_eq!(
                read_marker(&reopened_tel, tenant, node.out_tel_ref),
                *marker
            );
            assert_eq!(read_marker(&reopened_tel, tenant, node.in_tel_ref), *marker);

            for (directory, physical, page_no) in [
                (&reopened_props_dir, &reopened_props_io, node.property_ref),
                (&reopened_tel_dir, &reopened_tel_io, node.out_tel_ref),
                (&reopened_tel_dir, &reopened_tel_io, node.in_tel_ref),
            ] {
                let offset = directory.resolve_data_page(page_no).unwrap();
                let mut bytes = [0_u8; PAGE_SIZE];
                physical
                    .read_page(PageId::new(offset / PAGE_SIZE as u64), &mut bytes)
                    .unwrap();
                assert_eq!(
                    u64::from_le_bytes(
                        bytes[PageHeader::SIZE..PageHeader::SIZE + 8]
                            .try_into()
                            .unwrap()
                    ),
                    *marker
                );
            }
        }
    }
}

/// #1500 P0 — INV-S2.4 offset-derivation race, amplified RED-on-revert gate.
///
/// Pre-fix, `place()` read the durable extent mapping BEFORE taking the
/// allocator state lock. A concurrent writer's complete
/// place→commit→install→finish cycle landing inside that window removed the
/// shared pending entry, so the late writer's `or_insert_with` re-ran against
/// its STALE `None` mapping and derived a SECOND `fetch_add` physical offset
/// for the SAME logical extent. The conflicting `ExtentAlloc` rode the late
/// writer's already-durable commit bundle: live apply failed with
/// `WalCorruption` ("ExtentAlloc conflicts with generation 1 mapping …") and
/// every future recovery replay of that WAL fails the same way — a POISONED
/// STORE, exactly the outcome INV-S2.4's "any outcome correctness-neutral"
/// forbids (INV-S2.2's conflict corruption exists to catch genuinely
/// inconsistent durable state, not as a routine result of a healthy race).
///
/// Observed 1/30 runs of the ratified two-writer gate in plain CI isolation
/// (`WalCorruption { lsn: 500, reason: "ExtentAlloc conflicts with generation
/// 1 mapping ExtentAllocation { logical_extent: 9, physical_offset:
/// 144998400, pairing: 9 }" }`). This gate races a FRESH logical extent per
/// iteration so a revert goes RED with near-certainty instead of 1-in-30.
#[test]
fn affinity_place_install_race_never_derives_conflicting_offsets() {
    let tenant = TenantId::new(41);
    let fixture = tempfile::tempdir().unwrap();
    let props_physical = Arc::new(PosixPageIo::create(fixture.path().join("props.store")).unwrap());
    let tel_physical = Arc::new(PosixPageIo::create(fixture.path().join("tel.store")).unwrap());
    let props_directory = Arc::new(ExtentDirectory::new(tenant, STORE_PROPS, props_physical, 8));
    let tel_directory = Arc::new(ExtentDirectory::new(tenant, STORE_TEL, tel_physical, 8));
    // 64 data frames: 16 installers pin data pages CONCURRENTLY (installs run
    // outside the allocator state lock); the ratified 2-writer gate's 8-frame
    // pool exhausts under this writer count.
    let props_data = Arc::new(ExtentDataPageStore::new(props_directory.clone(), 64));
    let tel_data = Arc::new(ExtentDataPageStore::new(tel_directory.clone(), 64));
    let dpt = Arc::new(DirtyPageTable::new());
    let pairer = Arc::new(
        PairedAffinityAllocator::new_recovered(props_data.clone(), tel_data.clone(), dpt.clone())
            .unwrap(),
    );
    // 16 oversubscribed writers per fresh extent: the pre-fix window needs a
    // winner's WHOLE place→install→finish cycle inside a loser's
    // [mapping-read → state-lock] gap, which two matched threads rarely open
    // (the fix5a lesson: thread count ≫ cores is the amplifier).
    for iteration in 0..1_500_u64 {
        let record_page = iteration * 256 + 7;
        let barrier = Arc::new(Barrier::new(16));
        let joins: Vec<_> = (0..16_u64)
            .map(|writer| {
                let pairer = pairer.clone();
                let barrier = barrier.clone();
                thread::spawn(move || -> arcgraph_core::Result<()> {
                    barrier.wait();
                    let base_lsn = Lsn::new(1_000 + iteration * 100 + writer * 10);
                    let placement = pairer.place(record_page, base_lsn)?;
                    let commit_lsn = placement.last_op_lsn();
                    placement.install_committed(commit_lsn)?;
                    Ok(())
                })
            })
            .collect();
        for join in joins {
            join.join().unwrap().unwrap_or_else(|error| {
                panic!(
                    "iteration {iteration}: affinity pairing race must be \
                     correctness-neutral (INV-S2.4) — a conflicting ExtentAlloc \
                     in a committed bundle poisons recovery: {error}"
                )
            });
        }
        // Exactly one durable mapping per store, at the dense next offset.
        for directory in [&props_directory, &tel_directory] {
            let mapping = directory.mapping(iteration).unwrap().unwrap_or_else(|| {
                panic!("iteration {iteration}: racing writers left no durable mapping")
            });
            assert_eq!(
                mapping.physical_offset,
                DIRECTORY_HEAD_BYTES + iteration * EXTENT_BYTES,
                "iteration {iteration}: mapping must be the dense next offset",
            );
        }
    }
}

#[test]
fn extent_directory_no_resident_owner_scales_with_extents() {
    const EXTENTS: u64 = 1_000;
    for tenant in [TenantId::new(41), TenantId::new(73)] {
        assert_ne!(tenant, TenantId::DEFAULT);
        let fixture = tempfile::tempdir().unwrap();
        let store_path = fixture.path().join("nodes.store");
        let physical = Arc::new(PosixPageIo::create(&store_path).unwrap());
        let directory = Arc::new(ExtentDirectory::new(tenant, STORE_RECORD, physical, 4));
        let dpt = Arc::new(DirtyPageTable::new());
        for logical_extent in 0..EXTENTS {
            let allocation = ExtentAllocation {
                logical_extent,
                physical_offset: DIRECTORY_HEAD_BYTES + logical_extent * EXTENT_BYTES,
                pairing: logical_extent as u32,
            };
            let lsn = Lsn::new(logical_extent + 1);
            let op = DeltaIntent::extent_alloc(STORE_RECORD, tenant, allocation)
                .assign(lsn, lsn)
                .unwrap();
            assert_eq!(
                directory.apply_extent_alloc(&op, &dpt).unwrap(),
                ExtentApplyOutcome::Applied
            );
        }
        let census = directory.residency_census();
        assert_eq!(census.resident_extent_owners, 0);
        assert_eq!(census.hot_pages, 3);
        assert!(census.hot_pages < EXTENTS as usize);
        assert_eq!(dpt.len(), 3);

        let inert = Arc::new(ExtentDirectory::new(
            tenant,
            STORE_RECORD + 100,
            Arc::new(arcgraph_storage::io::InMemoryPageIo::new()),
            2,
        ));
        let checkpointer = WriteBehindCheckpointer::new(dpt.clone(), inert.clone(), inert)
            .with_directory_target(STORE_RECORD, directory.clone())
            .with_doublewrite_area(Arc::new(DoublewriteArea::new(fixture.path())));
        assert_eq!(
            checkpointer
                .flush_pass_with_doublewrite(Lsn::new(EXTENTS + 1))
                .unwrap()
                .flushed_pages,
            3
        );
        assert!(dpt.is_empty());

        let reopened = ExtentDirectory::new(
            tenant,
            STORE_RECORD,
            Arc::new(PosixPageIo::open(&store_path).unwrap()),
            2,
        );
        for logical_extent in [0, 338, 339, EXTENTS - 1] {
            assert_eq!(
                reopened.mapping(logical_extent).unwrap(),
                Some(ExtentAllocation {
                    logical_extent,
                    physical_offset: DIRECTORY_HEAD_BYTES + logical_extent * EXTENT_BYTES,
                    pairing: logical_extent as u32,
                })
            );
        }
        assert_eq!(reopened.residency_census().resident_extent_owners, 0);
        assert!(reopened.residency_census().hot_pages <= 2);
    }
}

/// FIX-2 (fable Slice-2 required gate): empty-slot ExtentAlloc on a shared
/// directory page whose recLSN was bumped by a neighbor must (a) install
/// (entry-presence idempotence) and (b) LOWER the DPT recLSN to cover the earlier
/// op — else checkpoint→crash→redo loses it (silent data loss).
#[test]
fn extent_alloc_empty_slot_installs_and_lowers_reclsn_under_shared_page() {
    for tenant in [TenantId::new(41), TenantId::new(73)] {
        assert_ne!(tenant, TenantId::DEFAULT);
        let fixture = tempfile::tempdir().unwrap();
        let store_path = fixture.path().join("nodes.store");
        let physical = Arc::new(PosixPageIo::create(store_path).unwrap());
        let directory = Arc::new(ExtentDirectory::new(tenant, STORE_RECORD, physical, 8));
        let dpt = Arc::new(DirtyPageTable::new());

        let neighbor = ExtentAllocation {
            logical_extent: 7,
            physical_offset: DIRECTORY_HEAD_BYTES,
            pairing: 99,
        };
        let earlier = ExtentAllocation {
            logical_extent: 3,
            physical_offset: DIRECTORY_HEAD_BYTES + EXTENT_BYTES,
            pairing: 42,
        };
        let mut seq = 0u64;
        let mut apply = |alloc: ExtentAllocation, lsn: Lsn| {
            seq += 1;
            let wal_dir = fixture.path().join(format!("wal_{seq}"));
            std::fs::create_dir_all(&wal_dir).unwrap();
            write_real_wal(&wal_dir, tenant, extent_bundle(tenant, alloc, lsn), lsn);
            let records: Vec<_> = WalRecoveryReader::open(&wal_dir)
                .unwrap()
                .collect::<arcgraph_core::Result<Vec<_>>>()
                .unwrap();
            let decoded =
                decode_commit_bundle_v9(&records[0].payload, records[0].tenant_id).unwrap();
            directory.apply_extent_alloc(&decoded.deltas[0], &dpt)
        };

        assert_eq!(
            apply(neighbor, Lsn::new(200)).unwrap(),
            ExtentApplyOutcome::Applied
        );
        assert_eq!(
            apply(earlier, Lsn::new(100)).unwrap(),
            ExtentApplyOutcome::Applied,
            "empty-slot install below shared-page recLSN must not brick",
        );
        assert_eq!(directory.mapping(3).unwrap(), Some(earlier));
        assert_eq!(directory.mapping(7).unwrap(), Some(neighbor));

        // (b) recLSN lowered to <=100 so a checkpoint covers the earliest change.
        let snaps = dpt.snapshot();
        let shared = snaps
            .iter()
            .find(|s| s.key.tenant_id == tenant)
            .expect("shared directory page dirty entry");
        assert!(
            shared.rec_lsn.raw() <= 100,
            "recLSN must be lowered to cover op-100 (got {}) — else redo hole",
            shared.rec_lsn.raw(),
        );
        // Redo idempotence: re-applying the already-installed op is an entry-presence skip.
        assert_eq!(
            apply(earlier, Lsn::new(100)).unwrap(),
            ExtentApplyOutcome::Idempotent
        );
    }
}

/// ADR-230-amendment-05 boundary gate: the extent census is capped at
/// MAX_EXTENTS_PER_STORE (head-sizing basis, ext4-safe). The last in-range
/// extent resolves; exceeding the cap is a LOUD Err (never wrap/silently grow).
#[allow(clippy::assertions_on_constants)] // intentional compile-time invariant guards (head/ceiling)
#[test]
fn extent_cap_edge_boundary_and_data_ceiling_under_ext4() {
    use arcgraph_storage::extent::{DIRECTORY_HEAD_BYTES, EXTENT_BYTES, MAX_EXTENTS_PER_STORE};
    // Data ceiling stays well under ext4's 16 TiB file cap.
    let data_ceiling = DIRECTORY_HEAD_BYTES + MAX_EXTENTS_PER_STORE * EXTENT_BYTES;
    const EXT4_FILE_CAP: u64 = 16 * (1 << 40); // 16 TiB
    assert!(
        data_ceiling < EXT4_FILE_CAP,
        "store data ceiling {data_ceiling} must stay under the ext4 16 TiB file cap",
    );
    // Head stays small (~138 MiB), not the id-space ~9.5 PiB.
    assert!(
        DIRECTORY_HEAD_BYTES < (1 << 30),
        "head must be well under 1 GiB"
    );

    let tenant = TenantId::new(41);
    let fixture = tempfile::tempdir().unwrap();
    let physical = Arc::new(PosixPageIo::create(fixture.path().join("nodes.store")).unwrap());
    let directory = Arc::new(ExtentDirectory::new(tenant, STORE_RECORD, physical, 8));
    let dpt = Arc::new(DirtyPageTable::new());

    // Last in-range extent (cap-1) resolves; the cap itself is a LOUD Err.
    let last = ExtentAllocation {
        logical_extent: MAX_EXTENTS_PER_STORE - 1,
        physical_offset: DIRECTORY_HEAD_BYTES,
        pairing: 1,
    };
    let over = ExtentAllocation {
        logical_extent: MAX_EXTENTS_PER_STORE,
        physical_offset: DIRECTORY_HEAD_BYTES + EXTENT_BYTES,
        pairing: 1,
    };
    let apply = |alloc: ExtentAllocation, lsn: Lsn| {
        let wal_dir = fixture.path().join(format!("wal_{}", alloc.logical_extent));
        std::fs::create_dir_all(&wal_dir).unwrap();
        write_real_wal(&wal_dir, tenant, extent_bundle(tenant, alloc, lsn), lsn);
        let records: Vec<_> = WalRecoveryReader::open(&wal_dir)
            .unwrap()
            .collect::<arcgraph_core::Result<Vec<_>>>()
            .unwrap();
        let decoded = decode_commit_bundle_v9(&records[0].payload, records[0].tenant_id).unwrap();
        directory.apply_extent_alloc(&decoded.deltas[0], &dpt)
    };
    assert_eq!(
        apply(last, Lsn::new(100)).unwrap(),
        ExtentApplyOutcome::Applied
    );
    assert!(
        apply(over, Lsn::new(101)).is_err(),
        "logical_extent == MAX_EXTENTS_PER_STORE must be a LOUD Err, not wrap/grow",
    );
}
