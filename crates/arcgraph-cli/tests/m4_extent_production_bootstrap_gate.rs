//! M4 Slice-2 production-bootstrap wiring gates.
//!
//! These tests deliberately use the public `bootstrap_storage_backend`
//! surface. They never pre-register an extent directory or replay target.

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arcgraph_cli::bootstrap::{BootstrapMode, bootstrap_storage_backend};
use arcgraph_cli::data_dir_migration::{current_generation, upgrade_data_dir};
use arcgraph_core::record::{NodeRecord, PageHeader};
use arcgraph_core::{LabelId, Lsn, NodeId, PAGE_SIZE, PageId, PageType, TenantId};
use arcgraph_storage::extent::{
    DIRECTORY_HEAD_BYTES, EXTENT_BYTES, ExtentAllocation, ExtentDirectory,
    production_extent_store_path,
};
use arcgraph_storage::io::{PageIo, PosixPageIo};
use arcgraph_storage::primary_index::RecordKind;
use arcgraph_storage::records::{NODE_CAPACITY, SlotId, SlottedPageRef};
use arcgraph_storage::redo::DeltaPageStore;
use arcgraph_storage::wal::{
    DeltaIntent, DeltaOp, DeltaOpKind, STORE_PROPS, STORE_RECORD, STORE_TEL, WalRecordType,
    encode_commit_bundle_v9,
};
use bytes::Bytes;
use tempfile::tempdir;

const CHILD_ENV: &str = "ARCGRAPH_M4_EXTENT_BOOTSTRAP_CHILD";
const AFFINITY_CHILD_ENV: &str = "ARCGRAPH_M4_AFFINITY_ABORT_CHILD";
const ROOT_ENV: &str = "ARCGRAPH_M4_EXTENT_BOOTSTRAP_ROOT";
const READY: &str = "M4_EXTENT_WAL_DURABLE";
const AFFINITY_READY: &str = "M4_AFFINITY_WAL_DURABLE";
const AFFINITY_REFS: &str = "M4_AFFINITY_REFS";
const TENANT: TenantId = TenantId::new(41);
const LOGICAL_EXTENT: u64 = 2;
const WAIT: Duration = Duration::from_secs(20);

fn synced_marker(path: &Path) {
    let mut file = File::create(path).expect("create marker");
    file.write_all(b"ready").expect("write marker");
    file.sync_all().expect("sync marker");
}

fn wait_for(path: &Path) {
    let deadline = Instant::now() + WAIT;
    while !path.exists() {
        assert!(Instant::now() < deadline, "timed out waiting for child");
        std::thread::sleep(Duration::from_millis(2));
    }
}

fn crash_bundle(commit_lsn: Lsn) -> (ExtentAllocation, NodeRecord, Vec<DeltaOp>, Vec<u8>) {
    let page_no = LOGICAL_EXTENT * 256;
    let id = page_no * u64::from(NODE_CAPACITY) + 1;
    let node = NodeRecord::new(NodeId::new(id), LabelId::new(7), commit_lsn);
    let allocation = ExtentAllocation {
        logical_extent: LOGICAL_EXTENT,
        physical_offset: DIRECTORY_HEAD_BYTES,
        pairing: 0x41,
    };
    let first = commit_lsn.raw() - 2;
    let deltas = vec![
        DeltaIntent::extent_alloc(STORE_RECORD, TENANT, allocation)
            .assign(Lsn::new(first), commit_lsn)
            .unwrap(),
        DeltaIntent::page_alloc(STORE_RECORD, TENANT, page_no, PageType::Node, 1)
            .assign(Lsn::new(first + 1), commit_lsn)
            .unwrap(),
        DeltaOp::new(
            DeltaOpKind::PutRecord,
            STORE_RECORD,
            TENANT,
            page_no,
            1,
            commit_lsn,
            Bytes::copy_from_slice(&node.to_bytes()),
        )
        .unwrap(),
    ];
    let payload = encode_commit_bundle_v9(
        commit_lsn,
        TENANT,
        &HashMap::new(),
        &[],
        &deltas,
        &[],
        &[],
        &[],
        &[],
        &[],
    )
    .unwrap();
    (allocation, node, deltas, payload)
}

fn crash_intents() -> (ExtentAllocation, NodeRecord, Vec<DeltaIntent>) {
    let page_no = LOGICAL_EXTENT * 256;
    let id = page_no * u64::from(NODE_CAPACITY) + 1;
    let node = NodeRecord::new(NodeId::new(id), LabelId::new(7), Lsn::ZERO);
    let allocation = ExtentAllocation {
        logical_extent: LOGICAL_EXTENT,
        physical_offset: DIRECTORY_HEAD_BYTES,
        pairing: 0x41,
    };
    let intents = vec![
        DeltaIntent::extent_alloc(STORE_RECORD, TENANT, allocation),
        DeltaIntent::page_alloc(STORE_RECORD, TENANT, page_no, PageType::Node, 1),
        DeltaIntent {
            kind: DeltaOpKind::PutRecord,
            store_id: STORE_RECORD,
            tenant_id: TENANT,
            page_no,
            slot: 1,
            payload: Bytes::copy_from_slice(&node.to_bytes()),
        },
    ];
    (allocation, node, intents)
}

fn wrong_marker_page(page_no: u64, lsn: Lsn) -> [u8; PAGE_SIZE] {
    let mut bytes = [0_u8; PAGE_SIZE];
    bytes[PageHeader::SIZE..PageHeader::SIZE + 8]
        .copy_from_slice(&0xBAD0_D1CE_CAFE_BABEu64.to_le_bytes());
    let mut header = PageHeader::new(PageId::new(page_no), PageType::Node, TENANT);
    header.lsn = lsn.raw();
    header.checksum = crc32c::crc32c(&bytes[PageHeader::SIZE..]);
    bytes[..PageHeader::SIZE].copy_from_slice(&header.to_bytes());
    bytes
}

fn child(root: &Path) -> ! {
    let mode = BootstrapMode::Durable {
        data_dir: root.to_path_buf(),
    };
    let (backend, guard) = bootstrap_storage_backend(&mode).expect("production bootstrap child");
    let runtime = guard
        .extent_store(TENANT, STORE_RECORD)
        .expect("production bootstrap registered non-default extent owner");
    let commit_lsn = Lsn::new(backend.txn_manager().current_lsn().raw() + 3);
    let (_allocation, _node, deltas, payload) = crash_bundle(commit_lsn);
    let written = guard
        .wal_handle()
        .expect("production WAL handle")
        .append_at(
            commit_lsn,
            WalRecordType::CommitBundle,
            TENANT.raw(),
            0,
            TENANT,
            payload,
        )
        .expect("real WAL fsync");
    assert_eq!(written, commit_lsn);
    runtime.apply_extent_alloc(&deltas[0]).unwrap();
    for delta in &deltas[1..] {
        runtime.apply_data_delta(delta, commit_lsn).unwrap();
    }
    std::fs::write(
        root.join("M4_EXTENT_COMMIT_LSN"),
        commit_lsn.raw().to_string(),
    )
    .unwrap();
    synced_marker(&root.join(READY));
    loop {
        std::thread::park();
    }
}

fn affinity_abort_child(root: &Path) -> ! {
    let mode = BootstrapMode::Durable {
        data_dir: root.to_path_buf(),
    };
    let (backend, guard) =
        bootstrap_storage_backend(&mode).expect("production affinity bootstrap child");
    let props = guard
        .extent_store(TENANT, STORE_PROPS)
        .expect("production props extent owner");
    let tel = guard
        .extent_store(TENANT, STORE_TEL)
        .expect("production TEL extent owner");
    assert!(Arc::ptr_eq(
        props.dirty_page_table(),
        tel.dirty_page_table()
    ));
    let pairer = guard
        .affinity_allocator(TENANT)
        .expect("production affinity allocator");
    let record_page = 9 * 256 + 7;
    let first_op_lsn = Lsn::new(backend.txn_manager().current_lsn().raw() + 1);

    let aborted = pairer
        .place(record_page, first_op_lsn)
        .expect("provisional winner placement");
    let aborted_refs = (
        aborted.property_page,
        aborted.out_tel_page,
        aborted.in_tel_page,
    );
    assert_eq!(aborted_refs.0 % 256, 1);
    drop(aborted);
    assert!(props.dirty_page_table().is_empty());
    assert_eq!(props.directory().mapping(9).unwrap(), None);
    assert_eq!(tel.directory().mapping(9).unwrap(), None);

    let placement = pairer
        .place(record_page, first_op_lsn)
        .expect("loser reclaims aborted lane");
    let refs = (
        placement.property_page,
        placement.out_tel_page,
        placement.in_tel_page,
    );
    assert_eq!(refs, aborted_refs, "aborted lane was not reusable");
    assert_eq!(placement.extent_allocs.len(), 2);
    assert_eq!(placement.page_inits.len(), 3);
    let commit_lsn = placement.last_op_lsn();
    let ops = placement.wal_ops();
    let payload = encode_commit_bundle_v9(
        commit_lsn,
        TENANT,
        &HashMap::new(),
        &[],
        &ops,
        &[],
        &[],
        &[],
        &[],
        &[],
    )
    .unwrap();
    guard
        .wal_handle()
        .expect("production WAL handle")
        .append_at(
            commit_lsn,
            WalRecordType::CommitBundle,
            TENANT.raw(),
            0,
            TENANT,
            payload,
        )
        .expect("affinity real WAL fsync");
    let placement = placement
        .install_committed(commit_lsn)
        .expect("post-fsync affinity install");
    assert_eq!(props.dirty_page_table().len(), 5);

    // Fresh head initialization is cache-only until checkpoint. A parallel
    // production directory over the home file must still see no mapping;
    // crash recovery below must reconstruct it from the real WAL.
    for (runtime, store_id) in [(&props, STORE_PROPS), (&tel, STORE_TEL)] {
        let generation = current_generation(root).unwrap().unwrap();
        let path = production_extent_store_path(&generation, TENANT, store_id).unwrap();
        let fresh = ExtentDirectory::new(
            TENANT,
            store_id,
            Arc::new(PosixPageIo::open(&path).unwrap()),
            2,
        );
        assert_eq!(fresh.mapping(9).unwrap(), None);
        assert!(runtime.directory().mapping(9).unwrap().is_some());
    }
    std::fs::write(
        root.join(AFFINITY_REFS),
        format!(
            "{} {} {}",
            placement.property_page, placement.out_tel_page, placement.in_tel_page
        ),
    )
    .unwrap();
    synced_marker(&root.join(AFFINITY_READY));
    loop {
        std::thread::park();
    }
}

#[test]
fn m4_extent_production_bootstrap_child() {
    if std::env::var_os(CHILD_ENV).is_none() {
        return;
    }
    let root = std::env::var_os(ROOT_ENV).expect("child root");
    child(Path::new(&root));
}

#[test]
fn m4_affinity_abort_child() {
    if std::env::var_os(AFFINITY_CHILD_ENV).is_none() {
        return;
    }
    let root = std::env::var_os(ROOT_ENV).expect("child root");
    affinity_abort_child(Path::new(&root));
}

#[test]
fn committed_extent_alloc_replay_mapping_live_and_readable_group_b() {
    let tmp = tempdir().unwrap();
    let root = tmp.path().join("db");
    upgrade_data_dir(&root).expect("build v9 production generation");
    let generation = current_generation(&root).unwrap().unwrap();
    std::fs::create_dir_all(
        generation
            .join(arcgraph_storage::m3_migration::M3_TENANTS_DIR)
            .join(TENANT.raw().to_string()),
    )
    .unwrap();

    let mut child = Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg("m4_extent_production_bootstrap_child")
        .arg("--nocapture")
        .env(CHILD_ENV, "1")
        .env(ROOT_ENV, &root)
        .spawn()
        .expect("spawn production writer");
    wait_for(&root.join(READY));
    child.kill().expect("inject pre-checkpoint crash");
    let status = child.wait().expect("reap crash child");
    assert!(!status.success(), "crash child exited cleanly");

    let mode = BootstrapMode::Durable {
        data_dir: root.clone(),
    };
    let (_backend, guard) =
        bootstrap_storage_backend(&mode).expect("production bootstrap recovery");
    let runtime = guard
        .extent_store(TENANT, STORE_RECORD)
        .expect("production reopen retained extent owner");
    let commit_lsn = Lsn::new(
        std::fs::read_to_string(root.join("M4_EXTENT_COMMIT_LSN"))
            .unwrap()
            .parse()
            .unwrap(),
    );
    let (allocation, node, _, _) = crash_bundle(commit_lsn);
    assert_eq!(
        runtime.directory().mapping(LOGICAL_EXTENT).unwrap(),
        Some(allocation)
    );
    let (page_no, slot) = RecordKind::Node.address(node.id).unwrap();
    assert_eq!(
        runtime.directory().resolve_data_page(page_no).unwrap(),
        allocation.physical_offset
    );
    let bytes = runtime
        .data()
        .read_page_for_redo(TENANT, PageId::new(page_no))
        .unwrap()
        .unwrap();
    let page = SlottedPageRef::open(bytes.as_ref()).unwrap();
    assert_eq!(
        page.read_node(SlotId(slot)).unwrap().unwrap().to_bytes(),
        node.to_bytes()
    );
}

#[test]
fn production_bootstrap_restores_extent_data_dwb_through_directory() {
    let tmp = tempdir().unwrap();
    let root = tmp.path().join("db");
    upgrade_data_dir(&root).expect("build v9 production generation");
    let generation = current_generation(&root).unwrap().unwrap();
    std::fs::create_dir_all(
        generation
            .join(arcgraph_storage::m3_migration::M3_TENANTS_DIR)
            .join(TENANT.raw().to_string()),
    )
    .unwrap();

    let mode = BootstrapMode::Durable {
        data_dir: root.clone(),
    };
    let (backend, guard) =
        bootstrap_storage_backend(&mode).expect("production DWB writer bootstrap");
    let runtime = guard
        .extent_store(TENANT, STORE_RECORD)
        .expect("production record extent owner");
    let (allocation, node_template, intents) = crash_intents();
    let commit_intents = intents.clone();
    let commit_lsn = backend
        .txn_manager()
        .begin(TENANT)
        .commit_with_bundle(move |_, _, _, _, mutation_log| {
            mutation_log.delta_intents.extend(commit_intents);
            Ok(Vec::new())
        })
        .expect("production v9 commit + real WAL fsync");
    let first_op_lsn = commit_lsn.raw() - intents.len() as u64 + 1;
    let ops = intents
        .into_iter()
        .enumerate()
        .map(|(index, intent)| {
            intent
                .assign(Lsn::new(first_op_lsn + index as u64), commit_lsn)
                .unwrap()
        })
        .collect::<Vec<_>>();
    runtime.apply_extent_alloc(&ops[0]).unwrap();
    for op in &ops[1..] {
        runtime.apply_data_delta(op, commit_lsn).unwrap();
    }
    assert_eq!(
        guard
            .checkpointer()
            .expect("production checkpointer")
            .checkpoint()
            .expect("directory-first production checkpoint"),
        commit_lsn
    );
    drop(backend);
    drop(guard);

    let page_no = LOGICAL_EXTENT * 256;
    let store_path = production_extent_store_path(&generation, TENANT, STORE_RECORD).unwrap();
    let physical = PosixPageIo::open(&store_path).unwrap();
    // Fault injection after the DWB fsync: the mapped home is torn. A valid,
    // newer wrong marker at physical PageId(logical_page) makes the M3
    // logical-as-physical bypass skip restoration at precisely the wrong
    // location, so the negative control cannot pass accidentally.
    physical
        .write_page(
            PageId::new(page_no),
            &wrong_marker_page(page_no, Lsn::new(commit_lsn.raw() + 100)),
        )
        .unwrap();
    physical
        .write_page(
            PageId::new(allocation.physical_offset / PAGE_SIZE as u64),
            &[0_u8; PAGE_SIZE],
        )
        .unwrap();
    physical.flush().unwrap();

    let (_backend, guard) =
        bootstrap_storage_backend(&mode).expect("production DWB recovery bootstrap");
    let runtime = guard
        .extent_store(TENANT, STORE_RECORD)
        .expect("production reopen retained record extent owner");
    assert_eq!(
        runtime.directory().mapping(LOGICAL_EXTENT).unwrap(),
        Some(allocation)
    );
    let mut expected = node_template;
    expected.created_lsn = commit_lsn.raw();
    let (resolved_page, slot) = RecordKind::Node.address(expected.id).unwrap();
    let bytes = runtime
        .data()
        .read_page_for_redo(TENANT, PageId::new(resolved_page))
        .unwrap()
        .unwrap();
    let page = SlottedPageRef::open(bytes.as_ref()).unwrap();
    assert_eq!(
        page.read_node(SlotId(slot)).unwrap().unwrap().to_bytes(),
        expected.to_bytes()
    );
}

#[test]
fn allocator_counter_recovery_does_not_rehand_replayed_unflushed_offsets() {
    let tmp = tempdir().unwrap();
    let root = tmp.path().join("db");
    upgrade_data_dir(&root).expect("build v9 production generation");
    let generation = current_generation(&root).unwrap().unwrap();
    std::fs::create_dir_all(
        generation
            .join(arcgraph_storage::m3_migration::M3_TENANTS_DIR)
            .join(TENANT.raw().to_string()),
    )
    .unwrap();

    let mut child = Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg("m4_affinity_abort_child")
        .arg("--nocapture")
        .env(AFFINITY_CHILD_ENV, "1")
        .env(ROOT_ENV, &root)
        .spawn()
        .expect("spawn production affinity writer");
    wait_for(&root.join(AFFINITY_READY));
    child.kill().expect("inject pre-checkpoint affinity crash");
    let status = child.wait().expect("reap affinity crash child");
    assert!(!status.success(), "affinity crash child exited cleanly");

    let refs = std::fs::read_to_string(root.join(AFFINITY_REFS)).unwrap();
    let refs = refs
        .split_whitespace()
        .map(|raw| raw.parse::<u64>().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(refs, vec![9 * 256 + 1, 9 * 256 + 2, 9 * 256 + 3]);

    let mode = BootstrapMode::Durable {
        data_dir: root.clone(),
    };
    let (_backend, guard) =
        bootstrap_storage_backend(&mode).expect("production affinity recovery bootstrap");
    let props = guard.extent_store(TENANT, STORE_PROPS).unwrap();
    let tel = guard.extent_store(TENANT, STORE_TEL).unwrap();
    assert_eq!(
        props.directory().mapping(9).unwrap(),
        Some(ExtentAllocation {
            logical_extent: 9,
            physical_offset: DIRECTORY_HEAD_BYTES,
            pairing: 9,
        })
    );
    assert_eq!(
        tel.directory().mapping(9).unwrap(),
        Some(ExtentAllocation {
            logical_extent: 9,
            physical_offset: DIRECTORY_HEAD_BYTES,
            pairing: 9,
        })
    );
    for (runtime, page_no, page_type) in [
        (&props, refs[0], PageType::PropSlotted),
        (&tel, refs[1], PageType::Tel),
        (&tel, refs[2], PageType::Tel),
    ] {
        let bytes = runtime
            .data()
            .read_page_for_redo(TENANT, PageId::new(page_no))
            .unwrap()
            .unwrap();
        let header_bytes: &[u8; PageHeader::SIZE] = bytes[..PageHeader::SIZE].try_into().unwrap();
        let header = PageHeader::from_bytes(header_bytes).unwrap();
        assert_eq!(header.page_id, page_no);
        assert_eq!(header.tenant_id, TENANT.raw());
        assert_eq!(header.page_type, page_type.as_byte());
        assert_eq!(crc32c::crc32c(&bytes[PageHeader::SIZE..]), header.checksum);
    }

    let pairer = guard
        .affinity_allocator(TENANT)
        .expect("recovered production affinity allocator");
    let next = pairer
        .place(10 * 256 + 7, Lsn::new(10_000))
        .expect("allocate after replay");
    let next_offsets: HashMap<_, _> = next
        .extent_allocs
        .iter()
        .map(|op| {
            (
                op.store_id,
                ExtentAllocation::decode(&op.payload, op.op_lsn)
                    .unwrap()
                    .physical_offset,
            )
        })
        .collect();
    assert_eq!(
        next_offsets[&STORE_PROPS],
        DIRECTORY_HEAD_BYTES + EXTENT_BYTES,
        "replayed props offset was handed out twice"
    );
    assert_eq!(
        next_offsets[&STORE_TEL],
        DIRECTORY_HEAD_BYTES + EXTENT_BYTES,
        "replayed TEL offset was handed out twice"
    );
}
