//! M4 Slice-3a durability gates: the v5 source generation is immutable,
//! the v6 WAL is fresh, and CURRENT/VERSION form an atomic-or-resumable swap.

use std::collections::{BTreeSet, HashMap};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use arcgraph_cli::bootstrap::{BootstrapMode, bootstrap_storage_backend};
#[cfg(feature = "fault-injection")]
use arcgraph_cli::data_dir_migration::{
    BuildLeakInjector, DurableGenerationSwap, GenerationCleanupFault, GenerationPinRegistry,
    INDEX_VECTOR_CENSUS_FILE, MigrationBuildRendezvous, PinAcquisitionRendezvous,
    cleanup_old_generation_after_drain, naive_unrevalidated_pin_for_red_control,
    pin_current_generation, upgrade_data_dir,
};
use arcgraph_cli::data_dir_migration::{
    MigrationFault, MigrationOutcome, current_generation, upgrade_quiesced_v5_to_v6,
};
#[cfg(all(feature = "fault-injection", not(debug_assertions)))]
use arcgraph_cli::data_dir_migration::{TenantLoadLeg, tenant_load_wal_dir};
use arcgraph_core::record::{NodeRecord, RelRecord};
use arcgraph_core::{
    LabelId, Lsn, NodeId, PAGE_SIZE, PageHeader, PageId, PageType, RelId, TenantId, TypeId,
};
use arcgraph_mcp::storage::{
    StorageBackend, StorageHybridSearcher, StorageIngestProvider, StorageNeighborhoodExplorer,
    StorageNodeInspector, StorageRawQueryExecutor, StorageSchemaProvider,
};
use arcgraph_mcp::tools::explore::Neighborhood;
use arcgraph_mcp::transport::handle_raw_envelope;
use arcgraph_mcp::{Dispatcher, SessionScope};
use arcgraph_storage::address::{AddressError, MAX_ID};
use arcgraph_storage::blob::BlobStore;
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::checkpoint::{
    CheckpointSidecar, CheckpointSnapshot, incremental_checkpoint, incremental_metadata_path,
    write_sidecar_atomic,
};
use arcgraph_storage::crud::{
    CrudStore, PropertyData, commit, create_node, crud_allocator_seed_handle,
    delete_node_with_store, read_node_with_store,
};
use arcgraph_storage::extent::{
    DIRECTORY_HEAD_BYTES, EXTENT_BYTES, EXTENT_PAGES, ExtentAllocation, ExtentApplyOutcome,
    ExtentDataPageStore, ExtentDirectory, MAX_EXTENTS_PER_STORE, PairedAffinityAllocator,
    production_extent_store_path,
};
use arcgraph_storage::idempotency::IdempotencyStore;
use arcgraph_storage::intern::{InternTable, intern_logged};
use arcgraph_storage::io::{InMemoryPageIo, PosixPageIo};
use arcgraph_storage::m4_migration::{
    LoaderMigrationFrontier, LoaderTarget, M4_EXTENT_STORE_IDS, load_v5_generation,
    load_v6_physical_base, loader_record_address, read_extent_ledger, recover_next_physical_offset,
};
use arcgraph_storage::manifest::{DATA_DIR_VERSION_M4, DataDirManifest, RECORD_FORMAT_DIRECT_M4};
use arcgraph_storage::page_alloc::PageAllocator;
#[cfg(all(feature = "fault-injection", unix))]
use arcgraph_storage::page_store::{
    BufferedRecordPageStore, PageStoreIdentity, PerTenantBufferPool, PerTenantBufferPoolConfig,
    TenantFilePageIo, TenantPageIo,
};
use arcgraph_storage::permissions::PermissionIndex;
use arcgraph_storage::primary_index::{PrimaryPageStore, RecordKind};
use arcgraph_storage::prop_block::{PropBlockBuilder, PropBlockView, PropValue, TypedBagParts};
use arcgraph_storage::property::BlobRef;
use arcgraph_storage::record_store::RecordPageStore;
use arcgraph_storage::records::{
    NODE_CAPACITY, PROP_BAG_MAX_BYTES, REL_CAPACITY, SlotId, SlottedPage, SlottedPageRef,
};
use arcgraph_storage::redo::DirtyPageTable;
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{
    AllocatorAdvance, AllocatorKind, BUNDLE_FORMAT_V9, BUNDLE_FORMAT_V10, DeltaIntent, DeltaOpKind,
    STORE_NODE_BINDINGS, STORE_PROPS, STORE_RECORD, STORE_RELS, STORE_TEL, SegmentHeader,
    WalRecord, WalRecordType, WalRecoveryReader, decode_commit_bundle_v10, encode_commit_bundle_v9,
    encode_commit_bundle_v10, segment_filename,
};
#[cfg(all(feature = "fault-injection", not(debug_assertions)))]
use arcgraph_storage::wal::{WalConfig, WalWriter};
use arcgraph_storage::{
    AddressedRecordStore, AddressedStoreError, BlobPageFlushTarget, DoublewriteArea,
    OWNER_ROWS_PER_PAGE, OwnerRow, OwnerRowClass, PageFlushTarget, WriteBehindCheckpointer,
};
use bytes::Bytes;
use tempfile::tempdir;

const BIN: &str = env!("CARGO_BIN_EXE_arcgraph");
const MIGRATION_LSN: Lsn = Lsn::new(100);
const TENANTS: [TenantId; 2] = [TenantId::new(41), TenantId::new(73)];
const CHILD_ROOT: &str = "ARCGRAPH_M4_MIGRATION_KILL_ROOT";
const CHILD_FAULT: &str = "ARCGRAPH_M4_MIGRATION_KILL_FAULT";
#[cfg(feature = "fault-injection")]
const ATTACH_COMPLETE_ROOT: &str = "ARCGRAPH_M5_ATTACH_COMPLETE_ROOT";
#[cfg(feature = "fault-injection")]
const OLD_GENERATION_CLEANUP_ROOT: &str = "ARCGRAPH_M5_OLD_GENERATION_CLEANUP_ROOT";
#[cfg(feature = "fault-injection")]
const PRODUCTION_CLEANUP_CRASH: &str = "ARCGRAPH_M5_PRODUCTION_CLEANUP_CRASH";
#[cfg(all(feature = "fault-injection", not(debug_assertions)))]
const LEDGER_PROOF_ROOT: &str = "ARCGRAPH_M5_LEDGER_PROOF_ROOT";
#[cfg(all(feature = "fault-injection", not(debug_assertions)))]
const PRODUCTION_MISSING_LEDGER: &str = "ARCGRAPH_M5_PRODUCTION_MISSING_LEDGER";
#[cfg(all(feature = "fault-injection", not(debug_assertions), unix))]
const WAL_EXPOSURE_ROOT: &str = "ARCGRAPH_M5_WAL_EXPOSURE_ROOT";
#[cfg(all(feature = "fault-injection", unix))]
const TENANT_IDENTITY_ROOT: &str = "ARCGRAPH_M5_TENANT_IDENTITY_ROOT";
#[cfg(all(feature = "fault-injection", unix))]
const INDEX_VECTOR_ROOT: &str = "ARCGRAPH_M5_INDEX_VECTOR_ROOT";
const PROP_SIGKILL_CHILD: &str = "ARCGRAPH_M4_PROP_SIGKILL_CHILD";
const PROP_SIGKILL_ROOT: &str = "ARCGRAPH_M4_PROP_SIGKILL_ROOT";
const PROP_SIGKILL_BATCH: u64 = 384;
const OWNER_PROVENANCE_ROOT: &str = "ARCGRAPH_M4_OWNER_PROVENANCE_ROOT";
const OWNER_PHASE1_READY: &str = "OWNER_PHASE1_READY";
const OWNER_ACKED: &str = "OWNER_ACKED";
const OWNER_UNACKED_ID: u64 = 25;

fn node(id: u64, label: u32, lsn: u64) -> NodeRecord {
    NodeRecord::new(NodeId::new(id), LabelId::new(label), Lsn::new(lsn))
}

fn write_page(path: &Path, page_no: u64, bytes: &[u8; PAGE_SIZE]) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)
        .unwrap();
    file.seek(SeekFrom::Start(page_no * PAGE_SIZE as u64))
        .unwrap();
    file.write_all(bytes).unwrap();
    file.sync_all().unwrap();
}

fn record_pages(tenant: TenantId) -> ([u8; PAGE_SIZE], [u8; PAGE_SIZE]) {
    let high = u64::from(NODE_CAPACITY) * 256 + 1;
    let mut nodes = [0_u8; PAGE_SIZE];
    let mut header = PageHeader::new(PageId::new(7), PageType::Node, tenant);
    header.lsn = 80;
    {
        let mut page = SlottedPage::init(&mut nodes, header).unwrap();
        page.insert_node(&node(1, 11, 40)).unwrap();
        page.insert_node(&node(2, 12, 41)).unwrap();
        page.insert_node(&node(3, 13, 42)).unwrap();
        page.insert_node(&node(high, 14, 43)).unwrap();
        page.tombstone(SlotId(1)).unwrap();
    }

    let mut rels = [0_u8; PAGE_SIZE];
    let mut header = PageHeader::new(PageId::new(8), PageType::Rel, tenant);
    header.lsn = 81;
    {
        let mut page = SlottedPage::init(&mut rels, header).unwrap();
        page.insert_rel(&RelRecord::new(
            RelId::new(1),
            TypeId::new(9),
            NodeId::new(1),
            NodeId::new(3),
            Lsn::new(44),
        ))
        .unwrap();
        let mut deleted = RelRecord::new(
            RelId::new(2),
            TypeId::new(9),
            NodeId::new(1),
            NodeId::new(3),
            Lsn::new(45),
        );
        deleted.expired_lsn = 60;
        page.insert_rel(&deleted).unwrap();
    }
    (nodes, rels)
}

fn prop_page(tenant: TenantId, page_no: u64) -> [u8; PAGE_SIZE] {
    let mut bytes = [0_u8; PAGE_SIZE];
    let mut header = PageHeader::new(PageId::new(page_no), PageType::PropSlotted, tenant);
    header.lsn = 70;
    SlottedPage::init(&mut bytes, header)
        .unwrap()
        .put_bag_at(SlotId(0), format!("tenant-{}", tenant.raw()).as_bytes())
        .unwrap();
    bytes
}

/// M5-D2 / INV-M5.8 fixture extension (amendment §3.3(3)): a RICHER v5
/// property page carrying many mixed-size bags, so a dropped-props /
/// empty-extent loader regression cannot hide behind the single-bag
/// `prop_page` above. (The fixture's relationships already make it
/// TEL-bearing input; the v5 format itself has no on-disk TEL store.)
fn rich_prop_page(tenant: TenantId, page_no: u64) -> [u8; PAGE_SIZE] {
    let mut bytes = [0_u8; PAGE_SIZE];
    let mut header = PageHeader::new(PageId::new(page_no), PageType::PropSlotted, tenant);
    header.lsn = 70;
    let mut page = SlottedPage::init(&mut bytes, header).unwrap();
    for index in 0_u64..24 {
        let mut bag = vec![u8::try_from(index).unwrap(); 16 + (index as usize * 7) % 96];
        bag.extend_from_slice(&tenant.raw().to_le_bytes());
        bag.extend_from_slice(&page_no.to_le_bytes());
        bag.extend_from_slice(&index.to_le_bytes());
        page.insert_bag(&bag).unwrap();
    }
    bytes
}

fn write_stale_v5_wal(generation: &Path) {
    let wal = generation.join("wal");
    fs::create_dir(&wal).unwrap();
    let stale = node(1, 99, 90);
    let delta = arcgraph_storage::wal::DeltaOp::new(
        DeltaOpKind::PutRecord,
        STORE_RECORD,
        TENANTS[0],
        999,
        0,
        Lsn::new(90),
        Bytes::copy_from_slice(&stale.to_bytes()),
    )
    .unwrap();
    let bundle = encode_commit_bundle_v9(
        Lsn::new(90),
        TENANTS[0],
        &HashMap::new(),
        &[],
        &[delta],
        &[],
        &[],
        &[],
        &[],
        &[],
    )
    .unwrap();
    let record = WalRecord {
        record_type: WalRecordType::CommitBundle,
        txn_id: 90,
        lsn: Lsn::new(90),
        timestamp_ms: 0,
        tenant_id: TENANTS[0],
        payload: bundle,
    };
    let mut bytes = SegmentHeader {
        format_version: BUNDLE_FORMAT_V9,
    }
    .encode()
    .to_vec();
    record.encode(&mut bytes).unwrap();
    fs::write(wal.join(segment_filename(0)), bytes).unwrap();
    File::open(wal.join(segment_filename(0)))
        .unwrap()
        .sync_all()
        .unwrap();
}

fn v5_fixture(root: &Path) -> PathBuf {
    fs::create_dir_all(root).unwrap();
    let generation = root.join("gen-v9");
    fs::create_dir(&generation).unwrap();
    fs::write(root.join("CURRENT"), b"gen-v9\n").unwrap();
    arcgraph_storage::stamp_data_dir(&generation, 5).unwrap();
    let prior = DataDirManifest::m2_typed("2026-07-13T00:00:00Z".to_owned());
    arcgraph_storage::write_data_dir_manifest(
        &generation,
        &DataDirManifest::m3_delta_from(&prior, "2026-07-13T00:00:01Z".to_owned(), Lsn::new(30)),
    )
    .unwrap();
    fs::write(generation.join("LSN_SEED"), 31_u64.to_le_bytes()).unwrap();
    fs::write(generation.join("pages.db"), vec![0_u8; PAGE_SIZE]).unwrap();
    let sidecar = CheckpointSidecar::incremental(MIGRATION_LSN, MIGRATION_LSN, 0, 1);
    write_sidecar_atomic(&generation, &sidecar).unwrap();
    let mut metadata_header = [0_u8; 48];
    metadata_header[..4].copy_from_slice(b"AGCM");
    metadata_header[4..6].copy_from_slice(
        &arcgraph_storage::checkpoint::INCREMENTAL_METADATA_FORMAT_VERSION.to_le_bytes(),
    );
    metadata_header[8..16].copy_from_slice(&MIGRATION_LSN.raw().to_le_bytes());
    metadata_header[16..24].copy_from_slice(&MIGRATION_LSN.raw().to_le_bytes());
    metadata_header[24..32].copy_from_slice(&MIGRATION_LSN.raw().to_le_bytes());
    fs::write(
        incremental_metadata_path(&generation, MIGRATION_LSN, 1),
        metadata_header,
    )
    .unwrap();

    for tenant in TENANTS {
        let path = arcgraph_storage::m3_migration::m3_record_store_path(&generation, tenant);
        let (nodes, rels) = record_pages(tenant);
        write_page(&path, 0, &nodes);
        write_page(&path, 1, &rels);
    }
    write_page(
        &generation.join(arcgraph_storage::m3_migration::M3_PROPS_STORE_FILE),
        0,
        &prop_page(TENANTS[0], 0),
    );
    write_page(
        &generation.join(arcgraph_storage::m3_migration::M3_PROPS_STORE_FILE),
        1,
        &prop_page(TENANTS[1], 1),
    );
    // M5-D2 / INV-M5.8 fixture extension: property-rich pages per tenant
    // (see `rich_prop_page`) so the loader-vs-M4-lite byte differential
    // covers multi-bag STORE_PROPS content, not just one bag per tenant.
    write_page(
        &generation.join(arcgraph_storage::m3_migration::M3_PROPS_STORE_FILE),
        2,
        &rich_prop_page(TENANTS[0], 2),
    );
    write_page(
        &generation.join(arcgraph_storage::m3_migration::M3_PROPS_STORE_FILE),
        3,
        &rich_prop_page(TENANTS[1], 3),
    );
    write_stale_v5_wal(&generation);
    generation
}

fn production_v5_fixture(root: &Path) -> PathBuf {
    let generation = v5_fixture(root);

    // Replace the compact crash-fixture header with a production-encoded v9
    // checkpoint. The real bootstrap decoder must consume this before the
    // explicit CLI can quiesce and establish its final DPT-empty checkpoint.
    let txn = Arc::new(TxnManager::new());
    txn.seed_after_replay(MIGRATION_LSN);
    let primary = Arc::new(PrimaryPageStore::new());
    let records = Arc::new(RecordPageStore::new());
    let blob = Arc::new(BlobStore::new());
    let allocator = Arc::new(PageAllocator::new());
    let crud = Arc::new(CrudStore::new_with_existing_page_stores(
        None,
        None,
        Arc::clone(&allocator),
        Arc::clone(&records),
        Arc::clone(&blob),
    ));
    let intern = Arc::new(InternTable::new());
    let idempotency = Arc::new(IdempotencyStore::new());
    let permissions = Arc::new(PermissionIndex::new());
    let seed = crud_allocator_seed_handle(crud, allocator);
    let snapshot = CheckpointSnapshot {
        txn: &txn,
        primary_pages: &primary,
        record_pages: &records,
        blob: &blob,
        allocator_seed: seed.as_ref(),
        intern: &intern,
        idempotency: &idempotency,
        permissions: &permissions,
        permissions_tenant: TENANTS[0],
    };
    let page_io = Arc::new(InMemoryPageIo::new());
    let flush_target: Arc<dyn PageFlushTarget> =
        Arc::new(BlobPageFlushTarget::new(Arc::clone(&blob), page_io.clone()));
    let write_behind = WriteBehindCheckpointer::new(
        Arc::new(DirtyPageTable::new()),
        Arc::clone(&flush_target),
        flush_target,
    )
    .with_doublewrite_area(Arc::new(DoublewriteArea::new(&generation)));
    let high_node = u64::from(NODE_CAPACITY) * 256 + 1;
    let advances: Vec<_> = TENANTS
        .into_iter()
        .flat_map(|tenant| {
            [
                AllocatorAdvance {
                    tenant,
                    kind: AllocatorKind::Node,
                    new_high_water: high_node,
                },
                AllocatorAdvance {
                    tenant,
                    kind: AllocatorKind::Rel,
                    new_high_water: 2,
                },
            ]
        })
        .collect();
    incremental_checkpoint(
        &generation,
        &BufferPool::new(1, page_io),
        &snapshot,
        &write_behind,
        || (advances, None),
        Ok,
    )
    .unwrap();
    generation
}

fn advance_with_production_incremental_checkpoint(generation: &Path, frontier: Lsn) {
    let txn = Arc::new(TxnManager::new());
    txn.seed_after_replay(frontier);
    let primary = Arc::new(PrimaryPageStore::new());
    let records = Arc::new(RecordPageStore::new());
    let blob = Arc::new(BlobStore::new());
    let allocator = Arc::new(PageAllocator::new());
    let crud = Arc::new(CrudStore::new_with_existing_page_stores(
        None,
        None,
        Arc::clone(&allocator),
        Arc::clone(&records),
        Arc::clone(&blob),
    ));
    let intern = Arc::new(InternTable::new());
    let idempotency = Arc::new(IdempotencyStore::new());
    let permissions = Arc::new(PermissionIndex::new());
    let seed = crud_allocator_seed_handle(crud, allocator);
    let snapshot = CheckpointSnapshot {
        txn: &txn,
        primary_pages: &primary,
        record_pages: &records,
        blob: &blob,
        allocator_seed: seed.as_ref(),
        intern: &intern,
        idempotency: &idempotency,
        permissions: &permissions,
        permissions_tenant: TENANTS[0],
    };
    let page_io = Arc::new(InMemoryPageIo::new());
    let flush_target: Arc<dyn PageFlushTarget> =
        Arc::new(BlobPageFlushTarget::new(Arc::clone(&blob), page_io.clone()));
    let write_behind = WriteBehindCheckpointer::new(
        Arc::new(DirtyPageTable::new()),
        Arc::clone(&flush_target),
        flush_target,
    )
    .with_doublewrite_area(Arc::new(DoublewriteArea::new(generation)));
    incremental_checkpoint(
        generation,
        &BufferPool::new(1, page_io),
        &snapshot,
        &write_behind,
        || (Vec::new(), None),
        Ok,
    )
    .unwrap();
}

fn tree_bytes(root: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    fn walk(base: &Path, path: &Path, out: &mut Vec<(PathBuf, Vec<u8>)>) {
        let mut entries: Vec<_> = fs::read_dir(path).unwrap().map(Result::unwrap).collect();
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            if entry.file_type().unwrap().is_dir() {
                walk(base, &entry.path(), out);
            } else {
                out.push((
                    entry.path().strip_prefix(base).unwrap().to_owned(),
                    fs::read(entry.path()).unwrap(),
                ));
            }
        }
    }
    let mut out = Vec::new();
    walk(root, root, &mut out);
    out
}

fn commit_property_node(
    backend: &StorageBackend,
    tenant_id: TenantId,
    property: &PropertyData,
) -> (NodeId, Lsn) {
    let routed = backend
        .router()
        .route(TenantId::DEFAULT, arcgraph_core::PartitionId::ZERO)
        .unwrap();
    let mut tx = backend.txn_manager().begin(tenant_id);
    let id = create_node(
        routed.crud(),
        &mut tx,
        tenant_id,
        LabelId::new(99),
        property,
    )
    .unwrap();
    let lsn = commit(tx, routed.crud()).unwrap();
    (id, lsn)
}

fn assert_node_bag(backend: &StorageBackend, tenant_id: TenantId, id: NodeId, expected: &[u8]) {
    let routed = backend
        .router()
        .route(TenantId::DEFAULT, arcgraph_core::PartitionId::ZERO)
        .unwrap();
    let reader = backend.txn_manager().begin(tenant_id);
    let record = read_node_with_store(routed.crud(), &reader, id)
        .unwrap()
        .expect("post-swap property node disappeared after restart");
    let blob_ref = BlobRef::decode(record.property_ref)
        .expect("post-swap property record lost its overflow reference");
    let bag = routed
        .crud()
        .blob_store()
        .get_bag(tenant_id, blob_ref)
        .expect("post-swap property payload is unreadable after restart");
    assert_eq!(&*bag, expected);
    reader.abort();
}

fn typed_mcp_shape() -> (PropertyData, Vec<u8>) {
    let mut builder = PropBlockBuilder::new();
    builder
        .put(11, PropValue::Str("payments-api".to_owned()))
        .put(12, PropValue::Int(503))
        .put(13, PropValue::Bool(true));
    let encoded = builder.build().unwrap();
    assert!(
        encoded.overflow_payload().is_none(),
        "MCP-default gate shape unexpectedly spilled"
    );
    let block = encoded.into_block_bytes(None).unwrap();
    (
        PropertyData::TypedBlock(TypedBagParts {
            block: block.clone(),
            overflow: None,
        }),
        block,
    )
}

fn owner_provenance_payload(tenant: TenantId, class: OwnerRowClass, id: u64) -> Vec<u8> {
    format!("owner-v9 tenant={} class={class:?} id={id}", tenant.raw()).into_bytes()
}

fn owner_id_in_extent(logical_extent: u64, slot: u64) -> u64 {
    assert!(slot < OWNER_ROWS_PER_PAGE);
    logical_extent * EXTENT_PAGES * OWNER_ROWS_PER_PAGE + slot
}

fn owner_rows_for_extents(
    tenant: TenantId,
    class: OwnerRowClass,
    logical_extents: &[u64],
) -> Vec<OwnerRow> {
    logical_extents
        .iter()
        .enumerate()
        .map(|(index, logical_extent)| {
            let id = owner_id_in_extent(*logical_extent, index as u64 + 1);
            OwnerRow::new(class, id, owner_provenance_payload(tenant, class, id)).unwrap()
        })
        .collect()
}

fn assert_owner_rows(
    registry: &arcgraph_storage::OwnerRowRegistry,
    expected: &[(TenantId, OwnerRow)],
) {
    for (tenant, row) in expected {
        assert_eq!(
            registry.read(*tenant, row.class(), row.id()).unwrap(),
            Some(row.clone()),
            "owner row {:?}/{} changed or disappeared",
            row.class(),
            row.id()
        );
    }
}

fn assert_dense_owner_mappings(
    guard: &arcgraph_cli::bootstrap::DurabilityGuard,
    tenant: TenantId,
    logical_extents: &BTreeSet<u64>,
) {
    let runtime = guard
        .extent_store(tenant, STORE_NODE_BINDINGS)
        .expect("production node-binding extent owner is absent");
    let mut physical_offsets: Vec<_> = logical_extents
        .iter()
        .map(|logical_extent| {
            runtime
                .directory()
                .mapping(*logical_extent)
                .unwrap()
                .unwrap_or_else(|| panic!("logical extent {logical_extent} is unmapped"))
                .physical_offset
        })
        .collect();
    physical_offsets.sort_unstable();
    let expected: Vec<_> = (0..logical_extents.len() as u64)
        .map(|index| DIRECTORY_HEAD_BYTES + index * EXTENT_BYTES)
        .collect();
    assert_eq!(physical_offsets, expected);
    assert_eq!(
        runtime.directory().recover_next_physical_offset().unwrap(),
        DIRECTORY_HEAD_BYTES + logical_extents.len() as u64 * EXTENT_BYTES
    );
}

fn snapshot_owner_pages(
    guard: &arcgraph_cli::bootstrap::DurabilityGuard,
    rows: &[(TenantId, OwnerRow)],
) -> Vec<((u64, u16, u64), Vec<u8>)> {
    let targets: BTreeSet<_> = rows
        .iter()
        .map(|(tenant, row)| {
            let address = row.class().address(row.id()).unwrap();
            ((*tenant, row.class().store_id()), address.page_no)
        })
        .collect();
    targets
        .into_iter()
        .map(|((tenant, store_id), page_no)| {
            let runtime = guard
                .extent_store(tenant, store_id)
                .expect("owner replay target is not a production extent store");
            let image = PageFlushTarget::copy_page_pinned(
                runtime.data().as_ref(),
                tenant,
                PageId::new(page_no),
            )
            .unwrap()
            .expect("owner replay target page is absent");
            ((tenant.raw(), store_id, page_no), image.to_vec())
        })
        .collect()
}

fn fault_from_name(name: &str) -> MigrationFault {
    match name {
        "scratch" => MigrationFault::AfterScratchCreate,
        #[cfg(feature = "fault-injection")]
        "before-ledger" => MigrationFault::BeforeGenerationSync,
        "sync" => MigrationFault::AfterGenerationSync,
        "rename" => MigrationFault::AfterGenerationRename,
        "current" => MigrationFault::AfterCurrentSwap,
        "version-dir-fsync" => MigrationFault::VersionParentDirFsync,
        "version" => MigrationFault::AfterVersionStamp,
        _ => panic!("unknown fault {name}"),
    }
}

#[cfg(feature = "fault-injection")]
#[test]
fn m5_attach_complete_child() {
    let Ok(root) = std::env::var(ATTACH_COMPLETE_ROOT) else {
        return;
    };
    assert_eq!(
        upgrade_quiesced_v5_to_v6(Path::new(&root), MIGRATION_LSN, MigrationFault::None)
            .expect("complete attach child must commit the generation"),
        MigrationOutcome::Upgraded {
            migration_lsn: MIGRATION_LSN
        }
    );
}

#[cfg(feature = "fault-injection")]
#[test]
fn old_generation_cleanup_kill9_child() {
    let Ok(root) = std::env::var(OLD_GENERATION_CLEANUP_ROOT) else {
        return;
    };
    let root = Path::new(&root);
    upgrade_data_dir(root).expect("operator upgrade must preserve its fallback through swap");
    assert!(root.join("gen-v9").exists());
    let mode = BootstrapMode::Durable {
        data_dir: root.to_path_buf(),
    };
    let (backend, guard) = bootstrap_storage_backend(&mode).unwrap();
    drop(backend);
    // Drop establishes the later durable checkpoint, then the injected
    // production reaper fault leaves a resumable retirement tree.
    drop(guard);
    assert!(root.join(".gen-v9.cleanup").exists());
    #[cfg(unix)]
    unsafe {
        libc::kill(libc::getpid(), libc::SIGKILL);
    }
    #[cfg(not(unix))]
    std::process::abort();
}

#[test]
fn m4_kill9_child() {
    let (Ok(root), Ok(fault)) = (std::env::var(CHILD_ROOT), std::env::var(CHILD_FAULT)) else {
        return;
    };
    let error = upgrade_quiesced_v5_to_v6(Path::new(&root), MIGRATION_LSN, fault_from_name(&fault))
        .expect_err("child fault must interrupt migration");
    let error_chain = format!("{error:#}");
    if fault == "version-dir-fsync" {
        assert!(
            error_chain.contains("injected VERSION parent-directory fsync failure"),
            "unexpected VERSION fsync error chain: {error_chain}"
        );
    } else {
        assert!(error_chain.contains("injected migration crash"));
    }
    #[cfg(unix)]
    unsafe {
        libc::kill(libc::getpid(), libc::SIGKILL);
    }
    #[cfg(not(unix))]
    std::process::abort();
}

/// INV-M5.11 subprocess. The parent requires SIGKILL after real v10 WAL fsyncs
/// for both non-default existing tenants in the unpublished generation.
#[cfg(all(feature = "fault-injection", not(debug_assertions), unix))]
#[test]
fn wal_exposure_kill9_child() {
    let Ok(root) = std::env::var(WAL_EXPOSURE_ROOT) else {
        return;
    };
    let root = Path::new(&root);

    let building = root.join("gen-v10.building");
    fs::create_dir_all(&building).unwrap();
    let isolated_wal = tenant_load_wal_dir(
        root,
        TenantLoadLeg::ExistingRecluster {
            building_generation: &building,
        },
    )
    .unwrap();
    let live = current_generation(root).unwrap().unwrap().join("wal");
    assert_ne!(
        isolated_wal, live,
        "re-cluster leg must never receive the selected generation WAL"
    );
    fs::create_dir(&isolated_wal).unwrap();
    fs::write(
        isolated_wal.join(segment_filename(0)),
        SegmentHeader {
            format_version: BUNDLE_FORMAT_V10,
        }
        .encode(),
    )
    .unwrap();

    let writer = WalWriter::spawn_from(WalConfig::new(&isolated_wal), MIGRATION_LSN).unwrap();
    assert_eq!(writer.handle().format_version(), BUNDLE_FORMAT_V10);
    for (offset, tenant) in TENANTS.into_iter().enumerate() {
        let lsn = Lsn::new(MIGRATION_LSN.raw() + 1 + offset as u64);
        let stale = node(1, 177 + offset as u32, lsn.raw());
        let delta = arcgraph_storage::wal::DeltaOp::new(
            DeltaOpKind::PutRecord,
            STORE_RECORD,
            tenant,
            0,
            0,
            lsn,
            Bytes::copy_from_slice(&stale.to_bytes()),
        )
        .unwrap();
        let bundle = encode_commit_bundle_v10(
            lsn,
            tenant,
            &HashMap::new(),
            &[],
            &[delta],
            &[],
            &[],
            &[],
            &[],
            &[],
        )
        .unwrap();
        assert_eq!(
            writer
                .handle()
                .append_at(
                    lsn,
                    WalRecordType::CommitBundle,
                    lsn.raw(),
                    0,
                    tenant,
                    bundle,
                )
                .unwrap(),
            lsn
        );
    }

    unsafe {
        libc::kill(libc::getpid(), libc::SIGKILL);
    }
    unreachable!("SIGKILL did not terminate the WAL exposure child");
}

#[cfg(feature = "fault-injection")]
#[test]
fn attach_swap_requires_fsync_ledger() {
    // Arm 1: kill-9 before the complete-generation ledger acknowledges. The
    // beside-build may remain as scratch, but restart must still select only
    // the old generation; no half-live gen-v10 commit point may exist.
    let unacked = tempdir().unwrap();
    production_v5_fixture(unacked.path());
    let status = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "m4_kill9_child", "--nocapture"])
        .env(CHILD_ROOT, unacked.path())
        .env(CHILD_FAULT, "before-ledger")
        .status()
        .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        assert_eq!(status.signal(), Some(libc::SIGKILL));
    }
    let selected = current_generation(unacked.path()).unwrap().unwrap();
    assert_eq!(
        selected.file_name().unwrap(),
        "gen-v9",
        "an unacknowledged generation became live"
    );
    assert!(
        !unacked.path().join("gen-v10").exists(),
        "an unacknowledged generation crossed the publish boundary"
    );
    let mode = BootstrapMode::Durable {
        data_dir: unacked.path().to_path_buf(),
    };
    let (backend, guard) = bootstrap_storage_backend(&mode).unwrap();
    drop(backend);
    drop(guard);
    assert_eq!(
        current_generation(unacked.path())
            .unwrap()
            .unwrap()
            .file_name()
            .unwrap(),
        "gen-v9",
        "restart exposed an unacknowledged generation"
    );

    // Arm 2: a separate process completes the real recursive ledger and
    // consumes its proof at the CURRENT swap. A fresh production bootstrap
    // must recover both non-default tenants from the selected v6 generation.
    let acked = tempdir().unwrap();
    production_v5_fixture(acked.path());
    let status = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "m5_attach_complete_child", "--nocapture"])
        .env(ATTACH_COMPLETE_ROOT, acked.path())
        .status()
        .unwrap();
    assert!(status.success(), "ledger-acknowledged attach child failed");
    let selected = current_generation(acked.path()).unwrap().unwrap();
    assert_eq!(selected.file_name().unwrap(), "gen-v10");
    assert_eq!(
        arcgraph_storage::check_or_stamp_data_dir(&selected, true, false).unwrap(),
        DATA_DIR_VERSION_M4
    );

    let mode = BootstrapMode::Durable {
        data_dir: acked.path().to_path_buf(),
    };
    let (backend, guard) = bootstrap_storage_backend(&mode).unwrap();
    let tenant = backend
        .router()
        .route(TenantId::DEFAULT, arcgraph_core::PartitionId::ZERO)
        .unwrap();
    for migrated_tenant in TENANTS {
        let reader = backend.txn_manager().begin(migrated_tenant);
        assert!(
            read_node_with_store(tenant.crud(), &reader, NodeId::new(1))
                .unwrap()
                .is_some(),
            "ledger-acknowledged attach lost tenant {migrated_tenant:?}"
        );
        reader.abort();
    }
    drop(tenant);
    drop(backend);
    drop(guard);
}

/// INV-M5.6 release lane. The fixture is the two-non-default-tenant production
/// v5 layout and includes a real, recoverable v9 WAL record. The only seam is
/// the cfg-gated false proof handed to the shared per-leg publication object.
///
/// Run with:
/// `cargo test -p arcgraph-cli --release --features fault-injection --test m4_data_dir_migration_gate ledger_proof_consumed_in_release -- --exact`
#[cfg(all(feature = "fault-injection", not(debug_assertions)))]
#[test]
fn ledger_proof_consumed_in_release() {
    assert!(
        TENANTS.iter().all(|tenant| *tenant != TenantId::DEFAULT),
        "gate tenants must be non-default"
    );
    assert_ne!(TENANTS[0], TENANTS[1], "gate must be multi-tenant");

    let fixture = tempdir().unwrap();
    let predecessor = production_v5_fixture(fixture.path());
    assert!(
        WalRecoveryReader::open(predecessor.join("wal"))
            .unwrap()
            .next()
            .transpose()
            .unwrap()
            .is_some(),
        "gate fixture must contain a real source WAL record"
    );

    let status = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "ledger_proof_production_child", "--nocapture"])
        .env(LEDGER_PROOF_ROOT, fixture.path())
        .env(PRODUCTION_MISSING_LEDGER, "1")
        .status()
        .unwrap();
    assert!(
        status.success(),
        "production false-proof child failed: {status:?}"
    );
    assert_eq!(
        current_generation(fixture.path())
            .unwrap()
            .unwrap()
            .file_name()
            .unwrap(),
        "gen-v9",
        "false proof changed CURRENT in a release build"
    );
    assert!(
        !fixture.path().join("gen-v10").exists(),
        "false proof crossed the generation publication boundary"
    );

    let mode = BootstrapMode::Durable {
        data_dir: fixture.path().to_path_buf(),
    };
    let (backend, guard) = bootstrap_storage_backend(&mode)
        .expect("refused publication must leave the production predecessor bootable");
    let routed = backend
        .router()
        .route(TenantId::DEFAULT, arcgraph_core::PartitionId::ZERO)
        .unwrap();
    for tenant in TENANTS {
        let reader = backend.txn_manager().begin(tenant);
        assert!(
            read_node_with_store(routed.crud(), &reader, NodeId::new(1))
                .unwrap()
                .is_some(),
            "refused publication lost tenant {}",
            tenant.raw()
        );
    }
    drop(backend);
    drop(guard);
}

#[cfg(all(feature = "fault-injection", not(debug_assertions)))]
#[test]
fn ledger_proof_production_child() {
    let Ok(root) = std::env::var(LEDGER_PROOF_ROOT) else {
        return;
    };
    let error = upgrade_data_dir(Path::new(&root))
        .expect_err("release production publication must refuse a false durability proof");
    let error_chain = format!("{error:#}");
    assert!(
        error_chain.contains("complete durability ledger proof is absent"),
        "unexpected production false-proof refusal: {error_chain}"
    );
}

#[cfg(feature = "fault-injection")]
#[test]
fn old_generation_unlink_waits_for_drain() {
    // Arm 0: the operator swap must preserve the v5 fallback. A later boot's
    // successful checkpoint is the first point that may reap it.
    let production = tempdir().unwrap();
    let production_predecessor = production_v5_fixture(production.path());
    assert!(
        WalRecoveryReader::open(production_predecessor.join("wal"))
            .unwrap()
            .next()
            .is_some(),
        "production gate fixture did not exercise a real source WAL"
    );
    assert!(matches!(
        upgrade_data_dir(production.path()).unwrap(),
        MigrationOutcome::Upgraded { .. }
    ));
    assert!(
        production_predecessor.exists(),
        "production upgrade reaped its crash-recovery fallback too early"
    );
    assert!(
        !production.path().join(".gen-v9.cleanup").exists(),
        "production upgrade leaked its cleanup retirement tree"
    );
    let mode = BootstrapMode::Durable {
        data_dir: production.path().to_path_buf(),
    };
    let (backend, guard) = bootstrap_storage_backend(&mode).unwrap();
    drop(backend);
    drop(guard);
    assert!(
        !production_predecessor.exists(),
        "post-swap durable checkpoint never reaped the drained predecessor"
    );

    // Arm 1: a generation-qualified reader spans the real durable v5->v6
    // attach. Cleanup must park behind that reader and the reader must keep
    // serving both non-default tenants from the immutable v5 bytes.
    let fixture = tempdir().unwrap();
    let predecessor = production_v5_fixture(fixture.path());
    assert!(
        WalRecoveryReader::open(predecessor.join("wal"))
            .unwrap()
            .next()
            .is_some(),
        "gate fixture did not exercise a real source WAL"
    );
    let tenant_reads: Vec<_> = TENANTS
        .into_iter()
        .map(|tenant| {
            let path = arcgraph_storage::m3_migration::m3_record_store_path(&predecessor, tenant);
            let relative = path.strip_prefix(&predecessor).unwrap().to_path_buf();
            let expected = fs::read(&path).unwrap();
            (tenant, relative, expected)
        })
        .collect();

    let pins = GenerationPinRegistry::new();
    let reader = pins.pin(&predecessor).unwrap();
    assert_eq!(
        upgrade_quiesced_v5_to_v6(fixture.path(), MIGRATION_LSN, MigrationFault::None).unwrap(),
        MigrationOutcome::Upgraded {
            migration_lsn: MIGRATION_LSN
        }
    );
    let successor = current_generation(fixture.path()).unwrap().unwrap();
    assert_eq!(successor.file_name().unwrap(), "gen-v10");
    let too_early = cleanup_old_generation_after_drain(
        DurableGenerationSwap::verify(fixture.path(), &predecessor, &successor).unwrap(),
        &GenerationPinRegistry::new(),
        GenerationCleanupFault::None,
    )
    .expect_err("swap alone must not authorize predecessor reaping");
    assert!(
        format!("{too_early:#}").contains("post-swap successor checkpoint"),
        "unexpected early-reap refusal: {too_early:#}"
    );
    assert!(
        predecessor.exists(),
        "early-reap refusal removed the fallback"
    );

    let mode = BootstrapMode::Durable {
        data_dir: fixture.path().to_path_buf(),
    };
    let (backend, guard) = bootstrap_storage_backend(&mode).unwrap();
    guard
        .checkpointer()
        .expect("durable bootstrap must expose its checkpointer")
        .checkpoint()
        .unwrap();

    let swap = DurableGenerationSwap::verify(fixture.path(), &predecessor, &successor).unwrap();
    let cleanup_pins = pins.clone();
    let cleanup = std::thread::spawn(move || {
        cleanup_old_generation_after_drain(swap, &cleanup_pins, GenerationCleanupFault::None)
    });

    let wait_started = Instant::now();
    while !pins.cleanup_waiting(&predecessor) && wait_started.elapsed() < Duration::from_secs(5) {
        std::thread::yield_now();
    }
    assert!(
        pins.cleanup_waiting(&predecessor),
        "cleanup did not wait behind the pre-swap reader pin"
    );
    assert!(
        predecessor.is_dir(),
        "old generation was unlinked while its reader was pinned"
    );
    for (tenant, relative, expected) in &tenant_reads {
        assert_eq!(
            reader.read(relative).unwrap(),
            *expected,
            "pinned reader stopped serving old-generation bytes for {tenant:?}"
        );
    }

    drop(reader);
    cleanup.join().unwrap().unwrap();
    assert!(
        !predecessor.exists(),
        "old generation remained linked after its final pin drained"
    );
    assert!(
        !fixture.path().join(".gen-v9.cleanup").exists(),
        "completed cleanup leaked its retirement tree"
    );
    drop(backend);
    drop(guard);

    // Arm 2: kill-9 after the first durable unlink. Restart must discover the
    // retirement tree, safely finish it, and leave the committed successor
    // bootable for both non-default tenants.
    let crashed = tempdir().unwrap();
    let crashed_predecessor = production_v5_fixture(crashed.path());
    let status = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "old_generation_cleanup_kill9_child",
            "--nocapture",
        ])
        .env(OLD_GENERATION_CLEANUP_ROOT, crashed.path())
        .env(PRODUCTION_CLEANUP_CRASH, "1")
        .status()
        .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        assert_eq!(status.signal(), Some(libc::SIGKILL));
    }
    assert!(
        !crashed_predecessor.exists() && crashed.path().join(".gen-v9.cleanup").exists(),
        "kill-9 did not land after retirement and during unlink"
    );

    let mode = BootstrapMode::Durable {
        data_dir: crashed.path().to_path_buf(),
    };
    let (backend, guard) = bootstrap_storage_backend(&mode).unwrap();
    assert!(
        !crashed_predecessor.exists() && !crashed.path().join(".gen-v9.cleanup").exists(),
        "production restart cleanup leaked old-generation files"
    );
    assert_eq!(
        current_generation(crashed.path())
            .unwrap()
            .unwrap()
            .file_name()
            .unwrap(),
        "gen-v10"
    );
    let tenant = backend
        .router()
        .route(TenantId::DEFAULT, arcgraph_core::PartitionId::ZERO)
        .unwrap();
    for migrated_tenant in TENANTS {
        let reader = backend.txn_manager().begin(migrated_tenant);
        assert!(
            read_node_with_store(tenant.crud(), &reader, NodeId::new(1))
                .unwrap()
                .is_some(),
            "resumed cleanup lost tenant {migrated_tenant:?}"
        );
        reader.abort();
    }
    drop(tenant);
    drop(backend);
    drop(guard);
}

/// INV-M5.4 — drive the CURRENT swap INTO the pin-acquisition window.
///
/// A reader thread parks inside `pin_current_generation` between resolving
/// CURRENT (latching the predecessor) and revalidating; a swap thread then
/// publishes gen-v10 while the acquisition is provably mid-window, and only
/// afterwards is the parked acquisition released. The mechanism under test —
/// the revalidate-retry loop in `data_dir_migration::pin_current_generation`
/// — must observe the moved pointer, retry (`acquisition_attempts >= 2`),
/// and hand back a WHOLLY-new epoch, while a pin that completed before the
/// swap keeps its WHOLLY-old epoch. RED-on-revert: the naive no-revalidation
/// pin under this exact schedule returns the superseded predecessor — see
/// `attach_reader_red_control_naive_pin_latches_stale_generation`.
#[cfg(feature = "fault-injection")]
#[test]
fn attach_under_concurrent_reader() {
    let fixture = tempdir().unwrap();
    let predecessor = v5_fixture(fixture.path());
    let pins = GenerationPinRegistry::new();

    // Wholly-OLD arm: an epoch acquired before the swap keeps every byte of
    // its generation across the swap.
    let pre_swap = pin_current_generation(fixture.path(), &pins).unwrap();
    assert_eq!(pre_swap.generation(), predecessor);
    let old_paths: Vec<_> = TENANTS
        .into_iter()
        .map(|tenant| {
            arcgraph_storage::m3_migration::m3_record_store_path(&predecessor, tenant)
                .strip_prefix(&predecessor)
                .unwrap()
                .to_path_buf()
        })
        .collect();
    let old_expected: Vec<_> = old_paths
        .iter()
        .map(|relative| fs::read(predecessor.join(relative)).unwrap())
        .collect();
    let old_version = fs::read(predecessor.join("VERSION")).unwrap();

    // Racing arm: park a fresh acquisition mid-window (generation latched
    // from CURRENT, not yet revalidated).
    let rendezvous = PinAcquisitionRendezvous::install(fixture.path()).unwrap();
    let racing = {
        let root = fixture.path().to_path_buf();
        let pins = pins.clone();
        std::thread::spawn(move || {
            let reader = pin_current_generation(&root, &pins).unwrap();
            let version = reader.read(Path::new("VERSION")).unwrap();
            (reader.generation().to_path_buf(), version)
        })
    };
    let latched = rendezvous.wait_until_pin_latched().unwrap();
    assert_eq!(
        latched, predecessor,
        "racing acquisition latched something other than pre-swap CURRENT"
    );

    // Swap thread: rename CURRENT to the successor while the acquisition is
    // parked. Joining BEFORE the release orders the schedule deterministically:
    // latch(gen-v9) → swap(gen-v10) → revalidate.
    let swap = {
        let root = fixture.path().to_path_buf();
        std::thread::spawn(move || {
            upgrade_quiesced_v5_to_v6(&root, MIGRATION_LSN, MigrationFault::None)
        })
    };
    assert_eq!(
        swap.join().unwrap().unwrap(),
        MigrationOutcome::Upgraded {
            migration_lsn: MIGRATION_LSN
        }
    );
    let successor = current_generation(fixture.path()).unwrap().unwrap();
    assert_ne!(successor, predecessor);

    rendezvous.release();
    let (racing_generation, racing_version) = racing.join().unwrap();
    let attempts = rendezvous.acquisition_attempts();
    drop(rendezvous);

    // Wholly-NEW: the stale latch may never survive the swap, and the
    // revalidate-retry loop must actually have run.
    assert_eq!(
        racing_generation, successor,
        "racing pin kept a stale generation latch across the swap"
    );
    let successor_version = fs::read(successor.join("VERSION")).unwrap();
    assert_ne!(
        successor_version, old_version,
        "fixture generations are not distinguishable by VERSION"
    );
    assert_eq!(
        racing_version, successor_version,
        "racing pin returned torn or stale VERSION content"
    );
    assert!(
        attempts >= 2,
        "pin revalidate-retry loop never retried (acquisition attempts = {attempts})"
    );

    // Wholly-OLD: the pre-swap epoch is intact — never torn toward v6.
    assert_eq!(pre_swap.generation(), predecessor);
    for (relative, expected) in old_paths.iter().zip(&old_expected) {
        assert_eq!(
            &pre_swap.read(relative).unwrap(),
            expected,
            "pre-swap pinned epoch lost {} across the swap",
            relative.display()
        );
    }
    assert_eq!(pre_swap.read(Path::new("VERSION")).unwrap(), old_version);
    drop(pre_swap);

    let new_reader = pin_current_generation(fixture.path(), &pins).unwrap();
    assert_eq!(new_reader.generation(), successor);
    for tenant in TENANTS {
        let store = production_extent_store_path(&successor, tenant, STORE_RECORD).unwrap();
        let relative = store.strip_prefix(&successor).unwrap();
        let directory_page = new_reader.read_exact_at(relative, 0, PAGE_SIZE).unwrap();
        assert!(
            directory_page.iter().any(|byte| *byte != 0),
            "new epoch exposed an empty record extent directory for tenant {tenant:?}"
        );
    }
}

/// INV-M5.4 negative control — the naive, no-revalidation pin under the
/// EXACT schedule `attach_under_concurrent_reader` forces observes the
/// superseded generation and never retries. This is the in-tree proof that
/// the forced race bites: revert `pin_current_generation` to the naive shape
/// and the attach gate goes RED at its wholly-new assertions.
#[cfg(feature = "fault-injection")]
#[test]
fn attach_reader_red_control_naive_pin_latches_stale_generation() {
    let fixture = tempdir().unwrap();
    let predecessor = v5_fixture(fixture.path());
    let pins = GenerationPinRegistry::new();
    let rendezvous = PinAcquisitionRendezvous::install(fixture.path()).unwrap();
    let racing = {
        let root = fixture.path().to_path_buf();
        let pins = pins.clone();
        std::thread::spawn(move || {
            naive_unrevalidated_pin_for_red_control(&root, &pins)
                .map(|reader| reader.generation().to_path_buf())
        })
    };
    let latched = rendezvous.wait_until_pin_latched().unwrap();
    assert_eq!(latched, predecessor);

    let swap = {
        let root = fixture.path().to_path_buf();
        std::thread::spawn(move || {
            upgrade_quiesced_v5_to_v6(&root, MIGRATION_LSN, MigrationFault::None)
        })
    };
    assert_eq!(
        swap.join().unwrap().unwrap(),
        MigrationOutcome::Upgraded {
            migration_lsn: MIGRATION_LSN
        }
    );
    let successor = current_generation(fixture.path()).unwrap().unwrap();
    assert_ne!(successor, predecessor);

    rendezvous.release();
    let naive_generation = racing.join().unwrap().unwrap();
    assert_eq!(
        rendezvous.acquisition_attempts(),
        1,
        "the naive pin has no retry loop to run"
    );
    assert_eq!(
        naive_generation, predecessor,
        "under the forced mid-window swap the naive pin MUST exhibit the \
         stale-latch defect the production revalidate loop prevents"
    );
}

#[cfg(all(feature = "fault-injection", unix))]
#[test]
fn tenant_qualified_kill9_child() {
    let Ok(root) = std::env::var(TENANT_IDENTITY_ROOT) else {
        return;
    };
    let root = PathBuf::from(root);
    let mode = BootstrapMode::Durable {
        data_dir: root.clone(),
    };
    let (backend, guard) = bootstrap_storage_backend(&mode).unwrap();
    let mut ids = Vec::new();
    for (tenant, payload) in TENANTS.into_iter().zip([
        b"tenant-41-after-kill9".to_vec(),
        b"tenant-73-after-kill9".to_vec(),
    ]) {
        ids.push(commit_property_node(&backend, tenant, &PropertyData::Blob(payload)).0);
    }
    assert_eq!(ids[0], ids[1], "gate requires cross-tenant id reuse");
    guard.wal_handle().unwrap().flush().unwrap();
    let id_path = root.join("M5_TENANT_IDENTITY_ID");
    let mut id_file = File::create(&id_path).unwrap();
    id_file.write_all(&ids[0].raw().to_le_bytes()).unwrap();
    id_file.sync_all().unwrap();
    File::open(&root).unwrap().sync_all().unwrap();
    unsafe {
        libc::kill(libc::getpid(), libc::SIGKILL);
    }
    unreachable!("SIGKILL returned");
}

/// INV-M5.7 — the same page/id coordinates in two non-default tenants remain
/// distinct through the production cache key, WAL fsync, SIGKILL, recovery,
/// and simultaneous reads after reopen.
#[cfg(all(feature = "fault-injection", unix))]
#[test]
fn tenant_qualified_end_to_end() {
    assert!(TENANTS.iter().all(|tenant| *tenant != TenantId::DEFAULT));
    assert_ne!(TENANTS[0], TENANTS[1]);
    let fixture = tempdir().unwrap();
    let predecessor = production_v5_fixture(fixture.path());
    upgrade_data_dir(fixture.path()).unwrap();
    let successor = current_generation(fixture.path()).unwrap().unwrap();

    // Use the shipped tenant-file resolver, per-tenant buffer pool, and hot
    // record cache. Only the identities are inspected; no richer test double
    // can conceal a qualifier dropped by production key construction.
    let tenant_io: Arc<dyn TenantPageIo> = Arc::new(TenantFilePageIo::new(
        &successor,
        arcgraph_storage::m3_migration::M3_RECORD_STORE_FILE,
    ));
    let pools = Arc::new(PerTenantBufferPool::with_tenant_io(
        tenant_io,
        PerTenantBufferPoolConfig {
            frames_per_tenant: 2,
            write_fraction: 0.5,
        },
    ));
    let cache = BufferedRecordPageStore::with_identity(
        pools,
        PageStoreIdentity::for_generation(&successor, STORE_RECORD),
    );
    let page = PageId::new(1);
    let tenant_a = cache.page_key(TENANTS[0], page);
    let tenant_b = cache.page_key(TENANTS[1], page);
    assert_ne!(tenant_a, tenant_b, "cross-tenant cache keys alias");
    assert_eq!(tenant_a.store_id, STORE_RECORD);
    assert_ne!(
        tenant_a.generation_id,
        PageStoreIdentity::for_generation(&predecessor, STORE_RECORD).generation_id,
        "predecessor and successor cache generations alias"
    );

    let status = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "tenant_qualified_kill9_child", "--nocapture"])
        .env(TENANT_IDENTITY_ROOT, fixture.path())
        .status()
        .unwrap();
    use std::os::unix::process::ExitStatusExt;
    assert_eq!(status.signal(), Some(libc::SIGKILL));

    let id_bytes: [u8; 8] = fs::read(fixture.path().join("M5_TENANT_IDENTITY_ID"))
        .unwrap()
        .try_into()
        .unwrap();
    let shared_id = NodeId::new(u64::from_le_bytes(id_bytes));
    let mode = BootstrapMode::Durable {
        data_dir: fixture.path().to_path_buf(),
    };
    let (backend, guard) = bootstrap_storage_backend(&mode).unwrap();
    assert_eq!(guard.generation(), Some(successor.as_path()));
    let backend = Arc::new(backend);
    let rendezvous = Arc::new(Barrier::new(TENANTS.len() + 1));
    let readers: Vec<_> = TENANTS
        .into_iter()
        .zip([
            b"tenant-41-after-kill9".to_vec(),
            b"tenant-73-after-kill9".to_vec(),
        ])
        .map(|(tenant, expected)| {
            let backend = Arc::clone(&backend);
            let rendezvous = Arc::clone(&rendezvous);
            std::thread::spawn(move || {
                rendezvous.wait();
                assert_node_bag(&backend, tenant, shared_id, &expected);
            })
        })
        .collect();
    rendezvous.wait();
    for reader in readers {
        reader.join().unwrap();
    }
    drop(backend);
    drop(guard);
}

/// INV-M5.10 — pause the real beside-build after its page passes but before
/// publication. Both checkpoint forms must continue to census only the live
/// generation; the invisible directory is never a buffer/DPT/DWB/checkpointer
/// owner.
#[cfg(feature = "fault-injection")]
#[test]
fn mid_build_checkpoints_exclude_build_pages() {
    let fixture = tempdir().unwrap();
    let predecessor = production_v5_fixture(fixture.path());
    let mode = BootstrapMode::Durable {
        data_dir: fixture.path().to_path_buf(),
    };
    let (backend, guard) = bootstrap_storage_backend(&mode).unwrap();
    assert_eq!(guard.generation(), Some(predecessor.as_path()));
    let checkpointer = guard
        .checkpointer()
        .expect("production v5 fixture omitted its checkpointer");

    // Establish both baseline census shapes before any build page exists.
    let baseline_full = checkpointer
        .full_checkpoint_for_build_isolation_gate()
        .unwrap();
    let baseline_incremental = checkpointer
        .incremental_checkpoint_for_build_isolation_gate()
        .unwrap();
    let migration_lsn = baseline_incremental.checkpoint_lsn;
    let baseline = checkpointer.build_isolation_census().unwrap();
    assert_eq!(baseline.generation, predecessor);
    assert!(baseline.dpt.is_empty());
    assert!(baseline.doublewrite.is_empty());
    assert!(baseline.checkpointer_routes.contains(&(None, STORE_RECORD)));
    assert!(baseline.checkpointer_routes.contains(&(None, STORE_PROPS)));

    let live_dpt = guard
        .extent_store(TENANTS[0], STORE_RECORD)
        .expect("production v5 record owner is not wired")
        .dirty_page_table()
        .clone();
    // One armed leak per censused live structure: DPT, doublewrite, buffer
    // pool, checkpointer routes. (The fifth structure, the OQ-G counts, is
    // armed by ARCGRAPH_M5_ROUTE_BUILD_OQG_LIVE inside the gate checkpoint
    // helpers — see bootstrap.rs::armed_oqg_build_leak.) Every leak fires
    // only in its RED-control child process; unarmed runs inject nothing.
    let rendezvous = {
        let dwb_generation = predecessor.clone();
        let pool_checkpointer = checkpointer.clone();
        let route_checkpointer = checkpointer.clone();
        MigrationBuildRendezvous::install(
            fixture.path(),
            vec![
                BuildLeakInjector {
                    env: "ARCGRAPH_M5_ROUTE_BUILD_PAGE_LIVE",
                    inject: Box::new(move || {
                        live_dpt.mark_dirty(
                            arcgraph_storage::DirtyPageKey {
                                tenant_id: TENANTS[0],
                                store_id: STORE_RECORD,
                                page_no: 0x5c_10,
                            },
                            Lsn::new(1),
                        );
                    }),
                },
                BuildLeakInjector {
                    env: "ARCGRAPH_M5_ROUTE_BUILD_DWB_LIVE",
                    inject: Box::new(move || {
                        let key = arcgraph_storage::checkpoint::DoublewriteKey {
                            tenant_id: TENANTS[0],
                            store_id: STORE_RECORD,
                            page_no: 0x5c_11,
                        };
                        let mut page = [0u8; PAGE_SIZE];
                        let mut header =
                            PageHeader::new(PageId::new(0x5c_11), PageType::Node, TENANTS[0]);
                        header.checksum = crc32c::crc32c(&page[PageHeader::SIZE..]);
                        page[..PageHeader::SIZE].copy_from_slice(&header.to_bytes());
                        DoublewriteArea::new(&dwb_generation)
                            .stage_batch(&[(key, &page)])
                            .expect("stage build page into live doublewrite");
                    }),
                },
                BuildLeakInjector {
                    env: "ARCGRAPH_M5_ROUTE_BUILD_POOL_LIVE",
                    inject: Box::new(move || {
                        pool_checkpointer
                            .build_isolation_gate_map_phantom_page(0x5c_12)
                            .expect("map phantom build page into live buffer pool");
                    }),
                },
                BuildLeakInjector {
                    env: "ARCGRAPH_M5_ROUTE_BUILD_ROUTE_LIVE",
                    inject: Box::new(move || {
                        route_checkpointer
                            .build_isolation_gate_inject_route(Some(TENANTS[0]), 0x5c13)
                            .expect("inject build route into live checkpointer");
                    }),
                },
            ],
        )
        .unwrap()
    };
    let root = fixture.path().to_path_buf();
    let builder = std::thread::spawn(move || {
        upgrade_quiesced_v5_to_v6(&root, migration_lsn, MigrationFault::None)
    });
    rendezvous.wait_until_reached().unwrap();
    let building = fixture.path().join("gen-v10.building");
    assert!(building.is_dir(), "rendezvous did not stop a real build");

    // The deliberate live-route fault marks the DPT before this census. It is
    // the RED-on-revert control and must fail here, before a checkpointer can
    // flush away the evidence.
    let paused = checkpointer.build_isolation_census().unwrap();
    assert_eq!(paused.generation, predecessor);
    assert_eq!(
        paused.buffer_pool_pages, baseline.buffer_pool_pages,
        "invisible build page entered live buffer pool"
    );
    assert_eq!(
        paused.checkpointer_routes, baseline.checkpointer_routes,
        "invisible build acquired a live checkpointer route"
    );
    assert!(
        paused.dpt.is_empty(),
        "invisible build page entered live DPT"
    );
    assert!(
        paused.doublewrite.is_empty(),
        "invisible build page entered live doublewrite"
    );
    assert_ne!(paused.generation, building);

    let mid_full = checkpointer
        .full_checkpoint_for_build_isolation_gate()
        .unwrap();
    let mid_incremental = checkpointer
        .incremental_checkpoint_for_build_isolation_gate()
        .unwrap();
    assert_eq!(
        mid_full.counts, baseline_full.counts,
        "full OQ-G census changed while only invisible pages were built"
    );
    assert_eq!(
        mid_incremental.counts, baseline_incremental.counts,
        "incremental OQ-G census changed while only invisible pages were built"
    );
    let after = checkpointer.build_isolation_census().unwrap();
    assert!(after.dpt.is_empty());
    assert!(after.doublewrite.is_empty());
    assert_eq!(after.generation, predecessor);

    rendezvous.release();
    assert!(matches!(
        builder.join().unwrap().unwrap(),
        MigrationOutcome::Upgraded { migration_lsn: lsn } if lsn == migration_lsn
    ));
    assert_eq!(
        current_generation(fixture.path())
            .unwrap()
            .unwrap()
            .file_name()
            .unwrap(),
        "gen-v10"
    );
    drop(backend);
    drop(guard);
}

#[cfg(all(feature = "fault-injection", unix))]
#[test]
fn index_vector_attach_kill9_child() {
    let Ok(root) = std::env::var(INDEX_VECTOR_ROOT) else {
        return;
    };
    upgrade_data_dir(Path::new(&root)).unwrap();
    unsafe {
        libc::kill(libc::getpid(), libc::SIGKILL);
    }
    unreachable!("SIGKILL returned");
}

/// INV-M5.13 — preserved vector/index files and produced secondary/forward
/// indexes must all appear in the durable complete-file census before attach.
/// The production child dies immediately after attach; reopen therefore also
/// proves the census survived without a graceful shutdown flush.
#[cfg(all(feature = "fault-injection", unix))]
#[test]
fn index_vector_passes_fsync_before_attach() {
    let fixture = tempdir().unwrap();
    let source = production_v5_fixture(fixture.path());
    assert!(
        WalRecoveryReader::open(source.join("wal"))
            .unwrap()
            .next()
            .is_some(),
        "index/vector gate requires a real v5 WAL"
    );
    let artifacts = [
        (
            "vector-index/pass-0001.bin",
            b"vector-pass-complete".as_slice(),
        ),
        ("bm25/index/segment-41", b"bm25-pass-complete".as_slice()),
    ];
    for (relative, bytes) in artifacts {
        let path = source.join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut file = File::create(&path).unwrap();
        file.write_all(bytes).unwrap();
        file.sync_all().unwrap();
        File::open(path.parent().unwrap())
            .unwrap()
            .sync_all()
            .unwrap();
    }
    File::open(&source).unwrap().sync_all().unwrap();

    let status = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "index_vector_attach_kill9_child", "--nocapture"])
        .env(INDEX_VECTOR_ROOT, fixture.path())
        .status()
        .unwrap();
    use std::os::unix::process::ExitStatusExt;
    assert_eq!(
        status.signal(),
        Some(libc::SIGKILL),
        "index/vector attach child failed before its post-attach SIGKILL: {status:?}"
    );

    let selected = current_generation(fixture.path()).unwrap().unwrap();
    assert_eq!(selected.file_name().unwrap(), "gen-v10");
    for (relative, bytes) in artifacts {
        assert_eq!(
            fs::read(selected.join(relative)).unwrap(),
            bytes,
            "preserved pass output differs: {relative}"
        );
    }
    let census = fs::read_to_string(selected.join(INDEX_VECTOR_CENSUS_FILE)).unwrap();
    for required in ["vector-index/pass-0001.bin", "bm25/index/segment-41"] {
        assert!(
            census.contains(required),
            "census omitted {required}: {census}"
        );
    }
    assert!(
        census.lines().any(|line| line.contains("secondary.index")),
        "census omitted produced secondary-index store: {census}"
    );
    for line in census.lines() {
        let mut columns = line.split('\t');
        let relative = columns.next().unwrap();
        let size: u64 = columns.next().unwrap().parse().unwrap();
        let sha = columns.next().unwrap();
        assert!(columns.next().is_none(), "malformed census row: {line}");
        let file = selected.join(relative);
        assert_eq!(fs::metadata(&file).unwrap().len(), size);
        assert_eq!(sha.len(), 64, "census SHA-256 is malformed: {line}");
    }

    let mode = BootstrapMode::Durable {
        data_dir: fixture.path().to_path_buf(),
    };
    let (backend, guard) = bootstrap_storage_backend(&mode).unwrap();
    assert_eq!(guard.generation(), Some(selected.as_path()));
    drop(backend);
    drop(guard);
}

/// Re-run one gate test in a child process with fault env vars armed,
/// returning whether the child harness passed plus its combined output.
/// Armed RED controls use this so the fault environment never leaks into
/// the sibling tests of this (parent) process.
#[cfg(feature = "fault-injection")]
fn run_gate_with_armed_faults(test_name: &str, envs: &[(&str, &str)]) -> (bool, String) {
    let mut command = Command::new(std::env::current_exe().unwrap());
    command.args(["--exact", test_name, "--nocapture"]);
    for (key, value) in envs {
        command.env(key, value);
    }
    let output = command.output().unwrap();
    let mut combined = String::from_utf8_lossy(&output.stdout).into_owned();
    combined.push_str(&String::from_utf8_lossy(&output.stderr));
    (output.status.success(), combined)
}

/// INV-M5.10 armed RED control — `ARCGRAPH_M5_ROUTE_BUILD_PAGE_LIVE` routes
/// one invisible-build page into the live DPT at the build rendezvous. The
/// mid-build census gate must go RED at its DPT assertion; a green run here
/// would mean the leak detector is decorative.
#[cfg(feature = "fault-injection")]
#[test]
fn red_control_build_page_routed_to_live_dpt_fails_gate() {
    let (passed, output) = run_gate_with_armed_faults(
        "mid_build_checkpoints_exclude_build_pages",
        &[("ARCGRAPH_M5_ROUTE_BUILD_PAGE_LIVE", "1")],
    );
    assert!(
        !passed,
        "armed live-DPT build-page leak was not caught by the gate:\n{output}"
    );
    assert!(
        output.contains("invisible build page entered live DPT"),
        "gate went RED for the wrong reason:\n{output}"
    );
}

/// INV-M5.10 armed RED control — `ARCGRAPH_M5_ROUTE_BUILD_DWB_LIVE` stages
/// one invisible-build page into the LIVE doublewrite area at the build
/// rendezvous; the census gate must go RED at its doublewrite assertion.
#[cfg(feature = "fault-injection")]
#[test]
fn red_control_build_page_staged_to_live_doublewrite_fails_gate() {
    let (passed, output) = run_gate_with_armed_faults(
        "mid_build_checkpoints_exclude_build_pages",
        &[("ARCGRAPH_M5_ROUTE_BUILD_DWB_LIVE", "1")],
    );
    assert!(
        !passed,
        "armed live-doublewrite build-page leak was not caught by the gate:\n{output}"
    );
    assert!(
        output.contains("invisible build page entered live doublewrite"),
        "gate went RED for the wrong reason:\n{output}"
    );
}

/// INV-M5.10 armed RED control — `ARCGRAPH_M5_ROUTE_BUILD_POOL_LIVE` binds
/// one invisible-build page into the LIVE buffer pool at the build
/// rendezvous; the census gate must go RED at its buffer-pool assertion.
#[cfg(feature = "fault-injection")]
#[test]
fn red_control_build_page_mapped_to_live_buffer_pool_fails_gate() {
    let (passed, output) = run_gate_with_armed_faults(
        "mid_build_checkpoints_exclude_build_pages",
        &[("ARCGRAPH_M5_ROUTE_BUILD_POOL_LIVE", "1")],
    );
    assert!(
        !passed,
        "armed live-buffer-pool build-page leak was not caught by the gate:\n{output}"
    );
    assert!(
        output.contains("invisible build page entered live buffer pool"),
        "gate went RED for the wrong reason:\n{output}"
    );
}

/// INV-M5.10 armed RED control — `ARCGRAPH_M5_ROUTE_BUILD_ROUTE_LIVE`
/// registers one build-store route on the LIVE write-behind checkpointer at
/// the build rendezvous; the census gate must go RED at its route assertion.
#[cfg(feature = "fault-injection")]
#[test]
fn red_control_build_route_registered_on_live_checkpointer_fails_gate() {
    let (passed, output) = run_gate_with_armed_faults(
        "mid_build_checkpoints_exclude_build_pages",
        &[("ARCGRAPH_M5_ROUTE_BUILD_ROUTE_LIVE", "1")],
    );
    assert!(
        !passed,
        "armed live-checkpointer route leak was not caught by the gate:\n{output}"
    );
    assert!(
        output.contains("invisible build acquired a live checkpointer route"),
        "gate went RED for the wrong reason:\n{output}"
    );
}

/// INV-M5.10 armed RED control — `ARCGRAPH_M5_ROUTE_BUILD_OQG_LIVE`
/// simulates one invisible-build object entering the live OQ-G
/// retained-owner census while the build is in flight (baselines stay
/// unpolluted); the gate must go RED at its count-equality assertion.
#[cfg(feature = "fault-injection")]
#[test]
fn red_control_build_object_counted_in_live_oqg_census_fails_gate() {
    let (passed, output) = run_gate_with_armed_faults(
        "mid_build_checkpoints_exclude_build_pages",
        &[("ARCGRAPH_M5_ROUTE_BUILD_OQG_LIVE", "1")],
    );
    assert!(
        !passed,
        "armed live OQ-G census leak was not caught by the gate:\n{output}"
    );
    assert!(
        output.contains("full OQ-G census changed while only invisible pages were built"),
        "gate went RED for the wrong reason:\n{output}"
    );
}

/// INV-M5.13 armed RED control — `ARCGRAPH_M5_SKIP_INDEX_FSYNC` makes the
/// index/vector pass hand publication an unsynced proof. The commit point
/// must refuse to attach (the child dies before its post-attach SIGKILL)
/// and the gate must go RED; a green run would mean an unsynced census can
/// still publish.
#[cfg(all(feature = "fault-injection", unix))]
#[test]
fn red_control_skipped_index_fsync_fails_gate() {
    let (passed, output) = run_gate_with_armed_faults(
        "index_vector_passes_fsync_before_attach",
        &[("ARCGRAPH_M5_SKIP_INDEX_FSYNC", "1")],
    );
    assert!(
        !passed,
        "armed index/vector fsync skip was not caught by the gate:\n{output}"
    );
    assert!(
        output.contains("index/vector pass fsync proof is absent"),
        "gate went RED for the wrong reason:\n{output}"
    );
}

#[cfg(feature = "fault-injection")]
#[test]
fn generation_commit_point_identity_kill9_sweep() {
    // Every interruption before the terminal VERSION marker must boot the
    // stamped predecessor.  The candidate may be scratch or renamed, and
    // CURRENT may already contain its name, but it is not a committed
    // generation until its own VERSION exists.
    for fault in ["scratch", "before-ledger", "sync", "rename", "current"] {
        let fixture = tempdir().unwrap();
        let source = production_v5_fixture(fixture.path());
        assert!(
            WalRecoveryReader::open(source.join("wal"))
                .unwrap()
                .next()
                .is_some(),
            "fault={fault} fixture did not exercise a real source WAL"
        );

        let status = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "m4_kill9_child", "--nocapture"])
            .env(CHILD_ROOT, fixture.path())
            .env(CHILD_FAULT, fault)
            .status()
            .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            assert_eq!(status.signal(), Some(libc::SIGKILL), "fault={fault}");
        }

        assert!(
            !fixture.path().join("gen-v10/VERSION").exists()
                && !fixture.path().join("gen-v10.building/VERSION").exists(),
            "fault={fault} exposed VERSION before the generation commit finished"
        );
        assert_eq!(
            current_generation(fixture.path())
                .unwrap()
                .unwrap()
                .file_name()
                .unwrap(),
            "gen-v9",
            "fault={fault} made the unstamped candidate visible"
        );

        let mode = BootstrapMode::Durable {
            data_dir: fixture.path().to_path_buf(),
        };
        let (backend, guard) = bootstrap_storage_backend(&mode)
            .unwrap_or_else(|error| panic!("fault={fault} predecessor did not boot: {error:#}"));
        let tenant = backend
            .router()
            .route(TenantId::DEFAULT, arcgraph_core::PartitionId::ZERO)
            .unwrap();
        for migrated_tenant in TENANTS {
            let reader = backend.txn_manager().begin(migrated_tenant);
            assert!(
                read_node_with_store(tenant.crud(), &reader, NodeId::new(1))
                    .unwrap()
                    .is_some(),
                "fault={fault} rollback lost tenant {migrated_tenant:?}"
            );
            reader.abort();
        }
        drop(tenant);
        drop(backend);
        drop(guard);
    }

    // A separately committed leg is visible only after VERSION has completed,
    // and its manifest/checkpoint metadata is generation-local and correct.
    let committed = tempdir().unwrap();
    production_v5_fixture(committed.path());
    // INV-M5.22 cross-tool plant (M5-D1): a foreign fresh-load `.building`
    // dir must survive the ENTIRE v5→v6 migration — including its own
    // stale-scratch sweep — byte-identical. RED-on-revert: point the migrate
    // sweep at a shared `gen-` prefix and the sentinel disappears.
    let planted_loader_building = committed.path().join("gen-load-v6.building");
    fs::create_dir(&planted_loader_building).unwrap();
    fs::write(
        planted_loader_building.join("sentinel"),
        b"loader-owned bytes",
    )
    .unwrap();
    let status = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "m5_attach_complete_child", "--nocapture"])
        .env(ATTACH_COMPLETE_ROOT, committed.path())
        .status()
        .unwrap();
    assert!(status.success(), "post-VERSION commit child failed");
    assert_eq!(
        fs::read(planted_loader_building.join("sentinel")).unwrap(),
        b"loader-owned bytes",
        "the v5->v6 migration consumed or swept the loader's namespace (INV-M5.22)"
    );
    assert!(
        !committed
            .path()
            .join("gen-v10/gen-load-v6.building")
            .exists(),
        "the v5->v6 migration copied the loader's namespace into its generation"
    );
    fs::remove_dir_all(&planted_loader_building).unwrap();

    let selected = current_generation(committed.path()).unwrap().unwrap();
    assert_eq!(selected.file_name().unwrap(), "gen-v10");
    assert_eq!(
        fs::read_to_string(committed.path().join("CURRENT")).unwrap(),
        "gen-v10\n"
    );
    assert_eq!(
        arcgraph_storage::check_or_stamp_data_dir(&selected, true, false).unwrap(),
        DATA_DIR_VERSION_M4
    );
    let manifest = arcgraph_storage::read_data_dir_manifest(&selected)
        .unwrap()
        .unwrap();
    assert_eq!(manifest.data_dir_version, DATA_DIR_VERSION_M4);
    assert_eq!(
        manifest.tenant_census,
        Some(
            std::iter::once(TenantId::DEFAULT.raw())
                .chain(TENANTS.map(TenantId::raw))
                .collect()
        )
    );
    let checkpoint = arcgraph_storage::read_latest_sidecar(&selected)
        .unwrap()
        .unwrap();
    let metadata = incremental_metadata_path(
        &selected,
        checkpoint.checkpoint_lsn,
        checkpoint.metadata_generation,
    );
    assert!(
        metadata.is_file() && metadata.starts_with(&selected),
        "checkpoint metadata did not travel with gen-v10"
    );

    let mode = BootstrapMode::Durable {
        data_dir: committed.path().to_path_buf(),
    };
    let (backend, guard) = bootstrap_storage_backend(&mode).unwrap();
    let tenant = backend
        .router()
        .route(TenantId::DEFAULT, arcgraph_core::PartitionId::ZERO)
        .unwrap();
    for migrated_tenant in TENANTS {
        let reader = backend.txn_manager().begin(migrated_tenant);
        assert!(
            read_node_with_store(tenant.crud(), &reader, NodeId::new(1))
                .unwrap()
                .is_some(),
            "committed generation lost tenant {migrated_tenant:?}"
        );
        reader.abort();
    }
    drop(tenant);
    drop(backend);
    drop(guard);
}

#[test]
fn migrate_v5_to_v6_atomic_or_resumable() {
    let dirty = tempdir().unwrap();
    v5_fixture(dirty.path());
    let metadata = incremental_metadata_path(
        &current_generation(dirty.path()).unwrap().unwrap(),
        MIGRATION_LSN,
        1,
    );
    let mut dirty_header = fs::read(&metadata).unwrap();
    dirty_header[32..40].copy_from_slice(&1_u64.to_le_bytes());
    fs::write(metadata, dirty_header).unwrap();
    assert!(
        upgrade_quiesced_v5_to_v6(dirty.path(), MIGRATION_LSN, MigrationFault::None).is_err(),
        "non-empty DPT was accepted as a full-flush checkpoint"
    );
    assert_eq!(
        current_generation(dirty.path())
            .unwrap()
            .unwrap()
            .file_name()
            .unwrap(),
        "gen-v9"
    );

    let fixture = tempdir().unwrap();
    let source = v5_fixture(fixture.path());
    let before = tree_bytes(&source);
    assert_eq!(
        upgrade_quiesced_v5_to_v6(fixture.path(), MIGRATION_LSN, MigrationFault::None).unwrap(),
        MigrationOutcome::Upgraded {
            migration_lsn: MIGRATION_LSN
        }
    );
    assert_eq!(
        tree_bytes(&source),
        before,
        "v5 source generation was mutated"
    );
    let selected = current_generation(fixture.path()).unwrap().unwrap();
    assert_eq!(selected.file_name().unwrap(), "gen-v10");
    assert_eq!(
        arcgraph_storage::check_or_stamp_data_dir(&selected, true, false).unwrap(),
        DATA_DIR_VERSION_M4
    );
    assert_eq!(
        upgrade_quiesced_v5_to_v6(fixture.path(), Lsn::new(999), MigrationFault::None).unwrap(),
        MigrationOutcome::AlreadyUpgraded {
            migration_lsn: MIGRATION_LSN
        }
    );

    // A post-CURRENT generation remains absent until VERSION. Missing
    // artifacts in that invisible candidate do not affect predecessor boot;
    // force-stamping the corrupt candidate makes the corruption fail closed.
    let half_built = tempdir().unwrap();
    production_v5_fixture(half_built.path());
    assert!(
        upgrade_quiesced_v5_to_v6(
            half_built.path(),
            MIGRATION_LSN,
            MigrationFault::AfterCurrentSwap,
        )
        .is_err()
    );
    let selected = current_generation(half_built.path()).unwrap().unwrap();
    assert_eq!(selected.file_name().unwrap(), "gen-v9");
    let candidate = half_built.path().join("gen-v10");
    assert!(!candidate.join("VERSION").exists());
    let missing = production_extent_store_path(
        &candidate,
        TenantId::DEFAULT,
        arcgraph_storage::wal::STORE_GRANTS,
    )
    .unwrap();
    assert!(missing.is_file(), "invisible build omitted DEFAULT stores");
    fs::remove_file(&missing).unwrap();
    let mode = BootstrapMode::Durable {
        data_dir: half_built.path().to_path_buf(),
    };
    let (backend, guard) = bootstrap_storage_backend(&mode).unwrap();
    drop(backend);
    drop(guard);
    assert!(
        !candidate.join("VERSION").exists(),
        "half-built v6 generation was stamped before completeness validation"
    );
    assert!(!missing.exists());

    arcgraph_storage::stamp_data_dir(&candidate, DATA_DIR_VERSION_M4).unwrap();
    assert!(bootstrap_storage_backend(&mode).is_err());
    assert!(
        !missing.exists(),
        "bootstrap recreated a missing selected-generation store"
    );
}

#[test]
fn post_checkpoint_healthy_store_upgrade_returns_already_upgraded() {
    let fixture = tempdir().unwrap();
    production_v5_fixture(fixture.path());
    upgrade_quiesced_v5_to_v6(fixture.path(), MIGRATION_LSN, MigrationFault::None).unwrap();
    let generation = current_generation(fixture.path()).unwrap().unwrap();
    advance_with_production_incremental_checkpoint(&generation, Lsn::new(101));
    assert_eq!(
        arcgraph_storage::read_latest_sidecar(&generation)
            .unwrap()
            .unwrap()
            .checkpoint_lsn,
        Lsn::new(101)
    );
    assert_eq!(
        upgrade_quiesced_v5_to_v6(fixture.path(), Lsn::new(999), MigrationFault::None).unwrap(),
        MigrationOutcome::AlreadyUpgraded {
            migration_lsn: MIGRATION_LSN
        }
    );
}

#[test]
fn resume_verify_rejects_whole_tenant_loss() {
    let fixture = tempdir().unwrap();
    production_v5_fixture(fixture.path());
    upgrade_quiesced_v5_to_v6(fixture.path(), MIGRATION_LSN, MigrationFault::None).unwrap();
    let generation = current_generation(fixture.path()).unwrap().unwrap();
    fs::remove_dir_all(
        generation
            .join(arcgraph_storage::m3_migration::M3_TENANTS_DIR)
            .join(TENANTS[0].raw().to_string()),
    )
    .unwrap();
    let error = upgrade_quiesced_v5_to_v6(fixture.path(), Lsn::new(999), MigrationFault::None)
        .expect_err("a complete tenant disappeared from the selected v6 generation");
    assert!(
        format!("{error:#}").contains("tenant census"),
        "unexpected error: {error:#}"
    );
}

#[test]
fn resume_verify_rejects_header_only_checkpoint_metadata() {
    let fixture = tempdir().unwrap();
    production_v5_fixture(fixture.path());
    upgrade_quiesced_v5_to_v6(fixture.path(), MIGRATION_LSN, MigrationFault::None).unwrap();
    let generation = current_generation(fixture.path()).unwrap().unwrap();
    let checkpoint = arcgraph_storage::read_latest_sidecar(&generation)
        .unwrap()
        .unwrap();
    let metadata = incremental_metadata_path(
        &generation,
        checkpoint.checkpoint_lsn,
        checkpoint.metadata_generation,
    );
    assert!(fs::metadata(&metadata).unwrap().len() > 48);
    OpenOptions::new()
        .write(true)
        .open(&metadata)
        .unwrap()
        .set_len(48)
        .unwrap();
    let error = upgrade_quiesced_v5_to_v6(fixture.path(), MIGRATION_LSN, MigrationFault::None)
        .expect_err("header-only metadata must not satisfy resume verification");
    assert!(
        format!("{error:#}").contains("metadata checksum differs"),
        "unexpected error: {error:#}"
    );
}

#[test]
fn migrate_version_stamp_last_kill9_sweep() {
    for (name, pre_swap) in [
        ("scratch", true),
        ("sync", true),
        ("rename", true),
        ("current", true),
        ("version-dir-fsync", false),
        ("version", false),
    ] {
        let fixture = tempdir().unwrap();
        let source = if name == "version-dir-fsync" {
            production_v5_fixture(fixture.path())
        } else {
            v5_fixture(fixture.path())
        };
        let before = tree_bytes(&source);
        let status = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "m4_kill9_child", "--nocapture"])
            .env(CHILD_ROOT, fixture.path())
            .env(CHILD_FAULT, name)
            .status()
            .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            assert_eq!(status.signal(), Some(libc::SIGKILL), "fault={name}");
        }
        assert_eq!(
            tree_bytes(&source),
            before,
            "fault={name} mutated v5 source"
        );
        let selected = current_generation(fixture.path()).unwrap().unwrap();
        assert_eq!(
            selected.file_name().unwrap(),
            if pre_swap { "gen-v9" } else { "gen-v10" },
            "fault={name} selected a skewed generation"
        );
        if name == "current" {
            assert!(
                !fixture.path().join("gen-v10/VERSION").exists(),
                "VERSION was not last"
            );
        }
        if name == "version-dir-fsync" {
            assert!(
                selected.join("VERSION").exists(),
                "the injected parent-sync error must occur after VERSION rename"
            );
            fs::remove_file(selected.join("VERSION")).unwrap();
            File::open(&selected).unwrap().sync_all().unwrap();
        }
        let resumed =
            upgrade_quiesced_v5_to_v6(fixture.path(), MIGRATION_LSN, MigrationFault::None).unwrap();
        if name == "version" {
            assert_eq!(
                resumed,
                MigrationOutcome::AlreadyUpgraded {
                    migration_lsn: MIGRATION_LSN
                }
            );
        } else {
            assert_eq!(
                resumed,
                MigrationOutcome::Upgraded {
                    migration_lsn: MIGRATION_LSN
                }
            );
        }
        if name == "version-dir-fsync" {
            // Model the exact born-empty artifacts a first WAL open may make:
            // one record-free v10 segment-0 header plus optional wal.dek. Then
            // model a later power loss reverting VERSION while those WAL-dir
            // entries remain durable, and prove both migrate-rerun and serve
            // can reopen the generation.
            let mode = BootstrapMode::Durable {
                data_dir: fixture.path().to_path_buf(),
            };
            let wal_dir = selected.join("wal");
            let segment_zero = wal_dir.join(segment_filename(0));
            fs::write(
                &segment_zero,
                SegmentHeader {
                    format_version: BUNDLE_FORMAT_V10,
                }
                .encode(),
            )
            .unwrap();
            File::open(&segment_zero).unwrap().sync_all().unwrap();
            File::open(&wal_dir).unwrap().sync_all().unwrap();
            assert_eq!(
                fs::metadata(&segment_zero).unwrap().len(),
                SegmentHeader::SIZE as u64,
                "fixture WAL must contain a header and zero records"
            );
            fs::write(
                wal_dir.join(arcgraph_storage::WAL_DEK_SIDECAR_FILE),
                b"disabled-encryption-sidecar-fixture",
            )
            .unwrap();
            fs::remove_file(selected.join("VERSION")).unwrap();
            File::open(&selected).unwrap().sync_all().unwrap();

            let unexpected = wal_dir.join("not-born-empty");
            fs::write(&unexpected, b"reject me").unwrap();
            assert_eq!(
                upgrade_quiesced_v5_to_v6(fixture.path(), MIGRATION_LSN, MigrationFault::None)
                    .unwrap(),
                MigrationOutcome::Upgraded {
                    migration_lsn: MIGRATION_LSN
                }
            );
            assert!(
                !unexpected.exists(),
                "rollback retained an artifact from the unstamped candidate"
            );
            let (backend, guard) = bootstrap_storage_backend(&mode).unwrap();
            drop(backend);
            drop(guard);
        }
    }

    // Exercise production rollback: CURRENT contains the candidate name but
    // VERSION is absent, so bootstrap serves v5 and a later migration rebuilds.
    let resume = tempdir().unwrap();
    production_v5_fixture(resume.path());
    assert!(
        upgrade_quiesced_v5_to_v6(
            resume.path(),
            MIGRATION_LSN,
            MigrationFault::AfterCurrentSwap,
        )
        .is_err()
    );
    let selected = current_generation(resume.path()).unwrap().unwrap();
    assert_eq!(selected.file_name().unwrap(), "gen-v9");
    let candidate = resume.path().join("gen-v10");
    assert!(!candidate.join("VERSION").exists());
    let mode = BootstrapMode::Durable {
        data_dir: resume.path().to_path_buf(),
    };
    let (backend, guard) = bootstrap_storage_backend(&mode).unwrap();
    drop(backend);
    drop(guard);
    assert!(!candidate.join("VERSION").exists());
    let restart_lsn = arcgraph_storage::read_latest_sidecar(&selected)
        .unwrap()
        .unwrap()
        .checkpoint_lsn;
    assert_eq!(
        upgrade_quiesced_v5_to_v6(resume.path(), restart_lsn, MigrationFault::None).unwrap(),
        MigrationOutcome::Upgraded {
            migration_lsn: restart_lsn
        }
    );
    let selected = current_generation(resume.path()).unwrap().unwrap();
    assert_eq!(selected.file_name().unwrap(), "gen-v10");
    assert_eq!(
        arcgraph_storage::check_or_stamp_data_dir(&selected, true, false).unwrap(),
        DATA_DIR_VERSION_M4
    );
}

/// INV-M5.11 release gate for the implemented existing-tenant leg: two
/// non-default tenants cross real fsyncs and then die by SIGKILL. The child
/// leaves valid v10 deltas in its unpublished WAL; old-generation recovery
/// must ignore them, and the later production generation build must discard
/// rather than carry them across the atomic swap.
///
/// `cargo test -p arcgraph-cli --release --features fault-injection --test m4_data_dir_migration_gate wal_exposure_split_kill9_sweeps_and_stale_delta_control -- --exact --nocapture`
#[cfg(all(feature = "fault-injection", not(debug_assertions), unix))]
#[test]
fn wal_exposure_split_kill9_sweeps_and_stale_delta_control() {
    use std::os::unix::process::ExitStatusExt;

    assert!(TENANTS.iter().all(|tenant| *tenant != TenantId::DEFAULT));

    // Kill-9 + stale-delta control: the re-cluster child writes a real v10
    // delta into gen-v10.building and dies.  The selected v9 WAL must remain
    // byte-identical, old recovery must not observe label 177, and a clean
    // re-cluster must remove that candidate WAL instead of carrying it forward.
    let existing = tempdir().unwrap();
    let predecessor = production_v5_fixture(existing.path());
    let old_wal_before = tree_bytes(&predecessor.join("wal"));
    let status = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "wal_exposure_kill9_child", "--nocapture"])
        .env(WAL_EXPOSURE_ROOT, existing.path())
        .status()
        .unwrap();
    assert_eq!(
        status.signal(),
        Some(libc::SIGKILL),
        "existing-leg child exited before the kill-9 durability point: {status:?}"
    );
    assert_eq!(
        tree_bytes(&predecessor.join("wal")),
        old_wal_before,
        "re-cluster delta cross-contaminated the live tenant WAL"
    );
    assert!(
        tenant_load_wal_dir(
            existing.path(),
            TenantLoadLeg::ExistingRecluster {
                building_generation: &predecessor,
            },
        )
        .is_err(),
        "selected generation was accepted as a re-cluster WAL owner"
    );
    let candidate_wal = existing.path().join("gen-v10.building/wal");
    let candidate_records = WalRecoveryReader::open(&candidate_wal)
        .unwrap()
        .collect::<arcgraph_core::Result<Vec<_>>>()
        .unwrap();
    assert_eq!(candidate_records.len(), TENANTS.len());
    let candidate_tenants: BTreeSet<_> = candidate_records
        .iter()
        .map(|record| {
            let candidate_delta =
                decode_commit_bundle_v10(&record.payload, record.tenant_id).unwrap();
            assert_eq!(candidate_delta.deltas.len(), 1);
            assert_eq!(candidate_delta.deltas[0].tenant_id, record.tenant_id);
            record.tenant_id
        })
        .collect();
    assert_eq!(candidate_tenants, TENANTS.into_iter().collect());

    let old_mode = BootstrapMode::Durable {
        data_dir: existing.path().to_path_buf(),
    };
    let (backend, guard) = bootstrap_storage_backend(&old_mode).unwrap();
    let routed = backend
        .router()
        .route(TenantId::DEFAULT, arcgraph_core::PartitionId::ZERO)
        .unwrap();
    let reader = backend.txn_manager().begin(TENANTS[0]);
    let recovered = read_node_with_store(routed.crud(), &reader, NodeId::new(1))
        .unwrap()
        .expect("old generation lost its existing tenant");
    assert!(
        ![177, 178].contains(&recovered.label_id),
        "old recovery consumed the unpublished generation's stale delta"
    );
    reader.abort();
    drop(routed);
    drop(backend);
    drop(guard);

    assert!(matches!(
        upgrade_data_dir(existing.path()).unwrap(),
        MigrationOutcome::Upgraded { .. }
    ));
    let successor = current_generation(existing.path()).unwrap().unwrap();
    assert_eq!(successor.file_name().unwrap(), "gen-v10");
    assert!(
        WalRecoveryReader::open(successor.join("wal"))
            .unwrap()
            .next()
            .is_none(),
        "stale re-cluster WAL delta was carried into the committed generation"
    );
    let (backend, guard) = bootstrap_storage_backend(&old_mode).unwrap();
    let routed = backend
        .router()
        .route(TenantId::DEFAULT, arcgraph_core::PartitionId::ZERO)
        .unwrap();
    for tenant in TENANTS {
        let reader = backend.txn_manager().begin(tenant);
        let recovered = read_node_with_store(routed.crud(), &reader, NodeId::new(1))
            .unwrap()
            .expect("committed re-cluster lost an existing tenant");
        assert!(
            ![177, 178].contains(&recovered.label_id),
            "tenant={tenant:?}"
        );
        reader.abort();
    }
    drop(routed);
    drop(backend);
    drop(guard);
}

#[test]
fn v6_wal_empty_no_preswap_delta_replay() {
    let fixture = tempdir().unwrap();
    let source = production_v5_fixture(fixture.path());
    let source_wal_before = tree_bytes(&source.join("wal"));
    let old_records = WalRecoveryReader::open(source.join("wal"))
        .unwrap()
        .collect::<arcgraph_core::Result<Vec<_>>>()
        .unwrap();
    assert_eq!(
        old_records.len(),
        1,
        "fixture must carry a real stale v5 delta"
    );
    upgrade_quiesced_v5_to_v6(fixture.path(), MIGRATION_LSN, MigrationFault::None).unwrap();
    let generation = current_generation(fixture.path()).unwrap().unwrap();
    assert!(
        fs::read_dir(generation.join("wal"))
            .unwrap()
            .next()
            .is_none()
    );
    assert_eq!(
        tree_bytes(&source.join("wal")),
        source_wal_before,
        "migration truncated v5 WAL"
    );
    assert!(
        WalRecoveryReader::open(generation.join("wal"))
            .unwrap()
            .next()
            .is_none()
    );

    let mode = BootstrapMode::Durable {
        data_dir: fixture.path().to_path_buf(),
    };
    let (backend, guard) = bootstrap_storage_backend(&mode).unwrap();
    assert_eq!(
        guard.wal_handle().unwrap().format_version(),
        BUNDLE_FORMAT_V10,
        "fresh v6 WAL reopened with the pre-migration page-image codec"
    );
    let tenant = backend
        .router()
        .route(TenantId::DEFAULT, arcgraph_core::PartitionId::ZERO)
        .unwrap();
    let reader = backend.txn_manager().begin(TENANTS[0]);
    let migrated = read_node_with_store(tenant.crud(), &reader, NodeId::new(1))
        .unwrap()
        .expect("migrated record missing after fresh-WAL recovery");
    assert_eq!(
        migrated.label_id, 11,
        "stale v5 WAL delta collided with the v6 arithmetic slot"
    );
    reader.abort();
    drop(tenant);
    drop(backend);
    drop(guard);
}

#[test]
fn post_swap_blob_commit_and_extent_redo_reopen() {
    let fixture = tempdir().unwrap();
    production_v5_fixture(fixture.path());
    upgrade_quiesced_v5_to_v6(fixture.path(), MIGRATION_LSN, MigrationFault::None).unwrap();
    let generation = current_generation(fixture.path()).unwrap().unwrap();
    let mode = BootstrapMode::Durable {
        data_dir: fixture.path().to_path_buf(),
    };

    let (backend, guard) = bootstrap_storage_backend(&mode).unwrap();
    let property_bytes = b"post-swap non-default tenant property bag".to_vec();
    let (node_id, property_lsn) = commit_property_node(
        &backend,
        TENANTS[0],
        &PropertyData::Blob(property_bytes.clone()),
    );

    // Commit a production paired placement through the real v10 writer, then
    // deliberately skip live apply. Restart must use the same recovery DPT
    // for ExtentAlloc before applying its three PageAlloc operations.
    let first_op_lsn = Lsn::new(property_lsn.raw() + 1);
    let placement = guard
        .affinity_allocator(TENANTS[0])
        .unwrap()
        .place(17 * EXTENT_PAGES + 1, first_op_lsn)
        .unwrap();
    assert!(
        !placement.extent_allocs.is_empty(),
        "fixture failed to stage an unmapped ExtentAlloc"
    );
    let logical_extent = placement.property_page / EXTENT_PAGES;
    let extent_commit_lsn = placement.last_op_lsn();
    let extent_ops = placement.wal_ops();
    let extent_bundle = encode_commit_bundle_v10(
        extent_commit_lsn,
        TENANTS[0],
        &HashMap::new(),
        &[],
        &extent_ops,
        &[],
        &[],
        &[],
        &[],
        &[],
    )
    .unwrap();
    let wal = guard.wal_handle().unwrap();
    assert_eq!(
        wal.append_at(
            extent_commit_lsn,
            WalRecordType::CommitBundle,
            extent_commit_lsn.raw(),
            0,
            TENANTS[0],
            extent_bundle,
        )
        .unwrap(),
        extent_commit_lsn
    );
    drop(wal);
    drop(placement);
    drop(backend);
    drop(guard);

    let mut saw_prop_block = false;
    let mut saw_prop_page_alloc = false;
    let mut saw_extent_alloc = false;
    for record in WalRecoveryReader::open(generation.join("wal")).unwrap() {
        let record = record.unwrap();
        if record.record_type != WalRecordType::CommitBundle {
            continue;
        }
        for delta in decode_commit_bundle_v10(&record.payload, record.tenant_id)
            .unwrap()
            .deltas
        {
            saw_prop_block |= delta.tenant_id == TENANTS[0]
                && delta.store_id == STORE_PROPS
                && delta.kind == DeltaOpKind::PutPropBlock;
            saw_prop_page_alloc |= delta.tenant_id == TENANTS[0]
                && delta.store_id == STORE_PROPS
                && delta.kind == DeltaOpKind::PageAlloc;
            saw_extent_alloc |=
                delta.tenant_id == TENANTS[0] && delta.kind == DeltaOpKind::ExtentAlloc;
        }
    }
    assert!(saw_prop_block && saw_prop_page_alloc && saw_extent_alloc);

    let (backend, guard) = bootstrap_storage_backend(&mode).unwrap();
    assert_node_bag(&backend, TENANTS[0], node_id, &property_bytes);
    assert!(
        guard
            .extent_store(TENANTS[0], STORE_PROPS)
            .unwrap()
            .directory()
            .mapping(logical_extent)
            .unwrap()
            .is_some(),
        "committed ExtentAlloc was not installed by production replay"
    );
    drop(backend);
    drop(guard);
}

#[test]
fn post_swap_typed_block_commit_reopens() {
    let fixture = tempdir().unwrap();
    production_v5_fixture(fixture.path());
    upgrade_quiesced_v5_to_v6(fixture.path(), MIGRATION_LSN, MigrationFault::None).unwrap();
    let mode = BootstrapMode::Durable {
        data_dir: fixture.path().to_path_buf(),
    };
    let (typed, expected_block) = typed_mcp_shape();

    let (backend, guard) = bootstrap_storage_backend(&mode).unwrap();
    let (node_id, _) = commit_property_node(&backend, TENANTS[1], &typed);
    drop(backend);
    drop(guard);

    let (backend, guard) = bootstrap_storage_backend(&mode).unwrap();
    assert_node_bag(&backend, TENANTS[1], node_id, &expected_block);
    PropBlockView::parse(&expected_block).expect("reopened MCP-default typed block is corrupt");
    drop(backend);
    drop(guard);
}

#[test]
fn m4_physical_checkpoint_restart_byte_identical() {
    let fixture = tempdir().unwrap();
    production_v5_fixture(fixture.path());
    upgrade_quiesced_v5_to_v6(fixture.path(), MIGRATION_LSN, MigrationFault::None).unwrap();
    let generation = current_generation(fixture.path()).unwrap().unwrap();
    let mode = BootstrapMode::Durable {
        data_dir: fixture.path().to_path_buf(),
    };

    let (backend, guard) = bootstrap_storage_backend(&mode).unwrap();
    let blob = vec![0x4d; PROP_BAG_MAX_BYTES];
    let (blob_id, _) =
        commit_property_node(&backend, TENANTS[0], &PropertyData::Blob(blob.clone()));
    let (typed, typed_bytes) = typed_mcp_shape();
    let (typed_id, _) = commit_property_node(&backend, TENANTS[1], &typed);

    let owner_rows = Arc::clone(guard.owner_rows().unwrap());
    let idempotency = Arc::new(IdempotencyStore::page_backed(Arc::clone(&owner_rows)));
    idempotency
        .try_install_with_payload_hash(TENANTS[0], 1, "checkpoint-rel", 7_001, Some(0xc0de))
        .unwrap();
    let intern = InternTable::page_backed(Arc::clone(&owner_rows)).unwrap();
    let intern_id = intern_logged(
        &intern,
        &guard.wal_handle().unwrap(),
        TENANTS[1],
        "CheckpointCustomer",
    )
    .unwrap();
    let permissions = PermissionIndex::page_backed(
        Arc::clone(&owner_rows),
        Arc::clone(&idempotency),
        TENANTS[0],
    )
    .unwrap();
    permissions
        .try_apply_doc_acl(
            NodeId::new(7_002),
            ["alice".to_owned()].into_iter().collect(),
        )
        .unwrap();

    let dirty = guard
        .extent_store(TENANTS[0], STORE_RECORD)
        .unwrap()
        .dirty_page_table()
        .snapshot();
    let mut live_record_home = false;
    let mut observed_record_pages = Vec::new();
    for entry in dirty
        .iter()
        .filter(|entry| entry.key.tenant_id == TENANTS[0] && entry.key.store_id == STORE_RECORD)
    {
        let runtime = guard.extent_store(TENANTS[0], STORE_RECORD).unwrap();
        let page = arcgraph_storage::redo::DeltaPageStore::read_page_for_redo(
            runtime.data().as_ref(),
            TENANTS[0],
            PageId::new(entry.key.page_no),
        )
        .unwrap()
        .unwrap();
        let view = SlottedPageRef::open(page.as_ref()).unwrap();
        observed_record_pages.push((
            entry.key.page_no,
            view.header().page_type,
            view.iter_nodes()
                .map(|(_, record)| record.id)
                .collect::<Vec<_>>(),
        ));
        live_record_home |= view
            .iter_nodes()
            .any(|(_, record)| record.id == blob_id.raw());
    }
    assert!(
        live_record_home,
        "v10 Phase 3 omitted the Blob-shape record from its direct extent home: id={} pages={observed_record_pages:?} dirty={dirty:?}",
        blob_id.raw()
    );

    let checkpoint_lsn = guard
        .checkpointer()
        .expect("M4 physical generation omitted its incremental checkpointer")
        .checkpoint()
        .unwrap();
    assert!(checkpoint_lsn > MIGRATION_LSN);
    let sidecar = arcgraph_storage::read_latest_sidecar(&generation)
        .unwrap()
        .unwrap();
    assert_eq!(sidecar.checkpoint_lsn, checkpoint_lsn);
    assert!(sidecar.incremental_metadata);
    assert!(!sidecar.full_state_snapshot);
    {
        let txn = TxnManager::new();
        let addressed = AddressedRecordStore::new();
        let blobs = BlobStore::new();
        let report =
            load_v6_physical_base(&generation, checkpoint_lsn, &txn, &addressed, &blobs).unwrap();
        assert!(
            addressed.read_node(TENANTS[0], blob_id).unwrap().is_some(),
            "checkpoint home omitted the Blob-shape record; report={report:?} id={}",
            blob_id.raw()
        );
        assert!(
            addressed.read_node(TENANTS[1], typed_id).unwrap().is_some(),
            "checkpoint home omitted the TypedBlock-shape record"
        );
    }

    drop(permissions);
    drop(intern);
    drop(idempotency);
    drop(owner_rows);
    drop(backend);
    drop(guard);

    let (backend, guard) = bootstrap_storage_backend(&mode).unwrap();
    assert_node_bag(&backend, TENANTS[0], blob_id, &blob);
    assert_node_bag(&backend, TENANTS[1], typed_id, &typed_bytes);
    let owner_rows = Arc::clone(guard.owner_rows().unwrap());
    let idempotency = Arc::new(IdempotencyStore::page_backed(Arc::clone(&owner_rows)));
    assert_eq!(
        idempotency
            .try_get(TENANTS[0], 1, "checkpoint-rel")
            .unwrap()
            .unwrap()
            .internal_id,
        7_001
    );
    assert_eq!(
        idempotency
            .try_external_id_for(TENANTS[0], 1, 7_001)
            .unwrap()
            .as_deref(),
        Some("checkpoint-rel")
    );
    let intern = InternTable::page_backed(Arc::clone(&owner_rows)).unwrap();
    assert_eq!(
        intern.try_probe(TENANTS[1], "CheckpointCustomer").unwrap(),
        Some(intern_id)
    );
    assert_eq!(
        intern
            .try_resolve(TENANTS[1], intern_id)
            .unwrap()
            .as_deref()
            .map(String::as_str),
        Some("CheckpointCustomer")
    );
    let permissions = PermissionIndex::page_backed(
        Arc::clone(&owner_rows),
        Arc::clone(&idempotency),
        TENANTS[0],
    )
    .unwrap();
    assert!(
        permissions
            .effective("alice")
            .try_is_visible(NodeId::new(7_002))
            .unwrap()
    );
    assert_eq!(idempotency.resident_len(), 0);
    assert_eq!(idempotency.resident_reverse_len(), 0);
    assert_eq!(intern.resident_map_cardinalities(), [0, 0, 0]);
    assert_eq!(permissions.resident_map_cardinalities(), [0, 0, 0, 0, 0]);

    drop(permissions);
    drop(intern);
    drop(idempotency);
    drop(owner_rows);
    drop(backend);
    drop(guard);
}

#[cfg(unix)]
fn run_owner_provenance_child(root: &Path) -> ! {
    let mode = BootstrapMode::Durable {
        data_dir: root.to_path_buf(),
    };
    let (backend, guard) = bootstrap_storage_backend(&mode).unwrap();

    let blob = vec![0xa7; PROP_BAG_MAX_BYTES];
    let (blob_id, _) = commit_property_node(&backend, TENANTS[0], &PropertyData::Blob(blob));
    let (typed, _) = typed_mcp_shape();
    let (typed_id, _) = commit_property_node(&backend, TENANTS[1], &typed);

    let owner_rows = Arc::clone(
        guard
            .owner_rows()
            .expect("M4 durable bootstrap omitted owner-row registry"),
    );
    assert!(owner_rows.has_commit_runtime());
    for (index, class) in OwnerRowClass::ALL.into_iter().enumerate() {
        let tenant = TENANTS[index % TENANTS.len()];
        let id = 11 + index as u64;
        let row = OwnerRow::new(class, id, owner_provenance_payload(tenant, class, id)).unwrap();
        owner_rows.commit_row(tenant, row.clone()).unwrap();
        assert_eq!(owner_rows.read(tenant, class, id).unwrap(), Some(row));
    }

    // Exercise the logical facades that replace the ADR-229 metadata owners,
    // not only raw substrate rows. Every mutation below is a real v10 WAL
    // commit and none creates a record-cardinality resident mirror.
    let idempotency = Arc::new(IdempotencyStore::page_backed(Arc::clone(&owner_rows)));
    idempotency
        .try_install_with_payload_hash(TENANTS[0], 1, "durable-rel-binding", 1_001, Some(0x5a17))
        .unwrap();
    let intern = InternTable::page_backed(Arc::clone(&owner_rows)).unwrap();
    let intern_id = intern_logged(
        &intern,
        &guard.wal_handle().unwrap(),
        TENANTS[1],
        "DurableCustomer",
    )
    .unwrap();
    let permissions = PermissionIndex::page_backed(
        Arc::clone(&owner_rows),
        Arc::clone(&idempotency),
        TENANTS[0],
    )
    .unwrap();
    permissions
        .try_apply_doc_acl(
            NodeId::new(1_002),
            ["alice".to_owned()].into_iter().collect(),
        )
        .unwrap();
    assert_eq!(idempotency.resident_len(), 0);
    assert_eq!(idempotency.resident_reverse_len(), 0);
    assert_eq!(intern.resident_map_cardinalities(), [0, 0, 0]);
    assert_eq!(permissions.resident_map_cardinalities(), [0, 0, 0, 0, 0]);

    let acknowledged = root.join(OWNER_ACKED);
    fs::write(
        &acknowledged,
        format!("{} {} {}", blob_id.raw(), typed_id.raw(), intern_id.raw()),
    )
    .unwrap();
    File::open(&acknowledged).unwrap().sync_all().unwrap();

    let unacked = OwnerRow::new(
        OwnerRowClass::NodeBinding,
        OWNER_UNACKED_ID,
        owner_provenance_payload(TENANTS[0], OwnerRowClass::NodeBinding, OWNER_UNACKED_ID),
    )
    .unwrap();
    assert_eq!(
        owner_rows
            .read(TENANTS[0], OwnerRowClass::NodeBinding, OWNER_UNACKED_ID)
            .unwrap(),
        None
    );
    owner_rows.commit_row(TENANTS[0], unacked).unwrap();
    panic!("owner-row Phase-1 fault arm returned instead of waiting for SIGKILL");
}

#[cfg(unix)]
#[test]
fn owner_row_durable_provenance_across_restart() {
    if let Some(root) = std::env::var_os(OWNER_PROVENANCE_ROOT) {
        run_owner_provenance_child(Path::new(&root));
    }

    let fixture = tempdir().unwrap();
    production_v5_fixture(fixture.path());
    upgrade_quiesced_v5_to_v6(fixture.path(), MIGRATION_LSN, MigrationFault::None).unwrap();
    let ready = fixture.path().join(OWNER_PHASE1_READY);
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "owner_row_durable_provenance_across_restart",
            "--nocapture",
        ])
        .env(OWNER_PROVENANCE_ROOT, fixture.path())
        .env(
            "ARCGRAPH_OWNER_ROW_PHASE1_PAUSE_ID",
            OWNER_UNACKED_ID.to_string(),
        )
        .env("ARCGRAPH_OWNER_ROW_PHASE1_READY", &ready)
        .spawn()
        .unwrap();

    let deadline = Instant::now() + Duration::from_secs(30);
    let (blob_id, typed_id, intern_id) = loop {
        if let Ok(text) = fs::read_to_string(fixture.path().join(OWNER_ACKED))
            && ready.exists()
        {
            let ids: Vec<_> = text
                .split_whitespace()
                .map(|field| field.parse::<u64>().unwrap())
                .collect();
            if ids.len() == 3 {
                break (
                    NodeId::new(ids[0]),
                    NodeId::new(ids[1]),
                    arcgraph_core::StringId::new(ids[2] as u32),
                );
            }
        }
        assert!(
            child.try_wait().unwrap().is_none(),
            "owner provenance child exited before the Phase-1 kill window"
        );
        assert!(
            Instant::now() < deadline,
            "owner provenance child produced no acknowledged/Phase-1 state"
        );
        std::thread::sleep(Duration::from_millis(2));
    };

    let kill_result = unsafe { libc::kill(child.id() as i32, libc::SIGKILL) };
    assert_eq!(kill_result, 0, "kernel SIGKILL delivery failed");
    let status = child.wait().unwrap();
    use std::os::unix::process::ExitStatusExt;
    assert_eq!(status.signal(), Some(libc::SIGKILL));

    let mode = BootstrapMode::Durable {
        data_dir: fixture.path().to_path_buf(),
    };
    let (backend, guard) = bootstrap_storage_backend(&mode).unwrap();
    let owner_rows = guard
        .owner_rows()
        .expect("restarted M4 bootstrap omitted owner-row registry");
    for (index, class) in OwnerRowClass::ALL.into_iter().enumerate() {
        let tenant = TENANTS[index % TENANTS.len()];
        let id = 11 + index as u64;
        let expected =
            OwnerRow::new(class, id, owner_provenance_payload(tenant, class, id)).unwrap();
        assert_eq!(
            owner_rows.read(tenant, class, id).unwrap(),
            Some(expected),
            "acknowledged {class:?} owner row changed across SIGKILL restart"
        );
    }
    assert_eq!(
        owner_rows
            .read(TENANTS[0], OwnerRowClass::NodeBinding, OWNER_UNACKED_ID)
            .unwrap(),
        None,
        "unacknowledged owner row resurrected after SIGKILL restart"
    );

    let idempotency = Arc::new(IdempotencyStore::page_backed(Arc::clone(owner_rows)));
    let binding = idempotency
        .try_get(TENANTS[0], 1, "durable-rel-binding")
        .unwrap()
        .expect("acknowledged binding disappeared across SIGKILL restart");
    assert_eq!(binding.internal_id, 1_001);
    assert_eq!(binding.payload_hash, Some(0x5a17));
    assert_eq!(
        idempotency
            .try_external_id_for(TENANTS[0], 1, 1_001)
            .unwrap()
            .as_deref(),
        Some("durable-rel-binding")
    );
    let intern = InternTable::page_backed(Arc::clone(owner_rows)).unwrap();
    assert_eq!(
        intern.try_probe(TENANTS[1], "DurableCustomer").unwrap(),
        Some(intern_id)
    );
    assert_eq!(
        intern
            .try_resolve(TENANTS[1], intern_id)
            .unwrap()
            .as_deref()
            .map(String::as_str),
        Some("DurableCustomer")
    );
    let permissions =
        PermissionIndex::page_backed(Arc::clone(owner_rows), Arc::clone(&idempotency), TENANTS[0])
            .unwrap();
    assert!(
        permissions
            .effective("alice")
            .try_is_visible(NodeId::new(1_002))
            .unwrap()
    );
    assert!(!permissions.effective("bob").is_visible(NodeId::new(1_002)));
    assert_eq!(idempotency.resident_len(), 0);
    assert_eq!(idempotency.resident_reverse_len(), 0);
    assert_eq!(intern.resident_map_cardinalities(), [0, 0, 0]);
    assert_eq!(permissions.resident_map_cardinalities(), [0, 0, 0, 0, 0]);

    assert_node_bag(
        &backend,
        TENANTS[0],
        blob_id,
        &vec![0xa7; PROP_BAG_MAX_BYTES],
    );
    let (_, expected_typed) = typed_mcp_shape();
    assert_node_bag(&backend, TENANTS[1], typed_id, &expected_typed);
    PropBlockView::parse(&expected_typed).expect("TypedBlock control shape is malformed");
    drop(backend);
    drop(guard);
}

#[test]
fn multi_extent_plan_offsets_unique_dense_under_commit() {
    let fixture = tempdir().unwrap();
    production_v5_fixture(fixture.path());
    upgrade_quiesced_v5_to_v6(fixture.path(), MIGRATION_LSN, MigrationFault::None).unwrap();
    let mode = BootstrapMode::Durable {
        data_dir: fixture.path().to_path_buf(),
    };
    let (backend, guard) = bootstrap_storage_backend(&mode).unwrap();
    assert_eq!(
        guard.wal_handle().unwrap().format_version(),
        BUNDLE_FORMAT_V10
    );

    let blob = vec![0x6b; PROP_BAG_MAX_BYTES];
    let (blob_id, _) = commit_property_node(&backend, TENANTS[0], &PropertyData::Blob(blob));
    let (typed, expected_typed) = typed_mcp_shape();
    let (typed_id, _) = commit_property_node(&backend, TENANTS[1], &typed);

    let registry = Arc::clone(guard.owner_rows().unwrap());
    let initial_extents = [2, 7, 11];
    let initial = owner_rows_for_extents(TENANTS[0], OwnerRowClass::NodeBinding, &initial_extents);
    registry.commit_rows(TENANTS[0], initial.clone()).unwrap();
    let mut expected: Vec<_> = initial.into_iter().map(|row| (TENANTS[0], row)).collect();

    let batches = vec![
        (
            TENANTS[0],
            owner_rows_for_extents(TENANTS[0], OwnerRowClass::NodeBinding, &[20, 21]),
        ),
        (
            TENANTS[0],
            owner_rows_for_extents(TENANTS[0], OwnerRowClass::NodeBinding, &[24, 25]),
        ),
        (
            TENANTS[1],
            owner_rows_for_extents(TENANTS[1], OwnerRowClass::NodeBinding, &[30, 31]),
        ),
        (
            TENANTS[1],
            owner_rows_for_extents(TENANTS[1], OwnerRowClass::NodeBinding, &[34, 35]),
        ),
    ];
    let start = Arc::new(Barrier::new(batches.len()));
    std::thread::scope(|scope| {
        let mut joins = Vec::new();
        for (tenant, rows) in &batches {
            let registry = Arc::clone(&registry);
            let start = Arc::clone(&start);
            let rows = rows.clone();
            let tenant = *tenant;
            joins.push(scope.spawn(move || {
                start.wait();
                registry.commit_rows(tenant, rows).unwrap();
            }));
        }
        for join in joins {
            join.join().unwrap();
        }
    });
    expected.extend(
        batches
            .iter()
            .flat_map(|(tenant, rows)| rows.iter().cloned().map(|row| (*tenant, row))),
    );
    assert_owner_rows(&registry, &expected);

    let tenant_41_extents = BTreeSet::from([2, 7, 11, 20, 21, 24, 25]);
    let tenant_73_extents = BTreeSet::from([30, 31, 34, 35]);
    assert_dense_owner_mappings(&guard, TENANTS[0], &tenant_41_extents);
    assert_dense_owner_mappings(&guard, TENANTS[1], &tenant_73_extents);

    drop(registry);
    drop(backend);
    drop(guard);

    let (backend, guard) = bootstrap_storage_backend(&mode).unwrap();
    let registry = guard.owner_rows().unwrap();
    assert_owner_rows(registry, &expected);
    assert_dense_owner_mappings(&guard, TENANTS[0], &tenant_41_extents);
    assert_dense_owner_mappings(&guard, TENANTS[1], &tenant_73_extents);
    assert_node_bag(
        &backend,
        TENANTS[0],
        blob_id,
        &vec![0x6b; PROP_BAG_MAX_BYTES],
    );
    assert_node_bag(&backend, TENANTS[1], typed_id, &expected_typed);
    drop(backend);
    drop(guard);
}

#[test]
fn owner_row_wal_replay_idempotent() {
    let fixture = tempdir().unwrap();
    production_v5_fixture(fixture.path());
    upgrade_quiesced_v5_to_v6(fixture.path(), MIGRATION_LSN, MigrationFault::None).unwrap();
    let mode = BootstrapMode::Durable {
        data_dir: fixture.path().to_path_buf(),
    };
    let (backend, guard) = bootstrap_storage_backend(&mode).unwrap();
    assert_eq!(
        guard.wal_handle().unwrap().format_version(),
        BUNDLE_FORMAT_V10
    );

    let blob = vec![0xd3; PROP_BAG_MAX_BYTES];
    let (blob_id, _) = commit_property_node(&backend, TENANTS[0], &PropertyData::Blob(blob));
    let (typed, expected_typed) = typed_mcp_shape();
    let (typed_id, _) = commit_property_node(&backend, TENANTS[1], &typed);

    let registry = Arc::clone(guard.owner_rows().unwrap());
    let mut tenant_41_rows: Vec<_> = OwnerRowClass::ALL
        .into_iter()
        .enumerate()
        .map(|(index, class)| {
            let id = 3 + index as u64;
            OwnerRow::new(class, id, owner_provenance_payload(TENANTS[0], class, id)).unwrap()
        })
        .collect();
    let same_page_id = 25;
    tenant_41_rows.push(
        OwnerRow::new(
            OwnerRowClass::NodeBinding,
            same_page_id,
            owner_provenance_payload(TENANTS[0], OwnerRowClass::NodeBinding, same_page_id),
        )
        .unwrap(),
    );
    registry
        .commit_rows(TENANTS[0], tenant_41_rows.clone())
        .unwrap();
    let tenant_73_rows = vec![
        OwnerRow::new(
            OwnerRowClass::RelBinding,
            17,
            owner_provenance_payload(TENANTS[1], OwnerRowClass::RelBinding, 17),
        )
        .unwrap(),
        OwnerRow::new(
            OwnerRowClass::Grant,
            19,
            owner_provenance_payload(TENANTS[1], OwnerRowClass::Grant, 19),
        )
        .unwrap(),
    ];
    registry
        .commit_rows(TENANTS[1], tenant_73_rows.clone())
        .unwrap();
    let expected: Vec<_> = tenant_41_rows
        .into_iter()
        .map(|row| (TENANTS[0], row))
        .chain(tenant_73_rows.into_iter().map(|row| (TENANTS[1], row)))
        .collect();
    assert_owner_rows(&registry, &expected);
    let live_pages = snapshot_owner_pages(&guard, &expected);

    drop(registry);
    drop(backend);
    drop(guard);

    let (backend, guard) = bootstrap_storage_backend(&mode).unwrap();
    assert_owner_rows(guard.owner_rows().unwrap(), &expected);
    let first_replay = snapshot_owner_pages(&guard, &expected);
    assert_eq!(first_replay, live_pages, "first replay changed owner bytes");
    assert_node_bag(
        &backend,
        TENANTS[0],
        blob_id,
        &vec![0xd3; PROP_BAG_MAX_BYTES],
    );
    assert_node_bag(&backend, TENANTS[1], typed_id, &expected_typed);
    drop(backend);
    drop(guard);

    let (backend, guard) = bootstrap_storage_backend(&mode).unwrap();
    assert_owner_rows(guard.owner_rows().unwrap(), &expected);
    let second_replay = snapshot_owner_pages(&guard, &expected);
    assert_eq!(
        second_replay, first_replay,
        "second replay changed owner bytes"
    );
    assert_node_bag(
        &backend,
        TENANTS[0],
        blob_id,
        &vec![0xd3; PROP_BAG_MAX_BYTES],
    );
    assert_node_bag(&backend, TENANTS[1], typed_id, &expected_typed);
    drop(backend);
    drop(guard);
}

#[cfg(unix)]
fn run_prop_sigkill_child(root: &Path) -> ! {
    let mode = BootstrapMode::Durable {
        data_dir: root.to_path_buf(),
    };
    let (backend, _guard) = bootstrap_storage_backend(&mode).unwrap();
    let routed = backend
        .router()
        .route(TenantId::DEFAULT, arcgraph_core::PartitionId::ZERO)
        .unwrap();
    let marker = root.join("PROP_COMMITTING");
    let mut ledger = OpenOptions::new()
        .create(true)
        .append(true)
        .open(root.join("PROP_ACKED"))
        .unwrap();
    let mut batch = 0_u64;
    loop {
        let fill = 0x40_u8.wrapping_add(batch as u8);
        let property = PropertyData::Blob(vec![fill; PROP_BAG_MAX_BYTES]);
        let mut tx = backend.txn_manager().begin(TENANTS[0]);
        let mut first_id = None;
        for _ in 0..PROP_SIGKILL_BATCH {
            let id = create_node(
                routed.crud(),
                &mut tx,
                TENANTS[0],
                LabelId::new(101),
                &property,
            )
            .unwrap();
            first_id.get_or_insert(id);
        }
        fs::write(&marker, batch.to_string()).unwrap();
        File::open(&marker).unwrap().sync_all().unwrap();
        commit(tx, routed.crud()).unwrap();
        fs::remove_file(&marker).unwrap();
        writeln!(ledger, "{batch} {} {fill}", first_id.unwrap().raw()).unwrap();
        ledger.sync_all().unwrap();
        batch += 1;
    }
}

#[cfg(unix)]
#[test]
fn post_swap_blob_sigkill_mid_commit_reopens() {
    if std::env::var_os(PROP_SIGKILL_CHILD).is_some() {
        run_prop_sigkill_child(Path::new(&std::env::var_os(PROP_SIGKILL_ROOT).unwrap()));
    }

    let fixture = tempdir().unwrap();
    production_v5_fixture(fixture.path());
    upgrade_quiesced_v5_to_v6(fixture.path(), MIGRATION_LSN, MigrationFault::None).unwrap();
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "post_swap_blob_sigkill_mid_commit_reopens",
            "--nocapture",
        ])
        .env(PROP_SIGKILL_CHILD, "1")
        .env(PROP_SIGKILL_ROOT, fixture.path())
        .spawn()
        .unwrap();

    let deadline = Instant::now() + Duration::from_secs(30);
    let (acked_batch, acked_id, acked_fill) = loop {
        if let Ok(text) = fs::read_to_string(fixture.path().join("PROP_ACKED"))
            && let Some(line) = text.lines().next()
        {
            let fields: Vec<_> = line.split_whitespace().collect();
            if fields.len() == 3 {
                break (
                    fields[0].parse::<u64>().unwrap(),
                    NodeId::new(fields[1].parse::<u64>().unwrap()),
                    fields[2].parse::<u8>().unwrap(),
                );
            }
        }
        assert!(
            child.try_wait().unwrap().is_none(),
            "property child exited before one strict commit was acknowledged"
        );
        assert!(Instant::now() < deadline, "property child produced no ACK");
        std::thread::sleep(Duration::from_millis(2));
    };

    loop {
        if let Ok(text) = fs::read_to_string(fixture.path().join("PROP_COMMITTING"))
            && text
                .trim()
                .parse::<u64>()
                .is_ok_and(|batch| batch > acked_batch)
        {
            break;
        }
        assert!(
            child.try_wait().unwrap().is_none(),
            "property child exited before the kill window"
        );
        assert!(
            Instant::now() < deadline,
            "no mid-commit kill window observed"
        );
        std::thread::sleep(Duration::from_micros(200));
    }

    let kill_result = unsafe { libc::kill(child.id() as i32, libc::SIGKILL) };
    assert_eq!(kill_result, 0, "kernel SIGKILL delivery failed");
    let status = child.wait().unwrap();
    use std::os::unix::process::ExitStatusExt;
    assert_eq!(status.signal(), Some(libc::SIGKILL));

    let mode = BootstrapMode::Durable {
        data_dir: fixture.path().to_path_buf(),
    };
    let (backend, guard) = bootstrap_storage_backend(&mode).unwrap();
    assert_node_bag(
        &backend,
        TENANTS[0],
        acked_id,
        &vec![acked_fill; PROP_BAG_MAX_BYTES],
    );
    drop(backend);
    drop(guard);
}

fn expected_node_page(tenant: TenantId, page_no: u64) -> [u8; PAGE_SIZE] {
    let mut bytes = [0_u8; PAGE_SIZE];
    let mut header = PageHeader::new(PageId::new(page_no), PageType::Node, tenant);
    header.lsn = MIGRATION_LSN.raw();
    let mut page = SlottedPage::init(&mut bytes, header).unwrap();
    page.write_node_at_slot(SlotId(1), &node(1, 11, 40))
        .unwrap();
    page.permanent_tombstone_node_at_slot(SlotId(2), MIGRATION_LSN)
        .unwrap();
    page.write_node_at_slot(SlotId(3), &node(3, 13, 42))
        .unwrap();
    bytes
}

fn expected_high_node_page(tenant: TenantId) -> [u8; PAGE_SIZE] {
    let high = u64::from(NODE_CAPACITY) * 256 + 1;
    let mut bytes = [0_u8; PAGE_SIZE];
    let mut header = PageHeader::new(PageId::new(256), PageType::Node, tenant);
    header.lsn = MIGRATION_LSN.raw();
    SlottedPage::init(&mut bytes, header)
        .unwrap()
        .write_node_at_slot(SlotId(1), &node(high, 14, 43))
        .unwrap();
    bytes
}

fn expected_rel_page(tenant: TenantId) -> [u8; PAGE_SIZE] {
    let mut bytes = [0_u8; PAGE_SIZE];
    let mut header = PageHeader::new(PageId::new(0), PageType::Rel, tenant);
    header.lsn = MIGRATION_LSN.raw();
    let record = RelRecord::new(
        RelId::new(1),
        TypeId::new(9),
        NodeId::new(1),
        NodeId::new(3),
        Lsn::new(44),
    );
    let mut page = SlottedPage::init(&mut bytes, header).unwrap();
    page.write_rel_at_slot(SlotId(1), &record).unwrap();
    page.permanent_tombstone_rel_at_slot(SlotId(2), MIGRATION_LSN)
        .unwrap();
    bytes
}

fn expected_prop_page(tenant: TenantId, page_no: u64) -> [u8; PAGE_SIZE] {
    let mut bytes = prop_page(tenant, page_no);
    SlottedPage::open(&mut bytes)
        .unwrap()
        .apply_redo_if_newer(MIGRATION_LSN, |_| Ok::<(), std::convert::Infallible>(()))
        .unwrap();
    bytes
}

/// INV-M5.8: compare the M5 producer with the independent pre-existing
/// M4-lite producer, then retain the hand-encoded page oracle below. The
/// fixture includes non-default tenants, empty stores, permanent node/rel
/// tombstones, gap ids, and property pages. A loader-only encoding change
/// makes cargo test exit 101.
#[test]
fn loader_vs_m4lite_differential() {
    let fixture = tempdir().unwrap();
    let source = v5_fixture(fixture.path());
    let frontier = LoaderMigrationFrontier::new(MIGRATION_LSN).unwrap();
    let reference = fixture.path().join("m4lite-reference");
    fs::create_dir(&reference).unwrap();
    load_v5_generation(&source, &reference, frontier, LoaderTarget::M4LiteReference).unwrap();
    let reference_bytes = tree_bytes(&reference);
    for (name, target) in [
        ("fresh", LoaderTarget::Fresh),
        ("existing-recluster", LoaderTarget::ExistingRecluster),
    ] {
        let destination = fixture.path().join(format!("m5-{name}"));
        fs::create_dir(&destination).unwrap();
        load_v5_generation(&source, &destination, frontier, target).unwrap();
        assert!(
            tree_bytes(&destination) == reference_bytes,
            "M5 {name} bytes differ from the independent M4-lite producer"
        );
    }

    upgrade_quiesced_v5_to_v6(fixture.path(), MIGRATION_LSN, MigrationFault::None).unwrap();
    let generation = current_generation(fixture.path()).unwrap().unwrap();
    let manifest = arcgraph_storage::read_data_dir_manifest(&generation)
        .unwrap()
        .unwrap();
    assert_eq!(manifest.record_store_format, RECORD_FORMAT_DIRECT_M4);

    for store_id in M4_EXTENT_STORE_IDS {
        let path = production_extent_store_path(&generation, TenantId::DEFAULT, *store_id).unwrap();
        assert!(path.is_file(), "DEFAULT v6 store is missing: {store_id}");
        assert!(
            read_extent_ledger(&path, TenantId::DEFAULT, *store_id)
                .unwrap()
                .is_empty(),
            "DEFAULT fixture store unexpectedly contains extents: {store_id}"
        );
    }

    for tenant in TENANTS {
        for store_id in M4_EXTENT_STORE_IDS {
            let path = production_extent_store_path(&generation, tenant, *store_id).unwrap();
            assert!(path.is_file());
            if !matches!(*store_id, STORE_PROPS | STORE_RECORD | STORE_RELS) {
                assert!(
                    read_extent_ledger(&path, tenant, *store_id)
                        .unwrap()
                        .is_empty(),
                    "fixture's empty v6 store unexpectedly gained pages: tenant={tenant:?} store={store_id}"
                );
            }
        }
        let nodes_path = production_extent_store_path(&generation, tenant, STORE_RECORD).unwrap();
        let ledger = read_extent_ledger(&nodes_path, tenant, STORE_RECORD).unwrap();
        assert_eq!(ledger.len(), 2);
        assert_eq!(ledger[0].physical_offset, DIRECTORY_HEAD_BYTES);
        assert_eq!(
            ledger[1].physical_offset,
            DIRECTORY_HEAD_BYTES + EXTENT_BYTES
        );
        assert_ne!(ledger[0].physical_offset, ledger[1].physical_offset);

        let mut file = File::open(nodes_path).unwrap();
        let mut actual = [0_u8; PAGE_SIZE];
        file.seek(SeekFrom::Start(DIRECTORY_HEAD_BYTES)).unwrap();
        file.read_exact(&mut actual).unwrap();
        assert_eq!(actual, expected_node_page(tenant, 0), "tenant={tenant:?}");
        let view = SlottedPageRef::open(&actual).unwrap();
        assert_eq!(view.page_lsn(), MIGRATION_LSN);
        assert_eq!(
            view.read_node(SlotId(2)).unwrap(),
            None,
            "deleted id must not be readable"
        );
        assert!(
            view.is_permanent_tombstone(SlotId(2)).unwrap(),
            "deleted id must retain durable provenance"
        );

        file.seek(SeekFrom::Start(DIRECTORY_HEAD_BYTES + EXTENT_BYTES))
            .unwrap();
        file.read_exact(&mut actual).unwrap();
        assert_eq!(actual, expected_high_node_page(tenant));

        let rels_path = production_extent_store_path(&generation, tenant, STORE_RELS).unwrap();
        let mut rels_file = File::open(rels_path).unwrap();
        rels_file
            .seek(SeekFrom::Start(DIRECTORY_HEAD_BYTES))
            .unwrap();
        rels_file.read_exact(&mut actual).unwrap();
        assert_eq!(actual, expected_rel_page(tenant));
        let rel_view = SlottedPageRef::open(&actual).unwrap();
        assert_eq!(rel_view.read_rel(SlotId(2)).unwrap(), None);
        assert!(rel_view.is_permanent_tombstone(SlotId(2)).unwrap());

        let prop_page_no = if tenant == TENANTS[0] { 0 } else { 1 };
        let props_path = production_extent_store_path(&generation, tenant, STORE_PROPS).unwrap();
        let mut props_file = File::open(props_path).unwrap();
        props_file
            .seek(SeekFrom::Start(
                DIRECTORY_HEAD_BYTES + prop_page_no * PAGE_SIZE as u64,
            ))
            .unwrap();
        props_file.read_exact(&mut actual).unwrap();
        assert_eq!(actual, expected_prop_page(tenant, prop_page_no));
    }

    let txn = TxnManager::new();
    let addressed = AddressedRecordStore::new();
    let blobs = BlobStore::new();
    let loaded =
        load_v6_physical_base(&generation, MIGRATION_LSN, &txn, &addressed, &blobs).unwrap();
    assert_eq!(loaded.nodes, 6);
    assert_eq!(loaded.rels, 2);
    for tenant in TENANTS {
        assert!(
            addressed
                .read_node(tenant, NodeId::new(1))
                .unwrap()
                .is_some()
        );
        assert_eq!(addressed.read_node(tenant, NodeId::new(2)).unwrap(), None);
        assert!(
            addressed
                .read_node(tenant, NodeId::new(3))
                .unwrap()
                .is_some()
        );
        assert!(addressed.read_rel(tenant, RelId::new(1)).unwrap().is_some());
        assert_eq!(addressed.read_rel(tenant, RelId::new(2)).unwrap(), None);
    }

    // Keep the named differential in the production regime: attach the
    // generated extent stores, append through the real v10 WAL, then recover
    // that post-attach commit in a fresh production bootstrap.
    let wal_fixture = tempdir().unwrap();
    production_v5_fixture(wal_fixture.path());
    upgrade_quiesced_v5_to_v6(wal_fixture.path(), MIGRATION_LSN, MigrationFault::None).unwrap();
    let mode = BootstrapMode::Durable {
        data_dir: wal_fixture.path().to_path_buf(),
    };
    let (backend, guard) = bootstrap_storage_backend(&mode).unwrap();
    let routed = backend
        .router()
        .route(TenantId::DEFAULT, arcgraph_core::PartitionId::ZERO)
        .unwrap();
    let mut writer = backend.txn_manager().begin(TENANTS[0]);
    let wal_backed = create_node(
        routed.crud(),
        &mut writer,
        TENANTS[0],
        LabelId::new(88),
        &PropertyData::InlineU32Pair(8, 9),
    )
    .unwrap();
    commit(writer, routed.crud()).unwrap();
    drop(routed);
    drop(backend);
    drop(guard);

    let (backend, guard) = bootstrap_storage_backend(&mode).unwrap();
    let routed = backend
        .router()
        .route(TenantId::DEFAULT, arcgraph_core::PartitionId::ZERO)
        .unwrap();
    let reader = backend.txn_manager().begin(TENANTS[0]);
    assert!(
        read_node_with_store(routed.crud(), &reader, wal_backed)
            .unwrap()
            .is_some(),
        "real-WAL post-attach commit did not recover over differential bytes"
    );
    reader.abort();
    drop(routed);
    drop(backend);
    drop(guard);
}

/// INV-M5.8 RED-on-revert (M5-D2, amendment §7 gate table): a LOADER-ONLY
/// encoding divergence — modeled by the cfg-gated
/// `ARCGRAPH_M5_DROP_LOADER_PROPS` seam, which makes only the M5 producer
/// skip the property pages — must redden `loader_vs_m4lite_differential`
/// over the property-bearing fixture. Runs in the CI release
/// fault-injection lane alongside the other armed controls.
#[cfg(feature = "fault-injection")]
#[test]
fn loader_props_drop_reddens_m4lite_differential() {
    if std::env::var_os("ARCGRAPH_M5_DROP_LOADER_PROPS").is_some() {
        // Never recurse when this process IS the armed child.
        return;
    }
    let output = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "loader_vs_m4lite_differential", "--nocapture"])
        .env("ARCGRAPH_M5_DROP_LOADER_PROPS", "1")
        .output()
        .unwrap();
    assert!(
        !output.status.success(),
        "loader_vs_m4lite_differential stayed GREEN while the M5 producer \
         dropped its property pages — the INV-M5.8 differential is a no-op\nstdout:\n{}",
        String::from_utf8_lossy(&output.stdout),
    );
}

/// INV-M5.9 production-regime gate. The table is the ratified total domain
/// for both record classes. It exercises the exact function used by M5's
/// canonical writers, then performs a two-non-default-tenant re-cluster and a
/// real-WAL append/recovery. Replacing the loader link with a local off-by-one
/// formula makes cargo test exit 101.
#[test]
fn loader_address_uses_total_derivation() {
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
    for tenant in TENANTS {
        assert_ne!(tenant, TenantId::DEFAULT);
        for (id, expected) in node_vectors {
            assert_eq!(
                loader_record_address(RecordKind::Node, id),
                expected,
                "node tenant={tenant:?} id={id}"
            );
        }
        for (id, expected) in rel_vectors {
            assert_eq!(
                loader_record_address(RecordKind::Rel, id),
                expected,
                "rel tenant={tenant:?} id={id}"
            );
        }
        for kind in [RecordKind::Node, RecordKind::Rel] {
            assert_eq!(
                loader_record_address(kind, arcgraph_storage::crud::REL_TAG_BIT | 1),
                Err(AddressError::OutOfRange),
                "tagged id accepted for tenant={tenant:?} kind={kind:?}"
            );
        }
    }

    let fixture = tempdir().unwrap();
    production_v5_fixture(fixture.path());
    upgrade_quiesced_v5_to_v6(fixture.path(), MIGRATION_LSN, MigrationFault::None).unwrap();
    let mode = BootstrapMode::Durable {
        data_dir: fixture.path().to_path_buf(),
    };
    let (backend, guard) = bootstrap_storage_backend(&mode).unwrap();
    let routed = backend
        .router()
        .route(TenantId::DEFAULT, arcgraph_core::PartitionId::ZERO)
        .unwrap();
    let mut writer = backend.txn_manager().begin(TENANTS[1]);
    let node_id = create_node(
        routed.crud(),
        &mut writer,
        TENANTS[1],
        LabelId::new(89),
        &PropertyData::InlineU32Pair(13, 21),
    )
    .unwrap();
    commit(writer, routed.crud()).unwrap();
    drop(routed);
    drop(backend);
    drop(guard);

    let generation = current_generation(fixture.path()).unwrap().unwrap();
    assert!(
        fs::read_dir(generation.join("wal"))
            .unwrap()
            .next()
            .is_some(),
        "production address gate did not append through the real WAL"
    );
    let (backend, guard) = bootstrap_storage_backend(&mode).unwrap();
    let routed = backend
        .router()
        .route(TenantId::DEFAULT, arcgraph_core::PartitionId::ZERO)
        .unwrap();
    let reader = backend.txn_manager().begin(TENANTS[1]);
    assert!(
        read_node_with_store(routed.crud(), &reader, node_id)
            .unwrap()
            .is_some(),
        "addressed post-attach node did not survive real-WAL recovery"
    );
    reader.abort();
    drop(routed);
    drop(backend);
    drop(guard);
}

/// INV-M5.14 production-regime gate. Re-cluster must preserve the raw id as
/// identity: live bytes remain at the same id, retired ids remain permanent
/// tombstones, and never-issued gaps remain allocation-free NotFound results.
/// The dangling-reference check is the M4 safety rerun against the migrated
/// tombstone. Compacting id 2 and shifting id 3 into its hole makes cargo test
/// exit 101.
#[test]
fn recluster_preserves_ids_and_tombstones() {
    let fixture = tempdir().unwrap();
    production_v5_fixture(fixture.path());
    upgrade_quiesced_v5_to_v6(fixture.path(), MIGRATION_LSN, MigrationFault::None).unwrap();
    let generation = current_generation(fixture.path()).unwrap().unwrap();

    let txn = TxnManager::new();
    let addressed = Arc::new(AddressedRecordStore::new());
    let loaded = load_v6_physical_base(
        &generation,
        MIGRATION_LSN,
        &txn,
        addressed.as_ref(),
        &BlobStore::new(),
    )
    .unwrap();
    assert_eq!(loaded.nodes, 6);
    assert_eq!(loaded.rels, 2);
    let high = u64::from(NODE_CAPACITY) * 256 + 1;

    for tenant in TENANTS {
        assert_ne!(tenant, TenantId::DEFAULT);
        for expected in [node(1, 11, 40), node(3, 13, 42), node(high, 14, 43)] {
            let actual = addressed
                .read_node(tenant, NodeId::new(expected.id))
                .unwrap()
                .unwrap_or_else(|| panic!("node id {} moved or disappeared", expected.id));
            assert_eq!(
                actual.to_bytes(),
                expected.to_bytes(),
                "node id {} was renumbered or rewritten for tenant={tenant:?}",
                expected.id
            );
        }
        let expected_rel = RelRecord::new(
            RelId::new(1),
            TypeId::new(9),
            NodeId::new(1),
            NodeId::new(3),
            Lsn::new(44),
        );
        assert_eq!(
            addressed
                .read_rel(tenant, RelId::new(1))
                .unwrap()
                .unwrap()
                .to_bytes(),
            expected_rel.to_bytes(),
            "relationship id 1 moved for tenant={tenant:?}"
        );

        assert_eq!(addressed.read_node(tenant, NodeId::new(2)).unwrap(), None);
        assert_eq!(addressed.read_rel(tenant, RelId::new(2)).unwrap(), None);
        assert!(matches!(
            addressed.write_node(tenant, &node(2, 99, 1_000)),
            Err(AddressedStoreError::PermanentTombstone { id: 2, .. })
        ));
        let resurrected_rel = RelRecord::new(
            RelId::new(2),
            TypeId::new(99),
            NodeId::new(1),
            NodeId::new(3),
            Lsn::new(1_000),
        );
        assert!(matches!(
            addressed.write_rel(tenant, &resurrected_rel),
            Err(AddressedStoreError::PermanentTombstone { id: 2, .. })
        ));

        let node_pages = addressed.page_count(tenant, RecordKind::Node);
        let rel_pages = addressed.page_count(tenant, RecordKind::Rel);
        for _ in 0..2 {
            assert_eq!(addressed.read_node(tenant, NodeId::new(4)).unwrap(), None);
            assert_eq!(addressed.read_rel(tenant, RelId::new(3)).unwrap(), None);
        }
        assert_eq!(addressed.page_count(tenant, RecordKind::Node), node_pages);
        assert_eq!(addressed.page_count(tenant, RecordKind::Rel), rel_pages);

        // Re-run `dangling_ref_resolves_to_tombstone` against the migrated
        // hole. The TEL-shaped reference retains raw source id 2 while a
        // recovered allocator issues strictly above the source high-water.
        let dangling_tel_ref = RelRecord::new(
            RelId::new(77),
            TypeId::new(9),
            NodeId::new(2),
            NodeId::new(3),
            Lsn::new(1),
        );
        let recovered = CrudStore::new().with_addressed_record_store(Arc::clone(&addressed));
        recovered.apply_allocator_advance(AllocatorAdvance {
            tenant,
            kind: AllocatorKind::Node,
            new_high_water: high,
        });
        let occupant = recovered.alloc_node(tenant).unwrap();
        addressed
            .write_node(tenant, &node(occupant.raw(), 100, 1_001))
            .unwrap();
        assert_eq!(
            addressed
                .read_node(tenant, NodeId::new(dangling_tel_ref.src_id))
                .unwrap(),
            None,
            "dangling TEL ref resolved to a compacted occupant"
        );
        assert!(occupant.raw() > high);
    }

    // Named gate remains in the production regime through a real v10 WAL
    // append and fresh-process-equivalent bootstrap recovery.
    let mode = BootstrapMode::Durable {
        data_dir: fixture.path().to_path_buf(),
    };
    let (backend, guard) = bootstrap_storage_backend(&mode).unwrap();
    let routed = backend
        .router()
        .route(TenantId::DEFAULT, arcgraph_core::PartitionId::ZERO)
        .unwrap();
    let mut writer = backend.txn_manager().begin(TENANTS[0]);
    let post_attach = create_node(
        routed.crud(),
        &mut writer,
        TENANTS[0],
        LabelId::new(90),
        &PropertyData::InlineU32Pair(34, 55),
    )
    .unwrap();
    commit(writer, routed.crud()).unwrap();
    drop(routed);
    drop(backend);
    drop(guard);

    let (backend, guard) = bootstrap_storage_backend(&mode).unwrap();
    let routed = backend
        .router()
        .route(TenantId::DEFAULT, arcgraph_core::PartitionId::ZERO)
        .unwrap();
    let reader = backend.txn_manager().begin(TENANTS[0]);
    assert!(
        read_node_with_store(routed.crud(), &reader, post_attach)
            .unwrap()
            .is_some(),
        "identity-preserving generation lost its real-WAL post-attach write"
    );
    reader.abort();
    drop(routed);
    drop(backend);
    drop(guard);
}

#[test]
fn migrated_tombstone_provenance_refuses_stale_publish_across_restart() {
    let fixture = tempdir().unwrap();
    production_v5_fixture(fixture.path());
    upgrade_quiesced_v5_to_v6(fixture.path(), MIGRATION_LSN, MigrationFault::None).unwrap();
    let generation = current_generation(fixture.path()).unwrap().unwrap();

    // First open proves the durable base contains both deletion classes.
    {
        let txn = TxnManager::new();
        let addressed = AddressedRecordStore::new();
        load_v6_physical_base(
            &generation,
            MIGRATION_LSN,
            &txn,
            &addressed,
            &BlobStore::new(),
        )
        .unwrap();
        assert_eq!(
            addressed.read_node(TENANTS[0], NodeId::new(2)).unwrap(),
            None
        );
        assert_eq!(addressed.read_rel(TENANTS[0], RelId::new(2)).unwrap(), None);
    }

    // A fresh owner set models the next process. A high-LSN stale publisher
    // must still lose to the page-resident provenance; no per-boot map is
    // available to help this assertion pass.
    let txn = TxnManager::new();
    let addressed = AddressedRecordStore::new();
    load_v6_physical_base(
        &generation,
        MIGRATION_LSN,
        &txn,
        &addressed,
        &BlobStore::new(),
    )
    .unwrap();
    assert!(matches!(
        addressed.write_node(TENANTS[0], &node(2, 99, 1_000)),
        Err(arcgraph_storage::AddressedStoreError::PermanentTombstone {
            id: 2,
            tombstone_lsn: 100,
            ..
        })
    ));
    let revived_rel = RelRecord::new(
        RelId::new(2),
        TypeId::new(99),
        NodeId::new(1),
        NodeId::new(3),
        Lsn::new(1_000),
    );
    assert!(matches!(
        addressed.write_rel(TENANTS[0], &revived_rel),
        Err(arcgraph_storage::AddressedStoreError::PermanentTombstone {
            id: 2,
            tombstone_lsn: 100,
            ..
        })
    ));
}

fn real_wal_roundtrip_extent_op(
    wal_dir: &Path,
    tenant: TenantId,
    store_id: u16,
    allocation: ExtentAllocation,
    lsn: Lsn,
) -> arcgraph_storage::wal::DeltaOp {
    fs::create_dir(wal_dir).unwrap();
    let op = DeltaIntent::extent_alloc(store_id, tenant, allocation)
        .assign(lsn, lsn)
        .unwrap();
    let bundle = encode_commit_bundle_v10(
        lsn,
        tenant,
        &HashMap::new(),
        &[],
        std::slice::from_ref(&op),
        &[],
        &[],
        &[],
        &[],
        &[],
    )
    .unwrap();
    let record = WalRecord {
        record_type: WalRecordType::CommitBundle,
        txn_id: lsn.raw(),
        lsn,
        timestamp_ms: 0,
        tenant_id: tenant,
        payload: bundle,
    };
    let mut bytes = SegmentHeader {
        format_version: BUNDLE_FORMAT_V10,
    }
    .encode()
    .to_vec();
    record.encode(&mut bytes).unwrap();
    fs::write(wal_dir.join(segment_filename(0)), bytes).unwrap();
    File::open(wal_dir.join(segment_filename(0)))
        .unwrap()
        .sync_all()
        .unwrap();
    let recovered = WalRecoveryReader::open(wal_dir)
        .unwrap()
        .collect::<arcgraph_core::Result<Vec<_>>>()
        .unwrap();
    decode_commit_bundle_v10(&recovered[0].payload, tenant)
        .unwrap()
        .deltas[0]
        .clone()
}

#[test]
fn m4lite_extent_layout_unique_dense_and_counter_recovered() {
    let fixture = tempdir().unwrap();
    production_v5_fixture(fixture.path());
    upgrade_quiesced_v5_to_v6(fixture.path(), MIGRATION_LSN, MigrationFault::None).unwrap();
    let generation = current_generation(fixture.path()).unwrap().unwrap();
    let tenant = TENANTS[0];
    let path = production_extent_store_path(&generation, tenant, STORE_RECORD).unwrap();
    let recovered_next = recover_next_physical_offset(&path, tenant, STORE_RECORD).unwrap();
    assert_eq!(recovered_next, DIRECTORY_HEAD_BYTES + 2 * EXTENT_BYTES);

    let physical = Arc::new(PosixPageIo::open(&path).unwrap());
    let directory = Arc::new(ExtentDirectory::new(tenant, STORE_RECORD, physical, 8));
    let dpt = DirtyPageTable::new();
    let allocation = ExtentAllocation {
        logical_extent: 2,
        physical_offset: recovered_next,
        pairing: 2,
    };
    let op = real_wal_roundtrip_extent_op(
        &fixture.path().join("counter-wal"),
        tenant,
        STORE_RECORD,
        allocation,
        Lsn::new(101),
    );
    assert_eq!(
        directory.apply_extent_alloc(&op, &dpt).unwrap(),
        ExtentApplyOutcome::Applied
    );
    let image = directory
        .copy_page_pinned(tenant, PageId::new(op.page_no))
        .unwrap()
        .unwrap();
    directory
        .write_pages_home(&[(tenant, PageId::new(op.page_no), image)])
        .unwrap();
    drop(directory);

    let after_restart = recover_next_physical_offset(&path, tenant, STORE_RECORD).unwrap();
    assert_eq!(after_restart, DIRECTORY_HEAD_BYTES + 3 * EXTENT_BYTES);
    let ledger = read_extent_ledger(&path, tenant, STORE_RECORD).unwrap();
    let offsets: BTreeSet<_> = ledger.iter().map(|extent| extent.physical_offset).collect();
    assert_eq!(
        offsets.len(),
        ledger.len(),
        "physical offsets collided after restart"
    );

    let over = ExtentAllocation {
        logical_extent: MAX_EXTENTS_PER_STORE,
        physical_offset: after_restart,
        pairing: 0,
    };
    let over_op = real_wal_roundtrip_extent_op(
        &fixture.path().join("cap-wal"),
        tenant,
        STORE_RECORD,
        over,
        Lsn::new(102),
    );
    let reopened = ExtentDirectory::new(
        tenant,
        STORE_RECORD,
        Arc::new(PosixPageIo::open(path).unwrap()),
        8,
    );
    assert!(
        reopened
            .apply_extent_alloc(&over_op, &DirtyPageTable::new())
            .is_err()
    );

    // The production paired allocator owns both next-physical counters. Open
    // it from the migrated directories, persist its proposals through real
    // WAL-decoded ExtentAlloc ops, restart, and prove neither counter resets.
    let props_path = production_extent_store_path(&generation, tenant, STORE_PROPS).unwrap();
    let tel_path = production_extent_store_path(&generation, tenant, STORE_TEL).unwrap();
    let props_directory = Arc::new(ExtentDirectory::new(
        tenant,
        STORE_PROPS,
        Arc::new(PosixPageIo::open(&props_path).unwrap()),
        8,
    ));
    let tel_directory = Arc::new(ExtentDirectory::new(
        tenant,
        STORE_TEL,
        Arc::new(PosixPageIo::open(&tel_path).unwrap()),
        8,
    ));
    let props_data = Arc::new(ExtentDataPageStore::new(Arc::clone(&props_directory), 8));
    let tel_data = Arc::new(ExtentDataPageStore::new(Arc::clone(&tel_directory), 8));
    let dpt = Arc::new(DirtyPageTable::new());
    let pairer = PairedAffinityAllocator::new_recovered(
        Arc::clone(&props_data),
        Arc::clone(&tel_data),
        Arc::clone(&dpt),
    )
    .unwrap();
    let placement = pairer.place(2 * 256 + 7, Lsn::new(200)).unwrap();
    let proposals: HashMap<_, _> = placement
        .extent_allocs
        .iter()
        .map(|op| {
            let allocation = ExtentAllocation::decode(&op.payload, op.op_lsn).unwrap();
            (op.store_id, allocation)
        })
        .collect();
    assert_eq!(
        proposals[&STORE_PROPS].physical_offset,
        DIRECTORY_HEAD_BYTES + EXTENT_BYTES
    );
    assert_eq!(proposals[&STORE_TEL].physical_offset, DIRECTORY_HEAD_BYTES);
    for (ordinal, (store_id, allocation)) in proposals.into_iter().enumerate() {
        let op = real_wal_roundtrip_extent_op(
            &fixture
                .path()
                .join(format!("paired-counter-wal-{store_id}")),
            tenant,
            store_id,
            allocation,
            Lsn::new(300 + ordinal as u64),
        );
        let directory = if store_id == STORE_PROPS {
            props_directory.as_ref()
        } else {
            tel_directory.as_ref()
        };
        assert_eq!(
            directory.apply_extent_alloc(&op, dpt.as_ref()).unwrap(),
            ExtentApplyOutcome::Applied
        );
        let image = directory
            .copy_page_pinned(tenant, PageId::new(op.page_no))
            .unwrap()
            .unwrap();
        directory
            .write_pages_home(&[(tenant, PageId::new(op.page_no), image)])
            .unwrap();
    }
    drop(placement);
    drop(pairer);
    drop(props_data);
    drop(tel_data);
    drop(props_directory);
    drop(tel_directory);

    let mode = BootstrapMode::Durable {
        data_dir: fixture.path().to_path_buf(),
    };
    let (backend, guard) = bootstrap_storage_backend(&mode).unwrap();
    let pairer = guard
        .affinity_allocator(tenant)
        .expect("durable bootstrap must retain directory-recovered counters");
    let after_restart = pairer.place(3 * 256 + 7, Lsn::new(400)).unwrap();
    let recovered: HashMap<_, _> = after_restart
        .extent_allocs
        .iter()
        .map(|op| {
            (
                op.store_id,
                ExtentAllocation::decode(&op.payload, op.op_lsn).unwrap(),
            )
        })
        .collect();
    assert_eq!(
        recovered[&STORE_PROPS].physical_offset,
        DIRECTORY_HEAD_BYTES + 2 * EXTENT_BYTES
    );
    assert_eq!(
        recovered[&STORE_TEL].physical_offset,
        DIRECTORY_HEAD_BYTES + EXTENT_BYTES
    );
    drop(after_restart);
    drop(pairer);
    drop(backend);
    drop(guard);
}

#[test]
fn lsn_monotone_across_swap() {
    let fixture = tempdir().unwrap();
    v5_fixture(fixture.path());
    upgrade_quiesced_v5_to_v6(fixture.path(), MIGRATION_LSN, MigrationFault::None).unwrap();
    let generation = current_generation(fixture.path()).unwrap().unwrap();
    assert_eq!(
        arcgraph_cli::data_dir_migration::read_lsn_seed(&generation).unwrap(),
        MIGRATION_LSN.raw() + 1
    );
    let manager = TxnManager::new();
    manager.seed_after_replay(MIGRATION_LSN);
    let mut tx = manager.begin(TENANTS[0]);
    tx.write(999, Bytes::from_static(b"post-swap"));
    let post_swap = tx.commit().unwrap();
    assert!(
        post_swap > MIGRATION_LSN,
        "post-swap LSN reset suppressed redo"
    );
}

#[test]
fn migrate_upgrade_verb_cli_pinned() {
    // A normal production open may checkpoint v5, but it must not activate
    // M4. Only the explicit upgrade-data-dir sub-verb can swap to gen-v10.
    let on_open_fixture = tempdir().unwrap();
    production_v5_fixture(on_open_fixture.path());
    let mode = BootstrapMode::Durable {
        data_dir: on_open_fixture.path().to_path_buf(),
    };
    let (backend, guard) = bootstrap_storage_backend(&mode).unwrap();
    drop(backend);
    drop(guard);
    let source = current_generation(on_open_fixture.path()).unwrap().unwrap();
    assert_eq!(source.file_name().unwrap(), "gen-v9");
    assert_eq!(
        arcgraph_storage::check_or_stamp_data_dir(&source, true, false).unwrap(),
        5
    );

    let fixture = tempdir().unwrap();
    let source = production_v5_fixture(fixture.path());
    let source_wal = source.join("wal").join(segment_filename(0));
    let before_wal = fs::read(&source_wal).unwrap();
    let output = Command::new(BIN)
        .args([
            "migrate",
            "upgrade-data-dir",
            "--data-dir",
            fixture.path().to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let after_wal = fs::read(source_wal).unwrap();
    assert!(
        after_wal.len() >= before_wal.len() && after_wal.starts_with(&before_wal),
        "final checkpoint may append but must not truncate the pre-swap v5 WAL"
    );
    let generation = current_generation(fixture.path()).unwrap().unwrap();
    assert_eq!(generation.file_name().unwrap(), "gen-v10");
    assert_eq!(
        arcgraph_storage::check_or_stamp_data_dir(&generation, true, false).unwrap(),
        DATA_DIR_VERSION_M4
    );
    assert!(
        fs::read_dir(generation.join("wal"))
            .unwrap()
            .next()
            .is_none()
    );
}

/// INV-M5.2 production-regime gate. The two non-default tenants are migrated
/// by the shipped CLI through the real extent writer and real WAL, then an
/// immediate post-attach commit is recovered after closing the first process
/// state. Rebase loader pages above `MIGRATION_LSN`, or seed the live clock at
/// or below it, and this gate exits red (cargo test exit 101).
#[test]
fn loader_page_lsn_is_migration_lsn() {
    let fixture = tempdir().unwrap();
    production_v5_fixture(fixture.path());
    let output = Command::new(BIN)
        .args([
            "migrate",
            "upgrade-data-dir",
            "--data-dir",
            fixture.path().to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let generation = current_generation(fixture.path()).unwrap().unwrap();
    let migration_lsn = Lsn::new(
        arcgraph_storage::read_data_dir_manifest(&generation)
            .unwrap()
            .unwrap()
            .migration_lsn
            .unwrap(),
    );
    assert_eq!(
        arcgraph_cli::data_dir_migration::read_lsn_seed(&generation).unwrap(),
        migration_lsn.raw() + 1,
        "the live clock was re-based instead of continuing immediately after migration_lsn"
    );
    for tenant_id in TENANTS.into_iter().chain([TenantId::DEFAULT]) {
        for store_id in M4_EXTENT_STORE_IDS {
            let path = production_extent_store_path(&generation, tenant_id, *store_id).unwrap();
            let mut file = File::open(&path).unwrap();
            let file_len = file.metadata().unwrap().len();
            for extent in read_extent_ledger(&path, tenant_id, *store_id).unwrap() {
                for page_in_extent in 0..EXTENT_PAGES {
                    let mut bytes = [0_u8; PAGE_SIZE];
                    let offset = extent.physical_offset + page_in_extent * PAGE_SIZE as u64;
                    if offset + PAGE_SIZE as u64 > file_len {
                        break;
                    }
                    file.seek(SeekFrom::Start(offset)).unwrap();
                    file.read_exact(&mut bytes).unwrap();
                    if bytes.iter().all(|byte| *byte == 0) {
                        continue;
                    }
                    assert_eq!(
                        SlottedPageRef::open(&bytes).unwrap().page_lsn(),
                        migration_lsn,
                        "loader page escaped the migration frontier: tenant={tenant_id:?} store={store_id} extent={} page={page_in_extent}",
                        extent.logical_extent,
                    );
                }
            }
        }
    }
    let mode = BootstrapMode::Durable {
        data_dir: fixture.path().to_path_buf(),
    };
    let (backend, guard) = bootstrap_storage_backend(&mode).unwrap();
    let tenant = backend
        .router()
        .route(TenantId::DEFAULT, arcgraph_core::PartitionId::ZERO)
        .unwrap();
    for migrated_tenant in TENANTS {
        let reader = backend.txn_manager().begin(migrated_tenant);
        assert!(
            read_node_with_store(tenant.crud(), &reader, NodeId::new(1))
                .unwrap()
                .is_some(),
            "tenant {migrated_tenant:?} lost its migrated v5 record"
        );
        reader.abort();
    }
    let mut writer = backend.txn_manager().begin(TENANTS[0]);
    let created = create_node(
        tenant.crud(),
        &mut writer,
        TENANTS[0],
        LabelId::new(99),
        &PropertyData::InlineU32Pair(1, 2),
    )
    .unwrap();
    let post_swap = commit(writer, tenant.crud()).unwrap();
    assert!(post_swap > migration_lsn);

    let mut delete = backend.txn_manager().begin(TENANTS[0]);
    delete_node_with_store(tenant.crud(), &mut delete, NodeId::new(1)).unwrap();
    let delete_lsn = commit(delete, tenant.crud()).unwrap();
    assert!(delete_lsn > post_swap);
    drop(tenant);
    drop(backend);
    drop(guard);

    let (backend, guard) = bootstrap_storage_backend(&mode).unwrap();
    let tenant = backend
        .router()
        .route(TenantId::DEFAULT, arcgraph_core::PartitionId::ZERO)
        .unwrap();
    let reader = backend.txn_manager().begin(TENANTS[0]);
    assert_eq!(
        read_node_with_store(tenant.crud(), &reader, NodeId::new(1)).unwrap(),
        None,
        "post-swap delete of a migrated record was lost during v6 recovery"
    );
    assert_eq!(
        tenant
            .crud()
            .addressed_record_store()
            .unwrap()
            .read_node(TENANTS[0], NodeId::new(1))
            .unwrap(),
        None,
        "v6 recovery left deleted migrated bytes authoritative"
    );
    assert!(
        read_node_with_store(tenant.crud(), &reader, created)
            .unwrap()
            .is_some(),
        "post-swap commit disappeared after v6 WAL recovery"
    );
    assert!(
        tenant
            .crud()
            .addressed_record_store()
            .unwrap()
            .read_node(TENANTS[0], created)
            .unwrap()
            .is_some(),
        "v6 recovery left the arithmetic slot absent and fell back to MVCC"
    );
    reader.abort();
    drop(tenant);
    drop(backend);
    drop(guard);
}

// ---------------------------------------------------------------------------
// M4 Slice-3b-2 — PHYSICAL-arm ACL enforcement gate (#1490 survival).
//
// #1490 (953eda26) fixed the `graph.explore` ACL-bypass-by-traversal. Its own
// gates (`crates/arcgraph-mcp/tests/fix_1488_explore_acl.rs`,
// `fix_1490_bolt_acl.rs`) build the router as
// `MultiTenantRouter::new(catalog, crud, None)`; `router.rs` then falls back to
// `PermissionIndex::new()` — the RESIDENT arm of
// `EffectivePermissions::try_is_visible`.
//
// M4 Slice-3b-2 retires the resident owner maps: production
// (`arcgraph-cli/src/bootstrap.rs`, M4 branch) serves
// `PermissionIndex::page_backed(..)` — the PHYSICAL arm. Those two arms are
// disjoint code paths. Hard-wiring the PHYSICAL arm to `Ok(true)` — a total ACL
// bypass, every node visible to every principal — leaves all four #1490 gates
// GREEN, because they never execute it. Post-3b-2 the enforcement path that
// production actually runs therefore had no adversarial coverage.
//
// This gate closes that hole. It drives the #1490 enforcement surface
// (`graph.explore` / `graph.inspect`) over the REAL production M4 bootstrap —
// the same `bootstrap_storage_backend` the `serve` binary calls — and asserts
// UP FRONT that the served index is page-backed, so the gate can never silently
// regress onto an arm production does not run (the fixture-shape trap above).
//
// This gate covers the production path. It does NOT fix the root: the #1490
// gates themselves still take the resident default, because `arcgraph-mcp` has
// no access to the `arcgraph-cli` production bootstrap. Tracked in #1491
// (cross-crate fixture move — deliberately out of this slice's scope).
//
// It bites BOTH ways, so neither a fail-open nor a fail-closed regression can
// pass. Mutation-proven RED-on-revert against `permissions.rs::try_is_visible`:
//   PHYSICAL arm -> `Ok(true)`  (fail-open, the P0 bypass)
//       => `denied` leaks into alice's neighborhood        => FAILS (assert 3).
//   PHYSICAL arm -> `Ok(false)` (fail-closed / over-denial)
//       => alice loses her own authorized seed             => FAILS (assert 2).
// ---------------------------------------------------------------------------

type AclDispatcher = Dispatcher<
    StorageSchemaProvider,
    StorageNodeInspector,
    StorageNeighborhoodExplorer,
    StorageHybridSearcher,
    StorageIngestProvider,
    StorageRawQueryExecutor,
>;

fn acl_dispatcher(
    backend: &StorageBackend,
    tenant: TenantId,
    scope: SessionScope,
) -> AclDispatcher {
    Dispatcher::with_session_scope(
        tenant,
        scope,
        Arc::new(StorageSchemaProvider::new(backend.clone())),
        Arc::new(StorageNodeInspector::new(backend.clone())),
        Arc::new(StorageNeighborhoodExplorer::new(backend.clone())),
        Arc::new(StorageHybridSearcher::new(backend.clone())),
        Arc::new(StorageIngestProvider::new(backend.clone())),
        Arc::new(StorageRawQueryExecutor::new(backend.clone())),
    )
}

fn acl_tool_result(
    dispatcher: &AclDispatcher,
    name: &str,
    arguments: serde_json::Value,
) -> serde_json::Value {
    let response = handle_raw_envelope(
        dispatcher,
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1404,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments}
        }),
    )
    .expect("request, not notification");
    assert!(
        response["error"].is_null(),
        "tools/call failure: {response}"
    );
    assert_eq!(response["result"]["isError"], false, "{response}");
    serde_json::from_str(
        response["result"]["content"][0]["text"]
            .as_str()
            .expect("tool result text"),
    )
    .expect("inner tool result JSON")
}

/// Assert the served `PermissionIndex` is the M4 page-backed (PHYSICAL) arm.
///
/// This is the anti-vacuity guard. Without it a future refactor could drop the
/// gate back onto `PermissionIndex::new()` (RESIDENT) and it would keep passing
/// while covering nothing production runs.
fn assert_served_index_is_physical(backend: &StorageBackend) {
    let routed = backend
        .router()
        .route(TenantId::DEFAULT, arcgraph_core::PartitionId::ZERO)
        .unwrap();
    let permissions = routed.permissions();
    assert!(
        permissions.is_page_backed(),
        "M4 gate ran on the RESIDENT permission arm — production serves the \
         page-backed arm, so this gate would be vacuous (see module note)"
    );
    assert!(
        !permissions.has_resident_owner_maps(),
        "M4 Slice-3b-2 retires the resident owner maps; residency census: {:?}",
        permissions.resident_map_cardinalities()
    );
}

/// Ingest the ACL fixture through the production MCP ingest path:
/// `seed` (granted to alice) --LINKS_TO--> `denied` (granted to bob only).
fn ingest_physical_acl_fixture(backend: &StorageBackend) -> (u64, u64) {
    let power = acl_dispatcher(backend, TenantId::DEFAULT, SessionScope::Power);
    let result = acl_tool_result(
        &power,
        "graph.ingest",
        serde_json::json!({
            "tenant_id": TenantId::DEFAULT.raw(),
            "nodes": [
                {
                    "external_id": "m4-3b2-seed",
                    "label": "Document",
                    "properties": {"body": "SEED_VISIBLE_3B2"}
                },
                {
                    "external_id": "m4-3b2-denied",
                    "label": "Document",
                    "properties": {"body": "DENIED_SECRET_3B2"}
                }
            ],
            "relationships": [{
                "external_id": "m4-3b2-adjacent",
                "from_external_id": "m4-3b2-seed",
                "to_external_id": "m4-3b2-denied",
                "rel_type": "LINKS_TO",
                "properties": {}
            }],
            "acl_grants": [
                {"external_id": "m4-3b2-seed", "read_principals": ["alice"]},
                {"external_id": "m4-3b2-denied", "read_principals": ["bob"]}
            ],
            "format": "json"
        }),
    );
    let summary: serde_json::Value =
        serde_json::from_str(result["body"].as_str().expect("ingest body"))
            .expect("ingest summary");
    assert_eq!(summary["failed_count"], 0, "fixture ingest: {summary}");
    let seed = summary["records"][0]["internal_id"]
        .as_u64()
        .expect("seed id");
    let denied = summary["records"][1]["internal_id"]
        .as_u64()
        .expect("denied id");
    (seed, denied)
}

/// Explore `seed` at depth 1 as `principal`; return the neighborhood + raw body.
fn explore_as(backend: &StorageBackend, seed: u64, principal: &str) -> (Neighborhood, String) {
    let read = acl_dispatcher(backend, TenantId::DEFAULT, SessionScope::Read);
    let result = acl_tool_result(
        &read,
        "graph.explore",
        serde_json::json!({
            "tenant_id": TenantId::DEFAULT.raw(),
            "seed": seed,
            "max_depth": 1,
            "format": "json",
            "principal": principal
        }),
    );
    let body = result["body"].as_str().expect("explore body").to_owned();
    let neighborhood: Neighborhood = serde_json::from_str(&body).expect("neighborhood");
    (neighborhood, body)
}

#[test]
fn gate_m4_physical_arm_acl_enforcement_and_restart_durability() {
    let fixture = tempdir().unwrap();
    production_v5_fixture(fixture.path());
    upgrade_quiesced_v5_to_v6(fixture.path(), MIGRATION_LSN, MigrationFault::None).unwrap();
    let mode = BootstrapMode::Durable {
        data_dir: fixture.path().to_path_buf(),
    };

    let (backend, guard) = bootstrap_storage_backend(&mode).unwrap();

    // (1) ANTI-VACUITY: we are on the arm production actually serves.
    assert_served_index_is_physical(&backend);

    let (seed, denied) = ingest_physical_acl_fixture(&backend);

    // (2) BITES ON OVER-DENIAL: alice's own authorized seed must still be
    //     returned. A blanket fail-closed regression (PHYSICAL -> Ok(false))
    //     kills this assertion.
    let (alice_view, alice_body) = explore_as(&backend, seed, "alice");
    assert!(
        alice_view.nodes.iter().any(|node| node.id == seed),
        "over-denial: alice lost her own authorized seed: {alice_body}"
    );
    assert!(
        alice_body.contains("SEED_VISIBLE_3B2"),
        "over-denial: authorized content withheld: {alice_body}"
    );

    // (3) BITES ON FAIL-OPEN: bob's document must not leak to alice by
    //     traversal. This is the #1490 bypass. A fail-open regression
    //     (PHYSICAL -> Ok(true)) kills this assertion.
    assert!(
        alice_view.nodes.iter().all(|node| node.id != denied),
        "ACL BYPASS: denied node reachable by traversal on the M4 physical \
         arm: {alice_body}"
    );
    assert!(
        !alice_body.contains("DENIED_SECRET_3B2"),
        "ACL BYPASS: denied content leaked on the M4 physical arm: {alice_body}"
    );
    assert!(
        alice_view.edges.is_empty(),
        "denied incident edge must be omitted: {alice_body}"
    );

    // (4) The grant is real, not a blanket deny: bob DOES see his own document.
    let (bob_view, bob_body) = explore_as(&backend, denied, "bob");
    assert!(
        bob_view.nodes.iter().any(|node| node.id == denied),
        "bob lost his own authorized document: {bob_body}"
    );
    assert!(
        bob_body.contains("DENIED_SECRET_3B2"),
        "bob's authorized content withheld: {bob_body}"
    );

    drop(backend);
    drop(guard);

    // (5) INV-B0 durable owner-row provenance: the grants are page-resident,
    //     not process-resident. After a bare restart the SAME enforcement must
    //     hold — no deny-all (#1221) and no widen.
    let (backend, guard) = bootstrap_storage_backend(&mode).unwrap();
    assert_served_index_is_physical(&backend);

    let (alice_view, alice_body) = explore_as(&backend, seed, "alice");
    assert!(
        alice_view.nodes.iter().any(|node| node.id == seed),
        "restart deny-all: alice lost her seed after reopen: {alice_body}"
    );
    assert!(
        alice_view.nodes.iter().all(|node| node.id != denied),
        "restart ACL BYPASS: denied node reachable after reopen: {alice_body}"
    );
    assert!(
        !alice_body.contains("DENIED_SECRET_3B2"),
        "restart ACL BYPASS: denied content leaked after reopen: {alice_body}"
    );

    drop(backend);
    drop(guard);
}

// ---------------------------------------------------------------------------
// ROOT CAUSE C gate — allocator seed vs WAL replay: crash-restart must not
// reissue a durably-committed StringId / AclClassId.
//
// `InternTable::page_backed` / `PermissionIndex::page_backed` seed their in-RAM
// AtomicU32 counters at bootstrap from the OWNER_ALLOCATOR_MARKER_ID row read
// off the last CHECKPOINTED page image — i.e. BEFORE WAL replay. Commits made
// after that image advance the durable marker through their v10 owner-page
// delta, so replay restores the PAGE correctly; but the in-RAM counter only
// learns about them via the replayed `AllocatorAdvance`, and
// `CrudAllocatorSeedHandle::seed_from_advance` DROPPED both the InternString and
// AclClass kinds (`AllocatorKind::InternString | AllocatorKind::AclClass => {}`).
//
// Net effect: after a crash the counters are pinned at the pre-crash image's
// high-water and the very next allocation REISSUES ids that committed
// transactions already own — a never-reissue violation, on the #1404-close
// surface itself. Node/Rel allocators were always seeded this way (ADR-034 D-1);
// the two M4 owner allocators were not.
//
// The 40M rung and every graceful-restart gate are blind to this: it needs a
// crash between the commit and the next checkpoint.
//
// RED-on-revert: restore the dropped arm in `seed_from_advance` — the post-crash
// allocation reissues and this gate fails with "REISSUED".
// ---------------------------------------------------------------------------

const ALLOC_REUSE_ROOT: &str = "ARCGRAPH_M4_ALLOC_REUSE_ROOT";
const ALLOC_REUSE_ACKED: &str = "ALLOC_REUSE_ACKED";

/// Labels the child interns before the crash. Each distinct grant set below also
/// forces a fresh `AclClassId`.
const PRE_CRASH_LABELS: [&str; 3] = ["AllocDocA", "AllocDocB", "AllocDocC"];

fn ingest_labelled_acl_node(
    backend: &StorageBackend,
    external_id: &str,
    label: &str,
    principals: &[&str],
) {
    let power = acl_dispatcher(backend, TenantId::DEFAULT, SessionScope::Power);
    let result = acl_tool_result(
        &power,
        "graph.ingest",
        serde_json::json!({
            "tenant_id": TenantId::DEFAULT.raw(),
            "nodes": [{
                "external_id": external_id,
                "label": label,
                "properties": {"body": format!("payload-{external_id}")}
            }],
            "acl_grants": [{
                "external_id": external_id,
                "read_principals": principals
            }],
            "format": "json"
        }),
    );
    let summary: serde_json::Value =
        serde_json::from_str(result["body"].as_str().expect("ingest body")).expect("summary");
    assert_eq!(summary["failed_count"], 0, "ingest failed: {summary}");
}

/// Highest `StringId` currently bound to any of `names` (production probe).
fn max_interned_id(backend: &StorageBackend, names: &[&str]) -> u64 {
    names
        .iter()
        .map(|name| {
            backend
                .intern_table()
                .try_probe(TenantId::DEFAULT, name)
                .expect("intern forward lookup")
                .unwrap_or_else(|| panic!("label {name} was not interned"))
                .raw()
                .into()
        })
        .max()
        .expect("no names")
}

fn acl_class_allocator_next(backend: &StorageBackend) -> u32 {
    backend
        .router()
        .route(TenantId::DEFAULT, arcgraph_core::PartitionId::ZERO)
        .unwrap()
        .permissions()
        .class_allocator_next()
}

#[cfg(unix)]
fn run_alloc_reuse_child(root: &Path) -> ! {
    let mode = BootstrapMode::Durable {
        data_dir: root.to_path_buf(),
    };
    let (backend, _guard) = bootstrap_storage_backend(&mode).unwrap();

    // The allocators under test must be the M4 page-backed ones, or this gate
    // is vacuous.
    assert!(
        backend.intern_table().is_page_backed(),
        "intern allocator is not page-backed — gate would be vacuous"
    );

    // Commit interned strings + ACL grants. Each distinct grant set allocates a
    // fresh AclClassId; each new label a fresh StringId. The WAL is fsynced by
    // the commit; the owner PAGES are still write-behind.
    for (index, label) in PRE_CRASH_LABELS.iter().enumerate() {
        ingest_labelled_acl_node(
            &backend,
            &format!("alloc-reuse-{index}"),
            label,
            &[
                // distinct principal set per node -> distinct ACL class
                Box::leak(format!("principal-{index}").into_boxed_str()) as &str,
            ],
        );
    }

    let max_string = max_interned_id(&backend, &PRE_CRASH_LABELS);
    let next_class = acl_class_allocator_next(&backend);
    assert!(
        next_class > 0,
        "no ACL class was allocated — gate is vacuous"
    );

    // Publish what the crash must not forget, and make it durable.
    let acked = root.join(ALLOC_REUSE_ACKED);
    fs::write(&acked, format!("{max_string} {next_class}")).unwrap();
    File::open(&acked).unwrap().sync_all().unwrap();

    // CRASH before any checkpoint / graceful flush: the WAL holds the commits,
    // the owner pages and their allocator markers do not.
    unsafe {
        libc::kill(libc::getpid(), libc::SIGKILL);
    }
    unreachable!("SIGKILL did not terminate the allocator-reuse child");
}

#[cfg(unix)]
#[test]
fn gate_crash_restart_never_reissues_durable_intern_or_acl_class_ids() {
    if let Some(root) = std::env::var_os(ALLOC_REUSE_ROOT) {
        run_alloc_reuse_child(Path::new(&root));
    }

    let fixture = tempdir().unwrap();
    production_v5_fixture(fixture.path());
    upgrade_quiesced_v5_to_v6(fixture.path(), MIGRATION_LSN, MigrationFault::None).unwrap();

    let status = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "gate_crash_restart_never_reissues_durable_intern_or_acl_class_ids",
            "--nocapture",
        ])
        .env(ALLOC_REUSE_ROOT, fixture.path())
        .status()
        .unwrap();
    assert!(
        status.code().is_none(),
        "child must die by SIGKILL, not exit cleanly ({status:?})"
    );

    let acked = fs::read_to_string(fixture.path().join(ALLOC_REUSE_ACKED))
        .expect("child did not publish its pre-crash allocator high-water");
    let mut parts = acked.split_whitespace();
    let durable_max_string: u64 = parts.next().unwrap().parse().unwrap();
    let durable_next_class: u32 = parts.next().unwrap().parse().unwrap();

    // Restart through the PRODUCTION bootstrap: recovery replays the WAL, which
    // carries the AllocatorAdvance entries for both owner allocators.
    let mode = BootstrapMode::Durable {
        data_dir: fixture.path().to_path_buf(),
    };
    let (backend, guard) = bootstrap_storage_backend(&mode).unwrap();
    assert!(backend.intern_table().is_page_backed());

    // (1) ACL class allocator must not rewind behind what the crash made durable.
    let recovered_next_class = acl_class_allocator_next(&backend);
    assert!(
        recovered_next_class >= durable_next_class,
        "REISSUED: ACL class allocator rewound across the crash — next={recovered_next_class} \
         but {durable_next_class} was already durably committed. The replayed \
         AllocatorAdvance{{AclClass}} was dropped, so the next grant would reuse a \
         class id that live documents already reference."
    );

    // (2) The pre-crash interned names must still resolve to their committed ids
    //     (replay restored the owner rows) ...
    let recovered_max_string = max_interned_id(&backend, &PRE_CRASH_LABELS);
    assert_eq!(
        recovered_max_string, durable_max_string,
        "replay lost a durably-interned StringId"
    );

    // (3) ... and a FRESH intern after the crash must not reuse any of them.
    ingest_labelled_acl_node(
        &backend,
        "alloc-reuse-post-crash",
        "AllocDocPostCrash",
        &["principal-post-crash"],
    );
    let fresh = max_interned_id(&backend, &["AllocDocPostCrash"]);
    assert!(
        fresh > durable_max_string,
        "REISSUED: post-crash intern handed out StringId {fresh}, but {durable_max_string} \
         was already durably committed before the crash. The replayed \
         AllocatorAdvance{{InternString}} was dropped, so the allocator rewound to the \
         last checkpointed marker and reissued a live id."
    );

    drop(backend);
    drop(guard);
}
