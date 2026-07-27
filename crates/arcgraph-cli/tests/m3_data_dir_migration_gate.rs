//! M3 RE-4b gate: the explicit offline v4 -> v5 generation migration is
//! atomic-or-resumable and never mutates the v8 source generation.

use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::process::Command;
use std::sync::Arc;

use arcgraph_cli::bootstrap::{BootstrapMode, bootstrap_storage_backend};
use arcgraph_cli::data_dir_migration::{
    MigrationFault, MigrationOutcome, current_generation, upgrade_data_dir,
    upgrade_quiesced_data_dir,
};
use arcgraph_core::record::NodeRecord;
use arcgraph_core::{LabelId, Lsn, NodeId, PAGE_SIZE, PageId, PageType, PartitionId, TenantId};
use arcgraph_storage::BlobRef;
use arcgraph_storage::blob::BlobStore;
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::checkpoint::CheckpointSnapshot;
use arcgraph_storage::crud::{
    CrudStore, PropertyData, commit, create_node, crud_allocator_seed_handle, read_node_with_store,
};
use arcgraph_storage::idempotency::IdempotencyStore;
use arcgraph_storage::intern::InternTable;
use arcgraph_storage::io::InMemoryPageIo;
use arcgraph_storage::manifest::{DataDirManifest, WAL_FORMAT_DELTA_V9};
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::permissions::PermissionIndex;
use arcgraph_storage::primary_index::PrimaryPageStore;
use arcgraph_storage::record_store::RecordPageStore;
use arcgraph_storage::records::{PROP_BAG_MAX_BYTES, SlotId, SlottedPage, SlottedPageRef};
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{BUNDLE_FORMAT_V9, SegmentHeader, segment_filename};
use tempfile::tempdir;

const BIN: &str = env!("CARGO_BIN_EXE_arcgraph");

#[test]
fn production_v9_open_sweeps_orphan_temp_and_fsyncs_directory() {
    let dir = tempdir().unwrap();
    let root = dir.path().join("db");
    upgrade_data_dir(&root).unwrap();
    let generation = current_generation(&root).unwrap().unwrap();
    let sidecar = arcgraph_storage::checkpoint::read_latest_sidecar(&generation)
        .unwrap()
        .unwrap();
    let selected = arcgraph_storage::checkpoint::incremental_metadata_path(
        &generation,
        sidecar.checkpoint_lsn,
        sidecar.metadata_generation,
    );
    assert!(selected.is_file());
    let orphan = generation.join(format!(
        "CHECKPOINT.v9.{:016x}.tmp.production-open-gate",
        sidecar.checkpoint_lsn.raw().saturating_add(1)
    ));
    fs::write(&orphan, b"crash-orphan").unwrap();
    let fsync_before = arcgraph_storage::checkpoint::incremental_temp_sweep_dir_fsync_count();

    let mode = BootstrapMode::Durable {
        data_dir: root.clone(),
    };
    let (backend, guard) = bootstrap_storage_backend(&mode).unwrap();
    assert!(selected.is_file(), "startup removed the selected metadata");
    assert!(!orphan.exists(), "startup left the orphan metadata temp");
    assert_eq!(
        arcgraph_storage::checkpoint::incremental_temp_sweep_dir_fsync_count(),
        fsync_before + 1,
        "temp deletion must be followed by exactly one directory fsync"
    );
    drop(backend);
    drop(guard);
}

fn write_full_checkpoint(root: &std::path::Path, lsn: Lsn) {
    let txn = Arc::new(TxnManager::new());
    txn.seed_after_replay(lsn);
    let primary = Arc::new(PrimaryPageStore::new());
    let records = Arc::new(RecordPageStore::new());
    let blob = Arc::new(BlobStore::new());
    let (property_ref, page_images) = blob
        .stage_bag(arcgraph_core::TenantId::DEFAULT, 9001, b"typed-prop-block")
        .unwrap();
    assert!(page_images.is_empty());
    blob.publish_txn_slotted(9001).unwrap();
    blob.put(
        arcgraph_core::TenantId::DEFAULT,
        &vec![0xA5; PROP_BAG_MAX_BYTES + 1],
    )
    .unwrap();
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
    records
        .install_fresh(
            PageId::new(7),
            PageType::Node,
            arcgraph_core::TenantId::DEFAULT,
        )
        .unwrap();
    let mut node = NodeRecord::new(NodeId::new(7), LabelId::new(3), lsn);
    node.property_ref = property_ref.encode();
    let latch = records.latch(PageId::new(7)).unwrap();
    let mut guard = latch.write();
    SlottedPage::open(guard.as_mut())
        .unwrap()
        .put_node_at(SlotId(0), &node)
        .unwrap();
    drop(guard);
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
        permissions_tenant: arcgraph_core::TenantId::DEFAULT,
    };
    arcgraph_storage::checkpoint::checkpoint(
        root,
        &BufferPool::new(1, Arc::new(InMemoryPageIo::new())),
        &snapshot,
        Vec::new,
        lsn,
    )
    .unwrap();
}

fn v4_fixture(root: &std::path::Path) -> Vec<(String, Vec<u8>)> {
    fs::create_dir_all(root.join("wal")).unwrap();
    arcgraph_storage::stamp_data_dir(root, 4).unwrap();
    arcgraph_storage::write_data_dir_manifest(
        root,
        &DataDirManifest::m2_typed("2026-07-11T00:00:00Z".to_owned()),
    )
    .unwrap();
    fs::write(root.join("pages.db"), b"catalog-v8").unwrap();
    write_full_checkpoint(root, Lsn::new(41));
    fs::write(
        root.join("wal").join(segment_filename(0)),
        SegmentHeader { format_version: 8 }.encode(),
    )
    .unwrap();
    [
        "VERSION",
        "MANIFEST",
        "pages.db",
        "CHECKPOINT.snap",
        "CHECKPOINT",
        "wal/wal-0000000000.log",
    ]
    .into_iter()
    .map(|path| (path.to_owned(), fs::read(root.join(path)).unwrap()))
    .collect()
}

fn assert_v8_intact(root: &std::path::Path, before: &[(String, Vec<u8>)]) {
    for (path, bytes) in before {
        assert_eq!(
            fs::read(root.join(path)).unwrap(),
            *bytes,
            "migration mutated v8 source file {path}"
        );
    }
}

#[test]
fn upgrade_builds_v9_beside_v8_and_stamps_version_last() {
    let dir = tempdir().unwrap();
    let before = v4_fixture(dir.path());
    let outcome = upgrade_quiesced_data_dir(dir.path(), Lsn::new(41), MigrationFault::None)
        .expect("offline upgrade");
    assert_eq!(
        outcome,
        MigrationOutcome::Upgraded {
            migration_lsn: Lsn::new(41)
        }
    );
    assert_v8_intact(dir.path(), &before);

    let generation = current_generation(dir.path()).unwrap().unwrap();
    assert_eq!(generation, dir.path().join("gen-v9"));
    let manifest = arcgraph_storage::read_data_dir_manifest(&generation)
        .unwrap()
        .unwrap();
    assert_eq!(manifest.data_dir_version, 5);
    assert_eq!(manifest.wal_format, WAL_FORMAT_DELTA_V9);
    assert!(
        !generation.join("CHECKPOINT.snap").exists(),
        "normal v9 recovery must not retain the v4 full-snapshot decoder"
    );
    let checkpoint = arcgraph_storage::read_latest_sidecar(&generation)
        .unwrap()
        .unwrap();
    assert!(checkpoint.incremental_metadata);
    assert!(!checkpoint.full_state_snapshot);
    assert_eq!(checkpoint.checkpoint_lsn, Lsn::new(41));
    assert!(generation.join("props.store").is_file());
    let record_store =
        arcgraph_storage::m3_migration::m3_record_store_path(&generation, TenantId::DEFAULT);
    assert!(record_store.is_file());
    let mut record_file = fs::File::open(record_store).unwrap();
    record_file
        .seek(SeekFrom::Start(7 * PAGE_SIZE as u64))
        .unwrap();
    let mut record_page = [0u8; PAGE_SIZE];
    record_file.read_exact(&mut record_page).unwrap();
    let record_view = SlottedPageRef::open(&record_page).unwrap();
    assert_eq!(record_view.page_lsn(), Lsn::new(41));
    assert_eq!(record_view.read_node(SlotId(0)).unwrap().unwrap().id, 7);

    let mut props_file = fs::File::open(generation.join("props.store")).unwrap();
    props_file.seek(SeekFrom::Start(PAGE_SIZE as u64)).unwrap();
    let mut props_page = [0u8; PAGE_SIZE];
    props_file.read_exact(&mut props_page).unwrap();
    let props_view = SlottedPageRef::open(&props_page).unwrap();
    assert_eq!(props_view.page_lsn(), Lsn::new(41));
    assert_eq!(
        props_view.read_bag(SlotId(0)).unwrap().unwrap(),
        b"typed-prop-block"
    );
    assert_eq!(
        fs::read(generation.join("LSN_SEED")).unwrap(),
        42u64.to_le_bytes()
    );
    let header = fs::read(generation.join("wal").join(segment_filename(0))).unwrap();
    assert_eq!(
        SegmentHeader::decode(&header).unwrap().format_version,
        BUNDLE_FORMAT_V9
    );
    assert_eq!(
        arcgraph_storage::check_or_stamp_data_dir(&generation, true, false).unwrap(),
        5
    );

    assert_eq!(
        upgrade_quiesced_data_dir(dir.path(), Lsn::new(999), MigrationFault::None).unwrap(),
        MigrationOutcome::AlreadyUpgraded {
            migration_lsn: Lsn::new(41)
        },
        "restart must not reset or advance the migration boundary"
    );
}

#[test]
fn every_pre_swap_crash_keeps_v8_visible_and_restart_rebuilds() {
    for fault in [
        MigrationFault::AfterScratchCreate,
        MigrationFault::AfterGenerationSync,
        MigrationFault::AfterGenerationRename,
    ] {
        let dir = tempdir().unwrap();
        let mut before = v4_fixture(dir.path());
        write_full_checkpoint(dir.path(), Lsn::new(77));
        before
            .iter_mut()
            .find(|(path, _)| path == "CHECKPOINT.snap")
            .unwrap()
            .1 = fs::read(dir.path().join("CHECKPOINT.snap")).unwrap();
        before
            .iter_mut()
            .find(|(path, _)| path == "CHECKPOINT")
            .unwrap()
            .1 = fs::read(dir.path().join("CHECKPOINT")).unwrap();
        let err = upgrade_quiesced_data_dir(dir.path(), Lsn::new(77), fault)
            .expect_err("fault injection must stop the migration");
        assert!(format!("{err:#}").contains("injected migration crash"));
        assert_v8_intact(dir.path(), &before);
        assert!(
            current_generation(dir.path()).unwrap().is_none(),
            "pre-swap crash must leave v8 selected: {fault:?}"
        );
        assert_eq!(
            upgrade_quiesced_data_dir(dir.path(), Lsn::new(77), MigrationFault::None).unwrap(),
            MigrationOutcome::Upgraded {
                migration_lsn: Lsn::new(77)
            }
        );
    }
}

#[test]
fn crash_after_current_before_version_stamp_rolls_back_and_rebuilds() {
    let dir = tempdir().unwrap();
    let mut before = v4_fixture(dir.path());
    write_full_checkpoint(dir.path(), Lsn::new(123));
    before
        .iter_mut()
        .find(|(path, _)| path == "CHECKPOINT.snap")
        .unwrap()
        .1 = fs::read(dir.path().join("CHECKPOINT.snap")).unwrap();
    before
        .iter_mut()
        .find(|(path, _)| path == "CHECKPOINT")
        .unwrap()
        .1 = fs::read(dir.path().join("CHECKPOINT")).unwrap();
    upgrade_quiesced_data_dir(dir.path(), Lsn::new(123), MigrationFault::AfterCurrentSwap)
        .expect_err("injected post-swap crash");
    assert_v8_intact(dir.path(), &before);
    assert!(
        current_generation(dir.path()).unwrap().is_none(),
        "an unstamped first generation must not replace the legacy root"
    );
    let generation = dir.path().join("gen-v9");
    assert!(
        !generation.join("VERSION").exists(),
        "VERSION=5 is the last durable act"
    );
    assert_eq!(
        upgrade_quiesced_data_dir(dir.path(), Lsn::new(123), MigrationFault::None).unwrap(),
        MigrationOutcome::Upgraded {
            migration_lsn: Lsn::new(123)
        }
    );
    assert_eq!(
        arcgraph_storage::check_or_stamp_data_dir(&generation, true, false).unwrap(),
        5
    );
}

#[test]
fn normal_bootstrap_keeps_unstamped_v9_generation_absent() {
    let dir = tempdir().unwrap();
    v4_fixture(dir.path());
    write_full_checkpoint(dir.path(), Lsn::new(124));
    upgrade_quiesced_data_dir(dir.path(), Lsn::new(124), MigrationFault::AfterCurrentSwap)
        .expect_err("injected post-swap crash");
    let generation = dir.path().join("gen-v9");
    assert!(current_generation(dir.path()).unwrap().is_none());
    assert!(!generation.join("VERSION").exists());

    let mode = BootstrapMode::Durable {
        data_dir: dir.path().to_path_buf(),
    };
    let (backend, guard) = bootstrap_storage_backend(&mode)
        .expect("normal bootstrap must keep serving the legacy predecessor");
    drop(backend);
    drop(guard);
    assert!(current_generation(dir.path()).unwrap().is_none());
    assert!(!generation.join("VERSION").exists());

    let restart_lsn = arcgraph_storage::read_latest_sidecar(dir.path())
        .unwrap()
        .unwrap()
        .checkpoint_lsn;
    assert_eq!(
        upgrade_quiesced_data_dir(dir.path(), restart_lsn, MigrationFault::None).unwrap(),
        MigrationOutcome::Upgraded {
            migration_lsn: restart_lsn
        }
    );
    assert_eq!(
        arcgraph_storage::check_or_stamp_data_dir(&generation, true, false).unwrap(),
        5
    );
}

#[test]
fn operator_subcommand_upgrades_a_real_quiesced_v4_store() {
    let dir = tempdir().unwrap();
    let mode = BootstrapMode::Durable {
        data_dir: dir.path().to_path_buf(),
    };
    let (backend, guard) = bootstrap_storage_backend(&mode).unwrap();
    drop(backend);
    drop(guard);

    let output = Command::new(BIN)
        .args([
            "migrate",
            "upgrade-data-dir",
            "--data-dir",
            dir.path().to_str().unwrap(),
        ])
        .output()
        .expect("spawn explicit offline migration");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "upgrade command failed: status={:?}\nstdout={stdout}\nstderr={stderr}",
        output.status
    );
    assert!(stdout.contains("Upgraded"), "unexpected output: {stdout}");
    let generation = current_generation(dir.path()).unwrap().unwrap();
    assert_eq!(
        arcgraph_storage::check_or_stamp_data_dir(&generation, true, false).unwrap(),
        5
    );
    let checkpoint = arcgraph_storage::read_latest_sidecar(&generation)
        .unwrap()
        .unwrap();
    assert!(checkpoint.incremental_metadata);
}

#[test]
fn migrated_generation_reopens_reads_base_and_replays_new_v9_commit() {
    let dir = tempdir().unwrap();
    v4_fixture(dir.path());
    upgrade_quiesced_data_dir(dir.path(), Lsn::new(41), MigrationFault::None).unwrap();
    let mode = BootstrapMode::Durable {
        data_dir: dir.path().to_path_buf(),
    };

    let (backend, guard) = bootstrap_storage_backend(&mode).unwrap();
    let tenant = backend
        .router()
        .route(TenantId::DEFAULT, PartitionId::ZERO)
        .unwrap();
    let reader = backend.txn_manager().begin(TenantId::DEFAULT);
    let migrated = read_node_with_store(tenant.crud(), &reader, NodeId::new(7))
        .unwrap()
        .expect("migrated node must be served from record.store");
    assert_eq!(migrated.label_id, 3);
    reader.abort();

    let mut writer_tx = backend.txn_manager().begin(TenantId::DEFAULT);
    let post_migration_id = create_node(
        tenant.crud(),
        &mut writer_tx,
        TenantId::DEFAULT,
        LabelId::new(8),
        &PropertyData::Blob(b"post-migration-props".to_vec()),
    )
    .unwrap();
    let commit_lsn = commit(writer_tx, tenant.crud()).unwrap();
    assert!(commit_lsn.raw() > 41, "LSN clock reset across migration");
    drop(tenant);
    drop(backend);
    drop(guard);

    let generation = current_generation(dir.path()).unwrap().unwrap();
    let incremental = arcgraph_storage::checkpoint::read_latest_sidecar(&generation)
        .unwrap()
        .unwrap();
    assert!(incremental.incremental_metadata);
    assert!(incremental.checkpoint_lsn.raw() >= commit_lsn.raw());
    assert!(
        !generation.join("CHECKPOINT.snap").exists(),
        "v5 shutdown must not reintroduce whole-owner freeze capture"
    );

    let (reopened, guard) = bootstrap_storage_backend(&mode).unwrap();
    let tenant = reopened
        .router()
        .route(TenantId::DEFAULT, PartitionId::ZERO)
        .unwrap();
    let reader = reopened.txn_manager().begin(TenantId::DEFAULT);
    assert!(
        read_node_with_store(tenant.crud(), &reader, NodeId::new(7))
            .unwrap()
            .is_some()
    );
    let recovered = read_node_with_store(tenant.crud(), &reader, post_migration_id)
        .unwrap()
        .expect("post-migration v9 delta commit must survive a second process open");
    let property_ref = BlobRef::decode(recovered.property_ref).unwrap();
    assert_eq!(
        tenant
            .crud()
            .blob_store()
            .get(TenantId::DEFAULT, property_ref)
            .unwrap(),
        b"post-migration-props".as_slice()
    );
    reader.abort();
    drop(tenant);
    drop(reopened);
    drop(guard);
}
