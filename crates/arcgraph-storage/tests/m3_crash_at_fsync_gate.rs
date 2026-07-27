#![cfg(debug_assertions)]

//! Deterministic M3 GATE-2 crash/fsync-failure checkpointer race on production v9.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use arcgraph_core::{ArcGraphError, LabelId, Lsn, NodeId, PAGE_SIZE, PageId, TenantId};
use arcgraph_storage::blob::BlobStore;
use arcgraph_storage::crud::{
    CrudError, CrudStore, PropertyData, commit, create_node, node_mvcc_key, update_node,
};
use arcgraph_storage::io::{PageBuf, PageIo, PosixPageIo};
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::page_store::{
    BufferedRecordPageStore, PerTenantBufferPool, PerTenantBufferPoolConfig,
};
use arcgraph_storage::primary_index::{PrimaryIndex, PrimaryKey, PrimaryPageStore, RecordKind};
use arcgraph_storage::records::{SlotId, SlottedPageRef};
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::segment::{SegmentHeader, segment_filename};
use arcgraph_storage::wal::{
    BUNDLE_FORMAT_V9, PageStoreTarget, ReplayConfig, ReplayExecutor, STORE_RECORD, WalConfig,
    WalRecoveryReader, WalWriter,
};
use arcgraph_storage::{
    BlobPageFlushTarget, DeltaPageStore, DirtyPageTable, WriteBehindCheckpointer,
};
use tempfile::tempdir;

const CHILD_ENV: &str = "ARCGRAPH_M3_CRASH_AT_FSYNC_CHILD";
const DIR_ENV: &str = "ARCGRAPH_M3_CRASH_AT_FSYNC_DIR";
const MODE_ENV: &str = "ARCGRAPH_M3_CRASH_AT_FSYNC_MODE";
const PAUSE_ENV: &str = "ARCGRAPH_M3_TEST_PAUSE_BEFORE_FSYNC";
const FAIL_ENV: &str = "ARCGRAPH_M3_TEST_FSYNC_FAILURE";
const WAIT: Duration = Duration::from_secs(20);

#[derive(Clone, Copy, PartialEq, Eq)]
enum FaultMode {
    Crash,
    FsyncFailure,
}

impl FaultMode {
    fn from_env() -> Self {
        match std::env::var(MODE_ENV).as_deref() {
            Ok("crash") => Self::Crash,
            Ok("fsync-failure") => Self::FsyncFailure,
            other => panic!("invalid {MODE_ENV}: {other:?}"),
        }
    }
}

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

fn write_synced(path: &Path, bytes: &[u8]) {
    let mut file = File::create(path).unwrap();
    file.write_all(bytes).unwrap();
    file.sync_all().unwrap();
}

fn write_marker(path: &Path) {
    write_synced(path, b"ready");
}

fn read_home_page(path: &Path, page_id: PageId) -> Box<PageBuf> {
    let mut file = OpenOptions::new().read(true).open(path).unwrap();
    file.seek(SeekFrom::Start(page_id.raw() * PAGE_SIZE as u64))
        .unwrap();
    let mut page = Box::new([0u8; PAGE_SIZE]);
    file.read_exact(page.as_mut()).unwrap();
    page
}

fn read_baseline_page(dir: &Path) -> Box<PageBuf> {
    std::fs::read(dir.join("BASELINE_PAGE"))
        .unwrap()
        .try_into()
        .unwrap()
}

fn truncate_wal_to_durable_prefix(dir: &Path) {
    let durable_len: u64 = std::fs::read_to_string(dir.join("DURABLE_LEN"))
        .unwrap()
        .parse()
        .unwrap();
    OpenOptions::new()
        .write(true)
        .open(dir.join(segment_filename(0)))
        .unwrap()
        .set_len(durable_len)
        .unwrap();
}

fn child(dir: &Path, mode: FaultMode) -> ! {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(
        dir.join(segment_filename(0)),
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
    let writer = WalWriter::spawn_from(config(dir.to_path_buf()), manager.current_lsn()).unwrap();
    let wal = writer.handle();
    manager.attach_wal(wal.clone());
    primary.attach_wal(wal.clone());

    let record_home = dir.join("record.home");
    let record_io: Arc<dyn PageIo> = Arc::new(PosixPageIo::create(&record_home).unwrap());
    let pools = Arc::new(PerTenantBufferPool::with_config(
        record_io,
        PerTenantBufferPoolConfig {
            frames_per_tenant: 16,
            write_fraction: 0.0,
        },
    ));
    let buffered_records = Arc::new(BufferedRecordPageStore::with_cache_cap(pools, 16));
    let blobs = Arc::new(BlobStore::new());
    let mut store = CrudStore::new_with_existing_buffered_page_store(
        Some(Arc::clone(&primary)),
        Some(wal.clone()),
        Arc::clone(&allocator),
        Arc::clone(&buffered_records),
        Arc::clone(&blobs),
    );
    store.attach_wal(wal);
    let dpt = Arc::new(DirtyPageTable::new());
    store.attach_m3_dirty_page_table(Arc::clone(&dpt));
    let store = Arc::new(store);

    let mut baseline = manager.begin(TenantId::DEFAULT);
    let node = create_node(
        &store,
        &mut baseline,
        TenantId::DEFAULT,
        LabelId::new(1),
        &PropertyData::InlineU32Pair(1, 2),
    )
    .unwrap();
    let baseline_lsn = commit(baseline, &store).unwrap();
    let slot = primary
        .lookup(PrimaryKey::new(
            TenantId::DEFAULT,
            RecordKind::Node,
            node.raw(),
        ))
        .unwrap()
        .unwrap();
    let baseline_page = buffered_records
        .copy_page_pinned(slot.page)
        .unwrap()
        .unwrap();
    let reader = manager.begin(TenantId::DEFAULT);
    let baseline_mvcc = reader.read(node_mvcc_key(node)).unwrap();
    reader.abort();
    write_synced(&dir.join("BASELINE_PAGE"), baseline_page.as_ref());
    write_synced(&dir.join("BASELINE_MVCC"), baseline_mvcc.as_ref());
    write_synced(
        &dir.join("BASELINE_LSN"),
        baseline_lsn.raw().to_string().as_bytes(),
    );
    write_synced(
        &dir.join("BASELINE_PAGE_ID"),
        slot.page.raw().to_string().as_bytes(),
    );
    write_synced(
        &dir.join("DURABLE_LEN"),
        std::fs::metadata(dir.join(segment_filename(0)))
            .unwrap()
            .len()
            .to_string()
            .as_bytes(),
    );
    assert_eq!(dpt.len(), 1, "baseline must dirty exactly one record page");
    assert_eq!(dpt.snapshot()[0].key.store_id, STORE_RECORD);

    let props_home: Arc<dyn PageIo> =
        Arc::new(PosixPageIo::create(dir.join("props.home")).unwrap());
    let props = Arc::new(BlobPageFlushTarget::new(Arc::clone(&blobs), props_home));
    let checkpointer = Arc::new(WriteBehindCheckpointer::with_batch_pages(
        Arc::clone(&dpt),
        props,
        Arc::clone(&buffered_records) as Arc<dyn arcgraph_storage::PageFlushTarget>,
        1,
    ));

    write_marker(&dir.join("PAUSED.arm"));
    let update_store = Arc::clone(&store);
    let update_manager = Arc::clone(&manager);
    let update = std::thread::spawn(move || {
        let mut tx = update_manager.begin(TenantId::DEFAULT);
        match mode {
            FaultMode::Crash => {
                update_node(
                    &update_store,
                    &mut tx,
                    node,
                    &PropertyData::InlineU32Pair(9, 9),
                )
                .unwrap();
            }
            FaultMode::FsyncFailure => {
                create_node(
                    &update_store,
                    &mut tx,
                    TenantId::DEFAULT,
                    LabelId::new(1),
                    &PropertyData::InlineU32Pair(9, 9),
                )
                .unwrap();
            }
        }
        commit(tx, &update_store)
    });
    while !dir.join("PAUSED").exists() {
        std::thread::sleep(Duration::from_millis(1));
    }

    if buffered_records
        .copy_page_pinned(slot.page)
        .unwrap()
        .unwrap()
        != baseline_page
    {
        eprintln!("a v9 page changed before its WAL fsync proof");
        std::process::exit(101);
    }
    write_marker(&dir.join("CACHE_OLD"));

    let checkpoint_manager = Arc::clone(&manager);
    let checkpoint_dir = dir.to_path_buf();
    let checkpoint = std::thread::spawn(move || {
        write_marker(&checkpoint_dir.join("CHECKPOINTER_FLUSHING"));
        let report = checkpointer
            .flush_pass(checkpoint_manager.current_lsn())
            .unwrap();
        assert_eq!(report.snapshot_pages, 1);
        assert_eq!(report.flushed_pages, 1);
        write_marker(&checkpoint_dir.join("HOME_FLUSHED"));
        write_marker(&checkpoint_dir.join("CHECKPOINTER_WAITING"));
        let _freeze = checkpoint_manager.checkpoint_freeze();
        write_marker(&checkpoint_dir.join("CHECKPOINTER_ACQUIRED"));
    });

    match mode {
        FaultMode::Crash => loop {
            std::thread::park();
        },
        FaultMode::FsyncFailure => {
            let error = update
                .join()
                .unwrap()
                .expect_err("injected fsync must fail commit");
            assert!(
                matches!(
                    error,
                    CrudError::Mvcc(ArcGraphError::WalErrorRolledBack { .. })
                ),
                "fsync failure ACK must be WalErrorRolledBack, got {error:?}"
            );
            checkpoint.join().unwrap();
            assert!(dir.join("CHECKPOINTER_ACQUIRED").exists());
            assert!(
                dpt.is_empty(),
                "failed commit must not re-dirty the flushed page"
            );
            assert_eq!(
                buffered_records
                    .copy_page_pinned(slot.page)
                    .unwrap()
                    .unwrap(),
                baseline_page,
                "fsync failure left a physical page mutation"
            );
            assert_eq!(
                read_home_page(&record_home, slot.page),
                baseline_page,
                "racing checkpointer persisted a failed commit"
            );
            let reader = manager.begin(TenantId::DEFAULT);
            assert_eq!(
                reader.read(node_mvcc_key(node)).unwrap().as_ref(),
                baseline_mvcc.as_ref(),
                "failed commit changed the visible MVCC prefix"
            );
            reader.abort();
            write_marker(&dir.join("FAILURE_ROLLED_BACK"));

            drop(writer);
            truncate_wal_to_durable_prefix(dir);
            let new_writer =
                WalWriter::spawn_from(config(dir.to_path_buf()), manager.allocator_lsn()).unwrap();
            let wal = new_writer.handle();
            manager.attach_wal(wal.clone());
            primary.attach_wal(wal.clone());
            let mut store = match Arc::try_unwrap(store) {
                Ok(store) => store,
                Err(_) => panic!("test leaked a CrudStore owner"),
            };
            store.attach_wal(wal);
            let store = Arc::new(store);
            let mut followup = manager.begin(TenantId::DEFAULT);
            let followup_node = create_node(
                &store,
                &mut followup,
                TenantId::DEFAULT,
                LabelId::new(1),
                &PropertyData::InlineU32Pair(3, 4),
            )
            .unwrap();
            commit(followup, &store).unwrap();
            let followup_slot = primary
                .lookup(PrimaryKey::new(
                    TenantId::DEFAULT,
                    RecordKind::Node,
                    followup_node.raw(),
                ))
                .unwrap()
                .unwrap();
            assert_eq!(followup_slot.page, slot.page);
            assert_eq!(
                followup_slot.slot,
                SlotId(1),
                "failed commit leaked its reserved slot"
            );
            write_marker(&dir.join("FAILED_SLOT_REUSED"));
            new_writer.shutdown().unwrap();
            std::process::exit(0);
        }
    }
}

fn spawn_child(test_name: &str, dir: &Path, mode: FaultMode) -> Child {
    let marker = dir.join("PAUSED");
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .arg("--exact")
        .arg(test_name)
        .arg("--nocapture")
        .env(CHILD_ENV, "1")
        .env(DIR_ENV, dir)
        .env(
            MODE_ENV,
            match mode {
                FaultMode::Crash => "crash",
                FaultMode::FsyncFailure => "fsync-failure",
            },
        )
        .env(PAUSE_ENV, marker);
    match mode {
        FaultMode::Crash => {
            command.env_remove(FAIL_ENV);
        }
        FaultMode::FsyncFailure => {
            command.env(FAIL_ENV, "1");
        }
    }
    command.spawn().unwrap()
}

fn wait_for_markers(process: &mut Child, dir: &Path, markers: &[&str]) {
    let deadline = Instant::now() + WAIT;
    while !markers.iter().all(|marker| dir.join(marker).exists()) {
        assert!(
            Instant::now() < deadline,
            "child did not reach markers {markers:?}"
        );
        assert!(
            process.try_wait().unwrap().is_none(),
            "child exited before reaching markers {markers:?}"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn wait_for_success(process: &mut Child) {
    let deadline = Instant::now() + WAIT;
    loop {
        if let Some(status) = process.try_wait().unwrap() {
            assert!(status.success(), "child failed with {status}");
            return;
        }
        assert!(Instant::now() < deadline, "child did not exit");
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[derive(Default)]
struct TenantPages {
    pages: Mutex<HashMap<(TenantId, PageId), Box<PageBuf>>>,
}

impl DeltaPageStore for TenantPages {
    fn read_page_for_redo(
        &self,
        tenant: TenantId,
        page_id: PageId,
    ) -> arcgraph_core::Result<Option<Box<PageBuf>>> {
        Ok(self.pages.lock().unwrap().get(&(tenant, page_id)).cloned())
    }

    fn install_page_from_redo(
        &self,
        tenant: TenantId,
        page_id: PageId,
        page: Box<PageBuf>,
    ) -> arcgraph_core::Result<()> {
        self.pages.lock().unwrap().insert((tenant, page_id), page);
        Ok(())
    }
}

#[test]
fn crash_at_fsync_recovers_byte_identical_durable_prefix() {
    if std::env::var_os(CHILD_ENV).is_some() {
        child(
            &PathBuf::from(std::env::var_os(DIR_ENV).unwrap()),
            FaultMode::from_env(),
        );
    }
    let root = tempdir().unwrap();
    let dir = root.path().join("v9");
    let mut process = spawn_child(
        "crash_at_fsync_recovers_byte_identical_durable_prefix",
        &dir,
        FaultMode::Crash,
    );
    wait_for_markers(
        &mut process,
        &dir,
        &["CACHE_OLD", "HOME_FLUSHED", "CHECKPOINTER_WAITING"],
    );
    assert!(
        !dir.join("CHECKPOINTER_ACQUIRED").exists(),
        "checkpoint frontier crossed a commit paused before durability"
    );
    assert!(
        Command::new("kill")
            .args(["-9", &process.id().to_string()])
            .status()
            .unwrap()
            .success()
    );
    assert!(!process.wait().unwrap().success());

    let page_id = PageId::new(
        std::fs::read_to_string(dir.join("BASELINE_PAGE_ID"))
            .unwrap()
            .parse()
            .unwrap(),
    );
    let baseline_page = read_baseline_page(&dir);
    assert_eq!(
        read_home_page(&dir.join("record.home"), page_id),
        baseline_page,
        "checkpointer flushed a phantom pre-fsync page"
    );

    // SIGKILL leaves the kernel page cache alive. Truncating to the length
    // recorded after the last strict ACK models the power-loss half of the
    // crash-at-fsync boundary: the appended-but-unfsynced frame is absent.
    truncate_wal_to_durable_prefix(&dir);
    let props = Arc::new(TenantPages::default());
    let records = Arc::new(TenantPages::default());
    let target = PageStoreTarget::primary_only(Arc::new(PrimaryPageStore::new()))
        .with_delta_stores(
            props,
            Arc::clone(&records) as Arc<dyn DeltaPageStore>,
            Arc::new(DirtyPageTable::new()),
        );
    let manager = Arc::new(TxnManager::new());
    let mut replay = ReplayExecutor::new(
        ReplayConfig::with_wal_dir(&dir),
        Arc::clone(&manager),
        target,
    );
    let baseline_lsn = Lsn::new(
        std::fs::read_to_string(dir.join("BASELINE_LSN"))
            .unwrap()
            .parse()
            .unwrap(),
    );
    assert_eq!(
        replay.run(WalRecoveryReader::open(&dir).unwrap()).unwrap(),
        baseline_lsn
    );
    let baseline_mvcc = std::fs::read(dir.join("BASELINE_MVCC")).unwrap();
    let reader = manager.begin(TenantId::DEFAULT);
    let recovered_mvcc = reader.read(node_mvcc_key(NodeId::new(1))).unwrap();
    assert_eq!(recovered_mvcc.as_ref(), baseline_mvcc.as_slice());
    let record =
        arcgraph_core::record::NodeRecord::from_bytes(recovered_mvcc.as_ref().try_into().unwrap())
            .unwrap();
    assert_eq!((record.inline_u32a, record.inline_u32b), (1, 2));
    reader.abort();
    let recovered_pages = records.pages.lock().unwrap();
    assert_eq!(recovered_pages.len(), 1);
    assert_eq!(
        recovered_pages.get(&(TenantId::DEFAULT, page_id)).unwrap(),
        &baseline_page
    );
    let physical = SlottedPageRef::open(baseline_page.as_ref())
        .unwrap()
        .read_node(SlotId(0))
        .unwrap()
        .unwrap();
    assert_eq!((physical.inline_u32a, physical.inline_u32b), (1, 2));
}

#[test]
fn fsync_failure_keeps_page_old_unwinds_version_and_reuses_slot() {
    if std::env::var_os(CHILD_ENV).is_some() {
        child(
            &PathBuf::from(std::env::var_os(DIR_ENV).unwrap()),
            FaultMode::from_env(),
        );
    }
    let root = tempdir().unwrap();
    let dir = root.path().join("v9");
    let mut process = spawn_child(
        "fsync_failure_keeps_page_old_unwinds_version_and_reuses_slot",
        &dir,
        FaultMode::FsyncFailure,
    );
    wait_for_markers(
        &mut process,
        &dir,
        &["CACHE_OLD", "HOME_FLUSHED", "CHECKPOINTER_WAITING"],
    );
    assert!(
        !dir.join("CHECKPOINTER_ACQUIRED").exists(),
        "checkpoint frontier crossed an unresolved fsync"
    );
    write_marker(&dir.join("PAUSED.release"));
    wait_for_markers(
        &mut process,
        &dir,
        &[
            "CHECKPOINTER_ACQUIRED",
            "FAILURE_ROLLED_BACK",
            "FAILED_SLOT_REUSED",
        ],
    );
    wait_for_success(&mut process);
    let page_id = PageId::new(
        std::fs::read_to_string(dir.join("BASELINE_PAGE_ID"))
            .unwrap()
            .parse()
            .unwrap(),
    );
    assert_eq!(
        read_home_page(&dir.join("record.home"), page_id),
        read_baseline_page(&dir),
        "failed commit reached the checkpointer's durable home page"
    );
}
