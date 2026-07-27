//! Genuine subprocess SIGKILL gate on the v9 production commit format.

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use arcgraph_core::{LabelId, Lsn, NodeId, PageId, TenantId};
use arcgraph_storage::crud::{CrudStore, PropertyData, commit, create_node, node_mvcc_key};
use arcgraph_storage::io::PageBuf;
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::{PrimaryIndex, PrimaryPageStore};
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::segment::{SegmentHeader, segment_filename};
use arcgraph_storage::wal::{
    BUNDLE_FORMAT_V9, PageStoreTarget, ReplayConfig, ReplayExecutor, WalConfig, WalRecoveryReader,
    WalWriter,
};
use arcgraph_storage::{DeltaPageStore, DirtyPageTable};
use tempfile::tempdir;

const CHILD_ENV: &str = "ARCGRAPH_M3_V9_SIGKILL_CHILD";
const DIR_ENV: &str = "ARCGRAPH_M3_V9_SIGKILL_DIR";

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

fn child_workload(dir: &Path) -> ! {
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
    let mut store = CrudStore::new_with_index(None, Arc::clone(&primary), allocator);
    let writer = WalWriter::spawn_from(config(dir.to_path_buf()), manager.current_lsn()).unwrap();
    let wal = writer.handle();
    manager.attach_wal(wal.clone());
    primary.attach_wal(wal.clone());
    store.attach_wal(wal);
    let mut ledger = OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("ACKED"))
        .unwrap();

    let mut sequence = 0u64;
    loop {
        let mut tx = manager.begin(TenantId::DEFAULT);
        let id = create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(7),
            &PropertyData::InlineU32Pair(sequence as u32, 0xA5A5),
        )
        .unwrap();
        commit(tx, &store).unwrap();
        writeln!(ledger, "{}", id.raw()).unwrap();
        ledger.sync_all().unwrap();
        sequence += 1;
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
fn v9_strict_commit_sigkill_recovers_every_acknowledged_node() {
    if std::env::var_os(CHILD_ENV).is_some() {
        let dir = PathBuf::from(std::env::var_os(DIR_ENV).unwrap());
        child_workload(&dir);
    }

    let root = tempdir().unwrap();
    let dir = root.path().join("v9");
    let mut child = Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg("v9_strict_commit_sigkill_recovers_every_acknowledged_node")
        .arg("--nocapture")
        .env(CHILD_ENV, "1")
        .env(DIR_ENV, &dir)
        .spawn()
        .unwrap();

    let deadline = Instant::now() + Duration::from_secs(10);
    let acknowledged = loop {
        if let Ok(text) = std::fs::read_to_string(dir.join("ACKED")) {
            let ids: Vec<u64> = text.lines().filter_map(|line| line.parse().ok()).collect();
            if ids.len() >= 20 {
                break ids;
            }
        }
        assert!(
            Instant::now() < deadline,
            "child produced no durable prefix"
        );
        std::thread::sleep(Duration::from_millis(20));
    };

    let kill = Command::new("kill")
        .args(["-9", &child.id().to_string()])
        .status()
        .unwrap();
    assert!(kill.success(), "kernel SIGKILL delivery failed");
    let status = child.wait().unwrap();
    assert!(!status.success(), "child unexpectedly exited gracefully");

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
    let recovered = replay.run(WalRecoveryReader::open(&dir).unwrap()).unwrap();
    assert!(recovered > Lsn::ZERO);

    let reader = manager.begin(TenantId::DEFAULT);
    for id in &acknowledged {
        assert!(
            reader.read(node_mvcc_key(NodeId::new(*id))).is_some(),
            "strict-ACKed node {id} missing after SIGKILL recovery"
        );
    }
    reader.abort();
    assert!(
        !records.pages.lock().unwrap().is_empty(),
        "v9 recovery rebuilt no physical record pages"
    );
}
