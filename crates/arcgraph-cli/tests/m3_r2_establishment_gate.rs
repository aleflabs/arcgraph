//! M3 round-2 establishment gate: a Periodic commit may not enter retained
//! owner metadata before its WAL record is durable.
//!
//! This promotes the deterministic `skeptic_p04_periodic_phantom.rs` oracle
//! into the production bootstrap/checkpointer/recovery path. The child uses
//! the real v9 `CrudStore`, pauses the real WAL writer after append but before
//! fsync, and invokes the wired `DurableCheckpointer::checkpoint`. The parent
//! then kills the child, truncates the explicitly non-durable WAL suffix, and
//! reopens through production recovery. The acknowledged Periodic node must
//! be absent from both the served view and retained primary owner.
//!
//! RED-on-revert: without the WAL flush + deferred drain before sidecar
//! establishment, the checkpoint completes while fsync is paused and embeds
//! the node's primary entry. Recovery after dropping the WAL suffix observes
//! that dangling entry and this gate fails.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::Arc;
use std::time::{Duration, Instant};

use arcgraph_cli::bootstrap::{BootstrapMode, bootstrap_storage_backend};
use arcgraph_cli::data_dir_migration::{current_generation, upgrade_data_dir};
use arcgraph_core::{DurabilityTier, LabelId, NodeId, PartitionId, RelId, TenantId, TypeId};
use arcgraph_storage::crud::{
    PropertyData, commit, create_node, create_rel, delete_rel_with_store, read_node_with_store,
    read_rel_with_store,
};
use arcgraph_storage::primary_index::{PrimaryKey, RecordKind};
use arcgraph_storage::wal::{list_segments, segment_count, segment_filename};
use tempfile::tempdir;

const CHILD_ENV: &str = "ARCGRAPH_M3_R2_ESTABLISHMENT_CHILD";
const ROOT_ENV: &str = "ARCGRAPH_M3_R2_ESTABLISHMENT_ROOT";
const PAUSE_ENV: &str = "ARCGRAPH_M3_TEST_PAUSE_BEFORE_FSYNC";
const SEGMENT_BYTES_ENV: &str = "ARCGRAPH_M3_TEST_WAL_SEGMENT_BYTES";
const WAIT: Duration = Duration::from_secs(20);

fn write_synced(path: &Path, bytes: &[u8]) {
    let mut file = File::create(path).expect("create marker");
    file.write_all(bytes).expect("write marker");
    file.sync_all().expect("sync marker");
}

fn wait_for(path: &Path, what: &str) {
    let deadline = Instant::now() + WAIT;
    while !path.exists() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(2));
    }
}

fn wait_for_exit(child: &mut Child, timeout: Duration) -> Option<std::process::ExitStatus> {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = child.try_wait().expect("poll child") {
            return Some(status);
        }
        if Instant::now() >= deadline {
            return None;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}

fn crud_for(
    backend: &arcgraph_mcp::storage::StorageBackend,
) -> Arc<arcgraph_storage::crud::CrudStore> {
    Arc::clone(
        backend
            .router()
            .route(TenantId::DEFAULT, PartitionId::ZERO)
            .expect("route DEFAULT")
            .crud(),
    )
}

fn child(root: &Path) -> ! {
    let mode = BootstrapMode::Durable {
        data_dir: root.to_path_buf(),
    };
    let (backend, guard) = bootstrap_storage_backend(&mode).expect("bootstrap v9 child");
    let crud = crud_for(&backend);

    // Establish a physical base containing a relationship with a finite
    // upper visibility bound. This is the recovery-visibility leg of the
    // same production-regime restart gate.
    let mut create = backend.txn_manager().begin(TenantId::DEFAULT);
    let src = create_node(
        &crud,
        &mut create,
        TenantId::DEFAULT,
        LabelId::new(1),
        &PropertyData::Empty,
    )
    .expect("create source");
    let dst = create_node(
        &crud,
        &mut create,
        TenantId::DEFAULT,
        LabelId::new(1),
        &PropertyData::Empty,
    )
    .expect("create destination");
    let expired_rel = create_rel(
        &crud,
        &mut create,
        TenantId::DEFAULT,
        src,
        dst,
        TypeId::new(3),
        &PropertyData::Empty,
    )
    .expect("create relationship");
    commit(create, &crud).expect("commit relationship base");
    let mut delete = backend.txn_manager().begin(TenantId::DEFAULT);
    delete_rel_with_store(&crud, &mut delete, expired_rel).expect("stage relationship expiry");
    commit(delete, &crud).expect("commit relationship expiry");
    guard
        .checkpointer()
        .expect("wired checkpointer")
        .checkpoint()
        .expect("establish expired relationship physical base");
    write_synced(
        &root.join("EXPIRED_REL_ID"),
        expired_rel.raw().to_string().as_bytes(),
    );

    // With the debug-only tiny segment setting, these strict commits create
    // a real closed WAL prefix that the post-restart quiescent checkpoint
    // must reclaim after its lone Periodic commit drains.
    for i in 0..24u32 {
        let mut filler = backend.txn_manager().begin(TenantId::DEFAULT);
        create_node(
            &crud,
            &mut filler,
            TenantId::DEFAULT,
            LabelId::new(2),
            &PropertyData::InlineU32Pair(i, i.wrapping_mul(3)),
        )
        .expect("stage WAL rotation filler");
        commit(filler, &crud).expect("commit WAL rotation filler");
    }

    // Flip DEFAULT using a Strict SYSTEM transaction before arming the
    // pre-fsync pause, so the only paused fire belongs to the target commit.
    let mut tier_tx = backend.txn_manager().begin(TenantId::SYSTEM);
    backend
        .router()
        .catalog()
        .set_durability_tier(
            &mut tier_tx,
            TenantId::DEFAULT,
            DurabilityTier::Periodic { rpo_ms: 60_000 },
        )
        .expect("set DEFAULT Periodic");
    tier_tx.commit().expect("durable tier change");

    let generation = current_generation(root)
        .expect("read CURRENT")
        .expect("v9 generation");
    let wal_dir = generation.join("wal");
    let segments = list_segments(&wal_dir).expect("list baseline WAL");
    let active = *segments.last().expect("active WAL segment");
    let active_path = wal_dir.join(segment_filename(active));
    let durable_len = active_path.metadata().expect("stat durable WAL").len();
    write_synced(
        &root.join("DURABLE_WAL"),
        format!("{active} {durable_len}").as_bytes(),
    );
    write_synced(
        &root.join("DURABLE_BLOB_PAGES"),
        crud.blob_store()
            .logical_page_count()
            .to_string()
            .as_bytes(),
    );

    let pause = root.join("PAUSED_BEFORE_FSYNC");
    let mut arm = pause.as_os_str().to_os_string();
    arm.push(".arm");
    write_synced(Path::new(&arm), b"armed");

    let mut tx = backend.txn_manager().begin(TenantId::DEFAULT);
    let node_id = create_node(
        &crud,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(9),
        &PropertyData::Blob(vec![0xA5; 16 * 1024]),
    )
    .expect("stage Periodic node");
    let commit_lsn = commit(tx, &crud).expect("Periodic ACK before fsync");
    assert!(
        guard.last_durable_lsn().expect("durable watermark") < commit_lsn,
        "premise: Periodic commit must remain non-durable"
    );
    assert!(
        crud.primary()
            .expect("primary")
            .lookup(PrimaryKey::new(
                TenantId::DEFAULT,
                RecordKind::Node,
                node_id.raw(),
            ))
            .expect("primary lookup")
            .is_some(),
        "premise: Phase 1 already installed the primary entry"
    );
    write_synced(&root.join("NODE_ID"), node_id.raw().to_string().as_bytes());
    wait_for(&pause, "writer pre-fsync pause");

    write_synced(&root.join("CHECKPOINT_STARTED"), b"started");
    let checkpoint = guard.checkpointer().expect("wired durable checkpointer");
    checkpoint.checkpoint().expect("production v9 checkpoint");
    write_synced(&root.join("CHECKPOINT_ESTABLISHED"), b"established");

    // The buggy implementation reaches here with the writer still paused.
    // Keep every owner alive until the parent delivers the crash.
    loop {
        std::thread::park();
    }
}

#[test]
fn production_periodic_checkpoint_crash_cannot_restore_owner2_phantom() {
    let tmp = tempdir().expect("tempdir");
    let root = tmp.path().join("db");
    upgrade_data_dir(&root).expect("build initial v9 generation");

    let pause = root.join("PAUSED_BEFORE_FSYNC");
    let mut child = Command::new(std::env::current_exe().expect("current test executable"))
        .arg("--exact")
        .arg("m3_r2_establishment_child")
        .arg("--nocapture")
        .env(CHILD_ENV, "1")
        .env(ROOT_ENV, &root)
        .env(PAUSE_ENV, &pause)
        .env(SEGMENT_BYTES_ENV, "2048")
        .spawn()
        .expect("spawn establishment child");

    wait_for(&pause, "child writer pre-fsync pause");
    wait_for(
        &root.join("CHECKPOINT_STARTED"),
        "production checkpoint start",
    );

    // Correct behavior is to block establishment on the paused WAL flush.
    // The old path establishes immediately; five seconds is intentionally
    // far above this tiny fixture's normal checkpoint latency.
    let establishment_deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < establishment_deadline && !root.join("CHECKPOINT_ESTABLISHED").exists() {
        if let Some(status) = child.try_wait().expect("poll child") {
            panic!("child exited before crash injection: {status}");
        }
        std::thread::sleep(Duration::from_millis(2));
    }

    child.kill().expect("kill child at pre-fsync crash point");
    let status = wait_for_exit(&mut child, WAIT).expect("child exits after kill");
    assert!(!status.success(), "crash child unexpectedly exited cleanly");

    // Deterministically model power loss: discard bytes appended after the
    // last completed fsync. The marker was synced before the Periodic append.
    let durable =
        std::fs::read_to_string(root.join("DURABLE_WAL")).expect("read durable WAL marker");
    let mut parts = durable.split_whitespace();
    let segment: u64 = parts.next().expect("segment").parse().expect("segment u64");
    let len: u64 = parts.next().expect("length").parse().expect("length u64");
    assert!(parts.next().is_none(), "malformed durable WAL marker");
    let generation = current_generation(&root)
        .expect("read CURRENT after crash")
        .expect("v9 generation after crash");
    let wal_dir = generation.join("wal");
    for later in list_segments(&wal_dir).expect("list WAL for crash truncation") {
        if later > segment {
            std::fs::remove_file(wal_dir.join(segment_filename(later)))
                .expect("drop wholly non-durable WAL segment");
        }
    }
    let wal_path = wal_dir.join(segment_filename(segment));
    let wal = OpenOptions::new()
        .write(true)
        .open(&wal_path)
        .expect("open WAL for crash truncation");
    wal.set_len(len).expect("drop non-durable WAL suffix");
    wal.sync_all().expect("sync crash-truncated WAL");
    let _ = std::fs::remove_file(&pause);

    let node_id = NodeId::new(
        std::fs::read_to_string(root.join("NODE_ID"))
            .expect("read node id")
            .parse()
            .expect("node id u64"),
    );
    let mode = BootstrapMode::Durable {
        data_dir: root.clone(),
    };
    let (recovered, recovered_guard) =
        bootstrap_storage_backend(&mode).expect("production recovery after crash");
    let crud = crud_for(&recovered);
    let primary_hit = crud
        .primary()
        .expect("recovered primary")
        .lookup(PrimaryKey::new(
            TenantId::DEFAULT,
            RecordKind::Node,
            node_id.raw(),
        ))
        .expect("recovered primary lookup");
    assert!(
        primary_hit.is_none(),
        "checkpoint metadata persisted a primary entry whose Periodic WAL record was not durable"
    );
    let reader = recovered.txn_manager().begin(TenantId::DEFAULT);
    assert!(
        read_node_with_store(&crud, &reader, node_id)
            .expect("served read")
            .is_none(),
        "crash recovery served a Periodic node absent from the durable WAL prefix"
    );
    let durable_blob_pages: usize = std::fs::read_to_string(root.join("DURABLE_BLOB_PAGES"))
        .expect("read durable blob-page count")
        .parse()
        .expect("blob-page count usize");
    assert_eq!(
        crud.blob_store().logical_page_count(),
        durable_blob_pages,
        "checkpoint metadata restored store-5 pages belonging only to the discarded WAL suffix"
    );
    let expired_rel = RelId::new(
        std::fs::read_to_string(root.join("EXPIRED_REL_ID"))
            .expect("read expired relationship id")
            .parse()
            .expect("relationship id u64"),
    );
    assert!(
        read_rel_with_store(&crud, &reader, expired_rel)
            .expect("served relationship read")
            .is_none(),
        "recovery served a relationship past its finite expired_lsn"
    );
    reader.abort();

    let mut tier_tx = recovered.txn_manager().begin(TenantId::SYSTEM);
    recovered
        .router()
        .catalog()
        .set_durability_tier(
            &mut tier_tx,
            TenantId::DEFAULT,
            DurabilityTier::Periodic { rpo_ms: 60_000 },
        )
        .expect("restore DEFAULT Periodic tier after restart");
    tier_tx.commit().expect("durable Periodic tier restore");

    let mut lone = recovered.txn_manager().begin(TenantId::DEFAULT);
    let lone_id = create_node(
        &crud,
        &mut lone,
        TenantId::DEFAULT,
        LabelId::new(10),
        &PropertyData::InlineU32Pair(7, 11),
    )
    .expect("stage lone Periodic commit");
    let lone_lsn = commit(lone, &crud).expect("ack lone Periodic commit");
    let durable_deadline = Instant::now() + WAIT;
    while recovered_guard
        .last_durable_lsn()
        .expect("durable watermark")
        < lone_lsn
    {
        assert!(
            Instant::now() < durable_deadline,
            "lone Periodic commit did not fsync without follow-up traffic"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
    let generation = current_generation(&root)
        .expect("read CURRENT before reclaim")
        .expect("v9 generation before reclaim");
    let wal_dir = generation.join("wal");
    let segments_before = segment_count(&wal_dir).expect("count WAL before quiescent checkpoint");
    assert!(
        segments_before > 1,
        "premise: test must create a closed WAL prefix"
    );
    let frontier = recovered_guard
        .checkpointer()
        .expect("wired recovered checkpointer")
        .checkpoint()
        .expect("quiescent checkpoint after Periodic fsync");
    let segments_after = segment_count(&wal_dir).expect("count WAL after quiescent checkpoint");
    assert!(
        frontier >= lone_lsn,
        "checkpoint frontier {frontier:?} stayed clamped below durable Periodic commit {lone_lsn:?}"
    );
    assert!(
        segments_after < segments_before,
        "checkpoint establishment did not reclaim the closed WAL prefix: before={segments_before} after={segments_after}"
    );
    let reader = recovered.txn_manager().begin(TenantId::DEFAULT);
    assert!(
        read_node_with_store(&crud, &reader, lone_id)
            .expect("read lone Periodic node")
            .is_some(),
        "frontier advanced but lost the lone durable Periodic commit"
    );
}

#[test]
fn m3_r2_establishment_child() {
    if std::env::var_os(CHILD_ENV).is_none() {
        return;
    }
    let root = PathBuf::from(std::env::var_os(ROOT_ENV).expect("child root env"));
    child(&root);
}
