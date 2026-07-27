//! HIGH #810 — a single batched ingest (N records, one transaction, one
//! `crud::commit`) must fsync **once**, not once-per-record.
//!
//! # The bug (pre-fix)
//!
//! Every node/rel with a non-empty property bag encodes as
//! [`PropertyData::Blob`]. The durable blob write path
//! ([`BlobStore::put_logged_and_stage`]) emitted a standalone
//! `WalRecordType::PutBlob` record through the **synchronous**
//! [`WalHandle::append`] (which BLOCKS until fsync on the Strict tier) —
//! once per record. So a single transaction that creates N
//! property-bearing nodes fsynced N times (one PutBlob per node) PLUS
//! one final `CommitBundle` fsync = N+1 fsyncs for one logical commit.
//! On a durable store this collapsed throughput to ~170 rec/s (issue
//! #810 CZ-scale measurement: 1k nodes = 5.9s, 10k = 59.9s).
//!
//! The PutBlob record was redundant: the blob chain pages are ALSO
//! staged (`StagedEmit { kind: Blob, .. }`) and folded into the same
//! `CommitBundle` that the commit fsyncs once (drained by
//! `crud::commit` via `take_blob_emits`), and WAL replay reconstructs
//! the chain from the bundle's `BundlePageKind::Blob` entries
//! (`wal/replay.rs`). The standalone per-record PutBlob fsync bought
//! nothing for durability and defeated group commit.
//!
//! # The oracle
//!
//! [`WalFireMetrics::wal_t1_appends_total`] counts synchronous
//! (fsync-blocking) WAL appends. For ONE transaction of N
//! property-bearing nodes + ONE commit it must be exactly **1** (the
//! `CommitBundle` append) — independent of N. Pre-fix it was N+1.
//! This is a deterministic count (not a wall-clock threshold), so it is
//! the RED-flip oracle; wall-clock is reported for the throughput
//! narrative but never asserted.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arcgraph_core::{LabelId, TenantId, TypeId};
use arcgraph_storage::crud::{CrudStore, PropertyData, commit, create_node};
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::PrimaryIndex;
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{WalConfig, WalFireMetrics, WalWriter};
use tempfile::TempDir;

fn test_wal_config(dir: PathBuf) -> WalConfig {
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

fn build_stack() -> (
    TempDir,
    Arc<CrudStore>,
    Arc<TxnManager>,
    WalWriter,
    WalFireMetrics,
) {
    let dir = TempDir::new().unwrap();
    let writer = WalWriter::spawn(test_wal_config(dir.path().to_path_buf())).unwrap();
    let metrics = writer.fire_metrics();
    let handle = writer.handle();
    let mgr = Arc::new(TxnManager::with_wal(handle.clone()));
    let alloc = Arc::new(PageAllocator::new());
    let primary = Arc::new(
        PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&alloc), Some(handle.clone())).unwrap(),
    );
    let store = Arc::new(CrudStore::new_with_index(
        Some(handle.clone()),
        Arc::clone(&primary),
        alloc,
    ));
    (dir, store, mgr, writer, metrics)
}

/// A small, single-page JSON-shaped property blob — the shape every
/// property-bearing `graph.ingest` record takes (`property_data_for_json_map`
/// → `PropertyData::Blob`).
fn sample_blob(i: u32) -> PropertyData {
    PropertyData::Blob(format!(r#"{{"name":"node-{i}","seq":{i}}}"#).into_bytes())
}

/// RED-flip oracle for #810: a single transaction creating N
/// property-bearing nodes, then one commit, fsyncs ONCE — not N+1
/// times. Asserted on the deterministic synchronous-append counter so
/// it does not flake on wall-clock.
#[test]
fn batched_property_ingest_fsyncs_once_not_per_record() {
    let (_dir, store, mgr, writer, metrics) = build_stack();

    const N: u32 = 256;

    let sync_appends_baseline = metrics.wal_t1_appends_total();
    let fires_baseline = metrics.total_fires();

    let start = Instant::now();
    let mut tx = mgr.begin(TenantId::DEFAULT);
    for i in 0..N {
        create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(1), // single shared label — mirrors a uniform bulk load
            &sample_blob(i),
        )
        .unwrap();
    }
    commit(tx, &store).unwrap();
    let elapsed = start.elapsed();

    writer.shutdown().unwrap();

    let sync_appends = metrics.wal_t1_appends_total() - sync_appends_baseline;
    let fires = metrics.total_fires() - fires_baseline;
    let rec_per_s = f64::from(N) / elapsed.as_secs_f64();

    eprintln!(
        "[#810] N={N} blob-nodes in ONE tx+commit: sync_appends={sync_appends} \
         fires={fires} elapsed={elapsed:?} (~{rec_per_s:.0} rec/s)"
    );

    // The load-bearing assertion: exactly one synchronous (fsync-
    // blocking) WAL append for the whole batch — the `CommitBundle`.
    // Pre-fix this was N+1 (one PutBlob fsync per node + the commit).
    assert_eq!(
        sync_appends, 1,
        "a batched ingest of {N} property-bearing nodes must fsync ONCE \
         (the CommitBundle), got {sync_appends} synchronous appends — \
         per-record PutBlob fsync regressed (#810)"
    );
}

/// Same invariant for relationships (the `apply_to_rel` blob leg). A
/// transaction creating N property-bearing rels commits with one fsync.
#[test]
fn batched_property_rel_ingest_fsyncs_once_not_per_record() {
    use arcgraph_storage::crud::create_rel;

    let (_dir, store, mgr, writer, metrics) = build_stack();

    const N: u32 = 128;

    // Two endpoint nodes first (their own commit; not under measurement).
    let (a, b) = {
        let mut tx = mgr.begin(TenantId::DEFAULT);
        let a = create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        let b = create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
        commit(tx, &store).unwrap();
        (a, b)
    };

    let sync_appends_baseline = metrics.wal_t1_appends_total();

    let mut tx = mgr.begin(TenantId::DEFAULT);
    for i in 0..N {
        create_rel(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            a,
            b,
            TypeId::new(7),
            &sample_blob(i),
        )
        .unwrap();
    }
    commit(tx, &store).unwrap();

    writer.shutdown().unwrap();

    let sync_appends = metrics.wal_t1_appends_total() - sync_appends_baseline;
    assert_eq!(
        sync_appends, 1,
        "a batched ingest of {N} property-bearing rels must fsync ONCE, \
         got {sync_appends} — per-record PutBlob fsync regressed (#810)"
    );
}
