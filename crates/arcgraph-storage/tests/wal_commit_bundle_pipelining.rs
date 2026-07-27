//! ADR-031 regression: 8 concurrent writers pipeline into a single
//! `CommitBundle` fire per commit.
//!
//! Pre-fix (ADR-030 only): 2 sequential fires per commit
//! (`IndexPage` drain + `CommitMarker`); `mean batch-at-fire` was
//! 1.54 — the MVCC half pipelined 8-way but the IndexPage half
//! stayed ~1-per-fire because it went out INSIDE the index layer's
//! drain, hitting the WAL one record at a time from the serialized
//! drain loop.
//!
//! Post-ADR-031: 1 fire per commit, carrying a bundle payload of
//! `(MVCC writes + IndexPage entries)`. With 8 concurrent writers
//! each waiting for the single `wal.append(CommitBundle)` ack,
//! arrivals pile up into the group-commit window and
//! `mean batch-at-fire` approaches 8 (the writer count). The test
//! asserts a conservative ≥ 4.0 lower bound to tolerate scheduler
//! jitter under `cargo test` concurrency.
//!
//! Deadlock regression: 20 s watchdog. A broken refactor would
//! either hang (deadlock on `install_order`, the failure mode Slice
//! 4's deferred-persist fix addresses) or land below the batch-at-
//! fire floor.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use arcgraph_core::{LabelId, TenantId};
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

#[test]
fn eight_writers_pipeline_into_single_fire_per_commit() {
    let (_dir, store, mgr, writer, metrics) = build_stack();
    let fires_baseline = metrics.total_fires();
    let records_baseline = metrics.total_records_fired();

    const N_WRITERS: usize = 8;
    const PER_WRITER: u32 = 100;

    let stop = Arc::new(AtomicBool::new(false));
    let commits = Arc::new(AtomicU64::new(0));

    let mut handles = Vec::with_capacity(N_WRITERS);
    for w_id in 0..N_WRITERS {
        let store = Arc::clone(&store);
        let mgr = Arc::clone(&mgr);
        let commits = Arc::clone(&commits);
        let stop = Arc::clone(&stop);
        handles.push(
            thread::Builder::new()
                .name(format!("bundle-pipeline-{w_id}"))
                .spawn(move || {
                    for i in 0..PER_WRITER {
                        if stop.load(Ordering::Relaxed) {
                            return;
                        }
                        let mut tx = mgr.begin(TenantId::DEFAULT);
                        create_node(
                            &store,
                            &mut tx,
                            TenantId::DEFAULT,
                            LabelId::new(((w_id as u32) << 16) | i),
                            &PropertyData::Empty,
                        )
                        .unwrap();
                        commit(tx, &store).unwrap();
                        commits.fetch_add(1, Ordering::Relaxed);
                    }
                })
                .unwrap(),
        );
    }

    let watchdog_deadline = Instant::now() + Duration::from_secs(20);
    for h in handles {
        let remaining = watchdog_deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            stop.store(true, Ordering::Relaxed);
            panic!("writer stuck past 20s watchdog — likely deadlock on install_order");
        }
        h.join().unwrap();
    }

    let total_commits = commits.load(Ordering::Relaxed);
    assert_eq!(
        total_commits,
        (N_WRITERS as u64) * (PER_WRITER as u64),
        "every writer must complete its budget"
    );

    writer.shutdown().unwrap();

    let fires = metrics.total_fires() - fires_baseline;
    let records = metrics.total_records_fired() - records_baseline;
    let records_per_commit = records as f64 / total_commits as f64;
    let mean_batch_at_fire = records as f64 / fires as f64;

    // Post-fix records/commit must be close to 1.00 on a no-split
    // insert workload. Tolerate a small slack for the occasional
    // grow_root's deferred SYSTEM commit (each grow_root adds one
    // extra CommitBundle outside the main per-commit fire). Over
    // 800 commits with ~203-per-leaf fanout, grow_root fires at
    // most a couple of times.
    assert!(
        (0.95..=1.10).contains(&records_per_commit),
        "records/commit outside [0.95, 1.10]: {records_per_commit:.3} \
         (records={records}, commits={total_commits})"
    );

    // Pipelining lower bound: with 8 writers pipelining, each fire
    // should on average batch several bundles together. Conservative
    // floor 4.0 tolerates scheduler jitter under `cargo test`
    // concurrency while still catching a regression that collapses
    // to 1 record per fire.
    assert!(
        mean_batch_at_fire >= 4.0,
        "mean batch-at-fire {mean_batch_at_fire:.2} < 4.0 — pipelining regressed"
    );
}

#[test]
fn commit_bundle_pipelining_has_no_deadlock_watchdog() {
    // Watchdog-only regression: if Slice 4's deferred-persist fix
    // ever reverts or the bundle builder re-introduces a nested
    // commit, this test hangs (grow_root → persist_root_to_mvcc →
    // inner Phase 3 blocks on install_order awaiting the outer's
    // Phase 3 which can't run until the builder returns).
    //
    // ## W14ε flake-CLASS fix (issue #282 / sibling of #270)
    //
    // The test name carries the §3.5.1 `*_no_deadlock_*` pattern and
    // the watchdog is the only correctness signal — there is no
    // load-bearing throughput threshold to preserve. Pre-W14ε the
    // 10 s deadline was tight enough to flake under 2-vCPU CI
    // sibling-cargo contention. Per `docs/testing-strategy.md` §3.5.1
    // we use the 60 s class default: a real install_order deadlock
    // never completes regardless of budget, so widening the watchdog
    // reduces hardware-throughput false positives without weakening
    // the deadlock-regression signal.
    let (_dir, store, mgr, writer, _metrics) = build_stack();

    // Enough nodes in ONE transaction to force a root split + some
    // internal splits. The `bootstrap_primary_index_walks_rotated_
    // pages` test in the storage lib used to deadlock here; this
    // test is its external-integration twin.
    let mut tx = mgr.begin(TenantId::DEFAULT);
    // Leaf fanout is 203; insert ~2.5x to trigger split cascade +
    // grow_root.
    for i in 0..512u32 {
        create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(i),
            &PropertyData::Empty,
        )
        .unwrap();
    }

    // Watchdog via `thread::scope` — non-'static refs to mgr+tx
    // are allowed inside the scope. A separate watchdog thread
    // polls completion and fires a panic if the deadline is hit.
    // 60 s budget = §3.5.1 class default; sized for the slowest
    // plausible CPU-starved CI host (deadlock never completes
    // regardless of budget). See `docs/testing-strategy.md` §3.5.
    const WATCHDOG: Duration = Duration::from_secs(60);
    let deadline = Instant::now() + WATCHDOG;
    let done = Arc::new(AtomicBool::new(false));
    let done_watcher = Arc::clone(&done);
    thread::scope(|s| {
        let commit_handle = s.spawn(|| {
            let r = commit(tx, &store);
            done.store(true, Ordering::Release);
            r.expect("commit");
        });
        let watchdog = s.spawn(move || {
            while Instant::now() < deadline {
                if done_watcher.load(Ordering::Acquire) {
                    return;
                }
                thread::sleep(Duration::from_millis(100));
            }
            if !done_watcher.load(Ordering::Acquire) {
                panic!(
                    "commit_with_bundle(builder) hung past {WATCHDOG:?} — budget is the §3.5.1 \
                     class default (sized wide to absorb CI starvation), so non-completion is a \
                     strong-signal deadlock — likely a nested commit inside the builder \
                     re-introduced the install_order deadlock. See `docs/testing-strategy.md` §3.5."
                );
            }
        });
        commit_handle.join().unwrap();
        watchdog.join().unwrap();
    });
    writer.shutdown().unwrap();
}
