//! ADR-030 regression: concurrent writers must pipeline their
//! `IndexPage` WAL appends into the group-commit window instead of
//! serializing one-per-fire through the index's `write_gate`.
//!
//! Pre-fix (DEC-21 as written): `PrimaryIndex::write` held
//! `write_gate: Mutex<()>` across every internal
//! `emit_wal_for_bytes(...)` call. Because `wal.append` blocks until
//! the group-commit fires (max 1 ms window, batch ≤ 16), the gate
//! held-across-`append` serializes all writers one-per-fire. The
//! observable evidence is `mean batch-at-fire ≈ 1.0` for the
//! `IndexPage` stream even under heavy concurrency.
//!
//! Post-fix (ADR-030, three-phase commit inside the index):
//!
//!   1. mutate + install under gate + latches (holding no WAL);
//!   2. drop gate and latches;
//!   3. drain staged emits via `wal.append` — no locks held.
//!
//! Multiple writers now race into the append channel concurrently,
//! so the group-commit window sees batches with size ≥ 2.
//!
//! Load-bearing assertions:
//! - every key inserted is findable post-run (correctness);
//! - `total_records_fired / total_fires` ≥ 2.0 (the "fix is active"
//!   probe);
//! - a concurrent reader probe observes no torn reads (reuses the
//!   `concurrent_readers_never_see_partial_state` pattern).
//!
//! The watchdog is set to 20 s; a regression would either time out
//! (deadlock) or land below the mean-batch bar rather than hang CI.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use arcgraph_core::{PageId, TenantId};
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::{PageSlot, PrimaryIndex, PrimaryKey, RecordKind};
use arcgraph_storage::records::SlotId;
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

fn key(id: u64) -> PrimaryKey {
    PrimaryKey::new(TenantId::DEFAULT, RecordKind::Node, id)
}

fn slot(id: u64) -> PageSlot {
    PageSlot::new(PageId::new(id + 1), SlotId((id & 0xFF) as u16))
}

fn run_with_watchdog<F: FnOnce() + Send + 'static>(label: &str, wall_cap: Duration, f: F) {
    let done = Arc::new(AtomicBool::new(false));
    let done_for_worker = Arc::clone(&done);
    let handle = thread::Builder::new()
        .name(format!("watchdog-worker-{label}"))
        .spawn(move || {
            f();
            done_for_worker.store(true, Ordering::Release);
        })
        .expect("spawn worker");
    let deadline = Instant::now() + wall_cap;
    while Instant::now() < deadline {
        if done.load(Ordering::Acquire) {
            handle.join().expect("worker panicked");
            return;
        }
        thread::sleep(Duration::from_millis(50));
    }
    panic!(
        "{label} did not complete within {:?} — the budget is sized per \
         `docs/testing-strategy.md` §3.5 (Option A class default or Option B \
         forward-pin per caller), so non-completion is a strong-signal ADR-030 \
         regression (gate-across-append serialization), not a slow-host symptom.",
        wall_cap
    );
}

/// 8 writers × `PER_WRITER` upserts land, and the WAL writer's
/// mean batch-at-fire exceeds 2.0 — i.e. concurrent writers actually
/// coalesce into the group-commit window.
#[test]
fn primary_index_concurrent_writers_coalesce_wal_fires() {
    const WRITERS: u64 = 8;
    const PER_WRITER: u64 = 400;

    let tmp = TempDir::new().expect("tempdir");
    let cfg = test_wal_config(tmp.path().to_path_buf());
    let writer = WalWriter::spawn(cfg).expect("spawn wal writer");
    let wal = writer.handle();
    let fire_metrics: WalFireMetrics = writer.fire_metrics();
    let fire_metrics_for_assert = fire_metrics.clone();

    run_with_watchdog(
        "primary_index_concurrent_writers_coalesce_wal_fires",
        Duration::from_secs(20),
        move || {
            let txn_mgr = Arc::new(TxnManager::new());
            let alloc = Arc::new(PageAllocator::new());
            let idx =
                Arc::new(PrimaryIndex::new(txn_mgr, alloc, Some(wal)).expect("new primary idx"));

            // Snapshot the fires counter before the concurrent phase
            // so we measure the fix's impact on THIS workload, not on
            // `new()`'s bootstrap emits.
            let fires_before = fire_metrics.total_fires();
            let records_before = fire_metrics.total_records_fired();

            let mut handles = Vec::with_capacity(WRITERS as usize);
            for w in 0..WRITERS {
                let idx = Arc::clone(&idx);
                // Disjoint key ranges per writer. Each writer's 400
                // inserts span multiple leaves, so every writer hits
                // many `stage_emit` + `drain` cycles under contention.
                let start = w * PER_WRITER;
                let end = start + PER_WRITER;
                handles.push(thread::spawn(move || {
                    for i in start..end {
                        idx.insert(key(i), slot(i))
                            .unwrap_or_else(|e| panic!("writer {w} insert {i}: {e}"));
                    }
                }));
            }
            for h in handles {
                h.join().expect("writer thread panicked");
            }

            // Correctness: every key findable.
            let total = WRITERS * PER_WRITER;
            for i in 0..total {
                assert_eq!(
                    idx.lookup(key(i)).unwrap(),
                    Some(slot(i)),
                    "key {i} missing after concurrent writers"
                );
            }

            // Coalescence: compute mean batch-at-fire over the
            // concurrent-writer window. Pre-fix would give ≈ 1.0 (one
            // IndexPage record per fire, because the gate was held
            // across `wal.append`). Post-fix we expect well above 2.0
            // under 8-way contention with a 1 ms window.
            let fires_after = fire_metrics_for_assert.total_fires();
            let records_after = fire_metrics_for_assert.total_records_fired();
            let fires_delta = fires_after - fires_before;
            let records_delta = records_after - records_before;
            assert!(
                fires_delta > 0,
                "no fires observed during concurrent phase: records_delta={records_delta}"
            );
            let mean = records_delta as f64 / fires_delta as f64;
            assert!(
                mean >= 2.0,
                "mean batch-at-fire too low: {mean:.3} \
                 (records_delta={records_delta}, fires_delta={fires_delta}) — \
                 likely an ADR-030 regression (gate held across wal.append)"
            );
        },
    );

    drop(writer);
}

/// Concurrent readers during a heavy writer workload never observe a
/// torn read. Asserts the ADR-030 "install-before-drop-gate" ordering
/// — readers either see `None` (key not yet published) or the exact
/// slot a writer installed.
///
/// W14ε flake-CLASS fix (issue #282 class-consistency tweak): the
/// load-bearing assertion is correctness (no torn reads), NOT
/// throughput — so under `docs/testing-strategy.md` §3.5.1 the
/// watchdog is raised to the 60 s class default. The pre-W14ε 20 s
/// budget was never observed flaking (this test was not on the
/// #282 / #270 named list), but the workload shape — `run_with_
/// watchdog` wrapping a fixed-op-count thread soup — is identical
/// to the deadlock-regression siblings, and class-consistency keeps
/// future flake investigators from re-deriving why one watchdog
/// budget differs from its siblings.
#[test]
fn primary_index_readers_see_no_torn_state_with_wal() {
    const WRITERS: u64 = 2;
    const PER_WRITER: u64 = 400;
    const READERS: usize = 2;
    const READ_ITERS: usize = 2_000;

    let tmp = TempDir::new().expect("tempdir");
    let cfg = test_wal_config(tmp.path().to_path_buf());
    let writer = WalWriter::spawn(cfg).expect("spawn wal writer");
    let wal = writer.handle();

    run_with_watchdog(
        "primary_index_readers_see_no_torn_state_with_wal",
        Duration::from_secs(60),
        move || {
            let txn_mgr = Arc::new(TxnManager::new());
            let alloc = Arc::new(PageAllocator::new());
            let idx =
                Arc::new(PrimaryIndex::new(txn_mgr, alloc, Some(wal)).expect("new primary idx"));

            let total = WRITERS * PER_WRITER;
            let mut handles = Vec::with_capacity(WRITERS as usize + READERS);
            for w in 0..WRITERS {
                let idx = Arc::clone(&idx);
                handles.push(thread::spawn(move || {
                    for j in 0..PER_WRITER {
                        let id = j * WRITERS + w; // interleaved
                        idx.insert(key(id), slot(id))
                            .unwrap_or_else(|e| panic!("writer {w} insert {id}: {e}"));
                    }
                }));
            }
            for _ in 0..READERS {
                let idx = Arc::clone(&idx);
                handles.push(thread::spawn(move || {
                    for _ in 0..READ_ITERS {
                        for id in 0..total {
                            match idx.lookup(key(id)).unwrap() {
                                None => {}
                                Some(found) => assert_eq!(
                                    found,
                                    slot(id),
                                    "torn read on key {id}: saw {found:?}"
                                ),
                            }
                        }
                    }
                }));
            }
            for h in handles {
                h.join().expect("thread panicked");
            }

            for id in 0..total {
                assert_eq!(
                    idx.lookup(key(id)).unwrap(),
                    Some(slot(id)),
                    "key {id} missing after concurrent writers"
                );
            }
        },
    );

    drop(writer);
}
