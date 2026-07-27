//! ADR-030 regression mirror for `SecondaryIndex`: concurrent writers
//! must pipeline their `IndexPage` WAL appends into the group-commit
//! window instead of serializing one-per-fire through the tree's
//! `write_gate`.
//!
//! Matches the shape of
//! `arcgraph-storage::tests::primary_index_pipelined_wal_emits`, with
//! a secondary-specific twist: we also exercise the overflow-chain
//! append path (DEC-22 tail-cache), which under ADR-030 stages each
//! chain-tail/head mutation to `staged: Vec<StagedEmit>` and drains
//! after the gate drops. A regression here would either deadlock
//! (watchdog) or land mean batch-at-fire ≈ 1.0.
//!
//! Load-bearing assertions:
//! - every (key, node) tuple findable post-run;
//! - `total_records_fired / total_fires` ≥ 2.0 during the concurrent
//!   phase (the ADR-030 "fix is active" probe).
//!
//! ## W14ε flake-CLASS fix (issue #282)
//!
//! `secondary_index_overflow_chain_no_deadlock_with_wal` was
//! observed flaking on W13β PR #287 CI under sibling-cargo
//! contention — the 10 s watchdog was tight on 2-vCPU GitHub Actions
//! runners. Post-W14ε the deadlock-only tests use a 60 s watchdog;
//! a real deadlock never completes regardless of budget, so a
//! generous bound captures the property ("no deadlock") without
//! depending on host throughput.
//!
//! `secondary_index_concurrent_writers_coalesce_wal_fires` retains
//! its 20 s watchdog (already generous) AND its `mean >= 2.0`
//! assertion (the ADR-030 "fix is active" probe — load-bearing). If
//! that test flakes on heavy CI, the forward path is Option B per
//! `docs/testing-strategy.md` §"Hardware-throughput-threshold tests"
//! (workspace-level test sequencer / `serial_test`). It is
//! intentionally NOT touched by this slice because the assertion
//! IS the property — relaxing it would weaken ADR-030 evidence.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use arcgraph_core::{LabelId, NodeId, StringId, TenantId};
use arcgraph_index::{PropertyValue, SecondaryIndex, SecondaryKey};
use arcgraph_storage::page_alloc::PageAllocator;
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

fn key(label: u32, v: u32) -> SecondaryKey {
    SecondaryKey::new(
        TenantId::DEFAULT,
        LabelId::new(label),
        StringId::new(1),
        PropertyValue::U32(v),
    )
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

/// 8 writers × `PER_WRITER` disjoint-key inserts land, and the WAL
/// writer's mean batch-at-fire exceeds 2.0 for the concurrent phase.
#[test]
fn secondary_index_concurrent_writers_coalesce_wal_fires() {
    const WRITERS: u32 = 8;
    const PER_WRITER: u32 = 400;

    let tmp = TempDir::new().expect("tempdir");
    let cfg = test_wal_config(tmp.path().to_path_buf());
    let writer = WalWriter::spawn(cfg).expect("spawn wal writer");
    let wal = writer.handle();
    let fire_metrics: WalFireMetrics = writer.fire_metrics();
    let fire_metrics_for_assert = fire_metrics.clone();

    run_with_watchdog(
        "secondary_index_concurrent_writers_coalesce_wal_fires",
        Duration::from_secs(20),
        move || {
            let txn_mgr = Arc::new(TxnManager::new());
            let alloc = Arc::new(PageAllocator::new());
            let idx =
                Arc::new(SecondaryIndex::new(txn_mgr, alloc, Some(wal)).expect("new secondary"));

            let fires_before = fire_metrics.total_fires();
            let records_before = fire_metrics.total_records_fired();

            let mut handles = Vec::with_capacity(WRITERS as usize);
            for w in 0..WRITERS {
                let idx = Arc::clone(&idx);
                let start = w * PER_WRITER;
                let end = start + PER_WRITER;
                handles.push(thread::spawn(move || {
                    for v in start..end {
                        idx.insert(key(1, v), NodeId::new(u64::from(v) + 1))
                            .unwrap_or_else(|e| panic!("writer {w} insert {v}: {e}"));
                    }
                }));
            }
            for h in handles {
                h.join().expect("writer thread panicked");
            }

            let total = WRITERS * PER_WRITER;
            for v in 0..total {
                assert_eq!(
                    idx.lookup(key(1, v)).unwrap(),
                    vec![NodeId::new(u64::from(v) + 1)],
                    "value {v} missing after concurrent writers"
                );
            }

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

/// Duplicate-heavy append stress: 8 writers pile `NodeId`s into ONE
/// bucket, forcing overflow-chain growth. Every NodeId must be
/// findable post-run — nothing dropped, nothing duplicated — and the
/// workload must not deadlock under 10 s.
#[test]
fn secondary_index_overflow_chain_no_deadlock_with_wal() {
    const WRITERS: u64 = 8;
    const PER_WRITER: u64 = 200;

    let tmp = TempDir::new().expect("tempdir");
    let cfg = test_wal_config(tmp.path().to_path_buf());
    let writer = WalWriter::spawn(cfg).expect("spawn wal writer");
    let wal = writer.handle();

    // 60 s watchdog: generous floor for the slowest plausible CI host
    // (2-vCPU + sibling cargo contention). A real ADR-030 staged-emit
    // deadlock never completes regardless of budget, so the generous
    // bound reduces flake without weakening the property. See
    // module-level doc for the W14ε class-fix rationale.
    run_with_watchdog(
        "secondary_index_overflow_chain_no_deadlock_with_wal",
        Duration::from_secs(60),
        move || {
            let txn_mgr = Arc::new(TxnManager::new());
            let alloc = Arc::new(PageAllocator::new());
            let idx =
                Arc::new(SecondaryIndex::new(txn_mgr, alloc, Some(wal)).expect("new secondary"));

            let bucket = key(1, 7);
            let mut handles = Vec::with_capacity(WRITERS as usize);
            for w in 0..WRITERS {
                let idx = Arc::clone(&idx);
                let start = 1 + w * PER_WRITER; // NodeId 0 is reserved.
                let end = start + PER_WRITER;
                handles.push(thread::spawn(move || {
                    for i in start..end {
                        idx.insert(bucket, NodeId::new(i))
                            .unwrap_or_else(|e| panic!("writer {w} insert {i}: {e}"));
                    }
                }));
            }
            for h in handles {
                h.join().expect("writer thread panicked");
            }

            let hits = idx.lookup(bucket).unwrap();
            assert_eq!(
                hits.len(),
                (WRITERS * PER_WRITER) as usize,
                "duplicate-heavy overflow chain lost inserts"
            );
            for w in 0..WRITERS {
                for i in (1 + w * PER_WRITER)..(1 + (w + 1) * PER_WRITER) {
                    assert!(
                        hits.contains(&NodeId::new(i)),
                        "NodeId({i}) from writer {w} missing"
                    );
                }
            }
        },
    );

    drop(writer);
}
