//! ADR-030 regression: 8-writer stress against `PrimaryIndex` with
//! WAL enabled must not deadlock.
//!
//! The pre-fix DEC-21 shape (`write_gate` held across
//! `emit_wal_for_bytes` → `wal.append` → blocking on group-commit
//! fsync) could not deadlock on its own — every writer was globally
//! serialized. ADR-030 moves the gate release BEFORE the WAL append,
//! so the staged-emit plumbing (`StagedEmit` vector threaded through
//! `apply_leaf_op` / `apply_internal_insert` / `grow_root`, drained
//! outside the gate) introduces a new failure mode if wrong: a
//! writer could release a latch while still holding some other lock
//! that a peer needs.
//!
//! ## W14ε flake-CLASS fix (issue #282 / closes #270)
//!
//! Pre-W14ε this test was wall-clock-bounded: each writer looped for
//! 2 s and the assertion required ≥ 1 000 inserts (a hardware-throughput
//! threshold). On 2-vCPU GitHub Actions runners under sibling-cargo
//! contention the floor was occasionally missed, producing a
//! deterministic-looking failure for what was actually CPU starvation.
//!
//! Post-W14ε we test the property the test name suggests — "8 writers
//! make progress under contention without deadlock" — by:
//!
//!   - giving each writer a fixed `PER_WRITER` op count (no wall-clock
//!     loop). Total work is bounded, so the assertion is "every writer
//!     completed every assigned op", which is hardware-independent;
//!   - protecting the workload with a generous 60 s watchdog. A real
//!     ADR-030 regression manifests as the watchdog firing (genuine
//!     deadlock never completes regardless of budget), while a slow
//!     CI host simply takes longer to finish. The watchdog stays well
//!     above any plausible CPU-starvation scenario.
//!
//! See `docs/testing-strategy.md` §"Hardware-throughput-threshold tests"
//! for the class-level pattern. The Option B fallback (workspace-level
//! test sequencer / serial_test) is forward-pinned for tests that
//! genuinely require throughput-bound assertions (e.g.
//! `*_concurrent_writers_coalesce_wal_fires`).

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use arcgraph_core::{PageId, TenantId};
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::{PageSlot, PrimaryIndex, PrimaryKey, RecordKind};
use arcgraph_storage::records::SlotId;
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{WalConfig, WalWriter};
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
        "{label} did not complete within {:?} — the budget is the §3.5.1 \
         class default (sized wide to absorb CI starvation), so non-completion \
         is a strong-signal ADR-030 deadlock regression, not a slow-host \
         symptom. See `docs/testing-strategy.md` §3.5.",
        wall_cap
    );
}

#[test]
fn primary_index_8_writers_make_progress_for_2s() {
    // Name preserved as a stable CI identifier across the W14ε
    // flake-CLASS fix; the "_for_2s" suffix dates from the pre-W14ε
    // wall-clock-bounded shape and is now historical. The current
    // shape verifies the same property — 8 writers make progress under
    // contention without deadlock — via a fixed per-writer op count,
    // which is hardware-independent. See module-level doc for context.
    const WRITERS: u64 = 8;
    // Each writer commits PER_WRITER inserts. Sized to span multiple
    // leaves under interleaved keys (8 × 200 = 1 600 keys triggers
    // several splits per writer-pair) while staying under the 60 s
    // watchdog on the slowest plausible CI host. On dev hardware the
    // workload finishes in well under a second.
    const PER_WRITER: u64 = 200;
    const WATCHDOG: Duration = Duration::from_secs(60);

    let tmp = TempDir::new().expect("tempdir");
    let cfg = test_wal_config(tmp.path().to_path_buf());
    let writer = WalWriter::spawn(cfg).expect("spawn wal writer");
    let wal = writer.handle();

    run_with_watchdog(
        "primary_index_8_writers_make_progress_for_2s",
        WATCHDOG,
        move || {
            let txn_mgr = Arc::new(TxnManager::new());
            let alloc = Arc::new(PageAllocator::new());
            let idx =
                Arc::new(PrimaryIndex::new(txn_mgr, alloc, Some(wal)).expect("new primary idx"));

            let total_inserted = Arc::new(AtomicU64::new(0));
            let mut handles = Vec::with_capacity(WRITERS as usize);
            for w in 0..WRITERS {
                let idx = Arc::clone(&idx);
                let counter = Arc::clone(&total_inserted);
                handles.push(thread::spawn(move || {
                    for i in 0..PER_WRITER {
                        // Interleaved key space so writers contend on
                        // the same ancestor leaves (not partitioned
                        // into disjoint subtrees).
                        let id = i * WRITERS + w;
                        idx.insert(
                            PrimaryKey::new(TenantId::DEFAULT, RecordKind::Node, id),
                            PageSlot::new(PageId::new(id + 1), SlotId((id & 0xFF) as u16)),
                        )
                        .unwrap_or_else(|e| panic!("writer {w} insert {id}: {e}"));
                        counter.fetch_add(1, Ordering::Relaxed);
                    }
                }));
            }
            for h in handles {
                h.join().expect("writer thread panicked");
            }

            // PROGRESS-based oracle: every writer completed every
            // assigned op. This disproves both circular-wait deadlock
            // (some writer would never finish) and latch-order
            // inversion (some writer would error). Hardware-independent
            // — the only timing dependence is the outer watchdog,
            // which is sized for a real deadlock signal, not for a
            // throughput floor.
            let total = total_inserted.load(Ordering::Relaxed);
            let expected = WRITERS * PER_WRITER;
            assert_eq!(
                total, expected,
                "writer-progress total {total} ≠ expected {expected} — \
                 some writer dropped an insert mid-loop (regression)",
            );
        },
    );

    drop(writer);
}
