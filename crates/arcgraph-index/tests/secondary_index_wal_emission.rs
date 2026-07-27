//! Regression for the C-I deadlock in `SecondaryIndex::emit_wal_for`
//! surfaced by the M2-D5 alt review.
//!
//! Pre-fix: `emit_wal_for(page_id)` re-latched the page store with a
//! read lock while the caller still held a `write_arc`. parking_lot's
//! `RwLock` is not re-entrant, so every secondary write with
//! `Some(wal)` configured deadlocked the thread. The entire pre-fix
//! test suite missed the bug because every test passed `None`.
//!
//! Post-fix (DEC-21: `emit_wal_for_bytes(page_id, &PageBuf)`, called
//! under the held write guard — no re-latch), the same workload
//! completes in milliseconds and produces `WalRecordType::IndexPage`
//! records.
//!
//! ## W14ε flake-CLASS fix (issue #282 / closes #270)
//!
//! The watchdog was originally 10 s — comfortable on dev hardware
//! (workload finishes in tens of ms) but tight on 2-vCPU GitHub
//! Actions runners under sibling-cargo contention. Issue #270 caught
//! the pattern: the watchdog implicitly assumes a hardware throughput
//! floor and fires under CPU starvation even though no deadlock
//! exists. Post-W14ε the watchdog is 60 s — a real deadlock never
//! completes regardless of budget, so a generous bound captures the
//! property ("no deadlock") without depending on host throughput.
//!
//! See `docs/testing-strategy.md` §"Hardware-throughput-threshold tests".
//!
//! Coverage:
//! - single-entry insert (non-split path)
//! - split-forcing insert (stage-WAL-publish split path)
//! - fifth-duplicate → overflow-head allocation (overflow stage-WAL-publish)
//! - overflow-page-full → new-tail allocation (chain append path)
//! - remove in inline + remove in overflow chain (tombstone WAL paths)

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use arcgraph_core::{LabelId, NodeId, StringId, TenantId};
use arcgraph_index::{
    LEAF_CAPACITY, OVERFLOW_SLOTS_PER_PAGE, PropertyValue, SecondaryIndex, SecondaryKey,
};
use arcgraph_storage::page_alloc::PageAllocator;
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
        "{label} did not complete within {:?} — the budget is the §3.5.1 \
         class default (sized wide to absorb CI starvation), so non-completion \
         is a strong-signal C-I `emit_wal_for` re-latch deadlock regression, \
         not a slow-host symptom. See `docs/testing-strategy.md` §3.5.",
        wall_cap
    );
}

#[test]
fn secondary_index_with_wal_writes_do_not_deadlock() {
    let tmp = TempDir::new().expect("tempdir");
    let cfg = test_wal_config(tmp.path().to_path_buf());
    let writer = WalWriter::spawn(cfg).expect("spawn wal writer");
    let handle = writer.handle();

    // 60 s watchdog: generous floor for the slowest plausible CI host
    // (2-vCPU + sibling cargo contention). A real C-I `emit_wal_for`
    // re-latch deadlock never completes regardless of budget, so the
    // generous bound reduces flake without weakening the property.
    // See module-level doc for the W14ε class-fix rationale.
    run_with_watchdog(
        "secondary_index_with_wal_writes_do_not_deadlock",
        Duration::from_secs(60),
        move || {
            let txn_mgr = Arc::new(TxnManager::new());
            let alloc = Arc::new(PageAllocator::new());
            let idx = SecondaryIndex::new(txn_mgr, alloc, Some(handle))
                .expect("SecondaryIndex::new with WAL must not deadlock");

            // 1. Single-entry insert — exercises non-split leaf
            //    emit_wal_for_bytes under held guard.
            let k0 = key(1, 42);
            idx.insert(k0, NodeId::new(1)).expect("non-split insert");
            assert_eq!(idx.lookup(k0).unwrap(), vec![NodeId::new(1)]);

            // 2. Four inline duplicates on the same key (first empty
            //    slot path).
            for i in 2..=4u64 {
                idx.insert(k0, NodeId::new(i)).expect("inline-slot insert");
            }

            // 3. Fifth duplicate forces overflow-head allocation
            //    (stage-WAL-publish for the fresh overflow page +
            //    leaf WAL for updated overflow_head).
            idx.insert(k0, NodeId::new(5))
                .expect("overflow-head allocating insert");
            assert_eq!(idx.lookup(k0).unwrap().len(), 5);

            // 4. Saturate an overflow page, then push past to force
            //    new-tail allocation. Each append-to-existing-tail
            //    exercises the in-place-mutate + emit-under-guard
            //    path; the transition hits the allocate-successor
            //    path once.
            let k1 = key(1, 999);
            idx.insert(k1, NodeId::new(100)).unwrap();
            // Fill 4 inline + OVERFLOW_SLOTS_PER_PAGE overflow + 1 more
            // → overflow chain grows to a second page.
            for i in 101..=(100 + 4 + OVERFLOW_SLOTS_PER_PAGE as u64) {
                idx.insert(k1, NodeId::new(i))
                    .expect("append-to-tail insert must not deadlock");
            }
            let hits = idx.lookup(k1).unwrap();
            assert_eq!(hits.len(), 1 + 4 + OVERFLOW_SLOTS_PER_PAGE);

            // 5. Force at least one leaf split — exercises
            //    apply_leaf_insert's split path (new_id + leaf WAL
            //    under guard, install-after-WAL).
            for i in 0..=(u32::from(LEAF_CAPACITY) + 5) {
                idx.insert(key(2, i), NodeId::new(10_000 + u64::from(i)))
                    .expect("split-triggering insert must not deadlock");
            }

            // 6. Remove in inline position.
            assert!(idx.remove(k0, NodeId::new(1)).unwrap());
            // 7. Remove in overflow chain.
            assert!(idx.remove(k0, NodeId::new(5)).unwrap());
            let after = idx.lookup(k0).unwrap();
            assert!(!after.contains(&NodeId::new(1)));
            assert!(!after.contains(&NodeId::new(5)));
        },
    );

    drop(writer);
}
