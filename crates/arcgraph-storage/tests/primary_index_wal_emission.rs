//! Regression for the C-I deadlock in `PrimaryIndex::emit_wal_for`
//! surfaced by the M2-D5 alt review.
//!
//! Before the fix (`emit_wal_for(page_id)` re-latched the page store
//! with a read lock while the caller still held a `write_arc`),
//! constructing a `PrimaryIndex` with `Some(wal)` and performing any
//! write would hang forever — parking_lot's `RwLock` is not
//! re-entrant. The whole test suite missed the bug because every
//! existing test passed `None` for the WAL argument.
//!
//! After the fix (DEC-21: `emit_wal_for_bytes(page_id, &PageBuf)`,
//! called under the held write guard, no re-latch), the same workload
//! completes in milliseconds and produces `WalRecordType::IndexPage`
//! records in the segment file.
//!
//! ## W14ε flake-CLASS fix (issue #282)
//!
//! Originally a 10 s watchdog — comfortable on dev hardware but tight
//! on 2-vCPU GitHub Actions runners under sibling-cargo contention.
//! The watchdog implicitly assumes a hardware throughput floor and
//! fires under CPU starvation even though no deadlock exists.
//! Post-W14ε the watchdog is 60 s for class consistency with the
//! sibling deadlock-only tests; a real C-I re-latch deadlock never
//! completes regardless of budget, so the generous bound captures
//! the property without depending on host throughput.
//!
//! See `docs/testing-strategy.md` §"Hardware-throughput-threshold tests".

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use arcgraph_core::{PageId, TenantId};
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::{
    LEAF_CAPACITY, PageSlot, PrimaryIndex, PrimaryKey, RecordKind,
};
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
         is a strong-signal C-I `emit_wal_for` re-latch deadlock regression, \
         not a slow-host symptom. See `docs/testing-strategy.md` §3.5.",
        wall_cap
    );
}

#[test]
fn primary_index_with_wal_inserts_do_not_deadlock() {
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
        "primary_index_with_wal_inserts_do_not_deadlock",
        Duration::from_secs(60),
        move || {
            let txn_mgr = Arc::new(TxnManager::new());
            let alloc = Arc::new(PageAllocator::new());
            let idx = PrimaryIndex::new(txn_mgr, alloc, Some(handle))
                .expect("PrimaryIndex::new with WAL must not deadlock");

            // One non-splitting insert (the original reproducer).
            let key = PrimaryKey::new(TenantId::DEFAULT, RecordKind::Node, 42);
            idx.insert(key, PageSlot::new(PageId::new(100), SlotId(3)))
                .expect("insert must not deadlock");
            assert_eq!(
                idx.lookup(key).unwrap(),
                Some(PageSlot::new(PageId::new(100), SlotId(3)))
            );

            // Force at least one leaf split → exercises the split-path
            // emit_wal_for_bytes sites (new_buf + SELF under guard).
            for i in 0..=(u64::from(LEAF_CAPACITY) + 5) {
                let k = PrimaryKey::new(TenantId::DEFAULT, RecordKind::Node, 1_000 + i);
                idx.insert(
                    k,
                    PageSlot::new(PageId::new(1 + i), SlotId((i & 0xFF) as u16)),
                )
                .expect("split-triggering insert must not deadlock");
            }
        },
    );

    drop(writer);
}
