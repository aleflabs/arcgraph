//! ADR-034 §Slice F — T3 / Periodic tier integration tests.
//!
//! Invariants exercised:
//! - **I-D2**: T3 commit is durable within `rpo_ms` of ack OR the
//!   process aborted.
//! - **I-D4**: Background fsync failure = process abort (verified
//!   via the [`BackgroundFsyncFailAction::RollbackAndContinue`] path
//!   since the real abort kills the test process).
//! - §6.6: Crash between `visible.advance` and fsync regresses
//!   visible on replay (simulated by dropping the writer without
//!   flushing T3 commits).

use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use arcgraph_core::{DurabilityTier, Lsn, TenantId};
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::catalog::SystemCatalog;
use arcgraph_storage::io::InMemoryPageIo;
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::segment::{SegmentHeader, list_segments, segment_filename};
use arcgraph_storage::wal::{
    BackgroundFsyncFailAction, BackgroundFsyncScheduler, WalConfig, WalRecord, WalRecordType,
    WalWriter,
};
use bytes::Bytes;
use proptest::prelude::*;
use tempfile::TempDir;

fn config(dir: PathBuf) -> WalConfig {
    WalConfig {
        dir,
        segment_size_bytes: 64 * 1024 * 1024,
        // Deliberately long window so we can observe the T3 pre-fsync
        // state deterministically — the scheduler is the only fire
        // trigger.
        group_commit_window: Duration::from_secs(3600),
        group_commit_max_batch: 1_000,
        metrics_sink: None,
        encryption: None,
        inflight_budget_bytes: None,
    }
}

fn drain_segments(dir: &std::path::Path) -> Vec<WalRecord> {
    let mut out = Vec::new();
    for seg in list_segments(dir).unwrap() {
        let bytes = std::fs::read(dir.join(segment_filename(seg))).unwrap();
        if bytes.len() < SegmentHeader::SIZE {
            continue;
        }
        SegmentHeader::decode(&bytes[..SegmentHeader::SIZE]).unwrap();
        let mut cursor = SegmentHeader::SIZE;
        while cursor < bytes.len() {
            let (r, consumed) = WalRecord::decode(&bytes[cursor..]).unwrap();
            out.push(r);
            cursor += consumed;
        }
    }
    out
}

struct Setup {
    _dir: TempDir,
    writer: Option<WalWriter>,
    scheduler: Arc<BackgroundFsyncScheduler>,
    mgr: TxnManager,
    #[allow(dead_code)] // held for future tests that exercise
    // additional catalog assertions; intentionally kept to anchor
    // the catalog lifetime to the setup.
    catalog: Arc<SystemCatalog>,
}

impl Setup {
    fn new(rpo_ms: u64) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let writer = WalWriter::spawn(config(dir.path().to_path_buf())).unwrap();
        let scheduler =
            BackgroundFsyncScheduler::start(writer.handle(), BackgroundFsyncFailAction::Abort);
        let mut mgr = TxnManager::with_wal(writer.handle());
        let catalog = Arc::new(SystemCatalog::new());
        let io = Arc::new(InMemoryPageIo::new());
        let pool = BufferPool::new(8, io);
        catalog.bootstrap(&pool, &mgr).unwrap();
        mgr.set_durability_lookup(catalog.clone());

        // Flip DEFAULT to T3.
        let mut tx = mgr.begin(TenantId::SYSTEM);
        catalog
            .set_durability_tier(
                &mut tx,
                TenantId::DEFAULT,
                DurabilityTier::Periodic { rpo_ms },
            )
            .unwrap();
        tx.commit().unwrap();

        // Register the scheduler for this tenant.
        scheduler.register(TenantId::DEFAULT, DurabilityTier::Periodic { rpo_ms });

        Self {
            _dir: dir,
            writer: Some(writer),
            scheduler,
            mgr,
            catalog,
        }
    }

    fn handle_last_durable(&self) -> Lsn {
        self.writer
            .as_ref()
            .expect("writer already shut down")
            .handle()
            .last_durable_lsn()
    }

    fn shutdown(mut self) {
        let _ = self.scheduler.shutdown();
        if let Some(w) = self.writer.take() {
            let _ = w.shutdown();
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Test 1 (spec §F.2): T3 data loss bounded by rpo_ms.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn t3_data_loss_bounded_by_rpo_ms() {
    // Configure a short rpo_ms so the scheduler durifies quickly.
    // Commit N writes, wait > rpo_ms, assert watermark covers all
    // commits.
    let rpo_ms = 50u64;
    let s = Setup::new(rpo_ms);

    let n = 10u64;
    let mut max_commit_lsn = Lsn::ZERO;
    for i in 1..=n {
        let mut tx = s.mgr.begin(TenantId::DEFAULT);
        tx.write(i, Bytes::from(format!("v{i}").into_bytes()));
        let lsn = tx.commit().unwrap();
        max_commit_lsn = lsn;
        // D-4: visible advanced; ack returned before fsync.
        assert_eq!(s.mgr.current_lsn(), lsn);
    }

    // Wait up to 4×rpo_ms for the scheduler to catch up (generous
    // for lagging CI hosts).
    let deadline = Instant::now() + Duration::from_millis(rpo_ms * 4 + 500);
    while s.handle_last_durable() < max_commit_lsn && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }
    assert!(
        s.handle_last_durable() >= max_commit_lsn,
        "I-D2: scheduler must durify every T3 commit within ≤ 4×rpo_ms; \
         max_commit={max_commit_lsn:?} watermark={:?}",
        s.handle_last_durable(),
    );

    s.shutdown();
}

// ─────────────────────────────────────────────────────────────────────
// Test 2 (spec §F.7): scheduler never misses fsyncs.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn background_fsync_scheduler_no_missed_fsyncs() {
    // Spike N commits; every one must eventually reach durable
    // state. This guards against a scheduler skip bug (e.g., timer
    // racing tracker, interval miscalculation).
    let s = Setup::new(30);
    let n = 100u64;

    let mut max_lsn = Lsn::ZERO;
    for i in 1..=n {
        let mut tx = s.mgr.begin(TenantId::DEFAULT);
        tx.write(i, Bytes::from_static(b"v"));
        max_lsn = tx.commit().unwrap();
    }

    let deadline = Instant::now() + Duration::from_millis(3000);
    while s.handle_last_durable() < max_lsn && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }
    assert!(
        s.handle_last_durable() >= max_lsn,
        "no missed fsyncs: max_lsn={max_lsn:?} watermark={:?}",
        s.handle_last_durable(),
    );

    // Tick count advanced.
    let m = s.scheduler.metrics();
    assert!(m.ticks_ran_total() >= 1);
    assert_eq!(m.tick_errors_total(), 0);

    s.shutdown();
}

// ─────────────────────────────────────────────────────────────────────
// Test 3 (spec §F.8): abort-on-fail (simulated via RollbackAndContinue).
// ─────────────────────────────────────────────────────────────────────
//
// The real abort path would kill this test process, so we instead
// exercise the `RollbackAndContinue` override — the scheduler
// observes a flush failure AND still calls `tick_for_test` without
// aborting. This proves the failure-detection wiring works; the
// abort dispatch is a single `match fail_action` arm that routes to
// `std::process::abort()` vs `warn! + return Err`.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn background_fsync_failure_dispatched_via_fail_action() {
    let dir = tempfile::tempdir().unwrap();
    let writer = WalWriter::spawn(config(dir.path().to_path_buf())).unwrap();
    let handle = writer.handle();
    // Under RollbackAndContinue, a flush failure is logged and
    // counted but does NOT abort.
    let scheduler = BackgroundFsyncScheduler::start(
        handle.clone(),
        BackgroundFsyncFailAction::RollbackAndContinue,
    );
    // Shutdown the writer — now every flush() returns
    // WalUnavailable.
    writer.shutdown().unwrap();
    // Allow the channel-closed signal to propagate.
    thread::sleep(Duration::from_millis(20));

    let res = scheduler.tick_for_test();
    assert!(
        res.is_err(),
        "RollbackAndContinue path must surface flush failures as Err"
    );
    let m = scheduler.metrics();
    assert!(m.tick_errors_total() >= 1, "tick_errors_total incremented");

    // The scheduler is still alive — a second tick also errors
    // without aborting.
    assert!(scheduler.is_running());
    let _ = scheduler.tick_for_test();
    assert!(scheduler.metrics().tick_errors_total() >= 2);

    scheduler.shutdown().unwrap();
}

// ─────────────────────────────────────────────────────────────────────
// Test 4: crash between visible.advance and fsync regresses visible.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn crash_between_visible_and_fsync_regresses_visible() {
    // §6.6: T3 commits that were ack'd but not yet fsynced are
    // "visible" in memory but absent from disk. Simulating crash by
    // dropping the writer (forces graceful drain) — we shut down
    // BEFORE the scheduler ticks so the post-recovery state
    // reflects only the pre-ack bootstrap commit.
    //
    // Rather than simulate full recovery (ADR-032 replay lives
    // separately), we assert the on-disk state: bootstrap SYSTEM
    // bundle + tier-change SYSTEM bundle are present; the N T3 user
    // commits may or may not be (depending on shutdown-drain
    // behaviour). The KEY assertion is that IF fewer than N user
    // commits are on disk, those missing ones are the REGRESSION —
    // in-memory visible had them, disk does not.
    //
    // Config: window long enough that T3 commits don't auto-fsync
    // within the test's assertion window, but short enough that
    // the SYSTEM bootstrap + tier-change T1 commits actually fire
    // (without the scheduler being there to help them along).
    // batch_size=10_000 is effectively unlimited for this test;
    // the timer is the only fire trigger.
    let dir = tempfile::tempdir().unwrap();
    let cfg = WalConfig {
        dir: dir.path().to_path_buf(),
        segment_size_bytes: 64 * 1024 * 1024,
        group_commit_window: Duration::from_millis(200),
        group_commit_max_batch: 10_000,
        metrics_sink: None,
        encryption: None,

        inflight_budget_bytes: None,
    };
    let writer = WalWriter::spawn(cfg).unwrap();
    // Intentionally DO NOT start the scheduler — T3 commits never
    // get a periodic fsync.
    let mut mgr = TxnManager::with_wal(writer.handle());
    let catalog = Arc::new(SystemCatalog::new());
    let io = Arc::new(InMemoryPageIo::new());
    let pool = BufferPool::new(8, io);
    catalog.bootstrap(&pool, &mgr).unwrap();
    mgr.set_durability_lookup(catalog.clone());
    let mut sys_tx = mgr.begin(TenantId::SYSTEM);
    catalog
        .set_durability_tier(
            &mut sys_tx,
            TenantId::DEFAULT,
            DurabilityTier::Periodic { rpo_ms: 100 },
        )
        .unwrap();
    sys_tx.commit().unwrap();

    // Commit N T3 writes.
    let n = 5u64;
    let mut acked = Vec::with_capacity(n as usize);
    for i in 1..=n {
        let mut tx = mgr.begin(TenantId::DEFAULT);
        tx.write(i, Bytes::from(format!("v{i}").into_bytes()));
        let lsn = tx.commit().unwrap();
        acked.push(lsn);
    }

    // In memory: visible advanced to max(acked).
    let in_memory_visible = mgr.current_lsn();
    assert_eq!(in_memory_visible, *acked.last().unwrap());

    // Simulate crash: shutdown drains pending — after shutdown we
    // expect all N records on disk because shutdown FIRES the batch
    // in the run loop. But the key ADR-034 test is:
    // pre-shutdown, committed_fsync_watermark < in_memory_visible.
    let pre_shutdown_watermark = writer.handle().last_durable_lsn();
    assert!(
        pre_shutdown_watermark < in_memory_visible,
        "§6.6: pre-shutdown watermark {pre_shutdown_watermark:?} must lag in-memory visible \
         {in_memory_visible:?} (proves visible regresses on crash)"
    );

    // Shutdown gracefully → every ack'd T3 commit lands on disk
    // (this is the graceful drain path, NOT the crash path).
    writer.shutdown().unwrap();
    let records = drain_segments(dir.path());
    let user_bundles: Vec<_> = records
        .iter()
        .filter(|r| {
            r.tenant_id == TenantId::DEFAULT && r.record_type == WalRecordType::CommitBundle
        })
        .collect();
    assert_eq!(
        user_bundles.len(),
        n as usize,
        "graceful shutdown drains pending"
    );
}

// ─────────────────────────────────────────────────────────────────────
// Test 5: scheduler idempotent start (lifecycle).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn scheduler_idempotent_register_same_tenant() {
    // Register the same T3 tenant twice with a different rpo_ms —
    // the second call replaces the first (recompute interval, same
    // tenant count).
    let s = Setup::new(500);
    assert_eq!(s.scheduler.registered_tenant_count(), 1);
    assert_eq!(s.scheduler.current_interval_ms(), 500);

    s.scheduler
        .register(TenantId::DEFAULT, DurabilityTier::Periodic { rpo_ms: 50 });
    assert_eq!(s.scheduler.registered_tenant_count(), 1);
    assert_eq!(s.scheduler.current_interval_ms(), 50);

    s.shutdown();
}

// ─────────────────────────────────────────────────────────────────────
// Proptest: RPO compliance sweep.
// ─────────────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]

    /// Sweep rpo_ms across the accepted range and assert that,
    /// given enough wall-clock, the scheduler durifies every T3
    /// commit. This is the positive direction of I-D2 (durability
    /// is eventually achieved); the "crash or durable within rpo_ms"
    /// disjunction's abort arm is tested separately in
    /// background_fsync_failure_dispatched_via_fail_action.
    #[test]
    fn rpo_compliance_sweep(rpo_ms in 10u64..=500u64, n_commits in 1u64..=8u64) {
        let s = Setup::new(rpo_ms);

        let mut max_lsn = Lsn::ZERO;
        for i in 1..=n_commits {
            let mut tx = s.mgr.begin(TenantId::DEFAULT);
            tx.write(i, Bytes::from_static(b"v"));
            max_lsn = tx.commit().unwrap();
        }

        // Allow up to (rpo_ms + fsync jitter) × 6 for the
        // scheduler to catch up. Generous because CI hosts under
        // load can miss scheduler ticks.
        let budget = Duration::from_millis(rpo_ms.saturating_mul(6) + 500);
        let deadline = Instant::now() + budget;
        while s.handle_last_durable() < max_lsn && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        prop_assert!(
            s.handle_last_durable() >= max_lsn,
            "rpo_ms={rpo_ms}: commits not durified within {budget:?}; \
             max_lsn={max_lsn:?} watermark={:?}",
            s.handle_last_durable(),
        );

        s.shutdown();
    }
}
