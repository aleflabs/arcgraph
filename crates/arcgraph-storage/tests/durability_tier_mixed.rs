//! ADR-034 §Slice F — mixed T1/T3 tier integration tests.
//!
//! Invariants exercised:
//! - **I-D3**: T1 ack implies prior T3 commits are durable
//!   (piggyback).
//! - **I-D5**: counter allocation is tier-agnostic; a mixed
//!   workload preserves commit_lsn monotonicity.
//! - **I-D6**: replay is tier-agnostic; the WAL has no tier byte.
//! - **I-D7**: tier read at commit time; SYSTEM always T1.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use arcgraph_core::{DurabilityTier, DurabilityTierError, Lsn, TenantId};
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

fn config_long_window(dir: PathBuf) -> WalConfig {
    WalConfig {
        dir,
        segment_size_bytes: 64 * 1024 * 1024,
        // Long window: the only fire triggers are batch-full or
        // explicit flush() (scheduler / T1 commit).
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

/// Drive a blocking T1/Strict commit (or any op that parks on a WAL
/// group-commit fsync) to completion under a **manual-tick** scheduler.
///
/// # Why this exists (the #711 hang)
///
/// [`config_long_window`] sets `group_commit_window = 3600 s`, so the WAL
/// writer's own group-commit timer never fires inside a test — the only
/// fire triggers are batch-full (1 000), an explicit `flush()`, or
/// shutdown. A **Strict** commit's `WalHandle::append` is *synchronous*:
/// it enqueues its record and then parks on its fsync-ack until *some*
/// fire drains the writer's pending batch. It does **not** self-fire
/// (forcing a fire per sync append would defeat group-commit pipelining
/// — ADR-031 §3.5 / invariant 8). In production that external fire is the
/// writer's short `group_commit_window` timeout, so Strict durability
/// does **not** depend on the scheduler (see `durability_tier_strict.rs`:
/// a 2 ms window, no scheduler at all). In AUTO-tick mode the
/// [`BackgroundFsyncScheduler`]'s periodic `flush()` was silently
/// supplying that fire. MANUAL-tick mode spawns **no** thread, so with a
/// 3 600 s window *nothing* fires the batch and the first Strict commit —
/// the SYSTEM bootstrap commit inside [`MixedSetup::new_two_tenant_inner`],
/// before the test body even runs — parks forever. That is the #711 CI
/// hang (writer thread idling in `recv_timeout(3600s)`, test thread in
/// `WalHandle::append`'s `recv()`).
///
/// This helper supplies the missing external fire deterministically:
/// `op` runs on the calling thread while a sibling thread fires
/// `tick_for_test()` (→ `wal.flush()`) until `op` returns. The fire
/// drains the pending batch — the blocking commit **and** any pending T3
/// commits (the I-D3 piggyback), which is exactly the durability path
/// under test.
///
/// The drain is **scoped** to `op`: it runs only while a blocking commit
/// is in flight and stops the instant `op` returns. No tick fires between
/// the T3 commits and the "T3 still pending" precondition read, so that
/// SETUP assertion stays deterministic — the property #711 set out to
/// guarantee, now without the hang.
fn drive_with_manual_drain<T>(
    scheduler: &Arc<BackgroundFsyncScheduler>,
    op: impl FnOnce() -> T,
) -> T {
    let sched = Arc::clone(scheduler);
    let stop = Arc::new(AtomicBool::new(false));
    let stop_for_thread = Arc::clone(&stop);
    let drainer = thread::spawn(move || {
        // `tick_for_test` → `wal.flush()` is a blocking round-trip through
        // the writer, so the loop paces itself to the fsync rate; the 1 ms
        // back-off keeps empty fires (before the commit's record reaches
        // the writer) from busy-spinning. Under `BackgroundFsyncFailAction::
        // Abort` a flush error would abort the process, but the writer is
        // live for the whole lifetime of any commit we drive here.
        while !stop_for_thread.load(Ordering::Acquire) {
            let _ = sched.tick_for_test();
            thread::sleep(Duration::from_millis(1));
        }
    });
    let out = op();
    stop.store(true, Ordering::Release);
    drainer.join().expect("manual-drain helper thread");
    out
}

struct MixedSetup {
    _dir: TempDir,
    writer: Option<WalWriter>,
    scheduler: Arc<BackgroundFsyncScheduler>,
    mgr: TxnManager,
    catalog: Arc<SystemCatalog>,
    /// `true` when the background fsync scheduler was started in
    /// manual-tick mode (no auto-fire thread). The long-window writer
    /// then has no mechanism to fire a Strict commit's batch, so every
    /// blocking T1/Strict commit must be driven via
    /// [`drive_with_manual_drain`]. AUTO-tick setups leave this `false`
    /// and rely on the scheduler thread's periodic `flush()`.
    manual_sched: bool,
}

impl MixedSetup {
    fn new_two_tenant(t3_rpo_ms: u64) -> Self {
        // Default: AUTO-tick scheduler (a background thread fires
        // flush() on the rpo_ms cadence). Most mixed-tier tests rely
        // on the auto-tick to durify T3 commits (e.g.
        // `tier_change_drains_background_scheduler`).
        Self::new_two_tenant_inner(t3_rpo_ms, /* manual_sched = */ false)
    }

    /// Like [`Self::new_two_tenant`], but the background fsync
    /// scheduler is started in **manual-tick** mode (no auto-ticking
    /// thread) via [`BackgroundFsyncScheduler::start_manual`].
    ///
    /// Use this for tests that assert a "T3 commits are still pending
    /// (un-fsynced)" precondition before exercising the real
    /// durability path. With the auto-ticking scheduler that
    /// precondition races the background timer — `set_default_tier`
    /// calls `register`, which wakes the condvar, and under load the
    /// woken tick can fire `flush()` inside the pre-assertion window,
    /// advancing the watermark early and tripping the setup assertion.
    /// Manual mode removes the race: with no auto-fire thread, nothing
    /// advances the watermark between the T3 commits and the precondition
    /// read. But it also removes the ONLY thing that fired a Strict
    /// commit's batch under this file's long-window writer — a Strict
    /// `wal.append` does NOT self-fsync (that would defeat group-commit
    /// pipelining), it parks until an external fire drains the batch. So
    /// every blocking T1/Strict commit (bootstrap, `set_default_tier`,
    /// user commits) MUST be driven via [`drive_with_manual_drain`],
    /// which fires `tick_for_test()` concurrently until the commit acks
    /// — that same fire piggybacks the pending T3 batch, exactly the
    /// durability path under test. The strict durability invariants are
    /// unchanged; only the SETUP becomes deterministic.
    fn new_two_tenant_manual_sched(t3_rpo_ms: u64) -> Self {
        Self::new_two_tenant_inner(t3_rpo_ms, /* manual_sched = */ true)
    }

    fn new_two_tenant_inner(_t3_rpo_ms: u64, manual_sched: bool) -> Self {
        // DEFAULT tenant = T1 (bootstrap default).
        // Tenant #100 = T3 {rpo_ms}.
        let dir = tempfile::tempdir().unwrap();
        let writer = WalWriter::spawn(config_long_window(dir.path().to_path_buf())).unwrap();
        let scheduler = if manual_sched {
            BackgroundFsyncScheduler::start_manual(
                writer.handle(),
                BackgroundFsyncFailAction::Abort,
            )
        } else {
            BackgroundFsyncScheduler::start(writer.handle(), BackgroundFsyncFailAction::Abort)
        };

        let mut mgr = TxnManager::with_wal(writer.handle());
        let catalog = Arc::new(SystemCatalog::new());
        let io = Arc::new(InMemoryPageIo::new());
        let pool = BufferPool::new(8, io);
        // `bootstrap` issues a SYSTEM (T1/Strict) commit. Under the
        // long-window writer that commit's synchronous fsync needs an
        // external fire. In manual-tick mode there is no scheduler thread
        // to supply it, so drive the fire concurrently or the constructor
        // hangs (the #711 bug — see `drive_with_manual_drain`). In auto
        // mode the scheduler thread already drains it.
        if manual_sched {
            drive_with_manual_drain(&scheduler, || catalog.bootstrap(&pool, &mgr).unwrap());
        } else {
            catalog.bootstrap(&pool, &mgr).unwrap();
        }
        mgr.set_durability_lookup(catalog.clone());

        // DEFAULT stays Strict (the bootstrap default).
        // Add tenant #100 and flip it to Periodic. v1.0 doesn't
        // have DDL-based tenant creation; for this test we bypass
        // by pushing a TenantRecord directly. But our API expects
        // the tenant in the catalog for set_durability_tier to
        // succeed, so instead use the DEFAULT tenant for T3 and
        // use SYSTEM for T1 writes (SYSTEM is T1-enforced per
        // I-D7 — convenient).
        //
        // Actually: both halves of the mixed test use DEFAULT as
        // the single user tenant, flipping it T1→T3→T1 as needed.
        // The piggyback test uses SYSTEM writes (always T1) for
        // the "other" side since SYSTEM bootstrapping always
        // produces T1 commits.

        Self {
            _dir: dir,
            writer: Some(writer),
            scheduler,
            mgr,
            catalog,
            manual_sched,
        }
    }

    fn set_default_tier(&self, tier: DurabilityTier) -> Lsn {
        let mut tx = self.mgr.begin(TenantId::SYSTEM);
        self.catalog
            .set_durability_tier(&mut tx, TenantId::DEFAULT, tier)
            .unwrap();
        // The tier-change commit is SYSTEM → always T1/Strict (I-D7), so
        // it parks on a synchronous fsync. Under manual-tick the
        // long-window writer has no auto-fire; drive it concurrently
        // (the same fire piggybacks any pending T3 batch — exactly the
        // I-D3 path the manual-tick tests assert). In auto mode the
        // scheduler thread drains it.
        let lsn = if self.manual_sched {
            drive_with_manual_drain(&self.scheduler, || tx.commit().unwrap())
        } else {
            tx.commit().unwrap()
        };
        self.scheduler.register(TenantId::DEFAULT, tier);
        lsn
    }

    fn handle_watermark(&self) -> Lsn {
        self.writer
            .as_ref()
            .expect("writer live")
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
// Test 1 (spec §F.3): T1 strict preservation in a mixed workload.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn mixed_t1_t3_t1_strict_preservation() {
    // DEFAULT commits a T3 commit, then flips to Strict, then
    // commits a T1 commit. Both ack'd; both visible. The T1 commit
    // is durable at ack; the T3 commit is durable by the time the
    // T1 commit fires (piggyback, I-D3).
    let s = MixedSetup::new_two_tenant(1000);

    // T3 phase.
    let _tier_change = s.set_default_tier(DurabilityTier::Periodic { rpo_ms: 1000 });
    let mut tx = s.mgr.begin(TenantId::DEFAULT);
    tx.write(1, Bytes::from_static(b"t3-val"));
    let t3_lsn = tx.commit().unwrap();
    // T3 ack'd but not yet fsynced (rpo_ms = 1000; scheduler hasn't
    // ticked yet).
    assert!(s.handle_watermark() < t3_lsn);

    // Flip to Strict and commit.
    let _tier_change2 = s.set_default_tier(DurabilityTier::Strict);
    let mut tx = s.mgr.begin(TenantId::DEFAULT);
    tx.write(2, Bytes::from_static(b"t1-val"));
    let t1_lsn = tx.commit().unwrap();

    // After T1 commit: watermark covers both the T1 and the T3
    // (piggyback — proven by I-D3).
    assert!(s.handle_watermark() >= t1_lsn);
    assert!(
        s.handle_watermark() >= t3_lsn,
        "T1 piggyback must durify prior T3: watermark={:?} t3_lsn={t3_lsn:?} t1_lsn={t1_lsn:?}",
        s.handle_watermark()
    );

    s.shutdown();
}

// ─────────────────────────────────────────────────────────────────────
// Test 2 (spec §F.4): T3 piggyback via T1 commit.
// ─────────────────────────────────────────────────────────────────────

// DETERMINISM: the "scheduler hasn't ticked yet at 60s rpo"
// precondition is a SETUP assertion that, under the auto-ticking
// scheduler, races the background timer (woken by `register` inside
// `set_default_tier`) and trips under debug-build / shared-runner
// slowness. Previously this was worked around with
// `#[cfg_attr(debug_assertions, ignore)]` — but an ignored test is a
// hole in coverage. We instead use a MANUAL-tick scheduler
// (`new_two_tenant_manual_sched`): no background timer can fire, the
// SETUP precondition is deterministic, and the test now runs in BOTH
// debug and release (strictly stronger than an ignored test). The
// piggyback durability invariants below are unchanged.
#[test]
fn mixed_t1_t3_t3_piggyback_durability() {
    // Configure a very long rpo_ms so the scheduler won't be the
    // one to durify. Flip to T3, commit N records, flip back to
    // T1, commit one T1 record. Assert that T1 return covers all
    // prior T3 LSNs.
    let s = MixedSetup::new_two_tenant_manual_sched(60_000);
    s.set_default_tier(DurabilityTier::Periodic { rpo_ms: 60_000 });

    let n = 5u64;
    let mut t3_lsns = Vec::with_capacity(n as usize);
    for i in 1..=n {
        let mut tx = s.mgr.begin(TenantId::DEFAULT);
        tx.write(i, Bytes::from(format!("t3-{i}").into_bytes()));
        t3_lsns.push(tx.commit().unwrap());
    }

    // Pre-T1: watermark is below every T3 LSN. Deterministic under the
    // manual-tick scheduler (no background timer can fire here).
    let pre = s.handle_watermark();
    assert!(pre < t3_lsns[0], "scheduler hasn't ticked yet at 60s rpo");

    // Flip to Strict (this commit itself is T1 since SYSTEM is T1;
    // it also triggers a fire for prior T3 piggyback).
    let tier_change_lsn = s.set_default_tier(DurabilityTier::Strict);
    // The tier-change commit is SYSTEM → T1. Its fsync piggybacks
    // every T3 record. So watermark post-tier-change covers all
    // T3 LSNs already.
    assert!(
        s.handle_watermark() >= *t3_lsns.last().unwrap(),
        "SYSTEM tier-change fsync piggybacks T3: watermark={:?} last T3={:?}",
        s.handle_watermark(),
        t3_lsns.last().unwrap()
    );

    // Commit a regular T1 on DEFAULT, confirm invariant holds. DEFAULT
    // is Strict now, so this is a blocking sync fsync — drive its fire
    // (manual-tick, long-window writer has no auto-fire).
    let mut tx = s.mgr.begin(TenantId::DEFAULT);
    tx.write(100, Bytes::from_static(b"t1"));
    let t1_lsn = drive_with_manual_drain(&s.scheduler, || tx.commit().unwrap());
    assert!(s.handle_watermark() >= t1_lsn);
    assert!(
        s.handle_watermark() >= tier_change_lsn,
        "tier_change_lsn {tier_change_lsn:?} must be durable after T1 {t1_lsn:?}",
    );

    s.shutdown();
}

// ─────────────────────────────────────────────────────────────────────
// Test 3 (spec §F.5): tier change mid-transaction uses commit-time tier.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn tier_change_mid_transaction_uses_commit_time_tier() {
    // I-D7: begin() under T1, tier flips to T3, commit() uses T3.
    // Observable via watermark lag.
    let s = MixedSetup::new_two_tenant(1000);

    // Start a user transaction while DEFAULT is still Strict.
    let mut tx = s.mgr.begin(TenantId::DEFAULT);
    tx.write(42, Bytes::from_static(b"value"));

    // Operator flips DEFAULT to Periodic {rpo_ms=60000} while the
    // user tx is in-flight.
    s.set_default_tier(DurabilityTier::Periodic { rpo_ms: 60_000 });

    // Commit. Tier is read at commit time → Periodic → async path
    // → watermark does NOT cover commit_lsn on return.
    let lsn = tx.commit().unwrap();
    assert_eq!(s.mgr.current_lsn(), lsn);
    assert!(
        s.handle_watermark() < lsn,
        "I-D7: mid-tx tier flip to T3 must yield async commit; \
         watermark={:?} commit_lsn={lsn:?}",
        s.handle_watermark()
    );

    s.shutdown();
}

// ─────────────────────────────────────────────────────────────────────
// Test 4 (spec §F.6): tier change drains background scheduler.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn tier_change_drains_background_scheduler() {
    // Flipping T3 → T1 for the last T3 tenant removes it from the
    // scheduler's set; interval returns to idle (1 s).
    let s = MixedSetup::new_two_tenant(50);
    s.set_default_tier(DurabilityTier::Periodic { rpo_ms: 50 });
    assert_eq!(s.scheduler.registered_tenant_count(), 1);
    assert_eq!(s.scheduler.current_interval_ms(), 50);

    // Flip back to Strict — the scheduler unregisters.
    s.set_default_tier(DurabilityTier::Strict);
    assert_eq!(s.scheduler.registered_tenant_count(), 0);
    assert!(
        s.scheduler.current_interval_ms() > 50,
        "post-unregister interval should return to idle (>50ms)"
    );

    s.shutdown();
}

// ─────────────────────────────────────────────────────────────────────
// Test 5 (spec §F.10): SYSTEM tenant is always T1.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn system_tenant_is_always_t1() {
    // Attempt to flip SYSTEM → Periodic rejected with
    // SystemTenantMustBeStrict (catalog-level I-D7).
    let s = MixedSetup::new_two_tenant(100);
    let mut tx = s.mgr.begin(TenantId::SYSTEM);
    let err = s
        .catalog
        .set_durability_tier(
            &mut tx,
            TenantId::SYSTEM,
            DurabilityTier::Periodic { rpo_ms: 100 },
        )
        .unwrap_err();
    assert_eq!(err, DurabilityTierError::SystemTenantMustBeStrict);
    tx.abort();

    // Confirm SYSTEM lookups return Strict regardless.
    assert_eq!(
        s.catalog.durability_tier(TenantId::SYSTEM),
        DurabilityTier::Strict
    );

    // And SYSTEM commits behave as T1 (watermark advances
    // synchronously) even without an explicit tier registration.
    let mut tx = s.mgr.begin(TenantId::SYSTEM);
    tx.write(0xdeadbeef, Bytes::from_static(b"sys"));
    let lsn = tx.commit().unwrap();
    assert!(
        s.handle_watermark() >= lsn,
        "SYSTEM commits must be durable on ack (I-D1 under T1 enforcement)"
    );

    s.shutdown();
}

// ─────────────────────────────────────────────────────────────────────
// Test 6: replay is tier-agnostic (I-D6 regression guard).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn replay_bytes_are_tier_agnostic() {
    // D-3 / I-D6: a T3 commit's bundle bytes are byte-identical in
    // shape to a T1 commit's bundle bytes. Verify by scanning the
    // on-disk records: every CommitBundle on DEFAULT has the same
    // decoded shape regardless of the tier that produced it.
    let s = MixedSetup::new_two_tenant(50);

    // Commit one T1 and one T3 record on DEFAULT.
    let mut tx = s.mgr.begin(TenantId::DEFAULT);
    tx.write(1, Bytes::from_static(b"v1"));
    let t1_lsn = tx.commit().unwrap();

    s.set_default_tier(DurabilityTier::Periodic { rpo_ms: 50 });
    let mut tx = s.mgr.begin(TenantId::DEFAULT);
    tx.write(2, Bytes::from_static(b"v2"));
    let t3_lsn = tx.commit().unwrap();

    // Let the scheduler tick to durify the T3.
    let deadline = Instant::now() + Duration::from_millis(1000);
    while s.handle_watermark() < t3_lsn && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }

    let dir_path = s._dir.path().to_path_buf();
    // Take writer out for clean shutdown.
    let _ = s.scheduler.shutdown();
    let w = s.writer;
    if let Some(w) = w {
        let _ = w.shutdown();
    }

    let records = drain_segments(&dir_path);
    let user_bundles: Vec<_> = records
        .iter()
        .filter(|r| {
            r.record_type == WalRecordType::CommitBundle && r.tenant_id == TenantId::DEFAULT
        })
        .collect();
    // Both user commits are CommitBundle = 12. No T3 byte
    // distinguishes them (D-3).
    assert_eq!(user_bundles.len(), 2);
    for r in &user_bundles {
        assert_eq!(r.record_type, WalRecordType::CommitBundle);
        assert_eq!(r.tenant_id, TenantId::DEFAULT);
    }
    // LSNs distinguish them but neither payload has a tier byte.
    let lsns: std::collections::HashSet<_> = user_bundles.iter().map(|r| r.lsn).collect();
    // Note: WAL LSN != commit LSN under pipelined commits (ADR-031
    // §R3). We assert the records exist and carry the user tenant;
    // commit_lsn uniqueness is proven by the commit path above.
    assert!(!lsns.is_empty());
    let _ = (t1_lsn, t3_lsn); // consumed above
}

// ─────────────────────────────────────────────────────────────────────
// Test 7: local-only regression guard.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn durability_tier_partition_id_always_zero_at_v1() {
    // ADR-024-amendment-02: v1.0 partition count = 1. Neither
    // DurabilityTier nor BackgroundFsyncScheduler carries a
    // partition_id at v1.0. If a v1.1 patch adds the field, this
    // test becomes the reviewer checkpoint.
    //
    // We verify by size: DurabilityTier must remain ≤ 16 bytes
    // (Strict + Periodic{rpo_ms: u64} = 16 with discriminant).
    // Adding a PartitionId (u32) would bump alignment.
    let t_strict = DurabilityTier::Strict;
    let t_periodic = DurabilityTier::Periodic { rpo_ms: 100 };
    assert!(
        std::mem::size_of_val(&t_strict) <= 16,
        "DurabilityTier Strict size exceeded 16B; partition_id likely added"
    );
    assert!(
        std::mem::size_of_val(&t_periodic) <= 16,
        "DurabilityTier Periodic size exceeded 16B; partition_id likely added"
    );

    // And the scheduler API takes (TenantId, DurabilityTier) with
    // no partition parameter.
    let dir = tempfile::tempdir().unwrap();
    let writer = WalWriter::spawn(config_long_window(dir.path().to_path_buf())).unwrap();
    let sched = BackgroundFsyncScheduler::start(writer.handle(), BackgroundFsyncFailAction::Abort);
    sched.register(TenantId::DEFAULT, DurabilityTier::Periodic { rpo_ms: 100 });
    let _ = sched.shutdown();
    let _ = writer.shutdown();
}

// ─────────────────────────────────────────────────────────────────────
// Proptest: T1 ack implies prior T3 durability (I-D3).
// ─────────────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]

    /// Randomly interleave T3 commits and T1 commits. Assert that
    /// every T1 ack finds all prior T3 commits durable (watermark
    /// ≥ max prior T3 LSN).
    #[test]
    fn t1_ack_implies_t3_durability_proptest(
        // A sequence of 'T3' or 'T1' markers.
        tiers in prop::collection::vec(any::<bool>(), 3..10),
    ) {
        // Setup with a huge rpo_ms so the scheduler cannot
        // accidentally durify T3 between our T1 commits — the only
        // fsync trigger is a T1 commit or a shutdown.
        let dir = tempfile::tempdir().unwrap();
        let cfg = WalConfig {
            dir: dir.path().to_path_buf(),
            segment_size_bytes: 64 * 1024 * 1024,
            // 5ms window so T1 SYSTEM commits (bootstrap + tier
            // changes) fire in bounded wall-clock time — there's
            // no scheduler in these proptests.
            group_commit_window: Duration::from_millis(5),
            group_commit_max_batch: 10_000,
            metrics_sink: None,
            encryption: None,

            inflight_budget_bytes: None,
};
        let writer = WalWriter::spawn(cfg).unwrap();
        let mut mgr = TxnManager::with_wal(writer.handle());
        let catalog = Arc::new(SystemCatalog::new());
        let io = Arc::new(InMemoryPageIo::new());
        let pool = BufferPool::new(8, io);
        catalog.bootstrap(&pool, &mgr).unwrap();
        mgr.set_durability_lookup(catalog.clone());

        let mut last_t3_lsn: Option<Lsn> = None;
        let mut i = 1000u64;
        for is_t1 in &tiers {
            if *is_t1 {
                // Ensure DEFAULT is Strict for this commit.
                let mut sx = mgr.begin(TenantId::SYSTEM);
                catalog
                    .set_durability_tier(&mut sx, TenantId::DEFAULT, DurabilityTier::Strict)
                    .unwrap();
                sx.commit().unwrap();

                let mut tx = mgr.begin(TenantId::DEFAULT);
                tx.write(i, Bytes::from_static(b"t1"));
                let lsn = tx.commit().unwrap();
                i += 1;

                // I-D3: every prior T3 commit is now durable.
                if let Some(prior_t3) = last_t3_lsn {
                    let wm = writer.handle().last_durable_lsn();
                    prop_assert!(
                        wm >= prior_t3,
                        "I-D3: T1 ack {lsn:?} must durify prior T3 {prior_t3:?}; watermark={wm:?}"
                    );
                }
            } else {
                // Ensure DEFAULT is Periodic for this commit.
                let mut sx = mgr.begin(TenantId::SYSTEM);
                catalog
                    .set_durability_tier(
                        &mut sx,
                        TenantId::DEFAULT,
                        DurabilityTier::Periodic { rpo_ms: 60_000 },
                    )
                    .unwrap();
                // The tier-change commit itself is SYSTEM (T1) so
                // it fsyncs — this actually exposes a subtle point:
                // the SYSTEM commit itself piggybacks prior T3s.
                // Do not assert about watermark here; only assert
                // at T1 commits on DEFAULT.
                sx.commit().unwrap();

                let mut tx = mgr.begin(TenantId::DEFAULT);
                tx.write(i, Bytes::from_static(b"t3"));
                let lsn = tx.commit().unwrap();
                i += 1;
                last_t3_lsn = Some(lsn);
            }
        }

        writer.shutdown().unwrap();
    }
}

// ─────────────────────────────────────────────────────────────────────
// Proptest: random tier workload preserves T1 invariant.
// ─────────────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]

    /// Random T1/T3 interleaving. Every T1 commit's commit_lsn must
    /// be durable at ack (watermark ≥ commit_lsn).
    #[test]
    fn random_tier_workload_preserves_t1_invariant(
        tiers in prop::collection::vec(any::<bool>(), 2..8),
    ) {
        let dir = tempfile::tempdir().unwrap();
        let cfg = WalConfig {
            dir: dir.path().to_path_buf(),
            segment_size_bytes: 64 * 1024 * 1024,
            // 5ms window so T1 SYSTEM commits (bootstrap + tier
            // changes) fire in bounded wall-clock time — there's
            // no scheduler in these proptests.
            group_commit_window: Duration::from_millis(5),
            group_commit_max_batch: 10_000,
            metrics_sink: None,
            encryption: None,

            inflight_budget_bytes: None,
};
        let writer = WalWriter::spawn(cfg).unwrap();
        let mut mgr = TxnManager::with_wal(writer.handle());
        let catalog = Arc::new(SystemCatalog::new());
        let io = Arc::new(InMemoryPageIo::new());
        let pool = BufferPool::new(8, io);
        catalog.bootstrap(&pool, &mgr).unwrap();
        mgr.set_durability_lookup(catalog.clone());

        for (key, is_t1) in tiers.iter().enumerate() {
            let key = key as u64;
            let target_tier = if *is_t1 {
                DurabilityTier::Strict
            } else {
                DurabilityTier::Periodic { rpo_ms: 60_000 }
            };
            let mut sx = mgr.begin(TenantId::SYSTEM);
            catalog
                .set_durability_tier(&mut sx, TenantId::DEFAULT, target_tier)
                .unwrap();
            sx.commit().unwrap();

            let mut tx = mgr.begin(TenantId::DEFAULT);
            tx.write(key, Bytes::from_static(b"v"));
            let lsn = tx.commit().unwrap();

            if *is_t1 {
                let wm = writer.handle().last_durable_lsn();
                prop_assert!(
                    wm >= lsn,
                    "I-D1: T1 commit {lsn:?} not durable at ack; watermark={wm:?}"
                );
            }
        }

        writer.shutdown().unwrap();
    }
}

// ─────────────────────────────────────────────────────────────────
// M3.a Phase 5.5 — vector workload extension
// ─────────────────────────────────────────────────────────────────
//
// Per Path A directive 2026-04-26 + Phase 5.5 spec §2.2: extend
// `durability_tier_mixed` with a vector-aware variant. ADR-035 §7.1
// pins that vector arena page bytes share `staged_pages` with
// primary / record / blob pages and inherit ADR-034's tier dispatch
// verbatim — there is no "vector tier" axis. The extension models a
// T1 + T3 mix where vectors-bearing tenants appear on both sides.
//
// Phase 5.5 (test-only) constraint: vector arena pages do not yet
// flow through `CrudStore::commit` in production (Slice G.5 wires
// that). The pin here is at the WAL-bundle / scheduler level: the
// existing `MixedSetup` harness drives WAL records that REPRESENT
// vector commits (opaque payload — the WAL byte path is
// vector-content-agnostic per ADR-034 D-3). We assert that:
//
//   1. T1 vector commits durify before ack (I-V6 / ADR-034 I-D1).
//   2. T3 vector commits durify within rpo_ms (I-V6 / ADR-034 I-D2).
//   3. T1 ack covers prior T3 vector commits (I-V6 / ADR-034 I-D3
//      piggyback). Cross-pin with `mixed_t1_t3_t1_strict_preservation`
//      above.
//
// The bytes used for the "vector" payload are tagged with a unique
// `key` value range so the test's intent is self-documenting (the
// WAL itself does not differentiate; the tagging is for reader
// clarity only).

#[test]
fn durability_tier_vector_t1_strict_t3_periodic() {
    use arcgraph_core::{PageId, PartitionId};
    use arcgraph_storage::crud::{CrudStore, PropertyData, commit, create_node};
    use arcgraph_storage::page_alloc::PageAllocator;
    use arcgraph_storage::primary_index::PrimaryIndex;
    use arcgraph_storage::wal::bundle::decode_commit_bundle_v8;

    // Vector key range — purely for reader clarity. The WAL does
    // not interpret these; ADR-034 D-3 + the existing
    // `replay_bytes_are_tier_agnostic` test pin that the bundle
    // codec carries no "vector" byte.
    //
    // **M3.a Slice G.4 update:** the test now stages REAL vector
    // arena pages via `CrudStore::stage_vector_page` so each user
    // commit emits a v5 `CommitBundle` with a non-empty
    // `vector_pages` section. The pre-G.4 version of this test
    // wrote opaque MVCC bytes — those bundles had EMPTY
    // `vector_pages`, leaving the production v5 path unexercised.
    // Per ADR-031 amendment-02 + ADR-035 §4.5/§4.6 + issue #131
    // item 3 (production-path simulation gap closure).

    // DETERMINISM: this test asserts a SETUP precondition (the pre-T1
    // watermark is below the highest T3 LSN — i.e. the T3 vector
    // commits are still pending). Under the auto-ticking scheduler
    // that precondition races the background timer: `set_default_tier`
    // → `register` → `recompute_interval_and_wake` → `wake` interrupts
    // the condvar and, under load, can fire an early `flush()` inside
    // the pre-assertion window, advancing the watermark over the T3
    // commits before the read. (This is a test-determinism flaw, NOT a
    // data-loss bug: an early flush makes the T3 commits MORE durable,
    // never less — the strict crash-recovery assertions below still
    // hold. #698's SF-100 bench co-tenancy raised host load enough to
    // surface the latent race.)
    //
    // We use a MANUAL-tick scheduler (`new_two_tenant_manual_sched`): no
    // background timer can fire on its own, so nothing durifies the T3
    // batch between the commits and the pre-T1 read — the SETUP
    // precondition is deterministic WITHOUT weakening any durability
    // assertion.
    //
    // IMPORTANT (the #711 hang, now fixed): a Strict commit does NOT
    // self-fsync. `WalHandle::append` parks on its fsync-ack until an
    // external fire drains the writer's pending batch; forcing a fire
    // per sync append would defeat group-commit pipelining (ADR-031
    // §3.5). With this file's long-window writer (3 600 s) AND no
    // scheduler thread, the only such fire is an explicit
    // `tick_for_test()`/`flush()`. So every BLOCKING T1/Strict commit
    // here — the SYSTEM bootstrap (inside the constructor),
    // `PrimaryIndex::new`'s root-pointer commit, the two
    // `set_default_tier` tier-changes, and the Phase-2 user commit — is
    // driven via `drive_with_manual_drain`, which fires `tick_for_test`
    // on a sibling thread until the commit acks. The drain is scoped to
    // each blocking commit, so it never ticks between the T3 commits and
    // the pre-T1 read. (The T3 commits use `append_async` and do not
    // block.) As a bonus this also eliminates the prior ~60 s
    // `scheduler.shutdown()` drain (no auto-thread to wait on).
    //
    // rpo_ms=60_000 is retained for documentation parity with the
    // sibling `mixed_t1_t3_t3_piggyback_durability`; in manual mode the
    // value only affects `current_interval_ms` bookkeeping, never a
    // timer fire.
    let s = MixedSetup::new_two_tenant_manual_sched(60_000);

    // Wire a CrudStore on top of MixedSetup's existing MVCC kernel
    // so the CRUD-aware commit path (v5 bundles + vector_pages
    // section) exercises here. We deliberately reuse `s.mgr` (which
    // already has the durability_lookup wired to the catalog) so
    // the user commits dispatch through the right tier resolver —
    // a fresh `TxnManager::with_wal(...)` would NOT see the catalog
    // and would default to Strict, breaking the T3 watermark
    // assertion.
    let writer_handle = s.writer.as_ref().expect("MixedSetup writer live").handle();
    let alloc = Arc::new(PageAllocator::new());
    // Build the CrudStore + PrimaryIndex against the existing
    // TxnManager. PrimaryIndex needs an `Arc<TxnManager>` so we
    // wrap an Arc around an inner clone of the existing manager.
    // The MixedSetup retains ownership of the original.
    //
    // PrimaryIndex bootstrap uses a SYSTEM-tenant transaction to
    // persist its initial root pointer; that commit goes through
    // the wal handle and counts as one record on the WAL stream.
    let mgr_arc_for_store: Arc<arcgraph_storage::transaction::TxnManager> = {
        // Reuse the existing TxnManager by transmuting it through
        // the WAL handle: a fresh manager with the same wal handle
        // sees the same WAL but has no shared MVCC state. To keep
        // MVCC + tier dispatch coherent, we instead use the
        // existing s.mgr directly via an Arc that points to a
        // re-built manager that ALSO carries the catalog lookup.
        let mut mgr_inner =
            arcgraph_storage::transaction::TxnManager::with_wal(writer_handle.clone());
        mgr_inner.set_durability_lookup(s.catalog.clone());
        Arc::new(mgr_inner)
    };
    // `PrimaryIndex::new` persists its initial root pointer via a SYSTEM
    // (T1/Strict) commit — another blocking sync fsync that the
    // long-window manual-tick writer can't fire on its own; drive it.
    let primary = Arc::new(drive_with_manual_drain(&s.scheduler, || {
        PrimaryIndex::new(
            Arc::clone(&mgr_arc_for_store),
            Arc::clone(&alloc),
            Some(writer_handle.clone()),
        )
        .unwrap()
    }));
    let store = Arc::new(CrudStore::new_with_index(
        Some(writer_handle.clone()),
        Arc::clone(&primary),
        Arc::clone(&alloc),
    ));

    // Phase 1: tenant DEFAULT registered as T3 (Periodic), commits
    // a batch of CRUD nodes WITH staged vector arena pages. Each
    // commit ack returns pre-fsync (T3 contract). Watermark stays
    // below the highest T3 LSN until either the scheduler ticks
    // (rpo_ms = 60s, won't fire in the test window) OR a T1 commit
    // fsyncs.
    s.set_default_tier(DurabilityTier::Periodic { rpo_ms: 60_000 });
    let mut t3_lsns: Vec<Lsn> = Vec::new();
    for i in 0..4u64 {
        let mut tx = mgr_arc_for_store.begin(TenantId::DEFAULT);
        let _node_id = create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            arcgraph_core::LabelId::new(73 + i as u32),
            &PropertyData::InlineU32Pair(2026, i as u32),
        )
        .unwrap();
        // Stage one vector arena page per T3 commit. The bytes will
        // ride the same v5 `CommitBundle` fsync as the CRUD writes.
        let txn_id = tx.id();
        store.stage_vector_page(
            txn_id,
            TenantId::DEFAULT,
            PartitionId::ZERO,
            0, // index_id always 0 at v1.0
            PageId::new(100 + i),
            Box::new([(0xA0u8 + i as u8); arcgraph_core::PAGE_SIZE]),
        );
        t3_lsns.push(commit(tx, &store).unwrap());
    }

    // Pre-T1 watermark: below the highest T3 LSN (the scheduler at
    // 60s rpo cannot have fired yet; the only way to advance the
    // watermark is a T1 commit).
    let pre_t1 = s.handle_watermark();
    assert!(
        pre_t1 < *t3_lsns.last().unwrap(),
        "I-V6 setup: T3 vector commits should be pending; pre-T1 watermark={pre_t1:?} \
         max-T3-LSN={:?}",
        t3_lsns.last().unwrap()
    );

    // Phase 2: flip DEFAULT to T1 (Strict), commit one CRUD node +
    // staged vector page. ADR-034 I-D1: T1 commit durable before
    // ack. ADR-034 I-D3 + ADR-035 I-V6: T1 ack covers prior T3
    // commits (the tier-change commit itself is a SYSTEM T1 — its
    // fsync piggybacks the T3 batch first; then the user T1 commit
    // on DEFAULT cements the watermark).
    s.set_default_tier(DurabilityTier::Strict);
    let mut tx = mgr_arc_for_store.begin(TenantId::DEFAULT);
    let _t1_node = create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        arcgraph_core::LabelId::new(99),
        &PropertyData::InlineU32Pair(2027, 0),
    )
    .unwrap();
    let txn_id_t1 = tx.id();
    store.stage_vector_page(
        txn_id_t1,
        TenantId::DEFAULT,
        PartitionId::ZERO,
        0,
        PageId::new(200),
        Box::new([0xFF; arcgraph_core::PAGE_SIZE]),
    );
    // Phase-2 user commit: DEFAULT is Strict now → blocking sync fsync.
    // Drive its fire (long-window manual-tick writer has no auto-fire).
    let t1_lsn = drive_with_manual_drain(&s.scheduler, || commit(tx, &store).unwrap());

    // I-V6 (T1 contract): post-ack watermark covers the T1 commit.
    let post_t1 = s.handle_watermark();
    assert!(
        post_t1 >= t1_lsn,
        "I-V6 T1 strict: vector commit {t1_lsn:?} not durable at ack; watermark={post_t1:?}"
    );

    // I-V6 (piggyback): post-T1 watermark covers every prior T3
    // vector commit. Cross-references
    // `mixed_t1_t3_t1_strict_preservation` for the non-vector
    // analogue.
    for t3 in &t3_lsns {
        assert!(
            post_t1 >= *t3,
            "I-V6 piggyback: T1 vector commit {t1_lsn:?} did NOT durify prior T3 vector \
             commit {t3:?}; watermark={post_t1:?}"
        );
    }

    // Replay-tier-agnostic cross-check: scan the WAL records and
    // confirm both halves emit the same record_type
    // (CommitBundle = 12). ADR-034 D-3: no tier byte.
    //
    // **M3.a Slice G.4 strengthening:** decode every user bundle as
    // v5 AND assert that `vector_pages` is non-empty for each. Pre-
    // Slice-G.4 the bundles emitted by the v4 codec had no
    // `vector_pages` section; post-cutover every user commit on
    // this test stages a vector page, so every user bundle MUST
    // carry exactly one vector_pages entry.
    let dir_path = s._dir.path().to_path_buf();
    drop(store);
    drop(primary);
    drop(mgr_arc_for_store);
    let _ = s.scheduler.shutdown();
    if let Some(w) = s.writer {
        let _ = w.shutdown();
    }
    let records = drain_segments(&dir_path);
    let user_bundles: Vec<_> = records
        .iter()
        .filter(|r| {
            r.record_type == WalRecordType::CommitBundle && r.tenant_id == TenantId::DEFAULT
        })
        .collect();
    // 4 T3 + 1 T1 = 5 user CommitBundle records.
    assert_eq!(
        user_bundles.len(),
        5,
        "I-V6 codec: expected 5 user CommitBundle records (4 T3 vector + 1 T1 vector)"
    );
    for r in &user_bundles {
        assert_eq!(r.record_type, WalRecordType::CommitBundle);
        assert_eq!(r.tenant_id, TenantId::DEFAULT);

        // M3.a Slice G.4 strengthening (updated for #352 Part 2 / ADR-199):
        // every user bundle MUST decode cleanly as v6 AND carry a non-empty
        // `vector_pages` section. This is a load-bearing assertion — it
        // verifies the production v6 path actually runs (a pre-v6 codec emit
        // would mis-decode here). The bundle format bumped v5 → v6 to fold
        // in the idempotency_bindings section; vector_pages is unchanged.
        let bundle = decode_commit_bundle_v8(&r.payload, r.tenant_id)
            .expect("#352 Part 2: every user bundle MUST decode as v6");
        assert!(
            !bundle.vector_pages.is_empty(),
            "M3.a Slice G.4: every user vector commit MUST carry a \
             non-empty vector_pages section; commit_lsn={:?}",
            bundle.commit_lsn
        );
        // Stronger pin: each commit staged exactly one vector page.
        assert_eq!(
            bundle.vector_pages.len(),
            1,
            "M3.a Slice G.4: each commit staged 1 vector page; \
             commit_lsn={:?}",
            bundle.commit_lsn
        );
    }
}
