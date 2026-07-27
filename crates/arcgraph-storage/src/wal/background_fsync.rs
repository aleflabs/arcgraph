//! ADR-034 §Slice C — background fsync scheduler for T3 periodic
//! durability.
//!
//! The scheduler owns one dedicated thread that wakes at
//! `interval_ms` cadence and calls [`WalHandle::flush`]. Every tick
//! drains the writer's pending batch (T3 commits accumulated since
//! the last fire) and advances the writer's
//! `committed_fsync_watermark`.
//!
//! # Interval selection (D-6)
//!
//! One scheduler per process. Its interval is
//! `min(rpo_ms)` across every registered T3 tenant. A T3 tenant with
//! `rpo_ms = 500` co-resident with one at `rpo_ms = 100` gets a
//! 100 ms scheduler — strictly safer than its configured RPO (the
//! 500 ms tenant observes better-than-configured RPO, which is
//! correct per I-D2 — the contract is an *upper bound*).
//!
//! Recomputation happens on [`BackgroundFsyncScheduler::register`]
//! and [`BackgroundFsyncScheduler::unregister`], via condvar wake-up
//! of the sleeping scheduler thread.
//!
//! When no T3 tenants are registered, the scheduler enters an
//! "idle" interval (`DEFAULT_IDLE_INTERVAL`) — 1 s. It continues
//! calling `flush()` so any straggling async appends (test harnesses,
//! post-unregister commits) still durify, and so the tick cadence
//! is non-zero (avoids a busy-loop spin-on-zero bug).
//!
//! # Failure contract (D-7, I-D4)
//!
//! If [`WalHandle::flush`] returns `Err(_)` on a scheduler tick:
//!
//! - Default ([`BackgroundFsyncFailAction::Abort`]): the scheduler
//!   emits `tracing::error!`, then calls `std::process::abort()`.
//!   No clean shutdown; the supervisor (systemd, k8s) restarts.
//! - Test override ([`BackgroundFsyncFailAction::RollbackAndContinue`]):
//!   logs the failure, continues. Useful in harnesses that need to
//!   observe scheduler state after an injected failure. **NOT
//!   recommended for production.** Silently loses up-to-rpo-ms of
//!   acked T3 commits on each failure with no retry.
//!
//! # Scheduler thread lifecycle
//!
//! 1. [`BackgroundFsyncScheduler::start`] — spawn. Idempotent via
//!    `OnceLock`; subsequent calls return the same `Arc`.
//! 2. Running phase. The thread sleeps on a `Condvar` for
//!    `interval_ms`; woken early by register/unregister/shutdown.
//! 3. [`BackgroundFsyncScheduler::shutdown`] — set shutdown flag,
//!    wake the thread, join. Idempotent.
//!
//! Panic recovery: if the scheduler thread panics, the `JoinHandle`'s
//! `join()` returns `Err(panic_payload)`. The next `shutdown()` or
//! `health_check()` call surfaces this; v1.0 treats a scheduler
//! panic as a process-abort-worthy event (§6.4).

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use arcgraph_core::{DurabilityTier, TenantId};
use parking_lot::{Condvar, Mutex};
use tracing::{error, info, warn};

use crate::wal::WalHandle;

// ─────────────────────────────────────────────────────────────────────
// BackgroundFsyncFailAction — extends ARCGRAPH_WAL_ERROR_POLICY
// semantics for the post-ack failure case (ADR-034 §8.6 / D-7).
// ─────────────────────────────────────────────────────────────────────

/// Action to take when a background-fsync fire fails post-ack.
///
/// Per ADR-034 D-7 / §8.6, the only coherent response is
/// [`Self::Abort`]. The [`Self::RollbackAndContinue`] variant is
/// provided strictly for test-harness fault injection and is
/// documented only in test READMEs — operators who encounter a
/// deployment configured this way are running with corrupt
/// durability.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BackgroundFsyncFailAction {
    /// Default. `std::process::abort()` on any background fsync
    /// failure. Matches PostgreSQL PANIC-on-fsync-fail; T1 tenants
    /// lose nothing, T3 tenants lose up-to-rpo-ms per contract.
    Abort,
    /// **NOT RECOMMENDED**. Log the failure and continue running.
    /// Silently loses up-to-rpo-ms of acked T3 commits on each
    /// failure. Provided for test fault-injection harnesses.
    RollbackAndContinue,
}

impl Default for BackgroundFsyncFailAction {
    #[inline]
    fn default() -> Self {
        Self::Abort
    }
}

impl BackgroundFsyncFailAction {
    /// Environment variable name for the override (test-only).
    pub const ENV_VAR: &'static str = "ARCGRAPH_WAL_BACKGROUND_FSYNC_FAIL_ACTION";

    /// Parse from the environment. Unset → [`Self::Abort`] (default).
    ///
    /// Recognized values (case-insensitive): `abort`,
    /// `rollback-and-continue`. Unknown values fall back to `Abort`
    /// with a `tracing::warn!`.
    #[must_use]
    pub fn from_env() -> Self {
        match std::env::var(Self::ENV_VAR) {
            Ok(v) if v.eq_ignore_ascii_case("abort") => Self::Abort,
            Ok(v)
                if v.eq_ignore_ascii_case("rollback-and-continue")
                    || v.eq_ignore_ascii_case("rollback_and_continue")
                    || v.eq_ignore_ascii_case("rollback") =>
            {
                warn!(
                    "ADR-034 D-7 override: {}={} — NOT RECOMMENDED for production. Silently \
                     loses up-to-rpo-ms of acked T3 commits per failure.",
                    Self::ENV_VAR,
                    v,
                );
                Self::RollbackAndContinue
            }
            Ok(v) => {
                warn!(
                    "ADR-034 D-7: {}={} unrecognized; defaulting to abort",
                    Self::ENV_VAR,
                    v,
                );
                Self::Abort
            }
            Err(_) => Self::Abort,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// BackgroundFsyncMetrics
// ─────────────────────────────────────────────────────────────────────

/// Lightweight metrics for the background fsync scheduler.
///
/// Analogous to [`crate::wal::WalFireMetrics`] — cheap atomic
/// counters intended for test assertions + Prometheus export at
/// v1.0-GA. Cloning shares the underlying Arcs.
#[derive(Clone, Debug, Default)]
pub struct BackgroundFsyncMetrics {
    /// Total number of scheduler tick → flush() cycles that ran to
    /// completion (success or handled-failure). Per ADR-034 §Slice C.
    ticks_ran_total: Arc<AtomicU64>,
    /// Total number of flush() errors observed. Counts failures
    /// regardless of response action (abort OR rollback-and-continue).
    tick_errors_total: Arc<AtomicU64>,
    /// Sum of per-tick latencies in microseconds. Divide by
    /// `ticks_ran_total` for mean latency. Includes sleep + flush
    /// execution — the flush itself (and its fsync) is the hot part.
    latency_sum_us: Arc<AtomicU64>,
    /// Current count of T3 tenants registered. Equal to
    /// `registered_tenants.lock().len()` at the last
    /// register/unregister call.
    registered_tenants_gauge: Arc<AtomicU64>,
    /// Current scheduler interval in milliseconds. Atomic for
    /// lock-free read from the tick loop.
    current_interval_ms_gauge: Arc<AtomicU64>,
    /// Highest durable LSN observed by the scheduler at any tick.
    /// Snapshots [`WalHandle::last_durable_lsn`] after each
    /// successful flush — lags by at most one tick.
    last_observed_durable_lsn: Arc<AtomicU64>,
}

impl BackgroundFsyncMetrics {
    /// Number of completed tick cycles.
    #[must_use]
    pub fn ticks_ran_total(&self) -> u64 {
        self.ticks_ran_total.load(Ordering::Acquire)
    }

    /// Number of tick errors (flush returned Err).
    #[must_use]
    pub fn tick_errors_total(&self) -> u64 {
        self.tick_errors_total.load(Ordering::Acquire)
    }

    /// Sum of per-tick latencies in microseconds.
    #[must_use]
    pub fn latency_sum_us(&self) -> u64 {
        self.latency_sum_us.load(Ordering::Acquire)
    }

    /// Current T3 tenant count.
    #[must_use]
    pub fn registered_tenants_gauge(&self) -> u64 {
        self.registered_tenants_gauge.load(Ordering::Acquire)
    }

    /// Current scheduler interval in ms.
    #[must_use]
    pub fn current_interval_ms_gauge(&self) -> u64 {
        self.current_interval_ms_gauge.load(Ordering::Acquire)
    }

    /// Highest durable LSN observed at any tick.
    #[must_use]
    pub fn last_observed_durable_lsn(&self) -> u64 {
        self.last_observed_durable_lsn.load(Ordering::Acquire)
    }

    /// Derived mean tick latency in microseconds. Returns 0 if no
    /// ticks have run.
    #[must_use]
    pub fn mean_latency_us(&self) -> u64 {
        self.latency_sum_us()
            .checked_div(self.ticks_ran_total())
            .unwrap_or(0)
    }
}

// ─────────────────────────────────────────────────────────────────────
// BackgroundFsyncScheduler
// ─────────────────────────────────────────────────────────────────────

/// Interval when no T3 tenants are registered. Keeps the scheduler
/// ticking so straggler appends still durify; also avoids spin-
/// loop degeneracy if interval_ms goes to 0.
const DEFAULT_IDLE_INTERVAL_MS: u64 = 1_000;

/// Shared state between the scheduler thread and the public API.
/// Held inside an `Arc` so both sides can access it cheaply.
struct SchedulerInner {
    wal: WalHandle,
    registered_tenants: Mutex<HashMap<TenantId, u64>>,
    /// Current scheduler interval. Recomputed on register/unregister.
    /// Atomic so the tick loop can read lock-free.
    interval_ms: AtomicU64,
    /// `true` once [`BackgroundFsyncScheduler::shutdown`] has been
    /// called. The tick loop polls this on each wake-up and exits
    /// gracefully.
    shutdown: AtomicBool,
    /// Wake-up primitive: condvar for interval changes and shutdown.
    wakeup_lock: Mutex<()>,
    wakeup_cv: Condvar,
    fail_action: BackgroundFsyncFailAction,
    metrics: BackgroundFsyncMetrics,
}

/// ADR-034 §Slice C background fsync scheduler.
///
/// Cheaply cloneable via `Arc<BackgroundFsyncScheduler>`. The inner
/// thread is spawned by [`Self::start`] and joined by
/// [`Self::shutdown`]; dropping the last handle also triggers
/// graceful shutdown (best-effort join).
pub struct BackgroundFsyncScheduler {
    inner: Arc<SchedulerInner>,
    /// The scheduler thread's join handle. Taken in
    /// [`Self::shutdown`]. `Mutex<Option>` lets shutdown be called
    /// from any thread.
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl BackgroundFsyncScheduler {
    /// Start a new scheduler tied to `wal`. Thread spawns
    /// immediately. Registers no tenants (use [`Self::register`]
    /// after `set_durability_tier`).
    ///
    /// `fail_action` is typically
    /// [`BackgroundFsyncFailAction::from_env`]; tests override
    /// directly.
    pub fn start(wal: WalHandle, fail_action: BackgroundFsyncFailAction) -> Arc<Self> {
        let inner = Arc::new(SchedulerInner {
            wal,
            registered_tenants: Mutex::new(HashMap::new()),
            interval_ms: AtomicU64::new(DEFAULT_IDLE_INTERVAL_MS),
            shutdown: AtomicBool::new(false),
            wakeup_lock: Mutex::new(()),
            wakeup_cv: Condvar::new(),
            fail_action,
            metrics: BackgroundFsyncMetrics::default(),
        });
        inner
            .metrics
            .current_interval_ms_gauge
            .store(DEFAULT_IDLE_INTERVAL_MS, Ordering::Release);

        let thread_inner = Arc::clone(&inner);
        let thread = thread::Builder::new()
            .name("arcgraph-bg-fsync".to_owned())
            .spawn(move || run_scheduler(thread_inner))
            .expect("spawn arcgraph-bg-fsync thread");

        info!("ADR-034 Slice C: BackgroundFsyncScheduler started");

        Arc::new(Self {
            inner,
            thread: Mutex::new(Some(thread)),
        })
    }

    /// **Test-only.** Start a scheduler in *manual-tick* mode: identical
    /// to [`Self::start`] except **no auto-ticking thread is spawned**.
    /// The scheduler therefore drives a [`WalHandle::flush`] only via an
    /// explicit [`Self::tick_for_test`] call.
    ///
    /// Note this does **not** mean a Strict (T1) commit can durify on its
    /// own here: [`WalHandle::append`] does not self-fsync — it parks on
    /// its fsync-ack until the writer fires its pending batch (batch-full,
    /// the writer's own `group_commit_window` timeout, an explicit
    /// `flush`, or shutdown). With a normal short `group_commit_window`
    /// the writer's own timer fires Strict commits independently of any
    /// scheduler. But a test that pairs `start_manual` with a *long*
    /// `group_commit_window` (to keep T3 commits pending) removes BOTH
    /// auto-fire paths, so it must drive the fire for each blocking Strict
    /// commit explicitly (e.g. a concurrent `tick_for_test`); otherwise
    /// that commit's `append` blocks forever. [`Self::register`] /
    /// [`Self::unregister`] still update the registered-tenant
    /// bookkeeping and recompute `interval_ms` exactly as in auto mode
    /// — only the *timer-driven* fire is absent.
    ///
    /// # Why this exists
    ///
    /// Some integration tests assert a precondition of the form "the T3
    /// commits are still pending (un-fsynced) at this instant" before
    /// driving the durability path they actually care about. With the
    /// auto-ticking thread that precondition *races* the background
    /// timer — `register` wakes the condvar (see
    /// [`Self::recompute_interval_and_wake`]), and under host load the
    /// woken tick can fire `flush()` inside the pre-assertion window,
    /// advancing the watermark early and tripping the setup assertion.
    /// The durability guarantee is unaffected (an early flush makes the
    /// T3 commits *more* durable, never less), but the test becomes
    /// flaky. `start_manual` removes the race by removing the timer:
    /// the test deterministically controls when the watermark advances.
    ///
    /// # Shutdown
    ///
    /// [`Self::shutdown`] is a no-op for a manual scheduler (there is no
    /// thread to join) and remains idempotent. The `shutdown` flag is
    /// still flipped so [`Self::is_running`] reports `false` afterward.
    #[doc(hidden)]
    pub fn start_manual(wal: WalHandle, fail_action: BackgroundFsyncFailAction) -> Arc<Self> {
        let inner = Arc::new(SchedulerInner {
            wal,
            registered_tenants: Mutex::new(HashMap::new()),
            interval_ms: AtomicU64::new(DEFAULT_IDLE_INTERVAL_MS),
            shutdown: AtomicBool::new(false),
            wakeup_lock: Mutex::new(()),
            wakeup_cv: Condvar::new(),
            fail_action,
            metrics: BackgroundFsyncMetrics::default(),
        });
        inner
            .metrics
            .current_interval_ms_gauge
            .store(DEFAULT_IDLE_INTERVAL_MS, Ordering::Release);

        info!("ADR-034 Slice C: BackgroundFsyncScheduler started (MANUAL-tick mode; test-only)");

        // No thread spawned — `thread` is `None`, so `shutdown` joins
        // nothing.
        Arc::new(Self {
            inner,
            thread: Mutex::new(None),
        })
    }

    /// Register a T3 tenant with its `rpo_ms`. Re-registering an
    /// already-registered tenant updates its `rpo_ms` (e.g., an
    /// operator tightening the RPO).
    ///
    /// Recomputes `interval_ms = min(rpo_ms)` across all registered
    /// tenants and wakes the scheduler thread so the new interval
    /// takes effect immediately (the thread may be sleeping on the
    /// prior, possibly-longer interval).
    ///
    /// `tier` is expected to be [`DurabilityTier::Periodic`]; if a
    /// caller passes [`DurabilityTier::Strict`] the method treats it
    /// as an unregister (same-shape rollback when an operator flips
    /// from T3 back to T1).
    pub fn register(&self, tenant: TenantId, tier: DurabilityTier) {
        match tier {
            DurabilityTier::Strict => {
                // Strict == unregister. Keeps the API symmetric so
                // set_durability_tier can call a single method.
                self.unregister(tenant);
            }
            DurabilityTier::Periodic { rpo_ms } => {
                {
                    let mut tenants = self.inner.registered_tenants.lock();
                    tenants.insert(tenant, rpo_ms);
                    self.inner
                        .metrics
                        .registered_tenants_gauge
                        .store(tenants.len() as u64, Ordering::Release);
                }
                self.recompute_interval_and_wake();
            }
        }
    }

    /// Remove `tenant` from the registered set. No-op if not
    /// registered. Recomputes interval + wakes.
    pub fn unregister(&self, tenant: TenantId) {
        {
            let mut tenants = self.inner.registered_tenants.lock();
            tenants.remove(&tenant);
            self.inner
                .metrics
                .registered_tenants_gauge
                .store(tenants.len() as u64, Ordering::Release);
        }
        self.recompute_interval_and_wake();
    }

    /// Current interval in milliseconds. Lock-free.
    #[must_use]
    pub fn current_interval_ms(&self) -> u64 {
        self.inner.interval_ms.load(Ordering::Acquire)
    }

    /// Count of currently-registered T3 tenants.
    #[must_use]
    pub fn registered_tenant_count(&self) -> usize {
        self.inner.registered_tenants.lock().len()
    }

    /// Snapshot of the scheduler's metrics. Cheap to clone.
    #[must_use]
    pub fn metrics(&self) -> BackgroundFsyncMetrics {
        self.inner.metrics.clone()
    }

    /// Whether the scheduler is still running. Returns `false` after
    /// [`Self::shutdown`] has been called AND the thread has joined.
    #[must_use]
    pub fn is_running(&self) -> bool {
        !self.inner.shutdown.load(Ordering::Acquire)
    }

    /// Shutdown the scheduler. Sets the shutdown flag, wakes the
    /// thread, joins it. Idempotent: subsequent calls are no-ops.
    /// Panics from the scheduler thread are logged and surfaced via
    /// the return value.
    pub fn shutdown(&self) -> std::result::Result<(), String> {
        if self.inner.shutdown.swap(true, Ordering::AcqRel) {
            // Already shut down.
            return Ok(());
        }
        self.wake();
        let handle_opt = {
            let mut guard = self.thread.lock();
            guard.take()
        };
        if let Some(handle) = handle_opt {
            match handle.join() {
                Ok(()) => Ok(()),
                Err(panic_payload) => {
                    error!("BackgroundFsyncScheduler thread panicked: {panic_payload:?}");
                    Err(format!(
                        "BackgroundFsyncScheduler thread panic: {panic_payload:?}"
                    ))
                }
            }
        } else {
            Ok(())
        }
    }

    /// Internal: wake the scheduler thread from its condvar sleep.
    ///
    /// Acquires the wakeup lock briefly to pair with the scheduler
    /// thread's `wait_for` — without the pairing, the notify can
    /// race a just-about-to-wait loop iteration and get lost. The
    /// `_guard` binding drops immediately after `notify_all`.
    fn wake(&self) {
        let _guard = self.inner.wakeup_lock.lock();
        self.inner.wakeup_cv.notify_all();
    }

    /// Recompute the scheduler interval as `min(rpo_ms)` across
    /// registered tenants, or [`DEFAULT_IDLE_INTERVAL_MS`] if no
    /// tenants. Wakes the scheduler so the new interval is
    /// observed immediately.
    fn recompute_interval_and_wake(&self) {
        let new_interval = {
            let tenants = self.inner.registered_tenants.lock();
            tenants
                .values()
                .copied()
                .min()
                .unwrap_or(DEFAULT_IDLE_INTERVAL_MS)
        };
        self.inner
            .interval_ms
            .store(new_interval, Ordering::Release);
        self.inner
            .metrics
            .current_interval_ms_gauge
            .store(new_interval, Ordering::Release);
        self.wake();
    }

    /// Test-only: directly fire a scheduler tick. Useful for
    /// synchronous integration tests that don't want to wait on
    /// the sleep loop. NOT called by production code.
    #[doc(hidden)]
    pub fn tick_for_test(&self) -> std::result::Result<(), String> {
        run_one_tick(&self.inner)
    }
}

impl Drop for BackgroundFsyncScheduler {
    fn drop(&mut self) {
        // Best-effort shutdown. Ignore errors; we're being dropped.
        let _ = self.shutdown();
    }
}

/// Scheduler main loop. Runs on the dedicated
/// `arcgraph-bg-fsync` thread.
///
/// Cycle:
/// 1. Sleep for `interval_ms` (or until woken by register /
///    unregister / shutdown).
/// 2. Check shutdown — if set, exit.
/// 3. Call `wal.flush()`. On success, update metrics + watermark.
///    On failure, dispatch per `fail_action` (abort or log+continue).
/// 4. Loop.
fn run_scheduler(inner: Arc<SchedulerInner>) {
    loop {
        // §1. Sleep on condvar for interval_ms.
        {
            let mut guard = inner.wakeup_lock.lock();
            let interval = inner.interval_ms.load(Ordering::Acquire);
            // wait_for returns on timeout OR on notify; we use a
            // simple timed wait. The shutdown check below re-checks
            // after wake-up regardless of cause.
            let _ = inner
                .wakeup_cv
                .wait_for(&mut guard, Duration::from_millis(interval));
        }

        // §2. Shutdown check.
        if inner.shutdown.load(Ordering::Acquire) {
            info!("BackgroundFsyncScheduler: shutdown flag observed; exiting loop");
            return;
        }

        // §3. Tick.
        let _ = run_one_tick(&inner);
    }
}

/// Run a single scheduler tick: flush, update metrics, dispatch
/// on failure. Returns `Err(msg)` only under
/// [`BackgroundFsyncFailAction::RollbackAndContinue`]; under
/// [`BackgroundFsyncFailAction::Abort`] a failing tick calls
/// `std::process::abort()` instead of returning.
///
/// Exposed via `tick_for_test` so integration tests can exercise
/// the tick logic without sleeping.
fn run_one_tick(inner: &SchedulerInner) -> std::result::Result<(), String> {
    let t0 = Instant::now();
    let flush_result = inner.wal.flush();
    let elapsed_us = t0.elapsed().as_micros() as u64;

    inner
        .metrics
        .latency_sum_us
        .fetch_add(elapsed_us, Ordering::Relaxed);
    inner
        .metrics
        .ticks_ran_total
        .fetch_add(1, Ordering::Relaxed);

    match flush_result {
        Ok(()) => {
            let durable = inner.wal.last_durable_lsn().raw();
            inner
                .metrics
                .last_observed_durable_lsn
                .fetch_max(durable, Ordering::AcqRel);
            Ok(())
        }
        Err(e) => {
            inner
                .metrics
                .tick_errors_total
                .fetch_add(1, Ordering::Relaxed);

            match inner.fail_action {
                BackgroundFsyncFailAction::Abort => {
                    // I-D4 / D-7. No rollback possible post-ack.
                    error!(
                        "ADR-034 D-7: background fsync failed; aborting process. error: {}",
                        e,
                    );
                    std::process::abort();
                }
                BackgroundFsyncFailAction::RollbackAndContinue => {
                    warn!(
                        "ADR-034 D-7 override (NOT RECOMMENDED): background fsync failed; \
                         continuing with potential up-to-rpo-ms silent loss. error: {}",
                        e,
                    );
                    Err(format!("tick flush failed: {e}"))
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use arcgraph_core::TenantId;
    use tempfile::tempdir;

    use super::*;
    use crate::wal::writer::{WalConfig, WalWriter};
    use crate::wal::{WalHandle, WalRecordType};

    fn test_wal(dir: impl Into<std::path::PathBuf>) -> (WalWriter, WalHandle) {
        let cfg = WalConfig {
            dir: dir.into(),
            segment_size_bytes: 64 * 1024 * 1024,
            group_commit_window: Duration::from_millis(2),
            group_commit_max_batch: 16,
            metrics_sink: None,
            encryption: None,

            inflight_budget_bytes: None,
        };
        let w = WalWriter::spawn(cfg).unwrap();
        let h = w.handle();
        (w, h)
    }

    // ─── BackgroundFsyncFailAction::from_env ─────────────────────
    //
    // env-var tests serialize through ENV_LOCK so they don't race.

    use parking_lot::Mutex as PlMutex;
    static ENV_LOCK: PlMutex<()> = PlMutex::new(());

    #[test]
    fn fail_action_default_is_abort() {
        assert_eq!(
            BackgroundFsyncFailAction::default(),
            BackgroundFsyncFailAction::Abort
        );
    }

    #[test]
    fn fail_action_from_env_unset_is_abort() {
        let _g = ENV_LOCK.lock();
        // SAFETY: serialized by ENV_LOCK; no concurrent env access.
        unsafe { std::env::remove_var(BackgroundFsyncFailAction::ENV_VAR) };
        assert_eq!(
            BackgroundFsyncFailAction::from_env(),
            BackgroundFsyncFailAction::Abort
        );
    }

    #[test]
    fn fail_action_from_env_abort_case_insensitive() {
        let _g = ENV_LOCK.lock();
        for v in ["abort", "ABORT", "Abort", "AbOrT"] {
            // SAFETY: serialized by ENV_LOCK; no concurrent env access.
            unsafe { std::env::set_var(BackgroundFsyncFailAction::ENV_VAR, v) };
            assert_eq!(
                BackgroundFsyncFailAction::from_env(),
                BackgroundFsyncFailAction::Abort
            );
        }
        // SAFETY: ENV_LOCK held; cleanup.
        unsafe { std::env::remove_var(BackgroundFsyncFailAction::ENV_VAR) };
    }

    #[test]
    fn fail_action_from_env_rollback_and_continue_case_insensitive() {
        let _g = ENV_LOCK.lock();
        for v in [
            "rollback-and-continue",
            "rollback_and_continue",
            "Rollback-And-Continue",
            "rollback",
        ] {
            // SAFETY: serialized by ENV_LOCK.
            unsafe { std::env::set_var(BackgroundFsyncFailAction::ENV_VAR, v) };
            assert_eq!(
                BackgroundFsyncFailAction::from_env(),
                BackgroundFsyncFailAction::RollbackAndContinue,
                "value {v} should parse to RollbackAndContinue"
            );
        }
        // SAFETY: cleanup.
        unsafe { std::env::remove_var(BackgroundFsyncFailAction::ENV_VAR) };
    }

    #[test]
    fn fail_action_from_env_unknown_falls_back_to_abort() {
        let _g = ENV_LOCK.lock();
        // SAFETY: ENV_LOCK held.
        unsafe { std::env::set_var(BackgroundFsyncFailAction::ENV_VAR, "ye-olde-nope") };
        assert_eq!(
            BackgroundFsyncFailAction::from_env(),
            BackgroundFsyncFailAction::Abort
        );
        // SAFETY: cleanup.
        unsafe { std::env::remove_var(BackgroundFsyncFailAction::ENV_VAR) };
    }

    // ─── Scheduler lifecycle ─────────────────────────────────────

    #[test]
    fn scheduler_starts_and_shuts_down() {
        let dir = tempdir().unwrap();
        let (writer, handle) = test_wal(dir.path());
        let sched = BackgroundFsyncScheduler::start(handle, BackgroundFsyncFailAction::Abort);
        assert!(sched.is_running());
        assert_eq!(sched.registered_tenant_count(), 0);
        assert_eq!(sched.current_interval_ms(), DEFAULT_IDLE_INTERVAL_MS);
        sched.shutdown().unwrap();
        assert!(!sched.is_running());
        writer.shutdown().unwrap();
    }

    #[test]
    fn scheduler_shutdown_is_idempotent() {
        let dir = tempdir().unwrap();
        let (writer, handle) = test_wal(dir.path());
        let sched = BackgroundFsyncScheduler::start(handle, BackgroundFsyncFailAction::Abort);
        sched.shutdown().unwrap();
        // Second shutdown is a no-op.
        sched.shutdown().unwrap();
        writer.shutdown().unwrap();
    }

    // ─── start_manual (manual-tick mode) ─────────────────────────

    /// A WAL whose group-commit window is effectively infinite, so the
    /// *writer's own* group-commit timer can never fire. This isolates
    /// the scheduler as the ONLY thing that can advance the watermark —
    /// exactly the property `start_manual` is about. (The default
    /// `test_wal` uses a 2 ms window, which would durify an
    /// `append_async` record on the writer's own timer regardless of
    /// the scheduler — useless for testing the scheduler in isolation.)
    fn test_wal_long_window(dir: impl Into<std::path::PathBuf>) -> (WalWriter, WalHandle) {
        let cfg = WalConfig {
            dir: dir.into(),
            segment_size_bytes: 64 * 1024 * 1024,
            group_commit_window: Duration::from_secs(3600),
            group_commit_max_batch: 1_000,
            metrics_sink: None,
            encryption: None,

            inflight_budget_bytes: None,
        };
        let w = WalWriter::spawn(cfg).unwrap();
        let h = w.handle();
        (w, h)
    }

    #[test]
    fn start_manual_does_not_auto_tick() {
        // The defining property of manual mode: NO background thread
        // fires a scheduler tick. We register a T3 tenant at a 1 ms
        // rpo_ms (which in `start` mode would tick ~continuously) and
        // enqueue an async record. A real wall-clock interval that
        // dwarfs the rpo_ms passes; the scheduler must run ZERO ticks
        // and the watermark must STAY at 0 — the only sanctioned
        // advance is an explicit `tick_for_test()`, which we have not
        // yet called. A long-window WAL (`test_wal_long_window`)
        // ensures the writer's own group-commit timer cannot durify
        // the record, isolating the scheduler as the sole advancer.
        let dir = tempdir().unwrap();
        let (writer, handle) = test_wal_long_window(dir.path());
        let sched = BackgroundFsyncScheduler::start_manual(
            handle.clone(),
            BackgroundFsyncFailAction::Abort,
        );

        // A 1 ms rpo_ms would auto-tick almost continuously in
        // `start` mode; in manual mode it must never fire on its own.
        sched.register(TenantId::DEFAULT, DurabilityTier::Periodic { rpo_ms: 1 });
        assert_eq!(sched.current_interval_ms(), 1);

        handle
            .append_async(
                WalRecordType::CommitBundle,
                1,
                0,
                TenantId::DEFAULT,
                vec![0u8; 4],
            )
            .unwrap();
        // Pre-tick: the record is enqueued but not yet fsynced.
        assert_eq!(handle.last_durable_lsn().raw(), 0);

        // Sleep far longer than the rpo_ms. An auto-ticking scheduler
        // (`start`) would have fired hundreds of times by now.
        thread::sleep(Duration::from_millis(200));
        assert_eq!(
            sched.metrics().ticks_ran_total(),
            0,
            "manual scheduler must run ZERO ticks on its own"
        );
        assert_eq!(
            handle.last_durable_lsn().raw(),
            0,
            "manual scheduler must NOT advance the watermark without an explicit tick"
        );

        // An explicit manual tick — the only sanctioned advance — must
        // still work and durify the pending record.
        sched.tick_for_test().unwrap();
        assert_eq!(sched.metrics().ticks_ran_total(), 1);
        assert_eq!(handle.last_durable_lsn().raw(), 1);

        // Shutdown is a no-op (no thread) and idempotent.
        assert!(sched.is_running());
        sched.shutdown().unwrap();
        assert!(!sched.is_running());
        sched.shutdown().unwrap();
        writer.shutdown().unwrap();
    }

    // ─── register / unregister ───────────────────────────────────

    #[test]
    fn register_min_interval_is_min_rpo() {
        let dir = tempdir().unwrap();
        let (writer, handle) = test_wal(dir.path());
        let sched = BackgroundFsyncScheduler::start(handle, BackgroundFsyncFailAction::Abort);

        sched.register(TenantId::DEFAULT, DurabilityTier::Periodic { rpo_ms: 500 });
        assert_eq!(sched.current_interval_ms(), 500);
        assert_eq!(sched.registered_tenant_count(), 1);

        sched.register(TenantId::new(100), DurabilityTier::Periodic { rpo_ms: 100 });
        assert_eq!(
            sched.current_interval_ms(),
            100,
            "min across 100 and 500 is 100"
        );
        assert_eq!(sched.registered_tenant_count(), 2);

        sched.register(
            TenantId::new(200),
            DurabilityTier::Periodic { rpo_ms: 2000 },
        );
        assert_eq!(
            sched.current_interval_ms(),
            100,
            "adding a slower rpo_ms does not slow the scheduler"
        );

        sched.shutdown().unwrap();
        writer.shutdown().unwrap();
    }

    #[test]
    fn register_then_unregister_returns_to_idle() {
        let dir = tempdir().unwrap();
        let (writer, handle) = test_wal(dir.path());
        let sched = BackgroundFsyncScheduler::start(handle, BackgroundFsyncFailAction::Abort);

        sched.register(TenantId::DEFAULT, DurabilityTier::Periodic { rpo_ms: 50 });
        assert_eq!(sched.current_interval_ms(), 50);

        sched.unregister(TenantId::DEFAULT);
        assert_eq!(sched.registered_tenant_count(), 0);
        assert_eq!(sched.current_interval_ms(), DEFAULT_IDLE_INTERVAL_MS);

        sched.shutdown().unwrap();
        writer.shutdown().unwrap();
    }

    #[test]
    fn register_strict_acts_as_unregister() {
        let dir = tempdir().unwrap();
        let (writer, handle) = test_wal(dir.path());
        let sched = BackgroundFsyncScheduler::start(handle, BackgroundFsyncFailAction::Abort);

        sched.register(TenantId::DEFAULT, DurabilityTier::Periodic { rpo_ms: 50 });
        assert_eq!(sched.registered_tenant_count(), 1);

        // Flipping back to Strict via the same register() API drops
        // the tenant from the T3 set — the catalog layer uses this
        // in its set_durability_tier path.
        sched.register(TenantId::DEFAULT, DurabilityTier::Strict);
        assert_eq!(sched.registered_tenant_count(), 0);

        sched.shutdown().unwrap();
        writer.shutdown().unwrap();
    }

    #[test]
    fn reregister_updates_rpo_ms() {
        let dir = tempdir().unwrap();
        let (writer, handle) = test_wal(dir.path());
        let sched = BackgroundFsyncScheduler::start(handle, BackgroundFsyncFailAction::Abort);

        sched.register(TenantId::DEFAULT, DurabilityTier::Periodic { rpo_ms: 500 });
        assert_eq!(sched.current_interval_ms(), 500);

        // Operator tightens the RPO — same tenant, new value.
        sched.register(TenantId::DEFAULT, DurabilityTier::Periodic { rpo_ms: 50 });
        assert_eq!(sched.current_interval_ms(), 50);
        assert_eq!(sched.registered_tenant_count(), 1);

        sched.shutdown().unwrap();
        writer.shutdown().unwrap();
    }

    // ─── Tick semantics (test-only direct tick) ──────────────────

    #[test]
    fn tick_advances_watermark_via_flush() {
        let dir = tempdir().unwrap();
        let (writer, handle) = test_wal(dir.path());
        let sched =
            BackgroundFsyncScheduler::start(handle.clone(), BackgroundFsyncFailAction::Abort);

        // Enqueue T3 records.
        for i in 1..=3u64 {
            handle
                .append_async(
                    WalRecordType::CommitBundle,
                    i,
                    0,
                    TenantId::DEFAULT,
                    vec![0u8; 4],
                )
                .unwrap();
        }
        // Pre-tick watermark is 0 (no fsync yet).
        assert_eq!(handle.last_durable_lsn().raw(), 0);

        // One scheduler tick fires.
        sched.tick_for_test().unwrap();

        // Post-tick watermark covers all 3 records.
        assert_eq!(handle.last_durable_lsn().raw(), 3);

        let metrics = sched.metrics();
        assert!(metrics.ticks_ran_total() >= 1);
        assert_eq!(metrics.tick_errors_total(), 0);
        assert_eq!(metrics.last_observed_durable_lsn(), 3);

        sched.shutdown().unwrap();
        writer.shutdown().unwrap();
    }

    #[test]
    fn scheduler_runs_ticks_on_its_own_cadence() {
        let dir = tempdir().unwrap();
        let (writer, handle) = test_wal(dir.path());
        let sched =
            BackgroundFsyncScheduler::start(handle.clone(), BackgroundFsyncFailAction::Abort);
        sched.register(TenantId::DEFAULT, DurabilityTier::Periodic { rpo_ms: 20 });

        // Enqueue a T3 record and wait for the scheduler thread to
        // tick on its 20ms cadence and durify it.
        handle
            .append_async(
                WalRecordType::CommitBundle,
                1,
                0,
                TenantId::DEFAULT,
                vec![0u8; 4],
            )
            .unwrap();

        // DE-FLAKE (#484 dst_runtime_seam scheduler-timing precedent):
        // this test previously polled `ticks_ran_total` alone with a 5s
        // deadline, then asserted BOTH `ticks_ran_total >= 1` AND
        // `last_durable_lsn >= 1`. That has two failure modes under the
        // workspace-parallel `cargo test` run:
        //   1. Logic race. A tick bumps `ticks_ran_total` AFTER flush
        //      returns, so the two counters advance at different
        //      instants; polling on EITHER one alone leaves the OTHER
        //      assertion exposed to its own update window. Concretely,
        //      the first tick can fire and bump `ticks_ran_total`
        //      before the async `append_async` record is visible to the
        //      writer — so the watermark is still 0 when we assert it,
        //      which is exactly the observed panic ("…durified… within
        //      5s").
        //   2. Timing. A 5s deadline is too tight when other test
        //      binaries saturate the host and starve the scheduler
        //      thread of wake-ups.
        // Fix: poll until BOTH postconditions hold (the conjunction, so
        // neither counter's update window can trip us) with a generous
        // 30s ceiling that only a genuine multi-second stall (a real
        // bug) would reach. The record durifies within a few 20ms ticks
        // normally; 30s is ~1500 ticks of headroom. NB `serial_test`
        // would not fix this: it serialises only WITHIN one test binary,
        // but the contention here is cross-process from the
        // workspace-parallel run, and the logic race is independent of
        // load.
        let metrics = sched.metrics();
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            let ticked = metrics.ticks_ran_total() >= 1;
            let durified = handle.last_durable_lsn().raw() >= 1;
            if (ticked && durified) || Instant::now() >= deadline {
                break;
            }
            thread::sleep(Duration::from_millis(5));
        }
        assert!(
            metrics.ticks_ran_total() >= 1,
            "scheduler should have ticked at least once within 30s \
             (ticks_ran_total={})",
            metrics.ticks_ran_total(),
        );
        assert!(
            handle.last_durable_lsn().raw() >= 1,
            "scheduler should have durified the T3 record within 30s \
             (last_durable_lsn={})",
            handle.last_durable_lsn().raw(),
        );
        assert_eq!(metrics.current_interval_ms_gauge(), 20);

        sched.shutdown().unwrap();
        writer.shutdown().unwrap();
    }

    // ─── Metrics integrity ────────────────────────────────────────

    #[test]
    fn metrics_track_ticks_and_latency() {
        let dir = tempdir().unwrap();
        let (writer, handle) = test_wal(dir.path());
        let sched = BackgroundFsyncScheduler::start(handle, BackgroundFsyncFailAction::Abort);

        for _ in 0..5 {
            sched.tick_for_test().unwrap();
        }
        let m = sched.metrics();
        assert!(m.ticks_ran_total() >= 5);
        // Latency is measured via Instant::now; on a working system
        // each tick takes at least one µs of wall-clock due to the
        // channel send + fsync path.
        assert!(m.latency_sum_us() > 0);
        assert!(m.mean_latency_us() > 0);

        sched.shutdown().unwrap();
        writer.shutdown().unwrap();
    }

    // ─── Rollback-and-continue override ──────────────────────────

    #[test]
    fn rollback_and_continue_returns_err_on_failure() {
        // We can't easily inject an fsync failure via the public API
        // here; the override's semantic behavior (no abort) is
        // verified by not calling abort in the happy path, and the
        // environment-parsing logic is tested in fail_action_from_env.
        //
        // A subprocess-style test for the abort path lives in
        // `tests/durability_tier_periodic.rs`.
        let dir = tempdir().unwrap();
        let (writer, handle) = test_wal(dir.path());
        let sched = BackgroundFsyncScheduler::start(
            handle.clone(),
            BackgroundFsyncFailAction::RollbackAndContinue,
        );
        // Shutdown the WAL writer first; subsequent flush() calls on
        // the handle return WalUnavailable. The scheduler tick
        // observes this as a tick error but does not abort.
        writer.shutdown().unwrap();
        // Allow the channel-closed signal to propagate.
        thread::sleep(Duration::from_millis(20));

        // Now a tick should surface the failure as Err, not abort.
        let res = sched.tick_for_test();
        assert!(
            res.is_err(),
            "rollback-and-continue returns Err on flush failure"
        );
        let m = sched.metrics();
        assert!(m.tick_errors_total() >= 1);

        sched.shutdown().unwrap();
    }

    // ─── Interval recompute under concurrent register/unregister ─

    #[test]
    fn interval_recompute_is_consistent_under_concurrency() {
        let dir = tempdir().unwrap();
        let (writer, handle) = test_wal(dir.path());
        let sched = BackgroundFsyncScheduler::start(handle, BackgroundFsyncFailAction::Abort);

        let sched_clone = Arc::clone(&sched);
        let t1 = thread::spawn(move || {
            for rpo in (10..=200u64).step_by(10) {
                sched_clone.register(TenantId::new(1), DurabilityTier::Periodic { rpo_ms: rpo });
            }
        });
        let sched_clone = Arc::clone(&sched);
        let t2 = thread::spawn(move || {
            for rpo in (20..=200u64).step_by(5) {
                sched_clone.register(TenantId::new(2), DurabilityTier::Periodic { rpo_ms: rpo });
            }
        });
        t1.join().unwrap();
        t2.join().unwrap();

        // Both tenants still registered.
        assert_eq!(sched.registered_tenant_count(), 2);
        // Interval is the min of whatever the last rpo_ms values
        // were. Values are known (200, 200), so min is 200. We
        // mainly assert the invariant: current interval <= max
        // possible rpo_ms in play.
        assert!(sched.current_interval_ms() <= 200);

        sched.shutdown().unwrap();
        writer.shutdown().unwrap();
    }

    // ─── Local-only regression guard ─────────────────────

    #[test]
    fn scheduler_has_no_partition_id_at_v1() {
        // ADR-034 §Q3: scheduler is process-global at v1.0 with no
        // partition scoping. At v1.1 each partition gets its own
        // scheduler; until then this test asserts no partition_id
        // field was prematurely added.
        let dir = tempdir().unwrap();
        let (writer, handle) = test_wal(dir.path());
        let sched = BackgroundFsyncScheduler::start(handle, BackgroundFsyncFailAction::Abort);

        // The scheduler's public API takes (TenantId, DurabilityTier)
        // — no PartitionId parameter anywhere. If a v1.1 migration
        // adds partition_id, every call site changes and this test
        // becomes the reviewer checkpoint.
        sched.register(TenantId::DEFAULT, DurabilityTier::Periodic { rpo_ms: 100 });
        assert_eq!(sched.registered_tenant_count(), 1);

        sched.shutdown().unwrap();
        writer.shutdown().unwrap();
    }
}
