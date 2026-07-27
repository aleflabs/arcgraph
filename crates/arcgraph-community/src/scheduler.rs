//! Daily background refresh scheduler for community detection
//! (ADR-040 §D-7).
//!
//! Mirrors the `BackgroundFsyncScheduler` pattern (see
//! `crates/arcgraph-storage/src/wal/background_fsync.rs`): a
//! dedicated OS thread sleeps on a [`parking_lot::Condvar`] for the
//! configured interval, then drains the pending-refresh queue.
//! Per-tenant failures log `tracing::error!` and the tick continues
//! to the next tenant; one tenant's failure does NOT stop the tick.
//!
//! ## Composition with DF Leiden incremental
//!
//! The scheduler runs the **static** GVE-Leiden algorithm — it is
//! the "canonical reset" against which DF Leiden incremental updates
//! accumulate ε modularity drift. The two compose at the
//! membership-index level: the incremental algorithm mutates
//! [`BTreeMembershipIndex`] between refreshes; the scheduler
//! overwrites it with the canonical static result on each tick.
//!
//! ## Cadence (ADR-040 §D-7)
//!
//! Default: once per UTC day per tenant (24 hours). Configurable
//! via [`SchedulerConfig::interval`] at start time. Per-tenant
//! cadence is a v1.1 candidate; v1.0 ships a single global interval.
//!
//! Why daily? At a v1.0 sustained ingest rate of 5 K writes/sec
//! (PRD NFR-9), 10 K batches translates to ~24 hours, which is the
//! largest cadence at which DF Leiden's modularity-stability claim
//! holds with reasonable conservatism (ADR-040 §D-7).
//!
//! ## v1.0 limitations
//!
//! - One scheduler instance per process; per-tenant interval is a
//!   single global [`SchedulerConfig::interval`] (not per-tenant).
//! - Static refresh is sequential per ADR-040 §D-1.
//! - Hook panics are caught via [`std::panic::catch_unwind`] and
//!   surfaced as a per-tenant failure; the scheduler thread does
//!   NOT die on a hook panic.

use std::collections::{BTreeSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use arcgraph_core::{Lsn, TenantId};
use parking_lot::{Condvar, Mutex};
use tracing::{debug, error, info, warn};

use crate::graph::Graph;
use crate::leiden_static::{GveLeiden, LeidenParams};
use crate::membership_index::BTreeMembershipIndex;

// ─────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────

/// Refresh-hook trait — caller-supplied resolver from
/// [`TenantId`] to graph + membership index + Leiden parameters.
///
/// Implementations must be `Send + Sync` because they're invoked
/// from the scheduler's dedicated OS thread.
///
/// Returning `None` from [`Self::resolve`] is a *soft skip* — the
/// scheduler logs `tracing::debug!` and continues. This is the
/// designed signal for "tenant catalog removed it between
/// `register` and the tick" or "tenant has no edges yet" or
/// "materialisation from CrudStore failed at the production
/// hook".
///
/// Hook implementations **should not panic**; if they do, the
/// scheduler catches the panic, logs `tracing::error!`, increments
/// `total_refresh_failures`, and continues to the next tenant. A
/// future v1.1 may revisit this contract.
///
/// ## Per-tick re-materialisation (post amendment-05)
///
/// Per ADR-040 amendment-05 (Wave 9b Slice 4) the trait surface
/// returns owned `Arc<Graph>` + `Arc<BTreeMembershipIndex>` so
/// production hooks can re-materialise per-tenant Graphs from
/// `CrudStore` on each tick. The pre-amendment-05 borrowed-ref
/// shape (`RefreshInputs<'a>` with `&'a Graph`) cannot accommodate
/// per-tick re-materialisation inside `&'a self` — the borrow
/// checker rejects the lifetime extension and there is no sound
/// safety argument for an `unsafe` escape hatch (PR #235 review §a
/// confirmed via 4 additional storage-shape candidates including
/// crossbeam epoch, ouroboros, yoke, and append-only boxcar).
///
/// `Arc<Graph>` cloning is O(1) (atomic refcount bump). The
/// scheduler's `do_refresh` consumes `inputs.graph` via
/// `&inputs.graph` (deref through `Arc`); per-call cost is one Arc
/// clone (~5 ns).
///
/// ## Closes amendment-04 §D-2 residual concern by code change
///
/// Amendment-04 §D-2 deferred the "0 prod impls" residual concern
/// to M5/M6 with documented residual risk. PR #235 closed the
/// concern *structurally* by introducing the FROZEN-GRAPH
/// `ProductionRefreshHook`; amendment-05 closes the concern
/// *semantically* by retiring the FROZEN-GRAPH posture and
/// shipping per-tick re-mat. Post-amendment-05 there is 1
/// production impl with the right long-run shape.
///
/// ## Liveness and non-reentrancy
///
/// [`Self::resolve`] runs synchronously while the scheduler holds its
/// whole-tick serialization mutex. Implementations must not call
/// [`CommunityRefreshScheduler::tick`] directly or indirectly; re-entering
/// that non-reentrant mutex would deadlock. A blocking implementation extends
/// the tick without a hard deadline: [`SchedulerConfig::max_tick_duration`]
/// warns only after callback work returns; it neither cancels nor times out
/// that work.
pub trait RefreshHook: Send + Sync {
    /// Resolve the per-tenant refresh inputs.
    ///
    /// Returns `None` if the tenant is no longer eligible (catalog
    /// removed it between `register` and the tick, materialisation
    /// from the storage substrate failed, etc.); the scheduler
    /// treats `None` as a soft skip and continues.
    ///
    /// The returned `OwnedRefreshInputs` carries `Arc<Graph>` +
    /// `Arc<BTreeMembershipIndex>`; the Arcs decouple the inner
    /// state's lifetime from `&self`, permitting per-tick re-
    /// materialisation against the live storage substrate per
    /// ADR-040 amendment-05.
    fn resolve(&self, tenant: TenantId) -> Option<OwnedRefreshInputs>;
}

/// Per-refresh inputs resolved by [`RefreshHook::resolve`].
///
/// Owned (`Arc`-shared) per ADR-040 amendment-05: the `Arc<Graph>`
/// and `Arc<BTreeMembershipIndex>` decouple the per-tenant state's
/// lifetime from the hook's `&self`, permitting hooks to re-
/// materialise Graphs per tick. The Arcs drop at end of
/// `do_refresh` (refcount decrements to whatever the hook's own
/// cache holds; typically 1).
pub struct OwnedRefreshInputs {
    /// Graph snapshot for the tenant. The hook produces a fresh
    /// `Arc<Graph>` per tick (production posture); test hooks may
    /// return the same Arc across ticks if they hold a cached one.
    pub graph: Arc<Graph>,
    /// Membership index the [`GveLeiden::install_into`] result
    /// will overwrite. Typically the same workspace-wide
    /// `Arc<BTreeMembershipIndex>` the
    /// [`crate::SharedBTreeIndexProvider`] holds (so router-side
    /// reads observe the install).
    pub index: Arc<BTreeMembershipIndex>,
    /// Leiden parameters for this tenant. Typically
    /// [`LeidenParams::default`].
    pub params: LeidenParams,
    /// Count of leading vertex slots in `graph` that are phantom
    /// (i.e., correspond to `NodeId`s the production allocator
    /// never emits and that must NOT surface in the membership
    /// index). The scheduler forwards this to
    /// [`GveLeiden::install_into`] which drops the first
    /// `n_skip_prefix` entries from each level's emitted pairs.
    ///
    /// The engine's `CrudStoreGraphAdapter` (in
    /// `arcgraph-storage::engine::graph_adapter`) sizes graphs as
    /// `n = node_high_water + 1` so vertex `0` corresponds to the
    /// reserved `NodeId::ZERO` sentinel; that hook returns
    /// `n_skip_prefix = 1`. Standalone test hooks whose graphs
    /// use the natural `0..n` indexing return `0`.
    pub n_skip_prefix: u32,
}

/// Community-resident observability seam (ADR-202).
///
/// The scheduler notifies the observer **synchronously, once per
/// successful per-tenant refresh, AFTER [`GveLeiden::install_into`]
/// has returned** — i.e. the new community snapshot is installed and
/// visible to readers when the call fires. The observer is NEVER
/// invoked on soft-skip (hook returned `None`) or on a failed /
/// panicked refresh: "last run" honestly means "last refresh whose
/// result is actually installed".
///
/// This trait exists because the design-v2 §10.2
/// `arcgraph_leiden_last_run_seconds` producer lives HERE, beneath
/// `arcgraph-storage` in the dependency graph, so it cannot reach the
/// storage-resident ADR-045 `MetricsSink` (PD-7 bounded contexts —
/// `arcgraph-storage → arcgraph-community`, never the inverse). The
/// concrete impl is `arcgraph-mcp::transport::metrics::MetricsRegistry`,
/// which sets the per-tenant gauge to the current Unix time so the
/// shipped alert contract
/// `time() - arcgraph_leiden_last_run_seconds > (48 * 3600)`
/// (docs/grafana/alerts.yml `ArcGraphLeidenFreshnessStale`) evaluates
/// correctly. The observer deliberately receives no timestamp: it is
/// called synchronously at completion, and the impl owning the metric
/// representation also owns the clock read (mirrors
/// `MetricsSink::record_hot_vertex_warning`).
///
/// # Panic containment
///
/// Implementations **should not panic**. If one does, the scheduler
/// catches the panic, logs `tracing::error!`, and continues — the
/// refresh still counts as a success (it WAS installed) and the
/// scheduler thread survives. See `refresh_one_tenant`.
///
/// The callback runs synchronously under the whole-tick serialization mutex.
/// Implementations must not call [`CommunityRefreshScheduler::tick`] directly
/// or indirectly; doing so would deadlock. Blocking here can extend the tick
/// indefinitely because [`SchedulerConfig::max_tick_duration`] is advisory
/// and does not cancel or time out callback work.
///
/// # Trait minimality (`feedback_avoid_speculative_scaffolding.md`)
///
/// One method, one producer call site (`refresh_one_tenant`'s success
/// arm), one concrete impl — all shipped in the same PR per ADR-202.
/// Adding a method requires a same-PR producer caller + consumer, or
/// an ADR-202 follow-up.
pub trait RefreshObserver: Send + Sync + std::fmt::Debug + 'static {
    /// Called once per successful per-tenant community refresh,
    /// after the result is installed into the membership index.
    fn record_refresh_success(&self, tenant: TenantId);
}

/// Configuration for [`CommunityRefreshScheduler::start`].
#[derive(Debug, Clone, Copy)]
pub struct SchedulerConfig {
    /// Interval between ticks. ADR-040 §D-7 default = 24 hours.
    pub interval: Duration,
    /// Soft cap on tick duration. If a tick exceeds this, the
    /// scheduler emits `tracing::warn!`. Does NOT abort the tick.
    pub max_tick_duration: Duration,
    /// Initial value of the scheduler's install-LSN allocator
    /// (per ADR-041 §D-3b). The first scheduler tick allocates
    /// this LSN; subsequent ticks allocate `+1` each. Default is
    /// `Lsn::new(1)` so `Lsn::ZERO` is reserved as "before any
    /// install". Tests that manually pre-populate the membership
    /// index at low LSNs override this so the scheduler's
    /// allocator starts above the manual pre-populates.
    pub initial_install_lsn: Lsn,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            // ADR-040 §D-7: once per UTC day per tenant.
            interval: Duration::from_secs(24 * 60 * 60),
            // Soft cap: 1 minute. Beyond this we log a warning so an
            // operator notices a runaway tenant; the tick itself
            // continues to completion regardless.
            max_tick_duration: Duration::from_secs(60),
            // ADR-041 §D-3b: install LSNs start at 1 so Lsn::ZERO
            // is reserved as "before any install" — read_lsn=0
            // returns empty / None per the membership-index
            // history-binary-search contract.
            initial_install_lsn: Lsn::new(1),
        }
    }
}

/// Health snapshot for the scheduler.
///
/// Returned by [`CommunityRefreshScheduler::health_check`]. All
/// fields are point-in-time samples; the scheduler may advance to
/// a new tick between the call and the snapshot's read by the
/// caller. The snapshot itself is internally consistent (taken
/// under a single lock).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchedulerHealth {
    /// Number of tenants currently registered.
    pub registered_tenants: usize,
    /// `true` if no tick is currently in progress (the scheduler
    /// is sleeping or completed its last tick). `false` if a tick
    /// is mid-execution.
    pub last_tick_completed: bool,
    /// Total ticks initiated (success or partial-failure).
    pub total_ticks: u64,
    /// Sum of per-tenant failures across all ticks. Includes
    /// `RefreshHook::resolve` returning `None` (soft-skip is NOT
    /// counted), hook panics, and `GveLeiden` errors (the static
    /// algorithm currently never returns errors, but the slot is
    /// reserved for v1.1 fallibility).
    ///
    /// Soft skips (hook returned `None`) are NOT counted as
    /// failures — they're expected when a tenant's catalog state
    /// changes between register and the tick. Use
    /// [`Self::total_soft_skips`] for that count.
    pub total_refresh_failures: u64,
    /// Total soft skips (hook returned `None`) across all ticks.
    pub total_soft_skips: u64,
    /// `true` if the scheduler thread has been shut down.
    pub shut_down: bool,
}

// ─────────────────────────────────────────────────────────────────
// Internals
// ─────────────────────────────────────────────────────────────────

/// Internal mutable state shared between the public API and the
/// scheduler thread. Held inside `Arc` for cheap cloning.
struct SchedulerInner {
    /// Set of registered tenants. `BTreeSet` for stable iteration
    /// order on each tick (so test assertions and log output are
    /// deterministic).
    registered: Mutex<BTreeSet<TenantId>>,
    /// Pending tenants for the next tick. Filled at tick start by
    /// snapshotting `registered`, then drained one-by-one.
    pending: Mutex<VecDeque<TenantId>>,
    /// Wake-up primitive. Pairs with `wakeup_lock` per
    /// `parking_lot` convention.
    wakeup_lock: Mutex<()>,
    wakeup_cv: Condvar,
    /// Set to `true` by [`CommunityRefreshScheduler::shutdown`].
    /// The scheduler thread polls this on each wake-up.
    shutdown: AtomicBool,
    /// Set to `true` while a tick is mid-execution.
    tick_in_progress: AtomicBool,
    /// Serializes natural-cadence and forced ticks. The install-LSN allocator
    /// is monotonic, but reserving LSN N before a Leiden run does not by
    /// itself prevent a concurrent run from installing N+1 first. Holding
    /// this guard for the whole sweep also protects the shared pending queue.
    tick_serialization: Mutex<()>,
    /// Counter: total ticks fired (success or partial failure).
    total_ticks: AtomicU64,
    /// Counter: per-tenant failures across all ticks (NOT soft
    /// skips — see [`SchedulerHealth::total_refresh_failures`]).
    total_refresh_failures: AtomicU64,
    /// Counter: soft skips (hook returned `None`) across all ticks.
    total_soft_skips: AtomicU64,
    /// Configuration. Immutable after `start`.
    cfg: SchedulerConfig,
    /// Caller-supplied refresh hook. `Arc` because the scheduler
    /// thread holds a reference for the duration of its lifetime
    /// and we need cheap shared ownership.
    hook: Arc<dyn RefreshHook>,
    /// Optional ADR-202 observability seam. `None` (the default via
    /// [`CommunityRefreshScheduler::start`]) costs one nullable-ptr
    /// check per successful refresh — the same zero-overhead-when-
    /// unwired posture as ADR-045's `Option<Arc<dyn MetricsSink>>`.
    observer: Option<Arc<dyn RefreshObserver>>,
    /// Monotonic LSN allocator for install_level calls per
    /// ADR-041 §D-3b. The scheduler advances this on every
    /// successful refresh; the value tags the install in the
    /// membership-index history so cross-substrate read_lsn
    /// snapshots resolve to the correct visible install. Starts
    /// at 1 (LSN::ZERO is reserved as "before any install").
    /// Shared with the storage-layer LSN allocator in v1.1; at
    /// v1.0 it is locally allocated to keep the scheduler
    /// self-contained.
    next_install_lsn: AtomicU64,
}

// ─────────────────────────────────────────────────────────────────
// Public scheduler
// ─────────────────────────────────────────────────────────────────

/// Daily community-refresh scheduler (ADR-040 §D-7).
///
/// Lifecycle:
///
/// 1. [`Self::start`] spawns the dedicated OS thread named
///    `"arcgraph-community-refresh"`.
/// 2. Callers register tenants via [`Self::register`] / unregister
///    via [`Self::unregister`].
/// 3. The thread sleeps on a [`parking_lot::Condvar`] for
///    [`SchedulerConfig::interval`] and on each wake-up either
///    runs a tick (if `interval` elapsed) or re-checks the
///    shutdown flag.
/// 4. [`Self::shutdown`] sets the flag, wakes the thread, joins.
///    Idempotent.
/// 5. [`Drop`] calls `shutdown` so the thread is reaped on test
///    cleanup.
///
/// Cheaply cloneable as `Arc<CommunityRefreshScheduler>`.
pub struct CommunityRefreshScheduler {
    inner: Arc<SchedulerInner>,
    /// Join handle for the scheduler thread. Taken in
    /// [`Self::shutdown`]. `Mutex<Option>` lets shutdown be called
    /// from any thread and stay idempotent (the second call sees
    /// `None`).
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl CommunityRefreshScheduler {
    /// Start the scheduler. Spawns the dedicated OS thread.
    ///
    /// `hook` is invoked once per tenant per tick to resolve the
    /// per-tenant graph + membership index. Implementations are
    /// expected to be cheap (the scheduler holds no per-tenant
    /// state of its own).
    #[must_use]
    pub fn start(cfg: SchedulerConfig, hook: Arc<dyn RefreshHook>) -> Arc<Self> {
        Self::start_with_observer(cfg, hook, None)
    }

    /// Start the scheduler with an ADR-202 [`RefreshObserver`] wired.
    ///
    /// Identical to [`Self::start`] except every successful per-tenant
    /// refresh additionally notifies `observer` (see the trait docs
    /// for the exact contract). [`Self::start`] delegates here with
    /// `observer = None`, so existing callers are unaffected.
    ///
    /// The observer is a constructor parameter (not a
    /// [`SchedulerConfig`] field) because the config is deliberately
    /// `Copy` and an `Arc<dyn …>` field would break that; and not a
    /// post-`start` setter because the scheduler thread is already
    /// running by the time `start` returns (a setter would need extra
    /// synchronisation for no consumer benefit).
    #[must_use]
    pub fn start_with_observer(
        cfg: SchedulerConfig,
        hook: Arc<dyn RefreshHook>,
        observer: Option<Arc<dyn RefreshObserver>>,
    ) -> Arc<Self> {
        let inner = Arc::new(SchedulerInner {
            registered: Mutex::new(BTreeSet::new()),
            pending: Mutex::new(VecDeque::new()),
            wakeup_lock: Mutex::new(()),
            wakeup_cv: Condvar::new(),
            shutdown: AtomicBool::new(false),
            tick_in_progress: AtomicBool::new(false),
            tick_serialization: Mutex::new(()),
            total_ticks: AtomicU64::new(0),
            total_refresh_failures: AtomicU64::new(0),
            total_soft_skips: AtomicU64::new(0),
            cfg,
            hook,
            observer,
            // ADR-041 §D-3b: install LSNs start at the
            // configured `initial_install_lsn` (default 1).
            // Tests that manually pre-populate at low LSNs raise
            // this so the scheduler's allocator starts above the
            // pre-populates (preserving the strict-monotonic
            // install_level invariant).
            next_install_lsn: AtomicU64::new(cfg.initial_install_lsn.raw()),
        });

        let thread_inner = Arc::clone(&inner);
        let thread = thread::Builder::new()
            .name("arcgraph-community-refresh".to_owned())
            .spawn(move || run_scheduler(thread_inner))
            .expect("spawn arcgraph-community-refresh thread");

        info!(
            observer_wired = inner.observer.is_some(),
            "ADR-040 D-7: CommunityRefreshScheduler started (interval={:?})", cfg.interval,
        );

        Arc::new(Self {
            inner,
            thread: Mutex::new(Some(thread)),
        })
    }

    /// Register a tenant for daily refresh. No-op if already
    /// registered. Wakes the scheduler thread so the tenant is
    /// included in the next tick.
    pub fn register(&self, tenant: TenantId) {
        let inserted = {
            let mut tenants = self.inner.registered.lock();
            tenants.insert(tenant)
        };
        if inserted {
            debug!(
                "CommunityRefreshScheduler: registered tenant {:?}",
                tenant.raw(),
            );
        }
        self.wake();
    }

    /// Remove a tenant from the registered set. No-op if not
    /// registered.
    pub fn unregister(&self, tenant: TenantId) {
        let removed = {
            let mut tenants = self.inner.registered.lock();
            tenants.remove(&tenant)
        };
        if removed {
            debug!(
                "CommunityRefreshScheduler: unregistered tenant {:?}",
                tenant.raw(),
            );
        }
        self.wake();
    }

    /// Test-only forced tick. Snapshots the registered set into
    /// pending and runs one refresh sweep synchronously on the
    /// calling thread.
    ///
    /// Production code should not call this; it bypasses the
    /// dedicated thread. Useful for integration tests asserting
    /// end-to-end refresh semantics. Refresh hooks and observers must not
    /// re-enter this method while a tick is running; the serialization mutex
    /// is deliberately non-reentrant.
    pub fn tick(&self) {
        run_one_tick(&self.inner);
    }

    /// Shutdown the scheduler. Sets the shutdown flag, wakes the
    /// thread, joins it. Idempotent — subsequent calls are no-ops.
    pub fn shutdown(&self) {
        if self.inner.shutdown.swap(true, Ordering::AcqRel) {
            // Already shut down; nothing to do.
            return;
        }
        self.wake();
        let handle_opt = {
            let mut guard = self.thread.lock();
            guard.take()
        };
        if let Some(handle) = handle_opt {
            match handle.join() {
                Ok(()) => {
                    info!("CommunityRefreshScheduler: thread joined cleanly");
                }
                Err(panic_payload) => {
                    // Per ADR-040 §D-7 the scheduler is designed to
                    // contain hook panics via `catch_unwind`. A panic
                    // that propagates here is therefore a scheduler
                    // bug; we log and continue (we own no durability
                    // contract, so abort-on-panic is unnecessary).
                    error!(
                        "CommunityRefreshScheduler thread panicked: {:?}",
                        panic_payload,
                    );
                }
            }
        }
    }

    /// Snapshot of the scheduler's health.
    #[must_use]
    pub fn health_check(&self) -> SchedulerHealth {
        let registered_tenants = self.inner.registered.lock().len();
        SchedulerHealth {
            registered_tenants,
            last_tick_completed: !self.inner.tick_in_progress.load(Ordering::Acquire),
            total_ticks: self.inner.total_ticks.load(Ordering::Acquire),
            total_refresh_failures: self.inner.total_refresh_failures.load(Ordering::Acquire),
            total_soft_skips: self.inner.total_soft_skips.load(Ordering::Acquire),
            shut_down: self.inner.shutdown.load(Ordering::Acquire),
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
}

impl Drop for CommunityRefreshScheduler {
    fn drop(&mut self) {
        // Best-effort shutdown so we don't leak the thread on test
        // cleanup. Idempotent — safe even if the caller already
        // called `shutdown()`.
        self.shutdown();
    }
}

// ─────────────────────────────────────────────────────────────────
// Thread loop
// ─────────────────────────────────────────────────────────────────

/// Scheduler main loop. Runs on the dedicated
/// `arcgraph-community-refresh` thread.
///
/// Cycle:
/// 1. Sleep for `cfg.interval` (or until woken by register /
///    unregister / shutdown).
/// 2. Check shutdown — if set, exit.
/// 3. Run one tick.
/// 4. Loop.
fn run_scheduler(inner: Arc<SchedulerInner>) {
    loop {
        // §1. Sleep on condvar for `interval`.
        //
        // ORDERING (Mesa-semantics, lost-wakeup avoidance): the
        // shutdown check MUST happen INSIDE the `wakeup_lock`
        // critical section, BEFORE entering `wait_for`. Otherwise a
        // `shutdown()` that wins the race between thread-spawn and
        // first `wait_for` (it sets `shutdown=true`, then `wake()`
        // notifies an empty waiter list) is lost — the scheduler
        // would then enter `wait_for` and sleep the full interval.
        // `shutdown()` performs an `AcqRel` swap on the flag BEFORE
        // its `wake()` acquires this same lock, so any thread that
        // observes the lock via the same Acquire also observes the
        // flag set; checking the flag while holding the lock makes
        // the "release-lock-then-wait" atomic from this thread's
        // perspective. We re-check after `wait_for` returns (under
        // the re-acquired lock) so a `shutdown()` issued during the
        // wait is also caught before we drop the guard and tick.
        let mut guard = inner.wakeup_lock.lock();
        if inner.shutdown.load(Ordering::Acquire) {
            info!("CommunityRefreshScheduler: shutdown flag observed; exiting loop");
            return;
        }
        // `wait_for` returns on timeout OR on notify. We only tick
        // when the timeout fires (the natural cadence). A notify is
        // either a shutdown signal (handled below) or a
        // register/unregister wake — the latter merely informs the
        // tenant set; the next natural-cadence tick picks up the
        // change. Spurious wakes (rare under parking_lot, but
        // permitted by the API) likewise do not trigger a tick.
        let wait_res = inner.wakeup_cv.wait_for(&mut guard, inner.cfg.interval);

        // §2. Shutdown check (under the re-acquired lock).
        if inner.shutdown.load(Ordering::Acquire) {
            info!("CommunityRefreshScheduler: shutdown flag observed; exiting loop");
            return;
        }
        // Drop the guard before the (potentially long-running) tick
        // so concurrent register/unregister/wake calls don't block
        // on the wakeup lock for the tick's duration.
        drop(guard);

        // §3. Tick — only on the natural-cadence timeout. Notify
        //     paths (register/unregister/shutdown) loop back without
        //     ticking; shutdown is caught above, the others let the
        //     next interval tick observe the updated tenant set.
        if wait_res.timed_out() {
            run_one_tick(&inner);
        }
    }
}

/// Run a single scheduler tick. Snapshots the registered set into
/// the pending queue and drains it, calling the hook + GVE-Leiden
/// for each tenant. Per-tenant failures log and continue; one
/// tenant's failure does NOT stop the tick.
///
/// Exposed via [`CommunityRefreshScheduler::tick`] so integration
/// tests can exercise the tick logic without sleeping.
fn run_one_tick(inner: &SchedulerInner) {
    // `tick()` is synchronous on its caller, but it can overlap the dedicated
    // scheduler thread's natural-cadence tick. Serialize both entry points so
    // install-LSN reservation order is necessarily installation order and one
    // tick cannot clear or drain another tick's shared pending queue.
    let _tick_guard = inner.tick_serialization.lock();
    inner.tick_in_progress.store(true, Ordering::Release);
    inner.total_ticks.fetch_add(1, Ordering::Relaxed);

    let tick_start = Instant::now();

    // Snapshot the registered set into the pending queue. We
    // snapshot first so a `register` mid-tick doesn't perturb the
    // current sweep.
    {
        let registered = inner.registered.lock();
        let mut pending = inner.pending.lock();
        pending.clear();
        pending.extend(registered.iter().copied());
    }

    let pending_count = inner.pending.lock().len();
    info!(
        "CommunityRefreshScheduler: tick start ({} tenants pending)",
        pending_count,
    );

    // Drain the pending queue one tenant at a time.
    loop {
        let next = {
            let mut pending = inner.pending.lock();
            pending.pop_front()
        };
        let tenant = match next {
            Some(t) => t,
            None => break,
        };

        refresh_one_tenant(inner, tenant);
    }

    let elapsed = tick_start.elapsed();
    if elapsed > inner.cfg.max_tick_duration {
        warn!(
            "CommunityRefreshScheduler: tick exceeded max_tick_duration ({:?} > {:?})",
            elapsed, inner.cfg.max_tick_duration,
        );
    }

    info!(
        "CommunityRefreshScheduler: tick done ({} tenants, elapsed={:?})",
        pending_count, elapsed,
    );

    inner.tick_in_progress.store(false, Ordering::Release);
}

/// Refresh a single tenant. Resolves the hook, runs GVE-Leiden,
/// installs the result into the membership index. Catches hook
/// panics so one bad tenant doesn't take down the scheduler.
///
/// Logging contract:
/// - `tracing::debug!` on soft-skip (hook returned `None`)
/// - `tracing::info!` on success
/// - `tracing::error!` on hook panic / unexpected failure
fn refresh_one_tenant(inner: &SchedulerInner, tenant: TenantId) {
    // Catch hook panics. The hook is caller-supplied code, so a
    // bug there must not kill the scheduler thread (which would
    // silently stop ALL future tenant refreshes). We use
    // `AssertUnwindSafe` because the only state we care about
    // post-panic is the atomic counters, which are unwind-safe.
    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| do_refresh(inner, tenant)));

    match result {
        Ok(RefreshOutcome::Success) => {
            info!(
                "CommunityRefreshScheduler: tenant {:?} refreshed",
                tenant.raw(),
            );
            // ADR-202: notify the observability seam AFTER the
            // refresh result is installed (success arm only — never
            // on soft-skip / failure, so "last run" honestly means
            // "last installed result").
            //
            // Budget (PD-5): fires once per tenant per scheduler
            // cadence (default 24 h per ADR-040 §D-7) — ~0.01
            // calls/sec at 1 000 tenants. Per-call cost when wired:
            // one vtable dispatch + the impl's clock read + atomic
            // gauge store (≪ 1 µs, against a Leiden run measured in
            // seconds). When unwired: one nullable-ptr check.
            //
            // The notify runs OUTSIDE the refresh `catch_unwind`
            // (the refresh already succeeded) but inside its OWN
            // `catch_unwind`: a panicking observer must neither kill
            // the scheduler thread nor mislabel the succeeded
            // refresh as a failure (the health counters feed the
            // §10.3 runbook — corrupting them to report a metrics
            // bug would be a worse lie than the panic). Same
            // per-tenant panic-isolation discipline as the hook.
            if let Some(observer) = &inner.observer {
                let notify = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    observer.record_refresh_success(tenant);
                }));
                if let Err(panic_payload) = notify {
                    error!(
                        "CommunityRefreshScheduler: RefreshObserver panicked for tenant {:?} \
                         (refresh itself succeeded; scheduler continues): {:?}",
                        tenant.raw(),
                        panic_payload,
                    );
                }
            }
        }
        Ok(RefreshOutcome::SoftSkip) => {
            debug!(
                "CommunityRefreshScheduler: tenant {:?} soft-skipped (hook returned None)",
                tenant.raw(),
            );
            inner.total_soft_skips.fetch_add(1, Ordering::Relaxed);
        }
        Err(panic_payload) => {
            error!(
                "CommunityRefreshScheduler: tenant {:?} hook panicked: {:?}",
                tenant.raw(),
                panic_payload,
            );
            inner.total_refresh_failures.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Outcome of a single tenant refresh, reported back through
/// `catch_unwind`.
enum RefreshOutcome {
    Success,
    SoftSkip,
}

/// Do the actual refresh: hook → GveLeiden::run → install_into.
/// Separated out so [`refresh_one_tenant`] can wrap it in
/// `catch_unwind`.
///
/// Allocates a fresh `install_lsn` per ADR-041 §D-3b so the new
/// snapshot tags into the membership-index history at a unique
/// monotonic point.
fn do_refresh(inner: &SchedulerInner, tenant: TenantId) -> RefreshOutcome {
    let inputs = match inner.hook.resolve(tenant) {
        Some(i) => i,
        None => return RefreshOutcome::SoftSkip,
    };

    // Allocate the install LSN BEFORE the heavy Leiden run so a
    // mid-tick observer (a `current_lsn()` consult by some other
    // crate, hypothetically) sees the LSN already reserved. At
    // v1.0 the scheduler's allocator is local; v1.1 will replace
    // this with a `TransactionManager::allocate_install_lsn()`
    // call so the scheduler shares the storage allocator.
    let install_lsn = Lsn::new(inner.next_install_lsn.fetch_add(1, Ordering::AcqRel));

    // ADR-040 amendment-05: deref the Arcs for the borrow-flavoured
    // GveLeiden API. The Arcs (and their cached refs in the hook's
    // diagnostic cache, if any) survive end of fn drop; the
    // scheduler holds no cross-tick references to per-tick state.
    let result = GveLeiden::run(&inputs.graph, inputs.params);
    GveLeiden::install_into(
        &result,
        &inputs.index,
        tenant,
        install_lsn,
        inputs.n_skip_prefix,
    );
    RefreshOutcome::Success
}

// ─────────────────────────────────────────────────────────────────
// Tests — pin scheduler MECHANICS only. End-to-end scheduler+graph
// integration tests live in `tests/scheduler_integration.rs` (Wave 2).
// ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    /// Quick test config: 1-hour interval (longer than any test
    /// runs, so the scheduler thread never naturally ticks; tests
    /// drive ticks via `tick()`).
    fn test_cfg() -> SchedulerConfig {
        SchedulerConfig {
            interval: Duration::from_secs(3600),
            max_tick_duration: Duration::from_secs(60),
            initial_install_lsn: Lsn::new(1),
        }
    }

    /// Stub hook that records each call. Always returns `None` so
    /// we don't exercise GveLeiden inside the unit tests; that's
    /// pinned by `leiden_static_correctness.rs` and the integration
    /// suite.
    struct CountingHook {
        calls: StdMutex<Vec<TenantId>>,
    }

    impl CountingHook {
        fn new() -> Self {
            Self {
                calls: StdMutex::new(Vec::new()),
            }
        }

        fn calls(&self) -> Vec<TenantId> {
            self.calls
                .lock()
                .expect("counting hook lock poisoned (test bug)")
                .clone()
        }
    }

    impl RefreshHook for CountingHook {
        fn resolve(&self, tenant: TenantId) -> Option<OwnedRefreshInputs> {
            self.calls
                .lock()
                .expect("counting hook lock poisoned (test bug)")
                .push(tenant);
            // Soft-skip: returning None lets us test the
            // scheduler's tenant-iteration mechanics without
            // owning a Graph + index per call.
            None
        }
    }

    /// Hook that panics on every call. Exercises the
    /// `catch_unwind` path.
    struct PanickingHook;

    impl RefreshHook for PanickingHook {
        fn resolve(&self, _tenant: TenantId) -> Option<OwnedRefreshInputs> {
            panic!("intentional test panic");
        }
    }

    // ─── Lifecycle ─────────────────────────────────────────────

    /// Pins: scheduler starts, can be shut down, shutdown is
    /// idempotent across multiple calls AND across `Drop`.
    #[test]
    fn start_then_shutdown() {
        let hook: Arc<dyn RefreshHook> = Arc::new(CountingHook::new());
        let sched = CommunityRefreshScheduler::start(test_cfg(), hook);
        let h0 = sched.health_check();
        assert_eq!(h0.registered_tenants, 0);
        assert!(!h0.shut_down);
        assert!(h0.last_tick_completed);

        sched.shutdown();
        let h1 = sched.health_check();
        assert!(h1.shut_down);

        // Idempotent: second shutdown is a no-op.
        sched.shutdown();
        // Drop runs shutdown a third time — also a no-op.
    }

    // ─── Register / unregister ─────────────────────────────────

    /// Pins: register + unregister update the registered count
    /// without firing a tick (the scheduler's interval is 1h, so
    /// the thread never ticks naturally during the test).
    #[test]
    fn register_unregister() {
        let hook: Arc<dyn RefreshHook> = Arc::new(CountingHook::new());
        let sched = CommunityRefreshScheduler::start(test_cfg(), Arc::clone(&hook));

        sched.register(TenantId::DEFAULT);
        sched.register(TenantId::new(100));
        assert_eq!(sched.health_check().registered_tenants, 2);

        // Re-registering same tenant is a no-op count-wise.
        sched.register(TenantId::DEFAULT);
        assert_eq!(sched.health_check().registered_tenants, 2);

        sched.unregister(TenantId::DEFAULT);
        assert_eq!(sched.health_check().registered_tenants, 1);

        // Unregister of unknown tenant is a no-op.
        sched.unregister(TenantId::new(9999));
        assert_eq!(sched.health_check().registered_tenants, 1);

        sched.shutdown();
    }

    // ─── Tick semantics ────────────────────────────────────────

    /// Pins: forced tick with no registered tenants completes
    /// successfully and advances `total_ticks`.
    #[test]
    fn tick_with_zero_tenants() {
        let hook: Arc<dyn RefreshHook> = Arc::new(CountingHook::new());
        let sched = CommunityRefreshScheduler::start(test_cfg(), Arc::clone(&hook));

        let h_before = sched.health_check();
        assert_eq!(h_before.total_ticks, 0);

        sched.tick();

        let h_after = sched.health_check();
        assert_eq!(h_after.total_ticks, 1);
        assert_eq!(h_after.total_refresh_failures, 0);
        assert_eq!(h_after.total_soft_skips, 0);
        assert!(h_after.last_tick_completed);

        sched.shutdown();
    }

    /// Pins: forced tick with one registered tenant invokes the
    /// hook once for that tenant (and the soft-skip path is
    /// exercised since the test hook returns `None`).
    #[test]
    fn tick_drains_pending() {
        let counting = Arc::new(CountingHook::new());
        let hook: Arc<dyn RefreshHook> = Arc::clone(&counting) as Arc<dyn RefreshHook>;
        let sched = CommunityRefreshScheduler::start(test_cfg(), hook);

        sched.register(TenantId::DEFAULT);
        sched.register(TenantId::new(100));

        sched.tick();

        let calls = counting.calls();
        assert_eq!(calls.len(), 2, "hook should be called once per tenant");
        // BTreeSet iteration order is sorted by the inner u64,
        // so DEFAULT (1) precedes 100.
        assert_eq!(calls[0], TenantId::DEFAULT);
        assert_eq!(calls[1], TenantId::new(100));

        let h = sched.health_check();
        assert_eq!(h.total_ticks, 1);
        // CountingHook returns None → soft-skip, not failure.
        assert_eq!(h.total_refresh_failures, 0);
        assert_eq!(h.total_soft_skips, 2);

        sched.shutdown();
    }

    /// Pins: a panicking hook does NOT kill the scheduler thread
    /// and DOES increment `total_refresh_failures`. One tenant's
    /// panic must not stop the tick from continuing to subsequent
    /// tenants.
    #[test]
    fn tick_continues_on_hook_panic() {
        let hook: Arc<dyn RefreshHook> = Arc::new(PanickingHook);
        let sched = CommunityRefreshScheduler::start(test_cfg(), hook);

        sched.register(TenantId::DEFAULT);
        sched.register(TenantId::new(100));
        sched.register(TenantId::new(200));

        sched.tick();

        let h = sched.health_check();
        assert_eq!(h.total_ticks, 1);
        // All three tenants panicked → all three count as failures.
        assert_eq!(
            h.total_refresh_failures, 3,
            "panicking hook contributes one failure per tenant"
        );
        assert_eq!(h.total_soft_skips, 0);
        assert!(
            h.last_tick_completed,
            "tick must complete even when every hook panics"
        );

        sched.shutdown();
    }

    /// Pins: shutdown joins the thread cleanly, and a subsequent
    /// `Drop` does not double-join (idempotency at the
    /// shutdown/drop boundary).
    #[test]
    fn shutdown_then_drop_is_idempotent() {
        let hook: Arc<dyn RefreshHook> = Arc::new(CountingHook::new());
        let sched = CommunityRefreshScheduler::start(test_cfg(), hook);
        sched.shutdown();
        // Dropping `sched` here calls shutdown again; the
        // implementation must not panic. Implicit at end of scope.
    }

    // ─── Health snapshot integrity ─────────────────────────────

    /// Pins: health_check returns a self-consistent snapshot —
    /// counters move together, `last_tick_completed` reflects the
    /// post-tick state.
    #[test]
    fn health_check_reflects_state() {
        let counting = Arc::new(CountingHook::new());
        let hook: Arc<dyn RefreshHook> = Arc::clone(&counting) as Arc<dyn RefreshHook>;
        let sched = CommunityRefreshScheduler::start(test_cfg(), hook);

        sched.register(TenantId::DEFAULT);

        let h0 = sched.health_check();
        assert_eq!(h0.total_ticks, 0);
        assert_eq!(h0.total_refresh_failures, 0);
        assert_eq!(h0.total_soft_skips, 0);
        assert_eq!(h0.registered_tenants, 1);
        assert!(h0.last_tick_completed);
        assert!(!h0.shut_down);

        sched.tick();

        let h1 = sched.health_check();
        assert_eq!(h1.total_ticks, 1);
        assert_eq!(h1.total_soft_skips, 1); // CountingHook returns None
        assert!(h1.last_tick_completed);

        sched.shutdown();
        let h2 = sched.health_check();
        assert!(h2.shut_down);
    }

    // ─── ADR-202 RefreshObserver seam ──────────────────────────

    /// Hook that resolves EVERY tenant to a real (tiny) graph +
    /// shared membership index, so `do_refresh` reaches the
    /// `Success` outcome and the ADR-202 observer notify fires.
    struct SuccessHook {
        graph: Arc<Graph>,
        index: Arc<BTreeMembershipIndex>,
    }

    impl SuccessHook {
        fn new() -> Self {
            // Triangle: one community; enough for GveLeiden to run.
            let graph = Arc::new(Graph::from_edges_undirected(
                3,
                &[(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)],
            ));
            Self {
                graph,
                index: Arc::new(BTreeMembershipIndex::new()),
            }
        }
    }

    impl RefreshHook for SuccessHook {
        fn resolve(&self, _tenant: TenantId) -> Option<OwnedRefreshInputs> {
            Some(OwnedRefreshInputs {
                graph: Arc::clone(&self.graph),
                index: Arc::clone(&self.index),
                params: LeidenParams::default(),
                n_skip_prefix: 0,
            })
        }
    }

    /// ADR-202 test observer: records every notified tenant.
    #[derive(Debug)]
    struct CountingObserver {
        calls: StdMutex<Vec<TenantId>>,
    }

    impl CountingObserver {
        fn new() -> Self {
            Self {
                calls: StdMutex::new(Vec::new()),
            }
        }

        fn calls(&self) -> Vec<TenantId> {
            self.calls
                .lock()
                .expect("counting observer lock poisoned (test bug)")
                .clone()
        }
    }

    impl RefreshObserver for CountingObserver {
        fn record_refresh_success(&self, tenant: TenantId) {
            self.calls
                .lock()
                .expect("counting observer lock poisoned (test bug)")
                .push(tenant);
        }
    }

    /// ADR-202 test observer that panics on every call —
    /// exercises the D-5 panic-containment contract.
    #[derive(Debug)]
    struct PanickingObserver;

    impl RefreshObserver for PanickingObserver {
        fn record_refresh_success(&self, _tenant: TenantId) {
            panic!("intentional test observer panic");
        }
    }

    /// Pins ADR-202 D-1/D-5: the observer fires EXACTLY once per
    /// successful per-tenant refresh, with the right tenant, on
    /// every tick.
    #[test]
    fn observer_fires_once_per_successful_refresh() {
        let observer = Arc::new(CountingObserver::new());
        let sched = CommunityRefreshScheduler::start_with_observer(
            test_cfg(),
            Arc::new(SuccessHook::new()),
            Some(Arc::clone(&observer) as Arc<dyn RefreshObserver>),
        );

        sched.register(TenantId::DEFAULT);
        sched.register(TenantId::new(100));

        sched.tick();
        // Exact oracle: one call per tenant, BTreeSet order.
        assert_eq!(
            observer.calls(),
            vec![TenantId::DEFAULT, TenantId::new(100)],
            "observer must fire once per successful refresh"
        );

        sched.tick();
        assert_eq!(
            observer.calls().len(),
            4,
            "second tick notifies both tenants again"
        );

        let h = sched.health_check();
        assert_eq!(h.total_refresh_failures, 0);
        assert_eq!(h.total_soft_skips, 0);

        sched.shutdown();
    }

    /// Pins ADR-202 D-5: soft-skip (hook returned `None`) does NOT
    /// notify the observer — "last run" means "last INSTALLED
    /// result", and a skip installs nothing.
    #[test]
    fn observer_not_called_on_soft_skip() {
        let observer = Arc::new(CountingObserver::new());
        let sched = CommunityRefreshScheduler::start_with_observer(
            test_cfg(),
            Arc::new(CountingHook::new()), // always None → soft-skip
            Some(Arc::clone(&observer) as Arc<dyn RefreshObserver>),
        );

        sched.register(TenantId::DEFAULT);
        sched.tick();

        assert!(
            observer.calls().is_empty(),
            "soft-skip must not report a refresh success"
        );
        assert_eq!(sched.health_check().total_soft_skips, 1);

        sched.shutdown();
    }

    /// Pins ADR-202 D-5: a failed (panicked) refresh does NOT
    /// notify the observer.
    #[test]
    fn observer_not_called_on_hook_panic() {
        let observer = Arc::new(CountingObserver::new());
        let sched = CommunityRefreshScheduler::start_with_observer(
            test_cfg(),
            Arc::new(PanickingHook),
            Some(Arc::clone(&observer) as Arc<dyn RefreshObserver>),
        );

        sched.register(TenantId::DEFAULT);
        sched.tick();

        assert!(
            observer.calls().is_empty(),
            "failed refresh must not report a refresh success"
        );
        assert_eq!(sched.health_check().total_refresh_failures, 1);

        sched.shutdown();
    }

    /// Pins ADR-202 D-5 panic containment: a panicking observer
    /// (a) does not kill the tick — subsequent tenants are still
    /// refreshed; (b) does not mislabel the succeeded refresh as a
    /// failure; (c) leaves the scheduler alive for future ticks.
    #[test]
    fn panicking_observer_is_contained_and_not_a_refresh_failure() {
        let sched = CommunityRefreshScheduler::start_with_observer(
            test_cfg(),
            Arc::new(SuccessHook::new()),
            Some(Arc::new(PanickingObserver) as Arc<dyn RefreshObserver>),
        );

        sched.register(TenantId::DEFAULT);
        sched.register(TenantId::new(100));

        sched.tick();

        let h = sched.health_check();
        assert_eq!(h.total_ticks, 1);
        assert_eq!(
            h.total_refresh_failures, 0,
            "observer panic must NOT count as a refresh failure (the refresh succeeded)"
        );
        assert!(h.last_tick_completed, "tick must run to completion");

        // Scheduler is still alive: a second tick works.
        sched.tick();
        assert_eq!(sched.health_check().total_ticks, 2);

        sched.shutdown();
    }
}
