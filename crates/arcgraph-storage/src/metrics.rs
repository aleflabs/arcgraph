//! W16γ M6-07 — cross-bounded-context metrics sink contract.
//!
//! Storage producers (WAL writer, buffer pool) emit observability
//! events through the single [`MetricsSink`] trait defined here. The
//! concrete implementation lives in `arcgraph-mcp::transport::metrics`
//! (where the Prometheus `Registry` lives); the dependency direction
//! mcp→storage means storage never imports mcp, preserving the bounded
//! contexts.
//!
//! # Why a trait (vs. direct dep)
//!
//! `arcgraph-mcp::MetricsRegistry` cannot be referenced from
//! `arcgraph-storage` because that would invert the dependency edge:
//! `arcgraph-storage → arcgraph-mcp` is forbidden. A
//! `dyn MetricsSink` lets the producer call into `MetricsRegistry`
//! without knowing the concrete type. Cost analysis:
//!
//! - `Option<Arc<dyn MetricsSink>>::is_none()` branch — 1 nullable-ptr
//!   check (~1 ns; predicted by the branch predictor as taken when
//!   metrics aren't wired, which is the default).
//! - `dyn MetricsSink::record_*` call — 1 vtable lookup + 1 atomic
//!   increment inside the prometheus crate's `IntCounter` (~5–10 ns).
//!
//! Buffer pool hot-path budget (buffer.rs:10–13): cache-hit pin is
//! 23 ns single-threaded. Adding 5–10 ns when metrics are wired
//! inflates the path by 22–43%. Acceptable for v1.0-α (metrics are
//! opt-in; operators paying the cost have actively asked for it).
//!
//! # Trait minimality (`feedback_avoid_speculative_scaffolding.md`)
//!
//! The trait has five methods. Each has at least one real producer
//! caller in the PR that introduced it (no register-and-defer):
//!
//! - `record_wal_write` (W16γ) — fires on every `WalCmd::Append` /
//!   `WalCmd::AppendAsync` accept + on `fire()` fsync failure.
//! - `observe_wal_fsync_ms` (W16γ) — fires per-`fire()` once the
//!   segment `fsync` returns (success or pre-abort fail).
//! - `record_storage_page` (W16γ) — fires on every buffer pool pin
//!   path (hit / miss / eviction).
//! - `record_hot_vertex_warning` (W28 #582, this slice) — fires on
//!   every CRUD TEL / reverse-TEL overflow-block allocation
//!   (`CrudStore::tel_append` / `tel_append_reverse`). design-v2 §10.2
//!   line 721. This wire is the no-op-trampoline fix for the metric
//!   that W17δ #313 *registered* but never *fired*.
//! - `record_query_plan_choice` (W28 #582, this slice) — fires once
//!   per executed query plan from the `arcgraph-query` `QueryEngine`
//!   plan-build path. design-v2 §10.2 line 723.
//!
//! Adding a sixth method requires either a real producer caller in the
//! same PR or an explicit ADR follow-up. Adding sink methods before
//! the producer code lands is exactly the speculative-scaffold
//! anti-pattern that prior waves' fix-ups have caught. (The §10.2 line
//! 724 `arcgraph_leiden_last_run_seconds` metric is NOT on this
//! trait — and never will be: its producer lives in
//! `arcgraph-community`, which has no `arcgraph-storage` dependency
//! and so cannot reach this trait. It ships through the ADR-202
//! community-resident seam instead:
//! `arcgraph-community::scheduler::RefreshObserver`, implemented by
//! the same `arcgraph-mcp` `MetricsRegistry` that implements this
//! trait. Each bounded context owns its seam; the registry implements
//! them all.)
//!
//! # Design provenance
//!
//! - design-v2 §10.2 lines 701–708 — observability inventory.
//! - design-v2 §3.4 — buffer pool no-mmap rationale.
//! - design-v2 §4.2 — WAL group-commit semantics.
//! - ADR-034 §Slice B — durability tier producer paths.

use std::fmt::Debug;

use arcgraph_core::TenantId;

// ---------------------------------------------------------------------
// Label enums
// ---------------------------------------------------------------------

/// Outcome label for `wal_writes_total{outcome}`.
///
/// Per ADR-034 §Slice B, the WAL writer accepts two append flavors —
/// T1 (sync) blocks until fsync; T3 (async) acks at enqueue and
/// durifies via a piggyback / scheduler. The `FsyncFail` variant
/// fires when a `fire()` group-commit fsync returns `Err`; per
/// ADR-034 §6.2 the writer aborts when async records are in the
/// failed batch, so `FsyncFail` is observed precisely once per
/// crash-causing fsync and never afterward.
///
/// `#[non_exhaustive]` under the strict public-contract policy — future variants
/// (e.g., a per-tenant outcome split, or a `Rollback` for non-D-7
/// recoveries) are additive and don't break existing match arms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum WalWriteOutcome {
    /// `WalHandle::append` (T1 / Strict) was accepted into the
    /// pending batch.
    T1Sync,
    /// `WalHandle::append_async` (T3 / Periodic) was accepted into
    /// the pending batch.
    T3Async,
    /// A `fire()` fsync attempt returned `Err`. Counted exactly once
    /// per failed fire; ADR-034 §6.2 abort eats the next-in-flight
    /// fire's metric so this is a load-bearing signal that fsync
    /// path degradation is occurring.
    FsyncFail,
}

impl WalWriteOutcome {
    /// String form used as the prometheus `outcome` label value.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::T1Sync => "t1_sync",
            Self::T3Async => "t3_async",
            Self::FsyncFail => "fsync_fail",
        }
    }
}

/// Kind label for `storage_pages_total{kind}`.
///
/// Maps to the three observable buffer-pool events on the pin path:
///
/// - `Hit` — `try_fast_pin_{read,write}` returned `Some(guard)`
///   without taking the slow-path load lock.
/// - `Miss` — `load_into_fresh_frame` was invoked because the page
///   was not currently mapped.
/// - `Eviction` — `load_into_fresh_frame` evicted an existing
///   mapping (the victim frame held a different page).
///
/// `Eviction` and `Miss` co-occur: a miss against a full pool ALWAYS
/// evicts. The two counters are kept separate because a miss against
/// a cold slot (initial pool warm-up) reports `Miss` but not
/// `Eviction`. Per buffer.rs:586 (`load_into_fresh_frame`), the
/// `old.is_some()` branch is the eviction signal.
///
/// `#[non_exhaustive]` under the strict public-contract policy — future page-event variants
/// are additive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum StoragePageKind {
    /// Cache hit — page already mapped, fast-path returned.
    Hit,
    /// Cache miss — `load_into_fresh_frame` invoked.
    Miss,
    /// Eviction — load displaced an existing mapping.
    Eviction,
}

impl StoragePageKind {
    /// String form used as the prometheus `kind` label value.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Hit => "hit",
            Self::Miss => "miss",
            Self::Eviction => "eviction",
        }
    }
}

/// `plan_type` label for `arcgraph_query_plan_choice{plan_type}`.
///
/// design-v2 §10.2 **line 723** literally cites the metric as
/// `arcgraph_query_plan_choice{plan_type}` with the parenthetical
/// taxonomy `(binary/wcoj/free_join)`. The three variants below are
/// the verbatim §10.2 label space:
///
/// - `Binary` — left-deep / bushy *binary* (pairwise) join plans. This
///   is the ONLY paradigm the v1.0-α query engine produces. Per
///   `crates/arcgraph-query/src/planner/enumeration/mod.rs:20`
///   ("binary joins at v1.0; bushy deferred to v1.1"), every plan the
///   v1.0 enumerator + `pick_join_algorithms` pass emit is a binary
///   join tree (the physical `HashJoin` / `MergeJoin` split is a
///   *sub*-distinction BELOW the §10.2 paradigm granularity, so it is
///   not exposed on this label).
/// - `Wcoj` — worst-case-optimal (multi-way) join plans. **Reserved
///   for v1.1+**; the WCOJ executor does not exist at v1.0-α, so this
///   variant is never emitted by the current planner. It ships now
///   because §10.2 line 723 names it and the producer-side classifier
///   (`arcgraph_query`) is the single forward-compat extension point.
/// - `FreeJoin` — free-join (Wang et al. SIGMOD 2023) hybrid plans.
///   **Reserved for v1.1+** (same rationale as `Wcoj`).
///
/// This is NOT speculative scaffolding per
/// `feedback_avoid_speculative_scaffolding.md`: the `Binary` variant
/// has a real producer caller in THIS PR (the `QueryEngine` execute
/// plan-build path), and the enum is the label space of a single
/// metric — the v1.1 variants are the cite-text-mandated label values,
/// not unused trait surface.
///
/// `#[non_exhaustive]` under the strict public-contract policy — a future `plan_type`
/// (e.g., a vector-index-scan plan family) is additive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum QueryPlanType {
    /// Binary (pairwise) join plan — the v1.0-α default + only paradigm.
    Binary,
    /// Worst-case-optimal (multi-way) join plan — reserved for v1.1+.
    Wcoj,
    /// Free-join hybrid plan — reserved for v1.1+.
    FreeJoin,
}

impl QueryPlanType {
    /// String form used as the prometheus `plan_type` label value.
    ///
    /// Values are the verbatim design-v2 §10.2 line 723 taxonomy
    /// `(binary/wcoj/free_join)`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Binary => "binary",
            Self::Wcoj => "wcoj",
            Self::FreeJoin => "free_join",
        }
    }
}

// ---------------------------------------------------------------------
// MetricsSink trait
// ---------------------------------------------------------------------

/// Storage-side observability emit point.
///
/// The trait is `Send + Sync + Debug + 'static` because storage
/// producers hold the sink across thread boundaries (WAL writer is
/// its own thread; buffer pool is shared across the work-stealing
/// pool). `Debug` lets producers log the sink presence in startup
/// tracing without conditional compilation.
///
/// Concrete impl: `arcgraph-mcp::transport::metrics::MetricsRegistry`.
///
/// # When NOT to call through this trait
///
/// Producer code paths inside `arcgraph-mcp` (e.g., `serve_stdio` or
/// `serve_bolt_listener` setting `active_connections`) should call
/// the concrete `MetricsRegistry` directly — those producers already
/// know the concrete type and routing through `dyn MetricsSink`
/// would add unnecessary indirection.
pub trait MetricsSink: Send + Sync + Debug + 'static {
    /// Record one `wal_writes_total{outcome}` increment.
    ///
    /// Called by the WAL writer thread on every accepted append
    /// (`T1Sync` / `T3Async`) and on every failed fire fsync
    /// (`FsyncFail`).
    fn record_wal_write(&self, outcome: WalWriteOutcome);

    /// Observe one `wal_fsync_duration_ms` sample.
    ///
    /// Called by the WAL writer thread per `fire()` invocation,
    /// regardless of fire success or failure. `duration_ms` is the
    /// wall-clock duration of the `segment.fsync()` call (NOT the
    /// full fire including append + reply distribution).
    ///
    /// design-v2 §10.2 line 704 target: P99 ≤ 5 ms.
    fn observe_wal_fsync_ms(&self, duration_ms: f64);

    /// Record one `storage_pages_total{kind}` increment.
    ///
    /// Called by `BufferPool` on every pin path (hit / miss /
    /// eviction). Hot — see buffer.rs:10–13 budget.
    fn record_storage_page(&self, kind: StoragePageKind);

    /// Record one `arcgraph_hot_vertex_warnings_total{tenant}`
    /// increment (design-v2 §10.2 **line 721**).
    ///
    /// Called by the CRUD layer (`CrudStore::tel_append` /
    /// `tel_append_reverse`) on every TEL/reverse-TEL overflow-block
    /// allocation — the per-event signal that a vertex's (out- or
    /// in-) adjacency is approaching the supernode threshold
    /// (design-v2 §3.3). The metric is a *counter* keyed by `tenant`
    /// per the W17δ #313 decision (option (1): counter semantic +
    /// `{tenant}` cardinality). Note the §10.2 line 721 cite-text
    /// labels are `{vertex_id, ops_per_sec}`; the shipped metric uses
    /// `{tenant}` — a deliberate, already-shipped W17δ #313 deviation
    /// (per-vertex/ops-per-sec cardinality is unbounded; per-tenant is
    /// the bounded operational signal the §10.3 line 733 alert
    /// `rate(...[1m]) > 100` reads). This slice (Feature #582) FIRES
    /// the previously register-only metric — the no-op-trampoline fix
    /// per `feedback_noop_trampoline_anti_pattern.md`.
    fn record_hot_vertex_warning(&self, tenant: TenantId);

    /// Record one `arcgraph_query_plan_choice{plan_type}` increment
    /// (design-v2 §10.2 **line 723**).
    ///
    /// Called once per executed query plan by the `arcgraph-query`
    /// `QueryEngine` plan-build path. `plan_type` is the §10.2 line 723
    /// taxonomy `(binary/wcoj/free_join)` — see [`QueryPlanType`]. At
    /// v1.0-α the engine produces binary join plans exclusively, so the
    /// emitted value is always [`QueryPlanType::Binary`]; the wcoj /
    /// free_join values are the v1.1+ extension point.
    ///
    /// PD-7 bounded contexts: `arcgraph-query` depends ON
    /// `arcgraph-storage` (never the inverse), so the query producer
    /// calls this storage-resident trait through
    /// `Option<Arc<dyn MetricsSink>>` exactly as the WAL writer +
    /// buffer pool do. (The Leiden producer in `arcgraph-community`
    /// CANNOT — `arcgraph-community` has no `arcgraph-storage` edge —
    /// which is why `arcgraph_leiden_last_run_seconds`, §10.2 line 724,
    /// ships through the ADR-202 community-resident
    /// `RefreshObserver` seam instead of this trait; see the
    /// `arcgraph-mcp` metrics module's "§10.2 closure — Leiden
    /// freshness gauge (ADR-202)" section.)
    fn record_query_plan_choice(&self, plan_type: QueryPlanType);
}

// ---------------------------------------------------------------------
// Test-only counting sink (used by unit tests + property tests)
//
// Gated `#[cfg(test)]` per `feedback_avoid_speculative_scaffolding.md`:
// no production consumer of `CountingMetricsSink` exists today, so it
// is NOT part of the public API surface. Downstream test crates
// wanting a counting sink should either implement `MetricsSink`
// directly (3 methods) or motivate a `test-utils` cargo feature with
// the first real consumer.
// ---------------------------------------------------------------------

/// A `MetricsSink` impl that records every call into atomic counters.
///
/// Used as the test default when a producer wants to exercise the
/// `Some(_)` branch of `Option<Arc<dyn MetricsSink>>` without
/// requiring the prometheus-backed `MetricsRegistry` (which lives
/// upstream in `arcgraph-mcp`). The `AtomicU64` counters expose a
/// verification handle for tests that need to assert the producer
/// emitted the expected events.
#[cfg(test)]
#[derive(Debug, Default)]
pub(crate) struct CountingMetricsSink {
    /// `(T1Sync, T3Async, FsyncFail)` counters.
    pub(crate) wal_writes: [std::sync::atomic::AtomicU64; 3],
    /// Sum of `observe_wal_fsync_ms` observations × 1e6 (kept as u64
    /// for `AtomicU64`; divide by 1e6 to recover the milliseconds
    /// sum). Number of observations is in `wal_fsync_observations`.
    pub(crate) wal_fsync_ms_micros_sum: std::sync::atomic::AtomicU64,
    /// Number of `observe_wal_fsync_ms` calls.
    pub(crate) wal_fsync_observations: std::sync::atomic::AtomicU64,
    /// `(Hit, Miss, Eviction)` counters.
    pub(crate) storage_pages: [std::sync::atomic::AtomicU64; 3],
    /// Total `record_hot_vertex_warning` calls (tenant-agnostic count;
    /// per-tenant attribution is the `MetricsRegistry`'s job — the
    /// test fixture only needs the no-op-trampoline guard count).
    pub(crate) hot_vertex_warnings: std::sync::atomic::AtomicU64,
    /// `(Binary, Wcoj, FreeJoin)` `record_query_plan_choice` counters.
    pub(crate) query_plan_choices: [std::sync::atomic::AtomicU64; 3],
}

#[cfg(test)]
impl CountingMetricsSink {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn wal_writes_count(&self, outcome: WalWriteOutcome) -> u64 {
        let idx = match outcome {
            WalWriteOutcome::T1Sync => 0,
            WalWriteOutcome::T3Async => 1,
            WalWriteOutcome::FsyncFail => 2,
        };
        self.wal_writes[idx].load(std::sync::atomic::Ordering::Acquire)
    }

    pub(crate) fn storage_pages_count(&self, kind: StoragePageKind) -> u64 {
        let idx = match kind {
            StoragePageKind::Hit => 0,
            StoragePageKind::Miss => 1,
            StoragePageKind::Eviction => 2,
        };
        self.storage_pages[idx].load(std::sync::atomic::Ordering::Acquire)
    }

    pub(crate) fn wal_fsync_observation_count(&self) -> u64 {
        self.wal_fsync_observations
            .load(std::sync::atomic::Ordering::Acquire)
    }

    pub(crate) fn hot_vertex_warning_count(&self) -> u64 {
        self.hot_vertex_warnings
            .load(std::sync::atomic::Ordering::Acquire)
    }

    pub(crate) fn query_plan_choice_count(&self, plan_type: QueryPlanType) -> u64 {
        let idx = match plan_type {
            QueryPlanType::Binary => 0,
            QueryPlanType::Wcoj => 1,
            QueryPlanType::FreeJoin => 2,
        };
        self.query_plan_choices[idx].load(std::sync::atomic::Ordering::Acquire)
    }
}

#[cfg(test)]
impl MetricsSink for CountingMetricsSink {
    fn record_wal_write(&self, outcome: WalWriteOutcome) {
        let idx = match outcome {
            WalWriteOutcome::T1Sync => 0,
            WalWriteOutcome::T3Async => 1,
            WalWriteOutcome::FsyncFail => 2,
        };
        self.wal_writes[idx].fetch_add(1, std::sync::atomic::Ordering::AcqRel);
    }

    fn observe_wal_fsync_ms(&self, duration_ms: f64) {
        // 1e6 scaling factor: nanoseconds-precision summing without
        // losing sub-µs ticks on rapid fsync paths.
        let micros = (duration_ms * 1_000.0).round().max(0.0) as u64;
        self.wal_fsync_ms_micros_sum
            .fetch_add(micros, std::sync::atomic::Ordering::AcqRel);
        self.wal_fsync_observations
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
    }

    fn record_storage_page(&self, kind: StoragePageKind) {
        let idx = match kind {
            StoragePageKind::Hit => 0,
            StoragePageKind::Miss => 1,
            StoragePageKind::Eviction => 2,
        };
        self.storage_pages[idx].fetch_add(1, std::sync::atomic::Ordering::AcqRel);
    }

    fn record_hot_vertex_warning(&self, _tenant: TenantId) {
        self.hot_vertex_warnings
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
    }

    fn record_query_plan_choice(&self, plan_type: QueryPlanType) {
        let idx = match plan_type {
            QueryPlanType::Binary => 0,
            QueryPlanType::Wcoj => 1,
            QueryPlanType::FreeJoin => 2,
        };
        self.query_plan_choices[idx].fetch_add(1, std::sync::atomic::Ordering::AcqRel);
    }
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn wal_write_outcome_label_strings_are_canonical() {
        assert_eq!(WalWriteOutcome::T1Sync.as_str(), "t1_sync");
        assert_eq!(WalWriteOutcome::T3Async.as_str(), "t3_async");
        assert_eq!(WalWriteOutcome::FsyncFail.as_str(), "fsync_fail");
    }

    #[test]
    fn storage_page_kind_label_strings_are_canonical() {
        assert_eq!(StoragePageKind::Hit.as_str(), "hit");
        assert_eq!(StoragePageKind::Miss.as_str(), "miss");
        assert_eq!(StoragePageKind::Eviction.as_str(), "eviction");
    }

    /// Pin: `QueryPlanType` label strings are the verbatim design-v2
    /// §10.2 line 723 taxonomy `(binary/wcoj/free_join)`. Drift here
    /// silently breaks the Grafana panel label match.
    #[test]
    fn query_plan_type_label_strings_are_canonical() {
        assert_eq!(QueryPlanType::Binary.as_str(), "binary");
        assert_eq!(QueryPlanType::Wcoj.as_str(), "wcoj");
        assert_eq!(QueryPlanType::FreeJoin.as_str(), "free_join");
    }

    #[test]
    fn counting_sink_records_hot_vertex_and_plan_choice() {
        let sink = CountingMetricsSink::new();
        sink.record_hot_vertex_warning(TenantId::new(1));
        sink.record_hot_vertex_warning(TenantId::new(2));
        sink.record_hot_vertex_warning(TenantId::new(1));
        assert_eq!(sink.hot_vertex_warning_count(), 3);

        sink.record_query_plan_choice(QueryPlanType::Binary);
        sink.record_query_plan_choice(QueryPlanType::Binary);
        // Exact (==) oracles — deterministic counters.
        assert_eq!(sink.query_plan_choice_count(QueryPlanType::Binary), 2);
        assert_eq!(sink.query_plan_choice_count(QueryPlanType::Wcoj), 0);
        assert_eq!(sink.query_plan_choice_count(QueryPlanType::FreeJoin), 0);
    }

    #[test]
    fn counting_sink_records_per_outcome() {
        let sink = CountingMetricsSink::new();
        sink.record_wal_write(WalWriteOutcome::T1Sync);
        sink.record_wal_write(WalWriteOutcome::T1Sync);
        sink.record_wal_write(WalWriteOutcome::T3Async);
        sink.record_wal_write(WalWriteOutcome::FsyncFail);
        assert_eq!(sink.wal_writes_count(WalWriteOutcome::T1Sync), 2);
        assert_eq!(sink.wal_writes_count(WalWriteOutcome::T3Async), 1);
        assert_eq!(sink.wal_writes_count(WalWriteOutcome::FsyncFail), 1);
    }

    #[test]
    fn counting_sink_records_per_page_kind() {
        let sink = CountingMetricsSink::new();
        for _ in 0..5 {
            sink.record_storage_page(StoragePageKind::Hit);
        }
        sink.record_storage_page(StoragePageKind::Miss);
        sink.record_storage_page(StoragePageKind::Eviction);
        assert_eq!(sink.storage_pages_count(StoragePageKind::Hit), 5);
        assert_eq!(sink.storage_pages_count(StoragePageKind::Miss), 1);
        assert_eq!(sink.storage_pages_count(StoragePageKind::Eviction), 1);
    }

    #[test]
    fn counting_sink_observe_fsync_accumulates() {
        let sink = CountingMetricsSink::new();
        sink.observe_wal_fsync_ms(0.500); // 500 µs
        sink.observe_wal_fsync_ms(2.000); // 2 ms
        sink.observe_wal_fsync_ms(0.001); // 1 µs

        assert_eq!(sink.wal_fsync_observation_count(), 3);
        // Sum is 0.500 + 2.000 + 0.001 = 2.501 ms × 1000 µs/ms = 2501.
        let sum_micros = sink
            .wal_fsync_ms_micros_sum
            .load(std::sync::atomic::Ordering::Acquire);
        // Rounding: 0.500 → 500, 2.000 → 2000, 0.001 → 1; total 2501.
        assert_eq!(sum_micros, 2501);
    }

    #[test]
    fn counting_sink_is_send_sync_arc_dyn() {
        fn assert_send_sync<T: Send + Sync + 'static>() {}
        assert_send_sync::<CountingMetricsSink>();
        // The trait object must also be Send + Sync to thread through
        // producer configs (WAL writer thread + buffer pool's
        // potentially-shared pool).
        let sink: Arc<dyn MetricsSink> = Arc::new(CountingMetricsSink::new());
        sink.record_wal_write(WalWriteOutcome::T1Sync);
        sink.record_storage_page(StoragePageKind::Hit);
        sink.observe_wal_fsync_ms(1.0);
        sink.record_hot_vertex_warning(TenantId::new(1));
        sink.record_query_plan_choice(QueryPlanType::Binary);
    }
}
