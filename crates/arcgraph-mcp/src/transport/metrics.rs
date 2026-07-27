//! W15γ M6-06 (initial, PARTIAL) + W16γ M6-07 (producer-wire surface
//! landed; CLI-side operator observability forward-pinned to M6-08+):
//! Prometheus `/metrics` exporter (roadmap line 409 +
//! design-v2 §10.2).
//!
//! # Scope at v1.0-alpha
//!
//! At W16γ M6-07, this slice ships the **producer-wire surface**
//! for 5 of the 8 metrics named in design-v2 §10.2 (lines 701–708),
//! plus three operational supplements cross-cited by sister PR #307's
//! Grafana dashboards (`active_connections`, `wal_writes_total`,
//! `storage_pages_total`). "Producer wires" means: the storage
//! producers (WAL writer, buffer pool) and the mcp transports
//! (stdio, bolt, http) emit observability events through the
//! [`MetricsSink`](crate::transport::metrics) / `MetricsRegistry`
//! contract WHEN an operator threads a sink into construction.
//!
//! **Operator-observability closure is NOT in this slice.** The
//! v1.0-α `arcgraph` CLI binary constructs `WalConfig` /
//! `BufferPool` / `serve_*` with `metrics_sink: None` and never
//! instantiates a `MetricsRegistry`, so an operator running the
//! production binary today sees no data on these 5 new metrics. The
//! HTTP `/metrics` listener + `--metrics` flag + storage→registry
//! thread-through are forward-pinned to M6-08+ alongside the
//! `--config <toml>` schema and the HTTP cert/key wiring.
//!
//! W28 Feature #582 (ADR-045) wires two of the three cross-context
//! §10.2 metrics that prior slices forward-pinned: the hot-vertex
//! warnings counter (§10.2 line 721, producer in `arcgraph-storage`'s
//! CRUD TEL overflow path) and the query plan-choice counter (§10.2
//! line 723, producer in `arcgraph-query`'s `QueryEngine`). Both reach
//! the storage-resident `MetricsSink` because their producer crates
//! sit ABOVE `arcgraph-storage` in the dep graph (`arcgraph-query →
//! arcgraph-storage`; CRUD IS `arcgraph-storage`). The third —
//! `arcgraph_leiden_last_run_seconds` (§10.2 line 724) — ships via the
//! ADR-202 community-resident seam (`arcgraph-community::scheduler::`
//! `RefreshObserver`, implemented by this registry); see the
//! "§10.2 closure — Leiden freshness gauge (ADR-202)" section below.
//!
//! Cross-artifact drift detection (Grafana dashboard JSON +
//! Prometheus alerts.yml referencing only registered metric names)
//! is closure-tracked under issue #314 via sister PR #307's
//! `grafana_validation.rs` regression test.
//!
//! # Shipped metric inventory (W15γ + W16γ M6-07)
//!
//! Every metric name below appears verbatim in design-v2 §10.2
//! lines 701-708 OR is cross-cited as an operational supplement by
//! sister PR #307's Grafana dashboards:
//!
//! - `arcgraph_mcp_tool_invocations{tenant, tool, status}` — §10.2
//!   line 706. `IntCounterVec`. Incremented per JSON-RPC dispatch
//!   through the HTTP transport. `tool` = the JSON-RPC method name
//!   (e.g., `"graph.schema"`, `"graph.ingest"`); `status` ∈
//!   {`"ok"`, `"error"`}. The `tenant` label extends §10.2 per
//!   ADR-038 amendment-03 §TIER-2-c (per-tenant observability).
//! - `arcgraph_read_latency_ms{tenant, tool}` — §10.2 line 702.
//!   `HistogramVec` in **milliseconds** (per the `_ms` suffix in the
//!   §10.2 spec). Observed per read-class JSON-RPC dispatch.
//! - `arcgraph_write_latency_ms{tenant, tool}` — §10.2 line 701.
//!   `HistogramVec` in **milliseconds**. Observed per write-class
//!   JSON-RPC dispatch (today only `graph.ingest`). Splitting read
//!   vs write follows §10.2's two-metric inventory line 701-702 and
//!   the M5-12 rate-limit `OpClass` bucket alignment.
//! - `arcgraph_wal_fsync_duration_ms` — §10.2 line 704.
//!   `Histogram` in **milliseconds**. Observed once per WAL
//!   `fire()` group-commit (success OR pre-abort fail). Target P99
//!   ≤ 5 ms per §10.2 line 704. Wired via ADR-045
//!   `MetricsSink::observe_wal_fsync_ms` from `arcgraph-storage`.
//! - `arcgraph_wal_writes_total{outcome}` — operational counter
//!   (cross-cited by sister PR #307 Grafana). `IntCounterVec` keyed
//!   by ADR-034 §Slice B durability tier outcome:
//!   `outcome ∈ {"t1_sync", "t3_async", "fsync_fail"}`. Wired via
//!   ADR-045 `MetricsSink::record_wal_write`.
//! - `arcgraph_storage_pages_total{kind}` — operational counter
//!   (cross-cited by sister PR #307 Grafana). `IntCounterVec` keyed
//!   by buffer-pool event kind: `kind ∈ {"hit", "miss", "eviction"}`.
//!   The §10.2 line 703 `arcgraph_buffer_pool_hit_rate` signal is
//!   computed Grafana-side as PromQL
//!   `rate(arcgraph_storage_pages_total{kind="hit"}[5m]) /
//!    rate(arcgraph_storage_pages_total[5m])`. Wired via ADR-045
//!   `MetricsSink::record_storage_page`. Single counter pair (Hit +
//!   Miss + Eviction) is strictly stronger than a single
//!   pre-computed gauge because operators can compute rolling rate,
//!   exclude eviction-pressure, or split by recording-pool window.
//!
//!   **Operator note — slow-path Hit semantics (`buffer.rs` re-check):**
//!   under concurrent fault-in of the same cold page from N threads,
//!   the observed split is 1 Miss + (N-1) Hits, NOT N Misses. The
//!   (N-1) waiters take the slow path, acquire `load_lock` after the
//!   loader populated the page table, and their re-check succeeds
//!   without disk I/O — that's a real cache hit (the cache table
//!   served the lookup). The Hit-rate signal is therefore biased
//!   upward under concurrent cold-start. The ≥0.95 target is reached
//!   sooner than a strict "did the first lookup succeed" metric would
//!   suggest. A v1.1 follow-up may add `kind="slow_hit"` to let
//!   operators distinguish true fast-path hits from post-load_lock
//!   cache hits; v1.0 ships the documented semantics.
//! - `arcgraph_active_connections{transport}` — operational gauge
//!   (not in §10.2 directly; cited by sister PR #307 Grafana
//!   dashboards). `IntGaugeVec` keyed by transport. W16γ M6-07
//!   extends [`ConnectionTransport`] with the `Stdio` and `Bolt`
//!   variants matching the `serve_stdio` + `serve_bolt_listener`
//!   accept-loop wires (`http.rs:823 ActiveConnGuard` pattern
//!   mirrored). All three transports report through the same
//!   gauge.
//! - `arcgraph_leiden_last_run_seconds{tenant}` — §10.2 line 724
//!   (ADR-202). `IntGaugeVec`; Unix timestamp of the tenant's last
//!   successful community refresh. See the dedicated
//!   "§10.2 closure — Leiden freshness gauge (ADR-202)" section
//!   below.
//!
//! # Histogram emission vs §10.2 `{quantile}` cite-text
//!
//! design-v2 §10.2 lines 701-702 literally cite
//! `arcgraph_{write,read}_latency_ms{quantile}` — the `{quantile}` label
//! is Prometheus *summary* syntax (per-quantile label populated by the
//! collector). This implementation emits *histograms* instead — the
//! Prometheus text-exposition surface is `_bucket{le=...}` +
//! `_count` + `_sum` rows. The two surfaces are functionally
//! equivalent: a histogram lets the scraper compute any quantile via
//! PromQL `histogram_quantile(0.99, rate(arcgraph_read_latency_ms_bucket[5m]))`
//! whereas a summary fixes the per-quantile set at the producer side.
//! Histograms are the Prometheus best-practice for cross-instance
//! aggregation (you cannot average quantiles across replicas without
//! losing accuracy; histogram buckets aggregate exactly), so the
//! literal cite-text departure is intentional and tracked here.
//!
//! # §10.2 closure — Leiden freshness gauge (ADR-202)
//!
//! - `arcgraph_leiden_last_run_seconds{tenant}` — §10.2 **line 724**
//!   (community-detection freshness; §10.3 alert "Community detection
//!   freshness > 48h"). `IntGaugeVec` holding the **Unix timestamp
//!   (whole seconds) of the tenant's most recent successful community
//!   refresh**, so the shipped alert contract
//!   `time() - arcgraph_leiden_last_run_seconds > (48 * 3600)`
//!   (docs/grafana/alerts.yml `ArcGraphLeidenFreshnessStale`)
//!   evaluates correctly per-series. Was forward-pinned at W28 #582
//!   for a PD-7 bounded-context reason: the producer
//!   (`arcgraph-community::CommunityRefreshScheduler`, the
//!   `GveLeiden::run` + `install_into` site) sits BENEATH
//!   `arcgraph-storage` in the dep graph and cannot reach the
//!   storage-resident [`StorageMetricsSink`]. ADR-202 lands the
//!   anticipated community-resident seam: the scheduler notifies an
//!   `Option<Arc<dyn arcgraph_community::RefreshObserver>>` once per
//!   successful per-tenant refresh (success arm ONLY — soft-skips and
//!   failed/panicked refreshes never fire, so "last run" honestly
//!   means "last installed result"); this registry is the concrete
//!   impl and owns the clock read. Threading:
//!   `EngineConfig::refresh_observer` →
//!   `CommunityRefreshScheduler::start_with_observer`.
//!
//!   **Restart semantics (ADR-202 D-6):** the gauge is process-local;
//!   after a restart the `{tenant}` series are ABSENT until each
//!   tenant's first successful refresh in the new process (the alert
//!   expression evaluates to an empty vector — no claim until a real
//!   run happened). **Remaining operational gap (NOT this metric's):**
//!   the `arcgraph serve` binary does not run the community scheduler
//!   at v1.0-α; the gauge fires wherever `bootstrap_engine` composes
//!   the engine (embedded deployments today; the serve binary the day
//!   it bootstraps the engine — the wire is the already-shipped
//!   `EngineConfig` field). Scheduler-in-binary wiring stays tracked
//!   as its own Operations slice per ADR-202 D-8.
//!
//! # W17δ #313 — Hot-vertex metric registered (counter semantic)
//!
//! `arcgraph_hot_vertex_warnings_total{tenant}` is now registered as an
//! `IntCounterVec`. Per issue #313's option (1), the counter semantic is
//! chosen (the existing `tracing::warn!` site at
//! `arcgraph-storage/src/crud.rs::tel_append` fires per TEL overflow
//! event — a per-event counter, not a continuous ops/sec gauge). The
//! alerts.yml expr is correspondingly bound to
//! `rate(arcgraph_hot_vertex_warnings_total[1m]) > 100` — 100 warnings
//! per second is the §10.3 alert threshold mapped onto the counter rate.
//!
//! The PRODUCER wiring (CRUD calling `record_hot_vertex_warning`) is
//! forward-bound to the M4-08+ production-storage slice — at v1.0-α
//! the CRUD path does not hold an `Arc<MetricsSink>` and the TEL
//! overflow site only logs via `tracing::warn!`. The metric
//! registration + alert binding lights the SEAM so the M4-08+ slice
//! can plug in the producer without touching the alert config.
//!
//! # Why `prometheus` (not `metrics` or `opentelemetry`)
//!
//! - `prometheus` (tikv/rust-prometheus, Apache-2.0) is the canonical
//!   Rust binding for the Prometheus text-exposition format. The
//!   `TextEncoder` surface emits OpenMetrics-compatible text directly
//!   — no protobuf, no push-gateway, no external runtime.
//! - `metrics` (metrics-rs) is a metric-recording macro layer; an
//!   exporter still has to encode. M6-06 ships the exporter; we keep
//!   the surface minimal (no macros) so the call-sites are explicit
//!   and reviewable.
//! - `opentelemetry-rust` is heavyweight + bundles trace + log + metric
//!   into one runtime; design-v2 §10.2 line 699 mentions OpenTelemetry
//!   *traces* (forward to M6-08 tracing wiring) — for `/metrics` the
//!   thin `prometheus` crate is the right tool.
//!
//! Crate license verified Apache-2.0 (Prime Directive #1); the
//! per-build gate is `cargo deny check` (which carries the live
//! deny.toml allow-list, not a frozen point-in-time claim).
//!
//! # Listener-port deferral (port 9090 vs same-listener)
//!
//! design-v2 §10.2 line 699 says "Prometheus scrape endpoint on port
//! 9090 by default." The W15γ HTTP composition mounts `/metrics` on
//! the same listener as `/mcp` and `/healthz` (typically port 3000
//! per W14α defaults). That composition deliberately uses one listener
//! for the current HTTP surface. A dual-listener split is deferred
//! until operators can expose `/metrics` on an internal interface
//! while `/mcp` remains on a public one.
//!
//! # ADR provenance
//!
//! - **design-v2 §10.2** — Observability metric inventory (lines
//!   701-708).
//! - **design-v2 §9.4** — HTTP MCP transport surface; `/metrics`
//!   shares the same listener as `/mcp` + `/healthz` per the W14α
//!   composition; a future dual-listener split will separate public
//!   and internal interfaces.
//! - **ADR-038 amendment-03 §TIER-2-c (observability)** — per-tenant
//!   tagging via the `tenant` label.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use arcgraph_community::RefreshObserver as CommunityRefreshObserver;
use arcgraph_core::TenantId;
use arcgraph_storage::metrics::{
    MetricsSink as StorageMetricsSink, QueryPlanType, StoragePageKind, WalWriteOutcome,
};
use prometheus::{
    Encoder, Histogram, HistogramVec, IntCounterVec, IntGaugeVec, Registry, TextEncoder,
};
use thiserror::Error;

use crate::rate_limit::OpClass;

// ---------------------------------------------------------------------
// Public constants
// ---------------------------------------------------------------------

/// HTTP path the Prometheus scrape endpoint binds to.
///
/// Mounted on the same listener as [`crate::transport::http::PATH_MCP`]
/// and [`crate::transport::http::PATH_HEALTHZ`]; GET-only, returns the
/// Prometheus text-exposition format ([`prometheus::TextEncoder`]).
pub const PATH_METRICS: &str = "/metrics";

/// `Content-Type` header value for the Prometheus text-exposition
/// format (`text/plain; version=0.0.4`).
///
/// Per the [Prometheus exposition formats spec][prom-spec] §2: the
/// canonical content-type carries the version qualifier so the
/// scraper can branch on protocol version. v0.0.4 is the format the
/// `prometheus::TextEncoder` emits.
///
/// [prom-spec]: https://prometheus.io/docs/instrumenting/exposition_formats/
pub const CONTENT_TYPE_PROMETHEUS_TEXT: &str = "text/plain; version=0.0.4; charset=utf-8";

/// Histogram bucket boundaries (**milliseconds**) for
/// `arcgraph_wal_fsync_duration_ms`.
///
/// design-v2 §10.2 line 704 target: P99 ≤ 5 ms. design-v2 §A.3
/// budget table: WAL group-commit fsync window is 1 ms (default
/// `WalConfig::group_commit_window`). The bucket boundaries are
/// chosen to give:
///
/// - **Sub-ms resolution** at the inner steady-state (50µs … 500µs)
///   for fast SSD storage where well-tuned fsyncs land below 1 ms.
/// - **Detection band** at 1ms…20ms covering the §10.2 P99 target
///   plus the alerting runbook threshold (§10.3 "WAL fsync P99
///   > 10ms" mandates investigation).
/// - **Outlier ceiling** at 100ms catching disk-degraded /
///   fsync-stall scenarios; values past 100ms aggregate into +Inf
///   and trigger the alerting runbook.
///
/// 12 buckets covers the cardinality budget (each cell is a
/// histogram-bucket scrape row).
pub const WAL_FSYNC_BUCKETS_MS: [f64; 12] = [
    0.050, // 50µs — fast SSD floor
    0.100, // 100µs
    0.250, // 250µs
    0.500, // 500µs — sub-millisecond steady state
    1.0,   // 1ms — group_commit_window default
    2.5,   // 2.5ms
    5.0,   // 5ms — §10.2 P99 target
    10.0,  // 10ms — §10.3 alerting threshold
    20.0,  // 20ms
    50.0,  // 50ms — disk saturation onset
    100.0, // 100ms — disk degradation
    500.0, // 500ms — outlier ceiling
];

/// Default histogram bucket boundaries (**milliseconds**) for
/// `arcgraph_read_latency_ms` and `arcgraph_write_latency_ms`.
///
/// Covers design-v2 §10.5 P50/P99 targets:
///
/// - **IS1** P50 = 50us, P99 = 500us.
/// - **IS3** P50 = 500us, P99 = 5ms.
/// - **IS4–7** P50 = 2ms, P99 = 20ms.
/// - **IC1** P50 = 5ms, P99 = 50ms.
/// - **IC2–4** P50 = 50ms, P99 = 500ms.
/// - **IC5–6** P50 = 200ms, P99 = 2s.
/// - **IU1–8** P50 = 5ms, P99 = 50ms.
///
/// 16 buckets in geometric progression (factor ~2.5x) from 50us to
/// 5s; one extra bucket above 2s catches IC5 P99 + outliers without
/// truncating into +Inf. Units are **milliseconds** per the §10.2
/// `_ms` suffix on lines 701-702. The Prometheus default histogram
/// bucket set is calibrated for ~10ms-10s web requests and would
/// lose resolution in the sub-millisecond region where 4 of the 7
/// IS queries land.
pub const DEFAULT_LATENCY_BUCKETS_MS: [f64; 16] = [
    0.050,   // 50us — IS1 P50
    0.100,   // 100us
    0.200,   // 200us — IS2 P50
    0.500,   // 500us — IS3 P50 / IS1 P99
    1.0,     // 1ms
    2.0,     // 2ms — IS4–7 P50 / IS2 P99
    5.0,     // 5ms — IS3 P99 / IC1 P50 / IU P50
    10.0,    // 10ms
    20.0,    // 20ms — IS4–7 P99
    50.0,    // 50ms — IC1 P99 / IC2–4 P50 / IU P99
    100.0,   // 100ms
    200.0,   // 200ms — IC5–6 P50
    500.0,   // 500ms — IC2–4 P99
    1_000.0, // 1s
    2_000.0, // 2s — IC5–6 P99
    5_000.0, // 5s — outlier ceiling
];

/// MCP tool-invocation status label values for
/// `arcgraph_mcp_tool_invocations`.
///
/// Per design-v2 §10.2 line 706: `{tool, status}`. `status` carries
/// the dispatch result: success (the envelope's `result` branch
/// fires) or error (the envelope's `error` branch fires).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ToolInvocationStatus {
    /// Dispatch returned a successful JSON-RPC `result` envelope.
    Ok,
    /// Dispatch returned a JSON-RPC `error` envelope (any code).
    Error,
}

impl ToolInvocationStatus {
    /// String form used as the `status` label value.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Error => "error",
        }
    }
}

/// Transport label values for `arcgraph_active_connections`.
///
/// W16γ M6-07: all three v1.0-α transports report through this gauge
/// (the W14α `serve_http` + W13δ `serve_stdio` + W14δ `serve_bolt_listener`
/// accept-loops increment, RAII guards decrement). The `Stdio` and
/// `Bolt` variants ship together with their wires in this PR
/// (no speculative scaffolds — per `feedback_avoid_speculative_scaffolding.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConnectionTransport {
    /// W14α M5-02b HTTP transport. Wired at PR #309.
    Http,
    /// W13δ M5-01 stdio MCP transport. Wired at W16γ M6-07 PR.
    Stdio,
    /// W14δ M5-13 Bolt 5.0 TCP listener. Wired at W16γ M6-07 PR.
    Bolt,
}

impl ConnectionTransport {
    /// String form used as the `transport` label value.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Http => "http",
            Self::Stdio => "stdio",
            Self::Bolt => "bolt",
        }
    }
}

// ---------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------

/// Errors emitted by [`MetricsRegistry`] construction or scrape.
///
/// `#[non_exhaustive]` under the code-quality policy — adding a new variant is
/// additive (e.g., M6-07 may introduce a `LabelCardinalityExceeded`
/// guardrail).
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum MetricsError {
    /// `prometheus::Registry::register` rejected a metric (duplicate
    /// name, invalid label set, etc.).
    #[error("metrics registry: {0}")]
    Registry(#[from] prometheus::Error),
    /// Text-encoding failed. `prometheus::TextEncoder::encode` only
    /// fails on I/O errors against the underlying writer; the
    /// in-memory `Vec<u8>` path the exporter uses never returns I/O
    /// errors so this variant is forward-pinned for callers that may
    /// route encoding through a streaming writer.
    #[error("metrics text encoding: {0}")]
    Encode(String),
}

// ---------------------------------------------------------------------
// MetricsRegistry — the load-bearing public type
// ---------------------------------------------------------------------

/// Composes the four W15γ-shipped metrics over a `prometheus::Registry`.
///
/// Wrapped in [`Arc`] at the call-site (the HTTP transport's
/// `RequestPolicy` holds an `Option<Arc<MetricsRegistry>>`); shared
/// across the accept loop's request handlers. Internally each metric
/// is a `prometheus::*Vec` keyed on label values — concurrent writes
/// from per-request handlers are correct (the prometheus crate's
/// metric atomics handle the synchronization).
///
/// # Cloning
///
/// Constructing one registry per process is the canonical pattern; the
/// `prometheus::Registry` deduplicates metric names so cloning at the
/// `Arc<MetricsRegistry>` level (cheap pointer copy) is the right
/// pattern across worker threads.
#[derive(Debug)]
pub struct MetricsRegistry {
    registry: Registry,
    mcp_tool_invocations: IntCounterVec,
    read_latency_ms: HistogramVec,
    write_latency_ms: HistogramVec,
    active_connections: IntGaugeVec,
    /// W16γ M6-07 — §10.2 line 704 closure.
    wal_fsync_duration_ms: Histogram,
    /// W16γ M6-07 — operational supplement; cross-cited by PR #307.
    wal_writes_total: IntCounterVec,
    /// W16γ M6-07 — operational supplement (drives the §10.2 line
    /// 703 hit-rate Grafana panel via PromQL ratio).
    storage_pages_total: IntCounterVec,
    /// W17δ #313 (registered) + W28 #582 (FIRED) — §10.2 **line 721**
    /// hot-vertex warnings counter. Increment site:
    /// TEL/reverse-TEL overflow allocation in
    /// `arcgraph-storage/src/crud.rs::tel_append` /
    /// `tel_append_reverse` (per-event signal that a vertex's out-/in-
    /// adjacency is approaching the supernode threshold), wired through
    /// `MetricsSink::record_hot_vertex_warning` at W28 #582 (the no-op
    /// trampoline fix). Alert:
    /// `rate(arcgraph_hot_vertex_warnings_total[1m]) > 100` per §10.3
    /// **line 733** (100 warnings/sec sustained → operator
    /// investigation).
    hot_vertex_warnings_total: IntCounterVec,
    /// W28 #582 — §10.2 **line 723** query plan-choice counter.
    /// Increment site: the `arcgraph-query` `QueryEngine` plan-build
    /// path, once per executed plan, wired through
    /// `MetricsSink::record_query_plan_choice`. `plan_type` ∈
    /// {binary, wcoj, free_join} per the §10.2 line 723 cite-text
    /// `(binary/wcoj/free_join)`. v1.0-α emits only `binary` (the
    /// engine produces binary join plans exclusively); wcoj / free_join
    /// are the v1.1+ label values.
    query_plan_choice: IntCounterVec,
    /// ADR-202 — §10.2 **line 724** community-detection freshness
    /// gauge. Value = Unix timestamp (whole seconds) of the tenant's
    /// most recent SUCCESSFUL community refresh, set via the
    /// community-resident `RefreshObserver` seam (producer:
    /// `arcgraph-community::CommunityRefreshScheduler`'s success arm,
    /// after `install_into` returns). Alert contract:
    /// `time() - arcgraph_leiden_last_run_seconds > (48 * 3600)`
    /// (docs/grafana/alerts.yml `ArcGraphLeidenFreshnessStale`,
    /// design-v2 §10.3 "Community detection freshness > 48h").
    leiden_last_run_seconds: IntGaugeVec,
}

impl MetricsRegistry {
    /// Build a fresh [`MetricsRegistry`] with the W15γ-shipped metrics
    /// registered against a private `prometheus::Registry`.
    ///
    /// # Errors
    ///
    /// Returns [`MetricsError::Registry`] if any metric fails to
    /// register — in practice this only fires if a future addition
    /// duplicates an existing metric name (defensive at v1.0-alpha;
    /// load-bearing forward as the registry grows).
    pub fn new() -> Result<Self, MetricsError> {
        let registry = Registry::new();

        let mcp_tool_invocations = IntCounterVec::new(
            prometheus::Opts::new(
                "arcgraph_mcp_tool_invocations",
                "Per design-v2 §10.2 line 706: MCP tool invocation count by tenant, tool name (JSON-RPC method), and status (ok/error).",
            ),
            &["tenant", "tool", "status"],
        )?;
        registry.register(Box::new(mcp_tool_invocations.clone()))?;

        let read_latency_ms = HistogramVec::new(
            prometheus::HistogramOpts::new(
                "arcgraph_read_latency_ms",
                "Per design-v2 §10.2 line 702: read-class JSON-RPC dispatch latency (milliseconds) by tenant and tool.",
            )
            .buckets(DEFAULT_LATENCY_BUCKETS_MS.to_vec()),
            &["tenant", "tool"],
        )?;
        registry.register(Box::new(read_latency_ms.clone()))?;

        let write_latency_ms = HistogramVec::new(
            prometheus::HistogramOpts::new(
                "arcgraph_write_latency_ms",
                "Per design-v2 §10.2 line 701: write-class JSON-RPC dispatch latency (milliseconds) by tenant and tool.",
            )
            .buckets(DEFAULT_LATENCY_BUCKETS_MS.to_vec()),
            &["tenant", "tool"],
        )?;
        registry.register(Box::new(write_latency_ms.clone()))?;

        let active_connections = IntGaugeVec::new(
            prometheus::Opts::new(
                "arcgraph_active_connections",
                "Active connection count per transport (operational gauge cross-cited by sister PR #307 Grafana dashboards).",
            ),
            &["transport"],
        )?;
        registry.register(Box::new(active_connections.clone()))?;

        // W16γ M6-07 — §10.2 line 704 WAL fsync duration histogram.
        let wal_fsync_duration_ms = Histogram::with_opts(
            prometheus::HistogramOpts::new(
                "arcgraph_wal_fsync_duration_ms",
                "Per design-v2 §10.2 line 704: WAL group-commit fsync wall-clock duration (milliseconds). Target P99 ≤ 5 ms.",
            )
            .buckets(WAL_FSYNC_BUCKETS_MS.to_vec()),
        )?;
        registry.register(Box::new(wal_fsync_duration_ms.clone()))?;

        // W16γ M6-07 — operational supplement: WAL writes-by-outcome.
        // Cross-cited by sister PR #307 Grafana dashboards.
        let wal_writes_total = IntCounterVec::new(
            prometheus::Opts::new(
                "arcgraph_wal_writes_total",
                "W16γ M6-07: WAL writer accepted-append + fsync-fail counter. \
                 outcome ∈ {t1_sync, t3_async, fsync_fail} per ADR-034 §Slice B + §6.2.",
            ),
            &["outcome"],
        )?;
        registry.register(Box::new(wal_writes_total.clone()))?;

        // W16γ M6-07 — operational supplement: buffer pool page kinds.
        // Drives the §10.2 line 703 hit-rate panel via PromQL ratio.
        let storage_pages_total = IntCounterVec::new(
            prometheus::Opts::new(
                "arcgraph_storage_pages_total",
                "W16γ M6-07: BufferPool pin-path counter. kind ∈ {hit, miss, eviction}. \
                 The §10.2 line 703 hit-rate target ≥ 0.95 is computed Grafana-side as \
                 rate(kind=hit) / rate(total).",
            ),
            &["kind"],
        )?;
        registry.register(Box::new(storage_pages_total.clone()))?;

        // W17δ #313 (registered) + W28 #582 (FIRED) — §10.2 line 721
        // hot-vertex warnings counter. Per-tenant cardinality
        // (operators investigating workload patterns benefit from
        // per-tenant attribution). W28 #582 wires the producer call
        // site (`tel_append` / `tel_append_reverse` TEL overflow) via
        // `MetricsSink::record_hot_vertex_warning` — closing the W17δ
        // #313 no-op trampoline where this series was gather-able but
        // never incremented by any producer.
        let hot_vertex_warnings_total = IntCounterVec::new(
            prometheus::Opts::new(
                "arcgraph_hot_vertex_warnings_total",
                "W17δ #313 / §10.2 line 721: hot-vertex warning counter \
                 (per-event, per-tenant). Incremented when the TEL chain \
                 allocates an overflow block (vertex approaching supernode \
                 threshold). Alert: rate(...[1m]) > 100 per §10.3 line 733.",
            ),
            &["tenant"],
        )?;
        registry.register(Box::new(hot_vertex_warnings_total.clone()))?;

        // W28 #582 — §10.2 line 723 query plan-choice counter.
        // `plan_type` ∈ {binary, wcoj, free_join} per the §10.2 line
        // 723 cite-text `(binary/wcoj/free_join)`. Wired through
        // `MetricsSink::record_query_plan_choice` from the
        // `arcgraph-query` `QueryEngine` execute plan-build path. The
        // v1.0-α engine produces binary join plans exclusively, so
        // only the `binary` label is emitted today; the wcoj /
        // free_join cells materialise when those planners land (v1.1+).
        let query_plan_choice = IntCounterVec::new(
            prometheus::Opts::new(
                "arcgraph_query_plan_choice",
                "W28 #582 / §10.2 line 723: per-query plan-type choice \
                 counter. plan_type ∈ {binary, wcoj, free_join}. v1.0-α \
                 emits only `binary` (binary join plans exclusively); \
                 wcoj / free_join reserved for v1.1+ planners.",
            ),
            &["plan_type"],
        )?;
        registry.register(Box::new(query_plan_choice.clone()))?;

        // ADR-202 — §10.2 line 724 community-detection freshness
        // gauge. Per-tenant per ADR-038 amendment-03 §TIER-2-c (same
        // label convention as hot_vertex_warnings_total). The value
        // is the Unix timestamp of the last SUCCESSFUL refresh —
        // NOT a duration — so the shipped alerts.yml contract_expr
        // `time() - <gauge> > 48*3600` is arithmetically correct.
        let leiden_last_run_seconds = IntGaugeVec::new(
            prometheus::Opts::new(
                "arcgraph_leiden_last_run_seconds",
                "ADR-202 / §10.2 line 724: Unix timestamp (seconds) of \
                 the tenant's most recent successful community (Leiden) \
                 refresh. Absent until the first successful refresh in \
                 this process. Alert: time() - value > 48h per §10.3 \
                 (ArcGraphLeidenFreshnessStale).",
            ),
            &["tenant"],
        )?;
        registry.register(Box::new(leiden_last_run_seconds.clone()))?;

        Ok(Self {
            registry,
            mcp_tool_invocations,
            read_latency_ms,
            write_latency_ms,
            active_connections,
            wal_fsync_duration_ms,
            wal_writes_total,
            storage_pages_total,
            hot_vertex_warnings_total,
            query_plan_choice,
            leiden_last_run_seconds,
        })
    }

    /// Construct an `Arc<MetricsRegistry>` for the common per-process
    /// shared-ownership pattern.
    ///
    /// # Errors
    ///
    /// See [`Self::new`].
    pub fn shared() -> Result<Arc<Self>, MetricsError> {
        Ok(Arc::new(Self::new()?))
    }

    /// Record a JSON-RPC dispatch (counter + latency histogram).
    ///
    /// Increments `arcgraph_mcp_tool_invocations{tenant, tool, status}`
    /// by 1 and observes `duration_ms` into either
    /// `arcgraph_read_latency_ms{tenant, tool}` or
    /// `arcgraph_write_latency_ms{tenant, tool}` per `op_class`.
    pub fn record_dispatch(
        &self,
        tenant: TenantId,
        tool: &str,
        op_class: OpClass,
        status: ToolInvocationStatus,
        duration_ms: f64,
    ) {
        let tenant_label = tenant.raw().to_string();
        self.mcp_tool_invocations
            .with_label_values(&[&tenant_label, tool, status.as_str()])
            .inc();
        let latency_vec = match op_class {
            OpClass::Read => &self.read_latency_ms,
            OpClass::Write => &self.write_latency_ms,
        };
        latency_vec
            .with_label_values(&[&tenant_label, tool])
            .observe(duration_ms);
    }

    /// Set the active-connections gauge for the given transport.
    ///
    /// W16γ M6-07: all three v1.0-α transports are wired
    /// ([`ConnectionTransport::Http`] / `Stdio` / `Bolt`).
    pub fn set_active_connections(&self, transport: ConnectionTransport, value: u64) {
        let signed = i64::try_from(value).unwrap_or(i64::MAX);
        self.active_connections
            .with_label_values(&[transport.as_str()])
            .set(signed);
    }

    /// W16γ M6-07 — observe one `wal_fsync_duration_ms` sample.
    ///
    /// Exposed both as a method and through the
    /// [`StorageMetricsSink`] impl below; the method form is for
    /// intra-`arcgraph-mcp` callers that already know the concrete
    /// type (e.g., the test suite).
    pub fn observe_wal_fsync_ms(&self, duration_ms: f64) {
        self.wal_fsync_duration_ms.observe(duration_ms);
    }

    /// W16γ M6-07 — record one `wal_writes_total{outcome}` increment.
    pub fn record_wal_write(&self, outcome: WalWriteOutcome) {
        self.wal_writes_total
            .with_label_values(&[outcome.as_str()])
            .inc();
    }

    /// W16γ M6-07 — record one `storage_pages_total{kind}` increment.
    pub fn record_storage_page(&self, kind: StoragePageKind) {
        self.storage_pages_total
            .with_label_values(&[kind.as_str()])
            .inc();
    }

    /// W17δ #313 (method) + W28 #582 (producer wired) — record one
    /// `hot_vertex_warnings_total{tenant}` increment (§10.2 line 721).
    ///
    /// The producer for this increment is the TEL/reverse-TEL overflow
    /// site in `arcgraph-storage/src/crud.rs::tel_append` /
    /// `tel_append_reverse`. Since W28 #582 (ADR-045) the CRUD layer
    /// holds an `Option<Arc<dyn MetricsSink>>` and fires this method
    /// (via the [`StorageMetricsSink`] impl below) at the overflow-
    /// block-allocation site alongside the existing `tracing::warn!`.
    /// Before #582 this method existed but had NO producer caller — a
    /// registered-but-never-fired no-op trampoline.
    pub fn record_hot_vertex_warning(&self, tenant: TenantId) {
        self.hot_vertex_warnings_total
            .with_label_values(&[&tenant.raw().to_string()])
            .inc();
    }

    /// W28 #582 — record one `query_plan_choice{plan_type}` increment
    /// (§10.2 line 723).
    ///
    /// Exposed both as a method and through the [`StorageMetricsSink`]
    /// impl below; the producer is the `arcgraph-query` `QueryEngine`
    /// plan-build path (once per executed plan). `plan_type` carries
    /// the §10.2 line 723 taxonomy `(binary/wcoj/free_join)`.
    pub fn record_query_plan_choice(&self, plan_type: QueryPlanType) {
        self.query_plan_choice
            .with_label_values(&[plan_type.as_str()])
            .inc();
    }

    /// ADR-202 — set `arcgraph_leiden_last_run_seconds{tenant}` to
    /// the current Unix time (§10.2 line 724).
    ///
    /// Called (via the [`CommunityRefreshObserver`] impl below) by
    /// the `arcgraph-community` refresh scheduler once per successful
    /// per-tenant refresh, synchronously after the new community
    /// snapshot is installed. The registry owns the clock read (the
    /// observer contract passes no timestamp): the call is
    /// synchronous at the producer site, so impl-side `now()` equals
    /// completion time at second granularity — 5 orders of magnitude
    /// below the 48 h alert threshold.
    ///
    /// Clock caveats (ADR-202 D-2): wall-clock, NTP-adjustable (the
    /// standard `*_timestamp_seconds`-pattern caveat); a pre-epoch
    /// clock maps to 0, which reads maximally stale — fail-loud,
    /// never fail-silent.
    pub fn record_leiden_refresh_success(&self, tenant: TenantId) {
        let unix_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let signed = i64::try_from(unix_secs).unwrap_or(i64::MAX);
        self.leiden_last_run_seconds
            .with_label_values(&[&tenant.raw().to_string()])
            .set(signed);
    }

    /// Encode the registry as Prometheus text-exposition format.
    ///
    /// Returns the encoded bytes — the HTTP transport's
    /// `GET /metrics` handler wraps these in a `Response` with
    /// `Content-Type: ` [`CONTENT_TYPE_PROMETHEUS_TEXT`].
    ///
    /// # Errors
    ///
    /// Returns [`MetricsError::Encode`] if the `TextEncoder` rejects
    /// the gather (in practice unreachable on the in-memory writer
    /// path; the variant exists for forward compat).
    pub fn gather_text(&self) -> Result<Vec<u8>, MetricsError> {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buf = Vec::with_capacity(4096);
        encoder
            .encode(&metric_families, &mut buf)
            .map_err(|e| MetricsError::Encode(e.to_string()))?;
        Ok(buf)
    }

    /// Get a reference to the underlying `prometheus::Registry`.
    ///
    /// Exposed so M6-07 Grafana wiring can register additional
    /// metrics (the §10.2 metrics forward-pinned at this slice).
    #[must_use]
    pub fn registry(&self) -> &Registry {
        &self.registry
    }
}

impl Clone for MetricsRegistry {
    /// Cloning shares the underlying `prometheus::Registry` + metric
    /// vectors — the prometheus crate's types are `Clone` by Arc-share
    /// design, so this is `O(1)` and observers see the same counter
    /// updates regardless of which clone they hold.
    fn clone(&self) -> Self {
        Self {
            registry: self.registry.clone(),
            mcp_tool_invocations: self.mcp_tool_invocations.clone(),
            read_latency_ms: self.read_latency_ms.clone(),
            write_latency_ms: self.write_latency_ms.clone(),
            active_connections: self.active_connections.clone(),
            wal_fsync_duration_ms: self.wal_fsync_duration_ms.clone(),
            wal_writes_total: self.wal_writes_total.clone(),
            storage_pages_total: self.storage_pages_total.clone(),
            hot_vertex_warnings_total: self.hot_vertex_warnings_total.clone(),
            query_plan_choice: self.query_plan_choice.clone(),
            leiden_last_run_seconds: self.leiden_last_run_seconds.clone(),
        }
    }
}

// ---------------------------------------------------------------------
// W16γ M6-07 — `MetricsSink` impl bridging arcgraph-storage producers.
// ---------------------------------------------------------------------

/// ADR-045 §"Decision": `MetricsRegistry` is the concrete impl of
/// `arcgraph-storage::metrics::MetricsSink`. Producers in
/// `arcgraph-storage` (WAL writer + buffer pool) call through the
/// trait; the dep edge `mcp → storage` keeps PD-7 bounded contexts
/// satisfied.
impl StorageMetricsSink for MetricsRegistry {
    fn record_wal_write(&self, outcome: WalWriteOutcome) {
        Self::record_wal_write(self, outcome);
    }

    fn observe_wal_fsync_ms(&self, duration_ms: f64) {
        Self::observe_wal_fsync_ms(self, duration_ms);
    }

    fn record_storage_page(&self, kind: StoragePageKind) {
        Self::record_storage_page(self, kind);
    }

    fn record_hot_vertex_warning(&self, tenant: TenantId) {
        Self::record_hot_vertex_warning(self, tenant);
    }

    fn record_query_plan_choice(&self, plan_type: QueryPlanType) {
        Self::record_query_plan_choice(self, plan_type);
    }
}

// ---------------------------------------------------------------------
// ADR-202 — community-resident `RefreshObserver` impl.
// ---------------------------------------------------------------------

/// ADR-202 §"Decision" D-2: `MetricsRegistry` is the concrete impl of
/// `arcgraph-community::scheduler::RefreshObserver` — the
/// community-context analogue of the [`StorageMetricsSink`] impl
/// above. The dep edge `mcp → community` is cycle-free
/// (`arcgraph-community` sits beneath `arcgraph-storage`, which mcp
/// already depends on) and PD-7-clean: the cross-context call goes
/// through the community-published trait.
///
/// The scheduler invokes this once per successful per-tenant refresh
/// (success arm only — never on soft-skip or failure), synchronously
/// after `install_into` returns. Per-call cost: one vtable dispatch +
/// one `SystemTime::now()` + one labelled atomic store, on a path
/// that fires once per tenant per scheduler cadence (default 24 h) —
/// the coldest producer path in the system.
impl CommunityRefreshObserver for MetricsRegistry {
    fn record_refresh_success(&self, tenant: TenantId) {
        Self::record_leiden_refresh_success(self, tenant);
    }
}

// ---------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Pin: the registry constructs cleanly with the W15γ + W16γ
    /// metrics. A duplicate-name regression (e.g., adding two
    /// `arcgraph_mcp_tool_invocations` registrations) would surface
    /// here as a `MetricsError::Registry` before any consumer is
    /// wired.
    #[test]
    fn metrics_registry_init_registers_all_metrics_without_error() {
        let r = MetricsRegistry::new().expect("init");
        // Probe each metric by exercising its setter — if a metric
        // failed to register, the `with_label_values` would panic on
        // the missing collector. The body emitted by `gather_text`
        // must include every metric name.
        r.record_dispatch(
            TenantId::new(1),
            "graph.schema",
            OpClass::Read,
            ToolInvocationStatus::Ok,
            1.0,
        );
        r.record_dispatch(
            TenantId::new(1),
            "graph.ingest",
            OpClass::Write,
            ToolInvocationStatus::Ok,
            2.0,
        );
        r.set_active_connections(ConnectionTransport::Http, 4);
        // W16γ M6-07 new surfaces:
        r.observe_wal_fsync_ms(0.5);
        r.record_wal_write(WalWriteOutcome::T1Sync);
        r.record_storage_page(StoragePageKind::Hit);
        // W17δ #313 hot-vertex new surface:
        r.record_hot_vertex_warning(TenantId::new(1));
        // W28 #582 query plan-choice new surface:
        r.record_query_plan_choice(QueryPlanType::Binary);

        let text = r.gather_text().expect("gather");
        let text = String::from_utf8(text).expect("utf-8");
        assert!(text.contains("arcgraph_mcp_tool_invocations"));
        assert!(text.contains("arcgraph_read_latency_ms"));
        assert!(text.contains("arcgraph_write_latency_ms"));
        assert!(text.contains("arcgraph_active_connections"));
        assert!(text.contains("arcgraph_wal_fsync_duration_ms"));
        assert!(text.contains("arcgraph_wal_writes_total"));
        assert!(text.contains("arcgraph_storage_pages_total"));
        assert!(text.contains("arcgraph_hot_vertex_warnings_total"));
        assert!(text.contains("arcgraph_query_plan_choice"));
    }

    /// W17δ #313 closure — regression pin for the hot-vertex counter.
    ///
    /// Synthetic high-warning-rate fixture: 200 warning increments for
    /// tenant 1, 0 for tenant 2. The Prometheus rate computation can't
    /// be exercised in a unit test (it requires a TSDB + time
    /// progression), but we CAN pin the counter value the alert expr
    /// will read. The alert
    /// `rate(arcgraph_hot_vertex_warnings_total[1m]) > 100` would fire
    /// for tenant 1 (200 events / 60s ≈ 3.3/s × the 60s rate window
    /// span ≈ 200) once a real Prometheus scraper observes the rate.
    ///
    /// Per-tenant attribution is the load-bearing pin: tenant 2 must
    /// NOT see tenant 1's increments leak into its label slice.
    #[test]
    fn record_hot_vertex_warning_increments_per_tenant() {
        let r = MetricsRegistry::new().expect("init");
        for _ in 0..200 {
            r.record_hot_vertex_warning(TenantId::new(1));
        }
        // Tenant 2 sees zero increments — no call is made so the
        // prometheus crate doesn't materialize a cell for tenant=2.
        let text = String::from_utf8(r.gather_text().expect("gather")).expect("utf-8");
        assert!(
            text.contains(r#"arcgraph_hot_vertex_warnings_total{tenant="1"} 200"#),
            "tenant=1 hot-vertex warnings must be 200; gathered:\n{text}"
        );
        // Tenant 2 either absent or zero; the prometheus crate omits
        // zero-valued counter cells from the text exposition until
        // first increment. Either is acceptable; the load-bearing pin
        // is the ABSENCE of a "tenant=\"2\"" row with a non-zero value.
        assert!(
            !text.contains(r#"arcgraph_hot_vertex_warnings_total{tenant="2"}"#),
            "tenant=2 must NOT have a hot-vertex cell (zero increments); gathered:\n{text}"
        );
    }

    /// W17δ #313 closure — pin the alert-threshold-cross semantic.
    ///
    /// Normal-load fixture (1 increment) MUST NOT cross the threshold
    /// in any sane rate-window computation; high-load fixture (200
    /// increments in quick succession) MUST. We compare the raw
    /// counter delta as the model: the alert
    /// `rate(arcgraph_hot_vertex_warnings_total[1m]) > 100` is
    /// `100/s × 60s = 6000` increments in a 1m window at the threshold;
    /// 200 is the canonical "well-above-quiet, well-below-overflow"
    /// fixture per the §10.3 cite-text intent.
    #[test]
    fn hot_vertex_normal_load_vs_high_load_distinct_counter_deltas() {
        let normal = MetricsRegistry::new().expect("init");
        normal.record_hot_vertex_warning(TenantId::new(99));
        let normal_text = String::from_utf8(normal.gather_text().expect("g")).expect("utf-8");
        assert!(
            normal_text.contains(r#"arcgraph_hot_vertex_warnings_total{tenant="99"} 1"#),
            "normal-load fixture: exactly 1 increment for tenant=99",
        );

        let hot = MetricsRegistry::new().expect("init");
        for _ in 0..200 {
            hot.record_hot_vertex_warning(TenantId::new(99));
        }
        let hot_text = String::from_utf8(hot.gather_text().expect("g")).expect("utf-8");
        assert!(
            hot_text.contains(r#"arcgraph_hot_vertex_warnings_total{tenant="99"} 200"#),
            "high-load fixture: exactly 200 increments for tenant=99",
        );
    }

    /// Pin: `record_dispatch` increments per (tenant, tool, status).
    #[test]
    fn record_dispatch_increments_per_tenant_tool_status() {
        let r = MetricsRegistry::new().expect("init");
        r.record_dispatch(
            TenantId::new(1),
            "graph.schema",
            OpClass::Read,
            ToolInvocationStatus::Ok,
            1.0,
        );
        r.record_dispatch(
            TenantId::new(1),
            "graph.schema",
            OpClass::Read,
            ToolInvocationStatus::Ok,
            2.0,
        );
        r.record_dispatch(
            TenantId::new(1),
            "graph.ingest",
            OpClass::Write,
            ToolInvocationStatus::Error,
            5.0,
        );
        r.record_dispatch(
            TenantId::new(2),
            "graph.schema",
            OpClass::Read,
            ToolInvocationStatus::Ok,
            1.0,
        );

        let text = String::from_utf8(r.gather_text().expect("gather")).expect("utf-8");
        // Tenant=1 graph.schema ok: 2 increments.
        assert!(
            text.contains(
                r#"arcgraph_mcp_tool_invocations{status="ok",tenant="1",tool="graph.schema"} 2"#
            ),
            "tenant=1 graph.schema ok counter must be 2; gathered text was:\n{text}"
        );
        // Tenant=1 graph.ingest error: 1 increment.
        assert!(
            text.contains(
                r#"arcgraph_mcp_tool_invocations{status="error",tenant="1",tool="graph.ingest"} 1"#
            ),
            "tenant=1 graph.ingest error counter must be 1; gathered text was:\n{text}"
        );
        // Tenant=2 graph.schema ok: 1 increment.
        assert!(
            text.contains(
                r#"arcgraph_mcp_tool_invocations{status="ok",tenant="2",tool="graph.schema"} 1"#
            ),
            "tenant=2 graph.schema ok counter must be 1; gathered text was:\n{text}"
        );
    }

    /// Pin: read vs write op_class route to different histograms.
    #[test]
    fn record_dispatch_routes_op_class_to_correct_histogram() {
        let r = MetricsRegistry::new().expect("init");
        r.record_dispatch(
            TenantId::new(1),
            "graph.schema",
            OpClass::Read,
            ToolInvocationStatus::Ok,
            0.5,
        );
        r.record_dispatch(
            TenantId::new(1),
            "graph.ingest",
            OpClass::Write,
            ToolInvocationStatus::Ok,
            10.0,
        );

        let text = String::from_utf8(r.gather_text().expect("gather")).expect("utf-8");
        // Read histogram has the schema observation, NOT the ingest.
        assert!(
            text.contains(r#"arcgraph_read_latency_ms_count{tenant="1",tool="graph.schema"} 1"#),
            "read histogram must hold the graph.schema observation; text was:\n{text}"
        );
        assert!(
            !text.contains(r#"arcgraph_read_latency_ms_count{tenant="1",tool="graph.ingest"}"#),
            "read histogram must NOT contain a graph.ingest cell; text was:\n{text}"
        );
        // Write histogram has the ingest observation, NOT the schema.
        assert!(
            text.contains(r#"arcgraph_write_latency_ms_count{tenant="1",tool="graph.ingest"} 1"#),
            "write histogram must hold the graph.ingest observation; text was:\n{text}"
        );
        assert!(
            !text.contains(r#"arcgraph_write_latency_ms_count{tenant="1",tool="graph.schema"}"#),
            "write histogram must NOT contain a graph.schema cell; text was:\n{text}"
        );
    }

    /// Pin: the histogram bucket boundaries land at the design-v2
    /// §10.5 P50/P99 targets (in milliseconds) so a future regression
    /// that re-orders or drops boundaries fires this test.
    #[test]
    fn latency_histogram_buckets_cover_design_v2_10_5_targets() {
        // The bucket set MUST include the IS1 P50 (50us=0.050ms), IS3
        // P50 (500us=0.500ms), IS4-7 P50 (2ms), IC1 P50 (5ms), IC5
        // P50 (200ms), and IC5 P99 (2000ms) anchors — the regression-
        // tracking boundaries. A change here is a deliberate
        // observability policy change.
        let critical_anchors: &[f64] = &[
            0.050,   // IS1 P50 (50us in ms)
            0.500,   // IS3 P50 (500us in ms)
            2.0,     // IS4–7 P50
            5.0,     // IC1 P50
            200.0,   // IC5 P50
            2_000.0, // IC5 P99
        ];
        for anchor in critical_anchors {
            assert!(
                DEFAULT_LATENCY_BUCKETS_MS.contains(anchor),
                "DEFAULT_LATENCY_BUCKETS_MS must include {anchor}ms \
                 (design-v2 §10.5 anchor); buckets = {DEFAULT_LATENCY_BUCKETS_MS:?}"
            );
        }
        // Buckets must be monotonically increasing (prometheus invariant).
        for w in DEFAULT_LATENCY_BUCKETS_MS.windows(2) {
            assert!(w[0] < w[1], "buckets must be monotonic: {w:?}");
        }
    }

    /// Pin: `set_active_connections` per-transport isolation.
    #[test]
    fn set_active_connections_isolated_per_transport() {
        let r = MetricsRegistry::new().expect("init");
        r.set_active_connections(ConnectionTransport::Http, 5);

        let text = String::from_utf8(r.gather_text().expect("gather")).expect("utf-8");
        assert!(
            text.contains(r#"arcgraph_active_connections{transport="http"} 5"#),
            "http gauge must be 5; text was:\n{text}"
        );
    }

    /// Pin: `gather_text` emits Prometheus text-exposition format
    /// (line-oriented with `# HELP` and `# TYPE` comments). Exercises
    /// each of the four metrics with at least one observation so all
    /// four HELP/TYPE rows appear (prometheus::*Vec metrics only
    /// emit after the first label-tuple is observed).
    #[test]
    fn gather_text_emits_prometheus_text_exposition_format() {
        let r = MetricsRegistry::new().expect("init");
        // Observe one sample per metric so all four HELP/TYPE rows
        // appear in the text exposition output.
        r.record_dispatch(
            TenantId::new(1),
            "graph.schema",
            OpClass::Read,
            ToolInvocationStatus::Ok,
            1.0,
        );
        r.record_dispatch(
            TenantId::new(1),
            "graph.ingest",
            OpClass::Write,
            ToolInvocationStatus::Ok,
            5.0,
        );
        r.set_active_connections(ConnectionTransport::Http, 1);

        let text = String::from_utf8(r.gather_text().expect("gather")).expect("utf-8");
        // Per Prometheus text exposition format spec
        // (https://prometheus.io/docs/instrumenting/exposition_formats/),
        // each metric carries `# HELP <name> <description>` and
        // `# TYPE <name> <type>` comment lines preceding the samples.
        assert!(
            text.contains("# HELP arcgraph_mcp_tool_invocations"),
            "must emit HELP for mcp_tool_invocations; text was:\n{text}"
        );
        assert!(
            text.contains("# TYPE arcgraph_mcp_tool_invocations counter"),
            "mcp_tool_invocations TYPE must be 'counter'; text was:\n{text}"
        );
        assert!(
            text.contains("# TYPE arcgraph_read_latency_ms histogram"),
            "read_latency_ms TYPE must be 'histogram'; text was:\n{text}"
        );
        assert!(
            text.contains("# TYPE arcgraph_write_latency_ms histogram"),
            "write_latency_ms TYPE must be 'histogram'; text was:\n{text}"
        );
        assert!(
            text.contains("# TYPE arcgraph_active_connections gauge"),
            "active_connections TYPE must be 'gauge'; text was:\n{text}"
        );
    }

    /// Pin: histogram bucket samples are emitted (the `_bucket{le="..."}`
    /// rows) — verifies the histogram surface is functionally wired,
    /// not just registered.
    #[test]
    fn latency_histogram_emits_bucket_samples() {
        let r = MetricsRegistry::new().expect("init");
        // Observe 5 read-side samples at various latencies (in ms).
        for d in [0.030, 0.100, 0.400, 1.5, 30.0] {
            r.record_dispatch(
                TenantId::new(1),
                "graph.schema",
                OpClass::Read,
                ToolInvocationStatus::Ok,
                d,
            );
        }

        let text = String::from_utf8(r.gather_text().expect("gather")).expect("utf-8");
        // The histogram must emit cumulative counts at each bucket
        // boundary; the +Inf bucket carries the total count.
        assert!(
            text.contains(
                r#"arcgraph_read_latency_ms_bucket{tenant="1",tool="graph.schema",le="+Inf"} 5"#
            ),
            "read histogram +Inf bucket must carry the total count 5; text was:\n{text}"
        );
        // The histogram MUST emit a `_count` + `_sum` summary pair.
        assert!(
            text.contains(r#"arcgraph_read_latency_ms_count{tenant="1",tool="graph.schema"} 5"#),
            "read histogram _count must equal 5; text was:\n{text}"
        );
        assert!(
            text.contains(r#"arcgraph_read_latency_ms_sum{tenant="1",tool="graph.schema"}"#),
            "read histogram _sum must be emitted; text was:\n{text}"
        );
    }

    /// Pin: `MetricsRegistry::shared` returns an `Arc` whose strong
    /// count is 1 — callers can `.clone()` for distribution across
    /// worker threads.
    #[test]
    fn shared_returns_arc_with_strong_count_one() {
        let r = MetricsRegistry::shared().expect("shared");
        assert_eq!(Arc::strong_count(&r), 1);
        let r2 = r.clone();
        assert_eq!(Arc::strong_count(&r), 2);
        // Both Arcs observe the same counter increments.
        r.record_dispatch(
            TenantId::new(7),
            "graph.schema",
            OpClass::Read,
            ToolInvocationStatus::Ok,
            1.0,
        );
        let text1 = String::from_utf8(r.gather_text().expect("gather")).expect("utf-8");
        let text2 = String::from_utf8(r2.gather_text().expect("gather")).expect("utf-8");
        assert_eq!(text1, text2, "both Arc clones must see the same metrics");
    }

    /// Pin: tool-status + connection-transport label strings are
    /// stable per their canonical strings — drift in `as_str()`
    /// would silently break the Grafana panel labels.
    #[test]
    fn label_strings_are_canonical() {
        assert_eq!(ToolInvocationStatus::Ok.as_str(), "ok");
        assert_eq!(ToolInvocationStatus::Error.as_str(), "error");
        assert_eq!(ConnectionTransport::Http.as_str(), "http");
    }

    /// Pin: cold-start scrape behavior of the two prometheus metric
    /// classes used here:
    ///
    /// 1. `prometheus::*Vec` (HistogramVec / IntCounterVec /
    ///    IntGaugeVec) — emits NEITHER HELP nor TYPE nor samples
    ///    until the first label-tuple is observed. Cold-start
    ///    Grafana panels treat "metric absent" as "metric at zero".
    /// 2. `prometheus::Histogram` (bare) — ALWAYS emits zero-count
    ///    bucket rows at scrape, even cold. The bare Histogram is
    ///    used for `wal_fsync_duration_ms` because §10.2 line 704
    ///    cites it without labels; HistogramVec with a placeholder
    ///    label would be speculative scaffolding.
    ///
    /// This test pins both behaviors so future changes are
    /// deliberate observability-policy changes (an ADR-forcing
    /// regression).
    #[test]
    fn cold_registry_only_bare_histogram_emits_until_first_observation() {
        let r = MetricsRegistry::new().expect("init");
        let text = String::from_utf8(r.gather_text().expect("gather")).expect("utf-8");
        // The bare wal_fsync_duration_ms Histogram emits its zero-
        // bucket surface in cold scrape (no observations yet).
        assert!(
            text.contains("# HELP arcgraph_wal_fsync_duration_ms"),
            "bare Histogram emits HELP cold; text was:\n{text}"
        );
        assert!(
            text.contains("arcgraph_wal_fsync_duration_ms_count 0"),
            "bare Histogram emits _count=0 cold; text was:\n{text}"
        );
        // All *Vec metrics remain cold (no observations yet).
        assert!(
            !text.contains("# HELP arcgraph_mcp_tool_invocations"),
            "*Vec metrics emit no HELP cold; text was:\n{text}"
        );
        assert!(
            !text.contains("# HELP arcgraph_active_connections"),
            "*Vec metrics emit no HELP cold; text was:\n{text}"
        );
        assert!(
            !text.contains("# HELP arcgraph_wal_writes_total"),
            "*Vec metrics emit no HELP cold; text was:\n{text}"
        );
        assert!(
            !text.contains("# HELP arcgraph_storage_pages_total"),
            "*Vec metrics emit no HELP cold; text was:\n{text}"
        );

        // After ONE observation on the *Vec counter, that metric
        // appears in the text.
        r.record_dispatch(
            TenantId::new(1),
            "graph.schema",
            OpClass::Read,
            ToolInvocationStatus::Ok,
            1.0,
        );
        let text2 = String::from_utf8(r.gather_text().expect("gather")).expect("utf-8");
        assert!(
            text2.contains("# HELP arcgraph_mcp_tool_invocations"),
            "post-first-observation, mcp_tool_invocations HELP must appear; text was:\n{text2}"
        );
        assert!(
            text2.contains("# TYPE arcgraph_mcp_tool_invocations counter"),
            "post-first-observation, mcp_tool_invocations TYPE must appear; text was:\n{text2}"
        );
        // The active_connections metric is still cold (no observations);
        // its HELP/TYPE rows do NOT appear yet.
        assert!(
            !text2.contains("# HELP arcgraph_active_connections"),
            "active_connections must remain cold until its first observation; text was:\n{text2}"
        );
    }

    /// Pin: cloned registries share underlying state — `Clone` is
    /// `O(1)` (Arc-clone of the prometheus internals) and observers
    /// on either clone see the same counter updates.
    #[test]
    fn cloned_registry_shares_underlying_state() {
        let r1 = MetricsRegistry::new().expect("init");
        let r2 = r1.clone();
        r1.record_dispatch(
            TenantId::new(1),
            "graph.schema",
            OpClass::Read,
            ToolInvocationStatus::Ok,
            1.0,
        );
        r2.record_dispatch(
            TenantId::new(1),
            "graph.schema",
            OpClass::Read,
            ToolInvocationStatus::Ok,
            2.0,
        );
        let text = String::from_utf8(r1.gather_text().expect("gather")).expect("utf-8");
        assert!(
            text.contains(
                r#"arcgraph_mcp_tool_invocations{status="ok",tenant="1",tool="graph.schema"} 2"#
            ),
            "cloned registries must share state — counter must be 2; text was:\n{text}"
        );
    }

    // -----------------------------------------------------------------
    // W16γ M6-07 — new metric surface pins
    // -----------------------------------------------------------------

    /// Pin: extended `ConnectionTransport` label strings.
    #[test]
    fn connection_transport_label_strings_canonical_w16gamma() {
        assert_eq!(ConnectionTransport::Http.as_str(), "http");
        assert_eq!(ConnectionTransport::Stdio.as_str(), "stdio");
        assert_eq!(ConnectionTransport::Bolt.as_str(), "bolt");
    }

    /// Pin: `record_wal_write` increments per-outcome.
    #[test]
    fn record_wal_write_increments_per_outcome() {
        let r = MetricsRegistry::new().expect("init");
        r.record_wal_write(WalWriteOutcome::T1Sync);
        r.record_wal_write(WalWriteOutcome::T1Sync);
        r.record_wal_write(WalWriteOutcome::T1Sync);
        r.record_wal_write(WalWriteOutcome::T3Async);
        r.record_wal_write(WalWriteOutcome::FsyncFail);

        let text = String::from_utf8(r.gather_text().expect("gather")).expect("utf-8");
        assert!(
            text.contains(r#"arcgraph_wal_writes_total{outcome="t1_sync"} 3"#),
            "t1_sync counter must be 3; text was:\n{text}"
        );
        assert!(
            text.contains(r#"arcgraph_wal_writes_total{outcome="t3_async"} 1"#),
            "t3_async counter must be 1; text was:\n{text}"
        );
        assert!(
            text.contains(r#"arcgraph_wal_writes_total{outcome="fsync_fail"} 1"#),
            "fsync_fail counter must be 1; text was:\n{text}"
        );
    }

    /// Pin: `record_storage_page` increments per-kind.
    #[test]
    fn record_storage_page_increments_per_kind() {
        let r = MetricsRegistry::new().expect("init");
        for _ in 0..7 {
            r.record_storage_page(StoragePageKind::Hit);
        }
        r.record_storage_page(StoragePageKind::Miss);
        r.record_storage_page(StoragePageKind::Miss);
        r.record_storage_page(StoragePageKind::Eviction);

        let text = String::from_utf8(r.gather_text().expect("gather")).expect("utf-8");
        assert!(
            text.contains(r#"arcgraph_storage_pages_total{kind="hit"} 7"#),
            "hit counter must be 7; text was:\n{text}"
        );
        assert!(
            text.contains(r#"arcgraph_storage_pages_total{kind="miss"} 2"#),
            "miss counter must be 2; text was:\n{text}"
        );
        assert!(
            text.contains(r#"arcgraph_storage_pages_total{kind="eviction"} 1"#),
            "eviction counter must be 1; text was:\n{text}"
        );
    }

    /// Pin: `observe_wal_fsync_ms` emits `_bucket{le=...}` rows
    /// covering the §10.2 P99 target (5 ms) and the §10.3 alerting
    /// threshold (10 ms). Without those anchors a future bucket-set
    /// regression silently breaks the alerting runbook.
    #[test]
    fn wal_fsync_histogram_emits_buckets_covering_alert_thresholds() {
        let r = MetricsRegistry::new().expect("init");
        // Observe a spread across the bucket boundaries.
        for d in [0.030, 0.500, 1.0, 4.5, 9.0, 25.0] {
            r.observe_wal_fsync_ms(d);
        }
        let text = String::from_utf8(r.gather_text().expect("gather")).expect("utf-8");
        // The +Inf bucket carries the total count.
        assert!(
            text.contains(r#"arcgraph_wal_fsync_duration_ms_bucket{le="+Inf"} 6"#),
            "+Inf bucket count must be 6; text was:\n{text}"
        );
        // §10.2 P99 anchor.
        assert!(
            text.contains(r#"arcgraph_wal_fsync_duration_ms_bucket{le="5"}"#),
            "must emit a 5ms bucket boundary (§10.2 P99 target); text was:\n{text}"
        );
        // §10.3 alerting anchor.
        assert!(
            text.contains(r#"arcgraph_wal_fsync_duration_ms_bucket{le="10"}"#),
            "must emit a 10ms bucket boundary (§10.3 alerting threshold); text was:\n{text}"
        );
    }

    /// Pin: bucket boundaries cover the design-v2 §10.2 line 704 P99
    /// anchor + §10.3 alerting threshold. A change here is a
    /// deliberate observability-policy change; the bucket set is
    /// load-bearing for the alerting runbook.
    #[test]
    fn wal_fsync_buckets_anchor_design_v2_targets() {
        let anchors: &[f64] = &[
            0.500, // 500µs lower band
            1.0,   // group_commit_window
            5.0,   // §10.2 P99 target
            10.0,  // §10.3 alerting threshold
            100.0, // disk degradation
        ];
        for a in anchors {
            assert!(
                WAL_FSYNC_BUCKETS_MS.contains(a),
                "WAL_FSYNC_BUCKETS_MS must include {a} ms; buckets = {WAL_FSYNC_BUCKETS_MS:?}"
            );
        }
        for w in WAL_FSYNC_BUCKETS_MS.windows(2) {
            assert!(w[0] < w[1], "buckets must be monotonic: {w:?}");
        }
    }

    /// Pin: `set_active_connections` works for the new Stdio / Bolt
    /// variants and emits the canonical label values.
    #[test]
    fn set_active_connections_emits_stdio_and_bolt_labels() {
        let r = MetricsRegistry::new().expect("init");
        r.set_active_connections(ConnectionTransport::Stdio, 1);
        r.set_active_connections(ConnectionTransport::Bolt, 3);
        r.set_active_connections(ConnectionTransport::Http, 2);

        let text = String::from_utf8(r.gather_text().expect("gather")).expect("utf-8");
        assert!(
            text.contains(r#"arcgraph_active_connections{transport="stdio"} 1"#),
            "stdio gauge must be 1; text was:\n{text}"
        );
        assert!(
            text.contains(r#"arcgraph_active_connections{transport="bolt"} 3"#),
            "bolt gauge must be 3; text was:\n{text}"
        );
        assert!(
            text.contains(r#"arcgraph_active_connections{transport="http"} 2"#),
            "http gauge must be 2; text was:\n{text}"
        );
    }

    /// Pin: `MetricsRegistry` impls `arcgraph_storage::metrics::MetricsSink`.
    /// Verifies the trait bridge is wired and the trait-object call
    /// reaches the same underlying counters/histograms as the
    /// inherent methods.
    #[test]
    fn metrics_registry_satisfies_storage_metrics_sink_trait() {
        let r = Arc::new(MetricsRegistry::new().expect("init"));
        let sink: Arc<dyn arcgraph_storage::metrics::MetricsSink> = r.clone();
        sink.record_wal_write(WalWriteOutcome::T1Sync);
        sink.observe_wal_fsync_ms(0.75);
        sink.record_storage_page(StoragePageKind::Hit);
        // W28 #582 — the two new trait methods must reach the same
        // backing counters as the inherent methods.
        sink.record_hot_vertex_warning(TenantId::new(7));
        sink.record_query_plan_choice(QueryPlanType::Binary);

        // Inherent-method observe + trait-method observe must share
        // the same backing histogram / counter.
        let text = String::from_utf8(r.gather_text().expect("gather")).expect("utf-8");
        assert!(
            text.contains(r#"arcgraph_wal_writes_total{outcome="t1_sync"} 1"#),
            "trait-method record_wal_write must increment same counter; text was:\n{text}"
        );
        assert!(
            text.contains("arcgraph_wal_fsync_duration_ms_count 1"),
            "trait-method observe_wal_fsync_ms must increment count; text was:\n{text}"
        );
        assert!(
            text.contains(r#"arcgraph_storage_pages_total{kind="hit"} 1"#),
            "trait-method record_storage_page must increment same counter; text was:\n{text}"
        );
        assert!(
            text.contains(r#"arcgraph_hot_vertex_warnings_total{tenant="7"} 1"#),
            "trait-method record_hot_vertex_warning must increment same counter; text was:\n{text}"
        );
        assert!(
            text.contains(r#"arcgraph_query_plan_choice{plan_type="binary"} 1"#),
            "trait-method record_query_plan_choice must increment same counter; text was:\n{text}"
        );
    }

    /// W28 #582 — pin: `record_query_plan_choice` increments per
    /// `plan_type`, emitting the verbatim §10.2 line 723 label values.
    /// Strong oracle: exact (`==`) counter value per label (the
    /// prometheus text exposition renders exact integers).
    #[test]
    fn record_query_plan_choice_increments_per_plan_type() {
        let r = MetricsRegistry::new().expect("init");
        for _ in 0..4 {
            r.record_query_plan_choice(QueryPlanType::Binary);
        }
        r.record_query_plan_choice(QueryPlanType::Wcoj);

        let text = String::from_utf8(r.gather_text().expect("gather")).expect("utf-8");
        assert!(
            text.contains(r#"arcgraph_query_plan_choice{plan_type="binary"} 4"#),
            "binary plan_type counter must be 4; text was:\n{text}"
        );
        assert!(
            text.contains(r#"arcgraph_query_plan_choice{plan_type="wcoj"} 1"#),
            "wcoj plan_type counter must be 1; text was:\n{text}"
        );
        // free_join was never emitted → no cell materialises.
        assert!(
            !text.contains(r#"arcgraph_query_plan_choice{plan_type="free_join"}"#),
            "free_join must have no cell (never emitted); text was:\n{text}"
        );
    }

    // ─── ADR-202 — leiden_last_run_seconds ─────────────────────

    /// Extract the exposed `arcgraph_leiden_last_run_seconds{tenant=N}`
    /// value from the text exposition. Returns `None` when the series
    /// is absent.
    fn leiden_gauge_value(text: &str, tenant: u64) -> Option<i64> {
        let needle = format!(r#"arcgraph_leiden_last_run_seconds{{tenant="{tenant}"}} "#);
        text.lines()
            .find_map(|l| l.strip_prefix(&needle))
            .map(|v| v.trim().parse::<i64>().expect("gauge value parses as i64"))
    }

    /// ADR-202 D-2 — pin: the gauge holds a REAL Unix timestamp.
    /// Bounded real-value oracle (the strongest possible for a
    /// wall-clock metric): the exposed value must lie within the
    /// `[before, after]` Unix-second bracket of the recording call.
    /// This is exact, not relaxed — a duration-semantics or
    /// staleness-semantics implementation (the two wrong readings of
    /// the metric name) would expose a tiny value (≪ `before`) and
    /// fail loudly here.
    #[test]
    fn record_leiden_refresh_sets_unix_timestamp_within_call_bounds() {
        let r = MetricsRegistry::new().expect("init");

        let before = i64::try_from(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock after epoch")
                .as_secs(),
        )
        .expect("fits i64");
        r.record_leiden_refresh_success(TenantId::new(1));
        let after = i64::try_from(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock after epoch")
                .as_secs(),
        )
        .expect("fits i64");

        let text = String::from_utf8(r.gather_text().expect("gather")).expect("utf-8");
        let v = leiden_gauge_value(&text, 1).expect("series for tenant 1 must exist");
        assert!(
            (before..=after).contains(&v),
            "gauge must hold the recording-time Unix timestamp; got {v}, bounds [{before}, {after}]"
        );
    }

    /// ADR-202 D-3 — pin: per-tenant series independence. Recording
    /// tenant 1 materialises ONLY tenant 1's series; the alert
    /// contract `time() - gauge > 48h` therefore evaluates per
    /// tenant.
    #[test]
    fn record_leiden_refresh_is_per_tenant() {
        let r = MetricsRegistry::new().expect("init");
        r.record_leiden_refresh_success(TenantId::new(7));

        let text = String::from_utf8(r.gather_text().expect("gather")).expect("utf-8");
        assert!(
            leiden_gauge_value(&text, 7).is_some(),
            "tenant 7 series must exist; text was:\n{text}"
        );
        assert!(
            leiden_gauge_value(&text, 8).is_none(),
            "tenant 8 never refreshed → no series; text was:\n{text}"
        );
    }

    /// ADR-202 honesty guard — pin: a fresh registry exposes NO
    /// `arcgraph_leiden_last_run_seconds` series. The metric cannot
    /// claim a run that never happened (absent-until-first-success,
    /// D-6 restart semantics).
    #[test]
    fn leiden_gauge_absent_until_first_refresh() {
        let r = MetricsRegistry::new().expect("init");
        let text = String::from_utf8(r.gather_text().expect("gather")).expect("utf-8");
        // The HELP/TYPE header may legitimately be absent too (the
        // prometheus crate omits unmaterialised vecs entirely); the
        // load-bearing assertion is that no SERIES row exists.
        assert!(
            !text.contains(r#"arcgraph_leiden_last_run_seconds{"#),
            "no series may exist before the first successful refresh; text was:\n{text}"
        );
    }

    /// ADR-202 — pin: the trait route (`dyn RefreshObserver`, the
    /// exact shape the scheduler holds) lands on the same gauge as
    /// the inherent method, and repeated refreshes move the gauge
    /// forward (non-decreasing under a non-stepping clock).
    #[test]
    fn refresh_observer_trait_route_records_and_is_non_decreasing() {
        let r = Arc::new(MetricsRegistry::new().expect("init"));
        let observer: Arc<dyn CommunityRefreshObserver> = Arc::clone(&r) as _;

        observer.record_refresh_success(TenantId::new(3));
        let text1 = String::from_utf8(r.gather_text().expect("gather")).expect("utf-8");
        let v1 = leiden_gauge_value(&text1, 3).expect("series after first refresh");

        observer.record_refresh_success(TenantId::new(3));
        let text2 = String::from_utf8(r.gather_text().expect("gather")).expect("utf-8");
        let v2 = leiden_gauge_value(&text2, 3).expect("series after second refresh");

        assert!(
            v2 >= v1,
            "second refresh must not move the timestamp backward (v1={v1}, v2={v2})"
        );
    }
}
