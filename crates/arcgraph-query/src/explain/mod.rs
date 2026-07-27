//! M4-91 EXPLAIN / PROFILE pipeline.
//!
//! Lit at v1.0 per ADR-038 §2 D-19 + amendment-03 §TIER-1 GAP B.
//!
//! # Slice scope (M4-91)
//!
//! - **EXPLAIN** — fully implemented. Lowers `EXPLAIN <read_query>` to
//!   a planner-only path: parse → bind → type-check → cross-substrate
//!   validate → lower → cost (via the M4-51 walker) → project to
//!   [`PlanTree`]. **Does NOT acquire a snapshot LSN** (per D-18 rule
//!   1). The resulting [`PlanTree`] carries cost + cardinality +
//!   bindings + operator-specific annotations and renders deterministically
//!   via the [`std::fmt::Display`] impl.
//!
//! - **PROFILE** — fully implemented at W12γ. Lowers
//!   `PROFILE <read_query>` through the same plan-pipeline as
//!   EXPLAIN, then ALSO drives the executor + materialize tail per
//!   M4-08a (the W12γ M4-08a slice). Returns
//!   `(PlanTree, ExecutionMetrics)` per ADR-038 amendment-03
//!   §TIER-1 GAP B; the [`ExecutionMetrics`] slot's
//!   `wall_time_ms` + `rows_emitted` fields are populated end-to-end;
//!   `memory_bytes_high_water` stays `0` until M4-64a's
//!   `MemoryTracker` lands. Per-operator
//!   [`PlanTree`] annotations (per-op row counts / wall-time) are
//!   forward-deferred to M4-71's `RowCountObserver`. Caller-facing
//!   contract: PROFILE works for every read query EXPLAIN works on,
//!   subject to the executor's `NotImplemented` taxonomy at M4-63
//!   (aggregation / sort / etc.) — those plans surface
//!   `ExplainError::ArcQL(ArcQLError::NotImplemented { ... })` at
//!   the materialize call.
//!
//! # Public entry points
//!
//! - [`explain`] — free function consuming a query string + catalog
//!   reference. Strips the `EXPLAIN` AST wrapper if present, OR runs
//!   on a bare read query (matching production graph DBs that admit
//!   the entry point on any read query).
//! - [`profile`] — free function. Lit at W12γ; takes a query +
//!   catalog + executor substrate, runs the full pipeline (plan +
//!   execute + metrics), returns `(PlanTree, ExecutionMetrics)`.
//!   The signature gained an `&S: ExecutorSubstrate` parameter to
//!   light the executor-side; previous parser-only callers update
//!   call-sites accordingly.
//! - [`QueryEngine`] — thin `&CatalogProvider` wrapper exposing
//!   `explain` / `profile` / `execute` / `cancel` as methods, per
//!   amendment-03 §M5↔M4 contract surface. W12γ lights the `cancel`
//!   half via [`crate::cancel::CancellationRegistry`] and the
//!   `execute_with_deadline` variant per amendment-03 §TIER-1 GAP C.
//!
//! # Snapshot-LSN discipline
//!
//! EXPLAIN MUST NOT acquire a snapshot LSN per ADR-038 §2 D-18 rule 1:
//! the planner-only path is by design not on a transaction. The
//! current implementation routes through
//! [`crate::logical_plan::LogicalPlanLoweringVisitor::lower`] (default
//! `Lsn::MAX` — the "read-latest" sentinel; no actual LSN is taken at
//! the storage layer) then through the M4-51 cost walker (which calls
//! [`crate::semantic::CatalogProvider::snapshot`] once for cost-keying
//! consistency — that's a CATALOG-stats snapshot, not an MVCC LSN).
//! Pinned by the `explain_does_not_acquire_snapshot_lsn` integration
//! test that asserts the bound-query's reserved
//! `BoundQuery::snapshot_lsn` slot remains `None` after EXPLAIN.
//!
//! # Cross-PR coherence
//!
//! - The M4-51 cost walker's `capture_snapshot` IS called once via
//!   [`crate::planner::cost::estimate_costs`]; that captures the
//!   catalog stats snapshot (label cardinalities, total nodes, etc.)
//!   — a different entity from the MVCC snapshot LSN. The naming
//!   overlap is unfortunate but historical (predates ADR-041).
//!   EXPLAIN does NOT touch the storage substrate.
//!
//! # ADR provenance
//! - ADR-038 §2 D-19 — EXPLAIN/PROFILE clause contract.
//! - ADR-038 §2 D-18 — snapshot LSN binding (rule 1: EXPLAIN does
//!   not acquire).
//! - ADR-038 amendment-03 §TIER-1 GAP B — M4-91 sub-slice scope.
//! - ADR-038 amendment-03 §M5↔M4 contract surface —
//!   `QueryEngine::explain` / `QueryEngine::profile` shape.
//! - ADR-036 §D-25 — 5 ms M4-05 plan-build budget; EXPLAIN walks the
//!   plan exactly twice (cost + describe), well inside budget.

pub mod format;
pub mod plan_tree;

use std::sync::Arc;

use crate::ast::Statement;
use crate::error::Span;
use crate::executor::Value;
use crate::executor::eval::Parameters;
use crate::logical_plan::{
    LogicalPlanLoweringVisitor, rewrite_scan_to_property_index_scan,
    rewrite_unfiltered_count_to_count_store,
};
use crate::materialize::MaterializedResult;
use crate::parse;
use crate::planner::cache::{LookupOutcome, PlanCache, PlanCacheKey};
use crate::planner::cost::estimate_costs_with_frozen;
use crate::planner::enumeration::{FrozenCatalog, enumerate_join_order_with_frozen};
use crate::semantic::error::ArcQLError;
use crate::semantic::{BindingVisitor, CatalogProvider, CrossSubstrateValidator, TypeCheckVisitor};

pub use plan_tree::{PlanTree, PlanTreeOp};

pub(crate) const PLAN_ROW_COLUMNS: [&str; 5] =
    ["operator", "details", "est_cost", "est_rows", "depth"];

/// Per-query execution metrics surface.
///
/// # Field-population timeline
///
/// - **W12γ (M4-08a + M4-92, this slice)** — `wall_time_ms` +
///   `rows_emitted` are populated end-to-end by `crate::materialize`
///   per ADR-038 amendment-02 §M4.h; `memory_bytes_high_water` stays
///   `0` (forward to M4-64a's `MemoryTracker`).
/// - **W12β (M4-71, parallel slice)** — adds per-operator
///   `RowCountObserver` annotations on the [`PlanTree`]; the top-
///   level `ExecutionMetrics` shape stays stable.
/// - **M4-64a (forward)** — populates `memory_bytes_high_water` at
///   the executor-batch boundary.
///
/// Producing the type early (M4-91 stub) lets downstream surfaces
/// (M5-07 `graph.search` / M5-11 `graph.raw_query` / M5-13 Bolt) bind
/// to a fixed return type per amendment-03 §M5↔M4 contract surface —
/// W12γ wires the first two fields end-to-end; M4-64a + M4-71 wiring
/// is purely additive (populate the third field), never a breaking
/// type change.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExecutionMetrics {
    /// Total wall-time (ms) spent inside the executor batch loop,
    /// end-to-end. Populated by `crate::materialize` at W12γ;
    /// per-operator decomposition is forward-deferred to M4-71.
    pub wall_time_ms: u64,
    /// High-water-mark of bytes resident in the per-query memory
    /// budget. Forward-deferred to M4-64a's `MemoryTracker`; reports
    /// `0` at v1.0-alpha.
    pub memory_bytes_high_water: u64,
    /// Total rows materialized at the root of the plan. Populated
    /// by `crate::materialize` at W12γ.
    pub rows_emitted: u64,
}

/// Run EXPLAIN over an ArcQL source string.
///
/// Lowers `EXPLAIN <read_query>` (or a bare read query) through the
/// full planner-only pipeline:
///
/// 1. [`crate::parse`] → AST [`Statement`]
/// 2. Strip the `EXPLAIN` / `PROFILE` wrapper if present (per ADR-038
///    §2 D-19, EXPLAIN is a clause prefix; the inner read query is
///    what flows through the rest of the pipeline). PROFILE input
///    routes here too — both report the same plan tree; the
///    `is_profile` flag is what distinguishes execution intent.
/// 3. [`BindingVisitor::bind`] (M4-21)
/// 4. [`TypeCheckVisitor::check`] (M4-22)
/// 5. [`CrossSubstrateValidator::validate`] (M4-23)
/// 6. [`LogicalPlanLoweringVisitor::lower`] (M4-31..M4-33)
/// 7. [`crate::planner::cost::estimate_costs`] (M4-51)
/// 8. [`PlanTree::from_costed_plan`] (M4-91)
///
/// # No snapshot LSN
///
/// Per ADR-038 §2 D-18 rule 1, EXPLAIN does NOT acquire a snapshot LSN.
/// Steps 4-6 use `Lsn::MAX` (read-latest sentinel) at the LogicalPlan
/// level — but no storage substrate is contacted. The M4-21 binding
/// pass leaves `BoundQuery::snapshot_lsn = None`; this stays `None`
/// for the entire EXPLAIN run (pinned by the
/// `explain_does_not_acquire_snapshot_lsn` integration test).
///
/// # Errors
///
/// - [`ArcQLError::Binding`] — undeclared variable, unknown label / rel-type, ...
/// - [`ArcQLError::TypeCheck`] — operator-type mismatch, function
///   arity, ...
/// - [`ArcQLError::CrossSubstrate`] — substrate unavailable, RANK BY
///   HYBRID missing operand, ...
/// - [`ArcQLError::LogicalPlan`] — lowering reserved-variant rejection.
/// - [`ArcQLError::NotImplemented`] — unsupported operations.
///
/// Parse errors do NOT surface as `ArcQLError`; they bubble out as
/// [`crate::error::ParseError`]. Callers needing a unified error type
/// translate at the API boundary.
pub fn explain<C: CatalogProvider>(query: &str, catalog: &C) -> Result<PlanTree, ExplainError> {
    let stmt = parse(query).map_err(ExplainError::Parse)?;
    let inner_stmt = strip_explain_or_profile_wrapper(stmt);
    plan_tree_for(&inner_stmt, query, catalog, None).map_err(ExplainError::ArcQL)
}

fn lower_for_planning(
    bound: &crate::semantic::bound_ast::BoundStatement,
) -> Result<crate::logical_plan::LogicalPlan, Vec<ArcQLError>> {
    lower_for_planning_with_count_store(bound, true)
}

fn lower_for_planning_with_count_store(
    bound: &crate::semantic::bound_ast::BoundStatement,
    allow_count_store_rewrite: bool,
) -> Result<crate::logical_plan::LogicalPlan, Vec<ArcQLError>> {
    LogicalPlanLoweringVisitor::lower(bound).map(|plan| {
        if allow_count_store_rewrite {
            rewrite_unfiltered_count_to_count_store(plan)
        } else {
            plan
        }
    })
}

/// Run EXPLAIN over an ArcQL source string with an attached
/// per-tenant plan cache (M4-53; per ADR-038 amendment-03 §TIER-2-a).
///
/// Same pipeline as [`explain`] except the lower → enumerate → cost
/// half is short-circuited on a cache hit. Tenant identity flows
/// from `catalog.tenant()`; the cache MUST NOT be shared across
/// tenant boundaries other than through this entry point (the cache
/// itself enforces per-tenant LRU isolation per amendment-03
/// §TIER-2-a).
///
/// Snapshot-LSN discipline is preserved (per ADR-038 §2 D-18 rule 1):
/// the cache lookup reads only the catalog's `commits_observed`
/// stats-change watermark — not the MVCC snapshot LSN.
pub fn explain_with_cache<C: CatalogProvider>(
    query: &str,
    catalog: &C,
    cache: &PlanCache,
) -> Result<PlanTree, ExplainError> {
    let stmt = parse(query).map_err(ExplainError::Parse)?;
    let inner_stmt = strip_explain_or_profile_wrapper(stmt);
    plan_tree_for(&inner_stmt, query, catalog, Some(cache)).map_err(ExplainError::ArcQL)
}

/// PROFILE entry point — lit at W12γ.
///
/// Lowers `PROFILE <read_query>` (or a bare read query) through the
/// full plan-build pipeline (parse → bind → type-check → cross-
/// substrate → lower → enumerate joins → cost → [`PlanTree`]) and
/// THEN drives the executor + `crate::materialize` tail to produce
/// real [`ExecutionMetrics`].
///
/// # Snapshot-LSN discipline
///
/// PROFILE acquires the snapshot LSN LAZILY pre-first-batch (per
/// ADR-038 §2 D-18 rule 1 — the same execute-time path as `execute`),
/// unlike EXPLAIN which never acquires (rule 1's EXPLAIN exception).
///
/// # Per-operator annotations
///
/// At W12γ the [`PlanTree`] returned is the un-annotated plan tree
/// (same shape as EXPLAIN). Per-operator `row_count` / `wall_time_ms`
/// / `memory_bytes` annotations on each `PlanTreeOp` are forward-
/// deferred to W12β's M4-71 `RowCountObserver` slice. The TOP-level
/// [`ExecutionMetrics`] is populated end-to-end at this slice
/// (per amendment-03 §TIER-1 GAP B's "rows_emitted + wall_time_ms
/// at v1.0; per-operator at M4-71").
///
/// # Cancellation + per-query deadline
///
/// W12γ fix-up MED-2: PROFILE is symmetric with `execute` per ADR-
/// 038 §4.3 I-Q13 ("every v1.0 query is cancellable + per-query-
/// timeout-bounded"). The free function takes a
/// [`crate::cancel::CancellationRegistry`] reference (so an external
/// canceller can fire mid-PROFILE) and a [`std::time::Duration`]
/// deadline. The [`QueryEngine::profile`] method binds the engine's
/// registry and applies [`crate::DEFAULT_QUERY_TIMEOUT_MS`] (30s);
/// callers needing an explicit per-call override use this free
/// function directly.
///
/// # Errors
///
/// All [`ExplainError`] variants apply: `Parse` / `ArcQL` for
/// plan-time faults, `Cancelled` / `Substrate` / `ExecutionEval` for
/// executor-time faults (matching the per-arm translation in
/// `translate_execution_error`).
pub fn profile<C, S>(
    query: &str,
    catalog: &C,
    substrate: &S,
    registry: &crate::cancel::CancellationRegistry,
    deadline: std::time::Duration,
) -> Result<(PlanTree, ExecutionMetrics), ExplainError>
where
    C: CatalogProvider,
    S: crate::executor::ExecutorSubstrate,
{
    let stmt = parse(query).map_err(ExplainError::Parse)?;
    let inner_stmt = strip_explain_or_profile_wrapper(stmt);
    // Plan-time path: same as EXPLAIN, but we keep the costed plan so
    // we can drive the executor without re-walking the plan a second
    // time.
    let mut bound = BindingVisitor::bind(&inner_stmt, query, catalog).map_err(|errs| {
        ExplainError::ArcQL(first_or_internal(errs.into_iter().map(ArcQLError::from)))
    })?;
    TypeCheckVisitor::check(&mut bound, catalog)
        .map_err(|errs| ExplainError::ArcQL(first_or_internal_iter(errs)))?;
    CrossSubstrateValidator::validate(&bound, catalog)
        .map_err(|errs| ExplainError::ArcQL(first_or_internal_iter(errs)))?;
    let plan = lower_for_planning(&bound)
        .map_err(|errs| ExplainError::ArcQL(first_or_internal_iter(errs)))?;
    // #1366 (Phase 2): PROFILE executes the plan, so it must take the
    // same index path as execute — route indexed equality lookups here.
    let plan = rewrite_scan_to_property_index_scan(plan, catalog);
    let snapshot = catalog.snapshot();
    let frozen = FrozenCatalog::new(catalog, snapshot);
    let optimized = enumerate_join_order_with_frozen(plan, &frozen);
    // W25-M4-61b / ADR-097: resolve join-algorithm Auto → Hash / Merge
    // before costing + executing so PROFILE reports the picked
    // algorithm. Pass `&frozen` so the picker's snapshot lookup
    // returns the already-captured snapshot.
    let optimized = crate::planner::pick_join_algorithms(optimized, &frozen);
    let costed = estimate_costs_with_frozen(optimized.clone(), &frozen);
    let plan_tree = PlanTree::from_costed_plan(&costed);
    // Executor path: drive the same enumerated/optimized plan
    // through the M4-61 executor + M4-08a materialize tail. W12γ
    // fix-up MED-2: register against the supplied registry + spawn
    // the deadline timer so PROFILE is symmetric with execute under
    // I-Q13.
    let qid = crate::QueryId::new();
    let ctx = crate::executor::ExecutionContext::with_query_id(
        catalog.tenant(),
        catalog.partition(),
        qid,
    );
    let token = ctx.cancellation().clone();
    registry.register_with_token(qid, token.clone());
    let _deadline_handle = crate::cancel::spawn_deadline_timer(token, deadline);
    let _guard = RegistryGuard::new(registry, qid);
    let mat = crate::materialize::materialize(&optimized, substrate, &ctx)
        .map_err(translate_execution_error)?;
    Ok((plan_tree, mat.metrics))
}

/// Forwarding wrapper around `explain` / `profile` / `crate::materialize`
/// that fulfills the amendment-03 §M5↔M4 contract surface and
/// `QueryEngine` shape per ADR-038 amendment-02 §M4.h + amendment-03
/// §TIER-1 GAP B + §TIER-1 GAP C.
///
/// W12γ rounds out the surface:
/// - `explain` (M4-91 baseline) — plan-tree-only path.
/// - `profile` (M4-91 + W12γ) — plan-tree + executor-driven metrics.
/// - `execute` (M4-08a) — full materialization to
///   [`crate::MaterializedResult`].
/// - `execute_with_deadline` (M4-92) — same as `execute` plus a
///   deadline-bounded cancellation gate.
/// - `cancel(query_id)` (M4-92) — fires the cancellation token
///   registered for `query_id`, no-op on miss.
///
/// Wraps a borrowed [`CatalogProvider`] so the engine is cheap to
/// construct + has no `'static` requirement on the catalog (production
/// catalogs are typically backed by `Arc<dyn CatalogProvider>` per
/// crate boundary; this wrapper lets tests use stack-allocated
/// catalogs without lifetime gymnastics).
pub struct QueryEngine<'cat, C: CatalogProvider> {
    catalog: &'cat C,
    /// M4-53: optional per-tenant plan cache. When present, EXPLAIN
    /// short-circuits the lower → enumerate → cost half on cache hit
    /// per ADR-038 amendment-03 §TIER-2-a. When absent (default for
    /// short-lived / one-shot engines), every EXPLAIN re-plans —
    /// matching pre-M4-53 behavior.
    cache: Option<Arc<PlanCache>>,
    /// M4-92: cancellation registry. Default is a fresh per-engine
    /// registry; multi-engine deployments (M5-12 forward) share an
    /// `Arc`-backed registry across engines so a Bolt-side
    /// `RESET(query_id)` frame routes to the correct in-flight
    /// query irrespective of which engine instance is running it.
    cancellation: crate::cancel::CancellationRegistry,
    /// #1291 — optional per-tenant memory budget threaded into every
    /// [`crate::executor::ExecutionContext`] this engine constructs.
    /// `None` (the default — embedded / library posture) preserves the
    /// v1.0-alpha opt-in behavior: contexts get an unbounded
    /// [`crate::executor::MemoryBudget`] and operators fall back to the
    /// row-count runaway guard. The SERVED binary attaches a budget with
    /// a configured per-tenant byte cap (via
    /// [`crate::executor::MemoryBudget::set_per_tenant_cap`]) so a heavy
    /// query surfaces `ArcQLError::ResourceExhausted` instead of OOMing
    /// the process. Cloning the budget shares its inner state (`Arc`),
    /// so per-tenant accounting is consistent across every context this
    /// engine mints.
    budget: Option<crate::executor::MemoryBudget>,
}

impl<'cat, C: CatalogProvider> QueryEngine<'cat, C> {
    /// Construct a `QueryEngine` over the given catalog.
    ///
    /// Tenant identity flows from `catalog.tenant()` per
    /// [`CatalogProvider`]; v1.0 binds 1:1 (one engine per tenant).
    /// Multi-tenant routing (one `QueryEngine` over many catalogs)
    /// lights at the M5-12 rate-limit / per-tenant-pool slice.
    ///
    /// No plan cache is attached by default; callers that want caching
    /// must explicitly call [`Self::with_cache`]. A fresh
    /// per-engine [`crate::cancel::CancellationRegistry`] is created
    /// implicitly; multi-engine deployments share a registry via
    /// [`Self::with_cancellation_registry`].
    #[must_use]
    pub fn new(catalog: &'cat C) -> Self {
        Self {
            catalog,
            cache: None,
            cancellation: crate::cancel::CancellationRegistry::new(),
            budget: None,
        }
    }

    /// Attach a per-tenant plan cache to this engine.
    ///
    /// The cache is shared via [`Arc`] so multiple per-tenant engines
    /// (M5-12 forward) can route through the same cache while
    /// preserving the per-tenant LRU isolation per ADR-038
    /// amendment-03 §TIER-2-a.
    #[must_use]
    pub fn with_cache(mut self, cache: Arc<PlanCache>) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Borrow the attached cache, if any. Returns `None` when the
    /// engine was constructed without [`Self::with_cache`].
    #[must_use]
    pub fn cache(&self) -> Option<&PlanCache> {
        self.cache.as_deref()
    }

    /// Replace the per-engine cancellation registry with a shared
    /// [`crate::cancel::CancellationRegistry`]. Used by multi-engine
    /// (M5-12 forward) deployments where one router holds many per-
    /// tenant `QueryEngine` instances and a single Bolt-side
    /// `RESET(query_id)` frame must route correctly across them.
    ///
    /// At v1.0-alpha the default per-engine registry is sufficient;
    /// the [`Self::with_cancellation_registry`] method is the M5-12
    /// integration seam.
    #[must_use]
    pub fn with_cancellation_registry(
        mut self,
        registry: crate::cancel::CancellationRegistry,
    ) -> Self {
        self.cancellation = registry;
        self
    }

    /// Borrow the cancellation registry. Used by tests + the M5-12
    /// integration seam to introspect in-flight registrations.
    #[must_use]
    pub fn cancellation_registry(&self) -> &crate::cancel::CancellationRegistry {
        &self.cancellation
    }

    /// #1291 — attach a [`crate::executor::MemoryBudget`] threaded into
    /// every [`crate::executor::ExecutionContext`] this engine
    /// constructs (execute / execute-in-txn / execute-multi / profile
    /// paths).
    ///
    /// The caller configures per-tenant byte caps on the budget via
    /// [`crate::executor::MemoryBudget::set_per_tenant_cap`] BEFORE (or
    /// after — the cap read is per-reservation) attaching it. This is
    /// the served-binary enablement seam for the M4-64a per-tenant
    /// memory budget: without it, contexts default to an unbounded
    /// budget and blocking operators fall back to the
    /// `UNCAPPED_RUNAWAY_GUARD_ROWS` (≈4.29 B rows) runaway guard —
    /// effectively no ceiling (#1291).
    ///
    /// Cost (under the performance-budget discipline): one `Option` check + one `Arc`
    /// clone per query — nanoseconds against a ≥ parse+plan+execute
    /// query lifetime; no per-batch overhead is added by THIS seam
    /// (operators already consult the context's budget).
    #[must_use]
    pub fn with_memory_budget(mut self, budget: crate::executor::MemoryBudget) -> Self {
        self.budget = Some(budget);
        self
    }

    /// #1291 — borrow the attached memory budget, if any. Returns
    /// `None` when the engine was constructed without
    /// [`Self::with_memory_budget`] (embedded / library posture).
    #[must_use]
    pub fn memory_budget(&self) -> Option<&crate::executor::MemoryBudget> {
        self.budget.as_ref()
    }

    /// #1291 — apply the engine-attached budget (if any) to a freshly
    /// constructed [`crate::executor::ExecutionContext`]. Identity when
    /// no budget is attached (the context keeps its default unbounded
    /// budget — byte-for-byte the pre-#1291 behavior).
    fn apply_memory_budget(
        &self,
        ctx: crate::executor::ExecutionContext,
    ) -> crate::executor::ExecutionContext {
        match &self.budget {
            Some(budget) => ctx.with_budget(budget.clone()),
            None => ctx,
        }
    }

    /// Cancel the in-flight query identified by `query_id`. Returns
    /// `true` if the query was registered (and the token was fired);
    /// `false` if the query was not registered (already completed or
    /// never started). Per amendment-03 §M5↔M4 contract surface, this
    /// is the canonical "best-effort cancellation" entry-point that
    /// M5-07 / M5-11 / M5-13 bind to.
    ///
    /// Idempotent: a second `cancel` on the same `query_id` is a
    /// no-op on the underlying token (already tripped) but still
    /// returns `true` while the registry entry is present.
    pub fn cancel(&self, query_id: crate::QueryId) -> bool {
        self.cancellation.cancel(query_id)
    }

    /// EXPLAIN over `query`. See module-level [`explain`] free
    /// function for the canonical pipeline doc-link. Routes through
    /// [`explain_with_cache`] when a plan cache is attached.
    pub fn explain(&self, query: &str) -> Result<PlanTree, ExplainError> {
        match self.cache.as_deref() {
            Some(cache) => explain_with_cache(query, self.catalog, cache),
            None => explain(query, self.catalog),
        }
    }

    /// PROFILE over `query`. W12γ wires the executor + metrics tail;
    /// returns `(PlanTree, ExecutionMetrics)` per amendment-03 §TIER-1
    /// GAP B. See module-level [`profile`] free function for the
    /// canonical pipeline doc.
    ///
    /// W12γ fix-up MED-2: forwards the engine's
    /// [`crate::cancel::CancellationRegistry`] and applies
    /// [`crate::DEFAULT_QUERY_TIMEOUT_MS`] (30s) per ADR-038 §4.3
    /// I-Q13 (PROFILE is symmetric with `execute` under the v1.0
    /// "every query is cancellable + per-query-timeout-bounded"
    /// contract). Callers needing an explicit deadline use the free
    /// [`profile`] function directly.
    pub fn profile<S>(
        &self,
        query: &str,
        substrate: &S,
    ) -> Result<(PlanTree, ExecutionMetrics), ExplainError>
    where
        S: crate::executor::ExecutorSubstrate,
    {
        profile(
            query,
            self.catalog,
            substrate,
            &self.cancellation,
            std::time::Duration::from_millis(crate::cancel::DEFAULT_QUERY_TIMEOUT_MS),
        )
    }

    /// PROFILE over `query` + execute against `substrate` and annotate
    /// the plan tree with per-operator row counts + wall-time + memory
    /// high-water per ADR-038 amendment-03 §TIER-1 GAP B.
    ///
    /// # Pipeline
    ///
    /// 1. Parse + bind + type-check + cross-substrate validate.
    /// 2. Cache lookup (M4-53 hit path skips lower/enumerate/cost).
    /// 3. Cold-path lower → enumerate → cost (cache miss / stale).
    /// 4. **Cache insert** — PROFILE that runs the planner DOES populate the
    ///    cache. Pinned by `profile_populates_plan_cache_via_explain_path`
    ///    integration test.
    /// 5. Build a [`crate::observer::RowCountObserver`] anchored to the
    ///    costed plan; attach to a fresh [`crate::executor::ExecutionContext`].
    /// 6. Build the physical pipeline + drive `next_batch` to
    ///    completion. Snapshot LSN is acquired LAZILY at first batch
    ///    per amendment-03 §TIER-1 GAP E rule 1 (lazy capture; rule 2
    ///    is the distinct multi-statement LSN-sharing rule per M4-83).
    /// 7. Project observer state into [`ExecutionMetrics`] + return
    ///    `(PlanTree, ExecutionMetrics)`.
    ///
    /// # Errors
    ///
    /// Same taxonomy as [`Self::execute`]: Parse / ArcQL / Cancelled /
    /// Substrate / ExecutionEval. PROFILE-specific quirks: a runtime
    /// `ExecutionEval` mid-PROFILE surfaces the same way as a runtime
    /// `Eval` mid-EXECUTE (the per-operator metrics observed before
    /// the eval failure are discarded — the function returns Err).
    pub fn profile_with_substrate<S>(
        &self,
        query: &str,
        substrate: &S,
    ) -> Result<(PlanTree, ExecutionMetrics), ExplainError>
    where
        S: crate::executor::ExecutorSubstrate,
    {
        let stmt = parse(query).map_err(ExplainError::Parse)?;
        let inner_stmt = strip_explain_or_profile_wrapper(stmt);
        let mut bound = BindingVisitor::bind(&inner_stmt, query, self.catalog).map_err(|errs| {
            ExplainError::ArcQL(first_or_internal(errs.into_iter().map(ArcQLError::from)))
        })?;
        TypeCheckVisitor::check(&mut bound, self.catalog)
            .map_err(|errs| ExplainError::ArcQL(first_or_internal_iter(errs)))?;
        CrossSubstrateValidator::validate(&bound, self.catalog)
            .map_err(|errs| ExplainError::ArcQL(first_or_internal_iter(errs)))?;
        // Cache + plan: per Sin #5 decision, PROFILE populates the
        // cache via the same pipeline as EXPLAIN. Cache miss runs the
        // cold path; cache hit reuses the costed plan.
        let snapshot = self.catalog.snapshot();
        let stats_version = snapshot.commits_observed();
        let frozen = FrozenCatalog::new(self.catalog, snapshot);
        let costed: Arc<crate::planner::cost::CostedPlan> = if let Some(cache) =
            self.cache.as_deref()
        {
            let key = PlanCacheKey::from_ast(self.catalog.tenant(), &inner_stmt);
            match cache.lookup(&key, stats_version) {
                LookupOutcome::Hit(cached) => cached,
                LookupOutcome::Miss | LookupOutcome::Stale | LookupOutcome::InvariantViolation => {
                    let plan = lower_for_planning(&bound)
                        .map_err(|errs| ExplainError::ArcQL(first_or_internal_iter(errs)))?;
                    // #1366 (Phase 2): route the indexed point lookup so
                    // the cached costed plan carries the index path
                    // (PROFILE executes it).
                    let plan = rewrite_scan_to_property_index_scan(plan, self.catalog);
                    let optimized = enumerate_join_order_with_frozen(plan, &frozen);
                    // W25-M4-61b / ADR-097: pick algorithm before
                    // costing so the cached costed plan carries the
                    // resolved Hash / Merge choice. `&frozen` reuses
                    // the captured snapshot (single-snapshot discipline).
                    let optimized = crate::planner::pick_join_algorithms(optimized, &frozen);
                    let costed = Arc::new(estimate_costs_with_frozen(optimized, &frozen));
                    cache.insert(key, Arc::clone(&costed), stats_version);
                    costed
                }
            }
        } else {
            let plan = lower_for_planning(&bound)
                .map_err(|errs| ExplainError::ArcQL(first_or_internal_iter(errs)))?;
            let plan = rewrite_scan_to_property_index_scan(plan, self.catalog);
            let optimized = enumerate_join_order_with_frozen(plan, &frozen);
            let optimized = crate::planner::pick_join_algorithms(optimized, &frozen);
            Arc::new(estimate_costs_with_frozen(optimized, &frozen))
        };
        // Build observer + run the executor with it attached. Per
        // amendment-03 §TIER-1 GAP E rule 1 (lazy capture — rule 2 is
        // multi-statement LSN-sharing per M4-83), the LSN is acquired
        // LAZILY at first batch via `ExecutionContext::ensure_snapshot_lsn`.
        let observer = std::sync::Arc::new(crate::observer::RowCountObserver::from_plan_and_costs(
            costed.plan(),
            costed.costs(),
        ));
        let ctx = self.apply_memory_budget(
            crate::executor::ExecutionContext::new(self.catalog.tenant(), self.catalog.partition())
                .with_observer(std::sync::Arc::clone(&observer)),
        );
        let _rows = crate::executor::execute_with_context(costed.plan(), substrate, &ctx)
            .map_err(translate_execution_error)?;
        let plan_tree = PlanTree::from_costed_plan(&costed);
        let metrics = observer.execution_metrics();
        Ok((plan_tree, metrics))
    }

    /// EXECUTE over `query` against `substrate`. Per M4-61 / M4-62 /
    /// M4-08a / M4-92 (ADR-038 amendment-02 §M4.f + §M4.h + amendment-
    /// 03 §TIER-1 GAP B/C/D/E + §TIER-2-b/c).
    ///
    /// Routes through the full planner pipeline (parse → bind →
    /// type-check → cross-substrate validate → lower → enumerate
    /// joins) plus the M4-61 vectorized executor + M4-08a materialize
    /// tail. The snapshot LSN is acquired LAZILY pre-first-batch per
    /// ADR-038 §2 D-18 rule 1; EXPLAIN's no-LSN discipline (also rule
    /// 1's EXPLAIN exception) is NOT touched by this path.
    ///
    /// Returns [`crate::MaterializedResult`] (W12γ shape change per
    /// amendment-03 §M5↔M4 contract surface §11 D-9). The pre-W12γ
    /// `Vec<Vec<Value>>` shape is recoverable via
    /// [`crate::MaterializedResult::into_rows`] /
    /// [`crate::MaterializedResult::rows`].
    ///
    /// # Cancellation + per-query deadline
    ///
    /// W12γ M4-92 wires per-query cancellation: the executor's
    /// [`crate::executor::ExecutionContext`] cancellation token is
    /// registered against the engine's
    /// [`crate::cancel::CancellationRegistry`] under the
    /// [`crate::QueryId`]; a concurrent
    /// [`Self::cancel`] call fires the token, surfacing
    /// [`ExplainError::Cancelled`] at the next batch boundary. The
    /// registry entry is unregistered on query end (success, error,
    /// cancel, or panic — RAII guard) so the registry doesn't
    /// accumulate completed-query entries.
    ///
    /// W12γ fix-up MED-1: this entry-point applies
    /// [`crate::DEFAULT_QUERY_TIMEOUT_MS`] (30s) per ADR-038 §4.3
    /// I-Q13 ("every v1.0 query is cancellable + per-query-timeout-
    /// bounded"); a long-running or pathological query surfaces
    /// `ExplainError::Cancelled` after the default elapses. Use
    /// [`Self::execute_with_deadline`] for an explicit per-call
    /// override, or [`Self::execute_with_query_id`] for an
    /// unbounded-deadline + caller-id variant (test-seam only —
    /// production paths MUST go through `execute` or
    /// `execute_with_deadline` to honor I-Q13).
    ///
    /// # M5↔M4 contract surface
    ///
    /// Per amendment-03 §M5↔M4 contract, this entry point is the
    /// stable surface that future MCP / Bolt / HTTP entry points
    /// (M5-07 `graph.search` / M5-11 `graph.raw_query` / M5-13
    /// Bolt) bind to. v1.0-alpha materializes ALL rows; M4-82
    /// (forward) ships a sibling streaming cursor surface.
    ///
    /// # Errors
    ///
    /// - [`ExplainError::Parse`] for syntactic faults.
    /// - [`ExplainError::ArcQL`] for plan-time faults (binding, type-
    ///   check, cross-substrate, plan lowering, NotImplemented for
    ///   plan shapes the executor doesn't support yet — aggregation,
    ///   sort, etc., forward to M4-63).
    /// - [`ExplainError::Cancelled`] when the per-query cancellation
    ///   token trips at a batch boundary (explicit cancel OR the
    ///   default 30s deadline elapsed).
    /// - [`ExplainError::Substrate`] for substrate-access faults
    ///   (e.g., HNSW unavailable).
    /// - [`ExplainError::ExecutionEval`] for runtime evaluation
    ///   faults (division by zero, NaN comparison, NULL operand
    ///   reaching a non-NULL-tolerant context).
    ///
    /// W11Z fix-up MED-2 (PR #268 retro): pre-fix-up, every executor
    /// error round-tripped as
    /// `ArcQLError::NotImplemented { feature: "execute: ..." }` —
    /// hiding `Cancelled` and runtime `Eval` behind a "feature not
    /// implemented" diagnostic. The per-arm translation now preserves
    /// the M5↔M4 contract surface fidelity.
    pub fn execute<S>(
        &self,
        query: &str,
        substrate: &S,
    ) -> Result<crate::MaterializedResult, ExplainError>
    where
        S: crate::executor::ExecutorSubstrate,
    {
        // Mint a fresh QueryId here (UUIDv7) and route through the
        // explicit-id + default-deadline variant so the I-Q13
        // contract is honored on the canonical entry point. W12γ
        // fix-up MED-1: previously skipped the deadline timer.
        let qid = crate::QueryId::new();
        self.execute_with_query_id_and_deadline(
            qid,
            query,
            substrate,
            std::time::Duration::from_millis(crate::cancel::DEFAULT_QUERY_TIMEOUT_MS),
        )
    }

    /// EXECUTE with a caller-supplied [`crate::QueryId`].
    ///
    /// Lets the caller pre-mint the [`crate::QueryId`] before
    /// `execute_with_query_id` runs, so a sibling thread can call
    /// [`Self::cancel`] using a known id. This is the test seam the
    /// M4-92 integration tests bind to; M5-12 forward will use the
    /// same surface for caller-side request-id pinning (the MCP
    /// request-id == the executor query-id).
    ///
    /// Behavior matches [`Self::execute`] except (a) the QueryId is
    /// supplied externally, AND (b) NO default deadline applies —
    /// queries run until completion or explicit `cancel`. Production
    /// paths MUST go through [`Self::execute`] or
    /// [`Self::execute_with_deadline`] to honor the ADR-038 §4.3
    /// I-Q13 "every v1.0 query is timeout-bounded" contract; this
    /// surface is reserved for tests + the M5-12 caller-id-pinning
    /// integration (which adds its own deadline at the M5 layer).
    ///
    /// # Runtime guard (W12 retro INDEPENDENT REVIEW L1-NIT-7)
    ///
    /// This entry-point emits a `tracing::warn!` on every call so that
    /// a production caller that bypasses the I-Q13 contract is
    /// observable in the structured-log stream. We do NOT make the
    /// surface `#[cfg(any(test, doc))]` because the M5-12 + M4-71 +
    /// M4-72 integration paths legitimately need it for caller-id
    /// pinning; a strict cfg-gate would break those test surfaces.
    /// The `warn!` is the defensive runtime guard that makes the
    /// contract violation visible without breaking legitimate callers.
    pub fn execute_with_query_id<S>(
        &self,
        qid: crate::QueryId,
        query: &str,
        substrate: &S,
    ) -> Result<crate::MaterializedResult, ExplainError>
    where
        S: crate::executor::ExecutorSubstrate,
    {
        // W12 retro INDEPENDENT REVIEW L1-NIT-7: defensive runtime
        // guard so a caller bypassing the I-Q13 v1.0 timeout contract
        // is observable in the structured-log stream. The canonical
        // bounded path is `execute()` (mints fresh QueryId + applies
        // DEFAULT_QUERY_TIMEOUT_MS) or `execute_with_deadline()`
        // (caller-supplied deadline). This `_with_query_id` variant
        // is reserved for tests + the M5-12 caller-id-pinning
        // integration which adds its own deadline at the M5 layer.
        tracing::warn!(
            target: "arcgraph_query::explain",
            qid = ?qid,
            "execute_with_query_id called without deadline; this bypasses \
             I-Q13 v1.0 timeout contract (ADR-038 §4.3). Use execute() for \
             the canonical bounded path or execute_with_deadline() for an \
             explicit per-call deadline. This entry-point is reserved for \
             tests + M5-12 caller-id pinning (which adds its own deadline)."
        );
        if let Some(result) = self.try_explain_as_rows(query)? {
            return Ok(result);
        }
        let (optimized, columns) = self.plan_for_execute(query)?;
        // Construct an ExecutionContext bound to the supplied QueryId,
        // then register its cancellation token against the engine's
        // registry so a concurrent QueryEngine::cancel(qid) call can
        // fire it. #1291 — the engine-attached memory budget (if any)
        // rides along so per-tenant byte caps are enforced.
        let ctx = self.apply_memory_budget(crate::executor::ExecutionContext::with_query_id(
            self.catalog.tenant(),
            self.catalog.partition(),
            qid,
        ));
        self.cancellation
            .register_with_token(qid, ctx.cancellation().clone());
        // W12γ fix-up MED-3: RAII guard so registry entry releases on
        // panic too. The prior sequential `unregister` after
        // materialize ran on success / error / cancel paths — but a
        // panic during the materialize loop unwound past it, leaking
        // the entry. The guard's Drop impl runs on every exit.
        let _guard = RegistryGuard::new(&self.cancellation, qid);
        // #353 — stamp the user RETURN-alias column names onto the
        // result so the wire surfaces emit them (the MCP / Bolt
        // renderers read `MaterializedResult::columns`).
        crate::materialize::materialize(&optimized, substrate, &ctx)
            .map(|r| r.with_columns(columns))
            .map_err(translate_execution_error)
    }

    /// EXECUTE over `query` with a per-query deadline per ADR-038
    /// amendment-03 §TIER-1 GAP C + §2 D-17.
    ///
    /// Behaves like [`Self::execute`] except a background deadline
    /// timer (per [`crate::cancel::spawn_deadline_timer`]) fires the
    /// cancellation token after `deadline` elapses; the executor
    /// surfaces [`ExplainError::Cancelled`] at the next batch boundary.
    ///
    /// # v1.0 default
    ///
    /// v1.0-alpha exposes [`crate::DEFAULT_QUERY_TIMEOUT_MS`] (30s) as
    /// the canonical default; per-tenant overrides forward-bind to
    /// M5-12 rate-limit config.
    ///
    /// # Latency precision
    ///
    /// Fire-to-observed latency: `deadline` + (≤ batch wall-time) +
    /// (≤ OS jitter on the deadline timer thread, typically ≤ 10ms on
    /// macOS/Linux). Per ADR-036 §D-24 the 2048-row batch boundary is
    /// well inside the M5-12 cancel-latency budget.
    pub fn execute_with_deadline<S>(
        &self,
        query: &str,
        substrate: &S,
        deadline: std::time::Duration,
    ) -> Result<crate::MaterializedResult, ExplainError>
    where
        S: crate::executor::ExecutorSubstrate,
    {
        let qid = crate::QueryId::new();
        self.execute_with_query_id_and_deadline(qid, query, substrate, deadline)
    }

    /// **#797** — EXECUTE with a per-query deadline AND a parameter bag.
    /// The auto-commit (non-explicit-tx) entry point the Bolt RUN
    /// handler + the MCP `graph.raw_query` adapter bind to when the
    /// wire message carries a `parameters` map (`$name` bindings). Mints
    /// a fresh [`crate::QueryId`] then routes through
    /// [`Self::execute_with_query_id_and_deadline_and_parameters`].
    pub fn execute_with_deadline_and_parameters<S>(
        &self,
        query: &str,
        substrate: &S,
        deadline: std::time::Duration,
        parameters: Parameters,
    ) -> Result<crate::MaterializedResult, ExplainError>
    where
        S: crate::executor::ExecutorSubstrate,
    {
        let qid = crate::QueryId::new();
        self.execute_with_query_id_and_deadline_and_parameters(
            qid, query, substrate, deadline, parameters,
        )
    }

    /// **ADR-197 — EXECUTE a statement WITHIN a caller-held explicit
    /// transaction** (Bolt BEGIN…COMMIT). The statement's writes STAGE
    /// into `held` (no per-op commit) instead of auto-committing; the
    /// caller commits / aborts `held` later at the Bolt COMMIT /
    /// ROLLBACK message.
    ///
    /// Returns `(result, held)` — the moved-back transaction carries
    /// the buffered writes so the caller can run the NEXT statement in
    /// the same transaction (multi-statement BEGIN…COMMIT) or finalize
    /// it. On a query-layer error the transaction is still returned
    /// (the caller decides whether to abort it — the Bolt FSM moves to
    /// Failed and a RESET / ROLLBACK aborts).
    ///
    /// This is the EXPLICIT-mode counterpart to
    /// [`Self::execute_with_deadline`] (AUTO-COMMIT mode). The
    /// substrate write ops consult [`crate::executor::ExecutionContext::with_held_txn_mut`]
    /// to stage into `held`; read ops at v1.0-α still read at their own
    /// snapshot (cross-statement read-your-writes within one tx is
    /// forward-deferred — see ADR-197 §Open questions).
    pub fn execute_in_txn<S>(
        &self,
        query: &str,
        substrate: &S,
        held: Box<dyn crate::executor::substrate::HeldTxnHandle>,
        deadline: std::time::Duration,
    ) -> (
        Result<crate::MaterializedResult, ExplainError>,
        Box<dyn crate::executor::substrate::HeldTxnHandle>,
    )
    where
        S: crate::executor::ExecutorSubstrate,
    {
        // #797 — delegate to the parameter-aware impl with an empty bag.
        self.execute_in_txn_with_parameters(query, substrate, held, deadline, Parameters::new())
    }

    /// **#797 / ADR-197** — EXECUTE a statement WITHIN a caller-held
    /// explicit transaction, threading a per-query parameter bag. The
    /// EXPLICIT-mode counterpart to
    /// [`Self::execute_with_deadline_and_parameters`]; the Bolt
    /// `run_in_txn` handler binds to this when a RUN inside
    /// BEGIN…COMMIT carries `$name` parameters.
    pub fn execute_in_txn_with_parameters<S>(
        &self,
        query: &str,
        substrate: &S,
        held: Box<dyn crate::executor::substrate::HeldTxnHandle>,
        deadline: std::time::Duration,
        parameters: Parameters,
    ) -> (
        Result<crate::MaterializedResult, ExplainError>,
        Box<dyn crate::executor::substrate::HeldTxnHandle>,
    )
    where
        S: crate::executor::ExecutorSubstrate,
    {
        let qid = crate::QueryId::new();
        match self.try_explain_as_rows(query) {
            Ok(Some(result)) => return (Ok(result), held),
            Ok(None) => {}
            Err(e) => return (Err(e), held),
        }
        // Plan; on a plan-time error, return the untouched tx so the
        // caller can abort it.
        let (optimized, columns) = match self.plan_for_execute_in_explicit_txn(query) {
            Ok(p) => p,
            Err(e) => return (Err(e), held),
        };
        // #1291 — apply the engine-attached memory budget (if any) so
        // explicit-tx statements observe the same per-tenant byte cap
        // as the auto-commit path.
        let ctx = self.apply_memory_budget(
            crate::executor::ExecutionContext::with_query_id(
                self.catalog.tenant(),
                self.catalog.partition(),
                qid,
            )
            .with_held_txn(held)
            .with_parameters(parameters),
        );
        let token = ctx.cancellation().clone();
        self.cancellation.register_with_token(qid, token.clone());
        let _deadline_handle = crate::cancel::spawn_deadline_timer(token, deadline);
        let _guard = RegistryGuard::new(&self.cancellation, qid);
        // #353 — stamp the user RETURN-alias column names onto the
        // result (explicit-tx Bolt path; the `RunOutcome::fields`
        // renderer reads `MaterializedResult::columns`).
        let result = crate::materialize::materialize(&optimized, substrate, &ctx)
            .map(|r| r.with_columns(columns))
            .map_err(translate_execution_error);
        // Reclaim the (mutated) held tx so the caller can run the next
        // statement in it or commit / abort it. `take_held_txn` always
        // yields `Some` here because we installed it via
        // `with_held_txn` above and nothing else takes it.
        let held = ctx
            .take_held_txn()
            .expect("held tx installed by with_held_txn must be reclaimable");
        (result, held)
    }

    /// EXECUTE with a caller-supplied [`crate::QueryId`] and a per-
    /// query deadline. The deadline-bounded sibling of
    /// [`Self::execute_with_query_id`].
    pub fn execute_with_query_id_and_deadline<S>(
        &self,
        qid: crate::QueryId,
        query: &str,
        substrate: &S,
        deadline: std::time::Duration,
    ) -> Result<crate::MaterializedResult, ExplainError>
    where
        S: crate::executor::ExecutorSubstrate,
    {
        // #797 — delegate to the parameter-aware impl with an empty bag
        // (literal-only path; byte-for-byte the prior behavior).
        self.execute_with_query_id_and_deadline_and_parameters(
            qid,
            query,
            substrate,
            deadline,
            Parameters::new(),
        )
    }

    /// **#797** — EXECUTE with a caller-supplied [`crate::QueryId`], a
    /// per-query deadline, AND a per-query parameter bag (`$name` →
    /// [`crate::executor::Value`]).
    ///
    /// The canonical parameter-binding entry point: the parameter bag
    /// is installed on the [`crate::executor::ExecutionContext`] and
    /// resolved at runtime by the executor's evaluator (see
    /// [`crate::executor::Pipeline::build_with_parameters`]). Binding is
    /// a RUNTIME substitution, NOT a plan-time rewrite — so the M4-53
    /// plan cache key (`PlanCacheKey::from_ast`) stays param-agnostic
    /// and one cached costed plan serves every parameter value.
    pub fn execute_with_query_id_and_deadline_and_parameters<S>(
        &self,
        qid: crate::QueryId,
        query: &str,
        substrate: &S,
        deadline: std::time::Duration,
        parameters: Parameters,
    ) -> Result<crate::MaterializedResult, ExplainError>
    where
        S: crate::executor::ExecutorSubstrate,
    {
        if let Some(result) = self.try_explain_as_rows(query)? {
            return Ok(result);
        }
        let (optimized, columns) = self.plan_for_execute(query)?;
        // #1291 — apply the engine-attached memory budget (if any):
        // this is the canonical served path (MCP `graph.raw_query` +
        // Bolt auto-commit RUN), so the per-tenant byte cap gates here.
        let ctx = self.apply_memory_budget(
            crate::executor::ExecutionContext::with_query_id(
                self.catalog.tenant(),
                self.catalog.partition(),
                qid,
            )
            .with_parameters(parameters),
        );
        let token = ctx.cancellation().clone();
        self.cancellation.register_with_token(qid, token.clone());
        // W12γ fix-up MED-3: RAII guard for panic-safety (see sibling
        // `execute_with_query_id`). Rust drops in reverse declaration
        // order — `_guard` drops FIRST (releases the registry entry),
        // then `_deadline_handle` drops (timer thread observes
        // Disconnected, exits without firing). On panic-unwind both
        // run, preserving the no-leak invariant; the prior sequential
        // `unregister` did NOT.
        let _deadline_handle = crate::cancel::spawn_deadline_timer(token, deadline);
        let _guard = RegistryGuard::new(&self.cancellation, qid);
        // #353 — stamp the user RETURN-alias column names onto the
        // result so the MCP / Bolt wire renderers emit them. This is
        // the canonical deadline-bounded path the MCP `graph.raw_query`
        // adapter + the Bolt auto-commit `run` both route through.
        crate::materialize::materialize(&optimized, substrate, &ctx)
            .map(|r| r.with_columns(columns))
            .map_err(translate_execution_error)
    }

    /// Internal helper: parse + plan-build + enumerate joins for the
    /// execute / execute_with_deadline path. Shared so the
    /// cancellation registration dance lives in exactly one place.
    fn plan_for_execute(
        &self,
        query: &str,
    ) -> Result<(crate::logical_plan::LogicalPlan, Vec<String>), ExplainError> {
        self.plan_for_execute_optionally_costed(query, false)
            .map(|(plan, _cost, columns)| (plan, columns))
    }

    fn plan_for_execute_in_explicit_txn(
        &self,
        query: &str,
    ) -> Result<(crate::logical_plan::LogicalPlan, Vec<String>), ExplainError> {
        self.plan_for_execute_with_options(query, false)
    }

    /// Shared implementation of [`Self::plan_for_execute`] and
    /// [`Self::plan_and_cost_for_execute_for_test`]. Routes through the
    /// SAME parse → bind → typecheck → cross-substrate validate → lower
    /// → snapshot → frozen → enumerate pipeline, optionally followed by
    /// a cost walk against the same `FrozenCatalog`.
    ///
    /// Sharing the implementation between the production execute path
    /// (`compute_cost=false`) and the test-only cost accessor
    /// (`compute_cost=true`) eliminates the I-13 cost-equivalence
    /// test-side divergence-risk: any future refactor that changes how
    /// production captures its snapshot or wires the DP enumerator
    /// automatically applies to the test's "live planner" comparand too
    /// (closes PR #342 R1 §M-1).
    fn plan_for_execute_optionally_costed(
        &self,
        query: &str,
        compute_cost: bool,
    ) -> Result<(crate::logical_plan::LogicalPlan, Option<f64>, Vec<String>), ExplainError> {
        let stmt = parse(query).map_err(ExplainError::Parse)?;
        self.plan_for_execute_with_bound_options(query, stmt, true, compute_cost)
    }

    fn plan_for_execute_with_options(
        &self,
        query: &str,
        allow_count_store_rewrite: bool,
    ) -> Result<(crate::logical_plan::LogicalPlan, Vec<String>), ExplainError> {
        let stmt = parse(query).map_err(ExplainError::Parse)?;
        self.plan_for_execute_with_bound_options(query, stmt, allow_count_store_rewrite, false)
            .map(|(plan, _cost, columns)| (plan, columns))
    }

    fn plan_for_execute_with_bound_options(
        &self,
        query: &str,
        stmt: Statement,
        allow_count_store_rewrite: bool,
        compute_cost: bool,
    ) -> Result<(crate::logical_plan::LogicalPlan, Option<f64>, Vec<String>), ExplainError> {
        let inner_stmt = strip_explain_or_profile_wrapper(stmt);
        let mut bound = BindingVisitor::bind(&inner_stmt, query, self.catalog).map_err(|errs| {
            ExplainError::ArcQL(first_or_internal(errs.into_iter().map(ArcQLError::from)))
        })?;
        TypeCheckVisitor::check(&mut bound, self.catalog)
            .map_err(|errs| ExplainError::ArcQL(first_or_internal_iter(errs)))?;
        CrossSubstrateValidator::validate(&bound, self.catalog)
            .map_err(|errs| ExplainError::ArcQL(first_or_internal_iter(errs)))?;
        // #353 — derive the user-meaningful RETURN-alias column names
        // from the BOUND statement (the terminal RETURN / standalone
        // CALL-proc YIELD / SHOW columns), BEFORE lowering discards the
        // bound AST. These flow to `MaterializedResult::columns` and out
        // to the MCP / Bolt wire so consumers (langchain's Neo4jGraph,
        // any Bolt driver) key result records by the user's aliases
        // instead of synthesized `col_0..N`. Empty for a wildcard /
        // write-only / RETURN-less statement (the wire falls back to
        // `col_0..N` for the actual row width).
        let columns = crate::output_column_names(&bound);
        let plan = lower_for_planning_with_count_store(&bound, allow_count_store_rewrite)
            .map_err(|errs| ExplainError::ArcQL(first_or_internal_iter(errs)))?;
        // #1366 (Phase 2): route an equality on an Online-indexed
        // labelled property to a `PropertyIndexScan` (candidate lookup →
        // MVCC-verify) instead of the anchor `Scan + Filter`. Runs here
        // (post-lowering, with the catalog handle) before enumeration —
        // the RC-6 planner-visible gate lives in
        // `CatalogProvider::online_property_index`. A read-only fixture /
        // no-index tenant reports no index → the plan is unchanged.
        let plan = rewrite_scan_to_property_index_scan(plan, self.catalog);
        // Run join-ordering enumeration so the executor benefits
        // from cost-optimal plan shape (matches the EXPLAIN path's
        // single-FrozenCatalog discipline closing #261).
        let snapshot = self.catalog.snapshot();
        let frozen = crate::planner::enumeration::FrozenCatalog::new(self.catalog, snapshot);
        let optimized =
            crate::planner::enumeration::enumerate_join_order_with_frozen(plan, &frozen);
        // W25-M4-61b / ADR-097: cost-based picker resolves every
        // `LogicalJoin::algorithm = Auto` to a concrete Hash / Merge
        // pick using the SAME frozen catalog the enumerator + cost
        // walker consume — preserves the single-snapshot discipline
        // closing #261. Idempotent — joins already pinned to a
        // concrete algorithm pass through unchanged.
        let optimized = crate::planner::pick_join_algorithms(optimized, &frozen);
        if compute_cost {
            let costed = crate::planner::cost::estimate_costs_with_frozen(optimized, &frozen);
            let cost = costed.total_cost().total();
            let (returned_plan, _) = costed.into_parts();
            Ok((returned_plan, Some(cost), columns))
        } else {
            Ok((optimized, None, columns))
        }
    }

    fn try_explain_as_rows(
        &self,
        query: &str,
    ) -> Result<Option<crate::MaterializedResult>, ExplainError> {
        let stmt = parse(query).map_err(ExplainError::Parse)?;
        let Statement::Explain(inner) = stmt else {
            return Ok(None);
        };
        let inner_stmt = Statement::Read(inner);
        let plan = plan_tree_for(&inner_stmt, query, self.catalog, self.cache.as_deref())
            .map_err(ExplainError::ArcQL)?;
        Ok(Some(plan_tree_as_rows(&plan)))
    }

    /// Production-path twin of [`Self::plan_for_execute`] that ALSO
    /// returns the cost of the optimized plan, walked against the same
    /// `FrozenCatalog` the enumerator used. Exists so the I-13 cost-
    /// equivalence integration test
    /// (`tests/i13_cost_equivalence_under_fault.rs`) can verify that
    /// EXPLAIN's reported cost equals the cost the production planner
    /// would compute on this exact path (closes PR #342 R1 §M-1).
    ///
    /// # Discipline
    ///
    /// **Production code paths MUST NEVER call this.** Use
    /// [`Self::plan_for_execute`] instead, which avoids the cost-walk
    /// overhead the executor does not need. The `_for_test` suffix +
    /// `#[doc(hidden)]` mirror the
    /// [`enumerate_join_order_pick_max_for_proptest`] convention
    /// established at `crates/arcgraph-query/src/planner/enumeration/mod.rs`
    /// for test-only seams that integration tests need but production
    /// code MUST NOT touch.
    ///
    /// [`enumerate_join_order_pick_max_for_proptest`]: crate::planner::enumeration::enumerate_join_order_pick_max_for_proptest
    #[doc(hidden)]
    pub fn plan_and_cost_for_execute_for_test(
        &self,
        query: &str,
    ) -> Result<(crate::logical_plan::LogicalPlan, f64), ExplainError> {
        let (plan, cost, _columns) = self.plan_for_execute_optionally_costed(query, true)?;
        Ok((
            plan,
            cost.expect("compute_cost=true returns Some(cost) by construction"),
        ))
    }

    /// EXECUTE a multi-statement ArcQL query (`stmt1; stmt2; …`) per
    /// ADR-038 §5.4.1 closure (M4-83).
    ///
    /// Closes ADR-038 §5.4.1 multi-statement deferral. Lowers the chain
    /// through:
    ///
    /// 1. [`crate::parse_multi`] — `Vec<Statement>`.
    /// 2. [`BindingVisitor::bind_multi`] — cross-statement carry-over scoping.
    /// 3. Per-statement [`TypeCheckVisitor`] + [`CrossSubstrateValidator`] +
    ///    [`crate::logical_plan::LogicalPlanLoweringVisitor`] + DP join
    ///    enumeration (the same plan-time pipeline single-statement EXECUTE runs).
    /// 4. [`crate::materialize::materialize_multi`] — sequential `materialize`
    ///    invocations sharing ONE [`crate::executor::ExecutionContext`] so the
    ///    snapshot LSN captured at statement 1's first batch flows through every
    ///    subsequent statement (per amendment-03 §TIER-1 GAP E rule 2).
    ///
    /// Single-statement input is admissible (the chain may be of length
    /// 1); the returned `Vec<MaterializedResult>` always has the same
    /// length as the parsed statement list.
    ///
    /// # Cancellation + per-query deadline
    ///
    /// One [`crate::QueryId`] is minted per multi-statement call; one
    /// [`crate::cancel::CancellationRegistry`] entry covers the entire
    /// chain. The default 30s deadline applies to the WHOLE chain, not
    /// per-statement (matches the I-Q13 contract: cancel applies to a
    /// query, multi-statement is one query). Callers needing
    /// per-statement deadlines run `execute_with_deadline` per
    /// statement; that's the v1.1 surface forward-link.
    ///
    /// # Errors
    ///
    /// Same taxonomy as [`Self::execute`]; the first-failing statement
    /// short-circuits the chain.
    pub fn execute_multi<S>(
        &self,
        query: &str,
        substrate: &S,
    ) -> Result<Vec<crate::MaterializedResult>, ExplainError>
    where
        S: crate::executor::ExecutorSubstrate,
    {
        let qid = crate::QueryId::new();
        self.execute_multi_with_query_id_and_deadline(
            qid,
            query,
            substrate,
            std::time::Duration::from_millis(crate::cancel::DEFAULT_QUERY_TIMEOUT_MS),
        )
    }

    /// EXECUTE multi-statement with a caller-supplied [`crate::QueryId`] +
    /// per-query deadline. Test seam mirroring
    /// [`Self::execute_with_query_id_and_deadline`] for the multi-statement
    /// path (single-engine entry — multi-engine deployments share a registry
    /// per the M5-12 forward-link).
    pub fn execute_multi_with_query_id_and_deadline<S>(
        &self,
        qid: crate::QueryId,
        query: &str,
        substrate: &S,
        deadline: std::time::Duration,
    ) -> Result<Vec<crate::MaterializedResult>, ExplainError>
    where
        S: crate::executor::ExecutorSubstrate,
    {
        let plans_and_columns = self.plans_for_execute_multi(query)?;
        // Split into the plan vec (for `materialize_multi`) and the
        // per-statement column-name vec (#353), preserving order.
        let (plans, per_stmt_columns): (Vec<crate::logical_plan::LogicalPlan>, Vec<Vec<String>>) =
            plans_and_columns.into_iter().unzip();
        // #1291 — apply the engine-attached memory budget (if any) so
        // the multi-statement chain observes the same per-tenant byte
        // cap as the single-statement execute paths (the shared ctx
        // covers every statement in the chain).
        // #1291 — apply the engine-attached memory budget (if any) so
        // the multi-statement chain observes the same per-tenant byte
        // cap as the single-statement execute paths (the shared ctx
        // covers every statement in the chain).
        let ctx = self.apply_memory_budget(crate::executor::ExecutionContext::with_query_id(
            self.catalog.tenant(),
            self.catalog.partition(),
            qid,
        ));
        let token = ctx.cancellation().clone();
        self.cancellation.register_with_token(qid, token.clone());
        // RAII registry guard — releases entry on success / error / panic
        // (mirrors the single-statement path's W12γ MED-3 closure).
        let _deadline_handle = crate::cancel::spawn_deadline_timer(token, deadline);
        let _guard = RegistryGuard::new(&self.cancellation, qid);
        crate::materialize::materialize_multi(&plans, substrate, &ctx)
            .map(|results| {
                // #353 — stamp each statement's RETURN-alias column
                // names onto its result. `materialize_multi` returns
                // results in plan order, so a positional zip pairs each
                // result with its statement's columns.
                results
                    .into_iter()
                    .zip(per_stmt_columns)
                    .map(|(r, cols)| r.with_columns(cols))
                    .collect()
            })
            .map_err(translate_execution_error)
    }

    /// Internal helper: parse_multi + per-statement plan-build +
    /// enumerate joins. Shared so the cancellation registration dance
    /// lives in exactly one place. Mirrors [`Self::plan_for_execute`].
    ///
    /// Per amendment-03 §TIER-1 GAP E rule 2 the snapshot-LSN sharing
    /// is a CONTEXT-level invariant (lives on `ExecutionContext`), NOT
    /// a plan-level invariant — each plan is independent at the
    /// logical-plan layer; the shared LSN flows through at execute time
    /// via the shared `ctx` threaded into `materialize_multi`.
    ///
    /// # W13γ fix-up LOW-2 (closes review-pr-285-final.md LOW-2)
    ///
    /// Within-chain plan-cache invalidation is **NOT** exercised at v1.0:
    /// statement-N's M4-71 stats-feedback row-count breach invalidates
    /// the plan-cache entry for the same query shape, but
    /// statement-N+1 of the SAME chain has already had its plan lowered
    /// upfront by this method's per-statement loop. Cross-chain (next
    /// call to `execute_multi`) DOES benefit from the invalidation;
    /// within-chain does not. This is fine at v1.0 but a v1.1+
    /// inflection point.
    ///
    /// Forward-pin: issue #NEW M4-83 + M4-72 within-chain plan-cache
    /// invalidation couples to PlanCache trait extraction (deferred to v1.2+
    /// persistent-cache
    /// slice).
    fn plans_for_execute_multi(
        &self,
        query: &str,
    ) -> Result<Vec<(crate::logical_plan::LogicalPlan, Vec<String>)>, ExplainError> {
        let stmts = crate::parse_multi(query).map_err(ExplainError::Parse)?;
        // PROFILE keeps the historical execute-the-inner-query
        // behavior. EXPLAIN is rejected here instead of stripped: this
        // internal-only batch API has no mixed plan-row/execution row
        // contract, and failing closed prevents EXPLAIN-wrapped writes
        // from silently mutating in defense-in-depth.
        if stmts
            .iter()
            .any(|stmt| matches!(stmt, Statement::Explain(_)))
        {
            return Err(ExplainError::ArcQL(ArcQLError::NotImplemented {
                feature: "EXPLAIN is not supported inside a multi-statement execute batch; use QueryEngine::explain or a single execute call".into(),
                section: "ADR-210 multi-statement execute safety".into(),
                target_version: "v1.0-alpha".into(),
                span: Span::point(1, 1),
            }));
        }
        // Strip PROFILE wrappers for the internal multi-statement
        // execute path so PROFILE continues to execute the inner query.
        let inner_stmts: Vec<crate::ast::Statement> = stmts
            .into_iter()
            .map(strip_explain_or_profile_wrapper)
            .collect();
        // bind_multi handles cross-statement carry-over scoping per
        // ADR-038 §5.4.1 closure paragraph + amendment-03 §TIER-1
        // GAP E rule 2 (semantic-layer carry-over; LSN-sharing is
        // executor-layer).
        let mut bound_stmts = BindingVisitor::bind_multi(&inner_stmts, query, self.catalog)
            .map_err(|errs| {
                ExplainError::ArcQL(first_or_internal(errs.into_iter().map(ArcQLError::from)))
            })?;
        // Per-statement type-check + cross-substrate + lower +
        // enumerate. Capture ONE catalog snapshot for the whole chain
        // (same internal-consistency discipline as the single-statement
        // path's #261 closure — every per-statement DP enumeration +
        // cost walker reads the same `FrozenCatalog`).
        let snapshot = self.catalog.snapshot();
        let frozen = crate::planner::enumeration::FrozenCatalog::new(self.catalog, snapshot);
        let mut plans: Vec<(crate::logical_plan::LogicalPlan, Vec<String>)> =
            Vec::with_capacity(bound_stmts.len());
        for bound in bound_stmts.iter_mut() {
            TypeCheckVisitor::check(bound, self.catalog)
                .map_err(|errs| ExplainError::ArcQL(first_or_internal_iter(errs)))?;
            CrossSubstrateValidator::validate(bound, self.catalog)
                .map_err(|errs| ExplainError::ArcQL(first_or_internal_iter(errs)))?;
            // #353 — derive each statement's column names from its
            // bound form before lowering. The last statement's columns
            // are what a multi-statement caller sees as the final
            // result shape; per-statement results each carry their own.
            let columns = crate::output_column_names(bound);
            let plan = lower_for_planning(bound)
                .map_err(|errs| ExplainError::ArcQL(first_or_internal_iter(errs)))?;
            // #1366 (Phase 2): route indexed point lookups per statement.
            let plan = rewrite_scan_to_property_index_scan(plan, self.catalog);
            let optimized =
                crate::planner::enumeration::enumerate_join_order_with_frozen(plan, &frozen);
            // W25-M4-61b / ADR-097: resolve Auto algorithm per
            // multi-statement plan so each downstream executor pull
            // sees a concrete pick. `&frozen` reuses the captured
            // multi-statement snapshot.
            let optimized = crate::planner::pick_join_algorithms(optimized, &frozen);
            plans.push((optimized, columns));
        }
        Ok(plans)
    }
}

/// Walk a [`PlanTree`] into a [`MaterializedResult`] whose rows describe
/// the plan in pre-order, one row per operator.
///
/// Each row is `[op, details, estimated_cost, estimated_card, depth]`
/// (columns pinned by `PLAN_ROW_COLUMNS`). This is the #952 plan-row
/// adapter shape: the same shape the top-level `EXPLAIN` statement
/// materializes. The walk is a single deterministic pre-order traversal
/// (children in source order, depth increasing from 0 at the root).
///
/// # Why `pub`
///
/// The MCP `graph.raw_query` `explain:true` verb-consolidation mode
/// (operator-ruled — stays at the ADR-004 10-tool cap, no new tool)
/// reuses this exact walk to serialize the plan tree returned by the
/// free [`explain`] fn into the `RawQueryRows` wire envelope. Exposing
/// the tested walk (rather than replicating it MCP-side) keeps the
/// `details_string()` formatter + `Cost::total()` / `Cardinality::rows()`
/// accessors private to this crate while still giving the consumer the
/// canonical plan-row projection.
#[must_use]
pub fn plan_tree_as_rows(plan: &PlanTree) -> MaterializedResult {
    fn walk(node: &PlanTree, depth: i64, rows: &mut Vec<Vec<Value>>) {
        rows.push(vec![
            Value::String(node.op.name().into()),
            Value::String(node.details_string()),
            Value::Float(node.estimated_cost.total()),
            Value::Float(node.estimated_card.rows()),
            Value::Integer(depth),
        ]);
        for child in &node.children {
            walk(child, depth + 1, rows);
        }
    }

    let mut rows = Vec::new();
    walk(plan, 0, &mut rows);
    MaterializedResult {
        rows,
        ..Default::default()
    }
    .with_columns(PLAN_ROW_COLUMNS.iter().map(|s| (*s).to_string()).collect())
}

/// W12γ fix-up MED-3: RAII Drop guard that unconditionally releases a
/// [`crate::cancel::CancellationRegistry`] entry on scope exit
/// (success, error, cancel, OR panic-unwind). Replaces the prior
/// sequential `registry.unregister(qid)` statement, which the panic
/// path skipped — leaving the registry entry behind and growing the
/// in-flight count without bound under recurring substrate panics.
///
/// Per `feedback_seqlock_panic_safety_primitive.md`, the canonical
/// panic-safety primitive is RAII (sister approach to crud.rs +
/// stats_rebuild.rs's `catch_unwind` + `AssertUnwindSafe` + manual
/// observe). Here the registry semantics are simpler — a single
/// "remove the qid" mutation — so a Drop guard is the idiomatic shape.
///
/// # Idempotence
///
/// `CancellationRegistry::unregister` is already idempotent on miss
/// (returns `false` on the second call). The guard does not gate on
/// "already unregistered"; if a future caller adds a manual
/// `unregister` before the guard scope exits, the second drop is a
/// no-op.
struct RegistryGuard<'a> {
    registry: &'a crate::cancel::CancellationRegistry,
    qid: crate::QueryId,
}

impl<'a> RegistryGuard<'a> {
    fn new(registry: &'a crate::cancel::CancellationRegistry, qid: crate::QueryId) -> Self {
        Self { registry, qid }
    }
}

impl<'a> Drop for RegistryGuard<'a> {
    fn drop(&mut self) {
        self.registry.unregister(self.qid);
    }
}

/// W11Z fix-up MED-2 (PR #268 retro): translate an
/// [`crate::executor::ExecutionError`] into [`ExplainError`] preserving
/// per-arm diagnostic context.
///
/// Per the M5↔M4 contract surface (per ADR-038 amendment-03 §M5↔M4),
/// each `ExecutionError` arm carries distinct diagnostic semantics
/// that the MCP / Bolt / HTTP renderers want to surface separately:
///
/// - `Cancelled` → `ExplainError::Cancelled` — Bolt response framing
///   wants a CANCELLATION frame, not a generic error frame.
/// - `Substrate(s)` → `ExplainError::Substrate(s)` — surfaces the
///   "vector index unavailable" detail to the caller.
/// - `Plan(arcql)` → `ExplainError::ArcQL(arcql)` — the executor
///   re-emitted a plan-time error; round-trip the original.
/// - `NotImplemented { ... }` → `ExplainError::ArcQL(ArcQLError::
///   NotImplemented { ... })` — preserves the executor-side
///   `target_slice` as `target_version`; preserves `section`.
/// - `Eval(s)` → `ExplainError::ExecutionEval(s)` — runtime fault
///   distinct from plan-time type errors.
///
/// Previously the dispatch was a coarse `NotImplemented` blanket that
/// hid every variant behind a "feature not implemented" diagnostic —
/// the MED-2 finding fix.
fn translate_execution_error(e: crate::executor::ExecutionError) -> ExplainError {
    use crate::executor::ExecutionError as Exec;
    match e {
        Exec::Cancelled => ExplainError::Cancelled,
        Exec::Substrate(s) => ExplainError::Substrate(s),
        Exec::Plan(arcql) => ExplainError::ArcQL(arcql),
        // The direct executor API retains the structured spill taxonomy.
        // ExplainError predates OOC scratch and has no typed spill arm, so
        // this higher rendering boundary preserves the complete diagnostic
        // text as a server-side execution fault.
        Exec::Spill(spill) => ExplainError::ExecutionEval(spill.to_string()),
        Exec::NotImplemented {
            feature,
            target_slice,
            section,
        } => ExplainError::ArcQL(ArcQLError::NotImplemented {
            feature,
            section,
            target_version: target_slice,
            span: crate::error::Span::point(1, 1),
        }),
        Exec::Eval(s) => ExplainError::ExecutionEval(s),
        // #797 — preserve the CLIENT-fault classification across the
        // executor → explain boundary so the wire surfaces emit a
        // client error (Bolt ParameterMissing / MCP -32602), not the
        // server-fault bucket the generic `ExecutionEval` arm renders.
        Exec::MissingParameter { name } => ExplainError::MissingParameter { name },
    }
}

/// Public error type for the EXPLAIN / PROFILE / EXECUTE entry points.
///
/// Variant taxonomy:
///
/// - [`Self::Parse`] — pre-bind syntactic failure. The wrapped
///   [`crate::error::ParseError`] carries `Span::point` coordinates
///   into the source.
/// - [`Self::ArcQL`] — semantic / lowering / cost-walk failure (or
///   the M4-91 PROFILE NotImplemented stub). The wrapped
///   [`ArcQLError`] carries its own span via
///   [`ArcQLError::span_byte_range`].
/// - [`Self::Cancelled`] — the per-query cancellation token tripped
///   during `QueryEngine::execute`. Distinct from any plan-time
///   error.
/// - [`Self::Substrate`] — substrate-access fault during
///   `QueryEngine::execute` (e.g., HNSW unavailable, BM25 index
///   missing). v1.0-alpha stub substrates only surface
///   [`crate::executor::SubstrateAccessError::TenantUnknown`] /
///   [`crate::executor::SubstrateAccessError::IndexUnavailable`];
///   production wiring at M4-08+ may surface
///   [`crate::executor::SubstrateAccessError::Io`].
/// - [`Self::ExecutionEval`] — runtime evaluation fault during
///   execute (division by zero, NaN comparison, NULL operand
///   reaching a non-NULL-tolerant context). Distinct from
///   [`ArcQLError::TypeCheck`] (a plan-time type fault).
///
/// W11Z fix-up MED-2 (PR #268 retro): the EXPLAIN-only `Parse` /
/// `ArcQL` taxonomy was extended with `Cancelled` / `Substrate` /
/// `ExecutionEval` for the EXECUTE path. Previously every executor
/// error round-tripped as
/// `ArcQLError::NotImplemented { feature: "execute: ..." }` — which
/// hid `Cancelled` and a runtime `Eval` ("division by zero") behind
/// a "feature not implemented" diagnostic. The per-arm translation
/// preserves the M5↔M4 contract surface exhaustiveness: M5-07
/// `graph.search` / M5-11 `graph.raw_query` / M5-13 Bolt all
/// pattern-match against this enum.
///
/// `#[non_exhaustive]` on the umbrella allows future variant
/// additions without a breaking change to non-pattern-matching
/// callers; the per-MCP/Bolt/HTTP renderers handle exhaustive matches
/// with a `_ => ...` rendering for "unrecognized error variant —
/// fall back to Display".
///
/// `From` impls let callers `?` through both Parse + ArcQL layers
/// per the prior contract. Discipline: tests + integration callers
/// use the Display impl; programmatic callers `match` on the variant.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ExplainError {
    #[error("parse error: {0}")]
    Parse(#[from] crate::error::ParseError),
    #[error("{0}")]
    ArcQL(#[from] ArcQLError),
    /// Per-query cancellation token tripped mid-execute (W11Z MED-2).
    #[error("query cancelled")]
    Cancelled,
    /// Substrate-access fault mid-execute (W11Z MED-2).
    #[error("substrate access error: {0}")]
    Substrate(#[from] crate::executor::SubstrateAccessError),
    /// Runtime evaluation fault mid-execute (division by zero, NaN
    /// comparison, NULL operand reaching a non-NULL-tolerant context).
    /// Distinct from a plan-time type-check error (W11Z MED-2).
    #[error("runtime evaluation error: {0}")]
    ExecutionEval(String),
    /// **#797 / ADR-147 Phase 2** — a `$name` parameter referenced by
    /// the statement had no binding in the supplied parameter bag. A
    /// CLIENT fault (distinct from `ExecutionEval`, which is a
    /// server-side runtime fault): the wire surfaces map it to a client
    /// error (Bolt `Neo.ClientError.Statement.ParameterMissing`; MCP
    /// `-32602` invalid params), NOT to the
    /// `Neo.DatabaseError`/`-32006` server-fault bucket the generic
    /// `ExecutionEval` would have rendered.
    #[error("missing parameter: ${name}")]
    MissingParameter { name: String },
}

// ---------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------

/// Strip the `EXPLAIN` / `PROFILE` AST wrapper if present, returning
/// the inner [`Statement`]. Preserves bare read queries and index DDL.
///
/// The grammar already enforces that the inner of an EXPLAIN /
/// PROFILE wrapper is a read query (see grammar.pest comment); this
/// function honors that invariant by mapping `Statement::Explain(q)`
/// → `Statement::Read(q)` (and similarly for PROFILE). The downstream
/// passes never need to know the wrapper bit was set.
fn strip_explain_or_profile_wrapper(stmt: Statement) -> Statement {
    match stmt {
        Statement::Explain(q) | Statement::Profile(q) => Statement::Read(q),
        other => other,
    }
}

/// Run the post-parse pipeline + DP join-ordering + cost walk + project to
/// `PlanTree`.
///
/// # Wave 9d M4-52b wiring (W9b CRIT-1 closure)
///
/// Per the W9b cross-PR review of PR #240 (M4-91 EXPLAIN) + PR #242 (M4-52
/// DP join enumeration), the M4-52 DP enumerator must run between
/// `LogicalPlanLoweringVisitor::lower` and the cost walker so the
/// EXPLAIN-rendered [`PlanTree`] reflects the cost-optimal join order
/// (not the M4-31 input order). EXPLAIN is the FIRST and ONLY production
/// consumer of the M4-52 enumerator until M4-61 (executor) lands and
/// consumes the post-DP plan at runtime.
///
/// # Wave 10b single-FrozenCatalog threading (issue #261 closure)
///
/// Per W9d retro Agent A §A-LOW-1, this function previously captured
/// THREE independent [`crate::semantic::CatalogSnapshot`]s within one
/// EXPLAIN call — one for the cache watermark, one for the M4-52 DP
/// enumerator, one for the M4-51 cost walker. Under v1.1+ multi-tenant
/// concurrent writers the three could drift, producing apples-to-
/// oranges cost annotations within a single EXPLAIN. The fix captures
/// ONE snapshot (and the corresponding [`FrozenCatalog`] adapter) BEFORE
/// the cache lookup and threads it through the cache watermark check,
/// the DP enumerator, and the cost walker via the
/// [`enumerate_join_order_with_frozen`] /
/// [`estimate_costs_with_frozen`] sister-shims. Concurrent commits CAN
/// still happen between snapshot capture and cache insert, but the
/// stamped watermark and the post-enumeration cost both reflect the
/// SAME captured snapshot — internally consistent within each call.
/// EXPLAIN does NOT acquire an MVCC snapshot LSN (per ADR-038 §2 D-18
/// rule 1; pinned by `explain_does_not_acquire_snapshot_lsn`).
///
/// # Wave 10a M4-53 cache wiring (per ADR-038 amendment-03 §TIER-2-a)
///
/// When `cache.is_some()`, the validation passes (bind / typecheck /
/// cross-substrate) ALWAYS run — they catch input-time errors that
/// must surface deterministically regardless of cache state and produce
/// stable diagnostics across hit / miss. Then the cache is consulted:
///
/// - **Hit + fresh watermark** — return [`PlanTree::from_costed_plan`]
///   over the cached costed plan. The lower → enumerate → cost half
///   is skipped.
/// - **Miss / stale / invariant violation** — fall through to the cold
///   path (lower → enumerate → cost) and insert into the cache stamped
///   with the watermark captured at lookup.
///
/// The watermark is captured ONCE per call from the same snapshot that
/// flows into the DP enumerator and cost walker; concurrent commits
/// between lookup and insert CAN result in a stamped value that is one
/// or more behind the catalog's true current value at insert time; this
/// is benign (the next lookup invalidates and replans).
fn plan_tree_for<C: CatalogProvider>(
    stmt: &Statement,
    source: &str,
    catalog: &C,
    cache: Option<&PlanCache>,
) -> Result<PlanTree, ArcQLError> {
    let mut bound = BindingVisitor::bind(stmt, source, catalog)
        .map_err(|errs| first_or_internal(errs.into_iter().map(ArcQLError::from)))?;
    TypeCheckVisitor::check(&mut bound, catalog).map_err(first_or_internal_iter)?;
    CrossSubstrateValidator::validate(&bound, catalog).map_err(first_or_internal_iter)?;

    // Single snapshot capture per EXPLAIN call (closes #261). The
    // captured snapshot is shared by the cache watermark check, the
    // M4-52 DP enumerator, and the M4-51 cost walker — every per-key
    // cardinality read across the three stages observes the same
    // point-in-time view.
    let snapshot = catalog.snapshot();
    let stats_version = snapshot.commits_observed();
    let frozen = FrozenCatalog::new(catalog, snapshot);

    if let Some(cache) = cache {
        let key = PlanCacheKey::from_ast(catalog.tenant(), stmt);
        match cache.lookup(&key, stats_version) {
            LookupOutcome::Hit(cached) => return Ok(PlanTree::from_costed_plan(&cached)),
            LookupOutcome::Miss | LookupOutcome::Stale | LookupOutcome::InvariantViolation => {
                let plan = lower_for_planning(&bound).map_err(first_or_internal_iter)?;
                // #1366 (Phase 2): route the indexed point lookup so
                // EXPLAIN shows `PropertyIndexScan` (the operator can
                // confirm the index path is live).
                let plan = rewrite_scan_to_property_index_scan(plan, catalog);
                let optimized = enumerate_join_order_with_frozen(plan, &frozen);
                // W25-M4-61b / ADR-097: pick join algorithm before
                // costing so EXPLAIN output reflects the picked Hash
                // / Merge variant + the cache stores the post-picker
                // plan. Passes `&frozen` (NOT `catalog`) so the
                // picker's internal snapshot lookup returns the
                // already-captured snapshot — preserves the single-
                // snapshot discipline closing #261.
                let optimized = crate::planner::pick_join_algorithms(optimized, &frozen);
                let costed = Arc::new(estimate_costs_with_frozen(optimized, &frozen));
                let pt = PlanTree::from_costed_plan(&costed);
                cache.insert(key, costed, stats_version);
                return Ok(pt);
            }
        }
    }

    let plan = lower_for_planning(&bound).map_err(first_or_internal_iter)?;
    let plan = rewrite_scan_to_property_index_scan(plan, catalog);
    let optimized = enumerate_join_order_with_frozen(plan, &frozen);
    let optimized = crate::planner::pick_join_algorithms(optimized, &frozen);
    let costed = estimate_costs_with_frozen(optimized, &frozen);
    Ok(PlanTree::from_costed_plan(&costed))
}

/// Coerce a non-empty error-vec into the first error. The pipeline
/// passes accumulate ALL errors per amendment-03 §TIER-1 GAP E; the
/// EXPLAIN entry point exposes only the first (most-actionable) one
/// per the M5-07 / M5-11 / M5-13 single-error response shape. Callers
/// that need the full vec can drop into the lower-level passes
/// directly.
fn first_or_internal_iter(errs: Vec<ArcQLError>) -> ArcQLError {
    first_or_internal(errs.into_iter())
}

fn first_or_internal<I: Iterator<Item = ArcQLError>>(mut iter: I) -> ArcQLError {
    iter.next().unwrap_or_else(|| ArcQLError::NotImplemented {
        // INVARIANT: the upstream passes return
        // Err only on a non-empty error vec. This branch is
        // unreachable; if a future refactor breaks the invariant we
        // surface a defensive NotImplemented rather than panicking.
        feature: "EXPLAIN pipeline internal-error-vec-empty".into(),
        section: "M4-91 internal invariant".into(),
        target_version: "(internal)".into(),
        span: Span::point(1, 1),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use crate::planner::cost::{Cardinality, Cost};
    use crate::semantic::StubCatalogProvider;

    fn cat() -> StubCatalogProvider {
        StubCatalogProvider::new()
            .with_labels(["Person", "Doc"])
            .with_rel_types(["KNOWS"])
            .with_properties(["age", "name"])
    }

    #[test]
    fn explain_simple_match_returns_plan_tree() {
        let pt = explain("MATCH (n:Person) RETURN n", &cat()).expect("explain");
        // Pipeline produces Project(Scan(Person)) (the M4-31 lowering
        // baseline).
        assert_eq!(pt.op, PlanTreeOp::Project);
        assert_eq!(pt.children.len(), 1);
        assert_eq!(pt.children[0].op, PlanTreeOp::Scan);
    }

    #[test]
    fn plan_tree_as_rows_preorder_and_details_are_deterministic() {
        let mut scan_annotations = BTreeMap::new();
        scan_annotations.insert("label".to_string(), "Person".to_string());
        scan_annotations.insert("read_lsn".to_string(), "42".to_string());
        let scan = PlanTree {
            op: PlanTreeOp::Scan,
            bindings: vec!["b0".to_string()],
            estimated_cost: Cost::new(10.5),
            estimated_card: Cardinality::new(7.0),
            children: vec![],
            annotations: scan_annotations,
        };
        let mut project_annotations = BTreeMap::new();
        project_annotations.insert("items".to_string(), "1".to_string());
        let project = PlanTree {
            op: PlanTreeOp::Project,
            bindings: vec![],
            estimated_cost: Cost::new(11.5),
            estimated_card: Cardinality::new(7.0),
            children: vec![scan],
            annotations: project_annotations,
        };

        let rows = plan_tree_as_rows(&project);
        assert_eq!(rows.columns, PLAN_ROW_COLUMNS.map(str::to_string));
        assert_eq!(
            rows.rows,
            vec![
                vec![
                    Value::String("Project".into()),
                    Value::String("[items=1]".into()),
                    Value::Float(11.5),
                    Value::Float(7.0),
                    Value::Integer(0),
                ],
                vec![
                    Value::String("Scan".into()),
                    Value::String("b0 [label=Person, read_lsn=42]".into()),
                    Value::Float(10.5),
                    Value::Float(7.0),
                    Value::Integer(1),
                ],
            ]
        );
        assert_eq!(rows, plan_tree_as_rows(&project));
    }

    #[test]
    fn explain_keyword_prefix_strips_the_wrapper() {
        let pt_a = explain("EXPLAIN MATCH (n:Person) RETURN n", &cat()).expect("explain wrapped");
        let pt_b = explain("MATCH (n:Person) RETURN n", &cat()).expect("explain bare");
        // EXPLAIN wrapper is purely a control bit; the resulting
        // PlanTree must be identical to the bare-query case.
        assert_eq!(pt_a, pt_b);
    }

    #[test]
    fn profile_returns_plan_tree_and_metrics_at_w12gamma() {
        // W12γ M4-91 PROFILE wire-up pin (replaces the previous
        // NotImplemented stub per amendment-03 §TIER-1 GAP B).
        // PROFILE on a simple MATCH executes the plan AND returns
        // (PlanTree, ExecutionMetrics) — the metrics are populated
        // end-to-end by M4-08a's materialize tail.
        let s = crate::executor::StubExecutorSubstrate::new();
        let registry = crate::cancel::CancellationRegistry::new();
        let (pt, metrics) = profile(
            "PROFILE MATCH (n:Person) RETURN n",
            &cat(),
            &s,
            &registry,
            std::time::Duration::from_millis(crate::cancel::DEFAULT_QUERY_TIMEOUT_MS),
        )
        .expect("profile");
        assert_eq!(pt.op, PlanTreeOp::Project);
        // Metrics: rows_emitted is 0 because the stub substrate
        // carries no Person nodes; wall_time_ms is non-deterministic
        // but populated end-to-end (we only assert the field is
        // accessible — the surface is what's load-bearing).
        let _ = metrics.wall_time_ms;
        // Forward-bind: M4-64a populates this; v1.0-alpha = 0.
        assert_eq!(metrics.memory_bytes_high_water, 0);
        // No-leak: registry drained on PROFILE end (W12γ fix-up MED-2 +
        // MED-3).
        assert!(registry.is_empty());
    }

    #[test]
    fn profile_on_bare_read_query_returns_plan_tree_and_metrics() {
        // Symmetry: PROFILE accepts a bare read query (no wrapper).
        let s = crate::executor::StubExecutorSubstrate::new();
        let registry = crate::cancel::CancellationRegistry::new();
        let (pt, _metrics) = profile(
            "MATCH (n:Person) RETURN n",
            &cat(),
            &s,
            &registry,
            std::time::Duration::from_millis(crate::cancel::DEFAULT_QUERY_TIMEOUT_MS),
        )
        .expect("profile");
        assert_eq!(pt.op, PlanTreeOp::Project);
    }

    #[test]
    fn profile_surfaces_parse_errors_before_executing() {
        // PROFILE with broken syntax — must surface ParseError, NOT
        // a successful execution. The discipline matches every other
        // entry point: parse failures are reported first.
        let s = crate::executor::StubExecutorSubstrate::new();
        let registry = crate::cancel::CancellationRegistry::new();
        let err = profile(
            "PROFILE this is not arcql",
            &cat(),
            &s,
            &registry,
            std::time::Duration::from_millis(crate::cancel::DEFAULT_QUERY_TIMEOUT_MS),
        )
        .expect_err("parse error");
        assert!(
            matches!(err, ExplainError::Parse(_)),
            "expected ParseError, got {err:?}"
        );
    }

    #[test]
    fn explain_propagates_binding_error() {
        // A M4-21 BindingError flows through EXPLAIN. (Was an unknown-label
        // error; since ADR-038 amendment-12 made unknown labels permissive,
        // this uses an undeclared variable `x` — still a hard BindingError.)
        let err = explain("MATCH (n) RETURN x", &cat()).expect_err("binding error");
        match err {
            ExplainError::ArcQL(ArcQLError::Binding(_)) => {}
            other => panic!("expected Binding error, got {other:?}"),
        }
    }

    #[test]
    fn query_engine_explain_routes_through_free_function() {
        let c = cat();
        let engine = QueryEngine::new(&c);
        let pt = engine
            .explain("MATCH (n:Person) RETURN n")
            .expect("explain");
        assert_eq!(pt.op, PlanTreeOp::Project);
    }

    #[test]
    fn query_engine_profile_routes_through_free_function_and_returns_metrics() {
        let c = cat();
        let s = crate::executor::StubExecutorSubstrate::new();
        let engine = QueryEngine::new(&c);
        let (pt, _metrics) = engine
            .profile("PROFILE MATCH (n:Person) RETURN n", &s)
            .expect("profile");
        assert_eq!(pt.op, PlanTreeOp::Project);
    }

    #[test]
    fn execution_metrics_default_is_zero() {
        let m = ExecutionMetrics::default();
        assert_eq!(m.wall_time_ms, 0);
        assert_eq!(m.memory_bytes_high_water, 0);
        assert_eq!(m.rows_emitted, 0);
    }

    #[test]
    fn explain_with_cache_routes_through_cache_on_repeated_query() {
        // M4-53 wiring smoke: the second EXPLAIN of the same query
        // through `explain_with_cache` produces the SAME plan tree as
        // the first AND populates the per-tenant cache. A bare
        // `explain` without a cache produces an equivalent plan tree
        // (cached vs uncached output is structurally identical).
        let c = cat();
        let cache = PlanCache::new();
        let pt_a = explain_with_cache("MATCH (n:Person) RETURN n", &c, &cache).expect("a");
        let pt_b = explain_with_cache("MATCH (n:Person) RETURN n", &c, &cache).expect("b");
        let pt_uncached = explain("MATCH (n:Person) RETURN n", &c).expect("uncached");
        assert_eq!(pt_a, pt_b);
        assert_eq!(pt_a, pt_uncached);
        // Cache populated for the engine's tenant.
        assert_eq!(cache.len_for(c.tenant()), 1);
    }

    // -----------------------------------------------------------------
    // W11Z fix-up MED-2 (PR #268 retro): per-arm executor-error
    // translation pins.
    // -----------------------------------------------------------------

    #[test]
    fn translate_execution_error_cancellation_preserves_distinction() {
        // Pre-fix-up, Cancelled rendered as
        // ArcQLError::NotImplemented { feature: "execute: query cancelled" }.
        // Post-fix-up: ExplainError::Cancelled, distinct variant.
        let translated = translate_execution_error(crate::executor::ExecutionError::Cancelled);
        assert_eq!(translated, ExplainError::Cancelled);
    }

    #[test]
    fn translate_execution_error_substrate_round_trips() {
        let inner = crate::executor::SubstrateAccessError::IndexUnavailable("vector".into());
        let translated =
            translate_execution_error(crate::executor::ExecutionError::Substrate(inner.clone()));
        assert_eq!(translated, ExplainError::Substrate(inner));
    }

    #[test]
    fn translate_execution_error_eval_round_trips_as_execution_eval() {
        let translated = translate_execution_error(crate::executor::ExecutionError::Eval(
            "division by zero".into(),
        ));
        assert_eq!(
            translated,
            ExplainError::ExecutionEval("division by zero".into())
        );
    }

    #[test]
    fn translate_execution_error_not_implemented_preserves_target_slice() {
        let translated =
            translate_execution_error(crate::executor::ExecutionError::NotImplemented {
                feature: "LogicalPlan::Aggregate".into(),
                target_slice: "M4-63".into(),
                section: "ADR-038 amendment-02 §M4.g".into(),
            });
        match translated {
            ExplainError::ArcQL(ArcQLError::NotImplemented {
                feature,
                section,
                target_version,
                ..
            }) => {
                assert_eq!(feature, "LogicalPlan::Aggregate");
                assert_eq!(section, "ADR-038 amendment-02 §M4.g");
                assert_eq!(
                    target_version, "M4-63",
                    "executor-side target_slice must round-trip into ArcQLError::target_version"
                );
            }
            other => panic!("expected ArcQL(NotImplemented), got {other:?}"),
        }
    }

    #[test]
    fn translate_execution_error_plan_round_trips_as_arcql() {
        // Executor's `Plan` arm carries an already-formed ArcQLError
        // (rare path — planner + executor both run, executor saw a
        // re-emitted plan-time error). Round-trip preserves the
        // ArcQLError unchanged.
        let inner = ArcQLError::NotImplemented {
            feature: "test".into(),
            section: "test §".into(),
            target_version: "v-test".into(),
            span: Span::point(1, 1),
        };
        let translated =
            translate_execution_error(crate::executor::ExecutionError::Plan(inner.clone()));
        assert_eq!(translated, ExplainError::ArcQL(inner));
    }

    // -----------------------------------------------------------------
    // W12γ fix-up MED-1/MED-2 — default-deadline route-through pins.
    // -----------------------------------------------------------------

    #[test]
    fn execute_default_path_registers_qid_under_engine_registry() {
        // Pin: QueryEngine::execute mints a QueryId and registers it
        // against the engine's registry. We can't easily peek at the
        // registry MID-execute (the substrate has 0 nodes — execute
        // returns immediately), but the post-execute drain combined
        // with the integration-level deadline-fire pin
        // (cancel_integration::execute_default_path_applies_30s_timeout)
        // proves the route-through.
        //
        // This unit pin proves that execute completes and leaves the
        // registry empty — the no-leak invariant.
        let c = cat();
        let s = crate::executor::StubExecutorSubstrate::new();
        let engine = QueryEngine::new(&c);
        let _r = engine
            .execute("MATCH (n:Person) RETURN n", &s)
            .expect("execute");
        assert!(
            engine.cancellation_registry().is_empty(),
            "execute must drain its registry entry"
        );
    }

    #[test]
    fn engine_profile_default_path_drains_registry() {
        // MED-2 unit pin: QueryEngine::profile registers against the
        // engine's registry (per the route-through we wired) and
        // drains on success.
        let c = cat();
        let s = crate::executor::StubExecutorSubstrate::new();
        let engine = QueryEngine::new(&c);
        let _r = engine
            .profile("PROFILE MATCH (n:Person) RETURN n", &s)
            .expect("profile");
        assert!(
            engine.cancellation_registry().is_empty(),
            "profile must drain its registry entry"
        );
    }
}
