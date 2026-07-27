//! M4-08a single-statement result materialization.
//!
//! Lit at v1.0 per ADR-038 amendment-02 §M4.h + amendment-03 §M5↔M4
//! contract surface.
//!
//! # Slice scope (M4-08a)
//!
//! - **`MaterializedResult`** — the v1.0 [`crate::QueryEngine::execute`]
//!   return type. Carries:
//!   - `rows: Vec<Vec<Value>>` — the full executor output, drained
//!     batch-by-batch into a single Vec per amendment-02 §M4.h
//!     "single-statement materialization (TOON + JSON serialization
//!     for MCP; per-tenant memory budget)". Streaming / cursor
//!     iteration is forward-deferred to M4-82 (M4-08b); v1.0 ships
//!     the full-Vec form.
//!   - `metrics: ExecutionMetrics` — top-level timing + memory + row
//!     count, wired here so the M4-91 PROFILE entry-point can wrap
//!     `MaterializedResult` inline. Per-operator metrics
//!     (M4-71 `RowCountObserver`) populate at the M4-71 wiring layer
//!     — the field set on [`crate::ExecutionMetrics`] is already
//!     forward-pinned for it.
//!
//! # Why this slice depends on cancellation (M4-92 in same wave)
//!
//! Per amendment-03 §M5↔M4 contract surface §11 D-9:
//!
//! > Implementation lands across M4-08a (`execute` row-materialization
//! > tail), M4-08b (`execute` streaming-cursor tail), M4-91
//! > (`explain` + `profile`), M4-92 (`cancel` + `timeout` plumbing).
//!
//! M4-08a is the materialization-tail surface that the M4-92
//! cancellation registry uses as its registration scope — register
//! the [`crate::QueryId`] before the first `next_batch`, unregister
//! at materialization end (success, error, or cancel). The W12γ
//! slice fuses M4-08a + M4-92 because the lifecycle is co-extensive:
//! cancellation is meaningful only for the duration of the materialize
//! loop.
//!
//! # Memory budget
//!
//! v1.0-alpha materializes ALL rows into a single `Vec` — no per-
//! tenant memory budget enforcement at this slice. M4-64a (forward)
//! plugs in a `MemoryTracker` at the executor-batch boundary that
//! drives the `metrics.memory_bytes_high_water` field. The
//! `MaterializedResult::rows` field is a candidate for streaming
//! when M4-82 lights; the M4-08a return type stays stable
//! (the streaming variant ships as a sibling
//! `MaterializedCursor` type, NOT a Vec replacement, per the
//! 7-slice 3-strike scaffolding rule — we do not generalize
//! prematurely).
//!
//! # PROFILE wiring (replaces previous NotImplemented body)
//!
//! Per amendment-03 §TIER-1 GAP B + this slice's prompt:
//!
//! > M4-91 PROFILE wire: previous `NotImplemented` body is replaced —
//! > PROFILE now returns `(PlanTree, ExecutionMetrics)` per
//! > amendment-03 §TIER-1 GAP B
//!
//! The PROFILE entry-point now: parse → bind → type-check → cross-
//! substrate validate → lower → cost (PlanTree) → THEN ALSO build the
//! operator pipeline + run [`materialize`] → return
//! `(PlanTree, ExecutionMetrics)` where the metrics are populated
//! from [`MaterializedResult::metrics`]. Per-operator annotations
//! (forward-link to M4-71 `RowCountObserver`) are NOT in this slice;
//! the `wall_time_ms` + `rows_emitted` fields on `ExecutionMetrics`
//! are populated end-to-end at this slice. `memory_bytes_high_water`
//! is a forward-link to M4-64a (`MemoryTracker`).
//!
//! # ADR provenance
//! - **ADR-038 amendment-02 §M4.h** — primary M4-08a (M4-81) cite.
//! - **ADR-038 amendment-03 §TIER-1 GAP B** — M4-91 PROFILE return
//!   shape `(PlanTree, ExecutionMetrics)`.
//! - **ADR-038 amendment-03 §M5↔M4 contract surface §11 D-9** —
//!   `execute` returns `MaterializedResult`-shaped value; M4-08a is
//!   the implementation row.
//! - **bounded-context policy** — implementer-vs-orchestrator discipline;
//!   this slice was implemented directly by a spawned implementer
//!   agent (W12γ).

use std::cell::Cell;
use std::time::Instant;

use crate::executor::budget::{MemoryBudget, estimate_row_bytes};
use crate::executor::error::ExecutionError;
use crate::executor::pipeline::Pipeline;
use crate::executor::value::Value;
use crate::executor::{ExecutionContext, ExecutorSubstrate};
use crate::explain::ExecutionMetrics;
use crate::logical_plan::LogicalPlan;
use crate::semantic::error::ArcQLError;
use arcgraph_core::TenantId;

/// Single-statement query result per ADR-038 amendment-02 §M4.h +
/// amendment-03 §M5↔M4 contract surface.
///
/// # Field semantics
///
/// - **`rows`** — the executor's output rows, drained into a single
///   Vec by [`materialize`]. Each inner `Vec<Value>` is one row per
///   the projection at the root of the plan. Empty Vec (`rows == &[]`)
///   denotes a successful empty result (e.g., `MATCH ... WHERE false
///   RETURN n` → 0 rows). Streaming variants ship at M4-82 as a
///   sibling type — this Vec form is the stable return for v1.0.
///
/// - **`metrics`** — top-level execution metrics. At v1.0-alpha the
///   populated fields are:
///   - `wall_time_ms` — Instant-based wall-clock elapsed inside
///     [`materialize`] (excludes plan-time; PROFILE's plan-time is
///     amortized via the pre-execute phases).
///   - `rows_emitted` — equal to `rows.len()` at this slice; the
///     forward M4-82 streaming-cursor surface decouples
///     `rows_emitted` from the materialized count (the cursor may
///     emit more rows than the v1.0 Vec carries). Future M4-71
///     overlay rebinds `rows_emitted` to per-operator emit counts.
///   - `memory_bytes_high_water` — `0` at this slice. Forward-link
///     to M4-64a's `MemoryTracker`.
///
/// # Why a struct and not a tuple?
///
/// A struct lets us add fields (e.g., a `query_id` for tracing
/// correlation; the M4-83 `multi_result_idx` for multi-statement
/// queries) without breaking pattern-matching call sites. Per the
/// 7-slice 3-strike scaffolding rule, we ship the struct now because
/// the M4-83 multi-statement extension is named in roadmap.md and
/// the field-add is concrete (not speculative) — the struct shape
/// has THREE in-flight consumers within M4 alone (M4-08a +
/// M4-91 PROFILE + M4-83 `Vec<MaterializedResult>` for the multi-
/// statement query surface).
///
/// # Serde / TOON / JSON
///
/// Per amendment-02 §M4.h, the M4-08a payload "TOON + JSON
/// serialization for MCP" — the W11ε arcgraph-mcp serializers
/// (per `feedback_writeup_loc_precision.md` PR #271 cite) consume
/// `MaterializedResult` directly. Serde derive is forward-deferred
/// to the consumer side: `arcgraph-mcp` already serializes
/// `Vec<Vec<Value>>` end-to-end; the wrapper struct lifts cleanly
/// into the same path (no behavior change for existing MCP
/// callers — the proptest in
/// `tests/m4_08a_materialize_proptest.rs` exercises a roundtrip
/// through `serde_json` on the rows half via the existing
/// `Value`'s `serde::Serialize` impl).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MaterializedResult {
    /// Executor output rows, drained from the operator pipeline.
    pub rows: Vec<Vec<Value>>,
    /// #353 — the ordered, user-meaningful result-column NAMES (the
    /// RETURN aliases / implicit column names), one per column of each
    /// row in [`Self::rows`].
    ///
    /// `RETURN a, r, b` → `["a", "r", "b"]`; `RETURN n.name AS name` →
    /// `["name"]`; `RETURN n.name` → `["n.name"]`; `RETURN count(*)` →
    /// `["count(*)"]`. Derived from the bound terminal RETURN clause
    /// (or standalone `CALL proc` YIELD / `SHOW` columns) by
    /// [`crate::output_column_names`] and stamped onto the result by the
    /// `QueryEngine::execute*` entry-points.
    ///
    /// Empty when the column names are unknown for the path — a
    /// write-only statement with no RETURN (`CREATE (n)`), a
    /// directly-constructed [`MaterializedResult`] (test fixtures), or a
    /// `materialize()` call that did not go through a `QueryEngine`
    /// entry-point. Wire renderers (MCP `RawQueryRows`, Bolt
    /// `RunOutcome::fields`) fall back to synthesized `col_0..N` labels
    /// only when this is empty AND the rows are non-empty (so a
    /// no-column result still gets a stable shape), so an unpopulated
    /// `columns` never regresses the pre-#353 behavior.
    ///
    /// Invariant when populated: `columns.len()` equals each row's
    /// width. The extractor derives the count from the projection items
    /// (wildcards excluded — see [`crate::output_column_names`]'s
    /// wildcard handling), so a `RETURN *` query (whose width is
    /// data-dependent) leaves this empty and falls back to `col_0..N`.
    pub columns: Vec<String>,
    /// Top-level execution metrics. Populated end-to-end by
    /// [`materialize`]; per-operator annotations are forward-deferred
    /// to M4-71's `RowCountObserver`.
    pub metrics: ExecutionMetrics,
    /// W13β M4-81 — partial-result indicator.
    ///
    /// `None` denotes the materialize loop completed: every operator-
    /// produced row was admitted by the per-tenant memory budget and
    /// is present in [`Self::rows`]. `Some(error)` denotes a partial
    /// result: the materialize loop encountered the per-tenant budget
    /// ceiling mid-stream; [`Self::rows`] carries the rows accumulated
    /// up to (but not including) the row that would have crossed the
    /// cap, and the embedded error names the budget ceiling +
    /// projected bytes per W12α `MemoryBudget` convention.
    ///
    /// # Why a field on the success path (not Err on the Result)
    ///
    /// Per ADR-038 amendment-02 §M4.h, a budget-exhausted query
    /// surfaces "partial result + `ArcQLError::ResourceExhausted`"
    /// per the W12α convention. Rust's `Result<T, E>` discards data on
    /// `Err`, which would force the M5-07 / M5-11 / M5-13 renderers to
    /// re-execute the query to recover the partial rows. Instead,
    /// budget exhaustion routes through this field so renderers can
    /// surface "best-effort response: `<N>` rows + budget exhausted"
    /// without re-execution. Real executor errors (Cancelled /
    /// Substrate / Eval) still flow as Err on
    /// [`materialize`]'s return — those are NOT partial-result paths.
    ///
    /// # Forward-binding
    ///
    /// M4-82 (streaming cursor) propagates the same field through its
    /// `close()` outcome — a cursor that closes due to budget
    /// exhaustion records the same `truncation` indicator. The
    /// surface stays uniform across single-batch (M4-81) and streaming
    /// (M4-82) materialization.
    pub truncation: Option<ArcQLError>,
}

impl MaterializedResult {
    /// Borrow the rows slice. The canonical shorthand at call sites
    /// that previously typed `Vec<Vec<Value>>` directly.
    #[inline]
    #[must_use]
    pub fn rows(&self) -> &[Vec<Value>] {
        &self.rows
    }

    /// Consume the wrapper and return the rows half. Used at v1.0
    /// MCP-renderer boundaries that already iterate `Vec<Vec<Value>>`
    /// without inspecting metrics.
    #[inline]
    #[must_use]
    pub fn into_rows(self) -> Vec<Vec<Value>> {
        self.rows
    }

    /// Borrow the metrics half.
    #[inline]
    #[must_use]
    pub fn metrics(&self) -> &ExecutionMetrics {
        &self.metrics
    }

    /// #353 — borrow the ordered result-column names (the user RETURN
    /// aliases / implicit column names). Empty when unknown for the path
    /// (write-only statement with no RETURN, a `RETURN *` wildcard whose
    /// width is data-dependent, or a directly-constructed result); the
    /// wire renderers fall back to `col_0..N` in that case. See
    /// [`Self::columns`] for the populated-invariant.
    #[inline]
    #[must_use]
    pub fn columns(&self) -> &[String] {
        &self.columns
    }

    /// #353 — stamp the result-column names. Called by the
    /// `QueryEngine::execute*` entry-points with the names derived from
    /// the bound terminal RETURN clause via
    /// [`crate::output_column_names`]. Builder-style (returns `self`) so
    /// it composes with the `materialize(...).map(...)` chains in the
    /// engine.
    #[inline]
    #[must_use]
    pub fn with_columns(mut self, columns: Vec<String>) -> Self {
        self.columns = columns;
        self
    }

    /// Number of rows materialized. Equivalent to
    /// `self.rows.len()` and `self.metrics.rows_emitted` at this
    /// slice; M4-82 streaming-cursor will decouple `metrics.rows_emitted`
    /// from `rows.len()` (the cursor may emit more rows than the v1.0
    /// Vec carries, e.g., when the cursor consumer back-pressures
    /// before exhausting the stream).
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Empty-result helper.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// W13β M4-81 — `true` iff the result is a partial materialization
    /// produced by hitting the per-tenant memory-budget ceiling.
    ///
    /// Renderers should surface a "result truncated by per-tenant
    /// budget" diagnostic when `is_truncated()`; the [`Self::truncation`]
    /// field carries the embedded [`ArcQLError::ResourceExhausted`]
    /// with the byte-numbers the renderer surfaces to the user.
    #[inline]
    #[must_use]
    pub fn is_truncated(&self) -> bool {
        self.truncation.is_some()
    }

    /// W13β M4-81 — borrow the truncation cause if present.
    #[inline]
    #[must_use]
    pub fn truncation(&self) -> Option<&ArcQLError> {
        self.truncation.as_ref()
    }
}

/// Drive `plan` through the M4-61 vectorized executor with `ctx` and
/// `substrate`, materializing all rows into a [`MaterializedResult`].
///
/// # Snapshot-LSN discipline
///
/// Per ADR-038 §2 D-18 rule 1, the snapshot LSN is acquired LAZILY at
/// the first `next_batch` call (via
/// [`ExecutionContext::ensure_snapshot_lsn`]); this function does NOT
/// acquire eagerly. EXPLAIN's no-LSN discipline (also rule 1's EXPLAIN
/// exception) is honored because EXPLAIN never calls this function.
///
/// **W13β M4-81 — release at materialization-end.** Per amendment-03
/// §TIER-1 GAP E rule 4 ("Snapshot LSN released at query-end /
/// cursor-close"), this function holds a [`crate::executor::SnapshotLsnGuard`]
/// for the duration of the materialize loop. Drop of the guard
/// (success, error, partial-result, or panic-unwind) calls
/// [`ExecutionContext::release_snapshot_lsn`]. The RAII discipline
/// matches the W12γ MED-3 `RegistryGuard` pattern in `explain.rs`.
///
/// # Cancellation
///
/// The operator pipeline's per-batch cancellation gate (per
/// amendment-02 §M4.f) surfaces [`ExecutionError::Cancelled`] which
/// this function propagates upward. The v1.0-alpha materialization
/// loop does NOT carry a per-row cancellation check — the 2048-row
/// batch boundary is well inside the M5-12 cancel-latency budget
/// per ADR-036 §D-24. The `BudgetReservationGuard` AND
/// [`crate::executor::SnapshotLsnGuard`] both drop on the cancellation
/// path (they're stack-bound RAII guards) so a cancelled query leaks
/// neither budget bytes nor the snapshot-LSN slot.
///
/// # Memory budget
///
/// Per W13β M4-81 + ADR-038 amendment-02 §M4.h ("per-tenant memory
/// budget"), each row produced by the operator pipeline is charged
/// against [`ExecutionContext::budget`] via
/// [`MemoryBudget::try_reserve_unscoped`] before it enters the result
/// Vec. If the per-tenant cap would be exceeded, the function STOPS
/// the loop, releases the bytes accumulated so far via the RAII
/// guard, and returns `Ok(MaterializedResult { truncation: Some(...),
/// ... })` — a PARTIAL result whose rows are the prefix admitted by
/// the budget, with the [`ArcQLError::ResourceExhausted`] embedded
/// for renderer diagnostics.
///
/// At v1.0-alpha the default per-tenant budget cap is `None`
/// (unbounded) per [`MemoryBudget::new`]; budget enforcement is
/// active only when an explicit cap is configured via
/// [`ExecutionContext::with_budget`] /
/// [`MemoryBudget::set_per_tenant_cap`]. M5-12 rate-limit config
/// flips the default to a configured cap at server-startup time.
///
/// # Metrics population
///
/// At this slice the populated `ExecutionMetrics` fields are:
/// - `wall_time_ms` — `Instant`-based wall-clock elapsed inside the
///   batch loop; excludes plan-time (which the caller bears).
/// - `rows_emitted` — equal to `rows.len()` post-loop.
/// - `memory_bytes_high_water` — total bytes reserved by THIS
///   materialize call (sum of per-row [`estimate_row_bytes`] over
///   admitted rows). Per the budget surface's per-tenant counter
///   semantics, this is THIS call's contribution, NOT the per-tenant
///   total (which may include concurrent queries' bytes).
///
/// # Errors
///
/// Forwards [`ExecutionError`] from the underlying
/// [`crate::executor::execute_with_context`]; cancellation surfaces
/// as [`ExecutionError::Cancelled`]. Budget exhaustion does NOT
/// surface as Err — it's the partial-result path documented above.
pub fn materialize<S>(
    plan: &LogicalPlan,
    substrate: &S,
    ctx: &ExecutionContext,
) -> Result<MaterializedResult, ExecutionError>
where
    S: ExecutorSubstrate,
{
    // W13β fix-up M-1: reject on a context whose snapshot LSN was
    // previously released (close-then-reopen pattern). See
    // `crate::cursor` module docs §"Close-then-reopen REJECTS at v1.0"
    // for the rule-5 rationale; consistency with `StreamingCursor::open`
    // — both surfaces refuse to silently re-acquire a fresh LSN on a
    // consumed context.
    if ctx.lsn_consumed() {
        return Err(ctx.lsn_consumed_error("materialize::materialize"));
    }
    // Drop order (Rust drops locals in reverse declaration order):
    // `_lsn_guard` is the LAST-declared local in this scope, so it is
    // dropped LAST — keeping the LSN alive until the inner body's
    // BudgetReservationGuard has finished its drop-time release. This
    // matches amendment-03 §TIER-1 GAP E rule 4 ("released at query-end")
    // — query-end is THIS function's return.
    let _lsn_guard = ctx.snapshot_lsn_guard();
    materialize_with_outer_lsn_held(plan, substrate, ctx)
}

/// Body of [`materialize`] that ASSUMES an outer scope already holds
/// the [`crate::executor::SnapshotLsnGuard`] and has performed the
/// `lsn_consumed` rejection check.
///
/// # Why a private split
///
/// W13γ M4-83's [`materialize_multi`] requires that the snapshot LSN
/// is held for the duration of ALL statements per ADR-038 §2 D-18 rule 2
/// (and amendment-03 §TIER-1 GAP E rule 2): "Same snapshot LSN held for
/// all statements in a multi-statement query". If each inner
/// [`materialize`] call took its own RAII guard, the first call's drop
/// would release the LSN AND light the W13β fix-up M-1 `lsn_consumed`
/// latch — causing the second call to reject before observing the first
/// call's captured LSN.
///
/// Splitting the public entry-point's "guard + latch-check" prologue
/// from the body lets [`materialize_multi`] hoist BOTH to its own
/// outer scope (one guard + one latch check across all statements),
/// while preserving the SINGLE-statement [`materialize`] behavior:
/// drop releases LSN at function return, lights the latch, REJECTS
/// any subsequent reuse on the same context.
fn materialize_with_outer_lsn_held<S>(
    plan: &LogicalPlan,
    substrate: &S,
    ctx: &ExecutionContext,
) -> Result<MaterializedResult, ExecutionError>
where
    S: ExecutorSubstrate,
{
    // # D-2 (ADR-147 §D-8 / W26-θ Phase 5) — statement-scoped autocommit
    //   transaction budget
    //
    // Latency/atomicity budget: a `CREATE (a)-[:R]->(b)` spine writes 2
    // nodes + 1 rel. Pre-D-2 the AUTO-COMMIT substrate did begin→op→commit
    // PER op → 3 durable CommitBundles / 3 fsyncs → both slow (~1.8× the
    // same-form ingest gap vs neo4j per `docs/perf/gap-neo4j-ingest-batch.md`
    // §3) AND a correctness hole (a crash/error after commit #2 leaves a
    // partial 2-of-3 spine, non-atomic). D-2 routes ALL write ops of ONE
    // statement through ONE transaction: begin-once here, every op STAGES
    // (the mature ADR-197 held-txn / EXPLICIT path), commit-once at the
    // end → 1 fsync, all-or-nothing on any mid-statement fault.
    //
    // A statement-scoped txn opens IFF the plan mutates the graph AND no
    // held txn is already installed (an explicit Bolt BEGIN…COMMIT owns
    // its own multi-statement lifetime per ADR-197 — D-2 must NOT nest a
    // statement txn inside it, or it would double-commit / commit the
    // outer tx early). The substrate's `begin_statement` is a NO-OP for
    // read-only substrates + the v1.0-α stub, so read plans + non-durable
    // fixtures keep byte-for-byte behavior.
    // NN-4 (#1384) re-spin, Fix 1 — acquire the MERGE get-or-create
    // serialization guard(s) BEFORE `begin_statement`. This ordering is
    // load-bearing: on the D-2 path `begin_statement` installs a
    // `BoltHeldTxn` whose pinned snapshot the MERGE match probe reads at.
    // A guard acquired only INSIDE `MergeOp::next_batch` (after
    // `begin_statement`) would let the loser pin a stale pre-commit
    // snapshot and still double-create — the deeper root cause behind the
    // ultracode-verify REJECT. Acquiring here first means the loser BLOCKS
    // before pinning its snapshot; once it proceeds the winner has already
    // committed (see the post-commit drop below), so the loser's fresh
    // snapshot sees the winner's node → match branch → no double-create.
    //
    // Acquire ordering + the `?` here: the shipped `merge_guard`
    // (`CrudExecutorSubstrate::merge_guard`, a `parking_lot` `lock_arc`) is
    // INFALLIBLE — it never returns `Err`, so `acquire_merge_guards` cannot
    // fail on the production path and this `?` never fires in practice. The
    // `MergeGuardDrain` is therefore bound on the NEXT line (AFTER the
    // acquire), and covers every POST-ACQUIRE exit of THIS function — the
    // `begin_statement` error `?`, the commit error `?`, the
    // success/rollback fall-through, and panic-unwind — dropping the stashed
    // guard(s) AFTER `commit_statement`/`rollback_statement` has run (the
    // guard outlives the commit because the drain binding is dropped at
    // function scope end, which is after the commit/rollback block below).
    // That post-commit drop is what makes the guard SPAN the commit.
    //
    // NOTE (partial-hold, UNREACHABLE): if a FUTURE fallible substrate made
    // `acquire_merge_guards` Err mid-multi-key acquire, the guards stashed
    // BEFORE the error would leak (the drain is not yet bound). This cannot
    // happen with the shipped infallible guard. A fallible substrate would
    // need to bind the drain before the acquire (or have
    // `acquire_merge_guards` hold guards in a local `Vec` transferred to
    // `ctx` only on `Ok`) to close it. See the `MergeGuardDrain` doc.
    crate::executor::ops::acquire_merge_guards(plan, substrate, ctx)?;
    let _merge_guard_drain = MergeGuardDrain { ctx };

    let stmt_scoped = plan.writes() && !ctx.has_held_txn();
    if stmt_scoped {
        substrate.begin_statement(ctx)?;
    }

    let result = drive_pipeline(plan, substrate, ctx);

    if stmt_scoped {
        match &result {
            Ok(_) => {
                // Success → commit the whole spine ONCE (one WAL
                // CommitBundle / fsync; #963 HNSW + BM25 hooks fired once;
                // CDC sees one commit). A commit failure converts a Ok
                // drive into an Err (the writes are discarded by the
                // substrate's commit-error path — no partial spine).
                // `_merge_guard_drain` is still alive across this commit, so
                // the MERGE guard(s) are held until AFTER the node is
                // durable, then released when this function returns (the
                // loser re-probes the winner's committed node → match).
                substrate.commit_statement(ctx)?;
            }
            Err(_) => {
                // Any mid-statement fault → roll back the ENTIRE spine
                // (mirrors the ADR-197 Bolt ROLLBACK / RESET abort path).
                // Neither node of a failed 2-node-1-rel statement commits.
                // The guard(s) release AFTER this rollback (at function
                // return) so a loser re-probes only once the failed create
                // has been discarded (it will then legitimately create).
                substrate.rollback_statement(ctx);
            }
        }
    }
    // else: held-txn mode (explicit Bolt BEGIN…COMMIT) OR a read plan that
    // acquired no statement txn. A read plan stashes no guard (the drain is
    // a harmless no-op). In held-txn mode the create is STAGED (commits at
    // the explicit COMMIT, out of THIS scope) — the drain still releases
    // the per-statement guard at the END of THIS statement (its
    // concurrent-explicit-txn edge is a documented SI limitation, not
    // closed by the pessimistic per-statement lock; see
    // `MergeOp::next_batch`), preventing the guard from leaking to
    // context-drop across the transaction's statements.

    result
}

/// **NN-4 (#1384) re-spin, Fix 1** — RAII drain that releases the MERGE
/// serialization guard(s) stashed on the [`ExecutionContext`] when it
/// drops. Bound in [`materialize_with_outer_lsn_held`] AFTER the guards are
/// acquired (before `begin_statement`) so — because Rust drops locals in
/// reverse declaration order at function scope end — the guard(s) release
/// AFTER `commit_statement`/`rollback_statement` runs. This is exactly what
/// makes the MERGE guard SPAN the commit: the loser blocked in
/// `merge_guard` unblocks only once the winner's node is durable, then
/// re-probes + sees it (match branch, no double-create). The RAII shape
/// also releases on the POST-ACQUIRE `?` error paths (`begin_statement` /
/// commit) and on panic-unwind, so no per-key lock ever leaks to
/// context-drop. (It does NOT cover an error from `acquire_merge_guards`
/// itself — the binding is AFTER that acquire — but the shipped guard is
/// infallible, so that path is unreachable; see the acquire-site comment in
/// [`materialize_with_outer_lsn_held`].)
struct MergeGuardDrain<'c> {
    ctx: &'c crate::executor::ExecutionContext,
}

impl Drop for MergeGuardDrain<'_> {
    fn drop(&mut self) {
        // Dropping the drained `Vec` releases the per-key mutexes; an empty
        // `Vec` (read / keyless plan) is a harmless no-op.
        drop(self.ctx.take_merge_guards());
    }
}

/// D-2 — the pipeline-drive body of [`materialize_with_outer_lsn_held`],
/// extracted so the statement-scoped-txn wrapper (begin / commit / roll
/// back) has a single `Result` to route on. Assumes the outer scope owns
/// the snapshot-LSN guard (per [`materialize_with_outer_lsn_held`]'s
/// contract) and, when the plan writes in AUTO-COMMIT mode, that a
/// statement txn was begun on `ctx` before this call — so every substrate
/// write op STAGES into it instead of auto-committing per op.
fn drive_pipeline<S>(
    plan: &LogicalPlan,
    substrate: &S,
    ctx: &ExecutionContext,
) -> Result<MaterializedResult, ExecutionError>
where
    S: ExecutorSubstrate,
{
    let start = Instant::now();
    // #797 — bake the per-query parameter bag into the operator tree so
    // `BoundExpression::Parameter { name }` resolves at runtime. Empty
    // for literal-only queries (byte-for-byte identical to the prior
    // `Pipeline::build`, which forwards `Parameters::new()`).
    let mut op = Pipeline::build_with_parameters(plan, ctx.parameters())?;
    // The per-tenant byte counter is decremented when this guard drops
    // at function return (or panic-unwind). Order against the outer
    // `_lsn_guard` doesn't matter for correctness — the two guards are
    // independent — but per the M4-81 doc, budget release should
    // observably precede LSN release. Guard is declared HERE (inside
    // the body) so the public [`materialize`] sees its drop order
    // before the outer LSN guard.
    let budget_guard = BudgetReservationGuard::new(ctx.budget(), ctx.tenant());
    let mut rows: Vec<Vec<Value>> = Vec::new();
    let mut truncation: Option<ArcQLError> = None;
    'outer: loop {
        let batch = op.next_batch(ctx, substrate)?;
        if batch.is_empty() {
            break;
        }
        for row in batch.into_rows() {
            let bytes = estimate_row_bytes(&row) as u64;
            match ctx
                .budget()
                .try_reserve_unscoped(ctx.tenant(), bytes, "M4-81 materialize")
            {
                Ok(()) => {
                    budget_guard.add_bytes(bytes);
                    rows.push(row);
                }
                Err(ExecutionError::Plan(arcql @ ArcQLError::ResourceExhausted { .. })) => {
                    // Per W13β M4-81: budget exhausted → stop the
                    // loop, return partial result + the embedded
                    // ArcQLError. The triggering row is DROPPED (its
                    // bytes were not reserved per W12α convention —
                    // try_reserve_unscoped does not bump on rejection).
                    truncation = Some(arcql);
                    break 'outer;
                }
                Err(other) => {
                    // Other ExecutionError variants (Substrate / Eval
                    // / NotImplemented) propagate as Err. The
                    // budget_guard + lsn_guard release on stack
                    // unwind via Drop.
                    return Err(other);
                }
            }
        }
    }
    let wall_time_ms = start.elapsed().as_millis() as u64;
    let rows_emitted = rows.len() as u64;
    let memory_bytes_high_water = budget_guard.bytes();
    Ok(MaterializedResult {
        rows,
        // #353 — `materialize` works at the LogicalPlan level and does
        // not know the bound RETURN-item display names; the
        // `QueryEngine::execute*` entry-points stamp `columns` from the
        // bound statement after this returns. Left empty here.
        columns: Vec::new(),
        metrics: ExecutionMetrics {
            wall_time_ms,
            // W13β M4-81: this is THIS call's contribution to the
            // per-tenant counter (sum of admitted rows' estimated
            // bytes), NOT the per-tenant total. M4-64a's
            // `MemoryBudget::peak_bytes` reads the per-tenant
            // high-water mark; this field is per-query.
            memory_bytes_high_water,
            rows_emitted,
        },
        truncation,
    })
}

/// Drive each plan in `plans` through the executor sequentially,
/// SHARING a single [`ExecutionContext`] so the snapshot LSN captured
/// at the first statement's first batch is observed by every
/// subsequent statement's operators.
///
/// Closes ADR-038 §5.4.1 multi-statement deferral (M4-83) per
/// ADR-038 §2 D-18 rule 2 + amendment-03 §TIER-1 GAP E rule 2: "Same
/// snapshot LSN held for all statements in a multi-statement query.
/// The LSN is acquired before statement 1's first batch pull and held
/// until the last statement's result is materialized. This guarantees
/// that a multi-statement query observes a consistent point-in-time
/// graph."
///
/// # Snapshot-LSN invariant (load-bearing)
///
/// The shared `ctx` field [`ExecutionContext::snapshot_lsn`] is an
/// `Arc<Mutex<Option<Lsn>>>` — first-batch acquire writes through
/// every clone of the context; subsequent statements observe the same
/// captured value via the mutex-guarded `Option::is_some()` branch.
/// Because we thread the SAME `ctx` into every `materialize` call, the
/// shared-LSN invariant holds without per-statement opt-in.
///
/// # Memory budget — accumulates across statements
///
/// Per W13γ fix-up LOW-1 (closes review-pr-285-final.md LOW-1): the
/// W12α [`crate::executor::MemoryBudget`] (anchored on the shared
/// `ctx` per [`ExecutionContext::budget`]) ACCUMULATES across
/// statements rather than resetting per-statement. Every statement's
/// operator stack draws from the SAME per-tenant byte budget — if
/// statement-1 reserves 80% of the cap, statement-2 sees the
/// remaining 20%; if statement-1 RELEASES its bytes pre-completion
/// (e.g., a streaming operator drops its spillover queue at EOS),
/// statement-2 sees the budget restored.
///
/// This matches the "one query = one chain" semantic per amendment-03
/// §TIER-1 GAP E rule 2: the snapshot LSN, cancellation token,
/// memory budget, and observer state ALL travel on the shared
/// [`ExecutionContext`]. Per-statement reset is forward-deferred to
/// v1.1+ if the M5-12 rate-limit / per-tenant-pool config requires it
/// (currently no consumer pin per amendment-03 §"Implicit dependency
/// edges" item 4).
///
/// The choice is documented here rather than at the
/// [`crate::executor::MemoryBudget`] surface because the
/// "accumulate vs reset" distinction is a multi-statement semantic
/// — single-statement materialization has no choice to make. The
/// W12α budget primitive is unchanged; only the multi-statement
/// composition rule is pinned.
///
/// # Error semantics
///
/// Failure of any statement aborts the whole multi-statement query;
/// no partial commit (v1.0 ArcQL is read-only per amendment-03 §TIER-1
/// GAP A — there is no "commit" surface to be partial about, but the
/// snapshot-LSN release semantics are: the LSN is released when the
/// shared `ctx` drops, regardless of which statement faulted). The
/// successful prefix's [`MaterializedResult`] vec is discarded — the
/// caller sees `Err(...)` only.
///
/// # Cancellation
///
/// The shared `ctx`'s [`crate::executor::CancellationToken`] is checked
/// at every batch boundary across every statement; firing the token
/// during statement N surfaces `ExecutionError::Cancelled` at the next
/// batch boundary and aborts subsequent statements (they never start —
/// the early-return `?` propagates upward).
///
/// # Per-statement metrics
///
/// Each [`MaterializedResult`] in the returned vec carries its own
/// [`ExecutionMetrics`] with `wall_time_ms` measured from the start of
/// THAT statement's `materialize` call (not cumulative). Callers
/// summarizing the whole multi-statement query sum the per-statement
/// `wall_time_ms` + `rows_emitted` themselves (the M4-71 row-count
/// observer integration is per-statement, not aggregate).
///
/// # Errors
///
/// Same taxonomy as [`materialize`]; the first-failing statement's
/// error short-circuits.
pub fn materialize_multi<S>(
    plans: &[LogicalPlan],
    substrate: &S,
    ctx: &ExecutionContext,
) -> Result<Vec<MaterializedResult>, ExecutionError>
where
    S: ExecutorSubstrate,
{
    // W13β fix-up M-1 + W13γ M4-83 — close-then-reopen rejection at the
    // OUTER scope. The single-statement [`materialize`] entry-point
    // would also reject, but checking here keeps the multi-statement
    // error site precise (the renderer's `feature` slot reads
    // `materialize::materialize_multi`, not `::materialize`).
    if ctx.lsn_consumed() {
        return Err(ctx.lsn_consumed_error("materialize::materialize_multi"));
    }
    // W13γ M4-83 + W13β fix-up M-1 reconciliation: hold a single
    // outer-scope LSN guard for the duration of ALL statements (per
    // amendment-03 §TIER-1 GAP E rule 2 "Same snapshot LSN held for all
    // statements in a multi-statement query"). The inner
    // [`materialize_with_outer_lsn_held`] body does NOT take its own
    // guard — so the first statement's first batch acquires a fresh
    // LSN via `ensure_snapshot_lsn`, every subsequent statement
    // observes the SAME captured LSN, and the latch ONLY lights when
    // this outer guard drops at function return. Single-statement
    // [`materialize`] still honors rule-4 release-at-query-end via its
    // own per-call guard.
    let _outer_lsn_guard = ctx.snapshot_lsn_guard();
    let mut results = Vec::with_capacity(plans.len());
    for plan in plans {
        // Cancellation, the per-tenant memory budget, and the
        // CancellationToken are Arc-shared via the same `ctx`; the
        // shared LSN survives across iterations because the
        // `_outer_lsn_guard` above keeps the slot live.
        let result = materialize_with_outer_lsn_held(plan, substrate, ctx)?;
        results.push(result);
    }
    Ok(results)
}

/// RAII guard that releases this materialize call's accumulated
/// per-tenant byte reservations on drop.
///
/// Per W13β M4-81 + amendment-03 §TIER-1 GAP E rule 4 ("Snapshot LSN
/// released at query-end / cursor-close. … Snapshot release is
/// unconditional and idempotent"): the byte reservations follow the
/// same RAII discipline as the snapshot-LSN slot. Drop releases the
/// accumulated bytes on success, error, partial-result, AND
/// panic-unwind paths.
///
/// # Why interior mutability
///
/// The materialize loop bumps `bytes` per-row from inside the loop
/// while holding `&` (not `&mut`) borrows on the guard. `Cell<u64>`
/// gives the per-row mutator path (`add_bytes`) without forcing the
/// loop to take the guard `&mut` — keeping the guard's lifetime
/// aligned with the function's stack scope.
///
/// # Why not [`crate::executor::MemoryReservation`]
///
/// W12α's [`crate::executor::MemoryReservation`] is constructed via
/// [`MemoryBudget::try_reserve`] — which itself reserves bytes
/// THROUGH the budget (one Mutex acquisition per reservation). The
/// materialize loop already reserves per-row via
/// [`MemoryBudget::try_reserve_unscoped`] (that path is the
/// W12α-pinned API for operators with multi-row spillover); wrapping
/// each row in an additional `MemoryReservation` would double-charge
/// the budget. This guard accumulates the already-reserved bytes
/// in a single drop-time release.
struct BudgetReservationGuard<'b> {
    budget: &'b MemoryBudget,
    tenant: TenantId,
    bytes: Cell<u64>,
}

impl<'b> BudgetReservationGuard<'b> {
    fn new(budget: &'b MemoryBudget, tenant: TenantId) -> Self {
        Self {
            budget,
            tenant,
            bytes: Cell::new(0),
        }
    }

    /// Bump the reservation by `bytes`. Caller has already reserved
    /// `bytes` via [`MemoryBudget::try_reserve_unscoped`]; this method
    /// only updates the guard's drop-time release count.
    #[inline]
    fn add_bytes(&self, bytes: u64) {
        self.bytes.set(self.bytes.get().saturating_add(bytes));
    }

    /// Read the accumulated reservation count without releasing.
    #[inline]
    fn bytes(&self) -> u64 {
        self.bytes.get()
    }
}

impl<'b> Drop for BudgetReservationGuard<'b> {
    fn drop(&mut self) {
        let total = self.bytes.get();
        if total > 0 {
            self.budget.release(self.tenant, total);
        }
    }
}

/// #353 — derive the ordered, user-meaningful RESULT-COLUMN NAMES for a
/// bound statement, matching openCypher / Neo4j implicit-column naming.
///
/// This is the authoritative column-name source for the wire surfaces
/// (MCP `RawQueryRows::columns`, Bolt `RunOutcome::fields`). It is
/// computed from the BOUND statement — NOT the lowered `LogicalPlan` —
/// because the bound terminal clause (the user's last `RETURN`, or a
/// standalone `CALL proc … YIELD …`, or `SHOW …`) is exactly what the
/// user asked for, whereas the lowered plan wraps the terminal `Project`
/// under `Sort`/`Distinct`/`Limit`/`Skip` nodes (and represents
/// `UNION`/`CALL`/`SHOW` output differently), which would make a
/// plan-walk fragile.
///
/// # Naming rule (per `BoundProjectionItem::display_name`)
///
/// - explicit `AS alias` → the alias;
/// - bare variable (`RETURN n`) → the variable name;
/// - any other un-aliased expression (`RETURN n.name`, `count(*)`) →
///   the verbatim source text the parser captured.
///
/// # When this returns EMPTY (wire falls back to `col_0..N`)
///
/// - **Wildcard projection** (`RETURN *`, or any item being `*` — e.g.
///   `RETURN *, x`): the output width is data-dependent on the runtime
///   row schema, which is not known at this layer. Returning empty is
///   the honest choice — we never fabricate a partial / wrong-width name
///   list. The wire renders `col_0..N` for the actual row width.
/// - **Write-only statement with no terminal RETURN** (`CREATE (n)`,
///   `MATCH (n) SET n.x = 1`): there is no projection; the result rows
///   (if any) are an implementation artifact, so no user column names
///   exist.
/// - **Index DDL**, which returns no rows.
///
/// The returned `Vec` length, when non-empty, equals the result row
/// width — the caller stamps it onto [`MaterializedResult::columns`].
#[must_use]
pub fn output_column_names(stmt: &crate::semantic::bound_ast::BoundStatement) -> Vec<String> {
    use crate::semantic::bound_ast::BoundStatement;
    match stmt {
        BoundStatement::Read(q) => output_column_names_from_query(q),
        // UNION (ADR-185): every arm exposes the SAME column-name set
        // (openCypher v9 §8, enforced at bind), so arm-0 is
        // representative. An empty `arms` cannot occur (bind requires
        // ≥2), but guard defensively.
        BoundStatement::Union(u) => u
            .arms
            .first()
            .map(output_column_names_from_query)
            .unwrap_or_default(),
        // Index creation executes but returns no rows or columns.
        BoundStatement::IndexDdl(_) => Vec::new(),
    }
}

/// #353 — column-name derivation for a single bound read query: find
/// the terminal output-producing clause (skipping standalone tail
/// `ORDER BY`/`SKIP`/`LIMIT` clauses, which never change the column
/// set) and derive names from it.
fn output_column_names_from_query(q: &crate::semantic::bound_ast::BoundQuery) -> Vec<String> {
    use crate::semantic::bound_ast::BoundClause;
    // Scan from the end for the terminal column-defining clause. Tail
    // `ORDER BY` / `SKIP` / `LIMIT` clauses (and a trailing WHERE-less
    // tail) operate on the prior clause's columns without redefining
    // them, so they are transparent for column-name purposes.
    for clause in q.clauses.iter().rev() {
        match clause {
            BoundClause::Return(r) => return projection_column_names(&r.items),
            // A standalone `CALL proc(...) YIELD a, b` (no trailing
            // RETURN) surfaces the YIELD'd columns as the result. The
            // YIELD binding's `name` is already the alias when `YIELD x
            // AS y` was used (the binder sets `binding_name = alias ||
            // column`).
            BoundClause::CallProcedure(c) => {
                return c.yields.iter().map(|y| y.var.name.clone()).collect();
            }
            // A standalone `SHOW …` surfaces its fixed column set.
            BoundClause::Show(s) => {
                return s.columns.iter().map(|v| v.name.clone()).collect();
            }
            // Tail-only clauses are transparent to the column set —
            // keep scanning leftwards for the real terminal clause.
            BoundClause::TailOrderBy(..)
            | BoundClause::TailSkip(..)
            | BoundClause::TailLimit(..) => {}
            // Any other terminal clause (a write op with no RETURN, a
            // WITH that is not the last clause, etc.) means there is no
            // user-facing projection at the tail → no column names.
            // (A WITH is never the LAST clause of a valid read query —
            // it must be followed by more clauses — so reaching a
            // non-tail, non-RETURN/CALL/SHOW clause here means a
            // write-only statement: no projection.)
            _ => return Vec::new(),
        }
    }
    Vec::new()
}

/// #353 — map an ordered list of bound projection items to their
/// display names, or EMPTY if any item is a wildcard (data-dependent
/// width — see [`output_column_names`]).
fn projection_column_names(
    items: &[crate::semantic::bound_ast::BoundProjectionItem],
) -> Vec<String> {
    use crate::semantic::bound_ast::BoundProjectionKind;
    // A wildcard expands to the runtime row schema; its width is unknown
    // at this layer, so we cannot produce a correct-width name list —
    // bail to the `col_0..N` fallback rather than emit a partial list.
    if items
        .iter()
        .any(|it| matches!(it.kind, BoundProjectionKind::Wildcard { .. }))
    {
        return Vec::new();
    }
    items
        .iter()
        .enumerate()
        .map(|(i, it)| it.display_name(i))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // 1. MaterializedResult shape pin
    // -----------------------------------------------------------------

    #[test]
    fn materialized_result_default_is_empty_zero_metrics() {
        // Pin: the Default impl produces an empty result with zero-
        // metrics. M4-91 PROFILE constructs a Default-form result if
        // the executor failed to produce ANY rows AND the caller
        // wants non-Err semantics (rare path; planner-side errors
        // surface before this).
        let r = MaterializedResult::default();
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
        assert_eq!(r.metrics.rows_emitted, 0);
        assert_eq!(r.metrics.wall_time_ms, 0);
        assert_eq!(r.metrics.memory_bytes_high_water, 0);
    }

    // -----------------------------------------------------------------
    // 2. ExecutionMetrics shape pin (forward-link surface intact)
    // -----------------------------------------------------------------

    #[test]
    fn execution_metrics_field_set_intact() {
        // Pin: the `ExecutionMetrics` struct still exposes the three
        // forward-pinned fields per amendment-03 §M5↔M4 contract
        // surface. M4-71 wiring populates `memory_bytes_high_water`
        // from M4-64a; this slice populates `wall_time_ms` +
        // `rows_emitted` end-to-end.
        let m = ExecutionMetrics {
            wall_time_ms: 17,
            memory_bytes_high_water: 0,
            rows_emitted: 4,
        };
        assert_eq!(m.wall_time_ms, 17);
        assert_eq!(m.rows_emitted, 4);
        assert_eq!(m.memory_bytes_high_water, 0);
    }

    // -----------------------------------------------------------------
    // 3. Empty-result handling pin
    // -----------------------------------------------------------------

    #[test]
    fn materialized_result_borrow_helpers() {
        // Pin: the `.rows()` / `.into_rows()` / `.metrics()` helpers
        // round-trip cleanly. v1.0 MCP renderers iterate via
        // `.rows()` (borrow) at the response-framing boundary; tests
        // that previously typed `Vec<Vec<Value>>` keep that shape via
        // `.into_rows()`.
        let r = MaterializedResult {
            rows: vec![vec![Value::Integer(1)], vec![Value::Integer(2)]],
            metrics: ExecutionMetrics {
                wall_time_ms: 3,
                memory_bytes_high_water: 0,
                rows_emitted: 2,
            },
            truncation: None,
            // #353 — column names default to empty for a directly-
            // constructed result; this test exercises the row/metrics
            // borrow helpers only.
            columns: Vec::new(),
        };
        assert_eq!(r.rows().len(), 2);
        assert_eq!(r.columns().len(), 0);
        assert_eq!(r.metrics().wall_time_ms, 3);
        assert_eq!(r.len(), 2);
        assert!(!r.is_truncated());
        assert!(r.truncation().is_none());
        let rows = r.into_rows();
        assert_eq!(rows.len(), 2);
    }

    // -----------------------------------------------------------------
    // 4. Large-result chunked materialization pin (≥ BATCH_ROWS rows)
    // -----------------------------------------------------------------
    //
    // Drives the materialize loop across multiple BATCH_ROWS-sized
    // batches; pins the loop's correctness across the batch boundary
    // (the M4-61 executor's per-2048-row chunking).

    #[test]
    fn materialize_drains_multiple_batches_into_single_vec() {
        use crate::executor::batch::{BATCH_ROWS, Batch};

        // The materialize loop asserts batch-boundary correctness by
        // exhaustion; here we drive the equivalent loop on a synthetic
        // per-batch chunked row stream (no executor bind required;
        // executor wiring is covered end-to-end by the integration
        // tests). The pin is: walk N > BATCH_ROWS rows in chunks of
        // ≤ BATCH_ROWS each, drain into a single Vec, observe
        // ordering preserved.
        let total_rows: usize = BATCH_ROWS * 2 + 7; // 4103 rows → 3 batches
        let mut remaining = total_rows;
        let mut emitted: u64 = 0;
        let mut rows: Vec<Vec<Value>> = Vec::new();
        loop {
            let take = remaining.min(BATCH_ROWS);
            let mut b = Batch::with_capacity(1);
            for _ in 0..take {
                let v = Value::Integer(emitted as i64);
                emitted += 1;
                assert!(b.push_row(vec![v]), "push_row at batch capacity");
            }
            remaining -= take;
            if b.is_empty() {
                break;
            }
            for row in b.into_rows() {
                rows.push(row);
            }
            if remaining == 0 {
                // Final empty batch to mirror the materialize-loop
                // termination condition.
                let final_b = Batch::with_capacity(1);
                if final_b.is_empty() {
                    break;
                }
            }
        }
        assert_eq!(rows.len(), total_rows);
        // Sanity: row 0 carries Integer(0), row N-1 carries Integer(N-1)
        // — multi-batch ordering preserved across batch boundaries.
        assert_eq!(rows[0][0], Value::Integer(0));
        assert_eq!(
            rows[total_rows - 1][0],
            Value::Integer(total_rows as i64 - 1)
        );
    }

    // -----------------------------------------------------------------
    // 5. M4-83 multi-statement materialize shape pin
    // -----------------------------------------------------------------
    //
    // Pin: `materialize_multi` over an empty `plans` slice returns an
    // empty `Vec<MaterializedResult>` without error and without
    // capturing a snapshot LSN. The non-trivial path (≥1 plan) is
    // exercised end-to-end by
    // `tests/m4_83_multi_statement_integration.rs`.

    #[test]
    fn materialize_multi_empty_plans_returns_empty_vec() {
        use crate::executor::{ExecutionContext, StubExecutorSubstrate};
        use arcgraph_core::{PartitionId, TenantId};
        let sub = StubExecutorSubstrate::new();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let results: Vec<MaterializedResult> =
            materialize_multi(&[], &sub, &ctx).expect("materialize_multi over empty slice");
        assert!(results.is_empty());
        // Defense in depth: an empty plan slice MUST NOT eagerly
        // capture the snapshot LSN (the lazy-capture invariant per
        // ADR-038 §2 D-18 rule 1 is held only by `materialize`'s first
        // batch — empty slice never reaches a first batch).
        assert_eq!(ctx.snapshot_lsn(), None, "empty multi must not capture LSN");
    }

    // -----------------------------------------------------------------
    // 6. W13γ fix-up LOW-1 — MemoryBudget accumulates across statements
    // -----------------------------------------------------------------

    /// Closes review-pr-285-final.md LOW-1: pin the documented
    /// "MemoryBudget accumulates across statements" choice via a
    /// direct primitive-level test.
    ///
    /// The shared `ctx` carries one `MemoryBudget`; reservations from
    /// statement-1 reduce the headroom seen by statement-2. We don't
    /// drive a full multi-statement materialize_multi here (no
    /// executor operator currently calls `try_reserve_unscoped` —
    /// the budget is wired forward to M4-64a's MemoryTracker per
    /// `executor::context::ExecutionContext::budget`'s rustdoc); we
    /// instead pin the primitive: a single shared `MemoryBudget`
    /// reachable through `ctx.budget()` carries reservation state
    /// across calls. A regression that re-creates the budget per
    /// statement (e.g., a refactor that calls
    /// `ExecutionContext::with_budget(MemoryBudget::new())` inside
    /// `materialize_multi`'s for-loop) would manifest as
    /// statement-2 seeing a fresh-cap budget; this test catches that
    /// shape via reservation persistence.
    #[test]
    fn materialize_multi_shares_memory_budget_across_statements() {
        use crate::executor::{ExecutionContext, MemoryBudget};
        use arcgraph_core::{PartitionId, TenantId};
        let cap_bytes = 1_024u64;
        let tenant = TenantId::DEFAULT;
        let budget = MemoryBudget::with_per_tenant_cap(tenant, cap_bytes);
        let ctx = ExecutionContext::new(tenant, PartitionId::ZERO).with_budget(budget);
        // "Statement 1" reserves 800 bytes via the same handle a real
        // operator would acquire.
        ctx.budget()
            .try_reserve_unscoped(tenant, 800, "stmt1_op")
            .expect("stmt-1 reservation succeeds (800 ≤ 1024)");
        // "Statement 2" attempts to reserve 300 bytes — should FAIL
        // because only 224 bytes remain after stmt-1's 800 reservation.
        // If the budget had reset between statements, this would
        // succeed (300 ≤ 1024 fresh cap).
        let r2 = ctx.budget().try_reserve_unscoped(tenant, 300, "stmt2_op");
        assert!(
            r2.is_err(),
            "stmt-2's 300-byte reservation MUST FAIL — only 224 bytes \
             remain after stmt-1's 800-byte reservation; if the budget \
             reset per statement this assertion would fail. The shared \
             ctx semantic across statements is the load-bearing \
             invariant per ADR-038 amendment-03 §TIER-1 GAP E rule 2."
        );
        // "Statement 3" attempts to reserve 200 bytes — fits in the
        // remaining 224. Pins that the residual budget is correctly
        // 224, not 1024 (the full cap) and not 0 (over-debited).
        ctx.budget()
            .try_reserve_unscoped(tenant, 200, "stmt3_op")
            .expect("stmt-3 reservation succeeds (200 ≤ 224 remaining)");
        // After stmt-1 RELEASES its bytes (e.g., a streaming operator
        // drops its spillover queue at EOS), stmt-N+1 sees the budget
        // restored. Pins the bidirectional accumulation: not just
        // monotonically-debited, but properly Arc-shared so releases
        // propagate.
        ctx.budget().release(tenant, 800);
        ctx.budget()
            .try_reserve_unscoped(tenant, 700, "stmt4_op")
            .expect("post-release: 700 ≤ (1024 - 200) = 824 remaining");
    }

    // -----------------------------------------------------------------
    // 7. D-2 (ADR-147 §D-8) — statement-scoped autocommit txn wrapper
    // -----------------------------------------------------------------

    use crate::executor::substrate::HeldTxnHandle;
    use crate::executor::{BoundEdge, BoundNode, RankedHit, SubstrateAccessError};
    use crate::logical_plan::{Direction, LogicalCreateNode, LogicalPlan, LogicalScan};
    use arcgraph_core::{LabelId, Lsn, NodeId, PartitionId, TenantId, TypeId};
    use std::sync::Mutex;

    /// A recording substrate that logs the D-2 statement-txn lifecycle
    /// calls (`begin`/`commit`/`rollback`) so the wrapper's dispatch is
    /// assertable. `fail_create_node` injects a mid-statement fault to
    /// exercise the rollback route. A `Mutex<Vec<_>>` log satisfies the
    /// trait's `Send + Sync` bound without `unsafe` (uncontended — the
    /// materialize drive is single-threaded).
    #[derive(Default)]
    struct RecordingSubstrate {
        events: Mutex<Vec<&'static str>>,
        fail_create_node: bool,
    }

    impl RecordingSubstrate {
        fn log(&self) -> Vec<&'static str> {
            self.events.lock().expect("uncontended test log").clone()
        }
        fn push(&self, ev: &'static str) {
            self.events.lock().expect("uncontended test log").push(ev);
        }
    }

    impl ExecutorSubstrate for RecordingSubstrate {
        fn begin_statement(&self, _ctx: &ExecutionContext) -> Result<(), SubstrateAccessError> {
            self.push("begin");
            Ok(())
        }
        fn commit_statement(&self, _ctx: &ExecutionContext) -> Result<(), SubstrateAccessError> {
            self.push("commit");
            Ok(())
        }
        fn rollback_statement(&self, _ctx: &ExecutionContext) {
            self.push("rollback");
        }
        fn create_node(
            &self,
            _tenant: TenantId,
            _label: Option<&str>,
            _properties: &[(String, Value)],
            _ctx: &ExecutionContext,
        ) -> Result<NodeId, SubstrateAccessError> {
            self.push("create_node");
            if self.fail_create_node {
                return Err(SubstrateAccessError::Io(
                    "injected create_node fault".into(),
                ));
            }
            Ok(NodeId::new(1))
        }
        fn scan_nodes(
            &self,
            _tenant: TenantId,
            _label: Option<LabelId>,
            _read_lsn: Lsn,
        ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
            Ok(Vec::new())
        }
        fn expand(
            &self,
            _tenant: TenantId,
            _from: NodeId,
            _rel_type: Option<TypeId>,
            _direction: Direction,
            _read_lsn: Lsn,
        ) -> Result<Vec<BoundEdge>, SubstrateAccessError> {
            Ok(Vec::new())
        }
        fn expand_cursor(
            &self,
            _tenant: TenantId,
            _from: NodeId,
            _rel_type: Option<TypeId>,
            _direction: Direction,
            _read_lsn: Lsn,
        ) -> Result<crate::executor::BoundEdgeCursor, SubstrateAccessError> {
            Ok(Box::new(std::iter::empty()))
        }
        fn vector_search(
            &self,
            _t: TenantId,
            _p: &str,
            _q: &[f32],
            _k: u64,
            _l: Lsn,
        ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
            Ok(Vec::new())
        }
        fn bm25_search(
            &self,
            _t: TenantId,
            _p: &str,
            _q: &str,
            _k: u64,
            _l: Lsn,
        ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
            Ok(Vec::new())
        }
        fn community_members(
            &self,
            _t: TenantId,
            _c: i64,
            _l: Lsn,
        ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
            Ok(Vec::new())
        }
    }

    /// A no-op held-txn handle for the nesting-guard test (models a Bolt
    /// BEGIN…COMMIT already owning the tx lifetime).
    #[derive(Debug)]
    struct FakeHeld;
    impl HeldTxnHandle for FakeHeld {
        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }
        fn snapshot_lsn(&self) -> Lsn {
            Lsn::MAX
        }
    }

    fn write_plan() -> LogicalPlan {
        LogicalPlan::CreateNode(LogicalCreateNode {
            var: Some(crate::semantic::bound_ast::BindingId::new(0)),
            label: Some("User".into()),
            properties: Vec::new(),
            input: None,
            span: crate::error::Span::point(0, 0),
        })
    }

    fn read_plan() -> LogicalPlan {
        LogicalPlan::Scan(LogicalScan {
            label: None,
            var: crate::semantic::bound_ast::BindingId::new(0),
            read_lsn: Lsn::MAX,
            span: crate::error::Span::point(0, 0),
        })
    }

    #[test]
    fn d2_write_statement_begins_and_commits_once() {
        // A write plan in AUTO-COMMIT mode: materialize opens ONE
        // statement txn (begin) BEFORE the ops, and commits ONCE after
        // the drive succeeds. The op stages BETWEEN begin and commit.
        let sub = RecordingSubstrate::default();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        materialize(&write_plan(), &sub, &ctx).expect("write materialize OK");
        assert_eq!(
            sub.log(),
            vec!["begin", "create_node", "commit"],
            "D-2: begin-once → stage → commit-once for an AUTO-COMMIT write"
        );
    }

    #[test]
    fn d2_read_statement_opens_no_statement_txn() {
        // A read plan does not mutate → no begin/commit/rollback.
        let sub = RecordingSubstrate::default();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        materialize(&read_plan(), &sub, &ctx).expect("read materialize OK");
        assert!(
            sub.log().is_empty(),
            "a read-only statement opens no statement txn; got {:?}",
            sub.log()
        );
    }

    #[test]
    fn d2_write_statement_failing_op_rolls_back() {
        // A write whose op FAILS routes to rollback (not commit); the
        // whole statement is discarded.
        let sub = RecordingSubstrate {
            fail_create_node: true,
            ..Default::default()
        };
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let err = materialize(&write_plan(), &sub, &ctx).expect_err("op fault surfaces");
        assert!(format!("{err:?}").contains("injected create_node fault"));
        assert_eq!(
            sub.log(),
            vec!["begin", "create_node", "rollback"],
            "D-2: a failing op routes to rollback, NOT commit — full statement rollback"
        );
    }

    #[test]
    fn d2_write_inside_held_txn_does_not_nest_statement_txn() {
        // NESTING GUARD: if a held txn is already installed (an explicit
        // Bolt BEGIN…COMMIT owns the multi-statement lifetime), D-2 must
        // NOT open a statement txn — the explicit COMMIT commits. So NO
        // begin/commit/rollback fires from the materialize wrapper (the
        // op still stages into the held tx via the substrate's own path).
        let sub = RecordingSubstrate::default();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO)
            .with_held_txn(Box::new(FakeHeld));
        materialize(&write_plan(), &sub, &ctx).expect("held-txn write materialize OK");
        assert_eq!(
            sub.log(),
            vec!["create_node"],
            "D-2 must NOT open a statement txn inside an explicit held tx \
             (no begin/commit/rollback); the op stages into the held tx"
        );
        // The held tx is untouched by D-2 (the explicit COMMIT reclaims it).
        assert!(
            ctx.has_held_txn(),
            "the explicit held tx survives — D-2 did not consume it"
        );
    }
}
