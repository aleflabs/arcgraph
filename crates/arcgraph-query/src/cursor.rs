//! M4-82 streaming cursor + yield-batch protocol.
//!
//! Lit at v1.0 per ADR-038 amendment-02 §M4.h ("streaming cursor /
//! yield-batch protocol for large results") + amendment-03 §M5↔M4
//! contract surface §11 D-9 ("Implementation lands across M4-08a
//! `execute` row-materialization tail, M4-08b `execute` streaming-
//! cursor tail, M4-91 `explain` + `profile`, M4-92 `cancel` + `timeout`
//! plumbing").
//!
//! # Slice scope (M4-82 — M4-08b)
//!
//! - **[`StreamingCursor`]** — the v1.0-alpha cursor type. Wraps a
//!   [`Pipeline`]-built [`PhysicalOperator`] tree + an
//!   [`ExecutionContext`] + a borrowed [`ExecutorSubstrate`]. Lifecycle:
//!   1. **`open(plan, ctx, substrate)`** — builds the pipeline + binds
//!      the cursor to the supplied context. The snapshot LSN is NOT
//!      acquired eagerly here (per ADR-038 §2 D-18 rule 1 lazy
//!      capture / execute-time, pre-first-batch); it lazies-out at the
//!      first `next_batch` call.
//!   2. **`next_batch()` loop** — emits `Vec<Row>` per call until
//!      exhausted (Returns `Ok(None)` on EOS). Per amendment-03
//!      §TIER-1 GAP C, the cursor checks the
//!      [`crate::CancellationToken`] at every batch boundary; on
//!      cancel, drops in-flight rows + releases LSN.
//!   3. **`close()` / `Drop`** — releases the snapshot LSN, the
//!      per-tenant budget bytes accumulated during streaming, the
//!      buffer-pool pins (forward-method — substrate-side responsibility
//!      at v1.0-alpha), and the plan-cache read lock (forward-method —
//!      v1.0-alpha cache uses brief Mutex sections, not long-lived
//!      RwLock reads).
//!
//! # Why a sibling type to [`crate::MaterializedResult`] (not a Vec replacement)
//!
//! Per ADR-038 amendment-02 §M4.h + the W13β prompt's "the streaming
//! variant ships as a sibling `MaterializedCursor` type, NOT a Vec
//! replacement, per the 7-slice 3-strike scaffolding rule — we do not
//! generalize prematurely":
//!
//! 1. **`MaterializedResult`** is the eager, bounded-memory return
//!    type — appropriate for queries the caller sizes via the M4-81
//!    per-tenant budget cap. The MCP renderer reads `rows()` once.
//! 2. **`StreamingCursor`** is the unbounded, back-pressure-friendly
//!    return type — appropriate for large-result queries that cross
//!    the budget cap or whose row count is unknown a priori. The MCP
//!    renderer pulls `next_batch()` per response chunk, allowing
//!    incremental wire emission.
//!
//! Both types share the same memory-budget primitive
//! ([`crate::executor::MemoryBudget`]) and the same snapshot-LSN
//! discipline ([`crate::executor::SnapshotLsnGuard`]). The contract
//! surface stays uniform across the two return shapes.
//!
//! # RAII close discipline (W12γ MED-3 lesson)
//!
//! Per `feedback_seqlock_panic_safety_primitive.md` — "RAII guard is
//! the canonical panic-safety primitive when cleanup is a single
//! mutation" — the cursor's [`Drop`] impl calls a `close_internal`
//! method that is idempotent across:
//!
//! - **Explicit `close()` followed by `Drop`** — `close()` sets the
//!   `closed` flag; `Drop`'s `close_internal` is a no-op.
//! - **`Drop` without `close()`** — the disconnect / panic-unwind /
//!   "caller forgot to close" path. `close_internal` releases the
//!   resources via the same code path as explicit `close()`.
//! - **Drop after partial cancellation** — when `next_batch` returns
//!   `Err(Cancelled)`, the cursor sets `closed = true` so subsequent
//!   `next_batch` calls return a "cursor closed" error and `Drop`'s
//!   `close_internal` is a no-op.
//!
//! The pattern matches the W12γ MED-3 `RegistryGuard` in
//! `explain.rs` — both follow "no leak on panic" discipline.
//!
//! # M4-72 replan ↔ cursor handoff
//!
//! Per the W13β spawn prompt's "M4-72 replan path from W12β: cursor
//! must INTEROPERATE with replan (a mid-stream replan should resume
//! the cursor on the new plan; verify there's a clean handoff API or
//! document the limitation)":
//!
//! v1.0-alpha LIMITATION: the cursor does NOT support mid-stream
//! replan handoff. The [`crate::observer::ReplanController`] is a
//! POST-EXECUTE controller at v1.0-alpha (per its module docs:
//! "replan is a POST-EXECUTE step: the original execution
//! materializes its results, the controller reads the observer's
//! breaches, replan-then-re-execute happens against the new plan if
//! breach detected"). The cursor's `next_batch` does NOT consult the
//! observer's threshold breaches mid-stream.
//!
//! The forward-method shape is documented at
//! [`StreamingCursor::replan_in_place`] (returns a `NotImplemented`
//! ArcQLError until a v1.1 slice lights the mid-stream-handoff
//! infrastructure — that slice's natural inflection point is the
//! `MidQueryState` opaque token already pinned in
//! `crate::observer::replan::MidQueryState`).
//!
//! # Close-then-reopen REJECTS at v1.0 (W13β fix-up M-1)
//!
//! v1.0 contract per ADR-038 amendment-03 §TIER-1 GAP E rule 5
//! ("All operators in a single `ExecutionContext` share the same
//! snapshot LSN; replan does NOT re-acquire"): once a
//! `StreamingCursor` (or `materialize::materialize`) releases the
//! snapshot LSN on a context, that context is *consumed*. A
//! subsequent [`StreamingCursor::open`] (or `materialize` call) on
//! the same context — even via a clone — REJECTS with
//! [`crate::semantic::error::ArcQLError::Internal`] rather than
//! silently re-acquiring a fresh LSN.
//!
//! Why reject: at production-LSN-binding time (M4-08+ when
//! `ensure_snapshot_lsn` returns real WAL state instead of the
//! `Lsn::MAX` v1.0-alpha sentinel), close-then-reopen would observe
//! a different point-in-time on the second cursor than the first —
//! breaking openCypher snapshot semantics. The "replan reuses the
//! same LSN" pattern requires either (a) the caller does NOT call
//! `close()` between the original cursor and the replan'd cursor
//! (the snapshot-LSN slot stays `Some(_)` so the second cursor's
//! lazy `ensure_snapshot_lsn` returns the captured value) — but at
//! v1.0-alpha there is no documented user-accessible API for that
//! shape; or (b) a v1.1 `replan_seam` API that hands `ExecutionContext`
//! ownership across cursors before `close_internal` runs. See
//! [`StreamingCursor::replan_in_place`] for the v1.1 inflection-point
//! stub.
//!
//! Detection: [`crate::executor::ExecutionContext::lsn_consumed`] is
//! a sticky `AtomicBool` set when [`crate::executor::ExecutionContext::release_snapshot_lsn`]
//! observes a `Some(_)` slot. The latch survives clones (shared
//! `Arc<AtomicBool>`), so a "clone the ctx; close cursor1; reopen
//! cursor2 on the clone" sequence rejects at cursor2.open's preflight
//! check.
//!
//! # Why no separate `materialize_cursor` constructor on QueryEngine
//!
//! Per the 7-slice 3-strike pattern + this slice's bounded scope: the
//! [`crate::QueryEngine`] surface at W13β does NOT add a
//! `cursor(query)` method. Adding one is a v1.1 inflection point
//! when the M5 server tier needs streaming-cursor end-to-end. v1.0-
//! alpha exposes [`StreamingCursor::open`] as a free constructor; the
//! M5 wiring slice adds the `QueryEngine::cursor` method when the
//! second consumer (MCP / Bolt server) lights.
//!
//! # ADR provenance
//! - **ADR-038 amendment-02 §M4.h** — primary M4-82 cite.
//! - **ADR-038 amendment-03 §TIER-1 GAP E rule 4** — snapshot-LSN
//!   release at cursor-close.
//! - **ADR-038 amendment-03 §TIER-1 GAP C** — cancellation contract
//!   (per-batch boundary check; drops in-flight state on trip).
//! - **ADR-038 amendment-03 §M5↔M4 contract surface §11 D-9** —
//!   M4-08b is the streaming-cursor tail of `execute`.
//! - **`feedback_seqlock_panic_safety_primitive.md`** — RAII
//!   panic-safety discipline.
//! - **`feedback_avoid_speculative_scaffolding.md`** — sibling type
//!   discipline (NOT generalize prematurely).

use crate::executor::budget::estimate_row_bytes;
use crate::executor::error::ExecutionError;
use crate::executor::{ExecutionContext, ExecutorSubstrate, PhysicalOperator, Pipeline, Value};
use crate::logical_plan::LogicalPlan;
use crate::semantic::error::ArcQLError;
use arcgraph_core::TenantId;

/// M4-82 streaming cursor — yield-batch surface for large-result
/// queries.
///
/// See module docs for design rationale. Construct via
/// [`Self::open`]; iterate via [`Self::next_batch`]; finalize via
/// [`Self::close`] (or rely on [`Drop`]).
///
/// # Lifetimes
///
/// `'sub` is the substrate borrow lifetime — the substrate's
/// `&dyn ExecutorSubstrate` (or concrete type) MUST outlive the
/// cursor. v1.0-alpha tests use `StubExecutorSubstrate` which is
/// stack-allocated; production wiring at M4-08+ uses an
/// `Arc<dyn ExecutorSubstrate>` carried by the M5 server tier.
///
/// # Send / Sync
///
/// The cursor is NOT `Send` (it owns a `&'sub S` borrow + an
/// `ExecutionContext` whose internal state is `Arc<Mutex<_>>`-backed).
/// At v1.0-alpha the cursor is consumed on the same thread that
/// constructed it. M5-12's async server wraps the cursor in a Tokio
/// task; the wrapper handles the cross-thread boundary via
/// `tokio::sync::Mutex` or per-tenant single-thread executors per
/// the design-v2 §4.1 thread-per-core convention.
pub struct StreamingCursor<'sub, S: ExecutorSubstrate> {
    /// Operator pipeline state machine. Built once at `open()` time.
    op: PhysicalOperator,
    /// Per-query execution context (tenant, partition, query_id,
    /// snapshot-LSN slot, cancellation token, budget). Owned by the
    /// cursor so the cursor's `Drop` can call
    /// [`ExecutionContext::release_snapshot_lsn`] on cleanup.
    ctx: ExecutionContext,
    /// Borrowed substrate handle for `next_batch` dispatch. Lifetime
    /// `'sub` ensures the substrate outlives the cursor.
    substrate: &'sub S,
    /// Cached tenant ID — copied at construction so the close path
    /// doesn't need to re-borrow `ctx`.
    tenant: TenantId,
    /// Per-tenant budget bytes the cursor is CURRENTLY HOLDING
    /// (charged via [`MemoryBudget::try_reserve_unscoped`] but not yet
    /// released back via [`MemoryBudget::release`]).
    ///
    /// Per W13β fix-up M-2 (release-on-emit), `next_batch` charges,
    /// emits, and releases within a single call — so this field is
    /// `0` at the next-batch boundary on the success path. Non-zero
    /// values appear ONLY in the narrow panic window between
    /// `try_reserve_unscoped` and the symmetric `release` call;
    /// `close_internal` reads this field as a backstop and releases
    /// any leaked bytes on Drop / cancel / error paths so the
    /// per-tenant counter naturally heals across panic-unwind.
    bytes_reserved: u64,
    /// Lifecycle flag. `true` after `close()` / panic-unwind cleanup
    /// has run; subsequent `next_batch` calls return
    /// [`StreamingCursorError::Closed`].
    closed: bool,
    /// Total rows emitted so far across all `next_batch` calls.
    /// Diagnostic + future M4-91 PROFILE per-cursor metrics surface.
    rows_emitted: u64,
}

impl<'sub, S: ExecutorSubstrate> StreamingCursor<'sub, S> {
    /// Open a streaming cursor over `plan` with the supplied
    /// `ctx` + `substrate`.
    ///
    /// # Snapshot-LSN discipline
    ///
    /// The cursor's `open()` does NOT acquire the snapshot LSN
    /// eagerly — per ADR-038 §2 D-18 rule 1, the LSN lazies-out at
    /// the first `next_batch` call (which the operator pipeline
    /// triggers via [`ExecutionContext::ensure_snapshot_lsn`]).
    /// EXPLAIN's no-LSN discipline (also rule 1's EXPLAIN exception)
    /// is honored because EXPLAIN never opens a cursor.
    ///
    /// # Pipeline-build errors
    ///
    /// Forwards [`ExecutionError`] from
    /// [`Pipeline::build`]. Most often this surfaces
    /// [`ExecutionError::NotImplemented`] for a plan operator
    /// reserved for a forward slice (e.g., the v1.0-alpha
    /// `Aggregate` / `Sort` paths that are admissible at the
    /// pipeline-build layer). The caller pattern-matches per the
    /// M5↔M4 contract surface.
    pub fn open(
        plan: &LogicalPlan,
        ctx: ExecutionContext,
        substrate: &'sub S,
    ) -> Result<Self, ExecutionError> {
        // W13β fix-up M-1: reject close-then-reopen. A context whose
        // snapshot LSN was previously released by a sibling cursor /
        // materialize call carries the consumption latch; opening a
        // new cursor on it would silently re-acquire a fresh LSN at
        // M4-08+ production wiring (when `ensure_snapshot_lsn` reads
        // real WAL state) — violating ADR-038 amendment-03 §TIER-1
        // GAP E rule 5. See module docs §"Close-then-reopen REJECTS
        // at v1.0".
        if ctx.lsn_consumed() {
            return Err(ctx.lsn_consumed_error("StreamingCursor::open"));
        }
        // #797 — param-aware build (the streaming cursor reads the same
        // per-query parameter bag the materialize path does; empty for
        // non-parameterized cursors).
        let op = Pipeline::build_with_parameters(plan, ctx.parameters())?;
        let tenant = ctx.tenant();
        Ok(Self {
            op,
            ctx,
            substrate,
            tenant,
            bytes_reserved: 0,
            closed: false,
            rows_emitted: 0,
        })
    }

    /// Pull the next batch of rows from the cursor.
    ///
    /// Returns:
    /// - **`Ok(Some(rows))`** — `rows` is a `Vec<Vec<Value>>` of size
    ///   ≤ [`crate::executor::BATCH_ROWS`] (per the executor's
    ///   factorized batch convention). Empty `rows` is NOT returned;
    ///   the cursor surfaces EOS via `Ok(None)` instead.
    /// - **`Ok(None)`** — end-of-stream. The cursor is exhausted;
    ///   subsequent `next_batch` calls also return `Ok(None)`. The
    ///   caller may still call `close()` to finalize cleanup
    ///   eagerly (Drop runs the same path otherwise).
    /// - **`Err(ExecutionError::Cancelled)`** — the
    ///   [`crate::CancellationToken`] tripped at the batch boundary.
    ///   The cursor is auto-closed (subsequent `next_batch` returns
    ///   `Err(Cancelled)` until the cursor is dropped). In-flight
    ///   rows are discarded.
    /// - **`Err(ExecutionError::Plan(ArcQLError::ResourceExhausted))`** —
    ///   the per-tenant budget cap was exceeded by the current
    ///   batch's bytes. Per the M4-81 partial-result convention, the
    ///   batch's rows are NOT admitted (the budget tracker did not
    ///   bump). The cursor is auto-closed.
    /// - **`Err(ExecutionError::*)`** — any other executor error
    ///   (Substrate / Eval / NotImplemented). The cursor is auto-
    ///   closed.
    ///
    /// # Cancellation discipline
    ///
    /// Per amendment-03 §TIER-1 GAP C, this method checks the
    /// [`crate::CancellationToken`] BEFORE invoking the operator
    /// pipeline's `next_batch`. The redundant check (the operator
    /// dispatcher checks too) is defense-in-depth — keeping the
    /// cursor's API contract stable even if a future operator
    /// refactor moves the dispatcher's check.
    ///
    /// # Memory budget (W13β fix-up M-2 — release-on-emit)
    ///
    /// The cursor charges the WHOLE batch's bytes against the per-
    /// tenant budget after the batch is produced (as a single bump
    /// rather than per-row, to amortize the Mutex acquisition cost),
    /// then RELEASES those bytes back to the per-tenant counter
    /// IMMEDIATELY before returning the rows to the caller. The
    /// per-tenant `current_bytes` reading therefore reflects "bytes
    /// currently in flight INSIDE the cursor" (zero between batches),
    /// not "bytes the cursor has emitted" (monotonic over the cursor's
    /// lifetime).
    ///
    /// Why charge-then-release rather than no-charge: the charge step
    /// preserves cap enforcement — a batch whose bytes would push the
    /// per-tenant total past the configured cap rejects with
    /// `ArcQLError::ResourceExhausted` (the rejecting `try_reserve`
    /// does NOT bump the counter per W12α's
    /// `MemoryBudget::try_reserve_unscoped` convention). Once the
    /// charge succeeds, the rows are about to leave the cursor's
    /// ownership, so the counter releases — back-pressure-aware
    /// callers see a counter that mirrors actual cursor working-set
    /// pressure rather than cumulative emit total.
    ///
    /// The `bytes_reserved` field tracks "bytes currently held by the
    /// cursor" — non-zero between the charge step and the release
    /// step within `next_batch`, zero on the next-batch boundary.
    /// Drop-time `close_internal` releases `bytes_reserved` as a
    /// backstop for the panic-mid-batch path (between charge and
    /// release).
    pub fn next_batch(&mut self) -> Result<Option<Vec<Vec<Value>>>, ExecutionError> {
        if self.closed {
            // W13β fix-up N-1: lifecycle-invariant violation, not a
            // deferred-feature signal. Use ArcQLError::Internal per
            // PR #287 review NIT-1; the M5-tier transport renderers
            // get a "client misuse" diagnostic distinct from
            // NotImplemented's "deferred feature" rendering.
            return Err(ExecutionError::Plan(ArcQLError::Internal {
                feature: "StreamingCursor::next_batch".into(),
                reason: "next_batch called on a closed cursor (M4-82 lifecycle invariant)".into(),
                span: crate::error::Span::point(0, 0),
            }));
        }
        // Cancellation gate — defense-in-depth (the operator
        // dispatcher checks too per `executor::ops::mod` line 110).
        if self.ctx.cancellation().check().is_err() {
            self.close_internal();
            return Err(ExecutionError::Cancelled);
        }
        let batch = match self.op.next_batch(&self.ctx, self.substrate) {
            Ok(b) => b,
            Err(e) => {
                // Auto-close on any executor error so subsequent
                // next_batch calls don't drive a half-broken
                // pipeline.
                self.close_internal();
                return Err(e);
            }
        };
        if batch.is_empty() {
            // EOS — close eagerly so the caller's bookkeeping (LSN
            // release, budget release) matches "stream exhausted".
            self.close_internal();
            return Ok(None);
        }
        let rows = batch.into_rows();
        // Charge the whole batch's bytes as a single budget reserve.
        let bytes_this_batch: u64 = rows.iter().map(|r| estimate_row_bytes(r) as u64).sum();
        match self.ctx.budget().try_reserve_unscoped(
            self.tenant,
            bytes_this_batch,
            "M4-82 cursor next_batch",
        ) {
            Ok(()) => {
                // Track the in-flight bytes so a panic between here
                // and the release call below is covered by Drop's
                // close_internal backstop release.
                self.bytes_reserved = self.bytes_reserved.saturating_add(bytes_this_batch);
            }
            Err(e) => {
                // ResourceExhausted OR any other Err — auto-close
                // and propagate. The current batch's rows are NOT
                // emitted (caller can recover partial state from
                // prior batches' returned rows; the cursor is the
                // streaming surface for that — we do not re-emit
                // here).
                self.close_internal();
                return Err(e);
            }
        }
        self.rows_emitted = self.rows_emitted.saturating_add(rows.len() as u64);
        // W13β fix-up M-2: release the just-reserved bytes back to
        // the per-tenant counter before handing rows to the caller.
        // The rows are about to leave the cursor's ownership; the
        // budget counter should reflect that. The transient
        // self.bytes_reserved bump above is the panic-window guard
        // for Drop's backstop release; we zero both sides here.
        self.ctx.budget().release(self.tenant, bytes_this_batch);
        self.bytes_reserved = self.bytes_reserved.saturating_sub(bytes_this_batch);
        Ok(Some(rows))
    }

    /// Total rows emitted so far across all [`Self::next_batch`]
    /// calls. Diagnostic surface; future M4-91 PROFILE per-cursor
    /// metrics consumer (when M4-91 grows a cursor-mode renderer per
    /// amendment-03 §TIER-2-c).
    #[inline]
    #[must_use]
    pub fn rows_emitted(&self) -> u64 {
        self.rows_emitted
    }

    /// `true` iff the cursor has been closed (explicitly via
    /// [`Self::close`], by EOS, by cancellation, or by error).
    #[inline]
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.closed
    }

    /// Per-tenant budget bytes the cursor is CURRENTLY HOLDING
    /// (charged but not yet released).
    ///
    /// Per W13β fix-up M-2 (release-on-emit), each `next_batch`
    /// charges-then-releases within the call, so this returns `0` on
    /// the next-batch boundary on the success path. Non-zero values
    /// appear ONLY in the narrow panic window between
    /// `try_reserve_unscoped` and the symmetric `release` call.
    /// `close_internal` reads this field as a backstop and releases
    /// the bytes on Drop / cancel / error paths so the per-tenant
    /// counter naturally heals across panic-unwind.
    #[inline]
    #[must_use]
    pub fn bytes_reserved(&self) -> u64 {
        self.bytes_reserved
    }

    /// Close the cursor explicitly — releases the snapshot LSN, the
    /// per-tenant budget bytes, and (forward-method) the buffer-pool
    /// pins + plan-cache read lock. Idempotent: calling on a closed
    /// cursor is a no-op.
    ///
    /// Per the RAII discipline in module docs §"RAII close
    /// discipline", the [`Drop`] impl runs this same path; explicit
    /// `close()` is the canonical "flush bookkeeping eagerly" call.
    /// The `mut self` consumption ensures the cursor is unusable
    /// afterward at the type level (a future `next_batch` call would
    /// fail to compile, NOT just at runtime).
    pub fn close(mut self) -> Result<(), ExecutionError> {
        self.close_internal();
        Ok(())
    }

    /// Forward-method: in-place mid-stream replan handoff.
    ///
    /// At v1.0-alpha this returns
    /// [`ArcQLError::NotImplemented`] — the
    /// [`crate::observer::ReplanController`] is a POST-EXECUTE
    /// controller; mid-stream handoff requires v1.1 sub-plan
    /// splitting + intermediate-state preservation. See module docs
    /// §"M4-72 replan ↔ cursor handoff" for the inflection-point
    /// rationale.
    ///
    /// Callers needing replan close the cursor, run replan, and open
    /// a fresh cursor — see `tests/wave_13_beta_streaming_to_observer_transit_pin.rs`
    /// for the end-to-end shape.
    pub fn replan_in_place(&mut self) -> Result<(), ExecutionError> {
        Err(ExecutionError::Plan(ArcQLError::NotImplemented {
            feature: "StreamingCursor::replan_in_place mid-stream replan handoff".into(),
            section: "M4-82 → v1.1 cursor mid-stream handoff inflection point".into(),
            target_version: "v1.1".into(),
            span: crate::error::Span::point(0, 0),
        }))
    }

    /// Internal idempotent close path. Called by:
    /// - [`Self::close`] (consumes `self`).
    /// - [`Drop::drop`] (panic-unwind / disconnect-before-close).
    /// - [`Self::next_batch`] on EOS / Err (auto-close).
    ///
    /// Invariants:
    /// 1. After return, `self.closed == true`.
    /// 2. The per-tenant budget counter is decremented by
    ///    `self.bytes_reserved` (and `self.bytes_reserved` is reset
    ///    to 0 so a double-call is a no-op on the budget). Per W13β
    ///    fix-up M-2 (release-on-emit), `self.bytes_reserved` is
    ///    typically `0` on entry — non-zero values appear ONLY in
    ///    the panic window between `try_reserve_unscoped` and the
    ///    symmetric `release` inside `next_batch`; this path is the
    ///    backstop for that narrow window.
    /// 3. The snapshot-LSN slot on `self.ctx` is reset to `None` via
    ///    [`ExecutionContext::release_snapshot_lsn`] (idempotent on
    ///    already-released — no double-decrement on the LSN side).
    ///    The release sets the context's `lsn_consumed` latch per
    ///    W13β fix-up M-1 — a subsequent `StreamingCursor::open`
    ///    on the same context (or any clone) rejects with
    ///    `ArcQLError::Internal`.
    /// 4. Forward-method: buffer-pool pins are managed by the
    ///    substrate; v1.0-alpha stub substrates hold no pins. Plan-
    ///    cache read locks are scoped to lookup function bodies (no
    ///    long-lived RwLock reads at v1.0); both are forward-deferred
    ///    cleanup tasks for the M4-08+ production wiring slice.
    ///    // TODO(#290): wire substrate buffer-pool pin release +
    ///    // plan-cache read-lock release at M4-08+ production wiring.
    fn close_internal(&mut self) {
        if self.closed {
            return;
        }
        // Release per-tenant budget bytes accumulated across all
        // next_batch calls. Idempotent under a doubled close: we
        // zero `bytes_reserved` so a subsequent close_internal is a
        // no-op on the budget side.
        if self.bytes_reserved > 0 {
            self.ctx.budget().release(self.tenant, self.bytes_reserved);
            self.bytes_reserved = 0;
        }
        // Release snapshot LSN. ExecutionContext::release_snapshot_lsn
        // is idempotent (W13β M4-81 contract: "release-on-already-
        // released is a no-op").
        self.ctx.release_snapshot_lsn();
        self.closed = true;
        tracing::debug!(
            target: "arcgraph_query::cursor",
            tenant = self.tenant.raw(),
            rows_emitted = self.rows_emitted,
            "StreamingCursor closed (LSN released, budget bytes returned)",
        );
    }
}

impl<'sub, S: ExecutorSubstrate> Drop for StreamingCursor<'sub, S> {
    fn drop(&mut self) {
        // RAII close per `feedback_seqlock_panic_safety_primitive.md`
        // — runs on stack-unwind (panic) AND the
        // disconnect-before-close path. The W12γ MED-3 RegistryGuard
        // pattern is the sister surface in `explain.rs`.
        self.close_internal();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::value::NodeView;
    use crate::executor::{MemoryBudget, StubExecutorSubstrate};
    use crate::semantic::{
        BindingVisitor, CatalogProvider, CrossSubstrateValidator, StubCatalogProvider,
        TypeCheckVisitor,
    };
    use arcgraph_core::{LabelId, NodeId, TenantId};

    fn cat_basic() -> StubCatalogProvider {
        StubCatalogProvider::new()
            .with_labels(["Person"])
            .with_rel_types(["KNOWS"])
            .with_properties(["name", "age"])
    }

    fn substrate_with_n_persons(n: u64) -> StubExecutorSubstrate {
        let mut s = StubExecutorSubstrate::new();
        for i in 1..=n {
            s = s.with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(i), Some(LabelId::new(1)))
                    .with_property("age", Value::Integer(i as i64 * 5)),
            );
        }
        s
    }

    fn lower_to_plan(query: &str, catalog: &StubCatalogProvider) -> LogicalPlan {
        let stmt = crate::parse(query).expect("parse");
        let mut bound = BindingVisitor::bind(&stmt, query, catalog).expect("bind");
        TypeCheckVisitor::check(&mut bound, catalog).expect("type-check");
        CrossSubstrateValidator::validate(&bound, catalog).expect("cross-substrate");
        crate::logical_plan::LogicalPlanLoweringVisitor::lower(&bound).expect("lower")
    }

    // -----------------------------------------------------------------
    // 1. Yield-batch protocol pin — multi-call next_batch returns
    //    Some(rows) until EOS, then Ok(None) terminator.
    // -----------------------------------------------------------------

    #[test]
    fn yield_batch_protocol_emits_rows_then_none_terminator() {
        let s = substrate_with_n_persons(3);
        let cat = cat_basic();
        let plan = lower_to_plan("MATCH (n:Person) RETURN n", &cat);
        let ctx = ExecutionContext::new(cat.tenant(), cat.partition());
        let mut cursor = StreamingCursor::open(&plan, ctx, &s).expect("open");
        // First batch carries the rows.
        let batch1 = cursor.next_batch().expect("first batch");
        assert!(batch1.is_some(), "first call yields Some(rows)");
        assert_eq!(batch1.expect("first").len(), 3, "all 3 rows in one batch");
        // Second call yields None — EOS terminator.
        let batch2 = cursor.next_batch().expect("second batch (EOS)");
        assert!(batch2.is_none(), "EOS terminator");
        // Subsequent calls — cursor is closed; next_batch returns
        // Err(ArcQLError::Internal "next_batch on a closed cursor")
        // per W13β fix-up N-1 (lifecycle-invariant taxonomy).
        let batch3 = cursor.next_batch();
        match batch3 {
            Err(ExecutionError::Plan(ArcQLError::Internal { feature, .. })) => {
                assert_eq!(feature, "StreamingCursor::next_batch");
            }
            other => panic!("post-EOS: expected Internal lifecycle error, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------
    // 2. Cursor lifecycle pin — open → next_batch loop → close().
    //    is_closed() reflects state across the lifecycle.
    // -----------------------------------------------------------------

    #[test]
    fn cursor_lifecycle_open_iterate_close() {
        let s = substrate_with_n_persons(5);
        let cat = cat_basic();
        let plan = lower_to_plan("MATCH (n:Person) RETURN n", &cat);
        let ctx = ExecutionContext::new(cat.tenant(), cat.partition());
        let mut cursor = StreamingCursor::open(&plan, ctx, &s).expect("open");
        assert!(!cursor.is_closed(), "open cursor is not closed");
        let _ = cursor.next_batch().expect("first batch");
        assert!(!cursor.is_closed(), "post-batch, pre-EOS: still open");
        cursor.close().expect("close");
        // The cursor is consumed by close(); we cannot inspect
        // is_closed() here. The Drop test below covers the post-
        // close-via-Drop introspection.
    }

    // -----------------------------------------------------------------
    // 3. Cursor-close-on-disconnect pin — Drop without close() runs
    //    the same cleanup path.
    // -----------------------------------------------------------------

    #[test]
    fn cursor_close_on_disconnect_releases_lsn_via_drop() {
        let s = substrate_with_n_persons(2);
        let cat = cat_basic();
        let plan = lower_to_plan("MATCH (n:Person) RETURN n", &cat);
        // Construct context outside the cursor scope so we can
        // observe the LSN slot post-Drop.
        let ctx = ExecutionContext::new(cat.tenant(), cat.partition());
        // Pre-open: no LSN.
        assert!(ctx.snapshot_lsn().is_none());
        // We cannot directly clone-and-share the context across the
        // cursor + the post-drop assert because StreamingCursor::open
        // takes ctx by value. Use a clone (the snapshot-LSN slot is
        // Arc<Mutex<Option<Lsn>>>-shared).
        let ctx_observer = ctx.clone();
        {
            let mut cursor = StreamingCursor::open(&plan, ctx, &s).expect("open");
            let _ = cursor.next_batch().expect("first batch");
            assert!(
                ctx_observer.snapshot_lsn().is_some(),
                "during streaming: LSN captured"
            );
            // Drop here (cursor goes out of scope) — close_internal runs.
        }
        // Post-drop: the LSN slot is reset to None via the cursor's
        // Drop → close_internal → ctx.release_snapshot_lsn path.
        assert!(
            ctx_observer.snapshot_lsn().is_none(),
            "Drop-on-disconnect releases LSN via close_internal"
        );
    }

    // -----------------------------------------------------------------
    // 4. Cursor-close-on-cancel pin — cancel mid-stream surfaces
    //    Cancelled + auto-closes.
    // -----------------------------------------------------------------

    #[test]
    fn cursor_close_on_cancel_releases_resources() {
        let s = substrate_with_n_persons(7);
        let cat = cat_basic();
        let plan = lower_to_plan("MATCH (n:Person) RETURN n", &cat);
        let ctx = ExecutionContext::new(cat.tenant(), cat.partition());
        let token = ctx.cancellation().clone();
        let ctx_observer = ctx.clone();
        let mut cursor = StreamingCursor::open(&plan, ctx, &s).expect("open");
        // Trip the cancellation BEFORE the first batch.
        token.cancel();
        // First next_batch call observes the trip + auto-closes.
        let result = cursor.next_batch();
        assert!(
            matches!(result, Err(ExecutionError::Cancelled)),
            "post-cancel next_batch surfaces Cancelled, got {result:?}"
        );
        assert!(cursor.is_closed(), "auto-closed on Cancelled");
        // Resources released: LSN.
        assert!(
            ctx_observer.snapshot_lsn().is_none(),
            "post-cancel: LSN released"
        );
        // Subsequent next_batch returns "closed" error.
        let result2 = cursor.next_batch();
        assert!(result2.is_err());
    }

    // -----------------------------------------------------------------
    // 5. Cursor-close-on-drop pin — explicit close() followed by
    //    Drop is a no-op on the budget counter (idempotent).
    // -----------------------------------------------------------------

    #[test]
    fn cursor_close_then_drop_does_not_double_release_budget() {
        // W13β fix-up M-2 (release-on-emit): each next_batch charges
        // the budget, then releases before returning rows. The
        // bytes_reserved field is `0` between calls; the budget
        // counter never accumulates across the cursor's lifetime.
        // The "no double-release" load-bearing claim still holds:
        // close-after-emit + Drop are both no-ops on the budget.
        let s = substrate_with_n_persons(5);
        let cat = cat_basic();
        let plan = lower_to_plan("MATCH (n:Person) RETURN n", &cat);
        let budget = MemoryBudget::new();
        let ctx = ExecutionContext::new(cat.tenant(), cat.partition()).with_budget(budget.clone());
        let mut cursor = StreamingCursor::open(&plan, ctx, &s).expect("open");
        let rows = cursor
            .next_batch()
            .expect("first batch")
            .expect("non-empty");
        assert!(!rows.is_empty(), "non-empty batch emitted");
        // Per release-on-emit: the cursor holds NO bytes between
        // batches; the budget counter mirrors actual in-flight
        // pressure (zero here).
        assert_eq!(
            cursor.bytes_reserved(),
            0,
            "release-on-emit: cursor holds no bytes between batches"
        );
        assert_eq!(
            budget.current_bytes(cat.tenant()),
            0,
            "release-on-emit: budget counter is zero between batches"
        );
        cursor.close().expect("close is idempotent on the budget");
        // After close (which consumed the cursor), the budget counter
        // is still 0 — no leak, no double-release. Drop runs at the
        // end of close() (consumed self) but close_internal is
        // idempotent.
        assert_eq!(
            budget.current_bytes(cat.tenant()),
            0,
            "post-close: budget counter at 0"
        );
    }

    // -----------------------------------------------------------------
    // 6. Cursor-close-on-panic pin — panic-unwind in caller drops
    //    the cursor, releasing resources via Drop. Mirrors W12γ
    //    MED-3 RegistryGuard panic-safety pin.
    // -----------------------------------------------------------------

    #[test]
    fn cursor_close_on_panic_unwind_releases_lsn_via_drop() {
        use std::panic::AssertUnwindSafe;
        let s = substrate_with_n_persons(3);
        let cat = cat_basic();
        let plan = lower_to_plan("MATCH (n:Person) RETURN n", &cat);
        let ctx = ExecutionContext::new(cat.tenant(), cat.partition());
        let ctx_observer = ctx.clone();
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            let mut cursor = StreamingCursor::open(&plan, ctx, &s).expect("open");
            let _ = cursor.next_batch().expect("first batch");
            // Caller-side panic: simulates a substrate-level fault
            // (e.g., out-of-disk) propagating up past the cursor.
            // Cursor's Drop runs during stack-unwind.
            panic!("synthetic panic for cursor Drop pin");
        }));
        assert!(result.is_err(), "panic propagated");
        // Cursor's Drop ran during unwind — LSN released.
        assert!(
            ctx_observer.snapshot_lsn().is_none(),
            "panic-unwind dropped cursor; LSN released"
        );
    }

    // -----------------------------------------------------------------
    // 7. Cursor-error-propagates-to-close pin — an error from
    //    op.next_batch auto-closes the cursor + propagates the
    //    error to the caller.
    // -----------------------------------------------------------------

    #[test]
    fn cursor_error_propagates_to_close_via_auto_close() {
        // Use a TINY budget cap to force a ResourceExhausted on the
        // first batch. The cursor auto-closes on Err.
        let s = substrate_with_n_persons(20);
        let cat = cat_basic();
        let plan = lower_to_plan("MATCH (n:Person) RETURN n", &cat);
        let budget = MemoryBudget::with_per_tenant_cap(cat.tenant(), 64);
        let ctx = ExecutionContext::new(cat.tenant(), cat.partition()).with_budget(budget.clone());
        let ctx_observer = ctx.clone();
        let mut cursor = StreamingCursor::open(&plan, ctx, &s).expect("open");
        let result = cursor.next_batch();
        // The 20 rows × ~bytes-per-row exceeds the 64-byte cap on
        // the first batch — try_reserve_unscoped surfaces
        // ResourceExhausted.
        match result {
            Err(ExecutionError::Plan(ArcQLError::ResourceExhausted { .. })) => {
                // Auto-closed.
                assert!(cursor.is_closed());
            }
            other => panic!("expected ResourceExhausted, got {other:?}"),
        }
        // Resources released: LSN.
        assert!(
            ctx_observer.snapshot_lsn().is_none(),
            "post-error: LSN released"
        );
        // Budget counter at 0 (the rejecting batch's bytes weren't
        // reserved per W12α convention).
        assert_eq!(budget.current_bytes(cat.tenant()), 0);
    }

    // -----------------------------------------------------------------
    // 8. replan_in_place forward-method pin
    // -----------------------------------------------------------------

    #[test]
    fn replan_in_place_returns_not_implemented_at_v1_0_alpha() {
        // The v1.0-alpha cursor does NOT support mid-stream replan.
        // The forward-method API surface returns
        // ArcQLError::NotImplemented per the inflection-point doc.
        let s = substrate_with_n_persons(2);
        let cat = cat_basic();
        let plan = lower_to_plan("MATCH (n:Person) RETURN n", &cat);
        let ctx = ExecutionContext::new(cat.tenant(), cat.partition());
        let mut cursor = StreamingCursor::open(&plan, ctx, &s).expect("open");
        let result = cursor.replan_in_place();
        match result {
            Err(ExecutionError::Plan(ArcQLError::NotImplemented { target_version, .. })) => {
                assert_eq!(target_version, "v1.1");
            }
            other => panic!("expected NotImplemented v1.1, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------
    // 9. W13β fix-up M-1 — close-then-reopen REJECTS at v1.0
    // -----------------------------------------------------------------

    #[test]
    fn open_rejects_close_then_reopen_via_consumed_context_latch() {
        // ADR-038 amendment-03 §TIER-1 GAP E rule 5: "All operators in
        // a single ExecutionContext share the same snapshot LSN; replan
        // does NOT re-acquire." A close-then-reopen sequence on the
        // same context (or a clone) would silently re-acquire a fresh
        // LSN at M4-08+ production wiring. v1.0-alpha LSN sentinel is
        // Lsn::MAX so the divergence is observationally null TODAY,
        // but the latch makes the rejection visible NOW so M4-08+
        // doesn't introduce a silent correctness regression.
        let s = substrate_with_n_persons(3);
        let cat = cat_basic();
        let plan = lower_to_plan("MATCH (n:Person) RETURN n", &cat);
        let ctx = ExecutionContext::new(cat.tenant(), cat.partition());
        // Clone shares the consumption latch (Arc<AtomicBool>).
        let ctx_clone = ctx.clone();
        // First cursor — runs to EOS, releases LSN, sets latch.
        let mut cursor1 = StreamingCursor::open(&plan, ctx, &s).expect("open #1");
        while let Some(_rows) = cursor1.next_batch().expect("next_batch") {}
        // Latch should be set on the clone (shared via Arc).
        assert!(
            ctx_clone.lsn_consumed(),
            "after cursor1 close: clone observes lsn_consumed = true"
        );
        // Reopen on the clone REJECTS with ArcQLError::Internal —
        // close-then-reopen is a documented v1.0 invariant violation.
        // (StreamingCursor is not Debug — match on the err half only.)
        match StreamingCursor::open(&plan, ctx_clone, &s) {
            Ok(_) => panic!("expected ArcQLError::Internal on consumed ctx, got Ok"),
            Err(ExecutionError::Plan(ArcQLError::Internal {
                feature, reason, ..
            })) => {
                assert_eq!(feature, "StreamingCursor::open");
                assert!(
                    reason.contains("rule 5"),
                    "rejection reason cites rule 5; got: {reason}"
                );
            }
            Err(other) => panic!("expected ArcQLError::Internal, got {other:?}"),
        }
    }

    #[test]
    fn open_succeeds_on_fresh_context_after_unrelated_cursor_close() {
        // The latch is per-context — a fresh context (different Arc-
        // backed slot) is NOT consumed by a sibling context's close.
        // This pin guards against a misimplementation that would
        // accidentally make the latch process-global.
        let s = substrate_with_n_persons(2);
        let cat = cat_basic();
        let plan = lower_to_plan("MATCH (n:Person) RETURN n", &cat);
        // Drive cursor #1 to close on its own context.
        let ctx_a = ExecutionContext::new(cat.tenant(), cat.partition());
        let mut cursor_a = StreamingCursor::open(&plan, ctx_a, &s).expect("open #A");
        while let Some(_rows) = cursor_a.next_batch().expect("next_batch") {}
        // Different context — latch defaults to false.
        let ctx_b = ExecutionContext::new(cat.tenant(), cat.partition());
        assert!(!ctx_b.lsn_consumed(), "fresh context: latch is false");
        let _cursor_b = StreamingCursor::open(&plan, ctx_b, &s).expect("open #B succeeds");
    }

    // -----------------------------------------------------------------
    // 10. W13β fix-up M-2 — release-on-emit budget semantics
    // -----------------------------------------------------------------

    #[test]
    fn next_batch_release_on_emit_keeps_budget_counter_at_zero_between_batches() {
        // Per W13β fix-up M-2 (PR #287 review M-2): the cursor's
        // bytes_reserved should release as rows are emitted to the
        // consumer (not held until cursor close). Drive a multi-batch
        // stream + assert the budget counter sits at 0 BETWEEN every
        // batch — proving "in-flight bytes" semantics, not the
        // pre-fix-up "cumulative emit total" semantics.
        use crate::executor::batch::BATCH_ROWS;
        let n = (BATCH_ROWS * 2 + 7) as u64; // 3 batches.
        let s = substrate_with_n_persons(n);
        let cat = cat_basic();
        let plan = lower_to_plan("MATCH (n:Person) RETURN n", &cat);
        let budget = MemoryBudget::new();
        let ctx = ExecutionContext::new(cat.tenant(), cat.partition()).with_budget(budget.clone());
        let mut cursor = StreamingCursor::open(&plan, ctx, &s).expect("open");
        let mut batches_seen = 0usize;
        while let Some(rows) = cursor.next_batch().expect("next_batch") {
            assert!(!rows.is_empty(), "non-empty batch");
            batches_seen += 1;
            // Per release-on-emit: the budget counter is released
            // BEFORE next_batch returns. The reading here observes 0.
            assert_eq!(
                budget.current_bytes(cat.tenant()),
                0,
                "release-on-emit: budget counter back to 0 after batch {batches_seen} emit"
            );
            assert_eq!(
                cursor.bytes_reserved(),
                0,
                "release-on-emit: cursor holds no bytes between batches"
            );
        }
        assert!(batches_seen >= 2, "drove >=2 batches; got {batches_seen}");
        // Peak high-water mark IS preserved (release does not
        // decrement peak — see budget.rs:unit_5_release_decrements_current_not_peak).
        assert!(
            budget.peak_bytes(cat.tenant()) > 0,
            "peak high-water mark reflects the largest in-flight batch"
        );
    }

    #[test]
    fn concurrent_query_observes_zero_pinned_bytes_between_cursor_batches() {
        // Per W13β fix-up M-2 second-order effect: a back-pressure-
        // aware sibling query on the same tenant should NOT observe
        // the cursor's lifetime-cumulative emit total in
        // current_bytes. Pin: between two next_batch calls, a sibling
        // observer (simulating a concurrent query against the same
        // per-tenant counter) sees current_bytes == 0.
        use crate::executor::batch::BATCH_ROWS;
        let n = (BATCH_ROWS * 2) as u64; // 2 batches minimum.
        let s = substrate_with_n_persons(n);
        let cat = cat_basic();
        let plan = lower_to_plan("MATCH (n:Person) RETURN n", &cat);
        let budget = MemoryBudget::new();
        // Sibling observer — separate Arc clone of the budget; same
        // per-tenant counter shared.
        let sibling = budget.clone();
        let ctx = ExecutionContext::new(cat.tenant(), cat.partition()).with_budget(budget);
        let mut cursor = StreamingCursor::open(&plan, ctx, &s).expect("open");
        // First batch produced + emitted; sibling sees 0.
        let _b1 = cursor
            .next_batch()
            .expect("first batch")
            .expect("non-empty");
        assert_eq!(
            sibling.current_bytes(cat.tenant()),
            0,
            "sibling sees 0 in-flight bytes after batch 1 emit"
        );
        // Second batch produced + emitted; sibling still sees 0.
        let _b2 = cursor
            .next_batch()
            .expect("second batch")
            .expect("non-empty");
        assert_eq!(
            sibling.current_bytes(cat.tenant()),
            0,
            "sibling sees 0 in-flight bytes after batch 2 emit"
        );
    }
}
