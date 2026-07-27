//! Per-query execution context for the M4-61 vectorized executor.
//!
//! [`ExecutionContext`] carries the cross-cutting state every operator
//! needs: tenant + partition stamp, query identity, lazy snapshot LSN,
//! cancellation token, and a [`tracing::Span`] tagged with the query
//! identity. Per ADR-038 amendment-02 §M4.f + amendment-03 §TIER-1
//! GAP E, the snapshot LSN is acquired LAZILY — pre-first-batch, not
//! at construction time — so EXPLAIN's no-LSN discipline (D-18 rule 1)
//! and PROFILE / direct execute's lazy-LSN discipline (also D-18 rule 1
//! — execute-time, pre-first-batch acquire) both flow from a single
//! context type.
//!
//! # Cancellation surface
//!
//! [`CancellationToken`] is a thin `Arc<AtomicBool>` shared between
//! the query-issuer thread and the operator pipeline. Operators check
//! the token at BATCH boundaries (not row boundaries) per
//! amendment-02 §M4.f — the 2048-row batch is well inside the
//! cancel-latency budget per ADR-036 §D-24. The forward-pin lets
//! M4-92 (Bolt-side client cancel + M4-83 multi-statement boundary)
//! plug in a richer cancellation source without re-shaping operator
//! code.
//!
//! # Tracing
//!
//! Every [`ExecutionContext`] creates a `tracing::Span` at info level
//! tagged with `query_id` (UUIDv7), `tenant`, and `partition`. The
//! span is the stable correlation handle for slow-query log emission
//! (forward to M4-71) + EXPLAIN/PROFILE telemetry. The span is
//! `Clone`-friendly via [`tracing::Span::current`]; operators enter
//! the span at `next_batch` to thread per-operator events.
//!
//! # ADR provenance
//! - **ADR-038 amendment-02 §M4.f** — primary M4-61 cite.
//! - **ADR-038 amendment-03 §TIER-1 GAP E** — snapshot-LSN execute-
//!   time-binding contract.
//! - **ADR-038 §2 D-18** — snapshot-LSN binding rules
//!   (rule 1: acquired at execute-time, pre-first-batch, incl. EXPLAIN
//!   no-acquire exception; rule 2: same snapshot LSN across all statements
//!   in a multi-statement query, M4-83).
//! - **ADR-036 §D-24** — cancel-latency budget; 2048-row batch fits.
//! - **ADR-024 amendment-02** — partition_id is always
//!   [`PartitionId::ZERO`] at v1.0.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use arcgraph_core::{Lsn, PartitionId, TenantId};
use parking_lot::Mutex;

use crate::error::Span;
use crate::executor::budget::MemoryBudget;
use crate::executor::error::ExecutionError;
use crate::executor::eval::Parameters;
use crate::executor::value::Value;
use crate::observer::RowCountObserver;
use crate::semantic::bound_ast::BindingId;
use crate::semantic::error::ArcQLError;

/// One `CALL { … }` correlation frame (ADR-192 #623): the imported
/// bindings + their values for the current driving row at one CALL
/// nesting level. Pushed by [`crate::executor::ops::CallOp`] before it
/// drives the subquery body for a driving row, popped when that row's
/// body is exhausted. Read by
/// [`crate::executor::ops::CorrelationSeedOp`].
pub(crate) type CorrelationFrame = Vec<(BindingId, Value)>;

/// UUIDv7 query identifier.
///
/// Wraps [`uuid::Uuid`]; the v7 timestamp prefix gives a natural
/// total ordering by issue time. Used for diagnostic correlation
/// across `tracing` spans, slow-query log, EXPLAIN/PROFILE artifacts,
/// and the future M4-71 row-count observer feedback loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QueryId(pub uuid::Uuid);

impl QueryId {
    /// Mint a fresh UUIDv7 for the current wall-clock instant.
    ///
    /// Per `draft-ietf-uuidrev-rfc4122bis` §6.6 the v7 layout is
    /// `<timestamp_ms:48><ver:4><rand_a:12><var:2><rand_b:62>` — two
    /// IDs minted in rapid succession sort by timestamp; ties break
    /// on the random tail. The `uuid` crate's `Uuid::now_v7()` is
    /// the canonical generator.
    #[must_use]
    pub fn new() -> Self {
        Self(uuid::Uuid::now_v7())
    }

    /// Construct from a raw UUIDv7. Used by tests that need a
    /// deterministic identifier; production callers should use
    /// [`Self::new`].
    #[must_use]
    pub fn from_uuid(u: uuid::Uuid) -> Self {
        Self(u)
    }

    /// Read the underlying UUID.
    #[must_use]
    pub fn as_uuid(self) -> uuid::Uuid {
        self.0
    }
}

impl Default for QueryId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for QueryId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // UUID's hyphenated lowercase rendering — matches the canonical
        // RFC 9562 form. Stable across `tracing`-emitted JSON logs.
        write!(f, "{}", self.0)
    }
}

/// Cancellation handle shared between the query issuer and the
/// operator pipeline.
///
/// Cheap to clone — internally an `Arc<AtomicBool>`. Operators
/// check the token at batch boundaries via
/// [`CancellationToken::is_cancelled`] and surface
/// [`CancellationError`] when tripped; the pipeline translates this
/// into [`crate::executor::ExecutionError::Cancelled`].
///
/// # Forward note (M4-92)
///
/// At v1.0-alpha the only trip source is [`CancellationToken::cancel`]
/// (caller-side). M4-92 plumbing extends this to:
/// - Bolt-side client cancel (`PROTOCOL` reset frame).
/// - M4-83 multi-statement boundary (one statement may cancel the
///   next).
/// - Per-tenant query-time SLO timeout.
///
/// The forward-pin keeps the operator-side surface stable: future
/// trip sources flip the same `AtomicBool`.
#[derive(Debug, Clone, Default)]
pub struct CancellationToken {
    flag: Arc<AtomicBool>,
}

impl CancellationToken {
    /// Construct a fresh, un-tripped token.
    #[must_use]
    pub fn new() -> Self {
        Self {
            flag: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Trip the token. Idempotent — repeated calls have no effect
    /// beyond the first.
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::Release);
    }

    /// Return `true` if the token has been tripped.
    #[inline]
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }

    /// Convenience: return `Err(CancellationError)` if tripped, else
    /// `Ok(())`. The standard idiom at every operator's batch
    /// boundary.
    #[inline]
    pub fn check(&self) -> Result<(), CancellationError> {
        if self.is_cancelled() {
            Err(CancellationError)
        } else {
            Ok(())
        }
    }
}

/// Marker error returned by [`CancellationToken::check`] when tripped.
///
/// Translated to [`crate::executor::ExecutionError::Cancelled`] at
/// the operator boundary; carrying a separate type lets operator code
/// use `?` without losing the cancellation-vs-substrate-fault
/// distinction at the call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("query cancelled at batch boundary")]
pub struct CancellationError;

/// Cloneable, type-erased access to the explicit transaction installed on an
/// [`ExecutionContext`].
///
/// Production streaming cursors use this narrow handle instead of cloning the
/// whole context (which also contains thread-affine MERGE guards). Each access
/// holds the same short, uncontended mutex as [`ExecutionContext::with_held_txn_mut`];
/// the transaction remains owned by the context and is still reclaimed by the
/// Bolt COMMIT/ROLLBACK path after cursor execution finishes.
#[derive(Clone)]
pub struct HeldTxnAccess {
    inner: Arc<Mutex<Option<Box<dyn crate::executor::substrate::HeldTxnHandle>>>>,
}

impl std::fmt::Debug for HeldTxnAccess {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HeldTxnAccess").finish_non_exhaustive()
    }
}

impl HeldTxnAccess {
    /// Run `f` against the installed handle, or return `None` if it was
    /// reclaimed before the cursor finished.
    pub fn with_mut<R>(
        &self,
        f: impl FnOnce(&mut dyn crate::executor::substrate::HeldTxnHandle) -> R,
    ) -> Option<R> {
        let mut guard = self.inner.lock();
        guard.as_mut().map(|held| f(held.as_mut()))
    }
}

/// Per-query execution context.
///
/// Constructed once per [`crate::execute`] call (or its
/// [`crate::execute_with_context`] sibling for tests / future
/// shared-context callers); threaded into every operator's
/// `next_batch`. The context owns:
///
/// - **Identity**: tenant, partition, query_id (via the field
///   accessors).
/// - **Snapshot LSN**: lazily acquired per ADR-038 §2 D-18 rule 1
///   (execute-time, pre-first-batch) via [`Self::ensure_snapshot_lsn`].
/// - **Cancellation**: a [`CancellationToken`] checked per batch.
/// - **Tracing**: a `tracing::Span` tagged with the identity fields.
///
/// # Snapshot LSN — capture timing
///
/// Per ADR-038 §2 D-18 rule 1 + amendment-03 §TIER-1 GAP E:
/// 1. EXPLAIN does NOT acquire (rule 1's EXPLAIN exception — pinned by
///    the `explain_does_not_acquire_snapshot_lsn` integration test in
///    `tests/m4_91_explain_integration.rs`).
/// 2. The execute path acquires LAZILY, pre-first-batch, and HOLDS
///    until the query ends (rule 1's main clause). Multiple operators
///    within the same query observe the same LSN value (the lazy-init
///    is process-monotonic per the AtomicU64 protocol below).
///
/// The lazy-init protocol uses a `Mutex<Option<Lsn>>` (NOT a
/// double-checked AtomicU64) for two reasons:
/// - The LSN itself is `Lsn(u64)`, not `Option<Lsn>`; encoding the
///   "not yet captured" state inside the LSN value would steal
///   `Lsn::MAX` (already used as the read-latest sentinel) or a
///   sentinel value like `Lsn::ZERO` (already the "never seen"
///   floor).
/// - Capture is once-per-query, not per-batch — the lock contention
///   is bounded to the very first `next_batch` call. After capture,
///   reads are uncontended.
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    tenant: TenantId,
    partition: PartitionId,
    query_id: QueryId,
    /// Lazily-acquired snapshot LSN. `None` until the first call to
    /// [`Self::ensure_snapshot_lsn`]; `Some(Lsn)` thereafter, held for
    /// the rest of the query lifetime.
    snapshot_lsn: Arc<Mutex<Option<Lsn>>>,
    /// W13β fix-up M-1 — single-shot consumption latch.
    ///
    /// Set to `true` the first time [`Self::release_snapshot_lsn`] runs
    /// while the LSN slot held a `Some(_)` value (i.e. an actual release,
    /// not a no-op release on an unacquired context). Once set, the
    /// latch is sticky: it never flips back to `false`. Cloning the
    /// context propagates the latch (`Arc<AtomicBool>`-shared), so a
    /// `cursor.close() → ctx.clone() → StreamingCursor::open(...)`
    /// sequence rejects rather than silently re-acquiring a fresh LSN
    /// — which would violate ADR-038 amendment-03 §TIER-1 GAP E rule 5
    /// ("All operators in a single `ExecutionContext` share the same
    /// snapshot LSN; replan (M4-72) does NOT re-acquire"; the 1:1
    /// sister cite in the parent ADR is §2 D-18 rule 5 with the
    /// briefer canonical wording "Replan (M4-72) does NOT re-acquire")
    /// at production-
    /// LSN-binding time (M4-08+,
    /// when the captured value stops being the `Lsn::MAX` v1.0-alpha
    /// sentinel and starts reflecting real WAL state).
    ///
    /// The latch lights ONLY on the explicit-release path; an
    /// `ExecutionContext` that is dropped without anyone calling
    /// `release_snapshot_lsn` (e.g. EXPLAIN — never acquires) leaves
    /// the latch `false` so the context behaves as never-used.
    lsn_consumed: Arc<AtomicBool>,
    cancellation: CancellationToken,
    span: tracing::Span,
    /// Wall-clock-monotonic batch counter. Bumped by
    /// [`Self::next_batch_seq`] which operators call AT THE TOP of
    /// each `next_batch` for slow-query log emission.
    batch_seq: Arc<AtomicU64>,
    /// W12α / M4-64a per-tenant memory budget per amendment-03
    /// §Structural-1. Default is unbounded (no cap configured); M5-12
    /// rate-limit config will override per-tenant caps via
    /// [`MemoryBudget::set_per_tenant_cap`] at server-startup time.
    /// Operators with spillover queues consume the budget; tests
    /// exercise the byte-cap path via [`Self::with_budget`].
    budget: MemoryBudget,
    /// M4-71 row-count observer. `None` for executions that don't need
    /// observability (most non-PROFILE non-replan paths). When `Some`,
    /// the dispatcher in [`crate::executor::ops::PhysicalOperator::next_batch`]
    /// records per-batch metrics via
    /// [`crate::observer::dispatcher::record_dispatch`].
    ///
    /// `Arc`-shared so post-execute readers (PROFILE renderer / replan
    /// controller) observe the SAME accumulated state the dispatcher
    /// wrote.
    observer: Option<Arc<RowCountObserver>>,
    /// **ADR-192 (#623) — `CALL { … }` correlation-frame stack.** The
    /// per-driving-row imported-binding values for the active `CALL { … }`
    /// subqueries, one frame per nesting level (innermost on top).
    /// [`crate::executor::ops::CallOp`] pushes a frame before driving the
    /// subquery body for a driving row and pops it when that row's body is
    /// exhausted; [`crate::executor::ops::CorrelationSeedOp`] reads the
    /// imported values via [`Self::correlation_value`] (nearest-frame-wins
    /// lookup, so nested correlated subqueries see both their own and the
    /// enclosing imports). `Arc<Mutex<…>>`-shared for interior mutability
    /// behind the `&ExecutionContext` operators hold — uncontended (a
    /// single query executes on one thread; the lock just enables
    /// push/pop/read through a shared borrow, mirroring `snapshot_lsn`).
    correlation: Arc<Mutex<Vec<CorrelationFrame>>>,
    /// **ADR-197 — Bolt explicit-transaction held transaction.**
    ///
    /// `None` (the default) → AUTO-COMMIT mode: substrate write ops
    /// open + commit their own per-call transaction (the v1.0-α
    /// one-call-one-tx path, byte-for-byte preserved). `Some(tx)` →
    /// EXPLICIT mode: substrate write ops STAGE into this one held
    /// transaction (no per-op commit); the Bolt handler commits it at
    /// COMMIT and aborts it at ROLLBACK / RESET / drop.
    ///
    /// `Arc<Mutex<Option<…>>>`-shared for interior mutability behind
    /// the `&ExecutionContext` operators hold (same idiom as
    /// `snapshot_lsn` / `correlation`). The Bolt handler installs the
    /// `OwnedTxn` via [`Self::with_held_txn`] before `execute`, and
    /// reclaims it via [`Self::take_held_txn`] after `execute` returns
    /// so it can `commit()` / `abort()` it at the COMMIT / ROLLBACK
    /// message. A single connection executes one statement at a time,
    /// so the lock is uncontended.
    ///
    /// Carried opaquely as `Box<dyn HeldTxnHandle>` so the concrete
    /// `arcgraph_storage::transaction::OwnedTxn` is NOT a production
    /// dependency of `arcgraph-query` (bounded-context policy published-trait
    /// boundary; `arcgraph-storage` is a dev-dep-only edge here).
    held_txn: Arc<Mutex<Option<Box<dyn crate::executor::substrate::HeldTxnHandle>>>>,
    /// **NN-4 (#1384) re-spin — MERGE get-or-create serialization guards
    /// that must SPAN the statement commit.**
    ///
    /// The query DRIVER ([`crate::executor::ops::acquire_merge_guards`],
    /// called from [`crate::materialize::materialize`] /
    /// [`crate::executor::execute_with_context`]) acquires the
    /// per-`(tenant, key)` serialization guard(s) BEFORE the statement's
    /// read snapshot is pinned (before `begin_statement`) and STASHES them
    /// here so the guard outlives the create's COMMIT. Under the D-2
    /// statement-scoped autocommit wrap a MERGE create only STAGES inside
    /// `next_batch` and COMMITS at `commit_statement`; the match probe reads
    /// at the snapshot pinned by `begin_statement`. If the guard were taken
    /// only INSIDE `MergeOp::next_batch` (after `begin_statement`, the
    /// pre-respin behavior) the loser would pin its snapshot BEFORE the
    /// winner committed, so even after blocking on the guard its re-probe
    /// would read the stale pre-commit snapshot and double-create. Acquiring
    /// before the snapshot pin + holding until [`Self::take_merge_guards`]
    /// runs — AFTER `commit_statement`/`rollback_statement` — closes that
    /// window on the production auto-commit path.
    ///
    /// `Arc<Mutex<Vec<…>>>`-shared for interior mutability behind the
    /// `&ExecutionContext` operators hold (same idiom as `held_txn` /
    /// `correlation`). A `Vec` because a path-shape MERGE acquires guards
    /// for BOTH endpoints (Fix 3); node-shape acquires one. The whole
    /// statement executes on ONE thread (the guard is `!Send`, held +
    /// dropped on the acquiring thread, never crossing a thread boundary
    /// or straddling an `.await`), so this slot is uncontended — the lock
    /// just enables stash/drain through a shared borrow.
    ///
    /// # Lock order (re-verified after the respin — no deadlock)
    ///
    /// The merge-key lock stays strictly OUTER of the MVCC commit gate:
    /// no commit path (`crud::commit` → `commit_gate`) ever acquires a
    /// merge-key lock, so the order is always `merge_key_lock →
    /// commit_gate`, never the reverse — even though the guard is now
    /// held ACROSS `commit_statement` (the commit takes `commit_gate`
    /// while the outer merge-key lock is held; the reverse never happens).
    /// Two MERGEs on the SAME endpoint-key set acquire per-key guards in
    /// CANONICAL TOTAL ORDER (sorted keys — Fix 3) so two path-MERGEs
    /// naming the same two keys in opposite pattern order cannot invert
    /// against each other either.
    merge_guards: Arc<Mutex<Vec<Box<dyn crate::executor::substrate::MergeGuard>>>>,
    /// **#797 / ADR-147 Phase 2 — per-query parameter bag (`$name`).**
    ///
    /// The Bolt RUN / MCP `graph.raw_query` entry points convert the
    /// wire parameter map into [`Parameters`] (string-keyed
    /// [`Value`]s) and install it here via [`Self::with_parameters`].
    /// [`crate::materialize::materialize`] /
    /// [`crate::executor::execute_with_context`] read it back via
    /// [`Self::parameters`] and pass it to
    /// [`crate::executor::Pipeline::build_with_parameters`], which bakes
    /// it into the per-operator [`crate::executor::eval::evaluate`]
    /// calls so `BoundExpression::Parameter { name }` resolves to its
    /// bound literal at runtime.
    ///
    /// Default empty (`Parameters::new()`) — every existing
    /// non-parameterized execute path constructs an empty bag, so
    /// behavior is byte-for-byte preserved for literal-only queries.
    /// Binding stays a RUNTIME substitution (NOT a plan-time rewrite)
    /// so the M4-53 plan cache keeps its param-agnostic AST key
    /// (`PlanCacheKey::from_ast` normalizes `$name`) AND its cached
    /// costed plan stays value-independent — distinct parameter values
    /// reuse the same cached plan.
    parameters: Parameters,
}

impl ExecutionContext {
    /// Construct a fresh execution context for `(tenant, partition)`.
    ///
    /// Mints a fresh [`QueryId`] (UUIDv7), creates a fresh
    /// [`CancellationToken`], and opens a `tracing::Span` tagged
    /// with the identity fields. Snapshot LSN is NOT acquired here —
    /// it lazies-out via [`Self::ensure_snapshot_lsn`].
    #[must_use]
    pub fn new(tenant: TenantId, partition: PartitionId) -> Self {
        let query_id = QueryId::new();
        let span = tracing::info_span!(
            "arcgraph_query::executor",
            query_id = %query_id,
            tenant = tenant.raw(),
            partition = partition.raw(),
        );
        Self {
            tenant,
            partition,
            query_id,
            snapshot_lsn: Arc::new(Mutex::new(None)),
            lsn_consumed: Arc::new(AtomicBool::new(false)),
            cancellation: CancellationToken::new(),
            span,
            batch_seq: Arc::new(AtomicU64::new(0)),
            budget: MemoryBudget::new(),
            observer: None,
            correlation: Arc::new(Mutex::new(Vec::new())),
            held_txn: Arc::new(Mutex::new(None)),
            // NN-4 (#1384) re-spin — the stashed `Box<dyn MergeGuard>` is
            // deliberately `!Send` (thread-affine `parking_lot::ArcMutexGuard`,
            // per the `MergeGuard` trait doc), which makes this `Arc<Mutex<…>>`
            // non-Send/Sync. That is BY DESIGN: the whole statement executes
            // synchronously on ONE thread (acquire → stash → commit → drain,
            // never crossing a thread boundary), and each thread constructs
            // its OWN `ExecutionContext`, so the guard never travels between
            // threads. `Arc` (not `Rc`) matches the `held_txn` / `correlation`
            // field idiom (shared interior mutability across the context's
            // operator-tree clones). The `arc_with_non_send_sync` lint is
            // silenced with this documented rationale.
            #[allow(clippy::arc_with_non_send_sync)]
            merge_guards: Arc::new(Mutex::new(Vec::new())),
            parameters: Parameters::new(),
        }
    }

    /// Construct a context with a caller-supplied query identity.
    /// Useful for tests that need deterministic span output.
    #[must_use]
    pub fn with_query_id(tenant: TenantId, partition: PartitionId, query_id: QueryId) -> Self {
        let mut ctx = Self::new(tenant, partition);
        ctx.query_id = query_id;
        ctx.span = tracing::info_span!(
            "arcgraph_query::executor",
            query_id = %query_id,
            tenant = tenant.raw(),
            partition = partition.raw(),
        );
        ctx
    }

    /// Override the cancellation token. The default is a fresh
    /// untripped token; tests / M4-92 wiring use this to share a
    /// single token across many contexts (e.g., one per multi-
    /// statement query in M4-83 forward).
    #[must_use]
    pub fn with_cancellation(mut self, token: CancellationToken) -> Self {
        self.cancellation = token;
        self
    }

    /// Override the per-tenant memory budget. The default is unbounded
    /// (no per-tenant cap configured); tests + M5-12 rate-limit
    /// wiring use this to inject a cap-configured budget. The budget
    /// is `Arc`-backed so cloning shares state across the operator
    /// pipeline.
    #[must_use]
    pub fn with_budget(mut self, budget: MemoryBudget) -> Self {
        self.budget = budget;
        self
    }

    /// Borrow the per-tenant memory budget. Operators with spillover
    /// queues consume this to track per-row byte allocations.
    #[inline]
    #[must_use]
    pub fn budget(&self) -> &MemoryBudget {
        &self.budget
    }

    /// **#797** — install the per-query parameter bag (`$name` →
    /// [`Value`]). The Bolt RUN / MCP `graph.raw_query` entry points
    /// build this from the wire parameter map; the executor's
    /// [`crate::executor::eval::evaluate`] resolves
    /// `BoundExpression::Parameter` against it at runtime. The default
    /// (no call) is an empty bag — preserving literal-only behavior.
    #[must_use]
    pub fn with_parameters(mut self, parameters: Parameters) -> Self {
        self.parameters = parameters;
        self
    }

    /// Borrow the per-query parameter bag. Read by
    /// [`crate::materialize::materialize`] /
    /// [`crate::executor::execute_with_context`] at
    /// [`crate::executor::Pipeline::build_with_parameters`] time.
    #[inline]
    #[must_use]
    pub fn parameters(&self) -> &Parameters {
        &self.parameters
    }

    /// Attach an [`RowCountObserver`] for per-batch metrics + 10×
    /// threshold detection per ADR-038 amendment-02 §M4.g.
    ///
    /// The observer is `Arc`-shared so the EXECUTE caller (PROFILE
    /// renderer or [`crate::observer::ReplanController`]) can read the
    /// accumulated state after `execute()` returns. Defaults to `None`
    /// — only PROFILE / replan paths attach an observer.
    ///
    /// # M4-71 ↔ executor coupling
    ///
    /// The dispatcher in [`crate::executor::ops::PhysicalOperator::next_batch`]
    /// reads this slot once per batch and (when present) calls
    /// [`crate::observer::dispatcher::record_dispatch`] to record the
    /// batch's row count, wall-time, and high-water memory. The hot
    /// path takes ZERO observer cost when no observer is attached
    /// (the option-presence check is one branch on a non-null Arc
    /// pointer comparison).
    #[must_use]
    pub fn with_observer(mut self, observer: Arc<RowCountObserver>) -> Self {
        self.observer = Some(observer);
        self
    }

    /// Borrow the attached row-count observer if any.
    #[inline]
    #[must_use]
    pub fn observer(&self) -> Option<&Arc<RowCountObserver>> {
        self.observer.as_ref()
    }

    /// Tenant identity.
    #[inline]
    #[must_use]
    pub fn tenant(&self) -> TenantId {
        self.tenant
    }

    /// **ADR-197.** Builder: install a Bolt explicit-transaction held
    /// transaction so substrate write ops STAGE into it (EXPLICIT
    /// mode) instead of auto-committing per call. The Bolt handler
    /// calls this before `execute_*`, then [`Self::take_held_txn`]
    /// after, so it can `commit()` / `abort()` the (mutated) tx at the
    /// COMMIT / ROLLBACK message. The tx is carried opaquely behind
    /// [`crate::executor::substrate::HeldTxnHandle`].
    #[must_use]
    pub fn with_held_txn(self, txn: Box<dyn crate::executor::substrate::HeldTxnHandle>) -> Self {
        // ADR-197-amendment-01 D-5: seed the context's snapshot LSN
        // from the held transaction so `ensure_snapshot_lsn` REPORTS
        // the pinned snapshot explicit-mode reads actually observe
        // (the visibility itself is enforced by reading through the
        // held tx — amendment D-1; this accessor is the observable /
        // assertable half). Constant across the transaction's
        // statements: every statement's context re-seeds from the
        // SAME handle, whose LSN was fixed at BEGIN.
        *self.snapshot_lsn.lock() = Some(txn.snapshot_lsn());
        *self.held_txn.lock() = Some(txn);
        self
    }

    /// **D-2 (ADR-147 §D-8) — install a held transaction through a
    /// shared borrow.**
    ///
    /// The `&self` counterpart to the builder-style [`Self::with_held_txn`],
    /// used by the D-2 statement-scoped autocommit path: the substrate's
    /// [`crate::executor::ExecutorSubstrate::begin_statement`] opens ONE
    /// transaction at statement start and installs it here so the AUTO-
    /// COMMIT write ops STAGE into it (the mature ADR-197 EXPLICIT path)
    /// for the duration of the statement, committing ONCE at
    /// [`crate::executor::ExecutorSubstrate::commit_statement`].
    ///
    /// Like [`Self::with_held_txn`] this seeds the context's snapshot LSN
    /// from the handle (ADR-197-amendment-01 D-5) so held-txn reads
    /// (`scan_nodes_with_context` etc.) observe the transaction's pinned
    /// snapshot — the identical read-your-writes visibility the explicit-tx
    /// path relies on for a `MATCH … CREATE` spine.
    ///
    /// # Panics
    ///
    /// Never installs OVER an existing held txn: the caller
    /// (`crate::materialize`) guards on `!self.has_held_txn()` before
    /// calling, so an explicit Bolt BEGIN…COMMIT transaction is never
    /// clobbered by a statement-scoped install.
    pub fn install_held_txn(&self, txn: Box<dyn crate::executor::substrate::HeldTxnHandle>) {
        *self.snapshot_lsn.lock() = Some(txn.snapshot_lsn());
        *self.held_txn.lock() = Some(txn);
    }

    /// **ADR-197.** Reclaim the held transaction after execution so the
    /// caller (Bolt handler) can `commit()` / `abort()` it. Returns
    /// `None` in auto-commit mode (no held tx was installed) or if a
    /// prior call already took it. The slot is left empty.
    #[must_use]
    pub fn take_held_txn(&self) -> Option<Box<dyn crate::executor::substrate::HeldTxnHandle>> {
        self.held_txn.lock().take()
    }

    /// **ADR-197.** Whether this execution is in EXPLICIT-transaction
    /// mode (a held tx is installed). Auto-commit mode returns `false`.
    /// Substrate write ops use this to branch between staging into the
    /// held tx and the one-call-one-tx commit path.
    #[inline]
    #[must_use]
    pub fn has_held_txn(&self) -> bool {
        self.held_txn.lock().is_some()
    }

    /// **ADR-197.** Run `f` with a mutable borrow of the held
    /// transaction handle, if one is installed (EXPLICIT mode).
    /// Returns `Some(f(&mut handle))` in explicit mode, `None` in
    /// auto-commit mode. Substrate write ops call this to downcast the
    /// handle back to the concrete `OwnedTxn` and stage their CRUD
    /// operation into the one held transaction (e.g.
    /// `crud::create_node(&crud, owned.txn_mut(), …)`) WITHOUT
    /// committing — the commit happens later at the Bolt COMMIT
    /// message. The lock is held only for the duration of `f` (the
    /// connection executes one statement at a time, so it is
    /// uncontended).
    pub fn with_held_txn_mut<R>(
        &self,
        f: impl FnOnce(&mut dyn crate::executor::substrate::HeldTxnHandle) -> R,
    ) -> Option<R> {
        let mut guard = self.held_txn.lock();
        guard.as_mut().map(|b| f(b.as_mut()))
    }

    /// Clone the narrow explicit-transaction access handle used by owned
    /// production cursors.
    ///
    /// Unlike cloning [`ExecutionContext`], this handle contains no
    /// thread-affine MERGE guards and is therefore safe to store in a
    /// [`crate::executor::substrate::BoundEdgeCursor`]. The cursor must still
    /// treat a missing/finalized transaction as an execution error.
    #[must_use]
    pub fn held_txn_access(&self) -> HeldTxnAccess {
        HeldTxnAccess {
            inner: Arc::clone(&self.held_txn),
        }
    }

    /// **NN-4 (#1384) re-spin.** Stash a MERGE serialization guard so it
    /// SPANS the statement commit rather than dropping at pipeline exit.
    ///
    /// The query driver's [`crate::executor::ops::acquire_merge_guards`]
    /// calls this after acquiring each guard (BEFORE the statement's
    /// snapshot pin); the guard is drained + dropped by the driver
    /// ([`crate::materialize`] or [`crate::executor::execute_with_context`])
    /// AFTER the create is durable (post-`commit_statement` on the D-2
    /// auto-commit path, or post-loop on the eager path where each op
    /// auto-committed). See the [`Self::merge_guards`] field doc for the
    /// durability-window rationale.
    ///
    /// A path-shape MERGE stashes MORE than one guard (one per endpoint
    /// key — Fix 3); node-shape stashes one. `None` guards (stub /
    /// read-only substrate) are never stashed (the caller only calls this
    /// with a real guard).
    pub(crate) fn stash_merge_guard(&self, guard: Box<dyn crate::executor::substrate::MergeGuard>) {
        self.merge_guards.lock().push(guard);
    }

    /// **NN-4 (#1384) re-spin.** Drain the stashed MERGE serialization
    /// guards, transferring ownership to the caller so they drop at the
    /// caller's chosen point — AFTER the statement's writes are durable.
    ///
    /// The query driver calls this AFTER `commit_statement` /
    /// `rollback_statement` (auto-commit / D-2 path) or after the eager
    /// materialize loop ([`crate::executor::execute_with_context`], where
    /// each op auto-committed inside `next_batch`). Dropping the returned
    /// `Vec` releases the per-key mutexes, unblocking the next racer —
    /// which then re-probes and sees the winner's now-COMMITTED node.
    ///
    /// Idempotent: a second call returns an empty `Vec` (the slot is left
    /// empty). A read plan / non-MERGE statement drains an empty `Vec`
    /// (no-op). This is deliberately a DRAIN (not a peek): the guards
    /// MUST be moved out and dropped at the driver's post-commit point,
    /// never held to context-drop (which for a multi-statement query
    /// would leak the lock across statements).
    #[must_use = "the drained MergeGuards release their locks on drop; bind them so \
                  they drop at the post-commit point, or the lock is released early"]
    pub(crate) fn take_merge_guards(&self) -> Vec<Box<dyn crate::executor::substrate::MergeGuard>> {
        std::mem::take(&mut *self.merge_guards.lock())
    }

    /// Local partition identity. Always [`PartitionId::ZERO`].
    #[inline]
    #[must_use]
    pub fn partition(&self) -> PartitionId {
        self.partition
    }

    /// Query identity.
    #[inline]
    #[must_use]
    pub fn query_id(&self) -> QueryId {
        self.query_id
    }

    /// Borrow the cancellation token. Operators clone this cheaply
    /// at start-of-batch and call [`CancellationToken::check`] before
    /// pulling more rows from the substrate.
    #[inline]
    #[must_use]
    pub fn cancellation(&self) -> &CancellationToken {
        &self.cancellation
    }

    /// Borrow the per-query tracing span.
    #[inline]
    #[must_use]
    pub fn tracing_span(&self) -> &tracing::Span {
        &self.span
    }

    /// Acquire (if needed) and return the snapshot LSN.
    ///
    /// Per ADR-038 §2 D-18 rule 1: the FIRST call to this function
    /// per context captures the LSN at execute-time, pre-first-batch;
    /// subsequent calls return the same captured value. v1.0-alpha captures `Lsn::MAX` (read-latest
    /// sentinel — no MVCC writer is running yet); production wiring
    /// at M4-08+ will route through `arcgraph_storage::wal::current_lsn()`.
    ///
    /// # Concurrency
    ///
    /// Multiple operators within the same plan may call this — once
    /// the LSN has been captured, all subsequent calls observe the
    /// SAME value (the in-mutex `Option::is_none()` check guards
    /// against double-capture). The lock is held briefly (one atomic
    /// store + the LSN value); contention is bounded to the very
    /// first `next_batch` call.
    pub fn ensure_snapshot_lsn(&self) -> Lsn {
        let mut guard = self.snapshot_lsn.lock();
        match *guard {
            Some(lsn) => lsn,
            None => {
                let lsn = Lsn::MAX;
                *guard = Some(lsn);
                tracing::debug!(
                    target: "arcgraph_query::executor::context",
                    parent: &self.span,
                    lsn = lsn.raw(),
                    "snapshot LSN captured pre-first-batch (D-18 rule 1)",
                );
                lsn
            }
        }
    }

    /// Read the snapshot LSN if already captured. Returns `None` if
    /// [`Self::ensure_snapshot_lsn`] has never been called.
    ///
    /// Used by tests + the EXPLAIN test pin to assert the LSN was
    /// (or was NOT) acquired during a query.
    #[must_use]
    pub fn snapshot_lsn(&self) -> Option<Lsn> {
        *self.snapshot_lsn.lock()
    }

    /// Release the captured snapshot LSN per ADR-038 §2 D-18 rule 4
    /// + amendment-03 §TIER-1 GAP E rule 4: "Snapshot LSN released at
    ///   query-end / cursor-close. … Snapshot release is unconditional
    ///   and idempotent (release-on-already-released is a no-op)."
    ///
    /// At v1.0-alpha the captured LSN is [`Lsn::MAX`] (read-latest
    /// sentinel); release is conceptual until the M4-08+ wiring layer
    /// routes through `arcgraph_storage::wal::release_snapshot(lsn)`.
    /// The slot is reset to `None` so a future re-use of the context
    /// (rare path — for example, a per-query test harness that drives
    /// multiple back-to-back materializations on the same context) can
    /// re-acquire fresh.
    ///
    /// # Idempotence
    ///
    /// Calling release on a context that never acquired (no
    /// `ensure_snapshot_lsn` call) is a no-op. Calling release twice
    /// is a no-op on the second call. The `release_snapshot_lsn`
    /// method does not return an error nor a bool — the contract is
    /// "best-effort release with last-write-wins idempotence".
    ///
    /// # Forward-method
    ///
    /// M4-08+ wiring at the production storage layer: the no-op slot
    /// reset becomes a real `wal_router.release_snapshot(lsn)` call;
    /// the API surface stays stable (no signature change).
    pub fn release_snapshot_lsn(&self) {
        let mut guard = self.snapshot_lsn.lock();
        if guard.is_some() {
            tracing::debug!(
                target: "arcgraph_query::executor::context",
                parent: &self.span,
                "snapshot LSN released (D-18 rule 4 / amendment-03 §TIER-1 GAP E rule 4)",
            );
            // W13β fix-up M-1: light the consumption latch on the actual
            // release path (slot was Some). Subsequent
            // `StreamingCursor::open` / `materialize::materialize` calls
            // on this context (or any clone) will reject — see
            // [`Self::lsn_consumed`] for the rule-5 rationale.
            self.lsn_consumed.store(true, Ordering::Release);
        }
        *guard = None;
    }

    /// W13β fix-up M-1 — `true` iff this context's snapshot LSN was
    /// previously captured AND released (i.e. a prior cursor or
    /// materialize call ran to completion / cleanup on this context).
    ///
    /// `StreamingCursor::open` and [`crate::materialize::materialize`]
    /// reject on consumed contexts to prevent re-acquiring a fresh LSN
    /// — which at production-LSN-binding time (M4-08+) would observe a
    /// different point-in-time than the originating cursor saw,
    /// breaking openCypher snapshot semantics per ADR-038 amendment-03
    /// §TIER-1 GAP E rule 5.
    ///
    /// Cloning the context propagates the latch: an `Arc<AtomicBool>`
    /// is shared across clones, so a sibling context (open before close)
    /// that watches the latch flip after the original closes can
    /// detect "the LSN-bearing cursor I was sharing with has now
    /// closed" — though in practice the rejection happens at `open`
    /// time, before any operator runs.
    #[inline]
    #[must_use]
    pub fn lsn_consumed(&self) -> bool {
        self.lsn_consumed.load(Ordering::Acquire)
    }

    /// W13β fix-up M-1 — construct a structured "context already
    /// consumed" error for the cursor / materialize entry points to
    /// return when [`Self::lsn_consumed`] is set.
    ///
    /// Carrying the constructor here keeps the rejection-error shape
    /// uniform across both consumers (cursor + materialize) — the
    /// renderers (M5-07 / M5-11 / M5-13) pattern-match on the
    /// `feature` slot to surface a "client misuse: ExecutionContext
    /// re-used after cursor close" diagnostic.
    pub(crate) fn lsn_consumed_error(&self, feature: &'static str) -> ExecutionError {
        ExecutionError::Plan(ArcQLError::Internal {
            feature: feature.to_owned(),
            reason:
                "ExecutionContext snapshot LSN was previously released; rule 5 (D-18 / TIER-1 GAP E) \
                 forbids re-acquiring a fresh LSN on a re-used context — open a new ExecutionContext \
                 instead of close-then-reopen"
                    .to_owned(),
            span: Span::point(0, 0),
        })
    }

    /// Construct a [`SnapshotLsnGuard`] bound to this context. Drop
    /// of the guard calls [`Self::release_snapshot_lsn`].
    ///
    /// The guard is the canonical RAII surface for query-scoped
    /// snapshot-LSN ownership. M4-81 (single-batch materialize) holds
    /// the guard for the duration of the materialize loop; M4-82
    /// (streaming cursor) holds the guard in [`crate::StreamingCursor`]
    /// from open-time until [`crate::StreamingCursor::close`] (or
    /// [`Drop`]). Both paths inherit the panic-unwind safety of RAII
    /// per `feedback_seqlock_panic_safety_primitive.md`'s "RAII guard
    /// is the canonical panic-safety primitive when the cleanup is a
    /// single mutation" sub-rule (the W12γ MED-3 pattern).
    #[must_use = "SnapshotLsnGuard releases the LSN when dropped; bind to a name to keep the LSN alive"]
    pub fn snapshot_lsn_guard(&self) -> SnapshotLsnGuard<'_> {
        SnapshotLsnGuard { ctx: self }
    }

    /// Bump the batch counter and return the new value. Operators
    /// call this once at the top of each `next_batch` to maintain a
    /// per-query batch sequence number for slow-query log emission +
    /// future M4-71 per-batch row-count observation.
    pub fn next_batch_seq(&self) -> u64 {
        self.batch_seq.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Read the current batch counter without bumping. Useful for
    /// tests that verify a particular operator was driven N times.
    #[inline]
    #[must_use]
    pub fn batches_executed(&self) -> u64 {
        self.batch_seq.load(Ordering::Relaxed)
    }

    // ---------- ADR-192 (#623) — CALL{} correlation frames ----------

    /// Push a `CALL { … }` correlation frame (the imported bindings +
    /// their values for one driving row). [`crate::executor::ops::CallOp`]
    /// pushes BEFORE driving the subquery body for a driving row.
    pub(crate) fn push_correlation_frame(&self, frame: CorrelationFrame) {
        self.correlation.lock().push(frame);
    }

    /// Pop the innermost `CALL { … }` correlation frame.
    /// [`crate::executor::ops::CallOp`] pops when a driving row's body is
    /// exhausted (balanced with [`Self::push_correlation_frame`]).
    pub(crate) fn pop_correlation_frame(&self) {
        self.correlation.lock().pop();
    }

    /// Resolve an imported binding's value from the correlation-frame
    /// stack, nearest-frame-wins (innermost CALL level first). Returns
    /// `None` if no active frame provides `binding` (e.g. an
    /// uninitialized seed outside any CALL drive — the schema-only
    /// state). Read by [`crate::executor::ops::CorrelationSeedOp`] when it
    /// emits the per-driving-row seed row.
    #[must_use]
    pub(crate) fn correlation_value(&self, binding: BindingId) -> Option<Value> {
        let stack = self.correlation.lock();
        for frame in stack.iter().rev() {
            if let Some((_, v)) = frame.iter().find(|(b, _)| *b == binding) {
                return Some(v.clone());
            }
        }
        None
    }
}

/// RAII guard that releases the [`ExecutionContext`]'s captured
/// snapshot LSN on drop per ADR-038 §2 D-18 rule 4 + amendment-03
/// §TIER-1 GAP E rule 4.
///
/// The guard is borrow-bound to the [`ExecutionContext`] (`'ctx`); a
/// caller cannot move the guard past the context's lifetime. The
/// guard's [`Drop`] calls [`ExecutionContext::release_snapshot_lsn`]
/// unconditionally — a query that never captured an LSN drops a
/// no-op guard, matching the idempotent contract.
///
/// # Panic-unwind safety
///
/// Per `feedback_seqlock_panic_safety_primitive.md`, RAII guards are
/// the canonical panic-safety primitive when the cleanup is a single
/// mutation (here, a `Mutex<Option<Lsn>>` reset). A panic anywhere in
/// the materialize loop (or cursor-next-batch loop) drops the guard
/// during stack unwind, releasing the LSN. The W12γ MED-3 RAII
/// `RegistryGuard` pattern in `explain.rs` is the sister surface for
/// the per-query `CancellationRegistry` entry — both follow the same
/// "no leak on panic" discipline.
#[must_use = "SnapshotLsnGuard releases the LSN when dropped; bind to a name to keep the LSN alive"]
pub struct SnapshotLsnGuard<'ctx> {
    ctx: &'ctx ExecutionContext,
}

impl<'ctx> Drop for SnapshotLsnGuard<'ctx> {
    fn drop(&mut self) {
        self.ctx.release_snapshot_lsn();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ADR-197-amendment-01 D-5 pin: installing a held transaction
    /// seeds the context's snapshot LSN from the handle, so
    /// `ensure_snapshot_lsn` REPORTS the pinned snapshot explicit-mode
    /// reads observe — constant across the transaction's statements
    /// (each statement re-seeds from the SAME handle).
    #[test]
    fn with_held_txn_seeds_snapshot_lsn_from_handle() {
        #[derive(Debug)]
        struct FakeHeld(Lsn);
        impl crate::executor::substrate::HeldTxnHandle for FakeHeld {
            fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
                self
            }
            fn snapshot_lsn(&self) -> Lsn {
                self.0
            }
        }

        let pinned = Lsn::new(42);
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO)
            .with_held_txn(Box::new(FakeHeld(pinned)));
        assert_eq!(
            ctx.snapshot_lsn(),
            Some(pinned),
            "with_held_txn must seed the pinned snapshot"
        );
        assert_eq!(
            ctx.ensure_snapshot_lsn(),
            pinned,
            "ensure_snapshot_lsn reports the pinned LSN, not the Lsn::MAX placeholder"
        );
    }

    #[test]
    fn fresh_context_has_no_snapshot_lsn() {
        // ADR-038 §2 D-18 rule 1 pin: a context without an
        // ensure_snapshot_lsn call MUST report None — capture only
        // happens at execute-time, pre-first-batch. EXPLAIN's
        // no-snapshot-LSN discipline depends on this.
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        assert_eq!(ctx.snapshot_lsn(), None);
    }

    #[test]
    fn ensure_snapshot_lsn_captures_once_and_holds() {
        // ADR-038 §2 D-18 rule 1: lazy capture (execute-time,
        // pre-first-batch), hold for query life.
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let a = ctx.ensure_snapshot_lsn();
        let b = ctx.ensure_snapshot_lsn();
        assert_eq!(a, b, "two ensure calls return the same captured LSN");
        assert_eq!(ctx.snapshot_lsn(), Some(a));
    }

    #[test]
    fn ensure_snapshot_lsn_is_lsn_max_at_v1_alpha() {
        // v1.0-alpha placeholder: read-latest sentinel until the
        // production wiring at M4-08+ routes through
        // `arcgraph_storage::wal::current_lsn`.
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        assert_eq!(ctx.ensure_snapshot_lsn(), Lsn::MAX);
    }

    #[test]
    fn cancellation_token_default_is_un_tripped() {
        let t = CancellationToken::new();
        assert!(!t.is_cancelled());
        assert!(t.check().is_ok());
    }

    #[test]
    fn cancellation_token_trips_idempotently() {
        let t = CancellationToken::new();
        t.cancel();
        assert!(t.is_cancelled());
        // Idempotent — repeated calls preserve tripped state.
        t.cancel();
        assert!(t.is_cancelled());
        assert_eq!(t.check(), Err(CancellationError));
    }

    #[test]
    fn cancellation_token_clone_shares_state() {
        // Cancellation is wired via Arc — cloning shares the underlying
        // flag. Operator pipelines clone the token from
        // ExecutionContext and pass it down; the issuer thread's
        // `cancel()` MUST trip every clone.
        let a = CancellationToken::new();
        let b = a.clone();
        a.cancel();
        assert!(b.is_cancelled());
    }

    #[test]
    fn context_is_clone_and_shares_cancellation() {
        // Cloning context shares the cancellation flag (Arc-backed)
        // and the snapshot-LSN slot (also Arc<Mutex>) — the contract
        // is "many operator clones, one cancellation source, one
        // snapshot LSN".
        let a = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let b = a.clone();
        a.cancellation().cancel();
        assert!(b.cancellation().is_cancelled());

        // Snapshot LSN captured via one clone shows up via the other.
        let lsn = a.ensure_snapshot_lsn();
        assert_eq!(b.snapshot_lsn(), Some(lsn));
    }

    #[test]
    fn query_id_is_uuid_v7() {
        let id = QueryId::new();
        let u = id.as_uuid();
        // RFC 9562 §6.6: version nibble at byte 6 high-nibble = 7.
        assert_eq!(u.get_version_num(), 7);
    }

    #[test]
    fn query_id_display_renders_canonical_form() {
        // Hyphenated lowercase, 36 chars — RFC 9562 canonical form.
        let id = QueryId::new();
        let s = id.to_string();
        assert_eq!(s.len(), 36);
        assert_eq!(s.matches('-').count(), 4);
    }

    #[test]
    fn batch_seq_increments_monotonically() {
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        assert_eq!(ctx.batches_executed(), 0);
        assert_eq!(ctx.next_batch_seq(), 1);
        assert_eq!(ctx.next_batch_seq(), 2);
        assert_eq!(ctx.batches_executed(), 2);
    }

    #[test]
    fn with_query_id_preserves_caller_supplied_uuid() {
        // Tests that need deterministic span output can pin the
        // query-id explicitly.
        let pinned = QueryId::from_uuid(uuid::uuid!("01010101-0101-7101-8101-010101010101"));
        let ctx = ExecutionContext::with_query_id(TenantId::DEFAULT, PartitionId::ZERO, pinned);
        assert_eq!(ctx.query_id(), pinned);
    }

    #[test]
    fn partition_default_is_zero_at_v1() {
        // ADR-024 amendment-02 invariant pin: v1.0 stamps
        // PartitionId::ZERO. The context honors whatever the catalog
        // supplies; the pin lives in the StubCatalogProvider default.
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        assert_eq!(ctx.partition(), PartitionId::ZERO);
    }

    // -----------------------------------------------------------------
    // W13β M4-81 / M4-82 — snapshot-LSN release pin set
    // -----------------------------------------------------------------

    #[test]
    fn release_snapshot_lsn_clears_captured_slot() {
        // ADR-038 §2 D-18 rule 4 / amendment-03 §TIER-1 GAP E rule 4:
        // "Snapshot LSN released at query-end / cursor-close." The
        // release method clears the lazily-captured LSN slot; a
        // subsequent `ensure` re-captures fresh.
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let _ = ctx.ensure_snapshot_lsn();
        assert!(ctx.snapshot_lsn().is_some());
        ctx.release_snapshot_lsn();
        assert!(
            ctx.snapshot_lsn().is_none(),
            "release clears the captured slot"
        );
    }

    #[test]
    fn release_snapshot_lsn_is_idempotent_on_unacquired() {
        // Calling release on a context that never acquired must be a
        // no-op (the EXPLAIN path constructs a context, never calls
        // ensure, and may still walk through an LSN guard via the
        // cursor surface — the contract: release-on-unacquired = no
        // panic, no error).
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        assert!(ctx.snapshot_lsn().is_none());
        ctx.release_snapshot_lsn();
        ctx.release_snapshot_lsn();
        assert!(
            ctx.snapshot_lsn().is_none(),
            "release stays None after no-op"
        );
    }

    #[test]
    fn release_snapshot_lsn_is_idempotent_on_double_release() {
        // Per the W12γ MED-3 RAII guard pattern, calling release after
        // a prior release is a no-op; this lets the cursor's `close()`
        // method coexist with the Drop guard without double-release
        // bugs.
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let _ = ctx.ensure_snapshot_lsn();
        ctx.release_snapshot_lsn();
        ctx.release_snapshot_lsn();
        assert!(ctx.snapshot_lsn().is_none());
    }

    #[test]
    fn snapshot_lsn_guard_releases_on_drop() {
        // The RAII surface — drop releases per
        // `feedback_seqlock_panic_safety_primitive.md`'s "RAII guard
        // is the canonical panic-safety primitive when cleanup is a
        // single mutation" sub-rule.
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let _ = ctx.ensure_snapshot_lsn();
        assert!(ctx.snapshot_lsn().is_some());
        {
            let _guard = ctx.snapshot_lsn_guard();
            assert!(
                ctx.snapshot_lsn().is_some(),
                "guard alive: LSN still captured"
            );
        }
        assert!(ctx.snapshot_lsn().is_none(), "guard dropped: LSN released");
    }

    #[test]
    fn snapshot_lsn_guard_releases_on_panic_unwind() {
        // Panic-safety pin: a panic INSIDE the guard's scope still
        // releases the LSN via Drop during stack unwind. Mirrors the
        // W12γ MED-3 RegistryGuard panic-unwind discipline. Uses
        // catch_unwind so the test process survives.
        use std::panic::AssertUnwindSafe;
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let _ = ctx.ensure_snapshot_lsn();
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            let _guard = ctx.snapshot_lsn_guard();
            panic!("synthetic panic for unwind-safety pin");
        }));
        assert!(result.is_err(), "expected the panic to propagate");
        assert!(
            ctx.snapshot_lsn().is_none(),
            "panic-unwind dropped the guard, releasing the LSN"
        );
    }

    #[test]
    fn snapshot_lsn_guard_re_acquire_after_release_yields_fresh_lsn_at_ctx_layer() {
        // Per ADR-038 amendment-03 §TIER-1 GAP E rule 5 ("replan does
        // NOT re-acquire snapshot LSN") the re-acquire path is NOT
        // exercised on the replan loop. W13β fix-up M-1 wires the
        // [`Self::lsn_consumed`] latch on top of the ctx-level slot:
        // direct callers of `ensure_snapshot_lsn` after `release_snapshot_lsn`
        // STILL see a re-acquire (preserving the ctx-level
        // `Mutex<Option<Lsn>>` semantics for the back-to-back-query
        // test path); but the cursor + materialize entry points
        // consult `lsn_consumed()` and reject before any operator
        // runs. This pin asserts the ctx-level loose contract; the
        // strict cursor/materialize contract is pinned in
        // `lsn_consumed_*` siblings below + in cursor.rs / materialize.rs.
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let first = ctx.ensure_snapshot_lsn();
        ctx.release_snapshot_lsn();
        let second = ctx.ensure_snapshot_lsn();
        // v1.0-alpha LSN sentinel is Lsn::MAX so first == second on
        // value; the captured-slot transition (Some → None → Some) is
        // what the pin asserts.
        assert_eq!(first, second);
        assert!(ctx.snapshot_lsn().is_some());
    }

    // -----------------------------------------------------------------
    // W13β fix-up M-1 — `lsn_consumed` consumption latch pin set
    // -----------------------------------------------------------------

    #[test]
    fn lsn_consumed_starts_false_on_fresh_context() {
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        assert!(!ctx.lsn_consumed());
    }

    #[test]
    fn lsn_consumed_lights_on_first_actual_release() {
        // Latch lights when release_snapshot_lsn observes a Some(_)
        // slot (i.e. an actual release, not a no-op). Sticky thereafter.
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let _ = ctx.ensure_snapshot_lsn();
        assert!(!ctx.lsn_consumed(), "pre-release: latch is false");
        ctx.release_snapshot_lsn();
        assert!(ctx.lsn_consumed(), "post-release: latch is true");
        // Even after a re-ensure (the ctx-level back-to-back-query
        // path), the latch remains set.
        let _ = ctx.ensure_snapshot_lsn();
        assert!(ctx.lsn_consumed(), "re-acquire does NOT clear latch");
        ctx.release_snapshot_lsn();
        assert!(ctx.lsn_consumed(), "second release: still set");
    }

    #[test]
    fn lsn_consumed_does_not_light_on_no_op_release() {
        // Releasing on a never-acquired context is a no-op; the latch
        // stays false. EXPLAIN's no-LSN discipline (D-18 rule 1)
        // depends on this — an EXPLAIN-then-cursor-on-same-ctx
        // sequence MUST be permitted (EXPLAIN never acquires; a
        // subsequent cursor sees lsn_consumed=false and proceeds).
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        ctx.release_snapshot_lsn();
        assert!(!ctx.lsn_consumed(), "no-op release: latch stays false");
        ctx.release_snapshot_lsn();
        assert!(!ctx.lsn_consumed(), "second no-op release: still false");
    }

    #[test]
    fn lsn_consumed_propagates_through_clone() {
        // The latch is `Arc<AtomicBool>`-backed; clones share the
        // same flag. The close-then-reopen detection at
        // `StreamingCursor::open` relies on this — the second
        // cursor's preflight check on a clone of the original ctx
        // observes the latch flip.
        let a = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let b = a.clone();
        assert!(!a.lsn_consumed());
        assert!(!b.lsn_consumed());
        let _ = a.ensure_snapshot_lsn();
        a.release_snapshot_lsn();
        assert!(a.lsn_consumed(), "a: latch set");
        assert!(b.lsn_consumed(), "b clone observes latch via Arc share");
    }

    #[test]
    fn lsn_consumed_propagates_through_guard_drop() {
        // The RAII `SnapshotLsnGuard` is the canonical release path
        // (materialize + cursor both use it). Pin: guard.drop fires
        // release_snapshot_lsn which lights the latch.
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        {
            let _guard = ctx.snapshot_lsn_guard();
            let _ = ctx.ensure_snapshot_lsn();
            assert!(
                !ctx.lsn_consumed(),
                "guard alive + LSN captured: latch false"
            );
        }
        assert!(
            ctx.lsn_consumed(),
            "guard dropped → release fired → latch lit"
        );
    }

    #[test]
    fn lsn_consumed_error_carries_feature_label() {
        // The constructor's `feature` slot is the surface name the
        // M5-tier renderer surfaces in the diagnostic; pin the
        // round-trip.
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let err = ctx.lsn_consumed_error("TestFeature::open");
        match err {
            ExecutionError::Plan(ArcQLError::Internal {
                feature, reason, ..
            }) => {
                assert_eq!(feature, "TestFeature::open");
                assert!(
                    reason.contains("rule 5"),
                    "reason cites rule 5; got: {reason}"
                );
                assert!(
                    reason.contains("close-then-reopen"),
                    "reason names the misuse pattern; got: {reason}"
                );
            }
            other => panic!("expected Internal, got {other:?}"),
        }
    }
}
