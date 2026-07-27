//! M4-61 / M4-62 / M4-63 / M4-64a vectorized executor.
//!
//! # Slice scope
//!
//! - **M4-61 (M4-06a)**: ExecutionContext + 2048-row factorized
//!   [`Batch`] cursor + simple operators
//!   ([`ScanOp`](ops::scan::ScanOp) / [`ExpandOp`](ops::expand::ExpandOp) /
//!   [`FilterOp`](ops::filter::FilterOp) /
//!   [`ProjectOp`](ops::project::ProjectOp)) + per-batch cancellation
//!   gate per ADR-038 amendment-02 §M4.f.
//! - **M4-62 (M4-06b)**: Hybrid retrieval orchestration
//!   ([`RankByHybridOp`](ops::rank_by_hybrid::RankByHybridOp))
//!   composing M3.a HNSW + M3.b BM25 + M3.d community via the
//!   [`ExecutorSubstrate`] adapter (whose production binding will
//!   delegate to `arcgraph_storage::router::TenantHandle` per
//!   ADR-037 D-1 at the M4-08+ wiring layer); OPTIONAL MATCH
//!   [`OptionalExpandOp`](ops::optional_expand::OptionalExpandOp)
//!   per amendment-03 §TIER-1 GAP D + ADR-006 amendment-01 §A-2;
//!   3VL NULL handling per amendment-03 §TIER-2-b
//!   (NULL propagation in WHERE / AND / OR / NOT predicates per
//!   ADR-038 §2 D-20).
//! - **M4-63 (M4-06c — this slice)**: Aggregation
//!   ([`AggregateOp`](ops::aggregate::AggregateOp) with
//!   COUNT/SUM/AVG/MIN/MAX/COLLECT per amendment-03 §TIER-2-b 3VL
//!   aggregate semantics) + sort
//!   ([`SortOp`](ops::sort::SortOp), stable + spillover via budget) +
//!   path ([`NamedShortestPathOp`](ops::path::NamedShortestPathOp),
//!   single-source + bidirectional BFS only — no DFS/A* at v1.0) +
//!   limit ([`LimitOp`](ops::limit::LimitOp)) operators per
//!   amendment-02 §M4.f.
//! - **M4-64a (M4-06d-1 — this slice)**: Per-tenant memory budget
//!   ([`budget::MemoryBudget`]) per amendment-03 §Structural-1.
//!   Tracking-allocator-style accounting; per-batch enforcement via
//!   [`budget::MemoryBudget::try_reserve_unscoped`]; M5-12 rate-limit
//!   config consumes [`budget::MemoryBudget::set_per_tenant_cap`]
//!   forward-method. Replaces the W11Z #272
//!   [`ops::expand::SPILLOVER_MAX_ROWS`] row-count cap with a proper
//!   per-tenant byte budget. For tenants WITHOUT a configured byte cap
//!   (uncapped = no memory limit) the accumulators grow with the actual
//!   cardinality, guarded against a true runaway by
//!   [`ops::expand::UNCAPPED_RUNAWAY_GUARD_ROWS`] (#980 lifted the old
//!   131 072-row `BUDGET_FALLBACK_ROWS` valve that broke legitimate
//!   large traversals / joins / sorts / PROFILE result sets).
//!
//! Forward-deferred slices (do NOT live here):
//!
//! - **M4-64b** — SIMD predicate evaluation. The
//!   [`Batch::row_count`] / dense-`Vec`-per-binding row layout is
//!   the forward-pin for the M4-64b columnar specialization;
//!   this slice ships a row-major fallback inside `eval`. The
//!   factorized intermediate re-shape will reuse the
//!   [`budget::MemoryBudget`] surface unchanged.
//! - **M4-71** — row-count observer feedback to the M4-04 catalog
//!   stats. The
//!   [`crate::explain::ExecutionMetrics`] type already carries the
//!   field set; M4-71 wires the observer.
//! - **M4-92** — execution-time cancellation triggers (Bolt-side
//!   client cancel; M4-83 multi-statement boundary). The
//!   [`context::CancellationToken`] surface is forward-pinned here so
//!   M4-92's wiring layer plugs in without touching operator code.
//!
//! # Why a CONCRETE pipeline (no `Operator` trait)
//!
//! Per the 7-slice 3-strike pattern — M4-21 binding visitor (custom
//! struct), M4-22 typecheck visitor (custom; speculative
//! `BoundAstVisitor` deleted), M4-23 cross-substrate validator
//! (custom), M4-31..M4-33 plan-lowering visitors (custom), M4-51 cost
//! walker (custom) — the executor is the SEVENTH consumer in this
//! lineage. The same discipline applies: the
//! [`ops::PhysicalOperator`] type is a CONCRETE enum, NOT a trait
//! abstraction. M4-72, the v1.1 second-executor inflection point,
//! re-evaluates trait extraction
//! when a second consumer arrives. See
//! `feedback_avoid_speculative_scaffolding.md`.
//!
//! # Why an [`ExecutorSubstrate`] trait (despite no-traits-without-≥2-consumers)
//!
//! The substrate-access trait is NOT speculative scaffolding because
//! it has FIVE in-slice consumers within this same milestone:
//! [`ScanOp`](ops::scan::ScanOp) / [`ExpandOp`](ops::expand::ExpandOp)
//! / [`RankByHybridOp`](ops::rank_by_hybrid::RankByHybridOp) (which
//! reads vector + BM25) / community lookup. The trait's role is
//! parallel to the long-standing
//! [`crate::semantic::CatalogProvider`] pattern: a query-side abstract
//! seam that production wiring (`arcgraph-storage` at M4-08+) and tests
//! (`StubExecutorSubstrate`) implement separately. The 3-strike
//! discipline rejects traits with ONE imagined consumer; this trait has
//! five in-slice real consumers, satisfying the multi-consumer bar.
//!
//! # ADR provenance
//! - **ADR-038 amendment-02 §M4.f** — primary M4.f executor scope
//!   citation; M4-61 (M4-06a) + M4-62 (M4-06b) decomposition.
//! - **ADR-038 amendment-03 §TIER-1 GAP D** — OPTIONAL MATCH null-row
//!   semantics at execution time.
//! - **ADR-038 amendment-03 §TIER-1 GAP E** — snapshot-LSN field on
//!   `BoundQuery` populated at execute-time (lazy, pre-first-batch).
//! - **ADR-038 amendment-03 §TIER-2-b** — 3VL NULL handling lock.
//! - **ADR-038 amendment-03 §TIER-2-c** — RANK BY HYBRID 3-substrate
//!   composition.
//! - **ADR-038 §2 D-20** — Cypher 3VL truth tables.
//! - **ADR-006 amendment-01 §A-2** — OPTIONAL MATCH lowers to a
//!   left-outer join.
//! - **ADR-037 §D-1** — `TenantHandle` per-tenant substrate
//!   composition (the production binding [`ExecutorSubstrate`] will
//!   delegate to at M4-08+).
//! - **bounded-context policy** — implementer-vs-orchestrator discipline; this
//!   slice was implemented directly by a spawned implementer agent.

pub mod batch;
pub mod budget;
pub mod context;
pub mod error;
pub mod eval;
pub mod fusion;
pub mod ops;
pub mod pipeline;
pub(crate) mod projection;
pub mod simd;
pub mod substrate;
pub mod three_vl;
pub mod value;

pub use batch::{BATCH_ROWS, Batch};
pub use budget::{
    BUDGET_FALLBACK_ROWS, MemoryBudget, MemoryReservation, estimate_row_bytes, estimate_value_bytes,
};
pub use context::{
    CancellationError, CancellationToken, ExecutionContext, QueryId, SnapshotLsnGuard,
};
pub use error::{ExecutionError, ExecutorSpillError, ExecutorSpillFailureKind};
pub use fusion::rrf_fuse;
pub use ops::PhysicalOperator;
pub use pipeline::Pipeline;
pub use substrate::{
    BoundEdge, BoundEdgeCursor, BoundNode, ExecutorSubstrate, RankedHit, StubExecutorSubstrate,
    SubstrateAccessError,
};
pub use three_vl::ThreeValued;
pub use value::Value;

use crate::executor::ops::expand::UNCAPPED_RUNAWAY_GUARD_ROWS;
use crate::logical_plan::LogicalPlan;
use crate::semantic::CatalogProvider;

/// Drive a [`LogicalPlan`] through the vectorized executor end-to-end
/// and materialize ALL rows.
///
/// This is the v1.0-alpha M5↔M4 execute entry point per ADR-038
/// amendment-03 §M5↔M4 contract surface. v1.0-alpha collects the full
/// row stream into a `Vec<Vec<Value>>`; M4-08a (forward) lights the
/// streaming cursor surface. Each row is the operator's own column
/// projection (typically the `Project` node's items when the plan is
/// rooted at a projection).
///
/// # Snapshot-LSN discipline (ADR-038 §2 D-18 rule 1)
///
/// The execute path acquires a snapshot LSN LAZILY — pre-first-batch,
/// not at construction time (rule 1: "Acquired at execute-time, before
/// first batch pull"). The acquisition flows through
/// [`ExecutionContext::ensure_snapshot_lsn`]; once acquired, every
/// operator pulling from the substrate observes the same point-in-time
/// view for the remainder of the query. v1.0-alpha uses [`arcgraph_core::Lsn::MAX`]
/// (read-latest) as the acquired value because no MVCC writer is
/// running yet (M5/storage executor wiring binds the real LSN at
/// M4-08+); the API contract is what's load-bearing here.
///
/// # Cancellation
///
/// Each operator's `next_batch` polls the cancellation token at
/// batch boundaries (NOT row boundaries) per amendment-02 §M4.f
/// "per-operator batch-boundary check". A canceled query surfaces
/// [`ExecutionError::Cancelled`]. Per-row cancel granularity ships at
/// M4-92 if the v1.1 SLO budget requires it; v1.0-alpha's 2048-row
/// [`Batch`] is well inside the M5-12 cancel-latency budget per
/// ADR-036 §D-24.
///
/// # Errors
///
/// - [`ExecutionError::Cancelled`] — cancellation token tripped.
/// - [`ExecutionError::Substrate`] — substrate-access fault (e.g.,
///   index unavailable; v1.0-alpha stub substrates rarely surface
///   these; production binding may).
/// - [`ExecutionError::NotImplemented`] — the plan contains an operator
///   reserved for M4-63 (aggregation / sort / DISTINCT / UNWIND /
///   path / dynamic LIMIT). The variant carries a forward-link cite.
pub fn execute<C, S>(
    plan: &LogicalPlan,
    catalog: &C,
    substrate: &S,
) -> Result<Vec<Vec<Value>>, ExecutionError>
where
    C: CatalogProvider,
    S: ExecutorSubstrate,
{
    let ctx = ExecutionContext::new(catalog.tenant(), catalog.partition());
    execute_with_context(plan, substrate, &ctx)
}

/// Drive a [`LogicalPlan`] through the executor with a caller-supplied
/// [`ExecutionContext`].
///
/// The escape hatch for tests + future M4-92 cancellation wiring that
/// needs to share an [`ExecutionContext`] across multiple plans (e.g.,
/// to inject a pre-tripped [`CancellationToken`] or a custom
/// [`tracing::Span`] parent).
///
/// Per ADR-038 §2 D-18 rule 1, the snapshot LSN is acquired at the
/// FIRST [`PhysicalOperator::next_batch`] call (lazy / pre-first-batch
/// — see [`ExecutionContext::ensure_snapshot_lsn`]); this entry point
/// does NOT acquire one eagerly.
pub fn execute_with_context<S>(
    plan: &LogicalPlan,
    substrate: &S,
    ctx: &ExecutionContext,
) -> Result<Vec<Vec<Value>>, ExecutionError>
where
    S: ExecutorSubstrate,
{
    // NN-4 (#1384) re-spin, Fix 1 — acquire the MERGE get-or-create
    // serialization guard(s) BEFORE building/driving the pipeline. This
    // eager entry point has NO D-2 statement-txn wrap — each write op
    // auto-commits inside its own `next_batch`, and reads use the
    // read-latest `Lsn::MAX` snapshot — so a guard held from here until the
    // scope-guard drain spans the whole match→create→commit critical
    // section on this path too. The shipped `merge_guard` is INFALLIBLE, so
    // the acquire `?` here never fires in production; the `MergeGuardDrain`
    // is bound on the NEXT line (AFTER the acquire) and drops (releasing the
    // guards) on every POST-ACQUIRE exit — the `?` early-returns below, the
    // runaway early-return, the success return, and a panic-unwind — so no
    // per-key lock ever leaks to context-drop. (It does NOT cover an error
    // from the acquire itself — bound after it — but that path is
    // unreachable with the infallible guard.) See `acquire_merge_guards`.
    crate::executor::ops::acquire_merge_guards(plan, substrate, ctx)?;
    let _merge_guard_drain = MergeGuardDrain { ctx };
    // #797 — thread the context's parameter bag into the operator tree
    // (empty for non-parameterized executions → identical to the prior
    // `Pipeline::build`).
    let mut op = Pipeline::build_with_parameters(plan, ctx.parameters())?;
    // #980 NIT-1 — this eager result-Vec feeds the public `execute()`
    // Vec API + the PROFILE path (`profile_with_substrate`). It is the
    // 7th accumulator that carried the mis-tuned fixed `BUDGET_FALLBACK_ROWS`
    // (= 131 072) ceiling. Pre-fix it clipped UNCONDITIONALLY (no
    // `has_cap` check), so `PROFILE MATCH ()-[r]->() RETURN r` over
    // > 131 072 edges errored with the same "would reserve 0" symptom as
    // the 6 operator-level paths — and it ALSO clipped budgeted-tenant
    // PROFILE early (a `has_cap`-blindness inconsistency vs the ops fix).
    // Resolve it the SAME way the ops do: an uncapped budget means "no
    // memory limit," so the Vec grows with the actual result cardinality,
    // guarded against a true runaway by `UNCAPPED_RUNAWAY_GUARD_ROWS`;
    // when a per-tenant byte cap IS configured the operator-layer byte
    // budget enforces it (each row is debited there) and this loop
    // imposes no row-count clip.
    let has_cap = ctx.budget().has_cap(ctx.tenant());
    let mut all_rows: Vec<Vec<Value>> = Vec::new();
    loop {
        let batch = op.next_batch(ctx, substrate)?;
        if batch.is_empty() {
            break;
        }
        for row in batch.into_rows() {
            if !has_cap && all_rows.len() >= UNCAPPED_RUNAWAY_GUARD_ROWS {
                return Err(result_row_fallback_err(all_rows.len()));
            }
            all_rows.push(row);
        }
    }
    Ok(all_rows)
}

/// **NN-4 (#1384) re-spin, Fix 1** — RAII drain that releases the MERGE
/// serialization guard(s) stashed on the [`ExecutionContext`] when it
/// drops. Bound in [`execute_with_context`] AFTER the guards are acquired
/// so the guards are released on EVERY exit path (`?` early-return, the
/// runaway early-return, the success return, panic-unwind), unblocking the
/// next racer once the winner's create is durable (each op auto-committed
/// inside `next_batch` on this non-D-2 path). Draining (not leaving to
/// context-drop) prevents a long-lived ctx from leaking the lock past the
/// statement.
struct MergeGuardDrain<'c> {
    ctx: &'c ExecutionContext,
}

impl Drop for MergeGuardDrain<'_> {
    fn drop(&mut self) {
        // Dropping the drained `Vec` releases the per-key mutexes; an empty
        // `Vec` (read / keyless plan) is a harmless no-op.
        drop(self.ctx.take_merge_guards());
    }
}

fn result_row_fallback_err(rows: usize) -> ExecutionError {
    ExecutionError::Plan(crate::semantic::error::ArcQLError::ResourceExhausted {
        feature: "execute_with_context result runaway-guard".to_owned(),
        requested_bytes: 0,
        // #980 — lifted runaway-protection ceiling, not the old 131 072
        // valve that broke PROFILE / eager-Vec over large result sets.
        cap_bytes: UNCAPPED_RUNAWAY_GUARD_ROWS as u64,
        projected_bytes: rows as u64 + 1,
        span: crate::error::Span::point(0, 0),
    })
}
