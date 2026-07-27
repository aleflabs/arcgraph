//! W13β M4-81 multi-batch materialization integration tests per
//! ADR-038 amendment-02 §M4.h + amendment-03 §M5↔M4 contract surface +
//! §TIER-1 GAP E rule 4.
//!
//! # Pin set (≥4 integration tests per spawn prompt)
//!
//! 1. **`single_statement_materialization_end_to_end`** —
//!    end-to-end: a multi-batch query (rows > BATCH_ROWS) produces a
//!    `MaterializedResult` with all rows accumulated, no truncation,
//!    and per-query `memory_bytes_high_water` populated.
//! 2. **`per_tenant_memory_budget_enforcement_truncates_with_resource_exhausted`**
//!    — a query whose cumulative row bytes exceed the per-tenant
//!    budget cap returns a partial `MaterializedResult` with
//!    `truncation == Some(ArcQLError::ResourceExhausted)` and rows
//!    capped at the prefix admitted by the cap.
//! 3. **`cancellation_during_materialization_releases_budget_and_lsn`**
//!    — firing the cancellation token mid-loop surfaces
//!    `Cancelled` AND drops the BudgetReservationGuard +
//!    SnapshotLsnGuard via stack-unwind so the per-tenant counter
//!    returns to 0 + the snapshot-LSN slot resets to None.
//! 4. **`snapshot_lsn_released_at_materialization_end`** — the
//!    canonical ADR-038 §2 D-18 rule 4 / amendment-03 §TIER-1 GAP E
//!    rule 4 pin: a successful materialize call leaves the
//!    `ExecutionContext::snapshot_lsn()` slot at `None`.
//!
//! # ADR provenance
//! - **ADR-038 amendment-02 §M4.h** — primary M4-81 cite.
//! - **ADR-038 amendment-03 §TIER-1 GAP E rule 4** — snapshot-LSN
//!   release at query-end / cursor-close.
//! - **ADR-038 amendment-03 §Structural-1** — per-tenant memory
//!   budget enforcement (W12α surface this slice consumes).
//! - **`feedback_seqlock_panic_safety_primitive.md`** — RAII guard
//!   discipline for resource cleanup (W12γ MED-3 RegistryGuard
//!   pattern is the sister surface).

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use arcgraph_core::{LabelId, Lsn, NodeId, TenantId, TypeId};
use arcgraph_query::executor::ExecutionError;
use arcgraph_query::executor::value::NodeView;
use arcgraph_query::executor::{
    BATCH_ROWS, BoundEdge, BoundNode, CancellationToken, ExecutionContext, ExecutorSubstrate,
    MemoryBudget, RankedHit, StubExecutorSubstrate, SubstrateAccessError, Value,
    execute_with_context,
};
use arcgraph_query::logical_plan::{Direction, LogicalPlanLoweringVisitor};
use arcgraph_query::semantic::error::ArcQLError;
use arcgraph_query::semantic::{
    BindingVisitor, CatalogProvider, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};
use arcgraph_query::{materialize, parse};

// ---------------------------------------------------------------------
// Fixtures — shared with cancel_integration.rs / materialize_integration.rs
// ---------------------------------------------------------------------

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
                .with_property("name", Value::String(format!("p{i}")))
                .with_property("age", Value::Integer(i as i64 * 5)),
        );
    }
    s
}

/// Compile `query` into a [`crate::logical_plan::LogicalPlan`] for the
/// tests that drive [`materialize`] directly (without
/// `QueryEngine::execute` which adds its own LSN-guard via
/// `ExecutionContext::with_query_id`). Mirror of the
/// `QueryEngine::plan_for_execute` helper without join-ordering — the
/// integration scope below doesn't exercise the cost optimizer.
fn lower_to_plan(
    query: &str,
    catalog: &StubCatalogProvider,
) -> arcgraph_query::logical_plan::LogicalPlan {
    let stmt = parse(query).expect("parse");
    let mut bound = BindingVisitor::bind(&stmt, query, catalog).expect("bind");
    TypeCheckVisitor::check(&mut bound, catalog).expect("type-check");
    CrossSubstrateValidator::validate(&bound, catalog).expect("cross-substrate");
    LogicalPlanLoweringVisitor::lower(&bound).expect("lower")
}

// =====================================================================
// Pin 1 — End-to-end multi-batch materialization
// =====================================================================

#[test]
fn single_statement_materialization_end_to_end() {
    // Multi-batch path pin: rows > BATCH_ROWS forces the executor's
    // batch loop to walk multiple iterations. The materialize loop
    // accumulates all rows into a single MaterializedResult.
    let n = (BATCH_ROWS as u64 * 2) + 13; // 4109 rows → 3 batches
    let s = substrate_with_n_persons(n);
    let cat = cat_basic();
    let plan = lower_to_plan("MATCH (n:Person) RETURN n", &cat);
    let ctx = ExecutionContext::new(cat.tenant(), cat.partition());
    let result = materialize::materialize(&plan, &s, &ctx).expect("materialize");
    assert_eq!(
        result.len() as u64,
        n,
        "all rows materialized across multi-batch executor loop"
    );
    assert_eq!(result.metrics.rows_emitted, n, "rows_emitted matches");
    assert!(
        result.metrics.memory_bytes_high_water > 0,
        "non-empty result must report non-zero per-query bytes"
    );
    assert!(
        !result.is_truncated(),
        "complete materialization → no truncation"
    );
    // Snapshot LSN released post-materialize (RAII guard dropped).
    assert!(
        ctx.snapshot_lsn().is_none(),
        "post-materialize: snapshot LSN slot released"
    );
}

// =====================================================================
// Pin 2 — Per-tenant memory-budget enforcement
// =====================================================================

#[test]
fn per_tenant_memory_budget_enforcement_truncates_with_resource_exhausted() {
    // Configure a TINY per-tenant budget cap (1024 bytes). Each
    // person row carries Value::Node{id, label, properties: {age,
    // name}} which `estimate_row_bytes` reports at ~ a few hundred
    // bytes per row (depends on platform). The materialize loop hits
    // the cap mid-stream, returns a PARTIAL MaterializedResult with
    // truncation populated.
    let n = 50; // far more than fits in 1024 bytes
    let s = substrate_with_n_persons(n);
    let cat = cat_basic();
    let plan = lower_to_plan("MATCH (n:Person) RETURN n", &cat);
    // Configure a strict per-tenant budget.
    let budget = MemoryBudget::with_per_tenant_cap(cat.tenant(), 1024);
    let ctx = ExecutionContext::new(cat.tenant(), cat.partition()).with_budget(budget.clone());
    let result = materialize::materialize(&plan, &s, &ctx).expect("partial-result is Ok");
    // Truncation populated:
    assert!(
        result.is_truncated(),
        "budget exhausted → truncation populated"
    );
    let trunc = result.truncation().expect("Some(arcql)");
    assert!(
        matches!(trunc, ArcQLError::ResourceExhausted { .. }),
        "truncation carries ResourceExhausted: {trunc:?}"
    );
    // Partial rows: 0 ≤ rows < n.
    assert!(
        result.len() < n as usize,
        "rows < total ({} < {n})",
        result.len()
    );
    // No-leak: the BudgetReservationGuard's Drop ran post-materialize,
    // releasing the per-tenant counter back to 0.
    assert_eq!(
        budget.current_bytes(cat.tenant()),
        0,
        "BudgetReservationGuard.drop released the per-query bytes"
    );
    // Snapshot-LSN slot released too (RAII guard dropped).
    assert!(
        ctx.snapshot_lsn().is_none(),
        "LSN released on partial-result path"
    );
}

// =====================================================================
// Pin 3 — Cancellation-during-materialization releases budget + LSN
// =====================================================================

/// Slow-substrate adapter used to give the cancellation token time to
/// fire during a scan_nodes call. Borrowed from cancel_integration.rs
/// pattern.
struct SlowSubstrate {
    base: StubExecutorSubstrate,
    per_call_ms: u64,
    calls_seen: Arc<AtomicU64>,
}

impl SlowSubstrate {
    fn new(base: StubExecutorSubstrate, per_call_ms: u64) -> Self {
        Self {
            base,
            per_call_ms,
            calls_seen: Arc::new(AtomicU64::new(0)),
        }
    }
}

impl ExecutorSubstrate for SlowSubstrate {
    fn scan_nodes(
        &self,
        tenant: TenantId,
        label: Option<LabelId>,
        read_lsn: Lsn,
    ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
        thread::sleep(Duration::from_millis(self.per_call_ms));
        self.calls_seen.fetch_add(1, Ordering::AcqRel);
        self.base.scan_nodes(tenant, label, read_lsn)
    }

    fn expand(
        &self,
        tenant: TenantId,
        from: NodeId,
        rel_type: Option<TypeId>,
        direction: Direction,
        read_lsn: Lsn,
    ) -> Result<Vec<BoundEdge>, SubstrateAccessError> {
        self.base
            .expand(tenant, from, rel_type, direction, read_lsn)
    }

    fn vector_search(
        &self,
        tenant: TenantId,
        property: &str,
        query_vec: &[f32],
        k: u64,
        read_lsn: Lsn,
    ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
        self.base
            .vector_search(tenant, property, query_vec, k, read_lsn)
    }

    fn bm25_search(
        &self,
        tenant: TenantId,
        property: &str,
        query_text: &str,
        k: u64,
        read_lsn: Lsn,
    ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
        self.base
            .bm25_search(tenant, property, query_text, k, read_lsn)
    }

    fn community_members(
        &self,
        tenant: TenantId,
        community_id: i64,
        read_lsn: Lsn,
    ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
        self.base.community_members(tenant, community_id, read_lsn)
    }
}

#[test]
fn cancellation_during_materialization_releases_budget_and_lsn() {
    // Slow substrate sleeps 200ms inside scan_nodes. A sibling thread
    // fires the cancellation token after 30ms; the substrate sleep is
    // uninterruptible, but the next batch-boundary check observes the
    // tripped token and surfaces ExecutionError::Cancelled. The
    // BudgetReservationGuard + SnapshotLsnGuard drop on stack unwind,
    // releasing the per-tenant counter + clearing the LSN slot.
    let base = substrate_with_n_persons(BATCH_ROWS as u64 * 5);
    let slow = SlowSubstrate::new(base, 200);
    let cat = cat_basic();
    let plan = lower_to_plan("MATCH (n:Person) RETURN n", &cat);
    // Configured (but unbounded) budget so we can read its counter.
    let budget = MemoryBudget::new();
    let token = CancellationToken::new();
    let ctx = ExecutionContext::new(cat.tenant(), cat.partition())
        .with_budget(budget.clone())
        .with_cancellation(token.clone());

    // Sibling thread fires the cancellation 30ms in.
    let token_canceller = token.clone();
    let canceller = thread::spawn(move || {
        thread::sleep(Duration::from_millis(30));
        token_canceller.cancel();
    });

    let start = Instant::now();
    let result = materialize::materialize(&plan, &slow, &ctx);
    let elapsed = start.elapsed();
    canceller.join().expect("canceller thread");

    match result {
        Err(ExecutionError::Cancelled) => {
            // Per amendment-03 §M5↔M4 contract: cancellation surfaces
            // via the dedicated variant (NOT NotImplemented blanket).
        }
        Err(other) => panic!("expected ExecutionError::Cancelled, got {other:?} after {elapsed:?}"),
        Ok(r) => panic!(
            "cancellation must surface as Err, not Ok({}) rows after {elapsed:?}",
            r.len()
        ),
    }

    // No-leak invariant: BudgetReservationGuard.drop ran on stack
    // unwind, releasing any bytes accumulated pre-cancel.
    assert_eq!(
        budget.current_bytes(cat.tenant()),
        0,
        "post-cancel: budget counter released"
    );
    // SnapshotLsnGuard.drop ran on stack unwind, releasing the LSN.
    assert!(
        ctx.snapshot_lsn().is_none(),
        "post-cancel: snapshot-LSN slot released"
    );
    // Sanity: the cancel must observe AFTER the substrate's 200ms
    // sleep completes (at ~200ms); the upper bound is generous.
    assert!(
        elapsed >= Duration::from_millis(150),
        "cancelled too early ({elapsed:?})"
    );
}

// =====================================================================
// Pin 4 — Snapshot-LSN release at materialization-end (canonical D-18 rule 4)
// =====================================================================

#[test]
fn snapshot_lsn_released_at_materialization_end() {
    // ADR-038 §2 D-18 rule 4 + amendment-03 §TIER-1 GAP E rule 4
    // canonical pin: a successful materialize call leaves the
    // ExecutionContext::snapshot_lsn() slot at None. The
    // SnapshotLsnGuard's Drop is the load-bearing cleanup.
    let s = substrate_with_n_persons(7);
    let cat = cat_basic();
    let plan = lower_to_plan("MATCH (n:Person) RETURN n", &cat);
    let ctx = ExecutionContext::new(cat.tenant(), cat.partition());
    // Pre-materialize: LSN slot is None.
    assert!(ctx.snapshot_lsn().is_none(), "pre-materialize: no LSN");
    let _ = materialize::materialize(&plan, &s, &ctx).expect("materialize");
    // Post-materialize: the SnapshotLsnGuard's Drop ran, clearing the
    // slot. This is the canonical contract pin.
    assert!(
        ctx.snapshot_lsn().is_none(),
        "post-materialize: SnapshotLsnGuard.drop released the LSN"
    );
}

// =====================================================================
// Bonus pin — empty-substrate materialization releases LSN + 0 bytes
// =====================================================================

#[test]
fn empty_substrate_materialization_releases_resources() {
    // The empty-substrate path also runs the materialize loop (one
    // batch returning empty); the guards' Drop fires regardless.
    let s = StubExecutorSubstrate::new();
    let cat = cat_basic();
    let plan = lower_to_plan("MATCH (n:Person) RETURN n", &cat);
    let budget = MemoryBudget::new();
    let ctx = ExecutionContext::new(cat.tenant(), cat.partition()).with_budget(budget.clone());
    let result = materialize::materialize(&plan, &s, &ctx).expect("materialize");
    assert!(result.is_empty());
    assert_eq!(result.metrics.memory_bytes_high_water, 0);
    assert!(!result.is_truncated());
    assert!(ctx.snapshot_lsn().is_none());
    assert_eq!(budget.current_bytes(cat.tenant()), 0);
}

// =====================================================================
// Bonus pin — execute_with_context (non-materialize path) does NOT
// release the LSN by itself; only `materialize` carries the guard.
// Pinned so a future change doesn't accidentally couple the two paths.
// =====================================================================

#[test]
fn execute_with_context_does_not_release_lsn_at_function_end() {
    // execute_with_context is the lower-level primitive that
    // materialize() composes; it returns Vec<Vec<Value>> and does
    // NOT carry the M4-81 LSN guard. Tests + future consumers calling
    // this path directly (e.g., the M4-72 replan post-execute walk)
    // observe the LSN slot still populated until they choose to
    // release. Pinned to prevent accidental coupling.
    let s = substrate_with_n_persons(3);
    let cat = cat_basic();
    let plan = lower_to_plan("MATCH (n:Person) RETURN n", &cat);
    let ctx = ExecutionContext::new(cat.tenant(), cat.partition());
    let _rows = execute_with_context(&plan, &s, &ctx).expect("execute_with_context");
    // The LSN was acquired by the executor's first next_batch call;
    // execute_with_context does NOT release it.
    assert!(
        ctx.snapshot_lsn().is_some(),
        "execute_with_context leaves LSN captured (release is materialize's responsibility)"
    );
    // Manual release per the new API contract.
    ctx.release_snapshot_lsn();
    assert!(ctx.snapshot_lsn().is_none());
}

// =====================================================================
// W13β fix-up M-1 — materialize REJECTS close-then-reopen at the API
// =====================================================================

#[test]
fn materialize_rejects_consumed_context_per_rule_5() {
    // PR #287 review M-1: ADR-038 amendment-03 §TIER-1 GAP E rule 5
    // ("All operators in a single ExecutionContext share the same
    // snapshot LSN; replan does NOT re-acquire") forbids re-acquiring
    // a fresh LSN on a context whose snapshot LSN was previously
    // released. `materialize::materialize` consults the
    // [`ExecutionContext::lsn_consumed`] latch and rejects with
    // ArcQLError::Internal rather than silently re-acquiring.
    let s = substrate_with_n_persons(5);
    let cat = cat_basic();
    let plan = lower_to_plan("MATCH (n:Person) RETURN n", &cat);
    let ctx = ExecutionContext::new(cat.tenant(), cat.partition());
    // First materialize call — succeeds; the SnapshotLsnGuard's Drop
    // releases the LSN AND lights the consumption latch.
    let _ = materialize::materialize(&plan, &s, &ctx).expect("materialize #1 succeeds");
    assert!(
        ctx.lsn_consumed(),
        "post-materialize: lsn_consumed latch lit"
    );
    // Second materialize on the SAME ctx — REJECTS with Internal
    // (rather than re-acquiring a fresh LSN per rule 5).
    let result = materialize::materialize(&plan, &s, &ctx);
    match result {
        Err(ExecutionError::Plan(ArcQLError::Internal {
            feature, reason, ..
        })) => {
            assert_eq!(feature, "materialize::materialize");
            assert!(
                reason.contains("rule 5"),
                "rejection cites rule 5; got: {reason}"
            );
        }
        other => panic!("expected ArcQLError::Internal on consumed ctx, got {other:?}"),
    }
}

#[test]
fn materialize_then_cursor_open_on_same_ctx_rejects() {
    // Cross-surface consistency pin (M-1): once one materialize call
    // consumes a context, a sibling `StreamingCursor::open` on the
    // same ctx (or a clone) ALSO rejects — both surfaces share the
    // single-shot context invariant. Future M5-tier renderers can
    // bucket "client misuse: ExecutionContext re-used after release"
    // uniformly across both shapes.
    use arcgraph_query::StreamingCursor;
    let s = substrate_with_n_persons(2);
    let cat = cat_basic();
    let plan = lower_to_plan("MATCH (n:Person) RETURN n", &cat);
    let ctx = ExecutionContext::new(cat.tenant(), cat.partition());
    let ctx_clone = ctx.clone();
    let _ = materialize::materialize(&plan, &s, &ctx).expect("materialize #1");
    // Cursor on the clone (same Arc-shared latch) — REJECTS.
    // (StreamingCursor is not Debug — match on the err half only.)
    match StreamingCursor::open(&plan, ctx_clone, &s) {
        Ok(_) => panic!("expected Internal on consumed-ctx cursor open, got Ok"),
        Err(ExecutionError::Plan(ArcQLError::Internal { feature, .. })) => {
            assert_eq!(feature, "StreamingCursor::open");
        }
        Err(other) => panic!("expected Internal, got {other:?}"),
    }
}
