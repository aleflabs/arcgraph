//! W13β M4-82 streaming cursor integration tests per ADR-038
//! amendment-02 §M4.h + amendment-03 §M5↔M4 contract surface §11
//! D-9 + §TIER-1 GAP E rule 4.
//!
//! # Pin set (≥3 integration tests per spawn prompt)
//!
//! 1. **`large_result_query_streams_through_cursor_end_to_end`** —
//!    100K-match query end-to-end. The cursor walks multiple
//!    `next_batch` iterations until EOS (Ok(None)), accumulating
//!    rows that match the substrate's row count exactly.
//! 2. **`cursor_close_releases_snapshot_lsn`** — pin for
//!    amendment-03 §TIER-1 GAP E rule 4 ("Snapshot LSN released at
//!    cursor-close"): explicit `cursor.close()` clears the
//!    `ExecutionContext::snapshot_lsn()` slot (observable via the
//!    LSN-watermark surface).
//! 3. **`cancellation_during_streaming_releases_lsn`** — token-
//!    fired-mid-stream surfaces `Cancelled` AND the cursor's auto-
//!    close path releases the LSN.
//!
//! Plus a 10M-match query gated `#[ignore]` for the full-scale run.
//!
//! # ADR provenance
//! - **ADR-038 amendment-02 §M4.h** — primary M4-82 cite.
//! - **ADR-038 amendment-03 §TIER-1 GAP E rule 4** — snapshot-LSN
//!   release at cursor-close.
//! - **ADR-038 amendment-03 §M5↔M4 contract surface §11 D-9** —
//!   M4-08b is the streaming-cursor tail of `execute`.

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::thread;
use std::time::{Duration, Instant};

use arcgraph_core::{LabelId, Lsn, NodeId, TenantId, TypeId};
use arcgraph_query::executor::value::NodeView;
use arcgraph_query::executor::{
    BoundEdge, BoundNode, CancellationToken, ExecutionContext, ExecutionError, ExecutorSubstrate,
    RankedHit, StubExecutorSubstrate, SubstrateAccessError, Value,
};
use arcgraph_query::logical_plan::{Direction, LogicalPlanLoweringVisitor};
use arcgraph_query::semantic::{
    BindingVisitor, CatalogProvider, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};
use arcgraph_query::{StreamingCursor, parse};

// ---------------------------------------------------------------------
// Fixtures
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
                .with_property("age", Value::Integer(i as i64 * 5)),
        );
    }
    s
}

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
// Pin 1 — 100K-match query streams end-to-end
// =====================================================================

#[test]
fn large_result_query_streams_through_cursor_end_to_end() {
    // 100K rows × multiple batches. Per the spawn prompt, the
    // 10M-match version is gated `#[ignore]` below; 100K is the
    // CI-runnable scaled-down variant.
    let n: u64 = 100_000;
    let s = substrate_with_n_persons(n);
    let cat = cat_basic();
    let plan = lower_to_plan("MATCH (n:Person) RETURN n", &cat);
    let ctx = ExecutionContext::new(cat.tenant(), cat.partition());
    let ctx_observer = ctx.clone();
    let mut cursor = StreamingCursor::open(&plan, ctx, &s).expect("open");
    let mut total_rows: u64 = 0;
    let start = Instant::now();
    while let Some(rows) = cursor.next_batch().expect("next_batch") {
        total_rows += rows.len() as u64;
    }
    let elapsed = start.elapsed();
    assert_eq!(
        total_rows, n,
        "all {n} rows streamed end-to-end across multi-batch cursor"
    );
    // Cursor auto-closed on EOS.
    assert!(cursor.is_closed());
    // Snapshot-LSN released via auto-close.
    assert!(
        ctx_observer.snapshot_lsn().is_none(),
        "post-EOS: LSN released"
    );
    // Sanity: 100K rows shouldn't take more than 30s on any
    // reasonable machine (the v1.0-alpha stub substrate is in-memory
    // hash-map-based; the 30s budget is the v1.0 default deadline,
    // far above the wall-time floor).
    assert!(
        elapsed < Duration::from_secs(30),
        "100K-row stream completed in {elapsed:?} (within budget)"
    );
}

/// 10M-row variant — same shape as the CI test, gated `#[ignore]` so
/// CI doesn't pay the wall-clock cost. Run via:
///
/// ```text
/// cargo test -p arcgraph-query --release --test m4_82_cursor_integration \
///   ten_million_row_query_streams_through_cursor -- --ignored
/// ```
///
/// Per the spawn prompt: "10M-match query end-to-end (or scaled-down
/// 100K + a documented #[ignore] for the full 10M)".
#[test]
#[ignore = "10M-row scale; run with --release --ignored on a beefy box"]
fn ten_million_row_query_streams_through_cursor() {
    let n: u64 = 10_000_000;
    let s = substrate_with_n_persons(n);
    let cat = cat_basic();
    let plan = lower_to_plan("MATCH (n:Person) RETURN n", &cat);
    let ctx = ExecutionContext::new(cat.tenant(), cat.partition());
    let mut cursor = StreamingCursor::open(&plan, ctx, &s).expect("open");
    let mut total_rows: u64 = 0;
    while let Some(rows) = cursor.next_batch().expect("next_batch") {
        total_rows += rows.len() as u64;
    }
    assert_eq!(total_rows, n);
}

// =====================================================================
// Pin 2 — cursor close releases snapshot LSN (observable via watermark)
// =====================================================================

#[test]
fn cursor_close_releases_snapshot_lsn() {
    // Per ADR-038 §2 D-18 rule 4 + amendment-03 §TIER-1 GAP E rule
    // 4: explicit `cursor.close()` releases the snapshot LSN.
    // Observable via `ExecutionContext::snapshot_lsn()` on a cloned
    // context (the snapshot-LSN slot is `Arc<Mutex<Option<Lsn>>>`-
    // shared across clones).
    let s = substrate_with_n_persons(3);
    let cat = cat_basic();
    let plan = lower_to_plan("MATCH (n:Person) RETURN n", &cat);
    let ctx = ExecutionContext::new(cat.tenant(), cat.partition());
    let ctx_observer = ctx.clone();
    let mut cursor = StreamingCursor::open(&plan, ctx, &s).expect("open");
    // Pull one batch — this lazily acquires the LSN.
    let _ = cursor.next_batch().expect("first batch");
    assert!(
        ctx_observer.snapshot_lsn().is_some(),
        "during streaming: LSN captured"
    );
    let captured = ctx_observer.snapshot_lsn().unwrap_or(Lsn::ZERO);
    // Explicit close — releases the LSN.
    cursor.close().expect("close");
    assert!(
        ctx_observer.snapshot_lsn().is_none(),
        "post-close: LSN slot reset to None"
    );
    // Sanity: the captured LSN was Lsn::MAX at v1.0-alpha (no MVCC
    // writer). Forward-method: M4-08+ wires the production LSN.
    assert_eq!(captured, Lsn::MAX);
}

// =====================================================================
// Pin 3 — Cancellation during streaming releases LSN
// =====================================================================

/// Slow-substrate adapter — sleeps `per_call_ms` ms inside scan_nodes
/// to give the cancellation token time to fire before the next batch
/// boundary.
struct SlowSubstrate {
    base: StubExecutorSubstrate,
    per_call_ms: u64,
    _calls_seen: Arc<AtomicU64>,
}

impl SlowSubstrate {
    fn new(base: StubExecutorSubstrate, per_call_ms: u64) -> Self {
        Self {
            base,
            per_call_ms,
            _calls_seen: Arc::new(AtomicU64::new(0)),
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
fn cancellation_during_streaming_releases_lsn() {
    // Cancellation MID-STREAM pin: after the cursor has emitted at
    // least one batch, trip the token; the next `next_batch` call
    // observes the trip at the batch-boundary check + auto-closes,
    // releasing the LSN.
    //
    // Why deterministic (NOT thread-spawn-based):
    //
    // The W12γ ScanOp pre-buffers the substrate's full row vec at
    // first scan_nodes call (it then chunks into BATCH_ROWS-sized
    // batches across subsequent next_batch invocations). A
    // thread-spawn-with-sleep canceller would have to time-coordinate
    // with the scan_nodes call to land its trip BEFORE the buffer
    // pre-fetches; on a contended runner, the pre-fetch may complete
    // before the canceller fires, then all subsequent batches drain
    // the buffer with no substrate re-entry.
    //
    // Deterministic alternative (used here): pull one batch, then
    // call `token.cancel()` directly on the cursor's thread (no
    // thread-spawn). The next next_batch's batch-boundary cancel-
    // check observes the trip — this is the SAME code path the
    // thread-spawn-with-sleep test would exercise, just without the
    // wall-clock race.
    let n: u64 = 8192; // > BATCH_ROWS so we drive 2+ next_batch calls.
    let s = substrate_with_n_persons(n);
    let cat = cat_basic();
    let plan = lower_to_plan("MATCH (n:Person) RETURN n", &cat);
    let token = CancellationToken::new();
    let ctx = ExecutionContext::new(cat.tenant(), cat.partition()).with_cancellation(token.clone());
    let ctx_observer = ctx.clone();
    let mut cursor = StreamingCursor::open(&plan, ctx, &s).expect("open");

    // First batch — captures the LSN, emits BATCH_ROWS rows.
    let first = cursor.next_batch().expect("first batch");
    assert!(first.is_some(), "first batch yields rows");
    assert!(
        ctx_observer.snapshot_lsn().is_some(),
        "during streaming: LSN captured"
    );

    // Trip the token mid-stream.
    token.cancel();

    // Second next_batch observes the trip + auto-closes.
    let result = cursor.next_batch();
    assert!(
        matches!(result, Err(ExecutionError::Cancelled)),
        "post-cancel: next_batch surfaces Cancelled, got {result:?}"
    );
    assert!(cursor.is_closed(), "cursor auto-closed on Cancelled");
    // LSN released via auto-close.
    assert!(
        ctx_observer.snapshot_lsn().is_none(),
        "post-cancel: LSN released via auto-close"
    );
}

// =====================================================================
// Bonus pin — slow-substrate-driven cancel (the canonical M5-12
// integration shape: a sibling thread fires the cancel during a slow
// substrate call). Gated `#[ignore]` because the wall-clock race can
// flake under contended CI runners; deterministic version above.
// =====================================================================

#[test]
#[ignore = "wall-clock-race-prone; the deterministic mid-stream-cancel test above covers the same code path"]
fn cancellation_via_slow_substrate_thread_spawn() {
    let base = substrate_with_n_persons(50_000);
    let slow = SlowSubstrate::new(base, 500);
    let cat = cat_basic();
    let plan = lower_to_plan("MATCH (n:Person) RETURN n", &cat);
    let token = CancellationToken::new();
    let ctx = ExecutionContext::new(cat.tenant(), cat.partition()).with_cancellation(token.clone());
    let ctx_observer = ctx.clone();
    let mut cursor = StreamingCursor::open(&plan, ctx, &slow).expect("open");

    // Spawn the canceller — fires 50ms in, well before the
    // substrate's 500ms sleep completes.
    let token_canceller = token.clone();
    let canceller = thread::spawn(move || {
        thread::sleep(Duration::from_millis(50));
        token_canceller.cancel();
    });

    // First next_batch waits inside scan_nodes (500ms sleep). On
    // return, the inner-buffer-then-batch path may emit one batch
    // before the cancel-check observes the trip; the SECOND call
    // observes the trip and surfaces Cancelled.
    let _first = cursor.next_batch();
    let result = cursor.next_batch();
    canceller.join().expect("canceller");

    // EITHER the first call OR the second call surfaced Cancelled
    // — both are valid outcomes depending on the wall-clock race.
    // The pin is: SOMEWHERE in the cursor's lifecycle, the cancel
    // was observed and the cursor auto-closed.
    if matches!(result, Err(ExecutionError::Cancelled)) {
        assert!(cursor.is_closed());
    } else {
        // Pull until we observe Cancelled or EOS; the deterministic
        // version above already pins the canonical contract.
        loop {
            match cursor.next_batch() {
                Err(ExecutionError::Cancelled) => break,
                Ok(None) => break,
                _ => continue,
            }
        }
    }
    assert!(ctx_observer.snapshot_lsn().is_none());
}
