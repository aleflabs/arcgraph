//! W13β wave-level cross-PR transit pin per
//! `feedback_anchor_to_consumer_transit_pinning.md` ("Producer-
//! consumer pairs in same wave need ≥1 end-to-end transit pin").
//!
//! # Pin scope
//!
//! W13β fuses three previously-shipped surfaces into one pipeline:
//! 1. **W12β M4-71** [`crate::observer::RowCountObserver`] —
//!    per-operator row-count + wall-time + memory metrics.
//! 2. **W12γ M4-08a** [`arcgraph_query::MaterializedResult`] —
//!    the eager Vec-form return type (W13β extended with the
//!    `truncation` field + memory-budget enforcement).
//! 3. **W13β M4-82** [`arcgraph_query::StreamingCursor`] —
//!    streaming yield-batch surface.
//!
//! The transit pin asserts the contract surface holds end-to-end:
//! a streaming cursor with an attached observer accumulates the
//! same row counts the materialize tail would produce; the
//! observer's `execution_metrics()` reflects the per-batch
//! observations across the cursor's `next_batch` walk.
//!
//! # Why this pin matters (per `feedback_anchor_to_consumer_transit_pinning.md`)
//!
//! Producer-consumer pairs that ship same-wave can drift on contract
//! shape if no end-to-end test exercises them together. M4-71
//! observer wires into the executor's `next_batch` dispatcher; M4-82
//! cursor calls the same dispatcher. A future change to the cursor
//! that bypasses the dispatcher (e.g., direct ScanOp call) would
//! silently disable the observer hook — this pin catches it.
//!
//! # Pin set
//!
//! 1. **`m4_71_observer_records_metrics_through_streaming_cursor`** —
//!    a cursor with attached observer accumulates per-operator
//!    metrics across multi-batch streaming. The observer's
//!    `execution_metrics()` reflects cumulative row counts at EOS.
//! 2. **`materialized_result_metrics_match_cursor_observer_metrics`** —
//!    the same query, run through both `materialize()` AND
//!    `StreamingCursor`, produces equivalent row counts. The
//!    M4-08a Vec form and the M4-82 streaming form converge on
//!    the same observable result-cardinality.
//! 3. **`m4_72_replan_via_observer_after_cursor_close`** — the
//!    canonical "cursor close → ReplanController" handoff per the
//!    cursor's module docs §"M4-72 replan ↔ cursor handoff": the
//!    caller closes the cursor, runs `ReplanController::replan()`,
//!    and observes the replan path is reachable + the observer's
//!    breaches are populated.
//! 4. **`m4_71_observer_plus_materialized_result_plus_cursor_one_pipeline`**
//!    (W13β fix-up N-3) — a single pipeline that exercises ALL
//!    three W13β surfaces together: observer + materialize +
//!    cursor. The "one pipeline" interpretation per the W13β spawn
//!    brief; the previous Pins 1–3 each cover 2-of-3 surfaces.
//!
//! # ADR provenance
//! - **ADR-038 amendment-02 §M4.h** — primary M4-82 cite.
//! - **ADR-038 amendment-02 §M4.g** — M4-71 observer scope.
//! - **`feedback_anchor_to_consumer_transit_pinning.md`** —
//!   producer-consumer end-to-end transit pin discipline.

use std::sync::Arc;

use arcgraph_core::{LabelId, NodeId, TenantId};
use arcgraph_query::executor::value::NodeView;
use arcgraph_query::executor::{ExecutionContext, StubExecutorSubstrate, Value};
use arcgraph_query::logical_plan::LogicalPlanLoweringVisitor;
use arcgraph_query::observer::{ReplanController, RowCountObserver};
use arcgraph_query::planner::cost::estimate_costs;
use arcgraph_query::planner::enumeration::enumerate_join_order;
use arcgraph_query::semantic::{
    BindingVisitor, CatalogProvider, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};
use arcgraph_query::{StreamingCursor, materialize, parse};

fn cat_basic() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["Person"])
        .with_rel_types(["KNOWS"])
        .with_properties(["name", "age"])
        .with_label_cardinality(LabelId::new(1), 100)
        .with_total_node_count(100)
}

fn substrate_with_n_persons(n: u64) -> StubExecutorSubstrate {
    let mut s = StubExecutorSubstrate::new();
    for i in 1..=n {
        s = s.with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(i), Some(LabelId::new(1)))
                .with_property("age", Value::Integer(i as i64)),
        );
    }
    s
}

// =====================================================================
// Pin 1 — M4-71 observer records metrics through M4-82 streaming cursor
// =====================================================================

#[test]
fn m4_71_observer_records_metrics_through_streaming_cursor() {
    // Build the costed plan so the observer can be initialized with
    // estimated cardinalities (per W12β
    // RowCountObserver::from_plan_and_costs convention).
    let cat = cat_basic();
    let n: u64 = 5_000; // > BATCH_ROWS so multi-batch streaming.
    let s = substrate_with_n_persons(n);
    let stmt = parse("MATCH (n:Person) RETURN n").expect("parse");
    let mut bound = BindingVisitor::bind(&stmt, "MATCH (n:Person) RETURN n", &cat).expect("bind");
    TypeCheckVisitor::check(&mut bound, &cat).expect("type-check");
    CrossSubstrateValidator::validate(&bound, &cat).expect("cross-substrate");
    let plan = LogicalPlanLoweringVisitor::lower(&bound).expect("lower");
    let optimized = enumerate_join_order(plan, &cat);
    let costed = estimate_costs(optimized, &cat);
    // Construct observer + attach to context.
    let observer = Arc::new(RowCountObserver::from_plan_and_costs(
        costed.plan(),
        costed.costs(),
    ));
    let ctx =
        ExecutionContext::new(cat.tenant(), cat.partition()).with_observer(Arc::clone(&observer));
    // Open the cursor on the SAME costed plan the observer was built
    // around.
    let mut cursor = StreamingCursor::open(costed.plan(), ctx, &s).expect("open");
    let mut total: u64 = 0;
    while let Some(rows) = cursor.next_batch().expect("next_batch") {
        total += rows.len() as u64;
    }
    assert_eq!(total, n, "cursor streamed all rows");
    assert_eq!(cursor.rows_emitted(), n);
    // Observer recorded per-operator metrics via the dispatcher hook.
    let metrics = observer.execution_metrics();
    assert_eq!(
        metrics.rows_emitted, n,
        "observer's execution_metrics().rows_emitted == cursor.rows_emitted()"
    );
    assert!(
        metrics.wall_time_ms > 0 || total < 100,
        "wall_time_ms populated for non-trivial query (got {}ms across {n} rows)",
        metrics.wall_time_ms
    );
}

// =====================================================================
// Pin 2 — MaterializedResult metrics match cursor + observer metrics
// =====================================================================

#[test]
fn materialized_result_metrics_match_cursor_observer_metrics() {
    // Same query, two paths:
    //   (a) materialize() — eager Vec form (W12γ M4-08a).
    //   (b) StreamingCursor — streaming form (W13β M4-82).
    // The cumulative row counts must agree. This is the canonical
    // "M4-08a vs M4-08b convergence" pin.
    let cat = cat_basic();
    let n: u64 = 4_500;
    let s = substrate_with_n_persons(n);
    let stmt = parse("MATCH (n:Person) RETURN n").expect("parse");
    let mut bound = BindingVisitor::bind(&stmt, "MATCH (n:Person) RETURN n", &cat).expect("bind");
    TypeCheckVisitor::check(&mut bound, &cat).expect("type-check");
    CrossSubstrateValidator::validate(&bound, &cat).expect("cross-substrate");
    let plan = LogicalPlanLoweringVisitor::lower(&bound).expect("lower");
    // Path (a): materialize — eager Vec.
    let ctx_mat = ExecutionContext::new(cat.tenant(), cat.partition());
    let mat_result = materialize::materialize(&plan, &s, &ctx_mat).expect("materialize");
    assert_eq!(mat_result.metrics.rows_emitted, n);
    assert_eq!(mat_result.len() as u64, n);
    assert!(!mat_result.is_truncated());
    // Path (b): StreamingCursor — streaming.
    let ctx_cur = ExecutionContext::new(cat.tenant(), cat.partition());
    let mut cursor = StreamingCursor::open(&plan, ctx_cur, &s).expect("open");
    let mut cur_total: u64 = 0;
    while let Some(rows) = cursor.next_batch().expect("next_batch") {
        cur_total += rows.len() as u64;
    }
    // Convergence: both paths see the same row count.
    assert_eq!(
        cur_total, mat_result.metrics.rows_emitted,
        "M4-08a Vec form and M4-08b cursor form agree on row count"
    );
    assert_eq!(cursor.rows_emitted(), n);
}

// =====================================================================
// Pin 3 — Cursor close → ReplanController handoff (M4-72 wiring)
// =====================================================================

#[test]
fn m4_72_replan_via_observer_after_cursor_close() {
    // Per the cursor module docs §"M4-72 replan ↔ cursor handoff",
    // mid-stream replan is forward-deferred to v1.1; the v1.0-alpha
    // contract is "close → replan → reopen". This pin walks the
    // canonical handoff: the observer accumulates breaches during
    // streaming; after `cursor.close()`, the
    // ReplanController::replan() is reachable + the observer's
    // breaches are populated.
    let cat = cat_basic();
    let n: u64 = 1_500;
    let s = substrate_with_n_persons(n);
    let stmt = parse("MATCH (n:Person) RETURN n").expect("parse");
    let mut bound = BindingVisitor::bind(&stmt, "MATCH (n:Person) RETURN n", &cat).expect("bind");
    TypeCheckVisitor::check(&mut bound, &cat).expect("type-check");
    CrossSubstrateValidator::validate(&bound, &cat).expect("cross-substrate");
    let bound_arc = Arc::new(bound);
    let plan = LogicalPlanLoweringVisitor::lower(&bound_arc).expect("lower");
    let optimized = enumerate_join_order(plan, &cat);
    let costed = Arc::new(estimate_costs(optimized, &cat));
    // Observer with intentionally-LOW estimated cardinality so the
    // observed rows trip the 10× threshold (catalog reports 100 nodes
    // total; substrate has 1500). That triggers an
    // UnderEstimate breach via the W12β observer's threshold logic.
    let observer = Arc::new(RowCountObserver::from_plan_and_costs(
        costed.plan(),
        costed.costs(),
    ));
    let ctx =
        ExecutionContext::new(cat.tenant(), cat.partition()).with_observer(Arc::clone(&observer));
    // Stream through the cursor.
    let mut cursor = StreamingCursor::open(costed.plan(), ctx, &s).expect("open");
    while let Some(_rows) = cursor.next_batch().expect("next_batch") {
        // Drain.
    }
    // Close (idempotent vs the EOS-auto-close path; explicit call
    // here is the canonical caller-owned sequence).
    cursor.close().expect("close");
    // Build the replan controller, attached to the SAME observer.
    let controller = ReplanController::new(
        &cat,
        None, // no plan cache attached
        Arc::clone(&observer),
        bound_arc,
        Arc::clone(&costed),
        None, // no original cache key
    );
    // Observer's threshold breaches should be populated (the 1500
    // observed rows vs the 100-row catalog estimate trips the 10×
    // threshold).
    let breaches = observer.threshold_breaches();
    assert!(
        !breaches.is_empty(),
        "observer accumulated threshold breaches across cursor stream"
    );
    // Replan reachable: the call returns Ok(_) — the controller can
    // walk its full pipeline. Whether the new plan diverges from the
    // original depends on the synthetic catalog shape; the load-bearing
    // pin is "the handoff API is reachable + replan() doesn't panic".
    let outcome = controller.replan().expect("replan");
    let _ = outcome; // outcome.is_some() depends on plan-shape divergence
    assert_eq!(
        controller.replan_count(),
        1,
        "replan attempt count incremented"
    );
}

// =====================================================================
// Pin 4 (W13β fix-up N-3) — observer + materialize + cursor in ONE pipeline
// =====================================================================

#[test]
fn m4_71_observer_plus_materialized_result_plus_cursor_one_pipeline() {
    // Per PR #287 review N-3: "Each pin covers 2-of-3 surfaces; none
    // covers 3-of-3. Defensible split (each pin is focused), but
    // doesn't match the brief's literal 'one pipeline' phrasing."
    // This pin exercises ALL THREE W13β surfaces together:
    //
    //   1. M4-71 RowCountObserver attached to ctx_mat — populates
    //      `observer_mat.execution_metrics()` during materialize.
    //   2. M4-08a `materialize::materialize` — produces the
    //      MaterializedResult with `metrics.rows_emitted` and
    //      memory_bytes_high_water populated end-to-end.
    //   3. M4-82 StreamingCursor on a SIBLING ExecutionContext (per
    //      W13β fix-up M-1: a single ctx is single-shot — close-
    //      then-reopen rejects), with its OWN observer attached, that
    //      streams the same query end-to-end.
    //
    // Convergence: BOTH observers + the materialize metrics + the
    // cursor's rows_emitted accumulator all agree on the row count.
    // The pin ensures the contract surface stays uniform across the
    // three return shapes / observation hooks.
    let cat = cat_basic();
    let n: u64 = 4_096; // > BATCH_ROWS so cursor walks ≥ 2 batches.
    let s = substrate_with_n_persons(n);
    let stmt = parse("MATCH (n:Person) RETURN n").expect("parse");
    let mut bound = BindingVisitor::bind(&stmt, "MATCH (n:Person) RETURN n", &cat).expect("bind");
    TypeCheckVisitor::check(&mut bound, &cat).expect("type-check");
    CrossSubstrateValidator::validate(&bound, &cat).expect("cross-substrate");
    let plan = LogicalPlanLoweringVisitor::lower(&bound).expect("lower");
    let optimized = enumerate_join_order(plan, &cat);
    let costed = estimate_costs(optimized, &cat);

    // ---- Surface 1+2: materialize with observer attached ----
    let observer_mat = Arc::new(RowCountObserver::from_plan_and_costs(
        costed.plan(),
        costed.costs(),
    ));
    let ctx_mat = ExecutionContext::new(cat.tenant(), cat.partition())
        .with_observer(Arc::clone(&observer_mat));
    let mat_result =
        arcgraph_query::materialize::materialize(costed.plan(), &s, &ctx_mat).expect("materialize");
    assert_eq!(mat_result.metrics.rows_emitted, n);
    assert_eq!(mat_result.len() as u64, n);
    assert!(!mat_result.is_truncated());
    let observer_mat_metrics = observer_mat.execution_metrics();
    assert_eq!(
        observer_mat_metrics.rows_emitted, n,
        "observer (attached during materialize) recorded {n} rows"
    );

    // ---- Surface 3: cursor with its OWN observer on a fresh ctx ----
    // Per W13β fix-up M-1, ctx_mat is consumed; we use a fresh
    // ExecutionContext for the cursor's leg of the pipeline.
    let observer_cursor = Arc::new(RowCountObserver::from_plan_and_costs(
        costed.plan(),
        costed.costs(),
    ));
    let ctx_cursor = ExecutionContext::new(cat.tenant(), cat.partition())
        .with_observer(Arc::clone(&observer_cursor));
    let mut cursor =
        arcgraph_query::StreamingCursor::open(costed.plan(), ctx_cursor, &s).expect("open");
    let mut cur_total: u64 = 0;
    while let Some(rows) = cursor.next_batch().expect("next_batch") {
        cur_total += rows.len() as u64;
    }
    assert_eq!(cur_total, n, "cursor streamed all rows");
    assert_eq!(cursor.rows_emitted(), n);
    let observer_cursor_metrics = observer_cursor.execution_metrics();
    assert_eq!(
        observer_cursor_metrics.rows_emitted, n,
        "observer (attached during cursor) recorded {n} rows"
    );

    // ---- Convergence: all four observation paths agree ----
    assert_eq!(observer_mat_metrics.rows_emitted, n);
    assert_eq!(observer_cursor_metrics.rows_emitted, n);
    assert_eq!(mat_result.metrics.rows_emitted, n);
    assert_eq!(cursor.rows_emitted(), n);
}
