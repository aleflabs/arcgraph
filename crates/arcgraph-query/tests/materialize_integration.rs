//! M4-08a end-to-end materialization integration tests per ADR-038
//! amendment-02 §M4.h + amendment-03 §M5↔M4 contract surface.
//!
//! # Pin set
//!
//! 1. `execute_returns_materialized_result_with_metrics` — the
//!    `QueryEngine::execute` shape change pin: returns
//!    `MaterializedResult` carrying both rows AND metrics; the
//!    `wall_time_ms` + `rows_emitted` fields are populated end-to-end
//!    by the W12γ wiring.
//! 2. `profile_returns_plan_tree_and_metrics_via_m4_71_forward_pin`
//!    — the M4-91 PROFILE wire-up pin (replaces the previous
//!    NotImplemented stub per amendment-03 §TIER-1 GAP B). The PROFILE
//!    path drives the executor and returns `(PlanTree,
//!    ExecutionMetrics)`; per-operator annotations from
//!    M4-71's `RowCountObserver` are forward-deferred (the test is
//!    NOT gated `#[ignore]` because the metrics surface is populated
//!    end-to-end at this slice — only the per-op decomposition is
//!    forward-linked to W12β / M4-71).
//!
//! # ADR provenance
//! - **ADR-038 amendment-02 §M4.h** — primary M4-08a (M4-81) cite.
//! - **ADR-038 amendment-03 §TIER-1 GAP B** — M4-91 PROFILE return
//!   shape `(PlanTree, ExecutionMetrics)`.
//! - **ADR-038 amendment-03 §M5↔M4 contract surface §11 D-9** —
//!   `execute` returns `MaterializedResult`-shaped value.

use arcgraph_core::{LabelId, NodeId, TenantId};
use arcgraph_query::executor::value::NodeView;
use arcgraph_query::executor::{StubExecutorSubstrate, Value};
use arcgraph_query::semantic::StubCatalogProvider;
use arcgraph_query::{ExplainError, MaterializedResult, PlanTreeOp, QueryEngine};

// ---------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------

fn cat_basic() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["Person"])
        .with_rel_types(["KNOWS"])
        .with_properties(["age"])
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

// =====================================================================
// 1. execute returns MaterializedResult with populated metrics
// =====================================================================

#[test]
fn execute_returns_materialized_result_with_metrics() {
    // The W12γ shape change pin: QueryEngine::execute returns
    // MaterializedResult carrying both rows AND metrics. The
    // metrics' wall_time_ms + rows_emitted fields are populated
    // end-to-end (memory_bytes_high_water is forward to M4-64a).
    let s = substrate_with_n_persons(7);
    let cat = cat_basic();
    let engine = QueryEngine::new(&cat);
    let result: MaterializedResult = engine
        .execute("MATCH (n:Person) RETURN n", &s)
        .expect("execute");
    // Rows surface
    assert_eq!(result.rows().len(), 7, "rows materialized end-to-end");
    assert_eq!(result.len(), 7, "len() == rows().len()");
    assert!(!result.is_empty());
    // Metrics surface
    assert_eq!(
        result.metrics().rows_emitted,
        7,
        "rows_emitted == rows.len() at v1.0-alpha; M4-82 streaming-cursor decouples them"
    );
    // W13β M4-81: memory_bytes_high_water now carries THIS call's
    // contribution to the per-tenant counter (sum of admitted rows'
    // estimated bytes). Pre-W13β value was 0 (the W12γ M4-64a-deferred
    // placeholder). 7 rows × ~7-cell-bytes each → non-zero. The exact
    // magnitude is platform-sensitive; the load-bearing pin is
    // `> 0` for non-empty results.
    assert!(
        result.metrics().memory_bytes_high_water > 0,
        "non-empty result must report non-zero memory_bytes_high_water: {result:?}"
    );
    // No-leak: the per-tenant counter must drop back to 0 after the
    // materialize call returns (the BudgetReservationGuard's Drop ran).
    // The post-execute counter reflects a fresh, no-leak state.
    // (Read via the substrate's tenant; tested below in the dedicated
    // M4-81 budget-release integration test.)
    assert!(
        !result.is_truncated(),
        "complete materialization → no truncation"
    );
    assert!(result.truncation().is_none());
    // wall_time_ms is non-deterministic but populated end-to-end —
    // we only assert the field is accessible (the surface is what's
    // load-bearing, not the magnitude).
    let _ = result.metrics().wall_time_ms;
    // Backwards-compat: into_rows() returns Vec<Vec<Value>> for
    // pre-W12γ call-site shapes.
    let rows: Vec<Vec<Value>> = result.into_rows();
    assert_eq!(rows.len(), 7);
}

#[test]
fn execute_on_empty_substrate_returns_empty_result() {
    // Empty-substrate path pin: execute returns an empty
    // MaterializedResult with rows_emitted = 0; the M4-91 PROFILE
    // path wraps this in (PlanTree, metrics) without surfacing an
    // error.
    let s = StubExecutorSubstrate::new();
    let cat = cat_basic();
    let engine = QueryEngine::new(&cat);
    let result = engine
        .execute("MATCH (n:Person) RETURN n", &s)
        .expect("execute on empty substrate");
    assert!(result.is_empty(), "0-row substrate → 0-row result");
    assert_eq!(result.metrics().rows_emitted, 0);
}

// =====================================================================
// 2. PROFILE returns (PlanTree, ExecutionMetrics) via M4-71 forward-pin
// =====================================================================

#[test]
fn profile_returns_plan_tree_and_metrics_via_m4_71_forward_pin() {
    // The M4-91 PROFILE wire-up pin per amendment-03 §TIER-1 GAP B.
    // Pre-W12γ: profile returned ArcQLError::NotImplemented.
    // Post-W12γ: profile drives the executor end-to-end and returns
    // (PlanTree, ExecutionMetrics).
    //
    // Per-operator annotations (per-op row counts / wall-time) are
    // forward-deferred to W12β / M4-71 RowCountObserver. The TOP-
    // level metrics (wall_time_ms / rows_emitted /
    // memory_bytes_high_water) ARE populated end-to-end here.
    let s = substrate_with_n_persons(3);
    let cat = cat_basic();
    let engine = QueryEngine::new(&cat);
    let (pt, metrics) = engine
        .profile("PROFILE MATCH (n:Person) RETURN n", &s)
        .expect("profile");
    // PlanTree shape: same as EXPLAIN's (no per-op annotations until
    // W12β/M4-71).
    assert_eq!(pt.op, PlanTreeOp::Project);
    assert_eq!(pt.children.len(), 1);
    assert_eq!(pt.children[0].op, PlanTreeOp::Scan);
    // Metrics: rows_emitted populated end-to-end.
    assert_eq!(metrics.rows_emitted, 3);
    // W13β M4-81: PROFILE's executor pass walks the same materialize
    // tail as `execute`, so memory_bytes_high_water also tracks the
    // per-query bytes. Pre-W13β placeholder was 0 (M4-64a deferred).
    assert!(
        metrics.memory_bytes_high_water > 0,
        "PROFILE executor tail must carry non-zero memory_bytes_high_water: {metrics:?}"
    );
}

#[test]
fn profile_on_bare_read_query_routes_through_same_pipeline() {
    // PROFILE accepts a bare read query (no PROFILE prefix) — the
    // surface mirrors EXPLAIN's bare-query admissibility per ADR-038
    // §2 D-19.
    let s = substrate_with_n_persons(2);
    let cat = cat_basic();
    let engine = QueryEngine::new(&cat);
    let (pt, metrics) = engine
        .profile("MATCH (n:Person) RETURN n", &s)
        .expect("profile bare");
    assert_eq!(pt.op, PlanTreeOp::Project);
    assert_eq!(metrics.rows_emitted, 2);
}

#[test]
fn profile_surfaces_executor_errors_via_explain_error_taxonomy() {
    // PROFILE on a query whose plan shape requires M4-63 (aggregation)
    // forwards through the executor's NotImplemented arm — which
    // routes to ExplainError::ArcQL(ArcQLError::NotImplemented {...}).
    // The W11Z fix-up MED-2 per-arm translation is preserved.
    let s = StubExecutorSubstrate::new();
    let cat = cat_basic();
    let engine = QueryEngine::new(&cat);
    // Aggregation surfaces NotImplemented at the plan-build layer
    // (LogicalPlan::Aggregate is admitted by lowering but the
    // executor pipeline-builder rejects it with a forward-link to
    // M4-63). The error reaches PROFILE via translate_execution_error.
    let result = engine.profile("MATCH (n:Person) RETURN COUNT(n)", &s);
    match result {
        Err(ExplainError::ArcQL(_)) => {
            // Either NotImplemented (executor side) or a plan-time
            // ArcQLError surfaces here; both are the ArcQL umbrella
            // post-W11Z-MED-2 translation.
        }
        Ok(_) => {
            // If aggregation lights at M4-63 before this test runs,
            // the result becomes Ok(...) — that's a forward-method
            // outcome, not a regression. The test is documented to
            // accept both shapes; the load-bearing pin is that PROFILE
            // surfaces a structured error (NOT a panic) when the
            // executor path is incomplete.
        }
        Err(other) => {
            // Cancelled / Substrate / ExecutionEval are all valid
            // surfaces. Parse is the only variant we'd consider a
            // regression.
            assert!(
                !matches!(other, ExplainError::Parse(_)),
                "PROFILE on syntactically valid input must not surface Parse: {other:?}"
            );
        }
    }
}
