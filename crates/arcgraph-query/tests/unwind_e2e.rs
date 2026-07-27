//! **ADR-038 D-28 §7 / #618** — `UNWIND <list> AS <var>` END-TO-END
//! tests (openCypher v9 §6.7).
//!
//! These exercise the FULL pipeline (parse → bind → type-check →
//! cross-substrate → lower → execute) for UNWIND, complementing the
//! direct-operator strong-oracle unit tests in
//! `src/executor/ops/unwind.rs`. They are the ADR-133 §D-4 Query-class
//! active-verification surface for this slice: real queries, STRONG `==`
//! oracles over the exact result rows.
//!
//! # Coverage vs. the openCypher Unwind1.feature scenarios
//!
//! PASSING here (leading UNWIND — no preceding WITH):
//! - Unwind1 [1] list, [8] empty list, [9] null, [10] duplicates,
//!   plus UNWIND composed after a real MATCH.
//!
//! PASSING here since #746 (the WITH→UNWIND chain — Unwind1
//! [3]/[7]/[11]/[13] and the prompt's `WITH [10,20] AS xs UNWIND xs ...`
//! case): #746 established the binder↔`ProjectOp` output-binding-id
//! contract (the binder assigns each projection's output id; `ProjectOp`
//! emits the column under THAT id instead of a fresh synthetic), so a
//! downstream `UNWIND` / `RETURN` / `MATCH` resolves the WITH-projected
//! name. Before #746 ANY `WITH x ... <ref to x>` (even
//! `MATCH (n) WITH n RETURN n`) errored with "binding ... missing from
//! row schema"; the regression guard
//! `with_then_unwind_projection_binding_resolves` below pins the fix.

use arcgraph_core::{LabelId, NodeId, TenantId};
use arcgraph_query::executor::value::NodeView;
use arcgraph_query::executor::{ExecutionContext, StubExecutorSubstrate, Value};
use arcgraph_query::logical_plan::LogicalPlanLoweringVisitor;
use arcgraph_query::semantic::{
    BindingVisitor, CatalogProvider, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};
use arcgraph_query::{materialize, parse};

const LABEL_X: u32 = 1;

fn cat() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["X"])
        .with_properties(["g"])
}

fn node(id: u64) -> NodeView {
    NodeView::new(NodeId::new(id), Some(LabelId::new(LABEL_X)))
}

/// Full pipeline → result rows (panics on any stage error).
fn run(query: &str, s: &StubExecutorSubstrate, c: &StubCatalogProvider) -> Vec<Vec<Value>> {
    let plan = lower(query, c);
    let ctx = ExecutionContext::new(c.tenant(), c.partition());
    materialize::materialize(&plan, s, &ctx)
        .expect("materialize")
        .rows()
        .to_vec()
}

fn lower(query: &str, c: &StubCatalogProvider) -> arcgraph_query::logical_plan::LogicalPlan {
    let stmt = parse(query).expect("parse");
    let mut bound = BindingVisitor::bind(&stmt, query, c).expect("bind");
    TypeCheckVisitor::check(&mut bound, c).expect("type-check");
    CrossSubstrateValidator::validate(&bound, c).expect("cross-substrate");
    LogicalPlanLoweringVisitor::lower(&bound).expect("lower")
}

fn rows_of_ints(xs: &[i64]) -> Vec<Vec<Value>> {
    xs.iter().map(|n| vec![Value::Integer(*n)]).collect()
}

// =====================================================================
// Leading UNWIND (no preceding MATCH) — the driving-row path.
// =====================================================================

#[test]
fn unwind_list_literal_returns_each_element_in_order() {
    // Unwind1 [1] + the prompt's required (b1) e2e oracle.
    // `UNWIND [1, 2, 3] AS x RETURN x` ⇒ exactly [1],[2],[3].
    let s = StubExecutorSubstrate::new();
    let rows = run("UNWIND [1, 2, 3] AS x RETURN x", &s, &cat());
    assert_eq!(rows, rows_of_ints(&[1, 2, 3]));
}

#[test]
fn unwind_empty_list_returns_zero_rows() {
    // Unwind1 [8]: `UNWIND [] AS empty RETURN empty` ⇒ no rows.
    let s = StubExecutorSubstrate::new();
    let rows = run("UNWIND [] AS empty RETURN empty", &s, &cat());
    assert_eq!(rows, Vec::<Vec<Value>>::new());
}

#[test]
fn unwind_null_returns_zero_rows() {
    // Unwind1 [9]: `UNWIND null AS nil RETURN nil` ⇒ no rows (NOT error).
    let s = StubExecutorSubstrate::new();
    let rows = run("UNWIND null AS nil RETURN nil", &s, &cat());
    assert_eq!(rows, Vec::<Vec<Value>>::new());
}

#[test]
fn unwind_list_with_duplicates_preserves_them() {
    // Unwind1 [10]: duplicates are NOT collapsed; order preserved.
    let s = StubExecutorSubstrate::new();
    let rows = run(
        "UNWIND [1, 1, 2, 2, 3, 3, 4, 4, 5, 5] AS d RETURN d",
        &s,
        &cat(),
    );
    assert_eq!(rows, rows_of_ints(&[1, 1, 2, 2, 3, 3, 4, 4, 5, 5]));
}

#[test]
fn unwind_nested_list_returns_each_inner_list() {
    // `UNWIND [[1,2],[3,4]] AS pair RETURN pair` ⇒ two list-valued rows.
    let s = StubExecutorSubstrate::new();
    let rows = run("UNWIND [[1, 2], [3, 4]] AS pair RETURN pair", &s, &cat());
    assert_eq!(
        rows,
        vec![
            vec![Value::List(vec![Value::Integer(1), Value::Integer(2)])],
            vec![Value::List(vec![Value::Integer(3), Value::Integer(4)])],
        ]
    );
}

// =====================================================================
// UNWIND composed AFTER a real MATCH (cartesian over upstream rows).
// =====================================================================

#[test]
fn unwind_after_match_is_cartesian_and_preserves_match_binding() {
    // Two matched nodes × a 2-element list ⇒ 4 rows; `RETURN x` projects
    // the unwound element (the upstream `n` binding stays in scope and is
    // referenceable — proven by the next test).
    let s = StubExecutorSubstrate::new()
        .with_node(TenantId::DEFAULT, node(1))
        .with_node(TenantId::DEFAULT, node(2));
    let rows = run("MATCH (n:X) UNWIND [10, 20] AS x RETURN x", &s, &cat());
    // in any order per openCypher; sort the projected integers.
    let mut got: Vec<i64> = rows
        .iter()
        .map(|r| match r[0] {
            Value::Integer(v) => v,
            ref o => panic!("expected Integer, got {o:?}"),
        })
        .collect();
    got.sort_unstable();
    assert_eq!(got, vec![10, 10, 20, 20], "2 nodes × [10,20]");
}

#[test]
fn unwind_after_match_keeps_match_binding_referenceable() {
    // `RETURN n, x` proves BOTH the upstream MATCH binding `n` AND the
    // unwound `x` are in scope after UNWIND (Unwind1 [12] shape, without
    // the WITH/collect that the projection-binding bug blocks).
    let s = StubExecutorSubstrate::new().with_node(TenantId::DEFAULT, node(1));
    let rows = run("MATCH (n:X) UNWIND [7, 8] AS x RETURN n, x", &s, &cat());
    assert_eq!(rows.len(), 2, "1 node × [7,8]");
    for r in &rows {
        assert_eq!(r.len(), 2, "two columns: n, x");
        assert!(matches!(&r[0], Value::Node(nv) if nv.id == NodeId::new(1)));
    }
    let mut xs: Vec<i64> = rows
        .iter()
        .map(|r| match r[1] {
            Value::Integer(v) => v,
            ref o => panic!("expected Integer x, got {o:?}"),
        })
        .collect();
    xs.sort_unstable();
    assert_eq!(xs, vec![7, 8]);
}

// =====================================================================
// #746 REGRESSION GUARD: WITH→UNWIND now resolves end-to-end. This was
// the foundation-train tripwire pinning the PRE-EXISTING,
// UNWIND-INDEPENDENT projection-binding bug; #746 established the
// binder↔ProjectOp output-binding-id contract, so it is FLIPPED to
// assert the correct result (per `feedback_review_oracle_relaxations` —
// guard the fix with the correct oracle, do NOT merely delete).
// =====================================================================

#[test]
fn with_then_unwind_projection_binding_resolves() {
    // `WITH [10, 20] AS xs UNWIND xs AS x RETURN x * x` ⇒ [100],[400].
    //
    // Before #746 this ERRORED with "binding ... missing from row
    // schema": the WITH `ProjectOp` emitted a fresh SYNTHETIC output
    // binding-id for `xs` that did not match the post-WITH scope id the
    // downstream UNWIND resolved `xs` to. The #746 contract makes the
    // binder assign the projection's output id and `ProjectOp` emit the
    // column under THAT id, so the UNWIND (and any other downstream
    // reference) finds it. UNWIND preserves list order, so the squared
    // elements come back in order.
    let s = StubExecutorSubstrate::new();
    let rows = run(
        "WITH [10, 20] AS xs UNWIND xs AS x RETURN x * x",
        &s,
        &cat(),
    );
    assert_eq!(
        rows,
        rows_of_ints(&[100, 400]),
        "WITH-projected list unwound + squared (the #746 fix)"
    );
}
