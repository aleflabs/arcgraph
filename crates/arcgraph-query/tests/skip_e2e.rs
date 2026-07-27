//! **#842 part A** — `SKIP N` (offset pagination) END-TO-END tests
//! (openCypher v9 §6.6).
//!
//! These exercise the FULL pipeline (parse → bind → type-check →
//! cross-substrate → lower → execute) for literal `SKIP`, complementing
//! the direct-operator strong-oracle unit tests in
//! `src/executor/ops/skip.rs`. They are the ADR-133 §D-4 Query-class
//! active-verification surface for this slice: real queries, STRONG `==`
//! oracles over the exact result rows.
//!
//! # What #842 reported (and these pin as FIXED)
//!
//! `SKIP` errored `-32005` in EVERY position — top-level RETURN and
//! WITH-stage, with or without ORDER BY — while `LIMIT` worked. Offset
//! pagination (`SKIP n LIMIT m`, the universal "page k" idiom) was
//! therefore impossible. Each test below is a query that returned
//! `-32005` on main `73275af8` (the `LogicalPlan::Skip(_) =>
//! NotImplemented` arm at `executor/pipeline.rs`) and now returns the
//! correct rows. Driving rows come from `UNWIND` (no graph fixture
//! needed; UNWIND preserves list order, so SKIP's "drop the first N in
//! stream order" semantics yield a deterministic, order-exact oracle).

use arcgraph_query::executor::{ExecutionContext, StubExecutorSubstrate, Value};
use arcgraph_query::logical_plan::LogicalPlanLoweringVisitor;
use arcgraph_query::semantic::{
    BindingVisitor, CatalogProvider, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};
use arcgraph_query::{materialize, parse};

fn cat() -> StubCatalogProvider {
    StubCatalogProvider::new()
}

/// Full pipeline → result rows (panics on any stage error).
fn run(query: &str, s: &StubExecutorSubstrate, c: &StubCatalogProvider) -> Vec<Vec<Value>> {
    let stmt = parse(query).expect("parse");
    let mut bound = BindingVisitor::bind(&stmt, query, c).expect("bind");
    TypeCheckVisitor::check(&mut bound, c).expect("type-check");
    CrossSubstrateValidator::validate(&bound, c).expect("cross-substrate");
    let plan = LogicalPlanLoweringVisitor::lower(&bound).expect("lower");
    let ctx = ExecutionContext::new(c.tenant(), c.partition());
    materialize::materialize(&plan, s, &ctx)
        .expect("materialize")
        .rows()
        .to_vec()
}

fn rows_of_ints(xs: &[i64]) -> Vec<Vec<Value>> {
    xs.iter().map(|n| vec![Value::Integer(*n)]).collect()
}

#[test]
fn return_skip_drops_leading_rows_in_order() {
    // `RETURN x SKIP 2` over [1..=5] ⇒ [3],[4],[5] (NOT just "3 rows").
    let s = StubExecutorSubstrate::new();
    let rows = run("UNWIND [1, 2, 3, 4, 5] AS x RETURN x SKIP 2", &s, &cat());
    assert_eq!(rows, rows_of_ints(&[3, 4, 5]));
}

#[test]
fn return_skip_then_limit_is_pagination() {
    // THE langchain / "page 2" idiom: `SKIP 2 LIMIT 2` over [1..=5] ⇒
    // [3],[4] (offset 2, window 2). Lowers to Limit(Skip(child)).
    let s = StubExecutorSubstrate::new();
    let rows = run(
        "UNWIND [1, 2, 3, 4, 5] AS x RETURN x SKIP 2 LIMIT 2",
        &s,
        &cat(),
    );
    assert_eq!(rows, rows_of_ints(&[3, 4]));
}

#[test]
fn return_skip_zero_returns_all_rows() {
    // `SKIP 0` is a no-op offset — every row, in order.
    let s = StubExecutorSubstrate::new();
    let rows = run("UNWIND [1, 2, 3] AS x RETURN x SKIP 0", &s, &cat());
    assert_eq!(rows, rows_of_ints(&[1, 2, 3]));
}

#[test]
fn return_skip_at_or_past_count_returns_empty() {
    // `SKIP N >= count` ⇒ empty result (skip everything).
    let s = StubExecutorSubstrate::new();
    let rows = run("UNWIND [1, 2, 3] AS x RETURN x SKIP 5", &s, &cat());
    assert_eq!(rows, Vec::<Vec<Value>>::new());
    // Boundary: SKIP exactly equal to the row count is also empty.
    let rows_eq = run("UNWIND [1, 2, 3] AS x RETURN x SKIP 3", &s, &cat());
    assert_eq!(rows_eq, Vec::<Vec<Value>>::new());
}

#[test]
fn with_stage_skip_then_return() {
    // The issue's `WITH a SKIP 1 RETURN a.id` shape: a WITH-stage SKIP
    // (arrives as a tail SKIP clause after the WITH projection) feeds the
    // downstream RETURN. [1..=5] skip 1 ⇒ [2],[3],[4],[5].
    let s = StubExecutorSubstrate::new();
    let rows = run(
        "UNWIND [1, 2, 3, 4, 5] AS x WITH x SKIP 1 RETURN x",
        &s,
        &cat(),
    );
    assert_eq!(rows, rows_of_ints(&[2, 3, 4, 5]));
}

#[test]
fn order_by_then_skip_sorts_before_offset() {
    // The issue's `... ORDER BY a.bal SKIP 1` shape: ORDER BY sorts, THEN
    // SKIP drops from the sorted stream. [3,1,2] sorted asc = [1,2,3];
    // skip 1 ⇒ [2],[3]. Proves SKIP composes after Sort (not before).
    let s = StubExecutorSubstrate::new();
    let rows = run(
        "UNWIND [3, 1, 2] AS x WITH x ORDER BY x SKIP 1 RETURN x",
        &s,
        &cat(),
    );
    assert_eq!(rows, rows_of_ints(&[2, 3]));
}

#[test]
fn order_by_skip_limit_full_pagination_window() {
    // ORDER BY + SKIP + LIMIT together — the fully-specified pagination
    // query. [5,4,3,2,1] sorted asc = [1,2,3,4,5]; SKIP 1 LIMIT 2 ⇒
    // [2],[3] (the deterministic "page 2, size 2" of an ordered result).
    let s = StubExecutorSubstrate::new();
    let rows = run(
        "UNWIND [5, 4, 3, 2, 1] AS x WITH x ORDER BY x SKIP 1 LIMIT 2 RETURN x",
        &s,
        &cat(),
    );
    assert_eq!(rows, rows_of_ints(&[2, 3]));
}
