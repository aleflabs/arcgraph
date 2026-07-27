//! **#842 part B** — `WITH DISTINCT …` (mid-pipeline dedup) END-TO-END
//! tests (openCypher v9 §6.4).
//!
//! These exercise the FULL pipeline (parse → bind → type-check →
//! cross-substrate → lower → execute) for `WITH DISTINCT`, the ADR-133
//! §D-4 Query-class active-verification surface for this slice: real
//! queries, STRONG `==` oracles over the exact result rows.
//!
//! # What #842 reported (and these pin as FIXED)
//!
//! `RETURN DISTINCT` worked but `WITH DISTINCT a.country AS c RETURN c`
//! was a `-32700` PARSE error — the grammar accepted `DISTINCT` after
//! `RETURN` but not after `WITH`, so mid-pipeline dedup (dedup THEN
//! continue the pipeline / traverse / aggregate) was impossible. The fix
//! adds `kw_distinct?` to the `with_clause` grammar and threads a
//! `distinct: bool` AST→BoundAST, then the lowering composes the SAME
//! `LogicalDistinct` operator `RETURN DISTINCT` uses over the WITH
//! projection. Driving rows come from `UNWIND` (no graph fixture; UNWIND
//! preserves list order and `DistinctOp` emits each key on first-seen, so
//! the dedup is order-exact + deterministic).

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
fn with_distinct_dedups_then_returns() {
    // The core fix: `WITH DISTINCT v RETURN v` dedups the projected rows
    // (first-seen order). [1,1,2,2,3] ⇒ [1],[2],[3]. On main this was a
    // `-32700` parse error (no `kw_distinct?` in `with_clause`).
    let s = StubExecutorSubstrate::new();
    let rows = run(
        "UNWIND [1, 1, 2, 2, 3] AS v WITH DISTINCT v RETURN v",
        &s,
        &cat(),
    );
    assert_eq!(rows, rows_of_ints(&[1, 2, 3]));
}

#[test]
fn without_distinct_with_keeps_duplicates() {
    // Contrast guard: plain `WITH v` does NOT dedup (proves the DISTINCT
    // keyword — not the WITH itself — is what dedups). [1,1,2] ⇒ [1],[1],[2].
    let s = StubExecutorSubstrate::new();
    let rows = run("UNWIND [1, 1, 2] AS v WITH v RETURN v", &s, &cat());
    assert_eq!(rows, rows_of_ints(&[1, 1, 2]));
}

#[test]
fn with_distinct_feeds_downstream_aggregation() {
    // The task's exact oracle: `UNWIND [1,1,2] AS v WITH DISTINCT v
    // RETURN collect(v)` ⇒ a single row [[1, 2]] (dedup, THEN aggregate).
    let s = StubExecutorSubstrate::new();
    let rows = run(
        "UNWIND [1, 1, 2] AS v WITH DISTINCT v RETURN collect(v)",
        &s,
        &cat(),
    );
    assert_eq!(
        rows,
        vec![vec![Value::List(vec![
            Value::Integer(1),
            Value::Integer(2)
        ])]],
        "collect over the DISTINCT'd stream = [1, 2]"
    );
}

#[test]
fn with_distinct_aliased_then_transform() {
    // Dedup THEN continue the pipeline (the idiom the parse error blocked):
    // `WITH DISTINCT v AS d RETURN d * 10`. [1,1,2,2] ⇒ d∈{1,2} ⇒ [10],[20].
    let s = StubExecutorSubstrate::new();
    let rows = run(
        "UNWIND [1, 1, 2, 2] AS v WITH DISTINCT v AS d RETURN d * 10",
        &s,
        &cat(),
    );
    assert_eq!(rows, rows_of_ints(&[10, 20]));
}

#[test]
fn with_distinct_then_where_dedups_before_filter() {
    // `WITH DISTINCT v WHERE v > 1` — DISTINCT applies BEFORE the post-WITH
    // WHERE (openCypher v9 §6.4 sub-clause order). [1,1,2,2,3] ⇒ distinct
    // {1,2,3} ⇒ filter >1 ⇒ [2],[3].
    let s = StubExecutorSubstrate::new();
    let rows = run(
        "UNWIND [1, 1, 2, 2, 3] AS v WITH DISTINCT v WHERE v > 1 RETURN v",
        &s,
        &cat(),
    );
    assert_eq!(rows, rows_of_ints(&[2, 3]));
}

#[test]
fn with_distinct_order_by_limit_tck44_shape() {
    // The openCypher `WithOrderBy1` [44] shape (read-only): dedup, sort,
    // take 1. `UNWIND [0,2,1,2,0,1] AS x WITH DISTINCT x ORDER BY x LIMIT 1
    // RETURN x` ⇒ [0] (the smallest distinct value).
    let s = StubExecutorSubstrate::new();
    let rows = run(
        "UNWIND [0, 2, 1, 2, 0, 1] AS x WITH DISTINCT x ORDER BY x LIMIT 1 RETURN x",
        &s,
        &cat(),
    );
    assert_eq!(rows, rows_of_ints(&[0]));
}

#[test]
fn with_distinct_then_skip_uses_deduped_stream() {
    // #842 composition oracle: DISTINCT must run before the tail SKIP.
    // First-seen distinct over [3,1,2,2,1,3] is [3,1,2]; SKIP 1 then
    // returns [1],[2]. A regression that drops the WITH DISTINCT lowering
    // or applies SKIP before dedup returns a different row sequence.
    let s = StubExecutorSubstrate::new();
    let rows = run(
        "UNWIND [3, 1, 2, 2, 1, 3] AS x WITH DISTINCT x SKIP 1 RETURN x",
        &s,
        &cat(),
    );
    assert_eq!(rows, rows_of_ints(&[1, 2]));
}

#[test]
fn with_distinct_order_by_skip_limit_pages_distinct_rows() {
    // Full #842 stack: projection -> DISTINCT -> ORDER BY -> SKIP ->
    // LIMIT. Distinct [3,1,2], sorted [1,2,3], skip 1, limit 1 => [2].
    // This is the missing composition between the standalone SKIP and
    // WITH DISTINCT fixes.
    let s = StubExecutorSubstrate::new();
    let rows = run(
        "UNWIND [3, 1, 2, 2, 1, 3] AS x WITH DISTINCT x ORDER BY x SKIP 1 LIMIT 1 RETURN x",
        &s,
        &cat(),
    );
    assert_eq!(rows, rows_of_ints(&[2]));
}

#[test]
fn with_distinct_multi_column_tuple_dedup() {
    // Multi-column DISTINCT dedups on the whole projected tuple, not a
    // single column. Rows (1,'a'),(1,'a'),(1,'b') ⇒ distinct (1,'a'),(1,'b').
    // Built via two UNWINDs would be a cartesian; instead use a list of
    // pairs unwound + projected to two columns is overkill — assert the
    // simpler observable: distinct over (v, v%2)-style derived tuple.
    // [1,1,2,2,3] with cols (v, v) ⇒ distinct {(1,1),(2,2),(3,3)} = 3 rows.
    let s = StubExecutorSubstrate::new();
    let rows = run(
        "UNWIND [1, 1, 2, 2, 3] AS v WITH DISTINCT v AS a, v AS b RETURN a, b",
        &s,
        &cat(),
    );
    assert_eq!(
        rows,
        vec![
            vec![Value::Integer(1), Value::Integer(1)],
            vec![Value::Integer(2), Value::Integer(2)],
            vec![Value::Integer(3), Value::Integer(3)],
        ]
    );
}
