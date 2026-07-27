//! TCK `expressions/list/List11` [3] "Create an empty list if range
//! direction and step direction are inconsistent" — END-TO-END.
//!
//! # What this pins
//!
//! Scenario [3] is the first List11 row that EXERCISES `range()` over a
//! grid of (stop, step) pairs and self-checks each result against the
//! direction rule with a `collect()` aggregate nested inside an `ALL(...)`
//! quantifier:
//!
//! ```cypher
//! WITH 0 AS start, [1, 2, 500, 1000, 1500] AS stopList,
//!      [-1000, -3, -2, -1, 1, 2, 3, 1000] AS stepList
//! UNWIND stopList AS stop
//! UNWIND stepList AS step
//! WITH start, stop, step, range(start, stop, step) AS list
//! WITH start, stop, step, list, sign(stop-start) <> sign(step) AS empty
//! RETURN ALL(ok IN collect((size(list) = 0) = empty) WHERE ok) AS okay
//! ```
//!
//! On pre-fix `main` the `range()` builtin ALREADY produced the correct
//! empty/non-empty lists (an inconsistent step direction yields `[]`); the
//! scenario nonetheless failed `RowsMismatch` because the final projection
//! raised `AmbiguousAggregationExpression` at BIND time. The implicit
//! grouping-key walk (`agg_has_nongrouping_ref`) treated the `ALL`
//! quantifier's SCOPED iteration variable `ok` — referenced in `WHERE ok`
//! — as a FREE outer-scope reference that had to be a grouping key. It is
//! not: `ok` is bound locally by the quantifier (ADR-188), so openCypher
//! v9 §6.4 (which governs FREE references only) exempts it. The fix adds
//! the scoped iteration variable(s) of `ListPredicate` /
//! `ListComprehension` / `Reduce` to the grouping-key set for the body
//! recursion.
//!
//! # Oracle
//!
//! The scenario's own `RETURN ALL(...) AS okay` is a self-checking
//! oracle: it is `true` IFF, for every (stop, step) pair, `range()`'s
//! emptiness EXACTLY matches `sign(stop-start) <> sign(step)`. A single
//! `true` row is the TCK-expected result. We additionally probe the
//! `range()` builtin directly so a regression in EITHER the eval (empty
//! list) OR the binder (aggregate-in-quantifier) is localized.

use arcgraph_query::QueryEngine;
use arcgraph_query::executor::value::Value;
use arcgraph_query::executor::{ExecutorSubstrate, StubExecutorSubstrate};
use arcgraph_query::semantic::StubCatalogProvider;

fn run<S: ExecutorSubstrate>(query: &str, c: &StubCatalogProvider, s: &S) -> Vec<Vec<Value>> {
    let engine = QueryEngine::new(c);
    let r = engine.execute(query, s).expect("execute");
    r.rows
}

fn ints(vs: &[i64]) -> Value {
    Value::List(vs.iter().copied().map(Value::Integer).collect())
}

// =====================================================================
// THE target — full scenario [3], self-checking ALL(...) oracle.
// =====================================================================

#[test]
fn list11_3_inconsistent_step_yields_empty_self_check() {
    let rows = run(
        "WITH 0 AS start, [1, 2, 500, 1000, 1500] AS stopList, \
         [-1000, -3, -2, -1, 1, 2, 3, 1000] AS stepList \
         UNWIND stopList AS stop \
         UNWIND stepList AS step \
         WITH start, stop, step, range(start, stop, step) AS list \
         WITH start, stop, step, list, sign(stop-start) <> sign(step) AS empty \
         RETURN ALL(ok IN collect((size(list) = 0) = empty) WHERE ok) AS okay",
        &StubCatalogProvider::new(),
        &StubExecutorSubstrate::new(),
    );
    assert_eq!(rows.len(), 1, "the aggregating projection folds to one row");
    assert_eq!(
        rows[0],
        vec![Value::Boolean(true)],
        "for every (stop, step), range() emptiness must equal \
         sign(stop-start) <> sign(step)"
    );
}

// =====================================================================
// Localizing probes — range() eval direction rule (eval, not binder).
// =====================================================================

#[test]
fn range_inconsistent_direction_is_empty() {
    let c = StubCatalogProvider::new();
    let s = StubExecutorSubstrate::new();
    // up-target, down-step → can't ascend with a negative step.
    assert_eq!(run("RETURN range(0, 5, -1) AS l", &c, &s)[0][0], ints(&[]));
    // down-target, up-step → can't descend with a positive step.
    assert_eq!(run("RETURN range(5, 0, 1) AS l", &c, &s)[0][0], ints(&[]));
}

#[test]
fn range_consistent_direction_still_correct() {
    let c = StubCatalogProvider::new();
    let s = StubExecutorSubstrate::new();
    // Ascending with +step.
    assert_eq!(
        run("RETURN range(0, 5, 2) AS l", &c, &s)[0][0],
        ints(&[0, 2, 4])
    );
    // Descending with -step.
    assert_eq!(
        run("RETURN range(5, 0, -2) AS l", &c, &s)[0][0],
        ints(&[5, 3, 1])
    );
    // Single-element (start == stop).
    assert_eq!(run("RETURN range(0, 0, 1) AS l", &c, &s)[0][0], ints(&[0]));
    // Default step, full ascending range.
    assert_eq!(
        run("RETURN range(0, 5) AS l", &c, &s)[0][0],
        ints(&[0, 1, 2, 3, 4, 5])
    );
}

// =====================================================================
// Localizing probe — aggregate-inside-quantifier binds (the binder fix).
// =====================================================================

#[test]
fn aggregate_inside_all_quantifier_binds_and_evaluates() {
    // Minimal shape of the [3] final projection: collect(...) inside
    // ALL(ok IN ... WHERE ok). `ok` is the quantifier's scoped variable,
    // exempt from the grouping-key rule. Pre-fix this raised
    // AmbiguousAggregationExpression at bind time.
    let rows = run(
        "UNWIND [true, true, true] AS b \
         RETURN ALL(ok IN collect(b) WHERE ok) AS okay",
        &StubCatalogProvider::new(),
        &StubExecutorSubstrate::new(),
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0], vec![Value::Boolean(true)]);
}
