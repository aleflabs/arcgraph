//! **openCypher v9 §6.6 — an AGGREGATE inside an ORDER BY sort key**, END-TO-END
//! through `QueryEngine::execute`.
//!
//! # The gap this pins
//!
//! openCypher permits an aggregate to appear INSIDE an ORDER BY sort
//! expression, where the aggregate references the SAME post-aggregation output
//! that the RETURN / WITH projection produces:
//!
//! ```cypher
//! MATCH (me:Person)--(you:Person)
//! RETURN me.age AS age, count(you.age) AS cnt
//! ORDER BY me.age + count(you.age)   -- aggregate INSIDE the sort key
//! ```
//!
//! The aggregate `count(you.age)` is computed ONCE alongside the RETURN/WITH
//! aggregates, and the sort key references that computed column. ArcGraph's
//! binder previously REJECTED any aggregate appearing in the ORDER BY sort
//! expression (`UnexpectedError`), because the existing ORDER BY resolution
//! (#857 projected expr, #884 non-projected in-scope, #767 projection output)
//! only handled NON-aggregate sort keys. This file is the RED-on-revert proof
//! that an inline aggregate in the sort key now resolves against the
//! post-aggregation output scope.
//!
//! # Golden oracle = vendored TCK
//!
//! - `clauses/with-orderBy/WithOrderBy4.feature` [17]/[18] (WITH form)
//! - `clauses/return-orderby/ReturnOrderBy6.feature` [2]/[3] (RETURN form)
//!
//! All four use `Given an empty graph` and expect an EMPTY result set — so the
//! oracle is "compiles + runs + returns 0 rows" (no fixture/data dependency).
//! The PARALLEL "Fail if …" scenarios ([19]/[20], [4]/[5]) still expect a
//! compile-time error and are pinned here too (so the fix does not over-accept).

use arcgraph_query::QueryEngine;
use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::executor::value::Value;
use arcgraph_query::semantic::StubCatalogProvider;

/// Execute against a bare catalog + empty substrate (labels are permissive, so
/// `:Person` binds to empty). Returns Ok(row count) / Err(debug string) of the
/// FULL pipeline (parse → bind → type-check → cross-substrate → lower → exec).
fn run(cypher: &str) -> Result<usize, String> {
    let cat = StubCatalogProvider::new();
    let s = StubExecutorSubstrate::new();
    QueryEngine::new(&cat)
        .execute(cypher, &s)
        .map(|r| r.rows.len())
        .map_err(|e| format!("{e:?}"))
}

/// As [`run`], but returns the FULL result rows (for ordering oracles over a
/// POPULATED graph — `UNWIND` synthesizes rows with no substrate fixture).
fn run_rows(cypher: &str) -> Vec<Vec<Value>> {
    let cat = StubCatalogProvider::new();
    let s = StubExecutorSubstrate::new();
    QueryEngine::new(&cat)
        .execute(cypher, &s)
        .unwrap_or_else(|e| panic!("expected OK for `{cypher}`, got error: {e:?}"))
        .rows
}

/// Extract the first column as integers, IN RESULT ORDER (the order IS the
/// assertion — the sort key is a computed aggregate).
fn col0_ints(rows: Vec<Vec<Value>>) -> Vec<i64> {
    rows.into_iter()
        .map(|r| match &r[0] {
            Value::Integer(n) => *n,
            other => panic!("expected Integer in col0, got {other:?}"),
        })
        .collect()
}

/// Assert the query COMPILES + runs and returns EXACTLY `expected_rows` rows
/// (the four target scenarios run over an empty graph → 0 rows).
fn assert_rows(cypher: &str, expected_rows: usize) {
    match run(cypher) {
        Ok(n) => assert_eq!(
            n, expected_rows,
            "`{cypher}` should return {expected_rows} rows, got {n}"
        ),
        Err(e) => panic!("expected OK ({expected_rows} rows) for `{cypher}`, got error: {e}"),
    }
}

// =====================================================================
// PASS — an aggregate INSIDE the ORDER BY sort key, resolving against the
// post-aggregation output of the RETURN / WITH projection.
// =====================================================================

/// WithOrderBy4 [17] — projected VARIABLE (`age`) plus an inline aggregate in
/// the sort key, in the WITH form.
#[test]
fn with_orderby4_17_projected_variable_inside_orderby_aggregate() {
    assert_rows(
        "MATCH (me:Person)--(you:Person) \
         WITH me.age AS age, count(you.age) AS cnt \
         ORDER BY age, age + count(you.age) \
         RETURN age",
        0,
    );
}

/// WithOrderBy4 [18] — projected PROPERTY ACCESS (`me.age`) plus an inline
/// aggregate in the sort key, in the WITH form.
#[test]
fn with_orderby4_18_projected_property_inside_orderby_aggregate() {
    assert_rows(
        "MATCH (me:Person)--(you:Person) \
         WITH me.age AS age, count(you.age) AS cnt \
         ORDER BY me.age + count(you.age) \
         RETURN age",
        0,
    );
}

/// ReturnOrderBy6 [2] — returned ALIAS (`age`) plus an inline aggregate in the
/// sort key, in the RETURN form.
#[test]
fn return_orderby6_2_returned_alias_inside_orderby_aggregate() {
    assert_rows(
        "MATCH (me:Person)--(you:Person) \
         RETURN me.age AS age, count(you.age) AS cnt \
         ORDER BY age, age + count(you.age)",
        0,
    );
}

/// ReturnOrderBy6 [3] — returned PROPERTY ACCESS (`me.age`) plus an inline
/// aggregate in the sort key, in the RETURN form.
#[test]
fn return_orderby6_3_returned_property_inside_orderby_aggregate() {
    assert_rows(
        "MATCH (me:Person)--(you:Person) \
         RETURN me.age AS age, count(you.age) AS cnt \
         ORDER BY me.age + count(you.age)",
        0,
    );
}

// =====================================================================
// FAIL — the parallel "Fail if …" scenarios must STILL be rejected, so the
// accept-path does not over-accept (no grouping-key context for the aggregate).
// =====================================================================

/// WithOrderBy4 [19] / ReturnOrderBy6 [4] — a NON-projected variable (`me.age`)
/// used inside an ORDER BY aggregate item must fail (UndefinedVariable).
#[test]
fn nonprojected_variable_inside_orderby_aggregate_is_rejected() {
    // WITH form: only `agg` is projected; `me.age` is not in scope post-WITH.
    assert!(
        run("MATCH (me:Person)--(you:Person) \
             WITH count(you.age) AS agg \
             ORDER BY me.age + count(you.age) \
             RETURN *")
        .is_err(),
        "ORDER BY over a non-projected variable inside an aggregate must be rejected"
    );
    // RETURN form analogue.
    assert!(
        run("MATCH (me:Person)--(you:Person) \
             RETURN count(you.age) AS agg \
             ORDER BY me.age + count(you.age)")
        .is_err(),
        "ORDER BY over a non-returned variable inside an aggregate must be rejected"
    );
}

// =====================================================================
// REAL ordering oracle — the empty-graph TCK shapes return 0 rows for
// ANY implementation (the Aggregate folds over zero input), so they pass
// trivially even if the inline aggregate is NOT lifted into the
// Aggregate node. These POPULATED cases (via `UNWIND`, no substrate
// fixture) prove the sort actually orders by the COMPUTED aggregate
// column — the fix is real, not a fixture mask (#1228 lesson). Before the
// lowering fix these errored `NotImplemented: aggregation function count`
// (the Sort op tried to evaluate `count(*)` row-wise).
// =====================================================================

/// `ORDER BY count(*)` — the BARE inline aggregate as the sort key. Counts
/// per group: 1→2, 2→1, 3→3. Ascending by count ⇒ x=2, x=1, x=3.
#[test]
fn populated_order_by_bare_aggregate_ascending() {
    let rows = run_rows("UNWIND [1, 1, 2, 3, 3, 3] AS x RETURN x, count(*) AS c ORDER BY count(*)");
    assert_eq!(
        col0_ints(rows),
        vec![2, 1, 3],
        "ORDER BY count(*) ascending must order groups by their row count"
    );
}

/// `ORDER BY x + count(*) DESC` — a COMPOUND key mixing a grouping key (`x`)
/// and an inline aggregate. Keys: x=1→1+2=3, x=2→2+1=3, x=3→3+3=6. DESC ⇒
/// x=3 (key 6) first; the two key-3 groups follow in either order.
#[test]
fn populated_order_by_grouping_key_plus_aggregate_descending() {
    let rows = run_rows(
        "UNWIND [1, 1, 2, 3, 3, 3] AS x RETURN x, count(*) AS c ORDER BY x + count(*) DESC",
    );
    let xs = col0_ints(rows);
    assert_eq!(xs.len(), 3, "three groups");
    assert_eq!(xs[0], 3, "DESC: x=3 (computed key 6) must sort first");
    assert!(
        xs[1..].contains(&1) && xs[1..].contains(&2),
        "the two key-3 groups (x=1, x=2) follow, got {xs:?}"
    );
}

/// Multi-key sort `ORDER BY x, x + count(*)` — the first key is the bare
/// grouping variable (a tiebreaker) and the SECOND mixes the grouping key
/// `x` with an inline aggregate, mirroring the TCK `ORDER BY age, age +
/// count(...)` two-key shape ([17]/[2]). Counts: 1→3, 2→2, 3→1. Ascending by
/// `x` ⇒ x=1, x=2, x=3 (the second key is consistent, so the order is by x).
#[test]
fn populated_order_by_grouping_var_then_grouping_var_plus_aggregate() {
    let rows =
        run_rows("UNWIND [1, 1, 1, 2, 2, 3] AS x RETURN x, count(*) AS c ORDER BY x, x + count(*)");
    assert_eq!(
        col0_ints(rows),
        vec![1, 2, 3],
        "ascending by the grouping var, with an inline-aggregate second key"
    );
}

// =====================================================================
// #1053 R1 REGRESSION — ORDER BY <agg>(<grouping-key projected under its OWN
// name>). The R1 adversarial review found that `ORDER BY count(x)` where `x`
// is BOTH the grouping key (projected as `x`, not aliased) AND the aggregate's
// argument was ACCEPTED by the binder then CRASHED at execution with
// `ExecutionEval("binding BindingId(1) missing from row schema")`: the binder
// resolved the sort-key aggregate's argument `x` to the grouping key's
// post-aggregation OUTPUT binding, but `AggregateOp` evaluates an aggregate
// argument against the PRE-grouping INPUT rows (where only the input binding
// exists). The base d9bddd03 REJECTED this shape — so the first cut was an
// accept→crash regression. The lowering now remaps the lifted sort-key
// aggregate's argument back to the grouping key's input expression. These cases
// were RED (the BindingId crash) before the remap fix; the prior populated
// oracles missed them because they used `count(*)` (no arg) or a DISTINCT-var
// arg (`count(you.age)`), never a grouping-key-as-arg.
// =====================================================================

/// RETURN form — `ORDER BY count(x)` where `x` is the grouping key projected
/// under its own name. Counts: 1→2, 2→1, 3→3. Ascending by count ⇒ x=2, x=1,
/// x=3. Asserts the FULL (x, c) rows so the order AND the aggregate value are
/// both pinned (not merely "no crash").
#[test]
fn r1_return_order_by_aggregate_of_own_name_grouping_key() {
    let rows = run_rows("UNWIND [1, 1, 2, 3, 3, 3] AS x RETURN x, count(x) AS c ORDER BY count(x)");
    let pairs: Vec<(i64, i64)> = rows
        .iter()
        .map(|r| match (&r[0], &r[1]) {
            (Value::Integer(x), Value::Integer(c)) => (*x, *c),
            other => panic!("expected (Integer, Integer) row, got {other:?}"),
        })
        .collect();
    assert_eq!(
        pairs,
        vec![(2, 1), (1, 2), (3, 3)],
        "ORDER BY count(x) ascending with x the own-name grouping key + the agg arg"
    );
}

/// RETURN form — `ORDER BY sum(x)` (a non-count aggregate over the own-name
/// grouping key). Sums: 1→2, 2→2, 3→9. Ascending ⇒ x=1, x=2, x=3.
#[test]
fn r1_return_order_by_sum_of_own_name_grouping_key() {
    let rows = run_rows("UNWIND [1, 1, 2, 3, 3, 3] AS x RETURN x, sum(x) AS c ORDER BY sum(x)");
    let pairs: Vec<(i64, i64)> = rows
        .iter()
        .map(|r| match (&r[0], &r[1]) {
            (Value::Integer(x), Value::Integer(c)) => (*x, *c),
            other => panic!("expected (Integer, Integer) row, got {other:?}"),
        })
        .collect();
    assert_eq!(
        pairs,
        vec![(1, 2), (2, 2), (3, 9)],
        "ORDER BY sum(x) ascending over the own-name grouping key"
    );
}

/// RETURN form — `ORDER BY x + count(x)` (the grouping key BOTH outside AND
/// inside the aggregate). Keys: 1+2=3, 2+1=3, 3+3=6. Ascending ⇒ the two key-3
/// groups (x=1, x=2) then x=3. The OUTER `x` must map to the Aggregate output
/// column while the INNER `x` (the agg arg) maps back to the input — proving
/// the remap touches ONLY the argument.
#[test]
fn r1_return_order_by_grouping_key_plus_aggregate_of_own_name() {
    let rows =
        run_rows("UNWIND [1, 1, 2, 3, 3, 3] AS x RETURN x, count(x) AS c ORDER BY x + count(x)");
    let xs = col0_ints(rows);
    assert_eq!(xs.len(), 3, "three groups");
    assert_eq!(xs[2], 3, "x=3 (key 6) sorts last ascending");
    assert!(
        xs[..2].contains(&1) && xs[..2].contains(&2),
        "the two key-3 groups (x=1, x=2) sort first, got {xs:?}"
    );
}

/// WITH form — `WITH x AS x, count(x) AS c ORDER BY count(x)` (the explicit
/// `x AS x` still projects the grouping key under its own name). The deferred
/// WITH input-frame keeps `x` bindable for the aggregate's argument; the
/// lowering remaps it back to the input. Counts: 1→2, 2→1, 3→3 ⇒ x=2, x=1, x=3.
#[test]
fn r1_with_order_by_aggregate_of_own_name_grouping_key() {
    let rows = run_rows(
        "UNWIND [1, 1, 2, 3, 3, 3] AS x WITH x AS x, count(x) AS c ORDER BY count(x) RETURN x",
    );
    assert_eq!(
        col0_ints(rows),
        vec![2, 1, 3],
        "WITH x AS x … ORDER BY count(x) ascending by count"
    );
}
