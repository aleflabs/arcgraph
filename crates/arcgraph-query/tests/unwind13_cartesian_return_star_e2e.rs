//! TCK `clauses/unwind/Unwind1` [13] "Multiple unwinds after each other"
//! — END-TO-END, plus the openCypher `RETURN *` alphabetical-ordering
//! oracle the [13] failure surfaced.
//!
//! # The bug this pins
//!
//! On pre-fix `main`, sequential `UNWIND`s composed correctly (the
//! cartesian product was right) and `RETURN *` carried every in-scope
//! binding — but it emitted the columns in pipeline-DECLARATION order
//! (`xs, ys, zs, x, y, z`) rather than the openCypher wildcard rule:
//! **ALPHABETICAL by variable name** (Cypher 9 §6.1 — "`RETURN *`
//! returns all variables, in alphabetical order"). So `[13]`, whose
//! expected table is column-ordered `| x | xs | y | ys | z | zs |`,
//! failed `RowsMismatch` purely on column ORDER — every row's VALUES
//! were already correct.
//!
//! Scenario [11] (`WITH [1,2,3] AS list UNWIND list AS x RETURN *`,
//! expected `| list | x |`) masked the bug: `list < x` is already both
//! alphabetical AND declaration order, so the two coincide. [13] is the
//! first multi-variable `RETURN *` where alphabetical ≠ declaration.
//!
//! # Why exact-rows (post-stringify) IS the proof
//!
//! The TCK differ strips the expected table's HEADER and compares the
//! remaining data rows as a multiset of stringified cells, POSITIONALLY
//! within each row. So column order within a row is load-bearing: a
//! correct row multiset under the [13] expected order proves BOTH the
//! cartesian composition (the 8-row set) AND the alphabetical column
//! order. We stringify with the same canonical-cell renderer the TCK
//! conformance harness uses (`render_tck` — lists render `[1, 2]` with a
//! space after the comma, matching the `.feature` table), so this oracle
//! is byte-faithful to the conformance harness.

use arcgraph_core::{LabelId, NodeId, TenantId};
use arcgraph_query::QueryEngine;
use arcgraph_query::executor::value::{NodeView, Value};
use arcgraph_query::executor::{ExecutorSubstrate, StubExecutorSubstrate};
use arcgraph_query::semantic::StubCatalogProvider;

/// Render a `Value` in the TCK canonical cell form — mirrors
/// `crates/arcgraph-tck/tests/full_eligible_conformance.rs::render_tck`
/// for the value kinds this test produces (integers + lists). Lists
/// render `[1, 2]` (space after comma), exactly as the `.feature`
/// expected table writes them, so this oracle is byte-faithful to the
/// conformance harness's row compare.
fn stringify(v: &Value) -> String {
    match v {
        Value::Null => "null".to_string(),
        Value::Boolean(b) => b.to_string(),
        Value::Integer(i) => i.to_string(),
        Value::String(s) => s.clone(),
        Value::List(items) => {
            let inner: Vec<String> = items.iter().map(stringify).collect();
            format!("[{}]", inner.join(", "))
        }
        other => format!("<unrenderable:{other:?}>"),
    }
}

/// Full pipeline → stringified rows (one `Vec<String>` per row, cells in
/// the engine's column order). Panics on any stage error.
fn run_str<S: ExecutorSubstrate>(query: &str, c: &StubCatalogProvider, s: &S) -> Vec<Vec<String>> {
    let engine = QueryEngine::new(c);
    let r = engine.execute(query, s).expect("execute");
    r.rows
        .iter()
        .map(|row| row.iter().map(stringify).collect())
        .collect()
}

/// Multiset (sorted) view of the rows — the TCK "in any order" compare.
fn as_multiset(mut rows: Vec<Vec<String>>) -> Vec<Vec<String>> {
    rows.sort();
    rows
}

// =====================================================================
// THE target: Unwind1 [13] — triple UNWIND cartesian + RETURN *.
// =====================================================================

#[test]
fn unwind1_13_multiple_unwinds_cartesian_return_star() {
    // WITH [1, 2] AS xs, [3, 4] AS ys, [5, 6] AS zs
    // UNWIND xs AS x  UNWIND ys AS y  UNWIND zs AS z
    // RETURN *
    //
    // Expected (TCK table, header `| x | xs | y | ys | z | zs |`) — the
    // 2×2×2 cartesian product, columns ALPHABETICAL:
    let actual = run_str(
        "WITH [1, 2] AS xs, [3, 4] AS ys, [5, 6] AS zs \
         UNWIND xs AS x UNWIND ys AS y UNWIND zs AS z RETURN *",
        &StubCatalogProvider::new(),
        &StubExecutorSubstrate::new(),
    );

    // Expected rows in TCK column order: x | xs | y | ys | z | zs.
    let expected: Vec<Vec<String>> = [
        ["1", "[1, 2]", "3", "[3, 4]", "5", "[5, 6]"],
        ["1", "[1, 2]", "3", "[3, 4]", "6", "[5, 6]"],
        ["1", "[1, 2]", "4", "[3, 4]", "5", "[5, 6]"],
        ["1", "[1, 2]", "4", "[3, 4]", "6", "[5, 6]"],
        ["2", "[1, 2]", "3", "[3, 4]", "5", "[5, 6]"],
        ["2", "[1, 2]", "3", "[3, 4]", "6", "[5, 6]"],
        ["2", "[1, 2]", "4", "[3, 4]", "5", "[5, 6]"],
        ["2", "[1, 2]", "4", "[3, 4]", "6", "[5, 6]"],
    ]
    .iter()
    .map(|r| r.iter().map(|s| s.to_string()).collect())
    .collect();

    assert_eq!(actual.len(), 8, "2×2×2 cartesian product = 8 rows");
    assert_eq!(
        as_multiset(actual),
        as_multiset(expected),
        "rows must match the TCK [13] expected table, including the \
         ALPHABETICAL column order (x, xs, y, ys, z, zs)"
    );
}

// =====================================================================
// Localizing probes — these isolate WHERE the bug was NOT.
// =====================================================================

#[test]
fn double_unwind_cartesian_without_star_is_correct() {
    // `WITH … UNWIND … UNWIND … RETURN x, y` — explicit columns, no
    // wildcard. The cartesian composition was ALWAYS correct; this guards
    // it stays correct (the fix touches only the wildcard path).
    let actual = run_str(
        "WITH [1, 2] AS xs, [3, 4] AS ys \
         UNWIND xs AS x UNWIND ys AS y RETURN x, y",
        &StubCatalogProvider::new(),
        &StubExecutorSubstrate::new(),
    );
    let expected: Vec<Vec<String>> = [["1", "3"], ["1", "4"], ["2", "3"], ["2", "4"]]
        .iter()
        .map(|r| r.iter().map(|s| s.to_string()).collect())
        .collect();
    assert_eq!(as_multiset(actual), as_multiset(expected));
}

#[test]
fn single_unwind_return_star_is_alphabetical() {
    // `WITH [1,2] AS xs UNWIND xs AS x RETURN *` — the simplest
    // multi-column `RETURN *`. Alphabetical order is `x` before `xs`
    // (so column 0 is the unwound scalar, column 1 the list). On pre-fix
    // main this emitted declaration order (`xs, x`) → reversed columns.
    let actual = run_str(
        "WITH [1, 2] AS xs UNWIND xs AS x RETURN *",
        &StubCatalogProvider::new(),
        &StubExecutorSubstrate::new(),
    );
    let expected: Vec<Vec<String>> = [["1", "[1, 2]"], ["2", "[1, 2]"]]
        .iter()
        .map(|r| r.iter().map(|s| s.to_string()).collect())
        .collect();
    assert_eq!(
        as_multiset(actual),
        as_multiset(expected),
        "RETURN * emits `x` (alphabetically first) before `xs`"
    );
}

#[test]
fn unwind1_11_return_star_does_not_prune_context() {
    // Unwind1 [11] — `WITH [1,2,3] AS list UNWIND list AS x RETURN *`,
    // expected `| list | x |`. Here declaration order (`list, x`) AND
    // alphabetical (`list < x`) coincide, so this scenario passed BEFORE
    // and MUST still pass after the reorder (no regression).
    let actual = run_str(
        "WITH [1, 2, 3] AS list UNWIND list AS x RETURN *",
        &StubCatalogProvider::new(),
        &StubExecutorSubstrate::new(),
    );
    let expected: Vec<Vec<String>> = [["[1, 2, 3]", "1"], ["[1, 2, 3]", "2"], ["[1, 2, 3]", "3"]]
        .iter()
        .map(|r| r.iter().map(|s| s.to_string()).collect())
        .collect();
    assert_eq!(as_multiset(actual), as_multiset(expected));
}

// =====================================================================
// RETURN * alphabetical-ordering oracle on NAMED node bindings — proves
// the rule generalizes past UNWIND scalars (Return7 [1] shape, minus the
// path value which is a separate surface).
// =====================================================================

#[test]
fn return_star_orders_named_node_bindings_alphabetically() {
    // Two named accounts bound declaration-order `b` then `a` via a WITH
    // alias swap; `RETURN *` must still emit `a` before `b` (alphabetical,
    // NOT declaration). Proves the reorder keys on the variable NAME, not
    // the pipeline position.
    let cat = StubCatalogProvider::new()
        .with_labels(["Account"])
        .with_properties(["id"]);
    let sub = StubExecutorSubstrate::new().with_node(
        TenantId::DEFAULT,
        NodeView::new(NodeId::new(1), Some(LabelId::new(1))).with_property("id", Value::Integer(7)),
    );
    // MATCH binds `n`; WITH re-aliases to `b` then derives `a` — so the
    // post-WITH declaration order is `b, a`, but RETURN * → `a, b`.
    let actual = run_str(
        "MATCH (n:Account) WITH n.id AS b, n.id + 100 AS a RETURN *",
        &cat,
        &sub,
    );
    assert_eq!(actual.len(), 1, "one matched account");
    // Column order MUST be a(=107) then b(=7) — alphabetical, not the
    // `b, a` declaration order.
    assert_eq!(
        actual[0],
        vec!["107".to_string(), "7".to_string()],
        "RETURN * emits `a` before `b` (alphabetical, not declaration order)"
    );
}
