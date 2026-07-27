//! **#773 G4/G5 (Customer-Zero AML)** — `count(*)` + `count(DISTINCT)` +
//! `collect(DISTINCT)` END-TO-END (openCypher v9 §3 aggregation).
//!
//! # What this pins
//!
//! Before this slice, `count(*)` and `count(DISTINCT x)` PARSE-FAILED
//! (`-32700`): the `function_call` grammar admitted `expression`-only
//! arguments — no `*`, no `DISTINCT`. Customer-Zero AML rollups need them
//! ("the canonical row count"; "distinct counterparties / countries per
//! day"). This file drives REAL ArcQL through the FULL engine
//! (`QueryEngine::execute`: parse → bind → type-check → cross-substrate →
//! lower → execute — the same path the TCK ratchet uses) and asserts the
//! EXACT result rows, not merely "no error".
//!
//! # The load-bearing semantics
//!
//! - `count(*)` counts ROWS, including rows whose counted property is
//!   NULL — distinct from `count(expr)`, which excludes NULL `expr`.
//! - `count(DISTINCT x)` / `collect(DISTINCT x)` deduplicate the non-NULL
//!   values before the count / collect (NULL is excluded, as for the
//!   non-distinct forms).
//! - `count(i)` (var arg), `collect(x)` (non-distinct), and
//!   `RETURN DISTINCT` are UNCHANGED (no regression).
//!
//! # ADR-133 §D-4 "Query" active-verification gate
//!
//! Hermetic (in-process `StubExecutorSubstrate` + `StubCatalogProvider`);
//! exact-row oracles are the openCypher aggregation semantics + the CZ
//! `cz_finance.py` G4/G5 repros.

use arcgraph_core::{LabelId, NodeId, TenantId};
use arcgraph_query::QueryEngine;
use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::executor::value::{NodeView, Value};
use arcgraph_query::semantic::StubCatalogProvider;

// ---------------------------------------------------------------------
// Pipeline helpers
// ---------------------------------------------------------------------

/// CZ AML schema: `Issue` (Label 1, `component`), `Account` (Label 2,
/// `country`). The order of `with_labels` assigns ids monotonically from
/// 1, so `Issue == LabelId(1)`, `Account == LabelId(2)`.
fn cz_catalog() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["Issue", "Account"])
        .with_properties(["component", "country", "v"])
}

fn run(cypher: &str, substrate: &StubExecutorSubstrate) -> Vec<Vec<Value>> {
    let catalog = cz_catalog();
    let engine = QueryEngine::new(&catalog);
    engine.execute(cypher, substrate).expect("execute").rows
}

/// Execute against an EMPTY substrate (for UNWIND-driven queries).
fn run_unwind(cypher: &str) -> Vec<Vec<Value>> {
    run(cypher, &StubExecutorSubstrate::new())
}

fn s_str(v: &str) -> Value {
    Value::String(v.to_string())
}

/// Issues with the given `component` values (NodeIds 1..=n, Label 1).
fn issues_with_components(components: &[Option<&str>]) -> StubExecutorSubstrate {
    let mut s = StubExecutorSubstrate::new();
    for (i, c) in components.iter().enumerate() {
        let mut nv = NodeView::new(NodeId::new((i + 1) as u64), Some(LabelId::new(1)));
        if let Some(c) = c {
            nv = nv.with_property("component", s_str(c));
        }
        s = s.with_node(TenantId::DEFAULT, nv);
    }
    s
}

/// Accounts with the given `country` values (NodeIds 1..=n, Label 2).
fn accounts_with_countries(countries: &[Option<&str>]) -> StubExecutorSubstrate {
    let mut s = StubExecutorSubstrate::new();
    for (i, c) in countries.iter().enumerate() {
        let mut nv = NodeView::new(NodeId::new((i + 1) as u64), Some(LabelId::new(2)));
        if let Some(c) = c {
            nv = nv.with_property("country", s_str(c));
        }
        s = s.with_node(TenantId::DEFAULT, nv);
    }
    s
}

/// Sort result rows by their first cell's canonical debug rendering so a
/// grouped-aggregate assertion does not depend on group-emission order.
fn sorted(mut rows: Vec<Vec<Value>>) -> Vec<Vec<Value>> {
    rows.sort_by_key(|r| format!("{:?}", r.first()));
    rows
}

// =====================================================================
// G4 — count(*) (the canonical row count)
// =====================================================================

#[test]
fn cz773_g4_count_star_total_rows() {
    // `MATCH (i:Issue) RETURN count(*)` → [[3]] (3 Issues).
    let s = issues_with_components(&[Some("auth"), Some("auth"), Some("db")]);
    assert_eq!(
        run("MATCH (i:Issue) RETURN count(*)", &s),
        vec![vec![Value::Integer(3)]]
    );
}

#[test]
fn cz773_g4_count_star_grouped_by_component() {
    // `MATCH (i:Issue) RETURN i.component, count(*)` → grouped row count
    // (auth×2, db×1). Sorted by component for a deterministic oracle.
    let s = issues_with_components(&[Some("auth"), Some("auth"), Some("db")]);
    let rows = sorted(run("MATCH (i:Issue) RETURN i.component, count(*)", &s));
    assert_eq!(
        rows,
        vec![
            vec![s_str("auth"), Value::Integer(2)],
            vec![s_str("db"), Value::Integer(1)],
        ]
    );
}

#[test]
fn cz773_g4_count_star_counts_rows_including_null_property() {
    // count(*) counts ROWS even when the property is NULL — UNWIND of 3
    // maps, one with a NULL `v`. count(*) = 3; count(m.v) = 2.
    assert_eq!(
        run_unwind("UNWIND [{v: 1}, {v: null}, {v: 2}] AS m RETURN count(*)"),
        vec![vec![Value::Integer(3)]],
        "count(*) counts the NULL-property row"
    );
    assert_eq!(
        run_unwind("UNWIND [{v: 1}, {v: null}, {v: 2}] AS m RETURN count(m.v)"),
        vec![vec![Value::Integer(2)]],
        "count(expr) excludes the NULL — the contrast"
    );
}

#[test]
fn cz773_g4_count_star_over_unwind_rows() {
    // count(*) over a 4-element UNWIND → 4 (hermetic, fully deterministic).
    assert_eq!(
        run_unwind("UNWIND [10, 20, 30, 40] AS x RETURN count(*)"),
        vec![vec![Value::Integer(4)]]
    );
}

// =====================================================================
// G5 — count(DISTINCT) / collect(DISTINCT) (distinct counterparties)
// =====================================================================

#[test]
fn cz773_g5_count_distinct_countries() {
    // `MATCH (a:Account) RETURN count(DISTINCT a.country)` over
    // [US, US, UK] → [[2]].
    let s = accounts_with_countries(&[Some("US"), Some("US"), Some("UK")]);
    assert_eq!(
        run("MATCH (a:Account) RETURN count(DISTINCT a.country)", &s),
        vec![vec![Value::Integer(2)]]
    );
}

#[test]
fn cz773_g5_count_distinct_excludes_null() {
    // [US, US, NULL, UK] → 2 (NULL excluded, as for count(expr)).
    let s = accounts_with_countries(&[Some("US"), Some("US"), None, Some("UK")]);
    assert_eq!(
        run("MATCH (a:Account) RETURN count(DISTINCT a.country)", &s),
        vec![vec![Value::Integer(2)]]
    );
}

#[test]
fn cz773_g5_collect_distinct_countries() {
    // `MATCH (a:Account) RETURN collect(DISTINCT a.country)` over
    // [US, US, UK] (scan in NodeId order) → [US, UK] (deduped, first-seen).
    let s = accounts_with_countries(&[Some("US"), Some("US"), Some("UK")]);
    let rows = run("MATCH (a:Account) RETURN collect(DISTINCT a.country)", &s);
    assert_eq!(rows.len(), 1);
    match &rows[0][0] {
        Value::List(items) => assert_eq!(items, &vec![s_str("US"), s_str("UK")]),
        other => panic!("expected List; got {other:?}"),
    }
}

#[test]
fn cz773_g5_count_distinct_over_unwind() {
    // Hermetic: count(DISTINCT) over a string UNWIND.
    assert_eq!(
        run_unwind("UNWIND ['US', 'US', 'UK', 'US'] AS c RETURN count(DISTINCT c)"),
        vec![vec![Value::Integer(2)]]
    );
}

// =====================================================================
// No-regression — count(var) / collect(x) / RETURN DISTINCT unchanged
// =====================================================================

#[test]
fn cz773_regression_count_var_arg() {
    // `count(i)` (var arg) still works → [[3]].
    let s = issues_with_components(&[Some("auth"), Some("auth"), Some("db")]);
    assert_eq!(
        run("MATCH (i:Issue) RETURN count(i)", &s),
        vec![vec![Value::Integer(3)]]
    );
}

#[test]
fn cz773_regression_collect_non_distinct_keeps_duplicates() {
    // `collect(a.country)` (non-distinct) keeps all 3 values.
    let s = accounts_with_countries(&[Some("US"), Some("US"), Some("UK")]);
    let rows = run("MATCH (a:Account) RETURN collect(a.country)", &s);
    match &rows[0][0] {
        Value::List(items) => assert_eq!(items.len(), 3, "non-distinct collect keeps duplicates"),
        other => panic!("expected List; got {other:?}"),
    }
}

#[test]
fn cz773_regression_return_distinct_unchanged() {
    // `RETURN DISTINCT a.country` over [US, US, UK] → 2 distinct rows.
    let s = accounts_with_countries(&[Some("US"), Some("US"), Some("UK")]);
    let rows = run("MATCH (a:Account) RETURN DISTINCT a.country", &s);
    assert_eq!(rows.len(), 2, "RETURN DISTINCT collapses the duplicate US");
}

// =====================================================================
// Validity gate — DISTINCT on a non-aggregating function is rejected
// (no silent discard of the modifier).
// =====================================================================

#[test]
fn cz773_distinct_on_non_aggregate_rejected() {
    // `size(DISTINCT x)` is NOT a valid openCypher form; type-check
    // rejects it rather than silently behaving as `size(x)`.
    let catalog = cz_catalog();
    let engine = QueryEngine::new(&catalog);
    let s = StubExecutorSubstrate::new();
    let r = engine.execute("UNWIND [[1, 2, 3]] AS l RETURN size(DISTINCT l)", &s);
    assert!(
        r.is_err(),
        "size(DISTINCT x) MUST reject (DISTINCT only valid on aggregates), got {r:?}"
    );
}
