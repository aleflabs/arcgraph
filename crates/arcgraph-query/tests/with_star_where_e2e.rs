//! **ADR-197 (#802) R1 finding #4** — `WITH * WHERE …` wildcard-passthrough
//! binding, END-TO-END.
//!
//! The ADR-197 part-b root-cause fix #1: on main `UNWIND [1,2,3] AS x
//! WITH * WHERE x > 1 RETURN x` failed `UndeclaredVariable x` — the `WITH
//! *` wildcard carried NO bindings into the post-WITH scope. The fix
//! (`bind_with_clause`, `semantic/binding.rs`) re-declares every in-scope
//! binding into the new scope PRESERVING their original binding ids, so
//! the post-WITH `Filter` resolves to the SAME id the `Project`'s
//! wildcard-passthrough emits — the #767/#749 "predicate resolves to the
//! projection OUTPUT id" discipline, extended to the wildcard.
//!
//! These ship with the fix but had NO committed in-crate test (only the
//! external, non-CI langchain acceptance whose `… UNWIND other … WITH *
//! WHERE …` REL_QUERY needs it). testing strategy: a feature ships with its
//! test. Every oracle drives a REAL query through `QueryEngine::execute`
//! (parse → bind → type-check → cross-substrate → lower → execute) and
//! asserts the EXACT rows.
//!
//! # Why exact-rows IS the binding-id-preservation proof
//!
//! If `WITH *` dropped the binding, the WHERE would `UndeclaredVariable`
//! at bind (no rows). If it carried the binding but under the STALE
//! pre-WITH id (not the projection-output id), the `Filter` keyed on the
//! pre-WITH id would be "missing from row schema" at runtime (the G1
//! `-32006` Eval class — see `cz773_with_where_e2e.rs`). A correct
//! filtered rowset therefore proves BOTH: the wildcard carried the
//! binding AND its id aligns with the projection output.

use arcgraph_core::{LabelId, NodeId, TenantId};
use arcgraph_query::QueryEngine;
use arcgraph_query::executor::value::{NodeView, Value};
use arcgraph_query::executor::{ExecutorSubstrate, StubExecutorSubstrate};
use arcgraph_query::semantic::StubCatalogProvider;

const ACCOUNT: u32 = 1; // first label ⇒ LabelId::new(1)

fn account(id: u64, balance: i64) -> NodeView {
    NodeView::new(NodeId::new(id), Some(LabelId::new(ACCOUNT)))
        .with_property("id", Value::Integer(id as i64))
        .with_property("balance", Value::Integer(balance))
}

/// Three accounts with balances [30000, 60000, 80000]; only the 60k +
/// 80k accounts (ids 2, 3) clear the `> 50000` threshold.
fn account_substrate() -> StubExecutorSubstrate {
    StubExecutorSubstrate::new()
        .with_node(TenantId::DEFAULT, account(1, 30000))
        .with_node(TenantId::DEFAULT, account(2, 60000))
        .with_node(TenantId::DEFAULT, account(3, 80000))
}

fn account_cat() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["Account"])
        .with_properties(["id", "balance"])
}

/// Full pipeline → result rows (panics on any stage error).
fn run<S: ExecutorSubstrate>(query: &str, c: &StubCatalogProvider, s: &S) -> Vec<Vec<Value>> {
    let engine = QueryEngine::new(c);
    engine.execute(query, s).expect("execute").rows
}

fn int(v: &Value) -> i64 {
    match v {
        Value::Integer(n) => *n,
        other => panic!("expected Integer, got {other:?}"),
    }
}

fn col0_sorted(rows: &[Vec<Value>]) -> Vec<i64> {
    let mut v: Vec<i64> = rows
        .iter()
        .map(|r| {
            assert_eq!(r.len(), 1, "expected single-column rows, got {r:?}");
            int(&r[0])
        })
        .collect();
    v.sort_unstable();
    v
}

// =====================================================================
// THE finding-#4 example: UNWIND + WITH * WHERE carries the scalar var.
// =====================================================================

#[test]
fn with_star_carries_unwind_scalar_into_where() {
    // `UNWIND [10,20] AS x WITH * WHERE x > 15 RETURN x` → [[20]]. The
    // wildcard carries `x` into the post-WITH WHERE; the filter selects
    // 20 and DROPS 10 (so it is not a vacuous pass-through). On pre-fix
    // main this `UndeclaredVariable x`'d at bind.
    let rows = run(
        "UNWIND [10, 20] AS x WITH * WHERE x > 15 RETURN x",
        &StubCatalogProvider::new(),
        &StubExecutorSubstrate::new(),
    );
    assert_eq!(rows.len(), 1, "exactly one surviving row; got {rows:?}");
    assert_eq!(int(&rows[0][0]), 20, "WHERE x > 15 selects 20, drops 10");
}

#[test]
fn with_star_then_filter_is_not_a_noop_passthrough() {
    // Stronger non-vacuity: a 4-element list where the filter selects a
    // strict subset (proves the WHERE runs against the projected rows,
    // not a no-op that passes everything). [5,12,18,25] WHERE x >= 15 →
    // {18, 25}.
    let rows = run(
        "UNWIND [5, 12, 18, 25] AS x WITH * WHERE x >= 15 RETURN x",
        &StubCatalogProvider::new(),
        &StubExecutorSubstrate::new(),
    );
    assert_eq!(col0_sorted(&rows), vec![18, 25]);
}

// =====================================================================
// Multi-var: MATCH (a) WITH * WHERE … carries the node binding `a`.
// =====================================================================

#[test]
fn with_star_carries_match_node_binding_into_where() {
    // `MATCH (a:Account) WITH * WHERE a.balance > 50000 RETURN a.id`. The
    // wildcard carries the NODE binding `a` (not a scalar) into the WHERE
    // + the RETURN; the filter selects the 60k + 80k accounts (ids 2, 3).
    let rows = run(
        "MATCH (a:Account) WITH * WHERE a.balance > 50000 RETURN a.id",
        &account_cat(),
        &account_substrate(),
    );
    assert_eq!(col0_sorted(&rows), vec![2, 3]);
}

#[test]
fn with_star_carries_multiple_bindings_into_where_and_return() {
    // Two carried bindings: the node `a` AND a computed alias `bal`.
    // `MATCH (a:Account) WITH a, a.balance AS bal WITH * WHERE bal > 50000
    // RETURN a.id, bal`. The wildcard must carry BOTH `a` and `bal`
    // (preserving their projected ids) so the WHERE resolves `bal` and the
    // RETURN resolves both. Selects accounts 2 (60000) + 3 (80000).
    let mut rows = run(
        "MATCH (a:Account) WITH a, a.balance AS bal \
         WITH * WHERE bal > 50000 \
         RETURN a.id, bal",
        &account_cat(),
        &account_substrate(),
    );
    rows.sort_by_key(|r| int(&r[0]));
    assert_eq!(
        rows.len(),
        2,
        "two accounts clear the threshold; got {rows:?}"
    );
    assert_eq!(
        (int(&rows[0][0]), int(&rows[0][1])),
        (2, 60000),
        "account 2 carries both id and the projected balance alias"
    );
    assert_eq!((int(&rows[1][0]), int(&rows[1][1])), (3, 80000));
}
