//! Leading `OPTIONAL MATCH` null-extension e2e (#996-followup, residual of
//! the #771/#996 OPTIONAL-MATCH cluster).
//!
//! openCypher 9 §6.5: a query whose FIRST reading clause is an
//! `OPTIONAL MATCH` (i.e. it has no preceding driving rows) over a graph
//! with no matching pattern must still emit exactly **one** row with every
//! optional variable bound to `null` — the left-outer join is taken against
//! an implicit single-row (unit) driving table, NOT against an empty table.
//!
//! The prior lowering took the `None => filtered` shortcut for a leading
//! OPTIONAL MATCH, producing a bare `Scan(n)` with no driving row, so an
//! empty graph returned **0 rows** (RowsMismatch). This costs 8 eligible
//! TCK scenarios: clauses/match Match7[1], Match7[10]; expressions/graph
//! Graph6[3], Graph6[7], Graph3[7], Graph9[3]; expressions/null Null1[3],
//! Null2[3].
//!
//! These are EXACT-ROW oracles (the strong-oracle bar). The non-empty case
//! pins that the unit-row left side must NOT add a spurious extra row when
//! the right side is non-empty (the count must equal the node count, not
//! node-count + 1).
//!
//! # ADR provenance
//! - ADR-006 amendment-01 §A-2 — OPTIONAL MATCH lowers to left-outer join.
//! - openCypher 9 §6.5 — leading OPTIONAL MATCH over no-match emits one
//!   null-extended row.

use arcgraph_core::{LabelId, NodeId, TenantId};
use arcgraph_query::QueryEngine;
use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::executor::value::{NodeView, Value};
use arcgraph_query::semantic::StubCatalogProvider;

fn cat() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["Person"])
        .with_properties(["name"])
}

/// A truly empty graph: no nodes, no edges.
fn empty_substrate() -> StubExecutorSubstrate {
    StubExecutorSubstrate::new()
}

/// Two nodes, no edges.
fn two_node_substrate() -> StubExecutorSubstrate {
    StubExecutorSubstrate::new()
        .with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(1), Some(LabelId::new(1)))
                .with_property("name", Value::String("a".into())),
        )
        .with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(2), Some(LabelId::new(1)))
                .with_property("name", Value::String("b".into())),
        )
}

fn run(query: &str, substrate: &StubExecutorSubstrate) -> Vec<Vec<Value>> {
    let cat = cat();
    let engine = QueryEngine::new(&cat);
    engine
        .execute(query, substrate)
        .unwrap_or_else(|e| panic!("execute failed for `{query}`: {e:?}"))
        .rows()
        .to_vec()
}

// ---------------------------------------------------------------------
// EMPTY GRAPH — a leading OPTIONAL MATCH must emit exactly ONE row with
// the optional variable bound to NULL (openCypher 9 §6.5).
// ---------------------------------------------------------------------
#[test]
fn leading_optional_match_empty_graph_emits_one_null_row() {
    let rows = run("OPTIONAL MATCH (n) RETURN n", &empty_substrate());
    assert_eq!(
        rows.len(),
        1,
        "leading OPTIONAL MATCH over empty graph must emit exactly 1 row (got {})",
        rows.len()
    );
    assert_eq!(
        rows[0],
        vec![Value::Null],
        "the single row must bind `n` to NULL"
    );
}

#[test]
fn leading_optional_match_empty_graph_property_is_null() {
    let rows = run("OPTIONAL MATCH (n) RETURN n.missing", &empty_substrate());
    assert_eq!(rows.len(), 1, "exactly one null-extended row");
    assert_eq!(
        rows[0],
        vec![Value::Null],
        "`n.missing` on a NULL `n` is NULL (Cypher 3VL)"
    );
}

#[test]
fn leading_optional_match_empty_graph_is_null_predicate_true() {
    let rows = run(
        "OPTIONAL MATCH (n) RETURN n.missing IS NULL",
        &empty_substrate(),
    );
    assert_eq!(rows.len(), 1, "exactly one null-extended row");
    assert_eq!(
        rows[0],
        vec![Value::Boolean(true)],
        "`n.missing IS NULL` must be true when `n` is NULL"
    );
}

#[test]
fn leading_optional_match_empty_graph_labels_is_null() {
    let rows = run("OPTIONAL MATCH (n) RETURN labels(n)", &empty_substrate());
    assert_eq!(rows.len(), 1, "exactly one null-extended row");
    assert_eq!(
        rows[0],
        vec![Value::Null],
        "`labels(null)` is NULL per the eval contract"
    );
}

// ---------------------------------------------------------------------
// NON-EMPTY GRAPH — the unit-row left side must NOT add a spurious extra
// row: 2 nodes ⇒ exactly 2 rows (NOT 3). The real rows pass through.
// ---------------------------------------------------------------------
#[test]
fn leading_optional_match_non_empty_graph_passes_through_real_rows() {
    let rows = run("OPTIONAL MATCH (n) RETURN n.name", &two_node_substrate());
    assert_eq!(
        rows.len(),
        2,
        "2 nodes ⇒ exactly 2 rows; the unit-row left must NOT add a spurious 3rd row (got {})",
        rows.len()
    );
    let mut names: Vec<String> = rows
        .iter()
        .map(|r| match &r[0] {
            Value::String(s) => s.clone(),
            other => panic!("expected String name, got {other:?}"),
        })
        .collect();
    names.sort();
    assert_eq!(
        names,
        vec!["a".to_string(), "b".to_string()],
        "both real nodes must pass through"
    );
}
