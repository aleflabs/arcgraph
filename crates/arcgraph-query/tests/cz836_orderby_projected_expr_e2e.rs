//! **#836 (CZ customer-found, HIGH)** — RETURN-clause `ORDER BY` over a
//! PROJECTED EXPRESSION, END-TO-END with a real node fixture.
//!
//! # The bug this pins
//!
//! The customer's literal query `RETURN p.name ORDER BY p.name` errored at
//! runtime with `Eval("binding BindingId(0) missing from row schema")`.
//! `RETURN p.name AS n ORDER BY n` (alias) and `WITH … ORDER BY` already
//! worked; the gap was ordering by the SAME unaliased expression that was
//! projected. Lowering places `Sort` OVER `Project`, and `Project`
//! replaces the row schema with the projection output ids — so a sort key
//! that references `p`'s PRE-projection binding is "missing from row
//! schema". The binder fix (`semantic/binding.rs`) matches the ORDER BY
//! key structurally against the RETURN projection expressions and resolves
//! it to the projected column's `output_id` (openCypher v9 §6.6: the sort
//! sees the projection OUTPUT — an unaliased expression orders by its
//! implicit column).
//!
//! # ADR-133 §D-4 "Query" active-verification gate
//!
//! This is the node/name active-verification fixture (the
//! `ga_orderby_binding_e2e.rs` PART F suite is the UNWIND-map RED-on-revert
//! proof; this file exercises the customer's EXACT `MATCH (p:Person) RETURN
//! p.name ORDER BY p.name` shape). Every assertion drives a REAL ArcQL
//! query through the FULL pipeline (`QueryEngine::execute`: parse → bind →
//! type-check → cross-substrate → lower → execute) and asserts the EXACT
//! row sequence against a HAND-SORTED oracle — ORDER BY is about ordering,
//! so the oracle verifies the sequence (ASC + DESC), not merely "no error".

use arcgraph_core::{LabelId, NodeId, TenantId};
use arcgraph_query::QueryEngine;
use arcgraph_query::executor::value::{NodeView, Value};
use arcgraph_query::executor::{ExecutorSubstrate, StubExecutorSubstrate};
use arcgraph_query::semantic::StubCatalogProvider;

const PERSON: u32 = 1; // first label ⇒ LabelId::new(1)

fn person(id: u64, name: &str) -> NodeView {
    NodeView::new(NodeId::new(id), Some(LabelId::new(PERSON)))
        .with_property("name", Value::String(name.to_string()))
}

/// Five people whose names are deliberately INSERTED OUT OF ORDER, so a
/// passing oracle proves the sort actually reordered the scan output (the
/// stub scan yields nodes in insertion / NodeId order).
fn person_substrate() -> StubExecutorSubstrate {
    StubExecutorSubstrate::new()
        .with_node(TenantId::DEFAULT, person(1, "Charlie"))
        .with_node(TenantId::DEFAULT, person(2, "Alice"))
        .with_node(TenantId::DEFAULT, person(3, "Eve"))
        .with_node(TenantId::DEFAULT, person(4, "Bob"))
        .with_node(TenantId::DEFAULT, person(5, "Dave"))
}

fn person_cat() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["Person"])
        .with_properties(["name"])
}

/// Full pipeline → result rows (panics on any stage error — so the #836
/// runtime `Eval("binding … missing from row schema")` surfaces here).
fn run<S: ExecutorSubstrate>(query: &str, c: &StubCatalogProvider, s: &S) -> Vec<Vec<Value>> {
    let engine = QueryEngine::new(c);
    engine.execute(query, s).expect("execute").rows
}

/// Assert single-column string rows, return the names IN RESULT ORDER (NOT
/// re-sorted — the order IS the assertion).
fn names_in_order(rows: Vec<Vec<Value>>) -> Vec<String> {
    rows.into_iter()
        .map(|r| {
            assert_eq!(r.len(), 1, "expected single-column rows, got {r:?}");
            match &r[0] {
                Value::String(s) => s.clone(),
                other => panic!("expected String name, got {other:?}"),
            }
        })
        .collect()
}

// =====================================================================
// The customer's EXACT query shape: `RETURN p.name ORDER BY p.name`.
// =====================================================================

#[test]
fn cz836_match_return_name_orderby_name_ascending() {
    let cat = person_cat();
    let sub = person_substrate();
    let rows = run("MATCH (p:Person) RETURN p.name ORDER BY p.name", &cat, &sub);

    // Hand-sorted oracle (ASC, lexicographic) over the inserted names.
    let oracle = vec!["Alice", "Bob", "Charlie", "Dave", "Eve"];
    assert_eq!(
        names_in_order(rows),
        oracle,
        "RETURN p.name ORDER BY p.name must return names in ascending order"
    );
}

#[test]
fn cz836_match_return_name_orderby_name_descending() {
    let cat = person_cat();
    let sub = person_substrate();
    let rows = run(
        "MATCH (p:Person) RETURN p.name ORDER BY p.name DESC",
        &cat,
        &sub,
    );

    // Hand-sorted oracle (DESC) — the exact reverse of the ASC oracle.
    let oracle = vec!["Eve", "Dave", "Charlie", "Bob", "Alice"];
    assert_eq!(
        names_in_order(rows),
        oracle,
        "RETURN p.name ORDER BY p.name DESC must return names in descending order"
    );
}

#[test]
fn cz836_match_return_name_orderby_name_default_equals_ascending() {
    // No direction keyword = ASC (openCypher default); proves the
    // default-direction path resolves the projected expression too.
    let cat = person_cat();
    let sub = person_substrate();
    let asc = run("MATCH (p:Person) RETURN p.name ORDER BY p.name", &cat, &sub);
    assert_eq!(
        names_in_order(asc),
        vec!["Alice", "Bob", "Charlie", "Dave", "Eve"],
    );
}
