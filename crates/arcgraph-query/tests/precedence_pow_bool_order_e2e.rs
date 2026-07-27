//! openCypher Precedence cluster conformance: exponentiation and Boolean
//! comparison ordering.
//!
//! Each query drives the full `QueryEngine::execute` path and asserts an exact
//! single-cell oracle. The exponentiation cases pin precedence/associativity
//! (`^` tighter than `*`, left-associative, unary `-` tighter than `^`) and the
//! Boolean cases pin `false < true` comparison ordering.

use arcgraph_query::QueryEngine;
use arcgraph_query::executor::{StubExecutorSubstrate, Value};
use arcgraph_query::semantic::StubCatalogProvider;

fn cell(query: &str) -> Value {
    let catalog = StubCatalogProvider::new();
    let substrate = StubExecutorSubstrate::new();
    let engine = QueryEngine::new(&catalog);
    let rows = engine.execute(query, &substrate).expect("execute").rows;
    assert_eq!(
        rows.len(),
        1,
        "expected exactly 1 row for `{query}`, got {}",
        rows.len()
    );
    assert_eq!(
        rows[0].len(),
        1,
        "expected exactly 1 column for `{query}`, got {}",
        rows[0].len()
    );
    rows[0][0].clone()
}

#[test]
fn exponentiation_returns_float_and_obeys_precedence() {
    assert_eq!(cell("RETURN 4 ^ 3 AS a"), Value::Float(64.0));
    assert_eq!(cell("RETURN -3 ^ 2 AS a"), Value::Float(9.0));
    assert_eq!(cell("RETURN 2 ^ 3 ^ 2 AS a"), Value::Float(64.0));
    assert_eq!(cell("RETURN 4 ^ 3 * 2 AS a"), Value::Float(128.0));
}

#[test]
fn boolean_order_comparisons_follow_false_before_true() {
    assert_eq!(cell("RETURN false >= false AS a"), Value::Boolean(true));
    assert_eq!(cell("RETURN false < true AS a"), Value::Boolean(true));
    assert_eq!(
        cell("RETURN NOT false >= false AS a"),
        Value::Boolean(false)
    );
}
