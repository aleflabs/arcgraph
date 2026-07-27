//! #1016 — openCypher comparison type semantics, end-to-end.
//!
//! These assertions drive real queries through `QueryEngine::execute`
//! and pin the vendored TCK corpus rows from Comparison2 [3][4][5],
//! Precedence3 [6], and Comparison1 [8]. Incompatible equality is
//! definite false/true; incompatible ordering is null, not an evaluator
//! "incomparable types" error.

use arcgraph_query::QueryEngine;
use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::executor::value::Value;
use arcgraph_query::semantic::StubCatalogProvider;

fn cell(cypher: &str) -> Value {
    let catalog = StubCatalogProvider::new();
    let substrate = StubExecutorSubstrate::new();
    let engine = QueryEngine::new(&catalog);
    let rows = engine
        .execute(cypher, &substrate)
        .unwrap_or_else(|e| panic!("execute `{cypher}`: {e:?}"))
        .rows;
    assert_eq!(rows.len(), 1, "expected one row for `{cypher}`");
    assert_eq!(rows[0].len(), 1, "expected one column for `{cypher}`");
    rows[0][0].clone()
}

#[test]
fn incompatible_ordering_and_equality_follow_open_cypher_3vl() {
    assert_eq!(cell("RETURN 1 > 'a' AS r"), Value::Null);
    assert_eq!(cell("RETURN [1, 2] = true AS r"), Value::Boolean(false));
    assert_eq!(cell("RETURN 1 = 'a' AS r"), Value::Boolean(false));
    assert_eq!(cell("RETURN [1, 2] <> [3, 4] AS r"), Value::Boolean(true));
}

#[test]
fn list_ordering_matches_comparison2_4_rows() {
    assert_eq!(cell("RETURN [1, 0] >= [1] AS r"), Value::Boolean(true));
    assert_eq!(cell("RETURN [1, 2] >= [1, null] AS r"), Value::Null);
    assert_eq!(
        cell("RETURN [1, 2] >= [3, null] AS r"),
        Value::Boolean(false)
    );
}

#[test]
fn precedence3_6_list_comparison_shape() {
    assert_eq!(cell("RETURN [1, 2] = [3, 4] AS r"), Value::Boolean(false));
    assert_eq!(cell("RETURN [1, 2] < [3, 4] AS r"), Value::Boolean(true));
    assert_eq!(
        cell("RETURN [1, 2] < ([3, 4] IN [[3, 4], false]) AS r"),
        Value::Null
    );
}

#[test]
fn nan_comparisons_match_comparison1_8_and_comparison2_5() {
    assert_eq!(cell("RETURN 0.0 / 0.0 = 1 AS r"), Value::Boolean(false));
    assert_eq!(
        cell("RETURN 0.0 / 0.0 <> 0.0 / 0.0 AS r"),
        Value::Boolean(true)
    );
    assert_eq!(cell("RETURN 0.0 / 0.0 > 1 AS r"), Value::Boolean(false));
    assert_eq!(cell("RETURN 0.0 / 0.0 <= 1.0 AS r"), Value::Boolean(false));
    assert_eq!(cell("RETURN 0.0 / 0.0 > 'a' AS r"), Value::Null);
}

#[test]
fn established_comparisons_and_null_propagation_are_preserved() {
    assert_eq!(cell("RETURN 1 < 2 AS r"), Value::Boolean(true));
    assert_eq!(cell("RETURN 1.5 < 2 AS r"), Value::Boolean(true));
    assert_eq!(cell("RETURN 'a' < 'b' AS r"), Value::Boolean(true));
    assert_eq!(cell("RETURN 1 = 1 AS r"), Value::Boolean(true));
    assert_eq!(cell("RETURN [1, 2] = [1, 2] AS r"), Value::Boolean(true));
    assert_eq!(cell("RETURN 1 < null AS r"), Value::Null);
    assert_eq!(cell("RETURN null = 1 AS r"), Value::Null);
}
