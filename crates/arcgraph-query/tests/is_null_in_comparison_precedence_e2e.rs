//! openCypher Boolean precedence regression: postfix predicates (`IS NULL`,
//! `IN`) bind tighter than binary comparison operators.
//!
//! The TCK Boolean1/2/5 cluster uses expressions shaped like
//! `(<bool-expr>) IS NULL = (<bool-expr>) IS NULL`. The parser must build
//! `(lhs IS NULL) = (rhs IS NULL)`, not a left-to-right comparison/predicate
//! chain.

use arcgraph_query::QueryEngine;
use arcgraph_query::executor::{StubExecutorSubstrate, Value};
use arcgraph_query::semantic::StubCatalogProvider;

fn rows(query: &str) -> Vec<Vec<Value>> {
    let catalog = StubCatalogProvider::new();
    let substrate = StubExecutorSubstrate::new();
    let engine = QueryEngine::new(&catalog);
    engine.execute(query, &substrate).expect("execute").rows
}

fn cell(query: &str) -> Value {
    let rows = rows(query);
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
fn is_null_postfix_binds_tighter_than_equality() {
    assert_eq!(
        cell("RETURN false IS NULL = false IS NULL AS r"),
        Value::Boolean(true)
    );
    assert_eq!(
        cell("RETURN true IS NULL = false IS NULL AS r"),
        Value::Boolean(true)
    );
    assert_eq!(
        cell("RETURN null IS NULL = false IS NULL AS r"),
        Value::Boolean(false)
    );
}

#[test]
fn boolean1_commutative_null_shape_returns_true_for_every_row() {
    let rows = rows(
        "UNWIND [true,false,null] AS a \
         UNWIND [true,false,null] AS b \
         WITH a,b \
         WHERE a IS NULL OR b IS NULL \
         RETURN a, b, (a AND b) IS NULL = (b AND a) IS NULL AS result",
    );

    assert_eq!(rows.len(), 5, "Boolean1[5] row shape must produce 5 rows");
    for row in rows {
        assert_eq!(row.len(), 3, "expected a, b, result row, got {row:?}");
        assert_eq!(
            row[2],
            Value::Boolean(true),
            "Boolean1[5] result must be true for row {row:?}"
        );
    }
}

#[test]
fn single_sided_is_null_and_in_regressions_still_parse() {
    assert_eq!(
        cell("RETURN false = true IS NULL AS r"),
        Value::Boolean(true)
    );
    assert_eq!(cell("RETURN 3 IN [1,2,3] AS r"), Value::Boolean(true));
    assert_eq!(
        cell("RETURN 3 IN [1,2,3] = (4 IN [1,2,3]) AS r"),
        Value::Boolean(false)
    );
}

#[test]
fn ordinary_comparisons_still_parse() {
    assert_eq!(cell("RETURN 1 + 2 < 3 + 4 AS r"), Value::Boolean(true));
    assert_eq!(cell("RETURN 1 < 2 AS r"), Value::Boolean(true));
}
