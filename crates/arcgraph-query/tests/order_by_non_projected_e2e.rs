//! **#864** — ORDER BY by a NON-projected in-scope property (the sister-gap
//! to #857's projected-column ORDER BY), END-TO-END through
//! `QueryEngine::execute`.
//!
//! `RETURN e.id ORDER BY e.n` is valid openCypher — the sort key need NOT
//! appear in RETURN. #857 fixed the case where the key IS a projected column;
//! this fixes the non-projected key via a HIDDEN sort column (computed in the
//! projection, sorted by, then trimmed off). Strong oracle: assert EXACT rows
//! in EXACT order (`==`), with the sort key NOT in the output, and preserve
//! both the #857 projected case and the DISTINCT-with-non-projected-key error.

use arcgraph_core::{LabelId, NodeId, TenantId};
use arcgraph_query::QueryEngine;
use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::executor::value::{NodeView, Value};
use arcgraph_query::semantic::StubCatalogProvider;

const E: u32 = 1;

fn node(id: u64, name: &str, n: i64) -> NodeView {
    NodeView::new(NodeId::new(id), Some(LabelId::new(E)))
        .with_property("id", Value::String(name.to_string()))
        .with_property("n", Value::Integer(n))
}

/// INSERTION order = cherry, apple, banana — deliberately distinct from both
/// alphabetical-by-id (apple, banana, cherry) and n-order (banana, cherry,
/// apple), so an assertion on order is discriminating (a no-op "sort" would
/// leave insertion order, a wrong-column sort would give alphabetical).
fn graph() -> StubExecutorSubstrate {
    StubExecutorSubstrate::new()
        .with_node(TenantId::DEFAULT, node(3, "cherry", 2))
        .with_node(TenantId::DEFAULT, node(1, "apple", 3))
        .with_node(TenantId::DEFAULT, node(2, "banana", 1))
}

fn catalog() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["E"])
        .with_properties(["id", "n"])
}

/// Returns the result rows IN ORDER (single-column id projections).
fn ids_in_order(cypher: &str) -> Vec<String> {
    let res = QueryEngine::new(&catalog())
        .execute(cypher, &graph())
        .unwrap_or_else(|e| panic!("execute `{cypher}`: {e:?}"));
    // The sort key must NOT leak into the output — exactly one column.
    for row in &res.rows {
        assert_eq!(
            row.len(),
            1,
            "sort key must not appear in output for `{cypher}`"
        );
    }
    res.rows
        .into_iter()
        .map(|r| match &r[0] {
            Value::String(s) => s.clone(),
            other => panic!("expected String id, got {other:?}"),
        })
        .collect()
}

fn v(s: &[&str]) -> Vec<String> {
    s.iter().map(|x| x.to_string()).collect()
}

// =====================================================================
// PART A — the #864 bug: ORDER BY a non-projected property. EXACT order.
// =====================================================================

#[test]
fn order_by_non_projected_ascending() {
    // n: apple=3, banana=1, cherry=2 ⇒ asc by n ⇒ banana, cherry, apple.
    assert_eq!(
        ids_in_order("MATCH (e:E) RETURN e.id ORDER BY e.n"),
        v(&["banana", "cherry", "apple"])
    );
}

#[test]
fn order_by_non_projected_descending() {
    assert_eq!(
        ids_in_order("MATCH (e:E) RETURN e.id ORDER BY e.n DESC"),
        v(&["apple", "cherry", "banana"])
    );
}

#[test]
fn order_by_non_projected_then_limit() {
    // Sort happens BEFORE limit; the hidden column is dropped after.
    assert_eq!(
        ids_in_order("MATCH (e:E) RETURN e.id ORDER BY e.n LIMIT 2"),
        v(&["banana", "cherry"])
    );
}

// (No SKIP test: `SKIP` is `NotImplemented` engine-wide at v1.0-α — ADR-038
// §2 D-28, deferred to M4-72 — independent of #864. The LIMIT test above
// covers the sort-before-trim/limit ordering interaction.)

#[test]
fn order_by_non_projected_compound_expr() {
    // A compound non-projected key (`e.n * -1`) sorts as one hidden column —
    // descending by n ⇒ apple(−3), cherry(−2), banana(−1).
    assert_eq!(
        ids_in_order("MATCH (e:E) RETURN e.id ORDER BY e.n * -1"),
        v(&["apple", "cherry", "banana"])
    );
}

// =====================================================================
// PART B — multi-key mixing projected + non-projected.
// =====================================================================

#[test]
fn order_by_non_projected_primary_projected_secondary() {
    // `ORDER BY e.n, e.id`: n is non-projected (hidden), e.id is projected
    // (#857). Primary n ⇒ banana, cherry, apple.
    assert_eq!(
        ids_in_order("MATCH (e:E) RETURN e.id ORDER BY e.n, e.id"),
        v(&["banana", "cherry", "apple"])
    );
}

// =====================================================================
// PART C — no regression: #857 projected-column ORDER BY still works.
// =====================================================================

#[test]
fn order_by_projected_column_unaffected() {
    // `ORDER BY e.id` — the projected column (#857). Alphabetical.
    assert_eq!(
        ids_in_order("MATCH (e:E) RETURN e.id ORDER BY e.id"),
        v(&["apple", "banana", "cherry"])
    );
}

#[test]
fn order_by_projected_alias_unaffected() {
    // `RETURN e.n AS k ORDER BY k` — order by the projected ALIAS (#857).
    let res = QueryEngine::new(&catalog())
        .execute("MATCH (e:E) RETURN e.n AS k ORDER BY k", &graph())
        .expect("execute");
    let ns: Vec<i64> = res
        .rows
        .into_iter()
        .map(|r| match &r[0] {
            Value::Integer(i) => *i,
            other => panic!("got {other:?}"),
        })
        .collect();
    assert_eq!(ns, vec![1, 2, 3]);
}

// =====================================================================
// PART D — preserve the openCypher errors: DISTINCT (and aggregation) may
// NOT order by a non-output value (the hidden column must not smuggle a
// dropped binding past the DISTINCT / GROUP-BY boundary).
// =====================================================================

#[test]
fn distinct_with_non_projected_sort_key_is_error() {
    let r = QueryEngine::new(&catalog())
        .execute("MATCH (e:E) RETURN DISTINCT e.id ORDER BY e.n", &graph());
    assert!(
        r.is_err(),
        "ORDER BY a non-projected key under DISTINCT must error (openCypher), got {r:?}"
    );
}
