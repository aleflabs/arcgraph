//! GA function-registry — active end-to-end verification (#618).
//!
//! # ADR-133 §D-4 "Query" active-verification gate
//!
//! Three GA-floor read-path fixes, each exercised through the FULL
//! `QueryEngine::execute` pipeline (parse → bind → type-check →
//! cross-substrate → lower → enumerate → execute) against a real
//! single-Person fixture, asserting the returned rows equal the
//! openCypher-golden values — NOT merely "no error":
//!
//! 1. **Case-insensitive function lookup + dispatch.** openCypher
//!    functions are case-insensitive (`range`/`RANGE`/`Range`,
//!    `abs`/`ABS`, `toInteger`/`TOINTEGER` all denote the same
//!    function — openCypher v9 §3). Pre-#618 the registry `lookup` was
//!    case-sensitive, so `RANGE(1,3)` failed type-check with
//!    `unknown function RANGE` even though the engine computes `range`.
//!    The DISCRIMINATING oracle is mis-cased == canonical-cased result.
//! 2. **`properties(node|rel|map)` → property map** (was unregistered).
//! 3. **`keys(map)`** — `keys` now accepts a MAP (was Node/Rel only).
//!
//! # Why MATCH-wrapped, not bare `RETURN f(x)`
//!
//! At v1.0-alpha a `RETURN`-only statement lowers to `Project(Empty)`
//! and `EmptyOp` emits zero rows. So every query wraps the projection
//! in `MATCH (n:Person)` over a single-Person fixture to obtain exactly
//! one row to project the function over.

use std::collections::BTreeMap;

use arcgraph_core::{LabelId, NodeId, TenantId};
use arcgraph_query::QueryEngine;
use arcgraph_query::executor::value::NodeView;
use arcgraph_query::executor::{StubExecutorSubstrate, Value};
use arcgraph_query::semantic::StubCatalogProvider;

/// Single-Person fixture: `name='alice'` (String), `age=30` (Integer).
/// Two scalar properties keep the `properties(n)` golden map exact.
fn substrate() -> StubExecutorSubstrate {
    StubExecutorSubstrate::new().with_node(
        TenantId::DEFAULT,
        NodeView::new(NodeId::new(1), Some(LabelId::new(1)))
            .with_property("name", Value::String("alice".into()))
            .with_property("age", Value::Integer(30)),
    )
}

fn catalog() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["Person"])
        .with_properties(["name", "age"])
}

/// Execute `query` through the full `QueryEngine` pipeline against the
/// single-Person fixture and return all result rows.
fn run(query: &str) -> Vec<Vec<Value>> {
    let s = substrate();
    let c = catalog();
    let engine = QueryEngine::new(&c);
    engine.execute(query, &s).expect("execute").rows
}

/// Execute `query`, asserting exactly one row + one column, return the cell.
fn cell(query: &str) -> Value {
    let rows = run(query);
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

fn vints(ns: &[i64]) -> Value {
    Value::List(ns.iter().copied().map(Value::Integer).collect())
}

fn vstrs(ss: &[&str]) -> Value {
    Value::List(ss.iter().map(|s| Value::String((*s).into())).collect())
}

fn map_of(pairs: &[(&str, Value)]) -> Value {
    let mut m = BTreeMap::new();
    for (k, v) in pairs {
        m.insert((*k).to_string(), v.clone());
    }
    Value::Map(m)
}

// =====================================================================
// 1. Case-insensitive function lookup + dispatch (#618)
//    Discriminating oracle: mis-cased == canonical-cased result.
// =====================================================================

#[test]
fn e2e_case_insensitive_function_calls() {
    // `RANGE(1,3)` resolves + dispatches to `range` -> [1,2,3].
    assert_eq!(
        cell("MATCH (n:Person) RETURN RANGE(1, 3)"),
        vints(&[1, 2, 3])
    );
    // Mixed-case `Range` likewise.
    assert_eq!(
        cell("MATCH (n:Person) RETURN Range(1, 3)"),
        vints(&[1, 2, 3])
    );
    // `Abs(-5)` / `ABS(-5)` -> 5 (Integer-preserving).
    assert_eq!(cell("MATCH (n:Person) RETURN Abs(-5)"), Value::Integer(5));
    assert_eq!(cell("MATCH (n:Person) RETURN ABS(-5)"), Value::Integer(5));
    // `TOINTEGER('42')` -> 42 (canonical is camelCase `toInteger`).
    assert_eq!(
        cell("MATCH (n:Person) RETURN TOINTEGER('42')"),
        Value::Integer(42)
    );
    // `TOUPPER` over the name property -> 'ALICE'.
    assert_eq!(
        cell("MATCH (n:Person) RETURN TOUPPER(n.name)"),
        Value::String("ALICE".into())
    );
}

#[test]
fn e2e_miscased_equals_canonical_cased() {
    // THE discriminating proof: each function returns the SAME value for
    // its canonical casing and a mis-casing.
    assert_eq!(
        cell("MATCH (n:Person) RETURN range(1, 3)"),
        cell("MATCH (n:Person) RETURN RANGE(1, 3)")
    );
    assert_eq!(
        cell("MATCH (n:Person) RETURN abs(-5)"),
        cell("MATCH (n:Person) RETURN ABS(-5)")
    );
    assert_eq!(
        cell("MATCH (n:Person) RETURN toInteger('42')"),
        cell("MATCH (n:Person) RETURN TOINTEGER('42')")
    );
}

// =====================================================================
// 2. properties(node|rel|map) (#618)
// =====================================================================

#[test]
fn e2e_properties_on_map_literal() {
    // properties({a:1, b:2}) -> {a:1, b:2} (identity on a map).
    assert_eq!(
        cell("MATCH (n:Person) RETURN properties({a: 1, b: 2})"),
        map_of(&[("a", Value::Integer(1)), ("b", Value::Integer(2))])
    );
    // properties(null) -> null.
    assert_eq!(
        cell("MATCH (n:Person) RETURN properties(null)"),
        Value::Null
    );
    // properties({}) -> {} (empty map).
    assert_eq!(
        cell("MATCH (n:Person) RETURN properties({})"),
        Value::Map(BTreeMap::new())
    );
}

#[test]
fn e2e_properties_on_node() {
    // properties(n) -> the node's full property bag as a Map (sorted).
    assert_eq!(
        cell("MATCH (n:Person) RETURN properties(n)"),
        map_of(&[
            ("age", Value::Integer(30)),
            ("name", Value::String("alice".into())),
        ])
    );
}

// =====================================================================
// 3. keys(map) (#618) — keys now accepts a MAP (was Node/Rel only)
// =====================================================================

#[test]
fn e2e_keys_on_map() {
    // keys({a:1, b:2}) -> ['a','b'] (sorted).
    assert_eq!(
        cell("MATCH (n:Person) RETURN keys({a: 1, b: 2})"),
        vstrs(&["a", "b"])
    );
    // Declaration order is irrelevant — keys are sorted.
    assert_eq!(
        cell("MATCH (n:Person) RETURN keys({b: 1, a: 2})"),
        vstrs(&["a", "b"])
    );
    // keys({}) -> [].
    assert_eq!(cell("MATCH (n:Person) RETURN keys({})"), vstrs(&[]));
    // keys(properties(map)) composes (properties is identity on a map).
    assert_eq!(
        cell("MATCH (n:Person) RETURN keys(properties({x: 1, y: 2, z: 3}))"),
        vstrs(&["x", "y", "z"])
    );
    // keys on the node still works (regression — Node arg unaffected).
    assert_eq!(
        cell("MATCH (n:Person) RETURN keys(n)"),
        vstrs(&["age", "name"])
    );
}

// =====================================================================
// Regression — canonical-cased + unaffected functions still work
// =====================================================================

#[test]
fn e2e_regression_canonical_unaffected() {
    // Canonical lowercase `range` still works.
    assert_eq!(
        cell("MATCH (n:Person) RETURN range(1, 3)"),
        vints(&[1, 2, 3])
    );
    // An unaffected function (`size`) is unchanged.
    assert_eq!(
        cell("MATCH (n:Person) RETURN size([1, 2, 3])"),
        Value::Integer(3)
    );
    // A genuinely-unknown function still errors at type-check (case-fold
    // does NOT admit non-builtins).
    let s = substrate();
    let c = catalog();
    let engine = QueryEngine::new(&c);
    let result = engine.execute("MATCH (n:Person) RETURN no_such_fn(1)", &s);
    assert!(
        result.is_err(),
        "an unknown function must still be rejected"
    );
}
