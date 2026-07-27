//! W28 conformance — active end-to-end verification for the additive
//! scalar / string / math / list / type-conversion built-in functions
//! (Task #652, Feature #648, Epic #519).
//!
//! # ADR-133 §D-4 "Query" active-verification gate
//!
//! Every assertion below loads a real fixture (a `StubExecutorSubstrate`
//! Person node carrying String / Integer / Float / List properties),
//! runs a real ArcQL query through the FULL pipeline
//! (`QueryEngine::execute`: parse → bind → type-check → cross-substrate
//! → lower → enumerate → execute), and asserts the returned rows equal
//! the **openCypher-golden** values — NOT merely "no error". The golden
//! values are the exact expected results from the vendored openCypher
//! TCK feature files at
//! `crates/arcgraph-tck/tck/features/expressions/{string,mathematical,list,typeConversion}`
//! (cited per query).
//!
//! # Why MATCH-wrapped, not bare `RETURN f(x)`
//!
//! At v1.0-alpha a `RETURN`-only statement lowers to `Project(Empty)`
//! and `EmptyOp` emits zero rows (a bare `RETURN 1` yields no rows). So
//! every query wraps the projection in `MATCH (n:Person)` over a
//! single-Person fixture to obtain exactly one row to project the
//! function over. The TCK's `RETURN f(...)` form is reproduced verbatim
//! in the projection list; the MATCH only supplies the row.

use arcgraph_core::{LabelId, NodeId, TenantId};
use arcgraph_query::QueryEngine;
use arcgraph_query::executor::value::NodeView;
use arcgraph_query::executor::{StubExecutorSubstrate, Value};
use arcgraph_query::semantic::StubCatalogProvider;

/// Single-Person fixture: `name='alice'` (String), `age=-7` (Integer),
/// `score=12.96` (Float — `sqrt(12.96) == 3.6` exactly in f64),
/// `tags=['a','b','c']` (List).
fn substrate() -> StubExecutorSubstrate {
    StubExecutorSubstrate::new().with_node(
        TenantId::DEFAULT,
        NodeView::new(NodeId::new(1), Some(LabelId::new(1)))
            .with_property("name", Value::String("alice".into()))
            .with_property("age", Value::Integer(-7))
            .with_property("score", Value::Float(12.96))
            .with_property(
                "tags",
                Value::List(vec![
                    Value::String("a".into()),
                    Value::String("b".into()),
                    Value::String("c".into()),
                ]),
            ),
    )
}

fn catalog() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["Person"])
        .with_properties(["name", "age", "score", "tags", "missing"])
}

/// Execute `query` through the full `QueryEngine` pipeline against the
/// single-Person fixture and return all result rows.
fn run(query: &str) -> Vec<Vec<Value>> {
    let s = substrate();
    let c = catalog();
    let engine = QueryEngine::new(&c);
    engine.execute(query, &s).expect("execute").rows
}

/// Execute `query`, asserting exactly one row + one column, and return
/// that single projected cell.
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

fn vstrs(ss: &[&str]) -> Value {
    Value::List(ss.iter().map(|s| Value::String((*s).into())).collect())
}

fn vints(ns: &[i64]) -> Value {
    Value::List(ns.iter().copied().map(Value::Integer).collect())
}

// =====================================================================
// String built-ins (TCK expressions/string/*)
// =====================================================================

#[test]
fn e2e_string_functions() {
    // toUpper / toLower / trim family over a real String property + literal.
    assert_eq!(
        cell("MATCH (n:Person) RETURN toUpper(n.name)"),
        Value::String("ALICE".into())
    );
    assert_eq!(
        cell("MATCH (n:Person) RETURN toLower('aBc')"),
        Value::String("abc".into())
    );
    assert_eq!(
        cell("MATCH (n:Person) RETURN trim('  hi  ')"),
        Value::String("hi".into())
    );
    // TCK String3: reverse('raksO') = 'Oskar'.
    assert_eq!(
        cell("MATCH (n:Person) RETURN reverse('raksO')"),
        Value::String("Oskar".into())
    );
    // TCK String1: substring('0123456789', 1) = '123456789'.
    assert_eq!(
        cell("MATCH (n:Person) RETURN substring('0123456789', 1)"),
        Value::String("123456789".into())
    );
    // substring over the property; left/right; replace.
    assert_eq!(
        cell("MATCH (n:Person) RETURN substring(n.name, 1)"),
        Value::String("lice".into())
    );
    assert_eq!(
        cell("MATCH (n:Person) RETURN left(n.name, 2)"),
        Value::String("al".into())
    );
    assert_eq!(
        cell("MATCH (n:Person) RETURN right(n.name, 3)"),
        Value::String("ice".into())
    );
    assert_eq!(
        cell("MATCH (n:Person) RETURN replace(n.name, 'a', 'A')"),
        Value::String("Alice".into())
    );
}

#[test]
fn e2e_split_tck_string4() {
    // TCK String4: split('one1two', '1') = ['one', 'two'].
    assert_eq!(
        cell("MATCH (n:Person) RETURN split('one1two', '1')"),
        vstrs(&["one", "two"])
    );
}

// =====================================================================
// Math built-ins (TCK expressions/mathematical/*)
// =====================================================================

#[test]
fn e2e_math_functions() {
    // TCK Mathematical11: abs(-1) = 1; abs over a negative Integer property.
    assert_eq!(cell("MATCH (n:Person) RETURN abs(-1)"), Value::Integer(1));
    assert_eq!(
        cell("MATCH (n:Person) RETURN abs(n.age)"),
        Value::Integer(7)
    );
    // TCK Mathematical13: sqrt(12.96) = 3.6 (exact in f64); also over property.
    assert_eq!(
        cell("MATCH (n:Person) RETURN sqrt(12.96)"),
        Value::Float(3.6)
    );
    assert_eq!(
        cell("MATCH (n:Person) RETURN sqrt(n.score)"),
        Value::Float(3.6)
    );
    // sign of the negative property; ceil/floor/round.
    assert_eq!(
        cell("MATCH (n:Person) RETURN sign(n.age)"),
        Value::Integer(-1)
    );
    assert_eq!(cell("MATCH (n:Person) RETURN ceil(0.1)"), Value::Float(1.0));
    assert_eq!(
        cell("MATCH (n:Person) RETURN floor(0.9)"),
        Value::Float(0.0)
    );
    assert_eq!(
        cell("MATCH (n:Person) RETURN round(2.5)"),
        Value::Float(3.0)
    );
    // exact transcendental anchors + constants.
    assert_eq!(cell("MATCH (n:Person) RETURN exp(0)"), Value::Float(1.0));
    assert_eq!(
        cell("MATCH (n:Person) RETURN log10(1000)"),
        Value::Float(3.0)
    );
    assert_eq!(cell("MATCH (n:Person) RETURN cos(0)"), Value::Float(1.0));
    assert_eq!(
        cell("MATCH (n:Person) RETURN pi()"),
        Value::Float(std::f64::consts::PI)
    );
}

// =====================================================================
// Type-conversion built-ins (TCK expressions/typeConversion/*)
// =====================================================================

#[test]
fn e2e_type_conversions() {
    // TCK TypeConversion2: toInteger(82.9) = 82; '1.7' -> 1; 'foo' -> null.
    assert_eq!(
        cell("MATCH (n:Person) RETURN toInteger(82.9)"),
        Value::Integer(82)
    );
    assert_eq!(
        cell("MATCH (n:Person) RETURN toInteger('1.7')"),
        Value::Integer(1)
    );
    assert_eq!(
        cell("MATCH (n:Person) RETURN toInteger('foo')"),
        Value::Null
    );
    // TCK TypeConversion3: toFloat('5') = 5.0; toFloat over Integer property.
    assert_eq!(
        cell("MATCH (n:Person) RETURN toFloat('5')"),
        Value::Float(5.0)
    );
    assert_eq!(
        cell("MATCH (n:Person) RETURN toFloat(n.age)"),
        Value::Float(-7.0)
    );
    // TCK TypeConversion1: toBoolean('true') = true; invalid string -> null.
    assert_eq!(
        cell("MATCH (n:Person) RETURN toBoolean('true')"),
        Value::Boolean(true)
    );
    assert_eq!(
        cell("MATCH (n:Person) RETURN toBoolean('f alse')"),
        Value::Null
    );
    // TCK TypeConversion4: toString(42) = '42'; toString(true) = 'true';
    // toString over Integer property.
    assert_eq!(
        cell("MATCH (n:Person) RETURN toString(42)"),
        Value::String("42".into())
    );
    assert_eq!(
        cell("MATCH (n:Person) RETURN toString(true)"),
        Value::String("true".into())
    );
    assert_eq!(
        cell("MATCH (n:Person) RETURN toString(n.age)"),
        Value::String("-7".into())
    );
}

// =====================================================================
// List / scalar built-ins (TCK expressions/list/*)
// =====================================================================

#[test]
fn e2e_list_functions() {
    // TCK List6: `size` (Any arg-kind) over a real List property = 3.
    assert_eq!(
        cell("MATCH (n:Person) RETURN size(n.tags)"),
        Value::Integer(3)
    );
    // size of a String property = char count.
    assert_eq!(
        cell("MATCH (n:Person) RETURN size(n.name)"),
        Value::Integer(5)
    );
    // head / last / tail (List arg-kind) over a `range(...)` result —
    // statically a List(Integer), so the List arg-kind accepts it.
    // (The stub catalog types `n.tags` as String, so a property arg
    // would FALSE-POSITIVE reject under the pre-existing List
    // constraint — the same conformance edge the math family hit. TCK
    // List7/List8/List9.)
    assert_eq!(
        cell("MATCH (n:Person) RETURN head(range(0, 5))"),
        Value::Integer(0)
    );
    assert_eq!(
        cell("MATCH (n:Person) RETURN last(range(0, 5))"),
        Value::Integer(5)
    );
    assert_eq!(
        cell("MATCH (n:Person) RETURN tail(range(1, 3))"),
        vints(&[2, 3])
    );
    // TCK List11: range(0, 10) inclusive; explicit negative step.
    assert_eq!(
        cell("MATCH (n:Person) RETURN range(0, 10)"),
        vints(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    );
    assert_eq!(
        cell("MATCH (n:Person) RETURN range(10, -10, -3)"),
        vints(&[10, 7, 4, 1, -2, -5, -8])
    );
}

// =====================================================================
// NULL propagation end-to-end (a real absent-property miss -> NULL)
// =====================================================================

#[test]
fn e2e_null_propagation_and_coalesce() {
    // n.missing is absent -> Null; the function propagates NULL.
    assert_eq!(
        cell("MATCH (n:Person) RETURN toUpper(n.missing)"),
        Value::Null
    );
    assert_eq!(cell("MATCH (n:Person) RETURN abs(n.missing)"), Value::Null);
    assert_eq!(cell("MATCH (n:Person) RETURN size(n.missing)"), Value::Null);
    // coalesce falls through the absent property to the literal fallback.
    assert_eq!(
        cell("MATCH (n:Person) RETURN coalesce(n.missing, 'fallback')"),
        Value::String("fallback".into())
    );
    // coalesce picks the present property over the fallback.
    assert_eq!(
        cell("MATCH (n:Person) RETURN coalesce(n.name, 'fallback')"),
        Value::String("alice".into())
    );
}

// =====================================================================
// Multi-function composition + multi-column projection in one query
// =====================================================================

#[test]
fn e2e_multi_function_single_query() {
    // Several new functions in one RETURN, exercised over the fixture in
    // a single end-to-end execution. Golden row: ['ALICE', 7, 3.6, '-7'].
    let rows =
        run("MATCH (n:Person) RETURN toUpper(n.name), abs(n.age), sqrt(n.score), toString(n.age)");
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0],
        vec![
            Value::String("ALICE".into()),
            Value::Integer(7),
            Value::Float(3.6),
            Value::String("-7".into()),
        ]
    );

    // Nested composition: toUpper(substring(reverse(n.name), 1)).
    // reverse('alice')='ecila'; substring(_,1)='cila'; toUpper='CILA'.
    assert_eq!(
        cell("MATCH (n:Person) RETURN toUpper(substring(reverse(n.name), 1))"),
        Value::String("CILA".into())
    );
}
