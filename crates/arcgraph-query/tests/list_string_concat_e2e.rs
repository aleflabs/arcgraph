//! **#621 / Epic #618** — openCypher v9 §3 `+` overloaded for LIST and
//! STRING concatenation, END-TO-END.
//!
//! # ADR-133 §D-4 "Query" active-verification gate
//!
//! Every assertion drives a REAL ArcQL query through the FULL pipeline
//! (`QueryEngine::execute`: parse → bind → type-check → cross-substrate
//! → lower → execute) — the EXACT path the TCK conformance ratchet
//! (`arcgraph-tck/tests/full_eligible_conformance.rs`) uses — and asserts
//! the returned cell equals the **openCypher-golden** value, NOT merely
//! "no error".
//!
//! The list oracles are the vendored TCK feature files
//! `crates/arcgraph-tck/tck/features/expressions/list/List4.feature`
//! (scenario [1] same-type list concat, [2] list+scalar append) and
//! `…/list/List6.feature` (scenario [3] `size()` of concatenated literal
//! lists). The string-concat + null-propagation + Add-only-scoping rows
//! pin the rest of the openCypher `+` overload contract.
//!
//! `+` is a SINGLE chokepoint: before this slice the `BinOp::Add` arm
//! rejected every non-numeric operand at type-check, blocking BOTH list
//! and string concat. The fix is `Add`-ONLY — `Sub`/`Mul`/`Div`/`Mod`
//! over a list/string still error (PART H pins that), and the numeric
//! `1 + 2` path is byte-identical (PART G pins that).

use arcgraph_query::QueryEngine;
use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::executor::value::Value;
use arcgraph_query::semantic::StubCatalogProvider;

// ---------------------------------------------------------------------
// Helpers — bare `RETURN <expr>` over a fresh EMPTY substrate.
// ---------------------------------------------------------------------

/// Execute `cypher` through the full engine against an EMPTY substrate
/// and return all result rows.
fn run(cypher: &str) -> Vec<Vec<Value>> {
    let catalog = StubCatalogProvider::new();
    let substrate = StubExecutorSubstrate::new();
    let engine = QueryEngine::new(&catalog);
    engine.execute(cypher, &substrate).expect("execute").rows
}

/// Execute `cypher`, assert exactly one row + one column, return the cell.
fn cell(cypher: &str) -> Value {
    let rows = run(cypher);
    assert_eq!(rows.len(), 1, "expected exactly one row for `{cypher}`");
    assert_eq!(
        rows[0].len(),
        1,
        "expected exactly one column for `{cypher}`"
    );
    rows[0][0].clone()
}

/// Execute `cypher` expecting a COMPILE-TIME rejection (the type-check
/// layer rejects the operand shape — used to pin that concat is `Add`-ONLY
/// and that the other arithmetic operators stay numeric-only).
fn expect_rejected(cypher: &str) {
    let catalog = StubCatalogProvider::new();
    let substrate = StubExecutorSubstrate::new();
    let engine = QueryEngine::new(&catalog);
    assert!(
        engine.execute(cypher, &substrate).is_err(),
        "expected `{cypher}` to be rejected at type-check"
    );
}

fn int(n: i64) -> Value {
    Value::Integer(n)
}

fn list(items: Vec<Value>) -> Value {
    Value::List(items)
}

fn s(text: &str) -> Value {
    Value::String(text.to_string())
}

// =====================================================================
// PART A — list + list concatenation (List4 [1]).
// =====================================================================

#[test]
fn list_concat_same_type() {
    // List4 [1]: `RETURN [1, 10, 100] + [4, 5]` → `[1, 10, 100, 4, 5]`.
    assert_eq!(
        cell("RETURN [1, 10, 100] + [4, 5] AS foo"),
        list(vec![int(1), int(10), int(100), int(4), int(5)]),
    );
}

#[test]
fn list_concat_preserves_order_a_then_b() {
    // Concatenation is `a` THEN `b` (NOT a set-union / sorted merge):
    // `[3, 1] + [2]` is `[3, 1, 2]`, order-preserving with duplicates.
    assert_eq!(
        cell("RETURN [3, 1] + [2, 1] AS foo"),
        list(vec![int(3), int(1), int(2), int(1)]),
    );
}

// =====================================================================
// PART B — list + element APPEND (List4 [2]).
// =====================================================================

#[test]
fn list_append_scalar_same_type() {
    // List4 [2]: `RETURN [false, true] + false` → `[false, true, false]`.
    assert_eq!(
        cell("RETURN [false, true] + false AS foo"),
        list(vec![
            Value::Boolean(false),
            Value::Boolean(true),
            Value::Boolean(false),
        ]),
    );
}

#[test]
fn list_append_scalar_heterogeneous() {
    // `[1, 2] + 'x'` appends the string element → `[1, 2, 'x']` (lists are
    // heterogeneous at runtime; element type widens to the permissive
    // sentinel at type-check but the value is exact).
    assert_eq!(
        cell("RETURN [1, 2] + 'x' AS foo"),
        list(vec![int(1), int(2), s("x")]),
    );
}

// =====================================================================
// PART C — element + list PREPEND.
// =====================================================================

#[test]
fn scalar_prepend_to_list() {
    // `RETURN 1 + [2, 3]` → `[1, 2, 3]` (element + list = prepend, NOT
    // numeric add — the chokepoint must route the List rhs to concat).
    assert_eq!(
        cell("RETURN 1 + [2, 3] AS foo"),
        list(vec![int(1), int(2), int(3)]),
    );
}

// =====================================================================
// PART D — empty-list edges.
// =====================================================================

#[test]
fn empty_list_concat_edges() {
    assert_eq!(cell("RETURN [] + [1] AS foo"), list(vec![int(1)]));
    assert_eq!(cell("RETURN [1] + [] AS foo"), list(vec![int(1)]));
    assert_eq!(cell("RETURN [] + [] AS foo"), list(vec![]));
}

// =====================================================================
// PART E — string + string concatenation.
// =====================================================================

#[test]
fn string_concat() {
    // `RETURN 'ab' + 'cd'` → `'abcd'`.
    assert_eq!(cell("RETURN 'ab' + 'cd' AS foo"), s("abcd"));
    // Empty-string edges.
    assert_eq!(cell("RETURN '' + 'x' AS foo"), s("x"));
    assert_eq!(cell("RETURN 'x' + '' AS foo"), s("x"));
    // Left-associative chain.
    assert_eq!(cell("RETURN 'a' + 'b' + 'c' AS foo"), s("abc"));
}

// =====================================================================
// PART F — NULL propagation (openCypher `+` is null-propagating, 3VL).
// `null + x = null` and `x + null = null` for EVERY operand shape — list,
// string, and numeric. These FAIL if null is coerced to an empty
// list/string or to 0.
// =====================================================================

#[test]
fn null_propagation_through_plus() {
    assert_eq!(cell("RETURN null + [1] AS foo"), Value::Null);
    assert_eq!(cell("RETURN [1] + null AS foo"), Value::Null);
    assert_eq!(cell("RETURN null + 'a' AS foo"), Value::Null);
    assert_eq!(cell("RETURN 'a' + null AS foo"), Value::Null);
    assert_eq!(cell("RETURN null + 1 AS foo"), Value::Null);
    assert_eq!(cell("RETURN 1 + null AS foo"), Value::Null);
    assert_eq!(cell("RETURN null + null AS foo"), Value::Null);
}

// =====================================================================
// PART G — numeric `+` REGRESSION: the pre-#621 numeric path is UNCHANGED.
// =====================================================================

#[test]
fn numeric_add_unchanged() {
    assert_eq!(cell("RETURN 1 + 2 AS foo"), int(3));
    assert_eq!(cell("RETURN 1.5 + 2 AS foo"), Value::Float(3.5));
    assert_eq!(cell("RETURN 2 + 1.5 AS foo"), Value::Float(3.5));
    assert_eq!(cell("RETURN 1.5 + 2.5 AS foo"), Value::Float(4.0));
    // Chained numeric add still numeric (no list/string in sight).
    assert_eq!(cell("RETURN 1 + 2 + 3 AS foo"), int(6));
}

// =====================================================================
// PART H — concat is `Add`-ONLY. `Sub`/`Mul`/`Div`/`Mod` over a
// list/string are STILL a type error (the chokepoint must NOT relax the
// other arithmetic operators).
// =====================================================================

#[test]
fn non_add_arithmetic_on_lists_still_rejected() {
    expect_rejected("RETURN [1] - [2] AS foo");
    expect_rejected("RETURN [1] * [2] AS foo");
    expect_rejected("RETURN [1, 2] / [3] AS foo");
    expect_rejected("RETURN [1] % [2] AS foo");
}

#[test]
fn non_add_arithmetic_on_strings_still_rejected() {
    expect_rejected("RETURN 'a' - 'b' AS foo");
    expect_rejected("RETURN 'a' * 'b' AS foo");
    expect_rejected("RETURN 'a' / 'b' AS foo");
    expect_rejected("RETURN 'a' % 'b' AS foo");
}

// =====================================================================
// PART I — `size()` over a concatenation (List6 [3]).
// =====================================================================

#[test]
fn size_of_concatenated_literal_lists() {
    // List6 [3]: `RETURN size([[], []] + [[]])` → `3` (three empty lists).
    assert_eq!(cell("RETURN size([[], []] + [[]]) AS l"), int(3));
}

#[test]
fn size_of_flat_concatenation() {
    // size() over a flat concat: `size([1, 2] + [3, 4, 5])` → `5`.
    assert_eq!(cell("RETURN size([1, 2] + [3, 4, 5]) AS l"), int(5));
}

// =====================================================================
// PART J — nesting / composition: concat results compose with further
// concat and with list functions.
// =====================================================================

#[test]
fn nested_list_concatenation() {
    // `([1] + [2]) + [3]` → `[1, 2, 3]` (concat of a concat result).
    assert_eq!(
        cell("RETURN ([1] + [2]) + [3] AS foo"),
        list(vec![int(1), int(2), int(3)]),
    );
    // Left-associative without parens: `[1] + [2] + [3]` → `[1, 2, 3]`.
    assert_eq!(
        cell("RETURN [1] + [2] + [3] AS foo"),
        list(vec![int(1), int(2), int(3)]),
    );
}

#[test]
fn concat_of_nested_lists_is_not_flattened() {
    // List concat does NOT flatten: `[[1]] + [[2]]` → `[[1], [2]]`, a
    // 2-element list of lists (size 2), NOT `[1, 2]`.
    assert_eq!(
        cell("RETURN [[1]] + [[2]] AS foo"),
        list(vec![list(vec![int(1)]), list(vec![int(2)])]),
    );
    assert_eq!(cell("RETURN size([[1]] + [[2]]) AS l"), int(2));
}
