//! **#621 / Epic #618** — openCypher v9 §3.3.5 (`IN` deep 3VL list
//! membership) + §3.4 (list subscript / slice) END-TO-END verification.
//!
//! # ADR-133 §D-4 "Query" active-verification gate
//!
//! Every assertion drives a REAL ArcQL query through the FULL pipeline
//! (`QueryEngine::execute`: parse → bind → type-check → cross-substrate →
//! lower → execute) against a fresh empty `StubExecutorSubstrate` — the
//! EXACT path the TCK conformance ratchet
//! (`arcgraph-tck/tests/full_eligible_conformance.rs::execute_empty`)
//! uses — and asserts the returned row equals the **openCypher-golden**
//! value from the vendored TCK feature file
//! `crates/arcgraph-tck/tck/features/expressions/list/List5.feature`
//! (scenario number cited per query), NOT merely "no error".
//!
//! # Bare `RETURN` now yields one row
//!
//! These are bare `RETURN <expr>` statements (no MATCH), exactly as the
//! TCK writes them. Per the openCypher v9 §6 unit-relation semantic, a
//! leading `RETURN` evaluates over a single driving row, so each query
//! returns exactly one row. (This slice added that unit-row behavior at
//! the leading-`RETURN` lowering site; pre-slice a bare `RETURN` lowered
//! to `Project(Empty)` and yielded zero rows.)

use arcgraph_query::QueryEngine;
use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::executor::value::Value;
use arcgraph_query::semantic::StubCatalogProvider;

/// Execute `cypher` through the full engine against a fresh EMPTY
/// substrate (no nodes) and return all result rows.
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

fn ints(xs: &[i64]) -> Value {
    Value::List(xs.iter().map(|n| Value::Integer(*n)).collect())
}

// =====================================================================
// PART B — `IN` deep 3VL list equality (List5 [5]-[41])
// =====================================================================

#[test]
fn in_scalar_type_mismatch_is_false_not_null() {
    // [5] `1 IN ['1', 2]` ⇒ false (Integer vs String is a definite
    // mismatch, NOT null).
    assert_eq!(cell("RETURN 1 IN ['1', 2] AS r"), Value::Boolean(false));
}

#[test]
fn in_nested_list_match_is_true() {
    // [10] `[1, 2] IN [1, [1, 2]]` ⇒ true (deep list equality).
    assert_eq!(
        cell("RETURN [1, 2] IN [1, [1, 2]] AS r"),
        Value::Boolean(true)
    );
}

#[test]
fn in_nested_list_order_matters() {
    // [11] `[1, 2] IN [1, [2, 1]]` ⇒ false (order-sensitive).
    assert_eq!(
        cell("RETURN [1, 2] IN [1, [2, 1]] AS r"),
        Value::Boolean(false)
    );
}

#[test]
fn in_nested_list_string_mismatch_is_false() {
    // [6] `[1, 2] IN [1, [1, '2']]` ⇒ false (Integer 2 ≠ String '2').
    assert_eq!(
        cell("RETURN [1, 2] IN [1, [1, '2']] AS r"),
        Value::Boolean(false)
    );
}

#[test]
fn in_found_despite_null_is_true() {
    // [24] `3 IN [1, null, 3]` ⇒ true (definite match short-circuits
    // past the null element).
    assert_eq!(cell("RETURN 3 IN [1, null, 3] AS r"), Value::Boolean(true));
}

#[test]
fn in_not_found_with_null_is_null() {
    // [25] `4 IN [1, null, 3]` ⇒ null (no match + a null comparison).
    assert_eq!(cell("RETURN 4 IN [1, null, 3] AS r"), Value::Null);
}

#[test]
fn in_nested_null_comparison_is_null() {
    // [34] `[1, 2] IN [[null, 2], [1, 3]]` ⇒ null. `[1,2]=[null,2]` is
    // 3VL-unknown (pos0 null, pos1 2=2 true, no definite mismatch);
    // `[1,2]=[1,3]` is false. No true, ≥1 unknown ⇒ null.
    assert_eq!(
        cell("RETURN [1, 2] IN [[null, 2], [1, 3]] AS r"),
        Value::Null
    );
}

#[test]
fn in_identical_list_with_null_element_is_null() {
    // [31] `[1, 2, null] IN [1, [1, 2, null]]` ⇒ null (the matching
    // candidate compares null=null at pos2 ⇒ unknown, not true).
    assert_eq!(
        cell("RETURN [1, 2, null] IN [1, [1, 2, null]] AS r"),
        Value::Null
    );
}

#[test]
fn in_true_match_dominates_earlier_null_candidate() {
    // [32] `[1, 2] IN [[null, 2], [1, 2]]` ⇒ true (second candidate is a
    // definite match; the first's unknown does not block a later true).
    assert_eq!(
        cell("RETURN [1, 2] IN [[null, 2], [1, 2]] AS r"),
        Value::Boolean(true)
    );
}

#[test]
fn in_definite_mismatch_dominates_nested_null() {
    // [28] `[1, 2] IN [[null, 'foo']]` ⇒ false. `[1,2]=[null,'foo']`:
    // pos1 `2='foo'` is a definite type-mismatch ⇒ the whole list cmp is
    // FALSE (definite mismatch dominates the pos0 null) ⇒ no unknown ⇒
    // false.
    assert_eq!(
        cell("RETURN [1, 2] IN [[null, 'foo']] AS r"),
        Value::Boolean(false)
    );
}

#[test]
fn in_different_length_is_false_despite_null() {
    // [23] `[1] IN [[1, null]]` ⇒ false (length 1 ≠ 2 ⇒ definite false,
    // even though the candidate contains a null).
    assert_eq!(
        cell("RETURN [1] IN [[1, null]] AS r"),
        Value::Boolean(false)
    );
}

#[test]
fn in_null_needle_is_null() {
    // [20] `null IN [null]` ⇒ null.
    assert_eq!(cell("RETURN null IN [null] AS r"), Value::Null);
}

#[test]
fn in_empty_list_membership() {
    // [35] `[] IN [[]]` ⇒ true; [36] `[] IN []` ⇒ false.
    assert_eq!(cell("RETURN [] IN [[]] AS r"), Value::Boolean(true));
    assert_eq!(cell("RETURN [] IN [] AS r"), Value::Boolean(false));
}

#[test]
fn in_empty_list_with_trailing_null_is_null() {
    // [40] `[] IN [1, 2, null]` ⇒ null (no `[]` match; null present).
    assert_eq!(cell("RETURN [] IN [1, 2, null] AS r"), Value::Null);
}

#[test]
fn in_nested_empty_lists_match() {
    // [41] `[[], []] IN [1, [[], []]]` ⇒ true.
    assert_eq!(
        cell("RETURN [[], []] IN [1, [[], []]] AS r"),
        Value::Boolean(true)
    );
}

#[test]
fn in_failing_on_non_list_rhs_rejects_at_compile_time() {
    // [42] `1 IN <non-list>` ⇒ SyntaxError InvalidArgumentType at compile
    // time. The type-checker rejects a concrete non-list RHS; the engine
    // surfaces a compile-phase error (NOT a runtime error / rows).
    for invalid in ["true", "123", "123.4", "'foo'", "{x: []}"] {
        let q = format!("RETURN 1 IN {invalid} AS r");
        let catalog = StubCatalogProvider::new();
        let substrate = StubExecutorSubstrate::new();
        let engine = QueryEngine::new(&catalog);
        let res = engine.execute(&q, &substrate);
        assert!(res.is_err(), "`{q}` must be rejected, got {res:?}");
    }
}

// =====================================================================
// `=` / `<>` 3VL nested-null equality (shared `values_equal_3vl`)
// =====================================================================

#[test]
fn equality_nested_null_is_null() {
    // `[1,2] = [null,2]` ⇒ null (3VL); `[1,2,null] = [1,2,null]` ⇒ null;
    // `[1,2] = [1,3]` ⇒ false (definite mismatch).
    assert_eq!(cell("RETURN [1, 2] = [null, 2] AS r"), Value::Null);
    assert_eq!(cell("RETURN [1, 2, null] = [1, 2, null] AS r"), Value::Null);
    assert_eq!(cell("RETURN [1, 2] = [1, 3] AS r"), Value::Boolean(false));
}

// =====================================================================
// PART A — list subscript / slice (List5 [2]/[4] + §3.4 semantics)
// =====================================================================

#[test]
fn subscript_literal_list_index() {
    // `[10, 20, 30][1]` ⇒ 20 (0-based).
    assert_eq!(cell("RETURN [10, 20, 30][1] AS r"), Value::Integer(20));
}

#[test]
fn subscript_negative_index_from_end() {
    // `[10, 20, 30][-1]` ⇒ 30 (last element).
    assert_eq!(cell("RETURN [10, 20, 30][-1] AS r"), Value::Integer(30));
}

#[test]
fn subscript_out_of_range_is_null() {
    // `[10, 20, 30][9]` ⇒ null (out of range, NOT an error).
    assert_eq!(cell("RETURN [10, 20, 30][9] AS r"), Value::Null);
    assert_eq!(cell("RETURN [10, 20, 30][-9] AS r"), Value::Null);
}

#[test]
fn subscript_nested_literal_in_membership() {
    // [2] `3 IN [[1, 2, 3]][0]` ⇒ true (subscript yields the inner list).
    assert_eq!(
        cell("RETURN 3 IN [[1, 2, 3]][0] AS r"),
        Value::Boolean(true)
    );
}

#[test]
fn slice_literal_list() {
    // `[10, 20, 30][0..2]` ⇒ [10, 20] (end exclusive).
    assert_eq!(cell("RETURN [10, 20, 30][0..2] AS r"), ints(&[10, 20]));
}

#[test]
fn slice_open_bounds() {
    // `[10,20,30][..2]` ⇒ [10,20]; `[10,20,30][1..]` ⇒ [20,30];
    // `[10,20,30][..]` ⇒ whole list.
    assert_eq!(cell("RETURN [10, 20, 30][..2] AS r"), ints(&[10, 20]));
    assert_eq!(cell("RETURN [10, 20, 30][1..] AS r"), ints(&[20, 30]));
    assert_eq!(cell("RETURN [10, 20, 30][..] AS r"), ints(&[10, 20, 30]));
}

#[test]
fn slice_negative_bounds() {
    // `[10,20,30,40][1..-1]` ⇒ [20,30] (drop first + last).
    assert_eq!(cell("RETURN [10, 20, 30, 40][1..-1] AS r"), ints(&[20, 30]));
}

#[test]
fn slice_in_membership_literal() {
    // [4] `3 IN [1, 2, 3][0..1]` ⇒ false (`[1,2,3][0..1]` = [1]).
    assert_eq!(
        cell("RETURN 3 IN [1, 2, 3][0..1] AS r"),
        Value::Boolean(false)
    );
}

// =====================================================================
// Sanity: the unit-relation fix — a bare RETURN yields exactly one row.
// =====================================================================

#[test]
fn bare_return_yields_one_row() {
    assert_eq!(cell("RETURN 1 AS x"), Value::Integer(1));
}
