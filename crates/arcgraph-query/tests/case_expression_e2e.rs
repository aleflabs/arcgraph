//! **#621 / Epic #618** — openCypher v9 §3.6 `CASE` expression, BOTH the
//! SIMPLE form (`CASE x WHEN v THEN r … ELSE d END`) and the SEARCHED form
//! (`CASE WHEN cond THEN r … ELSE d END`), END-TO-END.
//!
//! # ADR-133 §D-4 "Query" active-verification gate
//!
//! Every assertion drives a REAL ArcQL query through the FULL pipeline
//! (`QueryEngine::execute`: parse → bind → type-check → cross-substrate →
//! lower → execute) — the EXACT path the TCK conformance ratchet
//! (`arcgraph-tck/tests/full_eligible_conformance.rs`) uses — and asserts the
//! returned cell equals the **openCypher-golden** value from the vendored TCK
//! feature file
//! `crates/arcgraph-tck/tck/features/expressions/conditional/Conditional2.feature`
//! ("Conditional2 - Case Expression", Scenario Outline `[1] Simple cases over
//! integers`, all 12 example rows), NOT merely "no error".
//!
//! The two LOAD-BEARING semantics this pins (the discriminating oracles):
//!  1. **Simple-CASE type-mismatch → ELSE, not error.** Conditional2 [1]'s
//!     `'0'` / `true` / `10.1` rows compare a string / bool / float against
//!     integer WHENs; openCypher VALUE equality is a definite NON-match
//!     (`Some(false)`), so each falls through to `ELSE 'something else'`. A
//!     naïve type-checker that rejected the cross-type comparison would
//!     false-reject the primary oracle.
//!  2. **Searched-CASE null condition → not matched.** A `null` WHEN
//!     condition is NOT true under 3VL, so it does not match (the
//!     WHERE-filter discipline) and the CASE falls to ELSE.

use arcgraph_core::{LabelId, NodeId, TenantId};
use arcgraph_query::QueryEngine;
use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::executor::value::{NodeView, Value};
use arcgraph_query::semantic::StubCatalogProvider;

// ---------------------------------------------------------------------
// Helpers — bare `RETURN <expr>` over a fresh EMPTY substrate (the
// Conditional2 "Given an empty graph" precondition).
// ---------------------------------------------------------------------

/// Execute `cypher` through the full engine against an EMPTY substrate and
/// return all result rows.
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

fn s(text: &str) -> Value {
    Value::String(text.to_string())
}

// =====================================================================
// PART A — SIMPLE form: basic match / no-match→ELSE / no-match no-ELSE→null.
// =====================================================================

#[test]
fn simple_case_matches_a_when() {
    // First-arm match, middle-arm match, last-arm match — the equality
    // dispatch returns the FIRST matching THEN.
    assert_eq!(
        cell("RETURN CASE 1 WHEN 0 THEN 'zero' WHEN 1 THEN 'one' ELSE 'other' END AS r"),
        s("one"),
    );
    assert_eq!(
        cell("RETURN CASE 0 WHEN 0 THEN 'zero' WHEN 1 THEN 'one' ELSE 'other' END AS r"),
        s("zero"),
    );
}

#[test]
fn simple_case_no_match_falls_to_else() {
    // 5 matches no WHEN → ELSE.
    assert_eq!(
        cell("RETURN CASE 5 WHEN 0 THEN 'zero' ELSE 'other' END AS r"),
        s("other"),
    );
}

#[test]
fn simple_case_no_match_no_else_is_null() {
    // No matching WHEN and NO ELSE arm ⇒ null (openCypher §3.6).
    assert_eq!(
        cell("RETURN CASE 5 WHEN 0 THEN 'zero' END AS r"),
        Value::Null,
    );
}

#[test]
fn simple_case_first_match_wins_short_circuit() {
    // Two WHENs would match `1`; the FIRST (in source order) wins.
    assert_eq!(
        cell("RETURN CASE 1 WHEN 1 THEN 'first' WHEN 1 THEN 'second' END AS r"),
        s("first"),
    );
}

// =====================================================================
// PART B — SIMPLE form type-mismatch → ELSE (NOT an error). The
// Conditional2 [1] discriminator: `'0'` / `true` / `10.1` vs integer WHENs.
// These FAIL if the type-checker over-constrains the test-vs-WHEN types.
// =====================================================================

#[test]
fn simple_case_string_vs_int_when_is_no_match() {
    // '0' (String) ≠ 0 (Integer) under openCypher VALUE equality → ELSE.
    assert_eq!(
        cell("RETURN CASE '0' WHEN 0 THEN 'int-zero' ELSE 'else' END AS r"),
        s("else"),
    );
}

#[test]
fn simple_case_bool_vs_int_when_is_no_match() {
    // true (Boolean) ≠ 1 (Integer) → ELSE.
    assert_eq!(
        cell("RETURN CASE true WHEN 1 THEN 'one' ELSE 'else' END AS r"),
        s("else"),
    );
}

#[test]
fn simple_case_float_vs_int_when_is_no_match() {
    // 10.1 (Float) ≠ 10 (Integer) → ELSE (10.1 != 10.0 numerically).
    assert_eq!(
        cell("RETURN CASE 10.1 WHEN 10 THEN 'ten' ELSE 'else' END AS r"),
        s("else"),
    );
}

#[test]
fn simple_case_float_that_equals_int_when_matches() {
    // 10.0 (Float) == 10 (Integer) numerically → MATCHES (the converse of
    // the 10.1 row — proves the cross-numeric equality is REAL, not a blanket
    // "different variant ⇒ no match").
    assert_eq!(
        cell("RETURN CASE 10.0 WHEN 10 THEN 'ten' ELSE 'else' END AS r"),
        s("ten"),
    );
}

#[test]
fn simple_case_null_test_matches_no_when() {
    // A null test compares `null = v` ⇒ null (not true) for every WHEN, so no
    // branch matches → ELSE (the safe openCypher behaviour).
    assert_eq!(
        cell("RETURN CASE null WHEN 0 THEN 'zero' ELSE 'else' END AS r"),
        s("else"),
    );
    // …and a null WHEN value likewise never matches a non-null test.
    assert_eq!(
        cell("RETURN CASE 0 WHEN null THEN 'nullmatch' ELSE 'else' END AS r"),
        s("else"),
    );
}

// =====================================================================
// PART C — the PRIMARY ORACLE: Conditional2 [1] "Simple cases over
// integers", all 12 example rows verbatim. Mirrors the vendored feature
// file `expressions/conditional/Conditional2.feature` (the ratchet target).
// =====================================================================

/// Build the exact Conditional2 [1] query for a given `<value>` substitution.
fn conditional2_query(value: &str) -> String {
    format!(
        "RETURN CASE {value} \
            WHEN -10 THEN 'minus ten' \
            WHEN 0 THEN 'zero' \
            WHEN 1 THEN 'one' \
            WHEN 5 THEN 'five' \
            WHEN 10 THEN 'ten' \
            WHEN 3000 THEN 'three thousand' \
            ELSE 'something else' \
          END AS result"
    )
}

#[test]
fn conditional2_simple_cases_over_integers_all_12_golden_rows() {
    // (value, expected result) — the 12 Examples rows, verbatim.
    let golden: [(&str, &str); 12] = [
        ("-10", "minus ten"),
        ("0", "zero"),
        ("1", "one"),
        ("5", "five"),
        ("10", "ten"),
        ("3000", "three thousand"),
        ("-30", "something else"),
        ("3", "something else"),
        ("3001", "something else"),
        ("'0'", "something else"),  // String vs Integer → ELSE
        ("true", "something else"), // Boolean vs Integer → ELSE
        ("10.1", "something else"), // Float vs Integer → ELSE
    ];
    for (value, expected) in golden {
        let q = conditional2_query(value);
        assert_eq!(
            cell(&q),
            s(expected),
            "Conditional2 [1] row `value = {value}` should be `{expected}`",
        );
    }
}

// =====================================================================
// PART D — SEARCHED form: first-true wins / no-match→ELSE / null-cond→ELSE.
// =====================================================================

#[test]
fn searched_case_returns_first_true_branch() {
    // false THEN 'a' (skipped) → true THEN 'b' (taken).
    assert_eq!(
        cell("RETURN CASE WHEN false THEN 'a' WHEN true THEN 'b' ELSE 'c' END AS r"),
        s("b"),
    );
}

#[test]
fn searched_case_no_true_branch_falls_to_else() {
    assert_eq!(
        cell("RETURN CASE WHEN false THEN 'a' ELSE 'c' END AS r"),
        s("c"),
    );
}

#[test]
fn searched_case_null_condition_is_not_matched() {
    // A `null` WHEN condition is NOT true under 3VL (the WHERE-filter
    // discipline) → it does not match → ELSE. This FAILS if the searched
    // form coerces null → false-as-a-value but more importantly if it ever
    // treated a null/Unknown condition as "matched".
    assert_eq!(
        cell("RETURN CASE WHEN null THEN 'a' ELSE 'c' END AS r"),
        s("c"),
    );
}

#[test]
fn searched_case_no_match_no_else_is_null() {
    assert_eq!(
        cell("RETURN CASE WHEN false THEN 'a' END AS r"),
        Value::Null,
    );
}

#[test]
fn searched_case_comparison_conditions() {
    // Conditions are real boolean expressions, evaluated in order.
    assert_eq!(
        cell("RETURN CASE WHEN 1 > 2 THEN 'gt' WHEN 1 < 2 THEN 'lt' ELSE 'eq' END AS r"),
        s("lt"),
    );
}

// =====================================================================
// PART E — DUAL POSITION: CASE inside a WHERE-ladder over a real fixture.
// Proves the construct composes in WHERE context (not just RETURN).
// =====================================================================

const LABEL_X: u32 = 1;

fn node(id: u64) -> NodeView {
    NodeView::new(NodeId::new(id), Some(LabelId::new(LABEL_X)))
}

#[test]
fn searched_case_in_where_filters_rows() {
    // Three nodes a=-1,0,2. Predicate `CASE WHEN n.a > 0 THEN true ELSE false
    // END`:
    //   a=-1 → false → dropped
    //   a=0  → false → dropped
    //   a=2  → true  → KEPT
    let substrate = StubExecutorSubstrate::new()
        .with_node(
            TenantId::DEFAULT,
            node(1).with_property("a", Value::Integer(-1)),
        )
        .with_node(
            TenantId::DEFAULT,
            node(2).with_property("a", Value::Integer(0)),
        )
        .with_node(
            TenantId::DEFAULT,
            node(3).with_property("a", Value::Integer(2)),
        );
    let catalog = StubCatalogProvider::new()
        .with_labels(["X"])
        .with_properties(["a"]);
    let engine = QueryEngine::new(&catalog);
    let rows = engine
        .execute(
            "MATCH (n:X) WHERE CASE WHEN n.a > 0 THEN true ELSE false END RETURN n.a",
            &substrate,
        )
        .expect("execute")
        .rows;
    assert_eq!(rows.len(), 1, "exactly one node passes the CASE predicate");
    assert_eq!(rows[0][0], Value::Integer(2), "the surviving node is a=2");
}

#[test]
fn simple_case_projects_per_row_value() {
    // SIMPLE CASE in RETURN position over a fixture: map each node's `a` to a
    // label. a=1→'one', a=2→'two', a=7→'many'.
    let substrate = StubExecutorSubstrate::new()
        .with_node(
            TenantId::DEFAULT,
            node(1).with_property("a", Value::Integer(1)),
        )
        .with_node(
            TenantId::DEFAULT,
            node(2).with_property("a", Value::Integer(2)),
        )
        .with_node(
            TenantId::DEFAULT,
            node(3).with_property("a", Value::Integer(7)),
        );
    let catalog = StubCatalogProvider::new()
        .with_labels(["X"])
        .with_properties(["a"]);
    let engine = QueryEngine::new(&catalog);
    let mut labels: Vec<Value> = engine
        .execute(
            "MATCH (n:X) RETURN CASE n.a WHEN 1 THEN 'one' WHEN 2 THEN 'two' ELSE 'many' END AS lbl",
            &substrate,
        )
        .expect("execute")
        .rows
        .into_iter()
        .map(|mut r| r.remove(0))
        .collect();
    labels.sort_by_key(|v| match v {
        Value::String(x) => x.clone(),
        _ => String::new(),
    });
    assert_eq!(labels, vec![s("many"), s("one"), s("two")]);
}

// =====================================================================
// PART F — composition: nested CASE + CASE with arithmetic THEN/test.
// =====================================================================

#[test]
fn nested_case_in_then_arm() {
    // The THEN result is itself a CASE — proves `case_expr` recurses through
    // `primary_atom` cleanly (the inner END must not close the outer CASE).
    assert_eq!(
        cell(
            "RETURN CASE WHEN true THEN CASE WHEN false THEN 'x' ELSE 'inner-else' END \
             ELSE 'outer-else' END AS r"
        ),
        s("inner-else"),
    );
}

#[test]
fn simple_case_with_arithmetic_test_and_result() {
    // Test is an arithmetic expression (`1 + 1` = 2); THEN is arithmetic too.
    assert_eq!(
        cell("RETURN CASE 1 + 1 WHEN 2 THEN 10 * 2 ELSE 0 END AS r"),
        Value::Integer(20),
    );
}

#[test]
fn case_short_circuits_non_taken_then() {
    // The non-taken ELSE arm `1 / 0` would error if evaluated — short-circuit
    // means only the matching THEN runs. Proves we do NOT eagerly evaluate
    // every branch.
    assert_eq!(
        cell("RETURN CASE WHEN true THEN 42 ELSE 1 / 0 END AS r"),
        Value::Integer(42),
    );
}
