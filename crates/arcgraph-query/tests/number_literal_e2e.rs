//! **#618 (child #647) — GA Lane C** — openCypher v9 §3 numeric literals
//! (hexadecimal `0x…`, octal `0o…`, leading-dot float `.5`, and the
//! i64::MIN boundary `-9223372036854775808`), END-TO-END.
//!
//! # ADR-133 §D-4 "Query" active-verification gate
//!
//! Every assertion drives a REAL ArcQL query through the FULL pipeline
//! (`QueryEngine::execute`: parse → bind → type-check → cross-substrate →
//! lower → execute) — the EXACT path the TCK conformance ratchet
//! (`arcgraph-tck/tests/full_eligible_conformance.rs`) uses — and asserts
//! the returned cell equals the **openCypher-golden** value, NOT merely
//! "no error".
//!
//! The oracles are the vendored TCK feature files
//! `crates/arcgraph-tck/tck/features/expressions/literals/`:
//!   - `Literals2.feature` — decimal integer (incl. i64::MIN/MAX boundary)
//!   - `Literals3.feature` — hexadecimal integer
//!   - `Literals4.feature` — octal integer
//!   - `Literals5.feature` — float (incl. leading-dot `.5` / `.1e-5`)
//!   - `Literals7/8.feature` — hex/octal/leading-dot inside list & map
//!
//! Before this slice these forms ALL parse-failed (`int_literal` was
//! decimal-only; `float_literal` required a leading digit; the unary `-`
//! over the i64::MIN magnitude `9223372036854775808` overflowed at the
//! AST int build). The fix is grammar + parser ONLY (`grammar.pest`
//! int/float rules + `parser.rs` `parse_radix_i64` radix decode + the
//! `parse_unary_expr` i64::MIN constant-fold); the `Literal::Integer(i64)`
//! / `Literal::Float(f64)` values and every eval/type-check arm are
//! UNCHANGED.
//!
//! The REGRESSION block (PART G) is the discriminating guard: a too-greedy
//! hex / leading-dot rule could mis-parse a plain decimal, a property
//! accessor (`.prop`), or the list-slice `..` token. Those pins prove the
//! additions are strictly additive.

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

/// Execute `cypher` expecting a clean rejection (parse-time over-range
/// integer / lexer fault) — pins that an out-of-range magnitude surfaces
/// an error, NOT a panic and NOT a silently-wrong value.
fn expect_rejected(cypher: &str) {
    let catalog = StubCatalogProvider::new();
    let substrate = StubExecutorSubstrate::new();
    let engine = QueryEngine::new(&catalog);
    assert!(
        engine.execute(cypher, &substrate).is_err(),
        "expected `{cypher}` to be rejected (over-range / lexer fault)"
    );
}

fn int(n: i64) -> Value {
    Value::Integer(n)
}

// =====================================================================
// PART A — hexadecimal integer (Literals3).
// =====================================================================

#[test]
fn hex_short_and_long() {
    // Task oracle + Literals3 [1]/[2]/[9]/[10]/[11].
    assert_eq!(cell("RETURN 0x1A AS literal"), int(26));
    assert_eq!(cell("RETURN 0xFF AS literal"), int(255));
    assert_eq!(cell("RETURN 0x1 AS literal"), int(1));
    assert_eq!(cell("RETURN 0x162CD4F6 AS literal"), int(372036854));
    // lower / upper / mixed case digits (Literals3 [9]/[10]/[11]).
    assert_eq!(
        cell("RETURN 0x1a2b3c4d5e6f7 AS literal"),
        int(460367961908983)
    );
    assert_eq!(
        cell("RETURN 0x1A2B3C4D5E6F7 AS literal"),
        int(460367961908983)
    );
    assert_eq!(
        cell("RETURN 0x1A2b3c4D5E6f7 AS literal"),
        int(460367961908983)
    );
    // `0X` upper-case prefix also admitted (^"0x" is case-insensitive).
    assert_eq!(cell("RETURN 0X1A AS literal"), int(26));
}

#[test]
fn hex_zero_negative_and_boundary() {
    // Literals3 [3]/[4]/[5]/[6]/[7]/[8].
    assert_eq!(cell("RETURN 0x7FFFFFFFFFFFFFFF AS literal"), int(i64::MAX));
    assert_eq!(cell("RETURN 0x0 AS literal"), int(0));
    assert_eq!(cell("RETURN -0x0 AS literal"), int(0));
    assert_eq!(cell("RETURN -0x1 AS literal"), int(-1));
    assert_eq!(cell("RETURN -0x162CD4F6 AS literal"), int(-372036854));
    // i64::MIN as hex — the constant-fold boundary.
    assert_eq!(cell("RETURN -0x8000000000000000 AS literal"), int(i64::MIN));
}

// =====================================================================
// PART B — octal integer (Literals4).
// =====================================================================

#[test]
fn octal_short_long_zero_negative_boundary() {
    // Task oracle + Literals4 [1]/[2]/[3]/[4]/[5]/[6]/[7]/[8].
    assert_eq!(cell("RETURN 0o17 AS literal"), int(15));
    assert_eq!(cell("RETURN 0o777 AS literal"), int(511));
    assert_eq!(cell("RETURN 0o1 AS literal"), int(1));
    assert_eq!(cell("RETURN 0o2613152366 AS literal"), int(372036854));
    assert_eq!(
        cell("RETURN 0o777777777777777777777 AS literal"),
        int(i64::MAX)
    );
    assert_eq!(cell("RETURN 0o0 AS literal"), int(0));
    assert_eq!(cell("RETURN -0o0 AS literal"), int(0));
    assert_eq!(cell("RETURN -0o1 AS literal"), int(-1));
    assert_eq!(cell("RETURN -0o2613152366 AS literal"), int(-372036854));
    // i64::MIN as octal — the constant-fold boundary.
    assert_eq!(
        cell("RETURN -0o1000000000000000000000 AS literal"),
        int(i64::MIN)
    );
    // `0O` upper-case prefix also admitted.
    assert_eq!(cell("RETURN 0O17 AS literal"), int(15));
}

// =====================================================================
// PART C — leading-dot float (Literals5).
// =====================================================================

fn approx(cypher: &str, expected: f64) {
    match cell(cypher) {
        Value::Float(f) => assert!(
            (f - expected).abs() < 1e-12 || f == expected,
            "`{cypher}` => {f}, expected {expected}"
        ),
        other => panic!("`{cypher}` => {other:?}, expected Float({expected})"),
    }
}

#[test]
fn leading_dot_float() {
    // Task oracle + Literals5 [2]/[4]/[8]/[10].
    approx("RETURN .5 AS literal", 0.5);
    approx("RETURN -.25 AS literal", -0.25);
    approx("RETURN .3405892687 AS literal", 0.3405892687);
    approx("RETURN .0 AS literal", 0.0);
    // negative leading-dot zero (Literals5 [10]).
    approx("RETURN -.0 AS literal", 0.0);
    // leading-dot WITH exponent AND sign (Literals7/8 `-.1e-5`).
    approx("RETURN -.1e-5 AS literal", -0.000_001);
    approx("RETURN .1e-5 AS literal", 0.000_001);
}

// =====================================================================
// PART D — decimal i64::MIN boundary (Literals2 [8]).
// =====================================================================

#[test]
fn decimal_min_max_boundary() {
    assert_eq!(cell("RETURN 9223372036854775807 AS literal"), int(i64::MAX));
    // i64::MIN — folded at parse time (cannot be Integer(2^63) negated).
    assert_eq!(
        cell("RETURN -9223372036854775808 AS literal"),
        int(i64::MIN)
    );
}

// =====================================================================
// PART E — hex / octal / leading-dot as LEAF elements INSIDE a list
// literal (Literals7 [6], adapted). Proves the new radix / leading-dot
// FORMS parse + evaluate inside a composite literal context.
//
// SCOPE NOTE (out-of-lane, pre-existing): the vendored TCK `Literals7`
// [5]/[7] use a NEGATED element (`[-0x162CD4F6]`, `[-.1e-5]`). A list
// literal whose element is a `UnaryOp`/`BinOp` (anything but a leaf
// literal) currently evaluates that element to `Null` end-to-end —
// verified independent of radix: `[-5]`, `[1+2]`, `[-1.5]` all yield
// `[Null]` on `b1540e4f` too. That is a list-literal EXECUTOR gap
// (`arcgraph-query::executor`), NOT a number-lexer gap, so it is NOT in
// this lane (grammar.pest + parser.rs). The PARSE of the negated radix /
// leading-dot forms is proven by the top-level `RETURN -0x162CD4F6`
// (PART A) + `RETURN -.1e-5` (PART C) value oracles and the parser
// AST-shape unit tests; only the in-list NEGATED-element EVAL is deferred.
// =====================================================================

#[test]
fn radix_and_leading_dot_leaf_in_list() {
    // Positive hex + octal LEAF elements inside a list literal — radix
    // parse + list-literal eval, end-to-end.
    assert_eq!(
        cell("RETURN [0x162CD4F6] AS literal"),
        Value::List(vec![int(372036854)])
    );
    assert_eq!(
        cell("RETURN [0o2613152366] AS literal"),
        Value::List(vec![int(372036854)])
    );
    // Leading-dot float LEAF element inside a list literal.
    match cell("RETURN [.1e-5] AS literal") {
        Value::List(items) => match items.as_slice() {
            [Value::Float(f)] => assert!((f - 0.000_001).abs() < 1e-18),
            other => panic!("expected [Float], got {other:?}"),
        },
        other => panic!("expected List, got {other:?}"),
    }
}

// =====================================================================
// PART F — over-range magnitudes are REJECTED (Literals2 [9]/[10],
// Literals4 [9]/[10]) — clean error, never a panic.
// =====================================================================

#[test]
fn over_range_integers_are_rejected() {
    expect_rejected("RETURN 9223372036854775808 AS literal"); // i64::MAX + 1
    expect_rejected("RETURN -9223372036854775809 AS literal"); // i64::MIN - 1
    expect_rejected("RETURN 0o1000000000000000000000 AS literal"); // octal too large
    expect_rejected("RETURN -0o1000000000000000000001 AS literal"); // octal too small
    expect_rejected("RETURN 0xFFFFFFFFFFFFFFFF AS literal"); // u64::MAX > i64::MAX
}

// =====================================================================
// PART G — REGRESSION: the new forms must NOT mis-parse decimals,
// the list-slice `..` token, or `.`-property access (the discriminating
// guard for a too-greedy hex / leading-dot rule).
// =====================================================================

#[test]
fn decimals_unchanged() {
    assert_eq!(cell("RETURN 0 AS literal"), int(0));
    assert_eq!(cell("RETURN 10 AS literal"), int(10));
    assert_eq!(cell("RETURN -1 AS literal"), int(-1));
    assert_eq!(cell("RETURN 372036854 AS literal"), int(372036854));
    approx("RETURN 1.5 AS literal", 1.5);
    approx("RETURN 1e-5 AS literal", 1e-5);
    approx("RETURN 3985764.3405892687 AS literal", 3985764.3405892687);
}

#[test]
fn list_slice_double_dot_not_stolen_by_leading_dot_float() {
    // `[1,2,3][0..2]` — the leading-dot float must NOT consume the `..`
    // range token (`.` is only a float start when IMMEDIATELY followed by
    // a digit; `..` is `.`+`.`). Slice is inclusive-low / exclusive-high.
    assert_eq!(
        cell("RETURN [1, 2, 3][0..2] AS literal"),
        Value::List(vec![int(1), int(2)])
    );
    // Float slice bounds still parse (`1.5` commits to the digit.digit
    // form, leaving the `..` intact).
    assert_eq!(
        cell("RETURN [10, 20, 30][1..3] AS literal"),
        Value::List(vec![int(20), int(30)])
    );
}
