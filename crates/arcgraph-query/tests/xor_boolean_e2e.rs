//! **#621 / Epic #618** — openCypher v9 §boolean `XOR` exclusive-
//! disjunction (3-valued logic) + the `OR < XOR < AND` precedence
//! ladder, END-TO-END.
//!
//! # ADR-133 §D-4 "Query" active-verification gate
//!
//! Every assertion drives a REAL ArcQL query through the FULL pipeline
//! (`QueryEngine::execute`: parse → bind → type-check → cross-substrate
//! → lower → execute) — the EXACT path the TCK conformance ratchet
//! (`arcgraph-tck/tests/full_eligible_conformance.rs`) uses — and
//! asserts the returned cell equals the **openCypher-golden** value from
//! the vendored TCK feature files
//! `crates/arcgraph-tck/tck/features/expressions/boolean/Boolean3.feature`
//! and `…/expressions/precedence/Precedence1.feature` (scenario cited
//! per query), NOT merely "no error".
//!
//! These complement the direct 3VL truth-table unit tests in
//! `src/executor/three_vl.rs` (which pin `ThreeValued::xor` at the
//! strongest oracle); here we prove the grammar (BOTH the `expr_*` and
//! `where_*` xor-levels), the parser fold, the type-check, and the
//! evaluator all compose correctly through a real query.

use arcgraph_core::{LabelId, NodeId, TenantId};
use arcgraph_query::QueryEngine;
use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::executor::value::{NodeView, Value};
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
/// layer rejects a non-boolean XOR operand, exactly like AND/OR — the
/// `InvalidArgumentType` SyntaxError that Boolean3 [8] requires).
fn expect_rejected(cypher: &str) {
    let catalog = StubCatalogProvider::new();
    let substrate = StubExecutorSubstrate::new();
    let engine = QueryEngine::new(&catalog);
    assert!(
        engine.execute(cypher, &substrate).is_err(),
        "expected `{cypher}` to be rejected (non-boolean XOR operand)"
    );
}

// =====================================================================
// PART A — basic 2-valued truth table (Boolean3 [1], non-null rows)
// =====================================================================

#[test]
fn xor_true_xor_false_is_true() {
    assert_eq!(cell("RETURN true XOR false AS r"), Value::Boolean(true));
}

#[test]
fn xor_true_xor_true_is_false() {
    assert_eq!(cell("RETURN true XOR true AS r"), Value::Boolean(false));
}

#[test]
fn xor_false_xor_false_is_false() {
    assert_eq!(cell("RETURN false XOR false AS r"), Value::Boolean(false));
}

#[test]
fn xor_false_xor_true_is_true() {
    assert_eq!(cell("RETURN false XOR true AS r"), Value::Boolean(true));
}

// =====================================================================
// PART B — 3-valued NULL propagation (Boolean3 [1]; the [5]/[7]
// discriminator). `_ XOR null = null` and `null XOR _ = null` — XOR has
// NO short-circuit, so ANY null operand yields null. These FAIL if XOR
// is implemented as a 2VL projection (which would coerce null → false).
// =====================================================================

#[test]
fn xor_null_xor_true_is_null() {
    assert_eq!(cell("RETURN null XOR true AS r"), Value::Null);
}

#[test]
fn xor_true_xor_null_is_null() {
    assert_eq!(cell("RETURN true XOR null AS r"), Value::Null);
}

#[test]
fn xor_false_xor_null_is_null() {
    assert_eq!(cell("RETURN false XOR null AS r"), Value::Null);
}

#[test]
fn xor_null_xor_null_is_null() {
    assert_eq!(cell("RETURN null XOR null AS r"), Value::Null);
}

// =====================================================================
// PART C — precedence ladder `OR < XOR < AND` (Precedence1 [1]/[2]).
// These are the DISCRIMINATING oracles: they FAIL if the xor-level is
// slotted at the wrong rung of the precedence ladder.
// =====================================================================

#[test]
fn xor_binds_tighter_than_or() {
    // Precedence1 [1]: `true OR true XOR true` parses as
    // `true OR (true XOR true)` = `true OR false` = true.
    // If XOR were at/below OR it would parse as
    // `(true OR true) XOR true` = `true XOR true` = false.
    assert_eq!(
        cell("RETURN true OR true XOR true AS r"),
        Value::Boolean(true),
        "XOR must bind tighter than OR (Precedence1 [1] column a)"
    );
    // The oracle's b/c columns pin the parenthesized groupings.
    assert_eq!(
        cell("RETURN true OR (true XOR true) AS r"),
        Value::Boolean(true),
        "Precedence1 [1] column b"
    );
    assert_eq!(
        cell("RETURN (true OR true) XOR true AS r"),
        Value::Boolean(false),
        "Precedence1 [1] column c"
    );
}

#[test]
fn and_binds_tighter_than_xor() {
    // Precedence1 [2]: `true XOR false AND false` parses as
    // `true XOR (false AND false)` = `true XOR false` = true.
    // If AND were looser than XOR it would parse as
    // `(true XOR false) AND false` = `true AND false` = false.
    assert_eq!(
        cell("RETURN true XOR false AND false AS r"),
        Value::Boolean(true),
        "AND must bind tighter than XOR (Precedence1 [2] column a)"
    );
    assert_eq!(
        cell("RETURN true XOR (false AND false) AS r"),
        Value::Boolean(true),
        "Precedence1 [2] column b"
    );
    assert_eq!(
        cell("RETURN (true XOR false) AND false AS r"),
        Value::Boolean(false),
        "Precedence1 [2] column c"
    );
}

// =====================================================================
// PART D — left-associative chaining (Boolean3 [2]).
// =====================================================================

#[test]
fn xor_chain_is_left_associative() {
    // (true XOR true) XOR true = false XOR true = true.
    assert_eq!(
        cell("RETURN true XOR true XOR true AS r"),
        Value::Boolean(true)
    );
    // (true XOR true) XOR false = false XOR false = false.
    assert_eq!(
        cell("RETURN true XOR true XOR false AS r"),
        Value::Boolean(false)
    );
    // (true XOR false) XOR true = true XOR true = false.
    assert_eq!(
        cell("RETURN true XOR false XOR true AS r"),
        Value::Boolean(false)
    );
    // Any null in the chain → null (no short-circuit).
    assert_eq!(cell("RETURN true XOR true XOR null AS r"), Value::Null);
}

// =====================================================================
// PART E — non-boolean operand is a COMPILE-TIME error (Boolean3 [8]),
// mirroring AND/OR. Only both-non-null cases are asserted here: like
// AND/OR, a NULL operand short-circuits the type-check to Null (so
// `null XOR 'foo'` yields null, not an error — the existing AND/OR
// 3VL-null discipline this slice mirrors exactly).
// =====================================================================

#[test]
fn xor_integer_operand_is_rejected() {
    expect_rejected("RETURN 1 XOR true AS r");
}

#[test]
fn xor_rhs_integer_operand_is_rejected() {
    expect_rejected("RETURN true XOR 123 AS r");
}

#[test]
fn xor_string_operand_is_rejected() {
    expect_rejected("RETURN 'foo' XOR false AS r");
}

// =====================================================================
// PART F — the WHERE-context (dual) ladder. `where_xor_expr` must work
// independently of `expr_xor_expr`; this exercises the WHERE half.
//
// XOR operands must be Boolean (mirroring AND/OR — `is_boolean_compatible`
// is shared across all three operators). A raw property `n.flag` resolves
// to the stub catalog's dynamic property type, so we XOR two COMPARISONS
// (each yields Boolean) — the type-correct way to drive a property-derived
// boolean through the WHERE xor-level.
// =====================================================================

const LABEL_X: u32 = 1;

fn node(id: u64) -> NodeView {
    NodeView::new(NodeId::new(id), Some(LabelId::new(LABEL_X)))
}

#[test]
fn xor_in_where_filters_via_xor_of_comparisons() {
    // Three nodes g=1,2,3. Predicate `(n.g >= 2) XOR (n.g >= 3)`:
    //   g=1: false XOR false = false → dropped
    //   g=2: true  XOR false = true  → KEPT
    //   g=3: true  XOR true  = false → dropped
    // Only g=2 survives — a single-survivor discriminator proving the
    // `where_xor_expr` level parses + folds + evaluates (and that it is
    // distinct from the OR/AND levels, since the result differs from
    // both `OR` (would keep all 3) and `AND` (would keep g=2,g=3)).
    let substrate = StubExecutorSubstrate::new()
        .with_node(
            TenantId::DEFAULT,
            node(1).with_property("g", Value::Integer(1)),
        )
        .with_node(
            TenantId::DEFAULT,
            node(2).with_property("g", Value::Integer(2)),
        )
        .with_node(
            TenantId::DEFAULT,
            node(3).with_property("g", Value::Integer(3)),
        );
    let catalog = StubCatalogProvider::new()
        .with_labels(["X"])
        .with_properties(["g"]);
    let engine = QueryEngine::new(&catalog);
    let rows = engine
        .execute(
            "MATCH (n:X) WHERE (n.g >= 2) XOR (n.g >= 3) RETURN n.g",
            &substrate,
        )
        .expect("execute")
        .rows;
    assert_eq!(rows.len(), 1, "exactly one node passes the XOR predicate");
    assert_eq!(rows[0][0], Value::Integer(2), "the surviving node is g=2");
}
