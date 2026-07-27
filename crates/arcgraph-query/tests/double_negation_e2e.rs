//! **#1050** — openCypher v9 §boolean `NOT NOT` double-negation,
//! END-TO-END (TCK `expressions/boolean/Boolean4.feature` scenario [2]:
//! `RETURN NOT NOT true AS nnt, NOT NOT false AS nnf, NOT NOT null AS nnn`
//! → `true, false, null`).
//!
//! # ADR-133 §D-4 "Query" active-verification gate
//!
//! Every assertion drives a REAL ArcQL query through the FULL pipeline
//! (`QueryEngine::execute`: parse → bind → type-check → cross-substrate
//! → lower → execute) — the EXACT path the TCK conformance ratchet
//! (`arcgraph-tck/tests/full_eligible_conformance.rs`) uses — and
//! asserts the returned cell equals the **openCypher-golden** value.
//!
//! ## Why this is TWO seams, not one
//!
//! The grammar previously capped `kw_not` at a single optional occurrence
//! (`kw_not?`) on both the `where_not_expr` and `expr_not_expr` rules, so
//! `NOT NOT true` PARSE-FAILED ("expected primary_atom"). Lifting the cap
//! to `kw_not*` alone is INSUFFICIENT: the parser's `parse_not_expr`
//! collapsed N NOTs to a single boolean (`inners.iter().any(..)`), so two
//! NOTs would have folded to ONE `Not` layer → `NOT NOT true` would
//! evaluate to **false** (eval-wrong, latent under the parse-fail). The
//! parser must COUNT the `kw_not` occurrences and build a genuinely nested
//! `Not(Not(x))` AST. This test pins the full parity (`NOT^0..4`) so a
//! grammar-only fix (parse-fail → eval-wrong) cannot pass green.

use arcgraph_query::QueryEngine;
use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::executor::value::Value;
use arcgraph_query::semantic::StubCatalogProvider;

// ---------------------------------------------------------------------
// Helper — bare `RETURN <expr>` over a fresh EMPTY substrate.
// ---------------------------------------------------------------------

/// Execute `cypher`, assert exactly one row + one column, return the cell.
fn cell(cypher: &str) -> Value {
    let catalog = StubCatalogProvider::new();
    let substrate = StubExecutorSubstrate::new();
    let engine = QueryEngine::new(&catalog);
    let rows = engine.execute(cypher, &substrate).expect("execute").rows;
    assert_eq!(rows.len(), 1, "expected exactly one row for `{cypher}`");
    assert_eq!(
        rows[0].len(),
        1,
        "expected exactly one column for `{cypher}`"
    );
    rows[0][0].clone()
}

// =====================================================================
// PART A — Boolean4 [2] golden: `NOT NOT <lit>` is the IDENTITY for the
// 2-valued operands and propagates NULL (3VL). These are the exact three
// cells the TCK scenario [2] returns.
// =====================================================================

#[test]
fn not_not_true_is_true() {
    assert_eq!(cell("RETURN NOT NOT true AS nnt"), Value::Boolean(true));
}

#[test]
fn not_not_false_is_false() {
    assert_eq!(cell("RETURN NOT NOT false AS nnf"), Value::Boolean(false));
}

#[test]
fn not_not_null_is_null() {
    // 3VL: NOT null = null, NOT null = null — NULL propagates through
    // every NOT layer. A parity-bool collapse would NOT change this, but
    // a 2VL projection (null → false) would wrongly yield `true`.
    assert_eq!(cell("RETURN NOT NOT null AS nnn"), Value::Null);
}

// =====================================================================
// PART B — full parity ladder `NOT^0..4`. The DISCRIMINATING oracles:
// an even number of NOTs is the identity; an odd number negates. A naive
// "collapse N NOTs to one boolean" fix passes [2]'s `NOT NOT true` only
// by accident if it collapses to false — it does NOT; it collapses to
// `Not(true)` = false (eval-WRONG). The triple/quad cases below pin that
// each NOT layer is a REAL nested AST node, not a parity bit.
// =====================================================================

#[test]
fn single_not_true_is_false() {
    // Regression guard: the single-NOT case must still negate.
    assert_eq!(cell("RETURN NOT true AS r"), Value::Boolean(false));
}

#[test]
fn not_not_not_true_is_false() {
    // 3 NOTs (odd) → negate: Not(Not(Not(true))) = Not(Not(false))
    //   = Not(true) = false.
    assert_eq!(cell("RETURN NOT NOT NOT true AS r"), Value::Boolean(false));
}

#[test]
fn not_not_not_not_true_is_true() {
    // 4 NOTs (even) → identity.
    assert_eq!(
        cell("RETURN NOT NOT NOT NOT true AS r"),
        Value::Boolean(true)
    );
}

#[test]
fn not_not_not_false_is_true() {
    // 3 NOTs (odd) on false → negate → true.
    assert_eq!(cell("RETURN NOT NOT NOT false AS r"), Value::Boolean(true));
}

// =====================================================================
// PART C — the Boolean4 [2] scenario verbatim: all three columns in ONE
// projection (the exact TCK query). Proves the multi-column shape parses
// and each column folds independently.
// =====================================================================

#[test]
fn boolean4_scenario_2_verbatim() {
    let catalog = StubCatalogProvider::new();
    let substrate = StubExecutorSubstrate::new();
    let engine = QueryEngine::new(&catalog);
    let rows = engine
        .execute(
            "RETURN NOT NOT true AS nnt, NOT NOT false AS nnf, NOT NOT null AS nnn",
            &substrate,
        )
        .expect("execute")
        .rows;
    assert_eq!(rows.len(), 1, "exactly one row");
    assert_eq!(rows[0].len(), 3, "three columns");
    assert_eq!(rows[0][0], Value::Boolean(true), "nnt = NOT NOT true");
    assert_eq!(rows[0][1], Value::Boolean(false), "nnf = NOT NOT false");
    assert_eq!(rows[0][2], Value::Null, "nnn = NOT NOT null");
}

// =====================================================================
// PART D — the WHERE-context (dual) ladder. `where_not_expr` must fold
// nested NOTs independently of `expr_not_expr`. `WHERE NOT NOT (n.g >= 2)`
// is the identity of `n.g >= 2`, so it keeps exactly the g>=2 nodes — a
// discriminator distinct from a single NOT (which would keep g<2).
// =====================================================================

#[test]
fn double_not_in_where_is_identity() {
    use arcgraph_core::{LabelId, NodeId, TenantId};
    use arcgraph_query::executor::value::NodeView;

    const LABEL_X: u32 = 1;
    let node = |id: u64| NodeView::new(NodeId::new(id), Some(LabelId::new(LABEL_X)));

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

    // NOT NOT (n.g >= 2) == (n.g >= 2): keeps g=2 and g=3.
    let rows = engine
        .execute(
            "MATCH (n:X) WHERE NOT NOT (n.g >= 2) RETURN n.g",
            &substrate,
        )
        .expect("execute")
        .rows;
    assert_eq!(
        rows.len(),
        2,
        "NOT NOT (g>=2) is the identity → keeps g=2 and g=3"
    );

    // Single NOT (n.g >= 2): keeps g=1 (the complement) — proves NOT NOT
    // is NOT the same fold as a single NOT.
    let rows_single = engine
        .execute("MATCH (n:X) WHERE NOT (n.g >= 2) RETURN n.g", &substrate)
        .expect("execute")
        .rows;
    assert_eq!(
        rows_single.len(),
        1,
        "single NOT (g>=2) keeps only g=1 (the complement)"
    );
    assert_eq!(rows_single[0][0], Value::Integer(1));
}
