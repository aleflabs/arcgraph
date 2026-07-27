//! **#773** — openCypher v9 §3.3.6 string-comparison predicates
//! `STARTS WITH` / `ENDS WITH` / `CONTAINS`, END-TO-END.
//!
//! # ADR-133 §D-4 "Query" active-verification gate
//!
//! Every assertion drives a REAL ArcQL query through the FULL pipeline
//! (`QueryEngine::execute`: parse → bind → type-check → cross-substrate →
//! lower → execute) — the EXACT path the TCK conformance ratchet
//! (`arcgraph-tck/tests/full_eligible_conformance.rs`) uses — and asserts the
//! returned cell/rows equal the **openCypher-golden** value, NOT merely "no
//! error".
//!
//! Golden semantics (openCypher v9 §3.3.6; the runtime oracle for the
//! type-mismatch + precedence rows is the vendored TCK file
//! `crates/arcgraph-tck/tck/features/expressions/precedence/Precedence4.feature`
//! scenario `[4]`):
//!   * prefix / suffix / substring match, CASE-SENSITIVE, codepoint-correct;
//!   * any-operand `null` ⇒ `null` (3-valued logic);
//!   * a NON-string operand ⇒ `null` (NOT an error, NOT `false`);
//!   * comparison-precedence tier — binds TIGHTER than `OR`/`AND`/`XOR`.
//!
//! Coverage spans BOTH grammar ladders: RETURN/projection position
//! (`expr_special_pred`) AND WHERE-filter position (`special_pred`).

use arcgraph_core::{LabelId, NodeId, TenantId};
use arcgraph_query::QueryEngine;
use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::executor::value::{NodeView, Value};
use arcgraph_query::semantic::StubCatalogProvider;

// ---------------------------------------------------------------------
// Helpers — bare `RETURN <expr>` over a fresh EMPTY substrate.
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

// =====================================================================
// PART A — basic prefix / suffix / substring truth (RETURN position).
// Exact boolean VALUES (==), not "no error".
// =====================================================================

#[test]
fn starts_with_true_and_false() {
    assert_eq!(
        cell("RETURN 'hello' STARTS WITH 'he' AS r"),
        Value::Boolean(true)
    );
    assert_eq!(
        cell("RETURN 'hello' STARTS WITH 'x' AS r"),
        Value::Boolean(false)
    );
    // Whole-string prefix and equal-string prefix are both true.
    assert_eq!(
        cell("RETURN 'hello' STARTS WITH 'hello' AS r"),
        Value::Boolean(true)
    );
}

#[test]
fn ends_with_true_and_false() {
    assert_eq!(
        cell("RETURN 'hello' ENDS WITH 'lo' AS r"),
        Value::Boolean(true)
    );
    assert_eq!(
        cell("RETURN 'hello' ENDS WITH 'x' AS r"),
        Value::Boolean(false)
    );
    assert_eq!(
        cell("RETURN 'hello' ENDS WITH 'hello' AS r"),
        Value::Boolean(true)
    );
}

#[test]
fn contains_true_and_false() {
    assert_eq!(
        cell("RETURN 'hello' CONTAINS 'ell' AS r"),
        Value::Boolean(true)
    );
    assert_eq!(
        cell("RETURN 'hello' CONTAINS 'xyz' AS r"),
        Value::Boolean(false)
    );
    assert_eq!(
        cell("RETURN 'hello' CONTAINS 'hello' AS r"),
        Value::Boolean(true)
    );
}

#[test]
fn empty_needle_is_always_true() {
    // openCypher: the empty string is a prefix/suffix/substring of every
    // string (Rust `str` predicates agree). Pins the edge so a future
    // "non-empty needle" regression trips.
    assert_eq!(
        cell("RETURN 'hello' STARTS WITH '' AS r"),
        Value::Boolean(true)
    );
    assert_eq!(
        cell("RETURN 'hello' ENDS WITH '' AS r"),
        Value::Boolean(true)
    );
    assert_eq!(
        cell("RETURN 'hello' CONTAINS '' AS r"),
        Value::Boolean(true)
    );
}

// =====================================================================
// PART B — 3-valued NULL propagation. ANY null operand ⇒ null.
// =====================================================================

#[test]
fn null_operand_yields_null() {
    assert_eq!(cell("RETURN null STARTS WITH 'he' AS r"), Value::Null);
    assert_eq!(cell("RETURN 'hello' STARTS WITH null AS r"), Value::Null);
    assert_eq!(cell("RETURN null ENDS WITH 'lo' AS r"), Value::Null);
    assert_eq!(cell("RETURN 'hello' ENDS WITH null AS r"), Value::Null);
    assert_eq!(cell("RETURN null CONTAINS 'ell' AS r"), Value::Null);
    assert_eq!(cell("RETURN 'hello' CONTAINS null AS r"), Value::Null);
    assert_eq!(cell("RETURN null CONTAINS null AS r"), Value::Null);
}

// =====================================================================
// PART C — non-string operand ⇒ null (NOT an error, NOT false). This is
// the openCypher type-mismatch-in-string-predicate rule and the
// load-bearing semantics behind Precedence4 [4]. These FAIL if the
// implementation errors or coerces to false.
// =====================================================================

#[test]
fn non_string_operand_yields_null() {
    assert_eq!(cell("RETURN true STARTS WITH 'abc' AS r"), Value::Null);
    assert_eq!(cell("RETURN 'abc' STARTS WITH true AS r"), Value::Null);
    assert_eq!(cell("RETURN 5 CONTAINS 'a' AS r"), Value::Null);
    assert_eq!(cell("RETURN 'abc' ENDS WITH 7 AS r"), Value::Null);
}

// =====================================================================
// PART D — multi-byte UTF-8 (codepoint-correct substring matching).
// =====================================================================

#[test]
fn multibyte_utf8_matches() {
    assert_eq!(
        cell("RETURN 'héllo' CONTAINS 'éll' AS r"),
        Value::Boolean(true)
    );
    assert_eq!(
        cell("RETURN 'héllo' STARTS WITH 'hé' AS r"),
        Value::Boolean(true)
    );
    assert_eq!(
        cell("RETURN 'héllo' ENDS WITH 'llo' AS r"),
        Value::Boolean(true)
    );
    // A 4-byte codepoint (emoji) substring.
    assert_eq!(
        cell("RETURN 'a😀b' CONTAINS '😀' AS r"),
        Value::Boolean(true)
    );
    assert_eq!(
        cell("RETURN 'a😀b' STARTS WITH 'a😀' AS r"),
        Value::Boolean(true)
    );
}

// =====================================================================
// PART E — case-sensitivity (openCypher string predicates are case-
// SENSITIVE; Rust `str` predicates agree).
// =====================================================================

#[test]
fn case_sensitive() {
    assert_eq!(
        cell("RETURN 'Hello' STARTS WITH 'h' AS r"),
        Value::Boolean(false)
    );
    assert_eq!(
        cell("RETURN 'Hello' STARTS WITH 'H' AS r"),
        Value::Boolean(true)
    );
    assert_eq!(
        cell("RETURN 'Hello' CONTAINS 'ELLO' AS r"),
        Value::Boolean(false)
    );
    assert_eq!(
        cell("RETURN 'Hello' ENDS WITH 'O' AS r"),
        Value::Boolean(false)
    );
}

// =====================================================================
// PART F — concatenated / parenthesized RHS parses (RHS = `add_expr`).
// =====================================================================

#[test]
fn concatenated_rhs_operand() {
    // RHS `'pre' + 'fix'` = 'prefix'; confirms the RHS is a full add_expr.
    assert_eq!(
        cell("RETURN 'prefix-rest' STARTS WITH 'pre' + 'fix' AS r"),
        Value::Boolean(true)
    );
}

// =====================================================================
// PART G — precedence: string predicate binds TIGHTER than the binary
// boolean operators. This is the GOLDEN replication of vendored TCK
// `expressions/precedence/Precedence4.feature` scenario [4] — the exact
// eligible scenario this slice flips. Expected a=true, b=null, c=true,
// d=null. These FAIL if the operator is slotted at the wrong precedence
// rung OR if non-string ⇒ null is wrong.
// =====================================================================

#[test]
fn precedence4_scenario_4_golden() {
    let rows = run(
        "RETURN ('abc' STARTS WITH null OR true) = (('abc' STARTS WITH null) OR true) AS a, \
                ('abc' STARTS WITH null OR true) <> ('abc' STARTS WITH (null OR true)) AS b, \
                (true OR null STARTS WITH 'abc') = (true OR (null STARTS WITH 'abc')) AS c, \
                (true OR null STARTS WITH 'abc') <> ((true OR null) STARTS WITH 'abc') AS d",
    );
    assert_eq!(rows.len(), 1, "single row");
    assert_eq!(rows[0].len(), 4, "four columns a,b,c,d");
    assert_eq!(rows[0][0], Value::Boolean(true), "column a");
    assert_eq!(rows[0][1], Value::Null, "column b");
    assert_eq!(rows[0][2], Value::Boolean(true), "column c");
    assert_eq!(rows[0][3], Value::Null, "column d");
}

#[test]
fn binds_tighter_than_or_minimal() {
    // `'abc' STARTS WITH 'a' OR false` must parse as
    // `('abc' STARTS WITH 'a') OR false` = true OR false = true.
    // If STARTS WITH were looser than OR, the RHS `'a' OR false` would be a
    // (boolean) operand to STARTS WITH and the whole thing would differ.
    assert_eq!(
        cell("RETURN 'abc' STARTS WITH 'a' OR false AS r"),
        Value::Boolean(true)
    );
    // Explicit grouping pins the intended parse.
    assert_eq!(
        cell("RETURN ('abc' STARTS WITH 'a') OR false AS r"),
        Value::Boolean(true)
    );
}

// =====================================================================
// PART H — WHERE-filter context (the `special_pred` ladder, distinct from
// the `expr_special_pred` RETURN ladder) over a tiny string-property
// graph. This is the ADR-133 §D-4 ACTIVE-VERIFICATION recipe: build a
// graph, filter through the real path, assert returned rows vs a
// hand-filtered oracle.
// =====================================================================

const LABEL_PERSON: u32 = 1;

fn person(id: u64, name: &str) -> NodeView {
    NodeView::new(NodeId::new(id), Some(LabelId::new(LABEL_PERSON)))
        .with_property("name", Value::String(name.to_string()))
}

/// Names: Alice, Anna, Bob, Carol. Hand-filtered oracles below.
fn people_substrate() -> StubExecutorSubstrate {
    StubExecutorSubstrate::new()
        .with_node(TenantId::DEFAULT, person(1, "Alice"))
        .with_node(TenantId::DEFAULT, person(2, "Anna"))
        .with_node(TenantId::DEFAULT, person(3, "Bob"))
        .with_node(TenantId::DEFAULT, person(4, "Carol"))
}

fn run_people(cypher: &str) -> Vec<String> {
    let substrate = people_substrate();
    let catalog = StubCatalogProvider::new()
        .with_labels(["Person"])
        .with_properties(["name"]);
    let engine = QueryEngine::new(&catalog);
    let rows = engine.execute(cypher, &substrate).expect("execute").rows;
    let mut names: Vec<String> = rows
        .into_iter()
        .map(|r| match &r[0] {
            Value::String(name) => name.clone(),
            other => panic!("expected String name cell, got {other:?}"),
        })
        .collect();
    names.sort(); // multiset → deterministic compare against sorted oracle
    names
}

#[test]
fn where_starts_with_filters() {
    // Oracle: names starting with 'A' = {Alice, Anna}.
    assert_eq!(
        run_people("MATCH (n:Person) WHERE n.name STARTS WITH 'A' RETURN n.name"),
        vec!["Alice".to_string(), "Anna".to_string()]
    );
}

#[test]
fn where_ends_with_filters() {
    // Oracle: names ending with 'b' = {Bob}. (Case-sensitive: 'Bob' ends
    // with 'b', not 'B'.)
    assert_eq!(
        run_people("MATCH (n:Person) WHERE n.name ENDS WITH 'b' RETURN n.name"),
        vec!["Bob".to_string()]
    );
}

#[test]
fn where_contains_filters() {
    // Oracle: names containing 'ar' = {Carol}.
    assert_eq!(
        run_people("MATCH (n:Person) WHERE n.name CONTAINS 'ar' RETURN n.name"),
        vec!["Carol".to_string()]
    );
}

#[test]
fn where_starts_with_no_match_is_empty() {
    // Oracle: names starting with 'Z' = {} (empty, not an error).
    assert_eq!(
        run_people("MATCH (n:Person) WHERE n.name STARTS WITH 'Z' RETURN n.name"),
        Vec::<String>::new()
    );
}

// =====================================================================
// PART I — string predicate as a PROJECTED expression (RETURN position)
// over real node data. Proves the predicate works as a returned Boolean,
// not only as a filter.
// =====================================================================

#[test]
fn projected_string_predicate_over_nodes() {
    let substrate = people_substrate();
    let catalog = StubCatalogProvider::new()
        .with_labels(["Person"])
        .with_properties(["name"]);
    let engine = QueryEngine::new(&catalog);
    let rows = engine
        .execute(
            "MATCH (n:Person) RETURN n.name AS name, n.name STARTS WITH 'A' AS starts_a",
            &substrate,
        )
        .expect("execute")
        .rows;
    // Oracle: (Alice,true) (Anna,true) (Bob,false) (Carol,false).
    let mut got: Vec<(String, Value)> = rows
        .into_iter()
        .map(|r| match (&r[0], &r[1]) {
            (Value::String(name), b) => (name.clone(), b.clone()),
            other => panic!("unexpected row shape {other:?}"),
        })
        .collect();
    got.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(
        got,
        vec![
            ("Alice".to_string(), Value::Boolean(true)),
            ("Anna".to_string(), Value::Boolean(true)),
            ("Bob".to_string(), Value::Boolean(false)),
            ("Carol".to_string(), Value::Boolean(false)),
        ]
    );
}
