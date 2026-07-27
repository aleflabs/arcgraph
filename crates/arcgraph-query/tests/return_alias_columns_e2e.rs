//! **#353 (HIGH critical-path, customer-zero)** — user-provided RETURN
//! aliases surface as `MaterializedResult::columns` (the implicit
//! column-name rule), END-TO-END through `QueryEngine::execute`.
//!
//! # What this pins
//!
//! Before this slice, `MaterializedResult` carried only rows + metrics
//! — no column names — so the wire renderers (MCP `RawQueryRows`, Bolt
//! `RunOutcome::fields`) synthesized `col_0..N`. langchain's
//! `Neo4jGraph` keys result dicts by the column name, so it received
//! `{'col_0': ...}` instead of `{'name': ...}` and mis-bound. This file
//! drives REAL ArcQL through the FULL engine (`QueryEngine::execute`:
//! parse → bind → type-check → cross-substrate → lower → execute — the
//! same path the MCP / Bolt wire surfaces use) and asserts the EXACT
//! `columns` against the openCypher/Neo4j implicit-column-name rule.
//!
//! # The naming rule (openCypher / Neo4j)
//!
//! - bare variable (`RETURN n`) → the variable name (`"n"`);
//! - explicit `AS alias` → the alias;
//! - un-aliased expression (`RETURN n.name`, `count(*)`) → the
//!   verbatim source text (`"n.name"`, `"count(*)"`);
//! - `RETURN *` → EMPTY `columns` (data-dependent width; the wire falls
//!   back to `col_0..N`).
//!
//! # The DISCRIMINATING oracle (RED-without-the-fix)
//!
//! `RETURN n.name AS name` → `columns == ["name"]`. On `origin/main`
//! without this fix `MaterializedResult` had no `columns` field at all
//! (the wire produced `["col_0"]`). The exact-equality assertion here
//! is RED on any regression that drops the alias.
//!
//! # ADR-133 §D-4 "Query" active-verification gate
//!
//! Hermetic (in-process `StubExecutorSubstrate` + `StubCatalogProvider`);
//! the exact-`columns` oracle is the openCypher implicit-column-name
//! rule + the `~/cz-apps/langchain-neo4j-test/cz_806_test.py` repro
//! (which keys `RETURN a.name AS a, b.name AS b` records by `a` / `b`).

use arcgraph_core::{LabelId, NodeId, TenantId};
use arcgraph_query::QueryEngine;
use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::executor::value::{NodeView, Value};
use arcgraph_query::semantic::StubCatalogProvider;

fn catalog() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["Person"])
        .with_properties(["name", "age"])
}

/// One `Person` (Label 1) with `name = 'Ada'`, `age = 36` so
/// `RETURN n.name`-style queries have a row to project.
fn one_person() -> StubExecutorSubstrate {
    StubExecutorSubstrate::new().with_node(
        TenantId::DEFAULT,
        NodeView::new(NodeId::new(1), Some(LabelId::new(1)))
            .with_property("name", Value::String("Ada".into()))
            .with_property("age", Value::Integer(36)),
    )
}

/// Execute `cypher` and return the result's `columns` (the user RETURN
/// aliases / implicit names).
fn columns(cypher: &str) -> Vec<String> {
    let catalog = catalog();
    let engine = QueryEngine::new(&catalog);
    engine
        .execute(cypher, &one_person())
        .expect("execute")
        .columns
}

/// Execute `cypher` and return BOTH columns and rows so a test can pin
/// that the column NAMES change without disturbing the row DATA.
fn columns_and_rows(cypher: &str) -> (Vec<String>, Vec<Vec<Value>>) {
    let catalog = catalog();
    let engine = QueryEngine::new(&catalog);
    let r = engine.execute(cypher, &one_person()).expect("execute");
    (r.columns, r.rows)
}

// ---------------------------------------------------------------------
// The discriminating oracle (RED-without-the-fix)
// ---------------------------------------------------------------------

#[test]
fn explicit_alias_surfaces_as_column_name() {
    // `RETURN n.name AS name` → column ["name"], NOT ["col_0"]. This is
    // the langchain Neo4jGraph drop-in linchpin (#353): the result dict
    // must be keyed by `name`.
    assert_eq!(
        columns("MATCH (n:Person) RETURN n.name AS name"),
        vec!["name"]
    );
}

// ---------------------------------------------------------------------
// The four acceptance shapes from the #353 scope
// ---------------------------------------------------------------------

#[test]
fn bare_variables_use_their_names() {
    // `RETURN a, r, b`-shape: a bare variable's implicit column name is
    // its own name. (We use Person vars a, b on a self-pattern-free
    // query — two bare variable references over the same row.)
    assert_eq!(
        columns("MATCH (n:Person) WITH n AS a, n AS b RETURN a, b"),
        vec!["a", "b"]
    );
}

#[test]
fn explicit_alias_on_property_access() {
    assert_eq!(
        columns("MATCH (n:Person) RETURN n.name AS person_name"),
        vec!["person_name"]
    );
}

#[test]
fn unaliased_property_access_uses_source_text() {
    // `RETURN n.name` (no AS) → the expression's source text "n.name".
    assert_eq!(columns("MATCH (n:Person) RETURN n.name"), vec!["n.name"]);
}

#[test]
fn unaliased_count_star_uses_source_text() {
    // `RETURN count(*)` → "count(*)" — exercises the aggregate path:
    // the lowering rewrites the RETURN item to a passthrough VariableRef
    // over the Aggregate's output, which would lose the implicit name
    // unless `source_text` is threaded through the rewrite.
    assert_eq!(
        columns("MATCH (n:Person) RETURN count(*)"),
        vec!["count(*)"]
    );
}

#[test]
fn aliased_count_star_uses_alias() {
    assert_eq!(columns("MATCH (n:Person) RETURN count(*) AS c"), vec!["c"]);
}

// ---------------------------------------------------------------------
// Mixed + multi-column + aggregate-with-group-by
// ---------------------------------------------------------------------

#[test]
fn mixed_bare_unaliased_and_aliased() {
    // `RETURN a, a.name, a.name AS person_name` → exact 3-name vec in
    // declaration order: bare-var, source-text, explicit-alias.
    assert_eq!(
        columns("MATCH (n:Person) WITH n AS a RETURN a, a.name, a.name AS person_name"),
        vec!["a", "a.name", "person_name"]
    );
}

#[test]
fn group_by_key_and_aggregate_both_named() {
    // `RETURN a.age, count(*)` → group-by key keeps its source text
    // ("a.age") and the aggregate keeps its ("count(*)"). Exercises BOTH
    // passthrough_item call sites (the group-by key + the aggregation).
    assert_eq!(
        columns("MATCH (n:Person) WITH n AS a RETURN a.age, count(*)"),
        vec!["a.age", "count(*)"]
    );
}

// ---------------------------------------------------------------------
// Whitespace normalization (Neo4j collapses insignificant whitespace)
// ---------------------------------------------------------------------

#[test]
fn unaliased_expression_normalizes_internal_whitespace() {
    // `RETURN n.age  +  1` (extra spaces) → the implicit name collapses
    // runs of whitespace to single spaces: "n.age + 1".
    assert_eq!(
        columns("MATCH (n:Person) RETURN n.age  +  1"),
        vec!["n.age + 1"]
    );
}

// ---------------------------------------------------------------------
// Wildcard → EMPTY columns (data-dependent width; wire → col_0..N)
// ---------------------------------------------------------------------

#[test]
fn wildcard_yields_no_column_names() {
    // `RETURN *` expands to the runtime row schema; the width is not
    // known at the bound-AST layer, so `columns` is EMPTY (the wire
    // renders `col_0..N` for the actual width). The honest choice — we
    // never fabricate a partial/wrong-width name list.
    assert!(
        columns("MATCH (n:Person) RETURN *").is_empty(),
        "wildcard projection must leave columns empty (data-dependent width)"
    );
}

// ---------------------------------------------------------------------
// ORDER BY / LIMIT are transparent to the column set
// ---------------------------------------------------------------------

#[test]
fn order_by_and_limit_do_not_change_columns() {
    // ORDER BY / LIMIT wrap the terminal projection but never redefine
    // its columns: `RETURN n.name AS name ORDER BY name LIMIT 10` still
    // exposes ["name"].
    assert_eq!(
        columns("MATCH (n:Person) RETURN n.name AS name ORDER BY name LIMIT 10"),
        vec!["name"]
    );
}

// ---------------------------------------------------------------------
// Row DATA is unchanged — only the column NAMES are added (#353 risk)
// ---------------------------------------------------------------------

#[test]
fn row_data_unchanged_only_names_added() {
    // The whole point of #353: thread NAMES through without disturbing
    // the row values. `RETURN n.name AS name` → one row `["Ada"]`,
    // named `name`.
    let (cols, rows) = columns_and_rows("MATCH (n:Person) RETURN n.name AS name");
    assert_eq!(cols, vec!["name"]);
    assert_eq!(rows, vec![vec![Value::String("Ada".into())]]);
}

// ---------------------------------------------------------------------
// Write-only statement (no RETURN) → no column names
// ---------------------------------------------------------------------

#[test]
fn write_only_no_return_has_no_column_names() {
    // `CREATE (n:Person {name:'Bob'})` has no projection; any result
    // rows are an implementation artifact, so there are no user column
    // names. (The wire falls back to col_0..N for whatever width.)
    let catalog = catalog();
    let engine = QueryEngine::new(&catalog);
    let r = engine
        .execute(
            "CREATE (n:Person {name: 'Bob'})",
            &StubExecutorSubstrate::new(),
        )
        .expect("execute");
    assert!(
        r.columns.is_empty(),
        "write-only statement with no RETURN must have no column names; got {:?}",
        r.columns
    );
}
