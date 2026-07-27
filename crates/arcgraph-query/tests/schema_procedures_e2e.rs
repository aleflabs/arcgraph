//! **ADR-197 (#802) R1 finding #3** — schema-introspection procedure
//! END-TO-END tests: `CALL apoc.meta.data / apoc.schema.nodes / db.labels
//! / db.propertyKeys / db.relationshipTypes / db.schema.visualization`
//! + `SHOW CONSTRAINTS / INDEXES / DATABASES`.
//!
//! These are the in-crate oracle for the ADR-197 part-b langchain-neo4j
//! `refresh_schema` surface (the prior coverage was the external,
//! non-CI langchain acceptance ONLY — testing strategy requires an in-crate
//! test per feature). They exercise the FULL front-end + executor (parse
//! → bind → type-check → cross-substrate → lower → materialize) with
//! STRONG `==` oracles over the result rows + the YIELD-WHERE binding.
//!
//! v1.0-α the apoc/db label-enumeration procedures return EMPTY rowsets
//! (correct for an empty graph; best-effort-empty for a populated graph
//! pending a substrate catalog-introspection method — documented in
//! `procedure_call.rs` + ADR-197 §Open-questions, R1 finding #5). The
//! NON-empty shapes (`SHOW DATABASES` → 1 db row, `db.schema.visualization`
//! → 1 empty-structure row) prove the proc/SHOW materialization produces
//! correctly-shaped rows — not just a vacuous empty.

use arcgraph_query::executor::ExecutionContext;
use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::executor::value::Value;
use arcgraph_query::logical_plan::LogicalPlanLoweringVisitor;
use arcgraph_query::semantic::error::BindingError;
use arcgraph_query::semantic::{
    BindingVisitor, CatalogProvider, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};
use arcgraph_query::{materialize, parse};

fn cat() -> StubCatalogProvider {
    StubCatalogProvider::new()
}

/// Full pipeline → result rows (panics on any stage error). The schema
/// procedures are substrate-independent (canned rows), so an empty stub
/// substrate suffices.
fn run(query: &str, c: &StubCatalogProvider) -> Vec<Vec<Value>> {
    let plan = lower(query, c);
    let s = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(c.tenant(), c.partition());
    materialize::materialize(&plan, &s, &ctx)
        .expect("materialize")
        .rows()
        .to_vec()
}

fn lower(query: &str, c: &StubCatalogProvider) -> arcgraph_query::logical_plan::LogicalPlan {
    let stmt = parse(query).expect("parse");
    let mut bound = BindingVisitor::bind(&stmt, query, c).expect("bind");
    TypeCheckVisitor::check(&mut bound, c).expect("type-check");
    CrossSubstrateValidator::validate(&bound, c).expect("cross-substrate");
    LogicalPlanLoweringVisitor::lower(&bound).expect("lower")
}

/// Bind-only — for the tests that assert a BIND error.
fn bind_err(query: &str, c: &StubCatalogProvider) -> Vec<BindingError> {
    let stmt = parse(query).expect("parse");
    match BindingVisitor::bind(&stmt, query, c) {
        Ok(_) => panic!("expected a bind error for: {query}"),
        Err(errs) => errs,
    }
}

fn string_cell(v: &Value) -> &str {
    match v {
        Value::String(s) => s.as_str(),
        other => panic!("expected String, got {other:?}"),
    }
}

// =====================================================================
// apoc.meta.data — the langchain-critical YIELD-WHERE surface.
// =====================================================================

#[test]
fn apoc_meta_data_yield_where_parses_binds_executes() {
    // The langchain `refresh_schema` shape (simplified — drops the
    // collect/map-projection tail which is orthogonal to the YIELD-WHERE
    // grammar). The YIELD'd columns flow into the WHERE + RETURN like
    // UNWIND. v1.0-α returns 0 rows (empty/best-effort schema), which is
    // exactly what makes refresh_schema SUCCEED.
    let rows = run(
        "CALL apoc.meta.data() \
         YIELD label, other, elementType, type, property \
         WHERE NOT type = 'RELATIONSHIP' \
         RETURN label, type",
        &cat(),
    );
    assert_eq!(
        rows.len(),
        0,
        "apoc.meta.data is best-effort-empty at v1.0-α (ADR-197 finding #5); \
         the load-bearing proof is that the YIELD-WHERE-RETURN flow parses + \
         binds + executes WITHOUT error"
    );
}

#[test]
fn apoc_meta_data_yield_where_with_and_predicate_parses() {
    // The fuller langchain predicate: `WHERE NOT type = "RELATIONSHIP"
    // AND elementType = "node"`. Double-quoted string literals (Cypher
    // accepts both quote styles) + a compound AND predicate over two
    // yielded columns.
    let rows = run(
        "CALL apoc.meta.data() \
         YIELD label, elementType, type, property \
         WHERE NOT type = \"RELATIONSHIP\" AND elementType = \"node\" \
         RETURN label, property",
        &cat(),
    );
    assert_eq!(rows.len(), 0, "best-effort-empty at v1.0-α");
}

#[test]
fn apoc_meta_data_where_binds_against_yielded_columns_only() {
    // The DISCRIMINATING YIELD-WHERE binding oracle. A WHERE that
    // references a column NOT in the YIELD list must FAIL at bind
    // (`type` is not yielded) — proving the WHERE resolves against the
    // procedure's YIELD'd output bindings, not a free-for-all.
    let errs = bind_err(
        "CALL apoc.meta.data() YIELD label WHERE NOT type = 'RELATIONSHIP' RETURN label",
        &cat(),
    );
    assert!(
        errs.iter()
            .any(|e| matches!(e, BindingError::UndeclaredVariable { .. })),
        "WHERE referencing a non-YIELD'd column `type` must be an undeclared-variable \
         bind error; got {errs:?}"
    );

    // And the positive: yielding `type` makes the SAME WHERE bind + run.
    let rows = run(
        "CALL apoc.meta.data() YIELD label, type WHERE NOT type = 'RELATIONSHIP' RETURN label",
        &cat(),
    );
    assert_eq!(rows.len(), 0, "binds + executes once `type` is yielded");
}

// =====================================================================
// db.labels / db.propertyKeys / db.relationshipTypes — enumeration
// procedures (YIELD a single column; empty at v1.0-α).
// =====================================================================

#[test]
fn db_labels_parses_and_executes() {
    let rows = run("CALL db.labels() YIELD label RETURN label", &cat());
    assert_eq!(rows.len(), 0, "db.labels best-effort-empty at v1.0-α");
}

#[test]
fn db_property_keys_parses_and_executes() {
    let rows = run(
        "CALL db.propertyKeys() YIELD propertyKey RETURN propertyKey",
        &cat(),
    );
    assert_eq!(rows.len(), 0, "db.propertyKeys best-effort-empty at v1.0-α");
}

#[test]
fn db_relationship_types_parses_and_executes() {
    let rows = run(
        "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType",
        &cat(),
    );
    assert_eq!(
        rows.len(),
        0,
        "db.relationshipTypes best-effort-empty at v1.0-α"
    );
}

// =====================================================================
// SHOW CONSTRAINTS / INDEXES / DATABASES.
// =====================================================================

#[test]
fn show_constraints_parses_and_executes_empty() {
    // langchain wraps SHOW CONSTRAINTS in try/except + tolerates empty.
    let rows = run("SHOW CONSTRAINTS", &cat());
    assert_eq!(
        rows.len(),
        0,
        "no constraint catalog at v1.0-α → empty rowset"
    );
}

#[test]
fn show_indexes_parses_and_executes_empty() {
    let rows = run("SHOW INDEXES", &cat());
    assert_eq!(rows.len(), 0, "no index catalog at v1.0-α → empty rowset");
}

#[test]
fn show_databases_returns_one_default_db_row() {
    // The NON-empty SHOW oracle: proves the SHOW materialization
    // produces correctly-shaped rows (not a vacuous empty). One row for
    // the single default database; column 0 is `name`.
    let rows = run("SHOW DATABASES", &cat());
    assert_eq!(rows.len(), 1, "SHOW DATABASES → exactly one default-db row");
    assert_eq!(
        string_cell(&rows[0][0]),
        "neo4j",
        "the default database name column"
    );
}

#[test]
fn db_schema_visualization_returns_one_row() {
    // The NON-empty proc oracle: one row carrying empty nodes +
    // relationships lists (langchain tolerates a partial visualization).
    let rows = run(
        "CALL db.schema.visualization() YIELD nodes, relationships RETURN nodes, relationships",
        &cat(),
    );
    assert_eq!(rows.len(), 1, "db.schema.visualization → one (empty) row");
    assert!(
        matches!(rows[0][0], Value::List(ref l) if l.is_empty()),
        "nodes column is an empty list; got {:?}",
        rows[0][0]
    );
    assert!(
        matches!(rows[0][1], Value::List(ref l) if l.is_empty()),
        "relationships column is an empty list; got {:?}",
        rows[0][1]
    );
}

// =====================================================================
// Bind-error paths (the binder's procedure catalog + YIELD validation).
// =====================================================================

#[test]
fn unknown_procedure_rejected_at_bind() {
    let errs = bind_err("CALL db.bogusProcedure() YIELD x RETURN x", &cat());
    assert!(
        errs.iter()
            .any(|e| matches!(e, BindingError::UnknownProcedure { .. })),
        "an unknown dotted procedure name must be an UnknownProcedure bind error; got {errs:?}"
    );
}

#[test]
fn invalid_yield_column_rejected_at_bind() {
    // `db.labels` YIELDs only `label`; YIELDing a bogus column rejects.
    let errs = bind_err(
        "CALL db.labels() YIELD notAColumn RETURN notAColumn",
        &cat(),
    );
    assert!(
        errs.iter()
            .any(|e| matches!(e, BindingError::InvalidYieldColumn { .. })),
        "YIELDing a column the procedure does not output must be an InvalidYieldColumn \
         bind error; got {errs:?}"
    );
}
