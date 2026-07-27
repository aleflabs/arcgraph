//! Null-anchored `OPTIONAL MATCH` named-path null-extension e2e
//! (TCK `expressions/path/Path1[1]` + `Path2[3]`, sibling of #1051/#1243).
//!
//! openCypher 9 §6.5 + the `nodes()`/`relationships()` path-accessor
//! contract: when a preceding clause binds the anchor variable `a` to
//! **NULL** (`WITH null AS a`), the following
//!
//! ```cypher
//! OPTIONAL MATCH p = (a)-[r]->()
//! ```
//!
//! cannot match any pattern (a null anchor matches nothing), so the
//! OPTIONAL MATCH must NULL-extend: emit exactly **one** row with the
//! path variable `p` bound to `null`. Then `nodes(p)` / `relationships(p)`
//! on the null path are themselves `null` (already proven at the
//! `fn_nodes(Value::Null)` / `fn_relationships(Value::Null)` eval layer).
//!
//! This is DISTINCT from the #1243 *leading* OPTIONAL MATCH (no driving
//! rows) case: here a preceding `WITH` supplies exactly one driving row
//! whose `a` column is NULL, and the OPTIONAL named-path right side must
//! still null-extend that single row rather than ERROR or drop it.
//!
//! The two TCK scenarios this unblocks:
//! - `expressions/path/Path1.feature [1]` — `nodes()` on null path.
//! - `expressions/path/Path2.feature [3]` — `relationships()` on null path.
//!
//! Both are EXACT-ROW oracles: exactly one row, `[null, null]`.
//!
//! # ADR provenance
//! - ADR-006 amendment-01 §A-2 — OPTIONAL MATCH lowers to left-outer join.
//! - openCypher 9 §6.5 — OPTIONAL MATCH null-extends on no match.

use arcgraph_core::{LabelId, NodeId, RelId, TenantId, TypeId};
use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::executor::value::{NodeView, RelView, Value};
use arcgraph_query::semantic::{ArcQLError, BindingError, StubCatalogProvider};
use arcgraph_query::{ExplainError, QueryEngine};

fn cat() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["Person"])
        .with_properties(["name"])
        .with_rel_types(["KNOWS"])
}

/// A non-empty graph (one edge `1 -[KNOWS]-> 2`) — proves the null-anchor
/// null-extension does NOT depend on the graph being empty: even with a
/// matchable edge present, a NULL anchor `a` cannot bind, so the OPTIONAL
/// pattern matches nothing for the null-driving row.
fn one_edge_substrate() -> StubExecutorSubstrate {
    StubExecutorSubstrate::new()
        .with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(1), Some(LabelId::new(1)))
                .with_property("name", Value::String("a".into())),
        )
        .with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(2), Some(LabelId::new(1)))
                .with_property("name", Value::String("b".into())),
        )
        .with_edge(
            TenantId::DEFAULT,
            RelView::new(
                RelId::new(10),
                NodeId::new(1),
                NodeId::new(2),
                Some(TypeId::new(1)),
            ),
        )
}

fn run(query: &str, substrate: &StubExecutorSubstrate) -> Vec<Vec<Value>> {
    let cat = cat();
    let engine = QueryEngine::new(&cat);
    engine
        .execute(query, substrate)
        .unwrap_or_else(|e| panic!("execute failed for `{query}`: {e:?}"))
        .rows()
        .to_vec()
}

// ---------------------------------------------------------------------
// Path1[1] — `nodes()` on a null path from a null-anchored OPTIONAL MATCH.
// ---------------------------------------------------------------------
#[test]
fn null_anchor_optional_named_path_nodes_is_null() {
    let rows = run(
        "WITH null AS a \
         OPTIONAL MATCH p = (a)-[r]->() \
         RETURN nodes(p), nodes(null)",
        &one_edge_substrate(),
    );
    assert_eq!(
        rows.len(),
        1,
        "null-anchored OPTIONAL MATCH must null-extend the single driving \
         row into exactly 1 row (got {})",
        rows.len()
    );
    assert_eq!(
        rows[0],
        vec![Value::Null, Value::Null],
        "`p` binds to null (null anchor matches nothing) ⇒ nodes(p) is null; \
         nodes(null) is null"
    );
}

// ---------------------------------------------------------------------
// Path2[3] — `relationships()` on the same null path.
// ---------------------------------------------------------------------
#[test]
fn null_anchor_optional_named_path_relationships_is_null() {
    let rows = run(
        "WITH null AS a \
         OPTIONAL MATCH p = (a)-[r]->() \
         RETURN relationships(p), relationships(null)",
        &one_edge_substrate(),
    );
    assert_eq!(
        rows.len(),
        1,
        "null-anchored OPTIONAL MATCH must null-extend the single driving \
         row into exactly 1 row (got {})",
        rows.len()
    );
    assert_eq!(
        rows[0],
        vec![Value::Null, Value::Null],
        "`p` binds to null ⇒ relationships(p) is null; relationships(null) is null"
    );
}

// ---------------------------------------------------------------------
// Bonus: returning the path variable `p` itself must be null (the root of
// the accessor null-ness). Pins that the binding — not just the accessor —
// is null.
// ---------------------------------------------------------------------
#[test]
fn null_anchor_optional_named_path_var_itself_is_null() {
    let rows = run(
        "WITH null AS a \
         OPTIONAL MATCH p = (a)-[r]->() \
         RETURN p",
        &one_edge_substrate(),
    );
    assert_eq!(rows.len(), 1, "exactly one null-extended row");
    assert_eq!(
        rows[0],
        vec![Value::Null],
        "the path variable `p` itself must be null"
    );
}

// =====================================================================
// ADVERSARIAL — a NON-null value bound as a node anchor MUST still raise
// `VariableTypeConflict`, EVEN under OPTIONAL MATCH. The null-anchor
// carve-out is keyed on the static NULL type (which unifies with NODE),
// NOT on all-values-under-OPTIONAL: a non-null scalar / list / map does
// NOT unify with NODE, so `(a)` is a static type error regardless of
// OPTIONAL-ness (OPTIONAL changes runtime null-extension, not the static
// type signature `(a): NODE`). This mirrors TCK `clauses/match/Match1[11]`
// (which pins the non-null matrix under a *required* MATCH) and extends it
// to the OPTIONAL form. Without the `NullValue`-narrowed gate these would
// WRONGLY return rows (R1 delta-defect).
// =====================================================================

/// Compile/bind a query expected to be REJECTED; return the `ArcQLError`.
/// Panics if it instead succeeded (the defect this lane closes).
fn reject(query: &str) -> ArcQLError {
    let cat = cat();
    let engine = QueryEngine::new(&cat);
    match engine.execute(query, &one_edge_substrate()) {
        Ok(res) => panic!(
            "expected COMPILE-time VariableTypeConflict for `{query}`, \
             but it returned {} row(s): {:?}",
            res.rows().len(),
            res.rows()
        ),
        Err(ExplainError::ArcQL(e)) => e,
        Err(other) => {
            panic!("expected `ExplainError::ArcQL(..)` for `{query}`, got: {other}")
        }
    }
}

fn assert_type_conflict(query: &str) {
    match reject(query) {
        ArcQLError::Binding(BindingError::VariableTypeConflict { .. }) => {}
        other => panic!("expected VariableTypeConflict for `{query}`, got: {other}"),
    }
}

#[test]
fn nonnull_integer_anchor_under_optional_still_conflicts() {
    assert_type_conflict("WITH 123 AS a OPTIONAL MATCH (a)-[r]->() RETURN a, r");
}

#[test]
fn nonnull_list_anchor_under_optional_still_conflicts() {
    assert_type_conflict("WITH [1, 2] AS a OPTIONAL MATCH (a)-->() RETURN a");
}

#[test]
fn nonnull_map_anchor_under_optional_still_conflicts() {
    assert_type_conflict("WITH {x: 1} AS a OPTIONAL MATCH (a)-->() RETURN a");
}

#[test]
fn nonnull_string_anchor_under_optional_still_conflicts() {
    assert_type_conflict("WITH 'foo' AS a OPTIONAL MATCH (a)-->() RETURN a");
}

#[test]
fn nonnull_float_anchor_under_optional_still_conflicts() {
    assert_type_conflict("WITH 123.4 AS a OPTIONAL MATCH (a)-->() RETURN a");
}

#[test]
fn nonnull_boolean_anchor_under_optional_still_conflicts() {
    assert_type_conflict("WITH true AS a OPTIONAL MATCH (a)-->() RETURN a");
}

#[test]
fn nonnull_named_path_anchor_under_optional_still_conflicts() {
    // The exact Path1/Path2 SHAPE (named path) but with a non-null anchor
    // — proves the carve-out is null-typed, not OPTIONAL-named-path-wide.
    assert_type_conflict("WITH 123 AS a OPTIONAL MATCH p = (a)-[r]->() RETURN nodes(p)");
}

#[test]
fn nonnull_integer_anchor_under_required_match_still_conflicts() {
    // Match1[11] proper — required MATCH, unchanged by this lane.
    assert_type_conflict("WITH 123 AS a MATCH (a) RETURN a");
}
