//! **ADR-188 / #620 (list-half)** — openCypher v9 §3.5 list
//! comprehension `[x IN list WHERE p | e]` END-TO-END tests.
//!
//! These exercise the FULL pipeline (parse → bind → type-check →
//! cross-substrate → lower → execute) for the list-comprehension
//! special form, mirroring the #687 list-predicate e2e shape
//! (`tests/list_predicate_e2e.rs`). They complement the direct-evaluator
//! unit tests in `src/executor/eval.rs` (which pin the §3.5 semantics at
//! the strongest oracle); here we prove the grammar production, the
//! scoped binder, the type-check (`List(elem-or-proj type)`), the
//! cross-substrate walk, the lowering ripple, and the evaluator all
//! compose correctly through a real query.
//!
//! All oracles are STRONG `==` over the result rows. Each test that
//! depends on a load-bearing §3.5 semantic (3VL filter — only `true`
//! keeps; null-list ⇒ null; empty ⇒ empty; order preserved; identity
//! projection) asserts the exact value.
//!
//! Map-comprehension is OUT OF SCOPE (deferred to a future `Value::Map`
//! landing per ADR-188 Decision 5); only the LIST half ships here.
//!
//! # Why a list-valued PROPERTY for the NULL cases
//!
//! Inline list literals (`[1, null, 3]`) parse and evaluate, but the
//! property-list form (`n.scores`) also lets us inject NULL elements via
//! a `Value::List` containing `Value::Null` and exercise the
//! comprehension over a non-literal source list.

use arcgraph_core::{LabelId, NodeId, TenantId};
use arcgraph_query::executor::value::NodeView;
use arcgraph_query::executor::{ExecutionContext, StubExecutorSubstrate, Value};
use arcgraph_query::logical_plan::LogicalPlanLoweringVisitor;
use arcgraph_query::semantic::{
    BindingVisitor, CatalogProvider, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};
use arcgraph_query::{materialize, parse};

// ---------------------------------------------------------------------
// Fixtures (mirror list_predicate_e2e.rs)
// ---------------------------------------------------------------------

const LABEL_X: u32 = 1;

fn cat() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["X"])
        .with_properties(["g", "scores"])
}

fn node(id: u64) -> NodeView {
    NodeView::new(NodeId::new(id), Some(LabelId::new(LABEL_X)))
}

fn ints(xs: &[i64]) -> Value {
    Value::List(xs.iter().map(|n| Value::Integer(*n)).collect())
}

/// `[before..., NULL, after...]` as a `Value::List`.
fn ints_with_null(before: &[i64], after: &[i64]) -> Value {
    let mut v: Vec<Value> = before.iter().map(|n| Value::Integer(*n)).collect();
    v.push(Value::Null);
    v.extend(after.iter().map(|n| Value::Integer(*n)));
    Value::List(v)
}

/// Bind + type-check + validate + lower + materialize, returning rows.
fn run(query: &str, s: &StubExecutorSubstrate, c: &StubCatalogProvider) -> Vec<Vec<Value>> {
    let stmt = parse(query).expect("parse");
    let mut bound = BindingVisitor::bind(&stmt, query, c).expect("bind");
    TypeCheckVisitor::check(&mut bound, c).expect("type-check");
    CrossSubstrateValidator::validate(&bound, c).expect("cross-substrate");
    let plan = LogicalPlanLoweringVisitor::lower(&bound).expect("lower");
    let ctx = ExecutionContext::new(c.tenant(), c.partition());
    materialize::materialize(&plan, s, &ctx)
        .expect("materialize")
        .rows()
        .to_vec()
}

/// Single node with `g` + a `scores` list property.
fn one_node_with_scores(g: i64, scores: Value) -> StubExecutorSubstrate {
    StubExecutorSubstrate::new().with_node(
        TenantId::DEFAULT,
        node(1)
            .with_property("g", Value::Integer(g))
            .with_property("scores", scores),
    )
}

/// A single node with just `g` (no list property) — for inline-list
/// comprehension tests that don't need a property source.
fn one_node(g: i64) -> StubExecutorSubstrate {
    StubExecutorSubstrate::new().with_node(
        TenantId::DEFAULT,
        node(1).with_property("g", Value::Integer(g)),
    )
}

/// Assert the single returned row is exactly one `Value::List` column
/// equal to `want`.
fn assert_one_list_row(rows: &[Vec<Value>], want: Value) {
    assert_eq!(rows.len(), 1, "expected exactly one row, got {rows:?}");
    assert_eq!(rows[0].len(), 1, "expected exactly one column");
    assert_eq!(rows[0][0], want);
}

// =====================================================================
// RETURN-projection consumer site — the four §3.5 combinations
// =====================================================================

#[test]
fn e2e_filter_then_map() {
    // [x IN [1,2,3] WHERE x > 1 | x * 10] ⇒ [20, 30]
    let s = one_node(7);
    let rows = run(
        "MATCH (n:X) RETURN [x IN [1, 2, 3] WHERE x > 1 | x * 10] AS ys",
        &s,
        &cat(),
    );
    assert_one_list_row(&rows, ints(&[20, 30]));
}

#[test]
fn e2e_map_only_no_where() {
    // [x IN [1,2,3] | x * 10] ⇒ [10, 20, 30] (map every element)
    let s = one_node(7);
    let rows = run(
        "MATCH (n:X) RETURN [x IN [1, 2, 3] | x * 10] AS ys",
        &s,
        &cat(),
    );
    assert_one_list_row(&rows, ints(&[10, 20, 30]));
}

#[test]
fn e2e_filter_only_identity_projection() {
    // [x IN [1,2,3,4] WHERE x > 2] ⇒ [3, 4] (filter, project x itself)
    let s = one_node(7);
    let rows = run(
        "MATCH (n:X) RETURN [x IN [1, 2, 3, 4] WHERE x > 2] AS ys",
        &s,
        &cat(),
    );
    assert_one_list_row(&rows, ints(&[3, 4]));
}

#[test]
fn e2e_identity_no_where_no_projection() {
    // [x IN [1,2,3]] ⇒ [1, 2, 3] (identity over the whole list)
    let s = one_node(7);
    let rows = run("MATCH (n:X) RETURN [x IN [1, 2, 3]] AS ys", &s, &cat());
    assert_one_list_row(&rows, ints(&[1, 2, 3]));
}

// =====================================================================
// Edge cases (§3.5 + ADR-188 Decision 4)
// =====================================================================

#[test]
fn e2e_empty_list_yields_empty_list() {
    // [x IN [] WHERE x > 1 | x * 10] ⇒ [] (NOT null)
    let s = one_node(7);
    let rows = run(
        "MATCH (n:X) RETURN [x IN [] WHERE x > 1 | x * 10] AS ys",
        &s,
        &cat(),
    );
    assert_one_list_row(&rows, Value::List(Vec::new()));
}

#[test]
fn e2e_null_list_yields_null() {
    // [x IN n.scores | x * 10] where n.scores is NULL (property missing)
    // ⇒ null (the null-list rule). The node has no `scores` property, so
    // `n.scores` evaluates to NULL.
    let s = one_node(7);
    let rows = run(
        "MATCH (n:X) RETURN [x IN n.scores | x * 10] AS ys",
        &s,
        &cat(),
    );
    assert_one_list_row(&rows, Value::Null);
}

#[test]
fn e2e_null_element_filtered_out_by_where() {
    // n.scores = [1, NULL, 3]; [x IN n.scores WHERE x > 1 | x] —
    //   1 > 1 ⇒ false (out); null > 1 ⇒ null/Unknown (FILTERED OUT,
    //   only `true` keeps); 3 > 1 ⇒ true (in) ⇒ [3].
    // BITES on a wrong impl that keeps the null element or surfaces a
    // null in the result.
    let s = one_node_with_scores(5, ints_with_null(&[1], &[3]));
    let rows = run(
        "MATCH (n:X) RETURN [x IN n.scores WHERE x > 1 | x] AS ys",
        &s,
        &cat(),
    );
    assert_one_list_row(&rows, ints(&[3]));
}

#[test]
fn e2e_null_element_preserved_when_no_where() {
    // n.scores = [1, NULL, 3]; [x IN n.scores | x] — no WHERE, so the
    // NULL element is KEPT as a value (identity projection). ⇒ [1,null,3].
    let s = one_node_with_scores(5, ints_with_null(&[1], &[3]));
    let rows = run("MATCH (n:X) RETURN [x IN n.scores | x] AS ys", &s, &cat());
    assert_one_list_row(
        &rows,
        Value::List(vec![Value::Integer(1), Value::Null, Value::Integer(3)]),
    );
}

#[test]
fn e2e_order_preserved_when_filtering() {
    // [x IN [5,1,4,2,3] WHERE x > 2 | x] ⇒ [5, 4, 3] (SOURCE order, not
    // sorted; elements ≤ 2 dropped).
    let s = one_node(7);
    let rows = run(
        "MATCH (n:X) RETURN [x IN [5, 1, 4, 2, 3] WHERE x > 2 | x] AS ys",
        &s,
        &cat(),
    );
    assert_one_list_row(&rows, ints(&[5, 4, 3]));
}

// =====================================================================
// Source list = a PROPERTY (non-literal source list)
// =====================================================================

#[test]
fn e2e_over_property_list_map() {
    // n.scores = [10,20,30]; [x IN n.scores | x * 2] ⇒ [20, 40, 60].
    let s = one_node_with_scores(5, ints(&[10, 20, 30]));
    let rows = run(
        "MATCH (n:X) RETURN [x IN n.scores | x * 2] AS ys",
        &s,
        &cat(),
    );
    assert_one_list_row(&rows, ints(&[20, 40, 60]));
}

#[test]
fn e2e_projection_references_outer_binding() {
    // The projection may reference an OUTER row binding (`n.g`), proving
    // the scoped var `x` and the outer `n` coexist in the extended row.
    // n.g = 100, n.scores = [1,2,3]; [x IN n.scores | x + n.g]
    //   ⇒ [101, 102, 103].
    let s = one_node_with_scores(100, ints(&[1, 2, 3]));
    let rows = run(
        "MATCH (n:X) RETURN [x IN n.scores | x + n.g] AS ys",
        &s,
        &cat(),
    );
    assert_one_list_row(&rows, ints(&[101, 102, 103]));
}

#[test]
fn e2e_where_references_outer_binding() {
    // The WHERE filter may reference an OUTER binding too:
    // n.g = 2, n.scores = [1,2,3]; [x IN n.scores WHERE x > n.g | x]
    //   ⇒ [3] (only 3 > 2).
    let s = one_node_with_scores(2, ints(&[1, 2, 3]));
    let rows = run(
        "MATCH (n:X) RETURN [x IN n.scores WHERE x > n.g | x] AS ys",
        &s,
        &cat(),
    );
    assert_one_list_row(&rows, ints(&[3]));
}

// =====================================================================
// Nested comprehension
// =====================================================================

#[test]
fn e2e_nested_comprehension() {
    // [x IN [1,2] | [y IN [10,20] | x + y]]
    //   x=1 ⇒ [11,21]; x=2 ⇒ [12,22] ⇒ [[11,21],[12,22]].
    // Exercises the nested-slot mechanism end-to-end: inner `y` lands one
    // slot past the outer `x`, and the binder's reverse scope-walk gives
    // inner-shadows-outer.
    let s = one_node(7);
    let rows = run(
        "MATCH (n:X) RETURN [x IN [1, 2] | [y IN [10, 20] | x + y]] AS yss",
        &s,
        &cat(),
    );
    assert_one_list_row(&rows, Value::List(vec![ints(&[11, 21]), ints(&[12, 22])]));
}

#[test]
fn e2e_nested_with_inner_filter() {
    // [x IN [1,2,3] WHERE x > 1 | [y IN [10,20] WHERE y > 10 | x + y]]
    //   outer keeps x ∈ {2,3}; inner keeps y = 20 only.
    //   x=2 ⇒ [22]; x=3 ⇒ [23] ⇒ [[22],[23]].
    let s = one_node(7);
    let rows = run(
        "MATCH (n:X) RETURN [x IN [1, 2, 3] WHERE x > 1 | [y IN [10, 20] WHERE y > 10 | x + y]] AS yss",
        &s,
        &cat(),
    );
    assert_one_list_row(&rows, Value::List(vec![ints(&[22]), ints(&[23])]));
}

// =====================================================================
// Coexistence: a plain list LITERAL still works alongside comprehensions
// =====================================================================

#[test]
fn e2e_plain_list_literal_still_returns_list() {
    // The disambiguation must not regress plain list literals: a bare
    // `[1, 2, 3]` RETURNs as the list value unchanged.
    let s = one_node(7);
    let rows = run("MATCH (n:X) RETURN [1, 2, 3] AS xs", &s, &cat());
    assert_one_list_row(&rows, ints(&[1, 2, 3]));
}
