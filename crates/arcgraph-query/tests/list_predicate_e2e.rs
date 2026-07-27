//! **ADR-188 / #687** — openCypher v9 list-predicate functions
//! (`all`/`any`/`none`/`single`) + `reduce` END-TO-END tests.
//!
//! These exercise the FULL pipeline (parse → bind → type-check →
//! cross-substrate → lower → execute) for the list-predicate special
//! forms, mirroring the #622 UNION e2e shape
//! (`tests/w28_union_all_e2e.rs`). They complement the direct-evaluator
//! unit tests in `src/executor/eval.rs` (which pin the Decision 4 3VL
//! truth table at the strongest oracle); here we prove the grammar
//! productions, the scoped binder, the type-check, and the evaluator
//! all compose correctly through a real query.
//!
//! All oracles are STRONG `==` over the result rows / row counts. Each
//! test that depends on the PE-corrected `single`-with-NULL semantics
//! asserts the exact ADR value and is documented as load-bearing.
//!
//! # Why a list-valued PROPERTY for the NULL cases
//!
//! Inline list literals (`[1, null, 3]`) parse and evaluate, but the
//! property-list form (`n.scores`) lets us inject NULL elements via a
//! `Value::List` containing `Value::Null` and exercise the WHERE-filter
//! and RETURN-projection consumer sites with the same node fixture.

use arcgraph_core::{LabelId, NodeId, TenantId};
use arcgraph_query::executor::value::NodeView;
use arcgraph_query::executor::{ExecutionContext, StubExecutorSubstrate, Value};
use arcgraph_query::logical_plan::LogicalPlanLoweringVisitor;
use arcgraph_query::semantic::{
    BindingVisitor, CatalogProvider, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};
use arcgraph_query::{materialize, parse};

// ---------------------------------------------------------------------
// Fixtures
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

/// A list with a NULL element at `null_at` (0-based), the rest = `xs`
/// values spliced around it. We build `[before..., NULL, after...]`.
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

fn single_g(rows: &[Vec<Value>]) -> Vec<i64> {
    rows.iter()
        .map(|r| match &r[0] {
            Value::Integer(n) => *n,
            other => panic!("expected Integer g, got {other:?}"),
        })
        .collect()
}

// =====================================================================
// WHERE-filter consumer site — inline list literals
// =====================================================================

#[test]
fn e2e_all_inline_true_keeps_row() {
    let s = one_node_with_scores(7, ints(&[1, 2, 3]));
    let rows = run(
        "MATCH (n:X) WHERE all(x IN [1, 2, 3] WHERE x > 0) RETURN n.g",
        &s,
        &cat(),
    );
    assert_eq!(
        single_g(&rows),
        vec![7],
        "all([1,2,3] WHERE x>0) ⇒ true ⇒ row kept"
    );
}

#[test]
fn e2e_all_inline_false_drops_row() {
    let s = one_node_with_scores(7, ints(&[1, 2, 3]));
    let rows = run(
        "MATCH (n:X) WHERE all(x IN [1, 2, 3] WHERE x > 1) RETURN n.g",
        &s,
        &cat(),
    );
    assert!(
        rows.is_empty(),
        "all([1,2,3] WHERE x>1) ⇒ false ⇒ row dropped"
    );
}

#[test]
fn e2e_any_inline_true_keeps_row() {
    let s = one_node_with_scores(7, ints(&[1, 2, 3]));
    let rows = run(
        "MATCH (n:X) WHERE any(x IN [1, 2, 3] WHERE x = 2) RETURN n.g",
        &s,
        &cat(),
    );
    assert_eq!(single_g(&rows), vec![7]);
}

#[test]
fn e2e_none_inline_keeps_row_when_no_match() {
    let s = one_node_with_scores(7, ints(&[1, 2, 3]));
    let rows = run(
        "MATCH (n:X) WHERE none(x IN [1, 2, 3] WHERE x > 100) RETURN n.g",
        &s,
        &cat(),
    );
    assert_eq!(single_g(&rows), vec![7], "none([1,2,3] WHERE x>100) ⇒ true");
}

// =====================================================================
// WHERE-filter consumer site — list-valued PROPERTY (incl NULL elems)
// =====================================================================

#[test]
fn e2e_all_over_property_list() {
    // n.scores = [10,20,30]; all(x IN n.scores WHERE x >= 10) ⇒ true.
    let s = one_node_with_scores(5, ints(&[10, 20, 30]));
    let rows = run(
        "MATCH (n:X) WHERE all(x IN n.scores WHERE x >= 10) RETURN n.g",
        &s,
        &cat(),
    );
    assert_eq!(single_g(&rows), vec![5]);
}

#[test]
fn e2e_all_over_property_list_with_null_unknown_filters_row() {
    // n.scores = [20, NULL, 30]; all(x IN n.scores WHERE x > 1) ⇒ NULL
    // (no false, but a null). A WHERE that is NULL FILTERS the row (3VL:
    // Unknown ⇒ row not emitted). So the row is DROPPED — proving the
    // 3VL `null` flows from the list-predicate through the WHERE-filter
    // discrimination correctly.
    let s = one_node_with_scores(5, ints_with_null(&[20], &[30]));
    let rows = run(
        "MATCH (n:X) WHERE all(x IN n.scores WHERE x > 1) RETURN n.g",
        &s,
        &cat(),
    );
    assert!(
        rows.is_empty(),
        "all([20,null,30] WHERE x>1) ⇒ NULL ⇒ WHERE filters the row (3VL)"
    );
}

#[test]
fn e2e_all_over_property_list_definite_false_dominates_null_filters_row() {
    // n.scores = [0, NULL, 30]; all(x IN n.scores WHERE x > 1) ⇒ false
    // (0 is a definite false dominating the null) ⇒ row dropped.
    let s = one_node_with_scores(5, ints_with_null(&[0], &[30]));
    let rows = run(
        "MATCH (n:X) WHERE all(x IN n.scores WHERE x > 1) RETURN n.g",
        &s,
        &cat(),
    );
    assert!(
        rows.is_empty(),
        "definite false dominates null ⇒ false ⇒ dropped"
    );
}

// =====================================================================
// RETURN-projection consumer site — the list-predicate / reduce VALUE
// =====================================================================

#[test]
fn e2e_all_in_return_projects_boolean() {
    let s = one_node_with_scores(7, ints(&[1, 2, 3]));
    let rows = run(
        "MATCH (n:X) RETURN all(x IN [1, 2, 3] WHERE x > 0) AS ok",
        &s,
        &cat(),
    );
    assert_eq!(rows, vec![vec![Value::Boolean(true)]]);
}

#[test]
fn e2e_single_in_return_exactly_one_true() {
    let s = one_node_with_scores(7, ints(&[1, 2, 3]));
    let rows = run(
        "MATCH (n:X) RETURN single(x IN [1, 2, 3] WHERE x = 2) AS uniq",
        &s,
        &cat(),
    );
    assert_eq!(rows, vec![vec![Value::Boolean(true)]]);
}

#[test]
fn e2e_single_one_definite_witness_plus_null_is_true() {
    // LOAD-BEARING (PE-corrected Decision 4-single):
    // single(x IN [2,null,3] WHERE x = 2) ⇒ TRUE — one definite witness
    // dominates the null. End-to-end through RETURN projection.
    let s = one_node_with_scores(7, ints_with_null(&[2], &[3]));
    let rows = run(
        "MATCH (n:X) RETURN single(x IN n.scores WHERE x = 2) AS uniq",
        &s,
        &cat(),
    );
    assert_eq!(
        rows,
        vec![vec![Value::Boolean(true)]],
        "single([2,null,3] WHERE x=2) MUST project TRUE (one definite \
         witness dominates the null; ADR-188 Decision 4-single)"
    );
}

#[test]
fn e2e_single_zero_true_plus_null_is_null() {
    // LOAD-BEARING (PE-corrected Decision 4-single):
    // single(x IN [1,null,3] WHERE x = 2) ⇒ NULL — zero definite trues,
    // a null could be the single match. End-to-end through RETURN.
    let s = one_node_with_scores(7, ints_with_null(&[1], &[3]));
    let rows = run(
        "MATCH (n:X) RETURN single(x IN n.scores WHERE x = 2) AS uniq",
        &s,
        &cat(),
    );
    assert_eq!(
        rows,
        vec![vec![Value::Null]],
        "single([1,null,3] WHERE x=2) MUST project NULL (zero definite \
         trues + a null could be the single match; ADR-188 Decision \
         4-single)"
    );
}

#[test]
fn e2e_reduce_sum_in_return() {
    let s = one_node_with_scores(7, ints(&[1, 2, 3, 4]));
    let rows = run(
        "MATCH (n:X) RETURN reduce(s = 0, x IN [1, 2, 3, 4] | s + x) AS total",
        &s,
        &cat(),
    );
    assert_eq!(rows, vec![vec![Value::Integer(10)]]);
}

#[test]
fn e2e_reduce_over_property_list() {
    // reduce(s = 0, x IN n.scores | s + x) over [10,20,30] ⇒ 60.
    let s = one_node_with_scores(7, ints(&[10, 20, 30]));
    let rows = run(
        "MATCH (n:X) RETURN reduce(s = 0, x IN n.scores | s + x) AS total",
        &s,
        &cat(),
    );
    assert_eq!(rows, vec![vec![Value::Integer(60)]]);
}

#[test]
fn e2e_reduce_null_element_propagates() {
    // reduce(s = 0, x IN [1,null,3] | s + x) ⇒ NULL (pure fold; null
    // propagates as an ordinary value). End-to-end through RETURN.
    let s = one_node_with_scores(7, ints_with_null(&[1], &[3]));
    let rows = run(
        "MATCH (n:X) RETURN reduce(s = 0, x IN n.scores | s + x) AS total",
        &s,
        &cat(),
    );
    assert_eq!(rows, vec![vec![Value::Null]]);
}

#[test]
fn e2e_reduce_widens_to_float() {
    // reduce(s = 0, x IN [1.5, 2.5] | s + x) ⇒ 4.0 (Int acc + Float body
    // WIDENS to Float — Decision 3-reduce-widening / OQ-5; the
    // type-check ACCEPTS by widening rather than rejecting, and the
    // runtime produces a Float).
    let s = one_node_with_scores(7, ints(&[1]));
    let rows = run(
        "MATCH (n:X) RETURN reduce(s = 0, x IN [1.5, 2.5] | s + x) AS total",
        &s,
        &cat(),
    );
    assert_eq!(rows, vec![vec![Value::Float(4.0)]]);
}

// =====================================================================
// Nested — all inside any (cross-scope variable reference)
// =====================================================================

#[test]
fn e2e_nested_all_inside_any() {
    // any(x IN [1,2] WHERE all(y IN [10,20] WHERE y > x)) ⇒ true.
    // The inner predicate `y > x` references BOTH scoped vars (the outer
    // `x` is visible inside the inner scope — the binder's reverse
    // scope-walk + the evaluator's extended-row composition).
    let s = one_node_with_scores(7, ints(&[1]));
    let rows = run(
        "MATCH (n:X) RETURN any(x IN [1, 2] WHERE all(y IN [10, 20] WHERE y > x)) AS ok",
        &s,
        &cat(),
    );
    assert_eq!(rows, vec![vec![Value::Boolean(true)]]);
}

// =====================================================================
// Type-check rejection paths (the reject side of OQ-5 + list operand)
// =====================================================================

#[test]
fn e2e_reduce_int_acc_string_body_rejected_at_typecheck() {
    // reduce(s = 0, x IN [1, 2] | 'z') — the body types to String while
    // the accumulator is Integer. `reduce_join_type(Integer, String)` is
    // genuinely non-assignable (not both numeric, not equal) ⇒
    // TypeCheckError. This is the REJECT side of OQ-5 (Decision
    // 3-reduce-widening: genuinely non-assignable folds remain a
    // TypeCheckError) — it MUST error, not silently accept.
    //
    // NOTE: we use a String-LITERAL body (`'z'`) rather than `s + x`
    // over a `['a','b']` list, because at v1.0 a list literal's element
    // type erases to `Null` (`literal_type(List) ⇒ List(Null)`), so an
    // element-type-driven `Int + String` fold would 3VL-propagate to
    // `Null` rather than reaching the reject path. The string-literal
    // body types to a concrete `String`, which is what triggers
    // `reduce_join_type`'s non-assignable rejection.
    let query = "MATCH (n:X) RETURN reduce(s = 0, x IN [1, 2] | 'z') AS bad";
    let stmt = parse(query).expect("parse");
    let c = cat();
    let mut bound = BindingVisitor::bind(&stmt, query, &c).expect("bind");
    let res = TypeCheckVisitor::check(&mut bound, &c);
    assert!(
        res.is_err(),
        "reduce with Int acc + String body MUST fail type-check (OQ-5 reject side)"
    );
}

#[test]
fn e2e_reduce_int_acc_float_body_accepts_via_widening() {
    // reduce(s = 0, x IN [1] | s + 1.5) — the body `s + 1.5` types to
    // Float (Int + Float → Float). `reduce_join_type(Integer, Float)`
    // WIDENS to the numeric join `Float` rather than rejecting (OQ-5
    // accept side — a false-reject is as much a conformance failure as a
    // false-accept). The type-check MUST succeed AND the runtime MUST
    // produce a Float.
    let query = "MATCH (n:X) RETURN reduce(s = 0, x IN [1] | s + 1.5) AS ok";
    let stmt = parse(query).expect("parse");
    let c = cat();
    let mut bound = BindingVisitor::bind(&stmt, query, &c).expect("bind");
    let res = TypeCheckVisitor::check(&mut bound, &c);
    assert!(
        res.is_ok(),
        "reduce with Int acc + Float body MUST type-check via widening (OQ-5 accept side)"
    );
    // And end-to-end: 0 + 1.5 = 1.5 (one element).
    let s = one_node_with_scores(7, ints(&[1]));
    let rows = run(query, &s, &cat());
    assert_eq!(rows, vec![vec![Value::Float(1.5)]]);
}

#[test]
fn e2e_all_over_non_list_rejected_at_typecheck() {
    // all(x IN 5 WHERE x > 0) — the list operand `5` is an Integer, not
    // List(_)/Null. The type-check MUST reject (Decision 3 list-operand
    // rule), not surface a silent runtime surprise.
    let query = "MATCH (n:X) WHERE all(x IN 5 WHERE x > 0) RETURN n.g";
    let stmt = parse(query).expect("parse");
    let c = cat();
    let mut bound = BindingVisitor::bind(&stmt, query, &c).expect("bind");
    let res = TypeCheckVisitor::check(&mut bound, &c);
    assert!(
        res.is_err(),
        "all() over a non-list operand MUST fail type-check (Decision 3)"
    );
}

// =====================================================================
// Parse rejection — the WHERE is required for the four quantifiers
// =====================================================================

#[test]
fn e2e_all_without_where_rejected_at_parse() {
    // `all(x IN [1,2,3])` with NO WHERE — the four quantifiers REQUIRE a
    // predicate (Decision 4 tables key on it). We reject at PARSE time
    // with a precise message rather than defaulting to a silent `true`.
    let res = parse("MATCH (n:X) WHERE all(x IN [1, 2, 3]) RETURN n.g");
    assert!(
        res.is_err(),
        "all(x IN list) without WHERE MUST fail to parse (no silent default)"
    );
}

#[test]
fn e2e_lowercase_property_named_single_still_parses() {
    // The keyword wrappers are case-insensitive but NOT added to the
    // `keyword` identifier-exclusion set (which is case-sensitive
    // UPPERCASE), so a lowercase property access `n.single` MUST still
    // parse as a property — proving we didn't over-reserve the new
    // keywords (ADR-188 Consequences).
    let c = StubCatalogProvider::new()
        .with_labels(["X"])
        .with_properties(["single", "any", "none", "reduce"]);
    for prop in ["single", "any", "none", "reduce"] {
        let q = format!("MATCH (n:X) RETURN n.{prop}");
        assert!(
            parse(&q).is_ok(),
            "lowercase property n.{prop} MUST still parse as a property access"
        );
        // And it binds + type-checks (a property access, not a keyword).
        let stmt = parse(&q).expect("parse");
        let bound = BindingVisitor::bind(&stmt, &q, &c);
        assert!(bound.is_ok(), "n.{prop} MUST bind as a property access");
    }
}
