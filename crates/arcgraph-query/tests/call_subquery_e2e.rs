//! **ADR-192 / #623** — `CALL { <subquery> }` correlated brace-subquery
//! END-TO-END tests (Cypher 25 — a deliberate beyond-openCypher-v9
//! capability extension).
//!
//! These exercise the FULL pipeline (parse → bind → type-check →
//! cross-substrate → lower → execute) for `CALL { … }`. They ARE the
//! ADR-192 oracle (tests 1-13): `CALL{}` is NOT in the vendored
//! openCypher v9 TCK (v9's `CALL` is a procedure call, scoped out), so
//! the capability is proven HERE — by real queries with STRONG `==`
//! oracles over the exact result rows — NOT by a TCK bucket.
//!
//! The ADR-133 §D-4 Query-class active-verification surface for this
//! slice. Coverage of ADR-192's binding test plan:
//! - **Test 1** leading uncorrelated CALL{} (`== 1` output row, D-5a).
//! - **Test 2** correlated implicit import (per-driving-row expansion).
//! - **Test 3** cardinality multiply (a driving row repeated k-fold).
//! - **Test 4** empty-subquery DROPS the driving row (inner correlation).
//! - **Test 5** correlated AGGREGATE PRESERVES the driving row (deg=0,
//!   NOT dropped) — the D-7/D-8 correctness hinge; a naive
//!   measure-on-input CallOp would wrongly drop it.
//! - **Test 6** scoping fence (an inner non-returned var is not visible
//!   after `}`).
//! - **Test 7** RETURN-collision with an outer variable → bind error.
//! - **Test 8** write-in-CALL{} → `WriteInCallSubqueryNotSupported`.
//! - **Test 9** nested CALL{} (correlated import composes across levels).
//! - **Test 10** UNION inside the body.
//! - **Test 13** body-tail LIMIT is per-driving-row scoped (resets per
//!   row, does not leak across driving rows).
//!
//! Test 11 (cross-batch `Pending` carry-over) is a body that fans out
//! past `BATCH_ROWS` for ONE driving row — exercised directly + with a
//! strong no-truncation/no-dup oracle in the `CallOp` unit test
//! `call_crosses_batch_boundary_without_truncation_or_duplication`
//! (`src/executor/ops/call.rs`); not re-built here because synthesizing
//! 2048-plus substrate rows per driving row in an integration fixture
//! adds no oracle strength over the unit test.

use arcgraph_core::{LabelId, NodeId, RelId, TenantId, TypeId};
use arcgraph_query::executor::ExecutionContext;
use arcgraph_query::executor::value::{NodeView, RelView, Value};
use arcgraph_query::logical_plan::LogicalPlanLoweringVisitor;
use arcgraph_query::semantic::error::BindingError;
use arcgraph_query::semantic::{
    BindingVisitor, CatalogProvider, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};
use arcgraph_query::{materialize, parse};

const SERVICE: u32 = 1; // first label ⇒ LabelId::new(1)
const CALLS: u32 = 1; // first rel-type ⇒ TypeId::new(1)

fn cat() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["Service", "Other"])
        .with_rel_types(["CALLS"])
        .with_properties(["k"])
}

fn svc(id: u64) -> NodeView {
    NodeView::new(NodeId::new(id), Some(LabelId::new(SERVICE)))
}

fn calls(rel_id: u64, from: u64, to: u64) -> RelView {
    RelView::new(
        RelId::new(rel_id),
        NodeId::new(from),
        NodeId::new(to),
        Some(TypeId::new(CALLS)),
    )
}

/// The call graph fixture:
///   s1 -CALLS-> s2
///   s2 -CALLS-> s3
///   s2 -CALLS-> s1
///   s3 has NO outgoing CALLS.
/// Out-degrees (CALLS): s1=1, s2=2, s3=0.
fn call_graph() -> StubCatalogProvider {
    cat()
}

fn call_graph_substrate() -> arcgraph_query::executor::StubExecutorSubstrate {
    use arcgraph_query::executor::StubExecutorSubstrate;
    StubExecutorSubstrate::new()
        .with_node(TenantId::DEFAULT, svc(1))
        .with_node(TenantId::DEFAULT, svc(2))
        .with_node(TenantId::DEFAULT, svc(3))
        .with_edge(TenantId::DEFAULT, calls(100, 1, 2))
        .with_edge(TenantId::DEFAULT, calls(101, 2, 3))
        .with_edge(TenantId::DEFAULT, calls(102, 2, 1))
}

/// Full pipeline → result rows (panics on any stage error).
fn run(
    query: &str,
    s: &arcgraph_query::executor::StubExecutorSubstrate,
    c: &StubCatalogProvider,
) -> Vec<Vec<Value>> {
    let plan = lower(query, c);
    let ctx = ExecutionContext::new(c.tenant(), c.partition());
    materialize::materialize(&plan, s, &ctx)
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

/// Full pipeline → result rows OR the EXECUTE-stage error rendered as a
/// String (for the `test5_*` characterization tripwire). Parse / bind /
/// type-check / lower still `expect` (those stages must succeed); only
/// the materialize stage's error is surfaced.
fn run_result(
    query: &str,
    s: &arcgraph_query::executor::StubExecutorSubstrate,
    c: &StubCatalogProvider,
) -> Result<Vec<Vec<Value>>, String> {
    let plan = lower(query, c);
    let ctx = ExecutionContext::new(c.tenant(), c.partition());
    materialize::materialize(&plan, s, &ctx)
        .map(|m| m.rows().to_vec())
        .map_err(|e| format!("{e:?}"))
}

/// Bind-only — for the tests that assert a BIND error (tests 6/7/8).
fn bind_err(query: &str, c: &StubCatalogProvider) -> Vec<BindingError> {
    let stmt = parse(query).expect("parse");
    match BindingVisitor::bind(&stmt, query, c) {
        Ok(_) => panic!("expected a bind error for: {query}"),
        Err(errs) => errs,
    }
}

fn node_id(v: &Value) -> u64 {
    match v {
        Value::Node(n) => n.id.raw(),
        other => panic!("expected Node, got {other:?}"),
    }
}

fn int(v: &Value) -> i64 {
    match v {
        Value::Integer(n) => *n,
        other => panic!("expected Integer, got {other:?}"),
    }
}

// =====================================================================
// Test 1 — leading uncorrelated CALL{} ⇒ EXACTLY 1 output row (D-5a).
// =====================================================================

#[test]
fn test1_leading_uncorrelated_call_emits_exactly_one_row() {
    // `CALL { RETURN 1 AS x } RETURN x` (ADR-192 test 1 / the spawn-brief
    // form) — the leading-clause unit driving row (D-5a) makes the body
    // run EXACTLY once; the body's single-literal projection yields one
    // row. The `== 1` row-count is the STRONG oracle: WITHOUT the D-5a
    // unit row a leading CALL{} has zero driving rows ⇒ zero output ⇒
    // this assertion FAILS (the EXACT #618 UNWIND gap, closed here by the
    // `lower_call` `prev.unwrap_or_else(LogicalEmpty)` idiom).
    //
    // (ADR-192's test-1 table also shows a `count(*)`-bodied variant.
    // `count(*)` is NOT a v1.0 grammar surface — see
    // `aggregation_lowering_integration.rs` §282 "the v1.0 grammar does
    // not light count(*)" — AND full-pipeline aggregate EXECUTION is
    // blocked by a PRE-EXISTING Project-over-Aggregate bug, characterized
    // in `test5_*` below. So the non-aggregate `RETURN 1 AS x` form is the
    // executable D-5a oracle; the aggregating leading-CALL{} composes the
    // moment that orthogonal aggregate-exec bug is fixed.)
    let s = call_graph_substrate();
    let rows = run("CALL { RETURN 1 AS x } RETURN x", &s, &call_graph());
    assert_eq!(
        rows.len(),
        1,
        "leading CALL{{}} emits EXACTLY one row (D-5a)"
    );
    assert_eq!(int(&rows[0][0]), 1, "the body's single-literal projection");
}

// =====================================================================
// Test 2 / Test 3 — correlated implicit import + cardinality multiply.
// =====================================================================

#[test]
fn test2_correlated_import_expands_per_driving_row() {
    // `MATCH (a:Service) CALL { MATCH (a)-[:CALLS]->(b) RETURN b } RETURN a, b`
    // — `a` implicitly imported (Cypher 25, NO importing-WITH); per-`a`
    // CALLS expansion; output schema `a ++ b`. Expected (a,b) pairs
    // (s2 contributes TWO — the cardinality multiply, test 3): (1,2),
    // (2,3), (2,1). s3 (out-degree 0) is ABSENT (test 4 drop).
    let s = call_graph_substrate();
    let rows = run(
        "MATCH (a:Service) CALL { MATCH (a)-[:CALLS]->(b) RETURN b } RETURN a, b",
        &s,
        &call_graph(),
    );
    let mut pairs: Vec<(u64, u64)> = rows
        .iter()
        .map(|r| {
            assert_eq!(r.len(), 2, "output schema is [a, b]");
            (node_id(&r[0]), node_id(&r[1]))
        })
        .collect();
    pairs.sort_unstable();
    assert_eq!(
        pairs,
        vec![(1, 2), (2, 1), (2, 3)],
        "per-driving-row CALLS expansion; s2 multiplied 2× (test 3); s3 dropped (test 4)"
    );
}

// =====================================================================
// Test 4 — empty subquery DROPS the driving row (inner correlation).
// =====================================================================

#[test]
fn test4_empty_subquery_drops_driving_row() {
    // s3 has no outgoing CALLS; its non-aggregating subquery returns 0
    // rows ⇒ s3 is DROPPED (CALL{} is an inner correlation, NOT an
    // OPTIONAL/left-join). Assert s3 (id 3) appears in NO output row.
    let s = call_graph_substrate();
    let rows = run(
        "MATCH (a:Service) CALL { MATCH (a)-[:CALLS]->(b) RETURN b } RETURN a, b",
        &s,
        &call_graph(),
    );
    assert!(
        rows.iter().all(|r| node_id(&r[0]) != 3),
        "s3 (no CALLS) is DROPPED by the non-aggregating empty subquery"
    );
    assert_eq!(rows.len(), 3, "only s1 (1×) + s2 (2×) survive");
}

// =====================================================================
// Test 5 — correlated AGGREGATE PRESERVES the driving row (THE D-8 hinge).
//
// The D-8 correctness hinge — an aggregating body PRESERVES the driving
// row even over an EMPTY correlated set (deg=0, NOT dropped) — is proven
// at the OPERATOR level by the `CallOp` unit test
// `call_preserves_driving_row_when_body_returns_one_row_the_d8_hinge`
// (`src/executor/ops/call.rs`): `CallOp` measures cardinality on the
// body's OUTPUT (D-6), so a body that returns 1 row PRESERVES its
// driving row and a body that returns 0 rows DROPS it — and a NAIVE
// measure-on-input impl FAILS that unit test. That is the load-bearing
// CALL-specific behavior.
//
// The END-TO-END aggregating-body form below (`RETURN count(b)`) was
// BLOCKED by a PRE-EXISTING, CALL-INDEPENDENT executor bug (full-pipeline
// aggregate execution surfaced `Eval("binding … missing from row
// schema")` — `RETURN <aggregate>` lowers to a `Project` OVER an
// `Aggregate` and the `Project` re-evaluated the aggregate expression
// against the `Aggregate`'s fresh-id output schema, where the arg
// binding was absent; reproduced with NO CALL at all, e.g.
// `MATCH (n:X) RETURN count(n)`). #746/#749 fixed it by establishing the
// binder↔`ProjectOp` output-binding-id contract — the `CallOp` already
// drives + counts the body's OUTPUT correctly (D-6), so
// CALL{}-with-aggregate composes end-to-end with NO further CALL change.
//
// This test was the TRIPWIRE pinning the blocked state; it is now FLIPPED
// to assert the D-8 preserve semantics (s1→1, s2→2, s3→0; s3 PRESERVED
// via the openCypher count=0 identity row).
// =====================================================================

#[test]
fn test5_correlated_aggregate_d8_preserve_semantics() {
    // #746/#749 REGRESSION GUARD (was
    // `test5_correlated_aggregate_blocked_by_preexisting_aggregate_exec_bug`).
    // For each `:Service` node `a`, the correlated subquery counts its
    // `:CALLS` out-neighbours; the outer RETURN pairs each service with
    // its out-degree. The D-8 PRESERVE shape: s3 has NO out-edges, but
    // the correlated aggregate emits the openCypher count=0 identity row
    // for the empty inner match, so the outer s3 row SURVIVES with deg=0
    // (a measure-on-INPUT CallOp bug would instead drop it). Before
    // #746/#749 this errored ("missing from row schema"); the
    // binder↔ProjectOp output-binding-id contract unblocked it with NO
    // CALL change. Flipped per `feedback_review_oracle_relaxations`
    // (assert the correct result; do not delete the guard).
    let s = call_graph_substrate();
    let rows = run_result(
        "MATCH (a:Service) CALL { MATCH (a)-[:CALLS]->(b) RETURN count(b) AS deg } RETURN a, deg",
        &s,
        &call_graph(),
    )
    .expect("CALL{}-with-aggregate executes end-to-end (#746/#749)");
    let mut degrees: Vec<(u64, i64)> = rows.iter().map(|r| (node_id(&r[0]), int(&r[1]))).collect();
    degrees.sort_unstable();
    assert_eq!(
        degrees,
        vec![(1, 1), (2, 2), (3, 0)],
        "D-8 preserve: s1→1, s2→2, s3→0 (s3 PRESERVED via the count=0 identity row)"
    );
}

// =====================================================================
// Test 6 — scoping fence: an inner non-returned var is out of scope.
// =====================================================================

#[test]
fn test6_inner_non_returned_var_not_visible_after_brace() {
    // `r` is declared INSIDE the subquery (the rel binding) but NOT
    // returned, so it is OUT of scope after `}`. Referencing it in the
    // outer RETURN is an UndeclaredVariable bind error (only the
    // RETURNed `b` escapes the fence — D-4).
    let errs = bind_err(
        "MATCH (a:Service) CALL { MATCH (a)-[r:CALLS]->(b) RETURN b } RETURN a, r",
        &call_graph(),
    );
    assert!(
        errs.iter().any(|e| matches!(
            e,
            BindingError::UndeclaredVariable { name, .. } if name == "r"
        )),
        "inner non-returned `r` must be undeclared after `}}`; got {errs:?}"
    );
}

#[test]
fn test6_returned_var_is_visible_after_brace() {
    // Complement to test 6: the RETURNed `b` DOES escape the fence and
    // is referenceable in the outer RETURN (no bind error) — proving the
    // fence lets exactly the returned columns through.
    let s = call_graph_substrate();
    let rows = run(
        "MATCH (a:Service) CALL { MATCH (a)-[:CALLS]->(b) RETURN b } RETURN b",
        &s,
        &call_graph(),
    );
    // 3 surviving (a,b) expansions ⇒ 3 `b`s (the outer RETURN projects
    // only b). Every row is a single Node column.
    assert_eq!(rows.len(), 3);
    for r in &rows {
        assert_eq!(r.len(), 1, "outer RETURN b ⇒ one column");
        let _ = node_id(&r[0]);
    }
}

// =====================================================================
// Test 7 — RETURN-collision with an outer variable → bind error.
// =====================================================================

#[test]
fn test7_return_collision_with_outer_variable_is_bind_error() {
    // `MATCH (x) CALL { … RETURN y AS x }` — the subquery RETURN column
    // `x` collides with the outer-bound `x`. Per D-4 this reuses
    // `DuplicateBinding` (the body's returned column is re-declared in
    // the outer scope, where the existing `x` triggers the duplicate).
    let errs = bind_err(
        "MATCH (x:Service) CALL { MATCH (x)-[:CALLS]->(y) RETURN y AS x } RETURN x",
        &call_graph(),
    );
    assert!(
        errs.iter().any(|e| matches!(
            e,
            BindingError::DuplicateBinding { name, .. } if name == "x"
        )),
        "subquery RETURN column `x` must collide with outer `x` (DuplicateBinding); got {errs:?}"
    );
}

// =====================================================================
// Test 8 — write-in-CALL{} → WriteInCallSubqueryNotSupported (D-9).
// =====================================================================

#[test]
fn test8_write_clause_in_call_body_is_rejected() {
    // A CREATE inside CALL{} is rejected at bind (v1.0-α read-only fence,
    // D-9; write-in-CALL is forward-deferred to v1.1).
    let errs = bind_err(
        "MATCH (a:Service) CALL { CREATE (b:Service) } RETURN a",
        &call_graph(),
    );
    assert!(
        errs.iter()
            .any(|e| matches!(e, BindingError::WriteInCallSubqueryNotSupported { .. })),
        "CREATE inside CALL{{}} must be rejected (D-9); got {errs:?}"
    );
}

#[test]
fn test8_merge_in_call_body_is_rejected() {
    // The fence covers the whole write family — MERGE too.
    let errs = bind_err(
        "MATCH (a:Service) CALL { MERGE (b:Service) } RETURN a",
        &call_graph(),
    );
    assert!(
        errs.iter()
            .any(|e| matches!(e, BindingError::WriteInCallSubqueryNotSupported { .. })),
        "MERGE inside CALL{{}} must be rejected (D-9); got {errs:?}"
    );
}

#[test]
fn test8_nested_write_clause_in_call_body_is_rejected() {
    // The D-9 read-only fence is RECURSIVE. `call_body_has_write_clause`
    // scans only the IMMEDIATE level, but each nested CALL{} body is
    // re-scanned when its clause is bound (recursion through
    // `bind_call_clause`, binding.rs). So a CREATE nested TWO CALL levels
    // deep — `CALL { CALL { CREATE … } }` — still trips the fence. Extends
    // the top-level `test8_write_clause_in_call_body_is_rejected` to the
    // nested case (#744 R1, LOW-2).
    let errs = bind_err(
        "MATCH (a:Service) CALL { CALL { CREATE (b:Service) } } RETURN a",
        &call_graph(),
    );
    assert!(
        errs.iter()
            .any(|e| matches!(e, BindingError::WriteInCallSubqueryNotSupported { .. })),
        "CREATE inside a NESTED CALL{{}} must be rejected (D-9, recursive); got {errs:?}"
    );
}

// =====================================================================
// Test 9 — nested CALL{}: correlated import composes across levels.
// =====================================================================

#[test]
fn test9_nested_call_correlated_import_composes() {
    // `UNWIND [1,2] AS n CALL { CALL { RETURN n AS inner } RETURN inner AS sq } RETURN n, sq`
    // — `n` is implicitly imported TWO CALL levels deep (top → outer
    // body → inner body). The inner subquery's `inner` column flows out
    // as the outer subquery's `sq`. Expected (n, sq): (1,1), (2,2). This
    // exercises the correlation-frame STACK (inner + outer frames
    // coexist; nearest-frame-wins resolves `n`).
    let s = call_graph_substrate(); // unused by the literal-driven body
    let rows = run(
        "UNWIND [1, 2] AS n CALL { CALL { RETURN n AS inner } RETURN inner AS sq } RETURN n, sq",
        &s,
        &call_graph(),
    );
    let mut got: Vec<(i64, i64)> = rows
        .iter()
        .map(|r| {
            assert_eq!(r.len(), 2, "output schema is [n, sq]");
            (int(&r[0]), int(&r[1]))
        })
        .collect();
    got.sort_unstable();
    assert_eq!(
        got,
        vec![(1, 1), (2, 2)],
        "nested correlated import: sq == inner == n at every level"
    );
}

// =====================================================================
// Test 10 — UNION inside the subquery body.
// =====================================================================

#[test]
fn test10_union_inside_call_body() {
    // `CALL { RETURN 1 AS x UNION RETURN 2 AS x } RETURN x` — the body
    // is a UNION of two single-row arms; the union'd rows come back per
    // (here single, leading) driving row. Expected x ∈ {1, 2}.
    let s = call_graph_substrate();
    let rows = run(
        "CALL { RETURN 1 AS x UNION RETURN 2 AS x } RETURN x",
        &s,
        &call_graph(),
    );
    let mut xs: Vec<i64> = rows.iter().map(|r| int(&r[0])).collect();
    xs.sort_unstable();
    assert_eq!(xs, vec![1, 2], "UNION inside CALL{{}} returns both arms");
}

// =====================================================================
// Test (implicit-import expression) — the Cypher-25 implicit-import form.
// =====================================================================

#[test]
fn correlated_import_in_return_expression() {
    // `UNWIND [1,2,3] AS n CALL { RETURN n*n AS sq } RETURN n, sq` — the
    // body implicitly imports `n` (NO importing-WITH — the Cypher-25
    // form) and uses it in a RETURN expression. Expected (n, sq):
    // (1,1), (2,4), (3,9).
    let s = call_graph_substrate();
    let rows = run(
        "UNWIND [1, 2, 3] AS n CALL { RETURN n * n AS sq } RETURN n, sq",
        &s,
        &call_graph(),
    );
    let mut got: Vec<(i64, i64)> = rows.iter().map(|r| (int(&r[0]), int(&r[1]))).collect();
    got.sort_unstable();
    assert_eq!(got, vec![(1, 1), (2, 4), (3, 9)]);
}

// =====================================================================
// Test 13 — body-tail LIMIT is PER-DRIVING-ROW scoped.
// =====================================================================

#[test]
fn test13_body_limit_is_per_driving_row_scoped() {
    // `MATCH (a:Service) CALL { MATCH (a)-[:CALLS]->(b) RETURN b LIMIT 1 } RETURN a, b`
    // — the body's LIMIT 1 applies PER driving row (it RESETS per row,
    // does NOT leak across rows). s1 (1 target) ⇒ 1 row; s2 (2 targets)
    // ⇒ 1 row (limited, NOT 0 — the limit did not get "used up" by s1);
    // s3 (0 targets) ⇒ dropped. So exactly 2 rows, one per surviving
    // driving row, each with ≤1 `b`. If the LIMIT leaked across driving
    // rows, s2 would yield 0 rows after s1 consumed the budget → only 1
    // total — this assertion FAILS on that bug.
    let s = call_graph_substrate();
    let rows = run(
        "MATCH (a:Service) CALL { MATCH (a)-[:CALLS]->(b) RETURN b LIMIT 1 } RETURN a, b",
        &s,
        &call_graph(),
    );
    let mut driving: Vec<u64> = rows.iter().map(|r| node_id(&r[0])).collect();
    driving.sort_unstable();
    assert_eq!(
        driving,
        vec![1, 2],
        "LIMIT 1 resets per driving row: s1 + s2 each contribute exactly 1 (s3 dropped)"
    );
}
