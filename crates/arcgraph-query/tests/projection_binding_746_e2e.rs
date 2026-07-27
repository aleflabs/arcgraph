//! **#746** — binder↔`ProjectOp` output-binding-id contract, END-TO-END.
//!
//! These exercise the FULL pipeline (parse → bind → type-check →
//! cross-substrate → lower → execute) for the two surfaces the #746
//! projection-binding-id mismatch blocked at runtime:
//!
//! 1. **Project-over-Aggregate** — `RETURN <agg>` lowers to
//!    `Project(Aggregate(..))`; before #746 the outer `Project`
//!    re-evaluated `count(n)` against the `Aggregate`'s synthetic-id
//!    output and errored "binding ... missing from row schema". The
//!    lowering now rewrites the projection to a `VariableRef` of the
//!    aggregation's output id, and the `Aggregate` emits under that id.
//! 2. **WITH-projection** — `WITH <expr> AS a ... <ref to a>`; before
//!    #746 the WITH `Project` emitted a synthetic id for `a` that the
//!    downstream clause could not resolve.
//!
//! STRONG `==` oracles over the exact result rows (ADR-133 §D-4
//! Query-class active-verification surface for this slice). Aggregate /
//! group output ordering is not guaranteed, so multi-row oracles sort
//! before comparing.

use arcgraph_core::{LabelId, NodeId, TenantId};
use arcgraph_query::executor::value::NodeView;
use arcgraph_query::executor::{ExecutionContext, StubExecutorSubstrate, Value};
use arcgraph_query::logical_plan::LogicalPlanLoweringVisitor;
use arcgraph_query::semantic::{
    BindingVisitor, CatalogProvider, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};
use arcgraph_query::{materialize, parse};

const LABEL_X: u32 = 1;

fn cat() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["X", "Y"])
        .with_properties(["x", "city"])
}

/// Three `:X` nodes: x∈{10,20,30}, city∈{"A","A","B"}. No `:Y` nodes
/// (so a `:Y` scan is the empty-input single-row-aggregate case).
fn substrate() -> StubExecutorSubstrate {
    let x = |id: u64, xv: i64, city: &str| {
        NodeView::new(NodeId::new(id), Some(LabelId::new(LABEL_X)))
            .with_property("x", Value::Integer(xv))
            .with_property("city", Value::String(city.into()))
    };
    StubExecutorSubstrate::new()
        .with_node(TenantId::DEFAULT, x(1, 10, "A"))
        .with_node(TenantId::DEFAULT, x(2, 20, "A"))
        .with_node(TenantId::DEFAULT, x(3, 30, "B"))
}

/// Full pipeline → result rows (panics on any stage error).
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

fn int1(rows: &[Vec<Value>]) -> i64 {
    assert_eq!(rows.len(), 1, "expected exactly one row: {rows:?}");
    assert_eq!(rows[0].len(), 1, "expected exactly one column: {rows:?}");
    match rows[0][0] {
        Value::Integer(n) => n,
        ref o => panic!("expected Integer, got {o:?}"),
    }
}

fn sorted_ints(rows: &[Vec<Value>]) -> Vec<i64> {
    let mut got: Vec<i64> = rows
        .iter()
        .map(|r| match r[0] {
            Value::Integer(n) => n,
            ref o => panic!("expected Integer, got {o:?}"),
        })
        .collect();
    got.sort_unstable();
    got
}

// =====================================================================
// 1. Project-over-Aggregate — the headline repro.
// =====================================================================

#[test]
fn match_return_count_executes() {
    // `MATCH (n:X) RETURN count(n)` ⇒ [[3]]. THE headline #746 repro:
    // a plain aggregate with NO new feature involved. Before #746 this
    // errored at runtime with "binding ... missing from row schema".
    let s = substrate();
    let rows = run("MATCH (n:X) RETURN count(n)", &s, &cat());
    assert_eq!(int1(&rows), 3, "3 :X nodes");
}

#[test]
fn match_return_count_empty_input_single_row_zero() {
    // `MATCH (n:Y) RETURN count(n)` ⇒ [[0]] — the single-row aggregate
    // empty-input rule (amendment-03 §TIER-2-b): no `:Y` nodes, no
    // group-by, so ONE row with count 0. Proves the Project-over-empty-
    // Aggregate path resolves the aggregate output id too.
    let s = substrate();
    let rows = run("MATCH (n:Y) RETURN count(n)", &s, &cat());
    assert_eq!(int1(&rows), 0, "no :Y nodes ⇒ count 0, one row");
}

#[test]
fn match_return_group_by_with_aggregate() {
    // `MATCH (n:X) RETURN n.city AS city, count(n) AS c` ⇒ implicit
    // GROUP BY city: A→2, B→1. Exercises BOTH the group-by column output
    // id AND the aggregation output id flowing to the layered Project.
    let s = substrate();
    let rows = run(
        "MATCH (n:X) RETURN n.city AS city, count(n) AS c",
        &s,
        &cat(),
    );
    let mut pairs: Vec<(String, i64)> = rows
        .iter()
        .map(|r| match (&r[0], &r[1]) {
            (Value::String(city), Value::Integer(c)) => (city.clone(), *c),
            other => panic!("expected (String, Integer), got {other:?}"),
        })
        .collect();
    pairs.sort();
    assert_eq!(pairs, vec![("A".to_string(), 2), ("B".to_string(), 1)]);
}

#[test]
fn match_return_aggregate_then_group_reorders_columns() {
    // `RETURN count(n) AS c, n.city AS city` — the aggregation is the
    // FIRST output column but the Aggregate emits group-key columns
    // first. The layered Project must REORDER (agg-then-group → source
    // order) via the #746 output-id passthrough. ⇒ (c, city): (2,"A"),
    // (1,"B").
    let s = substrate();
    let rows = run(
        "MATCH (n:X) RETURN count(n) AS c, n.city AS city",
        &s,
        &cat(),
    );
    let mut pairs: Vec<(i64, String)> = rows
        .iter()
        .map(|r| match (&r[0], &r[1]) {
            (Value::Integer(c), Value::String(city)) => (*c, city.clone()),
            other => panic!("expected (Integer, String), got {other:?}"),
        })
        .collect();
    pairs.sort_by(|a, b| a.1.cmp(&b.1));
    assert_eq!(pairs, vec![(2, "A".to_string()), (1, "B".to_string())]);
}

// =====================================================================
// 2. WITH-projection — the second blocked surface.
// =====================================================================

#[test]
fn match_with_projection_then_return() {
    // `MATCH (n:X) WITH n.x AS a RETURN a` ⇒ [10],[20],[30]. The WITH
    // `Project` emits `a` under the binder-assigned output id; the
    // downstream `RETURN a` resolves `a` to the SAME id (before #746 the
    // synthetic id mismatch errored).
    let s = substrate();
    let rows = run("MATCH (n:X) WITH n.x AS a RETURN a", &s, &cat());
    assert_eq!(sorted_ints(&rows), vec![10, 20, 30]);
}

#[test]
fn unwind_then_with_passthrough_then_return() {
    // `UNWIND [1,2,3] AS x WITH x RETURN x` ⇒ [1],[2],[3]. WITH
    // passthrough of the unwound element: the WITH `Project` re-emits
    // `x` under the post-WITH binding id the downstream RETURN resolves.
    let s = substrate();
    let rows = run("UNWIND [1, 2, 3] AS x WITH x RETURN x", &s, &cat());
    assert_eq!(sorted_ints(&rows), vec![1, 2, 3]);
}

#[test]
fn match_with_aggregate_then_return_reference() {
    // `MATCH (n:X) WITH count(n) AS c RETURN c` ⇒ [[3]]. The WITH-level
    // aggregate's output id IS the post-WITH binding `c`; the layered
    // Project passes it through under `c`, and the downstream `RETURN c`
    // resolves the SAME id. Chains the Project-over-Aggregate fix with
    // the WITH-projection fix.
    let s = substrate();
    let rows = run("MATCH (n:X) WITH count(n) AS c RETURN c", &s, &cat());
    assert_eq!(int1(&rows), 3, "3 :X nodes, projected through WITH");
}
