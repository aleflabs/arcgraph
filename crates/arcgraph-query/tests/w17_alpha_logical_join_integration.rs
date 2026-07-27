//! W17α M4-08+ — `LogicalJoin` executor integration tests.
//!
//! Pins:
//!
//! 1. `multi_pattern_match_executes_end_to_end_via_query_engine` —
//!    parse → bind → typecheck → lower → enumerate → execute on a
//!    multi-pattern MATCH `MATCH (a:Person), (a)-[:KNOWS]->(b)`
//!    returns the expected row set.
//! 2. `multi_pattern_match_with_no_shared_binding_returns_cartesian` —
//!    Cartesian-shape MATCH `MATCH (a:Person), (b:Person)` emits
//!    every (a, b) pair.
//! 3. `multi_pattern_match_against_empty_substrate_returns_zero_rows` —
//!    empty substrate yields zero joined rows (no NotImplemented).
//! 4. `three_pattern_chain_executes_via_two_joins` — a chain
//!    `MATCH (a)-[r1]->(b), (b)-[r2]->(c)` lowers to two joined
//!    expand patterns; the executor walks both joins.
//! 5. `multi_pattern_match_observer_attributes_hash_join_kind` — the
//!    M4-71 row-count observer counts batches under the dedicated
//!    `OperatorKind::HashJoin` slug, not `Empty`.

#![allow(clippy::too_many_lines)]

use arcgraph_core::{LabelId, NodeId, RelId, TenantId, TypeId};
use arcgraph_query::QueryEngine;
use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::executor::value::{NodeView, RelView, Value};
use arcgraph_query::observer::{OperatorKind, RowCountObserver};
use arcgraph_query::semantic::StubCatalogProvider;

fn cat_person_knows() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["Person"])
        .with_rel_types(["KNOWS"])
        .with_properties(["name", "age"])
}

/// 3 Persons (1=Alice, 2=Bob, 3=Carol). Alice -[KNOWS]-> Bob;
/// Alice -[KNOWS]-> Carol. Bob and Carol have no outbound edges.
fn substrate_alice_knows_bob_and_carol() -> StubExecutorSubstrate {
    StubExecutorSubstrate::new()
        .with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(1), Some(LabelId::new(1))),
        )
        .with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(2), Some(LabelId::new(1))),
        )
        .with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(3), Some(LabelId::new(1))),
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
        .with_edge(
            TenantId::DEFAULT,
            RelView::new(
                RelId::new(11),
                NodeId::new(1),
                NodeId::new(3),
                Some(TypeId::new(1)),
            ),
        )
}

#[test]
fn multi_pattern_match_executes_end_to_end_via_query_engine() {
    // Multi-pattern MATCH with a shared variable lowers to a
    // LogicalJoin on `a`. The QueryEngine routes parse → bind →
    // typecheck → lower → enumerate → execute; the W17α executor
    // body emits real rows (NOT NotImplemented).
    //
    // Expected: 2 rows — Alice/Bob and Alice/Carol.
    let s = substrate_alice_knows_bob_and_carol();
    let cat = cat_person_knows();
    let engine = QueryEngine::new(&cat);
    let result = engine
        .execute("MATCH (a:Person), (a)-[r:KNOWS]->(b) RETURN a, r, b", &s)
        .expect("execute multi-pattern");
    assert_eq!(
        result.len(),
        2,
        "Alice has 2 KNOWS edges; expected 2 joined rows"
    );
    // Each row's `a` is Alice (id=1) and `b` is in {Bob=2, Carol=3}.
    for row in result.rows() {
        assert_eq!(row.len(), 3, "RETURN a, r, b → 3 columns");
        match &row[0] {
            Value::Node(n) => assert_eq!(n.id, NodeId::new(1), "a is Alice"),
            other => panic!("expected Node for `a`, got {other:?}"),
        }
        match &row[2] {
            Value::Node(n) => assert!(
                n.id == NodeId::new(2) || n.id == NodeId::new(3),
                "b is Bob or Carol; got {:?}",
                n.id
            ),
            other => panic!("expected Node for `b`, got {other:?}"),
        }
    }
}

#[test]
fn multi_pattern_match_with_no_shared_binding_returns_cartesian() {
    // `MATCH (a:Person), (b:Person)` has NO shared bindings →
    // Cartesian join. With 3 persons in the substrate, expected
    // row count = 3 × 3 = 9.
    let s = substrate_alice_knows_bob_and_carol();
    let cat = cat_person_knows();
    let engine = QueryEngine::new(&cat);
    let result = engine
        .execute("MATCH (a:Person), (b:Person) RETURN a, b", &s)
        .expect("execute cartesian");
    assert_eq!(result.len(), 9, "3 × 3 cartesian product");
}

#[test]
fn multi_pattern_match_against_empty_substrate_returns_zero_rows() {
    // Empty substrate; multi-pattern query must execute (not
    // NotImplemented) and emit zero rows.
    let s = StubExecutorSubstrate::new();
    let cat = cat_person_knows();
    let engine = QueryEngine::new(&cat);
    let result = engine
        .execute("MATCH (a:Person), (a)-[r:KNOWS]->(b) RETURN a, b", &s)
        .expect("execute empty");
    assert_eq!(result.len(), 0, "empty substrate produces zero joined rows");
}

#[test]
fn three_pattern_chain_executes_via_two_joins() {
    // Build a chain Alice -[KNOWS]-> Bob -[KNOWS]-> Carol.
    let s = StubExecutorSubstrate::new()
        .with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(1), Some(LabelId::new(1))),
        )
        .with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(2), Some(LabelId::new(1))),
        )
        .with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(3), Some(LabelId::new(1))),
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
        .with_edge(
            TenantId::DEFAULT,
            RelView::new(
                RelId::new(11),
                NodeId::new(2),
                NodeId::new(3),
                Some(TypeId::new(1)),
            ),
        );
    let cat = cat_person_knows();
    let engine = QueryEngine::new(&cat);
    let result = engine
        .execute(
            "MATCH (a:Person)-[r1:KNOWS]->(b), (b)-[r2:KNOWS]->(c) RETURN a, b, c",
            &s,
        )
        .expect("execute 3-pattern chain");
    assert_eq!(result.len(), 1, "Alice -> Bob -> Carol is the single chain");
    let row = &result.rows()[0];
    match (&row[0], &row[1], &row[2]) {
        (Value::Node(a), Value::Node(b), Value::Node(c)) => {
            assert_eq!(a.id, NodeId::new(1));
            assert_eq!(b.id, NodeId::new(2));
            assert_eq!(c.id, NodeId::new(3));
        }
        other => panic!("expected (Node, Node, Node); got {other:?}"),
    }
}

#[test]
fn multi_pattern_match_observer_attributes_hash_join_kind() {
    // Drive a multi-pattern query with an attached RowCountObserver,
    // then assert the per-kind metrics include the dedicated
    // HashJoin slug. This pins the M4-71 observer-side wiring for
    // W17α / M4-08+ — multi-pattern queries no longer attribute
    // their batches to `OperatorKind::Empty`.
    use arcgraph_query::executor::{ExecutionContext, execute_with_context};
    use arcgraph_query::planner::cost::estimate_costs;
    use arcgraph_query::semantic::{BindingVisitor, CatalogProvider, TypeCheckVisitor};
    use std::sync::Arc;

    let s = substrate_alice_knows_bob_and_carol();
    let cat = cat_person_knows();

    const QUERY: &str = "MATCH (a:Person), (a)-[r:KNOWS]->(b) RETURN a, b";

    let stmt = arcgraph_query::parser::parse(QUERY).expect("parse");
    let mut bound = BindingVisitor::bind(&stmt, QUERY, &cat).expect("bind");
    TypeCheckVisitor::check(&mut bound, &cat).expect("typecheck");
    let lowered =
        arcgraph_query::logical_plan::LogicalPlanLoweringVisitor::lower(&bound).expect("lower");
    let costed = estimate_costs(lowered.clone(), &cat);
    let (plan_for_exec, cost_tree) = costed.into_parts();

    let observer = Arc::new(RowCountObserver::from_plan_and_costs(
        &plan_for_exec,
        &cost_tree,
    ));
    let ctx =
        ExecutionContext::new(cat.tenant(), cat.partition()).with_observer(Arc::clone(&observer));
    let _result = execute_with_context(&plan_for_exec, &s, &ctx).expect("execute");

    // Verify the observer attributed at least one batch to HashJoin.
    let metrics = observer.metrics();
    let hash_join = metrics
        .iter()
        .find(|m| m.op_kind == Some(OperatorKind::HashJoin))
        .expect("HashJoin entry in per-kind metrics");
    assert!(
        hash_join.batches >= 1,
        "HashJoin must record at least one batch (got {})",
        hash_join.batches
    );
    assert!(
        hash_join.observed_rows >= 2,
        "HashJoin observed_rows must reflect the 2 joined output rows (got {})",
        hash_join.observed_rows
    );
}
