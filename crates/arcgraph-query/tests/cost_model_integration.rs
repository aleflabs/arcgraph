//! Integration tests for the M4-51 (M4-05a) cost model.
//!
//! Two end-to-end scenarios:
//!
//! 1. **Simple plan: Scan → Filter → Project → Limit.**
//!    Pins the cost-walker's per-operator threading: scan cardinality
//!    derived from `CatalogProvider::label_cardinality`, filter
//!    selectivity composed via the predicate walker, project + limit
//!    cost-additive but cardinality-preserving / cardinality-capping.
//!
//! 2. **Hybrid retrieval: Project → Limit → RankByHybrid (Vector +
//!    Text).**
//!    Pins the cost-walker's hybrid-leaf handling: per-operand cost
//!    aggregation + RRF fusion-step + output cardinality bounded by
//!    the largest operand's K. Mirrors the LDBC SNB-class
//!    hybrid-retrieval shape M4-05 + M4-06 will execute.
//!
//! # ADR provenance
//! - ADR-038 amendment-02 §M4.e — M4-51 slice contract.
//! - ADR-036 §D-25 — 5 ms M4-05 plan-build budget; the cost model
//!   must produce a coherent cost output well within budget.
//! - ADR-038 §2 D-25 — `CatalogProvider::snapshot()` cross-key
//!   consistency contract (consumer side; M4-04e producer in
//!   `arcgraph-storage`).
//! - ADR-038 §2 D-27 — `SelectivityEstimator` predicate-class
//!   surface (M4-42 input).

use arcgraph_core::{LabelId, Lsn, TypeId};
use arcgraph_query::ast::BinOp;
use arcgraph_query::error::Span;
use arcgraph_query::logical_plan::{
    HybridOperand, HybridOperandKind, LogicalEmpty, LogicalFilter, LogicalLimit, LogicalPlan,
    LogicalProject, LogicalRankByHybrid, LogicalScan,
};
use arcgraph_query::planner::cost::{CostedPlan, estimate_costs};
use arcgraph_query::semantic::StubCatalogProvider;
use arcgraph_query::semantic::bound_ast::{BindingId, BoundExpression};

fn span() -> Span {
    Span::point(1, 1)
}

fn var_ref(id: u64) -> BoundExpression {
    BoundExpression::VariableRef {
        name: format!("v{}", id),
        binding_id: BindingId::new(id),
        span: span(),
        type_info: None,
    }
}

#[test]
fn cost_model_end_to_end_scan_filter_project_limit() {
    // Pipeline:
    //   Scan(label=Person)        — 1_000 rows (catalog cardinality)
    //   ↓
    //   Filter(n.x = $p)          — selectivity = 1/total = 1/10_000 = 0.0001
    //   ↓                          → 1_000 * 0.0001 = 0.1 rows
    //   Project(...)              — cardinality preserved
    //   ↓
    //   Limit(50)                  — capped at min(0.1, 50) = 0.1
    //
    // The cost should be the sum of per-operator costs:
    //   Scan:    1_000 * SCAN_COST_PER_ROW = 1_000 * 1.0 = 1_000
    //   Filter:  1_000 * FILTER_COST_PER_ROW = 1_000 * 0.1 = 100
    //   Project: 1_000 * PROJECT_COST_PER_ROW = 1_000 * 0.05 = 50  (input is post-filter)
    //   Limit:   0.1 * LIMIT_COST_PER_ROW = 0.1 * 0.01 ≈ 0.001
    //
    // Total ≈ 1_150 (dominated by scan).
    let person = LabelId::new(1);
    let cat = StubCatalogProvider::new()
        .with_total_node_count(10_000)
        .with_label_cardinality(person, 1_000);

    let scan = LogicalPlan::Scan(LogicalScan {
        label: Some(person),
        var: BindingId::new(0),
        read_lsn: Lsn::MAX,
        span: span(),
    });

    let predicate = BoundExpression::BinaryOp {
        op: BinOp::Eq,
        lhs: Box::new(var_ref(0)),
        rhs: Box::new(BoundExpression::Parameter {
            name: "p".into(),
            span: span(),
            type_info: None,
        }),
        span: span(),
        type_info: None,
    };
    let filter = LogicalPlan::Filter(LogicalFilter {
        input: Box::new(scan),
        predicate,
        span: span(),
    });
    let project = LogicalPlan::Project(LogicalProject {
        input: Box::new(filter),
        items: Vec::new(),
        span: span(),
    });
    let limit = LogicalPlan::Limit(LogicalLimit {
        input: Box::new(project),
        count: 50,
        span: span(),
    });

    let costed: CostedPlan = estimate_costs(limit, &cat);

    // Output cardinality: filter shrinks input by 1/10_000;
    // post-limit cap is min(0.1, 50) = 0.1.
    assert!(
        (costed.output_card().rows() - 0.1).abs() < 1e-9,
        "expected post-limit cardinality 0.1, got {}",
        costed.output_card().rows()
    );

    // Total cost rough-shape — dominated by the Scan term.
    let total = costed.total_cost().total();
    assert!(
        (1_000.0..2_000.0).contains(&total),
        "scan-dominated cost should be in [1000, 2000), got {total}"
    );

    // Walk the cost tree: structural shape should be 4 nested unary
    // wrappers (limit→project→filter→scan), so the root has 1
    // child, that has 1 child, etc.
    let mut node = costed.costs();
    let mut depth = 0;
    while !node.children.is_empty() {
        depth += 1;
        node = &node.children[0];
    }
    assert_eq!(
        depth, 3,
        "expected 4-deep linear cost tree (3 unary wrappers above the scan leaf)"
    );
}

#[test]
fn cost_model_end_to_end_hybrid_rank_by_vector_plus_text() {
    // Pipeline:
    //   Limit(20)
    //   ↓
    //   Project(...)
    //   ↓
    //   RankByHybrid(VECTOR(K=10), TEXT(K=20))   — leaf node
    //
    // Per-operand costs:
    //   Vector: 10 * VECTOR_NEAR_COST_PER_K = 10 * 30 = 300
    //   Text:   20 * TEXT_MATCH_COST_PER_K = 20 * 20 = 400
    //   Fusion: max(K) * n_operands * FUSION_COST_PER_ROW
    //         = 20 * 2 * 0.2 = 8
    //   Total leaf = 708
    //
    // Output cardinality from RankByHybrid: max(K) = 20.
    // Project preserves cardinality; Limit caps at min(20, 20) = 20.
    let cat = StubCatalogProvider::new()
        .with_total_node_count(1_000_000)
        .with_total_rel_count(5_000_000)
        .with_vector_index()
        .with_bm25_index();

    let rank = LogicalPlan::RankByHybrid(LogicalRankByHybrid {
        operands: vec![
            HybridOperand {
                kind: HybridOperandKind::Vector,
                var: BindingId::new(0),
                property: "embedding".into(),
                query: BoundExpression::Parameter {
                    name: "q".into(),
                    span: span(),
                    type_info: None,
                },
                k: 10,
                read_lsn: Lsn::MAX,
                span: span(),
            },
            HybridOperand {
                kind: HybridOperandKind::Text,
                var: BindingId::new(0),
                property: "content".into(),
                query: BoundExpression::Parameter {
                    name: "q".into(),
                    span: span(),
                    type_info: None,
                },
                k: 20,
                read_lsn: Lsn::MAX,
                span: span(),
            },
        ],
        score_binding: None,
        fusion: None,
        span: span(),
    });
    let project = LogicalPlan::Project(LogicalProject {
        input: Box::new(rank),
        items: Vec::new(),
        span: span(),
    });
    let limit = LogicalPlan::Limit(LogicalLimit {
        input: Box::new(project),
        count: 20,
        span: span(),
    });

    let costed = estimate_costs(limit, &cat);

    // Output: 20 (K capped).
    assert_eq!(costed.output_card().rows(), 20.0);

    // Total cost includes the hybrid leaf (~708) plus project (20*0.05=1) plus limit (~0.2).
    let total = costed.total_cost().total();
    assert!(
        (700.0..720.0).contains(&total),
        "hybrid-dominated cost should be near 708, got {total}"
    );
}

#[test]
fn cost_model_handles_empty_catalog_with_default_selectivity() {
    // Cold-start: no stats collected. The cost model must produce a
    // finite, non-negative cost without panicking — DEFAULT_*_SELECTIVITY
    // fallbacks per M4-42 contract.
    let cat = StubCatalogProvider::new();
    let scan = LogicalPlan::Scan(LogicalScan {
        label: Some(LabelId::new(7)),
        var: BindingId::new(0),
        read_lsn: Lsn::MAX,
        span: span(),
    });
    let costed = estimate_costs(scan, &cat);
    let total = costed.total_cost().total();
    assert!(
        total.is_finite(),
        "cost should be finite under cold-start, got {total}"
    );
    assert!(
        total >= 0.0,
        "cost should be non-negative under cold-start, got {total}"
    );
    // Output cardinality should also be finite + non-negative.
    let card = costed.output_card().rows();
    assert!(card.is_finite() && card >= 0.0);
}

#[test]
fn cost_model_consumes_snapshot_consistently_across_plan_walk() {
    // The walker calls snapshot() ONCE; readings throughout the walk
    // see the same cross-key-consistent view. This integration test
    // pins the contract via a plan with two scans on different
    // labels in a join — both read from the same snapshot.
    let person = LabelId::new(1);
    let doc = LabelId::new(2);
    let _knows = TypeId::new(1);
    let cat = StubCatalogProvider::new()
        .with_total_node_count(100_000)
        .with_total_rel_count(500_000)
        .with_label_cardinality(person, 30_000)
        .with_label_cardinality(doc, 70_000);

    use arcgraph_query::logical_plan::{JoinAlgorithm, JoinCondition, LogicalJoin};
    let p_scan = LogicalPlan::Scan(LogicalScan {
        label: Some(person),
        var: BindingId::new(0),
        read_lsn: Lsn::MAX,
        span: span(),
    });
    let d_scan = LogicalPlan::Scan(LogicalScan {
        label: Some(doc),
        var: BindingId::new(1),
        read_lsn: Lsn::MAX,
        span: span(),
    });
    let join = LogicalPlan::Join(LogicalJoin {
        left: Box::new(p_scan),
        right: Box::new(d_scan),
        on: JoinCondition::SharedBindings(vec![BindingId::new(2)]),
        algorithm: JoinAlgorithm::Auto,
        span: span(),
    });
    let costed = estimate_costs(join, &cat);

    // Both scans see consistent label-cards from the same snapshot.
    // Join output: (30k * 70k) / max = 70k × 30k / 70k = 30_000.
    assert!(
        (costed.output_card().rows() - 30_000.0).abs() < 1e-3,
        "join cardinality consistency: expected 30_000, got {}",
        costed.output_card().rows()
    );
}

#[test]
#[ignore = "perf bench — runs only with --ignored; prints wall-clock to stdout for review-packet"]
fn cost_model_meets_5ms_plan_build_budget() {
    // ADR-036 §D-25 pins the M4-05 plan parse + cost row at 5 ms.
    // M4-51's contribution is the cost walk; this test exercises a
    // representative LDBC-SNB-class plan and asserts wall-clock
    // budget. Numbers also captured in the review packet.
    use arcgraph_query::logical_plan::{
        Direction, JoinAlgorithm, JoinCondition, LogicalExpand, LogicalJoin,
    };
    let person = LabelId::new(1);
    let doc = LabelId::new(2);
    let knows = TypeId::new(1);
    let cat = StubCatalogProvider::new()
        .with_total_node_count(1_000_000)
        .with_total_rel_count(5_000_000)
        .with_label_cardinality(person, 300_000)
        .with_label_cardinality(doc, 700_000)
        .with_rel_type_cardinality(knows, 2_000_000);

    // LDBC SNB-shaped query:
    //   MATCH (p:Person)-[:KNOWS]->(p2:Person)
    //   WHERE p.age > $threshold
    //   RETURN p2.name ORDER BY p.age LIMIT 50
    let p_scan = LogicalPlan::Scan(LogicalScan {
        label: Some(person),
        var: BindingId::new(0),
        read_lsn: Lsn::MAX,
        span: span(),
    });
    let p2_scan = LogicalPlan::Scan(LogicalScan {
        label: Some(person),
        var: BindingId::new(2),
        read_lsn: Lsn::MAX,
        span: span(),
    });
    let expand = LogicalPlan::Expand(LogicalExpand {
        from: BindingId::new(0),
        to: BindingId::new(2),
        direction: Direction::LeftToRight,
        rel_type: Some(knows),
        length_range: None,
        rel_var: None,
        span: span(),
    });
    let join1 = LogicalPlan::Join(LogicalJoin {
        left: Box::new(p_scan),
        right: Box::new(expand),
        on: JoinCondition::SharedBindings(vec![BindingId::new(0)]),
        algorithm: JoinAlgorithm::Auto,
        span: span(),
    });
    let join2 = LogicalPlan::Join(LogicalJoin {
        left: Box::new(join1),
        right: Box::new(p2_scan),
        on: JoinCondition::SharedBindings(vec![BindingId::new(2)]),
        algorithm: JoinAlgorithm::Auto,
        span: span(),
    });
    let predicate = BoundExpression::BinaryOp {
        op: BinOp::Gt,
        lhs: Box::new(var_ref(0)),
        rhs: Box::new(BoundExpression::Parameter {
            name: "threshold".into(),
            span: span(),
            type_info: None,
        }),
        span: span(),
        type_info: None,
    };
    let filter = LogicalPlan::Filter(LogicalFilter {
        input: Box::new(join2),
        predicate,
        span: span(),
    });
    use arcgraph_query::logical_plan::{LogicalSort, OrderByItem, SortDirection};
    let sort = LogicalPlan::Sort(LogicalSort {
        input: Box::new(filter),
        order_by: vec![OrderByItem {
            expr: var_ref(0),
            direction: SortDirection::Asc,
            span: span(),
        }],
        span: span(),
    });
    let limit = LogicalPlan::Limit(LogicalLimit {
        input: Box::new(sort),
        count: 50,
        span: span(),
    });

    // Warm-up the snapshot path (one untimed call).
    let _ = arcgraph_query::planner::cost::estimate_costs(limit.clone(), &cat);

    let iters = 1_000usize;
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let _ = arcgraph_query::planner::cost::estimate_costs(limit.clone(), &cat);
    }
    let elapsed = start.elapsed();
    let avg_us = (elapsed.as_micros() as f64) / iters as f64;
    eprintln!(
        "cost_model_meets_5ms_plan_build_budget: {} iters in {:?} → avg = {:.3} µs",
        iters, elapsed, avg_us
    );
    // Each individual call MUST be well under 5 ms (5_000 µs).
    assert!(
        avg_us < 5_000.0,
        "M4-05 plan-build budget exceeded: avg = {avg_us} µs > 5_000 µs"
    );
}

#[test]
fn cost_model_empty_logical_plan_has_zero_cost() {
    // Defensive sentinel: the M4-31 LogicalEmpty case (degenerate
    // empty-clauses query) must produce zero cost without panicking.
    let cat = StubCatalogProvider::new();
    let plan = LogicalPlan::Empty(LogicalEmpty { span: span() });
    let costed = estimate_costs(plan, &cat);
    assert_eq!(costed.total_cost().total(), 0.0);
    assert_eq!(costed.output_card().rows(), 0.0);
}
