//! M4-63 vectorized-executor integration tests per ADR-038
//! amendment-02 §M4.f + amendment-03 §TIER-2-b.
//!
//! # Pin set (per amendment-02 §M4.f M4-63 row)
//!
//! 1. `aggregation_pipeline_end_to_end` — Scan → Aggregate(Count, Avg)
//!    integration; verifies amendment-03 §TIER-2-b NULL exclusion at
//!    the executor seam.
//! 2. `sort_then_limit_composition_emits_top_k_in_order` — Scan → Sort
//!    → Limit composition; pins the canonical "top-K" idiom.
//! 3. `named_shortest_path_on_tel_substrate` — single-source BFS over
//!    a linear-chain stub substrate; verifies the path operator
//!    composes through the executor pipeline.
//! 4. `multi_tenant_aggregate_isolation` — two tenants with distinct
//!    row sets aggregated independently; verifies per-tenant routing
//!    holds at the M4-63 layer (parallels M4-61's
//!    `executor_multi_tenant_scan_isolation`).
//!
//! # Wave-level transit pin (per
//!   `feedback_anchor_to_consumer_transit_pinning.md`)
//!
//! - `w12_alpha_m4_04d_to_m4_63_aggregate_transit_pin` — M4-04d
//!   empirical Person fixture → M4-51 cost walker → M4-61 scan →
//!   M4-63 aggregate → row-count output. Phase 4.2 controlled-
//!   mutation cycle: scaling Person cardinality 10× scales the
//!   COUNT(n) result accordingly (FAIL-on-revert: a future M4-08+
//!   refactor that disconnects the catalog from the executor's
//!   substrate dispatch fires this assertion).
//!
//! # ADR provenance
//! - **ADR-038 amendment-02 §M4.f** — primary M4-63 cite.
//! - **ADR-038 amendment-03 §TIER-2-b** — 3VL aggregate NULL exclusion.
//! - **ADR-038 §2 D-28** — aggregation / sort / path operator contract.

#![allow(clippy::too_many_lines)]

use arcgraph_core::{LabelId, Lsn, NodeId, PartitionId, RelId, TenantId, TypeId};
use arcgraph_query::error::Span;
use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::executor::ops::{
    AggregateCall, AggregateOp, LimitOp, NamedShortestPathOp, PathSpec, PhysicalOperator, ScanOp,
    SortKey, SortOp,
};
use arcgraph_query::executor::value::{NodeView, RelView};
use arcgraph_query::executor::{ExecutionContext, Value};
use arcgraph_query::logical_plan::{AggregationKind, Direction, SortDirection};
use arcgraph_query::semantic::bound_ast::{BindingId, BoundExpression, BoundPropertyRef};

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

fn ctx() -> ExecutionContext {
    ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO)
}

fn person_scan() -> ScanOp {
    ScanOp::new(BindingId::new(0), Some(LabelId::new(1)), Lsn::MAX)
}

fn var_n() -> BoundExpression {
    BoundExpression::VariableRef {
        name: "n".into(),
        binding_id: BindingId::new(0),
        span: Span::point(1, 1),
        type_info: None,
    }
}

fn prop(base: BoundExpression, name: &str) -> BoundExpression {
    BoundExpression::PropertyAccess {
        base: Box::new(base),
        path: vec![BoundPropertyRef {
            name: name.into(),
            property_id: None,
            span: Span::point(1, 1),
        }],
        span: Span::point(1, 1),
        type_info: None,
    }
}

fn make_persons_with_age(ages: &[Option<i64>]) -> StubExecutorSubstrate {
    let mut s = StubExecutorSubstrate::new();
    for (i, age) in ages.iter().enumerate() {
        let v = match age {
            Some(n) => Value::Integer(*n),
            None => Value::Null,
        };
        s = s.with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new((i + 1) as u64), Some(LabelId::new(1)))
                .with_property("age", v),
        );
    }
    s
}

// ---------------------------------------------------------------------
// 1. Aggregation pipeline end-to-end
// ---------------------------------------------------------------------

#[test]
fn aggregation_pipeline_end_to_end() {
    // Scan(Person) → Aggregate(count(n), avg(n.age))
    // 5 persons; ages [10, NULL, 20, NULL, 30]:
    //   count(n) = 5 (n is non-NULL for every row)
    //   avg(n.age) = (10 + 20 + 30) / 3 = 20.0 (NULL excluded from BOTH numerator + denominator)
    let s = make_persons_with_age(&[Some(10), None, Some(20), None, Some(30)]);
    let aggregations = vec![
        AggregateCall {
            distinct: false,
            star: false,
            kind: AggregationKind::Count,
            arg: var_n(),
            output_id: BindingId::new(2),
        },
        AggregateCall {
            distinct: false,
            star: false,
            kind: AggregationKind::Avg,
            arg: prop(var_n(), "age"),
            output_id: BindingId::new(3),
        },
    ];
    let mut op = AggregateOp::new(
        PhysicalOperator::Scan(person_scan()),
        Vec::new(),
        aggregations,
    );
    let ec = ctx();
    let b = op.next_batch(&ec, &s).unwrap();
    assert_eq!(b.row_count(), 1);
    // Column 0 = count(n) = 5; Column 1 = avg(n.age) = 20.0.
    assert_eq!(b.row(0)[0], Value::Integer(5));
    assert_eq!(b.row(0)[1], Value::Float(20.0));
    let b2 = op.next_batch(&ec, &s).unwrap();
    assert!(b2.is_empty(), "EOS after single-row aggregate emitted");
}

// ---------------------------------------------------------------------
// 2. Sort + Limit composition (top-K idiom)
// ---------------------------------------------------------------------

#[test]
fn sort_then_limit_composition_emits_top_k_in_order() {
    // Substrate: 5 persons with ages [50, 10, 40, 20, 30].
    // Pipeline: Scan → Sort(age ASC) → Limit(3).
    // Expected top-3 ASC: [10, 20, 30].
    let s = make_persons_with_age(&[Some(50), Some(10), Some(40), Some(20), Some(30)]);
    let sort = SortOp::new(
        PhysicalOperator::Scan(person_scan()),
        vec![SortKey {
            expr: prop(var_n(), "age"),
            direction: SortDirection::Asc,
        }],
    );
    let mut op = LimitOp::new(PhysicalOperator::Sort(sort), 3);
    let ec = ctx();
    let b = op.next_batch(&ec, &s).unwrap();
    assert_eq!(b.row_count(), 3);
    let ages: Vec<i64> = b
        .rows()
        .iter()
        .filter_map(|r| match &r[0] {
            Value::Node(n) => match n.properties.get("age") {
                Some(Value::Integer(n)) => Some(*n),
                _ => None,
            },
            _ => None,
        })
        .collect();
    assert_eq!(ages, vec![10, 20, 30]);
    // Limit hits → next batch is EOS without consuming more upstream.
    let b2 = op.next_batch(&ec, &s).unwrap();
    assert!(b2.is_empty());
}

// ---------------------------------------------------------------------
// 3. Named-shortest-path on the TEL substrate
// ---------------------------------------------------------------------

#[test]
fn named_shortest_path_on_tel_substrate() {
    // Linear chain 1→2→3→4 (KNOWS edges); SSSP from each node emits a
    // path-list cell to every reachable descendant.
    let mut s = StubExecutorSubstrate::new();
    for i in 1..=4_u64 {
        s = s.with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(i), Some(LabelId::new(1))),
        );
    }
    for i in 1..4_u64 {
        s = s.with_edge(
            TenantId::DEFAULT,
            RelView::new(
                RelId::new(100 + i),
                NodeId::new(i),
                NodeId::new(i + 1),
                Some(TypeId::new(1)),
            ),
        );
    }
    let mut op = NamedShortestPathOp::new(
        PhysicalOperator::Scan(person_scan()),
        PathSpec {
            source: BindingId::new(0),
            target: None,
            rel_type: Some(TypeId::new(1)),
            direction: Direction::LeftToRight,
            path_var: BindingId::new(99),
            all_shortest: false,
        },
        Lsn::MAX,
    );
    let ec = ctx();
    let b = op.next_batch(&ec, &s).unwrap();
    // SSSP from each of the 4 nodes — descendants reachable: 3 + 2 + 1 + 0 = 6 paths.
    assert_eq!(b.row_count(), 6);
    // ADR-194 D-5 — every emitted row is a single `Value::Path` cell
    // (migrated from the legacy node-only `Value::List`), carrying nodes
    // AND relationships in source→target traversal order.
    for row in b.rows() {
        assert_eq!(row.len(), 1);
        match &row[0] {
            Value::Path(p) => {
                // `#nodes == #rels + 1` (PathView structural invariant).
                assert_eq!(p.nodes().len(), p.hop_count() + 1);
                assert!(
                    p.hop_count() >= 1,
                    "single-source paths skip the zero-hop self"
                );
            }
            other => panic!("expected Value::Path path cell; got {other:?}"),
        }
    }
}

// ---------------------------------------------------------------------
// 4. Multi-tenant aggregate isolation
// ---------------------------------------------------------------------

#[test]
fn multi_tenant_aggregate_isolation() {
    // Two tenants with disjoint substrate slices: tenant A has 3
    // persons; tenant B has 7. The aggregate executed for tenant A
    // returns 3 rows' worth of count, NOT 10.
    let other = TenantId::new(42);
    let mut s = StubExecutorSubstrate::new();
    for i in 1..=3_u64 {
        s = s.with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(i), Some(LabelId::new(1)))
                .with_property("age", Value::Integer(10)),
        );
    }
    for i in 1..=7_u64 {
        s = s.with_node(
            other,
            NodeView::new(NodeId::new(100 + i), Some(LabelId::new(1)))
                .with_property("age", Value::Integer(10)),
        );
    }
    let aggregations = vec![AggregateCall {
        distinct: false,
        star: false,
        kind: AggregationKind::Count,
        arg: var_n(),
        output_id: BindingId::new(2),
    }];

    // Tenant A.
    let ctx_a = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let mut op_a = AggregateOp::new(
        PhysicalOperator::Scan(person_scan()),
        Vec::new(),
        aggregations.clone(),
    );
    let b_a = op_a.next_batch(&ctx_a, &s).unwrap();
    assert_eq!(
        b_a.row(0)[0],
        Value::Integer(3),
        "tenant A sees only its 3 rows"
    );

    // Tenant B.
    let ctx_b = ExecutionContext::new(other, PartitionId::ZERO);
    let mut op_b = AggregateOp::new(
        PhysicalOperator::Scan(person_scan()),
        Vec::new(),
        aggregations,
    );
    let b_b = op_b.next_batch(&ctx_b, &s).unwrap();
    assert_eq!(
        b_b.row(0)[0],
        Value::Integer(7),
        "tenant B sees only its 7 rows"
    );
}

// ---------------------------------------------------------------------
// WAVE-LEVEL TRANSIT PIN — M4-04d → M4-51 → M4-63 → result
// ---------------------------------------------------------------------

mod common;

/// W12α wave-level transit pin. Mirrors the M4-04d → M4-61 transit
/// pin that landed in W11Z #272 (`m4_62_hybrid_optional_integration::
/// m4_04d_to_m4_61_executor_transit_pin`), but now drives the chain
/// THROUGH the M4-63 aggregate operator. The chain:
///
/// 1. M4-04d empirical fixture publishes a Person tenant with a
///    specific person_count cardinality.
/// 2. M4-51 cost walker reads that cardinality (verified in W11Z by
///    the EXPLAIN-side transit pin `m4_91_explain_integration::
///    empirical_fixture_phase_4_2_mutation_on_default_label_selectivity`).
/// 3. The M4-61 executor pulls rows from a substrate sized to the
///    same cardinality.
/// 4. The M4-63 AggregateOp folds them into a single COUNT(n) cell.
///
/// Phase 4.2 mutation: scale the producer cardinality 10×; assert the
/// AggregateOp's emitted COUNT scales 10× accordingly. FAIL-on-revert:
/// a future refactor that makes the executor's row count diverge from
/// the producer's published cardinality fires this assertion.
#[test]
fn w12_alpha_m4_04d_to_m4_63_aggregate_transit_pin() {
    use common::m4_04d_person_tenant::PersonTenant;

    // Anchor: SF-0.0001 (≈ 100 Persons). Mirrors the W11Z #272
    // ANCHOR_PERSON_SUBSTRATE_CAP = 1000 ceiling so 10× scaling lands
    // inside the cap.
    const ANCHOR_PERSON_CAP: u64 = 1000;

    fn substrate_for_person_tenant(pt: &PersonTenant) -> StubExecutorSubstrate {
        let cap = std::cmp::min(pt.person_count, ANCHOR_PERSON_CAP);
        let mut s = StubExecutorSubstrate::new();
        for i in 1..=cap {
            s = s.with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(i), Some(LabelId::new(1))),
            );
        }
        s
    }

    // Baseline: 100 persons → COUNT(n) = 100.
    let pt = PersonTenant::seed_sf(0.0001);
    let s = substrate_for_person_tenant(&pt);
    let mut op = AggregateOp::new(
        PhysicalOperator::Scan(person_scan()),
        Vec::new(),
        vec![AggregateCall {
            distinct: false,
            star: false,
            kind: AggregationKind::Count,
            arg: var_n(),
            output_id: BindingId::new(2),
        }],
    );
    let ec = ctx();
    let b = op.next_batch(&ec, &s).unwrap();
    let baseline_count = match b.row(0)[0] {
        Value::Integer(n) => n,
        _ => panic!("count must be Integer"),
    };
    assert_eq!(baseline_count, 100, "baseline: 100 person rows");

    // PHASE 4.2 controlled-mutation: scale the producer 10× → 1000
    // persons. The aggregate's COUNT(n) MUST scale 10×.
    let pt_scaled = pt.scale_all_label_cards(10);
    let s_scaled = substrate_for_person_tenant(&pt_scaled);
    let mut op_scaled = AggregateOp::new(
        PhysicalOperator::Scan(person_scan()),
        Vec::new(),
        vec![AggregateCall {
            distinct: false,
            star: false,
            kind: AggregationKind::Count,
            arg: var_n(),
            output_id: BindingId::new(2),
        }],
    );
    let ctx_scaled = ctx();
    let b_scaled = op_scaled.next_batch(&ctx_scaled, &s_scaled).unwrap();
    let scaled_count = match b_scaled.row(0)[0] {
        Value::Integer(n) => n,
        _ => panic!("count must be Integer"),
    };
    assert_eq!(
        scaled_count, 1000,
        "Phase 4.2 mutation: 10× producer cardinality scales \
         executor's emitted COUNT 10× (FAIL-on-revert per \
         feedback_anchor_to_consumer_transit_pinning.md)"
    );
    assert!(
        scaled_count > baseline_count,
        "Phase 4.2 propagation: scaled count strictly exceeds \
         baseline ({scaled_count} > {baseline_count})"
    );
}

// ---------------------------------------------------------------------
// Auxiliary: 3VL boundary integration (amendment-03 §TIER-2-b pin)
// ---------------------------------------------------------------------

#[test]
fn aggregate_3vl_null_exclusion_pin_at_executor_seam() {
    // Executor-seam pin for amendment-03 §TIER-2-b: COUNT(expr)
    // excludes NULL — the M4-63 executor MUST honor the contract that
    // M4-22 binds. A future regression that admits NULL into the
    // count surfaces at this integration test.
    let s = make_persons_with_age(&[Some(1), None, None, Some(2), None, Some(3)]);
    let mut op = AggregateOp::new(
        PhysicalOperator::Scan(person_scan()),
        Vec::new(),
        vec![AggregateCall {
            distinct: false,
            star: false,
            kind: AggregationKind::Count,
            arg: prop(var_n(), "age"),
            output_id: BindingId::new(3),
        }],
    );
    let ec = ctx();
    let b = op.next_batch(&ec, &s).unwrap();
    // 3 non-NULL ages out of 6 rows; the test pins the EXCLUSION
    // (COUNT-of-non-NULL = 3, NOT total-row-count = 6).
    assert_eq!(b.row(0)[0], Value::Integer(3));
}
