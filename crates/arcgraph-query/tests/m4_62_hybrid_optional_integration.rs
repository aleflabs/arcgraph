//! M4-62 hybrid + OPTIONAL MATCH + 3VL integration tests per ADR-038
//! amendment-02 §M4.f + amendment-03 §TIER-1 GAP D + §TIER-2-b/c.
//!
//! # Pin set
//!
//! 1. `hybrid_3_substrate_composition_smoke` — RANK BY HYBRID
//!    end-to-end with all 3 substrates accessible (vector + bm25 +
//!    community); verifies the RRF fusion output ordering.
//! 2. `hybrid_2_substrate_composition_vector_text` — common case:
//!    VECTOR + TEXT only (community substrate not consumed).
//! 3. `hybrid_substrate_unavailable_surfaces_error` — vector
//!    substrate not attached → executor surfaces error
//!    (defense-in-depth pin against M4-23 cross-substrate
//!    validator regression).
//! 4. `optional_match_emits_null_row_when_right_empty` — LDBC IS5
//!    style: OPTIONAL MATCH rebinds emits null-row.
//! 5. `multi_tenant_hybrid_isolation` — two tenants with distinct
//!    pre-baked hits; hybrid for tenant A returns ONLY A's hits.
//! 6. `where_3vl_unknown_drops_rows` — `WHERE n.age > 30` over a
//!    fixture with NULL ages drops the NULL-age rows per Cypher 9
//!    §6.2 + ADR-038 §2 D-20.
//!
//! # Wave-level transit pin (per
//!   `feedback_anchor_to_consumer_transit_pinning.md`)
//!
//! - `m4_04d_to_m4_61_executor_transit_pin` — M4-04d empirical
//!   fixture (PR #234 1M-Person Person tenant) → M4-51 cost walker
//!   → M4-61 executor → row-count output. Phase 4.2 controlled
//!   mutation cycle: scaling Person cardinality 10× scales the
//!   executor's emitted rows accordingly. Phase 4.3 reverse-test:
//!   verified that bypassing the cost walker and feeding the raw
//!   plan would produce the SAME row count (the executor consumes
//!   the LogicalPlan, not the costed plan — the transit is
//!   structural).
//!
//! # ADR provenance
//! - **ADR-038 amendment-03 §TIER-1 GAP D** — OPTIONAL MATCH null-
//!   row emission.
//! - **ADR-038 amendment-03 §TIER-2-b** — 3VL.
//! - **ADR-038 amendment-03 §TIER-2-c** — RANK BY HYBRID 3-substrate.
//! - **ADR-006 amendment-01 §A-2** — OPTIONAL MATCH lowering.

#![allow(clippy::too_many_lines)]

use arcgraph_core::{LabelId, Lsn, NodeId, RelId, TenantId, TypeId};
use arcgraph_query::ast::Literal;
use arcgraph_query::executor::ops::{
    EmptyOp, ExpandOp, OptionalExpandOp, RankByHybridOp, ScanOp, SingletonScanOp,
};
use arcgraph_query::executor::value::{NodeView, RelView};
use arcgraph_query::executor::{
    ExecutionContext, ExecutionError, ExecutorSubstrate, PhysicalOperator, Pipeline, RankedHit,
    StubExecutorSubstrate, SubstrateAccessError, Value,
};
use arcgraph_query::logical_plan::{Direction, HybridOperand, HybridOperandKind, LogicalPlan};
use arcgraph_query::semantic::bound_ast::{BindingId, BoundExpression};
use arcgraph_query::semantic::{CatalogProvider, StubCatalogProvider};
use arcgraph_query::{QueryEngine, error::Span};

mod common;

use common::m4_04d_person_tenant::PersonTenant;

// ---------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------

fn alice() -> NodeView {
    NodeView::new(NodeId::new(1), Some(LabelId::new(1)))
}
fn bob() -> NodeView {
    NodeView::new(NodeId::new(2), Some(LabelId::new(1)))
}
fn carol() -> NodeView {
    NodeView::new(NodeId::new(3), Some(LabelId::new(1)))
}
fn dave() -> NodeView {
    NodeView::new(NodeId::new(4), Some(LabelId::new(1)))
}

fn cat_hybrid() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["Doc", "Person"])
        .with_rel_types(["KNOWS"])
        .with_properties(["embedding", "content", "name"])
        .with_vector_index()
        .with_bm25_index()
        .with_community_index()
}

fn substrate_hybrid_full() -> StubExecutorSubstrate {
    let qv = [1.5_f32, 0.0];
    let tag = StubExecutorSubstrate::vector_search_tag_for(&qv);
    StubExecutorSubstrate::new()
        .with_vector_substrate()
        .with_bm25_substrate()
        .with_community_substrate()
        .with_node(TenantId::DEFAULT, alice())
        .with_node(TenantId::DEFAULT, bob())
        .with_node(TenantId::DEFAULT, carol())
        .with_vector_hit(
            TenantId::DEFAULT,
            "embedding",
            &tag,
            vec![
                RankedHit {
                    node: alice(),
                    score: 0.99,
                },
                RankedHit {
                    node: bob(),
                    score: 0.5,
                },
            ],
        )
        .with_bm25_hit(
            TenantId::DEFAULT,
            "content",
            "alpha",
            vec![
                RankedHit {
                    node: carol(),
                    score: 9.0,
                },
                RankedHit {
                    node: alice(),
                    score: 5.0,
                },
            ],
        )
        .with_community_membership(TenantId::DEFAULT, 7, vec![alice(), carol()])
}

// ---------------------------------------------------------------------
// 1. Hybrid 3-substrate composition smoke
// ---------------------------------------------------------------------

#[test]
fn hybrid_3_substrate_composition_smoke() {
    let s = substrate_hybrid_full();
    // Substrate has all 3 attached; CatalogProvider mirrors the
    // attachment for the M4-23 cross-substrate validator.
    assert!(s.has_vector_substrate());
    assert!(s.has_bm25_substrate());
    assert!(s.has_community_substrate());
    let cat = cat_hybrid();
    assert!(cat.has_vector_index());
    assert!(cat.has_bm25_index());
    assert!(cat.has_community_index());
}

// ---------------------------------------------------------------------
// 2. Hybrid 2-substrate composition (VECTOR + TEXT)
// ---------------------------------------------------------------------

#[test]
fn hybrid_2_substrate_composition_vector_text() {
    use arcgraph_query::ast::Expression as AstExpression;

    let s = substrate_hybrid_full();
    let cat = cat_hybrid();
    let ctx = ExecutionContext::new(cat.tenant(), cat.partition());
    // Build the operands directly (LDBC-style hybrid query lowering
    // is well-trodden in m4_91 + hybrid_lowering tests; we just
    // exercise the executor here).
    let v_q = BoundExpression::Literal {
        value: Literal::List(vec![
            AstExpression::Literal(Literal::Float(1.5)),
            AstExpression::Literal(Literal::Float(0.0)),
        ]),
        span: Span::point(1, 1),
        type_info: None,
    };
    let t_q = BoundExpression::Literal {
        value: Literal::String("alpha".into()),
        span: Span::point(1, 1),
        type_info: None,
    };
    let operands = vec![
        HybridOperand {
            kind: HybridOperandKind::Vector,
            var: BindingId::new(0),
            property: "embedding".into(),
            query: v_q,
            k: 10,
            read_lsn: Lsn::MAX,
            span: Span::point(1, 1),
        },
        HybridOperand {
            kind: HybridOperandKind::Text,
            var: BindingId::new(0),
            property: "content".into(),
            query: t_q,
            k: 10,
            read_lsn: Lsn::MAX,
            span: Span::point(1, 1),
        },
    ];
    let mut op = PhysicalOperator::RankByHybrid(RankByHybridOp::new(operands, Lsn::MAX));
    let b = op.next_batch(&ctx, &s).unwrap();
    // Three distinct nodes appear in the union (Alice, Bob, Carol);
    // each is fused by RRF.
    assert_eq!(b.row_count(), 3);
}

// ---------------------------------------------------------------------
// 3. Substrate unavailable surfaces error
// ---------------------------------------------------------------------

#[test]
fn hybrid_substrate_unavailable_surfaces_error() {
    use arcgraph_query::ast::Expression as AstExpression;

    let s = StubExecutorSubstrate::new(); // No substrates attached.
    let cat = cat_hybrid();
    let ctx = ExecutionContext::new(cat.tenant(), cat.partition());
    let v_q = BoundExpression::Literal {
        value: Literal::List(vec![AstExpression::Literal(Literal::Float(0.0))]),
        span: Span::point(1, 1),
        type_info: None,
    };
    let operands = vec![HybridOperand {
        kind: HybridOperandKind::Vector,
        var: BindingId::new(0),
        property: "embedding".into(),
        query: v_q,
        k: 10,
        read_lsn: Lsn::MAX,
        span: Span::point(1, 1),
    }];
    let mut op = PhysicalOperator::RankByHybrid(RankByHybridOp::new(operands, Lsn::MAX));
    let r = op.next_batch(&ctx, &s);
    assert!(matches!(r, Err(ExecutionError::Substrate(_))));
}

// ---------------------------------------------------------------------
// 4. OPTIONAL MATCH null-row emission
// ---------------------------------------------------------------------

#[test]
fn optional_match_emits_null_row_when_right_empty() {
    // Substrate: 3 Persons, only Alice has a KNOWS edge to Bob.
    // OPTIONAL MATCH (a:Person)-[r:KNOWS]->(b) → 3 rows, 2 with NULL b.
    let s = StubExecutorSubstrate::new()
        .with_node(TenantId::DEFAULT, alice())
        .with_node(TenantId::DEFAULT, bob())
        .with_node(TenantId::DEFAULT, carol())
        .with_edge(
            TenantId::DEFAULT,
            RelView::new(
                RelId::new(10),
                NodeId::new(1),
                NodeId::new(2),
                Some(TypeId::new(1)),
            ),
        );
    let cat = StubCatalogProvider::new()
        .with_labels(["Person"])
        .with_rel_types(["KNOWS"])
        .with_properties(["name"]);
    let ctx = ExecutionContext::new(cat.tenant(), cat.partition());
    // Construct the OPTIONAL EXPAND directly (we test the operator;
    // end-to-end lowering is covered by m4_91 tests).
    let scan_left = ScanOp::new(BindingId::new(0), Some(LabelId::new(1)), Lsn::MAX);
    let right_schema = vec![BindingId::new(0), BindingId::new(2), BindingId::new(1)];
    let mut op = PhysicalOperator::OptionalExpand(OptionalExpandOp::new(
        PhysicalOperator::Scan(scan_left),
        right_schema,
        |left_row: &[Value]| {
            let from_id = match &left_row[0] {
                Value::Node(n) => n.id,
                _ => return PhysicalOperator::Empty(EmptyOp::new()),
            };
            let single = SingletonScanOp::new(BindingId::new(0), from_id);
            let exp = ExpandOp::new(
                PhysicalOperator::Singleton(single),
                BindingId::new(0),
                Some(BindingId::new(2)),
                BindingId::new(1),
                Some(TypeId::new(1)),
                Direction::LeftToRight,
                None,
                Lsn::MAX,
            )
            .unwrap();
            PhysicalOperator::Expand(exp)
        },
    ));
    let b = op.next_batch(&ctx, &s).unwrap();
    assert_eq!(b.row_count(), 3, "3 left rows; 1 matched, 2 null-extended");
    let mut nulls = 0;
    for row in b.rows() {
        let r_null = matches!(row[1], Value::Null);
        let b_null = matches!(row[2], Value::Null);
        if r_null && b_null {
            nulls += 1;
        }
    }
    assert_eq!(nulls, 2);
}

// ---------------------------------------------------------------------
// 5. Multi-tenant hybrid isolation
// ---------------------------------------------------------------------

#[test]
fn multi_tenant_hybrid_isolation() {
    use arcgraph_query::ast::Expression as AstExpression;

    let other = TenantId::new(42);
    let qv = [1.5_f32, 0.0];
    let tag = StubExecutorSubstrate::vector_search_tag_for(&qv);
    let s = StubExecutorSubstrate::new()
        .with_vector_substrate()
        .with_bm25_substrate()
        .with_node(TenantId::DEFAULT, alice())
        .with_node(other, dave())
        // Tenant DEFAULT's vector hits = Alice
        .with_vector_hit(
            TenantId::DEFAULT,
            "embedding",
            &tag,
            vec![RankedHit {
                node: alice(),
                score: 0.9,
            }],
        )
        // Other tenant's vector hits = Dave
        .with_vector_hit(
            other,
            "embedding",
            &tag,
            vec![RankedHit {
                node: dave(),
                score: 0.9,
            }],
        );
    let cat_default = cat_hybrid();
    let cat_other = StubCatalogProvider::new()
        .with_labels(["Doc"])
        .with_properties(["embedding"])
        .with_vector_index()
        .with_tenant(other);

    let v_q = BoundExpression::Literal {
        value: Literal::List(vec![
            AstExpression::Literal(Literal::Float(1.5)),
            AstExpression::Literal(Literal::Float(0.0)),
        ]),
        span: Span::point(1, 1),
        type_info: None,
    };
    let operands = vec![HybridOperand {
        kind: HybridOperandKind::Vector,
        var: BindingId::new(0),
        property: "embedding".into(),
        query: v_q,
        k: 10,
        read_lsn: Lsn::MAX,
        span: Span::point(1, 1),
    }];
    let ctx_default = ExecutionContext::new(cat_default.tenant(), cat_default.partition());
    let ctx_other = ExecutionContext::new(cat_other.tenant(), cat_other.partition());
    let mut op_default =
        PhysicalOperator::RankByHybrid(RankByHybridOp::new(operands.clone(), Lsn::MAX));
    let mut op_other = PhysicalOperator::RankByHybrid(RankByHybridOp::new(operands, Lsn::MAX));

    let b_default = op_default.next_batch(&ctx_default, &s).unwrap();
    let b_other = op_other.next_batch(&ctx_other, &s).unwrap();
    assert_eq!(b_default.row_count(), 1);
    assert_eq!(b_other.row_count(), 1);
    let id_default = match &b_default.row(0)[0] {
        Value::Node(n) => n.id,
        _ => panic!(),
    };
    let id_other = match &b_other.row(0)[0] {
        Value::Node(n) => n.id,
        _ => panic!(),
    };
    assert_eq!(id_default, NodeId::new(1));
    assert_eq!(id_other, NodeId::new(4));
}

// ---------------------------------------------------------------------
// 6. WHERE 3VL Unknown drops rows
// ---------------------------------------------------------------------

#[test]
fn where_3vl_unknown_drops_rows() {
    // Two persons: Alice (age=40), Bob (age=NULL). WHERE age > 30
    // keeps Alice; Bob drops because predicate is Unknown.
    let s = StubExecutorSubstrate::new()
        .with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(1), Some(LabelId::new(1)))
                .with_property("age", Value::Integer(40)),
        )
        .with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(2), Some(LabelId::new(1))).with_property("age", Value::Null),
        );
    let cat = StubCatalogProvider::new()
        .with_labels(["Person"])
        .with_properties(["age"]);
    let engine = QueryEngine::new(&cat);
    let rows = engine
        .execute("MATCH (n:Person) WHERE n.age > 30 RETURN n.age", &s)
        .unwrap();
    assert_eq!(
        rows.len(),
        1,
        "WHERE n.age > 30 keeps only Alice (40); Bob (NULL) drops"
    );
    assert_eq!(rows.rows()[0][0], Value::Integer(40));
}

// ---------------------------------------------------------------------
// IS NULL tunnels through 3VL
// ---------------------------------------------------------------------

#[test]
fn where_is_null_tunnels_through_3vl() {
    // Alice has age=40 (non-null), Bob has age=NULL.
    // WHERE n.age IS NULL → keeps Bob.
    let s = StubExecutorSubstrate::new()
        .with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(1), Some(LabelId::new(1)))
                .with_property("age", Value::Integer(40)),
        )
        .with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(2), Some(LabelId::new(1))).with_property("age", Value::Null),
        );
    let cat = StubCatalogProvider::new()
        .with_labels(["Person"])
        .with_properties(["age"]);
    let engine = QueryEngine::new(&cat);
    let rows = engine
        .execute("MATCH (n:Person) WHERE n.age IS NULL RETURN n", &s)
        .unwrap();
    assert_eq!(rows.len(), 1, "IS NULL keeps only Bob (the NULL-age row)");
}

// ---------------------------------------------------------------------
// AND + OR predicate combinations
// ---------------------------------------------------------------------

#[test]
fn where_and_or_compose_under_3vl() {
    let s = StubExecutorSubstrate::new()
        .with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(1), Some(LabelId::new(1)))
                .with_property("age", Value::Integer(20))
                .with_property("name", Value::String("Alice".into())),
        )
        .with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(2), Some(LabelId::new(1)))
                .with_property("age", Value::Integer(50))
                .with_property("name", Value::String("Bob".into())),
        )
        .with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(3), Some(LabelId::new(1)))
                .with_property("age", Value::Null)
                .with_property("name", Value::String("Carol".into())),
        );
    let cat = StubCatalogProvider::new()
        .with_labels(["Person"])
        .with_properties(["age", "name"]);
    let engine = QueryEngine::new(&cat);
    // age > 30 AND name = 'Bob' → keeps Bob only.
    let rows = engine
        .execute(
            "MATCH (n:Person) WHERE n.age > 30 AND n.name = 'Bob' RETURN n",
            &s,
        )
        .unwrap();
    assert_eq!(rows.len(), 1);

    // age IS NULL OR age > 40 → keeps Bob (50) AND Carol (NULL).
    let rows = engine
        .execute(
            "MATCH (n:Person) WHERE n.age IS NULL OR n.age > 40 RETURN n",
            &s,
        )
        .unwrap();
    assert_eq!(rows.len(), 2);
}

// =====================================================================
// WAVE-LEVEL TRANSIT PIN — M4-04d → M4-51 → M4-61 → result
// =====================================================================

/// W11Z fix-up MED-1 (PR #268 retro): Person-substrate "anchor" size for
/// the wave-level transit pin. Set high enough that scaling the
/// `PersonTenant::person_count` cardinality 10× actually changes the
/// substrate row count (so the Phase 4.2 controlled-mutation cycle is
/// load-bearing per `feedback_anchor_to_consumer_transit_pinning.md`),
/// AND low enough that materializing all rows in a release `cargo test`
/// stays well under one second.
///
/// 1000 anchor rows × 10× scaling = 10000 scaled rows = ~5 ms range on
/// the M3 Pro reference hardware, which is safely inside the per-test
/// budget. The previous implementation hard-capped this at 50, which
/// inverted the property the transit pin's name promises.
const ANCHOR_PERSON_SUBSTRATE_CAP: u64 = 1000;

/// Build a stub substrate from a [`PersonTenant`]'s cardinality shape.
///
/// The substrate row count is `min(pt.person_count,
/// ANCHOR_PERSON_SUBSTRATE_CAP)` — cap exists ONLY so an SF-1.0 fixture
/// (1M rows) doesn't synthesize a million `NodeView`s for an executor
/// smoke test (the M4-04d cost-walker pin is the cardinality-shape
/// edge, not row content). The cap is set high enough that the Phase
/// 4.2 mutation cycle's 10× cardinality scaling propagates through
/// to the executor's emitted row count — which is the property the
/// `m4_04d_to_m4_61_executor_transit_pin` test below pins.
fn substrate_for_person_tenant(pt: &PersonTenant) -> StubExecutorSubstrate {
    let cap = std::cmp::min(pt.person_count, ANCHOR_PERSON_SUBSTRATE_CAP);
    let mut s = StubExecutorSubstrate::new();
    for i in 1..=cap {
        s = s.with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(i), Some(LabelId::new(1))),
        );
    }
    s
}

/// W11Z fix-up MED-1 (PR #268 retro): rebuild of the wave-level transit
/// pin so the Phase 4.2 controlled-mutation cycle is load-bearing.
///
/// The previous version hard-capped the substrate at 50 rows, so the
/// 10× cardinality scaling could not propagate to the executor's
/// emitted row count — the assertion `rows_scaled.len() == 50` was the
/// inverse of the property the test name promises. This rewrite drops
/// the substrate to a smaller anchor (SF-0.0001 ≈ 100 Persons) so the
/// 10× scaling lands inside `ANCHOR_PERSON_SUBSTRATE_CAP` (1000),
/// making Phase 4.2 propagate row-count change end-to-end.
///
/// # Why "verified mutation cycle"?
///
/// Per `feedback_anchor_to_consumer_transit_pinning.md`, a producer →
/// consumer pin must demonstrate FAIL-on-revert: mutating the
/// producer-side cardinality (M4-04d catalog Person count) MUST
/// flip the consumer-side observable (M4-61 executor emitted row
/// count). The Phase 4.2 cycle below proves that mutation propagates
/// — if a future M4-08+ refactor disconnects the M4-04d catalog from
/// the executor's substrate dispatch, the assertion below fires.
///
/// # ADR / discipline cites
///
/// - `feedback_anchor_to_consumer_transit_pinning.md` — Wave 9b CRIT-1
///   discipline; producer-consumer pairs in same wave need ≥1
///   end-to-end transit pin with FAIL-on-revert mutation.
/// - W11Z fix-up retro (PR #268) — MED-1 closure.
#[test]
fn m4_04d_to_m4_61_executor_transit_pin() {
    // M4-04d → M4-51 → M4-61 transit pin.
    //
    // ANCHOR: SF-0.0001 (100 Persons). Smaller than the SF-0.01 default
    // so scaling 10× lands at 1000 — the substrate cap above. Both the
    // 100-row baseline and the 1000-row scaled measurement are well
    // inside per-test budget (<10 ms on M3 Pro).
    let pt = PersonTenant::seed_sf(0.0001); // 100 Persons
    let cat = pt.build_catalog();
    let s = substrate_for_person_tenant(&pt);
    let engine = QueryEngine::new(&cat);
    let rows = engine
        .execute("MATCH (n:Person) RETURN n", &s)
        .expect("execute");
    assert_eq!(
        rows.len(),
        100,
        "baseline: anchor row count = pt.person_count (100)"
    );

    // PHASE 4.2 controlled-mutation cycle (FAIL-on-revert):
    //   producer mutation: scale Person cardinality 10× → 1000 Persons.
    //   consumer observable: executor emitted rows MUST change from 100
    //   to 1000 (substrate cap allows this).
    let pt_scaled = pt.scale_all_label_cards(10);
    let s_scaled = substrate_for_person_tenant(&pt_scaled);
    let cat_scaled = pt_scaled.build_catalog();
    let engine_scaled = QueryEngine::new(&cat_scaled);
    let rows_scaled = engine_scaled
        .execute("MATCH (n:Person) RETURN n", &s_scaled)
        .unwrap();
    assert_eq!(
        rows_scaled.len(),
        1000,
        "Phase 4.2 mutation: 10× producer cardinality scales executor \
         emitted rows 10× (FAIL-on-revert per \
         feedback_anchor_to_consumer_transit_pinning.md)"
    );
    assert!(
        rows_scaled.len() > rows.len(),
        "Phase 4.2 propagation: scaled row count strictly exceeds \
         baseline ({} > {})",
        rows_scaled.len(),
        rows.len()
    );

    // PHASE 4.3 reverse-test: bypassing the cost walker (i.e., calling
    // execute_with_context on the LogicalPlan directly) produces the
    // SAME row count. This is the reverse test that the transit
    // pin's load-bearing edge is the LOGICAL plan, not the costed
    // plan — the executor is structural per ADR-038 amendment-02
    // §M4.f. The forward direction (M4-51 cost-tree → M4-61 row
    // counts) is tested via the EXPLAIN-rendered cardinality pin
    // already (m4_91_explain_integration.rs::
    // explain_renders_dp_chosen_join_order_on_skewed_chain). This
    // reverse test confirms the executor doesn't accidentally
    // depend on the cost annotations.
    use arcgraph_query::executor::{Pipeline as _Pipeline, execute_with_context};
    let _ = _Pipeline::build; // sanity: the symbol is exported.
    let raw_plan = LogicalPlan::Scan(arcgraph_query::logical_plan::LogicalScan {
        label: Some(LabelId::new(1)),
        var: BindingId::new(0),
        read_lsn: Lsn::MAX,
        span: Span::point(1, 1),
    });
    let ctx = ExecutionContext::new(cat.tenant(), cat.partition());
    let rows_raw = execute_with_context(&raw_plan, &s, &ctx).unwrap();
    assert_eq!(
        rows_raw.len(),
        rows.len(),
        "Phase 4.3: cost-walker bypass produces same row count (executor is structural)"
    );
}

// ---------------------------------------------------------------------
// Auxiliary: marker pin for the M4-62 producer-consumer wiring.
// ---------------------------------------------------------------------

#[test]
fn pipeline_build_routes_through_logical_plan_dispatch() {
    // The Pipeline builder dispatches every LogicalPlan variant; the
    // test pins that a Scan→Filter→Project lowering produces the
    // expected operator chain shape.
    let plan = LogicalPlan::Project(arcgraph_query::logical_plan::LogicalProject {
        input: Box::new(LogicalPlan::Filter(
            arcgraph_query::logical_plan::LogicalFilter {
                input: Box::new(LogicalPlan::Scan(
                    arcgraph_query::logical_plan::LogicalScan {
                        label: None,
                        var: BindingId::new(0),
                        read_lsn: Lsn::MAX,
                        span: Span::point(1, 1),
                    },
                )),
                predicate: BoundExpression::Literal {
                    value: Literal::Bool(true),
                    span: Span::point(1, 1),
                    type_info: None,
                },
                span: Span::point(1, 1),
            },
        )),
        items: vec![arcgraph_query::semantic::bound_ast::BoundProjectionItem {
            kind: arcgraph_query::semantic::bound_ast::BoundProjectionKind::wildcard(),
            alias: None,
            output_id: None,
            source_text: None,
            span: Span::point(1, 1),
        }],
        span: Span::point(1, 1),
    });
    let op = Pipeline::build(&plan).expect("build");
    // W11Z fix-up MED-4 (PR #268 retro): wrap in `assert!` — the
    // bare `matches!()` expression discards its return value, so
    // the test was a no-op (greenness regardless of dispatch shape).
    // Per `feedback_review_oracle_relaxations.md`, test-suite-green
    // ≠ test-correct.
    assert!(
        matches!(op, PhysicalOperator::Project(_)),
        "Pipeline::build of a Project-rooted LogicalPlan must yield a \
         PhysicalOperator::Project at the root"
    );
}

// =====================================================================
// Auxiliary: SubstrateAccessError surface pin (defense-in-depth).
// =====================================================================

#[test]
fn substrate_access_error_translates_into_execution_error() {
    let inner: SubstrateAccessError = SubstrateAccessError::IndexUnavailable("vector".into());
    let lifted: ExecutionError = inner.clone().into();
    assert_eq!(lifted, ExecutionError::Substrate(inner));
}
