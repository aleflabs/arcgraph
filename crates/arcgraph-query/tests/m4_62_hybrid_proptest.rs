//! M4-62 hybrid + 3VL proptest per ADR-038 amendment-03 §TIER-2-b/c.
//!
//! # Pin set
//!
//! 1. `prop_hybrid_fusion_stability_under_permutation` — RRF fusion
//!    is rank-based: permuting the OPERAND order (vector first vs.
//!    text first) does NOT change the fused output set; the score
//!    ordering may shift on ties (broken by NodeId ascending), but
//!    the SET of nodes returned is invariant under operand
//!    permutation.
//! 2. `prop_three_valued_associativity` — AND / OR are associative +
//!    commutative under 3VL; NOT is involutive on True/False (Unknown
//!    is its own image).
//!
//! # Cite
//!
//! - ADR-038 amendment-03 §TIER-2-b/c.
//! - Cormack SIGIR 2009 (RRF rank-equivalence under linear
//!   re-weighting).

use arcgraph_core::{LabelId, Lsn, NodeId, PartitionId, TenantId};
use arcgraph_query::ast::{Expression as AstExpression, Literal};
use arcgraph_query::error::Span;
use arcgraph_query::executor::ops::RankByHybridOp;
use arcgraph_query::executor::value::NodeView;
use arcgraph_query::executor::{
    ExecutionContext, PhysicalOperator, RankedHit, StubExecutorSubstrate, ThreeValued, Value,
};
use arcgraph_query::logical_plan::{HybridOperand, HybridOperandKind};
use arcgraph_query::semantic::bound_ast::{BindingId, BoundExpression};
use proptest::prelude::*;

fn make_node(id: u64) -> NodeView {
    NodeView::new(NodeId::new(id), Some(LabelId::new(1)))
}

fn make_substrate(vector_ids: &[u64], bm25_ids: &[u64]) -> StubExecutorSubstrate {
    let qv = [1.5_f32, 0.0];
    let tag = StubExecutorSubstrate::vector_search_tag_for(&qv);
    let mut s = StubExecutorSubstrate::new()
        .with_vector_substrate()
        .with_bm25_substrate();
    for id in vector_ids.iter().chain(bm25_ids.iter()) {
        s = s.with_node(TenantId::DEFAULT, make_node(*id));
    }
    let v_hits: Vec<RankedHit> = vector_ids
        .iter()
        .map(|id| RankedHit {
            node: make_node(*id),
            score: *id as f64,
        })
        .collect();
    let b_hits: Vec<RankedHit> = bm25_ids
        .iter()
        .map(|id| RankedHit {
            node: make_node(*id),
            score: *id as f64,
        })
        .collect();
    s.with_vector_hit(TenantId::DEFAULT, "embedding", &tag, v_hits)
        .with_bm25_hit(TenantId::DEFAULT, "content", "alpha", b_hits)
}

fn vector_query() -> BoundExpression {
    BoundExpression::Literal {
        value: Literal::List(vec![
            AstExpression::Literal(Literal::Float(1.5)),
            AstExpression::Literal(Literal::Float(0.0)),
        ]),
        span: Span::point(1, 1),
        type_info: None,
    }
}

fn text_query() -> BoundExpression {
    BoundExpression::Literal {
        value: Literal::String("alpha".into()),
        span: Span::point(1, 1),
        type_info: None,
    }
}

fn vec_op() -> HybridOperand {
    HybridOperand {
        kind: HybridOperandKind::Vector,
        var: BindingId::new(0),
        property: "embedding".into(),
        query: vector_query(),
        k: 100,
        read_lsn: Lsn::MAX,
        span: Span::point(1, 1),
    }
}

fn text_op() -> HybridOperand {
    HybridOperand {
        kind: HybridOperandKind::Text,
        var: BindingId::new(0),
        property: "content".into(),
        query: text_query(),
        k: 100,
        read_lsn: Lsn::MAX,
        span: Span::point(1, 1),
    }
}

fn run_hybrid(operands: Vec<HybridOperand>, sub: &StubExecutorSubstrate) -> Vec<NodeId> {
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let mut op = PhysicalOperator::RankByHybrid(RankByHybridOp::new(operands, Lsn::MAX));
    let mut ids = Vec::new();
    loop {
        let b = op.next_batch(&ctx, sub).unwrap();
        if b.is_empty() {
            break;
        }
        for row in b.rows() {
            match &row[0] {
                Value::Node(n) => ids.push(n.id),
                _ => panic!("expected Node"),
            }
        }
    }
    ids
}

proptest! {
    /// PROP-1: RRF fusion output SET is invariant under operand
    /// permutation.
    ///
    /// RRF computes `Σ 1 / (k + rank_i)` over each retriever's
    /// rank — the sum is commutative, so swapping retriever order
    /// does NOT change any node's score. The resulting set + order
    /// (modulo tie-breaks) is identical.
    ///
    /// We use small fixed N so the fixture stays trivial and the
    /// proptest's randomization exercises the operand-order axis,
    /// not substrate population.
    #[test]
    fn prop_hybrid_fusion_stability_under_permutation(
        v_count in 1usize..=5,
        t_count in 1usize..=5,
    ) {
        let vector_ids: Vec<u64> = (1..=v_count as u64).collect();
        let bm25_ids: Vec<u64> = (10..(10 + t_count as u64)).collect();
        let s = make_substrate(&vector_ids, &bm25_ids);
        let ids_a = run_hybrid(vec![vec_op(), text_op()], &s);
        let ids_b = run_hybrid(vec![text_op(), vec_op()], &s);
        // Convert to sets for set-equality (the order may differ on
        // ties; tie-break is NodeId asc which IS deterministic but
        // RRF score equality across permutations requires identical
        // order).
        let mut a_sorted = ids_a.clone();
        let mut b_sorted = ids_b.clone();
        a_sorted.sort_by_key(|n| n.raw());
        b_sorted.sort_by_key(|n| n.raw());
        prop_assert_eq!(a_sorted, b_sorted, "fusion output set is permutation-invariant");
        // RRF is also order-equivalent for non-tied scores; since
        // every NodeId appears in at most one retriever's list (the
        // vector_ids and bm25_ids ranges are disjoint), each node's
        // RRF score is `1/(60+r)` which is unique per (retriever,
        // rank) pair. The two outputs MUST have identical order.
        prop_assert_eq!(ids_a, ids_b, "non-tied scores → identical order");
    }

    /// PROP-2: 3VL is associative + commutative; NOT is involutive
    /// on True/False (Unknown is its own image per ADR-038 §2 D-20).
    #[test]
    fn prop_three_valued_associativity(
        a_idx in 0usize..3,
        b_idx in 0usize..3,
        c_idx in 0usize..3,
    ) {
        let states = [ThreeValued::True, ThreeValued::False, ThreeValued::Unknown];
        let a = states[a_idx];
        let b = states[b_idx];
        let c = states[c_idx];
        // Commutativity.
        prop_assert_eq!(a.and(b), b.and(a));
        prop_assert_eq!(a.or(b), b.or(a));
        // Associativity.
        prop_assert_eq!(a.and(b).and(c), a.and(b.and(c)));
        prop_assert_eq!(a.or(b).or(c), a.or(b.or(c)));
        // De Morgan's laws.
        prop_assert_eq!(a.and(b).not(), a.not().or(b.not()));
        prop_assert_eq!(a.or(b).not(), a.not().and(b.not()));
        // Involution (NOT NOT = identity for True/False; Unknown
        // is its own image — but NOT NOT(Unknown) still equals
        // Unknown).
        prop_assert_eq!(a.not().not(), a);
    }
}
