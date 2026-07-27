//! [`RankByHybridOp`] + [`FusionOp`] — M4-62 hybrid retrieval.
//!
//! [`RankByHybridOp`] composes M3.a (HNSW vector) + M3.b (BM25 text)
//! retrieval substrates per ADR-038 amendment-03 §TIER-2-c. v1.0-alpha
//! lights the VECTOR + TEXT operands ONLY — `HybridOperandKind` carries
//! `Vector` / `Text` variants, no `Community` variant. The M3.d
//! community substrate is reached via a separate filter-shaped plan
//! node ([`crate::logical_plan::LogicalCommunityLookup`], a `M4-62b`
//! forward-pin), NOT through the `RankByHybridOp` operand list.
//!
//! # W11Z fix-up LOW-1 (PR #268 retro)
//!
//! The pre-fix-up doc-comment claimed "composes the three substrates"
//! and "the 3-substrate accessibility check still runs", but
//! [`RankByHybridOp`]'s internal fusion consumes only VECTOR + TEXT
//! operands (no community lookup). Reframed here to match the
//! implementation: 2-substrate composition; community is a
//! forward-pin to M4-62b.
//! The trait method
//! [`crate::executor::ExecutorSubstrate::community_members`] is
//! present for parallel-symmetry with `CatalogProvider` and the
//! anticipated M4-62b consumer; no in-slice operator consumes it
//! (forward-pin documented at the trait method too).
//!
//! [`FusionOp`] computes Reciprocal-Rank Fusion (RRF) per Cormack
//! SIGIR 2009: `score = Σ 1 / (k + rank_i)`.
//!
//! # ADR provenance
//! - **ADR-038 amendment-03 §TIER-2-c** — RANK BY HYBRID 2-substrate
//!   composition (VECTOR + TEXT at v1.0-alpha; COMMUNITY operand
//!   forward-pinned to M4-62b LogicalCommunityLookup).
//! - **ADR-037 §D-1** — `TenantHandle` per-tenant substrate
//!   composition (the production binding via this op's
//!   [`ExecutorSubstrate`] consumer reaches the same handle at
//!   M4-08+).
//! - **Cormack SIGIR 2009** — RRF fusion algorithm.

use arcgraph_core::Lsn;

use crate::executor::batch::{BATCH_ROWS, Batch};
use crate::executor::context::ExecutionContext;
use crate::executor::error::ExecutionError;
use crate::executor::eval::{Parameters, evaluate};
use crate::executor::fusion::rrf_fuse;
use crate::executor::ops::PhysicalOperator;
use crate::executor::substrate::{ExecutorSubstrate, RankedHit};
use crate::executor::value::Value;
use crate::logical_plan::{HybridOperand, HybridOperandKind};
use crate::semantic::bound_ast::BindingId;

/// `RANK BY HYBRID(...)` orchestration operator.
///
/// Pulls each operand's top-K from the underlying substrate, fuses
/// them by RRF (per the attached fusion spec — falls back to k=60 if
/// none attached, matching the ADR-006 amendment-01 §A-2 default),
/// emits the fused ranking as `[node]` rows, or `[node, score]` when
/// the source clause declares a score binding, in score-descending
/// order.
///
/// # Schema
///
/// Output schema = `[node_var]` by default, or
/// `[node_var, score_var]` for `RANK BY HYBRID(...) AS score_var`.
#[derive(Debug)]
pub struct RankByHybridOp {
    /// The bound variable carrying the fused ranked-node payload.
    /// Inferred from the operand list (all hybrid operands share the
    /// same root binding per M4-32 lowering invariant; we record the
    /// first operand's `var`). Mirrored in `schema[0]`.
    #[allow(dead_code)]
    binding: BindingId,
    /// Operand list (per ADR-038 §2 D-3 v1.0 admits VECTOR + TEXT).
    operands: Vec<HybridOperand>,
    /// MVCC visibility key.
    plan_read_lsn: Lsn,
    /// RRF smoothing constant. Defaults to 60 (Cormack SIGIR 2009).
    /// Bound by the fusion-clause's `k` param when present.
    rrf_k: u64,
    /// Optional binding exposing the fused score.
    score_binding: Option<BindingId>,
    /// Per-query parameter bag.
    parameters: Parameters,
    /// Buffered fused-ranking output. Primed at first batch.
    buffer: Option<Vec<RankedHit>>,
    /// Cursor into the buffer.
    cursor: usize,
    /// Cached schema.
    schema: Vec<BindingId>,
}

impl RankByHybridOp {
    /// Construct a `RankByHybridOp` from operand list + plan_read_lsn.
    /// `rrf_k` defaults to 60; pass [`Self::with_fusion_k`] to override.
    #[must_use]
    pub fn new(operands: Vec<HybridOperand>, plan_read_lsn: Lsn) -> Self {
        // The hybrid root binding is shared across operands per the
        // M4-32 lowering invariant; pick the first operand's.
        let binding = operands
            .first()
            .map(|o| o.var)
            .unwrap_or_else(|| BindingId::new(0));
        Self {
            binding,
            operands,
            plan_read_lsn,
            rrf_k: 60,
            score_binding: None,
            parameters: Parameters::new(),
            buffer: None,
            cursor: 0,
            schema: vec![binding],
        }
    }

    /// Override the RRF smoothing constant.
    #[must_use]
    pub fn with_fusion_k(mut self, k: u64) -> Self {
        self.rrf_k = k;
        self
    }

    /// Expose each result's fused score under `binding`.
    #[must_use]
    pub fn with_score_binding(mut self, binding: BindingId) -> Self {
        self.score_binding = Some(binding);
        self.schema.push(binding);
        self
    }

    /// Inject a per-query parameter bag.
    #[must_use]
    pub fn with_parameters(mut self, parameters: Parameters) -> Self {
        self.parameters = parameters;
        self
    }

    /// Output schema.
    pub fn schema(&self) -> &[BindingId] {
        &self.schema
    }

    /// Pull the next batch.
    pub fn next_batch<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<Batch, ExecutionError> {
        ctx.cancellation().check()?;

        if self.buffer.is_none() {
            // Snapshot LSN acquired pre-first-substrate-call.
            let _ = ctx.ensure_snapshot_lsn();
            self.buffer = Some(self.fuse(ctx, substrate)?);
        }
        let buf = self.buffer.as_ref().expect("primed above");
        if self.cursor >= buf.len() {
            return Ok(Batch::empty(self.schema.len()));
        }
        let mut batch = Batch::with_capacity(self.schema.len());
        let take = (buf.len() - self.cursor).min(BATCH_ROWS);
        for hit in &buf[self.cursor..self.cursor + take] {
            let mut row = vec![Value::Node(hit.node.clone())];
            if self.score_binding.is_some() {
                row.push(Value::Float(hit.score));
            }
            let _ = batch.push_row(row);
        }
        self.cursor += take;
        Ok(batch)
    }

    /// Run all operands + fuse.
    fn fuse<S: ExecutorSubstrate>(
        &self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<Vec<RankedHit>, ExecutionError> {
        // Do not pre-gate on `has_vector_substrate()` /
        // `has_bm25_substrate()`: those legacy availability mirrors do
        // not carry a tenant, while search availability is per-tenant.
        // The authoritative calls below do carry `ctx.tenant()` and
        // already fail loudly with `IndexUnavailable` naming the
        // missing tenant substrate (or provider). A tenant-free
        // defense-in-depth check can therefore contradict both the
        // semantic catalog and the real served search path.
        // Pull each operand's top-K.
        let mut all_hits: Vec<Vec<RankedHit>> = Vec::with_capacity(self.operands.len());
        for op in &self.operands {
            ctx.cancellation().check()?;
            let hits = match op.kind {
                HybridOperandKind::Vector => {
                    let qv = self.resolve_query_vector(&op.query)?;
                    substrate.vector_search(
                        ctx.tenant(),
                        &op.property,
                        &qv,
                        op.k,
                        self.plan_read_lsn,
                    )?
                }
                HybridOperandKind::Text => {
                    let qt = self.resolve_query_text(&op.query)?;
                    substrate.bm25_search(
                        ctx.tenant(),
                        &op.property,
                        &qt,
                        op.k,
                        self.plan_read_lsn,
                    )?
                }
            };
            all_hits.push(hits);
        }
        Ok(rrf_fuse(&all_hits, self.rrf_k))
    }

    /// Resolve a hybrid operand's vector query expression to a
    /// `Vec<f32>`. v1.0-alpha admits parameter / list-literal forms;
    /// other shapes surface a runtime eval error.
    fn resolve_query_vector(
        &self,
        expr: &crate::semantic::bound_ast::BoundExpression,
    ) -> Result<Vec<f32>, ExecutionError> {
        let lookup = |_: BindingId| None;
        let v = evaluate(expr, &[], &lookup, &self.parameters)?;
        match v {
            Value::List(elems) => {
                let mut out: Vec<f32> = Vec::with_capacity(elems.len());
                for e in elems {
                    match e {
                        Value::Float(f) => out.push(f as f32),
                        Value::Integer(i) => out.push(i as f32),
                        Value::Null => {
                            return Err(ExecutionError::Eval(
                                "vector operand contains NULL".into(),
                            ));
                        }
                        _ => {
                            return Err(ExecutionError::Eval(
                                "vector operand element is non-numeric".into(),
                            ));
                        }
                    }
                }
                Ok(out)
            }
            Value::Null => Err(ExecutionError::Eval(
                "vector operand resolved to NULL".into(),
            )),
            _ => Err(ExecutionError::Eval(
                "vector operand must be a list (or list parameter)".into(),
            )),
        }
    }

    /// Resolve a hybrid operand's text query expression to a String.
    fn resolve_query_text(
        &self,
        expr: &crate::semantic::bound_ast::BoundExpression,
    ) -> Result<String, ExecutionError> {
        let lookup = |_: BindingId| None;
        let v = evaluate(expr, &[], &lookup, &self.parameters)?;
        match v {
            Value::String(s) => Ok(s),
            Value::Null => Err(ExecutionError::Eval("text operand resolved to NULL".into())),
            _ => Err(ExecutionError::Eval(
                "text operand must be a string (or string parameter)".into(),
            )),
        }
    }
}

// =====================================================================
// FusionOp — `WITH FUSION = RRF(k = N)` standalone.
// =====================================================================

/// `WITH FUSION = RRF(k = N)` standalone fusion node.
///
/// At v1.0 the only fusion algorithm lit is RRF (per ADR-038 §2 D-9).
/// This op is a thin pass-through over a [`RankByHybridOp`] when the
/// fusion appears alongside a hybrid block; for now its presence as a
/// top-level operator is rare (the M4-32 lowering may emit
/// `Fusion(RankByHybrid(...))` or just `RankByHybrid(...)` depending
/// on the source-side clause sequencing). When present, this op
/// just re-emits the upstream rows unchanged — the fusion happens
/// inside [`RankByHybridOp`].
#[derive(Debug)]
pub struct FusionOp {
    child: Box<PhysicalOperator>,
    schema: Vec<BindingId>,
}

impl FusionOp {
    /// Construct a `FusionOp` over a child. The child's schema is
    /// preserved.
    pub fn new(child: PhysicalOperator) -> Self {
        let schema = child.schema().to_vec();
        Self {
            child: Box::new(child),
            schema,
        }
    }

    /// Output schema (= input schema).
    pub fn schema(&self) -> &[BindingId] {
        &self.schema
    }

    /// Pass-through next_batch.
    pub fn next_batch<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<Batch, ExecutionError> {
        ctx.cancellation().check()?;
        self.child.next_batch(ctx, substrate)
    }
}

#[cfg(test)]
mod tests {
    use arcgraph_core::{LabelId, NodeId, PartitionId, TenantId};

    use super::*;
    use crate::ast::Literal;
    use crate::error::Span;
    use crate::executor::substrate::StubExecutorSubstrate;
    use crate::executor::value::NodeView;
    use crate::semantic::bound_ast::BoundExpression;

    fn alice() -> NodeView {
        NodeView::new(NodeId::new(1), Some(LabelId::new(1)))
    }
    fn bob() -> NodeView {
        NodeView::new(NodeId::new(2), Some(LabelId::new(1)))
    }
    fn carol() -> NodeView {
        NodeView::new(NodeId::new(3), Some(LabelId::new(1)))
    }

    fn vector_query() -> BoundExpression {
        BoundExpression::Literal {
            value: Literal::List(vec![
                crate::ast::Expression::Literal(Literal::Float(1.5)),
                crate::ast::Expression::Literal(Literal::Float(0.0)),
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

    fn vec_op(binding: BindingId) -> HybridOperand {
        HybridOperand {
            kind: HybridOperandKind::Vector,
            var: binding,
            property: "embedding".into(),
            query: vector_query(),
            k: 10,
            read_lsn: Lsn::MAX,
            span: Span::point(1, 1),
        }
    }

    fn text_op(binding: BindingId) -> HybridOperand {
        HybridOperand {
            kind: HybridOperandKind::Text,
            var: binding,
            property: "content".into(),
            query: text_query(),
            k: 10,
            read_lsn: Lsn::MAX,
            span: Span::point(1, 1),
        }
    }

    fn fixture_substrate() -> StubExecutorSubstrate {
        // Vector top-K (3.14 → tag): Alice, Bob, Carol
        // BM25 top-K ("alpha"):       Carol, Alice, Bob
        // Test-only sentinel; using `1.5` (NOT π) so the
        // `clippy::approx_constant` heuristic doesn't fire — the
        // value is meaningful only as a stable substrate-tag input.
        let qv = [1.5_f32, 0.0];
        let tag = StubExecutorSubstrate::vector_search_tag_for(&qv);
        StubExecutorSubstrate::new()
            .with_vector_substrate()
            .with_bm25_substrate()
            .with_community_substrate()
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
                    RankedHit {
                        node: carol(),
                        score: 0.1,
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
                    RankedHit {
                        node: bob(),
                        score: 1.0,
                    },
                ],
            )
    }

    #[test]
    fn rank_by_hybrid_3_substrate_check_passes_when_attached() {
        let s = fixture_substrate();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let mut op = RankByHybridOp::new(
            vec![vec_op(BindingId::new(0)), text_op(BindingId::new(0))],
            Lsn::MAX,
        );
        let b = op.next_batch(&ctx, &s).unwrap();
        // Three nodes fused → three rows.
        assert_eq!(b.row_count(), 3);
    }

    #[test]
    fn rank_by_hybrid_substrate_unavailable_surfaces_error() {
        // Vector substrate not attached → the authoritative,
        // tenant-aware vector_search call names what is missing.
        let s = StubExecutorSubstrate::new();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let mut op = RankByHybridOp::new(vec![vec_op(BindingId::new(0))], Lsn::MAX);
        let r = op.next_batch(&ctx, &s);
        assert_eq!(
            r,
            Err(ExecutionError::Substrate(
                crate::executor::substrate::SubstrateAccessError::IndexUnavailable("vector".into())
            ))
        );
    }

    #[test]
    fn rrf_fusion_is_score_descending() {
        // Both lists rank: vec=A,B,C; bm25=C,A,B with k=60.
        // RRF score:
        //   Alice = 1/61 + 1/62 = 0.0163934 + 0.0161290 = 0.0325224
        //   Bob   = 1/62 + 1/63 = 0.0161290 + 0.0158730 = 0.0320020
        //   Carol = 1/63 + 1/61 = 0.0158730 + 0.0163934 = 0.0322664
        // Order: Alice > Carol > Bob
        let s = fixture_substrate();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let mut op = RankByHybridOp::new(
            vec![vec_op(BindingId::new(0)), text_op(BindingId::new(0))],
            Lsn::MAX,
        );
        let b = op.next_batch(&ctx, &s).unwrap();
        let ids: Vec<NodeId> = b
            .rows()
            .iter()
            .map(|r| match &r[0] {
                Value::Node(n) => n.id,
                _ => panic!("expected Node"),
            })
            .collect();
        assert_eq!(
            ids,
            vec![NodeId::new(1), NodeId::new(3), NodeId::new(2)],
            "RRF order: Alice(1) > Carol(3) > Bob(2)"
        );
    }

    #[test]
    fn score_binding_emits_exact_fused_score_next_to_node() {
        let s = fixture_substrate();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let mut op = RankByHybridOp::new(
            vec![vec_op(BindingId::new(0)), text_op(BindingId::new(0))],
            Lsn::MAX,
        )
        .with_score_binding(BindingId::new(1));
        let b = op.next_batch(&ctx, &s).unwrap();

        assert_eq!(op.schema(), &[BindingId::new(0), BindingId::new(1)]);
        let ids_and_scores: Vec<(NodeId, f64)> = b
            .rows()
            .iter()
            .map(|row| {
                let Value::Node(node) = &row[0] else {
                    panic!("expected Node")
                };
                let Value::Float(score) = &row[1] else {
                    panic!("expected Float")
                };
                (node.id, *score)
            })
            .collect();
        assert_eq!(
            ids_and_scores,
            vec![
                (NodeId::new(1), 1.0 / 61.0 + 1.0 / 62.0),
                (NodeId::new(3), 1.0 / 63.0 + 1.0 / 61.0),
                (NodeId::new(2), 1.0 / 62.0 + 1.0 / 63.0),
            ]
        );
    }

    #[test]
    fn rrf_fusion_handles_substrates_with_disjoint_results() {
        // Vector returns Alice; BM25 returns Bob. Both should appear
        // in the fusion output with rank-1 contributions only.
        // Test-only sentinel; using `1.5` (NOT π) so the
        // `clippy::approx_constant` heuristic doesn't fire — the
        // value is meaningful only as a stable substrate-tag input.
        let qv = [1.5_f32, 0.0];
        let tag = StubExecutorSubstrate::vector_search_tag_for(&qv);
        let s = StubExecutorSubstrate::new()
            .with_vector_substrate()
            .with_bm25_substrate()
            .with_vector_hit(
                TenantId::DEFAULT,
                "embedding",
                &tag,
                vec![RankedHit {
                    node: alice(),
                    score: 0.9,
                }],
            )
            .with_bm25_hit(
                TenantId::DEFAULT,
                "content",
                "alpha",
                vec![RankedHit {
                    node: bob(),
                    score: 5.0,
                }],
            );
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let mut op = RankByHybridOp::new(
            vec![vec_op(BindingId::new(0)), text_op(BindingId::new(0))],
            Lsn::MAX,
        );
        let b = op.next_batch(&ctx, &s).unwrap();
        let ids: Vec<NodeId> = b
            .rows()
            .iter()
            .map(|r| match &r[0] {
                Value::Node(n) => n.id,
                _ => panic!(),
            })
            .collect();
        // Both at rank 1 → tied score; tie-break by NodeId ascending.
        assert_eq!(ids, vec![NodeId::new(1), NodeId::new(2)]);
    }

    #[test]
    fn fusion_op_passes_through() {
        // Standalone FusionOp acts as a pass-through; presence is
        // load-bearing only at the plan-tree shape level.
        let s = fixture_substrate();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let inner = RankByHybridOp::new(
            vec![vec_op(BindingId::new(0)), text_op(BindingId::new(0))],
            Lsn::MAX,
        );
        let mut op = FusionOp::new(PhysicalOperator::RankByHybrid(inner));
        let b = op.next_batch(&ctx, &s).unwrap();
        assert_eq!(b.row_count(), 3);
    }

    #[test]
    fn rank_by_hybrid_propagates_cancel() {
        let s = fixture_substrate();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        ctx.cancellation().cancel();
        let mut op = RankByHybridOp::new(
            vec![vec_op(BindingId::new(0)), text_op(BindingId::new(0))],
            Lsn::MAX,
        );
        let r = op.next_batch(&ctx, &s);
        assert_eq!(r, Err(ExecutionError::Cancelled));
    }
}
