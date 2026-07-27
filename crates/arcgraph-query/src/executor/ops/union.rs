//! [`UnionOp`] — `UNION ALL` set-op concatenation (ADR-185, #649-A1,
//! W28).
//!
//! Lowers from [`crate::logical_plan::LogicalUnion`]. Concatenates the
//! arms' row streams in source order — openCypher v9 §8 `UNION ALL`
//! (keep duplicates). A bare `UNION` (distinct) wraps this op in a
//! [`crate::executor::ops::DistinctOp`] in #649-A2; the dedup logic is
//! deliberately NOT buried here (PE FROZEN CONTRACT item 2).
//!
//! # Streaming + memory — back-of-envelope (PD#5)
//!
//! Pure streaming: the op holds a cursor into the current arm and
//! forwards that arm's batches until EOS, then advances to the next
//! arm. O(1) extra memory beyond one in-flight batch — NO
//! materialization (that is the Distinct/Sort node ABOVE the union, if
//! any). Latency: a single pass over Σ arm rows.
//!
//! # Column realignment (openCypher v9 §8 — order-independent)
//!
//! §8 requires the arms to expose the same column NAME set but permits
//! a different column ORDER; the result columns follow the FIRST arm's
//! order. The bind pass computes a per-arm permutation
//! ([`crate::semantic::bound_ast::BoundUnionQuery::column_orders`]),
//! threaded here verbatim: `column_orders[i][j]` is the position in arm
//! `i`'s row that supplies canonical output column `j`. Arm 0's
//! permutation is the identity (fast path — its batches pass through
//! untouched); a non-identity arm has each row re-projected into
//! canonical order. The output schema is arm 0's schema.
//!
//! # ADR provenance
//!
//! - **ADR-185 §8** — primary cite (openCypher v9 §8 Set operations).

use crate::executor::batch::Batch;
use crate::executor::context::ExecutionContext;
use crate::executor::error::ExecutionError;
use crate::executor::ops::PhysicalOperator;
use crate::executor::substrate::ExecutorSubstrate;
use crate::executor::value::Value;
use crate::semantic::bound_ast::BindingId;

/// `UNION ALL` concatenation operator.
pub struct UnionOp {
    /// The union arms, drained left-to-right.
    arms: Vec<PhysicalOperator>,
    /// Per-arm column permutation (see module docs). `column_orders[i]`
    /// maps canonical output column `j` → source position in arm `i`.
    column_orders: Vec<Vec<usize>>,
    /// Output schema (= arm 0's schema; columns in arm-0 order).
    schema: Vec<BindingId>,
    /// Index of the arm currently being drained.
    current: usize,
}

impl std::fmt::Debug for UnionOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnionOp")
            .field("arm_count", &self.arms.len())
            .field("schema", &self.schema)
            .field("current_arm", &self.current)
            .finish()
    }
}

impl UnionOp {
    /// Construct a [`UnionOp`] from the arms + per-arm column orders.
    /// The output schema is arm 0's schema (`arms` is non-empty by
    /// grammar — a union has ≥2 arms; an empty `arms` degrades to an
    /// empty schema rather than panicking).
    #[must_use]
    pub fn new(arms: Vec<PhysicalOperator>, column_orders: Vec<Vec<usize>>) -> Self {
        let schema = arms
            .first()
            .map(|a| a.schema().to_vec())
            .unwrap_or_default();
        Self {
            arms,
            column_orders,
            schema,
            current: 0,
        }
    }

    /// Output schema.
    #[must_use]
    pub fn schema(&self) -> &[BindingId] {
        &self.schema
    }

    /// Pull the next batch. Forwards the current arm's batches (column-
    /// realigned if its permutation is non-identity), advancing to the
    /// next arm on EOS, until all arms are exhausted.
    pub fn next_batch<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<Batch, ExecutionError> {
        ctx.cancellation().check()?;
        loop {
            let idx = self.current;
            if idx >= self.arms.len() {
                // All arms exhausted.
                return Ok(Batch::empty(self.schema.len()));
            }
            let batch = self.arms[idx].next_batch(ctx, substrate)?;
            if batch.is_empty() {
                // Current arm EOS — advance to the next arm.
                self.current += 1;
                continue;
            }
            // Realign columns to canonical (arm-0) order if this arm's
            // permutation is non-identity. The borrow of
            // `self.column_orders` here is disjoint from the (now-ended)
            // mutable borrow of `self.arms[idx]` above.
            match self.column_orders.get(idx) {
                Some(perm) if !is_identity_permutation(perm) => {
                    let mut out = Batch::with_capacity(self.schema.len());
                    for row in batch.into_rows() {
                        let mut new_row: Vec<Value> = Vec::with_capacity(perm.len());
                        for &src in perm {
                            // Defensive: a well-formed permutation indexes
                            // in-range (bind validated the name sets are
                            // equal); fall back to NULL on the impossible
                            // out-of-range case rather than panicking.
                            new_row.push(row.get(src).cloned().unwrap_or(Value::Null));
                        }
                        if !out.push_row(new_row) {
                            return Err(ExecutionError::Eval(
                                "UnionOp: batch overflow during column realignment".into(),
                            ));
                        }
                    }
                    return Ok(out);
                }
                // Identity (arm 0, or a same-order arm) — pass through.
                _ => return Ok(batch),
            }
        }
    }
}

/// `true` iff `perm == [0, 1, …, perm.len()-1]` (the no-op reordering).
fn is_identity_permutation(perm: &[usize]) -> bool {
    perm.iter().enumerate().all(|(i, &p)| i == p)
}

#[cfg(test)]
mod tests {
    use arcgraph_core::{LabelId, Lsn, NodeId, PartitionId, TenantId};

    use super::*;
    use crate::executor::ops::ScanOp;
    use crate::executor::substrate::StubExecutorSubstrate;
    use crate::executor::value::NodeView;

    fn ctx() -> ExecutionContext {
        ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO)
    }

    // Two label buckets so two scans return disjoint node sets.
    fn two_label_substrate() -> StubExecutorSubstrate {
        StubExecutorSubstrate::new()
            .with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(1), Some(LabelId::new(1))),
            )
            .with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(2), Some(LabelId::new(2))),
            )
    }

    fn scan(label: u32, binding: u64) -> ScanOp {
        ScanOp::new(BindingId::new(binding), Some(LabelId::new(label)), Lsn::MAX)
    }

    fn drain(
        op: &mut UnionOp,
        ctx: &ExecutionContext,
        s: &StubExecutorSubstrate,
    ) -> Vec<Vec<Value>> {
        let mut rows = Vec::new();
        loop {
            let b = op.next_batch(ctx, s).unwrap();
            if b.is_empty() {
                break;
            }
            rows.extend(b.into_rows());
        }
        rows
    }

    #[test]
    fn union_all_concatenates_arms_in_order() {
        let s = two_label_substrate();
        // arm 0: label 1 (node 1); arm 1: label 2 (node 2).
        let arms = vec![
            PhysicalOperator::Scan(scan(1, 0)),
            PhysicalOperator::Scan(scan(2, 0)),
        ];
        // Single column each, same order → identity permutations.
        let column_orders = vec![vec![0], vec![0]];
        let mut op = UnionOp::new(arms, column_orders);
        let rows = drain(&mut op, &ctx(), &s);
        let ids: Vec<u64> = rows
            .iter()
            .map(|r| match &r[0] {
                Value::Node(n) => n.id.raw(),
                _ => panic!("Node"),
            })
            .collect();
        // Arm-0 rows first, then arm-1 rows (concat, order preserved).
        assert_eq!(ids, vec![1, 2]);
    }

    #[test]
    fn union_all_keeps_duplicates() {
        let s = two_label_substrate();
        // Both arms scan label 1 (node 1) → UNION ALL keeps the dup.
        let arms = vec![
            PhysicalOperator::Scan(scan(1, 0)),
            PhysicalOperator::Scan(scan(1, 0)),
        ];
        let mut op = UnionOp::new(arms, vec![vec![0], vec![0]]);
        let rows = drain(&mut op, &ctx(), &s);
        assert_eq!(rows.len(), 2, "UNION ALL keeps the duplicate row");
    }

    #[test]
    fn union_propagates_cancel() {
        let s = two_label_substrate();
        let ctx = ctx();
        ctx.cancellation().cancel();
        let arms = vec![
            PhysicalOperator::Scan(scan(1, 0)),
            PhysicalOperator::Scan(scan(2, 0)),
        ];
        let mut op = UnionOp::new(arms, vec![vec![0], vec![0]]);
        assert_eq!(op.next_batch(&ctx, &s), Err(ExecutionError::Cancelled));
    }
}
