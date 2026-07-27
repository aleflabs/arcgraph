//! [`SkipOp`] — `SKIP N` operator (#842 part A).
//!
//! Lowers from [`crate::logical_plan::LogicalSkip`]. Discards the first
//! `N` upstream rows, then forwards every remaining row unchanged
//! (offset-pagination semantics per Cypher 9 §6.6). The companion of
//! [`crate::executor::ops::LimitOp`] — `SKIP 5 LIMIT 10` (the "page 3"
//! idiom) composes `Limit(Skip(child))` and the two together implement
//! offset+keyset pagination, which #842 reported was impossible (SKIP
//! erroring `-32005` everywhere while LIMIT worked).
//!
//! # Boundary cases
//!
//! - `SKIP 0` — forwards every upstream row unchanged (no-op offset).
//! - `SKIP N == upstream_count` — discards everything; emits EOS.
//! - `SKIP N > upstream_count` — discards everything; emits EOS.
//! - `SKIP N` straddling a batch boundary — discards whole upstream
//!   batches until the offset is consumed, then emits a partial batch
//!   carrying exactly the post-offset rows, then streams the rest.
//!
//! # Why an internal pull loop (vs. one-batch-per-call)
//!
//! An empty batch is the EOS sentinel (see [`Batch::empty`]). If `SKIP`
//! consumes an entire upstream batch it must NOT return empty — that
//! would signal EOS to the consumer mid-skip. Instead it loops, pulling
//! and discarding fully-skipped upstream batches, until it either
//! reaches a batch with surviving rows (returns the remainder) or
//! upstream genuinely exhausts (returns empty = real EOS). The loop is
//! bounded by the number of upstream batches.
//!
//! # Memory + latency — back-of-envelope (PD#5)
//!
//! [`SkipOp`] is non-blocking and allocation-light: it forwards rows as
//! they arrive without buffering, so retained state is O(1) (the single
//! `remaining` counter) — NOT O(skipped-rows) nor O(input). Latency is
//! O(skipped-rows) for the discard phase (a counter decrement per
//! upstream batch, not per row, in the whole-batch-skip path) then O(1)
//! per forwarded batch (pure pass-through). No memory-budget reservation
//! is needed — the upstream operator's reservation flow already covers
//! row lifetimes (same rationale as `LimitOp`). The cancellation token
//! is checked at every loop iteration (defense-in-depth, mirrors
//! `LimitOp`'s per-batch-boundary check).
//!
//! # Why a separate operator (vs. an attribute on Project)
//!
//! Same rationale as `LimitOp`: SKIP can sit between any two operators
//! in the M4-32 lowering tree (the lowering places it after Sort /
//! Distinct, before Limit). A standalone op keeps the M4-05 cost-walker's
//! plan-rewrite freedom and matches the codebase's one-op-per-file style.
//!
//! # Scope (literal count only)
//!
//! This op consumes [`crate::logical_plan::LogicalSkip`], which the M4-33
//! lowering produces only for a LITERAL `SKIP <int>` (see
//! `crate::logical_plan::lowering::lower_skip_or_limit_with_span`). A
//! parameter / expression `SKIP $n` lowers to
//! [`crate::logical_plan::LogicalDynamicLimit`] instead and remains
//! `NotImplemented` at the pipeline build (composes with #797's
//! parameter threading — a follow-up).
//!
//! # ADR provenance
//!
//! - **ADR-038 §2 D-28** — SKIP / LIMIT operator contract (this op
//!   closes the D-28 SKIP deferral the LIMIT slice (M4-63) left open).
//! - **Cypher 9 §6.6** — SKIP / LIMIT semantics + ordering.

use crate::executor::batch::Batch;
use crate::executor::context::ExecutionContext;
use crate::executor::error::ExecutionError;
use crate::executor::ops::PhysicalOperator;
use crate::executor::substrate::ExecutorSubstrate;
use crate::semantic::bound_ast::BindingId;

/// `SKIP N` operator.
#[derive(Debug)]
pub struct SkipOp {
    child: Box<PhysicalOperator>,
    /// Rows still to discard before forwarding begins. Decremented as
    /// upstream rows are consumed; once 0, the operator is a pure
    /// pass-through.
    remaining: u64,
    /// Cached output schema (= input schema; SKIP preserves columns).
    schema: Vec<BindingId>,
}

impl SkipOp {
    /// Construct a [`SkipOp`].
    #[must_use]
    pub fn new(child: PhysicalOperator, count: u64) -> Self {
        let schema = child.schema().to_vec();
        Self {
            child: Box::new(child),
            remaining: count,
            schema,
        }
    }

    /// Output schema (= input schema).
    pub fn schema(&self) -> &[BindingId] {
        &self.schema
    }

    /// Pull the next batch.
    ///
    /// Discards upstream rows until the `SKIP` offset is consumed, then
    /// forwards remaining rows. Loops over (and discards) fully-skipped
    /// upstream batches so a consumed batch never surfaces as a
    /// premature EOS (see the module rustdoc).
    pub fn next_batch<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<Batch, ExecutionError> {
        loop {
            ctx.cancellation().check()?;
            let upstream = self.child.next_batch(ctx, substrate)?;
            if upstream.is_empty() {
                // Genuine upstream EOS — forward it. (Covers both
                // "skipped everything" and "nothing left after offset".)
                return Ok(upstream);
            }
            if self.remaining == 0 {
                // Offset fully consumed: pure pass-through.
                return Ok(upstream);
            }
            let row_count = upstream.row_count() as u64;
            if row_count <= self.remaining {
                // Discard the whole batch; loop to pull the next one.
                self.remaining -= row_count;
                continue;
            }
            // Partial-skip case: drop the first `remaining` rows, return
            // the rest. `to_skip < row_count`, so the survivors fit the
            // batch they came from.
            let to_skip = self.remaining as usize;
            self.remaining = 0;
            let mut out = Batch::with_capacity(self.schema.len());
            for row in upstream.into_rows().into_iter().skip(to_skip) {
                if !out.push_row(row) {
                    return Err(ExecutionError::Eval(
                        "SkipOp: batch overflow during post-offset push".into(),
                    ));
                }
            }
            return Ok(out);
        }
    }
}

#[cfg(test)]
mod tests {
    use arcgraph_core::{LabelId, Lsn, NodeId, PartitionId, TenantId};

    use super::*;
    use crate::executor::ops::{LimitOp, ScanOp};
    use crate::executor::substrate::StubExecutorSubstrate;
    use crate::executor::value::{NodeView, Value};

    fn make_n_persons(n: u64) -> StubExecutorSubstrate {
        let mut s = StubExecutorSubstrate::new();
        for i in 1..=n {
            s = s.with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(i), Some(LabelId::new(1))),
            );
        }
        s
    }

    fn person_scan() -> ScanOp {
        ScanOp::new(BindingId::new(0), Some(LabelId::new(1)), Lsn::MAX)
    }

    fn ctx() -> ExecutionContext {
        ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO)
    }

    /// Drain a batch of single-node rows into the ordered node ids — the
    /// "correct rows" oracle (row_count alone would not catch an
    /// off-by-one in WHICH rows survive the offset).
    fn node_ids(batch: &Batch) -> Vec<u64> {
        batch
            .rows()
            .iter()
            .map(|row| match &row[0] {
                Value::Node(nv) => nv.id.raw(),
                other => panic!("expected Value::Node, got {other:?}"),
            })
            .collect()
    }

    #[test]
    fn skip_zero_forwards_all_rows() {
        // SKIP 0 boundary: pass through every row, in order.
        let s = make_n_persons(5);
        let mut op = SkipOp::new(PhysicalOperator::Scan(person_scan()), 0);
        let ctx = ctx();
        let b1 = op.next_batch(&ctx, &s).unwrap();
        assert_eq!(node_ids(&b1), vec![1, 2, 3, 4, 5], "SKIP 0 forwards all");
        let b2 = op.next_batch(&ctx, &s).unwrap();
        assert!(b2.is_empty(), "EOS after upstream exhausted");
    }

    #[test]
    fn skip_two_drops_first_two_keeps_correct_rows() {
        // SKIP 2 over [1..=5]: emit nodes 3,4,5 (NOT just "3 rows").
        let s = make_n_persons(5);
        let mut op = SkipOp::new(PhysicalOperator::Scan(person_scan()), 2);
        let ctx = ctx();
        let b1 = op.next_batch(&ctx, &s).unwrap();
        assert_eq!(node_ids(&b1), vec![3, 4, 5], "SKIP 2 drops rows 1-2");
        let b2 = op.next_batch(&ctx, &s).unwrap();
        assert!(b2.is_empty(), "EOS after the post-offset rows drain");
    }

    #[test]
    fn skip_equal_to_count_emits_empty() {
        // SKIP N == upstream count: discard everything, emit EOS.
        let s = make_n_persons(3);
        let mut op = SkipOp::new(PhysicalOperator::Scan(person_scan()), 3);
        let ctx = ctx();
        let b1 = op.next_batch(&ctx, &s).unwrap();
        assert!(b1.is_empty(), "SKIP == count discards all rows");
    }

    #[test]
    fn skip_greater_than_count_emits_empty() {
        // SKIP N > upstream count: discard everything, emit EOS (does
        // NOT underflow the counter — saturating semantics).
        let s = make_n_persons(2);
        let mut op = SkipOp::new(PhysicalOperator::Scan(person_scan()), 100);
        let ctx = ctx();
        let b1 = op.next_batch(&ctx, &s).unwrap();
        assert!(b1.is_empty(), "SKIP > count discards all rows");
    }

    #[test]
    fn skip_then_limit_paginates() {
        // `SKIP 1 LIMIT 2` over [1..=5] = page 2 (size 2) = nodes 2,3.
        // Lowering composes Limit(Skip(child)); build the same stack.
        let s = make_n_persons(5);
        let skip = SkipOp::new(PhysicalOperator::Scan(person_scan()), 1);
        let mut op = LimitOp::new(PhysicalOperator::Skip(skip), 2);
        let ctx = ctx();
        let b1 = op.next_batch(&ctx, &s).unwrap();
        assert_eq!(node_ids(&b1), vec![2, 3], "SKIP 1 LIMIT 2 = nodes 2,3");
        let b2 = op.next_batch(&ctx, &s).unwrap();
        assert!(b2.is_empty(), "LIMIT short-circuits after 2 rows");
    }

    #[test]
    fn skip_propagates_cancel() {
        let s = make_n_persons(5);
        let ctx = ctx();
        ctx.cancellation().cancel();
        let mut op = SkipOp::new(PhysicalOperator::Scan(person_scan()), 2);
        let r = op.next_batch(&ctx, &s);
        assert_eq!(r, Err(ExecutionError::Cancelled));
    }

    #[test]
    fn skip_straddles_batch_boundary_correctly() {
        // Substrate has BATCH_ROWS + 5 rows; SKIP BATCH_ROWS + 2.
        // The op discards the whole first batch (BATCH_ROWS rows) plus
        // 2 more from the second, then emits the 3 survivors, then EOS.
        use crate::executor::BATCH_ROWS;
        let total = (BATCH_ROWS + 5) as u64;
        let s = make_n_persons(total);
        let skip = (BATCH_ROWS + 2) as u64;
        let mut op = SkipOp::new(PhysicalOperator::Scan(person_scan()), skip);
        let ctx = ctx();
        let b1 = op.next_batch(&ctx, &s).unwrap();
        // Survivors are the last 3 nodes (ids total-2, total-1, total).
        assert_eq!(
            node_ids(&b1),
            vec![total - 2, total - 1, total],
            "post-offset survivors straddle the batch boundary"
        );
        let b2 = op.next_batch(&ctx, &s).unwrap();
        assert!(b2.is_empty(), "EOS after survivors drain");
    }
}
