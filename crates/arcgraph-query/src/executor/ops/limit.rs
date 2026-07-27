//! [`LimitOp`] — `LIMIT N` operator (M4-63).
//!
//! Lowers from [`crate::logical_plan::LogicalLimit`]. Forwards the
//! first `N` upstream rows then short-circuits to EOS without pulling
//! more upstream batches (early-termination semantics per Cypher 9
//! §6.6).
//!
//! # Boundary cases
//!
//! - `LIMIT 0` — emits no rows; closes upstream immediately.
//! - `LIMIT N == upstream_count` — emits all rows then EOS.
//! - `LIMIT N > upstream_count` — emits all upstream rows + EOS.
//! - `LIMIT N` straddling a batch boundary — emits a partial batch
//!   carrying exactly the remaining rows, then EOS.
//!
//! # Memory + cancellation
//!
//! [`LimitOp`] is non-blocking: it forwards rows as they arrive
//! without buffering. No memory budget reservation is needed — the
//! upstream operator's reservation flow already covers row lifetimes.
//! The cancellation token is checked at every batch boundary
//! (defense-in-depth per W11Z #272 LOW-5).
//!
//! # Why a separate operator (vs. an attribute on Project)
//!
//! Because LIMIT can sit between any two operators in the M4-32
//! lowering tree (between Sort and Project, between Aggregate and
//! Sort, etc.). Folding it into Project would constrain the M4-05
//! cost-walker's plan-rewrite freedom.
//!
//! # Forward-pin: dynamic LIMIT
//!
//! Parameter-driven LIMIT (`LIMIT $n`) lowers to
//! [`crate::logical_plan::LogicalDynamicLimit`]; that variant is the
//! [`crate::executor::pipeline::Pipeline`] builder's responsibility to
//! evaluate at construction time and pass the resolved integer to
//! [`LimitOp::new`]. The operator itself is parameter-agnostic.
//!
//! # ADR provenance
//!
//! - **ADR-038 amendment-02 §M4.f** — primary M4-63 cite.
//! - **ADR-038 §2 D-28** — LIMIT operator contract.
//! - **Cypher 9 §6.6** — LIMIT semantics + early termination.

use crate::executor::batch::Batch;
use crate::executor::context::ExecutionContext;
use crate::executor::error::ExecutionError;
use crate::executor::ops::PhysicalOperator;
use crate::executor::substrate::ExecutorSubstrate;
use crate::semantic::bound_ast::BindingId;

/// `LIMIT N` operator.
#[derive(Debug)]
pub struct LimitOp {
    child: Box<PhysicalOperator>,
    /// Remaining rows to emit. Bumped down per row pushed; once 0,
    /// the operator returns EOS immediately on every subsequent call.
    remaining: u64,
    /// Cached output schema (= input schema; LIMIT preserves columns).
    schema: Vec<BindingId>,
    /// Have we hit the limit + emitted EOS?
    done: bool,
}

impl LimitOp {
    /// Construct a [`LimitOp`].
    #[must_use]
    pub fn new(child: PhysicalOperator, count: u64) -> Self {
        let schema = child.schema().to_vec();
        Self {
            child: Box::new(child),
            remaining: count,
            schema,
            done: count == 0,
        }
    }

    /// Output schema (= input schema).
    pub fn schema(&self) -> &[BindingId] {
        &self.schema
    }

    /// Pull the next batch.
    ///
    /// Forwards upstream rows up to `remaining`. When `remaining == 0`,
    /// returns the EOS sentinel without pulling more upstream batches
    /// (early-termination per Cypher 9 §6.6).
    pub fn next_batch<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<Batch, ExecutionError> {
        ctx.cancellation().check()?;
        if self.done || self.remaining == 0 {
            self.done = true;
            return Ok(Batch::empty(self.schema.len()));
        }
        let upstream = self.child.next_batch(ctx, substrate)?;
        if upstream.is_empty() {
            self.done = true;
            return Ok(Batch::empty(self.schema.len()));
        }
        // Decide how many of the upstream rows to forward.
        let to_take_usize = match usize::try_from(self.remaining) {
            Ok(n) => n.min(upstream.row_count()),
            Err(_) => upstream.row_count(),
        };
        // If we take ALL upstream rows AND remaining > upstream count,
        // we just forward the whole batch unchanged.
        if to_take_usize == upstream.row_count() {
            self.remaining = self.remaining.saturating_sub(upstream.row_count() as u64);
            if self.remaining == 0 {
                self.done = true;
            }
            return Ok(upstream);
        }
        // Partial-batch case: build a slice of `to_take_usize` rows.
        let mut out = Batch::with_capacity(self.schema.len());
        for row in upstream.into_rows().into_iter().take(to_take_usize) {
            if !out.push_row(row) {
                return Err(ExecutionError::Eval(
                    "LimitOp: batch overflow during sized push".into(),
                ));
            }
        }
        self.remaining = self.remaining.saturating_sub(to_take_usize as u64);
        if self.remaining == 0 {
            self.done = true;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use arcgraph_core::{LabelId, Lsn, NodeId, PartitionId, TenantId};

    use super::*;
    use crate::executor::ops::ScanOp;
    use crate::executor::substrate::StubExecutorSubstrate;
    use crate::executor::value::NodeView;

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

    // -------------------------------------------------------------
    // Boundary cases (12 unit tests target — 4 here cover Limit)
    // -------------------------------------------------------------

    #[test]
    fn limit_zero_emits_no_rows() {
        // LIMIT 0 boundary: emit zero rows, close upstream immediately.
        let s = make_n_persons(5);
        let mut op = LimitOp::new(PhysicalOperator::Scan(person_scan()), 0);
        let ctx = ctx();
        let b = op.next_batch(&ctx, &s).unwrap();
        assert!(b.is_empty(), "LIMIT 0 emits zero rows");
    }

    #[test]
    fn limit_n_equals_upstream_emits_all_rows() {
        // LIMIT N == upstream count: emit all rows then EOS.
        let s = make_n_persons(3);
        let mut op = LimitOp::new(PhysicalOperator::Scan(person_scan()), 3);
        let ctx = ctx();
        let b1 = op.next_batch(&ctx, &s).unwrap();
        assert_eq!(b1.row_count(), 3);
        let b2 = op.next_batch(&ctx, &s).unwrap();
        assert!(b2.is_empty());
    }

    #[test]
    fn limit_n_greater_than_upstream_emits_all_then_eos() {
        // LIMIT N > upstream count: forward all upstream rows + EOS;
        // does NOT emit a partial-NULL batch.
        let s = make_n_persons(2);
        let mut op = LimitOp::new(PhysicalOperator::Scan(person_scan()), 100);
        let ctx = ctx();
        let b1 = op.next_batch(&ctx, &s).unwrap();
        assert_eq!(b1.row_count(), 2);
        let b2 = op.next_batch(&ctx, &s).unwrap();
        assert!(b2.is_empty(), "EOS after upstream exhausted");
    }

    #[test]
    fn limit_n_less_than_upstream_truncates_within_batch() {
        // LIMIT 2 with 5 upstream rows: emit 2, then EOS (without
        // pulling another batch).
        let s = make_n_persons(5);
        let mut op = LimitOp::new(PhysicalOperator::Scan(person_scan()), 2);
        let ctx = ctx();
        let b1 = op.next_batch(&ctx, &s).unwrap();
        assert_eq!(b1.row_count(), 2);
        let b2 = op.next_batch(&ctx, &s).unwrap();
        assert!(b2.is_empty(), "EOS after limit hit");
    }

    #[test]
    fn limit_propagates_cancel() {
        let s = make_n_persons(5);
        let ctx = ctx();
        ctx.cancellation().cancel();
        let mut op = LimitOp::new(PhysicalOperator::Scan(person_scan()), 3);
        let r = op.next_batch(&ctx, &s);
        assert_eq!(r, Err(ExecutionError::Cancelled));
    }

    #[test]
    fn limit_straddles_batch_boundary_correctly() {
        // Substrate has BATCH_ROWS + 5 rows; LIMIT BATCH_ROWS + 3.
        // First batch returns BATCH_ROWS rows; second returns 3 rows;
        // third is EOS.
        use crate::executor::BATCH_ROWS;
        let total = BATCH_ROWS + 5;
        let s = make_n_persons(total as u64);
        let limit = (BATCH_ROWS + 3) as u64;
        let mut op = LimitOp::new(PhysicalOperator::Scan(person_scan()), limit);
        let ctx = ctx();
        let b1 = op.next_batch(&ctx, &s).unwrap();
        assert_eq!(b1.row_count(), BATCH_ROWS);
        let b2 = op.next_batch(&ctx, &s).unwrap();
        assert_eq!(b2.row_count(), 3);
        let b3 = op.next_batch(&ctx, &s).unwrap();
        assert!(b3.is_empty());
    }
}
