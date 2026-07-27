//! [`ScanOp`] — sequential node-scan operator (M4-61).
//!
//! Lowers from [`crate::logical_plan::LogicalScan`]. Reads
//! [`crate::executor::ExecutorSubstrate::scan_nodes`] once at
//! first-batch time + paginates the result vec out in
//! [`crate::executor::BATCH_ROWS`]-sized chunks. Population happens
//! lazily so the snapshot LSN is acquired AT first-batch (per
//! ADR-038 §2 D-18 rule 1: execute-time, pre-first-batch) — not at
//! construction time.
//!
//! # Schema
//!
//! Output schema is `[var]` — a single binding (the node-pattern
//! variable). Each row is `[Value::Node(NodeView)]`.
//!
//! # ADR provenance
//! - ADR-038 §2 D-24 — `LogicalScan` operator contract.
//! - ADR-038 amendment-02 §M4.f — M4-61 simple-operator slice scope.
//! - ADR-041 §D-4 — MVCC visibility key threaded via
//!   [`crate::logical_plan::LogicalScan::read_lsn`].

use arcgraph_core::{LabelId, Lsn};

use crate::executor::batch::{BATCH_ROWS, Batch};
use crate::executor::context::ExecutionContext;
use crate::executor::error::ExecutionError;
use crate::executor::substrate::{BoundNode, ExecutorSubstrate};
use crate::executor::value::Value;
use crate::semantic::bound_ast::BindingId;

/// Sequential node-scan operator.
#[derive(Debug)]
pub struct ScanOp {
    /// Variable bound by this scan. Mirrored in `schema[0]` for
    /// per-batch lookup; the field is preserved for diagnostic +
    /// future M4-71 row-count-observer attribution.
    #[allow(dead_code)]
    binding: BindingId,
    /// Optional label filter.
    label: Option<LabelId>,
    /// MVCC read LSN copied from
    /// [`crate::logical_plan::LogicalScan::read_lsn`]. v1.0-alpha
    /// stub substrates ignore this; production wiring threads it
    /// through CRUD scans at M4-08+.
    plan_read_lsn: Lsn,
    /// Cached per-batch schema (length-1: just the binding).
    schema: Vec<BindingId>,
    /// v2 M2 (design §M2.3) — plan-time-derived property projection.
    /// `Some(names)` ⇒ the whole plan provably consumes ONLY these
    /// properties of the scanned variable (see
    /// `executor::projection::scan_projection_for_chain`), so the
    /// substrate may materialize just those key_ids. `None` ⇒ full-bag
    /// scan (the safe default).
    projection: Option<Vec<String>>,
    /// Buffered scan result. `None` until first-batch primes it.
    buffer: Option<Vec<BoundNode>>,
    /// Cursor into the buffer.
    cursor: usize,
}

impl ScanOp {
    /// Construct a fresh `ScanOp` from a [`crate::logical_plan::LogicalScan`].
    #[must_use]
    pub fn new(binding: BindingId, label: Option<LabelId>, plan_read_lsn: Lsn) -> Self {
        Self {
            binding,
            label,
            plan_read_lsn,
            schema: vec![binding],
            projection: None,
            buffer: None,
            cursor: 0,
        }
    }

    /// v2 M2 (design §M2.3) — attach a plan-time property projection.
    /// The pipeline calls this ONLY when
    /// `executor::projection::scan_projection_for_chain` proved the
    /// whole plan consumes nothing beyond `names` from this scan's
    /// variable (a restricted bag is observation-equivalent).
    #[must_use]
    pub fn with_projection(mut self, names: Vec<String>) -> Self {
        self.projection = Some(names);
        self
    }

    /// Output schema. Always `[binding]`.
    pub fn schema(&self) -> &[BindingId] {
        &self.schema
    }

    /// Pull the next batch.
    pub fn next_batch<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<Batch, ExecutionError> {
        // Defense-in-depth cancel check inside the operator (in
        // addition to the dispatcher's check).
        ctx.cancellation().check()?;

        // Lazy buffer prime — at FIRST batch, acquire the snapshot
        // LSN per ADR-038 §2 D-18 rule 1 + populate the buffer from
        // the substrate.
        if self.buffer.is_none() {
            let _exec_lsn = ctx.ensure_snapshot_lsn();
            // The plan-side `read_lsn` is the canonical read key; the
            // exec-side `_exec_lsn` is the snapshot of the executor's
            // ambient context. v1.0-alpha they're both `Lsn::MAX`;
            // production wiring binds the real LSN at M4-08+.
            let nodes = match &self.projection {
                // v2 M2 (design §M2.3): the projected scan materializes
                // only the plan-consumed properties (zero-decode on the
                // typed substrate); the default impl over-fetches,
                // which is always correct.
                Some(names) => substrate.scan_nodes_projected_with_context(
                    ctx,
                    self.label,
                    self.plan_read_lsn,
                    names,
                )?,
                None => substrate.scan_nodes_with_context(ctx, self.label, self.plan_read_lsn)?,
            };
            self.buffer = Some(nodes);
        }
        let buf = self.buffer.as_ref().expect("primed above");
        if self.cursor >= buf.len() {
            // EOS sentinel.
            return Ok(Batch::empty(self.schema.len()));
        }
        let mut batch = Batch::with_capacity(self.schema.len());
        let take = (buf.len() - self.cursor).min(BATCH_ROWS);
        for node in &buf[self.cursor..self.cursor + take] {
            // Defensive: a Batch::push_row that fails would mean a
            // capacity mismatch — we sized `take` to fit so this
            // never trips, but we surface it loudly if it does.
            if !batch.push_row(vec![Value::Node(node.node.clone())]) {
                return Err(ExecutionError::Eval(
                    "ScanOp: batch overflow during sized push".into(),
                ));
            }
        }
        self.cursor += take;
        Ok(batch)
    }
}

/// Zero-schema source with two modes: the **EOS sentinel** (0 rows —
/// [`EmptyOp::new`]) and the **unit driving row** (1 zero-column row —
/// [`EmptyOp::unit`]). Both lower from [`crate::logical_plan::LogicalEmpty`];
/// which mode the pipeline builds depends on the role the
/// `LogicalEmpty` plays.
///
/// # Two roles of `LogicalEmpty` (#618)
///
/// `LogicalEmpty` is overloaded in the lowering, and the two roles need
/// opposite row counts:
///
/// 1. **Leading-clause driving table** ([`EmptyOp::unit`]). The leaf the
///    lowering wires under a *leading* clause with no preceding MATCH —
///    `RETURN 1`, `WITH [1, 2, 3] AS l ...`, `UNWIND [1, 2, 3] AS x ...`.
///    openCypher's relational model starts every query from a single
///    "unit" row (one row, zero columns); a MATCH (a [`ScanOp`]) is what
///    multiplies or zeroes that driving row. A leading projection /
///    unwind must run against exactly ONE driving row, or it produces
///    zero output rows (`RETURN 1` → ∅, `UNWIND [1, 2, 3] AS x` → ∅) —
///    wrong. The [`crate::executor::Pipeline`] builds the generic
///    `LogicalEmpty` arm as [`EmptyOp::unit`] to realize the contract the
///    lowering already documents (`lower_create`: "a `LogicalEmpty` whose
///    `next_batch` returns exactly one empty row (the trigger)") + the
///    openCypher v9 §6.7 / §6 leading-clause semantics. [`UnwindOp`](super::UnwindOp)
///    (#618) is the first consumer whose conformance tests make the
///    prior silent-∅ behaviour observable end-to-end.
///
/// 2. **Provably-empty MERGE match-branch** ([`EmptyOp::new`]). When
///    `MERGE (n:UninternedLabel)` lowers its match-branch to
///    `LogicalEmpty` (ADR-151 §lower_merge_node_scan case 2 — no live
///    node can carry an uninterned label, so the probe is an O(1) EOS,
///    NOT an O(node_high_water) scan-and-discard), [`MergeOp`](super::MergeOp)
///    pulls it to exhaustion → empty → fires the create-branch. This
///    role REQUIRES 0 rows; the Merge build arm constructs it as
///    [`EmptyOp::new`] explicitly so the generic unit-row default does
///    not leak in (which would make MERGE believe it matched).
///
/// Self-seeding write-op sources ([`super::CreateNodeOp`] etc.) do NOT
/// depend on either mode — the pipeline builds them as leaves and they
/// emit their own trigger row.
#[derive(Debug)]
pub struct EmptyOp {
    schema: Vec<BindingId>,
    /// `true` → emit one zero-column unit row before EOS (leading-clause
    /// driving table); `false` → straight EOS (sentinel / MERGE probe).
    emit_unit_row: bool,
    /// Set after the unit row (if any) is emitted; subsequent calls EOS.
    emitted: bool,
}

impl EmptyOp {
    /// Construct an EOS-sentinel `EmptyOp` — emits **zero** rows. The
    /// MERGE provably-empty match-branch probe + the degenerate
    /// empty-result sentinel. Backwards-compatible: this is the
    /// pre-#618 behaviour, unchanged.
    #[must_use]
    pub fn new() -> Self {
        Self {
            schema: Vec::new(),
            emit_unit_row: false,
            emitted: false,
        }
    }

    /// Construct a unit-driving-row `EmptyOp` — emits **one** zero-column
    /// row then EOS. The openCypher driving table for a leading no-MATCH
    /// clause (see the type-level docs, role 1; #618).
    #[must_use]
    pub fn unit() -> Self {
        Self {
            schema: Vec::new(),
            emit_unit_row: true,
            emitted: false,
        }
    }

    /// Output schema. Empty.
    pub fn schema(&self) -> &[BindingId] {
        &self.schema
    }

    /// Emit the unit row on first call (unit mode only), then EOS.
    ///
    /// W11Z fix-up LOW-5 (PR #268 retro): every operator's `next_batch`
    /// MUST call `ctx.cancellation().check()?` per ADR-038 amendment-02
    /// §M4.f — even the structurally trivial `EmptyOp`. Defense-in-
    /// depth: a future code path that constructs `EmptyOp` directly
    /// (bypassing `PhysicalOperator::next_batch`'s dispatcher-side
    /// check at `ops/mod.rs`) would otherwise miss cancellation. The
    /// cancel check fires BEFORE the unit row is emitted, so a tripped
    /// token returns `Cancelled` rather than the driving row.
    pub fn next_batch<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        _substrate: &S,
    ) -> Result<Batch, ExecutionError> {
        ctx.cancellation().check()?;
        if !self.emit_unit_row || self.emitted {
            return Ok(Batch::empty(self.schema.len()));
        }
        self.emitted = true;
        // The single zero-column "unit" row (openCypher driving table's
        // empty tuple) that seeds a leading no-MATCH clause. `schema` is
        // empty, so the row carries zero cells.
        let mut batch = Batch::with_capacity(self.schema.len());
        let pushed = batch.push_row(Vec::new());
        debug_assert!(pushed, "unit row must fit a fresh batch");
        Ok(batch)
    }
}

impl Default for EmptyOp {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use arcgraph_core::{NodeId, PartitionId, TenantId};

    use super::*;
    use crate::executor::substrate::StubExecutorSubstrate;
    use crate::executor::value::NodeView;

    fn fixture_substrate() -> StubExecutorSubstrate {
        let mut s = StubExecutorSubstrate::new();
        for i in 1..=5_u64 {
            s = s.with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(i), Some(LabelId::new(1))),
            );
        }
        s
    }

    #[test]
    fn scan_emits_all_rows_then_eos() {
        let s = fixture_substrate();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let mut op = ScanOp::new(BindingId::new(0), Some(LabelId::new(1)), Lsn::MAX);
        let b1 = op.next_batch(&ctx, &s).unwrap();
        assert_eq!(b1.row_count(), 5);
        let b2 = op.next_batch(&ctx, &s).unwrap();
        assert!(b2.is_empty(), "second batch is EOS");
    }

    #[test]
    fn scan_acquires_snapshot_lsn_at_first_batch() {
        // ADR-038 §2 D-18 rule 1 pin: the LSN must be acquired at
        // first batch, not at construction.
        let s = fixture_substrate();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let mut op = ScanOp::new(BindingId::new(0), None, Lsn::MAX);
        assert_eq!(ctx.snapshot_lsn(), None, "pre-first-batch: not acquired");
        let _ = op.next_batch(&ctx, &s).unwrap();
        assert!(
            ctx.snapshot_lsn().is_some(),
            "post-first-batch: LSN acquired"
        );
    }

    #[test]
    fn scan_pre_cancellation_skips_substrate_call() {
        let s = fixture_substrate();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        ctx.cancellation().cancel();
        let mut op = ScanOp::new(BindingId::new(0), None, Lsn::MAX);
        let r = op.next_batch(&ctx, &s);
        assert_eq!(r, Err(ExecutionError::Cancelled));
    }

    #[test]
    fn scan_paginates_when_buffer_exceeds_batch_rows() {
        // Build a substrate with 2*BATCH_ROWS + 7 rows; verify three
        // batches are produced (size 2048, 2048, 7) then EOS.
        let extra = 7;
        let total = BATCH_ROWS * 2 + extra;
        let mut s = StubExecutorSubstrate::new();
        for i in 1..=total as u64 {
            s = s.with_node(TenantId::DEFAULT, NodeView::new(NodeId::new(i), None));
        }
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let mut op = ScanOp::new(BindingId::new(0), None, Lsn::MAX);
        let b1 = op.next_batch(&ctx, &s).unwrap();
        let b2 = op.next_batch(&ctx, &s).unwrap();
        let b3 = op.next_batch(&ctx, &s).unwrap();
        let b4 = op.next_batch(&ctx, &s).unwrap();
        assert_eq!(b1.row_count(), BATCH_ROWS);
        assert_eq!(b2.row_count(), BATCH_ROWS);
        assert_eq!(b3.row_count(), extra);
        assert!(b4.is_empty());
    }

    #[test]
    fn empty_op_new_always_returns_eos() {
        // EOS-sentinel mode (the MERGE provably-empty probe + degenerate
        // sentinel): zero rows, unchanged pre-#618 behaviour.
        let s = StubExecutorSubstrate::new();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let mut op = EmptyOp::new();
        let b = op.next_batch(&ctx, &s).unwrap();
        assert!(b.is_empty(), "EmptyOp::new() emits zero rows");
        // Idempotent EOS.
        assert!(op.next_batch(&ctx, &s).unwrap().is_empty());
    }

    #[test]
    fn empty_op_unit_emits_one_unit_row_then_eos() {
        // #618 unit-driver mode: the leading-clause driving table is a
        // SINGLE zero-column unit row, NOT zero rows. The first batch
        // carries exactly one 0-cell row; the second is the EOS sentinel.
        let s = StubExecutorSubstrate::new();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let mut op = EmptyOp::unit();
        let b1 = op.next_batch(&ctx, &s).unwrap();
        assert_eq!(b1.row_count(), 1, "one unit driving row");
        assert_eq!(b1.column_count(), 0, "zero-column unit tuple");
        assert!(b1.row(0).is_empty(), "the unit row carries no cells");
        let b2 = op.next_batch(&ctx, &s).unwrap();
        assert!(b2.is_empty(), "EOS after the single unit row");
    }

    /// W11Z fix-up LOW-5 (PR #268 retro): `EmptyOp::next_batch` MUST
    /// honor the cancellation token even though it's structurally
    /// trivial — defense-in-depth per amendment-02 §M4.f's "every
    /// operator's next_batch checks at batch boundary" mandate.
    #[test]
    fn empty_op_pre_cancellation_returns_cancelled() {
        let s = StubExecutorSubstrate::new();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        ctx.cancellation().cancel();
        let mut op = EmptyOp::new();
        let r = op.next_batch(&ctx, &s);
        assert_eq!(r, Err(ExecutionError::Cancelled));
    }
}
