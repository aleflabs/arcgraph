//! [`OptionalExpandOp`] — OPTIONAL MATCH null-row emission (M4-62).
//!
//! Lowers from [`crate::logical_plan::LogicalLeftOuterJoin`] (which
//! itself lowers from `OPTIONAL MATCH` per ADR-006 amendment-01 §A-2).
//! For each LEFT row, evaluates the RIGHT-side sub-pipeline; if it
//! produces no rows, EMITS the LEFT row extended with NULL columns
//! for every fresh RIGHT-side binding.
//!
//! # Why this is the M4-62 OPTIONAL MATCH op
//!
//! Per amendment-03 §TIER-1 GAP D the executor-side OPTIONAL MATCH
//! contract is "for any LEFT row whose RIGHT match is empty, emit
//! a row carrying NULL for every fresh RIGHT-introduced binding".
//! That's exactly the left-outer-join semantics the M4-32 lowering
//! produces, so the executor lights it as a single op (named
//! `OptionalExpandOp` per the spawn brief, but the underlying
//! semantics is left-outer-join).
//!
//! # Schema
//!
//! Output schema = `left_schema ++ right_fresh_bindings_in_order`.
//! "Fresh" = binding NOT in `left_schema`. Re-references of pre-
//! existing bindings (e.g., the LEFT-side `(a)` reused by the RIGHT
//! `(a)-[:R]-(b)` pattern) stay non-nullable per the M4-22b
//! `may_be_null` propagation rule (the binding-time flag is the
//! source of truth; this operator just emits the row shape).
//!
//! # ADR provenance
//! - **ADR-006 amendment-01 §A-2** — OPTIONAL MATCH lowers to
//!   left-outer-join.
//! - **ADR-038 amendment-03 §TIER-1 GAP D** — execution-time
//!   null-row emission contract.
//! - **ADR-038 §2 D-21 / M4-22b** — `may_be_null` propagation rule
//!   (binding-time, not exec-time).

use std::collections::VecDeque;

use arcgraph_core::{Lsn, NodeId};

use crate::executor::batch::Batch;
use crate::executor::budget::estimate_row_bytes;
use crate::executor::context::ExecutionContext;
use crate::executor::error::ExecutionError;
use crate::executor::ops::PhysicalOperator;
use crate::executor::ops::expand::UNCAPPED_RUNAWAY_GUARD_ROWS;
use crate::executor::ops::expand_spill::{ExpandSpillQueue, ExpandSpillTarget};
use crate::executor::substrate::ExecutorSubstrate;
use crate::executor::value::Value;
use crate::semantic::bound_ast::BindingId;

/// OPTIONAL MATCH operator (left-outer-join).
///
/// The `right` operator MUST be re-runnable (i.e., its `next_batch`
/// can be driven from a fresh state for each LEFT row). v1.0-alpha
/// achieves this by accepting a `right_factory` closure that builds a
/// fresh `right` operator per LEFT row.
///
/// # Test-friendly construction
///
/// The factory pattern lets tests build a `right` operator that
/// already encodes the substrate-driven expansion (e.g., a
/// `ScanOp(Person) → ExpandOp(KNOWS) → ProjectOp(...)` mini-tree)
/// per the LEFT row. Production wiring at M4-08+ will re-shape this
/// once the M4-32 lowering's right-side sub-pipeline can be
/// parameterized — for now, the closure is the indirection.
/// Right-factory closure type: rebuilds the right-side sub-pipeline
/// given the LEFT row's columns. Pulled out as a `type` alias so
/// `clippy::type_complexity` doesn't fire on the field declaration.
pub type RightSidePipelineFactory = Box<dyn Fn(&[Value]) -> PhysicalOperator + Send + Sync>;

pub struct OptionalExpandOp {
    left: Box<PhysicalOperator>,
    /// Builder for the right-side sub-pipeline. Called once per LEFT
    /// row with the LEFT row's columns (so the right side can
    /// pre-bind the join column).
    right_factory: RightSidePipelineFactory,
    /// Schema of the right-side sub-pipeline output (typically the
    /// `[r, b]` pair for `OPTIONAL MATCH (a)-[r]-(b)`).
    right_schema: Vec<BindingId>,
    /// Indices in `right_schema` that are FRESH (i.e., not in
    /// `left_schema`). These are the columns appended to each LEFT
    /// row in the output; for empty-right-match rows, these are
    /// filled with NULL.
    right_fresh_indices: Vec<usize>,
    /// Cached output schema = `left_schema ++ right_fresh_bindings`.
    schema: Vec<BindingId>,
    /// Shared-binding equality predicates resolved as
    /// `(left_row_index, right_row_index)`. These preserve the
    /// left-outer-join correlation when the OPTIONAL pattern reuses
    /// more than the source endpoint, e.g. `(a)-[r]->(c)` with both
    /// `a` and `c` already bound.
    join_pairs: Vec<(usize, usize)>,
    /// Spillover from a partially-consumed LEFT row's many right-
    /// side outputs.
    ///
    /// W11Z fix-up MED-3 (PR #268 retro): switched to `VecDeque`
    /// (O(1) `pop_front`); same shape as
    /// [`super::expand::ExpandOp::spillover`]. W12α / M4-64a: each row
    /// carries its budget reservation byte count alongside the row;
    /// bound by per-tenant [`crate::executor::MemoryBudget`] when
    /// configured, falling back to the
    /// [`UNCAPPED_RUNAWAY_GUARD_ROWS`] runaway guard otherwise (#980).
    spillover: VecDeque<SpilledRow>,
    /// OOC-4 symmetry with ExpandOp; absent on the legacy path.
    spill_queue: Option<Box<ExpandSpillQueue>>,
    terminal_error: Option<ExecutionError>,
    /// Have we observed an EOS batch from the left?
    left_done: bool,
}

/// One spilled row + its budget reservation. Mirrors
/// [`super::expand::SpilledRow`] (kept module-local to honor the
/// 7-slice 3-strike rule — no shared trait until a third consumer
/// arrives).
#[derive(Debug)]
struct SpilledRow {
    row: Vec<Value>,
    reserved_bytes: u64,
}

impl std::fmt::Debug for OptionalExpandOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OptionalExpandOp")
            .field("left", &self.left)
            .field("right_schema", &self.right_schema)
            .field("right_fresh_indices", &self.right_fresh_indices)
            .field("schema", &self.schema)
            .field("join_pairs", &self.join_pairs)
            .finish()
    }
}

impl OptionalExpandOp {
    /// Construct an `OptionalExpandOp`.
    ///
    /// `right_schema` is the schema of the operator returned by
    /// `right_factory`; the constructor auto-derives the fresh-binding
    /// indices by comparing against `left.schema()`.
    pub fn new<F>(left: PhysicalOperator, right_schema: Vec<BindingId>, right_factory: F) -> Self
    where
        F: Fn(&[Value]) -> PhysicalOperator + Send + Sync + 'static,
    {
        Self::new_with_join_pairs(left, right_schema, Vec::new(), right_factory)
    }

    /// Construct an `OptionalExpandOp` with resolved shared-binding
    /// join predicates.
    pub fn new_with_join_pairs<F>(
        left: PhysicalOperator,
        right_schema: Vec<BindingId>,
        join_pairs: Vec<(usize, usize)>,
        right_factory: F,
    ) -> Self
    where
        F: Fn(&[Value]) -> PhysicalOperator + Send + Sync + 'static,
    {
        let left_schema: Vec<BindingId> = left.schema().to_vec();
        let right_fresh_indices: Vec<usize> = right_schema
            .iter()
            .enumerate()
            .filter(|(_, b)| !left_schema.contains(b))
            .map(|(i, _)| i)
            .collect();
        let mut schema = left_schema.clone();
        for &idx in &right_fresh_indices {
            schema.push(right_schema[idx]);
        }
        Self {
            left: Box::new(left),
            right_factory: Box::new(right_factory),
            right_schema,
            right_fresh_indices,
            schema,
            join_pairs,
            spillover: VecDeque::new(),
            spill_queue: None,
            terminal_error: None,
            left_done: false,
        }
    }

    /// Output schema (= left ++ right-fresh).
    pub fn schema(&self) -> &[BindingId] {
        &self.schema
    }

    /// Enable the OOC-4 FIFO spillover queue for OPTIONAL MATCH output.
    pub fn with_spillover_target(
        mut self,
        target: Option<ExpandSpillTarget>,
    ) -> Result<Self, ExecutionError> {
        self.spill_queue = target.map(|target| Box::new(ExpandSpillQueue::new(target)));
        Ok(self)
    }

    /// Pull the next batch.
    ///
    /// # Spillover bound (W12α / M4-64a integration)
    ///
    /// An OPTIONAL MATCH whose right-side sub-pipeline produces many
    /// rows per LEFT row overflows into the spillover. The spillover
    /// is bounded by:
    ///
    /// 1. **OOC-4 target:** a configured [`crate::executor::MemoryBudget`]
    ///    cap is mandatory. The resident FIFO prefix is charged to that cap
    ///    and overflow is streamed through OOC-1 runs.
    /// 2. **Legacy capped path:** without an OOC target, each overflow row
    ///    reserves [`estimate_row_bytes`]; exceeding the cap surfaces
    ///    [`crate::semantic::error::ArcQLError::ResourceExhausted`] via
    ///    [`ExecutionError::Plan`].
    /// 3. **Runaway-protection guard** ([`UNCAPPED_RUNAWAY_GUARD_ROWS`],
    ///    #980) when no per-tenant cap is configured (v1.0-alpha
    ///    default). Exceeding
    ///    surfaces
    ///    [`crate::semantic::error::ArcQLError::ResourceExhausted`]
    ///    (W12α fix-up LOW-4 promoted from `ExecutionError::Eval`).
    pub fn next_batch<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<Batch, ExecutionError> {
        if let Some(error) = &self.terminal_error {
            return Err(error.clone());
        }
        let result = (|| {
            if let Some(queue) = self.spill_queue.as_mut() {
                queue.prepare(ctx)?;
            }
            self.next_batch_inner(ctx, substrate)
        })();
        match result {
            Ok(batch) => {
                if batch.is_empty() {
                    self.spill_queue = None;
                }
                Ok(batch)
            }
            Err(error) => {
                if self.spill_queue.is_some() {
                    self.spill_queue = None;
                    self.terminal_error = Some(error.clone());
                }
                Err(error)
            }
        }
    }

    fn next_batch_inner<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<Batch, ExecutionError> {
        ctx.cancellation().check()?;
        let mut out = Batch::with_capacity(self.schema.len());
        let budget = ctx.budget().clone();
        let has_cap = budget.has_cap(ctx.tenant());

        // Drain spillover first. Each pop releases its reservation.
        while !out.is_full() {
            if let Some(queue) = self.spill_queue.as_mut() {
                match queue.pop()? {
                    Some(row) => {
                        if !out.push_row(row) {
                            return Err(ExecutionError::Eval(
                                "OptionalExpandOp: OOC-4 drain overflow despite fullness guard"
                                    .into(),
                            ));
                        }
                    }
                    None => break,
                }
            } else {
                match self.spillover.pop_front() {
                    Some(spilled) => {
                        if spilled.reserved_bytes > 0 {
                            budget.release(ctx.tenant(), spilled.reserved_bytes);
                        }
                        let _ = out.push_row(spilled.row);
                    }
                    None => break,
                }
            }
        }
        if out.is_full() {
            return Ok(out);
        }

        loop {
            ctx.cancellation().check()?;
            if self.left_done && self.spillover_is_empty() {
                break;
            }
            if !has_cap && self.spillover_len() >= UNCAPPED_RUNAWAY_GUARD_ROWS {
                return Err(spillover_fallback_err(self.spillover_len()));
            }
            if !self.left_done {
                let left_batch = self.left.next_batch(ctx, substrate)?;
                if left_batch.is_empty() {
                    self.left_done = true;
                    continue;
                }
                for left_row in left_batch.into_rows() {
                    let mut right = (self.right_factory)(&left_row);
                    let mut emitted_any_right = false;
                    loop {
                        ctx.cancellation().check()?;
                        let right_batch = right.next_batch(ctx, substrate)?;
                        if right_batch.is_empty() {
                            break;
                        }
                        for right_row in right_batch.into_rows() {
                            if !self.matches_join_pairs(&left_row, &right_row) {
                                continue;
                            }
                            emitted_any_right = true;
                            let mut joined = left_row.clone();
                            for &idx in &self.right_fresh_indices {
                                joined.push(right_row[idx].clone());
                            }
                            if !out.push_row(joined.clone()) {
                                self.push_spilled(ctx, &budget, has_cap, joined)?;
                            }
                        }
                    }
                    if !emitted_any_right {
                        // OPTIONAL MATCH null-row emission per
                        // amendment-03 §TIER-1 GAP D + ADR-006
                        // amendment-01 §A-2: extend left row with
                        // NULLs for every fresh right-side binding.
                        let mut joined = left_row.clone();
                        for _ in &self.right_fresh_indices {
                            joined.push(Value::Null);
                        }
                        if !out.push_row(joined.clone()) {
                            self.push_spilled(ctx, &budget, has_cap, joined)?;
                        }
                    }
                }
            } else {
                break;
            }
            if out.is_full() {
                break;
            }
        }
        Ok(out)
    }

    fn spillover_is_empty(&self) -> bool {
        self.spill_queue
            .as_ref()
            .map_or_else(|| self.spillover.is_empty(), |queue| queue.is_empty())
    }

    fn spillover_len(&self) -> usize {
        self.spill_queue
            .as_ref()
            .map_or_else(|| self.spillover.len(), |queue| queue.len())
    }

    fn matches_join_pairs(&self, left_row: &[Value], right_row: &[Value]) -> bool {
        self.join_pairs.iter().all(|&(left_idx, right_idx)| {
            match (left_row.get(left_idx), right_row.get(right_idx)) {
                (Some(left), Some(right)) => left == right,
                _ => false,
            }
        })
    }

    /// Push a row to spillover, reserving budget when configured or
    /// applying the row-count fallback when not.
    fn push_spilled(
        &mut self,
        ctx: &ExecutionContext,
        budget: &crate::executor::MemoryBudget,
        has_cap: bool,
        row: Vec<Value>,
    ) -> Result<(), ExecutionError> {
        if let Some(queue) = self.spill_queue.as_mut() {
            return queue.push(ctx, row);
        }
        let reserved_bytes = if has_cap {
            let bytes = estimate_row_bytes(&row) as u64;
            budget.try_reserve_unscoped(ctx.tenant(), bytes, "OptionalExpandOp spillover")?;
            bytes
        } else {
            if self.spillover.len() >= UNCAPPED_RUNAWAY_GUARD_ROWS {
                return Err(spillover_fallback_err(self.spillover.len()));
            }
            0
        };
        self.spillover.push_back(SpilledRow {
            row,
            reserved_bytes,
        });
        Ok(())
    }
}

/// Render the W11Z #272 row-count-fallback error consistently.
///
/// W12α fix-up LOW-4 (PR #277 retro): promoted from
/// [`ExecutionError::Eval`] to
/// [`crate::semantic::error::ArcQLError::ResourceExhausted`] mirroring
/// the [`super::expand::ExpandOp`] surface — see
/// [`super::expand`]'s `spillover_fallback_err` doc for rationale.
fn spillover_fallback_err(rows: usize) -> ExecutionError {
    ExecutionError::Plan(crate::semantic::error::ArcQLError::ResourceExhausted {
        feature: "OptionalExpandOp runaway-guard".to_owned(),
        requested_bytes: 0,
        // #980 — lifted runaway-protection ceiling, not the old valve.
        cap_bytes: UNCAPPED_RUNAWAY_GUARD_ROWS as u64,
        projected_bytes: rows as u64,
        span: crate::error::Span::point(0, 0),
    })
}

#[cfg(test)]
mod tests {
    use arcgraph_core::{LabelId, Lsn, NodeId, PartitionId, RelId, TenantId, TypeId};

    use super::*;
    use crate::executor::ops::{ExpandOp, ScanOp};
    use crate::executor::substrate::StubExecutorSubstrate;
    use crate::executor::value::{NodeView, RelView};
    use crate::logical_plan::Direction;

    fn fixture() -> StubExecutorSubstrate {
        // Two persons. Alice has a KNOWS edge to Bob; Carol has none.
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
    }

    #[test]
    fn optional_expand_emits_null_for_unmatched_left_rows() {
        // LEFT: scan Person → 3 rows (Alice, Bob, Carol).
        // RIGHT: OPTIONAL MATCH (a)-[:KNOWS]->(b) for each `a`.
        // - Alice (a=1) → 1 KNOWS edge to Bob → 1 right row → 1 joined row.
        // - Bob   (a=2) → 0 KNOWS edges → null row.
        // - Carol (a=3) → 0 KNOWS edges → null row.
        // Expected: 3 rows total, 2 with NULL `b`.
        let s = fixture();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let scan_left = ScanOp::new(BindingId::new(0), Some(LabelId::new(1)), Lsn::MAX);
        // The right_factory: for each left row, build a sub-pipeline
        // that scans the LEFT row's `a` and expands KNOWS.
        // v1.0-alpha test trick: encode the LEFT row's `a` as a
        // single-row scan substrate input. Since our stub's scan
        // returns ALL nodes, we instead drive the right side
        // through a custom mini-pipeline rooted at a "synthesized
        // single-row scan" — but that doesn't exist as an op. So
        // for the test we use a one-shot approach: build a Scan
        // (Person) → Expand(KNOWS) → Project(rel, dst); since the
        // ExpandOp pulls neighbors via the substrate keyed on
        // `from_node.id`, we want to filter to the LEFT row's `a`
        // only. The simplest way is to build a LeftRowFilterOp
        // upstream; but that's not in scope. So we use a closure
        // that fabricates a one-row mini-substrate.
        //
        // Alternative: parameterize ExpandOp directly via a
        // single-row "Scan(left.a)" — implemented as a TaggedSourceOp.
        // For test simplicity we use a ScanOp + filter to the LEFT
        // row's id.
        let right_schema = vec![BindingId::new(0), BindingId::new(2), BindingId::new(1)];
        let mut op = OptionalExpandOp::new(
            PhysicalOperator::Scan(scan_left),
            right_schema,
            move |left_row: &[Value]| {
                let from_id = match &left_row[0] {
                    Value::Node(n) => n.id,
                    _ => panic!("expected Node in LEFT row[0]"),
                };
                // SubstrateBackedSingletonScanOp synthesized via a
                // SingletonOp: emits exactly one row carrying the
                // LEFT node, then EOS — which feeds ExpandOp.
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
                .expect("expand build");
                PhysicalOperator::Expand(exp)
            },
        );
        let b = op.next_batch(&ctx, &s).unwrap();
        // 3 left rows; 1 matched, 2 null-extended.
        assert_eq!(b.row_count(), 3);
        let mut null_count = 0;
        let mut matched_count = 0;
        for row in b.rows() {
            // schema is [a, r, b] (left=[a], right_fresh=[r,b]).
            assert_eq!(row.len(), 3);
            // Last column is `b`; if null then it's a null-row.
            let is_null = matches!(row[2], Value::Null) && matches!(row[1], Value::Null);
            if is_null {
                null_count += 1;
            } else {
                matched_count += 1;
                // Matched row's b must be Bob (id=2).
                let b_id = match &row[2] {
                    Value::Node(n) => n.id,
                    _ => panic!(),
                };
                assert_eq!(b_id, NodeId::new(2));
            }
        }
        assert_eq!(matched_count, 1);
        assert_eq!(null_count, 2);
    }

    #[test]
    fn optional_expand_propagates_cancel() {
        let s = fixture();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        ctx.cancellation().cancel();
        let scan_left = ScanOp::new(BindingId::new(0), None, Lsn::MAX);
        let mut op = OptionalExpandOp::new(
            PhysicalOperator::Scan(scan_left),
            vec![BindingId::new(0), BindingId::new(1)],
            |_left| {
                // Unused — cancel trips before the factory runs.
                PhysicalOperator::Empty(crate::executor::ops::EmptyOp::new())
            },
        );
        let r = op.next_batch(&ctx, &s);
        assert_eq!(r, Err(ExecutionError::Cancelled));
    }
}

// =====================================================================
// Singleton scan helper for the OptionalExpand right-side factory.
// =====================================================================
//
// A `SingletonScanOp` emits exactly ONE node (looked up by id from the
// substrate) then EOS. Used by OPTIONAL MATCH right-side factories
// to root a sub-pipeline at the LEFT row's `from` node.
//
// This lives in the optional_expand module (NOT ops/scan.rs) because
// it's a v1.0-alpha bridge that exists ONLY so the OPTIONAL MATCH op
// can express "scan a particular node + expand from it" without
// adding a full predicate-on-scan helper. It will be re-shaped at
// M4-32 / M4-72 forward when the OPTIONAL-MATCH right-side
// sub-pipeline is parameterized at the LogicalPlan level.

/// Single-row scan that emits the substrate-resolved node for the
/// supplied id, then EOS.
#[derive(Debug)]
pub struct SingletonScanOp {
    binding: BindingId,
    target_id: NodeId,
    schema: Vec<BindingId>,
    emitted: bool,
}

impl SingletonScanOp {
    /// Construct a singleton scan keyed on `target_id`.
    #[must_use]
    pub fn new(binding: BindingId, target_id: NodeId) -> Self {
        Self {
            binding,
            target_id,
            schema: vec![binding],
            emitted: false,
        }
    }

    /// Output schema. Always `[binding]`.
    pub fn schema(&self) -> &[BindingId] {
        &self.schema
    }

    /// Pull the single row, then EOS.
    pub fn next_batch<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<Batch, ExecutionError> {
        ctx.cancellation().check()?;
        if self.emitted {
            return Ok(Batch::empty(self.schema.len()));
        }
        let _ = ctx.ensure_snapshot_lsn();
        // Look up the node via a label-free scan + filter (test
        // substrate is small; production wiring will use a
        // direct CRUD lookup).
        let nodes = substrate.scan_nodes(ctx.tenant(), None, Lsn::MAX)?;
        let mut batch = Batch::with_capacity(self.schema.len());
        for n in nodes {
            if n.node.id == self.target_id {
                let _ = batch.push_row(vec![Value::Node(n.node)]);
                break;
            }
        }
        self.emitted = true;
        Ok(batch)
    }
}

// Bind `binding` so the field is read; suppresses unused-field clippy
// hint without making the field public.
impl SingletonScanOp {
    /// The bound variable.
    #[must_use]
    pub fn binding(&self) -> BindingId {
        self.binding
    }
}
