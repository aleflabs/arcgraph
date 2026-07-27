//! [`MergeJoinOp`] — sort-merge equi-join executor (W25-M4-61b / ADR-097).
//!
//! Lowers from [`crate::logical_plan::LogicalJoin`] when the
//! [`crate::planner::pick_join_algorithms`] cost-based picker selects
//! [`crate::logical_plan::JoinAlgorithm::MergeJoin`]. Implements a
//! buffered sort-merge join: drains BOTH children eagerly into per-row
//! buffers, sorts each buffer by the shared-key fingerprint, then
//! walks the two cursors emitting one joined row per matching pair.
//!
//! # Why sort-merge alongside hash
//!
//! The W17α hash-join is the natural choice when one side is small
//! enough for the BUILD bucket map to fit in memory. Sort-merge is
//! strictly better when:
//!
//! - **Both sides are already sorted** on the join key (e.g., a
//!   forward-deferred M4-72 + storage-side sort-property annotation
//!   reveals "this Scan's rows are emitted in NodeId order because
//!   the underlying CRUD layer iterates the primary index"). The
//!   sort prefix collapses to zero and the merge cost is sub-tuple
//!   per row — strictly cheaper than hash-join's hash-table-touch.
//! - **The hash-side build would exceed the per-tenant memory budget**.
//!   Sort-merge needs only one full buffer per side (no bucket
//!   overhead); the per-tenant byte-budget reservation tracks each
//!   row's `estimate_row_bytes` linearly.
//! - **The merge result must preserve sort order downstream** (a
//!   downstream `SortOp` over the same key collapses to a no-op).
//!   v1.0-α does NOT yet propagate sort-property annotations to
//!   downstream operators; the merge output's natural sortedness is
//!   an additive forward optimization slot (M4-72).
//!
//! Per ADR-097 §"Algorithm picker policy", the v1.0-α picker compares
//! [`crate::planner::cost::operator::cost_hash_join`] vs
//! [`crate::planner::cost::operator::cost_merge_join`] and emits
//! whichever is cheaper. Cartesian (`SharedBindings([])`) ALWAYS picks
//! hash — merge-join is undefined without a comparison key.
//!
//! # Schema
//!
//! Output schema = `left_schema ++ right_fresh_bindings`. Identical to
//! the hash-join sibling (closes the cross-substrate equivalence
//! contract — same input plan + same substrate yields IDENTICAL row
//! sets modulo emission order).
//!
//! # Equality semantics — Cypher 3VL
//!
//! Reuses [`super::join::join_key_fingerprint`] for the key comparator.
//! Per ADR-006 amendment-01 + ADR-038 §2 D-20: NULL ≠ NULL (rows with
//! NULL in any shared key column are suppressed); NaN ≠ NaN (rows with
//! NaN float keys are suppressed). The fingerprint helper returns
//! `None` for these cases.
//!
//! # Memory budget (M4-64a integration)
//!
//! Each side's buffer reserves bytes against
//! [`crate::executor::MemoryBudget`] per row inserted. When the
//! per-tenant cap is configured, exceeding it surfaces
//! [`crate::semantic::error::ArcQLError::ResourceExhausted`]
//! routed via [`ExecutionError::Plan`]. When no cap is configured
//! (uncapped budget = no memory limit), each side's row count is
//! bounded only by the actual cardinality, guarded against a true
//! runaway by [`super::expand::UNCAPPED_RUNAWAY_GUARD_ROWS`] (#980
//! lifted the old `SPILLOVER_MAX_ROWS` 131 072-row valve that broke
//! legitimate large joins; matching the sibling join / spillover paths).
//!
//! # Match-emission spillover
//!
//! A single fingerprint may have many rows on both sides (M:N match).
//! When the output batch fills mid-match, the surplus rows queue into
//! `MergeJoinOp::spillover`, which the next `next_batch` call drains
//! FIFO. Same byte-budget discipline as the hash-join sibling.
//!
//! # ADR provenance
//! - **ADR-097** — W25-M4-61b executor JOIN cost-based picker.
//! - **ADR-038 §2 D-24** — `LogicalJoin` lowering surface.
//! - **ADR-038 amendment-02 §M4.f** — executor pull-based batch
//!   discipline.
//! - **ADR-006 amendment-01** — Cypher 3VL equality semantics.

use std::collections::VecDeque;

use crate::executor::batch::Batch;
use crate::executor::budget::estimate_row_bytes;
use crate::executor::context::ExecutionContext;
use crate::executor::error::ExecutionError;
use crate::executor::ops::PhysicalOperator;
use crate::executor::ops::expand::UNCAPPED_RUNAWAY_GUARD_ROWS;
use crate::executor::ops::join::join_key_fingerprint;
use crate::executor::substrate::ExecutorSubstrate;
use crate::executor::value::Value;
use crate::semantic::bound_ast::BindingId;

/// Sort-merge equi-join executor.
pub struct MergeJoinOp {
    left: Box<PhysicalOperator>,
    right: Box<PhysicalOperator>,
    /// Indices into `left.schema()` for the shared bindings (the
    /// merge-walk's left cursor key).
    left_shared_indices: Vec<usize>,
    /// Indices into `right.schema()` for the shared bindings.
    right_shared_indices: Vec<usize>,
    /// Indices into `right.schema()` for the bindings NOT present in
    /// `left.schema()` (the fresh right-side columns appended per
    /// joined row).
    right_fresh_indices: Vec<usize>,
    /// Output schema = `left_schema ++ right_fresh_bindings`.
    schema: Vec<BindingId>,
    /// Drain + sort state. `None` until the first `next_batch` call
    /// populates it; subsequent calls re-use the sorted buffers.
    state: Option<MergeState>,
    /// Cumulative bytes the buffers reserved against the per-tenant
    /// memory budget. Released on `Drop` / completion-after-EOS via
    /// [`Self::release_buffer_reservation`].
    buffer_reserved_bytes: u64,
    /// Match-emission spillover (overflow from a partially-emitted
    /// fingerprint cluster). Each `SpilledRow` carries its byte
    /// reservation so each release is paired with each pop.
    spillover: VecDeque<SpilledRow>,
}

/// Internal: per-side sorted buffer + merge cursor state.
struct MergeState {
    /// LEFT-side rows, sorted by key fingerprint. Each entry is the
    /// `(fingerprint, row)` pair; rows with `None` fingerprint were
    /// already suppressed during drain.
    left_rows: Vec<(String, Vec<Value>)>,
    /// RIGHT-side rows, sorted by key fingerprint.
    right_rows: Vec<(String, Vec<Value>)>,
    /// Merge cursor: next LEFT row to examine.
    li: usize,
    /// Merge cursor: next RIGHT row to examine.
    ri: usize,
    /// Cursor within the current left × right cluster cross-product.
    /// Stored as `(left_pos, right_pos)` indices into the LEFT cluster
    /// and the RIGHT cluster; the next emit cycles through right_pos
    /// first (inner loop), then advances left_pos (outer loop).
    cluster_emit: Option<ClusterEmit>,
}

/// In-progress cluster cross-product emission state.
#[derive(Debug, Clone, Copy)]
struct ClusterEmit {
    /// LEFT cluster `[left_start, left_end)` indices into
    /// `MergeState::left_rows`.
    left_start: usize,
    left_end: usize,
    /// RIGHT cluster `[right_start, right_end)` indices into
    /// `MergeState::right_rows`.
    right_start: usize,
    right_end: usize,
    /// Inner cross-product cursor — next pair to emit is
    /// `(left_rows[left_start + l_off], right_rows[right_start + r_off])`.
    l_off: usize,
    r_off: usize,
}

/// One spilled output row + its budget reservation. Mirror of the
/// sibling spillover shape in [`super::join::HashJoinOp`].
#[derive(Debug)]
struct SpilledRow {
    row: Vec<Value>,
    /// Bytes reserved against the per-tenant budget for this row.
    /// `0` when no cap was set at push time (the row-count fallback
    /// applied).
    reserved_bytes: u64,
}

impl std::fmt::Debug for MergeJoinOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MergeJoinOp")
            .field("left", &self.left)
            .field("right", &self.right)
            .field("left_shared_indices", &self.left_shared_indices)
            .field("right_shared_indices", &self.right_shared_indices)
            .field("right_fresh_indices", &self.right_fresh_indices)
            .field("schema", &self.schema)
            .field("buffer_reserved_bytes", &self.buffer_reserved_bytes)
            .field("state_initialized", &self.state.is_some())
            .field("spillover_len", &self.spillover.len())
            .finish()
    }
}

impl MergeJoinOp {
    /// Construct a `MergeJoinOp` from left + right children + the
    /// list of shared bindings. The shared list MUST be non-empty —
    /// Cartesian (empty SharedBindings) is structurally invalid for
    /// merge-join. Both inputs MUST produce each shared binding in
    /// their schemas; constructor returns [`ExecutionError::Eval`]
    /// on planner-contract violation.
    pub fn new(
        left: PhysicalOperator,
        right: PhysicalOperator,
        shared: Vec<BindingId>,
    ) -> Result<Self, ExecutionError> {
        if shared.is_empty() {
            return Err(ExecutionError::Eval(
                "MergeJoinOp: empty shared bindings (Cartesian) cannot be \
                 sort-merged — the planner picker MUST route Cartesian joins \
                 to HashJoinOp"
                    .to_owned(),
            ));
        }
        let left_schema: Vec<BindingId> = left.schema().to_vec();
        let right_schema: Vec<BindingId> = right.schema().to_vec();

        let mut left_shared_indices: Vec<usize> = Vec::with_capacity(shared.len());
        let mut right_shared_indices: Vec<usize> = Vec::with_capacity(shared.len());
        for b in &shared {
            let li = left_schema
                .iter()
                .position(|s| s == b)
                .ok_or_else(|| missing_binding_err("left", *b))?;
            let ri = right_schema
                .iter()
                .position(|s| s == b)
                .ok_or_else(|| missing_binding_err("right", *b))?;
            left_shared_indices.push(li);
            right_shared_indices.push(ri);
        }

        let right_fresh_indices: Vec<usize> = right_schema
            .iter()
            .enumerate()
            .filter(|(_, b)| !left_schema.contains(b))
            .map(|(i, _)| i)
            .collect();

        let mut schema = left_schema;
        for &idx in &right_fresh_indices {
            schema.push(right_schema[idx]);
        }

        Ok(Self {
            left: Box::new(left),
            right: Box::new(right),
            left_shared_indices,
            right_shared_indices,
            right_fresh_indices,
            schema,
            state: None,
            buffer_reserved_bytes: 0,
            spillover: VecDeque::new(),
        })
    }

    /// Output schema (= `left_schema ++ right_fresh_bindings`).
    #[must_use]
    pub fn schema(&self) -> &[BindingId] {
        &self.schema
    }

    /// Pull the next batch.
    ///
    /// # State machine
    ///
    /// 1. **DRAIN-and-SORT (first call)**: drain LEFT and RIGHT
    ///    children fully, fingerprinting each row's shared columns;
    ///    rows with `None` fingerprint (3VL NULL / NaN) are dropped.
    ///    Sort both buffers stably by fingerprint. Reserves byte
    ///    budget per row; surfaces `ResourceExhausted` on overflow.
    /// 2. **MERGE-WALK**: advance the dual cursor — when
    ///    `left_rows[li].0 < right_rows[ri].0` advance `li`; when
    ///    `>` advance `ri`; when `==` identify the maximal LEFT and
    ///    RIGHT clusters sharing the fingerprint and emit the
    ///    cross-product into the output batch.
    /// 3. **CLUSTER spillover**: when the cross-product overflows
    ///    a single batch, spill the remainder into
    ///    `MergeJoinOp::spillover`.
    /// 4. **EOS**: empty batch when both cursors are past their
    ///    respective ends AND `spillover.is_empty()`.
    pub fn next_batch<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<Batch, ExecutionError> {
        ctx.cancellation().check()?;

        // Lazy DRAIN-and-SORT on the first call.
        if self.state.is_none() {
            self.drain_and_sort(ctx, substrate)?;
        }

        let mut out = Batch::with_capacity(self.schema.len());
        let budget = ctx.budget().clone();
        let has_cap = budget.has_cap(ctx.tenant());

        // Drain spillover first so partially-emitted clusters finish
        // before we advance the merge cursor.
        while !out.is_full() {
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
        if out.is_full() {
            return Ok(out);
        }

        // Drain any in-progress cluster emit (resumed from a prior
        // batch that filled mid-cluster — the `cluster_emit` slot
        // carries the cross-product cursor across calls).
        self.emit_in_progress_cluster(ctx, &budget, has_cap, &mut out)?;
        if out.is_full() {
            return Ok(out);
        }

        // Merge-walk loop: advance dual cursors + emit clusters.
        loop {
            ctx.cancellation().check()?;
            let state = self
                .state
                .as_mut()
                .expect("drain_and_sort populated state before merge-walk");
            if state.li >= state.left_rows.len() || state.ri >= state.right_rows.len() {
                break;
            }
            let lkey = state.left_rows[state.li].0.as_str();
            let rkey = state.right_rows[state.ri].0.as_str();
            match lkey.cmp(rkey) {
                std::cmp::Ordering::Less => {
                    state.li += 1;
                }
                std::cmp::Ordering::Greater => {
                    state.ri += 1;
                }
                std::cmp::Ordering::Equal => {
                    // Identify the maximal LEFT and RIGHT clusters
                    // sharing this fingerprint.
                    let lkey_owned = lkey.to_owned();
                    let left_start = state.li;
                    let mut left_end = state.li + 1;
                    while left_end < state.left_rows.len()
                        && state.left_rows[left_end].0 == lkey_owned
                    {
                        left_end += 1;
                    }
                    let right_start = state.ri;
                    let mut right_end = state.ri + 1;
                    while right_end < state.right_rows.len()
                        && state.right_rows[right_end].0 == lkey_owned
                    {
                        right_end += 1;
                    }
                    // Set cluster_emit cursor at the start of the
                    // cross-product; advance the dual cursor past
                    // the cluster so the next iteration looks past
                    // it.
                    state.cluster_emit = Some(ClusterEmit {
                        left_start,
                        left_end,
                        right_start,
                        right_end,
                        l_off: 0,
                        r_off: 0,
                    });
                    state.li = left_end;
                    state.ri = right_end;
                    self.emit_in_progress_cluster(ctx, &budget, has_cap, &mut out)?;
                    if out.is_full() {
                        return Ok(out);
                    }
                }
            }
        }

        // EOS path: release buffer reservation.
        if out.is_empty() {
            self.release_buffer_reservation(ctx, &budget);
        }
        Ok(out)
    }

    /// Drain both children fully into per-side buffers, sort by key
    /// fingerprint. Rows with `None` fingerprint (NULL / NaN keys) are
    /// dropped per Cypher 3VL.
    fn drain_and_sort<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<(), ExecutionError> {
        let budget = ctx.budget().clone();
        let has_cap = budget.has_cap(ctx.tenant());

        let left_rows = drain_into_buffer(
            &mut self.left,
            &self.left_shared_indices,
            ctx,
            substrate,
            &budget,
            has_cap,
            &mut self.buffer_reserved_bytes,
            "MergeJoinOp left-side buffer",
        )?;
        let right_rows = drain_into_buffer(
            &mut self.right,
            &self.right_shared_indices,
            ctx,
            substrate,
            &budget,
            has_cap,
            &mut self.buffer_reserved_bytes,
            "MergeJoinOp right-side buffer",
        )?;

        // Stable sort by fingerprint — preserves insertion order
        // within each cluster for determinism.
        let mut left_rows = left_rows;
        let mut right_rows = right_rows;
        left_rows.sort_by(|a, b| a.0.cmp(&b.0));
        right_rows.sort_by(|a, b| a.0.cmp(&b.0));

        self.state = Some(MergeState {
            left_rows,
            right_rows,
            li: 0,
            ri: 0,
            cluster_emit: None,
        });
        Ok(())
    }

    /// Emit pairs from any in-progress cluster cross-product into
    /// `out`. When `out` fills before the cluster is exhausted, the
    /// remaining cross-product pairs spill into `self.spillover` so
    /// the next `next_batch` call resumes the cluster before advancing
    /// the dual cursor.
    fn emit_in_progress_cluster(
        &mut self,
        ctx: &ExecutionContext,
        budget: &crate::executor::MemoryBudget,
        has_cap: bool,
        out: &mut Batch,
    ) -> Result<(), ExecutionError> {
        // Take ownership of the cursor so we can advance it without
        // borrow conflicts with `self.spillover` mutations.
        let state = self
            .state
            .as_mut()
            .expect("emit_in_progress_cluster called before state initialized");
        let mut emit = match state.cluster_emit.take() {
            Some(e) => e,
            None => return Ok(()),
        };
        // Cross-product iteration: outer = LEFT, inner = RIGHT.
        while emit.left_start + emit.l_off < emit.left_end {
            let left_row = &state.left_rows[emit.left_start + emit.l_off].1;
            while emit.right_start + emit.r_off < emit.right_end {
                let right_row = &state.right_rows[emit.right_start + emit.r_off].1;
                let mut joined: Vec<Value> = left_row.clone();
                for &idx in &self.right_fresh_indices {
                    joined.push(right_row[idx].clone());
                }
                if !out.push_row(joined.clone()) {
                    // Output batch full — spill the row into the
                    // spillover queue + advance the emit cursor so
                    // resumption picks up the NEXT pair.
                    let reserved_bytes = if has_cap {
                        let bytes = estimate_row_bytes(&joined) as u64;
                        budget.try_reserve_unscoped(
                            ctx.tenant(),
                            bytes,
                            "MergeJoinOp cluster-emit spillover",
                        )?;
                        bytes
                    } else {
                        if self.spillover.len() >= UNCAPPED_RUNAWAY_GUARD_ROWS {
                            return Err(spillover_fallback_err(self.spillover.len()));
                        }
                        0
                    };
                    self.spillover.push_back(SpilledRow {
                        row: joined,
                        reserved_bytes,
                    });
                }
                emit.r_off += 1;
                if out.is_full() {
                    // Persist the cursor so the next call resumes.
                    state.cluster_emit = Some(emit);
                    return Ok(());
                }
            }
            // Advance outer cursor; reset inner.
            emit.l_off += 1;
            emit.r_off = 0;
        }
        Ok(())
    }

    /// Release the buffer-side byte reservation if any. Called at
    /// EOS (clean exit) and on drop (defensive — in case the
    /// operator is destroyed mid-stream).
    fn release_buffer_reservation(
        &mut self,
        ctx: &ExecutionContext,
        budget: &crate::executor::MemoryBudget,
    ) {
        if self.buffer_reserved_bytes > 0 {
            budget.release(ctx.tenant(), self.buffer_reserved_bytes);
            self.buffer_reserved_bytes = 0;
        }
    }
}

impl Drop for MergeJoinOp {
    fn drop(&mut self) {
        // No body: see [`super::join::HashJoinOp`] Drop rustdoc —
        // session-level release is the canonical path.
    }
}

/// Drain a child operator into a `(fingerprint, row)` buffer,
/// suppressing rows whose fingerprint is `None` (NULL / NaN per Cypher
/// 3VL). Reserves byte budget per row + applies row-count fallback
/// when no cap is configured.
#[allow(clippy::too_many_arguments)]
fn drain_into_buffer<S: ExecutorSubstrate>(
    child: &mut PhysicalOperator,
    shared_indices: &[usize],
    ctx: &ExecutionContext,
    substrate: &S,
    budget: &crate::executor::MemoryBudget,
    has_cap: bool,
    accumulated_bytes: &mut u64,
    label: &'static str,
) -> Result<Vec<(String, Vec<Value>)>, ExecutionError> {
    let mut rows: Vec<(String, Vec<Value>)> = Vec::new();
    loop {
        ctx.cancellation().check()?;
        let batch = child.next_batch(ctx, substrate)?;
        if batch.is_empty() {
            break;
        }
        for row in batch.into_rows() {
            let fingerprint = match join_key_fingerprint(&row, shared_indices) {
                Some(fp) => fp,
                None => continue, // 3VL NULL / NaN suppression.
            };
            if has_cap {
                let row_bytes = estimate_row_bytes(&row) as u64;
                budget.try_reserve_unscoped(ctx.tenant(), row_bytes, label)?;
                *accumulated_bytes = accumulated_bytes.saturating_add(row_bytes);
            } else if rows.len() >= UNCAPPED_RUNAWAY_GUARD_ROWS {
                return Err(buffer_fallback_err(label, rows.len()));
            }
            rows.push((fingerprint, row));
        }
    }
    Ok(rows)
}

/// Build a "shared binding missing from schema" planner-contract
/// violation error.
fn missing_binding_err(side: &str, b: BindingId) -> ExecutionError {
    ExecutionError::Eval(format!(
        "MergeJoinOp: shared binding {b:?} not present in {side} schema (planner-contract violation)"
    ))
}

/// Render the runaway-guard error for the per-side buffer drain (#980 —
/// lifted runaway-protection ceiling, not the old 131 072-row valve).
fn buffer_fallback_err(label: &str, rows: usize) -> ExecutionError {
    ExecutionError::Plan(crate::semantic::error::ArcQLError::ResourceExhausted {
        feature: format!("{label} runaway-guard"),
        requested_bytes: 0,
        cap_bytes: UNCAPPED_RUNAWAY_GUARD_ROWS as u64,
        projected_bytes: rows as u64,
        span: crate::error::Span::point(0, 0),
    })
}

/// Render the runaway-guard error for cluster-emit spillover (#980).
fn spillover_fallback_err(rows: usize) -> ExecutionError {
    ExecutionError::Plan(crate::semantic::error::ArcQLError::ResourceExhausted {
        feature: "MergeJoinOp cluster-emit spillover runaway-guard".to_owned(),
        requested_bytes: 0,
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

    /// Three persons + two edges Alice→Bob, Alice→Carol. Mirror of
    /// the sibling HashJoinOp fixture for cross-operator equivalence.
    fn fixture() -> StubExecutorSubstrate {
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
            .with_edge(
                TenantId::DEFAULT,
                RelView::new(
                    RelId::new(11),
                    NodeId::new(1),
                    NodeId::new(3),
                    Some(TypeId::new(1)),
                ),
            )
    }

    #[test]
    fn merge_equi_join_matches_on_shared_binding_like_hash() {
        // SAME query shape as the sibling HashJoin test:
        // LEFT = MATCH (a:Person) → 3 rows.
        // RIGHT = MATCH (a:Person)-[r:KNOWS]->(b) → 2 rows.
        // Shared = [a] → equi-join: Alice's LEFT row matches both
        // right rows; Bob + Carol have no right matches.
        let s = fixture();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let a = BindingId::new(0);
        let r = BindingId::new(1);
        let b = BindingId::new(2);

        let left = PhysicalOperator::Scan(ScanOp::new(a, Some(LabelId::new(1)), Lsn::MAX));
        let right_scan = PhysicalOperator::Scan(ScanOp::new(a, Some(LabelId::new(1)), Lsn::MAX));
        let right_exp = ExpandOp::new(
            right_scan,
            a,
            Some(r),
            b,
            Some(TypeId::new(1)),
            Direction::LeftToRight,
            None,
            Lsn::MAX,
        )
        .expect("expand build");
        let right = PhysicalOperator::Expand(right_exp);

        let mut op = MergeJoinOp::new(left, right, vec![a]).expect("merge join construction");
        assert_eq!(op.schema(), &[a, r, b]);
        let out = op.next_batch(&ctx, &s).expect("join batch");
        assert_eq!(out.row_count(), 2);
        for row in out.rows() {
            assert!(matches!(&row[0], Value::Node(n) if n.id == NodeId::new(1)));
            assert!(matches!(&row[1], Value::Relationship(_)));
            match &row[2] {
                Value::Node(n) => assert!(n.id == NodeId::new(2) || n.id == NodeId::new(3)),
                other => panic!("unexpected b-row: {other:?}"),
            }
        }
        let eos = op.next_batch(&ctx, &s).expect("eos");
        assert!(eos.is_empty());
    }

    #[test]
    fn merge_rejects_cartesian_at_construction() {
        // MergeJoin cannot run without a join key — Cartesian routes
        // to HashJoinOp at the planner picker layer. Construction
        // surfaces an Eval error.
        let a = BindingId::new(0);
        let b = BindingId::new(1);
        let left = PhysicalOperator::Scan(ScanOp::new(a, None, Lsn::MAX));
        let right = PhysicalOperator::Scan(ScanOp::new(b, None, Lsn::MAX));
        let err = MergeJoinOp::new(left, right, Vec::new()).expect_err("must reject Cartesian");
        match err {
            ExecutionError::Eval(msg) => {
                assert!(msg.contains("Cartesian"), "msg = {msg}");
            }
            other => panic!("expected Eval error, got {other:?}"),
        }
    }

    #[test]
    fn merge_empty_left_yields_empty_output() {
        let s = StubExecutorSubstrate::new();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let a = BindingId::new(0);
        let left = PhysicalOperator::Scan(ScanOp::new(a, None, Lsn::MAX));
        let right = PhysicalOperator::Scan(ScanOp::new(a, None, Lsn::MAX));
        let mut op = MergeJoinOp::new(left, right, vec![a]).expect("ok");
        let out = op.next_batch(&ctx, &s).expect("ok");
        assert!(out.is_empty());
    }

    #[test]
    fn merge_empty_right_yields_empty_output() {
        let s = fixture();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let a = BindingId::new(0);
        let left = PhysicalOperator::Scan(ScanOp::new(a, Some(LabelId::new(1)), Lsn::MAX));
        // No nodes labeled 99 → empty right.
        let right = PhysicalOperator::Scan(ScanOp::new(a, Some(LabelId::new(99)), Lsn::MAX));
        let mut op = MergeJoinOp::new(left, right, vec![a]).expect("ok");
        let out = op.next_batch(&ctx, &s).expect("ok");
        assert!(out.is_empty());
    }

    #[test]
    fn merge_missing_shared_binding_is_planner_contract_violation() {
        let a = BindingId::new(0);
        let b = BindingId::new(1);
        let bogus = BindingId::new(99);
        let left = PhysicalOperator::Scan(ScanOp::new(a, None, Lsn::MAX));
        let right = PhysicalOperator::Scan(ScanOp::new(b, None, Lsn::MAX));
        let err = MergeJoinOp::new(left, right, vec![bogus])
            .expect_err("contract violation must surface");
        assert!(matches!(err, ExecutionError::Eval(_)));
    }

    #[test]
    fn merge_cancel_during_drain_short_circuits() {
        let s = fixture();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        ctx.cancellation().cancel();
        let a = BindingId::new(0);
        let left = PhysicalOperator::Scan(ScanOp::new(a, None, Lsn::MAX));
        let right = PhysicalOperator::Scan(ScanOp::new(a, None, Lsn::MAX));
        let mut op = MergeJoinOp::new(left, right, vec![a]).expect("ok");
        let r = op.next_batch(&ctx, &s);
        assert_eq!(r, Err(ExecutionError::Cancelled));
    }

    #[test]
    fn merge_three_pattern_chain_matches_hash() {
        // Build a fixture where Alice -[r1]-> Bob -[r2]-> Carol.
        // Two patterns: (a)-[r1]->(b) AND (b)-[r2]->(c) joined on b.
        let s = StubExecutorSubstrate::new()
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
                    RelId::new(100),
                    NodeId::new(1),
                    NodeId::new(2),
                    Some(TypeId::new(1)),
                ),
            )
            .with_edge(
                TenantId::DEFAULT,
                RelView::new(
                    RelId::new(101),
                    NodeId::new(2),
                    NodeId::new(3),
                    Some(TypeId::new(1)),
                ),
            );

        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let a = BindingId::new(0);
        let r1 = BindingId::new(1);
        let b = BindingId::new(2);
        let r2 = BindingId::new(3);
        let c = BindingId::new(4);

        let scan_a = PhysicalOperator::Scan(ScanOp::new(a, Some(LabelId::new(1)), Lsn::MAX));
        let left_pattern = PhysicalOperator::Expand(
            ExpandOp::new(
                scan_a,
                a,
                Some(r1),
                b,
                Some(TypeId::new(1)),
                Direction::LeftToRight,
                None,
                Lsn::MAX,
            )
            .expect("left expand"),
        );
        let scan_b = PhysicalOperator::Scan(ScanOp::new(b, Some(LabelId::new(1)), Lsn::MAX));
        let right_pattern = PhysicalOperator::Expand(
            ExpandOp::new(
                scan_b,
                b,
                Some(r2),
                c,
                Some(TypeId::new(1)),
                Direction::LeftToRight,
                None,
                Lsn::MAX,
            )
            .expect("right expand"),
        );
        let mut op = MergeJoinOp::new(left_pattern, right_pattern, vec![b]).expect("ok");
        assert_eq!(op.schema(), &[a, r1, b, r2, c]);
        let out = op.next_batch(&ctx, &s).expect("batch");
        assert_eq!(out.row_count(), 1);
        let row = &out.rows()[0];
        let a_id = match &row[0] {
            Value::Node(n) => n.id,
            _ => panic!(),
        };
        let b_id = match &row[2] {
            Value::Node(n) => n.id,
            _ => panic!(),
        };
        let c_id = match &row[4] {
            Value::Node(n) => n.id,
            _ => panic!(),
        };
        assert_eq!(a_id, NodeId::new(1));
        assert_eq!(b_id, NodeId::new(2));
        assert_eq!(c_id, NodeId::new(3));
    }
}
