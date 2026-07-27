//! [`DeleteOp`] — write-op operator for `DELETE var (, var)*` and
//! `DETACH DELETE var (, var)*` per ADR-149 (W26-θ Phase 3).
//!
//! Lowers from [`crate::logical_plan::LogicalDelete`]. The operator
//! holds an upstream sub-pipeline (`input_op`) producing one row per
//! MATCH-bound trigger; per row it tombstones each item's resolved
//! id via the substrate's `delete_node` or `delete_rel` (per the
//! item's `DeleteKind` discriminator).
//!
//! # Terminal drain-to-EOS (the #717 fix)
//!
//! DELETE is a **terminal** write clause at Phase 3 — it is the
//! pipeline root and never has a write-op consumer above it (a
//! `DELETE n SET n.x` chain is nonsensical and the lowering wires
//! DELETE's `input` to the *prior* MATCH/CREATE, never to another
//! write-op — cf. [`crate::executor::Pipeline::build`]'s
//! `mark_writeop_input_stacked`, which only flips SET/REMOVE, not
//! DELETE). Unlike the SET/REMOVE ops (which compose when stacked per
//! #709), DELETE has only the terminal shape, so the operator
//! DRAINS its upstream fully:
//!
//! 1. On the first `next_batch`, repeatedly pull batches from
//!    `input_op` and tombstone every row's items until the upstream
//!    is genuinely exhausted (an empty batch). For each row, for each
//!    item: resolve the row's cell at the item's `binding` schema slot
//!    to a `NodeId` / `RelId`; dispatch to
//!    `substrate.delete_node(detach=...)` or `substrate.delete_rel(...)`
//!    per `DeleteKind`.
//! 2. Once the upstream is exhausted, transition to EOS and return an
//!    EMPTY output batch — DELETE emits **0 rows** to the driver
//!    (openCypher v9 §6 RETURN-less terminal-write contract; same as
//!    terminal SET/REMOVE per ADR-149/150 §D + ADR-182). Subsequent
//!    `next_batch` calls return empty.
//!
//! The internal drain is load-bearing: the materialize loop
//! (`crate::executor::execute_with_context` / `crate::materialize`)
//! breaks on the FIRST empty batch, so an op that returned empty after
//! only the first upstream batch would silently SKIP the deletes for
//! every later batch (any match set > [`crate::executor::BATCH_ROWS`]
//! rows). Pre-#717, `next_batch` returned `Batch::empty(0)` after
//! processing a single batch, so a `DELETE` matching > BATCH_ROWS rows
//! only tombstoned the FIRST page — a silent partial-delete
//! data-corruption (#717). Draining to real EOS in one call deletes
//! ALL matched rows across ALL pages while still emitting 0 rows.
//! Cancellation is re-checked per drain iteration so a long delete
//! stays interruptible.
//!
//! # Schema
//!
//! The output schema is EMPTY at Phase 3. RETURN-after-DELETE
//! (openCypher v9 §6's `DELETE n RETURN n` shape) is forward-pinned
//! to Phase 4+ per ADR-149 §"Forward-deferred".
//!
//! # ADR provenance
//! - **ADR-149** — primary spec (W26-θ Phase 3).
//! - **ADR-147** §D-7 — production-substrate convention (per-tenant
//!   Transaction; default trait impl returns `IndexUnavailable`).
//! - **ADR-031** + **ADR-033** — per-tenant `Transaction` discipline
//!   (commit + rollback).
//! - **ADR-018** — MVCC tombstone semantics for `delete_node` /
//!   `delete_rel`.

use arcgraph_core::{NodeId, RelId};

use crate::executor::batch::Batch;
use crate::executor::context::ExecutionContext;
use crate::executor::error::ExecutionError;
use crate::executor::ops::{PhysicalOperator, schema_index};
use crate::executor::substrate::ExecutorSubstrate;
use crate::executor::value::Value;
use crate::logical_plan::DeleteKind;
use crate::semantic::bound_ast::BindingId;

/// One bound DELETE item — the item's binding identifier + its
/// Node-vs-Rel substrate-dispatch discriminator, captured at
/// pipeline-build time.
#[derive(Debug, Clone)]
pub struct DeleteItemSpec {
    /// Schema slot the item refers to in the upstream's per-row
    /// layout.
    pub binding: BindingId,
    /// Substrate dispatch discriminator.
    pub kind: DeleteKind,
}

/// DELETE / DETACH DELETE executor op (ADR-149 W26-θ Phase 3).
#[derive(Debug)]
pub struct DeleteOp {
    /// Upstream sub-pipeline producing the MATCH-bound rows.
    input_op: Box<PhysicalOperator>,
    /// Per-item Bound DELETE specs (in source order).
    items: Vec<DeleteItemSpec>,
    /// `true` for `DETACH DELETE ...` — drives the substrate's
    /// cascade-rel-tombstone behavior for Node-typed items.
    detach: bool,
    /// Cached output schema — empty at Phase 3 per ADR-149 §D-9.
    schema: Vec<BindingId>,
    /// EOS flag — set after the terminal drain completes (the upstream
    /// returned an empty batch and every matched row has been
    /// tombstoned). Once set, subsequent `next_batch` calls return an
    /// empty batch.
    eos: bool,
}

impl DeleteOp {
    /// Construct a fresh [`DeleteOp`] from a
    /// [`crate::logical_plan::LogicalDelete`].
    #[must_use]
    pub fn new(input_op: PhysicalOperator, items: Vec<DeleteItemSpec>, detach: bool) -> Self {
        Self {
            input_op: Box::new(input_op),
            items,
            detach,
            schema: Vec::new(),
            eos: false,
        }
    }

    /// Output schema — empty at Phase 3.
    pub fn schema(&self) -> &[BindingId] {
        &self.schema
    }

    /// Tombstone every item of each row of `batch` via the substrate.
    /// Per-call-per-transaction discipline lives inside the substrate
    /// (each call opens + commits a per-tenant transaction); the
    /// executor's per-row narrowing is intentional per ADR-149 §D-8.
    fn apply_batch<S: ExecutorSubstrate>(
        &self,
        ctx: &ExecutionContext,
        substrate: &S,
        batch: &Batch,
        upstream_schema: &[BindingId],
    ) -> Result<(), ExecutionError> {
        for row_idx in 0..batch.row_count() {
            let row = batch.row(row_idx);
            for item in &self.items {
                let idx = schema_index(upstream_schema, item.binding).ok_or_else(|| {
                    ExecutionError::Eval(format!(
                        "DeleteOp: item binding {:?} not in upstream schema {:?}",
                        item.binding, upstream_schema
                    ))
                })?;
                let cell = row.get(idx).ok_or_else(|| {
                    ExecutionError::Eval(format!("DeleteOp: row missing cell at index {idx}"))
                })?;
                match item.kind {
                    DeleteKind::Node => {
                        let node_id = node_id_from_value(cell)?;
                        substrate
                            .delete_node(ctx.tenant(), node_id, self.detach, ctx)
                            .map_err(ExecutionError::Substrate)?;
                    }
                    DeleteKind::Rel => {
                        let rel_id = rel_id_from_value(cell)?;
                        substrate
                            .delete_rel(ctx.tenant(), rel_id, ctx)
                            .map_err(ExecutionError::Substrate)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Pull the next batch — DRAIN the upstream fully, tombstoning every
    /// matched row across ALL pages, then return an empty batch.
    ///
    /// DELETE is terminal at Phase 3 (no downstream rows), so it emits
    /// **0 rows** to the driver. The drain is internal because the
    /// materialize loop breaks on the FIRST empty batch — returning
    /// empty after only the first upstream batch would skip the deletes
    /// for every later batch (any match set > [`crate::executor::BATCH_ROWS`]
    /// rows), the #717 silent partial-delete bug. Returns EOS-empty
    /// once the upstream is exhausted; subsequent calls return empty.
    pub fn next_batch<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<Batch, ExecutionError> {
        ctx.cancellation().check()?;
        if self.eos {
            return Ok(Batch::empty(0));
        }

        // Acquire snapshot LSN — defense-in-depth (parallel to
        // CreateNodeOp / CreateRelOp; the outer materialize already
        // holds the LSN guard).
        let _exec_lsn = ctx.ensure_snapshot_lsn();

        // Terminal: drain the WHOLE upstream, tombstoning every matched
        // row, then emit empty (0 result rows). The internal loop is
        // necessary because the driver breaks on the first empty batch —
        // returning empty after only batch 1 would skip deletes for rows
        // in later batches (> BATCH_ROWS matches, the #717 bug).
        // Cancellation is re-checked per iteration so a long delete stays
        // interruptible.
        loop {
            ctx.cancellation().check()?;
            let upstream_batch = self.input_op.next_batch(ctx, substrate)?;
            if upstream_batch.is_empty() {
                // EOS reached; transition our own state and emit empty.
                self.eos = true;
                return Ok(Batch::empty(0));
            }
            let upstream_schema = self.input_op.schema().to_vec();
            self.apply_batch(ctx, substrate, &upstream_batch, &upstream_schema)?;
        }
    }
}

/// Extract the `NodeId` from a `Value::Node` cell. Surfaces a clean
/// `ExecutionError::Eval` otherwise (defense-in-depth — the
/// type-check pass already enforced Node typing on every Node-kind
/// DELETE item per ADR-149 §D-4).
fn node_id_from_value(v: &Value) -> Result<NodeId, ExecutionError> {
    match v {
        Value::Node(n) => Ok(n.id),
        other => Err(ExecutionError::Eval(format!(
            "DeleteOp: expected Node cell, got {other:?}"
        ))),
    }
}

/// Extract the `RelId` from a `Value::Relationship` cell.
fn rel_id_from_value(v: &Value) -> Result<RelId, ExecutionError> {
    match v {
        Value::Relationship(r) => Ok(r.id),
        other => Err(ExecutionError::Eval(format!(
            "DeleteOp: expected Relationship cell, got {other:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use arcgraph_core::{LabelId, NodeId, PartitionId, RelId, TenantId, TypeId};

    use super::*;
    use crate::ast::CreateRelDirection;
    use crate::executor::ops::{CreateNodeOp, CreateRelOp};
    use crate::executor::substrate::StubExecutorSubstrate;
    use crate::executor::value::{NodeView, RelView};
    use crate::logical_plan::LogicalCreateEndpoint;

    fn mk_create_node(var: BindingId, label: &str) -> PhysicalOperator {
        PhysicalOperator::CreateNode(CreateNodeOp::new(
            Some(var),
            Some(label.to_string()),
            Vec::new(),
        ))
    }

    #[test]
    fn delete_op_drains_upstream_and_emits_zero_rows() {
        // A trivial DELETE shape: upstream emits one row with a
        // CREATE-introduced Node value; the terminal DELETE DRAINS the
        // upstream (the #717 drain-to-EOS contract) — applying the
        // tombstone but emitting 0 rows to the driver — and we observe
        // scan_nodes no longer sees the deleted node after.
        let tenant = TenantId::DEFAULT;
        let label = LabelId::new(1024);
        let pre = NodeView::new(NodeId::new(1), Some(label));
        let s = StubExecutorSubstrate::new().with_node(tenant, pre.clone());
        let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);

        // Build a single-row upstream via CreateNodeOp (it emits one
        // row binding the new node-id; we then DELETE that binding).
        let create = mk_create_node(BindingId::new(0), "User");
        let items = vec![DeleteItemSpec {
            binding: BindingId::new(0),
            kind: DeleteKind::Node,
        }];
        let mut op = DeleteOp::new(create, items, false);
        // The first call drains the WHOLE upstream (the single CREATE
        // row + its EOS) and emits 0 rows — a terminal DELETE yields no
        // result rows (openCypher / ADR-149/150 §D / ADR-182). Pre-#717
        // this returned empty after the first batch WITHOUT draining the
        // rest; the drain now happens in one call.
        let b1 = op.next_batch(&ctx, &s).expect("first batch OK");
        assert!(
            b1.is_empty(),
            "terminal DELETE drains its rows and emits 0 rows, got {} row(s)",
            b1.row_count()
        );
        let b2 = op.next_batch(&ctx, &s).expect("second batch OK");
        assert!(b2.is_empty(), "second pull settles into EOS");
        // The CREATE-d node was DELETEd; pre-baked `pre` still exists.
        let nodes = s
            .scan_nodes(tenant, None, arcgraph_core::Lsn::MAX)
            .expect("scan_nodes OK");
        assert_eq!(
            nodes.len(),
            1,
            "pre-baked node remains; CREATE-d node tombstoned"
        );
        assert_eq!(nodes[0].node.id, pre.id);
    }

    #[test]
    fn delete_op_drains_all_pages_across_multiple_batches() {
        // **#717 regression (focused operator-level unit test).** A
        // terminal DELETE over an upstream that emits MORE THAN ONE
        // batch must tombstone EVERY row across ALL pages — not just the
        // first BATCH_ROWS. We drive a hand-built multi-batch upstream
        // (a `MultiBatchStubScan` emitting > BATCH_ROWS pre-baked Node
        // rows) directly into the DeleteOp.
        //
        // Pre-#717, `next_batch` returned `Batch::empty(0)` after the
        // FIRST non-empty batch, so only the first BATCH_ROWS rows were
        // tombstoned — rows on later pages were silently NOT deleted
        // (silent partial-delete data-corruption). The drain-to-EOS fix
        // tombstones all of them.
        use crate::executor::batch::BATCH_ROWS;

        let tenant = TenantId::DEFAULT;
        let label = LabelId::new(1024);
        // BATCH_ROWS * 2 + 7 = 4103 nodes → 3 upstream batches
        // (2048 + 2048 + 7). Mirrors materialize/scan multi-batch pins.
        let total = (BATCH_ROWS * 2 + 7) as u64;
        let mut s = StubExecutorSubstrate::new();
        for i in 1..=total {
            s = s.with_node(tenant, NodeView::new(NodeId::new(i), Some(label)));
        }
        let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);

        // A scan over the pre-baked nodes emits them in BATCH_ROWS-sized
        // pages — the canonical multi-batch upstream. ScanOp reads
        // scan_nodes ONCE at first-batch + paginates the cached vec, so
        // mid-drain tombstones do not perturb the iteration set.
        let scan = PhysicalOperator::Scan(crate::executor::ops::ScanOp::new(
            BindingId::new(0),
            Some(label),
            arcgraph_core::Lsn::MAX,
        ));
        let items = vec![DeleteItemSpec {
            binding: BindingId::new(0),
            kind: DeleteKind::Node,
        }];
        let mut op = DeleteOp::new(scan, items, false);

        // Drive the op the way the materialize loop does: pull until the
        // first empty batch. The terminal DELETE drains internally, so
        // the very first call tombstones ALL pages and returns empty.
        let mut emitted_rows = 0usize;
        loop {
            let b = op.next_batch(&ctx, &s).expect("batch OK");
            if b.is_empty() {
                break;
            }
            emitted_rows += b.row_count();
        }
        assert_eq!(
            emitted_rows, 0,
            "terminal DELETE emits 0 rows to the driver (openCypher contract)"
        );

        // The load-bearing assertion: EVERY matched node across ALL
        // pages is tombstoned, not just the first BATCH_ROWS. Pre-fix
        // this would observe `total - BATCH_ROWS` (= 2055) survivors.
        let remaining = s
            .scan_nodes(tenant, Some(label), arcgraph_core::Lsn::MAX)
            .expect("scan_nodes OK");
        assert_eq!(
            remaining.len(),
            0,
            "ALL {total} matched nodes deleted across all pages; \
             {} survived (pre-#717 partial-delete leaves the > BATCH_ROWS tail)",
            remaining.len()
        );
    }

    #[test]
    fn delete_op_pre_cancellation_short_circuits() {
        let s = StubExecutorSubstrate::new();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        ctx.cancellation().cancel();
        let create = mk_create_node(BindingId::new(0), "User");
        let items = vec![DeleteItemSpec {
            binding: BindingId::new(0),
            kind: DeleteKind::Node,
        }];
        let mut op = DeleteOp::new(create, items, false);
        let r = op.next_batch(&ctx, &s);
        assert_eq!(r, Err(ExecutionError::Cancelled));
    }

    #[test]
    fn delete_op_rel_kind_extracts_rel_id_from_relationship_value() {
        // Smoke: rel-kind dispatch reads Value::Relationship cell.
        // We wire a CreateRelOp upstream, then DELETE the binding.
        let tenant = TenantId::DEFAULT;
        let s = StubExecutorSubstrate::new();
        let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);

        let source = mk_create_node(BindingId::new(0), "User");
        let target = mk_create_node(BindingId::new(1), "User");
        let rel_op = CreateRelOp::new(
            Some(BindingId::new(2)),
            "KNOWS".into(),
            Vec::new(),
            source,
            BindingId::new(0),
            LogicalCreateEndpoint::Fresh,
            target,
            BindingId::new(1),
            LogicalCreateEndpoint::Fresh,
            CreateRelDirection::LeftToRight,
        );

        let items = vec![DeleteItemSpec {
            binding: BindingId::new(2),
            kind: DeleteKind::Rel,
        }];
        let mut op = DeleteOp::new(PhysicalOperator::CreateRel(rel_op), items, false);
        let b1 = op.next_batch(&ctx, &s).expect("first batch OK");
        assert!(b1.is_empty());
        // The rel was tombstoned at the stub layer; subsequent expand
        // from src ↔ dst is 0 (the stub's delete_rel removes from the
        // adjacency in the create_state).
        let _ = NodeView::new(NodeId::new(1), Some(LabelId::new(0)));
        let _ = RelView::new(
            RelId::new(1),
            NodeId::new(1),
            NodeId::new(2),
            Some(TypeId::new(1)),
        );
    }

    #[test]
    fn delete_op_node_kind_rejects_non_node_cell() {
        // Defense-in-depth: a malformed upstream cell surfaces a
        // clean ExecutionError::Eval; the type-check pass already
        // gates against this at lower-time per ADR-149 §D-4.
        let s = StubExecutorSubstrate::new();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        // Use CreateRelOp to emit a Value::Relationship row, then ask
        // DeleteOp to interpret it as a Node-kind — should surface Eval.
        let source = mk_create_node(BindingId::new(0), "User");
        let target = mk_create_node(BindingId::new(1), "User");
        let rel_op = CreateRelOp::new(
            Some(BindingId::new(2)),
            "KNOWS".into(),
            Vec::new(),
            source,
            BindingId::new(0),
            LogicalCreateEndpoint::Fresh,
            target,
            BindingId::new(1),
            LogicalCreateEndpoint::Fresh,
            CreateRelDirection::LeftToRight,
        );
        let items = vec![DeleteItemSpec {
            binding: BindingId::new(2),
            kind: DeleteKind::Node, // MISMATCH on purpose
        }];
        let mut op = DeleteOp::new(PhysicalOperator::CreateRel(rel_op), items, false);
        let r = op.next_batch(&ctx, &s);
        assert!(matches!(r, Err(ExecutionError::Eval(_))));
    }
}
