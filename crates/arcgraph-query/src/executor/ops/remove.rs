//! [`RemoveOp`] — write-op operator for `REMOVE <item> (, <item>)*`
//! per ADR-150 (W26-θ Phase 4).
//!
//! Lowers from [`crate::logical_plan::LogicalRemove`]. Symmetric to
//! [`crate::executor::ops::SetOp`] — pulls upstream MATCH-bound rows,
//! dispatches each item's removal (property or label) via the
//! substrate's `remove_node` / `remove_rel`.
//!
//! # Terminal-vs-stacked emission (the #709 fix, R1-narrowed)
//!
//! Identical contract to [`crate::executor::ops::SetOp`]: a REMOVE op is
//! either **stacked** (it has a ROW-CONSUMER above — a write-op for the
//! outer clause of `SET … REMOVE …` / `REMOVE … REMOVE …` (#709), OR a
//! `Project` / `Aggregate` / `Unwind` row-consumer for
//! `REMOVE … RETURN …` / `REMOVE … WITH …` / `REMOVE … RETURN count(a)` /
//! `REMOVE … UNWIND …` (#772)) or **terminal** (the pipeline root / no
//! row-consumer above). The
//! `terminal` flag (set at [`crate::executor::Pipeline::build`] time)
//! selects:
//!
//! - **Stacked** (`terminal == false`): pull ONE upstream batch, apply
//!   each item's removal per row (mirroring the removal onto the row's
//!   entity view so a RETURN/WITH projects the post-REMOVE bag, #772),
//!   PASS THE ROWS THROUGH (output schema = input schema) so a stacked
//!   outer write-op composes (#709) or a `Project` projects the rows
//!   (#772). The empty upstream batch is propagated as EOS.
//! - **Terminal** (`terminal == true`): DRAIN the upstream fully (apply
//!   every batch's removals) then emit an EMPTY batch — a RETURN-less
//!   terminal write yields **0 rows** (openCypher v9 + ADR-149/150 §D +
//!   ADR-182 v1.0-α contract). Draining in one call is required because
//!   the materialize loop breaks on the first empty batch.
//!
//! Pre-#709, REMOVE swallowed its rows and returned `Batch::empty(0)`
//! UNCONDITIONALLY — a *stacked* REMOVE (e.g. the `Remove(Set(Scan))`
//! lowering of `SET n.a = 1 REMOVE n.a`) read the inner op's empty batch
//! as upstream-EOS and never ran, silently dropping the REMOVE and
//! leaving the SET-written value in place (#709, HIGH correctness). The
//! naive pass-through fix composed stacked writes but made a terminal
//! REMOVE emit a row, breaking the openCypher TCK RowSet gate. The
//! `terminal` flag keeps BOTH correct.
//!
//! # Schema
//!
//! The output schema EQUALS the input (upstream) schema — REMOVE binds
//! no new columns. A **stacked** REMOVE re-emits the rows (carrying the
//! post-REMOVE bag, via the in-view mirror) for its row-consumer; a
//! **terminal** REMOVE drains them and emits none. `REMOVE … RETURN …` /
//! `REMOVE … WITH …` lower to `Project(Remove(…))`, and the `Project`
//! build arm flips the REMOVE child to **stacked** (#772); the aggregate
//! forms (`REMOVE … RETURN count(a)` / `… WITH <agg> …`) lower to
//! `Project(Aggregate(Remove(…)))` so the `Aggregate` arm does the flip,
//! and `REMOVE … UNWIND …` lowers to `Unwind(Remove(…))` so the `Unwind`
//! arm does. A bare `REMOVE …` with no row-consumer above it stays
//! terminal → 0 rows (the openCypher v9 / ADR-149/150 §D / ADR-182
//! contract).
//!
//! # ADR provenance
//! - **ADR-150** — primary spec (W26-θ Phase 4).
//! - **ADR-147** §D-7 — production-substrate convention (per-tenant
//!   Transaction; default trait impl returns `IndexUnavailable`).
//! - **ADR-031** + **ADR-033** — per-tenant `Transaction` discipline.
//! - **ADR-018** — MVCC version-chain semantics for `update_node` /
//!   `update_rel`.

use arcgraph_core::{NodeId, RelId};

use crate::executor::batch::Batch;
use crate::executor::context::ExecutionContext;
use crate::executor::error::ExecutionError;
use crate::executor::ops::{PhysicalOperator, schema_index};
use crate::executor::substrate::{ExecutorSubstrate, RemoveNodeMutation, RemoveRelMutation};
use crate::executor::value::{NodeView, RelView, Value};
use crate::logical_plan::{LogicalRemoveMutation, SetTargetKind};
use crate::semantic::bound_ast::BindingId;

/// One bound REMOVE item.
#[derive(Debug, Clone)]
pub struct RemoveItemSpec {
    pub binding: BindingId,
    pub kind: SetTargetKind,
    pub mutation: LogicalRemoveMutation,
}

/// REMOVE executor op (ADR-150 W26-θ Phase 4; #709 fix, R1-narrowed).
#[derive(Debug)]
pub struct RemoveOp {
    input_op: Box<PhysicalOperator>,
    items: Vec<RemoveItemSpec>,
    /// Cached output schema — EQUALS the input schema (REMOVE binds no
    /// new columns). See the module-level terminal-vs-stacked note.
    schema: Vec<BindingId>,
    /// `true` when this REMOVE is the pipeline root / has no row-consumer
    /// above it → it DRAINS the upstream and emits **0 rows**. `false`
    /// when **stacked** under a row-consumer — another write-op (#709) OR
    /// a `Project` / `Aggregate` / `Unwind` (`REMOVE … RETURN …` /
    /// `… WITH …` / `RETURN count(a)` / `REMOVE … UNWIND …`, #772) → it
    /// passes its mutated rows through so the consumer composes / projects /
    /// folds. Set at [`crate::executor::Pipeline::build`] time:
    /// [`Self::new`] defaults terminal; the build flips via
    /// [`Self::mark_stacked`].
    terminal: bool,
    eos: bool,
}

impl RemoveOp {
    /// Construct a fresh **terminal** [`RemoveOp`] from a
    /// [`crate::logical_plan::LogicalRemove`]. Terminal is the default
    /// (RETURN-less root REMOVE → 0 rows);
    /// [`crate::executor::Pipeline::build`] flips a REMOVE wired as a
    /// row-consumer's `input` (another write-op #709, or a `Project` /
    /// `Aggregate` / `Unwind` #772) to stacked via [`Self::mark_stacked`].
    #[must_use]
    pub fn new(input_op: PhysicalOperator, items: Vec<RemoveItemSpec>) -> Self {
        // Output schema = input schema (REMOVE binds no columns), so a
        // stacked outer write-op can resolve its item bindings against
        // the rows this op forwards (the #709 composition fix).
        let schema = input_op.schema().to_vec();
        Self {
            input_op: Box::new(input_op),
            items,
            schema,
            terminal: true,
            eos: false,
        }
    }

    /// Mark this op as **stacked** (pass-through) — it has a row-consumer
    /// above it. Called by [`crate::executor::Pipeline::build`] for a
    /// REMOVE wired as another write-op's `input` (#709) OR as a
    /// `Project` / `Aggregate` / `Unwind` row-consumer's `input` (#772),
    /// and by the stacked-composition unit tests.
    pub fn mark_stacked(&mut self) {
        self.terminal = false;
    }

    /// `true` iff this op is terminal (drains + emits 0 rows). Exposed
    /// for the terminal-vs-stacked row-cardinality pin test.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        self.terminal
    }

    /// Output schema — equals the input schema (REMOVE binds no columns).
    pub fn schema(&self) -> &[BindingId] {
        &self.schema
    }

    /// Apply every item's removal to each row of `batch` via the
    /// substrate. Shared by the stacked + terminal paths.
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
                        "RemoveOp: item binding {:?} not in upstream schema {:?}",
                        item.binding, upstream_schema
                    ))
                })?;
                let cell = row.get(idx).ok_or_else(|| {
                    ExecutionError::Eval(format!("RemoveOp: row missing cell at index {idx}"))
                })?;
                match item.kind {
                    SetTargetKind::Node => {
                        let node_id = node_id_from_value(cell)?;
                        let mutation = build_node_mutation(&item.mutation);
                        substrate
                            .remove_node(ctx.tenant(), node_id, &mutation, ctx)
                            .map_err(ExecutionError::Substrate)?;
                    }
                    SetTargetKind::Rel => {
                        let rel_id = rel_id_from_value(cell)?;
                        let mutation = build_rel_mutation(&item.mutation)?;
                        substrate
                            .remove_rel(ctx.tenant(), rel_id, &mutation, ctx)
                            .map_err(ExecutionError::Substrate)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Apply every item's removal to each row via the substrate **and
    /// mirror** the removal onto the row's in-memory `NodeView` /
    /// `RelView`, so a downstream row-consumer observes the post-REMOVE
    /// bag — `REMOVE a.x RETURN a.x` projects `NULL`, not the stale
    /// pre-REMOVE value (#772). Symmetric to [`super::set::SetOp`]'s
    /// `apply_rows_stacked` (RC-2 per ADR-151-amendment-01 §D-2). Used by
    /// the STACKED path; the terminal path drains its rows (no consumer)
    /// and applies via [`Self::apply_batch`] WITHOUT the mirror (the rows
    /// are discarded, so mirroring would be wasted work; the substrate
    /// removal — the only durable effect — is identical on both paths).
    fn apply_rows_stacked<S: ExecutorSubstrate>(
        &self,
        ctx: &ExecutionContext,
        substrate: &S,
        rows: &mut [Vec<Value>],
        upstream_schema: &[BindingId],
    ) -> Result<(), ExecutionError> {
        for row in rows.iter_mut() {
            for item in &self.items {
                let idx = schema_index(upstream_schema, item.binding).ok_or_else(|| {
                    ExecutionError::Eval(format!(
                        "RemoveOp: item binding {:?} not in upstream schema {:?}",
                        item.binding, upstream_schema
                    ))
                })?;
                let cell = row.get_mut(idx).ok_or_else(|| {
                    ExecutionError::Eval(format!("RemoveOp: row missing cell at index {idx}"))
                })?;
                match item.kind {
                    SetTargetKind::Node => {
                        let node_id = node_id_from_value(cell)?;
                        let mutation = build_node_mutation(&item.mutation);
                        substrate
                            .remove_node(ctx.tenant(), node_id, &mutation, ctx)
                            .map_err(ExecutionError::Substrate)?;
                        // #772 RC-2 — mirror onto the passed-through view.
                        if let Value::Node(view) = cell {
                            apply_node_removal_to_view(view, &mutation);
                        }
                    }
                    SetTargetKind::Rel => {
                        let rel_id = rel_id_from_value(cell)?;
                        let mutation = build_rel_mutation(&item.mutation)?;
                        substrate
                            .remove_rel(ctx.tenant(), rel_id, &mutation, ctx)
                            .map_err(ExecutionError::Substrate)?;
                        if let Value::Relationship(view) = cell {
                            apply_rel_removal_to_view(view, &mutation);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Pull the next batch.
    ///
    /// - **Stacked** (`!terminal`): consume ONE upstream batch, apply
    ///   each item per row (mirroring the removal onto the row's view per
    ///   `Self::apply_rows_stacked`), PASS THE ROWS THROUGH (output
    ///   schema = input schema) so a stacked outer write-op composes
    ///   (#709) or a `Project` / `Aggregate` / `Unwind` row-consumer
    ///   projects / folds / expands the post-REMOVE rows (#772). The empty
    ///   upstream batch is propagated as EOS.
    /// - **Terminal**: DRAIN the upstream fully then emit an EMPTY batch
    ///   — a RETURN-less terminal write yields 0 rows (openCypher /
    ///   ADR-149/150 §D / ADR-182). Draining in one call is required
    ///   because the materialize loop breaks on the first empty batch.
    pub fn next_batch<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<Batch, ExecutionError> {
        ctx.cancellation().check()?;
        if self.eos {
            return Ok(Batch::empty(self.schema.len()));
        }

        let _exec_lsn = ctx.ensure_snapshot_lsn();

        if self.terminal {
            // Terminal: drain the WHOLE upstream, applying removals to
            // every matched row, then emit empty (0 result rows). The
            // internal loop is necessary because the driver breaks on the
            // first empty batch — returning empty after only batch 1 would
            // skip removals for rows in later batches (> BATCH_ROWS
            // matches). Cancellation re-checked per iteration.
            loop {
                ctx.cancellation().check()?;
                let upstream_batch = self.input_op.next_batch(ctx, substrate)?;
                if upstream_batch.is_empty() {
                    self.eos = true;
                    return Ok(Batch::empty(self.schema.len()));
                }
                let upstream_schema = self.input_op.schema().to_vec();
                self.apply_batch(ctx, substrate, &upstream_batch, &upstream_schema)?;
            }
        }

        // Stacked: apply this op's removals to one batch + mirror them onto
        // the rows (#772 — so a `Project`/RETURN/WITH or a stacked outer
        // write-op observes the post-REMOVE bag), then pass the rows
        // through. `from_rows` returns `None` only on `row_count() >
        // BATCH_ROWS` — impossible for a batch we just pulled; the guard
        // is defense-in-depth.
        let upstream_batch = self.input_op.next_batch(ctx, substrate)?;
        if upstream_batch.is_empty() {
            self.eos = true;
            return Ok(Batch::empty(self.schema.len()));
        }
        let upstream_schema = self.input_op.schema().to_vec();
        let mut rows = upstream_batch.into_rows();
        self.apply_rows_stacked(ctx, substrate, &mut rows, &upstream_schema)?;
        Batch::from_rows(rows).ok_or_else(|| {
            ExecutionError::Eval(
                "RemoveOp: pass-through batch exceeded BATCH_ROWS (unreachable — \
                 upstream honoured the row cap)"
                    .into(),
            )
        })
    }
}

fn build_node_mutation(m: &LogicalRemoveMutation) -> RemoveNodeMutation {
    match m {
        LogicalRemoveMutation::Property(name) => RemoveNodeMutation::Property(name.clone()),
        LogicalRemoveMutation::LabelRemove(labels) => {
            RemoveNodeMutation::LabelRemove(labels.clone())
        }
    }
}

fn build_rel_mutation(m: &LogicalRemoveMutation) -> Result<RemoveRelMutation, ExecutionError> {
    match m {
        LogicalRemoveMutation::Property(name) => Ok(RemoveRelMutation::Property(name.clone())),
        LogicalRemoveMutation::LabelRemove(_) => Err(ExecutionError::Eval(
            "RemoveOp: label-remove mutation rejected on Relationship binding (Phase 4 per \
             ADR-150 §D-4; type-check should have rejected this earlier)"
                .into(),
        )),
    }
}

fn node_id_from_value(v: &Value) -> Result<NodeId, ExecutionError> {
    match v {
        Value::Node(n) => Ok(n.id),
        other => Err(ExecutionError::Eval(format!(
            "RemoveOp: expected Node cell, got {other:?}"
        ))),
    }
}

fn rel_id_from_value(v: &Value) -> Result<RelId, ExecutionError> {
    match v {
        Value::Relationship(r) => Ok(r.id),
        other => Err(ExecutionError::Eval(format!(
            "RemoveOp: expected Relationship cell, got {other:?}"
        ))),
    }
}

/// Mirror a [`RemoveNodeMutation`] onto an in-memory [`NodeView`]'s
/// property bag — the post-REMOVE row state for `REMOVE … RETURN …`
/// (#772; RC-2 per ADR-151-amendment-01 §D-2, the removal-side companion
/// to [`super::set::apply_node_mutation_to_view`]). `Property` drops the
/// key; `LabelRemove` has no property-bag effect (the multi-label
/// `NodeView` shape is forward-pinned to v1.1 per ADR-150 §D-9, exactly as
/// SET's `LabelAdd` mirror is a property-bag no-op).
fn apply_node_removal_to_view(node: &mut NodeView, mutation: &RemoveNodeMutation) {
    match mutation {
        RemoveNodeMutation::Property(name) => {
            node.properties.remove(name);
        }
        RemoveNodeMutation::LabelRemove(_) => {}
    }
}

/// Mirror a [`RemoveRelMutation`] onto an in-memory [`RelView`]'s property
/// bag — the rel-side companion to [`apply_node_removal_to_view`] (#772).
fn apply_rel_removal_to_view(rel: &mut RelView, mutation: &RemoveRelMutation) {
    match mutation {
        RemoveRelMutation::Property(name) => {
            rel.properties.remove(name);
        }
    }
}

#[cfg(test)]
mod tests {
    use arcgraph_core::{LabelId, NodeId, PartitionId, TenantId};

    use super::*;
    use crate::ast::Literal;
    use crate::error::Span;
    use crate::executor::ops::{CreateNodeOp, SetItemSpec, SetOp};
    use crate::executor::substrate::StubExecutorSubstrate;
    use crate::executor::value::NodeView;
    use crate::logical_plan::LogicalSetMutation;
    use crate::semantic::bound_ast::BoundExpression;

    fn mk_create_node(var: BindingId, label: &str) -> PhysicalOperator {
        PhysicalOperator::CreateNode(CreateNodeOp::new(
            Some(var),
            Some(label.to_string()),
            Vec::new(),
        ))
    }

    #[test]
    fn terminal_remove_op_applies_then_emits_zero_rows() {
        let tenant = TenantId::DEFAULT;
        let pre = NodeView::new(NodeId::new(1), Some(LabelId::new(1)));
        let s = StubExecutorSubstrate::new().with_node(tenant, pre);
        let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);
        let create = mk_create_node(BindingId::new(0), "User");
        let items = vec![RemoveItemSpec {
            binding: BindingId::new(0),
            kind: SetTargetKind::Node,
            mutation: LogicalRemoveMutation::Property("age".into()),
        }];
        // Terminal (#709 R1-narrowing): REMOVE applies the removal but
        // DRAINS the row — a RETURN-less terminal write yields 0 result
        // rows (openCypher / ADR-149/150 §D / ADR-182).
        let mut op = RemoveOp::new(create, items);
        assert!(op.is_terminal(), "RemoveOp::new defaults to terminal");
        assert_eq!(op.schema(), &[BindingId::new(0)], "schema == input schema");
        let b1 = op.next_batch(&ctx, &s).expect("first batch OK");
        assert!(
            b1.is_empty(),
            "terminal REMOVE drains its rows and emits 0 rows, got {} row(s)",
            b1.row_count()
        );
        let b2 = op.next_batch(&ctx, &s).expect("second batch settles EOS");
        assert!(b2.is_empty(), "EOS after the drain");
    }

    /// **#709 regression (focused unit test).** A `Remove(Set(Create))`
    /// stack models `MATCH (n) SET n.a = 1 REMOVE n.a`. Pre-fix, the
    /// inner SET returned `Batch::empty(0)` → the outer REMOVE read it
    /// as upstream-EOS and never ran, so `a` stayed `1` (the REMOVE was
    /// silently dropped). Post-fix: the inner SET is STACKED (passes its
    /// row through), the outer REMOVE is TERMINAL (clears `a`, drains,
    /// emits 0 rows).
    #[test]
    fn stacked_set_then_remove_composes_to_terminal_outer() {
        let tenant = TenantId::DEFAULT;
        let s = StubExecutorSubstrate::new();
        let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);
        let create = mk_create_node(BindingId::new(0), "User");
        let mut set = SetOp::new(
            create,
            vec![SetItemSpec {
                binding: BindingId::new(0),
                kind: SetTargetKind::Node,
                mutation: LogicalSetMutation::PropertyAssign {
                    name: "a".into(),
                    value: BoundExpression::Literal {
                        value: Literal::Integer(1),
                        span: Span::point(1, 1),
                        type_info: None,
                    },
                },
            }],
        );
        // Inner SET has a write-op consumer above it → STACKED.
        set.mark_stacked();
        assert!(!set.is_terminal(), "inner SET is stacked");
        let mut remove = RemoveOp::new(
            PhysicalOperator::Set(set),
            vec![RemoveItemSpec {
                binding: BindingId::new(0),
                kind: SetTargetKind::Node,
                mutation: LogicalRemoveMutation::Property("a".into()),
            }],
        );
        assert!(remove.is_terminal(), "outer REMOVE is terminal (root)");
        // The terminal outer REMOVE drains the inner SET (which passes its
        // mutated row up) and emits 0 rows — but composition still clears
        // the SET-written `a`.
        let b1 = remove.next_batch(&ctx, &s).expect("first batch OK");
        assert!(
            b1.is_empty(),
            "terminal outer REMOVE drains + emits 0 rows, got {} row(s)",
            b1.row_count()
        );
        let b2 = remove
            .next_batch(&ctx, &s)
            .expect("second batch settles EOS");
        assert!(b2.is_empty());

        // SET a=1 then REMOVE a on the SAME node → `a` is absent.
        // (Pre-#709-fix the inner's empty batch was read as EOS → the
        // REMOVE never ran → a=1 persisted; this proves composition.)
        let node_id = NodeId::new((1u64 << 32) + 1);
        let bag = s
            .node_properties(tenant, node_id)
            .expect("SET recorded a property bag");
        assert_eq!(
            bag.get("a"),
            None,
            "stacked SET a=1 then REMOVE a must clear `a`, got {:?}",
            bag.get("a")
        );
    }

    #[test]
    fn remove_op_label_routes_through_substrate() {
        let s = StubExecutorSubstrate::new();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let create = mk_create_node(BindingId::new(0), "User");
        let items = vec![RemoveItemSpec {
            binding: BindingId::new(0),
            kind: SetTargetKind::Node,
            mutation: LogicalRemoveMutation::LabelRemove(vec!["VIP".into()]),
        }];
        let mut op = RemoveOp::new(create, items);
        let _ = op.next_batch(&ctx, &s).expect("first batch OK");
        let _ = op.next_batch(&ctx, &s).expect("second batch settles EOS");
    }

    #[test]
    fn remove_op_pre_cancellation_short_circuits() {
        let s = StubExecutorSubstrate::new();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        ctx.cancellation().cancel();
        let create = mk_create_node(BindingId::new(0), "User");
        let items = vec![RemoveItemSpec {
            binding: BindingId::new(0),
            kind: SetTargetKind::Node,
            mutation: LogicalRemoveMutation::Property("age".into()),
        }];
        let mut op = RemoveOp::new(create, items);
        let r = op.next_batch(&ctx, &s);
        assert_eq!(r, Err(ExecutionError::Cancelled));
    }
}
