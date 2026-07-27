//! [`SetOp`] — write-op operator for `SET <item> (, <item>)*` per
//! ADR-150 (W26-θ Phase 4).
//!
//! Lowers from [`crate::logical_plan::LogicalSet`]. The operator
//! holds an upstream sub-pipeline (`input_op`) producing one row per
//! MATCH-bound trigger; per row it applies each item's mutation
//! (property assign / merge / replace / label add) via the
//! substrate's `set_node` / `set_rel` (per the item's
//! [`SetTargetKind`] discriminator).
//!
//! # Terminal-vs-stacked emission (the #709 fix, R1-narrowed)
//!
//! A SET op is either **stacked** (it has a ROW-CONSUMER above it — a
//! write-op for the outer clause of `SET … SET …` / `SET … REMOVE …`
//! (#709), OR a `Project` / `Aggregate` / `Unwind` row-consumer for
//! `SET … RETURN …` / `SET … WITH …` / `SET … RETURN count(a)` /
//! `SET … UNWIND …` (#772)) or **terminal** (it is the pipeline root /
//! has no row-consumer above it). The two shapes have DIFFERENT
//! row-cardinality contracts, and the
//! op's `terminal` flag (set at [`crate::executor::Pipeline::build`] time)
//! selects between them:
//!
//! - **Stacked** (`terminal == false`): on each `next_batch`, pull ONE
//!   batch from `input_op`, apply each item's mutation per row, MIRROR the
//!   mutation onto the row's in-memory entity view, and PASS THE (now
//!   post-SET) rows THROUGH to the consumer (output schema = input schema;
//!   only the property values change, not the column layout). The empty
//!   upstream batch is the EOS sentinel and is propagated. This is what
//!   lets a stacked outer SET/REMOVE see the rows and apply its own
//!   mutation in source order (#709), AND lets a `Project` above the SET
//!   project the post-SET property values for `SET … RETURN …` (#772 —
//!   the mirror keeps the projected `a.x` in lock-step with the
//!   substrate; see `SetOp::apply_rows_stacked`).
//! - **Terminal** (`terminal == true`): DRAIN the upstream fully —
//!   repeatedly pull batches and apply each row's mutation until the
//!   upstream is exhausted — then return an EMPTY batch. A RETURN-less
//!   terminal write yields **0 rows** to the driver (openCypher v9 +
//!   ADR-149/150 §D + ADR-182 v1.0-α contract: a terminal SET/REMOVE
//!   produces no result rows). Draining in one call is necessary because
//!   the materialize loop ([`crate::executor::execute_with_context`] /
//!   `crate::materialize`) breaks on the FIRST empty batch — so a
//!   terminal op that returned empty after only the first upstream batch
//!   would skip the mutations for every later batch (> [`Batch`]
//!   `BATCH_ROWS` rows). The internal drain applies to ALL matched rows.
//!
//! # Why the flag (and not unconditional pass-through)
//!
//! Pre-#709, SET swallowed its rows and returned `Batch::empty(0)`
//! UNCONDITIONALLY. That made a *stacked* outer SET/REMOVE (the
//! `Set(v=1, Set(v=0, Scan))` lowering of `SET n.a = 0 SET n.a = 1`)
//! read the inner op's empty batch as upstream-EOS and never apply its
//! own items — only the INNERMOST clause ran, persisting the FIRST write
//! and violating Cypher last-writer-wins (#709, HIGH correctness). The
//! naive fix (pass through UNCONDITIONALLY) composed stacked writes but
//! made a RETURN-less *terminal* write emit a row to the driver, breaking
//! the openCypher TCK write-op RowSet conformance gate (terminal SET →
//! 0 rows). The `terminal` flag keeps BOTH correct: stacked composes
//! (rows flow between write-ops), terminal yields 0 rows. The
//! substrate's per-key `insert` (last call wins — see
//! [`crate::executor::substrate::ExecutorSubstrate::set_node`]) provides
//! last-writer-wins regardless of which op is terminal.
//!
//! # Schema
//!
//! The output schema EQUALS the input (upstream) schema — SET binds no
//! new columns, it only mutates the substrate. A **stacked** SET re-emits
//! the same rows (carrying the post-SET property values, via the in-view
//! mirror) for its row-consumer; a **terminal** SET drains them and emits
//! none. `SET … RETURN …` / `SET … WITH …` lower to `Project(Set(…))`,
//! and the `Project` build arm flips the SET child to **stacked** (#772),
//! so the RETURN/WITH projects the mutated rows. The aggregate forms
//! (`SET … RETURN count(a)` / `sum(a.x)` / `… WITH <agg> …`) lower to
//! `Project(Aggregate(Set(…)))` — the `Aggregate` is the SET's direct
//! parent, so the `Aggregate` build arm (not the `Project`) does the flip
//! (else the terminal SET drains and the aggregate folds over 0 rows →
//! `count(a)=0` / `sum=NULL`); `SET … UNWIND …` lowers to `Unwind(Set(…))`
//! and the `Unwind` arm flips it. A bare `SET …` with no row-consumer
//! above it stays terminal → 0 rows (the openCypher v9 / ADR-149/150 §D /
//! ADR-182 RETURN-less terminal-write contract).
//!
//! # ADR provenance
//! - **ADR-150** — primary spec (W26-θ Phase 4).
//! - **ADR-147** §D-7 — production-substrate convention (per-tenant
//!   Transaction; default trait impl returns `IndexUnavailable`).
//! - **ADR-031** + **ADR-033** — per-tenant `Transaction` discipline
//!   (commit + rollback).
//! - **ADR-018** — MVCC version-chain semantics for `update_node` /
//!   `update_rel`.

use arcgraph_core::{NodeId, RelId};

use crate::executor::batch::Batch;
use crate::executor::context::ExecutionContext;
use crate::executor::error::ExecutionError;
use crate::executor::ops::{PhysicalOperator, schema_index};
use crate::executor::substrate::{ExecutorSubstrate, SetNodeMutation, SetRelMutation};
use crate::executor::value::{NodeView, RelView, Value};
use crate::logical_plan::{LogicalSetMutation, SetTargetKind};
use crate::semantic::bound_ast::{BindingId, BoundExpression};

/// One bound SET item — the binding identifier + the Node-vs-Rel
/// substrate-dispatch discriminator + the materialized mutation,
/// captured at pipeline-build time.
#[derive(Debug, Clone)]
pub struct SetItemSpec {
    pub binding: BindingId,
    pub kind: SetTargetKind,
    pub mutation: LogicalSetMutation,
}

/// SET executor op (ADR-150 W26-θ Phase 4; #709 fix, R1-narrowed).
#[derive(Debug)]
pub struct SetOp {
    /// Upstream sub-pipeline producing the MATCH-bound rows.
    input_op: Box<PhysicalOperator>,
    /// Per-item bound SET specs (in source order).
    items: Vec<SetItemSpec>,
    /// Cached output schema — EQUALS the input schema (SET binds no new
    /// columns). For a **stacked** op this is the pass-through layout a
    /// row-consumer (write-op #709, or `Project` / `Aggregate` / `Unwind`
    /// #772) resolves its bindings against; for a **terminal** op the
    /// schema width still describes the (empty) EOS batch.
    schema: Vec<BindingId>,
    /// `true` when this SET is the pipeline root / has no row-consumer
    /// above it → it DRAINS the upstream and emits **0 rows** (the
    /// openCypher / ADR-149/150 §D / ADR-182 terminal-write contract).
    /// `false` when it is **stacked** under a row-consumer — another
    /// write-op (`SET … SET …` / `SET … REMOVE …`, #709 last-writer-wins)
    /// OR a `Project` / `Aggregate` / `Unwind` (`SET … RETURN …` /
    /// `… WITH …` / `RETURN count(a)` / `SET … UNWIND …`, #772) — → it
    /// passes its mutated rows through so the consumer composes / projects /
    /// folds. Set at [`crate::executor::Pipeline::build`] time: [`Self::new`]
    /// defaults to terminal; the build flips it via [`Self::mark_stacked`]
    /// when the op is wired as a write-op's / `Project`'s / `Aggregate`'s /
    /// `Unwind`'s `input`.
    terminal: bool,
    /// EOS flag — set when upstream returned an empty batch on the
    /// prior pull (stacked path) or after the terminal drain completes.
    eos: bool,
}

impl SetOp {
    /// Construct a fresh **terminal** [`SetOp`] from a
    /// [`crate::logical_plan::LogicalSet`].
    ///
    /// Terminal is the default because the common shape is a RETURN-less
    /// `MATCH … SET …` whose SET is the pipeline root (0 result rows).
    /// [`crate::executor::Pipeline::build`] flips a SET that is wired as a
    /// row-consumer's `input` (another write-op #709, or a `Project` /
    /// `Aggregate` / `Unwind` #772) to stacked via [`Self::mark_stacked`].
    #[must_use]
    pub fn new(input_op: PhysicalOperator, items: Vec<SetItemSpec>) -> Self {
        // Output schema = input schema: SET mutates the substrate and
        // binds no new columns, so its column layout is the upstream's.
        // Capturing it here lets a stacked outer SET/REMOVE
        // `schema_index` its item bindings against the rows this op
        // forwards (the #709 composition fix).
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
    /// above it, so it forwards its mutated rows instead of draining them.
    /// Called by [`crate::executor::Pipeline::build`] for a SET/REMOVE
    /// wired as another write-op's `input` (#709) OR as a `Project` /
    /// `Aggregate` / `Unwind` row-consumer's `input` (#772), and by the
    /// stacked-composition unit tests that construct the operator tree by
    /// hand.
    pub fn mark_stacked(&mut self) {
        self.terminal = false;
    }

    /// `true` iff this op is terminal (drains + emits 0 rows). Exposed
    /// for the terminal-vs-stacked row-cardinality pin test.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        self.terminal
    }

    /// Output schema — equals the input schema (SET binds no columns).
    pub fn schema(&self) -> &[BindingId] {
        &self.schema
    }

    /// Apply every item's mutation to each row of `batch` via the
    /// substrate. Shared by the stacked (pass-through) + terminal (drain)
    /// paths so the per-row dispatch lives in ONE place.
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
                        "SetOp: item binding {:?} not in upstream schema {:?}",
                        item.binding, upstream_schema
                    ))
                })?;
                let cell = row.get(idx).ok_or_else(|| {
                    ExecutionError::Eval(format!("SetOp: row missing cell at index {idx}"))
                })?;
                match item.kind {
                    SetTargetKind::Node => {
                        let node_id = node_id_from_value(cell)?;
                        let mutation = build_node_mutation(&item.mutation)?;
                        substrate
                            .set_node(ctx.tenant(), node_id, &mutation, ctx)
                            .map_err(ExecutionError::Substrate)?;
                    }
                    SetTargetKind::Rel => {
                        let rel_id = rel_id_from_value(cell)?;
                        let mutation = build_rel_mutation(&item.mutation)?;
                        substrate
                            .set_rel(ctx.tenant(), rel_id, &mutation, ctx)
                            .map_err(ExecutionError::Substrate)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Apply every item's mutation to each row via the substrate **and
    /// mirror** the mutation onto the row's in-memory `NodeView` /
    /// `RelView`, so a downstream row-consumer observes the POST-SET
    /// property values — not the stale pre-SET bag. This is the SAME
    /// substrate-write-then-view-mirror contract MERGE's
    /// RETURN-after-MERGE uses (RC-2 per ADR-151-amendment-01 §D-2; see
    /// [`super::merge::fire_actions`] + [`apply_node_mutation_to_view`]).
    ///
    /// Used by the STACKED path ONLY (#772): a stacked SET passes its rows
    /// through to a consumer — a `Project` for `SET … RETURN …` /
    /// `SET … WITH …`, or a stacked outer write-op — which evaluates
    /// property access against the row-local `NodeView` bag
    /// ([`crate::executor::eval::evaluate`]'s `PropertyAccess` arm), so the
    /// mirror is what makes `SET a.x = v RETURN a.x` project `v` (not
    /// stale `NULL`). The TERMINAL path drains its rows (no consumer) and
    /// applies via [`Self::apply_batch`] WITHOUT the mirror — mirroring
    /// rows that are immediately discarded would be wasted work, and the
    /// substrate write (the only durable effect of a terminal SET) is
    /// identical on both paths.
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
                        "SetOp: item binding {:?} not in upstream schema {:?}",
                        item.binding, upstream_schema
                    ))
                })?;
                let cell = row.get_mut(idx).ok_or_else(|| {
                    ExecutionError::Eval(format!("SetOp: row missing cell at index {idx}"))
                })?;
                match item.kind {
                    SetTargetKind::Node => {
                        let node_id = node_id_from_value(cell)?;
                        let mutation = build_node_mutation(&item.mutation)?;
                        substrate
                            .set_node(ctx.tenant(), node_id, &mutation, ctx)
                            .map_err(ExecutionError::Substrate)?;
                        // #772 RC-2 — mirror onto the passed-through view so
                        // a RETURN/WITH (Project) over this stacked SET reads
                        // the post-SET value.
                        if let Value::Node(view) = cell {
                            apply_node_mutation_to_view(view, &mutation);
                        }
                    }
                    SetTargetKind::Rel => {
                        let rel_id = rel_id_from_value(cell)?;
                        let mutation = build_rel_mutation(&item.mutation)?;
                        substrate
                            .set_rel(ctx.tenant(), rel_id, &mutation, ctx)
                            .map_err(ExecutionError::Substrate)?;
                        if let Value::Relationship(view) = cell {
                            apply_rel_mutation_to_view(view, &mutation);
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
    ///   each item per row (mirroring the mutation onto the row's view per
    ///   `Self::apply_rows_stacked`), and PASS THE ROWS THROUGH (output
    ///   schema = input schema) so a stacked outer write-op composes
    ///   (#709) or a `Project` (RETURN/WITH) projects the post-SET rows
    ///   (#772). The empty upstream batch is propagated as EOS.
    /// - **Terminal**: DRAIN the upstream fully (apply every batch's
    ///   rows) then emit an EMPTY batch — a RETURN-less terminal write
    ///   yields 0 rows (openCypher / ADR-149/150 §D / ADR-182). Draining
    ///   in one call is required because the materialize loop breaks on
    ///   the first empty batch.
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
            // Terminal: drain the WHOLE upstream, applying mutations to
            // every matched row, then emit empty (0 result rows). The
            // internal loop is necessary because the driver breaks on the
            // first empty batch — returning empty after only batch 1
            // would skip mutations for rows in later batches (> BATCH_ROWS
            // matches). Cancellation is re-checked per iteration so a
            // long drain stays interruptible.
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

        // Stacked: apply this op's mutations to one batch + mirror them onto
        // the rows (#772 — so a `Project`/RETURN/WITH or a stacked outer
        // write-op observes the post-SET values), then pass the rows through.
        // The batch came from `input_op` whose schema == `self.schema`, so
        // the row width already matches the pass-through schema. `from_rows`
        // returns `None` only if `row_count() > BATCH_ROWS`, which cannot
        // happen for a batch we just pulled (it honoured the cap upstream) —
        // the `ok_or_else` guard is defense-in-depth.
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
                "SetOp: pass-through batch exceeded BATCH_ROWS (unreachable — \
                 upstream honoured the row cap)"
                    .into(),
            )
        })
    }
}

/// Materialize a [`LogicalSetMutation`] into a [`SetNodeMutation`].
/// Literal expressions are evaluated to runtime `Value`s; non-literal
/// expressions surface a clean `ExecutionError::Eval` (defense-in-
/// depth — the type-check pass already enforced literal-only values
/// per ADR-150 §D-4).
///
/// `pub(super)` — also consumed by the Phase 5 [`super::merge::MergeOp`]
/// per ADR-151 §D-7 (action firing reuses Phase 4's mutation
/// materialization).
pub(super) fn build_node_mutation(
    m: &LogicalSetMutation,
) -> Result<SetNodeMutation, ExecutionError> {
    match m {
        LogicalSetMutation::PropertyAssign { name, value } => Ok(SetNodeMutation::PropertyAssign {
            name: name.clone(),
            value: literal_to_value(value)?,
        }),
        LogicalSetMutation::PropertyReplace(entries) => {
            let mut out = Vec::with_capacity(entries.len());
            for (k, v) in entries {
                out.push((k.clone(), literal_to_value(v)?));
            }
            Ok(SetNodeMutation::PropertyReplace(out))
        }
        LogicalSetMutation::PropertyMerge(entries) => {
            let mut out = Vec::with_capacity(entries.len());
            for (k, v) in entries {
                out.push((k.clone(), literal_to_value(v)?));
            }
            Ok(SetNodeMutation::PropertyMerge(out))
        }
        LogicalSetMutation::LabelAdd(labels) => Ok(SetNodeMutation::LabelAdd(labels.clone())),
    }
}

/// Materialize a [`LogicalSetMutation`] into a [`SetRelMutation`] —
/// like [`build_node_mutation`] but rejects `LabelAdd` (rels do not
/// carry labels per ADR-150 §D-4; the type-check pass should already
/// have rejected this case).
///
/// `pub(super)` — also consumed by the Phase 5 [`super::merge::MergeOp`]
/// per ADR-151 §D-7.
pub(super) fn build_rel_mutation(m: &LogicalSetMutation) -> Result<SetRelMutation, ExecutionError> {
    match m {
        LogicalSetMutation::PropertyAssign { name, value } => Ok(SetRelMutation::PropertyAssign {
            name: name.clone(),
            value: literal_to_value(value)?,
        }),
        LogicalSetMutation::PropertyReplace(entries) => {
            let mut out = Vec::with_capacity(entries.len());
            for (k, v) in entries {
                out.push((k.clone(), literal_to_value(v)?));
            }
            Ok(SetRelMutation::PropertyReplace(out))
        }
        LogicalSetMutation::PropertyMerge(entries) => {
            let mut out = Vec::with_capacity(entries.len());
            for (k, v) in entries {
                out.push((k.clone(), literal_to_value(v)?));
            }
            Ok(SetRelMutation::PropertyMerge(out))
        }
        LogicalSetMutation::LabelAdd(_) => Err(ExecutionError::Eval(
            "SetOp: label-add mutation rejected on Relationship binding (Phase 4 per ADR-150 \
             §D-4; type-check should have rejected this earlier)"
                .into(),
        )),
    }
}

/// Mirror a [`SetNodeMutation`] onto an in-memory [`NodeView`]'s
/// property bag — the **SINGLE SOURCE OF TRUTH** for the post-SET row
/// state per ADR-151-amendment-01 §D-2 (RC-2).
///
/// The Phase 5 [`super::merge::MergeOp`] RETURN-after-MERGE emission
/// path calls this immediately after dispatching the SAME
/// `SetNodeMutation` to `substrate.set_node`, so the row the MERGE
/// emits reflects `ON CREATE SET` / `ON MATCH SET` mutations (without
/// it, the emitted view is the *pre-SET* scan / create-branch bag and
/// `MERGE … ON MATCH SET n.x = 2 RETURN n.x` would return stale/Null).
///
/// The mutation is consumed AFTER [`build_node_mutation`] has already
/// performed the literal-lift, so the apply is exact under the
/// literal-only narrowing — there is no second evaluation path that
/// could diverge. The three property variants mirror the substrate's
/// own semantics 1:1 (`PropertyAssign` → per-key insert;
/// `PropertyReplace` → clear-then-set; `PropertyMerge` → additive
/// insert — cf. the `StubExecutorSubstrate::set_node` /
/// `arcgraph_storage::crud::update_node` implementations). `LabelAdd`
/// is a no-op on the property bag: the production substrate surfaces
/// `IndexUnavailable` for it (forward-pinned to v1.1 per ADR-150 §D-9)
/// so `fire_actions` errors before this mirror runs, AND the
/// multi-label `NodeView` shape is itself forward-pinned — so there is
/// nothing exact to mirror at v1.0-α.
///
/// `pub(super)` — the executor-ops-internal post-mutation row mirror,
/// shared by [`super::merge`] (and available to any future
/// RETURN-after-SET landing) so the property-apply logic lives in ONE
/// place.
pub(super) fn apply_node_mutation_to_view(node: &mut NodeView, mutation: &SetNodeMutation) {
    match mutation {
        SetNodeMutation::PropertyAssign { name, value } => {
            node.properties.insert(name.clone(), value.clone());
        }
        SetNodeMutation::PropertyReplace(entries) => {
            node.properties.clear();
            for (k, v) in entries {
                node.properties.insert(k.clone(), v.clone());
            }
        }
        SetNodeMutation::PropertyMerge(entries) => {
            for (k, v) in entries {
                node.properties.insert(k.clone(), v.clone());
            }
        }
        // See the doc comment: forward-pinned to v1.1; no property-bag
        // effect, and `fire_actions` never reaches the mirror on this
        // variant against the production substrate.
        SetNodeMutation::LabelAdd(_) => {}
    }
}

/// Mirror a [`SetRelMutation`] onto an in-memory [`RelView`]'s property
/// bag — the rel-side companion to [`apply_node_mutation_to_view`]
/// (ADR-151-amendment-01 §D-2). Rels carry no labels at v1.0-α so there
/// is no `LabelAdd` variant. Same single-source-of-truth contract.
pub(super) fn apply_rel_mutation_to_view(rel: &mut RelView, mutation: &SetRelMutation) {
    match mutation {
        SetRelMutation::PropertyAssign { name, value } => {
            rel.properties.insert(name.clone(), value.clone());
        }
        SetRelMutation::PropertyReplace(entries) => {
            rel.properties.clear();
            for (k, v) in entries {
                rel.properties.insert(k.clone(), v.clone());
            }
        }
        SetRelMutation::PropertyMerge(entries) => {
            for (k, v) in entries {
                rel.properties.insert(k.clone(), v.clone());
            }
        }
    }
}

/// Convert a `BoundExpression::Literal` to a runtime `Value`. Returns
/// a clean `ExecutionError::Eval` for non-literal expressions (the
/// type-check pass per ADR-150 §D-4 rejects these before reaching the
/// executor; this guard is defensive) and for composite literals that
/// do not round-trip losslessly at v1.0-α.
///
/// The literal → `Value` materialization is delegated to the shared
/// [`super::literal_lift::literal_value`] helper. Per
/// ADR-152-amendment-02 §D-1 a `List`-of-scalars literal lifts to
/// [`Value::List`] (SET shares the same write-op property-bag gate as
/// CREATE — see ADR-152-amendment-02 §D-1); `Map` + temporal / decimal
/// remain deferred per §D-2 and surface the clean composite-literal
/// Eval error.
fn literal_to_value(e: &BoundExpression) -> Result<Value, ExecutionError> {
    // #870 — the shared folder admits a bare `Literal` AND a NEGATIVE numeric
    // literal (`UnaryOp(Neg, <numeric literal>)`, e.g. `SET n.x = -3`); it
    // returns `None` for a genuine non-literal or a fenced/deferred composite.
    super::literal_lift::bound_literal_value(e).ok_or_else(|| {
        ExecutionError::Eval(
            "SetOp: SET property value is not a persistable literal at Phase 4 \
             (ADR-150 §D-4): only a scalar, a negative numeric literal, or a \
             List-of-scalars per ADR-152-amendment-02 §D-1; Map + temporal / decimal \
             deferred per §D-2 — type-check should have rejected a non-literal earlier"
                .into(),
        )
    })
}

/// Extract the `NodeId` from a `Value::Node` cell. Surfaces a clean
/// `ExecutionError::Eval` otherwise (defense-in-depth — the type-
/// check pass already enforced Node typing on every Node-kind SET
/// item per ADR-150 §D-4).
///
/// `pub(super)` — also consumed by the Phase 5 [`super::merge::MergeOp`]
/// per ADR-151 §D-7.
pub(super) fn node_id_from_value(v: &Value) -> Result<NodeId, ExecutionError> {
    match v {
        Value::Node(n) => Ok(n.id),
        other => Err(ExecutionError::Eval(format!(
            "SetOp: expected Node cell, got {other:?}"
        ))),
    }
}

/// Extract the `RelId` from a `Value::Relationship` cell.
///
/// `pub(super)` — also consumed by the Phase 5 [`super::merge::MergeOp`]
/// per ADR-151 §D-7.
pub(super) fn rel_id_from_value(v: &Value) -> Result<RelId, ExecutionError> {
    match v {
        Value::Relationship(r) => Ok(r.id),
        other => Err(ExecutionError::Eval(format!(
            "SetOp: expected Relationship cell, got {other:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use arcgraph_core::{LabelId, NodeId, PartitionId, TenantId};

    use super::*;
    use crate::ast::Literal;
    use crate::error::Span;
    use crate::executor::ops::CreateNodeOp;
    use crate::executor::substrate::StubExecutorSubstrate;
    use crate::executor::value::NodeView;

    fn mk_create_node(var: BindingId, label: &str) -> PhysicalOperator {
        PhysicalOperator::CreateNode(CreateNodeOp::new(
            Some(var),
            Some(label.to_string()),
            Vec::new(),
        ))
    }

    fn lit_str(s: &str) -> BoundExpression {
        BoundExpression::Literal {
            value: Literal::String(s.into()),
            span: Span::point(1, 1),
            type_info: None,
        }
    }

    fn lit_int(i: i64) -> BoundExpression {
        BoundExpression::Literal {
            value: Literal::Integer(i),
            span: Span::point(1, 1),
            type_info: None,
        }
    }

    #[test]
    fn terminal_set_op_applies_then_emits_zero_rows() {
        let tenant = TenantId::DEFAULT;
        let pre = NodeView::new(NodeId::new(1), Some(LabelId::new(1)));
        let s = StubExecutorSubstrate::new().with_node(tenant, pre.clone());
        let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);
        let create = mk_create_node(BindingId::new(0), "User");
        let items = vec![SetItemSpec {
            binding: BindingId::new(0),
            kind: SetTargetKind::Node,
            mutation: LogicalSetMutation::PropertyAssign {
                name: "name".into(),
                value: lit_str("Alice"),
            },
        }];
        // Terminal (#709 R1-narrowing): the CreateNode emits one `[Node]`
        // row; the terminal SET applies its mutation but DRAINS the row —
        // a RETURN-less terminal write yields 0 result rows (openCypher /
        // ADR-149/150 §D / ADR-182). The substrate mutation still happens.
        let mut op = SetOp::new(create, items);
        assert!(op.is_terminal(), "SetOp::new defaults to terminal");
        assert_eq!(op.schema(), &[BindingId::new(0)], "schema == input schema");
        let created_id = NodeId::new((1u64 << 32) + 1);
        let b1 = op.next_batch(&ctx, &s).expect("first batch OK");
        assert!(
            b1.is_empty(),
            "terminal SET drains its rows and emits 0 rows, got {} row(s)",
            b1.row_count()
        );
        // The mutation was applied despite emitting no rows.
        let bag = s
            .node_properties(tenant, created_id)
            .expect("terminal SET recorded a property bag");
        assert_eq!(
            bag.get("name"),
            Some(&Value::String("Alice".into())),
            "terminal SET applied the mutation, got {:?}",
            bag.get("name")
        );
        let b2 = op.next_batch(&ctx, &s).expect("second batch settles EOS");
        assert!(b2.is_empty(), "EOS after the drain");
    }

    /// **#709 regression (focused unit test).** A STACKED inner
    /// [`SetOp`] must pass its mutated rows through to the terminal outer
    /// [`SetOp`], which applies its OWN mutation — proving the
    /// composition wiring directly (the proptest exercises it end-to-end
    /// through parse→lower; this pins the operator contract).
    ///
    /// Models `MATCH (n) SET n.a = 0 SET n.a = 1` lowered to
    /// `Set(items=[a=1], Set(items=[a=0], Create))`. Pre-fix, the inner
    /// SET returned `Batch::empty(0)` → the outer SET read it as
    /// upstream-EOS and never applied `a=1`, persisting `a=0`
    /// (first-writer-wins). Post-fix: the inner SET is STACKED (passes its
    /// row through), the outer SET is TERMINAL (applies `a=1`, drains, and
    /// emits 0 rows); the substrate's last call (`a=1`) wins.
    #[test]
    fn stacked_set_composes_through_to_terminal_outer_set() {
        let tenant = TenantId::DEFAULT;
        let s = StubExecutorSubstrate::new();
        let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);
        let create = mk_create_node(BindingId::new(0), "User");
        let mut inner = SetOp::new(
            create,
            vec![SetItemSpec {
                binding: BindingId::new(0),
                kind: SetTargetKind::Node,
                mutation: LogicalSetMutation::PropertyAssign {
                    name: "a".into(),
                    value: lit_int(0),
                },
            }],
        );
        // The inner SET has a write-op consumer above it → STACKED
        // (pass-through). `Pipeline::build` flips this in production; the
        // hand-built tree flips it explicitly.
        inner.mark_stacked();
        assert!(!inner.is_terminal(), "inner SET is stacked");
        let mut outer = SetOp::new(
            PhysicalOperator::Set(inner),
            vec![SetItemSpec {
                binding: BindingId::new(0),
                kind: SetTargetKind::Node,
                mutation: LogicalSetMutation::PropertyAssign {
                    name: "a".into(),
                    value: lit_int(1),
                },
            }],
        );
        assert!(outer.is_terminal(), "outer SET is terminal (root)");
        // The terminal outer SET drains the inner op (which passes its
        // mutated row up) and emits 0 rows — but composition still applies
        // a=1 over the inner's a=0.
        let b1 = outer.next_batch(&ctx, &s).expect("first batch OK");
        assert!(
            b1.is_empty(),
            "terminal outer SET drains + emits 0 rows, got {} row(s)",
            b1.row_count()
        );
        let b2 = outer
            .next_batch(&ctx, &s)
            .expect("second batch settles EOS");
        assert!(b2.is_empty());

        // Last-writer-wins: the substrate's per-key insert applied a=0
        // (inner) THEN a=1 (outer) on the SAME node — final value is 1.
        // (Pre-#709-fix the inner's empty batch was read as EOS and the
        // outer never ran → a=0 persisted; this proves composition.)
        let node_id = NodeId::new((1u64 << 32) + 1);
        let bag = s
            .node_properties(tenant, node_id)
            .expect("SET recorded a property bag");
        assert_eq!(
            bag.get("a"),
            Some(&Value::Integer(1)),
            "stacked SET a=0 then a=1 must persist 1 (last-writer-wins), got {:?}",
            bag.get("a")
        );
    }

    #[test]
    fn set_op_pre_cancellation_short_circuits() {
        let s = StubExecutorSubstrate::new();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        ctx.cancellation().cancel();
        let create = mk_create_node(BindingId::new(0), "User");
        let items = vec![SetItemSpec {
            binding: BindingId::new(0),
            kind: SetTargetKind::Node,
            mutation: LogicalSetMutation::LabelAdd(vec!["VIP".into()]),
        }];
        let mut op = SetOp::new(create, items);
        let r = op.next_batch(&ctx, &s);
        assert_eq!(r, Err(ExecutionError::Cancelled));
    }

    #[test]
    fn set_op_label_add_routes_through_substrate() {
        let tenant = TenantId::DEFAULT;
        let s = StubExecutorSubstrate::new();
        let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);
        let create = mk_create_node(BindingId::new(0), "User");
        let items = vec![SetItemSpec {
            binding: BindingId::new(0),
            kind: SetTargetKind::Node,
            mutation: LogicalSetMutation::LabelAdd(vec!["VIP".into()]),
        }];
        let mut op = SetOp::new(create, items);
        let _ = op.next_batch(&ctx, &s).expect("first batch OK");
        let _ = op.next_batch(&ctx, &s).expect("second batch settles EOS");
    }

    #[test]
    fn set_op_non_literal_property_value_surfaces_eval_error() {
        // Defense-in-depth: a programmatic LogicalSetMutation with a
        // non-literal value surfaces a clean Eval error at the
        // executor — the type-check pass would normally reject this
        // shape but the executor double-checks.
        let s = StubExecutorSubstrate::new();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let create = mk_create_node(BindingId::new(0), "User");
        let items = vec![SetItemSpec {
            binding: BindingId::new(0),
            kind: SetTargetKind::Node,
            mutation: LogicalSetMutation::PropertyAssign {
                name: "name".into(),
                value: BoundExpression::Parameter {
                    name: "p".into(),
                    span: Span::point(1, 1),
                    type_info: None,
                },
            },
        }];
        let mut op = SetOp::new(create, items);
        let r = op.next_batch(&ctx, &s);
        assert!(matches!(r, Err(ExecutionError::Eval(_))));
    }
}
