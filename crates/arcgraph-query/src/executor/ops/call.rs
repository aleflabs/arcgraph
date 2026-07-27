//! [`CallOp`] + [`CorrelationSeedOp`] — `CALL { <subquery> }` correlated
//! brace-subquery (ADR-192, #623).
//!
//! **Provenance: `CALL { … }` is Cypher 25 — a deliberate
//! beyond-openCypher-v9 capability extension.** The vendored v9 TCK has
//! ZERO `CALL{}` scenarios (v9's `CALL` is a procedure call, scoped out).
//! The capability is proven by ADR-192's own test plan (tests 1-11),
//! NOT by a TCK bucket.
//!
//! Lowers from [`crate::logical_plan::LogicalCall`] +
//! [`crate::logical_plan::LogicalCorrelationSeed`].
//!
//! # The two operators
//!
//! - [`CorrelationSeedOp`] — a one-row table whose columns are a
//!   `CALL { … }` body's IMPORTED bindings. It reads the current driving
//!   row's imported values from the [`ExecutionContext`]'s
//!   correlation-frame stack (pushed by [`CallOp`]) and emits exactly one
//!   row, then EOS. It is the leading-clause `prev` of the lowered
//!   subquery body, so the body's first clause threads the imports
//!   through the existing lowering (a MATCH-led body equi-joins the seed
//!   on the imported start variable; a WITH/RETURN-led body reads the
//!   imports directly off the seed). An empty import set degenerates to a
//!   zero-column unit row (≡ the leading-clause [`super::EmptyOp::unit`]).
//!
//! - [`CallOp`] — the correlated per-driving-row executor (the
//!   [`super::UnwindOp`] analogue). For EACH driving row pulled from its
//!   `child`: it pushes a correlation frame (the imported bindings bound
//!   to that row's values), (re-)builds + drives the subquery `body`
//!   sub-plan, and emits `driving_row ++ body_output` for EACH body
//!   output row (UNION-ALL across driving rows). It carries a `pending`
//!   cross-batch cursor (mirroring [`super::UnwindOp`]) so a body that
//!   produces more rows than fit in one output batch resumes correctly.
//!
//! # Cardinality (ADR-192 D-7 / D-8 — the correctness hinge)
//!
//! The join cardinality `k` is measured on the body's **OUTPUT** (the
//! drained rows), NOT its input (D-6). `CallOp` drives the body to
//! exhaustion and emits one output row per body output row:
//!
//! - body returns `k` rows → driving row multiplied `k`-fold (D-7);
//! - body returns **0** rows → driving row **DROPPED** (D-7 — `CALL{}` is
//!   an inner correlation, NOT an OPTIONAL/left-join);
//! - **aggregating body** over an EMPTY correlated set → the body's
//!   terminal [`super::AggregateOp`] STILL emits its empty-input identity
//!   row (e.g. `count(*) → 0`, `aggregate.rs` §amendment-03 §TIER-2-b), so
//!   the body returns 1 row → the driving row is **PRESERVED** with the
//!   aggregate (D-8). This falls out of "measure k on the body OUTPUT"
//!   for free — `CallOp` needs no special aggregating-body detection
//!   because the body's own `AggregateOp` produces the identity row. A
//!   NAIVE impl that measured `k` on the body's INPUT (the correlated
//!   expansion) would instead DROP the driving row — failing ADR-192
//!   test 5. `CallOp` is correct by construction because it drains the
//!   body and counts its OUTPUT.
//!
//! # Body output relabel (the projection fresh-id bridge)
//!
//! The body's terminal projection ([`super::ProjectOp`] /
//! [`super::AggregateOp`]) mints FRESH synthetic output binding-ids that
//! do not match the binder's ids for the returned column names. `CallOp`
//! therefore IGNORES the body op's output schema ids and labels its
//! output's body columns with `returned` — the binder-declared OUTER-scope
//! ids (ADR-192 D-4) — taking the values POSITIONALLY from each body row.
//! This is what makes the enclosing query's reference to a returned
//! column resolve. Schema = `child_schema ++ returned`.
//!
//! # ADR provenance
//! - **ADR-192** — the approved design this slice realizes (D-3 import,
//!   D-4 fence/returned, D-5/D-5a lowering, D-6/D-7/D-8 cardinality).
//! - **`UnwindOp`** (`executor/ops/unwind.rs`) — the correlated-per-row +
//!   `pending` cross-batch structural template (#618).
//! - **`OptionalExpandOp`** (`executor/ops/optional_expand.rs`) — the
//!   per-driving-row sub-pipeline rebuild (factory) precedent.

use std::collections::VecDeque;

use crate::executor::batch::Batch;
use crate::executor::context::ExecutionContext;
use crate::executor::error::ExecutionError;
use crate::executor::ops::PhysicalOperator;
use crate::executor::ops::schema_index;
use crate::executor::substrate::ExecutorSubstrate;
use crate::executor::value::Value;
use crate::semantic::bound_ast::BindingId;

// =====================================================================
// CorrelationSeedOp — the per-driving-row correlation seed.
// =====================================================================

/// One-row correlation seed for a `CALL { … }` body (ADR-192 #623).
///
/// Emits exactly ONE row carrying the imported bindings' values for the
/// current driving row (read from the [`ExecutionContext`]'s
/// correlation-frame stack, which [`CallOp`] pushes), then EOS. The
/// output schema is the imported bindings; an empty import set yields a
/// zero-column unit row (≡ [`super::EmptyOp::unit`]).
#[derive(Debug)]
pub struct CorrelationSeedOp {
    /// Output schema = the imported bindings (one column each).
    schema: Vec<BindingId>,
    /// Set after the single row is emitted; subsequent calls EOS.
    emitted: bool,
}

impl CorrelationSeedOp {
    /// Construct a correlation seed over `imported` bindings.
    #[must_use]
    pub fn new(imported: Vec<BindingId>) -> Self {
        Self {
            schema: imported,
            emitted: false,
        }
    }

    /// Output schema (the imported bindings).
    pub fn schema(&self) -> &[BindingId] {
        &self.schema
    }

    /// Emit the single seed row (the current driving row's imported
    /// values) on the first call, then EOS.
    pub fn next_batch<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        _substrate: &S,
    ) -> Result<Batch, ExecutionError> {
        ctx.cancellation().check()?;
        if self.emitted {
            return Ok(Batch::empty(self.schema.len()));
        }
        self.emitted = true;
        // Read each imported binding's value from the active correlation
        // frame. A missing value (no active frame — e.g. the operator is
        // driven outside any CALL, which the lowering never does) is
        // NULL, defensively.
        let row: Vec<Value> = self
            .schema
            .iter()
            .map(|b| ctx.correlation_value(*b).unwrap_or(Value::Null))
            .collect();
        let mut batch = Batch::with_capacity(self.schema.len());
        let pushed = batch.push_row(row);
        debug_assert!(pushed, "seed row must fit a fresh batch");
        Ok(batch)
    }
}

// =====================================================================
// CallOp — correlated per-driving-row subquery execution.
// =====================================================================

/// Factory rebuilding the subquery `body` sub-pipeline. Called once per
/// driving row to get a fresh body operator (operators carry per-run
/// cursor state, so they cannot be reset — they are rebuilt, exactly as
/// [`super::OptionalExpandOp`]'s right-side factory does). The body's
/// [`CorrelationSeedOp`] reads the per-row imports from the
/// [`ExecutionContext`] frame [`CallOp`] pushes, so the factory itself is
/// argument-free.
pub type CallBodyFactory = Box<dyn Fn() -> Result<PhysicalOperator, ExecutionError> + Send + Sync>;

/// The driving row currently being processed + its live body sub-pipeline.
struct Pending {
    /// The driving row's cells (the inherited columns), cloned into each
    /// emitted output row as the leading prefix.
    prefix: Vec<Value>,
    /// The live subquery body sub-pipeline for THIS driving row. Boxed —
    /// a `PhysicalOperator::Call` embeds `Pending`, so an unboxed
    /// `PhysicalOperator` here would make the enum infinitely sized.
    body: Box<PhysicalOperator>,
    /// Body output rows pulled but not yet emitted (cross-batch carry).
    buffer: VecDeque<Vec<Value>>,
    /// Set once the body returns its EOS empty batch.
    body_done: bool,
}

impl std::fmt::Debug for Pending {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pending")
            .field("prefix_len", &self.prefix.len())
            .field("buffered", &self.buffer.len())
            .field("body_done", &self.body_done)
            .finish()
    }
}

/// `CALL { <subquery> }` correlated executor (ADR-192 #623).
pub struct CallOp {
    /// Driving (outer) child producing the rows the subquery runs over.
    child: Box<PhysicalOperator>,
    /// Rebuilds the subquery body sub-pipeline per driving row.
    body_factory: CallBodyFactory,
    /// Outer bindings imported into the subquery (ADR-192 D-3). Together
    /// with `child_schema` this drives the per-driving-row correlation
    /// frame [`CallOp`] pushes onto the [`ExecutionContext`].
    imported: Vec<BindingId>,
    /// Cached child schema (for the per-driving-row imported-value
    /// extraction).
    child_schema: Vec<BindingId>,
    /// Output schema: `child_schema ++ returned`.
    schema: Vec<BindingId>,
    /// Width of the body output columns (= `returned.len()`). Each body
    /// output row is appended to the driving-row prefix.
    body_width: usize,
    /// The driving row mid-subquery + its live body (carry-over across
    /// `next_batch` when the output batch fills mid-subquery).
    pending: Option<Pending>,
    /// `true` while a correlation frame is pushed for the current
    /// `pending` (so we pop it EXACTLY once when the pending is retired).
    frame_pushed: bool,
    /// Not-yet-processed driving rows of the child batch we last pulled.
    buffered_input: VecDeque<Vec<Value>>,
    /// Set once the child returns its EOS empty batch.
    child_done: bool,
}

impl std::fmt::Debug for CallOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CallOp")
            .field("child", &self.child)
            .field("imported", &self.imported)
            .field("schema", &self.schema)
            .field("body_width", &self.body_width)
            .field("pending", &self.pending)
            .field("frame_pushed", &self.frame_pushed)
            .field("child_done", &self.child_done)
            .finish()
    }
}

impl CallOp {
    /// Construct a `CallOp` over `child`. The output schema is the
    /// child's schema with the body's `returned` columns appended
    /// (ADR-192 D-4 + D-6).
    #[must_use]
    pub fn new(
        child: PhysicalOperator,
        body_factory: CallBodyFactory,
        imported: Vec<BindingId>,
        returned: Vec<BindingId>,
    ) -> Self {
        let child_schema = child.schema().to_vec();
        let body_width = returned.len();
        let mut schema = child_schema.clone();
        schema.extend_from_slice(&returned);
        Self {
            child: Box::new(child),
            body_factory,
            imported,
            child_schema,
            schema,
            body_width,
            pending: None,
            frame_pushed: false,
            buffered_input: VecDeque::new(),
            child_done: false,
        }
    }

    /// Output schema (`child_schema ++ returned`).
    pub fn schema(&self) -> &[BindingId] {
        &self.schema
    }

    /// Build the correlation frame for a driving row — the imported
    /// bindings bound to that row's values (looked up by column index in
    /// the child schema).
    fn frame_for(&self, driving_row: &[Value]) -> Vec<(BindingId, Value)> {
        self.imported
            .iter()
            .map(|b| {
                let v = schema_index(&self.child_schema, *b)
                    .and_then(|i| driving_row.get(i).cloned())
                    .unwrap_or(Value::Null);
                (*b, v)
            })
            .collect()
    }

    /// Retire the current `pending`: pop its correlation frame (once) and
    /// clear the slot.
    fn retire_pending(&mut self, ctx: &ExecutionContext) {
        if self.frame_pushed {
            ctx.pop_correlation_frame();
            self.frame_pushed = false;
        }
        self.pending = None;
    }

    /// Pull the next batch of correlated subquery rows.
    ///
    /// Thin wrapper over `Self::next_batch_inner` upholding the
    /// **correlation-frame balance invariant on the error path** (#744 R1,
    /// LOW-1). A frame is pushed for a driving row (step 3) BEFORE the body
    /// is driven (step 1) and intentionally OUTLIVES a single `next_batch`
    /// call when a multi-batch body suspends mid-subquery (the `Ok(out)`
    /// carry); it is popped on normal exhaustion by
    /// `Self::retire_pending`. But on ANY error return — a body-drive
    /// fault, or a cancellation check that trips while a suspended frame is
    /// live — the `?`-propagated error would otherwise strand that frame on
    /// the [`ExecutionContext`] correlation stack. This wrapper pops it via
    /// `retire_pending` (idempotent — acts only while `frame_pushed`), so
    /// the stack stays balanced even when the query aborts mid-subquery.
    pub fn next_batch<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<Batch, ExecutionError> {
        let result = self.next_batch_inner(ctx, substrate);
        if result.is_err() && self.frame_pushed {
            // Error abort with a live correlation frame ⇒ pop it so the
            // ExecutionContext correlation stack stays balanced (LOW-1).
            self.retire_pending(ctx);
        }
        result
    }

    /// Inner driver. See [`Self::next_batch`] for the error-path frame
    /// guard that wraps this.
    fn next_batch_inner<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<Batch, ExecutionError> {
        ctx.cancellation().check()?;
        let mut out = Batch::with_capacity(self.schema.len());

        loop {
            // 1. Drain the current pending body into the output batch.
            if self.pending.is_some() {
                // Borrow-split: drain in a sub-scope so we can call
                // `retire_pending(&mut self)` after.
                let exhausted = {
                    let p = self.pending.as_mut().expect("is_some checked");
                    loop {
                        ctx.cancellation().check()?;
                        // Drain the buffer into out.
                        while !out.is_full() {
                            match p.buffer.pop_front() {
                                Some(body_row) => {
                                    debug_assert_eq!(
                                        body_row.len(),
                                        self.body_width,
                                        "body output row width must equal returned-column count"
                                    );
                                    let mut row =
                                        Vec::with_capacity(p.prefix.len() + body_row.len());
                                    row.extend_from_slice(&p.prefix);
                                    row.extend(body_row);
                                    let pushed = out.push_row(row);
                                    debug_assert!(pushed, "guarded by !out.is_full()");
                                }
                                None => break,
                            }
                        }
                        if out.is_full() {
                            // Output full mid-subquery — keep the pending
                            // (and its pushed frame) for the next call.
                            return Ok(out);
                        }
                        if p.body_done {
                            break; // buffer drained + body EOS ⇒ pending done
                        }
                        // Pull the next body batch.
                        let body_batch = p.body.next_batch(ctx, substrate)?;
                        if body_batch.is_empty() {
                            p.body_done = true;
                        } else {
                            p.buffer = VecDeque::from(body_batch.into_rows());
                        }
                    }
                    true
                };
                if exhausted {
                    self.retire_pending(ctx);
                }
            }
            if out.is_full() {
                break;
            }

            // 2. Get the next driving row (from the buffered batch, or by
            //    pulling a fresh batch from the child).
            let next_row = match self.buffered_input.pop_front() {
                Some(row) => row,
                None => {
                    if self.child_done {
                        break; // nothing pending, nothing buffered, child EOS
                    }
                    let child_batch = self.child.next_batch(ctx, substrate)?;
                    if child_batch.is_empty() {
                        self.child_done = true;
                        break;
                    }
                    self.buffered_input = VecDeque::from(child_batch.into_rows());
                    continue;
                }
            };

            // 3. Push the correlation frame for this driving row, build a
            //    fresh body sub-pipeline, and stage the pending. The frame
            //    is pushed BEFORE the body runs so the body's
            //    `CorrelationSeedOp` reads this row's imports; it stays
            //    pushed until the pending is retired (step 1), spanning
            //    however many output batches the body fans into.
            let frame = self.frame_for(&next_row);
            ctx.push_correlation_frame(frame);
            self.frame_pushed = true;
            let body = match (self.body_factory)() {
                Ok(b) => b,
                Err(e) => {
                    // Build failure: pop the frame we just pushed so the
                    // stack stays balanced, then surface the error.
                    ctx.pop_correlation_frame();
                    self.frame_pushed = false;
                    return Err(e);
                }
            };
            self.pending = Some(Pending {
                prefix: next_row,
                body: Box::new(body),
                buffer: VecDeque::new(),
                body_done: false,
            });
            // Loop back to drain (step 1).
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use arcgraph_core::{LabelId, Lsn, NodeId, PartitionId, TenantId};

    use super::*;
    use crate::ast::{Expression, Literal};
    use crate::error::Span;
    use crate::executor::batch::BATCH_ROWS;
    use crate::executor::ops::{AggregateCall, AggregateOp, EmptyOp, ScanOp, UnwindOp};
    use crate::executor::substrate::StubExecutorSubstrate;
    use crate::executor::value::NodeView;
    use crate::logical_plan::AggregationKind;
    use crate::semantic::bound_ast::BoundExpression;

    // ---- helpers ----------------------------------------------------

    const IMP: BindingId = BindingId::new(0); // imported / driving binding
    const RET: BindingId = BindingId::new(9); // returned body column
    const AGG_IN: BindingId = BindingId::new(5); // body-internal expansion var

    fn ctx() -> ExecutionContext {
        ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO)
    }

    fn int_list_expr(values: &[i64]) -> BoundExpression {
        BoundExpression::Literal {
            value: Literal::List(
                values
                    .iter()
                    .map(|n| Expression::Literal(Literal::Integer(*n)))
                    .collect(),
            ),
            span: Span::point(1, 1),
            type_info: None,
        }
    }

    /// A body factory that emits `k` rows (each a single Integer column),
    /// via `UNWIND [0..k] AS RET` over the unit row. Uncorrelated — it
    /// ignores the frame, which is fine for exercising `CallOp`'s
    /// drive-and-count + relabel + cross-batch mechanics.
    fn k_row_body_factory(k: usize) -> CallBodyFactory {
        Box::new(move || {
            let values: Vec<i64> = (0..k as i64).collect();
            Ok(PhysicalOperator::Unwind(UnwindOp::new(
                PhysicalOperator::Empty(EmptyOp::unit()),
                int_list_expr(&values),
                RET,
            )))
        })
    }

    /// A body factory that emits ZERO rows (an empty `UNWIND []`). The
    /// non-aggregating-empty case (ADR-192 D-7 — driving row DROPPED).
    fn empty_body_factory() -> CallBodyFactory {
        Box::new(|| {
            Ok(PhysicalOperator::Unwind(UnwindOp::new(
                PhysicalOperator::Empty(EmptyOp::unit()),
                int_list_expr(&[]),
                RET,
            )))
        })
    }

    /// A body factory that emits EXACTLY ONE row REGARDLESS of the
    /// correlated input — the AGGREGATING-body analogue (a group-less
    /// aggregate emits its empty-input identity row even over an empty
    /// correlated set, so the body returns 1 row → driving row PRESERVED,
    /// ADR-192 D-8). Realized here as `UNWIND [0] AS RET` (always 1 row).
    fn one_row_body_factory() -> CallBodyFactory {
        k_row_body_factory(1)
    }

    /// A `VariableRef` bound expression over `b` (for an aggregate arg).
    fn var_ref(b: BindingId) -> BoundExpression {
        BoundExpression::VariableRef {
            name: "x".into(),
            binding_id: b,
            span: Span::point(1, 1),
            type_info: None,
        }
    }

    /// A body factory whose body AGGREGATES `n` correlated-expansion rows
    /// down to EXACTLY ONE output row: `UNWIND [0..n] AS x` (`n` rows) →
    /// `count(x)` (1 row, no GROUP BY). The body's **INPUT** cardinality is
    /// `n`; its **OUTPUT** cardinality is `1`. This is the ADR-192 D-8
    /// hinge shape — the aggregating collapse where input ≠ output, so a
    /// CallOp that measures `k` on the body OUTPUT (correct, D-6) and one
    /// that measures it on the body INPUT (the bug) give DIFFERENT row
    /// counts (1 vs `n`). Used by the discriminating D-8 hinge test.
    fn agg_body_factory(n: usize) -> CallBodyFactory {
        Box::new(move || {
            let values: Vec<i64> = (0..n as i64).collect();
            let unwind = PhysicalOperator::Unwind(UnwindOp::new(
                PhysicalOperator::Empty(EmptyOp::unit()),
                int_list_expr(&values),
                AGG_IN,
            ));
            Ok(PhysicalOperator::Aggregate(AggregateOp::new(
                unwind,
                Vec::new(), // no GROUP BY ⇒ a single aggregate output row
                vec![AggregateCall {
                    kind: AggregationKind::Count,
                    // #746: aggregate output id. This D-8 hinge test
                    // measures the body's OUTPUT row count (1), not the
                    // value under a specific id, so any stable id ≠ AGG_IN
                    // suffices.
                    output_id: BindingId::new(6),
                    arg: var_ref(AGG_IN),
                    distinct: false,
                    star: false,
                }],
            )))
        })
    }

    /// A body factory whose body ERRORS when first driven: `UNWIND` of a
    /// non-list scalar surfaces [`ExecutionError::Eval`]
    /// (`unwind.rs::unwind_non_list_error`) the moment [`CallOp`] drives
    /// it — AFTER the correlation frame has been pushed. The fault-
    /// injection fixture for the LOW-1 frame-leak-on-error guard.
    fn erroring_body_factory() -> CallBodyFactory {
        Box::new(|| {
            Ok(PhysicalOperator::Unwind(UnwindOp::new(
                PhysicalOperator::Empty(EmptyOp::unit()),
                BoundExpression::Literal {
                    value: Literal::Integer(5), // a scalar, NOT a list ⇒ drive-time Eval error
                    span: Span::point(1, 1),
                    type_info: None,
                },
                RET,
            )))
        })
    }

    /// A scan child binding `IMP` over label 1.
    fn scan_child() -> PhysicalOperator {
        PhysicalOperator::Scan(ScanOp::new(IMP, Some(LabelId::new(1)), Lsn::MAX))
    }

    fn two_node_substrate() -> StubExecutorSubstrate {
        StubExecutorSubstrate::new()
            .with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(1), Some(LabelId::new(1))),
            )
            .with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(2), Some(LabelId::new(1))),
            )
    }

    fn drain(
        op: &mut CallOp,
        ctx: &ExecutionContext,
        s: &StubExecutorSubstrate,
    ) -> Vec<Vec<Value>> {
        let mut rows = Vec::new();
        loop {
            let b = op.next_batch(ctx, s).expect("next_batch ok");
            if b.is_empty() {
                break;
            }
            rows.extend(b.into_rows());
        }
        rows
    }

    // ---- CorrelationSeedOp ------------------------------------------

    #[test]
    fn correlation_seed_emits_one_row_of_imported_values_then_eos() {
        let ctx = ctx();
        let s = StubExecutorSubstrate::new();
        ctx.push_correlation_frame(vec![(IMP, Value::Integer(42))]);
        let mut seed = CorrelationSeedOp::new(vec![IMP]);
        assert_eq!(seed.schema(), &[IMP]);
        let b1 = seed.next_batch(&ctx, &s).unwrap();
        assert_eq!(b1.rows(), &[vec![Value::Integer(42)]]);
        let b2 = seed.next_batch(&ctx, &s).unwrap();
        assert!(b2.is_empty(), "EOS after the single seed row");
        ctx.pop_correlation_frame();
    }

    #[test]
    fn correlation_seed_empty_import_is_a_unit_row() {
        // An uncorrelated CALL{} ⇒ zero-column one-row table (≡ unit row).
        let ctx = ctx();
        let s = StubExecutorSubstrate::new();
        ctx.push_correlation_frame(vec![]);
        let mut seed = CorrelationSeedOp::new(vec![]);
        let b1 = seed.next_batch(&ctx, &s).unwrap();
        assert_eq!(b1.row_count(), 1);
        assert_eq!(b1.column_count(), 0);
        assert!(seed.next_batch(&ctx, &s).unwrap().is_empty());
        ctx.pop_correlation_frame();
    }

    #[test]
    fn correlation_seed_nearest_frame_wins() {
        // Nested frames: an inner frame shadows nothing it lacks; the
        // outer binding resolves from the lower frame (ADR-192 test 9
        // nesting). Stack: [outer{IMP=1}, inner{RET=2}]; seed over both.
        let ctx = ctx();
        let s = StubExecutorSubstrate::new();
        ctx.push_correlation_frame(vec![(IMP, Value::Integer(1))]);
        ctx.push_correlation_frame(vec![(RET, Value::Integer(2))]);
        let mut seed = CorrelationSeedOp::new(vec![IMP, RET]);
        let b = seed.next_batch(&ctx, &s).unwrap();
        assert_eq!(b.rows(), &[vec![Value::Integer(1), Value::Integer(2)]]);
        ctx.pop_correlation_frame();
        ctx.pop_correlation_frame();
    }

    // ---- CallOp cardinality (D-7 / D-8) -----------------------------

    #[test]
    fn call_drops_driving_row_when_body_returns_zero_rows() {
        // ADR-192 D-7 / test 4: a NON-aggregating body returning 0 rows
        // for a driving row ⇒ that driving row is DROPPED (inner
        // correlation, not optional). 2 driving rows, empty body ⇒ 0 out.
        let s = two_node_substrate();
        let ctx = ctx();
        let mut op = CallOp::new(scan_child(), empty_body_factory(), vec![IMP], vec![RET]);
        let rows = drain(&mut op, &ctx, &s);
        assert_eq!(rows, Vec::<Vec<Value>>::new(), "both driving rows dropped");
    }

    #[test]
    fn call_preserves_driving_row_when_nonaggregating_body_returns_one_row() {
        // PRESERVE SMOKE (NOT the D-8 discriminator). A body that returns
        // exactly 1 row preserves the driving row. Modeled as `UNWIND [0]`
        // — body INPUT = 1, body OUTPUT = 1. Because input == output here,
        // this case CANNOT distinguish a measure-on-OUTPUT CallOp (correct,
        // D-6) from a measure-on-INPUT one — BOTH yield ×1. It is a smoke
        // that "1 body row ⇒ driving row kept", nothing more. The genuine
        // D-8 hinge — where body INPUT ≠ OUTPUT so the bug is observable —
        // is `call_d8_hinge_measures_body_output_not_input_aggregating_collapse`
        // below. (The complementary divergent case is the DROP test
        // `call_drops_driving_row_when_body_returns_zero_rows`: input 1,
        // output 0.) 2 driving rows × 1 body row ⇒ 2 out.
        let s = two_node_substrate();
        let ctx = ctx();
        let mut op = CallOp::new(scan_child(), one_row_body_factory(), vec![IMP], vec![RET]);
        let rows = drain(&mut op, &ctx, &s);
        assert_eq!(rows.len(), 2, "both driving rows PRESERVED");
        for r in &rows {
            assert_eq!(r.len(), 2, "schema = [driving node, returned column]");
            assert!(matches!(r[0], Value::Node(_)), "driving node preserved");
            assert_eq!(r[1], Value::Integer(0), "appended body column");
        }
    }

    #[test]
    fn call_d8_hinge_measures_body_output_not_input_aggregating_collapse() {
        // ADR-192 D-8 (the correctness hinge) — STRONG-ORACLE form.
        //
        // An AGGREGATING body collapses its correlated-expansion INPUT
        // (here N=3 rows) into ONE OUTPUT row (the aggregate result row).
        // `CallOp` measures the join cardinality `k` on the body's OUTPUT
        // (D-6), so each driving row is preserved EXACTLY ONCE (×1) — NOT
        // ×N. This is the D-8 case that the empty-correlated-set identity
        // row (`count(*) → 0` preserving the row) generalizes: the body
        // returns 1 row, so the driving row survives once.
        //
        // WHY THIS IS THE DISCRIMINATOR (and the prior `UNWIND [0]` shape
        // was not): the body INPUT (3) DIVERGES from the body OUTPUT (1).
        // A naive CallOp that measured `k` on the body INPUT (the
        // correlated expansion) would emit each driving row 3× (WRONG);
        // measuring on OUTPUT emits it 1× (RIGHT). The two impls give
        // DIFFERENT answers here, so this test CAN fail on the bug — the
        // strong-oracle property the input=output=1 smoke lacked. Verified
        // RED under a temporary measure-on-input CallOp stub (draining the
        // aggregate's child, the 3-row expansion) → 6 rows; GREEN on the
        // shipped measure-on-output impl → 2 rows (see #744 R1 fix-up
        // report). The appended `count == 3` simultaneously proves the
        // body genuinely consumed 3 input rows (the divergence is real,
        // not a degenerate N=1).
        let s = two_node_substrate();
        let ctx = ctx();
        let mut op = CallOp::new(scan_child(), agg_body_factory(3), vec![IMP], vec![RET]);
        let rows = drain(&mut op, &ctx, &s);
        assert_eq!(
            rows.len(),
            2,
            "2 driving rows × 1 aggregate-output row ⇒ 2 (each preserved ONCE, not ×3)"
        );
        let mut node_ids: Vec<u64> = Vec::new();
        for r in &rows {
            assert_eq!(r.len(), 2, "schema = [driving node, returned aggregate]");
            match &r[0] {
                Value::Node(n) => node_ids.push(n.id.raw()),
                o => panic!("col0 must be the preserved driving node, got {o:?}"),
            }
            assert_eq!(
                r[1],
                Value::Integer(3),
                "appended aggregate = count of the 3-row body INPUT (output cardinality 1)"
            );
        }
        node_ids.sort_unstable();
        assert_eq!(
            node_ids,
            vec![1, 2],
            "each of the 2 driving rows preserved EXACTLY once (×1, not ×3 ⇒ no node duplicated)"
        );
    }

    #[test]
    fn call_multiplies_driving_row_by_body_cardinality() {
        // ADR-192 D-7 / test 3: a body returning k=3 rows ⇒ each driving
        // row repeated 3×, each joined to one body row. 2 driving × 3 ⇒ 6.
        let s = two_node_substrate();
        let ctx = ctx();
        let mut op = CallOp::new(scan_child(), k_row_body_factory(3), vec![IMP], vec![RET]);
        let rows = drain(&mut op, &ctx, &s);
        assert_eq!(rows.len(), 6, "2 driving rows × 3 body rows");
        // Group the appended body element per driving node.
        let mut per_node: std::collections::BTreeMap<u64, Vec<i64>> = Default::default();
        for r in &rows {
            let id = match &r[0] {
                Value::Node(n) => n.id.raw(),
                o => panic!("col0 must be the preserved driving node, got {o:?}"),
            };
            let e = match &r[1] {
                Value::Integer(e) => *e,
                o => panic!("col1 must be the body element, got {o:?}"),
            };
            per_node.entry(id).or_default().push(e);
        }
        for (_id, mut es) in per_node {
            es.sort_unstable();
            assert_eq!(es, vec![0, 1, 2], "each driving row sees the full body");
        }
    }

    #[test]
    fn call_schema_is_child_plus_returned() {
        let op = CallOp::new(scan_child(), one_row_body_factory(), vec![IMP], vec![RET]);
        assert_eq!(op.schema(), &[IMP, RET]);
    }

    // ---- cross-batch Pending carry-over (D-6 / test 11) -------------

    #[test]
    fn call_crosses_batch_boundary_without_truncation_or_duplication() {
        // A single driving row whose subquery body emits MORE rows than
        // fit in one output batch — the pending cursor resumes EXACTLY
        // across `next_batch` calls (no row dropped or duplicated). One
        // driving node, body of BATCH_ROWS + 5 rows.
        let s = StubExecutorSubstrate::new().with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(1), Some(LabelId::new(1))),
        );
        let ctx = ctx();
        let k = BATCH_ROWS + 5;
        let mut op = CallOp::new(scan_child(), k_row_body_factory(k), vec![IMP], vec![RET]);

        let b1 = op.next_batch(&ctx, &s).unwrap();
        assert_eq!(b1.row_count(), BATCH_ROWS, "first batch exactly full");
        let b2 = op.next_batch(&ctx, &s).unwrap();
        assert_eq!(
            b2.row_count(),
            5,
            "carry-over remainder in the second batch"
        );
        let b3 = op.next_batch(&ctx, &s).unwrap();
        assert!(b3.is_empty(), "EOS after the body is exhausted");

        // Every body element exactly once, in order (no trunc / no dup).
        let mut all: Vec<i64> = Vec::with_capacity(k);
        for r in b1.rows().iter().chain(b2.rows().iter()) {
            match r[1] {
                Value::Integer(v) => all.push(v),
                ref o => panic!("integer body element, got {o:?}"),
            }
        }
        let expected: Vec<i64> = (0..k as i64).collect();
        assert_eq!(all, expected, "every body row exactly once, in order");
    }

    // ---- correlation-frame balance ----------------------------------

    #[test]
    fn call_balances_correlation_frames_after_full_drain() {
        // After a full drive, the correlation-frame stack is empty (every
        // pushed frame popped) — the push/pop balance invariant. We can
        // observe it indirectly: a fresh CorrelationSeedOp over IMP after
        // the drain finds NO frame ⇒ NULL.
        let s = two_node_substrate();
        let ctx = ctx();
        let mut op = CallOp::new(scan_child(), one_row_body_factory(), vec![IMP], vec![RET]);
        let _ = drain(&mut op, &ctx, &s);
        let mut probe = CorrelationSeedOp::new(vec![IMP]);
        let b = probe.next_batch(&ctx, &s).unwrap();
        assert_eq!(
            b.rows(),
            &[vec![Value::Null]],
            "all frames popped ⇒ no imported value resolves"
        );
    }

    #[test]
    fn call_pops_frame_when_body_errors_mid_drive_low1() {
        // LOW-1 (fault injection): the correlation frame is pushed for a
        // driving row (step 3) BEFORE the body is driven (step 1). If the
        // body ERRORS mid-drive — here `UNWIND` of a non-list scalar ⇒
        // `ExecutionError::Eval` — a naive `?` propagation would strand
        // that frame on the `ExecutionContext` correlation stack. The
        // error-path guard MUST pop it so the stack stays balanced.
        //
        // STRONG ORACLE: reverting the guard makes this RED —
        // `correlation_value(IMP)` then resolves the leaked frame's node
        // instead of `None`. (#744 R1 fix-up.)
        let s = two_node_substrate();
        let ctx = ctx();
        let mut op = CallOp::new(scan_child(), erroring_body_factory(), vec![IMP], vec![RET]);
        let r = op.next_batch(&ctx, &s);
        assert!(
            matches!(r, Err(ExecutionError::Eval(_))),
            "body drive must error (non-list UNWIND); got {r:?}"
        );
        // The frame for the driving row must have been popped: with the
        // stack balanced, no active frame provides `IMP` ⇒ None.
        assert_eq!(
            ctx.correlation_value(IMP),
            None,
            "frame popped on the body-error path ⇒ correlation stack balanced (no leak)"
        );
    }

    // ---- cancellation -----------------------------------------------

    #[test]
    fn call_propagates_cancellation() {
        let s = two_node_substrate();
        let ctx = ctx();
        ctx.cancellation().cancel();
        let mut op = CallOp::new(scan_child(), one_row_body_factory(), vec![IMP], vec![RET]);
        assert_eq!(op.next_batch(&ctx, &s), Err(ExecutionError::Cancelled));
    }

    #[test]
    fn call_with_empty_child_emits_nothing() {
        // No driving rows ⇒ the subquery never runs ⇒ no output.
        let s = StubExecutorSubstrate::new(); // no nodes ⇒ scan is empty
        let ctx = ctx();
        let mut op = CallOp::new(scan_child(), one_row_body_factory(), vec![IMP], vec![RET]);
        assert_eq!(drain(&mut op, &ctx, &s), Vec::<Vec<Value>>::new());
    }
}
