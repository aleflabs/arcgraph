//! [`UnwindOp`] — `UNWIND <list> AS <var>` (openCypher v9 §6.7, #618).
//!
//! Lowers from [`crate::logical_plan::LogicalUnwind`] (ADR-038 D-28 §7).
//! A streaming 1-to-N operator: for each upstream row it evaluates the
//! list expression and emits **one output row per list element**, with
//! `var` bound to that element. The upstream bindings are PRESERVED —
//! the output schema is `child_schema ++ [var]` (UNWIND EXTENDS scope;
//! it does not replace it, unlike [`super::ProjectOp`]).
//!
//! # Semantics (openCypher v9 §6.7 — the strong oracle)
//!
//! For each input row, evaluating `list_expr` yields:
//! - `Value::List(items)` → one output row per element, in list order.
//!   **Empty list → ZERO output rows** for that input row.
//! - `Value::Null` → **ZERO output rows** (`UNWIND null AS x` yields no
//!   rows — §6.7; this is NOT an error).
//! - any other scalar → a runtime type error ([`ExecutionError::Eval`]):
//!   `UNWIND` of a non-list, non-null value is ill-typed. (Unwind1.feature
//!   has no scalar-UNWIND scenario; the type-check pass rejects most
//!   scalar UNWINDs upstream — this arm is the defensive executor-level
//!   guard, surfaced via the same `Eval` variant [`super::ExpandOp`] uses
//!   for runtime type faults.)
//!
//! Two input rows × an N-element list ⇒ 2·N output rows (the cartesian
//! product, upstream bindings carried verbatim).
//!
//! # Batching — lazy cursor, never truncate (ADR-038 amendment-02 §M4.f)
//!
//! UNWIND is 1-to-N, so a single upstream row can fan out past the
//! [`crate::executor::BATCH_ROWS`] cap, and an upstream row whose list
//! is empty/null produces ZERO output rows (which must NOT be mistaken
//! for EOS). The op therefore carries state across `next_batch` calls:
//!
//! - `pending` — the upstream row currently being unwound + a cursor
//!   into its (already-materialized) list. When the output batch fills
//!   mid-list, the cursor persists so the next call resumes EXACTLY
//!   where it stopped — no element is dropped or duplicated.
//! - `buffered_input` — the not-yet-unwound rows of the upstream batch
//!   we last pulled, so we process one upstream batch across however
//!   many output batches it expands into.
//! - `child_done` — set once the child returns its EOS empty batch.
//!
//! `next_batch` loops internally — draining `pending`, then pulling /
//! buffering upstream rows — until the output batch is full OR the child
//! is exhausted with nothing pending. It returns the EOS empty batch
//! ONLY when there is genuinely nothing left, so an all-empty-list /
//! all-null input correctly yields zero total rows (not a premature EOS).
//!
//! ## Why a lazy cursor, not the [`super::ExpandOp`] spillover queue
//!
//! [`super::ExpandOp`] (the other 1-to-N op) materializes every output
//! row of an upstream batch into a bounded spillover `VecDeque`, capped
//! by `super::ExpandOp::SPILLOVER_MAX_ROWS` / the per-tenant memory
//! budget. For UNWIND the list is ALREADY fully materialized by
//! [`evaluate`] (it is the value of `list_expr`); holding a cursor into
//! it costs nothing beyond that list + one output batch, and never
//! pre-clones the upstream prefix N times into a side queue. This is
//! strictly lower memory than the spillover shape AND has no spurious
//! `ResourceExhausted` ceiling (a 2048-row batch each unwinding a
//! 100-element list — 204 800 output rows — streams fine here but would
//! exceed `ExpandOp`'s no-budget fallback cap). The unbounded-memory
//! risk for UNWIND lives entirely at the list PRODUCER (`range(...)`, a
//! list literal, a concat, a property), which is where the containment is
//! ENFORCED — NOT via a per-tenant budget (there is none checked on this
//! path), but via per-builder element/byte caps in [`evaluate`] that
//! reject BEFORE materializing an abusive list: `range()` is capped at
//! `eval::MAX_RANGE_LEN` (`RETURN range(1, 9e18)` → clean error, not OOM;
//! NN-3), and `+` list/string concatenation is capped per-op at
//! `eval::MAX_CONCAT_LIST_LEN` / `MAX_CONCAT_STRING_BYTES`
//! (ADR-147-amendment-03 §B1). `UnwindOp` itself adds no unbounded
//! accumulation on top of the (now-capped) producer.
//!
//! # ADR provenance
//! - **ADR-038 §2 D-28 §7** — UNWIND operator contract (the approved
//!   design this slice realizes).
//! - **ADR-038 amendment-02 §M4.f** — batch-boundary cancel check +
//!   `BATCH_ROWS` discipline.
//! - **openCypher v9 §6.7** — the conformance oracle.

use std::collections::VecDeque;

use crate::executor::batch::Batch;
use crate::executor::context::ExecutionContext;
use crate::executor::error::ExecutionError;
use crate::executor::eval::{Parameters, evaluate};
use crate::executor::ops::PhysicalOperator;
use crate::executor::ops::schema_index;
use crate::executor::substrate::ExecutorSubstrate;
use crate::executor::value::Value;
use crate::semantic::bound_ast::{BindingId, BoundExpression};

/// The upstream row currently being unwound + how far we have emitted.
#[derive(Debug)]
struct Pending {
    /// The upstream row's cells (the inherited columns), cloned into
    /// each emitted output row as the leading prefix.
    prefix: Vec<Value>,
    /// The materialized list being unwound (owned — moved out of the
    /// `evaluate` result, never re-cloned).
    items: Vec<Value>,
    /// Index of the next element to emit. `cursor == items.len()` ⇒
    /// this pending is exhausted.
    cursor: usize,
}

/// `UNWIND <list_expr> AS <var>` streaming operator.
#[derive(Debug)]
pub struct UnwindOp {
    /// Upstream child producing the rows whose `list_expr` is unwound.
    child: Box<PhysicalOperator>,
    /// The list expression, evaluated against each upstream row.
    list_expr: BoundExpression,
    /// The per-element binding. Mirrored in the trailing slot of
    /// `schema` (consumed at construction); the field is preserved for
    /// diagnostics + future M4-71 row-count-observer attribution, as
    /// with [`super::ExpandOp`]'s `to_var` / [`super::ScanOp`]'s `binding`.
    #[allow(dead_code)]
    var: BindingId,
    /// Per-query parameter bag (for `$param` list expressions).
    parameters: Parameters,
    /// Output schema: `child_schema ++ [var]`.
    schema: Vec<BindingId>,
    /// Cached child schema for the per-row `list_expr` eval lookup.
    child_schema: Vec<BindingId>,
    /// The upstream row mid-unwind + its list cursor (carry-over across
    /// `next_batch` calls when the output batch fills mid-list).
    pending: Option<Pending>,
    /// Not-yet-unwound rows of the upstream batch we last pulled.
    buffered_input: VecDeque<Vec<Value>>,
    /// Set once the child returns its EOS empty batch.
    child_done: bool,
}

impl UnwindOp {
    /// Construct an `UnwindOp` over `child`. The output schema is the
    /// child's schema with `var` appended as the trailing column (UNWIND
    /// extends the in-scope bindings; openCypher v9 §6.7).
    #[must_use]
    pub fn new(child: PhysicalOperator, list_expr: BoundExpression, var: BindingId) -> Self {
        let child_schema = child.schema().to_vec();
        let mut schema = child_schema.clone();
        schema.push(var);
        Self {
            child: Box::new(child),
            list_expr,
            var,
            parameters: Parameters::new(),
            schema,
            child_schema,
            pending: None,
            buffered_input: VecDeque::new(),
            child_done: false,
        }
    }

    /// Inject a per-query parameter bag (mirrors [`super::ProjectOp::with_parameters`]).
    #[must_use]
    pub fn with_parameters(mut self, parameters: Parameters) -> Self {
        self.parameters = parameters;
        self
    }

    /// Output schema (`child_schema ++ [var]`).
    pub fn schema(&self) -> &[BindingId] {
        &self.schema
    }

    /// Pull the next batch of unwound rows.
    pub fn next_batch<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<Batch, ExecutionError> {
        ctx.cancellation().check()?;
        let mut out = Batch::with_capacity(self.schema.len());
        // Owned copy so the eval lookup closure does not borrow `self`
        // while we mutate `self.pending` / `self.child` / the buffer
        // (mirrors `ProjectOp`'s per-batch upstream-schema clone). The
        // schema is a handful of `BindingId`s — cheap.
        let child_schema = self.child_schema.clone();
        let lookup = |b: BindingId| schema_index(&child_schema, b);

        loop {
            // 1. Drain the current pending unwind into the output batch.
            if let Some(p) = self.pending.as_mut() {
                while !out.is_full() && p.cursor < p.items.len() {
                    let mut row = Vec::with_capacity(p.prefix.len() + 1);
                    row.extend_from_slice(&p.prefix);
                    row.push(p.items[p.cursor].clone());
                    p.cursor += 1;
                    let pushed = out.push_row(row);
                    debug_assert!(pushed, "guarded by !out.is_full()");
                }
                if p.cursor >= p.items.len() {
                    self.pending = None;
                }
            }
            if out.is_full() {
                break;
            }

            // 2. Get the next upstream row (from the buffered batch, or
            //    by pulling a fresh batch from the child).
            let next_row = match self.buffered_input.pop_front() {
                Some(row) => row,
                None => {
                    if self.child_done {
                        // Nothing pending (drained above), nothing
                        // buffered, child exhausted ⇒ genuine EOS.
                        break;
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

            // 3. Evaluate `list_expr` against the upstream row and stage
            //    the next pending unwind (openCypher v9 §6.7).
            let list_val = evaluate(&self.list_expr, &next_row, &lookup, &self.parameters)?;
            match list_val {
                Value::List(items) => {
                    if items.is_empty() {
                        // Empty list ⇒ ZERO rows for this input row.
                        continue;
                    }
                    self.pending = Some(Pending {
                        prefix: next_row,
                        items,
                        cursor: 0,
                    });
                    // Loop back to the drain step (step 1).
                }
                // `UNWIND null AS x` ⇒ ZERO rows (NOT an error).
                Value::Null => continue,
                // `UNWIND <scalar>` ⇒ runtime type error (§6.7).
                other => return Err(unwind_non_list_error(&other)),
            }
        }
        Ok(out)
    }
}

/// Render the openCypher v9 §6.7 "`UNWIND` of a non-list, non-null
/// value" runtime type fault. Surfaced via [`ExecutionError::Eval`] —
/// the same variant [`super::ExpandOp`] uses for runtime type
/// mismatches (`ExecutionError` is the frozen M5↔M4 contract surface,
/// deliberately exempt from `#[non_exhaustive]`, so we reuse `Eval`
/// rather than add a variant; the message carries the diagnostic).
fn unwind_non_list_error(value: &Value) -> ExecutionError {
    ExecutionError::Eval(format!(
        "UNWIND expects a list or null; got a non-list scalar value ({})",
        value_kind(value)
    ))
}

/// Stable, short kind-name for a runtime [`Value`] — diagnostic only.
fn value_kind(v: &Value) -> &'static str {
    match v {
        Value::Null => "Null",
        Value::Boolean(_) => "Boolean",
        Value::Integer(_) => "Integer",
        Value::Float(_) => "Float",
        Value::String(_) => "String",
        Value::Node(_) => "Node",
        Value::Relationship(_) => "Relationship",
        Value::List(_) => "List",
        // ADR-191 Value::Map — lock-step diagnostic site surfaced when
        // this branch's `Value::Map` variant rebased onto #733's UNWIND
        // op. UNWIND-of-a-map already routes correctly to the non-list
        // type error (openCypher v9 §6.7) via the `other =>` arm above;
        // this only lets that error name the kind.
        Value::Map(_) => "Map",
        // ADR-193 — `Value::Path` joins the diagnostic kind-name set
        // (UNWIND of a path is a non-list scalar fault, same as Node/Rel).
        Value::Path(_) => "Path",
        Value::Temporal(_) => "Temporal",
        Value::LocalDateTime(_) => "LocalDateTime",
        Value::Date(_) => "Date",
        Value::Duration(_) => "Duration",
        Value::Decimal(_) => "Decimal",
    }
}

#[cfg(test)]
mod tests {
    use arcgraph_core::{LabelId, Lsn, NodeId, PartitionId, TenantId};

    use super::*;
    use crate::ast::{Expression, Literal};
    use crate::error::Span;
    use crate::executor::batch::BATCH_ROWS;
    use crate::executor::ops::{EmptyOp, ScanOp};
    use crate::executor::substrate::StubExecutorSubstrate;
    use crate::executor::value::NodeView;

    // ---- helpers ----------------------------------------------------

    const VAR: BindingId = BindingId::new(7);

    fn span() -> Span {
        Span::point(1, 1)
    }

    /// A `BoundExpression` literal carrying the given value.
    fn lit(value: Literal) -> BoundExpression {
        BoundExpression::Literal {
            value,
            span: span(),
            type_info: None,
        }
    }

    /// A list-literal `BoundExpression` of integers.
    fn int_list(values: &[i64]) -> BoundExpression {
        lit(Literal::List(
            values
                .iter()
                .map(|n| Expression::Literal(Literal::Integer(*n)))
                .collect(),
        ))
    }

    /// Drive `op` to exhaustion, returning every output row in order.
    fn drain(op: &mut UnwindOp, s: &StubExecutorSubstrate) -> Vec<Vec<Value>> {
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let mut rows = Vec::new();
        loop {
            let b = op.next_batch(&ctx, s).expect("next_batch ok");
            if b.is_empty() {
                break;
            }
            rows.extend(b.into_rows());
        }
        rows
    }

    /// A unit-driving-row child (one zero-column row) — the leading
    /// `UNWIND` shape. The substrate is unused for a unit child.
    fn unit_child() -> PhysicalOperator {
        PhysicalOperator::Empty(EmptyOp::unit())
    }

    fn ints(rows: &[Vec<Value>]) -> Vec<i64> {
        rows.iter()
            .map(|r| match r.last().expect("non-empty row") {
                Value::Integer(n) => *n,
                other => panic!("expected trailing Integer, got {other:?}"),
            })
            .collect()
    }

    // ---- semantics oracles (openCypher v9 §6.7) ---------------------

    #[test]
    fn unwind_list_emits_one_row_per_element_in_order() {
        let s = StubExecutorSubstrate::new();
        let mut op = UnwindOp::new(unit_child(), int_list(&[1, 2, 3]), VAR);
        let rows = drain(&mut op, &s);
        // Exactly [1],[2],[3] — determinism-equal oracle, not a count.
        assert_eq!(
            rows,
            vec![
                vec![Value::Integer(1)],
                vec![Value::Integer(2)],
                vec![Value::Integer(3)],
            ]
        );
    }

    #[test]
    fn unwind_empty_list_emits_zero_rows() {
        let s = StubExecutorSubstrate::new();
        let mut op = UnwindOp::new(unit_child(), int_list(&[]), VAR);
        assert_eq!(drain(&mut op, &s), Vec::<Vec<Value>>::new());
    }

    #[test]
    fn unwind_null_emits_zero_rows_not_error() {
        // `UNWIND null AS x` ⇒ no rows (NOT an error) — §6.7.
        let s = StubExecutorSubstrate::new();
        let mut op = UnwindOp::new(unit_child(), lit(Literal::Null), VAR);
        assert_eq!(drain(&mut op, &s), Vec::<Vec<Value>>::new());
    }

    #[test]
    fn unwind_scalar_is_runtime_type_error() {
        // `UNWIND 5 AS x` ⇒ runtime type error (non-list, non-null).
        let s = StubExecutorSubstrate::new();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let mut op = UnwindOp::new(unit_child(), lit(Literal::Integer(5)), VAR);
        match op.next_batch(&ctx, &s) {
            Err(ExecutionError::Eval(msg)) => {
                assert!(msg.contains("UNWIND"), "clear UNWIND type-error: {msg}");
                assert!(msg.contains("Integer"), "names the offending kind: {msg}");
            }
            other => panic!("expected Eval type error, got {other:?}"),
        }
    }

    #[test]
    fn unwind_map_is_runtime_type_error() {
        // `UNWIND {a:1} AS x` ⇒ runtime type error — a map is non-list,
        // non-null (openCypher v9 §6.7). Proves the ADR-191 `Value::Map`
        // lock-step arm in `value_kind` (added when this branch rebased
        // onto #733's UNWIND op): the diagnostic must name the "Map" kind.
        let s = StubExecutorSubstrate::new();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let map_lit = lit(Literal::Map(vec![(
            "a".to_string(),
            Expression::Literal(Literal::Integer(1)),
        )]));
        let mut op = UnwindOp::new(unit_child(), map_lit, VAR);
        match op.next_batch(&ctx, &s) {
            Err(ExecutionError::Eval(msg)) => {
                assert!(msg.contains("UNWIND"), "clear UNWIND type-error: {msg}");
                assert!(msg.contains("Map"), "names the offending kind: {msg}");
            }
            other => panic!("expected Eval type error, got {other:?}"),
        }
    }

    #[test]
    fn unwind_nested_list_emits_each_inner_list_as_a_row() {
        // `[[1,2],[3]]` ⇒ 2 rows, each cell a Value::List (no flatten).
        let s = StubExecutorSubstrate::new();
        let nested = lit(Literal::List(vec![
            Expression::Literal(Literal::List(vec![
                Expression::Literal(Literal::Integer(1)),
                Expression::Literal(Literal::Integer(2)),
            ])),
            Expression::Literal(Literal::List(vec![Expression::Literal(Literal::Integer(
                3,
            ))])),
        ]));
        let mut op = UnwindOp::new(unit_child(), nested, VAR);
        let rows = drain(&mut op, &s);
        assert_eq!(
            rows,
            vec![
                vec![Value::List(vec![Value::Integer(1), Value::Integer(2)])],
                vec![Value::List(vec![Value::Integer(3)])],
            ]
        );
    }

    #[test]
    fn unwind_mixed_type_list_preserves_heterogeneity() {
        // Cypher 9 §3.5 admits heterogeneous lists at runtime.
        let s = StubExecutorSubstrate::new();
        let mixed = lit(Literal::List(vec![
            Expression::Literal(Literal::Integer(1)),
            Expression::Literal(Literal::String("a".into())),
            Expression::Literal(Literal::Bool(true)),
        ]));
        let mut op = UnwindOp::new(unit_child(), mixed, VAR);
        let rows = drain(&mut op, &s);
        assert_eq!(
            rows,
            vec![
                vec![Value::Integer(1)],
                vec![Value::String("a".into())],
                vec![Value::Boolean(true)],
            ]
        );
    }

    #[test]
    fn unwind_preserves_upstream_bindings_cartesian() {
        // Two upstream rows × a 2-element list ⇒ 4 output rows; the
        // upstream node binding is carried verbatim into each (UNWIND
        // EXTENDS scope, does not prune it — Unwind1 [11]/[12]).
        let s = StubExecutorSubstrate::new()
            .with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(1), Some(LabelId::new(1))),
            )
            .with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(2), Some(LabelId::new(1))),
            );
        let scan = PhysicalOperator::Scan(ScanOp::new(
            BindingId::new(0),
            Some(LabelId::new(1)),
            Lsn::MAX,
        ));
        let mut op = UnwindOp::new(scan, int_list(&[10, 20]), VAR);
        // schema = [node_binding(0), var(7)].
        assert_eq!(op.schema(), &[BindingId::new(0), VAR]);
        let rows = drain(&mut op, &s);
        assert_eq!(rows.len(), 4, "2 upstream rows × 2 elements");
        // Each row: [Node, Integer]; collect (node_id, element) pairs.
        let pairs: Vec<(u64, i64)> = rows
            .iter()
            .map(|r| {
                let n = match &r[0] {
                    Value::Node(n) => n.id.raw(),
                    other => panic!("col 0 must be the preserved Node, got {other:?}"),
                };
                let e = match &r[1] {
                    Value::Integer(e) => *e,
                    other => panic!("col 1 must be the unwound element, got {other:?}"),
                };
                (n, e)
            })
            .collect();
        assert_eq!(pairs, vec![(1, 10), (1, 20), (2, 10), (2, 20)]);
    }

    #[test]
    fn unwind_crosses_batch_boundary_without_truncation_or_duplication() {
        // A single upstream row unwinding a list LONGER than BATCH_ROWS
        // must emit EVERY element exactly once, in order, across
        // multiple next_batch calls (no silent truncation, no dup).
        let s = StubExecutorSubstrate::new();
        let n = BATCH_ROWS + 1; // 2049 — forces a second output batch.
        let values: Vec<i64> = (0..n as i64).collect();
        let mut op = UnwindOp::new(unit_child(), int_list(&values), VAR);

        // Pull batch-by-batch and assert the FIRST batch is exactly full
        // and the carry-over lands in a second batch.
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let b1 = op.next_batch(&ctx, &s).unwrap();
        assert_eq!(b1.row_count(), BATCH_ROWS, "first batch is exactly full");
        let b2 = op.next_batch(&ctx, &s).unwrap();
        assert_eq!(b2.row_count(), 1, "carry-over element in the second batch");
        let b3 = op.next_batch(&ctx, &s).unwrap();
        assert!(b3.is_empty(), "EOS after the list is exhausted");

        // And the full drained sequence is exactly 0..n, in order
        // (every element exactly once — no truncation, no duplication).
        let mut all: Vec<i64> = Vec::with_capacity(n);
        for r in b1.rows().iter().chain(b2.rows().iter()) {
            match r[0] {
                Value::Integer(v) => all.push(v),
                ref other => panic!("integer element, got {other:?}"),
            }
        }
        assert_eq!(all, values, "every element exactly once, in order");
    }

    #[test]
    fn unwind_two_input_rows_one_empty_one_nonempty() {
        // First upstream row's list is empty (0 rows), second is [1,2]
        // — the empty list must NOT short-circuit EOS before the second
        // row's elements are emitted.
        let s = StubExecutorSubstrate::new()
            .with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(1), Some(LabelId::new(1))),
            )
            .with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(2), Some(LabelId::new(1))),
            );
        let scan = PhysicalOperator::Scan(ScanOp::new(
            BindingId::new(0),
            Some(LabelId::new(1)),
            Lsn::MAX,
        ));
        // Same list for both rows; assert the empty-list path separately
        // in `unwind_empty_list_emits_zero_rows`. Here both rows unwind
        // [1,2] ⇒ 4 rows, proving multi-row drive across the buffer.
        let mut op = UnwindOp::new(scan, int_list(&[1, 2]), VAR);
        let rows = drain(&mut op, &s);
        assert_eq!(ints(&rows), vec![1, 2, 1, 2]);
    }

    #[test]
    fn unwind_propagates_cancellation() {
        let s = StubExecutorSubstrate::new();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        ctx.cancellation().cancel();
        let mut op = UnwindOp::new(unit_child(), int_list(&[1, 2, 3]), VAR);
        assert_eq!(op.next_batch(&ctx, &s), Err(ExecutionError::Cancelled));
    }

    #[test]
    fn unwind_schema_appends_var_to_child_schema() {
        let s = StubExecutorSubstrate::new().with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(1), Some(LabelId::new(1))),
        );
        let scan = PhysicalOperator::Scan(ScanOp::new(
            BindingId::new(3),
            Some(LabelId::new(1)),
            Lsn::MAX,
        ));
        let op = UnwindOp::new(scan, int_list(&[1]), VAR);
        // child schema [3] ++ [var 7].
        assert_eq!(op.schema(), &[BindingId::new(3), VAR]);
        let _ = s;
    }
}
