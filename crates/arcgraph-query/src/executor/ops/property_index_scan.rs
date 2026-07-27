//! [`PropertyIndexScanOp`] — #1366 (Phase 2) indexed point-lookup op.
//!
//! Lowers from [`crate::logical_plan::LogicalPropertyIndexScan`]. This is
//! the PAYOFF operator: instead of the `Scan(label) + Filter(prop=value)`
//! path that materializes `O(node_high_water)` rows and filters after the
//! fact, this op does a B+tree point lookup + MVCC-verify, emitting
//! `O(matches)` rows. It closes #1366's read-path OOM mechanism for
//! indexed anchors and the ~5183ms / ~820× Neo4j point-lookup A/B lead
//! (design §"Planner and executor wiring" + §"Closing #1366").
//!
//! # Candidate-then-verify (the load-bearing invariant, ADR-023)
//!
//! At first-batch time the op:
//!
//! 1. resolves the lookup [`Value`] from the plan's `value` expression
//!    against the per-query parameter bag (a literal or a `$param`);
//! 2. calls
//!    [`crate::executor::ExecutorSubstrate::property_index_lookup_with_context`],
//!    which returns the ALREADY MVCC-verified + deduplicated nodes for
//!    `(label, property = value)`. The seam owns the candidate lookup,
//!    the hydrate-through-snapshot, the label + property re-check, and
//!    the dedup-by-NodeId — **the index NEVER determines visibility**;
//!    stale / ghost / snapshot-invisible candidates are dropped there,
//!    not surfaced;
//! 3. applies the `residual` filter (OTHER predicates on the same
//!    binding), if any, over the verified rows — so a residual match is
//!    always over a live, snapshot-visible node;
//! 4. pages the surviving rows out in [`BATCH_ROWS`]-sized chunks.
//!
//! # Schema
//!
//! Output schema is `[var]` — a single binding (the node-pattern
//! variable), identical to [`super::ScanOp`], so a `PropertyIndexScan`
//! is a drop-in replacement for the anchor scan the planner removed.
//!
//! # Empty-result contract + the unkeyable-value SCAN FALLBACK (#1415)
//!
//! A KEYABLE value that is absent from the index (no candidates), or a
//! candidate set that is entirely stale / invisible, yields ZERO rows —
//! never an error. This matches the full-scan path's "no matching node ⇒
//! empty result" contract exactly (the identical-results correctness
//! gate).
//!
//! But an UNKEYABLE resolved value — one with no canonical index key: a
//! fractional / out-of-i64-range `Float`, a NEGATIVE `Integer`, a `List`
//! / `Map` — is a DIFFERENT case. It reaches this op because a `$param`
//! is admitted UNCONDITIONALLY at plan time (its runtime type is unknown
//! until it binds). For such a value the index lookup returns EMPTY, but
//! that empty is NOT the answer: a full scan's `Filter(prop = v)` still
//! matches it via `values_equal_3vl` (`10.5`, `[1,2]`, `-5` all compare).
//! Treating the empty index result as "no matches" would SILENTLY drop
//! rows (the #1415 REJECT-class wrong-results bug). So the op consults
//! [`crate::executor::ExecutorSubstrate::value_is_indexable`] BEFORE the
//! lookup and, when the value is NOT keyable, falls back to a Scan+Filter
//! over the label (`Self::scan_fallback`) — the SAME rows the un-routed
//! path would produce. Keyable values never take the fallback, so the
//! index fast-path (and its perf win) is untouched.

use arcgraph_core::{LabelId, Lsn};

use crate::executor::batch::{BATCH_ROWS, Batch};
use crate::executor::context::ExecutionContext;
use crate::executor::error::ExecutionError;
use crate::executor::eval::{Parameters, evaluate};
use crate::executor::ops::schema_index;
use crate::executor::substrate::{BoundNode, ExecutorSubstrate};
use crate::executor::three_vl::ThreeValued;
use crate::executor::value::Value;
use crate::semantic::bound_ast::{BindingId, BoundExpression};

/// Indexed point-lookup operator (candidate lookup → MVCC-verify →
/// residual → page).
#[derive(Debug)]
pub struct PropertyIndexScanOp {
    /// Variable bound by this lookup. Mirrored in `schema[0]` for
    /// per-batch layout; retained for diagnostic + observer attribution
    /// (mirrors [`super::ScanOp::binding`]).
    #[allow(dead_code)]
    binding: BindingId,
    /// The label the index is declared on.
    label: LabelId,
    /// The indexed property name (recheck key on the substrate side).
    property: String,
    /// The lookup value expression (literal or `$param`), resolved to a
    /// concrete [`Value`] at first-batch time.
    value_expr: BoundExpression,
    /// OTHER predicates on the same binding, applied as a post-verify
    /// filter. `None` when the index predicate is the whole filter.
    residual: Option<BoundExpression>,
    /// MVCC read LSN copied from the plan. v1.0-α stub substrates ignore
    /// it; production threads it into the hydrate path.
    plan_read_lsn: Lsn,
    /// Per-query parameter bag (for a `$param` lookup value + a residual
    /// referencing parameters).
    parameters: Parameters,
    /// Cached per-batch schema (length-1: just the binding).
    schema: Vec<BindingId>,
    /// Buffered verified rows. `None` until first-batch primes it.
    buffer: Option<Vec<BoundNode>>,
    /// Cursor into the buffer.
    cursor: usize,
}

impl PropertyIndexScanOp {
    /// Construct a fresh op from a
    /// [`crate::logical_plan::LogicalPropertyIndexScan`]'s fields.
    #[must_use]
    pub fn new(
        binding: BindingId,
        label: LabelId,
        property: String,
        value_expr: BoundExpression,
        residual: Option<BoundExpression>,
        plan_read_lsn: Lsn,
    ) -> Self {
        Self {
            binding,
            label,
            property,
            value_expr,
            residual,
            plan_read_lsn,
            parameters: Parameters::new(),
            schema: vec![binding],
            buffer: None,
            cursor: 0,
        }
    }

    /// Inject a per-query parameter bag. Default is empty.
    #[must_use]
    pub fn with_parameters(mut self, parameters: Parameters) -> Self {
        self.parameters = parameters;
        self
    }

    /// Output schema. Always `[binding]` (drop-in for [`super::ScanOp`]).
    pub fn schema(&self) -> &[BindingId] {
        &self.schema
    }

    /// Resolve the lookup value expression to a concrete [`Value`].
    ///
    /// The expression is a literal or a `$param` reference — it does NOT
    /// reference any row binding (the planner only routes an equality
    /// against a literal-or-parameter here), so it evaluates against an
    /// EMPTY row + a no-op schema-lookup. A NULL result (an unbound
    /// `$param` or an explicit `null`) means "no equality match is
    /// possible" — Cypher `x = null` is `null`, never `true` — so the
    /// lookup yields zero rows (handled by the caller).
    fn resolve_value(&self) -> Result<Value, ExecutionError> {
        let empty_row: Vec<Value> = Vec::new();
        let no_binding = |_b: BindingId| -> Option<usize> { None };
        evaluate(&self.value_expr, &empty_row, &no_binding, &self.parameters)
    }

    /// Apply the `residual` predicate over one verified row. Returns
    /// `true` when the row passes (or when there is no residual). Mirrors
    /// [`super::FilterOp`]'s scalar 3VL semantics: `False` / `Unknown`
    /// drop the row.
    fn residual_passes(&self, row: &[Value]) -> Result<bool, ExecutionError> {
        let Some(residual) = &self.residual else {
            return Ok(true);
        };
        let schema = self.schema.clone();
        let lookup = move |b: BindingId| schema_index(&schema, b);
        let v = evaluate(residual, row, &lookup, &self.parameters)?;
        Ok(ThreeValued::from_value(&v).passes_filter())
    }

    /// #1415 SCAN-FALLBACK. Reproduce the un-routed `Scan(label) +
    /// Filter(prop = value)` path for a resolved runtime value that has NO
    /// canonical index key (so [`ExecutorSubstrate::property_index_lookup_with_context`]
    /// would return an empty candidate set — which is NOT the correct
    /// answer for a scan-matchable value like `10.5` / `[1,2]` / `-5`).
    ///
    /// Scans the label through the SAME snapshot the index path uses, then
    /// filters each hydrated node by `n.<property> = value` under the exact
    /// engine `=` 3VL / `values_equal_3vl` coercion (by evaluating a
    /// synthesized `PropertyAccess = value_expr` equality — byte-identical
    /// to the planner's `Filter`), THEN applies the same `residual`. The
    /// result is exactly the set the full-scan path would produce.
    fn scan_fallback<S: ExecutorSubstrate>(
        &self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<Vec<BoundNode>, ExecutionError> {
        // The equality predicate the un-routed path would have kept as its
        // Filter: `n.<property> = value_expr`, rooted at THIS op's binding.
        // Evaluating it over a `[Node]` row reuses the identical engine `=`
        // semantics (`values_equal_3vl`) — so a Float/List/Map/negative-int
        // value matches exactly as the full scan does.
        let equality = self.fallback_equality_predicate();
        let scanned =
            substrate.scan_nodes_with_context(ctx, Some(self.label), self.plan_read_lsn)?;
        let mut rows = Vec::new();
        for bn in scanned {
            let row = vec![Value::Node(bn.node)];
            // Equality filter (engine `=`, 3VL: False/Unknown drop).
            let eq = evaluate(&equality, &row, &self.eq_binding_lookup(), &self.parameters)?;
            if !ThreeValued::from_value(&eq).passes_filter() {
                continue;
            }
            // AND the residual (other predicates on the same binding).
            if self.residual_passes(&row)? {
                rows.push(bn_from_row(row));
            }
        }
        Ok(rows)
    }

    /// Build the fallback equality `n.<property> = value_expr`, rooted at
    /// this op's binding — the exact predicate the un-routed
    /// `Scan + Filter` path carries.
    fn fallback_equality_predicate(&self) -> BoundExpression {
        use crate::semantic::bound_ast::BoundPropertyRef;
        let span = crate::error::Span::point(1, 1);
        let lhs = BoundExpression::PropertyAccess {
            base: Box::new(BoundExpression::VariableRef {
                name: String::new(),
                binding_id: self.binding,
                span: span.clone(),
                type_info: None,
            }),
            path: vec![BoundPropertyRef {
                name: self.property.clone(),
                property_id: None,
                span: span.clone(),
            }],
            span: span.clone(),
            type_info: None,
        };
        BoundExpression::BinaryOp {
            op: crate::ast::BinOp::Eq,
            lhs: Box::new(lhs),
            rhs: Box::new(self.value_expr.clone()),
            span,
            type_info: None,
        }
    }

    /// Binding→row-index lookup for the fallback equality: the op's schema
    /// is length-1 (`[binding]`), so `self.binding` maps to column 0.
    fn eq_binding_lookup(&self) -> impl Fn(BindingId) -> Option<usize> + '_ {
        let schema = self.schema.clone();
        move |b: BindingId| schema_index(&schema, b)
    }

    /// Pull the next batch.
    pub fn next_batch<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<Batch, ExecutionError> {
        // Defense-in-depth cancel check (dispatcher already checks).
        ctx.cancellation().check()?;

        // Lazy prime — acquire the snapshot LSN at FIRST batch (ADR-038
        // §2 D-18 rule 1) + do the candidate lookup + verify once.
        if self.buffer.is_none() {
            let _exec_lsn = ctx.ensure_snapshot_lsn();
            let value = self.resolve_value()?;
            // A NULL lookup value can never equality-match (Cypher 3VL)
            // → empty result, no substrate call.
            if matches!(value, Value::Null) {
                self.buffer = Some(Vec::new());
            } else if !substrate.value_is_indexable(&value) {
                // #1415 SCAN-FALLBACK (correctness): the resolved runtime
                // value has NO canonical index key — a fractional /
                // out-of-i64-range `Float`, a NEGATIVE `Integer`, a `List`
                // / `Map`. `property_index_lookup_with_context` would
                // return EMPTY for it (the lookup drops a `None`-key
                // value), which is NOT "no matches": a full scan's
                // `Filter(prop = v)` still matches such a value via
                // `values_equal_3vl` (e.g. `10.5`, `[1,2]`, `-5`). Because
                // a `$param` is admitted to this op UNCONDITIONALLY at
                // plan time (its runtime type is unknown until it binds),
                // the ONLY correct answer here is to reproduce the
                // un-routed path: Scan the label + Filter by the same
                // equality. Keyable values NEVER take this branch, so the
                // index fast-path (and its perf win) is untouched.
                self.buffer = Some(self.scan_fallback(ctx, substrate)?);
            } else {
                // The seam returns MVCC-verified + deduped nodes; the
                // index never determines visibility (candidate-then-
                // verify happens inside the seam). We then apply the
                // residual filter over the verified rows.
                let verified = substrate.property_index_lookup_with_context(
                    ctx,
                    self.label,
                    &self.property,
                    &value,
                    self.plan_read_lsn,
                )?;
                let mut rows = Vec::with_capacity(verified.len());
                for bn in verified {
                    let row = vec![Value::Node(bn.node)];
                    if self.residual_passes(&row)? {
                        rows.push(bn_from_row(row));
                    }
                }
                self.buffer = Some(rows);
            }
        }

        let buf = self.buffer.as_ref().expect("primed above");
        if self.cursor >= buf.len() {
            return Ok(Batch::empty(self.schema.len()));
        }
        let mut batch = Batch::with_capacity(self.schema.len());
        let take = (buf.len() - self.cursor).min(BATCH_ROWS);
        for node in &buf[self.cursor..self.cursor + take] {
            if !batch.push_row(vec![Value::Node(node.node.clone())]) {
                return Err(ExecutionError::Eval(
                    "PropertyIndexScanOp: batch overflow during sized push".into(),
                ));
            }
        }
        self.cursor += take;
        Ok(batch)
    }
}

/// Reconstruct a [`BoundNode`] from a single-cell verified row. The row
/// is `[Value::Node(view)]` by construction (the op's schema is
/// length-1); a non-Node cell is an internal invariant break (the seam
/// only returns node rows), so we defensively keep the node when present.
fn bn_from_row(row: Vec<Value>) -> BoundNode {
    match row.into_iter().next() {
        Some(Value::Node(node)) => BoundNode { node },
        // Unreachable: the op only ever builds `[Value::Node(_)]` rows.
        other => unreachable!("PropertyIndexScanOp row must be [Node], got {other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use arcgraph_core::{NodeId, PartitionId, TenantId};

    use super::*;
    use crate::ast::Literal;
    use crate::error::Span;
    use crate::executor::substrate::StubExecutorSubstrate;
    use crate::executor::value::NodeView;

    fn lit_str(s: &str) -> BoundExpression {
        BoundExpression::Literal {
            value: Literal::String(s.to_string()),
            span: Span::point(1, 1),
            type_info: None,
        }
    }

    /// Seed a stub with `n` User nodes each carrying a unique `email`,
    /// declare the index, and register each node as its own candidate.
    fn fixture(n: u64, label: LabelId) -> StubExecutorSubstrate {
        let mut s =
            StubExecutorSubstrate::new().with_property_index(TenantId::DEFAULT, label, "email");
        for i in 1..=n {
            let email = format!("u{i}@x.com");
            let node = NodeView::new(NodeId::new(i), Some(label))
                .with_property("email", Value::String(email.clone()));
            s = s
                .with_node(TenantId::DEFAULT, node)
                .with_property_index_candidate(
                    TenantId::DEFAULT,
                    label,
                    "email",
                    &Value::String(email),
                    NodeId::new(i),
                );
        }
        s
    }

    fn op(label: LabelId, value: &str) -> PropertyIndexScanOp {
        PropertyIndexScanOp::new(
            BindingId::new(0),
            label,
            "email".to_string(),
            lit_str(value),
            None,
            Lsn::MAX,
        )
    }

    #[test]
    fn point_lookup_emits_the_single_match() {
        let label = LabelId::new(1);
        let s = fixture(5, label);
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let mut op = op(label, "u3@x.com");
        let b = op.next_batch(&ctx, &s).unwrap();
        assert_eq!(b.row_count(), 1);
        let n = match &b.row(0)[0] {
            Value::Node(n) => n,
            _ => panic!("expected Node"),
        };
        assert_eq!(n.id, NodeId::new(3));
        // Second batch is EOS.
        assert!(op.next_batch(&ctx, &s).unwrap().is_empty());
    }

    #[test]
    fn absent_value_yields_empty() {
        let label = LabelId::new(1);
        let s = fixture(5, label);
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let mut op = op(label, "nobody@x.com");
        assert!(op.next_batch(&ctx, &s).unwrap().is_empty());
    }

    #[test]
    fn duplicate_candidate_slots_dedup_to_one_row() {
        // The Phase-1 insert-only + backfill overlap can leave the SAME
        // NodeId in the candidate list twice. The verify+dedup must emit
        // exactly ONE row.
        let label = LabelId::new(1);
        let email = "dup@x.com";
        let s = StubExecutorSubstrate::new()
            .with_property_index(TenantId::DEFAULT, label, "email")
            .with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(7), Some(label))
                    .with_property("email", Value::String(email.into())),
            )
            .with_property_index_candidate(
                TenantId::DEFAULT,
                label,
                "email",
                &Value::String(email.into()),
                NodeId::new(7),
            )
            // Second (duplicate) slot for the SAME node id.
            .with_property_index_candidate(
                TenantId::DEFAULT,
                label,
                "email",
                &Value::String(email.into()),
                NodeId::new(7),
            );
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let mut op = op(label, email);
        let b = op.next_batch(&ctx, &s).unwrap();
        assert_eq!(b.row_count(), 1, "duplicate slots must dedup to one row");
    }

    #[test]
    fn stale_candidate_pointing_to_wrong_property_is_dropped() {
        // A GHOST candidate: the B+tree slot points at a node whose
        // CURRENT property no longer equals the looked-up value (an
        // insert-only stale entry). The verify recheck must drop it.
        let label = LabelId::new(1);
        let looked_up = "old@x.com";
        let s = StubExecutorSubstrate::new()
            .with_property_index(TenantId::DEFAULT, label, "email")
            // Node 9's CURRENT email is "new@x.com" (property changed).
            .with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(9), Some(label))
                    .with_property("email", Value::String("new@x.com".into())),
            )
            // But a stale slot still lists node 9 under "old@x.com".
            .with_property_index_candidate(
                TenantId::DEFAULT,
                label,
                "email",
                &Value::String(looked_up.into()),
                NodeId::new(9),
            );
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let mut op = op(label, looked_up);
        assert!(
            op.next_batch(&ctx, &s).unwrap().is_empty(),
            "stale ghost candidate must be dropped by the verify recheck"
        );
    }

    #[test]
    fn candidate_with_wrong_label_is_dropped() {
        // A hash-collision analog: the candidate hydrates to a node that
        // carries a DIFFERENT label than the index is declared on. The
        // label recheck drops it.
        let index_label = LabelId::new(1);
        let other_label = LabelId::new(2);
        let email = "x@x.com";
        let s = StubExecutorSubstrate::new()
            .with_property_index(TenantId::DEFAULT, index_label, "email")
            .with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(11), Some(other_label))
                    .with_property("email", Value::String(email.into())),
            )
            .with_property_index_candidate(
                TenantId::DEFAULT,
                index_label,
                "email",
                &Value::String(email.into()),
                NodeId::new(11),
            );
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let mut op = op(index_label, email);
        assert!(
            op.next_batch(&ctx, &s).unwrap().is_empty(),
            "wrong-label candidate must be dropped"
        );
    }

    #[test]
    fn no_declared_index_yields_empty() {
        // The stub was never told the index exists → the seam serves no
        // candidates (defensive; the planner would not route here).
        let label = LabelId::new(1);
        let s = StubExecutorSubstrate::new().with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(1), Some(label))
                .with_property("email", Value::String("a@x.com".into())),
        );
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let mut op = op(label, "a@x.com");
        assert!(op.next_batch(&ctx, &s).unwrap().is_empty());
    }

    #[test]
    fn residual_filter_narrows_verified_rows() {
        // Two nodes share the looked-up email; a residual `age > 30`
        // keeps only the one that passes.
        let label = LabelId::new(1);
        let email = "shared@x.com";
        let s = StubExecutorSubstrate::new()
            .with_property_index(TenantId::DEFAULT, label, "email")
            .with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(1), Some(label))
                    .with_property("email", Value::String(email.into()))
                    .with_property("age", Value::Integer(40)),
            )
            .with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(2), Some(label))
                    .with_property("email", Value::String(email.into()))
                    .with_property("age", Value::Integer(20)),
            )
            .with_property_index_candidate(
                TenantId::DEFAULT,
                label,
                "email",
                &Value::String(email.into()),
                NodeId::new(1),
            )
            .with_property_index_candidate(
                TenantId::DEFAULT,
                label,
                "email",
                &Value::String(email.into()),
                NodeId::new(2),
            );
        // residual: n.age > 30
        let residual = BoundExpression::BinaryOp {
            op: crate::ast::BinOp::Gt,
            lhs: Box::new(BoundExpression::PropertyAccess {
                base: Box::new(BoundExpression::VariableRef {
                    name: "n".into(),
                    binding_id: BindingId::new(0),
                    span: Span::point(1, 1),
                    type_info: None,
                }),
                path: vec![crate::semantic::bound_ast::BoundPropertyRef {
                    name: "age".into(),
                    property_id: None,
                    span: Span::point(1, 1),
                }],
                span: Span::point(1, 1),
                type_info: None,
            }),
            rhs: Box::new(BoundExpression::Literal {
                value: Literal::Integer(30),
                span: Span::point(1, 1),
                type_info: None,
            }),
            span: Span::point(1, 1),
            type_info: None,
        };
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let mut op = PropertyIndexScanOp::new(
            BindingId::new(0),
            label,
            "email".to_string(),
            lit_str(email),
            Some(residual),
            Lsn::MAX,
        );
        let b = op.next_batch(&ctx, &s).unwrap();
        assert_eq!(b.row_count(), 1, "residual keeps only the age>30 node");
        let n = match &b.row(0)[0] {
            Value::Node(n) => n,
            _ => panic!(),
        };
        assert_eq!(n.id, NodeId::new(1));
    }

    #[test]
    fn pre_cancellation_skips_substrate_call() {
        let label = LabelId::new(1);
        let s = fixture(3, label);
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        ctx.cancellation().cancel();
        let mut op = op(label, "u1@x.com");
        assert_eq!(op.next_batch(&ctx, &s), Err(ExecutionError::Cancelled));
    }

    #[test]
    fn acquires_snapshot_lsn_at_first_batch() {
        let label = LabelId::new(1);
        let s = fixture(3, label);
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        assert_eq!(ctx.snapshot_lsn(), None, "pre-first-batch: not acquired");
        let mut op = op(label, "u1@x.com");
        let _ = op.next_batch(&ctx, &s).unwrap();
        assert!(ctx.snapshot_lsn().is_some(), "post-first-batch: acquired");
    }

    // ─── #1415 op-level scan-fallback for UNKEYABLE resolved values ─────
    // A value with no canonical index key (fractional Float, List/Map
    // param, negative Integer param) must NOT be answered by the empty
    // index candidate set — the op falls back to Scan+Filter and returns
    // the SAME rows a full scan would. (The load-bearing production-seam
    // equivalence is proven by the arcgraph-mcp proptest; these are the
    // fast stub-level guards that the op takes the fallback branch.)

    fn lit_float(f: f64) -> BoundExpression {
        BoundExpression::Literal {
            value: Literal::Float(f),
            span: Span::point(1, 1),
            type_info: None,
        }
    }

    fn param(name: &str) -> BoundExpression {
        BoundExpression::Parameter {
            name: name.to_string(),
            span: Span::point(1, 1),
            type_info: None,
        }
    }

    /// A node whose `price` is a FRACTIONAL float (19.99). The index is
    /// declared but the value is unkeyable (RC-5 Float drop) → no
    /// candidate is ever seeded. Pre-fix the op would take the empty
    /// candidate set and emit ZERO rows; the fallback scans + filters and
    /// finds the node (matching a full scan's `price = 19.99`).
    #[test]
    fn fractional_float_literal_falls_back_to_scan() {
        let label = LabelId::new(1);
        let s = StubExecutorSubstrate::new()
            .with_property_index(TenantId::DEFAULT, label, "price")
            .with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(1), Some(label))
                    .with_property("price", Value::Float(19.99)),
            )
            .with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(2), Some(label))
                    .with_property("price", Value::Float(5.0)),
            );
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let mut op = PropertyIndexScanOp::new(
            BindingId::new(0),
            label,
            "price".to_string(),
            lit_float(19.99),
            None,
            Lsn::MAX,
        );
        let b = op.next_batch(&ctx, &s).unwrap();
        assert_eq!(
            b.row_count(),
            1,
            "fractional-float fallback must find the match"
        );
        let n = match &b.row(0)[0] {
            Value::Node(n) => n,
            _ => panic!(),
        };
        assert_eq!(n.id, NodeId::new(1));
    }

    /// A `$p` param bound to a `List` value. Unkeyable → fallback scan.
    #[test]
    fn list_param_falls_back_to_scan() {
        let label = LabelId::new(1);
        let list = Value::List(vec![Value::Integer(1), Value::Integer(2)]);
        let s = StubExecutorSubstrate::new()
            .with_property_index(TenantId::DEFAULT, label, "tags")
            .with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(3), Some(label)).with_property("tags", list.clone()),
            );
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let mut params = Parameters::new();
        params.insert("p".into(), list);
        let mut op = PropertyIndexScanOp::new(
            BindingId::new(0),
            label,
            "tags".to_string(),
            param("p"),
            None,
            Lsn::MAX,
        )
        .with_parameters(params);
        let b = op.next_batch(&ctx, &s).unwrap();
        assert_eq!(b.row_count(), 1, "list-param fallback must find the match");
        assert!(matches!(&b.row(0)[0], Value::Node(n) if n.id == NodeId::new(3)));
    }

    /// A `$p` param bound to a NEGATIVE Integer (unkeyable in RC-5:
    /// the u56 slot has no sign bit). Fallback scan finds the match.
    #[test]
    fn negative_integer_param_falls_back_to_scan() {
        let label = LabelId::new(1);
        let s = StubExecutorSubstrate::new()
            .with_property_index(TenantId::DEFAULT, label, "balance")
            .with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(4), Some(label))
                    .with_property("balance", Value::Integer(-5)),
            )
            .with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(5), Some(label))
                    .with_property("balance", Value::Integer(5)),
            );
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let mut params = Parameters::new();
        params.insert("p".into(), Value::Integer(-5));
        let mut op = PropertyIndexScanOp::new(
            BindingId::new(0),
            label,
            "balance".to_string(),
            param("p"),
            None,
            Lsn::MAX,
        )
        .with_parameters(params);
        let b = op.next_batch(&ctx, &s).unwrap();
        assert_eq!(b.row_count(), 1, "negative-int fallback must find only -5");
        assert!(matches!(&b.row(0)[0], Value::Node(n) if n.id == NodeId::new(4)));
    }

    /// KEYABLE values still USE the index (no perf regression): an
    /// INTEGRAL float `10.0` is keyable (coerces to the int bucket), so
    /// the op consults the index candidate set — NOT the scan. We seed a
    /// candidate but NO node in the scan store for that path, so a match
    /// here proves the index path (not the scan) served it.
    #[test]
    fn integral_float_uses_index_not_scan() {
        let label = LabelId::new(1);
        // A node whose stored `age` is the INTEGER 10 (integral-float
        // lookup 10.0 must find stored int 10 via the coercing recheck).
        let s = StubExecutorSubstrate::new()
            .with_property_index(TenantId::DEFAULT, label, "age")
            .with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(6), Some(label)).with_property("age", Value::Integer(10)),
            )
            .with_property_index_candidate(
                TenantId::DEFAULT,
                label,
                "age",
                &Value::Integer(10),
                NodeId::new(6),
            );
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        // Confirm the op treats 10.0 as keyable (index path).
        assert!(s.value_is_indexable(&Value::Float(10.0)), "10.0 is keyable");
        let mut op = PropertyIndexScanOp::new(
            BindingId::new(0),
            label,
            "age".to_string(),
            lit_float(10.0),
            None,
            Lsn::MAX,
        );
        let b = op.next_batch(&ctx, &s).unwrap();
        assert_eq!(
            b.row_count(),
            1,
            "integral-float 10.0 finds stored int 10 via the index"
        );
        assert!(matches!(&b.row(0)[0], Value::Node(n) if n.id == NodeId::new(6)));
    }
}
