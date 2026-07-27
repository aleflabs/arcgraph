//! [`FilterOp`] — WHERE / WITH WHERE predicate filter (M4-61).
//!
//! Evaluates the predicate against each upstream row and emits the
//! row IFF the predicate evaluates to [`crate::executor::ThreeValued::True`].
//! Rows where the predicate is `False` or `Unknown` are dropped per
//! Cypher 9 §6.2 (NULL operands → drop the row at the WHERE
//! boundary).
//!
//! # Schema pass-through
//!
//! Output schema = input schema (no new columns).
//!
//! # SIMD fast-path (W13α / M4-64b)
//!
//! When the predicate matches the canonical hot-path shape
//! `<VarRef.PropertyAccess[single segment]> <BinOp[Eq|Ne|Lt|Le|Gt|Ge]>
//! <IntLiteral>` the constructor caches a `SimdShape` descriptor and
//! the per-batch hot loop routes through
//! [`crate::executor::simd::filter::simd_filter_i64_cmp`]. Other
//! predicate shapes (boolean ops, IS NULL, IN, function calls,
//! mixed-type comparisons) fall through to the scalar
//! [`crate::executor::eval::evaluate`] path. Per ADR-038 amendment-03
//! §TIER-2-b the SIMD path preserves 3VL NULL semantics: NULL property
//! cells produce ThreeValued::Unknown which the SIMD path drops, the
//! same as the scalar path.
//!
//! # ADR provenance
//! - ADR-038 §2 D-20 — 3VL truth tables; `WHERE` row drop is the
//!   "Unknown → effective false" projection at the filter boundary.
//! - ADR-038 amendment-03 §TIER-2-b — M4-62 3VL implementation.
//! - ADR-038 amendment-02 §M4.f + amendment-03 §Structural-1 —
//!   M4-64b SIMD on FilterOp predicate evaluation.

use crate::ast::{BinOp, Literal};
use crate::executor::batch::Batch;
use crate::executor::context::ExecutionContext;
use crate::executor::error::ExecutionError;
use crate::executor::eval::{Parameters, evaluate};
use crate::executor::ops::PhysicalOperator;
use crate::executor::ops::schema_index;
use crate::executor::simd::filter::{CmpOp, simd_filter_i64_cmp};
use crate::executor::substrate::ExecutorSubstrate;
use crate::executor::three_vl::ThreeValued;
use crate::executor::value::Value;
use crate::semantic::bound_ast::{BindingId, BoundExpression, BoundPropertyRef};

/// SIMD fast-path descriptor cached at FilterOp construction.
///
/// When `Some`, the per-batch hot loop routes through
/// [`simd_filter_i64_cmp`]. When `None`, the scalar
/// [`evaluate`] path runs.
#[derive(Debug, Clone)]
struct SimdShape {
    /// Variable binding the property reads from (column source).
    binding_id: BindingId,
    /// Property name to read off the bound node / relationship.
    property_name: String,
    /// SIMD-side comparison operator.
    op: CmpOp,
    /// RHS i64 literal value.
    target: i64,
}

/// WHERE / WITH WHERE predicate filter operator.
#[derive(Debug)]
pub struct FilterOp {
    child: Box<PhysicalOperator>,
    predicate: BoundExpression,
    parameters: Parameters,
    schema: Vec<BindingId>,
    /// SIMD fast-path descriptor when the predicate matches the
    /// canonical hot-path shape; `None` otherwise (scalar fallback).
    simd_shape: Option<SimdShape>,
}

impl FilterOp {
    /// Construct a `FilterOp` from a child operator + predicate.
    /// The child's schema is preserved.
    #[must_use]
    pub fn new(child: PhysicalOperator, predicate: BoundExpression) -> Self {
        let schema = child.schema().to_vec();
        let simd_shape = detect_simd_shape(&predicate);
        Self {
            child: Box::new(child),
            predicate,
            parameters: Parameters::new(),
            schema,
            simd_shape,
        }
    }

    /// Inject a per-query parameter bag. Default is empty.
    #[must_use]
    pub fn with_parameters(mut self, parameters: Parameters) -> Self {
        self.parameters = parameters;
        self
    }

    /// Output schema (= input schema).
    pub fn schema(&self) -> &[BindingId] {
        &self.schema
    }

    /// Whether the SIMD fast-path is active for this filter (i.e.,
    /// the predicate matched the canonical hot-path shape at
    /// construction). Tests + EXPLAIN use this to assert the route.
    #[must_use]
    pub fn uses_simd_path(&self) -> bool {
        self.simd_shape.is_some()
    }

    /// Pull the next batch.
    pub fn next_batch<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<Batch, ExecutionError> {
        ctx.cancellation().check()?;
        // Repeatedly pull from child until we get a non-empty filtered
        // batch OR child EOS. Don't return an empty batch unless child
        // is exhausted (matching Scan/Expand's semantics).
        loop {
            ctx.cancellation().check()?;
            let upstream = self.child.next_batch(ctx, substrate)?;
            if upstream.is_empty() {
                return Ok(Batch::empty(self.schema.len()));
            }
            let out = if let Some(shape) = &self.simd_shape {
                self.simd_filter_batch(shape, upstream)?
            } else {
                self.scalar_filter_batch(upstream)?
            };
            if !out.is_empty() {
                return Ok(out);
            }
            // Empty after filter; pull another upstream batch.
        }
    }

    /// Scalar fallback path. Evaluates the predicate per row.
    fn scalar_filter_batch(&self, upstream: Batch) -> Result<Batch, ExecutionError> {
        let mut out = Batch::with_capacity(self.schema.len());
        let schema = self.schema.clone();
        let lookup = move |b: BindingId| schema_index(&schema, b);
        for row in upstream.into_rows() {
            let v = evaluate(&self.predicate, &row, &lookup, &self.parameters)?;
            let tv = ThreeValued::from_value(&v);
            if tv.passes_filter() {
                let _ = out.push_row(row);
            }
        }
        Ok(out)
    }

    /// SIMD fast-path. Extracts the property column into a packed
    /// `Vec<i64>` + parallel null mask, runs
    /// [`simd_filter_i64_cmp`], and emits rows by mask.
    ///
    /// Falls back row-by-row to the scalar evaluator when the column
    /// at runtime contains a non-Integer / non-Node value (a planner
    /// invariant violation; the scalar path surfaces a clean error).
    fn simd_filter_batch(
        &self,
        shape: &SimdShape,
        upstream: Batch,
    ) -> Result<Batch, ExecutionError> {
        let col_idx = match schema_index(&self.schema, shape.binding_id) {
            Some(idx) => idx,
            None => {
                // Should be caught at construction-time by the
                // semantic visitor; defensive scalar fallback so a
                // schema-shape escape doesn't silently change behavior.
                return self.scalar_filter_batch(upstream);
            }
        };

        let rows = upstream.into_rows();
        let n = rows.len();
        let mut values: Vec<i64> = Vec::with_capacity(n);
        let mut is_null: Vec<bool> = Vec::with_capacity(n);
        let mut fallback_to_scalar = false;

        for row in &rows {
            let cell = row.get(col_idx).cloned().unwrap_or(Value::Null);
            // Walk: cell must be Value::Node(n) or Value::Relationship(r);
            // then read the named property; must be Integer or Null.
            let prop_value = match &cell {
                Value::Node(n) => n.properties.get(&shape.property_name).cloned(),
                Value::Relationship(r) => r.properties.get(&shape.property_name).cloned(),
                Value::Null => None,
                _ => {
                    // Cell is neither a node nor a relationship — the
                    // SIMD path's pattern doesn't apply here; defer to
                    // the scalar path so the consistent
                    // `apply_binop`-side error is surfaced.
                    fallback_to_scalar = true;
                    break;
                }
            };
            match prop_value.unwrap_or(Value::Null) {
                Value::Integer(i) => {
                    values.push(i);
                    is_null.push(false);
                }
                Value::Null => {
                    // Per amendment-03 §TIER-2-b: NULL operand → drop.
                    values.push(0);
                    is_null.push(true);
                }
                _ => {
                    // Non-Integer property at this column — the SIMD
                    // shape's i64 assumption doesn't hold; fall back
                    // so the scalar path delivers a uniform error /
                    // result.
                    fallback_to_scalar = true;
                    break;
                }
            }
        }

        if fallback_to_scalar {
            // Re-build a Batch from the rows so the scalar path can
            // walk them.
            let upstream = Batch::from_rows(rows).ok_or_else(|| {
                ExecutionError::Eval("SIMD fallback: failed to rebuild batch".into())
            })?;
            return self.scalar_filter_batch(upstream);
        }

        let mask = simd_filter_i64_cmp(&values, &is_null, shape.target, shape.op);

        // Emit rows by mask.
        let mut out = Batch::with_capacity(self.schema.len());
        for (i, row) in rows.into_iter().enumerate() {
            if mask[i] {
                let _ = out.push_row(row);
            }
        }
        Ok(out)
    }
}

/// Detect the SIMD-friendly canonical predicate shape:
/// `<VarRef.PropertyAccess[single seg]> <BinOp[Eq|Ne|Lt|Le|Gt|Ge]>
/// <Literal::Integer>`.
///
/// Returns `Some(SimdShape)` when the shape matches; `None` otherwise.
/// Mismatched shapes route to the scalar
/// [`crate::executor::eval::evaluate`] path.
fn detect_simd_shape(predicate: &BoundExpression) -> Option<SimdShape> {
    let (op, lhs, rhs) = match predicate {
        BoundExpression::BinaryOp { op, lhs, rhs, .. } => (op.clone(), lhs.as_ref(), rhs.as_ref()),
        _ => return None,
    };
    let cmp = match op {
        BinOp::Eq => CmpOp::Eq,
        BinOp::Neq => CmpOp::Ne,
        BinOp::Lt => CmpOp::Lt,
        BinOp::Le => CmpOp::Le,
        BinOp::Gt => CmpOp::Gt,
        BinOp::Ge => CmpOp::Ge,
        // AND / OR / arithmetic / etc. — not SIMD-shape.
        _ => return None,
    };

    // LHS must be PropertyAccess { base: VariableRef, path: [single segment] }.
    let (binding_id, property_name) = match lhs {
        BoundExpression::PropertyAccess { base, path, .. } => {
            let bid = match base.as_ref() {
                BoundExpression::VariableRef { binding_id, .. } => *binding_id,
                _ => return None,
            };
            // v1.0 SIMD path: single-segment property access only.
            let segs: &[BoundPropertyRef] = path;
            if segs.len() != 1 {
                return None;
            }
            (bid, segs[0].name.clone())
        }
        _ => return None,
    };

    // RHS must be Literal::Integer.
    let target = match rhs {
        BoundExpression::Literal {
            value: Literal::Integer(i),
            ..
        } => *i,
        _ => return None,
    };

    Some(SimdShape {
        binding_id,
        property_name,
        op: cmp,
        target,
    })
}

#[cfg(test)]
mod tests {
    use arcgraph_core::{LabelId, Lsn, NodeId, PartitionId, TenantId};

    use super::*;
    use crate::ast::{BinOp, Literal};
    use crate::error::Span;
    use crate::executor::ops::ScanOp;
    use crate::executor::substrate::StubExecutorSubstrate;
    use crate::executor::value::{NodeView, Value};

    fn fixture() -> StubExecutorSubstrate {
        let mut s = StubExecutorSubstrate::new();
        for i in 1..=5_u64 {
            s = s.with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(i), Some(LabelId::new(1)))
                    .with_property("age", Value::Integer(i as i64 * 10)),
            );
        }
        s
    }

    fn predicate_age_gt_30(node_binding: BindingId) -> BoundExpression {
        // n.age > 30
        BoundExpression::BinaryOp {
            op: BinOp::Gt,
            lhs: Box::new(BoundExpression::PropertyAccess {
                base: Box::new(BoundExpression::VariableRef {
                    name: "n".into(),
                    binding_id: node_binding,
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
        }
    }

    #[test]
    fn filter_drops_rows_where_predicate_false() {
        let s = fixture();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let scan = ScanOp::new(BindingId::new(0), None, Lsn::MAX);
        let mut filt = FilterOp::new(
            PhysicalOperator::Scan(scan),
            predicate_age_gt_30(BindingId::new(0)),
        );
        let b = filt.next_batch(&ctx, &s).unwrap();
        // ages 10,20,30,40,50; predicate `> 30` keeps 40,50.
        assert_eq!(b.row_count(), 2);
        for row in b.rows() {
            let n = match &row[0] {
                Value::Node(n) => n,
                _ => panic!("expected Node"),
            };
            let age = n.properties.get("age").cloned().unwrap();
            match age {
                Value::Integer(a) => assert!(a > 30, "row age must pass predicate: {a}"),
                _ => panic!("expected Integer age"),
            }
        }
    }

    #[test]
    fn filter_drops_rows_where_predicate_unknown() {
        // Replace a node's age with NULL → predicate returns Unknown
        // → row dropped.
        let mut s = StubExecutorSubstrate::new();
        s = s
            .with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(1), None).with_property("age", Value::Integer(50)),
            )
            .with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(2), None).with_property("age", Value::Null),
            );
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let scan = ScanOp::new(BindingId::new(0), None, Lsn::MAX);
        let mut filt = FilterOp::new(
            PhysicalOperator::Scan(scan),
            predicate_age_gt_30(BindingId::new(0)),
        );
        let b = filt.next_batch(&ctx, &s).unwrap();
        // Only the integer-50 row passes; the NULL-age row's
        // predicate is Unknown → dropped.
        assert_eq!(b.row_count(), 1);
        let n = match &b.row(0)[0] {
            Value::Node(n) => n,
            _ => panic!(),
        };
        assert_eq!(n.id, NodeId::new(1));
    }

    #[test]
    fn filter_propagates_cancel() {
        let s = fixture();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        ctx.cancellation().cancel();
        let scan = ScanOp::new(BindingId::new(0), None, Lsn::MAX);
        let mut filt = FilterOp::new(
            PhysicalOperator::Scan(scan),
            predicate_age_gt_30(BindingId::new(0)),
        );
        let r = filt.next_batch(&ctx, &s);
        assert_eq!(r, Err(ExecutionError::Cancelled));
    }

    // --------- W13α / M4-64b SIMD fast-path pins ----------

    #[test]
    fn simd_shape_detected_for_canonical_predicate() {
        // Pin: the canonical `<n.prop> <BinOp> <IntLit>` shape
        // triggers the SIMD path at construction.
        let scan = ScanOp::new(BindingId::new(0), None, Lsn::MAX);
        let filt = FilterOp::new(
            PhysicalOperator::Scan(scan),
            predicate_age_gt_30(BindingId::new(0)),
        );
        assert!(filt.uses_simd_path(), "canonical shape must route to SIMD");
    }

    #[test]
    fn simd_shape_not_detected_for_non_canonical_predicate() {
        // Pin: a function-call predicate (non-canonical) MUST NOT
        // route to SIMD; it falls back to the scalar evaluator.
        let scan = ScanOp::new(BindingId::new(0), None, Lsn::MAX);
        // `id(n) > 30` (function call on LHS, not a PropertyAccess).
        let pred = BoundExpression::BinaryOp {
            op: BinOp::Gt,
            lhs: Box::new(BoundExpression::FunctionCall {
                name: "id".into(),
                args: vec![BoundExpression::VariableRef {
                    name: "n".into(),
                    binding_id: BindingId::new(0),
                    span: Span::point(1, 1),
                    type_info: None,
                }],
                distinct: false,
                star: false,
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
        let filt = FilterOp::new(PhysicalOperator::Scan(scan), pred);
        assert!(
            !filt.uses_simd_path(),
            "non-canonical shape MUST fall through to scalar"
        );
    }

    #[test]
    fn simd_path_matches_scalar_path_on_same_input() {
        // Pin: the SIMD-routed FilterOp produces the same row output
        // as the scalar-routed FilterOp on identical input. This is
        // the equivalence promise at the operator level.
        let s = fixture();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);

        let scan_simd = ScanOp::new(BindingId::new(0), None, Lsn::MAX);
        let mut simd_filt = FilterOp::new(
            PhysicalOperator::Scan(scan_simd),
            predicate_age_gt_30(BindingId::new(0)),
        );
        assert!(simd_filt.uses_simd_path());
        let simd_batch = simd_filt.next_batch(&ctx, &s).unwrap();

        // Construct an equivalent scalar-routed FilterOp by wrapping
        // the predicate in `(<canonical>) AND TRUE` so it's no longer
        // canonical (BinaryOp::And is not in the SIMD CmpOp set).
        let scan_scalar = ScanOp::new(BindingId::new(0), None, Lsn::MAX);
        let pred_scalar = BoundExpression::BinaryOp {
            op: BinOp::And,
            lhs: Box::new(predicate_age_gt_30(BindingId::new(0))),
            rhs: Box::new(BoundExpression::Literal {
                value: Literal::Bool(true),
                span: Span::point(1, 1),
                type_info: None,
            }),
            span: Span::point(1, 1),
            type_info: None,
        };
        let mut scalar_filt = FilterOp::new(PhysicalOperator::Scan(scan_scalar), pred_scalar);
        assert!(!scalar_filt.uses_simd_path());
        let scalar_batch = scalar_filt.next_batch(&ctx, &s).unwrap();

        assert_eq!(simd_batch.row_count(), scalar_batch.row_count());
        for (sr, scr) in simd_batch.rows().iter().zip(scalar_batch.rows()) {
            assert_eq!(sr, scr);
        }
    }

    #[test]
    fn simd_path_drops_null_property_rows() {
        // Pin: even on the SIMD path, NULL property cells produce
        // ThreeValued::Unknown → row dropped. Verifies amendment-03
        // §TIER-2-b 3VL preservation.
        let mut s = StubExecutorSubstrate::new();
        s = s
            .with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(1), None).with_property("age", Value::Integer(50)),
            )
            .with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(2), None).with_property("age", Value::Null),
            )
            .with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(3), None).with_property("age", Value::Integer(20)),
            );
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let scan = ScanOp::new(BindingId::new(0), None, Lsn::MAX);
        let mut filt = FilterOp::new(
            PhysicalOperator::Scan(scan),
            predicate_age_gt_30(BindingId::new(0)),
        );
        assert!(filt.uses_simd_path());
        let b = filt.next_batch(&ctx, &s).unwrap();
        assert_eq!(b.row_count(), 1);
        let n = match &b.row(0)[0] {
            Value::Node(n) => n,
            _ => panic!(),
        };
        assert_eq!(n.id, NodeId::new(1));
    }
}
