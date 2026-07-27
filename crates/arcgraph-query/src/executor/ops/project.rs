//! [`ProjectOp`] — RETURN / WITH projection (M4-61).
//!
//! Replaces the upstream row schema with the projection-item list:
//! evaluates each projection expression against the upstream row and
//! emits a new row of the projection results.
//!
//! # Wildcard projection
//!
//! `RETURN *` (or `WITH *`) emits all upstream bindings unchanged
//! (preserves the schema). The wildcard is encoded by
//! [`crate::semantic::BoundProjectionKind::Wildcard`].
//!
//! # Schema generation (#746 binder↔ProjectOp contract)
//!
//! For non-wildcard items, the output column carries the
//! BINDER-ASSIGNED [`BindingId`]
//! ([`crate::semantic::bound_ast::BoundProjectionItem::output_id`]).
//! The binder mints this id when it binds the WITH / RETURN projection
//! (for a WITH-projected name it is the id the post-WITH scope
//! `declare()`s), so a downstream consumer — a 2nd `Project` over an
//! `Aggregate`, a `MATCH`/`RETURN`/`UNWIND` after a `WITH` — that
//! resolves the column by name gets the SAME id this operator emits.
//!
//! Earlier (pre-#746) this operator minted executor-local SYNTHETIC ids
//! (`fresh_id_base + column_index`, high-half u64) that no downstream
//! `resolve()` agreed with — so `MATCH (n) WITH n.x AS a RETURN a` and
//! `RETURN count(n)` (which lowers to `Project(Aggregate(..))`) failed
//! at runtime with `Eval("binding … missing from row schema")`. The
//! `fresh_id_base` is now a DEFENSIVE FALLBACK only — used solely for an
//! `Expr` item that somehow carries no binder id (e.g. a
//! hand-constructed test operator); the real bind path always assigns
//! one.

use crate::executor::batch::Batch;
use crate::executor::context::ExecutionContext;
use crate::executor::error::ExecutionError;
use crate::executor::eval::{Parameters, evaluate};
use crate::executor::ops::PhysicalOperator;
use crate::executor::ops::schema_index;
use crate::executor::substrate::ExecutorSubstrate;
use crate::semantic::bound_ast::{BindingId, BoundProjectionItem, BoundProjectionKind};

/// RETURN / WITH projection operator.
#[derive(Debug)]
pub struct ProjectOp {
    child: Box<PhysicalOperator>,
    items: Vec<BoundProjectionItem>,
    parameters: Parameters,
    /// Output schema. For `RETURN *` this mirrors the input schema;
    /// for `RETURN expr1, expr2, ...` each column carries the
    /// binder-assigned `output_id` of its projection item (#746),
    /// falling back to a synthetic `fresh_id_base + i` only for an
    /// `Expr` item with no binder id.
    schema: Vec<BindingId>,
    /// DEFENSIVE fallback base for the synthetic-id path (#746): used
    /// only when an `Expr` projection item carries no binder-assigned
    /// `output_id`. Values share the high half so a fallback id never
    /// collides with a bind-pass id (those grow from 0). The real bind
    /// path always sets `output_id`, so this is dormant in production.
    fresh_id_base: u64,
}

impl ProjectOp {
    /// Construct a `ProjectOp`. The child's schema is consulted for
    /// the wildcard (`*`) case; for explicit items, the projection
    /// synthesizes fresh binding IDs.
    #[must_use]
    pub fn new(child: PhysicalOperator, items: Vec<BoundProjectionItem>) -> Self {
        // The fresh-ID base is per-instance. Use a high half so we
        // never collide with bind-pass IDs (they grow from 0).
        let fresh_id_base = 0xFFFF_FFFF_0000_0000_u64;
        let schema = derive_schema(child.schema(), &items, fresh_id_base);
        Self {
            child: Box::new(child),
            items,
            parameters: Parameters::new(),
            schema,
            fresh_id_base,
        }
    }

    /// Inject a per-query parameter bag.
    #[must_use]
    pub fn with_parameters(mut self, parameters: Parameters) -> Self {
        self.parameters = parameters;
        self
    }

    /// Output schema.
    pub fn schema(&self) -> &[BindingId] {
        &self.schema
    }

    /// Pull the next batch.
    pub fn next_batch<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<Batch, ExecutionError> {
        ctx.cancellation().check()?;
        let upstream = self.child.next_batch(ctx, substrate)?;
        if upstream.is_empty() {
            return Ok(Batch::empty(self.schema.len()));
        }
        let mut out = Batch::with_capacity(self.schema.len());
        let upstream_schema = self.child.schema().to_vec();
        let upstream_lookup = move |b: BindingId| schema_index(&upstream_schema, b);
        for row in upstream.into_rows() {
            let mut new_row: Vec<crate::executor::value::Value> =
                Vec::with_capacity(self.schema.len());
            for item in &self.items {
                match &item.kind {
                    BoundProjectionKind::Wildcard { order } => {
                        // `RETURN *` / `WITH *` — emit the in-scope
                        // bindings in the binder-supplied ALPHABETICAL
                        // name order (openCypher v9 §6.1). An empty
                        // `order` (hand-built item) falls back to verbatim
                        // child-row passthrough (the pre-fix behavior).
                        for col in wildcard_columns(self.child.schema(), order) {
                            new_row.push(row[col].clone());
                        }
                    }
                    BoundProjectionKind::Expr(e) => {
                        let v = evaluate(e, &row, &upstream_lookup, &self.parameters)?;
                        new_row.push(v);
                    }
                }
            }
            // Sanity: rows must match the cached schema width.
            // A wildcard's row size depends on the upstream's row
            // shape; we require uniformity within a single batch
            // (the upstream guarantees this).
            if new_row.len() != self.schema.len() {
                // If the only item is a wildcard and the upstream
                // schema length matches, the schema was set correctly
                // at construction. Otherwise we need to refresh the
                // schema to match the wildcard's actual width.
                self.schema = derive_schema(self.child.schema(), &self.items, self.fresh_id_base);
                debug_assert_eq!(new_row.len(), self.schema.len());
            }
            let _ = out.push_row(new_row);
        }
        Ok(out)
    }
}

/// The child-schema column indices a `RETURN *` / `WITH *` wildcard
/// emits, in output order.
///
/// `order` is the binder-supplied list of in-scope binding ids in
/// openCypher wildcard output order (ALPHABETICAL by variable name —
/// Cypher 9 §6.1). For each id we resolve its position in the physical
/// child schema (which is in pipeline-DECLARATION order) and emit that
/// column index. The result reorders the child columns into name order
/// and drops anonymous bindings (which are never in `order`).
///
/// Fallbacks preserve the pre-fix verbatim-passthrough behavior:
/// - an EMPTY `order` (a hand-built test item, or a path the binder did
///   not populate) → every child column, in child-schema order;
/// - an `order` id NOT present in the child schema (defensive — should
///   not occur for a well-formed plan) → skipped, so the wildcard never
///   indexes out of the row.
fn wildcard_columns(child_schema: &[BindingId], order: &[BindingId]) -> Vec<usize> {
    if order.is_empty() {
        return (0..child_schema.len()).collect();
    }
    order
        .iter()
        .filter_map(|id| child_schema.iter().position(|c| c == id))
        .collect()
}

fn derive_schema(
    child_schema: &[BindingId],
    items: &[BoundProjectionItem],
    fresh_id_base: u64,
) -> Vec<BindingId> {
    let mut schema: Vec<BindingId> = Vec::new();
    let mut fresh_id = fresh_id_base;
    for item in items {
        match &item.kind {
            BoundProjectionKind::Wildcard { order } => {
                for col in wildcard_columns(child_schema, order) {
                    schema.push(child_schema[col]);
                }
            }
            BoundProjectionKind::Expr(_) => {
                // #746: emit the column under the BINDER-ASSIGNED id so
                // downstream consumers (a 2nd Project over an Aggregate,
                // a MATCH/RETURN/UNWIND after a WITH) resolve it to the
                // same id. The synthetic `fresh_id` is a defensive
                // fallback for an Expr item with no binder id only.
                let id = item.output_id.unwrap_or_else(|| {
                    let synthetic = BindingId::new(fresh_id);
                    fresh_id += 1;
                    synthetic
                });
                schema.push(id);
            }
        }
    }
    schema
}

#[cfg(test)]
mod tests {
    use arcgraph_core::{LabelId, Lsn, NodeId, PartitionId, TenantId};

    use super::*;
    use crate::ast::Literal;
    use crate::error::Span;
    use crate::executor::ops::ScanOp;
    use crate::executor::substrate::StubExecutorSubstrate;
    use crate::executor::value::{NodeView, Value};

    fn fixture() -> StubExecutorSubstrate {
        StubExecutorSubstrate::new().with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(1), Some(LabelId::new(1)))
                .with_property("age", Value::Integer(30)),
        )
    }

    #[test]
    fn project_wildcard_passes_through_schema_and_rows() {
        let s = fixture();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let scan = ScanOp::new(BindingId::new(0), None, Lsn::MAX);
        let item = BoundProjectionItem {
            kind: BoundProjectionKind::wildcard(),
            alias: None,
            output_id: None,
            source_text: None,
            span: Span::point(1, 1),
        };
        let mut proj = ProjectOp::new(PhysicalOperator::Scan(scan), vec![item]);
        let b = proj.next_batch(&ctx, &s).unwrap();
        assert_eq!(b.row_count(), 1);
        assert_eq!(b.column_count(), 1);
        match &b.row(0)[0] {
            Value::Node(n) => assert_eq!(n.id, NodeId::new(1)),
            _ => panic!("expected node value via wildcard projection"),
        }
    }

    #[test]
    fn project_literal_emits_constant_column() {
        let s = fixture();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let scan = ScanOp::new(BindingId::new(0), None, Lsn::MAX);
        let item = BoundProjectionItem {
            kind: BoundProjectionKind::Expr(BoundExpressionLitInt(7)),
            alias: Some("seven".into()),
            // #746: a binder-assigned output id; the row value (7) is
            // what this test asserts, but the column now carries this id.
            output_id: Some(BindingId::new(42)),
            source_text: None,
            span: Span::point(1, 1),
        };
        let mut proj = ProjectOp::new(PhysicalOperator::Scan(scan), vec![item]);
        let b = proj.next_batch(&ctx, &s).unwrap();
        assert_eq!(b.row_count(), 1);
        assert_eq!(b.row(0)[0], Value::Integer(7));
    }

    // Helper: build a literal-integer BoundExpression.
    #[allow(non_snake_case)]
    fn BoundExpressionLitInt(n: i64) -> crate::semantic::bound_ast::BoundExpression {
        crate::semantic::bound_ast::BoundExpression::Literal {
            value: Literal::Integer(n),
            span: Span::point(1, 1),
            type_info: None,
        }
    }

    #[test]
    fn project_propagates_cancel() {
        let s = fixture();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        ctx.cancellation().cancel();
        let scan = ScanOp::new(BindingId::new(0), None, Lsn::MAX);
        let item = BoundProjectionItem {
            kind: BoundProjectionKind::wildcard(),
            alias: None,
            output_id: None,
            source_text: None,
            span: Span::point(1, 1),
        };
        let mut proj = ProjectOp::new(PhysicalOperator::Scan(scan), vec![item]);
        let r = proj.next_batch(&ctx, &s);
        assert_eq!(r, Err(ExecutionError::Cancelled));
    }
}
