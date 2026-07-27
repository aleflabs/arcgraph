//! [`PlainPathOp`] — `MATCH p = (a)-[..]->(b)` plain named-path
//! materialization (ADR-193 D-4/D-5/D-6).
//!
//! Lowers from [`crate::logical_plan::LogicalNamedPath`] with
//! [`crate::logical_plan::PathAlgorithm::Plain`]. Unlike
//! [`crate::executor::ops::path::NamedShortestPathOp`] (which RE-traverses
//! the substrate via BFS), a plain named path is materialized DIRECTLY
//! from the MATCH-bound rows the child subtree (Scan + Expand chain)
//! already produced: for each row it assembles a [`Value::Path`] from the
//! pattern's bound node/relationship slots in pattern order, and appends
//! it as a NEW bound column (`path_var`).
//!
//! # Schema
//!
//! Output schema = `child_schema ++ [path_var]`. The plain path is
//! additive (the underlying `a`, `r`, `b` bindings stay visible), so
//! `RETURN p, a, b` works.
//!
//! # Var-length composition (D-5)
//!
//! When a segment is a var-length expand (`p = (a)-[*1..3]->(b)`), the
//! relationship column carries a `Value::List(Vec<Value::Relationship>)`
//! in traversal order (ADR-186 RC-2). The op walks that ordered rel-list
//! and materializes each INTERMEDIATE node from the adjacent
//! relationship endpoints (the endpoint NOT shared with the previous
//! relationship), preserving the `#nodes = #rels + 1` invariant. The
//! final node is the bound `to_var` node from the row; intermediate
//! nodes are hydrated through the substrate when it supports point
//! reads, and otherwise degrade to id-only [`NodeView`]s.
//!
//! # ADR provenance
//!
//! - **ADR-193 D-4/D-5/D-6** — plain-path execution + var-length
//!   composition + zero-length path.
//! - **ADR-186 §RC-2** — the var-length `ExpandOp` ordered-rel output
//!   this op composes.

use std::collections::HashMap;

use arcgraph_core::NodeId;

use crate::executor::batch::Batch;
use crate::executor::context::ExecutionContext;
use crate::executor::error::ExecutionError;
use crate::executor::ops::PhysicalOperator;
use crate::executor::ops::schema_index;
use crate::executor::substrate::ExecutorSubstrate;
use crate::executor::value::{NodeView, PathSegment, PathView, RelView, Value};
use crate::logical_plan::PlainPathShape;
use crate::semantic::bound_ast::BindingId;

/// `MATCH p = (a)-[..]->(b)` plain named-path operator (ADR-193 D-4).
pub struct PlainPathOp {
    child: Box<PhysicalOperator>,
    /// The ordered element-binding sequence (from lowering).
    shape: PlainPathShape,
    /// Output schema = `child_schema ++ [path_var]`.
    schema: Vec<BindingId>,
    /// Cached child schema for per-element column lookup.
    child_schema: Vec<BindingId>,
}

impl std::fmt::Debug for PlainPathOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PlainPathOp")
            .field("segments", &self.shape.segments.len())
            .field("schema_width", &self.schema.len())
            .finish()
    }
}

impl PlainPathOp {
    /// Construct a `PlainPathOp` appending `path_var` to the child schema.
    #[must_use]
    pub fn new(child: PhysicalOperator, shape: PlainPathShape, path_var: BindingId) -> Self {
        let child_schema = child.schema().to_vec();
        let mut schema = child_schema.clone();
        schema.push(path_var);
        Self {
            child: Box::new(child),
            shape,
            schema,
            child_schema,
        }
    }

    /// Output schema (`child_schema ++ [path_var]`).
    #[must_use]
    pub fn schema(&self) -> &[BindingId] {
        &self.schema
    }

    /// Pull the next batch — append a materialized `Value::Path` column
    /// to every upstream row.
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
        let mut node_memo: HashMap<NodeId, NodeView> = HashMap::new();
        for row in upstream.into_rows() {
            let path = self.build_path(ctx, substrate, &mut node_memo, &row)?;
            let mut new_row = row;
            new_row.push(Value::Path(path));
            let _ = out.push_row(new_row);
        }
        Ok(out)
    }

    /// Assemble the [`PathView`] for one upstream row from the bound
    /// node/relationship slots in pattern order (D-4/D-5/D-6).
    fn build_path<S: ExecutorSubstrate>(
        &self,
        ctx: &ExecutionContext,
        substrate: &S,
        node_memo: &mut HashMap<NodeId, NodeView>,
        row: &[Value],
    ) -> Result<PathView, ExecutionError> {
        let start = self.node_at(row, self.shape.start, "path start node")?;
        let mut prev_id = start.id;
        let mut path = PathView::new(start);
        for seg in &self.shape.segments {
            let end_node = self.node_at(row, seg.end, "path segment end node")?;
            let rel_cell = self.cell(row, seg.rel, "path segment relationship")?;
            if seg.var_length {
                // ADR-186 RC-2 — the rel column is a Value::List in
                // traversal order; walk it materializing intermediate
                // nodes from adjacent endpoints (D-5).
                let rels = expect_rel_list(rel_cell)?;
                if rels.is_empty() {
                    // `*0` — zero hops; the segment's end node equals its
                    // start (no PathSegment is added). Advance `prev_id`
                    // to the bound end node (== prev_id) so a following
                    // segment composes correctly.
                    prev_id = end_node.id;
                    continue;
                }
                let last = rels.len() - 1;
                for (i, rel) in rels.into_iter().enumerate() {
                    let seg_end = if i == last {
                        // Last hop lands on the bound `to_var` node (full
                        // NodeView with label + properties).
                        end_node.clone()
                    } else {
                        // Prime-Directive-5 budget (#965 / ADR-211):
                        // O(path_len) point-reads per emitted path row,
                        // bounded by varlen depth cap + #1040 supernode
                        // fan-out firewall + per-batch memo (combinatorial
                        // paths share intermediates).
                        self.hydrate_intermediate(
                            ctx,
                            substrate,
                            node_memo,
                            other_endpoint(&rel, prev_id),
                        )?
                    };
                    prev_id = seg_end.id;
                    path.segments.push(PathSegment { rel, end: seg_end });
                }
            } else {
                // Single-hop segment — the rel column is a scalar
                // Value::Relationship.
                let rel = expect_rel(rel_cell)?;
                prev_id = end_node.id;
                path.segments.push(PathSegment { rel, end: end_node });
            }
        }
        Ok(path)
    }

    fn hydrate_intermediate<S: ExecutorSubstrate>(
        &self,
        ctx: &ExecutionContext,
        substrate: &S,
        node_memo: &mut HashMap<NodeId, NodeView>,
        id: NodeId,
    ) -> Result<NodeView, ExecutionError> {
        if let Some(node) = node_memo.get(&id) {
            return Ok(node.clone());
        }
        let node = match substrate.node_by_id_with_context(ctx, id)? {
            Some(bound) => bound.node,
            None => NodeView::new(id, None),
        };
        node_memo.insert(id, node.clone());
        Ok(node)
    }

    /// Read a bound node from `row` at the column for `binding`.
    fn node_at(
        &self,
        row: &[Value],
        binding: BindingId,
        ctx: &str,
    ) -> Result<NodeView, ExecutionError> {
        match self.cell(row, binding, ctx)? {
            Value::Node(n) => Ok(n.clone()),
            other => Err(ExecutionError::Eval(format!(
                "PlainPathOp: expected a Node for {ctx}, got a non-node cell ({other:?})"
            ))),
        }
    }

    /// Look up `binding`'s cell in `row` via the cached child schema.
    fn cell<'r>(
        &self,
        row: &'r [Value],
        binding: BindingId,
        ctx: &str,
    ) -> Result<&'r Value, ExecutionError> {
        let idx = schema_index(&self.child_schema, binding).ok_or_else(|| {
            ExecutionError::Eval(format!(
                "PlainPathOp: {ctx} binding {binding:?} not found in child schema \
                 {schema:?}",
                schema = self.child_schema
            ))
        })?;
        row.get(idx).ok_or_else(|| {
            ExecutionError::Eval(format!(
                "PlainPathOp: row missing cell at index {idx} for {ctx}"
            ))
        })
    }
}

/// The endpoint of `rel` NOT equal to `prev` (direction-aware
/// next-node resolution for var-length walks, D-5). For a self-loop
/// (`from == to == prev`) returns `prev` (direction-agnostic).
fn other_endpoint(rel: &RelView, prev: NodeId) -> NodeId {
    if rel.from == prev { rel.to } else { rel.from }
}

/// Extract a `Vec<RelView>` from a var-length rel cell
/// (`Value::List(Vec<Value::Relationship>)`, ADR-186 RC-2).
fn expect_rel_list(cell: &Value) -> Result<Vec<RelView>, ExecutionError> {
    match cell {
        Value::List(items) => {
            let mut rels = Vec::with_capacity(items.len());
            for it in items {
                match it {
                    Value::Relationship(r) => rels.push(r.clone()),
                    other => {
                        return Err(ExecutionError::Eval(format!(
                            "PlainPathOp: var-length rel list element is not a relationship \
                             ({other:?})"
                        )));
                    }
                }
            }
            Ok(rels)
        }
        other => Err(ExecutionError::Eval(format!(
            "PlainPathOp: expected a var-length relationship list, got {other:?}"
        ))),
    }
}

/// Extract a single [`RelView`] from a scalar rel cell.
fn expect_rel(cell: &Value) -> Result<RelView, ExecutionError> {
    match cell {
        Value::Relationship(r) => Ok(r.clone()),
        other => Err(ExecutionError::Eval(format!(
            "PlainPathOp: expected a Relationship cell, got {other:?}"
        ))),
    }
}
