//! [`CreateRelOp`] — write-op operator for `CREATE (a)-[r:LABEL
//! {props}]->(b) RETURN r?` per ADR-148 (W26-θ Phase 2).
//!
//! Lowers from [`crate::logical_plan::LogicalCreateRel`]. The
//! operator holds two upstream sub-pipelines (`source_op` +
//! `target_op`) — typically Phase 1 [`super::CreateNodeOp`]s for
//! inline-CREATE endpoints, but future Phase 5 lights MATCH→CREATE
//! by allowing any node-producing upstream.
//!
//! On first `next_batch` invocation:
//!
//! 1. Pulls one row from `source_op` to resolve the source NodeId
//!    (via the schema slot at `source_idx`).
//! 2. Pulls one row from `target_op` to resolve the target NodeId.
//! 3. Materializes literal property values via the shared Phase 1
//!    `literal_to_value` helper (Phase 2 inherits the literal-only
//!    narrowing per ADR-147 §D-4).
//! 4. Calls [`ExecutorSubstrate::create_rel`] with the tenant +
//!    canonical (source → target) endpoints + label + materialized
//!    properties. The substrate opens a per-tenant `Transaction`
//!    (production wiring per ADR-031 + ADR-033) and commits at call
//!    boundary.
//! 5. Emits ONE row binding the new `RelId` to the operator's
//!    optional variable (or an EMPTY row when the CREATE-rel was
//!    anonymous). Subsequent `next_batch` calls return the EOS empty
//!    batch.
//!
//! # Direction canonicalization
//!
//! Phase 2 grammar admits `LeftToRight` (`-[..]->`) and `RightToLeft`
//! (`<-[..]-`); the substrate's `create_rel` always takes
//! `(source, target)` in source-to-target wire order. When the AST
//! direction is `RightToLeft`, this operator swaps source/target
//! BEFORE the substrate call so the stored rel always points
//! source-to-target in canonical orientation. The AST direction is
//! preserved in the LogicalPlan for EXPLAIN / cache-key purposes.
//!
//! # Schema
//!
//! - When `var` is `Some(binding)` → schema = `[binding]`; the row
//!   is `[Value::Rel(RelView{id, ...})]`.
//! - When `var` is `None` (anonymous rel) → schema is empty `[]`;
//!   the row is a 0-column tuple. The empty row is still ONE row
//!   from the executor's perspective (the openCypher TCK Eligible
//!   signal for "1 relationship created").
//!
//! # ADR provenance
//! - **ADR-148** — primary spec (W26-θ Phase 2).
//! - **ADR-147** §D-7 — production-substrate convention (per-tenant
//!   Transaction; intern table for type-name; default trait impl
//!   returns `IndexUnavailable`).
//! - **ADR-031** + **ADR-033** — per-tenant `Transaction` discipline
//!   (commit + rollback; consolidated lifecycle ADR forward-pinned
//!   to v1.1+).
//! - **ADR-041 §D-4** — MVCC visibility key (forward-pin; not used
//!   for the write at Phase 2 but consumed by the substrate at
//!   commit-LSN-stamping).
//! - **issue #356** — strict-schema property typing.
//! - **ADR-152-amendment-02 §D-1** — composite `List`-of-scalars
//!   property values lift here via `super::literal_lift::literal_value`
//!   (`Map` + temporal / decimal remain deferred per §D-2).

use crate::ast::CreateRelDirection;
use crate::executor::batch::Batch;
use crate::executor::context::ExecutionContext;
use crate::executor::error::ExecutionError;
use crate::executor::eval::Parameters;
use crate::executor::ops::{PhysicalOperator, schema_index};
use crate::executor::substrate::ExecutorSubstrate;
use crate::executor::value::{RelView, Value};
use crate::logical_plan::LogicalCreateEndpoint;
use crate::semantic::bound_ast::{BindingId, BoundExpression};

/// CREATE-rel executor op (ADR-148 W26-θ Phase 2).
#[derive(Debug)]
pub struct CreateRelOp {
    /// Optional binding for the new rel.
    var: Option<BindingId>,
    /// Mandatory label name (Phase 2 per ADR-148 §D-1). Forwarded
    /// verbatim to the substrate's `create_rel`; the substrate
    /// handles interning per ADR-148 §D-7.
    label: String,
    /// Phase 2 literal-only property values (per ADR-147 §D-4
    /// inherited); the op materializes each `BoundExpression::Literal`
    /// to a `Value` at first-batch time.
    properties: Vec<(String, BoundExpression)>,
    /// Upstream sub-pipeline producing the source NodeId.
    source_op: Box<PhysicalOperator>,
    /// Binding within `source_op`'s schema that carries the source
    /// NodeId.
    source_binding: BindingId,
    source_endpoint: LogicalCreateEndpoint,
    /// Upstream sub-pipeline producing the target NodeId.
    target_op: Box<PhysicalOperator>,
    /// Binding within `target_op`'s schema that carries the target
    /// NodeId.
    target_binding: BindingId,
    target_endpoint: LogicalCreateEndpoint,
    /// AST-side direction. Used to canonicalize (source, target) →
    /// (src, dst) wire order at the substrate call (RightToLeft
    /// swaps; LeftToRight passes through).
    direction: CreateRelDirection,
    /// Optional upstream sub-pipeline (issue #832 — multi-item CREATE
    /// chains). When present, the op is a streaming transform: per
    /// upstream row it pulls the source + target endpoints, writes one
    /// rel, and emits the upstream row EXTENDED with the new binding.
    /// This is DISTINCT from `source_op` / `target_op` (the endpoint
    /// producers); `input` carries the PRIOR CREATE item's row stream
    /// so it executes too. When `None` (chain leaf or MERGE
    /// create-branch) the op emits exactly one row on first call.
    input: Option<Box<PhysicalOperator>>,
    /// Cached schema: when `input` is present, `input.schema()`
    /// EXTENDED with `[var]`; otherwise `[var]` when `var` is `Some`,
    /// empty otherwise.
    schema: Vec<BindingId>,
    /// ADR-147-amendment-03 (D-1) — per-query parameter bag (threaded so
    /// a CREATE-rel property `$param` resolves at materialization time).
    parameters: Parameters,
    /// EOS flag for the leaf path (`input == None`): set after the
    /// single-row emission. Unused on the streaming path.
    emitted: bool,
}

impl CreateRelOp {
    /// Construct a fresh `CreateRelOp` from a
    /// [`crate::logical_plan::LogicalCreateRel`].
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        var: Option<BindingId>,
        label: String,
        properties: Vec<(String, BoundExpression)>,
        source_op: PhysicalOperator,
        source_binding: BindingId,
        source_endpoint: LogicalCreateEndpoint,
        target_op: PhysicalOperator,
        target_binding: BindingId,
        target_endpoint: LogicalCreateEndpoint,
        direction: CreateRelDirection,
    ) -> Self {
        let schema = match var {
            Some(b) => vec![b],
            None => Vec::new(),
        };
        Self {
            var,
            label,
            properties,
            source_op: Box::new(source_op),
            source_binding,
            source_endpoint,
            target_op: Box::new(target_op),
            target_binding,
            target_endpoint,
            direction,
            input: None,
            schema,
            parameters: Parameters::new(),
            emitted: false,
        }
    }

    /// Attach the per-query parameter bag (ADR-147-amendment-03 §D-1).
    /// Mirrors `CreateSpineOp::with_parameters`.
    #[must_use]
    pub fn with_parameters(mut self, parameters: Parameters) -> Self {
        self.parameters = parameters;
        self
    }

    /// Attach an upstream sub-pipeline, turning this leaf op into a
    /// streaming create-rel (issue #832 — multi-item CREATE chains).
    /// The schema becomes the upstream schema EXTENDED with this op's
    /// binding (if any). Mirrors [`super::CreateNodeOp::with_input`].
    #[must_use]
    pub fn with_input(mut self, input: PhysicalOperator) -> Self {
        let mut schema: Vec<BindingId> = input.schema().to_vec();
        if let Some(b) = self.var {
            schema.push(b);
        }
        self.schema = schema;
        self.input = Some(Box::new(input));
        self
    }

    /// Output schema.
    pub fn schema(&self) -> &[BindingId] {
        &self.schema
    }

    /// Pull the next batch.
    ///
    /// - **Leaf** (`input == None`): one row on first call, EOS
    ///   thereafter (the original Phase-2 behavior).
    /// - **Streaming** (`input == Some`, issue #832): per upstream row,
    ///   write one rel and emit the upstream row EXTENDED with the new
    ///   binding; EOS when the upstream is dry. This lets a prior
    ///   CREATE item in a multi-item `CREATE …,(a)-[:R]->(b)` execute.
    pub fn next_batch<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<Batch, ExecutionError> {
        // Defense-in-depth cancel check inside the operator.
        ctx.cancellation().check()?;

        // Streaming path (issue #832): an upstream chain feeds this op.
        // The `match` scopes the mutable borrow of `self.input` to the
        // pull alone, so `do_create_rel` is free in the loop below.
        let upstream: Option<(Batch, Vec<BindingId>)> = match self.input.as_mut() {
            Some(input) => {
                let schema = input.schema().to_vec();
                Some((input.next_batch(ctx, substrate)?, schema))
            }
            None => None,
        };
        if let Some((in_batch, input_schema)) = upstream {
            if in_batch.is_empty() {
                return Ok(Batch::empty(self.schema.len()));
            }
            let _exec_lsn = ctx.ensure_snapshot_lsn();
            let mut out = Batch::with_capacity(self.schema.len());
            for i in 0..in_batch.row_count() {
                let mut row = in_batch.row(i).to_vec();
                // #1123 / PD-5 budget: this is intentionally O(input
                // rows) writes, bounded by upstream cardinality. CREATE
                // adds no new cap here; existing spillover semantics stay
                // on the upstream operators and Batch boundary.
                if let Some(cell) =
                    self.do_create_rel(ctx, substrate, Some((&row, &input_schema)))?
                {
                    row.push(cell);
                }
                if !out.push_row(row) {
                    return Err(ExecutionError::Eval(
                        "CreateRelOp: batch push overflow (chained CREATE)".into(),
                    ));
                }
            }
            return Ok(out);
        }

        // Leaf path (input == None) — emit exactly ONE row, then EOS.
        if self.emitted {
            return Ok(Batch::empty(self.schema.len()));
        }
        // Acquire snapshot LSN per ADR-038 §2 D-18 rule 1 (defensive
        // — the outer materialize loop already holds the LSN guard).
        let _exec_lsn = ctx.ensure_snapshot_lsn();
        let cell = self.do_create_rel(ctx, substrate, None)?;
        self.emitted = true;
        // When the CREATE-rel was anonymous (`var` is None), `cell` is
        // None and the row is a 0-column tuple — still ONE row per the
        // openCypher TCK "1 relationship created" semantic.
        let row = match cell {
            Some(v) => vec![v],
            None => Vec::new(),
        };
        let mut batch = Batch::with_capacity(self.schema.len());
        if !batch.push_row(row) {
            return Err(ExecutionError::Eval(
                "CreateRelOp: batch push overflow".into(),
            ));
        }
        Ok(batch)
    }

    /// Resolve endpoints, write ONE rel, and return the binding cell to
    /// emit: `Some(Value::Relationship(..))` when this op binds a
    /// variable, else `None`. Shared by the leaf and streaming paths
    /// (the #832 fix only changes how many times + what prefix the row
    /// carries, never the rel write itself).
    fn do_create_rel<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
        input_row: Option<(&[Value], &[BindingId])>,
    ) -> Result<Option<Value>, ExecutionError> {
        // Pull the source NodeId from the upstream sub-pipeline.
        let source_id = self.pull_node_id(ctx, substrate, UpstreamSide::Source, input_row)?;
        // Pull the target NodeId from the upstream sub-pipeline.
        let target_id = self.pull_node_id(ctx, substrate, UpstreamSide::Target, input_row)?;

        // Direction canonicalization: RightToLeft swaps source/target
        // BEFORE the substrate call so the stored rel always points
        // source → target in canonical (src, dst) wire order. The
        // AST direction is preserved on the LogicalPlan for EXPLAIN.
        let (src_id, dst_id) = match self.direction {
            CreateRelDirection::LeftToRight => (source_id, target_id),
            CreateRelDirection::RightToLeft => (target_id, source_id),
        };

        // Materialize property values (ADR-147-amendment-03 §D-1 — the
        // shared const-fast-path-else-`evaluate` + runtime value-type
        // gate, identical to the live `CreateSpineOp` path so the two
        // never drift). Resolve against the upstream row when streaming.
        let (mat_row, mat_schema): (&[Value], &[BindingId]) = match input_row {
            Some((row, schema)) => (row, schema),
            None => (&[], &[]),
        };
        let materialized = super::literal_lift::materialize_create_properties(
            "CreateRelOp",
            &self.properties,
            mat_row,
            mat_schema,
            &self.parameters,
        )?;

        // Substrate write — opens + commits a per-tenant transaction
        // per ADR-031 + ADR-033 (production wiring; stub bookkeeps in-memory).
        let rel_id = substrate
            .create_rel(
                ctx.tenant(),
                src_id,
                dst_id,
                self.label.as_str(),
                &materialized,
                ctx,
            )
            .map_err(ExecutionError::Substrate)?;

        if self.var.is_some() {
            // Build a RelView carrying the new id + the resolved
            // endpoints + materialized property bag (ADR-152 §D-1 —
            // RETURN-after-CREATE row carries the literal properties).
            // The TypeId is `None` at this layer — the substrate
            // did intern the type-name but the executor doesn't
            // round-trip the result at Phase 2 (the downstream
            // RETURN projection reads the rel-id, not the rel-type,
            // at v1.0-α — same convention as Phase 1).
            let mut rel = RelView::new(rel_id, src_id, dst_id, None);
            // #871 (rel sister of facet 3) — carry the rel-type NAME so
            // `CREATE ()-[r:KNOWS]->() RETURN type(r)` surfaces `'KNOWS'`
            // (and the serializers emit `"KNOWS"`) instead of null. The
            // op holds the verbatim type name (`self.label`); the numeric
            // TypeId stays `None` per the Phase-2 convention above.
            rel.rel_type_name = Some(self.label.clone());
            for (k, v) in &materialized {
                rel.properties.insert(k.clone(), v.clone());
            }
            Ok(Some(Value::Relationship(rel)))
        } else {
            Ok(None)
        }
    }

    /// Pull one row from the source / target sub-pipeline and project
    /// out the NodeId at the expected schema slot.
    fn pull_node_id<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
        side: UpstreamSide,
        input_row: Option<(&[Value], &[BindingId])>,
    ) -> Result<arcgraph_core::NodeId, ExecutionError> {
        let (op, binding, endpoint, side_name) = match side {
            UpstreamSide::Source => (
                &mut self.source_op,
                self.source_binding,
                self.source_endpoint,
                "source",
            ),
            UpstreamSide::Target => (
                &mut self.target_op,
                self.target_binding,
                self.target_endpoint,
                "target",
            ),
        };
        if let LogicalCreateEndpoint::RowBinding(row_binding) = endpoint {
            let (row, schema) = input_row.ok_or_else(|| {
                ExecutionError::Eval(format!(
                    "CreateRelOp: {side_name} row-bound endpoint has no input row"
                ))
            })?;
            return node_id_from_row(row, schema, row_binding, side_name);
        }
        if input_row.is_some() {
            op.rearm_create_endpoint_leaf();
        }
        let batch = op.next_batch(ctx, substrate)?;
        if batch.is_empty() {
            return Err(ExecutionError::Eval(format!(
                "CreateRelOp: {side_name} sub-pipeline produced no row (expected 1)"
            )));
        }
        let row = batch.row(0);
        let schema = op.schema();
        let idx = schema_index(schema, binding).ok_or_else(|| {
            ExecutionError::Eval(format!(
                "CreateRelOp: {side_name} binding {binding:?} not in upstream schema {schema:?}"
            ))
        })?;
        let cell = row.get(idx).ok_or_else(|| {
            ExecutionError::Eval(format!(
                "CreateRelOp: {side_name} row missing cell at index {idx}"
            ))
        })?;
        node_id_from_cell(cell, side_name)
    }
}

fn node_id_from_row(
    row: &[Value],
    schema: &[BindingId],
    binding: BindingId,
    side_name: &str,
) -> Result<arcgraph_core::NodeId, ExecutionError> {
    let idx = schema_index(schema, binding).ok_or_else(|| {
        ExecutionError::Eval(format!(
            "CreateRelOp: {side_name} binding {binding:?} not in input schema {schema:?}"
        ))
    })?;
    let cell = row.get(idx).ok_or_else(|| {
        ExecutionError::Eval(format!(
            "CreateRelOp: {side_name} input row missing cell at index {idx}"
        ))
    })?;
    node_id_from_cell(cell, side_name)
}

fn node_id_from_cell(
    cell: &Value,
    side_name: &str,
) -> Result<arcgraph_core::NodeId, ExecutionError> {
    match cell {
        Value::Node(node) => Ok(node.id),
        other => Err(ExecutionError::Eval(format!(
            "CreateRelOp: {side_name} cell is not a Node value: {other:?}"
        ))),
    }
}

/// Discriminator for the source / target pull side.
#[derive(Debug, Clone, Copy)]
enum UpstreamSide {
    Source,
    Target,
}

#[cfg(test)]
mod tests {
    use arcgraph_core::{LabelId, PartitionId, TenantId};

    use super::*;
    use crate::ast::Literal;
    use crate::executor::ops::CreateNodeOp;
    use crate::executor::substrate::StubExecutorSubstrate;

    fn mk_create_node(var: BindingId, label: &str) -> PhysicalOperator {
        PhysicalOperator::CreateNode(CreateNodeOp::new(
            Some(var),
            Some(label.to_string()),
            Vec::new(),
        ))
    }

    #[test]
    fn create_rel_left_to_right_writes_and_emits_one_row() {
        let s = StubExecutorSubstrate::new();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let source = mk_create_node(BindingId::new(0), "User");
        let target = mk_create_node(BindingId::new(1), "User");
        let mut op = CreateRelOp::new(
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
        let b1 = op.next_batch(&ctx, &s).expect("first batch OK");
        assert_eq!(b1.row_count(), 1, "first batch: one CREATE-rel row");
        let b2 = op.next_batch(&ctx, &s).expect("second batch OK");
        assert!(b2.is_empty(), "second batch: EOS");
    }

    #[test]
    fn create_rel_anonymous_emits_zero_column_row() {
        let s = StubExecutorSubstrate::new();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let source = mk_create_node(BindingId::new(0), "User");
        let target = mk_create_node(BindingId::new(1), "User");
        let mut op = CreateRelOp::new(
            None,
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
        let b1 = op.next_batch(&ctx, &s).expect("first batch OK");
        assert_eq!(b1.row_count(), 1);
        let r = b1.row(0);
        assert!(r.is_empty(), "anonymous CREATE-rel row is 0-column");
    }

    #[test]
    fn create_rel_right_to_left_swaps_endpoints_before_substrate() {
        // The substrate's create_rel always sees source→target wire
        // order; a RightToLeft AST swaps before the substrate call.
        // Smoke: just assert the row count is correct (the actual
        // swap-correctness is asserted by mcp_create_rel_e2e via
        // `expand` round-trip).
        let s = StubExecutorSubstrate::new();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let source = mk_create_node(BindingId::new(0), "User");
        let target = mk_create_node(BindingId::new(1), "User");
        let mut op = CreateRelOp::new(
            Some(BindingId::new(2)),
            "KNOWS".into(),
            Vec::new(),
            source,
            BindingId::new(0),
            LogicalCreateEndpoint::Fresh,
            target,
            BindingId::new(1),
            LogicalCreateEndpoint::Fresh,
            CreateRelDirection::RightToLeft,
        );
        let b1 = op.next_batch(&ctx, &s).expect("first batch OK");
        assert_eq!(b1.row_count(), 1);
    }

    #[test]
    fn create_rel_with_literal_properties_succeeds() {
        let s = StubExecutorSubstrate::new();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let source = mk_create_node(BindingId::new(0), "User");
        let target = mk_create_node(BindingId::new(1), "User");
        let props = vec![(
            "since".to_string(),
            BoundExpression::Literal {
                value: Literal::Integer(2024),
                span: crate::error::Span::point(1, 1),
                type_info: None,
            },
        )];
        let mut op = CreateRelOp::new(
            Some(BindingId::new(2)),
            "KNOWS".into(),
            props,
            source,
            BindingId::new(0),
            LogicalCreateEndpoint::Fresh,
            target,
            BindingId::new(1),
            LogicalCreateEndpoint::Fresh,
            CreateRelDirection::LeftToRight,
        );
        let b1 = op.next_batch(&ctx, &s).expect("first batch OK");
        assert_eq!(b1.row_count(), 1);
        let _ = LabelId::new(0); // silence unused-import lint
    }

    #[test]
    fn create_rel_pre_cancellation_skips_substrate_call() {
        let s = StubExecutorSubstrate::new();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        ctx.cancellation().cancel();
        let source = mk_create_node(BindingId::new(0), "User");
        let target = mk_create_node(BindingId::new(1), "User");
        let mut op = CreateRelOp::new(
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
        let r = op.next_batch(&ctx, &s);
        assert_eq!(r, Err(ExecutionError::Cancelled));
    }
}
