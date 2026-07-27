//! Composite CREATE-spine executor.
//!
//! `Pipeline::build_create_spine` gathers a contiguous
//! `CreateNode`/`CreateRel` logical spine. This op executes that spine
//! iteratively per input row instead of materializing it as N nested
//! physical operators. Work stays O(items * rows) substrate writes,
//! while pull frames are O(1) in CREATE-chain depth (#1123 R2).

use arcgraph_core::{LabelId, NodeId};

use crate::ast::CreateRelDirection;
use crate::executor::batch::Batch;
use crate::executor::context::ExecutionContext;
use crate::executor::error::ExecutionError;
use crate::executor::eval::Parameters;
use crate::executor::ops::{PhysicalOperator, schema_index};
use crate::executor::substrate::ExecutorSubstrate;
use crate::executor::value::{NodeView, RelView, Value};
use crate::logical_plan::LogicalCreateEndpoint;
use crate::semantic::bound_ast::{BindingId, BoundExpression};

#[derive(Debug)]
pub enum CreateSpineItem {
    Node(CreateSpineNode),
    Rel(Box<CreateSpineRel>),
}

#[derive(Debug)]
pub struct CreateSpineNode {
    pub var: Option<BindingId>,
    pub label: Option<String>,
    pub properties: Vec<(String, BoundExpression)>,
}

#[derive(Debug)]
pub struct CreateSpineRel {
    pub var: Option<BindingId>,
    pub label: String,
    pub properties: Vec<(String, BoundExpression)>,
    pub source_op: PhysicalOperator,
    pub source_binding: BindingId,
    pub source_visible: bool,
    pub source_endpoint: LogicalCreateEndpoint,
    pub target_op: PhysicalOperator,
    pub target_binding: BindingId,
    pub target_visible: bool,
    pub target_endpoint: LogicalCreateEndpoint,
    pub direction: CreateRelDirection,
}

#[derive(Debug)]
pub struct CreateSpineOp {
    input: Option<Box<PhysicalOperator>>,
    items: Vec<CreateSpineItem>,
    schema: Vec<BindingId>,
    /// ADR-147-amendment-03 (D-1) — the per-query parameter bag, threaded
    /// from the pipeline so that CREATE property values referencing
    /// `$param` resolve at materialization time (the same bag
    /// `UnwindOp` / `ProjectOp` carry).
    parameters: Parameters,
    emitted: bool,
}

impl CreateSpineOp {
    #[must_use]
    pub fn new(
        input: Option<PhysicalOperator>,
        items: Vec<CreateSpineItem>,
        expose_path_endpoint_bindings: bool,
    ) -> Self {
        let mut schema = input
            .as_ref()
            .map(|input| input.schema().to_vec())
            .unwrap_or_default();
        for item in &items {
            match item {
                CreateSpineItem::Node(item) => {
                    if let Some(binding) = item.var {
                        schema.push(binding);
                    }
                }
                CreateSpineItem::Rel(item) => {
                    if matches!(item.source_endpoint, LogicalCreateEndpoint::Fresh)
                        && item.source_visible
                        && expose_path_endpoint_bindings
                    {
                        schema.push(item.source_binding);
                    }
                    if matches!(item.target_endpoint, LogicalCreateEndpoint::Fresh)
                        && item.target_visible
                        && expose_path_endpoint_bindings
                    {
                        schema.push(item.target_binding);
                    }
                    if let Some(binding) = item.var {
                        schema.push(binding);
                    }
                }
            }
        }
        Self {
            input: input.map(Box::new),
            items,
            schema,
            parameters: Parameters::new(),
            emitted: false,
        }
    }

    /// Attach the per-query parameter bag (ADR-147-amendment-03 §D-1).
    /// Mirrors `UnwindOp::with_parameters` / `ProjectOp::with_parameters`;
    /// the pipeline threads the same bag it installs on those ops so a
    /// CREATE property `$param` resolves identically.
    #[must_use]
    pub fn with_parameters(mut self, parameters: Parameters) -> Self {
        self.parameters = parameters;
        self
    }

    pub fn schema(&self) -> &[BindingId] {
        &self.schema
    }

    pub(crate) fn rearm_leaf_endpoint(&mut self) {
        if self.input.is_none() {
            self.emitted = false;
        }
    }

    pub fn next_batch<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<Batch, ExecutionError> {
        ctx.cancellation().check()?;

        let upstream = match self.input.as_mut() {
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
                let mut work_row = in_batch.row(i).to_vec();
                let mut work_schema = input_schema.clone();
                self.execute_items(ctx, substrate, &mut work_row, &mut work_schema)?;
                let out_row = project_output_row(&work_row, &work_schema, &self.schema)?;
                if !out.push_row(out_row) {
                    return Err(ExecutionError::Eval(
                        "CreateSpineOp: batch push overflow".into(),
                    ));
                }
            }
            return Ok(out);
        }

        if self.emitted {
            return Ok(Batch::empty(self.schema.len()));
        }
        let _exec_lsn = ctx.ensure_snapshot_lsn();
        let mut work_row = Vec::new();
        let mut work_schema = Vec::new();
        self.execute_items(ctx, substrate, &mut work_row, &mut work_schema)?;
        let row = project_output_row(&work_row, &work_schema, &self.schema)?;
        self.emitted = true;

        let mut batch = Batch::with_capacity(self.schema.len());
        if !batch.push_row(row) {
            return Err(ExecutionError::Eval(
                "CreateSpineOp: batch push overflow".into(),
            ));
        }
        Ok(batch)
    }

    fn execute_items<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
        row: &mut Vec<Value>,
        row_schema: &mut Vec<BindingId>,
    ) -> Result<(), ExecutionError> {
        // Disjoint-field borrow: `&self.parameters` (immutable) alongside
        // `&mut self.items` is allowed because they are distinct fields.
        let parameters = &self.parameters;
        for item in &mut self.items {
            match item {
                CreateSpineItem::Node(item) => {
                    // ADR-147-amendment-03 (D-1): materialize property
                    // values against the CURRENT work row (the unwound
                    // element + any earlier-bound cells) + the param bag.
                    if let Some(cell) = create_node_cell(
                        ctx,
                        substrate,
                        item.var,
                        item.label.as_deref(),
                        &item.properties,
                        row,
                        row_schema,
                        parameters,
                    )? {
                        let binding = item.var.expect("node cell exists only with binding");
                        row.push(cell);
                        row_schema.push(binding);
                    }
                }
                CreateSpineItem::Rel(item) => {
                    let source = resolve_endpoint(
                        ctx,
                        substrate,
                        EndpointSpec {
                            op: &mut item.source_op,
                            binding: item.source_binding,
                            endpoint: item.source_endpoint,
                            visible: item.source_visible,
                            side_name: "source",
                        },
                        row,
                        row_schema,
                    )?;
                    let target = resolve_endpoint(
                        ctx,
                        substrate,
                        EndpointSpec {
                            op: &mut item.target_op,
                            binding: item.target_binding,
                            endpoint: item.target_endpoint,
                            visible: item.target_visible,
                            side_name: "target",
                        },
                        row,
                        row_schema,
                    )?;
                    if let Some(cell) = create_rel_cell(
                        ctx,
                        substrate,
                        item.var,
                        item.label.as_str(),
                        &item.properties,
                        source,
                        target,
                        item.direction.clone(),
                        row,
                        row_schema,
                        parameters,
                    )? {
                        let binding = item.var.expect("rel cell exists only with binding");
                        row.push(cell);
                        row_schema.push(binding);
                    }
                }
            }
        }
        Ok(())
    }
}

fn project_output_row(
    work_row: &[Value],
    work_schema: &[BindingId],
    output_schema: &[BindingId],
) -> Result<Vec<Value>, ExecutionError> {
    let mut row = Vec::with_capacity(output_schema.len());
    for binding in output_schema {
        let idx = schema_index(work_schema, *binding).ok_or_else(|| {
            ExecutionError::Eval(format!(
                "CreateSpineOp: output binding {binding:?} not in work schema {work_schema:?}"
            ))
        })?;
        let cell = work_row.get(idx).ok_or_else(|| {
            ExecutionError::Eval(format!(
                "CreateSpineOp: output row missing cell at index {idx}"
            ))
        })?;
        row.push(cell.clone());
    }
    Ok(row)
}

struct EndpointSpec<'a> {
    op: &'a mut PhysicalOperator,
    binding: BindingId,
    endpoint: LogicalCreateEndpoint,
    visible: bool,
    side_name: &'static str,
}

fn resolve_endpoint<S: ExecutorSubstrate>(
    ctx: &ExecutionContext,
    substrate: &S,
    spec: EndpointSpec<'_>,
    row: &mut Vec<Value>,
    row_schema: &mut Vec<BindingId>,
) -> Result<NodeId, ExecutionError> {
    if let LogicalCreateEndpoint::RowBinding(binding) = spec.endpoint {
        return node_id_from_row(row, row_schema, binding, spec.side_name);
    }

    spec.op.rearm_create_endpoint_leaf();
    let batch = spec.op.next_batch(ctx, substrate)?;
    if batch.is_empty() {
        return Err(ExecutionError::Eval(format!(
            "CreateSpineOp: {} sub-pipeline produced no row (expected 1)",
            spec.side_name
        )));
    }
    let endpoint_schema = spec.op.schema();
    let idx = schema_index(endpoint_schema, spec.binding).ok_or_else(|| {
        ExecutionError::Eval(format!(
            "CreateSpineOp: {} binding {:?} not in upstream schema {:?}",
            spec.side_name, spec.binding, endpoint_schema
        ))
    })?;
    let cell = batch.row(0).get(idx).ok_or_else(|| {
        ExecutionError::Eval(format!(
            "CreateSpineOp: {} row missing cell at index {idx}",
            spec.side_name
        ))
    })?;
    let id = node_id_from_cell(cell, spec.side_name)?;
    if spec.visible {
        row.push(cell.clone());
        row_schema.push(spec.binding);
    }
    Ok(id)
}

#[allow(clippy::too_many_arguments)]
fn create_node_cell<S: ExecutorSubstrate>(
    ctx: &ExecutionContext,
    substrate: &S,
    var: Option<BindingId>,
    label: Option<&str>,
    properties: &[(String, BoundExpression)],
    row: &[Value],
    row_schema: &[BindingId],
    params: &Parameters,
) -> Result<Option<Value>, ExecutionError> {
    let materialized = super::literal_lift::materialize_create_properties(
        "CreateSpineOp: node",
        properties,
        row,
        row_schema,
        params,
    )?;
    let node_id = substrate
        .create_node(ctx.tenant(), label, &materialized, ctx)
        .map_err(ExecutionError::Substrate)?;

    if var.is_some() {
        let mut node = NodeView::new(node_id, label_id_from_name(label));
        node.label_name = label.map(str::to_owned);
        for (k, v) in &materialized {
            node.properties.insert(k.clone(), v.clone());
        }
        Ok(Some(Value::Node(node)))
    } else {
        Ok(None)
    }
}

#[allow(clippy::too_many_arguments)]
fn create_rel_cell<S: ExecutorSubstrate>(
    ctx: &ExecutionContext,
    substrate: &S,
    var: Option<BindingId>,
    label: &str,
    properties: &[(String, BoundExpression)],
    source: NodeId,
    target: NodeId,
    direction: CreateRelDirection,
    row: &[Value],
    row_schema: &[BindingId],
    params: &Parameters,
) -> Result<Option<Value>, ExecutionError> {
    let (src_id, dst_id) = match direction {
        CreateRelDirection::LeftToRight => (source, target),
        CreateRelDirection::RightToLeft => (target, source),
    };
    let materialized = super::literal_lift::materialize_create_properties(
        "CreateSpineOp: rel",
        properties,
        row,
        row_schema,
        params,
    )?;
    let rel_id = substrate
        .create_rel(ctx.tenant(), src_id, dst_id, label, &materialized, ctx)
        .map_err(ExecutionError::Substrate)?;

    if var.is_some() {
        let mut rel = RelView::new(rel_id, src_id, dst_id, None);
        rel.rel_type_name = Some(label.to_owned());
        for (k, v) in &materialized {
            rel.properties.insert(k.clone(), v.clone());
        }
        Ok(Some(Value::Relationship(rel)))
    } else {
        Ok(None)
    }
}

fn node_id_from_row(
    row: &[Value],
    schema: &[BindingId],
    binding: BindingId,
    side_name: &str,
) -> Result<NodeId, ExecutionError> {
    let idx = schema_index(schema, binding).ok_or_else(|| {
        ExecutionError::Eval(format!(
            "CreateSpineOp: {side_name} binding {binding:?} not in input schema {schema:?}"
        ))
    })?;
    let cell = row.get(idx).ok_or_else(|| {
        ExecutionError::Eval(format!(
            "CreateSpineOp: {side_name} input row missing cell at index {idx}"
        ))
    })?;
    node_id_from_cell(cell, side_name)
}

fn node_id_from_cell(cell: &Value, side_name: &str) -> Result<NodeId, ExecutionError> {
    match cell {
        Value::Node(node) => Ok(node.id),
        other => Err(ExecutionError::Eval(format!(
            "CreateSpineOp: {side_name} cell is not a Node value: {other:?}"
        ))),
    }
}

fn label_id_from_name(_name: Option<&str>) -> Option<LabelId> {
    None
}
