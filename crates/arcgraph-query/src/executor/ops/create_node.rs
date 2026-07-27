//! [`CreateNodeOp`] — write-op operator for `CREATE (var?:Label?
//! {props}) RETURN var?` per ADR-147 (W26-θ Phase 1).
//!
//! Lowers from [`crate::logical_plan::LogicalCreateNode`]. On first
//! `next_batch` invocation:
//!
//! 1. Acquires the snapshot LSN (defensive — the executor's outer
//!    `materialize` already holds the LSN guard, but every leaf op
//!    bumps the AtomicLsn handle anyway per ADR-038 §2 D-18 rule 1).
//! 2. Evaluates each property value's [`crate::ast::Literal`] to a
//!    [`crate::executor::value::Value`] (Phase 1 literal-only
//!    narrowing per ADR-147 §D-4 enforced at type-check).
//! 3. Calls [`ExecutorSubstrate::create_node`] with the tenant +
//!    label + materialized properties. The substrate opens a
//!    per-tenant `Transaction` (production wiring per ADR-031 +
//!    ADR-033) and commits at call boundary.
//! 4. Emits ONE row binding the new `NodeId` to the operator's
//!    optional variable (or an EMPTY row when the CREATE spec was
//!    anonymous like `CREATE (:User)`). Subsequent `next_batch`
//!    calls return the EOS empty batch.
//!
//! # Schema
//!
//! - When `var` is `Some(binding)` → schema = `[binding]`; the row
//!   is `[Value::Node(NodeView{id, label})]`.
//! - When `var` is `None` (anonymous) → schema is empty `[]`; the
//!   row is a 0-column tuple. The empty row is still ONE row from
//!   the executor's perspective (the openCypher TCK Eligible signal
//!   for "1 node created").
//!
//! # ADR provenance
//! - **ADR-147** — primary spec (W26-θ Phase 1).
//! - **ADR-031** + **ADR-033** — per-tenant `Transaction` discipline
//!   (commit + rollback; consolidated lifecycle ADR forward-pinned
//!   to v1.1+; the production substrate's `create_node` opens +
//!   commits).
//! - **ADR-041 §D-4** — MVCC visibility key (forward-pin; not used
//!   for the write at Phase 1 but consumed by the substrate at
//!   commit-LSN-stamping).
//! - **issue #356** — strict-schema property typing.
//! - **ADR-152-amendment-02 §D-1** — composite `List`-of-scalars
//!   property values lift here via `super::literal_lift::literal_value`
//!   (`Map` + temporal / decimal remain deferred per §D-2).

use arcgraph_core::LabelId;

use crate::executor::batch::Batch;
use crate::executor::context::ExecutionContext;
use crate::executor::error::ExecutionError;
use crate::executor::eval::Parameters;
use crate::executor::ops::PhysicalOperator;
use crate::executor::substrate::ExecutorSubstrate;
use crate::executor::value::{NodeView, Value};
use crate::semantic::bound_ast::{BindingId, BoundExpression};

/// CREATE-node executor op (ADR-147 W26-θ Phase 1).
#[derive(Debug)]
pub struct CreateNodeOp {
    /// Optional binding for the new node.
    var: Option<BindingId>,
    /// Optional label name. Forwarded verbatim to the substrate's
    /// `create_node`; the substrate handles interning per ADR-147
    /// §D-7.
    label: Option<String>,
    /// Phase 1 literal-only property values (per ADR-147 §D-4); the
    /// op materializes each `BoundExpression::Literal` to a `Value`
    /// at first-batch time.
    properties: Vec<(String, BoundExpression)>,
    /// Optional upstream sub-pipeline (issue #832). When present, this
    /// op is a streaming transform: per upstream row it performs ONE
    /// create and emits the upstream row EXTENDED with the new
    /// binding. This is how a multi-item `CREATE (a),(b),(c)` lowers —
    /// a left-deep chain where every item executes. When `None` (the
    /// chain leaf, a `CreateRel` endpoint, or a `Merge` create-branch)
    /// the op emits exactly ONE row on first call (the original
    /// leaf behavior, unchanged).
    input: Option<Box<PhysicalOperator>>,
    /// Cached schema: when `input` is present, `input.schema()`
    /// EXTENDED with `[var]`; otherwise `[var]` when `var` is `Some`,
    /// empty otherwise.
    schema: Vec<BindingId>,
    /// Cached upstream schema (`input.schema()`) for the streaming path,
    /// so property evaluation can resolve previously-bound row references
    /// (ADR-147-amendment-03 §D-1). Empty on the leaf path.
    child_schema: Vec<BindingId>,
    /// ADR-147-amendment-03 (D-1) — per-query parameter bag (threaded so
    /// a CREATE property `$param` resolves at materialization time).
    parameters: Parameters,
    /// EOS flag for the leaf path (`input == None`): set after the
    /// single-row emission. Unused on the streaming path, which is
    /// driven by the upstream's EOS.
    emitted: bool,
}

impl CreateNodeOp {
    /// Construct a fresh `CreateNodeOp` from a
    /// [`crate::logical_plan::LogicalCreateNode`].
    #[must_use]
    pub fn new(
        var: Option<BindingId>,
        label: Option<String>,
        properties: Vec<(String, BoundExpression)>,
    ) -> Self {
        let schema = match var {
            Some(b) => vec![b],
            None => Vec::new(),
        };
        Self {
            var,
            label,
            properties,
            input: None,
            schema,
            child_schema: Vec::new(),
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
    /// streaming create (issue #832 — multi-item CREATE chains). The
    /// schema becomes the upstream schema EXTENDED with this op's
    /// binding (if any), so a downstream RETURN / Project sees every
    /// chained binding. Mirrors [`super::CreateRelOp::with_input`].
    #[must_use]
    pub fn with_input(mut self, input: PhysicalOperator) -> Self {
        let child_schema: Vec<BindingId> = input.schema().to_vec();
        let mut schema = child_schema.clone();
        if let Some(b) = self.var {
            schema.push(b);
        }
        self.schema = schema;
        self.child_schema = child_schema;
        self.input = Some(Box::new(input));
        self
    }

    /// Output schema.
    pub fn schema(&self) -> &[BindingId] {
        &self.schema
    }

    pub(crate) fn rearm_leaf_endpoint(&mut self) {
        if self.input.is_none() {
            self.emitted = false;
        }
    }

    /// Pull the next batch.
    ///
    /// - **Leaf** (`input == None`): one row on first call, EOS
    ///   thereafter (the original Phase-1 behavior).
    /// - **Streaming** (`input == Some`, issue #832): per upstream row,
    ///   perform ONE create and emit the upstream row EXTENDED with the
    ///   new binding; EOS when the upstream is dry. This is how a
    ///   multi-item `CREATE (a),(b),(c)` chain executes every item.
    pub fn next_batch<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<Batch, ExecutionError> {
        // Defense-in-depth cancel check inside the operator.
        ctx.cancellation().check()?;

        // Streaming path (issue #832): an upstream chain feeds this op.
        // The `match` scopes the mutable borrow of `self.input` to the
        // pull alone, so `do_create` (which borrows `&self`) is free in
        // the loop below.
        let upstream: Option<Batch> = match self.input.as_mut() {
            Some(input) => Some(input.next_batch(ctx, substrate)?),
            None => None,
        };
        if let Some(in_batch) = upstream {
            if in_batch.is_empty() {
                // Upstream exhausted → propagate EOS.
                return Ok(Batch::empty(self.schema.len()));
            }
            // Acquire snapshot LSN per ADR-038 §2 D-18 rule 1.
            let _exec_lsn = ctx.ensure_snapshot_lsn();
            let mut out = Batch::with_capacity(self.schema.len());
            for i in 0..in_batch.row_count() {
                // Carry the upstream bindings, then append this op's
                // new node binding (if any). One create per row.
                let mut row = in_batch.row(i).to_vec();
                // ADR-147-amendment-03 (D-1): materialize against the
                // upstream prefix so a CREATE property can reference a
                // previously-bound row value / `$param`.
                if let Some(cell) = self.do_create(ctx, substrate, &row)? {
                    row.push(cell);
                }
                if !out.push_row(row) {
                    return Err(ExecutionError::Eval(
                        "CreateNodeOp: batch push overflow (chained CREATE)".into(),
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
        // Leaf path: no upstream row — property values resolve against an
        // empty row + the param bag (a `$param` still binds).
        let cell = self.do_create(ctx, substrate, &[])?;
        self.emitted = true;
        // When the CREATE was anonymous (`var` is None), `cell` is
        // None and the row is a 0-column tuple — still ONE row per the
        // openCypher TCK "1 node created" semantic.
        let row = match cell {
            Some(v) => vec![v],
            None => Vec::new(),
        };
        let mut batch = Batch::with_capacity(self.schema.len());
        if !batch.push_row(row) {
            return Err(ExecutionError::Eval(
                "CreateNodeOp: batch push overflow".into(),
            ));
        }
        Ok(batch)
    }

    /// Perform ONE substrate create-node and return the binding cell to
    /// emit: `Some(Value::Node(..))` when this op binds a variable, or
    /// `None` for an anonymous CREATE (the emitted row carries no extra
    /// cell). Shared by the leaf and streaming paths so both perform
    /// an identical write (the #832 fix only changes how many times +
    /// what prefix the row carries, never the create itself).
    fn do_create<S: ExecutorSubstrate>(
        &self,
        ctx: &ExecutionContext,
        substrate: &S,
        row: &[Value],
    ) -> Result<Option<Value>, ExecutionError> {
        // Materialize property values (ADR-147-amendment-03 §D-1 — the
        // shared const-fast-path-else-`evaluate` + runtime value-type
        // gate, identical to the live `CreateSpineOp` path so the two
        // never drift). `child_schema` is empty on the leaf path.
        let materialized = super::literal_lift::materialize_create_properties(
            "CreateNodeOp",
            &self.properties,
            row,
            &self.child_schema,
            &self.parameters,
        )?;

        // Substrate write — opens + commits a per-tenant transaction
        // per ADR-031 + ADR-033 (production wiring; stub bookkeeps in-memory).
        let node_id = substrate
            .create_node(ctx.tenant(), self.label.as_deref(), &materialized, ctx)
            .map_err(ExecutionError::Substrate)?;

        if self.var.is_some() {
            // Build a NodeView carrying the new id + materialized
            // property bag (ADR-152 §D-1 — the RETURN-after-CREATE row
            // carries the literal properties the user wrote; the
            // executor doesn't round-trip the substrate's interning
            // result here because RETURN binds node-id; downstream
            // consumers project properties via PropertyAccess against
            // the materialized bag).
            let mut node = NodeView::new(node_id, label_id_from_name(self.label.as_deref()));
            // #871 facet 3 — carry the label NAME onto the RETURN-after-
            // CREATE node so `CREATE (d:Widget) RETURN d` surfaces
            // `labels(d) == ['Widget']` (and the Bolt / JSON serializers
            // emit `["Widget"]`) instead of `[]`. The op already holds
            // the verbatim name (`self.label`); no catalog round-trip is
            // needed here (unlike the read path, which reverse-resolves
            // the LabelId via the intern table). The numeric `label`
            // stays `None` per `label_id_from_name` — RETURN binds by id,
            // and the name is the load-bearing display field.
            node.label_name = self.label.clone();
            for (k, v) in &materialized {
                node.properties.insert(k.clone(), v.clone());
            }
            Ok(Some(Value::Node(node)))
        } else {
            Ok(None)
        }
    }
}

/// Best-effort label-id reconstruction for the emitted row's
/// `NodeView`. At Phase 1 the substrate may or may not have allocated
/// a stable LabelId before this op runs (the stub substrate intern
/// table allocates lazily; the production substrate intern table
/// allocates at `create_node` call); the row carries `None` when we
/// don't know.
///
/// The downstream RETURN projection consumes `Value::Node(NodeView)`
/// by id (not by label), so returning `None` here is safe — the row's
/// node-id is the load-bearing field. A v1.1 amendment can plumb the
/// substrate-side allocated LabelId through the `create_node` return
/// value if RETURN-side label round-trip surfaces a need.
fn label_id_from_name(_name: Option<&str>) -> Option<LabelId> {
    None
}

#[cfg(test)]
mod tests {
    use arcgraph_core::{Lsn, PartitionId, TenantId};

    use super::*;
    use crate::ast::Literal;
    use crate::error::Span;
    use crate::executor::substrate::StubExecutorSubstrate;

    fn mk_literal_expr(lit: Literal) -> BoundExpression {
        BoundExpression::Literal {
            value: lit,
            span: Span::point(1, 1),
            type_info: None,
        }
    }

    #[test]
    fn create_node_emits_one_row_then_eos() {
        let s = StubExecutorSubstrate::new();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let mut op = CreateNodeOp::new(Some(BindingId::new(0)), Some("User".into()), Vec::new());
        let b1 = op.next_batch(&ctx, &s).unwrap();
        assert_eq!(b1.row_count(), 1, "first batch: one CREATE row");
        let b2 = op.next_batch(&ctx, &s).unwrap();
        assert!(b2.is_empty(), "second batch: EOS");
    }

    #[test]
    fn create_node_anonymous_emits_zero_column_row() {
        let s = StubExecutorSubstrate::new();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        // No var, no label, no properties → still emits one row per
        // openCypher v9 "1 node created" semantic. The row is a
        // 0-column tuple.
        let mut op = CreateNodeOp::new(None, None, Vec::new());
        let b1 = op.next_batch(&ctx, &s).unwrap();
        assert_eq!(b1.row_count(), 1);
        let b2 = op.next_batch(&ctx, &s).unwrap();
        assert!(b2.is_empty());
    }

    #[test]
    fn create_node_with_literal_properties_succeeds() {
        let s = StubExecutorSubstrate::new();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let props = vec![
            ("id".to_string(), mk_literal_expr(Literal::Integer(42))),
            (
                "name".to_string(),
                mk_literal_expr(Literal::String("Alice".into())),
            ),
        ];
        let mut op = CreateNodeOp::new(Some(BindingId::new(0)), Some("User".into()), props);
        let b1 = op.next_batch(&ctx, &s).unwrap();
        assert_eq!(b1.row_count(), 1);
    }

    #[test]
    fn create_node_round_trips_through_scan_nodes() {
        // ADR-147 Phase 1 smoke: CREATE then MATCH-by-label round
        // trip surfaces the new node in scan_nodes.
        let s = StubExecutorSubstrate::new();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let mut op = CreateNodeOp::new(Some(BindingId::new(0)), Some("User".into()), Vec::new());
        let _ = op.next_batch(&ctx, &s).unwrap();
        let nodes = s.scan_nodes(TenantId::DEFAULT, None, Lsn::MAX).unwrap();
        assert_eq!(nodes.len(), 1, "scan_nodes observes the CREATE-d node");
    }

    #[test]
    fn create_node_pre_cancellation_skips_substrate_call() {
        let s = StubExecutorSubstrate::new();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        ctx.cancellation().cancel();
        let mut op = CreateNodeOp::new(Some(BindingId::new(0)), Some("User".into()), Vec::new());
        let r = op.next_batch(&ctx, &s);
        assert_eq!(r, Err(ExecutionError::Cancelled));
    }
}
