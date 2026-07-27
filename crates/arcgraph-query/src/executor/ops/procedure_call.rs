//! [`ProcedureCallOp`] — `CALL <proc>(args) [YIELD …]` / `SHOW …`
//! (ADR-197, #802).
//!
//! Lowers from [`crate::logical_plan::LogicalProcedureCall`]. A
//! generating operator: it drains its (one-unit-row) child to drive
//! execution, then emits the procedure's / SHOW's result rows — one
//! [`crate::executor::batch::Batch`] (the result sets are small), then
//! EOS. Each output row places the projected column values at their
//! [`BindingId`] slots in the op's schema, so following Filter /
//! Project operators consume them exactly like any other bindings.
//!
//! # v1.0-α scope + the honest limitation (ADR-197)
//!
//! These are SCHEMA-INTROSPECTION procedures the langchain-neo4j
//! `refresh_schema` driver sends on connect. The langchain INIT path
//! runs them against a freshly-connected graph; for an EMPTY graph the
//! correct result of `apoc.meta.data` / `db.labels` / … is ZERO rows,
//! which is exactly what this op returns at v1.0-α.
//!
//! **For a POPULATED graph the apoc/db label/rel-type/property
//! procedures are presently best-effort EMPTY** — distinct-label /
//! rel-type / property-key enumeration requires an
//! [`ExecutorSubstrate`] catalog-introspection method that is a
//! forward-follow (tracked alongside #802). This is documented (not a
//! silent no-op): the op returns an empty rowset with the correct
//! YIELD column shape so `refresh_schema` SUCCEEDS (an empty schema is
//! a valid schema), and the limitation is recorded here + in ADR-197
//! §Open-questions. `SHOW DATABASES` returns the single default-db row;
//! `db.schema.visualization` returns one empty-structure row (langchain
//! tolerates a partial/empty visualization).
//!
//! # #830 (D1 + D4) — the `Neo4jVector` search half
//!
//! Two of these procedures are NOT schema-introspection but the
//! langchain-neo4j `Neo4jVector` bootstrap + search path:
//!
//! - **`dbms.components()` (D1)** — STATIC version handshake. Returns
//!   one `(name, versions, edition)` row. The `versions[0]` value
//!   (`5.26.0`) is chosen to clear the langchain-neo4j vector gates
//!   (`is_version_5_23_or_above` → `>= (5, 23, 0)`). See
//!   `Self::dbms_components_rows`.
//! - **`db.index.vector.queryNodes(indexName, k, queryVector)` (D4)** —
//!   DYNAMIC KNN search. Evaluates its three args and calls
//!   [`ExecutorSubstrate::vector_search`] on the property resolved through the
//!   tenant's vector-index catalog (with `DEFAULT_VECTOR_PROPERTY` fallback).
//!   On an unavailable vector substrate it returns a structured
//!   `SubstrateAccessError` (NEVER silent-empty — langchain must see a
//!   real error or real hits). See `Self::vector_query_nodes_rows`.
//!
//! The D2/D3 `SHOW VECTOR INDEXES` / `CREATE VECTOR INDEX` DDL populates the
//! per-tenant catalog consumed by the D4 search path.
//!
//! # ADR provenance
//! - **ADR-197 (#802)** — the procedure-call + SHOW surface (the
//!   langchain-neo4j managed-transaction drop-in's schema half).
//! - **#830 / ADR-198 OQ-7** — the `dbms.components` + `db.index.vector
//!   .queryNodes` proc-bodies (the `Neo4jVector` search half; D1 + D4).

use crate::executor::batch::Batch;
use crate::executor::context::ExecutionContext;
use crate::executor::error::ExecutionError;
use crate::executor::eval::{Parameters, evaluate};
use crate::executor::ops::PhysicalOperator;
use crate::executor::substrate::ExecutorSubstrate;
use crate::executor::value::Value;
use crate::logical_plan::types::ProcedureSource;
use crate::semantic::bound_ast::{BindingId, BoundExpression, ProcedureKind};

/// The fallback served vector-property name.
///
/// CROSS-REFERENCES `arcgraph_mcp::tools::search::DEFAULT_VECTOR_PROPERTY`
/// (`crates/arcgraph-mcp/src/tools/search.rs`) as the source-of-truth;
/// it is duplicated here as a `const` rather than imported to AVOID an
/// `arcgraph-query → arcgraph-mcp` dependency (the wrong bounded-context
/// direction — `arcgraph-mcp` already depends on `arcgraph-query`;
/// bounded-context policy). **These two consts MUST stay in sync:** if the
/// source-of-truth value changes, update this duplicate too, or
/// `db.index.vector.queryNodes` resolves to a property holding no
/// vectors → silent-empty on the served search path (the exact failure
/// class #830 otherwise guards against). The mcp-side const carries the
/// reciprocal pointer back to here. `db.index.vector.queryNodes`
/// resolves registered names through the tenant's vector-index catalog;
/// only an unregistered name falls back to this convention for backward
/// compatibility with out-of-band served indexes. **#830 D4.**
const DEFAULT_VECTOR_PROPERTY: &str = "embedding";

/// `CALL <proc>(…) [YIELD …]` / `SHOW …` generating operator.
#[derive(Debug)]
pub struct ProcedureCallOp {
    /// The unit-row child (drained once to drive a single execution).
    child: Box<PhysicalOperator>,
    /// Procedure / SHOW source.
    source: ProcedureSource,
    /// Output schema — the projected column binding ids, in order.
    schema: Vec<BindingId>,
    /// The column NAMES paired with the schema (so the op knows which
    /// procedure column each schema slot is). Same length / order as
    /// `schema`.
    column_names: Vec<String>,
    /// The bound procedure-call argument expressions, in source order
    /// (empty for SHOW + zero-arg procedures). Evaluated at
    /// `next_batch` time for arg-bearing procedures
    /// (`db.index.vector.queryNodes`); the static procedures ignore
    /// them. **#830 D4** — the load-bearing thread that carries the args
    /// into the proc-body (bound + carried since ADR-197, now interpreted).
    args: Vec<BoundExpression>,
    /// Per-query parameter bag for evaluating `args` that reference a
    /// `$param` (e.g. langchain's `$top_k * $effective_search_ratio` k
    /// argument). Defaults to empty; set via [`Self::with_parameters`].
    parameters: Parameters,
    /// Set once the child's unit row has been consumed + the result
    /// rows emitted (the op emits its whole result in one batch).
    emitted: bool,
}

impl ProcedureCallOp {
    /// Build from the lowered [`crate::logical_plan::LogicalProcedureCall`]
    /// fields.
    #[must_use]
    pub fn new(
        child: PhysicalOperator,
        source: ProcedureSource,
        columns: Vec<(String, BindingId)>,
        args: Vec<BoundExpression>,
    ) -> Self {
        let (column_names, schema): (Vec<String>, Vec<BindingId>) = columns.into_iter().unzip();
        Self {
            child: Box::new(child),
            source,
            schema,
            column_names,
            args,
            parameters: Parameters::new(),
            emitted: false,
        }
    }

    /// Inject a per-query parameter bag (for `$param`-referencing
    /// procedure arguments — e.g. langchain's
    /// `$top_k * $effective_search_ratio`). **#830 D4.**
    #[must_use]
    pub fn with_parameters(mut self, parameters: Parameters) -> Self {
        self.parameters = parameters;
        self
    }

    /// Output schema.
    #[must_use]
    pub fn schema(&self) -> &[BindingId] {
        &self.schema
    }

    /// Pull the next batch — the full result on first call, EOS after.
    pub fn next_batch<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<Batch, ExecutionError> {
        ctx.cancellation().check()?;
        if self.emitted {
            return Ok(Batch::empty(self.schema.len()));
        }
        // Drain the child's single driving (unit) row — the procedure
        // runs once. (For v1.0-α the only consumer is a leading CALL /
        // SHOW, so the child is the unit row; draining it keeps the
        // generating-op contract uniform.)
        let _driver = self.child.next_batch(ctx, substrate)?;
        self.emitted = true;

        let rows = self.build_rows(ctx, substrate)?;
        Ok(Batch::from_rows(rows).unwrap_or_else(|| Batch::empty(self.schema.len())))
    }

    /// Materialize the result rows aligned to `self.schema`. Each row's
    /// cell `i` is the value of the `self.column_names[i]` column.
    ///
    /// Fallible + substrate-threading: the arg-bearing
    /// `db.index.vector.queryNodes` proc evaluates its args + calls the
    /// vector substrate (which can error), so this threads `ctx` +
    /// `substrate` through (the static procs ignore them). **#830 D4.**
    fn build_rows<S: ExecutorSubstrate>(
        &self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<Vec<Vec<Value>>, ExecutionError> {
        // Pre-compute the per-procedure result as a Vec of
        // `(column_name → value)` maps, then align to the schema order.
        let raw_rows: Vec<Vec<(&str, Value)>> = match &self.source {
            ProcedureSource::Procedure(kind) => self.procedure_rows(*kind, ctx, substrate)?,
            ProcedureSource::Show(kind) => Self::show_rows(*kind, ctx, substrate),
        };
        Ok(raw_rows
            .into_iter()
            .map(|row| {
                self.column_names
                    .iter()
                    .map(|col| {
                        row.iter()
                            .find(|(name, _)| name == col)
                            .map(|(_, v)| v.clone())
                            .unwrap_or(Value::Null)
                    })
                    .collect()
            })
            .collect())
    }

    /// Per-procedure result rows.
    ///
    /// The schema-introspection procedures (ADR-197) are STATIC; the
    /// `Neo4jVector` procedures (#830) are `dbms.components` (a static
    /// version row) + `db.index.vector.queryNodes` (DYNAMIC — args +
    /// substrate). See the module-level honest-limitation notes.
    fn procedure_rows<S: ExecutorSubstrate>(
        &self,
        kind: ProcedureKind,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<Vec<Vec<(&'static str, Value)>>, ExecutionError> {
        match kind {
            // apoc.meta.data / apoc.schema.nodes / db.labels /
            // db.relationshipTypes / db.propertyKeys → ZERO rows at
            // v1.0-α (correct for an empty graph; best-effort empty for
            // a populated graph pending substrate catalog-introspection
            // — module doc + ADR-197 §Open-questions).
            ProcedureKind::ApocMetaData
            | ProcedureKind::ApocSchemaNodes
            | ProcedureKind::DbLabels
            | ProcedureKind::DbRelationshipTypes
            | ProcedureKind::DbPropertyKeys => Ok(Vec::new()),
            // db.schema.visualization → one row with empty nodes/rels
            // lists (langchain tolerates a partial/empty visualization).
            ProcedureKind::DbSchemaVisualization => Ok(vec![vec![
                ("nodes", Value::List(Vec::new())),
                ("relationships", Value::List(Vec::new())),
            ]]),
            // #830 D1: static version handshake row. Arity-tolerant by
            // design — the static path never inspects `self.args`, so any
            // extra args to `dbms.components(...)` are ignored (real Neo4j
            // errors on extra args, but langchain-neo4j calls it zero-arg,
            // so the tolerance is unobservable on the customer-zero path).
            // Contrast `db.index.vector.queryNodes` below, which DOES
            // arity-check because its args are load-bearing. R1 #861 Finding #3.
            ProcedureKind::DbmsComponents => Ok(Self::dbms_components_rows()),
            // #830 D4: dynamic KNN search — eval args + vector_search.
            ProcedureKind::DbIndexVectorQueryNodes => self.vector_query_nodes_rows(ctx, substrate),
        }
    }

    /// **#830 D1** — `dbms.components()` static rows: one
    /// `(name, versions, edition)` row.
    ///
    /// `versions[0] = "5.26.0"` is the load-bearing value.
    /// Neo4j-compatible clients parse `records[0]["versions"][0]` into
    /// an int tuple and gate the
    /// vector surface on it — `has_vector_index_support` needs
    /// `>= (5, 11, 0)` and, critically, `is_version_5_23_or_above` needs
    /// `>= (5, 23, 0)` to route `db.index.vector.queryNodes` to the
    /// SUPPORTED vector path (below 5.23 langchain takes a legacy /
    /// unsupported path). `5.26.0` (a real Neo4j 5.x LTS line) clears
    /// both gates. `name = "Neo4j Kernel"` matches Neo4j's real
    /// component name (the drop-in contract); `edition = "community"`
    /// keeps `is_enterprise` false. The chosen value is asserted
    /// byte-for-byte AND re-parsed via the langchain gate in
    /// `tests/vector_proc_e2e.rs`.
    fn dbms_components_rows() -> Vec<Vec<(&'static str, Value)>> {
        vec![vec![
            ("name", Value::String("Neo4j Kernel".to_string())),
            (
                "versions",
                Value::List(vec![Value::String("5.26.0".to_string())]),
            ),
            ("edition", Value::String("community".to_string())),
        ]]
    }

    /// **#830 D4** — `db.index.vector.queryNodes(indexName, k,
    /// queryVector)` dynamic KNN search rows.
    ///
    /// Evaluates the three bound args (index name → catalog property; k →
    /// top-K; query vector → `Vec<f32>`), then calls
    /// [`ExecutorSubstrate::vector_search`] on the served vector
    /// property and emits one `(node, score)` row per ranked hit
    /// (score-descending, k-truncated by the substrate).
    ///
    /// # Index-name resolution
    ///
    /// The `indexName` arg resolves through the per-tenant vector-index
    /// catalog populated by `CREATE VECTOR INDEX`. An unregistered name falls
    /// back to [`DEFAULT_VECTOR_PROPERTY`] for compatibility with an
    /// out-of-band served index created before catalog registration.
    ///
    /// # Errors
    ///
    /// - [`ExecutionError::Eval`] on wrong arg arity / non-string index
    ///   name / non-integer k / non-list query vector (clean error,
    ///   never a panic).
    /// - [`ExecutionError::Substrate`] when the vector substrate is
    ///   unavailable — surfaced structurally, NEVER swallowed to empty
    ///   rows (langchain must see a real error or real hits). Mirrors
    ///   [`crate::executor::ops::rank_by_hybrid::RankByHybridOp`].
    fn vector_query_nodes_rows<S: ExecutorSubstrate>(
        &self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<Vec<Vec<(&'static str, Value)>>, ExecutionError> {
        // Arity: exactly (indexName, k, queryVector).
        if self.args.len() != 3 {
            return Err(ExecutionError::Eval(format!(
                "db.index.vector.queryNodes expects 3 arguments \
                 (indexName, k, queryVector); got {}",
                self.args.len()
            )));
        }
        // The arg expressions reference no input bindings (a leading
        // CALL's unit row carries none); `$param`s resolve via
        // `self.parameters`.
        let lookup = |_: BindingId| None;

        // arg 0 — index name (catalog-resolved below; validated as a string so
        // a mis-typed call is a clean error, not a confusing downstream one).
        let index_name = match evaluate(&self.args[0], &[], &lookup, &self.parameters)? {
            Value::String(s) => s,
            Value::Null => {
                return Err(ExecutionError::Eval(
                    "db.index.vector.queryNodes: indexName (arg 1) resolved to NULL".into(),
                ));
            }
            _ => {
                return Err(ExecutionError::Eval(
                    "db.index.vector.queryNodes: indexName (arg 1) must be a string".into(),
                ));
            }
        };

        // arg 1 — k (top-K). langchain sends `$top_k * $effective_search_ratio`
        // (Integer × Integer ⇒ Integer); also accept an integral Float
        // defensively (arithmetic widening). A fractional / negative k is
        // a clean error.
        let k: u64 = match evaluate(&self.args[1], &[], &lookup, &self.parameters)? {
            Value::Integer(i) if i >= 0 => i as u64,
            Value::Float(f) if f >= 0.0 && f.fract() == 0.0 => f as u64,
            Value::Integer(_) | Value::Float(_) => {
                return Err(ExecutionError::Eval(
                    "db.index.vector.queryNodes: k (arg 2) must be a non-negative integer".into(),
                ));
            }
            Value::Null => {
                return Err(ExecutionError::Eval(
                    "db.index.vector.queryNodes: k (arg 2) resolved to NULL".into(),
                ));
            }
            _ => {
                return Err(ExecutionError::Eval(
                    "db.index.vector.queryNodes: k (arg 2) must be a non-negative integer".into(),
                ));
            }
        };

        // arg 2 — query vector (List<number> → Vec<f32>). Mirrors
        // RankByHybridOp::resolve_query_vector.
        let query_vec: Vec<f32> = match evaluate(&self.args[2], &[], &lookup, &self.parameters)? {
            Value::List(elems) => {
                let mut out: Vec<f32> = Vec::with_capacity(elems.len());
                for e in elems {
                    match e {
                        Value::Float(f) => out.push(f as f32),
                        Value::Integer(i) => out.push(i as f32),
                        Value::Null => {
                            return Err(ExecutionError::Eval(
                                "db.index.vector.queryNodes: query vector (arg 3) contains NULL"
                                    .into(),
                            ));
                        }
                        _ => {
                            return Err(ExecutionError::Eval(
                                "db.index.vector.queryNodes: query vector (arg 3) element is \
                                 non-numeric"
                                    .into(),
                            ));
                        }
                    }
                }
                out
            }
            Value::Null => {
                return Err(ExecutionError::Eval(
                    "db.index.vector.queryNodes: query vector (arg 3) resolved to NULL".into(),
                ));
            }
            _ => {
                return Err(ExecutionError::Eval(
                    "db.index.vector.queryNodes: query vector (arg 3) must be a list".into(),
                ));
            }
        };

        // #830 / ADR-200 — resolve the index NAME → its property
        // TRUTHFULLY via the per-tenant vector-index catalog (the
        // `CREATE VECTOR INDEX` registration). When no catalog entry
        // matches, fall back to the served-convention property
        // (`embedding`) — back-compat with the pre-catalog advisory-name
        // behavior (#861): a `queryNodes` before any `CREATE`, or
        // against an out-of-band served index, still resolves to the
        // single served vector property. This closes R1 #861 Finding #1's
        // residual (the advisory shim becomes a real lookup).
        let resolved = substrate.resolve_vector_index(ctx.tenant(), &index_name);
        let property: String = resolved
            .as_ref()
            .map(|e| e.property.clone())
            .unwrap_or_else(|| DEFAULT_VECTOR_PROPERTY.to_string());
        tracing::debug!(
            target: "arcgraph_query::executor::procedure_call",
            index_name = %index_name,
            property = %property,
            resolved_from_catalog = resolved.is_some(),
            "db.index.vector.queryNodes: resolved index name → property via the #830/ADR-200 \
             vector-index catalog (falls back to the served-convention property when the name \
             is unregistered)"
        );

        // Unavailable-substrate handling: we deliberately do NOT pre-gate
        // on `has_vector_substrate()` (the same posture used by
        // `RankByHybridOp::fuse`).
        // That trait method is tenant-free and the production
        // `CrudExecutorSubstrate` answers it `false` unconditionally
        // (availability is per-tenant, resolved INSIDE `vector_search` via
        // the router handle + bound `SubstrateSearchProvider`) — so a
        // pre-gate would wrongly block the SERVED path. Instead we rely on
        // `vector_search` itself to surface a structured
        // `SubstrateAccessError` when unavailable: the stub returns
        // `IndexUnavailable("vector")` when no vector substrate is
        // attached, and `CrudExecutorSubstrate` returns `IndexUnavailable`
        // when the router has no vector handle or no provider is bound.
        // Either way the `?` propagates a real error — NEVER a
        // silent-empty (langchain must see a real error or real hits;
        // `feedback_review_oracle_relaxations.md`).
        //
        // MVCC visibility key — captured pre-substrate-call (same idiom as
        // `RankByHybridOp::next_batch`).
        let read_lsn = ctx.ensure_snapshot_lsn();
        let hits = substrate.vector_search(ctx.tenant(), &property, &query_vec, k, read_lsn)?;
        Ok(hits
            .into_iter()
            .map(|h| {
                vec![
                    ("node", Value::Node(h.node)),
                    ("score", Value::Float(h.score)),
                ]
            })
            .collect())
    }

    /// Per-SHOW-kind result rows.
    ///
    /// `SHOW VECTOR INDEXES` (#830 / ADR-200) reads the per-tenant
    /// vector-index catalog via the substrate; the other SHOW kinds are
    /// static (their catalogs are not surfaced at v1.0-α).
    fn show_rows<S: ExecutorSubstrate>(
        kind: crate::ast::ShowKind,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Vec<Vec<(&'static str, Value)>> {
        use crate::ast::ShowKind;
        match kind {
            // No secondary-index / constraint catalog surfaced at
            // v1.0-α → empty rowset (langchain's refresh_schema wraps
            // SHOW CONSTRAINTS in try/except and tolerates empty).
            ShowKind::Constraints => Vec::new(),
            // #830 (ADR-198 §OQ-7 / ADR-200) — `SHOW VECTOR INDEXES`
            // reflects the per-tenant vector-index catalog. Before any
            // `CREATE VECTOR INDEX` the catalog is empty → zero rows
            // (the "no such index yet" signal clients use before
            // creating the index);
            // after a CREATE it reflects the registered entry over the
            // declared columns (`name, type, entityType, labelsOrTypes,
            // properties, options`) so the client's idempotent
            // create-or-skip sees the index.
            ShowKind::Indexes | ShowKind::VectorIndexes => substrate
                .list_vector_indexes(ctx.tenant())
                .iter()
                .map(vector_index_show_row)
                .collect(),
            // One row for the single default database.
            ShowKind::Databases => vec![vec![
                ("name", Value::String("neo4j".into())),
                ("address", Value::String("localhost:7687".into())),
                ("role", Value::String("primary".into())),
                ("currentStatus", Value::String("online".into())),
            ]],
        }
    }
}

/// **#830 / ADR-200.** Build a `SHOW VECTOR INDEXES` result row from a
/// vector-index catalog entry. The column set + shape match what
/// Neo4j-compatible vector clients read:
/// `name`, `type` (`"VECTOR"`), `entityType` (`"NODE"`),
/// `labelsOrTypes[0]`, `properties[0]`, and
/// `options["indexConfig"]["vector.dimensions"]` (the dimension the
/// client validates against its embedding function) — see
/// common client-side existing-index inspection.
fn vector_index_show_row(
    e: &crate::executor::substrate::VectorIndexCatalogEntry,
) -> Vec<(&'static str, Value)> {
    use std::collections::BTreeMap;
    let mut index_config: BTreeMap<String, Value> = BTreeMap::new();
    if let Some(d) = e.dimensions {
        index_config.insert(
            "vector.dimensions".to_string(),
            Value::Integer(i64::from(d)),
        );
    }
    if let Some(sim) = &e.similarity_function {
        index_config.insert(
            "vector.similarity_function".to_string(),
            Value::String(sim.clone()),
        );
    }
    let mut options: BTreeMap<String, Value> = BTreeMap::new();
    options.insert("indexConfig".to_string(), Value::Map(index_config));
    vec![
        ("name", Value::String(e.name.clone())),
        ("type", Value::String("VECTOR".to_string())),
        ("entityType", Value::String("NODE".to_string())),
        (
            "labelsOrTypes",
            Value::List(vec![Value::String(e.label.clone())]),
        ),
        (
            "properties",
            Value::List(vec![Value::String(e.property.clone())]),
        ),
        ("options", Value::Map(options)),
    ]
}

#[cfg(test)]
mod tests {
    //! **#830 D1 + D4** op-level unit tests — construct the op directly
    //! (a unit-row [`EmptyOp`] child + a pre-baked vector stub) and drive
    //! `next_batch`, isolating the proc-body from the parser/binder. The
    //! full parse→bind→lower→execute oracles live in
    //! `tests/vector_proc_e2e.rs`.

    use arcgraph_core::{LabelId, NodeId, PartitionId, TenantId};

    use super::*;
    use crate::ast::{Expression, Literal};
    use crate::error::Span;
    use crate::executor::ops::EmptyOp;
    use crate::executor::substrate::{RankedHit, StubExecutorSubstrate, SubstrateAccessError};
    use crate::executor::value::NodeView;
    use crate::semantic::bound_ast::BoundExpression;

    fn lit_str(s: &str) -> BoundExpression {
        BoundExpression::Literal {
            value: Literal::String(s.to_string()),
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
    fn lit_vec(xs: &[f64]) -> BoundExpression {
        BoundExpression::Literal {
            value: Literal::List(
                xs.iter()
                    .map(|x| Expression::Literal(Literal::Float(*x)))
                    .collect(),
            ),
            span: Span::point(1, 1),
            type_info: None,
        }
    }

    fn node(id: u64, label: u32) -> NodeView {
        NodeView::new(NodeId::new(id), Some(LabelId::new(label)))
    }

    /// Build a `db.index.vector.queryNodes` op with `(node, score)`
    /// output columns + the given args, over a unit-row child.
    fn query_nodes_op(args: Vec<BoundExpression>) -> ProcedureCallOp {
        ProcedureCallOp::new(
            PhysicalOperator::Empty(EmptyOp::unit()),
            ProcedureSource::Procedure(ProcedureKind::DbIndexVectorQueryNodes),
            vec![
                ("node".to_string(), BindingId::new(0)),
                ("score".to_string(), BindingId::new(1)),
            ],
            args,
        )
    }

    fn vector_stub(hits: Vec<RankedHit>) -> StubExecutorSubstrate {
        let tag = StubExecutorSubstrate::vector_search_tag_for(&[1.5_f32, 0.0]);
        StubExecutorSubstrate::new()
            .with_vector_substrate()
            .with_vector_hit(TenantId::DEFAULT, "embedding", &tag, hits)
    }

    fn ctx() -> ExecutionContext {
        ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO)
    }

    #[test]
    fn dbms_components_rows_exact_and_clears_5_23_gate() {
        // The static D1 row, asserted directly off the private builder.
        let rows = ProcedureCallOp::dbms_components_rows();
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row[0], ("name", Value::String("Neo4j Kernel".to_string())));
        assert_eq!(
            row[1],
            (
                "versions",
                Value::List(vec![Value::String("5.26.0".to_string())])
            )
        );
        assert_eq!(row[2], ("edition", Value::String("community".to_string())));

        // Re-parse versions[0] the way langchain's get_version does and
        // assert it clears is_version_5_23_or_above (>= (5,23,0)).
        let Value::List(versions) = &row[1].1 else {
            panic!("versions must be a List");
        };
        let Value::String(v0) = &versions[0] else {
            panic!("versions[0] must be a String");
        };
        let tuple: Vec<i64> = v0
            .split('-')
            .next()
            .unwrap()
            .split('.')
            .map(|p| p.parse::<i64>().unwrap())
            .collect();
        assert!(
            tuple.as_slice() >= [5, 23, 0].as_slice(),
            "versions[0] must clear the langchain 5.23 vector gate; got {tuple:?}"
        );
    }

    #[test]
    fn query_nodes_op_emits_exact_hits_in_order() {
        let mut op = query_nodes_op(vec![lit_str("any"), lit_int(3), lit_vec(&[1.5, 0.0])]);
        let s = vector_stub(vec![
            RankedHit {
                node: node(1, 1),
                score: 0.99,
            },
            RankedHit {
                node: node(2, 1),
                score: 0.50,
            },
            RankedHit {
                node: node(3, 1),
                score: 0.10,
            },
        ]);
        let batch = op.next_batch(&ctx(), &s).expect("next_batch");
        assert_eq!(
            batch.rows().to_vec(),
            vec![
                vec![Value::Node(node(1, 1)), Value::Float(0.99)],
                vec![Value::Node(node(2, 1)), Value::Float(0.50)],
                vec![Value::Node(node(3, 1)), Value::Float(0.10)],
            ]
        );
    }

    #[test]
    fn query_nodes_op_truncates_to_k() {
        // Pre-bake 5, ask k=3 → exactly 3.
        let mut op = query_nodes_op(vec![lit_str("any"), lit_int(3), lit_vec(&[1.5, 0.0])]);
        let s = vector_stub(
            (1..=5)
                .map(|i| RankedHit {
                    node: node(i, 1),
                    score: 1.0 / i as f64,
                })
                .collect(),
        );
        let batch = op.next_batch(&ctx(), &s).expect("next_batch");
        assert_eq!(batch.row_count(), 3, "k=3 truncates 5 hits to 3 rows");
    }

    #[test]
    fn query_nodes_op_substrate_off_is_structured_error() {
        // No vector substrate attached → structured error, NOT empty.
        let mut op = query_nodes_op(vec![lit_str("any"), lit_int(2), lit_vec(&[1.5, 0.0])]);
        let s = StubExecutorSubstrate::new();
        let r = op.next_batch(&ctx(), &s);
        assert!(
            matches!(
                r,
                Err(ExecutionError::Substrate(SubstrateAccessError::IndexUnavailable(ref w)))
                    if w == "vector"
            ),
            "expected Substrate(IndexUnavailable(\"vector\")); got {r:?}"
        );
    }

    #[test]
    fn query_nodes_op_wrong_arity_is_clean_error() {
        // 2 args → clean Eval error (never a panic / index-out-of-bounds).
        let mut op = query_nodes_op(vec![lit_str("any"), lit_int(2)]);
        let s = vector_stub(vec![]);
        let r = op.next_batch(&ctx(), &s);
        assert!(
            matches!(r, Err(ExecutionError::Eval(ref m)) if m.contains("expects 3 arguments")),
            "expected Eval(arity); got {r:?}"
        );
    }

    #[test]
    fn query_nodes_op_non_integer_k_is_clean_error() {
        // k as a string → clean Eval error (no panic).
        let mut op = query_nodes_op(vec![lit_str("any"), lit_str("three"), lit_vec(&[1.5, 0.0])]);
        let s = vector_stub(vec![]);
        let r = op.next_batch(&ctx(), &s);
        assert!(
            matches!(r, Err(ExecutionError::Eval(ref m)) if m.contains("k (arg 2)")),
            "expected Eval(k type); got {r:?}"
        );
    }
}
