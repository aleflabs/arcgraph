//! [`CreateVectorIndexOp`] — accept-and-register write-op for
//! `CREATE VECTOR INDEX <name> [IF NOT EXISTS] FOR (var:Label) ON
//! var.prop [OPTIONS {…}]` per **#830 / ADR-198 §OQ-7 / ADR-200**.
//!
//! Lowers from `crate::logical_plan::LogicalCreateVectorIndex`. On the
//! first `next_batch` invocation:
//!
//! 1. Resolves the index NAME. Neo4j-compatible vector clients may
//!    pass the name as a `$param`
//!    (`CREATE VECTOR INDEX $name …`); a literal name is also admitted.
//!    A `$param` name resolves against the per-query parameter bag.
//! 2. Extracts `vector.dimensions` + `vector.similarity_function` from
//!    the raw `OPTIONS` map. The real client emits
//!    `OPTIONS { indexConfig: { `vector.dimensions`: toInteger($dimensions),
//!    `vector.similarity_function`: $similarity_fn } }` — so the values
//!    are `$param`s (one wrapped in `toInteger(…)`), resolved here
//!    against the parameter bag.
//! 3. Validates dimensions > 0 when present (a clean typed error, not
//!    a panic, on a malformed / non-positive value).
//! 4. Registers a metadata entry in the per-tenant vector-index catalog
//!    via [`ExecutorSubstrate::register_vector_index`], honoring
//!    `IF NOT EXISTS` idempotency.
//!
//! # Why metadata-only (no heavyweight build)
//!
//! The served HNSW index BUILD is auto-on-ingest (#765 PART-1 — the
//! `SubstrateSearchProvider` builds the per-tenant index from every node
//! carrying the vector property). `CREATE VECTOR INDEX` therefore does
//! NOT trigger a build; it registers the name → label/property/dims/
//! similarity metadata so `SHOW VECTOR INDEXES` reflects it and
//! `db.index.vector.queryNodes(name, …)` resolves `name → property`
//! truthfully (ADR-198 §OQ-7: "does the VECTOR INDEX DDL register
//! against the same per-tenant served index?" — yes).
//!
//! # Schema
//!
//! Emits ZERO rows (Neo4j `CREATE VECTOR INDEX` returns an empty result
//! — the side effect is the registration; counters, not rows). Output
//! schema is the empty column set.
//!
//! # ADR provenance
//! - **#830 / ADR-198 §OQ-7** — the Neo4j-ecosystem vector surface
//!   split; this is the D2/D3 catalog half (mgr-dev grammar landed in
//!   #862; the substrate-binding catalog is ADR-200).
//! - **ADR-200** — the minimal in-memory per-tenant vector-index
//!   catalog design.
//! - **#765 PART-1** — auto-on-ingest served-HNSW build (why no build
//!   here).

use crate::ast::{Expression, IndexNameRef, Literal};
use crate::executor::batch::Batch;
use crate::executor::context::ExecutionContext;
use crate::executor::error::ExecutionError;
use crate::executor::eval::Parameters;
use crate::executor::substrate::{ExecutorSubstrate, VectorIndexCatalogEntry};
use crate::executor::value::Value;
use crate::semantic::bound_ast::BindingId;

/// CREATE VECTOR INDEX accept-and-register executor op (#830 / ADR-200).
#[derive(Debug)]
pub struct CreateVectorIndexOp {
    /// The index name (`$param` or literal — resolved at first-batch).
    name: IndexNameRef,
    /// `IF NOT EXISTS` present (idempotent create).
    if_not_exists: bool,
    /// The node label in `FOR (var:Label)`.
    label: String,
    /// The indexed vector property in `ON var.prop`.
    property: String,
    /// The raw `OPTIONS { … }` map (ADR-198 §OQ-7), or `None`. Parsed
    /// for `vector.dimensions` + `vector.similarity_function` at
    /// first-batch (the values may be `$param`s).
    options: Option<Expression>,
    /// Per-query parameter bag for `$name` / `$dimensions` /
    /// `$similarity_fn` resolution. Defaults to empty; set via
    /// [`Self::with_parameters`].
    parameters: Parameters,
    /// EOS flag: set once the registration has been performed.
    done: bool,
}

impl CreateVectorIndexOp {
    /// Build from the lowered
    /// `crate::logical_plan::LogicalCreateVectorIndex` fields.
    #[must_use]
    pub fn new(
        name: IndexNameRef,
        if_not_exists: bool,
        label: String,
        property: String,
        options: Option<Expression>,
    ) -> Self {
        Self {
            name,
            if_not_exists,
            label,
            property,
            options,
            parameters: Parameters::new(),
            done: false,
        }
    }

    /// Inject a per-query parameter bag (for the `$name` / `$dimensions`
    /// / `$similarity_fn` the real client passes).
    #[must_use]
    pub fn with_parameters(mut self, parameters: Parameters) -> Self {
        self.parameters = parameters;
        self
    }

    /// Output schema — empty (a DDL emits zero rows / zero columns).
    #[must_use]
    pub fn schema(&self) -> &[BindingId] {
        &[]
    }

    /// Pull the next batch — performs the registration on first call,
    /// emits ZERO rows, EOS thereafter.
    pub fn next_batch<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<Batch, ExecutionError> {
        // Defense-in-depth cancel check inside the operator.
        ctx.cancellation().check()?;
        if self.done {
            return Ok(Batch::empty(0));
        }

        // Acquire snapshot LSN per ADR-038 §2 D-18 rule 1 (defensive —
        // the outer materialize loop already holds the LSN guard).
        let _exec_lsn = ctx.ensure_snapshot_lsn();

        let name = self.resolve_name()?;
        let (dimensions, similarity_function) = self.extract_options()?;

        let entry = VectorIndexCatalogEntry {
            name,
            label: self.label.clone(),
            property: self.property.clone(),
            dimensions,
            // Clone so the local stays available for the structured log
            // below (Option<u32> `dimensions` is Copy; the String-bearing
            // similarity needs the clone).
            similarity_function: similarity_function.clone(),
        };

        // Metadata-only registration — NOT a build (the served HNSW
        // auto-builds on ingest per #765 PART-1). Honors IF NOT EXISTS
        // idempotency; a non-IF-NOT-EXISTS create on an existing name
        // propagates `SubstrateAccessError::IndexAlreadyExists`.
        let registration = substrate
            .register_vector_index(ctx.tenant(), entry, self.if_not_exists)
            .map_err(ExecutionError::Substrate)?;

        tracing::debug!(
            target: "arcgraph_query::executor::create_vector_index",
            label = %self.label,
            property = %self.property,
            dimensions = ?dimensions,
            similarity = ?similarity_function,
            registration = ?registration,
            "CREATE VECTOR INDEX registered (metadata-only; served HNSW build is auto-on-ingest)"
        );

        self.done = true;
        // CREATE VECTOR INDEX returns an empty result (0 rows).
        Ok(Batch::empty(0))
    }

    /// Resolve the index name — a literal verbatim, or a `$param`
    /// against the parameter bag (must be a string).
    fn resolve_name(&self) -> Result<String, ExecutionError> {
        match &self.name {
            IndexNameRef::Literal(s) => Ok(s.clone()),
            IndexNameRef::Param(p) => match self.parameters.get(p) {
                Some(Value::String(s)) => Ok(s.clone()),
                Some(Value::Null) | None => Err(ExecutionError::Eval(format!(
                    "CREATE VECTOR INDEX: index name parameter ${p} is unbound or NULL"
                ))),
                Some(_) => Err(ExecutionError::Eval(format!(
                    "CREATE VECTOR INDEX: index name parameter ${p} must be a string"
                ))),
            },
        }
    }

    /// Extract `(dimensions, similarity_function)` from the raw
    /// `OPTIONS` map. Returns `(None, None)` when OPTIONS is absent or
    /// omits `indexConfig`. A malformed shape surfaces a clean
    /// [`ExecutionError::Eval`] — never a panic.
    fn extract_options(&self) -> Result<(Option<u32>, Option<String>), ExecutionError> {
        let Some(opts) = &self.options else {
            return Ok((None, None));
        };
        let entries = match opts {
            Expression::Literal(Literal::Map(m)) => m,
            _ => {
                return Err(ExecutionError::Eval(
                    "CREATE VECTOR INDEX: OPTIONS must be a map literal".into(),
                ));
            }
        };
        // `indexConfig` is the nested map the real client emits. Other
        // top-level OPTIONS keys (e.g. `indexProvider`) are accepted +
        // ignored at v1.0-α.
        let Some(index_config) = entries
            .iter()
            .find(|(k, _)| k == "indexConfig")
            .map(|(_, v)| v)
        else {
            return Ok((None, None));
        };
        let cfg = match index_config {
            Expression::Literal(Literal::Map(m)) => m,
            _ => {
                return Err(ExecutionError::Eval(
                    "CREATE VECTOR INDEX: OPTIONS.indexConfig must be a map".into(),
                ));
            }
        };
        let mut dimensions = None;
        let mut similarity_function = None;
        for (k, v) in cfg {
            match k.as_str() {
                "vector.dimensions" => {
                    dimensions = Some(self.resolve_dimensions(v)?);
                }
                "vector.similarity_function" => {
                    similarity_function = Some(self.resolve_similarity(v)?);
                }
                // Unknown indexConfig keys (e.g. `vector.quantization`)
                // are accepted + ignored at v1.0-α.
                _ => {}
            }
        }
        Ok((dimensions, similarity_function))
    }

    /// Resolve a `vector.dimensions` value expression → a positive
    /// `u32`. Handles a literal int, a `$param`, and the real client's
    /// `toInteger($dimensions)` wrapper.
    fn resolve_dimensions(&self, expr: &Expression) -> Result<u32, ExecutionError> {
        let v = self.eval_option_scalar(expr, "vector.dimensions")?;
        let as_i64 = match v {
            Value::Integer(i) => i,
            Value::Float(f) if f.fract() == 0.0 => f as i64,
            _ => {
                return Err(ExecutionError::Eval(
                    "CREATE VECTOR INDEX: OPTIONS vector.dimensions must be an integer".into(),
                ));
            }
        };
        if as_i64 <= 0 || as_i64 > i64::from(u32::MAX) {
            return Err(ExecutionError::Eval(format!(
                "CREATE VECTOR INDEX: OPTIONS vector.dimensions must be a positive integer; got {as_i64}"
            )));
        }
        Ok(as_i64 as u32)
    }

    /// Resolve a `vector.similarity_function` value expression → a
    /// String. Handles a literal string + a `$param`.
    fn resolve_similarity(&self, expr: &Expression) -> Result<String, ExecutionError> {
        match self.eval_option_scalar(expr, "vector.similarity_function")? {
            Value::String(s) => Ok(s),
            _ => Err(ExecutionError::Eval(
                "CREATE VECTOR INDEX: OPTIONS vector.similarity_function must be a string".into(),
            )),
        }
    }

    /// Evaluate the narrow set of `OPTIONS`-value expressions the real
    /// client + the issue repro emit: a scalar literal, a `$param`, and
    /// a `toInteger(…)` / `toFloat(…)` wrapper around either. Anything
    /// else is a clean typed error (NOT a panic) — the OPTIONS surface
    /// is config, not a full query expression.
    fn eval_option_scalar(&self, expr: &Expression, key: &str) -> Result<Value, ExecutionError> {
        match expr {
            Expression::Literal(Literal::Integer(i)) => Ok(Value::Integer(*i)),
            Expression::Literal(Literal::Float(f)) => Ok(Value::Float(*f)),
            Expression::Literal(Literal::String(s)) => Ok(Value::String(s.clone())),
            Expression::Parameter(p) => self.parameters.get(p).cloned().ok_or_else(|| {
                ExecutionError::Eval(format!(
                    "CREATE VECTOR INDEX: OPTIONS {key} parameter ${p} is unbound"
                ))
            }),
            Expression::FunctionCall { name, args, .. }
                if name.eq_ignore_ascii_case("toInteger")
                    || name.eq_ignore_ascii_case("toFloat") =>
            {
                if args.len() != 1 {
                    return Err(ExecutionError::Eval(format!(
                        "CREATE VECTOR INDEX: OPTIONS {key}: {name}(…) expects exactly one argument"
                    )));
                }
                let inner = self.eval_option_scalar(&args[0], key)?;
                if name.eq_ignore_ascii_case("toInteger") {
                    coerce_to_integer(inner, key)
                } else {
                    coerce_to_float(inner, key)
                }
            }
            _ => Err(ExecutionError::Eval(format!(
                "CREATE VECTOR INDEX: OPTIONS {key} has an unsupported value expression \
                 (expected a literal, a $param, or toInteger(…)/toFloat(…))"
            ))),
        }
    }
}

/// `toInteger(x)` coercion over the OPTIONS scalar set (int / float /
/// numeric-string).
fn coerce_to_integer(v: Value, key: &str) -> Result<Value, ExecutionError> {
    match v {
        Value::Integer(i) => Ok(Value::Integer(i)),
        Value::Float(f) => Ok(Value::Integer(f as i64)),
        Value::String(s) => s.trim().parse::<i64>().map(Value::Integer).map_err(|_| {
            ExecutionError::Eval(format!(
                "CREATE VECTOR INDEX: OPTIONS {key}: toInteger('{s}') is not an integer"
            ))
        }),
        _ => Err(ExecutionError::Eval(format!(
            "CREATE VECTOR INDEX: OPTIONS {key}: toInteger(…) argument is non-numeric"
        ))),
    }
}

/// `toFloat(x)` coercion over the OPTIONS scalar set.
fn coerce_to_float(v: Value, key: &str) -> Result<Value, ExecutionError> {
    match v {
        Value::Integer(i) => Ok(Value::Float(i as f64)),
        Value::Float(f) => Ok(Value::Float(f)),
        Value::String(s) => s.trim().parse::<f64>().map(Value::Float).map_err(|_| {
            ExecutionError::Eval(format!(
                "CREATE VECTOR INDEX: OPTIONS {key}: toFloat('{s}') is not a number"
            ))
        }),
        _ => Err(ExecutionError::Eval(format!(
            "CREATE VECTOR INDEX: OPTIONS {key}: toFloat(…) argument is non-numeric"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use arcgraph_core::{PartitionId, TenantId};

    use super::*;
    use crate::executor::substrate::{
        StubExecutorSubstrate, SubstrateAccessError, VectorIndexRegistration,
    };

    fn ctx() -> ExecutionContext {
        ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO)
    }

    /// `OPTIONS { indexConfig: { `vector.dimensions`: <dim>,
    /// `vector.similarity_function`: <sim> } }` AST builder.
    fn options_map(dim: Expression, sim: Expression) -> Expression {
        Expression::Literal(Literal::Map(vec![(
            "indexConfig".to_string(),
            Expression::Literal(Literal::Map(vec![
                ("vector.dimensions".to_string(), dim),
                ("vector.similarity_function".to_string(), sim),
            ])),
        )]))
    }

    fn int_lit(i: i64) -> Expression {
        Expression::Literal(Literal::Integer(i))
    }
    fn str_lit(s: &str) -> Expression {
        Expression::Literal(Literal::String(s.to_string()))
    }
    fn to_integer(inner: Expression) -> Expression {
        Expression::FunctionCall {
            name: "toInteger".to_string(),
            args: vec![inner],
            distinct: false,
            star: false,
        }
    }

    #[test]
    fn create_registers_entry_with_literal_options() {
        let s = StubExecutorSubstrate::new();
        let mut op = CreateVectorIndexOp::new(
            IndexNameRef::Literal("cz806vec".into()),
            true,
            "CzChunk".into(),
            "embedding".into(),
            Some(options_map(int_lit(16), str_lit("cosine"))),
        );
        let b = op.next_batch(&ctx(), &s).unwrap();
        assert_eq!(b.row_count(), 0, "CREATE VECTOR INDEX emits zero rows");
        // Strong oracle: exact registered entry.
        let listed = s.list_vector_indexes(TenantId::DEFAULT);
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].name, "cz806vec");
        assert_eq!(listed[0].label, "CzChunk");
        assert_eq!(listed[0].property, "embedding");
        assert_eq!(listed[0].dimensions, Some(16));
        assert_eq!(listed[0].similarity_function.as_deref(), Some("cosine"));
        // EOS on the second pull.
        assert!(op.next_batch(&ctx(), &s).unwrap().is_empty());
    }

    #[test]
    fn create_resolves_param_name_and_param_options() {
        // The EXACT real-client form: $name + toInteger($dimensions) +
        // $similarity_fn.
        let s = StubExecutorSubstrate::new();
        let mut params = Parameters::new();
        params.insert("name".to_string(), Value::String("cz806vec".into()));
        params.insert("dimensions".to_string(), Value::Integer(16));
        params.insert("similarity_fn".to_string(), Value::String("cosine".into()));
        let mut op = CreateVectorIndexOp::new(
            IndexNameRef::Param("name".into()),
            true,
            "CzChunk".into(),
            "embedding".into(),
            Some(options_map(
                to_integer(Expression::Parameter("dimensions".into())),
                Expression::Parameter("similarity_fn".into()),
            )),
        )
        .with_parameters(params);
        op.next_batch(&ctx(), &s).unwrap();
        let listed = s.list_vector_indexes(TenantId::DEFAULT);
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].name, "cz806vec");
        assert_eq!(listed[0].dimensions, Some(16));
        assert_eq!(listed[0].similarity_function.as_deref(), Some("cosine"));
    }

    #[test]
    fn if_not_exists_is_idempotent_no_dup() {
        let s = StubExecutorSubstrate::new();
        let mk = || {
            CreateVectorIndexOp::new(
                IndexNameRef::Literal("cz".into()),
                true,
                "CzChunk".into(),
                "embedding".into(),
                Some(options_map(int_lit(16), str_lit("cosine"))),
            )
        };
        mk().next_batch(&ctx(), &s).unwrap();
        // Second IF NOT EXISTS create — no error, no dup.
        mk().next_batch(&ctx(), &s).unwrap();
        assert_eq!(
            s.list_vector_indexes(TenantId::DEFAULT).len(),
            1,
            "IF NOT EXISTS re-create must not duplicate the entry"
        );
    }

    #[test]
    fn create_without_if_not_exists_on_existing_is_typed_error() {
        let s = StubExecutorSubstrate::new();
        CreateVectorIndexOp::new(
            IndexNameRef::Literal("cz".into()),
            true,
            "CzChunk".into(),
            "embedding".into(),
            None,
        )
        .next_batch(&ctx(), &s)
        .unwrap();
        // Re-create WITHOUT IF NOT EXISTS → typed IndexAlreadyExists.
        let mut op = CreateVectorIndexOp::new(
            IndexNameRef::Literal("cz".into()),
            false,
            "CzChunk".into(),
            "embedding".into(),
            None,
        );
        let err = op.next_batch(&ctx(), &s).unwrap_err();
        assert!(
            matches!(
                err,
                ExecutionError::Substrate(SubstrateAccessError::IndexAlreadyExists { ref name })
                    if name == "cz"
            ),
            "expected typed IndexAlreadyExists; got {err:?}"
        );
    }

    #[test]
    fn dimensions_zero_is_clean_error_not_panic() {
        let s = StubExecutorSubstrate::new();
        let mut op = CreateVectorIndexOp::new(
            IndexNameRef::Literal("cz".into()),
            true,
            "CzChunk".into(),
            "embedding".into(),
            Some(options_map(int_lit(0), str_lit("cosine"))),
        );
        let err = op.next_batch(&ctx(), &s).unwrap_err();
        assert!(
            matches!(err, ExecutionError::Eval(ref m) if m.contains("positive integer")),
            "dims=0 → clean Eval error; got {err:?}"
        );
        assert_eq!(
            s.list_vector_indexes(TenantId::DEFAULT).len(),
            0,
            "a rejected CREATE registers nothing"
        );
    }

    #[test]
    fn malformed_options_is_clean_error_not_panic() {
        let s = StubExecutorSubstrate::new();
        // OPTIONS that is not a map literal → clean error.
        let mut op = CreateVectorIndexOp::new(
            IndexNameRef::Literal("cz".into()),
            true,
            "CzChunk".into(),
            "embedding".into(),
            Some(Expression::Literal(Literal::Integer(7))),
        );
        let err = op.next_batch(&ctx(), &s).unwrap_err();
        assert!(
            matches!(err, ExecutionError::Eval(ref m) if m.contains("OPTIONS must be a map")),
            "malformed OPTIONS → clean Eval error; got {err:?}"
        );
    }

    #[test]
    fn no_options_registers_with_none_dims_and_similarity() {
        let s = StubExecutorSubstrate::new();
        let mut op = CreateVectorIndexOp::new(
            IndexNameRef::Literal("myIdx".into()),
            false,
            "Doc".into(),
            "vec".into(),
            None,
        );
        let reg_before = s.list_vector_indexes(TenantId::DEFAULT).len();
        assert_eq!(reg_before, 0);
        op.next_batch(&ctx(), &s).unwrap();
        let listed = s.list_vector_indexes(TenantId::DEFAULT);
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].dimensions, None);
        assert_eq!(listed[0].similarity_function, None);
    }

    #[test]
    fn unbound_param_name_is_clean_error() {
        let s = StubExecutorSubstrate::new();
        let mut op = CreateVectorIndexOp::new(
            IndexNameRef::Param("name".into()),
            true,
            "CzChunk".into(),
            "embedding".into(),
            None,
        );
        // No parameters bound → unbound $name.
        let err = op.next_batch(&ctx(), &s).unwrap_err();
        assert!(
            matches!(err, ExecutionError::Eval(ref m) if m.contains("unbound")),
            "unbound $name → clean Eval error; got {err:?}"
        );
    }

    #[test]
    fn pre_cancellation_skips_registration() {
        let s = StubExecutorSubstrate::new();
        let ctx = ctx();
        ctx.cancellation().cancel();
        let mut op = CreateVectorIndexOp::new(
            IndexNameRef::Literal("cz".into()),
            true,
            "CzChunk".into(),
            "embedding".into(),
            None,
        );
        assert_eq!(op.next_batch(&ctx, &s), Err(ExecutionError::Cancelled));
        assert_eq!(s.list_vector_indexes(TenantId::DEFAULT).len(), 0);
    }

    #[test]
    fn registration_outcome_created_then_already_exists() {
        // Direct substrate-level oracle on the IF NOT EXISTS outcome enum.
        let s = StubExecutorSubstrate::new();
        let entry = VectorIndexCatalogEntry {
            name: "cz".into(),
            label: "CzChunk".into(),
            property: "embedding".into(),
            dimensions: Some(16),
            similarity_function: Some("cosine".into()),
        };
        assert_eq!(
            s.register_vector_index(TenantId::DEFAULT, entry.clone(), true)
                .unwrap(),
            VectorIndexRegistration::Created
        );
        assert_eq!(
            s.register_vector_index(TenantId::DEFAULT, entry, true)
                .unwrap(),
            VectorIndexRegistration::AlreadyExists
        );
    }

    #[test]
    fn vector_index_catalog_is_cross_tenant_isolated() {
        // R1 #872 F2 — adversarial cross-tenant isolation. The in-memory
        // catalog is `HashMap<TenantId, …>` (structurally per-tenant), but
        // every OTHER test here uses `TenantId::DEFAULT`, so nothing proves
        // the isolation. Multi-tenancy is a core correctness property on
        // this NEW surface: register in tenant A → prove tenant B sees
        // NOTHING (empty SHOW + no name→property resolution), so a
        // `db.index.vector.queryNodes("shared", …)` in B can NEVER resolve
        // to A's PRIVATE property — it falls back to the served convention
        // (`embedding`). Exact-assertion oracle (with a positive control so
        // it cannot pass vacuously).
        let s = StubExecutorSubstrate::new();
        let tenant_a = TenantId::new(101);
        let tenant_b = TenantId::new(202);

        // Register "shared" in tenant A via the FULL op path, with a
        // DISTINCTIVE property ("vec_tenant_a") that differs from the served
        // convention ("embedding") — so any cross-tenant leak is unmistakable.
        let ctx_a = ExecutionContext::new(tenant_a, PartitionId::ZERO);
        let mut op = CreateVectorIndexOp::new(
            IndexNameRef::Literal("shared".into()),
            false,
            "CzChunk".into(),
            "vec_tenant_a".into(),
            None,
        );
        op.next_batch(&ctx_a, &s).unwrap();

        // Positive control — tenant A sees its own entry + property, so the
        // index genuinely registered (the isolation assertions below are not
        // vacuously true against an empty catalog).
        let a_listed = s.list_vector_indexes(tenant_a);
        assert_eq!(a_listed.len(), 1, "tenant A registered exactly one index");
        assert_eq!(a_listed[0].name, "shared");
        assert_eq!(a_listed[0].property, "vec_tenant_a");
        assert_eq!(
            s.resolve_vector_index(tenant_a, "shared")
                .expect("tenant A resolves its own index")
                .property,
            "vec_tenant_a",
            "tenant A's queryNodes(\"shared\") truthfully resolves to A's property"
        );

        // ISOLATION — tenant B sees an empty catalog AND cannot resolve A's
        // index name. The `None` is exactly what makes a queryNodes("shared")
        // in tenant B fall back to the served convention instead of leaking
        // A's private "vec_tenant_a".
        assert!(
            s.list_vector_indexes(tenant_b).is_empty(),
            "tenant B's SHOW VECTOR INDEXES must be empty — A's entry is PRIVATE to A"
        );
        assert!(
            s.resolve_vector_index(tenant_b, "shared").is_none(),
            "tenant B must NOT resolve A's index name → queryNodes falls back to \
             convention, never A's property"
        );

        // Third isolation pin — the DEFAULT tenant (used by every other test
        // in this module) sees neither A's nor B's catalog.
        assert!(
            s.list_vector_indexes(TenantId::DEFAULT).is_empty(),
            "the DEFAULT tenant sees neither A's nor B's catalog"
        );
    }
}
