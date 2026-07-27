//! [`CreatePropertyIndexOp`] — accept-register-and-backfill write-op for
//! `CREATE INDEX <name> [IF NOT EXISTS] FOR (var:Label) ON (var.prop)`
//! (#1366, task #248 — Phase 1; `docs/design/property-index-design.md`
//! §Maintenance).
//!
//! Lowers from `crate::logical_plan::LogicalCreatePropertyIndex`. On
//! the first `next_batch` invocation it resolves the index NAME (a
//! literal or a `$param`) and calls
//! [`crate::executor::ExecutorSubstrate::create_property_index`], which:
//!
//! 1. registers the index in the durable property-index catalog as
//!    `Building`,
//! 2. backfills the MVCC-visible nodes once (extract the declared
//!    property → canonical key → insert into the secondary B+tree),
//! 3. flips `Online` co-committed with the final backfill watermark.
//!
//! `IF NOT EXISTS` is idempotent (a re-create is a no-op).
//!
//! # Schema
//!
//! Emits ZERO rows (a DDL returns an empty result — the side effect is
//! the register+backfill). Output schema is the empty column set.
//!
//! # Not query-enabled (Phase 2)
//!
//! This op only builds the index. There is no planner
//! `PropertyIndexScan` at Phase 1 — the catalog carries
//! `Building`/`Online` so the Phase-2 planner can gate visibility.

use crate::ast::IndexNameRef;
use crate::executor::batch::Batch;
use crate::executor::context::ExecutionContext;
use crate::executor::error::ExecutionError;
use crate::executor::eval::Parameters;
use crate::executor::substrate::ExecutorSubstrate;
use crate::executor::value::Value;
use crate::semantic::bound_ast::BindingId;

/// CREATE INDEX (property index) accept-register-and-backfill executor
/// op (#1366, task #248).
#[derive(Debug)]
pub struct CreatePropertyIndexOp {
    /// The index name (`$param` or literal — resolved at first-batch).
    name: IndexNameRef,
    /// `IF NOT EXISTS` present (idempotent create).
    if_not_exists: bool,
    /// The node label in `FOR (var:Label)`.
    label: String,
    /// The indexed property in `ON (var.prop)`.
    property: String,
    /// Per-query parameter bag for a `$name` resolution.
    parameters: Parameters,
    /// EOS flag: set once the register+backfill has been performed.
    done: bool,
}

impl CreatePropertyIndexOp {
    /// Build from the lowered
    /// `crate::logical_plan::LogicalCreatePropertyIndex` fields.
    #[must_use]
    pub fn new(name: IndexNameRef, if_not_exists: bool, label: String, property: String) -> Self {
        Self {
            name,
            if_not_exists,
            label,
            property,
            parameters: Parameters::new(),
            done: false,
        }
    }

    /// Inject a per-query parameter bag (for a `$name`).
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

    /// Pull the next batch — performs the register+backfill on first
    /// call, emits ZERO rows, EOS thereafter.
    pub fn next_batch<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<Batch, ExecutionError> {
        ctx.cancellation().check()?;
        if self.done {
            return Ok(Batch::empty(0));
        }
        // Defensive snapshot pin (the outer materialize loop already
        // holds the LSN guard — mirrors CreateVectorIndexOp).
        let _exec_lsn = ctx.ensure_snapshot_lsn();

        let name = self.resolve_name()?;
        let registration = substrate
            .create_property_index(
                ctx.tenant(),
                &name,
                self.if_not_exists,
                &self.label,
                &self.property,
            )
            .map_err(ExecutionError::Substrate)?;

        tracing::debug!(
            target: "arcgraph_query::executor::create_property_index",
            index = %name,
            label = %self.label,
            property = %self.property,
            registration = ?registration,
            "CREATE INDEX (property index) registered + backfilled (#1366)"
        );

        self.done = true;
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
                    "CREATE INDEX: index name parameter ${p} is unbound or NULL"
                ))),
                Some(_) => Err(ExecutionError::Eval(format!(
                    "CREATE INDEX: index name parameter ${p} must be a string"
                ))),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use arcgraph_core::{PartitionId, TenantId};

    use super::*;
    use crate::executor::substrate::{
        PropertyIndexRegistration, StubExecutorSubstrate, SubstrateAccessError,
    };

    fn ctx() -> ExecutionContext {
        ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO)
    }

    #[test]
    fn create_calls_substrate_and_emits_zero_rows() {
        let s = StubExecutorSubstrate::new();
        let mut op = CreatePropertyIndexOp::new(
            IndexNameRef::Literal("email_idx".into()),
            true,
            "User".into(),
            "email".into(),
        );
        let b = op.next_batch(&ctx(), &s).unwrap();
        assert_eq!(b.row_count(), 0, "CREATE INDEX emits zero rows");
        // Stub registers it — a re-create IF NOT EXISTS is a no-op.
        assert_eq!(
            s.create_property_index(TenantId::DEFAULT, "email_idx", true, "User", "email")
                .unwrap(),
            PropertyIndexRegistration::AlreadyExists
        );
        // EOS on the second pull.
        assert!(op.next_batch(&ctx(), &s).unwrap().is_empty());
    }

    #[test]
    fn create_without_if_not_exists_on_existing_is_typed_error() {
        let s = StubExecutorSubstrate::new();
        // First create.
        s.create_property_index(TenantId::DEFAULT, "e", false, "User", "email")
            .unwrap();
        // Re-create WITHOUT IF NOT EXISTS → typed IndexAlreadyExists.
        let err = s
            .create_property_index(TenantId::DEFAULT, "e", false, "User", "email")
            .unwrap_err();
        assert!(
            matches!(err, SubstrateAccessError::IndexAlreadyExists { ref name } if name == "e"),
            "expected typed IndexAlreadyExists; got {err:?}"
        );
    }

    #[test]
    fn unbound_param_name_is_clean_error() {
        let s = StubExecutorSubstrate::new();
        let mut op = CreatePropertyIndexOp::new(
            IndexNameRef::Param("name".into()),
            true,
            "User".into(),
            "email".into(),
        );
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
        let mut op = CreatePropertyIndexOp::new(
            IndexNameRef::Literal("e".into()),
            true,
            "User".into(),
            "email".into(),
        );
        assert_eq!(op.next_batch(&ctx, &s), Err(ExecutionError::Cancelled));
    }
}
