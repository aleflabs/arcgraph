//! W17α M4-08+ — production-side
//! [`BoltQueryHandler`]
//! impl backed by [`super::StorageBackend`] +
//! [`arcgraph_query::QueryEngine`] +
//! [`super::CrudExecutorSubstrate`].
//!
//! The handler routes every inbound Bolt `RUN` message through the
//! same parse → bind → typecheck → cross-substrate validate → lower
//! → plan → execute pipeline as the MCP `graph.raw_query` tool. The
//! Bolt-side wire shape ([`RunOutcome`]) is derived from the
//! materialized result: the `fields` column names are the user's
//! RETURN-item display names (aliases / bare-var names / implicit
//! source-text) per #353 (falling back to `col_0..N` only for a
//! wildcard / write-only result), and row data converts via the
//! existing [`crate::transport::bolt::value::exec_to_pack_with_tenant`]
//! bridge.
//!
//! Every `RUN` also enters `crate::read_acl::authorize_read`. Principal
//! sessions execute over a permission-decorated substrate, so filtering
//! happens before projection, aggregation, ordering, or limiting can erase
//! node provenance. A principal-less non-Power session is refused at `RUN`
//! with `Neo.ClientError.Security.Unauthorized`; `scheme="none"` may complete
//! HELLO for driver compatibility but cannot read stored content.
//!
//! # Authentication
//!
//! Two postures:
//!
//! - **OAuth-enforced** (production) — when constructed via
//!   [`StorageBoltHandler::with_oauth`], the handler routes every
//!   HELLO through the shared [`crate::auth::oauth_pkce::OAuthConfig`]
//!   verifier (ADR-044 + ADR-049). Only `scheme="bearer"` with a
//!   valid JWT is admitted; the `none` / `basic` schemes are
//!   REJECTED with `Neo.ClientError.Security.Unauthorized`. Tenant
//!   derivation per ADR-011 §M7-03: the first scope carrying an
//!   `@tenant_id` suffix decides the session's tenant id (numeric
//!   suffixes decode verbatim, non-numeric ones hash-derive a
//!   deterministic tenant id via FNV-1a — see
//!   [`crate::transport::bolt::tenant_id_for_suffix`]).
//! - **Embedded / dev** (default) — accepts the `none` / `basic` /
//!   `bearer` schemes; every authenticated session scopes to
//!   [`TenantId::DEFAULT`] per the existing
//!   [`crate::transport::bolt::StubBoltHandler`] precedent. The
//!   default trip-demo posture.
//!
//! # Cancellation
//!
//! v1.0-α [`BoltQueryHandler::cancel`] is deliberately a no-op (per
//! R1 review LOW-4 on PR #349) for BOTH `qid = Some(_)` AND
//! `qid = None`. The per-handler [`CancellationRegistry`] is kept and
//! threaded into every per-RUN `QueryEngine` for forward-compatibility,
//! but the `qid → QueryId → registry.cancel()` value-bridge is
//! forward-deferred to M5-12 (issue #354). In-flight RUNs DO observe
//! cancellation — through the engine's own `CancellationToken`
//! polling (SIGTERM-driven `cancel_all` from the CLI binary, deadline
//! expiry from `execute_with_deadline`) — but NOT through the
//! `BoltQueryHandler::cancel` entry point.
//!
//! W18ε sweep (W17 retro §2.5 Pattern E closure): this module doc
//! previously claimed "a `RESET(qid)` frame routes to the corresponding
//! in-flight token", which the [`StorageBoltHandler::cancel`] body did
//! not deliver. The doc is now restated to match the actual behavior.
//!
//! # Cypher subset
//!
//! Per ADR-006 amendment-01 + ADR-038, the openCypher subset already
//! accepted by `arcgraph_query::parse` IS the supported surface; the
//! handler does NO Cypher → ArcQL textual rewriting — the grammars
//! are intentionally identical.

use std::collections::BTreeMap;
use std::sync::Arc;

use arcgraph_core::{PartitionId, TenantId};
use arcgraph_query::QueryEngine;
use arcgraph_query::cancel::CancellationRegistry;
use arcgraph_query::executor::substrate::HeldTxnHandle;

use crate::MCPError;
use crate::auth::oauth_pkce::OAuthConfig;
use crate::read_acl::{PermissionEnforcedSubstrate, ReadAccess, authorize_read};
use crate::scope::SessionScope;
use crate::transport::bolt::auth::{BoltOAuthValidator, tenant_id_from_claims};
use crate::transport::bolt::error::BoltError;
use crate::transport::bolt::handler::{BoltQueryHandler, BoltSessionAuth, RunOutcome};
use crate::transport::bolt::packstream::PackValue;
use crate::transport::bolt::value::{exec_to_pack_with_tenant, pack_params_to_exec};

use super::StorageBackend;
use super::adapters::value_to_json;
use super::substrate::{BoltHeldTxn, CrudExecutorSubstrate, SubstrateSearchProvider};

/// Production-side [`BoltQueryHandler`] backed by the same storage
/// substrate the MCP-side adapters consume.
///
/// Shareable via `Arc` per the trait's `Send + Sync + 'static` bound.
pub struct StorageBoltHandler {
    backend: StorageBackend,
    substrate: CrudExecutorSubstrate,
    /// Shared cancellation registry threaded into every per-RUN
    /// `QueryEngine` for forward-compatibility. v1.0-α the
    /// [`StorageBoltHandler::cancel`] entry point does NOT bridge
    /// `qid → QueryId → registry.cancel()` (forward-deferred to
    /// M5-12, issue #354); cancellation reaches in-flight RUNs via
    /// the engine's own [`arcgraph_query::cancel::CancellationToken`]
    /// polling (SIGTERM-driven `cancel_all` from the CLI; deadline
    /// expiry from `execute_with_deadline`). Holding the registry
    /// here keeps the value-bridge wire ready for the M5-12 slice
    /// without a v1.0-α breaking change.
    cancellation: Arc<CancellationRegistry>,
    /// W19γ ADR-049 — optional OAuth validator. When `Some`, every
    /// HELLO is verified against the shared OAuthConfig (same JWKS,
    /// same scope vocabulary as the HTTP/TLS transport). When `None`
    /// (the default — trip-demo posture), the legacy
    /// `none`/`basic`/`bearer` schemes are accepted without
    /// signature verification. Production deployments wire OAuth via
    /// [`StorageBoltHandler::with_oauth`].
    oauth: Option<Arc<BoltOAuthValidator>>,
    /// #1291 — optional per-tenant memory cap (bytes) applied to every
    /// RUN (auto-commit AND explicit-tx). When `Some(cap)`, each RUN
    /// mints a per-query [`arcgraph_query::executor::MemoryBudget`]
    /// with the cap configured for the session tenant and attaches it
    /// to the per-RUN `QueryEngine` — a heavy query surfaces
    /// `Neo.TransientError.General.OutOfMemoryError` instead of OOMing
    /// the served process. `None` (the default — embedded / test
    /// posture) preserves the pre-#1291 opt-in behavior. The served
    /// binary wires this from `ARCGRAPH_TENANT_MEMORY_CAP_BYTES`
    /// (default 1 GiB).
    per_tenant_memory_cap_bytes: Option<u64>,
}

impl std::fmt::Debug for StorageBoltHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StorageBoltHandler")
            .field("backend", &self.backend)
            .field("substrate", &self.substrate)
            .field("cancellation", &"<Arc<CancellationRegistry>>")
            .field(
                "oauth",
                &self.oauth.as_ref().map(|_| "<BoltOAuthValidator>"),
            )
            .finish()
    }
}

impl StorageBoltHandler {
    /// Construct a Bolt handler over a shared backend.
    #[must_use]
    pub fn new(backend: StorageBackend) -> Self {
        let substrate = CrudExecutorSubstrate::new(
            Arc::clone(backend.router()),
            Arc::clone(backend.txn_manager()),
            Arc::clone(backend.intern_table()),
        );
        Self {
            backend,
            substrate,
            cancellation: Arc::new(CancellationRegistry::new()),
            oauth: None,
            per_tenant_memory_cap_bytes: None,
        }
    }

    /// #1291 — enable the per-tenant memory budget with `cap_bytes` as
    /// the byte ceiling for every session tenant. Each RUN mints a
    /// per-query [`arcgraph_query::executor::MemoryBudget`] with the
    /// cap configured for the requesting tenant so blocking operators
    /// (sort / join / aggregate / distinct / expand spillover) and the
    /// materialize tail enforce a REAL byte ceiling instead of the
    /// ≈4.29 B-row `UNCAPPED_RUNAWAY_GUARD_ROWS` fallback.
    /// Builder-style; chains after [`Self::new`].
    #[must_use]
    pub fn with_per_tenant_memory_cap(mut self, cap_bytes: u64) -> Self {
        self.per_tenant_memory_cap_bytes = Some(cap_bytes);
        self
    }

    /// #1291 — attach the per-query memory budget (cap set for
    /// `tenant`) to `engine` when a cap is configured; identity
    /// otherwise. Shared by the auto-commit [`Self::run`] and
    /// explicit-tx [`Self::run_in_txn`] paths so both enforce the same
    /// ceiling.
    fn apply_memory_cap<'cat, C>(
        &self,
        engine: QueryEngine<'cat, C>,
        tenant: TenantId,
    ) -> QueryEngine<'cat, C>
    where
        C: arcgraph_query::semantic::CatalogProvider,
    {
        match self.per_tenant_memory_cap_bytes {
            Some(cap) => engine.with_memory_budget(
                arcgraph_query::executor::MemoryBudget::with_per_tenant_cap(tenant, cap),
            ),
            None => engine,
        }
    }

    /// Builder: enable OAuth 2.1 + PKCE Bearer-token validation on
    /// every HELLO per ADR-049. Pass an `Arc<OAuthConfig>` so the
    /// same config can drive the HTTP/TLS transport (W16β ADR-044)
    /// concurrently — the JWKS + scope vocabulary + issuer / audience
    /// are unified across transports.
    ///
    /// When wired, the handler REJECTS `scheme="none"` and
    /// `scheme="basic"` with `Neo.ClientError.Security.Unauthorized`.
    /// Only valid bearer JWTs with ≥1 of `{arcgraph.read, arcgraph.write}`
    /// in their scope claim are admitted.
    #[must_use]
    pub fn with_oauth(mut self, config: Arc<OAuthConfig>) -> Self {
        self.oauth = Some(Arc::new(BoltOAuthValidator::new(config)));
        self
    }

    /// #765 PART-1 — bind the served vector-search provider into the wrapped
    /// substrate so a Bolt `RANK BY vector(n.embedding, $qv)` Cypher query runs
    /// real HNSW KNN (symmetric with the MCP `graph.search` + `graph.raw_query`
    /// wiring; the Bolt handler runs the same `QueryEngine` over the same
    /// `CrudExecutorSubstrate` seam). Builder-style; chains after [`Self::new`].
    #[must_use]
    pub fn with_search_provider(mut self, provider: Arc<dyn SubstrateSearchProvider>) -> Self {
        self.substrate = self.substrate.with_search_provider(provider);
        self
    }

    /// Whether OAuth validation is enabled on this handler.
    #[must_use]
    pub fn oauth_enforced(&self) -> bool {
        self.oauth.is_some()
    }

    /// Borrow the wrapped substrate. Exposed for tests + the
    /// integration layer.
    pub fn substrate(&self) -> &CrudExecutorSubstrate {
        &self.substrate
    }

    /// Borrow the cancellation registry. M5-12 forward will replace
    /// the per-handler registry with a workspace-shared one so
    /// cancellations route correctly across multiple per-tenant
    /// handlers.
    pub fn cancellation_registry(&self) -> &Arc<CancellationRegistry> {
        &self.cancellation
    }

    /// Resolve one Bolt statement's ADR-212 visibility through the same
    /// [`authorize_read`] choke point as graph.inspect/explore/search.
    fn read_access(&self, session: &BoltSessionAuth) -> Result<ReadAccess, BoltError> {
        let tenant = session.tenant();
        authorize_read("Bolt RUN", session.principal(), session.scope(), || {
            let handle = self
                .backend
                .router()
                .route(tenant, PartitionId::ZERO)
                .map_err(|error| {
                    MCPError::TenantUnknown(format!("Bolt ACL routing failed: {error}"))
                })?;
            Ok(Some(Arc::clone(handle.permissions())))
        })
        .map_err(translate_read_acl_to_bolt)
    }
}

impl BoltQueryHandler for StorageBoltHandler {
    fn authenticate(
        &self,
        scheme: Option<&str>,
        principal: Option<&str>,
        credentials: Option<&str>,
    ) -> Result<BoltSessionAuth, BoltError> {
        // W19γ ADR-049: when OAuth is enforced, route through the
        // shared validator. The validator enforces bearer-scheme +
        // signature + HELLO-time scope policy; tenant derivation per
        // ADR-011 §M7-03.
        if let Some(validator) = &self.oauth {
            let claims = validator.authenticate_hello(scheme, principal, credentials)?;
            return Ok(BoltSessionAuth::new(
                tenant_id_from_claims(&claims),
                principal.filter(|p| !p.is_empty()).map(str::to_owned),
                SessionScope::from_scope_claim(&claims.scope),
            ));
        }
        let scheme = scheme.unwrap_or("none");
        match scheme {
            "none" => Ok(BoltSessionAuth::new(
                TenantId::DEFAULT,
                None,
                SessionScope::Read,
            )),
            "basic" | "bearer" => {
                if principal.unwrap_or("").is_empty() {
                    return Err(BoltError::Unauthorized(format!(
                        "principal required for `{scheme}` auth"
                    )));
                }
                Ok(BoltSessionAuth::new(
                    TenantId::DEFAULT,
                    principal.map(str::to_owned),
                    SessionScope::Read,
                ))
            }
            other => Err(BoltError::Unauthorized(format!(
                "unsupported auth scheme: {other}"
            ))),
        }
    }

    fn run(
        &self,
        session: &BoltSessionAuth,
        cypher: &str,
        parameters: &BTreeMap<String, PackValue>,
    ) -> Result<RunOutcome, BoltError> {
        let tenant = session.tenant();
        // Defense-in-depth: the dispatcher's auth path already
        // routed `tenant` from `authenticate`; we re-route through
        // the storage backend so a future caller that mis-wires the
        // tenant value catches the routing error here.
        if self
            .backend
            .router()
            .route(tenant, PartitionId::ZERO)
            .is_err()
        {
            return Err(BoltError::Internal(format!(
                "tenant {tenant:?} not routable via storage"
            )));
        }

        // #797 — convert the RUN `parameters` map into the executor's
        // parameter bag BEFORE planning. A shape rejection (Node / Rel /
        // Path / Bytes value) surfaces as a client `InvalidParameter`
        // error naming the bad key.
        let params = params_to_bag(parameters)?;

        // Build the per-call catalog as for the MCP `RawQueryExecutor`.
        let cat = super::adapters::build_catalog_for_tenant(tenant, &self.backend);
        // #1291 — attach the per-tenant memory budget (if configured)
        // so a heavy RUN errors with the budget taxonomy instead of
        // OOMing the served process.
        let engine = self.apply_memory_cap(
            QueryEngine::new(&cat).with_cancellation_registry((*self.cancellation).clone()),
            tenant,
        );
        let access = self.read_access(session)?;
        let substrate = PermissionEnforcedSubstrate::new(self.substrate.clone(), access.clone());

        let result = engine
            .execute_with_deadline_and_parameters(
                cypher,
                &substrate,
                std::time::Duration::from_millis(arcgraph_query::cancel::DEFAULT_QUERY_TIMEOUT_MS),
                params,
            )
            .map_err(translate_explain_to_bolt)?;

        Ok(materialized_to_outcome(&result, tenant, &access))
    }

    fn cancel(&self, _tenant: TenantId, qid: Option<i64>) {
        // v1.0-α deliberately-no-op: per R1 review LOW-4 (PR #349)
        // the per-handler `CancellationRegistry` is kept (threaded
        // into the engine for forward-compatibility) but the
        // `qid → QueryId → registry.cancel()` value bridge is
        // forward-deferred to M5-12 (issue #354). The dispatcher's
        // rate-limiter still throttles the spam path; an in-flight
        // RUN observes cancellation through the engine's own
        // CancellationToken polling, not through this entry point.
        let _ = qid;
    }

    // ── ADR-197 explicit transactions ──

    fn begin_txn(
        &self,
        tenant: TenantId,
        _mode: Option<&str>,
        _db: Option<&str>,
    ) -> Result<Box<dyn HeldTxnHandle>, BoltError> {
        // Defense-in-depth tenant routing check (symmetric with run()).
        if self
            .backend
            .router()
            .route(tenant, PartitionId::ZERO)
            .is_err()
        {
            return Err(BoltError::Internal(format!(
                "tenant {tenant:?} not routable via storage"
            )));
        }
        // Open ONE owned MVCC transaction (ADR-197 layer 1) the
        // connection holds across RUNs until COMMIT / ROLLBACK. `mode`
        // (r/w) + `db` are accepted; v1.0-α is single-db and does not
        // gate writes on `mode` (the openCypher statement itself
        // decides read vs write).
        let crud = self.substrate.crud_for(tenant).map_err(|e| {
            BoltError::Internal(format!("failed to resolve transaction store: {e}"))
        })?;
        let owned = self.backend.txn_manager().begin_owned(tenant);
        Ok(Box::new(BoltHeldTxn::new_with_abort_store(owned, crud)))
    }

    fn run_in_txn(
        &self,
        session: &BoltSessionAuth,
        cypher: &str,
        parameters: &BTreeMap<String, PackValue>,
        held: Box<dyn HeldTxnHandle>,
    ) -> (Result<RunOutcome, BoltError>, Box<dyn HeldTxnHandle>) {
        let tenant = session.tenant();
        if self
            .backend
            .router()
            .route(tenant, PartitionId::ZERO)
            .is_err()
        {
            return (
                Err(BoltError::Internal(format!(
                    "tenant {tenant:?} not routable via storage"
                ))),
                held,
            );
        }
        // #797 — bind RUN parameters; on a shape rejection return the
        // UNTOUCHED held tx so the caller can continue / abort it (the
        // Bolt FSM moves to Failed; RESET / ROLLBACK aborts).
        let params = match params_to_bag(parameters) {
            Ok(p) => p,
            Err(e) => return (Err(e), held),
        };
        let cat = super::adapters::build_catalog_for_tenant(tenant, &self.backend);
        // #1291 — same budget attachment as the auto-commit path.
        let engine = self.apply_memory_cap(
            QueryEngine::new(&cat).with_cancellation_registry((*self.cancellation).clone()),
            tenant,
        );
        let access = match self.read_access(session) {
            Ok(access) => access,
            Err(error) => return (Err(error), held),
        };
        let substrate = PermissionEnforcedSubstrate::new(self.substrate.clone(), access.clone());
        // ADR-197: stage into the held tx (no commit). `execute_in_txn`
        // installs the handle on a fresh ExecutionContext, runs
        // materialize (write ops stage into the held tx via the
        // substrate's `run_txn`/`stage_or_commit` EXPLICIT branch), and
        // returns the moved-back handle.
        let (result, held) = engine.execute_in_txn_with_parameters(
            cypher,
            &substrate,
            held,
            std::time::Duration::from_millis(arcgraph_query::cancel::DEFAULT_QUERY_TIMEOUT_MS),
            params,
        );
        let outcome = result
            .map(|r| materialized_to_outcome(&r, tenant, &access))
            .map_err(translate_explain_to_bolt);
        (outcome, held)
    }

    fn commit_txn(&self, held: Box<dyn HeldTxnHandle>) -> Result<String, BoltError> {
        // ADR-197 #802 R1 finding #1 — commit the held tx through the
        // SAME FULL crud::commit machinery the auto-commit path uses
        // (primary-index dual-write via take_installs + install_create +
        // primary.upsert_deferred, WAL CommitBundle, CDC flush_commit,
        // TEL drain), NOT the MVCC-version-store-only `OwnedTxn::commit`.
        // The managed-tx writes buffered their installs/CDC/TEL into the
        // substrate's per-tenant CrudStore keyed by the tx id during
        // run_in_txn staging; `commit_held_txn` routes back to that SAME
        // store and drains them under one CommitBundle fsync — so an
        // explicit-tx COMMIT is byte-for-byte the auto-commit semantics,
        // differing only in tx lifetime. On MVCC write-write conflict /
        // commit failure the error maps to the Bolt transaction-error
        // taxonomy.
        let lsn = self
            .substrate
            .commit_bolt_held_handle(held)
            .map_err(|e| match e {
                // #907 — an explicit-tx (driver `execute_write`) COMMIT
                // that loses the OCC race is RETRIABLE: map the typed
                // conflict to `TransientConflict` (→ `Neo.TransientError.*`),
                // NOT the fatal `Internal` → `Neo.DatabaseError`.
                arcgraph_query::executor::SubstrateAccessError::Conflict { target } => {
                    BoltError::TransientConflict { target }
                }
                other => BoltError::Internal(format!("transaction commit failed: {other}")),
            })?;
        Ok(format!("arcgraph:{}", lsn.raw()))
    }

    fn rollback_txn(&self, mut held: Box<dyn HeldTxnHandle>) {
        // Abort the held tx — discards all staged writes (the real
        // ROLLBACK). If the move-out somehow fails (already taken, or
        // not a BoltHeldTxn), dropping the box still aborts via
        // OwnedTxn's Drop (no leak).
        match held.as_any_mut().downcast_mut::<BoltHeldTxn>() {
            Some(_) => {
                let held = held
                    .as_any_mut()
                    .downcast_mut::<BoltHeldTxn>()
                    .expect("type checked above");
                let replacement = BoltHeldTxn::new_empty_for_finalized_handle();
                let held = std::mem::replace(held, replacement);
                held.abort();
            }
            None => {
                if let Ok(owned) = take_owned(held) {
                    owned.abort()
                }
            }
        }
    }
}

/// ADR-197 — MOVE the concrete [`OwnedTxn`](arcgraph_storage::transaction::OwnedTxn)
/// out of the opaque `Box<dyn HeldTxnHandle>` the server stores on the
/// connection, so COMMIT / ROLLBACK can CONSUME it (`commit(self)` /
/// `abort(self)`).
///
/// Trait-object upcasting (`Box<dyn HeldTxnHandle>` → `Box<dyn Any>`)
/// is not stable at the 1.85 MSRV, so we go through the
/// [`HeldTxnHandle::as_any_mut`] downcast seam + `Option::take` to
/// move the `OwnedTxn` out of the boxed `BoltHeldTxn`. The box itself
/// is dropped after; the moved-out `OwnedTxn` carries the staged
/// write-set.
fn take_owned(
    mut held: Box<dyn HeldTxnHandle>,
) -> Result<arcgraph_storage::transaction::OwnedTxn, BoltError> {
    held.as_any_mut()
        .downcast_mut::<BoltHeldTxn>()
        .and_then(BoltHeldTxn::take_owned)
        .ok_or_else(|| {
            BoltError::Internal(
                "held-txn handle is not a BoltHeldTxn, or already finalized (ADR-197)".into(),
            )
        })
}

/// ADR-197 — shared `MaterializedResult` → [`RunOutcome`] conversion
/// for both the auto-commit [`StorageBoltHandler::run`] and the
/// explicit-tx [`StorageBoltHandler::run_in_txn`] paths.
///
/// #353 — `fields` now carries the user's RETURN-item display names
/// (aliases / bare-var names / implicit source-text) surfaced by
/// `MaterializedResult::columns`, via the SAME
/// [`super::adapters::column_names_for_result`] resolver the MCP
/// `RawQueryRows` renderer uses (single source of truth — both wire
/// paths emit identical column names). The Bolt RUN SUCCESS metadata's
/// `fields` list is what the neo4j driver keys each record by, so
/// `RETURN n.name AS name` now yields a record keyed `name` (langchain's
/// Neo4jGraph drop-in) instead of `col_0`. Falls back to `col_0..N` only
/// when the engine reports no names (wildcard / write-only). Rows
/// convert via `exec_to_pack_with_tenant`.
fn materialized_to_outcome(
    result: &arcgraph_query::MaterializedResult,
    tenant: TenantId,
    access: &ReadAccess,
) -> RunOutcome {
    let tenant_slug = tenant.raw().to_string();
    let fields: Vec<String> = super::adapters::column_names_for_result(result);
    let mut records: Vec<Vec<PackValue>> = Vec::with_capacity(result.rows().len());
    for row in result.rows() {
        if !access.allows_row(row) {
            continue;
        }
        records.push(
            row.iter()
                .map(|v| exec_to_pack_with_tenant(v, &tenant_slug))
                .collect(),
        );
    }
    RunOutcome {
        fields,
        records,
        qid: None,
    }
}

/// Bolt-protocol mapping for failures from the shared ADR-212 read seam.
///
/// Missing/empty principals are authentication failures and therefore use
/// `Neo.ClientError.Security.Unauthorized` on RUN. A missing tenant permission
/// index is a server wiring fault and stays `Neo.DatabaseError.General.UnknownError`;
/// both postures fail closed and emit no records.
fn translate_read_acl_to_bolt(error: MCPError) -> BoltError {
    let detail = error.to_string();
    match error {
        MCPError::Forbidden { .. } | MCPError::InvalidParams(_) => {
            BoltError::Unauthorized(format!("Bolt principal ACL refused RUN: {detail}"))
        }
        _ => BoltError::Internal(format!("Bolt principal ACL unavailable: {detail}")),
    }
}

/// Translate `arcgraph_query::ExplainError` to a `BoltError` per the
/// Bolt-side error taxonomy. `ExplainError` is `#[non_exhaustive]` so
/// the catch-all routes unknown variants to `BoltError::Internal` for
/// forward-additive safety.
fn translate_explain_to_bolt(err: arcgraph_query::ExplainError) -> BoltError {
    use arcgraph_query::ExplainError;
    use arcgraph_query::executor::SubstrateAccessError;
    use arcgraph_query::semantic::error::ArcQLError;
    match err {
        ExplainError::Parse(e) => BoltError::Syntax(format!("{e}")),
        ExplainError::ArcQL(e @ ArcQLError::ResourceExhausted { .. }) => {
            BoltError::ResourceExhausted(format!("{e}"))
        }
        ExplainError::ArcQL(e) => BoltError::Syntax(format!("{e}")),
        ExplainError::ExecutionEval(detail) => BoltError::Internal(detail),
        // #907 — a write-write MVCC conflict is a RETRIABLE transient
        // transaction error, NOT a fatal DatabaseError. Detect the TYPED
        // `SubstrateAccessError::Conflict` variant (threaded from
        // `crud::commit`'s `CrudError::Mvcc`; never string-matched) and
        // map it to `BoltError::TransientConflict` (→ `Neo.TransientError.*`,
        // which drivers auto-retry). This arm MUST precede the generic
        // `Substrate(_)` arm below.
        ExplainError::Substrate(SubstrateAccessError::Conflict { target }) => {
            BoltError::TransientConflict { target }
        }
        ExplainError::Substrate(detail) => BoltError::Internal(format!("substrate: {detail}")),
        ExplainError::Cancelled => BoltError::Cancelled,
        // #797 — a missing `$name` is a CLIENT fault → ParameterMissing
        // (`Neo.ClientError.Statement.ParameterMissing`), NOT the
        // `Neo.DatabaseError`-class Internal bucket that the pre-fix
        // `Eval("missing parameter")` → `ExecutionEval` path rendered.
        ExplainError::MissingParameter { name } => BoltError::ParameterMissing(format!("${name}")),
        other => BoltError::Internal(format!("query-layer error: {other:?}")),
    }
}

/// #797 — convert the inbound Bolt RUN `parameters` map into the
/// executor parameter bag, mapping a shape rejection (a Node /
/// Relationship / Path PackStream struct, or raw Bytes, as a value) to
/// a client [`BoltError::InvalidParameter`] naming the offending key.
fn params_to_bag(
    parameters: &BTreeMap<String, PackValue>,
) -> Result<arcgraph_query::executor::eval::Parameters, BoltError> {
    pack_params_to_exec(parameters)
        .map_err(|(name, e)| BoltError::InvalidParameter(format!("parameter `{name}`: {e}")))
}

/// Convenience: render an executor row as a JSON array for
/// diagnostic surfaces (the Bolt path uses `exec_to_pack_with_tenant`
/// directly, but logs / metrics paths reach for `serde_json::Value`).
#[doc(hidden)]
#[must_use]
pub fn row_to_json(row: &[arcgraph_query::executor::Value]) -> serde_json::Value {
    serde_json::Value::Array(row.iter().map(value_to_json).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arcgraph_storage::InternTable;
    use arcgraph_storage::buffer::BufferPool;
    use arcgraph_storage::catalog::SystemCatalog;
    use arcgraph_storage::crud::CrudStore;
    use arcgraph_storage::io::InMemoryPageIo;
    use arcgraph_storage::router::MultiTenantRouter;
    use arcgraph_storage::transaction::TxnManager;

    fn fixture() -> StorageBackend {
        let io = Arc::new(InMemoryPageIo::new());
        let pool = BufferPool::new(8, io);
        let mgr = Arc::new(TxnManager::new());
        let catalog = Arc::new(SystemCatalog::new());
        catalog.bootstrap(&pool, &mgr).expect("bootstrap");
        let crud = Arc::new(CrudStore::new());
        let router = Arc::new(MultiTenantRouter::new(catalog, Arc::clone(&crud), None));
        let intern = Arc::new(InternTable::new());
        StorageBackend::new(router, mgr, intern)
    }

    fn power_session() -> BoltSessionAuth {
        BoltSessionAuth::new(TenantId::DEFAULT, None, SessionScope::Power)
    }

    #[test]
    fn rollback_drains_slotted_transaction_scratch() {
        let backend = fixture();
        let crud = Arc::clone(
            backend
                .router()
                .route(TenantId::DEFAULT, PartitionId::ZERO)
                .expect("default tenant")
                .crud(),
        );
        let handler = StorageBoltHandler::new(backend);
        let owned = handler.backend.txn_manager().begin_owned(TenantId::DEFAULT);
        crud.blob_store()
            .stage_bag(TenantId::DEFAULT, owned.id(), b"abort-me")
            .expect("stage slotted bag");
        assert_eq!(crud.blob_store().txn_slotted_scratch_count(), 1);
        handler.rollback_txn(Box::new(BoltHeldTxn::new_with_abort_store(
            owned,
            Arc::clone(&crud),
        )));
        assert_eq!(crud.blob_store().txn_slotted_scratch_count(), 0);
    }

    #[test]
    fn connection_drop_loop_drains_slotted_transaction_scratch() {
        let backend = fixture();
        let crud = Arc::clone(
            backend
                .router()
                .route(TenantId::DEFAULT, PartitionId::ZERO)
                .expect("default tenant")
                .crud(),
        );
        for _ in 0..100 {
            let owned = backend.txn_manager().begin_owned(TenantId::DEFAULT);
            crud.blob_store()
                .stage_bag(TenantId::DEFAULT, owned.id(), b"drop-me")
                .expect("stage slotted bag");
            drop(BoltHeldTxn::new_with_abort_store(owned, Arc::clone(&crud)));
        }
        assert_eq!(crud.blob_store().txn_slotted_scratch_count(), 0);
    }

    #[test]
    fn authenticate_accepts_none_scheme_without_principal() {
        let h = StorageBoltHandler::new(fixture());
        let t = h.authenticate(Some("none"), None, None).expect("auth");
        assert_eq!(t.tenant(), TenantId::DEFAULT);
        assert_eq!(t.principal(), None);
        assert_eq!(t.scope(), SessionScope::Read);
    }

    #[test]
    fn authenticate_rejects_basic_with_empty_principal() {
        let h = StorageBoltHandler::new(fixture());
        let err = h
            .authenticate(Some("basic"), Some(""), None)
            .expect_err("rejects");
        assert!(matches!(err, BoltError::Unauthorized(_)));
    }

    #[test]
    fn authenticate_accepts_bearer_with_principal() {
        let h = StorageBoltHandler::new(fixture());
        let t = h
            .authenticate(Some("bearer"), Some("alice"), Some("tok"))
            .expect("auth");
        assert_eq!(t.tenant(), TenantId::DEFAULT);
        assert_eq!(t.principal(), Some("alice"));
    }

    /// #871 — a real Bolt `RUN … RETURN <node>` packs the node's label
    /// NAME into the Bolt 5.0 Node struct's labels field (the field a
    /// JS / Python `neo4j` driver reads as `node.labels`), never the
    /// opaque `"LabelId(N)"` debug form. This drives the FULL Bolt query
    /// path end-to-end: `run` → `QueryEngine::execute` → the CREATE op's
    /// name-carry → `materialized_to_outcome` → `exec_to_pack_with_tenant`
    /// → `pack_node_with_tenant`. CREATE-RETURN is used (no MATCH /
    /// catalog-stats dependency) so the assertion isolates the
    /// serialization path.
    #[test]
    fn bolt_run_create_returned_node_packs_label_name() {
        let h = StorageBoltHandler::new(fixture());
        let out = h
            .run(
                &power_session(),
                "CREATE (d:Widget) RETURN d",
                &std::collections::BTreeMap::new(),
            )
            .expect("RUN CREATE … RETURN d");
        assert_eq!(out.records.len(), 1, "one record for one created node");
        let node = &out.records[0][0];
        match node {
            PackValue::Struct { fields, .. } => {
                // Bolt 5.0 Node: fields = [id, labels, properties, element_id].
                assert_eq!(
                    fields[1],
                    PackValue::List(vec![PackValue::String("Widget".into())]),
                    "Bolt node labels must be the resolved name ['Widget'], not 'LabelId(N)'"
                );
            }
            other => panic!("RETURN d must pack a Node struct, got {other:?}"),
        }
    }

    /// #1291 — a Bolt RUN whose sort working set exceeds the configured
    /// per-tenant memory cap surfaces the Bolt budget taxonomy
    /// (`BoltError::ResourceExhausted` →
    /// `Neo.TransientError.General.OutOfMemoryError`), NOT an OOM / the
    /// ≈4.29 B-row runaway fallback. RED-on-revert: removing the
    /// `apply_memory_cap` call in `run` (or the `QueryEngine` budget
    /// threading) lets the query succeed → `expect_err` fails.
    #[test]
    fn bolt_run_over_budget_query_surfaces_resource_exhausted() {
        let h = StorageBoltHandler::new(fixture()).with_per_tenant_memory_cap(64 * 1024);
        let err = h
            .run(
                &power_session(),
                "UNWIND range(1, 20000) AS x RETURN x ORDER BY x",
                &std::collections::BTreeMap::new(),
            )
            .expect_err("over-budget RUN must trip the per-tenant byte cap");
        assert!(
            matches!(err, BoltError::ResourceExhausted(_)),
            "expected BoltError::ResourceExhausted; got {err:?}"
        );
    }

    /// #1291 — under-cap RUN succeeds with the budget attached; the
    /// uncapped default handler stays opt-in (admits the same heavy
    /// query — the pre-#1291 posture the embedded path keeps).
    #[test]
    fn bolt_run_under_budget_query_succeeds_and_uncapped_stays_opt_in() {
        let capped = StorageBoltHandler::new(fixture()).with_per_tenant_memory_cap(64 * 1024);
        let out = capped
            .run(
                &power_session(),
                "UNWIND range(1, 10) AS x RETURN x ORDER BY x",
                &std::collections::BTreeMap::new(),
            )
            .expect("under-budget RUN succeeds");
        assert_eq!(out.records.len(), 10);

        let uncapped = StorageBoltHandler::new(fixture());
        let out = uncapped
            .run(
                &power_session(),
                "UNWIND range(1, 20000) AS x RETURN x ORDER BY x",
                &std::collections::BTreeMap::new(),
            )
            .expect("uncapped handler admits the heavy query (opt-in posture)");
        assert_eq!(out.records.len(), 20000);
    }

    #[test]
    fn oauth_enforced_handler_rejects_basic_scheme() {
        // W19γ ADR-049 — when OAuth is wired, `basic` is REJECTED
        // even with a non-empty principal (the legacy handler accepts
        // it; the OAuth gate must override).
        use crate::auth::oauth_pkce::{JsonWebKey, JsonWebKeySet, OAuthConfig};
        use jsonwebtoken::{Algorithm, DecodingKey};

        let jwks = JsonWebKeySet::new(vec![JsonWebKey {
            kid: "k1".into(),
            algorithm: Algorithm::RS256,
            decoding_key: DecodingKey::from_secret(b"x"),
        }])
        .unwrap();
        let cfg = Arc::new(OAuthConfig::new(
            "https://issuer.example/".into(),
            vec!["arcgraph-bolt".into()],
            jwks,
        ));
        let h = StorageBoltHandler::new(fixture()).with_oauth(cfg);
        assert!(h.oauth_enforced());
        let err = h
            .authenticate(Some("basic"), Some("alice"), Some("pw"))
            .expect_err("OAuth rejects basic");
        assert!(matches!(err, BoltError::Unauthorized(_)));
        assert!(format!("{err}").contains("requires `bearer`"));
    }

    #[test]
    fn oauth_enforced_handler_rejects_none_scheme() {
        use crate::auth::oauth_pkce::{JsonWebKey, JsonWebKeySet, OAuthConfig};
        use jsonwebtoken::{Algorithm, DecodingKey};

        let jwks = JsonWebKeySet::new(vec![JsonWebKey {
            kid: "k1".into(),
            algorithm: Algorithm::RS256,
            decoding_key: DecodingKey::from_secret(b"x"),
        }])
        .unwrap();
        let cfg = Arc::new(OAuthConfig::new(
            "https://issuer.example/".into(),
            vec!["arcgraph-bolt".into()],
            jwks,
        ));
        let h = StorageBoltHandler::new(fixture()).with_oauth(cfg);
        let err = h
            .authenticate(Some("none"), None, None)
            .expect_err("OAuth rejects none");
        assert!(matches!(err, BoltError::Unauthorized(_)));
    }

    #[test]
    fn oauth_enforced_handler_rejects_garbage_bearer() {
        // A malformed JWT (truncated) reaches the OAuth verifier and
        // is rejected — proves the OAuth path is wired through.
        use crate::auth::oauth_pkce::{JsonWebKey, JsonWebKeySet, OAuthConfig};
        use jsonwebtoken::{Algorithm, DecodingKey};

        let jwks = JsonWebKeySet::new(vec![JsonWebKey {
            kid: "k1".into(),
            algorithm: Algorithm::RS256,
            decoding_key: DecodingKey::from_secret(b"x"),
        }])
        .unwrap();
        let cfg = Arc::new(OAuthConfig::new(
            "https://issuer.example/".into(),
            vec!["arcgraph-bolt".into()],
            jwks,
        ));
        let h = StorageBoltHandler::new(fixture()).with_oauth(cfg);
        let err = h
            .authenticate(Some("bearer"), None, Some("not-a-jwt"))
            .expect_err("OAuth rejects garbage bearer");
        assert!(matches!(err, BoltError::Unauthorized(_)));
    }

    #[test]
    fn run_on_empty_substrate_returns_empty_rows_for_label_free_query() {
        // A label-free query routes through the executor against an
        // empty substrate; the executor surfaces zero rows with a
        // SUCCESS RunOutcome (NOT a FAILURE). Label-anchored queries
        // require the catalog to know the label name; the per-call
        // catalog is seeded from the catalog-stats snapshot, which is
        // empty for a fresh tenant — so we use a label-free MATCH
        // here to exercise the executor path without dragging the
        // catalog-seed contract into the test.
        let h = StorageBoltHandler::new(fixture());
        let outcome = h
            .run(&power_session(), "MATCH (n) RETURN n", &BTreeMap::new())
            .expect("run");
        assert!(outcome.records.is_empty());
    }

    #[test]
    fn run_returns_syntax_error_for_malformed_cypher() {
        let h = StorageBoltHandler::new(fixture());
        let err = h
            .run(
                &power_session(),
                "THIS IS NOT VALID CYPHER xx yy zz",
                &BTreeMap::new(),
            )
            .expect_err("malformed");
        assert!(
            matches!(err, BoltError::Syntax(_)),
            "expected Syntax, got {err:?}"
        );
    }

    /// #907 — the auto-commit RUN error boundary maps a TYPED MVCC
    /// `SubstrateAccessError::Conflict` to the retriable
    /// `BoltError::TransientConflict` (carrying the contention target),
    /// while a GENUINE substrate I/O fault still maps to the fatal
    /// `BoltError::Internal` (→ `Neo.DatabaseError`) — no over-broadening.
    #[test]
    fn translate_explain_conflict_to_transient_but_io_stays_internal() {
        use arcgraph_query::ExplainError;
        use arcgraph_query::executor::SubstrateAccessError;

        let mapped =
            translate_explain_to_bolt(ExplainError::Substrate(SubstrateAccessError::Conflict {
                target: "key:42".into(),
            }));
        assert!(
            matches!(&mapped, BoltError::TransientConflict { target } if target == "key:42"),
            "conflict must map to TransientConflict carrying the target, got {mapped:?}"
        );
        assert_eq!(
            mapped.neo4j_code(),
            "Neo.TransientError.Transaction.DeadlockDetected"
        );

        // No over-broadening: a genuine substrate I/O fault stays Internal.
        let io = translate_explain_to_bolt(ExplainError::Substrate(SubstrateAccessError::Io(
            "disk fault".into(),
        )));
        assert!(
            matches!(io, BoltError::Internal(_)),
            "genuine substrate I/O must stay Internal (→ DatabaseError), got {io:?}"
        );
        assert_eq!(io.neo4j_code(), "Neo.DatabaseError.General.UnknownError");
    }

    #[test]
    fn translate_explain_resource_exhausted_to_bolt_resource_class() {
        use arcgraph_query::ExplainError;
        use arcgraph_query::semantic::error::ArcQLError;

        let mapped =
            translate_explain_to_bolt(ExplainError::ArcQL(ArcQLError::ResourceExhausted {
                feature: "HashJoinOp build-side".into(),
                requested_bytes: 1,
                cap_bytes: 0,
                projected_bytes: 1,
                span: arcgraph_query::Span::point(0, 0),
            }));

        assert!(
            matches!(mapped, BoltError::ResourceExhausted(_)),
            "ResourceExhausted must not map to Syntax, got {mapped:?}"
        );
        assert_eq!(
            mapped.neo4j_code(),
            "Neo.TransientError.General.OutOfMemoryError"
        );
    }

    /// #907 — a REAL write-write MVCC conflict driven END-TO-END through
    /// the handler's explicit-tx `commit_txn` (the driver
    /// `execute_write` COMMIT path) surfaces as a retriable
    /// `Neo.TransientError.*` with a clean, non-leaking message. The
    /// backend is built inline so the test keeps the shared `TxnManager`
    /// handle to stage a deterministic conflict (mirrors `fixture()`).
    ///
    /// RED-on-revert: revert `commit_txn`'s conflict arm (or
    /// `commit_err_to_substrate`) and the loser's COMMIT maps to the
    /// fatal `Neo.DatabaseError.General.UnknownError` with the leaked
    /// "transaction commit failed: …" message — both asserts fail.
    #[test]
    fn commit_txn_maps_mvcc_conflict_to_retriable_transient_error() {
        use bytes::Bytes;
        let io = Arc::new(InMemoryPageIo::new());
        let pool = BufferPool::new(8, io);
        let mgr = Arc::new(TxnManager::new());
        let catalog = Arc::new(SystemCatalog::new());
        catalog.bootstrap(&pool, &mgr).expect("bootstrap");
        let crud = Arc::new(CrudStore::new());
        let router = Arc::new(MultiTenantRouter::new(catalog, Arc::clone(&crud), None));
        let intern = Arc::new(InternTable::new());
        let h = StorageBoltHandler::new(StorageBackend::new(router, Arc::clone(&mgr), intern));

        const KEY: u64 = 9;
        // Loser begins FIRST → its snapshot precedes the winner's commit.
        let mut loser = mgr.begin_owned(TenantId::DEFAULT);
        loser.txn_mut().write(KEY, Bytes::from_static(b"loser"));
        let mut winner = mgr.begin_owned(TenantId::DEFAULT);
        winner.txn_mut().write(KEY, Bytes::from_static(b"winner"));

        h.commit_txn(Box::new(BoltHeldTxn::new(winner)))
            .expect("winner commits cleanly");
        let err = h
            .commit_txn(Box::new(BoltHeldTxn::new(loser)))
            .expect_err("OCC loser must conflict at COMMIT");

        assert!(
            err.neo4j_code().starts_with("Neo.TransientError."),
            "execute_write COMMIT conflict must be retriable; got {}",
            err.neo4j_code()
        );
        let msg = err.message();
        for leak in [
            "substrate",
            "MVCC commit failed",
            "transaction commit failed",
            "i/o",
        ] {
            assert!(!msg.contains(leak), "must not leak {leak:?}; got {msg:?}");
        }
    }

    /// #928 / #907 — a REAL write-write MVCC conflict driven through the
    /// AUTO-COMMIT Bolt RUN path (no explicit tx, no COMMIT message) surfaces
    /// as retriable `Neo.TransientError.Transaction.DeadlockDetected` with a
    /// clean message. The bounded concurrent workload forces the original
    /// repro shape: `session.run("MATCH … SET …")` races other auto-commit
    /// writers against the same matched records, so the loser is produced by
    /// `StorageBoltHandler::run` → `QueryEngine` → `CrudExecutorSubstrate`
    /// `run_txn`/`stage_or_commit`, not by `commit_txn`.
    ///
    /// RED-on-revert: revert `commit_err_to_substrate` / the #907 conflict
    /// classification and the losing RUN maps to `Neo.DatabaseError.*` and/or
    /// leaks `substrate` / `internal` / `MVCC commit failed`, failing below.
    #[test]
    fn run_auto_commit_mvcc_conflict_maps_to_retriable_transient_error() {
        use std::sync::{Barrier, mpsc};
        use std::thread;

        const NODES: usize = 96;
        const WORKERS: usize = 12;
        const ATTEMPTS: usize = 12;

        let h = Arc::new(StorageBoltHandler::new(fixture()));
        for i in 0..NODES {
            h.run(
                &power_session(),
                &format!("CREATE (n:Hot {{x: {i}}})"),
                &BTreeMap::new(),
            )
            .expect("seed hot node through auto-commit RUN");
        }

        let mut transient = None;
        for attempt in 0..ATTEMPTS {
            let barrier = Arc::new(Barrier::new(WORKERS));
            let (tx, rx) = mpsc::channel();
            let mut joins = Vec::with_capacity(WORKERS);

            for worker in 0..WORKERS {
                let h = Arc::clone(&h);
                let barrier = Arc::clone(&barrier);
                let tx = tx.clone();
                joins.push(thread::spawn(move || {
                    barrier.wait();
                    let result = h.run(
                        &power_session(),
                        &format!("MATCH (n:Hot) SET n.x = {}", attempt * WORKERS + worker),
                        &BTreeMap::new(),
                    );
                    tx.send(result.map(|_| ())).expect("send RUN result");
                }));
            }
            drop(tx);

            for result in rx {
                if let Err(err) = result {
                    if err.neo4j_code() == "Neo.TransientError.Transaction.DeadlockDetected" {
                        transient = Some(err);
                    } else {
                        panic!(
                            "auto-commit RUN conflict must not map to a fatal Bolt error; \
                             got code={} message={:?}",
                            err.neo4j_code(),
                            err.message()
                        );
                    }
                }
            }
            for join in joins {
                join.join().expect("RUN worker thread");
            }
            if transient.is_some() {
                break;
            }
        }

        let err = transient.expect(
            "bounded concurrent auto-commit RUN workload must produce at least one real MVCC conflict",
        );
        assert_eq!(
            err.neo4j_code(),
            "Neo.TransientError.Transaction.DeadlockDetected"
        );
        let msg = err.message();
        for leak in ["substrate", "internal", "MVCC commit failed"] {
            assert!(!msg.contains(leak), "must not leak {leak:?}; got {msg:?}");
        }
    }

    #[test]
    fn cancel_is_no_op_for_unknown_qid() {
        let h = StorageBoltHandler::new(fixture());
        // Per the trait, cancel is best-effort + no-op on miss.
        h.cancel(TenantId::DEFAULT, Some(99));
        h.cancel(TenantId::DEFAULT, None);
    }

    #[test]
    fn cancel_does_not_bridge_qid_into_registry_at_v1_0_alpha() {
        // W18ε sweep — W17 retro §2.5 Pattern E regression pin.
        //
        // BEFORE W18ε, the module-level doc claimed:
        //   "a RESET(qid) frame routes to the corresponding in-flight token"
        // but the body discarded `qid` and the trait-default no-op kicked in.
        //
        // This test pins the v1.0-α behavior so a future doc regression
        // (which would re-introduce the false claim while the body still
        // discards qid) flips this test red.
        //
        // CONTRACT BEING PINNED (matches the rewritten module + field doc):
        //
        //   When a QueryId is registered in the handler's
        //   CancellationRegistry, calling `BoltQueryHandler::cancel(t,
        //   Some(some_i64))` does NOT cancel that QueryId's token —
        //   because v1.0-α has no `i64 → QueryId` value bridge.
        //
        // When M5-12 (issue #354) lights the bridge, this test will need
        // to be updated alongside the bridge — the doc + body + test all
        // move together. That's the desired coupling.
        use arcgraph_query::executor::QueryId;

        let h = StorageBoltHandler::new(fixture());
        // Pre-register a QueryId / token in the handler's registry, as
        // an in-flight RUN would.
        let qid = QueryId::new();
        let token = h.cancellation_registry().register(qid);
        assert!(
            !token.is_cancelled(),
            "freshly-registered token must not be pre-cancelled"
        );

        // Call cancel with an arbitrary Option<i64> qid — the W18ε pin:
        // the body discards qid (no `i64 → QueryId` bridge at v1.0-α).
        h.cancel(TenantId::DEFAULT, Some(12345));

        // The registered token MUST still be untripped.
        assert!(
            !token.is_cancelled(),
            "v1.0-α: cancel(qid) is a no-op; the registered token must NOT be tripped"
        );

        // Same with `None` (the "RESET without qid" path).
        h.cancel(TenantId::DEFAULT, None);
        assert!(
            !token.is_cancelled(),
            "v1.0-α: cancel(None) is also a no-op; the registered token must NOT be tripped"
        );

        // Confirm the engine-side cancellation path still works (the
        // path the module doc cites as the actual mechanism): direct
        // registry.cancel(qid) DOES trip the token. This proves the
        // bridge isn't fundamentally missing — only the
        // `BoltQueryHandler::cancel` entry point doesn't route through.
        assert!(
            h.cancellation_registry().cancel(qid),
            "registry.cancel(qid) returns true when the token was registered"
        );
        assert!(
            token.is_cancelled(),
            "engine-side registry.cancel(qid) DOES trip the token — \
             the doc's 'in-flight RUNs observe cancellation through the engine's \
             own CancellationToken polling' claim is mechanically verified here"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// ADR-197 (#802) — Bolt explicit-transaction FAULT-INJECTION suite.
//
// These drive the REAL `StorageBoltHandler` (real MVCC, real held-tx)
// over an in-process duplex pair through the production
// [`crate::transport::bolt::handle_pair`] loop — the SAME FSM +
// packstream + handler path a `serve --bolt` connection takes. The
// load-bearing discriminator is `rollback_aborts_the_write`: revert the
// abort in `StorageBoltHandler::rollback_txn` (make it a no-op /
// commit) and that test goes RED — proving the explicit transaction is
// REAL, not a no-op that auto-commits anyway
// (`feedback_load_bearing_pr_requires_fault_injection_tests` +
// `feedback_noop_trampoline_anti_pattern`).
// ─────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod txn_fault_injection {
    use super::*;
    use crate::transport::bolt::handshake::{MAGIC_PREAMBLE, SERVER_ACCEPT_V5_0};
    use crate::transport::bolt::message::{ClientMessage, encode_client};
    use crate::transport::bolt::{read_chunked_message, write_chunked_message};
    use arcgraph_storage::InternTable;
    use arcgraph_storage::buffer::BufferPool;
    use arcgraph_storage::catalog::SystemCatalog;
    use arcgraph_storage::crud::CrudStore;
    use arcgraph_storage::io::InMemoryPageIo;
    use arcgraph_storage::router::MultiTenantRouter;
    use arcgraph_storage::transaction::TxnManager;
    use std::collections::BTreeMap;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    fn backend() -> StorageBackend {
        let io = Arc::new(InMemoryPageIo::new());
        let pool = BufferPool::new(8, io);
        let mgr = Arc::new(TxnManager::new());
        let catalog = Arc::new(SystemCatalog::new());
        catalog.bootstrap(&pool, &mgr).expect("bootstrap");
        let crud = Arc::new(CrudStore::new());
        let router = Arc::new(MultiTenantRouter::new(catalog, Arc::clone(&crud), None));
        let intern = Arc::new(InternTable::new());
        StorageBackend::new(router, mgr, intern)
    }

    fn hello() -> ClientMessage {
        ClientMessage::Hello {
            user_agent: Some("fault-inject/1".into()),
            scheme: Some("none".into()),
            principal: None,
            credentials: None,
            routing: None,
            extras: BTreeMap::new(),
        }
    }

    /// Fault-injection tests exercise transaction mechanics, not end-user ACLs.
    /// Make that SYSTEM-TRUSTED disposition explicit without adding a
    /// production auth scheme: only this `#[cfg(test)]` wrapper upgrades HELLO
    /// to a principal-less Power session, then delegates every operation to the
    /// real storage handler.
    struct SystemTrustedTestHandler(Arc<StorageBoltHandler>);

    impl BoltQueryHandler for SystemTrustedTestHandler {
        fn authenticate(
            &self,
            _scheme: Option<&str>,
            _principal: Option<&str>,
            _credentials: Option<&str>,
        ) -> Result<BoltSessionAuth, BoltError> {
            Ok(BoltSessionAuth::new(
                TenantId::DEFAULT,
                None,
                SessionScope::Power,
            ))
        }

        fn run(
            &self,
            session: &BoltSessionAuth,
            cypher: &str,
            parameters: &BTreeMap<String, PackValue>,
        ) -> Result<RunOutcome, BoltError> {
            self.0.run(session, cypher, parameters)
        }

        fn cancel(&self, tenant: TenantId, qid: Option<i64>) {
            self.0.cancel(tenant, qid);
        }

        fn begin_txn(
            &self,
            tenant: TenantId,
            mode: Option<&str>,
            db: Option<&str>,
        ) -> Result<Box<dyn HeldTxnHandle>, BoltError> {
            self.0.begin_txn(tenant, mode, db)
        }

        fn run_in_txn(
            &self,
            session: &BoltSessionAuth,
            cypher: &str,
            parameters: &BTreeMap<String, PackValue>,
            held: Box<dyn HeldTxnHandle>,
        ) -> (Result<RunOutcome, BoltError>, Box<dyn HeldTxnHandle>) {
            self.0.run_in_txn(session, cypher, parameters, held)
        }

        fn commit_txn(&self, held: Box<dyn HeldTxnHandle>) -> Result<String, BoltError> {
            self.0.commit_txn(held)
        }

        fn rollback_txn(&self, held: Box<dyn HeldTxnHandle>) {
            self.0.rollback_txn(held);
        }
    }
    fn run(q: &str) -> ClientMessage {
        ClientMessage::Run {
            query: q.into(),
            parameters: BTreeMap::new(),
            extra: BTreeMap::new(),
        }
    }
    fn pull() -> ClientMessage {
        ClientMessage::Pull { n: -1, qid: None }
    }
    fn begin() -> ClientMessage {
        ClientMessage::Begin {
            extra: BTreeMap::new(),
        }
    }

    /// One decoded server reply frame: its tag + (for SUCCESS/RECORD)
    /// the payload, enough to assert pass/fail + extract a scalar.
    #[derive(Debug)]
    struct Frame {
        tag: u8,
        value: PackValue,
    }

    /// Drive a full client session (handshake + the given messages)
    /// against a fresh `StorageBoltHandler` over a duplex pair; return
    /// every decoded server reply frame in order.
    async fn drive(handler: Arc<StorageBoltHandler>, msgs: Vec<ClientMessage>) -> Vec<Frame> {
        use crate::transport::bolt::handle_pair;
        let (client, server) = tokio::io::duplex(64 * 1024);
        let (cr, mut cw) = tokio::io::split(client);
        let (sr, sw) = tokio::io::split(server);
        let server_handler = Arc::new(SystemTrustedTestHandler(handler));
        let server_task = tokio::spawn(async move { handle_pair(server_handler, sr, sw).await });

        // Handshake.
        let mut req = Vec::new();
        req.extend_from_slice(&MAGIC_PREAMBLE);
        req.extend_from_slice(&[0x00, 0x00, 0x00, 0x05]);
        req.extend_from_slice(&[0; 12]);
        cw.write_all(&req).await.unwrap();
        let mut resp = [0u8; 4];
        let mut cr = cr;
        cr.read_exact(&mut resp).await.unwrap();
        assert_eq!(resp, SERVER_ACCEPT_V5_0, "handshake must accept v5.0");

        let want = msgs
            .iter()
            .filter(|m| !matches!(m, ClientMessage::Goodbye))
            .count();
        for m in &msgs {
            let mut buf = Vec::new();
            encode_client(&mut buf, m).unwrap();
            write_chunked_message(&mut cw, &buf).await.unwrap();
        }
        let mut frames = Vec::new();
        for _ in 0..(want * 64).max(16) {
            match read_chunked_message(&mut cr).await {
                Ok(Some(payload)) => frames.push(decode_server_frame(&payload)),
                Ok(None) | Err(_) => break,
            }
        }
        drop(cw);
        let _ = server_task.await;
        frames
    }

    /// Decode a server reply payload into a [`Frame`] (`tag` + the
    /// first struct field as `value`).
    fn decode_server_frame(payload: &[u8]) -> Frame {
        use crate::transport::bolt::packstream::decode;
        let (val, _) = decode(payload, 0).unwrap();
        match val {
            PackValue::Struct { tag, mut fields } => Frame {
                tag,
                value: fields.drain(..).next().unwrap_or(PackValue::Null),
            },
            other => panic!("server frame not a struct: {other:?}"),
        }
    }

    use crate::transport::bolt::message::{TAG_FAILURE, TAG_RECORD, TAG_SUCCESS};

    /// Run a fresh AUTO-COMMIT `MATCH (n) RETURN count(n)` against the
    /// handler and return the count (the discriminating oracle). Uses a
    /// SEPARATE session so it observes only COMMITTED state.
    async fn committed_count(handler: Arc<StorageBoltHandler>) -> i64 {
        let frames = drive(
            handler,
            vec![
                hello(),
                run("MATCH (n) RETURN count(n) AS c"),
                pull(),
                ClientMessage::Goodbye,
            ],
        )
        .await;
        // Frames: SUCCESS(hello), SUCCESS(run), RECORD([count]), SUCCESS(pull).
        let rec = frames
            .iter()
            .find(|f| f.tag == TAG_RECORD)
            .expect("a RECORD with the count");
        match &rec.value {
            PackValue::List(items) => match items.first() {
                Some(PackValue::Integer(n)) => *n,
                other => panic!("count cell not an integer: {other:?}"),
            },
            other => panic!("RECORD payload not a list: {other:?}"),
        }
    }

    fn record_ints(frames: &[Frame]) -> Vec<i64> {
        frames
            .iter()
            .filter(|f| f.tag == TAG_RECORD)
            .filter_map(|f| match &f.value {
                PackValue::List(items) => match items.first() {
                    Some(PackValue::Integer(n)) => Some(*n),
                    None => None,
                    other => panic!("record cell not an integer: {other:?}"),
                },
                other => panic!("RECORD payload not a list: {other:?}"),
            })
            .collect()
    }

    fn count_tags(frames: &[Frame], tag: u8) -> usize {
        frames.iter().filter(|f| f.tag == tag).count()
    }

    #[tokio::test]
    async fn rollback_aborts_the_write() {
        // THE load-bearing discriminator: BEGIN → RUN CREATE → ROLLBACK
        // → (new auto-commit) count → 0. Revert the abort in
        // StorageBoltHandler::rollback_txn and this goes RED.
        let h = Arc::new(StorageBoltHandler::new(backend()));
        let frames = drive(
            Arc::clone(&h),
            vec![
                hello(),
                begin(),
                run("CREATE (n)"),
                pull(),
                ClientMessage::Rollback,
                ClientMessage::Goodbye,
            ],
        )
        .await;
        // No FAILURE on the happy explicit-tx path.
        assert_eq!(
            count_tags(&frames, TAG_FAILURE),
            0,
            "BEGIN→CREATE→ROLLBACK must not FAILURE; frames={frames:?}"
        );
        // The rolled-back CREATE left NO committed node.
        assert_eq!(
            committed_count(Arc::clone(&h)).await,
            0,
            "ROLLBACK must discard the CREATE — 0 committed nodes (the \
             load-bearing discriminator; revert the abort → this is 1)"
        );
    }

    #[tokio::test]
    async fn commit_persists_the_write() {
        let h = Arc::new(StorageBoltHandler::new(backend()));
        let frames = drive(
            Arc::clone(&h),
            vec![
                hello(),
                begin(),
                run("CREATE (n)"),
                pull(),
                ClientMessage::Commit,
                ClientMessage::Goodbye,
            ],
        )
        .await;
        assert_eq!(count_tags(&frames, TAG_FAILURE), 0, "frames={frames:?}");
        // COMMIT replied SUCCESS with a bookmark.
        let commit_ok = frames.iter().rev().find(|f| f.tag == TAG_SUCCESS);
        assert!(commit_ok.is_some(), "COMMIT must reply SUCCESS");
        assert_eq!(
            committed_count(Arc::clone(&h)).await,
            1,
            "COMMIT must persist the CREATE — 1 committed node"
        );
    }

    #[tokio::test]
    async fn reset_mid_tx_aborts_the_write() {
        // BEGIN → RUN CREATE → RESET → the tx is aborted (count 0) +
        // connection back to Ready.
        let h = Arc::new(StorageBoltHandler::new(backend()));
        let _ = drive(
            Arc::clone(&h),
            vec![
                hello(),
                begin(),
                run("CREATE (n)"),
                pull(),
                ClientMessage::Reset,
                ClientMessage::Goodbye,
            ],
        )
        .await;
        assert_eq!(
            committed_count(Arc::clone(&h)).await,
            0,
            "RESET mid-tx must abort the held tx — 0 committed nodes"
        );
    }

    #[tokio::test]
    async fn connection_drop_mid_tx_aborts_the_write() {
        // BEGIN → RUN CREATE → (drop the connection — no COMMIT) →
        // reconnect → count 0 (no leaked/committed partial tx). The
        // drop is modeled by ending the session without COMMIT/ROLLBACK
        // (GOODBYE closes the connection; the held tx Drops → aborts).
        let h = Arc::new(StorageBoltHandler::new(backend()));
        let _ = drive(
            Arc::clone(&h),
            vec![
                hello(),
                begin(),
                run("CREATE (n)"),
                pull(),
                ClientMessage::Goodbye, // close WITHOUT commit/rollback
            ],
        )
        .await;
        assert_eq!(
            committed_count(Arc::clone(&h)).await,
            0,
            "connection drop mid-tx must abort (Drop-aborts) — 0 committed nodes"
        );
    }

    #[tokio::test]
    async fn multi_statement_txn_commits_both() {
        // BEGIN → CREATE a → CREATE b → COMMIT → count 2 (both in one tx).
        let h = Arc::new(StorageBoltHandler::new(backend()));
        let frames = drive(
            Arc::clone(&h),
            vec![
                hello(),
                begin(),
                run("CREATE (a)"),
                pull(),
                run("CREATE (b)"),
                pull(),
                ClientMessage::Commit,
                ClientMessage::Goodbye,
            ],
        )
        .await;
        assert_eq!(count_tags(&frames, TAG_FAILURE), 0, "frames={frames:?}");
        assert_eq!(
            committed_count(Arc::clone(&h)).await,
            2,
            "multi-statement BEGIN→CREATE a→CREATE b→COMMIT must commit BOTH — 2 nodes"
        );
    }

    #[tokio::test]
    async fn multi_statement_txn_rollback_aborts_both() {
        // BEGIN → CREATE a → CREATE b → ROLLBACK → count 0 (atomic).
        let h = Arc::new(StorageBoltHandler::new(backend()));
        let _ = drive(
            Arc::clone(&h),
            vec![
                hello(),
                begin(),
                run("CREATE (a)"),
                pull(),
                run("CREATE (b)"),
                pull(),
                ClientMessage::Rollback,
                ClientMessage::Goodbye,
            ],
        )
        .await;
        assert_eq!(
            committed_count(Arc::clone(&h)).await,
            0,
            "multi-statement ROLLBACK must abort BOTH writes — 0 committed nodes"
        );
    }

    #[tokio::test]
    async fn commit_without_open_tx_is_failure_not_panic() {
        // COMMIT with no open tx → FAILURE (not a panic); the FSM is in
        // Ready, so COMMIT is a protocol violation.
        let h = Arc::new(StorageBoltHandler::new(backend()));
        let frames = drive(
            Arc::clone(&h),
            vec![hello(), ClientMessage::Commit, ClientMessage::Goodbye],
        )
        .await;
        assert!(
            count_tags(&frames, TAG_FAILURE) >= 1,
            "COMMIT without an open tx must FAILURE; frames={frames:?}"
        );
    }

    #[tokio::test]
    async fn run_in_failed_state_is_ignored() {
        // A FAILURE (bad Cypher) drops the connection to Failed; the
        // next RUN is IGNORED (not processed) until RESET.
        let h = Arc::new(StorageBoltHandler::new(backend()));
        let frames = drive(
            Arc::clone(&h),
            vec![
                hello(),
                run("THIS IS NOT VALID CYPHER zz"), // → FAILURE → Failed
                run("CREATE (n)"),                  // → IGNORED
                ClientMessage::Goodbye,
            ],
        )
        .await;
        assert!(
            count_tags(&frames, TAG_FAILURE) >= 1,
            "bad Cypher must FAILURE; frames={frames:?}"
        );
        // The IGNORED reply tag (0x7E) appears for the post-failure RUN.
        use crate::transport::bolt::message::TAG_IGNORED;
        assert!(
            count_tags(&frames, TAG_IGNORED) >= 1,
            "RUN in Failed state must be IGNORED; frames={frames:?}"
        );
        // And the IGNORED CREATE left nothing committed.
        // (RESET first to clear Failed so the count query can run.)
        assert_eq!(
            committed_count(Arc::clone(&h)).await,
            0,
            "the IGNORED CREATE in Failed state must not commit"
        );
    }

    #[tokio::test]
    async fn begin_while_in_tx_is_failure_not_panic() {
        // BEGIN while a tx is already open → FAILURE (FSM TxReady rejects
        // a second BEGIN), not a panic.
        let h = Arc::new(StorageBoltHandler::new(backend()));
        let frames = drive(
            Arc::clone(&h),
            vec![
                hello(),
                begin(),
                begin(), // second BEGIN → FAILURE
                ClientMessage::Goodbye,
            ],
        )
        .await;
        assert!(
            count_tags(&frames, TAG_FAILURE) >= 1,
            "a second BEGIN while in a tx must FAILURE; frames={frames:?}"
        );
    }

    #[tokio::test]
    async fn count_inside_explicit_txn_sees_uncommitted_writes() {
        // #978: catalog_stats / counts-store is committed-only. A
        // count query executed while a Bolt held transaction is open
        // must use the transaction-visible scan path, so it includes
        // the writes staged by earlier RUNs in the same BEGIN.
        let h = Arc::new(StorageBoltHandler::new(backend()));
        let frames = drive(
            Arc::clone(&h),
            vec![
                hello(),
                begin(),
                run("CREATE (a)-[:T]->(b)"),
                pull(),
                run("MATCH (n) RETURN count(n) AS c"),
                pull(),
                run("MATCH ()-[r]->() RETURN count(r) AS c"),
                pull(),
                ClientMessage::Rollback,
                ClientMessage::Goodbye,
            ],
        )
        .await;
        assert_eq!(count_tags(&frames, TAG_FAILURE), 0, "frames={frames:?}");
        assert_eq!(
            record_ints(&frames),
            vec![2, 1],
            "counts inside the open explicit transaction must include \
             uncommitted node and relationship writes; frames={frames:?}"
        );
        assert_eq!(
            committed_count(Arc::clone(&h)).await,
            0,
            "ROLLBACK must still discard the staged writes"
        );
    }

    // ─────────────────────────────────────────────────────────────────
    // ADR-197 (#802) R1 finding #2 — PRODUCTION-SUBSTRATE fault injection.
    //
    // The suite ABOVE drives `CrudStore::new()` — NO primary index — which
    // is the ONE config that CANNOT exhibit R1 finding #1 (the explicit-tx
    // COMMIT skipping the primary-index dual-write): with no primary index
    // there is nothing to dual-write, so an
    // MVCC-only commit is indistinguishable from a full commit. Those
    // tests are therefore a FALSE-GREEN for the production commit path.
    //
    // These tests wire the production primary B-tree index
    // (`new_with_index`, the dual-write target). They drive the SAME real
    // Bolt FSM (BEGIN/RUN/COMMIT over `handle_pair`) and assert the COMMIT
    // landed in the primary index.
    //
    // LOAD-BEARING (finding #1 regression):
    // `commit_persists_into_primary_index` is RED against the
    // pre-fix `commit_txn` (which called the MVCC-only `OwnedTxn::commit`)
    // and GREEN after the fix routes COMMIT through `crud::commit`. The
    // WAL-append + recovery leg is proven by the storage-crate
    // `bolt_explicit_tx_wal_replay.rs` round-trip (the OwnedTxn →
    // `crud::commit` → WAL → `recover_from_wal` path).
    // ─────────────────────────────────────────────────────────────────
    use arcgraph_storage::page_alloc::PageAllocator;
    use arcgraph_storage::primary_index::{PrimaryIndex, PrimaryKey, RecordKind};

    /// Build a backend wired with the production primary B-tree index
    /// (the dual-write target). Returns the backend plus the primary-index
    /// handle so a test can assert directly on the surface R1 finding #1
    /// skips. The `TxnManager` is shared with the primary
    /// index (its root-pointer sidechannel writes ride the held tx's
    /// CommitBundle on the SAME manager).
    fn backend_production() -> (StorageBackend, Arc<PrimaryIndex>) {
        let io = Arc::new(InMemoryPageIo::new());
        let pool = BufferPool::new(8, io);
        let mgr = Arc::new(TxnManager::new());
        let catalog = Arc::new(SystemCatalog::new());
        catalog.bootstrap(&pool, &mgr).expect("bootstrap");
        let alloc = Arc::new(PageAllocator::new());
        let primary = Arc::new(
            PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&alloc), None).expect("primary index"),
        );
        let crud = Arc::new(CrudStore::new_with_index(
            None,
            Arc::clone(&primary),
            Arc::clone(&alloc),
        ));
        let router = Arc::new(MultiTenantRouter::new(catalog, Arc::clone(&crud), None));
        let intern = Arc::new(InternTable::new());
        (
            StorageBackend::new(router, Arc::clone(&mgr), intern),
            primary,
        )
    }

    /// Count the live `RecordKind::Node` entries the primary index holds
    /// for `tenant` over id range `1..=max_id` — the read-accelerator
    /// surface the explicit-tx COMMIT must dual-write into. With R1
    /// finding #1 unfixed this is ZERO (the install was never drained by
    /// `take_installs`); after the fix it equals the committed CREATE
    /// count. Range-scan (not a fixed id) so it does not pin the exact
    /// allocator-assigned NodeId.
    fn primary_node_count(primary: &PrimaryIndex, tenant: TenantId, max_id: u64) -> usize {
        (1..=max_id)
            .filter(|id| {
                primary
                    .lookup(PrimaryKey::new(tenant, RecordKind::Node, *id))
                    .expect("primary lookup")
                    .is_some()
            })
            .count()
    }

    #[tokio::test]
    async fn commit_persists_into_primary_index() {
        // THE finding-#1 regression test. BEGIN → CREATE (n) → COMMIT,
        // then assert the node landed in the primary index — the surface
        // the MVCC-only commit skipped.
        //
        // RED against the pre-fix `commit_txn` (which called the
        // MVCC-only `OwnedTxn::commit`): `primary_node_count == 0` while
        // the MVCC count is still 1
        // (exactly the silent half-commit). GREEN after COMMIT routes
        // through `crud::commit`.
        let (be, primary) = backend_production();
        let h = Arc::new(StorageBoltHandler::new(be));
        let frames = drive(
            Arc::clone(&h),
            vec![
                hello(),
                begin(),
                run("CREATE (n)"),
                pull(),
                ClientMessage::Commit,
                ClientMessage::Goodbye,
            ],
        )
        .await;
        assert_eq!(count_tags(&frames, TAG_FAILURE), 0, "frames={frames:?}");
        // MVCC-visible too (the false-green oracle the no-index test used —
        // true both before AND after the fix; NOT discriminating).
        assert_eq!(
            committed_count(Arc::clone(&h)).await,
            1,
            "the CREATE is MVCC-visible post-commit"
        );
        // The discriminating oracle is zero with finding #1 unfixed.
        assert_eq!(
            primary_node_count(&primary, TenantId::DEFAULT, 16),
            1,
            "COMMIT must dual-write the CREATE into the primary index \
             (R1 finding #1: the MVCC-only commit leaves this at 0)"
        );
    }

    #[tokio::test]
    async fn multi_statement_commit_persists_both_into_primary_index() {
        // BEGIN → CREATE a → CREATE b → COMMIT: BOTH managed-tx writes
        // dual-write into the primary index through one atomic
        // multi-statement commit.
        let (be, primary) = backend_production();
        let h = Arc::new(StorageBoltHandler::new(be));
        let frames = drive(
            Arc::clone(&h),
            vec![
                hello(),
                begin(),
                run("CREATE (a)"),
                pull(),
                run("CREATE (b)"),
                pull(),
                ClientMessage::Commit,
                ClientMessage::Goodbye,
            ],
        )
        .await;
        assert_eq!(count_tags(&frames, TAG_FAILURE), 0, "frames={frames:?}");
        assert_eq!(committed_count(Arc::clone(&h)).await, 2, "MVCC count");
        assert_eq!(
            primary_node_count(&primary, TenantId::DEFAULT, 16),
            2,
            "both managed-tx CREATEs must dual-write into the primary index"
        );
    }

    #[tokio::test]
    async fn rollback_discards_from_primary_index() {
        // ROLLBACK with the production substrate: the CREATE reaches
        // neither the primary index nor MVCC. The abort path discards the
        // MVCC write; the buffered install is keyed by the aborted tx id
        // and never drained by `crud::commit`.
        let (be, primary) = backend_production();
        let h = Arc::new(StorageBoltHandler::new(be));
        let _ = drive(
            Arc::clone(&h),
            vec![
                hello(),
                begin(),
                run("CREATE (n)"),
                pull(),
                ClientMessage::Rollback,
                ClientMessage::Goodbye,
            ],
        )
        .await;
        assert_eq!(
            committed_count(Arc::clone(&h)).await,
            0,
            "ROLLBACK must discard the CREATE — 0 MVCC-committed nodes"
        );
        assert_eq!(
            primary_node_count(&primary, TenantId::DEFAULT, 16),
            0,
            "ROLLBACK must leave the primary index empty"
        );
    }
}
