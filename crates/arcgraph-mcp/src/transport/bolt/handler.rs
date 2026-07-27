//! W14δ M5-13 — `BoltQueryHandler` adapter trait.
//!
//! The Bolt server is generic over a query handler so:
//!
//! 1. Tests can stub a deterministic handler (no QueryEngine wiring
//!    required for the v1.0-α scaffold tests).
//! 2. Production binding at M5-12+ implements the trait on
//!    [`arcgraph_query::QueryEngine`] (where the registry +
//!    cancellation token + plan-cache live) without coupling the
//!    transport surface to query-internal types.
//!
//! The trait shape parallels MCP's [`crate::tools::schema::SchemaProvider`]
//! / [`crate::tools::inspect::NodeInspector`] adapters: the
//! Bolt-side surface owns the trait; concrete impls live wherever
//! the implementer prefers.
//!
//! # Cypher → ArcQL translation
//!
//! Per the spawn prompt's "v1.0-α the openCypher subset already
//! parsed by W13γ M4-83 multi-statement is the supported surface;
//! reject Cypher constructs outside that subset with FAILURE
//! message + clear error text", the canonical handler implementation
//! pipes the Cypher source verbatim through
//! [`arcgraph_query::parse_multi`] and lets the parser report
//! syntax errors. There is no Cypher → ArcQL textual rewrite — the
//! grammars are intentionally identical at this slice.

use std::collections::BTreeMap;

use arcgraph_core::TenantId;
use arcgraph_query::executor::substrate::HeldTxnHandle;

use super::error::BoltError;
use super::packstream::PackValue;
use crate::scope::SessionScope;

/// Result of a successful RUN — column field names + a materialized
/// row stream the server drains via PULL.
///
/// v1.0-α materializes ALL rows up-front (per the spawn prompt's
/// scaffolding boundary); M4-82 streaming-cursor wiring at M5-12+
/// will decouple `RunOutcome::records` from "fully-materialized" so
/// large result sets stream without buffering.
#[derive(Debug, Clone)]
pub struct RunOutcome {
    /// Column names in the projection. Surfaced in the RUN SUCCESS
    /// metadata `fields` slot per Bolt §"Result-stream metadata".
    pub fields: Vec<String>,
    /// Row stream. Each `Vec<PackValue>` is one record; the field-
    /// position MUST match `fields`. Empty vec is a valid empty
    /// result.
    pub records: Vec<Vec<PackValue>>,
    /// Optional run-id (Bolt's `qid`). v1.0-α auto-commit always
    /// returns `None` since there is no concurrent run; v1.1+
    /// explicit-tx slice will populate.
    pub qid: Option<i64>,
}

/// Identity and privilege bound to one authenticated Bolt connection.
///
/// The server stores this whole value after HELLO and threads it into every
/// auto-commit and explicit-transaction RUN. Keeping principal + scope beside
/// the tenant prevents a transport implementation from authenticating an
/// identity and then accidentally executing as tenant-only SYSTEM-TRUSTED.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoltSessionAuth {
    tenant: TenantId,
    principal: Option<String>,
    scope: SessionScope,
}

impl BoltSessionAuth {
    #[must_use]
    pub fn new(tenant: TenantId, principal: Option<String>, scope: SessionScope) -> Self {
        Self {
            tenant,
            principal,
            scope,
        }
    }

    #[must_use]
    pub const fn tenant(&self) -> TenantId {
        self.tenant
    }

    #[must_use]
    pub fn principal(&self) -> Option<&str> {
        self.principal.as_deref()
    }

    #[must_use]
    pub const fn scope(&self) -> SessionScope {
        self.scope
    }
}

/// Adapter trait the Bolt listener binds to. Implementations
/// translate inbound Cypher RUN messages into RunOutcome / FAILURE.
pub trait BoltQueryHandler: Send + Sync + 'static {
    /// Authenticate a HELLO. v1.0-α default: accept any non-empty
    /// `principal` for `scheme="basic"`; accept any HELLO for
    /// `scheme="none"` (or unset). Production binding at M5-12+
    /// validates against the OAuth 2.1 / PKCE catalog (M5-03).
    ///
    /// Returns the full [`BoltSessionAuth`] bound to the connection. The
    /// default tenant derivation remains [`TenantId::DEFAULT`], but the
    /// principal and scope must survive HELLO so RUN can enforce ADR-212.
    fn authenticate(
        &self,
        scheme: Option<&str>,
        principal: Option<&str>,
        credentials: Option<&str>,
    ) -> Result<BoltSessionAuth, BoltError>;

    /// Execute a Cypher RUN. The Cypher source is the verbatim
    /// `query` field from the RUN message; parameters are the
    /// `parameters` map (already normalized to PackValue). The
    /// handler is responsible for parameter-shape validation
    /// (returning a Syntax error for unsupported shapes).
    fn run(
        &self,
        session: &BoltSessionAuth,
        cypher: &str,
        parameters: &BTreeMap<String, PackValue>,
    ) -> Result<RunOutcome, BoltError>;

    /// Cancel an in-flight query (corresponds to a Bolt RESET while
    /// in `Streaming`). Default impl is a no-op — handlers that
    /// don't track per-query cancellation tokens get the simple
    /// "RESET = clear bound run, the next RUN starts fresh"
    /// semantics. Production binding at M5-12+ overrides to fire
    /// the per-query token via
    /// [`arcgraph_query::QueryEngine::cancel`].
    fn cancel(&self, _tenant: TenantId, _qid: Option<i64>) {}

    // ── ADR-197 explicit transactions (BEGIN/COMMIT/ROLLBACK) ──

    /// **ADR-197** — open an explicit transaction (Bolt BEGIN). Returns
    /// an opaque held-transaction handle the server stores on the
    /// connection and threads back into [`Self::run_in_txn`] /
    /// [`Self::commit_txn`] / [`Self::rollback_txn`].
    ///
    /// `mode` is the BEGIN `extra.mode` (`"r"`/`"w"`; honored for
    /// read/write routing where applicable). `_db` is the target db
    /// (v1.0-α single-db; accepted, not acted on).
    ///
    /// Default impl REJECTS — a substrate without explicit-tx support
    /// surfaces `Neo.ClientError.Request.Invalid` so a managed-tx
    /// client gets a clear error rather than silent auto-commit.
    /// [`StorageBoltHandler`](crate::storage::bolt) overrides it.
    fn begin_txn(
        &self,
        _tenant: TenantId,
        _mode: Option<&str>,
        _db: Option<&str>,
    ) -> Result<Box<dyn HeldTxnHandle>, BoltError> {
        Err(BoltError::ProtocolViolation(
            "explicit transactions (BEGIN) not supported by this handler".into(),
        ))
    }

    /// **ADR-197** — execute a RUN within the open explicit transaction.
    /// The statement STAGES into `held` (no commit). Returns
    /// `(result, held)` — the moved-back handle carries the buffered
    /// writes so the connection can run the NEXT statement in the same
    /// transaction or finalize it. The handle is returned on BOTH the
    /// Ok and Err paths so the server can always abort it at
    /// ROLLBACK / RESET / drop.
    ///
    /// Default impl returns an error + the untouched handle.
    fn run_in_txn(
        &self,
        _session: &BoltSessionAuth,
        _cypher: &str,
        _parameters: &BTreeMap<String, PackValue>,
        held: Box<dyn HeldTxnHandle>,
    ) -> (Result<RunOutcome, BoltError>, Box<dyn HeldTxnHandle>) {
        (
            Err(BoltError::ProtocolViolation(
                "explicit transactions not supported by this handler".into(),
            )),
            held,
        )
    }

    /// **ADR-197** — commit the open explicit transaction (Bolt
    /// COMMIT). Consumes `held`. Returns an opaque bookmark token for
    /// the SUCCESS `{bookmark}` reply.
    ///
    /// Default impl rejects.
    fn commit_txn(&self, _held: Box<dyn HeldTxnHandle>) -> Result<String, BoltError> {
        Err(BoltError::ProtocolViolation(
            "explicit transactions not supported by this handler".into(),
        ))
    }

    /// **ADR-197** — abort the open explicit transaction (Bolt
    /// ROLLBACK, RESET-mid-tx, connection-drop). Consumes `held` and
    /// discards all staged writes. Infallible — abort is total (a
    /// failed abort is a server bug, not a client error); the default
    /// impl drops the handle (its `Drop` aborts the underlying tx).
    fn rollback_txn(&self, held: Box<dyn HeldTxnHandle>) {
        // Dropping the handle aborts the underlying transaction via
        // `OwnedTxn`'s Drop (the canonical no-leak path). Concrete
        // impls may override to call `OwnedTxn::abort()` explicitly for
        // a clearer audit trail.
        drop(held);
    }
}

// ─────────────────────────────────────────────────────────────────────
// Stub handler (used by integration tests + the in-tree client)
// ─────────────────────────────────────────────────────────────────────

/// Stub query handler with deterministic responses. Used by the
/// crate's integration tests (where wiring a real `QueryEngine` +
/// `ExecutorSubstrate` would be out-of-scope) and exposed publicly
/// so downstream consumers can validate their Bolt drivers against
/// a known-good fixture.
#[derive(Debug, Clone, Default)]
pub struct StubBoltHandler {
    /// If `Some(error)`, [`Self::run`] always returns this error —
    /// useful for tests asserting the FAILURE path.
    pub forced_error: Option<StubFault>,
    /// Whether [`Self::authenticate`] requires a non-empty principal
    /// (default `true`). Tests toggle off to exercise the "no auth"
    /// scheme.
    pub require_principal: bool,
}

/// Discrete fault classes a [`StubBoltHandler`] can be configured to
/// return. Maps 1:1 to the FAILURE-bearing [`BoltError`] variants the
/// integration tests want to pin.
#[derive(Debug, Clone)]
pub enum StubFault {
    /// Force a `Neo.ClientError.Statement.SyntaxError`.
    Syntax(String),
    /// Force a `Neo.ClientError.Transaction.Terminated`.
    Cancelled,
}

impl StubBoltHandler {
    /// Construct a handler that accepts any non-empty principal +
    /// returns a 2-row, 2-column synthetic dataset for every RUN.
    pub fn accepting() -> Self {
        Self {
            forced_error: None,
            require_principal: true,
        }
    }
}

impl BoltQueryHandler for StubBoltHandler {
    fn authenticate(
        &self,
        scheme: Option<&str>,
        principal: Option<&str>,
        _credentials: Option<&str>,
    ) -> Result<BoltSessionAuth, BoltError> {
        let scheme = scheme.unwrap_or("none");
        match scheme {
            "none" => Ok(BoltSessionAuth::new(
                TenantId::DEFAULT,
                None,
                SessionScope::Read,
            )),
            "basic" | "bearer" => {
                if self.require_principal && principal.unwrap_or("").is_empty() {
                    return Err(BoltError::Unauthorized(
                        "principal required for basic auth".into(),
                    ));
                }
                Ok(BoltSessionAuth::new(
                    TenantId::DEFAULT,
                    principal.filter(|p| !p.is_empty()).map(str::to_owned),
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
        _session: &BoltSessionAuth,
        cypher: &str,
        _parameters: &BTreeMap<String, PackValue>,
    ) -> Result<RunOutcome, BoltError> {
        if let Some(fault) = &self.forced_error {
            return Err(match fault {
                StubFault::Syntax(s) => BoltError::Syntax(s.clone()),
                StubFault::Cancelled => BoltError::Cancelled,
            });
        }
        // Deterministic "echo" behavior:
        //   - "RETURN 1" → 1 row, 1 column "n", value=1.
        //   - "RETURN 1, 2" → 1 row, 2 columns.
        //   - any other query → 1 row, 1 column "value", value=cypher_string.
        if cypher == "RETURN 1" {
            return Ok(RunOutcome {
                fields: vec!["n".into()],
                records: vec![vec![PackValue::Integer(1)]],
                qid: None,
            });
        }
        if cypher.starts_with("RETURN ") && cypher.contains(',') {
            return Ok(RunOutcome {
                fields: vec!["a".into(), "b".into()],
                records: vec![vec![PackValue::Integer(1), PackValue::Integer(2)]],
                qid: None,
            });
        }
        Ok(RunOutcome {
            fields: vec!["value".into()],
            records: vec![vec![PackValue::String(cypher.to_string())]],
            qid: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_authenticate_accepts_basic_with_principal() {
        let h = StubBoltHandler::accepting();
        let t = h
            .authenticate(Some("basic"), Some("alice"), Some("pw"))
            .unwrap();
        assert_eq!(t.tenant(), TenantId::DEFAULT);
        assert_eq!(t.principal(), Some("alice"));
    }

    #[test]
    fn stub_authenticate_rejects_basic_with_empty_principal() {
        let h = StubBoltHandler::accepting();
        let err = h.authenticate(Some("basic"), Some(""), None).unwrap_err();
        assert!(matches!(err, BoltError::Unauthorized(_)));
    }

    #[test]
    fn stub_authenticate_accepts_none_scheme_without_principal() {
        let h = StubBoltHandler::accepting();
        let t = h.authenticate(Some("none"), None, None).unwrap();
        assert_eq!(t.tenant(), TenantId::DEFAULT);
        assert_eq!(t.principal(), None);
    }

    #[test]
    fn stub_run_return_1_emits_single_int_row() {
        let h = StubBoltHandler::accepting();
        let out = h
            .run(
                &BoltSessionAuth::new(TenantId::DEFAULT, Some("alice".into()), SessionScope::Read),
                "RETURN 1",
                &BTreeMap::new(),
            )
            .unwrap();
        assert_eq!(out.fields, vec!["n".to_string()]);
        assert_eq!(out.records.len(), 1);
        assert_eq!(out.records[0], vec![PackValue::Integer(1)]);
    }

    #[test]
    fn stub_run_with_forced_syntax_fault_surfaces_syntax_error() {
        let h = StubBoltHandler {
            forced_error: Some(StubFault::Syntax("bad".into())),
            require_principal: true,
        };
        let err = h
            .run(
                &BoltSessionAuth::new(TenantId::DEFAULT, Some("alice".into()), SessionScope::Read),
                "x",
                &BTreeMap::new(),
            )
            .unwrap_err();
        assert!(matches!(err, BoltError::Syntax(_)));
    }
}
