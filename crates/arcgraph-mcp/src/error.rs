//! W13δ M5-01 / M5-04 / M5-05 — MCP-level error taxonomy + ExecutionError mapping.
//!
//! Codec-local per `docs/codec-error-translation.md`: callers of the
//! MCP transport pattern-match on [`MCPError`]; the inner
//! [`arcgraph_query`] error types stay private to their crate.
//!
//! # Error-code mapping (per ADR-038 amendment-03 §M5↔M4 contract surface)
//!
//! The JSON-RPC 2.0 spec reserves -32099..-32000 for server-defined
//! errors. The MCP spec (2025-11-25) inherits the JSON-RPC envelope.
//! ArcGraph maps the executor-side variants as follows:
//!
//! | ExecutionError variant                              | MCP code  | message               |
//! |-----------------------------------------------------|-----------|-----------------------|
//! | `Cancelled`                                         | `-32001`  | "request cancelled"   |
//! | `Substrate(SubstrateAccessError::TenantUnknown(_))` | `-32003`  | "tenant unknown"      |
//! | `Substrate(SubstrateAccessError::IndexUnavailable)` | `-32004`  | "index unavailable"   |
//! | `Substrate(SubstrateAccessError::Io(_))`            | `-32006`  | "substrate I/O"       |
//! | `Plan(ArcQLError)` / `NotImplemented`               | `-32005`  | rendered via `data`   |
//! | `Eval(_)`                                           | `-32006`  | "execution eval"      |
//!
//! Plus the protocol-layer codes shared with JSON-RPC:
//!
//! | MCPError variant                  | MCP code  |
//! |-----------------------------------|-----------|
//! | `ParseError`                      | `-32700`  |
//! | `InvalidRequest`                  | `-32600`  |
//! | `MethodNotFound`                  | `-32601`  |
//! | `InvalidParams`                   | `-32602`  |
//! | `InternalError`                   | `-32603`  |
//! | `Unauthorized`                    | `-32002`  |
//! | `RateLimited`                     | `-32007`  |
//!
//! `Unauthorized` is the cross-tenant rejection surface called out in
//! the W13δ spawn prompt's M5-05 acceptance ("cross-tenant access
//! rejected with `MCPError::Unauthorized`"). Its -32002 slot does not
//! collide with any of the executor codes.
//!
//! # Why a top-level enum (not a re-export of `ExecutionError`)
//!
//! `ExecutionError` lives in `arcgraph-query`; the MCP layer adds
//! protocol-level variants (`ParseError` from the JSON-RPC framer,
//! `MethodNotFound` from the dispatcher, `Unauthorized` from the tenant-
//! gate). Defining a thin codec-local enum keeps the mapping
//! deterministic + exhaustive at the boundary, exactly per
//! `docs/codec-error-translation.md` discipline.

use arcgraph_query::ExplainError;
use arcgraph_query::executor::{ExecutionError, SubstrateAccessError};
use arcgraph_query::semantic::error::ArcQLError;
use serde_json::Value as JsonValue;
use thiserror::Error;

use crate::serializers::SerializerError;

// ─────────────────────────────────────────────────────────────────────
// MCP error code constants
// ─────────────────────────────────────────────────────────────────────

/// JSON-RPC 2.0 / MCP error code for a malformed envelope (invalid
/// JSON or framing fault). Per JSON-RPC 2.0 spec §5.1.
pub const CODE_PARSE_ERROR: i32 = -32700;

/// JSON-RPC 2.0 / MCP error code for a structurally-invalid request
/// (well-formed JSON but missing required envelope fields).
pub const CODE_INVALID_REQUEST: i32 = -32600;

/// JSON-RPC 2.0 / MCP error code for an unknown method name.
pub const CODE_METHOD_NOT_FOUND: i32 = -32601;

/// JSON-RPC 2.0 / MCP error code for invalid params (well-formed
/// envelope, but tool-specific argument schema violation).
pub const CODE_INVALID_PARAMS: i32 = -32602;

/// JSON-RPC 2.0 / MCP error code for an unexpected server-side error
/// not otherwise classified.
pub const CODE_INTERNAL_ERROR: i32 = -32603;

/// MCP server-defined: per-query cancellation token tripped (M4-92).
/// Maps from [`ExecutionError::Cancelled`] / [`ExplainError::Cancelled`].
pub const CODE_CANCELLED: i32 = -32001;

/// MCP server-defined: cross-tenant access attempted by a session
/// scoped to a different tenant. Surfaced by the W13δ M5-05 cross-
/// tenant guard before any storage access.
pub const CODE_UNAUTHORIZED: i32 = -32002;

/// MCP server-defined: tenant identity not known to the substrate.
/// Maps from [`SubstrateAccessError::TenantUnknown`].
pub const CODE_TENANT_UNKNOWN: i32 = -32003;

/// MCP server-defined: requested substrate (vector / bm25 / community)
/// is not attached for this tenant. Maps from
/// [`SubstrateAccessError::IndexUnavailable`].
pub const CODE_INDEX_UNAVAILABLE: i32 = -32004;

/// MCP server-defined: ArcQL plan-time error (binding / type-check /
/// cross-substrate / lowering / NotImplemented). Maps from
/// [`ExecutionError::Plan`] / [`ExecutionError::NotImplemented`] /
/// [`ExplainError::ArcQL`] / [`ExplainError::Parse`]. The original
/// [`ArcQLError`] (or [`arcgraph_query::ParseError`]) is rendered as
/// the `data` field on the JSON-RPC error envelope so MCP clients
/// can surface the inner detail.
pub const CODE_QUERY_ERROR: i32 = -32005;

/// MCP server-defined: catch-all for runtime evaluation faults +
/// substrate I/O faults. Distinct from -32004 (index unavailable);
/// substrate-side I/O errors at production wiring time route here.
pub const CODE_EXECUTION_EVAL: i32 = -32006;

/// MCP server-defined: per-tenant rate-limit exhausted (W14γ M5-12).
/// The error envelope's `data` carries the suggested back-off in
/// milliseconds (`{"retry_after_ms": <u64>}`) so the caller can
/// honor the standard MCP retry-after surface.
pub const CODE_RATE_LIMITED: i32 = -32007;

/// MCP server-defined: session scope is insufficient for the requested
/// tool (W16ζ M5-11 per ADR-004 amendment-03 §D-3). Distinct from
/// [`CODE_UNAUTHORIZED`] (-32002, cross-tenant rejection): -32008 means
/// "you're in the right tenant but missing the required scope". The
/// error envelope's `data` carries `{"required_scope": "<slug>"}` so
/// MCP clients route on the slug without parsing the message.
pub const CODE_FORBIDDEN: i32 = -32008;

/// MCP server-defined: a query-path resource budget tripped its ceiling.
/// Surfaced as a structured error rather than risking an out-of-memory
/// failure.
/// The error envelope's `data` carries the human-readable budget detail so MCP
/// clients can route without parsing the message.
pub const CODE_BUDGET_EXCEEDED: i32 = -32009;

// ─────────────────────────────────────────────────────────────────────
// Public MCPError taxonomy
// ─────────────────────────────────────────────────────────────────────

/// Codec-local error type for the MCP transport + tool-dispatch
/// surfaces.
///
/// `#[non_exhaustive]` permits adding a new MCP-level variant in a future slice (M5-02
/// streamable HTTP transport, M5-03 OAuth, M5-13 Bolt) MUST not
/// regress source-compat for downstream pattern-matchers; the
/// `_ => …` catch-all in renderers stays valid.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum MCPError {
    /// JSON-RPC envelope was unparseable (malformed JSON, illegal
    /// `Content-Length` header, framing layer fault). Maps to MCP
    /// error code [`CODE_PARSE_ERROR`].
    #[error("parse error: {0}")]
    ParseError(String),

    /// JSON-RPC envelope parsed but missing required fields (`jsonrpc`,
    /// `method`, `id`) or carrying an unsupported `jsonrpc` version.
    /// Maps to MCP error code [`CODE_INVALID_REQUEST`].
    #[error("invalid request: {0}")]
    InvalidRequest(String),

    /// Caller invoked a tool name not in the active catalog (or a
    /// recognized-but-unwired Tier-2/v1.1 slot). Maps to MCP error code
    /// [`CODE_METHOD_NOT_FOUND`]. The inner [`String`] is the submitted
    /// method slug; [`Self::data`] renders it into a structured #901
    /// recovery payload — `{method, did_you_mean?, available_methods}` —
    /// so an agent can self-correct a near-miss without `tools/list`
    /// (still unavailable per #846).
    #[error("method not found: {0}")]
    MethodNotFound(String),

    /// Caller invoked a recognized catalog method that this dispatcher
    /// instance did not wire. Kept at JSON-RPC -32601 so method absence
    /// remains semantically method-not-found, while `data.reason`
    /// distinguishes deployment wiring from typos.
    #[error("method not found: {tool} (not wired in this deployment)")]
    MethodNotWired {
        /// Recognized method slug.
        tool: String,
        /// Human-readable deployment dependency.
        requires: &'static str,
    },

    /// Tool-specific argument schema violation (e.g., `graph.inspect`
    /// missing required `node_id`). Maps to MCP error code
    /// [`CODE_INVALID_PARAMS`].
    #[error("invalid params: {0}")]
    InvalidParams(String),

    /// Unhandled internal failure (serializer error during response
    /// rendering, transport-level I/O failure on stdout). Maps to MCP
    /// error code [`CODE_INTERNAL_ERROR`].
    #[error("internal error: {0}")]
    InternalError(String),

    /// Per-query cancellation token tripped mid-execution. Maps to MCP
    /// error code [`CODE_CANCELLED`].
    #[error("request cancelled")]
    Cancelled,

    /// Cross-tenant access attempted (the session is scoped to tenant
    /// A but the request body asked for tenant B). Maps to MCP error
    /// code [`CODE_UNAUTHORIZED`].
    #[error("unauthorized: cross-tenant access denied")]
    Unauthorized,

    /// Tenant identity not known to the substrate (e.g., the catalog
    /// binding hasn't been registered for this tenant). Maps to MCP
    /// error code [`CODE_TENANT_UNKNOWN`].
    #[error("tenant unknown: {0}")]
    TenantUnknown(String),

    /// The requested substrate (vector / bm25 / community) is not
    /// attached for this tenant. Maps to MCP error code
    /// [`CODE_INDEX_UNAVAILABLE`].
    #[error("index unavailable: {0}")]
    IndexUnavailable(String),

    /// ArcQL plan-time error (binding / type-check / cross-substrate /
    /// lowering / NotImplemented). Maps to MCP error code
    /// [`CODE_QUERY_ERROR`]. The inner [`String`] is the rendered
    /// detail; structured fields are forwarded via [`MCPError::data`]
    /// when available.
    #[error("query error: {0}")]
    QueryError(String),

    /// Runtime executor evaluation fault OR substrate I/O fault. Maps
    /// to MCP error code [`CODE_EXECUTION_EVAL`].
    #[error("execution eval: {0}")]
    ExecutionEval(String),

    /// Per-tenant rate-limit exhausted (W14γ M5-12). Carries the
    /// suggested back-off in milliseconds; the JSON-RPC error
    /// envelope's `data` slot serializes as
    /// `{"retry_after_ms": <u64>}` so MCP clients can implement the
    /// standard retry-after wait without parsing the message string.
    /// Maps to MCP error code [`CODE_RATE_LIMITED`].
    #[error("rate limited; retry after {retry_after_ms}ms")]
    RateLimited {
        /// Suggested back-off in milliseconds.
        retry_after_ms: u64,
    },

    /// Session scope insufficient for the requested tool (W16ζ M5-11
    /// per ADR-004 amendment-03 §D-3). At v1.0-alpha this is the
    /// canonical rejection surface for `graph.raw_query` invoked from
    /// a non-power session. Distinct from [`Self::Unauthorized`] which
    /// covers cross-tenant rejection.
    ///
    /// Maps to MCP error code [`CODE_FORBIDDEN`]. The JSON-RPC error
    /// envelope's `data` slot serializes as
    /// `{"required_scope": "<slug>"}` per the design-v2 §9.4 scope
    /// nomenclature so MCP clients route on the slug without parsing
    /// the message string.
    #[error("forbidden: required scope {required_scope}")]
    Forbidden {
        /// The design-v2 §9.4 scope slug required for the requested
        /// tool (e.g. `"arcgraph.power"` for `graph.raw_query`).
        required_scope: &'static str,
    },

    /// A query-path resource budget tripped its ceiling. The tool surfaces
    /// this structured error rather than risking an out-of-memory failure.
    ///
    /// Maps to MCP error code [`CODE_BUDGET_EXCEEDED`]. The JSON-RPC error
    /// envelope's `data` slot serializes the budget detail string so MCP
    /// clients route without parsing the message.
    #[error("budget exceeded: {detail}")]
    BudgetExceeded {
        /// Human-readable budget detail (which budget + consumed vs ceiling).
        detail: String,
    },
}

impl MCPError {
    /// JSON-RPC error code for this variant. See the table at the
    /// module-level docs for the mapping.
    #[inline]
    #[must_use]
    pub fn code(&self) -> i32 {
        match self {
            MCPError::ParseError(_) => CODE_PARSE_ERROR,
            MCPError::InvalidRequest(_) => CODE_INVALID_REQUEST,
            MCPError::MethodNotFound(_) | MCPError::MethodNotWired { .. } => CODE_METHOD_NOT_FOUND,
            MCPError::InvalidParams(_) => CODE_INVALID_PARAMS,
            MCPError::InternalError(_) => CODE_INTERNAL_ERROR,
            MCPError::Cancelled => CODE_CANCELLED,
            MCPError::Unauthorized => CODE_UNAUTHORIZED,
            MCPError::TenantUnknown(_) => CODE_TENANT_UNKNOWN,
            MCPError::IndexUnavailable(_) => CODE_INDEX_UNAVAILABLE,
            MCPError::QueryError(_) => CODE_QUERY_ERROR,
            MCPError::ExecutionEval(_) => CODE_EXECUTION_EVAL,
            MCPError::RateLimited { .. } => CODE_RATE_LIMITED,
            MCPError::Forbidden { .. } => CODE_FORBIDDEN,
            MCPError::BudgetExceeded { .. } => CODE_BUDGET_EXCEEDED,
        }
    }

    /// Optional structured `data` payload for the JSON-RPC error
    /// envelope. Returns `Some(json)` when the variant carries
    /// programmatic detail clients should consume (e.g., the rendered
    /// ArcQL diagnostic for a query error); `None` otherwise.
    ///
    /// MCP clients render `data` per the JSON-RPC spec §5.1; missing
    /// `data` fields are valid and idiomatic.
    #[must_use]
    pub fn data(&self) -> Option<JsonValue> {
        match self {
            MCPError::QueryError(detail)
            | MCPError::ExecutionEval(detail)
            | MCPError::TenantUnknown(detail)
            | MCPError::IndexUnavailable(detail)
            | MCPError::ParseError(detail)
            | MCPError::InvalidRequest(detail)
            | MCPError::InvalidParams(detail)
            | MCPError::InternalError(detail) => Some(JsonValue::String(detail.clone())),
            // #901 — a method-not-found is the agent's primary recovery
            // surface (`tools/list` is still unavailable per #846). Instead
            // of the bare method-name string origin/main returned, emit a
            // STRUCTURED payload: the submitted `method`, the ranked
            // `did_you_mean` near-misses, and the full `available_methods`
            // catalog — so the caller can deterministically self-correct a
            // typo / missing `graph.` namespace / camelCase drift rather than
            // retry random variants. The catalog + ranking live in
            // `crate::transport` (the dispatcher's single source of truth).
            MCPError::MethodNotFound(method) => {
                let mut obj = serde_json::Map::new();
                obj.insert("method".into(), JsonValue::String(method.clone()));
                let did_you_mean = crate::transport::nearest_methods(method);
                if !did_you_mean.is_empty() {
                    obj.insert(
                        "did_you_mean".into(),
                        JsonValue::Array(did_you_mean.into_iter().map(JsonValue::String).collect()),
                    );
                }
                obj.insert(
                    "available_methods".into(),
                    JsonValue::Array(
                        crate::transport::KNOWN_METHODS
                            .iter()
                            .map(|&m| JsonValue::String(m.to_string()))
                            .collect(),
                    ),
                );
                Some(JsonValue::Object(obj))
            }
            MCPError::MethodNotWired { tool, requires } => Some(serde_json::json!({
                "reason": "not wired in this deployment",
                "tool": tool,
                "requires": requires,
            })),
            MCPError::Cancelled | MCPError::Unauthorized => None,
            MCPError::RateLimited { retry_after_ms } => Some(serde_json::json!({
                "retry_after_ms": retry_after_ms,
            })),
            MCPError::Forbidden { required_scope } => Some(serde_json::json!({
                "required_scope": required_scope,
            })),
            MCPError::BudgetExceeded { detail } => Some(JsonValue::String(detail.clone())),
        }
    }

    /// Compact human-readable message field for the JSON-RPC error
    /// envelope. Drops the variant payload to satisfy MCP clients that
    /// want a one-line status line; the full rendered detail is
    /// available via [`Self::data`].
    #[must_use]
    pub fn message(&self) -> &'static str {
        match self {
            MCPError::ParseError(_) => "parse error",
            MCPError::InvalidRequest(_) => "invalid request",
            MCPError::MethodNotFound(_) | MCPError::MethodNotWired { .. } => "method not found",
            MCPError::InvalidParams(_) => "invalid params",
            MCPError::InternalError(_) => "internal error",
            MCPError::Cancelled => "request cancelled",
            MCPError::Unauthorized => "unauthorized",
            MCPError::TenantUnknown(_) => "tenant unknown",
            MCPError::IndexUnavailable(_) => "index unavailable",
            MCPError::QueryError(_) => "query error",
            MCPError::ExecutionEval(_) => "execution eval",
            MCPError::RateLimited { .. } => "rate limited",
            MCPError::Forbidden { .. } => "forbidden",
            MCPError::BudgetExceeded { .. } => "budget exceeded",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// RateLimitError → MCPError mapping (W14γ M5-12)
// ─────────────────────────────────────────────────────────────────────

impl From<crate::rate_limit::RateLimitError> for MCPError {
    fn from(e: crate::rate_limit::RateLimitError) -> Self {
        match e {
            crate::rate_limit::RateLimitError::Exceeded { retry_after } => MCPError::RateLimited {
                // Saturating cast: any back-off > u64::MAX ms is
                // pathologically misconfigured; clamp rather than
                // overflow.
                retry_after_ms: u64::try_from(retry_after.as_millis()).unwrap_or(u64::MAX),
            },
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// ExecutionError → MCPError mapping (the spawn prompt's load-bearing table)
// ─────────────────────────────────────────────────────────────────────

impl From<ExecutionError> for MCPError {
    fn from(e: ExecutionError) -> Self {
        match e {
            ExecutionError::Cancelled => MCPError::Cancelled,
            ExecutionError::Substrate(s) => substrate_to_mcp(s),
            // #980 Part 2 — route through `arcql_to_mcp` so a
            // `ResourceExhausted` reaches the dedicated -32009
            // resource class (not the generic -32005 QueryError that
            // reads as a malformed-query fault), matching the
            // `From<ExplainError>` arm.
            ExecutionError::Plan(arcql) => arcql_to_mcp(arcql),
            ExecutionError::Spill(spill) => match spill {
                arcgraph_query::ExecutorSpillError::ResourceExhausted { .. } => {
                    MCPError::BudgetExceeded {
                        detail: spill.to_string(),
                    }
                }
                other => MCPError::ExecutionEval(other.to_string()),
            },
            ExecutionError::NotImplemented {
                feature,
                target_slice,
                section,
            } => MCPError::QueryError(format!(
                "not implemented: {feature} (forward to {target_slice} per {section})"
            )),
            ExecutionError::Eval(detail) => MCPError::ExecutionEval(detail),
            // #797 — a missing `$name` is a CLIENT parameter fault →
            // -32602 invalid params (mirrors the `From<ExplainError>`
            // arm), NOT the -32006 execution-eval bucket.
            ExecutionError::MissingParameter { name } => {
                MCPError::InvalidParams(format!("missing parameter: ${name}"))
            }
        }
    }
}

impl From<ExplainError> for MCPError {
    fn from(e: ExplainError) -> Self {
        match e {
            ExplainError::Parse(p) => MCPError::QueryError(p.to_string()),
            ExplainError::ArcQL(arcql) => arcql_to_mcp(arcql),
            ExplainError::Cancelled => MCPError::Cancelled,
            ExplainError::Substrate(s) => substrate_to_mcp(s),
            ExplainError::ExecutionEval(detail) => MCPError::ExecutionEval(detail),
            // #797 — a missing `$name` is a CLIENT parameter fault →
            // -32602 invalid params (mirrors the #786 DimensionMismatch /
            // #830 IndexAlreadyExists decisions), NOT the -32006
            // execution-eval / InternalError server-fault bucket the
            // catch-all below would render.
            ExplainError::MissingParameter { name } => {
                MCPError::InvalidParams(format!("missing parameter: ${name}"))
            }
            // ExplainError is `#[non_exhaustive]` per its W11Z fix-up
            // MED-2 surface; new variants land additively via the
            // M5↔M4 contract surface (per ADR-038 amendment-03). The
            // wildcard arm preserves source-compat by routing
            // future variants to the catch-all `InternalError`
            // bucket; the spawn prompt's enumerated 5-code mapping
            // still holds for every variant present today.
            other => MCPError::InternalError(format!("unmapped ExplainError variant: {other}")),
        }
    }
}

impl From<SubstrateAccessError> for MCPError {
    fn from(s: SubstrateAccessError) -> Self {
        substrate_to_mcp(s)
    }
}

impl From<SerializerError> for MCPError {
    fn from(s: SerializerError) -> Self {
        MCPError::InternalError(format!("serializer: {s}"))
    }
}

fn substrate_to_mcp(s: SubstrateAccessError) -> MCPError {
    match s {
        SubstrateAccessError::TenantUnknown(t) => MCPError::TenantUnknown(format!("{t:?}")),
        SubstrateAccessError::IndexUnavailable(name) => MCPError::IndexUnavailable(name),
        SubstrateAccessError::Io(detail) => {
            MCPError::ExecutionEval(format!("substrate I/O: {detail}"))
        }
        // #786 — a `query_vec` whose dimension differs from the index's
        // established dimension is a CLIENT parameter error (-32602 invalid
        // params), not a server-side execution fault. Surface the exact dims
        // so the caller sees "dimension N does not match index dimension M"
        // instead of the cryptic -32006 "execution eval" the generic `Io`
        // bucket rendered (the original #786 symptom).
        SubstrateAccessError::DimensionMismatch {
            property,
            query_dim,
            index_dim,
        } => MCPError::InvalidParams(format!(
            "query_vec dimension {query_dim} does not match index dimension \
             {index_dim} for property `{property}`"
        )),
        // #830 / ADR-200 — a `CREATE VECTOR INDEX <name>` (WITHOUT
        // `IF NOT EXISTS`) naming an already-registered index is a CLIENT
        // error (the caller asked to create a duplicate), not a server-side
        // execution fault. Mirror the #786 `DimensionMismatch` decision:
        // surface -32602 invalid params carrying the precise name, NOT the
        // cryptic -32006 "execution eval" the generic `Io` bucket (and the
        // `other` catch-all below) would render. Mirrors Neo4j's
        // `Neo.ClientError.Schema.EquivalentSchemaRuleAlreadyExists` (a
        // `ClientError`). The binding `err @` preserves the variant's exact
        // `#[error(..)]` Display message ("a vector index named `X` already
        // exists") with no drift.
        err @ SubstrateAccessError::IndexAlreadyExists { .. } => {
            MCPError::InvalidParams(err.to_string())
        }
        // #907 — a write-write MVCC conflict is a logical serialization
        // conflict the client should RETRY, not a server fault. Emit a
        // clean "retry the transaction" detail (no "substrate fault: …"
        // catch-all leak) so the MCP `raw_query` boundary is consistent
        // with the Bolt `Neo.TransientError.*` mapping. MCP (JSON-RPC)
        // has no transient-class concept like Bolt's status codes, so the
        // retriable signal here is the clean message; we deliberately do
        // NOT widen the MCP taxonomy in this PR.
        SubstrateAccessError::Conflict { .. } => MCPError::ExecutionEval(
            "the transaction could not complete due to a concurrent write conflict; \
             retry the transaction"
                .into(),
        ),
        // `SubstrateAccessError` is `#[non_exhaustive]`. Future
        // production-substrate variants (e.g., M4-08+
        // CRUD-routing failures) route to the catch-all execution-
        // eval bucket until their canonical MCP code is pinned.
        other => MCPError::ExecutionEval(format!("substrate fault: {other}")),
    }
}

fn arcql_to_mcp(e: ArcQLError) -> MCPError {
    match e {
        // #980 Part 2 — a resource-exhaustion (memory budget / runaway
        // guard tripped) on a VALID query is NOT a query/syntax fault. It
        // is a resource/runtime class: route it to the dedicated
        // [`CODE_BUDGET_EXCEEDED`] (-32009) bucket so MCP clients see a
        // resource signal, never the generic `QueryError` (-32005) that
        // reads like a malformed-query diagnostic (the same mis-class the
        // Bolt surface fixed by mapping ResourceExhausted →
        // `Neo.TransientError.General.OutOfMemoryError`, never
        // `Neo.ClientError.Statement.SyntaxError`).
        e @ ArcQLError::ResourceExhausted { .. } => MCPError::BudgetExceeded {
            detail: e.to_string(),
        },
        other => MCPError::QueryError(other.to_string()),
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use arcgraph_core::TenantId;

    #[test]
    fn cancelled_maps_to_minus_32001() {
        let err: MCPError = ExecutionError::Cancelled.into();
        assert_eq!(err.code(), -32001);
        assert_eq!(err.message(), "request cancelled");
        assert!(err.data().is_none(), "Cancelled has no data payload");
    }

    #[test]
    fn substrate_tenant_unknown_maps_to_minus_32003() {
        let err: MCPError =
            ExecutionError::Substrate(SubstrateAccessError::TenantUnknown(TenantId::new(42)))
                .into();
        assert_eq!(err.code(), -32003);
        assert_eq!(err.message(), "tenant unknown");
        // Structured data: rendered TenantId.
        let data = err.data().expect("data populated for tenant-unknown");
        let s = data.as_str().expect("data is a JSON string");
        assert!(s.contains("42") || s.contains("Tenant"), "rendered: {s}");
    }

    #[test]
    fn substrate_index_unavailable_maps_to_minus_32004() {
        let err: MCPError =
            ExecutionError::Substrate(SubstrateAccessError::IndexUnavailable("vector".into()))
                .into();
        assert_eq!(err.code(), -32004);
        let data = err.data().expect("data carries substrate name");
        assert_eq!(data.as_str().unwrap(), "vector");
    }

    #[test]
    fn substrate_io_routes_to_minus_32006_eval() {
        // Per the spawn prompt: I/O faults at production wiring time
        // route to -32006 (execution eval) — distinct from -32004
        // (index unavailable) which is an availability problem at
        // routing time.
        let err: MCPError = ExecutionError::Substrate(SubstrateAccessError::Io(
            "page cache miss + WAL replay failed".into(),
        ))
        .into();
        assert_eq!(err.code(), -32006);
        let data = err.data().expect("eval carries detail");
        assert!(data.as_str().unwrap().contains("substrate I/O"));
    }

    #[test]
    fn substrate_index_already_exists_maps_to_minus_32602_invalid_params() {
        // R1 #872 F1 — boundary test (per the #786 precedent). The
        // `IndexAlreadyExists` doc claims the MCP boundary renders a
        // PRECISE client-facing error "distinct from `Io`". PROVE it: a
        // `CREATE VECTOR INDEX <name>` duplicate (without IF NOT EXISTS)
        // surfaces as -32602 invalid params via the DEDICATED arm
        // (mirroring #786 DimensionMismatch + Neo4j's
        // EquivalentSchemaRuleAlreadyExists), NOT the -32006 execution-
        // eval catch-all the generic `Io` bucket renders. Closes the
        // no-op-trampoline gap: the doc previously overclaimed a
        // precision the mapping did not provide.
        let err: MCPError = ExecutionError::Substrate(SubstrateAccessError::IndexAlreadyExists {
            name: "cz806vec".into(),
        })
        .into();
        assert_eq!(
            err.code(),
            -32602,
            "dedicated InvalidParams arm, not the -32006 catch-all"
        );
        assert_ne!(
            err.code(),
            -32006,
            "must NOT fall through to the Io / ExecutionEval bucket"
        );
        assert!(
            matches!(err, MCPError::InvalidParams(_)),
            "dedicated InvalidParams variant, not ExecutionEval; got {err:?}"
        );
        assert_eq!(err.message(), "invalid params");
        // The variant's exact Display message is preserved in the detail.
        let data = err
            .data()
            .expect("invalid-params carries the precise detail");
        let s = data.as_str().expect("string payload");
        assert!(
            s.contains("cz806vec") && s.contains("already exists"),
            "preserves the exact `IndexAlreadyExists` Display message; got {s}"
        );
    }

    #[test]
    fn substrate_mvcc_conflict_renders_clean_retry_message_no_internal_leak() {
        // #907 — a write-write MVCC conflict on the MCP `raw_query`
        // boundary surfaces a clean "retry the transaction" detail, NOT
        // the "substrate fault: …" catch-all (which would leak the
        // "substrate" layer term). Pins the MCP-side classification is
        // consistent with the Bolt `Neo.TransientError.*` mapping.
        let err: MCPError = ExecutionError::Substrate(SubstrateAccessError::Conflict {
            target: "key:6404".into(),
        })
        .into();
        let data = err.data().expect("conflict carries a detail");
        let s = data.as_str().expect("string payload");
        assert!(s.contains("retry"), "detail should advise retry; got {s}");
        for leak in [
            "substrate",
            "MVCC commit failed",
            "key:6404",
            "substrate fault",
        ] {
            assert!(
                !s.contains(leak),
                "MCP detail must not leak {leak:?}; got {s}"
            );
        }
    }

    #[test]
    fn plan_arcql_maps_to_minus_32005_with_data() {
        let arcql = ArcQLError::NotImplemented {
            feature: "unsupported clause".into(),
            section: "query language".into(),
            target_version: "v1.1".into(),
            span: arcgraph_query::error::Span::point(1, 1),
        };
        let err: MCPError = ExecutionError::Plan(arcql).into();
        assert_eq!(err.code(), -32005);
        let data = err.data().expect("query-error always carries data");
        let s = data.as_str().expect("string payload");
        assert!(s.contains("unsupported clause"), "rendered: {s}");
    }

    /// #980 Part 2 — a `ResourceExhausted` (memory budget / runaway
    /// guard tripped) on a VALID query must classify as a resource fault
    /// (-32009 BudgetExceeded), NOT the generic -32005 QueryError that
    /// reads like a malformed query. Pins BOTH entry paths:
    /// `ExecutionError::Plan` and `ExplainError::ArcQL`. RED-on-revert:
    /// restore `MCPError::QueryError(arcql.to_string())` and this fails.
    #[test]
    fn resource_exhausted_maps_to_budget_exceeded_not_query_error() {
        let mk = || ArcQLError::ResourceExhausted {
            feature: "HashJoinOp build-side runaway-guard".into(),
            requested_bytes: 0,
            cap_bytes: 4_294_967_296,
            projected_bytes: 4_294_967_296,
            span: arcgraph_query::error::Span::point(0, 0),
        };
        // Path A: executor error.
        let a: MCPError = ExecutionError::Plan(mk()).into();
        assert_eq!(a.code(), -32009, "ExecutionError::Plan path must be -32009");
        assert!(matches!(a, MCPError::BudgetExceeded { .. }));
        // Path B: explain/execute error.
        let b: MCPError = ExplainError::ArcQL(mk()).into();
        assert_eq!(b.code(), -32009, "ExplainError::ArcQL path must be -32009");
        assert!(matches!(b, MCPError::BudgetExceeded { .. }));
        // Neither code is the parse / query-error class.
        assert_ne!(a.code(), CODE_PARSE_ERROR);
        assert_ne!(a.code(), CODE_QUERY_ERROR);
    }

    #[test]
    fn execution_not_implemented_maps_to_minus_32005() {
        // M4-63 forward-deferred operators (Aggregate, Sort, etc.) at
        // the executor level surface as ExecutionError::NotImplemented;
        // they MUST land in the -32005 query-error bucket too — the
        // user-visible message is "your query asked for a feature we
        // don't have", which is a query error in the MCP taxonomy.
        let err: MCPError = ExecutionError::NotImplemented {
            feature: "LogicalPlan::Aggregate".into(),
            target_slice: "M4-63".into(),
            section: "ADR-038 amendment-02 §M4.g".into(),
        }
        .into();
        assert_eq!(err.code(), -32005);
        let data = err.data().expect("not-implemented carries forward-link");
        assert!(
            data.as_str().unwrap().contains("M4-63"),
            "renders forward-link"
        );
    }

    #[test]
    fn eval_maps_to_minus_32006() {
        let err: MCPError = ExecutionError::Eval("division by zero".into()).into();
        assert_eq!(err.code(), -32006);
        assert_eq!(err.data().unwrap().as_str().unwrap(), "division by zero");
    }

    #[test]
    fn explain_error_round_trips_through_substrate_arm() {
        // ExplainError::Substrate is the M5-public surface of
        // SubstrateAccessError — must hit the same code as the bare
        // ExecutionError::Substrate path (otherwise -32003/-32004
        // diverge depending on which crate's error you started from).
        let err: MCPError =
            ExplainError::Substrate(SubstrateAccessError::IndexUnavailable("bm25".into())).into();
        assert_eq!(err.code(), -32004);
    }

    #[test]
    fn explain_error_parse_routes_to_query_error() {
        // ExplainError::Parse is a syntactic ArcQL fault — query-error
        // bucket per the table.
        let parse_err = arcgraph_query::ParseError::Pest {
            message: "expected RETURN".into(),
            span: arcgraph_query::error::Span::point(1, 1),
        };
        let err: MCPError = ExplainError::Parse(parse_err).into();
        assert_eq!(err.code(), -32005);
    }

    #[test]
    fn unauthorized_does_not_collide_with_executor_codes() {
        // -32002 is the W13δ M5-05 cross-tenant rejection slot; must
        // not collide with -32001 (cancelled), -32003 (tenant
        // unknown), -32004 (index unavailable), -32005 (query error),
        // -32006 (execution eval), -32007 (rate limited; W14γ), or
        // -32008 (forbidden; W16ζ).
        let unauth = MCPError::Unauthorized;
        assert_eq!(unauth.code(), -32002);
        for taken in [-32001, -32003, -32004, -32005, -32006, -32007, -32008] {
            assert_ne!(unauth.code(), taken, "code collision with {taken}");
        }
    }

    #[test]
    fn forbidden_maps_to_minus_32008_with_required_scope_data() {
        // W16ζ M5-11 surface (ADR-004 amendment-03 §D-3): scope
        // rejection lands at -32008 with `{"required_scope":"<slug>"}`
        // data so MCP clients route on the slug without parsing the
        // message string.
        let err = MCPError::Forbidden {
            required_scope: "arcgraph.power",
        };
        assert_eq!(err.code(), -32008);
        assert_eq!(err.message(), "forbidden");
        let data = err.data().expect("required_scope data populated");
        assert_eq!(data["required_scope"], "arcgraph.power");
    }

    #[test]
    fn forbidden_code_distinct_from_other_server_defined_codes() {
        // -32008 must not collide with -32001..-32007 (the previously-
        // taken server-defined slots).
        let forbidden = MCPError::Forbidden {
            required_scope: "arcgraph.power",
        };
        assert_eq!(forbidden.code(), -32008);
        for taken in [-32001, -32002, -32003, -32004, -32005, -32006, -32007] {
            assert_ne!(forbidden.code(), taken, "code collision with {taken}");
        }
    }

    #[test]
    fn rate_limited_maps_to_minus_32007_with_retry_after_data() {
        // W14γ M5-12 surface: RateLimitError::Exceeded → MCPError::
        // RateLimited; code -32007; data carries `retry_after_ms`.
        let err: MCPError = crate::rate_limit::RateLimitError::Exceeded {
            retry_after: std::time::Duration::from_millis(250),
        }
        .into();
        assert_eq!(err.code(), -32007);
        assert_eq!(err.message(), "rate limited");
        let data = err.data().expect("retry-after data populated");
        assert_eq!(data["retry_after_ms"], 250);
    }

    #[test]
    fn protocol_codes_are_distinct() {
        // JSON-RPC 2.0 spec reserved range — must keep them apart so
        // clients don't confuse "method not found" with "invalid
        // params" or with our server-defined codes.
        let parse = MCPError::ParseError("x".into());
        let invalid = MCPError::InvalidRequest("x".into());
        let method = MCPError::MethodNotFound("x".into());
        let params = MCPError::InvalidParams("x".into());
        let internal = MCPError::InternalError("x".into());
        let codes = [
            parse.code(),
            invalid.code(),
            method.code(),
            params.code(),
            internal.code(),
        ];
        let mut seen = std::collections::HashSet::new();
        for c in codes {
            assert!(seen.insert(c), "duplicate code {c}");
        }
    }

    #[test]
    fn method_not_found_data_carries_suggestions_and_catalog() {
        // #901 — a near-miss method name surfaces a STRUCTURED recovery
        // payload (NOT the bare method string origin/main returned): the
        // submitted `method`, ranked `did_you_mean` candidates containing the
        // closest real tool, and the full `available_methods` catalog.
        // RED-on-revert: on origin/main `data()` groups MethodNotFound into
        // the plain-string arm, so `data` is `"graph.explor"` and both
        // `did_you_mean` / `available_methods` are absent.
        let err = MCPError::MethodNotFound("graph.explor".into());
        assert_eq!(err.code(), -32601);
        assert_eq!(err.message(), "method not found");
        let data = err
            .data()
            .expect("method-not-found carries structured #901 data");
        assert_eq!(data["method"], "graph.explor");
        let did = data["did_you_mean"]
            .as_array()
            .expect("did_you_mean array present");
        assert!(
            did.iter().any(|v| v == "graph.explore"),
            "closest catalog tool `graph.explore` suggested; got {did:?}"
        );
        let avail = data["available_methods"]
            .as_array()
            .expect("available_methods catalog present");
        for m in ["graph.schema", "graph.search", "graph.raw_query"] {
            assert!(
                avail.iter().any(|v| v == m),
                "{m} in catalog; got {avail:?}"
            );
        }
    }

    #[test]
    fn method_not_found_unmatched_omits_did_you_mean_but_keeps_catalog() {
        // A method with no near-miss still gets the catalog (the fallback
        // recovery path) but NO garbage `did_you_mean` — better than a bare
        // string, and honest about there being no close suggestion.
        let err = MCPError::MethodNotFound("zzzzzzzzzz".into());
        let data = err.data().expect("data present");
        assert!(
            data.get("did_you_mean").is_none(),
            "unrelated input must not produce a garbage suggestion; got {data}"
        );
        assert_eq!(
            data["available_methods"].as_array().unwrap().len(),
            crate::transport::KNOWN_METHODS.len(),
            "complete public catalog still emitted as the fallback recovery path"
        );
    }
}
