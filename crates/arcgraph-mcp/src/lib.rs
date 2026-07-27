//! Protocol surface for ArcGraph.
//!
//! Scope: MCP server (stdio + streamable HTTP + WebSocket), five Tier-1
//! and one Tier-2 tool, OAuth 2.1 + PKCE, rate limiting, TOON / YAML / JSON
//! serializers, Bolt protocol for driver compatibility.
//!
//! Total MCP tool cap: 10. See ADR-004.
//!
//! # W13δ / W13ε / W14α / W14β slices (landed)
//!
//! - **M5-01** stdio MCP transport ([`transport::stdio`]): Content-
//!   Length-framed JSON-RPC 2.0 envelopes; SIGTERM graceful drain;
//!   per-request tracing span with `request_id` + `method` +
//!   `tenant_id` per ADR-038 amendment-03 §TIER-2-c.
//! - **M5-02** TLS hot-reload resolver ([`tls`], W13ε).
//! - **M5-02b** HTTP/TLS transport composition ([`transport::http`],
//!   W14α): hyper 1.x + tokio-rustls + the W13ε `HotReloadResolver`;
//!   header- and SAN-based tenant strategies; 30s deadline timer
//!   (W12γ); SIGTERM `cancel_all` drain; cross-tenant fence at the
//!   transport boundary. Roadmap M5-02 "streamable HTTP transport"
//!   substance lands here; roadmap M5-03 OAuth substance is forward.
//! - **M5-04** `graph.schema` Tier-1 tool ([`tools::schema`]): per-
//!   tenant schema as YAML / TOON / JSON.
//! - **M5-05** `graph.inspect` Tier-1 tool ([`tools::inspect`]): per-
//!   node properties + 1-hop neighborhood; cross-tenant rejection
//!   surfaces [`MCPError::Unauthorized`] (-32002).
//! - **M5-06** `graph.explore` Tier-1 tool ([`tools::explore`]): N-hop
//!   neighborhood graph; depth-cap at [`tools::MAX_EXPLORE_DEPTH`]
//!   (W14β).
//! - **M5-07** `graph.search` Tier-1 tool ([`tools::search`]): RRF-
//!   fused hybrid retrieval over vector + BM25 substrates; substrate-
//!   availability rejection on tenants without an attached index
//!   (W14β).
//!
//! # W14γ slice (this commit)
//!
//! - **M5-08** `graph.ingest` Tier-1 tool ([`tools::ingest`]): first
//!   WRITE-side surface; batch ingest of nodes + relationships,
//!   per-record idempotency on `external_id`, ADR-031 §Decision
//!   group-commit durability, MCP cross-call reads-after-write via
//!   amendment-03 §TIER-1 GAP E rule 1 + LSN monotonicity. v1.0-α
//!   wire shape ratified by ADR-004 amendment-01.
//! - **M5-12** per-tenant token-bucket rate-limit
//!   ([`rate_limit::RateLimiter`]): `(TenantId, OpClass)`-keyed
//!   buckets; design-v2 §9.4 defaults of 100 read req/min + 10
//!   write req/min per tenant (ratified by ADR-004 amendment-02);
//!   per-tenant override via `set_per_tenant`; rejection surfaces
//!   [`MCPError::RateLimited`] (-32007) with `retry_after_ms` data.
//!
//! # Transport / auth / TCK
//!
//! - **M5-03** OAuth 2.1 + PKCE + Bearer-token scope enforcement
//!   (forward — design-v2 §9.4 line 665; the `arcgraph.{read,write,
//!   power,admin}` scope set). Origin-header allowlist defaults +
//!   bind-address hardening land in this slice; the token surface is
//!   the load-bearing forward.
//! - **M5-13** Bolt 5.0 protocol scaffold ([`transport::bolt`]) — W14δ.
//! - **M5-tck** openCypher TCK harness (forward).
//!
//! # W16ζ slice (this commit — M5-11)
//!
//! - **M5-11** `graph.raw_query` Tier-2 power-user MCP tool
//!   ([`tools::raw_query`]) — direct ArcQL execution gated by
//!   [`SessionScope::Power`]. Stub auth at v1.0-α admits any principal;
//!   the scope check is logical-only and binds to the M5-03 OAuth
//!   slice (Bearer-token-derived scope replaces the stub default).
//!   Per ADR-004 amendment-03 §D-1.
//! - **`MCPError::Forbidden`** (-32008) lands as the canonical scope-
//!   rejection surface per ADR-004 amendment-03 §D-3.
//! - **`SessionScope`** enum (`#[non_exhaustive]`, [`scope`]) lands
//!   with two variants: `Read` (default) + `Power`. `Write` + `Admin`
//!   forward-pinned per `feedback_avoid_speculative_scaffolding.md`.
//!
//! The active catalog contains five Tier-1 tools and one Tier-2 tool.
//! ADR-004 enforces a hard maximum of ten.

#![recursion_limit = "256"]

pub mod auth;
pub mod error;
pub mod jsonrpc;
pub mod rate_limit;
mod read_acl;
pub mod scope;
pub mod serializers;
pub mod storage;
pub mod tls;
pub mod tools;
pub mod transport;

pub use error::{
    CODE_CANCELLED, CODE_EXECUTION_EVAL, CODE_FORBIDDEN, CODE_INDEX_UNAVAILABLE,
    CODE_INTERNAL_ERROR, CODE_INVALID_PARAMS, CODE_INVALID_REQUEST, CODE_METHOD_NOT_FOUND,
    CODE_PARSE_ERROR, CODE_QUERY_ERROR, CODE_RATE_LIMITED, CODE_TENANT_UNKNOWN, CODE_UNAUTHORIZED,
    MCPError,
};
pub use jsonrpc::{
    JSONRPC_VERSION, JsonRpcErrorObject, JsonRpcErrorResponse, JsonRpcRequest, JsonRpcResponse,
    MAX_MESSAGE_BYTES, decode_request, read_message, write_message,
};
pub use rate_limit::{
    ClassPolicy, DEFAULT_READ_CAPACITY, DEFAULT_READ_REFILL_PER_SEC, DEFAULT_WRITE_CAPACITY,
    DEFAULT_WRITE_REFILL_PER_SEC, OpClass, RateLimitConfig, RateLimitError, RateLimiter,
    TenantPolicy,
};
pub use scope::SessionScope;
pub use tools::explore::{
    DEFAULT_EXPLORE_DEPTH, DEFAULT_EXPLORE_LIMIT, ExploreRequest, MAX_EXPLORE_DEPTH,
    MAX_EXPLORE_LIMIT, Neighborhood, NeighborhoodEdge, NeighborhoodExplorer, NeighborhoodNode,
    explore_tool,
};
pub use tools::ingest::{
    IngestBatch, IngestError, IngestProvider, IngestRecordOutcome, IngestRequest, IngestSummary,
    NodeIngest, RelIngest, ingest_tool,
};
pub use tools::inspect::{
    InspectRequest, NeighborDirection, NeighborInfo, NodeInspection, NodeInspector, inspect_tool,
};
pub use tools::raw_query::{
    DEFAULT_RAW_QUERY_MAX_ROWS, MAX_RAW_QUERY_BYTES, MAX_RAW_QUERY_MAX_ROWS, RawQueryExecutor,
    RawQueryRequest, RawQueryRows, raw_query_tool,
};
pub use tools::schema::{
    GraphSchema, IndexDescriptor, IndexKind, LabelInfo, PropertyDescriptor, RelTypeInfo,
    SchemaProvider, SchemaRequest, schema_tool,
};
pub use tools::search::{
    AvailableSubstrates, DEFAULT_SEARCH_K, HybridSearcher, MAX_SEARCH_K, SUBSTRATE_SLUG_BM25,
    SUBSTRATE_SLUG_VECTOR, SearchHit, SearchRequest, SearchResult, search_tool, substrate_kinds,
};
pub use tools::{ResponseFormat, render_response};
pub use transport::http::{
    DEFAULT_REQUEST_DEADLINE, ExitReason as HttpExitReason, HEADER_ORIGIN, HEADER_TENANT,
    HttpServerConfig, PATH_HEALTHZ, PATH_MCP, ServeStats as HttpServeStats, TenantStrategy,
    TransportError, client_verifier_for_roots, serve_http,
};
// W16β M5-03 — OAuth 2.1 + PKCE Bearer-token verification (ADR-044 /
// design-v2 §9.4 line 665).
pub use auth::oauth_pkce::{
    Audiences, BEARER_PREFIX, CODE_VERIFIER_DEFAULT_LEN, CODE_VERIFIER_MAX_LEN,
    CODE_VERIFIER_MIN_LEN, CodeVerifier, DEFAULT_CLOCK_SKEW_SECS, HEADER_AUTHORIZATION, JsonWebKey,
    JsonWebKeySet, OAuthConfig, OAuthError, SCOPE_ADMIN, SCOPE_POWER, SCOPE_READ, SCOPE_WRITE,
    TokenClaims, code_challenge_s256, code_verifier_new, code_verifier_with_len, enforce_scope,
    extract_bearer_token, oauth_error_to_www_authenticate, parse_scope_claim, scope_for_method,
    unix_now_secs, validate_code_verifier, verify_bearer_header, verify_bearer_token,
};
// W15γ M6-06 — Prometheus `/metrics` exporter (design-v2 §10.2).
pub use transport::bulkhead::{BulkheadConfig, BulkheadOutcome, DispatchBulkhead, default_permits};
pub use transport::metrics::{
    CONTENT_TYPE_PROMETHEUS_TEXT, ConnectionTransport, DEFAULT_LATENCY_BUCKETS_MS, MetricsError,
    MetricsRegistry, PATH_METRICS, ToolInvocationStatus,
};
pub use transport::stdio::{ExitReason, ServeStats, serve_stdio, shutdown_on_term};
pub use transport::{
    Dispatcher, METHOD_GRAPH_EXPLORE, METHOD_GRAPH_INGEST, METHOD_GRAPH_INSPECT,
    METHOD_GRAPH_RAW_QUERY, METHOD_GRAPH_SCHEMA, METHOD_GRAPH_SEARCH, handle_raw_envelope,
    handle_raw_envelope_with_scope, op_class_for_method,
};

// W14δ M5-13 — Bolt 5.0 protocol scaffold for Neo4j-driver compat.
pub use transport::bolt::{
    BoltError, BoltQueryHandler, BoltServeStats, BoltServerConfig, BoltSessionAuth, BoltVersion,
    ClientMessage, ConnFsm, ConnState, HandlerOutcome, PackValue, RunOutcome, ServerMessage,
    StubBoltHandler, StubFault, handle_pair, serve_bolt_listener,
};
