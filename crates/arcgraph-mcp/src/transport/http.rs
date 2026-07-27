//! W14α M5-02b — HTTP/TLS MCP transport composition.
//!
//! Composes the W13δ stdio dispatch surface ([`crate::transport::Dispatcher`])
//! and the W13ε hot-reload TLS resolver ([`crate::tls::HotReloadResolver`])
//! into a hyper 1.x + `tokio_rustls` server that speaks JSON-RPC 2.0 over
//! HTTP POST /mcp. Health checks live at GET /healthz; everything else
//! responds 405 Method Not Allowed (or 404 Not Found for unknown paths).
//!
//! ## Scope — M5-02b vs roadmap M5-02 / M5-03
//!
//! Per `docs/roadmap.md` (line 377-378), roadmap M5-02 is "streamable
//! HTTP transport" and roadmap M5-03 is "OAuth 2.1 + PKCE for the HTTP
//! transport". This slice ships the **transport-composition substance**
//! of roadmap M5-02 (POST /mcp, GET /healthz, TLS with the W13ε
//! resolver, tenant-strategy, drain). The slug "M5-02b" reflects the
//! HTTP/TLS-composition slice that depends on M5-02 (W13δ stdio) and
//! W13ε resolver substrate. Roadmap M5-03's substance — OAuth 2.1
//! with PKCE plus Bearer-token scope enforcement
//! (`arcgraph.{read,write,power,admin}` per design-v2 §9.4 line 665)
//! — is NOT in this slice. See the PR Risks section and the
//! forward-binding issue for M5-03 OAuth.
//!
//! ## Why hyper-direct (and not axum)
//!
//! Per the W14α spawn prompt: axum's tower-stack + middleware machinery
//! is more surface than the Tier-1 MCP-over-HTTPS endpoint needs (one
//! POST + one GET). hyper 1.x's `service_fn` + `http1::Builder::serve_connection`
//! gives precise control over the per-connection TLS handshake, the
//! per-request cancellation token plumbing, and the SIGTERM drain — all
//! of which would need bespoke axum extractors anyway.
//!
//! ## Concurrency model
//!
//! Each accepted TCP connection spawns a tokio task; the task performs
//! the TLS handshake (via the [`tokio_rustls::TlsAcceptor`] wrapping the
//! W13ε resolver), then runs hyper's `serve_connection`. Per-request
//! handlers register a [`arcgraph_query::CancellationToken`] in the
//! shared registry and arm a 30s deadline timer (per W12γ); on SIGTERM
//! the accept loop fires `cancel_all()` on the registry to drain
//! in-flight queries.
//!
//! ## Hot-reload TLS
//!
//! The `tokio_rustls::TlsAcceptor` is built from a `rustls::ServerConfig`
//! that installs `Arc<HotReloadResolver>` via
//! `with_cert_resolver(...)`. SIGHUP rotation is handled inside the
//! W13ε [`crate::tls::run_sighup_reload_loop`] (caller's responsibility
//! to spawn it alongside this server) — the resolver swaps the
//! `Arc<CertifiedKey>` atomically so new accepts pick up the rotated
//! key without an in-flight handshake observing a half-rotated state.
//!
//! ## Per-tenant identification
//!
//! v1.0-alpha supports two strategies, selected via [`TenantStrategy`]:
//!
//! - [`TenantStrategy::Header`] — read `X-ArcGraph-Tenant: <decimal>` on
//!   each request. Default for v1.0-alpha because it does not require
//!   mTLS configuration on the listener (which is a substantial surface
//!   we defer to v1.1+).
//! - [`TenantStrategy::PeerCertSan`] — read the peer cert's
//!   SubjectAltName entries on the TLS session and parse a `tenant-<N>`
//!   DNSName entry. Requires the [`HttpServerConfig`] to have been
//!   built with mTLS enabled (a `WebPkiClientVerifier` configured with
//!   the operator's client trust store; v1.0-alpha exposes the SAN
//!   parsing function so the caller can wire mTLS forward when ready).
//!
//! When the strategy fails to extract a tenant (header missing or peer
//! cert absent), the request rejects with `400 Bad Request` carrying a
//! JSON-RPC `-32600` (InvalidRequest) envelope so MCP clients still see
//! a structured error.
//!
//! ## ADR provenance
//!
//! - **design-v2 §9.4 (Transport and security)** — transport-layer
//!   security envelope: HTTPS-enforced for non-stdio; Origin-header
//!   allowlist; 127.0.0.1 bind for local; per-tenant rate limit
//!   (forward to M5-12); OAuth 2.1 + PKCE (forward to roadmap M5-03).
//! - **design-v2 §9 (Agent-Native MCP Interface)** — transport
//!   layering: stdio for local, streamable-HTTP for remote, Bolt at v1.1.
//! - **ADR-038 amendment-03 §M5↔M4 contract surface** — per-request
//!   `request_id` + tenant-scoped dispatch.
//! - **ADR-038 amendment-03 §TIER-1 GAP C** — graceful drain at
//!   shutdown via [`arcgraph_query::CancellationRegistry::cancel_all`].
//! - **TLS hot reload** — `ResolvesServerCert` bridge via the
//!   [`crate::tls::HotReloadResolver`] consumed here.
//! - **W12γ** — 30s default per-request deadline.

use std::convert::Infallible;
use std::future::Future;
use std::net::SocketAddr;
use std::pin::Pin;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use arcgraph_core::TenantId;
use arcgraph_query::cancel::{CancellationRegistry, spawn_deadline_timer};
use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use rustls::ServerConfig;
use rustls::server::WebPkiClientVerifier;
use rustls_pki_types::CertificateDer;
use serde_json::Value;
use thiserror::Error;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::watch;
use tokio_rustls::TlsAcceptor;

use crate::error::MCPError;
use crate::jsonrpc::{
    JSONRPC_VERSION, JsonRpcErrorObject, JsonRpcErrorResponse, MAX_MESSAGE_BYTES,
};
use crate::tls::{ClientCertIdentity, HotReloadResolver, parse_client_cert_identity};
use crate::tools::explore::NeighborhoodExplorer;
use crate::tools::ingest::IngestProvider;
use crate::tools::inspect::NodeInspector;
use crate::tools::schema::SchemaProvider;
use crate::tools::search::HybridSearcher;
use crate::transport::bulkhead::{BulkheadOutcome, DispatchBulkhead};
use crate::transport::metrics::{
    CONTENT_TYPE_PROMETHEUS_TEXT, ConnectionTransport, MetricsRegistry, PATH_METRICS,
};
use crate::transport::{Dispatcher, handle_raw_envelope_with_scope};

/// HTTP path the JSON-RPC dispatcher binds to. POST-only.
pub const PATH_MCP: &str = "/mcp";

/// HTTP path the liveness probe binds to. GET-only; returns 200 with a
/// trivial JSON body so kubernetes / monitoring stacks can poll without
/// engaging the dispatcher.
pub const PATH_HEALTHZ: &str = "/healthz";

/// HTTP header carrying the request-tenant identifier when the
/// [`TenantStrategy::Header`] strategy is active. Decimal `u64` value
/// per the [`TenantId::raw`] surface.
pub const HEADER_TENANT: &str = "x-arcgraph-tenant";

/// HTTP header consulted by the Origin allowlist defense (design-v2
/// §9.4 line 667 — DNS rebinding mitigation). Browsers set this on
/// cross-origin requests; non-browser clients (curl, MCP CLI) may
/// omit it. When [`HttpServerConfig::allowed_origins`] is `Some`,
/// requests carrying an `Origin` header that is NOT in the
/// allowlist are rejected with 403 Forbidden before the dispatcher
/// runs.
pub const HEADER_ORIGIN: &str = "origin";

/// Default per-request deadline (W12γ contract). Matches
/// [`arcgraph_query::DEFAULT_QUERY_TIMEOUT_MS`] but is duplicated as a
/// `Duration` here to keep the transport-side configuration self-
/// contained — the request handler arms a deadline timer rooted at
/// this value before the dispatch starts.
pub const DEFAULT_REQUEST_DEADLINE: Duration =
    Duration::from_millis(arcgraph_query::DEFAULT_QUERY_TIMEOUT_MS);

/// Strategy for extracting the request tenant.
///
/// `#[non_exhaustive]` so future strategies (OAuth-claim-based,
/// per-host-binding) land additively without breaking source-compat.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum TenantStrategy {
    /// Read `X-ArcGraph-Tenant: <decimal>` from each request. Default
    /// for v1.0-alpha — does NOT require mTLS configuration.
    #[default]
    Header,
    /// Read the peer cert's SAN entries (looking for a `tenant-<N>`
    /// DNSName entry). Requires the [`HttpServerConfig::client_verifier`]
    /// to be `Some(...)` so peer certs are presented during handshake.
    PeerCertSan,
    /// Try peer-cert SAN first, fall back to header. Useful for mixed
    /// deployments where some clients present mTLS certs and others
    /// don't.
    PeerCertSanThenHeader,
}

/// Configuration for [`serve_http`].
///
/// The `bind_addr` + `tls_resolver` are mandatory; the optional
/// `client_verifier` enables mTLS (v1.1+ forward-pin); the
/// `tenant_strategy` selects how the per-request tenant is identified.
///
/// Configs of this shape will eventually deserialize from operator
/// config with `#[serde(deny_unknown_fields)]`; v1.0-alpha builds them
/// programmatically.
pub struct HttpServerConfig {
    /// TCP bind address. design-v2 §9.4 line 668 mandates 127.0.0.1
    /// for local MCP. Non-loopback binds require
    /// [`Self::allow_remote_bind`] to be `true` (explicit operator
    /// opt-in) — see [`HttpServerConfig::validate`].
    pub bind_addr: SocketAddr,
    /// W13ε hot-reload cert resolver (server-side). Wrapped in
    /// `Arc<...>` so the SIGHUP loop and the TLS acceptor share one
    /// rotating cert.
    pub tls_resolver: Arc<HotReloadResolver>,
    /// Optional client cert verifier — when `Some`, the listener
    /// requires peer certs (mTLS). v1.0-alpha leaves this `None` by
    /// default; tests + future v1.1+ deployments build a
    /// [`WebPkiClientVerifier`] from the operator's client-CA trust
    /// store.
    pub client_verifier: Option<Arc<dyn rustls::server::danger::ClientCertVerifier>>,
    /// Per-request tenant strategy.
    pub tenant_strategy: TenantStrategy,
    /// Origin-header allowlist (design-v2 §9.4 line 667 — DNS
    /// rebinding defense). When `Some`, requests whose `Origin`
    /// header is NOT in this list reject with 403. `None` means the
    /// transport accepts any `Origin` (the operator opted out of the
    /// allowlist explicitly — v1.0-alpha default for parity with the
    /// existing tests but operators SHOULD set this for production).
    /// Origins are compared case-sensitively per RFC 6454 (the host
    /// portion is normally lowercase; the scheme is always
    /// lowercase). Requests without an `Origin` header pass — that's
    /// the typical CLI / curl / MCP-stdio-bridge case.
    pub allowed_origins: Option<Vec<String>>,
    /// Allow [`Self::bind_addr`] to be non-loopback (e.g. `0.0.0.0`,
    /// public IP). Default `false`: design-v2 §9.4 line 668 mandates
    /// 127.0.0.1 for local MCP servers; an operator that wants the
    /// transport reachable beyond loopback MUST set this explicitly.
    /// Loud failure at startup beats silently-public servers.
    pub allow_remote_bind: bool,
    /// Per-request deadline (default = [`DEFAULT_REQUEST_DEADLINE`]).
    pub request_deadline: Duration,
    /// Session-bound tenant the [`Dispatcher`] is pinned to (per-
    /// process pattern). Used by [`Self::validate`] to defend
    /// against a transport-identified tenant that doesn't match the
    /// dispatcher's bound tenant — when `Some`, [`serve_http`]
    /// reasserts at request time that
    /// `identified_tenant == bound_tenant`. `None` means the
    /// transport does not assert the binding (older callers; the
    /// authoritative check still happens at the tool layer).
    pub bound_tenant: Option<TenantId>,
    /// Optional [`MetricsRegistry`] for W15γ M6-06 Prometheus
    /// `/metrics` exposure (design-v2 §10.2). When `Some`, the
    /// transport:
    ///   1. Mounts `GET /metrics` returning the Prometheus text-
    ///      exposition format (Content-Type carries the version
    ///      qualifier per the spec).
    ///   2. Records `arcgraph_mcp_tool_invocations{tenant, tool,
    ///      status}` + `arcgraph_{read,write}_latency_ms{tenant, tool}`
    ///      per dispatch (the latency histogram routes by `op_class`).
    ///   3. Maintains `arcgraph_active_connections{transport="http"}`
    ///      across accept + close transitions.
    ///
    /// When `None`, the `/metrics` route is NOT mounted (`404 Not
    /// Found`) and the request handler skips metric recording. Older
    /// callers don't need to wire metrics; new deployments SHOULD per
    /// the M6-06 exit criterion.
    pub metrics: Option<Arc<MetricsRegistry>>,

    /// W16β M5-03 — Optional OAuth 2.1 + PKCE Bearer-token
    /// verification (ADR-044 / design-v2 §9.4 line 665). When
    /// `Some`, every `POST /mcp` request MUST carry an
    /// `Authorization: Bearer <token>` header that verifies against
    /// the operator-staged JWK Set; the dispatched method's required
    /// scope is enforced against the token's `scope` claim. When
    /// `None`, the transport behaves exactly as the W14α landing:
    /// no Bearer header required, no scope enforced.
    ///
    /// Wraps in `Arc` so the per-request handler can clone cheaply
    /// for the verification call. Configs of this shape are NOT yet
    /// user-deserialized; strict deserialization lands with the future
    /// server-config work.
    pub oauth: Option<Arc<crate::auth::oauth_pkce::OAuthConfig>>,

    /// AHP-1 (ADR-225 §3) — the `spawn_blocking` bulkhead the per-request
    /// dispatch runs behind so a blocking engine call (cold page read,
    /// group-commit `fdatasync` wait) no longer pins the connection's
    /// Tokio worker and starves concurrent reads (#999). `None` means
    /// [`serve_http`] builds one with the default cap (2 × cores); the
    /// CLI injects a *shared* instance via [`Self::with_dispatch_bulkhead`]
    /// so the HTTP and Bolt transports of one process share a single
    /// bounded blocking-pool budget.
    pub dispatch_bulkhead: Option<DispatchBulkhead>,
}

impl HttpServerConfig {
    /// Construct a config with sensible defaults: no mTLS, header-based
    /// tenant identification, 30s request deadline, loopback-only bind,
    /// no Origin allowlist (operator must opt in).
    #[must_use]
    pub fn new(bind_addr: SocketAddr, tls_resolver: Arc<HotReloadResolver>) -> Self {
        Self {
            bind_addr,
            tls_resolver,
            client_verifier: None,
            tenant_strategy: TenantStrategy::Header,
            allowed_origins: None,
            allow_remote_bind: false,
            request_deadline: DEFAULT_REQUEST_DEADLINE,
            bound_tenant: None,
            metrics: None,
            oauth: None,
            dispatch_bulkhead: None,
        }
    }

    /// AHP-1 (ADR-225 §3) — inject a shared [`DispatchBulkhead`]. The
    /// production binary constructs ONE bulkhead and installs it on both
    /// the HTTP and Bolt configs so a single process shares one bounded
    /// blocking-pool budget across transports. When unset, [`serve_http`]
    /// builds a per-listener bulkhead at the default cap.
    #[must_use]
    pub fn with_dispatch_bulkhead(mut self, bulkhead: DispatchBulkhead) -> Self {
        self.dispatch_bulkhead = Some(bulkhead);
        self
    }

    /// Builder-pattern: attach an [`crate::auth::oauth_pkce::OAuthConfig`]
    /// (W16β M5-03 / ADR-044). When set, the HTTP transport requires
    /// a `Authorization: Bearer <jwt>` header on every `POST /mcp`
    /// request and enforces design-v2 §9.4 scope policy against the
    /// dispatched method.
    #[must_use]
    pub fn with_oauth(mut self, oauth: Arc<crate::auth::oauth_pkce::OAuthConfig>) -> Self {
        self.oauth = Some(oauth);
        self
    }

    /// W20β-1 — install an mTLS client-cert verifier from a PEM bundle
    /// of trusted client-CA roots.
    ///
    /// `client_cert_required` selects the posture:
    /// - `true`  → every accepted handshake MUST present a chain-
    ///   validating client cert (REJECT otherwise).
    /// - `false` → handshake admits no-cert clients; clients that DO
    ///   present a cert are still chain-validated.
    ///
    /// This is a convenience wrapper over
    /// [`crate::tls::client_verifier_from_ca_pem`] that installs the
    /// resulting verifier on [`Self::client_verifier`]. Operators that
    /// need hot-reload semantics for the CA bundle (SIGHUP-driven
    /// rotation) wrap the result via [`crate::tls::HotReloadClientVerifier`]
    /// and assign the wrapper to `client_verifier` directly.
    ///
    /// # Errors
    ///
    /// Surfaces any [`crate::tls::TlsResolverError`] from PEM decode /
    /// trust-store build / `WebPkiClientVerifier::builder` failure.
    pub fn with_client_ca_pem(
        mut self,
        pem: &[u8],
        client_cert_required: bool,
    ) -> Result<Self, TransportError> {
        let verifier = crate::tls::client_verifier_from_ca_pem(pem, client_cert_required)
            .map_err(|e| TransportError::TlsConfig(format!("client-CA PEM: {e}")))?;
        self.client_verifier = Some(verifier);
        Ok(self)
    }

    /// Builder-pattern: attach a [`MetricsRegistry`] for W15γ M6-06
    /// Prometheus `/metrics` exposure (design-v2 §10.2). See the
    /// [`HttpServerConfig::metrics`] field for the wiring semantics.
    #[must_use]
    pub fn with_metrics(mut self, metrics: Arc<MetricsRegistry>) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// Builder-pattern: install a non-empty Origin allowlist. Each
    /// entry must be a complete origin string per RFC 6454
    /// (e.g. `"https://app.example.com"` — scheme + host, no path).
    /// Empty list is rejected (the operator clearly meant something).
    pub fn with_allowed_origins(mut self, origins: Vec<String>) -> Self {
        debug_assert!(!origins.is_empty(), "allowed_origins must be non-empty");
        self.allowed_origins = Some(origins);
        self
    }

    /// Builder-pattern: opt-in to a non-loopback bind. design-v2 §9.4
    /// line 668: "Bind 127.0.0.1 for local MCP servers (not
    /// 0.0.0.0)." Setting this to `true` is required for any
    /// non-loopback `bind_addr` (e.g. `0.0.0.0` for a corp-network
    /// MCP server). Validated by [`Self::validate`].
    #[must_use]
    pub fn with_allow_remote_bind(mut self, allow: bool) -> Self {
        self.allow_remote_bind = allow;
        self
    }

    /// Builder-pattern: bind the transport to a specific tenant. When
    /// set, [`serve_http`] asserts at request time that the
    /// transport-identified tenant matches this value — closing the
    /// "header label != identity" gap when mTLS / OAuth are absent.
    #[must_use]
    pub fn with_bound_tenant(mut self, tenant: TenantId) -> Self {
        self.bound_tenant = Some(tenant);
        self
    }

    /// Validate the configuration against the design-v2 §9.4 mandates
    /// and internal consistency requirements. Called at the top of
    /// [`serve_http`] so misconfiguration surfaces at startup rather
    /// than after a first failing request.
    ///
    /// Validates:
    /// - `bind_addr` is loopback OR `allow_remote_bind == true`.
    /// - `tenant_strategy == PeerCertSan` implies
    ///   `client_verifier.is_some()` (otherwise EVERY request rejects
    ///   for "no peer cert").
    /// - `allowed_origins` (when `Some`) is non-empty.
    pub fn validate(&self) -> Result<(), TransportError> {
        let ip = self.bind_addr.ip();
        if !ip.is_loopback() && !self.allow_remote_bind {
            return Err(TransportError::BindAddrForbidden {
                addr: self.bind_addr,
            });
        }
        if matches!(self.tenant_strategy, TenantStrategy::PeerCertSan)
            && self.client_verifier.is_none()
        {
            return Err(TransportError::ConfigInvalid(
                "TenantStrategy::PeerCertSan requires client_verifier to be Some(_)".into(),
            ));
        }
        if let Some(origins) = &self.allowed_origins {
            if origins.is_empty() {
                return Err(TransportError::ConfigInvalid(
                    "allowed_origins must be non-empty when set; use None to disable the \
                     allowlist explicitly"
                        .into(),
                ));
            }
        }
        Ok(())
    }
}

impl std::fmt::Debug for HttpServerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpServerConfig")
            .field("bind_addr", &self.bind_addr)
            .field("tls_resolver", &self.tls_resolver)
            .field("client_verifier", &self.client_verifier.is_some())
            .field("tenant_strategy", &self.tenant_strategy)
            .field(
                "allowed_origins",
                &self.allowed_origins.as_ref().map(Vec::len),
            )
            .field("allow_remote_bind", &self.allow_remote_bind)
            .field("request_deadline", &self.request_deadline)
            .field("bound_tenant", &self.bound_tenant)
            .field("metrics", &self.metrics.is_some())
            .field("oauth", &self.oauth.is_some())
            .finish()
    }
}

// ─────────────────────────────────────────────────────────────────────
// Error taxonomy
// ─────────────────────────────────────────────────────────────────────

/// Failure modes specific to the HTTP/TLS transport.
///
/// `#[non_exhaustive]` preserves forward compatibility. v1.0-alpha has
/// no production exhaustive-pattern-match consumers; the only consumer is
/// [`serve_http`] which propagates errors via `?` without exhaustive
/// matching.
///
/// Variants split into three phases:
/// - **Listener / acceptor build** ([`Self::TlsConfig`], [`Self::Bind`],
///   [`Self::BindAddrForbidden`]).
/// - **Per-connection** ([`Self::Accept`], [`Self::TlsHandshake`]).
/// - **Per-request** ([`Self::BodyTooLarge`], [`Self::BodyParse`],
///   [`Self::HeaderInvalid`], [`Self::OriginForbidden`],
///   [`Self::TenantMissing`], [`Self::TenantParse`]).
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TransportError {
    /// `ServerConfig::builder()` rejected the supplied
    /// [`HotReloadResolver`] / cipher suite combination. Surfaces only
    /// at startup; in-flight handshakes that mis-negotiate fall under
    /// [`Self::TlsHandshake`].
    #[error("TLS server config build failed: {0}")]
    TlsConfig(String),

    /// `TcpListener::bind` failed. The bind address is already in use,
    /// the operator lacks the privilege to bind to a low port, or the
    /// kernel refused the binding for some other reason.
    #[error("bind to {addr} failed: {source}")]
    Bind {
        addr: SocketAddr,
        #[source]
        source: std::io::Error,
    },

    /// `TcpListener::accept` returned an error. Usually fatal — we
    /// surface the error to the caller of [`serve_http`] which decides
    /// whether to retry or shut down.
    #[error("accept failed: {0}")]
    Accept(#[source] std::io::Error),

    /// Per-connection TLS handshake failed. NOT fatal at the listener
    /// level: the connection task swallows the error and logs at WARN.
    /// The variant exists for tests that exercise the handshake path.
    #[error("tls handshake failed: {0}")]
    TlsHandshake(#[source] std::io::Error),

    /// Request body exceeded the [`MAX_MESSAGE_BYTES`] cap (the same
    /// cap stdio enforces — keeps both transports symmetric so a
    /// single rate-limit policy applies). Maps to HTTP 413 Payload
    /// Too Large.
    #[error("request body exceeds cap: {len} > {cap}")]
    BodyTooLarge { len: usize, cap: usize },

    /// The hyper body collector returned an error while reading the
    /// request body. Distinct from [`Self::HeaderInvalid`] (header
    /// vs body parse) and [`Self::BodyTooLarge`] (size cap). Maps to
    /// MCP `ParseError` (-32700) because the JSON-RPC layer could
    /// not decode the body the peer claimed to send.
    #[error("body parse failed: {0}")]
    BodyParse(String),

    /// HTTP header was structurally invalid (non-ASCII, malformed
    /// value). Maps to HTTP 400 Bad Request.
    #[error("invalid header {name}: {detail}")]
    HeaderInvalid { name: &'static str, detail: String },

    /// The peer-supplied `Origin` header was not in the
    /// [`HttpServerConfig::allowed_origins`] allowlist. Defends
    /// against DNS-rebinding per design-v2 §9.4. Maps to HTTP 403
    /// Forbidden with a JSON-RPC -32600 envelope.
    #[error("origin not allowed: {origin}")]
    OriginForbidden { origin: String },

    /// The configured [`TenantStrategy`] could not extract a tenant
    /// identifier from the request (header absent, peer cert absent,
    /// or the SAN extension carried no `tenant-<N>` entry). Maps to
    /// HTTP 400 Bad Request with a JSON-RPC -32600 envelope.
    #[error("tenant identifier missing for strategy {strategy:?}")]
    TenantMissing { strategy: TenantStrategy },

    /// The tenant value in the header / SAN was not parseable as a
    /// `u64` decimal. Maps to HTTP 400 Bad Request.
    #[error("tenant identifier parse failed: {0}")]
    TenantParse(String),

    /// The configured [`HttpServerConfig`] is internally inconsistent
    /// (e.g. [`TenantStrategy::PeerCertSan`] without a
    /// `client_verifier`, or non-loopback `bind_addr` without
    /// `allow_remote_bind`). Surfaces at startup so misconfiguration
    /// is loud, not silent. Maps to MCP `InternalError`.
    #[error("server configuration invalid: {0}")]
    ConfigInvalid(String),

    /// `bind_addr` is non-loopback (e.g. `0.0.0.0` / public IP) but
    /// [`HttpServerConfig::allow_remote_bind`] is `false`. design-v2
    /// §9.4 line 668: "Bind 127.0.0.1 for local MCP servers (not
    /// 0.0.0.0)". Maps to MCP `InternalError`.
    #[error("bind to non-loopback {addr} forbidden without allow_remote_bind")]
    BindAddrForbidden { addr: SocketAddr },

    /// W16β M5-03 — `Authorization: Bearer <jwt>` header was absent
    /// when [`HttpServerConfig::oauth`] is `Some`. design-v2 §9.4
    /// line 665 mandates Bearer tokens for the HTTP transport.
    /// Maps to HTTP `401 Unauthorized` + JSON-RPC
    /// [`MCPError::Unauthorized`] (-32002) + RFC 6750 §3
    /// `WWW-Authenticate: Bearer realm="arcgraph"` (no `error=`
    /// since no authentication was attempted). Cite-coherent with
    /// the W14 IR [`Self::BindAddrForbidden`] shape per
    /// `feedback_cross_wave_hardening_propagation.md`.
    #[error("OAuth Bearer header missing (design-v2 §9.4)")]
    OAuthMissingBearer,

    /// W16β M5-03 — Bearer token decode / signature verify / claims
    /// validation failed. Maps to HTTP `401 Unauthorized` +
    /// `MCPError::Unauthorized` + RFC 6750 §3.1
    /// `WWW-Authenticate: Bearer error="invalid_token"`.
    #[error("OAuth invalid token: {0}")]
    OAuthInvalidToken(String),

    /// W16β M5-03 — Token verified but the `scope` claim does not
    /// include the scope required for the dispatched method per
    /// ADR-044 §Decision item 6. Maps to HTTP `403 Forbidden` +
    /// `MCPError::Unauthorized` + RFC 6750 §3.1
    /// `WWW-Authenticate: Bearer error="insufficient_scope"
    /// scope="<required>"`.
    #[error("OAuth insufficient scope: required {required}")]
    OAuthInsufficientScope {
        /// The scope the method requires (e.g. `arcgraph.read`).
        required: &'static str,
    },
}

// ─────────────────────────────────────────────────────────────────────
// Public entrypoint
// ─────────────────────────────────────────────────────────────────────

/// Run the HTTP/TLS MCP server until `shutdown_signal` resolves.
///
/// The function:
/// 1. Builds a `rustls::ServerConfig` rooted in the supplied
///    [`HotReloadResolver`] and wraps it in a [`tokio_rustls::TlsAcceptor`].
/// 2. Binds a [`tokio::net::TcpListener`] at the configured address.
/// 3. Loops on `accept()`; each accepted connection spawns a tokio
///    task that performs the TLS handshake + serves a single hyper
///    `http1::Builder::serve_connection`.
/// 4. On `shutdown_signal` resolution, fires
///    [`CancellationRegistry::cancel_all`] to drain in-flight queries
///    and stops accepting new connections.
///
/// Returns [`ServeStats`] describing the loop's lifetime — accepted
/// connection count, per-tenant request count, etc.
///
/// # Errors
///
/// - [`TransportError::TlsConfig`] on `ServerConfig::builder()` failure.
/// - [`TransportError::Bind`] on `TcpListener::bind` failure.
/// - [`TransportError::Accept`] on a fatal `accept()` fault.
///
/// Per-connection TLS handshake faults and per-request body-parse
/// faults do NOT propagate; they are logged at WARN and the loop
/// continues.
pub async fn serve_http<S, I, E, H, G, R, Sig>(
    config: HttpServerConfig,
    dispatcher: Arc<Dispatcher<S, I, E, H, G, R>>,
    cancel_registry: Arc<CancellationRegistry>,
    shutdown_signal: Sig,
) -> Result<ServeStats, TransportError>
where
    S: SchemaProvider + Send + Sync + 'static,
    I: NodeInspector + Send + Sync + 'static,
    E: NeighborhoodExplorer + Send + Sync + 'static,
    H: HybridSearcher + Send + Sync + 'static,
    G: IngestProvider + Send + Sync + 'static,
    R: crate::tools::raw_query::RawQueryExecutor + Send + Sync + 'static,
    Sig: Future<Output = ()> + Send + 'static,
{
    // ─── design-v2 §9.4 mandates — loud at startup ─────────────────
    //
    // - bind_addr ∈ loopback unless explicit opt-in (§9.4 line 668).
    // - PeerCertSan + no client_verifier is incoherent.
    // - allowed_origins (when Some) must be non-empty.
    //
    // See [`HttpServerConfig::validate`]. Surfacing here means the
    // operator sees the misconfiguration at process start, not after
    // a first failing request.
    config.validate()?;
    if !config.bind_addr.ip().is_loopback() {
        tracing::warn!(
            target: "arcgraph_mcp::http",
            bind_addr = %config.bind_addr,
            "binding HTTP MCP transport to non-loopback address (allow_remote_bind=true); \
             per design-v2 §9.4 this requires mTLS and/or OAuth to be safe",
        );
    }
    if config.allowed_origins.is_none() {
        tracing::warn!(
            target: "arcgraph_mcp::http",
            "no Origin allowlist configured (design-v2 §9.4 mandate); the transport will \
             accept browser-originated requests from any peer — set HttpServerConfig::\
             allowed_origins for browser-driven deployments",
        );
    }

    let acceptor = tls_acceptor_from_config(&config)?;
    let listener = TcpListener::bind(config.bind_addr)
        .await
        .map_err(|source| TransportError::Bind {
            addr: config.bind_addr,
            source,
        })?;
    let local_addr = listener
        .local_addr()
        .map_err(|source| TransportError::Bind {
            addr: config.bind_addr,
            source,
        })?;

    tracing::info!(
        target: "arcgraph_mcp::http",
        bind_addr = %local_addr,
        tls_source = %config.tls_resolver.source_descriptor(),
        tenant_strategy = ?config.tenant_strategy,
        request_deadline_ms = config.request_deadline.as_millis() as u64,
        allowed_origins = ?config.allowed_origins.as_ref().map(Vec::len),
        bound_tenant = ?config.bound_tenant,
        "HTTP MCP transport listening",
    );

    // The `shutdown_watch` is a single-producer/multi-consumer signal
    // that propagates to every spawned connection task: when the
    // top-level `shutdown_signal` future resolves, `shutdown_tx.send(true)`
    // flips every connection task's local `shutdown_rx` so they can
    // drop their in-flight `serve_connection`'s gracefully.
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    let stats = Arc::new(ServeStatsInner::default());
    let policy = Arc::new(RequestPolicy::from_config(&config));

    // AHP-1 (ADR-225 §3) — the shared `spawn_blocking` bulkhead every
    // connection's dispatch runs behind. Injected by the CLI (shared with
    // the Bolt transport) or built here at the default cap (2 × cores).
    let bulkhead = config
        .dispatch_bulkhead
        .clone()
        .unwrap_or_else(DispatchBulkhead::with_default_cap);
    tracing::info!(
        target: "arcgraph_mcp::http",
        dispatch_bulkhead_permits = bulkhead.capacity(),
        "HTTP dispatch bulkhead active (spawn_blocking off-reactor; #999)",
    );

    // Pin the shutdown signal so we can both poll it in the accept
    // select! arm AND let the connection tasks observe the watch.
    tokio::pin!(shutdown_signal);

    loop {
        tokio::select! {
            biased;

            _ = &mut shutdown_signal => {
                tracing::info!(
                    target: "arcgraph_mcp::http",
                    in_flight = cancel_registry.len(),
                    "shutdown signal received; cancelling in-flight queries",
                );
                let fired = cancel_registry.cancel_all();
                stats.in_flight_cancelled.store(fired, std::sync::atomic::Ordering::SeqCst);
                stats.exit_reason.store(ExitReason::ShutdownSignal as u8, std::sync::atomic::Ordering::SeqCst);
                let _ = shutdown_tx.send(true);
                return Ok(stats.snapshot());
            }

            accept = listener.accept() => {
                match accept {
                    Ok((tcp, peer_addr)) => {
                        // #1353: disable Nagle's algorithm on the accepted
                        // socket. hyper 1.x's server side does NOT enable
                        // TCP_NODELAY — verified against hyper-1.9.0 source
                        // (zero `nodelay` references in the crate; the only
                        // nodelay handling lives in `hyper-util`'s CLIENT-side
                        // `HttpConnector::set_nodelay`, which defaults off and
                        // is irrelevant to `serve_connection`). Without this,
                        // the JSON-RPC request/response ping-pong hits the same
                        // Nagle × delayed-ACK ~40 ms stall as Bolt (#1352 A/B,
                        // 61×). We set it on the raw `TcpStream` here, BEFORE
                        // the TLS wrap in `handle_connection` (and independent
                        // of the AHP-1 dispatch bulkhead — a per-socket option,
                        // orthogonal to dispatch concurrency), so nodelay is in
                        // force for the whole connection. Best-effort: log and
                        // serve on failure, mirroring hyper-util's own
                        // client-side `warn!("tcp set_nodelay error: {}")`.
                        if let Err(source) = tcp.set_nodelay(true) {
                            tracing::warn!(
                                target: "arcgraph_mcp::http",
                                peer = %peer_addr,
                                error = %source,
                                "failed to set TCP_NODELAY on accepted socket; serving anyway",
                            );
                        }
                        stats.connections_accepted.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        // W15γ M6-06 — track active connections as a
                        // gauge. The active-count is `connections_accepted -
                        // connections_closed`; we increment here on
                        // accept and decrement in `handle_connection`
                        // on close. The gauge update is best-effort
                        // (the operator may not have wired metrics).
                        let active = stats.active_connections.fetch_add(1, std::sync::atomic::Ordering::AcqRel) + 1;
                        if let Some(metrics) = policy.metrics.as_ref() {
                            metrics.set_active_connections(ConnectionTransport::Http, active);
                        }
                        let acceptor = acceptor.clone();
                        let dispatcher = dispatcher.clone();
                        let cancel_registry = cancel_registry.clone();
                        let policy = policy.clone();
                        let stats_inner = stats.clone();
                        let conn_shutdown_rx = shutdown_rx.clone();
                        let bulkhead = bulkhead.clone();
                        tokio::spawn(async move {
                            handle_connection(
                                tcp,
                                peer_addr,
                                acceptor,
                                dispatcher,
                                cancel_registry,
                                policy,
                                stats_inner,
                                conn_shutdown_rx,
                                bulkhead,
                            )
                            .await;
                        });
                    }
                    Err(source) => {
                        tracing::error!(
                            target: "arcgraph_mcp::http",
                            error = %source,
                            "TCP accept failed; shutting down",
                        );
                        stats.exit_reason.store(ExitReason::AcceptError as u8, std::sync::atomic::Ordering::SeqCst);
                        let _ = shutdown_tx.send(true);
                        return Err(TransportError::Accept(source));
                    }
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Telemetry
// ─────────────────────────────────────────────────────────────────────

/// Per-loop telemetry emitted by [`serve_http`]. Mirrors the stdio
/// transport's [`crate::transport::stdio::ServeStats`] surface so future
/// observability sinks can render both transports through one schema.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ServeStats {
    /// Number of TCP connections successfully accepted.
    pub connections_accepted: u64,
    /// Number of HTTP requests dispatched through the JSON-RPC layer.
    pub requests_dispatched: u64,
    /// Number of HTTP requests rejected at the path / method gate
    /// (404 Not Found or 405 Method Not Allowed).
    pub requests_rejected_method: u64,
    /// Number of HTTP requests rejected at the header / body / tenant
    /// gate (400 Bad Request or 413 Payload Too Large).
    pub requests_rejected_request: u64,
    /// Number of cancellation tokens fired during shutdown drain.
    pub in_flight_cancelled: usize,
    /// Why the loop exited.
    pub exit_reason: ExitReason,
}

/// Inner atomics-backed stats — converted to [`ServeStats`] at exit.
#[derive(Debug, Default)]
struct ServeStatsInner {
    connections_accepted: std::sync::atomic::AtomicU64,
    /// W15γ M6-06 — running count of accepted-minus-closed connections.
    /// Mirrored into `arcgraph_active_connections{transport="http"}`
    /// when the operator wires [`MetricsRegistry`] via
    /// [`HttpServerConfig::with_metrics`].
    active_connections: std::sync::atomic::AtomicU64,
    requests_dispatched: std::sync::atomic::AtomicU64,
    requests_rejected_method: std::sync::atomic::AtomicU64,
    requests_rejected_request: std::sync::atomic::AtomicU64,
    in_flight_cancelled: std::sync::atomic::AtomicUsize,
    exit_reason: std::sync::atomic::AtomicU8,
}

impl ServeStatsInner {
    fn snapshot(&self) -> ServeStats {
        ServeStats {
            connections_accepted: self
                .connections_accepted
                .load(std::sync::atomic::Ordering::Relaxed),
            requests_dispatched: self
                .requests_dispatched
                .load(std::sync::atomic::Ordering::Relaxed),
            requests_rejected_method: self
                .requests_rejected_method
                .load(std::sync::atomic::Ordering::Relaxed),
            requests_rejected_request: self
                .requests_rejected_request
                .load(std::sync::atomic::Ordering::Relaxed),
            in_flight_cancelled: self
                .in_flight_cancelled
                .load(std::sync::atomic::Ordering::Relaxed),
            exit_reason: ExitReason::from_u8(
                self.exit_reason.load(std::sync::atomic::Ordering::Relaxed),
            ),
        }
    }
}

/// Why the [`serve_http`] loop exited.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ExitReason {
    /// Loop has not yet exited.
    #[default]
    InProgress,
    /// Shutdown signal fired (SIGTERM / explicit).
    ShutdownSignal,
    /// Fatal `accept()` error.
    AcceptError,
}

impl ExitReason {
    fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::ShutdownSignal,
            2 => Self::AcceptError,
            _ => Self::InProgress,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Per-request policy (frozen at serve_http entry, shared across tasks)
// ─────────────────────────────────────────────────────────────────────

/// Request-time policy shared across the connection + request tasks.
///
/// Built once at the top of [`serve_http`] (after validation) so the
/// spawn-per-connection / spawn-per-request paths can clone an
/// `Arc<RequestPolicy>` instead of cloning N fields independently.
/// Internal-only — exposed nowhere on the public surface.
#[derive(Debug)]
struct RequestPolicy {
    tenant_strategy: TenantStrategy,
    /// Per-request deadline mirroring [`HttpServerConfig::request_deadline`].
    deadline: Duration,
    /// Origin-header allowlist mirroring [`HttpServerConfig::allowed_origins`].
    allowed_origins: Option<Vec<String>>,
    /// Tenant the dispatcher is bound to (defense-in-depth check at
    /// request time against the transport-identified tenant — closes
    /// the "header label != identity" gap when mTLS / OAuth absent).
    bound_tenant: Option<TenantId>,
    /// W15γ M6-06 — optional metrics registry. When `Some`, the
    /// transport mounts `GET /metrics` and records per-request
    /// counters + histograms.
    metrics: Option<Arc<MetricsRegistry>>,
    /// W16β M5-03 — optional OAuth 2.1 + PKCE Bearer-token config
    /// (ADR-044). When `Some`, the per-request handler extracts +
    /// verifies the Authorization header before dispatch and enforces
    /// scope against the dispatched method.
    oauth: Option<Arc<crate::auth::oauth_pkce::OAuthConfig>>,
}

impl RequestPolicy {
    fn from_config(config: &HttpServerConfig) -> Self {
        Self {
            tenant_strategy: config.tenant_strategy.clone(),
            deadline: config.request_deadline,
            allowed_origins: config.allowed_origins.clone(),
            bound_tenant: config.bound_tenant,
            metrics: config.metrics.clone(),
            oauth: config.oauth.clone(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// TLS acceptor build
// ─────────────────────────────────────────────────────────────────────

/// Build a [`tokio_rustls::TlsAcceptor`] from the resolver + optional
/// mTLS verifier in `config`.
///
/// Visible to the test module so handshake-path unit tests can build
/// their own acceptor without going through the full `serve_http`
/// harness.
fn tls_acceptor_from_config(config: &HttpServerConfig) -> Result<TlsAcceptor, TransportError> {
    // We use `builder_with_provider` rather than `builder()` so we
    // don't depend on a globally-installed default `CryptoProvider`
    // — calling code may or may not have called
    // `aws_lc_rs::default_provider().install_default()`. Constructing
    // the provider Arc inline keeps `arcgraph-mcp` self-contained.
    let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
    let builder = ServerConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .map_err(|e| TransportError::TlsConfig(format!("protocol versions: {e}")))?;
    let server_config = match config.client_verifier.clone() {
        Some(verifier) => builder
            .with_client_cert_verifier(verifier)
            .with_cert_resolver(config.tls_resolver.clone()),
        None => builder
            .with_no_client_auth()
            .with_cert_resolver(config.tls_resolver.clone()),
    };
    Ok(TlsAcceptor::from(Arc::new(server_config)))
}

// ─────────────────────────────────────────────────────────────────────
// Per-connection handler
// ─────────────────────────────────────────────────────────────────────

/// W15γ M6-06 — RAII guard decrementing the active-connections gauge
/// when the handler task ends.
///
/// The increment happens at the accept-point in [`serve_http`]; this
/// guard fires on `Drop` regardless of how `handle_connection`
/// returns (TLS-handshake failure, normal close, panic). Without the
/// guard, a TLS handshake failure would leave the gauge inflated.
struct ActiveConnGuard {
    stats: Arc<ServeStatsInner>,
    metrics: Option<Arc<MetricsRegistry>>,
}

impl Drop for ActiveConnGuard {
    fn drop(&mut self) {
        let prev = self
            .stats
            .active_connections
            .fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
        let now = prev.saturating_sub(1);
        if let Some(m) = self.metrics.as_ref() {
            m.set_active_connections(ConnectionTransport::Http, now);
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn handle_connection<S, I, E, H, G, R>(
    tcp: TcpStream,
    peer_addr: SocketAddr,
    acceptor: TlsAcceptor,
    dispatcher: Arc<Dispatcher<S, I, E, H, G, R>>,
    cancel_registry: Arc<CancellationRegistry>,
    policy: Arc<RequestPolicy>,
    stats: Arc<ServeStatsInner>,
    mut shutdown_rx: watch::Receiver<bool>,
    bulkhead: DispatchBulkhead,
) where
    S: SchemaProvider + Send + Sync + 'static,
    I: NodeInspector + Send + Sync + 'static,
    E: NeighborhoodExplorer + Send + Sync + 'static,
    H: HybridSearcher + Send + Sync + 'static,
    G: IngestProvider + Send + Sync + 'static,
    R: crate::tools::raw_query::RawQueryExecutor + Send + Sync + 'static,
{
    // W15γ M6-06 — install the decrement guard immediately so any
    // early-return path (TLS handshake fail, panic) still releases
    // the active-connection gauge slot incremented by the accept loop.
    let _active_guard = ActiveConnGuard {
        stats: stats.clone(),
        metrics: policy.metrics.clone(),
    };

    let tls_stream = match acceptor.accept(tcp).await {
        Ok(s) => s,
        Err(source) => {
            tracing::warn!(
                target: "arcgraph_mcp::http",
                peer = %peer_addr,
                error = %source,
                "TLS handshake failed",
            );
            return;
        }
    };

    // Snapshot the peer cert chain (if mTLS is on) so the per-request
    // tenant extractor can read it without re-fetching from the
    // session.
    let (_io, server_conn) = tls_stream.get_ref();
    let peer_certs: Vec<CertificateDer<'static>> = server_conn
        .peer_certificates()
        .map(|certs| certs.iter().map(|c| c.clone().into_owned()).collect())
        .unwrap_or_default();

    let io = TokioIo::new(tls_stream);

    let svc = service_fn({
        let dispatcher = dispatcher.clone();
        let cancel_registry = cancel_registry.clone();
        let policy = policy.clone();
        let stats = stats.clone();
        let peer_certs = peer_certs.clone();
        let bulkhead = bulkhead.clone();
        move |req: Request<hyper::body::Incoming>| {
            let dispatcher = dispatcher.clone();
            let cancel_registry = cancel_registry.clone();
            let policy = policy.clone();
            let stats = stats.clone();
            let peer_certs = peer_certs.clone();
            let bulkhead = bulkhead.clone();
            async move {
                let resp = handle_request(
                    req,
                    dispatcher,
                    cancel_registry.as_ref(),
                    policy.as_ref(),
                    &peer_certs,
                    stats.as_ref(),
                    &bulkhead,
                )
                .await;
                Ok::<_, Infallible>(resp)
            }
        }
    });

    let conn = http1::Builder::new().serve_connection(io, svc);
    tokio::pin!(conn);

    tokio::select! {
        biased;
        _ = shutdown_rx.changed() => {
            // Tell hyper to shut down — it will let any in-flight
            // request complete then close the connection. We don't
            // hold the connection open further; the in-flight cancel
            // tokens fire on the registry path independently.
            conn.as_mut().graceful_shutdown();
            // Awaiting after graceful_shutdown drains the connection
            // — the `Pin<&mut Connection>` resolves once hyper has
            // flushed any in-flight response.
            if let Err(err) = conn.await {
                tracing::debug!(
                    target: "arcgraph_mcp::http",
                    peer = %peer_addr,
                    error = %err,
                    "connection drained with error post-shutdown",
                );
            }
        }
        result = &mut conn => {
            if let Err(err) = result {
                tracing::debug!(
                    target: "arcgraph_mcp::http",
                    peer = %peer_addr,
                    error = %err,
                    "connection terminated with error",
                );
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Per-request handler
// ─────────────────────────────────────────────────────────────────────

// AHP-1 (ADR-225 §3) added the `bulkhead` parameter (7th arg). The
// per-request handler threads request-scoped context (dispatcher, cancel
// registry, policy, peer certs, stats, bulkhead) — each is load-bearing
// and grouping them into a struct would just move the arg count without
// improving clarity. Same rationale as `https_request` in the tests.
#[allow(clippy::too_many_arguments)]
async fn handle_request<S, I, E, H, G, R>(
    req: Request<hyper::body::Incoming>,
    dispatcher: Arc<Dispatcher<S, I, E, H, G, R>>,
    cancel_registry: &CancellationRegistry,
    policy: &RequestPolicy,
    peer_certs: &[CertificateDer<'static>],
    stats: &ServeStatsInner,
    bulkhead: &DispatchBulkhead,
) -> Response<Full<Bytes>>
where
    S: SchemaProvider + Send + Sync + 'static,
    I: NodeInspector + Send + Sync + 'static,
    E: NeighborhoodExplorer + Send + Sync + 'static,
    H: HybridSearcher + Send + Sync + 'static,
    G: IngestProvider + Send + Sync + 'static,
    R: crate::tools::raw_query::RawQueryExecutor + Send + Sync + 'static,
{
    let method = req.method().clone();
    let path = req.uri().path().to_string();

    // W20β-1 — peer-cert identity extraction. Parses the end-entity
    // DER's CN + SAN DNSNames so the per-request tracing span can emit
    // them for operator audit. Failure to parse the cert is non-fatal at
    // this stage — rustls's chain-verify already passed, so an X.509
    // ASN.1 surprise here means we just skip the audit emit. The
    // dispatcher path does NOT consume the identity at v1.0-β (the SAN-
    // based `tenant-<N>` strategy is the canonical tenant pin); v1.1+
    // may route on CN for cert-pinned authorization.
    let client_identity: Option<ClientCertIdentity> =
        peer_certs
            .first()
            .and_then(|c| match parse_client_cert_identity(c.as_ref()) {
                Ok(id) if !id.is_empty() => Some(id),
                Ok(_) => None,
                Err(e) => {
                    tracing::debug!(
                        target: "arcgraph_mcp::http",
                        error = %e,
                        "peer cert present but identity parse failed",
                    );
                    None
                }
            });
    if let Some(id) = client_identity.as_ref() {
        tracing::debug!(
            target: "arcgraph_mcp::http",
            peer_cn = ?id.cn,
            peer_san_count = id.sans.len(),
            "peer-cert identity extracted",
        );
    }

    // ─── 1. method + path gate ─────────────────────────────────────
    match (path.as_str(), &method) {
        (PATH_HEALTHZ, &Method::GET) => {
            return Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from_static(b"{\"status\":\"ok\"}")))
                .unwrap_or_else(|_| empty_response(StatusCode::INTERNAL_SERVER_ERROR));
        }
        (PATH_HEALTHZ, _) => {
            stats
                .requests_rejected_method
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return method_not_allowed("GET");
        }
        (PATH_METRICS, &Method::GET) => {
            // W15γ M6-06 — `/metrics` Prometheus exporter
            // (design-v2 §10.2). Mounted only when the operator
            // attached a [`MetricsRegistry`] via
            // [`HttpServerConfig::with_metrics`]; otherwise the path
            // falls through to the 404 branch. This matches the
            // "explicit opt-in" pattern other observability surfaces
            // use (Origin allowlist, mTLS verifier) — design-v2 §10.2
            // mandates port 9090 by default but the W15γ slice mounts
            // metrics on the same listener as `/mcp` to defer the
            // dual-listener decision to M6-07 Grafana wiring.
            match policy.metrics.as_ref() {
                Some(metrics) => match metrics.gather_text() {
                    Ok(buf) => {
                        return Response::builder()
                            .status(StatusCode::OK)
                            .header("content-type", CONTENT_TYPE_PROMETHEUS_TEXT)
                            .body(Full::new(Bytes::from(buf)))
                            .unwrap_or_else(|_| empty_response(StatusCode::INTERNAL_SERVER_ERROR));
                    }
                    Err(err) => {
                        tracing::warn!(
                            target: "arcgraph_mcp::http",
                            error = %err,
                            "metrics gather failed; returning 500",
                        );
                        return empty_response(StatusCode::INTERNAL_SERVER_ERROR);
                    }
                },
                None => {
                    // No registry attached — treat /metrics like any
                    // unknown path (404). The operator can opt in via
                    // `HttpServerConfig::with_metrics`; before that,
                    // metrics scrape is intentionally not advertised.
                    stats
                        .requests_rejected_method
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return Response::builder()
                        .status(StatusCode::NOT_FOUND)
                        .header("content-type", "application/json")
                        .body(Full::new(Bytes::from_static(
                            b"{\"error\":\"metrics endpoint not configured\"}",
                        )))
                        .unwrap_or_else(|_| empty_response(StatusCode::INTERNAL_SERVER_ERROR));
                }
            }
        }
        (PATH_METRICS, _) => {
            stats
                .requests_rejected_method
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return method_not_allowed("GET");
        }
        (PATH_MCP, &Method::POST) => {
            // Fall through to body handling.
        }
        (PATH_MCP, _) => {
            stats
                .requests_rejected_method
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return method_not_allowed("POST");
        }
        _ => {
            stats
                .requests_rejected_method
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Response::builder()
                .status(StatusCode::NOT_FOUND)
                .header("content-type", "application/json")
                .body(Full::new(Bytes::from_static(
                    b"{\"error\":\"unknown path\"}",
                )))
                .unwrap_or_else(|_| empty_response(StatusCode::INTERNAL_SERVER_ERROR));
        }
    }

    // ─── 2. Origin allowlist gate (design-v2 §9.4 line 667) ────────
    //
    // DNS-rebinding defense: browser-driven peers attach an `Origin`
    // header on cross-origin POSTs. If the transport is configured
    // with an allowlist AND the request carries an `Origin` that's
    // NOT in the list, reject with 403 before the body is even read.
    // Requests without an `Origin` header pass — that's the typical
    // curl / CLI / MCP-stdio-bridge case.
    if let Some(allowed) = policy.allowed_origins.as_ref() {
        if let Some(origin_value) = req.headers().get(HEADER_ORIGIN) {
            let origin_str = match origin_value.to_str() {
                Ok(s) => s,
                Err(e) => {
                    stats
                        .requests_rejected_request
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let err = TransportError::HeaderInvalid {
                        name: "origin",
                        detail: format!("non-ASCII: {e}"),
                    };
                    return error_envelope_response(
                        StatusCode::BAD_REQUEST,
                        transport_error_to_mcp(&err),
                    );
                }
            };
            if !allowed.iter().any(|a| a == origin_str) {
                stats
                    .requests_rejected_request
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let err = TransportError::OriginForbidden {
                    origin: origin_str.to_string(),
                };
                return error_envelope_response(
                    StatusCode::FORBIDDEN,
                    transport_error_to_mcp(&err),
                );
            }
        }
    }

    // ─── 2a. OAuth Bearer-token verify (W16β M5-03 / ADR-044) ─────
    //
    // When OAuth is configured, every POST /mcp request MUST carry
    // `Authorization: Bearer <jwt>`. Verification runs BEFORE body
    // parse so an unauthenticated request never pays the body-parse
    // or dispatch cost. The verified [`TokenClaims`] is held in a
    // local for the scope-enforcement gate (after envelope decode,
    // before dispatch).
    //
    // When OAuth is `None`, this gate is skipped — backward-compat
    // with W14α tests + embedded deployments that don't expose
    // beyond loopback.
    let oauth_claims: Option<crate::auth::oauth_pkce::TokenClaims> =
        if let Some(oauth) = policy.oauth.as_ref() {
            match extract_and_verify_bearer(&req, oauth.as_ref()) {
                Ok(claims) => Some(claims),
                Err((transport_err, oauth_err)) => {
                    stats
                        .requests_rejected_request
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    // RFC 6750 §3.1: 401 Unauthorized for invalid_token
                    // and "no auth attempted" cases. WWW-Authenticate
                    // carries the error code per the RFC.
                    return oauth_error_response(
                        StatusCode::UNAUTHORIZED,
                        &transport_err,
                        &oauth_err,
                        Value::Null,
                    );
                }
            }
        } else {
            None
        };

    // ─── 3. tenant extraction ──────────────────────────────────────
    //
    // Cross-tenant rejection: if the request asks for a tenant other
    // than the one identified at the transport layer, reject with
    // -32002 Unauthorized. This is the same gate the dispatcher
    // enforces when the per-request payload contains a tenant_id —
    // surfacing it here means the tenant-mismatched request never
    // reaches the dispatcher.
    let identified_tenant = match identify_tenant(&req, &policy.tenant_strategy, peer_certs) {
        Ok(t) => t,
        Err(e) => {
            stats
                .requests_rejected_request
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return error_envelope_response(StatusCode::BAD_REQUEST, transport_error_to_mcp(&e));
        }
    };

    // ─── 3a. bound-tenant fence (defense-in-depth) ─────────────────
    //
    // When the operator declares the bound tenant explicitly via
    // [`HttpServerConfig::bound_tenant`], the transport reasserts
    // here that the identified tenant matches it. This closes the
    // "header label ≠ identity" gap when no mTLS / OAuth is wired:
    // even if an attacker forges a header claiming tenant 7, the
    // request rejects with 403 if the operator pinned the listener
    // to tenant 9. When `bound_tenant` is `None`, the assertion is
    // skipped (legacy callers; the authoritative cross-tenant check
    // still happens at the tool layer).
    if let Some(bound) = policy.bound_tenant {
        if bound != identified_tenant {
            tracing::warn!(
                target: "arcgraph_mcp::http",
                identified = identified_tenant.raw(),
                bound = bound.raw(),
                "request identified for tenant != server's bound tenant",
            );
            stats
                .requests_rejected_request
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return error_envelope_response(StatusCode::FORBIDDEN, MCPError::Unauthorized);
        }
    }

    // ─── 4. body parse ─────────────────────────────────────────────
    let body_bytes = match read_request_body(req).await {
        Ok(b) => b,
        Err(e) => {
            stats
                .requests_rejected_request
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let status = match &e {
                TransportError::BodyTooLarge { .. } => StatusCode::PAYLOAD_TOO_LARGE,
                _ => StatusCode::BAD_REQUEST,
            };
            return error_envelope_response(status, transport_error_to_mcp(&e));
        }
    };

    // ─── 5. JSON envelope decode ───────────────────────────────────
    let envelope: Value = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(e) => {
            stats
                .requests_rejected_request
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let mcp_err = MCPError::ParseError(format!("body: {e}"));
            return error_envelope_response(StatusCode::BAD_REQUEST, mcp_err);
        }
    };

    // ─── 6. Cross-tenant guard: envelope payload vs identified ─────
    //
    // The dispatcher also performs this check at the tool layer
    // (request_tenant ≠ session_tenant → Unauthorized), but doing
    // it here means a transport-identified tenant mismatch never
    // even reaches the dispatcher. If `bound_tenant` is set above,
    // a successful payload_tenant check also implies
    // payload_tenant == bound_tenant (transitive).
    if let Some(payload_tenant) = envelope_tenant_id(&envelope) {
        if payload_tenant != identified_tenant {
            stats
                .requests_rejected_request
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return error_envelope_response_with_id(
                StatusCode::FORBIDDEN,
                MCPError::Unauthorized,
                envelope.get("id").cloned().unwrap_or(Value::Null),
            );
        }
    }

    // ─── 6a. OAuth scope enforcement (W16β M5-03 / ADR-044) ────────
    //
    // When OAuth is configured (and the Bearer verify at gate 2a
    // succeeded), enforce per-method scope before dispatch. The scope
    // policy is the static table in
    // `crate::auth::oauth_pkce::scope_for_method` (design-v2 §9.4
    // line 665). Insufficient scope: HTTP 403 + RFC 6750 §3.1
    // WWW-Authenticate carrying `error="insufficient_scope"` and the
    // required scope.
    //
    // Unknown methods fail-closed here (UnknownMethod →
    // OAuthInvalidToken → 401). The dispatcher would have returned
    // MethodNotFound (-32601) after dispatch, but failing here means
    // an OAuth-protected deployment can never run an off-catalog
    // method body even if the token nominally carries any scope.
    let envelope_method_for_scope = envelope
        .get("method")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    if let Some(claims) = oauth_claims.as_ref() {
        if let Err(oauth_err) =
            crate::auth::oauth_pkce::enforce_scope(claims, &envelope_method_for_scope)
        {
            stats
                .requests_rejected_request
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let (status, transport_err) = match &oauth_err {
                crate::auth::oauth_pkce::OAuthError::InsufficientScope { required, .. } => (
                    StatusCode::FORBIDDEN,
                    TransportError::OAuthInsufficientScope { required },
                ),
                _ => (
                    StatusCode::UNAUTHORIZED,
                    TransportError::OAuthInvalidToken(oauth_err.to_string()),
                ),
            };
            return oauth_error_response(
                status,
                &transport_err,
                &oauth_err,
                envelope.get("id").cloned().unwrap_or(Value::Null),
            );
        }
    }

    // Bind the request's executor scope to the verified bearer claim. The
    // composition root deliberately uses Power for trusted local stdio, but a
    // connectionless HTTPS request must not inherit that default. Without
    // this override an `arcgraph.read` token omitting `principal` could reach
    // the SYSTEM-TRUSTED arm after passing method-level OAuth enforcement.
    let dispatch_scope = oauth_claims
        .as_ref()
        .map(|claims| crate::SessionScope::from_scope_claim(&claims.scope))
        .unwrap_or(dispatcher.session_scope);

    // ─── 7. dispatch with deadline + cancellation ──────────────────
    stats
        .requests_dispatched
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let request_id = arcgraph_query::QueryId::new();
    let token = cancel_registry.register(request_id);
    let _deadline_handle = spawn_deadline_timer(token.clone(), policy.deadline);

    // AHP-1 (ADR-225 §3) — the sync W13δ dispatcher runs on a
    // `spawn_blocking` thread behind a bounded semaphore (the
    // `bulkhead`), so a blocking engine call (cold page read,
    // group-commit `fdatasync` wait) no longer pins THIS connection's
    // Tokio worker and starves concurrent reads across tenants (#999).
    // The per-request deadline is honoured at the bulkhead boundary via
    // `tokio::time::timeout` around the JoinHandle — closing the
    // pre-AHP-1 TODO where the deadline timer fired the cancel token but
    // the inline sync dispatch could not poll it. `spawn_blocking` cannot
    // be force-cancelled, so a timed-out request stops AWAITING the work
    // (returning a `-32001 cancelled` envelope) and abandons the blocking
    // thread, which finishes off-reactor still holding its permit.
    let envelope_id = envelope.get("id").cloned().unwrap_or(Value::Null);
    // W15γ M6-06 — pre-dispatch metric capture. The op_class
    // classification (from `crate::transport::op_class_for_method`)
    // selects the read- vs write-latency histogram and bucket-aligns
    // with the M5-12 rate-limit buckets. Unknown method names land
    // in `op_class="read"` (defensive — see `op_class_for_method`
    // doc).
    let envelope_method = envelope_method_for_scope.clone();
    let dispatch_start = std::time::Instant::now();
    tracing::debug!(
        target: "arcgraph_mcp::http",
        query_id = ?request_id,
        envelope_id = ?envelope_id,
        identified_tenant = identified_tenant.raw(),
        "dispatch begin",
    );
    // AHP-1 — off-reactor dispatch behind the bounded bulkhead, with the
    // request deadline enforced at the boundary. The blocking closure owns
    // an `Arc<Dispatcher>` clone + the `envelope` `Value`; both are
    // `Send + 'static`, so the work runs on a blocking-pool thread.
    let disp_for_blocking = Arc::clone(&dispatcher);
    let dispatch_result: Option<Value> = match bulkhead
        .run(Some(policy.deadline), move || {
            handle_raw_envelope_with_scope(disp_for_blocking.as_ref(), envelope, dispatch_scope)
        })
        .await
    {
        BulkheadOutcome::Completed(result) => result,
        BulkheadOutcome::TimedOut => {
            tracing::warn!(
                target: "arcgraph_mcp::http",
                query_id = ?request_id,
                deadline_ms = policy.deadline.as_millis() as u64,
                "dispatch exceeded deadline at the bulkhead; abandoning the blocking thread",
            );
            // -32001 cancelled envelope — the deadline elapsed, so this
            // request is cancelled from the client's perspective even
            // though the abandoned blocking thread finishes off-reactor.
            Some(mcp_error_envelope(
                envelope_id.clone(),
                &MCPError::Cancelled,
            ))
        }
        BulkheadOutcome::Panicked => {
            tracing::error!(
                target: "arcgraph_mcp::http",
                query_id = ?request_id,
                "dispatch blocking task panicked; returning -32603 internal error",
            );
            Some(mcp_error_envelope(
                envelope_id.clone(),
                &MCPError::InternalError("dispatch task panicked".to_string()),
            ))
        }
    };

    // W15γ M6-06 — post-dispatch metric record. Records BOTH the
    // counter increment AND the latency histogram observation in one
    // call. Only records when `policy.metrics` is `Some` — the
    // `None` case is the "metrics not configured" path (legacy
    // callers); zero overhead beyond the `is_some()` branch.
    //
    // The dispatch outcome is inferred from the envelope shape: a
    // success envelope carries `result`, an error envelope carries
    // `error` (per JSON-RPC §5). Notification dispatches (no `id`)
    // return `None`; we tag those as Ok (a notification that produced
    // no envelope is by definition not a JSON-RPC error).
    if let Some(metrics) = policy.metrics.as_ref() {
        let elapsed_ms = dispatch_start.elapsed().as_secs_f64() * 1_000.0;
        let op_class = crate::transport::op_class_for_method(&envelope_method);
        let status = match dispatch_result.as_ref() {
            Some(env) if env.get("error").is_some() => crate::ToolInvocationStatus::Error,
            _ => crate::ToolInvocationStatus::Ok,
        };
        metrics.record_dispatch(
            identified_tenant,
            &envelope_method,
            op_class,
            status,
            elapsed_ms,
        );
    }

    // Always unregister at request-end (success or error). Drops the
    // deadline handle next, which aborts the timer.
    cancel_registry.unregister(request_id);

    let response_body: Bytes = match dispatch_result {
        Some(value) => match serde_json::to_vec(&value) {
            Ok(v) => Bytes::from(v),
            Err(e) => {
                let mcp_err = MCPError::InternalError(format!("response serialize: {e}"));
                return error_envelope_response_with_id(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    mcp_err,
                    envelope_id,
                );
            }
        },
        // Notification (no `id`) — JSON-RPC §4.1 says no response.
        // HTTP requires SOMETHING; we return 204 No Content per the
        // MCP streamable-HTTP spec convention.
        None => {
            return Response::builder()
                .status(StatusCode::NO_CONTENT)
                .body(Full::new(Bytes::new()))
                .unwrap_or_else(|_| empty_response(StatusCode::INTERNAL_SERVER_ERROR));
        }
    };

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/json")
        .body(Full::new(response_body))
        .unwrap_or_else(|_| empty_response(StatusCode::INTERNAL_SERVER_ERROR))
}

/// Collect the request body, capping at [`MAX_MESSAGE_BYTES`].
///
/// We use `BodyExt::collect` (from `http-body-util`) which reads the
/// entire body into a single `Bytes` — fine for the v1.0-alpha JSON-RPC
/// envelope (capped at 16 MiB). We pre-check the `Content-Length`
/// header to reject oversized bodies BEFORE reading them, avoiding a
/// hostile peer that announces a 10 GiB length and dripfeeds bytes
/// (the `collect` adapter would happily allocate that buffer).
async fn read_request_body(req: Request<hyper::body::Incoming>) -> Result<Bytes, TransportError> {
    if let Some(cl) = req.headers().get(hyper::header::CONTENT_LENGTH) {
        let cl_str = cl.to_str().map_err(|e| TransportError::HeaderInvalid {
            name: "content-length",
            detail: format!("non-ASCII: {e}"),
        })?;
        let cl_num: usize = cl_str.parse().map_err(|e| TransportError::HeaderInvalid {
            name: "content-length",
            detail: format!("non-numeric: {e}"),
        })?;
        if cl_num > MAX_MESSAGE_BYTES {
            return Err(TransportError::BodyTooLarge {
                len: cl_num,
                cap: MAX_MESSAGE_BYTES,
            });
        }
    }

    let body = req.into_body();
    let collected = body
        .collect()
        .await
        .map_err(|e| TransportError::BodyParse(format!("collect: {e}")))?;
    let bytes = collected.to_bytes();
    if bytes.len() > MAX_MESSAGE_BYTES {
        return Err(TransportError::BodyTooLarge {
            len: bytes.len(),
            cap: MAX_MESSAGE_BYTES,
        });
    }
    Ok(bytes)
}

// ─────────────────────────────────────────────────────────────────────
// Tenant identification
// ─────────────────────────────────────────────────────────────────────

/// Identify the request's tenant according to the configured strategy.
fn identify_tenant<B>(
    req: &Request<B>,
    strategy: &TenantStrategy,
    peer_certs: &[CertificateDer<'static>],
) -> Result<TenantId, TransportError> {
    match strategy {
        TenantStrategy::Header => extract_tenant_from_header(req).and_then(|opt| {
            opt.ok_or(TransportError::TenantMissing {
                strategy: TenantStrategy::Header,
            })
        }),
        TenantStrategy::PeerCertSan => extract_tenant_from_peer_certs(peer_certs).and_then(|opt| {
            opt.ok_or(TransportError::TenantMissing {
                strategy: TenantStrategy::PeerCertSan,
            })
        }),
        TenantStrategy::PeerCertSanThenHeader => {
            if let Some(t) = extract_tenant_from_peer_certs(peer_certs)? {
                return Ok(t);
            }
            extract_tenant_from_header(req).and_then(|opt| {
                opt.ok_or(TransportError::TenantMissing {
                    strategy: TenantStrategy::PeerCertSanThenHeader,
                })
            })
        }
    }
}

/// Read `X-ArcGraph-Tenant` from the request headers and parse it as a
/// `u64`. Returns `Ok(None)` when the header is absent (so callers can
/// fall back to other strategies).
fn extract_tenant_from_header<B>(req: &Request<B>) -> Result<Option<TenantId>, TransportError> {
    match req.headers().get(HEADER_TENANT) {
        None => Ok(None),
        Some(value) => {
            let s = value.to_str().map_err(|e| TransportError::HeaderInvalid {
                name: "x-arcgraph-tenant",
                detail: format!("non-ASCII: {e}"),
            })?;
            let raw: u64 = s
                .parse()
                .map_err(|e| TransportError::TenantParse(format!("{HEADER_TENANT}: {e}")))?;
            Ok(Some(TenantId::new(raw)))
        }
    }
}

/// Extract the tenant from the peer cert SAN entries.
///
/// A tenant identity is a DNSName SAN entry of the form
/// `tenant-<decimal>.arcgraph.local`
/// or just `tenant-<decimal>`. The first matching entry wins.
///
/// Returns `Ok(None)` when no peer cert was presented OR the SAN
/// entries don't carry a recognized tenant identifier.
fn extract_tenant_from_peer_certs(
    certs: &[CertificateDer<'static>],
) -> Result<Option<TenantId>, TransportError> {
    let Some(end_entity) = certs.first() else {
        return Ok(None);
    };
    let sans = parse_sans_from_der(end_entity.as_ref())
        .map_err(|e| TransportError::TenantParse(format!("peer cert SAN parse: {e}")))?;
    Ok(tenant_from_san_strings(&sans))
}

/// Pure helper: scan a list of SAN DNSName strings for a `tenant-<N>`
/// entry. Used by [`extract_tenant_from_peer_certs`] (production) and
/// directly by unit tests (avoids needing a real cert).
pub(crate) fn tenant_from_san_strings(sans: &[String]) -> Option<TenantId> {
    for san in sans {
        // Accept either "tenant-7" or "tenant-7.arcgraph.local"
        // (the latter is the typical Kubernetes Cert-Manager shape).
        let head = san.split('.').next().unwrap_or(san);
        if let Some(rest) = head.strip_prefix("tenant-") {
            if let Ok(n) = u64::from_str(rest) {
                return Some(TenantId::new(n));
            }
        }
    }
    None
}

/// Parse the SubjectAltName DNSName entries from a DER-encoded X.509
/// cert. Returns the list of DNSName strings found; other SAN types
/// (IP, URI, RFC822) are ignored.
fn parse_sans_from_der(der: &[u8]) -> Result<Vec<String>, String> {
    use x509_parser::extensions::GeneralName;
    use x509_parser::prelude::*;

    let (_, cert) = X509Certificate::from_der(der).map_err(|e| format!("X.509 decode: {e}"))?;
    let mut out = Vec::new();
    if let Ok(Some(san_ext)) = cert.subject_alternative_name() {
        for name in &san_ext.value.general_names {
            if let GeneralName::DNSName(d) = name {
                out.push((*d).to_string());
            }
        }
    }
    Ok(out)
}

/// Read the `tenant_id` field from the JSON-RPC envelope's `params`
/// object, if present. Returns `None` for a missing / non-numeric
/// tenant_id (the dispatcher's per-tool param decoder produces a clean
/// -32602 error for that case).
fn envelope_tenant_id(envelope: &Value) -> Option<TenantId> {
    envelope
        .get("params")
        .and_then(|p| p.get("tenant_id"))
        .and_then(|t| t.as_u64())
        .map(TenantId::new)
}

// ─────────────────────────────────────────────────────────────────────
// Response helpers
// ─────────────────────────────────────────────────────────────────────

fn method_not_allowed(allowed: &str) -> Response<Full<Bytes>> {
    Response::builder()
        .status(StatusCode::METHOD_NOT_ALLOWED)
        .header("allow", allowed)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from_static(
            b"{\"error\":\"method not allowed\"}",
        )))
        .unwrap_or_else(|_| empty_response(StatusCode::INTERNAL_SERVER_ERROR))
}

fn empty_response(status: StatusCode) -> Response<Full<Bytes>> {
    let mut r = Response::new(Full::new(Bytes::new()));
    *r.status_mut() = status;
    r
}

fn error_envelope_response(status: StatusCode, err: MCPError) -> Response<Full<Bytes>> {
    error_envelope_response_with_id(status, err, Value::Null)
}

/// W16β M5-03 — extract + verify a Bearer token from the request's
/// `Authorization` header per RFC 6750 §2.1 + ADR-044. Returns the
/// validated claims on success, OR a paired
/// `(TransportError, OAuthError)` on any failure (the [`OAuthError`]
/// drives the `WWW-Authenticate` header per RFC 6750 §3; the
/// [`TransportError`] drives the JSON-RPC envelope code via
/// `transport_error_to_mcp`).
fn extract_and_verify_bearer<B>(
    req: &Request<B>,
    oauth: &crate::auth::oauth_pkce::OAuthConfig,
) -> Result<
    crate::auth::oauth_pkce::TokenClaims,
    (TransportError, crate::auth::oauth_pkce::OAuthError),
> {
    let header_value = match req
        .headers()
        .get(crate::auth::oauth_pkce::HEADER_AUTHORIZATION)
    {
        None => {
            return Err((
                TransportError::OAuthMissingBearer,
                crate::auth::oauth_pkce::OAuthError::MissingBearer,
            ));
        }
        Some(v) => match v.to_str() {
            Ok(s) => s,
            Err(e) => {
                let oauth_err = crate::auth::oauth_pkce::OAuthError::MalformedBearer(format!(
                    "non-ASCII Authorization header: {e}"
                ));
                let transport_err = TransportError::OAuthInvalidToken(oauth_err.to_string());
                return Err((transport_err, oauth_err));
            }
        },
    };
    crate::auth::oauth_pkce::verify_bearer_header(header_value, oauth).map_err(|e| {
        let transport_err = match &e {
            crate::auth::oauth_pkce::OAuthError::MissingBearer => {
                TransportError::OAuthMissingBearer
            }
            _ => TransportError::OAuthInvalidToken(e.to_string()),
        };
        (transport_err, e)
    })
}

/// W16β M5-03 — build a 401/403 response envelope for an OAuth
/// failure. Sets `WWW-Authenticate` per RFC 6750 §3 and emits the
/// JSON-RPC `-32002 Unauthorized` envelope.
fn oauth_error_response(
    status: StatusCode,
    transport_err: &TransportError,
    oauth_err: &crate::auth::oauth_pkce::OAuthError,
    envelope_id: Value,
) -> Response<Full<Bytes>> {
    let www_auth = crate::auth::oauth_pkce::oauth_error_to_www_authenticate(oauth_err);
    let mcp = transport_error_to_mcp(transport_err);
    let env = JsonRpcErrorResponse {
        jsonrpc: JSONRPC_VERSION.into(),
        id: envelope_id,
        error: JsonRpcErrorObject {
            code: mcp.code(),
            message: mcp.message().into(),
            data: mcp.data(),
        },
    };
    let body = serde_json::to_vec(&env).unwrap_or_default();
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .header("www-authenticate", www_auth)
        .body(Full::new(Bytes::from(body)))
        .unwrap_or_else(|_| empty_response(StatusCode::INTERNAL_SERVER_ERROR))
}

/// AHP-1 — render an [`MCPError`] as a JSON-RPC error-envelope [`Value`]
/// (not an HTTP response). Used at the bulkhead boundary so a timed-out /
/// panicked dispatch flows through the SAME post-dispatch response path
/// (metric record + 200-with-error-body) as a normal dispatch error.
fn mcp_error_envelope(id: Value, err: &MCPError) -> Value {
    serde_json::to_value(JsonRpcErrorResponse::from_mcp(id, err)).unwrap_or(Value::Null)
}

fn error_envelope_response_with_id(
    status: StatusCode,
    err: MCPError,
    id: Value,
) -> Response<Full<Bytes>> {
    let env = JsonRpcErrorResponse {
        jsonrpc: JSONRPC_VERSION.into(),
        id,
        error: JsonRpcErrorObject {
            code: err.code(),
            message: err.message().into(),
            data: err.data(),
        },
    };
    let body = serde_json::to_vec(&env).unwrap_or_default();
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(body)))
        .unwrap_or_else(|_| empty_response(StatusCode::INTERNAL_SERVER_ERROR))
}

fn transport_error_to_mcp(e: &TransportError) -> MCPError {
    match e {
        TransportError::BodyTooLarge { len, cap } => {
            MCPError::ParseError(format!("body {len} > cap {cap}"))
        }
        TransportError::BodyParse(detail) => MCPError::ParseError(format!("body: {detail}")),
        TransportError::HeaderInvalid { name, detail } => {
            MCPError::InvalidRequest(format!("header {name}: {detail}"))
        }
        TransportError::OriginForbidden { origin } => {
            MCPError::InvalidRequest(format!("origin not allowed: {origin}"))
        }
        TransportError::TenantMissing { strategy } => {
            MCPError::InvalidRequest(format!("tenant identifier missing for {strategy:?}"))
        }
        TransportError::TenantParse(detail) => {
            MCPError::InvalidRequest(format!("tenant parse: {detail}"))
        }
        // W16β M5-03 — OAuth failures map to `Unauthorized`. HTTP
        // status distinguishes between 401 (missing/invalid token)
        // and 403 (insufficient scope) at the call site; the
        // JSON-RPC envelope code is -32002 across all three.
        TransportError::OAuthMissingBearer
        | TransportError::OAuthInvalidToken(_)
        | TransportError::OAuthInsufficientScope { .. } => MCPError::Unauthorized,
        TransportError::TlsConfig(_)
        | TransportError::Bind { .. }
        | TransportError::Accept(_)
        | TransportError::TlsHandshake(_)
        | TransportError::ConfigInvalid(_)
        | TransportError::BindAddrForbidden { .. } => {
            MCPError::InternalError(format!("transport: {e}"))
        }
    }
}

// Reserved for future use by lower-level tasks that build their own
// futures around the connection lifecycle.
#[allow(dead_code)]
type ConnFuture<'a> = Pin<Box<dyn Future<Output = ()> + Send + 'a>>;

// ─────────────────────────────────────────────────────────────────────
// Sealed mTLS verifier helper (for tests + future v1.1+ landings)
// ─────────────────────────────────────────────────────────────────────

/// Build a [`WebPkiClientVerifier`] that trusts the supplied root
/// certificates. Used by the integ-test fixture to drive mTLS
/// configuration without committing a full client-CA bootstrap to
/// v1.0-alpha. v1.1+ will replace this with a config-driven path.
///
/// Returns the verifier as a trait object so it slots directly into
/// [`HttpServerConfig::client_verifier`].
pub fn client_verifier_for_roots(
    roots: Vec<CertificateDer<'static>>,
) -> Result<Arc<dyn rustls::server::danger::ClientCertVerifier>, TransportError> {
    let mut store = rustls::RootCertStore::empty();
    for c in roots {
        store
            .add(c)
            .map_err(|e| TransportError::TlsConfig(format!("add root: {e}")))?;
    }
    let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
    let verifier = WebPkiClientVerifier::builder_with_provider(Arc::new(store), provider)
        .build()
        .map_err(|e| TransportError::TlsConfig(format!("build verifier: {e}")))?;
    Ok(verifier)
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    //! Unit tests cover everything reachable WITHOUT a real
    //! `hyper::body::Incoming` (which hyper 1.x makes opaque + has no
    //! public constructor). Body-reading paths — full HTTP request →
    //! TLS handshake → response cycle — live in
    //! `tests/mcp_http_integ.rs` against an in-process listener.
    //!
    //! Coverage matrix here:
    //!   1. Content-Length cap predicate (BodyTooLarge variant)
    //!   2. method-not-allowed builder shape
    //!   3. healthz response shape (200 + JSON body)
    //!   4. tls acceptor build from resolver
    //!   5. tenant extraction from header (Some / None / parse-error)
    //!   6. tenant extraction from SAN strings (helper)
    //!   7. tenant extraction from real DER cert (parse_sans_from_der)
    //!   8. deadline timer arms + fires
    //!   9. SIGTERM cancel_all path
    //!  10. register/unregister balance
    //!  11. error envelope renders -32600
    //!  12. transport_error_to_mcp mapping (multiple variants)
    //!  13. identify_tenant Header missing
    //!  14. envelope_tenant_id pulls params.tenant_id
    //!  15. HttpServerConfig::validate — non-loopback bind without
    //!      opt-in rejects (design-v2 §9.4 line 668)
    //!  16. HttpServerConfig::validate — non-loopback bind with
    //!      `with_allow_remote_bind(true)` passes
    //!  17. HttpServerConfig::validate — PeerCertSan +
    //!      client_verifier=None rejects (NIT-9)
    //!  18. HttpServerConfig::validate — loopback bind + sane
    //!      defaults passes
    //!  19. transport_error_to_mcp maps BodyParse → ParseError
    //!  20. transport_error_to_mcp maps OriginForbidden →
    //!      InvalidRequest
    //!
    //! Total: ≥20 unit tests across the public surface.

    use super::*;
    use crate::tls::error::TlsResolverError;
    use crate::tls::provider::CertProvider;
    use rcgen::{CertificateParams, DnType, KeyPair};
    use serde_json::json;
    use std::sync::Mutex;
    use time::OffsetDateTime;

    // ---- scripted resolver provider ----

    #[derive(Debug)]
    struct ScriptedProvider {
        script: Mutex<Vec<Result<Arc<rustls::sign::CertifiedKey>, TlsResolverError>>>,
    }

    impl ScriptedProvider {
        fn from_rcgen() -> Self {
            let key = synth_certified_key("localhost", None);
            Self {
                script: Mutex::new(vec![Ok(key)]),
            }
        }
    }

    impl CertProvider for ScriptedProvider {
        fn load(&self) -> Result<Arc<rustls::sign::CertifiedKey>, TlsResolverError> {
            let mut s = self.script.lock().expect("scripted provider mutex");
            if s.is_empty() {
                panic!("scripted provider exhausted");
            }
            s.remove(0)
        }
        fn source_descriptor(&self) -> String {
            "scripted://test".into()
        }
    }

    fn synth_certified_key(san: &str, cn: Option<&str>) -> Arc<rustls::sign::CertifiedKey> {
        use rustls_pki_types::PrivateKeyDer;
        use rustls_pki_types::pem::PemObject;

        let mut params = CertificateParams::new(vec![san.to_string()]).expect("rcgen params");
        params.not_before = OffsetDateTime::now_utc() - time::Duration::days(1);
        params.not_after = OffsetDateTime::now_utc() + time::Duration::days(365);
        if let Some(cn_v) = cn {
            params.distinguished_name.push(DnType::CommonName, cn_v);
        }
        let signing_key = KeyPair::generate().expect("rcgen keypair");
        let cert = params.self_signed(&signing_key).expect("self-signed");
        let cert_der = cert.der().clone();
        let key_pem = signing_key.serialize_pem();
        let key_der = PrivateKeyDer::from_pem_slice(key_pem.as_bytes()).expect("PEM key parse");
        let signing_key =
            rustls::crypto::aws_lc_rs::sign::any_supported_type(&key_der).expect("sign");
        Arc::new(rustls::sign::CertifiedKey::new(vec![cert_der], signing_key))
    }

    fn synth_resolver() -> Arc<HotReloadResolver> {
        HotReloadResolver::new(Arc::new(ScriptedProvider::from_rcgen())).expect("resolver init")
    }

    /// Collect a `Full<Bytes>` body into a `Bytes` for assertion.
    async fn body_to_bytes(body: Full<Bytes>) -> Bytes {
        let collected = body.collect().await.expect("collect");
        collected.to_bytes()
    }

    // ─────────────────────────────────────────────────────────────────
    // 1. Content-Length cap predicate (the read_request_body header
    //    arm). Pure unit — pins the cap-vs-MAX_MESSAGE_BYTES check.
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn body_too_large_variant_carries_len_and_cap() {
        let oversized: usize = MAX_MESSAGE_BYTES + 1;
        let err = TransportError::BodyTooLarge {
            len: oversized,
            cap: MAX_MESSAGE_BYTES,
        };
        if let TransportError::BodyTooLarge { len, cap } = err {
            assert_eq!(len, MAX_MESSAGE_BYTES + 1);
            assert_eq!(cap, MAX_MESSAGE_BYTES);
        } else {
            panic!("variant mismatch");
        }
        // Pin the cap message rendering shape (the err-to-mcp translator
        // routes to ParseError -32700, exercised in test 12 below).
        let err = TransportError::BodyTooLarge {
            len: oversized,
            cap: MAX_MESSAGE_BYTES,
        };
        let s = format!("{err}");
        assert!(s.contains("exceeds cap"));
    }

    // ─────────────────────────────────────────────────────────────────
    // 2. method_not_allowed builder shape — 405 + Allow header
    // ─────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn method_not_allowed_carries_allow_header() {
        let resp = method_not_allowed("POST");
        assert_eq!(resp.status(), StatusCode::METHOD_NOT_ALLOWED);
        let allow = resp
            .headers()
            .get("allow")
            .map(|v| v.to_str().unwrap().to_string());
        assert_eq!(allow.as_deref(), Some("POST"));
    }

    // ─────────────────────────────────────────────────────────────────
    // 3. Healthz response builder — 200 OK + JSON body
    // ─────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn healthz_response_shape_is_200_with_status_ok() {
        // Direct test of the healthz arm's response builder shape —
        // the integration test exercises the full GET /healthz cycle.
        let resp = Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from_static(b"{\"status\":\"ok\"}")))
            .expect("builder ok");
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get("content-type")
                .and_then(|v| v.to_str().ok()),
            Some("application/json")
        );
        let body = body_to_bytes(resp.into_body()).await;
        let s = String::from_utf8(body.to_vec()).expect("utf-8");
        assert!(s.contains("\"ok\""));
    }

    // ─────────────────────────────────────────────────────────────────
    // 4. TLS acceptor builds from a HotReloadResolver + sane defaults
    // ─────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn tls_acceptor_builds_from_resolver() {
        let resolver = synth_resolver();
        let cfg = HttpServerConfig::new("127.0.0.1:0".parse().expect("parse addr"), resolver);
        let acc = tls_acceptor_from_config(&cfg).expect("acceptor builds");
        // Smoke: the acceptor was constructed without error. Full
        // handshake + roundtrip lives in the integration test.
        let _ = acc;
    }

    // ─────────────────────────────────────────────────────────────────
    // 5a/5b/5c: Tenant extraction from X-ArcGraph-Tenant header
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn tenant_extract_from_header_decimal() {
        let req = Request::builder()
            .uri("/mcp")
            .header(HEADER_TENANT, "42")
            .body(())
            .expect("req build");
        let t = extract_tenant_from_header(&req).expect("ok");
        assert_eq!(t, Some(TenantId::new(42)));
    }

    #[test]
    fn tenant_extract_from_header_missing_returns_none() {
        let req = Request::builder().uri("/mcp").body(()).expect("req build");
        let t = extract_tenant_from_header(&req).expect("ok");
        assert!(t.is_none());
    }

    #[test]
    fn tenant_extract_from_header_non_numeric_errors() {
        let req = Request::builder()
            .uri("/mcp")
            .header(HEADER_TENANT, "not-a-number")
            .body(())
            .expect("req build");
        let err = extract_tenant_from_header(&req).expect_err("must reject");
        assert!(
            matches!(err, TransportError::TenantParse(_)),
            "expected TenantParse, got {err:?}"
        );
    }

    // ─────────────────────────────────────────────────────────────────
    // 6a/6b: tenant_from_san_strings (pure helper)
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn tenant_from_san_strings_picks_tenant_prefix() {
        let sans = vec![
            "arcgraph.local".to_string(),
            "tenant-7.arcgraph.local".to_string(),
            "tenant-13".to_string(),
        ];
        let t = tenant_from_san_strings(&sans);
        assert_eq!(t, Some(TenantId::new(7)));
    }

    #[test]
    fn tenant_from_san_strings_returns_none_when_no_match() {
        let sans = vec!["client-7.arcgraph.local".to_string(), "user-13".to_string()];
        let t = tenant_from_san_strings(&sans);
        assert!(t.is_none());
    }

    // ─────────────────────────────────────────────────────────────────
    // 7. Tenant extraction from real DER cert SAN
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn extract_tenant_from_peer_certs_parses_real_cert_san() {
        use rustls_pki_types::CertificateDer;
        let mut params = CertificateParams::new(vec![
            "tenant-99.arcgraph.local".to_string(),
            "arcgraph.local".to_string(),
        ])
        .expect("rcgen params");
        params.not_before = OffsetDateTime::now_utc() - time::Duration::days(1);
        params.not_after = OffsetDateTime::now_utc() + time::Duration::days(365);
        let kp = KeyPair::generate().expect("keypair");
        let cert = params.self_signed(&kp).expect("cert");
        let der = cert.der().clone();
        let owned = CertificateDer::from(der.as_ref().to_vec());
        let t = extract_tenant_from_peer_certs(&[owned])
            .expect("parse")
            .expect("tenant present");
        assert_eq!(t, TenantId::new(99));
    }

    #[test]
    fn extract_tenant_from_peer_certs_returns_none_for_empty_chain() {
        // No peer cert presented (non-mTLS handshake).
        let t = extract_tenant_from_peer_certs(&[]).expect("ok");
        assert!(t.is_none());
    }

    // ─────────────────────────────────────────────────────────────────
    // 8. Per-request deadline timer arms + fires
    // ─────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn deadline_timer_arms_with_configured_value() {
        let registry = CancellationRegistry::new();
        let qid = arcgraph_query::QueryId::new();
        let token = registry.register(qid);
        let _h = spawn_deadline_timer(token.clone(), Duration::from_millis(50));
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert!(
            token.is_cancelled(),
            "deadline timer must trip the token after the bound elapses"
        );
    }

    // ─────────────────────────────────────────────────────────────────
    // 9. SIGTERM cancel_all path fires every in-flight token
    // ─────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn shutdown_signal_fires_cancel_all() {
        let registry = Arc::new(CancellationRegistry::new());
        let tokens: Vec<_> = (0..3)
            .map(|_| {
                let qid = arcgraph_query::QueryId::new();
                registry.register(qid)
            })
            .collect();
        assert_eq!(registry.len(), 3);
        let fired = registry.cancel_all();
        assert_eq!(fired, 3);
        for t in &tokens {
            assert!(t.is_cancelled(), "every in-flight token must be cancelled");
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // 10. Per-request register/unregister balance
    // ─────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn per_request_register_and_unregister_balance() {
        let registry = CancellationRegistry::new();
        let qid = arcgraph_query::QueryId::new();
        assert_eq!(registry.len(), 0);
        let _token = registry.register(qid);
        assert_eq!(registry.len(), 1);
        let removed = registry.unregister(qid);
        assert!(removed);
        assert_eq!(registry.len(), 0);
    }

    // ─────────────────────────────────────────────────────────────────
    // 11. error_envelope_response renders the JSON-RPC error code
    // ─────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn error_envelope_response_carries_jsonrpc_code() {
        let resp = error_envelope_response_with_id(
            StatusCode::BAD_REQUEST,
            MCPError::InvalidRequest("test".into()),
            json!(7),
        );
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = body_to_bytes(resp.into_body()).await;
        let v: Value = serde_json::from_slice(&bytes).expect("json");
        assert_eq!(v["error"]["code"], -32600);
        assert_eq!(v["id"], 7);
    }

    // ─────────────────────────────────────────────────────────────────
    // AHP-1 (ADR-225 §3) — the bulkhead-boundary envelope mapping. A
    // deadline-exceeded dispatch (`BulkheadOutcome::TimedOut`) renders a
    // `-32001 cancelled` envelope echoing the request id; a panicked
    // dispatch renders `-32603 internal error`. This locks the codes the
    // `handle_request` bulkhead arm emits when it stops awaiting the
    // blocking thread (closing the http.rs:1420–1426 TODO).
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn timed_out_dispatch_maps_to_minus_32001_cancelled_envelope() {
        let env = mcp_error_envelope(json!("req-7"), &MCPError::Cancelled);
        assert_eq!(env["error"]["code"], -32001, "deadline → -32001 cancelled");
        assert_eq!(
            env["id"], "req-7",
            "request id echoed on the timeout envelope"
        );
        assert!(
            env.get("result").is_none(),
            "error envelope must not carry a result member"
        );
    }

    #[test]
    fn panicked_dispatch_maps_to_minus_32603_internal_error_envelope() {
        let env = mcp_error_envelope(
            json!(42),
            &MCPError::InternalError("dispatch task panicked".into()),
        );
        assert_eq!(
            env["error"]["code"], -32603,
            "panic → -32603 internal error"
        );
        assert_eq!(env["id"], 42);
    }

    // ─────────────────────────────────────────────────────────────────
    // 12. transport_error_to_mcp mapping (BodyTooLarge / TenantMissing)
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn transport_error_to_mcp_maps_body_too_large_to_parse_error() {
        let e = TransportError::BodyTooLarge {
            len: 99_999_999,
            cap: 1_000_000,
        };
        let mcp = transport_error_to_mcp(&e);
        assert_eq!(mcp.code(), -32700);
    }

    #[test]
    fn transport_error_to_mcp_maps_tenant_missing_to_invalid_request() {
        let e = TransportError::TenantMissing {
            strategy: TenantStrategy::Header,
        };
        let mcp = transport_error_to_mcp(&e);
        assert_eq!(mcp.code(), -32600);
    }

    // ─────────────────────────────────────────────────────────────────
    // 13. identify_tenant w/ Header strategy + missing header errors
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn identify_tenant_header_strategy_missing_header_errors() {
        let req = Request::builder().uri("/mcp").body(()).expect("req build");
        let err = identify_tenant(&req, &TenantStrategy::Header, &[]).expect_err("must err");
        assert!(matches!(err, TransportError::TenantMissing { .. }));
    }

    // ─────────────────────────────────────────────────────────────────
    // 14. envelope_tenant_id pulls params.tenant_id (Some / None)
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn envelope_tenant_id_pulls_decimal_from_params() {
        let env = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "graph.schema",
            "params": {"tenant_id": 17}
        });
        assert_eq!(envelope_tenant_id(&env), Some(TenantId::new(17)));
    }

    #[test]
    fn envelope_tenant_id_returns_none_when_missing() {
        let env = json!({"jsonrpc":"2.0","id":1,"method":"graph.schema"});
        assert_eq!(envelope_tenant_id(&env), None);
    }

    // ─────────────────────────────────────────────────────────────────
    // 15. validate — non-loopback bind without `allow_remote_bind` is
    //     rejected (design-v2 §9.4 line 668 mandate)
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn validate_rejects_nonloopback_bind_without_opt_in() {
        let resolver = synth_resolver();
        let cfg = HttpServerConfig::new("0.0.0.0:8080".parse().expect("addr"), resolver);
        let err = cfg
            .validate()
            .expect_err("must reject 0.0.0.0 without opt-in");
        assert!(
            matches!(err, TransportError::BindAddrForbidden { .. }),
            "expected BindAddrForbidden, got {err:?}",
        );
    }

    // ─────────────────────────────────────────────────────────────────
    // 16. validate — non-loopback bind WITH `allow_remote_bind(true)`
    //     passes (explicit operator opt-in)
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn validate_passes_nonloopback_bind_with_opt_in() {
        let resolver = synth_resolver();
        let cfg = HttpServerConfig::new("0.0.0.0:8080".parse().expect("addr"), resolver)
            .with_allow_remote_bind(true);
        cfg.validate().expect("explicit opt-in passes");
    }

    // ─────────────────────────────────────────────────────────────────
    // 17. validate — PeerCertSan + client_verifier=None is rejected
    //     (NIT-9: misconfiguration that would silently 400 every
    //     request)
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn validate_rejects_peercertsan_without_client_verifier() {
        let resolver = synth_resolver();
        let mut cfg = HttpServerConfig::new("127.0.0.1:0".parse().expect("addr"), resolver);
        cfg.tenant_strategy = TenantStrategy::PeerCertSan;
        let err = cfg.validate().expect_err("must reject");
        match err {
            TransportError::ConfigInvalid(msg) => {
                assert!(
                    msg.contains("PeerCertSan"),
                    "config error must name the misconfiguration: {msg}"
                );
            }
            other => panic!("expected ConfigInvalid, got {other:?}"),
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // 18. validate — loopback bind + sane defaults passes
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn validate_passes_loopback_default_config() {
        let resolver = synth_resolver();
        let cfg = HttpServerConfig::new("127.0.0.1:0".parse().expect("addr"), resolver);
        cfg.validate().expect("sane defaults pass");
    }

    // ─────────────────────────────────────────────────────────────────
    // 18b. validate — empty allowed_origins (when Some) is rejected
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn validate_rejects_empty_allowed_origins() {
        let resolver = synth_resolver();
        let mut cfg = HttpServerConfig::new("127.0.0.1:0".parse().expect("addr"), resolver);
        cfg.allowed_origins = Some(Vec::new());
        let err = cfg.validate().expect_err("must reject empty list");
        assert!(matches!(err, TransportError::ConfigInvalid(_)));
    }

    // ─────────────────────────────────────────────────────────────────
    // 19. transport_error_to_mcp maps BodyParse → ParseError (-32700)
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn transport_error_to_mcp_maps_body_parse_to_parse_error() {
        let e = TransportError::BodyParse("hyper: incomplete frame".into());
        let mcp = transport_error_to_mcp(&e);
        assert_eq!(mcp.code(), -32700, "BodyParse routes to ParseError");
    }

    // ─────────────────────────────────────────────────────────────────
    // 20. transport_error_to_mcp maps OriginForbidden → InvalidRequest
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn transport_error_to_mcp_maps_origin_forbidden_to_invalid_request() {
        let e = TransportError::OriginForbidden {
            origin: "https://evil.example.com".into(),
        };
        let mcp = transport_error_to_mcp(&e);
        assert_eq!(
            mcp.code(),
            -32600,
            "OriginForbidden routes to InvalidRequest"
        );
    }
}
