//! W13δ M5-01 / W14α M5-02b / W14β M5-06 + M5-07 / W14γ M5-08 + M5-12 / W14δ M5-13 —
//! MCP transport implementations.
//!
//! Hosts the stdio transport ([`stdio`] submodule), the HTTP/TLS
//! transport ([`http`] submodule), and the Bolt protocol scaffold
//! ([`bolt`] submodule). The dispatch surface ([`Dispatcher`]) is
//! shared across MCP transports so a tool only implements its logic
//! once; the Bolt transport has its own adapter trait
//! ([`bolt::BoltQueryHandler`]) since Bolt is not an MCP-tool surface
//! (per ADR-004 the 10-tool cap covers MCP tools only — Bolt is a
//! driver-compat transport).
//!
//! # W14γ (M5-08 + M5-12)
//!
//! - The dispatcher composes a third adapter: [`crate::tools::ingest::IngestProvider`]
//!   (M5-08, the first WRITE-side Tier-1 tool).
//! - The dispatcher composes an optional [`crate::rate_limit::RateLimiter`]
//!   (M5-12). When present, every request consults the per-(tenant,
//!   op_class) bucket BEFORE the tool body runs; exhaustion surfaces
//!   [`MCPError::RateLimited`] (-32007) with `retry_after_ms`.
//!
//! # W14δ (M5-13)
//!
//! - Adds the Bolt 5.0 protocol scaffold ([`bolt`] submodule) for
//!   Neo4j-driver compatibility. Independent of the MCP dispatcher;
//!   uses its own [`bolt::BoltQueryHandler`] adapter.
//!
//! # W16ζ (this slice — M5-11)
//!
//! - The dispatcher composes a sixth adapter:
//!   [`crate::tools::raw_query::RawQueryExecutor`] (M5-11, the first
//!   Tier-2 power-user surface).
//! - The dispatcher carries a [`crate::SessionScope`] field (stub at
//!   v1.0-α; M5-03 OAuth swaps in Bearer-token-driven derivation).
//!   The new `Dispatcher::with_session_scope` constructor is the
//!   fail-closed entry point per ADR-004 amendment-03 §D-1; the legacy
//!   `new` / `with_rate_limiter` constructors default to
//!   `SessionScope::Power` for backward-compat with W13δ / W14β /
//!   W14γ test fixtures (the M5-03 swap-in canonicalizes the
//!   fail-closed posture).
//! - Dispatch arm for `graph.raw_query` rejects non-power sessions
//!   with [`MCPError::Forbidden`] (-32008) BEFORE the executor body
//!   runs.
//!
//! # ADR provenance
//! - **ADR-004 §Decision** — six-tool MCP surface under a hard cap of
//!   ten.
//! - **design-v2 §9 (Agent-Native MCP Interface)** — transport
//!   layering: stdio for local, streamable-HTTP for remote (W14α
//!   M5-02b composition), Bolt for Neo4j-driver compat (W14δ M5-13).
//! - **design-v2 §9.4 (Transport and security)** — HTTPS-enforced,
//!   Origin allowlist, 127.0.0.1 bind, OAuth 2.1 (forward roadmap
//!   M5-03), Bearer-token scope enforcement.
//! - **design-v2 §16.3** — "Bolt protocol (openCypher driver
//!   compatibility)" listed as an M4-5 deliverable; W14δ M5-13
//!   delivers the v1.0-α scaffold.
//! - **ADR-038 amendment-03 §M5↔M4 contract surface** — the
//!   `QueryEngine` surface MCP tools + Bolt RUN handlers bind to.
//! - **ADR-038 amendment-03 §TIER-1 GAP A** — pinned `graph.ingest`
//!   as the v1.0 data-modification surface.

pub mod bolt;
pub mod bulkhead;
pub mod http;
pub mod metrics;
pub mod stdio;

use std::sync::Arc;

use arcgraph_core::TenantId;
use arcgraph_query::CancellationToken;
use serde_json::Value;

use crate::error::MCPError;
use crate::jsonrpc::{
    JSONRPC_VERSION, JsonRpcErrorResponse, JsonRpcRequest, JsonRpcResponse, decode_request,
    id_or_null,
};
use crate::rate_limit::{OpClass, RateLimiter};
use crate::scope::SessionScope;
use crate::tools::explore::{ExploreRequest, NeighborhoodExplorer, explore_tool};
use crate::tools::ingest::{IngestProvider, IngestRequest, ingest_tool};
use crate::tools::inspect::{InspectRequest, NodeInspector, inspect_tool};
use crate::tools::raw_query::{RawQueryExecutor, RawQueryRequest, raw_query_tool};
use crate::tools::schema::{SchemaProvider, SchemaRequest, schema_tool};
use crate::tools::search::{HybridSearcher, SearchRequest, search_tool};

/// Method-name slug for the `graph.schema` tool.
pub const METHOD_GRAPH_SCHEMA: &str = "graph.schema";

/// Method-name slug for the `graph.inspect` tool.
pub const METHOD_GRAPH_INSPECT: &str = "graph.inspect";

/// Method-name slug for the `graph.explore` tool (W14β M5-06).
pub const METHOD_GRAPH_EXPLORE: &str = "graph.explore";

/// Method-name slug for the `graph.search` tool (W14β M5-07).
pub const METHOD_GRAPH_SEARCH: &str = "graph.search";

/// Method-name slug for the `graph.ingest` tool (W14γ M5-08).
pub const METHOD_GRAPH_INGEST: &str = "graph.ingest";

/// Method-name slug for the `graph.raw_query` Tier-2 tool (W16ζ M5-11).
pub const METHOD_GRAPH_RAW_QUERY: &str = "graph.raw_query";

/// The complete catalog of wire-callable JSON-RPC method slugs the
/// [`Dispatcher`] recognizes (the `match req.method.as_str()` arms in
/// [`Dispatcher::dispatch_inner`]). Single source of truth for the #901
/// method-not-found near-miss surface: it backs BOTH the ranked
/// `did_you_mean` hints and the `available_methods` catalog emitted in the
/// -32601 error `data`. MUST stay in sync with the dispatch arms — the
/// `known_methods_catalog_is_canonical` unit test pins the invariant.
///
/// An agent that typo'd toward any recognized slug gets a deterministic
/// recovery hint regardless of per-deployment provider wiring.
pub(crate) const KNOWN_METHODS: &[&str] = &[
    METHOD_GRAPH_SCHEMA,
    METHOD_GRAPH_INSPECT,
    METHOD_GRAPH_EXPLORE,
    METHOD_GRAPH_SEARCH,
    METHOD_GRAPH_INGEST,
    METHOD_GRAPH_RAW_QUERY,
];

/// ADR-004 §Decision — the MCP tool-surface hard cap: at most **10 active
/// tools** across v1.0–v1.2. Ratified against doc drift by
/// ADR-004-amendment-01; the load-bearing evidence is the MCP-Bench /
/// WildToolBench / BFCL v3 "~10-tool accuracy cliff" — do NOT lift the cap
/// without re-benchmarking. Runtime-enforced by
/// `enforce_adr_004_tool_cap` at every dispatcher wiring point (#1294).
pub const ADR_004_TOOL_CAP: usize = 10;

/// #1294 — runtime enforcement of the ADR-004 10-tool cap.
///
/// Before this guard the cap was a docs-and-test commitment only: nothing
/// at runtime stopped a dispatcher from serving an over-cap catalog.
/// `wired` is a wire-callable catalog in the shape
/// [`Dispatcher::wired_methods`] returns. It exists so an 11th active tool
/// cannot ship silently.
///
/// Cost budget: one O(catalog ≤ 13) pass of short-string compares per call
/// (~tens of ns) — negligible against the JSON-RPC envelope work on every
/// path that materializes the catalog.
///
/// # Panics
///
/// Panics when the active count exceeds [`ADR_004_TOOL_CAP`], naming the
/// offending count + slugs. Wiring-time invocation (constructors + every
/// provider `with_*` builder) makes an over-cap misconfiguration reject AT
/// STARTUP rather than silently serve an 11th tool — mirroring the
/// `#[serde(deny_unknown_fields)]` config strict-mode philosophy.
fn enforce_adr_004_tool_cap(wired: &[&'static str]) {
    assert!(
        wired.len() <= ADR_004_TOOL_CAP,
        "ADR-004 10-tool cap violated: {count} active MCP tools wired \
         (cap = {ADR_004_TOOL_CAP}): {wired:?}. Adding an MCP tool requires \
         a new ADR; do not lift the cap without re-benchmarking \
         the ~10-tool accuracy cliff (ADR-004 §Evidence).",
        count = wired.len(),
    );
}

/// MCP protocol revisions this server understands, newest first. During
/// `initialize` we echo the client's requested revision when it is in this
/// list; otherwise we select the newest supported revision.
pub const MCP_SUPPORTED_PROTOCOL_VERSIONS: &[&str] = &["2025-06-18", "2025-03-26", "2024-11-05"];

const MCP_LATEST_PROTOCOL_VERSION: &str = MCP_SUPPORTED_PROTOCOL_VERSIONS[0];

const METHOD_MCP_INITIALIZE: &str = "initialize";
const METHOD_MCP_TOOLS_LIST: &str = "tools/list";
const METHOD_MCP_TOOLS_CALL: &str = "tools/call";
const MCP_NOTIFICATION_PREFIX: &str = "notifications/";

/// Dependency-free Levenshtein edit distance (two-row DP, O(|a|·|b|) time,
/// O(|b|) space). Per the Apache-2.0 Prime Directive (workspace Apache-2.0 licensing policy) a
/// 12-slug catalog never justifies a crate; a ~20-line classic suffices.
/// Operates on Unicode scalar values — exact for the ASCII method slugs and
/// safe for arbitrary client input.
///
/// Budget: invoked only at error-render time (never the success hot path) —
/// 12 catalog entries × distance over ≤~16-char slugs is sub-microsecond,
/// with no allocation beyond the two row buffers.
fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr: Vec<usize> = vec![0; b.len() + 1];
    for (i, &ca) in a.iter().enumerate() {
        curr[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let cost = usize::from(ca != cb);
            curr[j + 1] = (prev[j + 1] + 1) // deletion
                .min(curr[j] + 1) // insertion
                .min(prev[j] + cost); // substitution
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()]
}

/// Normalize a method slug for near-miss comparison: lowercase, drop a
/// leading `graph.` namespace, and strip `_` / `.` separators. Collapses the
/// three dominant LLM near-miss classes (#901) toward their canonical target
/// so edit distance ranks the intended tool first:
/// - missing namespace — `schema` → `graph.schema`
/// - camelCase — `graph.rawQuery` → `graph.raw_query`
/// - snake/typo drift — `graph.explor` → `graph.explore`
fn normalize_method(m: &str) -> String {
    let lower = m.to_ascii_lowercase();
    let stem = lower.strip_prefix("graph.").unwrap_or(&lower);
    stem.chars().filter(|c| *c != '_' && *c != '.').collect()
}

/// Up to 3 [`KNOWN_METHODS`] slugs that are near-misses for `method`, ranked
/// by normalized edit distance then name (#901). Empty when nothing is close
/// enough — the `available_methods` catalog in the -32601 `data` is the
/// fallback recovery path. Never suggests `method` itself, so a recognized-
/// but-unwired slug doesn't get a useless "did you mean &lt;the same name&gt;?".
pub(crate) fn nearest_methods(method: &str) -> Vec<String> {
    let norm_in = normalize_method(method);
    let mut scored: Vec<(usize, &'static str)> = KNOWN_METHODS
        .iter()
        .filter_map(|&m| {
            if m == method {
                return None; // never echo the caller's own slug
            }
            let norm_m = normalize_method(m);
            let dist = levenshtein(&norm_in, &norm_m);
            // Length-relative gate keyed to the SHORTER normalized slug, so a
            // missing-suffix near-miss (`inspectnode` → `inspect`) still
            // qualifies while unrelated input (`foobar`) is rejected.
            let gate = norm_in.len().min(norm_m.len()) / 2 + 1;
            (dist <= gate).then_some((dist, m))
        })
        .collect();
    scored.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(b.1)));
    scored
        .into_iter()
        .take(3)
        .map(|(_, m)| m.to_string())
        .collect()
}

/// Classify a JSON-RPC method name into its [`OpClass`].
///
/// Mirrors the W14γ M5-12 rate-limit classification: the only write-
/// side method at v1.0-alpha is `graph.ingest`; every other method
/// (including `graph.raw_query` and unknown / typo'd names) is
/// read-class. This is the same classification the [`Dispatcher`] uses
/// for rate-limit-bucket selection — exposed as a free function so the
/// W15γ M6-06 Prometheus exporter can route `arcgraph_read_latency_ms`
/// / `arcgraph_write_latency_ms` observations to the same buckets.
///
/// W16ζ M5-11 classification rationale: `graph.raw_query` is bound to
/// the read bucket because ArcQL v1.0 is read-only (CREATE / MERGE /
/// DELETE are deferred to v1.1+ per ADR-006 amendment-01 §D-2). When
/// write-clause ArcQL lands (v1.1+), this classifier MUST be amended
/// to inspect the query AST and bucket write-clause raw queries to
/// `OpClass::Write`.
#[must_use]
pub fn op_class_for_method(method: &str) -> OpClass {
    match method {
        // W14γ M5-08 — `graph.ingest` is the v1.0-α write surface.
        METHOD_GRAPH_INGEST => OpClass::Write,
        // All other methods (including `graph.raw_query` and unknown /
        // typo'd names) are read-class.
        _ => OpClass::Read,
    }
}

/// Per-session MCP dispatcher.
///
/// Composes the adapter impls for the Tier-1 tools landed so far
/// ([`SchemaProvider`] M5-04, [`NodeInspector`] M5-05,
/// [`NeighborhoodExplorer`] W14β M5-06, [`HybridSearcher`] W14β M5-07,
/// [`IngestProvider`] W14γ M5-08), the session's bound tenant, and an
/// optional [`RateLimiter`] (W14γ M5-12). Routes JSON-RPC requests to
/// the correct tool.
///
/// `Arc`-ed shared state lets a future M5-02 streamable-HTTP server
/// share one dispatcher across many connections; v1.0-alpha stdio
/// uses a single dispatcher pinned to the parent process's tenant.
///
/// # Cancellation
///
/// Each dispatched call mints a fresh [`CancellationToken`] (per
/// JSON-RPC request) and threads it into the tools' cancellation-
/// aware adapters ([`NeighborhoodExplorer::explore`] /
/// [`HybridSearcher::search`]). v1.0-alpha stdio dispatch is sync;
/// the token is bound on the spawning task so a future async
/// transport (M5-02) can trip it from the SIGTERM drain. The schema /
/// inspect tools do not yet thread a token (their adapter shapes
/// predate W14β); a future slice can plumb tokens through those
/// surfaces too.
///
/// # Adding a new tool
///
/// The dispatcher routes the five Tier-1 tools and the power-scoped
/// `graph.raw_query` tool. Catalog changes extend
/// [`Dispatcher::dispatch`] and must stay under the ADR-004 hard cap.
pub struct Dispatcher<S, I, E, H, G, R>
where
    S: SchemaProvider + 'static,
    I: NodeInspector + 'static,
    E: NeighborhoodExplorer + 'static,
    H: HybridSearcher + 'static,
    G: IngestProvider + 'static,
    R: RawQueryExecutor + 'static,
{
    /// The session's bound tenant. Cross-tenant requests reject as
    /// [`MCPError::Unauthorized`] before any tool body runs.
    pub session_tenant: TenantId,
    /// The session's bound scope (W16ζ M5-11 per ADR-004 amendment-03
    /// §D-1). Stub at v1.0-α; the M5-03 OAuth slice swaps in
    /// Bearer-token-derived derivation. Power-tier tools
    /// (`graph.raw_query`) require [`SessionScope::Power`]; otherwise
    /// they reject as [`MCPError::Forbidden`] (-32008) BEFORE the
    /// tool body runs. Defaults to [`SessionScope::Power`] for the
    /// legacy `new` / `with_rate_limiter` constructors (backward-
    /// compat with W13δ / W14β / W14γ test fixtures);
    /// `with_session_scope` is the canonical fail-closed entry point.
    pub session_scope: SessionScope,
    /// Schema-tool adapter.
    pub schema_provider: Arc<S>,
    /// Inspect-tool adapter.
    pub node_inspector: Arc<I>,
    /// Explore-tool adapter (W14β M5-06).
    pub neighborhood_explorer: Arc<E>,
    /// Search-tool adapter (W14β M5-07).
    pub hybrid_searcher: Arc<H>,
    /// Ingest-tool adapter (W14γ M5-08).
    pub ingest_provider: Arc<G>,
    /// Raw-query-tool adapter (W16ζ M5-11). First Tier-2 power-user
    /// adapter in the catalog.
    pub raw_query_executor: Arc<R>,
    /// Optional per-tenant rate-limiter (W14γ M5-12). When `None`,
    /// rate-limit gating is disabled (useful for embedded /
    /// single-tenant deployments). When `Some`, every request
    /// consults the bucket BEFORE the tool body.
    pub rate_limiter: Option<RateLimiter>,
}

impl<S, I, E, H, G, R> Dispatcher<S, I, E, H, G, R>
where
    S: SchemaProvider + 'static,
    I: NodeInspector + 'static,
    E: NeighborhoodExplorer + 'static,
    H: HybridSearcher + 'static,
    G: IngestProvider + 'static,
    R: RawQueryExecutor + 'static,
{
    /// Construct a dispatcher bound to a tenant + adapter set.
    /// Rate-limiting is disabled.
    ///
    /// **Backward-compat default**: this constructor seeds
    /// [`SessionScope::Power`] so the W13δ / W14β / W14γ test fixtures
    /// (and any pre-W16ζ deployment) retain unfettered access to
    /// `graph.raw_query`. Per ADR-004 amendment-03 §D-1 the canonical
    /// fail-closed entry point is [`Self::with_session_scope`]; the
    /// M5-03 OAuth slice canonicalizes the fail-closed posture by
    /// replacing this default with a Bearer-token-derived derivation
    /// across all constructors.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        session_tenant: TenantId,
        schema_provider: Arc<S>,
        node_inspector: Arc<I>,
        neighborhood_explorer: Arc<E>,
        hybrid_searcher: Arc<H>,
        ingest_provider: Arc<G>,
        raw_query_executor: Arc<R>,
    ) -> Self {
        let dispatcher = Self {
            session_tenant,
            session_scope: SessionScope::Power,
            schema_provider,
            node_inspector,
            neighborhood_explorer,
            hybrid_searcher,
            ingest_provider,
            raw_query_executor,
            rate_limiter: None,
        };
        // #1294 — ADR-004 cap guard at construction.
        dispatcher.assert_within_adr_004_cap();
        dispatcher
    }

    /// Construct a dispatcher with rate-limiting enabled.
    ///
    /// Same `SessionScope::Power` backward-compat default as
    /// [`Self::new`]; the canonical fail-closed entry point is
    /// [`Self::with_session_scope`] per ADR-004 amendment-03 §D-1.
    ///
    /// The argument count exceeds clippy's default limit because the
    /// Tier-1 catalog already requires 5 adapter generics, plus the
    /// new Tier-2 raw-query adapter (W16ζ M5-11), plus tenant and
    /// limiter. A builder would obscure the per-adapter wiring at the
    /// only two call sites (CLI binary plus this crate's tests), so
    /// we keep the constructor flat and `#[allow]` the lint locally.
    #[allow(clippy::too_many_arguments)]
    pub fn with_rate_limiter(
        session_tenant: TenantId,
        schema_provider: Arc<S>,
        node_inspector: Arc<I>,
        neighborhood_explorer: Arc<E>,
        hybrid_searcher: Arc<H>,
        ingest_provider: Arc<G>,
        raw_query_executor: Arc<R>,
        rate_limiter: RateLimiter,
    ) -> Self {
        let dispatcher = Self {
            session_tenant,
            session_scope: SessionScope::Power,
            schema_provider,
            node_inspector,
            neighborhood_explorer,
            hybrid_searcher,
            ingest_provider,
            raw_query_executor,
            rate_limiter: Some(rate_limiter),
        };
        // #1294 — ADR-004 cap guard at construction.
        dispatcher.assert_within_adr_004_cap();
        dispatcher
    }

    /// Construct a dispatcher with an explicit session scope (W16ζ
    /// M5-11 per ADR-004 amendment-03 §D-1).
    ///
    /// This is the **fail-closed** entry point: callers MUST pass a
    /// scope explicitly. A non-power scope rejects `graph.raw_query`
    /// with [`MCPError::Forbidden`] (-32008) BEFORE the executor body
    /// runs. Pair with [`Self::with_session_scope_and_rate_limiter`]
    /// when both scope-gating + rate-limiting are required (production
    /// shape).
    ///
    /// HTTP derives [`SessionScope`] per request from Bearer-token claims and
    /// calls [`Self::dispatch_with_scope`]; Bolt carries its authenticated
    /// scope in `BoltSessionAuth`. Stdio/embedded composition roots use this
    /// dispatcher-bound scope.
    #[allow(clippy::too_many_arguments)]
    pub fn with_session_scope(
        session_tenant: TenantId,
        session_scope: SessionScope,
        schema_provider: Arc<S>,
        node_inspector: Arc<I>,
        neighborhood_explorer: Arc<E>,
        hybrid_searcher: Arc<H>,
        ingest_provider: Arc<G>,
        raw_query_executor: Arc<R>,
    ) -> Self {
        let dispatcher = Self {
            session_tenant,
            session_scope,
            schema_provider,
            node_inspector,
            neighborhood_explorer,
            hybrid_searcher,
            ingest_provider,
            raw_query_executor,
            rate_limiter: None,
        };
        // #1294 — ADR-004 cap guard at construction.
        dispatcher.assert_within_adr_004_cap();
        dispatcher
    }

    /// Construct a dispatcher with both an explicit session scope AND
    /// rate-limiting (W16ζ M5-11). The production shape for v1.0-α
    /// deployments running the full Tier-1 + Tier-2 surface.
    #[allow(clippy::too_many_arguments)]
    pub fn with_session_scope_and_rate_limiter(
        session_tenant: TenantId,
        session_scope: SessionScope,
        schema_provider: Arc<S>,
        node_inspector: Arc<I>,
        neighborhood_explorer: Arc<E>,
        hybrid_searcher: Arc<H>,
        ingest_provider: Arc<G>,
        raw_query_executor: Arc<R>,
        rate_limiter: RateLimiter,
    ) -> Self {
        let dispatcher = Self {
            session_tenant,
            session_scope,
            schema_provider,
            node_inspector,
            neighborhood_explorer,
            hybrid_searcher,
            ingest_provider,
            raw_query_executor,
            rate_limiter: Some(rate_limiter),
        };
        // #1294 — ADR-004 cap guard at construction.
        dispatcher.assert_within_adr_004_cap();
        dispatcher
    }

    /// Classify a method name into the read/write op class used to
    /// key the rate-limit bucket. Unknown methods default to
    /// `Read` — they'll fall through to `MethodNotFound` below the
    /// rate-limit gate, which is the correct order: an unknown
    /// method should still count against a tenant's total budget so
    /// a tenant can't probe the server with method-name typos for
    /// free.
    fn op_class_for_method(method: &str) -> OpClass {
        op_class_for_method(method)
    }

    /// Methods actually callable on this dispatcher instance. The always-wired
    /// tools are listed first; provider-backed tools appear only when their
    /// provider state is present.
    fn wired_methods(&self) -> Vec<&'static str> {
        let methods = vec![
            METHOD_GRAPH_SCHEMA,
            METHOD_GRAPH_INSPECT,
            METHOD_GRAPH_EXPLORE,
            METHOD_GRAPH_SEARCH,
            METHOD_GRAPH_INGEST,
            METHOD_GRAPH_RAW_QUERY,
        ];
        // #1294 — ADR-004 cap guard: this Vec IS the served tool catalog
        // (tools/list, tools/call gating, -32601 catalog rendering), so an
        // over-cap catalog can never escape this fn. Constructors + the
        // provider `with_*` builders also run the guard so a misconfigured
        // deployment rejects at STARTUP, not on its first request.
        enforce_adr_004_tool_cap(&methods);
        methods
    }

    /// #1294 — run the ADR-004 cap guard against the current catalog.
    /// Called at every wiring point (constructors + provider `with_*`
    /// builders) so an over-cap catalog panics at STARTUP (construction),
    /// not on the first `tools/list`. [`Self::wired_methods`] enforces
    /// internally; materializing the catalog here triggers it.
    fn assert_within_adr_004_cap(&self) {
        let _ = self.wired_methods();
    }

    /// Dispatch a JSON-RPC request envelope and return the
    /// rendered response envelope (success or error). Notification
    /// requests (no `id`) return `Ok(None)` — the caller skips
    /// writing a response.
    ///
    /// # Tracing
    ///
    /// Per ADR-038 amendment-03 §TIER-2-c the dispatcher opens a
    /// per-request span tagged with `request_id`, `method`, and
    /// `tenant_id`. The span context propagates into tool bodies
    /// via the structured-fields mechanism on `tracing`.
    pub fn dispatch(&self, req: JsonRpcRequest) -> Option<Value> {
        self.dispatch_with_scope(req, self.session_scope)
    }

    /// Dispatch with a transport-authenticated request scope.
    ///
    /// Connectionless transports such as HTTPS derive authorization on each
    /// request, so they must not inherit the composition root's Power default.
    /// Stdio and embedded callers use [`Self::dispatch`], which preserves the
    /// dispatcher-bound scope.
    pub fn dispatch_with_scope(
        &self,
        req: JsonRpcRequest,
        session_scope: SessionScope,
    ) -> Option<Value> {
        let request_id = req.id.clone();
        let method = req.method.clone();
        let span = tracing::info_span!(
            "mcp_request",
            request_id = ?request_id,
            method = %method,
            tenant_id = self.session_tenant.raw(),
            session_scope = session_scope.slug(),
        );
        let _g = span.enter();

        // Notifications (no `id`) — fire-and-forget per JSON-RPC §4.1.
        let response_id = match request_id.clone() {
            Some(v) => v,
            None => {
                let _ = self.dispatch_inner(&req, session_scope);
                return None;
            }
        };

        let result = self.dispatch_inner(&req, session_scope);
        let envelope = match result {
            Ok(v) => serde_json::to_value(JsonRpcResponse::success(response_id, v)),
            Err(e) => {
                tracing::warn!(
                    target: "arcgraph_mcp::dispatcher",
                    code = e.code(),
                    error = %e,
                    "MCP request error",
                );
                serde_json::to_value(JsonRpcErrorResponse::from_mcp(response_id, &e))
            }
        };
        let envelope = envelope.unwrap_or_else(|e| {
            // Fallback: if even the error envelope fails to
            // serialize, emit a minimal -32603 internal error.
            serde_json::to_value(JsonRpcErrorResponse::from_mcp(
                id_or_null(req.id),
                &MCPError::InternalError(format!("response envelope serialize: {e}")),
            ))
            .unwrap_or(Value::Null)
        });
        Some(envelope)
    }

    fn dispatch_inner(
        &self,
        req: &JsonRpcRequest,
        session_scope: SessionScope,
    ) -> Result<Value, MCPError> {
        // W14γ M5-12: rate-limit gate. Consulted BEFORE the tool body
        // so an exhausted tenant can't drive substrate work via a
        // request that would have failed validation downstream. The
        // gate is defense-in-depth — production deployments without
        // a configured limiter still get the cross-tenant guard.
        if let Some(limiter) = self.rate_limiter.as_ref() {
            let class = Self::op_class_for_method(&req.method);
            if let Err(e) = limiter.try_consume(self.session_tenant, class) {
                return Err(MCPError::from(e));
            }
        }

        match req.method.as_str() {
            METHOD_GRAPH_SCHEMA => {
                let params: SchemaRequest = serde_json::from_value(req.params.clone())
                    .map_err(|e| MCPError::InvalidParams(format!("graph.schema: {e}")))?;
                schema_tool(self.schema_provider.as_ref(), self.session_tenant, params)
            }
            METHOD_GRAPH_INSPECT => {
                let params: InspectRequest = serde_json::from_value(req.params.clone())
                    .map_err(|e| MCPError::InvalidParams(format!("graph.inspect: {e}")))?;
                inspect_tool(
                    self.node_inspector.as_ref(),
                    self.session_tenant,
                    session_scope,
                    params,
                )
            }
            METHOD_GRAPH_EXPLORE => {
                let params: ExploreRequest = serde_json::from_value(req.params.clone())
                    .map_err(|e| MCPError::InvalidParams(format!("graph.explore: {e}")))?;
                // v1.0-alpha: mint a fresh token per request. A
                // future M5-02 transport will plumb in a session-
                // scoped token so the SIGTERM drain can trip
                // in-flight requests.
                let token = CancellationToken::new();
                explore_tool(
                    self.neighborhood_explorer.as_ref(),
                    self.session_tenant,
                    // #1488 — mirror graph.search's #1293 scope
                    // threading so a non-power session omitting
                    // `principal` fails CLOSED (-32008), never through
                    // the unfiltered SYSTEM-TRUSTED path.
                    session_scope,
                    &token,
                    params,
                )
            }
            METHOD_GRAPH_SEARCH => {
                let params: SearchRequest = serde_json::from_value(req.params.clone())
                    .map_err(|e| MCPError::InvalidParams(format!("graph.search: {e}")))?;
                let token = CancellationToken::new();
                // #1293 — thread the transport-authenticated scope so a non-power
                // session omitting `principal` fails CLOSED (-32008)
                // instead of running the unfiltered SYSTEM-TRUSTED
                // path. Mirrors the `graph.raw_query` scope-threading
                // shape.
                search_tool(
                    self.hybrid_searcher.as_ref(),
                    self.session_tenant,
                    session_scope,
                    &token,
                    params,
                )
            }
            METHOD_GRAPH_INGEST => {
                let params: IngestRequest = serde_json::from_value(req.params.clone())
                    .map_err(|e| MCPError::InvalidParams(format!("graph.ingest: {e}")))?;
                ingest_tool(self.ingest_provider.as_ref(), self.session_tenant, params)
            }
            METHOD_GRAPH_RAW_QUERY => {
                // W16ζ M5-11: Tier-2 power-user surface. The scope
                // check + cross-tenant guard + caps run INSIDE
                // raw_query_tool (cross-tenant before scope to avoid
                // leaking scope info to a cross-tenant probe). The
                // rate-limit gate already ran above (read class per
                // op_class_for_method — see W16ζ M5-11 classification
                // rationale on that fn's doc-comment).
                let params: RawQueryRequest = serde_json::from_value(req.params.clone())
                    .map_err(|e| MCPError::InvalidParams(format!("graph.raw_query: {e}")))?;
                let token = CancellationToken::new();
                raw_query_tool(
                    self.raw_query_executor.as_ref(),
                    self.session_tenant,
                    session_scope,
                    &token,
                    params,
                )
            }
            other => Err(MCPError::MethodNotFound(other.to_string())),
        }
    }
}

fn mcp_success(id: Value, result: Value) -> Option<Value> {
    Some(serde_json::to_value(JsonRpcResponse::success(id, result)).unwrap_or(Value::Null))
}

fn mcp_error(id: Value, err: MCPError) -> Option<Value> {
    Some(serde_json::to_value(JsonRpcErrorResponse::from_mcp(id, &err)).unwrap_or(Value::Null))
}

fn negotiate_mcp_protocol_version(params: &Value) -> &'static str {
    let requested = params
        .get("protocolVersion")
        .and_then(Value::as_str)
        .unwrap_or(MCP_LATEST_PROTOCOL_VERSION);
    MCP_SUPPORTED_PROTOCOL_VERSIONS
        .iter()
        .copied()
        .find(|version| *version == requested)
        .unwrap_or(MCP_LATEST_PROTOCOL_VERSION)
}

fn initialize_result(params: &Value) -> Value {
    serde_json::json!({
        "protocolVersion": negotiate_mcp_protocol_version(params),
        "capabilities": {
            "tools": {
                "listChanged": false
            }
        },
        "serverInfo": {
            "name": "arcgraph",
            "version": env!("CARGO_PKG_VERSION")
        }
    })
}

fn tool_descriptor(method: &str) -> Option<Value> {
    let (description, input_schema) = match method {
        METHOD_GRAPH_SCHEMA => (
            "Return the per-tenant graph schema: labels, relationship types, and property/index metadata.",
            schema_input_schema(),
        ),
        METHOD_GRAPH_INSPECT => (
            "Return one node's property bag and 1-hop neighborhood.",
            inspect_input_schema(),
        ),
        METHOD_GRAPH_EXPLORE => (
            "Return an N-hop neighborhood graph rooted at a seed node, with bounded depth and output size.",
            explore_input_schema(),
        ),
        METHOD_GRAPH_SEARCH => (
            "Run hybrid BM25/vector retrieval for a tenant and return ranked hits with optional label filtering.",
            search_input_schema(),
        ),
        METHOD_GRAPH_INGEST => (
            "Batch-ingest node and relationship records for a tenant, returning per-record outcomes and a commit LSN when records commit.",
            ingest_input_schema(),
        ),
        METHOD_GRAPH_RAW_QUERY => (
            "Execute an ArcQL/openCypher-subset query for power-scope sessions, with row caps and structured query faults. Pass explain:true to return the query plan (parse/bind/cost-only, never executed) instead of running the query.",
            raw_query_input_schema(),
        ),
        _ => return None,
    };
    Some(serde_json::json!({
        "name": method,
        "description": description,
        "inputSchema": input_schema,
    }))
}

fn object_schema(properties: Value, required: &[&str]) -> Value {
    serde_json::json!({
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": false
    })
}

fn format_schema() -> Value {
    serde_json::json!({
        "type": "string",
        "enum": ["toon", "yaml", "json"]
    })
}

fn tenant_schema() -> Value {
    serde_json::json!({"type": "integer", "minimum": 0})
}

fn string_array_schema() -> Value {
    serde_json::json!({"type": "array", "items": {"type": "string"}})
}

fn schema_input_schema() -> Value {
    object_schema(
        serde_json::json!({
            "tenant_id": tenant_schema(),
            "format": format_schema()
        }),
        &["tenant_id"],
    )
}

fn inspect_input_schema() -> Value {
    object_schema(
        serde_json::json!({
            "tenant_id": tenant_schema(),
            "node_id": {"type": "integer", "minimum": 0},
            "format": format_schema(),
            "principal": {"type": "string", "minLength": 1}
        }),
        &["tenant_id", "node_id"],
    )
}

fn explore_input_schema() -> Value {
    object_schema(
        serde_json::json!({
            "tenant_id": tenant_schema(),
            "seed": {"type": "integer", "minimum": 0},
            "max_depth": {"type": "integer", "minimum": 0},
            "max_results": {"type": "integer", "minimum": 1},
            "rel_types": string_array_schema(),
            "direction": {"type": "string", "enum": ["out", "in", "both"]},
            "format": format_schema(),
            "principal": {"type": "string", "minLength": 1}
        }),
        &["tenant_id", "seed"],
    )
}

fn search_input_schema() -> Value {
    object_schema(
        serde_json::json!({
            "tenant_id": tenant_schema(),
            "query": {"type": "string"},
            "query_vec": {"type": "array", "items": {"type": "number"}},
            "k": {"type": "integer", "minimum": 1},
            "label_filter": string_array_schema(),
            "ef_search": {"type": "integer", "minimum": 1},
            "format": format_schema(),
            "principal": {"type": "string", "minLength": 1}
        }),
        &["tenant_id"],
    )
}

fn ingest_input_schema() -> Value {
    object_schema(
        serde_json::json!({
            "tenant_id": tenant_schema(),
            "nodes": {
                "type": "array",
                "items": object_schema(
                    serde_json::json!({
                        "external_id": {"type": "string"},
                        "label": {"type": "string"},
                        "properties": {"type": "object"}
                    }),
                    &["label"],
                )
            },
            "relationships": {
                "type": "array",
                "items": object_schema(
                    serde_json::json!({
                        "external_id": {"type": "string"},
                        "from_external_id": {"type": "string"},
                        "to_external_id": {"type": "string"},
                        "rel_type": {"type": "string"},
                        "properties": {"type": "object"}
                    }),
                    &["from_external_id", "to_external_id", "rel_type"],
                )
            },
            // #1181 (MUST-CON-07, ADR-212 §D-4 Seam-1): per-doc read-ACL
            // grants applied via PermissionIndex::apply_doc_acl AFTER the
            // records commit.
            // `read_principals` null/absent ⇒ UNCLASSIFIED (skipped).
            "acl_grants": {
                "type": "array",
                "items": object_schema(
                    serde_json::json!({
                        "external_id": {"type": "string"},
                        "read_principals": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    }),
                    &["external_id"],
                )
            },
            "format": format_schema()
        }),
        &["tenant_id"],
    )
}

fn raw_query_input_schema() -> Value {
    object_schema(
        serde_json::json!({
            "tenant_id": tenant_schema(),
            "query": {"type": "string"},
            "max_rows": {"type": "integer", "minimum": 1},
            "format": format_schema(),
            "explain": {
                "type": "boolean",
                "description": "when true, return the query plan instead of executing"
            }
        }),
        &["tenant_id", "query"],
    )
}

fn handle_mcp_tools_call<S, I, E, H, G, R>(
    dispatcher: &Dispatcher<S, I, E, H, G, R>,
    req: JsonRpcRequest,
    response_id: Value,
    session_scope: SessionScope,
) -> Option<Value>
where
    S: SchemaProvider + 'static,
    I: NodeInspector + 'static,
    E: NeighborhoodExplorer + 'static,
    H: HybridSearcher + 'static,
    G: IngestProvider + 'static,
    R: RawQueryExecutor + 'static,
{
    let Some(name) = req.params.get("name").and_then(Value::as_str) else {
        return mcp_error(
            response_id,
            MCPError::InvalidParams("tools/call: missing string `name`".into()),
        );
    };
    if !dispatcher.wired_methods().contains(&name) {
        return mcp_error(
            response_id,
            MCPError::InvalidParams(format!("tools/call: unknown tool {name:?}")),
        );
    }
    let arguments = req
        .params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    let synthesized = JsonRpcRequest {
        jsonrpc: JSONRPC_VERSION.into(),
        method: name.to_string(),
        params: arguments,
        id: Some(response_id.clone()),
    };
    let Some(tool_response) = dispatcher.dispatch_with_scope(synthesized, session_scope) else {
        return mcp_error(
            response_id,
            MCPError::InternalError("tools/call synthesized request produced no response".into()),
        );
    };
    if let Some(result) = tool_response.get("result") {
        return mcp_success(
            response_id,
            serde_json::json!({
                "content": [{
                    "type": "text",
                    "text": serde_json::to_string(result).unwrap_or_else(|_| "null".into())
                }],
                "isError": false
            }),
        );
    }
    if let Some(error) = tool_response.get("error") {
        return mcp_success(
            response_id,
            serde_json::json!({
                "content": [{
                    "type": "text",
                    "text": serde_json::to_string(error).unwrap_or_else(|_| "{}".into())
                }],
                "isError": true
            }),
        );
    }
    mcp_error(
        response_id,
        MCPError::InternalError("tools/call dispatch returned malformed envelope".into()),
    )
}

/// Decode a raw JSON-RPC envelope value + dispatch it through
/// `dispatcher`, returning the rendered response envelope (or
/// `None` for a notification).
///
/// On a malformed envelope, returns an error envelope with the
/// originating id (if any) per JSON-RPC §5.1.
pub fn handle_raw_envelope<S, I, E, H, G, R>(
    dispatcher: &Dispatcher<S, I, E, H, G, R>,
    envelope: Value,
) -> Option<Value>
where
    S: SchemaProvider + 'static,
    I: NodeInspector + 'static,
    E: NeighborhoodExplorer + 'static,
    H: HybridSearcher + 'static,
    G: IngestProvider + 'static,
    R: RawQueryExecutor + 'static,
{
    handle_raw_envelope_with_scope(dispatcher, envelope, dispatcher.session_scope)
}

/// Decode and dispatch a raw JSON-RPC envelope using a
/// transport-authenticated request scope.
///
/// HTTPS uses this entry point after JWT verification so a read-scoped token
/// cannot inherit the composition root's local-operator Power scope. Other
/// callers should use [`handle_raw_envelope`].
pub fn handle_raw_envelope_with_scope<S, I, E, H, G, R>(
    dispatcher: &Dispatcher<S, I, E, H, G, R>,
    envelope: Value,
    session_scope: SessionScope,
) -> Option<Value>
where
    S: SchemaProvider + 'static,
    I: NodeInspector + 'static,
    E: NeighborhoodExplorer + 'static,
    H: HybridSearcher + 'static,
    G: IngestProvider + 'static,
    R: RawQueryExecutor + 'static,
{
    let id_for_error = envelope.get("id").cloned().unwrap_or(Value::Null);
    match decode_request(envelope) {
        Ok(req) => {
            if req.method.starts_with(MCP_NOTIFICATION_PREFIX) {
                return None;
            }
            let Some(response_id) = req.id.clone() else {
                return dispatcher.dispatch_with_scope(req, session_scope);
            };
            match req.method.as_str() {
                METHOD_MCP_INITIALIZE => mcp_success(response_id, initialize_result(&req.params)),
                METHOD_MCP_TOOLS_LIST => {
                    let tools = dispatcher
                        .wired_methods()
                        .into_iter()
                        .filter_map(tool_descriptor)
                        .collect::<Vec<_>>();
                    mcp_success(response_id, serde_json::json!({ "tools": tools }))
                }
                METHOD_MCP_TOOLS_CALL => {
                    handle_mcp_tools_call(dispatcher, req, response_id, session_scope)
                }
                _ => dispatcher.dispatch_with_scope(req, session_scope),
            }
        }
        Err(e) => {
            let env = JsonRpcErrorResponse::from_mcp(id_for_error, &e);
            Some(serde_json::to_value(env).unwrap_or(Value::Null))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_methods_catalog_is_canonical() {
        assert_eq!(
            KNOWN_METHODS,
            [
                METHOD_GRAPH_SCHEMA,
                METHOD_GRAPH_INSPECT,
                METHOD_GRAPH_EXPLORE,
                METHOD_GRAPH_SEARCH,
                METHOD_GRAPH_INGEST,
                METHOD_GRAPH_RAW_QUERY,
            ]
        );
        assert!(KNOWN_METHODS.len() <= ADR_004_TOOL_CAP);
        for method in KNOWN_METHODS {
            assert!(
                tool_descriptor(method).is_some(),
                "missing descriptor for {method}"
            );
        }
    }

    #[test]
    fn ingest_is_the_only_write_class_method() {
        for method in KNOWN_METHODS {
            let expected = if *method == METHOD_GRAPH_INGEST {
                OpClass::Write
            } else {
                OpClass::Read
            };
            assert_eq!(op_class_for_method(method), expected);
        }
    }

    #[test]
    fn near_miss_hints_only_name_public_methods() {
        assert_eq!(nearest_methods("graph.shema")[0], METHOD_GRAPH_SCHEMA);
        assert!(nearest_methods("unknown.method").is_empty());
    }

    #[test]
    #[should_panic(expected = "ADR-004 10-tool cap violated")]
    fn cap_guard_rejects_eleventh_active_tool() {
        enforce_adr_004_tool_cap(&[
            "graph.one",
            "graph.two",
            "graph.three",
            "graph.four",
            "graph.five",
            "graph.six",
            "graph.seven",
            "graph.eight",
            "graph.nine",
            "graph.ten",
            "graph.eleven",
        ]);
    }
}
