//! W14β M5-07 — `graph.search` Tier-1 MCP tool.
//!
//! Hybrid retrieval over the vector + BM25 substrates per
//! [ADR-036](../../../../docs/adr/ADR-036-hybrid-retrieval-architecture.md).
//! Input is a free-form query string (BM25 path) and / or an optional
//! query vector (HNSW path); output is a ranked node list with RRF-
//! fused scores.
//!
//! # Surface seam — `HybridSearcher`
//!
//! Following the W13δ M5-04 / M5-05 pattern: the MCP layer defines a
//! local adapter trait ([`HybridSearcher`]) rather than reaching
//! across the arcgraph-query bounded-context to bind directly to
//! [`arcgraph_query::executor::ops::rank_by_hybrid::RankByHybridOp`].
//! Production wiring at M4-08+ implements this trait on the storage
//! tenant handle (which composes the vector +
//! BM25 substrates the
//! [`arcgraph_query::executor::ExecutorSubstrate::vector_search`] /
//! `bm25_search` impls already read from).
//!
//! `feedback_avoid_speculative_scaffolding.md` applies: define the
//! trait at first consumer (this tool), avoid speculatively
//! extending arcgraph-query with a "hybrid search convenience" API
//! that has no other consumer.
//!
//! # Substrate availability — W14β spawn-prompt acceptance
//!
//! The spawn prompt pins a substrate-availability check that rejects
//! when the tenant has no vector or text index. The MCP layer
//! delegates the check to the [`HybridSearcher::available_substrates`]
//! adapter method: if neither vector nor BM25 is attached, the tool
//! returns [`MCPError::IndexUnavailable`] (-32004) BEFORE the searcher
//! body runs. If only one is attached, the tool runs in single-
//! substrate mode (RRF over one operand is a degenerate but well-
//! defined fusion: the sole substrate's rank order is the final
//! order).
//!
//! # Snapshot-LSN discipline
//!
//! Per ADR-038 amendment-03 §TIER-1 GAP E rule 1 ("Snapshot LSN
//! acquired at execute-time, before the first operator pulls a
//! batch"), the searcher's storage-side reads MUST acquire a snapshot
//! LSN before the first batch pull. Same shape as
//! [`crate::tools::inspect::NodeInspector`] /
//! [`crate::tools::explore::NeighborhoodExplorer`]. (Rule 2 is the
//! distinct multi-statement-LSN-sharing rule consumed by M4-83.)
//!
//! # Cross-tenant guard
//!
//! Same as the sibling tools: `request.tenant_id == session_tenant`
//! check runs BEFORE any searcher call.
//!
//! # Permission-aware retrieval — ADR-212 §D-4 Seam 1 (stage-1)
//!
//! When [`SearchRequest::principal`] is present, the tool resolves the
//! principal's effective permission set ONCE (per ADR-212 §D-5
//! statement-granularity freshness) via
//! [`HybridSearcher::permission_index`] and filters EVERY candidate —
//! vector leg, BM25 leg, hybrid fusion — through
//! `EffectivePermissions::is_visible` BEFORE response assembly. Top-k
//! legs over-fetch ([`ACL_OVERFETCH_FACTOR`], bounded refill ≤
//! [`ACL_OVERFETCH_ROUNDS`] rounds) so filtered results still approach
//! `k`; under-fill returns FEWER results, never unfiltered ones.
//!
//! Fail-closed postures (ADR-212 §D-4):
//! - principal present but the searcher exposes NO permission index →
//!   [`MCPError::IndexUnavailable`] (fail-LOUD, never silently
//!   unfiltered — the ADR-197-amendment-01 / #822 lesson);
//! - untagged content → UNCLASSIFIED → invisible (storage invariant);
//! - **forbidden ≡ not-found** — a query matching only restricted
//!   content returns the SAME response shape as a query matching
//!   nothing (no per-doc `PermissionDenied` existence oracle).
//!
//! `principal: None` is the principal-less SYSTEM-TRUSTED path
//! (internal operations and administration — ADR-212 §D-1): results
//! are unfiltered. Per #1293 the trusted path is gated on the
//! session's EXPLICIT [`SessionScope::Power`] marker (the same marker
//! gating `graph.raw_query`) —
//! NOT on "principal happened to be absent": a non-power session
//! omitting `principal` refuses with [`MCPError::Forbidden`] (-32008)
//! BEFORE any substrate probe, never runs unfiltered. Under power
//! scope the arm stays byte-identical to pre-ADR-212. Per the
//! §D-6 interim fail-closed-by-subtraction posture, end-user-facing
//! deployments MUST NOT hand end-user principals a token scoped to
//! any retrieval surface that does not thread `principal`. At
//! #1488/#1490, `graph.search`, `graph.inspect`, `graph.explore`, and Bolt
//! `RUN` all enter the shared `crate::read_acl::authorize_read` seam.
//! Bolt additionally decorates the
//! executor substrate so scalar projections, aggregates, ordering, and
//! limits cannot discard node provenance before filtering. `raw_query`
//! remains an explicit Power-only surface. Binding `principal` to the
//! per-request AUTHENTICATED identity (rejecting client
//! self-assertion under a power session) is the #1279-gated
//! architectural follow-on; this gate closes only the
//! absent-principal fail-OPEN default.
//!
//! # ADR provenance
//! - **ADR-004 §"Tier 1 (agent-facing, default)"** — `graph.search()`
//!   is the fourth Tier-1 tool in the 10-tool catalog.
//! - **ADR-036** — hybrid retrieval architecture (vector + BM25
//!   composition; RRF fusion).
//! - **ADR-038 amendment-02 §M4.b (M4-23)** — RANK BY HYBRID requires
//!   VECTOR+TEXT+K cross-substrate validation; substrate-availability
//!   checks land at the semantic-analyzer boundary. (TIER-2-c is the
//!   sibling observability section; the substrate-composition rule
//!   lives in amendment-02 §M4.b.)
//! - **ADR-038 amendment-03 §Structural-2** — M4-62 composes M3.a +
//!   M3.b + M3.d substrate via `TenantHandle` (the implicit-edge
//!   summary for the executor-side hybrid composition).
//! - **ADR-035** — HNSW vector index.
//! - **ADR-039** — BM25 text index.

use std::sync::Arc;

use arcgraph_core::{NodeId, TenantId};
use arcgraph_query::CancellationToken;
use arcgraph_storage::permissions::PermissionIndex;
use serde::{Deserialize, Serialize};

use crate::error::MCPError;
use crate::read_acl::{PERMISSION_INDEX_SLUG as SHARED_PERMISSION_INDEX_SLUG, authorize_read};
use crate::scope::SessionScope;
use crate::tools::ResponseFormat;
use crate::tools::schema::IndexKind;

/// Default top-K when an [`SearchRequest`] omits the field.
///
/// Pinned at 10 — matches the M3.c "anchor top-10 + expand 1-hop"
/// hybrid path that the production wiring composes against (per
/// ADR-036 §"M3.c hybrid exit gate"). Callers can override on the
/// request.
pub const DEFAULT_SEARCH_K: u32 = 10;

/// Hard cap on top-K to short-circuit hostile requests.
///
/// Pinned at 1000 — large enough for any reasonable agent-side
/// pagination, small enough to bound substrate IO. v1.0-alpha treats
/// this as a compile-time constant; M5-12 rate-limit slice may
/// move it to a per-tenant config value.
pub const MAX_SEARCH_K: u32 = 1000;

/// Hard cap on the query-time `ef_search` beam width (#816a).
///
/// Pinned at 4096 — generous headroom over [`MAX_SEARCH_K`] (so a
/// caller can always set `ef_search ≥ k` for the recall-vs-latency
/// curve Qdrant `hnsw_ef` / Milvus `ef` expose) while bounding the
/// per-query beam cost against a hostile value (`u32::MAX` would walk
/// the whole arena). A request above this rejects as
/// [`MCPError::InvalidParams`]; an omitted `ef_search` uses the engine
/// default (`HnswParams::ef_search` = 128) — unchanged behavior.
pub const MAX_SEARCH_EF: u32 = 4096;

/// ADR-212 §D-4 Seam-1 over-fetch factor: when a `principal` scopes
/// the request, each substrate leg fetches `k × ACL_OVERFETCH_FACTOR`
/// candidates so the post-visibility-filter result still approaches
/// `k`. Per the ADR's §4 budget the multiplier is engine-controlled
/// and bounded; stage-4 pre-filtered ANN removes it.
pub const ACL_OVERFETCH_FACTOR: u32 = 4;

/// ADR-212 §D-4 Seam-1 bounded-refill cap: at most this many fetch
/// rounds total (round 2 widens to `k × FACTOR²`). After the last
/// round the tool returns the visible candidates found so far —
/// under-fill returns FEWER than `k` results, never unfiltered ones.
pub const ACL_OVERFETCH_ROUNDS: u32 = 2;

/// Slug carried by the [`MCPError::IndexUnavailable`] raised when a
/// `principal`-scoped request reaches a searcher that exposes no
/// permission index ([`HybridSearcher::permission_index`] returned
/// `None`). Mirrors [`SUBSTRATE_SLUG_VECTOR`] / [`SUBSTRATE_SLUG_BM25`]
/// so clients can route on the slug.
pub const PERMISSION_INDEX_SLUG: &str = SHARED_PERMISSION_INDEX_SLUG;

/// The node property holding a vector embedding, by convention, for the served
/// HNSW substrate (#765 PART-1). An MCP client ingests a vector via `graph.ingest`
/// (`{"embedding": [0.12, …]}`); the served `SubstrateSearchProvider` builds the
/// per-tenant HNSW from every node carrying it, and `graph.search` with a
/// `query_vec` runs KNN over it. A v1 product contract reusing the existing ingest
/// property bag (no new wire field); one source of truth for the MCP `graph.search`
/// body + the `arcgraph-cli` provider. A configurable property name is a follow-on.
///
/// **Sync guard (#830 D4 / R1 #861 Finding #1):** this const is the
/// source-of-truth, but `arcgraph-query` duplicates its VALUE as a
/// private `const DEFAULT_VECTOR_PROPERTY` in
/// `crates/arcgraph-query/src/executor/ops/procedure_call.rs` — used to
/// resolve the advisory index name of `db.index.vector.queryNodes` to
/// the served vector property. It is duplicated (not imported) to avoid
/// an `arcgraph-query → arcgraph-mcp` dependency (the wrong
/// bounded-context direction; `arcgraph-mcp` already depends on
/// `arcgraph-query`). **The two MUST stay in sync:** if you change this
/// value, update the query-side duplicate too, or `queryNodes` will
/// search a property holding no vectors → silent-empty on the served
/// search path.
pub const DEFAULT_VECTOR_PROPERTY: &str = "embedding";

/// Substrate-availability descriptor — what the searcher knows it
/// can compose against, per tenant.
///
/// Mirrors the [`crate::tools::schema::IndexKind`] tags so a client
/// that read `graph.schema` first can match the `graph.search`
/// "substrate unavailable" error message against the schema's
/// `indexes` slot. We omit the [`IndexKind::Community`] variant from
/// the [`AvailableSubstrates`] flags because v1.0-alpha hybrid search
/// composes ONLY vector + text — community detection ships at v1.0
/// per ADR-036 §D-6 ("Community detection — GVE-Leiden + DF Leiden at
/// v1.0; retrieval surface" — static + incremental + membership index)
/// but the executor-side `LogicalCommunityLookup` lowering rides on
/// M4-32 / M4-62b per ADR-038 amendment-02 §M4.c (Hybrid retrieval
/// lowering). Community lookups surface to the MCP layer as a
/// forward-pin to M4-62b.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AvailableSubstrates {
    /// HNSW vector index attached + readable.
    pub vector: bool,
    /// BM25 text index attached + readable.
    pub bm25: bool,
}

impl AvailableSubstrates {
    /// `true` if at least one substrate is available; the searcher
    /// body can produce some output when this is true.
    #[must_use]
    pub fn any(&self) -> bool {
        self.vector || self.bm25
    }

    /// Construct a `none-available` instance — convenience for
    /// production impls that want to return a clean "no substrate
    /// attached" rather than `bool::default()` gymnastics.
    #[must_use]
    pub fn none() -> Self {
        Self::default()
    }
}

/// Adapter trait read by the [`search_tool`] entry point.
///
/// Implementations live OUTSIDE this crate: tests stub it in-line;
/// production wiring at M4-08+ implements it on the storage tenant
/// handle.
///
/// # Per-tenant scoping
///
/// `tenant: TenantId` parameter matches the sibling tool traits.
///
/// # `Send + Sync`
///
/// MCP transport runs on a tokio runtime; the searcher must be
/// shareable across awaits.
///
/// # Cancellation + snapshot-LSN contracts
///
/// Same shape as
/// [`crate::tools::explore::NeighborhoodExplorer`]: production impls
/// MUST honor the cancellation token at hop / batch boundaries and
/// acquire a snapshot LSN before the first substrate pull.
pub trait HybridSearcher: Send + Sync {
    /// Return the substrate-availability flags for `tenant`. Called
    /// by the tool entry point BEFORE the search to detect the "no
    /// substrate attached" rejection path.
    ///
    /// # Cancellation
    ///
    /// `cancel` is the same per-request token threaded into
    /// [`HybridSearcher::search`]. v1.0-alpha stub impls return cached
    /// flags in O(1) and ignore the token; production impls MAY make a
    /// substrate-handle / partition-router lookup that takes
    /// non-trivial wall-time, so the token MUST be checked at any
    /// blocking boundary and the impl MUST short-circuit with
    /// [`MCPError::Cancelled`] on a tripped token. Symmetric with
    /// [`HybridSearcher::search`]'s cancellation contract; closes
    /// W14β PR #292 review MED-2 (asymmetry-with-`search`).
    fn available_substrates(
        &self,
        tenant: TenantId,
        cancel: &CancellationToken,
    ) -> Result<AvailableSubstrates, MCPError>;

    /// Run the hybrid search on `tenant`, returning ranked hits.
    ///
    /// `query_text` is the BM25 input (empty string disables the
    /// BM25 operand; production impls SHOULD treat empty as "skip BM25"
    /// rather than emitting an empty-query top-K). `query_vec` is the
    /// optional vector operand. `k` is the top-K cap; impls MUST NOT
    /// return more than `k` hits. `cancel` is the cancellation token.
    ///
    /// Returns ranked hits in score-descending order (RRF score is
    /// rank-based; the absolute number is rank-fusion-relative).
    fn search(
        &self,
        tenant: TenantId,
        query_text: &str,
        query_vec: Option<&[f32]>,
        k: u32,
        cancel: &CancellationToken,
    ) -> Result<Vec<SearchHit>, MCPError>;

    /// #815 / #816a — like [`Self::search`] but with the label filter
    /// pushed into the substrate traversal and an optional query-time
    /// `ef_search` recall knob.
    ///
    /// - `label_filter`: non-empty allowlist of label NAMES. Production
    ///   impls resolve them to the substrate's id space and push the
    ///   predicate INTO the HNSW beam, so a SELECTIVE filter returns `k`
    ///   true matches rather than collapsing recall (#815). `None` /
    ///   empty = no filter.
    /// - `ef_search`: query-time HNSW beam width (#816a); `None` = the
    ///   engine default; higher → higher recall.
    ///
    /// The default impl is back-compat: it IGNORES both knobs and
    /// delegates to [`Self::search`]. The `graph.search` tool still
    /// applies the label post-filter at the MCP boundary, so a stub impl
    /// that does not override this keeps its current behavior. Production
    /// [`crate::storage::StorageHybridSearcher`] OVERRIDES it to push the
    /// filter + `ef_search` down into the served HNSW.
    // 8 args: the pushdown knobs (label_filter + ef_search) parallel the
    // existing `search` shape; a struct adds ceremony without clarity at the
    // call sites — same allow precedent as `HnswGraph::search_with_rescore`.
    #[allow(clippy::too_many_arguments)]
    fn search_filtered(
        &self,
        tenant: TenantId,
        query_text: &str,
        query_vec: Option<&[f32]>,
        k: u32,
        label_filter: Option<&[String]>,
        ef_search: Option<u32>,
        cancel: &CancellationToken,
    ) -> Result<Vec<SearchHit>, MCPError> {
        // Back-compat default: knobs unsupported here → distance-only
        // top-k; the MCP boundary applies the label post-filter.
        let _ = (label_filter, ef_search);
        self.search(tenant, query_text, query_vec, k, cancel)
    }

    /// ADR-212 §D-4 Seam-1 — the per-tenant source-ACL permission
    /// index backing principal-scoped enforcement, or `None` when this
    /// searcher cannot enforce (no storage binding).
    ///
    /// The default is `Ok(None)`, which is FAIL-CLOSED at the tool
    /// boundary: a request carrying [`SearchRequest::principal`]
    /// against a `None`-returning searcher rejects with
    /// [`MCPError::IndexUnavailable`] (slug
    /// [`PERMISSION_INDEX_SLUG`]) — it is NEVER served unfiltered.
    /// Principal-less requests never call this method, so stub impls
    /// that ignore it keep their existing behavior on every existing
    /// test. Production [`crate::storage::StorageHybridSearcher`]
    /// overrides this to expose
    /// `TenantHandle::permissions()` (ADR-037-amendment-02).
    fn permission_index(
        &self,
        tenant: TenantId,
        cancel: &CancellationToken,
    ) -> Result<Option<Arc<PermissionIndex>>, MCPError> {
        let _ = (tenant, cancel);
        Ok(None)
    }
}

/// One ranked hit returned from [`HybridSearcher::search`].
///
/// Serializes as a uniform-shape record (TOON tabular-friendly per
/// design-v2 §9.3).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchHit {
    /// The matched node id.
    pub node_id: u64,
    /// Optional label.
    pub label: Option<String>,
    /// RRF-fused score (or single-substrate rank score when only one
    /// substrate is composed). Higher is better.
    pub score: f64,
}

// ─────────────────────────────────────────────────────────────────────
// Request envelope
// ─────────────────────────────────────────────────────────────────────

/// Request params for the `graph.search` tool.
///
/// `#[serde(deny_unknown_fields)]` under the code-quality policy.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SearchRequest {
    /// The tenant to search within.
    pub tenant_id: u64,
    /// Free-form BM25 query string. Omitted or empty string disables
    /// the BM25 operand (the search runs in vector-only mode if a
    /// `query_vec` is present); a request with empty/missing `query`
    /// AND no `query_vec` rejects as [`MCPError::InvalidParams`].
    #[serde(default)]
    pub query: String,
    /// Optional vector operand (HNSW path). Must be non-empty when
    /// present.
    #[serde(default)]
    pub query_vec: Option<Vec<f32>>,
    /// Top-K cap. Defaults to [`DEFAULT_SEARCH_K`]; values above
    /// [`MAX_SEARCH_K`] reject as [`MCPError::InvalidParams`].
    #[serde(default)]
    pub k: Option<u32>,
    /// Optional label allowlist filter — only hits whose label is in
    /// this list are returned. Empty Vec = "no filter". Production
    /// impls push the filter down into the substrate when possible.
    #[serde(default)]
    pub label_filter: Option<Vec<String>>,
    /// Optional query-time HNSW beam width (`ef_search`) — the
    /// recall-vs-latency knob (#816a) that Qdrant exposes as `hnsw_ef`
    /// and Milvus as `ef`. `None` (omitted) uses the engine default
    /// (`HnswParams::ef_search` = 128), preserving prior behavior;
    /// `Some(n)` trades recall for latency (higher → higher recall).
    /// Validated against [`MAX_SEARCH_EF`]; `0` rejects as
    /// [`MCPError::InvalidParams`]. Additive + non-breaking.
    #[serde(default)]
    pub ef_search: Option<u32>,
    /// Optional render-format hint. Defaults to TOON — search results
    /// are uniform-shape rows (the design-v2 §9.3 token-savings path).
    #[serde(default)]
    pub format: Option<ResponseFormat>,
    /// ADR-212 §D-1 — the end-user principal this retrieval is issued
    /// ON BEHALF OF (the explicit on-behalf-of parameter for
    /// embedded / CLI / AEB-service callers; the authenticated-subject
    /// derivation rides the #761 auth lineage as identity work
    /// completes). When present, every hit is filtered through the
    /// principal's effective source-ACL permission set BEFORE response
    /// assembly (module docs §"Permission-aware retrieval"); an empty
    /// string rejects as [`MCPError::InvalidParams`]. When absent, the
    /// request is admitted ONLY on a [`SessionScope::Power`] session
    /// (the principal-less SYSTEM-TRUSTED path, unfiltered); a
    /// non-power session omitting this field refuses with
    /// [`MCPError::Forbidden`] — fail-closed per #1293 / ADR-212 §D-6
    /// (absence of a principal is NOT a trusted marker).
    #[serde(default)]
    pub principal: Option<String>,
}

/// `IndexKind` echo on the substrate-missing error path. Public so
/// downstream MCP clients can route on the slug without re-parsing
/// the error string.
pub const SUBSTRATE_SLUG_VECTOR: &str = "vector";
pub const SUBSTRATE_SLUG_BM25: &str = "bm25";

/// Body shape returned in the JSON-RPC `result` slot for `graph.search`.
///
/// Wraps the top-K list alongside the echo'd `k` cap so a client
/// reading a single response envelope can pin "did the searcher
/// honor my k?" without re-counting the array.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchResult {
    /// The honored top-K (≤ requested k; may be smaller when fewer
    /// matches exist).
    pub k: u32,
    /// Ranked hits, score-descending.
    pub hits: Vec<SearchHit>,
}

// ─────────────────────────────────────────────────────────────────────
// Tool entry point
// ─────────────────────────────────────────────────────────────────────

/// `graph.search` — return RRF-fused ranked hits as JSON-RPC `result`.
///
/// # Cross-tenant guard
///
/// Same shape as the sibling Tier-1 tools.
///
/// # Input-validation order
///
/// 1. Cross-tenant guard.
/// 2. Principal-less scope gate (#1293) — an ABSENT `principal` on a
///    session whose scope does not admit power refuses as
///    [`MCPError::Forbidden`] (-32008) BEFORE any further validation
///    or substrate probe. Cross-tenant runs first so a cross-tenant
///    probe cannot distinguish scopes (the `graph.raw_query` ordering
///    per ADR-004 amendment-03).
/// 3. `k` cap.
/// 4. "At least one operand" — empty query AND no vector → reject.
///    Empty vector (zero-len `Vec<f32>`) also rejects per the
///    [`SearchRequest::query_vec`] doc contract.
/// 5. Substrate-availability — if the tenant has neither vector nor
///    BM25 attached, reject as [`MCPError::IndexUnavailable`].
/// 6. Substrate-vs-operand match — vector operand against a tenant
///    that has no vector substrate, or text operand with no usable
///    vector fallback against a tenant that has no BM25 substrate,
///    MUST reject as [`MCPError::IndexUnavailable`] with the slug
///    naming the missing substrate. A request with both text and
///    vector degrades to vector-only when BM25 is unavailable and the
///    vector substrate is attached.
///
/// # Cancellation
///
/// `cancel` is the cancellation token bound to this request.
///
/// # Errors
///
/// - [`MCPError::Unauthorized`] — cross-tenant request.
/// - [`MCPError::Forbidden`] — principal-less request on a non-power
///   session (#1293 fail-closed default; the unfiltered SYSTEM-TRUSTED
///   path requires the explicit [`SessionScope::Power`] marker).
/// - [`MCPError::InvalidParams`] — `k > MAX_SEARCH_K`, empty
///   query + no vector, or empty `query_vec`.
/// - [`MCPError::IndexUnavailable`] — no substrate attached OR a
///   substrate-vs-operand mismatch.
/// - [`MCPError::TenantUnknown`] — provider has no binding for the
///   tenant.
/// - [`MCPError::Cancelled`] — the cancellation token tripped.
/// - [`MCPError::InternalError`] — serializer encode failure.
pub fn search_tool<S: HybridSearcher + ?Sized>(
    searcher: &S,
    session_tenant: TenantId,
    session_scope: SessionScope,
    cancel: &CancellationToken,
    req: SearchRequest,
) -> Result<serde_json::Value, MCPError> {
    let request_tenant = TenantId::new(req.tenant_id);
    if request_tenant != session_tenant {
        return Err(MCPError::Unauthorized);
    }

    let access = authorize_read(
        "graph.search",
        req.principal.as_deref(),
        session_scope,
        || searcher.permission_index(request_tenant, cancel),
    )?;

    let k = req.k.unwrap_or(DEFAULT_SEARCH_K);
    if k > MAX_SEARCH_K {
        return Err(MCPError::InvalidParams(format!(
            "graph.search: k={k} exceeds hard cap {MAX_SEARCH_K}"
        )));
    }
    if k == 0 {
        return Err(MCPError::InvalidParams(
            "graph.search: k must be ≥ 1".into(),
        ));
    }
    // #816a — validate the optional ef_search recall knob. Reject
    // gracefully (InvalidParams), never panic. Omitted → engine default.
    if let Some(ef) = req.ef_search {
        if ef == 0 {
            return Err(MCPError::InvalidParams(
                "graph.search: ef_search must be ≥ 1 when present".into(),
            ));
        }
        if ef > MAX_SEARCH_EF {
            return Err(MCPError::InvalidParams(format!(
                "graph.search: ef_search={ef} exceeds hard cap {MAX_SEARCH_EF}"
            )));
        }
    }

    let has_text = !req.query.is_empty();
    let has_vec = match &req.query_vec {
        Some(v) if !v.is_empty() => true,
        Some(_) => {
            return Err(MCPError::InvalidParams(
                "graph.search: query_vec must be non-empty when present".into(),
            ));
        }
        None => false,
    };
    if !has_text && !has_vec {
        return Err(MCPError::InvalidParams(
            "graph.search: at least one of `query` (non-empty) or `query_vec` (non-empty) is required".into(),
        ));
    }

    // ADR-212 §D-4 Seam-1 — validate the on-behalf-of principal and
    // resolve its effective permission set ONCE per request (§D-5
    // statement-granularity freshness). Fail-LOUD when a
    // principal-scoped request reaches a searcher that cannot enforce:
    // it is rejected, never served unfiltered (the #822
    // silent-degradation lesson).
    let avail = searcher.available_substrates(request_tenant, cancel)?;
    if !avail.any() {
        return Err(MCPError::IndexUnavailable(
            "tenant has no vector or bm25 substrate attached".into(),
        ));
    }
    if has_vec && !avail.vector {
        return Err(MCPError::IndexUnavailable(SUBSTRATE_SLUG_VECTOR.into()));
    }
    if has_text && !avail.bm25 && !has_vec {
        return Err(MCPError::IndexUnavailable(SUBSTRATE_SLUG_BM25.into()));
    }
    let query_text = if has_text && avail.bm25 {
        req.query.as_str()
    } else {
        ""
    };

    // #815 / #816a — push the label filter + ef_search DOWN into the
    // searcher (filter-during-search + recall knob). The default trait
    // impl falls back to the unfiltered `search`, and the
    // belt-and-suspenders MCP-boundary post-filter below still runs, so
    // a stub that does not override `search_filtered` is unchanged.
    //
    // ADR-212 §D-4 Seam-1: under a principal, the legs OVER-FETCH
    // (`k × ACL_OVERFETCH_FACTOR`, widening once to `k × FACTOR²`,
    // ≤ ACL_OVERFETCH_ROUNDS substrate calls, never beyond
    // MAX_SEARCH_K) and every candidate is filtered through
    // `is_visible` BEFORE response assembly. Under-fill returns fewer
    // than `k` hits — never unfiltered ones. The principal-less arm is
    // byte-identical to the pre-ADR-212 path and — per the #1293 gate
    // above — only reachable on a power-scope session.
    let hits = if access.is_system_trusted() {
        searcher.search_filtered(
            request_tenant,
            query_text,
            req.query_vec.as_deref(),
            k,
            req.label_filter.as_deref(),
            req.ef_search,
            cancel,
        )?
    } else {
        let mut round: u32 = 1;
        let mut fetch_k = k
            .saturating_mul(ACL_OVERFETCH_FACTOR)
            .min(MAX_SEARCH_K)
            .max(k);
        loop {
            let raw = searcher.search_filtered(
                request_tenant,
                query_text,
                req.query_vec.as_deref(),
                fetch_k,
                req.label_filter.as_deref(),
                req.ef_search,
                cancel,
            )?;
            // The substrate returned fewer candidates than asked:
            // it is exhausted; a wider refetch cannot find more.
            let exhausted = raw.len() < fetch_k as usize;
            let visible: Vec<SearchHit> = raw
                .into_iter()
                .filter(|h| access.allows(NodeId::new(h.node_id)))
                .collect();
            // Fill check mirrors the MCP-boundary label post-filter
            // below so a refill round accounts for BOTH predicates
            // (a stub searcher may not push the label filter down).
            let fill = match req.label_filter {
                Some(ref allow) if !allow.is_empty() => visible
                    .iter()
                    .filter(|h| match &h.label {
                        Some(l) => allow.iter().any(|a| a == l),
                        None => false,
                    })
                    .count(),
                _ => visible.len(),
            };
            let widened = k
                .saturating_mul(ACL_OVERFETCH_FACTOR.saturating_pow(2))
                .min(MAX_SEARCH_K);
            if fill >= k as usize
                || exhausted
                || round >= ACL_OVERFETCH_ROUNDS
                || widened == fetch_k
            {
                break visible;
            }
            round += 1;
            fetch_k = widened;
        }
    };

    // Honor the caller's label filter at the MCP boundary as a
    // belt-and-suspenders pin against a stub impl that ignores the
    // filter. Production impls push the filter down into the
    // substrate (so this is then a no-op — every returned hit already
    // matches); the MCP boundary keeps the call-site contract
    // verifiable. Empty allowlist = no-op.
    let filtered: Vec<SearchHit> = match req.label_filter {
        Some(ref allow) if !allow.is_empty() => hits
            .into_iter()
            .filter(|h| match &h.label {
                Some(l) => allow.iter().any(|a| a == l),
                None => false,
            })
            .collect(),
        _ => hits,
    };

    // Defensive truncation: enforce the caller's `k` at the MCP
    // boundary so a stub fixture (or a future production impl that
    // returns more than `k` on a fast path) cannot violate the wire
    // contract.
    let mut hits = filtered;
    if hits.len() > k as usize {
        hits.truncate(k as usize);
    }
    let result = SearchResult { k, hits };

    let format = req.format.unwrap_or(ResponseFormat::Toon);
    let value = serde_json::to_value(&result)
        .map_err(|e| MCPError::InternalError(format!("search result serialize: {e}")))?;
    crate::tools::render_response(format, &value)
}

/// Convenience: convert [`AvailableSubstrates`] into the matching
/// [`crate::tools::schema::IndexKind`] descriptors so the
/// `graph.schema` and `graph.search` tools render a consistent slug
/// set across MCP clients. v1.0-alpha emits only `vector` + `bm25`;
/// community is forward-pinned.
#[must_use]
pub fn substrate_kinds(a: AvailableSubstrates) -> Vec<IndexKind> {
    let mut v = Vec::with_capacity(2);
    if a.vector {
        v.push(IndexKind::Vector);
    }
    if a.bm25 {
        v.push(IndexKind::Bm25);
    }
    v
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Stub searcher: returns a caller-bakd hit list for a matching
    /// tenant + records the bm25 / vector availability. ADR-212: an
    /// optional [`PermissionIndex`] (None = the trait's fail-closed
    /// default) + a fetch log recording every `k` the tool asked for
    /// (pins the §D-4 over-fetch/refill round shape).
    #[derive(Debug, Clone)]
    struct StubSearcher {
        tenant: TenantId,
        avail: AvailableSubstrates,
        hits: Vec<SearchHit>,
        perms: Option<Arc<PermissionIndex>>,
        fetch_log: Arc<std::sync::Mutex<Vec<u32>>>,
    }

    impl StubSearcher {
        fn new(tenant: TenantId) -> Self {
            Self {
                tenant,
                avail: AvailableSubstrates {
                    vector: true,
                    bm25: true,
                },
                hits: vec![],
                perms: None,
                fetch_log: Arc::new(std::sync::Mutex::new(Vec::new())),
            }
        }
        fn with_avail(mut self, a: AvailableSubstrates) -> Self {
            self.avail = a;
            self
        }
        fn with_hits(mut self, h: Vec<SearchHit>) -> Self {
            self.hits = h;
            self
        }
        fn with_permissions(mut self, p: Arc<PermissionIndex>) -> Self {
            self.perms = Some(p);
            self
        }
        fn fetches(&self) -> Vec<u32> {
            self.fetch_log.lock().expect("fetch log lock").clone()
        }
    }

    impl HybridSearcher for StubSearcher {
        fn available_substrates(
            &self,
            tenant: TenantId,
            cancel: &CancellationToken,
        ) -> Result<AvailableSubstrates, MCPError> {
            if cancel.is_cancelled() {
                return Err(MCPError::Cancelled);
            }
            if tenant != self.tenant {
                return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
            }
            Ok(self.avail)
        }

        fn search(
            &self,
            tenant: TenantId,
            _query_text: &str,
            _query_vec: Option<&[f32]>,
            k: u32,
            cancel: &CancellationToken,
        ) -> Result<Vec<SearchHit>, MCPError> {
            if cancel.is_cancelled() {
                return Err(MCPError::Cancelled);
            }
            if tenant != self.tenant {
                return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
            }
            self.fetch_log.lock().expect("fetch log lock").push(k);
            let mut hits = self.hits.clone();
            hits.truncate(k as usize);
            Ok(hits)
        }

        fn permission_index(
            &self,
            tenant: TenantId,
            cancel: &CancellationToken,
        ) -> Result<Option<Arc<PermissionIndex>>, MCPError> {
            if cancel.is_cancelled() {
                return Err(MCPError::Cancelled);
            }
            if tenant != self.tenant {
                return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
            }
            Ok(self.perms.clone())
        }
    }

    fn fixture_hits() -> Vec<SearchHit> {
        vec![
            SearchHit {
                node_id: 1,
                label: Some("Document".into()),
                score: 0.95,
            },
            SearchHit {
                node_id: 2,
                label: Some("Document".into()),
                score: 0.81,
            },
            SearchHit {
                node_id: 3,
                label: Some("Person".into()),
                score: 0.55,
            },
        ]
    }

    #[test]
    fn search_tool_returns_hits_for_text_query() {
        let s = StubSearcher::new(TenantId::new(7)).with_hits(fixture_hits());
        let req = SearchRequest {
            tenant_id: 7,
            query: "machine learning".into(),
            query_vec: None,
            k: Some(3),
            label_filter: None,
            ef_search: None,
            format: Some(ResponseFormat::Json),
            principal: None,
        };
        let token = CancellationToken::new();
        let resp = search_tool(&s, TenantId::new(7), SessionScope::Power, &token, req).expect("ok");
        assert_eq!(resp["format"], "json");
        let body = resp["body"].as_str().unwrap();
        assert!(body.contains("\"k\":3"));
        // Top hit at the head of the array.
        assert!(body.contains("\"node_id\":1"));
    }

    #[test]
    fn search_tool_rejects_empty_query_with_no_vector() {
        let s = StubSearcher::new(TenantId::new(7));
        let req = SearchRequest {
            tenant_id: 7,
            query: "".into(),
            query_vec: None,
            k: Some(10),
            label_filter: None,
            ef_search: None,
            format: None,
            principal: None,
        };
        let token = CancellationToken::new();
        let err = search_tool(&s, TenantId::new(7), SessionScope::Power, &token, req)
            .expect_err("must reject");
        assert_eq!(err.code(), -32602);
        match err {
            MCPError::InvalidParams(msg) => {
                assert!(msg.contains("query"));
            }
            other => panic!("expected InvalidParams, got {other:?}"),
        }
    }

    #[test]
    fn search_tool_rejects_empty_vector_with_text() {
        // `query_vec: Some(vec![])` should reject; empty vec is a
        // malformed operand. Even if `query` is non-empty, we want a
        // clean error rather than silently dropping the empty vector.
        let s = StubSearcher::new(TenantId::new(7));
        let req = SearchRequest {
            tenant_id: 7,
            query: "x".into(),
            query_vec: Some(vec![]),
            k: Some(10),
            label_filter: None,
            ef_search: None,
            format: None,
            principal: None,
        };
        let token = CancellationToken::new();
        let err = search_tool(&s, TenantId::new(7), SessionScope::Power, &token, req)
            .expect_err("empty vec rejects");
        assert_eq!(err.code(), -32602);
    }

    #[test]
    fn search_tool_supports_text_only() {
        // BM25-only query: vector substrate absent on the tenant.
        let avail = AvailableSubstrates {
            vector: false,
            bm25: true,
        };
        let s = StubSearcher::new(TenantId::new(7))
            .with_avail(avail)
            .with_hits(fixture_hits());
        let req = SearchRequest {
            tenant_id: 7,
            query: "graph".into(),
            query_vec: None,
            k: Some(2),
            label_filter: None,
            ef_search: None,
            format: Some(ResponseFormat::Json),
            principal: None,
        };
        let token = CancellationToken::new();
        let resp = search_tool(&s, TenantId::new(7), SessionScope::Power, &token, req).expect("ok");
        let body = resp["body"].as_str().unwrap();
        assert!(body.contains("\"k\":2"));
    }

    #[test]
    fn search_tool_supports_vector_only() {
        // Vector-only query: BM25 absent on the tenant. Caller sends
        // empty `query` + a non-empty vector.
        let avail = AvailableSubstrates {
            vector: true,
            bm25: false,
        };
        let s = StubSearcher::new(TenantId::new(7))
            .with_avail(avail)
            .with_hits(fixture_hits());
        let req = SearchRequest {
            tenant_id: 7,
            query: "".into(),
            query_vec: Some(vec![0.1, 0.2, 0.3]),
            k: Some(2),
            label_filter: None,
            ef_search: None,
            format: Some(ResponseFormat::Json),
            principal: None,
        };
        let token = CancellationToken::new();
        let resp = search_tool(&s, TenantId::new(7), SessionScope::Power, &token, req).expect("ok");
        let body = resp["body"].as_str().unwrap();
        assert!(body.contains("\"k\":2"));
    }

    #[test]
    fn search_tool_degrades_hybrid_to_vector_when_bm25_unavailable() {
        // #916: agents naturally send both text + vector. If BM25 is
        // down but vector is attached, return vector hits rather than
        // hard-erroring with IndexUnavailable("bm25").
        let avail = AvailableSubstrates {
            vector: true,
            bm25: false,
        };
        let s = StubSearcher::new(TenantId::new(7))
            .with_avail(avail)
            .with_hits(vec![SearchHit {
                node_id: 42,
                label: Some("Document".into()),
                score: 0.98,
            }]);
        let req = SearchRequest {
            tenant_id: 7,
            query: "Alpha".into(),
            query_vec: Some(vec![0.1, 0.2, 0.3]),
            k: Some(1),
            label_filter: None,
            ef_search: None,
            format: Some(ResponseFormat::Json),
            principal: None,
        };
        let token = CancellationToken::new();
        let resp = search_tool(&s, TenantId::new(7), SessionScope::Power, &token, req)
            .expect("vector fallback");
        let body: SearchResult =
            serde_json::from_str(resp["body"].as_str().expect("json body")).expect("result body");
        assert_eq!(body.hits.len(), 1);
        assert_eq!(body.hits[0].node_id, 42);
        assert_eq!(body.hits[0].score, 0.98);
    }

    #[test]
    fn search_tool_rejects_hybrid_when_vector_operand_has_no_vector_substrate() {
        let avail = AvailableSubstrates {
            vector: false,
            bm25: true,
        };
        let s = StubSearcher::new(TenantId::new(7)).with_avail(avail);
        let req = SearchRequest {
            tenant_id: 7,
            query: "Alpha".into(),
            query_vec: Some(vec![0.1, 0.2, 0.3]),
            k: Some(1),
            label_filter: None,
            ef_search: None,
            format: Some(ResponseFormat::Json),
            principal: None,
        };
        let token = CancellationToken::new();
        let err = search_tool(&s, TenantId::new(7), SessionScope::Power, &token, req)
            .expect_err("vector missing");
        assert_eq!(err.code(), -32004);
        match err {
            MCPError::IndexUnavailable(slug) => assert_eq!(slug, SUBSTRATE_SLUG_VECTOR),
            other => panic!("expected IndexUnavailable(vector), got {other:?}"),
        }
    }

    #[test]
    fn search_request_accepts_vector_only_without_query_field() {
        let req: SearchRequest = serde_json::from_value(serde_json::json!({
            "tenant_id": 7,
            "query_vec": [0.1, 0.2, 0.3],
            "k": 1,
            "format": "json",
        }))
        .expect("query defaults to empty string");
        assert!(req.query.is_empty());

        let avail = AvailableSubstrates {
            vector: true,
            bm25: false,
        };
        let s = StubSearcher::new(TenantId::new(7))
            .with_avail(avail)
            .with_hits(vec![SearchHit {
                node_id: 42,
                label: Some("Document".into()),
                score: 0.98,
            }]);
        let token = CancellationToken::new();
        let resp = search_tool(&s, TenantId::new(7), SessionScope::Power, &token, req)
            .expect("vector-only ok");
        let body: SearchResult =
            serde_json::from_str(resp["body"].as_str().expect("json body")).expect("result body");
        assert_eq!(body.hits.len(), 1);
        assert_eq!(body.hits[0].node_id, 42);
        assert_eq!(body.hits[0].score, 0.98);
    }

    #[test]
    fn search_tool_rejects_request_with_neither_query_nor_vector() {
        let req: SearchRequest = serde_json::from_value(serde_json::json!({
            "tenant_id": 7,
        }))
        .expect("query defaults to empty string");
        let s = StubSearcher::new(TenantId::new(7));
        let token = CancellationToken::new();
        let err = search_tool(&s, TenantId::new(7), SessionScope::Power, &token, req)
            .expect_err("no operands");
        assert_eq!(err.code(), -32602);
        match err {
            MCPError::InvalidParams(msg) => {
                assert!(msg.contains("query"));
                assert!(msg.contains("query_vec"));
            }
            other => panic!("expected InvalidParams, got {other:?}"),
        }
    }

    #[test]
    fn search_tool_supports_hybrid() {
        // Both substrates attached + a non-empty text + non-empty
        // vector. Should produce a result envelope.
        let s = StubSearcher::new(TenantId::new(7)).with_hits(fixture_hits());
        let req = SearchRequest {
            tenant_id: 7,
            query: "agentic graphs".into(),
            query_vec: Some(vec![0.1, 0.2]),
            k: Some(3),
            label_filter: None,
            ef_search: None,
            format: Some(ResponseFormat::Json),
            principal: None,
        };
        let token = CancellationToken::new();
        let resp = search_tool(&s, TenantId::new(7), SessionScope::Power, &token, req).expect("ok");
        let body = resp["body"].as_str().unwrap();
        assert!(body.contains("node_id"));
    }

    #[test]
    fn search_tool_rejects_when_no_substrate_attached() {
        let s = StubSearcher::new(TenantId::new(7)).with_avail(AvailableSubstrates::none());
        let req = SearchRequest {
            tenant_id: 7,
            query: "x".into(),
            query_vec: None,
            k: Some(5),
            label_filter: None,
            ef_search: None,
            format: None,
            principal: None,
        };
        let token = CancellationToken::new();
        let err = search_tool(&s, TenantId::new(7), SessionScope::Power, &token, req)
            .expect_err("no substrate");
        assert_eq!(err.code(), -32004);
    }

    #[test]
    fn search_tool_rejects_vector_operand_when_vector_substrate_missing() {
        let avail = AvailableSubstrates {
            vector: false,
            bm25: true,
        };
        let s = StubSearcher::new(TenantId::new(7)).with_avail(avail);
        let req = SearchRequest {
            tenant_id: 7,
            query: "".into(),
            query_vec: Some(vec![0.1]),
            k: Some(5),
            label_filter: None,
            ef_search: None,
            format: None,
            principal: None,
        };
        let token = CancellationToken::new();
        let err = search_tool(&s, TenantId::new(7), SessionScope::Power, &token, req)
            .expect_err("vec missing");
        assert_eq!(err.code(), -32004);
        match err {
            MCPError::IndexUnavailable(slug) => assert_eq!(slug, "vector"),
            other => panic!("expected IndexUnavailable, got {other:?}"),
        }
    }

    #[test]
    fn search_tool_rejects_text_operand_when_bm25_substrate_missing() {
        let avail = AvailableSubstrates {
            vector: true,
            bm25: false,
        };
        let s = StubSearcher::new(TenantId::new(7)).with_avail(avail);
        let req = SearchRequest {
            tenant_id: 7,
            query: "x".into(),
            query_vec: None,
            k: Some(5),
            label_filter: None,
            ef_search: None,
            format: None,
            principal: None,
        };
        let token = CancellationToken::new();
        let err = search_tool(&s, TenantId::new(7), SessionScope::Power, &token, req)
            .expect_err("bm25 missing");
        assert_eq!(err.code(), -32004);
        match err {
            MCPError::IndexUnavailable(slug) => assert_eq!(slug, "bm25"),
            other => panic!("expected IndexUnavailable, got {other:?}"),
        }
    }

    #[test]
    fn search_tool_rejects_cross_tenant_request_with_unauthorized() {
        let s = StubSearcher::new(TenantId::new(7));
        let req = SearchRequest {
            tenant_id: 8,
            query: "x".into(),
            query_vec: None,
            k: Some(5),
            label_filter: None,
            ef_search: None,
            format: None,
            principal: None,
        };
        let token = CancellationToken::new();
        let err = search_tool(&s, TenantId::new(7), SessionScope::Power, &token, req)
            .expect_err("cross-tenant");
        assert_eq!(err.code(), -32002);
        assert!(matches!(err, MCPError::Unauthorized));
    }

    #[test]
    fn search_tool_rejects_k_over_cap() {
        let s = StubSearcher::new(TenantId::new(7));
        let req = SearchRequest {
            tenant_id: 7,
            query: "x".into(),
            query_vec: None,
            k: Some(MAX_SEARCH_K + 1),
            label_filter: None,
            ef_search: None,
            format: None,
            principal: None,
        };
        let token = CancellationToken::new();
        let err =
            search_tool(&s, TenantId::new(7), SessionScope::Power, &token, req).expect_err("k cap");
        assert_eq!(err.code(), -32602);
    }

    #[test]
    fn search_tool_rejects_k_zero() {
        let s = StubSearcher::new(TenantId::new(7));
        let req = SearchRequest {
            tenant_id: 7,
            query: "x".into(),
            query_vec: None,
            k: Some(0),
            label_filter: None,
            ef_search: None,
            format: None,
            principal: None,
        };
        let token = CancellationToken::new();
        let err = search_tool(&s, TenantId::new(7), SessionScope::Power, &token, req)
            .expect_err("k=0 reject");
        assert_eq!(err.code(), -32602);
    }

    #[test]
    fn search_tool_surfaces_cancelled_when_token_tripped() {
        let s = StubSearcher::new(TenantId::new(7)).with_hits(fixture_hits());
        let req = SearchRequest {
            tenant_id: 7,
            query: "x".into(),
            query_vec: None,
            k: Some(5),
            label_filter: None,
            ef_search: None,
            format: None,
            principal: None,
        };
        let token = CancellationToken::new();
        token.cancel();
        let err = search_tool(&s, TenantId::new(7), SessionScope::Power, &token, req)
            .expect_err("cancelled");
        assert_eq!(err.code(), -32001);
        assert!(matches!(err, MCPError::Cancelled));
    }

    #[test]
    fn search_tool_default_format_is_toon() {
        let s = StubSearcher::new(TenantId::new(7)).with_hits(fixture_hits());
        let req = SearchRequest {
            tenant_id: 7,
            query: "x".into(),
            query_vec: None,
            k: Some(2),
            label_filter: None,
            ef_search: None,
            format: None,
            principal: None,
        };
        let token = CancellationToken::new();
        let resp = search_tool(&s, TenantId::new(7), SessionScope::Power, &token, req).expect("ok");
        assert_eq!(resp["format"], "toon");
    }

    #[test]
    fn search_tool_default_k_is_ten() {
        // Pin DEFAULT_SEARCH_K=10 behavior: a request that omits `k`
        // bakes 10 into the result envelope.
        let mut many: Vec<SearchHit> = Vec::new();
        for i in 0..20 {
            many.push(SearchHit {
                node_id: i + 1,
                label: Some("Doc".into()),
                score: 1.0 - (i as f64) * 0.01,
            });
        }
        let s = StubSearcher::new(TenantId::new(7)).with_hits(many);
        let req = SearchRequest {
            tenant_id: 7,
            query: "x".into(),
            query_vec: None,
            k: None,
            label_filter: None,
            ef_search: None,
            format: Some(ResponseFormat::Json),
            principal: None,
        };
        let token = CancellationToken::new();
        let resp = search_tool(&s, TenantId::new(7), SessionScope::Power, &token, req).expect("ok");
        let body = resp["body"].as_str().unwrap();
        assert!(body.contains("\"k\":10"));
    }

    #[test]
    fn search_tool_applies_label_filter() {
        let s = StubSearcher::new(TenantId::new(7)).with_hits(fixture_hits());
        let req = SearchRequest {
            tenant_id: 7,
            query: "x".into(),
            query_vec: None,
            k: Some(10),
            label_filter: Some(vec!["Document".into()]),
            ef_search: None,
            format: Some(ResponseFormat::Json),
            principal: None,
        };
        let token = CancellationToken::new();
        let resp = search_tool(&s, TenantId::new(7), SessionScope::Power, &token, req).expect("ok");
        let body = resp["body"].as_str().unwrap();
        // Person hit dropped; Document hits retained.
        assert!(body.contains("\"node_id\":1"));
        assert!(body.contains("\"node_id\":2"));
        assert!(!body.contains("\"node_id\":3"), "Person filtered out");
    }

    #[test]
    fn search_tool_treats_empty_label_filter_as_no_filter() {
        // PR #292 review NIT-3 — pin the `Some(vec![])` empty-allowlist
        // semantics: per the SearchRequest::label_filter doc convention
        // ("Empty Vec = 'no filter'"), an empty allowlist MUST NOT
        // exclude any hits. All three fixture hits (2 Document + 1
        // Person) survive.
        let s = StubSearcher::new(TenantId::new(7)).with_hits(fixture_hits());
        let req = SearchRequest {
            tenant_id: 7,
            query: "x".into(),
            query_vec: None,
            k: Some(10),
            label_filter: Some(vec![]),
            ef_search: None,
            format: Some(ResponseFormat::Json),
            principal: None,
        };
        let token = CancellationToken::new();
        let resp = search_tool(&s, TenantId::new(7), SessionScope::Power, &token, req).expect("ok");
        let body = resp["body"].as_str().unwrap();
        assert!(body.contains("\"node_id\":1"), "Document #1 retained");
        assert!(body.contains("\"node_id\":2"), "Document #2 retained");
        assert!(
            body.contains("\"node_id\":3"),
            "Person retained (empty allowlist == no filter)"
        );
    }

    #[test]
    fn available_substrates_surfaces_cancelled_when_token_tripped() {
        // PR #292 review MED-1 + MED-2 — pin the new
        // `available_substrates(&self, tenant, &CancellationToken)`
        // contract: a tripped token short-circuits the gate-side
        // availability check with MCPError::Cancelled BEFORE the search
        // body runs. Without this, a slow production
        // `available_substrates` (substrate-handle / partition-router
        // lookup) would block until completion even after a per-request
        // cancel.
        let s = StubSearcher::new(TenantId::new(7)).with_hits(fixture_hits());
        let req = SearchRequest {
            tenant_id: 7,
            query: "x".into(),
            query_vec: None,
            k: Some(5),
            label_filter: None,
            ef_search: None,
            format: None,
            principal: None,
        };
        let token = CancellationToken::new();
        token.cancel();
        let err = search_tool(&s, TenantId::new(7), SessionScope::Power, &token, req)
            .expect_err("cancelled");
        assert_eq!(err.code(), -32001);
        assert!(matches!(err, MCPError::Cancelled));
    }

    #[test]
    fn search_request_rejects_unknown_field() {
        let v = serde_json::json!({
            "tenant_id": 1,
            "query": "x",
            "top_k": 5,  // typo of `k`
        });
        let res: Result<SearchRequest, _> = serde_json::from_value(v);
        assert!(res.is_err(), "typo must reject");
    }

    #[test]
    fn substrate_kinds_maps_to_index_kind() {
        let a = AvailableSubstrates {
            vector: true,
            bm25: true,
        };
        let kinds = substrate_kinds(a);
        assert!(kinds.contains(&IndexKind::Vector));
        assert!(kinds.contains(&IndexKind::Bm25));
        let none = substrate_kinds(AvailableSubstrates::none());
        assert!(none.is_empty());
    }

    // ─────────────────────────────────────────────────────────────────
    // ADR-212 §D-4 Seam-1 — principal-scoped enforcement
    // ─────────────────────────────────────────────────────────────────

    use arcgraph_storage::permissions::PUBLIC_PRINCIPAL;
    use std::collections::BTreeSet;

    fn grants(items: &[&str]) -> BTreeSet<String> {
        items.iter().map(|s| (*s).to_owned()).collect()
    }

    /// Index over the unit corpus: 1=alice-only, 2=bob-only, 3=PUBLIC,
    /// node 4 deliberately NEVER tagged (UNCLASSIFIED).
    fn unit_corpus_index() -> Arc<PermissionIndex> {
        let idx = PermissionIndex::new();
        idx.apply_doc_acl(arcgraph_core::NodeId::new(1), grants(&["alice"]));
        idx.apply_doc_acl(arcgraph_core::NodeId::new(2), grants(&["bob"]));
        idx.apply_doc_acl(arcgraph_core::NodeId::new(3), grants(&[PUBLIC_PRINCIPAL]));
        Arc::new(idx)
    }

    /// Four hits the substrate legs can ALL find: the restricted ones
    /// outrank the rest (the adversarial ordering for the filter).
    fn unit_corpus_hits() -> Vec<SearchHit> {
        vec![
            SearchHit {
                node_id: 1,
                label: Some("Document".into()),
                score: 0.99,
            },
            SearchHit {
                node_id: 2,
                label: Some("Document".into()),
                score: 0.88,
            },
            SearchHit {
                node_id: 3,
                label: Some("Document".into()),
                score: 0.77,
            },
            SearchHit {
                node_id: 4,
                label: Some("Document".into()),
                score: 0.66,
            },
        ]
    }

    fn principal_req(principal: &str, k: u32) -> SearchRequest {
        SearchRequest {
            tenant_id: 7,
            query: "shared keyword".into(),
            query_vec: None,
            k: Some(k),
            label_filter: None,
            ef_search: None,
            format: Some(ResponseFormat::Json),
            principal: Some(principal.into()),
        }
    }

    fn hits_of(resp: &serde_json::Value) -> Vec<u64> {
        let body: SearchResult =
            serde_json::from_str(resp["body"].as_str().expect("json body")).expect("result body");
        body.hits.iter().map(|h| h.node_id).collect()
    }

    #[test]
    fn principal_scoped_search_filters_restricted_and_untagged_hits() {
        let s = StubSearcher::new(TenantId::new(7))
            .with_hits(unit_corpus_hits())
            .with_permissions(unit_corpus_index());
        let token = CancellationToken::new();

        // Bob: alice-only (1) and UNCLASSIFIED (4) are invisible; his
        // own doc (2) + PUBLIC (3) survive, order preserved.
        let resp = search_tool(
            &s,
            TenantId::new(7),
            SessionScope::Power,
            &token,
            principal_req("bob", 10),
        )
        .expect("ok");
        assert_eq!(hits_of(&resp), vec![2, 3]);

        // Alice (positive control): sees 1 + 3, never bob's 2, never
        // the untagged 4.
        let resp = search_tool(
            &s,
            TenantId::new(7),
            SessionScope::Power,
            &token,
            principal_req("alice", 10),
        )
        .expect("ok");
        assert_eq!(hits_of(&resp), vec![1, 3]);

        // A principal the index has never seen: PUBLIC only.
        let resp = search_tool(
            &s,
            TenantId::new(7),
            SessionScope::Power,
            &token,
            principal_req("mallory", 10),
        )
        .expect("ok");
        assert_eq!(hits_of(&resp), vec![3]);
    }

    #[test]
    fn principal_less_request_is_unfiltered_system_trusted_path() {
        // ADR-212 §D-1: `principal: None` is the system-trusted path —
        // byte-identical to pre-ADR-212 behavior even when the searcher
        // HAS a permission index. Per #1293 the trusted path requires
        // the EXPLICIT power-scope marker (asserted here); the
        // absent-principal + non-power combination fails closed
        // (`absent_principal_non_power_session_fails_closed_1293`).
        let s = StubSearcher::new(TenantId::new(7))
            .with_hits(unit_corpus_hits())
            .with_permissions(unit_corpus_index());
        let req = SearchRequest {
            principal: None,
            ..principal_req("ignored", 10)
        };
        let token = CancellationToken::new();
        let resp = search_tool(&s, TenantId::new(7), SessionScope::Power, &token, req).expect("ok");
        assert_eq!(hits_of(&resp), vec![1, 2, 3, 4]);
        // And exactly ONE substrate fetch at the caller's k (no
        // over-fetch on the principal-less arm).
        assert_eq!(s.fetches(), vec![10]);
    }

    // ─────────────────────────────────────────────────────────────────
    // #1293 — absent-principal fail-closed gate (fail-OPEN → fail-CLOSED)
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn absent_principal_non_power_session_fails_closed_1293() {
        // #1293 RED-on-revert pin: an ABSENT principal on a non-power
        // session MUST refuse with Forbidden (-32008) — NOT run the
        // unfiltered SYSTEM-TRUSTED path. Reverting the gate returns
        // the full unfiltered corpus (including alice-only + untagged
        // hits) to a session holding no trusted marker.
        let s = StubSearcher::new(TenantId::new(7))
            .with_hits(unit_corpus_hits())
            .with_permissions(unit_corpus_index());
        let req = SearchRequest {
            principal: None,
            ..principal_req("ignored", 10)
        };
        let token = CancellationToken::new();
        let err = search_tool(&s, TenantId::new(7), SessionScope::Read, &token, req)
            .expect_err("fail-closed: absent principal on read scope");
        assert_eq!(err.code(), -32008);
        match &err {
            MCPError::Forbidden { required_scope } => {
                assert_eq!(*required_scope, SessionScope::Power.slug());
            }
            other => panic!("expected Forbidden, got {other:?}"),
        }
        // The searcher body MUST NOT have been invoked — the gate
        // fires before any substrate probe.
        assert_eq!(s.fetches(), Vec::<u32>::new(), "no substrate fetch");
    }

    #[test]
    fn absent_principal_fails_closed_for_read_scope_1293() {
        // A read-scoped request may not enter the principal-less,
        // system-trusted path.
        let token = CancellationToken::new();
        let scope = SessionScope::Read;
        let s = StubSearcher::new(TenantId::new(7)).with_hits(unit_corpus_hits());
        let req = SearchRequest {
            principal: None,
            ..principal_req("ignored", 10)
        };
        let err = search_tool(&s, TenantId::new(7), scope, &token, req)
            .expect_err("fail-closed for non-power scope");
        assert_eq!(err.code(), -32008, "scope {scope:?} must refuse");
        assert_eq!(
            s.fetches(),
            Vec::<u32>::new(),
            "scope {scope:?} leaked a fetch"
        );
    }

    #[test]
    fn principal_scoped_request_admits_and_filters_at_read_scope_1293() {
        // The end-user path: a READ-scope session carrying a principal
        // is admitted and filtered exactly as before — the #1293 gate
        // constrains only the principal-LESS arm. (Power scope is not
        // required to search on-behalf-of a principal.)
        let s = StubSearcher::new(TenantId::new(7))
            .with_hits(unit_corpus_hits())
            .with_permissions(unit_corpus_index());
        let token = CancellationToken::new();
        let resp = search_tool(
            &s,
            TenantId::new(7),
            SessionScope::Read,
            &token,
            principal_req("bob", 10),
        )
        .expect("principal-scoped read-scope request admits");
        assert_eq!(hits_of(&resp), vec![2, 3], "bob sees his doc + PUBLIC only");
    }

    #[test]
    fn cross_tenant_rejects_before_the_1293_scope_gate() {
        // Ordering pin (the graph.raw_query discipline, ADR-004
        // amendment-03): a cross-tenant probe on a read-scope session
        // surfaces Unauthorized (-32002), NOT Forbidden (-32008) — the
        // guard must not leak scope information cross-tenant.
        let s = StubSearcher::new(TenantId::new(7));
        let req = SearchRequest {
            tenant_id: 8,
            query: "x".into(),
            query_vec: None,
            k: Some(5),
            label_filter: None,
            ef_search: None,
            format: None,
            principal: None,
        };
        let token = CancellationToken::new();
        let err = search_tool(&s, TenantId::new(7), SessionScope::Read, &token, req)
            .expect_err("cross-tenant");
        assert_eq!(
            err.code(),
            -32002,
            "cross-tenant guard fires before scope gate"
        );
    }

    #[test]
    fn principal_scoped_search_fails_loud_without_permission_index() {
        // The trait default returns Ok(None): a principal-scoped
        // request against it MUST reject (never serve unfiltered).
        let s = StubSearcher::new(TenantId::new(7)).with_hits(unit_corpus_hits());
        let token = CancellationToken::new();
        let err = search_tool(
            &s,
            TenantId::new(7),
            SessionScope::Power,
            &token,
            principal_req("bob", 10),
        )
        .expect_err("fail-closed");
        assert_eq!(err.code(), -32004);
        match err {
            MCPError::IndexUnavailable(msg) => {
                assert!(
                    msg.starts_with(PERMISSION_INDEX_SLUG),
                    "slug-routable message, got: {msg}"
                );
            }
            other => panic!("expected IndexUnavailable, got {other:?}"),
        }
    }

    #[test]
    fn empty_principal_rejects_as_invalid_params() {
        let s = StubSearcher::new(TenantId::new(7))
            .with_hits(unit_corpus_hits())
            .with_permissions(unit_corpus_index());
        let token = CancellationToken::new();
        let err = search_tool(
            &s,
            TenantId::new(7),
            SessionScope::Power,
            &token,
            principal_req("", 10),
        )
        .expect_err("empty principal");
        assert_eq!(err.code(), -32602);
    }

    #[test]
    fn forbidden_is_byte_indistinguishable_from_not_found() {
        // ADR-212 §D-4 / §D-8(d) error-shape pin: a query whose every
        // match is restricted returns the SAME bytes as a query that
        // matches nothing — no existence oracle on the denied path.
        let only_alices = vec![
            SearchHit {
                node_id: 1,
                label: Some("Document".into()),
                score: 0.99,
            },
            SearchHit {
                node_id: 4,
                label: Some("Document".into()),
                score: 0.66,
            },
        ];
        let token = CancellationToken::new();

        let s_restricted = StubSearcher::new(TenantId::new(7))
            .with_hits(only_alices)
            .with_permissions(unit_corpus_index());
        let forbidden = search_tool(
            &s_restricted,
            TenantId::new(7),
            SessionScope::Power,
            &token,
            principal_req("bob", 10),
        )
        .expect("forbidden-as-empty");

        let s_empty = StubSearcher::new(TenantId::new(7))
            .with_hits(vec![])
            .with_permissions(unit_corpus_index());
        let not_found = search_tool(
            &s_empty,
            TenantId::new(7),
            SessionScope::Power,
            &token,
            principal_req("bob", 10),
        )
        .expect("no matches");

        assert_eq!(
            serde_json::to_string(&forbidden).expect("ser"),
            serde_json::to_string(&not_found).expect("ser"),
            "forbidden ≡ not-found: response envelopes must be byte-identical"
        );
    }

    #[test]
    fn overfetch_refill_fills_k_when_top_hits_are_restricted() {
        // 8 alice-ranked-first docs + 2 bob docs at the tail. Bob asks
        // k=2: round 1 fetches k×4=8 → zero visible; round 2 widens to
        // k×16=32 → the 2 bob docs surface. Refill is bounded at 2
        // substrate calls.
        let idx = PermissionIndex::new();
        let mut hits = Vec::new();
        for n in 1..=8u64 {
            idx.apply_doc_acl(arcgraph_core::NodeId::new(n), grants(&["alice"]));
            hits.push(SearchHit {
                node_id: n,
                label: Some("Document".into()),
                score: 1.0 - (n as f64) * 0.01,
            });
        }
        for n in 9..=10u64 {
            idx.apply_doc_acl(arcgraph_core::NodeId::new(n), grants(&["bob"]));
            hits.push(SearchHit {
                node_id: n,
                label: Some("Document".into()),
                score: 0.5 - (n as f64) * 0.01,
            });
        }
        let s = StubSearcher::new(TenantId::new(7))
            .with_hits(hits)
            .with_permissions(Arc::new(idx));
        let token = CancellationToken::new();
        let resp = search_tool(
            &s,
            TenantId::new(7),
            SessionScope::Power,
            &token,
            principal_req("bob", 2),
        )
        .expect("refill finds bob's docs");
        assert_eq!(hits_of(&resp), vec![9, 10]);
        assert_eq!(s.fetches(), vec![8, 32], "k×4 then k×16, exactly 2 rounds");
    }

    #[test]
    fn overfetch_stops_after_one_round_when_substrate_exhausted() {
        // Only 3 hits exist, all restricted to alice. Bob asks k=2:
        // round 1 fetches 8, gets 3 (< 8 ⇒ exhausted) → NO second
        // call; result is EMPTY (under-fill, never unfiltered).
        let s = StubSearcher::new(TenantId::new(7))
            .with_hits(unit_corpus_hits()[..1].to_vec())
            .with_permissions(unit_corpus_index());
        let token = CancellationToken::new();
        let resp = search_tool(
            &s,
            TenantId::new(7),
            SessionScope::Power,
            &token,
            principal_req("bob", 2),
        )
        .expect("empty, not error");
        assert_eq!(hits_of(&resp), Vec::<u64>::new());
        assert_eq!(
            s.fetches(),
            vec![8],
            "exhausted substrate ⇒ no refill round"
        );
    }

    #[test]
    fn overfetch_single_round_when_first_fetch_fills_k() {
        // Bob's docs are visible inside the first k×4 window: exactly
        // one substrate call.
        let s = StubSearcher::new(TenantId::new(7))
            .with_hits(unit_corpus_hits())
            .with_permissions(unit_corpus_index());
        let token = CancellationToken::new();
        let resp = search_tool(
            &s,
            TenantId::new(7),
            SessionScope::Power,
            &token,
            principal_req("bob", 1),
        )
        .expect("ok");
        assert_eq!(hits_of(&resp), vec![2]);
        assert_eq!(s.fetches(), vec![4], "k×4 once; fill reached");
    }

    #[test]
    fn principal_scoped_search_respects_label_filter_in_refill_accounting() {
        // Visibility AND label predicates both reduce fill; the loop
        // must account for both (a visible-but-wrong-label hit does
        // not satisfy k).
        let idx = PermissionIndex::new();
        idx.apply_doc_acl(arcgraph_core::NodeId::new(1), grants(&["bob"]));
        idx.apply_doc_acl(arcgraph_core::NodeId::new(2), grants(&["bob"]));
        let hits = vec![
            SearchHit {
                node_id: 1,
                label: Some("Person".into()),
                score: 0.9,
            },
            SearchHit {
                node_id: 2,
                label: Some("Document".into()),
                score: 0.8,
            },
        ];
        let s = StubSearcher::new(TenantId::new(7))
            .with_hits(hits)
            .with_permissions(Arc::new(idx));
        let mut req = principal_req("bob", 1);
        req.label_filter = Some(vec!["Document".into()]);
        let token = CancellationToken::new();
        let resp = search_tool(&s, TenantId::new(7), SessionScope::Power, &token, req).expect("ok");
        assert_eq!(
            hits_of(&resp),
            vec![2],
            "visible Person hit must not satisfy k"
        );
    }

    // ─────────────────────────────────────────────────────────────────
    // ADR-212 §D-8(c) — enforcement-coverage pin
    // ─────────────────────────────────────────────────────────────────

    /// ADR-212 §D-6 + §D-7 enforcement-coverage pin (§D-8(c), the
    /// `cross_crate_boundary.rs` pin pattern applied to enforcement).
    ///
    /// Every wire-callable method in the live catalog
    /// ([`crate::transport::KNOWN_METHODS`]) MUST be exactly one of:
    ///
    /// - **ENFORCED** — threads `principal` through ADR-212 visibility
    ///   enforcement at every node-content response boundary;
    /// - **SCOPED-BY-SUBTRACTION** — documented un-enforced surface
    ///   that deployment profiles MUST keep admin-/service-scoped for
    ///   end-user principals until its named ADR-212 §D-7 stage lands.
    ///
    /// A NEW retrieval tool that is neither enforced nor classified
    /// here fails this pin. The exclusion list is ITSELF part of the
    /// pinned set (#1104 R1 NIT-3): growing it without amending
    /// ADR-212 fails the pin — every entry cites its §D-7 stage (or
    /// names the non-retrieval rationale).
    #[test]
    fn adr212_coverage_pin() {
        const ENFORCED: &[&str] = &["graph.inspect", "graph.explore", "graph.search"];
        // (method, disposition) — keep in lockstep with ADR-212 §D-6 /
        // §D-7. Dispositions are doc-strings, asserted non-empty so a
        // lazy addition cannot silently classify.
        const SCOPED_BY_SUBTRACTION: &[(&str, &str)] = &[
            (
                "graph.schema",
                "S11 metadata stance: tenant-shared at stages 1-3; stage-4 decision",
            ),
            (
                "graph.ingest",
                "write surface — retrieval enforcement N/A; write-authz is engine RBAC (ADR-011)",
            ),
            (
                "graph.raw_query",
                "S1 ArcQL; executor decorator at §D-7 stage-2 (already power-scope-gated per ADR-004-am-03)",
            ),
        ];

        let enforced: std::collections::BTreeSet<&str> = ENFORCED.iter().copied().collect();
        let scoped: std::collections::BTreeMap<&str, &str> =
            SCOPED_BY_SUBTRACTION.iter().copied().collect();

        // Disjoint.
        for m in &enforced {
            assert!(
                !scoped.contains_key(m),
                "{m} is both ENFORCED and SCOPED_BY_SUBTRACTION"
            );
        }
        // Exact cover of the live catalog, both directions.
        for m in crate::transport::KNOWN_METHODS {
            assert!(
                enforced.contains(m) || scoped.contains_key(m),
                "live tool `{m}` is neither ADR-212-enforced nor documented as \
                 scoped-by-subtraction — classify it (and amend ADR-212) before shipping"
            );
        }
        for m in enforced.iter().chain(scoped.keys()) {
            assert!(
                crate::transport::KNOWN_METHODS.contains(m),
                "pinned method `{m}` is no longer in the live catalog — prune the pin"
            );
        }
        for (m, why) in &scoped {
            assert!(
                !why.trim().is_empty(),
                "scoped-by-subtraction entry `{m}` must cite its stage/rationale"
            );
        }
    }
}
