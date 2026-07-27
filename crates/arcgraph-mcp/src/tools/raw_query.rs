//! W16ζ M5-11 — `graph.raw_query` Tier-2 power-user MCP tool.
//!
//! Direct ArcQL execution per design-v2 §9.2 entry 7:
//! > "execute an ArcQL query directly. Requires elevated permission
//! > (OAuth scope `arcgraph.power`)".
//!
//! Per ADR-004 §"Tier 2" entry 7 + ADR-004 amendment-03 §D-1 this is
//! the v1.0-alpha power-user escape hatch — the canonical extension
//! path design-v2 §15 line 924 names ("The extension path is
//! `graph.raw_query` (Tier 2, scoped). Agents with `arcgraph.power`
//! scope can run arbitrary ArcQL").
//!
//! # Surface seam — `RawQueryExecutor`
//!
//! Following the W13δ M5-04 / M5-05 + W14β M5-06 / M5-07 pattern: the
//! MCP layer defines a local adapter trait ([`RawQueryExecutor`])
//! rather than reaching across the arcgraph-query bounded-context to
//! bind directly to [`arcgraph_query::QueryEngine`]. Production wiring
//! at M4-08+ implements this trait on a per-tenant catalog + substrate
//! handle by routing through
//! [`arcgraph_query::QueryEngine::execute_with_deadline`].
//!
//! `feedback_avoid_speculative_scaffolding.md` applies: define the
//! trait at first consumer (this tool), avoid speculatively extending
//! arcgraph-query with a "raw query convenience" API that has no other
//! consumer.
//!
//! # Scope-gating contract
//!
//! Per design-v2 §9.5 JD stress test (line 682):
//! > "Agent tries `graph.raw_query(\"DROP EVERYTHING\")`. Mitigation:
//! > `raw_query` requires `arcgraph.power` scope; without it, the call
//! > is rejected at transport."
//!
//! The dispatcher passes a [`crate::SessionScope`] to
//! [`raw_query_tool`]; non-power sessions reject with
//! [`crate::MCPError::Forbidden`] (-32008) BEFORE any executor call.
//! At v1.0-alpha the scope is a stub (M5-03 OAuth is forward); the
//! W16ζ integration test exercises both paths (power session accepts,
//! read session rejects).
//!
//! # Defense-in-depth caps
//!
//! Per `feedback_security_class_first_network_surface.md`:
//!
//! 1. **Query bytes cap**: [`MAX_RAW_QUERY_BYTES`] = 1 MiB. Defense-
//!    in-depth on top of [`crate::MAX_MESSAGE_BYTES`] = 16 MiB (the
//!    JSON-RPC envelope-level cap). Oversized queries reject as
//!    [`crate::MCPError::InvalidParams`] (-32602) BEFORE the parser
//!    body runs.
//! 2. **Row cap**: [`MAX_RAW_QUERY_MAX_ROWS`] = 10_000. The caller's
//!    `max_rows` is clamped; result envelopes set `truncated: true`
//!    when the executor's row stream exceeded `max_rows`. This pins
//!    the memory-budget floor — a power-user query returning 10M rows
//!    cannot OOM the JSON-RPC writer.
//! 3. **Cross-tenant guard**: `request.tenant_id == session_tenant`
//!    check runs BEFORE the scope check + executor call. Same shape
//!    as the W13δ / W14β / W14γ Tier-1 tools.
//!
//! # Snapshot-LSN + cancellation
//!
//! Inherited from the [`RawQueryExecutor`] impl. Production wires
//! `QueryEngine::execute_with_deadline` which:
//! - acquires the snapshot LSN at execute-time before first-batch pull
//!   (per ADR-038 amendment-03 §TIER-1 GAP E rule 1);
//! - honors the per-query deadline + cancellation token (per M4-92).
//!
//! # Write-op exposure (ADR-153 W27-β)
//!
//! Per ADR-153 §D-1 the tool ADMITS any ArcQL clause that parses,
//! including the 5 W26-θ write-op clauses (CREATE node, CREATE rel,
//! DELETE / DETACH DELETE, SET / REMOVE, MERGE) wired through
//! ADR-147..151. The tool does NOT inspect query content at the MCP
//! layer; the parser + binder + type-check pipeline owns clause-level
//! admission. Write-op results return through the same `RawQueryRows`
//! envelope, with the post-W27-β [`WriteSummary`] field summarizing
//! side-effects per openCypher v9 conventions (1 statement = 1 tx;
//! commit-or-rollback per ADR-031 + ADR-033). The forward-deferred
//! shapes (read-only opt-in flag; multi-statement batching; streaming
//! write results; explicit txn boundary control) land at v1.1+ per
//! ADR-153 §"Forward-deferred".
//!
//! # ADR provenance
//! - **ADR-004 §"Tier 2 (power-user, scoped)" entry 7** — direct ArcQL
//!   execution; requires `arcgraph.power` scope.
//! - **ADR-004 amendment-03 §D-1, §D-2, §D-3, §D-4** — v1.0-alpha
//!   wire shape, stub-auth posture, `Forbidden` error variant,
//!   forward-deferred `params` slot.
//! - **ADR-153 §D-1..§D-7** — W27-β MCP `graph.raw_query` write-op
//!   contract: admission + envelope shape + tx lifecycle + tenant
//!   scoping + v1.0-α posture inheritance from ADR-152 sister.
//! - **ADR-147..151** — the 5 W26-θ executor-layer write-op ADRs the
//!   raw_query tool exposes (CREATE node / CREATE rel / DELETE / SET +
//!   REMOVE / MERGE).
//! - **ADR-031 + ADR-033** — per-tenant `Transaction` discipline
//!   (commit-or-rollback at the executor's substrate boundary; ADR-153
//!   §D-5 inherits this for the raw_query write-op path).
//! - **design-v2 §9.2 entry 7, §9.4, §9.5** — canonical raw_query
//!   shape, scope set, security contract.
//! - **design-v2 §15 line 924** — power-user pattern justification.
//! - **`docs/roadmap.md` M5-11** — "Implement Tier-2 tool:
//!   `graph.raw_query` (power scope only) | arcgraph-mcp | M | M4,
//!   M5-03 | Security test: rejected without `arcgraph.power` scope".
//! - **ADR-038 amendment-03 §M5↔M4 contract surface** — the executor
//!   surface raw_query binds to via the [`RawQueryExecutor`] adapter.

use arcgraph_core::TenantId;
use arcgraph_query::CancellationToken;
use serde::{Deserialize, Serialize};

use crate::error::MCPError;
use crate::scope::SessionScope;
use crate::tools::ResponseFormat;

// ─────────────────────────────────────────────────────────────────────
// Caps
// ─────────────────────────────────────────────────────────────────────

/// Maximum query-string byte length admitted by `graph.raw_query`.
///
/// Defense-in-depth on top of the envelope-level
/// [`crate::MAX_MESSAGE_BYTES`] = 16 MiB cap per
/// `feedback_security_class_first_network_surface.md`. The 1 MiB
/// per-query cap matches the typical query-length distribution (LDBC
/// SNB Interactive-Short queries are <500 bytes; pathological queries
/// above 1 MiB are almost certainly an attack vector or a generator
/// bug). Per ADR-004 amendment-03 §D-1 point 2.
pub const MAX_RAW_QUERY_BYTES: usize = 1024 * 1024;

/// Default row cap when [`RawQueryRequest::max_rows`] is omitted.
///
/// 1000 rows matches the typical Cypher REPL use-case (Neo4j Browser
/// default). Callers needing more rows set `max_rows` up to
/// [`MAX_RAW_QUERY_MAX_ROWS`].
pub const DEFAULT_RAW_QUERY_MAX_ROWS: u32 = 1000;

/// Hard cap on [`RawQueryRequest::max_rows`].
///
/// 10_000 rows is the v1.0-alpha memory-budget floor — a power-user
/// query returning more rows is forced to paginate or use the streaming
/// surface (forward-pinned to v1.1+ per ADR-004 amendment-03 forward-
/// amendment-hooks §4). Per ADR-004 amendment-03 §D-1 point 3.
pub const MAX_RAW_QUERY_MAX_ROWS: u32 = 10_000;

// ─────────────────────────────────────────────────────────────────────
// Trait surface
// ─────────────────────────────────────────────────────────────────────

/// Adapter trait read by the [`raw_query_tool`] entry point.
///
/// Implementations live OUTSIDE this crate: tests stub it in-line;
/// production wiring at M4-08+ implements it on a per-tenant catalog +
/// substrate handle, routing through
/// [`arcgraph_query::QueryEngine::execute_with_deadline`] (which honors
/// the M4-92 deadline + cancellation + ADR-038 amendment-03 §TIER-1
/// GAP E rule 1 snapshot-LSN contract).
///
/// # Per-tenant scoping
///
/// `tenant: TenantId` parameter matches the sibling tool traits. The
/// MCP layer has already enforced `tenant == session_tenant` BEFORE
/// invoking the executor; the executor MAY still re-check defensively
/// but the cross-tenant rejection rule is owned by the MCP layer.
///
/// # `Send + Sync`
///
/// MCP transport runs on a tokio runtime; the executor must be
/// shareable across awaits.
///
/// # Cancellation contract
///
/// Production impls MUST honor the cancellation token at batch
/// boundaries and short-circuit with
/// [`crate::MCPError::Cancelled`] (-32001) on a tripped token.
pub trait RawQueryExecutor: Send + Sync {
    /// Execute `query` on `tenant`, returning materialized rows.
    ///
    /// Impls MUST NOT return more than `max_rows` rows; the MCP layer
    /// defensively truncates anyway as a belt-and-suspenders pin. If
    /// the executor's row stream exceeds `max_rows`, set
    /// `RawQueryRows::truncated = true`.
    fn execute(
        &self,
        tenant: TenantId,
        query: &str,
        max_rows: u32,
        cancel: &CancellationToken,
    ) -> Result<RawQueryRows, MCPError>;

    /// Build the QUERY PLAN for `query` on `tenant` WITHOUT executing it,
    /// returning the plan tree serialized into the [`RawQueryRows`] wire
    /// shape (one row per plan operator; columns
    /// `[op, details, estimated_cost, estimated_card, depth]` — the #952
    /// plan-row adapter shape).
    ///
    /// This backs the `graph.raw_query` `explain:true` verb-consolidation
    /// mode (operator-ruled — keeps the ADR-004 10-tool cap, no separate
    /// `graph.explain` tool). The production impl routes through the free
    /// `arcgraph_query::explain` fn, which runs only the
    /// parse → bind → type-check → lower → cost pipeline (no snapshot LSN,
    /// no substrate I/O, per ADR-038 §2 D-18 rule 1) — so `explain` is
    /// side-effect-free even for write-op clauses.
    ///
    /// # Default impl
    ///
    /// Returns [`MCPError::MethodNotFound`] (-32601) so test / fixture
    /// stubs that don't model plan introspection don't all break when
    /// this method lands. The PRODUCTION executor
    /// (`StorageRawQueryExecutor`) OVERRIDES it.
    fn explain(&self, tenant: TenantId, query: &str) -> Result<RawQueryRows, MCPError> {
        let _ = (tenant, query);
        Err(MCPError::MethodNotFound(
            "graph.raw_query: explain mode not supported by this executor".into(),
        ))
    }
}

/// Write-effect summary returned with every [`RawQueryRows`] envelope
/// per ADR-153 §D-2 (W27-β).
///
/// Each counter tracks side-effects produced by the executor during
/// this single `graph.raw_query` invocation, aggregated per openCypher
/// v9 conventions. A pure-read query (e.g.,
/// `MATCH (n) RETURN n`) returns the zero value
/// ([`WriteSummary::is_empty`] = `true`). A pure-write query
/// (`CREATE (n:User)`) returns non-zero counters with an empty rows
/// slot. A read-write composition (`CREATE (n) RETURN n`) returns
/// BOTH the writes counters and the RETURN rows.
///
/// # Counter semantics
///
/// - `nodes_created` — one per `ExecutorSubstrate::create_node` call.
/// - `nodes_deleted` — one per `ExecutorSubstrate::delete_node` call
///   (the tombstone counts once even when `detach = true` cascades
///   through attached rels; those attached rels increment
///   `rels_deleted` separately).
/// - `rels_created` — one per `ExecutorSubstrate::create_rel` call.
/// - `rels_deleted` — one per `ExecutorSubstrate::delete_rel` call
///   (including the `DETACH DELETE` cascade through attached rels per
///   ADR-149 §D-3).
/// - `properties_set` — number of property keys written by every
///   `SET` clause variant per ADR-150 §D-4:
///   `PropertyAssign` → 1; `PropertyReplace(N)` → N;
///   `PropertyMerge(N)` → N. Reflects KEYS touched, not bytes.
/// - `properties_removed` — number of property keys removed per
///   `REMOVE`. `RemoveNodeMutation::Property` → 1.
/// - `labels_added` — number of label NAMES added per
///   `SET n:L1:L2:...` (one count per name; at v1.0-α the production
///   substrate surfaces `IndexUnavailable` per ADR-150 §D-9 — the
///   counter still increments on the in-memory test substrate so the
///   wire contract is testable end-to-end).
/// - `labels_removed` — number of label NAMES removed per
///   `REMOVE n:L1:L2`. Same v1.0-α posture as `labels_added`.
///
/// # `[Eq]` + `[Copy]`
///
/// The struct carries 8 `u64`s; both `Eq` and `Copy` are derived.
/// Renderers reach `WriteSummary` by value in formatting helpers; the
/// integration tests assert equality against literal values.
///
/// # ADR provenance
/// - **ADR-153 §D-2** — wire shape (8 counters; `writes`-keyed entry on
///   `RawQueryRows`).
/// - **ADR-147..151 §D-x §"Counting semantics"** — per-clause counting
///   rules each ADR ratified in W26-θ; ADR-153 cross-references and
///   composes.
/// - **openCypher v9 §"Statistics"** — naming convention adopted
///   (Neo4j Browser uses the same 8 names).
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct WriteSummary {
    /// Nodes created via `CREATE (n[:Label] {props})` per ADR-147.
    pub nodes_created: u64,
    /// Nodes tombstoned via `DELETE n` / `DETACH DELETE n` per ADR-149.
    pub nodes_deleted: u64,
    /// Relationships created via `CREATE (a)-[r:TYPE {props}]->(b)` per
    /// ADR-148.
    pub rels_created: u64,
    /// Relationships tombstoned via `DELETE r` OR the DETACH cascade of
    /// `DETACH DELETE n` per ADR-149.
    pub rels_deleted: u64,
    /// Property keys written via `SET n.k = v` / `SET n = {...}` /
    /// `SET n += {...}` per ADR-150.
    pub properties_set: u64,
    /// Property keys removed via `REMOVE n.k` per ADR-150.
    pub properties_removed: u64,
    /// Label names added via `SET n:L1:L2:...` per ADR-150 §D-9
    /// posture.
    pub labels_added: u64,
    /// Label names removed via `REMOVE n:L1:L2:...` per ADR-150 §D-9
    /// posture.
    pub labels_removed: u64,
}

impl WriteSummary {
    /// `true` when every counter is zero — the canonical
    /// "pure-read query" signal renderers consume to suppress the
    /// `writes:{...}` block from a TOON / YAML payload.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    /// Sum of every counter. Convenience for renderers that surface
    /// a single "N changes committed" line.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.nodes_created
            .saturating_add(self.nodes_deleted)
            .saturating_add(self.rels_created)
            .saturating_add(self.rels_deleted)
            .saturating_add(self.properties_set)
            .saturating_add(self.properties_removed)
            .saturating_add(self.labels_added)
            .saturating_add(self.labels_removed)
    }
}

/// Materialized result returned by [`RawQueryExecutor::execute`].
///
/// Each row is a heterogeneous tuple of JSON-encoded cells (the
/// caller's projection determines the shape). v1.0-alpha admits any
/// projection that ArcQL supports; the executor's
/// [`arcgraph_query::executor::Value`] is projected through
/// `Value::to_json_value` per the W13β M4-81 serializer bridge.
///
/// The post-W27-β `writes: WriteSummary` field summarizes side-effects
/// per ADR-153 §D-2 (zero for pure-read queries; non-zero for any of
/// the 5 W26-θ write-op clauses); the field is `#[serde(default)]` so
/// pre-W27-β client envelopes deserialize cleanly (forward-compat with
/// the cross-version wire contract under the code-quality policy).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RawQueryRows {
    /// Optional projected column names. v1.0-alpha production impls
    /// SHOULD populate this from the ArcQL `RETURN` clause's
    /// projection list; test stubs MAY leave it `None` (the wire shape
    /// still admits row data without columns).
    #[serde(default)]
    pub columns: Option<Vec<String>>,
    /// Materialized rows. Each row is a JSON array of cells.
    pub rows: Vec<serde_json::Value>,
    /// Number of rows in [`Self::rows`]. Convenience for MCP clients
    /// that want to surface a row-count without re-iterating the
    /// `rows` slot.
    pub row_count: usize,
    /// `true` when the executor's row stream exceeded `max_rows` and
    /// the result was truncated. Clients seeing `truncated: true`
    /// SHOULD re-run with a larger `max_rows` or paginate via a future
    /// streaming surface.
    pub truncated: bool,
    /// Write-effect counters per ADR-153 §D-2. Zero for pure-read
    /// queries; populated by the executor at substrate-write boundaries
    /// for any of the 5 W26-θ write-op clauses. `#[serde(default)]` so
    /// pre-W27-β client envelopes deserialize cleanly.
    #[serde(default)]
    pub writes: WriteSummary,
}

// ─────────────────────────────────────────────────────────────────────
// Request envelope
// ─────────────────────────────────────────────────────────────────────

/// Request params for the `graph.raw_query` Tier-2 tool.
///
/// `#[serde(deny_unknown_fields)]` under the strict public-contract policy — typo'd
/// fields (or pre-amendment `params` from a future-version client)
/// reject cleanly at deserialize-time rather than silently dropping.
///
/// Per ADR-004 amendment-03 §D-4 the `params` slot is forward-deferred
/// to a future amendment alongside the parser-side parameter slice;
/// v1.0-alpha admits parameter-less queries only.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RawQueryRequest {
    /// The tenant to execute the query within.
    pub tenant_id: u64,
    /// The ArcQL source string. v1.0-alpha admits the openCypher subset
    /// pinned by ADR-006 + ADR-038. Capped at [`MAX_RAW_QUERY_BYTES`].
    pub query: String,
    /// Optional row cap. Defaults to [`DEFAULT_RAW_QUERY_MAX_ROWS`];
    /// values above [`MAX_RAW_QUERY_MAX_ROWS`] reject as
    /// [`MCPError::InvalidParams`].
    #[serde(default)]
    pub max_rows: Option<u32>,
    /// Optional render-format hint. Defaults to JSON — raw query
    /// results are heterogeneous by nature (caller-arbitrary
    /// projections) and JSON is the most permissive renderer. TOON /
    /// YAML pivot through `serde_json::Value` per the W11ε serializer
    /// convention.
    #[serde(default)]
    pub format: Option<ResponseFormat>,
    /// When `true`, return the QUERY PLAN (the `EXPLAIN` plan tree) for
    /// `query` instead of EXECUTING it. The plan is serialized to the
    /// same [`RawQueryRows`] envelope, one row per plan operator, with
    /// columns `[op, details, estimated_cost, estimated_card, depth]`
    /// (the #952 plan-row adapter shape). No snapshot LSN is acquired
    /// and no substrate write/read runs — the plan is built by the
    /// parse → bind → type-check → lower → cost pipeline only (per
    /// ADR-038 §2 D-18 rule 1).
    ///
    /// This is the operator-ruled verb-consolidation of plan
    /// introspection INTO `graph.raw_query` (the roadmap §"Notes for
    /// engineering" #3 verb-discrimination precedent) — it keeps the
    /// ADR-004 10-tool cap (NO separate `graph.explain` tool is wired).
    /// `explain:true` inherits the same `arcgraph.power` scope gate as
    /// the execute path (a plan can leak schema / cardinality).
    ///
    /// Absent / `false` ⇒ the request executes exactly as before
    /// (`#[serde(deny_unknown_fields)]`-safe: a request that omits the
    /// field deserializes to `false`).
    #[serde(default)]
    pub explain: bool,
}

// ─────────────────────────────────────────────────────────────────────
// Tool entry point
// ─────────────────────────────────────────────────────────────────────

/// `graph.raw_query` — return raw-query rows as JSON-RPC `result`.
///
/// # Validation order
///
/// 1. **Cross-tenant guard** — request `tenant_id` must equal
///    `session_tenant`; otherwise [`MCPError::Unauthorized`] (-32002).
/// 2. **Scope check** — `session_scope` must admit power tools
///    (i.e. [`SessionScope::Power`]); otherwise [`MCPError::Forbidden`]
///    (-32008) with `data.required_scope = "arcgraph.power"`.
/// 3. **Query-bytes cap** — `query.len() > MAX_RAW_QUERY_BYTES` rejects
///    as [`MCPError::InvalidParams`] (-32602) BEFORE the executor body
///    runs. Per `feedback_security_class_first_network_surface.md`.
/// 4. **Empty-query guard** — empty `query` rejects as
///    [`MCPError::InvalidParams`] (an empty string is not valid
///    ArcQL; rejecting at the MCP layer surfaces a cleaner diagnostic
///    than the parser-side `ParseError`).
/// 5. **Explain-mode branch** — when `req.explain == true`, return the
///    QUERY PLAN (the `EXPLAIN` plan tree serialized into `RawQueryRows`)
///    via [`RawQueryExecutor::explain`] instead of executing. The scope
///    gate (step 2) already ran, so `explain:true` is Power-only just
///    like execute (a plan can leak schema / cardinality). This is the
///    operator-ruled verb-consolidation that keeps the ADR-004 10-tool
///    cap (NO separate `graph.explain` tool). The explain path builds
///    the plan side-effect-free (no snapshot LSN, no substrate I/O per
///    ADR-038 §2 D-18 rule 1), so the row-cap clamp (step 6) is
///    execute-only.
/// 6. **Row-cap clamp** — `max_rows > MAX_RAW_QUERY_MAX_ROWS` rejects
///    as [`MCPError::InvalidParams`].
/// 7. **Executor invocation** — passes through `cancel` so a SIGTERM
///    drain (M5-12 forward) trips the in-flight query at the next
///    batch boundary.
///
/// # Cancellation
///
/// `cancel` is the cancellation token bound to this request. The
/// dispatcher mints a fresh token per JSON-RPC request; production
/// MCP transport (M5-12 forward) shares the token with the SIGTERM
/// drain so an in-flight query trips at the next batch boundary.
///
/// # Errors
///
/// - [`MCPError::Unauthorized`] (-32002) — cross-tenant request.
/// - [`MCPError::Forbidden`] (-32008) — session scope insufficient.
/// - [`MCPError::InvalidParams`] (-32602) — oversized query, empty
///   query, or `max_rows` above the hard cap.
/// - [`MCPError::QueryError`] (-32005) — ArcQL parse / bind / type-
///   check error (surfaced via the executor's `ExecutionError::Plan`
///   → `MCPError::QueryError` mapping per W13δ codec-local error
///   translation).
/// - [`MCPError::Cancelled`] (-32001) — cancellation token tripped.
/// - [`MCPError::ExecutionEval`] (-32006) — runtime evaluation fault
///   (e.g., substrate I/O fault).
/// - [`MCPError::InternalError`] (-32603) — serializer encode failure.
pub fn raw_query_tool<R: RawQueryExecutor>(
    executor: &R,
    session_tenant: TenantId,
    session_scope: SessionScope,
    cancel: &CancellationToken,
    req: RawQueryRequest,
) -> Result<serde_json::Value, MCPError> {
    // Step 1: cross-tenant guard. Same shape as the sibling Tier-1
    // tools; rejects BEFORE the scope check so a cross-tenant probe
    // doesn't leak scope information.
    let request_tenant = TenantId::new(req.tenant_id);
    if request_tenant != session_tenant {
        return Err(MCPError::Unauthorized);
    }

    // Step 2: scope check. Per ADR-004 amendment-03 §D-1 + design-v2
    // §9.5 line 682 the power-tier admission is strict equality on
    // SessionScope::Power.
    if !session_scope.admits_power() {
        return Err(MCPError::Forbidden {
            required_scope: SessionScope::Power.slug(),
        });
    }

    // Step 3: query-bytes cap. Defense-in-depth on top of
    // crate::MAX_MESSAGE_BYTES. Per
    // feedback_security_class_first_network_surface.md.
    if req.query.len() > MAX_RAW_QUERY_BYTES {
        return Err(MCPError::InvalidParams(format!(
            "graph.raw_query: query length {} exceeds cap {MAX_RAW_QUERY_BYTES} bytes",
            req.query.len()
        )));
    }

    // Step 4: empty-query guard. Rejecting at the MCP layer surfaces a
    // cleaner -32602 diagnostic than the parser-side -32005
    // ParseError.
    if req.query.is_empty() {
        return Err(MCPError::InvalidParams(
            "graph.raw_query: query must be non-empty".into(),
        ));
    }

    // Step 5: explain-mode branch (operator-ruled verb-consolidation —
    // stays at the ADR-004 10-tool cap, NO separate graph.explain tool).
    // The scope gate (Step 2) already ran: explain:true is Power-only
    // just like execute (a plan tree can leak schema / cardinality). The
    // plan is built by the free `arcgraph_query::explain` fn through the
    // production executor's `explain` override — it acquires NO snapshot
    // LSN and contacts NO substrate (ADR-038 §2 D-18 rule 1), so the
    // row-cap clamp below is execute-only (a plan tree is bounded by the
    // query's operator count, not by data cardinality).
    if req.explain {
        let result = executor.explain(request_tenant, &req.query)?;
        let format = req.format.unwrap_or(ResponseFormat::Json);
        let value = serde_json::to_value(&result).map_err(|e| {
            MCPError::InternalError(format!("raw_query explain result serialize: {e}"))
        })?;
        return crate::tools::render_response(format, &value);
    }

    // Step 6: row-cap clamp.
    let max_rows = req.max_rows.unwrap_or(DEFAULT_RAW_QUERY_MAX_ROWS);
    if max_rows > MAX_RAW_QUERY_MAX_ROWS {
        return Err(MCPError::InvalidParams(format!(
            "graph.raw_query: max_rows={max_rows} exceeds hard cap {MAX_RAW_QUERY_MAX_ROWS}"
        )));
    }
    if max_rows == 0 {
        return Err(MCPError::InvalidParams(
            "graph.raw_query: max_rows must be ≥ 1".into(),
        ));
    }

    // Step 7: executor invocation. Production impl routes through
    // QueryEngine::execute_with_deadline which honors M4-92 deadline +
    // cancellation + ADR-038 amendment-03 §TIER-1 GAP E rule 1
    // snapshot-LSN contract.
    let mut result = executor.execute(request_tenant, &req.query, max_rows, cancel)?;

    // Defensive truncation: enforce `max_rows` at the MCP boundary so a
    // stub fixture (or a future production impl that returns more than
    // `max_rows` on a fast path) cannot violate the wire contract.
    if result.rows.len() > max_rows as usize {
        result.rows.truncate(max_rows as usize);
        result.truncated = true;
    }
    // Re-derive row_count from the (possibly truncated) rows slot so
    // the wire shape is internally consistent.
    result.row_count = result.rows.len();

    let format = req.format.unwrap_or(ResponseFormat::Json);
    let value = serde_json::to_value(&result)
        .map_err(|e| MCPError::InternalError(format!("raw_query result serialize: {e}")))?;
    crate::tools::render_response(format, &value)
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Stub executor: returns a caller-baked row list for a matching
    /// tenant + records the requested max_rows.
    ///
    /// Not `Clone` (MCPError is not `Clone` by design — payloads carry
    /// thiserror-derived stack info we don't want to duplicate); the
    /// test fixtures consume the stub by reference.
    #[derive(Debug)]
    struct StubExecutor {
        tenant: TenantId,
        rows: Vec<serde_json::Value>,
        columns: Option<Vec<String>>,
        force_error: Option<MCPError>,
        writes: WriteSummary,
        /// Plan rows the `explain` override returns. `None` ⇒ the
        /// default trait impl (MethodNotFound) is exercised.
        explain_rows: Option<Vec<serde_json::Value>>,
    }

    impl StubExecutor {
        fn new(tenant: TenantId) -> Self {
            Self {
                tenant,
                rows: vec![],
                columns: None,
                force_error: None,
                writes: WriteSummary::default(),
                explain_rows: None,
            }
        }
        fn with_rows(mut self, rows: Vec<serde_json::Value>) -> Self {
            self.rows = rows;
            self
        }
        /// Make the stub override `explain` to return a baked plan-row
        /// list (so the explain:true branch can be unit-tested without
        /// the production catalog / QueryEngine).
        fn with_explain_rows(mut self, rows: Vec<serde_json::Value>) -> Self {
            self.explain_rows = Some(rows);
            self
        }
        fn with_columns(mut self, cols: Vec<String>) -> Self {
            self.columns = Some(cols);
            self
        }
        fn with_forced_error(mut self, e: MCPError) -> Self {
            self.force_error = Some(e);
            self
        }
        fn with_writes(mut self, w: WriteSummary) -> Self {
            self.writes = w;
            self
        }
    }

    impl RawQueryExecutor for StubExecutor {
        fn execute(
            &self,
            tenant: TenantId,
            _query: &str,
            max_rows: u32,
            cancel: &CancellationToken,
        ) -> Result<RawQueryRows, MCPError> {
            if let Some(e) = &self.force_error {
                // Clone-of-Forbidden-with-static-str is cheap; for the
                // wider taxonomy this stub returns Internal as a
                // surrogate (test doesn't depend on payload fidelity).
                return Err(match e {
                    MCPError::Cancelled => MCPError::Cancelled,
                    MCPError::Unauthorized => MCPError::Unauthorized,
                    MCPError::QueryError(s) => MCPError::QueryError(s.clone()),
                    MCPError::ExecutionEval(s) => MCPError::ExecutionEval(s.clone()),
                    MCPError::Forbidden { required_scope } => {
                        MCPError::Forbidden { required_scope }
                    }
                    other => MCPError::InternalError(format!("forced: {other}")),
                });
            }
            if cancel.is_cancelled() {
                return Err(MCPError::Cancelled);
            }
            if tenant != self.tenant {
                return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
            }
            let mut emitted = self.rows.clone();
            let truncated = emitted.len() > max_rows as usize;
            if truncated {
                emitted.truncate(max_rows as usize);
            }
            let row_count = emitted.len();
            Ok(RawQueryRows {
                columns: self.columns.clone(),
                rows: emitted,
                row_count,
                truncated,
                writes: self.writes,
            })
        }

        fn explain(&self, tenant: TenantId, _query: &str) -> Result<RawQueryRows, MCPError> {
            if tenant != self.tenant {
                return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
            }
            // `None` ⇒ fall through to the default trait impl so the
            // MethodNotFound default path stays test-covered.
            match &self.explain_rows {
                Some(rows) => Ok(RawQueryRows {
                    // Mirror the production #952 plan-row adapter column
                    // names (arcgraph_query::explain::PLAN_ROW_COLUMNS).
                    columns: Some(vec![
                        "operator".into(),
                        "details".into(),
                        "est_cost".into(),
                        "est_rows".into(),
                        "depth".into(),
                    ]),
                    rows: rows.clone(),
                    row_count: rows.len(),
                    truncated: false,
                    writes: WriteSummary::default(),
                }),
                None => Err(MCPError::MethodNotFound(
                    "graph.raw_query: explain mode not supported by this executor".into(),
                )),
            }
        }
    }

    fn fixture_rows() -> Vec<serde_json::Value> {
        vec![json!([1, "Alice"]), json!([2, "Bob"]), json!([3, "Carol"])]
    }

    fn fixture_req(tenant_id: u64, query: &str) -> RawQueryRequest {
        RawQueryRequest {
            tenant_id,
            query: query.into(),
            max_rows: None,
            format: Some(ResponseFormat::Json),
            explain: false,
        }
    }

    #[test]
    fn raw_query_tool_returns_rows_on_power_session() {
        // Canonical happy path: power-scope session, valid tenant,
        // simple query, executor returns 3 rows.
        let s = StubExecutor::new(TenantId::new(7))
            .with_rows(fixture_rows())
            .with_columns(vec!["id".into(), "name".into()]);
        let req = fixture_req(7, "MATCH (n:Person) RETURN n.id, n.name");
        let token = CancellationToken::new();
        let resp =
            raw_query_tool(&s, TenantId::new(7), SessionScope::Power, &token, req).expect("ok");
        assert_eq!(resp["format"], "json");
        let body = resp["body"].as_str().unwrap();
        assert!(body.contains("\"row_count\":3"), "row_count: body={body}");
        assert!(body.contains("Alice"), "first row visible");
        assert!(body.contains("Carol"), "last row visible");
        assert!(body.contains("\"truncated\":false"), "not truncated");
    }

    #[test]
    fn raw_query_tool_rejects_read_scope_with_forbidden() {
        // The W16ζ M5-11 spawn-prompt acceptance gate: a read-scope
        // session MUST reject graph.raw_query with -32008 BEFORE the
        // executor body runs. Per ADR-004 amendment-03 §D-1 +
        // design-v2 §9.5 JD stress test.
        let s = StubExecutor::new(TenantId::new(7)).with_rows(fixture_rows());
        let req = fixture_req(7, "MATCH (n) RETURN n");
        let token = CancellationToken::new();
        let err = raw_query_tool(&s, TenantId::new(7), SessionScope::Read, &token, req)
            .expect_err("read scope must reject");
        assert_eq!(err.code(), -32008);
        match err {
            MCPError::Forbidden { required_scope } => {
                assert_eq!(required_scope, "arcgraph.power");
            }
            other => panic!("expected Forbidden, got {other:?}"),
        }
    }

    #[test]
    fn raw_query_tool_rejects_cross_tenant_before_scope_check() {
        // A cross-tenant probe with insufficient scope MUST surface
        // Unauthorized (-32002), NOT Forbidden (-32008). The cross-
        // tenant guard runs FIRST to prevent leaking scope info to a
        // probe from a different tenant.
        let s = StubExecutor::new(TenantId::new(7)).with_rows(fixture_rows());
        let req = fixture_req(8, "MATCH (n) RETURN n");
        let token = CancellationToken::new();
        let err = raw_query_tool(&s, TenantId::new(7), SessionScope::Read, &token, req)
            .expect_err("cross-tenant rejects");
        assert_eq!(
            err.code(),
            -32002,
            "cross-tenant guard runs BEFORE scope check"
        );
        assert!(matches!(err, MCPError::Unauthorized));
    }

    #[test]
    fn raw_query_tool_rejects_empty_query() {
        let s = StubExecutor::new(TenantId::new(7));
        let req = fixture_req(7, "");
        let token = CancellationToken::new();
        let err = raw_query_tool(&s, TenantId::new(7), SessionScope::Power, &token, req)
            .expect_err("empty rejects");
        assert_eq!(err.code(), -32602);
    }

    #[test]
    fn raw_query_tool_rejects_oversized_query() {
        // Pin the MAX_RAW_QUERY_BYTES = 1 MiB cap. A query of exactly
        // MAX_RAW_QUERY_BYTES + 1 bytes MUST reject as -32602 BEFORE
        // the executor body runs.
        let s = StubExecutor::new(TenantId::new(7));
        let oversized: String = "x".repeat(MAX_RAW_QUERY_BYTES + 1);
        let req = RawQueryRequest {
            tenant_id: 7,
            query: oversized,
            max_rows: None,
            format: None,
            explain: false,
        };
        let token = CancellationToken::new();
        let err = raw_query_tool(&s, TenantId::new(7), SessionScope::Power, &token, req)
            .expect_err("oversized rejects");
        assert_eq!(err.code(), -32602);
        match err {
            MCPError::InvalidParams(msg) => {
                assert!(msg.contains("exceeds cap"), "msg={msg}");
            }
            other => panic!("expected InvalidParams, got {other:?}"),
        }
    }

    #[test]
    fn raw_query_tool_rejects_max_rows_zero() {
        let s = StubExecutor::new(TenantId::new(7));
        let req = RawQueryRequest {
            tenant_id: 7,
            query: "MATCH (n) RETURN n".into(),
            max_rows: Some(0),
            format: None,
            explain: false,
        };
        let token = CancellationToken::new();
        let err = raw_query_tool(&s, TenantId::new(7), SessionScope::Power, &token, req)
            .expect_err("max_rows=0 rejects");
        assert_eq!(err.code(), -32602);
    }

    #[test]
    fn raw_query_tool_rejects_max_rows_above_cap() {
        // Pin the MAX_RAW_QUERY_MAX_ROWS = 10_000 hard cap.
        let s = StubExecutor::new(TenantId::new(7));
        let req = RawQueryRequest {
            tenant_id: 7,
            query: "MATCH (n) RETURN n".into(),
            max_rows: Some(MAX_RAW_QUERY_MAX_ROWS + 1),
            format: None,
            explain: false,
        };
        let token = CancellationToken::new();
        let err = raw_query_tool(&s, TenantId::new(7), SessionScope::Power, &token, req)
            .expect_err("max_rows cap rejects");
        assert_eq!(err.code(), -32602);
    }

    #[test]
    fn raw_query_tool_respects_max_rows_with_truncation() {
        // Executor returns 3 rows; caller asks for max_rows=2. The MCP
        // boundary must truncate to 2 rows AND set truncated=true.
        let s = StubExecutor::new(TenantId::new(7)).with_rows(fixture_rows());
        let req = RawQueryRequest {
            tenant_id: 7,
            query: "MATCH (n) RETURN n".into(),
            max_rows: Some(2),
            format: Some(ResponseFormat::Json),
            explain: false,
        };
        let token = CancellationToken::new();
        let resp =
            raw_query_tool(&s, TenantId::new(7), SessionScope::Power, &token, req).expect("ok");
        let body = resp["body"].as_str().unwrap();
        assert!(body.contains("\"row_count\":2"), "row_count=2: body={body}");
        assert!(
            body.contains("\"truncated\":true"),
            "truncated: body={body}"
        );
        assert!(body.contains("Alice"));
        assert!(body.contains("Bob"));
        assert!(!body.contains("Carol"), "third row dropped");
    }

    #[test]
    fn raw_query_tool_surfaces_cancelled_when_token_tripped() {
        let s = StubExecutor::new(TenantId::new(7)).with_rows(fixture_rows());
        let req = fixture_req(7, "MATCH (n) RETURN n");
        let token = CancellationToken::new();
        token.cancel();
        let err = raw_query_tool(&s, TenantId::new(7), SessionScope::Power, &token, req)
            .expect_err("cancelled");
        assert_eq!(err.code(), -32001);
        assert!(matches!(err, MCPError::Cancelled));
    }

    #[test]
    fn raw_query_tool_surfaces_executor_query_error_as_minus_32005() {
        // Executor returns QueryError (the M5↔M4 mapping for ArcQL
        // parser / binder / type-checker faults). Surfaces as
        // -32005 with the message propagated.
        let s = StubExecutor::new(TenantId::new(7))
            .with_forced_error(MCPError::QueryError("unknown label X".into()));
        let req = fixture_req(7, "MATCH (n:X) RETURN n");
        let token = CancellationToken::new();
        let err = raw_query_tool(&s, TenantId::new(7), SessionScope::Power, &token, req)
            .expect_err("query error");
        assert_eq!(err.code(), -32005);
    }

    #[test]
    fn raw_query_tool_default_format_is_json() {
        // Per ADR-004 amendment-03 §D-1 point 4: raw_query defaults to
        // JSON (heterogeneous projections favor JSON over TOON).
        let s = StubExecutor::new(TenantId::new(7)).with_rows(fixture_rows());
        let req = RawQueryRequest {
            tenant_id: 7,
            query: "MATCH (n) RETURN n".into(),
            max_rows: None,
            format: None,
            explain: false,
        };
        let token = CancellationToken::new();
        let resp =
            raw_query_tool(&s, TenantId::new(7), SessionScope::Power, &token, req).expect("ok");
        assert_eq!(resp["format"], "json");
    }

    #[test]
    fn raw_query_request_rejects_unknown_field() {
        // strict public-contract policy #[serde(deny_unknown_fields)]: a typo'd
        // field (or pre-amendment `params` from a future client) must
        // reject at deserialize-time.
        let v = json!({
            "tenant_id": 7,
            "query": "MATCH (n) RETURN n",
            "params": {"x": 1}  // forward-deferred per ADR-004 amendment-03 §D-4
        });
        let res: Result<RawQueryRequest, _> = serde_json::from_value(v);
        assert!(res.is_err(), "params typo must reject");
    }

    #[test]
    fn raw_query_tool_default_max_rows_is_1000() {
        // Pin DEFAULT_RAW_QUERY_MAX_ROWS = 1000 — a request that omits
        // `max_rows` MUST use the default, NOT MAX_RAW_QUERY_MAX_ROWS.
        // We pin this by constructing a fixture with rows > 1000 and
        // asserting truncation kicks in at 1000.
        let many: Vec<serde_json::Value> = (0..1500)
            .map(|i| json!([i as u64, format!("row-{i}")]))
            .collect();
        let s = StubExecutor::new(TenantId::new(7)).with_rows(many);
        let req = RawQueryRequest {
            tenant_id: 7,
            query: "MATCH (n) RETURN n".into(),
            max_rows: None, // -> DEFAULT_RAW_QUERY_MAX_ROWS = 1000
            format: Some(ResponseFormat::Json),
            explain: false,
        };
        let token = CancellationToken::new();
        let resp =
            raw_query_tool(&s, TenantId::new(7), SessionScope::Power, &token, req).expect("ok");
        let body = resp["body"].as_str().unwrap();
        assert!(
            body.contains("\"row_count\":1000"),
            "default cap at 1000: body excerpt: {}",
            &body[..body.len().min(200)]
        );
        assert!(body.contains("\"truncated\":true"));
    }

    // ─────────────────────────────────────────────────────────────────
    // explain:true verb-consolidation mode (operator-ruled — stays at
    // the ADR-004 10-tool cap; NO separate graph.explain tool)
    // ─────────────────────────────────────────────────────────────────

    /// One baked plan row, shaped like the #952 plan-row adapter output
    /// (`[op, details, estimated_cost, estimated_card, depth]`).
    fn fixture_plan_rows() -> Vec<serde_json::Value> {
        vec![
            json!(["Project", "b0", 12.0, 3.0, 0]),
            json!(["Scan", ":Person", 4.0, 3.0, 1]),
        ]
    }

    #[test]
    fn raw_query_tool_explain_true_returns_plan_rows_not_executed_rows() {
        // explain:true MUST return the PLAN rows (via executor.explain),
        // NOT the executed-query rows (via executor.execute). The stub's
        // execute would return "Alice/Bob/Carol"; the explain override
        // returns the plan tree.
        let s = StubExecutor::new(TenantId::new(7))
            .with_rows(fixture_rows())
            .with_explain_rows(fixture_plan_rows());
        let req = RawQueryRequest {
            tenant_id: 7,
            query: "MATCH (n:Person) RETURN n".into(),
            max_rows: None,
            format: Some(ResponseFormat::Json),
            explain: true,
        };
        let token = CancellationToken::new();
        let resp =
            raw_query_tool(&s, TenantId::new(7), SessionScope::Power, &token, req).expect("ok");
        let body = resp["body"].as_str().unwrap();
        // Plan operators present (NOT executed-query data rows).
        assert!(body.contains("Project"), "plan op visible: {body}");
        assert!(body.contains("Scan"), "plan op visible: {body}");
        // Plan-row columns present.
        assert!(body.contains("est_cost"), "plan columns: {body}");
        assert!(body.contains("est_rows"), "plan columns: {body}");
        assert!(body.contains("\"row_count\":2"), "2 plan rows: {body}");
        // NOT the executed-query rows.
        assert!(!body.contains("Alice"), "must NOT execute: {body}");
        assert!(!body.contains("Carol"), "must NOT execute: {body}");
    }

    #[test]
    fn raw_query_tool_explain_false_executes_unchanged() {
        // explain:false (and absent) MUST execute exactly as before —
        // returning the executed-query data rows, not a plan.
        let s = StubExecutor::new(TenantId::new(7))
            .with_rows(fixture_rows())
            .with_explain_rows(fixture_plan_rows());
        let req = RawQueryRequest {
            tenant_id: 7,
            query: "MATCH (n) RETURN n".into(),
            max_rows: None,
            format: Some(ResponseFormat::Json),
            explain: false,
        };
        let token = CancellationToken::new();
        let resp =
            raw_query_tool(&s, TenantId::new(7), SessionScope::Power, &token, req).expect("ok");
        let body = resp["body"].as_str().unwrap();
        // Executed-query data rows present.
        assert!(body.contains("Alice"), "executes: {body}");
        assert!(body.contains("Carol"), "executes: {body}");
        // NOT a plan.
        assert!(!body.contains("Project"), "must not be a plan: {body}");
        assert!(!body.contains("est_cost"), "must not be a plan: {body}");
    }

    #[test]
    fn raw_query_tool_explain_default_when_field_absent_executes() {
        // A request envelope that OMITS `explain` deserializes to
        // explain=false (#[serde(default)]) and executes. Pins the
        // deny_unknown_fields-safe forward-compat shape.
        let v = json!({
            "tenant_id": 7,
            "query": "MATCH (n) RETURN n",
            "format": "json"
            // note: no `explain` key
        });
        let req: RawQueryRequest = serde_json::from_value(v).expect("absent explain parses");
        assert!(!req.explain, "absent explain defaults to false");

        let s = StubExecutor::new(TenantId::new(7))
            .with_rows(fixture_rows())
            .with_explain_rows(fixture_plan_rows());
        let token = CancellationToken::new();
        let resp =
            raw_query_tool(&s, TenantId::new(7), SessionScope::Power, &token, req).expect("ok");
        let body = resp["body"].as_str().unwrap();
        assert!(
            body.contains("Alice"),
            "executes when explain absent: {body}"
        );
        assert!(!body.contains("Project"), "not a plan: {body}");
    }

    #[test]
    fn raw_query_tool_explain_true_rejects_read_scope_with_forbidden() {
        // explain:true inherits the Power-scope gate (a plan can leak
        // schema / cardinality). A read-scope session MUST reject with
        // -32008 BEFORE the explain branch runs — the gate holds.
        let s = StubExecutor::new(TenantId::new(7)).with_explain_rows(fixture_plan_rows());
        let req = RawQueryRequest {
            tenant_id: 7,
            query: "MATCH (n) RETURN n".into(),
            max_rows: None,
            format: None,
            explain: true,
        };
        let token = CancellationToken::new();
        let err = raw_query_tool(&s, TenantId::new(7), SessionScope::Read, &token, req)
            .expect_err("explain:true read scope must reject");
        assert_eq!(err.code(), -32008);
        match err {
            MCPError::Forbidden { required_scope } => {
                assert_eq!(required_scope, "arcgraph.power");
            }
            other => panic!("expected Forbidden, got {other:?}"),
        }
    }

    #[test]
    fn raw_query_tool_explain_true_rejects_cross_tenant_before_scope() {
        // The cross-tenant guard runs BEFORE the scope check AND before
        // the explain branch, so a cross-tenant explain:true probe with
        // insufficient scope surfaces Unauthorized (-32002), not
        // Forbidden and not a plan.
        let s = StubExecutor::new(TenantId::new(7)).with_explain_rows(fixture_plan_rows());
        let req = RawQueryRequest {
            tenant_id: 8,
            query: "MATCH (n) RETURN n".into(),
            max_rows: None,
            format: None,
            explain: true,
        };
        let token = CancellationToken::new();
        let err = raw_query_tool(&s, TenantId::new(7), SessionScope::Read, &token, req)
            .expect_err("cross-tenant rejects");
        assert_eq!(err.code(), -32002);
        assert!(matches!(err, MCPError::Unauthorized));
    }

    #[test]
    fn raw_query_tool_explain_true_default_impl_is_method_not_found() {
        // When an executor does NOT override `explain` (here: stub
        // without `with_explain_rows`), explain:true surfaces the
        // default trait impl's MethodNotFound (-32601) — NOT a silent
        // execute, NOT a panic.
        let s = StubExecutor::new(TenantId::new(7)).with_rows(fixture_rows());
        let req = RawQueryRequest {
            tenant_id: 7,
            query: "MATCH (n) RETURN n".into(),
            max_rows: None,
            format: None,
            explain: true,
        };
        let token = CancellationToken::new();
        let err = raw_query_tool(&s, TenantId::new(7), SessionScope::Power, &token, req)
            .expect_err("default explain impl rejects");
        assert_eq!(err.code(), -32601);
        assert!(matches!(err, MCPError::MethodNotFound(_)));
    }

    // ─────────────────────────────────────────────────────────────────
    // ADR-153 W27-β — WriteSummary wire-shape tests
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn write_summary_default_is_all_zero_and_is_empty() {
        // Per ADR-153 §D-2: pure-read query returns a zero WriteSummary
        // whose `is_empty()` predicate evaluates true (renderers
        // suppress the `writes:{...}` block in that case).
        let ws = WriteSummary::default();
        assert_eq!(ws.nodes_created, 0);
        assert_eq!(ws.nodes_deleted, 0);
        assert_eq!(ws.rels_created, 0);
        assert_eq!(ws.rels_deleted, 0);
        assert_eq!(ws.properties_set, 0);
        assert_eq!(ws.properties_removed, 0);
        assert_eq!(ws.labels_added, 0);
        assert_eq!(ws.labels_removed, 0);
        assert!(ws.is_empty());
        assert_eq!(ws.total(), 0);
    }

    #[test]
    fn write_summary_total_sums_all_counters() {
        let ws = WriteSummary {
            nodes_created: 1,
            nodes_deleted: 2,
            rels_created: 3,
            rels_deleted: 4,
            properties_set: 5,
            properties_removed: 6,
            labels_added: 7,
            labels_removed: 8,
        };
        assert!(!ws.is_empty());
        assert_eq!(ws.total(), 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8);
    }

    #[test]
    fn raw_query_rows_serializes_writes_when_non_empty() {
        // ADR-153 §D-2 wire shape: a write-touched envelope surfaces
        // `writes:{...}` with non-zero counters on the JSON wire.
        let rows = RawQueryRows {
            columns: None,
            rows: vec![json!([1u64])],
            row_count: 1,
            truncated: false,
            writes: WriteSummary {
                nodes_created: 1,
                ..Default::default()
            },
        };
        let v = serde_json::to_value(&rows).expect("serialize");
        assert_eq!(v["writes"]["nodes_created"], 1);
        assert_eq!(v["writes"]["nodes_deleted"], 0);
    }

    #[test]
    fn raw_query_rows_deserializes_pre_w27_beta_envelope_without_writes() {
        // Forward-compat per ADR-153 §"Wire-shape evolution": a pre-
        // W27-β envelope that omits the `writes` field deserializes
        // cleanly to a default (all-zero) WriteSummary, matching the
        // `#[serde(default)]` on the RawQueryRows.writes field.
        let pre_w27 = json!({
            "columns": ["id"],
            "rows": [[1]],
            "row_count": 1,
            "truncated": false
            // note: no `writes` key
        });
        let parsed: RawQueryRows = serde_json::from_value(pre_w27).expect("backward-compat parse");
        assert_eq!(parsed.row_count, 1);
        assert!(
            parsed.writes.is_empty(),
            "missing writes deserializes to zero summary; got {:?}",
            parsed.writes
        );
    }

    #[test]
    fn raw_query_tool_propagates_writes_to_wire_body() {
        // End-to-end pin: an executor that populates `writes` must
        // surface those counters through `raw_query_tool` → wire body.
        // Used by the W27-β integration tests as an in-process smoke
        // before exercising the production substrate path.
        let s = StubExecutor::new(TenantId::new(7))
            .with_rows(vec![json!([42u64])])
            .with_writes(WriteSummary {
                nodes_created: 1,
                ..Default::default()
            });
        let req = fixture_req(7, "CREATE (n:User) RETURN n");
        let token = CancellationToken::new();
        let resp =
            raw_query_tool(&s, TenantId::new(7), SessionScope::Power, &token, req).expect("ok");
        let body = resp["body"].as_str().unwrap();
        assert!(
            body.contains("\"nodes_created\":1"),
            "writes propagate to body: {body}"
        );
        assert!(body.contains("\"row_count\":1"));
    }
}
