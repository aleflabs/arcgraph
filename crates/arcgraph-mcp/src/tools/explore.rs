//! W14β M5-06 — `graph.explore` Tier-1 MCP tool.
//!
//! Returns an N-hop neighborhood graph (nodes + edges) rooted at a
//! seed node id, with depth bounded by a per-request `max_depth` (and
//! a hard cap enforced server-side) AND output bounded by a per-request
//! `max_results` combined node+edge cap (W30 #900).
//!
//! # Surface seam — `NeighborhoodExplorer`
//!
//! Following the W13δ M5-04 / M5-05 pattern: the MCP layer defines a
//! local adapter trait ([`NeighborhoodExplorer`]) rather than reaching
//! across the arcgraph-query bounded-context to bind directly to
//! [`arcgraph_query::executor::ExecutorSubstrate`] or
//! [`arcgraph_query::executor::ops::expand::ExpandOp`]. Production
//! wiring at M4-08+ implements this trait on the storage tenant
//! handle (which already carries the per-tenant adjacency surface that
//! [`ExpandOp`](arcgraph_query::executor::ops::expand) reads from).
//!
//! `feedback_avoid_speculative_scaffolding.md` applies: define the
//! trait at first consumer (this tool) instead of speculatively
//! extending arcgraph-query's `ExecutorSubstrate` with a "give me a
//! neighborhood graph" convenience method that has no other consumer.
//!
//! # Depth-cap discipline (W14β spawn-prompt acceptance)
//!
//! The spawn prompt pins a default `max_depth = 2` with a hard cap of
//! 5. The request envelope ([`ExploreRequest`]) defaults the field;
//! [`explore_tool`] surfaces [`MCPError::InvalidParams`] when the
//! caller asks for more than [`MAX_EXPLORE_DEPTH`] hops. This pre-
//! validation runs BEFORE the explorer is touched — a hostile request
//! that asks for `max_depth = 1_000_000` cannot tip a stub fixture
//! into combinatorial enumeration.
//!
//! # Output-cap discipline (W30 #900 — token-blowup defense)
//!
//! Depth-capping bounds how FAR the walk reaches but NOT how WIDE: a
//! single high-fanout node (issue #900 repro: a hub with 300 leaves →
//! 301 nodes + 300 edges at depth 1) produced an unbounded response
//! with no `truncated` signal, blowing the agent's token budget. The
//! [`ExploreRequest::max_results`] field caps the COMBINED node+edge
//! count of the response, defaulting to [`DEFAULT_EXPLORE_LIMIT`] and
//! rejecting values above [`MAX_EXPLORE_LIMIT`] with
//! [`MCPError::InvalidParams`]. When the cap clips anything,
//! [`Neighborhood::truncated`] is set `true` so the agent knows the
//! view is partial. This MIRRORS the `graph.raw_query` row-cap
//! convention exactly ([`crate::tools::raw_query`]: a single
//! `max_rows` cap over a flat row stream + a `truncated: bool` on the
//! result envelope) per ADR-004 amendment-03 §D-1 — it does NOT invent
//! a new pagination/cursor shape (that is forward-pinned to v1.1+).
//!
//! # Snapshot-LSN discipline
//!
//! Per ADR-038 amendment-03 §TIER-1 GAP E rule 1, the explorer's
//! storage-side reads MUST acquire a snapshot LSN before the first
//! batch pull (matching the executor's single-statement materialize
//! tail — rule 1 is "Snapshot LSN acquired at execute-time, before
//! the first operator pulls a batch"; rule 2 is the multi-statement
//! LSN-sharing rule). The trait does not expose the LSN to MCP
//! callers — it's an implementation detail of the
//! [`NeighborhoodExplorer::explore`] body. v1.0-alpha stub impls have
//! no MVCC layer; production wiring at M4-08+ acquires per
//! amendment-03 rule 1 (same shape as
//! [`crate::tools::inspect::NodeInspector`]).
//!
//! # Access-control boundary
//!
//! The tool checks `request.tenant_id == session_tenant` BEFORE any
//! explorer call. Cross-tenant requests reject as
//! [`MCPError::Unauthorized`] (-32002). Principal-scoped requests then
//! resolve one effective-permission snapshot and authorize the seed and
//! EVERY returned node. Denied nodes, their incident edges, and paths
//! that exist only through them are omitted; a denied seed is identical
//! to a missing seed. A principal-less request is admitted only for an
//! explicit [`SessionScope::Power`] session; all other sessions fail
//! closed with [`MCPError::Forbidden`] (-32008).
//!
//! # ADR provenance
//! - **ADR-004 §"Tier 1 (agent-facing, default)"** — `graph.explore()`
//!   is the third Tier-1 tool in the 10-tool catalog.
//! - **ADR-004 amendment-03 §D-1** — the row/output-cap convention
//!   (`DEFAULT_*` + `MAX_*` consts + `truncated: bool`) `graph.explore`
//!   mirrors from `graph.raw_query` for the #900 output cap.
//! - **ADR-036 §"hybrid retrieval architecture"** — `k_hop_local` /
//!   `bidirectional_shortest` surface that production wiring composes
//!   against at M4-08+.
//! - **ADR-038 amendment-03 §TIER-1 GAP E rule 1** — snapshot-LSN
//!   acquired at execute-time before first batch pull (the binding
//!   for read-only tools).
//! - **ADR-037 D-1** — per-tenant routing; the cross-tenant guard
//!   inherits this posture.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::Arc;

use arcgraph_core::{NodeId, TenantId};
use arcgraph_query::CancellationToken;
use arcgraph_storage::permissions::PermissionIndex;
use serde::{Deserialize, Serialize};

use crate::error::MCPError;
use crate::read_acl::{ReadAccess, authorize_read};
use crate::scope::SessionScope;
use crate::tools::ResponseFormat;
use crate::tools::inspect::NeighborDirection;

/// Default `max_depth` when an [`ExploreRequest`] omits the field.
///
/// Pinned at 2 per the W14β spawn-prompt acceptance ("default depth
/// 2"). A future ADR may revisit this; the constant is exported so
/// downstream MCP clients can default-fill identically.
pub const DEFAULT_EXPLORE_DEPTH: u32 = 2;

/// Hard cap on `max_depth` accepted by [`explore_tool`].
///
/// Pinned at 5 per the W14β spawn-prompt acceptance ("config-cap at
/// 5"). v1.0-alpha treats this as a compile-time constant; a future
/// M5-12 rate-limit slice may move it to a per-tenant config value,
/// in which case [`explore_tool`] gains a runtime override
/// parameter. Today the cap exists to short-circuit a hostile or
/// runaway request before the explorer body runs.
pub const MAX_EXPLORE_DEPTH: u32 = 5;

/// Default cap on the COMBINED node+edge count returned by
/// [`explore_tool`] when [`ExploreRequest::max_results`] is omitted.
///
/// Mirrors [`crate::tools::raw_query::DEFAULT_RAW_QUERY_MAX_ROWS`] =
/// 1000 — the same v1.0-alpha token-budget default the
/// `graph.raw_query` Tier-2 tool pins for a flat row stream. Per
/// ADR-004 amendment-03 §D-1 the explore output cap reuses the
/// raw_query cap convention rather than inventing a new pagination
/// shape. A high-fanout hub (issue #900 repro: a 300-leaf hub yields
/// 301 nodes and 300 edges) is bounded to this default unless the
/// caller raises `max_results` up to [`MAX_EXPLORE_LIMIT`].
pub const DEFAULT_EXPLORE_LIMIT: u32 = 1000;

/// Hard cap on [`ExploreRequest::max_results`].
///
/// Mirrors [`crate::tools::raw_query::MAX_RAW_QUERY_MAX_ROWS`] =
/// 10_000 — the v1.0-alpha memory/token-budget floor. A caller asking
/// for more than this is rejected with [`MCPError::InvalidParams`]
/// (-32602) and must narrow the `rel_types` filter, lower `max_depth`,
/// or wait for the v1.1+ streaming/cursor surface. Per ADR-004
/// amendment-03 §D-1.
pub const MAX_EXPLORE_LIMIT: u32 = 10_000;

/// Traversal direction for [`NeighborhoodExplorer::explore`] (ADR-217).
///
/// The v1.0-alpha explorer walked **outbound only** (`scan_out`). For
/// "sink" topologies — where the interesting neighbors all point *into* a
/// hub (for example, an `Account` reached by inbound relationship
/// types that an outbound walk never follows) — an
/// outbound-only neighborhood is empty in the rich direction. This enum
/// lets a caller opt into inbound ([`crud::scan_in`](arcgraph_storage::crud),
/// ADR-131 reverse adjacency) or both.
///
/// # Backward compatibility (ADR-217)
///
/// [`Default`] is [`ExploreDirection::Out`], and [`ExploreRequest::direction`]
/// is `#[serde(default)]`, so a request that omits the field behaves
/// exactly as before (outbound BFS). `#[non_exhaustive]` ensures a future
/// variant is not a breaking change.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum ExploreDirection {
    /// Outbound only — the seed is each edge's `from` (the v1.0-alpha
    /// default; `scan_out`). Unchanged behavior.
    #[default]
    Out,
    /// Inbound only — the seed is each edge's `to` (`scan_in`, ADR-131
    /// reverse adjacency). Edges tag [`NeighborDirection::In`].
    In,
    /// Both directions, de-duplicated by `RelId`. Outbound edges tag
    /// [`NeighborDirection::Out`], inbound edges [`NeighborDirection::In`].
    Both,
}

/// Adapter trait read by the [`explore_tool`] entry point.
///
/// Implementations live OUTSIDE this crate: tests stub it in-line;
/// production wiring at M4-08+ implements it on the storage tenant
/// handle (which already carries the per-tenant adjacency surface
/// that [`arcgraph_query::executor::ExecutorSubstrate::expand`]
/// reads from).
///
/// # Per-tenant scoping
///
/// `tenant: TenantId` parameter matches the
/// [`crate::tools::schema::SchemaProvider`] /
/// [`crate::tools::inspect::NodeInspector`] pattern — a single
/// `NeighborhoodExplorer` instance can serve multiple tenants under a
/// shared MCP router (forward-method per M5-12).
///
/// # `Send + Sync`
///
/// MCP transport runs on a tokio runtime; the explorer must be
/// shareable across awaits.
///
/// # Output cap is NOT the explorer's concern
///
/// The `max_results` output cap (W30 #900) is applied by
/// [`explore_capped`] at the tool boundary, NOT by the explorer.
/// Impls enumerate the full `max_depth`-bounded neighborhood and
/// return [`Neighborhood::truncated`] = `false`; the tool boundary
/// owns the cap and re-sets `truncated`. (Pushing the cap down into
/// the explorer for a true memory bound on pathological fanout is a
/// v1.1+ efficiency follow-up — at v1.0-α the wire/token blowup is the
/// actual defect, and a single neighborhood is memory-bounded by node
/// degree.)
///
/// # Cancellation contract — IMPLEMENTOR HARD REQUIREMENT
///
/// The [`CancellationToken`] argument is the same surface read by
/// arcgraph-query's executor operators (per
/// [`arcgraph_query::executor::context::CancellationToken`]).
/// Production impls MUST check `token.is_cancelled()` at hop /
/// batch boundaries and short-circuit with
/// [`MCPError::Cancelled`]; the stub impls in this module's tests
/// model that contract for the
/// `explore_tool_surfaces_cancelled_when_token_tripped` unit test.
///
/// # Snapshot-LSN contract — IMPLEMENTOR HARD REQUIREMENT
///
/// Per ADR-038 amendment-03 §TIER-1 GAP E rule 1 ("Snapshot LSN
/// acquired at execute-time, before the first operator pulls a
/// batch"), an `explore()` call MUST acquire a snapshot LSN before
/// the first hop and hold it for the life of the call. The trait
/// shape DELIBERATELY DOES NOT carry the LSN as a parameter —
/// v1.0-alpha stubs have no MVCC layer, and the production storage
/// handle is the natural source of LSN acquisition. The contract is
/// enforced by convention and the M4-08+ wiring slice's end-to-end
/// tests, not by the type signature.
pub trait NeighborhoodExplorer: Send + Sync {
    /// Explore the N-hop neighborhood of `seed` in `tenant`.
    ///
    /// `max_depth` is the hop-cap; impls MUST NOT pull edges past
    /// this depth. `rel_filter`, when `Some`, restricts the
    /// enumerated relationships to the given rel-type names (impls
    /// translate to internal `TypeId`s at the catalog binding).
    ///
    /// `direction` (ADR-217) selects outbound (`scan_out`, the v1.0-alpha
    /// default), inbound (`scan_in`, ADR-131 reverse adjacency), or both.
    /// Impls that do not support inbound (no reverse index) MAY treat
    /// [`ExploreDirection::In`] / [`ExploreDirection::Both`] as outbound,
    /// but the storage impl honors it via `crud::scan_in`.
    ///
    /// `cancel` is the cancellation token the caller plumbs in;
    /// production impls check it at hop boundaries.
    ///
    /// Errors as [`MCPError::TenantUnknown`] for an unbound tenant,
    /// [`MCPError::QueryError`] for a missing seed node id (rendered
    /// as "seed not found" inside the query-error bucket — distinct
    /// from "tenant unknown"), or [`MCPError::ExecutionEval`] /
    /// [`MCPError::IndexUnavailable`] for substrate-level faults.
    ///
    /// Implementors MUST honor the snapshot-LSN + cancellation
    /// contracts on the trait doc comment above, and SHOULD leave
    /// [`Neighborhood::truncated`] = `false` — the output cap is
    /// applied at the tool boundary by [`explore_capped`].
    // ADR-217 added `direction`, taking the method to 7 params; the
    // alternative (a request struct) would churn every impl + call site
    // for one enum. The trait is the published explorer contract.
    #[allow(clippy::too_many_arguments)]
    fn explore(
        &self,
        tenant: TenantId,
        seed: u64,
        max_depth: u32,
        rel_filter: Option<&[String]>,
        direction: ExploreDirection,
        cancel: &CancellationToken,
    ) -> Result<Neighborhood, MCPError>;

    /// ADR-212 / ADR-218 — the per-tenant permission index used to
    /// authorize every node returned by `graph.explore`.
    ///
    /// The default is deliberately unavailable. A principal-scoped
    /// request against an explorer that does not override this seam is
    /// rejected with [`MCPError::IndexUnavailable`], never served
    /// unfiltered. The principal-less SYSTEM-TRUSTED path does not call
    /// this method and is admitted only for [`SessionScope::Power`].
    fn permission_index(
        &self,
        tenant: TenantId,
        cancel: &CancellationToken,
    ) -> Result<Option<Arc<PermissionIndex>>, MCPError> {
        let _ = (tenant, cancel);
        Ok(None)
    }
}

/// Per-seed neighborhood result returned by
/// [`NeighborhoodExplorer::explore`].
///
/// Serializes as a JSON / YAML / TOON tree per M5-06. The shape is
/// `{ seed, max_depth, nodes: [...], edges: [...], truncated }` —
/// nodes are uniform-shape records (TOON tabular friendly), edges
/// likewise.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Neighborhood {
    /// The seed node id (echoed for client-side disambiguation).
    pub seed: u64,
    /// The maximum hop count that was honored on this call. The
    /// v1.0-alpha contract is identity — impls MUST echo the requested
    /// `max_depth` here even when the underlying enumeration
    /// short-circuited (e.g. all neighbors enumerated before the cap).
    /// A future ADR may relax this so an impl can signal a lower
    /// achieved-depth when short-circuiting; the wire shape is
    /// forward-compatible with that change since downstream clients
    /// already treat this as `≤ requested`.
    pub max_depth: u32,
    /// All nodes visited during the walk, including the seed.
    pub nodes: Vec<NeighborhoodNode>,
    /// All edges traversed during the walk. May reference the same
    /// node id more than once when an undirected adjacency is
    /// observed from both sides; the explorer impl is responsible for
    /// the de-dup discipline (production impls de-dup on `RelId`).
    pub edges: Vec<NeighborhoodEdge>,
    /// `true` when the [`explore_tool`] output cap (`max_results`)
    /// clipped nodes and/or edges from this neighborhood (W30 #900).
    ///
    /// Mirrors [`crate::tools::raw_query::RawQueryRows::truncated`]:
    /// clients seeing `truncated: true` SHOULD re-run with a larger
    /// `max_results`, a narrower `rel_types` filter, or a smaller
    /// `max_depth`. Owned by [`explore_capped`] — the
    /// [`NeighborhoodExplorer`] enumerates the full neighborhood and
    /// leaves this `false`; the tool boundary sets it.
    ///
    /// `#[serde(default)]` so a pre-#900 envelope omitting the field
    /// deserializes cleanly (mirrors the
    /// [`crate::tools::raw_query::RawQueryRows::writes`] forward-compat
    /// convention under the code-quality policy cross-version wire contract).
    #[serde(default)]
    pub truncated: bool,
}

/// One node entry in a [`Neighborhood`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NeighborhoodNode {
    /// The node id.
    pub id: u64,
    /// Optional label. Single-label per ADR-038 §2 D-1 v1.0 grammar.
    pub label: Option<String>,
    /// Hop distance from the seed (0 for the seed itself, 1 for
    /// direct neighbors, etc.).
    pub depth: u32,
    /// Property bag — same shape as
    /// [`crate::tools::inspect::NodeInspection::properties`] for
    /// client-side reuse.
    pub properties: BTreeMap<String, serde_json::Value>,
}

/// One edge entry in a [`Neighborhood`].
///
/// Edges are topology-only at v1.0-α: there is deliberately NO
/// `properties` field. Node-property hydration landed with #894, but
/// edge/relationship-property hydration is a scoped-out follow-up
/// (tracked separately) — `graph.explore` returns relationship topology
/// (`from` / `to` / `rel_type` / `direction`) but not relationship
/// property bags. Adding a `properties` field here is the forward path
/// when that follow-up lands.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NeighborhoodEdge {
    /// The edge's `from` node id.
    pub from: u64,
    /// The edge's `to` node id.
    pub to: u64,
    /// Optional rel-type name.
    pub rel_type: Option<String>,
    /// Direction tag — `"out"` / `"in"` / `"undirected"`. Reuses the
    /// [`NeighborDirection`] enum so the wire shape is consistent
    /// across the `graph.inspect` and `graph.explore` tools.
    pub direction: NeighborDirection,
}

// ─────────────────────────────────────────────────────────────────────
// Request envelope
// ─────────────────────────────────────────────────────────────────────

/// Request params for the `graph.explore` tool.
///
/// `#[serde(deny_unknown_fields)]` under the code-quality policy config-strict-mode
/// convention.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ExploreRequest {
    /// The tenant to explore within.
    pub tenant_id: u64,
    /// The seed node id.
    pub seed: u64,
    /// Maximum hop count. Defaults to [`DEFAULT_EXPLORE_DEPTH`] when
    /// omitted; values above [`MAX_EXPLORE_DEPTH`] reject as
    /// [`MCPError::InvalidParams`].
    #[serde(default)]
    pub max_depth: Option<u32>,
    /// Optional cap on the COMBINED node+edge count in the response
    /// (W30 #900). Defaults to [`DEFAULT_EXPLORE_LIMIT`] when omitted;
    /// values above [`MAX_EXPLORE_LIMIT`] reject as
    /// [`MCPError::InvalidParams`]. Mirrors
    /// [`crate::tools::raw_query::RawQueryRequest::max_rows`].
    #[serde(default)]
    pub max_results: Option<u32>,
    /// Optional rel-type allowlist. Empty Vec = "no filter"; impls
    /// only restrict when this is `Some(non_empty)`.
    #[serde(default)]
    pub rel_types: Option<Vec<String>>,
    /// Optional traversal direction (ADR-217). Defaults to
    /// [`ExploreDirection::Out`] when omitted — so a pre-ADR-217 request
    /// walks outbound-only exactly as before.
    /// `in` / `both` opt into inbound ([`crud::scan_in`](arcgraph_storage::crud),
    /// ADR-131) so a "sink" hub's inbound neighbors become reachable.
    #[serde(default)]
    pub direction: Option<ExploreDirection>,
    /// Optional render-format hint. Defaults to TOON — explore
    /// results are uniform-shape rows (the design-v2 §9.3 token-
    /// savings path; the M5-09 TOON encoder gets its motivating
    /// shape here).
    #[serde(default)]
    pub format: Option<ResponseFormat>,
    /// ADR-212 — end-user principal on whose behalf this traversal is
    /// issued. When present, the seed and every returned node are
    /// filtered through the tenant's effective permission set. When
    /// absent, only an explicit [`SessionScope::Power`] session may use
    /// the unfiltered SYSTEM-TRUSTED path (#1488 / #1293 convention).
    #[serde(default)]
    pub principal: Option<String>,
}

// ─────────────────────────────────────────────────────────────────────
// Output cap
// ─────────────────────────────────────────────────────────────────────

/// Apply the `max_results` output cap to a freshly-enumerated
/// [`Neighborhood`], in place. Returns `true` iff the cap clipped at
/// least one node or edge.
///
/// # Truncation policy (deterministic + documented)
///
/// The cap bounds the **combined** node+edge count (mirroring
/// `graph.raw_query`'s single `max_rows` cap over a flat row stream).
/// When `nodes.len() + edges.len() > cap` we:
///
/// 1. **Nodes first, closest-to-seed kept.** Stable-sort nodes by
///    `depth` (seed = depth 0 sorts first), then keep at most `cap`
///    nodes. This guarantees the seed and its nearest neighbors — the
///    most relevant subset for an agent summarizing a local
///    neighborhood — survive the cut, regardless of the order the
///    explorer emitted them in. `sort_by_key` is stable, so within a
///    depth the explorer's original order is preserved (determinism).
/// 2. **Coherent edges only.** Drop any edge whose endpoint was cut in
///    step 1 (a dangling edge referencing an absent node is noise for
///    the agent), then keep at most `cap - nodes.len()` of the
///    surviving edges.
///
/// Rationale for nodes-first (vs proportional): nodes carry the
/// entities (label + properties) an agent reasons over; edges are
/// recoverable by re-querying with a higher `max_results` or a
/// narrower filter. Capping the *combined* count (not nodes and edges
/// independently) is what actually bounds the token/byte budget the
/// issue #900 repro blew through.
fn apply_output_cap(neighborhood: &mut Neighborhood, max_results: u32) -> bool {
    let cap = max_results as usize;
    if neighborhood.nodes.len() + neighborhood.edges.len() <= cap {
        return false;
    }
    // 1. Nodes-first, closest-to-seed: stable sort by depth then clip.
    neighborhood.nodes.sort_by_key(|n| n.depth);
    neighborhood.nodes.truncate(cap);
    // 2. Coherent edges: drop edges referencing a cut node, then clip
    //    to the remaining budget (`cap - kept_nodes`).
    let kept: std::collections::HashSet<u64> = neighborhood.nodes.iter().map(|n| n.id).collect();
    neighborhood
        .edges
        .retain(|e| kept.contains(&e.from) && kept.contains(&e.to));
    let edge_budget = cap.saturating_sub(neighborhood.nodes.len());
    neighborhood.edges.truncate(edge_budget);
    true
}

/// Remove every denied node and every edge incident to one, then retain
/// only the authorized component reachable from the seed under the
/// request's traversal direction.
///
/// Recomputing depths is security-relevant: a visible node may have both
/// a permitted long path and a shorter path through a denied node. Keeping
/// the explorer's original depth would disclose the denied shortcut even
/// after its node and edges were removed.
fn retain_visible_reachable(
    neighborhood: &mut Neighborhood,
    access: &ReadAccess,
    traversal_direction: ExploreDirection,
) {
    let visible: HashSet<u64> = neighborhood
        .nodes
        .iter()
        .filter(|node| access.allows(NodeId::new(node.id)))
        .map(|node| node.id)
        .collect();

    neighborhood
        .edges
        .retain(|edge| visible.contains(&edge.from) && visible.contains(&edge.to));

    let mut adjacency: HashMap<u64, Vec<u64>> = HashMap::new();
    for edge in &neighborhood.edges {
        match traversal_direction {
            ExploreDirection::Out => adjacency.entry(edge.from).or_default().push(edge.to),
            ExploreDirection::In => adjacency.entry(edge.to).or_default().push(edge.from),
            ExploreDirection::Both => {
                adjacency.entry(edge.from).or_default().push(edge.to);
                adjacency.entry(edge.to).or_default().push(edge.from);
            }
        }
    }

    let mut depth_by_node: HashMap<u64, u32> = HashMap::new();
    let mut frontier: VecDeque<u64> = VecDeque::new();
    if visible.contains(&neighborhood.seed) {
        depth_by_node.insert(neighborhood.seed, 0);
        frontier.push_back(neighborhood.seed);
    }
    while let Some(current) = frontier.pop_front() {
        let next_depth = depth_by_node[&current].saturating_add(1);
        if let Some(neighbors) = adjacency.get(&current) {
            for &neighbor in neighbors {
                if let std::collections::hash_map::Entry::Vacant(slot) =
                    depth_by_node.entry(neighbor)
                {
                    slot.insert(next_depth);
                    frontier.push_back(neighbor);
                }
            }
        }
    }

    let max_depth = neighborhood.max_depth;
    neighborhood.nodes.retain_mut(|node| {
        let Some(depth) = depth_by_node.get(&node.id) else {
            return false;
        };
        if *depth > max_depth {
            return false;
        }
        node.depth = *depth;
        true
    });
    let reachable: HashSet<u64> = neighborhood.nodes.iter().map(|node| node.id).collect();
    neighborhood
        .edges
        .retain(|edge| reachable.contains(&edge.from) && reachable.contains(&edge.to));
}

// ─────────────────────────────────────────────────────────────────────
// Tool entry point
// ─────────────────────────────────────────────────────────────────────

/// Core of [`explore_tool`]: run the cross-tenant + depth-cap +
/// output-cap validation, invoke the explorer, and apply the
/// `max_results` output cap (setting [`Neighborhood::truncated`]).
/// Returns the capped [`Neighborhood`] WITHOUT rendering it.
///
/// Shared by [`explore_tool`] and callers that need the structured
/// [`Neighborhood`] before rendering.
///
/// # Validation order
///
/// 1. **Cross-tenant guard** — `tenant_id != session_tenant` rejects
///    as [`MCPError::Unauthorized`] (-32002) before the explorer call.
/// 2. **Principal-less scope gate** (#1488, mirroring #1293) — an
///    absent principal on a non-power session rejects as
///    [`MCPError::Forbidden`] (-32008), before validation or storage.
/// 3. **Depth-cap guard** — `max_depth > MAX_EXPLORE_DEPTH` rejects as
///    [`MCPError::InvalidParams`] (-32602).
/// 4. **Output-cap guard** — `max_results > MAX_EXPLORE_LIMIT` (or `==
///    0`) rejects as [`MCPError::InvalidParams`] (-32602). The message
///    names `max_results` AND the cap so a caller can self-correct.
/// 5. **Permission resolution + seed gate** — principal-scoped calls
///    resolve one effective-permission snapshot. A denied seed returns
///    the same query-error shape as a missing seed without touching storage.
/// 6. **Explorer invocation + per-node gate** — every node is checked;
///    denied nodes, incident edges, and paths through them are omitted.
/// 7. **Output cap** — `apply_output_cap` clips nodes+edges to
///    `max_results` and sets [`Neighborhood::truncated`].
pub fn explore_capped<E: NeighborhoodExplorer + ?Sized>(
    explorer: &E,
    session_tenant: TenantId,
    session_scope: SessionScope,
    cancel: &CancellationToken,
    req: &ExploreRequest,
) -> Result<Neighborhood, MCPError> {
    let request_tenant = TenantId::new(req.tenant_id);
    if request_tenant != session_tenant {
        return Err(MCPError::Unauthorized);
    }

    let access = authorize_read(
        "graph.explore",
        req.principal.as_deref(),
        session_scope,
        || explorer.permission_index(request_tenant, cancel),
    )?;

    let max_depth = req.max_depth.unwrap_or(DEFAULT_EXPLORE_DEPTH);
    if max_depth > MAX_EXPLORE_DEPTH {
        return Err(MCPError::InvalidParams(format!(
            "graph.explore: max_depth={max_depth} exceeds hard cap {MAX_EXPLORE_DEPTH}"
        )));
    }
    let max_results = req.max_results.unwrap_or(DEFAULT_EXPLORE_LIMIT);
    if max_results > MAX_EXPLORE_LIMIT {
        return Err(MCPError::InvalidParams(format!(
            "graph.explore: max_results={max_results} exceeds hard cap {MAX_EXPLORE_LIMIT}"
        )));
    }
    if max_results == 0 {
        return Err(MCPError::InvalidParams(
            "graph.explore: max_results must be ≥ 1".into(),
        ));
    }
    // Seed authorization is an early optimization and existence-oracle
    // defense, not the whole policy. The per-neighbor filter below is
    // independently load-bearing (#1488 RED-on-revert gate).
    if !access.allows(NodeId::new(req.seed)) {
        return Err(MCPError::QueryError(format!("seed {} not found", req.seed)));
    }

    let rel_filter = req.rel_types.as_deref();
    // ADR-217: default to outbound when the field is omitted — a
    // pre-ADR-217 request behaves exactly as before.
    let direction = req.direction.unwrap_or_default();
    let mut neighborhood = explorer.explore(
        request_tenant,
        req.seed,
        max_depth,
        rel_filter,
        direction,
        cancel,
    )?;
    // Preserve the pre-ADR-212 SYSTEM-TRUSTED result byte-for-byte. The
    // reachability recomputation is needed only after a principal filter has
    // removed nodes; running it on an unfiltered provider could reinterpret a
    // provider-owned depth annotation or discard an intentionally partial
    // neighborhood.
    if !access.is_system_trusted() {
        retain_visible_reachable(&mut neighborhood, &access, direction);
    }
    // Output cap owns `truncated` (the explorer leaves it false).
    neighborhood.truncated = apply_output_cap(&mut neighborhood, max_results);
    Ok(neighborhood)
}

/// Serialize + render a (possibly capped) [`Neighborhood`] into the
/// standard `{format, body}` envelope.
pub fn render_neighborhood(
    format: ResponseFormat,
    neighborhood: &Neighborhood,
) -> Result<serde_json::Value, MCPError> {
    let value = serde_json::to_value(neighborhood)
        .map_err(|e| MCPError::InternalError(format!("neighborhood serialize: {e}")))?;
    crate::tools::render_response(format, &value)
}

/// `graph.explore` — return per-seed neighborhood as JSON-RPC `result`.
///
/// # Cross-tenant guard
///
/// Same shape as [`crate::tools::schema::schema_tool`] /
/// [`crate::tools::inspect::inspect_tool`]: cross-tenant requests
/// reject as [`MCPError::Unauthorized`] before any explorer call.
///
/// # Depth-cap guard
///
/// Requests with `max_depth > MAX_EXPLORE_DEPTH` reject as
/// [`MCPError::InvalidParams`] BEFORE the explorer is touched. The
/// rejection message names the offending value AND the cap so a
/// caller can self-correct without a second round-trip.
///
/// # Output-cap guard (W30 #900)
///
/// Requests with `max_results > MAX_EXPLORE_LIMIT` (or `== 0`) reject
/// as [`MCPError::InvalidParams`]. Otherwise the combined node+edge
/// count is capped to `max_results` (default
/// [`DEFAULT_EXPLORE_LIMIT`]) and [`Neighborhood::truncated`] is set
/// when the cap clipped anything.
///
/// # Cancellation
///
/// `cancel` is the cancellation token bound to this request. Callers
/// (the dispatcher) plumb in a fresh token per JSON-RPC request; if
/// the token trips mid-explore the impl surfaces
/// [`MCPError::Cancelled`].
///
/// # Errors
///
/// - [`MCPError::Unauthorized`] — cross-tenant request.
/// - [`MCPError::InvalidParams`] — `max_depth > MAX_EXPLORE_DEPTH`, or
///   `max_results > MAX_EXPLORE_LIMIT` / `max_results == 0`.
/// - [`MCPError::TenantUnknown`] / [`MCPError::QueryError`] —
///   propagated from [`NeighborhoodExplorer::explore`].
/// - [`MCPError::Cancelled`] — the cancellation token tripped.
/// - [`MCPError::InternalError`] — serializer encode failure.
pub fn explore_tool<E: NeighborhoodExplorer + ?Sized>(
    explorer: &E,
    session_tenant: TenantId,
    session_scope: SessionScope,
    cancel: &CancellationToken,
    req: ExploreRequest,
) -> Result<serde_json::Value, MCPError> {
    let neighborhood = explore_capped(explorer, session_tenant, session_scope, cancel, &req)?;
    let format = req.format.unwrap_or(ResponseFormat::Toon);
    render_neighborhood(format, &neighborhood)
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::read_acl::PERMISSION_INDEX_SLUG;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// In-memory fixture: tenant → seed → Neighborhood.
    #[derive(Debug, Clone, Default)]
    struct StubExplorer {
        bound_tenant: Option<TenantId>,
        seeds: std::collections::HashMap<u64, Neighborhood>,
        permissions: Option<Arc<PermissionIndex>>,
    }

    impl StubExplorer {
        fn new(tenant: TenantId) -> Self {
            Self {
                bound_tenant: Some(tenant),
                seeds: Default::default(),
                permissions: None,
            }
        }
        fn with_seed(mut self, n: Neighborhood) -> Self {
            self.seeds.insert(n.seed, n);
            self
        }

        fn with_permissions(mut self, permissions: Arc<PermissionIndex>) -> Self {
            self.permissions = Some(permissions);
            self
        }
    }

    impl NeighborhoodExplorer for StubExplorer {
        fn explore(
            &self,
            tenant: TenantId,
            seed: u64,
            max_depth: u32,
            _rel_filter: Option<&[String]>,
            _direction: ExploreDirection,
            cancel: &CancellationToken,
        ) -> Result<Neighborhood, MCPError> {
            // Per the trait's cancellation contract: check before
            // touching the (would-be storage-side) fixture.
            if cancel.is_cancelled() {
                return Err(MCPError::Cancelled);
            }
            match self.bound_tenant {
                Some(t) if t == tenant => match self.seeds.get(&seed).cloned() {
                    Some(mut n) => {
                        // Truncate fixture to the caller's max_depth
                        // (sticking to the trait contract that the
                        // impl honors max_depth, not the tool body).
                        n.nodes.retain(|nn| nn.depth <= max_depth);
                        let allowed: std::collections::HashSet<u64> =
                            n.nodes.iter().map(|nn| nn.id).collect();
                        n.edges
                            .retain(|e| allowed.contains(&e.from) && allowed.contains(&e.to));
                        n.max_depth = max_depth;
                        Ok(n)
                    }
                    None => Err(MCPError::QueryError(format!("seed {seed} not found"))),
                },
                _ => Err(MCPError::TenantUnknown(format!("{tenant:?}"))),
            }
        }

        fn permission_index(
            &self,
            _tenant: TenantId,
            cancel: &CancellationToken,
        ) -> Result<Option<Arc<PermissionIndex>>, MCPError> {
            if cancel.is_cancelled() {
                return Err(MCPError::Cancelled);
            }
            Ok(self.permissions.clone())
        }
    }

    /// Counting explorer — fires a stub neighborhood and increments a
    /// call counter so tests can pin "we never reached the explorer".
    #[derive(Debug)]
    struct CountingExplorer {
        bound_tenant: TenantId,
        seed_id: u64,
        calls: Arc<AtomicUsize>,
        permissions: Option<Arc<PermissionIndex>>,
    }

    impl NeighborhoodExplorer for CountingExplorer {
        fn explore(
            &self,
            tenant: TenantId,
            seed: u64,
            max_depth: u32,
            _rel_filter: Option<&[String]>,
            _direction: ExploreDirection,
            _cancel: &CancellationToken,
        ) -> Result<Neighborhood, MCPError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if tenant != self.bound_tenant {
                return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
            }
            Ok(Neighborhood {
                seed,
                max_depth,
                nodes: vec![NeighborhoodNode {
                    id: self.seed_id,
                    label: Some("Person".into()),
                    depth: 0,
                    properties: BTreeMap::new(),
                }],
                edges: vec![],
                truncated: false,
            })
        }

        fn permission_index(
            &self,
            _tenant: TenantId,
            _cancel: &CancellationToken,
        ) -> Result<Option<Arc<PermissionIndex>>, MCPError> {
            Ok(self.permissions.clone())
        }
    }

    fn neighborhood_fixture() -> Neighborhood {
        let mut props_a: BTreeMap<String, serde_json::Value> = BTreeMap::new();
        props_a.insert("name".into(), serde_json::json!("Alice"));
        let mut props_b: BTreeMap<String, serde_json::Value> = BTreeMap::new();
        props_b.insert("name".into(), serde_json::json!("Bob"));
        let mut props_c: BTreeMap<String, serde_json::Value> = BTreeMap::new();
        props_c.insert("name".into(), serde_json::json!("Carol"));
        Neighborhood {
            seed: 1,
            max_depth: 2,
            nodes: vec![
                NeighborhoodNode {
                    id: 1,
                    label: Some("Person".into()),
                    depth: 0,
                    properties: props_a,
                },
                NeighborhoodNode {
                    id: 2,
                    label: Some("Person".into()),
                    depth: 1,
                    properties: props_b,
                },
                NeighborhoodNode {
                    id: 3,
                    label: Some("Person".into()),
                    depth: 2,
                    properties: props_c,
                },
            ],
            edges: vec![
                NeighborhoodEdge {
                    from: 1,
                    to: 2,
                    rel_type: Some("KNOWS".into()),
                    direction: NeighborDirection::Out,
                },
                NeighborhoodEdge {
                    from: 2,
                    to: 3,
                    rel_type: Some("KNOWS".into()),
                    direction: NeighborDirection::Out,
                },
            ],
            truncated: false,
        }
    }

    /// High-fanout hub fixture (issue #900 repro shape): a seed at
    /// depth 0 plus `leaves` depth-1 leaves, each connected by one
    /// out-edge. Total = `(leaves + 1)` nodes + `leaves` edges.
    fn high_fanout_fixture(seed: u64, leaves: u64) -> Neighborhood {
        let mut nodes = vec![NeighborhoodNode {
            id: seed,
            label: Some("Hub".into()),
            depth: 0,
            properties: BTreeMap::new(),
        }];
        let mut edges = Vec::new();
        for i in 0..leaves {
            let leaf = seed + 1 + i;
            nodes.push(NeighborhoodNode {
                id: leaf,
                label: Some("Leaf".into()),
                depth: 1,
                properties: BTreeMap::new(),
            });
            edges.push(NeighborhoodEdge {
                from: seed,
                to: leaf,
                rel_type: Some("LINKS".into()),
                direction: NeighborDirection::Out,
            });
        }
        Neighborhood {
            seed,
            max_depth: 1,
            nodes,
            edges,
            truncated: false,
        }
    }

    #[test]
    fn explore_tool_returns_seed_plus_neighbors_at_depth_2() {
        let e = StubExplorer::new(TenantId::new(1)).with_seed(neighborhood_fixture());
        let req = ExploreRequest {
            tenant_id: 1,
            seed: 1,
            max_depth: Some(2),
            max_results: None,
            rel_types: None,
            direction: None,
            format: Some(ResponseFormat::Json),
            principal: None,
        };
        let token = CancellationToken::new();
        let resp =
            explore_tool(&e, TenantId::new(1), SessionScope::Power, &token, req).expect("ok");
        assert_eq!(resp["format"], "json");
        let body = resp["body"].as_str().expect("json body");
        assert!(body.contains("Alice"), "seed in body");
        assert!(body.contains("Bob"), "depth-1 neighbor in body");
        assert!(body.contains("Carol"), "depth-2 neighbor in body");
    }

    #[test]
    fn explore_tool_rejects_max_depth_over_hard_cap() {
        // Hostile request: depth=999. MUST reject as InvalidParams
        // BEFORE any explorer call. Verify "no explorer call" by using
        // an explorer with a zeroed call counter; we expect 0 calls
        // after the rejection.
        let calls = Arc::new(AtomicUsize::new(0));
        let e = CountingExplorer {
            bound_tenant: TenantId::new(1),
            seed_id: 1,
            calls: calls.clone(),
            permissions: None,
        };
        let req = ExploreRequest {
            tenant_id: 1,
            seed: 1,
            max_depth: Some(999),
            max_results: None,
            rel_types: None,
            direction: None,
            format: None,
            principal: None,
        };
        let token = CancellationToken::new();
        let err = explore_tool(&e, TenantId::new(1), SessionScope::Power, &token, req)
            .expect_err("must reject");
        assert_eq!(err.code(), -32602);
        match &err {
            MCPError::InvalidParams(msg) => {
                assert!(msg.contains("999"));
                assert!(msg.contains(&MAX_EXPLORE_DEPTH.to_string()));
            }
            other => panic!("expected InvalidParams, got {other:?}"),
        }
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "explorer must not be called"
        );
    }

    #[test]
    fn explore_tool_rejects_cross_tenant_request_with_unauthorized() {
        // Session = tenant 1, request asks for tenant 2 — MUST reject
        // BEFORE any explorer call. Verify via counter as above.
        let calls = Arc::new(AtomicUsize::new(0));
        let e = CountingExplorer {
            bound_tenant: TenantId::new(1),
            seed_id: 1,
            calls: calls.clone(),
            permissions: None,
        };
        let req = ExploreRequest {
            tenant_id: 2,
            seed: 1,
            max_depth: Some(2),
            max_results: None,
            rel_types: None,
            direction: None,
            format: None,
            principal: None,
        };
        let token = CancellationToken::new();
        let err = explore_tool(&e, TenantId::new(1), SessionScope::Power, &token, req)
            .expect_err("must reject");
        assert_eq!(err.code(), -32002);
        assert!(matches!(err, MCPError::Unauthorized));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn explore_tool_default_depth_is_two() {
        // Pin the DEFAULT_EXPLORE_DEPTH binding: omitting the
        // max_depth field on the wire defaults to 2. The fixture
        // contains depth-0 / 1 / 2 nodes — all three MUST appear in
        // the body.
        let e = StubExplorer::new(TenantId::new(1)).with_seed(neighborhood_fixture());
        let req = ExploreRequest {
            tenant_id: 1,
            seed: 1,
            max_depth: None,
            max_results: None,
            rel_types: None,
            direction: None,
            format: Some(ResponseFormat::Json),
            principal: None,
        };
        let token = CancellationToken::new();
        let resp =
            explore_tool(&e, TenantId::new(1), SessionScope::Power, &token, req).expect("ok");
        let body = resp["body"].as_str().unwrap();
        assert!(body.contains("Alice"));
        assert!(body.contains("Bob"));
        assert!(body.contains("Carol"));
    }

    #[test]
    fn explore_tool_default_format_is_toon() {
        // Pin the W14β wire contract: graph.explore default = TOON
        // (uniform-shape rows; design-v2 §9.3 token-savings path).
        let e = StubExplorer::new(TenantId::new(1)).with_seed(neighborhood_fixture());
        let req = ExploreRequest {
            tenant_id: 1,
            seed: 1,
            max_depth: Some(1),
            max_results: None,
            rel_types: None,
            direction: None,
            format: None,
            principal: None,
        };
        let token = CancellationToken::new();
        let resp =
            explore_tool(&e, TenantId::new(1), SessionScope::Power, &token, req).expect("ok");
        assert_eq!(resp["format"], "toon");
    }

    #[test]
    fn explore_tool_surfaces_cancelled_when_token_tripped() {
        // Trip the token BEFORE the call; the stub honors the
        // cancellation contract and short-circuits with
        // MCPError::Cancelled (-32001).
        let e = StubExplorer::new(TenantId::new(1)).with_seed(neighborhood_fixture());
        let req = ExploreRequest {
            tenant_id: 1,
            seed: 1,
            max_depth: Some(2),
            max_results: None,
            rel_types: None,
            direction: None,
            format: None,
            principal: None,
        };
        let token = CancellationToken::new();
        token.cancel();
        let err = explore_tool(&e, TenantId::new(1), SessionScope::Power, &token, req)
            .expect_err("cancelled");
        assert_eq!(err.code(), -32001);
        assert!(matches!(err, MCPError::Cancelled));
    }

    #[test]
    fn explore_tool_propagates_seed_not_found_as_query_error() {
        let e = StubExplorer::new(TenantId::new(1));
        let req = ExploreRequest {
            tenant_id: 1,
            seed: 999,
            max_depth: Some(2),
            max_results: None,
            rel_types: None,
            direction: None,
            format: None,
            principal: None,
        };
        let token = CancellationToken::new();
        let err = explore_tool(&e, TenantId::new(1), SessionScope::Power, &token, req)
            .expect_err("missing seed");
        assert_eq!(err.code(), -32005);
        match err {
            MCPError::QueryError(msg) => assert!(msg.contains("999")),
            other => panic!("expected QueryError, got {other:?}"),
        }
    }

    #[test]
    fn explore_request_rejects_unknown_field() {
        // code-quality policy strict-mode discipline.
        let v = serde_json::json!({
            "tenant_id": 1,
            "seed": 1,
            "depth": 2,  // typo of `max_depth`
        });
        let res: Result<ExploreRequest, _> = serde_json::from_value(v);
        assert!(res.is_err(), "typo must reject");
    }

    #[test]
    fn explore_tool_honors_smaller_max_depth_truncating_neighborhood() {
        // Caller asks for depth=1 against a depth-2 fixture; the
        // depth-2 node (Carol) MUST be dropped, but Alice + Bob remain.
        let e = StubExplorer::new(TenantId::new(1)).with_seed(neighborhood_fixture());
        let req = ExploreRequest {
            tenant_id: 1,
            seed: 1,
            max_depth: Some(1),
            max_results: None,
            rel_types: None,
            direction: None,
            format: Some(ResponseFormat::Json),
            principal: None,
        };
        let token = CancellationToken::new();
        let resp =
            explore_tool(&e, TenantId::new(1), SessionScope::Power, &token, req).expect("ok");
        let body = resp["body"].as_str().unwrap();
        assert!(body.contains("Alice"));
        assert!(body.contains("Bob"));
        assert!(!body.contains("Carol"), "depth-2 node truncated");
    }

    // ─────────────────────────────────────────────────────────────────
    // #1488 — per-principal traversal authorization
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn principal_scoped_explore_omits_denied_neighbor_and_paths_through_it_1488() {
        // S(1) is visible to alice, D(2) is denied, and V(3) is visible
        // to alice but reachable only through D. Seed-only gating would
        // leak both D's content and V's denied-path-derived depth.
        let permissions = Arc::new(PermissionIndex::new());
        permissions.apply_doc_acl(
            NodeId::new(1),
            std::collections::BTreeSet::from(["alice".to_owned()]),
        );
        permissions.apply_doc_acl(
            NodeId::new(2),
            std::collections::BTreeSet::from(["bob".to_owned()]),
        );
        permissions.apply_doc_acl(
            NodeId::new(3),
            std::collections::BTreeSet::from(["alice".to_owned()]),
        );
        let e = StubExplorer::new(TenantId::new(1))
            .with_seed(neighborhood_fixture())
            .with_permissions(permissions);
        let req = ExploreRequest {
            tenant_id: 1,
            seed: 1,
            max_depth: Some(2),
            max_results: None,
            rel_types: None,
            direction: None,
            format: Some(ResponseFormat::Json),
            principal: Some("alice".into()),
        };
        let token = CancellationToken::new();
        let resp = explore_tool(&e, TenantId::new(1), SessionScope::Read, &token, req)
            .expect("principal-scoped traversal");
        let body = resp["body"].as_str().expect("json body");
        let neighborhood: Neighborhood = serde_json::from_str(body).expect("neighborhood");
        assert_eq!(
            neighborhood
                .nodes
                .iter()
                .map(|node| node.id)
                .collect::<Vec<_>>(),
            vec![1],
            "denied D and visible V behind D must both be absent"
        );
        assert!(neighborhood.edges.is_empty(), "no denied incident edge");
        assert!(
            !body.contains("Bob"),
            "denied neighbor content must not leak"
        );
        assert!(
            !body.contains("Carol"),
            "content reachable only through a denied node must not leak"
        );
    }

    #[test]
    fn principal_scoped_explore_recomputes_depth_over_authorized_alternate_path_1488() {
        // The explorer first reaches V(3) in two hops through denied D(2),
        // but alice also has a three-hop route S(1)->A(4)->B(5)->V(3).
        // ACL filtering must keep the authorized route and replace V's
        // pre-filter depth=2 with depth=3, revealing no denied shortcut.
        let permissions = Arc::new(PermissionIndex::new());
        for id in [1, 3, 4, 5] {
            permissions.apply_doc_acl(
                NodeId::new(id),
                std::collections::BTreeSet::from(["alice".to_owned()]),
            );
        }
        permissions.apply_doc_acl(
            NodeId::new(2),
            std::collections::BTreeSet::from(["bob".to_owned()]),
        );
        let nodes = [
            (1, 0, "seed"),
            (2, 1, "denied"),
            (4, 1, "allowed-a"),
            (3, 2, "visible-target"),
            (5, 2, "allowed-b"),
        ]
        .into_iter()
        .map(|(id, depth, body)| NeighborhoodNode {
            id,
            label: Some("Document".into()),
            depth,
            properties: BTreeMap::from([("body".into(), serde_json::json!(body))]),
        })
        .collect();
        let edges = [(1, 2), (1, 4), (2, 3), (4, 5), (5, 3)]
            .into_iter()
            .map(|(from, to)| NeighborhoodEdge {
                from,
                to,
                rel_type: Some("LINKS_TO".into()),
                direction: NeighborDirection::Out,
            })
            .collect();
        let e = StubExplorer::new(TenantId::new(1))
            .with_seed(Neighborhood {
                seed: 1,
                max_depth: 3,
                nodes,
                edges,
                truncated: false,
            })
            .with_permissions(permissions);
        let req = ExploreRequest {
            tenant_id: 1,
            seed: 1,
            max_depth: Some(3),
            max_results: None,
            rel_types: None,
            direction: Some(ExploreDirection::Out),
            format: None,
            principal: Some("alice".into()),
        };
        let token = CancellationToken::new();
        let neighborhood = explore_capped(&e, TenantId::new(1), SessionScope::Read, &token, &req)
            .expect("authorized alternate path");

        assert_eq!(
            neighborhood
                .nodes
                .iter()
                .map(|node| node.id)
                .collect::<Vec<_>>(),
            vec![1, 4, 3, 5]
        );
        assert_eq!(
            neighborhood
                .nodes
                .iter()
                .find(|node| node.id == 3)
                .expect("visible target")
                .depth,
            3,
            "depth must be recomputed without the denied shortcut"
        );
        assert!(
            neighborhood
                .edges
                .iter()
                .all(|edge| edge.from != 2 && edge.to != 2),
            "all incident edges to the denied node must be absent"
        );

        // The same raw fixture places the target at pre-filter depth 2
        // through the denied shortcut. If the caller caps at 2, the
        // authorized three-hop alternate route is over-depth and the
        // target must be omitted rather than returned with depth 3.
        let mut capped_req = req;
        capped_req.max_depth = Some(2);
        let capped = explore_capped(
            &e,
            TenantId::new(1),
            SessionScope::Read,
            &token,
            &capped_req,
        )
        .expect("authorized graph under tighter depth cap");
        assert!(
            capped.nodes.iter().all(|node| node.id != 3),
            "ACL-safe recomputed depth must still honor max_depth"
        );
    }

    #[test]
    fn absent_principal_non_power_explore_fails_closed_minus_32008_1488() {
        let calls = Arc::new(AtomicUsize::new(0));
        let e = CountingExplorer {
            bound_tenant: TenantId::new(1),
            seed_id: 1,
            calls: calls.clone(),
            permissions: None,
        };
        let req = ExploreRequest {
            tenant_id: 1,
            seed: 1,
            max_depth: Some(1),
            max_results: None,
            rel_types: None,
            direction: None,
            format: Some(ResponseFormat::Json),
            principal: None,
        };
        let token = CancellationToken::new();
        let err = explore_tool(&e, TenantId::new(1), SessionScope::Read, &token, req)
            .expect_err("missing principal must fail closed");
        assert_eq!(err.code(), -32008);
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "fail-closed gate must run before explorer storage"
        );
    }

    #[test]
    fn principal_scoped_explore_without_permission_index_fails_closed_1488() {
        let e = StubExplorer::new(TenantId::new(1)).with_seed(neighborhood_fixture());
        let req = ExploreRequest {
            tenant_id: 1,
            seed: 1,
            max_depth: Some(1),
            max_results: None,
            rel_types: None,
            direction: None,
            format: Some(ResponseFormat::Json),
            principal: Some("alice".into()),
        };
        let token = CancellationToken::new();
        let err = explore_tool(&e, TenantId::new(1), SessionScope::Read, &token, req)
            .expect_err("missing permission index must never run unfiltered");
        assert_eq!(err.code(), -32004);
        assert!(format!("{err}").contains(PERMISSION_INDEX_SLUG));
    }

    #[test]
    fn denied_seed_matches_missing_seed_without_storage_1488() {
        let permissions = Arc::new(PermissionIndex::new());
        permissions.apply_doc_acl(
            NodeId::new(1),
            std::collections::BTreeSet::from(["bob".to_owned()]),
        );
        let calls = Arc::new(AtomicUsize::new(0));
        let e = CountingExplorer {
            bound_tenant: TenantId::new(1),
            seed_id: 1,
            calls: calls.clone(),
            permissions: Some(permissions),
        };
        let req = ExploreRequest {
            tenant_id: 1,
            seed: 1,
            max_depth: Some(1),
            max_results: None,
            rel_types: None,
            direction: None,
            format: Some(ResponseFormat::Json),
            principal: Some("alice".into()),
        };
        let token = CancellationToken::new();
        let err = explore_tool(&e, TenantId::new(1), SessionScope::Read, &token, req)
            .expect_err("denied seed must look missing");
        assert_eq!(err.code(), -32005);
        assert_eq!(format!("{err}"), "query error: seed 1 not found");
        assert_eq!(calls.load(Ordering::SeqCst), 0, "no storage call");
    }

    // ─────────────────────────────────────────────────────────────────
    // W30 #900 — output-cap tests (mirror raw_query.rs:728-746)
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn explore_tool_caps_high_fanout_hub_and_signals_truncated() {
        // issue #900 repro: a depth-1 hub with 300 leaves yields 301
        // nodes + 300 edges = 601 items. With max_results=50 the
        // COMBINED output MUST be bounded to <= 50 AND `truncated:true`
        // MUST be set. (RED-on-revert anchor: this assertion FAILS when
        // the apply_output_cap call in explore_capped is neutered —
        // the body returns the full 601 items.)
        let e = StubExplorer::new(TenantId::new(1)).with_seed(high_fanout_fixture(1, 300));
        let req = ExploreRequest {
            tenant_id: 1,
            seed: 1,
            max_depth: Some(1),
            max_results: Some(50),
            rel_types: None,
            direction: None,
            format: Some(ResponseFormat::Json),
            principal: None,
        };
        let token = CancellationToken::new();
        let resp =
            explore_tool(&e, TenantId::new(1), SessionScope::Power, &token, req).expect("ok");
        let body = resp["body"].as_str().expect("json body");
        let n: Neighborhood = serde_json::from_str(body).expect("parse neighborhood");
        assert!(
            n.nodes.len() + n.edges.len() <= 50,
            "combined output bounded to cap; got {} nodes + {} edges",
            n.nodes.len(),
            n.edges.len()
        );
        assert!(n.truncated, "truncated must be set when the cap clipped");
        assert!(
            n.nodes.iter().any(|node| node.id == 1),
            "seed (depth 0) retained under nodes-first policy"
        );
        assert!(
            body.contains("\"truncated\":true"),
            "truncated:true on the wire: {}",
            &body[..body.len().min(120)]
        );
    }

    #[test]
    fn explore_tool_default_max_results_bounds_to_default_limit() {
        // max_results omitted on a 2001-item hub (1000 leaves → 1001
        // nodes + 1000 edges) → bounded to DEFAULT_EXPLORE_LIMIT (1000)
        // with truncated=true.
        let e = StubExplorer::new(TenantId::new(1)).with_seed(high_fanout_fixture(1, 1000));
        let req = ExploreRequest {
            tenant_id: 1,
            seed: 1,
            max_depth: Some(1),
            max_results: None, // -> DEFAULT_EXPLORE_LIMIT
            rel_types: None,
            direction: None,
            format: Some(ResponseFormat::Json),
            principal: None,
        };
        let token = CancellationToken::new();
        let resp =
            explore_tool(&e, TenantId::new(1), SessionScope::Power, &token, req).expect("ok");
        let body = resp["body"].as_str().unwrap();
        let n: Neighborhood = serde_json::from_str(body).unwrap();
        assert!(
            n.nodes.len() + n.edges.len() <= DEFAULT_EXPLORE_LIMIT as usize,
            "combined output bounded to the default cap; got {} + {}",
            n.nodes.len(),
            n.edges.len()
        );
        assert!(n.truncated, "2001-item hub truncated at default cap");
    }

    #[test]
    fn explore_tool_rejects_max_results_above_cap() {
        // Pin the MAX_EXPLORE_LIMIT = 10_000 hard cap. The rejection
        // message MUST name `max_results` AND the cap.
        let e = StubExplorer::new(TenantId::new(1)).with_seed(neighborhood_fixture());
        let req = ExploreRequest {
            tenant_id: 1,
            seed: 1,
            max_depth: Some(1),
            max_results: Some(MAX_EXPLORE_LIMIT + 1),
            rel_types: None,
            direction: None,
            format: None,
            principal: None,
        };
        let token = CancellationToken::new();
        let err = explore_tool(&e, TenantId::new(1), SessionScope::Power, &token, req)
            .expect_err("must reject");
        assert_eq!(err.code(), -32602);
        match err {
            MCPError::InvalidParams(msg) => {
                assert!(msg.contains("max_results"), "names the param: {msg}");
                assert!(
                    msg.contains(&MAX_EXPLORE_LIMIT.to_string()),
                    "names the cap: {msg}"
                );
            }
            other => panic!("expected InvalidParams, got {other:?}"),
        }
    }

    #[test]
    fn explore_tool_rejects_max_results_zero() {
        // Mirror raw_query's max_rows==0 rejection — a zero cap is a
        // client bug, not "return nothing".
        let e = StubExplorer::new(TenantId::new(1)).with_seed(neighborhood_fixture());
        let req = ExploreRequest {
            tenant_id: 1,
            seed: 1,
            max_depth: Some(1),
            max_results: Some(0),
            rel_types: None,
            direction: None,
            format: None,
            principal: None,
        };
        let token = CancellationToken::new();
        let err = explore_tool(&e, TenantId::new(1), SessionScope::Power, &token, req)
            .expect_err("must reject");
        assert_eq!(err.code(), -32602);
    }

    #[test]
    fn explore_tool_small_neighborhood_not_truncated() {
        // The 3-node/2-edge fixture (5 items) under the default cap →
        // no truncation; the wire shape carries truncated:false.
        let e = StubExplorer::new(TenantId::new(1)).with_seed(neighborhood_fixture());
        let req = ExploreRequest {
            tenant_id: 1,
            seed: 1,
            max_depth: Some(2),
            max_results: None,
            rel_types: None,
            direction: None,
            format: Some(ResponseFormat::Json),
            principal: None,
        };
        let token = CancellationToken::new();
        let resp =
            explore_tool(&e, TenantId::new(1), SessionScope::Power, &token, req).expect("ok");
        let body = resp["body"].as_str().unwrap();
        let n: Neighborhood = serde_json::from_str(body).unwrap();
        assert!(!n.truncated, "small neighborhood not truncated");
        assert_eq!(n.nodes.len(), 3);
        assert_eq!(n.edges.len(), 2);
        assert!(
            body.contains("\"truncated\":false"),
            "truncated:false on the wire: {body}"
        );
    }

    #[test]
    fn explore_tool_truncation_policy_is_nodes_first_seed_preserved() {
        // Policy pin: nodes-first combined-cap. A hub (seed depth 0 +
        // 300 leaves depth 1) capped at 10 keeps the seed + 9 closest
        // nodes and 0 edges (nodes ate the budget); combined == 10.
        let e = StubExplorer::new(TenantId::new(1)).with_seed(high_fanout_fixture(1, 300));
        let req = ExploreRequest {
            tenant_id: 1,
            seed: 1,
            max_depth: Some(1),
            max_results: Some(10),
            rel_types: None,
            direction: None,
            format: Some(ResponseFormat::Json),
            principal: None,
        };
        let token = CancellationToken::new();
        let resp =
            explore_tool(&e, TenantId::new(1), SessionScope::Power, &token, req).expect("ok");
        let n: Neighborhood = serde_json::from_str(resp["body"].as_str().unwrap()).unwrap();
        assert_eq!(n.nodes.len(), 10, "nodes-first fills the cap");
        assert_eq!(n.edges.len(), 0, "no edge budget left after nodes");
        assert!(
            n.nodes.iter().any(|x| x.id == 1),
            "seed (depth 0) preserved by the stable depth sort"
        );
        assert!(n.truncated);
    }

    #[test]
    fn apply_output_cap_keeps_coherent_edges_within_budget() {
        // Direct unit test of the policy helper: a small graph where
        // nodes fit but edges overflow. cap=4 on 2 nodes + 5 edges (all
        // between the 2 nodes) → keep 2 nodes, edge_budget=2, keep 2
        // coherent edges, drop 3. truncated=true.
        let mut n = Neighborhood {
            seed: 1,
            max_depth: 1,
            nodes: vec![
                NeighborhoodNode {
                    id: 1,
                    label: None,
                    depth: 0,
                    properties: BTreeMap::new(),
                },
                NeighborhoodNode {
                    id: 2,
                    label: None,
                    depth: 1,
                    properties: BTreeMap::new(),
                },
            ],
            edges: (0..5)
                .map(|_| NeighborhoodEdge {
                    from: 1,
                    to: 2,
                    rel_type: Some("R".into()),
                    direction: NeighborDirection::Out,
                })
                .collect(),
            truncated: false,
        };
        let clipped = apply_output_cap(&mut n, 4);
        assert!(clipped, "cap clipped edges");
        assert_eq!(n.nodes.len(), 2, "both nodes fit");
        assert_eq!(n.edges.len(), 2, "edges clipped to cap - nodes = 2");
    }

    #[test]
    fn apply_output_cap_drops_edges_with_cut_endpoints() {
        // When nodes are cut, edges referencing a cut node are dropped
        // (no dangling edges). seed(0) + leaf2(1) + leaf3(1); cap=1
        // keeps only the seed; both edges reference cut leaves → 0
        // edges, 1 node, truncated.
        let mut n = Neighborhood {
            seed: 1,
            max_depth: 1,
            nodes: vec![
                NeighborhoodNode {
                    id: 1,
                    label: None,
                    depth: 0,
                    properties: BTreeMap::new(),
                },
                NeighborhoodNode {
                    id: 2,
                    label: None,
                    depth: 1,
                    properties: BTreeMap::new(),
                },
                NeighborhoodNode {
                    id: 3,
                    label: None,
                    depth: 1,
                    properties: BTreeMap::new(),
                },
            ],
            edges: vec![
                NeighborhoodEdge {
                    from: 1,
                    to: 2,
                    rel_type: None,
                    direction: NeighborDirection::Out,
                },
                NeighborhoodEdge {
                    from: 1,
                    to: 3,
                    rel_type: None,
                    direction: NeighborDirection::Out,
                },
            ],
            truncated: false,
        };
        let clipped = apply_output_cap(&mut n, 1);
        assert!(clipped);
        assert_eq!(n.nodes.len(), 1);
        assert_eq!(n.nodes[0].id, 1, "seed kept");
        assert_eq!(n.edges.len(), 0, "edges to cut leaves dropped");
    }

    /// Records the `rel_filter` argument observed by the adapter so the
    /// PR #292 review LOW-1 unit test can pin the wire-plumbing
    /// `req.rel_types → NeighborhoodExplorer::explore`.
    #[derive(Debug)]
    struct RelFilterRecordingExplorer {
        bound_tenant: TenantId,
        last_filter: std::sync::Mutex<Option<Vec<String>>>,
        // ADR-217: also record the direction the tool boundary plumbed.
        last_direction: std::sync::Mutex<Option<ExploreDirection>>,
    }

    impl NeighborhoodExplorer for RelFilterRecordingExplorer {
        fn explore(
            &self,
            tenant: TenantId,
            seed: u64,
            max_depth: u32,
            rel_filter: Option<&[String]>,
            direction: ExploreDirection,
            _cancel: &CancellationToken,
        ) -> Result<Neighborhood, MCPError> {
            *self.last_filter.lock().unwrap() = rel_filter.map(<[String]>::to_vec);
            *self.last_direction.lock().unwrap() = Some(direction);
            if tenant != self.bound_tenant {
                return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
            }
            Ok(Neighborhood {
                seed,
                max_depth,
                nodes: vec![NeighborhoodNode {
                    id: seed,
                    label: Some("Person".into()),
                    depth: 0,
                    properties: BTreeMap::new(),
                }],
                edges: vec![],
                truncated: false,
            })
        }
    }

    #[test]
    fn explore_tool_plumbs_rel_filter_through_to_adapter() {
        // PR #292 review LOW-1 — pin the `req.rel_types →
        // NeighborhoodExplorer::explore`'s `rel_filter` plumbing. A bug
        // like "tool body drops the rel_filter" or "tool body passes the
        // wrong field" would surface as a None / mismatched recorded
        // value here.
        let e = RelFilterRecordingExplorer {
            bound_tenant: TenantId::new(1),
            last_filter: std::sync::Mutex::new(None),
            last_direction: std::sync::Mutex::new(None),
        };
        let req = ExploreRequest {
            tenant_id: 1,
            seed: 1,
            max_depth: Some(1),
            max_results: None,
            rel_types: Some(vec!["KNOWS".into(), "WORKS_WITH".into()]),
            direction: None,
            format: Some(ResponseFormat::Json),
            principal: None,
        };
        let token = CancellationToken::new();
        explore_tool(&e, TenantId::new(1), SessionScope::Power, &token, req).expect("ok");
        let observed = e.last_filter.lock().unwrap().clone();
        assert_eq!(
            observed,
            Some(vec!["KNOWS".to_string(), "WORKS_WITH".to_string()]),
            "rel_filter must arrive at the adapter unchanged"
        );
    }

    #[test]
    fn explore_capped_defaults_direction_to_out_when_omitted() {
        // ADR-217 backward-compat pin: a request omitting `direction`
        // plumbs ExploreDirection::Out — byte-identical to the
        // v1.0-alpha walk.
        let e = RelFilterRecordingExplorer {
            bound_tenant: TenantId::new(1),
            last_filter: std::sync::Mutex::new(None),
            last_direction: std::sync::Mutex::new(None),
        };
        let req = ExploreRequest {
            tenant_id: 1,
            seed: 1,
            max_depth: Some(1),
            max_results: None,
            rel_types: None,
            direction: None,
            format: None,
            principal: None,
        };
        let token = CancellationToken::new();
        explore_capped(&e, TenantId::new(1), SessionScope::Power, &token, &req).expect("ok");
        assert_eq!(
            *e.last_direction.lock().unwrap(),
            Some(ExploreDirection::Out),
            "omitted direction must default to Out (no behavior change)"
        );
    }

    #[test]
    fn explore_capped_plumbs_explicit_direction_both() {
        // ADR-217: an explicit `direction:both` reaches the explorer
        // unchanged (the demo's opt-in to bidirectional traversal).
        let e = RelFilterRecordingExplorer {
            bound_tenant: TenantId::new(1),
            last_filter: std::sync::Mutex::new(None),
            last_direction: std::sync::Mutex::new(None),
        };
        let req = ExploreRequest {
            tenant_id: 1,
            seed: 1,
            max_depth: Some(1),
            max_results: None,
            rel_types: None,
            direction: Some(ExploreDirection::Both),
            format: None,
            principal: None,
        };
        let token = CancellationToken::new();
        explore_capped(&e, TenantId::new(1), SessionScope::Power, &token, &req).expect("ok");
        assert_eq!(
            *e.last_direction.lock().unwrap(),
            Some(ExploreDirection::Both),
        );
    }

    #[test]
    fn explore_request_direction_round_trips_on_the_wire() {
        // Wire pin: `direction` deserializes from the lowercase token, and
        // an omitted field is None (→ Out at explore_capped).
        let v = serde_json::json!({ "tenant_id": 1, "seed": 1, "direction": "both" });
        let req: ExploreRequest = serde_json::from_value(v).expect("parse");
        assert_eq!(req.direction, Some(ExploreDirection::Both));
        let v2 = serde_json::json!({ "tenant_id": 1, "seed": 1 });
        let req2: ExploreRequest = serde_json::from_value(v2).expect("parse");
        assert_eq!(req2.direction, None, "omitted → None");
        // Default of the enum itself is Out.
        assert_eq!(ExploreDirection::default(), ExploreDirection::Out);
    }

    #[test]
    fn explore_tool_returns_only_seed_at_depth_zero() {
        // PR #292 review NIT-2 — pin the depth=0 boundary. A zero-hop
        // request returns only the seed (no edges traversed). The
        // fixture's `nodes.retain(|n| n.depth <= max_depth)` handles
        // this; assert that only the depth-0 node (Alice) appears.
        let e = StubExplorer::new(TenantId::new(1)).with_seed(neighborhood_fixture());
        let req = ExploreRequest {
            tenant_id: 1,
            seed: 1,
            max_depth: Some(0),
            max_results: None,
            rel_types: None,
            direction: None,
            format: Some(ResponseFormat::Json),
            principal: None,
        };
        let token = CancellationToken::new();
        let resp =
            explore_tool(&e, TenantId::new(1), SessionScope::Power, &token, req).expect("ok");
        let body = resp["body"].as_str().unwrap();
        assert!(body.contains("Alice"), "seed (depth=0) present");
        assert!(!body.contains("Bob"), "depth=1 node excluded at depth=0");
        assert!(!body.contains("Carol"), "depth=2 node excluded at depth=0");
        // The wire shape echoes the requested max_depth (per LOW-2
        // identity contract).
        assert!(
            body.contains("\"max_depth\":0"),
            "max_depth=0 echoed; body={body}"
        );
    }
}
