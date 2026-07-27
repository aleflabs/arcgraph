//! W14β M5-06 + M5-07 integration tests — end-to-end `graph.explore`
//! and `graph.search` over the JSON-RPC dispatcher surface.
//!
//! Acceptance per the W14β spawn prompt:
//!   1. End-to-end explore on a Person fixture (multi-hop).
//!   2. End-to-end search on a "community-detected" fixture (label
//!      filter + ranking).
//!   3. Cross-tool tenant isolation (the dispatcher MUST reject
//!      cross-tenant explores AND cross-tenant searches identically).
//!   4. Cancel-during-search (a token tripped between
//!      `handle_raw_envelope` invocations on a long-running search
//!      surfaces -32001 in the response envelope).
//!
//! All four tests drive the dispatcher through [`handle_raw_envelope`]
//! — the same entry point [`arcgraph_mcp::serve_stdio`] uses — so the
//! integration coverage matches the production transport's wire shape.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use arcgraph_core::TenantId;
use arcgraph_mcp::tools::explore::{Neighborhood, NeighborhoodEdge, NeighborhoodNode};
use arcgraph_mcp::tools::ingest::{IngestBatch, IngestProvider, IngestSummary};
use arcgraph_mcp::tools::inspect::{NeighborDirection, NodeInspection, NodeInspector};
use arcgraph_mcp::tools::schema::{
    GraphSchema, IndexDescriptor, IndexKind, LabelInfo, RelTypeInfo, SchemaProvider,
};
use arcgraph_mcp::tools::search::{AvailableSubstrates, SearchHit};
use arcgraph_mcp::{
    Dispatcher, HybridSearcher, MCPError, NeighborhoodExplorer, handle_raw_envelope,
};
use arcgraph_query::CancellationToken;
use serde_json::{Value, json};

// ─────────────────────────────────────────────────────────────────────
// Fixture: PersonTenant — three Person nodes (Alice → Bob → Carol) +
// two Document nodes attached as KNOWS-of-INTEREST anchors.
// ─────────────────────────────────────────────────────────────────────

struct PersonExplorer {
    tenant: TenantId,
}

impl NeighborhoodExplorer for PersonExplorer {
    fn explore(
        &self,
        tenant: TenantId,
        seed: u64,
        max_depth: u32,
        rel_filter: Option<&[String]>,
        _direction: arcgraph_mcp::tools::explore::ExploreDirection,
        cancel: &CancellationToken,
    ) -> Result<Neighborhood, MCPError> {
        if cancel.is_cancelled() {
            return Err(MCPError::Cancelled);
        }
        if tenant != self.tenant {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        if seed != 1 {
            return Err(MCPError::QueryError(format!("seed {seed} not found")));
        }
        let mut alice: BTreeMap<String, Value> = BTreeMap::new();
        alice.insert("name".into(), json!("Alice"));
        let mut bob: BTreeMap<String, Value> = BTreeMap::new();
        bob.insert("name".into(), json!("Bob"));
        let mut carol: BTreeMap<String, Value> = BTreeMap::new();
        carol.insert("name".into(), json!("Carol"));
        let mut nodes = vec![
            NeighborhoodNode {
                id: 1,
                label: Some("Person".into()),
                depth: 0,
                properties: alice,
            },
            NeighborhoodNode {
                id: 2,
                label: Some("Person".into()),
                depth: 1,
                properties: bob,
            },
            NeighborhoodNode {
                id: 3,
                label: Some("Person".into()),
                depth: 2,
                properties: carol,
            },
        ];
        let mut edges = vec![
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
        ];
        nodes.retain(|n| n.depth <= max_depth);
        let allowed: std::collections::HashSet<u64> = nodes.iter().map(|n| n.id).collect();
        edges.retain(|e| allowed.contains(&e.from) && allowed.contains(&e.to));
        if let Some(allow) = rel_filter {
            if !allow.is_empty() {
                edges.retain(|e| match &e.rel_type {
                    Some(rt) => allow.contains(rt),
                    None => false,
                });
            }
        }
        Ok(Neighborhood {
            seed,
            max_depth,
            truncated: false,
            nodes,
            edges,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────
// Fixture: CommunitySearcher — a search that returns Document hits
// pre-clustered by a "community-detected" label set. The community
// substrate itself is not bound at v1.0-alpha (it's a forward-pin to
// M4-62b LogicalCommunityLookup per ADR-038 amendment-02 §M4.c
// hybrid-retrieval lowering; community detection availability per
// ADR-036 §D-6); the fixture stands in for the production wiring's
// eventual output shape.
// ─────────────────────────────────────────────────────────────────────

struct CommunitySearcher {
    tenant: TenantId,
    /// Trip this externally to fire a Cancelled response BEFORE the
    /// search body runs. Models the post-token-trip path used by the
    /// cancel-during-search test.
    pre_trip: Arc<AtomicBool>,
}

impl HybridSearcher for CommunitySearcher {
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
        Ok(AvailableSubstrates {
            vector: true,
            bm25: true,
        })
    }

    fn search(
        &self,
        tenant: TenantId,
        query_text: &str,
        _query_vec: Option<&[f32]>,
        k: u32,
        cancel: &CancellationToken,
    ) -> Result<Vec<SearchHit>, MCPError> {
        if self.pre_trip.load(Ordering::SeqCst) {
            cancel.cancel();
        }
        if cancel.is_cancelled() {
            return Err(MCPError::Cancelled);
        }
        if tenant != self.tenant {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        // "Community-detected" fixture: two Document clusters with
        // RRF-fused scores; the query text picks which cluster wins.
        let mut hits = if query_text.contains("incident") {
            vec![
                SearchHit {
                    node_id: 101,
                    label: Some("Document".into()),
                    score: 0.95,
                },
                SearchHit {
                    node_id: 102,
                    label: Some("Document".into()),
                    score: 0.88,
                },
                SearchHit {
                    node_id: 201,
                    label: Some("Document".into()),
                    score: 0.42,
                },
            ]
        } else {
            vec![
                SearchHit {
                    node_id: 201,
                    label: Some("Document".into()),
                    score: 0.92,
                },
                SearchHit {
                    node_id: 202,
                    label: Some("Document".into()),
                    score: 0.85,
                },
            ]
        };
        hits.truncate(k as usize);
        Ok(hits)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Fixture: minimal stubs for the unrelated schema + inspect adapters
// the dispatcher requires.
// ─────────────────────────────────────────────────────────────────────

struct DummySchema(TenantId);
impl SchemaProvider for DummySchema {
    fn schema(&self, tenant: TenantId) -> Result<GraphSchema, MCPError> {
        if tenant != self.0 {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        Ok(GraphSchema {
            tenant_id: tenant.raw(),
            labels: vec![LabelInfo {
                name: "Person".into(),
                cardinality: None,
                properties: vec![],
            }],
            rel_types: vec![RelTypeInfo {
                name: "KNOWS".into(),
                cardinality: None,
            }],
            indexes: vec![IndexDescriptor {
                kind: IndexKind::Vector,
                available: true,
            }],
            total_node_count: None,
            total_rel_count: None,
        })
    }
}

struct DummyInspect(TenantId);
impl NodeInspector for DummyInspect {
    fn inspect(&self, tenant: TenantId, node_id: u64) -> Result<NodeInspection, MCPError> {
        if tenant != self.0 {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        Ok(NodeInspection {
            id: node_id,
            label: Some("Person".into()),
            properties: BTreeMap::new(),
            neighbors: vec![],
        })
    }
}

/// Stub ingest provider — the M5-06/M5-07 integ tests don't exercise
/// ingest; the impl exists only to satisfy the W14γ-merged
/// dispatcher's `IngestProvider` generic.
struct DummyIngest(TenantId);
impl IngestProvider for DummyIngest {
    fn ingest(&self, tenant: TenantId, _batch: IngestBatch) -> Result<IngestSummary, MCPError> {
        if tenant != self.0 {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        Err(MCPError::InternalError(
            "stub ingest not exercised by W14β M5-06/M5-07 integ".into(),
        ))
    }
}

/// Stub raw-query executor — the M5-06/M5-07 integ tests don't
/// exercise raw_query; the impl exists only to satisfy the W16ζ-merged
/// dispatcher's `RawQueryExecutor` generic.
struct DummyRawQuery(TenantId);
impl arcgraph_mcp::tools::raw_query::RawQueryExecutor for DummyRawQuery {
    fn execute(
        &self,
        tenant: TenantId,
        _query: &str,
        _max_rows: u32,
        _cancel: &CancellationToken,
    ) -> Result<arcgraph_mcp::tools::raw_query::RawQueryRows, MCPError> {
        if tenant != self.0 {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        Err(MCPError::InternalError(
            "stub raw_query not exercised by W14β M5-06/M5-07 integ".into(),
        ))
    }
}

fn build_dispatcher(
    tenant: u64,
    pre_trip: Arc<AtomicBool>,
) -> Dispatcher<
    DummySchema,
    DummyInspect,
    PersonExplorer,
    CommunitySearcher,
    DummyIngest,
    DummyRawQuery,
> {
    let t = TenantId::new(tenant);
    Dispatcher::new(
        t,
        Arc::new(DummySchema(t)),
        Arc::new(DummyInspect(t)),
        Arc::new(PersonExplorer { tenant: t }),
        Arc::new(CommunitySearcher {
            tenant: t,
            pre_trip,
        }),
        Arc::new(DummyIngest(t)),
        Arc::new(DummyRawQuery(t)),
    )
}

// ─────────────────────────────────────────────────────────────────────
// Test 1 — end-to-end graph.explore over the JSON-RPC envelope.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn integ_explore_returns_seed_plus_two_hops_on_person_fixture() {
    let d = build_dispatcher(7, Arc::new(AtomicBool::new(false)));
    let env = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "graph.explore",
        "params": {"tenant_id": 7, "seed": 1, "max_depth": 2, "format": "json"}
    });
    let resp = handle_raw_envelope(&d, env).expect("response present");
    assert_eq!(resp["id"], 1);
    let body = resp["result"]["body"]
        .as_str()
        .expect("explore result.body is a string");
    assert!(body.contains("Alice"), "seed visible in body");
    assert!(body.contains("Bob"), "depth-1 neighbor visible");
    assert!(body.contains("Carol"), "depth-2 neighbor visible");
    // The wire shape echoes the requested max_depth.
    assert!(
        body.contains("\"max_depth\":2"),
        "max_depth echoed: body={body}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// Test 2 — end-to-end graph.search over the community-detected
// fixture. The Document cluster ranking + label_filter pass through
// the JSON-RPC envelope intact.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn integ_search_returns_ranked_documents_on_community_fixture() {
    let d = build_dispatcher(7, Arc::new(AtomicBool::new(false)));
    let env = json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "graph.search",
        "params": {
            "tenant_id": 7,
            "query": "incident response",
            "k": 3,
            "label_filter": ["Document"],
            "format": "json"
        }
    });
    let resp = handle_raw_envelope(&d, env).expect("response present");
    assert_eq!(resp["id"], 2);
    let body = resp["result"]["body"]
        .as_str()
        .expect("search result.body is a string");
    // Community-1 cluster ranked top (101, 102), Community-2 (201)
    // brings up the tail at 0.42; the envelope must echo k=3.
    assert!(body.contains("\"k\":3"), "k echoed: body={body}");
    // O-K (W28-S3): assert the RANKED ORDER, not just presence. The
    // fixture ranks Community-1 (101 @ 0.95, 102 @ 0.88) above
    // Community-2 (201 @ 0.42); the JSON-serialized hit array preserves
    // rank order, so the byte offsets of each `node_id` marker must be
    // strictly increasing 101 → 102 → 201. The prior three `contains`
    // presence checks passed for ANY permutation (a searcher that
    // returned the hits reversed/unsorted would not have been caught),
    // and omitted the tail hit (201) entirely.
    let p101 = body
        .find("\"node_id\":101")
        .unwrap_or_else(|| panic!("node 101 (rank 0) missing: body={body}"));
    let p102 = body
        .find("\"node_id\":102")
        .unwrap_or_else(|| panic!("node 102 (rank 1) missing: body={body}"));
    let p201 = body
        .find("\"node_id\":201")
        .unwrap_or_else(|| panic!("node 201 (rank 2 / tail) missing: body={body}"));
    assert!(
        p101 < p102 && p102 < p201,
        "ranked order must be 101 (0.95) → 102 (0.88) → 201 (0.42); got \
         offsets 101@{p101} 102@{p102} 201@{p201}; body={body}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// Test 3 — cross-tool tenant isolation. The dispatcher binds to
// tenant 7; both graph.explore AND graph.search MUST reject a request
// asking for tenant 8 with -32002 BEFORE any adapter call. We test
// both tools on the same dispatcher to confirm the guard is uniform.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn integ_cross_tool_tenant_isolation_rejects_both_explore_and_search() {
    let d = build_dispatcher(7, Arc::new(AtomicBool::new(false)));
    let explore_env = json!({
        "jsonrpc": "2.0",
        "id": 10,
        "method": "graph.explore",
        "params": {"tenant_id": 8, "seed": 1}
    });
    let search_env = json!({
        "jsonrpc": "2.0",
        "id": 11,
        "method": "graph.search",
        "params": {"tenant_id": 8, "query": "x"}
    });
    let r1 = handle_raw_envelope(&d, explore_env).expect("response");
    let r2 = handle_raw_envelope(&d, search_env).expect("response");
    assert_eq!(
        r1["error"]["code"], -32002,
        "explore cross-tenant -> Unauthorized"
    );
    assert_eq!(
        r2["error"]["code"], -32002,
        "search cross-tenant -> Unauthorized"
    );
    assert_eq!(r1["id"], 10);
    assert_eq!(r2["id"], 11);
    // Both errors carry the same message slug so a router can
    // template on the code uniformly.
    assert_eq!(r1["error"]["message"], "unauthorized");
    assert_eq!(r2["error"]["message"], "unauthorized");
}

// ─────────────────────────────────────────────────────────────────────
// Test 4 — search surfaces Cancelled when the per-request token is
// pre-tripped by the adapter body BEFORE the search yields a hit.
// This models the post-token-trip path: the dispatcher's per-request
// `CancellationToken` is fresh-per-request at v1.0-alpha (no external
// SIGTERM-style trip; M5-02 streamable-HTTP forward-binds a session-
// scoped token + `$/cancelRequest` end-to-end), so the integration
// test self-trips the token via `pre_trip: AtomicBool` to exercise the
// dispatch → adapter → error-envelope path. The response envelope
// MUST carry -32001 (Cancelled) and propagate the original request id.
// PR #292 review NIT-1: rename clarified — the test does NOT model an
// external mid-flight cancel; M5-02 will replace it with one that does.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn integ_search_surfaces_cancelled_when_token_pretripped() {
    let pre_trip = Arc::new(AtomicBool::new(true));
    let d = build_dispatcher(7, pre_trip);
    let env = json!({
        "jsonrpc": "2.0",
        "id": 42,
        "method": "graph.search",
        "params": {"tenant_id": 7, "query": "incident", "k": 3}
    });
    let resp = handle_raw_envelope(&d, env).expect("response");
    assert_eq!(resp["id"], 42);
    assert_eq!(
        resp["error"]["code"], -32001,
        "cancelled response code: {resp}"
    );
    assert_eq!(resp["error"]["message"], "request cancelled");
}
