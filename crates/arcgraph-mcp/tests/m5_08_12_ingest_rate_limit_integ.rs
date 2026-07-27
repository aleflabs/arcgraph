//! W14γ M5-08 + M5-12 integration tests.
//!
//! End-to-end coverage for the new write-side Tier-1 tool
//! (`graph.ingest`) and the per-tenant token-bucket rate-limit gate
//! that sits in front of every dispatcher request.
//!
//! Per the spawn prompt's §Tests acceptance:
//! 1. End-to-end ingest then graph.inspect roundtrip (reads-after-
//!    write contract).
//! 2. Rate-limit fires at threshold (write-class drains; the next
//!    request is rejected as -32007 with retry-after).
//! 3. Cross-tenant rate-limit isolation (drained tenant 7 doesn't
//!    affect tenant 8 sharing the same RateLimiter instance).
//! 4. Ingest + rate-limit composition: write is rate-limited but
//!    reads still serve.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::Mutex;

use arcgraph_core::TenantId;
use arcgraph_mcp::tools::explore::{Neighborhood, NeighborhoodExplorer};
use arcgraph_mcp::tools::ingest::{
    IngestBatch, IngestProvider, IngestRecordOutcome, IngestSummary,
};
use arcgraph_mcp::tools::inspect::{NeighborDirection, NeighborInfo, NodeInspection};
use arcgraph_mcp::tools::schema::{
    GraphSchema, IndexDescriptor, IndexKind, LabelInfo, RelTypeInfo,
};
use arcgraph_mcp::tools::search::{AvailableSubstrates, HybridSearcher, SearchHit};
use arcgraph_mcp::{
    Dispatcher, JsonRpcRequest, MCPError, NodeInspector, OpClass, RateLimiter, SchemaProvider,
};
use arcgraph_query::CancellationToken;
use serde_json::{Value, json};

// ─────────────────────────────────────────────────────────────────────
// Stubs — a `ProviderBundle` lets the ingest stub stash committed
// records so the inspect stub observes them on subsequent calls
// (the reads-after-write proof).
// ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Default, Clone)]
struct CommittedNode {
    internal_id: u64,
    label: String,
    properties: BTreeMap<String, Value>,
}

#[derive(Debug, Default)]
struct InMemoryStore {
    /// tenant -> nodes
    nodes: Mutex<std::collections::HashMap<u64, Vec<CommittedNode>>>,
    /// tenant -> last-issued LSN
    lsn: Mutex<u64>,
    /// Per-call observation LSN log. Every inspect / schema call
    /// snapshots the CURRENT lsn into this log; the snapshot-LSN
    /// oracle reads from here to prove the consumer (read tool) saw
    /// the producer's (ingest) `commit_lsn` per amendment-03 §TIER-1
    /// GAP E rule 1 + LSN monotonicity.
    observed_snapshot_lsns: Mutex<Vec<u64>>,
}

impl InMemoryStore {
    /// Acquire a snapshot LSN — mirrors the production read-tool's
    /// "snapshot LSN acquired before first batch pull" contract per
    /// amendment-03 §TIER-1 GAP E rule 1. Returns the current LSN
    /// and pushes it to the observation log.
    fn acquire_snapshot_lsn(&self) -> u64 {
        let cur = *self.lsn.lock().unwrap();
        self.observed_snapshot_lsns.lock().unwrap().push(cur);
        cur
    }
}

#[derive(Debug, Clone)]
struct StoreSchema {
    tenant: TenantId,
    store: Arc<InMemoryStore>,
}
impl SchemaProvider for StoreSchema {
    fn schema(&self, tenant: TenantId) -> Result<GraphSchema, MCPError> {
        if tenant != self.tenant {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        // Acquire snapshot LSN before first batch pull (amendment-03
        // §TIER-1 GAP E rule 1) — observed by the snapshot-LSN
        // oracle test below.
        let _snapshot_lsn = self.store.acquire_snapshot_lsn();
        let nodes = self.store.nodes.lock().unwrap();
        let by_tenant = nodes.get(&tenant.raw()).cloned().unwrap_or_default();
        let count = by_tenant.len() as u64;
        Ok(GraphSchema {
            tenant_id: tenant.raw(),
            labels: if count == 0 {
                vec![]
            } else {
                vec![LabelInfo {
                    name: "Person".into(),
                    cardinality: Some(count),
                    properties: vec![],
                }]
            },
            rel_types: vec![RelTypeInfo {
                name: "KNOWS".into(),
                cardinality: None,
            }],
            indexes: vec![IndexDescriptor {
                kind: IndexKind::Bm25,
                available: true,
            }],
            total_node_count: Some(count),
            total_rel_count: Some(0),
        })
    }
}

#[derive(Debug, Clone)]
struct StoreInspect {
    tenant: TenantId,
    store: Arc<InMemoryStore>,
}
impl NodeInspector for StoreInspect {
    fn inspect(&self, tenant: TenantId, node_id: u64) -> Result<NodeInspection, MCPError> {
        if tenant != self.tenant {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        // Acquire snapshot LSN before first batch pull (amendment-03
        // §TIER-1 GAP E rule 1).
        let _snapshot_lsn = self.store.acquire_snapshot_lsn();
        let nodes = self.store.nodes.lock().unwrap();
        let by_tenant = nodes.get(&tenant.raw());
        let n = by_tenant
            .and_then(|v| v.iter().find(|n| n.internal_id == node_id))
            .ok_or_else(|| MCPError::QueryError(format!("node {node_id} not found")))?;
        Ok(NodeInspection {
            id: n.internal_id,
            label: Some(n.label.clone()),
            properties: n.properties.clone(),
            neighbors: vec![NeighborInfo {
                node_id: 0,
                label: None,
                rel_type: None,
                direction: NeighborDirection::Out,
            }]
            .into_iter()
            .filter(|_| false) // no neighbors in this stub
            .collect(),
        })
    }
}

#[derive(Debug, Clone)]
struct StoreIngest {
    tenant: TenantId,
    store: Arc<InMemoryStore>,
}
impl IngestProvider for StoreIngest {
    fn ingest(&self, tenant: TenantId, batch: IngestBatch) -> Result<IngestSummary, MCPError> {
        if tenant != self.tenant {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        let mut nodes = self.store.nodes.lock().unwrap();
        let mut lsn = self.store.lsn.lock().unwrap();
        let by_tenant = nodes.entry(tenant.raw()).or_default();
        let mut records = Vec::new();
        let mut inserted = 0u64;
        let mut commit_lsn: Option<u64> = None;
        for n in batch.nodes {
            let internal_id = 1000 + by_tenant.len() as u64;
            *lsn += 1;
            let new_lsn = *lsn;
            commit_lsn = Some(commit_lsn.map_or(new_lsn, |c| c.max(new_lsn)));
            by_tenant.push(CommittedNode {
                internal_id,
                label: n.label.clone(),
                properties: n.properties.clone(),
            });
            inserted += 1;
            records.push(IngestRecordOutcome::Inserted {
                internal_id,
                external_id: n.external_id,
            });
        }
        for r in batch.relationships {
            // No rel store in this stub; all rels succeed for the
            // test fixture.
            let internal_id = 2000 + records.len() as u64;
            *lsn += 1;
            let new_lsn = *lsn;
            commit_lsn = Some(commit_lsn.map_or(new_lsn, |c| c.max(new_lsn)));
            inserted += 1;
            records.push(IngestRecordOutcome::Inserted {
                internal_id,
                external_id: r.external_id,
            });
        }
        Ok(IngestSummary {
            records,
            inserted_count: inserted,
            failed_count: 0,
            commit_lsn,
            dropped_acl_grants: Vec::new(),
        })
    }
}

/// Stub neighborhood explorer — the M5-08+M5-12 integ tests don't
/// exercise explore directly; the impl exists to satisfy the
/// dispatcher's `NeighborhoodExplorer` generic added in W14β.
#[derive(Debug, Clone)]
struct StoreExplore {
    tenant: TenantId,
}
impl NeighborhoodExplorer for StoreExplore {
    fn explore(
        &self,
        tenant: TenantId,
        _seed: u64,
        _max_depth: u32,
        _rel_filter: Option<&[String]>,
        _direction: arcgraph_mcp::tools::explore::ExploreDirection,
        _cancel: &CancellationToken,
    ) -> Result<Neighborhood, MCPError> {
        if tenant != self.tenant {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        Err(MCPError::InternalError(
            "stub explore not exercised by W14γ ingest+rate-limit integ".into(),
        ))
    }
}

/// Stub hybrid searcher — same role as `StoreExplore` for the W14β
/// `HybridSearcher` generic.
#[derive(Debug, Clone)]
struct StoreSearch {
    tenant: TenantId,
}
impl HybridSearcher for StoreSearch {
    fn available_substrates(
        &self,
        tenant: TenantId,
        _cancel: &CancellationToken,
    ) -> Result<AvailableSubstrates, MCPError> {
        if tenant != self.tenant {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        Ok(AvailableSubstrates::none())
    }
    fn search(
        &self,
        tenant: TenantId,
        _query_text: &str,
        _query_vec: Option<&[f32]>,
        _k: u32,
        _cancel: &CancellationToken,
    ) -> Result<Vec<SearchHit>, MCPError> {
        if tenant != self.tenant {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        Err(MCPError::InternalError(
            "stub search not exercised by W14γ ingest+rate-limit integ".into(),
        ))
    }
}

/// Stub raw-query executor — W14γ ingest+rate-limit integ tests don't
/// exercise raw_query; the impl exists only to satisfy the W16ζ-merged
/// dispatcher's `RawQueryExecutor` generic.
struct StoreRawQuery {
    tenant: TenantId,
}
impl arcgraph_mcp::tools::raw_query::RawQueryExecutor for StoreRawQuery {
    fn execute(
        &self,
        tenant: TenantId,
        _query: &str,
        _max_rows: u32,
        _cancel: &CancellationToken,
    ) -> Result<arcgraph_mcp::tools::raw_query::RawQueryRows, MCPError> {
        if tenant != self.tenant {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        Err(MCPError::InternalError(
            "stub raw_query not exercised by W14γ ingest+rate-limit integ".into(),
        ))
    }
}

fn dispatcher_for(
    tenant: u64,
    store: Arc<InMemoryStore>,
    limiter: Option<RateLimiter>,
) -> Dispatcher<StoreSchema, StoreInspect, StoreExplore, StoreSearch, StoreIngest, StoreRawQuery> {
    let t = TenantId::new(tenant);
    let s = StoreSchema {
        tenant: t,
        store: Arc::clone(&store),
    };
    let i = StoreInspect {
        tenant: t,
        store: Arc::clone(&store),
    };
    let e = StoreExplore { tenant: t };
    let h = StoreSearch { tenant: t };
    let g = StoreIngest {
        tenant: t,
        store: Arc::clone(&store),
    };
    let r = StoreRawQuery { tenant: t };
    match limiter {
        Some(l) => Dispatcher::with_rate_limiter(
            t,
            Arc::new(s),
            Arc::new(i),
            Arc::new(e),
            Arc::new(h),
            Arc::new(g),
            Arc::new(r),
            l,
        ),
        None => Dispatcher::new(
            t,
            Arc::new(s),
            Arc::new(i),
            Arc::new(e),
            Arc::new(h),
            Arc::new(g),
            Arc::new(r),
        ),
    }
}

fn ingest_request(tenant: u64, label: &str, ext: &str) -> JsonRpcRequest {
    JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(1)),
        method: "graph.ingest".into(),
        params: json!({
            "tenant_id": tenant,
            "nodes": [{
                "label": label,
                "external_id": ext,
            }]
        }),
    }
}

fn schema_request(tenant: u64) -> JsonRpcRequest {
    JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(2)),
        method: "graph.schema".into(),
        params: json!({"tenant_id": tenant}),
    }
}

fn inspect_request(tenant: u64, node_id: u64) -> JsonRpcRequest {
    JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(3)),
        method: "graph.inspect".into(),
        params: json!({"tenant_id": tenant, "node_id": node_id}),
    }
}

// ─────────────────────────────────────────────────────────────────────
// 1. End-to-end ingest then inspect — reads-after-write proof
// ─────────────────────────────────────────────────────────────────────

#[test]
fn integ_ingest_then_inspect_roundtrip_observes_committed_node() {
    // Producer → consumer transit pin per
    // feedback_anchor_to_consumer_transit_pinning.md:
    //   PRODUCER: graph.ingest's IngestSummary.commit_lsn
    //   CONSUMER: graph.inspect's snapshot LSN acquired before first
    //             batch pull (per amendment-03 §TIER-1 GAP E rule 1)
    // Contract: consumer's snapshot LSN >= producer's commit_lsn
    // (LSN monotonicity).
    //
    // We exercise the transit pin by:
    //   (a) ingesting through dispatcher A;
    //   (b) extracting commit_lsn from the response body;
    //   (c) inspecting through dispatcher B (separate context) —
    //       proves the LSN flows across MCP sessions, not just
    //       within one dispatcher instance;
    //   (d) asserting the consumer's snapshot LSN >= producer's
    //       commit_lsn.
    let store = Arc::new(InMemoryStore::default());
    let d_ingest = dispatcher_for(7, Arc::clone(&store), None);

    let ingest_resp = d_ingest
        .dispatch(ingest_request(7, "Person", "k1"))
        .expect("resp");
    assert!(ingest_resp.get("result").is_some(), "ingest must succeed");

    // Pull the internal_id + commit_lsn out of the response body.
    let body_str = ingest_resp["result"]["body"].as_str().unwrap();
    let body_json: Value = serde_json::from_str(body_str).unwrap();
    let internal_id = body_json["records"][0]["internal_id"]
        .as_u64()
        .expect("internal_id in inserted record");
    let producer_commit_lsn = body_json["commit_lsn"]
        .as_u64()
        .expect("commit_lsn must be Some(L) after successful ingest");
    assert!(
        producer_commit_lsn >= 1,
        "commit_lsn must advance from 0 baseline after first commit; got {producer_commit_lsn}"
    );
    // Sanity: producer's reported commit_lsn matches store's lsn.
    assert_eq!(
        producer_commit_lsn,
        *store.lsn.lock().unwrap(),
        "producer's commit_lsn must equal store's LSN at commit"
    );

    // (c) Inspect via a SEPARATE dispatcher context — proves the
    // reads-after-write contract holds across MCP sessions, not just
    // within one cached dispatcher.
    let observations_before = store.observed_snapshot_lsns.lock().unwrap().len();
    let d_inspect = dispatcher_for(7, Arc::clone(&store), None);
    let inspect_resp = d_inspect
        .dispatch(inspect_request(7, internal_id))
        .expect("inspect resp");
    let body = inspect_resp["result"]["body"].as_str().unwrap();
    assert!(body.contains("Person"), "inspect body must include label");
    assert!(
        body.contains(&internal_id.to_string()),
        "inspect body must include id"
    );

    // (d) Transit pin: the consumer's snapshot LSN >= producer's
    // commit_lsn. The fresh d_inspect dispatcher's first observation
    // is at index `observations_before`.
    let observed = store.observed_snapshot_lsns.lock().unwrap().clone();
    assert!(
        observed.len() > observations_before,
        "inspect call must record a snapshot LSN observation"
    );
    let consumer_snapshot_lsn = observed[observations_before];
    assert!(
        consumer_snapshot_lsn >= producer_commit_lsn,
        "amendment-03 §TIER-1 GAP E rule 1 + LSN monotonicity: \
         consumer snapshot LSN ({consumer_snapshot_lsn}) must be \
         >= producer commit_lsn ({producer_commit_lsn})"
    );

    // graph.schema should now report 1 Person AND also acquire a
    // snapshot LSN >= the producer's commit_lsn.
    let schema_resp = d_inspect.dispatch(schema_request(7)).expect("schema resp");
    let sb = schema_resp["result"]["body"].as_str().unwrap();
    assert!(
        sb.contains("Person"),
        "schema body must reflect committed node"
    );
    let observed_after_schema = store.observed_snapshot_lsns.lock().unwrap();
    let schema_snapshot_lsn = *observed_after_schema
        .last()
        .expect("schema must record a snapshot LSN observation");
    assert!(
        schema_snapshot_lsn >= producer_commit_lsn,
        "schema's snapshot LSN ({schema_snapshot_lsn}) must also be >= \
         producer commit_lsn ({producer_commit_lsn})"
    );
}

// ─────────────────────────────────────────────────────────────────────
// 2. Rate-limit fires at threshold
// ─────────────────────────────────────────────────────────────────────

#[test]
fn integ_rate_limit_fires_at_write_threshold() {
    // Tighten the write-class to 2 tokens / 0.0 refill (so the
    // bucket can't refill during the test). 2 inserts succeed; the
    // 3rd hits -32007.
    let store = Arc::new(InMemoryStore::default());
    let limiter = RateLimiter::new();
    limiter.set_per_tenant(TenantId::new(7), OpClass::Write, 2, 0.0);
    let d = dispatcher_for(7, Arc::clone(&store), Some(limiter));

    let r1 = d.dispatch(ingest_request(7, "P", "k1")).expect("resp");
    assert!(r1.get("result").is_some(), "1st must succeed");
    let r2 = d.dispatch(ingest_request(7, "P", "k2")).expect("resp");
    assert!(r2.get("result").is_some(), "2nd must succeed");
    let r3 = d.dispatch(ingest_request(7, "P", "k3")).expect("resp");
    assert_eq!(r3["error"]["code"], -32007, "3rd must hit rate-limit");
    assert!(
        r3["error"]["data"].get("retry_after_ms").is_some(),
        "retry-after-ms data populated"
    );
}

// ─────────────────────────────────────────────────────────────────────
// 3. Cross-tenant rate-limit isolation at the integration boundary
// ─────────────────────────────────────────────────────────────────────

#[test]
fn integ_rate_limit_isolation_drained_tenant_does_not_affect_other() {
    // Both dispatchers share the same RateLimiter; tenant 7's write
    // bucket drains, but tenant 8's bucket is untouched.
    let store = Arc::new(InMemoryStore::default());
    let limiter = RateLimiter::new();
    // Tenant 7 capped tight; tenant 8 uses defaults (10 write/s).
    limiter.set_per_tenant(TenantId::new(7), OpClass::Write, 1, 0.0);

    let d7 = dispatcher_for(7, Arc::clone(&store), Some(limiter.clone()));
    let d8 = dispatcher_for(8, Arc::clone(&store), Some(limiter));

    // Drain tenant 7.
    let _ = d7.dispatch(ingest_request(7, "P", "k7-1")).unwrap();
    let r7_2 = d7.dispatch(ingest_request(7, "P", "k7-2")).unwrap();
    assert_eq!(r7_2["error"]["code"], -32007);

    // Tenant 8 unaffected — at default 10 tokens, it serves 5
    // requests easily.
    for i in 0..5 {
        let r = d8
            .dispatch(ingest_request(8, "P", &format!("k8-{i}")))
            .unwrap();
        assert!(r.get("result").is_some(), "tenant 8 #{i} must succeed");
    }
}

// ─────────────────────────────────────────────────────────────────────
// 4. Ingest + rate-limit composition — writes throttled, reads not
// ─────────────────────────────────────────────────────────────────────

#[test]
fn integ_writes_throttled_reads_continue_to_serve() {
    // Drain the write bucket; reads still serve from the same
    // dispatcher (the read bucket is independent per the M5-12
    // op-class isolation invariant).
    let store = Arc::new(InMemoryStore::default());
    let limiter = RateLimiter::new();
    limiter.set_per_tenant(TenantId::new(7), OpClass::Write, 1, 0.0);
    let d = dispatcher_for(7, Arc::clone(&store), Some(limiter));

    // Write the seed record while we still have a write token.
    let r1 = d.dispatch(ingest_request(7, "Person", "seed")).unwrap();
    assert!(r1.get("result").is_some());

    // Drain the write bucket — second ingest fails.
    let r2 = d.dispatch(ingest_request(7, "Person", "k2")).unwrap();
    assert_eq!(r2["error"]["code"], -32007);

    // Reads continue serving — schema + inspect both work even
    // though writes are throttled.
    let s = d.dispatch(schema_request(7)).unwrap();
    assert!(s.get("result").is_some(), "schema reads must serve");
    let body_str = r1["result"]["body"].as_str().unwrap();
    let body_json: Value = serde_json::from_str(body_str).unwrap();
    let id = body_json["records"][0]["internal_id"].as_u64().unwrap();
    let i = d.dispatch(inspect_request(7, id)).unwrap();
    assert!(i.get("result").is_some(), "inspect reads must serve");
}
