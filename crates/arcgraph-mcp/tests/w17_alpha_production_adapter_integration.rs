//! W17α M4-08+ — production storage-adapter integration tests.
//!
//! Pins the end-to-end MCP wire shape against the workspace's
//! production storage adapters (`StorageSchemaProvider`,
//! `StorageNodeInspector`, `StorageNeighborhoodExplorer`,
//! `StorageHybridSearcher`, `StorageIngestProvider`,
//! `StorageRawQueryExecutor`) routed through the live
//! [`arcgraph_mcp::Dispatcher`].
//!
//! Each test:
//!
//! 1. Bootstraps a fresh per-process storage backend (catalog +
//!    buffer pool + txn manager + CRUD store + intern table).
//! 2. Wires the adapter bundle into a `Dispatcher` bound to
//!    `TenantId::DEFAULT`.
//! 3. Drives a JSON-RPC request through `dispatcher.dispatch(...)`.
//! 4. Asserts the response envelope shape pins the contract.
//!
//! The integration test does NOT exercise the subprocess binary
//! (that's `mcp_stdio_integ.rs`); it exercises the dispatcher tower
//! that the binary wraps, with REAL storage adapters substituted for
//! the previous stub adapters.

use std::sync::Arc;

use arcgraph_core::{PartitionId, TenantId};
use arcgraph_mcp::jsonrpc::JsonRpcRequest;
use arcgraph_mcp::storage::{
    StorageBackend, StorageHybridSearcher, StorageIngestProvider, StorageNeighborhoodExplorer,
    StorageNodeInspector, StorageRawQueryExecutor, StorageSchemaProvider,
};
use arcgraph_mcp::{Dispatcher, RateLimiter, SessionScope};
use arcgraph_storage::InternTable;
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::catalog::SystemCatalog;
use arcgraph_storage::crud::CrudStore;
use arcgraph_storage::io::InMemoryPageIo;
use arcgraph_storage::router::MultiTenantRouter;
use arcgraph_storage::transaction::TxnManager;
use serde_json::json;

type TestDispatcher = Dispatcher<
    StorageSchemaProvider,
    StorageNodeInspector,
    StorageNeighborhoodExplorer,
    StorageHybridSearcher,
    StorageIngestProvider,
    StorageRawQueryExecutor,
>;

fn fresh_backend() -> StorageBackend {
    let io = Arc::new(InMemoryPageIo::new());
    let pool = BufferPool::new(8, io);
    let mgr = Arc::new(TxnManager::new());
    let catalog = Arc::new(SystemCatalog::new());
    catalog.bootstrap(&pool, &mgr).expect("catalog bootstrap");
    let crud = Arc::new(CrudStore::new());
    let router = Arc::new(MultiTenantRouter::new(catalog, Arc::clone(&crud), None));
    let intern = Arc::new(InternTable::new());
    StorageBackend::new(router, mgr, intern)
}

fn fresh_dispatcher() -> TestDispatcher {
    let backend = fresh_backend();
    Dispatcher::with_session_scope_and_rate_limiter(
        TenantId::DEFAULT,
        SessionScope::Power,
        Arc::new(StorageSchemaProvider::new(backend.clone())),
        Arc::new(StorageNodeInspector::new(backend.clone())),
        Arc::new(StorageNeighborhoodExplorer::new(backend.clone())),
        Arc::new(StorageHybridSearcher::new(backend.clone())),
        Arc::new(StorageIngestProvider::new(backend.clone())),
        Arc::new(StorageRawQueryExecutor::new(backend)),
        RateLimiter::new(),
    )
}

#[test]
fn graph_schema_returns_empty_schema_for_fresh_tenant() {
    let d = fresh_dispatcher();
    let req = JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(1)),
        method: "graph.schema".into(),
        params: json!({ "tenant_id": 1, "format": "json" }),
    };
    let resp = d.dispatch(req).expect("dispatch returns envelope");
    let result = &resp["result"];
    let body = result["body"].as_str().expect("body");
    // Fresh tenant → empty schema.
    assert!(body.contains("\"labels\":[]"));
    assert!(body.contains("\"rel_types\":[]"));
}

#[test]
fn graph_ingest_then_graph_inspect_round_trips_a_node() {
    let d = fresh_dispatcher();

    // 1. Ingest a single Person node with external_id "alice".
    let ingest_req = JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(1)),
        method: "graph.ingest".into(),
        params: json!({
            "tenant_id": 1,
            "nodes": [
                {
                    "external_id": "alice",
                    "label": "Person",
                    "properties": { "name": "Alice", "team": "incident-response" }
                }
            ],
            "relationships": [],
            "format": "json"
        }),
    };
    let ingest_resp = d.dispatch(ingest_req).expect("ingest envelope");
    let result = &ingest_resp["result"];
    let body = result["body"].as_str().expect("body");
    let summary: serde_json::Value = serde_json::from_str(body).expect("parse body");
    assert_eq!(summary["inserted_count"], 1);
    assert_eq!(summary["failed_count"], 0);
    let internal_id = summary["records"][0]["internal_id"]
        .as_u64()
        .expect("internal_id");

    // 2. Inspect the returned internal id.
    let inspect_req = JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(2)),
        method: "graph.inspect".into(),
        params: json!({
            "tenant_id": 1,
            "node_id": internal_id,
            "format": "json"
        }),
    };
    let inspect_resp = d.dispatch(inspect_req).expect("inspect envelope");
    let body = inspect_resp["result"]["body"]
        .as_str()
        .expect("inspect body");
    // The body contains the Person label and the returned id.
    assert!(body.contains("\"label\":\"Person\""));
    assert!(body.contains(&format!("\"id\":{internal_id}")));

    // #894 (HIGH, data-loss): the ingested property bag MUST round-trip
    // through ingest → inspect. Pre-fix `graph.inspect` returned
    // `"properties":{}` even though the data was stored (raw_query could
    // filter on it) — a silent data-loss / hallucination-bait surface.
    // Strong oracle: exact equality with the ingested bag (also proves
    // the storage-internal `inline_u32a/b` slots do NOT leak as
    // user-facing property names — R1 review MED-6, PR #349).
    let inspection: serde_json::Value =
        serde_json::from_str(body).expect("parse inspect body as JSON");
    assert_eq!(
        inspection["properties"],
        json!({ "name": "Alice", "team": "incident-response" }),
        "inspect must hydrate the ingested property bag (#894); body={body}"
    );
}

#[test]
fn graph_explore_after_ingest_emits_full_neighborhood() {
    let d = fresh_dispatcher();
    // Ingest 3 Persons + 2 KNOWS edges (alice→bob, alice→carol).
    let ingest_req = JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(1)),
        method: "graph.ingest".into(),
        params: json!({
            "tenant_id": 1,
            "nodes": [
                { "external_id": "alice", "label": "Person", "properties": { "name": "Alice" } },
                { "external_id": "bob", "label": "Person", "properties": { "name": "Bob" } },
                { "external_id": "carol", "label": "Person", "properties": { "name": "Carol" } }
            ],
            "relationships": [
                {
                    "from_external_id": "alice",
                    "to_external_id": "bob",
                    "rel_type": "KNOWS",
                    "properties": {}
                },
                {
                    "from_external_id": "alice",
                    "to_external_id": "carol",
                    "rel_type": "KNOWS",
                    "properties": {}
                }
            ],
            "format": "json"
        }),
    };
    let ingest_resp = d.dispatch(ingest_req).expect("ingest");
    let body = ingest_resp["result"]["body"].as_str().expect("body");
    let summary: serde_json::Value = serde_json::from_str(body).expect("parse");
    assert_eq!(summary["inserted_count"], 5);

    // Find Alice's internal id.
    let alice_id = summary["records"]
        .as_array()
        .expect("records array")
        .iter()
        .find(|r| r["external_id"] == "alice")
        .expect("alice record")["internal_id"]
        .as_u64()
        .expect("alice id");
    let bob_id = summary["records"]
        .as_array()
        .expect("records array")
        .iter()
        .find(|r| r["external_id"] == "bob")
        .expect("bob record")["internal_id"]
        .as_u64()
        .expect("bob id");

    // Explore from Alice at depth 1 — must reach Bob + Carol.
    let explore_req = JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(2)),
        method: "graph.explore".into(),
        params: json!({
            "tenant_id": 1,
            "seed": alice_id,
            "max_depth": 1,
            "format": "json"
        }),
    };
    let explore_resp = d.dispatch(explore_req).expect("explore");
    let body = explore_resp["result"]["body"].as_str().expect("body");
    let n: serde_json::Value = serde_json::from_str(body).expect("parse");
    // Alice + Bob + Carol → 3 nodes.
    let nodes = n["nodes"].as_array().expect("nodes");
    assert_eq!(nodes.len(), 3, "alice + 2 neighbors");
    // 2 KNOWS edges.
    let edges = n["edges"].as_array().expect("edges");
    assert_eq!(edges.len(), 2);

    // #894 (HIGH, data-loss): the SEED node AND the NEIGHBOR nodes MUST
    // carry their hydrated property bags. Pre-fix every explore node came
    // back `"properties":{}` (silent data-loss). Strong oracle: locate
    // each node by id (order-independent) and assert exact-match props.
    let seed_node = nodes
        .iter()
        .find(|x| x["id"].as_u64() == Some(alice_id))
        .expect("seed node present in neighborhood");
    assert_eq!(seed_node["depth"], 0, "seed is depth 0");
    assert_eq!(
        seed_node["properties"],
        json!({ "name": "Alice" }),
        "explore must hydrate the SEED property bag (#894); body={body}"
    );
    let bob_node = nodes
        .iter()
        .find(|x| x["id"].as_u64() == Some(bob_id))
        .expect("bob neighbor present in neighborhood");
    assert_eq!(bob_node["depth"], 1, "bob is a depth-1 neighbor");
    assert_eq!(
        bob_node["properties"],
        json!({ "name": "Bob" }),
        "explore must hydrate NEIGHBOR property bags (#894); body={body}"
    );
}

#[test]
fn graph_search_reports_no_substrate_for_unwired_tenant() {
    let d = fresh_dispatcher();
    let req = JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(1)),
        method: "graph.search".into(),
        params: json!({
            "tenant_id": 1,
            "query": "alice",
            "k": 5,
            "format": "json"
        }),
    };
    let resp = d.dispatch(req).expect("envelope");
    // No substrate attached → -32004 IndexUnavailable.
    let err_code = resp["error"]["code"].as_i64().expect("code");
    assert_eq!(err_code, -32004);
}

#[test]
fn graph_raw_query_executes_label_free_match() {
    let d = fresh_dispatcher();
    // Ingest 2 nodes (no labels needed since we use a label-free
    // MATCH in the raw query).
    let ingest_req = JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(1)),
        method: "graph.ingest".into(),
        params: json!({
            "tenant_id": 1,
            "nodes": [
                { "external_id": "a", "label": "A", "properties": {} },
                { "external_id": "b", "label": "A", "properties": {} }
            ],
            "relationships": [],
            "format": "json"
        }),
    };
    let _ = d.dispatch(ingest_req).expect("ingest");

    // Now run a label-free MATCH via raw_query.
    let raw_req = JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(2)),
        method: "graph.raw_query".into(),
        params: json!({
            "tenant_id": 1,
            "query": "MATCH (n) RETURN n",
            "max_rows": 100,
            "format": "json"
        }),
    };
    let resp = d.dispatch(raw_req).expect("raw envelope");
    // No error envelope.
    assert!(resp["error"].is_null(), "raw_query failed: {resp:?}");
    let body = resp["result"]["body"].as_str().expect("body");
    let rows: serde_json::Value = serde_json::from_str(body).expect("parse");
    assert_eq!(rows["row_count"], 2, "expected 2 rows; body={body}");
}

#[test]
fn graph_raw_query_multi_pattern_join_executes_end_to_end() {
    // Pin: the W17α LogicalJoin executor is reachable via the MCP
    // raw_query path. We ingest 3 nodes + 1 edge then run a
    // multi-pattern MATCH that lowers to a Join.
    let d = fresh_dispatcher();
    let ingest_req = JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(1)),
        method: "graph.ingest".into(),
        params: json!({
            "tenant_id": 1,
            "nodes": [
                { "external_id": "a", "label": "A", "properties": {} },
                { "external_id": "b", "label": "A", "properties": {} }
            ],
            "relationships": [
                {
                    "from_external_id": "a",
                    "to_external_id": "b",
                    "rel_type": "REL",
                    "properties": {}
                }
            ],
            "format": "json"
        }),
    };
    let _ = d.dispatch(ingest_req).expect("ingest");

    // Multi-pattern MATCH — shared `a` binding → LogicalJoin.
    let raw_req = JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(2)),
        method: "graph.raw_query".into(),
        params: json!({
            "tenant_id": 1,
            "query": "MATCH (a), (a)-[r]->(b) RETURN a, b",
            "max_rows": 100,
            "format": "json"
        }),
    };
    let resp = d.dispatch(raw_req).expect("raw envelope");
    assert!(resp["error"].is_null(), "raw_query failed: {resp:?}");
    let body = resp["result"]["body"].as_str().expect("body");
    let rows: serde_json::Value = serde_json::from_str(body).expect("parse");
    assert_eq!(rows["row_count"], 1, "single (a,b) join row; body={body}");
}

/// R1 MED-2 (PR #349): pin the production raw-query path's
/// property-name behavior. The v1.0-α catalog seed is the in-memory
/// `StubCatalogProvider` (aliased as `InMemoryCatalogProvider` for
/// the production-named surface); property names are not enumerated
/// upfront, so a `n.name = 'X'` predicate exercises the binding
/// pass's dynamic-name fallback. The test pins the v1.0-α
/// behavior shape so a future v1.1 catalog impl can be detected as
/// a behavior change (the relaxed-oracle anti-pattern from
/// `feedback_review_oracle_relaxations.md` is what this test
/// defends against).
#[test]
fn graph_raw_query_property_name_filter_pins_v1_0_alpha_behavior() {
    let d = fresh_dispatcher();
    // Ingest 2 nodes, only one of which carries a `name` property
    // equal to "Alice". The wire-shape property bag is encoded via
    // `property_data_for_json_map` → `PropertyData::Blob` (JSON
    // bytes).
    //
    // **W27-α / ADR-152 closes the property-bag round-trip** —
    // previously the storage layer did NOT decode the blob at scan
    // time (issue #356, forward-pinned to v1.2). With ADR-152's
    // §D-3 wire-through, `CrudExecutorSubstrate::scan_nodes` now
    // decodes the persisted JSON blob back into the executor's
    // `NodeView.properties` BTreeMap so `WHERE n.name = 'Alice'`
    // evaluates correctly via the existing PropertyAccess +
    // BinaryOp::Eq machinery.
    //
    // This test now pins the **post-ADR-152 v1.0-α-permissive
    // behavior**: exactly the matching node (Alice) returns; the
    // non-matching node (Bob) is suppressed. A future regression
    // that drops the property bag again will flip the test red.
    let ingest_req = JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(1)),
        method: "graph.ingest".into(),
        params: json!({
            "tenant_id": 1,
            "nodes": [
                { "external_id": "alice", "label": "Person", "properties": { "name": "Alice" } },
                { "external_id": "bob",   "label": "Person", "properties": { "name": "Bob"   } }
            ],
            "relationships": [],
            "format": "json"
        }),
    };
    let _ = d.dispatch(ingest_req).expect("ingest");

    // Property-name filter — post-ADR-152 the executor's substrate-
    // backed property resolution decodes the JSON blob at scan time
    // (ADR-152 §D-3 `scan_nodes` + `record_property_bag`) so
    // `n.name = 'Alice'` matches the Alice row and the row count is 1.
    //
    // R1 LOW-1 (W27-α #492): this is now a SUPPORTED path, so the
    // oracle asserts success EXPLICITLY — it does NOT accept an error
    // envelope. The pre-ADR-152 error-envelope acceptance is dropped:
    // tolerating it would silently re-admit the dropped-property-bag
    // regression this test defends against (the relaxed-oracle
    // anti-pattern from `feedback_review_oracle_relaxations.md`).
    let raw_req = JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(2)),
        method: "graph.raw_query".into(),
        params: json!({
            "tenant_id": 1,
            "query": "MATCH (n) WHERE n.name = 'Alice' RETURN n",
            "max_rows": 100,
            "format": "json"
        }),
    };
    let resp = d.dispatch(raw_req).expect("envelope");
    // Post-ADR-152: the property-name filter is a supported path — it
    // MUST succeed (no error envelope).
    assert!(
        resp["error"].is_null(),
        "post-ADR-152 property-name filter is a supported path; must not error: {resp:?}"
    );
    let body = resp["result"]["body"].as_str().expect("body");
    let rows: serde_json::Value = serde_json::from_str(body).expect("parse");
    // The predicate matches EXACTLY the Alice row; Bob (name="Bob") is
    // suppressed. Asserting the exact count pins the property-only
    // match — a regression that drops the bag (→ 0 rows) or fails to
    // suppress Bob (→ 2 rows) flips the test red.
    assert_eq!(
        rows["row_count"], 1,
        "post-ADR-152 v1.0-α property-name filter returns exactly the matching node \
         (Alice present, Bob suppressed; closes issue #356 forward-pin); body={body}"
    );
}

#[test]
fn cross_tenant_request_rejects_with_unauthorized() {
    let d = fresh_dispatcher();
    let req = JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(1)),
        method: "graph.schema".into(),
        params: json!({ "tenant_id": 42, "format": "json" }),
    };
    let resp = d.dispatch(req).expect("envelope");
    let err_code = resp["error"]["code"].as_i64().expect("code");
    assert_eq!(err_code, -32002, "cross-tenant must surface Unauthorized");
}

#[test]
fn backend_routes_default_tenant_only_at_v1_0_alpha() {
    // Pin: the per-process backend bootstraps only the default
    // tenant. Routing an unknown tenant directly yields
    // SubstrateAccessError::TenantUnknown (translated to
    // MCPError::TenantUnknown at the adapter boundary).
    let backend = fresh_backend();
    let unknown = TenantId::new(9999);
    assert!(
        backend.router().route(unknown, PartitionId::ZERO).is_err(),
        "unknown tenant must not route"
    );
    assert!(
        backend
            .router()
            .route(TenantId::DEFAULT, PartitionId::ZERO)
            .is_ok(),
        "DEFAULT tenant must route"
    );
}
