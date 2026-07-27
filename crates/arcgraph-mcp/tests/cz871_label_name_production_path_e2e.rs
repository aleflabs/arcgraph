//! **#871** — production read-path proof: a Bolt/MCP client reads a
//! node's label NAME (and a rel's type NAME), resolved from the REAL
//! storage intern table, end-to-end through the `graph.raw_query`
//! surface the production stdio / HTTP / Bolt transports all wrap.
//!
//! This is the strong half of the fix: `CrudExecutorSubstrate::scan` /
//! `expand` reverse-resolve the storage-allocated `LabelId` / `TypeId`
//! back to the interned name via the intern table (the same mechanism
//! `inspect` / `explore` already use), so `labels(n)` / `type(r)` and the
//! node/rel wire serialization carry `"Account"` / `"KNOWS"` — never the
//! opaque `"LabelId(1)"` / `"TypeId(1)"` debug form CZ found over both
//! the JS `neo4j-driver` and the Python `neo4j` Bolt drivers.
//!
//! The `graph.raw_query` JSON body row serializer is `adapters::value_to_json`
//! (W13β M4-81 serializer bridge) — the SAME `NodeView`/`RelView` the Bolt
//! `pack_node_with_tenant` encodes — so this exercises the shared
//! materialization + the MCP serializer; the Bolt serializer is pinned by
//! the `transport::bolt::value` unit suite.

use std::sync::Arc;

use arcgraph_core::TenantId;
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
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::PrimaryIndex;
use arcgraph_storage::router::MultiTenantRouter;
use arcgraph_storage::transaction::TxnManager;
use serde_json::{Value, json};

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
    let pool = BufferPool::new(64, io);
    let mgr = Arc::new(TxnManager::new());
    let catalog = Arc::new(SystemCatalog::new());
    catalog.bootstrap(&pool, &mgr).expect("catalog bootstrap");
    let allocator = Arc::new(PageAllocator::new());
    let primary = Arc::new(
        PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&allocator), None)
            .expect("PrimaryIndex::new"),
    );
    let crud = Arc::new(CrudStore::new_with_index(None, primary, allocator));
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

/// Ingest two `Account` nodes with `acc-1 -[:KNOWS]-> acc-2`. The
/// label / rel-type names are interned by storage at ingest; the read
/// path must reverse-resolve them.
fn ingest_accounts(d: &TestDispatcher) {
    let req = JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(1)),
        method: "graph.ingest".into(),
        params: json!({
            "tenant_id": 1,
            "nodes": [
                { "external_id": "acc-1", "label": "Account", "properties": {} },
                { "external_id": "acc-2", "label": "Account", "properties": {} }
            ],
            "relationships": [
                { "from_external_id": "acc-1", "to_external_id": "acc-2",
                  "rel_type": "KNOWS", "properties": {} }
            ],
            "format": "json"
        }),
    };
    let resp = d.dispatch(req).expect("ingest dispatch");
    assert!(resp["error"].is_null(), "ingest must succeed: {resp:?}");
}

fn raw_query_rows(d: &TestDispatcher, query: &str) -> Value {
    let req = JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(2)),
        method: "graph.raw_query".into(),
        params: json!({
            "tenant_id": 1,
            "query": query,
            "max_rows": 100,
            "format": "json"
        }),
    };
    let resp = d.dispatch(req).expect("raw_query dispatch");
    assert!(resp["error"].is_null(), "raw_query failed: {resp:?}");
    let body = resp["result"]["body"].as_str().expect("body string");
    serde_json::from_str(body).expect("parse raw_query body")
}

// =====================================================================
// FACET 2 (production read path) — a MATCH-returned node serializes its
// label NAME, resolved from the storage intern table.
// =====================================================================

#[test]
fn match_returned_node_carries_label_name_over_mcp_production_path() {
    let d = fresh_dispatcher();
    ingest_accounts(&d);

    let rows = raw_query_rows(&d, "MATCH (s:Account) RETURN s");
    assert_eq!(rows["row_count"], 2, "two Account nodes; body={rows}");
    let cells = rows["rows"].as_array().expect("rows array");
    // Every returned node MUST carry labels == ["Account"] (the NAME).
    for row in cells {
        let node = &row.as_array().expect("row array")[0];
        assert_eq!(
            node.get("labels"),
            Some(&json!(["Account"])),
            "production MATCH node must surface labels=['Account'], got {node}"
        );
        // Regression guard: never the opaque id debug form.
        assert_ne!(node.get("labels"), Some(&json!(["LabelId(1)"])));
    }
}

// =====================================================================
// FACET 1 + SISTER (production read path) — `labels(n)` / `type(r)` over
// the real substrate return the interned NAMES.
// =====================================================================

#[test]
fn labels_and_type_resolve_names_over_mcp_production_path() {
    let d = fresh_dispatcher();
    ingest_accounts(&d);

    let rows = raw_query_rows(
        &d,
        "MATCH (a:Account)-[r:KNOWS]->(b:Account) RETURN labels(a), type(r)",
    );
    assert_eq!(rows["row_count"], 1, "one KNOWS edge; body={rows}");
    let row0 = &rows["rows"].as_array().expect("rows")[0];
    let cells = row0.as_array().expect("row cells");
    // labels(a) == ["Account"]; type(r) == "KNOWS" — the NAMES.
    assert_eq!(cells[0], json!(["Account"]), "labels(a) name; body={rows}");
    assert_eq!(cells[1], json!("KNOWS"), "type(r) name; body={rows}");
}

// =====================================================================
// FACET 2 (rel) — a MATCH-returned relationship serializes its type
// NAME over the production path.
// =====================================================================

#[test]
fn match_returned_rel_carries_type_name_over_mcp_production_path() {
    let d = fresh_dispatcher();
    ingest_accounts(&d);

    let rows = raw_query_rows(&d, "MATCH (a:Account)-[r:KNOWS]->(b) RETURN r");
    assert_eq!(rows["row_count"], 1, "one KNOWS edge; body={rows}");
    let rel = &rows["rows"].as_array().expect("rows")[0]
        .as_array()
        .expect("row")[0];
    assert_eq!(
        rel.get("type"),
        Some(&json!("KNOWS")),
        "production MATCH rel must surface type='KNOWS', got {rel}"
    );
    assert_ne!(rel.get("type"), Some(&json!("TypeId(1)")));
}
