// Each test binary uses only the helpers it needs; allow the
// shared module to expose the full set without per-binary dead-code
// warnings.
#![allow(dead_code)]

//! Shared test harness for the W27-β `graph.raw_query` write-op
//! integration tests (ADR-153).
//!
//! Each of the 7 sibling test files in `tests/raw_query_write_*` uses
//! `mod common;` to pull this helper in. Cargo treats
//! `tests/raw_query_write_common/mod.rs` as a regular module (NOT its
//! own test binary) because of the trailing `mod.rs`; an alternative
//! `tests/raw_query_write_common.rs` would compile as a SEPARATE test
//! binary and lose the `pub fn` re-use.
//!
//! # Fixture shape
//!
//! [`fresh_backend`] mirrors the
//! `crates/arcgraph-mcp/tests/m4_08_finalize_end_to_end.rs::fresh_backend`
//! pattern: in-memory storage with the [`PrimaryIndex`] wired so the
//! `CatalogStats` hook fires inside `crud::commit` (without it,
//! label / rel-type cardinalities never surface in the catalog and
//! the binding pass falls through to dynamic-name resolution → silent
//! zero rows for any label-anchored MATCH; see W23-M4-08-FINALIZE
//! retro for the founding incident).
//!
//! [`fresh_dispatcher`] composes the production-shaped adapter bundle
//! around the backend with the Power-scope session so `graph.raw_query`
//! routes through end-to-end. The 5 sibling adapters
//! (Schema / Inspector / Explorer / Search / Ingest) are wired to the
//! same backend so a `graph.ingest` call followed by `graph.raw_query`
//! observe the same per-tenant state.
//!
//! # Why through the `Dispatcher` (not direct `raw_query_tool`)?
//!
//! The audit-2026-05-27 boundary requirement is "writes exposed via the
//! MCP `graph.raw_query` SURFACE." Going through the JSON-RPC
//! `Dispatcher::dispatch` exercises BOTH the `raw_query_tool` boundary
//! AND the per-tenant guard / scope check / rate-limit gate. The
//! existing `raw_query_integ.rs` regression suite already pins the
//! direct-`raw_query_tool` invariants; the W27-β slice wants the
//! end-to-end pin.

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

/// Production-shaped Dispatcher type alias.
pub type TestDispatcher = Dispatcher<
    StorageSchemaProvider,
    StorageNodeInspector,
    StorageNeighborhoodExplorer,
    StorageHybridSearcher,
    StorageIngestProvider,
    StorageRawQueryExecutor,
>;

/// Build a fresh in-memory backend with PrimaryIndex wired so the
/// `CatalogStats` hook fires inside `crud::commit`. Without the index
/// wiring, label / rel-type cardinalities never reach the catalog and
/// `MATCH (n:Label)` queries silently return 0 rows.
pub fn fresh_backend() -> StorageBackend {
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

/// Build a Power-scope Dispatcher over the backend with the 6
/// production adapters wired. The fixture binds the session to
/// `TenantId::DEFAULT` (= tenant_id 1) so request envelopes use that
/// value.
pub fn fresh_dispatcher() -> TestDispatcher {
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

/// Dispatch a `graph.raw_query` JSON-RPC envelope. Returns the raw
/// response value (caller asserts on `result` / `error`).
///
/// The `tenant_id` in the envelope is always `1` to match the
/// `TenantId::DEFAULT` binding in [`fresh_dispatcher`].
pub fn raw_query(d: &TestDispatcher, query: &str) -> Value {
    let req = JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(1)),
        method: "graph.raw_query".into(),
        params: json!({
            "tenant_id": 1,
            "query": query,
            "max_rows": 100,
            "format": "json"
        }),
    };
    d.dispatch(req).expect("raw_query dispatch")
}

/// Parse the response body into a `serde_json::Value` for assertions.
/// Panics with a clear message if the envelope carries an error or the
/// body is missing.
pub fn parse_body(resp: &Value) -> Value {
    assert!(
        resp["error"].is_null(),
        "raw_query failed with error envelope: {resp:?}"
    );
    let body = resp["result"]["body"]
        .as_str()
        .unwrap_or_else(|| panic!("raw_query body is missing or not a string: {resp:?}"));
    serde_json::from_str(body)
        .unwrap_or_else(|e| panic!("raw_query body is not valid JSON: {body} (err={e:?})"))
}

/// Helper: assert the response's writes summary matches the expected
/// counter values exactly. Takes a tuple of 8 `u64` counters in the
/// canonical WriteSummary order: `(nodes_created, nodes_deleted,
/// rels_created, rels_deleted, properties_set, properties_removed,
/// labels_added, labels_removed)`. Tuple instead of 9 positional
/// parameters per the `too_many_arguments` clippy lint.
pub fn assert_writes(parsed: &Value, expected: (u64, u64, u64, u64, u64, u64, u64, u64)) {
    let (
        nodes_created,
        nodes_deleted,
        rels_created,
        rels_deleted,
        properties_set,
        properties_removed,
        labels_added,
        labels_removed,
    ) = expected;
    let w = &parsed["writes"];
    assert_eq!(
        w["nodes_created"], nodes_created,
        "writes.nodes_created: {w:?}"
    );
    assert_eq!(
        w["nodes_deleted"], nodes_deleted,
        "writes.nodes_deleted: {w:?}"
    );
    assert_eq!(
        w["rels_created"], rels_created,
        "writes.rels_created: {w:?}"
    );
    assert_eq!(
        w["rels_deleted"], rels_deleted,
        "writes.rels_deleted: {w:?}"
    );
    assert_eq!(
        w["properties_set"], properties_set,
        "writes.properties_set: {w:?}"
    );
    assert_eq!(
        w["properties_removed"], properties_removed,
        "writes.properties_removed: {w:?}"
    );
    assert_eq!(
        w["labels_added"], labels_added,
        "writes.labels_added: {w:?}"
    );
    assert_eq!(
        w["labels_removed"], labels_removed,
        "writes.labels_removed: {w:?}"
    );
}
