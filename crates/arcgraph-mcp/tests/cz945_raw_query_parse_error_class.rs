//! #945 — syntactically invalid ArcQL inside a valid JSON-RPC
//! `graph.raw_query` envelope is a query-domain fault (-32005), not a
//! JSON-RPC frame parse fault (-32700).

use std::sync::Arc;

use arcgraph_core::TenantId;
use arcgraph_mcp::storage::{
    StorageBackend, StorageHybridSearcher, StorageIngestProvider, StorageNeighborhoodExplorer,
    StorageNodeInspector, StorageRawQueryExecutor, StorageSchemaProvider,
};
use arcgraph_mcp::{
    CODE_INTERNAL_ERROR, CODE_INVALID_PARAMS, CODE_PARSE_ERROR, CODE_QUERY_ERROR, Dispatcher,
    SessionScope, handle_raw_envelope, serve_stdio,
};
use arcgraph_query::CancellationRegistry;
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

fn dispatcher() -> TestDispatcher {
    let backend = fresh_backend();
    Dispatcher::with_session_scope(
        TenantId::DEFAULT,
        SessionScope::Power,
        Arc::new(StorageSchemaProvider::new(backend.clone())),
        Arc::new(StorageNodeInspector::new(backend.clone())),
        Arc::new(StorageNeighborhoodExplorer::new(backend.clone())),
        Arc::new(StorageHybridSearcher::new(backend.clone())),
        Arc::new(StorageIngestProvider::new(backend.clone())),
        Arc::new(StorageRawQueryExecutor::new(backend)),
    )
}

fn raw_query_envelope(query: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": "rq-945",
        "method": "graph.raw_query",
        "params": {
            "tenant_id": 1,
            "query": query,
            "max_rows": 100,
            "format": "json"
        }
    })
}

fn frame(payload: &str) -> Vec<u8> {
    let mut out = format!("Content-Length: {}\r\n\r\n", payload.len()).into_bytes();
    out.extend_from_slice(payload.as_bytes());
    out
}

fn first_stdio_json(output: &[u8]) -> Value {
    let s = std::str::from_utf8(output).expect("stdio output utf8");
    let (_, body) = s.split_once("\r\n\r\n").expect("framed response body");
    serde_json::from_str(body).expect("json response body")
}

fn assert_raw_query_parse_error_is_query_error(resp: &Value) {
    assert_eq!(resp["id"], "rq-945");
    assert_eq!(resp["error"]["code"], CODE_QUERY_ERROR);
    assert_ne!(resp["error"]["code"], CODE_PARSE_ERROR);
    assert_eq!(resp["error"]["message"], "query error");
    let data = resp["error"]["data"]
        .as_str()
        .expect("pest detail carried in error.data");
    assert!(
        data.contains("pest parse error at 1:"),
        "line/column detail missing: {data}"
    );
    assert!(
        data.contains("expected"),
        "expected-token detail missing: {data}"
    );
}

fn assert_raw_query_missing_parameter_is_invalid_params(resp: &Value) {
    assert_eq!(resp["id"], "rq-945");
    assert_eq!(resp["error"]["code"], CODE_INVALID_PARAMS);
    assert_ne!(resp["error"]["code"], CODE_INTERNAL_ERROR);
    assert_eq!(resp["error"]["message"], "invalid params");
    assert_eq!(resp["error"]["data"], "missing parameter: $missing");
}

#[test]
fn dispatcher_raw_query_arcql_parse_fault_is_query_error() {
    let d = dispatcher();
    let resp = handle_raw_envelope(&d, raw_query_envelope("MATCH (n:Person RETURN n"))
        .expect("dispatcher response");
    assert_raw_query_parse_error_is_query_error(&resp);
}

#[test]
fn dispatcher_raw_query_missing_parameter_is_invalid_params() {
    let d = dispatcher();
    let resp = handle_raw_envelope(&d, raw_query_envelope("RETURN $missing AS y"))
        .expect("dispatcher response");
    assert_raw_query_missing_parameter_is_invalid_params(&resp);
}

#[tokio::test]
async fn stdio_raw_query_arcql_parse_fault_is_query_error() {
    let d = dispatcher();
    let cr = CancellationRegistry::new();
    let payload = raw_query_envelope("MATCH (n:Person RETURN n").to_string();
    let input = frame(&payload);
    let mut output = Vec::new();
    let (_tx, rx) = tokio::sync::oneshot::channel::<()>();
    let shutdown = async move {
        let _ = rx.await;
    };

    let stats = serve_stdio(
        std::sync::Arc::new(d),
        &cr,
        &input[..],
        &mut output,
        shutdown,
        None,
    )
    .await
    .expect("stdio exits after EOF");

    assert_eq!(stats.messages_in, 1);
    assert_eq!(stats.messages_out, 1);
    assert_eq!(stats.parse_errors, 0);
    let resp = first_stdio_json(&output);
    assert_raw_query_parse_error_is_query_error(&resp);
}

#[tokio::test]
async fn stdio_raw_query_missing_parameter_is_invalid_params() {
    let d = dispatcher();
    let cr = CancellationRegistry::new();
    let payload = raw_query_envelope("RETURN $missing AS y").to_string();
    let input = frame(&payload);
    let mut output = Vec::new();
    let (_tx, rx) = tokio::sync::oneshot::channel::<()>();
    let shutdown = async move {
        let _ = rx.await;
    };

    let stats = serve_stdio(
        std::sync::Arc::new(d),
        &cr,
        &input[..],
        &mut output,
        shutdown,
        None,
    )
    .await
    .expect("stdio exits after EOF");

    assert_eq!(stats.messages_in, 1);
    assert_eq!(stats.messages_out, 1);
    assert_eq!(stats.parse_errors, 0);
    let resp = first_stdio_json(&output);
    assert_raw_query_missing_parameter_is_invalid_params(&resp);
}

#[tokio::test]
async fn stdio_malformed_json_frame_still_returns_parse_error() {
    let d = dispatcher();
    let cr = CancellationRegistry::new();
    let bad = "Content-Length: 5\r\n\r\n{bad}";
    let mut output = Vec::new();
    let (_tx, rx) = tokio::sync::oneshot::channel::<()>();
    let shutdown = async move {
        let _ = rx.await;
    };

    let stats = serve_stdio(
        std::sync::Arc::new(d),
        &cr,
        bad.as_bytes(),
        &mut output,
        shutdown,
        None,
    )
    .await
    .expect("stdio exits after EOF");

    assert_eq!(stats.messages_in, 0);
    assert_eq!(stats.messages_out, 0);
    assert_eq!(stats.parse_errors, 1);
    let resp = first_stdio_json(&output);
    assert_eq!(resp["id"], Value::Null);
    assert_eq!(resp["error"]["code"], CODE_PARSE_ERROR);
    assert_eq!(resp["error"]["message"], "parse error");
}
