//! #1638 release-blocker regression — a legacy JSON property bag written by
//! MCP must materialize identically through MCP and the Bolt wire.
//!
//! The fixture deliberately grants the ingested node to the Bolt principal.
//! MCP reads through the production dispatcher; Bolt reads through a real
//! protocol handshake and `RUN`/`PULL` session backed by the same store.
//! Reverting the requested-name projection fix changes the Bolt row from
//! `["Ada", 37]` to `[null, null]` while MCP remains correct.

use std::collections::BTreeMap;
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;

use arcgraph_core::TenantId;
use arcgraph_mcp::jsonrpc::JsonRpcRequest;
use arcgraph_mcp::storage::{
    StorageBackend, StorageBoltHandler, StorageHybridSearcher, StorageIngestProvider,
    StorageNeighborhoodExplorer, StorageNodeInspector, StorageRawQueryExecutor,
    StorageSchemaProvider,
};
use arcgraph_mcp::transport::bolt::{
    self, ClientMessage, MAGIC_PREAMBLE, PackValue, SERVER_ACCEPT_V5_0, decode, encode_client,
    message::{TAG_RECORD, TAG_SUCCESS},
    read_chunked_message, write_chunked_message,
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
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

const QUERY: &str = "MATCH (p:Person) RETURN p.name AS name, p.age AS age";

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
    let txn_manager = Arc::new(TxnManager::new());
    let catalog = Arc::new(SystemCatalog::new());
    catalog
        .bootstrap(&pool, &txn_manager)
        .expect("catalog bootstrap");
    let allocator = Arc::new(PageAllocator::new());
    let primary = Arc::new(
        PrimaryIndex::new(Arc::clone(&txn_manager), Arc::clone(&allocator), None)
            .expect("primary index"),
    );
    let crud = Arc::new(CrudStore::new_with_index(None, primary, allocator));
    let router = Arc::new(MultiTenantRouter::new(catalog, crud, None));
    StorageBackend::new(router, txn_manager, Arc::new(InternTable::new()))
}

fn dispatcher_over(backend: &StorageBackend) -> TestDispatcher {
    Dispatcher::with_session_scope_and_rate_limiter(
        TenantId::DEFAULT,
        SessionScope::Power,
        Arc::new(StorageSchemaProvider::new(backend.clone())),
        Arc::new(StorageNodeInspector::new(backend.clone())),
        Arc::new(StorageNeighborhoodExplorer::new(backend.clone())),
        Arc::new(StorageHybridSearcher::new(backend.clone())),
        Arc::new(StorageIngestProvider::new(backend.clone())),
        Arc::new(StorageRawQueryExecutor::new(backend.clone())),
        RateLimiter::new(),
    )
}

fn dispatch_body(dispatcher: &TestDispatcher, method: &str, params: Value) -> Value {
    let response = dispatcher
        .dispatch(JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(json!(1)),
            method: method.into(),
            params,
        })
        .unwrap_or_else(|| panic!("{method} dispatch returned no envelope"));
    assert!(
        response["error"].is_null(),
        "{method} returned an error envelope: {response:?}"
    );
    let body = response["result"]["body"]
        .as_str()
        .unwrap_or_else(|| panic!("{method} body missing: {response:?}"));
    serde_json::from_str(body).unwrap_or_else(|error| panic!("{method} body is not JSON: {error}"))
}

async fn start_bolt(
    backend: StorageBackend,
) -> (
    SocketAddr,
    tokio::sync::oneshot::Sender<()>,
    tokio::task::JoinHandle<()>,
) {
    let listener = tokio::net::TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
        .await
        .expect("bind Bolt regression server");
    let address = listener.local_addr().expect("Bolt regression address");
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(async move {
        bolt::serve_bolt_inner(
            Arc::new(StorageBoltHandler::new(backend)),
            listener,
            async move {
                let _ = shutdown_rx.await;
            },
            None,
        )
        .await
        .expect("Bolt regression server");
    });
    (address, shutdown_tx, server)
}

async fn send(stream: &mut TcpStream, message: &ClientMessage) -> PackValue {
    let mut encoded = Vec::new();
    encode_client(&mut encoded, message).expect("encode Bolt client message");
    write_chunked_message(stream, &encoded)
        .await
        .expect("write Bolt client message");
    let payload = read_chunked_message(stream)
        .await
        .expect("read Bolt server message")
        .expect("Bolt server closed before responding");
    decode(&payload, 0).expect("decode Bolt server message").0
}

fn assert_success(message: PackValue, operation: &str) {
    assert!(
        matches!(
            message,
            PackValue::Struct {
                tag: TAG_SUCCESS,
                ..
            }
        ),
        "{operation} must succeed, got {message:?}"
    );
}

async fn connect_bolt(address: SocketAddr) -> TcpStream {
    let mut stream = TcpStream::connect(address)
        .await
        .expect("connect Bolt regression client");
    let mut handshake = Vec::with_capacity(20);
    handshake.extend_from_slice(&MAGIC_PREAMBLE);
    handshake.extend_from_slice(&[0x00, 0x00, 0x00, 0x05]);
    handshake.extend_from_slice(&[0; 12]);
    stream
        .write_all(&handshake)
        .await
        .expect("write Bolt handshake");
    let mut accepted = [0_u8; 4];
    stream
        .read_exact(&mut accepted)
        .await
        .expect("read Bolt handshake");
    assert_eq!(accepted, SERVER_ACCEPT_V5_0);

    assert_success(
        send(
            &mut stream,
            &ClientMessage::Hello {
                user_agent: Some("fix-1638-regression/1".into()),
                scheme: Some("basic".into()),
                principal: Some("neo4j".into()),
                credentials: Some("test-only".into()),
                routing: None,
                extras: BTreeMap::new(),
            },
        )
        .await,
        "HELLO",
    );
    stream
}

async fn run_and_pull(stream: &mut TcpStream, query: &str) -> Vec<Vec<PackValue>> {
    assert_success(
        send(
            stream,
            &ClientMessage::Run {
                query: query.into(),
                parameters: BTreeMap::new(),
                extra: BTreeMap::new(),
            },
        )
        .await,
        "RUN",
    );

    let mut encoded = Vec::new();
    encode_client(&mut encoded, &ClientMessage::Pull { n: -1, qid: None }).expect("encode PULL");
    write_chunked_message(stream, &encoded)
        .await
        .expect("write PULL");
    let mut records = Vec::new();
    loop {
        let payload = read_chunked_message(stream)
            .await
            .expect("read PULL response")
            .expect("Bolt server closed during PULL");
        match decode(&payload, 0).expect("decode PULL response").0 {
            PackValue::Struct {
                tag: TAG_RECORD,
                mut fields,
            } => match fields.pop() {
                Some(PackValue::List(row)) if fields.is_empty() => records.push(row),
                other => panic!("malformed RECORD fields: {other:?}; remainder={fields:?}"),
            },
            PackValue::Struct {
                tag: TAG_SUCCESS, ..
            } => break,
            other => panic!("PULL must emit RECORD* then SUCCESS, got {other:?}"),
        }
    }
    records
}

fn bolt_rows_as_json(records: &[Vec<PackValue>]) -> Value {
    Value::Array(
        records
            .iter()
            .map(|row| {
                Value::Array(
                    row.iter()
                        .map(|cell| match cell {
                            PackValue::Null => Value::Null,
                            PackValue::Boolean(value) => Value::Bool(*value),
                            PackValue::Integer(value) => Value::from(*value),
                            PackValue::Float(value) => json!(value),
                            PackValue::String(value) => Value::String(value.clone()),
                            other => panic!("unexpected scalar Bolt cell: {other:?}"),
                        })
                        .collect(),
                )
            })
            .collect(),
    )
}

#[tokio::test]
async fn mcp_ingest_reads_equal_and_correct_over_mcp_and_bolt() {
    let backend = fresh_backend();
    let dispatcher = dispatcher_over(&backend);

    let ingest = dispatch_body(
        &dispatcher,
        "graph.ingest",
        json!({
            "tenant_id": 1,
            "format": "json",
            "nodes": [{
                "external_id": "person-ada",
                "label": "Person",
                "properties": { "name": "Ada", "age": 37 }
            }],
            "relationships": [],
            "acl_grants": [{
                "external_id": "person-ada",
                "read_principals": ["neo4j"]
            }]
        }),
    );
    assert_eq!(ingest["inserted_count"], 1, "MCP ingest: {ingest}");
    assert_eq!(ingest["failed_count"], 0, "MCP ingest: {ingest}");

    let mcp = dispatch_body(
        &dispatcher,
        "graph.raw_query",
        json!({
            "tenant_id": 1,
            "format": "json",
            "query": QUERY,
            "max_rows": 100
        }),
    );
    assert_eq!(mcp["columns"], json!(["name", "age"]));
    assert_eq!(mcp["rows"], json!([["Ada", 37]]), "MCP read: {mcp}");

    let (address, shutdown, server) = start_bolt(backend).await;
    let mut bolt = connect_bolt(address).await;
    let bolt_rows = bolt_rows_as_json(&run_and_pull(&mut bolt, QUERY).await);
    assert_eq!(
        bolt_rows, mcp["rows"],
        "MCP and Bolt must materialize the same ingested values"
    );
    assert_eq!(
        bolt_rows,
        json!([["Ada", 37]]),
        "Bolt must return values, never successful nulls"
    );

    drop(bolt);
    shutdown.send(()).expect("request Bolt server shutdown");
    server.await.expect("join Bolt regression server");
}
