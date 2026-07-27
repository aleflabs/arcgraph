//! #1490 RC-blocker gate — ADR-212 principal ACL enforcement over Bolt.
//!
//! Production-shaped chain:
//! Bolt 5.0 TCP handshake + HELLO principal → per-connection session auth →
//! production `StorageBoltHandler` → `QueryEngine` →
//! permission-enforced `CrudExecutorSubstrate` → PackStream records.
//!
//! The scalar projection and count assertions are intentional: filtering only
//! final Node-shaped rows would still leak `RETURN n.body` and `count(n)` after
//! the executor erased node provenance. The explicit-transaction arm pins the
//! independent `run_in_txn` call path as well as auto-commit `run`.

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::sync::Arc;

use arcgraph_core::TenantId;
use arcgraph_mcp::storage::{StorageBackend, StorageBoltHandler, StorageIngestProvider};
use arcgraph_mcp::tools::ingest::{
    AclGrant, IngestBatch, IngestProvider, IngestRecordOutcome, NodeIngest,
};
use arcgraph_mcp::transport::bolt::{
    self, ClientMessage, MAGIC_PREAMBLE, PackValue, SERVER_ACCEPT_V5_0, decode, encode_client,
    message::{TAG_FAILURE, TAG_RECORD, TAG_SUCCESS},
    read_chunked_message, write_chunked_message,
};
use arcgraph_storage::InternTable;
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::catalog::SystemCatalog;
use arcgraph_storage::crud::CrudStore;
use arcgraph_storage::io::InMemoryPageIo;
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::PrimaryIndex;
use arcgraph_storage::router::MultiTenantRouter;
use arcgraph_storage::transaction::TxnManager;
use serde_json::json;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

const ALICE_VISIBLE: &str = "ALICE_VISIBLE_1490";
const BOB_SECRET: &str = "BOB_DENIED_SECRET_1490";

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

fn seed_acl_fixture(backend: &StorageBackend) -> (i64, i64) {
    let nodes = [
        ("fix-1490-alice", ALICE_VISIBLE),
        ("fix-1490-bob", BOB_SECRET),
    ]
    .into_iter()
    .map(|(external_id, body)| NodeIngest {
        external_id: Some(external_id.into()),
        label: "Document".into(),
        properties: BTreeMap::from([("body".into(), json!(body))]),
    })
    .collect();
    let acl_grants = [("fix-1490-alice", "alice"), ("fix-1490-bob", "bob")]
        .into_iter()
        .map(|(external_id, principal)| AclGrant {
            external_id: external_id.into(),
            read_principals: Some(vec![principal.into()]),
        })
        .collect();
    let summary = StorageIngestProvider::new(backend.clone())
        .ingest(
            TenantId::DEFAULT,
            IngestBatch {
                nodes,
                relationships: Vec::new(),
                acl_grants,
            },
        )
        .expect("fixture ingest request");
    assert_eq!(summary.failed_count, 0, "fixture ingest: {summary:?}");
    assert_eq!(summary.inserted_count, 2, "fixture ingest: {summary:?}");
    assert!(
        summary.dropped_acl_grants.is_empty(),
        "all ACL grants must resolve: {summary:?}"
    );
    let ids: Vec<i64> = summary
        .records
        .iter()
        .map(|record| match record {
            IngestRecordOutcome::Inserted { internal_id, .. } => {
                i64::try_from(*internal_id).expect("test node id fits Bolt integer")
            }
            other => panic!("fixture node must be newly inserted: {other:?}"),
        })
        .collect();
    (ids[0], ids[1])
}

async fn start_server(backend: StorageBackend) -> (SocketAddr, tokio::sync::oneshot::Sender<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind Bolt test server");
    let addr = listener.local_addr().expect("Bolt test address");
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(async move {
        bolt::serve_bolt_inner(
            Arc::new(StorageBoltHandler::new(backend)),
            listener,
            async move {
                let _ = shutdown_rx.await;
            },
            None,
        )
        .await
        .expect("Bolt test server");
    });
    (addr, shutdown_tx)
}

async fn send(stream: &mut TcpStream, message: &ClientMessage) -> PackValue {
    let mut encoded = Vec::new();
    encode_client(&mut encoded, message).expect("encode client message");
    write_chunked_message(stream, &encoded)
        .await
        .expect("write client message");
    let payload = read_chunked_message(stream)
        .await
        .expect("read server message")
        .expect("server closed before response");
    decode(&payload, 0).expect("decode server message").0
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

async fn connect(addr: SocketAddr, scheme: &str, principal: Option<&str>) -> TcpStream {
    let mut stream = TcpStream::connect(addr).await.expect("connect Bolt client");
    let mut handshake = Vec::with_capacity(20);
    handshake.extend_from_slice(&MAGIC_PREAMBLE);
    handshake.extend_from_slice(&[0x00, 0x00, 0x00, 0x05]);
    handshake.extend_from_slice(&[0; 12]);
    stream
        .write_all(&handshake)
        .await
        .expect("write Bolt handshake");
    let mut accepted = [0u8; 4];
    stream
        .read_exact(&mut accepted)
        .await
        .expect("read Bolt handshake");
    assert_eq!(accepted, SERVER_ACCEPT_V5_0);

    let hello = ClientMessage::Hello {
        user_agent: Some("fix-1490-acl-gate/1".into()),
        scheme: Some(scheme.into()),
        principal: principal.map(str::to_owned),
        credentials: principal.map(|_| "test-only".into()),
        routing: None,
        extras: BTreeMap::new(),
    };
    assert_success(send(&mut stream, &hello).await, "HELLO");
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
            .expect("server closed during PULL");
        let message = decode(&payload, 0).expect("decode PULL response").0;
        match message {
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

#[tokio::test]
async fn gate_fix_1490_bolt_principal_filters_other_principal_auto_and_explicit() {
    let backend = fresh_backend();
    let (alice_id, bob_id) = seed_acl_fixture(&backend);
    assert_ne!(alice_id, bob_id, "fixture principals need distinct nodes");
    let (addr, shutdown) = start_server(backend).await;

    let mut alice = connect(addr, "basic", Some("alice")).await;
    let scalar_query = "MATCH (n:Document) RETURN id(n) AS node_id";
    assert_eq!(
        run_and_pull(&mut alice, scalar_query).await,
        vec![vec![PackValue::Integer(alice_id)]],
        "auto-commit scalar projection must exclude Bob's node id {bob_id}"
    );
    assert_eq!(
        run_and_pull(
            &mut alice,
            "MATCH (n:Document) RETURN count(n) AS visible_count",
        )
        .await,
        vec![vec![PackValue::Integer(1)]],
        "count store must not disclose denied nodes"
    );

    assert_success(
        send(
            &mut alice,
            &ClientMessage::Begin {
                extra: BTreeMap::new(),
            },
        )
        .await,
        "BEGIN",
    );
    assert_eq!(
        run_and_pull(&mut alice, scalar_query).await,
        vec![vec![PackValue::Integer(alice_id)]],
        "explicit-transaction scalar projection must exclude Bob's node id {bob_id}"
    );
    assert_success(send(&mut alice, &ClientMessage::Rollback).await, "ROLLBACK");

    let mut no_grants = connect(addr, "basic", Some("charlie")).await;
    assert!(
        run_and_pull(&mut no_grants, scalar_query).await.is_empty(),
        "a principal with no grants must receive no content"
    );
    assert_eq!(
        run_and_pull(
            &mut no_grants,
            "MATCH (n:Document) RETURN count(n) AS visible_count",
        )
        .await,
        vec![vec![PackValue::Integer(0)]],
        "a principal with no grants must not get a count oracle"
    );

    shutdown.send(()).expect("stop Bolt test server");
}

#[tokio::test]
async fn gate_fix_1490_bolt_none_scheme_run_fails_closed_unauthorized() {
    let backend = fresh_backend();
    seed_acl_fixture(&backend);
    let (addr, shutdown) = start_server(backend).await;
    let mut anonymous = connect(addr, "none", None).await;

    let failure = send(
        &mut anonymous,
        &ClientMessage::Run {
            query: "MATCH (n:Document) RETURN n.body AS body".into(),
            parameters: BTreeMap::new(),
            extra: BTreeMap::new(),
        },
    )
    .await;
    match failure {
        PackValue::Struct {
            tag: TAG_FAILURE,
            fields,
        } => {
            let [PackValue::Map(metadata)] = fields.as_slice() else {
                panic!("FAILURE metadata shape: {fields:?}");
            };
            assert_eq!(
                metadata.get("code"),
                Some(&PackValue::String(
                    "Neo.ClientError.Security.Unauthorized".into()
                )),
                "principal-less content RUN must use Bolt's security failure surface"
            );
        }
        other => panic!("principal-less content RUN must fail, got {other:?}"),
    }

    shutdown.send(()).expect("stop Bolt test server");
}
