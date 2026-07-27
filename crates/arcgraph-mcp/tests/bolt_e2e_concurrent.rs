//! W14δ M5-13 integration test: concurrent Bolt connections.
//!
//! Bolt is one-message-at-a-time per connection per the spec, but
//! the listener MUST handle multiple concurrent connections (one
//! `tokio::spawn`-ed task each) without serialization. This test
//! pins that: N clients connect concurrently, each runs a full
//! HANDSHAKE → HELLO → RUN → PULL → GOODBYE session, all complete
//! successfully.
//!
//! The test also stress-pins the FSM: each connection has its own
//! state, so a RUN+PULL race across two connections must NOT
//! cross-contaminate the active-stream state.

use std::collections::BTreeMap;
use std::sync::Arc;

use arcgraph_mcp::transport::bolt::{
    self, ClientMessage, MAGIC_PREAMBLE, PackValue, SERVER_ACCEPT_V5_0, StubBoltHandler, decode,
    encode_client, message::TAG_RECORD, message::TAG_SUCCESS, read_chunked_message,
    write_chunked_message,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

async fn full_session_against(addr: std::net::SocketAddr, query: &str) -> usize {
    let mut stream = TcpStream::connect(addr).await.unwrap();
    // Handshake.
    let mut req = Vec::new();
    req.extend_from_slice(&MAGIC_PREAMBLE);
    req.extend_from_slice(&[0x00, 0x00, 0x00, 0x05]);
    req.extend_from_slice(&[0; 12]);
    stream.write_all(&req).await.unwrap();
    let mut resp = [0u8; 4];
    stream.read_exact(&mut resp).await.unwrap();
    assert_eq!(resp, SERVER_ACCEPT_V5_0);
    // HELLO.
    let mut buf = Vec::new();
    encode_client(
        &mut buf,
        &ClientMessage::Hello {
            user_agent: Some(format!("concurrent-test/{query}")),
            scheme: Some("none".into()),
            principal: None,
            credentials: None,
            routing: None,
            extras: BTreeMap::new(),
        },
    )
    .unwrap();
    write_chunked_message(&mut stream, &buf).await.unwrap();
    let _ = read_chunked_message(&mut stream).await.unwrap().unwrap();
    // RUN.
    buf.clear();
    encode_client(
        &mut buf,
        &ClientMessage::Run {
            query: query.into(),
            parameters: BTreeMap::new(),
            extra: BTreeMap::new(),
        },
    )
    .unwrap();
    write_chunked_message(&mut stream, &buf).await.unwrap();
    let payload = read_chunked_message(&mut stream).await.unwrap().unwrap();
    let (val, _) = decode(&payload, 0).unwrap();
    assert!(matches!(val, PackValue::Struct { tag, .. } if tag == TAG_SUCCESS));
    // PULL.
    buf.clear();
    encode_client(&mut buf, &ClientMessage::Pull { n: -1, qid: None }).unwrap();
    write_chunked_message(&mut stream, &buf).await.unwrap();
    let mut record_count = 0;
    loop {
        let payload = read_chunked_message(&mut stream).await.unwrap().unwrap();
        let (val, _) = decode(&payload, 0).unwrap();
        match val {
            PackValue::Struct { tag, .. } if tag == TAG_RECORD => {
                record_count += 1;
            }
            PackValue::Struct { tag, .. } if tag == TAG_SUCCESS => break,
            other => panic!("unexpected frame: {other:?}"),
        }
    }
    // GOODBYE.
    buf.clear();
    encode_client(&mut buf, &ClientMessage::Goodbye).unwrap();
    write_chunked_message(&mut stream, &buf).await.unwrap();
    let _ = stream.read(&mut [0u8; 1]).await.unwrap();
    record_count
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ten_concurrent_connections_complete_without_cross_contamination() {
    let handler = Arc::new(StubBoltHandler::accepting());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let listener_task = tokio::spawn(async move {
        let _ = bolt::serve_bolt_inner(
            handler,
            listener,
            async move {
                let _ = rx.await;
            },
            None,
        )
        .await;
    });
    // Spawn 10 concurrent client tasks.
    let mut handles = Vec::new();
    for i in 0..10 {
        // Alternate between the two stub responses to also exercise
        // the per-connection-state independence: "RETURN 1" → 1
        // record, "RETURN 1, 2" → 1 record but different shape.
        let q = if i % 2 == 0 {
            "RETURN 1"
        } else {
            "RETURN 1, 2"
        };
        handles.push(tokio::spawn(
            async move { full_session_against(addr, q).await },
        ));
    }
    let mut total_records = 0;
    for h in handles {
        total_records += h.await.unwrap();
    }
    // 10 sessions × 1 record each = 10 records.
    assert_eq!(total_records, 10);
    let _ = tx.send(());
    let _ = listener_task.await;
}
