//! W14δ M5-13 integration test: end-to-end Bolt session over a real
//! TCP listener.
//!
//! This is the integration counterpart to the in-process duplex test
//! in `transport::bolt::server::tests::full_hello_run_pull_session_...`.
//! It binds an actual `127.0.0.1:0` TCP listener, runs the v1.0-α
//! server scaffold, and drives an in-tree Bolt 5.0 client across the
//! TCP socket. The pin: the full HANDSHAKE → HELLO → RUN → PULL →
//! GOODBYE sequence works without the duplex shortcut.
//!
//! The spawn prompt asked for a "bolt-client in-tree" integration
//! test. The crates.io `bolt-client` v0.11.0 dependency does NOT
//! support Bolt 5.0 (it pins 4.x); rather than add an old client we
//! built an in-tree client out of the same PackStream + chunk-framing
//! + message codec the server uses. That's what "in-tree" means
//!   here: zero external client deps + bidirectional codec exercise.

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use arcgraph_mcp::transport::bolt::{
    self, ClientMessage, MAGIC_PREAMBLE, PackValue, SERVER_ACCEPT_V5_0, StubBoltHandler, decode,
    encode_client, message::TAG_RECORD, message::TAG_SUCCESS, read_chunked_message,
    write_chunked_message,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// Helper: run the listener on a fresh ephemeral port, return its
/// SocketAddr + a oneshot to signal shutdown.
async fn spawn_listener(
    handler: Arc<StubBoltHandler>,
) -> (SocketAddr, tokio::sync::oneshot::Sender<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(async move {
        let shutdown = async move {
            let _ = rx.await;
        };
        let _ = bolt::serve_bolt_inner(handler, listener, shutdown, None).await;
    });
    (addr, tx)
}

async fn handshake(stream: &mut TcpStream) {
    let mut req = Vec::new();
    req.extend_from_slice(&MAGIC_PREAMBLE);
    req.extend_from_slice(&[0x00, 0x00, 0x00, 0x05]); // Bolt 5.0
    req.extend_from_slice(&[0; 12]);
    stream.write_all(&req).await.unwrap();
    let mut resp = [0u8; 4];
    stream.read_exact(&mut resp).await.unwrap();
    assert_eq!(resp, SERVER_ACCEPT_V5_0);
}

async fn send_message(stream: &mut TcpStream, msg: &ClientMessage) {
    let mut buf = Vec::new();
    encode_client(&mut buf, msg).unwrap();
    write_chunked_message(stream, &buf).await.unwrap();
}

async fn read_reply(stream: &mut TcpStream) -> PackValue {
    let payload = read_chunked_message(stream).await.unwrap().unwrap();
    let (val, n) = decode(&payload, 0).unwrap();
    assert_eq!(n, payload.len());
    val
}

#[tokio::test]
async fn full_session_over_tcp_listener() {
    let handler = Arc::new(StubBoltHandler::accepting());
    let (addr, shutdown) = spawn_listener(handler).await;
    // Connect.
    let mut stream = tokio::time::timeout(Duration::from_secs(2), TcpStream::connect(addr))
        .await
        .expect("connect timed out")
        .expect("connect failed");
    // HANDSHAKE.
    handshake(&mut stream).await;
    // HELLO.
    let mut hello_extras = BTreeMap::new();
    hello_extras.insert(
        "bolt_agent".into(),
        PackValue::Map({
            let mut m = BTreeMap::new();
            m.insert(
                "product".into(),
                PackValue::String("arcgraph-test/1.0".into()),
            );
            m
        }),
    );
    send_message(
        &mut stream,
        &ClientMessage::Hello {
            user_agent: Some("arcgraph-test/1.0".into()),
            scheme: Some("basic".into()),
            principal: Some("alice".into()),
            credentials: Some("secret".into()),
            routing: None,
            extras: hello_extras,
        },
    )
    .await;
    let reply = read_reply(&mut stream).await;
    match reply {
        PackValue::Struct { tag, fields } => {
            assert_eq!(tag, TAG_SUCCESS, "HELLO reply must be SUCCESS");
            let meta = match fields.into_iter().next() {
                Some(PackValue::Map(m)) => m,
                _ => panic!("SUCCESS body is not a map"),
            };
            assert!(meta.contains_key("connection_id"));
            assert!(meta.contains_key("server"));
        }
        other => panic!("expected struct, got {other:?}"),
    }
    // RUN.
    send_message(
        &mut stream,
        &ClientMessage::Run {
            query: "RETURN 1".into(),
            parameters: BTreeMap::new(),
            extra: BTreeMap::new(),
        },
    )
    .await;
    let reply = read_reply(&mut stream).await;
    match reply {
        PackValue::Struct { tag, fields } => {
            assert_eq!(tag, TAG_SUCCESS);
            let meta = match fields.into_iter().next() {
                Some(PackValue::Map(m)) => m,
                _ => panic!(),
            };
            // RUN SUCCESS carries field-name list.
            let fields = meta
                .get("fields")
                .and_then(|v| match v {
                    PackValue::List(l) => Some(l),
                    _ => None,
                })
                .expect("fields list present");
            assert_eq!(fields.len(), 1);
            assert!(matches!(&fields[0], PackValue::String(s) if s == "n"));
        }
        other => panic!("expected RUN SUCCESS, got {other:?}"),
    }
    // PULL.
    send_message(&mut stream, &ClientMessage::Pull { n: -1, qid: None }).await;
    // Drain RECORD(s) then final SUCCESS.
    let mut record_count = 0;
    loop {
        let reply = read_reply(&mut stream).await;
        match reply {
            PackValue::Struct { tag, .. } if tag == TAG_RECORD => {
                record_count += 1;
            }
            PackValue::Struct { tag, fields } if tag == TAG_SUCCESS => {
                let meta = match fields.into_iter().next() {
                    Some(PackValue::Map(m)) => m,
                    _ => panic!(),
                };
                assert_eq!(meta.get("has_more"), Some(&PackValue::Boolean(false)));
                break;
            }
            other => panic!("unexpected frame: {other:?}"),
        }
    }
    assert_eq!(record_count, 1, "stub returns exactly 1 row for RETURN 1");
    // GOODBYE — server closes without replying.
    send_message(&mut stream, &ClientMessage::Goodbye).await;
    // The next read should return clean EOF.
    let n = stream.read(&mut [0u8; 1]).await.unwrap();
    assert_eq!(n, 0, "server closed after GOODBYE");
    let _ = shutdown.send(());
}

#[tokio::test]
async fn handshake_rejection_closes_socket_with_zero_response() {
    let handler = Arc::new(StubBoltHandler::accepting());
    let (addr, shutdown) = spawn_listener(handler).await;
    let mut stream = TcpStream::connect(addr).await.unwrap();
    // Send magic + four Bolt 4.x offers (none match 5.0).
    let mut req = Vec::new();
    req.extend_from_slice(&MAGIC_PREAMBLE);
    req.extend_from_slice(&[0x00, 0x00, 0x04, 0x04]);
    req.extend_from_slice(&[0x00, 0x00, 0x03, 0x04]);
    req.extend_from_slice(&[0x00, 0x00, 0x02, 0x04]);
    req.extend_from_slice(&[0x00, 0x00, 0x01, 0x04]);
    stream.write_all(&req).await.unwrap();
    // Read 4-byte zero response.
    let mut resp = [0u8; 4];
    stream.read_exact(&mut resp).await.unwrap();
    assert_eq!(resp, [0u8; 4]);
    // Server should close.
    let n = stream.read(&mut [0u8; 16]).await.unwrap();
    assert_eq!(n, 0, "server closed after rejection");
    let _ = shutdown.send(());
}
