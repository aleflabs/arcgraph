//! W16γ M6-07 — integration test: `serve_bolt_listener` emits
//! `arcgraph_active_connections{transport="bolt"}` through the
//! accept-loop incr + RAII guard decrement.
//!
//! Pin: after one connection completes its session and the
//! per-connection task drops, the gauge reads 0 (the RAII guard
//! fired). After two concurrent connections active, the gauge
//! reads 2.
//!
//! ADR-045 §"Decision": stdio/bolt active_connections is intra-mcp
//! (no MetricsSink trait); the accept-point increments + RAII guard
//! decrements mirror the `http.rs:823 ActiveConnGuard` pattern.
//!
//! Spawn-prompt constraint: this wire is at the ACCEPT loop, NOT at
//! the `BoltServerConfig::validate` security path from #321.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use arcgraph_mcp::transport::bolt::{self, MAGIC_PREAMBLE, SERVER_ACCEPT_V5_0, StubBoltHandler};
use arcgraph_mcp::transport::metrics::MetricsRegistry;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

async fn spawn_listener_with_metrics(
    handler: Arc<StubBoltHandler>,
    metrics: Arc<MetricsRegistry>,
) -> (SocketAddr, tokio::sync::oneshot::Sender<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(async move {
        let shutdown = async move {
            let _ = rx.await;
        };
        let _ = bolt::serve_bolt_inner(handler, listener, shutdown, Some(metrics)).await;
    });
    (addr, tx)
}

/// Pin: after a single connection establishes the handshake and
/// then closes, the bolt active_connections gauge returns to 0.
/// Tests the RAII guard's Drop fires on connection-task end.
#[tokio::test]
async fn bolt_active_connections_gauge_returns_to_zero_after_session() {
    let handler = Arc::new(StubBoltHandler::accepting());
    let metrics = MetricsRegistry::shared().expect("metrics init");
    let (addr, shutdown) = spawn_listener_with_metrics(handler, metrics.clone()).await;

    // Connect.
    let mut stream = tokio::time::timeout(Duration::from_secs(2), TcpStream::connect(addr))
        .await
        .expect("connect timed out")
        .expect("connect failed");
    // HANDSHAKE.
    let mut req = Vec::new();
    req.extend_from_slice(&MAGIC_PREAMBLE);
    req.extend_from_slice(&[0x00, 0x00, 0x00, 0x05]);
    req.extend_from_slice(&[0; 12]);
    stream.write_all(&req).await.unwrap();
    let mut resp = [0u8; 4];
    stream.read_exact(&mut resp).await.unwrap();
    assert_eq!(resp, SERVER_ACCEPT_V5_0);

    // Drop the client → server task exits → RAII guard decrements.
    drop(stream);
    // Allow the server task to observe the closed stream + Drop the
    // guard. Bolt's protocol loop is a select! with broadcast
    // shutdown; closing the TCP stream surfaces EOF on the next
    // read which exits the loop.
    //
    // 200ms is generous on CI: the loop polls in O(µs) — we just
    // need ONE scheduler tick for the spawned task to advance past
    // the protocol-loop-exit + run Drop.
    tokio::time::sleep(Duration::from_millis(200)).await;

    let text = String::from_utf8(metrics.gather_text().expect("gather")).expect("utf-8");
    assert!(
        text.contains(r#"arcgraph_active_connections{transport="bolt"} 0"#),
        "post-session, bolt gauge must be 0 (RAII guard fired); text was:\n{text}"
    );
    let _ = shutdown.send(());
}

/// Pin: with two concurrent open connections, the gauge reads 2
/// while both are alive; drops to 0 once both close.
#[tokio::test]
async fn bolt_active_connections_gauge_tracks_concurrent_sessions() {
    let handler = Arc::new(StubBoltHandler::accepting());
    let metrics = MetricsRegistry::shared().expect("metrics init");
    let (addr, shutdown) = spawn_listener_with_metrics(handler, metrics.clone()).await;

    // Open two connections + complete their handshakes so both
    // tasks are in the protocol-loop state where the active-conn
    // counter is at 2.
    let mut s1 = TcpStream::connect(addr).await.unwrap();
    let mut req = Vec::new();
    req.extend_from_slice(&MAGIC_PREAMBLE);
    req.extend_from_slice(&[0x00, 0x00, 0x00, 0x05]);
    req.extend_from_slice(&[0; 12]);
    s1.write_all(&req).await.unwrap();
    let mut resp = [0u8; 4];
    s1.read_exact(&mut resp).await.unwrap();
    assert_eq!(resp, SERVER_ACCEPT_V5_0);

    let mut s2 = TcpStream::connect(addr).await.unwrap();
    let mut req = Vec::new();
    req.extend_from_slice(&MAGIC_PREAMBLE);
    req.extend_from_slice(&[0x00, 0x00, 0x00, 0x05]);
    req.extend_from_slice(&[0; 12]);
    s2.write_all(&req).await.unwrap();
    let mut resp = [0u8; 4];
    s2.read_exact(&mut resp).await.unwrap();
    assert_eq!(resp, SERVER_ACCEPT_V5_0);

    // Give the server task a beat to record both increments.
    tokio::time::sleep(Duration::from_millis(100)).await;
    let text = String::from_utf8(metrics.gather_text().expect("gather")).expect("utf-8");
    assert!(
        text.contains(r#"arcgraph_active_connections{transport="bolt"} 2"#),
        "two concurrent sessions: gauge must be 2; text was:\n{text}"
    );

    // Close both. RAII guards decrement on task end.
    drop(s1);
    drop(s2);
    tokio::time::sleep(Duration::from_millis(200)).await;
    let text2 = String::from_utf8(metrics.gather_text().expect("gather")).expect("utf-8");
    assert!(
        text2.contains(r#"arcgraph_active_connections{transport="bolt"} 0"#),
        "both sessions closed: gauge must be 0; text was:\n{text2}"
    );

    let _ = shutdown.send(());
}

/// Pin: handshake rejection still releases the gauge slot. Without
/// the RAII guard, a `BoltError::HandshakeRejected` would leave the
/// gauge permanently inflated.
#[tokio::test]
async fn bolt_active_connections_gauge_releases_on_handshake_rejection() {
    let handler = Arc::new(StubBoltHandler::accepting());
    let metrics = MetricsRegistry::shared().expect("metrics init");
    let (addr, shutdown) = spawn_listener_with_metrics(handler, metrics.clone()).await;

    let mut stream = TcpStream::connect(addr).await.unwrap();
    // Send magic + four Bolt 4.x offers (none match 5.0).
    let mut req = Vec::new();
    req.extend_from_slice(&MAGIC_PREAMBLE);
    req.extend_from_slice(&[0x00, 0x00, 0x04, 0x04]);
    req.extend_from_slice(&[0x00, 0x00, 0x03, 0x04]);
    req.extend_from_slice(&[0x00, 0x00, 0x02, 0x04]);
    req.extend_from_slice(&[0x00, 0x00, 0x01, 0x04]);
    stream.write_all(&req).await.unwrap();
    // Read 4-byte zero rejection.
    let mut resp = [0u8; 4];
    stream.read_exact(&mut resp).await.unwrap();
    assert_eq!(resp, [0u8; 4]);
    // Server closes the connection on rejection; the spawned task
    // exits + RAII guard fires.
    let _ = stream.read(&mut [0u8; 16]).await;

    tokio::time::sleep(Duration::from_millis(200)).await;
    let text = String::from_utf8(metrics.gather_text().expect("gather")).expect("utf-8");
    assert!(
        text.contains(r#"arcgraph_active_connections{transport="bolt"} 0"#),
        "handshake rejection: gauge must release; text was:\n{text}"
    );

    let _ = shutdown.send(());
}
