//! W16 M6-10 (#310) — subprocess integration tests for the
//! `arcgraph health` subcommand.
//!
//! Each test spins up a tiny tokio TCP listener bound to an
//! ephemeral loopback port, hard-codes its scripted HTTP response,
//! and then spawns the `arcgraph` binary to probe it. Exit code +
//! stderr line are the assertion surface — same shape as a Docker
//! `HEALTHCHECK` consumer would see.
//!
//! These tests intentionally use `tokio::process::Command` instead
//! of `std::process::Command` so the binary spawn doesn't block the
//! test runtime's worker thread when the binary itself owns a
//! tokio runtime.

use std::net::SocketAddr;
use std::process::Stdio;
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::process::Command;
use tokio::time::timeout;

const SUBPROCESS_TIMEOUT: Duration = Duration::from_secs(15);

/// Spawn a one-shot HTTP stub that drains the request bytes (up to
/// `\r\n\r\n`) and then writes `response` back. Each accepted
/// connection serves exactly one canned response and then closes.
async fn start_stub_server(response: &'static [u8]) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral");
    let addr = listener.local_addr().expect("local_addr");
    tokio::spawn(async move {
        loop {
            let (mut stream, _peer) = match listener.accept().await {
                Ok(p) => p,
                Err(_) => return,
            };
            tokio::spawn(async move {
                // Drain enough of the request that the client's
                // `Connection: close` plus our reply isn't racing the
                // peer's send buffer. A 1KiB read covers any minimal
                // GET line the probe will write.
                let mut buf = [0u8; 1024];
                let _ = stream.read(&mut buf).await;
                let _ = stream.write_all(response).await;
                let _ = stream.flush().await;
                let _ = stream.shutdown().await;
            });
        }
    });
    addr
}

/// Pick a loopback `127.0.0.1:<port>` that is *unbound* — we open a
/// listener, take its port, drop the listener, and return the addr.
/// The kernel may reassign the port immediately, so callers should
/// only use this where a refused connection is the expected outcome.
async fn unbound_loopback_addr() -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral for unbound");
    let addr = listener.local_addr().expect("local_addr");
    drop(listener);
    addr
}

#[tokio::test]
async fn health_subcommand_exits_0_when_server_returns_200() {
    let addr = start_stub_server(
        b"HTTP/1.1 200 OK\r\n\
          Content-Type: application/json\r\n\
          Content-Length: 15\r\n\
          \r\n\
          {\"status\":\"ok\"}",
    )
    .await;

    let bin = env!("CARGO_BIN_EXE_arcgraph");
    let url = format!("http://{}/healthz", addr);
    let out = timeout(
        SUBPROCESS_TIMEOUT,
        Command::new(bin)
            .args(["health", "--addr", &url])
            .stdin(Stdio::null())
            .output(),
    )
    .await
    .expect("subprocess finished in time")
    .expect("spawn arcgraph health");

    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "expected exit 0; got {:?}\nstderr: {stderr}\nstdout: {stdout}",
        out.status
    );
}

#[tokio::test]
async fn health_subcommand_exits_1_when_server_returns_503() {
    let addr = start_stub_server(
        b"HTTP/1.1 503 Service Unavailable\r\n\
          Content-Type: text/plain\r\n\
          Content-Length: 11\r\n\
          \r\n\
          not-ready\r\n",
    )
    .await;

    let bin = env!("CARGO_BIN_EXE_arcgraph");
    let url = format!("http://{}/healthz", addr);
    let out = timeout(
        SUBPROCESS_TIMEOUT,
        Command::new(bin)
            .args(["health", "--addr", &url])
            .stdin(Stdio::null())
            .output(),
    )
    .await
    .expect("subprocess finished in time")
    .expect("spawn arcgraph health");

    assert!(
        !out.status.success(),
        "expected non-zero exit for 503; got {:?}",
        out.status
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("arcgraph health: HTTP 503"),
        "stderr should report the 503: {stderr}"
    );
}

#[tokio::test]
async fn health_subcommand_exits_1_when_server_unreachable() {
    // No listener — the connect should refuse (or, in the rare race
    // where the port is reassigned, the read will time out). Either
    // way `arcgraph health` exits non-zero with a stderr message.
    let addr = unbound_loopback_addr().await;
    let bin = env!("CARGO_BIN_EXE_arcgraph");
    let url = format!("http://{}/healthz", addr);

    let out = timeout(
        SUBPROCESS_TIMEOUT,
        Command::new(bin)
            .args(["health", "--addr", &url, "--timeout-ms", "1000"])
            .stdin(Stdio::null())
            .output(),
    )
    .await
    .expect("subprocess finished in time")
    .expect("spawn arcgraph health");

    assert!(
        !out.status.success(),
        "expected non-zero exit; got {:?}",
        out.status
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.starts_with("arcgraph health:"),
        "stderr should be prefixed: {stderr}"
    );
}

#[tokio::test]
async fn health_subcommand_respects_addr_flag() {
    // Two stubs on different ports — the probe must hit the one we
    // pass via --addr. We make stub B's response a 503 so a "wrong-
    // server-hit" would fail the assertion (`success()`).
    let healthy = start_stub_server(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n").await;
    let _unhealthy =
        start_stub_server(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n").await;

    let bin = env!("CARGO_BIN_EXE_arcgraph");
    let url = format!("http://{}/healthz", healthy);
    let out = timeout(
        SUBPROCESS_TIMEOUT,
        Command::new(bin)
            .args(["health", "--addr", &url])
            .stdin(Stdio::null())
            .output(),
    )
    .await
    .expect("subprocess finished in time")
    .expect("spawn arcgraph health");

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "expected exit 0 hitting the healthy stub; got {:?}\nstderr: {stderr}",
        out.status
    );
}

#[tokio::test]
async fn health_subcommand_rejects_https_scheme_with_forward_bind_message() {
    // The probe must not silently fall back to plaintext if the
    // operator points it at https://. v1.0-α surfaces a forward-bind
    // note pointing at M6-08+.
    let bin = env!("CARGO_BIN_EXE_arcgraph");
    let out = timeout(
        SUBPROCESS_TIMEOUT,
        Command::new(bin)
            .args(["health", "--addr", "https://127.0.0.1:8443/healthz"])
            .stdin(Stdio::null())
            .output(),
    )
    .await
    .expect("subprocess finished in time")
    .expect("spawn arcgraph health");

    assert!(
        !out.status.success(),
        "https:// must reject; got {:?}",
        out.status
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("M6-08+"),
        "forward-bind cite missing: {stderr}"
    );
}
