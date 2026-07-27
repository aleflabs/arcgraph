//! W15δ Bolt-TLS-wire integration tests — end-to-end TLS over the
//! Bolt 5.0 listener.
//!
//! These tests boot a real Bolt listener on `127.0.0.1:0`, wrap it in
//! a `tokio_rustls::TlsAcceptor` via `BoltServerConfig::with_tls`,
//! then drive an in-tree Bolt 5.0 client over a `tokio_rustls::
//! TlsConnector`. The fixtures are synthesized at runtime via
//! `rcgen` (mirrors the W14α `mcp_http_integ.rs` pattern).
//!
//! Coverage (≥5):
//!   1. End-to-end TLS handshake + Bolt HELLO/RUN session — HELLO +
//!      RUN + GOODBYE works over the wrapped TLS stream. (Sends
//!      HELLO + RUN + GOODBYE; PULL is exercised by the in-process
//!      `tests` module of `server.rs` which carries the full
//!      HELLO/RUN/PULL/GOODBYE pin — the integ harness's role is to
//!      pin the TLS-wrap-up, not duplicate the PULL coverage.)
//!   2. SIGHUP-style cert rotation mid-listener — a NEW connection
//!      after the resolver's `reload()` presents the rotated cert.
//!   3. Plain-TCP rejection for non-loopback bind without TLS — the
//!      `BoltServerConfig::validate()` gate fires at startup so a
//!      misconfigured `0.0.0.0` bind is loud, not silent.
//!   4. W13ε resolver consumption — `serve_bolt_listener` builds a
//!      `tokio_rustls::TlsAcceptor` from the supplied resolver and
//!      the in-flight connection observes the resolver's current
//!      cert (not a stale one).
//!   5. Wrong-cert handshake rejection (W15δ review MED-3) — a
//!      client built against an UNTRUSTED roots store cannot
//!      handshake AND the listener's connection-task drains
//!      cleanly, so a subsequent trusted-client connection
//!      succeeds (no listener crash, no listener-drain deadlock).

#![allow(clippy::expect_used)]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use arcgraph_mcp::transport::bolt::{
    self, BoltServerConfig, ClientMessage, MAGIC_PREAMBLE, PackValue, SERVER_ACCEPT_V5_0,
    StubBoltHandler, decode, encode_client, message::TAG_SUCCESS, read_chunked_message,
    write_chunked_message,
};
use rcgen::{CertificateParams, DnType, KeyPair};
use rustls::pki_types::{CertificateDer, ServerName};
use rustls::{ClientConfig, RootCertStore};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio_rustls::TlsConnector;

// ─────────────────────────────────────────────────────────────────
// rcgen-driven cert fixture (mirrors mcp_http_integ.rs)
// ─────────────────────────────────────────────────────────────────

struct CertFixture {
    _dir: tempfile::TempDir,
    cert_path: std::path::PathBuf,
    key_path: std::path::PathBuf,
    cert_der: CertificateDer<'static>,
}

fn make_cert_fixture(san: &str) -> CertFixture {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut params = CertificateParams::new(vec![san.to_string()]).expect("rcgen params");
    params.not_before = time::OffsetDateTime::now_utc() - time::Duration::days(1);
    params.not_after = time::OffsetDateTime::now_utc() + time::Duration::days(365);
    params.distinguished_name.push(DnType::CommonName, san);
    let kp = KeyPair::generate().expect("rcgen keypair");
    let cert = params.self_signed(&kp).expect("self-signed");
    let cert_path = dir.path().join("server.crt");
    let key_path = dir.path().join("server.key");
    std::fs::write(&cert_path, cert.pem()).expect("write cert");
    std::fs::write(&key_path, kp.serialize_pem()).expect("write key");
    let cert_der = CertificateDer::from(cert.der().as_ref().to_vec());
    CertFixture {
        _dir: dir,
        cert_path,
        key_path,
        cert_der,
    }
}

fn rotate_cert_fixture(fx: &CertFixture, san: &str) -> CertificateDer<'static> {
    let mut params = CertificateParams::new(vec![san.to_string()]).expect("rcgen params");
    params.not_before = time::OffsetDateTime::now_utc() - time::Duration::days(1);
    params.not_after = time::OffsetDateTime::now_utc() + time::Duration::days(365);
    params.distinguished_name.push(DnType::CommonName, san);
    let kp = KeyPair::generate().expect("rcgen keypair");
    let cert = params.self_signed(&kp).expect("self-signed");
    std::fs::write(&fx.cert_path, cert.pem()).expect("write rotated cert");
    std::fs::write(&fx.key_path, kp.serialize_pem()).expect("write rotated key");
    CertificateDer::from(cert.der().as_ref().to_vec())
}

fn build_client_config(trusted_certs: &[CertificateDer<'static>]) -> Arc<ClientConfig> {
    let mut roots = RootCertStore::empty();
    for cert in trusted_certs {
        roots.add(cert.clone()).expect("add root");
    }
    let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
    let cfg = ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .expect("protocol versions")
        .with_root_certificates(roots)
        .with_no_client_auth();
    Arc::new(cfg)
}

// ─────────────────────────────────────────────────────────────────
// Test harness: spawn a TLS-wrapped Bolt listener on 127.0.0.1:0
// ─────────────────────────────────────────────────────────────────

async fn spawn_tls_listener(
    handler: Arc<StubBoltHandler>,
    resolver: Arc<arcgraph_mcp::tls::HotReloadResolver>,
) -> (
    std::net::SocketAddr,
    tokio::sync::oneshot::Sender<()>,
    tokio::task::JoinHandle<()>,
) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    // Build the acceptor inline via the same path serve_bolt_listener
    // would. We bypass serve_bolt_listener so the test owns the
    // listener directly (and can close it without going through a
    // bind-by-string round-trip).
    let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
    let server_config = rustls::ServerConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .expect("protocol versions")
        .with_no_client_auth()
        .with_cert_resolver(resolver.clone());
    let acceptor = tokio_rustls::TlsAcceptor::from(Arc::new(server_config));
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let join = tokio::spawn(async move {
        let shutdown = async move {
            let _ = rx.await;
        };
        let _ = bolt::serve_bolt_inner_with_tls(handler, listener, Some(acceptor), shutdown, None)
            .await;
    });
    (addr, tx, join)
}

async fn tls_handshake(
    addr: std::net::SocketAddr,
    client_cfg: Arc<ClientConfig>,
    server_name: &str,
) -> tokio_rustls::client::TlsStream<TcpStream> {
    let tcp = TcpStream::connect(addr).await.expect("tcp connect");
    let connector = TlsConnector::from(client_cfg);
    let dnsname = ServerName::try_from(server_name.to_owned()).expect("dns name");
    connector
        .connect(dnsname, tcp)
        .await
        .expect("tls handshake")
}

async fn bolt_handshake<S>(stream: &mut S)
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    let mut req = Vec::new();
    req.extend_from_slice(&MAGIC_PREAMBLE);
    req.extend_from_slice(&[0x00, 0x00, 0x00, 0x05]); // Bolt 5.0
    req.extend_from_slice(&[0; 12]);
    stream.write_all(&req).await.unwrap();
    let mut resp = [0u8; 4];
    stream.read_exact(&mut resp).await.unwrap();
    assert_eq!(resp, SERVER_ACCEPT_V5_0);
}

async fn send_msg<S>(stream: &mut S, msg: &ClientMessage)
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    let mut buf = Vec::new();
    encode_client(&mut buf, msg).unwrap();
    write_chunked_message(stream, &buf).await.unwrap();
}

async fn read_reply<S>(stream: &mut S) -> PackValue
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    let payload = read_chunked_message(stream).await.unwrap().unwrap();
    let (val, _) = decode(&payload, 0).unwrap();
    val
}

// ─────────────────────────────────────────────────────────────────
// Test 1: end-to-end TLS handshake + Bolt HELLO/RUN session
//
// Renamed from `tls_handshake_then_full_bolt_session_succeeds` per
// the W15δ review LOW-5 finding: the test sends HELLO + RUN +
// GOODBYE but not PULL; PULL is exercised by `server.rs`'s in-process
// `tests::full_hello_run_pull_session_returns_record_and_success` —
// duplicating PULL here would only repeat the codec path. The
// load-bearing pin in this integ test is the **TLS-wrap-up**.
// ─────────────────────────────────────────────────────────────────

#[tokio::test]
async fn tls_handshake_then_hello_run_session_succeeds() {
    let fx = make_cert_fixture("localhost");
    let provider = Arc::new(arcgraph_mcp::tls::FileSystemCertProvider::new(
        fx.cert_path.clone(),
        fx.key_path.clone(),
        None,
    ));
    let resolver = arcgraph_mcp::tls::HotReloadResolver::new(provider).expect("resolver init");
    let handler = Arc::new(StubBoltHandler::accepting());
    let (addr, shutdown, _join) = spawn_tls_listener(handler, resolver).await;

    let client_cfg = build_client_config(std::slice::from_ref(&fx.cert_der));
    let mut tls = tokio::time::timeout(
        Duration::from_secs(2),
        tls_handshake(addr, client_cfg, "localhost"),
    )
    .await
    .expect("tls handshake timed out");

    bolt_handshake(&mut tls).await;
    send_msg(
        &mut tls,
        &ClientMessage::Hello {
            user_agent: Some("arcgraph-tls-test/1.0".into()),
            scheme: Some("none".into()),
            principal: None,
            credentials: None,
            routing: None,
            extras: BTreeMap::new(),
        },
    )
    .await;
    let reply = read_reply(&mut tls).await;
    match reply {
        PackValue::Struct { tag, .. } => {
            assert_eq!(tag, TAG_SUCCESS, "HELLO over TLS must SUCCESS");
        }
        other => panic!("expected struct, got {other:?}"),
    }
    send_msg(
        &mut tls,
        &ClientMessage::Run {
            query: "RETURN 1".into(),
            parameters: BTreeMap::new(),
            extra: BTreeMap::new(),
        },
    )
    .await;
    let reply = read_reply(&mut tls).await;
    match reply {
        PackValue::Struct { tag, .. } => assert_eq!(tag, TAG_SUCCESS, "RUN over TLS must SUCCESS"),
        other => panic!("expected RUN SUCCESS, got {other:?}"),
    }
    send_msg(&mut tls, &ClientMessage::Goodbye).await;
    drop(tls);
    let _ = shutdown.send(());
}

// ─────────────────────────────────────────────────────────────────
// Test 2: SIGHUP-style cert rotation mid-listener
// ─────────────────────────────────────────────────────────────────

#[tokio::test]
async fn cert_rotation_observed_by_new_connections() {
    let fx = make_cert_fixture("localhost");
    let provider = Arc::new(arcgraph_mcp::tls::FileSystemCertProvider::new(
        fx.cert_path.clone(),
        fx.key_path.clone(),
        None,
    ));
    let resolver = arcgraph_mcp::tls::HotReloadResolver::new(provider).expect("resolver init");
    let resolver_for_reload = resolver.clone();
    let handler = Arc::new(StubBoltHandler::accepting());
    let initial_cert = fx.cert_der.clone();
    let (addr, shutdown, _join) = spawn_tls_listener(handler, resolver).await;

    // Connection #1 trusts the initial cert.
    let client1_cfg = build_client_config(std::slice::from_ref(&initial_cert));
    let mut tls1 = tokio::time::timeout(
        Duration::from_secs(2),
        tls_handshake(addr, client1_cfg, "localhost"),
    )
    .await
    .expect("first tls handshake timed out");
    bolt_handshake(&mut tls1).await;
    drop(tls1);

    // Rotate: stage a new cert + key on disk, fire resolver.reload();
    // a fresh connection that trusts ONLY the rotated cert must
    // succeed.
    let rotated_cert = rotate_cert_fixture(&fx, "localhost");
    resolver_for_reload.reload().expect("reload after rotation");

    let client2_cfg = build_client_config(&[rotated_cert]);
    let mut tls2 = tokio::time::timeout(
        Duration::from_secs(2),
        tls_handshake(addr, client2_cfg, "localhost"),
    )
    .await
    .expect("post-rotation tls handshake timed out");
    bolt_handshake(&mut tls2).await;
    drop(tls2);

    let _ = shutdown.send(());
}

// ─────────────────────────────────────────────────────────────────
// Test 3: plain-TCP rejection for non-loopback bind without TLS
// ─────────────────────────────────────────────────────────────────

#[tokio::test]
async fn plain_tcp_rejected_for_non_loopback_bind() {
    let cfg = BoltServerConfig {
        bind: "0.0.0.0:7687".parse().unwrap(),
        allow_remote_bind: true, // explicit opt-in
        ..Default::default()
    };
    // No TLS configured + non-loopback bind → must reject.
    let err = cfg
        .validate()
        .expect_err("non-loopback plain-TCP must reject");
    let msg = format!("{err}");
    assert!(
        msg.contains("requires TLS"),
        "validate() must cite the TLS requirement: {msg}"
    );

    // Without allow_remote_bind, the same config is rejected on a
    // different ground (allow_remote_bind precedence).
    let cfg_no_allow = BoltServerConfig {
        bind: "0.0.0.0:7687".parse().unwrap(),
        ..Default::default()
    };
    let err = cfg_no_allow
        .validate()
        .expect_err("non-loopback without allow_remote_bind must reject");
    assert!(format!("{err}").contains("allow_remote_bind"));

    // Loopback plain TCP is accepted for backwards compatibility.
    let cfg_loopback = BoltServerConfig::default();
    cfg_loopback
        .validate()
        .expect("loopback default plain-TCP is accepted");

    // Non-loopback + allow_remote + TLS configured → accepted.
    let fx = make_cert_fixture("localhost");
    let provider = Arc::new(arcgraph_mcp::tls::FileSystemCertProvider::new(
        fx.cert_path,
        fx.key_path,
        None,
    ));
    let resolver = arcgraph_mcp::tls::HotReloadResolver::new(provider).expect("resolver init");
    let cfg_tls_remote = BoltServerConfig {
        bind: "0.0.0.0:7687".parse().unwrap(),
        allow_remote_bind: true,
        ..Default::default()
    }
    .with_tls(resolver);
    cfg_tls_remote
        .validate()
        .expect("non-loopback with TLS + allow_remote_bind is accepted");
}

// ─────────────────────────────────────────────────────────────────
// Test 4: W13ε resolver consumption — listener observes the
// resolver's current cert (not a stale one)
// ─────────────────────────────────────────────────────────────────

#[tokio::test]
async fn w13e_resolver_consumed_by_acceptor() {
    // Build the resolver with cert A; rotate to cert B BEFORE the
    // first connection. The connection's peer cert chain must be cert
    // B (not the resolver-init-time cert A). This proves the acceptor
    // walks the resolver per-handshake (not per-listener-build), so
    // the W13ε hot-reload semantic actually surfaces at the Bolt
    // transport.
    let fx = make_cert_fixture("localhost");
    let provider = Arc::new(arcgraph_mcp::tls::FileSystemCertProvider::new(
        fx.cert_path.clone(),
        fx.key_path.clone(),
        None,
    ));
    let resolver = arcgraph_mcp::tls::HotReloadResolver::new(provider).expect("resolver init");
    let resolver_for_reload = resolver.clone();
    let handler = Arc::new(StubBoltHandler::accepting());
    let (addr, shutdown, _join) = spawn_tls_listener(handler, resolver).await;

    // Rotate BEFORE any connection. The acceptor was built from
    // `resolver`, but the Arc<HotReloadResolver> is what it actually
    // consults at handshake time.
    let rotated_cert = rotate_cert_fixture(&fx, "localhost");
    resolver_for_reload.reload().expect("reload");

    // Trust ONLY the rotated cert — connection must succeed.
    let client_cfg = build_client_config(&[rotated_cert]);
    let mut tls = tokio::time::timeout(
        Duration::from_secs(2),
        tls_handshake(addr, client_cfg, "localhost"),
    )
    .await
    .expect("tls handshake timed out — resolver not consumed at handshake time");
    bolt_handshake(&mut tls).await;
    drop(tls);

    let _ = shutdown.send(());
}

// ─────────────────────────────────────────────────────────────────
// Test 5: wrong-cert handshake rejection does NOT crash the listener
//
// W15δ review MED-3: the four prior tests cover handshake-success
// + rotation-success + validate-rejection (config-time, not
// handshake-time) + resolver-consumption. The handshake-time
// rejection path (`server.rs:430` "TLS handshake failed" branch)
// was structurally unreachable from the integ harness — a
// regression that made `acc.accept` panic on certain malformed
// ClientHello variants would silently break the listener-drain
// without any test catching it.
//
// This test builds a client whose root store is EMPTY (trusts
// nothing), attempts the handshake, asserts the client errors,
// AND THEN runs a SECOND connection from a properly-trusting
// client over the SAME listener to prove the listener-drain
// logic survived the failed handshake. If the failed handshake
// pinned the listener, the second connection would time out.
// ─────────────────────────────────────────────────────────────────

#[tokio::test]
async fn tls_handshake_rejected_for_untrusted_client_does_not_crash_listener() {
    let fx = make_cert_fixture("localhost");
    let provider = Arc::new(arcgraph_mcp::tls::FileSystemCertProvider::new(
        fx.cert_path.clone(),
        fx.key_path.clone(),
        None,
    ));
    let resolver = arcgraph_mcp::tls::HotReloadResolver::new(provider).expect("resolver init");
    let handler = Arc::new(StubBoltHandler::accepting());
    let (addr, shutdown, _join) = spawn_tls_listener(handler, resolver).await;

    // Untrusted client: empty root store; the rotated/issued cert
    // is NOT in the trust set, so the handshake MUST reject at the
    // client side.
    let untrusted_cfg = build_client_config(&[]);
    let bad = tokio::time::timeout(Duration::from_secs(2), async {
        let tcp = TcpStream::connect(addr).await.expect("tcp connect");
        let connector = TlsConnector::from(untrusted_cfg);
        let dnsname = ServerName::try_from("localhost".to_owned()).expect("dns name");
        connector.connect(dnsname, tcp).await
    })
    .await
    .expect("untrusted handshake should fast-reject, not stall the 2s budget");
    assert!(
        bad.is_err(),
        "untrusted-root client expected to ERROR on tls handshake; got Ok"
    );

    // Recovery check: a properly-trusting client should still be
    // able to handshake + transact. If the failed handshake had
    // pinned the listener-drain (or panicked the accept loop), the
    // listener wouldn't accept this second connection.
    let trusted_cfg = build_client_config(std::slice::from_ref(&fx.cert_der));
    let mut tls = tokio::time::timeout(
        Duration::from_secs(2),
        tls_handshake(addr, trusted_cfg, "localhost"),
    )
    .await
    .expect("trusted second connection timed out — listener did not drain failed handshake");
    bolt_handshake(&mut tls).await;
    send_msg(
        &mut tls,
        &ClientMessage::Hello {
            user_agent: Some("arcgraph-tls-test/1.0".into()),
            scheme: Some("none".into()),
            principal: None,
            credentials: None,
            routing: None,
            extras: BTreeMap::new(),
        },
    )
    .await;
    let reply = read_reply(&mut tls).await;
    match reply {
        PackValue::Struct { tag, .. } => assert_eq!(
            tag, TAG_SUCCESS,
            "post-rejection trusted HELLO must SUCCESS — listener survived"
        ),
        other => panic!("expected SUCCESS struct, got {other:?}"),
    }
    drop(tls);
    let _ = shutdown.send(());
}
