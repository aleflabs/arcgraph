//! W20β-1 — Bolt mTLS smoke test.
//!
//! Mirrors `mtls_smoke.rs` for the HTTP transport. Spawns a real
//! `serve_bolt_inner_with_tls` listener configured with mTLS, drives a
//! `tokio_rustls::TlsConnector` with three postures:
//!
//! 1. **Valid client cert** (signed by the trusted CA): handshake
//!    completes, the Bolt HELLO succeeds, RUN returns SUCCESS.
//! 2. **Untrusted client cert** (signed by a DIFFERENT CA): handshake
//!    REJECTS at chain-verify or the connection drops without
//!    completing the Bolt session.
//! 3. **Missing client cert when required**: handshake REJECTS with the
//!    server's bad-certificate alert (`required = true` posture).
//!
//! The fault-injection regression discipline (per
//! `feedback_load_bearing_pr_requires_fault_injection_tests.md`) is the
//! reason for case 2's existence — the W19γ adversarial-test posture
//! extended to mTLS at the Bolt transport.

#![allow(clippy::expect_used)]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use arcgraph_mcp::transport::bolt::{
    self, BoltServerConfig, ClientMessage, MAGIC_PREAMBLE, PackValue, SERVER_ACCEPT_V5_0,
    StubBoltHandler, decode, encode_client, message::TAG_SUCCESS, read_chunked_message,
    write_chunked_message,
};
use rcgen::{
    BasicConstraints, CertificateParams, DnType, ExtendedKeyUsagePurpose, IsCa, Issuer, KeyPair,
    KeyUsagePurpose,
};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName, pem::PemObject};
use rustls::{ClientConfig, RootCertStore};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio_rustls::TlsConnector;

// ─────────────────────────────────────────────────────────────────────
// CA + cert chain fixture
// ─────────────────────────────────────────────────────────────────────

struct CaFixture {
    ca_cert_der: CertificateDer<'static>,
    ca_cert_pem: Vec<u8>,
    ca_kp: KeyPair,
    ca_params: CertificateParams,
}

fn make_ca(common_name: &str) -> CaFixture {
    let mut params = CertificateParams::new(vec![common_name.to_string()]).expect("rcgen params");
    params.not_before = time::OffsetDateTime::now_utc() - time::Duration::days(1);
    params.not_after = time::OffsetDateTime::now_utc() + time::Duration::days(365);
    params
        .distinguished_name
        .push(DnType::CommonName, common_name);
    params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    params.key_usages = vec![
        KeyUsagePurpose::KeyCertSign,
        KeyUsagePurpose::DigitalSignature,
        KeyUsagePurpose::CrlSign,
    ];
    let kp = KeyPair::generate().expect("ca kp");
    let cert = params.clone().self_signed(&kp).expect("ca self-sign");
    let der = cert.der().clone();
    let pem = cert.pem().into_bytes();
    CaFixture {
        ca_cert_der: CertificateDer::from(der.as_ref().to_vec()),
        ca_cert_pem: pem,
        ca_kp: kp,
        ca_params: params,
    }
}

struct LeafCert {
    cert_der: CertificateDer<'static>,
    key_der: PrivateKeyDer<'static>,
    cert_pem: String,
    key_pem: String,
}

fn make_leaf_signed_by(ca: &CaFixture, cn: &str, sans: &[&str], server_eku: bool) -> LeafCert {
    make_leaf_signed_by_with_window(
        ca,
        cn,
        sans,
        server_eku,
        time::OffsetDateTime::now_utc() - time::Duration::days(1),
        time::OffsetDateTime::now_utc() + time::Duration::days(180),
    )
}

fn make_leaf_signed_by_with_window(
    ca: &CaFixture,
    cn: &str,
    sans: &[&str],
    server_eku: bool,
    not_before: time::OffsetDateTime,
    not_after: time::OffsetDateTime,
) -> LeafCert {
    let san_vec: Vec<String> = sans.iter().map(|s| (*s).to_string()).collect();
    let mut params = CertificateParams::new(san_vec).expect("leaf params");
    params.not_before = not_before;
    params.not_after = not_after;
    params.distinguished_name.push(DnType::CommonName, cn);
    params.extended_key_usages = if server_eku {
        vec![ExtendedKeyUsagePurpose::ServerAuth]
    } else {
        vec![ExtendedKeyUsagePurpose::ClientAuth]
    };
    let kp = KeyPair::generate().expect("leaf kp");
    let issuer = Issuer::new(ca.ca_params.clone(), &ca.ca_kp);
    let cert = params.signed_by(&kp, &issuer).expect("sign leaf");
    let cert_pem = cert.pem();
    let cert_der = cert.der().clone();
    let key_pem = kp.serialize_pem();
    let key_der = PrivateKeyDer::from_pem_slice(key_pem.as_bytes()).expect("key DER");
    LeafCert {
        cert_der: CertificateDer::from(cert_der.as_ref().to_vec()),
        key_der,
        cert_pem,
        key_pem,
    }
}

fn stage_server_cert(
    ca: &CaFixture,
) -> (
    tempfile::TempDir,
    std::path::PathBuf,
    std::path::PathBuf,
    CertificateDer<'static>,
) {
    let leaf = make_leaf_signed_by(ca, "localhost", &["localhost"], true);
    let dir = tempfile::tempdir().expect("tempdir");
    let cert_path = dir.path().join("server.crt");
    let key_path = dir.path().join("server.key");
    let bundled = format!(
        "{}{}",
        leaf.cert_pem,
        std::str::from_utf8(&ca.ca_cert_pem).expect("ca pem utf-8")
    );
    std::fs::write(&cert_path, bundled).expect("write cert");
    std::fs::write(&key_path, leaf.key_pem).expect("write key");
    (dir, cert_path, key_path, leaf.cert_der)
}

fn build_client_no_auth(trusted_server_roots: &[CertificateDer<'static>]) -> Arc<ClientConfig> {
    let mut roots = RootCertStore::empty();
    for c in trusted_server_roots {
        roots.add(c.clone()).expect("add server root");
    }
    let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
    let cfg = ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .expect("protocol versions")
        .with_root_certificates(roots)
        .with_no_client_auth();
    Arc::new(cfg)
}

fn build_client_with_cert(
    trusted_server_roots: &[CertificateDer<'static>],
    client_cert: CertificateDer<'static>,
    client_key: PrivateKeyDer<'static>,
) -> Arc<ClientConfig> {
    let mut roots = RootCertStore::empty();
    for c in trusted_server_roots {
        roots.add(c.clone()).expect("add server root");
    }
    let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
    let cfg = ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .expect("protocol versions")
        .with_root_certificates(roots)
        .with_client_auth_cert(vec![client_cert], client_key)
        .expect("client auth cert");
    Arc::new(cfg)
}

// ─────────────────────────────────────────────────────────────────────
// Server harness — wraps the Bolt listener with mTLS.
// ─────────────────────────────────────────────────────────────────────

async fn spawn_mtls_bolt_listener(
    handler: Arc<StubBoltHandler>,
    cert_path: std::path::PathBuf,
    key_path: std::path::PathBuf,
    client_ca_pem: &[u8],
    client_cert_required: bool,
) -> (
    std::net::SocketAddr,
    tokio::sync::oneshot::Sender<()>,
    tokio::task::JoinHandle<()>,
) {
    let provider = Arc::new(arcgraph_mcp::tls::FileSystemCertProvider::new(
        cert_path, key_path, None,
    ));
    let resolver = arcgraph_mcp::tls::HotReloadResolver::new(provider).expect("resolver init");
    let client_verifier =
        arcgraph_mcp::tls::client_verifier_from_ca_pem(client_ca_pem, client_cert_required)
            .expect("client verifier");
    // Build the TLS acceptor directly so the test owns the listener.
    let crypto_provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
    let server_config = rustls::ServerConfig::builder_with_provider(crypto_provider)
        .with_safe_default_protocol_versions()
        .expect("protocol versions")
        .with_client_cert_verifier(client_verifier)
        .with_cert_resolver(resolver);
    let acceptor = tokio_rustls::TlsAcceptor::from(Arc::new(server_config));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
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

async fn tls_handshake_attempt(
    addr: std::net::SocketAddr,
    client_cfg: Arc<ClientConfig>,
    server_name: &str,
) -> Result<tokio_rustls::client::TlsStream<TcpStream>, String> {
    let tcp = TcpStream::connect(addr)
        .await
        .map_err(|e| format!("tcp: {e}"))?;
    let connector = TlsConnector::from(client_cfg);
    let dnsname = ServerName::try_from(server_name.to_owned()).map_err(|e| format!("dns: {e}"))?;
    connector
        .connect(dnsname, tcp)
        .await
        .map_err(|e| format!("handshake: {e}"))
}

async fn bolt_handshake_and_hello<S>(stream: &mut S) -> Result<(), String>
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    let mut req = Vec::new();
    req.extend_from_slice(&MAGIC_PREAMBLE);
    req.extend_from_slice(&[0x00, 0x00, 0x00, 0x05]);
    req.extend_from_slice(&[0; 12]);
    stream
        .write_all(&req)
        .await
        .map_err(|e| format!("write magic: {e}"))?;
    let mut resp = [0u8; 4];
    stream
        .read_exact(&mut resp)
        .await
        .map_err(|e| format!("read accept: {e}"))?;
    if resp != SERVER_ACCEPT_V5_0 {
        return Err(format!("unexpected bolt accept: {resp:?}"));
    }
    let hello = ClientMessage::Hello {
        user_agent: Some("arcgraph-mtls-test/1.0".into()),
        scheme: Some("none".into()),
        principal: None,
        credentials: None,
        routing: None,
        extras: BTreeMap::new(),
    };
    let mut buf = Vec::new();
    encode_client(&mut buf, &hello).map_err(|e| format!("encode hello: {e}"))?;
    write_chunked_message(stream, &buf)
        .await
        .map_err(|e| format!("write hello: {e}"))?;
    let payload = read_chunked_message(stream)
        .await
        .map_err(|e| format!("read hello reply: {e}"))?
        .ok_or("no hello reply")?;
    let (val, _) = decode(&payload, 0).map_err(|e| format!("decode reply: {e}"))?;
    match val {
        PackValue::Struct { tag, .. } if tag == TAG_SUCCESS => Ok(()),
        other => Err(format!("hello got non-success: {other:?}")),
    }
}

// ─────────────────────────────────────────────────────────────────────
// Test 1 — VALID client cert: handshake + HELLO succeed.
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn bolt_mtls_smoke_valid_client_cert_succeeds() {
    let ca = make_ca("ArcGraph Bolt CA");
    let (_dir, cert_path, key_path, server_cert_der) = stage_server_cert(&ca);
    let client = make_leaf_signed_by(&ca, "client-alice", &["client-alice.arcgraph.local"], false);

    let handler = Arc::new(StubBoltHandler::accepting());
    let (addr, shutdown, _join) = spawn_mtls_bolt_listener(
        handler,
        cert_path,
        key_path,
        &ca.ca_cert_pem,
        true, // required
    )
    .await;

    let client_cfg = build_client_with_cert(
        &[server_cert_der, ca.ca_cert_der.clone()],
        client.cert_der.clone(),
        client.key_der.clone_key(),
    );
    let mut tls = tokio::time::timeout(
        Duration::from_secs(2),
        tls_handshake_attempt(addr, client_cfg, "localhost"),
    )
    .await
    .expect("timeout")
    .expect("valid mTLS handshake should succeed");

    bolt_handshake_and_hello(&mut tls)
        .await
        .expect("HELLO over mTLS should succeed");
    drop(tls);
    let _ = shutdown.send(());
}

// ─────────────────────────────────────────────────────────────────────
// Test 2 — UNTRUSTED client cert (different CA) MUST be rejected.
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn bolt_mtls_smoke_untrusted_client_cert_rejects() {
    let server_ca = make_ca("ArcGraph Bolt CA");
    let attacker_ca = make_ca("Attacker CA");
    let (_dir, cert_path, key_path, server_cert_der) = stage_server_cert(&server_ca);
    let bad_client = make_leaf_signed_by(&attacker_ca, "rogue", &["rogue.attacker.example"], false);

    let handler = Arc::new(StubBoltHandler::accepting());
    let (addr, shutdown, _join) =
        spawn_mtls_bolt_listener(handler, cert_path, key_path, &server_ca.ca_cert_pem, true).await;

    let client_cfg = build_client_with_cert(
        &[server_cert_der, server_ca.ca_cert_der.clone()],
        bad_client.cert_der.clone(),
        bad_client.key_der.clone_key(),
    );

    let outcome = tokio::time::timeout(Duration::from_secs(3), async move {
        // Either the TLS handshake fails OUTRIGHT, or the handshake
        // completes (TLS 1.3 timing edge) and the subsequent Bolt
        // HELLO fails because the server dropped the connection
        // post-handshake. Both ARE rejection — the canonical proof
        // is "no successful HELLO over the bad cert."
        match tls_handshake_attempt(addr, client_cfg, "localhost").await {
            Err(_) => Ok::<_, String>(()), // handshake rejected outright
            Ok(mut tls) => match bolt_handshake_and_hello(&mut tls).await {
                Err(_) => Ok(()), // post-handshake drop
                Ok(()) => Err("BAD: untrusted client cert successfully HELLOed".into()),
            },
        }
    })
    .await
    .expect("timeout");

    outcome.expect("untrusted cert MUST be rejected somewhere in the cert/HELLO chain");
    let _ = shutdown.send(());
}

// ─────────────────────────────────────────────────────────────────────
// Test 3 — MISSING client cert (no-auth client config) MUST be rejected
// when the server's verifier is in `client_cert_required = true` mode.
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn bolt_mtls_smoke_missing_client_cert_when_required_rejects() {
    let ca = make_ca("ArcGraph Bolt CA");
    let (_dir, cert_path, key_path, server_cert_der) = stage_server_cert(&ca);

    let handler = Arc::new(StubBoltHandler::accepting());
    let (addr, shutdown, _join) = spawn_mtls_bolt_listener(
        handler,
        cert_path,
        key_path,
        &ca.ca_cert_pem,
        true, // required — no-cert clients MUST reject
    )
    .await;

    let client_cfg = build_client_no_auth(&[server_cert_der, ca.ca_cert_der.clone()]);

    let outcome = tokio::time::timeout(Duration::from_secs(3), async move {
        match tls_handshake_attempt(addr, client_cfg, "localhost").await {
            Err(_) => Ok::<_, String>(()),
            Ok(mut tls) => match bolt_handshake_and_hello(&mut tls).await {
                Err(_) => Ok(()),
                Ok(()) => Err("BAD: no-cert client successfully HELLOed".into()),
            },
        }
    })
    .await
    .expect("timeout");
    outcome.expect("no-cert client MUST be rejected when required");
    let _ = shutdown.send(());
}

// ─────────────────────────────────────────────────────────────────────
// Test 4 — EXPIRED client cert (not_after in the past) MUST be rejected.
//
// Per R1 M-3 + `feedback_load_bearing_pr_requires_fault_injection_tests.md`:
// each documented failure mode of mTLS chain-verify gets a per-failure-
// mode regression test. Same shape as the HTTP-side
// `mtls_smoke_expired_client_cert_rejects` — the cert is signed by the
// trusted CA (trust-store membership passes) but `not_after` is one day
// in the past, so rustls's WebPkiClientVerifier MUST reject at the
// validity-window check before the Bolt HELLO completes.
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn bolt_mtls_smoke_expired_client_cert_rejects() {
    let ca = make_ca("ArcGraph Bolt CA");
    let (_dir, cert_path, key_path, server_cert_der) = stage_server_cert(&ca);
    // Signed by the trusted CA, but with `not_after` in the past.
    let expired_client = make_leaf_signed_by_with_window(
        &ca,
        "client-expired",
        &["client-expired.arcgraph.local"],
        false,
        time::OffsetDateTime::now_utc() - time::Duration::days(30),
        time::OffsetDateTime::now_utc() - time::Duration::days(1),
    );

    let handler = Arc::new(StubBoltHandler::accepting());
    let (addr, shutdown, _join) =
        spawn_mtls_bolt_listener(handler, cert_path, key_path, &ca.ca_cert_pem, true).await;

    let client_cfg = build_client_with_cert(
        &[server_cert_der, ca.ca_cert_der.clone()],
        expired_client.cert_der.clone(),
        expired_client.key_der.clone_key(),
    );

    let outcome = tokio::time::timeout(Duration::from_secs(3), async move {
        // Same assertion discipline as the untrusted-CA case (Test 2):
        // EITHER (a) the TLS handshake fails outright on the validity-
        // window check, OR (b) the handshake completes (TLS 1.3 timing
        // edge) and the post-handshake Bolt HELLO fails because the
        // server dropped the connection. Both ARE rejection; the
        // canonical proof is "no successful HELLO over the expired
        // cert."
        match tls_handshake_attempt(addr, client_cfg, "localhost").await {
            Err(_) => Ok::<_, String>(()),
            Ok(mut tls) => match bolt_handshake_and_hello(&mut tls).await {
                Err(_) => Ok(()),
                Ok(()) => Err("BAD: expired client cert successfully HELLOed".into()),
            },
        }
    })
    .await
    .expect("timeout");

    outcome.expect("expired cert MUST be rejected somewhere in the cert/HELLO chain");
    let _ = shutdown.send(());
}

// ─────────────────────────────────────────────────────────────────────
// Test 5 — BoltServerConfig validates that mTLS + plain-TCP is rejected
// at startup (mTLS without server-TLS is incoherent — W20β-1 invariant).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn bolt_server_config_rejects_mtls_without_server_tls() {
    let ca = make_ca("ArcGraph Bolt CA");
    let cfg = BoltServerConfig::default()
        .with_client_ca_pem(&ca.ca_cert_pem, true)
        .expect("with_client_ca_pem");
    let err = cfg.validate().expect_err("mTLS w/o TLS must reject");
    let msg = format!("{err}");
    assert!(
        msg.contains("client_verifier") || msg.contains("with_tls") || msg.contains("server-TLS"),
        "expected mTLS-without-TLS rejection text, got: {msg}",
    );
}
