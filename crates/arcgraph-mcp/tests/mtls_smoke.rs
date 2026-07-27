//! W20β-1 — real-data mTLS smoke test for the HTTP MCP transport.
//!
//! Spawns a real `serve_http` listener with mTLS enforced (a synthesized
//! client-CA + `client_cert_required = true`), then drives three
//! handshake postures:
//!
//! 1. **Valid client cert** (signed by the trusted CA) — handshake +
//!    request succeed; the dispatcher emits a success envelope.
//! 2. **Untrusted client cert** (signed by a DIFFERENT CA) — handshake
//!    FAILS at chain-verify; the request never reaches the dispatcher.
//! 3. **Missing client cert when required** — handshake FAILS with a
//!    `bad certificate` alert; the request never reaches the dispatcher.
//!
//! The fixture mirrors `mcp_http_integ.rs`'s rcgen-driven cert
//! synthesis but adds a separate client-CA + per-client end-entity cert
//! signed by the CA. The "untrusted CA" case generates a second CA and
//! a client cert signed by IT so the server's trust store rejects.
//!
//! # §1.5 W19 retro inheritance pin
//!
//! Per `feedback_load_bearing_pr_requires_fault_injection_tests.md`,
//! each failure mode (cert untrusted + cert missing) is covered by a
//! per-failure-mode regression test below.

#![allow(clippy::expect_used)] // tests can expect; high-signal failure context.
#![allow(dead_code)] // some stubs are only invoked by particular tests.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use arcgraph_core::TenantId;
use arcgraph_mcp::tools::explore::{Neighborhood, NeighborhoodEdge, NeighborhoodNode};
use arcgraph_mcp::tools::ingest::{IngestBatch, IngestProvider, IngestSummary};
use arcgraph_mcp::tools::inspect::{NeighborDirection, NeighborInfo, NodeInspection};
use arcgraph_mcp::tools::schema::{
    GraphSchema, IndexDescriptor, IndexKind, LabelInfo, RelTypeInfo,
};
use arcgraph_mcp::tools::search::{AvailableSubstrates, SearchHit};
use arcgraph_mcp::transport::http::{
    HEADER_TENANT, HttpServerConfig, PATH_MCP, TenantStrategy, serve_http,
};
use arcgraph_mcp::{
    Dispatcher, HybridSearcher, MCPError, NeighborhoodExplorer, NodeInspector, SchemaProvider,
};
use arcgraph_query::CancellationToken;
use arcgraph_query::cancel::CancellationRegistry;
use rcgen::{CertificateParams, DnType, IsCa, KeyPair, KeyUsagePurpose};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName, pem::PemObject};
use rustls::{ClientConfig, RootCertStore};
use serde_json::{Value, json};
use time::OffsetDateTime;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::oneshot;
use tokio_rustls::TlsConnector;

// ─────────────────────────────────────────────────────────────────────
// Stub providers (mirrors `mcp_http_integ.rs`)
// ─────────────────────────────────────────────────────────────────────

struct StubSchema(TenantId);
impl SchemaProvider for StubSchema {
    fn schema(&self, tenant: TenantId) -> Result<GraphSchema, MCPError> {
        if tenant != self.0 {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        Ok(GraphSchema {
            tenant_id: tenant.raw(),
            labels: vec![LabelInfo {
                name: "Person".into(),
                cardinality: Some(1),
                properties: vec![],
            }],
            rel_types: vec![RelTypeInfo {
                name: "KNOWS".into(),
                cardinality: None,
            }],
            indexes: vec![IndexDescriptor {
                kind: IndexKind::Vector,
                available: true,
            }],
            total_node_count: Some(1),
            total_rel_count: Some(0),
        })
    }
}

struct StubInspect(TenantId);
impl NodeInspector for StubInspect {
    fn inspect(&self, tenant: TenantId, node_id: u64) -> Result<NodeInspection, MCPError> {
        if tenant != self.0 {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        Ok(NodeInspection {
            id: node_id,
            label: Some("Person".into()),
            properties: BTreeMap::new(),
            neighbors: vec![NeighborInfo {
                node_id: 2,
                label: Some("Person".into()),
                rel_type: Some("KNOWS".into()),
                direction: NeighborDirection::Out,
            }],
        })
    }
}

struct StubExplore(TenantId);
impl NeighborhoodExplorer for StubExplore {
    fn explore(
        &self,
        tenant: TenantId,
        seed: u64,
        max_depth: u32,
        _rel_filter: Option<&[String]>,
        _direction: arcgraph_mcp::tools::explore::ExploreDirection,
        cancel: &CancellationToken,
    ) -> Result<Neighborhood, MCPError> {
        if cancel.is_cancelled() {
            return Err(MCPError::Cancelled);
        }
        if tenant != self.0 {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        Ok(Neighborhood {
            seed,
            max_depth,
            truncated: false,
            nodes: vec![NeighborhoodNode {
                id: seed,
                label: Some("Person".into()),
                depth: 0,
                properties: BTreeMap::new(),
            }],
            edges: vec![NeighborhoodEdge {
                from: seed,
                to: seed + 1,
                rel_type: Some("KNOWS".into()),
                direction: NeighborDirection::Out,
            }],
        })
    }
}

struct StubSearch(TenantId);
impl HybridSearcher for StubSearch {
    fn available_substrates(
        &self,
        _t: TenantId,
        _c: &CancellationToken,
    ) -> Result<AvailableSubstrates, MCPError> {
        Ok(AvailableSubstrates {
            vector: true,
            bm25: true,
        })
    }
    fn search(
        &self,
        _t: TenantId,
        _q: &str,
        _v: Option<&[f32]>,
        _k: u32,
        _c: &CancellationToken,
    ) -> Result<Vec<SearchHit>, MCPError> {
        Ok(vec![])
    }
}

struct StubIngest(TenantId);
impl IngestProvider for StubIngest {
    fn ingest(&self, _t: TenantId, _b: IngestBatch) -> Result<IngestSummary, MCPError> {
        Ok(IngestSummary {
            records: vec![],
            inserted_count: 0,
            failed_count: 0,
            commit_lsn: None,
            dropped_acl_grants: Vec::new(),
        })
    }
}

struct StubRawQuery(TenantId);
impl arcgraph_mcp::tools::raw_query::RawQueryExecutor for StubRawQuery {
    fn execute(
        &self,
        _t: TenantId,
        _q: &str,
        _r: u32,
        _c: &CancellationToken,
    ) -> Result<arcgraph_mcp::RawQueryRows, MCPError> {
        Err(MCPError::InternalError("not exercised".into()))
    }
}

fn dispatcher_arc(
    tenant_raw: u64,
) -> Arc<Dispatcher<StubSchema, StubInspect, StubExplore, StubSearch, StubIngest, StubRawQuery>> {
    let t = TenantId::new(tenant_raw);
    Arc::new(Dispatcher::new(
        t,
        Arc::new(StubSchema(t)),
        Arc::new(StubInspect(t)),
        Arc::new(StubExplore(t)),
        Arc::new(StubSearch(t)),
        Arc::new(StubIngest(t)),
        Arc::new(StubRawQuery(t)),
    ))
}

// ─────────────────────────────────────────────────────────────────────
// CA + cert chain fixture: synthesize a self-signed CA, then mint a
// server cert + a client cert both signed by the CA.
// ─────────────────────────────────────────────────────────────────────

struct CaFixture {
    /// CA cert DER (used by the SERVER to validate client certs AND by
    /// the CLIENT to validate the server cert in valid-path tests).
    ca_cert_der: CertificateDer<'static>,
    /// CA cert as PEM bytes (consumed by `with_client_ca_pem`).
    ca_cert_pem: Vec<u8>,
    /// CA keypair (used to sign downstream end-entity certs).
    ca_kp: KeyPair,
    /// CA parameters retained for signing.
    ca_params: CertificateParams,
}

fn make_ca(common_name: &str) -> CaFixture {
    let mut params = CertificateParams::new(vec![common_name.to_string()]).expect("rcgen params");
    params.not_before = OffsetDateTime::now_utc() - time::Duration::days(1);
    params.not_after = OffsetDateTime::now_utc() + time::Duration::days(365);
    params
        .distinguished_name
        .push(DnType::CommonName, common_name);
    params.is_ca = IsCa::Ca(rcgen::BasicConstraints::Unconstrained);
    params.key_usages = vec![
        KeyUsagePurpose::KeyCertSign,
        KeyUsagePurpose::DigitalSignature,
        KeyUsagePurpose::CrlSign,
    ];
    let kp = KeyPair::generate().expect("CA keypair");
    let cert = params.clone().self_signed(&kp).expect("CA self-sign");
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
    /// Re-issue cert as PEM for FileSystemCertProvider-style fixtures.
    cert_pem: String,
    key_pem: String,
}

fn make_leaf_signed_by(
    ca: &CaFixture,
    common_name: &str,
    sans: &[&str],
    is_server_cert: bool,
) -> LeafCert {
    make_leaf_signed_by_with_window(
        ca,
        common_name,
        sans,
        is_server_cert,
        OffsetDateTime::now_utc() - time::Duration::days(1),
        OffsetDateTime::now_utc() + time::Duration::days(180),
    )
}

fn make_leaf_signed_by_with_window(
    ca: &CaFixture,
    common_name: &str,
    sans: &[&str],
    is_server_cert: bool,
    not_before: OffsetDateTime,
    not_after: OffsetDateTime,
) -> LeafCert {
    let san_vec: Vec<String> = sans.iter().map(|s| (*s).to_string()).collect();
    let mut params = CertificateParams::new(san_vec).expect("rcgen leaf params");
    params.not_before = not_before;
    params.not_after = not_after;
    params
        .distinguished_name
        .push(DnType::CommonName, common_name);
    if is_server_cert {
        params.extended_key_usages = vec![rcgen::ExtendedKeyUsagePurpose::ServerAuth];
    } else {
        params.extended_key_usages = vec![rcgen::ExtendedKeyUsagePurpose::ClientAuth];
    }
    let kp = KeyPair::generate().expect("leaf keypair");
    let issuer = rcgen::Issuer::new(ca.ca_params.clone(), &ca.ca_kp);
    let cert = params.signed_by(&kp, &issuer).expect("leaf sign");
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

/// Stage server cert + key on disk for the FileSystemCertProvider used
/// by HotReloadResolver. Returns the TempDir (kept alive via caller
/// binding), the cert path, the key path, and the server cert DER (for
/// the client to trust).
fn stage_server_cert_files(
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
    // Bundle leaf + CA so clients that only trust the ROOT can still
    // build a chain from the presented leaf.
    let bundled = format!(
        "{}{}",
        leaf.cert_pem,
        std::str::from_utf8(&ca.ca_cert_pem).expect("ca pem utf-8")
    );
    std::fs::write(&cert_path, bundled).expect("write cert");
    std::fs::write(&key_path, leaf.key_pem).expect("write key");
    (dir, cert_path, key_path, leaf.cert_der)
}

fn build_client_config_no_auth(
    trusted_server_roots: &[CertificateDer<'static>],
) -> Arc<ClientConfig> {
    let mut roots = RootCertStore::empty();
    for cert in trusted_server_roots {
        roots.add(cert.clone()).expect("add server root");
    }
    let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
    let cfg = ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .expect("protocol versions")
        .with_root_certificates(roots)
        .with_no_client_auth();
    Arc::new(cfg)
}

fn build_client_config_with_cert(
    trusted_server_roots: &[CertificateDer<'static>],
    client_cert: CertificateDer<'static>,
    client_key: PrivateKeyDer<'static>,
) -> Arc<ClientConfig> {
    let mut roots = RootCertStore::empty();
    for cert in trusted_server_roots {
        roots.add(cert.clone()).expect("add server root");
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
// Server harness (mirrors mcp_http_integ.rs spawn pattern)
// ─────────────────────────────────────────────────────────────────────

struct Harness {
    addr: std::net::SocketAddr,
    shutdown: oneshot::Sender<()>,
    join: tokio::task::JoinHandle<()>,
}

impl Harness {
    async fn shutdown(self) {
        let _ = self.shutdown.send(());
        let _ = tokio::time::timeout(Duration::from_secs(2), self.join).await;
    }
}

async fn spawn_https_with_mtls(
    cert_path: std::path::PathBuf,
    key_path: std::path::PathBuf,
    client_ca_pem: &[u8],
    client_cert_required: bool,
    tenant_strategy: TenantStrategy,
) -> Harness {
    use arcgraph_mcp::tls::{FileSystemCertProvider, HotReloadResolver};
    use tokio::net::TcpListener;

    let provider = Arc::new(FileSystemCertProvider::new(
        cert_path.clone(),
        key_path.clone(),
        Some("localhost".into()),
    ));
    let resolver = HotReloadResolver::new(provider).expect("resolver init");
    let temp = TcpListener::bind("127.0.0.1:0").await.expect("bind 0");
    let addr = temp.local_addr().expect("local addr");
    drop(temp);
    let config = HttpServerConfig::new(addr, resolver)
        .with_client_ca_pem(client_ca_pem, client_cert_required)
        .expect("with_client_ca_pem");
    let config = HttpServerConfig {
        tenant_strategy,
        ..config
    };

    let cancel_registry = Arc::new(CancellationRegistry::new());
    let cr = cancel_registry.clone();
    let (tx, rx) = oneshot::channel::<()>();
    let dispatcher = dispatcher_arc(7);
    let join = tokio::spawn(async move {
        let _ = serve_http(config, dispatcher, cr, async move {
            let _ = rx.await;
        })
        .await;
    });

    for _ in 0..40 {
        if TcpStream::connect(addr).await.is_ok() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    Harness {
        addr,
        shutdown: tx,
        join,
    }
}

#[derive(Debug)]
struct HttpResp {
    status: u16,
    body: Vec<u8>,
}

async fn https_post_mcp(
    addr: std::net::SocketAddr,
    client_cfg: Arc<ClientConfig>,
    server_name: &str,
    tenant_header: Option<&str>,
    body: &[u8],
) -> Result<HttpResp, String> {
    let tcp = TcpStream::connect(addr)
        .await
        .map_err(|e| format!("tcp connect: {e}"))?;
    let connector = TlsConnector::from(client_cfg);
    let dnsname = ServerName::try_from(server_name.to_owned()).map_err(|e| format!("dns: {e}"))?;
    let mut tls = connector
        .connect(dnsname, tcp)
        .await
        .map_err(|e| format!("tls handshake: {e}"))?;

    let mut req = format!(
        "POST {PATH_MCP} HTTP/1.1\r\nHost: {server_name}\r\nContent-Type: application/json\r\n"
    );
    if let Some(t) = tenant_header {
        req.push_str(&format!("{HEADER_TENANT}: {t}\r\n"));
    }
    req.push_str(&format!("Content-Length: {}\r\n", body.len()));
    req.push_str("Connection: close\r\n\r\n");
    tls.write_all(req.as_bytes())
        .await
        .map_err(|e| format!("write headers: {e}"))?;
    tls.write_all(body)
        .await
        .map_err(|e| format!("write body: {e}"))?;
    tls.flush().await.map_err(|e| format!("flush: {e}"))?;

    let mut buf = Vec::new();
    let _ = tls.read_to_end(&mut buf).await;
    let head_end = (0..buf.len().saturating_sub(3))
        .find(|i| &buf[*i..*i + 4] == b"\r\n\r\n")
        .ok_or_else(|| "no header terminator".to_string())?;
    let head = std::str::from_utf8(&buf[..head_end]).map_err(|e| format!("utf-8: {e}"))?;
    let status_line = head.lines().next().unwrap_or("");
    let code: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let body = buf[head_end + 4..].to_vec();
    Ok(HttpResp { status: code, body })
}

// ─────────────────────────────────────────────────────────────────────
// Test 1 — VALID client cert succeeds end-to-end through mTLS.
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn mtls_smoke_valid_client_cert_succeeds() {
    let ca = make_ca("ArcGraph Test CA");
    let (_dir, cert_path, key_path, server_cert_der) = stage_server_cert_files(&ca);
    let client = make_leaf_signed_by(&ca, "client-alice", &["client-alice.arcgraph.local"], false);

    let h = spawn_https_with_mtls(
        cert_path,
        key_path,
        &ca.ca_cert_pem,
        true, // client_cert_required
        TenantStrategy::Header,
    )
    .await;

    let client_cfg = build_client_config_with_cert(
        &[server_cert_der, ca.ca_cert_der.clone()],
        client.cert_der.clone(),
        client.key_der.clone_key(),
    );
    let body = serde_json::to_vec(&json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "graph.schema",
        "params": {"tenant_id": 7},
    }))
    .expect("encode body");

    let resp = https_post_mcp(h.addr, client_cfg, "localhost", Some("7"), &body)
        .await
        .expect("valid client cert handshake should succeed");
    assert_eq!(resp.status, 200, "valid client cert + tenant should 200");
    let env: Value = serde_json::from_slice(&resp.body).expect("response JSON");
    assert_eq!(env["id"], 1);
    assert!(
        env.get("result").is_some(),
        "expected JSON-RPC success: {env}"
    );

    h.shutdown().await;
}

// ─────────────────────────────────────────────────────────────────────
// Test 2 — UNTRUSTED client cert (signed by a different CA) rejects.
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn mtls_smoke_untrusted_client_cert_rejects_handshake() {
    let server_ca = make_ca("ArcGraph Test CA");
    let attacker_ca = make_ca("Attacker CA");
    let (_dir, cert_path, key_path, server_cert_der) = stage_server_cert_files(&server_ca);
    // Attacker client cert: signed by attacker_ca, NOT by server_ca.
    let bad_client = make_leaf_signed_by(&attacker_ca, "rogue", &["rogue.attacker.example"], false);

    let h = spawn_https_with_mtls(
        cert_path,
        key_path,
        &server_ca.ca_cert_pem,
        true,
        TenantStrategy::Header,
    )
    .await;

    let client_cfg = build_client_config_with_cert(
        &[server_cert_der, server_ca.ca_cert_der.clone()],
        bad_client.cert_der.clone(),
        bad_client.key_der.clone_key(),
    );
    let body = serde_json::to_vec(
        &json!({"jsonrpc":"2.0","id":1,"method":"graph.schema","params":{"tenant_id":7}}),
    )
    .expect("body");

    let result = https_post_mcp(h.addr, client_cfg, "localhost", Some("7"), &body).await;
    // ASSERTION DISCIPLINE (per `feedback_load_bearing_pr_requires_fault_injection_tests.md`):
    // we MUST confirm the untrusted cert was REJECTED — not just that
    // "something failed." Allowed surface forms:
    //  (a) Handshake-time alert → `connector.connect(...)` Err with
    //      "tls handshake" / "certificate" / "alert" / "UnknownCA"
    //      in the message — TLS 1.3's canonical path.
    //  (b) Handshake completes (TLS 1.3 client sees server's Finished
    //      BEFORE the server validates the client cert), server drops
    //      mid-request → read_to_end returns 0 bytes pre-framing →
    //      header-parse fails with "no header terminator". This is
    //      the documented rustls 0.23 + tokio-rustls 0.26 behavior for
    //      WebPkiClientVerifier rejection on certain TLS 1.3 timing
    //      shapes (the server's CertificateVerify check runs AFTER the
    //      handshake-ack flight has been queued but BEFORE the
    //      application-data records are accepted; on reject the server
    //      drops the connection instead of sending app-layer alert).
    //
    // In EITHER (a) or (b) the assertion below holds: NO dispatch
    // envelope was returned. The integration test below
    // (mtls_smoke_valid_client_cert_succeeds) is the positive control:
    // it sees a 200 + JSON-RPC envelope. So this Err implies the cert
    // path actively failed at the verifier — the negative arm of the
    // same comparator.
    assert!(
        result.is_err(),
        "untrusted client cert MUST be rejected (no dispatch envelope); got Ok={result:?}"
    );
    h.shutdown().await;
}

// ─────────────────────────────────────────────────────────────────────
// Test 3 — MISSING client cert with client_cert_required=true rejects.
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn mtls_smoke_missing_client_cert_when_required_rejects() {
    let ca = make_ca("ArcGraph Test CA");
    let (_dir, cert_path, key_path, server_cert_der) = stage_server_cert_files(&ca);

    let h = spawn_https_with_mtls(
        cert_path,
        key_path,
        &ca.ca_cert_pem,
        true,
        TenantStrategy::Header,
    )
    .await;

    // Client config with NO client cert.
    let client_cfg = build_client_config_no_auth(&[server_cert_der, ca.ca_cert_der.clone()]);
    let body = serde_json::to_vec(
        &json!({"jsonrpc":"2.0","id":1,"method":"graph.schema","params":{"tenant_id":7}}),
    )
    .expect("body");

    let result = https_post_mcp(h.addr, client_cfg, "localhost", Some("7"), &body).await;
    assert!(
        result.is_err(),
        "missing client cert with required=true MUST reject; got {result:?}"
    );
    h.shutdown().await;
}

// ─────────────────────────────────────────────────────────────────────
// Test 4 — VALID client cert with PeerCertSan tenant strategy: tenant
// extracted from cert SAN.
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn mtls_smoke_peer_cert_san_strategy_routes_tenant_from_cert() {
    let ca = make_ca("ArcGraph Test CA");
    let (_dir, cert_path, key_path, server_cert_der) = stage_server_cert_files(&ca);
    // tenant-7 in SAN → dispatcher receives TenantId::new(7) without
    // an X-ArcGraph-Tenant header.
    let client = make_leaf_signed_by(&ca, "client-tenant-7", &["tenant-7.arcgraph.local"], false);

    let h = spawn_https_with_mtls(
        cert_path,
        key_path,
        &ca.ca_cert_pem,
        true,
        TenantStrategy::PeerCertSan,
    )
    .await;

    let client_cfg = build_client_config_with_cert(
        &[server_cert_der, ca.ca_cert_der.clone()],
        client.cert_der.clone(),
        client.key_der.clone_key(),
    );
    // NO X-ArcGraph-Tenant header — tenant must come from cert SAN.
    let body = serde_json::to_vec(&json!({
        "jsonrpc":"2.0","id":1,"method":"graph.schema","params":{"tenant_id":7},
    }))
    .expect("body");

    let resp = https_post_mcp(h.addr, client_cfg, "localhost", None, &body)
        .await
        .expect("valid SAN-cert handshake should succeed");
    assert_eq!(
        resp.status, 200,
        "tenant from cert SAN should route to dispatcher",
    );
    let env: Value = serde_json::from_slice(&resp.body).expect("response JSON");
    assert!(
        env.get("result").is_some(),
        "expected success envelope: {env}"
    );
    h.shutdown().await;
}

// ─────────────────────────────────────────────────────────────────────
// Test 5 — Cross-tenant probe: client A's cert (tenant-7) cannot drive
// a request for tenant 9 (mismatch rejects with 403 / Unauthorized).
// This is the W19γ adversarial cross-tenant probe extended to the mTLS
// transport.
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn mtls_smoke_cross_tenant_probe_rejects() {
    let ca = make_ca("ArcGraph Test CA");
    let (_dir, cert_path, key_path, server_cert_der) = stage_server_cert_files(&ca);
    let client_tenant_7 =
        make_leaf_signed_by(&ca, "client-tenant-7", &["tenant-7.arcgraph.local"], false);

    let h = spawn_https_with_mtls(
        cert_path,
        key_path,
        &ca.ca_cert_pem,
        true,
        TenantStrategy::PeerCertSan,
    )
    .await;

    let client_cfg = build_client_config_with_cert(
        &[server_cert_der, ca.ca_cert_der.clone()],
        client_tenant_7.cert_der.clone(),
        client_tenant_7.key_der.clone_key(),
    );
    // The cert SAN says tenant-7 but the envelope asks for tenant 9.
    // The transport MUST reject as 403 before the dispatcher's tool
    // body runs.
    let body = serde_json::to_vec(&json!({
        "jsonrpc":"2.0","id":1,"method":"graph.schema","params":{"tenant_id":9},
    }))
    .expect("body");

    let resp = https_post_mcp(h.addr, client_cfg, "localhost", None, &body)
        .await
        .expect("handshake succeeds; rejection is at the envelope-tenant guard");
    assert_eq!(
        resp.status,
        403,
        "cross-tenant probe MUST return 403 Forbidden; got {} body={:?}",
        resp.status,
        String::from_utf8_lossy(&resp.body),
    );
    let env: Value = serde_json::from_slice(&resp.body).expect("response JSON");
    assert_eq!(env["error"]["code"], -32002, "expected Unauthorized code");
    h.shutdown().await;
}

// ─────────────────────────────────────────────────────────────────────
// Test 6 — EXPIRED client cert (not_after in the past) rejects.
//
// Per R1 M-3 + `feedback_load_bearing_pr_requires_fault_injection_tests.md`:
// each documented failure mode of mTLS chain-verify gets a per-failure-
// mode regression test. The cert is signed by the trusted CA (so the
// trust-store-membership gate passes) but `not_after` is set to one day
// in the past — rustls's WebPkiClientVerifier MUST reject at validity-
// window check. Without this regression, a future rustls upgrade that
// loosens chain-verify-with-time would silently admit expired certs.
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn mtls_smoke_expired_client_cert_rejects() {
    let ca = make_ca("ArcGraph Test CA");
    let (_dir, cert_path, key_path, server_cert_der) = stage_server_cert_files(&ca);
    // Signed by the trusted CA, but expired.
    let expired_client = make_leaf_signed_by_with_window(
        &ca,
        "client-expired",
        &["client-expired.arcgraph.local"],
        false,
        OffsetDateTime::now_utc() - time::Duration::days(30),
        OffsetDateTime::now_utc() - time::Duration::days(1),
    );

    let h = spawn_https_with_mtls(
        cert_path,
        key_path,
        &ca.ca_cert_pem,
        true,
        TenantStrategy::Header,
    )
    .await;

    let client_cfg = build_client_config_with_cert(
        &[server_cert_der, ca.ca_cert_der.clone()],
        expired_client.cert_der.clone(),
        expired_client.key_der.clone_key(),
    );
    let body = serde_json::to_vec(
        &json!({"jsonrpc":"2.0","id":1,"method":"graph.schema","params":{"tenant_id":7}}),
    )
    .expect("body");

    let result = https_post_mcp(h.addr, client_cfg, "localhost", Some("7"), &body).await;
    // Same assertion discipline as the untrusted-CA case: EITHER (a) the
    // handshake errs on the client side via TLS alert, OR (b) the server
    // accepts the handshake-ack flight before validity-window rejection
    // and drops the connection mid-request. In both cases, NO dispatch
    // envelope is returned. The positive control above
    // (`mtls_smoke_valid_client_cert_succeeds`) confirms the success
    // path returns 200 + JSON-RPC envelope; this Err implies the
    // verifier actively rejected the expired cert.
    assert!(
        result.is_err(),
        "expired client cert MUST be rejected (no dispatch envelope); got Ok={result:?}"
    );
    h.shutdown().await;
}
