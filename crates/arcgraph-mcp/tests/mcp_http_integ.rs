//! W14α M5-02b integration tests — end-to-end HTTP/TLS MCP transport.
//!
//! These tests boot a real `serve_http` instance on `127.0.0.1:0`,
//! drive a raw TLS client over `tokio_rustls`, and verify the
//! request/response cycle. The server uses an `rcgen` self-signed
//! cert + the W13ε [`HotReloadResolver`]; the client's trust store
//! contains the same cert so handshakes succeed without `WebPki`
//! unknown-issuer errors.
//!
//! Coverage:
//!   1. End-to-end TLS roundtrip — POST /mcp + valid envelope →
//!      JSON-RPC success response.
//!   2. Cert rotation mid-listener — the resolver swaps; a fresh
//!      connection picks up the rotated cert.
//!   3. GET /healthz returns 200 OK.
//!   4. Bad method (GET /mcp) → 405 Method Not Allowed.
//!   5. Missing tenant header → 400 with -32600 envelope.
//!   6. Cross-tenant payload vs. header → 403 with -32002 envelope.
//!   7. Origin allowlist: in-list passes, out-of-list rejects 403.
//!   8. Bound-tenant fence rejects forged header with 403.

#![allow(clippy::expect_used)] // tests are allowed to expect; failure context is high-signal here.

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
    HEADER_TENANT, HttpServerConfig, PATH_HEALTHZ, PATH_MCP, TenantStrategy, serve_http,
};
use arcgraph_mcp::{
    Dispatcher, HybridSearcher, MCPError, NeighborhoodExplorer, NodeInspector, SchemaProvider,
};
use arcgraph_query::CancellationToken;
use arcgraph_query::cancel::CancellationRegistry;
use rcgen::{CertificateParams, DnType, KeyPair};
use rustls::pki_types::{CertificateDer, ServerName};
use rustls::{ClientConfig, RootCertStore};
use serde_json::{Value, json};
use tempfile::TempDir;
use time::OffsetDateTime;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::oneshot;
use tokio_rustls::TlsConnector;

// ─────────────────────────────────────────────────────────────────────
// Stub providers (mirror the W13δ stdio integ ones)
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
                cardinality: Some(2),
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
            total_node_count: Some(2),
            total_rel_count: Some(1),
        })
    }
}

struct StubInspect(TenantId);
impl NodeInspector for StubInspect {
    fn inspect(&self, tenant: TenantId, node_id: u64) -> Result<NodeInspection, MCPError> {
        if tenant != self.0 {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        let mut p = BTreeMap::new();
        p.insert("name".into(), json!("Alice"));
        Ok(NodeInspection {
            id: node_id,
            label: Some("Person".into()),
            properties: p,
            neighbors: vec![NeighborInfo {
                node_id: 2,
                label: Some("Person".into()),
                rel_type: Some("KNOWS".into()),
                direction: NeighborDirection::Out,
            }],
        })
    }
}

// W14β M5-06: minimal NeighborhoodExplorer stub for HTTP-transport
// composition tests. The HTTP integ harness exercises the transport
// layer (path/method gate, TLS, Origin allowlist, tenant fence) — not
// the explorer body. A stub is sufficient.
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

// W14β M5-07: minimal HybridSearcher stub for HTTP-transport tests.
struct StubSearch(TenantId);
impl HybridSearcher for StubSearch {
    fn available_substrates(
        &self,
        tenant: TenantId,
        cancel: &CancellationToken,
    ) -> Result<AvailableSubstrates, MCPError> {
        if cancel.is_cancelled() {
            return Err(MCPError::Cancelled);
        }
        if tenant != self.0 {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        Ok(AvailableSubstrates {
            vector: true,
            bm25: true,
        })
    }
    fn search(
        &self,
        tenant: TenantId,
        _query_text: &str,
        _query_vec: Option<&[f32]>,
        k: u32,
        cancel: &CancellationToken,
    ) -> Result<Vec<SearchHit>, MCPError> {
        if cancel.is_cancelled() {
            return Err(MCPError::Cancelled);
        }
        if tenant != self.0 {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        let mut hits = vec![SearchHit {
            node_id: 1,
            label: Some("Document".into()),
            score: 0.9,
        }];
        hits.truncate(k as usize);
        Ok(hits)
    }
}

/// Stub ingest provider — the http integ tests don't exercise the
/// ingest tool; they need the trait to satisfy the dispatcher's
/// `IngestProvider` generic. Returns a tenant-unknown error so an
/// accidental ingest call would fail loudly rather than silently.
struct StubIngest(TenantId);
impl IngestProvider for StubIngest {
    fn ingest(&self, tenant: TenantId, _batch: IngestBatch) -> Result<IngestSummary, MCPError> {
        if tenant != self.0 {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        Err(MCPError::InternalError(
            "stub ingest not exercised by http integ tests".into(),
        ))
    }
}

/// Stub raw-query executor (W16ζ M5-11) — http integ tests do not
/// exercise the raw_query tool; the impl exists only to satisfy the
/// W16ζ-merged dispatcher's `RawQueryExecutor` generic.
struct StubRawQuery(TenantId);
impl arcgraph_mcp::tools::raw_query::RawQueryExecutor for StubRawQuery {
    fn execute(
        &self,
        tenant: TenantId,
        _query: &str,
        _max_rows: u32,
        _cancel: &CancellationToken,
    ) -> Result<arcgraph_mcp::tools::raw_query::RawQueryRows, MCPError> {
        if tenant != self.0 {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        Err(MCPError::InternalError(
            "stub raw_query not exercised by http integ tests".into(),
        ))
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
// rcgen-driven cert fixture for HTTPS
// ─────────────────────────────────────────────────────────────────────

struct ServerCertFixture {
    cert_der: CertificateDer<'static>,
}

/// Build a complete file-fixture pair (cert + key) on disk so the
/// server's `FileSystemCertProvider` can read them.
fn stage_filesystem_fixture() -> (
    TempDir,
    ServerCertFixture,
    std::path::PathBuf,
    std::path::PathBuf,
) {
    use std::path::PathBuf;
    let dir = tempfile::tempdir().expect("tempdir");
    let mut params = CertificateParams::new(vec!["localhost".to_string()]).expect("rcgen params");
    params.not_before = OffsetDateTime::now_utc() - time::Duration::days(1);
    params.not_after = OffsetDateTime::now_utc() + time::Duration::days(365);
    params
        .distinguished_name
        .push(DnType::CommonName, "localhost");
    let kp = KeyPair::generate().expect("rcgen keypair");
    let cert = params.self_signed(&kp).expect("self-signed");
    let cert_pem = cert.pem();
    let cert_der_owned = cert.der().clone();
    let cert_path: PathBuf = dir.path().join("server.crt");
    let key_path: PathBuf = dir.path().join("server.key");
    std::fs::write(&cert_path, &cert_pem).expect("write cert");
    std::fs::write(&key_path, kp.serialize_pem()).expect("write key");
    let fx = ServerCertFixture {
        cert_der: CertificateDer::from(cert_der_owned.as_ref().to_vec()),
    };
    (dir, fx, cert_path, key_path)
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

// ─────────────────────────────────────────────────────────────────────
// Test harness: spawn a serve_http on 127.0.0.1:0 and return port +
// shutdown sender + cancellation registry.
// ─────────────────────────────────────────────────────────────────────

struct HarnessHandle {
    addr: std::net::SocketAddr,
    shutdown: oneshot::Sender<()>,
    join: tokio::task::JoinHandle<()>,
}

async fn spawn_harness(
    config: HttpServerConfig,
    dispatcher: Arc<
        Dispatcher<StubSchema, StubInspect, StubExplore, StubSearch, StubIngest, StubRawQuery>,
    >,
) -> HarnessHandle {
    use tokio::net::TcpListener;
    // Pre-bind to grab the assigned port; the harness will rebind on
    // the same port inside `serve_http`. This is necessary because
    // `serve_http` returns AFTER listening, but we need the port BEFORE
    // returning to drive the client.
    let temp = TcpListener::bind(config.bind_addr)
        .await
        .expect("temp bind");
    let assigned = temp.local_addr().expect("local addr");
    drop(temp);

    let bind_config = HttpServerConfig {
        bind_addr: assigned,
        ..config
    };

    let cancel_registry = Arc::new(CancellationRegistry::new());
    let cr_clone = cancel_registry.clone();
    let (tx, rx) = oneshot::channel::<()>();
    let shutdown = async move {
        let _ = rx.await;
    };

    let join = tokio::spawn(async move {
        let _ = serve_http(bind_config, dispatcher, cr_clone, shutdown).await;
    });

    // Brief settle: spawn returned, but the listener may not be bound
    // yet. Poll for connectability with a 2s overall budget.
    for _ in 0..40 {
        if TcpStream::connect(assigned).await.is_ok() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    let _ = cancel_registry;
    HarnessHandle {
        addr: assigned,
        shutdown: tx,
        join,
    }
}

impl HarnessHandle {
    async fn shutdown(self) {
        let _ = self.shutdown.send(());
        let _ = tokio::time::timeout(Duration::from_secs(2), self.join).await;
    }
}

// ─────────────────────────────────────────────────────────────────────
// HTTP/1.1 raw client: handcrafted request over a TLS stream
// ─────────────────────────────────────────────────────────────────────

struct HttpResp {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

#[allow(clippy::too_many_arguments)]
async fn https_request(
    addr: std::net::SocketAddr,
    client_cfg: Arc<ClientConfig>,
    server_name: &str,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: &[u8],
) -> HttpResp {
    let tcp = TcpStream::connect(addr).await.expect("tcp connect");
    let connector = TlsConnector::from(client_cfg);
    let dnsname = ServerName::try_from(server_name.to_owned()).expect("dns name");
    let mut tls = connector
        .connect(dnsname, tcp)
        .await
        .expect("tls handshake");

    let mut req = format!("{method} {path} HTTP/1.1\r\nHost: {server_name}\r\n");
    for (name, value) in headers {
        req.push_str(&format!("{name}: {value}\r\n"));
    }
    req.push_str(&format!("Content-Length: {}\r\n", body.len()));
    req.push_str("Connection: close\r\n\r\n");
    tls.write_all(req.as_bytes()).await.expect("write headers");
    if !body.is_empty() {
        tls.write_all(body).await.expect("write body");
    }
    tls.flush().await.expect("flush");

    let mut buf = Vec::new();
    let _ = tls.read_to_end(&mut buf).await; // peer closes after one response.
    parse_http_response(&buf)
}

fn parse_http_response(bytes: &[u8]) -> HttpResp {
    let mut head_end = 0;
    for i in 0..bytes.len().saturating_sub(3) {
        if &bytes[i..i + 4] == b"\r\n\r\n" {
            head_end = i;
            break;
        }
    }
    assert!(head_end > 0, "no header terminator in response");
    let header_block = std::str::from_utf8(&bytes[..head_end]).expect("response headers utf-8");
    let mut lines = header_block.split("\r\n");
    let status_line = lines.next().unwrap_or("");
    let mut parts = status_line.split_whitespace();
    let _ver = parts.next();
    let code: u16 = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let mut headers = Vec::new();
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            headers.push((name.trim().to_string(), value.trim().to_string()));
        }
    }
    let body = bytes[head_end + 4..].to_vec();
    HttpResp {
        status: code,
        headers,
        body,
    }
}

// ─────────────────────────────────────────────────────────────────────
// Test 1: end-to-end TLS roundtrip — POST /mcp returns success
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn integ_post_mcp_returns_jsonrpc_success() {
    let (_dir, fx, cert_path, key_path) = stage_filesystem_fixture();
    let provider = Arc::new(arcgraph_mcp::tls::FileSystemCertProvider::new(
        cert_path, key_path, None,
    ));
    let resolver = arcgraph_mcp::tls::HotReloadResolver::new(provider).expect("resolver init");
    let cfg = HttpServerConfig::new("127.0.0.1:0".parse().unwrap(), resolver);
    let harness = spawn_harness(cfg, dispatcher_arc(7)).await;

    let client_cfg = build_client_config(std::slice::from_ref(&fx.cert_der));
    let body = serde_json::to_vec(&json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "graph.schema",
        "params": {"tenant_id": 7}
    }))
    .unwrap();
    let resp = https_request(
        harness.addr,
        client_cfg,
        "localhost",
        "POST",
        PATH_MCP,
        &[("content-type", "application/json"), (HEADER_TENANT, "7")],
        &body,
    )
    .await;
    assert_eq!(resp.status, 200, "POST /mcp must return 200 OK");
    let v: Value = serde_json::from_slice(&resp.body).expect("response is JSON");
    assert_eq!(v["id"], 1);
    assert!(v.get("result").is_some(), "success envelope");
    assert!(v["result"]["body"].is_string(), "rendered body");

    harness.shutdown().await;
}

#[tokio::test]
async fn integ_post_mcp_initialize_uses_shared_lifecycle_prerouter() {
    let (_dir, fx, cert_path, key_path) = stage_filesystem_fixture();
    let provider = Arc::new(arcgraph_mcp::tls::FileSystemCertProvider::new(
        cert_path, key_path, None,
    ));
    let resolver = arcgraph_mcp::tls::HotReloadResolver::new(provider).expect("resolver init");
    let cfg = HttpServerConfig::new("127.0.0.1:0".parse().unwrap(), resolver);
    let harness = spawn_harness(cfg, dispatcher_arc(7)).await;

    let client_cfg = build_client_config(std::slice::from_ref(&fx.cert_der));
    let body = serde_json::to_vec(&json!({
        "jsonrpc": "2.0",
        "id": "init-http",
        "method": "initialize",
        "params": {"protocolVersion": "2025-03-26"}
    }))
    .unwrap();
    let resp = https_request(
        harness.addr,
        client_cfg,
        "localhost",
        "POST",
        PATH_MCP,
        &[("content-type", "application/json"), (HEADER_TENANT, "7")],
        &body,
    )
    .await;
    assert_eq!(resp.status, 200, "POST /mcp initialize must return 200");
    let v: Value = serde_json::from_slice(&resp.body).expect("response is JSON");
    assert_eq!(v["id"], "init-http");
    assert_eq!(v["result"]["protocolVersion"], "2025-03-26");
    assert_eq!(v["result"]["capabilities"]["tools"]["listChanged"], false);
    assert_eq!(v["result"]["serverInfo"]["name"], "arcgraph");

    harness.shutdown().await;
}

// ─────────────────────────────────────────────────────────────────────
// Test 2: cert rotation mid-listener — resolver swap is observed by
//          fresh connections
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn integ_cert_rotation_mid_connection_is_observed_by_new_conns() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cert_path = dir.path().join("server.crt");
    let key_path = dir.path().join("server.key");

    // Initial cert (CN=localhost, SAN=localhost).
    let mut params = CertificateParams::new(vec!["localhost".to_string()]).unwrap();
    params.not_before = OffsetDateTime::now_utc() - time::Duration::days(1);
    params.not_after = OffsetDateTime::now_utc() + time::Duration::days(365);
    params
        .distinguished_name
        .push(DnType::CommonName, "localhost");
    let kp1 = KeyPair::generate().unwrap();
    let cert1 = params.self_signed(&kp1).unwrap();
    let cert1_der = CertificateDer::from(cert1.der().as_ref().to_vec());
    std::fs::write(&cert_path, cert1.pem()).unwrap();
    std::fs::write(&key_path, kp1.serialize_pem()).unwrap();

    let provider = Arc::new(arcgraph_mcp::tls::FileSystemCertProvider::new(
        cert_path.clone(),
        key_path.clone(),
        None,
    ));
    let resolver = arcgraph_mcp::tls::HotReloadResolver::new(provider).expect("resolver init");
    let resolver_for_reload = resolver.clone();

    let cfg = HttpServerConfig::new("127.0.0.1:0".parse().unwrap(), resolver);
    let harness = spawn_harness(cfg, dispatcher_arc(7)).await;

    // Connection #1 with the initial cert in the trust store works.
    let client1 = build_client_config(std::slice::from_ref(&cert1_der));
    let resp1 = https_request(
        harness.addr,
        client1,
        "localhost",
        "GET",
        PATH_HEALTHZ,
        &[],
        &[],
    )
    .await;
    assert_eq!(resp1.status, 200);

    // Rotate: stage a new cert with a fresh keypair on disk, fire
    // resolver.reload(), then prove a new connection picks it up.
    let mut params2 = CertificateParams::new(vec!["localhost".to_string()]).unwrap();
    params2.not_before = OffsetDateTime::now_utc() - time::Duration::days(1);
    params2.not_after = OffsetDateTime::now_utc() + time::Duration::days(365);
    params2
        .distinguished_name
        .push(DnType::CommonName, "localhost");
    let kp2 = KeyPair::generate().unwrap();
    let cert2 = params2.self_signed(&kp2).unwrap();
    let cert2_der = CertificateDer::from(cert2.der().as_ref().to_vec());
    std::fs::write(&cert_path, cert2.pem()).unwrap();
    std::fs::write(&key_path, kp2.serialize_pem()).unwrap();

    resolver_for_reload.reload().expect("reload after rotation");

    // Connection #2 trusts only the rotated cert; must succeed.
    let client2 = build_client_config(&[cert2_der]);
    let resp2 = https_request(
        harness.addr,
        client2,
        "localhost",
        "GET",
        PATH_HEALTHZ,
        &[],
        &[],
    )
    .await;
    assert_eq!(
        resp2.status, 200,
        "after rotation, the new cert is presented"
    );

    harness.shutdown().await;
}

// ─────────────────────────────────────────────────────────────────────
// Test 3: GET /healthz returns 200 + JSON status body
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn integ_healthz_returns_200_with_status_ok() {
    let (_dir, fx, cert_path, key_path) = stage_filesystem_fixture();
    let provider = Arc::new(arcgraph_mcp::tls::FileSystemCertProvider::new(
        cert_path, key_path, None,
    ));
    let resolver = arcgraph_mcp::tls::HotReloadResolver::new(provider).expect("resolver init");
    let cfg = HttpServerConfig::new("127.0.0.1:0".parse().unwrap(), resolver);
    let harness = spawn_harness(cfg, dispatcher_arc(7)).await;

    let client = build_client_config(std::slice::from_ref(&fx.cert_der));
    let resp = https_request(
        harness.addr,
        client,
        "localhost",
        "GET",
        PATH_HEALTHZ,
        &[],
        &[],
    )
    .await;
    assert_eq!(resp.status, 200);
    let s = String::from_utf8(resp.body).unwrap();
    assert!(s.contains("\"ok\""), "got: {s}");

    harness.shutdown().await;
}

// ─────────────────────────────────────────────────────────────────────
// Test 4: GET /mcp → 405 Method Not Allowed
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn integ_get_mcp_returns_405() {
    let (_dir, fx, cert_path, key_path) = stage_filesystem_fixture();
    let provider = Arc::new(arcgraph_mcp::tls::FileSystemCertProvider::new(
        cert_path, key_path, None,
    ));
    let resolver = arcgraph_mcp::tls::HotReloadResolver::new(provider).expect("resolver init");
    let cfg = HttpServerConfig::new("127.0.0.1:0".parse().unwrap(), resolver);
    let harness = spawn_harness(cfg, dispatcher_arc(7)).await;

    let client = build_client_config(std::slice::from_ref(&fx.cert_der));
    let resp = https_request(harness.addr, client, "localhost", "GET", PATH_MCP, &[], &[]).await;
    assert_eq!(resp.status, 405);
    let allow = resp
        .headers
        .iter()
        .find(|(n, _)| n.eq_ignore_ascii_case("allow"))
        .map(|(_, v)| v.as_str());
    assert_eq!(allow, Some("POST"));

    harness.shutdown().await;
}

// ─────────────────────────────────────────────────────────────────────
// Test 5: missing tenant header → 400 with -32600 envelope
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn integ_post_mcp_missing_tenant_header_returns_400() {
    let (_dir, fx, cert_path, key_path) = stage_filesystem_fixture();
    let provider = Arc::new(arcgraph_mcp::tls::FileSystemCertProvider::new(
        cert_path, key_path, None,
    ));
    let resolver = arcgraph_mcp::tls::HotReloadResolver::new(provider).expect("resolver init");
    let mut cfg = HttpServerConfig::new("127.0.0.1:0".parse().unwrap(), resolver);
    cfg.tenant_strategy = TenantStrategy::Header;
    let harness = spawn_harness(cfg, dispatcher_arc(7)).await;

    let client = build_client_config(std::slice::from_ref(&fx.cert_der));
    let body = serde_json::to_vec(&json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "graph.schema",
        "params": {"tenant_id": 7}
    }))
    .unwrap();
    let resp = https_request(
        harness.addr,
        client,
        "localhost",
        "POST",
        PATH_MCP,
        &[("content-type", "application/json")],
        &body,
    )
    .await;
    assert_eq!(resp.status, 400, "missing X-ArcGraph-Tenant must reject");
    let v: Value = serde_json::from_slice(&resp.body).expect("json");
    assert_eq!(v["error"]["code"], -32600);

    harness.shutdown().await;
}

// ─────────────────────────────────────────────────────────────────────
// Test 6: cross-tenant payload vs. transport-identified tenant
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn integ_cross_tenant_request_returns_403_unauthorized() {
    let (_dir, fx, cert_path, key_path) = stage_filesystem_fixture();
    let provider = Arc::new(arcgraph_mcp::tls::FileSystemCertProvider::new(
        cert_path, key_path, None,
    ));
    let resolver = arcgraph_mcp::tls::HotReloadResolver::new(provider).expect("resolver init");
    let cfg = HttpServerConfig::new("127.0.0.1:0".parse().unwrap(), resolver);
    let harness = spawn_harness(cfg, dispatcher_arc(7)).await;

    let client = build_client_config(std::slice::from_ref(&fx.cert_der));
    // Header says tenant 7, but envelope params say tenant 8.
    let body = serde_json::to_vec(&json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "graph.schema",
        "params": {"tenant_id": 8}
    }))
    .unwrap();
    let resp = https_request(
        harness.addr,
        client,
        "localhost",
        "POST",
        PATH_MCP,
        &[("content-type", "application/json"), (HEADER_TENANT, "7")],
        &body,
    )
    .await;
    assert_eq!(
        resp.status, 403,
        "cross-tenant request must surface 403 Forbidden"
    );
    let v: Value = serde_json::from_slice(&resp.body).expect("json");
    assert_eq!(v["error"]["code"], -32002, "Unauthorized = -32002");

    harness.shutdown().await;
}

// (Former test 7 — deadline timer registry-level smoke — was removed
// in W14α fix-up: identical-in-shape coverage already lives in
// `crates/arcgraph-mcp/src/transport/http.rs::tests::deadline_timer_arms_with_configured_value`.
// The full "slow substrate" end-to-end pin is M5-06+ once an async
// executor consumes the cancel-token at batch boundaries.)

// ─────────────────────────────────────────────────────────────────────
// Test 7 (new): Origin allowlist — in-list pass, out-of-list 403
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn integ_origin_allowlist_in_list_passes_out_of_list_rejects() {
    let (_dir, fx, cert_path, key_path) = stage_filesystem_fixture();
    let provider = Arc::new(arcgraph_mcp::tls::FileSystemCertProvider::new(
        cert_path, key_path, None,
    ));
    let resolver = arcgraph_mcp::tls::HotReloadResolver::new(provider).expect("resolver init");
    let cfg = HttpServerConfig::new("127.0.0.1:0".parse().unwrap(), resolver)
        .with_allowed_origins(vec!["https://app.example.com".into()]);
    let harness = spawn_harness(cfg, dispatcher_arc(7)).await;

    let client = build_client_config(std::slice::from_ref(&fx.cert_der));
    let body = serde_json::to_vec(&json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "graph.schema",
        "params": {"tenant_id": 7}
    }))
    .unwrap();

    // In-list origin → 200
    let resp_ok = https_request(
        harness.addr,
        client.clone(),
        "localhost",
        "POST",
        PATH_MCP,
        &[
            ("content-type", "application/json"),
            (HEADER_TENANT, "7"),
            ("Origin", "https://app.example.com"),
        ],
        &body,
    )
    .await;
    assert_eq!(resp_ok.status, 200, "in-list Origin must pass");

    // Out-of-list origin → 403
    let resp_bad = https_request(
        harness.addr,
        client.clone(),
        "localhost",
        "POST",
        PATH_MCP,
        &[
            ("content-type", "application/json"),
            (HEADER_TENANT, "7"),
            ("Origin", "https://evil.example.com"),
        ],
        &body,
    )
    .await;
    assert_eq!(
        resp_bad.status, 403,
        "out-of-list Origin must reject 403 (DNS-rebinding defense per design-v2 §9.4)"
    );
    let v: Value = serde_json::from_slice(&resp_bad.body).expect("json");
    assert_eq!(
        v["error"]["code"], -32600,
        "InvalidRequest for forbidden origin"
    );

    // No Origin header (CLI / curl style) → still passes
    let resp_no_origin = https_request(
        harness.addr,
        client,
        "localhost",
        "POST",
        PATH_MCP,
        &[("content-type", "application/json"), (HEADER_TENANT, "7")],
        &body,
    )
    .await;
    assert_eq!(
        resp_no_origin.status, 200,
        "no Origin header (non-browser client) passes the allowlist"
    );

    harness.shutdown().await;
}

// ─────────────────────────────────────────────────────────────────────
// Test 8 (new): bound_tenant fence — forged header rejected at transport
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn integ_bound_tenant_fence_rejects_forged_header() {
    let (_dir, fx, cert_path, key_path) = stage_filesystem_fixture();
    let provider = Arc::new(arcgraph_mcp::tls::FileSystemCertProvider::new(
        cert_path, key_path, None,
    ));
    let resolver = arcgraph_mcp::tls::HotReloadResolver::new(provider).expect("resolver init");
    // Server is bound to tenant 7; the dispatcher is also bound to
    // tenant 7. An attacker sending X-ArcGraph-Tenant: 99 should be
    // rejected at the transport boundary, BEFORE the dispatcher gets
    // a chance to see the request.
    let cfg = HttpServerConfig::new("127.0.0.1:0".parse().unwrap(), resolver)
        .with_bound_tenant(TenantId::new(7));
    let harness = spawn_harness(cfg, dispatcher_arc(7)).await;

    let client = build_client_config(std::slice::from_ref(&fx.cert_der));
    let body = serde_json::to_vec(&json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "graph.schema",
        "params": {"tenant_id": 99}
    }))
    .unwrap();
    let resp = https_request(
        harness.addr,
        client,
        "localhost",
        "POST",
        PATH_MCP,
        &[("content-type", "application/json"), (HEADER_TENANT, "99")],
        &body,
    )
    .await;
    assert_eq!(
        resp.status, 403,
        "forged header tenant != bound tenant → transport-level 403"
    );
    let v: Value = serde_json::from_slice(&resp.body).expect("json");
    assert_eq!(v["error"]["code"], -32002, "Unauthorized = -32002");

    harness.shutdown().await;
}
