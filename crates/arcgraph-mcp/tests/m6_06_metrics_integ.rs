//! W15γ M6-06 integration tests — `/metrics` Prometheus exporter.
//!
//! Boots a real `serve_http` instance on `127.0.0.1:0` (mirror of
//! `tests/mcp_http_integ.rs::spawn_harness`), attaches a
//! `MetricsRegistry`, drives traffic through `POST /mcp`, then scrapes
//! `GET /metrics` over the same TLS listener and asserts:
//!
//!   1. The scrape returns 200 with the canonical Prometheus
//!      text-exposition `Content-Type` (`text/plain; version=0.0.4;
//!      charset=utf-8`), the §10.2 metric names this slice ships are
//!      present, and the per-(tenant, tool, status) counter increments
//!      reflect the requests we issued — **all metrics are driven
//!      through the production code path** (the JSON-RPC dispatcher
//!      records via `record_dispatch`; the HTTP accept loop records
//!      via the `ActiveConnGuard` RAII drop). No pre-seeded values.
//!      (`integ_metrics_endpoint_scrape_after_dispatch_returns_expected_text`)
//!   2. `POST /metrics` rejects with `405 Method Not Allowed` —
//!      the endpoint is GET-only per the Prometheus convention.
//!      (`integ_metrics_endpoint_rejects_post_with_405`)
//!   3. When no [`MetricsRegistry`] is attached, `GET /metrics` 404s
//!      — the legacy callers' path.
//!      (`integ_metrics_endpoint_404_when_no_registry_configured`)
//!
//! These tests close the M6-06 partial exit criterion (the three
//! §10.2 metrics this slice wires + the operational
//! `active_connections` gauge). The remaining 5 §10.2 metrics are
//! forward-pinned to M6-07 per the per-metric forward-pin section in
//! `metrics.rs`.
//!
//! # ADR provenance
//!
//! - **roadmap M6-06** (`docs/roadmap.md` line 409): "all metrics in
//!   design-v2 §10.2" — this slice closes the partial; the
//!   producer-wires for the storage / WAL / index / Leiden §10.2
//!   metrics are M6-07.
//! - **design-v2 §10.2** — Observability metric inventory.
//! - **design-v2 §9.4** — HTTP transport surface.

#![allow(clippy::expect_used)]

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
use arcgraph_mcp::transport::http::{HEADER_TENANT, HttpServerConfig, serve_http};
use arcgraph_mcp::{
    CONTENT_TYPE_PROMETHEUS_TEXT, Dispatcher, HybridSearcher, MCPError, MetricsRegistry,
    NeighborhoodExplorer, NodeInspector, PATH_METRICS, SchemaProvider,
};
use arcgraph_query::CancellationToken;
use arcgraph_query::cancel::CancellationRegistry;
use rcgen::{CertificateParams, DnType, KeyPair};
use rustls::pki_types::{CertificateDer, ServerName};
use rustls::{ClientConfig, RootCertStore};
use serde_json::json;
use tempfile::TempDir;
use time::OffsetDateTime;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::oneshot;
use tokio_rustls::TlsConnector;

// ─────────────────────────────────────────────────────────────────────
// Stub providers (mirror mcp_http_integ.rs at minimal surface)
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
                label: None,
                depth: 0,
                properties: BTreeMap::new(),
            }],
            edges: vec![] as Vec<NeighborhoodEdge>,
        })
    }
}

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
            vector: false,
            bm25: false,
        })
    }
    fn search(
        &self,
        tenant: TenantId,
        _query_text: &str,
        _query_vec: Option<&[f32]>,
        _k: u32,
        cancel: &CancellationToken,
    ) -> Result<Vec<SearchHit>, MCPError> {
        if cancel.is_cancelled() {
            return Err(MCPError::Cancelled);
        }
        if tenant != self.0 {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        Ok(vec![])
    }
}

struct StubIngest(TenantId);
impl IngestProvider for StubIngest {
    fn ingest(&self, tenant: TenantId, _batch: IngestBatch) -> Result<IngestSummary, MCPError> {
        if tenant != self.0 {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        Err(MCPError::InternalError(
            "stub ingest not exercised by metrics integ tests".into(),
        ))
    }
}

/// Stub raw-query executor (W16ζ M5-11) — metrics integ tests do not
/// exercise raw_query; the impl exists only to satisfy the W16ζ-merged
/// dispatcher's `RawQueryExecutor` generic.
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
            "stub raw_query not exercised by metrics integ tests".into(),
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
// rcgen-driven cert fixture
// ─────────────────────────────────────────────────────────────────────

struct ServerCertFixture {
    cert_der: CertificateDer<'static>,
}

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
// Test harness
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
    // Settle: poll for connectability (2s budget).
    for _ in 0..40 {
        if TcpStream::connect(assigned).await.is_ok() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
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
// HTTP/1.1 raw client over TLS
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
    let _ = tls.read_to_end(&mut buf).await;
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

fn header_value<'a>(resp: &'a HttpResp, name: &str) -> Option<&'a str> {
    resp.headers
        .iter()
        .find(|(n, _)| n.eq_ignore_ascii_case(name))
        .map(|(_, v)| v.as_str())
}

// ─────────────────────────────────────────────────────────────────────
// Integration test 1 — end-to-end scrape after dispatch
// ─────────────────────────────────────────────────────────────────────

/// Boot the server with a MetricsRegistry, fire two `graph.schema`
/// dispatches (read-class) + one `graph.ingest` dispatch (write-class)
/// against it, then scrape `/metrics`. Verify the response status +
/// content-type + that the per-(tenant, tool, status) counter and
/// per-(tenant, tool) latency histograms reflect the requests we
/// issued — driven entirely through the production code path (no
/// pre-seeded values).
#[tokio::test]
async fn integ_metrics_endpoint_scrape_after_dispatch_returns_expected_text() {
    let (_dir, fx, cert_path, key_path) = stage_filesystem_fixture();
    let provider = Arc::new(arcgraph_mcp::tls::FileSystemCertProvider::new(
        cert_path, key_path, None,
    ));
    let resolver = arcgraph_mcp::tls::HotReloadResolver::new(provider).expect("resolver init");

    let metrics = MetricsRegistry::shared().expect("metrics init");
    // Per W15γ M6-06 fix-up H-3 closure: no pre-seeding. Every label
    // tuple that appears in scrape output is driven by an actual
    // production call-site (dispatcher.record_dispatch on the JSON-RPC
    // path; ActiveConnGuard::drop on the HTTP accept loop's
    // connection lifecycle). A future regression that decouples the
    // producer from the registry fires this test rather than passing
    // on pre-seed alone.

    let cfg = HttpServerConfig::new("127.0.0.1:0".parse().unwrap(), resolver)
        .with_metrics(metrics.clone());
    let harness = spawn_harness(cfg, dispatcher_arc(7)).await;

    let client_cfg = build_client_config(std::slice::from_ref(&fx.cert_der));

    // Fire two graph.schema reads so the tool_invocations counter
    // cell for (tenant=7, tool=graph.schema, status=ok) reaches 2.
    for i in 0..2 {
        let body = serde_json::to_vec(&json!({
            "jsonrpc": "2.0",
            "id": i,
            "method": "graph.schema",
            "params": {"tenant_id": 7}
        }))
        .unwrap();
        let resp = https_request(
            harness.addr,
            client_cfg.clone(),
            "localhost",
            "POST",
            "/mcp",
            &[("content-type", "application/json"), (HEADER_TENANT, "7")],
            &body,
        )
        .await;
        assert_eq!(resp.status, 200, "POST /mcp #{i} must return 200");
    }
    // Fire one graph.ingest write — the stub ingest provider returns
    // an InternalError so this dispatches as (tool=graph.ingest,
    // status=error). The write_latency_ms histogram receives one
    // observation regardless of success/error status.
    {
        let body = serde_json::to_vec(&json!({
            "jsonrpc": "2.0",
            "id": "ing-1",
            "method": "graph.ingest",
            "params": {"tenant_id": 7, "nodes": [{"label": "Person"}]}
        }))
        .unwrap();
        let resp = https_request(
            harness.addr,
            client_cfg.clone(),
            "localhost",
            "POST",
            "/mcp",
            &[("content-type", "application/json"), (HEADER_TENANT, "7")],
            &body,
        )
        .await;
        // The HTTP layer returns 200 with a JSON-RPC error envelope
        // (the dispatch surface returns the error envelope as the
        // response body; the HTTP status is success-shaped).
        assert_eq!(resp.status, 200, "POST /mcp ingest must return 200");
    }

    // Scrape /metrics.
    let scrape = https_request(
        harness.addr,
        client_cfg.clone(),
        "localhost",
        "GET",
        PATH_METRICS,
        &[],
        &[],
    )
    .await;
    assert_eq!(
        scrape.status,
        200,
        "GET /metrics must return 200; body: {:?}",
        String::from_utf8_lossy(&scrape.body)
    );
    assert_eq!(
        header_value(&scrape, "content-type"),
        Some(CONTENT_TYPE_PROMETHEUS_TEXT),
        "Content-Type must carry version qualifier"
    );

    let text = String::from_utf8(scrape.body).expect("scrape body utf-8");
    // The two POST /mcp graph.schema requests dispatched as
    // (tenant=7, tool=graph.schema, status=ok). The counter cell
    // must equal 2 — and it appears ONLY because the dispatcher
    // recorded via the production path.
    assert!(
        text.contains(
            r#"arcgraph_mcp_tool_invocations{status="ok",tenant="7",tool="graph.schema"} 2"#
        ),
        "mcp_tool_invocations(graph.schema, ok) cell must be 2; text was:\n{text}"
    );
    // The one POST /mcp graph.ingest dispatched as
    // (tenant=7, tool=graph.ingest, status=error). The stub provider
    // rejects with InternalError; the dispatch result is the error
    // envelope so the status label is "error".
    assert!(
        text.contains(
            r#"arcgraph_mcp_tool_invocations{status="error",tenant="7",tool="graph.ingest"} 1"#
        ),
        "mcp_tool_invocations(graph.ingest, error) cell must be 1; text was:\n{text}"
    );
    // The read-latency histogram must show 2 observations on the
    // graph.schema cell (the same dispatch that incremented the
    // counter; the production wire records both in one call).
    assert!(
        text.contains("# TYPE arcgraph_read_latency_ms histogram"),
        "read latency histogram TYPE must be present; text was:\n{text}"
    );
    assert!(
        text.contains(r#"arcgraph_read_latency_ms_count{tenant="7",tool="graph.schema"} 2"#),
        "read latency _count must equal 2; text was:\n{text}"
    );
    // The write-latency histogram must show 1 observation on the
    // graph.ingest cell.
    assert!(
        text.contains("# TYPE arcgraph_write_latency_ms histogram"),
        "write latency histogram TYPE must be present; text was:\n{text}"
    );
    assert!(
        text.contains(r#"arcgraph_write_latency_ms_count{tenant="7",tool="graph.ingest"} 1"#),
        "write latency _count must equal 1; text was:\n{text}"
    );
    // The HTTP transport increments the active-connections gauge for
    // each accepted connection. Three POST + one GET = four accepts
    // total, all of which dropped their ActiveConnGuard by the time
    // we scrape (the scrape itself is the only accept still active);
    // the gauge value at scrape time is therefore 1 (the in-flight
    // scrape itself). Note: the gauge ONLY appears in scrape output
    // because the HTTP transport called set_active_connections from
    // production code — no pre-seeding.
    assert!(
        text.contains(r#"arcgraph_active_connections{transport="http"} "#),
        "http active_connections gauge must appear (HTTP transport production wire); text was:\n{text}"
    );

    harness.shutdown().await;
}

// ─────────────────────────────────────────────────────────────────────
// Integration test 2 — POST /metrics rejects 405
// ─────────────────────────────────────────────────────────────────────

/// `/metrics` is GET-only per the Prometheus convention. A POST against
/// the path must reject with 405 Method Not Allowed + Allow: GET.
#[tokio::test]
async fn integ_metrics_endpoint_rejects_post_with_405() {
    let (_dir, fx, cert_path, key_path) = stage_filesystem_fixture();
    let provider = Arc::new(arcgraph_mcp::tls::FileSystemCertProvider::new(
        cert_path, key_path, None,
    ));
    let resolver = arcgraph_mcp::tls::HotReloadResolver::new(provider).expect("resolver init");
    let metrics = MetricsRegistry::shared().expect("metrics init");
    let cfg = HttpServerConfig::new("127.0.0.1:0".parse().unwrap(), resolver).with_metrics(metrics);
    let harness = spawn_harness(cfg, dispatcher_arc(7)).await;

    let client_cfg = build_client_config(std::slice::from_ref(&fx.cert_der));
    let resp = https_request(
        harness.addr,
        client_cfg,
        "localhost",
        "POST",
        PATH_METRICS,
        &[("content-type", "application/json")],
        b"{}",
    )
    .await;
    assert_eq!(
        resp.status, 405,
        "POST /metrics must reject with 405 Method Not Allowed"
    );
    assert_eq!(
        header_value(&resp, "allow"),
        Some("GET"),
        "405 response must carry Allow: GET"
    );

    harness.shutdown().await;
}

// ─────────────────────────────────────────────────────────────────────
// Integration test 3 — GET /metrics returns 404 when not configured
// ─────────────────────────────────────────────────────────────────────

/// `HttpServerConfig::with_metrics` opt-in surface: when the operator
/// does NOT attach a registry, the `/metrics` path falls through to
/// the 404 unknown-path branch. This is the "older callers" path —
/// existing tests in `mcp_http_integ.rs` use this default.
#[tokio::test]
async fn integ_metrics_endpoint_404_when_no_registry_configured() {
    let (_dir, fx, cert_path, key_path) = stage_filesystem_fixture();
    let provider = Arc::new(arcgraph_mcp::tls::FileSystemCertProvider::new(
        cert_path, key_path, None,
    ));
    let resolver = arcgraph_mcp::tls::HotReloadResolver::new(provider).expect("resolver init");
    // NO `with_metrics` call — registry stays None.
    let cfg = HttpServerConfig::new("127.0.0.1:0".parse().unwrap(), resolver);
    let harness = spawn_harness(cfg, dispatcher_arc(7)).await;

    let client_cfg = build_client_config(std::slice::from_ref(&fx.cert_der));
    let resp = https_request(
        harness.addr,
        client_cfg,
        "localhost",
        "GET",
        PATH_METRICS,
        &[],
        &[],
    )
    .await;
    assert_eq!(
        resp.status, 404,
        "GET /metrics must 404 when no MetricsRegistry is configured"
    );

    harness.shutdown().await;
}
