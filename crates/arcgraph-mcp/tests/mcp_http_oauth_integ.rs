//! W16β M5-03 integration tests — OAuth 2.1 + PKCE Bearer-token
//! verification + scope enforcement against a live `serve_http`
//! listener (ADR-044).
//!
//! The test fixture is deterministic and **does not hit the network**:
//! we mint our own EC P-256 keypair via `rcgen` (test dep), stage the
//! public key into the [`OAuthConfig`] JWK Set, sign test JWTs with
//! `jsonwebtoken::EncodingKey::from_ec_pem`, and present them as
//! Bearer tokens on a real HTTPS connection to the spawned listener.
//! ArcGraph never reaches out to an external Authorization Server —
//! all keys are operator-staged per ADR-044 §Decision item 5
//! (HTTP-fetching JWKS is the v1.1 forward-pin).
//!
//! Coverage:
//!   1. **Success** — valid token with sufficient scope → 200 OK +
//!      JSON-RPC success envelope.
//!   2. **Missing Bearer** — no Authorization header → 401 + RFC 6750
//!      §3 `WWW-Authenticate: Bearer realm="arcgraph"` (no error code).
//!   3. **Malformed scheme** — `Authorization: Basic ...` → 401 +
//!      `error="invalid_token"`.
//!   4. **Invalid signature** — token signed by a DIFFERENT key →
//!      401 + `error="invalid_token"`.
//!   5. **Expired token** — `exp` 60s in the past → 401 +
//!      `error="invalid_token"`.
//!   6. **Wrong issuer** — `iss` claim doesn't match config → 401 +
//!      `error="invalid_token"`.
//!   7. **Wrong audience** — `aud` outside config's allow-list →
//!      401 + `error="invalid_token"`.
//!   8. **Insufficient scope** — valid token with `arcgraph.read`
//!      attempts `graph.ingest` → 403 + `error="insufficient_scope"
//!      scope="arcgraph.write"`.
//!   9. **OAuth-disabled passthrough** — `oauth: None` continues to
//!      accept unauthenticated requests (backward-compat with W14α).
//!  10. **End-to-end PKCE-then-token** — exercises the full PKCE
//!      client helper path (verifier → challenge → mock AS exchange)
//!      then drives the resulting token through ArcGraph. This is
//!      the spawn-prompt §1 deliverable line ("end-to-end
//!      token-exchange + scoped-tool-invocation flow").

#![allow(clippy::expect_used)] // tests are allowed to expect; failure context is high-signal here.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use arcgraph_core::TenantId;
use arcgraph_mcp::auth::oauth_pkce::{
    CODE_VERIFIER_DEFAULT_LEN, CodeVerifier, JsonWebKey, JsonWebKeySet, OAuthConfig, SCOPE_READ,
    SCOPE_WRITE, code_challenge_s256, code_verifier_new, validate_code_verifier,
};
use arcgraph_mcp::tools::explore::{Neighborhood, NeighborhoodEdge, NeighborhoodNode};
use arcgraph_mcp::tools::ingest::{IngestBatch, IngestProvider, IngestSummary};
use arcgraph_mcp::tools::inspect::{NeighborDirection, NeighborInfo, NodeInspection};
use arcgraph_mcp::tools::schema::{
    GraphSchema, IndexDescriptor, IndexKind, LabelInfo, RelTypeInfo,
};
use arcgraph_mcp::tools::search::{AvailableSubstrates, SearchHit};
use arcgraph_mcp::transport::http::{HEADER_TENANT, HttpServerConfig, PATH_MCP, serve_http};
use arcgraph_mcp::{
    Dispatcher, HybridSearcher, MCPError, NeighborhoodExplorer, NodeInspector, RawQueryExecutor,
    RawQueryRows, SchemaProvider,
};
use arcgraph_query::CancellationToken;
use arcgraph_query::cancel::CancellationRegistry;
use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
use rcgen::{CertificateParams, DnType, KeyPair};
use rustls::pki_types::{CertificateDer, ServerName};
use rustls::{ClientConfig, RootCertStore};
use serde::Serialize;
use serde_json::{Value, json};
use tempfile::TempDir;
use time::OffsetDateTime;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::oneshot;
use tokio_rustls::TlsConnector;

// ─────────────────────────────────────────────────────────────────────
// Stub providers — same shape as mcp_http_integ.rs. Kept inline here
// so the test compiles without a shared-fixture refactor.
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

/// Stub ingest provider that returns a no-op summary so the
/// integ_oauth_sufficient_write_scope_allows_ingest test can verify
/// a SUCCESS path when the token carries `arcgraph.write`.
struct StubIngest(TenantId);
impl IngestProvider for StubIngest {
    fn ingest(&self, tenant: TenantId, _batch: IngestBatch) -> Result<IngestSummary, MCPError> {
        if tenant != self.0 {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        Ok(IngestSummary {
            records: vec![],
            inserted_count: 0,
            failed_count: 0,
            commit_lsn: None,
            dropped_acl_grants: Vec::new(),
        })
    }
}

/// Stub raw-query executor (W16ζ M5-11) — rejects every query so the
/// OAuth tests focus on transport + auth surface, not query execution.
/// Mirrors the v1.0-α stub in `arcgraph-cli`.
struct StubRawQuery(TenantId);
impl RawQueryExecutor for StubRawQuery {
    fn execute(
        &self,
        tenant: TenantId,
        _query: &str,
        _max_rows: u32,
        cancel: &CancellationToken,
    ) -> Result<RawQueryRows, MCPError> {
        if cancel.is_cancelled() {
            return Err(MCPError::Cancelled);
        }
        if tenant != self.0 {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        Err(MCPError::ExecutionEval(
            "graph.raw_query: not yet wired (test stub)".into(),
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
// TLS cert fixture
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
// JWT signing fixture — ECDSA P-256 (ES256), deterministic on key
// material but not on signatures (ECDSA includes a random `k`).
// ─────────────────────────────────────────────────────────────────────

/// A signing fixture: a P-256 key pair (private PEM + public PEM)
/// plus a `kid` selector for the JWK Set.
struct JwtFixture {
    encoding_key: EncodingKey,
    public_key_pem: String,
    kid: String,
}

impl JwtFixture {
    fn new(kid: &str) -> Self {
        // rcgen 0.14 KeyPair::generate() defaults to ECDSA P-256
        // SHA-256 — the exact algorithm ES256 uses (JWS Alg name).
        let kp = KeyPair::generate().expect("rcgen P-256 keypair");
        let priv_pem = kp.serialize_pem();
        let pub_pem = kp.public_key_pem();
        let encoding_key =
            EncodingKey::from_ec_pem(priv_pem.as_bytes()).expect("ES256 encoding key");
        Self {
            encoding_key,
            public_key_pem: pub_pem,
            kid: kid.to_string(),
        }
    }

    fn as_jwk(&self) -> JsonWebKey {
        let decoding_key = jsonwebtoken::DecodingKey::from_ec_pem(self.public_key_pem.as_bytes())
            .expect("ES256 decoding key");
        JsonWebKey {
            kid: self.kid.clone(),
            algorithm: Algorithm::ES256,
            decoding_key,
        }
    }

    fn mint(&self, claims: &TestClaims) -> String {
        let mut header = Header::new(Algorithm::ES256);
        header.kid = Some(self.kid.clone());
        encode(&header, claims, &self.encoding_key).expect("mint JWT")
    }
}

#[derive(Serialize)]
struct TestClaims {
    iss: String,
    aud: String,
    exp: u64,
    // `skip_serializing_if` is REQUIRED. jsonwebtoken's
    // `ClaimsForValidation` (validation.rs:194) deserializes `nbf`
    // via the `numeric_type` Visitor which only accepts `u64`/`f64`;
    // a JSON `null` falls through to `TryParse::FailedToParse`. With
    // `validate_nbf = true` (R1 HIGH-1 fix), a `nbf: null` claim is
    // rejected with `InvalidClaimFormat("nbf")` even when the test
    // doesn't care about nbf. Omitting the field entirely yields
    // `TryParse::NotPresent` which passes validation.
    #[serde(skip_serializing_if = "Option::is_none")]
    nbf: Option<u64>,
    scope: String,
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

fn make_oauth_config(jwk: JsonWebKey, issuer: &str, audience: &str) -> Arc<OAuthConfig> {
    let jwks = JsonWebKeySet::new(vec![jwk]).expect("non-empty jwks");
    Arc::new(OAuthConfig::new(
        issuer.to_string(),
        vec![audience.to_string()],
        jwks,
    ))
}

// ─────────────────────────────────────────────────────────────────────
// Spawn harness
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

fn header(resp: &HttpResp, name: &str) -> Option<String> {
    resp.headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(name))
        .map(|(_, v)| v.clone())
}

// ─────────────────────────────────────────────────────────────────────
// Common spawn helper — sets up TLS + OAuth + harness in one call.
// ─────────────────────────────────────────────────────────────────────

const ISSUER: &str = "https://idp.example.com";
const AUDIENCE: &str = "arcgraph";
const KID: &str = "test-key-1";

async fn spawn_with_oauth(tenant: u64) -> (HarnessHandle, JwtFixture, Arc<ClientConfig>, TempDir) {
    let (dir, fx, cert_path, key_path) = stage_filesystem_fixture();
    let provider = Arc::new(arcgraph_mcp::tls::FileSystemCertProvider::new(
        cert_path, key_path, None,
    ));
    let resolver = arcgraph_mcp::tls::HotReloadResolver::new(provider).expect("resolver init");

    let jwt_fx = JwtFixture::new(KID);
    let oauth_cfg = make_oauth_config(jwt_fx.as_jwk(), ISSUER, AUDIENCE);

    let cfg = HttpServerConfig::new("127.0.0.1:0".parse().unwrap(), resolver).with_oauth(oauth_cfg);
    let harness = spawn_harness(cfg, dispatcher_arc(tenant)).await;
    let client_cfg = build_client_config(std::slice::from_ref(&fx.cert_der));
    (harness, jwt_fx, client_cfg, dir)
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn integ_oauth_valid_read_token_invokes_schema() {
    let (harness, jwt, client_cfg, _dir) = spawn_with_oauth(7).await;
    let token = jwt.mint(&TestClaims {
        iss: ISSUER.to_string(),
        aud: AUDIENCE.to_string(),
        exp: unix_now() + 3600,
        nbf: None,
        scope: SCOPE_READ.to_string(),
    });
    let body = serde_json::to_vec(&json!({
        "jsonrpc": "2.0", "id": 1, "method": "graph.schema",
        "params": {"tenant_id": 7}
    }))
    .unwrap();
    let auth = format!("Bearer {token}");
    let resp = https_request(
        harness.addr,
        client_cfg,
        "localhost",
        "POST",
        PATH_MCP,
        &[
            ("content-type", "application/json"),
            (HEADER_TENANT, "7"),
            ("authorization", auth.as_str()),
        ],
        &body,
    )
    .await;
    assert_eq!(resp.status, 200, "valid read token must succeed");
    let v: Value = serde_json::from_slice(&resp.body).expect("response JSON");
    assert!(v.get("result").is_some(), "success envelope expected");
    harness.shutdown().await;
}

#[tokio::test]
async fn integ_oauth_missing_bearer_returns_401() {
    let (harness, _jwt, client_cfg, _dir) = spawn_with_oauth(7).await;
    let body = serde_json::to_vec(&json!({
        "jsonrpc": "2.0", "id": 1, "method": "graph.schema",
        "params": {"tenant_id": 7}
    }))
    .unwrap();
    let resp = https_request(
        harness.addr,
        client_cfg,
        "localhost",
        "POST",
        PATH_MCP,
        &[
            ("content-type", "application/json"),
            (HEADER_TENANT, "7"),
            // No Authorization header.
        ],
        &body,
    )
    .await;
    assert_eq!(resp.status, 401, "missing Bearer must reject 401");
    let www = header(&resp, "www-authenticate").expect("WWW-Authenticate header");
    assert_eq!(
        www, "Bearer realm=\"arcgraph\"",
        "RFC 6750 §3 — no error code when no auth attempted"
    );
    let v: Value = serde_json::from_slice(&resp.body).expect("response JSON");
    // -32002 Unauthorized per error.rs CODE_UNAUTHORIZED.
    assert_eq!(v["error"]["code"], -32002);
    harness.shutdown().await;
}

#[tokio::test]
async fn integ_oauth_malformed_scheme_returns_401_invalid_token() {
    let (harness, _jwt, client_cfg, _dir) = spawn_with_oauth(7).await;
    let body = serde_json::to_vec(&json!({
        "jsonrpc": "2.0", "id": 1, "method": "graph.schema",
        "params": {"tenant_id": 7}
    }))
    .unwrap();
    let resp = https_request(
        harness.addr,
        client_cfg,
        "localhost",
        "POST",
        PATH_MCP,
        &[
            ("content-type", "application/json"),
            (HEADER_TENANT, "7"),
            ("authorization", "Basic dXNlcjpwYXNz"),
        ],
        &body,
    )
    .await;
    assert_eq!(resp.status, 401);
    let www = header(&resp, "www-authenticate").expect("WWW-Authenticate header");
    assert!(
        www.contains("error=\"invalid_token\""),
        "WWW-Authenticate should signal invalid_token, got: {www}"
    );
    harness.shutdown().await;
}

#[tokio::test]
async fn integ_oauth_invalid_signature_returns_401() {
    let (harness, _jwt, client_cfg, _dir) = spawn_with_oauth(7).await;
    // Mint a token signed by a DIFFERENT key. The server's JWK Set
    // doesn't carry this key's pubkey → signature verify fails.
    let attacker = JwtFixture::new(KID); // same kid, different key material
    let token = attacker.mint(&TestClaims {
        iss: ISSUER.to_string(),
        aud: AUDIENCE.to_string(),
        exp: unix_now() + 3600,
        nbf: None,
        scope: SCOPE_READ.to_string(),
    });
    let body = serde_json::to_vec(&json!({
        "jsonrpc": "2.0", "id": 1, "method": "graph.schema",
        "params": {"tenant_id": 7}
    }))
    .unwrap();
    let auth = format!("Bearer {token}");
    let resp = https_request(
        harness.addr,
        client_cfg,
        "localhost",
        "POST",
        PATH_MCP,
        &[
            ("content-type", "application/json"),
            (HEADER_TENANT, "7"),
            ("authorization", auth.as_str()),
        ],
        &body,
    )
    .await;
    assert_eq!(resp.status, 401, "wrong-key signature must reject");
    let www = header(&resp, "www-authenticate").expect("WWW-Authenticate header");
    assert!(www.contains("error=\"invalid_token\""), "got: {www}");
    harness.shutdown().await;
}

#[tokio::test]
async fn integ_oauth_expired_token_returns_401() {
    let (harness, jwt, client_cfg, _dir) = spawn_with_oauth(7).await;
    let token = jwt.mint(&TestClaims {
        iss: ISSUER.to_string(),
        aud: AUDIENCE.to_string(),
        // exp 5 minutes in the past — well past the 30s default skew.
        exp: unix_now() - 300,
        nbf: None,
        scope: SCOPE_READ.to_string(),
    });
    let body = serde_json::to_vec(&json!({
        "jsonrpc": "2.0", "id": 1, "method": "graph.schema",
        "params": {"tenant_id": 7}
    }))
    .unwrap();
    let auth = format!("Bearer {token}");
    let resp = https_request(
        harness.addr,
        client_cfg,
        "localhost",
        "POST",
        PATH_MCP,
        &[
            ("content-type", "application/json"),
            (HEADER_TENANT, "7"),
            ("authorization", auth.as_str()),
        ],
        &body,
    )
    .await;
    assert_eq!(resp.status, 401);
    let www = header(&resp, "www-authenticate").expect("WWW-Authenticate header");
    assert!(www.contains("error=\"invalid_token\""), "got: {www}");
    harness.shutdown().await;
}

#[tokio::test]
async fn integ_oauth_wrong_issuer_returns_401() {
    let (harness, jwt, client_cfg, _dir) = spawn_with_oauth(7).await;
    let token = jwt.mint(&TestClaims {
        iss: "https://evil.example.com".to_string(),
        aud: AUDIENCE.to_string(),
        exp: unix_now() + 3600,
        nbf: None,
        scope: SCOPE_READ.to_string(),
    });
    let body = serde_json::to_vec(&json!({
        "jsonrpc": "2.0", "id": 1, "method": "graph.schema",
        "params": {"tenant_id": 7}
    }))
    .unwrap();
    let auth = format!("Bearer {token}");
    let resp = https_request(
        harness.addr,
        client_cfg,
        "localhost",
        "POST",
        PATH_MCP,
        &[
            ("content-type", "application/json"),
            (HEADER_TENANT, "7"),
            ("authorization", auth.as_str()),
        ],
        &body,
    )
    .await;
    assert_eq!(resp.status, 401, "wrong issuer must reject");
    harness.shutdown().await;
}

#[tokio::test]
async fn integ_oauth_wrong_audience_returns_401() {
    let (harness, jwt, client_cfg, _dir) = spawn_with_oauth(7).await;
    let token = jwt.mint(&TestClaims {
        iss: ISSUER.to_string(),
        aud: "different-service".to_string(),
        exp: unix_now() + 3600,
        nbf: None,
        scope: SCOPE_READ.to_string(),
    });
    let body = serde_json::to_vec(&json!({
        "jsonrpc": "2.0", "id": 1, "method": "graph.schema",
        "params": {"tenant_id": 7}
    }))
    .unwrap();
    let auth = format!("Bearer {token}");
    let resp = https_request(
        harness.addr,
        client_cfg,
        "localhost",
        "POST",
        PATH_MCP,
        &[
            ("content-type", "application/json"),
            (HEADER_TENANT, "7"),
            ("authorization", auth.as_str()),
        ],
        &body,
    )
    .await;
    assert_eq!(resp.status, 401, "wrong audience must reject");
    harness.shutdown().await;
}

#[tokio::test]
async fn integ_oauth_insufficient_scope_returns_403() {
    let (harness, jwt, client_cfg, _dir) = spawn_with_oauth(7).await;
    // Token carries only `arcgraph.read`, but `graph.ingest`
    // requires `arcgraph.write` per ADR-044 §Decision item 6.
    let token = jwt.mint(&TestClaims {
        iss: ISSUER.to_string(),
        aud: AUDIENCE.to_string(),
        exp: unix_now() + 3600,
        nbf: None,
        scope: SCOPE_READ.to_string(),
    });
    let body = serde_json::to_vec(&json!({
        "jsonrpc": "2.0", "id": 2, "method": "graph.ingest",
        "params": {
            "tenant_id": 7,
            "nodes": [],
            "relationships": []
        }
    }))
    .unwrap();
    let auth = format!("Bearer {token}");
    let resp = https_request(
        harness.addr,
        client_cfg,
        "localhost",
        "POST",
        PATH_MCP,
        &[
            ("content-type", "application/json"),
            (HEADER_TENANT, "7"),
            ("authorization", auth.as_str()),
        ],
        &body,
    )
    .await;
    assert_eq!(resp.status, 403, "read-only token can't write");
    let www = header(&resp, "www-authenticate").expect("WWW-Authenticate header");
    assert!(www.contains("error=\"insufficient_scope\""), "got: {www}");
    assert!(
        www.contains(SCOPE_WRITE),
        "scope hint should name write: {www}"
    );
    harness.shutdown().await;
}

#[tokio::test]
async fn integ_oauth_sufficient_write_scope_allows_ingest() {
    let (harness, jwt, client_cfg, _dir) = spawn_with_oauth(7).await;
    // Token carries BOTH read + write — write is required for
    // `graph.ingest`, and exercises the multi-scope path through
    // `parse_scope_claim`.
    let token = jwt.mint(&TestClaims {
        iss: ISSUER.to_string(),
        aud: AUDIENCE.to_string(),
        exp: unix_now() + 3600,
        nbf: None,
        scope: format!("{SCOPE_READ} {SCOPE_WRITE}"),
    });
    let body = serde_json::to_vec(&json!({
        "jsonrpc": "2.0", "id": 3, "method": "graph.ingest",
        "params": {
            "tenant_id": 7,
            "nodes": [],
            "relationships": []
        }
    }))
    .unwrap();
    let auth = format!("Bearer {token}");
    let resp = https_request(
        harness.addr,
        client_cfg,
        "localhost",
        "POST",
        PATH_MCP,
        &[
            ("content-type", "application/json"),
            (HEADER_TENANT, "7"),
            ("authorization", auth.as_str()),
        ],
        &body,
    )
    .await;
    assert_eq!(
        resp.status, 200,
        "read+write token must successfully ingest"
    );
    let v: Value = serde_json::from_slice(&resp.body).expect("response JSON");
    assert!(v.get("result").is_some(), "ingest success envelope");
    harness.shutdown().await;
}

#[tokio::test]
async fn integ_oauth_disabled_passthrough_unchanged_from_w14a() {
    // Backward-compat: oauth: None means no Bearer required, no scope
    // enforced. This is the W14α HTTP/TLS transport behavior — and
    // the existing mcp_http_integ.rs tests rely on it.
    let (_dir, fx, cert_path, key_path) = stage_filesystem_fixture();
    let provider = Arc::new(arcgraph_mcp::tls::FileSystemCertProvider::new(
        cert_path, key_path, None,
    ));
    let resolver = arcgraph_mcp::tls::HotReloadResolver::new(provider).expect("resolver init");
    let cfg = HttpServerConfig::new("127.0.0.1:0".parse().unwrap(), resolver);
    let harness = spawn_harness(cfg, dispatcher_arc(7)).await;
    let client_cfg = build_client_config(std::slice::from_ref(&fx.cert_der));

    let body = serde_json::to_vec(&json!({
        "jsonrpc": "2.0", "id": 1, "method": "graph.schema",
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
    assert_eq!(resp.status, 200, "oauth: None passes unauthenticated");
    harness.shutdown().await;
}

#[tokio::test]
async fn integ_oauth_pkce_end_to_end_flow() {
    // End-to-end PKCE-then-token flow (deterministic; no network).
    //
    // 1. CLIENT: generate code_verifier per RFC 7636 §4.1.
    // 2. CLIENT: derive S256 code_challenge per RFC 7636 §4.2.
    // 3. MOCK AS: would receive (code_challenge, code_challenge_method)
    //    on the authorization request, store them keyed by an
    //    authorization code.
    // 4. CLIENT: after redirect with the auth code, exchange it +
    //    code_verifier at the token endpoint.
    // 5. MOCK AS: re-derives S256(code_verifier), compares against
    //    the stored challenge; if equal, issues a signed JWT.
    // 6. CLIENT: presents the JWT as Bearer to ArcGraph.
    // 7. ARCGRAPH: verifies the JWT against its operator-staged
    //    JWK Set; dispatches.
    //
    // Steps 3-5 are mocked in-process below — we don't need a real
    // HTTP AS; we just exercise the verifier ↔ challenge equivalence
    // and then mint the resulting JWT against the same keypair.
    let (harness, jwt, client_cfg, _dir) = spawn_with_oauth(7).await;

    let verifier = code_verifier_new();
    assert_eq!(verifier.as_str().len(), CODE_VERIFIER_DEFAULT_LEN);
    validate_code_verifier(verifier.as_str()).expect("RFC 7636 §4.1 conformant");
    let challenge = code_challenge_s256(&verifier);
    // The MOCK AS-side: would have stored `challenge`; here we re-
    // derive from the verifier-string-on-wire (the canonical AS
    // roundtrip during token exchange per RFC 7636 §4.5-§4.6).
    let verifier_wire = verifier.as_str().to_string();
    let reconstructed = CodeVerifier::from_string(verifier_wire).expect("valid verifier");
    let re_derived = code_challenge_s256(&reconstructed);
    assert_eq!(re_derived, challenge, "S256 transform is deterministic");

    // The MOCK AS now mints the access token (skipping the redirect
    // hop; the test exercises the verify path on the ArcGraph side).
    let token = jwt.mint(&TestClaims {
        iss: ISSUER.to_string(),
        aud: AUDIENCE.to_string(),
        exp: unix_now() + 3600,
        nbf: None,
        scope: SCOPE_READ.to_string(),
    });

    let body = serde_json::to_vec(&json!({
        "jsonrpc": "2.0", "id": 1, "method": "graph.schema",
        "params": {"tenant_id": 7}
    }))
    .unwrap();
    let auth = format!("Bearer {token}");
    let resp = https_request(
        harness.addr,
        client_cfg,
        "localhost",
        "POST",
        PATH_MCP,
        &[
            ("content-type", "application/json"),
            (HEADER_TENANT, "7"),
            ("authorization", auth.as_str()),
        ],
        &body,
    )
    .await;
    assert_eq!(resp.status, 200, "PKCE-then-token flow must end in success");
    harness.shutdown().await;
}

// ─────────────────────────────────────────────────────────────────────
// R1 fix-up integ tests — MED-3 edge-case coverage + HIGH-1 regression.
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn integ_oauth_nbf_in_future_returns_401() {
    // R1 HIGH-1 regression: a token whose `nbf` claim is in the
    // future MUST NOT be accepted before its activation time per RFC
    // 7519 §4.1.5. Pre-fix the verifier ignored `nbf` (jsonwebtoken's
    // default `validate_nbf: false`); the fix opts validation in and
    // this test guards against future regression.
    let (harness, jwt, client_cfg, _dir) = spawn_with_oauth(7).await;
    let now = unix_now();
    let token = jwt.mint(&TestClaims {
        iss: ISSUER.to_string(),
        aud: AUDIENCE.to_string(),
        // exp in the future so we know we're not tripping on exp
        // (1h forward); nbf 1h in the future (well past 30s skew).
        exp: now + 7200,
        nbf: Some(now + 3600),
        scope: SCOPE_READ.to_string(),
    });
    let body = serde_json::to_vec(&json!({
        "jsonrpc": "2.0", "id": 1, "method": "graph.schema",
        "params": {"tenant_id": 7}
    }))
    .unwrap();
    let auth = format!("Bearer {token}");
    let resp = https_request(
        harness.addr,
        client_cfg,
        "localhost",
        "POST",
        PATH_MCP,
        &[
            ("content-type", "application/json"),
            (HEADER_TENANT, "7"),
            ("authorization", auth.as_str()),
        ],
        &body,
    )
    .await;
    assert_eq!(
        resp.status, 401,
        "RFC 7519 §4.1.5: token with nbf in future MUST NOT be accepted"
    );
    let www = header(&resp, "www-authenticate").expect("WWW-Authenticate header");
    assert!(
        www.contains("error=\"invalid_token\""),
        "nbf-in-future surfaces as invalid_token, got: {www}"
    );
    let v: Value = serde_json::from_slice(&resp.body).expect("response JSON");
    assert_eq!(v["error"]["code"], -32002);
    harness.shutdown().await;
}

#[tokio::test]
async fn integ_oauth_wrong_kid_returns_401() {
    // MED-3 item 3: a token whose `kid` header doesn't match any
    // entry in the JWK Set is rejected at resolve-time (the JWK
    // resolver returns None → InvalidToken).
    let (harness, jwt, client_cfg, _dir) = spawn_with_oauth(7).await;
    // Mint a token whose `kid` is forced to a value NOT in the JWK Set.
    let claims = TestClaims {
        iss: ISSUER.to_string(),
        aud: AUDIENCE.to_string(),
        exp: unix_now() + 3600,
        nbf: None,
        scope: SCOPE_READ.to_string(),
    };
    let mut jwt_header = Header::new(Algorithm::ES256);
    jwt_header.kid = Some("does-not-exist".to_string());
    let token = encode(&jwt_header, &claims, &jwt.encoding_key).expect("mint with bogus kid");
    let body = serde_json::to_vec(&json!({
        "jsonrpc": "2.0", "id": 1, "method": "graph.schema",
        "params": {"tenant_id": 7}
    }))
    .unwrap();
    let auth = format!("Bearer {token}");
    let resp = https_request(
        harness.addr,
        client_cfg,
        "localhost",
        "POST",
        PATH_MCP,
        &[
            ("content-type", "application/json"),
            (HEADER_TENANT, "7"),
            ("authorization", auth.as_str()),
        ],
        &body,
    )
    .await;
    assert_eq!(resp.status, 401, "unknown kid must reject");
    let www = header(&resp, "www-authenticate").expect("WWW-Authenticate header");
    assert!(www.contains("error=\"invalid_token\""), "got: {www}");
    harness.shutdown().await;
}

#[tokio::test]
async fn integ_oauth_alg_confusion_returns_401() {
    // MED-3 item 4: a token whose `alg` header matches the
    // `required_algorithms` whitelist but whose `kid` resolves to a
    // JWK with a DIFFERENT `algorithm` (header.alg != jwk.algorithm)
    // must be rejected — defense-in-depth against the classic
    // algorithm-confusion vector. Stage a JWK that claims RS256 but
    // wraps the actual ES256 decoding key; mint with header.alg=ES256
    // and the matching kid; the verifier's line-752 check catches
    // the mismatch.
    let (_dir, fx, cert_path, key_path) = stage_filesystem_fixture();
    let provider = Arc::new(arcgraph_mcp::tls::FileSystemCertProvider::new(
        cert_path, key_path, None,
    ));
    let resolver = arcgraph_mcp::tls::HotReloadResolver::new(provider).expect("resolver init");

    let jwt_fx = JwtFixture::new(KID);
    // Stage the JWK with algorithm=RS256, despite the underlying
    // decoding_key being EC-PEM. This intentionally misrepresents the
    // algorithm so the verifier hits its alg-mismatch defense.
    let bogus_jwk = JsonWebKey {
        kid: KID.to_string(),
        algorithm: Algorithm::RS256,
        decoding_key: jsonwebtoken::DecodingKey::from_ec_pem(jwt_fx.public_key_pem.as_bytes())
            .expect("ES256 decoding key wrapped under RS256 alg"),
    };
    let jwks = JsonWebKeySet::new(vec![bogus_jwk]).expect("non-empty");
    let oauth_cfg = Arc::new(OAuthConfig::new(
        ISSUER.to_string(),
        vec![AUDIENCE.to_string()],
        jwks,
    ));
    let cfg = HttpServerConfig::new("127.0.0.1:0".parse().unwrap(), resolver).with_oauth(oauth_cfg);
    let harness = spawn_harness(cfg, dispatcher_arc(7)).await;
    let client_cfg = build_client_config(std::slice::from_ref(&fx.cert_der));

    let token = jwt_fx.mint(&TestClaims {
        iss: ISSUER.to_string(),
        aud: AUDIENCE.to_string(),
        exp: unix_now() + 3600,
        nbf: None,
        scope: SCOPE_READ.to_string(),
    });
    let body = serde_json::to_vec(&json!({
        "jsonrpc": "2.0", "id": 1, "method": "graph.schema",
        "params": {"tenant_id": 7}
    }))
    .unwrap();
    let auth = format!("Bearer {token}");
    let resp = https_request(
        harness.addr,
        client_cfg,
        "localhost",
        "POST",
        PATH_MCP,
        &[
            ("content-type", "application/json"),
            (HEADER_TENANT, "7"),
            ("authorization", auth.as_str()),
        ],
        &body,
    )
    .await;
    assert_eq!(
        resp.status, 401,
        "alg-confusion (header.alg != jwk.algorithm) must reject"
    );
    let www = header(&resp, "www-authenticate").expect("WWW-Authenticate header");
    assert!(www.contains("error=\"invalid_token\""), "got: {www}");
    harness.shutdown().await;
}

#[tokio::test]
async fn integ_oauth_unknown_alg_returns_401() {
    // MED-3 gap-2: a token whose `alg` header is OUTSIDE the
    // operator-staged whitelist (e.g. HS256 — symmetric, expressly
    // excluded by the default whitelist + with_required_algorithms
    // panic guard) is rejected at the alg-whitelist check (line 739)
    // BEFORE the signature verify runs. Mint with HS256 + a shared
    // secret; the server's default `required_algorithms` excludes
    // HS256 so the check fires before we ever attempt verification.
    let (harness, _jwt, client_cfg, _dir) = spawn_with_oauth(7).await;
    let claims = TestClaims {
        iss: ISSUER.to_string(),
        aud: AUDIENCE.to_string(),
        exp: unix_now() + 3600,
        nbf: None,
        scope: SCOPE_READ.to_string(),
    };
    let mut jwt_header = Header::new(Algorithm::HS256);
    jwt_header.kid = Some(KID.to_string());
    let token = encode(
        &jwt_header,
        &claims,
        &EncodingKey::from_secret(b"any-shared-secret"),
    )
    .expect("HS256 mint");
    let body = serde_json::to_vec(&json!({
        "jsonrpc": "2.0", "id": 1, "method": "graph.schema",
        "params": {"tenant_id": 7}
    }))
    .unwrap();
    let auth = format!("Bearer {token}");
    let resp = https_request(
        harness.addr,
        client_cfg,
        "localhost",
        "POST",
        PATH_MCP,
        &[
            ("content-type", "application/json"),
            (HEADER_TENANT, "7"),
            ("authorization", auth.as_str()),
        ],
        &body,
    )
    .await;
    assert_eq!(
        resp.status, 401,
        "HS256 outside the asymmetric-only whitelist must reject"
    );
    let www = header(&resp, "www-authenticate").expect("WWW-Authenticate header");
    assert!(www.contains("error=\"invalid_token\""), "got: {www}");
    harness.shutdown().await;
}
