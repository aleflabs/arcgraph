//! #761 slice 1 — `serve --http` TLS end-to-end (ADR-133 §D-4 "Driver").
//!
//! Spawns the real `arcgraph` binary as a subprocess (the cargo-set
//! `CARGO_BIN_EXE_arcgraph`) with a staged `rcgen` self-signed cert/key,
//! then drives a real `tokio_rustls` TLS client through one MCP
//! JSON-RPC roundtrip over `POST /mcp` and asserts a well-formed success
//! envelope. This PROVES the CLI cert/key flags flow through
//! `FileSystemCertProvider` → `HotReloadResolver` → live TLS handshake →
//! MCP dispatch — no mocks where a real wire path is feasible (the spawn
//! prompt's e2e bar + `feedback_load_bearing_pr_requires_fault_injection_tests`).
//!
//! Plus two fault-injection regressions on the HTTP serve path
//! (≥1 per failure mode), each driving the actual binary:
//!   - **ADR-183 durable-by-default:** `--http` with neither `--data`
//!     nor `--in-memory` refuses to start — a GA server never silently
//!     comes up non-durable.
//!   - **design-v2 §9.4 line 668 + W14 retro IR L1-HIGH-4:** a
//!     non-loopback `--http` bind without `--allow-remote-http-bind`
//!     refuses to start (loopback-default). RED-on-revert: removing the
//!     bind gate lets `0.0.0.0` bind silently and this test hangs→fails.
//!
//! The clap-surface fault cases (`--http` without cert/key; tls flags on
//! a non-HTTP transport) + the `validate_http_bind` / `build_http_tls_resolver`
//! helper faults are covered by the in-process unit tests in
//! `src/bin/arcgraph.rs` (`#[cfg(test)] mod tests`).

#![allow(clippy::expect_used)] // tests are allowed to expect; high-signal failure context.

use std::net::SocketAddr;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use arcgraph_mcp::{HEADER_TENANT, PATH_MCP};
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
use tokio::process::{Child, Command};
use tokio::time::timeout;

/// How long to wait for the subprocess to bootstrap its in-memory store +
/// load the cert + bind the listener before giving up.
const STARTUP_BUDGET: Duration = Duration::from_secs(20);
const OAUTH_ISSUER: &str = "https://auth.example.test/";
const OAUTH_AUDIENCE: &str = "arcgraph-http-test";
const OAUTH_KID: &str = "http-test-key-1";

/// Grab a free loopback TCP port by binding `:0`, reading the assigned
/// port, then dropping the listener. A tiny TOCTOU window exists before
/// the subprocess rebinds the same port; acceptable for a hermetic test
/// (the standard subprocess-port idiom).
fn free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind :0 for free port");
    let port = listener.local_addr().expect("local_addr").port();
    drop(listener);
    port
}

/// A staged on-disk cert fixture: SAN=localhost self-signed cert + matching
/// key written as PEM, plus the cert DER for the client's trust store.
struct CertFixture {
    _dir: TempDir, // held so the tempdir (and its files) outlive the server.
    cert_path: std::path::PathBuf,
    key_path: std::path::PathBuf,
    cert_der: CertificateDer<'static>,
}

/// Stage a self-signed cert (CN/SAN = `localhost`) + matching private key
/// into a fresh tempdir as PEM files the binary's `FileSystemCertProvider`
/// reads. Mirrors `crates/arcgraph-mcp/tests/mcp_http_integ.rs`'s
/// `stage_filesystem_fixture`.
fn stage_cert_fixture() -> CertFixture {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut params = CertificateParams::new(vec!["localhost".to_string()]).expect("rcgen params");
    params.not_before = OffsetDateTime::now_utc() - time::Duration::days(1);
    params.not_after = OffsetDateTime::now_utc() + time::Duration::days(365);
    params
        .distinguished_name
        .push(DnType::CommonName, "localhost");
    let kp = KeyPair::generate().expect("rcgen keypair");
    let cert = params.self_signed(&kp).expect("self-signed");
    let cert_path = dir.path().join("server.crt");
    let key_path = dir.path().join("server.key");
    std::fs::write(&cert_path, cert.pem()).expect("write cert pem");
    std::fs::write(&key_path, kp.serialize_pem()).expect("write key pem");
    let cert_der = CertificateDer::from(cert.der().as_ref().to_vec());
    CertFixture {
        _dir: dir,
        cert_path,
        key_path,
        cert_der,
    }
}

/// Build a `tokio_rustls` client config whose trust store contains exactly
/// the staged self-signed cert, so the handshake succeeds without a
/// WebPki unknown-issuer error.
fn build_client_config(trusted: &CertificateDer<'static>) -> Arc<ClientConfig> {
    let mut roots = RootCertStore::empty();
    roots.add(trusted.clone()).expect("add trusted root");
    let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
    let cfg = ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .expect("protocol versions")
        .with_root_certificates(roots)
        .with_no_client_auth();
    Arc::new(cfg)
}

struct JwtFixture {
    encoding_key: EncodingKey,
    jwks_json: String,
}

impl JwtFixture {
    fn new() -> Self {
        let kp = KeyPair::generate().expect("rcgen P-256 keypair");
        let priv_pem = kp.serialize_pem();
        let encoding_key =
            EncodingKey::from_ec_pem(priv_pem.as_bytes()).expect("ES256 encoding key");
        let (x, y) = p256_xy_from_raw_point(kp.public_key_raw());
        let jwks_json = json!({
            "keys": [{
                "kty": "EC",
                "crv": "P-256",
                "x": base64url_no_pad(&x),
                "y": base64url_no_pad(&y),
                "use": "sig",
                "alg": "ES256",
                "kid": OAUTH_KID,
            }]
        })
        .to_string();
        Self {
            encoding_key,
            jwks_json,
        }
    }

    fn mint(&self, scope: &str, exp_offset_secs: i64) -> String {
        #[derive(Serialize)]
        struct Claims<'a> {
            iss: &'a str,
            aud: &'a str,
            exp: u64,
            scope: &'a str,
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let exp = if exp_offset_secs >= 0 {
            now + exp_offset_secs as u64
        } else {
            now.saturating_sub(exp_offset_secs.unsigned_abs())
        };
        let claims = Claims {
            iss: OAUTH_ISSUER,
            aud: OAUTH_AUDIENCE,
            exp,
            scope,
        };
        let mut header = Header::new(Algorithm::ES256);
        header.kid = Some(OAUTH_KID.to_string());
        encode(&header, &claims, &self.encoding_key).expect("mint JWT")
    }
}

fn p256_xy_from_raw_point(point: &[u8]) -> ([u8; 32], [u8; 32]) {
    assert_eq!(
        point.len(),
        65,
        "P-256 uncompressed point must carry 0x04 + 32-byte x + 32-byte y",
    );
    assert_eq!(point[0], 0x04, "P-256 point must be uncompressed");
    let mut x = [0u8; 32];
    let mut y = [0u8; 32];
    x.copy_from_slice(&point[1..33]);
    y.copy_from_slice(&point[33..65]);
    (x, y)
}

fn base64url_no_pad(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::with_capacity((bytes.len() * 4).div_ceil(3));
    let mut chunks = bytes.chunks_exact(3);
    for chunk in &mut chunks {
        let n = ((chunk[0] as u32) << 16) | ((chunk[1] as u32) << 8) | chunk[2] as u32;
        out.push(TABLE[((n >> 18) & 0x3f) as usize] as char);
        out.push(TABLE[((n >> 12) & 0x3f) as usize] as char);
        out.push(TABLE[((n >> 6) & 0x3f) as usize] as char);
        out.push(TABLE[(n & 0x3f) as usize] as char);
    }
    let rem = chunks.remainder();
    if rem.len() == 1 {
        let n = (rem[0] as u32) << 16;
        out.push(TABLE[((n >> 18) & 0x3f) as usize] as char);
        out.push(TABLE[((n >> 12) & 0x3f) as usize] as char);
    } else if rem.len() == 2 {
        let n = ((rem[0] as u32) << 16) | ((rem[1] as u32) << 8);
        out.push(TABLE[((n >> 18) & 0x3f) as usize] as char);
        out.push(TABLE[((n >> 12) & 0x3f) as usize] as char);
        out.push(TABLE[((n >> 6) & 0x3f) as usize] as char);
    }
    out
}

struct HttpResp {
    status: u16,
    body: Vec<u8>,
}

/// Do one `POST /mcp` over a real TLS connection and return the parsed
/// status + body. Handcrafted HTTP/1.1 (`Connection: close`) so the peer
/// closes after one response — mirrors the mcp-crate integ harness.
async fn https_post_mcp(
    addr: SocketAddr,
    client_cfg: Arc<ClientConfig>,
    headers: &[(&str, &str)],
    body: &[u8],
) -> HttpResp {
    use tokio_rustls::TlsConnector;
    let tcp = TcpStream::connect(addr).await.expect("tcp connect");
    let connector = TlsConnector::from(client_cfg);
    let dnsname = ServerName::try_from("localhost".to_owned()).expect("dns name");
    let mut tls = connector
        .connect(dnsname, tcp)
        .await
        .expect("TLS handshake (cert/key flowed through FileSystemCertProvider → resolver)");

    let mut req = format!("POST {PATH_MCP} HTTP/1.1\r\nHost: localhost\r\n");
    for (name, value) in headers {
        req.push_str(&format!("{name}: {value}\r\n"));
    }
    req.push_str(&format!("Content-Length: {}\r\n", body.len()));
    req.push_str("Connection: close\r\n\r\n");
    tls.write_all(req.as_bytes()).await.expect("write headers");
    tls.write_all(body).await.expect("write body");
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
    assert!(
        head_end > 0,
        "no header terminator in TLS response ({} bytes)",
        bytes.len()
    );
    let header_block = std::str::from_utf8(&bytes[..head_end]).expect("response headers utf-8");
    let status_line = header_block.split("\r\n").next().unwrap_or("");
    let code: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    HttpResp {
        status: code,
        body: bytes[head_end + 4..].to_vec(),
    }
}

/// Poll TCP-connect until the subprocess listener is up. Fails fast (with
/// the child's captured stderr) if the child exited during startup.
async fn await_listener(child: &mut Child, addr: SocketAddr) {
    let deadline = tokio::time::Instant::now() + STARTUP_BUDGET;
    loop {
        if let Some(status) = child.try_wait().expect("try_wait") {
            let stderr = drain_stderr(child).await;
            panic!("server exited early (status {status}) before listening:\n{stderr}");
        }
        if TcpStream::connect(addr).await.is_ok() {
            return;
        }
        if tokio::time::Instant::now() >= deadline {
            let stderr = drain_stderr(child).await;
            panic!("server did not listen within {STARTUP_BUDGET:?}:\n{stderr}");
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

async fn drain_stderr(child: &mut Child) -> String {
    let mut s = String::new();
    if let Some(mut err) = child.stderr.take() {
        let _ = err.read_to_string(&mut s).await;
    }
    s
}

// ─────────────────────────────────────────────────────────────────────
// ADR-133 §D-4 "Driver" e2e — the headline active-verification test.
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn serve_http_tls_e2e_roundtrip() {
    let bin = env!("CARGO_BIN_EXE_arcgraph");
    let fx = stage_cert_fixture();
    let addr: SocketAddr = format!("127.0.0.1:{}", free_port()).parse().unwrap();

    let mut child = Command::new(bin)
        .args([
            "serve",
            "--http",
            &addr.to_string(),
            "--tls-cert",
            fx.cert_path.to_str().unwrap(),
            "--tls-key",
            fx.key_path.to_str().unwrap(),
            // Hermetic: in-memory store (no disk durability needed to
            // prove the TLS wiring); admin + audit disabled so the test
            // doesn't contend for the default admin port / write to CWD.
            "--in-memory",
            "--admin-http",
            "",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .expect("spawn arcgraph serve --http");

    await_listener(&mut child, addr).await;

    // The dispatcher is pinned to TenantId::DEFAULT (raw = 1); the
    // transport's bound-tenant fence requires the header + payload tenant
    // to match it.
    let client = build_client_config(&fx.cert_der);
    let body = serde_json::to_vec(&json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "graph.schema",
        "params": { "tenant_id": 1 }
    }))
    .unwrap();

    let resp = timeout(
        Duration::from_secs(10),
        https_post_mcp(
            addr,
            client,
            &[("content-type", "application/json"), (HEADER_TENANT, "1")],
            &body,
        ),
    )
    .await
    .expect("TLS roundtrip completed within budget");

    // Oracle: a well-formed MCP success envelope over TLS.
    assert_eq!(resp.status, 200, "POST /mcp over TLS must return 200 OK");
    let v: Value = serde_json::from_slice(&resp.body).expect("response body is JSON");
    assert_eq!(v["id"], 1, "JSON-RPC id echoed");
    assert!(
        v.get("result").is_some() && v.get("error").is_none(),
        "well-formed MCP success envelope (no error): {v}"
    );

    child.kill().await.expect("kill server");
}

// ─────────────────────────────────────────────────────────────────────
// #761 slice 3 — CLI HTTP OAuth wire e2e.
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn serve_http_oauth_bearer_auth_e2e() {
    let bin = env!("CARGO_BIN_EXE_arcgraph");
    let fx = stage_cert_fixture();
    let jwt = JwtFixture::new();
    let jwt_dir = tempfile::tempdir().expect("jwt tempdir");
    let jwks_path = jwt_dir.path().join("jwks.json");
    std::fs::write(&jwks_path, &jwt.jwks_json).expect("write JWKS JSON");
    let valid_token = jwt.mint("arcgraph.read", 3600);
    let addr: SocketAddr = format!("127.0.0.1:{}", free_port()).parse().unwrap();

    let mut child = Command::new(bin)
        .args([
            "serve",
            "--http",
            &addr.to_string(),
            "--tls-cert",
            fx.cert_path.to_str().unwrap(),
            "--tls-key",
            fx.key_path.to_str().unwrap(),
            "--http-auth-jwks",
            jwks_path.to_str().unwrap(),
            "--http-auth-issuer",
            OAUTH_ISSUER,
            "--http-auth-audience",
            OAUTH_AUDIENCE,
            "--in-memory",
            "--admin-http",
            "",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .expect("spawn arcgraph serve --http with OAuth");

    await_listener(&mut child, addr).await;

    let body = serde_json::to_vec(&json!({
        "jsonrpc": "2.0",
        "id": 761,
        "method": "graph.schema",
        "params": { "tenant_id": 1 }
    }))
    .unwrap();

    let client = build_client_config(&fx.cert_der);
    let auth_header = format!("Bearer {valid_token}");
    let ok_resp = timeout(
        Duration::from_secs(10),
        https_post_mcp(
            addr,
            Arc::clone(&client),
            &[
                ("content-type", "application/json"),
                (HEADER_TENANT, "1"),
                ("authorization", &auth_header),
            ],
            &body,
        ),
    )
    .await
    .expect("valid bearer TLS roundtrip completed within budget");
    assert_eq!(
        ok_resp.status, 200,
        "valid bearer token must pass CLI flag -> OAuthConfig -> HttpServerConfig::with_oauth -> serve_http",
    );
    let ok_json: Value = serde_json::from_slice(&ok_resp.body).expect("response body is JSON");
    assert!(
        ok_json.get("result").is_some() && ok_json.get("error").is_none(),
        "valid bearer response must be JSON-RPC success: {ok_json}",
    );

    let missing_resp = timeout(
        Duration::from_secs(10),
        https_post_mcp(
            addr,
            Arc::clone(&client),
            &[("content-type", "application/json"), (HEADER_TENANT, "1")],
            &body,
        ),
    )
    .await
    .expect("missing bearer TLS roundtrip completed within budget");
    assert_eq!(
        missing_resp.status, 401,
        "RED-on-revert: without HttpServerConfig::with_oauth, this unauthenticated request would dispatch and return 200",
    );

    let invalid_resp = timeout(
        Duration::from_secs(10),
        https_post_mcp(
            addr,
            client,
            &[
                ("content-type", "application/json"),
                (HEADER_TENANT, "1"),
                ("authorization", "Bearer not.a.valid.jwt"),
            ],
            &body,
        ),
    )
    .await
    .expect("invalid bearer TLS roundtrip completed within budget");
    assert_eq!(
        invalid_resp.status, 401,
        "invalid bearer token must be rejected by existing serve_http OAuth enforcement",
    );

    child.kill().await.expect("kill server");
}

// ─────────────────────────────────────────────────────────────────────
// Fault injection 1 — ADR-183 durable-by-default refuse-to-start.
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn serve_http_without_storage_mode_refuses_to_start() {
    let bin = env!("CARGO_BIN_EXE_arcgraph");
    let fx = stage_cert_fixture(); // valid cert: the SOLE failure is the missing storage mode.
    let addr = format!("127.0.0.1:{}", free_port());

    let mut cmd = Command::new(bin);
    cmd.args([
        "serve",
        "--http",
        &addr,
        "--tls-cert",
        fx.cert_path.to_str().unwrap(),
        "--tls-key",
        fx.key_path.to_str().unwrap(),
        // NO --data, NO --in-memory.
        "--admin-http",
        "",
    ])
    .stdout(Stdio::piped())
    .stderr(Stdio::piped())
    .kill_on_drop(true);

    let out = timeout(STARTUP_BUDGET, cmd.output())
        .await
        .expect("process exited within budget (must refuse, not serve)")
        .expect("collect output");

    assert!(
        !out.status.success(),
        "ADR-183: serve --http must exit non-zero without --data XOR --in-memory"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--in-memory") || stderr.contains("--data"),
        "ADR-183 refusal must name the storage flags: {stderr}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// Fault injection 2 — loopback-default bind gate (RED-on-revert e2e).
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn serve_http_non_loopback_without_optin_refuses_to_start() {
    let bin = env!("CARGO_BIN_EXE_arcgraph");
    let fx = stage_cert_fixture(); // valid cert + --in-memory: the SOLE failure is the bind gate.
    let addr = format!("0.0.0.0:{}", free_port());

    let mut cmd = Command::new(bin);
    cmd.args([
        "serve",
        "--http",
        &addr, // non-loopback…
        "--tls-cert",
        fx.cert_path.to_str().unwrap(),
        "--tls-key",
        fx.key_path.to_str().unwrap(),
        "--in-memory",
        // …WITHOUT --allow-remote-http-bind.
        "--admin-http",
        "",
    ])
    .stdout(Stdio::piped())
    .stderr(Stdio::piped())
    .kill_on_drop(true);

    // If the gate is reverted, the binary binds 0.0.0.0 and serves
    // forever → cmd.output() never resolves → timeout fires → RED.
    let out = timeout(STARTUP_BUDGET, cmd.output())
        .await
        .expect("process exited within budget (must refuse, not bind 0.0.0.0)")
        .expect("collect output");

    assert!(
        !out.status.success(),
        "non-loopback --http bind without --allow-remote-http-bind must exit non-zero"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--allow-remote-http-bind"),
        "loopback-default refusal must cite the opt-in flag: {stderr}"
    );
}
