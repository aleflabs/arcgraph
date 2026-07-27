//! #761 slice 2 integration tests — Bolt HELLO OAuth bearer-token auth wired
//! into `StorageBoltHandler` via the `with_oauth` builder (ADR-049).
//!
//! Coverage:
//!   1. **Success** — valid bearer JWT → HELLO SUCCESS + RUN succeeds.
//!      This is the ADR-133 §D-4 "Driver" e2e row: full
//!      HANDSHAKE → HELLO(bearer) → RUN → PULL session over a real TCP listener
//!      against a real `StorageBoltHandler` backed by an in-memory substrate.
//!   2. **No bearer when OAuth enforced** — HELLO without credentials →
//!      FAILURE (Neo.ClientError.Security.Unauthorized). RED-on-revert: without
//!      `.with_oauth(...)` on the handler, this test fails because the dev-mode
//!      handler accepts any HELLO.
//!   3. **Invalid token** — HELLO with a syntactically-valid-looking but
//!      signature-invalid token → FAILURE.
//!   4. **Dev-mode preserved** — no OAuth config + loopback → basic-auth HELLO +
//!      RUN still works (backwards-compat; the test would fail if `with_oauth` were
//!      unconditionally applied).
//!   5. **Non-loopback gate** — `BoltServerConfig::validate()` rejects a
//!      non-loopback bind when `allow_remote_bind = false` (the default) →
//!      `BoltError::BindAddrForbidden`. This is the "serve --bolt 0.0.0.0:X
//!      without --allow-remote-bolt-bind → refused" gate.
//!
//! Fixture design (mirrors `mcp_http_oauth_integ.rs`):
//!   - ECDSA P-256 (ES256) keypair minted via `rcgen` at test-setup time.
//!   - JWT signed with `jsonwebtoken::encode` (the private PEM from rcgen).
//!   - Public key staged into `OAuthConfig` as a `JsonWebKey` with
//!     `DecodingKey::from_ec_pem`. ArcGraph never reaches the network —
//!     all key material is in-process (per ADR-044 §Decision item 5).

#![allow(clippy::expect_used)]

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use arcgraph_mcp::auth::oauth_pkce::{JsonWebKey, JsonWebKeySet, OAuthConfig, SCOPE_READ};
use arcgraph_mcp::storage::{StorageBackend, StorageBoltHandler};
use arcgraph_mcp::transport::bolt::{
    self, BoltError, BoltServerConfig, ClientMessage, MAGIC_PREAMBLE, PackValue,
    SERVER_ACCEPT_V5_0, decode, encode_client, message::TAG_FAILURE, message::TAG_SUCCESS,
    read_chunked_message, write_chunked_message,
};
use arcgraph_storage::InternTable;
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::catalog::SystemCatalog;
use arcgraph_storage::crud::CrudStore;
use arcgraph_storage::io::InMemoryPageIo;
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::PrimaryIndex;
use arcgraph_storage::router::MultiTenantRouter;
use arcgraph_storage::transaction::TxnManager;
use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
use rcgen::KeyPair;
use serde::Serialize;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

// ─────────────────────────────────────────────────────────────────────
// Storage fixture (mirrors return_alias_columns_wire_e2e.rs)
// ─────────────────────────────────────────────────────────────────────

fn fresh_backend() -> StorageBackend {
    let io = Arc::new(InMemoryPageIo::new());
    let pool = BufferPool::new(64, io);
    let mgr = Arc::new(TxnManager::new());
    let catalog = Arc::new(SystemCatalog::new());
    catalog.bootstrap(&pool, &mgr).expect("catalog bootstrap");
    let allocator = Arc::new(PageAllocator::new());
    let primary = Arc::new(
        PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&allocator), None).expect("PrimaryIndex"),
    );
    let crud = Arc::new(CrudStore::new_with_index(None, primary, allocator));
    let router = Arc::new(MultiTenantRouter::new(catalog, Arc::clone(&crud), None));
    let intern = Arc::new(InternTable::new());
    StorageBackend::new(router, mgr, intern)
}

// ─────────────────────────────────────────────────────────────────────
// JWT fixture — ECDSA P-256 (ES256), per mcp_http_oauth_integ.rs pattern
// ─────────────────────────────────────────────────────────────────────

const ISS: &str = "https://auth.example.test/";
const AUD: &str = "arcgraph-bolt-test";
const KID: &str = "bolt-test-key-1";

struct JwtFixture {
    encoding_key: EncodingKey,
    pub_pem: String,
}

impl JwtFixture {
    fn new() -> Self {
        let kp = KeyPair::generate().expect("rcgen P-256 keypair");
        let priv_pem = kp.serialize_pem();
        let pub_pem = kp.public_key_pem();
        let encoding_key =
            EncodingKey::from_ec_pem(priv_pem.as_bytes()).expect("ES256 encoding key");
        Self {
            encoding_key,
            pub_pem,
        }
    }

    fn as_jwk(&self) -> JsonWebKey {
        let decoding_key = jsonwebtoken::DecodingKey::from_ec_pem(self.pub_pem.as_bytes())
            .expect("ES256 decoding key");
        JsonWebKey {
            kid: KID.to_string(),
            algorithm: Algorithm::ES256,
            decoding_key,
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
            iss: ISS,
            aud: AUD,
            exp,
            scope,
        };
        let mut header = Header::new(Algorithm::ES256);
        header.kid = Some(KID.to_string());
        encode(&header, &claims, &self.encoding_key).expect("mint JWT")
    }
}

fn make_oauth_config(jwk: JsonWebKey) -> Arc<OAuthConfig> {
    let jwks = JsonWebKeySet::new(vec![jwk]).expect("non-empty jwks");
    Arc::new(OAuthConfig::new(
        ISS.to_string(),
        vec![AUD.to_string()],
        jwks,
    ))
}

// ─────────────────────────────────────────────────────────────────────
// Bolt client helpers (mirrors bolt_e2e_full_session.rs)
// ─────────────────────────────────────────────────────────────────────

async fn bolt_handshake(stream: &mut TcpStream) {
    let mut req = Vec::new();
    req.extend_from_slice(&MAGIC_PREAMBLE);
    req.extend_from_slice(&[0x00, 0x00, 0x00, 0x05]); // negotiate Bolt 5.0
    req.extend_from_slice(&[0; 12]);
    stream.write_all(&req).await.unwrap();
    let mut resp = [0u8; 4];
    stream.read_exact(&mut resp).await.unwrap();
    assert_eq!(resp, SERVER_ACCEPT_V5_0, "server must accept Bolt 5.0");
}

async fn bolt_send(stream: &mut TcpStream, msg: &ClientMessage) {
    let mut buf = Vec::new();
    encode_client(&mut buf, msg).unwrap();
    write_chunked_message(stream, &buf).await.unwrap();
}

async fn bolt_recv(stream: &mut TcpStream) -> PackValue {
    let payload = read_chunked_message(stream).await.unwrap().unwrap();
    let (val, n) = decode(&payload, 0).unwrap();
    assert_eq!(n, payload.len());
    val
}

/// Read the reply tag (the first byte of the Struct header).
async fn bolt_recv_tag(stream: &mut TcpStream) -> u8 {
    match bolt_recv(stream).await {
        PackValue::Struct { tag, .. } => tag,
        other => panic!("expected PackValue::Struct, got {other:?}"),
    }
}

async fn spawn_handler_listener<H>(
    handler: Arc<H>,
) -> (SocketAddr, tokio::sync::oneshot::Sender<()>)
where
    H: arcgraph_mcp::transport::bolt::BoltQueryHandler,
{
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

// ─────────────────────────────────────────────────────────────────────
// Test 1: valid bearer JWT → HELLO SUCCESS + RUN succeeds (ADR-133 §D-4 Driver)
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn oauth_valid_bearer_hello_and_run_succeeds() {
    let fixture = JwtFixture::new();
    let config = make_oauth_config(fixture.as_jwk());
    let token = fixture.mint(SCOPE_READ, 3600); // valid for 1h

    let handler =
        Arc::new(StorageBoltHandler::new(fresh_backend()).with_oauth(Arc::clone(&config)));
    // Sanity: oauth_enforced() reflects the wired config.
    assert!(
        handler.oauth_enforced(),
        "handler must report oauth_enforced after with_oauth"
    );

    let (addr, shutdown) = spawn_handler_listener(handler).await;
    let mut stream = tokio::time::timeout(Duration::from_secs(2), TcpStream::connect(addr))
        .await
        .expect("connect timeout")
        .expect("connect");

    // HANDSHAKE.
    bolt_handshake(&mut stream).await;

    // HELLO with scheme="bearer" + credentials=<JWT>.
    bolt_send(
        &mut stream,
        &ClientMessage::Hello {
            user_agent: Some("arcgraph-test/1.0".into()),
            scheme: Some("bearer".into()),
            // The verified bearer authenticates the session while the HELLO
            // principal binds ADR-212 row visibility for subsequent RUNs.
            principal: Some("oauth-test-principal".into()),
            credentials: Some(token),
            routing: None,
            extras: BTreeMap::new(),
        },
    )
    .await;
    let hello_reply = bolt_recv(&mut stream).await;
    let (tag, meta) = match hello_reply {
        PackValue::Struct { tag, fields } => {
            let m = match fields.into_iter().next() {
                Some(PackValue::Map(m)) => m,
                _ => panic!("HELLO reply body is not a map"),
            };
            (tag, m)
        }
        other => panic!("expected Struct, got {other:?}"),
    };
    assert_eq!(
        tag, TAG_SUCCESS,
        "HELLO with valid bearer must succeed; meta={meta:?}"
    );
    assert!(
        meta.contains_key("connection_id"),
        "HELLO SUCCESS must carry connection_id"
    );

    // RUN "RETURN 1" — proves the authenticated session can issue queries.
    bolt_send(
        &mut stream,
        &ClientMessage::Run {
            query: "RETURN 1".into(),
            parameters: BTreeMap::new(),
            extra: BTreeMap::new(),
        },
    )
    .await;
    let run_tag = bolt_recv_tag(&mut stream).await;
    assert_eq!(
        run_tag, TAG_SUCCESS,
        "RUN must succeed after successful OAuth HELLO"
    );

    let _ = shutdown.send(());
}

// ─────────────────────────────────────────────────────────────────────
// Test 2: no bearer when OAuth enforced → FAILURE (RED-on-revert)
//
// Without `.with_oauth(...)` the dev-mode handler accepts any HELLO,
// so this test proves the OAuth gate is ACTUALLY wired (not just compiled).
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn oauth_no_bearer_hello_rejected_when_enforced() {
    let fixture = JwtFixture::new();
    let config = make_oauth_config(fixture.as_jwk());

    let handler =
        Arc::new(StorageBoltHandler::new(fresh_backend()).with_oauth(Arc::clone(&config)));
    let (addr, shutdown) = spawn_handler_listener(handler).await;
    let mut stream = tokio::time::timeout(Duration::from_secs(2), TcpStream::connect(addr))
        .await
        .expect("connect timeout")
        .expect("connect");
    bolt_handshake(&mut stream).await;

    // HELLO with scheme="none" — no credentials at all.
    bolt_send(
        &mut stream,
        &ClientMessage::Hello {
            user_agent: Some("arcgraph-test/1.0".into()),
            scheme: Some("none".into()),
            principal: None,
            credentials: None,
            routing: None,
            extras: BTreeMap::new(),
        },
    )
    .await;
    let tag = bolt_recv_tag(&mut stream).await;
    assert_eq!(
        tag, TAG_FAILURE,
        "unauthenticated HELLO must be REJECTED when OAuth is enforced"
    );

    // Connection is now in Failed state; a subsequent RUN must be IGNORED
    // (per Bolt §"Session lifecycle" — Failed state ignores RUN until RESET).
    bolt_send(
        &mut stream,
        &ClientMessage::Run {
            query: "RETURN 1".into(),
            parameters: BTreeMap::new(),
            extra: BTreeMap::new(),
        },
    )
    .await;
    let run_tag = bolt_recv_tag(&mut stream).await;
    assert_eq!(
        run_tag,
        arcgraph_mcp::transport::bolt::message::TAG_IGNORED,
        "RUN after HELLO FAILURE must be IGNORED (Bolt Failed state)"
    );

    let _ = shutdown.send(());
}

// ─────────────────────────────────────────────────────────────────────
// Test 3: invalid token (wrong-key signature) → FAILURE
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn oauth_invalid_token_hello_rejected() {
    let config_fixture = JwtFixture::new();
    let config = make_oauth_config(config_fixture.as_jwk());

    // A DIFFERENT keypair mints the token — signature won't verify.
    let attacker_fixture = JwtFixture::new();
    let bad_token = attacker_fixture.mint(SCOPE_READ, 3600);

    let handler =
        Arc::new(StorageBoltHandler::new(fresh_backend()).with_oauth(Arc::clone(&config)));
    let (addr, shutdown) = spawn_handler_listener(handler).await;
    let mut stream = tokio::time::timeout(Duration::from_secs(2), TcpStream::connect(addr))
        .await
        .expect("connect timeout")
        .expect("connect");
    bolt_handshake(&mut stream).await;

    bolt_send(
        &mut stream,
        &ClientMessage::Hello {
            user_agent: Some("arcgraph-test/1.0".into()),
            scheme: Some("bearer".into()),
            principal: None,
            credentials: Some(bad_token),
            routing: None,
            extras: BTreeMap::new(),
        },
    )
    .await;
    let tag = bolt_recv_tag(&mut stream).await;
    assert_eq!(
        tag, TAG_FAILURE,
        "HELLO with wrong-key bearer must be REJECTED"
    );

    let _ = shutdown.send(());
}

// ─────────────────────────────────────────────────────────────────────
// Test 4: dev-mode preserved — no OAuth + loopback → basic-auth HELLO works
//
// Without `.with_oauth(...)` the handler accepts any HELLO. This test guards
// against accidentally making OAuth mandatory unconditionally — backwards-compat
// with the dev-mode posture documented in ADR-049.
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn dev_mode_noauth_loopback_hello_and_run_work() {
    // No .with_oauth() — dev mode.
    let handler = Arc::new(StorageBoltHandler::new(fresh_backend()));
    assert!(
        !handler.oauth_enforced(),
        "dev-mode handler must NOT enforce oauth"
    );

    let (addr, shutdown) = spawn_handler_listener(handler).await;
    let mut stream = tokio::time::timeout(Duration::from_secs(2), TcpStream::connect(addr))
        .await
        .expect("connect timeout")
        .expect("connect");
    bolt_handshake(&mut stream).await;

    // basic-auth HELLO — accepted in dev mode.
    bolt_send(
        &mut stream,
        &ClientMessage::Hello {
            user_agent: Some("arcgraph-test/1.0".into()),
            scheme: Some("basic".into()),
            principal: Some("devuser".into()),
            credentials: Some("devpass".into()),
            routing: None,
            extras: BTreeMap::new(),
        },
    )
    .await;
    let tag = bolt_recv_tag(&mut stream).await;
    assert_eq!(tag, TAG_SUCCESS, "dev-mode basic HELLO must succeed");

    bolt_send(
        &mut stream,
        &ClientMessage::Run {
            query: "RETURN 1".into(),
            parameters: BTreeMap::new(),
            extra: BTreeMap::new(),
        },
    )
    .await;
    let run_tag = bolt_recv_tag(&mut stream).await;
    assert_eq!(run_tag, TAG_SUCCESS, "dev-mode RUN must succeed");

    let _ = shutdown.send(());
}

// ─────────────────────────────────────────────────────────────────────
// Test 5: non-loopback bind without --allow-remote-bolt-bind → refused
//
// `BoltServerConfig::validate()` with `allow_remote_bind = false` and a
// non-loopback bind must return `BoltError::BindAddrForbidden`. This is the
// CLI's "serve --bolt 0.0.0.0:X without --allow-remote-bolt-bind → refused"
// gate (design-v2 §9.4, W14 retro IR L1-HIGH-4).
//
// We test `validate()` directly (no running server needed) since the CLI
// wires `allow_remote_bind = args.allow_remote_bolt_bind` into `BoltServerConfig`
// before calling `serve_bolt_listener` (which calls validate internally).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn noauth_nonloopback_bind_config_rejected_without_remote_flag() {
    let config = BoltServerConfig {
        bind: "0.0.0.0:7687".parse::<SocketAddr>().unwrap(),
        max_connections: 256,
        allow_remote_bind: false, // the CLI default; set by --allow-remote-bolt-bind
        tls: None,
        client_verifier: None,
        dispatch_bulkhead: None,
    };
    let err = config
        .validate()
        .expect_err("non-loopback bind with allow_remote_bind=false must be rejected");
    assert!(
        matches!(err, BoltError::BindAddrForbidden { .. }),
        "expected BindAddrForbidden, got {err:?}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// Test 6: loopback bind without OAuth → validate() succeeds (dev mode OK)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn loopback_noauth_config_validates_as_dev_mode() {
    let config = BoltServerConfig {
        bind: "127.0.0.1:7687".parse::<SocketAddr>().unwrap(),
        max_connections: 256,
        allow_remote_bind: false,
        tls: None,
        client_verifier: None,
        dispatch_bulkhead: None,
    };
    config
        .validate()
        .expect("loopback + no auth is valid dev mode; validate() must succeed");
}
