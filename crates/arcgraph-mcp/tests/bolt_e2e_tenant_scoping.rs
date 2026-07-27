//! W14δ M5-13 integration test: tenant scoping on the Bolt server.
//!
//! Per the spawn prompt's Hard boundaries: every HELLO authenticates
//! to a tenant. Subsequent RUNs MUST see that tenant in the
//! `BoltQueryHandler::run` call, NOT a different one.
//!
//! v1.0-α `StubBoltHandler` always scopes to `TenantId::DEFAULT`
//! (per the §"tenant derivation at v1.0-α is fixed" doc comment),
//! so we author a custom handler here that records the tenant
//! observed on each RUN and verifies it matches the HELLO-derived
//! tenant. The test is a pin against future regressions where the
//! tenant might be lost / overwritten in the per-connection state.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use arcgraph_core::TenantId;
use arcgraph_mcp::SessionScope;
use arcgraph_mcp::transport::bolt::{
    self, BoltError, BoltQueryHandler, BoltSessionAuth, ClientMessage, MAGIC_PREAMBLE, PackValue,
    RunOutcome, SERVER_ACCEPT_V5_0, decode, encode_client, message::TAG_SUCCESS,
    read_chunked_message, write_chunked_message,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Handler that maps `principal` to a tenant id and records the
/// tenant on every RUN call. Used to pin the tenant flow.
struct TenantScopingHandler {
    /// (principal_lookup, run_tenant_observed) — last RUN's tenant.
    last_seen: Arc<Mutex<Option<TenantId>>>,
}

impl TenantScopingHandler {
    fn new() -> (Self, Arc<Mutex<Option<TenantId>>>) {
        let last = Arc::new(Mutex::new(None));
        (
            Self {
                last_seen: last.clone(),
            },
            last,
        )
    }
}

impl BoltQueryHandler for TenantScopingHandler {
    fn authenticate(
        &self,
        _scheme: Option<&str>,
        principal: Option<&str>,
        _credentials: Option<&str>,
    ) -> Result<BoltSessionAuth, BoltError> {
        match principal {
            Some("alice") => Ok(BoltSessionAuth::new(
                TenantId::new(101),
                Some("alice".into()),
                SessionScope::Read,
            )),
            Some("bob") => Ok(BoltSessionAuth::new(
                TenantId::new(202),
                Some("bob".into()),
                SessionScope::Read,
            )),
            _ => Err(BoltError::Unauthorized("unknown principal".into())),
        }
    }

    fn run(
        &self,
        session: &BoltSessionAuth,
        _cypher: &str,
        _parameters: &BTreeMap<String, PackValue>,
    ) -> Result<RunOutcome, BoltError> {
        // Record the observed tenant so the test can assert.
        let tenant = session.tenant();
        *self.last_seen.lock().unwrap() = Some(tenant);
        // Echo the tenant raw id as a single-column row.
        Ok(RunOutcome {
            fields: vec!["t".into()],
            records: vec![vec![PackValue::Integer(tenant.raw() as i64)]],
            qid: None,
        })
    }
}

#[tokio::test]
async fn run_sees_tenant_derived_from_hello_principal() {
    let (handler, observed_tenant) = TenantScopingHandler::new();
    let handler = Arc::new(handler);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(async move {
        let _ = bolt::serve_bolt_inner(
            handler,
            listener,
            async move {
                let _ = rx.await;
            },
            None,
        )
        .await;
    });
    let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    // Handshake.
    let mut req = Vec::new();
    req.extend_from_slice(&MAGIC_PREAMBLE);
    req.extend_from_slice(&[0x00, 0x00, 0x00, 0x05]);
    req.extend_from_slice(&[0; 12]);
    stream.write_all(&req).await.unwrap();
    let mut resp = [0u8; 4];
    stream.read_exact(&mut resp).await.unwrap();
    assert_eq!(resp, SERVER_ACCEPT_V5_0);
    // HELLO as "alice" → tenant 101.
    let mut buf = Vec::new();
    encode_client(
        &mut buf,
        &ClientMessage::Hello {
            user_agent: Some("tenant-scoping-test/1".into()),
            scheme: Some("basic".into()),
            principal: Some("alice".into()),
            credentials: Some("pw".into()),
            routing: None,
            extras: BTreeMap::new(),
        },
    )
    .unwrap();
    write_chunked_message(&mut stream, &buf).await.unwrap();
    let _ = read_chunked_message(&mut stream).await.unwrap().unwrap();
    // RUN — handler should see tenant 101.
    buf.clear();
    encode_client(
        &mut buf,
        &ClientMessage::Run {
            query: "RETURN tenant".into(),
            parameters: BTreeMap::new(),
            extra: BTreeMap::new(),
        },
    )
    .unwrap();
    write_chunked_message(&mut stream, &buf).await.unwrap();
    let payload = read_chunked_message(&mut stream).await.unwrap().unwrap();
    let (val, _) = decode(&payload, 0).unwrap();
    assert!(matches!(val, PackValue::Struct { tag, .. } if tag == TAG_SUCCESS));
    assert_eq!(
        *observed_tenant.lock().unwrap(),
        Some(TenantId::new(101)),
        "RUN must see the tenant derived from HELLO principal"
    );
    let _ = tx.send(());
}

#[tokio::test]
async fn unknown_principal_rejects_hello_with_unauthorized_failure() {
    let (handler, _observed_tenant) = TenantScopingHandler::new();
    let handler = Arc::new(handler);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(async move {
        let _ = bolt::serve_bolt_inner(
            handler,
            listener,
            async move {
                let _ = rx.await;
            },
            None,
        )
        .await;
    });
    let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    let mut req = Vec::new();
    req.extend_from_slice(&MAGIC_PREAMBLE);
    req.extend_from_slice(&[0x00, 0x00, 0x00, 0x05]);
    req.extend_from_slice(&[0; 12]);
    stream.write_all(&req).await.unwrap();
    let mut resp = [0u8; 4];
    stream.read_exact(&mut resp).await.unwrap();
    let mut buf = Vec::new();
    encode_client(
        &mut buf,
        &ClientMessage::Hello {
            user_agent: Some("test/1".into()),
            scheme: Some("basic".into()),
            principal: Some("eve_attacker".into()),
            credentials: Some("evil".into()),
            routing: None,
            extras: BTreeMap::new(),
        },
    )
    .unwrap();
    write_chunked_message(&mut stream, &buf).await.unwrap();
    let payload = read_chunked_message(&mut stream).await.unwrap().unwrap();
    let (val, _) = decode(&payload, 0).unwrap();
    match val {
        PackValue::Struct { tag, fields } => {
            assert_eq!(
                tag,
                arcgraph_mcp::transport::bolt::message::TAG_FAILURE,
                "unknown principal must FAIL"
            );
            let meta = match fields.into_iter().next() {
                Some(PackValue::Map(m)) => m,
                _ => panic!(),
            };
            match meta.get("code") {
                Some(PackValue::String(s)) => {
                    assert_eq!(s, "Neo.ClientError.Security.Unauthorized");
                }
                _ => panic!("missing code"),
            }
        }
        _ => panic!(),
    }
    let _ = tx.send(());
}
