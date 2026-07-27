//! W14δ M5-13 integration test: Neo4j Python driver wire-shape pin.
//!
//! The spawn prompt's "Neo4j-Python-driver-equivalent E2E" pin is
//! the conformance gate for "from neo4j import GraphDatabase" usage
//! against ArcGraph. The official Python driver (and every other
//! Neo4j-shipped Bolt 5.0 driver) emits a deterministic frame
//! sequence on session establishment + every query; this test
//! replays that exact sequence and asserts the server replies
//! conform to the spec.
//!
//! # Why not depend on `neo4rs`?
//!
//! `neo4rs` is a Rust driver with the SAME wire semantics as the
//! Python driver. Adding it as a dev-dep would:
//! 1. pull in a substantial transitive surface (the driver targets
//!    full Neo4j 5.x — connection pooling, routing, prepared
//!    statements);
//! 2. risk pulling a transitive crate whose license is not on the
//!    `deny.toml` allowlist (e.g., a GPL utility) — which would
//!    block the cargo deny FULL gate the spawn prompt mandates;
//! 3. not increase coverage beyond what this hand-rolled wire pin
//!    already provides — both pins exercise the same protocol path.
//!
//! Forward-pin: M5-12 will add a sibling
//! `bolt_e2e_neo4rs.rs` test with `neo4rs` once the v1.1 surface
//! lights, because at that point we'll want a SECOND driver crate
//! exercising the protocol against real `QueryEngine` wiring.

use std::collections::BTreeMap;
use std::sync::Arc;

use arcgraph_core::TenantId;
use arcgraph_mcp::SessionScope;
use arcgraph_mcp::transport::bolt::{
    self, BoltError, BoltQueryHandler, BoltSessionAuth, ClientMessage, PackValue, RunOutcome,
    SERVER_ACCEPT_V5_0, StubBoltHandler, decode, encode_client, message::TAG_FAILURE,
    message::TAG_RECORD, message::TAG_SUCCESS, read_chunked_message, write_chunked_message,
};
use arcgraph_query::parse_multi;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// The exact magic-preamble + version-offer bytes the Neo4j Python
/// driver 5.x sends on connection open. Per the driver's
/// `bolt_protocol_version` constants the negotiated set is
/// 5.4 / 5.3 / 5.2 / 5.0 (offered as one range + the legacy 4.x
/// fallbacks; we only care about the first 5.0-capable offer).
const PYDRIVER_HANDSHAKE: [u8; 20] = [
    // magic
    0x60, 0x60, 0xB0, 0x17, //
    // offer 1: [00, 03, 04, 05] — Bolt 5.1..=5.4 (omits 5.0).
    0x00, 0x03, 0x04, 0x05, //
    // offer 2: [00, 00, 00, 05] — Bolt 5.0 exactly. ← server must
    // pick this one since offer 1 excludes 5.0.
    0x00, 0x00, 0x00, 0x05, //
    // offer 3: [00, 00, 04, 04] — Bolt 4.4 fallback.
    0x00, 0x00, 0x04, 0x04, //
    // offer 4: padding.
    0x00, 0x00, 0x00, 0x00, //
];

#[tokio::test]
async fn server_negotiates_v5_0_from_python_driver_offer_set() {
    let handler = Arc::new(StubBoltHandler::accepting());
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
    stream.write_all(&PYDRIVER_HANDSHAKE).await.unwrap();
    let mut resp = [0u8; 4];
    stream.read_exact(&mut resp).await.unwrap();
    assert_eq!(
        resp, SERVER_ACCEPT_V5_0,
        "server must pick Bolt 5.0 from the python-driver offer set"
    );
    let _ = tx.send(());
}

#[tokio::test]
async fn server_replies_to_python_driver_hello_shape() {
    // Real Neo4j 5.x Python driver HELLO includes:
    //   {
    //     "user_agent": "neo4j-python/5.x",
    //     "bolt_agent": { ... },
    //     "patch_bolt": [...],  (Bolt 5.3+)
    //     "notifications_minimum_severity": "WARNING",
    //     "notifications_disabled_categories": []
    //   }
    // followed by LOGON in 5.1+. At 5.0 the auth fields are in
    // HELLO. We exercise the 5.0 shape directly.
    let handler = Arc::new(StubBoltHandler::accepting());
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
    stream.write_all(&PYDRIVER_HANDSHAKE).await.unwrap();
    let mut resp = [0u8; 4];
    stream.read_exact(&mut resp).await.unwrap();
    assert_eq!(resp, SERVER_ACCEPT_V5_0);
    // HELLO with Python-driver-shape extras map.
    let mut extras: BTreeMap<String, PackValue> = BTreeMap::new();
    let mut bolt_agent = BTreeMap::new();
    bolt_agent.insert(
        "product".into(),
        PackValue::String("neo4j-python/5.7".into()),
    );
    bolt_agent.insert(
        "platform".into(),
        PackValue::String("Linux 5.15.0-78-generic".into()),
    );
    extras.insert("bolt_agent".into(), PackValue::Map(bolt_agent));
    extras.insert(
        "notifications_minimum_severity".into(),
        PackValue::String("WARNING".into()),
    );
    extras.insert(
        "notifications_disabled_categories".into(),
        PackValue::List(vec![]),
    );
    let hello = ClientMessage::Hello {
        user_agent: Some("neo4j-python/5.7.0 Python/3.11.5 (linux)".into()),
        scheme: Some("basic".into()),
        principal: Some("neo4j".into()),
        credentials: Some("password".into()),
        routing: None,
        extras,
    };
    let mut buf = Vec::new();
    encode_client(&mut buf, &hello).unwrap();
    write_chunked_message(&mut stream, &buf).await.unwrap();
    // Read HELLO reply.
    let payload = read_chunked_message(&mut stream).await.unwrap().unwrap();
    let (val, _) = decode(&payload, 0).unwrap();
    match val {
        PackValue::Struct { tag, fields } => {
            assert_eq!(tag, TAG_SUCCESS, "python driver expects SUCCESS on HELLO");
            let meta = match fields.into_iter().next() {
                Some(PackValue::Map(m)) => m,
                _ => panic!("SUCCESS body not a map"),
            };
            // The Python driver reads `server` to determine which
            // dialect rules to apply. Pin its presence + shape.
            match meta.get("server") {
                Some(PackValue::String(s)) => {
                    assert!(s.starts_with("ArcGraph/"), "server slug = {s}");
                }
                other => panic!("server field missing/invalid: {other:?}"),
            }
            // Drivers also read `connection_id` for diagnostics.
            assert!(matches!(
                meta.get("connection_id"),
                Some(PackValue::String(_))
            ));
        }
        other => panic!("expected struct, got {other:?}"),
    }
    let _ = tx.send(());
}

#[tokio::test]
async fn run_outside_supported_subset_emits_failure_with_python_driver_code() {
    // Python driver reads the FAILURE `code` field to decide
    // whether the error is retryable. Pin that ours uses the
    // canonical `Neo.ClientError.Statement.SyntaxError` slug for
    // queries the ArcQL parser would reject.
    let handler = Arc::new(StubBoltHandler {
        forced_error: Some(bolt::StubFault::Syntax(
            "Cypher construct outside supported subset".into(),
        )),
        require_principal: false,
    });
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
    stream.write_all(&PYDRIVER_HANDSHAKE).await.unwrap();
    let mut resp = [0u8; 4];
    stream.read_exact(&mut resp).await.unwrap();
    // HELLO.
    let mut buf = Vec::new();
    encode_client(
        &mut buf,
        &ClientMessage::Hello {
            user_agent: Some("neo4j-python/5.7".into()),
            scheme: Some("none".into()),
            principal: None,
            credentials: None,
            routing: None,
            extras: BTreeMap::new(),
        },
    )
    .unwrap();
    write_chunked_message(&mut stream, &buf).await.unwrap();
    let _ = read_chunked_message(&mut stream).await.unwrap().unwrap();
    // RUN (will fault).
    buf.clear();
    encode_client(
        &mut buf,
        &ClientMessage::Run {
            query: "MATCH (n) BOGUS RETURN n".into(),
            parameters: BTreeMap::new(),
            extra: BTreeMap::new(),
        },
    )
    .unwrap();
    write_chunked_message(&mut stream, &buf).await.unwrap();
    let payload = read_chunked_message(&mut stream).await.unwrap().unwrap();
    let (val, _) = decode(&payload, 0).unwrap();
    match val {
        PackValue::Struct { tag, fields } => {
            assert_eq!(tag, TAG_FAILURE);
            let meta = match fields.into_iter().next() {
                Some(PackValue::Map(m)) => m,
                _ => panic!(),
            };
            // Python driver code: "Neo.<Class>.<Category>.<Title>"
            let code = match meta.get("code") {
                Some(PackValue::String(s)) => s.clone(),
                _ => panic!("FAILURE missing code"),
            };
            assert_eq!(code, "Neo.ClientError.Statement.SyntaxError");
            // Driver surfaces `message` to caller code; pin it
            // contains the human-readable detail.
            match meta.get("message") {
                Some(PackValue::String(m)) => {
                    assert!(m.contains("outside supported subset"), "msg = {m}");
                }
                _ => panic!("FAILURE missing message"),
            }
        }
        _ => panic!("expected FAILURE"),
    }
    let _ = tx.send(());
}

#[tokio::test]
async fn cancellation_during_run_via_reset_succeeds() {
    // The Python driver fires RESET from a sibling thread to cancel
    // a long-running RUN. We exercise the protocol-level RESET
    // semantic: while in `Streaming` (post-RUN, pre-PULL drain),
    // a RESET clears the active stream and returns to `Ready`.
    let handler = Arc::new(StubBoltHandler::accepting());
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
    stream.write_all(&PYDRIVER_HANDSHAKE).await.unwrap();
    let mut resp = [0u8; 4];
    stream.read_exact(&mut resp).await.unwrap();
    let mut buf = Vec::new();
    // HELLO.
    encode_client(
        &mut buf,
        &ClientMessage::Hello {
            user_agent: Some("test/1".into()),
            scheme: Some("none".into()),
            principal: None,
            credentials: None,
            routing: None,
            extras: BTreeMap::new(),
        },
    )
    .unwrap();
    write_chunked_message(&mut stream, &buf).await.unwrap();
    let _ = read_chunked_message(&mut stream).await.unwrap().unwrap();
    // RUN.
    buf.clear();
    encode_client(
        &mut buf,
        &ClientMessage::Run {
            query: "RETURN 1".into(),
            parameters: BTreeMap::new(),
            extra: BTreeMap::new(),
        },
    )
    .unwrap();
    write_chunked_message(&mut stream, &buf).await.unwrap();
    let _ = read_chunked_message(&mut stream).await.unwrap().unwrap();
    // RESET mid-stream (before PULL).
    buf.clear();
    encode_client(&mut buf, &ClientMessage::Reset).unwrap();
    write_chunked_message(&mut stream, &buf).await.unwrap();
    let payload = read_chunked_message(&mut stream).await.unwrap().unwrap();
    let (val, _) = decode(&payload, 0).unwrap();
    match val {
        PackValue::Struct { tag, .. } => {
            assert_eq!(tag, TAG_SUCCESS, "RESET reply must be SUCCESS");
        }
        _ => panic!(),
    }
    // After RESET we should be back in Ready: a new RUN succeeds.
    buf.clear();
    encode_client(
        &mut buf,
        &ClientMessage::Run {
            query: "RETURN 2".into(),
            parameters: BTreeMap::new(),
            extra: BTreeMap::new(),
        },
    )
    .unwrap();
    write_chunked_message(&mut stream, &buf).await.unwrap();
    let payload = read_chunked_message(&mut stream).await.unwrap().unwrap();
    let (val, _) = decode(&payload, 0).unwrap();
    match val {
        PackValue::Struct { tag, .. } => {
            assert_eq!(tag, TAG_SUCCESS, "post-RESET RUN must succeed");
        }
        _ => panic!(),
    }
    // PULL drains the new stream.
    buf.clear();
    encode_client(&mut buf, &ClientMessage::Pull { n: -1, qid: None }).unwrap();
    write_chunked_message(&mut stream, &buf).await.unwrap();
    let mut saw_record = false;
    loop {
        let payload = read_chunked_message(&mut stream).await.unwrap().unwrap();
        let (val, _) = decode(&payload, 0).unwrap();
        match val {
            PackValue::Struct { tag, .. } if tag == TAG_RECORD => saw_record = true,
            PackValue::Struct { tag, .. } if tag == TAG_SUCCESS => break,
            other => panic!("unexpected frame: {other:?}"),
        }
    }
    assert!(saw_record, "post-RESET PULL drained at least one row");
    let _ = tx.send(());
}

// ─────────────────────────────────────────────────────────────────────
// Real-parser subset enforcement gate (round-2 H-1)
// ─────────────────────────────────────────────────────────────────────

/// Test fixture handler that wires the W13γ M4-83 `parse_multi`
/// surface into the Bolt RUN path. Used by
/// [`subset_violation_via_real_parser_surfaces_as_failure`] to prove
/// the parser-rejection → Bolt-FAILURE translation actually runs on
/// the production parser (vs the stub's forced-error path which never
/// invokes `parse_multi`).
///
/// This is NOT the M5-12 production handler (no `QueryEngine`,
/// no executor wiring, no plan-cache); it is a thin shim whose only
/// behavior is "parse the Cypher; on success return empty rows; on
/// `ParseError`, translate to `BoltError::Syntax`". Production wiring
/// at M5-12 replaces this shim with `QueryEngine::execute`.
struct RealParserHandler;

impl BoltQueryHandler for RealParserHandler {
    fn authenticate(
        &self,
        _scheme: Option<&str>,
        _principal: Option<&str>,
        _credentials: Option<&str>,
    ) -> Result<BoltSessionAuth, BoltError> {
        Ok(BoltSessionAuth::new(
            TenantId::DEFAULT,
            None,
            SessionScope::Read,
        ))
    }

    fn run(
        &self,
        _session: &BoltSessionAuth,
        cypher: &str,
        _parameters: &BTreeMap<String, PackValue>,
    ) -> Result<RunOutcome, BoltError> {
        // The translation step that the M5-12 production handler
        // will perform: pipe Cypher through `parse_multi`, surface
        // `ParseError` as `BoltError::Syntax`. The empty
        // `RunOutcome` on success keeps the test focused on the
        // FAILURE branch.
        match parse_multi(cypher) {
            Ok(_stmts) => Ok(RunOutcome {
                fields: Vec::new(),
                records: Vec::new(),
                qid: None,
            }),
            Err(e) => Err(BoltError::Syntax(e.to_string())),
        }
    }
}

/// H-1 gate: the spawn prompt's "RUN with Cypher outside W13γ M4-83
/// subset → FAILURE with clear error text" acceptance criterion,
/// exercised through the **actual** `arcgraph_query::parse_multi`
/// path rather than a stub-forced fault.
///
/// Pairs with `run_outside_supported_subset_emits_failure_with_python_driver_code`
/// (above): that test verifies the FAILURE serialization shape on the
/// wire is what the Python driver expects; this test verifies the
/// gate is **actually driven by the parser**, not by a stub fault.
#[tokio::test]
async fn subset_violation_via_real_parser_surfaces_as_failure() {
    // Pre-flight: prove the parser rejects this string in isolation,
    // so a failure of the wire path can't be confused with the
    // parser silently admitting the construct. This is the same
    // invariant H-1 was about: don't let a passing test mask a
    // missing gate.
    let parse_result = parse_multi("MATCH (n) BOGUS RETURN n");
    assert!(
        parse_result.is_err(),
        "real parser must reject unsupported clause syntax; \
         if this passes, the grammar admitted out-of-subset syntax and \
         the wire-path assertion below is no longer load-bearing"
    );

    let handler = Arc::new(RealParserHandler);
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
    stream.write_all(&PYDRIVER_HANDSHAKE).await.unwrap();
    let mut resp = [0u8; 4];
    stream.read_exact(&mut resp).await.unwrap();
    assert_eq!(resp, SERVER_ACCEPT_V5_0);
    // HELLO.
    let mut buf = Vec::new();
    encode_client(
        &mut buf,
        &ClientMessage::Hello {
            user_agent: Some("neo4j-python/5.7".into()),
            scheme: Some("none".into()),
            principal: None,
            credentials: None,
            routing: None,
            extras: BTreeMap::new(),
        },
    )
    .unwrap();
    write_chunked_message(&mut stream, &buf).await.unwrap();
    let _ = read_chunked_message(&mut stream).await.unwrap().unwrap();
    // RUN with the out-of-subset query. The handler pipes this
    // through `parse_multi`; the parser rejects → BoltError::Syntax
    // → wire FAILURE with Neo.ClientError.Statement.SyntaxError.
    buf.clear();
    encode_client(
        &mut buf,
        &ClientMessage::Run {
            query: "MATCH (n) BOGUS RETURN n".into(),
            parameters: BTreeMap::new(),
            extra: BTreeMap::new(),
        },
    )
    .unwrap();
    write_chunked_message(&mut stream, &buf).await.unwrap();
    let payload = read_chunked_message(&mut stream).await.unwrap().unwrap();
    let (val, _) = decode(&payload, 0).unwrap();
    match val {
        PackValue::Struct { tag, fields } => {
            assert_eq!(
                tag, TAG_FAILURE,
                "out-of-subset RUN must serialize as FAILURE"
            );
            let meta = match fields.into_iter().next() {
                Some(PackValue::Map(m)) => m,
                _ => panic!("FAILURE first field not map"),
            };
            let code = match meta.get("code") {
                Some(PackValue::String(s)) => s.clone(),
                _ => panic!("FAILURE missing code"),
            };
            assert_eq!(
                code, "Neo.ClientError.Statement.SyntaxError",
                "code must be Neo4j-canonical syntax-error slug"
            );
            // The message slot carries the parser's humanized
            // rejection. We don't pin the exact text (it's
            // pest's rendering, which may shift across pest
            // releases), but it MUST be non-empty so drivers can
            // surface it to caller code.
            match meta.get("message") {
                Some(PackValue::String(m)) => {
                    assert!(!m.is_empty(), "FAILURE message must not be empty");
                }
                _ => panic!("FAILURE missing message"),
            }
        }
        _ => panic!("expected FAILURE"),
    }
    let _ = tx.send(());
}
