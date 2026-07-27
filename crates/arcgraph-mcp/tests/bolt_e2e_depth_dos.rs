//! **#819 (HIGH, security/DoS)** — server-stays-up proof for the
//! expression-nesting-depth guard, end-to-end over Bolt.
//!
//! # What this proves (the actual vulnerability)
//!
//! Pre-#819, a RUN carrying a deeply-nested expression — bracket-style
//! (`((((1))))`, `[[[[1]]]]`, `CASE WHEN … END`) OR a bracket-less
//! unary-operator chain (`-+-+-+ … 1`, the #819 R1 residual) — drove
//! unbounded native-stack recursion in the parser ON THE SERVER'S
//! WORKER THREAD and aborted the WHOLE process with SIGABRT — an
//! unauthenticated remote DoS that took ArcGraph down for every
//! client/tenant via a ~600-byte–to–~4 KB query.
//!
//! This test drives the REAL `arcgraph_query::parse_multi` path (the
//! same path the M5-12 production handler uses) across a real TCP Bolt
//! 5.0 session and asserts the two halves of the fixed contract:
//!
//!   1. The deep RUN returns a clean Bolt **FAILURE**
//!      (`Neo.ClientError.Statement.SyntaxError`) — NOT a crash.
//!   2. The server is **STILL UP** afterward: a subsequent
//!      `RETURN 1` on the SAME connection succeeds.
//!
//! Because the server runs on this test's own process, the pre-fix
//! SIGABRT would kill the TEST PROCESS itself (signal 6) — so this test
//! failing-by-crashing on `origin/main` IS the discriminating proof
//! that the deep query no longer aborts the server. Post-fix, both
//! halves above hold and the test passes cleanly.
//!
//! The parser-/eval-level exhaustive coverage (all 3 forms, 300 / 1000
//! / 100k depth, just-under-cap, string-literal safety, end-to-end
//! eval) lives in
//! `crates/arcgraph-query/tests/expression_depth_dos.rs`.

#![allow(clippy::expect_used)]

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use arcgraph_core::TenantId;
use arcgraph_mcp::SessionScope;
use arcgraph_mcp::transport::bolt::{
    self, BoltError, BoltQueryHandler, BoltSessionAuth, ClientMessage, PackValue, RunOutcome,
    SERVER_ACCEPT_V5_0, decode, encode_client, message::TAG_FAILURE, message::TAG_SUCCESS,
    read_chunked_message, write_chunked_message,
};
use arcgraph_query::parse_multi;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// Bolt 5.0 handshake: magic preamble + a 5.0-capable version offer.
const HANDSHAKE: [u8; 20] = [
    0x60, 0x60, 0xB0, 0x17, // magic
    0x00, 0x00, 0x00, 0x05, // Bolt 5.0
    0x00, 0x00, 0x00, 0x00, //
    0x00, 0x00, 0x00, 0x00, //
    0x00, 0x00, 0x00, 0x00, //
];

/// Real-parser RUN handler (mirrors `RealParserHandler` in
/// `bolt_e2e_python_driver_shape.rs`): pipe the Cypher through the
/// production `parse_multi`, surface `ParseError` (incl.
/// `ExpressionTooDeep`) as `BoltError::Syntax` → wire FAILURE. On a
/// successful parse, return empty rows (SUCCESS) so the "server still
/// answers" half of the proof has a clean positive signal.
///
/// This is the surface the DoS lives on: the parser running on the
/// server's Tokio worker thread for an attacker-supplied query.
struct DepthDosHandler;

impl BoltQueryHandler for DepthDosHandler {
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

async fn spawn_server() -> (std::net::SocketAddr, tokio::sync::oneshot::Sender<()>) {
    let handler = Arc::new(DepthDosHandler);
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
    (addr, tx)
}

async fn handshake_and_hello(stream: &mut TcpStream) {
    stream.write_all(&HANDSHAKE).await.unwrap();
    let mut resp = [0u8; 4];
    stream.read_exact(&mut resp).await.unwrap();
    assert_eq!(resp, SERVER_ACCEPT_V5_0, "server must accept Bolt 5.0");
    let mut buf = Vec::new();
    encode_client(
        &mut buf,
        &ClientMessage::Hello {
            user_agent: Some("arcgraph-depth-dos-test/1.0".into()),
            scheme: Some("none".into()),
            principal: None,
            credentials: None,
            routing: None,
            extras: BTreeMap::new(),
        },
    )
    .unwrap();
    write_chunked_message(stream, &buf).await.unwrap();
    let _ = read_chunked_message(stream).await.unwrap().unwrap(); // HELLO SUCCESS
}

/// Send a RUN and return the decoded reply struct tag + (if FAILURE)
/// the `code` string.
async fn run_query(stream: &mut TcpStream, query: &str) -> (u8, Option<String>) {
    let mut buf = Vec::new();
    encode_client(
        &mut buf,
        &ClientMessage::Run {
            query: query.into(),
            parameters: BTreeMap::new(),
            extra: BTreeMap::new(),
        },
    )
    .unwrap();
    write_chunked_message(stream, &buf).await.unwrap();
    let payload = read_chunked_message(stream).await.unwrap().unwrap();
    let (val, _) = decode(&payload, 0).unwrap();
    match val {
        PackValue::Struct { tag, fields } if tag == TAG_FAILURE => {
            let code = match fields.into_iter().next() {
                Some(PackValue::Map(m)) => match m.get("code") {
                    Some(PackValue::String(s)) => Some(s.clone()),
                    _ => None,
                },
                _ => None,
            };
            (tag, code)
        }
        PackValue::Struct { tag, .. } => (tag, None),
        other => panic!("expected a Struct reply, got {other:?}"),
    }
}

/// The deep-nesting forms the issue documents (depth 300 — its crash
/// threshold). Each, sent over Bolt, must come back as a clean FAILURE
/// and leave the server alive.
fn deep_paren(d: usize) -> String {
    "RETURN ".to_string() + &"(".repeat(d) + "1" + &")".repeat(d)
}
fn deep_list(d: usize) -> String {
    "RETURN ".to_string() + &"[".repeat(d) + "1" + &"]".repeat(d)
}
fn deep_case(d: usize) -> String {
    "RETURN ".to_string() + &"CASE WHEN true THEN ".repeat(d) + "1" + &" END".repeat(d)
}
/// Alternating unary-prefix chain of `ops` operators (family (B), the
/// #819 R1 residual). NO bracket, so the bracket scan scored it 0 and a
/// ~4 KB chain SIGABRT-ed `parse_multi` on the worker thread pre-fix.
/// Uses 4000 operators — the R1 PROBE3 crash depth (the unary pest
/// cliff is ~3000 on a 2 MiB worker; 300 like the bracket forms would
/// NOT reach it, so this form needs its own deeper repro).
fn deep_unary(ops: usize) -> String {
    let chain: String = (0..ops)
        .map(|i| if i % 2 == 0 { '-' } else { '+' })
        .collect();
    format!("RETURN {chain}1")
}

/// THE server-stays-up proof. For each deep-nesting form: a single
/// ~600-byte deep query → clean FAILURE → the server STILL answers a
/// follow-up `RETURN 1` with SUCCESS, on the SAME connection.
#[tokio::test]
async fn d819_deep_query_fails_cleanly_and_server_stays_up() {
    let (addr, shutdown) = spawn_server().await;

    let mut stream = tokio::time::timeout(Duration::from_secs(5), TcpStream::connect(addr))
        .await
        .expect("connect timed out")
        .expect("connect failed");
    handshake_and_hello(&mut stream).await;

    for (form, deep) in [
        ("parens", deep_paren(300)),
        ("lists", deep_list(300)),
        ("case", deep_case(300)),
        // Family (B) — the R1 residual. 4000 operators (~4 KB): on the
        // pre-fix build this SIGABRT-ed `parse_multi` on the worker (the
        // bracket scan counted it 0), crashing THIS test process.
        ("unary", deep_unary(4000)),
    ] {
        // 1. The deep RUN must come back as a clean FAILURE — pre-fix
        //    this RUN SIGABRT-ed the whole process (which would crash
        //    THIS test). Reaching this assertion at all proves no abort.
        let (tag, code) = run_query(&mut stream, &deep).await;
        assert_eq!(
            tag, TAG_FAILURE,
            "{form}: a 300-deep nested expression must return a Bolt FAILURE (not a crash, not SUCCESS)"
        );
        assert_eq!(
            code.as_deref(),
            Some("Neo.ClientError.Statement.SyntaxError"),
            "{form}: the depth-cap rejection must surface as a syntax-error FAILURE code"
        );

        // After a FAILURE the Bolt state machine needs a RESET before
        // the next RUN on the same connection.
        let mut buf = Vec::new();
        encode_client(&mut buf, &ClientMessage::Reset).unwrap();
        write_chunked_message(&mut stream, &buf).await.unwrap();
        let _ = read_chunked_message(&mut stream).await.unwrap().unwrap(); // RESET SUCCESS

        // 2. The server is STILL UP: a normal query on the SAME
        //    connection succeeds. (If the deep RUN had crashed the
        //    worker, the connection would be dead and this would
        //    fail/hang.)
        let (ok_tag, _) = run_query(&mut stream, "RETURN 1").await;
        assert_eq!(
            ok_tag, TAG_SUCCESS,
            "{form}: after the deep-query FAILURE the server must still answer `RETURN 1` (it stayed up)"
        );

        // Reset again to a clean READY for the next form's RUN.
        buf.clear();
        encode_client(&mut buf, &ClientMessage::Reset).unwrap();
        write_chunked_message(&mut stream, &buf).await.unwrap();
        let _ = read_chunked_message(&mut stream).await.unwrap().unwrap();
    }

    let _ = shutdown.send(());
}
