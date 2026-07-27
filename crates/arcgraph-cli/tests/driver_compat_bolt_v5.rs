//! W24-DRIVERS-α — Layer 3 (subprocess TCP) Bolt 5.0 driver-compat
//! integration test.
//!
//! Per ADR-094 §D-4 "Driver-compat test triangle", this test closes the
//! v1.0-α driver-compat verification triangle by exercising the
//! production `arcgraph` CLI binary's `serve --bolt` path end-to-end:
//!
//! - Layer 1 (in-process): `crates/arcgraph-mcp/tests/bolt_e2e_python_driver_shape.rs`
//! - Layer 2 (in-tree TCP listener): `crates/arcgraph-mcp/tests/bolt_e2e_full_session.rs`
//! - **Layer 3 (this test): subprocess TCP** — spawn `arcgraph serve --bolt`;
//!   drive HANDSHAKE → HELLO → RUN → PULL → DISCARD → GOODBYE via the
//!   in-tree client codec; assert all 6 message classes round-trip.
//! - Layer 4 (operator-runnable, not CI): `tests/driver-compat/python/smoke.py`
//!
//! # Why subprocess + in-tree codec
//!
//! Layers 1 and 2 exercise the protocol against `bolt::serve_bolt_inner`
//! DIRECTLY — they bypass the CLI binary's argument parsing,
//! [`BoltServerConfig::validate`] startup gate, signal-handling shutdown
//! path, and per-task tracing instrumentation. A regression in that
//! wiring would not surface until production smoke.
//!
//! This test consumes `env!("CARGO_BIN_EXE_arcgraph")` per the W15α
//! M6-02 subprocess-test pattern, spawning the built binary with
//! `serve --bolt 127.0.0.1:<port>` and driving the Bolt 5.0 wire
//! protocol against the real listener. The in-tree client codec (same
//! types `crates/arcgraph-mcp/src/transport/bolt/` exports) is reused
//! so the test does not depend on `neo4rs` or another external Bolt
//! 5.0 driver crate (the existing
//! `bolt_e2e_python_driver_shape.rs` §"Why not depend on `neo4rs`?"
//! rationale applies identically here).
//!
//! # Env-gate + panic discipline
//!
//! Per `feedback_test_env_gate_panic_by_default.md` (W12δ HIGH-1) +
//! the W24-DRIVERS-α spawn prompt: the test PANICS by default if the
//! subprocess cannot be spawned, fails the wire round-trip, or hits
//! any unexpected protocol divergence. Set
//! `ARCGRAPH_DRIVER_COMPAT_SKIP_OK=1` to opt into a soft-skip that
//! exits with a clear message (the W24-DRIVERS-α workspace gauntlet
//! sets the variable so workspace-test invocations soft-skip; the
//! `cargo test -p arcgraph-cli --test driver_compat_bolt_v5` invocation
//! without the variable exercises the test).
//!
//! Soft-skip is NOT the default per the W12δ HIGH-1 founding incident
//! discipline: the bug class "test silently no-op'd when prerequisite
//! missing" is the worst of all worlds. PANIC by default; explicit
//! env-flag opt-out is the only safe shape.

use std::collections::BTreeMap;
use std::net::{SocketAddr, TcpListener as StdTcpListener};
use std::process::{Child, Stdio};
use std::time::{Duration, Instant};

use arcgraph_mcp::transport::bolt::{
    ClientMessage, MAGIC_PREAMBLE, PackValue, SERVER_ACCEPT_V5_0, decode, encode_client,
    message::{TAG_RECORD, TAG_SUCCESS},
    read_chunked_message, write_chunked_message,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

const ENV_DRIVER_COMPAT_SKIP_OK: &str = "ARCGRAPH_DRIVER_COMPAT_SKIP_OK";

/// Bounded wall-clock for subprocess spawn + listener readiness. Local
/// runs see ~200ms; CI sees ~1-2s. 30s is generous + matches the
/// W15α M6-02 subprocess-test budget.
const SPAWN_TIMEOUT: Duration = Duration::from_secs(30);

/// Bounded wall-clock for each wire round-trip step. The in-tree
/// listener is local-loopback; 5s is generous (real exchanges are
/// sub-millisecond).
const WIRE_STEP_TIMEOUT: Duration = Duration::from_secs(5);

/// The bolt magic + a one-version-offer handshake for Bolt 5.0. Same
/// shape the in-tree `bolt_e2e_full_session.rs::handshake` helper
/// emits — pinned here as a constant so the subprocess test reads
/// like a wire-protocol spec replay.
const HANDSHAKE_BYTES: [u8; 20] = {
    let mut buf = [0u8; 20];
    buf[0] = MAGIC_PREAMBLE[0];
    buf[1] = MAGIC_PREAMBLE[1];
    buf[2] = MAGIC_PREAMBLE[2];
    buf[3] = MAGIC_PREAMBLE[3];
    // Offer 1: [00, 00, 00, 05] — Bolt 5.0 exactly.
    buf[4] = 0x00;
    buf[5] = 0x00;
    buf[6] = 0x00;
    buf[7] = 0x05;
    // Offers 2-4: padding (zero) — server picks offer 1.
    buf
};

/// `RAII` guard for the spawned arcgraph subprocess. The Drop impl
/// sends SIGKILL on test scope exit so a panicking test does not leak
/// the child process. Graceful Ctrl-C path is preferred but the
/// SIGTERM-only `serve --bolt` exit cadence is bounded enough that
/// SIGKILL fallback is acceptable for a test (no on-disk state needs
/// flushing).
struct ChildGuard(Option<Child>);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if let Some(mut c) = self.0.take() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

/// Best-effort: bind 127.0.0.1:0, capture the OS-assigned port, drop
/// the listener so the subprocess can re-bind it. There IS a small
/// TOCTOU race window where another process could grab the port; we
/// accept it for a test (the race is microseconds on local-loopback,
/// and the test retries on bind-collision in `spawn_arcgraph_bolt`).
fn pick_loopback_port() -> u16 {
    let listener = StdTcpListener::bind(("127.0.0.1", 0))
        .expect("pick_loopback_port: bind 127.0.0.1:0 failed");
    let port = listener
        .local_addr()
        .expect("pick_loopback_port: local_addr failed")
        .port();
    drop(listener);
    port
}

/// Spawn `arcgraph serve --bolt 127.0.0.1:<port>` and return the
/// `Child` handle + the bound `SocketAddr`. Retries on bind-collision
/// up to 5 times, each with a freshly-picked port.
async fn spawn_arcgraph_bolt() -> (ChildGuard, SocketAddr) {
    let bin = env!("CARGO_BIN_EXE_arcgraph");
    let mut last_err: Option<String> = None;
    for attempt in 0..5 {
        let port = pick_loopback_port();
        let bind_str = format!("127.0.0.1:{port}");
        let addr: SocketAddr = bind_str.parse().expect("bind addr parses");
        // Inherit stderr so tracing output flows to test logs on failure;
        // null stdout to avoid pipe-fill backpressure on long-running
        // sessions (the serve subcommand does not write to stdout).
        // W28 / ADR-183 — `serve` refuses to start without an explicit
        // storage mode; the Bolt driver-compat smoke is ephemeral, so it
        // passes `--in-memory`.
        let child_res = std::process::Command::new(bin)
            .args(["serve", "--bolt", &bind_str, "--in-memory"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn();
        let mut child = match child_res {
            Ok(c) => c,
            Err(e) => {
                last_err = Some(format!("attempt {attempt}: spawn failed: {e}"));
                continue;
            }
        };
        // Poll the listener for readiness. The Bolt listener binds
        // synchronously inside `serve_bolt_listener`; we expect a
        // sub-second establishment window. Retry every 50ms up to
        // SPAWN_TIMEOUT.
        let deadline = Instant::now() + SPAWN_TIMEOUT;
        loop {
            if Instant::now() > deadline {
                // Subprocess never bound the port; kill + try a new port.
                let _ = child.kill();
                let _ = child.wait();
                last_err = Some(format!(
                    "attempt {attempt} port {port}: listener did not bind within {:?}",
                    SPAWN_TIMEOUT
                ));
                break;
            }
            // Check if subprocess died early (e.g., bind-collision).
            match child.try_wait() {
                Ok(Some(status)) => {
                    last_err = Some(format!(
                        "attempt {attempt} port {port}: subprocess exited early with {status:?}"
                    ));
                    break;
                }
                Ok(None) => { /* still running — keep polling */ }
                Err(e) => {
                    last_err = Some(format!(
                        "attempt {attempt} port {port}: try_wait failed: {e}"
                    ));
                    break;
                }
            }
            // Probe via TCP connect — bound + accept-loop running.
            if TcpStream::connect(addr).await.is_ok() {
                return (ChildGuard(Some(child)), addr);
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        // fell through — retry with a fresh port
    }
    panic!(
        "spawn_arcgraph_bolt: exhausted retries; last_err={:?}\n\n\
         Per feedback_test_env_gate_panic_by_default.md: set \
         ARCGRAPH_DRIVER_COMPAT_SKIP_OK=1 to opt into a soft-skip if \
         the host environment cannot bind a loopback port or build the \
         arcgraph binary.",
        last_err
    );
}

/// Drive the HANDSHAKE step of the round-trip.
async fn drive_handshake(stream: &mut TcpStream) {
    tokio::time::timeout(WIRE_STEP_TIMEOUT, stream.write_all(&HANDSHAKE_BYTES))
        .await
        .expect("HANDSHAKE write timeout")
        .expect("HANDSHAKE write failed");
    let mut resp = [0u8; 4];
    tokio::time::timeout(WIRE_STEP_TIMEOUT, stream.read_exact(&mut resp))
        .await
        .expect("HANDSHAKE response timeout")
        .expect("HANDSHAKE response read failed");
    assert_eq!(
        resp, SERVER_ACCEPT_V5_0,
        "subprocess server did not pick Bolt 5.0; got {resp:?}"
    );
}

/// Send one client message + read one server reply, returning the
/// decoded `PackValue` of the reply (server replies are Struct frames).
async fn round_trip_one(stream: &mut TcpStream, msg: &ClientMessage) -> PackValue {
    let mut buf = Vec::with_capacity(64);
    encode_client(&mut buf, msg).expect("encode client message");
    tokio::time::timeout(WIRE_STEP_TIMEOUT, write_chunked_message(stream, &buf))
        .await
        .expect("write_chunked_message timeout")
        .expect("write_chunked_message failed");
    let payload = tokio::time::timeout(WIRE_STEP_TIMEOUT, read_chunked_message(stream))
        .await
        .expect("read_chunked_message timeout")
        .expect("read_chunked_message I/O failed")
        .expect("read_chunked_message returned None (peer closed)");
    let (val, n) = decode(&payload, 0).expect("decode reply");
    assert_eq!(
        n,
        payload.len(),
        "reply decode consumed {n} of {} bytes",
        payload.len()
    );
    val
}

/// Read one server reply WITHOUT sending a client message — used after
/// GOODBYE to verify the server closed the socket (the next read sees
/// clean EOF, not a frame).
async fn read_eof(stream: &mut TcpStream) {
    let mut buf = [0u8; 1];
    match tokio::time::timeout(WIRE_STEP_TIMEOUT, stream.read(&mut buf)).await {
        Ok(Ok(0)) => { /* clean EOF — server closed after GOODBYE */ }
        Ok(Ok(n)) => panic!("expected EOF after GOODBYE; read {n} byte(s): {buf:?}"),
        Ok(Err(e)) => panic!("expected EOF after GOODBYE; read errored: {e}"),
        Err(_) => panic!("expected EOF after GOODBYE; read timed out"),
    }
}

/// Per ADR-094 §D-4: the W24-DRIVERS-α layer 3 verification surface.
/// Spawns the production `arcgraph` CLI binary with `serve --bolt`
/// + drives all 6 message classes (HANDSHAKE / HELLO / RUN / PULL /
/// DISCARD / GOODBYE) end-to-end through the subprocess TCP path.
///
/// # Wire-protocol-level assertions only
///
/// The CLI binary wires
/// [`arcgraph_mcp::storage::bolt::StorageBoltHandler`] (NOT the
/// [`arcgraph_mcp::transport::bolt::StubBoltHandler`] the in-process
/// tests use). The production handler executes Cypher through the
/// real `QueryEngine` against an empty substrate — for a fresh
/// session "RETURN 1" returns a SUCCESS with whatever the W17α
/// executor's projection-stage emits (currently: 0 rows + 0 columns
/// because `col_count` is derived from `result.rows().first()` per
/// `crates/arcgraph-mcp/src/storage/bolt.rs:252` — a pre-existing
/// limitation tracked at issue #353).
///
/// This test therefore asserts WIRE-LEVEL invariants only:
/// every reply is a SUCCESS Struct frame with the expected
/// metadata-Map shape (no FAILURE; `fields` list present in RUN
/// reply; `has_more` Boolean in PULL/DISCARD reply). Per-handler
/// result-shape assertions belong to the in-process layer-1 test
/// (`crates/arcgraph-mcp/tests/bolt_e2e_python_driver_shape.rs`)
/// which uses the deterministic StubBoltHandler.
///
/// # Two-RUN structure to exercise both PULL and DISCARD
///
/// The FSM (`crates/arcgraph-mcp/src/transport/bolt/state.rs:158`)
/// requires DISCARD to fire while in `Streaming`. After PULL n=-1
/// returns has_more=false the FSM transitions back to Ready;
/// DISCARD on Ready is a ProtocolViolation. To exercise BOTH PULL
/// and DISCARD in one session, the test issues two RUN cycles:
///
/// Cycle 1: RUN → PULL n=-1 → drains all (possibly 0) records →
///          SUCCESS{has_more=false} → back to Ready
/// Cycle 2: RUN → DISCARD n=-1 → drops all remaining records →
///          SUCCESS{has_more=false} → back to Ready
///
/// Both cycles exercise the same RUN code path; the second cycle
/// is the only path that admits DISCARD into the wire trace.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn subprocess_bolt_v5_full_six_message_round_trip() {
    if std::env::var(ENV_DRIVER_COMPAT_SKIP_OK).as_deref() == Ok("1") {
        eprintln!(
            "[driver_compat_bolt_v5] {ENV_DRIVER_COMPAT_SKIP_OK}=1 — soft-skip per \
             feedback_test_env_gate_panic_by_default.md opt-out. To exercise the \
             subprocess test: unset {ENV_DRIVER_COMPAT_SKIP_OK} and re-run."
        );
        return;
    }

    // 1. Spawn `arcgraph serve --bolt 127.0.0.1:<port>` + poll for
    //    listener readiness (bounded retry on bind race).
    let (_child, addr) = spawn_arcgraph_bolt().await;

    // 2. Connect to the subprocess listener.
    let mut stream = tokio::time::timeout(WIRE_STEP_TIMEOUT, TcpStream::connect(addr))
        .await
        .expect("TCP connect timeout")
        .expect("TCP connect failed");

    // 3. HANDSHAKE — pin: server picks Bolt 5.0.
    drive_handshake(&mut stream).await;

    // 4. HELLO — pin: SUCCESS with server-side connection metadata
    //    (connection_id + server slug present per ADR-094 §D-1
    //    driver-author contract).
    let hello_reply = round_trip_one(
        &mut stream,
        &ClientMessage::Hello {
            user_agent: Some("arcgraph-driver-compat-v5/1.0".into()),
            scheme: Some("basic".into()),
            principal: Some("compat-smoke".into()),
            credentials: Some("compat-secret".into()),
            routing: None,
            extras: BTreeMap::new(),
        },
    )
    .await;
    assert_hello_success(hello_reply);

    // 5. Cycle 1 — RUN "MATCH (n) RETURN n" — pin: SUCCESS with
    //    `fields` list present. We use MATCH because the production
    //    `StorageBoltHandler::run` test at
    //    `crates/arcgraph-mcp/src/storage/bolt.rs:444` pins this
    //    query as the "exercise the executor without depending on
    //    catalog seed" smoke; bare "RETURN 1" hits a pre-existing
    //    column_count = 0 case (issue #353) the wire-protocol test
    //    must not assume away.
    let run_reply_1 = round_trip_one(
        &mut stream,
        &ClientMessage::Run {
            query: "MATCH (n) RETURN n".into(),
            parameters: BTreeMap::new(),
            extra: BTreeMap::new(),
        },
    )
    .await;
    assert_run_success_with_fields_list(run_reply_1);

    // 6. PULL n=-1 — pin: SUCCESS with `has_more=false` (drains all
    //    rows; on an empty substrate the row count is 0, so the
    //    server emits zero RECORD frames + one tail SUCCESS).
    let pull_reply = round_trip_one(&mut stream, &ClientMessage::Pull { n: -1, qid: None }).await;
    // `assert_pull_or_discard_success` is permissive on has_more —
    // the wire-level assertion is "metadata Map carries has_more as
    // a Boolean," not a value claim that depends on substrate state.
    assert_pull_or_discard_success(pull_reply);

    // 7. Cycle 2 — RUN again (FSM is back at Ready post-PULL with
    //    has_more=false). The second RUN re-enters Streaming so
    //    DISCARD is admissible.
    let run_reply_2 = round_trip_one(
        &mut stream,
        &ClientMessage::Run {
            query: "MATCH (n) RETURN n".into(),
            parameters: BTreeMap::new(),
            extra: BTreeMap::new(),
        },
    )
    .await;
    assert_run_success_with_fields_list(run_reply_2);

    // 8. DISCARD n=-1 — pin: SUCCESS with `has_more=false` (drops
    //    all remaining rows). This is the canonical DISCARD message-
    //    class exercise; without the second RUN cycle, the FSM
    //    rejects DISCARD-on-Ready as ProtocolViolation per
    //    `crates/arcgraph-mcp/src/transport/bolt/state.rs:150`.
    let discard_reply =
        round_trip_one(&mut stream, &ClientMessage::Discard { n: -1, qid: None }).await;
    assert_pull_or_discard_success(discard_reply);

    // 8b. T18 (ADR-147-amendment-03, D-1) — drive a NON-LITERAL
    //     `UNWIND $rows AS r CREATE (n {v: r.v})` over the SERVED TCP
    //     path. This exercises the live `CreateSpineOp` with a `$rows`
    //     parameter through the production CLI binary's `serve --bolt`
    //     handler (the spine is where the amendment wires eval). A wire-
    //     level SUCCESS pins that the served path accepts + executes the
    //     lever (RED on revert: pre-amendment the served handler rejected
    //     the non-literal property with a FAILURE).
    let rows_param = PackValue::List(vec![
        PackValue::Map(
            [("v".to_string(), PackValue::Integer(10))]
                .into_iter()
                .collect(),
        ),
        PackValue::Map(
            [("v".to_string(), PackValue::Integer(20))]
                .into_iter()
                .collect(),
        ),
    ]);
    // DISCARD (not PULL) drops all result rows and replies with a single
    // tail SUCCESS regardless of the record count — the harness's
    // `round_trip_one` reads exactly one frame, so DISCARD keeps the
    // read-one-frame invariant while still draining the created-node
    // summary rows. The RUN SUCCESS + DISCARD SUCCESS jointly pin that
    // the served spine accepted + executed the non-literal-property batch.
    let run_reply_unwind = round_trip_one(
        &mut stream,
        &ClientMessage::Run {
            query: "UNWIND $rows AS r CREATE (n:BatchT18 {v: r.v})".into(),
            parameters: [("rows".to_string(), rows_param)].into_iter().collect(),
            extra: BTreeMap::new(),
        },
    )
    .await;
    assert_run_success_with_fields_list(run_reply_unwind);
    let discard_unwind =
        round_trip_one(&mut stream, &ClientMessage::Discard { n: -1, qid: None }).await;
    assert_pull_or_discard_success(discard_unwind);

    // 9. GOODBYE — pin: server closes the socket cleanly with NO reply.
    let mut goodbye_buf = Vec::with_capacity(8);
    encode_client(&mut goodbye_buf, &ClientMessage::Goodbye).expect("encode GOODBYE");
    tokio::time::timeout(
        WIRE_STEP_TIMEOUT,
        write_chunked_message(&mut stream, &goodbye_buf),
    )
    .await
    .expect("GOODBYE write timeout")
    .expect("GOODBYE write failed");
    read_eof(&mut stream).await;

    // Subprocess kept alive by the ChildGuard; Drop sends SIGKILL on
    // test scope exit. Per the testing discipline and the
    // W15α M6-02 subprocess-test pattern: graceful shutdown is via
    // SIGTERM (the production path), but for a test the SIGKILL
    // fallback is bounded (no on-disk state to flush at this scope).
}

/// Validate the HELLO reply's structure: tag = SUCCESS, metadata
/// includes `connection_id` + `server` (per ADR-094 §D-1 driver-author
/// contract: real Python drivers read `server` to determine dialect
/// rules; pin its presence + shape).
fn assert_hello_success(reply: PackValue) {
    match reply {
        PackValue::Struct { tag, fields } => {
            assert_eq!(
                tag, TAG_SUCCESS,
                "HELLO reply must be SUCCESS; got tag {tag:#x}"
            );
            let meta = first_map_field(fields, "HELLO SUCCESS");
            assert!(
                matches!(meta.get("connection_id"), Some(PackValue::String(_))),
                "HELLO SUCCESS must carry `connection_id` (driver diagnostics requirement); meta={meta:?}"
            );
            match meta.get("server") {
                Some(PackValue::String(s)) => {
                    assert!(
                        s.starts_with("ArcGraph/"),
                        "server slug must start with 'ArcGraph/' for driver dialect-rule selection; got `{s}`"
                    );
                }
                other => panic!("HELLO SUCCESS missing/invalid `server`: {other:?}"),
            }
        }
        other => panic!("expected HELLO reply struct; got {other:?}"),
    }
}

/// Validate the RUN reply at the wire level: tag = SUCCESS, metadata
/// `fields` is present + is a List (length may be 0+). Per the
/// per-test §"Wire-protocol-level assertions only" rustdoc, the test
/// MUST NOT assume StubBoltHandler-specific column names or counts;
/// the production handler emits `col_0..N` shaped names derived from
/// the first row's column count, which can be 0 on an empty substrate.
fn assert_run_success_with_fields_list(reply: PackValue) {
    match reply {
        PackValue::Struct { tag, fields } => {
            assert_eq!(
                tag, TAG_SUCCESS,
                "RUN reply must be SUCCESS at wire level; got tag {tag:#x}"
            );
            let meta = first_map_field(fields, "RUN SUCCESS");
            match meta.get("fields") {
                Some(PackValue::List(_)) => { /* present + correctly typed; length unrestricted */ }
                other => panic!(
                    "RUN SUCCESS metadata missing/invalid `fields` field (must be List); got {other:?}"
                ),
            }
        }
        other => panic!("expected RUN reply struct; got {other:?}"),
    }
}

/// Validate the PULL / DISCARD reply at the wire level: tag = SUCCESS,
/// metadata `has_more` is present + is a Boolean (value unrestricted —
/// depends on substrate state). Both messages share the SUCCESS shape
/// per `crates/arcgraph-mcp/src/transport/bolt/server.rs:787` reply
/// table.
fn assert_pull_or_discard_success(reply: PackValue) {
    match reply {
        PackValue::Struct { tag, fields } => {
            assert_eq!(
                tag, TAG_SUCCESS,
                "PULL/DISCARD reply must be SUCCESS at wire level; got tag {tag:#x}"
            );
            let meta = first_map_field(fields, "PULL/DISCARD SUCCESS");
            match meta.get("has_more") {
                Some(PackValue::Boolean(_)) => { /* present + correctly typed; value unrestricted */
                }
                other => panic!(
                    "PULL/DISCARD SUCCESS metadata missing/invalid `has_more` (must be Boolean); got {other:?}"
                ),
            }
        }
        other => panic!("expected PULL/DISCARD reply struct; got {other:?}"),
    }
}

/// Helper: pull the first field of a Struct frame and assert it is a
/// Map (every SUCCESS / FAILURE / RECORD frame's first field is the
/// metadata Map / records List per Bolt 5.0 message catalog).
fn first_map_field(mut fields: Vec<PackValue>, label: &str) -> BTreeMap<String, PackValue> {
    let first = fields
        .drain(0..1)
        .next()
        .unwrap_or_else(|| panic!("{label}: struct has no fields"));
    match first {
        PackValue::Map(m) => m,
        other => panic!("{label}: first field is not a Map: {other:?}"),
    }
}

// Smoke-pin: the test runs through ALL 6 Bolt 5.0 message classes the
// ADR-094 §D-4 triangle commits to. A separate "rejection" smoke
// proves the subprocess listener honors the spec's reject-with-zero
// response for non-5.0 offers — distinct concern (server behavior
// under bad input), so a separate test.
//
// Per the testing discipline, integration tests in `tests/`
// for every cross-crate API. The CLI ↔ Bolt transport ↔ in-tree codec
// path is cross-crate (arcgraph-cli → arcgraph-mcp → in-tree codec),
// so this test is in the right place per the M6-02 W15α pattern.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn subprocess_bolt_v5_rejects_v4_4_only_offer_set() {
    if std::env::var(ENV_DRIVER_COMPAT_SKIP_OK).as_deref() == Ok("1") {
        eprintln!(
            "[driver_compat_bolt_v5] {ENV_DRIVER_COMPAT_SKIP_OK}=1 — soft-skip per \
             feedback_test_env_gate_panic_by_default.md opt-out."
        );
        return;
    }

    let (_child, addr) = spawn_arcgraph_bolt().await;
    let mut stream = tokio::time::timeout(WIRE_STEP_TIMEOUT, TcpStream::connect(addr))
        .await
        .expect("TCP connect timeout")
        .expect("TCP connect failed");

    // ADR-094 §D-2 — Bolt 4.4 is rejected at v1.0-α. Replay an offer
    // set that ONLY includes 4.x versions (no 5.0); expect the
    // 4-byte zero response per Bolt §"Handshake".
    let mut req = Vec::with_capacity(20);
    req.extend_from_slice(&MAGIC_PREAMBLE);
    req.extend_from_slice(&[0x00, 0x00, 0x04, 0x04]); // Bolt 4.4
    req.extend_from_slice(&[0x00, 0x00, 0x03, 0x04]); // Bolt 4.3
    req.extend_from_slice(&[0x00, 0x00, 0x02, 0x04]); // Bolt 4.2
    req.extend_from_slice(&[0x00, 0x00, 0x01, 0x04]); // Bolt 4.1
    tokio::time::timeout(WIRE_STEP_TIMEOUT, stream.write_all(&req))
        .await
        .expect("4.x handshake write timeout")
        .expect("4.x handshake write failed");

    let mut resp = [0u8; 4];
    tokio::time::timeout(WIRE_STEP_TIMEOUT, stream.read_exact(&mut resp))
        .await
        .expect("rejection response timeout")
        .expect("rejection response read failed");
    assert_eq!(
        resp, [0u8; 4],
        "subprocess server must reject Bolt 4.x with the 4-byte zero \
         response per ADR-094 §D-2 (Bolt 4.4 deferred to v1.1+); got {resp:?}"
    );

    // Per the spec, the client is expected to close after a zero
    // response. We verify the server has dropped the socket cleanly
    // (next read returns EOF, not a frame).
    let mut buf = [0u8; 1];
    match tokio::time::timeout(WIRE_STEP_TIMEOUT, stream.read(&mut buf)).await {
        Ok(Ok(0)) => { /* clean EOF — server closed after rejection */ }
        Ok(Ok(n)) => panic!("expected EOF after rejection; read {n} byte(s)"),
        Ok(Err(_)) | Err(_) => { /* peer-reset / timeout acceptable here */ }
    }
}

/// Smoke-pin: the test harness's `RECORD` tag re-export is consumed by
/// downstream assertion sites; pin it touched so the unused-import
/// lint trips if the module's public surface changes. This pin is the
/// W14δ M5-13 surface-stability discipline applied to the
/// driver-compat layer 3 surface.
#[test]
fn driver_compat_module_uses_record_tag_constant() {
    let _ = TAG_RECORD;
}
