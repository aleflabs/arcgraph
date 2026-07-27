//! W13δ M5-01 fix-up — subprocess-based end-to-end stdio MCP test.
//!
//! Exercises the `arcgraph-mcp-stdio` binary as a true subprocess:
//! spawn it, write a Content-Length-framed JSON-RPC envelope to its
//! stdin, read the framed response from its stdout, verify the
//! response shape, then send SIGTERM (or close stdin) and confirm a
//! clean exit.
//!
//! # Why this lives in `arcgraph-cli/tests/`
//!
//! `env!("CARGO_BIN_EXE_arcgraph-mcp-stdio")` only resolves for tests
//! in the same package as the binary (per Cargo docs:
//! <https://doc.rust-lang.org/cargo/reference/environment-variables.html#environment-variables-cargo-sets-for-crates>).
//! The library-side integ tests in
//! `arcgraph-mcp/tests/mcp_stdio_integ.rs` already pin the in-memory
//! pipe path; this test pins the production-deployment shape.
//!
//! # What it pins
//!
//! 1. `cargo build --bin arcgraph-mcp-stdio` produces a runnable
//!    binary (the v1.0-alpha "MCP transport works end-to-end" surface
//!    the PR #286 review packet HIGH-2 asked for).
//! 2. The binary's stdout is a Content-Length-framed JSON-RPC stream
//!    (matches the Anthropic MCP spec 2025-11-25 stdio shape).
//! 3. A `graph.schema` request gets a valid JSON-RPC response
//!    envelope with the empty-tenant schema body.
//! 4. Closing stdin causes a clean shutdown (`ExitReason::PeerClosed`
//!    path in `serve_stdio`).

use std::process::Stdio;
use std::time::Duration;

use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio::time::timeout;

/// Per-test wallclock budget. Generous because the binary may be
/// recompiled on the first invocation.
const SUBPROCESS_TIMEOUT: Duration = Duration::from_secs(30);

/// Read one Content-Length-framed JSON-RPC envelope from `reader`.
///
/// Mirrors the framer in `arcgraph_mcp::jsonrpc::read_message` (which
/// is private to the library + used by `serve_stdio`); this test
/// reproduces the wire format inline so a regression in the framer
/// surfaces as a test failure here, not a silent passthrough.
async fn read_framed_response<R>(reader: &mut R) -> std::io::Result<Value>
where
    R: tokio::io::AsyncBufRead + Unpin,
{
    let mut content_length: Option<usize> = None;
    let mut line = String::new();
    loop {
        line.clear();
        let n = tokio::io::AsyncBufReadExt::read_line(reader, &mut line).await?;
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "EOF before headers complete",
            ));
        }
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            // End of headers.
            break;
        }
        if let Some(value) = trimmed.strip_prefix("Content-Length:") {
            let value = value.trim();
            content_length = Some(value.parse::<usize>().map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, format!("bad len: {e}"))
            })?);
        }
    }
    let len = content_length.ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "no Content-Length header found",
        )
    })?;
    let mut body = vec![0u8; len];
    AsyncReadExt::read_exact(reader, &mut body).await?;
    serde_json::from_slice(&body)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("bad JSON: {e}")))
}

/// Render a JSON value as a Content-Length-framed envelope (matches
/// `arcgraph_mcp::jsonrpc::write_message`).
fn frame_request(req: &Value) -> Vec<u8> {
    let body = serde_json::to_vec(req).expect("request always serializes");
    let mut out = Vec::with_capacity(body.len() + 32);
    out.extend_from_slice(format!("Content-Length: {}\r\n\r\n", body.len()).as_bytes());
    out.extend_from_slice(&body);
    out
}

/// `graph.schema` request to the empty-tenant binary.
#[tokio::test]
async fn integ_subprocess_graph_schema_returns_empty_tenant_envelope() {
    let bin = env!("CARGO_BIN_EXE_arcgraph-mcp-stdio");

    let mut child = Command::new(bin)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn arcgraph-mcp-stdio binary");

    let mut stdin = child.stdin.take().expect("stdin piped");
    let stdout = child.stdout.take().expect("stdout piped");

    // TenantId::DEFAULT == 1 (per crates/arcgraph-core/src/ids.rs:202).
    let request = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "graph.schema",
        "params": {
            "tenant_id": 1,
        }
    });

    let framed = frame_request(&request);

    timeout(SUBPROCESS_TIMEOUT, async {
        stdin
            .write_all(&framed)
            .await
            .expect("write framed request");
        stdin.flush().await.expect("flush framed request");
    })
    .await
    .expect("subprocess accepted request within timeout");

    let mut reader = BufReader::new(stdout);
    let response = timeout(SUBPROCESS_TIMEOUT, read_framed_response(&mut reader))
        .await
        .expect("subprocess emitted response within timeout")
        .expect("response framed correctly");

    // Pin the JSON-RPC envelope shape.
    assert_eq!(response.get("jsonrpc").and_then(Value::as_str), Some("2.0"));
    assert_eq!(response.get("id").and_then(Value::as_i64), Some(1));
    assert!(
        response.get("error").is_none(),
        "expected success response; got error: {response:?}"
    );

    // The result is `{"format": "yaml", "body": "<yaml-text>"}` per
    // `arcgraph_mcp::render_response` (M5-04 default-format = YAML).
    let result = response
        .get("result")
        .expect("response carries result slot");
    assert_eq!(
        result.get("format").and_then(Value::as_str),
        Some("yaml"),
        "graph.schema default format is YAML"
    );
    let body = result
        .get("body")
        .and_then(Value::as_str)
        .expect("body is a YAML string");
    assert!(body.contains("tenant_id"), "schema YAML mentions tenant_id");
    assert!(
        body.contains("labels"),
        "schema YAML mentions labels (even if empty)"
    );

    // Close stdin — the binary should exit cleanly via PeerClosed.
    drop(stdin);

    let exit_status = timeout(SUBPROCESS_TIMEOUT, child.wait())
        .await
        .expect("subprocess exited within timeout")
        .expect("wait returned ExitStatus");

    assert!(
        exit_status.success(),
        "binary exited non-zero: {exit_status:?}"
    );
}

// SIGTERM-drain path is library-tested at
// `crates/arcgraph-mcp/src/transport/stdio.rs::serve_stdio_returns_on
// _shutdown_signal` + `serve_stdio_flushes_writer_on_shutdown`. The
// binary just composes `shutdown_on_term()` (a SIGTERM-aware future
// per `transport/stdio.rs:242-277`) into `serve_stdio` — that
// composition is statically obvious in
// `crates/arcgraph-cli/src/bin/arcgraph_mcp_stdio.rs::run`. A
// subprocess-level SIGTERM test would require a new `nix` /
// `libc::kill` dep (PD-1 concern); the close-stdin (PeerClosed) test
// above suffices as the binary smoke pin. M5-02 streamable-HTTP +
// production deployment carry a runtime SIGTERM-drain Criterion bench.
