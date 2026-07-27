//! W13δ M5-01 — JSON-RPC 2.0 envelope + Content-Length stream framing.
//!
//! MCP's stdio transport (per the Anthropic MCP spec, 2025-11-25)
//! borrows the LSP / Microsoft Debug Adapter Protocol framing shape:
//! each message is preceded by a CRLF-delimited HTTP-like header
//! block whose `Content-Length: <N>` declares the byte length of the
//! JSON payload that follows the blank line. Concretely:
//!
//! ```text
//! Content-Length: 87\r\n
//! \r\n
//! {"jsonrpc":"2.0","id":1,"method":"graph.schema","params":{"tenant_id":1}}
//! ```
//!
//! Implementing the framing here (rather than reaching for a generic
//! LSP crate) keeps the dep surface small and avoids pulling in
//! `tower-lsp` / `lsp-server` (both pull async-plumbing surfaces we
//! don't need at v1.0-alpha). The shape is small enough (≈100 LOC)
//! that the maintenance cost is bounded.
//!
//! # Headers we accept / emit
//!
//! - **`Content-Length: <decimal>`** — REQUIRED. Decimal byte count of
//!   the payload that follows. Per LSP §3 we accept any decimal
//!   payload; the framer caps the value at [`MAX_MESSAGE_BYTES`] to
//!   prevent a hostile peer from claiming a multi-GB body.
//! - **`Content-Type: application/vscode-jsonrpc; charset=utf-8`** —
//!   OPTIONAL. We tolerate / ignore it on input; we don't emit it on
//!   output (the MCP spec does not mandate it; LSP §3 marks it
//!   optional).
//!
//! # Errors
//!
//! All framing-layer faults surface as
//! [`crate::MCPError::ParseError`] (envelope unparseable) or
//! [`crate::MCPError::InvalidRequest`] (envelope parsed but missing
//! required JSON-RPC fields).
//!
//! # ADR provenance
//! - **ADR-004 §"Tier-1 (agent-facing, default)"** — MCP tool surface
//!   (six active tools under a 10-tool cap); this module is the
//!   wire-format substrate.
//! - **ADR-038 amendment-03 §M5↔M4 contract surface** — the request
//!   envelope's `id` is the M5-side `request_id` that the per-query
//!   tracing span tags + the [`crate::MCPError::Cancelled`] / cancel
//!   surface routes through.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader};

use crate::error::MCPError;

/// Hard cap on a single message's `Content-Length` for the UNTRUSTED-peer
/// transports (HTTP / future WebSocket). Defends against a hostile peer
/// claiming a multi-GB body (the framer would otherwise `vec![0u8; n]`
/// that many bytes) — this is the allocation-DoS bound, not a
/// functional limit. v1.0-alpha cap: 16 MiB. This is the same
/// envelope-level cap ADR-004 amendment-03 §D-1 names as
/// `MAX_MESSAGE_BYTES = 16 MiB` (the envelope under which the Tier-2
/// `graph.raw_query` `MAX_RAW_QUERY_BYTES = 1 MiB` sits as
/// defense-in-depth); the Tier-1 read tools (`graph.schema` /
/// `graph.inspect` / `graph.search`) return result sets well inside it.
///
/// This is the cap [`crate::transport::http`] enforces on the inbound
/// request body (→ HTTP 413). The stdio transport — whose peer is the
/// trusted local MCP host that spawned the process — uses the larger
/// [`STDIO_MAX_MESSAGE_BYTES`] instead, because the bulk-data WRITE tool
/// `graph.ingest` (ADR-004 amendment-01, structured-record ingest) can
/// produce a frame far larger than the read surface (see #818).
///
/// Forward-method: the M5-12 rate-limit config will surface a
/// per-tenant override; v1.0-alpha keeps the two static caps.
pub const MAX_MESSAGE_BYTES: usize = 16 * 1024 * 1024;

/// Per-frame `Content-Length` cap for the **stdio** transport, whose peer
/// is the trusted local MCP host that spawned the process (design-v2 §9.4
/// "Transport and security" classes stdio as the local transport — "stdio
/// (for local)", with "HTTPS enforced for non-stdio transports" — so the
/// parent process, not a network peer, bounds the request rate here).
/// v1.0-alpha cap: **512 MiB**.
///
/// # Why a separate, larger cap (#818)
///
/// `graph.ingest` is the one MCP tool that IS bulk-data: a single batch of
/// N embedding-bearing nodes is `N × dim × ~19 bytes` of JSON on the wire
/// (a 128-d `f32` vector widens to ~18-char `f64` decimals + commas).
/// Under the shared 16 MiB [`MAX_MESSAGE_BYTES`] a single-batch ingest was
/// silently rejected by the framer above only ~6 300 × 128-d vectors
/// (16 MiB ÷ ~2 665 bytes/node) — the served HNSW then had nothing to
/// build, so `graph.search` returned empty with recall 0 (#818). The cap
/// belongs to the untrusted-network surface, not the trusted-local
/// bulk-ingest one.
///
/// # Back-of-envelope (PD#5)
///
/// Target: a 100 000 × 128-d single-batch ingest (the #818 scalability
/// floor) is `100 000 × ~2 665 ≈ 254 MiB` on the wire. 512 MiB gives ~2×
/// headroom for that floor and accommodates the production chunk size
/// (`cz_sift_recall.py` chunks at 25 000 ≈ 64 MiB). The cost is a
/// worst-case 512 MiB single-frame `vec![0u8; len]` on the trusted-local
/// stdio path (sequential dispatch at v1.0-α → one in flight at a time);
/// over-cap frames are drained (not allocated) and surfaced as a clear
/// error (never silent — see [`read_message_with_cap`]).
pub const STDIO_MAX_MESSAGE_BYTES: usize = 512 * 1024 * 1024;

/// Compile-time invariant (#818): the stdio cap MUST exceed the default
/// (untrusted-peer) cap — otherwise the bulk-ingest path is no wider than
/// the cliff it was raised to clear. A future edit that inverts the two
/// constants fails to compile here rather than silently re-introducing the
/// regression.
const _: () = assert!(STDIO_MAX_MESSAGE_BYTES > MAX_MESSAGE_BYTES);

/// JSON-RPC 2.0 protocol version literal (per spec §3 envelope).
pub const JSONRPC_VERSION: &str = "2.0";

/// Stdio framing mode detected from the first non-empty bytes on a
/// connection. `Content-Length` is the legacy LSP-style framing ArcGraph
/// already shipped; newline JSON is the MCP stdio shape used by standard SDK
/// clients (one JSON message per line, no embedded raw newlines).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StdioFramingMode {
    /// Legacy LSP-style `Content-Length: N\r\n\r\n<body>` framing.
    ContentLength,
    /// MCP stdio newline-delimited JSON framing.
    NewlineJson,
}

// ─────────────────────────────────────────────────────────────────────
// Envelope types
// ─────────────────────────────────────────────────────────────────────

/// JSON-RPC 2.0 request envelope.
///
/// Per JSON-RPC 2.0 spec §4: the `id` MAY be a number, a string, or
/// `null`. `params` is OPTIONAL; absent params are treated as the
/// empty object. Notifications (no `id`) are PERMITTED by the spec
/// but not produced by MCP clients in practice; we accept them on
/// input but the dispatch path treats them as fire-and-forget (no
/// response is sent).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JsonRpcRequest {
    /// MUST equal "2.0" per spec §3. The deserializer enforces.
    pub jsonrpc: String,
    /// Method name. For W13δ the recognized names are `graph.schema`
    /// and `graph.inspect`; future M5 sub-slices add more.
    pub method: String,
    /// Method-specific parameter object. Absent → null at the wire
    /// level; we normalize to `Value::Null` in the deserializer
    /// default.
    #[serde(default)]
    pub params: Value,
    /// Request id. `Some` → request-response pair; `None` →
    /// notification (no reply expected). Per spec §4.2 the id MUST be
    /// a number or string — we accept any JSON value to match real-
    /// world MCP clients (the id is opaque to the dispatcher; it just
    /// echoes it on the response).
    pub id: Option<Value>,
}

/// JSON-RPC 2.0 success response envelope.
///
/// Per spec §5 a response carries EITHER `result` OR `error`, never
/// both. The two variants are split here so misuse at the call-site
/// is structurally impossible.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JsonRpcResponse {
    /// MUST equal "2.0".
    pub jsonrpc: String,
    /// Echoed from the request envelope. `Value::Null` when the
    /// originating request id was missing or null (some clients send
    /// `null`).
    pub id: Value,
    /// Method-specific result value. Tools serialize their outputs
    /// (TOON / YAML / JSON per the format hint) and embed the
    /// rendered string under `{ "format": ..., "body": ... }` so
    /// clients can route on format.
    pub result: Value,
}

/// JSON-RPC 2.0 error response envelope.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JsonRpcErrorResponse {
    /// MUST equal "2.0".
    pub jsonrpc: String,
    /// Echoed from the request envelope (or `null` when the request
    /// itself was unparseable).
    pub id: Value,
    /// Structured error per JSON-RPC §5.1.
    pub error: JsonRpcErrorObject,
}

/// JSON-RPC 2.0 error object per spec §5.1.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JsonRpcErrorObject {
    /// Numeric error code per the table in [`crate::error`].
    pub code: i32,
    /// Short human-readable message.
    pub message: String,
    /// Optional structured detail.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl JsonRpcResponse {
    /// Build a success response for the given request id + result.
    #[must_use]
    pub fn success(id: Value, result: Value) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.into(),
            id,
            result,
        }
    }
}

impl JsonRpcErrorResponse {
    /// Build an error response from the given request id + MCPError.
    /// The MCPError's variant code + message + optional data populate
    /// the JSON-RPC error object.
    #[must_use]
    pub fn from_mcp(id: Value, err: &MCPError) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.into(),
            id,
            error: JsonRpcErrorObject {
                code: err.code(),
                message: err.message().into(),
                data: err.data(),
            },
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Content-Length framer
// ─────────────────────────────────────────────────────────────────────

/// Read one framed JSON-RPC message from `reader`, enforcing the default
/// [`MAX_MESSAGE_BYTES`] cap (the untrusted-peer surface).
///
/// The stdio serve loop calls [`read_message_with_cap`] with the larger
/// [`STDIO_MAX_MESSAGE_BYTES`] instead, because its peer is the trusted
/// local MCP host and `graph.ingest` is bulk-data (#818).
pub async fn read_message<R>(reader: &mut BufReader<R>) -> Result<Option<Value>, MCPError>
where
    R: AsyncRead + Unpin,
{
    read_message_with_cap(reader, MAX_MESSAGE_BYTES).await
}

/// Read one framed JSON-RPC message from `reader`, capping the declared
/// body length at `max_message_bytes`.
///
/// Returns `Ok(Some(json))` on a successful read, `Ok(None)` on a
/// clean EOF (the peer closed the stream after the last complete
/// message), or `Err(MCPError::ParseError)` on a framing or JSON
/// parse fault.
///
/// The framer tolerates blank lines + duplicate-CR before the header
/// block (LSP §3 admits CRLF / LF / CR line endings; some peers send
/// stray empty lines on session start). It does NOT tolerate out-of-
/// order headers; the `Content-Length` MUST appear at least once.
///
/// # Over-cap handling (#818)
///
/// A frame whose declared `Content-Length` exceeds `max_message_bytes`
/// is NOT allocated (`vec![0u8; len]` is the very memory amplification
/// the cap defends against). Instead the announced body is *drained* in
/// bounded chunks so the stream stays in sync for the next frame — a
/// single oversized frame surfaces ONE clear, actionable error and does
/// NOT desync or silently truncate the session — then an actionable
/// [`MCPError::ParseError`] naming the size + cap is returned.
///
/// # Latency budget
///
/// One `read_until(b'\n')` per header line + one `read_exact(N)` for
/// the body. v1.0-alpha targets sub-millisecond framing per message
/// for headers ≤ 256 bytes + body ≤ `max_message_bytes`. The
/// `BufReader` in front of `reader` amortizes the per-call syscall cost.
pub async fn read_message_with_cap<R>(
    reader: &mut BufReader<R>,
    max_message_bytes: usize,
) -> Result<Option<Value>, MCPError>
where
    R: AsyncRead + Unpin,
{
    // Step 1: read the header block (CRLF-delimited; terminated by an
    // empty line). We accept LF-only line endings in addition to
    // CRLF to be generous to peers running on platforms that strip
    // CR (the LSP spec explicitly says the line ending is CRLF, but
    // real-world peers vary).
    let mut content_length: Option<usize> = None;
    let mut header_buf = String::new();
    let mut saw_any_header = false;
    loop {
        header_buf.clear();
        let n = reader
            .read_line(&mut header_buf)
            .await
            .map_err(|e| MCPError::ParseError(format!("header read: {e}")))?;
        if n == 0 {
            // EOF. If we've already started reading headers, that's a
            // truncated framing. Otherwise it's a clean session close.
            if saw_any_header || content_length.is_some() {
                return Err(MCPError::ParseError(
                    "EOF after partial header block".into(),
                ));
            }
            return Ok(None);
        }
        // Trim any trailing CR/LF.
        let line = header_buf.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            // Blank separator line. If we have NOT seen any header
            // yet, tolerate it (some peers send an extra blank on
            // session start). Otherwise the blank line ends the
            // header block and we MUST have seen Content-Length —
            // a missing-Content-Length here is a framing fault, not
            // a tolerated blank.
            if !saw_any_header {
                continue;
            }
            if content_length.is_none() {
                return Err(MCPError::ParseError(
                    "header block ended without Content-Length".into(),
                ));
            }
            break;
        }
        saw_any_header = true;
        // Parse `Header-Name: value`.
        let Some((name, value)) = line.split_once(':') else {
            return Err(MCPError::ParseError(format!(
                "malformed header line: {line:?}"
            )));
        };
        let name = name.trim();
        let value = value.trim();
        if name.eq_ignore_ascii_case("content-length") {
            let n: usize = value
                .parse()
                .map_err(|e| MCPError::ParseError(format!("Content-Length parse: {e}")))?;
            // The cap is enforced AFTER the header block ends (below), once
            // the reader is positioned at the body — so an over-cap frame's
            // body can be drained to keep the stream in sync (#818) instead
            // of being left in the pipe to desync the next read.
            content_length = Some(n);
        } else if name.eq_ignore_ascii_case("content-type") {
            // Tolerated but ignored.
            continue;
        } else {
            // Unknown header — tolerate (LSP §3 says additional
            // headers are reserved for future use; ignoring is the
            // safest option).
            continue;
        }
    }

    // Step 2: read exactly `Content-Length` bytes for the body.
    //
    // M5-02 forward-bind (PR #286 review LOW-1): a peer that
    // announces `Content-Length: 16777215` and never sends bytes
    // hangs `read_exact` indefinitely. For STDIO (parent process is
    // the trusted local MCP host) this is acceptable — the parent is
    // the rate-limiter. For M5-02 streamable-HTTP / WebSocket where
    // the peer is untrusted, this becomes a DoS vector. The M5-02
    // transport MUST add a body-read timeout (e.g., 30s default;
    // per-tenant override via M5-12 rate-limit config).
    let len = content_length
        .ok_or_else(|| MCPError::ParseError("missing Content-Length header".into()))?;
    if len > max_message_bytes {
        // Over-cap: drain the announced body (bounded scratch, EOF-tolerant)
        // so the NEXT frame parses cleanly, then surface a clear, actionable
        // error. NEVER `vec![0u8; len]` here — draining reads in fixed-size
        // chunks, so the cap's allocation-amplification defense holds while
        // the session stays usable. This replaces the prior immediate-return
        // that left the oversized body in the pipe and desynced the session
        // (the #818 "silent, no log" failure on `graph.ingest`).
        drain_body(reader, len).await;
        return Err(MCPError::ParseError(format!(
            "Content-Length {len} bytes exceeds cap {max_message_bytes} bytes for this \
             transport; reduce the batch size / chunk the ingest"
        )));
    }
    let mut body = vec![0u8; len];
    reader
        .read_exact(&mut body)
        .await
        .map_err(|e| MCPError::ParseError(format!("body read: {e}")))?;

    // Step 3: parse the JSON envelope.
    serde_json::from_slice(&body).map(Some).map_err(|e| {
        MCPError::ParseError(format!(
            "JSON envelope parse: {e} (body = {} bytes)",
            body.len()
        ))
    })
}

/// Read one stdio message, auto-detecting the per-connection framing mode on
/// the first non-empty bytes and reusing that mode afterwards.
///
/// Detection is intentionally unambiguous for the two supported modes:
/// `Content-Length` (case-insensitive) selects the legacy LSP framer; `{`
/// selects MCP newline-delimited JSON. Empty leading lines are skipped before
/// detection to preserve the legacy reader's tolerance for startup whitespace.
pub async fn read_stdio_message_with_cap<R>(
    reader: &mut BufReader<R>,
    max_message_bytes: usize,
    mode: &mut Option<StdioFramingMode>,
) -> Result<Option<Value>, MCPError>
where
    R: AsyncRead + Unpin,
{
    if mode.is_none() {
        let detected = detect_stdio_framing(reader).await?;
        *mode = match detected {
            Some(mode) => Some(mode),
            None => return Ok(None),
        };
    }
    match mode.expect("stdio framing mode detected") {
        StdioFramingMode::ContentLength => read_message_with_cap(reader, max_message_bytes).await,
        StdioFramingMode::NewlineJson => read_newline_json_message(reader, max_message_bytes).await,
    }
}

async fn detect_stdio_framing<R>(
    reader: &mut BufReader<R>,
) -> Result<Option<StdioFramingMode>, MCPError>
where
    R: AsyncRead + Unpin,
{
    loop {
        let buf = reader
            .fill_buf()
            .await
            .map_err(|e| MCPError::ParseError(format!("stdio framing detect: {e}")))?;
        if buf.is_empty() {
            return Ok(None);
        }
        let skipped = buf
            .iter()
            .take_while(|byte| matches!(byte, b'\r' | b'\n' | b' ' | b'\t'))
            .count();
        if skipped > 0 {
            reader.consume(skipped);
            continue;
        }
        if buf[0] == b'{' {
            return Ok(Some(StdioFramingMode::NewlineJson));
        }
        if starts_with_ignore_ascii_case(buf, b"Content-Length") {
            return Ok(Some(StdioFramingMode::ContentLength));
        }
        return Err(MCPError::ParseError(
            "stdio framing detect: expected Content-Length header or newline JSON object".into(),
        ));
    }
}

fn starts_with_ignore_ascii_case(buf: &[u8], prefix: &[u8]) -> bool {
    buf.len() >= prefix.len()
        && buf[..prefix.len()]
            .iter()
            .zip(prefix)
            .all(|(a, b)| a.eq_ignore_ascii_case(b))
}

async fn read_newline_json_message<R>(
    reader: &mut BufReader<R>,
    max_message_bytes: usize,
) -> Result<Option<Value>, MCPError>
where
    R: AsyncRead + Unpin,
{
    let mut line = String::new();
    loop {
        line.clear();
        let n = reader
            .read_line(&mut line)
            .await
            .map_err(|e| MCPError::ParseError(format!("newline JSON read: {e}")))?;
        if n == 0 {
            return Ok(None);
        }
        if n > max_message_bytes {
            return Err(MCPError::ParseError(format!(
                "newline JSON message {n} bytes exceeds cap {max_message_bytes} bytes for this transport"
            )));
        }
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.trim().is_empty() {
            continue;
        }
        return serde_json::from_str(trimmed)
            .map(Some)
            .map_err(|e| MCPError::ParseError(format!("newline JSON parse: {e}")));
    }
}

/// Discard exactly `n` bytes from `reader` (an over-cap frame's body),
/// reading into a bounded scratch buffer so we never allocate the
/// oversized `n` up front (that allocation is the very DoS the cap
/// defends against). Best-effort + EOF-tolerant: a peer that lies about
/// the length and never sends the bytes simply EOFs, which we treat as
/// "drained" — the caller surfaces the over-cap error regardless. Used by
/// [`read_message_with_cap`] to keep the stream in sync after rejecting an
/// oversized frame (#818) so a single bad frame doesn't desync the
/// session.
async fn drain_body<R>(reader: &mut BufReader<R>, mut n: usize)
where
    R: AsyncRead + Unpin,
{
    let mut scratch = [0u8; 64 * 1024];
    while n > 0 {
        let want = n.min(scratch.len());
        match reader.read(&mut scratch[..want]).await {
            Ok(0) | Err(_) => break, // EOF or I/O error — stop draining.
            Ok(read) => n -= read,
        }
    }
}

/// Encode `payload` as a Content-Length-framed message and write it
/// to `writer`. Flushes on completion.
///
/// Errors as [`MCPError::InternalError`] on a stdout write failure.
/// The caller is responsible for propagating these to the client (in
/// the stdio model, an unwritable stdout means the parent process is
/// gone and we should shut down).
pub async fn write_message<W>(writer: &mut W, payload: &Value) -> Result<(), MCPError>
where
    W: AsyncWrite + Unpin,
{
    let body = serde_json::to_vec(payload)
        .map_err(|e| MCPError::InternalError(format!("response serialize: {e}")))?;
    let header = format!("Content-Length: {}\r\n\r\n", body.len());
    writer
        .write_all(header.as_bytes())
        .await
        .map_err(|e| MCPError::InternalError(format!("header write: {e}")))?;
    writer
        .write_all(&body)
        .await
        .map_err(|e| MCPError::InternalError(format!("body write: {e}")))?;
    writer
        .flush()
        .await
        .map_err(|e| MCPError::InternalError(format!("flush: {e}")))?;
    Ok(())
}

/// Write `payload` using the detected stdio framing mode.
pub async fn write_stdio_message<W>(
    writer: &mut W,
    payload: &Value,
    mode: StdioFramingMode,
) -> Result<(), MCPError>
where
    W: AsyncWrite + Unpin,
{
    match mode {
        StdioFramingMode::ContentLength => write_message(writer, payload).await,
        StdioFramingMode::NewlineJson => {
            let body = serde_json::to_string(payload)
                .map_err(|e| MCPError::InternalError(format!("response serialize: {e}")))?;
            writer
                .write_all(body.as_bytes())
                .await
                .map_err(|e| MCPError::InternalError(format!("newline JSON body write: {e}")))?;
            writer.write_all(b"\n").await.map_err(|e| {
                MCPError::InternalError(format!("newline JSON delimiter write: {e}"))
            })?;
            writer
                .flush()
                .await
                .map_err(|e| MCPError::InternalError(format!("flush: {e}")))?;
            Ok(())
        }
    }
}

/// Parse a JSON [`Value`] into a [`JsonRpcRequest`], rejecting
/// malformed envelopes with [`MCPError::InvalidRequest`].
pub fn decode_request(value: Value) -> Result<JsonRpcRequest, MCPError> {
    let req: JsonRpcRequest = serde_json::from_value(value)
        .map_err(|e| MCPError::InvalidRequest(format!("envelope deserialize: {e}")))?;
    if req.jsonrpc != JSONRPC_VERSION {
        return Err(MCPError::InvalidRequest(format!(
            "unsupported jsonrpc version: {:?}",
            req.jsonrpc
        )));
    }
    Ok(req)
}

/// Echo `id` for response framing — `Some(v) → v` else `Value::Null`
/// per JSON-RPC §5 (a response to a notification is NOT sent at all,
/// but the framer caller may still need a placeholder for an internal
/// log line).
#[must_use]
pub fn id_or_null(id: Option<Value>) -> Value {
    id.unwrap_or(Value::Null)
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tokio::io::BufReader;

    fn frame(payload: &str) -> Vec<u8> {
        let mut out = format!("Content-Length: {}\r\n\r\n", payload.len()).into_bytes();
        out.extend_from_slice(payload.as_bytes());
        out
    }

    #[tokio::test]
    async fn read_message_roundtrips_a_simple_request() {
        let payload = r#"{"jsonrpc":"2.0","id":1,"method":"graph.schema","params":{}}"#;
        let bytes = frame(payload);
        let mut r = BufReader::new(&bytes[..]);
        let v = read_message(&mut r).await.expect("read ok");
        let v = v.expect("Some");
        assert_eq!(v["method"], "graph.schema");
        assert_eq!(v["id"], 1);
    }

    #[tokio::test]
    async fn read_message_returns_none_on_clean_eof() {
        let mut r = BufReader::new(&[][..]);
        let v = read_message(&mut r).await.expect("clean EOF is Ok(None)");
        assert!(v.is_none());
    }

    #[tokio::test]
    async fn read_message_rejects_oversized_content_length() {
        let header = format!("Content-Length: {}\r\n\r\n", MAX_MESSAGE_BYTES + 1);
        let mut r = BufReader::new(header.as_bytes());
        let err = read_message(&mut r)
            .await
            .expect_err("oversized must reject");
        match err {
            MCPError::ParseError(msg) => assert!(msg.contains("exceeds cap"), "msg: {msg}"),
            other => panic!("expected ParseError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn stdio_cap_admits_a_frame_the_default_cap_rejects() {
        // #818 — the stdio cap is strictly larger than the default
        // (untrusted-peer) cap, so a `graph.ingest`-sized frame between the
        // two caps parses under the stdio cap. (The cap ORDERING itself is a
        // compile-time invariant — see the `const _: () = assert!(...)` below
        // the const definitions; a future edit that inverts it won't compile.)
        // A 17 MiB body (just past the 16 MiB default cap — the ~7 000 ×
        // 128-d ingest from #818) parses under the stdio cap.
        let body_len = 17 * 1024 * 1024;
        let payload = {
            // A valid JSON-RPC envelope padded to exactly `body_len` bytes via
            // a string field (so the framer reads a real body, not a stub).
            let prefix = r#"{"jsonrpc":"2.0","id":1,"method":"graph.ingest","pad":""#;
            let suffix = r#""}"#;
            let pad = body_len - prefix.len() - suffix.len();
            format!("{prefix}{}{suffix}", "x".repeat(pad))
        };
        assert_eq!(payload.len(), body_len);
        let mut framed = format!("Content-Length: {}\r\n\r\n", payload.len()).into_bytes();
        framed.extend_from_slice(payload.as_bytes());
        let mut r = BufReader::new(&framed[..]);
        // Default cap REJECTS it (the #818 cliff)...
        {
            let mut r_default = BufReader::new(&framed[..]);
            let err = read_message_with_cap(&mut r_default, MAX_MESSAGE_BYTES)
                .await
                .expect_err("17 MiB frame must exceed the 16 MiB default cap");
            assert!(matches!(err, MCPError::ParseError(_)));
        }
        // ...the stdio cap ADMITS it.
        let v = read_message_with_cap(&mut r, STDIO_MAX_MESSAGE_BYTES)
            .await
            .expect("stdio cap admits a 17 MiB frame")
            .expect("Some");
        assert_eq!(v["method"], "graph.ingest");
    }

    #[tokio::test]
    async fn read_message_with_cap_drains_oversized_body_then_resyncs() {
        // #818 fault-injection: an over-cap frame must be DRAINED (not left
        // in the pipe), so the FOLLOWING frame parses cleanly. Before the
        // fix the over-cap branch returned on the header and left the body
        // in the stream, desyncing every subsequent read. A small injected
        // cap exercises the drain path without paying real-MiB I/O.
        // Frame 1: a >cap body that must be drained.
        let big = r#"{"jsonrpc":"2.0","id":1,"method":"oversized-frame-to-drain"}"#;
        // Frame 2: a <=cap body that must parse AFTER the drain.
        let small = r#"{"jsonrpc":"2.0","id":2}"#;
        // Cap = exactly the small frame's length: frame-2 fits, frame-1 (which
        // is longer) is over-cap and must be drained.
        let cap = small.len();
        assert!(big.len() > cap, "frame-1 body must exceed the injected cap");

        let mut stream = format!("Content-Length: {}\r\n\r\n", big.len()).into_bytes();
        stream.extend_from_slice(big.as_bytes());
        stream.extend_from_slice(format!("Content-Length: {}\r\n\r\n", small.len()).as_bytes());
        stream.extend_from_slice(small.as_bytes());
        let mut r = BufReader::new(&stream[..]);

        // Frame 1 → over-cap error (actionable message).
        let err = read_message_with_cap(&mut r, cap)
            .await
            .expect_err("frame-1 exceeds cap");
        match err {
            MCPError::ParseError(msg) => {
                assert!(
                    msg.contains("exceeds cap"),
                    "actionable over-cap msg: {msg}"
                );
                assert!(
                    msg.contains("chunk the ingest"),
                    "message should guide the caller to chunk: {msg}"
                );
            }
            other => panic!("expected ParseError, got {other:?}"),
        }
        // Frame 2 → parses cleanly: PROVES frame-1's body was drained and
        // the stream re-synced (the load-bearing oracle).
        let v = read_message_with_cap(&mut r, cap)
            .await
            .expect("post-drain read ok")
            .expect("frame-2 is Some after drain");
        assert_eq!(v["id"], 2, "the frame AFTER the over-cap one must parse");
    }

    #[tokio::test]
    async fn read_message_tolerates_content_type_header() {
        let payload = r#"{"jsonrpc":"2.0","id":42,"method":"graph.inspect"}"#;
        let body = format!(
            "Content-Type: application/vscode-jsonrpc; charset=utf-8\r\n\
             Content-Length: {}\r\n\r\n{}",
            payload.len(),
            payload
        );
        let mut r = BufReader::new(body.as_bytes());
        let v = read_message(&mut r).await.expect("ok").expect("some");
        assert_eq!(v["id"], 42);
    }

    #[tokio::test]
    async fn read_message_rejects_missing_content_length() {
        // A header block with no Content-Length, followed by an empty
        // line, must error — we cannot frame a body without a length.
        let body = b"X-Other: value\r\n\r\n{\"jsonrpc\":\"2.0\"}";
        let mut r = BufReader::new(&body[..]);
        let err = read_message(&mut r).await.expect_err("must reject");
        match err {
            MCPError::ParseError(_) => {}
            other => panic!("expected ParseError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn write_message_emits_canonical_framing() {
        let payload = json!({"jsonrpc":"2.0","id":1,"result":{"ok":true}});
        let mut buf = Vec::new();
        write_message(&mut buf, &payload).await.expect("ok");
        let s = String::from_utf8(buf).expect("UTF-8");
        assert!(s.starts_with("Content-Length: "), "got: {s:?}");
        assert!(s.contains("\r\n\r\n"), "framing separator missing");
        assert!(s.contains("\"id\":1"), "id missing");
    }

    #[tokio::test]
    async fn read_then_write_roundtrip_preserves_envelope() {
        // End-to-end: write a response, read it back, verify the
        // envelope is preserved.
        let resp = JsonRpcResponse::success(json!(7), json!({"labels": ["Person"]}));
        let resp_value = serde_json::to_value(&resp).unwrap();
        let mut buf = Vec::new();
        write_message(&mut buf, &resp_value).await.unwrap();
        let mut r = BufReader::new(&buf[..]);
        let parsed = read_message(&mut r).await.unwrap().unwrap();
        assert_eq!(parsed, resp_value);
    }

    #[test]
    fn decode_request_rejects_wrong_version() {
        let v = json!({"jsonrpc":"1.0","id":1,"method":"graph.schema"});
        let err = decode_request(v).expect_err("must reject");
        match err {
            MCPError::InvalidRequest(msg) => {
                assert!(msg.contains("1.0") || msg.contains("unsupported"))
            }
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    #[test]
    fn decode_request_accepts_missing_params() {
        let v = json!({"jsonrpc":"2.0","id":1,"method":"graph.schema"});
        let req = decode_request(v).expect("ok");
        assert_eq!(req.method, "graph.schema");
        assert_eq!(req.params, Value::Null);
    }

    #[test]
    fn from_mcp_renders_error_envelope() {
        let mcp_err = MCPError::Cancelled;
        let env = JsonRpcErrorResponse::from_mcp(json!(1), &mcp_err);
        assert_eq!(env.error.code, -32001);
        assert_eq!(env.error.message, "request cancelled");
        assert!(env.error.data.is_none());
    }

    #[test]
    fn from_mcp_renders_query_error_with_data() {
        let mcp_err = MCPError::QueryError("unknown label `Person`".into());
        let env = JsonRpcErrorResponse::from_mcp(json!("req-7"), &mcp_err);
        assert_eq!(env.error.code, -32005);
        let data = env.error.data.expect("query-error carries data");
        assert!(data.as_str().unwrap().contains("Person"));
    }
}
