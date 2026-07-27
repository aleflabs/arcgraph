//! W26-γ-3 / ADR-136 — JSON-RPC malformed-frame negative tests.
//!
//! # Surface
//!
//! [`arcgraph_mcp::jsonrpc::read_message`] + [`decode_request`] +
//! [`MAX_MESSAGE_BYTES`]. The MCP stdio/HTTP framer reads
//! Content-Length-framed JSON-RPC 2.0 envelopes per LSP §3 +
//! JSON-RPC 2.0 §4-5.
//!
//! # Adversarial classes covered
//!
//! 1. **Content-Length overflow** — Content-Length value >
//!    [`MAX_MESSAGE_BYTES`] (16 MiB) rejects.
//! 2. **Missing Content-Length** — body present but no header.
//! 3. **Malformed Content-Length** — non-numeric value, negative,
//!    leading/trailing garbage.
//! 4. **Truncated body** — header says N bytes, body delivers < N.
//! 5. **Malformed JSON body** — header OK but body is not valid JSON.
//! 6. **JSON-RPC version drift** — `jsonrpc` field != "2.0".
//! 7. **Missing method** — request envelope without `method`.
//! 8. **Notification (no id)** — valid per spec but dispatched as
//!    fire-and-forget.
//! 9. **Embedded NULs** — null bytes in the body (NotAsciiPrintable
//!    class).
//! 10. **Header injection** — `Content-Length: 100\r\nX-Evil: bad`
//!     — extra headers tolerated but no smuggle.
//! 11. **CRLF vs LF** — both line endings tolerated per LSP §3.
//!
//! Under the testing strategy fuzz target dependency: the
//! `fuzz/fuzz_targets/mcp_message_fuzz.rs` target covers the
//! arbitrary-bytes surface; this test suite pins the specific
//! adversarial patterns.

use arcgraph_mcp::error::MCPError;
use arcgraph_mcp::jsonrpc::{JSONRPC_VERSION, MAX_MESSAGE_BYTES, decode_request, read_message};
use serde_json::{Value, json};
use tokio::io::BufReader;

/// Helper — build a Content-Length-framed payload.
fn frame(payload: &str) -> Vec<u8> {
    let mut out = format!("Content-Length: {}\r\n\r\n", payload.len()).into_bytes();
    out.extend_from_slice(payload.as_bytes());
    out
}

// =====================================================================
// 1. Content-Length overflow
// =====================================================================

#[tokio::test]
async fn content_length_above_cap_rejects() {
    let header = format!("Content-Length: {}\r\n\r\n", MAX_MESSAGE_BYTES + 1);
    let mut r = BufReader::new(header.as_bytes());
    let err = read_message(&mut r).await.unwrap_err();
    assert!(matches!(err, MCPError::ParseError(_)));
}

#[tokio::test]
async fn content_length_at_cap_does_not_overflow_parse() {
    // We don't read a 16-MiB body, but the header itself + 1-byte body
    // would be the boundary. Use a small body — the framer accepts
    // mismatched bodies if their actual bytes are present.
    // To test the cap boundary without paying 16 MiB I/O, set the
    // header to MAX_MESSAGE_BYTES exactly + give a 1-byte body, then
    // verify the header parse goes through (the body read would
    // block / fail downstream because there's no full body, but the
    // parse phase succeeds).
    let header = format!("Content-Length: {}\r\n\r\n", MAX_MESSAGE_BYTES);
    let mut r = BufReader::new(header.as_bytes());
    // EOF after header → read_message returns ParseError for partial
    // body (not for header overflow); the boundary check is OK.
    let err = read_message(&mut r).await.unwrap_err();
    match err {
        MCPError::ParseError(s) => {
            // Acceptable: any ParseError class. The important thing
            // is it's NOT the "exceeds cap" error.
            assert!(
                !s.contains("exceeds cap"),
                "Content-Length at exact cap must not surface 'exceeds cap': {s}"
            );
        }
        other => panic!("expected ParseError, got {other:?}"),
    }
}

// =====================================================================
// 2. Missing Content-Length
// =====================================================================

#[tokio::test]
async fn missing_content_length_rejects() {
    // Some other header but no Content-Length.
    let bytes = b"X-Foo: bar\r\n\r\n{}".to_vec();
    let mut r = BufReader::new(&bytes[..]);
    let err = read_message(&mut r).await.unwrap_err();
    assert!(matches!(err, MCPError::ParseError(_)));
}

// =====================================================================
// 3. Malformed Content-Length
// =====================================================================

#[tokio::test]
async fn nonnumeric_content_length_rejects() {
    let bytes = b"Content-Length: abc\r\n\r\n{}".to_vec();
    let mut r = BufReader::new(&bytes[..]);
    let err = read_message(&mut r).await.unwrap_err();
    assert!(matches!(err, MCPError::ParseError(_)));
}

#[tokio::test]
async fn negative_content_length_rejects() {
    let bytes = b"Content-Length: -10\r\n\r\n{}".to_vec();
    let mut r = BufReader::new(&bytes[..]);
    let err = read_message(&mut r).await.unwrap_err();
    assert!(matches!(err, MCPError::ParseError(_)));
}

#[tokio::test]
async fn fractional_content_length_rejects() {
    let bytes = b"Content-Length: 12.5\r\n\r\n{}".to_vec();
    let mut r = BufReader::new(&bytes[..]);
    let err = read_message(&mut r).await.unwrap_err();
    assert!(matches!(err, MCPError::ParseError(_)));
}

#[tokio::test]
async fn malformed_header_line_rejects() {
    // Header line without a `:` separator.
    let bytes = b"Content-Length 100\r\n\r\n{}".to_vec();
    let mut r = BufReader::new(&bytes[..]);
    let err = read_message(&mut r).await.unwrap_err();
    assert!(matches!(err, MCPError::ParseError(_)));
}

// =====================================================================
// 4. Truncated body
// =====================================================================

#[tokio::test]
async fn body_shorter_than_content_length_rejects() {
    // Header says 50 bytes; provide only 10.
    let bytes = b"Content-Length: 50\r\n\r\n{\"a\":\"bc\"}".to_vec();
    let mut r = BufReader::new(&bytes[..]);
    let err = read_message(&mut r).await.unwrap_err();
    assert!(matches!(err, MCPError::ParseError(_)));
}

// =====================================================================
// 5. Malformed JSON body
// =====================================================================

#[tokio::test]
async fn malformed_json_body_rejects() {
    let bytes = frame("{not-json}");
    let mut r = BufReader::new(&bytes[..]);
    let err = read_message(&mut r).await.unwrap_err();
    assert!(matches!(err, MCPError::ParseError(_)));
}

#[tokio::test]
async fn empty_body_rejects() {
    let bytes = b"Content-Length: 0\r\n\r\n".to_vec();
    let mut r = BufReader::new(&bytes[..]);
    // Zero-length body is OK at the framing layer; the JSON
    // deserialize of "" fails.
    let result = read_message(&mut r).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn json_array_at_top_level_parses_but_decode_rejects() {
    // The framer accepts any well-formed JSON value; the JSON-RPC
    // envelope decoder rejects non-object top-levels.
    let bytes = frame("[1, 2, 3]");
    let mut r = BufReader::new(&bytes[..]);
    let v = read_message(&mut r).await.unwrap().expect("frame ok");
    let err = decode_request(v).unwrap_err();
    assert!(matches!(err, MCPError::InvalidRequest(_)));
}

// =====================================================================
// 6. JSON-RPC version drift
// =====================================================================

#[test]
fn version_drift_rejects() {
    // Valid JSON, valid shape, but jsonrpc != "2.0".
    let env = json!({
        "jsonrpc": "1.0",
        "id": 1,
        "method": "graph.schema",
        "params": {}
    });
    let err = decode_request(env).unwrap_err();
    assert!(matches!(err, MCPError::InvalidRequest(_)));
}

#[test]
fn missing_jsonrpc_field_rejects() {
    let env = json!({
        "id": 1,
        "method": "graph.schema",
        "params": {}
    });
    let err = decode_request(env).unwrap_err();
    assert!(matches!(err, MCPError::InvalidRequest(_)));
}

// =====================================================================
// 7. Missing method
// =====================================================================

#[test]
fn missing_method_rejects() {
    let env = json!({
        "jsonrpc": JSONRPC_VERSION,
        "id": 1,
        "params": {}
    });
    let err = decode_request(env).unwrap_err();
    assert!(matches!(err, MCPError::InvalidRequest(_)));
}

#[test]
fn null_method_rejects() {
    let env = json!({
        "jsonrpc": JSONRPC_VERSION,
        "id": 1,
        "method": Value::Null,
        "params": {}
    });
    let err = decode_request(env).unwrap_err();
    assert!(matches!(err, MCPError::InvalidRequest(_)));
}

// =====================================================================
// 8. Notification (no id)
// =====================================================================

#[test]
fn notification_no_id_accepts() {
    let env = json!({
        "jsonrpc": JSONRPC_VERSION,
        "method": "graph.schema",
        "params": {}
    });
    let req = decode_request(env).expect("notification is valid JSON-RPC");
    assert!(req.id.is_none());
    assert_eq!(req.method, "graph.schema");
}

#[test]
fn null_id_accepts_as_notification_equivalent() {
    let env = json!({
        "jsonrpc": JSONRPC_VERSION,
        "id": Value::Null,
        "method": "graph.schema",
        "params": {}
    });
    let req = decode_request(env).expect("null-id is valid");
    // Per `Option<Value>` deserialization: a JSON `null` for an
    // `Option<Value>` field collapses to `None` (not `Some(Null)`).
    // Either shape is acceptable at the JSON-RPC envelope layer per
    // spec §5 (null id → notification-equivalent); the dispatcher
    // uses `id_or_null` to flatten back to Value::Null on response.
    assert!(req.id.is_none() || req.id.as_ref().map(|v| v.is_null()).unwrap_or(false));
}

// =====================================================================
// 9. Embedded NULs in body
// =====================================================================

#[tokio::test]
async fn null_byte_in_body_is_handled() {
    // A NUL byte inside a JSON string literal is invalid; serde_json
    // rejects it.
    let payload = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"a\u{0000}b\",\"params\":{}}";
    let bytes = frame(payload);
    let mut r = BufReader::new(&bytes[..]);
    // The framer reads the bytes successfully; the JSON parse will
    // either accept the NUL inside the string (per RFC 8259 §7) or
    // reject. Either is acceptable; the key is no panic.
    let _ = read_message(&mut r).await;
}

// =====================================================================
// 10. Header injection
// =====================================================================

#[tokio::test]
async fn extra_headers_tolerated() {
    // The framer admits any number of additional headers as long as
    // Content-Length is one of them.
    let body = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"graph.schema\",\"params\":{}}";
    let bytes = format!(
        "X-Foo: bar\r\nContent-Length: {}\r\nX-Baz: qux\r\nContent-Type: application/json\r\n\r\n{body}",
        body.len()
    );
    let mut r = BufReader::new(bytes.as_bytes());
    let v = read_message(&mut r).await.expect("read ok").expect("some");
    assert_eq!(v["method"], "graph.schema");
}

#[tokio::test]
async fn duplicate_content_length_takes_last() {
    // Some peers emit duplicate Content-Length; the framer treats the
    // last seen as authoritative (or returns ParseError). Either is
    // acceptable; the key is no smuggle / no body-bypass.
    let body = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"graph.schema\",\"params\":{}}";
    let bytes = format!(
        "Content-Length: 999999\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    let mut r = BufReader::new(bytes.as_bytes());
    let result = read_message(&mut r).await;
    // Either accept-with-last-wins OR ParseError is acceptable.
    match result {
        Ok(Some(v)) => assert_eq!(v["method"], "graph.schema"),
        Ok(None) => panic!("unexpected clean EOF"),
        Err(MCPError::ParseError(_)) => (), // acceptable
        Err(other) => panic!("unexpected error: {other:?}"),
    }
}

// =====================================================================
// 11. CRLF vs LF
// =====================================================================

#[tokio::test]
async fn lf_only_line_endings_tolerated() {
    let body = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"graph.schema\",\"params\":{}}";
    // LF-only (no CR).
    let bytes = format!("Content-Length: {}\n\n{body}", body.len());
    let mut r = BufReader::new(bytes.as_bytes());
    let v = read_message(&mut r).await.expect("read ok").expect("some");
    assert_eq!(v["method"], "graph.schema");
}

// =====================================================================
// 12. Stress — many small frames in sequence
// =====================================================================

#[tokio::test]
async fn many_frames_back_to_back() {
    let body = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"graph.schema\",\"params\":{}}";
    let one_frame = format!("Content-Length: {}\r\n\r\n{body}", body.len());
    let mut concat = Vec::new();
    for _ in 0..10 {
        concat.extend_from_slice(one_frame.as_bytes());
    }
    let mut r = BufReader::new(&concat[..]);
    for i in 0..10 {
        let v = read_message(&mut r)
            .await
            .expect("read ok")
            .unwrap_or_else(|| panic!("frame {i}: unexpected EOF"));
        assert_eq!(v["method"], "graph.schema");
    }
    // 11th read returns clean EOF.
    let none = read_message(&mut r).await.expect("clean EOF");
    assert!(none.is_none());
}
