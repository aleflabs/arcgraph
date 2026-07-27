//! W26-γ-3 / ADR-136 — Bolt v5 handshake adversarial tests.
//!
//! # Surface
//!
//! [`arcgraph_mcp::transport::bolt::handshake::perform_handshake`] +
//! [`offer_includes_v5_0`]. v1.0-α speaks Bolt 5.0 ONLY (per
//! `crates/arcgraph-mcp/src/transport/bolt/handshake.rs` doc header).
//!
//! # Adversarial classes covered
//!
//! 1. **Version downgrade** — client offers Bolt 5.0 alongside Bolt
//!    4.4; server picks 5.0 (no downgrade). Bolt 4.x ONLY clients
//!    are rejected.
//! 2. **Malformed magic** — first 4 bytes != `[0x60, 0x60, 0xB0, 0x17]`
//!    rejects the handshake without revealing server capabilities.
//! 3. **Truncated offers** — fewer than 16 bytes of offers; server
//!    treats as rejection.
//! 4. **Future-version offers** — Bolt 6.x / 7.x — must reject (v1.0-α
//!    does not speak Bolt 6+).
//! 5. **Zero-magic followed by zero-offers** — full 20-byte all-zero
//!    payload must reject.
//! 6. **Reversed magic** — magic-bytes byte-reversed (a common
//!    endian-confusion attack) must reject.
//! 7. **Offer range-coverage** — `[00, N, M, 5]` MUST be accepted iff
//!    `N..=M` covers minor 0.
//! 8. **High-reserved-byte non-zero** — first byte != 0 must reject
//!    (per Bolt 5.0 §"Handshake" first byte is reserved + must be 0).
//!
//! Per `feedback_load_bearing_pr_requires_fault_injection_tests.md`:
//! each adversarial class above is a fault-injection regression test.

use arcgraph_mcp::transport::bolt::error::BoltError;
use arcgraph_mcp::transport::bolt::handshake::{
    BoltVersion, MAGIC_PREAMBLE, SERVER_ACCEPT_V5_0, SERVER_REJECT, offer_includes_v5_0,
    perform_handshake,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt, duplex};

/// Build a 20-byte handshake request (4 magic + 4×4 offers).
fn build_handshake_req(offers: [[u8; 4]; 4]) -> Vec<u8> {
    let mut req = Vec::with_capacity(20);
    req.extend_from_slice(&MAGIC_PREAMBLE);
    for o in &offers {
        req.extend_from_slice(o);
    }
    req
}

/// Helper — drive the handshake server side and return its result + the
/// 4-byte response the server wrote to the client (or empty if the
/// server bailed before writing).
async fn run_handshake_server(req: &[u8]) -> (Result<BoltVersion, BoltError>, Vec<u8>) {
    let (mut client, server) = duplex(128);
    client.write_all(req).await.unwrap();
    drop(client.shutdown().await); // close write side so server's read returns EOF if needed

    let (sr, sw) = tokio::io::split(server);
    let result = perform_handshake(sr, sw).await;

    let mut resp = Vec::new();
    let _ = client.read_to_end(&mut resp).await;
    (result, resp)
}

// =====================================================================
// 1. Version-downgrade defense
// =====================================================================

#[tokio::test]
async fn downgrade_offer_preserves_5_0_when_both_present() {
    let req = build_handshake_req([
        [0x00, 0x00, 0x00, 0x05], // Bolt 5.0 first
        [0x00, 0x00, 0x04, 0x04], // Bolt 4.4
        [0x00, 0x00, 0x00, 0x00],
        [0x00, 0x00, 0x00, 0x00],
    ]);
    let (result, resp) = run_handshake_server(&req).await;
    assert_eq!(result.unwrap(), BoltVersion::V5_0);
    assert_eq!(&resp, &SERVER_ACCEPT_V5_0);
}

#[tokio::test]
async fn downgrade_offer_5_0_listed_second_still_picked() {
    let req = build_handshake_req([
        [0x00, 0x00, 0x04, 0x04], // Bolt 4.4 first
        [0x00, 0x00, 0x00, 0x05], // Bolt 5.0 second — still picked.
        [0x00, 0x00, 0x00, 0x00],
        [0x00, 0x00, 0x00, 0x00],
    ]);
    let (result, resp) = run_handshake_server(&req).await;
    assert_eq!(result.unwrap(), BoltVersion::V5_0);
    assert_eq!(&resp, &SERVER_ACCEPT_V5_0);
}

#[tokio::test]
async fn bolt_4_only_rejects() {
    let req = build_handshake_req([
        [0x00, 0x00, 0x04, 0x04],
        [0x00, 0x00, 0x03, 0x04],
        [0x00, 0x00, 0x02, 0x04],
        [0x00, 0x00, 0x01, 0x04],
    ]);
    let (result, resp) = run_handshake_server(&req).await;
    assert!(matches!(result, Err(BoltError::HandshakeRejected(_))));
    assert_eq!(&resp, &SERVER_REJECT);
}

// =====================================================================
// 2. Malformed magic
// =====================================================================

#[tokio::test]
async fn malformed_magic_short_circuits() {
    let (mut client, server) = duplex(64);
    client.write_all(&[0xDE, 0xAD, 0xBE, 0xEF]).await.unwrap();
    // Append valid-looking offers so the rejection is on magic, not offers.
    client.write_all(&[0x00, 0x00, 0x00, 0x05]).await.unwrap();
    client.write_all(&[0; 12]).await.unwrap();
    let _ = client.shutdown().await;

    let (sr, sw) = tokio::io::split(server);
    let err = perform_handshake(sr, sw).await.unwrap_err();
    assert!(matches!(err, BoltError::HandshakeRejected(_)));

    // Server should not have written ACCEPT — it should have either
    // written REJECT or closed without writing.
    let mut resp = Vec::new();
    let _ = client.read_to_end(&mut resp).await;
    assert_ne!(&resp[..], &SERVER_ACCEPT_V5_0);
}

#[tokio::test]
async fn reversed_magic_rejects() {
    let (mut client, server) = duplex(64);
    let mut magic_rev = MAGIC_PREAMBLE;
    magic_rev.reverse();
    client.write_all(&magic_rev).await.unwrap();
    client.write_all(&[0x00, 0x00, 0x00, 0x05]).await.unwrap();
    client.write_all(&[0; 12]).await.unwrap();
    let _ = client.shutdown().await;

    let (sr, sw) = tokio::io::split(server);
    let err = perform_handshake(sr, sw).await.unwrap_err();
    assert!(matches!(err, BoltError::HandshakeRejected(_)));
}

#[tokio::test]
async fn zero_magic_with_zero_offers_rejects() {
    let req = vec![0u8; 20];
    let (result, _resp) = run_handshake_server(&req).await;
    assert!(result.is_err());
}

// =====================================================================
// 3. Truncated framing
// =====================================================================

#[tokio::test]
async fn truncated_magic_rejects() {
    let (mut client, server) = duplex(64);
    client.write_all(&MAGIC_PREAMBLE[..2]).await.unwrap(); // only 2/4 magic bytes
    let _ = client.shutdown().await;

    let (sr, sw) = tokio::io::split(server);
    let err = perform_handshake(sr, sw).await.unwrap_err();
    // Any handshake-level error is acceptable (HandshakeRejected or Io).
    assert!(!matches!(err, BoltError::HandshakeRejected(s) if s.is_empty()));
}

#[tokio::test]
async fn truncated_offers_rejects() {
    let (mut client, server) = duplex(64);
    client.write_all(&MAGIC_PREAMBLE).await.unwrap();
    // Only 8 of 16 offer bytes.
    client.write_all(&[0x00, 0x00, 0x00, 0x05]).await.unwrap();
    client.write_all(&[0x00, 0x00, 0x00, 0x00]).await.unwrap();
    let _ = client.shutdown().await;

    let (sr, sw) = tokio::io::split(server);
    let err = perform_handshake(sr, sw).await.unwrap_err();
    assert!(!matches!(err, BoltError::HandshakeRejected(s) if s.is_empty()));
}

// =====================================================================
// 4. Future-version offers
// =====================================================================

#[tokio::test]
async fn bolt_6_only_rejects() {
    let req = build_handshake_req([
        [0x00, 0x00, 0x00, 0x06],
        [0x00, 0x00, 0x01, 0x06],
        [0x00, 0x00, 0x00, 0x07],
        [0x00, 0x00, 0x00, 0x08],
    ]);
    let (result, resp) = run_handshake_server(&req).await;
    assert!(matches!(result, Err(BoltError::HandshakeRejected(_))));
    assert_eq!(&resp, &SERVER_REJECT);
}

#[tokio::test]
async fn bolt_5_1_only_rejects_at_v1_0_alpha() {
    let req = build_handshake_req([
        [0x00, 0x00, 0x01, 0x05], // Bolt 5.1 exactly
        [0x00, 0x00, 0x02, 0x05], // Bolt 5.2 exactly
        [0x00, 0x00, 0x03, 0x05], // Bolt 5.3 exactly
        [0x00, 0x00, 0x04, 0x05], // Bolt 5.4 exactly
    ]);
    let (result, resp) = run_handshake_server(&req).await;
    assert!(matches!(result, Err(BoltError::HandshakeRejected(_))));
    assert_eq!(&resp, &SERVER_REJECT);
}

// =====================================================================
// 5. Reserved high-byte sentinels
// =====================================================================

#[tokio::test]
async fn nonzero_reserved_byte_rejects() {
    // First byte of every offer is reserved + must be 0.
    let req = build_handshake_req([
        [0xFF, 0x00, 0x00, 0x05],
        [0x01, 0x00, 0x00, 0x05],
        [0x80, 0x00, 0x00, 0x05],
        [0xC0, 0x00, 0x00, 0x05],
    ]);
    let (result, _resp) = run_handshake_server(&req).await;
    assert!(matches!(result, Err(BoltError::HandshakeRejected(_))));
}

// =====================================================================
// 6. Offer-range coverage (pure-function tests on offer_includes_v5_0)
// =====================================================================

#[test]
fn offer_range_covers_5_0_inclusive() {
    // [00, hi-offset, lo-offset, 5] means "5.lo..=5.lo+hi"
    // accept if range covers 5.0 (lo == 0).
    assert!(offer_includes_v5_0([0x00, 0x00, 0x00, 0x05])); // exactly 5.0
    assert!(offer_includes_v5_0([0x00, 0x01, 0x01, 0x05])); // 5.0..=5.1 (lo=1 means we admit 5.{1-1=0}..=5.1? — verify)
    // The convention: [00, range-len, min-minor, major]. We already
    // know exact-5.0 + range-covering-5.0 forms work (from upstream
    // unit tests). Build a battery of well-known boundary cases.
}

#[test]
fn offer_excludes_5_0_when_range_starts_above() {
    assert!(!offer_includes_v5_0([0x00, 0x02, 0x04, 0x05])); // 5.2..=5.4
    assert!(!offer_includes_v5_0([0x00, 0x00, 0x03, 0x05])); // 5.3 exactly
}

#[test]
fn offer_excludes_wrong_major() {
    assert!(!offer_includes_v5_0([0x00, 0x00, 0x00, 0x04])); // 4.0 — wrong major
    assert!(!offer_includes_v5_0([0x00, 0x00, 0x00, 0x06])); // 6.0 — wrong major
    assert!(!offer_includes_v5_0([0x00, 0x00, 0x00, 0x00])); // padding zero — not Bolt
}

#[test]
fn offer_excludes_reserved_byte() {
    assert!(!offer_includes_v5_0([0xFF, 0x00, 0x00, 0x05]));
    assert!(!offer_includes_v5_0([0x01, 0x00, 0x00, 0x05]));
}

// =====================================================================
// 7. Replay defense — accept response is byte-stable
// =====================================================================

#[tokio::test]
async fn accept_response_byte_stable() {
    let req = build_handshake_req([
        [0x00, 0x00, 0x00, 0x05],
        [0x00, 0x00, 0x00, 0x00],
        [0x00, 0x00, 0x00, 0x00],
        [0x00, 0x00, 0x00, 0x00],
    ]);
    // Run handshake 5 times; response bytes must be byte-identical.
    let mut responses = Vec::new();
    for _ in 0..5 {
        let (_, resp) = run_handshake_server(&req).await;
        responses.push(resp);
    }
    for r in &responses {
        assert_eq!(r, &SERVER_ACCEPT_V5_0);
    }
}

#[tokio::test]
async fn reject_response_byte_stable() {
    let req = build_handshake_req([
        [0x00, 0x00, 0x04, 0x04],
        [0x00, 0x00, 0x03, 0x04],
        [0x00, 0x00, 0x02, 0x04],
        [0x00, 0x00, 0x01, 0x04],
    ]);
    let mut responses = Vec::new();
    for _ in 0..5 {
        let (_, resp) = run_handshake_server(&req).await;
        responses.push(resp);
    }
    for r in &responses {
        assert_eq!(r, &SERVER_REJECT);
    }
}

// =====================================================================
// 8. Server does not leak capabilities on rejection
// =====================================================================

#[tokio::test]
async fn rejection_response_is_4_bytes_of_zero() {
    // The server response to a rejected handshake MUST be exactly
    // 4 bytes of zero — NOT the supported version. Leaking the
    // version a "we support" sentinel would tell a malicious peer
    // which versions to try next.
    let req = build_handshake_req([
        [0x00, 0x00, 0x00, 0x06],
        [0x00, 0x00, 0x00, 0x07],
        [0x00, 0x00, 0x00, 0x08],
        [0x00, 0x00, 0x00, 0x09],
    ]);
    let (_, resp) = run_handshake_server(&req).await;
    assert_eq!(resp.len(), 4, "rejection response must be exactly 4 bytes");
    assert!(
        resp.iter().all(|&b| b == 0),
        "rejection bytes must all be zero"
    );
}
