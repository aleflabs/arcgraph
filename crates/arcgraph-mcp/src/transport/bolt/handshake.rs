//! W14δ M5-13 — Bolt protocol HANDSHAKE.
//!
//! Per the Bolt 5.0 spec §"Handshake" the client opens a TCP
//! connection and immediately sends 20 bytes:
//!
//! - 4 bytes: magic preamble `0x60 0x60 0xB0 0x17`.
//! - 16 bytes: four 4-byte version offers in client-preference order.
//!   Each version is `0x00 0x00 minor major` (yes — minor BEFORE
//!   major in the wire layout per the Bolt 5+ "version range"
//!   convention; Bolt 4 sent `00 00 00 04`, Bolt 5+ sends
//!   `00 0F 04 05` to mean "5.0 through 5.4").
//!
//! The server replies with 4 bytes:
//!
//! - `0x00 0x00 minor major` for the chosen version, OR
//! - `0x00 0x00 0x00 0x00` to reject all offered versions (the client
//!   then closes the TCP connection).
//!
//! v1.0-α only speaks **Bolt 5.0** (per the spawn prompt's hard
//! boundary "Bolt 5.0 only at v1.0-α; no 4.4 / 5.1 / 5.2"). We
//! accept any client offer that includes `5.0` exactly and reject
//! everything else with the zero response.

use tokio::io::{AsyncReadExt, AsyncWriteExt};

use super::error::BoltError;

/// Bolt handshake magic preamble. Sent verbatim by the client at the
/// very start of every connection.
pub const MAGIC_PREAMBLE: [u8; 4] = [0x60, 0x60, 0xB0, 0x17];

/// Wire encoding of the Bolt 5.0 server response when accepting v5.0.
/// Layout: `[00, 00, 00, 05]` — first two bytes reserved, then minor
/// (`0x00`), then major (`0x05`).
pub const SERVER_ACCEPT_V5_0: [u8; 4] = [0x00, 0x00, 0x00, 0x05];

/// Wire encoding of the Bolt server "reject all" response.
pub const SERVER_REJECT: [u8; 4] = [0x00, 0x00, 0x00, 0x00];

/// Major / minor pair the server accepts at v1.0-α.
const SUPPORTED_MAJOR: u8 = 5;
const SUPPORTED_MINOR: u8 = 0;

/// Negotiated Bolt protocol version returned to the caller after a
/// successful handshake. v1.0-α only ever returns `5.0`; the type
/// stays open to accommodate future Bolt 5.x slices without a wire-
/// API break.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoltVersion {
    /// Major version (5 at v1.0-α).
    pub major: u8,
    /// Minor version (0 at v1.0-α).
    pub minor: u8,
}

impl BoltVersion {
    /// The single supported v1.0-α version.
    pub const V5_0: BoltVersion = BoltVersion {
        major: SUPPORTED_MAJOR,
        minor: SUPPORTED_MINOR,
    };
}

/// Decide whether `offer` (a 4-byte version block) lists Bolt 5.0
/// among its negotiated range. Bolt 5+ encodes a version as
/// `[00, range, minor, major]` where `range` is the count of older
/// minor versions the client also accepts. So `[00, 04, 04, 05]`
/// means "5.0..=5.4 acceptable". A range of `0` means "exact minor".
pub fn offer_includes_v5_0(offer: [u8; 4]) -> bool {
    if offer[0] != 0 {
        return false;
    }
    let range = offer[1];
    let minor = offer[2];
    let major = offer[3];
    if major != SUPPORTED_MAJOR {
        return false;
    }
    let lower = minor.saturating_sub(range);
    (lower..=minor).contains(&SUPPORTED_MINOR)
}

/// Run the Bolt handshake on an already-accepted TCP connection.
///
/// Reads 4 bytes for magic + 16 bytes for the four version offers.
/// On success, writes the 4-byte server-accept response and returns
/// the negotiated [`BoltVersion`]. On magic-mismatch / unsupported
/// offers / I/O fault, the function writes the zero-reject response
/// (where applicable) and returns [`BoltError::HandshakeRejected`].
///
/// # Caller contract
///
/// - The reader/writer must be the actual TCP socket (or a test pipe).
///   Buffering above the handshake level is the caller's job.
/// - On Err, the caller closes the socket — Bolt has no
///   "handshake retry" semantics within a connection.
pub async fn perform_handshake<R, W>(mut reader: R, mut writer: W) -> Result<BoltVersion, BoltError>
where
    R: tokio::io::AsyncRead + Unpin,
    W: tokio::io::AsyncWrite + Unpin,
{
    // 1. Read magic preamble.
    let mut magic = [0u8; 4];
    reader
        .read_exact(&mut magic)
        .await
        .map_err(|e| BoltError::Io(format!("handshake magic read: {e}")))?;
    if magic != MAGIC_PREAMBLE {
        // Per the spec, an invalid magic means "this is not a Bolt
        // client". Close without writing anything — the peer's
        // expectation is that the connection drops on a non-Bolt
        // greeting.
        return Err(BoltError::HandshakeRejected(format!(
            "invalid magic preamble: {magic:?}"
        )));
    }
    // 2. Read 16 bytes for four version offers.
    let mut offers = [0u8; 16];
    reader
        .read_exact(&mut offers)
        .await
        .map_err(|e| BoltError::Io(format!("handshake offers read: {e}")))?;
    // 3. Pick the first offer that includes Bolt 5.0.
    let mut chosen: Option<BoltVersion> = None;
    for chunk_idx in 0..4 {
        let off = chunk_idx * 4;
        let chunk = [
            offers[off],
            offers[off + 1],
            offers[off + 2],
            offers[off + 3],
        ];
        if chunk == [0; 4] {
            // Padding — client signalled "no more offers".
            continue;
        }
        if offer_includes_v5_0(chunk) {
            chosen = Some(BoltVersion::V5_0);
            break;
        }
    }
    // 4. Write response.
    let response = match chosen {
        Some(_) => SERVER_ACCEPT_V5_0,
        None => SERVER_REJECT,
    };
    writer
        .write_all(&response)
        .await
        .map_err(|e| BoltError::Io(format!("handshake response write: {e}")))?;
    writer
        .flush()
        .await
        .map_err(|e| BoltError::Io(format!("handshake response flush: {e}")))?;
    chosen.ok_or_else(|| BoltError::HandshakeRejected("no offer included Bolt 5.0".to_string()))
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::duplex;

    #[test]
    fn offer_includes_v5_0_recognizes_exact_minor() {
        // [00, 00, 00, 05] = "Bolt 5.0 exactly".
        assert!(offer_includes_v5_0([0x00, 0x00, 0x00, 0x05]));
    }

    #[test]
    fn offer_includes_v5_0_recognizes_range_covering_5_0() {
        // [00, 04, 04, 05] = "Bolt 5.0..=5.4 acceptable".
        assert!(offer_includes_v5_0([0x00, 0x04, 0x04, 0x05]));
        // [00, 03, 03, 05] = "Bolt 5.0..=5.3 acceptable".
        assert!(offer_includes_v5_0([0x00, 0x03, 0x03, 0x05]));
        // [00, 02, 02, 05] = "Bolt 5.0..=5.2 acceptable".
        assert!(offer_includes_v5_0([0x00, 0x02, 0x02, 0x05]));
    }

    #[test]
    fn offer_excludes_v5_0_when_range_does_not_cover() {
        // [00, 02, 04, 05] = "Bolt 5.2..=5.4" — does NOT cover 5.0.
        assert!(!offer_includes_v5_0([0x00, 0x02, 0x04, 0x05]));
        // [00, 00, 03, 05] = "Bolt 5.3 exactly".
        assert!(!offer_includes_v5_0([0x00, 0x00, 0x03, 0x05]));
    }

    #[test]
    fn offer_excludes_other_majors() {
        // Bolt 4.x offers MUST NOT match — we only speak 5.0.
        assert!(!offer_includes_v5_0([0x00, 0x00, 0x04, 0x04]));
        // Bolt 6.x (future) MUST NOT match.
        assert!(!offer_includes_v5_0([0x00, 0x00, 0x00, 0x06]));
    }

    #[test]
    fn offer_excludes_nonzero_high_byte() {
        // First byte is reserved and MUST be zero per the spec.
        assert!(!offer_includes_v5_0([0xFF, 0x00, 0x00, 0x05]));
    }

    #[tokio::test]
    async fn handshake_accepts_5_0_offer() {
        let (mut client, server) = duplex(64);
        // Client side: write magic + four offers (5.0 first, padding rest).
        let mut req = Vec::new();
        req.extend_from_slice(&MAGIC_PREAMBLE);
        req.extend_from_slice(&[0x00, 0x00, 0x00, 0x05]); // Bolt 5.0
        req.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // padding
        req.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
        req.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
        client.write_all(&req).await.unwrap();
        // Server side: run perform_handshake on the duplex.
        let (sr, sw) = tokio::io::split(server);
        let v = perform_handshake(sr, sw).await.unwrap();
        assert_eq!(v, BoltVersion::V5_0);
        // Client should now see the 4-byte accept response.
        let mut resp = [0u8; 4];
        client.read_exact(&mut resp).await.unwrap();
        assert_eq!(resp, SERVER_ACCEPT_V5_0);
    }

    #[tokio::test]
    async fn handshake_rejects_when_no_offer_matches() {
        let (mut client, server) = duplex(64);
        let mut req = Vec::new();
        req.extend_from_slice(&MAGIC_PREAMBLE);
        // Four Bolt 4.x offers — none match.
        req.extend_from_slice(&[0x00, 0x00, 0x04, 0x04]);
        req.extend_from_slice(&[0x00, 0x00, 0x03, 0x04]);
        req.extend_from_slice(&[0x00, 0x00, 0x02, 0x04]);
        req.extend_from_slice(&[0x00, 0x00, 0x01, 0x04]);
        client.write_all(&req).await.unwrap();
        let (sr, sw) = tokio::io::split(server);
        let err = perform_handshake(sr, sw).await.unwrap_err();
        assert!(matches!(err, BoltError::HandshakeRejected(_)));
        // Client should see the 4-byte zero-reject response.
        let mut resp = [0u8; 4];
        client.read_exact(&mut resp).await.unwrap();
        assert_eq!(resp, SERVER_REJECT);
    }

    #[tokio::test]
    async fn handshake_rejects_invalid_magic() {
        let (mut client, server) = duplex(64);
        // Bad magic — server should close without writing.
        client.write_all(&[0xDE, 0xAD, 0xBE, 0xEF]).await.unwrap();
        client.write_all(&[0; 16]).await.unwrap();
        let (sr, sw) = tokio::io::split(server);
        let err = perform_handshake(sr, sw).await.unwrap_err();
        assert!(matches!(err, BoltError::HandshakeRejected(_)));
    }
}
