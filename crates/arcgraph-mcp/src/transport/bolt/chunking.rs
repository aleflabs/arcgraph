//! W14δ M5-13 — Bolt chunked-framing layer.
//!
//! Bolt sits two layers deep on the wire:
//!
//! ```text
//! TCP → chunked frame → PackStream-encoded message
//! ```
//!
//! Each Bolt message is split into one or more chunks; each chunk is
//! a 2-byte big-endian length prefix followed by `length` body bytes.
//! A 2-byte zero (`0x00 0x00`) terminates the message — distinct from
//! a length-zero chunk (which is the terminator itself; a chunk MUST
//! have body length ≥ 1 if it precedes the terminator).
//!
//! The chunk size limit is **0xFFFF** bytes per chunk; messages
//! larger than 65535 bytes get split across multiple chunks
//! transparently. The v1.0-α implementation emits one chunk per
//! message until the message itself exceeds 0xFFFF bytes (in which
//! case it splits at the chunk boundary).
//!
//! # Why a separate framing layer
//!
//! The PackStream codec is byte-oriented but knows nothing about
//! message boundaries: a single PackStream value can be arbitrarily
//! large. The chunking layer is what lets the peer say "here ends
//! the current message; the next bytes start a new one". Bolt's
//! design choice (chunks rather than length-prefixed messages) lets
//! a server start emitting RECORD frames before it knows the total
//! result-set size — important for streaming.

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

use super::error::BoltError;

/// Maximum admissible chunk body length per the spec.
pub const MAX_CHUNK_LEN: usize = 0xFFFF;

/// Hard cap on the total dechunked size of a single Bolt message — sum
/// of every chunk body before the `0x0000` terminator. Defends against
/// a hostile peer that streams N×`MAX_CHUNK_LEN` chunks unbounded
/// (16 MiB ≈ 256 max-sized chunks). The 64KiB per-chunk wire limit by
/// itself does NOT bound total message size.
///
/// # Why 16 MiB
///
/// design-v2 §M5 ("Bolt protocol (openCypher driver compatibility)",
/// line 978) lists Bolt as a v1.0-α deliverable but does NOT publish a
/// numeric message-size cap. The 16 MiB value matches
/// [`crate::jsonrpc::MAX_MESSAGE_BYTES`] — the HTTP / stdio framing's
/// cap — so a single per-tenant rate-limit policy applies symmetrically
/// across transports. The MCP agent surface is not bulk-data: tool
/// responses are schema / inspect / search results bounded by per-
/// tenant memory budget per ADR-038 amendment-03 §TIER-2-c.
///
/// W14-retro IR L1-HIGH-3 Vector 1 (`fix/w14-retro-ir-bolt-security`):
/// without this cap, a pre-auth attacker (stub auth at v1.0-α admits
/// any principal) crashes the Bolt server with OOM by streaming
/// N×64KiB chunks.
pub const MAX_BOLT_MESSAGE_BYTES: usize = 16 * 1024 * 1024;

/// Read a complete Bolt message from `reader`, reassembling the
/// chunked body. Returns the dechunked PackStream-encoded bytes
/// (i.e., a single PackStream Struct) ready for [`super::packstream::decode`].
///
/// # Termination
///
/// The function returns when it sees the 2-byte zero terminator. An
/// EOF before the terminator is reported as [`BoltError::Io`] — the
/// peer closed mid-message, which is recoverable only by closing
/// our side.
///
/// # Size cap
///
/// The accumulated dechunked body is capped at
/// [`MAX_BOLT_MESSAGE_BYTES`]. A chunk whose body would cross the
/// boundary surfaces [`BoltError::MessageTooLarge`] BEFORE the chunk
/// is read into memory — the buffer never grows past the cap (the
/// would-be allocation is short-circuited at the size check, not via
/// a post-resize comparison).
///
/// # Returns
///
/// `Ok(Some(buf))` on a complete message. `Ok(None)` if EOF is hit
/// cleanly at a message boundary (the peer closed gracefully between
/// messages — caller terminates the connection).
pub async fn read_chunked_message<R>(reader: &mut R) -> Result<Option<Vec<u8>>, BoltError>
where
    R: AsyncRead + Unpin,
{
    let mut buf = Vec::new();
    let mut header = [0u8; 2];
    let mut first_iteration = true;
    loop {
        // Try to read the chunk header. EOF at the very first chunk
        // = clean disconnect; EOF mid-message = framing fault.
        match reader.read_exact(&mut header).await {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof && first_iteration => {
                return Ok(None);
            }
            Err(e) => {
                return Err(BoltError::Io(format!("chunk header: {e}")));
            }
        }
        first_iteration = false;
        let len = u16::from_be_bytes(header) as usize;
        if len == 0 {
            // Terminator. Buf carries the dechunked payload.
            return Ok(Some(buf));
        }
        let start = buf.len();
        // Cap check BEFORE allocation: refuse the chunk whose body
        // would push the running total past MAX_BOLT_MESSAGE_BYTES.
        // `start + len` cannot overflow on 64-bit since both summands
        // are ≤ MAX_BOLT_MESSAGE_BYTES + 0xFFFF (well under usize::MAX),
        // but the cap check is written `len > max - start` which is
        // saturating-friendly regardless of platform width.
        if len > MAX_BOLT_MESSAGE_BYTES - start {
            return Err(BoltError::MessageTooLarge {
                bytes: start + len,
                max: MAX_BOLT_MESSAGE_BYTES,
            });
        }
        buf.resize(start + len, 0);
        reader
            .read_exact(&mut buf[start..start + len])
            .await
            .map_err(|e| BoltError::Io(format!("chunk body: {e}")))?;
    }
}

/// Write a complete Bolt message to `writer`, splitting `payload`
/// across as many `0xFFFF`-byte chunks as needed and appending the
/// 2-byte terminator. Flushes at the end so the peer sees the
/// message as a unit.
pub async fn write_chunked_message<W>(writer: &mut W, payload: &[u8]) -> Result<(), BoltError>
where
    W: AsyncWrite + Unpin,
{
    let mut offset = 0;
    while offset < payload.len() {
        let len = std::cmp::min(MAX_CHUNK_LEN, payload.len() - offset);
        let header = (len as u16).to_be_bytes();
        writer
            .write_all(&header)
            .await
            .map_err(|e| BoltError::Io(format!("chunk header write: {e}")))?;
        writer
            .write_all(&payload[offset..offset + len])
            .await
            .map_err(|e| BoltError::Io(format!("chunk body write: {e}")))?;
        offset += len;
    }
    // Terminator.
    writer
        .write_all(&[0x00, 0x00])
        .await
        .map_err(|e| BoltError::Io(format!("chunk terminator write: {e}")))?;
    writer
        .flush()
        .await
        .map_err(|e| BoltError::Io(format!("flush: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::duplex;

    #[tokio::test]
    async fn single_chunk_message_roundtrips() {
        let payload = b"hello packstream-message-body";
        let (mut a, mut b) = duplex(1024);
        write_chunked_message(&mut a, payload).await.unwrap();
        let got = read_chunked_message(&mut b).await.unwrap().unwrap();
        assert_eq!(got, payload);
    }

    #[tokio::test]
    async fn empty_message_roundtrips() {
        // A message with empty body is still a valid framed unit:
        // the writer emits ONLY the terminator. The reader sees a
        // length-zero header at first chunk and returns Some(vec![]).
        let (mut a, mut b) = duplex(1024);
        write_chunked_message(&mut a, &[]).await.unwrap();
        let got = read_chunked_message(&mut b).await.unwrap().unwrap();
        assert_eq!(got, b"");
    }

    #[tokio::test]
    async fn large_message_splits_into_multiple_chunks() {
        // 200_000-byte message exceeds MAX_CHUNK_LEN by ~3×.
        let payload = vec![0xAB; 200_000];
        let (mut a, mut b) = duplex(8 * 1024 * 1024);
        write_chunked_message(&mut a, &payload).await.unwrap();
        let got = read_chunked_message(&mut b).await.unwrap().unwrap();
        assert_eq!(got, payload);
    }

    #[tokio::test]
    async fn read_chunked_returns_none_on_clean_eof() {
        // Peer closed without writing anything. Reader returns Ok(None).
        let (a, mut b) = duplex(1024);
        drop(a);
        let got = read_chunked_message(&mut b).await.unwrap();
        assert!(got.is_none());
    }

    #[tokio::test]
    async fn read_chunked_errors_on_mid_message_eof() {
        // Peer wrote a chunk header but no body, then closed.
        let (mut a, mut b) = duplex(1024);
        a.write_all(&[0x00, 0x05]).await.unwrap();
        drop(a);
        let err = read_chunked_message(&mut b).await.unwrap_err();
        assert!(matches!(err, BoltError::Io(_)));
    }

    #[tokio::test]
    async fn read_chunked_rejects_message_exceeding_max_bytes() {
        // W14-retro IR L1-HIGH-3 Vector 1 pin: a hostile peer streaming
        // > MAX_BOLT_MESSAGE_BYTES bytes across the per-chunk-capped
        // wire must hit a hard refusal BEFORE the buffer grows past
        // the cap. Compose 256 × MAX_CHUNK_LEN chunks (= 256 × 65535 =
        // 16,776,960 bytes = MAX_BOLT_MESSAGE_BYTES - 256) of valid
        // framing followed by one more 64 KiB chunk header; the 257th
        // header's `len` (65535) exceeds the remaining headroom (256),
        // so the cap check trips before any body is read. The duplex is
        // sized > 16 MiB so the writer doesn't backlog on the unread
        // bytes.
        let (mut a, mut b) = duplex(32 * 1024 * 1024);

        // Spawn a writer that emits 256 × MAX_CHUNK_LEN chunks
        // (= 16,776,960 bytes dechunked), then one more chunk header
        // (which trips the cap). No terminator — the reader is expected
        // to error out before reaching it.
        let writer = tokio::spawn(async move {
            let body = vec![0xCDu8; MAX_CHUNK_LEN];
            let header = (MAX_CHUNK_LEN as u16).to_be_bytes();
            for _ in 0..256 {
                a.write_all(&header).await.unwrap();
                a.write_all(&body).await.unwrap();
            }
            // 257th chunk header: cap check on `len > max - start`
            // with start = 16,776,960, len = 65535,
            // max - start = MAX_BOLT_MESSAGE_BYTES - 16,776,960 = 256
            // → 65535 > 256 → reject.
            a.write_all(&header).await.unwrap();
            // Keep the writer half alive briefly so the reader doesn't
            // observe an EOF before the cap check fires.
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            drop(a);
        });

        let err = read_chunked_message(&mut b)
            .await
            .expect_err("expected MessageTooLarge");
        match err {
            BoltError::MessageTooLarge { bytes, max } => {
                assert_eq!(max, MAX_BOLT_MESSAGE_BYTES);
                assert!(
                    bytes > MAX_BOLT_MESSAGE_BYTES,
                    "bytes ({bytes}) must exceed max ({MAX_BOLT_MESSAGE_BYTES})"
                );
            }
            other => panic!("expected MessageTooLarge, got {other:?}"),
        }
        // Awaiting writer post-error: it may complete or be cancelled
        // when the duplex pair is dropped — both are fine.
        let _ = writer.await;
    }

    #[tokio::test]
    async fn read_chunked_admits_message_at_max_bytes_boundary() {
        // W14-retro IR R1 MED-2 sister-pin to the rejection test above:
        // a message of EXACTLY MAX_BOLT_MESSAGE_BYTES MUST admit. Catches
        // an off-by-one regression in `read_chunked_message` (e.g.,
        // `len >= MAX - start` instead of `len > MAX - start`) that would
        // reject the boundary case while the rejection test still passes.
        // Mirrors the `decode_admits_depth_at_boundary` discipline pinning
        // the depth-cap's off-by-one.
        //
        // Compose chunks summing to EXACTLY MAX_BOLT_MESSAGE_BYTES:
        //   256 × MAX_CHUNK_LEN  = 256 × 65535       = 16,776,960 bytes
        //   + 1 chunk of 256 bytes                   =        256 bytes
        //   = MAX_BOLT_MESSAGE_BYTES (16,777,216 = 16 MiB)
        // Followed by the 2-byte terminator. The 257th chunk header's
        // cap-check evaluates `len > max - start` → `256 > 256` → admit.
        let (mut a, mut b) = duplex(32 * 1024 * 1024);

        let writer = tokio::spawn(async move {
            let body = vec![0xCDu8; MAX_CHUNK_LEN];
            let header = (MAX_CHUNK_LEN as u16).to_be_bytes();
            for _ in 0..256 {
                a.write_all(&header).await.unwrap();
                a.write_all(&body).await.unwrap();
            }
            // Final boundary-touching chunk: 256 bytes (= MAX_BOLT_MESSAGE_BYTES
            // - 256 × MAX_CHUNK_LEN). cap-check: 256 > (max - 16,776,960) =
            // 256 > 256 = false → admit.
            const TAIL: usize = MAX_BOLT_MESSAGE_BYTES - 256 * MAX_CHUNK_LEN;
            let tail_header = (TAIL as u16).to_be_bytes();
            let tail_body = vec![0xEFu8; TAIL];
            a.write_all(&tail_header).await.unwrap();
            a.write_all(&tail_body).await.unwrap();
            // Terminator.
            a.write_all(&[0x00, 0x00]).await.unwrap();
            drop(a);
        });

        let buf = read_chunked_message(&mut b)
            .await
            .expect("boundary-sized message must decode")
            .expect("boundary-sized message must yield Some(buf)");
        assert_eq!(buf.len(), MAX_BOLT_MESSAGE_BYTES);
        // Spot-check payload borders to confirm the dechunker reassembled
        // the streams correctly (not a same-length-but-wrong-bytes pass).
        assert_eq!(buf[0], 0xCD, "first chunk body should be 0xCD");
        assert_eq!(
            buf[256 * MAX_CHUNK_LEN - 1],
            0xCD,
            "last byte of the 256-chunk run should be 0xCD"
        );
        assert_eq!(
            buf[256 * MAX_CHUNK_LEN],
            0xEF,
            "first byte of the tail chunk should be 0xEF"
        );
        assert_eq!(
            buf[MAX_BOLT_MESSAGE_BYTES - 1],
            0xEF,
            "last byte at the boundary should be 0xEF"
        );
        writer.await.expect("writer task must complete cleanly");
    }

    #[test]
    fn max_bolt_message_bytes_matches_jsonrpc_cap() {
        // Cite-correctness pin: the Bolt cap MUST equal the HTTP /
        // stdio JSON-RPC framing's cap so a single per-tenant rate-
        // limit policy applies symmetrically across transports.
        assert_eq!(
            MAX_BOLT_MESSAGE_BYTES,
            crate::jsonrpc::MAX_MESSAGE_BYTES,
            "Bolt cap diverged from HTTP/stdio JSON-RPC cap"
        );
    }
}
