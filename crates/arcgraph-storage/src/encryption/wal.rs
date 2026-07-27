//! WAL payload encryption-at-rest (W20β-3 / ADR-052 §"WAL encryption").
//!
//! ## On-disk layout (encrypted payload prefix)
//!
//! ```text
//! offset  field           size  notes
//! 0       enc_magic       4     b"AEAD"
//! 4       key_version     2     u16 LE
//! 6       reserved        2     must be 0
//! 8       iv              12    AES-256-GCM nonce
//! 20      tag             16    AES-256-GCM authentication tag
//! 36      ciphertext      N     N = plaintext.len()
//! ```
//!
//! Total overhead: 36 bytes (header) per encrypted WAL record. The
//! outer `WalRecord` header (44 bytes carrying crc32c, length, record
//! type, lsn, tenant, etc.) stays in clear so recovery can route records
//! by tenant or LSN without the encryption key.
//!
//! ## IV / nonce construction (NIST SP 800-38D §8.2.2 random — #1111 fix)
//!
//! `iv` = a **fresh 96-bit value drawn from the OS CSPRNG per record**
//! (`/dev/urandom` via `keysource::read_csprng_iv` — the same proven
//! entropy source the DEK/KEK wrap path already uses; no DIY crypto per
//! ADR-051 §"Decision item 4" / workspace Apache-2.0 licensing policy). The IV is then STORED in
//! the 36-byte record header (offset 8..20) and read back verbatim at
//! decrypt — so randomizing the write-side IV is invisible to recovery.
//!
//! ### Why random and not deterministic `(segment_no, lsn)` (#1111, SEC-HIGH)
//!
//! The original construction derived the IV from `(segment_no, lsn)`. Its
//! uniqueness rested on a cross-module bootstrap convention — that the
//! framing LSN counter resumes at `last_durable + 1` across restarts. The
//! durable bootstrap (`arcgraph-cli::build_durable`) violates it: it uses
//! plain `WalWriter::spawn`, so the framing LSN counter **restarts at 0**
//! every process life (#825) while the writer re-attaches to the SAME
//! highest segment (`SegmentWriter::open`). Result: a restarted writer
//! re-emits `(segment_no = N, lsn = 1, 2, 3 …)` — pairs already consumed
//! in that segment — i.e. **AES-256-GCM nonce reuse under the same key**:
//! keystream recovery (XOR of plaintexts) + auth-key recovery (forgeries).
//! Two restarts inside one 64 MiB segment suffice. A random per-record IV
//! removes the dependency on the bootstrap convention entirely — the IV is
//! unique regardless of what `(segment_no, lsn)` the writer reuses.
//!
//! ### 96-bit collision safety (back-of-envelope, performance-budget discipline)
//!
//! With a uniform random 96-bit nonce the birthday bound puts the first
//! expected IV collision under one key near `2^48` records (`√(2^96)`).
//! At the v1.0-α baseline (≤ 50 MB/s, ~10⁴–10⁵ records/s) that is well
//! beyond any single key's service life — negligible. NIST SP 800-38D
//! §8.3 caps a single key at `2^32` *messages* for 96-bit-random nonces;
//! enforcing that bound is a key-rotation (KEK/DEK) policy concern
//! deferred to ADR-216 — a forward-pin, NOT implemented here.
//!
//! ## Failure semantics
//!
//! - Encryption failure at encode time → returns `ArcGraphError::Io`
//!   wrapping a synthetic error (encryption errors at encode time
//!   indicate aws-lc-rs internal state issues; never seen on the v1.0-α
//!   hot path). Encode-time failures abort the WAL append.
//! - Decryption failure at recovery time → returns
//!   `ArcGraphError::WalDecryptionFailed`. Silent fallback to
//!   plaintext is FORBIDDEN per
//!   `feedback_noop_trampoline_anti_pattern.md`.

use std::sync::Arc;

use arcgraph_core::{ArcGraphError, KeyVersion, Lsn, Result, SecretsProvider};
use parking_lot::Mutex;

use super::cipher::{AES_GCM_IV_LEN, AES_GCM_TAG_LEN};
use super::keyring::{ENCRYPTION_KEY_NAMESPACE_WAL, KeyRing, KeyRingError};
use super::keysource::read_csprng_iv;

/// 4-byte magic at offset 0 of every encrypted WAL payload.
/// `b"AEAD"` chosen because:
/// - 4 ASCII bytes distinguish encrypted from clear payloads in hex
///   dumps without context.
/// - The byte sequence `0x41 0x45 0x41 0x44` is extremely unlikely to
///   appear at the start of a CRUD payload by accident (M3.a binary
///   records start with type bytes 0..=17 per `WalRecordType`).
pub const WAL_ENCRYPTION_MAGIC: [u8; 4] = *b"AEAD";

/// Length of the encrypted payload header: 4B magic + 2B key_version
/// + 2B reserved + 12B IV + 16B tag = 36 bytes.
pub const WAL_PAYLOAD_HEADER_LEN: usize = 4 + 2 + 2 + AES_GCM_IV_LEN + AES_GCM_TAG_LEN;

/// Configuration for WAL payload encryption attached to a `WalConfig`.
///
/// Holds a [`KeyRing`] fronting a [`SecretsProvider`]. The WAL writer
/// thread takes ownership of this and uses it per-record at encode
/// time; the recovery path takes a clone of the provider + builds its
/// own `KeyRing` for decryption.
pub struct WalEncryption {
    keyring: Arc<Mutex<KeyRing>>,
}

impl std::fmt::Debug for WalEncryption {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WalEncryption").finish_non_exhaustive()
    }
}

impl Clone for WalEncryption {
    fn clone(&self) -> Self {
        Self {
            keyring: Arc::clone(&self.keyring),
        }
    }
}

impl WalEncryption {
    /// Construct WAL encryption from a [`SecretsProvider`] + the
    /// initial current key version. Namespace is fixed to
    /// [`ENCRYPTION_KEY_NAMESPACE_WAL`].
    ///
    /// Fail-fast: pre-fetches the current key on construction so a
    /// missing key surfaces at startup rather than at first append.
    pub fn new(
        provider: Arc<dyn SecretsProvider>,
        current_version: KeyVersion,
    ) -> std::result::Result<Self, KeyRingError> {
        let keyring = KeyRing::new(provider, ENCRYPTION_KEY_NAMESPACE_WAL, current_version);
        // Eager pre-fetch of the current cipher — fail-fast on missing key.
        let _ = keyring.current_cipher()?;
        Ok(Self {
            keyring: Arc::new(Mutex::new(keyring)),
        })
    }

    /// Construct from an already-built keyring. Useful when a single
    /// keyring is shared between WAL + page-store config wiring
    /// (rare but supported).
    #[must_use]
    pub fn from_keyring(keyring: Arc<Mutex<KeyRing>>) -> Self {
        Self { keyring }
    }

    /// Current key version. WAL records produced after `rotate_to()`
    /// fires use the new version.
    pub fn current_version(&self) -> KeyVersion {
        self.keyring.lock().current_version()
    }

    /// Rotate the in-memory current-version pointer. The caller MUST
    /// have already installed the new key via `SecretsProvider::set`
    /// at `arcgraph.wal.encryption_key.v<next.raw()>`. Returns the
    /// new version.
    pub fn rotate_to(&self, next: KeyVersion) -> std::result::Result<KeyVersion, KeyRingError> {
        let mut ring = self.keyring.lock();
        // Validate the new version is fetchable before advancing.
        let _ = ring.cipher_for_version(next)?;
        ring.advance_to(next);
        Ok(next)
    }

    /// Encrypt `plaintext` using the current key + a **fresh random
    /// 96-bit IV** drawn per record from the OS CSPRNG (#1111). Returns
    /// the on-disk payload bytes (the 36-byte header — magic, version,
    /// reserved, IV, tag — followed by the ciphertext). The IV is stored
    /// in the header (offset 8..20) and read back verbatim at decrypt,
    /// so randomizing it does not touch the recovery path.
    ///
    /// `(segment_no, lsn)` are still bound into the GCM AAD via
    /// `build_aad` — that anti-tamper / anti-replay invariant is
    /// independent of the IV and is preserved unchanged.
    pub fn encrypt(&self, segment_no: u64, lsn: Lsn, plaintext: &[u8]) -> Result<Vec<u8>> {
        let ring = self.keyring.lock();
        let current_version = ring.current_version();
        let cipher = ring.current_cipher().map_err(map_keyring_to_io)?;
        drop(ring);

        // Fresh random IV per record (#1111, SEC-HIGH): the previous
        // `(segment_no, lsn)`-derived IV reused nonces across restart
        // because the framing LSN counter resets to 0 (#825). A random
        // 96-bit nonce is unique regardless of the (segment_no, lsn)
        // the writer reuses. Drawn from the same `/dev/urandom` source
        // the DEK/KEK wrap path uses — no DIY crypto (ADR-051 §item-4).
        let iv = read_csprng_iv().map_err(|reason| {
            ArcGraphError::Io(std::io::Error::other(format!("wal iv csprng: {reason}")))
        })?;
        let aad = build_aad(current_version, segment_no, lsn);
        let ct_with_tag = cipher
            .encrypt(&iv, &aad, plaintext)
            .map_err(map_cipher_to_io)?;

        // ct_with_tag layout: [ciphertext NB] [tag 16B]
        let ct_len = ct_with_tag.len().saturating_sub(AES_GCM_TAG_LEN);
        let mut out = Vec::with_capacity(WAL_PAYLOAD_HEADER_LEN + ct_len);
        out.extend_from_slice(&WAL_ENCRYPTION_MAGIC);
        out.extend_from_slice(&current_version.raw().to_le_bytes());
        out.extend_from_slice(&[0u8, 0u8]); // reserved
        out.extend_from_slice(&iv);
        // tag goes BEFORE ciphertext in the on-disk layout so a reader
        // can validate the tag-bound metadata before allocating the
        // ciphertext buffer. aws-lc-rs's open_in_place expects tag
        // APPENDED — we splice the tag back to the tail at decrypt.
        let tag_start = ct_len;
        out.extend_from_slice(&ct_with_tag[tag_start..tag_start + AES_GCM_TAG_LEN]);
        out.extend_from_slice(&ct_with_tag[..tag_start]);
        Ok(out)
    }

    /// Decrypt an on-disk payload produced by [`Self::encrypt`].
    /// Validates the magic + reserved bytes; routes to the historical
    /// cipher by the stamped `key_version`; returns the recovered
    /// plaintext.
    ///
    /// Errors:
    /// - Payload doesn't start with [`WAL_ENCRYPTION_MAGIC`] →
    ///   `ArcGraphError::WalDecryptionFailed` with reason
    ///   "missing AEAD magic".
    /// - Reserved bytes non-zero → `WalDecryptionFailed` (future
    ///   versions may reclaim; today must be zero).
    /// - Key version not retrievable from the provider →
    ///   `WalDecryptionFailed` with the provider's reason chain.
    /// - AES-GCM tag mismatch → `WalDecryptionFailed` with reason
    ///   "tag mismatch".
    pub fn decrypt(&self, segment_no: u64, lsn: Lsn, payload: &[u8]) -> Result<Vec<u8>> {
        if payload.len() < WAL_PAYLOAD_HEADER_LEN {
            return Err(ArcGraphError::WalDecryptionFailed {
                lsn,
                key_version: 0,
                reason: format!(
                    "encrypted payload too short: got {} bytes, need at least {}",
                    payload.len(),
                    WAL_PAYLOAD_HEADER_LEN
                ),
            });
        }
        if payload[0..4] != WAL_ENCRYPTION_MAGIC {
            return Err(ArcGraphError::WalDecryptionFailed {
                lsn,
                key_version: 0,
                reason: format!(
                    "missing AEAD magic: got {:02x?}, expected {:02x?}",
                    &payload[0..4],
                    WAL_ENCRYPTION_MAGIC
                ),
            });
        }
        let key_version = u16::from_le_bytes([payload[4], payload[5]]);
        if payload[6] != 0 || payload[7] != 0 {
            return Err(ArcGraphError::WalDecryptionFailed {
                lsn,
                key_version,
                reason: format!(
                    "reserved bytes non-zero: 0x{:02x}{:02x}",
                    payload[6], payload[7]
                ),
            });
        }

        let mut iv = [0u8; AES_GCM_IV_LEN];
        iv.copy_from_slice(&payload[8..20]);
        let mut tag = [0u8; AES_GCM_TAG_LEN];
        tag.copy_from_slice(&payload[20..36]);
        let ct = &payload[WAL_PAYLOAD_HEADER_LEN..];

        let ring = self.keyring.lock();
        let cipher = ring
            .cipher_for_version(KeyVersion::new(key_version))
            .map_err(|e| map_keyring_to_wal_decryption(lsn, key_version, e))?;
        drop(ring);

        // Re-glue tag back to ciphertext tail for aws-lc-rs's
        // open_in_place which expects [ciphertext|tag] layout.
        let mut ct_with_tag = Vec::with_capacity(ct.len() + AES_GCM_TAG_LEN);
        ct_with_tag.extend_from_slice(ct);
        ct_with_tag.extend_from_slice(&tag);

        let aad = build_aad(KeyVersion::new(key_version), segment_no, lsn);
        cipher
            .decrypt(&iv, &aad, &ct_with_tag)
            .map_err(|e| ArcGraphError::WalDecryptionFailed {
                lsn,
                key_version,
                reason: format!("aead decryption failed: {e}"),
            })
    }
}

/// Module-level helper exposed to recovery code paths that don't have
/// a `WalEncryption` instance handy (e.g., the segment dispatcher
/// peeks the magic to detect encrypted records).
///
/// **PR #373 R1 §L-3 tightening:** to reduce false-positive
/// classification on adversarial clear payloads, the check requires
/// the full header length (`WAL_PAYLOAD_HEADER_LEN` = 36 bytes) plus
/// the `0` reserved bytes at offset 6..8. A clear payload that
/// happens to start with `b"AEAD"` but is shorter than 36 bytes, or
/// has non-zero reserved bytes, is classified as Clear (the encrypted
/// path requires the full header — a partial match cannot be an
/// encrypted payload by the on-disk layout). The 4-byte magic match
/// alone is insufficient: a tenant-supplied string payload starting
/// with the exact bytes `0x41 0x45 0x41 0x44 ... 0x00 0x00` would
/// otherwise mis-classify.
#[must_use]
pub fn is_encrypted_wal_payload(payload: &[u8]) -> bool {
    payload.len() >= WAL_PAYLOAD_HEADER_LEN
        && payload[0..4] == WAL_ENCRYPTION_MAGIC
        && payload[6] == 0
        && payload[7] == 0
}

/// Stand-alone encrypt helper for callers without a `WalEncryption`
/// instance (e.g., property tests).
pub fn encrypt_wal_payload(
    enc: &WalEncryption,
    segment_no: u64,
    lsn: Lsn,
    plaintext: &[u8],
) -> Result<Vec<u8>> {
    enc.encrypt(segment_no, lsn, plaintext)
}

/// Stand-alone decrypt helper.
pub fn decrypt_wal_payload(
    enc: &WalEncryption,
    segment_no: u64,
    lsn: Lsn,
    payload: &[u8],
) -> Result<Vec<u8>> {
    enc.decrypt(segment_no, lsn, payload)
}

/// Discriminator returned by [`PayloadEncryption::peek`] so callers
/// can route between clear + encrypted decode paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PayloadEncryption {
    /// Payload is plaintext as today.
    Clear,
    /// Payload starts with `WAL_ENCRYPTION_MAGIC` and the stamped
    /// `key_version` is `key_version`.
    Encrypted {
        /// The version stamped in the payload's header. Caller uses
        /// this to route to the historical cipher.
        key_version: u16,
    },
}

impl PayloadEncryption {
    /// Inspect `payload` to decide between [`Self::Clear`] +
    /// [`Self::Encrypted`].
    ///
    /// **Discriminator strength (PR #373 R1 §L-3 tightening):** the
    /// classifier requires (a) the 4-byte `AEAD` magic AND (b) zeroed
    /// reserved bytes at offset 6..8 AND (c) the full
    /// `WAL_PAYLOAD_HEADER_LEN` (36 bytes) of header. A tenant-string
    /// payload that happens to start with `b"AEAD"` but is shorter
    /// than 36 bytes, OR has non-zero bytes at offset 6..8, is
    /// classified as [`Self::Clear`]. This narrows the false-positive
    /// surface from "first 4 bytes match" to "first 8 bytes match a
    /// specific pattern AND payload is at least 36 bytes" — a
    /// `(2^16 - 1) / 2^16 ≈ 99.998 %` reduction in adversarial
    /// false-positive density for a string-prefix attacker.
    #[must_use]
    pub fn peek(payload: &[u8]) -> Self {
        if is_encrypted_wal_payload(payload) {
            Self::Encrypted {
                key_version: u16::from_le_bytes([payload[4], payload[5]]),
            }
        } else {
            Self::Clear
        }
    }

    /// Convenience: extract the key version if this is the encrypted
    /// variant. Returns `None` for `Clear`.
    #[must_use]
    pub fn as_key_version(self) -> Option<u16> {
        match self {
            Self::Encrypted { key_version } => Some(key_version),
            Self::Clear => None,
        }
    }
}

/// Build the AAD (additional authenticated data) for a WAL record's
/// encryption. AAD binds `(key_version, segment_no, lsn)` into the
/// GCM tag so an attacker cannot replay a ciphertext under a
/// different version / segment / LSN.
fn build_aad(key_version: KeyVersion, segment_no: u64, lsn: Lsn) -> Vec<u8> {
    let mut aad = Vec::with_capacity(2 + 8 + 8);
    aad.extend_from_slice(&key_version.raw().to_le_bytes());
    aad.extend_from_slice(&segment_no.to_le_bytes());
    aad.extend_from_slice(&lsn.raw().to_le_bytes());
    aad
}

fn map_keyring_to_io(err: KeyRingError) -> ArcGraphError {
    ArcGraphError::Io(std::io::Error::other(format!("wal keyring: {err}")))
}

fn map_cipher_to_io(err: super::cipher::CipherError) -> ArcGraphError {
    ArcGraphError::Io(std::io::Error::other(format!("wal cipher: {err}")))
}

fn map_keyring_to_wal_decryption(lsn: Lsn, key_version: u16, err: KeyRingError) -> ArcGraphError {
    ArcGraphError::WalDecryptionFailed {
        lsn,
        key_version,
        reason: format!("keyring resolve failed: {err}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encryption::keyring::install_random_key;
    use arcgraph_core::{EnvSecretsProvider, KeyVersion};

    fn unique_prefix(suffix: &str) -> String {
        let pid = std::process::id();
        let thread_id = std::thread::current().id();
        format!("ARCGRAPH_WAL_ENC_TEST_{pid}_{thread_id:?}_{suffix}_").replace([' ', '(', ')'], "_")
    }

    fn provider_with_v1(prefix: &str) -> Arc<dyn SecretsProvider> {
        let p: Arc<dyn SecretsProvider> = Arc::new(
            EnvSecretsProvider::without_startup_warn_for_tests(prefix.to_owned()),
        );
        install_random_key(&*p, ENCRYPTION_KEY_NAMESPACE_WAL, KeyVersion::ONE).expect("install v1");
        p
    }

    #[test]
    fn wal_payload_header_len_is_36() {
        assert_eq!(WAL_PAYLOAD_HEADER_LEN, 36);
    }

    #[test]
    fn round_trip_encrypt_decrypt() {
        let prefix = unique_prefix("round_trip");
        let provider = provider_with_v1(&prefix);
        let enc = WalEncryption::new(provider, KeyVersion::ONE).expect("init");
        let pt = b"some-arcgraph-payload-bytes";
        let ct = enc.encrypt(7, Lsn::new(42), pt).expect("encrypt");
        assert!(ct.len() > pt.len(), "ciphertext must be longer (overhead)");
        assert_eq!(ct.len(), pt.len() + WAL_PAYLOAD_HEADER_LEN);
        assert!(is_encrypted_wal_payload(&ct));
        let back = enc.decrypt(7, Lsn::new(42), &ct).expect("decrypt");
        assert_eq!(back, pt);
    }

    #[test]
    fn peek_returns_clear_for_plaintext() {
        assert_eq!(
            PayloadEncryption::peek(b"hello plain"),
            PayloadEncryption::Clear
        );
    }

    #[test]
    fn peek_returns_encrypted_for_aead_magic() {
        // Build a payload with full WAL_PAYLOAD_HEADER_LEN: magic +
        // key_version + reserved(0) + iv(0) + tag(0). PR #373 R1
        // §L-3 tightened classifier requires the full 36-byte
        // header to classify as Encrypted.
        let mut buf = vec![b'A', b'E', b'A', b'D']; // magic
        buf.extend_from_slice(&3u16.to_le_bytes()); // key_version
        buf.extend_from_slice(&[0u8, 0u8]); // reserved (must be 0)
        buf.extend_from_slice(&[0u8; AES_GCM_IV_LEN]); // iv placeholder
        buf.extend_from_slice(&[0u8; AES_GCM_TAG_LEN]); // tag placeholder
        assert_eq!(buf.len(), WAL_PAYLOAD_HEADER_LEN);
        match PayloadEncryption::peek(&buf) {
            PayloadEncryption::Encrypted { key_version } => assert_eq!(key_version, 3),
            other => panic!("expected Encrypted, got {other:?}"),
        }
    }

    /// PR #373 R1 §L-3: a clear payload starting with `b"AEAD"` but
    /// SHORTER than `WAL_PAYLOAD_HEADER_LEN` (36 B) must classify as
    /// `Clear` — the tightened discriminator narrows the false-
    /// positive surface from "4-byte prefix match" to "full-header
    /// shape match".
    #[test]
    fn peek_classifies_short_aead_prefix_as_clear() {
        // Less than 36 bytes — even with the AEAD magic, this is
        // NOT a valid encrypted payload.
        let buf = vec![b'A', b'E', b'A', b'D', 0, 0, 0, 0, 0, 0];
        assert!(
            buf.len() < WAL_PAYLOAD_HEADER_LEN,
            "test fixture: must be shorter than the encrypted header"
        );
        match PayloadEncryption::peek(&buf) {
            PayloadEncryption::Clear => {}
            other => panic!("expected Clear for short AEAD-prefixed payload, got {other:?}"),
        }
    }

    /// PR #373 R1 §L-3: a clear payload with the AEAD magic + valid
    /// length but NON-ZERO reserved bytes (offset 6..8) is still
    /// Clear — the discriminator also requires the reserved bytes
    /// to be zero.
    #[test]
    fn peek_classifies_nonzero_reserved_bytes_as_clear() {
        let mut buf = vec![b'A', b'E', b'A', b'D']; // magic
        buf.extend_from_slice(&3u16.to_le_bytes()); // key_version
        buf.extend_from_slice(&[0xFFu8, 0xFFu8]); // reserved NON-ZERO
        buf.extend_from_slice(&[0u8; AES_GCM_IV_LEN]); // iv placeholder
        buf.extend_from_slice(&[0u8; AES_GCM_TAG_LEN]); // tag placeholder
        assert_eq!(buf.len(), WAL_PAYLOAD_HEADER_LEN);
        match PayloadEncryption::peek(&buf) {
            PayloadEncryption::Clear => {}
            other => {
                panic!("expected Clear for AEAD-magic-but-nonzero-reserved payload, got {other:?}")
            }
        }
    }

    #[test]
    fn decrypt_rejects_missing_magic() {
        let prefix = unique_prefix("no_magic");
        let provider = provider_with_v1(&prefix);
        let enc = WalEncryption::new(provider, KeyVersion::ONE).unwrap();
        let bogus = vec![0xFFu8; WAL_PAYLOAD_HEADER_LEN + 8];
        let err = enc.decrypt(0, Lsn::new(1), &bogus).unwrap_err();
        match err {
            ArcGraphError::WalDecryptionFailed { reason, .. } => {
                assert!(reason.contains("AEAD magic"), "got: {reason}");
            }
            other => panic!("expected WalDecryptionFailed, got {other:?}"),
        }
    }

    #[test]
    fn decrypt_rejects_reserved_nonzero() {
        let prefix = unique_prefix("reserved");
        let provider = provider_with_v1(&prefix);
        let enc = WalEncryption::new(provider, KeyVersion::ONE).unwrap();
        let pt = b"hello";
        let mut ct = enc.encrypt(0, Lsn::new(1), pt).unwrap();
        ct[6] = 0xFF; // Tamper with reserved.
        let err = enc.decrypt(0, Lsn::new(1), &ct).unwrap_err();
        match err {
            ArcGraphError::WalDecryptionFailed { reason, .. } => {
                assert!(reason.contains("reserved bytes non-zero"), "got: {reason}");
            }
            other => panic!("expected WalDecryptionFailed, got {other:?}"),
        }
    }

    #[test]
    fn decrypt_rejects_tag_corruption() {
        let prefix = unique_prefix("tag_corrupt");
        let provider = provider_with_v1(&prefix);
        let enc = WalEncryption::new(provider, KeyVersion::ONE).unwrap();
        let pt = b"hello";
        let mut ct = enc.encrypt(0, Lsn::new(1), pt).unwrap();
        // Tag lives at offset 20..36; flip a tag byte.
        ct[25] ^= 0x80;
        let err = enc.decrypt(0, Lsn::new(1), &ct).unwrap_err();
        match err {
            ArcGraphError::WalDecryptionFailed { reason, .. } => {
                assert!(reason.contains("aead decryption failed"), "got: {reason}");
            }
            other => panic!("expected WalDecryptionFailed, got {other:?}"),
        }
    }

    #[test]
    fn decrypt_rejects_ciphertext_corruption() {
        let prefix = unique_prefix("ct_corrupt");
        let provider = provider_with_v1(&prefix);
        let enc = WalEncryption::new(provider, KeyVersion::ONE).unwrap();
        let pt = b"hello-arcgraph";
        let mut ct = enc.encrypt(0, Lsn::new(1), pt).unwrap();
        // Ciphertext lives at offset 36..; flip a ciphertext byte.
        ct[WAL_PAYLOAD_HEADER_LEN + 3] ^= 0x40;
        let err = enc.decrypt(0, Lsn::new(1), &ct).unwrap_err();
        assert!(matches!(err, ArcGraphError::WalDecryptionFailed { .. }));
    }

    /// Per ADR-052 §"AAD": decryption with a different segment_no /
    /// LSN must fail because those are bound into the GCM tag via
    /// AAD. This pins the anti-replay invariant — an attacker cannot
    /// move a ciphertext to a different (segment, lsn) slot.
    #[test]
    fn decrypt_rejects_different_lsn_aad() {
        let prefix = unique_prefix("aad_lsn");
        let provider = provider_with_v1(&prefix);
        let enc = WalEncryption::new(provider, KeyVersion::ONE).unwrap();
        let pt = b"hello";
        let ct = enc.encrypt(0, Lsn::new(1), pt).unwrap();
        let err = enc.decrypt(0, Lsn::new(2), &ct).unwrap_err();
        assert!(matches!(err, ArcGraphError::WalDecryptionFailed { .. }));
    }

    #[test]
    fn decrypt_rejects_different_segment_aad() {
        let prefix = unique_prefix("aad_seg");
        let provider = provider_with_v1(&prefix);
        let enc = WalEncryption::new(provider, KeyVersion::ONE).unwrap();
        let pt = b"hello";
        let ct = enc.encrypt(0, Lsn::new(1), pt).unwrap();
        let err = enc.decrypt(7, Lsn::new(1), &ct).unwrap_err();
        assert!(matches!(err, ArcGraphError::WalDecryptionFailed { .. }));
    }

    #[test]
    fn key_rotation_old_records_still_decryptable() {
        let prefix = unique_prefix("rotation");
        let provider = provider_with_v1(&prefix);
        let enc = WalEncryption::new(Arc::clone(&provider), KeyVersion::ONE).unwrap();

        // Encrypt under v1.
        let pt_v1 = b"v1-payload";
        let ct_v1 = enc.encrypt(0, Lsn::new(10), pt_v1).unwrap();
        assert_eq!(enc.current_version(), KeyVersion::ONE);

        // Install v2 in the provider + rotate.
        install_random_key(&*provider, ENCRYPTION_KEY_NAMESPACE_WAL, KeyVersion::new(2))
            .expect("install v2");
        enc.rotate_to(KeyVersion::new(2)).expect("rotate");
        assert_eq!(enc.current_version(), KeyVersion::new(2));

        // New record encrypts under v2.
        let pt_v2 = b"v2-payload";
        let ct_v2 = enc.encrypt(0, Lsn::new(20), pt_v2).unwrap();

        // Old record (v1) still decrypts.
        let back_v1 = enc.decrypt(0, Lsn::new(10), &ct_v1).unwrap();
        assert_eq!(back_v1, pt_v1);

        // New record (v2) decrypts.
        let back_v2 = enc.decrypt(0, Lsn::new(20), &ct_v2).unwrap();
        assert_eq!(back_v2, pt_v2);
    }

    #[test]
    fn missing_key_for_version_surfaces_decryption_failed() {
        let prefix = unique_prefix("missing_v");
        let provider = provider_with_v1(&prefix);
        let enc = WalEncryption::new(Arc::clone(&provider), KeyVersion::ONE).unwrap();
        let pt = b"hello";
        let mut ct = enc.encrypt(0, Lsn::new(1), pt).unwrap();
        // Tamper key_version byte to point at a non-existent version.
        ct[4] = 0xFE;
        ct[5] = 0xFE;
        let err = enc.decrypt(0, Lsn::new(1), &ct).unwrap_err();
        match err {
            ArcGraphError::WalDecryptionFailed {
                key_version,
                reason,
                ..
            } => {
                assert_eq!(key_version, 0xFEFE);
                assert!(reason.contains("keyring resolve failed"), "got: {reason}");
            }
            other => panic!("expected WalDecryptionFailed, got {other:?}"),
        }
    }

    /// Extract the 12-byte IV stamped in an encrypted WAL payload
    /// header (offset 8..20 per the on-disk layout).
    fn iv_of(payload: &[u8]) -> [u8; AES_GCM_IV_LEN] {
        let mut iv = [0u8; AES_GCM_IV_LEN];
        iv.copy_from_slice(&payload[8..20]);
        iv
    }

    /// #1111 (SEC-HIGH) — THE NONCE-REUSE REPRO (load-bearing security
    /// proof). Two writer incarnations re-emit the SAME `(segment_no,
    /// lsn)` pairs into the SAME segment — exactly the production
    /// restart scenario: `build_durable` uses plain `WalWriter::spawn`,
    /// so the framing LSN counter restarts at 0 (#825) while the writer
    /// re-attaches to the highest existing segment. With random IVs the
    /// two incarnations MUST stamp DISTINCT IVs (no nonce reuse).
    ///
    /// **RED-on-revert:** revert `encrypt` to the deterministic
    /// `derive_iv(segment_no, lsn)` and these IVs become IDENTICAL —
    /// the assertion below fires. That collision IS the AES-256-GCM
    /// nonce reuse this fix eliminates (keystream + auth-key recovery).
    #[test]
    fn nonce_not_reused_across_writer_restart_same_segment_lsn() {
        let prefix = unique_prefix("nonce_reuse_repro");
        let provider = provider_with_v1(&prefix);

        // First writer incarnation, segment 0, LSNs 1..=8.
        let enc_a = WalEncryption::new(Arc::clone(&provider), KeyVersion::ONE).expect("init a");
        // Second incarnation re-attaching to the SAME segment after a
        // restart that reset the framing LSN counter to 0 (#825) — so
        // it re-emits the IDENTICAL (segment_no, lsn) pairs.
        let enc_b = WalEncryption::new(Arc::clone(&provider), KeyVersion::ONE).expect("init b");

        for lsn in 1u64..=8 {
            let pt_a = format!("incarnation-A-record-{lsn}");
            let pt_b = format!("incarnation-B-record-{lsn}"); // DIFFERENT plaintext
            let ct_a = enc_a.encrypt(0, Lsn::new(lsn), pt_a.as_bytes()).unwrap();
            let ct_b = enc_b.encrypt(0, Lsn::new(lsn), pt_b.as_bytes()).unwrap();
            let iv_a = iv_of(&ct_a);
            let iv_b = iv_of(&ct_b);
            assert_ne!(
                iv_a, iv_b,
                "NONCE REUSE at (segment_no=0, lsn={lsn}): two writer \
                 incarnations stamped the SAME IV under the same key — \
                 with different plaintexts this is catastrophic AES-GCM \
                 nonce reuse (#1111). Random IVs must make these distinct."
            );
        }
    }

    /// #1111 round-trip: encrypting with a RANDOM IV still decrypts,
    /// because decrypt reads the STORED IV from the header (offset
    /// 8..20) rather than re-deriving it. Random write-side IVs do not
    /// touch the read/recovery path.
    #[test]
    fn round_trip_with_random_iv_decrypts() {
        let prefix = unique_prefix("random_iv_round_trip");
        let provider = provider_with_v1(&prefix);
        let enc = WalEncryption::new(provider, KeyVersion::ONE).expect("init");
        let pt = b"random-iv-payload-must-decrypt";

        // Encrypt the same (segment_no, lsn) twice → distinct random IVs
        // (proves the IV is NOT (segment_no, lsn)-derived) but BOTH
        // round-trip to the original plaintext via the stored IV.
        let ct1 = enc.encrypt(3, Lsn::new(99), pt).expect("encrypt 1");
        let ct2 = enc.encrypt(3, Lsn::new(99), pt).expect("encrypt 2");
        assert_ne!(
            iv_of(&ct1),
            iv_of(&ct2),
            "two encrypts of the same (segment, lsn) must use distinct random IVs"
        );
        assert_eq!(enc.decrypt(3, Lsn::new(99), &ct1).unwrap(), pt);
        assert_eq!(enc.decrypt(3, Lsn::new(99), &ct2).unwrap(), pt);
    }

    /// #1111 (rewrite of `iv_unique_across_simulated_writer_restart`,
    /// formerly PR #373 R1 §N-1). The original asserted cross-restart
    /// IV uniqueness via the monotonic-LSN-across-restart bootstrap
    /// convention — which production (`build_durable`) VIOLATES (#825).
    /// Rewritten to assert random-IV uniqueness regardless of
    /// `(segment_no, lsn)`: even when both counters fully repeat (the
    /// restart scenario), every produced IV is distinct because it is
    /// drawn fresh from the CSPRNG, not derived from the counters.
    #[test]
    fn iv_unique_across_simulated_writer_restart() {
        let prefix = unique_prefix("iv_restart_random");
        let provider = provider_with_v1(&prefix);
        let mut seen = std::collections::HashSet::new();

        // Two incarnations, each writing the EXACT SAME (segment_no,
        // lsn) coordinates — the worst case the deterministic IV could
        // not survive. Random IVs must keep every nonce unique.
        for _incarnation in 0..2 {
            let enc = WalEncryption::new(Arc::clone(&provider), KeyVersion::ONE).expect("init");
            for seg in 0u64..3 {
                for lsn in 1u64..50 {
                    let ct = enc.encrypt(seg, Lsn::new(lsn), b"payload").unwrap();
                    assert!(
                        seen.insert(iv_of(&ct)),
                        "IV collision at (seg={seg}, lsn={lsn}) across writer \
                         restart — random IV uniqueness violated; production \
                         reuses these coordinates after an LSN-reset restart"
                    );
                }
            }
        }
    }

    /// Sanity that the CSPRNG path is wired (not returning a constant):
    /// N independent encrypts of the same input yield N distinct IVs.
    #[test]
    fn random_ivs_are_all_distinct() {
        let prefix = unique_prefix("iv_statistical");
        let provider = provider_with_v1(&prefix);
        let enc = WalEncryption::new(provider, KeyVersion::ONE).expect("init");
        let mut seen = std::collections::HashSet::new();
        for _ in 0..256 {
            let ct = enc.encrypt(0, Lsn::new(1), b"x").unwrap();
            assert!(
                seen.insert(iv_of(&ct)),
                "duplicate IV among 256 random draws — CSPRNG path not wired \
                 (a constant/derived IV would collide immediately)"
            );
        }
    }
}
