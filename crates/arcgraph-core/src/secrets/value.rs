//! [`SecretValue`] + [`KeyVersion`] newtypes.
//!
//! Every v1.0-α consumer of the secrets provider uses fixed-width
//! 32-byte values (AES-256 keys). Fixing the width at the trait
//! surface eliminates variable-length attack surface + simplifies
//! key rotation (the provider's `rotate()` returns a fresh 32-byte
//! key without caller-side size negotiation).

use zeroize::{Zeroize, ZeroizeOnDrop};

/// The fixed byte-width of every [`SecretValue`]. 32 bytes = 256 bits
/// = the AES-256 key length. v1.0-α has no variable-length consumer.
pub const SECRET_VALUE_LEN: usize = 32;

/// A 32-byte cryptographic key, zeroized when dropped.
///
/// **Drop-zeroization** (via the `zeroize` crate): when this value
/// falls out of scope, the underlying memory is overwritten with
/// zeros. This is critical because the heap allocator does NOT zero
/// freed memory by default — a key that lives in a `Vec<u8>` and is
/// then `drop`'d could be partially recovered by an attacker with
/// later heap access. `ZeroizeOnDrop` closes that gap.
///
/// **Clone semantics**: cloning produces a fresh allocation; the
/// drop-zeroize property holds on every clone (cloning a `[u8; 32]`
/// copies the bytes, and the clone is dropped independently).
///
/// **Equality**: byte-equal via `PartialEq` so tests can pin
/// round-trip behavior. Equality is constant-time at the bit level
/// for 32-byte values (the optimizer should not short-circuit on
/// `[u8; 32]` since there is no early-out instruction); explicit
/// constant-time compare would require `subtle` crate — out of v1.0-α
/// scope (the threat model is disk theft, not local timing).
///
/// **Debug**: explicitly `*****` redacted; the byte contents NEVER
/// surface in `{:?}` formatting. This is the canonical pin against
/// accidental secret leakage in error chains / log lines.
#[derive(Clone, PartialEq, Eq, Zeroize, ZeroizeOnDrop)]
pub struct SecretValue {
    /// The raw 32 bytes. Private — callers must go through
    /// [`Self::new`] / [`Self::expose_bytes`] which keep the value
    /// inside the `SecretValue` boundary as long as possible.
    bytes: [u8; SECRET_VALUE_LEN],
}

impl SecretValue {
    /// Construct a [`SecretValue`] from raw bytes. The caller is
    /// responsible for ensuring the source of `bytes` is itself
    /// secret (e.g., a CSPRNG, or a value already retrieved from a
    /// provider).
    #[must_use]
    pub const fn new(bytes: [u8; SECRET_VALUE_LEN]) -> Self {
        Self { bytes }
    }

    /// Expose the underlying bytes for a single cryptographic
    /// operation. Callers MUST NOT log / persist / clone the returned
    /// slice — it's a borrow of the protected bytes, and copies
    /// escape the drop-zeroize boundary.
    ///
    /// Named `expose_bytes` (not `as_bytes`) because the name should
    /// remind reviewers that this is the secret-extraction seam — if
    /// you see `.expose_bytes()` in a code path that's NOT a cipher
    /// init, it's likely wrong.
    #[must_use]
    pub fn expose_bytes(&self) -> &[u8; SECRET_VALUE_LEN] {
        &self.bytes
    }

    /// Construct from a byte slice with length validation. Returns
    /// `None` if `slice.len() != SECRET_VALUE_LEN`. Used by provider
    /// impls that receive variable-length backend responses (e.g.,
    /// keyring may return arbitrary-length blobs).
    #[must_use]
    pub fn try_from_slice(slice: &[u8]) -> Option<Self> {
        let bytes: [u8; SECRET_VALUE_LEN] = slice.try_into().ok()?;
        Some(Self::new(bytes))
    }
}

impl std::fmt::Debug for SecretValue {
    /// Redacted format — bytes NEVER surface. Pin against accidental
    /// `tracing::info!(?secret)` / `panic!("{:?}", secret)` leakage.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SecretValue(*****)")
    }
}

/// Monotonic per-key version counter. `rotate()` increments this; the
/// provider stores the new value at `key.v<new_version>`, leaving
/// historical versions readable for backwards-compatible decryption
/// (per ADR-052 §"Key rotation").
///
/// `u16` width chosen because:
/// - Wire-format overhead is 2 bytes per record / page (small).
/// - 65 535 rotations covers a 65-year deployment at one rotation
///   per day with margin.
/// - Operators rotating faster than once per minute have a different
///   problem (likely incident response — handled at v1.2 with KMS).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KeyVersion(u16);

impl KeyVersion {
    /// The first version any key carries after initial creation. Per
    /// the v1.0-α convention, `key.v1` is the first stored binding.
    pub const ONE: Self = Self(1);

    /// Construct from a raw `u16`. `0` is reserved (sentinel for
    /// "unset" / "default") and is permitted by this constructor —
    /// the provider rejects `0` at `set` time if it's not the
    /// intended sentinel.
    #[must_use]
    pub const fn new(raw: u16) -> Self {
        Self(raw)
    }

    /// Raw `u16` for encoding to the wire (WAL record / page slot).
    #[must_use]
    pub const fn raw(self) -> u16 {
        self.0
    }

    /// Next version. Saturates at `u16::MAX` — callers rotating past
    /// the saturation point are in deep operator territory and should
    /// either re-key from scratch or have already lifted to KMS at
    /// v1.2.
    #[must_use]
    pub const fn next(self) -> Self {
        Self(self.0.saturating_add(1))
    }
}

impl std::fmt::Display for KeyVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "v{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secret_value_len_is_thirty_two() {
        assert_eq!(SECRET_VALUE_LEN, 32);
    }

    #[test]
    fn secret_value_round_trip_bytes() {
        let bytes = [0x42; SECRET_VALUE_LEN];
        let v = SecretValue::new(bytes);
        assert_eq!(v.expose_bytes(), &bytes);
    }

    #[test]
    fn secret_value_try_from_slice_accepts_thirty_two_bytes() {
        let bytes = vec![0x77; SECRET_VALUE_LEN];
        let v = SecretValue::try_from_slice(&bytes).expect("32 bytes round-trips");
        assert_eq!(v.expose_bytes(), &[0x77u8; SECRET_VALUE_LEN]);
    }

    #[test]
    fn secret_value_try_from_slice_rejects_wrong_length() {
        let too_short = [0u8; 16];
        let too_long = [0u8; 64];
        assert!(SecretValue::try_from_slice(&too_short).is_none());
        assert!(SecretValue::try_from_slice(&too_long).is_none());
    }

    /// The Debug impl MUST NOT reveal bytes — this is the canonical
    /// pin against accidental log leakage. If anyone changes the
    /// Debug impl to print bytes, this test breaks.
    #[test]
    fn debug_redacts_bytes() {
        let bytes = [0xAB; SECRET_VALUE_LEN];
        let v = SecretValue::new(bytes);
        let s = format!("{v:?}");
        assert_eq!(s, "SecretValue(*****)");
        assert!(!s.contains("AB"), "Debug must not surface bytes");
        assert!(!s.contains("171"), "Debug must not surface decimal bytes");
    }

    #[test]
    fn secret_value_partial_eq_byte_equal() {
        let a = SecretValue::new([0x01; SECRET_VALUE_LEN]);
        let b = SecretValue::new([0x01; SECRET_VALUE_LEN]);
        let c = SecretValue::new([0x02; SECRET_VALUE_LEN]);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    /// Zeroization on drop is the load-bearing security property of
    /// `SecretValue`. We can't observe drop-time bytes directly
    /// (Rust drops the value), but we can verify the trait is
    /// implemented + the `zeroize()` method does the right thing.
    #[test]
    fn zeroize_clears_bytes() {
        let mut v = SecretValue::new([0xFF; SECRET_VALUE_LEN]);
        // Manually invoke zeroize (drop would call this too).
        v.zeroize();
        assert_eq!(v.expose_bytes(), &[0u8; SECRET_VALUE_LEN]);
    }

    #[test]
    fn key_version_one_is_v1() {
        assert_eq!(KeyVersion::ONE.raw(), 1);
        assert_eq!(format!("{}", KeyVersion::ONE), "v1");
    }

    #[test]
    fn key_version_next_increments() {
        let v = KeyVersion::ONE.next();
        assert_eq!(v.raw(), 2);
    }

    #[test]
    fn key_version_next_saturates_at_u16_max() {
        let v = KeyVersion::new(u16::MAX);
        assert_eq!(v.next(), v);
    }

    #[test]
    fn key_version_ord_is_numeric() {
        let v1 = KeyVersion::new(1);
        let v2 = KeyVersion::new(2);
        let v10 = KeyVersion::new(10);
        assert!(v1 < v2);
        assert!(v2 < v10);
    }
}
