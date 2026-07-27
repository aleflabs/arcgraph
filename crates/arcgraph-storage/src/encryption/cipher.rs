//! Thin AES-256-GCM cipher over `aws-lc-rs::aead`.
//!
//! `aws-lc-rs` is the FIPS-140-3-candidate AEAD library used by
//! `rustls` + `jsonwebtoken` across the workspace (per ADR-049).
//! Promoting it as the storage-side AEAD adds zero net new transitive
//! deps. The interface here is deliberately small: `encrypt_in_place`
//! / `decrypt_in_place`, plus the IV / tag / key length constants.
//!
//! ## On the choice of in-place
//!
//! `aws-lc-rs::aead::LessSafeKey::seal_in_place_append_tag` appends
//! the 16-byte tag to the supplied buffer. Decryption uses
//! `open_in_place` which expects the ciphertext + appended tag.
//! Both are amortized O(n) over the payload with the AES-NI / NEON
//! Crypto-extension hot path; benchmarks live in
//! `benches/wal_encryption_overhead.rs`.

use aws_lc_rs::aead::{AES_256_GCM, Aad, LessSafeKey, Nonce, UnboundKey};
use thiserror::Error;

use arcgraph_core::SECRET_VALUE_LEN;

/// AES-256 key length in bytes. Equals
/// [`arcgraph_core::SECRET_VALUE_LEN`].
pub const AEAD_KEY_LEN: usize = SECRET_VALUE_LEN;

/// AES-256-GCM IV / nonce length in bytes (NIST SP 800-38D §5.2.1.1
/// recommends 96-bit IVs for the standard construction).
pub const AES_GCM_IV_LEN: usize = 12;

/// AES-256-GCM authentication tag length in bytes.
pub const AES_GCM_TAG_LEN: usize = 16;

/// Errors surfaced by the cipher boundary. Caller MUST map these to
/// `ArcGraphError::WalDecryptionFailed` / `PageDecryptionFailed`
/// per the surface where the error originates.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CipherError {
    /// `aws-lc-rs` rejected the key bytes (always indicates a logic
    /// bug — keys are fixed-width 32 bytes by construction).
    #[error("aead key initialization failed: {0}")]
    KeyInit(String),

    /// Decryption failed — tag mismatch, wrong key, wrong IV, or
    /// truncated ciphertext.
    #[error("aead decryption failed: {0}")]
    Decryption(String),

    /// Encryption failed — typically allocator failure or buffer-
    /// length overflow. Never seen on the v1.0-α hot path.
    #[error("aead encryption failed: {0}")]
    Encryption(String),
}

/// A keyed AES-256-GCM cipher.
///
/// Internal state holds the `aws-lc-rs::aead::LessSafeKey` which is
/// stateless for encryption / decryption (each call provides its own
/// nonce). Cloneable via `Aes256GcmCipher::from_key`; the underlying
/// key material is held inside the `LessSafeKey` and zeroized when
/// the cipher is dropped (aws-lc-rs's `UnboundKey::drop` clears the
/// key bytes).
pub struct Aes256GcmCipher {
    inner: LessSafeKey,
}

impl std::fmt::Debug for Aes256GcmCipher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Aes256GcmCipher")
            .field("algorithm", &"AES-256-GCM")
            .finish_non_exhaustive()
    }
}

impl Aes256GcmCipher {
    /// Construct from raw 32-byte key material. Callers typically
    /// route via [`super::keyring::KeyRing::cipher_for_version`] which
    /// holds the [`arcgraph_core::SecretValue`] across the call so the
    /// key never leaves the secrets boundary uncontrolled.
    pub fn from_key(key_bytes: &[u8; AEAD_KEY_LEN]) -> Result<Self, CipherError> {
        let unbound = UnboundKey::new(&AES_256_GCM, key_bytes)
            .map_err(|e| CipherError::KeyInit(format!("{e:?}")))?;
        Ok(Self {
            inner: LessSafeKey::new(unbound),
        })
    }

    /// Encrypt `plaintext` into a fresh `Vec<u8>` whose layout is
    /// `[ciphertext (plaintext.len() bytes)] [tag (16 bytes)]`. The
    /// caller frames the 12-byte `iv` separately + verifies the IV
    /// uniqueness contract.
    ///
    /// `aad` (additional authenticated data) is bound into the GCM
    /// tag — typically the layout magic + key_version so an attacker
    /// cannot replay a ciphertext under a different layout / version.
    pub fn encrypt(
        &self,
        iv: &[u8; AES_GCM_IV_LEN],
        aad: &[u8],
        plaintext: &[u8],
    ) -> Result<Vec<u8>, CipherError> {
        let encoded_len = plaintext
            .len()
            .checked_add(AES_GCM_TAG_LEN)
            .ok_or_else(|| CipherError::Encryption("ciphertext length overflow".to_owned()))?;
        // Reserve the final ciphertext+tag size up front. Besides avoiding a
        // growth copy, this lets spill account the single dynamic buffer
        // exactly before invoking the cipher.
        let mut out = Vec::with_capacity(encoded_len);
        out.extend_from_slice(plaintext);
        let nonce = Nonce::assume_unique_for_key(*iv);
        let aad = Aad::from(aad);
        self.inner
            .seal_in_place_append_tag(nonce, aad, &mut out)
            .map_err(|e| CipherError::Encryption(format!("{e:?}")))?;
        Ok(out)
    }

    /// Authenticate and decrypt an owned ciphertext buffer in place.
    ///
    /// The returned vector reuses the input allocation and is truncated to
    /// the plaintext length. Spill uses this boundary so its staging charge
    /// covers exactly one dynamic frame buffer instead of a ciphertext copy
    /// plus a second plaintext allocation.
    pub(crate) fn decrypt_owned(
        &self,
        iv: &[u8; AES_GCM_IV_LEN],
        aad: &[u8],
        mut ciphertext_with_tag: Vec<u8>,
    ) -> Result<Vec<u8>, CipherError> {
        if ciphertext_with_tag.len() < AES_GCM_TAG_LEN {
            return Err(CipherError::Decryption(format!(
                "ciphertext too short: got {} bytes, need at least {} for the tag",
                ciphertext_with_tag.len(),
                AES_GCM_TAG_LEN
            )));
        }
        let plaintext_len = ciphertext_with_tag.len() - AES_GCM_TAG_LEN;
        let allocation_start = ciphertext_with_tag.as_ptr();
        let nonce = Nonce::assume_unique_for_key(*iv);
        let aad = Aad::from(aad);
        let plaintext = self
            .inner
            .open_in_place(nonce, aad, &mut ciphertext_with_tag)
            .map_err(|e| CipherError::Decryption(format!("{e:?}")))?;
        if plaintext.len() != plaintext_len || plaintext.as_ptr() != allocation_start {
            return Err(CipherError::Decryption(
                "AEAD provider returned an unexpected plaintext window".to_owned(),
            ));
        }
        ciphertext_with_tag.truncate(plaintext_len);
        Ok(ciphertext_with_tag)
    }

    /// Decrypt `ciphertext_with_tag` (which MUST be exactly
    /// `plaintext_len + AES_GCM_TAG_LEN` bytes). On success returns
    /// the plaintext `Vec<u8>` (length = ciphertext_with_tag.len() -
    /// 16). On tag mismatch returns [`CipherError::Decryption`].
    pub fn decrypt(
        &self,
        iv: &[u8; AES_GCM_IV_LEN],
        aad: &[u8],
        ciphertext_with_tag: &[u8],
    ) -> Result<Vec<u8>, CipherError> {
        self.decrypt_owned(iv, aad, ciphertext_with_tag.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_key() -> [u8; AEAD_KEY_LEN] {
        let mut k = [0u8; AEAD_KEY_LEN];
        for (i, b) in k.iter_mut().enumerate() {
            *b = (i * 7 + 13) as u8;
        }
        k
    }

    fn fixture_iv() -> [u8; AES_GCM_IV_LEN] {
        let mut iv = [0u8; AES_GCM_IV_LEN];
        for (i, b) in iv.iter_mut().enumerate() {
            *b = (i + 1) as u8;
        }
        iv
    }

    #[test]
    fn constants_match_aws_lc_rs() {
        assert_eq!(AEAD_KEY_LEN, 32);
        assert_eq!(AES_GCM_IV_LEN, 12);
        assert_eq!(AES_GCM_TAG_LEN, 16);
    }

    #[test]
    fn encrypt_then_decrypt_roundtrip() {
        let key = fixture_key();
        let iv = fixture_iv();
        let aad = b"test-aad";
        let plaintext = b"hello arcgraph encryption";
        let cipher = Aes256GcmCipher::from_key(&key).expect("key init");
        let ct = cipher.encrypt(&iv, aad, plaintext).expect("encrypt");
        // Ciphertext = plaintext.len() + tag (16)
        assert_eq!(ct.len(), plaintext.len() + AES_GCM_TAG_LEN);
        let back = cipher.decrypt(&iv, aad, &ct).expect("decrypt");
        assert_eq!(back, plaintext);
    }

    #[test]
    fn decrypt_fails_on_wrong_key() {
        let key_a = fixture_key();
        let mut key_b = key_a;
        key_b[0] ^= 0xFF; // Flip a bit so it's a different key.
        let iv = fixture_iv();
        let aad = b"";
        let pt = b"secret";
        let cipher_a = Aes256GcmCipher::from_key(&key_a).unwrap();
        let cipher_b = Aes256GcmCipher::from_key(&key_b).unwrap();
        let ct = cipher_a.encrypt(&iv, aad, pt).unwrap();
        let err = cipher_b.decrypt(&iv, aad, &ct).unwrap_err();
        assert!(matches!(err, CipherError::Decryption(_)));
    }

    #[test]
    fn decrypt_fails_on_tag_corruption() {
        let key = fixture_key();
        let iv = fixture_iv();
        let aad = b"";
        let pt = b"secret";
        let cipher = Aes256GcmCipher::from_key(&key).unwrap();
        let mut ct = cipher.encrypt(&iv, aad, pt).unwrap();
        let last_idx = ct.len() - 1;
        ct[last_idx] ^= 0x01; // Flip a tag bit.
        let err = cipher.decrypt(&iv, aad, &ct).unwrap_err();
        assert!(matches!(err, CipherError::Decryption(_)));
    }

    #[test]
    fn decrypt_fails_on_ciphertext_corruption() {
        let key = fixture_key();
        let iv = fixture_iv();
        let aad = b"";
        let pt = b"secretsecretsecret";
        let cipher = Aes256GcmCipher::from_key(&key).unwrap();
        let mut ct = cipher.encrypt(&iv, aad, pt).unwrap();
        ct[2] ^= 0x80;
        let err = cipher.decrypt(&iv, aad, &ct).unwrap_err();
        assert!(matches!(err, CipherError::Decryption(_)));
    }

    #[test]
    fn decrypt_fails_on_iv_mismatch() {
        let key = fixture_key();
        let iv_a = fixture_iv();
        let mut iv_b = iv_a;
        iv_b[5] ^= 0xFF;
        let aad = b"";
        let pt = b"secret";
        let cipher = Aes256GcmCipher::from_key(&key).unwrap();
        let ct = cipher.encrypt(&iv_a, aad, pt).unwrap();
        let err = cipher.decrypt(&iv_b, aad, &ct).unwrap_err();
        assert!(matches!(err, CipherError::Decryption(_)));
    }

    #[test]
    fn decrypt_fails_on_aad_mismatch() {
        let key = fixture_key();
        let iv = fixture_iv();
        let pt = b"secret";
        let cipher = Aes256GcmCipher::from_key(&key).unwrap();
        let ct = cipher.encrypt(&iv, b"AAD-A", pt).unwrap();
        let err = cipher.decrypt(&iv, b"AAD-B", &ct).unwrap_err();
        assert!(matches!(err, CipherError::Decryption(_)));
    }

    #[test]
    fn decrypt_rejects_truncated_ciphertext() {
        let key = fixture_key();
        let cipher = Aes256GcmCipher::from_key(&key).unwrap();
        let iv = fixture_iv();
        let err = cipher.decrypt(&iv, b"", &[1, 2, 3]).unwrap_err();
        assert!(matches!(err, CipherError::Decryption(_)));
    }

    /// A 0-byte plaintext is legal — GCM admits empty messages. The
    /// resulting ciphertext is exactly the tag (16 bytes).
    #[test]
    fn empty_plaintext_roundtrips() {
        let key = fixture_key();
        let iv = fixture_iv();
        let cipher = Aes256GcmCipher::from_key(&key).unwrap();
        let ct = cipher.encrypt(&iv, b"aad", b"").unwrap();
        assert_eq!(ct.len(), AES_GCM_TAG_LEN);
        let pt = cipher.decrypt(&iv, b"aad", &ct).unwrap();
        assert!(pt.is_empty());
    }

    /// Property: two encryptions of the same plaintext under
    /// DIFFERENT IVs produce DIFFERENT ciphertexts. (Same IV under
    /// same key is a nonce-reuse fatality; AES-GCM provides no
    /// indistinguishability there. This test exercises the
    /// happy-path IV-randomization property.)
    #[test]
    fn different_ivs_produce_different_ciphertexts() {
        let key = fixture_key();
        let cipher = Aes256GcmCipher::from_key(&key).unwrap();
        let pt = b"same plaintext";
        let mut iv_a = [0u8; AES_GCM_IV_LEN];
        iv_a[0] = 1;
        let mut iv_b = [0u8; AES_GCM_IV_LEN];
        iv_b[0] = 2;
        let ct_a = cipher.encrypt(&iv_a, b"", pt).unwrap();
        let ct_b = cipher.encrypt(&iv_b, b"", pt).unwrap();
        assert_ne!(ct_a, ct_b);
    }
}
