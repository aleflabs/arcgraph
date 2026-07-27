//! W16β M5-03 — Property tests on the OAuth PKCE `code_verifier`
//! flow per RFC 7636 §4.1 + §4.2 (ADR-044).
//!
//! Coverage:
//!   1. **Length boundary** — any random `len ∈ [43, 128]` yields a
//!      `code_verifier` of EXACTLY that length and a valid character
//!      set.
//!   2. **Character-set conformance** — every generated verifier
//!      passes [`validate_code_verifier`] and contains only chars
//!      from the RFC 7636 §4.1 unreserved set
//!      (`[A-Za-z0-9-._~]`).
//!   3. **Out-of-range length rejection** — `len < 43` and `len > 128`
//!      both panic when fed to [`code_verifier_with_len`].
//!   4. **S256 transform stability** — `code_challenge_s256` is
//!      deterministic: the same verifier always yields the same
//!      challenge.
//!   5. **S256 transform discrimination** — two different verifiers
//!      almost always yield different challenges (SHA-256 collision
//!      space is 2^256 so any practical inputs differ).
//!   6. **`validate_code_verifier` reverse path** — any non-empty
//!      string outside the unreserved set rejects.
//!   7. **`CodeVerifier::from_string` roundtrips a generated
//!      verifier** — `code_verifier_new() → as_str().to_string() →
//!      CodeVerifier::from_string` yields the same byte content.

use arcgraph_mcp::auth::oauth_pkce::{
    CODE_VERIFIER_MAX_LEN, CODE_VERIFIER_MIN_LEN, CodeVerifier, code_challenge_s256,
    code_verifier_new, code_verifier_with_len, validate_code_verifier,
};
use proptest::prelude::*;

/// The RFC 7636 §4.1 unreserved set (`ALPHA / DIGIT / "-" / "." /
/// "_" / "~"`). Mirrors the private constant in the production
/// module; duplicated here so the proptest doesn't depend on a
/// public-API leak.
const UNRESERVED_BYTES: &[u8; 66] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~";

/// ASCII bytes in 0x20..=0x7E that are NOT in the RFC 7636 §4.1
/// unreserved set — i.e. printable, non-control, but illegal in a
/// `code_verifier`. Drawing the "bad" byte directly from this set
/// (instead of `prop_assume!`-filtering 0..128) keeps the test
/// deterministic and reject-free (proptest's global-reject ceiling
/// otherwise aborts the run — see #711).
const NON_UNRESERVED_PRINTABLE: &[u8] = b" !\"#$%&'()*+,/:;<=>?@[\\]^`{|}";

proptest! {
    /// Every length in the legal range produces a verifier of EXACTLY
    /// that length AND a chars-in-unreserved-set body.
    #[test]
    fn any_legal_len_produces_valid_verifier(len in CODE_VERIFIER_MIN_LEN..=CODE_VERIFIER_MAX_LEN) {
        let v = code_verifier_with_len(len);
        prop_assert_eq!(v.as_str().len(), len);
        prop_assert!(validate_code_verifier(v.as_str()).is_ok());
        for b in v.as_str().bytes() {
            prop_assert!(UNRESERVED_BYTES.contains(&b),
                "byte 0x{b:02x} not in RFC 7636 §4.1 unreserved set");
        }
    }

    /// Lengths below 43 panic per the assertion in
    /// [`code_verifier_with_len`].
    #[test]
    fn short_len_panics(len in 0_usize..CODE_VERIFIER_MIN_LEN) {
        let result = std::panic::catch_unwind(|| {
            let _ = code_verifier_with_len(len);
        });
        prop_assert!(result.is_err());
    }

    /// Lengths above 128 panic per the assertion in
    /// [`code_verifier_with_len`].
    #[test]
    fn long_len_panics(len in (CODE_VERIFIER_MAX_LEN + 1)..=(CODE_VERIFIER_MAX_LEN + 256)) {
        let result = std::panic::catch_unwind(|| {
            let _ = code_verifier_with_len(len);
        });
        prop_assert!(result.is_err());
    }

    /// The S256 transform is deterministic: applying it twice to the
    /// SAME [`CodeVerifier`] yields the same challenge.
    #[test]
    fn s256_is_deterministic(seed in any::<u64>()) {
        // Use the seed to make the test deterministic across runs;
        // we don't actually use rng here — we generate via the
        // production function then verify determinism by computing
        // twice.
        let _ = seed; // silence unused-var lint
        let v = code_verifier_new();
        let c1 = code_challenge_s256(&v);
        let c2 = code_challenge_s256(&v);
        prop_assert_eq!(c1, c2);
    }

    /// Two distinct verifiers (almost certainly) yield distinct
    /// challenges. SHA-256 collision space is 2^256; the proptest
    /// runner will surface ANY collision as a counterexample
    /// (which would indicate a serious bug).
    #[test]
    fn s256_discriminates(_seed in any::<u64>()) {
        let v1 = code_verifier_new();
        let v2 = code_verifier_new();
        // Skip the (vanishingly improbable) case that two random
        // verifiers collide on bytes.
        if v1.as_str() != v2.as_str() {
            let c1 = code_challenge_s256(&v1);
            let c2 = code_challenge_s256(&v2);
            prop_assert_ne!(c1, c2);
        }
    }

    /// Verifiers of legal length but containing at least one
    /// non-unreserved char reject in [`validate_code_verifier`].
    #[test]
    fn invalid_char_rejects(
        prefix_len in CODE_VERIFIER_MIN_LEN..CODE_VERIFIER_MAX_LEN,
        bad_idx in 0_usize..NON_UNRESERVED_PRINTABLE.len(),
    ) {
        // Draw the "bad" byte directly from the in-range invalid set
        // so every sample is illegal-by-construction — no
        // `prop_assume!`-filtering, hence no rejects (the prior
        // draw-then-filter left only 29 of 128 draws surviving, so
        // proptest hit its global-reject ceiling and aborted — #711).
        let bad_byte = NON_UNRESERVED_PRINTABLE[bad_idx];
        let mut s: String = "A".repeat(prefix_len);
        s.push(bad_byte as char);
        prop_assert!(
            validate_code_verifier(&s).is_err(),
            "verifier with non-unreserved byte 0x{bad_byte:02x} must reject"
        );
    }

    /// `CodeVerifier::from_string` roundtrips a generated verifier
    /// — bytes are preserved, validation passes, and S256 transform
    /// matches.
    #[test]
    fn from_string_roundtrip(_seed in any::<u64>()) {
        let v = code_verifier_new();
        let wire = v.as_str().to_string();
        let reconstructed = CodeVerifier::from_string(wire.clone()).expect("valid verifier");
        prop_assert_eq!(reconstructed.as_str(), wire.as_str());
        prop_assert_eq!(code_challenge_s256(&reconstructed), code_challenge_s256(&v));
    }

    /// `CodeVerifier::from_string` rejects out-of-range or
    /// non-unreserved input (closes the door on a hostile AS that
    /// might receive a malformed `code_verifier` over the wire).
    #[test]
    fn from_string_rejects_invalid(
        choice in 0_u8..3_u8,
    ) {
        let s = match choice {
            0 => "A".repeat(CODE_VERIFIER_MIN_LEN - 1),
            1 => "A".repeat(CODE_VERIFIER_MAX_LEN + 1),
            _ => "A".repeat(CODE_VERIFIER_MIN_LEN - 1) + "=",
        };
        prop_assert!(CodeVerifier::from_string(s).is_err());
    }
}
