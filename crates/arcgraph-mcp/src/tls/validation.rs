//! Pre-swap validation pipeline for the W13ε M5-02 TLS hot-reload
//! resolver.
//!
//! Validation is executed by [`super::provider::FileSystemCertProvider`]
//! BEFORE the new `Arc<CertifiedKey>` is installed in the [`super::resolver::HotReloadResolver`]
//! `ArcSwap`. The contract is "no half-swap": if any validation step
//! fails, the previous cert remains and the resolver keeps serving the
//! old key for new handshakes.
//!
//! ## Validation steps
//!
//! 1. **PEM cert chain present** — at least one CERTIFICATE block; the
//!    end-entity is at index 0 per rustls convention.
//! 2. **PEM private key present** — at least one PKCS#1 / PKCS#8 / SEC1
//!    block. `PrivateKeyDer::from_pem_reader` (via the
//!    `rustls_pki_types::pem::PemObject` trait) returns the first one
//!    found.
//! 3. **Signing key constructible** — `aws_lc_rs::sign::any_supported_type`
//!    accepts the key's algorithm + encoding.
//! 4. **Cert ↔ key match** — `CertifiedKey::keys_match` compares the
//!    private key's `SubjectPublicKeyInfo` against the end-entity cert's
//!    SPKI. (rustls 0.23.40 `crates/rustls/src/crypto/signer.rs:190`.)
//! 5. **Validity window** — end-entity cert's `notBefore <= now <= notAfter`.
//! 6. **Hostname (optional)** — when `expected_hostname` is configured,
//!    the cert's SubjectAltName must contain a matching DNSName entry.
//!    CommonName is checked as a fallback per RFC 6125 §6.4.4 — but
//!    only when SAN is absent (modern browsers / clients ignore CN if
//!    SAN exists; we mirror that behavior so a SAN-less rotation is
//!    flagged early).
//!
//! ## Why x509-parser instead of webpki
//!
//! `webpki` does full chain-to-trust-anchor validation, which is the
//! CLIENT-side concern. Server-side cert rotation cares about
//! "is this cert presentable?" — i.e., the local validity-window +
//! key-match invariants — not "does it chain to a trusted root?"
//! (the client's trust store is the authority for that). So
//! `x509-parser` is the right tool: pure local DER inspection.
//!
//! ## What this validation does NOT check
//!
//! - **Cert chain trust anchor.** Server-side resolver rotation does
//!   NOT validate the chain to a trusted root; that is the CLIENT's
//!   concern (the client's trust store is the authority). Adding
//!   chain-trust validation here would couple server config to a trust
//!   store the server doesn't manage. If the operator stages a cert
//!   that doesn't chain to client roots, clients will reject the
//!   handshake — that's the right place to fail. Future maintainers:
//!   do NOT add chain-trust validation here without an ADR; the
//!   omission is by design, not oversight.
//! - **Revocation (OCSP / CRL).** Out of scope for the v1.0 resolver
//!   surface. v1.1+ may add OCSP stapling driven by an
//!   `OcspProvider`-shaped sibling trait if customer demand emerges.
//! - **Cipher / TLS-version policy.** That lives in the
//!   `rustls::ServerConfig` builder at the transport sub-slice (W14+),
//!   not in this validation pipeline.

use std::sync::Arc;

use rustls::crypto::aws_lc_rs;
use rustls::sign::CertifiedKey;
use rustls_pki_types::{CertificateDer, PrivateKeyDer};
use x509_parser::extensions::GeneralName;
use x509_parser::oid_registry::OID_X509_COMMON_NAME;
use x509_parser::prelude::*;

use super::error::{TlsResolverError, TlsResolverResult};

/// Wall-clock provider, abstracted so tests can inject a deterministic
/// "now" without `std::time::SystemTime` flakiness on cert-validity
/// edges.
///
/// Per `feedback_avoid_speculative_scaffolding.md`, this trait IS
/// consumed in this slice: production paths use `SystemClock`, and the
/// validity-window-edge integration test (`tls_resolver.rs`) wires its
/// own local `Clock` impl through `build_certified_key`'s public
/// signature. We deliberately do NOT ship a public `FixedClock` —
/// downstream test code should write its own deterministic clock so
/// the public API doesn't leak a fragile timing primitive.
pub trait Clock: Send + Sync + std::fmt::Debug {
    /// Current time as Unix-epoch seconds (UTC).
    fn now_unix(&self) -> i64;
}

/// Production clock — wraps `std::time::SystemTime::now()`.
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_unix(&self) -> i64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        // SystemTime::now() can be earlier than UNIX_EPOCH only if the
        // host clock is set before 1970; rather than panic, clamp to 0
        // and the validity check will fail with a clear "now" value.
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| i64::try_from(d.as_secs()).unwrap_or(i64::MAX))
            .unwrap_or(0)
    }
}

/// Construct a validated `Arc<CertifiedKey>` from PEM-decoded inputs.
///
/// All validation steps from the module docstring run here; on the
/// happy path, the returned `Arc<CertifiedKey>` is safe to install in
/// the resolver's `ArcSwap`.
///
/// `expected_hostname`: when `Some`, validates SAN/CN match per RFC 6125.
/// When `None`, hostname checks are skipped (operator's contract is
/// "trust the cert as-presented", suitable for development clusters
/// or when external infra (e.g., reverse proxy) handles SNI).
///
/// `clock`: deterministic time source for the validity-window check.
pub fn build_certified_key(
    cert_chain: Vec<CertificateDer<'static>>,
    private_key: PrivateKeyDer<'static>,
    expected_hostname: Option<&str>,
    clock: &dyn Clock,
) -> TlsResolverResult<Arc<CertifiedKey>> {
    if cert_chain.is_empty() {
        // Reachable only if the caller passes an empty Vec; the
        // FileSystemCertProvider rejects empty chains earlier with
        // NoCertificatesFound. Defense-in-depth at the boundary.
        return Err(TlsResolverError::SigningKey {
            detail: "empty cert chain".into(),
        });
    }

    // Build signing key via aws-lc-rs provider.
    let signing_key = aws_lc_rs::sign::any_supported_type(&private_key).map_err(|e| {
        TlsResolverError::SigningKey {
            detail: format!("aws_lc_rs::any_supported_type rejected key: {e:?}"),
        }
    })?;

    let cert_key = Arc::new(CertifiedKey::new(cert_chain, signing_key));

    // Cross-check: private key SPKI matches end-entity cert SPKI.
    cert_key
        .keys_match()
        .map_err(|e| TlsResolverError::KeyMismatch {
            detail: format!("rustls keys_match: {e:?}"),
        })?;

    // Pull end-entity cert (index 0) for X.509 inspection.
    let end_entity = cert_key
        .end_entity_cert()
        .map_err(|e| TlsResolverError::SigningKey {
            detail: format!("rustls end_entity_cert: {e:?}"),
        })?;

    // Borrow the DER bytes for the lifetime of this function — the
    // x509-parser X509Certificate borrows from this slice. We don't
    // store any X509Certificate references beyond this scope.
    let der: &[u8] = end_entity.as_ref();
    let (_remainder, parsed) =
        parse_x509_certificate(der).map_err(|e| TlsResolverError::X509Parse {
            detail: format!("{e:?}"),
        })?;

    // Validity window check.
    let now = clock.now_unix();
    let not_before = parsed.validity().not_before.timestamp();
    let not_after = parsed.validity().not_after.timestamp();
    if now < not_before {
        return Err(TlsResolverError::ValidityWindow {
            phase: "not_before",
            cert_unix: not_before,
            at_unix: now,
        });
    }
    if now > not_after {
        return Err(TlsResolverError::ValidityWindow {
            phase: "not_after",
            cert_unix: not_after,
            at_unix: now,
        });
    }

    // Hostname check (optional).
    if let Some(expected) = expected_hostname {
        verify_hostname(&parsed, expected)?;
    }

    Ok(cert_key)
}

/// Hostname verification per RFC 6125 §6.4.4.
///
/// Behavior:
///   1. If SAN is present, ALL DNSName entries are scanned for an
///      exact-or-wildcard match. CN is ignored (modern clients do).
///   2. If SAN is absent, fall back to CommonName from the Subject DN.
///
/// Wildcard matching is left-most-label only: `*.example.com` matches
/// `foo.example.com` but NOT `foo.bar.example.com` (per RFC 6125
/// §6.4.3 rule 1).
fn verify_hostname(parsed: &X509Certificate<'_>, expected: &str) -> TlsResolverResult<()> {
    let san_ext = parsed
        .subject_alternative_name()
        .map_err(|e| TlsResolverError::X509Parse {
            detail: format!("SAN extension parse: {e:?}"),
        })?;

    if let Some(san) = san_ext {
        for name in &san.value.general_names {
            if let GeneralName::DNSName(dns) = name {
                if hostname_matches(dns, expected) {
                    return Ok(());
                }
            }
        }
        // SAN present but no DNSName matched — RFC 6125 §6.4.4 says
        // CN must NOT be used as a fallback when SAN exists.
        return Err(TlsResolverError::HostnameMismatch {
            expected: expected.to_string(),
        });
    }

    // SAN absent — fall back to CN.
    for cn_attr in parsed.subject().iter_attributes() {
        if cn_attr.attr_type() == &OID_X509_COMMON_NAME {
            if let Ok(cn) = cn_attr.as_str() {
                if hostname_matches(cn, expected) {
                    return Ok(());
                }
            }
        }
    }

    Err(TlsResolverError::HostnameMismatch {
        expected: expected.to_string(),
    })
}

/// Match a cert SAN/CN entry against an expected hostname.
///
/// Supports left-most wildcard (`*.example.com`) per RFC 6125 §6.4.3
/// rule 1: a single `*` in the leftmost label matches a single label,
/// not multiple. Returns false on any structural mismatch.
fn hostname_matches(pattern: &str, expected: &str) -> bool {
    if pattern.eq_ignore_ascii_case(expected) {
        return true;
    }
    // Wildcard handling — only leftmost label `*.rest` and only when
    // `expected` has the same number of labels.
    if let Some(rest) = pattern.strip_prefix("*.") {
        // `expected` must split into (label, rest) where rest matches.
        if let Some((_first, expected_rest)) = expected.split_once('.') {
            return rest.eq_ignore_ascii_case(expected_rest);
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hostname_matches_exact_case_insensitive() {
        assert!(hostname_matches("api.example.com", "api.example.com"));
        assert!(hostname_matches("API.EXAMPLE.COM", "api.example.com"));
        assert!(!hostname_matches("api.example.com", "other.example.com"));
    }

    #[test]
    fn hostname_matches_wildcard_one_label() {
        assert!(hostname_matches("*.example.com", "api.example.com"));
        // Multi-label wildcard rejection: RFC 6125 §6.4.3 rule 1.
        assert!(!hostname_matches("*.example.com", "deep.api.example.com"));
        // Wildcard does NOT match the apex domain.
        assert!(!hostname_matches("*.example.com", "example.com"));
    }

    #[test]
    fn hostname_matches_rejects_internal_wildcard() {
        // `foo.*.example.com` — internal wildcard is non-conformant
        // and our matcher rejects it. RFC 6125 §6.4.3 rule 2 forbids
        // wildcards anywhere except the leftmost label.
        assert!(!hostname_matches(
            "foo.*.example.com",
            "foo.api.example.com"
        ));
    }
}
