//! Error taxonomy for the W13ε M5-02 TLS hot-reload resolver.
//!
//! `TlsResolverError` is `#[non_exhaustive]`. All validation paths in
//! [`super::validation`]
//! and the SIGHUP loop in [`super::reload`] map to this taxonomy at the
//! public boundary; internal helpers propagate concrete typed errors
//! (e.g., `std::io::Error`, `rustls::Error`) and translate at the
//! module surface.

use std::io;
use std::path::PathBuf;

use thiserror::Error;

/// All failure modes emitted by the TLS hot-reload resolver.
///
/// Variants are ordered by where they fire in the load → validate →
/// swap pipeline so callers building diagnostic UIs can group by phase
/// (filesystem → parse → validate → install).
///
/// `#[non_exhaustive]`: future ACME / Vault / cert-manager backend
/// landings (v1.1+) may add backend-specific failure variants without
/// breaking the SemVer contract.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TlsResolverError {
    /// File-system error reading a cert or key file. The `path` field
    /// distinguishes which side of the pair failed so operators can
    /// hand the error directly to a runbook step ("the cert side
    /// can't be read; check `[mcp_http_tls].cert_path`").
    #[error("io error reading {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    /// PEM parsing failed. The provided file was readable but didn't
    /// contain valid PEM blocks (e.g., truncated, corrupted, or a
    /// non-PEM format like raw DER without armor).
    #[error("malformed PEM in {path}: {detail}")]
    MalformedPem { path: PathBuf, detail: String },

    /// PEM was valid but contained zero certificate blocks. A cert
    /// file MUST contain at least one CERTIFICATE block (the
    /// end-entity); intermediates may follow.
    #[error("no certificates found in {path}")]
    NoCertificatesFound { path: PathBuf },

    /// PEM was valid but contained zero private-key blocks of a
    /// supported type (PKCS#1 RSA, PKCS#8, SEC1 EC).
    #[error("no private key found in {path}")]
    NoPrivateKeyFound { path: PathBuf },

    /// The candidate private key was not in a format `aws_lc_rs`
    /// recognizes — e.g., an unknown algorithm OID, a malformed
    /// PKCS#8 encoding, or a cipher rustls' provider does not support.
    #[error("unsupported or malformed private key in {path}: {detail}")]
    UnsupportedKey { path: PathBuf, detail: String },

    /// The end-entity X.509 cert at index 0 of the chain failed
    /// `x509_parser` decode. Indicates a structurally invalid DER
    /// blob even though the PEM armor was correct.
    #[error("x509 parse failed for end-entity cert: {detail}")]
    X509Parse { detail: String },

    /// The cert's `notBefore` is in the future (we'd be presenting a
    /// not-yet-valid cert) or `notAfter` is in the past (expired).
    /// `phase` is `"not_before"` or `"not_after"` and `at_unix` is
    /// the wall-clock unix timestamp the validation observed.
    #[error("cert validity window violated ({phase}): cert={cert_unix}, now={at_unix}")]
    ValidityWindow {
        phase: &'static str,
        cert_unix: i64,
        at_unix: i64,
    },

    /// The configured `expected_hostname` was not present in the
    /// cert's SubjectAltName extension and did not match the
    /// CommonName fallback.
    #[error("expected hostname {expected:?} not found in cert SAN/CN")]
    HostnameMismatch { expected: String },

    /// Private key does not match the public key in the end-entity
    /// cert. Wraps rustls' `keys_match` verdict with a stable error
    /// shape so operators can branch without inspecting the rustls
    /// error string.
    #[error("private key does not match cert public key: {detail}")]
    KeyMismatch { detail: String },

    /// rustls rejected the candidate signing key during `CertifiedKey`
    /// construction. Catch-all for failures before our explicit
    /// `keys_match` invocation.
    #[error("rustls signing-key construction failed: {detail}")]
    SigningKey { detail: String },

    /// W20β-1 — rustls rejected the candidate client-CA trust store or
    /// the `WebPkiClientVerifier::builder` chain. Surfaces from
    /// [`super::client_verifier::client_verifier_from_ca_pem`] when the
    /// admissible-DER trust anchor list cannot be assembled (e.g., a
    /// cert with a bad key-usage extension). Distinct from
    /// [`Self::MalformedPem`] which fires for PEM-decode failures
    /// BEFORE the trust store is touched.
    #[error("client-cert verifier build failed: {detail}")]
    Validation { detail: String },
}

/// Convenience type alias for the resolver-internal pipeline.
pub type TlsResolverResult<T> = Result<T, TlsResolverError>;
