//! Shared cert/key fixture helpers for the W13ε M5-02 TLS test suite.
//!
//! All tests synthesize self-signed certs at runtime via `rcgen` rather
//! than committing PEM blobs to the repo: cert blobs in-tree have a
//! way of expiring before anyone notices, and per-run synthesis means
//! the integration tests exercise validity-window boundaries against
//! a known-fresh wall clock.

#![allow(dead_code)] // each test binary uses a different subset.

use std::path::PathBuf;

use rcgen::{CertificateParams, DnType, KeyPair};
use tempfile::TempDir;
use time::OffsetDateTime;

/// Materialized cert pair on disk, retained as long as `dir` is in
/// scope (TempDir cleans up on drop).
pub struct CertFixture {
    pub dir: TempDir,
    pub cert_path: PathBuf,
    pub key_path: PathBuf,
}

impl CertFixture {
    /// Default fixture: SAN = `["localhost"]`, validity ~ now ± 1 year.
    pub fn fresh_localhost() -> Self {
        Self::new_with_san_and_validity(
            &["localhost".to_string()],
            OffsetDateTime::now_utc() - time::Duration::days(1),
            OffsetDateTime::now_utc() + time::Duration::days(365),
            None,
        )
    }

    /// Fixture with a specific SAN list, validity window, and optional
    /// CommonName override (when CN is `Some`, the cert sets a
    /// CommonName attribute on the Subject DN).
    pub fn new_with_san_and_validity(
        sans: &[String],
        not_before: OffsetDateTime,
        not_after: OffsetDateTime,
        common_name: Option<&str>,
    ) -> Self {
        let mut params = CertificateParams::new(sans.to_vec()).expect("rcgen params construction");
        params.not_before = not_before;
        params.not_after = not_after;
        if let Some(cn) = common_name {
            params.distinguished_name.push(DnType::CommonName, cn);
        }
        let signing_key = KeyPair::generate().expect("rcgen keypair");
        let cert = params
            .self_signed(&signing_key)
            .expect("self-signed cert build");

        let dir = tempfile::tempdir().expect("tempdir");
        let cert_path = dir.path().join("server.crt");
        let key_path = dir.path().join("server.key");
        std::fs::write(&cert_path, cert.pem()).expect("write cert");
        std::fs::write(&key_path, signing_key.serialize_pem()).expect("write key");

        Self {
            dir,
            cert_path,
            key_path,
        }
    }

    /// Generate a fresh fixture but with a CN-only subject (no SAN).
    /// Used to test the SAN-absent → CN-fallback hostname path.
    pub fn cn_only(cn: &str) -> Self {
        Self::new_with_san_and_validity(
            &[],
            OffsetDateTime::now_utc() - time::Duration::days(1),
            OffsetDateTime::now_utc() + time::Duration::days(365),
            Some(cn),
        )
    }

    /// Overwrite both files with new cert/key content (used to drive
    /// the rotation test — the resolver's reload() pulls these new
    /// bytes).
    pub fn rotate_with_san(&self, sans: &[String]) {
        let mut params = CertificateParams::new(sans.to_vec()).expect("rcgen params construction");
        params.not_before = OffsetDateTime::now_utc() - time::Duration::days(1);
        params.not_after = OffsetDateTime::now_utc() + time::Duration::days(365);
        let signing_key = KeyPair::generate().expect("rcgen keypair");
        let cert = params
            .self_signed(&signing_key)
            .expect("self-signed cert build");
        std::fs::write(&self.cert_path, cert.pem()).expect("write rotated cert");
        std::fs::write(&self.key_path, signing_key.serialize_pem()).expect("write rotated key");
    }
}

/// Stash a malformed PEM blob at the given path — used to test
/// "reload keeps the old cert when the new files are corrupted".
pub fn write_malformed_pem(path: &std::path::Path) {
    std::fs::write(
        path,
        b"-----BEGIN CERTIFICATE-----\nNOT_BASE64@@@\n-----END CERTIFICATE-----\n",
    )
    .expect("write malformed");
}
