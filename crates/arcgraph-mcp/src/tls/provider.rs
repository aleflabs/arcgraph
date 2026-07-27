//! Cert-source abstraction for the W13ε M5-02 TLS hot-reload resolver.
//!
//! The [`CertProvider`] trait is the seam between "where do certs come
//! from?" and "how do we install them?". Production paths use
//! [`FileSystemCertProvider`] which reads PEM files staged on disk;
//! v1.1+ may add `AcmeCertProvider` (Let's Encrypt) or
//! `VaultCertProvider` (HashiCorp Vault) without touching the
//! [`super::resolver::HotReloadResolver`] surface.
//!
//! ## Trait-shape pre-decision (per `feedback_avoid_speculative_scaffolding.md` §"Trait-shape sub-rule")
//!
//! This trait IS shipped with a single in-slice consumer
//! ([`FileSystemCertProvider`]) which is the borderline case explicitly
//! endorsed by the W13ε spawn prompt:
//!
//! > ACME forward-pin trait is documented but has zero impls; ship the
//! > trait when CONSUMED per `feedback_avoid_speculative_scaffolding.md`
//! > — exception here because the trait is the natural boundary for
//! > v1.1 ACME work, and the stub impl IS the FileSystemCertProvider
//! > (the trait has 1 real consumer — borderline; document the
//! > rationale inline).
//!
//! The borrow-check sketch was performed: the trait method returns
//! `Result<Arc<CertifiedKey>, TlsResolverError>` (no lifetime params,
//! owned), which is the canonical shape for all 4 plausible production
//! producers (file-system, ACME, Vault, k8s cert-manager Secret-watch).
//! All 4 hand back fully-loaded `Arc<CertifiedKey>` after their own
//! validation; none need to borrow anything from the resolver. So the
//! trait shape is production-validated, not fixture-validated.

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rustls::sign::CertifiedKey;
use rustls_pki_types::pem::{Error as PemError, PemObject};
use rustls_pki_types::{CertificateDer, PrivateKeyDer};

use super::error::{TlsResolverError, TlsResolverResult};
use super::validation::{Clock, SystemClock, build_certified_key};

/// Source of cert+key material for the resolver.
///
/// `Send + Sync` so the resolver's `Arc<dyn CertProvider>` field can be
/// shared across the SIGHUP reload task and any future `tokio::spawn`'d
/// reload triggers.
pub trait CertProvider: Send + Sync + std::fmt::Debug {
    /// Load + validate the current cert+key pair.
    ///
    /// Implementations MUST run all validation steps (signing-key
    /// construction, key/cert match, validity window, hostname when
    /// configured) before returning Ok — the resolver assumes the
    /// returned `Arc<CertifiedKey>` is presentable and installs it
    /// directly into its `ArcSwap`.
    fn load(&self) -> TlsResolverResult<Arc<CertifiedKey>>;

    /// Human-readable source descriptor for tracing + diagnostics.
    /// Examples: `"file:/etc/arcgraph/server.crt+server.key"` or
    /// `"acme:example.com"`. Used in `tls.reload.{success,failed}`
    /// log lines.
    fn source_descriptor(&self) -> String;
}

/// File-system-backed cert provider — v1.0 default.
///
/// Reads PEM-encoded cert chain + private key from configured paths
/// on every `load()` call. Validation runs through the
/// [`super::validation::build_certified_key`] pipeline.
#[derive(Debug)]
pub struct FileSystemCertProvider {
    cert_path: PathBuf,
    key_path: PathBuf,
    expected_hostname: Option<String>,
    clock: Arc<dyn Clock>,
}

impl FileSystemCertProvider {
    /// Construct a new provider rooted at `cert_path` + `key_path`.
    ///
    /// `expected_hostname` enables RFC 6125 SAN/CN matching during
    /// reload validation; `None` skips hostname verification (suitable
    /// when SNI / hostname checks are handled by an upstream proxy or
    /// the cluster is single-tenant single-hostname).
    pub fn new(
        cert_path: impl Into<PathBuf>,
        key_path: impl Into<PathBuf>,
        expected_hostname: Option<String>,
    ) -> Self {
        Self {
            cert_path: cert_path.into(),
            key_path: key_path.into(),
            expected_hostname,
            clock: Arc::new(SystemClock),
        }
    }

    /// Test-hook constructor injecting a deterministic clock.
    ///
    /// Production paths use `new(...)` which wires `SystemClock`; this
    /// constructor exists for the `validity_window_*` test family in
    /// the integration suite where reproducible expiry-edge tests need
    /// pinned timestamps.
    #[cfg(test)]
    pub fn new_with_clock(
        cert_path: impl Into<PathBuf>,
        key_path: impl Into<PathBuf>,
        expected_hostname: Option<String>,
        clock: Arc<dyn Clock>,
    ) -> Self {
        Self {
            cert_path: cert_path.into(),
            key_path: key_path.into(),
            expected_hostname,
            clock,
        }
    }

    /// Load and validate without owning any state — useful for
    /// pre-flight validation before installing in a resolver.
    pub fn load_validated(&self) -> TlsResolverResult<Arc<CertifiedKey>> {
        let cert_chain = read_cert_chain(&self.cert_path)?;
        let private_key = read_private_key(&self.key_path)?;
        build_certified_key(
            cert_chain,
            private_key,
            self.expected_hostname.as_deref(),
            &*self.clock,
        )
    }
}

impl CertProvider for FileSystemCertProvider {
    fn load(&self) -> TlsResolverResult<Arc<CertifiedKey>> {
        self.load_validated()
    }

    fn source_descriptor(&self) -> String {
        format!(
            "file:{}+{}",
            self.cert_path.display(),
            self.key_path.display()
        )
    }
}

fn read_cert_chain(path: &Path) -> TlsResolverResult<Vec<CertificateDer<'static>>> {
    let file = File::open(path).map_err(|source| TlsResolverError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let reader = BufReader::new(file);
    let mut chain = Vec::new();
    for cert_result in CertificateDer::pem_reader_iter(reader) {
        let cert = cert_result.map_err(|e| translate_cert_pem_error(path, e))?;
        chain.push(cert);
    }
    if chain.is_empty() {
        return Err(TlsResolverError::NoCertificatesFound {
            path: path.to_path_buf(),
        });
    }
    Ok(chain)
}

fn read_private_key(path: &Path) -> TlsResolverResult<PrivateKeyDer<'static>> {
    let file = File::open(path).map_err(|source| TlsResolverError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let reader = BufReader::new(file);
    PrivateKeyDer::from_pem_reader(reader).map_err(|e| match e {
        PemError::NoItemsFound => TlsResolverError::NoPrivateKeyFound {
            path: path.to_path_buf(),
        },
        other => TlsResolverError::MalformedPem {
            path: path.to_path_buf(),
            detail: format!("private key parse: {other}"),
        },
    })
}

/// Translate a `rustls_pki_types::pem::Error` from cert parsing into
/// the resolver's local taxonomy. `NoItemsFound` from a single cert in
/// the chain is a structural error inside the iterator (only emitted
/// when no further sections decode); `read_cert_chain` short-circuits
/// on the first `Err`, so an empty chain (no CERTIFICATE blocks at all)
/// surfaces via the explicit `chain.is_empty()` check above as
/// `NoCertificatesFound`. Other variants (parse errors, base64 errors,
/// I/O errors mid-stream) all collapse to `MalformedPem`.
fn translate_cert_pem_error(path: &Path, err: PemError) -> TlsResolverError {
    TlsResolverError::MalformedPem {
        path: path.to_path_buf(),
        detail: format!("cert parse: {err}"),
    }
}
