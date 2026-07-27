//! W13ε M5-02 — `HotReloadResolver`.
//!
//! `HotReloadResolver` implements `rustls::server::ResolvesServerCert`
//! by holding the current `Arc<CertifiedKey>` in an `ArcSwap`. The TLS
//! handshake's `resolve()` callback hits `ArcSwap::load_full()` once per
//! accept; the reload path hits `ArcSwap::store()`. Per arc-swap docs
//! the read path is wait-free and the rotation is atomic — no observer
//! ever sees a half-rotated state.
//!
//! ## Latency budget
//!
//! - `resolve()` overhead: 1 atomic-load + 1 Arc-clone ≈ 5-15 ns on
//!   modern x86_64; the cost is dominated by the RustTLS handshake
//!   (10-100 µs depending on key type), so the resolver overhead is
//!   noise (< 0.1% of total handshake cost).
//! - `reload()` is on the slow path (SIGHUP-driven, expected ≤1/hour
//!   in normal production rotation cadence); validation cost
//!   (10-50 ms for cert chain parse +
//!   key match) is bounded by file size and crypto parser speed.
//!
//! ## "Keep old on failure" invariant
//!
//! If `reload()` fails (PEM parse, validation, key mismatch), the
//! `ArcSwap` is NOT updated and the resolver continues serving the
//! previous cert. Callers (the SIGHUP loop in [`super::reload`]) log
//! the failure but keep the listener up.

use std::sync::Arc;

use arc_swap::ArcSwap;
use rustls::server::{ClientHello, ResolvesServerCert};
use rustls::sign::CertifiedKey;

use super::error::TlsResolverResult;
use super::provider::CertProvider;

/// Hot-reloading server-cert resolver.
///
/// Construct via [`HotReloadResolver::new`] — initial load is mandatory
/// so the resolver always has a valid cert to present from the moment
/// it's installed in `rustls::ServerConfig`.
pub struct HotReloadResolver {
    current: ArcSwap<CertifiedKey>,
    provider: Arc<dyn CertProvider>,
}

impl std::fmt::Debug for HotReloadResolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HotReloadResolver")
            .field("provider", &self.provider.source_descriptor())
            .finish()
    }
}

impl HotReloadResolver {
    /// Build a resolver from a cert provider, performing the initial
    /// load synchronously. Returns an error if the initial load fails
    /// — the caller decides whether to abort startup or retry.
    pub fn new(provider: Arc<dyn CertProvider>) -> TlsResolverResult<Arc<Self>> {
        let initial = provider.load()?;
        Ok(Arc::new(Self {
            current: ArcSwap::from(initial),
            provider,
        }))
    }

    /// Trigger a reload from the underlying provider.
    ///
    /// On success, the new `Arc<CertifiedKey>` replaces the previous
    /// one in the `ArcSwap`. On failure, the previous cert is
    /// preserved (per the "keep old on failure" invariant) and the
    /// caller receives the error for logging / metric emission.
    pub fn reload(&self) -> TlsResolverResult<()> {
        let next = self.provider.load()?;
        self.current.store(next);
        Ok(())
    }

    /// Snapshot the currently-installed cert. Each call returns a
    /// fresh `Arc<CertifiedKey>` clone (not a `Guard`) so the cert
    /// can outlive the resolver's lifetime if needed (e.g., used in
    /// an in-flight handshake whose state machine clones the Arc into
    /// per-connection state).
    #[must_use]
    pub fn current(&self) -> Arc<CertifiedKey> {
        self.current.load_full()
    }

    /// Source descriptor passed-through from the provider — useful in
    /// tracing-event emission so log consumers can correlate reload
    /// events with the cert source (file path, ACME account, etc.).
    #[must_use]
    pub fn source_descriptor(&self) -> String {
        self.provider.source_descriptor()
    }
}

impl ResolvesServerCert for HotReloadResolver {
    fn resolve(&self, _client_hello: ClientHello<'_>) -> Option<Arc<CertifiedKey>> {
        Some(self.current.load_full())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::tls::error::TlsResolverError;
    use crate::tls::provider::CertProvider;

    /// In-memory provider for unit tests. Keeps a script of `Result`s
    /// to return on consecutive `load()` calls so the test can drive
    /// success / failure transitions without filesystem I/O.
    #[derive(Debug)]
    struct ScriptedProvider {
        script: Mutex<Vec<Result<Arc<CertifiedKey>, TlsResolverError>>>,
    }

    impl ScriptedProvider {
        fn new(script: Vec<Result<Arc<CertifiedKey>, TlsResolverError>>) -> Self {
            Self {
                script: Mutex::new(script),
            }
        }
    }

    impl CertProvider for ScriptedProvider {
        fn load(&self) -> Result<Arc<CertifiedKey>, TlsResolverError> {
            // Pop from front. Test must script enough entries for the
            // call sequence; running out is a test bug, not a runtime
            // condition the resolver should tolerate.
            let mut s = self.script.lock().expect("scripted provider mutex");
            if s.is_empty() {
                panic!(
                    "ScriptedProvider exhausted: test scripted too few \
                     load() responses for the resolver call sequence"
                );
            }
            s.remove(0)
        }

        fn source_descriptor(&self) -> String {
            "scripted://test".into()
        }
    }

    /// Synthesize a `CertifiedKey` directly via rcgen for unit tests
    /// that need a real key but don't want filesystem I/O. Each call
    /// produces a fresh keypair so we can distinguish "before" from
    /// "after" rotation by Arc-pointer equality (different
    /// `CertifiedKey`'s wrap different `signing_key`'s).
    fn synth_certified_key() -> Arc<CertifiedKey> {
        use rcgen::{CertifiedKey as RcgenCert, generate_simple_self_signed};
        use rustls_pki_types::PrivateKeyDer;
        use rustls_pki_types::pem::PemObject;
        let RcgenCert { cert, signing_key } =
            generate_simple_self_signed(vec!["localhost".to_string()]).expect("rcgen self-signed");
        let cert_der = cert.der().clone();
        // Pivot through PEM → `PrivateKeyDer::from_pem_slice` to get an
        // owned `PrivateKeyDer<'static>`; rcgen's `KeyPair::serialize_der`
        // returns SEC1/PKCS#8 raw bytes but typing the static lifetime
        // is easier via the PEM round-trip.
        let key_pem = signing_key.serialize_pem();
        let key_der =
            PrivateKeyDer::from_pem_slice(key_pem.as_bytes()).expect("PEM private key parse");
        let signing_key =
            rustls::crypto::aws_lc_rs::sign::any_supported_type(&key_der).expect("aws_lc_rs sign");
        Arc::new(CertifiedKey::new(vec![cert_der], signing_key))
    }

    #[test]
    fn new_returns_initial_cert_via_resolver_callback() {
        let initial = synth_certified_key();
        // Scripted entries: HotReloadResolver::new + 1 current() →
        // only the new() call hits the provider.
        let provider = Arc::new(ScriptedProvider::new(vec![Ok(initial.clone())]));
        let resolver = HotReloadResolver::new(provider).expect("initial load");
        let snapshot = resolver.current();
        assert!(Arc::ptr_eq(&snapshot, &initial));
    }

    #[test]
    fn reload_swaps_cert_atomically() {
        let initial = synth_certified_key();
        let next = synth_certified_key();
        let provider = Arc::new(ScriptedProvider::new(vec![
            Ok(initial.clone()),
            Ok(next.clone()),
        ]));
        let resolver = HotReloadResolver::new(provider).expect("initial load");
        assert!(Arc::ptr_eq(&resolver.current(), &initial));
        resolver.reload().expect("reload success");
        let after = resolver.current();
        assert!(Arc::ptr_eq(&after, &next));
        // Pointer-equality on `next` proves the swap installed the
        // exact `Arc` the provider returned (not a clone of `initial`).
        assert!(!Arc::ptr_eq(&after, &initial));
    }

    #[test]
    fn reload_keeps_old_cert_when_provider_fails() {
        let initial = synth_certified_key();
        let provider = Arc::new(ScriptedProvider::new(vec![
            Ok(initial.clone()),
            Err(TlsResolverError::KeyMismatch {
                detail: "test induced".into(),
            }),
        ]));
        let resolver = HotReloadResolver::new(provider).expect("initial load");

        let err = resolver.reload().expect_err("provider returns Err on 2nd");
        assert!(
            matches!(err, TlsResolverError::KeyMismatch { .. }),
            "expected KeyMismatch, got {err:?}"
        );
        // Critical: previous cert is still installed.
        let after = resolver.current();
        assert!(
            Arc::ptr_eq(&after, &initial),
            "resolver swapped despite reload error"
        );
    }

    #[test]
    fn resolve_callback_returns_current_cert() {
        // ResolvesServerCert::resolve takes a ClientHello; rustls
        // doesn't expose a public ClientHello constructor, so we
        // exercise the equivalent path via the public `current()`
        // accessor — both paths bottom out on the same ArcSwap load.
        let initial = synth_certified_key();
        let provider = Arc::new(ScriptedProvider::new(vec![Ok(initial.clone())]));
        let resolver = HotReloadResolver::new(provider).expect("initial load");
        let cert = resolver.current();
        assert_eq!(cert.cert.len(), 1);
    }

    #[test]
    fn source_descriptor_pass_through() {
        let initial = synth_certified_key();
        let provider = Arc::new(ScriptedProvider::new(vec![Ok(initial)]));
        let resolver = HotReloadResolver::new(provider).expect("initial load");
        assert_eq!(resolver.source_descriptor(), "scripted://test");
    }

    #[test]
    fn initial_load_failure_is_surfaced() {
        let provider = Arc::new(ScriptedProvider::new(vec![Err(
            TlsResolverError::NoCertificatesFound {
                path: std::path::PathBuf::from("/dev/null"),
            },
        )]));
        let res = HotReloadResolver::new(provider);
        assert!(matches!(
            res,
            Err(TlsResolverError::NoCertificatesFound { .. })
        ));
    }
}
