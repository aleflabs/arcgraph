//! Versioned [`KeyRing`] — fronts a `SecretsProvider` with a cache
//! of `KeyVersion → Aes256GcmCipher`.
//!
//! Designed for the WAL writer thread + page-store decorator's
//! hot path: every encryption op needs the **current** cipher; every
//! decryption op needs the cipher for the **version stamped in the
//! payload**. Without caching, each op would round-trip through the
//! keyring backend (D-Bus / Keychain / env). The cache turns that
//! into a single mutex acquisition per op (post-first-use).
//!
//! ## Concurrency
//!
//! `KeyRing` holds an inner `Mutex<HashMap<KeyVersion, Arc<Aes256GcmCipher>>>`.
//! On every `cipher_for_version(v)`:
//! 1. Acquire the mutex.
//! 2. If `v` is cached, clone the `Arc` + drop the mutex.
//! 3. If not cached, fetch from the provider, install in cache, drop
//!    the mutex.
//!
//! At W20β-3 v1.0-α this is the right granularity — the WAL writer
//! is single-threaded; the page-store decorator's hot path is
//! per-page (microsecond-scale ops, mutex acquisition overhead is in
//! the noise). v1.1 may lift to `arc-swap::ArcSwap` if profiling
//! shows contention.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use thiserror::Error;

use arcgraph_core::{KeyVersion, SecretValue, SecretsError, SecretsProvider};

use super::cipher::{Aes256GcmCipher, CipherError};

/// Standard namespace prefix for WAL encryption keys. Per ADR-052
/// §"Key namespace convention" the full key reads
/// `arcgraph.wal.encryption_key.v<N>`.
pub const ENCRYPTION_KEY_NAMESPACE_WAL: &str = "arcgraph.wal.encryption_key";

// v1.0-α narrowing (PR #373 R1): the page-store-encryption namespace
// (`arcgraph.page.encryption_key`) is intentionally NOT defined at v1.0-α
// per `feedback_avoid_speculative_scaffolding.md` — the v1.1 page-store
// encryption landing reintroduces it together with the encrypted page-IO
// impl + a corrected IV scheme.

/// Errors specific to the keyring layer. Distinct from
/// [`CipherError`] so the caller can distinguish "secret backend
/// missing the key version" from "cipher init failed" from "AEAD
/// decryption tag mismatch".
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum KeyRingError {
    /// The underlying `SecretsProvider` rejected the lookup.
    #[error("secrets provider error for {namespace}.v{version}: {source}")]
    Provider {
        /// Namespace + version that failed.
        namespace: String,
        /// The version requested.
        version: u16,
        /// Underlying provider error.
        #[source]
        source: SecretsError,
    },

    /// Cipher initialization failed (typically aws-lc-rs rejected the
    /// key bytes — should be impossible with fixed-width
    /// `SecretValue`).
    #[error("cipher init failed for {namespace}.v{version}: {source}")]
    Cipher {
        /// Namespace + version whose cipher init failed.
        namespace: String,
        /// The version requested.
        version: u16,
        /// Underlying cipher error.
        #[source]
        source: CipherError,
    },

    /// Mutex poisoned — fatal; should never happen on a non-panicking
    /// path.
    #[error("keyring mutex poisoned: {0}")]
    MutexPoisoned(String),
}

/// Maps `KeyVersion → Arc<Aes256GcmCipher>` with lazy population from
/// a `SecretsProvider`.
pub struct KeyRing {
    provider: Arc<dyn SecretsProvider>,
    namespace: String,
    current_version: KeyVersion,
    cache: Mutex<HashMap<u16, Arc<Aes256GcmCipher>>>,
}

impl std::fmt::Debug for KeyRing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KeyRing")
            .field("namespace", &self.namespace)
            .field("current_version", &self.current_version)
            .finish_non_exhaustive()
    }
}

impl KeyRing {
    /// Construct a key ring fronting `provider` for the given
    /// `namespace` (e.g., [`ENCRYPTION_KEY_NAMESPACE_WAL`]).
    /// `current_version` is the version new records / pages will be
    /// encrypted under; historical versions are fetched lazily from
    /// the provider on decryption.
    ///
    /// The constructor does NOT pre-fetch any version — the cache
    /// populates on first use. Callers that want to fail-fast on
    /// missing keys SHOULD call `cipher_for_version(current_version)`
    /// immediately after construction.
    #[must_use]
    pub fn new(
        provider: Arc<dyn SecretsProvider>,
        namespace: impl Into<String>,
        current_version: KeyVersion,
    ) -> Self {
        Self {
            provider,
            namespace: namespace.into(),
            current_version,
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// The namespace the key ring fronts.
    #[must_use]
    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    /// Current key version. New writes use this; decryption reads
    /// the version from the payload header.
    #[must_use]
    pub fn current_version(&self) -> KeyVersion {
        self.current_version
    }

    /// Increment the in-memory current-version pointer. The caller is
    /// responsible for having installed the new key in the
    /// underlying provider first (via `SecretsProvider::set` or
    /// `rotate`). Future writes encrypt under `next_version`;
    /// readback of old records still consults the cache + provider
    /// at the appropriate historical version.
    pub fn advance_to(&mut self, next: KeyVersion) {
        self.current_version = next;
    }

    /// Get the cipher for `version`. Returns the cached cipher if
    /// present; otherwise fetches the key from the underlying
    /// `SecretsProvider`, constructs the cipher, installs it in the
    /// cache, and returns a clone of the `Arc`.
    pub fn cipher_for_version(
        &self,
        version: KeyVersion,
    ) -> Result<Arc<Aes256GcmCipher>, KeyRingError> {
        let mut cache = self
            .cache
            .lock()
            .map_err(|e| KeyRingError::MutexPoisoned(format!("{e}")))?;
        if let Some(c) = cache.get(&version.raw()) {
            return Ok(Arc::clone(c));
        }

        let key = format!("{}.v{}", self.namespace, version.raw());
        let secret: SecretValue =
            self.provider
                .get(&key)
                .map_err(|source| KeyRingError::Provider {
                    namespace: self.namespace.clone(),
                    version: version.raw(),
                    source,
                })?;
        let cipher = Aes256GcmCipher::from_key(secret.expose_bytes()).map_err(|source| {
            KeyRingError::Cipher {
                namespace: self.namespace.clone(),
                version: version.raw(),
                source,
            }
        })?;
        let arc = Arc::new(cipher);
        cache.insert(version.raw(), Arc::clone(&arc));
        Ok(arc)
    }

    /// Convenience: get the cipher for the current write version.
    pub fn current_cipher(&self) -> Result<Arc<Aes256GcmCipher>, KeyRingError> {
        self.cipher_for_version(self.current_version)
    }

    /// Number of cached cipher instances. Diagnostic only.
    #[must_use]
    pub fn cache_size(&self) -> usize {
        self.cache.lock().map(|c| c.len()).unwrap_or(0)
    }
}

/// Generate + install a fresh AES-256 key at `namespace.v<version>`
/// in the given provider. Used by tests + bootstrap utilities. Reads
/// 32 bytes from `/dev/urandom` (Unix) and stores them via
/// `SecretsProvider::set`. On non-Unix the caller MUST install the
/// key explicitly.
pub fn install_random_key(
    provider: &dyn SecretsProvider,
    namespace: &str,
    version: KeyVersion,
) -> Result<(), KeyRingError> {
    let bytes = read_csprng_bytes().map_err(|reason| KeyRingError::Provider {
        namespace: namespace.to_owned(),
        version: version.raw(),
        source: SecretsError::Backend {
            key: format!("{namespace}.v{}", version.raw()),
            reason,
        },
    })?;
    let key = format!("{namespace}.v{}", version.raw());
    provider
        .set(&key, SecretValue::new(bytes))
        .map_err(|source| KeyRingError::Provider {
            namespace: namespace.to_owned(),
            version: version.raw(),
            source,
        })
}

#[cfg(unix)]
fn read_csprng_bytes() -> Result<[u8; super::AEAD_KEY_LEN], String> {
    use std::io::Read;
    let mut bytes = [0u8; super::AEAD_KEY_LEN];
    let mut f = std::fs::File::open("/dev/urandom")
        .map_err(|e| format!("open /dev/urandom failed: {e}"))?;
    f.read_exact(&mut bytes)
        .map_err(|e| format!("read /dev/urandom failed: {e}"))?;
    Ok(bytes)
}

#[cfg(not(unix))]
fn read_csprng_bytes() -> Result<[u8; super::AEAD_KEY_LEN], String> {
    Err("CSPRNG read unsupported on non-Unix; install key explicitly".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arcgraph_core::{EnvSecretsProvider, KeyVersion, SECRET_VALUE_LEN, SecretValue};

    fn unique_prefix(suffix: &str) -> String {
        let pid = std::process::id();
        let thread_id = std::thread::current().id();
        format!("ARCGRAPH_KEYRING_TEST_{pid}_{thread_id:?}_{suffix}_").replace([' ', '(', ')'], "_")
    }

    fn fixture_provider(prefix: &str) -> Arc<dyn SecretsProvider> {
        Arc::new(EnvSecretsProvider::without_startup_warn_for_tests(
            prefix.to_owned(),
        ))
    }

    fn install_fixture_key(p: &dyn SecretsProvider, key: &str, byte: u8) {
        p.set(key, SecretValue::new([byte; SECRET_VALUE_LEN]))
            .expect("install fixture key");
    }

    #[test]
    fn cipher_for_version_caches() {
        let prefix = unique_prefix("cache");
        let provider = fixture_provider(&prefix);
        install_fixture_key(&*provider, "ns.v1", 0xAB);
        let ring = KeyRing::new(Arc::clone(&provider), "ns", KeyVersion::ONE);
        assert_eq!(ring.cache_size(), 0);
        let c1 = ring.cipher_for_version(KeyVersion::ONE).expect("v1");
        assert_eq!(ring.cache_size(), 1);
        let c2 = ring.cipher_for_version(KeyVersion::ONE).expect("v1 cached");
        assert!(Arc::ptr_eq(&c1, &c2), "cache MUST return the same Arc");
        assert_eq!(ring.cache_size(), 1);
    }

    #[test]
    fn cipher_for_version_missing_propagates_provider_error() {
        let prefix = unique_prefix("missing");
        let provider = fixture_provider(&prefix);
        let ring = KeyRing::new(provider, "ns", KeyVersion::ONE);
        let err = ring.cipher_for_version(KeyVersion::ONE).unwrap_err();
        match err {
            KeyRingError::Provider { source, .. } => {
                assert!(matches!(source, SecretsError::NotFound { .. }));
            }
            other => panic!("expected Provider err, got {other:?}"),
        }
    }

    #[test]
    fn historical_version_decryption_via_cache() {
        let prefix = unique_prefix("history");
        let provider = fixture_provider(&prefix);
        install_fixture_key(&*provider, "ns.v1", 0x11);
        install_fixture_key(&*provider, "ns.v2", 0x22);

        let mut ring = KeyRing::new(Arc::clone(&provider), "ns", KeyVersion::ONE);
        let c1 = ring.cipher_for_version(KeyVersion::ONE).expect("v1");

        // Rotate: advance + new version.
        ring.advance_to(KeyVersion::new(2));
        assert_eq!(ring.current_version().raw(), 2);
        let c2 = ring.cipher_for_version(KeyVersion::new(2)).expect("v2");

        // Both ciphers in cache; v1 still readable for old records.
        assert_eq!(ring.cache_size(), 2);
        let c1_again = ring.cipher_for_version(KeyVersion::ONE).unwrap();
        assert!(Arc::ptr_eq(&c1, &c1_again));
        // v1 and v2 are different ciphers (different key bytes).
        assert!(!Arc::ptr_eq(&c1, &c2));
    }

    #[test]
    fn current_cipher_uses_current_version() {
        let prefix = unique_prefix("current");
        let provider = fixture_provider(&prefix);
        install_fixture_key(&*provider, "ns.v3", 0x33);
        let ring = KeyRing::new(Arc::clone(&provider), "ns", KeyVersion::new(3));
        let _ = ring.current_cipher().expect("v3 cipher");
        assert_eq!(ring.cache_size(), 1);
    }

    #[cfg(unix)]
    #[test]
    fn install_random_key_round_trips() {
        let prefix = unique_prefix("install");
        let provider = fixture_provider(&prefix);
        install_random_key(&*provider, "ns", KeyVersion::ONE).expect("install");
        let ring = KeyRing::new(Arc::clone(&provider), "ns", KeyVersion::ONE);
        let _ = ring.current_cipher().expect("freshly installed key works");
    }
}
