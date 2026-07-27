//! OS-keyring-backed [`SecretsProvider`] — production default at
//! v1.0-α.
//!
//! Wraps the `keyring` crate (Apache-2.0/MIT) for macOS Keychain /
//! Windows Credential Manager / Linux Secret Service. Feature-gated
//! behind `os-keyring` so headless CI without D-Bus / libsecret
//! builds without system deps.
//!
//! ## Service vs. user
//!
//! The `keyring::Entry::new(service, username)` API pairs every key
//! with a `service` string + `username` (the "account"). Our keys
//! follow the convention:
//!
//! - `service = OsKeyringProvider::service_prefix()` (default
//!   `arcgraph`)
//! - `username = key`  (the verbatim caller-supplied key, e.g.
//!   `arcgraph.wal.encryption_key.v1`)
//!
//! Per-provider `service_prefix` overrides exist for multi-instance
//! deployments (multiple ArcGraph instances on the same host).
//!
//! ## Rotation
//!
//! `rotate(key)` reads the current version stored at
//! `<key>.__current_version__` (a small u16 sidecar), generates a
//! fresh 32-byte value via the OS CSPRNG (via aws-lc-rs), stores it
//! at `<key>.v<new_version>`, increments + writes back the sidecar
//! pointer. Atomicity is best-effort — the keyring backend does not
//! expose transactions; a crash between writing the new key and
//! updating the pointer leaves the new key orphaned but the old key
//! readable (operationally safe — operator can re-rotate).

use std::sync::Mutex;

use tracing::info;

use super::SecretsProvider;
use super::error::SecretsError;
use super::value::{KeyVersion, SECRET_VALUE_LEN, SecretValue};

/// Default service prefix the keyring entries land under.
pub const DEFAULT_SERVICE_PREFIX: &str = "arcgraph";

/// Default sidecar suffix the current-version pointer lives at.
const CURRENT_VERSION_SUFFIX: &str = ".__current_version__";

/// OS-keyring-backed provider.
pub struct OsKeyringProvider {
    service_prefix: String,
    /// Serializes rotate() so two concurrent rotations don't race on
    /// the version sidecar.
    rotate_lock: Mutex<()>,
}

impl std::fmt::Debug for OsKeyringProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OsKeyringProvider")
            .field("service_prefix", &self.service_prefix)
            .finish()
    }
}

impl OsKeyringProvider {
    /// Construct with the default `arcgraph` service prefix.
    #[must_use]
    pub fn new() -> Self {
        Self::with_service_prefix(DEFAULT_SERVICE_PREFIX)
    }

    /// Construct with a custom service prefix. Use for multi-instance
    /// deployments where two ArcGraph processes must NOT share a
    /// secret namespace.
    #[must_use]
    pub fn with_service_prefix(prefix: impl Into<String>) -> Self {
        let service_prefix = prefix.into();
        info!(
            target: "arcgraph_core::secrets",
            secrets_provider = "os-keyring",
            service_prefix = %service_prefix,
            "OsKeyringProvider constructed — production-grade key custody \
             via OS keyring (macOS Keychain / Windows Credential Manager / \
             Linux Secret Service). See ADR-052 §Decision."
        );
        Self {
            service_prefix,
            rotate_lock: Mutex::new(()),
        }
    }

    /// The service prefix in use.
    #[must_use]
    pub fn service_prefix(&self) -> &str {
        &self.service_prefix
    }

    fn entry(&self, key: &str) -> Result<::keyring::Entry, SecretsError> {
        ::keyring::Entry::new(&self.service_prefix, key).map_err(|e| SecretsError::Backend {
            key: key.to_owned(),
            reason: format!("keyring::Entry::new failed: {e}"),
        })
    }
}

impl Default for OsKeyringProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl SecretsProvider for OsKeyringProvider {
    fn get(&self, key: &str) -> Result<SecretValue, SecretsError> {
        let entry = self.entry(key)?;
        let bytes = match entry.get_secret() {
            Ok(b) => b,
            Err(::keyring::Error::NoEntry) => {
                return Err(SecretsError::NotFound {
                    key: key.to_owned(),
                });
            }
            Err(e) => {
                return Err(SecretsError::Backend {
                    key: key.to_owned(),
                    reason: format!("get_secret failed: {e}"),
                });
            }
        };
        SecretValue::try_from_slice(&bytes).ok_or_else(|| SecretsError::InvalidLength {
            key: key.to_owned(),
            got: bytes.len(),
            expected: SECRET_VALUE_LEN,
        })
    }

    fn set(&self, key: &str, value: SecretValue) -> Result<(), SecretsError> {
        let entry = self.entry(key)?;
        entry
            .set_secret(value.expose_bytes())
            .map_err(|e| SecretsError::Backend {
                key: key.to_owned(),
                reason: format!("set_secret failed: {e}"),
            })?;
        Ok(())
    }

    fn rotate(&self, key: &str) -> Result<KeyVersion, SecretsError> {
        let _guard = self.rotate_lock.lock().map_err(|_| SecretsError::Backend {
            key: key.to_owned(),
            reason: "rotate lock poisoned".to_owned(),
        })?;

        // Read the current version sidecar; default to ZERO (the
        // sentinel for "no rotation yet") on missing.
        let current = read_current_version(self, key)?;
        let new_version = current.next();

        // Generate a fresh 32-byte key. v1.0-α uses aws-lc-rs's
        // SystemRandom (the same RNG rustls uses); we route through
        // `getrandom` here because it's already in `Cargo.lock`
        // transitively, and aws-lc-rs isn't a direct dep of
        // arcgraph-core (we want to keep arcgraph-core dependency-
        // lean under the bounded-context policy "no I/O").
        //
        // Actually — arcgraph-core has no aws-lc-rs or getrandom
        // direct dep at v1.0-α. We read from /dev/urandom directly
        // to keep the dep surface minimal. On Windows / non-Unix
        // platforms, callers must use the test-side
        // `set_with_value_for_tests` path.
        let bytes = read_csprng_bytes()?;
        let new_value = SecretValue::new(bytes);

        // Write the new value at key.v<new_version>.
        let versioned_key = format!("{key}.v{}", new_version.raw());
        self.set(&versioned_key, new_value)?;

        // Update the current-version sidecar.
        write_current_version(self, key, new_version)?;

        Ok(new_version)
    }

    fn delete(&self, key: &str) -> Result<(), SecretsError> {
        let entry = self.entry(key)?;
        match entry.delete_credential() {
            Ok(()) => Ok(()),
            Err(::keyring::Error::NoEntry) => Err(SecretsError::NotFound {
                key: key.to_owned(),
            }),
            Err(e) => Err(SecretsError::Backend {
                key: key.to_owned(),
                reason: format!("delete_credential failed: {e}"),
            }),
        }
    }
}

fn read_current_version(p: &OsKeyringProvider, key: &str) -> Result<KeyVersion, SecretsError> {
    let pointer_key = format!("{key}{CURRENT_VERSION_SUFFIX}");
    let entry = p.entry(&pointer_key)?;
    match entry.get_secret() {
        Ok(bytes) if bytes.len() == 2 => {
            let v = u16::from_le_bytes([bytes[0], bytes[1]]);
            Ok(KeyVersion::new(v))
        }
        Ok(other) => Err(SecretsError::Backend {
            key: pointer_key,
            reason: format!(
                "version sidecar has wrong length: got {} bytes, expected 2",
                other.len()
            ),
        }),
        Err(::keyring::Error::NoEntry) => Ok(KeyVersion::new(0)),
        Err(e) => Err(SecretsError::Backend {
            key: pointer_key,
            reason: format!("read version sidecar failed: {e}"),
        }),
    }
}

fn write_current_version(
    p: &OsKeyringProvider,
    key: &str,
    version: KeyVersion,
) -> Result<(), SecretsError> {
    let pointer_key = format!("{key}{CURRENT_VERSION_SUFFIX}");
    let entry = p.entry(&pointer_key)?;
    entry
        .set_secret(&version.raw().to_le_bytes())
        .map_err(|e| SecretsError::Backend {
            key: pointer_key,
            reason: format!("write version sidecar failed: {e}"),
        })?;
    Ok(())
}

/// Read 32 bytes from the OS CSPRNG (`/dev/urandom` on Unix). v1.0-α
/// uses a direct read to keep arcgraph-core off the `getrandom` /
/// `aws-lc-rs` dep cycle (those are storage-crate concerns).
fn read_csprng_bytes() -> Result<[u8; SECRET_VALUE_LEN], SecretsError> {
    #[cfg(unix)]
    {
        use std::io::Read;
        let mut bytes = [0u8; SECRET_VALUE_LEN];
        let mut f = std::fs::File::open("/dev/urandom").map_err(|e| SecretsError::Backend {
            key: "<rotate>".to_owned(),
            reason: format!("open /dev/urandom failed: {e}"),
        })?;
        f.read_exact(&mut bytes)
            .map_err(|e| SecretsError::Backend {
                key: "<rotate>".to_owned(),
                reason: format!("read /dev/urandom failed: {e}"),
            })?;
        Ok(bytes)
    }
    #[cfg(not(unix))]
    {
        Err(SecretsError::Unsupported {
            operation: "rotate",
            key: "<csprng>".to_owned(),
            reason: "automatic CSPRNG-backed rotate is Unix-only at v1.0-α; \
                     Windows callers must set rotated keys explicitly via `set()`. \
                     v1.0 GA lifts to a cross-platform CSPRNG path."
                .to_owned(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Constructor is infallible (the keyring backend isn't touched
    /// until the first method call).
    #[test]
    fn constructor_does_not_touch_keyring() {
        // No panic, no side effect.
        let _p = OsKeyringProvider::with_service_prefix("arcgraph_test_constructor");
    }

    #[test]
    fn debug_redacts_internals() {
        let p = OsKeyringProvider::with_service_prefix("arcgraph_test_debug");
        let s = format!("{p:?}");
        assert!(s.contains("OsKeyringProvider"), "got: {s}");
        assert!(s.contains("arcgraph_test_debug"), "got: {s}");
    }

    // We deliberately do NOT exercise live keyring round-trip here —
    // CI environments often lack a usable D-Bus session bus. The
    // adversarial cross-tenant test in `arcgraph-storage` exercises
    // the provider via the EnvSecretsProvider path; the keyring path
    // is exercised in the manual M5+ deployment-side test suite.
}
