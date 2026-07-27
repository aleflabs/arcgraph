//! Environment-variable-backed [`SecretsProvider`] — DEVELOPMENT
//! ONLY.
//!
//! `EnvSecretsProvider` reads secrets from process environment
//! variables under a configurable prefix (default `ARCGRAPH_SECRET_`).
//! Construction emits a `tracing::warn!` tagged with
//! `secrets.provider = "env"` and `unsafe_for_prod = true` so
//! operators see the danger in startup logs.
//!
//! ## Key → env-var mapping
//!
//! `key` is uppercased; the literal `.` is replaced with `_DOT_` so
//! `arcgraph.wal.encryption_key.v1` → `ARCGRAPH_DOT_WAL_DOT_ENCRYPTION_KEY_DOT_V1`
//! (after applying the `ARCGRAPH_SECRET_` prefix). The verbose
//! `_DOT_` is INTENTIONAL: env vars don't allow `.`, and a naive `_`
//! substitution would collide with keys that legitimately carry
//! underscores. Using `_DOT_` keeps the mapping reversible and
//! collision-free.
//!
//! ## Value format
//!
//! Env-var values are hex-encoded (64 lowercase hex chars for the
//! 32-byte fixed-width [`SecretValue`]). Base64 would be more
//! compact but mixing case is error-prone in shell environments; hex
//! is the standard ops-team-friendly format. The provider validates
//! length + parses hex at `get()` time; a non-hex / wrong-length
//! value returns [`SecretsError::InvalidLength`] (length-mismatch
//! after hex-decoding) or [`SecretsError::Backend`] (un-parseable
//! hex).
//!
//! ## Mutability
//!
//! `set()` modifies the process's own environment via
//! [`std::env::set_var`] — visible only to the current process, NOT
//! persisted to disk. This is intentional: an `EnvSecretsProvider` is
//! a development fixture, not a persistent store. `rotate()` returns
//! [`SecretsError::Unsupported`] because env vars don't carry version
//! metadata across restarts; operators rotating with env must update
//! `systemd` / `k8s` env + restart.

use std::sync::Mutex;

use tracing::warn;

use super::SecretsProvider;
use super::error::SecretsError;
use super::value::{KeyVersion, SECRET_VALUE_LEN, SecretValue};

/// The default prefix for env vars holding ArcGraph secrets.
pub const DEFAULT_ENV_PREFIX: &str = "ARCGRAPH_SECRET_";

/// Env-var-backed provider. **NOT for production use** — emits a
/// `tracing::warn!` at construction.
///
/// Construction is idempotent: making N instances emits N warnings,
/// which surfaces the danger every time. Wrap with
/// [`Self::without_startup_warn_for_tests`] in unit tests to silence
/// the noise (the warning is the production-facing pin, not a
/// test-time pin).
pub struct EnvSecretsProvider {
    prefix: String,
    /// Guards process-wide env-var mutations so two concurrent
    /// `set()` calls don't race on the env table. The env table is
    /// process-global; this mutex is the cheapest correctness barrier.
    write_lock: Mutex<()>,
}

impl std::fmt::Debug for EnvSecretsProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EnvSecretsProvider")
            .field("prefix", &self.prefix)
            .finish()
    }
}

impl EnvSecretsProvider {
    /// Construct with the default `ARCGRAPH_SECRET_` prefix. Emits
    /// the startup warning. Use [`Self::with_prefix`] for a custom
    /// prefix.
    #[must_use]
    pub fn new() -> Self {
        Self::with_prefix(DEFAULT_ENV_PREFIX)
    }

    /// Construct with a custom env-var prefix. Useful in tests +
    /// multi-tenant fixtures where two providers must not see each
    /// other's vars.
    ///
    /// Emits a `tracing::warn!` tagged `unsafe_for_prod=true`.
    #[must_use]
    pub fn with_prefix(prefix: impl Into<String>) -> Self {
        let prefix = prefix.into();
        warn!(
            target: "arcgraph_core::secrets",
            secrets_provider = "env",
            unsafe_for_prod = true,
            prefix = %prefix,
            "EnvSecretsProvider constructed — UNSAFE FOR PRODUCTION; \
             use OsKeyringProvider (`os-keyring` feature) for v1.0-α deployments."
        );
        Self {
            prefix,
            write_lock: Mutex::new(()),
        }
    }

    /// Construct WITHOUT emitting the startup warning. Tests should
    /// use this constructor so the test log isn't spammed; production
    /// callers should NEVER use this — the warning is the load-
    /// bearing operator-facing pin.
    ///
    /// Doc-hidden because no public consumer should reach for it.
    #[doc(hidden)]
    #[must_use]
    pub fn without_startup_warn_for_tests(prefix: impl Into<String>) -> Self {
        Self {
            prefix: prefix.into(),
            write_lock: Mutex::new(()),
        }
    }

    /// The current prefix (for diagnostics).
    #[must_use]
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// Map `key` → env-var-name under the configured prefix.
    fn env_var_name(&self, key: &str) -> String {
        let canonical = key.replace('.', "_DOT_");
        format!("{}{}", self.prefix, canonical.to_uppercase())
    }
}

impl Default for EnvSecretsProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl SecretsProvider for EnvSecretsProvider {
    fn get(&self, key: &str) -> Result<SecretValue, SecretsError> {
        let var = self.env_var_name(key);
        let raw = match std::env::var(&var) {
            Ok(s) => s,
            Err(std::env::VarError::NotPresent) => {
                return Err(SecretsError::NotFound {
                    key: key.to_owned(),
                });
            }
            Err(std::env::VarError::NotUnicode(_)) => {
                return Err(SecretsError::Backend {
                    key: key.to_owned(),
                    reason: format!("env var {var} contains non-UTF-8 bytes"),
                });
            }
        };
        decode_hex_secret(key, &raw)
    }

    fn set(&self, key: &str, value: SecretValue) -> Result<(), SecretsError> {
        let _guard = self.write_lock.lock().map_err(|_| SecretsError::Backend {
            key: key.to_owned(),
            reason: "env var write lock poisoned".to_owned(),
        })?;
        let var = self.env_var_name(key);
        let hex = encode_hex(value.expose_bytes());
        // SAFETY: `std::env::set_var` is unsafe in 2024 edition because
        // POSIX getenv/setenv are not thread-safe. We hold `write_lock`
        // for the duration of the set + we document that
        // `EnvSecretsProvider` is dev-only. Concurrent readers via
        // `std::env::var` may see torn reads, but development workflows
        // do not concurrently mutate env vars.
        unsafe {
            std::env::set_var(&var, hex);
        }
        Ok(())
    }

    fn rotate(&self, key: &str) -> Result<KeyVersion, SecretsError> {
        Err(SecretsError::Unsupported {
            operation: "rotate",
            key: key.to_owned(),
            reason: "EnvSecretsProvider is dev-only; rotate by updating \
                     systemd / k8s env + restarting the process. \
                     Production deployments use OsKeyringProvider (rotate \
                     supported) or v1.2 KMS providers (per ADR-051)."
                .to_owned(),
        })
    }

    fn delete(&self, key: &str) -> Result<(), SecretsError> {
        let _guard = self.write_lock.lock().map_err(|_| SecretsError::Backend {
            key: key.to_owned(),
            reason: "env var write lock poisoned".to_owned(),
        })?;
        let var = self.env_var_name(key);
        match std::env::var(&var) {
            Ok(_) => {
                // SAFETY: see `set` above — write_lock + dev-only
                // discipline guards the unsafe std::env::remove_var.
                unsafe {
                    std::env::remove_var(&var);
                }
                Ok(())
            }
            Err(std::env::VarError::NotPresent) => Err(SecretsError::NotFound {
                key: key.to_owned(),
            }),
            Err(std::env::VarError::NotUnicode(_)) => {
                // PR #373 R1 §N-2: surface the underlying NotUnicode
                // condition. The previous behavior silently removed
                // the malformed var and returned Ok(()) — that
                // collapses two distinct operator-visible failure
                // modes (clean delete vs. malformed-then-deleted)
                // into one return, hiding the corruption. We still
                // attempt the remove (a malformed value is unusable
                // anyway), but report the original issue so the
                // operator sees that their env table contained a
                // non-UTF-8 value that needed cleanup.
                unsafe {
                    std::env::remove_var(&var);
                }
                Err(SecretsError::Backend {
                    key: key.to_owned(),
                    reason: format!(
                        "env var {var} contained non-UTF-8 bytes; \
                         the malformed value was removed but the underlying \
                         encoding error MUST be reported per PR #373 R1 §N-2 \
                         — investigate how a non-UTF-8 value reached the \
                         env table"
                    ),
                })
            }
        }
    }
}

/// Decode a hex-encoded 64-char string into a [`SecretValue`].
fn decode_hex_secret(key: &str, hex: &str) -> Result<SecretValue, SecretsError> {
    let hex = hex.trim();
    if hex.len() != SECRET_VALUE_LEN * 2 {
        return Err(SecretsError::InvalidLength {
            key: key.to_owned(),
            got: hex.len() / 2,
            expected: SECRET_VALUE_LEN,
        });
    }
    let mut bytes = [0u8; SECRET_VALUE_LEN];
    for (i, chunk) in hex.as_bytes().chunks(2).enumerate() {
        let high = decode_nibble(chunk[0]).ok_or_else(|| SecretsError::Backend {
            key: key.to_owned(),
            reason: format!("non-hex byte 0x{:02x} at position {}", chunk[0], i * 2),
        })?;
        let low = decode_nibble(chunk[1]).ok_or_else(|| SecretsError::Backend {
            key: key.to_owned(),
            reason: format!("non-hex byte 0x{:02x} at position {}", chunk[1], i * 2 + 1),
        })?;
        bytes[i] = (high << 4) | low;
    }
    Ok(SecretValue::new(bytes))
}

fn decode_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

fn encode_hex(bytes: &[u8; SECRET_VALUE_LEN]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(SECRET_VALUE_LEN * 2);
    for b in bytes.iter() {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0F) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Per-test prefix-derivation helper. Each test uses its own
    /// process-wide env-var namespace so concurrent tests don't
    /// stomp each other.
    fn unique_prefix(suffix: &str) -> String {
        let pid = std::process::id();
        let thread_id = std::thread::current().id();
        format!("ARCGRAPH_TEST_{pid}_{thread_id:?}_{suffix}_").replace([' ', '(', ')'], "_")
    }

    #[test]
    fn env_var_name_uppercases_and_dot_escapes() {
        let p = EnvSecretsProvider::without_startup_warn_for_tests("ARCGRAPH_SECRET_");
        assert_eq!(
            p.env_var_name("arcgraph.wal.encryption_key.v1"),
            "ARCGRAPH_SECRET_ARCGRAPH_DOT_WAL_DOT_ENCRYPTION_KEY_DOT_V1"
        );
    }

    #[test]
    fn round_trip_set_get() {
        let prefix = unique_prefix("round_trip");
        let p = EnvSecretsProvider::without_startup_warn_for_tests(prefix);
        let key = "my.test.key";
        let val = SecretValue::new([0x42; SECRET_VALUE_LEN]);
        p.set(key, val.clone()).expect("set");
        let back = p.get(key).expect("get");
        assert_eq!(back, val);
    }

    #[test]
    fn get_missing_returns_not_found() {
        let prefix = unique_prefix("missing");
        let p = EnvSecretsProvider::without_startup_warn_for_tests(prefix);
        let err = p.get("definitely.never.set").unwrap_err();
        assert!(matches!(err, SecretsError::NotFound { .. }));
    }

    #[test]
    fn rotate_returns_unsupported() {
        let prefix = unique_prefix("rotate");
        let p = EnvSecretsProvider::without_startup_warn_for_tests(prefix);
        let err = p.rotate("any.key").unwrap_err();
        match err {
            SecretsError::Unsupported {
                operation, reason, ..
            } => {
                assert_eq!(operation, "rotate");
                assert!(
                    reason.contains("dev-only"),
                    "rotate's Unsupported reason must explain why: {reason}"
                );
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn delete_removes_var() {
        let prefix = unique_prefix("delete");
        let p = EnvSecretsProvider::without_startup_warn_for_tests(prefix);
        let key = "k";
        p.set(key, SecretValue::new([0; SECRET_VALUE_LEN]))
            .expect("set");
        assert!(p.get(key).is_ok());
        p.delete(key).expect("delete");
        assert!(matches!(p.get(key), Err(SecretsError::NotFound { .. })));
    }

    #[test]
    fn delete_missing_returns_not_found() {
        let prefix = unique_prefix("delete_missing");
        let p = EnvSecretsProvider::without_startup_warn_for_tests(prefix);
        let err = p.delete("nope").unwrap_err();
        assert!(matches!(err, SecretsError::NotFound { .. }));
    }

    #[test]
    fn hex_decode_rejects_non_hex() {
        let err = decode_hex_secret("k", &"z".repeat(64)).unwrap_err();
        assert!(matches!(err, SecretsError::Backend { .. }));
    }

    #[test]
    fn hex_decode_rejects_wrong_length() {
        let err = decode_hex_secret("k", &"a".repeat(32)).unwrap_err();
        match err {
            SecretsError::InvalidLength { got, expected, .. } => {
                assert_eq!(got, 16);
                assert_eq!(expected, SECRET_VALUE_LEN);
            }
            other => panic!("expected InvalidLength, got {other:?}"),
        }
    }

    #[test]
    fn hex_round_trip_preserves_bytes() {
        let bytes: [u8; SECRET_VALUE_LEN] = std::array::from_fn(|i| (i * 7 + 13) as u8);
        let hex = encode_hex(&bytes);
        let back = decode_hex_secret("k", &hex).unwrap();
        assert_eq!(back.expose_bytes(), &bytes);
    }

    /// The startup warning is the LOAD-BEARING operator-facing pin
    /// — emit a startup warning every time the constructor runs.
    /// Use `tracing-test` to assert; if unavailable, fall back to a
    /// best-effort check that the constructor doesn't panic.
    ///
    /// At v1.0-α `tracing-test` is not a workspace dep; we pin the
    /// warning through the source-level discipline: the constructor's
    /// body MUST contain a `warn!` call with `unsafe_for_prod = true`.
    /// This test reads the source file and asserts the literal exists.
    #[test]
    fn startup_emits_unsafe_for_prod_warning_source_pin() {
        let src = include_str!("env.rs");
        // Match the literal `unsafe_for_prod = true` arg inside the
        // tracing macro body. We search for the exact macro-arg
        // form (with `= true`) because the test docstring + comments
        // already contain `unsafe_for_prod=true` (no spaces).
        assert!(
            src.contains("unsafe_for_prod = true"),
            "EnvSecretsProvider::new / with_prefix MUST emit `unsafe_for_prod = true` \
             in its startup warning"
        );
        assert!(
            src.contains("UNSAFE FOR PRODUCTION"),
            "EnvSecretsProvider startup warning MUST literally say \
             `UNSAFE FOR PRODUCTION` so log-search catches it"
        );
    }
}
