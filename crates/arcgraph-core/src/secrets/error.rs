//! Error taxonomy for the secrets provider trait.

use thiserror::Error;

/// Every error a [`crate::secrets::SecretsProvider`] can produce.
///
/// `#[non_exhaustive]` lets future provider impls (KMS at v1.2 per
/// ADR-051) add new
/// variants without breaking downstream pattern matches.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SecretsError {
    /// No secret bound to the requested key.
    #[error("secret not found: {key}")]
    NotFound {
        /// The key the caller looked up (NOT the secret value — keys
        /// are safe to log; values are not).
        key: String,
    },

    /// The provider backend rejected the request (D-Bus down, keyring
    /// locked, file permission denied, etc.). The `reason` is
    /// operator-facing and MUST NOT include the secret value bytes.
    #[error("secrets backend error on key {key}: {reason}")]
    Backend {
        /// The key the operation was attempted against.
        key: String,
        /// Human-readable cause (provider-specific).
        reason: String,
    },

    /// The retrieved value is not the fixed-width 32 bytes a
    /// `SecretValue` requires. Typically surfaces when the provider
    /// holds a legacy variable-length token alongside fixed-width
    /// AES-256 keys.
    #[error("secret at key {key} has wrong length: got {got} bytes, expected {expected}")]
    InvalidLength {
        /// The key whose value failed length validation.
        key: String,
        /// Bytes read from the provider.
        got: usize,
        /// Bytes expected (32 for v1.0-α).
        expected: usize,
    },

    /// The provider does not support the requested operation. The two
    /// expected cases at v1.0-α:
    ///
    /// - [`crate::secrets::EnvSecretsProvider`] does not implement
    ///   [`crate::secrets::SecretsProvider::rotate`] (env vars are
    ///   read-only from the running process's perspective; operators
    ///   rotate by updating systemd / k8s env + restarting).
    /// - Future KMS providers (v1.2) may reject `set` if the backend
    ///   only exposes a read-only key-fetch interface.
    #[error("secrets backend does not support {operation} on key {key}: {reason}")]
    Unsupported {
        /// The operation that was attempted (`get` / `set` / `rotate`
        /// / `delete`).
        operation: &'static str,
        /// The key the operation was attempted against.
        key: String,
        /// Operator-facing rationale + lift target.
        reason: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_not_found_names_key() {
        let e = SecretsError::NotFound {
            key: "arcgraph.wal.encryption_key.v1".to_owned(),
        };
        let s = format!("{e}");
        assert!(s.contains("secret not found"), "got: {s}");
        assert!(s.contains("arcgraph.wal.encryption_key.v1"), "got: {s}");
    }

    #[test]
    fn display_backend_carries_reason() {
        let e = SecretsError::Backend {
            key: "test.key".to_owned(),
            reason: "D-Bus session bus not available".to_owned(),
        };
        let s = format!("{e}");
        assert!(s.contains("backend error"), "got: {s}");
        assert!(s.contains("D-Bus"), "got: {s}");
    }

    #[test]
    fn display_invalid_length_shows_sizes() {
        let e = SecretsError::InvalidLength {
            key: "k".to_owned(),
            got: 16,
            expected: 32,
        };
        let s = format!("{e}");
        assert!(s.contains("got 16"), "got: {s}");
        assert!(s.contains("expected 32"), "got: {s}");
    }

    #[test]
    fn display_unsupported_names_operation() {
        let e = SecretsError::Unsupported {
            operation: "rotate",
            key: "k".to_owned(),
            reason: "env vars are read-only; restart with new env".to_owned(),
        };
        let s = format!("{e}");
        assert!(s.contains("does not support rotate"), "got: {s}");
        assert!(s.contains("read-only"), "got: {s}");
    }

    /// `SecretsError::Backend.reason` MUST NOT include raw secret
    /// bytes — values are not safe to log. This is a documentation
    /// pin, not a runtime check, but the pin lives in tests so a
    /// future field addition surfaces the contract.
    #[test]
    fn backend_reason_field_is_documented_as_log_safe() {
        let docs = include_str!("error.rs");
        assert!(
            docs.contains("MUST NOT include the secret value bytes"),
            "the reason field's log-safety contract must stay documented"
        );
    }
}
