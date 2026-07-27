//! Key-MANAGEMENT abstraction (`KeySource`) layered above the
//! key-STORAGE `SecretsProvider` (ADR-216 §D-1).
//!
//! Where [`SecretsProvider`](crate::secrets::SecretsProvider) is the
//! flat `get/set/rotate/delete` STORAGE backend for key bytes, a
//! [`KeySource`] is the key-MANAGEMENT surface: it resolves the
//! **key-encryption key (KEK)** for a named [`KeyScope`] and performs
//! envelope wrap/unwrap of **data-encryption keys (DEKs)**. Concrete
//! sources differ ONLY in *where the KEK lives* and *who performs the
//! wrap* (local in-process AES-256-GCM key-wrap vs a remote
//! KMS/HSM `Encrypt`/`Decrypt` call).
//!
//! ## What lives here (vs `arcgraph-storage`)
//!
//! Per ADR-216 §D-1 + "What it locks in" #4, the `KeySource` *trait*
//! and its supporting on-disk envelope TYPES live in
//! `arcgraph-core::secrets` — the key-management abstraction is a core
//! bounded-context concern (a sibling of `SecretsProvider`), and
//! storage consumes it, never the reverse. The ONE v1 concrete source
//! (`SecretsProviderKeySource`) lives in `arcgraph-storage::encryption`
//! because it performs the AES-256-GCM key-wrap via
//! `arcgraph-storage::encryption::cipher::Aes256GcmCipher`, and
//! `arcgraph-core` deliberately carries NO `aws-lc-rs` dependency
//! (keeping core I/O-free + crypto-free). This is the bounded-context
//! resolution for #1180: trait + types in core; the wrap impl in
//! storage where the cipher already lives.
//!
//! ## Envelope model (ADR-216 §D-3)
//!
//! ```text
//!   KEK  (lives in the KeySource backend — keyring/KMS; NEVER on disk
//!         next to the WAL)
//!    │  wraps
//!    ▼
//!   DEK  (32-byte AES-256 key the existing WalEncryption/KeyRing
//!         consumes; persisted ONLY in WRAPPED form as a WrappedDek in
//!         the `<data_dir>/wal/wal.dek` sidecar)
//!    │  encrypts
//!    ▼
//!   WAL record payloads
//! ```
//!
//! KEK rotation re-wraps the DEK (one small sidecar file, O(1) — no
//! WAL rewrite). DEK rotation bumps the in-record `key_version` and
//! appends a new wrapped DEK to the sidecar (history stays decryptable
//! via the existing per-record version routing).
//!
//! ## See also
//!
//! - `arcgraph-storage::encryption::keysource::SecretsProviderKeySource` — the v1 concrete source.
//! - [`crate::secrets::SecretsProvider`] — the key-STORAGE sibling trait.

use thiserror::Error;

use super::error::SecretsError;
use super::value::SecretValue;

/// The key-MANAGEMENT abstraction (above `SecretsProvider`, which is the
/// key-STORAGE backend). A `KeySource` resolves the KEK for a scope and
/// performs envelope wrap/unwrap of DEKs. Concrete sources differ ONLY in
/// where the KEK lives and who performs the wrap (local process vs remote
/// KMS/HSM). Object-safe; held as `Arc<dyn KeySource>` in `build_durable`.
///
/// ## Intentionally synchronous + retrofit-safe (ADR-216 CF-1 / NIT-2)
///
/// This trait is **deliberately synchronous** and **MUST stay so**. v1's
/// `SecretsProviderKeySource` is a local in-process AES-256-GCM key-wrap
/// called only OFF the WAL hot path (at `build_durable` startup + at
/// rotation events). The v1.1 CMK/Vault/cloud-KMS impls make a *network*
/// call (`kms:Decrypt`, `transit/decrypt`) which is naturally async — but
/// those calls also happen only at `build_durable`/rotation, NOT on the WAL
/// append path. A v1.1 impl therefore bridges async→sync at ITS OWN
/// boundary (a dedicated blocking executor / `block_on` on a runtime handle
/// the KMS source owns, or a small bounded thread pool inside the source)
/// WITHOUT changing this trait's signature. The sync trait is thus
/// retrofit-safe: the impl owns the async↔sync bridge, the trait surface is
/// stable. Do NOT "fix" this trait to `async` — that would be a breaking
/// change to a GA surface for no benefit (the calls are off the hot path).
pub trait KeySource: Send + Sync + std::fmt::Debug {
    /// Stable identifier for audit logs + the on-disk wrapped-DEK header
    /// (e.g. "secrets-provider:os-keyring", "aws-kms:arn:...",
    /// "vault:transit/keys/arcgraph-wal"). Bound into the wrapped-DEK AAD
    /// so a DEK wrapped under one source cannot be unwrapped as another.
    fn key_source_id(&self) -> &str;

    /// Generate a fresh 32-byte DEK and return it WRAPPED under the
    /// current KEK for `scope`, together with the KEK version used.
    /// For a local source the wrap is an in-process AES-256-GCM key-wrap
    /// under the KEK; for a remote KMS the wrap is a `kms:Encrypt` /
    /// `transit/encrypt` call (the plaintext DEK is generated locally via
    /// CSPRNG OR returned by the KMS `GenerateDataKey` op — impl choice).
    fn generate_wrapped_dek(&self, scope: &KeyScope) -> Result<WrappedDek, KeySourceError>;

    /// Unwrap a previously wrapped DEK to its 32-byte plaintext, routing
    /// to the KEK version stamped in `wrapped`. Returns a `SecretValue`
    /// (zeroized on drop). For a remote KMS this is `kms:Decrypt` /
    /// `transit/decrypt`. MUST fail-closed (structured error) on any
    /// resolution failure — NEVER returns a zero key or a placeholder.
    fn unwrap_dek(&self, wrapped: &WrappedDek) -> Result<SecretValue, KeySourceError>;

    /// Rotate the KEK for `scope`: provision a new KEK version in the
    /// backing store/KMS and return the new version. Historical KEK
    /// versions MUST remain resolvable for `unwrap_dek` (operator policy
    /// governs eventual KEK destruction). Does NOT rewrite any WAL — see
    /// the envelope rotation design (D-3).
    fn rotate_kek(&self, scope: &KeyScope) -> Result<KekVersion, KeySourceError>;

    /// Probe that the current KEK for `scope` is resolvable. Called at
    /// `build_durable` for fail-fast startup (no key ⟹ refuse to serve,
    /// per ADR-033 — never serve plaintext WAL silently).
    fn health_check(&self, scope: &KeyScope) -> Result<(), KeySourceError>;
}

/// Standard namespace prefix for WAL encryption keys (mirrors
/// `arcgraph-storage::encryption::keyring::ENCRYPTION_KEY_NAMESPACE_WAL`).
///
/// Defined here so [`KeyScope::wal`] can name the scope without forcing
/// `arcgraph-core` to depend on `arcgraph-storage`. The two constants are
/// pinned byte-identical by
/// `arcgraph-storage::encryption::keysource`'s tests so a drift surfaces.
pub const ENCRYPTION_KEY_NAMESPACE_WAL: &str = "arcgraph.wal.encryption_key";

/// A typed key namespace — the scope a KEK governs (ADR-216 §D-1).
///
/// v1 has exactly one scope, [`KeyScope::wal`] →
/// `arcgraph.wal.encryption_key`. v1.1 extends with
/// `KeyScope::page_store()` (OQ-5) and optional per-tenant suffixing
/// (OQ-3). The namespace string is the KEK lookup key under the backing
/// [`SecretsProvider`](crate::secrets::SecretsProvider) (suffixed with the
/// KEK version at resolution time).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KeyScope {
    namespace: String,
}

impl KeyScope {
    /// The WAL-encryption scope (`arcgraph.wal.encryption_key`).
    ///
    /// Reuses [`ENCRYPTION_KEY_NAMESPACE_WAL`] (the same string
    /// `arcgraph-storage::encryption::keyring` uses for the per-record DEK
    /// namespace) so the KEK and the DEK share one coherent key-namespace
    /// convention.
    #[must_use]
    pub fn wal() -> Self {
        Self {
            namespace: ENCRYPTION_KEY_NAMESPACE_WAL.to_owned(),
        }
    }

    /// Construct a scope from an arbitrary namespace string. Used by v1.1
    /// extensions (`page_store`, per-tenant suffixing) and tests.
    #[must_use]
    pub fn from_namespace(namespace: impl Into<String>) -> Self {
        Self {
            namespace: namespace.into(),
        }
    }

    /// The namespace string. The KEK is stored/resolved under
    /// `<namespace>.kek.v<kek_version>` by the v1
    /// `SecretsProviderKeySource`.
    #[must_use]
    pub fn namespace(&self) -> &str {
        &self.namespace
    }
}

/// Monotonic per-scope KEK version. `u16` mirrors the existing
/// [`KeyVersion`](crate::secrets::KeyVersion) shape so the on-disk
/// wire-overhead stays 2 bytes and the rotation count budget matches
/// (65 535 rotations).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KekVersion(u16);

impl KekVersion {
    /// The first KEK version any scope carries after initial provisioning.
    pub const ONE: Self = Self(1);

    /// Construct from a raw `u16`.
    #[must_use]
    pub const fn new(raw: u16) -> Self {
        Self(raw)
    }

    /// Raw `u16` for encoding into the [`WrappedDek`] envelope + the AAD.
    #[must_use]
    pub const fn raw(self) -> u16 {
        self.0
    }

    /// Next version. Saturates at `u16::MAX` (an operator rotating past
    /// saturation has re-keyed from scratch / lifted to KMS already).
    #[must_use]
    pub const fn next(self) -> Self {
        Self(self.0.saturating_add(1))
    }
}

impl std::fmt::Display for KekVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "v{}", self.0)
    }
}

/// The algorithm used to wrap a DEK under a KEK (ADR-216 §D-1).
///
/// `#[non_exhaustive]`: a v1.1 KMS source wraps with the cloud KMS's
/// native algorithm (e.g. `kms:Encrypt`), which is added as a new variant
/// without breaking downstream matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum WrapAlg {
    /// In-process AES-256-GCM key-wrap of the 32-byte DEK under the KEK,
    /// with `key_source_id` + `kek_version` bound into the AAD. The v1
    /// `SecretsProviderKeySource` wrap algorithm.
    Aes256GcmKeyWrap,
}

impl WrapAlg {
    /// Stable 1-byte on-disk discriminant for the [`WrappedDek`] envelope.
    /// Distinct from `Display`/`Debug` so the wire format is independent of
    /// the human-readable name.
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        match self {
            Self::Aes256GcmKeyWrap => 1,
        }
    }

    /// Parse the 1-byte on-disk discriminant. Returns `None` for an
    /// unknown byte (a forward-incompatible / corrupted sidecar).
    #[must_use]
    pub const fn from_u8(b: u8) -> Option<Self> {
        match b {
            1 => Some(Self::Aes256GcmKeyWrap),
            _ => None,
        }
    }
}

/// The on-disk envelope: a DEK wrapped under a KEK (ADR-216 §D-1).
///
/// **The DEK plaintext is NEVER in this struct.** Only the wrapped bytes,
/// the KEK version used, the `key_source_id` that performed the wrap, and
/// the [`WrapAlg`]. Serialized into the WAL-encryption bootstrap sidecar
/// (`<data_dir>/wal/wal.dek`, ADR-216 §D-2) — NOT into each WAL record.
///
/// `key_source_id` + `kek_version` are bound into the wrap AAD (CF-2) so a
/// DEK wrapped under one source/version cannot be unwrapped as another, and
/// a tampered sidecar fails closed at `unwrap_dek` (structured error), not
/// silently honored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WrappedDek {
    /// The KEK version this DEK was wrapped under. `unwrap_dek` routes to
    /// the matching historical KEK version.
    pub kek_version: KekVersion,
    /// The `key_source_id` of the source that performed the wrap. Bound
    /// into the AAD; read back as UNTRUSTED at restart (CF-2).
    pub key_source_id: String,
    /// The wrapped DEK bytes (AES-256-GCM ciphertext + tag for the v1
    /// source). NEVER the plaintext DEK.
    pub wrapped_bytes: Vec<u8>,
    /// The algorithm used to wrap.
    pub wrap_alg: WrapAlg,
}

/// Every error a [`KeySource`] can produce (ADR-216 §D-1).
///
/// `#[non_exhaustive]` lets the v1.1 CMK/Vault/cloud-KMS impls add
/// KMS-shaped variants without breaking
/// downstream matches. Translated to
/// [`ArcGraphError::WalDecryptionFailed`](crate::ArcGraphError) /
/// startup `ArcGraphError::Io` at the storage boundary per the
/// codec-error-translation discipline.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum KeySourceError {
    /// The KEK for `scope` at `version` is not resolvable in the backing
    /// store/KMS. At `build_durable` this is a fail-closed startup error
    /// (ADR-033 — never serve plaintext WAL silently).
    #[error("KEK not found for scope {scope} at v{version}")]
    KekNotFound {
        /// The scope namespace whose KEK is missing.
        scope: String,
        /// The KEK version that could not be resolved.
        version: u16,
    },

    /// Wrapping a fresh DEK under the current KEK failed (AEAD seal error,
    /// or a backend `kms:Encrypt` failure at v1.1).
    #[error("DEK wrap failed: {reason}")]
    WrapFailed {
        /// Human-readable cause (MUST NOT include key bytes).
        reason: String,
    },

    /// Unwrapping a [`WrappedDek`] failed — AEAD tag mismatch (a tampered
    /// sidecar, CF-2), a `key_source_id`/`kek_version` AAD mismatch, or a
    /// truncated/corrupted envelope. ALWAYS fail-closed; NEVER returns a
    /// placeholder key.
    #[error("DEK unwrap failed: {reason}")]
    UnwrapFailed {
        /// Human-readable cause (MUST NOT include key bytes).
        reason: String,
    },

    /// The underlying [`SecretsProvider`](crate::secrets::SecretsProvider)
    /// (the KEK storage backend) rejected the lookup/store.
    #[error("key-source backend error: {source}")]
    Backend {
        /// The underlying provider error.
        #[source]
        source: SecretsError,
    },

    /// A remote KMS/Vault/HSM backend error. RESERVED for the v1.1
    /// CMK/Vault/cloud-KMS impls (named-not-stubbed per
    /// `feedback_avoid_speculative_scaffolding.md`); the v1
    /// `SecretsProviderKeySource` never produces it.
    #[error("KMS backend error from {provider}: {reason}")]
    Kms {
        /// The KMS/Vault provider id (e.g. "aws-kms", "vault").
        provider: String,
        /// Human-readable cause (MUST NOT include key bytes).
        reason: String,
    },

    /// The source does not support the requested operation (e.g.
    /// `rotate_kek` against a read-only env-backed dev source).
    #[error("key-source does not support {op}: {reason}")]
    Unsupported {
        /// The operation that was attempted.
        op: &'static str,
        /// Operator-facing rationale + lift target.
        reason: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// Compile-time pin: `dyn KeySource` is object-safe + Send + Sync +
    /// Debug. Without these the `Arc<dyn KeySource>` held in
    /// `build_durable` (and threaded into the WAL bootstrap dance) would
    /// not compile.
    #[test]
    fn key_source_is_object_safe_and_send_sync() {
        fn assert_object_safe(_: &dyn KeySource) {}
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Arc<dyn KeySource>>();
        // `assert_object_safe` is never called with a real value here
        // (the concrete source lives in arcgraph-storage); referencing it
        // forces the object-safety check at compile time.
        let _ = assert_object_safe;
    }

    #[test]
    fn key_scope_wal_reuses_namespace_constant() {
        assert_eq!(KeyScope::wal().namespace(), ENCRYPTION_KEY_NAMESPACE_WAL);
        assert_eq!(KeyScope::wal().namespace(), "arcgraph.wal.encryption_key");
    }

    #[test]
    fn kek_version_one_and_next() {
        assert_eq!(KekVersion::ONE.raw(), 1);
        assert_eq!(KekVersion::ONE.next().raw(), 2);
        assert_eq!(format!("{}", KekVersion::ONE), "v1");
        assert_eq!(KekVersion::new(u16::MAX).next(), KekVersion::new(u16::MAX));
    }

    #[test]
    fn wrap_alg_round_trips_through_u8() {
        assert_eq!(WrapAlg::Aes256GcmKeyWrap.as_u8(), 1);
        assert_eq!(WrapAlg::from_u8(1), Some(WrapAlg::Aes256GcmKeyWrap));
        assert_eq!(WrapAlg::from_u8(0), None);
        assert_eq!(WrapAlg::from_u8(255), None);
    }

    #[test]
    fn wrapped_dek_never_carries_plaintext_by_construction() {
        // A structural pin: the WrappedDek struct has no `plaintext` /
        // `dek` field. This test documents + enforces (via the field
        // list) that a DEK plaintext cannot be placed in the envelope.
        let w = WrappedDek {
            kek_version: KekVersion::ONE,
            key_source_id: "secrets-provider:env".to_owned(),
            wrapped_bytes: vec![0xAB; 48],
            wrap_alg: WrapAlg::Aes256GcmKeyWrap,
        };
        // The only byte vector is `wrapped_bytes` (ciphertext), never a
        // plaintext key. Equality round-trips for the sidecar codec.
        assert_eq!(w.clone(), w);
        assert_eq!(w.wrapped_bytes.len(), 48);
    }

    #[test]
    fn key_source_error_display_names_scope_and_version() {
        let e = KeySourceError::KekNotFound {
            scope: "arcgraph.wal.encryption_key".to_owned(),
            version: 3,
        };
        let s = format!("{e}");
        assert!(s.contains("arcgraph.wal.encryption_key"), "got: {s}");
        assert!(s.contains("v3"), "got: {s}");
    }
}
