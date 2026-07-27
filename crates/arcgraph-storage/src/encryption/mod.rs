//! AES-256-GCM encryption primitives for ArcGraph storage (W20β-3 / ADR-052).
//!
//! Three composing layers in the original ADR-052 design; the v1.0-α
//! narrowing in PR #373 R1 ships layers 1 + 2 only and defers layer 3
//! to v1.1:
//!
//! 1. [`cipher`] — thin `Aes256GcmCipher` over `aws-lc-rs::aead`. The
//!    only module that knows about AEAD primitives; everything else
//!    is layout + key plumbing.
//! 2. [`keyring`] — `KeyRing` that maps `KeyVersion` → cached cipher
//!    instance, fronted by a `SecretsProvider`. WAL encryption + the
//!    v1.1 page-store path will both consume this.
//! 3. [`wal`] — payload-level encryption with the on-disk magic +
//!    IV + tag layout documented in ADR-052 §Decision.
//!
//! **Page-store encryption (former §3 of ADR-052) is DEFERRED to v1.1.**
//! The `EncryptedFilePageIo` impl + its in-file tests were removed by
//! PR #373 R1 fix-up per `feedback_avoid_speculative_scaffolding.md`
//! — the surface had zero production consumers AND the deterministic
//! IV scheme (Joux 2006 nonce-reuse on rewritten pages) needs a
//! redesign. v1.1 reintroduces page-store encryption with a corrected
//! IV scheme that is safe for rewritten pages.
//!
//! ## Threading
//!
//! `KeyRing` + `Aes256GcmCipher` are both `Send + Sync` — the WAL
//! writer thread holds one across its lifetime. Mutex-protected
//! internal state (cached keys) is fine-grained enough not to
//! serialize the hot path — ADR-052 §Open questions discusses the
//! lock-free cache lift if profiling shows contention at v1.0 GA.
//!
//! ## Bounded-context discipline
//!
//! Under design-v2 §3.4's bounded-context rule, storage owns ALL I/O. The
//! encryption module deliberately depends ONLY on `arcgraph-core`'s
//! `SecretsProvider` trait + `aws-lc-rs`'s AEAD primitives — no
//! tokio, no async runtime. The WAL writer thread (sync, OS-thread)
//! drives encryption synchronously.

pub mod cipher;
pub mod keyring;
// ADR-216 §D-2: the v1 concrete `KeySource` (`SecretsProviderKeySource`)
// performing the in-process AES-256-GCM DEK key-wrap + the `wal.dek`
// envelope sidecar codec. Lives in storage (not core) because the wrap
// uses the storage-owned `Aes256GcmCipher` (core carries no `aws-lc-rs`);
// the trait + envelope types live in `arcgraph-core::secrets`.
pub mod keysource;
pub mod wal;

pub use cipher::{AEAD_KEY_LEN, AES_GCM_IV_LEN, AES_GCM_TAG_LEN, Aes256GcmCipher, CipherError};
pub use keyring::{ENCRYPTION_KEY_NAMESPACE_WAL, KeyRing, KeyRingError, install_random_key};
pub use keysource::{
    SECRETS_PROVIDER_KEY_SOURCE_PREFIX, SecretsProviderKeySource, SidecarCodecError,
    SidecarIoError, WAL_DEK_SIDECAR_FILE, WalDekSidecar, WalEncryptionBootstrap,
    WalEncryptionBootstrapError, bootstrap_wal_encryption, sidecar_path,
};
pub use wal::{
    PayloadEncryption, WAL_ENCRYPTION_MAGIC, WAL_PAYLOAD_HEADER_LEN, WalEncryption,
    decrypt_wal_payload, encrypt_wal_payload, is_encrypted_wal_payload,
};
