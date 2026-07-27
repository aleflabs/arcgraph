//! The v1 concrete [`KeySource`] — `SecretsProviderKeySource` — plus the
//! `wal.dek` envelope sidecar codec (ADR-216 §D-2 / §D-3).
//!
//! ## Bounded-context resolution (ADR-216 §D-1 + #1180)
//!
//! Per ADR-216 §D-1 the `KeySource` *trait* + the on-disk envelope TYPES
//! ([`WrappedDek`], [`KeyScope`], [`KekVersion`], [`WrapAlg`],
//! [`KeySourceError`]) live in `arcgraph-core::secrets` — the
//! key-management abstraction is a core bounded-context concern (sibling of
//! `SecretsProvider`). The ONE v1 concrete source —
//! [`SecretsProviderKeySource`] — lives HERE in `arcgraph-storage` because
//! the wrap is an in-process AES-256-GCM key-wrap performed via
//! [`super::cipher::Aes256GcmCipher`], and `arcgraph-core` deliberately
//! carries NO `aws-lc-rs` dependency (core stays I/O-free + crypto-free).
//! This is the clean dependency direction: storage CONSUMES the core trait,
//! never the reverse (ADR-216 "What it locks in" #4).
//!
//! ## Envelope wrap (ADR-216 §D-3, CF-2, CF-3)
//!
//! - **DEK** = a fresh 32-byte AES-256 key from the CSPRNG (CF-3: the same
//!   `/dev/urandom` path the existing `install_random_key` uses; ADR-051
//!   §"Decision item 4" — no DIY entropy).
//! - **KEK** = a 32-byte [`SecretValue`] fetched from the backing
//!   [`SecretsProvider`] at `<scope.namespace>.kek.v<kek_version>`.
//! - **Wrap** = `AES-256-GCM(KEK).seal(DEK)` with a fresh random 12-byte IV
//!   PREPENDED to the wrapped bytes, and the wrap AAD binding
//!   `key_source_id` + `kek_version` (CF-2 anti-tamper). On `unwrap_dek`
//!   the `key_source_id`/`kek_version` read from the (untrusted) sidecar
//!   are bound into the unwrap AAD — a tampered sidecar fails CLOSED with a
//!   structured [`KeySourceError::UnwrapFailed`], NOT silently honored.
//!   Mirrors `wal.rs::build_aad`'s `(key_version, segment_no, lsn)` binding.
//!
//! ## `wal.dek` sidecar (ADR-216 §D-2 / OQ-1)
//!
//! The sidecar at `<data_dir>/wal/wal.dek` holds a SET of [`WrappedDek`]
//! keyed by [`KeyVersion`] (the per-record WAL key version), so DEK
//! rotation appends a new wrapped DEK while history stays decryptable. The
//! plaintext DEK is NEVER written — only [`WrappedDek`] (the wrapped form).
//! Writes are atomic (temp-file + fsync + rename, OQ-1): a torn rewrite
//! leaves the OLD sidecar readable (err toward old-readable).

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arcgraph_core::{
    KekVersion, KeyScope, KeySource, KeySourceError, KeyVersion, SECRET_VALUE_LEN, SecretValue,
    SecretsError, SecretsProvider, WrapAlg, WrappedDek,
};

use super::cipher::{AES_GCM_IV_LEN, AES_GCM_TAG_LEN, Aes256GcmCipher};

/// File name of the wrapped-DEK envelope sidecar inside `<data_dir>/wal/`.
pub const WAL_DEK_SIDECAR_FILE: &str = "wal.dek";

/// The `key_source_id` prefix for [`SecretsProviderKeySource`]. The full id
/// is `secrets-provider:<provider-tag>` (e.g. `secrets-provider:os-keyring`
/// / `secrets-provider:env`). Bound into the wrap AAD (CF-2).
pub const SECRETS_PROVIDER_KEY_SOURCE_PREFIX: &str = "secrets-provider";

// ─────────────────────────────────────────────────────────────────────────
// SecretsProviderKeySource — the v1 concrete KeySource (ADR-216 §D-2)
// ─────────────────────────────────────────────────────────────────────────

/// The v1 production [`KeySource`]: resolves the KEK from a
/// [`SecretsProvider`] (OS keyring in prod, env in dev) and performs an
/// in-process AES-256-GCM key-wrap of the DEK under the KEK.
///
/// The KEK is a 32-byte [`SecretValue`] stored at
/// `<scope.namespace>.kek.v<kek_version>` in the provider. The DEK is
/// generated locally via CSPRNG (CF-3), wrapped, and returned as a
/// [`WrappedDek`] (never as plaintext). See the module docs for the
/// bounded-context rationale (this source lives in `arcgraph-storage`
/// because it owns the cipher).
pub struct SecretsProviderKeySource {
    provider: Arc<dyn SecretsProvider>,
    /// Stable audit id, bound into the wrap AAD (CF-2). e.g.
    /// `secrets-provider:os-keyring`.
    key_source_id: String,
    /// The KEK version new wraps use. KEK rotation bumps this; historical
    /// versions stay resolvable from the provider for `unwrap_dek`.
    current_kek_version: KekVersion,
}

impl std::fmt::Debug for SecretsProviderKeySource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SecretsProviderKeySource")
            .field("key_source_id", &self.key_source_id)
            .field("current_kek_version", &self.current_kek_version)
            .finish_non_exhaustive()
    }
}

impl SecretsProviderKeySource {
    /// Construct from a backing [`SecretsProvider`] + a provider tag (e.g.
    /// `"os-keyring"` / `"env"`) used to build the stable `key_source_id`.
    /// `current_kek_version` is the version new DEK wraps are performed
    /// under (typically [`KekVersion::ONE`] on first boot).
    #[must_use]
    pub fn new(
        provider: Arc<dyn SecretsProvider>,
        provider_tag: &str,
        current_kek_version: KekVersion,
    ) -> Self {
        Self {
            provider,
            key_source_id: format!("{SECRETS_PROVIDER_KEY_SOURCE_PREFIX}:{provider_tag}"),
            current_kek_version,
        }
    }

    /// The provider lookup key for the KEK at `kek_version` under `scope`:
    /// `<scope.namespace>.kek.v<kek_version>`. Distinct from the per-record
    /// DEK namespace (`<scope.namespace>.v<key_version>`) so the KEK and the
    /// DEK never collide in the provider's flat key space.
    fn kek_provider_key(scope: &KeyScope, kek_version: KekVersion) -> String {
        format!("{}.kek.v{}", scope.namespace(), kek_version.raw())
    }

    /// Resolve the KEK [`SecretValue`] for `scope` at `kek_version`,
    /// translating provider errors into [`KeySourceError`].
    fn resolve_kek(
        &self,
        scope: &KeyScope,
        kek_version: KekVersion,
    ) -> Result<SecretValue, KeySourceError> {
        let key = Self::kek_provider_key(scope, kek_version);
        self.provider.get(&key).map_err(|source| match source {
            SecretsError::NotFound { .. } => KeySourceError::KekNotFound {
                scope: scope.namespace().to_owned(),
                version: kek_version.raw(),
            },
            other => KeySourceError::Backend { source: other },
        })
    }

    /// Build the wrap AAD binding `key_source_id` + `kek_version` (CF-2).
    /// On `unwrap_dek` the `key_source_id`/`kek_version` are read from the
    /// (UNTRUSTED) sidecar and fed here, so a tampered sidecar produces a
    /// different AAD → AEAD tag mismatch → fail-closed.
    ///
    /// Layout: `[key_source_id bytes] [0x00 separator] [kek_version u16 LE]`.
    /// The `0x00` separator + the trailing fixed-width version prevent a
    /// prefix-collision attack (a `key_source_id` that is a prefix of
    /// another cannot produce the same AAD as the longer id).
    fn build_wrap_aad(key_source_id: &str, kek_version: KekVersion) -> Vec<u8> {
        let id_bytes = key_source_id.as_bytes();
        let mut aad = Vec::with_capacity(id_bytes.len() + 1 + 2);
        aad.extend_from_slice(id_bytes);
        aad.push(0x00);
        aad.extend_from_slice(&kek_version.raw().to_le_bytes());
        aad
    }

    /// Wrap an existing 32-byte DEK under the current KEK for `scope`. The
    /// shared core of [`KeySource::generate_wrapped_dek`] (which wraps a
    /// freshly-generated DEK) and [`Self::rewrap_dek`] (which re-wraps an
    /// existing DEK under a new KEK on rotation).
    ///
    /// A fresh random IV is read per wrap (CF-3) and PREPENDED to the
    /// wrapped bytes; the wrap AAD binds `key_source_id` + `kek_version`
    /// (CF-2).
    fn wrap_dek(&self, scope: &KeyScope, dek: &SecretValue) -> Result<WrappedDek, KeySourceError> {
        let kek = self.resolve_kek(scope, self.current_kek_version)?;
        let cipher = Aes256GcmCipher::from_key(kek.expose_bytes()).map_err(|e| {
            KeySourceError::WrapFailed {
                reason: format!("KEK cipher init: {e}"),
            }
        })?;

        // Fresh random IV per wrap (CF-3) — a unique IV removes any
        // nonce-reuse concern even across re-wraps of the same DEK on KEK
        // rotation.
        let iv = read_csprng_iv().map_err(|reason| KeySourceError::WrapFailed { reason })?;
        let aad = Self::build_wrap_aad(&self.key_source_id, self.current_kek_version);
        let ct_with_tag = cipher.encrypt(&iv, &aad, dek.expose_bytes()).map_err(|e| {
            KeySourceError::WrapFailed {
                reason: format!("DEK seal: {e}"),
            }
        })?;

        // wrapped_bytes layout: [iv (12B)] [ciphertext+tag].
        let mut wrapped_bytes = Vec::with_capacity(AES_GCM_IV_LEN + ct_with_tag.len());
        wrapped_bytes.extend_from_slice(&iv);
        wrapped_bytes.extend_from_slice(&ct_with_tag);

        Ok(WrappedDek {
            kek_version: self.current_kek_version,
            key_source_id: self.key_source_id.clone(),
            wrapped_bytes,
            wrap_alg: WrapAlg::Aes256GcmKeyWrap,
        })
    }

    /// Re-wrap an EXISTING DEK under this source's current KEK version
    /// (ADR-216 §D-3 KEK rotation: "re-wrap the existing DEK under the new
    /// KEK and overwrite `wal.dek`"). The DEK is UNCHANGED, so every
    /// historical + future WAL record stays decryptable — KEK rotation is
    /// O(1) over one small sidecar file, never O(WAL-size).
    ///
    /// The typical rotation sequence: (1) provision a new KEK version
    /// ([`KeySource::rotate_kek`]); (2) unwrap each DEK from `wal.dek` under
    /// the old KEK; (3) `rewrap_dek` it under the new-KEK source; (4)
    /// atomically overwrite `wal.dek` with the re-wrapped envelopes.
    ///
    /// # Errors
    ///
    /// [`KeySourceError`] if the current KEK is unresolvable or the AEAD
    /// seal fails.
    pub fn rewrap_dek(
        &self,
        scope: &KeyScope,
        dek: &SecretValue,
    ) -> Result<WrappedDek, KeySourceError> {
        self.wrap_dek(scope, dek)
    }
}

impl KeySource for SecretsProviderKeySource {
    fn key_source_id(&self) -> &str {
        &self.key_source_id
    }

    fn generate_wrapped_dek(&self, scope: &KeyScope) -> Result<WrappedDek, KeySourceError> {
        // CF-3 (ADR-051 §"Decision item 4"): the fresh DEK comes from the
        // existing Unix `/dev/urandom` CSPRNG path — NO DIY entropy. The
        // cross-platform lift (OQ-6) selects `getrandom`/`aws-lc-rs`
        // `SystemRandom` per the same ADR-051 §item-4 well-known-crate
        // table; v1 reuses the proven Unix read.
        let dek_bytes =
            read_csprng_dek().map_err(|reason| KeySourceError::WrapFailed { reason })?;
        let dek = SecretValue::new(dek_bytes);
        self.wrap_dek(scope, &dek)
    }

    fn unwrap_dek(&self, wrapped: &WrappedDek) -> Result<SecretValue, KeySourceError> {
        if wrapped.wrap_alg != WrapAlg::Aes256GcmKeyWrap {
            return Err(KeySourceError::UnwrapFailed {
                reason: format!("unsupported wrap_alg {:?}", wrapped.wrap_alg),
            });
        }
        if wrapped.wrapped_bytes.len() < AES_GCM_IV_LEN + AES_GCM_TAG_LEN {
            return Err(KeySourceError::UnwrapFailed {
                reason: format!(
                    "wrapped DEK too short: {} bytes (need >= {})",
                    wrapped.wrapped_bytes.len(),
                    AES_GCM_IV_LEN + AES_GCM_TAG_LEN
                ),
            });
        }

        // Route to the KEK version stamped in the wrapped envelope.
        let kek = self.resolve_kek(&KeyScope::wal(), wrapped.kek_version)?;
        let cipher = Aes256GcmCipher::from_key(kek.expose_bytes()).map_err(|e| {
            KeySourceError::UnwrapFailed {
                reason: format!("KEK cipher init: {e}"),
            }
        })?;

        let mut iv = [0u8; AES_GCM_IV_LEN];
        iv.copy_from_slice(&wrapped.wrapped_bytes[..AES_GCM_IV_LEN]);
        let ct_with_tag = &wrapped.wrapped_bytes[AES_GCM_IV_LEN..];

        // CF-2: bind the `key_source_id` + `kek_version` READ FROM THE
        // (untrusted) sidecar into the unwrap AAD. A tampered sidecar
        // (swapped id or version) yields a different AAD → AEAD tag
        // mismatch → fail-closed UnwrapFailed (never silently honored).
        let aad = Self::build_wrap_aad(&wrapped.key_source_id, wrapped.kek_version);
        let dek_bytes =
            cipher
                .decrypt(&iv, &aad, ct_with_tag)
                .map_err(|e| KeySourceError::UnwrapFailed {
                    reason: format!("DEK open: {e}"),
                })?;

        let dek = SecretValue::try_from_slice(&dek_bytes).ok_or_else(|| {
            KeySourceError::UnwrapFailed {
                reason: format!(
                    "unwrapped DEK has wrong length: {} (expected {})",
                    dek_bytes.len(),
                    SECRET_VALUE_LEN
                ),
            }
        })?;
        Ok(dek)
    }

    fn rotate_kek(&self, scope: &KeyScope) -> Result<KekVersion, KeySourceError> {
        // Rotation provisions a NEW KEK version in the provider. The v1
        // source delegates to `SecretsProvider::rotate`/`set`: a provider
        // that supports rotation (OsKeyringProvider) installs the new KEK;
        // an env-backed dev provider returns Unsupported (operators rotate
        // by updating systemd/k8s env + restart). The NEW KEK is installed
        // at `<namespace>.kek.v<next>` so historical versions stay
        // resolvable for `unwrap_dek` (ADR-216 §D-3 — old KEKs remain
        // until operator policy destroys them).
        let next = self.current_kek_version.next();
        let next_key = Self::kek_provider_key(scope, next);
        let next_kek_bytes =
            read_csprng_dek().map_err(|reason| KeySourceError::WrapFailed { reason })?;
        self.provider
            .set(&next_key, SecretValue::new(next_kek_bytes))
            .map_err(|source| match source {
                SecretsError::Unsupported { reason, .. } => KeySourceError::Unsupported {
                    op: "rotate_kek",
                    reason,
                },
                other => KeySourceError::Backend { source: other },
            })?;
        Ok(next)
    }

    fn health_check(&self, scope: &KeyScope) -> Result<(), KeySourceError> {
        // Fail-fast probe: the current KEK MUST be resolvable. A miss is a
        // hard startup error (ADR-033 — never serve plaintext WAL silently).
        let _ = self.resolve_kek(scope, self.current_kek_version)?;
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────
// wal.dek sidecar codec (ADR-216 §D-2 / OQ-1)
// ─────────────────────────────────────────────────────────────────────────

/// 4-byte magic at the head of every `wal.dek` sidecar file.
const SIDECAR_MAGIC: [u8; 4] = *b"ADEK"; // "Arcgraph DEK"
/// On-disk sidecar format version (bumped on any layout change).
const SIDECAR_FORMAT_VERSION: u8 = 1;

/// The in-memory model of the `wal.dek` sidecar: a SET of [`WrappedDek`]
/// keyed by the per-record WAL [`KeyVersion`] (ADR-216 §D-2).
///
/// DEK rotation inserts a new entry (a new `key_version` → its wrapped
/// DEK); history stays decryptable because the existing per-record
/// `key_version` routing in `WalEncryption`/`KeyRing` consults the matching
/// historical DEK. KEK rotation REPLACES the `WrappedDek` for an existing
/// `key_version` (same DEK, re-wrapped under the new KEK) — no WAL changes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WalDekSidecar {
    entries: BTreeMap<u16, WrappedDek>,
}

impl WalDekSidecar {
    /// An empty sidecar (no DEKs yet — first boot before
    /// `generate_wrapped_dek`).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert / replace the wrapped DEK for `key_version`. Replacing is the
    /// KEK-rotation path (same DEK, new wrap); inserting a new version is
    /// the DEK-rotation path.
    pub fn insert(&mut self, key_version: KeyVersion, wrapped: WrappedDek) {
        self.entries.insert(key_version.raw(), wrapped);
    }

    /// The wrapped DEK for `key_version`, if present.
    #[must_use]
    pub fn get(&self, key_version: KeyVersion) -> Option<&WrappedDek> {
        self.entries.get(&key_version.raw())
    }

    /// Number of wrapped DEKs held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` iff no DEK is held yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The highest `key_version` present (the current write version), if
    /// any. Used at restart to set `WalEncryption`'s current version.
    #[must_use]
    pub fn max_key_version(&self) -> Option<KeyVersion> {
        self.entries.keys().next_back().map(|v| KeyVersion::new(*v))
    }

    /// Iterate `(KeyVersion, &WrappedDek)` in ascending version order.
    pub fn iter(&self) -> impl Iterator<Item = (KeyVersion, &WrappedDek)> {
        self.entries.iter().map(|(v, w)| (KeyVersion::new(*v), w))
    }

    /// Encode the sidecar to a length-prefixed byte blob (ADR-216 OQ-1
    /// "simple bincode/length-prefixed format"). All multi-byte fields are
    /// little-endian.
    ///
    /// ```text
    ///   magic            4   b"ADEK"
    ///   format_version   1   u8 (= 1)
    ///   entry_count      4   u32 LE
    ///   per entry:
    ///     key_version    2   u16 LE   (the WAL per-record version)
    ///     kek_version    2   u16 LE
    ///     wrap_alg       1   u8       (WrapAlg discriminant)
    ///     id_len         4   u32 LE
    ///     key_source_id  id_len bytes (UTF-8)
    ///     wrapped_len    4   u32 LE
    ///     wrapped_bytes  wrapped_len bytes
    /// ```
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(9 + self.entries.len() * 64);
        out.extend_from_slice(&SIDECAR_MAGIC);
        out.push(SIDECAR_FORMAT_VERSION);
        out.extend_from_slice(&(self.entries.len() as u32).to_le_bytes());
        for (key_version, wrapped) in &self.entries {
            out.extend_from_slice(&key_version.to_le_bytes());
            out.extend_from_slice(&wrapped.kek_version.raw().to_le_bytes());
            out.push(wrapped.wrap_alg.as_u8());
            let id_bytes = wrapped.key_source_id.as_bytes();
            out.extend_from_slice(&(id_bytes.len() as u32).to_le_bytes());
            out.extend_from_slice(id_bytes);
            out.extend_from_slice(&(wrapped.wrapped_bytes.len() as u32).to_le_bytes());
            out.extend_from_slice(&wrapped.wrapped_bytes);
        }
        out
    }

    /// Decode a sidecar blob produced by [`Self::encode`]. Returns a
    /// structured [`SidecarCodecError`] on any malformed input (bad magic,
    /// unknown version, truncation, unknown `wrap_alg`).
    pub fn decode(bytes: &[u8]) -> Result<Self, SidecarCodecError> {
        let mut cur = Cursor { bytes, pos: 0 };
        let magic = cur.take(4)?;
        if magic != SIDECAR_MAGIC {
            return Err(SidecarCodecError::BadMagic);
        }
        let version = cur.take(1)?[0];
        if version != SIDECAR_FORMAT_VERSION {
            return Err(SidecarCodecError::UnknownVersion(version));
        }
        let entry_count = cur.take_u32()? as usize;
        let mut entries = BTreeMap::new();
        for _ in 0..entry_count {
            let key_version = cur.take_u16()?;
            let kek_version = KekVersion::new(cur.take_u16()?);
            let wrap_alg_byte = cur.take(1)?[0];
            let wrap_alg = WrapAlg::from_u8(wrap_alg_byte)
                .ok_or(SidecarCodecError::UnknownWrapAlg(wrap_alg_byte))?;
            let id_len = cur.take_u32()? as usize;
            let id_bytes = cur.take(id_len)?;
            let key_source_id = String::from_utf8(id_bytes.to_vec())
                .map_err(|_| SidecarCodecError::InvalidUtf8KeySourceId)?;
            let wrapped_len = cur.take_u32()? as usize;
            let wrapped_bytes = cur.take(wrapped_len)?.to_vec();
            entries.insert(
                key_version,
                WrappedDek {
                    kek_version,
                    key_source_id,
                    wrapped_bytes,
                    wrap_alg,
                },
            );
        }
        Ok(Self { entries })
    }

    /// Read + decode the sidecar at `<wal_dir>/wal.dek`. Returns `Ok(None)`
    /// if the sidecar does not exist (first boot).
    pub fn read_from_dir(wal_dir: &Path) -> Result<Option<Self>, SidecarIoError> {
        let path = sidecar_path(wal_dir);
        match std::fs::read(&path) {
            Ok(bytes) => Self::decode(&bytes)
                .map(Some)
                .map_err(|source| SidecarIoError::Decode { path, source }),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(source) => Err(SidecarIoError::Read { path, source }),
        }
    }

    /// Atomically (re)write the sidecar at `<wal_dir>/wal.dek` (ADR-216
    /// OQ-1: temp-file + fsync + rename). A torn rewrite leaves the OLD
    /// sidecar readable (the rename is the atomic commit point; a crash
    /// before rename leaves the temp file as garbage but the real file
    /// untouched) — err toward old-readable.
    ///
    /// NEVER writes a plaintext DEK/KEK: the only bytes on disk are the
    /// [`WrappedDek`] envelopes (wrapped form).
    pub fn write_to_dir(&self, wal_dir: &Path) -> Result<(), SidecarIoError> {
        let path = sidecar_path(wal_dir);
        let tmp_path = path.with_extension("dek.tmp");
        let encoded = self.encode();

        // 1. Write the full blob to a temp file + fsync it durable.
        {
            let mut f =
                std::fs::File::create(&tmp_path).map_err(|source| SidecarIoError::Write {
                    path: tmp_path.clone(),
                    source,
                })?;
            f.write_all(&encoded)
                .map_err(|source| SidecarIoError::Write {
                    path: tmp_path.clone(),
                    source,
                })?;
            f.sync_all().map_err(|source| SidecarIoError::Write {
                path: tmp_path.clone(),
                source,
            })?;
        }
        // 2. Atomic rename — the commit point. POSIX rename is atomic, so a
        //    reader sees either the OLD file or the FULLY-written new one,
        //    never a torn blend.
        std::fs::rename(&tmp_path, &path).map_err(|source| SidecarIoError::Write {
            path: path.clone(),
            source,
        })?;
        // 3. fsync the directory so the rename itself is durable (else a
        //    crash after rename but before the dir entry hits disk could
        //    lose the rename — leaving the OLD file, still readable: the
        //    SAFE direction per OQ-1).
        if let Some(parent) = path.parent() {
            if let Ok(dir) = std::fs::File::open(parent) {
                let _ = dir.sync_all();
            }
        }
        Ok(())
    }
}

/// Path of the `wal.dek` sidecar inside a WAL directory.
#[must_use]
pub fn sidecar_path(wal_dir: &Path) -> PathBuf {
    wal_dir.join(WAL_DEK_SIDECAR_FILE)
}

/// A tiny forward-only byte cursor for the sidecar decoder.
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], SidecarCodecError> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or(SidecarCodecError::Truncated)?;
        if end > self.bytes.len() {
            return Err(SidecarCodecError::Truncated);
        }
        let out = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn take_u16(&mut self) -> Result<u16, SidecarCodecError> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn take_u32(&mut self) -> Result<u32, SidecarCodecError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
}

/// Errors decoding a `wal.dek` sidecar blob.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SidecarCodecError {
    /// The 4-byte magic header did not match `b"ADEK"`.
    #[error("wal.dek sidecar: bad magic (not an ADEK envelope)")]
    BadMagic,
    /// The format-version byte is not a version this build understands.
    #[error("wal.dek sidecar: unknown format version {0}")]
    UnknownVersion(u8),
    /// The blob ended mid-field (a torn / truncated sidecar).
    #[error("wal.dek sidecar: truncated blob")]
    Truncated,
    /// An entry's `wrap_alg` discriminant is not a known [`WrapAlg`].
    #[error("wal.dek sidecar: unknown wrap_alg discriminant {0}")]
    UnknownWrapAlg(u8),
    /// A `key_source_id` field is not valid UTF-8.
    #[error("wal.dek sidecar: key_source_id is not valid UTF-8")]
    InvalidUtf8KeySourceId,
}

/// Errors reading / writing the `wal.dek` sidecar file.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SidecarIoError {
    /// Reading the sidecar file failed (permissions, IO).
    #[error("wal.dek sidecar read failed at {path}: {source}")]
    Read {
        /// The sidecar path.
        path: PathBuf,
        /// The underlying IO error.
        #[source]
        source: std::io::Error,
    },
    /// Writing / atomically renaming the sidecar failed.
    #[error("wal.dek sidecar write failed at {path}: {source}")]
    Write {
        /// The sidecar (or temp) path.
        path: PathBuf,
        /// The underlying IO error.
        #[source]
        source: std::io::Error,
    },
    /// The on-disk sidecar blob is malformed.
    #[error("wal.dek sidecar decode failed at {path}: {source}")]
    Decode {
        /// The sidecar path.
        path: PathBuf,
        /// The codec error.
        #[source]
        source: SidecarCodecError,
    },
}

// ─────────────────────────────────────────────────────────────────────────
// CSPRNG (CF-3 / ADR-051 §"Decision item 4")
// ─────────────────────────────────────────────────────────────────────────

/// Read 32 bytes of DEK/KEK material from the OS CSPRNG.
///
/// CF-3 (ADR-051 §"Decision item 4"): the entropy source is the existing
/// Unix `/dev/urandom` read (the same path `install_random_key` already
/// uses) — NO DIY / ad-hoc entropy. The cross-platform lift (OQ-6) selects
/// `getrandom`/`aws-lc-rs` `SystemRandom` per the ADR-051 §item-4
/// well-known-crate table; v1 reuses the proven Unix read.
#[cfg(unix)]
fn read_csprng_dek() -> Result<[u8; SECRET_VALUE_LEN], String> {
    use std::io::Read;
    let mut bytes = [0u8; SECRET_VALUE_LEN];
    let mut f = std::fs::File::open("/dev/urandom")
        .map_err(|e| format!("open /dev/urandom failed: {e}"))?;
    f.read_exact(&mut bytes)
        .map_err(|e| format!("read /dev/urandom failed: {e}"))?;
    Ok(bytes)
}

#[cfg(not(unix))]
fn read_csprng_dek() -> Result<[u8; SECRET_VALUE_LEN], String> {
    Err("CSPRNG read unsupported on non-Unix at v1; cross-platform lift is OQ-6".to_owned())
}

/// Read a fresh 12-byte AES-GCM IV from the OS CSPRNG (CF-3, same source as
/// [`read_csprng_dek`]).
///
/// `pub(crate)` so the per-record WAL write path
/// ([`super::wal::WalEncryption::encrypt`]) can draw a fresh random nonce
/// from the SAME proven `/dev/urandom` source — no DIY entropy, no second
/// CSPRNG code path (#1111 fix, ADR-051 §"Decision item 4" / ADR-052).
#[cfg(unix)]
pub(crate) fn read_csprng_iv() -> Result<[u8; AES_GCM_IV_LEN], String> {
    use std::io::Read;
    let mut bytes = [0u8; AES_GCM_IV_LEN];
    let mut f = std::fs::File::open("/dev/urandom")
        .map_err(|e| format!("open /dev/urandom failed: {e}"))?;
    f.read_exact(&mut bytes)
        .map_err(|e| format!("read /dev/urandom failed: {e}"))?;
    Ok(bytes)
}

#[cfg(not(unix))]
pub(crate) fn read_csprng_iv() -> Result<[u8; AES_GCM_IV_LEN], String> {
    Err("CSPRNG read unsupported on non-Unix at v1; cross-platform lift is OQ-6".to_owned())
}

// ─────────────────────────────────────────────────────────────────────────
// build_durable bootstrap dance (ADR-216 §D-2 / §D-4)
// ─────────────────────────────────────────────────────────────────────────

/// In-memory [`SecretsProvider`] holding UNWRAPPED DEKs in process memory
/// only (never on disk), keyed by the per-record WAL key namespace
/// (`arcgraph.wal.encryption_key.v<N>`) so the existing
/// [`WalEncryption`](super::wal::WalEncryption)/[`KeyRing`](super::keyring::KeyRing)
/// can consume them unchanged.
///
/// This is the crux of ADR-216 §D-2 step 4: the `KeySource` unwraps each
/// DEK from the sidecar into a plaintext [`SecretValue`] (zeroized on drop)
/// held here; `WalEncryption::new(Arc::new(this), version)` then reads the
/// DEK exactly as it reads any provider-backed key. The DEK plaintext lives
/// ONLY in these `SecretValue`s in RAM — it is never persisted (the
/// `wal.dek` sidecar holds the WRAPPED form).
#[derive(Debug)]
struct InMemoryDekProvider {
    /// `arcgraph.wal.encryption_key.v<N>` → unwrapped 32-byte DEK.
    keys: std::collections::HashMap<String, SecretValue>,
}

impl SecretsProvider for InMemoryDekProvider {
    fn get(&self, key: &str) -> Result<SecretValue, SecretsError> {
        self.keys
            .get(key)
            .cloned()
            .ok_or_else(|| SecretsError::NotFound {
                key: key.to_owned(),
            })
    }

    fn set(&self, key: &str, _value: SecretValue) -> Result<(), SecretsError> {
        Err(SecretsError::Unsupported {
            operation: "set",
            key: key.to_owned(),
            reason: "InMemoryDekProvider is read-only: DEKs are unwrapped \
                     from the wal.dek sidecar at bootstrap; rotation goes \
                     through the KeySource + sidecar, not this provider"
                .to_owned(),
        })
    }

    fn rotate(&self, key: &str) -> Result<KeyVersion, SecretsError> {
        Err(SecretsError::Unsupported {
            operation: "rotate",
            key: key.to_owned(),
            reason: "InMemoryDekProvider is read-only; rotate via the \
                     KeySource envelope path"
                .to_owned(),
        })
    }

    fn delete(&self, key: &str) -> Result<(), SecretsError> {
        Err(SecretsError::Unsupported {
            operation: "delete",
            key: key.to_owned(),
            reason: "InMemoryDekProvider is read-only".to_owned(),
        })
    }
}

/// Errors from the `build_durable` WAL-encryption bootstrap dance.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum WalEncryptionBootstrapError {
    /// The `KeySource` could not resolve / wrap / unwrap a key (fail-closed).
    #[error("key-source error during WAL-encryption bootstrap: {0}")]
    KeySource(#[from] KeySourceError),
    /// Reading / writing the `wal.dek` sidecar failed.
    #[error("wal.dek sidecar IO error during WAL-encryption bootstrap: {0}")]
    Sidecar(#[from] SidecarIoError),
    /// Constructing [`WalEncryption`](super::wal::WalEncryption) from the
    /// unwrapped DEK(s) failed (e.g. a cipher-init logic bug).
    #[error("WalEncryption construction failed: {0}")]
    WalEncryption(#[from] super::keyring::KeyRingError),
}

/// The result of [`bootstrap_wal_encryption`]: the encryption config to wire
/// into the WAL writer + recovery, plus whether the sidecar was freshly
/// generated (first boot) vs read from disk (restart).
#[derive(Debug)]
pub struct WalEncryptionBootstrap {
    /// The encryption config to pass to both
    /// `WalConfig::with_encryption(..)` AND every recovery
    /// `WalRecoveryReader::with_encryption(..)` (load-bearing: encrypt-on-
    /// write without decrypt-on-recover is unrecoverable WAL).
    pub encryption: super::wal::WalEncryption,
    /// The DEK key version new WAL records are stamped with.
    pub current_key_version: KeyVersion,
    /// `true` iff this boot generated a fresh DEK + sidecar (no prior
    /// `wal.dek` existed); `false` on a restart that read the sidecar.
    pub freshly_generated: bool,
}

/// The ADR-216 §D-2 bootstrap-sidecar dance: read-or-generate `wal.dek`,
/// unwrap the DEK(s) via the [`KeySource`], and construct a
/// [`WalEncryption`](super::wal::WalEncryption) ready to wire into the WAL
/// writer + the recovery readers.
///
/// Steps (ADR-216 §D-2):
/// 1. [`KeySource::health_check`] (the caller SHOULD do this first for a
///    fail-fast startup error; this fn re-checks implicitly via unwrap).
/// 2. If `<wal_dir>/wal.dek` exists: read it, unwrap each [`WrappedDek`] to
///    its plaintext DEK, build the in-memory DEK provider, construct
///    `WalEncryption` at the highest key version (restart path).
/// 3. If absent (first boot): `generate_wrapped_dek` → persist the sidecar
///    atomically → unwrap once → construct `WalEncryption::new(.., v1)`.
///
/// The plaintext DEK NEVER touches disk: it is unwrapped into an in-memory
/// [`SecretValue`] held by the DEK provider (zeroized on drop). The sidecar
/// on disk is the WRAPPED envelope only.
///
/// # Errors
///
/// [`WalEncryptionBootstrapError`] on a key-source failure (fail-closed —
/// never falls back to plaintext WAL, ADR-033), a sidecar IO/decode error,
/// or a `WalEncryption` construction error.
pub fn bootstrap_wal_encryption(
    key_source: &dyn KeySource,
    wal_dir: &Path,
) -> Result<WalEncryptionBootstrap, WalEncryptionBootstrapError> {
    use super::keyring::ENCRYPTION_KEY_NAMESPACE_WAL;
    use super::wal::WalEncryption;

    let existing = WalDekSidecar::read_from_dir(wal_dir)?;
    let (sidecar, freshly_generated) = match existing {
        Some(sc) if !sc.is_empty() => (sc, false),
        _ => {
            // First boot (no sidecar, or an empty one): generate a fresh
            // DEK wrapped under the current KEK, persist atomically.
            let scope = KeyScope::wal();
            let wrapped = key_source.generate_wrapped_dek(&scope)?;
            let mut sc = WalDekSidecar::new();
            sc.insert(KeyVersion::ONE, wrapped);
            sc.write_to_dir(wal_dir)?;
            (sc, true)
        }
    };

    // Unwrap every DEK in the sidecar into the in-memory provider, keyed by
    // the per-record WAL namespace the existing KeyRing reads.
    let mut keys = std::collections::HashMap::with_capacity(sidecar.len());
    for (key_version, wrapped) in sidecar.iter() {
        let dek = key_source.unwrap_dek(wrapped)?;
        let provider_key = format!("{ENCRYPTION_KEY_NAMESPACE_WAL}.v{}", key_version.raw());
        keys.insert(provider_key, dek);
    }
    let provider: Arc<dyn SecretsProvider> = Arc::new(InMemoryDekProvider { keys });

    let current_key_version = sidecar.max_key_version().unwrap_or(KeyVersion::ONE);
    let encryption = WalEncryption::new(provider, current_key_version)?;

    Ok(WalEncryptionBootstrap {
        encryption,
        current_key_version,
        freshly_generated,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arcgraph_core::EnvSecretsProvider;

    fn unique_prefix(suffix: &str) -> String {
        let pid = std::process::id();
        let thread_id = std::thread::current().id();
        format!("ARCGRAPH_KEYSOURCE_TEST_{pid}_{thread_id:?}_{suffix}_")
            .replace([' ', '(', ')'], "_")
    }

    /// Install a deterministic KEK at `<scope>.kek.v<version>` so tests
    /// don't depend on a real keyring.
    fn install_kek(p: &dyn SecretsProvider, scope: &KeyScope, version: KekVersion, byte: u8) {
        let key = format!("{}.kek.v{}", scope.namespace(), version.raw());
        p.set(&key, SecretValue::new([byte; SECRET_VALUE_LEN]))
            .expect("install KEK");
    }

    fn dev_source(
        prefix: &str,
        kek_version: KekVersion,
    ) -> (Arc<dyn SecretsProvider>, SecretsProviderKeySource) {
        let provider: Arc<dyn SecretsProvider> = Arc::new(
            EnvSecretsProvider::without_startup_warn_for_tests(prefix.to_owned()),
        );
        let src = SecretsProviderKeySource::new(Arc::clone(&provider), "env", kek_version);
        (provider, src)
    }

    #[test]
    fn key_source_id_has_provider_tag() {
        let (_p, src) = dev_source(&unique_prefix("id"), KekVersion::ONE);
        assert_eq!(src.key_source_id(), "secrets-provider:env");
    }

    #[test]
    fn generate_then_unwrap_round_trips_dek() {
        let prefix = unique_prefix("rt");
        let (provider, src) = dev_source(&prefix, KekVersion::ONE);
        let scope = KeyScope::wal();
        install_kek(&*provider, &scope, KekVersion::ONE, 0xC0);

        let wrapped = src.generate_wrapped_dek(&scope).expect("generate");
        // The wrapped envelope must NOT be the plaintext DEK (it's IV + AEAD).
        assert!(wrapped.wrapped_bytes.len() >= AES_GCM_IV_LEN + AES_GCM_TAG_LEN);
        assert_eq!(wrapped.kek_version, KekVersion::ONE);
        assert_eq!(wrapped.key_source_id, "secrets-provider:env");

        let dek = src.unwrap_dek(&wrapped).expect("unwrap");
        // A second unwrap yields the SAME DEK (deterministic).
        let dek2 = src.unwrap_dek(&wrapped).expect("unwrap2");
        assert_eq!(dek.expose_bytes(), dek2.expose_bytes());
    }

    #[test]
    fn health_check_fails_closed_when_kek_absent() {
        let prefix = unique_prefix("hc_absent");
        let (_provider, src) = dev_source(&prefix, KekVersion::ONE);
        // No KEK installed.
        let err = src.health_check(&KeyScope::wal()).unwrap_err();
        assert!(
            matches!(err, KeySourceError::KekNotFound { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn health_check_passes_when_kek_present() {
        let prefix = unique_prefix("hc_present");
        let (provider, src) = dev_source(&prefix, KekVersion::ONE);
        install_kek(&*provider, &KeyScope::wal(), KekVersion::ONE, 0x11);
        src.health_check(&KeyScope::wal()).expect("KEK present");
    }

    /// CF-2: tamper the `key_source_id` byte read back → unwrap fails CLOSED.
    #[test]
    fn unwrap_fails_closed_on_tampered_key_source_id() {
        let prefix = unique_prefix("tamper_id");
        let (provider, src) = dev_source(&prefix, KekVersion::ONE);
        let scope = KeyScope::wal();
        install_kek(&*provider, &scope, KekVersion::ONE, 0xAA);
        let mut wrapped = src.generate_wrapped_dek(&scope).expect("generate");
        // Swap the key_source_id to a different (attacker-chosen) source.
        wrapped.key_source_id = "secrets-provider:attacker".to_owned();
        let err = src.unwrap_dek(&wrapped).unwrap_err();
        assert!(
            matches!(err, KeySourceError::UnwrapFailed { .. }),
            "got {err:?}"
        );
    }

    /// CF-2: tamper the `kek_version` field → unwrap fails CLOSED (the
    /// version is bound into the AAD AND routes KEK resolution).
    #[test]
    fn unwrap_fails_closed_on_tampered_kek_version() {
        let prefix = unique_prefix("tamper_kekv");
        let (provider, src) = dev_source(&prefix, KekVersion::ONE);
        let scope = KeyScope::wal();
        install_kek(&*provider, &scope, KekVersion::ONE, 0xBB);
        // Install a DIFFERENT KEK at v2 so resolution succeeds but the AAD
        // mismatches (the strongest tamper case: resolvable wrong version).
        install_kek(&*provider, &scope, KekVersion::new(2), 0xCC);
        let mut wrapped = src.generate_wrapped_dek(&scope).expect("generate");
        wrapped.kek_version = KekVersion::new(2);
        let err = src.unwrap_dek(&wrapped).unwrap_err();
        assert!(
            matches!(err, KeySourceError::UnwrapFailed { .. }),
            "got {err:?}"
        );
    }

    /// CF-2: tamper a wrapped-bytes ciphertext byte → unwrap fails CLOSED.
    #[test]
    fn unwrap_fails_closed_on_tampered_ciphertext() {
        let prefix = unique_prefix("tamper_ct");
        let (provider, src) = dev_source(&prefix, KekVersion::ONE);
        let scope = KeyScope::wal();
        install_kek(&*provider, &scope, KekVersion::ONE, 0xDD);
        let mut wrapped = src.generate_wrapped_dek(&scope).expect("generate");
        let last = wrapped.wrapped_bytes.len() - 1;
        wrapped.wrapped_bytes[last] ^= 0x80;
        let err = src.unwrap_dek(&wrapped).unwrap_err();
        assert!(
            matches!(err, KeySourceError::UnwrapFailed { .. }),
            "got {err:?}"
        );
    }

    // ── sidecar codec ──────────────────────────────────────────────────

    fn fixture_wrapped(kek: u16, id: &str, byte: u8, len: usize) -> WrappedDek {
        WrappedDek {
            kek_version: KekVersion::new(kek),
            key_source_id: id.to_owned(),
            wrapped_bytes: vec![byte; len],
            wrap_alg: WrapAlg::Aes256GcmKeyWrap,
        }
    }

    #[test]
    fn sidecar_encode_decode_round_trips() {
        let mut sc = WalDekSidecar::new();
        sc.insert(
            KeyVersion::ONE,
            fixture_wrapped(1, "secrets-provider:env", 0x11, 48),
        );
        sc.insert(
            KeyVersion::new(2),
            fixture_wrapped(1, "secrets-provider:env", 0x22, 60),
        );
        let blob = sc.encode();
        let back = WalDekSidecar::decode(&blob).expect("decode");
        assert_eq!(back, sc);
        assert_eq!(back.len(), 2);
        assert_eq!(back.max_key_version(), Some(KeyVersion::new(2)));
    }

    #[test]
    fn sidecar_decode_rejects_bad_magic() {
        let blob = vec![b'X', b'X', b'X', b'X', 1, 0, 0, 0, 0];
        let err = WalDekSidecar::decode(&blob).unwrap_err();
        assert!(matches!(err, SidecarCodecError::BadMagic));
    }

    #[test]
    fn sidecar_decode_rejects_truncation() {
        let mut sc = WalDekSidecar::new();
        sc.insert(KeyVersion::ONE, fixture_wrapped(1, "id", 0x33, 40));
        let mut blob = sc.encode();
        blob.truncate(blob.len() - 5); // chop the tail
        let err = WalDekSidecar::decode(&blob).unwrap_err();
        assert!(matches!(err, SidecarCodecError::Truncated));
    }

    #[test]
    fn sidecar_atomic_write_then_read_round_trips() {
        let dir = tempdir();
        let mut sc = WalDekSidecar::new();
        sc.insert(
            KeyVersion::ONE,
            fixture_wrapped(1, "secrets-provider:env", 0x44, 52),
        );
        sc.write_to_dir(&dir).expect("write");
        let back = WalDekSidecar::read_from_dir(&dir)
            .expect("read")
            .expect("present");
        assert_eq!(back, sc);
    }

    #[test]
    fn sidecar_read_absent_is_none() {
        let dir = tempdir();
        let res = WalDekSidecar::read_from_dir(&dir).expect("read");
        assert!(res.is_none());
    }

    /// OQ-1 crash-safety: a torn rewrite (temp file present, no rename)
    /// leaves the OLD sidecar readable. We simulate the torn rewrite by
    /// writing a garbage temp file then confirming the real file is intact.
    #[test]
    fn sidecar_torn_rewrite_leaves_old_readable() {
        let dir = tempdir();
        // Write the original (good) sidecar.
        let mut original = WalDekSidecar::new();
        original.insert(
            KeyVersion::ONE,
            fixture_wrapped(1, "secrets-provider:env", 0x55, 48),
        );
        original.write_to_dir(&dir).expect("write original");

        // Simulate a torn rewrite: write a garbage temp file WITHOUT
        // renaming it over the real file (a crash before the rename
        // commit point). `write_to_dir`'s rename is the atomic commit, so a
        // crash before it leaves the temp as orphan garbage + the real file
        // untouched.
        let real = sidecar_path(&dir);
        let tmp = real.with_extension("dek.tmp");
        std::fs::write(&tmp, b"TORN-GARBAGE-NOT-A-VALID-SIDECAR").expect("write torn temp");

        // The real sidecar still decodes to the ORIGINAL (old-readable).
        let back = WalDekSidecar::read_from_dir(&dir)
            .expect("read")
            .expect("old sidecar still present");
        assert_eq!(
            back, original,
            "torn rewrite must leave the OLD wrapped DEK readable (OQ-1)"
        );
    }

    /// Hostile-grep guard: the sidecar bytes on disk must NEVER contain the
    /// plaintext DEK. We generate a real wrapped DEK, persist it, and assert
    /// the unwrapped DEK bytes do not appear verbatim in the file.
    #[test]
    fn sidecar_file_never_contains_plaintext_dek() {
        let prefix = unique_prefix("no_plaintext");
        let (provider, src) = dev_source(&prefix, KekVersion::ONE);
        let scope = KeyScope::wal();
        install_kek(&*provider, &scope, KekVersion::ONE, 0x77);
        let wrapped = src.generate_wrapped_dek(&scope).expect("generate");
        let dek = src.unwrap_dek(&wrapped).expect("unwrap");
        let dek_plaintext = dek.expose_bytes();

        let dir = tempdir();
        let mut sc = WalDekSidecar::new();
        sc.insert(KeyVersion::ONE, wrapped);
        sc.write_to_dir(&dir).expect("write");

        let on_disk = std::fs::read(sidecar_path(&dir)).expect("read raw bytes");
        // The 32-byte plaintext DEK must not appear as a contiguous window
        // anywhere in the persisted sidecar.
        let found = on_disk
            .windows(SECRET_VALUE_LEN)
            .any(|w| w == dek_plaintext.as_slice());
        assert!(
            !found,
            "plaintext DEK leaked into the wal.dek sidecar — REJECT"
        );
    }

    // ── build_durable bootstrap dance ──────────────────────────────────

    #[test]
    fn bootstrap_first_boot_generates_sidecar_then_restart_unwraps() {
        let prefix = unique_prefix("boot_rt");
        let (provider, src) = dev_source(&prefix, KekVersion::ONE);
        install_kek(&*provider, &KeyScope::wal(), KekVersion::ONE, 0x88);
        let dir = tempdir();

        // First boot: no sidecar → generate + persist.
        let boot1 = bootstrap_wal_encryption(&src, &dir).expect("first boot");
        assert!(
            boot1.freshly_generated,
            "first boot must generate a fresh DEK"
        );
        assert_eq!(boot1.current_key_version, KeyVersion::ONE);
        assert!(
            sidecar_path(&dir).exists(),
            "wal.dek must be persisted on first boot"
        );

        // Encrypt a payload under the first-boot DEK.
        let pt = b"restart-round-trip-payload";
        let ct = boot1
            .encryption
            .encrypt(0, arcgraph_core::Lsn::new(1), pt)
            .expect("encrypt");

        // Restart: a SECOND bootstrap over the same dir + a fresh KeySource
        // (same provider/KEK) reads the sidecar + unwraps → decrypts the
        // earlier ciphertext.
        let src2 = SecretsProviderKeySource::new(Arc::clone(&provider), "env", KekVersion::ONE);
        let boot2 = bootstrap_wal_encryption(&src2, &dir).expect("restart boot");
        assert!(
            !boot2.freshly_generated,
            "restart must read the existing sidecar, not regenerate"
        );
        let back = boot2
            .encryption
            .decrypt(0, arcgraph_core::Lsn::new(1), &ct)
            .expect("decrypt");
        assert_eq!(
            back, pt,
            "restart-unwrapped DEK must decrypt first-boot ciphertext"
        );
    }

    #[test]
    fn bootstrap_kek_rotation_rewraps_dek_unchanged_wal_decrypts() {
        // KEK rotation: re-wrap the SAME DEK under a new KEK, overwrite the
        // sidecar. All WAL still decrypts (the DEK is unchanged).
        let prefix = unique_prefix("kek_rot");
        let (provider, src) = dev_source(&prefix, KekVersion::ONE);
        install_kek(&*provider, &KeyScope::wal(), KekVersion::ONE, 0x10);
        let dir = tempdir();
        let boot = bootstrap_wal_encryption(&src, &dir).expect("first boot");
        let pt = b"kek-rotation-payload";
        let ct = boot
            .encryption
            .encrypt(0, arcgraph_core::Lsn::new(5), pt)
            .expect("encrypt");

        // Read the wrapped DEK, unwrap it (the plaintext DEK), then re-wrap
        // it under KEK v2 via a v2 KeySource — the envelope rotation.
        install_kek(&*provider, &KeyScope::wal(), KekVersion::new(2), 0x20);
        let dek = src
            .unwrap_dek(
                WalDekSidecar::read_from_dir(&dir)
                    .unwrap()
                    .unwrap()
                    .get(KeyVersion::ONE)
                    .unwrap(),
            )
            .expect("unwrap under v1");

        let src_v2 =
            SecretsProviderKeySource::new(Arc::clone(&provider), "env", KekVersion::new(2));
        // Re-wrap the SAME DEK bytes under KEK v2 (the envelope-rotation
        // capability). This is NOT a new DEK — the WAL still decrypts.
        let rewrapped = src_v2
            .rewrap_dek(&KeyScope::wal(), &dek)
            .expect("rewrap under v2");
        assert_eq!(rewrapped.kek_version, KekVersion::new(2));
        let mut sc = WalDekSidecar::new();
        sc.insert(KeyVersion::ONE, rewrapped);
        sc.write_to_dir(&dir)
            .expect("overwrite sidecar with re-wrapped DEK");

        // Restart under v2: the sidecar now wraps the same DEK under KEK v2;
        // the historical WAL ciphertext still decrypts.
        let boot2 = bootstrap_wal_encryption(&src_v2, &dir).expect("restart under v2");
        let back = boot2
            .encryption
            .decrypt(0, arcgraph_core::Lsn::new(5), &ct)
            .expect("decrypt");
        assert_eq!(
            back, pt,
            "KEK rotation must NOT change the DEK — WAL still decrypts"
        );
    }

    #[test]
    fn bootstrap_fails_closed_when_kek_absent() {
        // enabled but no KEK in provider → bootstrap refuses (fail-closed).
        let prefix = unique_prefix("boot_fail_closed");
        let (_provider, src) = dev_source(&prefix, KekVersion::ONE);
        let dir = tempdir();
        // No KEK installed.
        let err = bootstrap_wal_encryption(&src, &dir).unwrap_err();
        assert!(
            matches!(
                err,
                WalEncryptionBootstrapError::KeySource(KeySourceError::KekNotFound { .. })
            ),
            "got {err:?}"
        );
        // No plaintext-WAL fallback: the sidecar must NOT have been written
        // with an unwrappable DEK (generate succeeds only if KEK present).
        assert!(
            !sidecar_path(&dir).exists(),
            "fail-closed must not leave a sidecar behind"
        );
    }

    /// A minimal temp dir under the OS temp root, cleaned best-effort.
    fn tempdir() -> PathBuf {
        let pid = std::process::id();
        let tid = format!("{:?}", std::thread::current().id()).replace([' ', '(', ')'], "_");
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!("arcgraph-walDek-{pid}-{tid}-{nanos}"));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }
}
