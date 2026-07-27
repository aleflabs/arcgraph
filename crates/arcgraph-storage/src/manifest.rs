//! `<data_dir>/MANIFEST` — the v2 data-dir self-description stamp
//! (design `m1-m2-m4-m5-impl-designs.md` §0.1; ADR-230 OQ-E / RE-4b).
//!
//! # Why (v2 M1 — the first `data_dir_version` step)
//!
//! The pre-v2 store format was implicit: `pages.db` + `wal/` + the
//! 12-byte `VERSION` guard (`crate::data_dir_version`, SVC-2 #1302).
//! The v2 record-native migration is a SEQUENCE of on-disk format
//! steps (M1 slotted props → M2 typed blocks → M3 delta WAL → …), and
//! a mixed-step store (e.g. M1-packed props but still B-tree-indexed
//! records) must self-describe without inflating the coarse `VERSION`
//! integer space. The MANIFEST carries the per-substrate format
//! strings; the `VERSION` file stays the load-bearing boot REFUSAL
//! gate an old binary enforces (an old binary knows nothing of the
//! MANIFEST — it refuses on the `VERSION` integer alone).
//!
//! # Crash-atomicity
//!
//! [`write_data_dir_manifest`] is temp-write → fsync → rename → dir-fsync (the
//! ADR-229 checkpoint-sidecar pattern): a crash mid-write leaves
//! either the old MANIFEST (or none) or the complete new one — never
//! a torn file. The M1 migrate-on-open uses the final MANIFEST
//! rewrite (`props_store_format: "slotted-v1-migrating"` →
//! `"slotted-v1"`) as its SINGLE COMMIT POINT (design §0.2).
//!
//! # Strictness
//!
//! `#[serde(deny_unknown_fields)]` per design §0.1 + code-quality policy
//! config-strict-mode: a MANIFEST written by a FUTURE engine with
//! fields this binary does not understand refuses loudly at boot
//! instead of silently dropping semantics. (A future engine that adds
//! fields also bumps `data_dir_version`, so the `VERSION` guard
//! usually fires first; the serde gate is defense in depth.)
//!
//! # Budget (performance-budget discipline)
//!
//! One ≤ 4 KiB JSON file read at boot + one write per migration step
//! — not a hot path.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use arcgraph_core::Lsn;

/// Name of the manifest file inside `<data_dir>`.
pub const MANIFEST_FILE: &str = "MANIFEST";

/// `props_store_format` for the v2 M1 slotted small-blob store, fully
/// migrated (design §0.1: `"slotted-v1"`).
pub const PROPS_FORMAT_SLOTTED_V1: &str = "slotted-v1";

/// `props_store_format` while the M1 migrate-on-open re-encode is in
/// flight (resumable — see `crate::migrate`). A store carrying this
/// value re-enters the sweep on next open; the rewrite to
/// [`PROPS_FORMAT_SLOTTED_V1`] is the migration's single commit point.
pub const PROPS_FORMAT_SLOTTED_V1_MIGRATING: &str = "slotted-v1-migrating";

/// `props_store_format` for the v2 M2 typed property blocks, fully
/// migrated (ADR-230 row M2; design §M2.6: `data_dir_version` 3 → 4,
/// "`props_store_format` typed").
pub const PROPS_FORMAT_TYPED_V1: &str = "typed-v1";

/// `props_store_format` while the M2 migrate-on-open re-encode (M1
/// JSON slot payloads → typed blocks) is in flight (resumable — see
/// `crate::migrate::run_m2_migrate_on_open`). A store carrying this
/// value re-enters the sweep on next open; the rewrite to
/// [`PROPS_FORMAT_TYPED_V1`] is the migration's single commit point.
pub const PROPS_FORMAT_TYPED_V1_MIGRATING: &str = "typed-v1-migrating";

/// `record_store_format` at M1 (records are still B-tree-indexed
/// slotted pages; M4 flips this to direct-addressed).
pub const RECORD_FORMAT_BTREE_INDEXED: &str = "btree-indexed";

/// `record_store_format` after the M4 offline rewrite retires the primary
/// B-tree and makes arithmetic addressing authoritative.
pub const RECORD_FORMAT_DIRECT_M4: &str = "direct-addressed-v1";

/// `tel_ref_format` for every M4/M5 generation written BEFORE #1519
/// densify: `NodeRecord::out_tel_ref`/`in_tel_ref` and a TEL block's
/// `prev_block_ptr` carried a BARE `PageType::Tel` page id (one
/// dedicated page per (owner, type) adjacency block — the page-per-block
/// layout `docs/adr/` M5-D3 100M rung measured at ~200x blowup).
///
/// This is the value [`DataDirManifest::tel_ref_format`] defaults to via
/// `#[serde(default)]` when parsing an OLDER manifest that predates the
/// field entirely (every M4/M5 generation persisted before #1519 landed):
/// an absent field on disk truthfully means "this store's TEL refs are
/// bare page ids", never "unknown" — the whole point of the discriminator
/// is that silence must resolve to the OLD, pre-densify meaning rather
/// than being mistaken for the current one.
pub const TEL_REF_FORMAT_BARE_PAGE_ID: &str = "bare-page-id";

/// `tel_ref_format` for a generation whose STORE_TEL was written by the
/// #1519 densify packer: `out_tel_ref`/`in_tel_ref`/`prev_block_ptr`
/// carry an opaque `arcgraph_storage::m4_migration::encode_tel_ref`
/// `(page_no << 16 | slot)` pair, and sub-threshold blocks may share a
/// page behind a `TEL_PAGE_FLAG_PACKED` intra-page directory.
///
/// # Why this exists (SILENT-M6-CORRUPTION, #1519 BLOCK_FIX FIX 1)
///
/// #1519 changed the STORE_TEL ref encoding but left
/// [`crate::data_dir_version::DATA_DIR_VERSION_DIRECT_M4`] (the coarse
/// `VERSION` integer) UNCHANGED — both encodings are v6/M4 generations,
/// and the coarse guard only refuses a version it has never heard of, so
/// a pre-#1519 store is byte-indistinguishable from a post-#1519 store at
/// that granularity. A bare page id >= 65536 decodes under the NEW
/// `encode_tel_ref` inverse as a plausible-looking but WRONG
/// `(page_no, slot)` pair — a silent adjacency-corruption misread with no
/// refusal anywhere, exactly the class the M1→M2/M2→M3 `props_store_format`
/// MANIFEST markers exist to prevent for property payloads (module docs
/// above). This field gives STORE_TEL the same fine-grained, MANIFEST-level
/// discriminator: a load-bearing bump to the coarse `VERSION` integer would
/// have rippled through every hardcoded `SUPPORTED_DATA_DIR_VERSIONS` /
/// version-set test in the tree for a change that is scoped to ONE
/// substrate's ref encoding, so this follows the established
/// `props_store_format`/`record_store_format` precedent instead (module
/// docs, "why v2 M1" — the MANIFEST carries per-substrate format strings
/// precisely so a substrate-local format change need not inflate the
/// coarse counter).
pub const TEL_REF_FORMAT_PAGE_SLOT_V1: &str = "page-slot-v1";

/// `wal_format` at M1 (full page images in `CommitBundle`s; M3 flips
/// this to the delta WAL).
pub const WAL_FORMAT_PAGE_IMAGE: &str = "page-image";

/// `wal_format` for the M3 physiological/delta WAL generation.
pub const WAL_FORMAT_DELTA_V9: &str = "delta-v9";

/// `wal_format` for the M4 delta WAL with physical owner-row kinds enabled.
pub const WAL_FORMAT_DELTA_V10: &str = "delta-v10";

/// The design §0.1 monotone step counter value M1 stamps.
/// (v1.0 stores are the IMPLICIT version 2 — no MANIFEST on disk, and
/// their `VERSION` file carries the legacy integer 1. There is no
/// on-disk stamp value 2; see `crate::data_dir_version` for the
/// mapping.)
pub const DATA_DIR_VERSION_M1: u16 = 3;

/// The design §0.1 monotone step counter value M2 stamps (typed
/// property blocks — ADR-230 row M2, design §M2.6).
pub const DATA_DIR_VERSION_M2: u16 = 4;

/// The design §6 M3 generation counter value.
pub const DATA_DIR_VERSION_M3: u16 = 5;

/// The design §M4.2 direct-addressed, extent-backed generation counter.
pub const DATA_DIR_VERSION_M4: u16 = 6;

/// `<data_dir>/MANIFEST` body (design §0.1).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DataDirManifest {
    /// Monotone format-step counter (M1 = 3, M2 = 4, …). Mirrors the
    /// `VERSION` file integer for M1+ stores.
    pub data_dir_version: u16,
    /// Engine semver that last wrote this manifest (diagnostics only —
    /// compatibility decisions key off `data_dir_version`).
    pub engine_semver: String,
    /// Property-store representation. See the `PROPS_FORMAT_*` consts.
    pub props_store_format: String,
    /// Record-store representation. See [`RECORD_FORMAT_BTREE_INDEXED`].
    pub record_store_format: String,
    /// STORE_TEL ref-encoding representation. See
    /// [`TEL_REF_FORMAT_BARE_PAGE_ID`] / [`TEL_REF_FORMAT_PAGE_SLOT_V1`].
    ///
    /// `#[serde(default = "tel_ref_format_default_bare_page_id")]`: every
    /// manifest written before #1519 (M1 through pre-densify M4/M5) has NO
    /// such field on disk at all. Defaulting the ABSENT case to the OLD
    /// bare-page-id meaning (rather than, say, the current constant) is the
    /// load-bearing direction — an absent field must resolve to "this is an
    /// old store", never be silently treated as current, or the whole
    /// discriminator is defeated for exactly the stores it exists to catch.
    #[serde(default = "tel_ref_format_default_bare_page_id")]
    pub tel_ref_format: String,
    /// WAL representation. See [`WAL_FORMAT_PAGE_IMAGE`].
    pub wal_format: String,
    /// RFC3339 UTC timestamp of manifest creation.
    pub created_utc: String,
    /// RFC3339 UTC timestamp of the last migration-step completion (or
    /// creation, for a store born at the current version).
    pub last_migration_utc: String,
    /// Immutable v4→v5 cutover frontier. Present only for an M3 generation;
    /// it makes `LSN_SEED = migration_lsn + 1` independently verifiable after
    /// later incremental checkpoints advance their own frontier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub migration_lsn: Option<u64>,
    /// Exact sorted tenant ids present in the complete M4 generation.
    ///
    /// A resume must compare this durable census with the selected
    /// generation instead of trusting whatever tenant directories remain.
    /// Without it, losing one whole tenant directory is indistinguishable
    /// from a legitimately smaller generation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant_census: Option<Vec<u64>>,
    /// SHA-256 of the immutable first M4 incremental-metadata artifact.
    ///
    /// The checksum is a resume proof for the VERSION-last crash window. A
    /// later healthy checkpoint may advance the sidecar after VERSION exists,
    /// so AlreadyUpgraded validation checks presence/shape but deliberately
    /// does not pin the live checkpoint frontier to this first artifact.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint_metadata_sha256: Option<String>,
}

/// `serde(default)` fn for [`DataDirManifest::tel_ref_format`]: an absent
/// field on an older manifest means the pre-#1519 bare-page-id encoding
/// (see [`TEL_REF_FORMAT_BARE_PAGE_ID`] docs for why this direction, not
/// the current constant, is the correct default).
fn tel_ref_format_default_bare_page_id() -> String {
    TEL_REF_FORMAT_BARE_PAGE_ID.to_string()
}

impl DataDirManifest {
    /// A fresh M1 (`data_dir_version = 3`) manifest with
    /// `props_store_format = slotted-v1` (a store born at M1, or one
    /// whose migrate-on-open completed).
    #[must_use]
    pub fn m1_slotted(now_utc: String) -> Self {
        Self {
            data_dir_version: DATA_DIR_VERSION_M1,
            engine_semver: env!("CARGO_PKG_VERSION").to_string(),
            props_store_format: PROPS_FORMAT_SLOTTED_V1.to_string(),
            record_store_format: RECORD_FORMAT_BTREE_INDEXED.to_string(),
            tel_ref_format: TEL_REF_FORMAT_BARE_PAGE_ID.to_string(),
            wal_format: WAL_FORMAT_PAGE_IMAGE.to_string(),
            created_utc: now_utc.clone(),
            last_migration_utc: now_utc,
            migration_lsn: None,
            tenant_census: None,
            checkpoint_metadata_sha256: None,
        }
    }

    /// The migrating variant (`props_store_format =
    /// slotted-v1-migrating`) written BEFORE the first migration batch
    /// commits, so a crash mid-sweep resumes on next open.
    #[must_use]
    pub fn m1_migrating(now_utc: String) -> Self {
        let mut m = Self::m1_slotted(now_utc);
        m.props_store_format = PROPS_FORMAT_SLOTTED_V1_MIGRATING.to_string();
        m
    }

    /// True iff the M1 migrate-on-open sweep still needs to run (the
    /// manifest carries the migrating marker).
    #[must_use]
    pub fn m1_migration_in_flight(&self) -> bool {
        self.props_store_format == PROPS_FORMAT_SLOTTED_V1_MIGRATING
    }

    /// A fresh M2 (`data_dir_version = 4`) manifest with
    /// `props_store_format = typed-v1` (a store born at M2, or one
    /// whose M2 migrate-on-open completed).
    #[must_use]
    pub fn m2_typed(now_utc: String) -> Self {
        Self {
            data_dir_version: DATA_DIR_VERSION_M2,
            engine_semver: env!("CARGO_PKG_VERSION").to_string(),
            props_store_format: PROPS_FORMAT_TYPED_V1.to_string(),
            record_store_format: RECORD_FORMAT_BTREE_INDEXED.to_string(),
            tel_ref_format: TEL_REF_FORMAT_BARE_PAGE_ID.to_string(),
            wal_format: WAL_FORMAT_PAGE_IMAGE.to_string(),
            created_utc: now_utc.clone(),
            last_migration_utc: now_utc,
            migration_lsn: None,
            tenant_census: None,
            checkpoint_metadata_sha256: None,
        }
    }

    /// The M2 migrating variant (`props_store_format =
    /// typed-v1-migrating`) written BEFORE the first M2 migration batch
    /// commits, so a crash mid-sweep resumes on next open (design §0.2
    /// — the same contract shape as [`Self::m1_migrating`]).
    #[must_use]
    pub fn m2_migrating(now_utc: String) -> Self {
        let mut m = Self::m2_typed(now_utc);
        m.props_store_format = PROPS_FORMAT_TYPED_V1_MIGRATING.to_string();
        m
    }

    /// True iff the M2 migrate-on-open sweep still needs to run (the
    /// manifest carries the M2 migrating marker).
    #[must_use]
    pub fn m2_migration_in_flight(&self) -> bool {
        self.props_store_format == PROPS_FORMAT_TYPED_V1_MIGRATING
    }

    /// True iff the store's property payloads are fully typed (the M2
    /// end state — nothing left for the M2 sweep to do).
    #[must_use]
    pub fn props_fully_typed(&self) -> bool {
        self.props_store_format == PROPS_FORMAT_TYPED_V1
    }

    /// Build the complete M3 manifest placed inside the invisible generation.
    #[must_use]
    pub fn m3_delta_from(prior: &Self, now_utc: String, migration_lsn: Lsn) -> Self {
        Self {
            data_dir_version: DATA_DIR_VERSION_M3,
            engine_semver: env!("CARGO_PKG_VERSION").to_string(),
            props_store_format: PROPS_FORMAT_TYPED_V1.to_string(),
            record_store_format: RECORD_FORMAT_BTREE_INDEXED.to_string(),
            tel_ref_format: TEL_REF_FORMAT_BARE_PAGE_ID.to_string(),
            wal_format: WAL_FORMAT_DELTA_V9.to_string(),
            created_utc: prior.created_utc.clone(),
            last_migration_utc: now_utc,
            migration_lsn: Some(migration_lsn.raw()),
            tenant_census: None,
            checkpoint_metadata_sha256: None,
        }
    }

    /// Build the complete M4 manifest placed in the invisible v6 generation.
    #[must_use]
    pub fn m4_direct_from(
        prior: &Self,
        now_utc: String,
        migration_lsn: Lsn,
        tenant_census: Vec<u64>,
        checkpoint_metadata_sha256: String,
    ) -> Self {
        Self {
            data_dir_version: DATA_DIR_VERSION_M4,
            engine_semver: env!("CARGO_PKG_VERSION").to_string(),
            props_store_format: PROPS_FORMAT_TYPED_V1.to_string(),
            record_store_format: RECORD_FORMAT_DIRECT_M4.to_string(),
            tel_ref_format: TEL_REF_FORMAT_PAGE_SLOT_V1.to_string(),
            wal_format: WAL_FORMAT_DELTA_V10.to_string(),
            created_utc: prior.created_utc.clone(),
            last_migration_utc: now_utc,
            migration_lsn: Some(migration_lsn.raw()),
            tenant_census: Some(tenant_census),
            checkpoint_metadata_sha256: Some(checkpoint_metadata_sha256),
        }
    }

    /// The manifest for a v6 generation **born by the M5 leg-(c) offline
    /// bootstrap-load** (`arcgraph load` into a virgin data dir;
    /// `docs/design/M5D-REDESIGN-AMENDMENT.md` §2.6).
    ///
    /// Honest provenance, by construction: there is NO prior manifest to
    /// chain from — `created_utc == last_migration_utc == now` — instead of
    /// the fabricated `m2_typed` prior the superseded PR #1504 synthesized
    /// (amendment finding V-4). The tenant census travels here (INV-M5.3:
    /// metadata travels with its generation), and `migration_lsn` is the
    /// loader's INV-M5.2 frontier so `LSN_SEED = migration_lsn + 1` stays
    /// independently verifiable, exactly as for the migration legs.
    #[must_use]
    pub fn fresh_load(
        now_utc: String,
        migration_lsn: Lsn,
        tenant_census: Vec<u64>,
        checkpoint_metadata_sha256: String,
    ) -> Self {
        Self {
            data_dir_version: DATA_DIR_VERSION_M4,
            engine_semver: env!("CARGO_PKG_VERSION").to_string(),
            props_store_format: PROPS_FORMAT_TYPED_V1.to_string(),
            record_store_format: RECORD_FORMAT_DIRECT_M4.to_string(),
            tel_ref_format: TEL_REF_FORMAT_PAGE_SLOT_V1.to_string(),
            wal_format: WAL_FORMAT_DELTA_V10.to_string(),
            created_utc: now_utc.clone(),
            last_migration_utc: now_utc,
            migration_lsn: Some(migration_lsn.raw()),
            tenant_census: Some(tenant_census),
            checkpoint_metadata_sha256: Some(checkpoint_metadata_sha256),
        }
    }
}

/// Faults surfaced by manifest read/write. Fail-loud + fail-closed
/// (mirrors `crate::data_dir_version::DataDirVersionError`); the
/// bootstrap boundary translates into `anyhow` with context.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum DataDirManifestError {
    /// The MANIFEST exists but does not parse as the known schema
    /// (malformed JSON, or unknown fields from a future engine —
    /// `deny_unknown_fields` fail-closed).
    #[error(
        "`{MANIFEST_FILE}` at {path} is malformed or from an unknown future format: {reason}. \
         Refusing to open a data dir whose manifest this binary cannot fully interpret"
    )]
    Malformed {
        /// Path of the MANIFEST file.
        path: PathBuf,
        /// Parse failure detail.
        reason: String,
    },

    /// I/O error reading or writing the MANIFEST (permissions, ENOSPC…).
    #[error("i/o error on `{MANIFEST_FILE}` at {path}: {source}")]
    Io {
        /// Path of the MANIFEST file.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
}

/// Path of the MANIFEST file inside `data_dir`.
#[must_use]
pub fn manifest_path(data_dir: &Path) -> PathBuf {
    data_dir.join(MANIFEST_FILE)
}

/// Read `<data_dir>/MANIFEST`. `Ok(None)` when absent (a pre-M1 store,
/// or a fresh dir before its first stamp).
pub fn read_data_dir_manifest(
    data_dir: &Path,
) -> Result<Option<DataDirManifest>, DataDirManifestError> {
    let path = manifest_path(data_dir);
    let bytes = match fs::read(&path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(source) => return Err(DataDirManifestError::Io { path, source }),
    };
    serde_json::from_slice::<DataDirManifest>(&bytes)
        .map(Some)
        .map_err(|e| DataDirManifestError::Malformed {
            path,
            reason: e.to_string(),
        })
}

/// Crash-atomically write `<data_dir>/MANIFEST`: temp-write → fsync →
/// rename → best-effort dir fsync (the ADR-229 sidecar pattern). After
/// return the manifest is durable; a crash at ANY prior point leaves
/// the previous manifest (or none) intact — never a torn file. This
/// rename IS the M1 migration's single commit point (design §0.2).
pub fn write_data_dir_manifest(
    data_dir: &Path,
    manifest: &DataDirManifest,
) -> Result<(), DataDirManifestError> {
    let path = manifest_path(data_dir);
    let tmp = data_dir.join(".MANIFEST.tmp");
    let io_err = |p: &Path, source: std::io::Error| DataDirManifestError::Io {
        path: p.to_path_buf(),
        source,
    };
    let body =
        serde_json::to_vec_pretty(manifest).map_err(|e| DataDirManifestError::Malformed {
            path: path.clone(),
            reason: format!("manifest serialization failed (bug): {e}"),
        })?;
    {
        let mut f = fs::File::create(&tmp).map_err(|e| io_err(&tmp, e))?;
        f.write_all(&body).map_err(|e| io_err(&tmp, e))?;
        f.sync_all().map_err(|e| io_err(&tmp, e))?;
    }
    fs::rename(&tmp, &path).map_err(|e| io_err(&path, e))?;
    // Best-effort directory fsync so the rename itself is durable
    // before callers proceed (POSIX: rename durability requires the
    // parent dir sync). Failure here is logged, not fatal — the worst
    // case is the rename replaying to the PRE-write state on power
    // loss, which every caller tolerates by construction (old-or-new,
    // never torn).
    if let Ok(dir) = fs::File::open(data_dir) {
        if let Err(e) = dir.sync_all() {
            tracing::warn!(
                target: "arcgraph_storage::manifest",
                dir = %data_dir.display(),
                error = %e,
                "MANIFEST parent-dir fsync failed (rename durability best-effort)",
            );
        }
    }
    Ok(())
}

/// RFC3339 UTC "now" (second precision) with no external time dep —
/// the proleptic-Gregorian civil-from-days conversion (Howard
/// Hinnant's `civil_from_days` algorithm, exact for the full u64
/// epoch-seconds range used here).
#[must_use]
pub fn now_rfc3339_utc() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    rfc3339_from_epoch_secs(secs)
}

/// Format epoch seconds as `YYYY-MM-DDTHH:MM:SSZ`.
fn rfc3339_from_epoch_secs(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let (h, m, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    // civil_from_days (Hinnant): days since 1970-01-01 → (y, m, d).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097); // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let mth = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let y = if mth <= 2 { y + 1 } else { y };
    format!("{y:04}-{mth:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn manifest_roundtrip_and_absent() {
        let tmp = TempDir::new().expect("tempdir");
        assert!(
            read_data_dir_manifest(tmp.path())
                .expect("absent read ok")
                .is_none(),
            "no MANIFEST yet"
        );
        let m = DataDirManifest::m1_slotted(now_rfc3339_utc());
        write_data_dir_manifest(tmp.path(), &m).expect("write");
        let back = read_data_dir_manifest(tmp.path())
            .expect("read")
            .expect("present");
        assert_eq!(back, m, "roundtrip byte-equal semantics");
        assert!(!back.m1_migration_in_flight());
    }

    #[test]
    fn migrating_marker_roundtrip() {
        let tmp = TempDir::new().expect("tempdir");
        let m = DataDirManifest::m1_migrating(now_rfc3339_utc());
        write_data_dir_manifest(tmp.path(), &m).expect("write");
        let back = read_data_dir_manifest(tmp.path())
            .expect("read")
            .expect("present");
        assert!(back.m1_migration_in_flight(), "migrating marker survives");
        // Completion rewrite = single commit point.
        let done = DataDirManifest::m1_slotted(now_rfc3339_utc());
        write_data_dir_manifest(tmp.path(), &done).expect("rewrite");
        let back = read_data_dir_manifest(tmp.path())
            .expect("read")
            .expect("present");
        assert!(!back.m1_migration_in_flight(), "final rewrite lands");
    }

    #[test]
    fn unknown_fields_fail_closed() {
        // deny_unknown_fields: a future-format manifest refuses loudly.
        let tmp = TempDir::new().expect("tempdir");
        let mut v = serde_json::to_value(DataDirManifest::m1_slotted(now_rfc3339_utc())).unwrap();
        v.as_object_mut()
            .unwrap()
            .insert("future_field".into(), serde_json::json!(42));
        std::fs::write(manifest_path(tmp.path()), serde_json::to_vec(&v).unwrap()).unwrap();
        let err = read_data_dir_manifest(tmp.path()).expect_err("unknown field must refuse");
        assert!(
            matches!(err, DataDirManifestError::Malformed { .. }),
            "expected Malformed, got {err:?}"
        );
    }

    #[test]
    fn malformed_json_fails_closed() {
        let tmp = TempDir::new().expect("tempdir");
        std::fs::write(manifest_path(tmp.path()), b"{not json").unwrap();
        assert!(matches!(
            read_data_dir_manifest(tmp.path()),
            Err(DataDirManifestError::Malformed { .. })
        ));
    }

    #[test]
    fn rfc3339_formatter_known_values() {
        // Cross-checked against `date -u -r <secs>`.
        assert_eq!(rfc3339_from_epoch_secs(0), "1970-01-01T00:00:00Z");
        assert_eq!(rfc3339_from_epoch_secs(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(
            rfc3339_from_epoch_secs(1_751_328_000),
            "2025-07-01T00:00:00Z"
        );
        assert_eq!(
            rfc3339_from_epoch_secs(4_102_444_799),
            "2099-12-31T23:59:59Z"
        );
    }
}
