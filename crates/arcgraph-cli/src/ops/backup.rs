//! ADR-204 — cold backup + verified restore v1 (§5.4 stage-1).
//!
//! A backup is a plain directory: the ADR-183 durable allowlist
//! (`pages.db` + `wal/*`) copied byte-for-byte, plus a checksummed,
//! versioned `BACKUP_MANIFEST.json`. Restore verifies the manifest
//! FIRST (per-file size + SHA-256 — torn/partial/bit-rotted copies
//! fail loud before the target is touched), refuses non-empty
//! targets, places the files with full fsync discipline, and leaves
//! loading to the standard boot recovery (`recover_from_wal` — the
//! K-3 crash-campaign-proven path; there is deliberately NO second
//! restore engine, ADR-204 D-4).
//!
//! COLD-only (ADR-204 D-1): `create` takes the SAME exclusive
//! [`DataDirLock`] the server takes, so quiescence is structural — a
//! running server makes backup fail loud-and-early, and a mid-backup
//! server start is excluded for the backup's duration. Online/hot
//! backup is #405 stage-2 (needs the #849 B3(b) checkpoint design).
//!
//! Copy allowlist is deny-by-default (ADR-204 D-2): `LOCK` is runtime
//! state and never copied; future `secrets/` (OS-keyring trust
//! boundary per the backup runbook) is structurally excluded because
//! only the allowlist is read.
//!
//! Budget (PD-5): offline operator verb — wall-clock is I/O-bound on
//! the copy; SHA-256 at ~1–2 GB/s/core hashes a 10 GB dir in seconds.
//! Nothing here is on a serving path.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tracing::info;

use crate::data_lock::DataDirLock;

/// Manifest file name inside a backup directory.
pub const MANIFEST_FILE: &str = "BACKUP_MANIFEST.json";

/// Current manifest format version. Bumps are additive per the
/// ADR-031 CommitBundle versioning discipline (ADR-204 D-2) — e.g. a
/// future online-cut `head_lsn` watermark lands as v2.
pub const MANIFEST_FORMAT_VERSION: u32 = 1;

/// The page-store file name (mirrors `bootstrap::PAGE_STORE_FILE`).
const PAGES_FILE: &str = "pages.db";
/// The WAL segment sub-directory (mirrors `bootstrap`'s wal dir).
const WAL_DIR: &str = "wal";
/// The on-disk data-dir version stamp (SVC-2, #1302). Part of the durable
/// layout: it MUST travel with the backup so a restored dir carries its
/// source's version stamp — else the boot guard would (correctly) reject the
/// restored dir as a legacy/unstamped store. Restoring a version-incompatible
/// backup onto a newer binary then fails loud at boot exactly as intended
/// (upgrade-safety). Mirrors `arcgraph_storage::data_dir_version::VERSION_FILE`.
const VERSION_FILE: &str = arcgraph_storage::data_dir_version::VERSION_FILE;

/// Faults surfaced by [`backup_create`] / [`backup_restore`].
///
/// `#[non_exhaustive]` under the strict public-contract policy.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum BackupError {
    /// The source data dir is missing the durable layout (no `wal/`).
    #[error(
        "source `{0}` is not an ArcGraph data dir (no `wal/` sub-directory); \
         nothing to back up"
    )]
    NotADataDir(PathBuf),

    /// The source data dir's `LOCK` is held (a server is running) —
    /// COLD backup requires exclusivity (ADR-204 D-1).
    #[error(
        "cannot take a cold backup of `{0}`: its LOCK is held (is a server \
         running?). Stop the server, then retry (online backup is #405 stage-2)"
    )]
    SourceLocked(PathBuf),

    /// The backup destination already exists and is non-empty.
    #[error("backup destination `{0}` exists and is not empty; refusing to mix artifacts")]
    DestNotEmpty(PathBuf),

    /// The restore target already contains store state — restore
    /// NEVER overwrites an existing store (ADR-204 D-4.2).
    #[error(
        "restore target `{0}` already contains `{1}`; refusing to overwrite an \
         existing store (delete it explicitly first)"
    )]
    TargetHasStore(PathBuf, String),

    /// The backup directory has no readable manifest.
    #[error("`{0}` has no readable {MANIFEST_FILE}: {1}")]
    ManifestUnreadable(PathBuf, String),

    /// Manifest format version is newer than this binary supports.
    #[error(
        "manifest format_version {found} is not supported by this binary \
         (supports ≤ {MANIFEST_FORMAT_VERSION}); upgrade arcgraph to restore"
    )]
    ManifestVersionUnsupported {
        /// The version found in the manifest.
        found: u32,
    },

    /// A manifest-listed file is missing, truncated, or fails its
    /// SHA-256 — the backup is torn/corrupt; the target was NOT
    /// touched (ADR-204 D-4.1).
    #[error(
        "backup verification FAILED for `{file}`: {reason}; the backup is \
         torn/corrupt and the restore target was not touched"
    )]
    VerifyFailed {
        /// The offending manifest entry.
        file: String,
        /// Mismatch detail (missing / size / sha256).
        reason: String,
    },

    /// Underlying I/O fault (with the path for diagnosability).
    #[error("backup I/O on `{path}`: {source}")]
    Io {
        /// The path the operation was touching.
        path: PathBuf,
        /// The underlying error.
        #[source]
        source: std::io::Error,
    },
}

impl BackupError {
    fn io(path: impl Into<PathBuf>) -> impl FnOnce(std::io::Error) -> Self {
        let path = path.into();
        move |source| Self::Io { path, source }
    }
}

/// One allowlisted file captured in a backup.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ManifestEntry {
    /// Path RELATIVE to the backup root (`pages.db`, `wal/<segment>`).
    pub path: String,
    /// Size in bytes of the COPY (verify-what-you-wrote).
    pub size: u64,
    /// Lowercase-hex SHA-256 of the COPY.
    pub sha256: String,
}

/// `BACKUP_MANIFEST.json` — the verifiability contract (ADR-204 D-2).
///
/// `#[serde(deny_unknown_fields)]` is deliberately NOT applied: a v2
/// manifest (additive fields) must remain restorable by a v2 binary
/// while a v1 binary rejects it via `format_version` — the version
/// gate, not field-set strictness, is the compat contract here. This
/// is an artifact we write, not operator-typed config, so the strict-mode
/// rule for user-deserialized `*Config` structs does not apply.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupManifest {
    /// Manifest format version (see [`MANIFEST_FORMAT_VERSION`]).
    pub format_version: u32,
    /// Unix seconds at backup creation (wall clock; informational).
    pub created_at_unix: u64,
    /// `CARGO_PKG_VERSION` of the creating binary (informational).
    pub arcgraph_version: String,
    /// Basename of the source data dir (informational).
    pub source: String,
    /// Allowlisted files with sizes + checksums.
    pub files: Vec<ManifestEntry>,
}

/// Create a cold backup of `data_dir` into `dest` (ADR-204 D-1/D-2).
///
/// Takes the data dir's exclusive [`DataDirLock`] for the duration —
/// fails loud if a server holds it. `dest` must not exist or be an
/// empty directory. Returns the written manifest.
pub fn backup_create(data_dir: &Path, dest: &Path) -> Result<BackupManifest, BackupError> {
    let wal_src = data_dir.join(WAL_DIR);
    if !wal_src.is_dir() {
        return Err(BackupError::NotADataDir(data_dir.to_path_buf()));
    }
    // D-1: structural quiescence — same lock the server takes, BEFORE
    // reading any state. Held for the whole copy.
    let _lock = DataDirLock::acquire(data_dir)
        .map_err(|_| BackupError::SourceLocked(data_dir.to_path_buf()))?;

    if dest.exists() {
        let mut entries = fs::read_dir(dest).map_err(BackupError::io(dest))?;
        if entries.next().is_some() {
            return Err(BackupError::DestNotEmpty(dest.to_path_buf()));
        }
    } else {
        fs::create_dir_all(dest).map_err(BackupError::io(dest))?;
    }

    // Allowlist copy (deny-by-default): VERSION + MANIFEST + pages.db +
    // wal/*. LOCK is runtime state — never copied.
    let mut files: Vec<ManifestEntry> = Vec::new();
    // SVC-2 / #1302 — carry the data-dir version stamp so the restored dir is
    // NOT seen as a legacy/unstamped store at boot, and so a version-
    // incompatible backup fails loud when restored onto a newer binary.
    let version_src = data_dir.join(VERSION_FILE);
    if version_src.is_file() {
        files.push(copy_and_hash(
            &version_src,
            &dest.join(VERSION_FILE),
            VERSION_FILE,
        )?);
    }
    // v2 M1 (ADR-230) — carry the data-dir MANIFEST (per-substrate format
    // strings). A restore without it still boots (the §11 migrate-on-open
    // self-heals an absent MANIFEST via the idempotent sweep), but copying
    // it keeps the restored dir byte-exact and skips that no-op pass.
    let dd_manifest_src = data_dir.join(arcgraph_storage::manifest::MANIFEST_FILE);
    if dd_manifest_src.is_file() {
        files.push(copy_and_hash(
            &dd_manifest_src,
            &dest.join(arcgraph_storage::manifest::MANIFEST_FILE),
            arcgraph_storage::manifest::MANIFEST_FILE,
        )?);
    }
    let pages_src = data_dir.join(PAGES_FILE);
    if pages_src.is_file() {
        files.push(copy_and_hash(
            &pages_src,
            &dest.join(PAGES_FILE),
            PAGES_FILE,
        )?);
    }
    let wal_dst = dest.join(WAL_DIR);
    fs::create_dir_all(&wal_dst).map_err(BackupError::io(&wal_dst))?;
    let mut segs: Vec<PathBuf> = fs::read_dir(&wal_src)
        .map_err(BackupError::io(&wal_src))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file())
        .collect();
    segs.sort(); // deterministic manifest order
    for seg in segs {
        let name = seg
            .file_name()
            .and_then(|n| n.to_str())
            .map(ToOwned::to_owned)
            .ok_or_else(|| BackupError::VerifyFailed {
                file: seg.display().to_string(),
                reason: "non-UTF-8 WAL segment name".into(),
            })?;
        let rel = format!("{WAL_DIR}/{name}");
        files.push(copy_and_hash(&seg, &wal_dst.join(&name), &rel)?);
    }
    fsync_dir(&wal_dst)?;

    let manifest = BackupManifest {
        format_version: MANIFEST_FORMAT_VERSION,
        created_at_unix: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs()),
        arcgraph_version: env!("CARGO_PKG_VERSION").to_string(),
        source: data_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("data")
            .to_string(),
        files,
    };
    let manifest_path = dest.join(MANIFEST_FILE);
    let body = serde_json::to_vec_pretty(&manifest).expect("manifest serializes (owned data)");
    write_fsynced(&manifest_path, &body)?;
    fsync_dir(dest)?;

    info!(
        files = manifest.files.len(),
        dest = %dest.display(),
        "ADR-204: cold backup created"
    );
    Ok(manifest)
}

/// Restore a backup into a fresh `data_dir` (ADR-204 D-4):
/// verify → refuse → place; loading is the standard boot recovery.
pub fn backup_restore(from: &Path, data_dir: &Path) -> Result<BackupManifest, BackupError> {
    // 1. Verify FIRST — nothing is written until the whole backup
    //    checks out (typed error names the offending file).
    let manifest = read_manifest(from)?;
    for entry in &manifest.files {
        verify_entry(from, entry)?;
    }

    // 2. Refuse dangerous targets: never overwrite an existing store.
    for marker in [WAL_DIR, PAGES_FILE, "LOCK"] {
        if data_dir.join(marker).exists() {
            return Err(BackupError::TargetHasStore(
                data_dir.to_path_buf(),
                marker.to_string(),
            ));
        }
    }
    fs::create_dir_all(data_dir).map_err(BackupError::io(data_dir))?;
    // Exclude a concurrently-starting server while we place files
    // (same lock order as the server: lock before state).
    let _lock = DataDirLock::acquire(data_dir)
        .map_err(|_| BackupError::SourceLocked(data_dir.to_path_buf()))?;

    // 3. Place with fsync discipline (a restore that evaporates on
    //    power loss is not a restore).
    let wal_dst = data_dir.join(WAL_DIR);
    fs::create_dir_all(&wal_dst).map_err(BackupError::io(&wal_dst))?;
    for entry in &manifest.files {
        let src = from.join(&entry.path);
        let dst = data_dir.join(&entry.path);
        copy_fsynced(&src, &dst)?;
    }
    fsync_dir(&wal_dst)?;
    fsync_dir(data_dir)?;

    info!(
        files = manifest.files.len(),
        data_dir = %data_dir.display(),
        "ADR-204: backup restored (boot recovery will replay on next start)"
    );
    Ok(manifest)
}

/// Read + version-gate the manifest of a backup directory.
fn read_manifest(from: &Path) -> Result<BackupManifest, BackupError> {
    let path = from.join(MANIFEST_FILE);
    let body = fs::read(&path)
        .map_err(|e| BackupError::ManifestUnreadable(from.to_path_buf(), e.to_string()))?;
    let manifest: BackupManifest = serde_json::from_slice(&body)
        .map_err(|e| BackupError::ManifestUnreadable(from.to_path_buf(), e.to_string()))?;
    if manifest.format_version > MANIFEST_FORMAT_VERSION {
        return Err(BackupError::ManifestVersionUnsupported {
            found: manifest.format_version,
        });
    }
    Ok(manifest)
}

/// Verify one manifest entry against the backup directory's bytes.
fn verify_entry(from: &Path, entry: &ManifestEntry) -> Result<(), BackupError> {
    let path = from.join(&entry.path);
    let meta = fs::metadata(&path).map_err(|_| BackupError::VerifyFailed {
        file: entry.path.clone(),
        reason: "missing from backup".into(),
    })?;
    if meta.len() != entry.size {
        return Err(BackupError::VerifyFailed {
            file: entry.path.clone(),
            reason: format!("size {} != manifest {}", meta.len(), entry.size),
        });
    }
    let actual = sha256_file(&path)?;
    if actual != entry.sha256 {
        return Err(BackupError::VerifyFailed {
            file: entry.path.clone(),
            reason: format!("sha256 {actual} != manifest {}", entry.sha256),
        });
    }
    Ok(())
}

/// Copy `src` → `dst` (fsynced), then hash THE COPY (ADR-204 D-2:
/// verify-what-you-wrote) and return its manifest entry.
fn copy_and_hash(src: &Path, dst: &Path, rel: &str) -> Result<ManifestEntry, BackupError> {
    copy_fsynced(src, dst)?;
    let size = fs::metadata(dst).map_err(BackupError::io(dst))?.len();
    let sha256 = sha256_file(dst)?;
    Ok(ManifestEntry {
        path: rel.to_string(),
        size,
        sha256,
    })
}

/// Plain streamed copy + fsync of the destination file.
fn copy_fsynced(src: &Path, dst: &Path) -> Result<(), BackupError> {
    let mut input = fs::File::open(src).map_err(BackupError::io(src))?;
    let mut output = fs::File::create(dst).map_err(BackupError::io(dst))?;
    std::io::copy(&mut input, &mut output).map_err(BackupError::io(dst))?;
    output.sync_all().map_err(BackupError::io(dst))?;
    Ok(())
}

/// Write `body` to `path` and fsync the file.
fn write_fsynced(path: &Path, body: &[u8]) -> Result<(), BackupError> {
    let mut f = fs::File::create(path).map_err(BackupError::io(path))?;
    f.write_all(body).map_err(BackupError::io(path))?;
    f.sync_all().map_err(BackupError::io(path))?;
    Ok(())
}

/// fsync a directory so freshly-created entries are durable.
fn fsync_dir(dir: &Path) -> Result<(), BackupError> {
    let d = fs::File::open(dir).map_err(BackupError::io(dir))?;
    d.sync_all().map_err(BackupError::io(dir))?;
    Ok(())
}

/// Streamed SHA-256 of a file, lowercase hex.
fn sha256_file(path: &Path) -> Result<String, BackupError> {
    let mut f = fs::File::open(path).map_err(BackupError::io(path))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).map_err(BackupError::io(path))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(64);
    for b in digest {
        use std::fmt::Write as _;
        let _ = write!(hex, "{b:02x}");
    }
    Ok(hex)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_data_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        fs::create_dir_all(dir.path().join(WAL_DIR)).expect("wal dir");
        fs::write(
            dir.path().join(WAL_DIR).join("seg_0001.wal"),
            b"wal-bytes-1",
        )
        .expect("seg");
        fs::write(dir.path().join(PAGES_FILE), b"page-bytes").expect("pages");
        dir
    }

    /// Pin: create → manifest lists the allowlist (and ONLY it), with
    /// real sizes + checksums; LOCK never copied.
    #[test]
    fn create_writes_verifiable_manifest_and_excludes_lock() {
        let src = fake_data_dir();
        // A leftover LOCK in the source must not be copied.
        fs::write(src.path().join("LOCK"), b"pid").expect("lock file");
        let dest = tempfile::tempdir().expect("dest");
        let dest_dir = dest.path().join("b1");

        let m = backup_create(src.path(), &dest_dir).expect("create");
        let mut listed: Vec<&str> = m.files.iter().map(|f| f.path.as_str()).collect();
        listed.sort_unstable();
        assert_eq!(listed, vec![PAGES_FILE, "wal/seg_0001.wal"]);
        assert!(!dest_dir.join("LOCK").exists(), "LOCK is runtime state");
        assert_eq!(m.format_version, MANIFEST_FORMAT_VERSION);
        // Round-trip the manifest through verify (self-consistency).
        for e in &m.files {
            verify_entry(&dest_dir, e).expect("fresh backup verifies");
        }
    }

    /// SVC-2 / #1302 — the data-dir VERSION stamp is part of the backup
    /// allowlist: when the source carries a `VERSION` file, `backup_create`
    /// copies it (byte-identical) so the restored dir is NOT seen as a
    /// legacy/unstamped store at boot. RED-on-revert: drop the VERSION copy in
    /// `backup_create` → the manifest omits VERSION and this assertion fails.
    #[test]
    fn create_copies_version_stamp_when_present_1302() {
        let src = fake_data_dir();
        // Stamp the source dir with a real VERSION file (as durable bootstrap
        // would). Use the storage crate's own stamp writer to match the wire.
        arcgraph_storage::data_dir_version::stamp_data_dir(
            src.path(),
            arcgraph_storage::DATA_DIR_FORMAT_VERSION,
        )
        .expect("stamp source");
        let dest = tempfile::tempdir().expect("dest");
        let dest_dir = dest.path().join("bver");

        let m = backup_create(src.path(), &dest_dir).expect("create");
        let mut listed: Vec<&str> = m.files.iter().map(|f| f.path.as_str()).collect();
        listed.sort_unstable();
        assert!(
            listed.contains(&VERSION_FILE),
            "backup must include the VERSION stamp (#1302); listed={listed:?}"
        );
        // The copied stamp is byte-identical to the source.
        let src_bytes = fs::read(src.path().join(VERSION_FILE)).expect("read source VERSION");
        let dst_bytes = fs::read(dest_dir.join(VERSION_FILE)).expect("read backup VERSION");
        assert_eq!(
            src_bytes, dst_bytes,
            "backed-up VERSION must be byte-identical to the source"
        );
    }

    /// Pin (ADR-204 D-1): a held source LOCK makes create fail loud.
    #[test]
    fn create_fails_loud_when_source_locked() {
        let src = fake_data_dir();
        let _held = DataDirLock::acquire(src.path()).expect("hold lock");
        let dest = tempfile::tempdir().expect("dest");
        let err = backup_create(src.path(), &dest.path().join("b")).unwrap_err();
        assert!(matches!(err, BackupError::SourceLocked(_)), "got {err:?}");
    }

    /// Pin (ADR-204 D-4.1): a tampered byte fails verification with
    /// the offending file named, and the target is untouched.
    #[test]
    fn restore_rejects_tampered_backup_and_leaves_target_untouched() {
        let src = fake_data_dir();
        let dest = tempfile::tempdir().expect("dest");
        let bdir = dest.path().join("b");
        backup_create(src.path(), &bdir).expect("create");

        // Flip one byte in the copied WAL segment.
        let seg = bdir.join(WAL_DIR).join("seg_0001.wal");
        let mut bytes = fs::read(&seg).expect("read seg");
        bytes[0] ^= 0xFF;
        fs::write(&seg, &bytes).expect("tamper");

        let target = dest.path().join("restored");
        let err = backup_restore(&bdir, &target).unwrap_err();
        assert!(
            matches!(err, BackupError::VerifyFailed { ref file, .. } if file == "wal/seg_0001.wal"),
            "got {err:?}"
        );
        assert!(!target.exists(), "target untouched on verify failure");
    }

    /// Pin (ADR-204 D-4.2): restore refuses a target with store state.
    #[test]
    fn restore_refuses_existing_store() {
        let src = fake_data_dir();
        let dest = tempfile::tempdir().expect("dest");
        let bdir = dest.path().join("b");
        backup_create(src.path(), &bdir).expect("create");

        let target = fake_data_dir(); // already a store
        let err = backup_restore(&bdir, target.path()).unwrap_err();
        assert!(
            matches!(err, BackupError::TargetHasStore(_, _)),
            "got {err:?}"
        );
    }

    /// Pin: future manifest versions are rejected with the upgrade
    /// guidance (format gate, ADR-204 D-2).
    #[test]
    fn restore_rejects_newer_manifest_version() {
        let src = fake_data_dir();
        let dest = tempfile::tempdir().expect("dest");
        let bdir = dest.path().join("b");
        let mut m = backup_create(src.path(), &bdir).expect("create");
        m.format_version = MANIFEST_FORMAT_VERSION + 1;
        fs::write(
            bdir.join(MANIFEST_FILE),
            serde_json::to_vec_pretty(&m).expect("ser"),
        )
        .expect("rewrite manifest");

        let err = backup_restore(&bdir, &dest.path().join("r")).unwrap_err();
        assert!(
            matches!(err, BackupError::ManifestVersionUnsupported { found } if found == MANIFEST_FORMAT_VERSION + 1),
            "got {err:?}"
        );
    }
}
