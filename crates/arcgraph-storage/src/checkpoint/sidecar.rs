//! Checkpoint sidecar file — the durable checkpoint frontier record
//! (ADR-229 §Decision, OQ-1).
//!
//! The sidecar is a tiny, CRC-protected, fixed-layout file in the
//! data-dir that records the established checkpoint frontier for
//! **O(1) open-time lookup** — recovery reads it directly rather than
//! scanning the WAL for a `Checkpoint` record. It is written
//! crash-atomically (temp-file + fsync + rename + dir-fsync) so a crash
//! mid-write NEVER corrupts the previously-established checkpoint (the
//! both-or-neither contract of ADR-229 §Decision).
//!
//! # On-disk layout (v2, fixed 48 bytes)
//!
//! ```text
//! offset  field                 size  notes
//! 0       magic                 4     b"AGCK" (ArcGraph ChecKpoint)
//! 4       format_version        2     u16 LE; == CHECKPOINT_FORMAT_VERSION
//! 6       flags                 2     u16 LE; bit0 = full_state_snapshot,
//!                                      bit1 = v9 incremental metadata
//! 8       checkpoint_lsn        8     u64 LE; the durable frontier
//! 16      snapshot_last_wal_lsn 8     u64 LE; last WAL LSN in snapshot
//! 24      created_unix_ms       8     i64 LE; wall-clock (advisory)
//! 32      metadata_generation   8     immutable v9 metadata identity; 0 otherwise
//! 40      crc32c                4     over bytes [0..40]
//! 44      _reserved             4     must be 0 (forward-compat)
//! ```
//!
//! Version 1's 40-byte layout remains readable and maps incremental metadata
//! to generation 0 (the legacy frontier-only file name). The CRC covers the
//! complete v2 identity prefix so a torn generation selector cannot redirect
//! recovery.

use std::io::Write;
use std::path::{Path, PathBuf};

use arcgraph_core::{ArcGraphError, Lsn};
use thiserror::Error;

use crate::wal::segment::fsync_dir;

/// Sidecar file name in the data-dir. O(1)-lookup on open.
pub const CHECKPOINT_SIDECAR_FILE: &str = "CHECKPOINT";

/// Temp file used for the crash-atomic write (rename target).
const CHECKPOINT_SIDECAR_TMP: &str = "CHECKPOINT.tmp";

/// On-disk sidecar format version. Bumped on any layout change; an
/// unknown version is rejected (recovery falls back to from-zero).
pub const CHECKPOINT_FORMAT_VERSION: u16 = 2;

const CHECKPOINT_FORMAT_VERSION_V1: u16 = 1;

/// Magic bytes at offset 0 — "AGCK" (ArcGraph ChecKpoint). Distinguishes
/// "right file, wrong version" from "wrong file entirely".
pub const CHECKPOINT_MAGIC: [u8; 4] = *b"AGCK";

/// Fixed encoded size of the sidecar.
const SIDECAR_V1_SIZE: usize = 40;
const SIDECAR_SIZE: usize = 48;

/// `flags` bit0: the checkpoint carries a full-state snapshot, so
/// recovery MAY replay-from-frontier. When unset, the frontier is
/// advisory only and recovery replays from zero (ADR-229 OQ-2 —
/// anchoring is unsafe without the full-state snapshot at v1.0).
const FLAG_FULL_STATE_SNAPSHOT: u16 = 0x0001;

/// `flags` bit1: the checkpoint is backed by an immutable v9 incremental
/// metadata file. Unlike the legacy advisory-only shape, recovery may anchor
/// at its `redo_lsn` after restoring that metadata and the doublewrite area.
const FLAG_INCREMENTAL_METADATA: u16 = 0x0002;

/// Errors surfaced by the checkpoint subsystem. Codec-local per the
/// `docs/codec-error-translation.md` convention; translated to
/// [`ArcGraphError`] at the public boundary.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CheckpointError {
    /// The sidecar/snapshot file failed magic, version, CRC, or length
    /// validation. Recovery treats this as "no valid checkpoint" and
    /// falls back to a from-zero replay (the SAFE direction — replay
    /// more, never lose committed data).
    #[error("checkpoint file corrupt: {reason}")]
    Corrupt {
        /// Human-readable cause for operator diagnosis.
        reason: String,
    },

    /// The sidecar/snapshot references an on-disk format version this
    /// binary does not support.
    #[error("unsupported checkpoint format version {got} (this binary supports {supported})")]
    UnsupportedVersion {
        /// The version read from disk.
        got: u16,
        /// The version this binary understands.
        supported: u16,
    },

    /// I/O failure reading or writing a checkpoint file.
    #[error("checkpoint i/o: {0}")]
    Io(#[from] std::io::Error),

    /// The bounded blob tier could not stream a spill image while the
    /// checkpoint was being established.
    #[error("checkpoint blob spill: {0}")]
    Blob(#[from] crate::blob::BlobError),

    /// #1404 M0.x FIX-D — a streamed owner section's record count did not match
    /// the count header written before it (a capture-vs-install skew that the
    /// per-section capture guard is supposed to prevent). ABORTING the
    /// checkpoint on this is the defense-in-depth that turns a would-be
    /// corrupt-but-`Ok` snapshot (which #1365 WAL reclaim would act on → silent
    /// data loss) into a wasted checkpoint. The producer surfaces this BEFORE
    /// `finalize_atomic`, so the temp snapshot is discarded and the previous
    /// checkpoint stays valid; the next checkpoint retries.
    #[error(
        "checkpoint owner section '{owner}' count skew: header declared {header}, \
         streamed {streamed} (capture-vs-install race — checkpoint aborted, not established)"
    )]
    CountSkew {
        /// The owner section that skewed (e.g. `idempotency`, `intern`,
        /// `permissions`).
        owner: &'static str,
        /// The count written in the section header.
        header: u64,
        /// The count actually streamed.
        streamed: u64,
    },
}

impl From<CheckpointError> for ArcGraphError {
    fn from(e: CheckpointError) -> Self {
        match e {
            CheckpointError::Io(io) => ArcGraphError::Io(io),
            CheckpointError::Blob(blob) => blob.into(),
            // A corrupt/unsupported checkpoint is not a hard WAL
            // corruption — recovery falls back to from-zero replay.
            // We surface it as WalCorruption only if a caller escalates;
            // the recovery reader instead treats it as "no checkpoint".
            other => ArcGraphError::WalCorruption {
                lsn: Lsn::ZERO,
                reason: format!("checkpoint sidecar: {other}"),
            },
        }
    }
}

/// The durable checkpoint frontier record.
///
/// Established (durable) only when BOTH the state snapshot AND this
/// sidecar are on disk (ADR-229 §Decision crash-atomicity contract).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckpointSidecar {
    /// The checkpoint frontier: the highest commit LSN whose effects are
    /// durable in the checkpoint's state snapshot. Recovery replays only
    /// WAL records with `commit_lsn > checkpoint_lsn`.
    pub checkpoint_lsn: Lsn,
    /// The last WAL record LSN observed when the snapshot was taken.
    /// Advisory: lets recovery seed the WAL framing counter without a
    /// full drain when the WAL-since-checkpoint is empty.
    pub snapshot_last_wal_lsn: Lsn,
    /// Wall-clock creation time (Unix ms). Advisory / diagnostic only.
    pub created_unix_ms: i64,
    /// Whether this checkpoint carries a full-state snapshot. When
    /// `true`, recovery MAY replay-from-frontier; when `false`, the
    /// frontier is advisory and recovery replays from zero (ADR-229
    /// OQ-2 safe-by-construction anchoring gate).
    pub full_state_snapshot: bool,
    /// Whether this checkpoint is backed by v9 incremental metadata. This is
    /// mutually exclusive with [`Self::full_state_snapshot`].
    pub incremental_metadata: bool,
    /// Immutable metadata generation selected by this sidecar. Generation 0
    /// names the legacy frontier-only v9 metadata path.
    pub metadata_generation: u64,
}

impl CheckpointSidecar {
    /// Build a full-state checkpoint sidecar (the P1 anchoring shape).
    #[must_use]
    pub fn full_state(
        checkpoint_lsn: Lsn,
        snapshot_last_wal_lsn: Lsn,
        created_unix_ms: i64,
    ) -> Self {
        Self {
            checkpoint_lsn,
            snapshot_last_wal_lsn,
            created_unix_ms,
            full_state_snapshot: true,
            incremental_metadata: false,
            metadata_generation: 0,
        }
    }

    /// Build a v9 incremental-checkpoint sidecar. The immutable metadata file
    /// is named by `checkpoint_lsn`; writing this sidecar is the sole establish
    /// point after DWB, home pages, and metadata are durable.
    #[must_use]
    pub fn incremental(
        checkpoint_lsn: Lsn,
        snapshot_last_wal_lsn: Lsn,
        created_unix_ms: i64,
        metadata_generation: u64,
    ) -> Self {
        debug_assert!(metadata_generation > 0);
        Self {
            checkpoint_lsn,
            snapshot_last_wal_lsn,
            created_unix_ms,
            full_state_snapshot: false,
            incremental_metadata: true,
            metadata_generation,
        }
    }

    /// Encode to the fixed 40-byte on-disk layout.
    #[must_use]
    pub fn encode(&self) -> [u8; SIDECAR_SIZE] {
        let mut buf = [0u8; SIDECAR_SIZE];
        buf[0..4].copy_from_slice(&CHECKPOINT_MAGIC);
        buf[4..6].copy_from_slice(&CHECKPOINT_FORMAT_VERSION.to_le_bytes());
        debug_assert!(
            !(self.full_state_snapshot && self.incremental_metadata),
            "checkpoint sidecar modes are mutually exclusive"
        );
        let mut flags = 0;
        if self.full_state_snapshot {
            flags |= FLAG_FULL_STATE_SNAPSHOT;
        }
        if self.incremental_metadata {
            flags |= FLAG_INCREMENTAL_METADATA;
        }
        buf[6..8].copy_from_slice(&flags.to_le_bytes());
        buf[8..16].copy_from_slice(&self.checkpoint_lsn.raw().to_le_bytes());
        buf[16..24].copy_from_slice(&self.snapshot_last_wal_lsn.raw().to_le_bytes());
        buf[24..32].copy_from_slice(&self.created_unix_ms.to_le_bytes());
        buf[32..40].copy_from_slice(&self.metadata_generation.to_le_bytes());
        let crc = crc32c::crc32c(&buf[0..40]);
        buf[40..44].copy_from_slice(&crc.to_le_bytes());
        // buf[44..48] reserved = 0
        buf
    }

    /// Decode from the fixed on-disk layout. Validates magic, version,
    /// length, and CRC. A short or corrupt file is
    /// [`CheckpointError::Corrupt`] (recovery falls back to from-zero).
    pub fn decode(bytes: &[u8]) -> Result<Self, CheckpointError> {
        if bytes.len() < 6 {
            return Err(CheckpointError::Corrupt {
                reason: format!(
                    "sidecar too short: got {} bytes, expected at least 6",
                    bytes.len()
                ),
            });
        }
        if bytes[0..4] != CHECKPOINT_MAGIC {
            return Err(CheckpointError::Corrupt {
                reason: "bad sidecar magic (not AGCK)".to_owned(),
            });
        }
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        if !matches!(
            version,
            CHECKPOINT_FORMAT_VERSION_V1 | CHECKPOINT_FORMAT_VERSION
        ) {
            return Err(CheckpointError::UnsupportedVersion {
                got: version,
                supported: CHECKPOINT_FORMAT_VERSION,
            });
        }
        let expected_size = if version == CHECKPOINT_FORMAT_VERSION_V1 {
            SIDECAR_V1_SIZE
        } else {
            SIDECAR_SIZE
        };
        if bytes.len() != expected_size {
            return Err(CheckpointError::Corrupt {
                reason: format!(
                    "sidecar v{version} length is {}, expected {expected_size}",
                    bytes.len()
                ),
            });
        }
        let (crc_offset, crc_covered) = if version == CHECKPOINT_FORMAT_VERSION_V1 {
            (32, 32)
        } else {
            (40, 40)
        };
        let crc_stored = u32::from_le_bytes(
            bytes[crc_offset..crc_offset + 4]
                .try_into()
                .expect("sidecar length checked"),
        );
        let crc_computed = crc32c::crc32c(&bytes[0..crc_covered]);
        if crc_stored != crc_computed {
            return Err(CheckpointError::Corrupt {
                reason: format!(
                    "sidecar crc mismatch: stored 0x{crc_stored:08x}, computed 0x{crc_computed:08x}"
                ),
            });
        }
        let flags = u16::from_le_bytes([bytes[6], bytes[7]]);
        if flags & FLAG_FULL_STATE_SNAPSHOT != 0 && flags & FLAG_INCREMENTAL_METADATA != 0 {
            return Err(CheckpointError::Corrupt {
                reason: "sidecar sets both full-state and incremental checkpoint modes".to_owned(),
            });
        }
        let checkpoint_lsn = Lsn::new(u64::from_le_bytes(
            bytes[8..16].try_into().expect("bounds checked"),
        ));
        let snapshot_last_wal_lsn = Lsn::new(u64::from_le_bytes(
            bytes[16..24].try_into().expect("bounds checked"),
        ));
        let created_unix_ms = i64::from_le_bytes(bytes[24..32].try_into().expect("bounds checked"));
        let metadata_generation = if version == CHECKPOINT_FORMAT_VERSION_V1 {
            0
        } else {
            u64::from_le_bytes(bytes[32..40].try_into().expect("bounds checked"))
        };
        if flags & FLAG_INCREMENTAL_METADATA != 0
            && version == CHECKPOINT_FORMAT_VERSION
            && metadata_generation == 0
        {
            return Err(CheckpointError::Corrupt {
                reason: "v2 incremental sidecar selects metadata generation 0".to_owned(),
            });
        }
        if flags & FLAG_INCREMENTAL_METADATA == 0 && metadata_generation != 0 {
            return Err(CheckpointError::Corrupt {
                reason: "non-incremental sidecar selects a metadata generation".to_owned(),
            });
        }
        Ok(Self {
            checkpoint_lsn,
            snapshot_last_wal_lsn,
            created_unix_ms,
            full_state_snapshot: flags & FLAG_FULL_STATE_SNAPSHOT != 0,
            incremental_metadata: flags & FLAG_INCREMENTAL_METADATA != 0,
            metadata_generation,
        })
    }
}

/// Path of the sidecar within `data_dir`.
#[must_use]
pub fn sidecar_path(data_dir: &Path) -> PathBuf {
    data_dir.join(CHECKPOINT_SIDECAR_FILE)
}

/// Write the sidecar crash-atomically: temp-file + fsync + rename +
/// dir-fsync (ADR-229 §Decision; mirrors `truncate_torn_tail`).
///
/// The rename over the previous sidecar is atomic on POSIX, so a crash
/// leaves EITHER the previous sidecar (fully valid) OR the new one
/// (fully valid) — never a torn sidecar. This is the LAST step of the
/// producer, established only after the state snapshot is durable, so a
/// crash before this rename leaves the previous checkpoint valid.
pub fn write_sidecar_atomic(
    data_dir: &Path,
    sidecar: &CheckpointSidecar,
) -> Result<(), CheckpointError> {
    let tmp = data_dir.join(CHECKPOINT_SIDECAR_TMP);
    let final_path = sidecar_path(data_dir);
    let bytes = sidecar.encode();
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(&bytes)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, &final_path)?;
    fsync_dir(data_dir).map_err(arcgraph_err_to_io)?;
    Ok(())
}

/// `fsync_dir` returns [`ArcGraphError`]; the checkpoint boundary speaks
/// [`CheckpointError`]. A dir-fsync failure is always an I/O fault, so
/// re-wrap it as [`CheckpointError::Io`] (translating back to
/// [`ArcGraphError::Io`] at the public boundary is a round-trip no-op).
pub(super) fn arcgraph_err_to_io(e: ArcGraphError) -> CheckpointError {
    match e {
        ArcGraphError::Io(io) => CheckpointError::Io(io),
        other => CheckpointError::Io(std::io::Error::other(other.to_string())),
    }
}

/// Read the latest valid sidecar from `data_dir`, if any.
///
/// - `Ok(None)` — no sidecar present (fresh/legacy dir → recovery
///   replays from zero).
/// - `Ok(Some(_))` — a valid sidecar.
/// - `Err(Corrupt)` — a present-but-corrupt sidecar. Recovery treats
///   this as "no checkpoint" (from-zero replay, the SAFE direction) and
///   logs a warning; callers decide whether to surface it.
pub fn read_latest_sidecar(data_dir: &Path) -> Result<Option<CheckpointSidecar>, CheckpointError> {
    let path = sidecar_path(data_dir);
    match std::fs::read(&path) {
        Ok(bytes) => CheckpointSidecar::decode(&bytes).map(Some),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(CheckpointError::Io(e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn sample() -> CheckpointSidecar {
        CheckpointSidecar::full_state(Lsn::new(4242), Lsn::new(4243), 1_700_000_000_000)
    }

    #[test]
    fn encode_decode_roundtrip() {
        let s = sample();
        let bytes = s.encode();
        assert_eq!(bytes.len(), SIDECAR_SIZE);
        let back = CheckpointSidecar::decode(&bytes).unwrap();
        assert_eq!(back, s);
        assert!(back.full_state_snapshot);
        assert_eq!(back.checkpoint_lsn, Lsn::new(4242));
    }

    #[test]
    fn decode_rejects_short_file() {
        let err = CheckpointSidecar::decode(&[0u8; 10]).unwrap_err();
        assert!(matches!(err, CheckpointError::Corrupt { .. }));
    }

    #[test]
    fn decode_rejects_bad_magic() {
        let mut bytes = sample().encode();
        bytes[0] = b'X';
        // recompute crc so we isolate the magic check
        let crc = crc32c::crc32c(&bytes[0..40]);
        bytes[40..44].copy_from_slice(&crc.to_le_bytes());
        let err = CheckpointSidecar::decode(&bytes).unwrap_err();
        assert!(matches!(err, CheckpointError::Corrupt { .. }));
    }

    #[test]
    fn decode_rejects_crc_flip() {
        let mut bytes = sample().encode();
        bytes[8] ^= 0x01; // flip a byte of checkpoint_lsn, leave crc
        let err = CheckpointSidecar::decode(&bytes).unwrap_err();
        assert!(matches!(err, CheckpointError::Corrupt { .. }));
    }

    #[test]
    fn decode_rejects_unknown_version() {
        let mut bytes = sample().encode();
        bytes[4..6].copy_from_slice(&99u16.to_le_bytes());
        let err = CheckpointSidecar::decode(&bytes).unwrap_err();
        assert!(matches!(
            err,
            CheckpointError::UnsupportedVersion { got: 99, .. }
        ));
    }

    #[test]
    fn write_then_read_roundtrip() {
        let dir = tempdir().unwrap();
        let s = sample();
        write_sidecar_atomic(dir.path(), &s).unwrap();
        let back = read_latest_sidecar(dir.path()).unwrap().unwrap();
        assert_eq!(back, s);
    }

    #[test]
    fn incremental_generation_roundtrips_and_is_crc_protected() {
        let sidecar =
            CheckpointSidecar::incremental(Lsn::new(7), Lsn::new(8), 1_700_000_000_000, 42);
        let mut bytes = sidecar.encode();
        assert_eq!(CheckpointSidecar::decode(&bytes).unwrap(), sidecar);
        bytes[32] ^= 1;
        assert!(matches!(
            CheckpointSidecar::decode(&bytes),
            Err(CheckpointError::Corrupt { .. })
        ));
    }

    #[test]
    fn v1_incremental_sidecar_selects_legacy_generation_zero() {
        let mut bytes = [0u8; SIDECAR_V1_SIZE];
        bytes[0..4].copy_from_slice(&CHECKPOINT_MAGIC);
        bytes[4..6].copy_from_slice(&CHECKPOINT_FORMAT_VERSION_V1.to_le_bytes());
        bytes[6..8].copy_from_slice(&FLAG_INCREMENTAL_METADATA.to_le_bytes());
        bytes[8..16].copy_from_slice(&7u64.to_le_bytes());
        bytes[16..24].copy_from_slice(&8u64.to_le_bytes());
        bytes[24..32].copy_from_slice(&9i64.to_le_bytes());
        let crc = crc32c::crc32c(&bytes[0..32]);
        bytes[32..36].copy_from_slice(&crc.to_le_bytes());
        let sidecar = CheckpointSidecar::decode(&bytes).unwrap();
        assert!(sidecar.incremental_metadata);
        assert_eq!(sidecar.metadata_generation, 0);
    }

    #[test]
    fn read_missing_is_none() {
        let dir = tempdir().unwrap();
        assert!(read_latest_sidecar(dir.path()).unwrap().is_none());
    }

    #[test]
    fn atomic_rewrite_uses_latest() {
        let dir = tempdir().unwrap();
        write_sidecar_atomic(dir.path(), &sample()).unwrap();
        let newer = CheckpointSidecar::full_state(Lsn::new(9999), Lsn::new(10_000), 1);
        write_sidecar_atomic(dir.path(), &newer).unwrap();
        let back = read_latest_sidecar(dir.path()).unwrap().unwrap();
        assert_eq!(back.checkpoint_lsn, Lsn::new(9999));
        // no tmp file left behind
        assert!(!dir.path().join(CHECKPOINT_SIDECAR_TMP).exists());
    }

    #[test]
    fn corrupt_sidecar_on_disk_surfaces_error() {
        let dir = tempdir().unwrap();
        std::fs::write(sidecar_path(dir.path()), b"not a checkpoint").unwrap();
        let err = read_latest_sidecar(dir.path()).unwrap_err();
        assert!(matches!(err, CheckpointError::Corrupt { .. }));
    }
}
