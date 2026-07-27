//! Checkpoint-anchored recovery entry (ADR-229 §Decision — THE bound).
//!
//! On open, before the WAL replay, [`restore_latest_checkpoint`] reads
//! the latest valid checkpoint sidecar and, if it carries a full-state
//! snapshot, restores that snapshot into the served owners and returns
//! the frontier `checkpoint_lsn`. The WAL replay driver then replays
//! ONLY records with `commit_lsn > checkpoint_lsn` — the effects at/below
//! the frontier are already durable in the restored snapshot. This is
//! what bounds restart-recovery to `O(WAL-since-checkpoint)` instead of
//! `O(entire-history)` (the #849 rc-blocker).
//!
//! # Safe-by-construction (ADR-229 OQ-2)
//!
//! - No sidecar (fresh/legacy dir) → `Ok(None)` → recovery replays from
//!   `Lsn::ZERO` (the pre-ADR-229 behaviour, back-compat).
//! - Sidecar present but NOT `full_state_snapshot` → the frontier is
//!   advisory only; recovery still replays from zero (anchoring is
//!   unsafe without the full-state snapshot at v1.0).
//! - Sidecar present + `full_state_snapshot` but the snapshot file is
//!   missing/corrupt → treated as "no checkpoint" (from-zero replay, the
//!   SAFE direction — replay more, never lose committed data) + a
//!   `tracing::warn!`.
//! - Sidecar + snapshot both valid → restore + return the frontier.

use std::path::Path;

use arcgraph_core::Lsn;

use crate::checkpoint::sidecar::{CheckpointError, read_latest_sidecar};
use crate::checkpoint::snapshot::{
    CheckpointSnapshot, IncrementalCheckpointMetadata, SnapshotOwnerCounts,
    read_incremental_metadata, read_snapshot,
};
use crate::checkpoint::{DoublewriteArea, DoublewriteRestoreReport, DoublewriteRestoreTarget};

/// Outcome of a checkpoint-anchored open.
#[derive(Debug, Clone, Copy)]
pub struct CheckpointRestore {
    /// The frontier the WAL replay must anchor at: replay ONLY records
    /// with `commit_lsn > checkpoint_lsn`.
    pub checkpoint_lsn: Lsn,
    /// Per-owner counts restored from the snapshot.
    pub counts: SnapshotOwnerCounts,
}

/// v9 checkpoint-open result. Recovery must replay from `metadata.redo_lsn`,
/// not from the checkpoint frontier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IncrementalCheckpointRestore {
    pub metadata: IncrementalCheckpointMetadata,
    pub doublewrite: DoublewriteRestoreReport,
}

/// Restore the v9 checkpoint base, with the DWB scan deliberately first.
/// A corrupt v9 metadata file is a hard open failure: unlike the legacy full
/// snapshot, pre-redo WAL may already have been reclaimed below `redo_lsn`, so
/// a from-zero fallback is not necessarily available.
pub fn restore_latest_incremental_checkpoint(
    data_dir: &Path,
    snap: &CheckpointSnapshot<'_>,
    doublewrite: &DoublewriteArea,
    home: &mut dyn DoublewriteRestoreTarget,
) -> Result<Option<IncrementalCheckpointRestore>, CheckpointError> {
    crate::checkpoint::snapshot::sweep_incremental_metadata_temps(data_dir)?;
    let Some(sidecar) = read_latest_sidecar(data_dir)? else {
        return Ok(None);
    };
    if !sidecar.incremental_metadata {
        return Ok(None);
    }

    // IMPL-DEC-7: restore a torn/older home before metadata or any page-LSN
    // comparison. Reaching physical redo with a torn page is a double fault.
    let doublewrite_report = doublewrite
        .restore(home)
        .map_err(crate::checkpoint::sidecar::arcgraph_err_to_io)?;
    let metadata = read_incremental_metadata(
        data_dir,
        snap,
        sidecar.checkpoint_lsn,
        sidecar.metadata_generation,
    )?;
    Ok(Some(IncrementalCheckpointRestore {
        metadata,
        doublewrite: doublewrite_report,
    }))
}

/// Read + restore the latest valid full-state checkpoint from
/// `data_dir` into the owners of `snap`, returning the recovery
/// frontier. `Ok(None)` means "no valid checkpoint — replay the whole
/// WAL from zero".
///
/// This never hard-fails on a corrupt/absent checkpoint: a corrupt
/// sidecar or snapshot degrades to `Ok(None)` (from-zero replay) with a
/// warning, because falling back and replaying more WAL is always the
/// SAFE direction (the checkpoint is an optimization, not the source of
/// truth — the WAL is). It DOES surface a genuine I/O error (unreadable
/// present file) so an operator sees a disk fault rather than silent
/// data loss.
pub fn restore_latest_checkpoint(
    data_dir: &Path,
    snap: &CheckpointSnapshot<'_>,
) -> Result<Option<CheckpointRestore>, CheckpointError> {
    let sidecar = match read_latest_sidecar(data_dir) {
        Ok(Some(s)) => s,
        Ok(None) => return Ok(None), // fresh/legacy — from-zero replay
        Err(CheckpointError::Corrupt { reason }) => {
            tracing::warn!(
                target: "arcgraph_storage::checkpoint",
                reason = %reason,
                "checkpoint sidecar corrupt on open — falling back to from-zero WAL replay \
                 (SAFE: replay more, never lose committed data)",
            );
            return Ok(None);
        }
        Err(CheckpointError::UnsupportedVersion { got, supported }) => {
            tracing::warn!(
                target: "arcgraph_storage::checkpoint",
                got, supported,
                "checkpoint sidecar version unsupported — falling back to from-zero WAL replay",
            );
            return Ok(None);
        }
        Err(e @ CheckpointError::Io(_)) => return Err(e), // real disk fault — surface
        Err(e @ CheckpointError::Blob(_)) => return Err(e), // impossible on sidecar read; surface
        // #1404 M0.x FIX-D — `CountSkew` is a WRITE-path abort (the producer
        // aborts BEFORE establishing a skewed checkpoint), so it cannot arise
        // from `read_latest_sidecar`. If it somehow reaches here, treat it like
        // a corrupt sidecar and fall back to from-zero replay (the SAFE
        // direction — never anchor on a suspect checkpoint).
        Err(e @ CheckpointError::CountSkew { .. }) => {
            tracing::warn!(
                target: "arcgraph_storage::checkpoint",
                error = %e,
                "checkpoint sidecar reported a count skew on read (unexpected) — \
                 falling back to from-zero WAL replay (SAFE)",
            );
            return Ok(None);
        }
    };

    if !sidecar.full_state_snapshot {
        // Advisory frontier without a full-state snapshot — cannot anchor
        // safely at v1.0. Replay from zero.
        tracing::info!(
            target: "arcgraph_storage::checkpoint",
            checkpoint_lsn = sidecar.checkpoint_lsn.raw(),
            "checkpoint sidecar has no full-state snapshot flag — from-zero WAL replay",
        );
        return Ok(None);
    }

    // BLOCK-3: `read_snapshot` restores ONLY if magic + version + CRC +
    // header-LSN-matches-sidecar + full-structure all pass — ALL before
    // touching any owner. A mismatch / corrupt / structurally-bad snapshot
    // returns `Err(Corrupt)` with owners left PRISTINE (no TxnManager
    // watermark pollution) → we fall back to a genuine from-zero replay.
    // The sidecar's `checkpoint_lsn` is the expected frontier.
    match read_snapshot(data_dir, snap, sidecar.checkpoint_lsn) {
        Ok(Some((restored_lsn, counts))) => {
            debug_assert_eq!(restored_lsn, sidecar.checkpoint_lsn);
            tracing::info!(
                target: "arcgraph_storage::checkpoint",
                checkpoint_lsn = sidecar.checkpoint_lsn.raw(),
                mvcc_records = counts.mvcc_records,
                primary_pages = counts.primary_pages,
                record_pages = counts.record_pages,
                blob_pages = counts.blob_pages,
                "checkpoint restored — WAL replay anchored at checkpoint_lsn+1 (ADR-229 bound)",
            );
            Ok(Some(CheckpointRestore {
                checkpoint_lsn: sidecar.checkpoint_lsn,
                counts,
            }))
        }
        Ok(None) => {
            // Sidecar says full-state but the snapshot file is gone — a
            // torn establish. From-zero (SAFE).
            tracing::warn!(
                target: "arcgraph_storage::checkpoint",
                checkpoint_lsn = sidecar.checkpoint_lsn.raw(),
                "checkpoint sidecar present but snapshot file missing — from-zero WAL replay",
            );
            Ok(None)
        }
        Err(CheckpointError::Io(e)) => Err(CheckpointError::Io(e)),
        Err(e) => {
            // Corrupt / LSN-mismatch / structurally-bad — owners pristine.
            tracing::warn!(
                target: "arcgraph_storage::checkpoint",
                error = %e,
                "checkpoint snapshot invalid on open (corrupt / LSN-mismatch) — owners left \
                 pristine, falling back to from-zero WAL replay (SAFE)",
            );
            Ok(None)
        }
    }
}
