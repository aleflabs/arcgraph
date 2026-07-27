//! WAL segment reclamation (SVC-1 P2, ADR-229 §Segment reclamation).
//!
//! # Why this exists (the second half of the #849 rc-blocker)
//!
//! SVC-1 P1 (#1371) built the checkpoint **producer** + checkpoint-anchored
//! **recovery**, so a restart replays only `O(WAL-since-checkpoint)`. But
//! P1 alone does NOT shrink the WAL on disk — every segment ever written is
//! still there. CZ proved the concrete failure at 10M: a **167 GB WAL** that
//! filled the disk and could not restart. Bounding *recovery time* without
//! bounding *WAL size* leaves the disk-fill week-1 killer open. This module
//! is the piece that actually **deletes** WAL segments whose committed
//! effects are already durable in the checkpoint snapshot, so steady-state
//! WAL size stays bounded regardless of uptime.
//!
//! # THE correctness invariant (ADR-229 §Consequences — data loss if wrong)
//!
//! > *No committed effect below `checkpoint_lsn` may be absent from the
//! > snapshot when its WAL segment is reclaimed.*
//!
//! P1's checkpoint is a **full-state snapshot** capturing all 8
//! WAL-reconstructed owners at the frontier, so an effect at or below
//! `checkpoint_lsn` IS in the snapshot. This module's job is therefore
//! narrow and absolute: **delete a segment ONLY if the checkpoint has
//! provably captured every committed effect in it** — i.e. every
//! `CommitBundle` record in the segment has `commit_lsn <= checkpoint_lsn`.
//! Delete a segment holding a `commit_lsn > checkpoint_lsn` and that commit
//! is silently LOST on the next restart (the anchored replay skips at/below
//! the frontier and the segment is gone). Getting this wrong = silent data
//! loss. Every branch here is written to fail SAFE: when in doubt, keep the
//! segment (recovery replays a little more; never loses).
//!
//! # LSN spaces — the load-bearing subtlety (do NOT conflate)
//!
//! There are TWO independent LSN counters (verified against
//! `wal::writer` + `TxnManager`):
//!
//! - **`record.lsn`** — the WAL framing/append counter. Assigned by the
//!   writer thread (`lsn_counter += 1`). **Resets to 0 on every restart**
//!   (`build_durable` uses plain `WalWriter::spawn`). USELESS as a
//!   persistent reclamation floor.
//! - **`commit_lsn`** — the MVCC commit LSN carried INSIDE each
//!   `CommitBundle` payload (`DecodedCommitBundle::commit_lsn`), = the
//!   `TxnManager` watermark. This is the space the checkpoint frontier
//!   lives in and what recovery anchors on.
//!
//! Reclamation MUST compare against `commit_lsn`, so this module DECODES
//! each candidate segment's bundle records to derive its **max
//! `commit_lsn`** (there is no cheaper per-segment LSN index at v1.0). The
//! decode reuses the exact same [`decode_commit_bundle_for_version`] the
//! recovery path uses, dispatched by each segment's own header version —
//! so a segment we deem reclaimable is one recovery would also have read
//! identically.
//!
//! # Contiguous-prefix policy (preserve recovery's segment ordering)
//!
//! `WalRecoveryReader` replays segments in ascending order and treats a
//! decode failure in a **non-terminal** segment as hard corruption. We
//! therefore reclaim only a **contiguous prefix** of the lowest segments:
//! walk ascending, delete while a segment is provably-below-frontier, and
//! STOP at the first segment that is NOT (any `commit_lsn > frontier`, a
//! decode failure, or the active segment). This never leaves a hole in the
//! segment sequence, and it never deletes the segment "containing or after
//! `checkpoint_lsn`" (ADR-229 §Segment reclamation) because that segment —
//! and everything after it — is on the STOP side of the prefix.
//!
//! # Durability (mirror `truncate_torn_tail`)
//!
//! Each deletion is `remove_file` + a directory fsync, exactly mirroring
//! [`truncate_torn_tail`](crate::wal::truncate_torn_tail)'s durable-delete
//! pattern (recovery.rs) so a crash mid-reclamation leaves the directory in
//! a consistent state (a segment is either fully present or fully gone; the
//! dir-fsync makes the unlink durable). Reclamation runs ONLY AFTER the
//! checkpoint sidecar is durably established, so even a crash that loses the
//! dir-fsync (segment reappears) is harmless — the checkpoint is valid and
//! recovery replays the reappeared segment idempotently.

use std::path::Path;

use arcgraph_core::{ArcGraphError, Lsn, Result};

use crate::wal::bundle::decode_commit_bundle_for_version;
use crate::wal::record::{WalRecord, WalRecordType};
use crate::wal::segment::{SegmentHeader, fsync_dir, list_segments, segment_filename};

/// Outcome of a [`reclaim_segments_below`] pass — observability + the
/// bounded-WAL oracle surface. All counts are for a single pass.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReclaimReport {
    /// Segment numbers deleted this pass (the reclaimed contiguous prefix).
    pub deleted_segments: Vec<u64>,
    /// Total bytes freed (sum of the deleted segment files' sizes).
    pub bytes_freed: u64,
    /// The segment number reclamation STOPPED at (the first non-reclaimable
    /// segment — kept), or `None` if every non-active segment was reclaimed
    /// or there was nothing to consider.
    pub stopped_at_segment: Option<u64>,
    /// Why the pass stopped (diagnostic; never an error — stopping is the
    /// SAFE default).
    pub stop_reason: StopReason,
}

/// Why a reclamation pass stopped advancing the reclaimable prefix.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum StopReason {
    /// Reached the active (currently-appended) segment — never reclaimed.
    #[default]
    ReachedActiveSegment,
    /// A segment held a `commit_lsn > checkpoint_lsn` (not yet captured by
    /// the checkpoint) — this segment and everything after it are kept.
    AboveFrontier,
    /// A segment failed to decode (torn tail / corruption) — kept
    /// conservatively so recovery can still read it.
    UndecodableKept,
    /// Nothing to reclaim (0 or 1 segments, or `checkpoint_lsn == ZERO`).
    Nothing,
}

/// Reclaim (durably delete) the contiguous prefix of WAL segments whose
/// committed effects are ALL at or below `checkpoint_lsn` — the segments
/// the checkpoint's full-state snapshot has provably captured.
///
/// - `wal_dir` — the WAL segment directory.
/// - `checkpoint_lsn` — the durably-established checkpoint frontier
///   (**commit-LSN space**; see the module docs). `Lsn::ZERO` = no
///   checkpoint ⇒ reclaim nothing (the SAFE no-op).
///
/// # Safety contract (ADR-229 §Consequences — data loss if violated)
///
/// A segment is reclaimed IFF **every** `CommitBundle` record in it has
/// `commit_lsn <= checkpoint_lsn` AND it is not the active segment. The
/// active segment is the highest-numbered one (the writer only ever appends
/// to / rotates forward from the highest). We reclaim only a contiguous
/// ascending prefix and STOP at the first segment that is not
/// provably-below-frontier, is undecodable, or is the active segment — so
/// we never delete the segment "containing or after `checkpoint_lsn`", never
/// leave a hole in the sequence, and never delete a segment whose effects
/// the checkpoint has NOT captured.
///
/// MUST be called ONLY AFTER the checkpoint sidecar + snapshot are durably
/// established (both-or-neither) — reclaiming before the snapshot is durable
/// would drop a segment whose effects are not yet in any durable snapshot.
///
/// # Errors
///
/// Returns [`ArcGraphError::Io`] only on a genuine filesystem fault while
/// listing / stat-ing / unlinking / dir-fsyncing. A decode failure is NOT
/// an error — it STOPS the prefix (the segment is kept). A caller that gets
/// an `Err` should log it and continue serving: reclamation is an
/// optimization, and a failed pass leaves every segment intact (correct,
/// just not shrunk this time).
pub fn reclaim_segments_below(wal_dir: &Path, checkpoint_lsn: Lsn) -> Result<ReclaimReport> {
    // No checkpoint ⇒ nothing is provably captured ⇒ reclaim nothing.
    if checkpoint_lsn == Lsn::ZERO {
        return Ok(ReclaimReport {
            stop_reason: StopReason::Nothing,
            ..ReclaimReport::default()
        });
    }

    let segments = list_segments(wal_dir)?;
    // 0 or 1 segments: the single segment (if any) is the active one and is
    // never reclaimed. Nothing to do.
    if segments.len() <= 1 {
        return Ok(ReclaimReport {
            stop_reason: StopReason::Nothing,
            ..ReclaimReport::default()
        });
    }

    // The active segment is the highest-numbered — the writer only appends
    // to / rotates forward from it. It is NEVER reclaimable (records are
    // still landing in it and its frontier bound is not yet closed).
    let active_segment = *segments
        .last()
        .expect("len > 1 checked above, last is Some");

    let mut report = ReclaimReport::default();
    // Walk the ascending prefix. Delete while provably-below-frontier; STOP
    // (keep this segment + all after it) on the first that is not.
    for &seg_no in &segments {
        if seg_no == active_segment {
            report.stopped_at_segment = Some(seg_no);
            report.stop_reason = StopReason::ReachedActiveSegment;
            break;
        }
        match segment_max_commit_lsn(wal_dir, seg_no)? {
            SegmentScan::MaxCommitLsn(max) => {
                if max.raw() > checkpoint_lsn.raw() {
                    // This segment carries an effect the checkpoint has NOT
                    // captured — keep it (and everything after). SAFE stop.
                    report.stopped_at_segment = Some(seg_no);
                    report.stop_reason = StopReason::AboveFrontier;
                    break;
                }
                // Provably below the frontier: every commit in this segment
                // is captured by the snapshot. Durably delete it.
                let freed = delete_segment_durable(wal_dir, seg_no)?;
                report.deleted_segments.push(seg_no);
                report.bytes_freed = report.bytes_freed.saturating_add(freed);
            }
            SegmentScan::Undecodable => {
                // A torn / corrupt segment — do NOT delete it (recovery must
                // be able to read the prefix up to the tear). STOP here.
                report.stopped_at_segment = Some(seg_no);
                report.stop_reason = StopReason::UndecodableKept;
                break;
            }
        }
    }

    if !report.deleted_segments.is_empty() {
        tracing::info!(
            target: "arcgraph_storage::wal::reclaim",
            checkpoint_lsn = checkpoint_lsn.raw(),
            deleted = report.deleted_segments.len(),
            bytes_freed = report.bytes_freed,
            stopped_at = ?report.stopped_at_segment,
            stop_reason = ?report.stop_reason,
            "ADR-229 P2: reclaimed WAL segments below checkpoint frontier",
        );
    }
    Ok(report)
}

/// Result of scanning a segment for its maximum `commit_lsn`.
enum SegmentScan {
    /// The segment decoded cleanly; this is the max `commit_lsn` of every
    /// `CommitBundle` in it. A segment with NO commit bundles (only
    /// `Begin`/`Checkpoint`/etc.) yields `Lsn::ZERO` — it holds no committed
    /// effect, so it is trivially at/below any non-zero frontier.
    MaxCommitLsn(Lsn),
    /// The segment could not be fully decoded (torn tail / corruption). It
    /// is kept conservatively (never reclaimed).
    Undecodable,
}

/// Scan one WAL segment and return the maximum `commit_lsn` across all its
/// `CommitBundle` records, or [`SegmentScan::Undecodable`] if any record
/// fails to decode (torn / corrupt) — in which case the segment is kept.
///
/// This decodes exactly as recovery does: it reads the segment's own header
/// version and dispatches each `CommitBundle` payload through
/// [`decode_commit_bundle_for_version`], so a segment we classify as
/// reclaimable is one recovery would have read identically. We do NOT
/// short-circuit on the first bundle: we must observe the MAX `commit_lsn`
/// over the WHOLE segment, because a single record with
/// `commit_lsn > checkpoint_lsn` anywhere in the segment makes the entire
/// segment non-reclaimable.
///
/// Encrypted-WAL note: the outer `WalRecord` header (record_type + lsn +
/// framing) stays in clear even when payloads are AEAD-wrapped (see
/// `wal::writer::encode_for_fire`). `CommitBundle`'s `commit_lsn` is the
/// FIRST field of the (encrypted) payload, so with encryption on we cannot
/// read it without the key. We handle that fail-SAFE: an encrypted payload
/// that will not decode is treated as [`SegmentScan::Undecodable`] → the
/// segment is KEPT (never mis-reclaimed). Reclamation of encrypted WALs is
/// therefore a no-op today; wiring the DEK here is a follow-on (it only ever
/// makes reclamation MORE aggressive, never less safe).
fn segment_max_commit_lsn(wal_dir: &Path, seg_no: u64) -> Result<SegmentScan> {
    let path = wal_dir.join(segment_filename(seg_no));
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            // Raced away (already reclaimed by a concurrent pass — the
            // producer_mutex serializes checkpoints, but be defensive).
            // Treat as "nothing committed here" so the prefix can advance.
            return Ok(SegmentScan::MaxCommitLsn(Lsn::ZERO));
        }
        Err(e) => return Err(ArcGraphError::Io(e)),
    };

    // A segment shorter than its header is a torn creation tail — keep it
    // (recovery folds it into torn-tail semantics; never our job to delete).
    if bytes.len() < SegmentHeader::SIZE {
        return Ok(SegmentScan::Undecodable);
    }
    let header = match SegmentHeader::decode(&bytes[..SegmentHeader::SIZE]) {
        Ok(h) => h,
        // Bad magic / unsupported version: keep it — a version we cannot
        // decode is one we must not delete (an upgrade path may need it).
        Err(_) => return Ok(SegmentScan::Undecodable),
    };

    let mut cursor = SegmentHeader::SIZE;
    let mut max_commit = Lsn::ZERO;
    while cursor < bytes.len() {
        let (record, consumed) = match WalRecord::decode(&bytes[cursor..]) {
            Ok(pair) => pair,
            // Any decode failure (truncated tail, CRC, framing) → keep the
            // whole segment. We never reclaim a segment we cannot fully read
            // (a hidden `commit_lsn > frontier` past the tear would be lost).
            Err(_) => return Ok(SegmentScan::Undecodable),
        };
        if record.record_type == WalRecordType::CommitBundle {
            match decode_commit_bundle_for_version(
                &record.payload,
                header.format_version,
                record.tenant_id,
            ) {
                Ok(bundle) => {
                    if bundle.commit_lsn.raw() > max_commit.raw() {
                        max_commit = bundle.commit_lsn;
                    }
                }
                // Cannot decode this bundle (e.g. an encrypted payload with
                // no key, or a format we don't understand) → fail SAFE: keep
                // the segment. We must NOT reclaim a segment whose committed
                // frontier we cannot bound.
                Err(_) => return Ok(SegmentScan::Undecodable),
            }
        }
        cursor += consumed;
    }
    Ok(SegmentScan::MaxCommitLsn(max_commit))
}

/// Durably delete segment `seg_no`: `remove_file` + directory fsync,
/// mirroring [`truncate_torn_tail`](crate::wal::truncate_torn_tail)'s
/// durable-delete discipline (recovery.rs:56-63). Returns the freed byte
/// count (the file's size before deletion) for the [`ReclaimReport`].
///
/// The dir-fsync makes the unlink durable so a crash after this returns
/// leaves the segment truly gone. A crash BETWEEN `remove_file` and the
/// dir-fsync may resurrect the segment on some filesystems — which is
/// HARMLESS: the checkpoint is already durable, so recovery replays the
/// resurrected segment's below-frontier bundles idempotently (Lemma I2). We
/// never lose data by a lost unlink; we only fail to shrink the WAL that one
/// time.
fn delete_segment_durable(wal_dir: &Path, seg_no: u64) -> Result<u64> {
    let path = wal_dir.join(segment_filename(seg_no));
    let freed = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    match std::fs::remove_file(&path) {
        Ok(()) => {}
        // Already gone (raced) — treat as success; nothing to fsync-away.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(e) => return Err(ArcGraphError::Io(e)),
    }
    fsync_dir(wal_dir)?;
    Ok(freed)
}

/// The active (currently-appended) segment number in `wal_dir` — the
/// highest-numbered segment. `None` when the WAL directory holds no
/// segments. Exposed so a caller (the reclamation trigger) can reason about
/// which segment is off-limits without duplicating the "highest = active"
/// rule.
#[must_use]
pub fn active_segment_number(segments: &[u64]) -> Option<u64> {
    segments.last().copied()
}

/// Convenience: the number of segments currently on disk in `wal_dir`. Used
/// by tests + observability to assert the WAL is bounded.
pub fn segment_count(wal_dir: &Path) -> Result<usize> {
    Ok(list_segments(wal_dir)?.len())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use arcgraph_core::{PAGE_SIZE, PageId, TenantId};
    use bytes::Bytes;
    use tempfile::tempdir;

    use super::*;
    use crate::wal::bundle::encode_commit_bundle_v8;
    use crate::wal::segment::SegmentWriter;
    use crate::wal::writer::{WalConfig, WalWriter};

    fn wal_cfg(dir: &Path, segment_size_bytes: u64) -> WalConfig {
        WalConfig {
            dir: dir.to_path_buf(),
            segment_size_bytes,
            group_commit_window: std::time::Duration::from_millis(2),
            group_commit_max_batch: 1,
            metrics_sink: None,
            encryption: None,
            inflight_budget_bytes: None,
        }
    }

    /// Append one v8 CommitBundle at `commit_lsn` (small payload → each
    /// bundle in its own segment when segment_size is tiny).
    fn append_bundle(handle: &crate::wal::WalHandle, commit_lsn: u64) {
        let mut mvcc: HashMap<u64, Option<Bytes>> = HashMap::new();
        mvcc.insert(commit_lsn, Some(Bytes::from(format!("v{commit_lsn}"))));
        let staged: Vec<(_, PageId, TenantId, Box<[u8; PAGE_SIZE]>)> = vec![(
            crate::wal::BundlePageKind::PrimaryIndex,
            PageId::new(1000 + commit_lsn),
            TenantId::DEFAULT,
            Box::new([(commit_lsn % 256) as u8; PAGE_SIZE]),
        )];
        let payload = encode_commit_bundle_v8(
            Lsn::new(commit_lsn),
            TenantId::DEFAULT,
            &mvcc,
            &[],
            &staged,
            &[],
            &[],
            &[],
            &[],
        );
        handle
            .append(
                WalRecordType::CommitBundle,
                1,
                0,
                TenantId::DEFAULT,
                payload,
            )
            .unwrap();
    }

    /// Write `n` bundles (commit_lsn 1..=n), each large enough that with a
    /// tiny segment_size each lands in its own segment. Returns the WAL dir.
    fn write_n_segments(dir: &Path, n: u64) {
        // A bundle carries a full PAGE_SIZE page, so a segment_size just
        // under PAGE_SIZE forces one bundle per segment.
        let writer = WalWriter::spawn(wal_cfg(dir, 64)).unwrap();
        let handle = writer.handle();
        for lsn in 1..=n {
            append_bundle(&handle, lsn);
        }
        writer.shutdown().unwrap();
    }

    #[test]
    fn zero_frontier_reclaims_nothing() {
        let dir = tempdir().unwrap();
        write_n_segments(dir.path(), 5);
        let before = segment_count(dir.path()).unwrap();
        let report = reclaim_segments_below(dir.path(), Lsn::ZERO).unwrap();
        assert!(report.deleted_segments.is_empty());
        assert_eq!(report.stop_reason, StopReason::Nothing);
        assert_eq!(segment_count(dir.path()).unwrap(), before);
    }

    #[test]
    fn single_segment_never_reclaimed() {
        let dir = tempdir().unwrap();
        // One big segment holding several bundles.
        let writer = WalWriter::spawn(wal_cfg(dir.path(), 64 * 1024 * 1024)).unwrap();
        let handle = writer.handle();
        for lsn in 1..=5 {
            append_bundle(&handle, lsn);
        }
        writer.shutdown().unwrap();
        assert_eq!(segment_count(dir.path()).unwrap(), 1);
        // Frontier well above every commit — still must NOT delete the sole
        // (active) segment.
        let report = reclaim_segments_below(dir.path(), Lsn::new(100)).unwrap();
        assert!(report.deleted_segments.is_empty());
        assert_eq!(report.stop_reason, StopReason::Nothing);
        assert_eq!(segment_count(dir.path()).unwrap(), 1);
    }

    /// DISCRIMINATING active-segment-protection test (ULTRACODE w4y85944p
    /// REQUIRED FIX): isolate the active-segment guard so NOTHING ELSE can
    /// prevent deleting the live/highest segment.
    ///
    /// Every commit is BELOW the frontier (frontier is set ABOVE the max
    /// commit_lsn), so the `AboveFrontier` stop can NEVER fire — the loop
    /// would delete EVERY segment, INCLUDING the active (highest) one, if not
    /// for the `seg_no == active_segment` guard at the top of the loop. This
    /// is the scenario the `AboveFrontier` stop cannot mask: the active guard
    /// is the SOLE thing keeping the live segment alive.
    ///
    /// RED-on-revert: disable the active guard (remove the
    /// `if seg_no == active_segment { break }` early-out) → the active/highest
    /// segment is DELETED → `active_survives` is false → this test FAILS. That
    /// deletion is data loss + a bricked WAL (the writer's open segment
    /// vanished), which no other test here catches because they all rely on
    /// the `AboveFrontier` stop firing first.
    #[test]
    fn active_segment_protected_in_isolation_all_below_frontier() {
        let dir = tempdir().unwrap();
        // 5 segments, commit_lsn 1..=5 (one bundle per tiny segment).
        write_n_segments(dir.path(), 5);
        let segs_before = list_segments(dir.path()).unwrap();
        assert!(
            segs_before.len() >= 3,
            "need multiple segments to isolate the active guard, got {segs_before:?}",
        );
        let active = *segs_before.last().unwrap();

        // Frontier ABOVE every commit_lsn (max is 5) → the `AboveFrontier`
        // stop is UNREACHABLE. Every segment is provably-below-frontier, so
        // the ONLY reason the active segment is not deleted is the active
        // guard.
        let report = reclaim_segments_below(dir.path(), Lsn::new(1_000_000)).unwrap();

        // The active (highest) segment MUST survive.
        let after = list_segments(dir.path()).unwrap();
        let active_survives = after.contains(&active);
        assert!(
            active_survives,
            "active segment {active} was DELETED — the active-segment guard failed (data loss + \
             bricked WAL). remaining {after:?}, report {report:?}",
        );
        // And the pass stopped BECAUSE it reached the active segment — NOT
        // because of an AboveFrontier / Undecodable stop (which would mask an
        // active-guard regression). This pins the discriminating property.
        assert_eq!(
            report.stop_reason,
            StopReason::ReachedActiveSegment,
            "the ONLY safe stop here is ReachedActiveSegment (frontier is above every commit, so \
             AboveFrontier is unreachable); a different stop reason means something OTHER than \
             the active guard protected the segment — the test would then not discriminate",
        );
        assert_eq!(report.stopped_at_segment, Some(active));
        // Every NON-active segment below the frontier was reclaimed (the guard
        // protects ONLY the active one, not the whole WAL).
        for seg in &segs_before {
            if *seg != active {
                assert!(
                    report.deleted_segments.contains(seg),
                    "non-active below-frontier segment {seg} should have been reclaimed: {report:?}",
                );
                assert!(
                    !after.contains(seg),
                    "reclaimed segment {seg} still present"
                );
            }
        }
    }

    #[test]
    fn reclaims_prefix_below_frontier_keeps_active() {
        let dir = tempdir().unwrap();
        write_n_segments(dir.path(), 6); // segments ~0..=5, commit_lsn 1..=6
        let segs = list_segments(dir.path()).unwrap();
        assert!(segs.len() >= 3, "need multiple segments, got {segs:?}");
        let active = *segs.last().unwrap();

        // Frontier at commit_lsn 3 → segments whose max commit_lsn <= 3 are
        // reclaimable (the low prefix), stopping before the first segment
        // with a commit_lsn > 3, and never the active segment.
        let report = reclaim_segments_below(dir.path(), Lsn::new(3)).unwrap();
        assert!(
            !report.deleted_segments.is_empty(),
            "some low segments must be reclaimed at frontier=3: {report:?}"
        );
        // The active segment must survive.
        let after = list_segments(dir.path()).unwrap();
        assert!(
            after.contains(&active),
            "active segment {active} must never be reclaimed; remaining {after:?}"
        );
        // No deleted segment may still exist.
        for d in &report.deleted_segments {
            assert!(!after.contains(d), "deleted segment {d} still present");
        }
    }

    #[test]
    fn never_deletes_segment_at_or_above_frontier() {
        // THE data-loss boundary test: a segment holding commit_lsn ABOVE
        // the frontier must NEVER be deleted. RED-on-revert: if the guard
        // `max > frontier` is weakened (compare removed / made never-fire), a
        // segment with a commit above the frontier lands in the `deleted` set
        // → the assert below FIRES (data loss caught). We snapshot each
        // segment's max commit_lsn BEFORE reclamation, because a deleted
        // file reads back as ZERO — that hole is exactly what would MASK the
        // loss if we scanned post-deletion.
        let dir = tempdir().unwrap();
        write_n_segments(dir.path(), 6); // commit_lsn 1..=6 across segments
        let segs = list_segments(dir.path()).unwrap();

        let mut pre: std::collections::HashMap<u64, u64> = std::collections::HashMap::new();
        for seg in &segs {
            if let SegmentScan::MaxCommitLsn(max) =
                segment_max_commit_lsn(dir.path(), *seg).unwrap()
            {
                pre.insert(*seg, max.raw());
            }
        }

        let frontier = 2u64;
        let report = reclaim_segments_below(dir.path(), Lsn::new(frontier)).unwrap();

        // No DELETED segment may have held a commit above the frontier.
        for deleted in &report.deleted_segments {
            let max = *pre.get(deleted).expect("pre-scanned every segment");
            assert!(
                max <= frontier,
                "DATA LOSS: segment {deleted} held commit_lsn {max} > frontier {frontier} but \
                 was DELETED (report {report:?})",
            );
        }
        // And every segment with a commit above the frontier still exists.
        let after = list_segments(dir.path()).unwrap();
        for (seg, max) in &pre {
            if *max > frontier {
                assert!(
                    after.contains(seg),
                    "segment {seg} (max commit_lsn {max} > frontier {frontier}) must survive; \
                     remaining {after:?}",
                );
            }
        }
    }

    #[test]
    fn stops_at_undecodable_segment_keeps_it() {
        // A corrupt low segment must STOP the prefix — never deleted, and
        // nothing after it deleted either (contiguity).
        let dir = tempdir().unwrap();
        write_n_segments(dir.path(), 5);
        let segs = list_segments(dir.path()).unwrap();
        assert!(segs.len() >= 3);
        // Corrupt the FIRST (lowest) segment's first record body so its
        // decode fails, but keep a valid header.
        let low = segs[0];
        let path = dir.path().join(segment_filename(low));
        let mut bytes = std::fs::read(&path).unwrap();
        // Flip a byte just past the header + record header to break framing.
        let off = SegmentHeader::SIZE + WalRecord::HEADER_SIZE + 2;
        if off < bytes.len() {
            bytes[off] ^= 0xFF;
        }
        std::fs::write(&path, &bytes).unwrap();

        let report = reclaim_segments_below(dir.path(), Lsn::new(100)).unwrap();
        // Nothing deleted: the very first candidate is undecodable → stop.
        assert!(report.deleted_segments.is_empty(), "report {report:?}");
        assert_eq!(report.stop_reason, StopReason::UndecodableKept);
        assert_eq!(report.stopped_at_segment, Some(low));
        assert!(list_segments(dir.path()).unwrap().contains(&low));
    }

    #[test]
    fn segment_with_no_commit_bundles_is_below_any_frontier() {
        // A segment holding only non-CommitBundle records (e.g. Begin) has
        // max commit_lsn ZERO → reclaimable under any non-zero frontier.
        let dir = tempdir().unwrap();
        let writer = WalWriter::spawn(wal_cfg(dir.path(), 64)).unwrap();
        let handle = writer.handle();
        // Two Begin records (tiny) → likely two segments; then a bundle to
        // create an active segment above them.
        handle
            .append(WalRecordType::Begin, 1, 0, TenantId::DEFAULT, vec![0u8; 40])
            .unwrap();
        handle
            .append(WalRecordType::Begin, 2, 0, TenantId::DEFAULT, vec![0u8; 40])
            .unwrap();
        append_bundle(&handle, 7);
        writer.shutdown().unwrap();

        let scan0 = segment_max_commit_lsn(dir.path(), 0).unwrap();
        assert!(
            matches!(scan0, SegmentScan::MaxCommitLsn(l) if l == Lsn::ZERO),
            "a Begin-only segment has no committed effect → max commit_lsn ZERO"
        );
    }

    #[test]
    fn active_segment_number_is_highest() {
        assert_eq!(active_segment_number(&[0, 1, 2, 7]), Some(7));
        assert_eq!(active_segment_number(&[]), None);
    }

    #[test]
    fn deleted_segments_free_bytes_and_dir_stays_readable() {
        let dir = tempdir().unwrap();
        write_n_segments(dir.path(), 5);
        let report = reclaim_segments_below(dir.path(), Lsn::new(3)).unwrap();
        if !report.deleted_segments.is_empty() {
            assert!(report.bytes_freed > 0, "freed bytes must be > 0");
        }
        // The directory must still be openable by a SegmentWriter (no hole,
        // header intact on the survivors).
        let _w = SegmentWriter::open(dir.path(), 64 * 1024 * 1024).unwrap();
    }
}
