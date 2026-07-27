//! The checkpoint producer — the `checkpoint()` operation (ADR-229
//! §Decision).
//!
//! Establishes a checkpoint crash-atomically (both-or-neither):
//!
//! 1. **Catalog barrier.** Flush the catalog buffer pool durably to
//!    `pages.db` (`BufferPool::flush_all` → `PosixPageIo::flush` =
//!    `fdatasync`). This is the buffer.rs:719 "checkpointing must add its
//!    own barrier" step.
//! 2. **Full-state snapshot.** Serialize EVERY WAL-reconstructed owner
//!    (MVCC rows, primary/record/blob page images, allocator advances,
//!    intern names, idempotency bindings, permission grants) into a
//!    durable, crash-atomically-written snapshot file (OQ-2 — a miss is
//!    silent data loss; the snapshot covers all owners).
//! 3. **Sidecar.** Write the checkpoint frontier sidecar
//!    crash-atomically. This is the LAST step — the checkpoint is
//!    "established" only once the sidecar (which points at the durable
//!    snapshot from step 2) is renamed into place.
//!
//! A crash BETWEEN steps 2 and 3 leaves the PREVIOUS sidecar (or none)
//! valid; the orphaned new snapshot is overwritten by the next
//! checkpoint. A crash BETWEEN steps 1 and 2 likewise leaves the
//! previous checkpoint valid. NEVER a half-checkpoint that loses
//! committed data — recovery always falls back to the last fully
//! established checkpoint (or a from-zero replay) and replays more WAL.
//!
//! # Not the hot path (design-v2 §4.1)
//!
//! This runs on the background (Tokio work-stealing) pool or the
//! graceful-shutdown hook. It takes read guards on each owner's pages
//! but never blocks a foreground commit for the duration of the whole
//! snapshot (the snapshot is a best-effort point-in-time consistent at
//! `checkpoint_lsn`; a concurrent commit above the frontier is simply
//! replayed from the WAL on the next restart — correctness is preserved
//! because `checkpoint_lsn` is the last DURABLE commit at snapshot time).

use std::path::Path;

use arcgraph_core::Lsn;

use crate::buffer::BufferPool;
use crate::checkpoint::sidecar::{CheckpointError, CheckpointSidecar, write_sidecar_atomic};
use crate::checkpoint::snapshot::{
    CheckpointSnapshot, IncrementalMetadataCapture, IncrementalMetadataReport, SnapshotOwnerCounts,
    write_incremental_metadata_atomic,
};
use crate::checkpoint::write_behind::WriteBehindCheckpointer;
use crate::crud::DeferredV9Boundary;
use crate::wal::AllocatorAdvance;

/// Result of a successful `checkpoint()` — the established frontier +
/// what was captured (observability + the bounded-recovery oracle
/// surface).
#[derive(Debug, Clone, Copy)]
pub struct CheckpointReport {
    /// The established frontier: recovery now replays only WAL records
    /// with `commit_lsn > checkpoint_lsn`.
    pub checkpoint_lsn: Lsn,
    /// Per-owner entry counts captured in the snapshot.
    pub counts: SnapshotOwnerCounts,
}

/// Result of a durably-established v9 incremental checkpoint.
#[derive(Debug, Clone, Copy)]
pub struct IncrementalCheckpointReport {
    /// Highest installed commit represented by the checkpoint boundary.
    pub checkpoint_lsn: Lsn,
    /// ARIES replay/prune anchor (minimum post-flush DPT recLSN).
    pub redo_lsn: Lsn,
    /// Named lower bound for the outside-freeze metadata capture.
    pub capture_lsn: Lsn,
    /// Retained owner counts. MVCC and record-page counts are always zero.
    pub metadata: IncrementalMetadataReport,
}

/// Establish a v9/v10 incremental checkpoint over every registered physical
/// extent home.
///
/// Durability order is load-bearing:
///
/// 1. DWB fsync, then home-page writes + store fsync through the banked
///    write-behind pass;
/// 2. brief commit-freeze for the `{checkpoint_lsn, DPT}` boundary and its
///    allocator/deferred-apply capture;
/// 3. owner 2 + store-5 page images are captured under short commit freezes,
///    while owners 5-8 stream into an immutable metadata file;
/// 4. sidecar swap LAST, the sole establish point.
///
/// Owners 1/3/4's legacy full-state walks are structurally unreachable from
/// this path. Blob overflow is the Director-ruling exception: store 5 remains
/// a streamed PAGE-IMAGE section and never enters the DPT/delta set.
pub fn incremental_checkpoint<F, B>(
    data_dir: &Path,
    buffer_pool: &BufferPool,
    snap: &CheckpointSnapshot<'_>,
    write_behind: &WriteBehindCheckpointer,
    capture_boundary: F,
    mut establish_durability: B,
) -> Result<IncrementalCheckpointReport, CheckpointError>
where
    F: FnOnce() -> (Vec<AllocatorAdvance>, Option<DeferredV9Boundary>),
    B: FnMut(Lsn) -> Result<Lsn, CheckpointError>,
{
    crate::checkpoint::snapshot::sweep_incremental_metadata_temps(data_dir)?;
    // Drain already-fsynced Periodic applies before the write-behind pass
    // chooses its DPT/frontier. Without this first barrier, a quiescent queue
    // remains clamped for one extra checkpoint even though its WAL is already
    // durable, preventing both frontier advance and segment reclamation.
    let preflush_horizon = {
        let _freeze = snap.txn.checkpoint_freeze();
        snap.txn.current_lsn()
    };
    let preflush_durable_lsn = establish_durability(preflush_horizon)?;
    if preflush_durable_lsn < preflush_horizon {
        return Err(CheckpointError::Corrupt {
            reason: format!(
                "v9 pre-capture WAL horizon {} is below completed-commit horizon {}",
                preflush_durable_lsn.raw(),
                preflush_horizon.raw()
            ),
        });
    }

    // Page durability first. The strict entry point rejects a missing DWB;
    // metadata can never be established over an unprotected home pass.
    let pass_hint = snap.txn.current_lsn();
    let pass = write_behind
        .flush_pass_with_doublewrite(pass_hint)
        .map_err(crate::checkpoint::sidecar::arcgraph_err_to_io)?;

    // Catalog is still a separate durable owner during M3.
    buffer_pool
        .flush_all()
        .map_err(crate::checkpoint::sidecar::arcgraph_err_to_io)?;

    // The global freeze shrinks to the frontier + DPT observation. It is
    // released before any metadata-owner walk or synchronous metadata I/O.
    let (checkpoint_lsn, dpt, deferred, advances) = {
        let _freeze = snap.txn.checkpoint_freeze();
        let (advances, deferred) = capture_boundary();
        (
            snap.txn.current_lsn(),
            write_behind.metadata_dpt_snapshot(),
            deferred,
            advances,
        )
    };
    let checkpoint_lsn = deferred.map_or(checkpoint_lsn, |pending| {
        Lsn::new(
            checkpoint_lsn
                .raw()
                .min(pending.commit_lsn.raw().saturating_sub(1)),
        )
    });
    let dpt_redo_lsn = dpt
        .iter()
        .map(|entry| entry.rec_lsn)
        .min()
        .unwrap_or(checkpoint_lsn);
    let redo_lsn = deferred.map_or(dpt_redo_lsn, |pending| {
        Lsn::new(dpt_redo_lsn.raw().min(pending.redo_lsn.raw()))
    });

    // Metadata owners are absolute/idempotent; an outside-freeze capture may
    // overcapture effects above this named LSN and replay converges.
    let capture_lsn = snap.txn.current_lsn();
    let capture = IncrementalMetadataCapture {
        checkpoint_lsn,
        capture_lsn,
        redo_lsn,
        dpt: &dpt,
        advances: &advances,
    };
    let metadata = write_incremental_metadata_atomic(data_dir, snap, &capture)?;

    // Owner captures above may include a completed Periodic commit whose WAL
    // append was acknowledged before fsync. Take the completed-commit horizon
    // only after every retained owner is captured; the write guard waits out
    // any builder that could have contributed bytes. The production barrier
    // flushes WAL through this horizon and drains exact-durable deferred v9
    // applies before the sidecar can establish the metadata generation.
    let durability_horizon = {
        let _freeze = snap.txn.checkpoint_freeze();
        snap.txn.current_lsn()
    };
    let established_last_wal_lsn = establish_durability(durability_horizon)?;
    if established_last_wal_lsn < durability_horizon {
        return Err(CheckpointError::Corrupt {
            reason: format!(
                "v9 establishment WAL horizon {} is below captured commit horizon {}",
                established_last_wal_lsn.raw(),
                durability_horizon.raw()
            ),
        });
    }

    let sidecar = CheckpointSidecar::incremental(
        checkpoint_lsn,
        established_last_wal_lsn,
        now_unix_ms(),
        metadata.generation,
    );
    write_sidecar_atomic(data_dir, &sidecar)?;
    crate::checkpoint::snapshot::prune_incremental_metadata(
        data_dir,
        checkpoint_lsn,
        metadata.generation,
    )?;

    tracing::info!(
        target: "arcgraph_storage::checkpoint",
        checkpoint_lsn = checkpoint_lsn.raw(),
        redo_lsn = redo_lsn.raw(),
        capture_lsn = capture_lsn.raw(),
        metadata_generation = metadata.generation,
        flushed_pages = pass.flushed_pages,
        retained_redirties = pass.retained_redirties,
        dpt_entries = metadata.dpt_entries,
        primary_pages = metadata.counts.primary_pages,
        blob_overflow_pages = metadata.counts.blob_pages,
        allocator_advances = metadata.counts.allocator_advances,
        intern_names = metadata.counts.intern_names,
        idempotency_bindings = metadata.counts.idempotency_bindings,
        permission_docs = metadata.counts.permission_docs,
        metadata_peak_in_flight_bytes = metadata.max_in_flight,
        metadata_overflow_peak_resident_bytes = metadata.overflow_peak_resident,
        "M3 v9 incremental checkpoint established after DWB + home durability",
    );

    Ok(IncrementalCheckpointReport {
        checkpoint_lsn,
        redo_lsn,
        capture_lsn,
        metadata,
    })
}

/// Run a full-state checkpoint (ADR-229 §Decision), point-in-time
/// consistent against the commit path.
///
/// - `data_dir` — the durable data-dir (holds `pages.db`, `wal/`, and
///   the checkpoint sidecar + snapshot).
/// - `buffer_pool` — the catalog buffer pool (flushed to `pages.db`).
/// - `snap` — borrowed handles to every WAL-reconstructed owner (incl.
///   `snap.txn`, whose `checkpoint_freeze` guard this fn holds across the
///   whole capture).
/// - `collect_advances` — a closure invoked UNDER the commit-freeze that
///   returns the UNION of `PageAllocator::snapshot_advances` (page-kind
///   high-waters) + `CrudStore::snapshot_allocator_advances` (Node/Rel
///   high-waters). Draining under the freeze (AFTER the frontier read)
///   guarantees the captured allocator high-waters reflect every id
///   allocated by a commit visible at the frontier — closing BLOCK-1's
///   id-reuse skew window.
/// - `snapshot_last_wal_lsn` — the last durable WAL LSN (sidecar advisory).
///
/// # Consistency (BLOCK-1 + BLOCK-2 fix)
///
/// The entire capture — frontier read, MVCC walk, page-image byte-copy,
/// allocator drain — runs UNDER `snap.txn.checkpoint_freeze()` (the
/// commit/checkpoint WRITE guard). While held, no commit is between its
/// `counter.allocate()` and its `visible.store`, so:
/// - `checkpoint_lsn = current_lsn()` is a stable frontier;
/// - no id is allocated-but-absent-from-the-snapshot (BLOCK-1);
/// - no page image embeds a not-yet-WAL-durable commit (BLOCK-2);
/// - the whole owner set is captured against ONE quiescent instant.
///
/// Returns the [`CheckpointReport`]. The checkpoint is durably
/// established (both-or-neither) only when this returns `Ok` — the
/// snapshot is durable BEFORE the sidecar (the establishing step).
pub fn checkpoint<F>(
    data_dir: &Path,
    buffer_pool: &BufferPool,
    snap: &CheckpointSnapshot<'_>,
    collect_advances: F,
    snapshot_last_wal_lsn: Lsn,
) -> Result<CheckpointReport, CheckpointError>
where
    F: FnOnce() -> Vec<AllocatorAdvance>,
{
    // ── Capture phase — IN-RAM ONLY under the commit-freeze (REQ-2) ──
    //
    // The commit-freeze WRITE guard is held ONLY for the in-RAM capture:
    // the frontier read, allocator drain, MVCC walk, and the byte-copy of
    // RESIDENT pages. Page capture uses the NON-FAULTING
    // `iter_pages_resident_only` iterators, so the freeze NEVER blocks on a
    // synchronous disk `fault_in` — closing the ULTRACODE re-verify HIGH
    // availability regression (a periodic checkpoint at 10M with an
    // evicting buffer pool would otherwise stall EVERY foreground commit
    // for millions of disk reads). Any evicted page is RECORDED (id only)
    // under the guard and its durable disk image is read AFTER the guard
    // drops (below) — safe because a below-frontier page image is
    // immutable at the frontier, and any `> frontier` mutation is
    // idempotently re-applied by the anchored WAL replay. For the wired
    // pure-DashMap stores nothing is ever evicted, so the post-guard read
    // is a no-op and the snapshot is complete under the guard.
    //
    // # #1404 M0.5 — STREAMED snapshot (O(chunk) RSS, no whole-`Vec` spike)
    //
    // The whole-`Vec` path built ONE `Vec<u8>` holding the ENTIRE snapshot
    // (~18 GB @ 2M nodes) under the freeze, then wrote it whole → an RSS
    // spike 18.8→37 GB during a checkpoint burst → OOM @2M under a 40 G cap
    // (the 3rd #1404 memory term). We now STREAM the snapshot to disk in
    // O(chunk)-resident chunks: the temp file + its `BufWriter` sink are
    // opened BEFORE the freeze, the in-freeze sections stream UNDER the
    // freeze (same BLOCK-2 byte-consistency — page bytes copied while frozen
    // — but one page resident at a time, NOT the whole snapshot), the freeze
    // releases, then the post-guard evicted supplement + footer stream, and
    // the file is fsync'd ONCE + renamed. ADR-229 crash-atomicity is
    // byte-untouched (unique temp → fsync → rename; sidecar LAST establishes).
    let mut writer = crate::checkpoint::snapshot::StreamingSnapshotWrite::open(data_dir)?;
    let (checkpoint_lsn, counts, evicted) = {
        let _freeze = snap.txn.checkpoint_freeze();
        // Frontier read FIRST (under the freeze), THEN drain advances
        // (BLOCK-1 ordering: an advance drained after the frontier read,
        // while frozen, can only be ≥ the frontier-implied high-water).
        let checkpoint_lsn = snap.txn.current_lsn();
        let advances = collect_advances();
        // Stream the header + MVCC records + RESIDENT page sections +
        // allocator/intern/idempotency/permissions UNDER the freeze. Page
        // bytes are read + streamed one page at a time (BLOCK-2: copied
        // while frozen), so nothing but the `BufWriter` + one page is
        // resident. Evicted page-ids are RECORDED (not faulted; REQ-2) for
        // the post-guard supplement.
        let (counts, evicted) = crate::checkpoint::snapshot::encode_snapshot_streaming(
            snap,
            checkpoint_lsn,
            &advances,
            writer.sink(),
        )?;
        (checkpoint_lsn, counts, evicted)
    };
    // ── Guard RELEASED — commits resume. All work below is disk I/O
    //    (evicted-page backfill + supplement/footer stream + fsync) and must
    //    NOT hold the freeze. ──

    // REQ-2 post-guard backfill: STREAM the evicted pages' durable disk
    // images (OUTSIDE the freeze) as the evicted supplement, PAGE-BY-PAGE.
    // #1404 M0.5 ultracode-REJECT fix: the prior `read_evicted_page_images`
    // pre-collected ALL evicted images into a `Vec` (one owned
    // `Box<[u8; PAGE_SIZE]>` per evicted page, held at once) BEFORE streaming
    // — O(N-above-the-watermark) ≈ 74 GB @10M, the EXACT whole-`Vec` OOM class
    // #1404 exists to fix, on the on-by-default durable serve path. We now
    // read + emit + DROP each evicted page one-at-a-time (≤ one page +
    // BufWriter resident), matching the body-section streaming. The wire
    // layout is byte-identical (tag, count = evicted.blob.len(), then per-image
    // {owner_tag, tenant, pid, PAGE}), so the whole-`Vec` byte-identity
    // differential still holds. For the wired pure-DashMap stores `evicted` is
    // empty → a zero-count section (no disk read). The fail-loud primary/record
    // wiring-bug check is MOVED ahead of the write inside the streamer.
    let supplement_peak_resident =
        crate::checkpoint::snapshot::stream_evicted_supplement(writer.sink(), snap, &evicted)?;
    crate::checkpoint::snapshot::finalize_snapshot_streaming(writer.sink())?;

    let stream_stats = writer.stats();

    // Step 1 — catalog barrier: flush + fdatasync pages.db.
    buffer_pool
        .flush_all()
        .map_err(crate::checkpoint::sidecar::arcgraph_err_to_io)?;

    // Step 2 — full-state snapshot durable (crash-atomic: fsync the streamed
    // temp ONCE + rename + dir-fsync). Durable BEFORE the sidecar. A crash
    // before this rename leaves the partial temp orphaned (ignored on
    // recovery) — the crash-mid-stream contract.
    writer.finalize_atomic()?;

    // Step 3 — sidecar (crash-atomic). LAST step: established only once
    // this rename lands. A crash before here leaves the previous
    // checkpoint valid (both-or-neither).
    let created_unix_ms = now_unix_ms();
    let sidecar =
        CheckpointSidecar::full_state(checkpoint_lsn, snapshot_last_wal_lsn, created_unix_ms);
    write_sidecar_atomic(data_dir, &sidecar)?;

    tracing::info!(
        target: "arcgraph_storage::checkpoint",
        checkpoint_lsn = checkpoint_lsn.raw(),
        snapshot_last_wal_lsn = snapshot_last_wal_lsn.raw(),
        mvcc_records = counts.mvcc_records,
        primary_pages = counts.primary_pages,
        record_pages = counts.record_pages,
        blob_pages = counts.blob_pages,
        allocator_advances = counts.allocator_advances,
        intern_names = counts.intern_names,
        idempotency_bindings = counts.idempotency_bindings,
        permission_docs = counts.permission_docs,
        // #1404 M0.5 — bounded-resident proof: peak in-flight snapshot bytes
        // (O(chunk) = one page/record, NOT the O(total) whole-`Vec`) + the
        // total streamed body size. `snapshot_peak_in_flight_bytes` «
        // `snapshot_body_bytes` is the RSS win (no whole-in-RAM spike).
        snapshot_peak_in_flight_bytes = stream_stats.max_in_flight,
        snapshot_body_bytes = stream_stats.body_len,
        // #1404 M0.5 ultracode-REJECT fix — peak caller-resident EVICTED
        // SUPPLEMENT bytes: 0 when nothing evicted, else exactly `PAGE_SIZE`
        // (one page at a time), NOT the O(N-above-watermark) whole-`Vec`
        // ~74 GB @10M the pre-collect path held. O(1) in the evicted-count.
        snapshot_evicted_supplement_peak_resident_bytes = supplement_peak_resident,
        "ADR-229 checkpoint established (full-state snapshot streamed + sidecar durable)",
    );

    Ok(CheckpointReport {
        checkpoint_lsn,
        counts,
    })
}

fn now_unix_ms() -> i64 {
    // Determinism-oracle pin (INV-M5.24, cfg-gated + bounded per the
    // standing test-hook rule): `created_unix_ms` is the ONLY wall-clock
    // byte in the checkpoint sidecar, and pinning it lets the M5-D3
    // byte-identical gate compare whole generations with no exclusion
    // list. Compiled out of production builds.
    #[cfg(feature = "fault-injection")]
    if let Some(pinned) = std::env::var("ARCGRAPH_CHECKPOINT_UNIX_MS")
        .ok()
        .and_then(|raw| raw.parse().ok())
    {
        return pinned;
    }
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
        .unwrap_or(0)
}
