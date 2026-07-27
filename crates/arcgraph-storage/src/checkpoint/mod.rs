//! WAL checkpoint subsystem (SVC-1 P1, ADR-229).
//!
//! # Why this exists (the rc-blocker)
//!
//! ArcGraph's WAL is append-only with **no lifecycle bound** and every
//! restart replays the *entire* history. At 10M scale (#849, CZ-proven)
//! a 167 GB WAL could not restart in 8.5+ minutes — an un-restartable
//! store is a data-availability failure. A GA engine MUST restart in
//! bounded time. This module adds the **checkpoint producer** +
//! **checkpoint-anchored recovery** that bounds restart-recovery to
//! `O(WAL-since-checkpoint)` instead of `O(entire-history)`.
//!
//! # The v1.0 durability reality (ADR-229 OQ-2 — read this before you
//! touch anything)
//!
//! At v1.0, `pages.db` durably holds ONLY the catalog root page
//! (`SystemCatalog`, `CATALOG_PAGE_ID`). **Every other durable-state
//! owner is reconstructed from the WAL on restart** and lives in
//! in-memory `DashMap`-backed stores:
//!
//! - the MVCC version store (`TxnManager::versions`) — node/rel rows;
//! - the primary-index page store (`PrimaryPageStore`);
//! - the record page store (`RecordPageStore`);
//! - the BLOB store (`BlobStore`);
//! - the page allocator high-water marks (`PageAllocator`);
//! - the intern table (`InternTable`) — label/rel-type names;
//! - the idempotency store (`IdempotencyStore`);
//! - the per-tenant permission index (`PermissionIndex`).
//!
//! (Source-verified: `backup.rs` states the durable set is `pages.db +
//! wal/*` and restore "leaves loading to the standard boot recovery
//! (`recover_from_wal`)"; the page stores are pure `DashMap`s that never
//! call `write_page`.)
//!
//! **Consequence — the highest-risk invariant (OQ-2):** if recovery is
//! anchored to skip WAL records at or below a checkpoint frontier, then
//! ALL of the state above must be durable *outside* the WAL at that
//! frontier, or restart silently loses every committed effect the WAL
//! would otherwise have replayed. Flushing only `pages.db` (the catalog)
//! is NOT sufficient. A checkpoint that anchors recovery therefore MUST
//! capture a durable snapshot of **every** owner above.
//!
//! This module makes the anchoring **safe by construction**: the
//! [`CheckpointSidecar`] records `checkpoint_lsn` AND a
//! `full_state_snapshot` flag. Recovery only replays-from-frontier when
//! the checkpoint carries a full-state snapshot; a frontier without a
//! full-state snapshot degrades to a from-zero replay (see
//! [`crate::wal::recover_from_wal`]). This is the both-or-neither
//! crash-atomicity contract of ADR-229 §Decision.
//!
//! # Crash-atomicity (ADR-229 §Decision)
//!
//! A checkpoint is "established" only when BOTH the state snapshot AND
//! the sidecar are durable — both-or-neither. The producer:
//!
//! 1. flushes the catalog buffer pool (`pages.db`) durably;
//! 2. writes the full-state snapshot to a temp file, fsyncs it,
//!    atomically renames it into place, then fsyncs the directory;
//! 3. writes the sidecar to a temp file, fsyncs it, atomically renames
//!    it over the previous sidecar, then fsyncs the directory.
//!
//! A crash BETWEEN (2) and (3) leaves the PREVIOUS sidecar (or none)
//! pointing at the PREVIOUS (or no) frontier — recovery falls back and
//! replays more WAL; never a half-checkpoint that loses committed data.
//! The rename+dir-fsync durability mirrors `truncate_torn_tail`
//! (recovery.rs) and the vector-arena snapshot pattern.
//!
//! # Budget (PD#5)
//!
//! Checkpoint is a background operation on the Tokio work-stealing pool
//! (design-v2 §4.1) or the graceful-shutdown hook — NEVER the
//! thread-per-core hot path. Cost is I/O-bound on the state snapshot
//! (one page-image write per live page + one record per live MVCC row).
//! The default trigger interval (see [`crate::config::WalCheckpointConfig`])
//! is sized so the steady-state WAL-since-checkpoint (and therefore
//! restart-recovery replay) stays bounded regardless of uptime.

mod doublewrite;
mod producer;
mod recovery;
mod sidecar;
mod snapshot;
mod write_behind;

pub use doublewrite::{
    DOUBLEWRITE_FILE, DoublewriteArea, DoublewriteKey, DoublewriteRestoreReport,
    DoublewriteRestoreTarget, ExtentDirectoryDoublewriteHome, M3DoublewriteHome,
};
pub use producer::{
    CheckpointReport, IncrementalCheckpointReport, checkpoint, incremental_checkpoint,
};
pub use recovery::{
    IncrementalCheckpointRestore, restore_latest_checkpoint, restore_latest_incremental_checkpoint,
};
pub use sidecar::{
    CHECKPOINT_SIDECAR_FILE, CheckpointError, CheckpointSidecar, read_latest_sidecar,
    write_sidecar_atomic,
};
pub(crate) use snapshot::prune_incremental_metadata;
pub use snapshot::{
    CHECKPOINT_INCREMENTAL_PREFIX, CHECKPOINT_SNAPSHOT_FILE, CheckpointSnapshot,
    INCREMENTAL_METADATA_FORMAT_VERSION, IncrementalCheckpointMetadata, IncrementalMetadataReport,
    IncrementalOwnerVisitor, SnapshotOwnerCounts, incremental_metadata_path,
    incremental_temp_sweep_dir_fsync_count, read_incremental_metadata, read_snapshot,
    retire_incremental_lookup_owner_sections, sweep_incremental_metadata_temps,
    visit_incremental_metadata_owners, write_snapshot_atomic, write_snapshot_bytes_atomic,
};
pub use write_behind::{
    BlobPageFlushTarget, DEFAULT_WRITE_BEHIND_BATCH_PAGES, PageFlushTarget,
    WriteBehindCheckpointer, WriteBehindReport,
};
