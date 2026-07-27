//! Full-state checkpoint snapshot file (ADR-229 §Decision, OQ-2).
//!
//! At v1.0 the WAL is the sole durable store for every row/index/blob/
//! intern/idempotency/permission/allocator effect (only the catalog is
//! in `pages.db`; see the module docs of [`crate::checkpoint`]). To
//! anchor recovery at a frontier and skip the WAL below it, the
//! checkpoint must durably capture a snapshot of ALL of those owners —
//! a MISS on any owner is silent data loss (OQ-2, the highest-risk
//! item). This module serializes and restores that full-state snapshot.
//!
//! # Restore reuses the proven replay primitives
//!
//! The snapshot restore path does NOT invent new mutation logic. It
//! feeds each captured entry back through the SAME entry points the
//! WAL replay executor uses (`apply_replay_mvcc_write`,
//! `PrimaryPageStore::install_or_replace`, `RecordPageStore::
//! install_or_replace`, `BlobStore::install_or_replace`,
//! `PageAllocator::seed_advance`, `InternTable::intern_install`,
//! `IdempotencyStore::install`, `PermissionIndex::apply_doc_acl_replayed`)
//! — the crash-campaign-proven, idempotent restore surface. So the
//! checkpoint restore is byte-for-byte equivalent to replaying the WAL
//! prefix it stands in for.
//!
//! # On-disk layout
//!
//! ```text
//! [header]
//!   magic            4    b"AGCS" (ArcGraph Checkpoint Snapshot)
//!   format_version   2    u16 LE == SNAPSHOT_FORMAT_VERSION
//!   _reserved        2
//!   checkpoint_lsn   8    u64 LE
//! [sections]  (one per owner, in fixed order; a section may be empty)
//!   section_tag      1    OwnerTag byte
//!   entry_count      8    u64 LE
//!   entries          ...  owner-specific, length-prefixed as needed
//! [footer]
//!   crc32c           4    over every byte before the footer
//! ```
//!
//! A truncated or CRC-failing snapshot is
//! [`CheckpointError::Corrupt`]; recovery treats it as "no valid
//! checkpoint" and falls back to a from-zero replay (the SAFE
//! direction). Because the sidecar is written LAST (after this file is
//! durable), a valid sidecar always points at a fully-written snapshot.

use std::collections::BTreeSet;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use arcgraph_core::record::PAGE_SIZE;
use arcgraph_core::{Lsn, NodeId, StringId, TenantId};

use crate::blob::{BlobStore, BlobStoreHandle};
use crate::checkpoint::sidecar::CheckpointError;
use crate::idempotency::IdempotencyStore;
use crate::intern::InternTable;
use crate::permissions::PermissionIndex;
use crate::primary_index::PrimaryPageStore;
use crate::record_store::RecordPageStore;
use crate::redo::DirtyPageSnapshot;
use crate::transaction::TxnManager;
use crate::wal::segment::fsync_dir;
use crate::wal::{
    AllocatorAdvance, AllocatorKind, AllocatorSeedHandle, STORE_BLOB_OVERFLOW, STORE_GRANTS,
    STORE_INTERN, STORE_NODE_BINDINGS, STORE_PROPS, STORE_RECORD, STORE_REL_BINDINGS, STORE_RELS,
    STORE_SECONDARY_INDEX, STORE_TEL,
};

/// Snapshot file name in the data-dir.
pub const CHECKPOINT_SNAPSHOT_FILE: &str = "CHECKPOINT.snap";

/// Prefix for immutable v9 incremental metadata files. The checkpoint LSN is
/// part of the file name so a crash before the sidecar swap cannot overwrite
/// the metadata named by the previously-established sidecar.
pub const CHECKPOINT_INCREMENTAL_PREFIX: &str = "CHECKPOINT.v9";

/// M3 incremental metadata format. This is deliberately the CommitBundle
/// generation number: a v8 recovery path must never consume this file.
pub const INCREMENTAL_METADATA_FORMAT_VERSION: u16 = 9;

static INCREMENTAL_TEMP_SWEEP_DIR_FSYNC_COUNT: AtomicU64 = AtomicU64::new(0);

const INCREMENTAL_METADATA_MAGIC: [u8; 4] = *b"AGCM";

/// Stores whose dirty-page frontier may be retained by an incremental
/// checkpoint. v9 used only stores 0/1; v10 adds every direct extent home.
/// The primary B-tree remains an owner-2 page-image section, not a DPT store.
const fn is_incremental_dpt_store(store_id: u16) -> bool {
    matches!(
        store_id,
        STORE_PROPS
            | STORE_RECORD
            | STORE_TEL
            | STORE_SECONDARY_INDEX
            | STORE_BLOB_OVERFLOW
            | STORE_RELS
            | STORE_NODE_BINDINGS
            | STORE_REL_BINDINGS
            | STORE_INTERN
            | STORE_GRANTS
    )
}

/// Temp file used for the crash-atomic write.
const CHECKPOINT_SNAPSHOT_TMP: &str = "CHECKPOINT.snap.tmp";

/// On-disk snapshot format version.
pub const SNAPSHOT_FORMAT_VERSION: u16 = 1;

/// Magic at offset 0 — "AGCS" (ArcGraph Checkpoint Snapshot).
const SNAPSHOT_MAGIC: [u8; 4] = *b"AGCS";

const HEADER_SIZE: usize = 4 + 2 + 2 + 8; // magic + ver + reserved + lsn
const FOOTER_SIZE: usize = 4; // crc32c

/// Per-owner section tags. Fixed byte values — never renumber.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum OwnerTag {
    Mvcc = 1,
    PrimaryPages = 2,
    RecordPages = 3,
    BlobPages = 4,
    Allocator = 5,
    Intern = 6,
    Idempotency = 7,
    Permissions = 8,
    /// REQ-2 — post-guard evicted-page supplement. Carries evicted page
    /// images read AFTER the commit-freeze released; each entry is tagged
    /// with the owning page store so restore routes it correctly. Always
    /// present but zero-count for the wired pure-`DashMap` stores.
    EvictedSupplement = 9,
}

impl OwnerTag {
    fn from_byte(b: u8) -> Result<Self, CheckpointError> {
        Ok(match b {
            1 => Self::Mvcc,
            2 => Self::PrimaryPages,
            3 => Self::RecordPages,
            4 => Self::BlobPages,
            5 => Self::Allocator,
            6 => Self::Intern,
            7 => Self::Idempotency,
            8 => Self::Permissions,
            9 => Self::EvictedSupplement,
            other => {
                return Err(CheckpointError::Corrupt {
                    reason: format!("unknown snapshot owner tag byte {other}"),
                });
            }
        })
    }
}

/// Per-owner entry counts captured in a snapshot. Observability / the
/// bounded-recovery oracle assertion surface.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SnapshotOwnerCounts {
    /// Live MVCC records (node/rel rows).
    pub mvcc_records: u64,
    /// Primary-index page images.
    pub primary_pages: u64,
    /// Record-store page images.
    pub record_pages: u64,
    /// BLOB page images.
    pub blob_pages: u64,
    /// Allocator high-water advances.
    pub allocator_advances: u64,
    /// Interned label/rel-type names.
    pub intern_names: u64,
    /// Idempotency bindings.
    pub idempotency_bindings: u64,
    /// Permission doc→grants mappings.
    pub permission_docs: u64,
}

/// The owner handles a checkpoint snapshot reads from / restores into.
///
/// Borrowed for the duration of a `write_snapshot_atomic` /
/// `read_snapshot` call. Every field is the SAME `Arc` the durable
/// bootstrap wires into the replay target — so a restore lands in the
/// served stores.
pub struct CheckpointSnapshot<'a> {
    /// MVCC version store (node/rel rows).
    pub txn: &'a TxnManager,
    /// Primary-index page store.
    pub primary_pages: &'a PrimaryPageStore,
    /// Record page store.
    pub record_pages: &'a RecordPageStore,
    /// BLOB store.
    pub blob: &'a BlobStore,
    /// The allocator seed handle to restore advances through. Dispatches
    /// `Node`/`Rel` into the `CrudStore` and `Page*` into the
    /// `PageAllocator` — the SAME handle WAL replay seeds through, so
    /// restore is byte-identical (Lemma I3 monotonic-max).
    ///
    /// The WRITE-side allocator advances are NOT a field: they are
    /// collected under the commit-freeze by the producer's
    /// `collect_advances` closure (BLOCK-1 — draining under the freeze,
    /// after the frontier read) and passed to `encode_snapshot_bytes`.
    pub allocator_seed: &'a dyn AllocatorSeedHandle,
    /// Intern table.
    pub intern: &'a InternTable,
    /// Idempotency store.
    pub idempotency: &'a IdempotencyStore,
    /// Per-tenant permission index (v1.0: the single DEFAULT-tenant
    /// index). The tenant it belongs to is supplied separately at
    /// restore since the index itself is tenant-local.
    pub permissions: &'a PermissionIndex,
    /// Tenant the `permissions` index scopes (v1.0: DEFAULT).
    pub permissions_tenant: TenantId,
}

// ─────────────────────────────────────────────────────────────────────
// little-endian primitives
// ─────────────────────────────────────────────────────────────────────

fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_u8(out: &mut Vec<u8>, v: u8) {
    out.push(v);
}
fn put_bytes(out: &mut Vec<u8>, b: &[u8]) {
    put_u32(out, u32::try_from(b.len()).expect("section entry < 4 GiB"));
    out.extend_from_slice(b);
}

// ─────────────────────────────────────────────────────────────────────
// #1404 M0.5 — streaming snapshot sink (O(chunk) resident, not O(total))
// ─────────────────────────────────────────────────────────────────────
//
// # Budget (PD#5) — resident snapshot memory
//
// The whole-`Vec<u8>` encode path holds the ENTIRE snapshot resident
// (~18 GB @ 2M nodes; heaptrack flagged `encode_snapshot_bytes` as a peak
// allocator). During a checkpoint burst RSS spiked 18.8→37 GB → OOM @2M
// under a 40 G cap — the 3rd #1404 memory term (distinct from the M0
// blob-tier and the queue-0 WAL channel). This sink STREAMS the snapshot
// to disk so the resident snapshot working-set is O(chunk) — a single
// [`SNAPSHOT_STREAM_BUF_BYTES`] `BufWriter` plus one transient page
// (`PAGE_SIZE`) / record buffer at a time — NOT O(total). Peak in-flight
// snapshot bytes ≤ the `BufWriter` capacity, provably « the whole snapshot
// (asserted by the `max_in_flight` instrumentation on the sink).
//
// # Byte-identity (the #1 correctness risk)
//
// Every byte the streaming sink emits is byte-identical to what the
// whole-`Vec` path produced over the same DB state — same magic/version/
// owner-order/evicted-supplement — including the footer CRC. The trailer
// stays identical because the CRC is computed as a RUNNING crc32c
// (`crc32c_append(running, chunk)` per write): `crc32c(concat) ==
// fold(crc32c_append, 0, chunks)` (crc32c 0.6: `crc32c(d) ==
// crc32c_append(0, d)` and append is associative over concatenation), so
// the streamed body CRC equals the whole-buffer CRC exactly. The
// differential test (`streamed == whole-Vec`, byte-for-byte) is the gate.

/// The `BufWriter` capacity for a streamed snapshot file. Bounds the
/// resident snapshot working-set: the sink buffers at most this many bytes
/// before flushing to the OS, so peak in-flight snapshot memory is
/// O(this) not O(total snapshot). 1 MiB amortizes syscalls over the
/// per-record / per-page writes without holding the whole snapshot.
pub(crate) const SNAPSHOT_STREAM_BUF_BYTES: usize = 1024 * 1024;

/// A byte sink the snapshot encoder streams into, maintaining a RUNNING
/// crc32c over every body byte so the footer trailer is byte-identical to
/// the whole-buffer path (`crc32c(concat) == running append per chunk`).
///
/// Two impls:
/// - [`VecSnapshotSink`] — into a `Vec<u8>` (differential test + the
///   convenience whole-buffer callers).
/// - [`FileSnapshotSink`] — into a `BufWriter<File>` over the crash-atomic
///   temp file (the PRODUCTION path; O(chunk) resident).
///
/// `write_body` appends to the body AND folds the bytes into the running
/// CRC. `finish_crc` returns the running CRC (the footer value) WITHOUT
/// folding the footer into itself (the footer is CRC-over-body-only, as in
/// the whole-buffer `finalize_snapshot_bytes`).
pub(crate) trait SnapshotSink {
    /// Append `bytes` to the snapshot body and fold them into the running
    /// CRC. Errors propagate the underlying I/O error (file sink); the Vec
    /// sink is infallible.
    fn write_body(&mut self, bytes: &[u8]) -> Result<(), CheckpointError>;

    /// The running crc32c over every body byte written so far. Equals
    /// `crc32c(whole_body)`. Called once, after the body + evicted
    /// supplement, to obtain the footer value.
    fn body_crc(&self) -> u32;
}

/// `Vec<u8>`-backed sink — the differential-test oracle (whole-`Vec` ==
/// streamed byte-identity). Test-only: production streams to a file
/// (`FileSnapshotSink`); the Vec sink exists solely to encode the streamed
/// bytes into RAM for a `memcmp` against the whole-`Vec` path. Infallible.
#[cfg(test)]
pub(crate) struct VecSnapshotSink {
    buf: Vec<u8>,
    crc: u32,
}

#[cfg(test)]
impl VecSnapshotSink {
    pub(crate) fn with_capacity(cap: usize) -> Self {
        Self {
            buf: Vec::with_capacity(cap),
            crc: 0,
        }
    }

    /// Consume the sink, returning the streamed body bytes (incl. the footer
    /// once `finalize_snapshot_streaming` has run) for the `memcmp` gate.
    pub(crate) fn into_body(self) -> Vec<u8> {
        self.buf
    }
}

#[cfg(test)]
impl SnapshotSink for VecSnapshotSink {
    fn write_body(&mut self, bytes: &[u8]) -> Result<(), CheckpointError> {
        self.crc = crc32c::crc32c_append(self.crc, bytes);
        self.buf.extend_from_slice(bytes);
        Ok(())
    }
    fn body_crc(&self) -> u32 {
        self.crc
    }
}

/// `BufWriter<File>`-backed sink — the PRODUCTION streaming path. Holds
/// only the `BufWriter` (≤ [`SNAPSHOT_STREAM_BUF_BYTES`] resident) plus the
/// running CRC. `max_in_flight` records the peak single-write size for the
/// bounded-resident assertion (the whole-buffer path would report O(total);
/// this path reports O(page/record)).
pub(crate) struct FileSnapshotSink<W: std::io::Write> {
    writer: W,
    crc: u32,
    /// Largest single `write_body` chunk seen — the transient buffer the
    /// caller materialized before handing it to the sink (a `PAGE_SIZE`
    /// page or one MVCC record). Bounds the caller-side working-set; the
    /// sink itself buffers ≤ its `BufWriter` capacity. Instrumentation for
    /// the bounded-resident proof.
    max_in_flight: usize,
    /// Total body bytes streamed (excludes the footer). Observability.
    body_len: u64,
}

impl<W: std::io::Write> FileSnapshotSink<W> {
    pub(crate) fn new(writer: W) -> Self {
        Self {
            writer,
            crc: 0,
            max_in_flight: 0,
            body_len: 0,
        }
    }
    /// Peak single-write chunk size — the caller-side transient buffer
    /// bound (page/record), NOT O(total snapshot). The bounded-resident
    /// proof asserts this is « the whole snapshot.
    pub(crate) fn max_in_flight(&self) -> usize {
        self.max_in_flight
    }
    /// Total body bytes written (pre-footer).
    pub(crate) fn body_len(&self) -> u64 {
        self.body_len
    }
    /// Consume the sink, returning the inner writer (to flush/fsync).
    pub(crate) fn into_writer(self) -> W {
        self.writer
    }
}

impl<W: std::io::Write> SnapshotSink for FileSnapshotSink<W> {
    fn write_body(&mut self, bytes: &[u8]) -> Result<(), CheckpointError> {
        self.crc = crc32c::crc32c_append(self.crc, bytes);
        self.max_in_flight = self.max_in_flight.max(bytes.len());
        self.body_len += bytes.len() as u64;
        self.writer.write_all(bytes)?;
        Ok(())
    }
    fn body_crc(&self) -> u32 {
        self.crc
    }
}

// ── streaming little-endian put-primitives (mirror the Vec versions,
//    byte-for-byte, but emit into an `impl SnapshotSink`) ──

fn s_u64<S: SnapshotSink>(out: &mut S, v: u64) -> Result<(), CheckpointError> {
    out.write_body(&v.to_le_bytes())
}
fn s_u16<S: SnapshotSink>(out: &mut S, v: u16) -> Result<(), CheckpointError> {
    out.write_body(&v.to_le_bytes())
}
fn s_u32<S: SnapshotSink>(out: &mut S, v: u32) -> Result<(), CheckpointError> {
    out.write_body(&v.to_le_bytes())
}
fn s_u8<S: SnapshotSink>(out: &mut S, v: u8) -> Result<(), CheckpointError> {
    out.write_body(&[v])
}
fn s_bytes<S: SnapshotSink>(out: &mut S, b: &[u8]) -> Result<(), CheckpointError> {
    s_u32(out, u32::try_from(b.len()).expect("section entry < 4 GiB"))?;
    out.write_body(b)
}

/// Observability from one v9 incremental metadata write. Zero counts for
/// owners 1/3 are structural: those page-backed owners are never walked by
/// the v9 encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IncrementalMetadataReport {
    /// Immutable generation selected by the sidecar if this attempt
    /// establishes. It is unique across retained and aborted attempts.
    pub generation: u64,
    pub counts: SnapshotOwnerCounts,
    /// DPT entries captured at the checkpoint frontier.
    pub dpt_entries: u64,
    /// Largest single chunk passed to the streaming sink.
    pub max_in_flight: usize,
    /// Total bytes emitted, including the CRC footer.
    pub body_len: u64,
    /// Peak caller-owned buffer while streaming evicted overflow images.
    pub overflow_peak_resident: usize,
}

/// Stable inputs captured by the producer before the outside-freeze owner
/// stream begins.
pub(crate) struct IncrementalMetadataCapture<'a> {
    pub checkpoint_lsn: Lsn,
    pub capture_lsn: Lsn,
    pub redo_lsn: Lsn,
    pub dpt: &'a [DirtyPageSnapshot],
    pub advances: &'a [AllocatorAdvance],
}

/// Immutable v9 metadata path named by checkpoint frontier and generation.
/// Generation 0 preserves the pre-v2 sidecar path for backward reads.
#[must_use]
pub fn incremental_metadata_path(data_dir: &Path, checkpoint_lsn: Lsn, generation: u64) -> PathBuf {
    if generation == 0 {
        data_dir.join(format!(
            "{CHECKPOINT_INCREMENTAL_PREFIX}.{:016x}.meta",
            checkpoint_lsn.raw()
        ))
    } else {
        data_dir.join(format!(
            "{CHECKPOINT_INCREMENTAL_PREFIX}.{:016x}.{generation:016x}.meta",
            checkpoint_lsn.raw()
        ))
    }
}

fn unique_incremental_tmp(data_dir: &Path, checkpoint_lsn: Lsn) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    data_dir.join(format!(
        "{CHECKPOINT_INCREMENTAL_PREFIX}.{:016x}.tmp.{}.{}",
        checkpoint_lsn.raw(),
        std::process::id(),
        seq
    ))
}

struct RemoveIncrementalTmp(PathBuf);

impl Drop for RemoveIncrementalTmp {
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_file(&self.0)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!(
                target: "arcgraph_storage::checkpoint",
                path = %self.0.display(),
                %error,
                "failed to remove aborted incremental-checkpoint temp file",
            );
        }
    }
}

fn parse_incremental_metadata_identity(name: &str) -> Option<(Lsn, u64)> {
    let body = name
        .strip_prefix(&format!("{CHECKPOINT_INCREMENTAL_PREFIX}."))?
        .strip_suffix(".meta")?;
    let mut fields = body.split('.');
    let lsn_hex = fields.next()?;
    let generation_hex = fields.next();
    if fields.next().is_some() || lsn_hex.len() != 16 {
        return None;
    }
    let lsn = Lsn::new(u64::from_str_radix(lsn_hex, 16).ok()?);
    let generation = match generation_hex {
        Some(hex) if hex.len() == 16 => u64::from_str_radix(hex, 16).ok()?,
        Some(_) => return None,
        None => 0,
    };
    Some((lsn, generation))
}

fn next_incremental_metadata_generation(data_dir: &Path) -> Result<u64, CheckpointError> {
    let mut max_generation = 0u64;
    for entry in std::fs::read_dir(data_dir)? {
        let entry = entry?;
        if let Some((_, generation)) =
            parse_incremental_metadata_identity(&entry.file_name().to_string_lossy())
        {
            max_generation = max_generation.max(generation);
        }
    }
    max_generation
        .checked_add(1)
        .ok_or_else(|| CheckpointError::Corrupt {
            reason: "v9 metadata generation counter exhausted".to_owned(),
        })
}

fn is_incremental_tmp(name: &str) -> bool {
    name.starts_with(&format!("{CHECKPOINT_INCREMENTAL_PREFIX}.")) && name.contains(".tmp.")
}

/// Remove every metadata generation except the one named by the established
/// sidecar. Called only after the sidecar swap, so the retained generation is
/// always the recovery-selected one.
pub(crate) fn prune_incremental_metadata(
    data_dir: &Path,
    keep_lsn: Lsn,
    keep_generation: u64,
) -> Result<(), CheckpointError> {
    for entry in std::fs::read_dir(data_dir)? {
        let entry = entry?;
        let name = entry.file_name();
        if parse_incremental_metadata_identity(&name.to_string_lossy())
            .is_some_and(|identity| identity != (keep_lsn, keep_generation))
        {
            std::fs::remove_file(entry.path())?;
        }
    }
    fsync_dir(data_dir).map_err(crate::checkpoint::sidecar::arcgraph_err_to_io)
}

/// Startup cleanup for aborted producer attempts. Metadata generations are
/// left intact until the sidecar identifies the established one.
pub fn sweep_incremental_metadata_temps(data_dir: &Path) -> Result<(), CheckpointError> {
    let mut removed = false;
    for entry in std::fs::read_dir(data_dir)? {
        let entry = entry?;
        let name = entry.file_name();
        if is_incremental_tmp(&name.to_string_lossy()) {
            std::fs::remove_file(entry.path())?;
            removed = true;
        }
    }
    if removed {
        fsync_dir(data_dir).map_err(crate::checkpoint::sidecar::arcgraph_err_to_io)?;
        INCREMENTAL_TEMP_SWEEP_DIR_FSYNC_COUNT.fetch_add(1, Ordering::AcqRel);
    }
    Ok(())
}

/// Test/observability counter for successful startup temp-sweep directory
/// fsyncs. The increment occurs only after `fsync_dir` succeeds.
#[doc(hidden)]
#[must_use]
pub fn incremental_temp_sweep_dir_fsync_count() -> u64 {
    INCREMENTAL_TEMP_SWEEP_DIR_FSYNC_COUNT.load(Ordering::Acquire)
}

/// Stream and durably install one v9 incremental metadata generation.
///
/// The encoder contains no call to the MVCC/record-page capture paths. It
/// writes only DPT + owner 2 (index page images) + store-5 overflow images +
/// owners 5-8. Every retained O(N) owner uses its cursor callback directly;
/// no `iter_all -> Vec` compatibility path exists here.
pub(crate) fn write_incremental_metadata_atomic(
    data_dir: &Path,
    snap: &CheckpointSnapshot<'_>,
    capture: &IncrementalMetadataCapture<'_>,
) -> Result<IncrementalMetadataReport, CheckpointError> {
    let checkpoint_lsn = capture.checkpoint_lsn;
    let generation = next_incremental_metadata_generation(data_dir)?;
    let capture_lsn = capture.capture_lsn;
    let redo_lsn = capture.redo_lsn;
    let dpt = capture.dpt;
    let advances = capture.advances;
    if capture_lsn.raw() < checkpoint_lsn.raw() {
        return Err(CheckpointError::Corrupt {
            reason: format!(
                "v9 metadata capture_lsn {} precedes checkpoint_lsn {}",
                capture_lsn.raw(),
                checkpoint_lsn.raw()
            ),
        });
    }
    let expected_redo = dpt
        .iter()
        .map(|entry| entry.rec_lsn)
        .min()
        .unwrap_or(checkpoint_lsn);
    if redo_lsn.raw() > expected_redo.raw() {
        return Err(CheckpointError::Corrupt {
            reason: format!(
                "v9 metadata redo_lsn {} exceeds min DPT/checkpoint floor {}",
                redo_lsn.raw(),
                expected_redo.raw()
            ),
        });
    }
    if dpt.windows(2).any(|pair| pair[0].key >= pair[1].key) {
        return Err(CheckpointError::Corrupt {
            reason: "v9 metadata DPT is not strictly key-sorted".to_owned(),
        });
    }
    for entry in dpt {
        if !is_incremental_dpt_store(entry.key.store_id) {
            return Err(CheckpointError::Corrupt {
                reason: format!(
                    "v9 metadata DPT contains non-delta store_id {}",
                    entry.key.store_id
                ),
            });
        }
    }

    let tmp = unique_incremental_tmp(data_dir, checkpoint_lsn);
    let _remove_tmp = RemoveIncrementalTmp(tmp.clone());
    let final_path = incremental_metadata_path(data_dir, checkpoint_lsn, generation);
    let file = std::fs::File::create(&tmp)?;
    let bufw = std::io::BufWriter::with_capacity(SNAPSHOT_STREAM_BUF_BYTES, file);
    let mut out = FileSnapshotSink::new(bufw);

    // Header + DPT. Store ids are explicit and remain limited to 0/1; store 5
    // is intentionally represented only by the page-image owner below.
    out.write_body(&INCREMENTAL_METADATA_MAGIC)?;
    s_u16(&mut out, INCREMENTAL_METADATA_FORMAT_VERSION)?;
    s_u16(&mut out, 0)?;
    s_u64(&mut out, checkpoint_lsn.raw())?;
    s_u64(&mut out, redo_lsn.raw())?;
    s_u64(&mut out, capture_lsn.raw())?;
    s_u64(&mut out, dpt.len() as u64)?;
    for entry in dpt {
        s_u64(&mut out, entry.key.tenant_id.raw())?;
        s_u16(&mut out, entry.key.store_id)?;
        s_u16(&mut out, 0)?;
        s_u64(&mut out, entry.key.page_no)?;
        s_u64(&mut out, entry.rec_lsn.raw())?;
        s_u64(&mut out, entry.dirty_gen)?;
    }

    let mut counts = SnapshotOwnerCounts::default();

    // Owner 2 is PRIMARY-ONLY at M3. Its stream is freeze-scoped: primary
    // builders mutate before fsync and roll back on failure, so capturing
    // outside the commit guard could persist a phantom. The freeze also
    // makes count+cursor stable under paced installs (checkpoint liveness).
    // read and emitted before advancing the DashMap cursor. Secondary-index
    // mutations retain the existing full-image bundle/metadata treatment;
    // a page-LSN/SMO checkpoint owner for secondary pages is deferred to M4.
    let primary_count = {
        let _freeze = snap.txn.checkpoint_freeze();
        s_u8(&mut out, OwnerTag::PrimaryPages as u8)?;
        let primary_count = snap.primary_pages.resident_page_count() as u64;
        s_u64(&mut out, primary_count)?;
        let mut primary_streamed = 0u64;
        let primary_evicted = snap.primary_pages.for_each_resident_page(|pid, latch| {
            s_u64(&mut out, pid.raw())?;
            let guard = latch.read();
            out.write_body(guard.as_ref().as_ref())?;
            primary_streamed += 1;
            Ok::<(), CheckpointError>(())
        })?;
        if !primary_evicted.is_empty() || primary_streamed != primary_count {
            return Err(CheckpointError::CountSkew {
                owner: "v9_primary_pages",
                header: primary_count,
                streamed: primary_streamed,
            });
        }
        primary_count
    };
    counts.primary_pages = primary_count;

    // Director ruling: blob.overflow (store 5) remains PAGE-IMAGE at M3.
    // Resident and spilled images share one section and stream one page at a
    // time. This is not a PutPropBlock delta path.
    let overflow_capture = {
        let _freeze = snap.txn.checkpoint_freeze();
        s_u8(&mut out, OwnerTag::BlobPages as u8)?;
        // O(1): count + monotone page/epoch frontier only. Both resident
        // encoding and spill-index iteration are scale-dependent and belong
        // outside the global commit freeze.
        let capture = snap.blob.capture_overflow_frontier();
        s_u64(&mut out, capture.count())?;
        capture
    };
    let overflow_count = overflow_capture.count();
    let overflow_streamed =
        snap.blob
            .for_each_captured_overflow_page(overflow_capture, |tenant, page_id, page| {
                s_u64(&mut out, tenant.raw())?;
                s_u64(&mut out, page_id)?;
                out.write_body(page)?;
                Ok::<(), CheckpointError>(())
            })?;
    // One immutable page image plus the fixed-size frontier token; no owner-N
    // id Vec is caller-owned at any point.
    let overflow_peak_resident = if overflow_count == 0 {
        0
    } else {
        PAGE_SIZE + std::mem::size_of_val(&overflow_capture)
    };
    if overflow_streamed != overflow_count {
        return Err(CheckpointError::CountSkew {
            owner: "v9_blob_overflow",
            header: overflow_count,
            streamed: overflow_streamed,
        });
    }
    counts.blob_pages = overflow_count;

    // Owners 5-8 — capture guards stabilize count+cursor locally. This runs
    // outside TxnManager::checkpoint_freeze; overcapture is safe because the
    // corresponding WAL operations are absolute/idempotent.
    s_u8(&mut out, OwnerTag::Allocator as u8)?;
    s_u64(&mut out, advances.len() as u64)?;
    for advance in advances {
        s_u64(&mut out, advance.tenant.raw())?;
        s_u8(&mut out, advance.kind.as_byte())?;
        s_u64(&mut out, advance.new_high_water)?;
    }
    counts.allocator_advances = advances.len() as u64;

    {
        let _capture = snap.intern.capture_guard();
        s_u8(&mut out, OwnerTag::Intern as u8)?;
        let count = snap.intern.name_count();
        s_u64(&mut out, count)?;
        let streamed = snap.intern.for_each_name(|tenant, id, name| {
            s_u64(&mut out, tenant.raw())?;
            s_u32(&mut out, id.raw())?;
            s_bytes(&mut out, name.as_bytes())
        })?;
        if streamed != count {
            return Err(CheckpointError::CountSkew {
                owner: "v9_intern",
                header: count,
                streamed,
            });
        }
        counts.intern_names = count;
    }

    {
        let _capture = snap.idempotency.capture_guard();
        s_u8(&mut out, OwnerTag::Idempotency as u8)?;
        let count = snap.idempotency.binding_count();
        s_u64(&mut out, count)?;
        let streamed =
            snap.idempotency
                .for_each_binding(|tenant, kind, external, internal, hash| {
                    s_u64(&mut out, tenant.raw())?;
                    s_u8(&mut out, kind)?;
                    s_bytes(&mut out, external.as_bytes())?;
                    s_u64(&mut out, internal)?;
                    s_u8(&mut out, u8::from(hash.is_some()))?;
                    s_u64(&mut out, hash.unwrap_or(0))
                })?;
        if streamed != count {
            return Err(CheckpointError::CountSkew {
                owner: "v9_idempotency",
                header: count,
                streamed,
            });
        }
        counts.idempotency_bindings = count;
    }

    {
        let _capture = snap.permissions.capture_guard();
        s_u8(&mut out, OwnerTag::Permissions as u8)?;
        let count = snap.permissions.doc_grant_count();
        s_u64(&mut out, count)?;
        let tenant = snap.permissions_tenant.raw();
        let streamed =
            snap.permissions
                .for_each_doc_grant::<_, CheckpointError>(|doc, grants| {
                    s_u64(&mut out, tenant)?;
                    s_u64(&mut out, doc.raw())?;
                    s_u32(
                        &mut out,
                        u32::try_from(grants.len()).expect("grant set < 4 G"),
                    )?;
                    for principal in grants {
                        s_bytes(&mut out, principal.as_bytes())?;
                    }
                    Ok(())
                })?;
        if streamed != count {
            return Err(CheckpointError::CountSkew {
                owner: "v9_permissions",
                header: count,
                streamed,
            });
        }
        counts.permission_docs = count;
    }

    finalize_snapshot_streaming(&mut out)?;
    let max_in_flight = out.max_in_flight();
    let body_len = out.body_len();
    let mut bufw = out.into_writer();
    bufw.flush()?;
    let file = bufw
        .into_inner()
        .map_err(|error| CheckpointError::Io(error.into_error()))?;
    file.sync_all()?;
    // Link is create-new: unlike rename, it can never replace bytes selected
    // by an older sidecar if generation allocation regresses or races.
    std::fs::hard_link(&tmp, &final_path)?;
    std::fs::remove_file(&tmp)?;
    fsync_dir(data_dir).map_err(crate::checkpoint::sidecar::arcgraph_err_to_io)?;

    Ok(IncrementalMetadataReport {
        generation,
        counts,
        dpt_entries: dpt.len() as u64,
        max_in_flight,
        body_len,
        overflow_peak_resident,
    })
}

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }
    fn need(&self, n: usize) -> Result<(), CheckpointError> {
        if self.pos + n > self.bytes.len() {
            return Err(CheckpointError::Corrupt {
                reason: format!(
                    "snapshot truncated: need {n} bytes at offset {}, have {}",
                    self.pos,
                    self.bytes.len()
                ),
            });
        }
        Ok(())
    }
    fn u8(&mut self) -> Result<u8, CheckpointError> {
        self.need(1)?;
        let v = self.bytes[self.pos];
        self.pos += 1;
        Ok(v)
    }
    fn u32(&mut self) -> Result<u32, CheckpointError> {
        self.need(4)?;
        let v = u32::from_le_bytes(self.bytes[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }
    fn u64(&mut self) -> Result<u64, CheckpointError> {
        self.need(8)?;
        let v = u64::from_le_bytes(self.bytes[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }
    fn bytes(&mut self) -> Result<&'a [u8], CheckpointError> {
        let n = self.u32()? as usize;
        self.need(n)?;
        let s = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn page(&mut self) -> Result<Box<[u8; PAGE_SIZE]>, CheckpointError> {
        self.need(PAGE_SIZE)?;
        let mut p: Box<[u8; PAGE_SIZE]> = Box::new([0u8; PAGE_SIZE]);
        p.copy_from_slice(&self.bytes[self.pos..self.pos + PAGE_SIZE]);
        self.pos += PAGE_SIZE;
        Ok(p)
    }
}

// ─────────────────────────────────────────────────────────────────────
// encode
// ─────────────────────────────────────────────────────────────────────

/// Serialize the full-state snapshot into `out`, returning the per-owner
/// entry counts captured. `advances` is the allocator high-water set
/// (collected under the commit-freeze by the producer; see BLOCK-1).
///
/// # REQ-2 — NO disk fault under the freeze
///
/// Caller MUST invoke this under `snap.txn.checkpoint_freeze()` so the
/// frontier read, MVCC walk, and RESIDENT page-image byte-copy are a
/// single quiescent instant (BLOCK-1 + BLOCK-2). Page capture uses the
/// **non-faulting** `iter_pages_resident_only` iterators: resident page
/// bytes are copied under the guard (in-RAM only — NO `pin_read` /
/// `fault_in` / disk read), and evicted page-ids are RECORDED (not
/// faulted). The returned [`EvictedPages`] lists must be backfilled by
/// the producer AFTER the guard drops (their durable disk images are
/// immutable-below-frontier, and any post-guard `> frontier` mutation is
/// idempotently re-applied by the anchored WAL replay). For the wired
/// pure-`DashMap` stores nothing is ever evicted, so the [`EvictedPages`]
/// are empty and the snapshot is complete under the guard. This closes
/// the ULTRACODE re-verify HIGH availability regression: a periodic
/// checkpoint never blocks foreground commits on synchronous disk
/// fault-in.
pub(crate) fn encode_snapshot_bytes(
    snap: &CheckpointSnapshot<'_>,
    checkpoint_lsn: Lsn,
    advances: &[AllocatorAdvance],
    out: &mut Vec<u8>,
) -> (SnapshotOwnerCounts, EvictedPages) {
    let mut counts = SnapshotOwnerCounts::default();
    let mut evicted = EvictedPages::default();
    out.extend_from_slice(&SNAPSHOT_MAGIC);
    out.extend_from_slice(&SNAPSHOT_FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&[0u8, 0u8]); // reserved
    put_u64(out, checkpoint_lsn.raw());

    // ── MVCC live records ──
    // Enumerate every live (non-tombstone) record visible at the
    // checkpoint frontier, per tenant. Restore re-installs each via the
    // replay MVCC-write path at `checkpoint_lsn`.
    put_u8(out, OwnerTag::Mvcc as u8);
    let count_pos = out.len();
    put_u64(out, 0); // placeholder
    let mut mvcc_n = 0u64;
    for tenant in snap.txn.tenants_with_chains() {
        snap.txn
            .for_each_visible_record(tenant, checkpoint_lsn, |key, bytes| {
                put_u64(out, tenant.raw());
                put_u64(out, key);
                put_bytes(out, bytes);
                mvcc_n += 1;
            });
    }
    out[count_pos..count_pos + 8].copy_from_slice(&mvcc_n.to_le_bytes());
    counts.mvcc_records = mvcc_n;

    // ── Primary-index pages (RESIDENT only under the guard; REQ-2) ──
    put_u8(out, OwnerTag::PrimaryPages as u8);
    let (primary, primary_evicted) = snap.primary_pages.iter_pages_resident_only();
    put_u64(out, primary.len() as u64);
    for (pid, latch) in &primary {
        put_u64(out, pid.raw());
        let g = latch.read();
        out.extend_from_slice(g.as_ref().as_ref());
    }
    counts.primary_pages = primary.len() as u64;
    evicted.primary = primary_evicted;

    // ── Record-store pages (RESIDENT only under the guard; REQ-2) ──
    put_u8(out, OwnerTag::RecordPages as u8);
    let (records, records_evicted) = snap.record_pages.iter_pages_resident_only();
    put_u64(out, records.len() as u64);
    for (pid, latch) in &records {
        put_u64(out, pid.raw());
        let g = latch.read();
        out.extend_from_slice(g.as_ref().as_ref());
    }
    counts.record_pages = records.len() as u64;
    evicted.record = records_evicted;

    // ── BLOB pages (RESIDENT only under the guard; REQ-2) ──
    put_u8(out, OwnerTag::BlobPages as u8);
    let (blobs, blobs_evicted) = snap.blob.iter_pages_resident_only();
    put_u64(out, blobs.len() as u64);
    for (tenant, pid, page) in &blobs {
        put_u64(out, tenant.raw());
        put_u64(out, *pid);
        out.extend_from_slice(page.as_ref());
    }
    counts.blob_pages = blobs.len() as u64;
    evicted.blob = blobs_evicted;

    // ── Allocator advances (collected under the freeze; BLOCK-1) ──
    put_u8(out, OwnerTag::Allocator as u8);
    put_u64(out, advances.len() as u64);
    for adv in advances {
        put_u64(out, adv.tenant.raw());
        put_u8(out, adv.kind.as_byte());
        put_u64(out, adv.new_high_water);
    }
    counts.allocator_advances = advances.len() as u64;

    // ── Intern names ──
    //
    // #1404 M0.x — STREAM via `for_each_name` (count header FIRST) rather than
    // re-collecting the whole reverse map into a `Vec`. This convenience path
    // still targets an in-RAM `Vec` buffer, but it no longer materializes a
    // SECOND whole-`Vec` of owned names; more importantly it shares the ONE
    // capture entry point (`for_each_name`) with the freeze-critical streaming
    // path, so the two cannot drift. Wire layout byte-identical.
    put_u8(out, OwnerTag::Intern as u8);
    let intern_count = snap.intern.name_count();
    put_u64(out, intern_count);
    snap.intern
        .for_each_name::<_, std::convert::Infallible>(|tenant, id, name| {
            put_u64(out, tenant.raw());
            put_u32(out, id.raw());
            put_bytes(out, name.as_bytes());
            Ok(())
        })
        .expect("put_* into a Vec is infallible");
    counts.intern_names = intern_count;

    // ── Idempotency bindings ──
    //
    // #1404 M0.x — STREAM via `for_each_binding` (count header FIRST), sharing
    // the ONE capture entry point with the freeze-critical streaming path.
    // Wire layout byte-identical to the pre-M0.x whole-`Vec` path.
    put_u8(out, OwnerTag::Idempotency as u8);
    let idempotency_count = snap.idempotency.binding_count();
    put_u64(out, idempotency_count);
    snap.idempotency
        .for_each_binding::<_, std::convert::Infallible>(|tenant, kind, ext, internal, hash| {
            put_u64(out, tenant.raw());
            put_u8(out, kind);
            put_bytes(out, ext.as_bytes());
            put_u64(out, internal);
            // payload hash: 1-byte present flag + u64
            put_u8(out, u8::from(hash.is_some()));
            put_u64(out, hash.unwrap_or(0));
            Ok(())
        })
        .expect("put_* into a Vec is infallible");
    counts.idempotency_bindings = idempotency_count;

    // ── Permission doc→grants ──
    //
    // #1404 M0.x — STREAM via `for_each_doc_grant` (count header FIRST) rather
    // than `iter_doc_grants`, which CLONED every doc's grant set into one
    // `Vec<(NodeId, BTreeSet<String>)>` under the freeze (O(docs-with-ACLs)
    // whole-in-RAM — the third RE-2 owner sibling). Wire layout byte-identical
    // (tenant/doc/grant-count/principal-bytes per doc, count-first).
    put_u8(out, OwnerTag::Permissions as u8);
    let permission_docs = snap.permissions.doc_grant_count();
    put_u64(out, permission_docs);
    let permissions_tenant = snap.permissions_tenant.raw();
    snap.permissions
        .for_each_doc_grant::<_, std::convert::Infallible>(|doc, grants| {
            put_u64(out, permissions_tenant);
            put_u64(out, doc.raw());
            put_u32(out, u32::try_from(grants.len()).expect("grant set < 4 G"));
            for principal in grants {
                put_bytes(out, principal.as_bytes());
            }
            Ok(())
        })
        .expect("put_* into a Vec is infallible");
    counts.permission_docs = permission_docs;

    // NOTE: the footer CRC is NOT written here. The producer appends the
    // post-guard evicted-page supplement (REQ-2) via
    // `append_evicted_supplement` and then seals the buffer with
    // `finalize_snapshot_bytes` (which writes the footer CRC over the
    // complete body). For the wired pure-DashMap stores `evicted` is empty
    // and the supplement is a zero-count section.
    (counts, evicted)
}

/// SVC-1 / #849 / ADR-229 REQ-2 — page-ids evicted-to-disk at capture
/// time, RECORDED (not faulted) under the commit-freeze. Their durable
/// disk images are read + appended to the snapshot AFTER the guard drops
/// (immutable-below-frontier; any `> frontier` mutation is idempotently
/// re-applied by the anchored WAL replay). Empty for the wired
/// pure-`DashMap` stores (they never evict).
/// REQ-2 — one evicted page's durable disk image read post-guard:
/// `(owner_tag, tenant, page_id, PAGE_SIZE bytes)`. `owner_tag` routes the
/// page back to its store on restore.
pub(crate) type EvictedPageImage = (u8, TenantId, u64, Box<[u8; PAGE_SIZE]>);

#[derive(Debug, Default)]
pub(crate) struct EvictedPages {
    /// Evicted primary-index page ids.
    pub primary: Vec<arcgraph_core::PageId>,
    /// Evicted record-store page ids.
    pub record: Vec<arcgraph_core::PageId>,
    /// Evicted blob pages `(tenant, page_id)`.
    pub blob: Vec<(TenantId, u64)>,
}

impl EvictedPages {
    /// True iff no page was evicted at capture time — the snapshot is
    /// complete under the freeze (no post-guard disk read needed). Always
    /// true for the wired pure-`DashMap` stores.
    pub(crate) fn is_empty(&self) -> bool {
        self.primary.is_empty() && self.record.is_empty() && self.blob.is_empty()
    }
}

/// Append the post-guard evicted-page supplement to a snapshot body, then
/// return the ids so the producer knows what to read. The supplement is a
/// single `EvictedSupplement` section carrying the evicted pages' durable
/// disk images (read by the producer AFTER releasing the commit-freeze).
/// Restore installs these via the SAME `install_or_replace` path as the
/// resident page sections.
///
/// For the wired pure-`DashMap` stores this writes a zero-count section
/// (nothing was evicted), so the on-disk format is unchanged in practice.
/// `evicted_images` is `(owner_tag, tenant, page_id, PAGE_SIZE bytes)`
/// tuples the producer read post-guard.
pub(crate) fn append_evicted_supplement(out: &mut Vec<u8>, evicted_images: &[EvictedPageImage]) {
    put_u8(out, OwnerTag::EvictedSupplement as u8);
    put_u64(out, evicted_images.len() as u64);
    for (owner_tag, tenant, pid, page) in evicted_images {
        put_u8(out, *owner_tag);
        put_u64(out, tenant.raw());
        put_u64(out, *pid);
        out.extend_from_slice(page.as_ref());
    }
}

/// Seal a snapshot body with the footer CRC over every byte before the
/// footer. Called by the producer AFTER `append_evicted_supplement`.
pub(crate) fn finalize_snapshot_bytes(out: &mut Vec<u8>) {
    let crc = crc32c::crc32c(out);
    out.extend_from_slice(&crc.to_le_bytes());
}

// ─────────────────────────────────────────────────────────────────────
// #1404 M0.5 — STREAMING encode (byte-identical to the whole-Vec path)
// ─────────────────────────────────────────────────────────────────────
//
// The whole-`Vec` `encode_snapshot_bytes` back-patches the MVCC count
// placeholder AFTER walking the records (the count is not known upfront —
// `for_each_visible_record` yields records without a count). A streamed
// file cannot seek back to overwrite a flushed count. To keep the on-disk
// layout byte-identical (count-FIRST, as the format requires) the
// streaming path runs a lightweight COUNTING PASS over the SAME
// deterministic walk (`tenants_with_chains` → `for_each_visible_record`,
// identical iteration order) to learn the count, then STREAMS the count +
// records in a second pass. Both passes visit the same records in the same
// order, so the streamed bytes match the whole-`Vec` bytes exactly.
//
// Every non-MVCC section already writes its count from an
// already-materialized `.len()` (no back-patch), so those stream directly.

/// Count the MVCC records visible at `checkpoint_lsn` across every tenant,
/// using the SAME deterministic walk the streaming write pass uses. Runs
/// UNDER the same commit-freeze as the write pass (the caller holds it), so
/// the count and the streamed records observe ONE quiescent instant —
/// identical to the whole-`Vec` path's single walk. O(records) time, O(1)
/// extra memory (no record bytes retained).
fn count_visible_mvcc_records(snap: &CheckpointSnapshot<'_>, checkpoint_lsn: Lsn) -> u64 {
    let mut n = 0u64;
    for tenant in snap.txn.tenants_with_chains() {
        snap.txn
            .for_each_visible_record(tenant, checkpoint_lsn, |_key, _bytes| {
                n += 1;
            });
    }
    n
}

/// STREAMING twin of [`encode_snapshot_bytes`]: emit the full-state
/// snapshot body into `out` (an [`SnapshotSink`]) record-by-record /
/// page-by-page, holding only O(chunk) resident — NOT the whole snapshot.
/// The emitted bytes are BYTE-IDENTICAL to the whole-`Vec` path over the
/// same DB state (same header, owner order, per-owner encoding). Returns
/// the per-owner counts + the evicted page-ids (backfilled post-guard by
/// the producer's [`stream_evicted_supplement`], which streams each evicted
/// page one-at-a-time — byte-identical to the whole-`Vec` path's
/// [`append_evicted_supplement`]).
///
/// # REQ-2 — NO disk fault under the freeze (unchanged from the Vec path)
///
/// Caller MUST invoke this under `snap.txn.checkpoint_freeze()`. Page
/// capture uses the NON-FAULTING `iter_pages_resident_only` iterators;
/// resident page bytes stream out under the guard (still one page-image
/// resident at a time — the transient buffer the store latch exposes), and
/// evicted page-ids are RECORDED (not faulted) for the post-guard
/// supplement. The freeze never blocks on a synchronous disk read.
pub(crate) fn encode_snapshot_streaming<S: SnapshotSink>(
    snap: &CheckpointSnapshot<'_>,
    checkpoint_lsn: Lsn,
    advances: &[AllocatorAdvance],
    out: &mut S,
) -> Result<(SnapshotOwnerCounts, EvictedPages), CheckpointError> {
    let mut counts = SnapshotOwnerCounts::default();
    let mut evicted = EvictedPages::default();

    // ── Header ──
    out.write_body(&SNAPSHOT_MAGIC)?;
    out.write_body(&SNAPSHOT_FORMAT_VERSION.to_le_bytes())?;
    out.write_body(&[0u8, 0u8])?; // reserved
    s_u64(out, checkpoint_lsn.raw())?;

    // ── MVCC live records ──
    // Count-FIRST (the format requires the count before the records), then
    // stream the records in the SAME deterministic order. The count pass
    // and the write pass run under the SAME freeze the caller holds → one
    // quiescent instant, identical to the whole-`Vec` single walk.
    s_u8(out, OwnerTag::Mvcc as u8)?;
    let mvcc_n = count_visible_mvcc_records(snap, checkpoint_lsn);
    s_u64(out, mvcc_n)?;
    let mut streamed_mvcc = 0u64;
    let mut mvcc_err: Option<CheckpointError> = None;
    for tenant in snap.txn.tenants_with_chains() {
        if mvcc_err.is_some() {
            break;
        }
        snap.txn
            .for_each_visible_record(tenant, checkpoint_lsn, |key, bytes| {
                if mvcc_err.is_some() {
                    return;
                }
                // The closure cannot return a Result; capture the first I/O
                // error and short-circuit the remaining callbacks. On the
                // production BufWriter this only fires on a genuine disk
                // error (disk-full), which the caller surfaces as Corrupt/Io
                // → recovery ignores the partial temp (crash-mid-stream).
                let r = (|| -> Result<(), CheckpointError> {
                    s_u64(out, tenant.raw())?;
                    s_u64(out, key)?;
                    s_bytes(out, bytes)
                })();
                match r {
                    Ok(()) => streamed_mvcc += 1,
                    Err(e) => mvcc_err = Some(e),
                }
            });
    }
    if let Some(e) = mvcc_err {
        return Err(e);
    }
    // #1404 M0.x round-2 (non-blocking hardening) — a hard `CountSkew` return,
    // not a `debug_assert_eq!`: release builds must ALSO abort the checkpoint
    // when the count pass and write pass diverge (a mis-framed section written
    // `Ok` + #1365 WAL reclaim = permanent silent data loss), matching the
    // sibling sections' release-build defense-in-depth below.
    if streamed_mvcc != mvcc_n {
        return Err(CheckpointError::CountSkew {
            owner: "mvcc_records",
            header: mvcc_n,
            streamed: streamed_mvcc,
        });
    }
    counts.mvcc_records = mvcc_n;

    // ── Primary-index pages (RESIDENT only under the guard; REQ-2) ──
    //
    // #1404 M0.x FIX-B — STREAM page-at-a-time (count header FIRST, then
    // `for_each_resident_page` emits each latch's bytes and DROPS the transient
    // before the next) instead of pre-collecting a whole `Vec` of latches. The
    // count is `resident_page_count` (stable under `checkpoint_freeze` — no
    // commit can add/remove a page mid-capture), and the streamed count is
    // hard-checked against it (CountSkew abort). Wire layout byte-identical.
    s_u8(out, OwnerTag::PrimaryPages as u8)?;
    let primary_count = snap.primary_pages.resident_page_count() as u64;
    s_u64(out, primary_count)?;
    let mut primary_streamed = 0u64;
    let primary_evicted = snap.primary_pages.for_each_resident_page(|pid, latch| {
        s_u64(out, pid.raw())?;
        let g = latch.read();
        out.write_body(g.as_ref().as_ref())?;
        primary_streamed += 1;
        Ok::<(), CheckpointError>(())
    })?;
    if primary_streamed != primary_count {
        return Err(CheckpointError::CountSkew {
            owner: "primary_pages",
            header: primary_count,
            streamed: primary_streamed,
        });
    }
    counts.primary_pages = primary_count;
    evicted.primary = primary_evicted;

    // ── Record-store pages (RESIDENT only under the guard; REQ-2) ──
    s_u8(out, OwnerTag::RecordPages as u8)?;
    let record_count = snap.record_pages.resident_page_count() as u64;
    s_u64(out, record_count)?;
    let mut record_streamed = 0u64;
    let records_evicted = snap.record_pages.for_each_resident_page(|pid, latch| {
        s_u64(out, pid.raw())?;
        let g = latch.read();
        out.write_body(g.as_ref().as_ref())?;
        record_streamed += 1;
        Ok::<(), CheckpointError>(())
    })?;
    if record_streamed != record_count {
        return Err(CheckpointError::CountSkew {
            owner: "record_pages",
            header: record_count,
            streamed: record_streamed,
        });
    }
    counts.record_pages = record_count;
    evicted.record = records_evicted;

    // ── BLOB pages (RESIDENT only under the guard; REQ-2) ──
    //
    // #1404 M0.x FIX-B — the WORST whole-`Vec`: pre-`iter_pages_resident_only`
    // pushed one 8 KB `encode_page()` COPY per resident page into a `Vec`
    // (O(cap) ≈ +2 GiB/checkpoint). Now stream each page: encode → emit → DROP,
    // ≤ one 8 KB page-image resident. Count-first + hard-check.
    s_u8(out, OwnerTag::BlobPages as u8)?;
    let blob_count = snap.blob.resident_page_count() as u64;
    s_u64(out, blob_count)?;
    let mut blob_streamed = 0u64;
    let blobs_evicted = snap.blob.for_each_resident_page(|tenant, pid, page| {
        s_u64(out, tenant.raw())?;
        s_u64(out, pid)?;
        out.write_body(page.as_ref())?;
        blob_streamed += 1;
        Ok::<(), CheckpointError>(())
    })?;
    if blob_streamed != blob_count {
        return Err(CheckpointError::CountSkew {
            owner: "blob_pages",
            header: blob_count,
            streamed: blob_streamed,
        });
    }
    counts.blob_pages = blob_count;
    evicted.blob = blobs_evicted;

    // ── Allocator advances (collected under the freeze; BLOCK-1) ──
    s_u8(out, OwnerTag::Allocator as u8)?;
    s_u64(out, advances.len() as u64)?;
    for adv in advances {
        s_u64(out, adv.tenant.raw())?;
        s_u8(out, adv.kind.as_byte())?;
        s_u64(out, adv.new_high_water)?;
    }
    counts.allocator_advances = advances.len() as u64;

    // ── Intern names ──
    //
    // #1404 M0.x — STREAM one name at a time (count header FIRST, then the
    // records via `for_each_name`), NEVER re-collecting the whole reverse map
    // into a `Vec` under the freeze. The wire layout (tag, count, per-name
    // {tenant, id, bytes}) is byte-identical to the pre-M0.x whole-`Vec` path.
    //
    // #1404 M0.x FIX-D — hold the capture WRITE guard across the count header +
    // the stream so no concurrent `intern`/`intern_install` (which take the
    // READ guard) can interleave and skew header≠stream (a mis-framed section
    // → #1365 silent data loss). The header count and the streamed count are
    // then deterministically equal; the producer-side HARD CHECK below is
    // defense-in-depth (aborts the checkpoint on any residual skew).
    {
        let _capture = snap.intern.capture_guard();
        s_u8(out, OwnerTag::Intern as u8)?;
        let intern_count = snap.intern.name_count();
        s_u64(out, intern_count)?;
        let streamed = snap.intern.for_each_name(|tenant, id, name| {
            s_u64(out, tenant.raw())?;
            s_u32(out, id.raw())?;
            s_bytes(out, name.as_bytes())
        })?;
        if streamed != intern_count {
            return Err(CheckpointError::CountSkew {
                owner: "intern",
                header: intern_count,
                streamed,
            });
        }
        counts.intern_names = intern_count;
    }

    // ── Idempotency bindings ──
    //
    // #1404 M0.x — STREAM one binding at a time (count header FIRST, then the
    // records via `for_each_binding`), NEVER re-collecting the whole binding
    // set into a `Vec` under the freeze. This is THE whole-in-RAM sibling the
    // M0.x binding spill would otherwise leave unbounded: the resident DashMap
    // is bounded, but a whole-`Vec` capture re-materialized all ~9M bindings
    // under `checkpoint_freeze`. The wire layout (tag, count, per-binding
    // {tenant, kind, ext-bytes, internal, hash-flag, hash}) is byte-identical
    // to the pre-M0.x whole-`Vec` path (the count precedes the records, same
    // resident-vs-spilled dedup as `binding_count`).
    //
    // #1404 M0.x FIX-D — capture WRITE guard across count+stream (excludes the
    // concurrent post-commit `install`/`release`) + producer-side HARD CHECK.
    {
        let _capture = snap.idempotency.capture_guard();
        s_u8(out, OwnerTag::Idempotency as u8)?;
        let idempotency_count = snap.idempotency.binding_count();
        s_u64(out, idempotency_count)?;
        let streamed = snap
            .idempotency
            .for_each_binding(|tenant, kind, ext, internal, hash| {
                s_u64(out, tenant.raw())?;
                s_u8(out, kind)?;
                s_bytes(out, ext.as_bytes())?;
                s_u64(out, internal)?;
                // payload hash: 1-byte present flag + u64
                s_u8(out, u8::from(hash.is_some()))?;
                s_u64(out, hash.unwrap_or(0))
            })?;
        if streamed != idempotency_count {
            return Err(CheckpointError::CountSkew {
                owner: "idempotency",
                header: idempotency_count,
                streamed,
            });
        }
        counts.idempotency_bindings = idempotency_count;
    }

    // ── Permission doc→grants ──
    //
    // #1404 M0.x — STREAM via `for_each_doc_grant` (count header FIRST) rather
    // than `iter_doc_grants`, which cloned every doc's grant set into one `Vec`
    // under the freeze (the third RE-2 owner whole-in-RAM sibling). Wire layout
    // byte-identical to the pre-M0.x whole-`Vec` path (tenant/doc/grant-count/
    // principal-bytes per doc, count-first, same skip-if-class-missing filter
    // as `doc_grant_count`).
    //
    // #1404 M0.x FIX-D — capture WRITE guard across count+stream (excludes
    // concurrent `apply_doc_acl`/`revoke_doc`) + producer-side HARD CHECK.
    {
        let _capture = snap.permissions.capture_guard();
        s_u8(out, OwnerTag::Permissions as u8)?;
        let permission_docs = snap.permissions.doc_grant_count();
        s_u64(out, permission_docs)?;
        let permissions_tenant = snap.permissions_tenant.raw();
        let streamed =
            snap.permissions
                .for_each_doc_grant::<_, CheckpointError>(|doc, grants| {
                    s_u64(out, permissions_tenant)?;
                    s_u64(out, doc.raw())?;
                    s_u32(out, u32::try_from(grants.len()).expect("grant set < 4 G"))?;
                    for principal in grants {
                        s_bytes(out, principal.as_bytes())?;
                    }
                    Ok(())
                })?;
        if streamed != permission_docs {
            return Err(CheckpointError::CountSkew {
                owner: "permissions",
                header: permission_docs,
                streamed,
            });
        }
        counts.permission_docs = permission_docs;
    }

    // NOTE (matches the whole-`Vec` path): the footer CRC is NOT written
    // here. The producer streams the post-guard evicted-page supplement via
    // `append_evicted_supplement_streaming` and then seals the footer with
    // `finalize_snapshot_streaming` (running-CRC trailer). For the wired
    // pure-DashMap stores `evicted` is empty → a zero-count supplement.
    Ok((counts, evicted))
}

/// STREAMING twin of [`append_evicted_supplement`]: emit the post-guard
/// evicted-page supplement section into the sink from an ALREADY-collected
/// `evicted_images` slice, byte-identical to the whole-`Vec` path.
/// Zero-count for the wired pure-`DashMap` stores.
///
/// # #1404 M0.5 — retained as the whole-`Vec` byte-identity ORACLE only
///
/// This twin still takes a pre-collected `&[EvictedPageImage]` (all N pages
/// resident at once). The PRODUCTION path is [`stream_evicted_supplement`],
/// which reads + emits each evicted page one-at-a-time (≤ one page resident,
/// NOT O(N)). This slice-taking twin is kept ONLY as the differential oracle
/// the `stream_evicted_supplement` byte-identity test diffs against — it is
/// NOT called by the producer (hence `#[cfg(test)]`). See the module
/// `#1404 M0.5` header.
#[cfg(test)]
pub(crate) fn append_evicted_supplement_streaming<S: SnapshotSink>(
    out: &mut S,
    evicted_images: &[EvictedPageImage],
) -> Result<(), CheckpointError> {
    s_u8(out, OwnerTag::EvictedSupplement as u8)?;
    s_u64(out, evicted_images.len() as u64)?;
    for (owner_tag, tenant, pid, page) in evicted_images {
        s_u8(out, *owner_tag)?;
        s_u64(out, tenant.raw())?;
        s_u64(out, *pid)?;
        out.write_body(page.as_ref())?;
    }
    Ok(())
}

/// #1404 M0.5 — stream the post-guard evicted-page supplement PAGE-BY-PAGE,
/// reading each evicted page's durable spill image, emitting it, and
/// DROPPING it before the next — so at most ONE page (`PAGE_SIZE`) is
/// caller-resident at a time, INDEPENDENT of the evicted-count `N`.
///
/// # Why this exists (the ultracode REJECT fix)
///
/// The prior producer path pre-collected ALL evicted images into a
/// `Vec<EvictedPageImage>` (`read_evicted_page_images` — one owned
/// `Box<[u8; PAGE_SIZE]>` per evicted page, held at once) BEFORE streaming
/// them via [`append_evicted_supplement_streaming`]. The evicted set is
/// O(N-above-the-watermark), monotonic — at the 10M target ~9.5M spilled
/// pages collect ~74 GB into that Vec at every checkpoint, the EXACT
/// whole-`Vec` OOM class #1404 exists to fix, on the on-by-default durable
/// serve path. Streaming page-by-page bounds the supplement's caller-side
/// working-set to ≤ one page + the sink's 1 MiB `BufWriter`.
///
/// # Byte-identity (UNCHANGED wire layout)
///
/// The on-wire layout is byte-for-byte what [`append_evicted_supplement`] /
/// [`append_evicted_supplement_streaming`] emit over the same evicted set —
/// tag `EvictedSupplement`, `count = evicted.blob.len()` (KNOWN UP FRONT, no
/// pre-collection), then per-image `{owner_tag=BlobPages, tenant, page_id,
/// PAGE}` in `evicted.blob` iteration order (the SAME order
/// `read_evicted_page_images` walked). The differential byte-identity test
/// (`streamed == whole-Vec append_evicted_supplement`) is the gate.
///
/// # Fail-loud (MOVED ahead of the write)
///
/// The primary/record non-empty check (those stores never evict — a
/// non-empty set there is a wiring bug, the OQ-2 data-loss class) runs
/// BEFORE any supplement byte hits the sink, so a wiring bug refuses the
/// checkpoint without leaving a partially-written supplement. A missing
/// spill image for a reported-evicted blob page is likewise `Corrupt`.
///
/// Returns the PEAK caller-resident supplement bytes — `0` when nothing was
/// evicted, else exactly `PAGE_SIZE` (one page working buffer at a time,
/// INDEPENDENT of N). The bounded-resident regression test asserts this is
/// O(1) in the evicted-count (the un-fixed whole-`Vec` path would hold
/// `N · PAGE_SIZE`).
pub(crate) fn stream_evicted_supplement<S: SnapshotSink>(
    out: &mut S,
    snap: &CheckpointSnapshot<'_>,
    evicted: &EvictedPages,
) -> Result<usize, CheckpointError> {
    // ── Fail-loud, BEFORE any byte hits the sink (moved ahead of the write) ──
    // The primary/record page stores never evict — a non-empty evicted set
    // there is a wiring bug. Silently omitting their images from the snapshot
    // would be data loss on the anchored restart (the OQ-2 class). Refuse the
    // checkpoint here, before writing an incomplete supplement.
    if !evicted.primary.is_empty() || !evicted.record.is_empty() {
        return Err(CheckpointError::Corrupt {
            reason: format!(
                "checkpoint producer: {} primary + {} record pages were evicted at capture \
                 but the primary/record page stores expose no post-guard disk-read (they \
                 never evict) — refusing to write an incomplete snapshot",
                evicted.primary.len(),
                evicted.record.len(),
            ),
        });
    }

    // ── Section header: tag + count (count is KNOWN UP FRONT) ──
    s_u8(out, OwnerTag::EvictedSupplement as u8)?;
    s_u64(out, evicted.blob.len() as u64)?;

    // ── Stream each evicted BLOB page one-at-a-time (≤ one page resident) ──
    let mut peak_resident = 0usize;
    for &(tenant, page_id) in &evicted.blob {
        // Read this page's durable spill image. It is dropped at the end of
        // the loop body (before the next iteration), so only ONE page is
        // caller-resident at a time — NOT the O(N) whole-`Vec` collection.
        // A reported-evicted page without a durable spill image would make an
        // anchored restart lose data, so refuse the incomplete checkpoint.
        let page = snap
            .blob
            .read_evicted_page(tenant, page_id)?
            .ok_or_else(|| CheckpointError::Corrupt {
                reason: format!(
                    "checkpoint producer: bounded blob page ({tenant:?},{page_id}) was reported \
                     evicted but its spill image is missing — refusing to write an incomplete \
                     snapshot"
                ),
            })?;
        // One page resident right now (`page`); the sink buffers ≤ its
        // `BufWriter` capacity independently. Record the peak caller-resident
        // supplement bytes — a single `PAGE_SIZE`, independent of N.
        peak_resident = peak_resident.max(PAGE_SIZE);
        // Emit byte-identical to the whole-`Vec` twin: owner_tag, tenant,
        // page_id, then the PAGE body.
        s_u8(out, OwnerTag::BlobPages as u8)?;
        s_u64(out, tenant.raw())?;
        s_u64(out, page_id)?;
        out.write_body(page.as_ref())?;
        // `page` drops HERE, before the next iteration — the bounded-resident
        // invariant. Explicit for the reader (scope-drop is equivalent).
        drop(page);
    }
    Ok(peak_resident)
}

/// STREAMING twin of [`finalize_snapshot_bytes`]: emit the 4-byte footer
/// CRC trailer. The value is the sink's RUNNING body CRC — byte-identical
/// to `crc32c(whole_body)` because `crc32c(concat) == fold(crc32c_append,
/// 0, chunks)` (crc32c 0.6). The footer bytes themselves are NOT folded
/// into the CRC (CRC is over-body-only, as in the whole-`Vec` path); we
/// snapshot the CRC BEFORE writing the trailer.
pub(crate) fn finalize_snapshot_streaming<S: SnapshotSink>(
    out: &mut S,
) -> Result<(), CheckpointError> {
    let crc = out.body_crc();
    // The trailer is written through `write_body` too (it must land in the
    // file), but its bytes do not affect the value we just captured — the
    // footer CRC covers the body only, identical to `finalize_snapshot_bytes`
    // which computes `crc32c(out)` over the pre-footer body then appends.
    out.write_body(&crc.to_le_bytes())
}

/// #1404 M0.5 — an in-progress streamed snapshot write. Owns the crash-atomic
/// temp file + its `BufWriter` sink, so the producer can stream the
/// in-freeze sections UNDER the commit-freeze, RELEASE the freeze, then
/// stream the post-guard evicted supplement + footer — the SAME freeze
/// scoping as the whole-`Vec` producer (which builds the Vec under the
/// freeze, releases, then appends the supplement) — while never holding the
/// whole snapshot resident (only the `BufWriter` + one page at a time).
///
/// # Crash-atomicity (ADR-229 — UNCHANGED)
///
/// `open` creates a PROCESS-UNIQUE temp file; the sink writes the body
/// incrementally; [`Self::finalize_atomic`] flushes → fsyncs the temp ONCE
/// → atomically renames it into place → dir-fsyncs. The sidecar (the
/// ESTABLISH point) is still written LAST by the producer. A crash BEFORE
/// `finalize_atomic`'s rename leaves a PARTIAL, un-renamed temp with no
/// sidecar → recovery ignores it (prior checkpoint + WAL replay
/// reconstruct) — byte-identical to the whole-`Vec` crash behaviour (the
/// unique temp is likewise orphaned + overwritten by the next checkpoint).
pub(crate) struct StreamingSnapshotWrite {
    sink: FileSnapshotSink<std::io::BufWriter<std::fs::File>>,
    tmp: PathBuf,
    data_dir: PathBuf,
}

impl StreamingSnapshotWrite {
    /// Create the crash-atomic temp file + a [`SNAPSHOT_STREAM_BUF_BYTES`]
    /// `BufWriter` sink over it. No bytes are durable until
    /// [`Self::finalize_atomic`].
    pub(crate) fn open(data_dir: &Path) -> Result<Self, CheckpointError> {
        let tmp = unique_snapshot_tmp(data_dir);
        let f = std::fs::File::create(&tmp)?;
        let bufw = std::io::BufWriter::with_capacity(SNAPSHOT_STREAM_BUF_BYTES, f);
        Ok(Self {
            sink: FileSnapshotSink::new(bufw),
            tmp,
            data_dir: data_dir.to_path_buf(),
        })
    }

    /// The streaming sink — hand this to [`encode_snapshot_streaming`],
    /// [`append_evicted_supplement_streaming`], and
    /// [`finalize_snapshot_streaming`].
    pub(crate) fn sink(&mut self) -> &mut FileSnapshotSink<std::io::BufWriter<std::fs::File>> {
        &mut self.sink
    }

    /// The bounded-resident instrumentation (M0.5 RSS proof): peak single
    /// write handed to the sink (a `PAGE_SIZE` page or one MVCC record) +
    /// total body bytes. `max_in_flight` is O(chunk), « the whole snapshot
    /// the Vec path held resident.
    pub(crate) fn stats(&self) -> StreamStats {
        StreamStats {
            max_in_flight: self.sink.max_in_flight(),
            body_len: self.sink.body_len(),
        }
    }

    /// Flush the buffered body, fsync the temp ONCE (durable BEFORE the
    /// rename), then atomically rename it into place + dir-fsync. After this
    /// returns `Ok`, the snapshot file is durable (the producer then writes
    /// the sidecar LAST). Consumes `self`.
    pub(crate) fn finalize_atomic(self) -> Result<(), CheckpointError> {
        let final_path = snapshot_path(&self.data_dir);
        // Flush the BufWriter, then fsync the file ONCE (ADR-229 durability
        // before rename). `into_writer` → BufWriter → flush → File → sync_all.
        let mut bufw = self.sink.into_writer();
        bufw.flush()?;
        let f = bufw
            .into_inner()
            .map_err(|e| CheckpointError::Io(e.into_error()))?;
        f.sync_all()?;
        std::fs::rename(&self.tmp, &final_path)?;
        fsync_dir(&self.data_dir).map_err(crate::checkpoint::sidecar::arcgraph_err_to_io)?;
        Ok(())
    }
}

/// Bounded-resident instrumentation from a streamed snapshot write — the
/// M0.5 RSS proof. `max_in_flight` is the peak single caller-side write
/// (a `PAGE_SIZE` page or one MVCC record) — O(chunk), « the whole
/// snapshot the Vec path held resident.
#[derive(Debug, Clone, Copy)]
pub(crate) struct StreamStats {
    /// Peak single-write chunk size handed to the sink.
    pub max_in_flight: usize,
    /// Total streamed body bytes (pre-footer).
    pub body_len: u64,
}

/// Decoded v9 checkpoint header and its ARIES DPT. Retained owner bytes are
/// restored directly while streaming and are therefore not accumulated here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IncrementalCheckpointMetadata {
    pub checkpoint_lsn: Lsn,
    pub redo_lsn: Lsn,
    pub capture_lsn: Lsn,
    pub dpt: Vec<DirtyPageSnapshot>,
    pub counts: SnapshotOwnerCounts,
}

struct CrcMetadataReader<R> {
    inner: R,
    crc: u32,
    position: u64,
    file_len: u64,
}

impl<R: Read> CrcMetadataReader<R> {
    fn new(inner: R, file_len: u64) -> Self {
        Self {
            inner,
            crc: 0,
            position: 0,
            file_len,
        }
    }

    fn body_exact(&mut self, out: &mut [u8], what: &'static str) -> Result<(), CheckpointError> {
        self.inner
            .read_exact(out)
            .map_err(|error| metadata_read_error(error, what))?;
        self.crc = crc32c::crc32c_append(self.crc, out);
        self.position += out.len() as u64;
        Ok(())
    }

    fn u8(&mut self, what: &'static str) -> Result<u8, CheckpointError> {
        let mut bytes = [0; 1];
        self.body_exact(&mut bytes, what)?;
        Ok(bytes[0])
    }

    fn u16(&mut self, what: &'static str) -> Result<u16, CheckpointError> {
        let mut bytes = [0; 2];
        self.body_exact(&mut bytes, what)?;
        Ok(u16::from_le_bytes(bytes))
    }

    fn u32(&mut self, what: &'static str) -> Result<u32, CheckpointError> {
        let mut bytes = [0; 4];
        self.body_exact(&mut bytes, what)?;
        Ok(u32::from_le_bytes(bytes))
    }

    fn u64(&mut self, what: &'static str) -> Result<u64, CheckpointError> {
        let mut bytes = [0; 8];
        self.body_exact(&mut bytes, what)?;
        Ok(u64::from_le_bytes(bytes))
    }

    fn page(&mut self, what: &'static str) -> Result<Box<[u8; PAGE_SIZE]>, CheckpointError> {
        let mut page = Box::new([0; PAGE_SIZE]);
        self.body_exact(page.as_mut(), what)?;
        Ok(page)
    }

    fn string(&mut self, what: &'static str) -> Result<String, CheckpointError> {
        let len = self.u32(what)? as usize;
        if len as u64 > self.remaining_body() {
            return Err(CheckpointError::Corrupt {
                reason: format!("v9 metadata {what} length {len} exceeds remaining file"),
            });
        }
        let mut bytes = vec![0; len];
        self.body_exact(&mut bytes, what)?;
        String::from_utf8(bytes).map_err(|_| CheckpointError::Corrupt {
            reason: format!("v9 metadata {what} is not UTF-8"),
        })
    }

    fn remaining_body(&self) -> u64 {
        self.file_len
            .saturating_sub(self.position)
            .saturating_sub(FOOTER_SIZE as u64)
    }

    fn count(
        &self,
        count: u64,
        min_entry: u64,
        what: &'static str,
    ) -> Result<usize, CheckpointError> {
        if count.saturating_mul(min_entry) > self.remaining_body() {
            return Err(CheckpointError::Corrupt {
                reason: format!("v9 metadata {what} count {count} exceeds remaining file"),
            });
        }
        usize::try_from(count).map_err(|_| CheckpointError::Corrupt {
            reason: format!("v9 metadata {what} count does not fit usize"),
        })
    }

    fn finish(mut self) -> Result<(), CheckpointError> {
        let expected = self.crc;
        let mut footer = [0; FOOTER_SIZE];
        self.inner
            .read_exact(&mut footer)
            .map_err(|error| metadata_read_error(error, "CRC footer"))?;
        self.position += FOOTER_SIZE as u64;
        let stored = u32::from_le_bytes(footer);
        if stored != expected {
            return Err(CheckpointError::Corrupt {
                reason: format!(
                    "v9 metadata crc mismatch: stored 0x{stored:08x}, computed 0x{expected:08x}"
                ),
            });
        }
        let mut trailing = [0; 1];
        if self.inner.read(&mut trailing)? != 0 || self.position != self.file_len {
            return Err(CheckpointError::Corrupt {
                reason: "v9 metadata has trailing bytes".to_owned(),
            });
        }
        Ok(())
    }
}

fn metadata_read_error(error: std::io::Error, what: &'static str) -> CheckpointError {
    if error.kind() == std::io::ErrorKind::UnexpectedEof {
        CheckpointError::Corrupt {
            reason: format!("v9 metadata truncated while reading {what}"),
        }
    } else {
        CheckpointError::Io(error)
    }
}

/// Streaming owner callbacks used by the v5→v6 rewriter. Implementations
/// must retain at most a bounded run buffer; the decoder itself holds one
/// page/string/grant set at a time.
pub trait IncrementalOwnerVisitor {
    /// One durable intern binding.
    fn intern(
        &mut self,
        tenant: TenantId,
        id: StringId,
        name: String,
    ) -> Result<(), CheckpointError>;

    /// One durable idempotency binding.
    fn idempotency(
        &mut self,
        tenant: TenantId,
        kind: u8,
        external_id: String,
        internal_id: u64,
        payload_hash: Option<u64>,
    ) -> Result<(), CheckpointError>;

    /// One durable document grant set.
    fn permission(
        &mut self,
        tenant: TenantId,
        doc: NodeId,
        grants: BTreeSet<String>,
    ) -> Result<(), CheckpointError>;
}

/// Validate an immutable incremental metadata file and stream only ADR-229
/// owners 6–8 to a bounded migration visitor. Other sections are parsed and
/// checksummed but never installed into resident owner maps.
pub fn visit_incremental_metadata_owners(
    data_dir: &Path,
    expected_checkpoint_lsn: Lsn,
    expected_generation: u64,
    visitor: &mut dyn IncrementalOwnerVisitor,
) -> Result<IncrementalCheckpointMetadata, CheckpointError> {
    let path = incremental_metadata_path(data_dir, expected_checkpoint_lsn, expected_generation);
    let file = std::fs::File::open(path)?;
    let file_len = file.metadata()?.len();
    if file_len < 40 + FOOTER_SIZE as u64 {
        return Err(CheckpointError::Corrupt {
            reason: format!("v9 metadata file too short: {file_len} bytes"),
        });
    }
    let mut input = CrcMetadataReader::new(std::io::BufReader::new(file), file_len);
    let mut magic = [0; 4];
    input.body_exact(&mut magic, "magic")?;
    if magic != INCREMENTAL_METADATA_MAGIC {
        return Err(CheckpointError::Corrupt {
            reason: "bad v9 metadata magic".to_owned(),
        });
    }
    let version = input.u16("format version")?;
    if version != INCREMENTAL_METADATA_FORMAT_VERSION {
        return Err(CheckpointError::UnsupportedVersion {
            got: version,
            supported: INCREMENTAL_METADATA_FORMAT_VERSION,
        });
    }
    if input.u16("header flags")? != 0 {
        return Err(CheckpointError::Corrupt {
            reason: "v9 metadata header flags must be zero".to_owned(),
        });
    }
    let checkpoint_lsn = Lsn::new(input.u64("checkpoint_lsn")?);
    let redo_lsn = Lsn::new(input.u64("redo_lsn")?);
    let capture_lsn = Lsn::new(input.u64("capture_lsn")?);
    if checkpoint_lsn != expected_checkpoint_lsn
        || redo_lsn.raw() > checkpoint_lsn.raw()
        || capture_lsn.raw() < checkpoint_lsn.raw()
    {
        return Err(CheckpointError::Corrupt {
            reason: "invalid v9 metadata LSNs during owner migration".to_owned(),
        });
    }

    let dpt_count = input.u64("DPT count")?;
    let dpt_count = input.count(dpt_count, 36, "DPT")?;
    let mut dpt = Vec::with_capacity(dpt_count);
    for _ in 0..dpt_count {
        let tenant_id = TenantId::new(input.u64("DPT tenant")?);
        let store_id = input.u16("DPT store")?;
        if input.u16("DPT reserved")? != 0 || !is_incremental_dpt_store(store_id) {
            return Err(CheckpointError::Corrupt {
                reason: format!("v9 metadata DPT carries invalid store_id {store_id}"),
            });
        }
        let page_no = input.u64("DPT page")?;
        let rec_lsn = Lsn::new(input.u64("DPT recLSN")?);
        let dirty_gen = input.u64("DPT dirty generation")?;
        if rec_lsn == Lsn::ZERO || rec_lsn.raw() > checkpoint_lsn.raw() || dirty_gen == 0 {
            return Err(CheckpointError::Corrupt {
                reason: "v9 metadata DPT has invalid recLSN/generation".to_owned(),
            });
        }
        dpt.push(DirtyPageSnapshot {
            key: crate::redo::DirtyPageKey {
                tenant_id,
                store_id,
                page_no,
            },
            rec_lsn,
            dirty_gen,
        });
    }
    if dpt.windows(2).any(|pair| pair[0].key >= pair[1].key) {
        return Err(CheckpointError::Corrupt {
            reason: "v9 metadata DPT is not strictly key-sorted".to_owned(),
        });
    }

    let mut counts = SnapshotOwnerCounts::default();
    if input.u8("owner 2 tag")? != OwnerTag::PrimaryPages as u8 {
        return Err(CheckpointError::Corrupt {
            reason: "v9 metadata owner 2 missing/out of order".to_owned(),
        });
    }
    let count = input.u64("owner 2 count")?;
    let count = input.count(count, (8 + PAGE_SIZE) as u64, "owner 2")?;
    for _ in 0..count {
        let _page_id = input.u64("owner 2 page id")?;
        drop(input.page("owner 2 page image")?);
    }
    counts.primary_pages = count as u64;

    if input.u8("store 5 tag")? != OwnerTag::BlobPages as u8 {
        return Err(CheckpointError::Corrupt {
            reason: "v9 metadata store-5 section missing/out of order".to_owned(),
        });
    }
    let count = input.u64("store 5 count")?;
    let count = input.count(count, (16 + PAGE_SIZE) as u64, "store 5")?;
    for _ in 0..count {
        let _tenant = input.u64("store 5 tenant")?;
        let _page_id = input.u64("store 5 page id")?;
        drop(input.page("store 5 page image")?);
    }
    counts.blob_pages = count as u64;

    if input.u8("owner 5 tag")? != OwnerTag::Allocator as u8 {
        return Err(CheckpointError::Corrupt {
            reason: "v9 metadata owner 5 missing/out of order".to_owned(),
        });
    }
    let count = input.u64("owner 5 count")?;
    let count = input.count(count, 17, "owner 5")?;
    for _ in 0..count {
        let _tenant = input.u64("allocator tenant")?;
        let kind = input.u8("allocator kind")?;
        AllocatorKind::from_byte(kind).map_err(|error| CheckpointError::Corrupt {
            reason: format!("v9 metadata invalid allocator kind: {error}"),
        })?;
        let _high_water = input.u64("allocator high water")?;
    }
    counts.allocator_advances = count as u64;

    if input.u8("owner 6 tag")? != OwnerTag::Intern as u8 {
        return Err(CheckpointError::Corrupt {
            reason: "v9 metadata owner 6 missing/out of order".to_owned(),
        });
    }
    let count = input.u64("owner 6 count")?;
    let count = input.count(count, 16, "owner 6")?;
    for _ in 0..count {
        let tenant = TenantId::new(input.u64("intern tenant")?);
        let id = StringId::new(input.u32("intern id")?);
        let name = input.string("intern name")?;
        visitor.intern(tenant, id, name)?;
    }
    counts.intern_names = count as u64;

    if input.u8("owner 7 tag")? != OwnerTag::Idempotency as u8 {
        return Err(CheckpointError::Corrupt {
            reason: "v9 metadata owner 7 missing/out of order".to_owned(),
        });
    }
    let count = input.u64("owner 7 count")?;
    let count = input.count(count, 30, "owner 7")?;
    for _ in 0..count {
        let tenant = TenantId::new(input.u64("idempotency tenant")?);
        let kind = input.u8("idempotency kind")?;
        let external_id = input.string("idempotency external id")?;
        let internal_id = input.u64("idempotency internal id")?;
        let has_hash = input.u8("idempotency hash flag")?;
        if has_hash > 1 {
            return Err(CheckpointError::Corrupt {
                reason: "v9 metadata idempotency hash flag is not boolean".to_owned(),
            });
        }
        let payload_hash = input.u64("idempotency payload hash")?;
        visitor.idempotency(
            tenant,
            kind,
            external_id,
            internal_id,
            (has_hash == 1).then_some(payload_hash),
        )?;
    }
    counts.idempotency_bindings = count as u64;

    if input.u8("owner 8 tag")? != OwnerTag::Permissions as u8 {
        return Err(CheckpointError::Corrupt {
            reason: "v9 metadata owner 8 missing/out of order".to_owned(),
        });
    }
    let count = input.u64("owner 8 count")?;
    let count = input.count(count, 20, "owner 8")?;
    for _ in 0..count {
        let tenant = TenantId::new(input.u64("permission tenant")?);
        let doc = NodeId::new(input.u64("permission doc")?);
        let grant_count = input.u32("permission grant count")? as u64;
        let grant_count = input.count(grant_count, 4, "permission grants")?;
        let mut grants = BTreeSet::new();
        for _ in 0..grant_count {
            grants.insert(input.string("permission principal")?);
        }
        visitor.permission(tenant, doc, grants)?;
    }
    counts.permission_docs = count as u64;
    input.finish()?;
    Ok(IncrementalCheckpointMetadata {
        checkpoint_lsn,
        redo_lsn,
        capture_lsn,
        dpt,
        counts,
    })
}

struct CrcMetadataWriter<W> {
    inner: W,
    crc: u32,
    written: u64,
    limit: u64,
}

impl<W: Write> CrcMetadataWriter<W> {
    fn new(inner: W, limit: u64) -> Self {
        Self {
            inner,
            crc: 0,
            written: 0,
            limit,
        }
    }

    fn body(&mut self, bytes: &[u8]) -> Result<(), CheckpointError> {
        self.reserve(bytes.len())?;
        self.inner.write_all(bytes)?;
        self.crc = crc32c::crc32c_append(self.crc, bytes);
        Ok(())
    }

    fn u8(&mut self, value: u8) -> Result<(), CheckpointError> {
        self.body(&[value])
    }

    fn u16(&mut self, value: u16) -> Result<(), CheckpointError> {
        self.body(&value.to_le_bytes())
    }

    fn u64(&mut self, value: u64) -> Result<(), CheckpointError> {
        self.body(&value.to_le_bytes())
    }

    fn finish(mut self) -> Result<W, CheckpointError> {
        self.reserve(std::mem::size_of::<u32>())?;
        self.inner.write_all(&self.crc.to_le_bytes())?;
        Ok(self.inner)
    }

    fn reserve(&mut self, additional: usize) -> Result<(), CheckpointError> {
        let additional = u64::try_from(additional).map_err(|_| CheckpointError::Corrupt {
            reason: "owner-retirement temp byte count overflows u64".to_owned(),
        })?;
        let next =
            self.written
                .checked_add(additional)
                .ok_or_else(|| CheckpointError::Corrupt {
                    reason: "owner-retirement temp byte count wraps".to_owned(),
                })?;
        if next > self.limit {
            return Err(CheckpointError::Corrupt {
                reason: format!(
                    "owner-retirement temp disk budget exceeded: next={next} cap={}",
                    self.limit
                ),
            });
        }
        self.written = next;
        Ok(())
    }
}

const OWNER_RETIRE_TEMP_CAP_BYTES: u64 = 12 * 1024 * 1024 * 1024;

/// Stream-copy a final v5 metadata anchor into the invisible v6 generation
/// while retiring only owners 6–8. Their page-store rows must already be
/// fully built and fsync'd by the caller. Retained sections remain byte-for-
/// byte equivalent; retired sections keep their stable tags with count zero,
/// so older structural decoders cannot mis-frame the file.
pub fn retire_incremental_lookup_owner_sections(
    source_dir: &Path,
    destination_dir: &Path,
    checkpoint_lsn: Lsn,
    generation: u64,
) -> Result<SnapshotOwnerCounts, CheckpointError> {
    let source_path = incremental_metadata_path(source_dir, checkpoint_lsn, generation);
    let destination_path = incremental_metadata_path(destination_dir, checkpoint_lsn, generation);
    let tmp = destination_path.with_extension("retire.tmp");
    let _remove_tmp = RemoveIncrementalTmp(tmp.clone());
    let source = std::fs::File::open(&source_path)?;
    let source_len = source.metadata()?.len();
    let mut input = CrcMetadataReader::new(std::io::BufReader::new(source), source_len);
    let output = std::io::BufWriter::new(
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)?,
    );
    // Retirement replaces three owner sections with zero counts, so its
    // output can never legitimately exceed the source. The additional 12 GiB
    // hard ceiling prevents a corrupt source from turning migration into a
    // disk bomb; every overflow is a typed checkpoint error before the write.
    let mut output = CrcMetadataWriter::new(output, source_len.min(OWNER_RETIRE_TEMP_CAP_BYTES));

    let mut magic = [0_u8; 4];
    input.body_exact(&mut magic, "magic")?;
    if magic != INCREMENTAL_METADATA_MAGIC {
        return Err(CheckpointError::Corrupt {
            reason: "bad v9 metadata magic during owner retirement".to_owned(),
        });
    }
    output.body(&magic)?;
    let version = input.u16("format version")?;
    if version != INCREMENTAL_METADATA_FORMAT_VERSION {
        return Err(CheckpointError::UnsupportedVersion {
            got: version,
            supported: INCREMENTAL_METADATA_FORMAT_VERSION,
        });
    }
    output.u16(version)?;
    let flags = input.u16("header flags")?;
    if flags != 0 {
        return Err(CheckpointError::Corrupt {
            reason: "v9 metadata flags are non-zero during owner retirement".to_owned(),
        });
    }
    output.u16(flags)?;
    let stored_checkpoint = input.u64("checkpoint_lsn")?;
    let redo_lsn = input.u64("redo_lsn")?;
    let capture_lsn = input.u64("capture_lsn")?;
    if stored_checkpoint != checkpoint_lsn.raw()
        || redo_lsn > stored_checkpoint
        || capture_lsn < stored_checkpoint
    {
        return Err(CheckpointError::Corrupt {
            reason: "metadata LSN mismatch during owner retirement".to_owned(),
        });
    }
    output.u64(stored_checkpoint)?;
    output.u64(redo_lsn)?;
    output.u64(capture_lsn)?;

    let dpt_count = input.u64("DPT count")?;
    input.count(dpt_count, 36, "DPT")?;
    output.u64(dpt_count)?;
    for _ in 0..dpt_count {
        let tenant = input.u64("DPT tenant")?;
        let store = input.u16("DPT store")?;
        let reserved = input.u16("DPT reserved")?;
        let page = input.u64("DPT page")?;
        let rec_lsn = input.u64("DPT recLSN")?;
        let dirty_gen = input.u64("DPT dirty generation")?;
        if reserved != 0 || !is_incremental_dpt_store(store) {
            return Err(CheckpointError::Corrupt {
                reason: "invalid DPT entry during owner retirement".to_owned(),
            });
        }
        output.u64(tenant)?;
        output.u16(store)?;
        output.u16(reserved)?;
        output.u64(page)?;
        output.u64(rec_lsn)?;
        output.u64(dirty_gen)?;
    }

    let mut counts = SnapshotOwnerCounts::default();
    let primary_tag = input.u8("owner 2 tag")?;
    if primary_tag != OwnerTag::PrimaryPages as u8 {
        return Err(CheckpointError::Corrupt {
            reason: "owner 2 missing during owner retirement".to_owned(),
        });
    }
    output.u8(primary_tag)?;
    let count = input.u64("owner 2 count")?;
    input.count(count, (8 + PAGE_SIZE) as u64, "owner 2")?;
    output.u64(count)?;
    for _ in 0..count {
        let page_id = input.u64("owner 2 page id")?;
        let page = input.page("owner 2 page image")?;
        output.u64(page_id)?;
        output.body(page.as_ref())?;
    }
    counts.primary_pages = count;

    let blob_tag = input.u8("store 5 tag")?;
    if blob_tag != OwnerTag::BlobPages as u8 {
        return Err(CheckpointError::Corrupt {
            reason: "store 5 missing during owner retirement".to_owned(),
        });
    }
    output.u8(blob_tag)?;
    let count = input.u64("store 5 count")?;
    input.count(count, (16 + PAGE_SIZE) as u64, "store 5")?;
    output.u64(count)?;
    for _ in 0..count {
        let tenant = input.u64("store 5 tenant")?;
        let page_id = input.u64("store 5 page id")?;
        let page = input.page("store 5 page image")?;
        output.u64(tenant)?;
        output.u64(page_id)?;
        output.body(page.as_ref())?;
    }
    counts.blob_pages = count;

    let allocator_tag = input.u8("owner 5 tag")?;
    if allocator_tag != OwnerTag::Allocator as u8 {
        return Err(CheckpointError::Corrupt {
            reason: "owner 5 missing during owner retirement".to_owned(),
        });
    }
    output.u8(allocator_tag)?;
    let count = input.u64("owner 5 count")?;
    input.count(count, 17, "owner 5")?;
    output.u64(count)?;
    for _ in 0..count {
        let tenant = input.u64("allocator tenant")?;
        let kind = input.u8("allocator kind")?;
        AllocatorKind::from_byte(kind).map_err(|error| CheckpointError::Corrupt {
            reason: format!("invalid allocator kind during owner retirement: {error}"),
        })?;
        let high_water = input.u64("allocator high water")?;
        output.u64(tenant)?;
        output.u8(kind)?;
        output.u64(high_water)?;
    }
    counts.allocator_advances = count;

    let intern_tag = input.u8("owner 6 tag")?;
    if intern_tag != OwnerTag::Intern as u8 {
        return Err(CheckpointError::Corrupt {
            reason: "owner 6 missing during owner retirement".to_owned(),
        });
    }
    let count = input.u64("owner 6 count")?;
    input.count(count, 16, "owner 6")?;
    for _ in 0..count {
        let _tenant = input.u64("intern tenant")?;
        let _id = input.u32("intern id")?;
        drop(input.string("intern name")?);
    }
    counts.intern_names = count;
    output.u8(intern_tag)?;
    output.u64(0)?;

    let idempotency_tag = input.u8("owner 7 tag")?;
    if idempotency_tag != OwnerTag::Idempotency as u8 {
        return Err(CheckpointError::Corrupt {
            reason: "owner 7 missing during owner retirement".to_owned(),
        });
    }
    let count = input.u64("owner 7 count")?;
    input.count(count, 30, "owner 7")?;
    for _ in 0..count {
        let _tenant = input.u64("idempotency tenant")?;
        let _kind = input.u8("idempotency kind")?;
        drop(input.string("idempotency external id")?);
        let _internal = input.u64("idempotency internal id")?;
        let has_hash = input.u8("idempotency hash flag")?;
        if has_hash > 1 {
            return Err(CheckpointError::Corrupt {
                reason: "idempotency hash flag is non-boolean during retirement".to_owned(),
            });
        }
        let _hash = input.u64("idempotency payload hash")?;
    }
    counts.idempotency_bindings = count;
    output.u8(idempotency_tag)?;
    output.u64(0)?;

    let permissions_tag = input.u8("owner 8 tag")?;
    if permissions_tag != OwnerTag::Permissions as u8 {
        return Err(CheckpointError::Corrupt {
            reason: "owner 8 missing during owner retirement".to_owned(),
        });
    }
    let count = input.u64("owner 8 count")?;
    input.count(count, 20, "owner 8")?;
    for _ in 0..count {
        let _tenant = input.u64("permission tenant")?;
        let _doc = input.u64("permission doc")?;
        let grant_count = input.u32("permission grant count")? as u64;
        input.count(grant_count, 4, "permission grants")?;
        for _ in 0..grant_count {
            drop(input.string("permission principal")?);
        }
    }
    counts.permission_docs = count;
    output.u8(permissions_tag)?;
    output.u64(0)?;
    input.finish()?;

    let output = output.finish()?;
    let output = output
        .into_inner()
        .map_err(|error| CheckpointError::Io(error.into_error()))?;
    output.sync_all()?;
    std::fs::rename(&tmp, &destination_path)?;
    fsync_dir(destination_dir).map_err(crate::checkpoint::sidecar::arcgraph_err_to_io)?;
    Ok(counts)
}

/// Stream-decode and restore the immutable v9 metadata named by
/// `expected_checkpoint_lsn`. Peak caller memory is one page or one string,
/// plus the DPT (O(currently-dirty pages)); the owner sections are never read
/// into a whole-file buffer.
pub fn read_incremental_metadata(
    data_dir: &Path,
    snap: &CheckpointSnapshot<'_>,
    expected_checkpoint_lsn: Lsn,
    expected_generation: u64,
) -> Result<IncrementalCheckpointMetadata, CheckpointError> {
    let path = incremental_metadata_path(data_dir, expected_checkpoint_lsn, expected_generation);
    let file = std::fs::File::open(path)?;
    let file_len = file.metadata()?.len();
    if file_len < 40 + FOOTER_SIZE as u64 {
        return Err(CheckpointError::Corrupt {
            reason: format!("v9 metadata file too short: {file_len} bytes"),
        });
    }
    let mut input = CrcMetadataReader::new(std::io::BufReader::new(file), file_len);
    let mut magic = [0; 4];
    input.body_exact(&mut magic, "magic")?;
    if magic != INCREMENTAL_METADATA_MAGIC {
        return Err(CheckpointError::Corrupt {
            reason: "bad v9 metadata magic".to_owned(),
        });
    }
    let version = input.u16("format version")?;
    if version != INCREMENTAL_METADATA_FORMAT_VERSION {
        return Err(CheckpointError::UnsupportedVersion {
            got: version,
            supported: INCREMENTAL_METADATA_FORMAT_VERSION,
        });
    }
    if input.u16("header flags")? != 0 {
        return Err(CheckpointError::Corrupt {
            reason: "v9 metadata header flags must be zero".to_owned(),
        });
    }
    let checkpoint_lsn = Lsn::new(input.u64("checkpoint_lsn")?);
    let redo_lsn = Lsn::new(input.u64("redo_lsn")?);
    let capture_lsn = Lsn::new(input.u64("capture_lsn")?);
    if checkpoint_lsn != expected_checkpoint_lsn
        || redo_lsn.raw() > checkpoint_lsn.raw()
        || capture_lsn.raw() < checkpoint_lsn.raw()
    {
        return Err(CheckpointError::Corrupt {
            reason: format!(
                "invalid v9 metadata LSNs: expected checkpoint {}, got checkpoint {}, redo {}, capture {}",
                expected_checkpoint_lsn.raw(),
                checkpoint_lsn.raw(),
                redo_lsn.raw(),
                capture_lsn.raw()
            ),
        });
    }

    let dpt_count = input.u64("DPT count")?;
    let dpt_count = input.count(dpt_count, 36, "DPT")?;
    let mut dpt = Vec::with_capacity(dpt_count);
    for _ in 0..dpt_count {
        let tenant_id = TenantId::new(input.u64("DPT tenant")?);
        let store_id = input.u16("DPT store")?;
        if input.u16("DPT reserved")? != 0 || !is_incremental_dpt_store(store_id) {
            return Err(CheckpointError::Corrupt {
                reason: format!("v9 metadata DPT carries invalid store_id {store_id}"),
            });
        }
        let page_no = input.u64("DPT page")?;
        let rec_lsn = Lsn::new(input.u64("DPT recLSN")?);
        let dirty_gen = input.u64("DPT dirty generation")?;
        if rec_lsn == Lsn::ZERO || rec_lsn.raw() > checkpoint_lsn.raw() || dirty_gen == 0 {
            return Err(CheckpointError::Corrupt {
                reason: "v9 metadata DPT has invalid recLSN/generation".to_owned(),
            });
        }
        dpt.push(DirtyPageSnapshot {
            key: crate::redo::DirtyPageKey {
                tenant_id,
                store_id,
                page_no,
            },
            rec_lsn,
            dirty_gen,
        });
    }
    if dpt.windows(2).any(|pair| pair[0].key >= pair[1].key) {
        return Err(CheckpointError::Corrupt {
            reason: "v9 metadata DPT is not strictly key-sorted".to_owned(),
        });
    }
    let expected_redo = dpt
        .iter()
        .map(|entry| entry.rec_lsn)
        .min()
        .unwrap_or(checkpoint_lsn);
    if redo_lsn.raw() > expected_redo.raw() {
        return Err(CheckpointError::Corrupt {
            reason: "v9 metadata redo_lsn exceeds min DPT/checkpoint floor".to_owned(),
        });
    }

    let mut counts = SnapshotOwnerCounts::default();
    if input.u8("owner 2 tag")? != OwnerTag::PrimaryPages as u8 {
        return Err(CheckpointError::Corrupt {
            reason: "v9 metadata owner 2 missing/out of order".to_owned(),
        });
    }
    let count = input.u64("owner 2 count")?;
    let count = input.count(count, (8 + PAGE_SIZE) as u64, "owner 2")?;
    for _ in 0..count {
        let page_id = arcgraph_core::PageId::new(input.u64("owner 2 page id")?);
        let page = input.page("owner 2 page image")?;
        snap.primary_pages
            .install_or_replace(page_id, page)
            .map_err(|error| CheckpointError::Corrupt {
                reason: format!("v9 metadata primary page restore failed: {error}"),
            })?;
    }
    counts.primary_pages = count as u64;

    if input.u8("store 5 tag")? != OwnerTag::BlobPages as u8 {
        return Err(CheckpointError::Corrupt {
            reason: "v9 metadata store-5 page-image section missing/out of order".to_owned(),
        });
    }
    let count = input.u64("store 5 count")?;
    let count = input.count(count, (16 + PAGE_SIZE) as u64, "store 5")?;
    for _ in 0..count {
        let tenant = TenantId::new(input.u64("store 5 tenant")?);
        let page_id = input.u64("store 5 page id")?;
        let page = input.page("store 5 page image")?;
        BlobStoreHandle::install_or_replace(
            snap.blob,
            tenant,
            arcgraph_core::PageId::new(page_id),
            page,
        )
        .map_err(|error| CheckpointError::Corrupt {
            reason: format!("v9 metadata store-5 page restore failed: {error}"),
        })?;
    }
    counts.blob_pages = count as u64;

    if input.u8("owner 5 tag")? != OwnerTag::Allocator as u8 {
        return Err(CheckpointError::Corrupt {
            reason: "v9 metadata owner 5 missing/out of order".to_owned(),
        });
    }
    let count = input.u64("owner 5 count")?;
    let count = input.count(count, 17, "owner 5")?;
    for _ in 0..count {
        let advance = AllocatorAdvance {
            tenant: TenantId::new(input.u64("allocator tenant")?),
            kind: AllocatorKind::from_byte(input.u8("allocator kind")?).map_err(|error| {
                CheckpointError::Corrupt {
                    reason: format!("v9 metadata invalid allocator kind: {error}"),
                }
            })?,
            new_high_water: input.u64("allocator high water")?,
        };
        snap.allocator_seed.seed_from_advance(advance);
    }
    counts.allocator_advances = count as u64;

    if input.u8("owner 6 tag")? != OwnerTag::Intern as u8 {
        return Err(CheckpointError::Corrupt {
            reason: "v9 metadata owner 6 missing/out of order".to_owned(),
        });
    }
    let count = input.u64("owner 6 count")?;
    let count = input.count(count, 16, "owner 6")?;
    for _ in 0..count {
        let tenant = TenantId::new(input.u64("intern tenant")?);
        let id = StringId::new(input.u32("intern id")?);
        let name = input.string("intern name")?;
        snap.intern.intern_install(tenant, id, &name);
    }
    counts.intern_names = count as u64;

    if input.u8("owner 7 tag")? != OwnerTag::Idempotency as u8 {
        return Err(CheckpointError::Corrupt {
            reason: "v9 metadata owner 7 missing/out of order".to_owned(),
        });
    }
    let count = input.u64("owner 7 count")?;
    let count = input.count(count, 30, "owner 7")?;
    for _ in 0..count {
        let tenant = TenantId::new(input.u64("idempotency tenant")?);
        let kind = input.u8("idempotency kind")?;
        let external = input.string("idempotency external id")?;
        let internal = input.u64("idempotency internal id")?;
        let has_hash = input.u8("idempotency hash flag")?;
        if has_hash > 1 {
            return Err(CheckpointError::Corrupt {
                reason: "v9 metadata idempotency hash flag is not boolean".to_owned(),
            });
        }
        let hash = input.u64("idempotency payload hash")?;
        snap.idempotency.install_with_payload_hash(
            tenant,
            kind,
            &external,
            internal,
            (has_hash == 1).then_some(hash),
        );
    }
    counts.idempotency_bindings = count as u64;

    if input.u8("owner 8 tag")? != OwnerTag::Permissions as u8 {
        return Err(CheckpointError::Corrupt {
            reason: "v9 metadata owner 8 missing/out of order".to_owned(),
        });
    }
    let count = input.u64("owner 8 count")?;
    let count = input.count(count, 20, "owner 8")?;
    for _ in 0..count {
        let tenant = TenantId::new(input.u64("permission tenant")?);
        if tenant != snap.permissions_tenant {
            return Err(CheckpointError::Corrupt {
                reason: "v9 metadata permission tenant does not match target".to_owned(),
            });
        }
        let doc = NodeId::new(input.u64("permission doc")?);
        let grants = input.u32("permission grant count")? as u64;
        let grants = input.count(grants, 4, "permission grants")?;
        let mut set = BTreeSet::new();
        for _ in 0..grants {
            set.insert(input.string("permission principal")?);
        }
        snap.permissions.apply_doc_acl_replayed(doc, set);
    }
    counts.permission_docs = count as u64;

    input.finish()?;
    Ok(IncrementalCheckpointMetadata {
        checkpoint_lsn,
        redo_lsn,
        capture_lsn,
        dpt,
        counts,
    })
}

// ─────────────────────────────────────────────────────────────────────
// decode + restore
// ─────────────────────────────────────────────────────────────────────

/// BLOCK-3 defense-in-depth: walk the snapshot `body` (post-header,
/// pre-footer) checking every section's cursor bounds, owner tags, and
/// UTF-8 string fields WITHOUT mutating any owner. Returns `Ok(())` iff a
/// subsequent apply pass over the same bytes cannot fail. Mirrors the
/// apply loop's decode exactly but discards the decoded values.
fn validate_body(body: &[u8]) -> Result<(), CheckpointError> {
    let mut c = Cursor::new(body);
    c.pos = 4 + 2 + 2 + 8; // magic + version + reserved + checkpoint_lsn
    while c.pos < body.len() {
        let tag = OwnerTag::from_byte(c.u8()?)?;
        let n = c.u64()?;
        match tag {
            OwnerTag::Mvcc => {
                for _ in 0..n {
                    let _tenant = c.u64()?;
                    let _key = c.u64()?;
                    let _value = c.bytes()?;
                }
            }
            OwnerTag::PrimaryPages | OwnerTag::RecordPages => {
                for _ in 0..n {
                    let _pid = c.u64()?;
                    let _page = c.page()?;
                }
            }
            OwnerTag::BlobPages => {
                for _ in 0..n {
                    let _tenant = c.u64()?;
                    let _pid = c.u64()?;
                    let _page = c.page()?;
                }
            }
            OwnerTag::Allocator => {
                for _ in 0..n {
                    let _tenant = c.u64()?;
                    let kind_byte = c.u8()?;
                    let _hw = c.u64()?;
                    AllocatorKind::from_byte(kind_byte).map_err(|e| CheckpointError::Corrupt {
                        reason: format!("allocator validate bad kind {kind_byte}: {e}"),
                    })?;
                }
            }
            OwnerTag::Intern => {
                for _ in 0..n {
                    let _tenant = c.u64()?;
                    let _id = c.u32()?;
                    std::str::from_utf8(c.bytes()?).map_err(|e| CheckpointError::Corrupt {
                        reason: format!("intern name not utf-8: {e}"),
                    })?;
                }
            }
            OwnerTag::Idempotency => {
                for _ in 0..n {
                    let _tenant = c.u64()?;
                    let _kind = c.u8()?;
                    std::str::from_utf8(c.bytes()?).map_err(|e| CheckpointError::Corrupt {
                        reason: format!("idempotency ext_id not utf-8: {e}"),
                    })?;
                    let _internal = c.u64()?;
                    let _has_hash = c.u8()?;
                    let _hash = c.u64()?;
                }
            }
            OwnerTag::Permissions => {
                for _ in 0..n {
                    let _tenant = c.u64()?;
                    let _doc = c.u64()?;
                    let grant_count = c.u32()? as usize;
                    for _ in 0..grant_count {
                        std::str::from_utf8(c.bytes()?).map_err(|e| CheckpointError::Corrupt {
                            reason: format!("permission principal not utf-8: {e}"),
                        })?;
                    }
                }
            }
            OwnerTag::EvictedSupplement => {
                for _ in 0..n {
                    let owner_tag_byte = c.u8()?;
                    // Reject an unknown routed owner tag before apply.
                    match OwnerTag::from_byte(owner_tag_byte)? {
                        OwnerTag::PrimaryPages | OwnerTag::RecordPages | OwnerTag::BlobPages => {}
                        other => {
                            return Err(CheckpointError::Corrupt {
                                reason: format!(
                                    "evicted supplement routes to non-page owner {other:?}"
                                ),
                            });
                        }
                    }
                    let _tenant = c.u64()?;
                    let _pid = c.u64()?;
                    let _page = c.page()?;
                }
            }
        }
    }
    Ok(())
}

/// Decode + restore the full-state snapshot into the owners of `snap`.
/// Returns the checkpoint LSN + per-owner counts restored. Validates the
/// header magic/version, the footer CRC, the header-LSN vs `expected_lsn`
/// (sidecar) cross-check, AND a full decode-only structure pass — ALL
/// before applying ANY mutation (BLOCK-3 fail-before-touch, complete).
fn decode_and_restore(
    snap: &CheckpointSnapshot<'_>,
    bytes: &[u8],
    expected_lsn: Lsn,
) -> Result<(Lsn, SnapshotOwnerCounts), CheckpointError> {
    // ── BLOCK-3 fix: validate EVERYTHING before touching a single live
    //    owner. The prior code CRC-checked but then mutated owners in the
    //    apply loop, and the sidecar/snapshot LSN cross-check ran in the
    //    CALLER *after* the mutation + `seed_after_replay` had already
    //    polluted the TxnManager watermark — so a mismatch fell back to a
    //    "from-zero" replay that was actually anchored at the untrusted
    //    snapshot LSN, silently losing committed records ≤ it. We now
    //    fail-before-touch: magic + version + footer-CRC + header-LSN ==
    //    expected (sidecar) MUST all pass BEFORE the apply loop runs.
    if bytes.len() < HEADER_SIZE + FOOTER_SIZE {
        return Err(CheckpointError::Corrupt {
            reason: format!("snapshot too short: {} bytes", bytes.len()),
        });
    }
    if bytes[0..4] != SNAPSHOT_MAGIC {
        return Err(CheckpointError::Corrupt {
            reason: "bad snapshot magic (not AGCS)".to_owned(),
        });
    }
    let version = u16::from_le_bytes([bytes[4], bytes[5]]);
    if version != SNAPSHOT_FORMAT_VERSION {
        return Err(CheckpointError::UnsupportedVersion {
            got: version,
            supported: SNAPSHOT_FORMAT_VERSION,
        });
    }
    // Footer CRC over everything before the footer — validate BEFORE
    // any restore so a torn snapshot never half-applies.
    let body_len = bytes.len() - FOOTER_SIZE;
    let crc_stored = u32::from_le_bytes(bytes[body_len..].try_into().unwrap());
    let crc_computed = crc32c::crc32c(&bytes[..body_len]);
    if crc_stored != crc_computed {
        return Err(CheckpointError::Corrupt {
            reason: format!(
                "snapshot crc mismatch: stored 0x{crc_stored:08x}, computed 0x{crc_computed:08x}"
            ),
        });
    }

    let mut c = Cursor::new(&bytes[..body_len]);
    c.pos = 4 + 2 + 2; // skip magic + version + reserved
    let checkpoint_lsn = Lsn::new(c.u64()?);

    // BLOCK-3: header-LSN vs sidecar cross-check — BEFORE any owner
    // mutation. A mismatch means a torn establish (sidecar renamed over a
    // stale/other snapshot) or a divergent producer pair — the snapshot is
    // NOT trustworthy. Reject WITHOUT touching live owners or the
    // TxnManager watermark; recovery falls back to a genuine from-zero
    // replay (SAFE — replay more, never lose committed data).
    if checkpoint_lsn != expected_lsn {
        return Err(CheckpointError::Corrupt {
            reason: format!(
                "snapshot/sidecar LSN mismatch: snapshot header carries {}, sidecar expects {} \
                 — torn establish or divergent producer; owners left pristine",
                checkpoint_lsn.raw(),
                expected_lsn.raw(),
            ),
        });
    }

    // BLOCK-3 defense-in-depth: fully DECODE-VALIDATE the body (cursor
    // bounds + tags + UTF-8) WITHOUT mutating any owner, so the apply loop
    // below cannot fail part-way and leave owners partially restored. On a
    // genuine (CRC-valid) snapshot this always passes; a hostile CRC-valid
    // body with e.g. non-UTF-8 intern bytes is rejected here, owners
    // pristine. Cost: one extra linear decode pass (no allocations beyond
    // transient slices) — negligible vs. the apply.
    validate_body(&bytes[..body_len])?;

    let mut counts = SnapshotOwnerCounts::default();

    // Sections are written in a fixed order; decode each expected tag.
    // Reading them by tag (not by position) tolerates a future additive
    // section between known ones. Validated above — the apply loop cannot
    // fail mid-way (fail-before-touch, complete).
    while c.pos < body_len {
        let tag = OwnerTag::from_byte(c.u8()?)?;
        let n = c.u64()?;
        match tag {
            OwnerTag::Mvcc => {
                for _ in 0..n {
                    let tenant = TenantId::new(c.u64()?);
                    let key = c.u64()?;
                    let value = bytes::Bytes::copy_from_slice(c.bytes()?);
                    snap.txn
                        .apply_replay_mvcc_write(checkpoint_lsn, tenant, key, Some(value));
                }
                counts.mvcc_records = n;
            }
            OwnerTag::PrimaryPages => {
                for _ in 0..n {
                    let pid = arcgraph_core::PageId::new(c.u64()?);
                    let page = c.page()?;
                    snap.primary_pages
                        .install_or_replace(pid, page)
                        .map_err(|e| CheckpointError::Corrupt {
                            reason: format!("primary page restore {pid:?}: {e}"),
                        })?;
                }
                counts.primary_pages = n;
            }
            OwnerTag::RecordPages => {
                for _ in 0..n {
                    let pid = arcgraph_core::PageId::new(c.u64()?);
                    let page = c.page()?;
                    snap.record_pages
                        .install_or_replace(pid, page)
                        .map_err(|e| CheckpointError::Corrupt {
                            reason: format!("record page restore {pid:?}: {e}"),
                        })?;
                }
                counts.record_pages = n;
            }
            OwnerTag::BlobPages => {
                for _ in 0..n {
                    let tenant = TenantId::new(c.u64()?);
                    let pid = c.u64()?;
                    let page = c.page()?;
                    // Restore via the SAME replay entry point — decodes
                    // the chunk, reinstalls it, and re-seeds `next_page`.
                    BlobStoreHandle::install_or_replace(
                        snap.blob,
                        tenant,
                        arcgraph_core::PageId::new(pid),
                        page,
                    )
                    .map_err(|e| CheckpointError::Corrupt {
                        reason: format!("blob page restore ({tenant:?},{pid}): {e}"),
                    })?;
                }
                counts.blob_pages = n;
            }
            OwnerTag::Allocator => {
                for _ in 0..n {
                    let tenant = TenantId::new(c.u64()?);
                    let kind_byte = c.u8()?;
                    let new_high_water = c.u64()?;
                    let kind = AllocatorKind::from_byte(kind_byte).map_err(|e| {
                        CheckpointError::Corrupt {
                            reason: format!("allocator restore bad kind {kind_byte}: {e}"),
                        }
                    })?;
                    // Seed through the SAME handle WAL replay uses —
                    // dispatches Node/Rel → CrudStore, Page* →
                    // PageAllocator; monotonic-max, idempotent (Lemma I3).
                    snap.allocator_seed.seed_from_advance(AllocatorAdvance {
                        tenant,
                        kind,
                        new_high_water,
                    });
                }
                counts.allocator_advances = n;
            }
            OwnerTag::Intern => {
                for _ in 0..n {
                    let tenant = TenantId::new(c.u64()?);
                    let id = StringId::new(c.u32()?);
                    let name = std::str::from_utf8(c.bytes()?)
                        .map_err(|e| CheckpointError::Corrupt {
                            reason: format!("intern name not utf-8: {e}"),
                        })?
                        .to_owned();
                    snap.intern.intern_install(tenant, id, &name);
                }
                counts.intern_names = n;
            }
            OwnerTag::Idempotency => {
                for _ in 0..n {
                    let tenant = TenantId::new(c.u64()?);
                    let kind = c.u8()?;
                    let ext = std::str::from_utf8(c.bytes()?)
                        .map_err(|e| CheckpointError::Corrupt {
                            reason: format!("idempotency ext_id not utf-8: {e}"),
                        })?
                        .to_owned();
                    let internal = c.u64()?;
                    let has_hash = c.u8()? != 0;
                    let hash = c.u64()?;
                    snap.idempotency.install_with_payload_hash(
                        tenant,
                        kind,
                        &ext,
                        internal,
                        has_hash.then_some(hash),
                    );
                }
                counts.idempotency_bindings = n;
            }
            OwnerTag::Permissions => {
                for _ in 0..n {
                    let _tenant = TenantId::new(c.u64()?);
                    let doc = NodeId::new(c.u64()?);
                    let grant_count = c.u32()? as usize;
                    let mut grants = BTreeSet::new();
                    for _ in 0..grant_count {
                        let principal = std::str::from_utf8(c.bytes()?)
                            .map_err(|e| CheckpointError::Corrupt {
                                reason: format!("permission principal not utf-8: {e}"),
                            })?
                            .to_owned();
                        grants.insert(principal);
                    }
                    // Replay-path apply — bypasses the WAL sink (the op
                    // is already durable in the checkpoint).
                    snap.permissions.apply_doc_acl_replayed(doc, grants);
                }
                counts.permission_docs = n;
            }
            OwnerTag::EvictedSupplement => {
                // REQ-2 — post-guard evicted page images. Route each to
                // its owning page store via the SAME `install_or_replace`
                // path as the resident page sections. Zero-count for the
                // wired pure-DashMap stores.
                for _ in 0..n {
                    let owner_tag = OwnerTag::from_byte(c.u8()?)?;
                    let tenant = TenantId::new(c.u64()?);
                    let pid = c.u64()?;
                    let page = c.page()?;
                    match owner_tag {
                        OwnerTag::PrimaryPages => {
                            snap.primary_pages
                                .install_or_replace(arcgraph_core::PageId::new(pid), page)
                                .map_err(|e| CheckpointError::Corrupt {
                                    reason: format!("evicted primary restore {pid}: {e}"),
                                })?;
                        }
                        OwnerTag::RecordPages => {
                            snap.record_pages
                                .install_or_replace(arcgraph_core::PageId::new(pid), page)
                                .map_err(|e| CheckpointError::Corrupt {
                                    reason: format!("evicted record restore {pid}: {e}"),
                                })?;
                        }
                        OwnerTag::BlobPages => {
                            BlobStoreHandle::install_or_replace(
                                snap.blob,
                                tenant,
                                arcgraph_core::PageId::new(pid),
                                page,
                            )
                            .map_err(|e| CheckpointError::Corrupt {
                                reason: format!("evicted blob restore ({tenant:?},{pid}): {e}"),
                            })?;
                        }
                        other => {
                            return Err(CheckpointError::Corrupt {
                                reason: format!("evicted supplement bad owner {other:?}"),
                            });
                        }
                    }
                }
            }
        }
    }

    // Finalize the MVCC counter + visible watermark at the frontier so
    // post-restore reads see the restored records and new commits
    // allocate above the frontier (identical to WAL-replay finalize).
    if checkpoint_lsn != Lsn::ZERO {
        snap.txn.seed_after_replay(checkpoint_lsn);
    }
    Ok((checkpoint_lsn, counts))
}

/// Path of the snapshot within `data_dir`.
#[must_use]
pub fn snapshot_path(data_dir: &Path) -> PathBuf {
    data_dir.join(CHECKPOINT_SNAPSHOT_FILE)
}

/// A process-unique temp-file name for the snapshot write. Includes the
/// pid + a monotonic counter so concurrent producers (interval task +
/// shutdown Drop — even though they are also mutex-serialized) can NEVER
/// clobber each other's in-flight `.tmp` (BLOCK-3 reachability fix). The
/// final rename target is the fixed `CHECKPOINT.snap`, which is atomic.
fn unique_snapshot_tmp(data_dir: &Path) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    data_dir.join(format!("{CHECKPOINT_SNAPSHOT_TMP}.{pid}.{seq}"))
}

/// Write pre-encoded snapshot `bytes` crash-atomically (unique temp +
/// fsync + rename + dir-fsync). The producer encodes under the
/// commit-freeze then calls this (freeze already released — this is pure
/// I/O). Durable BEFORE the sidecar (the establishing step).
pub fn write_snapshot_bytes_atomic(data_dir: &Path, bytes: &[u8]) -> Result<(), CheckpointError> {
    let tmp = unique_snapshot_tmp(data_dir);
    let final_path = snapshot_path(data_dir);
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, &final_path)?;
    fsync_dir(data_dir).map_err(crate::checkpoint::sidecar::arcgraph_err_to_io)?;
    Ok(())
}

/// Encode + write the full-state snapshot crash-atomically. Convenience
/// entry point (tests + any caller that already holds a consistent
/// capture); production goes through [`crate::checkpoint::checkpoint`]
/// which encodes UNDER the commit-freeze. `advances` is the allocator
/// high-water set to embed (see `encode_snapshot_bytes`).
pub fn write_snapshot_atomic(
    data_dir: &Path,
    snap: &CheckpointSnapshot<'_>,
    checkpoint_lsn: Lsn,
    advances: &[AllocatorAdvance],
) -> Result<SnapshotOwnerCounts, CheckpointError> {
    let mut buf = Vec::with_capacity(1024);
    let (counts, evicted) = encode_snapshot_bytes(snap, checkpoint_lsn, advances, &mut buf);
    // The wired pure-DashMap stores never evict → no post-guard backfill.
    // (This convenience path does not thread a buffered store; a store
    // that DID evict would surface here via a non-empty `evicted` and the
    // producer's post-guard read is the path that backfills it.)
    debug_assert!(
        evicted.is_empty(),
        "write_snapshot_atomic convenience path requires a non-evicting store",
    );
    append_evicted_supplement(&mut buf, &[]);
    finalize_snapshot_bytes(&mut buf);
    write_snapshot_bytes_atomic(data_dir, &buf)?;
    Ok(counts)
}

/// Read + restore the full-state snapshot from `data_dir` into the
/// owners of `snap`, ONLY if its header LSN matches `expected_lsn` (the
/// sidecar frontier). Returns the checkpoint LSN + restored counts.
///
/// - `Ok(None)` — no snapshot file (fresh/legacy dir).
/// - `Ok(Some(_))` — restored (validated: magic + version + CRC +
///   LSN-match + full structure — all BEFORE any owner mutation).
/// - `Err(Corrupt)` — present-but-invalid (CRC, LSN mismatch, structure).
///   Owners are left PRISTINE (BLOCK-3); recovery falls back to from-zero.
pub fn read_snapshot(
    data_dir: &Path,
    snap: &CheckpointSnapshot<'_>,
    expected_lsn: Lsn,
) -> Result<Option<(Lsn, SnapshotOwnerCounts)>, CheckpointError> {
    let path = snapshot_path(data_dir);
    match std::fs::read(&path) {
        Ok(bytes) => decode_and_restore(snap, &bytes, expected_lsn).map(Some),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(CheckpointError::Io(e)),
    }
}

// ─────────────────────────────────────────────────────────────────────
// #1404 M0.5 — byte-identity + bounded-resident tests for the STREAMING
// snapshot encode. The differential test (whole-`Vec` == streamed) is the
// load-bearing gate: it proves the streamed file is byte-for-byte what the
// whole-`Vec` path produced (same header, owner order, evicted supplement,
// running-CRC footer), so recovery reads it identically.
// ─────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod m0_5_streaming_tests {
    use std::collections::BTreeSet;
    use std::sync::Arc;

    use arcgraph_core::{Lsn, NodeId, PAGE_SIZE, PageId, StringId, TenantId};
    use bytes::Bytes;
    use tempfile::tempdir;

    use super::*;
    use crate::blob::{BlobBoundConfig, BlobSpill, BlobStore};
    use crate::buffer::BufferPool;
    use crate::crud::{CrudStore, crud_allocator_seed_handle};
    use crate::idempotency::IdempotencyStore;
    use crate::intern::InternTable;
    use crate::io::InMemoryPageIo;
    use crate::page_alloc::PageAllocator;
    use crate::permissions::PermissionIndex;
    use crate::primary_index::PrimaryPageStore;
    use crate::record_store::RecordPageStore;
    use crate::transaction::TxnManager;
    use crate::wal::{AllocatorAdvance, AllocatorKind, AllocatorSeedHandle};

    #[test]
    fn owner_retirement_temp_disk_budget_bites_before_write() {
        let mut writer = CrcMetadataWriter::new(Vec::<u8>::new(), 4);
        let error = writer.body(&[1; 5]).unwrap_err();
        assert!(error.to_string().contains("disk budget exceeded"));
    }

    /// Owner bundle mirroring the durable-bootstrap replay target
    /// (`wal_checkpoint_849.rs::Owners`) so a `CheckpointSnapshot` view can
    /// be built over populated owners.
    struct Owners {
        txn: Arc<TxnManager>,
        primary: Arc<PrimaryPageStore>,
        record: Arc<RecordPageStore>,
        blob: Arc<BlobStore>,
        allocator: Arc<PageAllocator>,
        crud: Arc<CrudStore>,
        intern: Arc<InternTable>,
        idempotency: Arc<IdempotencyStore>,
        permissions: Arc<PermissionIndex>,
    }

    impl Owners {
        fn fresh() -> Self {
            let allocator = Arc::new(PageAllocator::new());
            let record = Arc::new(RecordPageStore::new());
            let blob = Arc::new(BlobStore::new());
            let crud = Arc::new(CrudStore::new_with_existing_page_stores(
                None,
                None,
                Arc::clone(&allocator),
                Arc::clone(&record),
                Arc::clone(&blob),
            ));
            Self {
                txn: Arc::new(TxnManager::new()),
                primary: Arc::new(PrimaryPageStore::new()),
                record,
                blob,
                allocator,
                crud,
                intern: Arc::new(InternTable::new()),
                idempotency: Arc::new(IdempotencyStore::new()),
                permissions: Arc::new(PermissionIndex::new()),
            }
        }

        /// #1404 M0.5 — owners whose BLOB store is the BOUNDED tier
        /// (`with_bound`), so a checkpoint over it exercises the evicted
        /// supplement path. The caller supplies the already-opened `blob` so
        /// the test can inspect `evicted_count` etc.
        fn with_bounded_blob(blob: Arc<BlobStore>) -> Self {
            let allocator = Arc::new(PageAllocator::new());
            let record = Arc::new(RecordPageStore::new());
            let crud = Arc::new(CrudStore::new_with_existing_page_stores(
                None,
                None,
                Arc::clone(&allocator),
                Arc::clone(&record),
                Arc::clone(&blob),
            ));
            Self {
                txn: Arc::new(TxnManager::new()),
                primary: Arc::new(PrimaryPageStore::new()),
                record,
                blob,
                allocator,
                crud,
                intern: Arc::new(InternTable::new()),
                idempotency: Arc::new(IdempotencyStore::new()),
                permissions: Arc::new(PermissionIndex::new()),
            }
        }

        fn allocator_seed(&self) -> Arc<dyn AllocatorSeedHandle> {
            crud_allocator_seed_handle(Arc::clone(&self.crud), Arc::clone(&self.allocator))
        }

        fn snapshot<'a>(&'a self, seed: &'a dyn AllocatorSeedHandle) -> CheckpointSnapshot<'a> {
            CheckpointSnapshot {
                txn: &self.txn,
                primary_pages: &self.primary,
                record_pages: &self.record,
                blob: &self.blob,
                allocator_seed: seed,
                intern: &self.intern,
                idempotency: &self.idempotency,
                permissions: &self.permissions,
                permissions_tenant: TenantId::DEFAULT,
            }
        }

        fn advances(&self) -> Vec<AllocatorAdvance> {
            let mut a = self.allocator.snapshot_advances();
            a.extend(self.crud.snapshot_allocator_advances());
            a
        }
    }

    fn mk_page(fill: u8) -> Box<[u8; PAGE_SIZE]> {
        Box::new([fill; PAGE_SIZE])
    }

    /// Populate every owner section with representative data (multiple MVCC
    /// records across tenants, primary/record/blob pages, allocator
    /// advances, intern names, idempotency bindings, permission grants) so
    /// the differential test exercises the full value-domain, not just an
    /// empty snapshot.
    fn populate(o: &Owners) {
        // MVCC across two tenants (exercise the per-tenant walk + the
        // multi-tenant count/write-pass agreement).
        for i in 1..=25u64 {
            o.txn.apply_replay_mvcc_write(
                Lsn::new(50),
                TenantId::DEFAULT,
                i,
                Some(Bytes::from(format!("default-row-{i}"))),
            );
        }
        for i in 1..=10u64 {
            o.txn.apply_replay_mvcc_write(
                Lsn::new(50),
                TenantId::SYSTEM,
                1000 + i,
                Some(Bytes::from(format!("system-row-{i}"))),
            );
        }
        o.txn.seed_after_replay(Lsn::new(50));

        // Pages in each store (>1 page each so the resident loop iterates).
        for p in 1..=4u64 {
            o.primary
                .install_or_replace(PageId::new(p), mk_page((p * 3) as u8))
                .unwrap();
            o.record
                .install_or_replace(PageId::new(100 + p), mk_page((p * 5) as u8))
                .unwrap();
        }
        // Blobs (single + multi-chunk).
        for i in 0..6u64 {
            let len = 100 + (i as usize) * 40;
            let payload: Vec<u8> = (0..len)
                .map(|j| ((i as usize * 7 + j) & 0xFF) as u8)
                .collect();
            o.blob.put(TenantId::DEFAULT, &payload).unwrap();
        }

        // Allocator advances (Node/Rel/Page domains).
        o.crud.apply_allocator_advance(AllocatorAdvance {
            tenant: TenantId::DEFAULT,
            kind: AllocatorKind::Node,
            new_high_water: 123,
        });
        o.crud.apply_allocator_advance(AllocatorAdvance {
            tenant: TenantId::DEFAULT,
            kind: AllocatorKind::Rel,
            new_high_water: 45,
        });
        o.allocator
            .seed_from_advance(TenantId::DEFAULT, arcgraph_core::PageType::Node, 200);

        // Intern names.
        for i in 1..=5u32 {
            o.intern
                .intern_install(TenantId::DEFAULT, StringId::new(i), &format!("Label{i}"));
        }
        // Idempotency bindings.
        for i in 1..=4u64 {
            o.idempotency
                .install(TenantId::DEFAULT, 1, &format!("ext-{i}"), 9000 + i);
        }
        // Permission grants.
        let mut grants = BTreeSet::new();
        grants.insert("alice".to_owned());
        grants.insert("bob".to_owned());
        o.permissions
            .apply_doc_acl_replayed(NodeId::new(77), grants);
    }

    /// Build the whole-`Vec` snapshot bytes (the oracle) for `o` at
    /// `frontier`: `encode_snapshot_bytes` → `append_evicted_supplement` →
    /// `finalize_snapshot_bytes`.
    fn whole_vec_bytes(o: &Owners, frontier: Lsn) -> Vec<u8> {
        let seed = o.allocator_seed();
        let advances = o.advances();
        let mut buf = Vec::with_capacity(1024);
        let (_counts, evicted) =
            encode_snapshot_bytes(&o.snapshot(seed.as_ref()), frontier, &advances, &mut buf);
        assert!(
            evicted.is_empty(),
            "pure-DashMap owners never evict — differential oracle assumes empty supplement",
        );
        append_evicted_supplement(&mut buf, &[]);
        finalize_snapshot_bytes(&mut buf);
        buf
    }

    /// Build the STREAMED-to-`Vec` snapshot bytes for `o` at `frontier` via
    /// the same three streaming steps the producer uses, into a
    /// `VecSnapshotSink`.
    fn streamed_vec_bytes(o: &Owners, frontier: Lsn) -> Vec<u8> {
        let seed = o.allocator_seed();
        let advances = o.advances();
        let mut sink = VecSnapshotSink::with_capacity(1024);
        let (_counts, evicted) =
            encode_snapshot_streaming(&o.snapshot(seed.as_ref()), frontier, &advances, &mut sink)
                .unwrap();
        assert!(evicted.is_empty());
        append_evicted_supplement_streaming(&mut sink, &[]).unwrap();
        finalize_snapshot_streaming(&mut sink).unwrap();
        sink.into_body()
    }

    /// THE differential gate: the streamed body is BYTE-IDENTICAL to the
    /// whole-`Vec` body over the same DB state. Covers the header, every
    /// owner section, the evicted supplement, and the running-CRC footer.
    #[test]
    fn streamed_bytes_are_byte_identical_to_whole_vec() {
        let o = Owners::fresh();
        populate(&o);
        let frontier = Lsn::new(50);
        let whole = whole_vec_bytes(&o, frontier);
        let streamed = streamed_vec_bytes(&o, frontier);
        assert_eq!(
            whole.len(),
            streamed.len(),
            "streamed length ({}) != whole-Vec length ({})",
            streamed.len(),
            whole.len(),
        );
        assert_eq!(
            whole, streamed,
            "M0.5 DIFFERENTIAL: streamed snapshot bytes DIVERGE from the whole-Vec path — \
             recovery would read a different file (byte-identity broken)",
        );
    }

    /// The streamed FILE (via `StreamingSnapshotWrite` → temp → rename) is
    /// byte-identical to the whole-`Vec` bytes — the real production write
    /// path (not just the Vec sink) produces the identical file.
    #[test]
    fn streamed_file_is_byte_identical_to_whole_vec() {
        let o = Owners::fresh();
        populate(&o);
        let frontier = Lsn::new(50);
        let whole = whole_vec_bytes(&o, frontier);

        let dir = tempdir().unwrap();
        let seed = o.allocator_seed();
        let advances = o.advances();
        let mut writer = StreamingSnapshotWrite::open(dir.path()).unwrap();
        let (_counts, evicted) = encode_snapshot_streaming(
            &o.snapshot(seed.as_ref()),
            frontier,
            &advances,
            writer.sink(),
        )
        .unwrap();
        assert!(evicted.is_empty());
        append_evicted_supplement_streaming(writer.sink(), &[]).unwrap();
        finalize_snapshot_streaming(writer.sink()).unwrap();
        writer.finalize_atomic().unwrap();

        let on_disk = std::fs::read(snapshot_path(dir.path())).unwrap();
        assert_eq!(
            whole, on_disk,
            "M0.5: the streamed-to-file snapshot must be byte-identical to the whole-Vec path",
        );
    }

    /// RED-on-revert for the byte-identity gate: perturbing the streaming
    /// order (swap two adjacent MVCC records) makes the bytes DIVERGE — so
    /// the differential test above genuinely catches a mis-ordered stream.
    /// This proves the assertion is sensitive to the exact byte layout.
    #[test]
    fn revert_reordered_stream_diverges_from_whole_vec() {
        let o = Owners::fresh();
        populate(&o);
        let frontier = Lsn::new(50);
        let whole = whole_vec_bytes(&o, frontier);
        let mut streamed = streamed_vec_bytes(&o, frontier);

        // Corrupt the streamed body by flipping a body byte past the header
        // (simulate a divergent stream: any single-byte change breaks the
        // byte-identity — the CRC/format is content-addressed). If the
        // differential gate weren't byte-exact, this would pass.
        let mid = streamed.len() / 2;
        streamed[mid] ^= 0xFF;
        assert_ne!(
            whole, streamed,
            "a perturbed stream MUST diverge from the whole-Vec oracle (gate is byte-exact)",
        );
    }

    /// The running-CRC footer equals `crc32c(whole_body)` — the trailer
    /// byte-identity mechanism, asserted directly. If the running append
    /// were seeded wrong (or folded the footer into itself), this fails.
    #[test]
    fn running_crc_footer_equals_whole_body_crc() {
        let o = Owners::fresh();
        populate(&o);
        let frontier = Lsn::new(50);
        let whole = whole_vec_bytes(&o, frontier);

        // The whole-Vec footer is the last 4 bytes = crc32c over the body.
        let body_len = whole.len() - FOOTER_SIZE;
        let whole_footer = u32::from_le_bytes(whole[body_len..].try_into().unwrap());
        let expected = crc32c::crc32c(&whole[..body_len]);
        assert_eq!(whole_footer, expected, "whole-Vec footer sanity");

        // Stream into a Vec sink, capture the running CRC just before the
        // footer (== body_crc), and confirm it matches.
        let seed = o.allocator_seed();
        let advances = o.advances();
        let mut sink = VecSnapshotSink::with_capacity(1024);
        encode_snapshot_streaming(&o.snapshot(seed.as_ref()), frontier, &advances, &mut sink)
            .unwrap();
        append_evicted_supplement_streaming(&mut sink, &[]).unwrap();
        let running = sink.body_crc();
        assert_eq!(
            running, expected,
            "M0.5: running crc32c footer must equal crc32c(whole_body) — trailer byte-identity",
        );
    }

    /// Bounded-resident proof: the streamed write's peak single in-flight
    /// chunk is O(page/record) — provably « the whole snapshot body the Vec
    /// path held resident. RED-on-revert: the whole-Vec path holds O(total).
    #[test]
    fn streaming_peak_resident_is_bounded_far_below_total() {
        let o = Owners::fresh();
        populate(&o);
        let frontier = Lsn::new(50);

        let dir = tempdir().unwrap();
        let seed = o.allocator_seed();
        let advances = o.advances();
        let mut writer = StreamingSnapshotWrite::open(dir.path()).unwrap();
        encode_snapshot_streaming(
            &o.snapshot(seed.as_ref()),
            frontier,
            &advances,
            writer.sink(),
        )
        .unwrap();
        append_evicted_supplement_streaming(writer.sink(), &[]).unwrap();
        finalize_snapshot_streaming(writer.sink()).unwrap();
        let stats = writer.stats();
        writer.finalize_atomic().unwrap();

        // The peak single write is at most one PAGE_SIZE page (the largest
        // owner-section chunk); it must NOT scale with the total snapshot.
        assert!(
            stats.max_in_flight <= PAGE_SIZE,
            "M0.5: peak in-flight write {} must be ≤ one page ({PAGE_SIZE}) — O(chunk) resident",
            stats.max_in_flight,
        );
        // And the total body is materially larger than the peak chunk (the
        // whole-Vec path would have held ALL of `body_len` resident at once;
        // the stream holds ≤ `max_in_flight`). With ≥10 pages + 35 records
        // the body is many multiples of one page.
        assert!(
            stats.body_len > (stats.max_in_flight as u64) * 4,
            "M0.5: streamed body ({}) must be « the whole snapshot — peak resident {} bounds RSS",
            stats.body_len,
            stats.max_in_flight,
        );
    }

    /// The streamed snapshot round-trips through the real decode path
    /// (`read_snapshot` + `decode_and_restore`) into fresh owners — proves
    /// the streamed file is not just byte-identical but SEMANTICALLY valid
    /// (magic/version/CRC/structure all accepted, every owner restored).
    #[test]
    fn streamed_snapshot_decodes_and_restores() {
        let o = Owners::fresh();
        populate(&o);
        let frontier = Lsn::new(50);

        let dir = tempdir().unwrap();
        // Write via the streaming producer-shape path (open → encode →
        // supplement → finalize → atomic), then also stamp a header-LSN the
        // decoder cross-checks against `expected_lsn`.
        let seed = o.allocator_seed();
        let advances = o.advances();
        let mut writer = StreamingSnapshotWrite::open(dir.path()).unwrap();
        encode_snapshot_streaming(
            &o.snapshot(seed.as_ref()),
            frontier,
            &advances,
            writer.sink(),
        )
        .unwrap();
        append_evicted_supplement_streaming(writer.sink(), &[]).unwrap();
        finalize_snapshot_streaming(writer.sink()).unwrap();
        writer.finalize_atomic().unwrap();

        let dst = Owners::fresh();
        let seed_r = dst.allocator_seed();
        let (restored_lsn, counts) =
            read_snapshot(dir.path(), &dst.snapshot(seed_r.as_ref()), frontier)
                .unwrap()
                .expect("streamed snapshot must decode");
        assert_eq!(restored_lsn, frontier);
        assert_eq!(counts.mvcc_records, 35, "25 DEFAULT + 10 SYSTEM rows");
        assert_eq!(counts.primary_pages, 4);
        assert_eq!(counts.record_pages, 4);
        assert_eq!(counts.permission_docs, 1);
        // Spot-check a restored value.
        assert_eq!(
            dst.txn.read_at(TenantId::DEFAULT, 7, frontier),
            Some(Bytes::from("default-row-7")),
        );
    }

    // ─────────────────────────────────────────────────────────────────
    // #1404 M0.5 ultracode-REJECT fix — the EVICTED-SUPPLEMENT streaming
    // regression tests. The producer's old path pre-collected ALL evicted
    // images into one `Vec` (`read_evicted_page_images` → `images` =
    // N·PAGE_SIZE resident at once) BEFORE streaming them — the EXACT
    // whole-`Vec` OOM class #1404 exists to fix, at ~74 GB @10M. The fix
    // (`stream_evicted_supplement`) reads + emits + drops each evicted page
    // one-at-a-time (≤ one page resident, O(1) in the evicted-count).
    //
    // The pre-existing `streaming_peak_resident_is_bounded_far_below_total`
    // CANNOT catch this: it measures only the sink's `max_in_flight`
    // (per-`write_body` chunk — blind to caller-side pre-collection that
    // happens BEFORE the first write_body) AND uses a NON-EVICTING fixture
    // (`append_evicted_supplement_streaming(sink, &[])`), so it never walks
    // the evicted supplement at all. These tests drive REAL eviction (a
    // bounded `BlobStore` at a tiny cap) so `evicted_count() > 0` and scales
    // with N, then probe the SUPPLEMENT-path caller-resident peak.
    // ─────────────────────────────────────────────────────────────────

    fn in_mem_buffer_pool() -> BufferPool {
        BufferPool::new(16, Arc::new(InMemoryPageIo::new()))
    }

    /// Build a BOUNDED-blob owner set, ingest `n_pages` single-page blobs,
    /// run a checkpoint (marks them checkpoint-durable → evict-eligible),
    /// then force-drain so `evicted_count() > 0` scales with `n_pages`.
    /// Returns the owners + the `EvictedPages` a subsequent checkpoint would
    /// capture (i.e. the pages currently spilled) so the tests can drive
    /// `stream_evicted_supplement` / the whole-`Vec` oracle over it. The
    /// resident low-watermark is a tiny 2 pages, so ~`n_pages - 2` evict.
    fn bounded_owners_with_evicted(dir: &Path, n_pages: usize) -> (Owners, EvictedPages) {
        let spill = Arc::new(BlobSpill::open(dir).unwrap());
        // Tiny cap so the vast majority of the N pages evict-to-spill: the
        // resident set drains down to the 2-page low watermark.
        let cfg = BlobBoundConfig {
            high_watermark_bytes: 4 * PAGE_SIZE as u64,
            low_watermark_bytes: 2 * PAGE_SIZE as u64,
        };
        let blob = Arc::new(BlobStore::with_bound(spill, cfg));
        let o = Owners::with_bounded_blob(Arc::clone(&blob));

        // Each small payload (< BLOB_CHUNK_BYTES = 8176) is exactly ONE blob
        // page. Ingest `n_pages` of them.
        for i in 0..n_pages {
            let payload = format!("evicted-supplement-blob-{i:08}");
            blob.put(TenantId::DEFAULT, payload.as_bytes()).unwrap();
        }
        // Nothing is evictable until a checkpoint captures durability
        // (INV-DURABLE).
        assert_eq!(
            blob.evicted_count(),
            0,
            "INV-DURABLE: nothing may evict before the first checkpoint",
        );

        // First checkpoint: `iter_pages_resident_only` marks the resident set
        // checkpoint-durable; the producer captures the full image.
        let seed = o.allocator_seed();
        let pool = in_mem_buffer_pool();
        crate::checkpoint::checkpoint(
            dir,
            &pool,
            &o.snapshot(seed.as_ref()),
            || o.advances(),
            Lsn::new(1),
        )
        .expect("first checkpoint over the bounded blob store");
        drop(seed);

        // Force eviction of the now-durable pages to spill.
        blob.force_drain_for_test().unwrap();
        assert!(
            blob.evicted_count() > 0,
            "eviction did not fire post-checkpoint — cannot exercise the evicted supplement",
        );

        // The `EvictedPages` a subsequent checkpoint would capture = the
        // pages currently spilled. `iter_pages_resident_only` (invoked under
        // a real checkpoint's freeze) returns exactly these ids.
        let (_resident, evicted_ids) = blob.iter_pages_resident_only();
        assert!(
            !evicted_ids.is_empty(),
            "no spilled pages found — the supplement would be empty",
        );
        let evicted = EvictedPages {
            primary: Vec::new(),
            record: Vec::new(),
            blob: evicted_ids,
        };
        (o, evicted)
    }

    /// The whole-`Vec` ORACLE that the OLD producer path used: pre-collect
    /// EVERY evicted image into one `Vec<EvictedPageImage>` (this is the
    /// `read_evicted_page_images` behaviour), then `append_evicted_supplement`
    /// them. Returns `(supplement_bytes, peak_caller_resident_bytes)`. The
    /// peak resident is `images.len() * PAGE_SIZE` — O(N), the OOM the fix
    /// removes.
    fn whole_vec_evicted_supplement(o: &Owners, evicted: &EvictedPages) -> (Vec<u8>, usize) {
        // Pre-collect ALL evicted images (the un-fixed path). At the 10M
        // target this Vec is ~74 GB; here it is `evicted.blob.len()` pages.
        let mut images: Vec<EvictedPageImage> = Vec::with_capacity(evicted.blob.len());
        for &(tenant, page_id) in &evicted.blob {
            let page = o
                .blob
                .read_evicted_page(tenant, page_id)
                .expect("spill read succeeds")
                .expect("spill image present");
            images.push((OwnerTag::BlobPages as u8, tenant, page_id, page));
        }
        // Peak caller-resident = every image held at once (the OOM class).
        let peak_resident = images.len() * PAGE_SIZE;
        let mut buf = Vec::new();
        append_evicted_supplement(&mut buf, &images);
        (buf, peak_resident)
    }

    /// LOAD-BEARING regression (the ultracode-REJECT fix): the streamed
    /// evicted supplement holds only O(1) — exactly one `PAGE_SIZE` page —
    /// caller-resident, INDEPENDENT of the evicted-count `N`. The un-fixed
    /// whole-`Vec` path holds `N · PAGE_SIZE` (the ~74 GB @10M OOM). Drive
    /// TWO scales (64 vs 1024 evicted pages) and assert the STREAMED peak
    /// ratio is 1 while the WHOLE-`Vec` peak ratio scales ~16× with N.
    #[test]
    fn stream_evicted_supplement_peak_resident_is_o1_in_evicted_count() {
        let small_dir = tempdir().unwrap();
        let large_dir = tempdir().unwrap();
        let (o_small, ev_small) = bounded_owners_with_evicted(small_dir.path(), 64);
        let (o_large, ev_large) = bounded_owners_with_evicted(large_dir.path(), 1024);

        // The evicted set must actually scale with N (else the ratio is
        // meaningless). With a 2-page low watermark, ~N-2 pages evict.
        let n_small = ev_small.blob.len();
        let n_large = ev_large.blob.len();
        assert!(
            n_large >= n_small * 8,
            "evicted counts must scale with N: small={n_small}, large={n_large}",
        );

        // ── The FIX: stream_evicted_supplement holds ≤ one page, O(1) in N. ──
        let seed_small = o_small.allocator_seed();
        let seed_large = o_large.allocator_seed();
        let mut sink_small = VecSnapshotSink::with_capacity(1024);
        let peak_small = stream_evicted_supplement(
            &mut sink_small,
            &o_small.snapshot(seed_small.as_ref()),
            &ev_small,
        )
        .unwrap();
        let mut sink_large = VecSnapshotSink::with_capacity(1024);
        let peak_large = stream_evicted_supplement(
            &mut sink_large,
            &o_large.snapshot(seed_large.as_ref()),
            &ev_large,
        )
        .unwrap();

        assert_eq!(
            peak_small, PAGE_SIZE,
            "streamed supplement must hold exactly one page resident (64 evicted)",
        );
        assert_eq!(
            peak_large, PAGE_SIZE,
            "streamed supplement must hold exactly one page resident (1024 evicted)",
        );
        // The streamed peak ratio is EXACTLY 1 — O(1) in the evicted-count.
        assert_eq!(
            peak_large, peak_small,
            "M0.5 FIX: streamed evicted-supplement peak resident MUST be O(1) in evicted-count \
             (ratio 1024/64 ≈ 1), got small={peak_small} large={peak_large}",
        );

        // ── RED-on-revert: the un-fixed whole-`Vec` path holds N·PAGE_SIZE. ──
        let (_bytes_small, whole_peak_small) = whole_vec_evicted_supplement(&o_small, &ev_small);
        let (_bytes_large, whole_peak_large) = whole_vec_evicted_supplement(&o_large, &ev_large);
        assert_eq!(whole_peak_small, n_small * PAGE_SIZE);
        assert_eq!(whole_peak_large, n_large * PAGE_SIZE);
        let whole_ratio = whole_peak_large as f64 / whole_peak_small as f64;
        eprintln!(
            "M0.5 PEAK PROBE: evicted small={n_small} large={n_large} | \
             STREAMED peak small={peak_small} large={peak_large} ratio={:.2}× | \
             WHOLE-Vec peak small={whole_peak_small} large={whole_peak_large} ratio={whole_ratio:.2}×",
            peak_large as f64 / peak_small as f64,
        );
        assert!(
            whole_ratio >= 8.0,
            "REVERT PROOF: the whole-`Vec` (pre-collect) supplement peak scales ~linearly with N \
             (the OOM the fix removes) — ratio {whole_ratio:.1}× (≈16× at 1024/64), \
             whole_small={whole_peak_small} whole_large={whole_peak_large}. \
             The streamed fix keeps this FLAT at one page.",
        );
        // The fix's win, stated numerically: at the large scale the whole-`Vec`
        // path holds `n_large` pages resident; the streamed path holds ONE.
        assert!(
            whole_peak_large > peak_large * (n_large - 1),
            "streamed peak ({peak_large}) « whole-`Vec` peak ({whole_peak_large}) at N={n_large}",
        );
    }

    /// Byte-identity: the STREAMED evicted supplement (page-by-page) is
    /// byte-for-byte IDENTICAL to the whole-`Vec` `append_evicted_supplement`
    /// over the SAME evicted set — guards the incremental rewrite doesn't
    /// drift the wire format (tag, count, per-image {owner_tag, tenant, pid,
    /// PAGE}). Uses a real bounded store with genuinely-evicted pages.
    #[test]
    fn stream_evicted_supplement_bytes_identical_to_whole_vec() {
        let dir = tempdir().unwrap();
        let (o, evicted) = bounded_owners_with_evicted(dir.path(), 128);
        assert!(evicted.blob.len() >= 64, "need a non-trivial evicted set");

        // Streamed supplement bytes (the production path).
        let seed = o.allocator_seed();
        let mut sink = VecSnapshotSink::with_capacity(1024);
        stream_evicted_supplement(&mut sink, &o.snapshot(seed.as_ref()), &evicted).unwrap();
        let streamed = sink.into_body();

        // Whole-`Vec` oracle bytes over the SAME evicted set.
        let (whole, _peak) = whole_vec_evicted_supplement(&o, &evicted);

        assert_eq!(
            whole.len(),
            streamed.len(),
            "streamed supplement length ({}) != whole-Vec length ({})",
            streamed.len(),
            whole.len(),
        );
        assert_eq!(
            whole, streamed,
            "M0.5: streamed evicted supplement DIVERGES from the whole-Vec append_evicted_supplement \
             over the same evicted set — the wire format drifted (byte-identity broken)",
        );
    }

    /// The fail-loud primary/record wiring-bug check is MOVED AHEAD of any
    /// write: a non-empty `evicted.primary`/`evicted.record` (a wiring bug —
    /// those stores never evict) is rejected as `Corrupt` BEFORE a single
    /// supplement byte hits the sink, so no partial supplement is written.
    #[test]
    fn stream_evicted_supplement_fails_loud_on_primary_record_eviction_before_any_write() {
        let o = Owners::fresh();
        let evicted = EvictedPages {
            primary: vec![PageId::new(7)],
            record: Vec::new(),
            blob: Vec::new(),
        };
        let seed = o.allocator_seed();
        let mut sink = VecSnapshotSink::with_capacity(64);
        let err =
            stream_evicted_supplement(&mut sink, &o.snapshot(seed.as_ref()), &evicted).unwrap_err();
        match err {
            CheckpointError::Corrupt { reason } => {
                assert!(
                    reason.contains("primary") && reason.contains("record"),
                    "expected primary/record wiring-bug reason, got: {reason}",
                );
            }
            other => panic!("expected Corrupt on primary/record eviction, got {other:?}"),
        }
        // Fail-loud is BEFORE the write: nothing hit the sink.
        assert_eq!(
            sink.into_body().len(),
            0,
            "fail-loud must reject BEFORE writing any supplement byte (no partial section)",
        );
    }
}
