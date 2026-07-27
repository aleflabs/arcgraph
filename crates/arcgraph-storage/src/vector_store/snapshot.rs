//! Vector arena snapshot flush — M3.a Slice G.2 (ADR-035 §4.5/§4.6).
//!
//! Per ADR-035 §4.5/§4.6 and `docs/design/vector-storage-layout.md`
//! §10.3, vector arenas are persisted via a periodic full-arena
//! snapshot to disk in the **ARCV** file format. Slice G.2 lands the
//! flush primitive plus its atomic-write protocol; Slice G.3 (in
//! parallel on this branch) lands the corresponding load path.
//!
//! # ARCV file layout (Slice G.2 v1)
//!
//! ```text
//!   Header                  64 bytes (fixed; one cache line)
//!   Section descriptors     32 bytes × section_count
//!   Section payloads        Concatenated; each starts on a 64-byte boundary
//!   Footer                  16 bytes (last 16 bytes of file)
//! ```
//!
//! ## Header (64 bytes, offset 0)
//!
//! ```text
//!   Offset  Size  Field
//!     0       4   magic = b"ARCV"
//!     4       2   version: u16 LE = 1
//!     6       1   encoding: u8                (0=F32, 1=F16, 2=SQ8, 3=Binary)
//!     7       1   index_type: u8              (0=HNSW, 1=DiskANN)
//!     8       4   dim: u32 LE
//!    12       4   section_count: u32 LE
//!    16       8   lsn: u64 LE
//!    24       8   vectors_count: u64 LE
//!    32       8   tenant_id: u64 LE
//!    40       8   index_id: u64 LE
//!    48      16   reserved (must be zero)
//! ```
//!
//! ## Section descriptor (32 bytes, immediately after header)
//!
//! ```text
//!   Offset  Size  Field
//!     0       2   kind: u16 LE                (0=Quantized, 1=Rescore, 2=Labels)
//!     2       2   flags: u16 LE = 0
//!     4       4   reserved: u32 = 0
//!     8       8   payload_offset: u64 LE      Absolute file offset
//!    16       8   payload_size: u64 LE        Bytes
//!    24       8   reserved: u64 = 0
//! ```
//!
//! ## Footer (16 bytes; last 16 bytes of file)
//!
//! ```text
//!   Offset (file_size - 16):
//!     0       8   total_file_size: u64 LE
//!     8       4   reserved: u32 = 0
//!    12       4   crc32c: u32 LE              CRC over bytes 0..(file_size - 4)
//! ```
//!
//! # Atomic write protocol (per ADR-035 §10.3 steps 2–11)
//!
//! 1. Build the file body in memory.
//! 2. Compute trailing CRC32C over `bytes[0 .. file_size - 4]`.
//! 3. Open `arena-{tenant}-{index}-{lsn}.snap.tmp` with
//!    `O_CREAT | O_WRONLY | O_TRUNC` (overwrites a stale `.tmp`
//!    from a crashed prior flush; new flushes always start clean).
//! 4. `write_all(buf)` then `sync_all()` (durability barrier).
//! 5. `fs::rename(.tmp, .snap)` — atomic on POSIX.
//! 6. `fsync(snapshot_dir)` — makes the rename durable.
//! 7. Stamp the snapshot's `lsn` in the [`SnapshotCatalog`] so
//!    subsequent CommitBundle deltas stage only post-snapshot
//!    pages (per ADR-035 §4.5 high-water contract).
//!
//! Crash-injection points are exposed via [`flush_snapshot_with_crash_point`]
//! for Path A boundary tests; the production entry point
//! [`flush_snapshot`] never crash-injects.
//!
//! # Local-only hooks (ADR-035 §8)
//!
//! Snapshot file paths key on `(tenant, index, lsn)`.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use arcgraph_core::{Lsn, PartitionId, TenantId};
use dashmap::DashMap;

use super::VectorStoreError;

// ─── Format constants ────────────────────────────────────────────

/// ASCII magic bytes at file offset 0. Disjoint from
/// [`crate::wal::WAL_SEGMENT_MAGIC`] (`b"AGWL"`) and from the spill
/// magic (`b"ARCGSPIL"`).
pub const ARCV_MAGIC: &[u8; 4] = b"ARCV";

/// On-disk ARCV format version.
pub const ARCV_FORMAT_VERSION: u16 = 1;

/// Header size in bytes. Fixed; matches one cache line × 1 (64).
pub const ARCV_HEADER_SIZE: usize = 64;

/// Section descriptor size in bytes. 32 bytes × section_count
/// follow the header.
pub const ARCV_SECTION_DESCRIPTOR_SIZE: usize = 32;

/// Footer size in bytes. The last 16 bytes of every ARCV file.
pub const ARCV_FOOTER_SIZE: usize = 16;

/// Trailing CRC32C size (the last 4 bytes of the footer).
pub const ARCV_TRAILING_CRC_SIZE: usize = 4;

/// Section payloads start on a 64-byte boundary. Matches the
/// vector-section / graph-section alignment in
/// `vector-storage-layout.md` §3 — kept consistent across G.2's
/// simplified layout and the full §10.3 spec so future format
/// extensions don't break alignment guarantees.
pub const ARCV_PAYLOAD_ALIGNMENT: usize = 64;

/// Maximum encoding code (0=F32, 1=F16, 2=SQ8, 3=Binary, 4=RaBitQ).
/// Encoding 4 follows the arcgraph-vector tag space. At v1.0-alpha the SSD
/// sidecar is the only producer; ARCV/arena construction still rejects RaBitQ.
pub const ARCV_MAX_ENCODING: u8 = 4;

/// Maximum index_type code (0=HNSW, 1=DiskANN).
pub const ARCV_MAX_INDEX_TYPE: u8 = 1;

/// Maximum supported dimension. Matches pgvector's production cap
/// per `vector-storage-layout.md` §2; supports OpenAI text-3-large
/// (3072-dim) with headroom.
pub const ARCV_MAX_DIM: u32 = 4096;

// ─── Section kinds ───────────────────────────────────────────────

/// Section kind byte for ARCV snapshot section descriptors.
///
/// Slice G.2 ships three kinds because the v1.0 SQ8 + binary
/// rescore-aware arenas need to round-trip:
///
/// - [`SectionKind::Quantized`] — encoded vector bytes (SQ8 / F32 /
///   F16 / binary; one section per arena).
/// - [`SectionKind::Rescore`] — full-precision vectors held when
///   the arena is configured with rescore (per ADR-035 AC-1a:
///   SQ8 + F32 default, binary + F32 opt-in).
/// - [`SectionKind::Labels`] — per-vector payload labels for the
///   filter-aware HNSW / DiskANN indexes (Slices F.2 / F.3).
///
/// Future extensions (e.g., quantizer codebooks, tombstones) add a
/// new variant + reserved kind code; the section_count field in
/// the header bounds the descriptor table so adding a kind is a
/// strictly-additive change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum SectionKind {
    /// Encoded vector bytes (per arena's encoding).
    Quantized = 0,
    /// Full-precision rescore source (F32 by default per AC-1a).
    Rescore = 1,
    /// Per-vector payload labels for filter-aware search.
    Labels = 2,
}

impl SectionKind {
    /// Decode from on-disk u16 LE. Returns `None` on unknown codes
    /// so the load path (Slice G.3) can flag forward-version files
    /// without panicking.
    #[inline]
    #[must_use]
    pub const fn from_u16(raw: u16) -> Option<Self> {
        match raw {
            0 => Some(Self::Quantized),
            1 => Some(Self::Rescore),
            2 => Some(Self::Labels),
            _ => None,
        }
    }

    /// On-disk byte code.
    #[inline]
    #[must_use]
    pub const fn as_u16(self) -> u16 {
        self as u16
    }
}

// ─── Public spec types ───────────────────────────────────────────

/// One section's bytes, ready to flush. The slice is borrowed for
/// the duration of [`flush_snapshot`]; callers do not need to clone
/// — the function copies bytes into its working buffer in a single
/// pass before any I/O.
#[derive(Debug, Clone, Copy)]
pub struct SnapshotSection<'a> {
    /// What kind of bytes these are.
    pub kind: SectionKind,
    /// Raw payload. Length unrestricted by G.2 (the arena layer
    /// owns dim × vector_count math); the only constraint is that
    /// `payload_offset + payload_size` fits in u64.
    pub bytes: &'a [u8],
}

/// Self-describing snapshot input. All fields are pre-validated by
/// the F.1 / F.2 / F.3 arena layers before they reach G.2; the
/// flush path re-validates encoding / index_type / dim ranges so a
/// bad upstream cannot silently produce a malformed ARCV file.
#[derive(Debug, Clone, Copy)]
pub struct SnapshotSpec<'a> {
    /// Per-tenant arena identifier (ADR-011 + ADR-035 §7.5
    /// `(tenant, index)` keying).
    pub tenant: TenantId,
    /// Local partition sentinel. Must be [`PartitionId::ZERO`].
    pub partition: PartitionId,
    /// Per-tenant index id. Stamped into the file path AND header
    /// so the load path verifies the bytes match the catalog
    /// record they were attributed to.
    pub index_id: u64,
    /// Snapshot LSN — the highest commit_lsn included in this
    /// snapshot. Stamped into the path and header per ADR-035
    /// §4.6 step 5; the catalog's `last_snapshot_high_water`
    /// advances to this LSN on success.
    pub lsn: Lsn,
    /// Encoding code (0=F32, 1=F16, 2=SQ8, 3=Binary, 4=RaBitQ). Validated
    /// against [`ARCV_MAX_ENCODING`].
    pub encoding: u8,
    /// Index type code (0=HNSW, 1=DiskANN). Validated against
    /// [`ARCV_MAX_INDEX_TYPE`].
    pub index_type: u8,
    /// Embedding dimension. Validated against [`ARCV_MAX_DIM`].
    pub dim: u32,
    /// Number of vectors captured at snapshot time. Stamped into
    /// the header for the G.3 sanity-check (per ADR-035 §4.6
    /// step 4: `arena.vectors_count == graph.node_count`).
    pub vectors_count: u64,
    /// Sections to write (one descriptor + one payload per entry).
    /// Order is preserved on disk; G.3 reads in descriptor order.
    pub sections: &'a [SnapshotSection<'a>],
}

// ─── Crash injection (test-only path) ─────────────────────────────

/// Crash injection points exposed via [`flush_snapshot_with_crash_point`].
///
/// Not used by [`flush_snapshot`]; the production path drives a
/// `None` crash point so the I/O sequence runs end-to-end. The
/// test harness uses these to verify Slice G.2's atomic-write
/// protocol leaves a graceful artifact at every interior step:
/// either no `.snap` file (rename never ran) OR a `.snap` file
/// that round-trips header + footer CRC.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CrashPoint {
    /// Crash after temp-file creation, before any bytes are
    /// written. Leaves a 0-byte `.tmp` on disk.
    AfterTempCreate,
    /// Crash after writing `n` bytes of the body and before the
    /// remaining bytes / fsync. Leaves a truncated `.tmp` whose
    /// length < `total_file_size` — the trailing CRC field is
    /// missing or partial, so the file is unambiguously corrupt.
    MidWrite(usize),
    /// Crash after `write_all` + `sync_all`, before the rename.
    /// Leaves a complete, byte-valid `.tmp` but no `.snap`.
    BeforeRename,
    /// Crash after the rename, before the directory fsync that
    /// makes the rename durable. Leaves a `.snap` that may or may
    /// not survive a power loss; recovery treats this as a
    /// "completed snapshot, fsync uncertainty" case.
    BeforeDirFsync,
    /// Crash after directory fsync, before the catalog stamp.
    /// Leaves a durable `.snap` but the catalog still points at
    /// the previous snapshot's LSN — recovery picks the older
    /// snapshot AND replays post-old-snapshot WAL deltas (which
    /// is correct because the new `.snap` is byte-identical to
    /// what those deltas would re-build).
    BeforeCatalogStamp,
}

// ─── Catalog ─────────────────────────────────────────────────────

/// Per-arena snapshot LSN catalog.
///
/// Maps `(tenant, index_id)` to the LSN of the latest successfully
/// flushed snapshot. The flush path stamps the LSN here AFTER the
/// rename + directory fsync succeed, so a crash before the stamp
/// leaves the catalog pointing at the prior snapshot — recovery
/// then picks that older `.snap` AND replays post-older-LSN WAL
/// deltas. (The newly-flushed `.snap` exists on disk but is
/// "orphan-but-correct"; G.3's cold-start logic ignores .snap
/// files whose LSN exceeds the catalog stamp.)
///
/// At v1.0 this is a thin DashMap wrapper; future extensions (per-
/// partition keying, generation counts for GC) extend the value
/// shape, not the public API.
///
/// # Why DashMap (per ADR-035 §4.6 + the user spec)
///
/// The catalog is read on every commit (to compute
/// `last_snapshot_high_water` for the delta-staging contract) and
/// written once per snapshot flush (every N=10 000 commits or on
/// bulk-load completion). Reads dominate writes by 4+ orders of
/// magnitude; DashMap's sharded read-mostly profile matches that
/// access pattern without locking the whole map for the hot
/// commit-path probe.
#[derive(Debug, Default)]
pub struct SnapshotCatalog {
    inner: DashMap<(TenantId, u64), Lsn>,
}

impl SnapshotCatalog {
    /// Empty catalog. Used by tests that drive the flush primitive
    /// directly; production wiring (Slice F.* `VectorArenaRegistry`)
    /// constructs a catalog when the arena registry is created.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up the latest snapshot LSN for an arena. Returns
    /// `None` if no snapshot has been flushed yet (cold start;
    /// G.3 falls back to bootstrap-from-MVCC per ADR-035 §10.5).
    #[must_use]
    pub fn latest_lsn(&self, tenant: TenantId, index_id: u64) -> Option<Lsn> {
        self.inner.get(&(tenant, index_id)).map(|v| *v)
    }

    /// Stamp a snapshot LSN. Called by [`flush_snapshot`] after
    /// the rename + directory fsync succeed. Idempotent at the
    /// `(tenant, index, lsn)` triple level; calling with a lower
    /// LSN than the current stamp is a no-op (the older value
    /// wins) — protects against stale-flush-completion races.
    pub fn stamp(&self, tenant: TenantId, index_id: u64, lsn: Lsn) {
        self.inner
            .entry((tenant, index_id))
            .and_modify(|cur| {
                if lsn > *cur {
                    *cur = lsn;
                }
            })
            .or_insert(lsn);
    }

    /// Number of arenas with at least one stamped snapshot. For
    /// metrics / diagnostics only.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the catalog is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

// ─── Snapshot policy ─────────────────────────────────────────────

/// Snapshot trigger predicate per ADR-035 §4.6.
///
/// The flush primitive [`flush_snapshot`] is called *when* the
/// policy fires; the policy itself is evaluated by the F.* arena
/// layer at transaction-commit time. Slice G.2 ships the predicate
/// alongside the flush so the cadence rules are colocated with the
/// thing they govern.
///
/// # Three triggers (per ADR-035 §4.6)
///
/// 1. **Periodic**: snapshot every `periodic_commits` commits since
///    the last snapshot (default N=10 000).
/// 2. **Bulk-load completion** (per OQ-V3 resolution): a single
///    transaction with ≥ `bulk_load_threshold` vector inserts
///    forces a snapshot at txn end (default N=1 000).
/// 3. **Schema change**: quantizer params changed (e.g., SQ8
///    retraining via `REINDEX <index>`). Always fires.
#[derive(Debug, Clone, Copy)]
pub struct SnapshotPolicy {
    /// Periodic-trigger threshold. Default 10 000 per ADR-035 §4.6.
    pub periodic_commits: u64,
    /// Bulk-load-trigger threshold. Default 1 000 per OQ-V3.
    pub bulk_load_threshold: usize,
}

impl Default for SnapshotPolicy {
    fn default() -> Self {
        Self {
            periodic_commits: 10_000,
            bulk_load_threshold: 1_000,
        }
    }
}

/// Which trigger fired (returned from [`SnapshotPolicy::should_snapshot`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotTrigger {
    /// Periodic threshold reached.
    Periodic,
    /// Single-txn vector inserts crossed the bulk-load threshold.
    BulkLoad,
    /// Quantizer params changed; rebuild + snapshot atomically.
    SchemaChange,
}

impl SnapshotPolicy {
    /// Decide whether to fire a snapshot at the end of a
    /// transaction. Inputs:
    ///
    /// - `commits_since_last`: commits committed since the last
    ///   snapshot for this arena.
    /// - `txn_inserts`: vector inserts in the just-committed
    ///   transaction.
    /// - `schema_changed`: quantizer params changed in this
    ///   transaction.
    ///
    /// Returns `Some(trigger)` to fire; `None` to defer. Trigger
    /// precedence: schema → bulk-load → periodic, matching
    /// ADR-035 §4.6's "force snapshot at txn end" bulk-load
    /// override of the steady-state cadence.
    #[must_use]
    pub fn should_snapshot(
        &self,
        commits_since_last: u64,
        txn_inserts: usize,
        schema_changed: bool,
    ) -> Option<SnapshotTrigger> {
        if schema_changed {
            return Some(SnapshotTrigger::SchemaChange);
        }
        if txn_inserts >= self.bulk_load_threshold {
            return Some(SnapshotTrigger::BulkLoad);
        }
        if commits_since_last >= self.periodic_commits {
            return Some(SnapshotTrigger::Periodic);
        }
        None
    }
}

// ─── Path helpers (local-only keying) ───────────────────

/// Build the path to a snapshot file:
/// `{dir}/arena-{tenant}-{index}-{lsn}.snap`.
#[must_use]
pub fn snapshot_path(snapshot_dir: &Path, tenant: TenantId, index_id: u64, lsn: Lsn) -> PathBuf {
    snapshot_dir.join(format!(
        "arena-{}-{}-{}.snap",
        tenant.raw(),
        index_id,
        lsn.raw()
    ))
}

/// Companion temp path used during the atomic write.
#[must_use]
pub fn snapshot_temp_path(
    snapshot_dir: &Path,
    tenant: TenantId,
    index_id: u64,
    lsn: Lsn,
) -> PathBuf {
    snapshot_dir.join(format!(
        "arena-{}-{}-{}.snap.tmp",
        tenant.raw(),
        index_id,
        lsn.raw()
    ))
}

// ─── Buffer assembly ─────────────────────────────────────────────

/// Plan: where each section's payload lives in the final file.
#[derive(Debug)]
struct SectionPlan {
    payload_offset: u64,
    payload_size: u64,
}

/// Build the full ARCV file in memory (header + descriptors +
/// payloads + footer with trailing CRC). Returns the buffer ready
/// to flush.
///
/// The body is built in-memory before any I/O so the trailing CRC
/// is computed once over a consistent buffer; the same pattern the
/// WAL spill writer uses (`spill.rs::write_spill_batch`) — keeps
/// the disk-side artifact byte-identical regardless of how many
/// `write` syscalls fragment it.
fn build_arcv_buffer(spec: &SnapshotSpec<'_>) -> Result<Vec<u8>, VectorStoreError> {
    // 1. Validate.
    if spec.partition != PartitionId::ZERO {
        return Err(VectorStoreError::InvalidSnapshotSpec(
            "partition must be PartitionId::ZERO".to_owned(),
        ));
    }
    if spec.encoding > ARCV_MAX_ENCODING {
        return Err(VectorStoreError::InvalidSnapshotSpec(format!(
            "encoding={} exceeds max {}",
            spec.encoding, ARCV_MAX_ENCODING
        )));
    }
    if spec.index_type > ARCV_MAX_INDEX_TYPE {
        return Err(VectorStoreError::InvalidSnapshotSpec(format!(
            "index_type={} exceeds max {}",
            spec.index_type, ARCV_MAX_INDEX_TYPE
        )));
    }
    if spec.dim == 0 || spec.dim > ARCV_MAX_DIM {
        return Err(VectorStoreError::InvalidSnapshotSpec(format!(
            "dim={} not in 1..={}",
            spec.dim, ARCV_MAX_DIM
        )));
    }
    if spec.sections.len() > u32::MAX as usize {
        return Err(VectorStoreError::InvalidSnapshotSpec(format!(
            "section_count={} exceeds u32::MAX",
            spec.sections.len()
        )));
    }

    // 2. Compute layout: where each payload lives.
    let section_count = spec.sections.len();
    let descriptor_block_size = ARCV_SECTION_DESCRIPTOR_SIZE * section_count;
    let mut cursor = (ARCV_HEADER_SIZE + descriptor_block_size) as u64;
    let mut plans: Vec<SectionPlan> = Vec::with_capacity(section_count);
    for section in spec.sections {
        cursor = align_up_u64(cursor, ARCV_PAYLOAD_ALIGNMENT as u64);
        let size = section.bytes.len() as u64;
        plans.push(SectionPlan {
            payload_offset: cursor,
            payload_size: size,
        });
        cursor = cursor.saturating_add(size);
    }
    // Footer goes at the very end; no alignment requirement.
    let footer_offset = cursor;
    let total_file_size = footer_offset + ARCV_FOOTER_SIZE as u64;

    // 3. Allocate buffer.
    let total_size = usize::try_from(total_file_size).map_err(|_| {
        VectorStoreError::InvalidSnapshotSpec(format!(
            "total_file_size={total_file_size} overflows usize"
        ))
    })?;
    let mut buf = vec![0u8; total_size];

    // 4. Write header at offset 0.
    write_header(&mut buf[..ARCV_HEADER_SIZE], spec);

    // 5. Write descriptor table at offset ARCV_HEADER_SIZE.
    for (i, (section, plan)) in spec.sections.iter().zip(plans.iter()).enumerate() {
        let off = ARCV_HEADER_SIZE + i * ARCV_SECTION_DESCRIPTOR_SIZE;
        write_section_descriptor(
            &mut buf[off..off + ARCV_SECTION_DESCRIPTOR_SIZE],
            section.kind,
            plan,
        );
    }

    // 6. Write payloads.
    for (section, plan) in spec.sections.iter().zip(plans.iter()) {
        let off = plan.payload_offset as usize;
        let end = off + plan.payload_size as usize;
        buf[off..end].copy_from_slice(section.bytes);
    }

    // 7. Write footer prefix (total_file_size + reserved). The CRC
    //    field is filled in step 8.
    let footer_off = footer_offset as usize;
    buf[footer_off..footer_off + 8].copy_from_slice(&total_file_size.to_le_bytes());
    // bytes [footer_off + 8 .. footer_off + 12] already zero; reserved.

    // 8. Compute CRC32C over bytes [0 .. total_size - ARCV_TRAILING_CRC_SIZE]
    //    and write into the trailing 4 bytes.
    let crc = crc32c::crc32c(&buf[..total_size - ARCV_TRAILING_CRC_SIZE]);
    let crc_off = total_size - ARCV_TRAILING_CRC_SIZE;
    buf[crc_off..crc_off + 4].copy_from_slice(&crc.to_le_bytes());

    Ok(buf)
}

#[inline]
fn align_up_u64(value: u64, alignment: u64) -> u64 {
    debug_assert!(alignment.is_power_of_two());
    (value + alignment - 1) & !(alignment - 1)
}

fn write_header(buf: &mut [u8], spec: &SnapshotSpec<'_>) {
    debug_assert_eq!(buf.len(), ARCV_HEADER_SIZE);
    buf[0..4].copy_from_slice(ARCV_MAGIC);
    buf[4..6].copy_from_slice(&ARCV_FORMAT_VERSION.to_le_bytes());
    buf[6] = spec.encoding;
    buf[7] = spec.index_type;
    buf[8..12].copy_from_slice(&spec.dim.to_le_bytes());
    let section_count = spec.sections.len() as u32;
    buf[12..16].copy_from_slice(&section_count.to_le_bytes());
    buf[16..24].copy_from_slice(&spec.lsn.raw().to_le_bytes());
    buf[24..32].copy_from_slice(&spec.vectors_count.to_le_bytes());
    buf[32..40].copy_from_slice(&spec.tenant.raw().to_le_bytes());
    buf[40..48].copy_from_slice(&spec.index_id.to_le_bytes());
    // bytes 48..64 = reserved (kept zeroed by the buffer-fill init).
}

fn write_section_descriptor(buf: &mut [u8], kind: SectionKind, plan: &SectionPlan) {
    debug_assert_eq!(buf.len(), ARCV_SECTION_DESCRIPTOR_SIZE);
    buf[0..2].copy_from_slice(&kind.as_u16().to_le_bytes());
    // bytes 2..4 = flags (zeroed)
    // bytes 4..8 = reserved (zeroed)
    buf[8..16].copy_from_slice(&plan.payload_offset.to_le_bytes());
    buf[16..24].copy_from_slice(&plan.payload_size.to_le_bytes());
    // bytes 24..32 = reserved (zeroed)
}

// ─── Public flush API ────────────────────────────────────────────

/// Flush an arena snapshot to disk.
///
/// See the module-level docs for the file format and the atomic-
/// write protocol (steps 1–7). On success the snapshot's LSN is
/// stamped in `catalog`; the returned `PathBuf` is the final
/// `.snap` path on disk.
///
/// # Errors
///
/// - [`VectorStoreError::InvalidSnapshotSpec`] if encoding /
///   index_type / dim are out of range, or if the byte layout
///   would overflow `usize`.
/// - [`VectorStoreError::SnapshotIo`] for any underlying I/O
///   failure (write / fsync / rename / dir-fsync). The atomic-
///   write protocol guarantees a graceful artifact at every
///   interior failure point: either no `.snap` file (rename never
///   ran) OR a `.snap` whose header + footer round-trip cleanly.
pub fn flush_snapshot(
    spec: &SnapshotSpec<'_>,
    snapshot_dir: &Path,
    catalog: &SnapshotCatalog,
) -> Result<PathBuf, VectorStoreError> {
    flush_snapshot_inner(spec, snapshot_dir, catalog, None)
}

/// Crash-injecting flush variant. Production callers use
/// [`flush_snapshot`]; this entry point exists so Path A boundary
/// tests can verify the on-disk artifact at every interior step
/// of the atomic-write protocol without resorting to actual
/// process kills.
pub fn flush_snapshot_with_crash_point(
    spec: &SnapshotSpec<'_>,
    snapshot_dir: &Path,
    catalog: &SnapshotCatalog,
    crash_at: CrashPoint,
) -> Result<PathBuf, VectorStoreError> {
    flush_snapshot_inner(spec, snapshot_dir, catalog, Some(crash_at))
}

fn flush_snapshot_inner(
    spec: &SnapshotSpec<'_>,
    snapshot_dir: &Path,
    catalog: &SnapshotCatalog,
    crash_at: Option<CrashPoint>,
) -> Result<PathBuf, VectorStoreError> {
    // Step 0: ensure the directory exists. Inexpensive; matches
    // `wal/spill.rs::write_spill_batch` defensiveness.
    fs::create_dir_all(snapshot_dir).map_err(|e| {
        VectorStoreError::SnapshotIo(format!(
            "create_dir_all({}) failed: {e}",
            snapshot_dir.display()
        ))
    })?;

    // Step 1+2: build the body in memory (validates spec).
    let buf = build_arcv_buffer(spec)?;

    let tmp_path = snapshot_temp_path(snapshot_dir, spec.tenant, spec.index_id, spec.lsn);
    let final_path = snapshot_path(snapshot_dir, spec.tenant, spec.index_id, spec.lsn);

    // Step 3: open .tmp with O_CREAT | O_WRONLY | O_TRUNC. Per
    // ADR-035 §10.3 step 2 — overwrites any stale .tmp from a
    // crashed prior flush so the new flush always starts clean.
    let mut tmp_file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&tmp_path)
        .map_err(|e| {
            VectorStoreError::SnapshotIo(format!("open({}) failed: {e}", tmp_path.display()))
        })?;

    if matches!(crash_at, Some(CrashPoint::AfterTempCreate)) {
        return Err(VectorStoreError::CrashInjected(CrashPoint::AfterTempCreate));
    }

    // Step 4a: write_all (or partial write under MidWrite injection).
    if let Some(CrashPoint::MidWrite(n)) = crash_at {
        let n = n.min(buf.len());
        tmp_file.write_all(&buf[..n]).map_err(|e| {
            VectorStoreError::SnapshotIo(format!(
                "partial write_all({}) failed: {e}",
                tmp_path.display()
            ))
        })?;
        // Note: do NOT sync_all here — the MidWrite scenario is
        // "kernel cache holds bytes; process dies". The partial
        // .tmp is what we want on disk for the test.
        return Err(VectorStoreError::CrashInjected(CrashPoint::MidWrite(n)));
    }
    tmp_file.write_all(&buf).map_err(|e| {
        VectorStoreError::SnapshotIo(format!("write_all({}) failed: {e}", tmp_path.display()))
    })?;

    // Step 4b: fsync the temp file. After this returns, the body
    // is durable — a crash now leaves a complete byte-valid .tmp.
    tmp_file.sync_all().map_err(|e| {
        VectorStoreError::SnapshotIo(format!("sync_all({}) failed: {e}", tmp_path.display()))
    })?;

    if matches!(crash_at, Some(CrashPoint::BeforeRename)) {
        return Err(VectorStoreError::CrashInjected(CrashPoint::BeforeRename));
    }

    // Step 5: atomic rename. POSIX rename(2) guarantees that any
    // observer either sees the old `.snap` (if one existed; G.2's
    // first flush has none) or the new one — never a half-renamed
    // path. On no prior `.snap`: observer either sees no file or
    // the new one.
    fs::rename(&tmp_path, &final_path).map_err(|e| {
        VectorStoreError::SnapshotIo(format!(
            "rename({} -> {}) failed: {e}",
            tmp_path.display(),
            final_path.display()
        ))
    })?;

    if matches!(crash_at, Some(CrashPoint::BeforeDirFsync)) {
        return Err(VectorStoreError::CrashInjected(CrashPoint::BeforeDirFsync));
    }

    // Step 6: fsync the directory so the rename is durable. POSIX
    // requires a directory fsync to make a name-change durable —
    // without it, a crash post-rename can revert to the pre-rename
    // dirent. Best-effort because some filesystems (notably tmpfs
    // and certain network mounts) reject directory fsync; the WAL
    // spill writer uses the same best-effort approach.
    if let Ok(dir) = File::open(snapshot_dir) {
        let _ = dir.sync_all();
    }

    if matches!(crash_at, Some(CrashPoint::BeforeCatalogStamp)) {
        return Err(VectorStoreError::CrashInjected(
            CrashPoint::BeforeCatalogStamp,
        ));
    }

    // Step 7: stamp the catalog. Per ADR-035 §4.5 high-water
    // contract: future commits delta-stage only post-snapshot
    // pages because next_vector_id has advanced past the stamped
    // LSN's recorded `last_snapshot_high_water` (the F.* layer
    // stamps both pieces; G.2 owns only the LSN field).
    catalog.stamp(spec.tenant, spec.index_id, spec.lsn);

    Ok(final_path)
}

// ─── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn sample_spec<'a>(sections: &'a [SnapshotSection<'a>]) -> SnapshotSpec<'a> {
        SnapshotSpec {
            tenant: TenantId::DEFAULT,
            partition: PartitionId::ZERO,
            index_id: 7,
            lsn: Lsn::new(42),
            encoding: 2,   // SQ8
            index_type: 0, // HNSW
            dim: 768,
            vectors_count: 0,
            sections,
        }
    }

    #[test]
    fn build_arcv_buffer_minimal_zero_sections_roundtrips() {
        let spec = sample_spec(&[]);
        let buf = build_arcv_buffer(&spec).unwrap();
        // Header (64) + footer (16) only.
        assert_eq!(buf.len(), ARCV_HEADER_SIZE + ARCV_FOOTER_SIZE);
        // Magic
        assert_eq!(&buf[0..4], ARCV_MAGIC);
        // Version
        assert_eq!(
            u16::from_le_bytes(buf[4..6].try_into().unwrap()),
            ARCV_FORMAT_VERSION
        );
        // section_count = 0
        assert_eq!(u32::from_le_bytes(buf[12..16].try_into().unwrap()), 0);
        // CRC verifies.
        let stored_crc = u32::from_le_bytes(buf[buf.len() - 4..].try_into().unwrap());
        let computed = crc32c::crc32c(&buf[..buf.len() - 4]);
        assert_eq!(stored_crc, computed);
    }

    #[test]
    fn build_arcv_buffer_rejects_bad_encoding() {
        let mut spec = sample_spec(&[]);
        spec.encoding = 99;
        let err = build_arcv_buffer(&spec).unwrap_err();
        assert!(matches!(err, VectorStoreError::InvalidSnapshotSpec(_)));
    }

    #[test]
    fn build_arcv_buffer_rejects_non_local_partition() {
        let mut spec = sample_spec(&[]);
        spec.partition = PartitionId::new(1);
        let err = build_arcv_buffer(&spec).unwrap_err();
        assert!(matches!(err, VectorStoreError::InvalidSnapshotSpec(_)));
    }

    #[test]
    fn build_arcv_buffer_rejects_zero_dim() {
        let mut spec = sample_spec(&[]);
        spec.dim = 0;
        let err = build_arcv_buffer(&spec).unwrap_err();
        assert!(matches!(err, VectorStoreError::InvalidSnapshotSpec(_)));
    }

    #[test]
    fn build_arcv_buffer_rejects_huge_dim() {
        let mut spec = sample_spec(&[]);
        spec.dim = ARCV_MAX_DIM + 1;
        let err = build_arcv_buffer(&spec).unwrap_err();
        assert!(matches!(err, VectorStoreError::InvalidSnapshotSpec(_)));
    }

    #[test]
    fn align_up_aligns_correctly() {
        assert_eq!(align_up_u64(0, 64), 0);
        assert_eq!(align_up_u64(1, 64), 64);
        assert_eq!(align_up_u64(64, 64), 64);
        assert_eq!(align_up_u64(65, 64), 128);
        assert_eq!(align_up_u64(127, 64), 128);
        assert_eq!(align_up_u64(128, 64), 128);
    }

    #[test]
    fn snapshot_path_uses_local_keying() {
        let dir = Path::new("/var/arcgraph/vectors");
        let path = snapshot_path(dir, TenantId::new(7), 9, Lsn::new(123));
        assert_eq!(path, Path::new("/var/arcgraph/vectors/arena-7-9-123.snap"));
    }

    #[test]
    fn flush_writes_header_and_footer() {
        let tmpdir = TempDir::new().unwrap();
        let catalog = SnapshotCatalog::new();
        let payload = vec![0u8; 32];
        let sections = [SnapshotSection {
            kind: SectionKind::Quantized,
            bytes: &payload,
        }];
        let spec = sample_spec(&sections);
        let path = flush_snapshot(&spec, tmpdir.path(), &catalog).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[0..4], ARCV_MAGIC);
        // Verify catalog stamped.
        assert_eq!(catalog.latest_lsn(TenantId::DEFAULT, 7), Some(Lsn::new(42)));
    }

    #[test]
    fn snapshot_policy_default_thresholds() {
        let policy = SnapshotPolicy::default();
        assert_eq!(policy.periodic_commits, 10_000);
        assert_eq!(policy.bulk_load_threshold, 1_000);
    }

    #[test]
    fn snapshot_policy_schema_change_wins() {
        let policy = SnapshotPolicy::default();
        // Even with low commits and no bulk-load, schema change fires.
        assert_eq!(
            policy.should_snapshot(0, 0, true),
            Some(SnapshotTrigger::SchemaChange)
        );
    }

    #[test]
    fn snapshot_policy_bulk_load_beats_periodic() {
        let policy = SnapshotPolicy::default();
        assert_eq!(
            policy.should_snapshot(50_000, 1_500, false),
            Some(SnapshotTrigger::BulkLoad)
        );
    }

    #[test]
    fn snapshot_policy_periodic_at_threshold() {
        let policy = SnapshotPolicy::default();
        assert_eq!(
            policy.should_snapshot(10_000, 10, false),
            Some(SnapshotTrigger::Periodic)
        );
    }

    #[test]
    fn snapshot_policy_no_trigger() {
        let policy = SnapshotPolicy::default();
        assert_eq!(policy.should_snapshot(100, 10, false), None);
    }

    #[test]
    fn catalog_stamp_keeps_higher_lsn() {
        let cat = SnapshotCatalog::new();
        cat.stamp(TenantId::DEFAULT, 1, Lsn::new(10));
        cat.stamp(TenantId::DEFAULT, 1, Lsn::new(5));
        assert_eq!(cat.latest_lsn(TenantId::DEFAULT, 1), Some(Lsn::new(10)));
        cat.stamp(TenantId::DEFAULT, 1, Lsn::new(20));
        assert_eq!(cat.latest_lsn(TenantId::DEFAULT, 1), Some(Lsn::new(20)));
    }

    #[test]
    fn section_kind_roundtrips() {
        for k in [
            SectionKind::Quantized,
            SectionKind::Rescore,
            SectionKind::Labels,
        ] {
            assert_eq!(SectionKind::from_u16(k.as_u16()), Some(k));
        }
        assert_eq!(SectionKind::from_u16(99), None);
    }
}
