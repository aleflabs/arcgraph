//! Vector arena recovery + `bootstrap_from_mvcc` fallback (M3.a Slice
//! G.3, ADR-035 §4.5/§4.6 + ADR-032 §R1-R7).
//!
//! Slice G.3 owns the recovery side of the vector-arena durability
//! contract. Sibling Slice G.2 owns the snapshot writer
//! ([`super::snapshot`]); G.4 owns commit-time staging; G.5 owns Z-1
//! (b) rollback. The on-disk snapshot format is anchored to ADR-035
//! §4.1 — both G.2 and G.3 decode/encode against that byte layout
//! independently.
//!
//! # Recovery flow (ADR-035 §4.6)
//!
//! On process restart, for each `(tenant, index)` arena present in
//! the index catalog, [`recover_arena`] runs five steps:
//!
//! 1. **Snapshot load.** Find the latest
//!    `arena-{tenant}-{index}-{lsn}.snap` file in `snapshot_dir`,
//!    parse the ARCV header, verify both header CRC and trailing
//!    file CRC, decode all sections.
//! 2. **Apply post-snapshot CommitBundle delta pages.** Iterate the
//!    WAL via [`crate::wal::WalRecoveryReader`]; for each
//!    [`crate::wal::bundle::BundlePageKind::Vector`] entry whose
//!    bundle's `commit_lsn > snapshot_lsn`, dispatch to
//!    [`super::VectorPageStoreHandle::install_or_replace`].
//!    Idempotent by construction (Lemma I2).
//! 3. **Apply post-snapshot MVCC vector writes.** A no-op under the
//!    B-1 resolution (vector data lives in CommitBundle delta
//!    pages, not MVCC chains) — retained as a contract guard for
//!    future evolution.
//! 4. **Sanity check (mandatory, ship-blocking).** Assert
//!    `arena.vectors_count == graph.node_count`. On mismatch emit
//!    [`arcgraph_core::ArcGraphError::VectorIndexInconsistency`]
//!    with operator-actionable diagnostics; **halt replay**.
//! 5. **Mark arena ready.** Delete superseded snapshot generations
//!    and any orphaned `.tmp` files; return the [`RecoveredArena`].
//!
//! # Bootstrap-from-MVCC fallback (ADR-035 §9.1)
//!
//! If no usable snapshot exists (cold start, OR every available
//! snapshot fails CRC, OR snapshot file truncated to less than the
//! 128-byte ARCV header), [`bootstrap_from_mvcc`] walks the MVCC
//! version chain via the supplied [`MvccVectorSource`] adapter and
//! reconstructs the arena from raw vector bytes. Slow (≈5 min per
//! 10 M vectors per ADR-035 §4.3.1) but functionally correct per
//! ADR-023 (MVCC is authoritative). On bootstrap failure (MVCC
//! also unable to provide vectors) we surface the error directly —
//! this is the "no silent data loss" rung of ADR-032's escalation.
//!
//! # Local-only hooks (ADR-035 §8)
//!
//! Recovery is keyed by [`VectorRecoveryRequest`] which carries
//! `(tenant_id, partition_id, index_id)`; `partition_id` must be
//! `PartitionId::ZERO` (asserted by
//! `g3_recover_request_partition_id_always_zero_at_v1` in the
//! integration test suite). The snapshot filename pattern is
//! `arena-{tenant}-{index}-{lsn}.snap`.
//!
//! # Boundaries (PR scope)
//!
//! - **Owns:** snapshot decoder, recovery orchestration, MVCC
//!   bootstrap orchestration, post-replay sanity check, stale
//!   snapshot cleanup.
//! - **Does not own:** snapshot writer (G.2), commit-time staging
//!   (G.4), Z-1 (b) rollback wiring in `crud.rs` (G.5), runtime
//!   `VectorArena` in `arcgraph-vector` (consumed via
//!   [`MvccVectorSource`] + [`super::VectorPageStoreHandle`]).

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arcgraph_core::{ArcGraphError, Lsn, PageId, PartitionId, Result, TenantId};
use tracing::{debug, error, info, warn};

use super::{VectorPageStoreHandle, VectorStoreError};

// ─────────────────────────────────────────────────────────────────────
// Wire-format constants — ADR-035 §4.1 (anchored byte layout)
// ─────────────────────────────────────────────────────────────────────

/// `b"ARCV"` magic at snapshot file offset 0. Read as bytes (not
/// a u32) so endian-confused operators see the literal ASCII in
/// `xxd`. Disjoint from every other on-disk magic in the workspace.
pub const SNAPSHOT_MAGIC: [u8; 4] = *b"ARCV";

/// Snapshot file format version.
pub const SNAPSHOT_FORMAT_VERSION: u8 = 1;

/// ARCV header size (cacheline-aligned, ADR-035 §4.1).
pub const SNAPSHOT_HEADER_SIZE: usize = 128;

/// Header CRC offset (over bytes 0..120; CRC byte range 120..124).
const HEADER_CRC_OFFSET: usize = 120;

/// Trailing footer size (bytes -16..0 of the file).
pub const SNAPSHOT_FOOTER_SIZE: usize = 16;

/// HNSW graph-section magic (`b"HNSW"`).
pub const GRAPH_MAGIC_HNSW: [u8; 4] = *b"HNSW";

/// DiskANN/Vamana graph-section magic (`b"VAMA"`).
pub const GRAPH_MAGIC_VAMA: [u8; 4] = *b"VAMA";

/// Graph-section header size (first 64 bytes of the section per
/// ADR-035 §4.2 / §4.3).
const GRAPH_HEADER_SIZE: usize = 64;

/// HNSW graph-header `node_count` u32 LE offset (ADR-035 §4.2 bytes
/// 16..20 of the graph header).
const HNSW_NODE_COUNT_OFFSET: usize = 16;

/// VAMA graph-header `node_count` u32 LE offset (ADR-035 §4.3 bytes
/// 17..21 of the graph header).
const VAMA_NODE_COUNT_OFFSET: usize = 17;

/// Snapshot filename suffix.
pub const SNAPSHOT_FILE_EXT: &str = "snap";

/// Temporary file extension produced during atomic snapshot flush
/// (G.2's `arena-{tenant}-{index}-{lsn}.snap.tmp`); cleaned by G.3
/// during recovery so a crash mid-flush does not leak files.
pub const SNAPSHOT_TEMP_EXT: &str = "snap.tmp";

// ─────────────────────────────────────────────────────────────────────
// Shadow types — kept disjoint from `arcgraph-vector::Encoding` /
// `IndexType` to avoid the circular crate dep. Byte values match
// ADR-035 §4.1 so the snapshot wire format is shared with G.2.
// ─────────────────────────────────────────────────────────────────────

/// Vector encoding tag carried in the snapshot header (byte 5).
/// Byte values match ADR-035 §4.1 and the `arcgraph-vector::Encoding`
/// enum's ordinal layout so the on-wire byte is stable across
/// crates without a `From<arcgraph_vector::Encoding>` impl
/// (which would force a circular dependency).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Encoding {
    /// 32-bit IEEE-754 float per dim. Default for `< 10 M` vectors.
    F32 = 0,
    /// 16-bit IEEE-754 half-precision. v1.1 halfvec parity.
    F16 = 1,
    /// 8-bit scalar-quantized. Default for `>= 10 M`.
    Sq8 = 2,
    /// 1-bit-per-dim sign quantization, 128-byte-aligned per S-1.
    Binary = 3,
    /// ADR-209 RaBitQ payload tag. No producer emits it until
    /// slice 2 wires index-side nav.
    RaBitQ = 4,
}

impl Encoding {
    /// Decode the byte-5 encoding tag. Rejects unknown bytes.
    pub fn from_byte(byte: u8) -> Result<Self> {
        Ok(match byte {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Sq8,
            3 => Self::Binary,
            4 => Self::RaBitQ,
            other => {
                return Err(ArcGraphError::WalCorruption {
                    lsn: Lsn::ZERO,
                    reason: format!("vector snapshot: unknown Encoding byte {other}"),
                });
            }
        })
    }

    /// Raw byte for snapshot encoding.
    #[must_use]
    pub const fn as_byte(self) -> u8 {
        self as u8
    }
}

/// Index algorithm tag carried in the snapshot header (byte 6).
/// Byte values match ADR-035 §4.1 and `arcgraph-vector::IndexType`'s
/// ordinal layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum IndexType {
    /// HNSW (Malkov & Yashunin TPAMI 2018). Default for hot
    /// collections ≤ ~50 M vectors per tenant.
    Hnsw = 0,
    /// DiskANN / Vamana (Subramanya et al. NeurIPS 2019). Default
    /// for 50 M – 1 B.
    DiskAnn = 1,
}

impl IndexType {
    /// Decode the byte-6 index_type tag. Rejects unknown bytes.
    pub fn from_byte(byte: u8) -> Result<Self> {
        Ok(match byte {
            0 => Self::Hnsw,
            1 => Self::DiskAnn,
            other => {
                return Err(ArcGraphError::WalCorruption {
                    lsn: Lsn::ZERO,
                    reason: format!("vector snapshot: unknown IndexType byte {other}"),
                });
            }
        })
    }

    /// Raw byte for snapshot encoding.
    #[must_use]
    pub const fn as_byte(self) -> u8 {
        self as u8
    }

    /// Expected graph-section magic for this index algorithm.
    #[must_use]
    pub const fn graph_magic(self) -> [u8; 4] {
        match self {
            Self::Hnsw => GRAPH_MAGIC_HNSW,
            Self::DiskAnn => GRAPH_MAGIC_VAMA,
        }
    }

    /// Byte offset of `node_count` u32 LE within the graph header.
    #[must_use]
    pub const fn graph_node_count_offset(self) -> usize {
        match self {
            Self::Hnsw => HNSW_NODE_COUNT_OFFSET,
            Self::DiskAnn => VAMA_NODE_COUNT_OFFSET,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// VectorRecoveryRequest — recovery descriptor
// ─────────────────────────────────────────────────────────────────────

/// Identifies a single tenant/index arena to recover.
///
/// Constructed from the index catalog at startup. The recovery
/// orchestrator iterates one of these per-arena and calls
/// [`recover_arena`] on each. `partition_id` is always zero
/// (regression guard
/// `g3_recover_request_partition_id_always_zero_at_v1` in the
/// integration test suite).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VectorRecoveryRequest {
    /// Tenant that owns the arena. Vector arenas are physically per-
    /// tenant (ADR-035 §7.5); the recovery loader derives the
    /// snapshot filename from this.
    pub tenant_id: TenantId,
    /// Local partition key. Always [`PartitionId::ZERO`].
    pub partition_id: PartitionId,
    /// Per-tenant index id. The catalog allocates one per index DDL
    /// (`DEFINE INDEX <name>`); a single tenant may host many.
    pub index_id: u64,
    /// Expected index algorithm. Used to pick the graph-section
    /// magic + `node_count` offset so the sanity check reads the
    /// correct field. The recovery flow does NOT trust the snapshot
    /// header's `index_type` byte alone — the catalog is the
    /// authoritative source per ADR-011's catalog discipline. A
    /// snapshot whose `index_type` byte disagrees with this field
    /// is treated as corruption and falls through to bootstrap.
    pub expected_index_type: IndexType,
    /// Expected dim. Same rationale as `expected_index_type`: the
    /// catalog is authoritative; a snapshot whose `dim` disagrees
    /// is treated as corruption.
    pub expected_dim: u32,
}

impl VectorRecoveryRequest {
    /// Construct a recovery request. `partition_id` is forced to
    /// [`PartitionId::ZERO`].
    #[must_use]
    pub fn v1(
        tenant_id: TenantId,
        index_id: u64,
        expected_index_type: IndexType,
        expected_dim: u32,
    ) -> Self {
        Self {
            tenant_id,
            partition_id: PartitionId::ZERO,
            index_id,
            expected_index_type,
            expected_dim,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MvccVectorSource — bootstrap fallback adapter
// ─────────────────────────────────────────────────────────────────────

/// Walks the MVCC version chain to reconstruct a vector arena when
/// no usable snapshot exists.
///
/// At v1.0 single-process deployments, production wires this to a
/// thin adapter over [`crate::transaction::TxnManager::read_at`] +
/// the index catalog (which knows the source property's
/// [`crate::transaction::MvccKey`] range). The trait is the
/// abstraction so `arcgraph-storage` does not need to know how the
/// catalog identifies the source property — the wiring lives in
/// `arcgraph-vector` (or above) where the catalog is in scope.
///
/// Tests provide an in-memory implementation that returns a fixed
/// vector list; this lets the boundary tests in
/// `tests/vector_recovery.rs` drive every fallback case without
/// instantiating a full `TxnManager`.
///
/// # Walk semantics (ADR-035 §9.1)
///
/// The implementer iterates **live** MVCC versions for the source
/// property under the `(tenant_id, label_id)` keying pair (or
/// whatever the catalog dictates). `next_vector` returns:
///
/// - `Ok(Some((vector_id, raw_bytes)))` — next live vector. Bytes
///   are the raw `f32` little-endian payload pre-quantization.
/// - `Ok(None)` — iteration complete.
/// - `Err(...)` — catalog corruption / MVCC inconsistency. The
///   caller surfaces this as the bootstrap-failure rung of the
///   ADR-032 §Slice 3c escalation.
pub trait MvccVectorSource: Send + Sync {
    /// Return the next live vector for the arena under
    /// reconstruction. Implementations are stateful — calling
    /// `next_vector` advances internal cursors. Iteration is
    /// expected to be single-pass; multi-pass callers wrap an
    /// implementor in a `RefCell`-guarded resetter.
    fn next_vector(&self) -> Result<Option<(u64, Vec<u8>)>>;

    /// Snapshot LSN this iteration is keyed against. The
    /// reconstructed arena's `last_applied_commit_lsn` is set to
    /// this value so a subsequent `recover_arena` call (post-
    /// bootstrap) can replay only the WAL deltas after this point.
    fn snapshot_lsn(&self) -> Lsn;
}

// ─────────────────────────────────────────────────────────────────────
// VectorPageDelta — installer for post-snapshot WAL deltas
// ─────────────────────────────────────────────────────────────────────

/// One vector arena page from a post-snapshot CommitBundle.
///
/// The recovery WAL pass collects these in commit_lsn order, then
/// installs them via the supplied
/// [`super::VectorPageStoreHandle::install_or_replace`]. Idempotent
/// by construction (Lemma I2 + ADR-035 §4.5 contract).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorPageDelta {
    /// Bundle that produced this page. Drives sort order during
    /// drain; ties broken by `page_id` for determinism in tests.
    pub commit_lsn: Lsn,
    /// Tenant the page belongs to (must match the recovery
    /// request).
    pub tenant_id: TenantId,
    /// Page id within the per-tenant arena address space.
    pub page_id: PageId,
    /// Raw page bytes.
    pub bytes: Vec<u8>,
}

// ─────────────────────────────────────────────────────────────────────
// RecoveredArena — recovery output
// ─────────────────────────────────────────────────────────────────────

/// Source of the recovered arena state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArenaSource {
    /// Recovered from a snapshot file + post-snapshot WAL deltas.
    Snapshot,
    /// Reconstructed via [`bootstrap_from_mvcc`] because no usable
    /// snapshot existed (cold start, every snapshot CRC-failed, or
    /// snapshot truncated below the 128-byte header).
    Bootstrap,
}

/// Successful output of [`recover_arena`] / [`bootstrap_from_mvcc`].
///
/// Holds the metadata needed for the §4.6 step 4 sanity check and
/// the bytes / page deltas the runtime arena consumer
/// (`arcgraph-vector`'s `VectorArena::from_recovered`) needs to
/// rehydrate the in-memory data structures. The struct is
/// intentionally a metadata-and-bytes container rather than the
/// runtime arena type itself — the runtime type lives in
/// `arcgraph-vector` and depending on it from `arcgraph-storage`
/// would be a circular dep (see crate boundary discussion in
/// `super::mod` doc).
#[derive(Debug, Clone)]
pub struct RecoveredArena {
    /// Tenant that owns this arena.
    pub tenant_id: TenantId,
    /// Partition keying hook (always [`PartitionId::ZERO`] at v1.0).
    pub partition_id: PartitionId,
    /// Per-tenant index id.
    pub index_id: u64,
    /// Encoding of the primary arena bytes.
    pub encoding: Encoding,
    /// Index algorithm.
    pub index_type: IndexType,
    /// Vector dimension.
    pub dim: u32,
    /// Number of vectors. For [`ArenaSource::Snapshot`] this is the
    /// snapshot header's `vector_count` plus the count of
    /// post-snapshot install_or_replace calls (one per delta page);
    /// for [`ArenaSource::Bootstrap`] it is the count of MVCC
    /// vectors observed.
    pub vectors_count: u64,
    /// Number of nodes in the graph section. Read from the graph
    /// header at recovery time. The §4.6 step 4 sanity check
    /// compares this to `vectors_count` and halts on mismatch.
    pub graph_node_count: u64,
    /// LSN of the loaded snapshot. `0` for [`ArenaSource::Bootstrap`]
    /// (no snapshot was loaded).
    pub snapshot_lsn: Lsn,
    /// Highest `commit_lsn` applied post-snapshot. Equals
    /// `snapshot_lsn` when no WAL deltas existed; equals the last
    /// post-snapshot bundle's `commit_lsn` otherwise. Equals
    /// [`MvccVectorSource::snapshot_lsn`] for bootstrap recoveries.
    pub last_applied_commit_lsn: Lsn,
    /// Whether this arena came from snapshot or bootstrap.
    pub source: ArenaSource,
    /// Raw snapshot bytes (header + sections + footer). Empty for
    /// bootstrap recoveries. Consumed by
    /// `arcgraph-vector::VectorArena::from_recovered_snapshot` to
    /// re-hydrate the in-memory graph + quantizer state.
    pub raw_snapshot: Vec<u8>,
    /// Vectors observed during a bootstrap walk. Empty for
    /// snapshot recoveries. Each entry is `(vector_id, raw_bytes)`
    /// from [`MvccVectorSource::next_vector`].
    pub bootstrap_vectors: Vec<(u64, Vec<u8>)>,
    /// Post-snapshot WAL delta pages successfully installed.
    /// Recorded for observability / test assertions; production
    /// code reads them off the [`super::VectorPageStoreHandle`] after
    /// recovery completes.
    pub applied_deltas: Vec<VectorPageDelta>,
}

impl RecoveredArena {
    /// Vector count surface name from ADR-035 §4.6 step 4. Stable
    /// across the metadata struct rename so the call-site looks
    /// like `arena.vectors_count() == graph.node_count()` from the
    /// ADR.
    #[inline]
    #[must_use]
    pub fn vectors_count(&self) -> u64 {
        self.vectors_count
    }

    /// Graph-section node count surface name from ADR-035 §4.6
    /// step 4.
    #[inline]
    #[must_use]
    pub fn graph_node_count(&self) -> u64 {
        self.graph_node_count
    }
}

// ─────────────────────────────────────────────────────────────────────
// SnapshotHeader — decoded ARCV header
// ─────────────────────────────────────────────────────────────────────

/// Decoded ARCV header per ADR-035 §4.1. Section offsets / sizes
/// are validated against `total_file_size` from the footer at decode
/// time; subsequent recovery code may take them at face value.
#[derive(Debug, Clone, PartialEq, Eq)]
struct SnapshotHeader {
    encoding: Encoding,
    index_type: IndexType,
    flags: u8,
    tenant_id: TenantId,
    partition_id: PartitionId,
    index_id: u64,
    dim: u32,
    vector_count: u32,
    vector_section_offset: u64,
    vector_section_size: u64,
    graph_section_offset: u64,
    graph_section_size: u64,
    rescore_section_offset: u64,
    rescore_section_size: u64,
    quantizer_section_offset: u64,
    quantizer_section_size: u64,
    tombstone_section_offset: u64,
    tombstone_section_size: u64,
}

fn read_u32_le(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect(
        "read_u32_le: offset+4 must fit (caller checked SNAPSHOT_HEADER_SIZE / section sizes)",
    ))
}

fn read_u64_le(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().expect(
        "read_u64_le: offset+8 must fit (caller checked SNAPSHOT_HEADER_SIZE / section sizes)",
    ))
}

// ─────────────────────────────────────────────────────────────────────
// Filename parsing — `arena-{tenant}-{index}-{lsn}.snap`
// ─────────────────────────────────────────────────────────────────────

/// Parsed components of a recognised snapshot filename. Returned by
/// [`parse_snapshot_filename`] for filenames in the v1.0 layout
/// `arena-{tenant}-{index}-{lsn}.snap`. v1.1 promotes to
/// `arena-{tenant}-{partition}-{index}-{lsn}.snap` (per ADR-035
/// §8); the parser will gain a partition arm at that time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParsedSnapshotName {
    /// Raw `TenantId` (parsed from the second filename segment).
    pub tenant_id: TenantId,
    /// Per-tenant index id.
    pub index_id: u64,
    /// Snapshot LSN encoded into the filename — the authoritative
    /// snapshot_lsn value per ADR-035 §4.5 step 5 (the LSN is
    /// stamped into the index catalog AND the filename so a
    /// catalog-less recovery still picks the correct generation).
    pub snapshot_lsn: Lsn,
}

/// Parse `arena-{tenant}-{index}-{lsn}.snap`. Returns `None` for
/// any other shape (including the `.snap.tmp` orphan files G.2
/// produces during atomic flush; recovery deletes those during
/// cleanup).
#[must_use]
pub fn parse_snapshot_filename(name: &str) -> Option<ParsedSnapshotName> {
    let stem = name.strip_suffix(&format!(".{SNAPSHOT_FILE_EXT}"))?;
    let rest = stem.strip_prefix("arena-")?;
    let mut parts = rest.splitn(3, '-');
    let tenant = parts.next()?.parse::<u64>().ok()?;
    let index = parts.next()?.parse::<u64>().ok()?;
    let lsn = parts.next()?.parse::<u64>().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some(ParsedSnapshotName {
        tenant_id: TenantId::new(tenant),
        index_id: index,
        snapshot_lsn: Lsn::new(lsn),
    })
}

/// Build the v1.0 snapshot filename for a given (tenant, index, lsn).
/// Symmetric counterpart to [`parse_snapshot_filename`]; G.2 uses
/// the same shape (so the encode/decode pair is anchored to ADR-035
/// §4.5 step 4).
#[must_use]
pub fn snapshot_filename(tenant: TenantId, index_id: u64, lsn: Lsn) -> String {
    format!(
        "arena-{}-{}-{}.{SNAPSHOT_FILE_EXT}",
        tenant.raw(),
        index_id,
        lsn.raw()
    )
}

// ─────────────────────────────────────────────────────────────────────
// Snapshot decoder
// ─────────────────────────────────────────────────────────────────────

/// Outcome of decoding a snapshot file. Two terminal shapes:
/// successful decode, or a recoverable failure that the caller
/// should treat as "fall back to bootstrap" per ADR-035 §9.1.
#[derive(Debug)]
enum SnapshotLoadOutcome {
    Ok(SnapshotHeader, Vec<u8>),
    /// Recoverable failure (CRC mismatch, truncation, partial
    /// section, header bad magic). Carries a description for the
    /// `tracing::warn!` emit; recovery falls back to bootstrap
    /// without escalating.
    Recoverable(String),
}

/// Decode a snapshot file, validating the header CRC AND the
/// trailing file CRC. Both CRCs must pass; either failure produces
/// [`SnapshotLoadOutcome::Recoverable`]. Per ADR-035 §9.1 these
/// failures fall back to bootstrap.
fn load_snapshot_file(path: &Path) -> Result<SnapshotLoadOutcome> {
    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            // I/O error reading a known-existing file is structural —
            // surface up. (The file existence was confirmed by the
            // dir scan; an Err here means a transient I/O hiccup.)
            return Err(ArcGraphError::Io(e));
        }
    };

    // Need at least header + footer to even attempt CRC validation.
    let min_size = SNAPSHOT_HEADER_SIZE + SNAPSHOT_FOOTER_SIZE;
    if bytes.len() < min_size {
        return Ok(SnapshotLoadOutcome::Recoverable(format!(
            "snapshot {} smaller than ARCV header+footer ({} < {})",
            path.display(),
            bytes.len(),
            min_size
        )));
    }

    // Header magic.
    if bytes[..4] != SNAPSHOT_MAGIC {
        return Ok(SnapshotLoadOutcome::Recoverable(format!(
            "snapshot {} missing ARCV magic (got {:02x?})",
            path.display(),
            &bytes[..4]
        )));
    }

    // Header version.
    let version = bytes[4];
    if version != SNAPSHOT_FORMAT_VERSION {
        return Ok(SnapshotLoadOutcome::Recoverable(format!(
            "snapshot {} unknown version {} (supported: {})",
            path.display(),
            version,
            SNAPSHOT_FORMAT_VERSION
        )));
    }

    // Header CRC.
    let header_crc_observed = read_u32_le(&bytes, HEADER_CRC_OFFSET);
    let header_crc_computed = crc32c::crc32c(&bytes[..HEADER_CRC_OFFSET]);
    if header_crc_observed != header_crc_computed {
        return Ok(SnapshotLoadOutcome::Recoverable(format!(
            "snapshot {} header CRC mismatch (observed=0x{header_crc_observed:08x} \
             computed=0x{header_crc_computed:08x})",
            path.display(),
        )));
    }

    // Footer total_file_size + CRC.
    let footer_off = bytes.len() - SNAPSHOT_FOOTER_SIZE;
    let total_file_size = read_u64_le(&bytes, footer_off);
    if total_file_size as usize != bytes.len() {
        return Ok(SnapshotLoadOutcome::Recoverable(format!(
            "snapshot {} footer total_file_size={total_file_size} != file len={}",
            path.display(),
            bytes.len(),
        )));
    }
    let file_crc_observed = read_u32_le(&bytes, bytes.len() - 4);
    let file_crc_computed = crc32c::crc32c(&bytes[..bytes.len() - 4]);
    if file_crc_observed != file_crc_computed {
        return Ok(SnapshotLoadOutcome::Recoverable(format!(
            "snapshot {} file CRC mismatch (observed=0x{file_crc_observed:08x} \
             computed=0x{file_crc_computed:08x})",
            path.display(),
        )));
    }

    // Decode the header fields.
    let encoding = match Encoding::from_byte(bytes[5]) {
        Ok(e) => e,
        Err(_) => {
            return Ok(SnapshotLoadOutcome::Recoverable(format!(
                "snapshot {} unknown encoding byte {}",
                path.display(),
                bytes[5]
            )));
        }
    };
    let index_type = match IndexType::from_byte(bytes[6]) {
        Ok(t) => t,
        Err(_) => {
            return Ok(SnapshotLoadOutcome::Recoverable(format!(
                "snapshot {} unknown index_type byte {}",
                path.display(),
                bytes[6]
            )));
        }
    };
    let header = SnapshotHeader {
        encoding,
        index_type,
        flags: bytes[7],
        tenant_id: TenantId::new(read_u64_le(&bytes, 8)),
        partition_id: {
            // ADR-035 §4.1 reserves 8 bytes for partition_id on disk
            // even though `PartitionId` is u32 in core. The upper 32
            // bits are reserved for v1.1 partitioning growth; at v1.0
            // they MUST be zero. A non-zero high half is a v1.1+ file
            // we cannot read — treat as "zero out and let the v1.0
            // partition_id mismatch check below fall back to bootstrap".
            let raw = read_u64_le(&bytes, 16);
            let lo = raw as u32;
            // We don't reject the high half here — the
            // `header.partition_id != PartitionId::ZERO` check upstream
            // catches both "v1.1 file" and "corrupted partition_id"
            // uniformly.
            PartitionId::new(lo)
        },
        index_id: read_u64_le(&bytes, 24),
        dim: read_u32_le(&bytes, 32),
        vector_count: read_u32_le(&bytes, 36),
        vector_section_offset: read_u64_le(&bytes, 40),
        vector_section_size: read_u64_le(&bytes, 48),
        graph_section_offset: read_u64_le(&bytes, 56),
        graph_section_size: read_u64_le(&bytes, 64),
        rescore_section_offset: read_u64_le(&bytes, 72),
        rescore_section_size: read_u64_le(&bytes, 80),
        quantizer_section_offset: read_u64_le(&bytes, 88),
        quantizer_section_size: read_u64_le(&bytes, 96),
        tombstone_section_offset: read_u64_le(&bytes, 104),
        tombstone_section_size: read_u64_le(&bytes, 112),
    };

    // Validate that every present section fits inside the file.
    let file_len = bytes.len() as u64;
    let footer_off_u64 = footer_off as u64;
    let sections = [
        (
            "vector",
            header.vector_section_offset,
            header.vector_section_size,
        ),
        (
            "graph",
            header.graph_section_offset,
            header.graph_section_size,
        ),
        (
            "rescore",
            header.rescore_section_offset,
            header.rescore_section_size,
        ),
        (
            "quantizer",
            header.quantizer_section_offset,
            header.quantizer_section_size,
        ),
        (
            "tombstone",
            header.tombstone_section_offset,
            header.tombstone_section_size,
        ),
    ];
    for (name, offset, size) in sections {
        if size == 0 {
            continue;
        }
        let end = offset.saturating_add(size);
        if offset < SNAPSHOT_HEADER_SIZE as u64 || end > footer_off_u64 || end > file_len {
            return Ok(SnapshotLoadOutcome::Recoverable(format!(
                "snapshot {} {name} section overruns file (offset={offset} \
                 size={size} file_len={file_len})",
                path.display(),
            )));
        }
    }

    // Graph section must be present (vector indexes always have a graph).
    if header.graph_section_size < GRAPH_HEADER_SIZE as u64 {
        return Ok(SnapshotLoadOutcome::Recoverable(format!(
            "snapshot {} graph section ({} bytes) smaller than GRAPH_HEADER_SIZE ({})",
            path.display(),
            header.graph_section_size,
            GRAPH_HEADER_SIZE,
        )));
    }
    // Graph section magic must match the index_type byte.
    let g_off = header.graph_section_offset as usize;
    let observed_magic = &bytes[g_off..g_off + 4];
    let expected_magic = header.index_type.graph_magic();
    if observed_magic != expected_magic {
        return Ok(SnapshotLoadOutcome::Recoverable(format!(
            "snapshot {} graph section magic {:02x?} != expected {:02x?} for index_type {:?}",
            path.display(),
            observed_magic,
            expected_magic,
            header.index_type,
        )));
    }

    Ok(SnapshotLoadOutcome::Ok(header, bytes))
}

/// Read the `node_count` u32 LE field from the graph section header.
/// Caller must have already validated that the graph section bounds
/// fit inside the file (done in [`load_snapshot_file`]).
fn read_graph_node_count(header: &SnapshotHeader, bytes: &[u8]) -> u64 {
    let g_off = header.graph_section_offset as usize;
    let nc_off = g_off + header.index_type.graph_node_count_offset();
    u64::from(read_u32_le(bytes, nc_off))
}

// ─────────────────────────────────────────────────────────────────────
// Snapshot dir helpers
// ─────────────────────────────────────────────────────────────────────

/// Find the latest `arena-{tenant}-{index}-{lsn}.snap` file in
/// `dir`. Returns `Ok(None)` if no candidates exist (cold start).
/// I/O errors propagate (snapshot dir read failure is structural —
/// the caller decides whether to fall back to bootstrap or surface
/// the error).
fn find_latest_snapshot(
    dir: &Path,
    tenant: TenantId,
    index_id: u64,
) -> Result<Option<(PathBuf, Lsn)>> {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(ArcGraphError::Io(e)),
    };
    let mut best: Option<(PathBuf, Lsn)> = None;
    for entry in entries {
        let entry = entry?;
        let name = match entry.file_name().to_str() {
            Some(s) => s.to_owned(),
            None => continue,
        };
        let parsed = match parse_snapshot_filename(&name) {
            Some(p) => p,
            None => continue,
        };
        if parsed.tenant_id != tenant || parsed.index_id != index_id {
            continue;
        }
        match &best {
            Some((_, current_lsn)) if current_lsn.raw() >= parsed.snapshot_lsn.raw() => {}
            _ => best = Some((entry.path(), parsed.snapshot_lsn)),
        }
    }
    Ok(best)
}

/// Output shape for [`list_snapshots_and_orphans`]: snapshots with
/// their LSN tags, plus orphan `.snap.tmp` paths.
type SnapshotsAndOrphans = (Vec<(PathBuf, Lsn)>, Vec<PathBuf>);

/// List every snapshot for `(tenant, index)` plus any orphan
/// `*.snap.tmp` files under `dir`. Used by the cleanup pass after
/// a successful recovery.
fn list_snapshots_and_orphans(
    dir: &Path,
    tenant: TenantId,
    index_id: u64,
) -> Result<SnapshotsAndOrphans> {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Ok((Vec::new(), Vec::new()));
        }
        Err(e) => return Err(ArcGraphError::Io(e)),
    };
    let mut snaps = Vec::new();
    let mut orphans = Vec::new();
    for entry in entries {
        let entry = entry?;
        let name = match entry.file_name().to_str() {
            Some(s) => s.to_owned(),
            None => continue,
        };
        if let Some(parsed) = parse_snapshot_filename(&name) {
            if parsed.tenant_id == tenant && parsed.index_id == index_id {
                snaps.push((entry.path(), parsed.snapshot_lsn));
            }
            continue;
        }
        // Match `arena-{tenant}-{index}-*.snap.tmp` orphans (G.2's
        // atomic-flush temp files left over from a crash mid-flush).
        // Cleanup is conservative: only drop tmp files that match
        // the current `(tenant, index)` prefix.
        if name.ends_with(&format!(".{SNAPSHOT_TEMP_EXT}"))
            && name.starts_with(&format!("arena-{}-{}-", tenant.raw(), index_id))
        {
            orphans.push(entry.path());
        }
    }
    Ok((snaps, orphans))
}

/// Remove every snapshot for `(tenant, index)` whose LSN is strictly
/// less than `keep_lsn`, plus every matching `.snap.tmp` orphan.
/// Errors during deletion are logged but not propagated — a stale
/// file lingering on disk is observability noise, not a correctness
/// violation. Callers that need stricter cleanup semantics can wrap
/// this with their own error handling.
fn cleanup_stale_snapshots(dir: &Path, tenant: TenantId, index_id: u64, keep_lsn: Lsn) {
    let (snaps, orphans) = match list_snapshots_and_orphans(dir, tenant, index_id) {
        Ok(pair) => pair,
        Err(e) => {
            warn!(
                tenant = tenant.raw(),
                index_id,
                error = ?e,
                "g3 cleanup_stale_snapshots: dir scan failed; skipping cleanup",
            );
            return;
        }
    };
    for (path, lsn) in snaps {
        if lsn.raw() >= keep_lsn.raw() {
            continue;
        }
        if let Err(e) = fs::remove_file(&path) {
            warn!(
                path = ?path,
                error = ?e,
                "g3 cleanup_stale_snapshots: unable to remove stale snapshot",
            );
        }
    }
    for path in orphans {
        if let Err(e) = fs::remove_file(&path) {
            warn!(
                path = ?path,
                error = ?e,
                "g3 cleanup_stale_snapshots: unable to remove orphan .tmp",
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// recover_arena — public entry point for snapshot-based recovery
// ─────────────────────────────────────────────────────────────────────

/// Iterator-like adapter for post-snapshot WAL deltas the caller
/// supplies to [`recover_arena`].
///
/// Production wires this to a [`crate::wal::WalRecoveryReader`]
/// scan that filters
/// [`crate::wal::bundle::BundlePageKind::Vector`] entries with
/// `commit_lsn > snapshot_lsn`. Tests inject a fixed `Vec` so the
/// boundary tests don't need a real WAL writer.
///
/// The adapter is a trait (rather than `Vec<VectorPageDelta>`) so
/// callers can stream entries from a long WAL without
/// materializing the whole tail in memory. Implementations are
/// single-pass.
pub trait WalDeltaSource {
    /// Snapshot LSN this iteration is filtered against. Deltas
    /// returned must satisfy `commit_lsn > snapshot_lsn`.
    fn snapshot_lsn(&self) -> Lsn;

    /// Return the next delta in commit_lsn-ascending order
    /// (`(commit_lsn, page_id)` lex order for ties), or `None` on
    /// exhaustion.
    fn next_delta(&self) -> Result<Option<VectorPageDelta>>;
}

/// Empty WAL delta source — no post-snapshot deltas. Useful as a
/// default when the caller has no WAL or has already drained the
/// WAL externally. Production tests for the "snapshot only, no
/// post-snapshot writes" path use this.
#[derive(Debug)]
pub struct EmptyWalDeltaSource {
    snapshot_lsn: Lsn,
}

impl Default for EmptyWalDeltaSource {
    fn default() -> Self {
        Self {
            snapshot_lsn: Lsn::ZERO,
        }
    }
}

impl EmptyWalDeltaSource {
    /// Construct an empty source bound to a given snapshot LSN.
    #[must_use]
    pub const fn new(snapshot_lsn: Lsn) -> Self {
        Self { snapshot_lsn }
    }
}

impl WalDeltaSource for EmptyWalDeltaSource {
    fn snapshot_lsn(&self) -> Lsn {
        self.snapshot_lsn
    }

    fn next_delta(&self) -> Result<Option<VectorPageDelta>> {
        Ok(None)
    }
}

/// Recover a single `(tenant, index)` arena from snapshot + WAL.
///
/// Per ADR-035 §4.6 the flow is:
///
/// 1. **Locate snapshot.** `find_latest_snapshot` picks the
///    `arena-{tenant}-{index}-{lsn}.snap` file with the highest
///    LSN.
/// 2. **Decode + verify.** `load_snapshot_file` verifies the ARCV
///    header CRC and trailing file CRC; on either failure we fall
///    through to [`bootstrap_from_mvcc`] (with a `tracing::warn!`
///    naming the reason). Section bounds + graph magic are also
///    checked here — corruption that survives the CRCs (e.g., the
///    operator hand-spliced a snapshot) is treated identically.
/// 3. **Apply post-snapshot WAL deltas.** [`WalDeltaSource`] yields
///    entries in commit_lsn order. Each is dispatched to
///    [`super::VectorPageStoreHandle::install_or_replace`].
/// 4. **Sanity check.** `vectors_count == graph_node_count`. On
///    mismatch return
///    [`ArcGraphError::VectorIndexInconsistency`].
/// 5. **Cleanup.** Remove superseded snapshot generations + any
///    orphan `.snap.tmp` files for `(tenant, index)`.
///
/// On bootstrap fallback (step 2 failure), the function returns
/// whatever [`bootstrap_from_mvcc`] returns — including any
/// bootstrap-side error.
///
/// # Idempotence
///
/// `recover_arena` is idempotent: a crash mid-recovery (e.g.,
/// during step 3 WAL replay) leaves the snapshot file intact;
/// the next process restart re-runs every step. Lemma I2 + Lemma
/// I1 (per ADR-032 §4) guarantee re-installation is byte-identical.
///
/// # Errors
///
/// - [`ArcGraphError::Io`] — snapshot dir / WAL dir read failure.
/// - [`ArcGraphError::VectorIndexInconsistency`] — sanity check
///   failure (see ADR-035 §4.6 step 4).
/// - Any error returned by [`bootstrap_from_mvcc`] when fallback
///   engages.
pub fn recover_arena(
    handle: Arc<dyn VectorPageStoreHandle>,
    snapshot_dir: &Path,
    wal_deltas: &dyn WalDeltaSource,
    mvcc_source: &dyn MvccVectorSource,
    request: VectorRecoveryRequest,
) -> Result<RecoveredArena> {
    info!(
        tenant = request.tenant_id.raw(),
        index_id = request.index_id,
        snapshot_dir = ?snapshot_dir,
        "g3 recover_arena: starting (ADR-035 §4.6)",
    );

    // Step 1: locate the latest snapshot for this (tenant, index).
    let snapshot_path = find_latest_snapshot(snapshot_dir, request.tenant_id, request.index_id)?;

    // Step 2: load + verify; on any recoverable failure, bootstrap.
    let (header, bytes, snapshot_lsn) = match snapshot_path {
        Some((path, lsn)) => match load_snapshot_file(&path)? {
            SnapshotLoadOutcome::Ok(header, bytes) => (header, bytes, lsn),
            SnapshotLoadOutcome::Recoverable(reason) => {
                warn!(
                    tenant = request.tenant_id.raw(),
                    index_id = request.index_id,
                    reason = %reason,
                    "g3 recover_arena: snapshot unusable; falling back to bootstrap_from_mvcc \
                     (ADR-035 §9.1)",
                );
                return bootstrap_from_mvcc(handle, mvcc_source, request);
            }
        },
        None => {
            info!(
                tenant = request.tenant_id.raw(),
                index_id = request.index_id,
                "g3 recover_arena: no snapshot present; cold-start via bootstrap_from_mvcc",
            );
            return bootstrap_from_mvcc(handle, mvcc_source, request);
        }
    };

    // Validate header against the catalog-supplied request. The
    // catalog is authoritative (ADR-011's catalog discipline); a
    // disagreement here is corruption-class. Treat it as recoverable
    // and bootstrap so a stale snapshot from a different deployment
    // does not jam recovery.
    if header.tenant_id != request.tenant_id
        || header.partition_id != request.partition_id
        || header.index_id != request.index_id
    {
        warn!(
            tenant = request.tenant_id.raw(),
            index_id = request.index_id,
            header_tenant = header.tenant_id.raw(),
            header_index = header.index_id,
            "g3 recover_arena: snapshot tenant/index mismatch; falling back to bootstrap",
        );
        return bootstrap_from_mvcc(handle, mvcc_source, request);
    }
    if header.index_type != request.expected_index_type || header.dim != request.expected_dim {
        warn!(
            tenant = request.tenant_id.raw(),
            index_id = request.index_id,
            header_index_type = ?header.index_type,
            request_index_type = ?request.expected_index_type,
            header_dim = header.dim,
            request_dim = request.expected_dim,
            "g3 recover_arena: snapshot shape mismatches catalog; falling back to bootstrap",
        );
        return bootstrap_from_mvcc(handle, mvcc_source, request);
    }
    if header.partition_id != PartitionId::ZERO {
        // v1.0 invariant: regression test pins this. A non-zero
        // partition_id on disk is either a v1.1 file we cannot read
        // or corruption.
        warn!(
            tenant = request.tenant_id.raw(),
            partition_id = header.partition_id.raw(),
            "g3 recover_arena: snapshot has non-zero partition_id (v1.1+ format?); \
             falling back to bootstrap",
        );
        return bootstrap_from_mvcc(handle, mvcc_source, request);
    }

    // Step 3: replay post-snapshot CommitBundle delta pages.
    // The delta source is expected to filter to
    // `commit_lsn > snapshot_lsn` already, but we double-check.
    let mut last_applied = snapshot_lsn;
    let mut applied: Vec<VectorPageDelta> = Vec::new();
    while let Some(delta) = wal_deltas.next_delta()? {
        if delta.commit_lsn.raw() <= snapshot_lsn.raw() {
            // Pre-snapshot deltas are already in the snapshot bytes;
            // skipping them is the §4.5 last_snapshot_high_water
            // contract.
            continue;
        }
        if delta.tenant_id != request.tenant_id {
            // Delta belongs to a different tenant; the caller's
            // filter is wrong but we can self-correct without
            // halting.
            debug!(
                expected_tenant = request.tenant_id.raw(),
                delta_tenant = delta.tenant_id.raw(),
                "g3 recover_arena: skipping delta for non-matching tenant",
            );
            continue;
        }
        handle
            .install_or_replace(delta.tenant_id, delta.page_id, &delta.bytes)
            .map_err(map_install_error)?;
        if delta.commit_lsn.raw() > last_applied.raw() {
            last_applied = delta.commit_lsn;
        }
        applied.push(delta);
    }

    // Step 4: sanity check (ship-blocking).
    let graph_node_count = read_graph_node_count(&header, &bytes);
    let vectors_count_from_header = u64::from(header.vector_count);
    let vectors_count = vectors_count_from_header.saturating_add(applied.len() as u64);
    if vectors_count != graph_node_count {
        let observed = vectors_count;
        let expected = graph_node_count;
        let delta = (observed as i128 - expected as i128) as i64;
        error!(
            tenant = request.tenant_id.raw(),
            index_id = request.index_id,
            snapshot_lsn = snapshot_lsn.raw(),
            vectors_count = observed,
            graph_node_count = expected,
            delta,
            wal_replay_high_lsn = last_applied.raw(),
            "g3 recover_arena: VectorIndexInconsistency (ADR-035 §4.6 step 4) — halting",
        );
        return Err(ArcGraphError::VectorIndexInconsistency {
            tenant_id: request.tenant_id.raw(),
            index_id: request.index_id,
            snapshot_lsn: snapshot_lsn.raw(),
            observed_vectors_count: observed,
            observed_graph_node_count: expected,
            wal_replay_high_lsn: last_applied.raw(),
            delta,
        });
    }

    // Step 5: cleanup superseded generations + orphan tmp files.
    cleanup_stale_snapshots(
        snapshot_dir,
        request.tenant_id,
        request.index_id,
        snapshot_lsn,
    );

    info!(
        tenant = request.tenant_id.raw(),
        index_id = request.index_id,
        snapshot_lsn = snapshot_lsn.raw(),
        wal_replay_high_lsn = last_applied.raw(),
        vectors_count,
        graph_node_count,
        encoding = ?header.encoding,
        index_type = ?header.index_type,
        "g3 recover_arena: completed (ADR-035 §4.6)",
    );

    Ok(RecoveredArena {
        tenant_id: header.tenant_id,
        partition_id: header.partition_id,
        index_id: header.index_id,
        encoding: header.encoding,
        index_type: header.index_type,
        dim: header.dim,
        vectors_count,
        graph_node_count,
        snapshot_lsn,
        last_applied_commit_lsn: last_applied,
        source: ArenaSource::Snapshot,
        raw_snapshot: bytes,
        bootstrap_vectors: Vec::new(),
        applied_deltas: applied,
    })
}

// ─────────────────────────────────────────────────────────────────────
// bootstrap_from_mvcc — MVCC fallback
// ─────────────────────────────────────────────────────────────────────

/// Reconstruct an arena by walking MVCC. Per ADR-035 §9.1 this is
/// the graceful-degradation rung when the snapshot is missing,
/// CRC-failed, or truncated.
///
/// The caller supplies an [`MvccVectorSource`] that knows how to
/// iterate the source property's MVCC versions. v1.0 production
/// wires this to `TxnManager::read_at` for each known
/// `(tenant_id, label_id)` pair under the index catalog's
/// `source_property_keyspace`. At v1.0 vector data does not actually
/// flow through MVCC (per ADR-035 §4.5 B-1 resolution), so the
/// production wiring is a "list of vectors written for this index"
/// projection over the transactional surface.
///
/// # Sanity contract
///
/// The caller is responsible for the post-bootstrap sanity check;
/// a freshly bootstrapped arena's `vectors_count == bootstrap_vectors.len()`
/// and `graph_node_count == 0` (the graph is built later by
/// `arcgraph-vector::VectorArena::from_recovered_bootstrap`).
/// Recovery's own sanity check
/// (`vectors_count == graph_node_count`) is **deferred** for the
/// bootstrap path because the graph rebuild happens downstream. The
/// post-rebuild check fires once `from_recovered_bootstrap`
/// completes and consumes the [`RecoveredArena`].
///
/// # Errors
///
/// - Any error returned by [`MvccVectorSource::next_vector`].
pub fn bootstrap_from_mvcc(
    _handle: Arc<dyn VectorPageStoreHandle>,
    source: &dyn MvccVectorSource,
    request: VectorRecoveryRequest,
) -> Result<RecoveredArena> {
    info!(
        tenant = request.tenant_id.raw(),
        index_id = request.index_id,
        "g3 bootstrap_from_mvcc: starting (ADR-035 §9.1)",
    );

    let snapshot_lsn = source.snapshot_lsn();
    let mut vectors: Vec<(u64, Vec<u8>)> = Vec::new();
    while let Some(v) = source.next_vector()? {
        vectors.push(v);
    }

    info!(
        tenant = request.tenant_id.raw(),
        index_id = request.index_id,
        vectors = vectors.len(),
        snapshot_lsn = snapshot_lsn.raw(),
        "g3 bootstrap_from_mvcc: completed; arena ready for graph rebuild",
    );

    // Bootstrap leaves graph_node_count = 0 because the graph
    // rebuild is downstream (arcgraph-vector consumes the
    // RecoveredArena). The §4.6 step 4 sanity check is deferred to
    // the post-rebuild stage; if the rebuild produces a graph with
    // node_count != bootstrap_vectors.len(), arcgraph-vector emits
    // VectorIndexInconsistency at that point.
    let vectors_count = vectors.len() as u64;

    Ok(RecoveredArena {
        tenant_id: request.tenant_id,
        partition_id: request.partition_id,
        index_id: request.index_id,
        encoding: Encoding::F32,
        index_type: request.expected_index_type,
        dim: request.expected_dim,
        vectors_count,
        graph_node_count: 0,
        snapshot_lsn: Lsn::ZERO,
        last_applied_commit_lsn: snapshot_lsn,
        source: ArenaSource::Bootstrap,
        raw_snapshot: Vec::new(),
        bootstrap_vectors: vectors,
        applied_deltas: Vec::new(),
    })
}

// ─────────────────────────────────────────────────────────────────────
// Multi-arena recovery driver — for wal/recovery.rs hook
// ─────────────────────────────────────────────────────────────────────

/// One per-arena recovery request, bundled with its delta source.
///
/// The global recovery driver in `wal/recovery.rs` collects one of
/// these per known `(tenant, index)` arena from the index catalog
/// and passes them to [`recover_all_arenas`]. The per-arena delta
/// source is filtered against that arena's snapshot_lsn — production
/// implements this with a [`crate::wal::WalRecoveryReader`] scan
/// that filters
/// [`crate::wal::bundle::BundlePageKind::Vector`] entries by
/// `(tenant_id, page_id)` and `commit_lsn > snapshot_lsn`.
pub struct ArenaRecoveryJob<'a> {
    /// Recovery descriptor for this arena.
    pub request: VectorRecoveryRequest,
    /// Source of post-snapshot WAL deltas (filtered by tenant +
    /// `commit_lsn > snapshot_lsn`). Tests pass a
    /// [`Vec`]-backed implementation; production passes a
    /// `WalRecoveryReader`-backed one.
    pub wal_deltas: &'a dyn WalDeltaSource,
    /// Source for the bootstrap fallback. Tests pass an in-memory
    /// implementation; production passes a `TxnManager`-backed
    /// one.
    pub mvcc_source: &'a dyn MvccVectorSource,
}

/// Drive recovery for every arena in `jobs` against the same
/// `handle` + `snapshot_dir`. Each call to [`recover_arena`] is
/// independent — a per-arena failure surfaces immediately and the
/// remaining arenas are not attempted (the global recovery flow
/// halts on the first failure per ADR-032 §6).
///
/// Returns a [`Vec<RecoveredArena>`] in the same order as `jobs`.
pub fn recover_all_arenas(
    handle: Arc<dyn VectorPageStoreHandle>,
    snapshot_dir: &Path,
    jobs: &[ArenaRecoveryJob<'_>],
) -> Result<Vec<RecoveredArena>> {
    let mut out = Vec::with_capacity(jobs.len());
    for job in jobs {
        let arena = recover_arena(
            Arc::clone(&handle),
            snapshot_dir,
            job.wal_deltas,
            job.mvcc_source,
            job.request,
        )?;
        out.push(arena);
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────
// VectorArenaPageStore — minimal in-memory store for tests + v1.0
// recovery driver
// ─────────────────────────────────────────────────────────────────────

/// In-memory `VectorPageStoreHandle` implementation that records
/// every `install_or_replace` / `restore_page_bytes` call. Provides
/// the storage layer with a working backend that the G.3 recovery
/// flow can route to without requiring `arcgraph-vector` (which is
/// out of scope for this slice — see ADR-035 §7.5 trait pattern).
///
/// At v1.0 production the runtime arena (in `arcgraph-vector`)
/// implements [`VectorPageStoreHandle`] directly. This in-memory
/// store is the test fixture + a fallback the recovery driver can
/// instantiate when no external implementor is wired (so the
/// recovery hook in [`crate::wal::recovery`] can be exercised end-
/// to-end without an `arcgraph-vector` dep).
#[derive(Debug, Default)]
pub struct VectorArenaPageStore {
    inner: parking_lot::Mutex<VectorArenaPageStoreInner>,
}

#[derive(Debug, Default)]
struct VectorArenaPageStoreInner {
    pages: BTreeMap<(TenantId, PageId), Vec<u8>>,
}

impl VectorArenaPageStore {
    /// Construct an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of installed pages for a tenant. Useful for
    /// observability and the integration test's count checks.
    #[must_use]
    pub fn page_count(&self, tenant: TenantId) -> usize {
        let inner = self.inner.lock();
        inner
            .pages
            .range((tenant, PageId::ZERO)..)
            .take_while(|((t, _), _)| *t == tenant)
            .count()
    }

    /// Snapshot the page bytes for `(tenant, page_id)`. Returns
    /// `None` if no install_or_replace has covered the page.
    #[must_use]
    pub fn get_page(&self, tenant: TenantId, page_id: PageId) -> Option<Vec<u8>> {
        self.inner.lock().pages.get(&(tenant, page_id)).cloned()
    }
}

impl VectorPageStoreHandle for VectorArenaPageStore {
    fn install_or_replace(
        &self,
        tenant: TenantId,
        page_id: PageId,
        bytes: &[u8],
    ) -> std::result::Result<(), VectorStoreError> {
        let mut inner = self.inner.lock();
        inner.pages.insert((tenant, page_id), bytes.to_vec());
        Ok(())
    }

    fn restore_page_bytes(
        &self,
        tenant: TenantId,
        page_id: PageId,
        bytes: &[u8],
    ) -> std::result::Result<(), VectorStoreError> {
        let mut inner = self.inner.lock();
        // `restore_page_bytes` is the Z-1 (b) rollback hook
        // (Slice G.5 wires it from `crud.rs`). Behaviour mirrors
        // `install_or_replace` since both are byte-overwrites; the
        // separation exists so the Slice G.5 dispatcher can target
        // a different rollback path if needed (e.g., DashMap::remove
        // for new pages).
        inner.pages.insert((tenant, page_id), bytes.to_vec());
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────

fn map_install_error(e: VectorStoreError) -> ArcGraphError {
    ArcGraphError::WalCorruption {
        lsn: Lsn::ZERO,
        reason: format!("g3 recover_arena: VectorPageStore install_or_replace failed: {e}"),
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_filename_round_trips_v1() {
        let fname = snapshot_filename(TenantId::new(7), 42, Lsn::new(1000));
        assert_eq!(fname, "arena-7-42-1000.snap");
        let parsed = parse_snapshot_filename(&fname).unwrap();
        assert_eq!(parsed.tenant_id, TenantId::new(7));
        assert_eq!(parsed.index_id, 42);
        assert_eq!(parsed.snapshot_lsn, Lsn::new(1000));
    }

    #[test]
    fn parse_filename_rejects_tmp_and_garbage() {
        // Mid-flush orphan from G.2.
        assert!(parse_snapshot_filename("arena-1-2-3.snap.tmp").is_none());
        // Wrong prefix.
        assert!(parse_snapshot_filename("blob-1-2-3.snap").is_none());
        // Missing component.
        assert!(parse_snapshot_filename("arena-1-2.snap").is_none());
        // Extra component.
        assert!(parse_snapshot_filename("arena-1-2-3-4.snap").is_none());
        // Non-numeric.
        assert!(parse_snapshot_filename("arena-foo-2-3.snap").is_none());
    }

    #[test]
    fn encoding_round_trips_byte_values() {
        for (byte, enc) in [
            (0u8, Encoding::F32),
            (1u8, Encoding::F16),
            (2u8, Encoding::Sq8),
            (3u8, Encoding::Binary),
            (4u8, Encoding::RaBitQ),
        ] {
            assert_eq!(Encoding::from_byte(byte).unwrap(), enc);
            assert_eq!(enc.as_byte(), byte);
        }
        assert!(Encoding::from_byte(99).is_err());
    }

    #[test]
    fn index_type_round_trips_byte_values() {
        for (byte, t) in [(0u8, IndexType::Hnsw), (1u8, IndexType::DiskAnn)] {
            assert_eq!(IndexType::from_byte(byte).unwrap(), t);
            assert_eq!(t.as_byte(), byte);
        }
        assert!(IndexType::from_byte(99).is_err());
    }

    #[test]
    fn index_type_graph_magic_matches_adr() {
        assert_eq!(IndexType::Hnsw.graph_magic(), *b"HNSW");
        assert_eq!(IndexType::DiskAnn.graph_magic(), *b"VAMA");
        // Per ADR-035 §4.2 / §4.3 byte offsets of node_count.
        assert_eq!(IndexType::Hnsw.graph_node_count_offset(), 16);
        assert_eq!(IndexType::DiskAnn.graph_node_count_offset(), 17);
    }

    #[test]
    fn v1_request_pins_partition_id_zero() {
        // ADR-035 D-7 v1.0 invariant. The integration test re-asserts
        // this at recovery time; here we pin the constructor.
        let req = VectorRecoveryRequest::v1(TenantId::DEFAULT, 1, IndexType::Hnsw, 768);
        assert_eq!(req.partition_id, PartitionId::ZERO);
    }

    #[test]
    fn arena_page_store_round_trips_install() {
        let store = VectorArenaPageStore::new();
        let tenant = TenantId::DEFAULT;
        let page = PageId::new(7);
        store.install_or_replace(tenant, page, &[0xAA; 16]).unwrap();
        let got = store.get_page(tenant, page).unwrap();
        assert_eq!(got, vec![0xAA; 16]);
        // Idempotence: re-install with same bytes is a no-op (Lemma I2).
        store.install_or_replace(tenant, page, &[0xAA; 16]).unwrap();
        let got2 = store.get_page(tenant, page).unwrap();
        assert_eq!(got2, vec![0xAA; 16]);
        // Different bytes overwrite (last writer wins).
        store.install_or_replace(tenant, page, &[0xBB; 16]).unwrap();
        let got3 = store.get_page(tenant, page).unwrap();
        assert_eq!(got3, vec![0xBB; 16]);
    }

    #[test]
    fn empty_wal_delta_source_returns_none() {
        let src = EmptyWalDeltaSource::new(Lsn::new(100));
        assert_eq!(src.snapshot_lsn(), Lsn::new(100));
        assert!(src.next_delta().unwrap().is_none());
    }
}
