//! M3.a Slice G.3 — Path A boundary tests for vector arena recovery.
//!
//! Per ADR-035 §4.5/§4.6 + ADR-032 §R1-R7 + ADR-033 Z-1 (b). Each
//! test drives a focused failure / recovery scenario through the
//! G.3 entry points and asserts the operator-actionable contract:
//!
//! - **Recovery from clean snapshot.** `g3_recover_from_clean_snapshot`
//!   + `g3_recover_with_post_snapshot_wal_delta`.
//! - **Missing / corrupt / partial snapshot fallback.**
//!   `g3_missing_snapshot_falls_back_to_bootstrap`,
//!   `g3_crc_corruption_falls_back_to_bootstrap`,
//!   `g3_truncated_snapshot_falls_back_to_bootstrap`,
//!   `g3_partial_section_falls_back`.
//! - **Mismatch sanity halt.** `g3_arena_count_mismatch_halts_replay`,
//!   `g3_inconsistency_diagnostics_actionable`.
//! - **Crash mid-recovery resume.**
//!   `g3_crash_mid_recovery_resumes_correctly`.
//! - **Multi-tenant mixed encodings.**
//!   `g3_recover_4_tenants_mixed_encodings`.
//! - **Bootstrap halt on corrupt MVCC.**
//!   `g3_bootstrap_failure_halts_with_error`.
//! - **Cleanup of stale snapshots / .tmp.**
//!   `g3_recover_cleans_up_old_snapshots`.
//! - **Z-1 (b) rollback integration with vector arena pages.**
//!   `g3_z1_rollback_with_vector_arena_pages`.
//! - **Local-only hooks.**
//!   `g3_recover_request_partition_id_always_zero_at_v1`.
//!
//! Each test is self-contained: writes its own snapshot file via
//! [`write_snapshot`] (the in-test ARCV encoder per ADR-035 §4.1)
//! into a `tempdir()`, drives [`recover_arena`] / [`bootstrap_from_mvcc`],
//! and checks the operator-facing assertions listed in the slice's
//! Path A spec.
//!
//! # Snapshot wire-format anchor
//!
//! The encoder in this file is the "test mirror" of the snapshot
//! format defined in ADR-035 §4.1 and decoded by `recovery.rs`.
//! When G.2 (snapshot writer) lands, both sides remain anchored to
//! the same byte layout — so a snapshot produced by G.2 is decoded
//! identically by this test's encoder ⇄ recovery decoder pair.
//!
//! Run:
//!   cargo test -p arcgraph-storage --test vector_recovery

use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arcgraph_core::record::PAGE_SIZE;
use arcgraph_core::{ArcGraphError, Lsn, PageId, PartitionId, TenantId};
use arcgraph_storage::mutation_log::{PageStoreKind, TxnMutationLog};
use arcgraph_storage::vector_store::VectorPageStoreHandle;
use arcgraph_storage::vector_store::recovery::{
    ArenaSource, Encoding, IndexType, MvccVectorSource, SNAPSHOT_FOOTER_SIZE,
    SNAPSHOT_FORMAT_VERSION, SNAPSHOT_HEADER_SIZE, VectorArenaPageStore, VectorPageDelta,
    VectorRecoveryRequest, WalDeltaSource, bootstrap_from_mvcc, recover_arena, snapshot_filename,
};
use parking_lot::Mutex;
use tempfile::TempDir;

// ─────────────────────────────────────────────────────────────────────
// Test fixtures
// ─────────────────────────────────────────────────────────────────────

/// Construct a synthetic ARCV-format snapshot. Mirrors the byte
/// layout in ADR-035 §4.1 + §4.2 / §4.3 (graph header). Writes the
/// file to `dir/arena-{tenant}-{index}-{lsn}.snap` and returns the
/// path.
///
/// The graph_node_count parameter is passed independently of
/// vector_count so tests can synthesize the §4.6 step 4 mismatch
/// scenario by setting them unequal. For all "happy path" tests
/// they must match.
struct SnapshotBuilder {
    tenant: TenantId,
    index_id: u64,
    snapshot_lsn: Lsn,
    encoding: Encoding,
    index_type: IndexType,
    dim: u32,
    vector_count: u32,
    graph_node_count: u32,
    /// Extra padding inserted into the vector section. Lets tests
    /// produce snapshots of varying sizes without changing the
    /// fundamental shape.
    pad_bytes: usize,
    /// When true, encode a graph header whose `node_count` field
    /// reads as `graph_node_count`. When false, write zeros into the
    /// node_count slot — used by the partial-section fallback test
    /// to produce a file that decodes its header but fails the
    /// graph-section-magic check.
    write_graph_header: bool,
    partition_id: PartitionId,
    /// Which header version byte to stamp; non-1 triggers the
    /// "unknown version" recoverable fallback path.
    format_version: u8,
}

impl SnapshotBuilder {
    fn new(tenant: TenantId, index_id: u64, snapshot_lsn: Lsn) -> Self {
        Self {
            tenant,
            index_id,
            snapshot_lsn,
            encoding: Encoding::F32,
            index_type: IndexType::Hnsw,
            dim: 8,
            vector_count: 4,
            graph_node_count: 4,
            pad_bytes: 0,
            write_graph_header: true,
            partition_id: PartitionId::ZERO,
            format_version: SNAPSHOT_FORMAT_VERSION,
        }
    }

    fn encoding(mut self, e: Encoding) -> Self {
        self.encoding = e;
        self
    }

    fn index_type(mut self, t: IndexType) -> Self {
        self.index_type = t;
        self
    }

    fn dim(mut self, d: u32) -> Self {
        self.dim = d;
        self
    }

    fn counts(mut self, vector_count: u32, graph_node_count: u32) -> Self {
        self.vector_count = vector_count;
        self.graph_node_count = graph_node_count;
        self
    }

    #[allow(dead_code)]
    fn pad_bytes(mut self, n: usize) -> Self {
        self.pad_bytes = n;
        self
    }

    fn omit_graph_header(mut self) -> Self {
        self.write_graph_header = false;
        self
    }

    fn build(&self) -> Vec<u8> {
        // Layout: header (128) | vector section | graph section | footer (16).
        // Sections are minimal — vectors as zero-filled bytes,
        // graph header (64 bytes) followed by the graph-section
        // tail (zero-filled). Pad bytes are appended to the vector
        // section to let tests produce files of varying sizes.
        let bytes_per_vec = self.encoding.bytes_per_vector_aligned(self.dim as usize);
        let vector_size = bytes_per_vec * self.vector_count as usize + self.pad_bytes;
        let graph_size = 64 + 16; // 64-byte header + tiny tail
        let vector_offset = SNAPSHOT_HEADER_SIZE;
        let graph_offset = vector_offset + vector_size;
        let total = SNAPSHOT_HEADER_SIZE + vector_size + graph_size + SNAPSHOT_FOOTER_SIZE;

        let mut out = vec![0u8; total];

        // Header.
        out[0..4].copy_from_slice(b"ARCV");
        out[4] = self.format_version;
        out[5] = self.encoding.as_byte();
        out[6] = self.index_type.as_byte();
        out[7] = 0; // flags
        out[8..16].copy_from_slice(&self.tenant.raw().to_le_bytes());
        out[16..24].copy_from_slice(&u64::from(self.partition_id.raw()).to_le_bytes());
        out[24..32].copy_from_slice(&self.index_id.to_le_bytes());
        out[32..36].copy_from_slice(&self.dim.to_le_bytes());
        out[36..40].copy_from_slice(&self.vector_count.to_le_bytes());
        out[40..48].copy_from_slice(&(vector_offset as u64).to_le_bytes());
        out[48..56].copy_from_slice(&(vector_size as u64).to_le_bytes());
        out[56..64].copy_from_slice(&(graph_offset as u64).to_le_bytes());
        out[64..72].copy_from_slice(&(graph_size as u64).to_le_bytes());
        // rescore + quantizer + tombstone sections absent (size = 0).
        // Header CRC over bytes 0..120 → write into 120..124.
        let header_crc = crc32c::crc32c(&out[..120]);
        out[120..124].copy_from_slice(&header_crc.to_le_bytes());

        // Vector section (already zero-filled).

        // Graph section: header at graph_offset.
        if self.write_graph_header {
            let magic = self.index_type.graph_magic();
            out[graph_offset..graph_offset + 4].copy_from_slice(&magic);
            // version byte
            out[graph_offset + 4] = 1;
            // node_count: HNSW = bytes 16..20 of header; VAMA = 17..21
            let nc_off = graph_offset + self.index_type.graph_node_count_offset();
            out[nc_off..nc_off + 4].copy_from_slice(&self.graph_node_count.to_le_bytes());
        }

        // Footer: total_file_size + CRC.
        let footer_off = total - SNAPSHOT_FOOTER_SIZE;
        out[footer_off..footer_off + 8].copy_from_slice(&(total as u64).to_le_bytes());
        // Trailing 4 bytes = file CRC over bytes 0..(total-4).
        let file_crc = crc32c::crc32c(&out[..total - 4]);
        out[total - 4..].copy_from_slice(&file_crc.to_le_bytes());

        out
    }
}

/// Test-only helper for `Encoding::bytes_per_vector_aligned`. The
/// recovery module's `Encoding` enum is the wire-side shadow of
/// `arcgraph-vector::Encoding`; we replicate the minimal sizing
/// helper here so the snapshot builder doesn't need to depend on
/// `arcgraph-vector` (which is downstream of `arcgraph-storage`
/// and therefore can't be a dev-dep here).
trait EncodingExt {
    fn bytes_per_vector_aligned(self, dim: usize) -> usize;
}

impl EncodingExt for Encoding {
    fn bytes_per_vector_aligned(self, dim: usize) -> usize {
        match self {
            Encoding::F32 => dim * 4,
            Encoding::F16 => dim * 2,
            Encoding::Sq8 => dim,
            Encoding::Binary => {
                // 1 bit per dim; round to 64-byte cache lines per
                // ADR-035 S-1.
                let unaligned = dim.div_ceil(8);
                unaligned.next_multiple_of(64)
            }
            Encoding::RaBitQ => {
                let unaligned = dim.div_ceil(8) + 8;
                unaligned.next_multiple_of(64)
            }
        }
    }
}

/// Trivial in-memory `MvccVectorSource` for tests. Returns vectors
/// from the underlying `Vec` once each in insertion order.
struct InMemoryMvccSource {
    vectors: Mutex<std::collections::VecDeque<(u64, Vec<u8>)>>,
    snapshot_lsn: Lsn,
    /// When set, the next call to `next_vector` returns this error
    /// instead of advancing — used by
    /// `g3_bootstrap_failure_halts_with_error` to drive the failure
    /// rung of ADR-032 Slice 3c escalation.
    error: Mutex<Option<ArcGraphError>>,
}

impl InMemoryMvccSource {
    fn new(snapshot_lsn: Lsn, vectors: Vec<(u64, Vec<u8>)>) -> Self {
        Self {
            vectors: Mutex::new(std::collections::VecDeque::from(vectors)),
            snapshot_lsn,
            error: Mutex::new(None),
        }
    }

    fn empty(snapshot_lsn: Lsn) -> Self {
        Self::new(snapshot_lsn, Vec::new())
    }

    fn with_error(error: ArcGraphError) -> Self {
        Self {
            vectors: Mutex::new(std::collections::VecDeque::new()),
            snapshot_lsn: Lsn::ZERO,
            error: Mutex::new(Some(error)),
        }
    }
}

impl MvccVectorSource for InMemoryMvccSource {
    fn next_vector(&self) -> Result<Option<(u64, Vec<u8>)>, ArcGraphError> {
        if let Some(e) = self.error.lock().take() {
            return Err(e);
        }
        Ok(self.vectors.lock().pop_front())
    }

    fn snapshot_lsn(&self) -> Lsn {
        self.snapshot_lsn
    }
}

/// Trivial in-memory `WalDeltaSource` for tests. Returns each
/// queued delta once in insertion order.
struct InMemoryWalDeltas {
    snapshot_lsn: Lsn,
    deltas: Mutex<std::collections::VecDeque<VectorPageDelta>>,
    /// Counter the test inspects to pin "we actually called the
    /// delta source" — the dispatch arm in `recover_arena` is the
    /// only caller, so a zero count means recovery never reached
    /// step 3.
    pulls: AtomicUsize,
}

impl InMemoryWalDeltas {
    fn new(snapshot_lsn: Lsn, deltas: Vec<VectorPageDelta>) -> Self {
        Self {
            snapshot_lsn,
            deltas: Mutex::new(std::collections::VecDeque::from(deltas)),
            pulls: AtomicUsize::new(0),
        }
    }

    fn empty(snapshot_lsn: Lsn) -> Self {
        Self::new(snapshot_lsn, Vec::new())
    }

    fn pulls(&self) -> usize {
        self.pulls.load(Ordering::Relaxed)
    }
}

impl WalDeltaSource for InMemoryWalDeltas {
    fn snapshot_lsn(&self) -> Lsn {
        self.snapshot_lsn
    }

    fn next_delta(&self) -> Result<Option<VectorPageDelta>, ArcGraphError> {
        self.pulls.fetch_add(1, Ordering::Relaxed);
        Ok(self.deltas.lock().pop_front())
    }
}

fn write_snapshot_file(dir: &Path, builder: &SnapshotBuilder) -> std::path::PathBuf {
    let bytes = builder.build();
    let path = dir.join(snapshot_filename(
        builder.tenant,
        builder.index_id,
        builder.snapshot_lsn,
    ));
    fs::write(&path, &bytes).expect("write snapshot");
    path
}

// ─────────────────────────────────────────────────────────────────────
// 1. Recovery from genuine snapshot
// ─────────────────────────────────────────────────────────────────────

#[test]
fn g3_recover_from_clean_snapshot() {
    // ADR-035 §4.6 happy path: snapshot decodes; no post-snapshot
    // WAL deltas; sanity check passes; recovered arena reports
    // the byte-perfect metadata.
    let dir = TempDir::new().unwrap();
    let tenant = TenantId::new(7);
    let req = VectorRecoveryRequest::v1(tenant, 42, IndexType::Hnsw, 8);
    let lsn = Lsn::new(1000);

    write_snapshot_file(
        dir.path(),
        &SnapshotBuilder::new(tenant, 42, lsn).counts(16, 16),
    );

    let store: Arc<dyn VectorPageStoreHandle> = Arc::new(VectorArenaPageStore::new());
    let deltas = InMemoryWalDeltas::empty(lsn);
    let mvcc = InMemoryMvccSource::empty(lsn);

    let arena = recover_arena(store, dir.path(), &deltas, &mvcc, req).expect("recover");

    assert_eq!(arena.source, ArenaSource::Snapshot);
    assert_eq!(arena.tenant_id, tenant);
    assert_eq!(arena.index_id, 42);
    assert_eq!(arena.encoding, Encoding::F32);
    assert_eq!(arena.index_type, IndexType::Hnsw);
    assert_eq!(arena.dim, 8);
    assert_eq!(arena.vectors_count(), 16);
    assert_eq!(arena.graph_node_count(), 16);
    assert_eq!(arena.snapshot_lsn, lsn);
    assert_eq!(arena.last_applied_commit_lsn, lsn);
    assert!(arena.applied_deltas.is_empty());
    assert!(arena.bootstrap_vectors.is_empty());
    // Sanity: the delta source was queried at least once even with
    // no deltas (the loop pulls until None).
    assert!(deltas.pulls() >= 1);
}

#[test]
fn g3_recover_with_post_snapshot_wal_delta() {
    // Snapshot at LSN 1000 with N=16 vectors / 16 graph nodes; 4
    // post-snapshot delta pages each carrying one new vector. The
    // §4.6 sanity check uses the snapshot's graph_node_count + the
    // count of delta pages (one new vector per page in this test)
    // when the §4.5 staging contract holds. To keep the wire
    // contract simple we set graph_node_count = vector_count + delta
    // count so the post-replay sanity check passes.
    let dir = TempDir::new().unwrap();
    let tenant = TenantId::new(11);
    let req = VectorRecoveryRequest::v1(tenant, 1, IndexType::Hnsw, 8);
    let snapshot_lsn = Lsn::new(500);

    // Snapshot reports 16 vectors / 20 graph nodes — i.e., the
    // graph already includes the 4 to-be-replayed deltas (a v1.0
    // §4.5 invariant that delta pages don't grow graph_node_count
    // independently — the snapshot's graph header is the
    // authoritative count for this test).
    write_snapshot_file(
        dir.path(),
        &SnapshotBuilder::new(tenant, 1, snapshot_lsn).counts(16, 20),
    );

    let deltas = vec![
        VectorPageDelta {
            commit_lsn: Lsn::new(501),
            tenant_id: tenant,
            page_id: PageId::new(101),
            bytes: vec![0x01; 64],
        },
        VectorPageDelta {
            commit_lsn: Lsn::new(502),
            tenant_id: tenant,
            page_id: PageId::new(102),
            bytes: vec![0x02; 64],
        },
        VectorPageDelta {
            commit_lsn: Lsn::new(503),
            tenant_id: tenant,
            page_id: PageId::new(103),
            bytes: vec![0x03; 64],
        },
        VectorPageDelta {
            commit_lsn: Lsn::new(504),
            tenant_id: tenant,
            page_id: PageId::new(104),
            bytes: vec![0x04; 64],
        },
    ];

    let store = Arc::new(VectorArenaPageStore::new());
    let store_handle: Arc<dyn VectorPageStoreHandle> = Arc::clone(&store) as _;
    let wal = InMemoryWalDeltas::new(snapshot_lsn, deltas);
    let mvcc = InMemoryMvccSource::empty(snapshot_lsn);

    let arena = recover_arena(store_handle, dir.path(), &wal, &mvcc, req).expect("recover");

    assert_eq!(arena.applied_deltas.len(), 4);
    assert_eq!(arena.last_applied_commit_lsn, Lsn::new(504));
    assert_eq!(arena.vectors_count(), 16 + 4);
    assert_eq!(arena.graph_node_count(), 20);
    // Each delta page reaches the store via install_or_replace.
    for page in 101u64..=104 {
        let got = store
            .get_page(tenant, PageId::new(page))
            .expect("page installed");
        assert_eq!(got.len(), 64);
    }
}

// ─────────────────────────────────────────────────────────────────────
// 2. Missing / corrupt / partial snapshot fallback
// ─────────────────────────────────────────────────────────────────────

#[test]
fn g3_missing_snapshot_falls_back_to_bootstrap() {
    // No snapshot file in the dir → bootstrap. Recall: bootstrap
    // walks MvccVectorSource and produces an arena with
    // source=Bootstrap, vectors_count = MVCC walk count,
    // graph_node_count = 0 (graph rebuilt downstream).
    let dir = TempDir::new().unwrap();
    let tenant = TenantId::new(13);
    let req = VectorRecoveryRequest::v1(tenant, 1, IndexType::Hnsw, 8);
    let store: Arc<dyn VectorPageStoreHandle> = Arc::new(VectorArenaPageStore::new());
    let wal = InMemoryWalDeltas::empty(Lsn::ZERO);
    let mvcc_vectors: Vec<(u64, Vec<u8>)> = (0u64..7).map(|i| (i, vec![i as u8; 32])).collect();
    let mvcc = InMemoryMvccSource::new(Lsn::new(2000), mvcc_vectors);

    let arena = recover_arena(store, dir.path(), &wal, &mvcc, req).expect("bootstrap");

    assert_eq!(arena.source, ArenaSource::Bootstrap);
    assert_eq!(arena.vectors_count(), 7);
    assert_eq!(arena.graph_node_count(), 0);
    assert_eq!(arena.last_applied_commit_lsn, Lsn::new(2000));
    assert_eq!(arena.bootstrap_vectors.len(), 7);
}

#[test]
fn g3_crc_corruption_falls_back_to_bootstrap() {
    // Flip a byte inside the snapshot's CRC region (the trailing
    // file_crc32c slot) → CRC mismatch on load → fallback. The
    // recovery layer does NOT escalate; it warns and bootstraps.
    let dir = TempDir::new().unwrap();
    let tenant = TenantId::new(17);
    let req = VectorRecoveryRequest::v1(tenant, 5, IndexType::Hnsw, 8);
    let path = write_snapshot_file(
        dir.path(),
        &SnapshotBuilder::new(tenant, 5, Lsn::new(100)).counts(8, 8),
    );

    // Flip the last CRC byte.
    let mut bytes = fs::read(&path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0xFF;
    fs::write(&path, &bytes).unwrap();

    let store: Arc<dyn VectorPageStoreHandle> = Arc::new(VectorArenaPageStore::new());
    let wal = InMemoryWalDeltas::empty(Lsn::ZERO);
    // 5 MVCC vectors → bootstrap recall validation: post-bootstrap
    // count == 5. (For real deployments this validates ADR-035 §9.1
    // "recall maintained" — the bootstrap walk reproduces the
    // logical vector set, just slowly.)
    let mvcc = InMemoryMvccSource::new(
        Lsn::new(100),
        (0u64..5).map(|i| (i, vec![i as u8; 32])).collect(),
    );

    let arena = recover_arena(store, dir.path(), &wal, &mvcc, req).expect("bootstrap");
    assert_eq!(arena.source, ArenaSource::Bootstrap);
    assert_eq!(arena.vectors_count(), 5, "recall preserved post-bootstrap");
}

#[test]
fn g3_truncated_snapshot_falls_back_to_bootstrap() {
    // Truncate the snapshot mid-file (below the footer). Header
    // CRC may pass; trailing file CRC fails → fallback.
    let dir = TempDir::new().unwrap();
    let tenant = TenantId::new(19);
    let req = VectorRecoveryRequest::v1(tenant, 1, IndexType::Hnsw, 8);
    let path = write_snapshot_file(
        dir.path(),
        &SnapshotBuilder::new(tenant, 1, Lsn::new(200)).counts(8, 8),
    );

    // Cut the file in half.
    let bytes = fs::read(&path).unwrap();
    let half = bytes.len() / 2;
    fs::write(&path, &bytes[..half]).unwrap();

    let store: Arc<dyn VectorPageStoreHandle> = Arc::new(VectorArenaPageStore::new());
    let wal = InMemoryWalDeltas::empty(Lsn::ZERO);
    let mvcc = InMemoryMvccSource::new(
        Lsn::new(200),
        (0u64..3).map(|i| (i, vec![i as u8; 32])).collect(),
    );

    let arena = recover_arena(store, dir.path(), &wal, &mvcc, req).expect("bootstrap");
    assert_eq!(arena.source, ArenaSource::Bootstrap);
    assert_eq!(arena.vectors_count(), 3);
}

#[test]
fn g3_partial_section_falls_back() {
    // Snapshot whose header is byte-perfect but graph section
    // header is missing (zeroed instead of "HNSW" magic). The
    // section-bounds check inside `load_snapshot_file` does NOT
    // catch this on its own (the bytes are inside the file); the
    // graph-magic check after section-bounds validation does.
    let dir = TempDir::new().unwrap();
    let tenant = TenantId::new(23);
    let req = VectorRecoveryRequest::v1(tenant, 1, IndexType::Hnsw, 8);
    let mut builder = SnapshotBuilder::new(tenant, 1, Lsn::new(300)).counts(8, 8);
    builder = builder.omit_graph_header();
    let _path = write_snapshot_file(dir.path(), &builder);

    let store: Arc<dyn VectorPageStoreHandle> = Arc::new(VectorArenaPageStore::new());
    let wal = InMemoryWalDeltas::empty(Lsn::ZERO);
    let mvcc = InMemoryMvccSource::new(
        Lsn::new(300),
        (0u64..2).map(|i| (i, vec![i as u8; 32])).collect(),
    );

    let arena = recover_arena(store, dir.path(), &wal, &mvcc, req).expect("bootstrap");
    assert_eq!(arena.source, ArenaSource::Bootstrap);
    assert_eq!(arena.vectors_count(), 2);
}

// ─────────────────────────────────────────────────────────────────────
// 3. Mismatch sanity check halt
// ─────────────────────────────────────────────────────────────────────

#[test]
fn g3_arena_count_mismatch_halts_replay() {
    // Snapshot reports 10 vectors but graph header reports 8 nodes
    // → §4.6 step 4 sanity check fails → halt with
    // VectorIndexInconsistency, NOT bootstrap fallback (this is a
    // ship-blocking correctness violation per ADR-035 §4.6).
    let dir = TempDir::new().unwrap();
    let tenant = TenantId::new(29);
    let req = VectorRecoveryRequest::v1(tenant, 7, IndexType::Hnsw, 8);
    write_snapshot_file(
        dir.path(),
        &SnapshotBuilder::new(tenant, 7, Lsn::new(500)).counts(10, 8),
    );

    let store: Arc<dyn VectorPageStoreHandle> = Arc::new(VectorArenaPageStore::new());
    let wal = InMemoryWalDeltas::empty(Lsn::new(500));
    let mvcc = InMemoryMvccSource::empty(Lsn::new(500));

    let err = recover_arena(store, dir.path(), &wal, &mvcc, req).expect_err("must halt");
    match err {
        ArcGraphError::VectorIndexInconsistency {
            tenant_id,
            index_id,
            snapshot_lsn,
            observed_vectors_count,
            observed_graph_node_count,
            wal_replay_high_lsn,
            delta,
        } => {
            assert_eq!(tenant_id, 29);
            assert_eq!(index_id, 7);
            assert_eq!(snapshot_lsn, 500);
            assert_eq!(observed_vectors_count, 10);
            assert_eq!(observed_graph_node_count, 8);
            assert_eq!(wal_replay_high_lsn, 500);
            assert_eq!(delta, 2);
        }
        other => panic!("expected VectorIndexInconsistency, got {other:?}"),
    }
}

#[test]
fn g3_inconsistency_diagnostics_actionable() {
    // The error's Display impl is operator-actionable: it names
    // tenant + index + snapshot_lsn + the count delta + the WAL
    // high-water LSN. All five fields are required for an
    // operator to triage a `VectorIndexInconsistency` page-fire.
    let dir = TempDir::new().unwrap();
    let tenant = TenantId::new(31);
    let req = VectorRecoveryRequest::v1(tenant, 13, IndexType::DiskAnn, 16);
    write_snapshot_file(
        dir.path(),
        &SnapshotBuilder::new(tenant, 13, Lsn::new(777))
            .index_type(IndexType::DiskAnn)
            .dim(16)
            .counts(50, 51),
    );

    let store: Arc<dyn VectorPageStoreHandle> = Arc::new(VectorArenaPageStore::new());
    let wal = InMemoryWalDeltas::empty(Lsn::new(777));
    let mvcc = InMemoryMvccSource::empty(Lsn::new(777));

    let err = recover_arena(store, dir.path(), &wal, &mvcc, req).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("tenant=31"), "msg: {msg}");
    assert!(msg.contains("index=13"), "msg: {msg}");
    assert!(msg.contains("snapshot_lsn=777"), "msg: {msg}");
    assert!(msg.contains("vectors_count=50"), "msg: {msg}");
    assert!(msg.contains("graph_node_count=51"), "msg: {msg}");
    assert!(msg.contains("delta=-1"), "msg: {msg}");
    assert!(msg.contains("bootstrap_from_mvcc"), "msg: {msg}");
}

// ─────────────────────────────────────────────────────────────────────
// 4. Crash mid-recovery (process restart)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn g3_crash_mid_recovery_resumes_correctly() {
    // Run recovery once successfully (the "first restart"). Then
    // simulate a crash mid-recovery by tearing down the store +
    // re-running recovery with the same snapshot + WAL deltas. Per
    // Lemma I1 + I2, the second pass produces a byte-identical
    // post-state.
    let dir = TempDir::new().unwrap();
    let tenant = TenantId::new(37);
    let req = VectorRecoveryRequest::v1(tenant, 1, IndexType::Hnsw, 8);
    let snapshot_lsn = Lsn::new(1000);

    write_snapshot_file(
        dir.path(),
        &SnapshotBuilder::new(tenant, 1, snapshot_lsn).counts(8, 10),
    );
    let deltas_seed = vec![
        VectorPageDelta {
            commit_lsn: Lsn::new(1001),
            tenant_id: tenant,
            page_id: PageId::new(1),
            bytes: vec![0xAA; 64],
        },
        VectorPageDelta {
            commit_lsn: Lsn::new(1002),
            tenant_id: tenant,
            page_id: PageId::new(2),
            bytes: vec![0xBB; 64],
        },
    ];

    // First recovery pass — completes.
    let store_a = Arc::new(VectorArenaPageStore::new());
    let store_a_handle: Arc<dyn VectorPageStoreHandle> = Arc::clone(&store_a) as _;
    let wal_a = InMemoryWalDeltas::new(snapshot_lsn, deltas_seed.clone());
    let mvcc_a = InMemoryMvccSource::empty(snapshot_lsn);
    let a = recover_arena(store_a_handle, dir.path(), &wal_a, &mvcc_a, req).unwrap();

    // Second pass — simulates restart-after-crash. Same snapshot
    // file, same WAL deltas. Should produce identical state.
    let store_b = Arc::new(VectorArenaPageStore::new());
    let store_b_handle: Arc<dyn VectorPageStoreHandle> = Arc::clone(&store_b) as _;
    let wal_b = InMemoryWalDeltas::new(snapshot_lsn, deltas_seed);
    let mvcc_b = InMemoryMvccSource::empty(snapshot_lsn);
    let b = recover_arena(store_b_handle, dir.path(), &wal_b, &mvcc_b, req).unwrap();

    assert_eq!(a.tenant_id, b.tenant_id);
    assert_eq!(a.index_id, b.index_id);
    assert_eq!(a.snapshot_lsn, b.snapshot_lsn);
    assert_eq!(a.last_applied_commit_lsn, b.last_applied_commit_lsn);
    assert_eq!(a.vectors_count(), b.vectors_count());
    assert_eq!(a.graph_node_count(), b.graph_node_count());
    assert_eq!(a.applied_deltas.len(), b.applied_deltas.len());
    // Page bytes byte-identical post-second-pass.
    for delta in &b.applied_deltas {
        let stored = store_b.get_page(tenant, delta.page_id).unwrap();
        assert_eq!(stored, delta.bytes);
    }
}

// ─────────────────────────────────────────────────────────────────────
// 5. Multi-tenant mixed encodings
// ─────────────────────────────────────────────────────────────────────

#[test]
fn g3_recover_4_tenants_mixed_encodings() {
    // 4 tenants, mixed encodings F32 / SQ8 / Binary / SQ8. Recover
    // all 4 sequentially; verify per-tenant isolation (no cross-
    // tenant leakage) and per-tenant correctness of encoding +
    // metadata.
    let dir = TempDir::new().unwrap();
    let tenants = [
        (TenantId::new(101), Encoding::F32),
        (TenantId::new(102), Encoding::Sq8),
        (TenantId::new(103), Encoding::Binary),
        (TenantId::new(104), Encoding::Sq8),
    ];
    for (i, (tenant, enc)) in tenants.iter().enumerate() {
        let lsn = Lsn::new(1_000 + i as u64);
        write_snapshot_file(
            dir.path(),
            &SnapshotBuilder::new(*tenant, 1, lsn)
                .encoding(*enc)
                .dim(if matches!(enc, Encoding::Binary) {
                    128
                } else {
                    8
                })
                .counts(4 + i as u32, 4 + i as u32),
        );
    }

    let store: Arc<dyn VectorPageStoreHandle> = Arc::new(VectorArenaPageStore::new());
    let mut recovered = Vec::new();
    for (i, (tenant, enc)) in tenants.iter().enumerate() {
        let dim = if matches!(enc, Encoding::Binary) {
            128
        } else {
            8
        };
        let req = VectorRecoveryRequest::v1(*tenant, 1, IndexType::Hnsw, dim);
        let wal = InMemoryWalDeltas::empty(Lsn::new(1_000 + i as u64));
        let mvcc = InMemoryMvccSource::empty(Lsn::new(1_000 + i as u64));
        let arena = recover_arena(Arc::clone(&store), dir.path(), &wal, &mvcc, req)
            .expect("per-tenant recover");
        recovered.push(arena);
    }

    assert_eq!(recovered.len(), 4);
    for (i, (tenant, enc)) in tenants.iter().enumerate() {
        let a = &recovered[i];
        assert_eq!(a.tenant_id, *tenant, "tenant isolation");
        assert_eq!(a.encoding, *enc, "encoding preserved");
        assert_eq!(a.vectors_count(), 4 + i as u64);
        assert_eq!(a.snapshot_lsn, Lsn::new(1_000 + i as u64));
        assert_eq!(a.source, ArenaSource::Snapshot);
    }
    // Cross-tenant leakage check: encoding never bled across.
    let encs: Vec<_> = recovered.iter().map(|a| a.encoding).collect();
    assert_eq!(
        encs,
        vec![
            Encoding::F32,
            Encoding::Sq8,
            Encoding::Binary,
            Encoding::Sq8
        ]
    );
}

// ─────────────────────────────────────────────────────────────────────
// 6. Bootstrap halt on corrupt MVCC
// ─────────────────────────────────────────────────────────────────────

#[test]
fn g3_bootstrap_failure_halts_with_error() {
    // No snapshot exists → bootstrap is invoked. The
    // MvccVectorSource is rigged to return an error on the first
    // `next_vector` call (synthetic MVCC corruption). Per ADR-032
    // Slice 3c the error surfaces uplevel; recovery does NOT
    // silently degrade.
    let dir = TempDir::new().unwrap();
    let tenant = TenantId::new(43);
    let req = VectorRecoveryRequest::v1(tenant, 1, IndexType::Hnsw, 8);
    let store: Arc<dyn VectorPageStoreHandle> = Arc::new(VectorArenaPageStore::new());
    let wal = InMemoryWalDeltas::empty(Lsn::ZERO);
    let mvcc = InMemoryMvccSource::with_error(ArcGraphError::WalCorruption {
        lsn: Lsn::ZERO,
        reason: "synthetic mvcc corruption".to_owned(),
    });

    let err = recover_arena(store, dir.path(), &wal, &mvcc, req).unwrap_err();
    match err {
        ArcGraphError::WalCorruption { reason, .. } => {
            assert!(reason.contains("synthetic"), "got: {reason}");
        }
        other => panic!("expected WalCorruption, got {other:?}"),
    }
}

// ─────────────────────────────────────────────────────────────────────
// 7. Cleanup of stale snapshots
// ─────────────────────────────────────────────────────────────────────

#[test]
fn g3_recover_cleans_up_old_snapshots() {
    // Three generations + one orphan .tmp. Recovery loads the
    // newest, then removes the older two AND the orphan. Per
    // ADR-035 §4.6 step 5 + §4.5 step 7 (snapshot writer also
    // GCs every two generations; recovery's pass is the catch-up
    // for files left over from a crash mid-flush).
    let dir = TempDir::new().unwrap();
    let tenant = TenantId::new(53);
    let req = VectorRecoveryRequest::v1(tenant, 1, IndexType::Hnsw, 8);

    write_snapshot_file(
        dir.path(),
        &SnapshotBuilder::new(tenant, 1, Lsn::new(10)).counts(4, 4),
    );
    write_snapshot_file(
        dir.path(),
        &SnapshotBuilder::new(tenant, 1, Lsn::new(20)).counts(8, 8),
    );
    write_snapshot_file(
        dir.path(),
        &SnapshotBuilder::new(tenant, 1, Lsn::new(30)).counts(12, 12),
    );

    // Orphan .tmp from a crashed flush mid-rename.
    let orphan_path = dir.path().join("arena-53-1-25.snap.tmp");
    fs::write(&orphan_path, b"orphan-bytes").unwrap();

    let store: Arc<dyn VectorPageStoreHandle> = Arc::new(VectorArenaPageStore::new());
    let wal = InMemoryWalDeltas::empty(Lsn::new(30));
    let mvcc = InMemoryMvccSource::empty(Lsn::new(30));

    let arena = recover_arena(store, dir.path(), &wal, &mvcc, req).unwrap();
    assert_eq!(arena.snapshot_lsn, Lsn::new(30));

    // Post-cleanup: only the newest snap remains; the .tmp orphan
    // is gone.
    let entries: Vec<String> = fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert!(
        entries.contains(&"arena-53-1-30.snap".to_owned()),
        "newest survives: {entries:?}"
    );
    assert!(
        !entries.contains(&"arena-53-1-10.snap".to_owned()),
        "lsn=10 removed: {entries:?}"
    );
    assert!(
        !entries.contains(&"arena-53-1-20.snap".to_owned()),
        "lsn=20 removed: {entries:?}"
    );
    assert!(
        !entries.contains(&"arena-53-1-25.snap.tmp".to_owned()),
        "orphan tmp removed: {entries:?}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// 8. Z-1 (b) rollback integration with vector arena pages
// ─────────────────────────────────────────────────────────────────────

#[test]
fn g3_z1_rollback_with_vector_arena_pages() {
    // ADR-033 Z-1 (b) integration: extend the existing rollback
    // pattern to vector arena pages. The TxnMutationLog can carry
    // (PageStoreKind::Vector, page_id, pre_W_bytes) entries; on
    // WAL fsync failure the rollback drainer dispatches to the
    // VectorPageStoreHandle::restore_page_bytes hook.
    //
    // Slice G.5 wires the crud.rs dispatch arm; G.3 owns the
    // trait-level contract: restore_page_bytes is byte-overwrite
    // (matching install_or_replace) so a double-call is idempotent.
    //
    // This test exercises the trait directly. It synthesizes the
    // mutation log entries that G.5's rollback drainer will produce
    // and verifies the per-page restore semantics: pre-W bytes are
    // restored exactly, and a second call (idempotent) leaves the
    // store in the same state.
    let store = VectorArenaPageStore::new();
    let tenant = TenantId::new(59);

    // Pre-W state: page 1 holds bytes X. The transaction's builder
    // mutated it to Y. WAL fsync fails. Rollback drains
    // page_mutations and calls restore_page_bytes(tenant, 1, X).
    let pre_w = vec![0xAA; 64];
    store
        .install_or_replace(tenant, PageId::new(1), &pre_w)
        .unwrap();

    // Builder phase: the transaction mutated the page in place
    // (would have called `install_or_replace` with the post-W
    // bytes via a builder helper). Capture the pre-W state into a
    // mutation log:
    let mut log = TxnMutationLog::new();
    let mut pre_w_buf = [0u8; PAGE_SIZE];
    pre_w_buf[..pre_w.len()].copy_from_slice(&pre_w);
    log.page_mutations
        .push((PageStoreKind::Vector, PageId::new(1), Box::new(pre_w_buf)));

    // Now mutate to post-W (simulate the builder's in-place edit).
    store
        .install_or_replace(tenant, PageId::new(1), &[0xBB; 64])
        .unwrap();
    assert_eq!(
        store.get_page(tenant, PageId::new(1)).unwrap(),
        vec![0xBB; 64]
    );

    // WAL fsync fails. Rollback drains the log: dispatches each
    // (PageStoreKind::Vector, page_id, pre_w) to
    // VectorPageStoreHandle::restore_page_bytes. We do this
    // explicitly here because Slice G.5 owns the production crud.rs
    // dispatch arm.
    for (kind, pid, bytes) in log.page_mutations.drain(..) {
        match kind {
            PageStoreKind::Vector => {
                store
                    .restore_page_bytes(tenant, pid, bytes.as_ref())
                    .unwrap();
            }
            other => panic!("unexpected kind {other:?}"),
        }
    }

    // Post-rollback: page 1 holds the pre-W bytes again (with the
    // PAGE_SIZE-padded width because the mutation log uses fixed-
    // size buffers).
    let restored = store.get_page(tenant, PageId::new(1)).unwrap();
    assert_eq!(restored.len(), PAGE_SIZE);
    assert_eq!(&restored[..pre_w.len()], &pre_w[..]);

    // Idempotence: a second restore call leaves state unchanged.
    let mut buf = [0u8; PAGE_SIZE];
    buf[..pre_w.len()].copy_from_slice(&pre_w);
    store
        .restore_page_bytes(tenant, PageId::new(1), &buf)
        .unwrap();
    let restored_again = store.get_page(tenant, PageId::new(1)).unwrap();
    assert_eq!(restored, restored_again);
}

// ─────────────────────────────────────────────────────────────────────
// 9. Local partition invariant
// ─────────────────────────────────────────────────────────────────────

#[test]
fn g3_recover_request_partition_id_always_zero_at_v1() {
    // Mirrors the Z-1 (b) `z1_partition_id_always_zero_at_v1`
    // regression: the constructor must produce partition_id == ZERO.
    let req = VectorRecoveryRequest::v1(TenantId::DEFAULT, 1, IndexType::Hnsw, 8);
    assert_eq!(req.partition_id, PartitionId::ZERO);
    assert_eq!(req.partition_id.raw(), 0);
}

// ─────────────────────────────────────────────────────────────────────
// 10. bootstrap_from_mvcc smoke (cold start path)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn g3_bootstrap_from_mvcc_cold_start() {
    // Direct call to bootstrap_from_mvcc (without going through
    // recover_arena's snapshot lookup) — the cold-start path the
    // operator triggers via ADR-035 §9.1 manual rebuild.
    let store: Arc<dyn VectorPageStoreHandle> = Arc::new(VectorArenaPageStore::new());
    let mvcc = InMemoryMvccSource::new(
        Lsn::new(123),
        (0u64..16).map(|i| (i, vec![i as u8; 32])).collect(),
    );
    let req = VectorRecoveryRequest::v1(TenantId::new(67), 1, IndexType::Hnsw, 8);
    let arena = bootstrap_from_mvcc(store, &mvcc, req).unwrap();
    assert_eq!(arena.source, ArenaSource::Bootstrap);
    assert_eq!(arena.vectors_count(), 16);
    assert_eq!(arena.bootstrap_vectors.len(), 16);
    assert_eq!(arena.snapshot_lsn, Lsn::ZERO);
    assert_eq!(arena.last_applied_commit_lsn, Lsn::new(123));
}

// ─────────────────────────────────────────────────────────────────────
// 11. Pre-snapshot deltas are filtered (idempotence guard)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn g3_pre_snapshot_deltas_skipped() {
    // Per ADR-035 §4.5 last_snapshot_high_water contract: a delta
    // whose commit_lsn ≤ snapshot_lsn is already in the snapshot
    // and MUST be skipped at recovery — replaying it would
    // double-install (idempotent under Lemma I2 but pathology-
    // class wasted work). The recovery loop self-defends against
    // a misfiltered WAL delta source.
    let dir = TempDir::new().unwrap();
    let tenant = TenantId::new(71);
    let req = VectorRecoveryRequest::v1(tenant, 1, IndexType::Hnsw, 8);
    let snapshot_lsn = Lsn::new(500);

    write_snapshot_file(
        dir.path(),
        &SnapshotBuilder::new(tenant, 1, snapshot_lsn).counts(4, 4),
    );

    let deltas = vec![
        // Pre-snapshot — must be skipped.
        VectorPageDelta {
            commit_lsn: Lsn::new(499),
            tenant_id: tenant,
            page_id: PageId::new(99),
            bytes: vec![0xFF; 64],
        },
        VectorPageDelta {
            commit_lsn: snapshot_lsn,
            tenant_id: tenant,
            page_id: PageId::new(100),
            bytes: vec![0xFE; 64],
        },
    ];

    let store = Arc::new(VectorArenaPageStore::new());
    let store_handle: Arc<dyn VectorPageStoreHandle> = Arc::clone(&store) as _;
    let wal = InMemoryWalDeltas::new(snapshot_lsn, deltas);
    let mvcc = InMemoryMvccSource::empty(snapshot_lsn);

    let arena = recover_arena(store_handle, dir.path(), &wal, &mvcc, req).unwrap();
    // No applied deltas — both filtered as ≤ snapshot_lsn.
    assert!(arena.applied_deltas.is_empty());
    // Pages weren't installed because the deltas were skipped.
    assert!(store.get_page(tenant, PageId::new(99)).is_none());
    assert!(store.get_page(tenant, PageId::new(100)).is_none());
}
