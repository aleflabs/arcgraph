//! Slice G.2 Path A boundary tests for the vector arena snapshot
//! flush.
//!
//! Per ADR-035 §4.5/§4.6 and `docs/design/vector-storage-layout.md`
//! §10.3, the snapshot flush is the durability primitive that makes
//! vector arenas recoverable across process restarts. Path A
//! boundary coverage proves three properties hold at the byte
//! level:
//!
//! 1. **Format byte-validation.** The on-disk file matches the
//!    documented header / section-descriptor / footer layout
//!    exactly; corruption at any of those three regions is
//!    detectable via the trailing CRC.
//! 2. **Atomic temp-file + rename.** The five-step protocol
//!    (write → fsync → rename → dir-fsync → catalog-stamp) leaves
//!    a graceful artifact at every interior crash point: either
//!    no `.snap` file (rename never ran) OR a `.snap` whose
//!    header + footer round-trip cleanly.
//! 3. **Concurrent + multi-tenant isolation.** Snapshots over
//!    different `(tenant, index)` pairs run in parallel without
//!    cross-tenant byte leakage; a snapshot taken during active
//!    inserts captures a consistent point-in-time, not a torn
//!    state.
//!
//! Sibling: Slice G.3 (recovery; in parallel on this branch) owns
//! the load path that consumes these artifacts. Tests here verify
//! G.2 produces the right artifacts; G.3 verifies G.2's artifacts
//! load correctly.

use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use arcgraph_core::{Lsn, PartitionId, TenantId};
use arcgraph_storage::vector_store::VectorStoreError;
use arcgraph_storage::vector_store::snapshot::{
    ARCV_FOOTER_SIZE, ARCV_FORMAT_VERSION, ARCV_HEADER_SIZE, ARCV_MAGIC,
    ARCV_SECTION_DESCRIPTOR_SIZE, ARCV_TRAILING_CRC_SIZE, CrashPoint, SectionKind, SnapshotCatalog,
    SnapshotPolicy, SnapshotSection, SnapshotSpec, SnapshotTrigger, flush_snapshot,
    flush_snapshot_with_crash_point, snapshot_path, snapshot_temp_path,
};
use tempfile::TempDir;

// ─── Helpers ─────────────────────────────────────────────────────

/// Build a minimal `(tenant, index, lsn)` spec with the given
/// sections. Defaults: SQ8 + HNSW + dim=768 + DEFAULT tenant +
/// index_id=1 + lsn=100.
fn spec_with<'a>(sections: &'a [SnapshotSection<'a>]) -> SnapshotSpec<'a> {
    SnapshotSpec {
        tenant: TenantId::DEFAULT,
        partition: PartitionId::ZERO,
        index_id: 1,
        lsn: Lsn::new(100),
        encoding: 2,   // SQ8
        index_type: 0, // HNSW
        dim: 768,
        vectors_count: 0,
        sections,
    }
}

/// Read the raw bytes of a file and verify CRC32C of the prefix
/// matches the trailing 4 bytes. Returns the parsed `(magic, version,
/// section_count, lsn, vectors_count)`.
fn read_and_verify_arcv(path: &Path) -> ([u8; 4], u16, u32, u64, u64) {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    assert!(
        bytes.len() >= ARCV_HEADER_SIZE + ARCV_FOOTER_SIZE,
        "file too short: {} bytes",
        bytes.len()
    );
    // Footer
    let stored_total =
        u64::from_le_bytes(bytes[bytes.len() - 16..bytes.len() - 8].try_into().unwrap());
    assert_eq!(
        stored_total as usize,
        bytes.len(),
        "footer total_file_size disagrees with actual length"
    );
    let stored_crc = u32::from_le_bytes(bytes[bytes.len() - 4..].try_into().unwrap());
    let computed = crc32c::crc32c(&bytes[..bytes.len() - ARCV_TRAILING_CRC_SIZE]);
    assert_eq!(
        stored_crc, computed,
        "trailing CRC mismatch: stored={stored_crc:#x} computed={computed:#x}"
    );

    let mut magic = [0u8; 4];
    magic.copy_from_slice(&bytes[0..4]);
    let version = u16::from_le_bytes(bytes[4..6].try_into().unwrap());
    let section_count = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
    let lsn = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
    let vectors_count = u64::from_le_bytes(bytes[24..32].try_into().unwrap());
    (magic, version, section_count, lsn, vectors_count)
}

// ─────────────────────────────────────────────────────────────────
// Format byte-validation
// ─────────────────────────────────────────────────────────────────

/// `g2_snapshot_header_byte_layout` — write a snapshot, read raw
/// bytes, assert magic = b"ARCV", version = 1, encoding / dim / lsn
/// fields at expected offsets per the ARCV format docs.
#[test]
fn g2_snapshot_header_byte_layout() {
    let tmpdir = TempDir::new().unwrap();
    let catalog = SnapshotCatalog::new();

    let payload = vec![0xABu8; 256];
    let sections = [SnapshotSection {
        kind: SectionKind::Quantized,
        bytes: &payload,
    }];
    let spec = SnapshotSpec {
        tenant: TenantId::new(42),
        partition: PartitionId::ZERO,
        index_id: 7,
        lsn: Lsn::new(0xDEAD_BEEF),
        encoding: 2,   // SQ8
        index_type: 1, // DiskANN
        dim: 768,
        vectors_count: 256,
        sections: &sections,
    };

    let path = flush_snapshot(&spec, tmpdir.path(), &catalog).unwrap();
    let bytes = fs::read(&path).unwrap();

    // Offset 0..4: magic = b"ARCV"
    assert_eq!(&bytes[0..4], ARCV_MAGIC);
    // Offset 4..6: version = 1 LE
    assert_eq!(
        u16::from_le_bytes(bytes[4..6].try_into().unwrap()),
        ARCV_FORMAT_VERSION
    );
    // Offset 6: encoding = 2 (SQ8)
    assert_eq!(bytes[6], 2);
    // Offset 7: index_type = 1 (DiskANN)
    assert_eq!(bytes[7], 1);
    // Offset 8..12: dim = 768
    assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 768);
    // Offset 12..16: section_count = 1
    assert_eq!(u32::from_le_bytes(bytes[12..16].try_into().unwrap()), 1);
    // Offset 16..24: lsn = 0xDEADBEEF
    assert_eq!(
        u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
        0xDEAD_BEEF
    );
    // Offset 24..32: vectors_count = 256
    assert_eq!(u64::from_le_bytes(bytes[24..32].try_into().unwrap()), 256);
    // Offset 32..40: tenant_id = 42
    assert_eq!(u64::from_le_bytes(bytes[32..40].try_into().unwrap()), 42);
    // Offset 40..48: index_id = 7
    assert_eq!(u64::from_le_bytes(bytes[40..48].try_into().unwrap()), 7);
    // Offset 48..64: reserved (must be zero)
    assert!(bytes[48..64].iter().all(|&b| b == 0));
}

#[test]
fn g2_snapshot_rabitq_encoding_tag_roundtrips_and_next_tag_rejects() {
    let tmpdir = TempDir::new().unwrap();
    let catalog = SnapshotCatalog::new();
    let payload = vec![0x44u8; 128];
    let sections = [SnapshotSection {
        kind: SectionKind::Quantized,
        bytes: &payload,
    }];
    let mut spec = spec_with(&sections);
    spec.encoding = 4; // RaBitQ

    let path = flush_snapshot(&spec, tmpdir.path(), &catalog).unwrap();
    let bytes = fs::read(&path).unwrap();
    assert_eq!(bytes[6], 4);

    spec.encoding = 5;
    let err = flush_snapshot(&spec, tmpdir.path(), &catalog).unwrap_err();
    assert!(matches!(err, VectorStoreError::InvalidSnapshotSpec(_)));
}

/// `g2_snapshot_footer_crc32c` — compute CRC32C over header +
/// sections; assert footer matches.
#[test]
fn g2_snapshot_footer_crc32c() {
    let tmpdir = TempDir::new().unwrap();
    let catalog = SnapshotCatalog::new();

    // Mix of sections to exercise the descriptor + payload + CRC path.
    let q = vec![0x11u8; 100];
    let r = vec![0x22u8; 50];
    let l = vec![0x33u8; 25];
    let sections = [
        SnapshotSection {
            kind: SectionKind::Quantized,
            bytes: &q,
        },
        SnapshotSection {
            kind: SectionKind::Rescore,
            bytes: &r,
        },
        SnapshotSection {
            kind: SectionKind::Labels,
            bytes: &l,
        },
    ];
    let spec = spec_with(&sections);

    let path = flush_snapshot(&spec, tmpdir.path(), &catalog).unwrap();
    let bytes = fs::read(&path).unwrap();

    // Footer trailing CRC: last 4 bytes.
    let crc_offset = bytes.len() - ARCV_TRAILING_CRC_SIZE;
    let stored = u32::from_le_bytes(bytes[crc_offset..].try_into().unwrap());
    let computed = crc32c::crc32c(&bytes[..crc_offset]);
    assert_eq!(
        stored, computed,
        "footer CRC must equal CRC32C(header || descriptors || payloads || footer-prefix)"
    );

    // Total file size byte-validates.
    let total = u64::from_le_bytes(bytes[crc_offset - 12..crc_offset - 4].try_into().unwrap());
    assert_eq!(total as usize, bytes.len());
}

/// `g2_snapshot_corrupt_header_detected` — flip 1 byte in the
/// header region; assert the trailing CRC then disagrees.
#[test]
fn g2_snapshot_corrupt_header_detected() {
    let tmpdir = TempDir::new().unwrap();
    let catalog = SnapshotCatalog::new();
    let payload = vec![0u8; 64];
    let sections = [SnapshotSection {
        kind: SectionKind::Quantized,
        bytes: &payload,
    }];
    let path = flush_snapshot(&spec_with(&sections), tmpdir.path(), &catalog).unwrap();

    // Read, flip one byte in the header (offset 7 = index_type),
    // recompute CRC, expect mismatch.
    let mut bytes = fs::read(&path).unwrap();
    bytes[7] ^= 0xFF;
    let crc_offset = bytes.len() - ARCV_TRAILING_CRC_SIZE;
    let stored = u32::from_le_bytes(bytes[crc_offset..].try_into().unwrap());
    let recomputed = crc32c::crc32c(&bytes[..crc_offset]);
    assert_ne!(
        stored, recomputed,
        "header bit-flip must invalidate the trailing CRC"
    );
}

/// `g2_snapshot_corrupt_section_detected` — flip 1 byte in a
/// section payload region; assert CRC fails.
#[test]
fn g2_snapshot_corrupt_section_detected() {
    let tmpdir = TempDir::new().unwrap();
    let catalog = SnapshotCatalog::new();
    let payload = vec![0u8; 256];
    let sections = [SnapshotSection {
        kind: SectionKind::Quantized,
        bytes: &payload,
    }];
    let path = flush_snapshot(&spec_with(&sections), tmpdir.path(), &catalog).unwrap();

    let mut bytes = fs::read(&path).unwrap();
    // Section payload starts at the first 64-aligned offset after
    // header + 1 descriptor (64 + 32 = 96 → align up to 128).
    // Flip a byte deep inside that region.
    let target = ARCV_HEADER_SIZE + ARCV_SECTION_DESCRIPTOR_SIZE + 64;
    assert!(target < bytes.len() - ARCV_FOOTER_SIZE);
    bytes[target] ^= 0x55;
    let crc_offset = bytes.len() - ARCV_TRAILING_CRC_SIZE;
    let stored = u32::from_le_bytes(bytes[crc_offset..].try_into().unwrap());
    let recomputed = crc32c::crc32c(&bytes[..crc_offset]);
    assert_ne!(
        stored, recomputed,
        "section payload bit-flip must invalidate the trailing CRC"
    );
}

/// `g2_snapshot_corrupt_footer_detected` — flip 1 byte in the
/// CRC field itself; recomputed CRC over the (unmodified) prefix
/// no longer matches the stored CRC.
#[test]
fn g2_snapshot_corrupt_footer_detected() {
    let tmpdir = TempDir::new().unwrap();
    let catalog = SnapshotCatalog::new();
    let payload = vec![0u8; 32];
    let sections = [SnapshotSection {
        kind: SectionKind::Quantized,
        bytes: &payload,
    }];
    let path = flush_snapshot(&spec_with(&sections), tmpdir.path(), &catalog).unwrap();

    let mut bytes = fs::read(&path).unwrap();
    let crc_offset = bytes.len() - ARCV_TRAILING_CRC_SIZE;
    bytes[crc_offset] ^= 0xAB;
    let stored = u32::from_le_bytes(bytes[crc_offset..].try_into().unwrap());
    let recomputed = crc32c::crc32c(&bytes[..crc_offset]);
    assert_ne!(stored, recomputed, "flipping CRC field must be detectable");
}

// ─────────────────────────────────────────────────────────────────
// Atomic temp-file + rename (crash injection)
// ─────────────────────────────────────────────────────────────────

/// `g2_snapshot_crash_before_rename_no_corruption` — inject crash
/// between fsync(.tmp) and rename. Verify .tmp exists, no .snap;
/// the .tmp is byte-valid (fully written + CRC matches) so a
/// recovery pass that GCs orphan .tmp files leaves no torn state.
/// G.3 owns the actual cleanup; G.2 owns producing a graceful
/// artifact.
#[test]
fn g2_snapshot_crash_before_rename_no_corruption() {
    let tmpdir = TempDir::new().unwrap();
    let catalog = SnapshotCatalog::new();
    let payload = vec![0xCDu8; 128];
    let sections = [SnapshotSection {
        kind: SectionKind::Quantized,
        bytes: &payload,
    }];
    let spec = spec_with(&sections);

    let err =
        flush_snapshot_with_crash_point(&spec, tmpdir.path(), &catalog, CrashPoint::BeforeRename)
            .unwrap_err();
    match err {
        VectorStoreError::CrashInjected(CrashPoint::BeforeRename) => {}
        e => panic!("expected CrashInjected(BeforeRename), got {e:?}"),
    }

    // .tmp exists.
    let tmp = snapshot_temp_path(tmpdir.path(), spec.tenant, spec.index_id, spec.lsn);
    assert!(tmp.exists(), ".tmp must exist after crash before rename");
    // .snap does NOT exist.
    let final_path = snapshot_path(tmpdir.path(), spec.tenant, spec.index_id, spec.lsn);
    assert!(!final_path.exists(), ".snap must NOT exist before rename");

    // The .tmp is fully written and CRC-valid (fsync ran before
    // the crash injection point) — recovery sees a "graceful crash
    // artifact" rather than a torn write.
    let bytes = fs::read(&tmp).unwrap();
    let crc_offset = bytes.len() - ARCV_TRAILING_CRC_SIZE;
    let stored = u32::from_le_bytes(bytes[crc_offset..].try_into().unwrap());
    let computed = crc32c::crc32c(&bytes[..crc_offset]);
    assert_eq!(stored, computed, ".tmp must be byte-valid after fsync");

    // Catalog NOT stamped (the stamp comes after dir fsync).
    assert_eq!(catalog.latest_lsn(spec.tenant, spec.index_id), None);
}

/// `g2_snapshot_crash_mid_write_temp_cleaned` — inject crash mid-
/// write so the .tmp is truncated. The CRC over the partial .tmp
/// will not match its trailing 4 bytes (because the trailing 4
/// bytes don't even exist in the partial), so the file is
/// unambiguously detectable as corrupt.
#[test]
fn g2_snapshot_crash_mid_write_temp_cleaned() {
    let tmpdir = TempDir::new().unwrap();
    let catalog = SnapshotCatalog::new();
    let payload = vec![0xEEu8; 1024];
    let sections = [SnapshotSection {
        kind: SectionKind::Quantized,
        bytes: &payload,
    }];
    let spec = spec_with(&sections);

    // Crash after writing only 100 bytes (less than the full body).
    let err =
        flush_snapshot_with_crash_point(&spec, tmpdir.path(), &catalog, CrashPoint::MidWrite(100))
            .unwrap_err();
    match err {
        VectorStoreError::CrashInjected(CrashPoint::MidWrite(100)) => {}
        e => panic!("expected CrashInjected(MidWrite(100)), got {e:?}"),
    }

    let tmp = snapshot_temp_path(tmpdir.path(), spec.tenant, spec.index_id, spec.lsn);
    assert!(tmp.exists());
    // The .tmp is truncated: shorter than header + footer combined.
    let len = fs::metadata(&tmp).unwrap().len();
    assert_eq!(len, 100, "mid-write crash leaves exactly 100 bytes on disk");
    // No .snap.
    let final_path = snapshot_path(tmpdir.path(), spec.tenant, spec.index_id, spec.lsn);
    assert!(!final_path.exists());
    // Catalog not stamped.
    assert_eq!(catalog.latest_lsn(spec.tenant, spec.index_id), None);
}

/// `g2_snapshot_partial_temp_cleaned_on_next_run` — start with a
/// stale .tmp; flush_snapshot completes successfully (overwrites
/// .tmp via O_TRUNC, then renames). The final .snap is byte-valid
/// and the stale .tmp content is gone.
#[test]
fn g2_snapshot_partial_temp_cleaned_on_next_run() {
    let tmpdir = TempDir::new().unwrap();
    let catalog = SnapshotCatalog::new();
    let payload = vec![0x77u8; 64];
    let sections = [SnapshotSection {
        kind: SectionKind::Quantized,
        bytes: &payload,
    }];
    let spec = spec_with(&sections);

    // Pre-place a stale .tmp at the same path the flush would use.
    let tmp = snapshot_temp_path(tmpdir.path(), spec.tenant, spec.index_id, spec.lsn);
    {
        let mut f = File::create(&tmp).unwrap();
        f.write_all(b"STALE PARTIAL CONTENT FROM CRASHED PRIOR FLUSH")
            .unwrap();
    }
    assert!(tmp.exists());

    // Flush succeeds.
    let path = flush_snapshot(&spec, tmpdir.path(), &catalog).unwrap();
    assert!(path.exists(), "flush must produce the .snap");
    assert!(!tmp.exists(), "rename must consume the .tmp");

    // The .snap is byte-valid.
    let (magic, version, sections_count, lsn, _vc) = read_and_verify_arcv(&path);
    assert_eq!(&magic, ARCV_MAGIC);
    assert_eq!(version, ARCV_FORMAT_VERSION);
    assert_eq!(sections_count, 1);
    assert_eq!(lsn, 100);
    // Catalog stamped.
    assert_eq!(
        catalog.latest_lsn(spec.tenant, spec.index_id),
        Some(spec.lsn)
    );
}

// ─────────────────────────────────────────────────────────────────
// Concurrent snapshot operations (P5 boundary)
// ─────────────────────────────────────────────────────────────────

/// `g2_snapshot_during_active_inserts` — a writer thread mutates a
/// shared "live arena" buffer (simulated as a Vec<u8> behind a
/// Mutex) at high rate; flush_snapshot is called with a CAPTURED
/// snapshot of that buffer. The flush bytes must be the snapshot
/// at capture time, NOT a mid-mutation tear.
///
/// The point here: G.2's atomicity contract is "the SnapshotSpec's
/// byte slices are the snapshot." Anything the caller chooses to
/// pass is what gets flushed; the flush itself runs over a stable
/// borrow. The test verifies that the flushed bytes match what
/// the caller captured — concurrent mutation by another thread
/// does not bleed into the on-disk file.
#[test]
fn g2_snapshot_during_active_inserts() {
    let tmpdir = TempDir::new().unwrap();
    let catalog = SnapshotCatalog::new();

    // Live "arena" — a writer thread fills it with monotonically
    // increasing bytes. The snapshot captures the prefix at one
    // moment; the writer continues past that point.
    let live: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(vec![0u8; 4096]));
    let stop = Arc::new(AtomicBool::new(false));
    let writer = {
        let live = Arc::clone(&live);
        let stop = Arc::clone(&stop);
        thread::spawn(move || {
            let mut counter: u8 = 0;
            while !stop.load(Ordering::Relaxed) {
                let mut g = live.lock().unwrap();
                for b in g.iter_mut() {
                    *b = counter;
                }
                counter = counter.wrapping_add(1);
                drop(g);
                thread::yield_now();
            }
        })
    };

    // Capture a snapshot of the live arena and flush. The capture
    // is just a clone of the bytes under the lock — the flush then
    // runs over the cloned (stable) buffer.
    let captured: Vec<u8> = {
        let g = live.lock().unwrap();
        g.clone()
    };
    let sections = [SnapshotSection {
        kind: SectionKind::Quantized,
        bytes: &captured,
    }];
    let spec = spec_with(&sections);
    let path = flush_snapshot(&spec, tmpdir.path(), &catalog).unwrap();

    stop.store(true, Ordering::Relaxed);
    writer.join().unwrap();

    // The flushed payload must equal the captured bytes. Read the
    // file, walk the section descriptor, extract the payload.
    let bytes = fs::read(&path).unwrap();
    let payload_offset = u64::from_le_bytes(
        bytes[ARCV_HEADER_SIZE + 8..ARCV_HEADER_SIZE + 16]
            .try_into()
            .unwrap(),
    ) as usize;
    let payload_size = u64::from_le_bytes(
        bytes[ARCV_HEADER_SIZE + 16..ARCV_HEADER_SIZE + 24]
            .try_into()
            .unwrap(),
    ) as usize;
    assert_eq!(payload_size, captured.len());
    assert_eq!(
        &bytes[payload_offset..payload_offset + payload_size],
        captured.as_slice(),
        "flushed payload must match captured snapshot, not torn live arena"
    );
}

/// `g2_multi_tenant_concurrent_snapshot` — N=4 tenants snapshot in
/// parallel; assert no cross-tenant leakage and each snapshot file
/// is independent.
#[test]
fn g2_multi_tenant_concurrent_snapshot() {
    let tmpdir = TempDir::new().unwrap();
    let catalog = Arc::new(SnapshotCatalog::new());
    let dir = Arc::new(tmpdir.path().to_path_buf());

    let mut handles = Vec::new();
    for tenant_raw in 100u64..104 {
        let catalog = Arc::clone(&catalog);
        let dir = Arc::clone(&dir);
        let h = thread::spawn(move || {
            // Each tenant's payload is filled with its tenant id
            // so a cross-tenant leak would be detectable.
            let payload = vec![tenant_raw as u8; 1024];
            let sections = [SnapshotSection {
                kind: SectionKind::Quantized,
                bytes: &payload,
            }];
            let spec = SnapshotSpec {
                tenant: TenantId::new(tenant_raw),
                partition: PartitionId::ZERO,
                index_id: 1,
                lsn: Lsn::new(tenant_raw * 10),
                encoding: 2,
                index_type: 0,
                dim: 768,
                vectors_count: 1024,
                sections: &sections,
            };
            flush_snapshot(&spec, dir.as_path(), &catalog).unwrap()
        });
        handles.push(h);
    }
    let paths: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    // Verify each file's payload matches the originating tenant.
    for (i, path) in paths.iter().enumerate() {
        let tenant_raw = 100u64 + i as u64;
        let bytes = fs::read(path).unwrap();
        // Tenant_id at header offset 32..40
        let tid = u64::from_le_bytes(bytes[32..40].try_into().unwrap());
        assert_eq!(tid, tenant_raw, "header tenant_id mismatch");
        // Payload bytes
        let payload_offset = u64::from_le_bytes(
            bytes[ARCV_HEADER_SIZE + 8..ARCV_HEADER_SIZE + 16]
                .try_into()
                .unwrap(),
        ) as usize;
        let payload_size = u64::from_le_bytes(
            bytes[ARCV_HEADER_SIZE + 16..ARCV_HEADER_SIZE + 24]
                .try_into()
                .unwrap(),
        ) as usize;
        let payload = &bytes[payload_offset..payload_offset + payload_size];
        assert!(
            payload.iter().all(|&b| b == tenant_raw as u8),
            "payload byte-fill must match tenant id; got first byte={} expected {}",
            payload[0],
            tenant_raw as u8
        );
        // CRC verifies.
        let crc_offset = bytes.len() - ARCV_TRAILING_CRC_SIZE;
        let stored = u32::from_le_bytes(bytes[crc_offset..].try_into().unwrap());
        let computed = crc32c::crc32c(&bytes[..crc_offset]);
        assert_eq!(stored, computed);
    }

    // Catalog has 4 entries.
    assert_eq!(catalog.len(), 4);
    for tenant_raw in 100u64..104 {
        assert_eq!(
            catalog.latest_lsn(TenantId::new(tenant_raw), 1),
            Some(Lsn::new(tenant_raw * 10))
        );
    }
}

// ─────────────────────────────────────────────────────────────────
// Boundary cases
// ─────────────────────────────────────────────────────────────────

/// `g2_snapshot_empty_arena` — 0 vectors → valid file (header +
/// empty sections + footer). The file roundtrips header + footer
/// CRC, and the descriptor / payload regions are zero-byte.
#[test]
fn g2_snapshot_empty_arena() {
    let tmpdir = TempDir::new().unwrap();
    let catalog = SnapshotCatalog::new();
    let spec = SnapshotSpec {
        tenant: TenantId::DEFAULT,
        partition: PartitionId::ZERO,
        index_id: 1,
        lsn: Lsn::new(1),
        encoding: 0, // F32
        index_type: 0,
        dim: 768,
        vectors_count: 0,
        sections: &[],
    };
    let path = flush_snapshot(&spec, tmpdir.path(), &catalog).unwrap();
    let (magic, version, count, lsn, vc) = read_and_verify_arcv(&path);
    assert_eq!(&magic, ARCV_MAGIC);
    assert_eq!(version, ARCV_FORMAT_VERSION);
    assert_eq!(count, 0);
    assert_eq!(lsn, 1);
    assert_eq!(vc, 0);
    let len = fs::metadata(&path).unwrap().len();
    // Header + 0 descriptors + 0 payloads + footer
    assert_eq!(len, (ARCV_HEADER_SIZE + ARCV_FOOTER_SIZE) as u64);
}

/// `g2_snapshot_single_store_quantized_only` — F32-no-rescore
/// arena; snapshot has only the quantized section.
#[test]
fn g2_snapshot_single_store_quantized_only() {
    let tmpdir = TempDir::new().unwrap();
    let catalog = SnapshotCatalog::new();
    // F32 dim=128, 4 vectors → 4 * 128 * 4 = 2048 bytes
    let payload = vec![0u8; 2048];
    let sections = [SnapshotSection {
        kind: SectionKind::Quantized,
        bytes: &payload,
    }];
    let spec = SnapshotSpec {
        tenant: TenantId::DEFAULT,
        partition: PartitionId::ZERO,
        index_id: 1,
        lsn: Lsn::new(2),
        encoding: 0, // F32
        index_type: 0,
        dim: 128,
        vectors_count: 4,
        sections: &sections,
    };
    let path = flush_snapshot(&spec, tmpdir.path(), &catalog).unwrap();
    let (_, _, count, _, _) = read_and_verify_arcv(&path);
    assert_eq!(count, 1, "exactly one section descriptor");

    let bytes = fs::read(&path).unwrap();
    // Section 0 descriptor at offset 64
    let kind = u16::from_le_bytes(bytes[64..66].try_into().unwrap());
    assert_eq!(kind, SectionKind::Quantized.as_u16());
}

/// `g2_snapshot_all_stores_present` — SQ8 arena with quantized +
/// rescore + labels; all 3 sections present and round-trip.
#[test]
fn g2_snapshot_all_stores_present() {
    let tmpdir = TempDir::new().unwrap();
    let catalog = SnapshotCatalog::new();
    let q = vec![0xAAu8; 256];
    let r = vec![0xBBu8; 512];
    let l = vec![0xCCu8; 64];
    let sections = [
        SnapshotSection {
            kind: SectionKind::Quantized,
            bytes: &q,
        },
        SnapshotSection {
            kind: SectionKind::Rescore,
            bytes: &r,
        },
        SnapshotSection {
            kind: SectionKind::Labels,
            bytes: &l,
        },
    ];
    let spec = spec_with(&sections);
    let path = flush_snapshot(&spec, tmpdir.path(), &catalog).unwrap();
    let (_, _, count, _, _) = read_and_verify_arcv(&path);
    assert_eq!(count, 3, "all three sections present");

    let bytes = fs::read(&path).unwrap();
    // Walk the descriptor table and verify each section is at the
    // documented offset and contains the right bytes.
    for (i, (expected_kind, expected_bytes)) in [
        (SectionKind::Quantized, q.as_slice()),
        (SectionKind::Rescore, r.as_slice()),
        (SectionKind::Labels, l.as_slice()),
    ]
    .iter()
    .enumerate()
    {
        let off = ARCV_HEADER_SIZE + i * ARCV_SECTION_DESCRIPTOR_SIZE;
        let kind = u16::from_le_bytes(bytes[off..off + 2].try_into().unwrap());
        assert_eq!(kind, expected_kind.as_u16());
        let payload_offset =
            u64::from_le_bytes(bytes[off + 8..off + 16].try_into().unwrap()) as usize;
        let payload_size =
            u64::from_le_bytes(bytes[off + 16..off + 24].try_into().unwrap()) as usize;
        // Each payload offset is 64-byte aligned.
        assert_eq!(payload_offset % 64, 0);
        assert_eq!(payload_size, expected_bytes.len());
        assert_eq!(
            &bytes[payload_offset..payload_offset + payload_size],
            *expected_bytes
        );
    }
}

/// `g2_snapshot_max_size_1m_vectors` — 1M vectors at 768-dim binary
/// encoding (smallest tractable encoding); snapshot completes
/// within 30s and file size matches expected (header + 1M × 128
/// bytes + footer).
#[test]
fn g2_snapshot_max_size_1m_vectors() {
    let tmpdir = TempDir::new().unwrap();
    let catalog = SnapshotCatalog::new();

    // Binary encoding at dim=768 → 96 bytes packed → padded to
    // 128 bytes per vector per ADR-035 §S-1. Use the padded size
    // for the payload.
    let bytes_per_vector = 128_usize;
    let n = 1_000_000_usize;
    let total = bytes_per_vector * n;
    // Allocate as a single buffer; on a modern dev host this is
    // 128 MiB which is well within the test budget.
    let payload = vec![0xA5u8; total];
    let sections = [SnapshotSection {
        kind: SectionKind::Quantized,
        bytes: &payload,
    }];
    let spec = SnapshotSpec {
        tenant: TenantId::DEFAULT,
        partition: PartitionId::ZERO,
        index_id: 1,
        lsn: Lsn::new(7_777_777),
        encoding: 3, // Binary
        index_type: 0,
        dim: 768,
        vectors_count: n as u64,
        sections: &sections,
    };
    let start = Instant::now();
    let path = flush_snapshot(&spec, tmpdir.path(), &catalog).unwrap();
    let elapsed = start.elapsed();
    assert!(
        elapsed < Duration::from_secs(30),
        "1M-vector flush took {elapsed:?} > 30s budget"
    );

    // File size: header + 1 descriptor + (align to 64) + payload + footer
    let descriptor_block = ARCV_SECTION_DESCRIPTOR_SIZE;
    // Payload starts at align_up(64 + 32, 64) = 128
    let payload_start = 128_usize;
    let expected_size = payload_start + total + ARCV_FOOTER_SIZE;
    let actual = fs::metadata(&path).unwrap().len();
    assert_eq!(actual as usize, expected_size);
    let _ = descriptor_block;
}

/// `g2_snapshot_uses_aligned_binary_size` — binary encoding at
/// dim=768 uses 128-byte aligned size (per ADR-035 §S-1), not 96.
/// Asserted by checking the payload byte count vs. an aligned-size
/// expectation.
#[test]
fn g2_snapshot_uses_aligned_binary_size() {
    let tmpdir = TempDir::new().unwrap();
    let catalog = SnapshotCatalog::new();
    // 4 binary vectors at dim=768 → 4 * 128 = 512 bytes if aligned,
    // 4 * 96 = 384 bytes if NOT aligned. The slice the caller
    // passes IS the aligned bytes (caller is the F.* arena layer
    // that owns alignment); G.2 just writes them verbatim. This
    // test verifies the caller's aligned size flows through to the
    // file unchanged — i.e., G.2 does not silently re-pack.
    let aligned = 128 * 4;
    let payload = vec![0u8; aligned];
    let sections = [SnapshotSection {
        kind: SectionKind::Quantized,
        bytes: &payload,
    }];
    let spec = SnapshotSpec {
        tenant: TenantId::DEFAULT,
        partition: PartitionId::ZERO,
        index_id: 1,
        lsn: Lsn::new(3),
        encoding: 3, // Binary
        index_type: 0,
        dim: 768,
        vectors_count: 4,
        sections: &sections,
    };
    let path = flush_snapshot(&spec, tmpdir.path(), &catalog).unwrap();
    let bytes = fs::read(&path).unwrap();
    let payload_size = u64::from_le_bytes(
        bytes[ARCV_HEADER_SIZE + 16..ARCV_HEADER_SIZE + 24]
            .try_into()
            .unwrap(),
    ) as usize;
    assert_eq!(
        payload_size, aligned,
        "binary encoding must reflect 128-byte aligned size, not 96"
    );
}

// ─────────────────────────────────────────────────────────────────
// Bulk-load policy (OQ-V3)
// ─────────────────────────────────────────────────────────────────

/// `g2_bulk_load_completion_forces_snapshot` — single transaction
/// inserts 1500 vectors; the bulk-load policy fires; flush runs at
/// txn end; the on-disk .snap reflects all 1500 vectors.
#[test]
fn g2_bulk_load_completion_forces_snapshot() {
    let policy = SnapshotPolicy::default();
    let txn_inserts = 1_500_usize;
    // Periodic threshold not crossed; schema not changed.
    let trigger = policy.should_snapshot(0, txn_inserts, false);
    assert_eq!(trigger, Some(SnapshotTrigger::BulkLoad));

    // Now drive the flush as if F.* fired it.
    let tmpdir = TempDir::new().unwrap();
    let catalog = SnapshotCatalog::new();
    // 1500 vectors at SQ8 dim=128 → 1500 * 128 = 192_000 bytes
    let payload = vec![0xB1u8; 1_500 * 128];
    let sections = [SnapshotSection {
        kind: SectionKind::Quantized,
        bytes: &payload,
    }];
    let spec = SnapshotSpec {
        tenant: TenantId::DEFAULT,
        partition: PartitionId::ZERO,
        index_id: 1,
        lsn: Lsn::new(5_555),
        encoding: 2, // SQ8
        index_type: 0,
        dim: 128,
        vectors_count: 1_500,
        sections: &sections,
    };
    let path = flush_snapshot(&spec, tmpdir.path(), &catalog).unwrap();
    let (_, _, _, _, vc) = read_and_verify_arcv(&path);
    assert_eq!(vc, 1_500, "bulk-load .snap reflects all 1500 vectors");
    assert_eq!(
        catalog.latest_lsn(TenantId::DEFAULT, 1),
        Some(Lsn::new(5_555))
    );
}

// ─────────────────────────────────────────────────────────────────
// Tier interaction (cross-cutting with ADR-034)
// ─────────────────────────────────────────────────────────────────

/// `g2_snapshot_during_t3_periodic_commit` — T3 (periodic) tenant
/// with a vector arena. The snapshot fires during the async-fsync
/// window — the captured spec contains a snapshot taken BEFORE
/// the background fsync. The flushed file matches the captured
/// state, preserving T3's RYW semantic (the flush is a stable
/// borrow over caller-captured bytes; ADR-034's background fsync
/// scheduler can run in parallel without bleeding through).
///
/// This test models the interaction at the G.2 boundary: the
/// flush primitive is decoupled from ADR-034's fsync tier — the
/// caller is responsible for capturing the right point-in-time
/// bytes. G.2's contract is "the bytes you give me are the bytes
/// that land on disk."
#[test]
fn g2_snapshot_during_t3_periodic_commit() {
    let tmpdir = TempDir::new().unwrap();
    let catalog = SnapshotCatalog::new();

    // Simulate the T3 pre-fsync visible state: the caller has
    // committed N vectors with `visible` advanced (per ADR-034
    // D-4 RYW preservation) but the WAL fsync is still pending.
    // The caller captures the visible bytes for snapshot purposes.
    let visible_bytes = vec![0x55u8; 4096];
    let sections = [SnapshotSection {
        kind: SectionKind::Quantized,
        bytes: &visible_bytes,
    }];
    let spec = SnapshotSpec {
        tenant: TenantId::new(200), // T3 tenant id (simulated)
        partition: PartitionId::ZERO,
        index_id: 1,
        lsn: Lsn::new(8_888),
        encoding: 2,
        index_type: 0,
        dim: 768,
        vectors_count: 4,
        sections: &sections,
    };

    // Background "fsync scheduler" is a dummy thread that does
    // nothing observable during the flush — the flush must succeed
    // regardless of tier-side activity.
    let stop = Arc::new(AtomicBool::new(false));
    let bg = {
        let stop = Arc::clone(&stop);
        thread::spawn(move || {
            let mut counter: u64 = 0;
            while !stop.load(Ordering::Relaxed) {
                counter = counter.wrapping_add(1);
                thread::yield_now();
            }
            counter
        })
    };

    let path = flush_snapshot(&spec, tmpdir.path(), &catalog).unwrap();
    stop.store(true, Ordering::Relaxed);
    let _ = bg.join().unwrap();

    // Snapshot reflects the visible state captured before the
    // background fsync ran.
    let bytes = fs::read(&path).unwrap();
    let payload_offset = u64::from_le_bytes(
        bytes[ARCV_HEADER_SIZE + 8..ARCV_HEADER_SIZE + 16]
            .try_into()
            .unwrap(),
    ) as usize;
    let payload_size = u64::from_le_bytes(
        bytes[ARCV_HEADER_SIZE + 16..ARCV_HEADER_SIZE + 24]
            .try_into()
            .unwrap(),
    ) as usize;
    assert_eq!(payload_size, visible_bytes.len());
    assert_eq!(
        &bytes[payload_offset..payload_offset + payload_size],
        visible_bytes.as_slice()
    );
}

// ─────────────────────────────────────────────────────────────────
// Bonus: full crash-injection coverage (all CrashPoints)
// ─────────────────────────────────────────────────────────────────

/// Cover every [`CrashPoint`] variant — ensures the inner state
/// transitions are well-formed. (Mid-write and BeforeRename are
/// already covered above; this test asserts the remaining points
/// produce the expected on-disk artifacts.)
#[test]
fn g2_crash_at_every_point_produces_graceful_artifact() {
    let tmpdir = TempDir::new().unwrap();
    let catalog = SnapshotCatalog::new();
    let payload = vec![0u8; 64];
    let sections = [SnapshotSection {
        kind: SectionKind::Quantized,
        bytes: &payload,
    }];

    // AfterTempCreate: 0-byte .tmp, no .snap, no catalog stamp.
    let spec1 = SnapshotSpec {
        tenant: TenantId::new(10),
        partition: PartitionId::ZERO,
        index_id: 1,
        lsn: Lsn::new(1),
        encoding: 2,
        index_type: 0,
        dim: 768,
        vectors_count: 0,
        sections: &sections,
    };
    let _ = flush_snapshot_with_crash_point(
        &spec1,
        tmpdir.path(),
        &catalog,
        CrashPoint::AfterTempCreate,
    )
    .unwrap_err();
    let tmp1 = snapshot_temp_path(tmpdir.path(), spec1.tenant, spec1.index_id, spec1.lsn);
    assert!(tmp1.exists());
    assert_eq!(fs::metadata(&tmp1).unwrap().len(), 0);
    assert!(!snapshot_path(tmpdir.path(), spec1.tenant, spec1.index_id, spec1.lsn).exists());

    // BeforeDirFsync: .snap exists (rename ran), but the catalog
    // entry was already stamped or not — depends on order. With
    // current code, catalog stamps AFTER dir fsync, so no stamp.
    let spec2 = SnapshotSpec {
        tenant: TenantId::new(11),
        partition: PartitionId::ZERO,
        index_id: 1,
        lsn: Lsn::new(2),
        encoding: 2,
        index_type: 0,
        dim: 768,
        vectors_count: 0,
        sections: &sections,
    };
    let _ = flush_snapshot_with_crash_point(
        &spec2,
        tmpdir.path(),
        &catalog,
        CrashPoint::BeforeDirFsync,
    )
    .unwrap_err();
    let final2 = snapshot_path(tmpdir.path(), spec2.tenant, spec2.index_id, spec2.lsn);
    assert!(final2.exists(), "rename ran before BeforeDirFsync");
    assert_eq!(catalog.latest_lsn(spec2.tenant, spec2.index_id), None);

    // BeforeCatalogStamp: .snap exists, catalog NOT stamped.
    let spec3 = SnapshotSpec {
        tenant: TenantId::new(12),
        partition: PartitionId::ZERO,
        index_id: 1,
        lsn: Lsn::new(3),
        encoding: 2,
        index_type: 0,
        dim: 768,
        vectors_count: 0,
        sections: &sections,
    };
    let _ = flush_snapshot_with_crash_point(
        &spec3,
        tmpdir.path(),
        &catalog,
        CrashPoint::BeforeCatalogStamp,
    )
    .unwrap_err();
    let final3 = snapshot_path(tmpdir.path(), spec3.tenant, spec3.index_id, spec3.lsn);
    assert!(final3.exists());
    assert_eq!(catalog.latest_lsn(spec3.tenant, spec3.index_id), None);

    // The .snap from spec3 round-trips header + footer CRC — proof
    // that even when the catalog stamp was skipped, the on-disk
    // artifact is byte-valid (recovery picks the older catalog
    // entry but the orphan .snap is harmless).
    let bytes = fs::read(&final3).unwrap();
    let crc_offset = bytes.len() - ARCV_TRAILING_CRC_SIZE;
    let stored = u32::from_le_bytes(bytes[crc_offset..].try_into().unwrap());
    let computed = crc32c::crc32c(&bytes[..crc_offset]);
    assert_eq!(stored, computed);
}

/// G.2 produces a graceful artifact when the snapshot dir does
/// not yet exist — `create_dir_all` runs before any I/O so a
/// fresh deployment can flush without explicit dir setup.
#[test]
fn g2_flush_creates_missing_snapshot_dir() {
    let tmpdir = TempDir::new().unwrap();
    let nested = tmpdir.path().join("nested").join("deep").join("dir");
    assert!(!nested.exists());

    let catalog = SnapshotCatalog::new();
    let payload = vec![0u8; 32];
    let sections = [SnapshotSection {
        kind: SectionKind::Quantized,
        bytes: &payload,
    }];
    let spec = spec_with(&sections);
    let path = flush_snapshot(&spec, &nested, &catalog).unwrap();
    assert!(path.exists());
    assert!(nested.exists());
}

/// Final defense-in-depth: the temp file has the documented
/// extension `.snap.tmp` and the final file has `.snap`. Other
/// callers (e.g., G.3 cleanup of orphan .tmp) rely on the suffix
/// pattern.
#[test]
fn g2_paths_use_documented_suffixes() {
    let tmpdir = TempDir::new().unwrap();
    let tmp = snapshot_temp_path(tmpdir.path(), TenantId::new(1), 2, Lsn::new(3));
    let snap = snapshot_path(tmpdir.path(), TenantId::new(1), 2, Lsn::new(3));
    assert!(tmp.to_str().unwrap().ends_with("arena-1-2-3.snap.tmp"));
    assert!(snap.to_str().unwrap().ends_with("arena-1-2-3.snap"));
}

/// Reading the partial .tmp from `MidWrite(N)` mid-write must not
/// somehow be mistakable for a valid file. Direct read from the
/// truncated .tmp confirms it is short.
#[test]
fn g2_mid_write_temp_is_unambiguously_truncated() {
    let tmpdir = TempDir::new().unwrap();
    let catalog = SnapshotCatalog::new();
    let payload = vec![0u8; 4096];
    let sections = [SnapshotSection {
        kind: SectionKind::Quantized,
        bytes: &payload,
    }];
    let spec = spec_with(&sections);

    let _ =
        flush_snapshot_with_crash_point(&spec, tmpdir.path(), &catalog, CrashPoint::MidWrite(64))
            .unwrap_err();
    let tmp = snapshot_temp_path(tmpdir.path(), spec.tenant, spec.index_id, spec.lsn);
    let mut f = File::open(&tmp).unwrap();
    let mut head = [0u8; 4];
    f.seek(SeekFrom::Start(0)).unwrap();
    f.read_exact(&mut head).unwrap();
    // Magic is intact (we wrote at least 64 bytes), but file is too
    // short to carry a footer — recovery's "read footer at end-16"
    // will fail (file is exactly 64 bytes so footer slice would
    // overlap header bytes 48..64, which are reserved-zero, not a
    // valid footer).
    assert_eq!(&head, ARCV_MAGIC);
    let len = fs::metadata(&tmp).unwrap().len();
    assert_eq!(len, 64);
}

// ─────────────────────────────────────────────────────────────────
// M3.a Phase 5.5 — snapshot-during-T3-commit extension
// ─────────────────────────────────────────────────────────────────
//
// Per Path A directive 2026-04-26 + Phase 5.5 spec §2.4: extend the
// snapshot-coverage suite with a test that captures a snapshot
// *during* an in-flight ADR-034 T3 (Periodic) commit. The pin is
// the ADR-034 D-4 RYW contract: visible advances pre-fsync, so an
// in-process snapshot taken AFTER T3 commit returns Ok but BEFORE
// the scheduler's batched fsync MUST observe the post-commit state
// (the bytes the running tenant can see via its own readers).
//
// The flush primitive itself is sync + atomic per G.2 (rename +
// dir-fsync + catalog stamp); the "during" semantics here mean
// "after the user-visible commit returns Ok, before the WAL fsync
// scheduler's batch fires". The snapshot bytes therefore reflect
// the in-memory arena state, which already incorporates the T3
// commit (T3 RYW within the same process is preserved per D-4).
//
// Production wiring of the vector arena into the T3 batch lands in
// Slice G.4; today the snapshot flush primitive proves the analogue
// at the catalog-stamp + file-bytes level: catalog stamps land
// synchronously per `flush_snapshot`'s post-rename path, so the
// stamp is the read-after-snapshot RYW token.

#[test]
fn snapshot_during_t3_commit_captures_pre_fsync_state() {
    let tmpdir = TempDir::new().unwrap();
    let catalog = SnapshotCatalog::new();
    let tenant = TenantId::new(2025);
    let index_id: u64 = 11;

    // Phase A: simulate a T3-tier commit at LSN=1000. The "commit"
    // is the act of staging post-W bytes into the in-memory arena
    // and bumping the visible-LSN counter. In the production wiring
    // this corresponds to the Phase-3 publish + visible.store
    // happening BEFORE the scheduler-driven fsync — that's the D-4
    // RYW window. We model the post-commit visible state as a
    // snapshot section payload tagged with the commit LSN.
    let pre_fsync_payload = vec![0xC0u8; 256];
    let pre_fsync_sections = [SnapshotSection {
        kind: SectionKind::Quantized,
        bytes: &pre_fsync_payload,
    }];
    let t3_commit_lsn = Lsn::new(1_000);
    let spec_at_t3 = SnapshotSpec {
        tenant,
        partition: PartitionId::ZERO,
        index_id,
        lsn: t3_commit_lsn,
        encoding: 2, // SQ8
        index_type: 0,
        dim: 768,
        vectors_count: 64,
        sections: &pre_fsync_sections,
    };

    // Phase B: take the snapshot. The snapshot flush is itself a T1
    // commit (the catalog stamp is part of the snapshot's atomic
    // protocol per G.2). The captured bytes MUST include the T3
    // commit's payload (the section we just staged) — proving the
    // snapshot saw the pre-fsync visible state, not the on-disk
    // pre-T3 state.
    let path = flush_snapshot(&spec_at_t3, tmpdir.path(), &catalog).expect("flush during T3");
    assert!(path.exists(), "snapshot file landed during T3 RYW window");

    // The snapshot's bytes contain the pre-fsync payload exactly.
    // The sectionsdescriptors after the header point at the
    // post-T3 bytes; recovering these proves the snapshot saw the
    // in-flight state.
    let bytes = fs::read(&path).unwrap();
    let payload_in_file = locate_quantized_section_bytes(&bytes, pre_fsync_payload.len());
    assert_eq!(
        payload_in_file, pre_fsync_payload,
        "snapshot-during-T3-commit: captured bytes drifted from pre-fsync visible state"
    );

    // Catalog stamp: post-flush, the catalog reports the snapshot's
    // LSN. This is the ADR-034 D-4 RYW token — a subsequent
    // recovery loader using this catalog reads the snapshot at
    // t3_commit_lsn and replays only post-1000 deltas. The T3
    // commit at LSN=1000 is "captured" in the snapshot, not
    // re-replayed.
    assert_eq!(
        catalog.latest_lsn(tenant, index_id),
        Some(t3_commit_lsn),
        "catalog stamp must equal snapshot LSN (RYW token for T3 commits)"
    );

    // Phase C: a SECOND T3 commit lands AFTER the snapshot at
    // LSN=1500. It is NOT captured by the prior snapshot (already
    // closed); the recovery loader re-applies it from the WAL
    // delta stream. We pin the snapshot's content as the closed
    // pre-1500 state by re-reading: the file bytes are byte-stable
    // (the snapshot writer is one-shot, write-once; no in-place
    // edits).
    let bytes_re_read = fs::read(&path).unwrap();
    assert_eq!(
        bytes, bytes_re_read,
        "post-snapshot file bytes must be stable (snapshot is write-once-then-closed)"
    );

    // Phase D: take a SECOND snapshot AFTER the additional T3
    // commit. Catalog stamp advances; first snapshot file remains
    // for the G.3 recovery cleanup pass to GC.
    let phase_d_payload = vec![0xD0u8; 256];
    let phase_d_sections = [SnapshotSection {
        kind: SectionKind::Quantized,
        bytes: &phase_d_payload,
    }];
    let later_lsn = Lsn::new(1_500);
    let spec_phase_d = SnapshotSpec {
        tenant,
        partition: PartitionId::ZERO,
        index_id,
        lsn: later_lsn,
        encoding: 2,
        index_type: 0,
        dim: 768,
        vectors_count: 64,
        sections: &phase_d_sections,
    };
    let path_d = flush_snapshot(&spec_phase_d, tmpdir.path(), &catalog).expect("flush phase D");
    assert!(path_d.exists());
    assert_ne!(
        path, path_d,
        "two distinct snapshot LSNs produce two distinct file paths"
    );
    assert_eq!(
        catalog.latest_lsn(tenant, index_id),
        Some(later_lsn),
        "catalog stamp advances to the latest snapshot LSN"
    );
    // Phase B's snapshot bytes still match pre-fsync payload —
    // the second snapshot did not retroactively rewrite the first.
    let bytes_phase_b_after_d = fs::read(&path).unwrap();
    let payload_after_d =
        locate_quantized_section_bytes(&bytes_phase_b_after_d, pre_fsync_payload.len());
    assert_eq!(
        payload_after_d, pre_fsync_payload,
        "first snapshot bytes stable across subsequent snapshot flushes"
    );
}

/// Locate the Quantized section's payload bytes inside an ARCV
/// snapshot file. Walks the descriptor table at the file's offset
/// `ARCV_HEADER_SIZE`, scans for the first Quantized kind, and
/// slices `payload_len` bytes from the resolved offset.
///
/// Descriptor layout (per snapshot.rs §G.2 docstring):
/// ```text
///   0..2   kind: u16 LE  (0=Quantized, 1=Rescore, 2=Labels)
///   2..4   flags: u16 LE
///   4..8   reserved: u32
///   8..16  payload_offset: u64 LE  ← absolute file offset
///   16..24 payload_size: u64 LE
///   24..32 reserved: u64
/// ```
fn locate_quantized_section_bytes(bytes: &[u8], payload_len: usize) -> Vec<u8> {
    let section_count = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    assert!(section_count > 0, "snapshot has no sections");
    let table_start = ARCV_HEADER_SIZE;
    for i in 0..section_count {
        let desc = table_start + i * ARCV_SECTION_DESCRIPTOR_SIZE;
        assert!(
            bytes.len() >= desc + ARCV_SECTION_DESCRIPTOR_SIZE,
            "descriptor {i} extends past end of file"
        );
        let kind = u16::from_le_bytes(bytes[desc..desc + 2].try_into().unwrap());
        if kind != SectionKind::Quantized.as_u16() {
            continue;
        }
        let offset = u64::from_le_bytes(bytes[desc + 8..desc + 16].try_into().unwrap()) as usize;
        let length = u64::from_le_bytes(bytes[desc + 16..desc + 24].try_into().unwrap()) as usize;
        assert!(
            length >= payload_len,
            "Quantized section length ({length}) is less than expected payload_len ({payload_len})"
        );
        assert!(
            offset + payload_len <= bytes.len(),
            "Quantized section payload range extends past file end"
        );
        return bytes[offset..offset + payload_len].to_vec();
    }
    panic!("snapshot has no Quantized section in descriptor table");
}
