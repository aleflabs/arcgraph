//! No `CommitBundle` record may span a segment boundary. Crash
//! atomicity relies on each `CommitBundle` being a single
//! CRC-protected unit; if the segment writer could split a record's
//! bytes across two segment files, a crash between segment N's fsync
//! and segment N+1's fsync would leave a torn-but-decodable prefix in
//! segment N that recovery couldn't detect via torn-tail
//! classification.
//!
//! Today `SegmentWriter::append` pre-rotates to a fresh segment
//! when the incoming record wouldn't fit — this test is a
//! belt-and-suspenders regression guard against any future
//! refactor that accidentally introduces mid-record splits.
//!
//! Test shape: drive a small segment size (~16 KiB), issue 10
//! bundles through the WAL writer, walk all segments, decode
//! every CommitBundle, assert each decode consumed exactly its
//! declared length inside a single segment's byte stream.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use arcgraph_core::{LabelId, TenantId};
use arcgraph_storage::crud::{CrudStore, PropertyData, commit, create_node};
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::PrimaryIndex;
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::segment::{SegmentHeader, list_segments, segment_filename};
use arcgraph_storage::wal::{WalConfig, WalRecord, WalRecordType, WalWriter};
use tempfile::TempDir;

fn tiny_segment_config(dir: PathBuf) -> WalConfig {
    WalConfig {
        dir,
        // 16 KiB — fits maybe one CommitBundle (~8 KB payload +
        // 44 B header ≈ 8.3 KB) per segment, so every commit
        // likely rotates.
        segment_size_bytes: 16 * 1024,
        group_commit_window: Duration::from_millis(1),
        group_commit_max_batch: 16,
        metrics_sink: None,
        encryption: None,
        inflight_budget_bytes: None,
    }
}

#[test]
fn no_commit_bundle_record_spans_a_segment_boundary() {
    let dir = TempDir::new().unwrap();
    let writer = WalWriter::spawn(tiny_segment_config(dir.path().to_path_buf())).unwrap();
    let handle = writer.handle();

    let mgr = Arc::new(TxnManager::with_wal(handle.clone()));
    let alloc = Arc::new(PageAllocator::new());
    let primary = Arc::new(
        PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&alloc), Some(handle.clone())).unwrap(),
    );
    let store = CrudStore::new_with_index(Some(handle.clone()), Arc::clone(&primary), alloc);

    // 10 commits — with a 16 KiB segment size and ~8 KB per
    // CommitBundle we expect at least 5 segments after the run.
    for i in 0..10u32 {
        let mut tx = mgr.begin(TenantId::DEFAULT);
        create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(i),
            &PropertyData::Empty,
        )
        .unwrap();
        commit(tx, &store).unwrap();
    }
    writer.shutdown().unwrap();

    // Walk each segment file INDEPENDENTLY. If any CommitBundle's
    // bytes were split across two files, per-segment decode would
    // fail with `InvalidRecordLength` (length field exceeds
    // remaining bytes) rather than returning a full record.
    let segs = list_segments(dir.path()).unwrap();
    assert!(
        segs.len() >= 2,
        "tiny segment size should rotate ≥ 1 time (got {} segments)",
        segs.len()
    );

    let mut total_bundles = 0usize;
    for seg in &segs {
        let path = dir.path().join(segment_filename(*seg));
        let bytes = std::fs::read(&path).unwrap();
        // Skip the fixed segment header (magic + format_version + reserved)
        // landed by ADR-031's companion fix #39 (PR #68). A segment shorter
        // than the header is a terminal crash artifact with no records.
        if bytes.len() < SegmentHeader::SIZE {
            continue;
        }
        SegmentHeader::decode(&bytes[..SegmentHeader::SIZE]).unwrap();
        let mut cursor = SegmentHeader::SIZE;
        while cursor < bytes.len() {
            // This MUST succeed for every record in the segment —
            // a mid-record split would leave `bytes[cursor..]` with
            // a length header claiming more bytes than remain,
            // which `WalRecord::decode` reports as
            // `InvalidRecordLength`.
            match WalRecord::decode(&bytes[cursor..]) {
                Ok((r, consumed)) => {
                    if r.record_type == WalRecordType::CommitBundle {
                        total_bundles += 1;
                    }
                    cursor += consumed;
                }
                Err(e) => {
                    panic!(
                        "segment {seg} at offset {cursor} has an incomplete record — \
                         CommitBundle must never span a segment boundary. Error: {e:?}"
                    );
                }
            }
        }
    }

    // Sanity: every commit produced exactly one user CommitBundle.
    // Bootstrap emits one additional SYSTEM CommitBundle + one
    // IndexPage legacy record, so total_bundles ≥ 10 (user) + 1
    // (SYSTEM bootstrap).
    assert!(
        total_bundles >= 11,
        "expected ≥ 11 CommitBundle records (10 user + 1 bootstrap), got {total_bundles}"
    );
}
