//! WAL format-versioning regression tests (issue #39).
//!
//! These exercise the operator-facing recovery path end-to-end: a
//! writer stamps the v1 segment header, and `WalRecoveryReader`
//! strictly rejects unknown versions / wrong magic with structured
//! errors so an operator sees "upgrade required" instead of "WAL corrupt".

use std::time::Duration;

use arcgraph_core::{ArcGraphError, TenantId};
use arcgraph_storage::wal::{
    CURRENT_WAL_FORMAT_VERSION, SUPPORTED_WAL_FORMAT_VERSIONS, SegmentHeader, WAL_SEGMENT_MAGIC,
    WalConfig, WalRecordType, WalRecoveryReader, WalWriter, segment_filename,
};
use tempfile::tempdir;

fn fast_config(dir: std::path::PathBuf) -> WalConfig {
    WalConfig {
        dir,
        segment_size_bytes: 16 * 1024 * 1024,
        group_commit_window: Duration::from_millis(2),
        group_commit_max_batch: 4,
        metrics_sink: None,
        encryption: None,
        inflight_budget_bytes: None,
    }
}

/// A fresh segment written by the writer carries the v1 header at
/// offset 0 and the format version matches `CURRENT_WAL_FORMAT_VERSION`.
#[test]
fn wal_segment_header_stamps_current_version() {
    let dir = tempdir().unwrap();
    let writer = WalWriter::spawn(fast_config(dir.path().to_path_buf())).unwrap();
    let handle = writer.handle();
    // One record so the segment is definitely flushed to disk.
    handle
        .append(WalRecordType::PutNode, 1, 0, TenantId::DEFAULT, vec![0xAB])
        .unwrap();
    writer.shutdown().unwrap();

    let path = dir.path().join(segment_filename(0));
    let bytes = std::fs::read(&path).unwrap();
    assert!(
        bytes.len() >= SegmentHeader::SIZE,
        "segment too short: {} bytes",
        bytes.len()
    );

    let header = SegmentHeader::decode(&bytes[..SegmentHeader::SIZE]).unwrap();
    assert_eq!(header.format_version, CURRENT_WAL_FORMAT_VERSION);
    assert_eq!(&bytes[0..4], &WAL_SEGMENT_MAGIC);
}

/// A segment with valid magic but an unsupported version fails
/// recovery with `WalFormatMismatch { found_version: 999,
/// supported_versions: SUPPORTED_WAL_FORMAT_VERSIONS }`. Distinct
/// from corruption; operator sees "upgrade required".
///
/// ADR-032 Slice 1: `SUPPORTED_WAL_FORMAT_VERSIONS` is now `[1, 2]`
/// (reader supports the v2 CommitBundle payload shape); the writer
/// still stamps v1 until Slice 2's cutover.
#[test]
fn wal_reader_rejects_unknown_version() {
    let dir = tempdir().unwrap();
    let mut header = SegmentHeader::current().encode();
    // Valid magic preserved, version bumped to one we don't know.
    header[4..6].copy_from_slice(&999u16.to_le_bytes());
    std::fs::write(dir.path().join(segment_filename(0)), header).unwrap();

    // Opening is where `advance_segment` is first called, so the
    // validation fires on `open` (before any iterator step).
    let err = WalRecoveryReader::open(dir.path()).unwrap_err();
    match err {
        ArcGraphError::WalFormatMismatch {
            found_version,
            supported_versions,
        } => {
            assert_eq!(found_version, 999);
            assert_eq!(supported_versions, SUPPORTED_WAL_FORMAT_VERSIONS);
        }
        other => panic!("expected WalFormatMismatch, got {other:?}"),
    }
}

/// A segment with wrong magic is distinct from "right file type,
/// wrong version" — reader emits `WalBadMagic`, not
/// `WalFormatMismatch`. Lets an operator tell "random file" from
/// "valid WAL from a different binary version".
#[test]
fn wal_reader_rejects_missing_magic() {
    let dir = tempdir().unwrap();
    let mut bogus = SegmentHeader::current().encode();
    bogus[0..4].copy_from_slice(b"XXXX"); // magic replaced
    std::fs::write(dir.path().join(segment_filename(0)), bogus).unwrap();

    let err = WalRecoveryReader::open(dir.path()).unwrap_err();
    match err {
        ArcGraphError::WalBadMagic { got, expected } => {
            assert_eq!(&got, b"XXXX");
            assert_eq!(&expected, b"AGWL");
            assert_eq!(expected, WAL_SEGMENT_MAGIC);
        }
        other => panic!("expected WalBadMagic, got {other:?}"),
    }
}

/// Regression: a non-terminal segment with an unknown version is a
/// hard error. Covers the rollback-hazard from the reviewer's PR #67
/// feedback — a pre-PR-#67 binary reading a post-PR-#67 WAL that has
/// since rotated to a newer segment must fail loudly.
#[test]
fn wal_reader_rejects_unknown_version_in_non_terminal_segment() {
    let dir = tempdir().unwrap();
    // Segment 0: valid v1 with one record.
    let writer = WalWriter::spawn(fast_config(dir.path().to_path_buf())).unwrap();
    writer
        .handle()
        .append(WalRecordType::PutNode, 1, 0, TenantId::DEFAULT, vec![0xCD])
        .unwrap();
    writer.shutdown().unwrap();

    // Segment 1: hand-crafted with valid magic but version=999.
    let mut header = SegmentHeader::current().encode();
    header[4..6].copy_from_slice(&999u16.to_le_bytes());
    std::fs::write(dir.path().join(segment_filename(1)), header).unwrap();

    // Segment 2: same as segment 1, kept so segment 1 is non-terminal.
    std::fs::write(
        dir.path().join(segment_filename(2)),
        SegmentHeader::current().encode(),
    )
    .unwrap();

    // The reader opens segment 0 first. Iterating should consume
    // segment 0's record and then hit segment 1 (non-terminal, bad
    // version) with `WalFormatMismatch`.
    let mut reader = WalRecoveryReader::open(dir.path()).unwrap();
    let first = reader.next().expect("segment 0 record").unwrap();
    assert_eq!(first.payload, vec![0xCD]);

    let err = reader
        .next()
        .expect("non-terminal segment 1 must error")
        .unwrap_err();
    assert!(
        matches!(
            err,
            ArcGraphError::WalFormatMismatch {
                found_version: 999,
                ..
            }
        ),
        "expected WalFormatMismatch{{999}}, got {err:?}"
    );
}
