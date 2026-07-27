//! W26-γ-2 D5#2 — Negative scenario: disk full mid-WAL-append (the
//! ENOSPC / disk-full class).
//!
//! Real-world incident: SQLite has fired numerous "WAL grew past
//! disk" bug reports; ZFS has a known "out of space during WAL
//! append" corruption class; MySQL's InnoDB ALWAYS surfaces ENOSPC
//! as a hard-crash rather than partial write (post-2018 fix).
//!
//! ArcGraph's analog: the WAL writer's append path MUST surface
//! `ArcGraphError::Io(...)` (the underlying ENOSPC) without partial
//! flush. The recovery path MUST treat a partial-tail WAL as a torn
//! tail (`ArcGraphError::InvalidRecordLength` or `WalCorruption`
//! at the WalRecord level).
//!
//! This test asserts the partial-write-rejection invariant at the
//! WalRecord codec layer (the load-bearing surface).

use std::io;

use arcgraph_core::error::ArcGraphError;
use arcgraph_core::{Lsn, PAGE_SIZE, PageId, TenantId};
use arcgraph_storage::wal::bundle::{StagedEmit, encode_commit_bundle};
use arcgraph_storage::wal::{WalRecord, WalRecordType};
use bytes::Bytes;

fn encoded_bundle_for_disk_full_probe() -> Vec<u8> {
    let mut writes = std::collections::HashMap::new();
    writes.insert(
        0xFE_FE_u64,
        Some(Bytes::from_static(b"disk-full-probe-payload")),
    );
    let mut page_bytes: Box<[u8; PAGE_SIZE]> = Box::new([0u8; PAGE_SIZE]);
    for (i, b) in page_bytes.iter_mut().enumerate() {
        *b = ((i ^ 0xAA) & 0xFF) as u8;
    }
    let staged = vec![StagedEmit {
        kind: arcgraph_storage::wal::BundlePageKind::PrimaryIndex,
        page_id: PageId::new(99),
        bytes: page_bytes,
    }];
    let payload = encode_commit_bundle(Lsn::new(1), &writes, &staged, TenantId::DEFAULT);
    let r = WalRecord {
        record_type: WalRecordType::CommitBundle,
        txn_id: 1,
        lsn: Lsn::new(1),
        timestamp_ms: 1_700_000_000_000,
        tenant_id: TenantId::DEFAULT,
        payload,
    };
    r.encode_to_vec().expect("encode")
}

#[test]
fn partial_wal_record_after_disk_full_decodes_as_invalid_record_length() {
    // Simulate "disk filled mid-write" by truncating the encoded
    // record to a fraction of its declared length. The decoder MUST
    // surface InvalidRecordLength, not silently return an empty
    // record or panic.
    let bytes = encoded_bundle_for_disk_full_probe();
    assert!(bytes.len() > 500, "sample must be substantial");
    let half = &bytes[..bytes.len() / 2];
    let err = WalRecord::decode(half).expect_err("partial WAL record must reject");
    match err {
        ArcGraphError::InvalidRecordLength { got, expected } => {
            assert!(got < expected, "partial write must report got < expected");
        }
        other => panic!("expected InvalidRecordLength on partial WAL; got {other:?}"),
    }
}

#[test]
fn one_byte_wal_record_after_disk_full_decodes_structured() {
    // Extreme partial write: only one byte made it to disk before
    // the disk-full error fired.
    let one_byte = vec![0u8];
    let err = WalRecord::decode(&one_byte).expect_err("one-byte WAL record must reject");
    assert!(
        matches!(
            err,
            ArcGraphError::InvalidRecordLength { .. } | ArcGraphError::WalCorruption { .. }
        ),
        "one-byte WAL must surface InvalidRecordLength or WalCorruption; got: {err:?}"
    );
}

#[test]
fn empty_wal_record_after_disk_full_decodes_structured() {
    // The most extreme partial write: 0 bytes were written before
    // the disk-full error fired.
    let empty = vec![];
    let err = WalRecord::decode(&empty).expect_err("empty WAL bytes must reject");
    // The decoder MUST surface a structured error, never a panic.
    match err {
        ArcGraphError::InvalidRecordLength { .. } | ArcGraphError::WalCorruption { .. } => {}
        other => panic!("expected InvalidRecordLength/WalCorruption, got {other:?}"),
    }
}

#[test]
fn disk_full_io_error_maps_to_arcgraph_io_variant() {
    // The underlying disk-full error is std::io::Error with kind
    // `StorageFull` (or `Other`); arcgraph-core's `From<io::Error>`
    // lifts it to ArcGraphError::Io. Pin the lift contract.
    let io_err = io::Error::new(io::ErrorKind::StorageFull, "ENOSPC");
    let lifted: ArcGraphError = io_err.into();
    assert!(matches!(lifted, ArcGraphError::Io(_)));
    let display = format!("{lifted}");
    assert!(
        display.contains("ENOSPC"),
        "io error must include ENOSPC; got: {display}"
    );
}

#[test]
fn disk_full_rolledback_variant_composes_with_io() {
    // The combined production path: WAL fsync fails with ENOSPC →
    // rollback wraps as WalErrorRolledBack { source: Io(ENOSPC) }.
    let inner = ArcGraphError::Io(io::Error::new(io::ErrorKind::StorageFull, "ENOSPC"));
    let outer = ArcGraphError::WalErrorRolledBack {
        source: Box::new(inner),
    };
    let display = format!("{outer}");
    assert!(display.contains("rolled back"));
    assert!(
        display.contains("ENOSPC"),
        "operator must see ENOSPC chain; got: {display}"
    );
}
