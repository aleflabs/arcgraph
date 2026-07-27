//! A torn or CRC-flipped `CommitBundle` record MUST be rejected by the
//! WAL record decoder, not silently returned with corrupt contents.
//!
//! Commit atomicity requires either a fully durable bundle (CRC +
//! length both pass) or a detected tear (one of those checks fires
//! and the decoder returns `InvalidRecordLength` or `WalCorruption`).
//! At the terminal segment, the recovery iterator converts
//! `InvalidRecordLength` into `TornTail { segment, offset }` (expected
//! crash recovery); `WalCorruption` remains a hard error elsewhere.

use arcgraph_core::PAGE_SIZE;
use arcgraph_core::PageId;
use arcgraph_core::{ArcGraphError, Lsn, TenantId};
use arcgraph_storage::wal::bundle::{StagedEmit, encode_commit_bundle};
use arcgraph_storage::wal::{WalRecord, WalRecordType};
use bytes::Bytes;

fn encoded_sample_bundle() -> Vec<u8> {
    let mut writes = std::collections::HashMap::new();
    writes.insert(0xCAFEu64, Some(Bytes::from_static(b"payload-ABC")));
    let mut page_bytes: Box<[u8; PAGE_SIZE]> = Box::new([0u8; PAGE_SIZE]);
    for (i, b) in page_bytes.iter_mut().enumerate() {
        *b = (i & 0xFF) as u8;
    }
    let staged = vec![StagedEmit {
        kind: arcgraph_storage::wal::BundlePageKind::PrimaryIndex,
        page_id: PageId::new(42),
        bytes: page_bytes,
    }];
    let payload = encode_commit_bundle(Lsn::new(7), &writes, &staged, TenantId::DEFAULT);

    let r = WalRecord {
        record_type: WalRecordType::CommitBundle,
        txn_id: 1,
        lsn: Lsn::new(1),
        timestamp_ms: 1_700_000_000_000,
        tenant_id: TenantId::DEFAULT,
        payload,
    };
    r.encode_to_vec().expect("sample bundle must encode")
}

#[test]
fn truncated_commit_bundle_is_invalid_record_length() {
    let bytes = encoded_sample_bundle();
    assert!(
        bytes.len() > WalRecord::HEADER_SIZE + 1024,
        "sample must be large enough to meaningfully truncate"
    );
    // Remove the last ~64 bytes — the decoder's length check reads
    // the `length` field (bytes 4..8), sees a declared total length
    // larger than `bytes.len()`, and returns InvalidRecordLength.
    let truncated = &bytes[..bytes.len() - 64];
    let err = WalRecord::decode(truncated).expect_err("truncated CommitBundle must not decode");
    match err {
        ArcGraphError::InvalidRecordLength { got, expected } => {
            assert!(
                got < expected,
                "InvalidRecordLength must report got < expected; got={got} expected={expected}"
            );
        }
        other => panic!("expected InvalidRecordLength, got {other:?}"),
    }
}

#[test]
fn crc_flipped_commit_bundle_is_wal_corruption() {
    let mut bytes = encoded_sample_bundle();
    // Flip one byte inside the CRC field (bytes 0..4). This causes
    // the stored-CRC to differ from the computed-CRC, which the
    // decoder reports as `WalCorruption { reason: "crc ..." }`.
    bytes[0] ^= 0xFF;
    let err = WalRecord::decode(&bytes).expect_err("CRC-flipped CommitBundle must not decode");
    match err {
        ArcGraphError::WalCorruption { reason, .. } => {
            assert!(
                reason.contains("crc"),
                "WalCorruption should cite 'crc' in reason; got: {reason}"
            );
        }
        other => panic!("expected WalCorruption, got {other:?}"),
    }
}

#[test]
fn payload_byte_flip_commit_bundle_is_wal_corruption() {
    // Belt-and-suspenders: a single-bit flip INSIDE the payload
    // (not the CRC) must also surface as WalCorruption because the
    // computed CRC no longer matches the stored one. This guards
    // against any future refactor that skips the CRC check for
    // `CommitBundle` specifically (e.g. if someone adds a "trusted
    // producer" shortcut).
    let mut bytes = encoded_sample_bundle();
    // Flip a byte at offset HEADER_SIZE + 100 (inside payload).
    let offset = WalRecord::HEADER_SIZE + 100;
    bytes[offset] ^= 0x01;
    let err = WalRecord::decode(&bytes).expect_err("payload-flipped CommitBundle must not decode");
    assert!(matches!(err, ArcGraphError::WalCorruption { .. }));
}
