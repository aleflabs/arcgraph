//! W26-γ-2 D5#7 — Negative scenario: process crash mid-recovery
//! (recovery-of-recovery / idempotent-recovery class).
//!
//! Real-world incident: ZooKeeper had a "recovery-during-recovery"
//! bug in 2013 where the second recovery attempt produced different
//! state than the first (non-idempotent recovery). Cassandra had a
//! similar class in 2016. The contract: recovery MUST be idempotent
//! across N back-to-back restart attempts.
//!
//! ArcGraph's analog: per K-2 (`tests/k2_fault_during_recovery.rs`)
//! the recovery contract is "byte-equal post-recovery state across
//! N consecutive cold-start recovery attempts." This test asserts
//! the structured error path AT the codec layer — every error a
//! second-pass recovery produces MUST be the SAME error class as the
//! first pass.

use std::io;

use arcgraph_core::error::ArcGraphError;
use arcgraph_core::{Lsn, PAGE_SIZE, PageId, TenantId};
use arcgraph_storage::wal::bundle::{StagedEmit, encode_commit_bundle};
use arcgraph_storage::wal::{WalRecord, WalRecordType};
use bytes::Bytes;

fn corrupted_bundle() -> Vec<u8> {
    let mut writes = std::collections::HashMap::new();
    writes.insert(0xCAFE_u64, Some(Bytes::from_static(b"x")));
    let mut page_bytes: Box<[u8; PAGE_SIZE]> = Box::new([0u8; PAGE_SIZE]);
    page_bytes[0] = 0xDE;
    let staged = vec![StagedEmit {
        kind: arcgraph_storage::wal::BundlePageKind::PrimaryIndex,
        page_id: PageId::new(1),
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
    let mut bytes = r.encode_to_vec().expect("encode");
    // Corrupt the CRC by flipping the first byte.
    bytes[0] ^= 0xFF;
    bytes
}

#[test]
fn recovery_of_recovery_produces_same_error_class() {
    // First pass: decode the corrupt bundle → WalCorruption.
    let bytes = corrupted_bundle();
    let first_err = WalRecord::decode(&bytes).expect_err("first pass must fail");
    // Second pass (replay the same bytes from disk): MUST produce
    // the SAME error class. Non-idempotent recovery would produce a
    // different error variant on the second attempt.
    let second_err = WalRecord::decode(&bytes).expect_err("second pass must fail");
    // Pin: both errors are WalCorruption.
    assert!(matches!(first_err, ArcGraphError::WalCorruption { .. }));
    assert!(matches!(second_err, ArcGraphError::WalCorruption { .. }));
}

#[test]
fn recovery_of_recovery_produces_same_error_message() {
    // Beyond variant equality, the operator-facing message must be
    // byte-identical across recovery passes. A message that includes
    // a nondeterministic field (timestamp, process id) would break
    // this invariant.
    let bytes = corrupted_bundle();
    let m1 = format!("{}", WalRecord::decode(&bytes).expect_err("first"));
    let m2 = format!("{}", WalRecord::decode(&bytes).expect_err("second"));
    assert_eq!(
        m1, m2,
        "recovery messages MUST be byte-identical across passes"
    );
}

#[test]
fn unrecoverable_orphans_error_variant_is_distinguishable_from_wal_corruption() {
    // ADR-032 §Slice 3c: an UnrecoverableOrphans error indicates the
    // FIRST recovery pass left orphan IndexPage records that the
    // SECOND-pass bootstrap_from_mvcc cannot recover. This is the
    // "recovery-of-recovery is unrecoverable" terminal class.
    let e = ArcGraphError::UnrecoverableOrphans {
        orphan_count: 7,
        reason: "bootstrap_from_mvcc: tenant 1 not in catalog".into(),
    };
    let display = format!("{e}");
    assert!(
        display.contains("manual recovery required"),
        "operator hint must mention manual recovery; got: {display}"
    );
    assert!(
        display.contains("bootstrap_from_mvcc"),
        "operator hint must cite the recovery API; got: {display}"
    );
    // Pattern match for operator dispatch.
    match e {
        ArcGraphError::UnrecoverableOrphans { orphan_count, .. } => {
            assert_eq!(orphan_count, 7);
        }
        _ => panic!("expected UnrecoverableOrphans"),
    }
}

#[test]
fn idempotent_decode_n_consecutive_passes() {
    // Run 5 back-to-back decode passes on the same corrupt bytes;
    // every pass MUST produce the same WalCorruption variant +
    // message. Non-idempotent decode (e.g., a refactor that
    // stateful-mutated a global on first failure) would fire.
    let bytes = corrupted_bundle();
    let messages: Vec<String> = (0..5)
        .map(|_| format!("{}", WalRecord::decode(&bytes).expect_err("pass")))
        .collect();
    // All messages MUST be identical.
    for i in 1..messages.len() {
        assert_eq!(messages[i], messages[0], "pass {i} diverged");
    }
}

#[test]
fn recovery_wal_error_rolled_back_first_and_second_pass_distinct_sources_propagate() {
    // Two distinct WAL errors → wrapped → unwrap source → identity.
    // Pin that the source-chain unwrapping is deterministic across
    // recovery passes.
    let inner1 = ArcGraphError::Io(io::Error::other("first pass eio"));
    let outer1 = ArcGraphError::WalErrorRolledBack {
        source: Box::new(inner1),
    };
    let inner2 = ArcGraphError::WalUnavailable;
    let outer2 = ArcGraphError::WalErrorRolledBack {
        source: Box::new(inner2),
    };
    // Different inner errors → different display strings (but same
    // outer variant + same prefix).
    let d1 = format!("{outer1}");
    let d2 = format!("{outer2}");
    assert_ne!(d1, d2);
    assert!(d1.starts_with("wal fsync failed"));
    assert!(d2.starts_with("wal fsync failed"));
}

#[test]
fn empty_wal_decode_is_idempotent_across_passes() {
    // Empty WAL → InvalidRecordLength on every pass.
    let empty = vec![];
    let p1 = WalRecord::decode(&empty).expect_err("first");
    let p2 = WalRecord::decode(&empty).expect_err("second");
    let p3 = WalRecord::decode(&empty).expect_err("third");
    assert!(matches!(
        p1,
        ArcGraphError::InvalidRecordLength { .. } | ArcGraphError::WalCorruption { .. }
    ));
    assert!(matches!(
        p2,
        ArcGraphError::InvalidRecordLength { .. } | ArcGraphError::WalCorruption { .. }
    ));
    assert!(matches!(
        p3,
        ArcGraphError::InvalidRecordLength { .. } | ArcGraphError::WalCorruption { .. }
    ));
    // Display strings byte-equal.
    let m1 = format!("{p1}");
    let m2 = format!("{p2}");
    let m3 = format!("{p3}");
    assert_eq!(m1, m2);
    assert_eq!(m2, m3);
}
