//! M6.2 OOC-1 / INV-M6.10 security gate.
//!
//! The retained-file and encryption-disable controls exist only behind the
//! off-by-default `fault-injection` feature and retention is hard-bounded in
//! the library. Production always encrypts when tenant encryption is enabled.

#![cfg(feature = "fault-injection")]

use std::fs;

use arcgraph_core::TenantId;
use arcgraph_storage::spill::{
    SpillEncryptionPolicy, SpillError, SpillManager, SpillManagerConfig, SpillQueryConfig,
};

const NEEDLE: &[u8] = b"ARCGRAPH-M6-SPILL-NEEDLE-9f8b7a6c5d4e3f2017c6b5a493827160";
const SECOND_CHUNK: &[u8] = b"strictly-monotonic-chunk-one";

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}

fn encrypted_query_config(query_id: u64) -> SpillQueryConfig {
    let mut config = SpillQueryConfig::new(TenantId::DEFAULT, query_id, 0, 1024 * 1024);
    config.encryption = SpillEncryptionPolicy {
        tenant_encryption_enabled: true,
        force_encryption: false,
    };
    config
}

/// SECURITY-critical decisive gate: plaintext absence, authenticated restore,
/// and explicit query-end key zeroization all hold in one forced-spill path.
#[test]
fn m6_spill_scratch_encrypted() {
    let root = tempfile::tempdir().unwrap();
    let manager =
        SpillManager::new_with_fault_injection(SpillManagerConfig::new(root.path()), 2, false)
            .unwrap();
    let query = manager
        .begin_query(encrypted_query_config(0x5EC0_0001))
        .unwrap();
    let probe = query
        .key_zeroize_probe_for_test()
        .expect("encrypted query must own one ephemeral key");
    let mut key_before_drop = probe.snapshot();
    assert_ne!(key_before_drop, [0_u8; 32]);

    let mut clean_writer = query.create_run().unwrap();
    assert!(
        clean_writer.is_encrypted(),
        "tenant encryption must mandate spill encryption"
    );
    clean_writer.append_batch(NEEDLE).unwrap();
    clean_writer.append_batch(SECOND_CHUNK).unwrap();
    let clean = clean_writer.finish().unwrap();

    let mut corrupt_writer = query.create_run().unwrap();
    corrupt_writer.append_batch(NEEDLE).unwrap();
    corrupt_writer.append_batch(SECOND_CHUNK).unwrap();
    let mut corrupt = corrupt_writer.finish().unwrap();

    let clean_base = clean.nonce_base_for_test();
    let corrupt_base = corrupt.nonce_base_for_test();
    assert_ne!(
        &clean_base[..8],
        &corrupt_base[..8],
        "two runs under one query key reused their nonce domain prefix"
    );
    assert_ne!(
        clean.nonce_for_chunk_for_test(0).unwrap(),
        clean.nonce_for_chunk_for_test(1).unwrap(),
        "monotonic chunks under one run reused a nonce"
    );

    let clean_path = clean
        .retained_path_for_test()
        .expect("bounded security-gate retention")
        .to_path_buf();
    let corrupt_path = corrupt
        .retained_path_for_test()
        .expect("bounded security-gate retention")
        .to_path_buf();

    for run in [&clean, &corrupt] {
        let path = run
            .retained_path_for_test()
            .expect("bounded security-gate retention");
        let raw = fs::read(path).unwrap();
        assert!(
            !contains(&raw, NEEDLE),
            "high-entropy tenant needle appeared in retained spill plaintext"
        );
        assert!(
            !contains(&raw, &key_before_drop),
            "ephemeral query key was persisted in the run framing"
        );
    }

    // Normal authenticated restore returns the exact batch bytes.
    let mut clean_reader = clean.into_reader(query.epoch()).unwrap();
    assert_eq!(clean_reader.next_batch().unwrap().unwrap().as_ref(), NEEDLE);
    assert_eq!(
        clean_reader.next_batch().unwrap().unwrap().as_ref(),
        SECOND_CHUNK
    );
    assert!(clean_reader.next_batch().unwrap().is_none());
    drop(clean_reader);

    // A single ciphertext-bit mutation must fail loudly at the GCM tag.
    corrupt.corrupt_first_payload_byte_for_test().unwrap();
    let mut corrupt_reader = corrupt.into_reader(query.epoch()).unwrap();
    assert!(matches!(
        corrupt_reader.next_batch(),
        Err(SpillError::AuthenticationFailed { chunk: 0 })
    ));
    assert!(matches!(
        corrupt_reader.next_batch(),
        Err(SpillError::CorruptFrame { chunk: 0, .. })
    ));
    drop(corrupt_reader);

    // The monotonic counter has a hard refusal boundary; it never wraps into
    // a previously consumed nonce. The seam only jumps to that boundary and
    // is absent without `fault-injection`.
    let mut exhausted_writer = query.create_run().unwrap();
    exhausted_writer.exhaust_chunk_counter_for_test();
    assert!(matches!(
        exhausted_writer.append_batch(b"must-not-write"),
        Err(SpillError::NonceExhausted)
    ));
    drop(exhausted_writer);

    // Do not let the test's inspection copy outlive the query. The cfg-only
    // probe remains and proves the actual query-end guard wrote zeros.
    key_before_drop.fill(0);
    drop(query);
    assert!(
        probe.is_zeroized(),
        "ephemeral spill key material survives query-end drop"
    );
    let report = manager.periodic_sweep().unwrap();
    assert!(report.removed_files >= 2);
    assert!(!clean_path.exists() && !corrupt_path.exists());
}

/// Decisive RED-on-revert control: the cfg-only mutation disables encryption
/// while the tenant bit is ON; the same needle becomes visible on disk. If
/// the positive gate did not inspect the real spill bytes, this would not be
/// observable and the control would fail.
#[test]
fn encryption_off_mutation_exposes_the_needle() {
    let root = tempfile::tempdir().unwrap();
    let manager =
        SpillManager::new_with_fault_injection(SpillManagerConfig::new(root.path()), 1, true)
            .unwrap();
    let query = manager
        .begin_query(encrypted_query_config(0x5EC0_0002))
        .unwrap();
    let mut writer = query.create_run().unwrap();
    assert!(!writer.is_encrypted(), "fault mutation did not engage");
    writer.append_batch(NEEDLE).unwrap();
    let run = writer.finish().unwrap();
    let raw = fs::read(run.retained_path_for_test().unwrap()).unwrap();
    assert!(
        contains(&raw, NEEDLE),
        "negative control: disabling encryption must make the needle visible"
    );
}

#[test]
fn retention_hook_is_hard_bounded() {
    let root = tempfile::tempdir().unwrap();
    assert!(matches!(
        SpillManager::new_with_fault_injection(SpillManagerConfig::new(root.path()), 9, false,),
        Err(SpillError::InvalidConfig(_))
    ));

    let manager =
        SpillManager::new_with_fault_injection(SpillManagerConfig::new(root.path()), 1, false)
            .unwrap();
    let mut config = SpillQueryConfig::new(TenantId::DEFAULT, 0x5EC0_0003, 0, 16 * 1024 * 1024);
    config.spill_quota_bytes = Some(16 * 1024 * 1024);
    let query = manager.begin_query(config).unwrap();
    let mut writer = query.create_run().unwrap();
    assert!(matches!(
        writer.append_batch(&vec![0xA7; 1024 * 1024]),
        Err(SpillError::BatchTooLarge { .. })
    ));
}

#[test]
fn retention_claim_limit_and_cleanup_are_enforced() {
    let root = tempfile::tempdir().unwrap();
    let manager =
        SpillManager::new_with_fault_injection(SpillManagerConfig::new(root.path()), 2, false)
            .unwrap();
    let mut config = SpillQueryConfig::new(TenantId::DEFAULT, 0x5EC0_0004, 0, 16 * 1024 * 1024);
    config.spill_quota_bytes = Some(16 * 1024 * 1024);
    let query = manager.begin_query(config).unwrap();
    let mut runs = Vec::new();
    let mut retained = Vec::new();
    for index in 0..3 {
        let mut writer = query.create_run().unwrap();
        writer.append_batch(&[index]).unwrap();
        let run = writer.finish().unwrap();
        match (index, run.retained_path_for_test()) {
            (0 | 1, Some(path)) => retained.push(path.to_path_buf()),
            (2, None) => {}
            other => panic!("retention count bound violated: {other:?}"),
        }
        runs.push(run);
    }
    drop(runs);
    drop(query);
    let report = manager.periodic_sweep().unwrap();
    assert_eq!(report.removed_files, 2);
    assert!(retained.iter().all(|path| !path.exists()));
}
