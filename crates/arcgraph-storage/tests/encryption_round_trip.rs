//! W20β-3 / ADR-052 encryption round-trip tests.
//!
//! Pins the WAL + page encryption integration with the production
//! commit path. Mirrors `wal_replay_round_trip.rs` (the canonical
//! recovery-correctness pin) but flips the WAL writer config to
//! `with_encryption(...)` and replays through the encryption-aware
//! recovery entry point.
//!
//! Tests:
//!
//! 1. `round_trip_encrypted_wal` — clean shutdown + recovery with
//!    encryption. Per-record AEAD wrap is transparent to the CRUD
//!    surface.
//! 2. `mixed_clear_and_encrypted_records` — switching encryption ON
//!    mid-stream: pre-rotation clear records + post-rotation
//!    encrypted records are both recovered correctly.
//! 3. `key_rotation_mid_session` — install v2, rotate, write under
//!    v2, recover all (both v1 + v2) records.
//! 4. `missing_key_surfaces_decryption_failed` — recovering with no
//!    encryption config on an encrypted WAL surfaces
//!    `WalDecryptionFailed` (NOT silent skip).
//! 5. `wrong_key_surfaces_decryption_failed` — recovering with a
//!    different key under the same version surfaces
//!    `WalDecryptionFailed` via the GCM tag.

use std::sync::Arc;

use arcgraph_core::{
    ArcGraphError, EnvSecretsProvider, KeyVersion, LabelId, SecretValue, SecretsProvider, TenantId,
};
use arcgraph_storage::crud::{
    CrudStore, PropertyData, commit, create_node, crud_allocator_seed_handle, read_node_with_store,
};
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::PrimaryIndex;
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{
    AllocatorSeedHandle, BlobStoreHandle, PageStoreTarget, PrimaryPageStoreHandle,
    RecordPageStoreHandle, WalConfig, WalWriter, recover_from_wal_encrypted,
};
use arcgraph_storage::{ENCRYPTION_KEY_NAMESPACE_WAL, WalEncryption};
use tempfile::TempDir;

fn unique_prefix(suffix: &str) -> String {
    let pid = std::process::id();
    let thread_id = std::thread::current().id();
    format!("ARCGRAPH_WAL_ROUND_TRIP_{pid}_{thread_id:?}_{suffix}_").replace([' ', '(', ')'], "_")
}

fn build_provider_with_wal_v1(prefix: &str) -> Arc<dyn SecretsProvider> {
    let p: Arc<dyn SecretsProvider> = Arc::new(EnvSecretsProvider::without_startup_warn_for_tests(
        prefix.to_owned(),
    ));
    arcgraph_storage::install_random_key(&*p, ENCRYPTION_KEY_NAMESPACE_WAL, KeyVersion::ONE)
        .expect("install v1");
    p
}

fn encrypted_wal_config(dir: &std::path::Path, enc: WalEncryption) -> WalConfig {
    WalConfig::new(dir.to_path_buf()).with_encryption(enc)
}

fn small_segment_encrypted_wal_config(dir: &std::path::Path, enc: WalEncryption) -> WalConfig {
    WalConfig {
        segment_size_bytes: 512,
        ..WalConfig::new(dir.to_path_buf()).with_encryption(enc)
    }
}

fn clear_wal_config(dir: &std::path::Path) -> WalConfig {
    WalConfig::new(dir.to_path_buf())
}

fn wal_segment_count(dir: &std::path::Path) -> usize {
    std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            name.starts_with("wal-") && name.ends_with(".log")
        })
        .count()
}

fn build_stack(cfg: WalConfig) -> Stack {
    let writer = WalWriter::spawn(cfg).unwrap();
    let handle = writer.handle();
    let mgr = Arc::new(TxnManager::with_wal(handle.clone()));
    let alloc = Arc::new(PageAllocator::new());
    let primary = Arc::new(
        PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&alloc), Some(handle.clone())).unwrap(),
    );
    let store = Arc::new(CrudStore::new_with_index(
        Some(handle.clone()),
        Arc::clone(&primary),
        Arc::clone(&alloc),
    ));
    (writer, mgr, primary, store)
}

/// Tuple alias for the stack components returned by [`build_stack`] +
/// [`recover_stack`]. Avoids clippy::type_complexity on the recover
/// helper's signature.
type Stack = (
    WalWriter,
    Arc<TxnManager>,
    Arc<PrimaryIndex>,
    Arc<CrudStore>,
);

/// Recover a stack. Returns the WalWriter + state on success, or the
/// `ArcGraphError` that surfaced from `recover_from_wal_encrypted`.
/// Non-recovery errors (e.g., PrimaryIndex setup) panic — the tests
/// use this helper only on paths where setup is already known good.
fn recover_stack(
    cfg: WalConfig,
    encryption: Option<WalEncryption>,
) -> Result<Stack, ArcGraphError> {
    let dir = cfg.dir.clone();
    let writer = WalWriter::spawn(cfg).expect("WalWriter::spawn");
    let handle = writer.handle();
    let mgr = Arc::new(TxnManager::with_wal(handle.clone()));
    let alloc = Arc::new(PageAllocator::new());
    let primary = Arc::new(
        PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&alloc), Some(handle.clone()))
            .expect("PrimaryIndex::new"),
    );
    let store = Arc::new(CrudStore::new_with_index(
        Some(handle.clone()),
        Arc::clone(&primary),
        Arc::clone(&alloc),
    ));
    let primary_handle: Arc<dyn PrimaryPageStoreHandle> =
        Arc::clone(primary.page_store()) as Arc<dyn PrimaryPageStoreHandle>;
    let records_handle: Arc<dyn RecordPageStoreHandle> =
        Arc::clone(store.records().expect("records")) as Arc<dyn RecordPageStoreHandle>;
    let blob_handle: Arc<dyn BlobStoreHandle> =
        Arc::clone(store.blob_store()) as Arc<dyn BlobStoreHandle>;
    let allocator_seed: Arc<dyn AllocatorSeedHandle> =
        crud_allocator_seed_handle(Arc::clone(&store), Arc::clone(&alloc));
    let target = PageStoreTarget::primary_only(primary_handle)
        .with_record_store(records_handle)
        .with_blob_store(blob_handle)
        .with_allocator_seed(allocator_seed);
    recover_from_wal_encrypted(&dir, Arc::clone(&mgr), target, None, encryption)?;
    Ok((writer, mgr, primary, store))
}

#[test]
fn round_trip_encrypted_wal() {
    let prefix = unique_prefix("round_trip");
    let provider = build_provider_with_wal_v1(&prefix);
    let enc = WalEncryption::new(Arc::clone(&provider), KeyVersion::ONE).expect("init encryption");

    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();

    // Pre-crash: encrypted commits.
    let cfg = encrypted_wal_config(&wal_dir, enc.clone());
    let (writer, mgr, primary, store) = build_stack(cfg);
    let mut tx = mgr.begin(TenantId::DEFAULT);
    let id = create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(7),
        &PropertyData::InlineU32Pair(11, 22),
    )
    .unwrap();
    commit(tx, &store).unwrap();
    writer.shutdown().unwrap();
    drop(store);
    drop(primary);
    drop(mgr);

    // The on-disk WAL bytes should NOT contain the plaintext (11,22)
    // pair as raw little-endian. Best-effort check that the encryption
    // is actually doing something.
    let segments: Vec<_> = std::fs::read_dir(&wal_dir).unwrap().collect();
    assert!(!segments.is_empty(), "WAL must have at least one segment");
    for seg in segments {
        let seg = seg.unwrap();
        let bytes = std::fs::read(seg.path()).unwrap();
        // Skip segment header (8 B) — search the body.
        let body = &bytes[8..];
        // 22 little-endian as a u32 ≠ in encrypted body.
        // We can't guarantee no incidental occurrence of one byte, but
        // the AEAD magic SHOULD appear in the encrypted records.
        let has_aead_magic = body.windows(4).any(|w| w == b"AEAD");
        assert!(
            has_aead_magic,
            "encrypted WAL segment must contain the AEAD magic somewhere"
        );
    }

    // Post-crash: recover + verify.
    let cfg2 = encrypted_wal_config(&wal_dir, enc.clone());
    let (writer2, mgr2, _primary2, store2) =
        recover_stack(cfg2, Some(enc)).expect("recover with encryption");
    let tx2 = mgr2.begin(TenantId::DEFAULT);
    let rec = read_node_with_store(&store2, &tx2, id)
        .unwrap()
        .expect("node should be readable post-replay");
    assert_eq!(rec.label_id, 7);
    assert_eq!(rec.inline_u32a, 11);
    assert_eq!(rec.inline_u32b, 22);
    writer2.shutdown().unwrap();
}

#[test]
fn encrypted_wal_round_trip_across_segment_rotations() {
    let prefix = unique_prefix("segment_rotation");
    let provider = build_provider_with_wal_v1(&prefix);
    let enc = WalEncryption::new(Arc::clone(&provider), KeyVersion::ONE).expect("init encryption");

    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();

    let cfg = small_segment_encrypted_wal_config(&wal_dir, enc.clone());
    let (writer, mgr, primary, store) = build_stack(cfg);
    let mut ids = Vec::new();
    for i in 0u32..24 {
        let mut tx = mgr.begin(TenantId::DEFAULT);
        let id = create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(i + 10),
            &PropertyData::InlineU32Pair(i * 3, i * 5),
        )
        .unwrap();
        commit(tx, &store).unwrap();
        ids.push((id, i));
    }
    writer.shutdown().unwrap();
    drop(store);
    drop(primary);
    drop(mgr);

    assert!(
        wal_segment_count(&wal_dir) >= 3,
        "test must cross at least two encrypted WAL segment rotations"
    );

    let cfg2 = small_segment_encrypted_wal_config(&wal_dir, enc.clone());
    let (writer2, mgr2, _primary2, store2) =
        recover_stack(cfg2, Some(enc)).expect("recover encrypted WAL across rotations");
    let tx2 = mgr2.begin(TenantId::DEFAULT);
    for (id, i) in ids {
        let rec = read_node_with_store(&store2, &tx2, id)
            .unwrap()
            .unwrap_or_else(|| panic!("node {id:?} should replay after encrypted WAL rotation"));
        assert_eq!(rec.inline_u32a, i * 3);
        assert_eq!(rec.inline_u32b, i * 5);
    }
    writer2.shutdown().unwrap();
}

/// Mixed-mode: a WAL written WITHOUT encryption, then later written
/// WITH encryption (after rotating up the writer). Recovery with
/// encryption config decodes BOTH (clear records pass through; AEAD
/// records get decrypted).
#[test]
fn mixed_clear_and_encrypted_records() {
    let prefix = unique_prefix("mixed");
    let provider = build_provider_with_wal_v1(&prefix);
    let enc = WalEncryption::new(Arc::clone(&provider), KeyVersion::ONE).unwrap();

    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();

    // Phase 1: clear WAL.
    let (writer, mgr, primary, store) = build_stack(clear_wal_config(&wal_dir));
    let mut tx = mgr.begin(TenantId::DEFAULT);
    let id_clear = create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(1),
        &PropertyData::InlineU32Pair(100, 200),
    )
    .unwrap();
    commit(tx, &store).unwrap();
    writer.shutdown().unwrap();
    drop(store);
    drop(primary);
    drop(mgr);

    // Phase 2: same WAL dir, now WITH encryption.
    let (writer2, mgr2, _primary2, store2) = recover_stack(
        encrypted_wal_config(&wal_dir, enc.clone()),
        Some(enc.clone()),
    )
    .expect("recover with encryption (mixed clear+enc)");
    // Verify the clear-phase record is still readable.
    let tx2 = mgr2.begin(TenantId::DEFAULT);
    let rec = read_node_with_store(&store2, &tx2, id_clear)
        .unwrap()
        .expect("clear-phase node readable post-mixed-recovery");
    assert_eq!(rec.inline_u32a, 100);
    drop(tx2);
    // Now write an encrypted commit.
    let mut tx3 = mgr2.begin(TenantId::DEFAULT);
    let id_enc = create_node(
        &store2,
        &mut tx3,
        TenantId::DEFAULT,
        LabelId::new(2),
        &PropertyData::InlineU32Pair(300, 400),
    )
    .unwrap();
    commit(tx3, &store2).unwrap();
    writer2.shutdown().unwrap();
    drop(store2);
    drop(_primary2);
    drop(mgr2);

    // Phase 3: recover with encryption and verify BOTH records survive.
    let (writer3, mgr3, _, store3) =
        recover_stack(encrypted_wal_config(&wal_dir, enc.clone()), Some(enc))
            .expect("recover post-mixed");
    let tx3 = mgr3.begin(TenantId::DEFAULT);
    let r1 = read_node_with_store(&store3, &tx3, id_clear)
        .unwrap()
        .unwrap();
    let r2 = read_node_with_store(&store3, &tx3, id_enc)
        .unwrap()
        .unwrap();
    assert_eq!(r1.inline_u32a, 100);
    assert_eq!(r1.inline_u32b, 200);
    assert_eq!(r2.inline_u32a, 300);
    assert_eq!(r2.inline_u32b, 400);
    writer3.shutdown().unwrap();
}

/// Mid-session key rotation: write under v1, install v2, rotate, write
/// under v2, recover all records. Old (v1) records still decryptable
/// via the in-record key_version routing.
#[test]
fn key_rotation_mid_session() {
    let prefix = unique_prefix("rotate");
    let provider = build_provider_with_wal_v1(&prefix);
    let enc = WalEncryption::new(Arc::clone(&provider), KeyVersion::ONE).unwrap();

    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();

    let (writer, mgr, _primary, store) = build_stack(encrypted_wal_config(&wal_dir, enc.clone()));
    let mut tx = mgr.begin(TenantId::DEFAULT);
    let id_v1 = create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(1),
        &PropertyData::InlineU32Pair(11, 22),
    )
    .unwrap();
    commit(tx, &store).unwrap();

    // Rotate: install v2 + advance.
    arcgraph_storage::install_random_key(
        &*provider,
        ENCRYPTION_KEY_NAMESPACE_WAL,
        KeyVersion::new(2),
    )
    .expect("install v2");
    enc.rotate_to(KeyVersion::new(2)).expect("rotate");
    assert_eq!(enc.current_version(), KeyVersion::new(2));

    // Write under v2.
    let mut tx2 = mgr.begin(TenantId::DEFAULT);
    let id_v2 = create_node(
        &store,
        &mut tx2,
        TenantId::DEFAULT,
        LabelId::new(2),
        &PropertyData::InlineU32Pair(33, 44),
    )
    .unwrap();
    commit(tx2, &store).unwrap();

    writer.shutdown().unwrap();
    drop(store);
    drop(_primary);
    drop(mgr);

    // Recover at v2 — both v1 + v2 records must read.
    let enc_post = WalEncryption::new(Arc::clone(&provider), KeyVersion::new(2)).unwrap();
    let (writer3, mgr3, _, store3) = recover_stack(
        encrypted_wal_config(&wal_dir, enc_post.clone()),
        Some(enc_post),
    )
    .expect("recover post-rotation");
    let tx3 = mgr3.begin(TenantId::DEFAULT);
    let r1 = read_node_with_store(&store3, &tx3, id_v1).unwrap().unwrap();
    let r2 = read_node_with_store(&store3, &tx3, id_v2).unwrap().unwrap();
    assert_eq!(r1.inline_u32a, 11);
    assert_eq!(r1.inline_u32b, 22);
    assert_eq!(r2.inline_u32a, 33);
    assert_eq!(r2.inline_u32b, 44);
    writer3.shutdown().unwrap();
}

/// Missing key on recovery: write encrypted WAL → drop provider →
/// open recovery WITHOUT encryption config → must surface
/// `WalDecryptionFailed`, NOT silent skip.
#[test]
fn missing_key_surfaces_wal_decryption_failed() {
    let prefix = unique_prefix("no_key");
    let provider = build_provider_with_wal_v1(&prefix);
    let enc = WalEncryption::new(Arc::clone(&provider), KeyVersion::ONE).unwrap();

    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();

    let (writer, mgr, _primary, store) = build_stack(encrypted_wal_config(&wal_dir, enc));
    let mut tx = mgr.begin(TenantId::DEFAULT);
    let _id = create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(1),
        &PropertyData::Empty,
    )
    .unwrap();
    commit(tx, &store).unwrap();
    writer.shutdown().unwrap();
    drop(store);
    drop(_primary);
    drop(mgr);

    // Recovery WITHOUT encryption config: must fail with WalDecryptionFailed.
    let err = match recover_stack(clear_wal_config(&wal_dir), None) {
        Ok(_) => panic!("recovery without key MUST fail"),
        Err(e) => e,
    };
    match err {
        ArcGraphError::WalDecryptionFailed { reason, .. } => {
            assert!(
                reason.contains("WalRecoveryReader has no encryption config"),
                "expected the structured 'no encryption config' message, got: {reason}"
            );
        }
        other => panic!("expected WalDecryptionFailed, got {other:?}"),
    }
}

/// Wrong key on recovery: write encrypted WAL → install a DIFFERENT
/// v1 key → recover → must surface `WalDecryptionFailed` via the GCM
/// tag mismatch.
#[test]
fn wrong_key_surfaces_wal_decryption_failed() {
    let prefix_a = unique_prefix("wrong_a");
    let prefix_b = unique_prefix("wrong_b");

    let provider_a = build_provider_with_wal_v1(&prefix_a);
    let enc_a = WalEncryption::new(Arc::clone(&provider_a), KeyVersion::ONE).unwrap();

    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();

    let (writer, mgr, _primary, store) = build_stack(encrypted_wal_config(&wal_dir, enc_a));
    let mut tx = mgr.begin(TenantId::DEFAULT);
    let _id = create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(1),
        &PropertyData::Empty,
    )
    .unwrap();
    commit(tx, &store).unwrap();
    writer.shutdown().unwrap();
    drop(store);
    drop(_primary);
    drop(mgr);

    // Recover with a DIFFERENT provider's v1 key (different bytes,
    // same version).
    let provider_b = build_provider_with_wal_v1(&prefix_b);
    let enc_b = WalEncryption::new(Arc::clone(&provider_b), KeyVersion::ONE).unwrap();
    let err = match recover_stack(encrypted_wal_config(&wal_dir, enc_b.clone()), Some(enc_b)) {
        Ok(_) => panic!("recovery with wrong key MUST fail"),
        Err(e) => e,
    };
    match err {
        ArcGraphError::WalDecryptionFailed { reason, .. } => {
            assert!(
                reason.contains("aead decryption failed") || reason.contains("tag"),
                "expected GCM tag mismatch, got: {reason}"
            );
        }
        other => panic!("expected WalDecryptionFailed (GCM tag), got {other:?}"),
    }
}

/// WAL writer's encryption knob can be set explicitly via the
/// builder + observed via the API. (Smoke-tests the builder ergonomics
/// reviewers care about.)
#[test]
fn wal_config_builder_sets_encryption() {
    let prefix = unique_prefix("builder");
    let provider = build_provider_with_wal_v1(&prefix);
    let enc = WalEncryption::new(Arc::clone(&provider), KeyVersion::ONE).unwrap();
    assert_eq!(enc.current_version(), KeyVersion::ONE);
    let tmp = TempDir::new().unwrap();
    let cfg = WalConfig::new(tmp.path().to_path_buf()).with_encryption(enc);
    assert!(cfg.encryption.is_some());
    let _ = SecretValue::new([0; arcgraph_core::SECRET_VALUE_LEN]); // silence unused
}
