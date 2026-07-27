//! ADR-216 §D-2 / #1180 — the GREEN-FLIP ORACLE acceptance test for
//! WAL-at-rest encryption (MUST-KG-04).
//!
//! This mirrors the CZ acceptance proof
//! (`ENCRYPTION_AUDIT_SECRET_PLAINTEXT_TEST`): a node carrying a plaintext
//! sentinel is committed through the production CRUD path, then the on-disk
//! `wal-*.log` segments are grepped for the sentinel.
//!
//! - **Encrypted** (the `KeySource` → `bootstrap_wal_encryption` → WAL
//!   writer path that #1180 wires into `build_durable`): **0 plaintext
//!   hits** — the payload is AES-256-GCM ciphertext at rest.
//! - **Plaintext** (no `.with_encryption(..)`, the pre-#1180 posture): the
//!   sentinel appears verbatim — this is the bug #1180 closes, and the
//!   RED-on-revert oracle.
//!
//! The encrypted path here uses the SAME `SecretsProviderKeySource` +
//! `bootstrap_wal_encryption` sidecar dance that `build_durable` performs,
//! so this test exercises the production wiring (not a hand-rolled
//! `WalEncryption`). The restart round-trip leg proves the sidecar-unwrapped
//! DEK decrypts the WAL on recovery (encrypt-on-write WITHOUT
//! decrypt-on-recover would be unrecoverable WAL).

use std::sync::Arc;

use arcgraph_core::{
    EnvSecretsProvider, KekVersion, KeyScope, KeySource, LabelId, SecretValue, SecretsProvider,
    TenantId,
};
use arcgraph_storage::crud::{
    CrudStore, PropertyData, commit, create_node, crud_allocator_seed_handle, read_node_with_store,
};
use arcgraph_storage::encryption::{SecretsProviderKeySource, bootstrap_wal_encryption};
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::PrimaryIndex;
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{
    AllocatorSeedHandle, BlobStoreHandle, PageStoreTarget, PrimaryPageStoreHandle,
    RecordPageStoreHandle, WalConfig, WalWriter, recover_from_wal_encrypted,
};
use arcgraph_storage::{WalEncryption, sidecar_path};
use tempfile::TempDir;

/// The plaintext sentinel — verbatim from the CZ acceptance proof
/// (ADR-216 §Evidence). Encrypted WAL must contain ZERO occurrences of it.
const PLAINTEXT_SENTINEL: &[u8] = b"ENCRYPTION_AUDIT_SECRET_PLAINTEXT_TEST";

fn unique_prefix(suffix: &str) -> String {
    let pid = std::process::id();
    let thread_id = std::thread::current().id();
    format!("ARCGRAPH_GREENFLIP_{pid}_{thread_id:?}_{suffix}_").replace([' ', '(', ')'], "_")
}

/// Install a KEK at `arcgraph.wal.encryption_key.kek.v1` so the
/// `SecretsProviderKeySource` can resolve + wrap.
fn provider_with_kek(prefix: &str) -> Arc<dyn SecretsProvider> {
    let p: Arc<dyn SecretsProvider> = Arc::new(EnvSecretsProvider::without_startup_warn_for_tests(
        prefix.to_owned(),
    ));
    let kek_key = format!("{}.kek.v1", KeyScope::wal().namespace());
    p.set(&kek_key, SecretValue::new([0x5A; 32]))
        .expect("install KEK");
    p
}

type Stack = (
    WalWriter,
    Arc<TxnManager>,
    Arc<PrimaryIndex>,
    Arc<CrudStore>,
);

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

/// Commit a node whose blob property carries the plaintext sentinel.
fn commit_sentinel_node(stack: &Stack) -> arcgraph_core::NodeId {
    let (_writer, mgr, _primary, store) = stack;
    let mut tx = mgr.begin(TenantId::DEFAULT);
    let id = create_node(
        store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(7),
        &PropertyData::Blob(PLAINTEXT_SENTINEL.to_vec()),
    )
    .unwrap();
    commit(tx, store).unwrap();
    id
}

/// Count the plaintext-sentinel occurrences across all `wal-*.log` segments
/// in `wal_dir` (the green-flip oracle).
fn sentinel_hits(wal_dir: &std::path::Path) -> usize {
    let mut hits = 0usize;
    for entry in std::fs::read_dir(wal_dir).expect("read wal dir") {
        let entry = entry.unwrap();
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with("wal-") && name.ends_with(".log") {
            let bytes = std::fs::read(entry.path()).unwrap();
            hits += bytes
                .windows(PLAINTEXT_SENTINEL.len())
                .filter(|w| *w == PLAINTEXT_SENTINEL)
                .count();
        }
    }
    hits
}

/// THE GREEN-FLIP ORACLE: an encrypted-WAL ingest yields ZERO plaintext
/// sentinel hits in the on-disk segments (ADR-216 §D-2; the MUST-KG-04
/// acceptance test). Uses the production `KeySource` → sidecar →
/// `WalEncryption` path.
#[test]
fn green_flip_encrypted_wal_has_zero_plaintext_sentinel_hits() {
    let prefix = unique_prefix("green");
    let provider = provider_with_kek(&prefix);
    let key_source = SecretsProviderKeySource::new(Arc::clone(&provider), "env", KekVersion::ONE);

    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();

    // Production bootstrap dance: generate the wrapped DEK + persist the
    // `wal.dek` sidecar + construct the WalEncryption.
    let boot = bootstrap_wal_encryption(&key_source, &wal_dir).expect("bootstrap encryption");
    assert!(boot.freshly_generated, "first boot generates a fresh DEK");
    assert!(
        sidecar_path(&wal_dir).exists(),
        "wal.dek sidecar must be persisted"
    );

    // Encrypt-on-write: the SAME WalEncryption wired into the WAL writer.
    let cfg = WalConfig::new(wal_dir.clone()).with_encryption(boot.encryption.clone());
    let stack = build_stack(cfg);
    let _id = commit_sentinel_node(&stack);
    stack.0.shutdown().unwrap();

    // THE ORACLE: 0 plaintext hits.
    let hits = sentinel_hits(&wal_dir);
    assert_eq!(
        hits, 0,
        "GREEN-FLIP ORACLE FAILED: the plaintext sentinel \
         `ENCRYPTION_AUDIT_SECRET_PLAINTEXT_TEST` appeared {hits} time(s) in the \
         encrypted WAL segments — encryption is NOT wired (ADR-216 §D-2 / #1180)"
    );

    // The wal.dek sidecar must ALSO never contain the sentinel (it holds
    // only the WRAPPED DEK, no record payloads).
    let sidecar_bytes = std::fs::read(sidecar_path(&wal_dir)).unwrap();
    assert!(
        !sidecar_bytes
            .windows(PLAINTEXT_SENTINEL.len())
            .any(|w| w == PLAINTEXT_SENTINEL),
        "the wal.dek sidecar must never contain record-payload plaintext"
    );
}

/// RED-on-revert oracle: the SAME ingest WITHOUT `.with_encryption(..)` (the
/// pre-#1180 plaintext-WAL posture) DOES leak the sentinel. This is the bug
/// #1180 closes — neutering the encryption wiring makes the green-flip test
/// above FAIL because the sentinel is found in plaintext.
#[test]
fn red_on_revert_plaintext_wal_leaks_sentinel() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();

    // No `.with_encryption(..)` — the plaintext posture the wiring removes.
    let cfg = WalConfig::new(wal_dir.clone());
    let stack = build_stack(cfg);
    let _id = commit_sentinel_node(&stack);
    stack.0.shutdown().unwrap();

    let hits = sentinel_hits(&wal_dir);
    assert!(
        hits >= 1,
        "RED-on-revert oracle: a plaintext WAL MUST leak the sentinel verbatim \
         (this is the bug #1180 closes). Got {hits} hits — if 0, the test fixture \
         is not actually writing the sentinel into the WAL payload."
    );
}

/// Restart round-trip with encryption ON (mirrors #1025 cross-restart, but
/// encrypted): commit through an encrypted WAL, shut down, then re-bootstrap
/// from the `wal.dek` sidecar and recover — the sidecar-unwrapped DEK
/// decrypts the WAL and the record reads back intact. Proves the recovery
/// path threads the SAME encryption (encrypt-on-write without
/// decrypt-on-recover = unrecoverable WAL).
#[test]
fn restart_round_trip_recovers_encrypted_wal_via_sidecar() {
    let prefix = unique_prefix("restart");
    let provider = provider_with_kek(&prefix);
    let key_source = SecretsProviderKeySource::new(Arc::clone(&provider), "env", KekVersion::ONE);

    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();

    // Boot 1: generate sidecar + encrypt-on-write a sentinel node.
    let boot1 = bootstrap_wal_encryption(&key_source, &wal_dir).expect("boot1");
    let cfg = WalConfig::new(wal_dir.clone()).with_encryption(boot1.encryption.clone());
    let stack = build_stack(cfg);
    let id = commit_sentinel_node(&stack);
    let (writer1, mgr1, primary1, store1) = stack;
    writer1.shutdown().unwrap();
    drop(store1);
    drop(primary1);
    drop(mgr1);

    // 0 plaintext hits on disk.
    assert_eq!(
        sentinel_hits(&wal_dir),
        0,
        "encrypted WAL must have 0 plaintext hits"
    );

    // Boot 2 (restart): a fresh KeySource over the same provider re-reads
    // the sidecar + unwraps the DEK → recovery decrypts the WAL.
    let key_source2 = SecretsProviderKeySource::new(Arc::clone(&provider), "env", KekVersion::ONE);
    let boot2 = bootstrap_wal_encryption(&key_source2, &wal_dir).expect("boot2");
    assert!(
        !boot2.freshly_generated,
        "restart must read the existing sidecar"
    );

    let store = recover_encrypted_stack(&wal_dir, boot2.encryption.clone())
        .expect("recover encrypted WAL via sidecar-unwrapped DEK");
    let tx = store.1.begin(TenantId::DEFAULT);
    let rec = read_node_with_store(&store.3, &tx, id)
        .unwrap()
        .expect("sentinel node must be readable post-restart");
    assert_eq!(rec.label_id, 7);
    store.0.shutdown().unwrap();
}

/// Fail-closed startup: the KeySource health_check refuses when the KEK is
/// absent — the bootstrap returns an error rather than producing a plaintext
/// WAL (ADR-033 fail-closed; no `wal.dek` is left behind).
#[test]
fn fail_closed_when_kek_absent_no_plaintext_wal() {
    let prefix = unique_prefix("failclosed");
    // Provider WITHOUT a KEK installed.
    let provider: Arc<dyn SecretsProvider> =
        Arc::new(EnvSecretsProvider::without_startup_warn_for_tests(prefix));
    let key_source = SecretsProviderKeySource::new(Arc::clone(&provider), "env", KekVersion::ONE);

    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();

    // The fail-fast probe `build_durable` performs first.
    assert!(
        key_source.health_check(&KeyScope::wal()).is_err(),
        "health_check must fail-closed when the KEK is absent (ADR-033)"
    );
    // And the bootstrap itself refuses (no sidecar, no plaintext WAL).
    let res = bootstrap_wal_encryption(&key_source, &wal_dir);
    assert!(
        res.is_err(),
        "bootstrap must fail-closed when the KEK is absent"
    );
    assert!(
        !sidecar_path(&wal_dir).exists(),
        "fail-closed must not leave a wal.dek behind"
    );
    assert_eq!(
        sentinel_hits(&wal_dir),
        0,
        "fail-closed must not have written any WAL"
    );
}

// ── helpers ────────────────────────────────────────────────────────────

fn recover_encrypted_stack(
    wal_dir: &std::path::Path,
    encryption: WalEncryption,
) -> Result<Stack, arcgraph_core::ArcGraphError> {
    let cfg = WalConfig::new(wal_dir.to_path_buf()).with_encryption(encryption.clone());
    let writer = WalWriter::spawn(cfg).expect("spawn");
    let handle = writer.handle();
    let mgr = Arc::new(TxnManager::with_wal(handle.clone()));
    let alloc = Arc::new(PageAllocator::new());
    let primary = Arc::new(
        PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&alloc), Some(handle.clone()))
            .expect("primary"),
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
    recover_from_wal_encrypted(wal_dir, Arc::clone(&mgr), target, None, Some(encryption))?;
    Ok((writer, mgr, primary, store))
}
