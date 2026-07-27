//! ADR-216 §D-4 / #1180 — END-TO-END `build_durable` WAL-encryption wiring.
//!
//! Drives the FULL production bootstrap surface
//! (`bootstrap_storage_backend_with_metrics_and_encryption`) with
//! `WalEncryptionConfig.enabled = true` against a real `--data <dir>`, then:
//!
//! - **green-flip oracle** — a node carrying the plaintext sentinel
//!   `ENCRYPTION_AUDIT_SECRET_PLAINTEXT_TEST` is committed through the served
//!   CRUD path; the on-disk `wal-*.log` segments contain ZERO plaintext
//!   hits (the MUST-KG-04 acceptance test, at the `build_durable` layer);
//! - **restart round-trip** — re-bootstrapping the SAME dir reads the
//!   `wal.dek` sidecar, unwraps the DEK, decrypts the WAL on recovery, and
//!   reads the node back intact (encrypt-on-write + decrypt-on-recover
//!   threaded symmetrically through `build_durable`);
//! - **disabled = plaintext** — the same ingest with encryption OFF (the
//!   v1.0-α default) leaks the sentinel (the RED-on-revert sibling pin).
//!
//! Tests use the `env` secrets provider (the dev provider) with the KEK
//! installed via a per-test `ARCGRAPH_SECRET_*` env var, so they run in
//! headless CI WITHOUT a real OS keyring (per
//! `feedback_test_env_gate_panic_by_default` — the OS-keyring path is the
//! prod default; a keyring-requiring test would gate behind an
//! env-flag-PANIC-by-default, but these tests deliberately use the
//! env provider so no keyring is needed).

use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use arcgraph_cli::bootstrap::{
    BootstrapMode, SecretsProviderKind, WalEncryptionConfig,
    bootstrap_storage_backend_with_metrics_and_encryption,
};
use arcgraph_core::{KeyScope, LabelId, PartitionId, TenantId};
use arcgraph_storage::crud::{CrudStore, PropertyData, commit, create_node, read_node_with_store};
use tempfile::TempDir;

const PLAINTEXT_SENTINEL: &[u8] = b"ENCRYPTION_AUDIT_SECRET_PLAINTEXT_TEST";

/// These tests mutate the process-global `ARCGRAPH_SECRET_*` env table (the
/// env provider reads it). Cargo runs tests in this binary on parallel
/// threads, so installing-vs-removing the KEK across tests races. Serialize
/// the env-touching tests behind one mutex.
fn env_guard() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|p| p.into_inner())
}

/// Map a provider key → the `EnvSecretsProvider` env-var name (default
/// `ARCGRAPH_SECRET_` prefix; `.`→`_DOT_`; uppercased).
fn env_var_for_key(key: &str) -> String {
    let canonical = key.replace('.', "_DOT_");
    format!("ARCGRAPH_SECRET_{}", canonical.to_uppercase())
}

/// Install a 32-byte KEK (hex) into the process env at the key the
/// `SecretsProviderKeySource` (env provider) reads:
/// `arcgraph.wal.encryption_key.kek.v1`.
fn install_kek_env() {
    let kek_key = format!("{}.kek.v1", KeyScope::wal().namespace());
    let var = env_var_for_key(&kek_key);
    // 64 hex chars = 32 bytes.
    let hex = "5a".repeat(32);
    // SAFETY: set_var is unsafe in 2024 edition (POSIX env not thread-safe).
    // These tests run single-threaded per-process for the env-var setup; the
    // env provider's own write_lock guards concurrent provider mutations.
    unsafe {
        std::env::set_var(&var, hex);
    }
}

fn enabled_env_config() -> WalEncryptionConfig {
    WalEncryptionConfig {
        enabled: true,
        key_source: arcgraph_cli::bootstrap::KeySourceKind::default(),
        secrets_provider: SecretsProviderKind::Env,
    }
}

fn crud_for(backend: &arcgraph_mcp::storage::StorageBackend, tenant: TenantId) -> Arc<CrudStore> {
    backend
        .router()
        .route(tenant, PartitionId::ZERO)
        .expect("route tenant")
        .crud()
        .clone()
}

fn commit_sentinel_node(
    backend: &arcgraph_mcp::storage::StorageBackend,
    crud: &Arc<CrudStore>,
) -> arcgraph_core::NodeId {
    let mut tx = backend.txn_manager().begin(TenantId::DEFAULT);
    let id = create_node(
        crud,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(7),
        &PropertyData::Blob(PLAINTEXT_SENTINEL.to_vec()),
    )
    .expect("create_node");
    commit(tx, crud).expect("commit");
    id
}

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

/// THE `build_durable` GREEN-FLIP ORACLE + restart round-trip: encryption
/// enabled → 0 plaintext hits on disk; restart recovers via the `wal.dek`
/// sidecar.
#[test]
fn build_durable_encrypted_green_flip_and_restart_round_trip() {
    let _env = env_guard();
    install_kek_env();
    let tmp = TempDir::new().expect("tempdir");
    let data_dir = tmp.path().join("db");
    let wal_dir = data_dir.join("wal");
    let cfg = enabled_env_config();

    // ── Session 1: durable bootstrap WITH encryption; commit a sentinel node.
    let id = {
        let (backend, guard) = bootstrap_storage_backend_with_metrics_and_encryption(
            &BootstrapMode::Durable {
                data_dir: data_dir.clone(),
            },
            None,
            &cfg,
        )
        .expect("durable bootstrap with encryption (session 1)");
        assert!(guard.is_durable(), "Durable mode must own a WAL writer");
        let crud = crud_for(&backend, TenantId::DEFAULT);
        let id = commit_sentinel_node(&backend, &crud);
        // Drop the guard → WAL drains + fsyncs + joins ("restart").
        id
    };

    // The wal.dek sidecar must exist (first-boot generated it).
    assert!(
        wal_dir.join("wal.dek").exists(),
        "build_durable must persist the wal.dek sidecar when encryption is enabled"
    );

    // GREEN-FLIP ORACLE: 0 plaintext sentinel hits in the encrypted WAL.
    let hits = sentinel_hits(&wal_dir);
    assert_eq!(
        hits, 0,
        "build_durable GREEN-FLIP ORACLE FAILED: plaintext sentinel appeared {hits} time(s) \
         in the encrypted WAL — encryption is NOT wired into build_durable (ADR-216 §D-4 / #1180)"
    );

    // ── Session 2: re-bootstrap the SAME dir WITH encryption → recovery
    //    unwraps the DEK via the sidecar + decrypts the WAL.
    let (backend2, _guard2) = bootstrap_storage_backend_with_metrics_and_encryption(
        &BootstrapMode::Durable {
            data_dir: data_dir.clone(),
        },
        None,
        &cfg,
    )
    .expect("durable bootstrap with encryption (session 2 — recover)");
    let crud2 = crud_for(&backend2, TenantId::DEFAULT);
    let tx = backend2.txn_manager().begin(TenantId::DEFAULT);
    let rec = read_node_with_store(&crud2, &tx, id)
        .expect("read node")
        .expect("sentinel node must be readable post-restart via the sidecar-unwrapped DEK");
    assert_eq!(rec.label_id, 7, "recovered node label must match");
}

/// Sibling pin: with encryption DISABLED (the v1.0-α default), the SAME
/// `build_durable` ingest leaks the sentinel in plaintext — the
/// RED-on-revert posture #1180 removes when an operator enables encryption.
#[test]
fn build_durable_disabled_default_leaks_plaintext_sentinel() {
    let tmp = TempDir::new().expect("tempdir");
    let data_dir = tmp.path().join("db");
    let wal_dir = data_dir.join("wal");

    let (backend, _guard) = bootstrap_storage_backend_with_metrics_and_encryption(
        &BootstrapMode::Durable {
            data_dir: data_dir.clone(),
        },
        None,
        // The DEFAULT config: enabled = false (OQ-2 — v1.0-α opt-in).
        &WalEncryptionConfig::default(),
    )
    .expect("durable bootstrap, encryption disabled");
    let crud = crud_for(&backend, TenantId::DEFAULT);
    let _id = commit_sentinel_node(&backend, &crud);
    drop(backend);

    // No sidecar when disabled.
    assert!(
        !wal_dir.join("wal.dek").exists(),
        "disabled encryption must NOT create a wal.dek sidecar"
    );
    // Plaintext WAL: the sentinel leaks (the bug encryption closes).
    assert!(
        sentinel_hits(&wal_dir) >= 1,
        "disabled (plaintext) WAL must leak the sentinel — this is the posture \
         WAL encryption removes when enabled (ADR-216 §D-4 / #1180)"
    );
}

/// Fail-closed: `enabled = true` with NO KEK in the provider → build_durable
/// REFUSES to start (ADR-033). No plaintext WAL is written.
#[test]
fn build_durable_enabled_without_kek_fails_closed() {
    let _env = env_guard();
    // Deliberately do NOT install the KEK env var. Use a unique env-provider
    // namespace so a KEK installed by a sibling test cannot leak in: the env
    // provider reads the process env directly, so we point at a key that is
    // guaranteed absent by clearing it first.
    let kek_key = format!("{}.kek.v1", KeyScope::wal().namespace());
    let var = env_var_for_key(&kek_key);
    // SAFETY: see install_kek_env. Clear any KEK a sibling test set so this
    // test deterministically observes the absent-KEK fail-closed path.
    unsafe {
        std::env::remove_var(&var);
    }

    let tmp = TempDir::new().expect("tempdir");
    let data_dir = tmp.path().join("db");
    let wal_dir = data_dir.join("wal");
    let cfg = enabled_env_config();

    let res = bootstrap_storage_backend_with_metrics_and_encryption(
        &BootstrapMode::Durable {
            data_dir: data_dir.clone(),
        },
        None,
        &cfg,
    );
    assert!(
        res.is_err(),
        "build_durable with encryption enabled but no KEK must FAIL-CLOSED (ADR-033), not serve"
    );
    // No plaintext WAL records were written (the failure is before the
    // writer spawns / commits). The wal.dek must NOT exist either.
    assert!(
        !wal_dir.join("wal.dek").exists(),
        "fail-closed startup must not leave a wal.dek behind"
    );
    if wal_dir.exists() {
        assert_eq!(
            sentinel_hits(&wal_dir),
            0,
            "fail-closed startup must not have written any plaintext WAL records"
        );
    }
}
