//! SVC-2 / #1302 — on-disk data-dir version stamp + boot-time guard
//! (upgrade-safety), exercised through the PRODUCTION bootstrap seam.
//!
//! These tests drive `arcgraph_cli::bootstrap::bootstrap_storage_backend`
//! over a real `--data <dir>` — the exact surface `arcgraph serve` reaches —
//! so they prove the version guard is wired into the durable-open path, not
//! just unit-tested in the codec. The core oracle is
//! [`boot_refuses_incompatible_on_disk_version_1302`]: a durable dir stamped
//! at the current version, tampered to a DIFFERENT version, fails LOUD on
//! re-open with the actionable incompatible-version error — instead of the
//! misparse the issue describes.
//!
//! Mirrors the WAL-format guard proven by
//! `crates/arcgraph-storage/tests/wal_format_versioning.rs` and the durable
//! bootstrap tests in `durable_bootstrap_restart.rs`.

use arcgraph_cli::bootstrap::{
    BootstrapMode, bootstrap_storage_backend,
    bootstrap_storage_backend_with_metrics_encryption_and_adopt,
};
use arcgraph_core::{LabelId, NodeId, PartitionId, TenantId};
use arcgraph_storage::crud::{CrudStore, PropertyData, commit, create_node, read_node_with_store};
use arcgraph_storage::data_dir_version::{
    DATA_DIR_FORMAT_VERSION, DATA_DIR_VERSION_MAGIC, VERSION_FILE, version_file_path,
};
use std::sync::Arc;
use tempfile::TempDir;

/// Bootstrap a durable dir with the explicit `--adopt-legacy-datadir` opt-in
/// (encryption OFF, no metrics — the adopt path under test). Mirrors what the
/// serve binary's `--adopt-legacy-datadir` flag threads through.
fn bootstrap_durable_adopt(
    data_dir: &std::path::Path,
) -> anyhow::Result<(
    arcgraph_mcp::storage::StorageBackend,
    arcgraph_cli::bootstrap::DurabilityGuard,
)> {
    bootstrap_storage_backend_with_metrics_encryption_and_adopt(
        &BootstrapMode::Durable {
            data_dir: data_dir.to_path_buf(),
        },
        None,
        &arcgraph_cli::bootstrap::WalEncryptionConfig::default(),
        true, // adopt_legacy
    )
}

/// The shared per-tenant `CrudStore` for `tenant` via the production router.
fn crud_for(backend: &arcgraph_mcp::storage::StorageBackend, tenant: TenantId) -> Arc<CrudStore> {
    backend
        .router()
        .route(tenant, PartitionId::ZERO)
        .expect("route tenant")
        .crud()
        .clone()
}

/// Commit one node under DEFAULT so the dir genuinely holds durable data
/// (a `pages.db` + a non-empty WAL) — a realistic "existing dir" for the
/// legacy / restart cases. Returns the committed `NodeId` for readback.
fn commit_one_node(backend: &arcgraph_mcp::storage::StorageBackend) -> arcgraph_core::NodeId {
    let crud = crud_for(backend, TenantId::DEFAULT);
    let mut tx = backend.txn_manager().begin(TenantId::DEFAULT);
    let id = create_node(
        &crud,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(7),
        &PropertyData::InlineU32Pair(11, 22),
    )
    .expect("create_node");
    commit(tx, &crud).expect("commit node");
    id
}

/// Encode a 12-byte VERSION file body at an arbitrary `version` (mirrors the
/// module's private `encode_version_file`, kept local to the test so we can
/// forge an incompatible stamp).
fn forge_version_body(version: u16) -> [u8; 12] {
    let mut out = [0u8; 12];
    out[0..8].copy_from_slice(DATA_DIR_VERSION_MAGIC);
    out[8..10].copy_from_slice(&version.to_le_bytes());
    out
}

// ─────────────────────────────────────────────────────────────────────
// Test (2)+(4): fresh dir stamps the current version; restart is a clean
// re-open (round-trip through the production bootstrap).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn fresh_durable_dir_is_stamped_and_restart_reopens_1302() {
    let tmp = TempDir::new().expect("tempdir");
    let data_dir = tmp.path().join("db");

    // ── Session 1: fresh durable bootstrap must stamp `<data_dir>/VERSION`.
    {
        let (backend, guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
            data_dir: data_dir.clone(),
        })
        .expect("durable bootstrap on a fresh dir must succeed");
        assert!(guard.is_durable());
        commit_one_node(&backend);
    }

    // The VERSION file is stamped at the current version with the right magic.
    let vpath = version_file_path(&data_dir);
    assert!(
        vpath.exists(),
        "fresh durable bootstrap must write <data_dir>/{VERSION_FILE}"
    );
    let bytes = std::fs::read(&vpath).expect("read VERSION");
    assert_eq!(&bytes[0..8], DATA_DIR_VERSION_MAGIC, "magic stamped");
    assert_eq!(
        u16::from_le_bytes([bytes[8], bytes[9]]),
        DATA_DIR_FORMAT_VERSION,
        "current version stamped"
    );

    // ── Session 2: restart over the SAME (now-stamped) dir is a clean re-open.
    let before = std::fs::read(&vpath).expect("read VERSION before restart");
    let (_backend2, _guard2) = bootstrap_storage_backend(&BootstrapMode::Durable {
        data_dir: data_dir.clone(),
    })
    .expect("restart over a same-version stamped dir must succeed (clean no-op)");
    let after = std::fs::read(&vpath).expect("read VERSION after restart");
    assert_eq!(
        before, after,
        "a same-version re-open must NOT rewrite the version stamp"
    );
}

// ─────────────────────────────────────────────────────────────────────
// Test (1): THE ORACLE — boot refuses an incompatible on-disk version.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn boot_refuses_incompatible_on_disk_version_1302() {
    let tmp = TempDir::new().expect("tempdir");
    let data_dir = tmp.path().join("db");

    // ── Session 1: stamp the dir at the current version + write real data.
    {
        let (backend, _guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
            data_dir: data_dir.clone(),
        })
        .expect("initial durable bootstrap");
        commit_one_node(&backend);
    }

    // TAMPER: rewrite `<data_dir>/VERSION` at an UNSUPPORTED version — models
    // an operator binary-swapping across a data-dir-format change.
    let vpath = version_file_path(&data_dir);
    let tampered = DATA_DIR_FORMAT_VERSION.wrapping_add(9);
    std::fs::write(&vpath, forge_version_body(tampered)).expect("tamper VERSION");

    // ── Session 2: re-open MUST fail LOUD (not misparse `pages.db`, not panic).
    //    (`DurabilityGuard` is not `Debug`, so match the Result directly
    //    rather than `.expect_err()` on the whole bootstrap tuple.)
    let err = match bootstrap_storage_backend(&BootstrapMode::Durable {
        data_dir: data_dir.clone(),
    }) {
        Ok(_) => panic!(
            "re-opening a data dir stamped at an INCOMPATIBLE version MUST be refused at \
             bootstrap (SVC-2 upgrade-safety, #1302) — NOT a silent misparse"
        ),
        Err(e) => e,
    };
    let msg = format!("{err:#}");
    assert!(
        msg.contains("incompatible on-disk data-dir version"),
        "error must name the incompatible-version condition; got: {msg}"
    );
    assert!(
        msg.contains(&tampered.to_string()),
        "error must report the found (tampered) version {tampered}; got: {msg}"
    );
    // #1345 R1 REQUIRED: must NOT send the operator to `arcgraph migrate`
    // (the Neo4j-import verb) — an incompatible dir needs a matching binary
    // or a restore, and can NOT be adopted.
    assert!(
        !msg.contains("arcgraph migrate"),
        "error must NOT point at `arcgraph migrate` (Neo4j-import verb); got: {msg}"
    );
    assert!(
        msg.contains("matching ArcGraph binary") || msg.contains("restore"),
        "error must point at a matching binary / restore; got: {msg}"
    );
    assert!(
        msg.contains("can NOT be adopted"),
        "error must say an incompatible version can NOT be adopted; got: {msg}"
    );
    assert!(
        msg.contains("1302"),
        "error should cite the root-cause issue #1302; got: {msg}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// Test (3): a LEGACY / pre-stamp dir (has data, no VERSION file), with NO
// explicit adopt, fails loud pointing at the REAL recovery path (the adopt
// flag) — NOT `arcgraph migrate`, NOT a silent proceed, NOT a silent
// auto-stamp. The "don't brick existing beta deployments silently" policy.
// ─────────────────────────────────────────────────────────────────────

/// Create a real durable dir, then delete its `VERSION` file to simulate a
/// pre-stamp (beta) dir that holds data but predates the on-disk version
/// guard. Returns the `<tmp>/db` path (kept alive by the returned `TempDir`)
/// and the committed `NodeId` (for post-adopt readback).
fn make_legacy_unstamped_dir() -> (TempDir, std::path::PathBuf, NodeId) {
    let tmp = TempDir::new().expect("tempdir");
    let data_dir = tmp.path().join("db");
    let node_id = {
        let (backend, _guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
            data_dir: data_dir.clone(),
        })
        .expect("initial durable bootstrap");
        commit_one_node(&backend)
    };
    let vpath = version_file_path(&data_dir);
    assert!(vpath.exists(), "precondition: dir was stamped");
    std::fs::remove_file(&vpath).expect("remove VERSION to simulate legacy dir");
    (tmp, data_dir, node_id)
}

#[test]
fn legacy_unstamped_data_dir_fails_loud_not_silent_1302() {
    let (_tmp, data_dir, _node_id) = make_legacy_unstamped_dir();
    let vpath = version_file_path(&data_dir);

    // ── Re-open with NO adopt flag MUST fail loud (legacy policy), and MUST
    //    NOT silently auto-stamp. (Match the Result directly — the bootstrap
    //    tuple is not `Debug`.)
    let err = match bootstrap_storage_backend(&BootstrapMode::Durable {
        data_dir: data_dir.clone(),
    }) {
        Ok(_) => panic!(
            "a data dir that holds data but has no VERSION stamp (a pre-stamp beta deployment) \
             MUST be refused — not silently proceeded (#1302)"
        ),
        Err(e) => e,
    };
    let msg = format!("{err:#}");
    assert!(
        msg.contains("predates the on-disk version guard"),
        "legacy error must explain the dir predates the stamp; got: {msg}"
    );
    // #1345 R1 REQUIRED: point at the REAL adopt path, NOT `arcgraph migrate`
    // (the Neo4j-import verb — a wrong/destructive dead-end for beta ops).
    assert!(
        !msg.contains("arcgraph migrate"),
        "legacy error must NOT point at `arcgraph migrate` (Neo4j-import verb); got: {msg}"
    );
    assert!(
        msg.contains("--adopt-legacy-datadir"),
        "legacy error must point at the real adopt path; got: {msg}"
    );
    // Critically: the guard must NOT have silently written a VERSION file.
    assert!(
        !vpath.exists(),
        "a legacy dir must NOT be silently auto-stamped (that would mark a possibly-incompatible \
         dir as compatible, defeating the guard)"
    );
}

// ─────────────────────────────────────────────────────────────────────
// #1345 R1 REQUIRED — the REAL recovery path: explicit adopt boots + stamps.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn legacy_unstamped_data_dir_adopts_with_explicit_flag_1302() {
    // A legacy unstamped dir + the explicit `--adopt-legacy-datadir` opt-in
    // MUST boot (the beta→GA recovery), stamp the current version, AND the
    // pre-existing data MUST still be readable (adopt does not touch pages/WAL).
    let (_tmp, data_dir, node_id) = make_legacy_unstamped_dir();
    let vpath = version_file_path(&data_dir);

    let (backend, guard) =
        bootstrap_durable_adopt(&data_dir).expect("explicit --adopt-legacy-datadir MUST boot");
    assert!(guard.is_durable(), "adopt must yield a durable substrate");

    // The dir is now stamped at the current version.
    assert!(vpath.exists(), "adopt must stamp the VERSION file");
    let bytes = std::fs::read(&vpath).expect("read VERSION");
    assert_eq!(
        &bytes[0..8],
        DATA_DIR_VERSION_MAGIC,
        "magic stamped on adopt"
    );
    assert_eq!(
        u16::from_le_bytes([bytes[8], bytes[9]]),
        DATA_DIR_FORMAT_VERSION,
        "adopt stamps the current version"
    );

    // The pre-existing committed node is still there (adopt recovered the WAL).
    let crud = crud_for(&backend, TenantId::DEFAULT);
    let tx = backend.txn_manager().begin(TenantId::DEFAULT);
    let node = read_node_with_store(&crud, &tx, node_id)
        .expect("read node after adopt")
        .expect("adopt must recover the pre-existing data (the committed node survives)");
    assert_eq!(node.label_id, 7, "recovered node label survives adopt");
    drop(guard);

    // A subsequent NORMAL boot (no adopt flag) is now a clean no-op — the
    // adopt is durable, so the operator does not need the flag again.
    let (_backend2, _guard2) = bootstrap_storage_backend(&BootstrapMode::Durable {
        data_dir: data_dir.clone(),
    })
    .expect("after adopt, a normal boot is a clean no-op");
}

#[test]
fn incompatible_version_is_never_adopted_even_with_flag_1302() {
    // #1345 R1 REQUIRED: adopt rescues ONLY an unstamped legacy dir. An
    // INCOMPATIBLE stamped version is refused even WITH --adopt-legacy-datadir
    // (the on-disk format really differs; stamping current would be a lie).
    let tmp = TempDir::new().expect("tempdir");
    let data_dir = tmp.path().join("db");
    {
        let (backend, _guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
            data_dir: data_dir.clone(),
        })
        .expect("initial durable bootstrap");
        commit_one_node(&backend);
    }
    // Tamper to an incompatible version.
    let vpath = version_file_path(&data_dir);
    let tampered = DATA_DIR_FORMAT_VERSION.wrapping_add(13);
    std::fs::write(&vpath, forge_version_body(tampered)).expect("tamper VERSION");

    // Even WITH the adopt flag, bootstrap MUST refuse.
    let err = match bootstrap_durable_adopt(&data_dir) {
        Ok(_) => panic!(
            "an incompatible stamped version MUST be refused even with --adopt-legacy-datadir \
             (adopt rescues an unstamped legacy dir, never a version mismatch) (#1302)"
        ),
        Err(e) => e,
    };
    let msg = format!("{err:#}");
    assert!(
        msg.contains("incompatible on-disk data-dir version"),
        "must be the incompatible-version error, not an adopt; got: {msg}"
    );
    // The stamp is untouched — adopt did not overwrite the incompatible version.
    let after = std::fs::read(&vpath).expect("read VERSION after refused adopt");
    assert_eq!(
        u16::from_le_bytes([after[8], after[9]]),
        tampered,
        "refused adopt must NOT overwrite the incompatible stamp"
    );
}

// ─────────────────────────────────────────────────────────────────────
// Negative pin: --in-memory has no data dir, so the guard is inert (no
// VERSION file is written, bootstrap succeeds unchanged).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn in_memory_bootstrap_has_no_version_stamp_1302() {
    let (_backend, guard) =
        bootstrap_storage_backend(&BootstrapMode::InMemory).expect("in-memory bootstrap");
    assert!(
        !guard.is_durable(),
        "in-memory mode owns no WAL writer and touches no data dir"
    );
    // Nothing to assert on disk — the version guard is a durable-only concern
    // (there is no `<data_dir>` in `--in-memory` mode). This pins that the
    // guard did not introduce a spurious file-write on the ephemeral path.
}
