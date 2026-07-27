//! v2 M1 — migrate-on-open + the crash-during-migration gate, driven
//! through the PRODUCTION bootstrap seam (build-plan §2 M1 EXIT 4;
//! design `m1-m2-m4-m5-impl-designs.md` §0.1/§0.2/§M1.4; ADR-230;
//! tracking #1430).
//!
//! Covers:
//! 1. **Backward-compat / migration happy path** — a genuine pre-M1
//!    store (DEC-4 chained bags, `VERSION = 1`, no MANIFEST) reopened
//!    by the M1 binary migrates on open: every bag reads back
//!    byte-identical, small bags re-encoded slotted (`slot_id >= 1`),
//!    large bags kept chained (`slot_id == 0`), `VERSION` re-stamped
//!    `3`, MANIFEST final (`slotted-v1`).
//! 2. **Idempotent reopen** — a second open of the migrated store is a
//!    clean no-op (bags unchanged, stamps unchanged).
//! 3. **The §0.2 kill-9 gate** — a subprocess reopens the pre-M1 store
//!    with the crash-injection hook armed and ABORTS at the hardest
//!    window (first batch committed + chains not yet reclaimed +
//!    MANIFEST still `slotted-v1-migrating`). The next NORMAL open
//!    RESUMES and completes; every bag byte-identical throughout —
//!    old-version-intact-or-fully-migrated, never torn (the §0.2
//!    atomic-or-RESUMABLE contract; both representations readable at
//!    every intermediate state).
//!
//! Fixture construction: a pre-M1 chained store is unreachable through
//! the M1 write path (small bags pack slotted), so the fixture builder
//! runs in a SUBPROCESS under `ARCGRAPH_M1_FORCE_CHAINED_BAGS=1` (the
//! test-only lever in `blob.rs`) and the parent then rewrites the
//! stamps to the pre-M1 shape (`VERSION = 1`, MANIFEST removed).

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use arcgraph_cli::bootstrap::{BootstrapMode, bootstrap_storage_backend};
use arcgraph_core::{LabelId, NodeId, PartitionId, TenantId};
use arcgraph_storage::crud::{CrudStore, PropertyData, commit, create_node, read_node_with_store};
use arcgraph_storage::property::BlobRef;
use arcgraph_storage::records::PROP_BAG_MAX_BYTES;
use arcgraph_storage::{
    DATA_DIR_VERSION_CHAINED_V1, read_data_dir_manifest, stamp_data_dir, version_file_path,
};
use tempfile::TempDir;

/// Env var the parent uses to hand the fixture/crash subprocesses the
/// data-dir path.
const ENV_TEST_DIR: &str = "ARCGRAPH_TEST_M1_MIGRATE_DIR";

/// Total nodes in the fixture — > one migration batch (512) so the
/// crash-after-batch-1 subprocess leaves a genuine PARTIAL (mixed
/// slotted + chained) store for the resume leg.
const FIXTURE_SMALL: u32 = 560;
/// Large (chained-forever) bags mixed in.
const FIXTURE_LARGE: u32 = 8;

fn crud_for(backend: &arcgraph_mcp::storage::StorageBackend, tenant: TenantId) -> Arc<CrudStore> {
    backend
        .router()
        .route(tenant, PartitionId::ZERO)
        .expect("route tenant")
        .crud()
        .clone()
}

/// Deterministic per-node payloads. Node ids are allocated densely from
/// 1 in creation order, so readback derives the expected bytes from the
/// id alone (no side-channel needed across processes).
fn small_payload(i: u32) -> Vec<u8> {
    format!(
        r#"{{"incident":"inc-{i:05}","sev":"P{}","svc":"api-{}"}}"#,
        i % 4,
        i % 17
    )
    .into_bytes()
}
fn large_payload(i: u32) -> Vec<u8> {
    vec![(i & 0xFF) as u8; PROP_BAG_MAX_BYTES + 1 + (i as usize % 3) * 4096]
}

/// Spawn this test binary re-targeted at `helper`, with `envs` set.
fn run_helper(helper: &str, dir: &Path, envs: &[(&str, &str)]) -> std::process::Output {
    let exe = std::env::current_exe().expect("test binary path");
    let mut cmd = Command::new(exe);
    cmd.args(["--exact", helper, "--ignored", "--nocapture"])
        .env(ENV_TEST_DIR, dir);
    for (k, v) in envs {
        cmd.env(k, v);
    }
    cmd.output().expect("spawn helper subprocess")
}

/// Build the pre-M1 fixture: subprocess writes a durable store with
/// FORCED-CHAINED bags + clean shutdown; parent rewrites the stamps to
/// the genuine pre-M1 shape (`VERSION = 1`, no MANIFEST).
fn build_pre_m1_fixture(dir: &Path) {
    let out = run_helper(
        "helper_build_chained_fixture",
        dir,
        &[("ARCGRAPH_M1_FORCE_CHAINED_BAGS", "1")],
    );
    assert!(
        out.status.success(),
        "fixture builder must succeed.\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    // Rewrite the stamps to the pre-M1 shape: chained-format VERSION,
    // no MANIFEST — byte-for-byte what a store written by a pre-M1
    // binary looks like (pages.db + wal/ + VERSION=1).
    stamp_data_dir(dir, DATA_DIR_VERSION_CHAINED_V1).expect("re-stamp VERSION=1");
    let manifest = dir.join("MANIFEST");
    if manifest.exists() {
        std::fs::remove_file(&manifest).expect("remove MANIFEST");
    }
}

/// Read every fixture bag through the production read path and assert
/// the expected representation + content.
///
/// # v2 M2 update (the ladder end-state — flagged for review)
///
/// Under the M2 binary a v1 store chains BOTH migrate-on-open sweeps
/// (1 → 3 → 4) in one boot, so this gate's post-migration oracle moved
/// with the ladder:
///
/// - **Small (JSON) bags** end TYPED (first byte 0x01) in slotted
///   slots, VALUE-identical through the production mcp read (the
///   representation changes by design; visible values must not —
///   design §0.3). Byte-equality of the JSON form now holds only for
///   the pre-migration fixture-verification pass.
/// - **Large RAW (non-JSON) bags** — this fixture's large payloads are
///   arbitrary bytes, the embedder-written crud-grain class — stay
///   BYTE-IDENTICAL and CHAINED forever: the M2 sweep's
///   `skipped_opaque` preservation disposition (see
///   `M2MigrateReport::skipped_opaque`). This gate is now ALSO the
///   opaque-preservation gate.
fn assert_all_bags(
    backend: &arcgraph_mcp::storage::StorageBackend,
    expect_small_migrated: bool,
) -> (u32, u32) {
    let crud = crud_for(backend, TenantId::DEFAULT);
    let tx = backend.txn_manager().begin(TenantId::DEFAULT);
    let intern = backend.intern_table();
    let (mut small_ok, mut chained_raw) = (0u32, 0u32);
    for i in 0..(FIXTURE_SMALL + FIXTURE_LARGE) {
        let id = NodeId::new(u64::from(i) + 1); // dense from 1, creation order
        let rec = read_node_with_store(&crud, &tx, id)
            .expect("read")
            .unwrap_or_else(|| panic!("node {id:?} must exist"));
        let bref = BlobRef::decode(rec.property_ref)
            .unwrap_or_else(|| panic!("node {id:?} must carry an overflow ref"));
        let got = crud
            .blob_store()
            .get(TenantId::DEFAULT, bref)
            .unwrap_or_else(|e| panic!("bag read for node {id:?}: {e}"));
        if i >= FIXTURE_SMALL {
            // Large RAW bag: byte-identical + chained, at EVERY stage
            // (the opaque-preservation disposition).
            assert_eq!(
                got.as_ref(),
                large_payload(i).as_slice(),
                "raw bag bytes must be identical for node {id:?} (opaque preservation)"
            );
            assert_eq!(bref.slot_id, 0, "raw bag must stay chained (node {id:?})");
            chained_raw += 1;
        } else if expect_small_migrated {
            // Post-boot (M1+M2 legs ran): typed slotted, value-identical.
            assert!(
                bref.slot_id >= 1,
                "small bag must be slotted post-migration (node {id:?})"
            );
            assert_eq!(
                got.first(),
                Some(&arcgraph_storage::prop_block::PROP_BLOCK_DISCRIMINANT),
                "small bag must be TYPED post-M2 (node {id:?})"
            );
            let bag = arcgraph_mcp::storage::property_payload::record_property_bag_checked(
                &rec,
                crud.blob_store(),
                intern,
                TenantId::DEFAULT,
            )
            .unwrap_or_else(|e| panic!("checked read for node {id:?}: {e}"));
            let expect: serde_json::Value =
                serde_json::from_slice(&small_payload(i)).expect("fixture JSON");
            let obj = expect.as_object().expect("fixture is an object");
            assert_eq!(bag.len(), obj.len(), "bag size for node {id:?}");
            for (k, jv) in obj {
                let got_v = bag
                    .get(k)
                    .unwrap_or_else(|| panic!("node {id:?} missing key {k}"));
                assert_eq!(
                    &got_v.to_json_value(),
                    jv,
                    "node {id:?} key {k} value-identity (the §0.3 anchor)"
                );
            }
            small_ok += 1;
        } else {
            // Pre-migration fixture verification: byte-identical JSON,
            // CHAINED (the forced-chained fixture precondition).
            assert_eq!(
                got.as_ref(),
                small_payload(i).as_slice(),
                "bag bytes must be identical for node {id:?} (fixture pass)"
            );
            assert_eq!(
                bref.slot_id, 0,
                "fixture precondition: bag chained (node {id:?})"
            );
            small_ok += 1;
        }
    }
    (small_ok, chained_raw)
}

fn version_of(dir: &Path) -> u16 {
    let bytes = std::fs::read(version_file_path(dir)).expect("read VERSION");
    u16::from_le_bytes([bytes[8], bytes[9]])
}

// ─── Subprocess helpers (ignored in normal runs) ─────────────────────

/// Fixture builder — runs under `ARCGRAPH_M1_FORCE_CHAINED_BAGS=1`.
#[test]
#[ignore = "subprocess helper (fixture builder) — spawned by the migration tests"]
fn helper_build_chained_fixture() {
    let dir = PathBuf::from(std::env::var(ENV_TEST_DIR).expect("fixture dir env"));
    assert_eq!(
        std::env::var("ARCGRAPH_M1_FORCE_CHAINED_BAGS").as_deref(),
        Ok("1"),
        "fixture builder must run under the forced-chained lever"
    );
    let (backend, guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
        data_dir: dir.clone(),
    })
    .expect("fixture bootstrap");
    assert!(guard.is_durable());
    let crud = crud_for(&backend, TenantId::DEFAULT);
    // Batched creates (realistic ingest shape). All bags CHAIN under
    // the lever — verified below before the fixture is accepted.
    let mut created = 0u32;
    while created < FIXTURE_SMALL {
        let n = (FIXTURE_SMALL - created).min(80);
        let mut tx = backend.txn_manager().begin(TenantId::DEFAULT);
        for _ in 0..n {
            create_node(
                &crud,
                &mut tx,
                TenantId::DEFAULT,
                LabelId::new(5),
                &PropertyData::Blob(small_payload(created)),
            )
            .expect("create small");
            created += 1;
        }
        commit(tx, &crud).expect("commit batch");
    }
    for i in FIXTURE_SMALL..(FIXTURE_SMALL + FIXTURE_LARGE) {
        let mut tx = backend.txn_manager().begin(TenantId::DEFAULT);
        create_node(
            &crud,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(5),
            &PropertyData::Blob(large_payload(i)),
        )
        .expect("create large");
        commit(tx, &crud).expect("commit large");
    }
    // Precondition proof: EVERY bag is chained (slot_id == 0 —
    // asserted per-bag inside assert_all_bags' pre-migration branch).
    let (small_ok, chained_raw) = assert_all_bags(&backend, false);
    assert_eq!(
        small_ok, FIXTURE_SMALL,
        "fixture precondition: all small bags verified"
    );
    assert_eq!(chained_raw, FIXTURE_LARGE);
    // guard drops here → graceful shutdown (+ shutdown checkpoint).
}

/// Crash-injected reopen — runs with the migration kill hook armed and
/// MUST die via `std::process::abort()` mid-sweep.
#[test]
#[ignore = "subprocess helper (crash-injected migration) — spawned by the kill-9 gate"]
fn helper_crash_migrating_reopen() {
    let dir = PathBuf::from(std::env::var(ENV_TEST_DIR).expect("fixture dir env"));
    assert!(
        std::env::var("ARCGRAPH_M1_MIGRATE_CRASH_AFTER_BATCHES").is_ok(),
        "crash helper must run with the injection hook armed"
    );
    // This bootstrap must NOT return: the sweep aborts the process
    // after its first batch commit.
    let _ = bootstrap_storage_backend(&BootstrapMode::Durable { data_dir: dir });
    panic!("the crash-injected migration must have aborted the process before this line");
}

// ─── The gates ────────────────────────────────────────────────────────

/// Happy path + idempotent reopen (EXIT gates: backward-compat, the
/// migration itself, and the §0.3 byte-equality anchor).
#[test]
fn m1_migrate_on_open_converts_pre_m1_store_byte_identically() {
    let tmp = TempDir::new().expect("tempdir");
    let dir = tmp.path().join("db");
    std::fs::create_dir_all(&dir).unwrap();
    build_pre_m1_fixture(&dir);
    assert_eq!(version_of(&dir), 1, "fixture is a genuine pre-M1 store");

    // ── Open 1: the M1 binary migrates on open.
    {
        let (backend, _guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
            data_dir: dir.clone(),
        })
        .expect("migrating open must succeed");
        let (slotted, chained) = assert_all_bags(&backend, true);
        assert_eq!(slotted, FIXTURE_SMALL, "every small bag re-encoded slotted");
        assert_eq!(chained, FIXTURE_LARGE, "large bags kept chained");
    }
    // v2 M2 ladder end-state: the M2 binary chains BOTH sweeps in one
    // boot — a v1 store lands at VERSION 4 / typed-v1 (the mid-ladder
    // VERSION=3 state is asserted by the crash-window gate below).
    assert_eq!(version_of(&dir), 4, "VERSION re-stamped 1 → 3 → 4");
    let manifest = read_data_dir_manifest(&dir)
        .expect("manifest readable")
        .expect("manifest present");
    assert!(
        !manifest.m1_migration_in_flight(),
        "MANIFEST must be final after the sweeps"
    );
    assert_eq!(manifest.props_store_format, "typed-v1");

    // ── Open 2: idempotent no-op reopen.
    {
        let (backend, _guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
            data_dir: dir.clone(),
        })
        .expect("migrated store reopens cleanly");
        let (slotted, chained) = assert_all_bags(&backend, true);
        assert_eq!((slotted, chained), (FIXTURE_SMALL, FIXTURE_LARGE));
    }
    assert_eq!(version_of(&dir), 4);
}

/// The §0.2 crash-during-migration gate: kill the process at the
/// hardest window (batch 1 committed, chains unreclaimed, MANIFEST
/// still `migrating`), then prove the next open RESUMES to completion
/// with every bag byte-identical. Old-data-intact-or-fully-migrated —
/// never torn, at every observed state.
#[test]
fn m1_crash_during_migration_resumes_and_completes_never_torn() {
    let tmp = TempDir::new().expect("tempdir");
    let dir = tmp.path().join("db");
    std::fs::create_dir_all(&dir).unwrap();
    build_pre_m1_fixture(&dir);

    // ── The kill-9-equivalent: abort after the FIRST migration batch.
    let out = run_helper(
        "helper_crash_migrating_reopen",
        &dir,
        &[("ARCGRAPH_M1_MIGRATE_CRASH_AFTER_BATCHES", "1")],
    );
    assert!(
        !out.status.success(),
        "the crash-injected reopen must have died mid-migration.\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    // The crashed store is mid-migration: VERSION already 3 (pre-M1
    // binaries locked out the moment slotted bytes could land) and the
    // MANIFEST still carries the RESUMABLE marker.
    assert_eq!(version_of(&dir), 3, "VERSION flips before the sweep");
    let mid = read_data_dir_manifest(&dir)
        .expect("manifest readable after crash")
        .expect("manifest present after crash");
    assert!(
        mid.m1_migration_in_flight(),
        "crash window must leave the migrating marker (the resume signal)"
    );

    // ── The resume: a normal open completes the M1 sweep AND (v2 M2)
    //    the M2 sweep that follows it — the ladder end-state.
    {
        let (backend, _guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
            data_dir: dir.clone(),
        })
        .expect("post-crash reopen must resume the migration");
        let (slotted, chained) = assert_all_bags(&backend, true);
        assert_eq!(
            (slotted, chained),
            (FIXTURE_SMALL, FIXTURE_LARGE),
            "resume must complete the re-encode with zero lost/torn bags"
        );
    }
    assert_eq!(version_of(&dir), 4, "ladder end-state after resume");
    let done = read_data_dir_manifest(&dir).unwrap().unwrap();
    assert!(
        !done.m1_migration_in_flight(),
        "MANIFEST final after resume"
    );
    assert_eq!(done.props_store_format, "typed-v1");
}
