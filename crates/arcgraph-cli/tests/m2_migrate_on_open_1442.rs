//! v2 M2 — migrate-on-open (JSON → typed blocks, `data_dir_version`
//! 3 → 4) + the §0.2 crash-during-migration gate, driven through the
//! PRODUCTION bootstrap seam (build-plan §2 M2 EXIT 4; design
//! `m1-m2-m4-m5-impl-designs.md` §0.1/§0.2/§M2.6; ADR-230 row M2;
//! tracking #1442).
//!
//! Covers:
//! 1. **Backward-compat / migration happy path** — a genuine M1-era
//!    (v3) store (JSON slotted bags + JSON chained large bags,
//!    `VERSION = 3`, MANIFEST `slotted-v1`) reopened by the M2 binary
//!    migrates on open: every bag reads back VALUE-identical through
//!    the production read path, every payload re-encoded typed (first
//!    byte 0x01), `VERSION` re-stamped `4`, MANIFEST final
//!    (`typed-v1`).
//! 2. **Idempotent reopen** — a second open is a clean no-op.
//! 3. **The §0.2 kill-9 gate (G4)** — a subprocess reopens the v3
//!    store with `ARCGRAPH_M2_MIGRATE_CRASH_AFTER_BATCHES=1` armed and
//!    ABORTS at the hardest window (first batch committed + superseded
//!    chains unreclaimed + MANIFEST still `typed-v1-migrating` +
//!    VERSION already 4). The parent asserts the crash-window stamps,
//!    then a NORMAL open RESUMES — reading the genuinely MIXED store
//!    (typed payloads from batch 1, JSON payloads from the remainder —
//!    the mixed-store read dispatch is what the resume sweep itself
//!    exercises to finish) — and completes; every value identical
//!    throughout. Old-value-intact-or-fully-migrated, never torn.
//! 4. **The 1 → 3 → 4 ladder** — a genuine PRE-M1 (v1, chained,
//!    no-MANIFEST) store opened by the M2 binary chains BOTH sweeps in
//!    one boot and lands at `VERSION = 4` / `typed-v1`,
//!    value-identical.
//!
//! Value-identity oracle: payloads are deterministic per node id (the
//! M1 gate's derivation trick), so every process can independently
//! derive the expected DECODED VALUES — the M2 oracle is
//! value-identity through the production mcp read path (the
//! representation changes by design; the VISIBLE values must not,
//! design §0.3 consistency-neutral).

use std::path::Path;
use std::sync::Arc;

use arcgraph_cli::bootstrap::{BootstrapMode, bootstrap_storage_backend};
use arcgraph_core::{LabelId, NodeId, PartitionId, TenantId};
use arcgraph_storage::crud::{CrudStore, PropertyData, commit, create_node, read_node_with_store};
use arcgraph_storage::prop_block::PROP_BLOCK_DISCRIMINANT;
use arcgraph_storage::property::BlobRef;
use arcgraph_storage::records::PROP_BAG_MAX_BYTES;
use arcgraph_storage::{
    DATA_DIR_FORMAT_VERSION, DATA_DIR_VERSION_CHAINED_V1, DATA_DIR_VERSION_SLOTTED_M1,
    manifest::{
        DataDirManifest, PROPS_FORMAT_SLOTTED_V1, PROPS_FORMAT_TYPED_V1,
        PROPS_FORMAT_TYPED_V1_MIGRATING, now_rfc3339_utc,
    },
    read_data_dir_manifest, stamp_data_dir, write_data_dir_manifest,
};
use tempfile::TempDir;

/// Env var handing subprocesses the data-dir path.
const ENV_TEST_DIR: &str = "ARCGRAPH_TEST_M2_MIGRATE_DIR";

/// More than one M2 migration batch (512) so the crash-after-batch-1
/// leg leaves a genuine PARTIAL (mixed typed + JSON) store.
const FIXTURE_SMALL: u32 = 560;
/// Large (M1 kept-chained) JSON bags mixed in — the M2 sweep migrates
/// them too (block + overflow).
const FIXTURE_LARGE: u32 = 6;

fn crud_for(backend: &arcgraph_mcp::storage::StorageBackend, tenant: TenantId) -> Arc<CrudStore> {
    backend
        .router()
        .route(tenant, PartitionId::ZERO)
        .expect("route tenant")
        .crud()
        .clone()
}

/// Deterministic JSON bag per node id (dense from 1, creation order).
fn small_payload(i: u32) -> Vec<u8> {
    format!(
        r#"{{"incident":"inc-{i:05}","open":{},"score":{}.5,"sev":"P{}"}}"#,
        i % 2 == 0,
        i % 90,
        i % 4
    )
    .into_bytes()
}

/// A large JSON bag (> PROP_BAG_MAX_BYTES) — M1-chained; the M2 sweep
/// re-encodes it to a typed block whose big value spills to overflow.
fn large_payload(i: u32) -> Vec<u8> {
    format!(
        r#"{{"incident":"inc-{i:05}","dump":"{}"}}"#,
        "x".repeat(PROP_BAG_MAX_BYTES + 512 + (i as usize % 3) * 1024)
    )
    .into_bytes()
}

/// The expected DECODED values for node `i`, derived independently
/// (the cross-process value-identity oracle).
fn expected_values(i: u32, large: bool) -> std::collections::BTreeMap<String, serde_json::Value> {
    let bytes = if large {
        large_payload(i)
    } else {
        small_payload(i)
    };
    match serde_json::from_slice::<serde_json::Value>(&bytes).expect("fixture JSON parses") {
        serde_json::Value::Object(m) => m.into_iter().collect(),
        other => panic!("fixture must be an object, got {other:?}"),
    }
}

fn run_helper(helper: &str, dir: &Path, envs: &[(&str, &str)]) -> std::process::Output {
    let exe = std::env::current_exe().expect("test binary path");
    let mut cmd = std::process::Command::new(exe);
    cmd.args(["--exact", helper, "--ignored", "--nocapture"])
        .env(ENV_TEST_DIR, dir);
    for (k, v) in envs {
        cmd.env(k, v);
    }
    cmd.output().expect("spawn helper subprocess")
}

/// Build a genuine M1-era (v3) store: JSON payloads written at the
/// crud grain (`PropertyData::Blob` — the M1 write shape, still fully
/// supported machinery), clean shutdown, then the stamps rewritten to
/// the M1 shape (`VERSION = 3`, MANIFEST `slotted-v1`). Byte-genuine:
/// exactly what an M1 binary's store looks like.
fn build_v3_fixture(dir: &Path) {
    let out = run_helper("helper_build_v3_fixture", dir, &[]);
    assert!(
        out.status.success(),
        "v3 fixture builder must succeed.\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    stamp_data_dir(dir, DATA_DIR_VERSION_SLOTTED_M1).expect("re-stamp VERSION=3");
    let mut m = DataDirManifest::m1_slotted(now_rfc3339_utc());
    m.props_store_format = PROPS_FORMAT_SLOTTED_V1.to_string();
    write_data_dir_manifest(dir, &m).expect("write M1 manifest");
}

/// Scan every fixture record through the production read path;
/// assert value-identity vs the derived oracle and count payload
/// representations. Returns `(typed, json)` payload counts.
fn assert_all_values(
    backend: &arcgraph_mcp::storage::StorageBackend,
    dir_label: &str,
) -> (u32, u32) {
    let tenant = TenantId::DEFAULT;
    let crud = crud_for(backend, tenant);
    let tx = backend.txn_manager().begin(tenant);
    let intern = backend.intern_table();
    let (mut typed, mut json) = (0u32, 0u32);
    for i in 0..(FIXTURE_SMALL + FIXTURE_LARGE) {
        let id = NodeId::new(u64::from(i) + 1);
        let rec = read_node_with_store(&crud, &tx, id)
            .expect("read")
            .unwrap_or_else(|| panic!("node {id:?} must exist ({dir_label})"));
        // Representation census (the payload's first byte).
        let bref = BlobRef::decode(rec.property_ref)
            .unwrap_or_else(|| panic!("node {id:?} must carry a blob ref ({dir_label})"));
        let raw = crud
            .blob_store()
            .get(tenant, bref)
            .expect("payload readable");
        match raw.first() {
            Some(&PROP_BLOCK_DISCRIMINANT) => typed += 1,
            Some(&b'{') => json += 1,
            other => panic!("unknown payload discriminant {other:?} ({dir_label})"),
        }
        // Value-identity through the PRODUCTION mcp read path.
        let bag = arcgraph_mcp::storage::property_payload::record_property_bag_checked(
            &rec,
            crud.blob_store(),
            intern,
            tenant,
        )
        .expect("checked read");
        let is_large = i >= FIXTURE_SMALL;
        let expect = expected_values(i, is_large);
        assert_eq!(
            bag.len(),
            expect.len(),
            "node {id:?} bag size ({dir_label})"
        );
        for (k, jv) in &expect {
            let got = bag
                .get(k)
                .unwrap_or_else(|| panic!("node {id:?} missing key {k} ({dir_label})"));
            assert_eq!(
                &got.to_json_value(),
                jv,
                "node {id:?} key {k} value ({dir_label})"
            );
        }
    }
    (typed, json)
}

fn open_durable(dir: &Path) -> (arcgraph_mcp::storage::StorageBackend, impl Drop) {
    let (backend, guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
        data_dir: dir.to_path_buf(),
    })
    .expect("durable bootstrap");
    (backend, guard)
}

// ─────────────────────────────────────────────────────────────────────
// Helpers (subprocess entry points; `#[ignore]`d so only the parent's
// `run_helper` invokes them)
// ─────────────────────────────────────────────────────────────────────

#[test]
#[ignore = "subprocess helper — invoked by the parent tests via run_helper"]
fn helper_build_v3_fixture() {
    let dir = std::env::var(ENV_TEST_DIR).expect("dir env");
    let (backend, guard) = open_durable(Path::new(&dir));
    let tenant = TenantId::DEFAULT;
    let crud = crud_for(&backend, tenant);
    let mgr = backend.txn_manager();
    // JSON bags at the crud grain — the M1 write shape.
    for i in 0..FIXTURE_SMALL {
        let mut tx = mgr.begin(tenant);
        create_node(
            &crud,
            &mut tx,
            tenant,
            LabelId::new(1),
            &PropertyData::Blob(small_payload(i)),
        )
        .expect("create small");
        commit(tx, &crud).expect("commit small");
    }
    for i in FIXTURE_SMALL..(FIXTURE_SMALL + FIXTURE_LARGE) {
        let mut tx = mgr.begin(tenant);
        create_node(
            &crud,
            &mut tx,
            tenant,
            LabelId::new(1),
            &PropertyData::Blob(large_payload(i)),
        )
        .expect("create large");
        commit(tx, &crud).expect("commit large");
    }
    drop(guard);
}

#[test]
#[ignore = "subprocess helper — invoked by the crash-gate test via run_helper"]
fn helper_crash_mid_m2_migration() {
    // Reopens the v3 store with the M2 crash hook armed (the env is
    // set by the PARENT); bootstrap runs the sweep, which aborts the
    // process after batch 1. This helper "fails" by design (abort).
    let dir = std::env::var(ENV_TEST_DIR).expect("dir env");
    let (_backend, _guard) = open_durable(Path::new(&dir));
    // Reaching here means the crash hook did NOT fire — the parent
    // asserts on the subprocess's abort status, so make the miss loud.
    panic!("crash hook did not fire — ARCGRAPH_M2_MIGRATE_CRASH_AFTER_BATCHES not honored");
}

// ─────────────────────────────────────────────────────────────────────
// The gates
// ─────────────────────────────────────────────────────────────────────

/// Happy path + idempotent reopen (backward-compat EXIT bullet: v3
/// fixture → migrate-on-open → value-identical; pre-M2 store fully
/// re-encoded; stamps final).
#[test]
fn m2_migrate_on_open_v3_store_value_identical_and_idempotent() {
    let tmp = TempDir::new().expect("tempdir");
    build_v3_fixture(tmp.path());

    // First open: the M2 sweep runs to completion.
    {
        let (backend, guard) = open_durable(tmp.path());
        let (typed, json) = assert_all_values(&backend, "post-migration");
        assert_eq!(
            typed,
            FIXTURE_SMALL + FIXTURE_LARGE,
            "every payload re-encoded typed"
        );
        assert_eq!(json, 0, "no JSON payload survives the completed sweep");
        drop(guard);
    }
    // Stamps: VERSION = 4, MANIFEST final typed-v1.
    let v = std::fs::read(arcgraph_storage::version_file_path(tmp.path())).expect("VERSION");
    assert_eq!(
        u16::from_le_bytes([v[8], v[9]]),
        DATA_DIR_FORMAT_VERSION,
        "VERSION re-stamped 4"
    );
    let m = read_data_dir_manifest(tmp.path())
        .expect("manifest read")
        .expect("manifest present");
    assert_eq!(m.props_store_format, PROPS_FORMAT_TYPED_V1);
    assert_eq!(m.data_dir_version, DATA_DIR_FORMAT_VERSION);

    // Second open: clean no-op, values still identical.
    {
        let (backend, guard) = open_durable(tmp.path());
        let (typed, json) = assert_all_values(&backend, "idempotent-reopen");
        assert_eq!(typed, FIXTURE_SMALL + FIXTURE_LARGE);
        assert_eq!(json, 0);
        drop(guard);
    }
}

/// G4 — the §0.2 kill-9 crash-during-migration gate at the hardest
/// window (batch committed, superseded chains unreclaimed, MANIFEST
/// migrating), then resume-to-completion. RED-on-revert: breaking the
/// sweep's resume (e.g. skipping the `already_typed` idempotence or
/// flipping the MANIFEST before the sweep) makes the value-identity or
/// stamp assertions fail.
#[test]
fn m2_crash_mid_migration_resumes_value_identical() {
    let tmp = TempDir::new().expect("tempdir");
    build_v3_fixture(tmp.path());

    // Kill-9 (process abort) after the FIRST migration batch commits.
    let out = run_helper(
        "helper_crash_mid_m2_migration",
        tmp.path(),
        &[("ARCGRAPH_M2_MIGRATE_CRASH_AFTER_BATCHES", "1")],
    );
    assert!(
        !out.status.success(),
        "the crash helper must abort (crash hook armed).\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    // The crash window's stamps: VERSION already 4 (pre-M2 binaries
    // locked out before the first typed byte), MANIFEST still the
    // resumable migrating marker — the §0.2 resume contract.
    let v = std::fs::read(arcgraph_storage::version_file_path(tmp.path())).expect("VERSION");
    assert_eq!(
        u16::from_le_bytes([v[8], v[9]]),
        DATA_DIR_FORMAT_VERSION,
        "crash window: VERSION already 4"
    );
    let m = read_data_dir_manifest(tmp.path())
        .expect("manifest read")
        .expect("manifest present");
    assert_eq!(
        m.props_store_format, PROPS_FORMAT_TYPED_V1_MIGRATING,
        "crash window: MANIFEST still migrating (resumable)"
    );

    // Resume: a NORMAL open completes the sweep over the MIXED store
    // (batch-1 payloads typed, the rest still JSON — the resume sweep
    // itself reads both classes) and every value is identical.
    {
        let (backend, guard) = open_durable(tmp.path());
        let (typed, json) = assert_all_values(&backend, "post-resume");
        assert_eq!(typed, FIXTURE_SMALL + FIXTURE_LARGE);
        assert_eq!(json, 0);
        drop(guard);
    }
    let m = read_data_dir_manifest(tmp.path())
        .expect("manifest read")
        .expect("manifest present");
    assert_eq!(
        m.props_store_format, PROPS_FORMAT_TYPED_V1,
        "final MANIFEST"
    );
}

/// The full ladder: a genuine PRE-M1 (v1, chained, no MANIFEST) store
/// chains BOTH migrate-on-open sweeps (1 → 3 → 4) in one boot.
#[test]
fn m2_v1_store_chains_both_sweeps_to_v4() {
    let tmp = TempDir::new().expect("tempdir");
    // The M1 gate's fixture recipe: forced-chained JSON writes, then
    // stamps rewritten to the pre-M1 shape.
    let out = run_helper(
        "helper_build_v3_fixture",
        tmp.path(),
        &[("ARCGRAPH_M1_FORCE_CHAINED_BAGS", "1")],
    );
    assert!(
        out.status.success(),
        "chained fixture builder must succeed.\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    stamp_data_dir(tmp.path(), DATA_DIR_VERSION_CHAINED_V1).expect("re-stamp VERSION=1");
    let manifest = tmp.path().join("MANIFEST");
    if manifest.exists() {
        std::fs::remove_file(&manifest).expect("remove MANIFEST");
    }

    let (backend, guard) = open_durable(tmp.path());
    let (typed, json) = assert_all_values(&backend, "v1-ladder");
    assert_eq!(typed, FIXTURE_SMALL + FIXTURE_LARGE);
    assert_eq!(json, 0);
    drop(guard);

    let v = std::fs::read(arcgraph_storage::version_file_path(tmp.path())).expect("VERSION");
    assert_eq!(
        u16::from_le_bytes([v[8], v[9]]),
        DATA_DIR_FORMAT_VERSION,
        "ladder end: VERSION 4"
    );
    let m = read_data_dir_manifest(tmp.path())
        .expect("manifest read")
        .expect("manifest present");
    assert_eq!(m.props_store_format, PROPS_FORMAT_TYPED_V1);
}
