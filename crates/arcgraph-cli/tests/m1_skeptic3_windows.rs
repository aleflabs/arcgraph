//! SKEPTIC-3 scratch gates (NOT FOR COMMIT) — migrate-on-open crash
//! windows the shipped gate does not exercise:
//!   w1: crash between migrating-MANIFEST write and VERSION re-stamp
//!   w2: crash after VERSION=3 re-stamp, before the first batch
//!   w3: crash after the LAST batch commit, before the final MANIFEST
//!   w4: crash MID-VERSION-STAMP (torn/zero-length VERSION file)
//!   w5: downgrade — full bootstrap over an unsupported stamped version
//!
//! Harness copied from tests/m1_migrate_on_open_1430.rs.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use arcgraph_cli::bootstrap::{BootstrapMode, bootstrap_storage_backend};
use arcgraph_core::{LabelId, NodeId, PartitionId, TenantId};
use arcgraph_storage::crud::{CrudStore, PropertyData, commit, create_node, read_node_with_store};
use arcgraph_storage::property::BlobRef;
use arcgraph_storage::records::PROP_BAG_MAX_BYTES;
use arcgraph_storage::{
    DATA_DIR_VERSION_CHAINED_V1, DataDirManifest, read_data_dir_manifest, stamp_data_dir,
    version_file_path, write_data_dir_manifest,
};
use tempfile::TempDir;

const ENV_TEST_DIR: &str = "ARCGRAPH_TEST_M1_S3_DIR";
const FIXTURE_SMALL: u32 = 560;
const FIXTURE_LARGE: u32 = 8;

fn crud_for(backend: &arcgraph_mcp::storage::StorageBackend, tenant: TenantId) -> Arc<CrudStore> {
    backend
        .router()
        .route(tenant, PartitionId::ZERO)
        .expect("route tenant")
        .crud()
        .clone()
}

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

fn build_pre_m1_fixture(dir: &Path) {
    let out = run_helper(
        "helper_s3_build_chained_fixture",
        dir,
        &[("ARCGRAPH_M1_FORCE_CHAINED_BAGS", "1")],
    );
    assert!(
        out.status.success(),
        "fixture builder must succeed.\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    stamp_data_dir(dir, DATA_DIR_VERSION_CHAINED_V1).expect("re-stamp VERSION=1");
    let manifest = dir.join("MANIFEST");
    if manifest.exists() {
        std::fs::remove_file(&manifest).expect("remove MANIFEST");
    }
}

/// v2 M2 oracle update (the ladder end-state — see
/// `m1_migrate_on_open_1430.rs`'s twin for the full rationale): small
/// JSON bags end TYPED slotted + VALUE-identical after a full boot
/// (the M2 sweep follows the M1 sweep); large RAW bags stay
/// byte-identical + chained forever (the M2 `skipped_opaque`
/// preservation disposition).
fn assert_all_bags(
    backend: &arcgraph_mcp::storage::StorageBackend,
    expect_small_migrated: bool,
) -> (u32, u32) {
    let crud = crud_for(backend, TenantId::DEFAULT);
    let tx = backend.txn_manager().begin(TenantId::DEFAULT);
    let intern = backend.intern_table();
    let (mut small_ok, mut chained_raw) = (0u32, 0u32);
    for i in 0..(FIXTURE_SMALL + FIXTURE_LARGE) {
        let id = NodeId::new(u64::from(i) + 1);
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
            assert_eq!(
                got.as_ref(),
                large_payload(i).as_slice(),
                "raw bag bytes must be identical for node {id:?} (opaque preservation)"
            );
            assert_eq!(bref.slot_id, 0, "raw bag must stay chained (node {id:?})");
            chained_raw += 1;
        } else if expect_small_migrated {
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
                assert_eq!(&got_v.to_json_value(), jv, "node {id:?} key {k} value");
            }
            small_ok += 1;
        } else {
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

fn migrating_manifest() -> DataDirManifest {
    DataDirManifest::m1_migrating(arcgraph_storage::manifest::now_rfc3339_utc())
}

// ─── Subprocess helpers ──────────────────────────────────────────────

#[test]
#[ignore = "subprocess helper (fixture builder)"]
fn helper_s3_build_chained_fixture() {
    let dir = PathBuf::from(std::env::var(ENV_TEST_DIR).expect("fixture dir env"));
    assert_eq!(
        std::env::var("ARCGRAPH_M1_FORCE_CHAINED_BAGS").as_deref(),
        Ok("1")
    );
    let (backend, guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
        data_dir: dir.clone(),
    })
    .expect("fixture bootstrap");
    assert!(guard.is_durable());
    let crud = crud_for(&backend, TenantId::DEFAULT);
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
    let (small_ok, chained_raw) = assert_all_bags(&backend, false);
    assert_eq!(
        small_ok, FIXTURE_SMALL,
        "fixture precondition: all small verified chained"
    );
    assert_eq!(chained_raw, FIXTURE_LARGE);
}

#[test]
#[ignore = "subprocess helper (crash-injected migration)"]
fn helper_s3_crash_migrating_reopen() {
    let dir = PathBuf::from(std::env::var(ENV_TEST_DIR).expect("fixture dir env"));
    assert!(std::env::var("ARCGRAPH_M1_MIGRATE_CRASH_AFTER_BATCHES").is_ok());
    let _ = bootstrap_storage_backend(&BootstrapMode::Durable { data_dir: dir });
    panic!("crash-injected migration must have aborted before this line");
}

// ─── The windows ─────────────────────────────────────────────────────

/// w1: crash landed AFTER the migrating MANIFEST but BEFORE the VERSION
/// re-stamp (bootstrap §11 step a → b gap). On-disk: VERSION=1,
/// MANIFEST=migrating. Reopen must resume + complete.
#[test]
fn w1_crash_between_migrating_manifest_and_version_stamp_resumes() {
    let tmp = TempDir::new().expect("tempdir");
    let dir = tmp.path().join("db");
    std::fs::create_dir_all(&dir).unwrap();
    build_pre_m1_fixture(&dir);
    write_data_dir_manifest(&dir, &migrating_manifest()).expect("simulate step (a)");
    assert_eq!(version_of(&dir), 1, "window precondition: VERSION still 1");

    {
        let (backend, _guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
            data_dir: dir.clone(),
        })
        .expect("w1 reopen must resume");
        let (slotted, chained) = assert_all_bags(&backend, true);
        assert_eq!((slotted, chained), (FIXTURE_SMALL, FIXTURE_LARGE));
    }
    assert_eq!(
        version_of(&dir),
        4,
        "v2 M2 ladder end-state (1→3→4 in one boot)"
    );
    assert!(
        !read_data_dir_manifest(&dir)
            .unwrap()
            .unwrap()
            .m1_migration_in_flight()
    );
}

/// w2: crash landed AFTER the VERSION=3 re-stamp but BEFORE the first
/// batch commit (step b → c gap). On-disk: VERSION=3,
/// MANIFEST=migrating, every bag still chained. Reopen must resume.
#[test]
fn w2_crash_after_version_stamp_before_first_batch_resumes() {
    let tmp = TempDir::new().expect("tempdir");
    let dir = tmp.path().join("db");
    std::fs::create_dir_all(&dir).unwrap();
    build_pre_m1_fixture(&dir);
    write_data_dir_manifest(&dir, &migrating_manifest()).expect("simulate step (a)");
    stamp_data_dir(&dir, 3).expect("simulate step (b)");

    {
        let (backend, _guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
            data_dir: dir.clone(),
        })
        .expect("w2 reopen must resume");
        let (slotted, chained) = assert_all_bags(&backend, true);
        assert_eq!((slotted, chained), (FIXTURE_SMALL, FIXTURE_LARGE));
    }
    assert_eq!(
        version_of(&dir),
        4,
        "v2 M2 ladder end-state (1→3→4 in one boot)"
    );
    assert!(
        !read_data_dir_manifest(&dir)
            .unwrap()
            .unwrap()
            .m1_migration_in_flight()
    );
}

/// w3: crash AFTER the LAST batch commit (560 smalls = batch 512 +
/// batch 48; larges skip) but BEFORE reclaim/checkpoint/final MANIFEST
/// — the post-encode-pre-stamp window. Reopen must be a no-op resume
/// that flips the MANIFEST, byte-identical bags.
#[test]
fn w3_crash_after_last_batch_before_final_manifest_resumes() {
    let tmp = TempDir::new().expect("tempdir");
    let dir = tmp.path().join("db");
    std::fs::create_dir_all(&dir).unwrap();
    build_pre_m1_fixture(&dir);

    let out = run_helper(
        "helper_s3_crash_migrating_reopen",
        &dir,
        &[("ARCGRAPH_M1_MIGRATE_CRASH_AFTER_BATCHES", "2")],
    );
    assert!(
        !out.status.success(),
        "crash-injected reopen must die after batch 2.\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(version_of(&dir), 3, "VERSION flipped before the sweep");
    assert!(
        read_data_dir_manifest(&dir)
            .expect("manifest readable after crash")
            .expect("manifest present after crash")
            .m1_migration_in_flight(),
        "post-encode pre-flip window: MANIFEST must still be migrating"
    );

    {
        let (backend, _guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
            data_dir: dir.clone(),
        })
        .expect("w3 reopen must resume (no-op sweep + MANIFEST flip)");
        let (slotted, chained) = assert_all_bags(&backend, true);
        assert_eq!((slotted, chained), (FIXTURE_SMALL, FIXTURE_LARGE));
    }
    assert_eq!(
        version_of(&dir),
        4,
        "v2 M2 ladder end-state (1→3→4 in one boot)"
    );
    assert!(
        !read_data_dir_manifest(&dir)
            .unwrap()
            .unwrap()
            .m1_migration_in_flight(),
        "resume must reach the single commit point"
    );
}

/// w4 — THE ATTACK: crash MID-VERSION-STAMP. `stamp_data_dir` is
/// `File::create` (TRUNCATES the existing 12-byte v1 stamp) →
/// `write_all` → `sync_all`; a kill between create and write_all leaves
/// a ZERO-LENGTH VERSION file (plain process kill, no power loss
/// needed — the truncate syscall completed, the write never issued).
/// Same end state as ENOSPC on the write. Per the §0.2 contract the
/// reopen must RESUME. Does it?
#[test]
fn w4_torn_version_stamp_mid_migration_must_resume() {
    let tmp = TempDir::new().expect("tempdir");
    let dir = tmp.path().join("db");
    std::fs::create_dir_all(&dir).unwrap();
    build_pre_m1_fixture(&dir);
    // Step (a) completed: migrating MANIFEST durable.
    write_data_dir_manifest(&dir, &migrating_manifest()).expect("simulate step (a)");
    // Step (b) torn: File::create truncated the v1 stamp, then the
    // process died before write_all — exactly what stamp_data_dir
    // leaves at that kill point.
    let vpath = version_file_path(&dir);
    let f = std::fs::File::create(&vpath).expect("truncate VERSION (simulated kill mid-stamp)");
    drop(f);
    assert_eq!(
        std::fs::metadata(&vpath).unwrap().len(),
        0,
        "window precondition: zero-length VERSION"
    );

    let res = bootstrap_storage_backend(&BootstrapMode::Durable {
        data_dir: dir.clone(),
    });
    match res {
        Ok((backend, _guard)) => {
            let (slotted, chained) = assert_all_bags(&backend, true);
            assert_eq!((slotted, chained), (FIXTURE_SMALL, FIXTURE_LARGE));
            println!("w4: RESUMED cleanly — property holds");
        }
        Err(e) => {
            panic!(
                "w4: REOPEN AFTER TORN VERSION STAMP DID NOT RESUME — store is \
                 unopenable until manual intervention.\nerror chain: {e:?}"
            );
        }
    }
}

/// w4b: the partial-write variant (magic landed, version bytes did
/// not) — same kill point one syscall later under a short write.
#[test]
fn w4b_partial_version_stamp_mid_migration_must_resume() {
    let tmp = TempDir::new().expect("tempdir");
    let dir = tmp.path().join("db");
    std::fs::create_dir_all(&dir).unwrap();
    build_pre_m1_fixture(&dir);
    write_data_dir_manifest(&dir, &migrating_manifest()).expect("simulate step (a)");
    let vpath = version_file_path(&dir);
    std::fs::write(&vpath, b"ARCGDDV1").expect("partial stamp (8 of 12 bytes)");

    let res = bootstrap_storage_backend(&BootstrapMode::Durable {
        data_dir: dir.clone(),
    });
    match res {
        Ok((backend, _guard)) => {
            let (slotted, chained) = assert_all_bags(&backend, true);
            assert_eq!((slotted, chained), (FIXTURE_SMALL, FIXTURE_LARGE));
            println!("w4b: RESUMED cleanly — property holds");
        }
        Err(e) => {
            panic!("w4b: REOPEN AFTER PARTIAL VERSION STAMP DID NOT RESUME.\nerror chain: {e:?}");
        }
    }
}

/// w5: downgrade through the FULL bootstrap — a data-full dir stamped
/// at an unsupported version must refuse cleanly (typed Incompatible,
/// no panic, no page parse). Models a pre-M1 binary (supported=[1])
/// meeting a v3 dir via the identical decode path.
#[test]
fn w5_full_bootstrap_refuses_unsupported_stamp_cleanly() {
    let tmp = TempDir::new().expect("tempdir");
    let dir = tmp.path().join("db");
    std::fs::create_dir_all(&dir).unwrap();
    build_pre_m1_fixture(&dir);
    // Fully migrate first (a genuine v3 store).
    {
        let (_backend, _guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
            data_dir: dir.clone(),
        })
        .expect("migrating open");
    }
    assert_eq!(
        version_of(&dir),
        4,
        "v2 M2 ladder end-state (1→3→4 in one boot)"
    );
    // Tamper the stamp to a version NO binary supports — the same
    // "found ∉ supported" branch a pre-M1 binary (supported=[1]) takes
    // when it reads found=3.
    stamp_data_dir(&dir, 9).expect("stamp unsupported version");

    let err = match bootstrap_storage_backend(&BootstrapMode::Durable {
        data_dir: dir.clone(),
    }) {
        Ok(_) => panic!("unsupported stamped version must refuse to open"),
        Err(e) => e,
    };
    let chain = format!("{err:?}");
    assert!(
        chain.contains("incompatible on-disk data-dir version"),
        "must surface the typed Incompatible error, got:\n{chain}"
    );
    println!("w5 error chain (clean refusal):\n{chain}");
}
