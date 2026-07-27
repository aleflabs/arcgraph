//! M5-D1 leg-(c) attach-spine gates (`docs/design/M5D-REDESIGN-AMENDMENT.md`
//! §2.7 + §7 M5-D1 row): the five-point `LoadFault` kill-9 sweep with its
//! restart matrix (INV-M5.1/.3/.6/.23), the virgin-precondition refusals
//! (INV-M5.21), the cross-tool generation-namespace plants (INV-M5.22), the
//! stale-WAL plant (INV-M5.11 fresh half), `AlreadyLoaded` idempotence, and
//! the release-lane fsync-bypass negative controls (INV-M5.6).
//!
//! Oracle independence note: generation names are asserted as on-disk string
//! literals here, deliberately NOT through the `generation_namespace`
//! registry — a registry re-pointing regression (INV-M5.22 RED-on-revert)
//! must redden these gates rather than move them.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use arcgraph_cli::bootstrap::{BootstrapMode, bootstrap_storage_backend};
use arcgraph_cli::data_dir_migration::{current_generation, upgrade_data_dir};
use arcgraph_cli::data_lock::DataDirLock;
use arcgraph_cli::m5_load::{
    LoadFault, LoadFormat, LoadOutcome, LoadRefusal, load_data_dir, load_data_dir_with_fault,
};
use arcgraph_core::{NodeId, TenantId};
use arcgraph_storage::crud::read_node_with_store;
use arcgraph_storage::wal::{BUNDLE_FORMAT_V10, SegmentHeader, segment_filename};
use tempfile::tempdir;

const CHILD_ROOT: &str = "ARCGRAPH_M5_LOAD_KILL_ROOT";
const CHILD_INPUT: &str = "ARCGRAPH_M5_LOAD_KILL_INPUT";
const CHILD_FAULT: &str = "ARCGRAPH_M5_LOAD_KILL_FAULT";
const COMPLETE_ROOT: &str = "ARCGRAPH_M5_LOAD_COMPLETE_ROOT";
const COMPLETE_INPUT: &str = "ARCGRAPH_M5_LOAD_COMPLETE_INPUT";
const COMPLETE_EXPECT: &str = "ARCGRAPH_M5_LOAD_COMPLETE_EXPECT";
#[cfg(all(feature = "fault-injection", not(debug_assertions)))]
const LEDGER_PROOF_ROOT: &str = "ARCGRAPH_M5_LOAD_LEDGER_PROOF_ROOT";
#[cfg(all(feature = "fault-injection", not(debug_assertions)))]
const LEDGER_PROOF_INPUT: &str = "ARCGRAPH_M5_LOAD_LEDGER_PROOF_INPUT";
#[cfg(feature = "fault-injection")]
const INDEX_SKIP_ROOT: &str = "ARCGRAPH_M5_LOAD_INDEX_SKIP_ROOT";
#[cfg(feature = "fault-injection")]
const INDEX_SKIP_INPUT: &str = "ARCGRAPH_M5_LOAD_INDEX_SKIP_INPUT";

const RACE_ROOT: &str = "ARCGRAPH_M5_RACE_ROOT";
const RACE_INPUT: &str = "ARCGRAPH_M5_RACE_INPUT";
const RACE_DELAY_US: &str = "ARCGRAPH_M5_RACE_DELAY_US";

const LOAD_TENANT: TenantId = TenantId::new(77);
const FINAL_GENERATION: &str = "gen-load-v6";
const BUILDING_GENERATION: &str = "gen-load-v6.building";

/// Three nodes (`a` < `b` < `c` by external-id byte order, so dense ids
/// 1/2/3) and two relationships. Hex-encoded per the native boundary.
fn native_input_fixture(dir: &Path) -> PathBuf {
    let path = dir.join("input.jsonl");
    let record = |kind: &str, ext: &str, label: u32, extra: &str| {
        format!(
            "{{\"kind\":\"{kind}\",\"external_id\":\"{ext}\",\"label_or_type\":{label},\
             \"float_bits\":\"3ff0000000000000\",\"opaque\":\"beef\"{extra}}}"
        )
    };
    let body = [
        record("node", "61", 7, ""), // "a"
        record("node", "62", 7, ""), // "b"
        record("node", "63", 9, ""), // "c"
        record(
            "relationship",
            "7231",
            3,
            ",\"source_id\":\"61\",\"target_id\":\"62\"",
        ), // "r1": a -> b
        record(
            "relationship",
            "7232",
            3,
            ",\"source_id\":\"62\",\"target_id\":\"63\"",
        ), // "r2": b -> c
    ]
    .join("\n");
    fs::write(&path, body).expect("write native input fixture");
    path
}

/// Complete tree census: sorted (relative path, file bytes) pairs. Dirs are
/// represented by their children; the byte-identical assertions compare the
/// full file SET plus every file's exact content.
fn tree_bytes(root: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    fn walk(base: &Path, path: &Path, out: &mut Vec<(PathBuf, Vec<u8>)>) {
        let mut entries: Vec<_> = fs::read_dir(path)
            .expect("read dir")
            .collect::<std::io::Result<Vec<_>>>()
            .expect("dir entries");
        entries.sort_by_key(std::fs::DirEntry::file_name);
        for entry in entries {
            let entry_path = entry.path();
            if entry.file_type().expect("file type").is_dir() {
                walk(base, &entry_path, out);
            } else {
                out.push((
                    entry_path
                        .strip_prefix(base)
                        .expect("relative")
                        .to_path_buf(),
                    fs::read(&entry_path).expect("read file"),
                ));
            }
        }
    }
    let mut out = Vec::new();
    if root.exists() {
        walk(root, root, &mut out);
    }
    out
}

/// Cold-open the committed store through PRODUCTION bootstrap and read node 1
/// under the loaded non-default tenant (the same oracle shape the leg-(b)
/// migration gates use).
fn assert_loaded_store_serves(root: &Path) {
    let mode = BootstrapMode::Durable {
        data_dir: root.to_path_buf(),
    };
    let (backend, guard) =
        bootstrap_storage_backend(&mode).expect("committed loaded store must boot");
    let routed = backend
        .router()
        .route(TenantId::DEFAULT, arcgraph_core::PartitionId::ZERO)
        .expect("route default partition");
    let reader = backend.txn_manager().begin(LOAD_TENANT);
    let node = read_node_with_store(routed.crud(), &reader, NodeId::new(1))
        .expect("read loaded node 1")
        .expect("loaded node 1 must exist");
    assert_eq!(node.label_id, 7, "node 1 label survives the load");
    reader.abort();
    drop(routed);
    drop(backend);
    drop(guard);
}

fn fault_from_name(name: &str) -> LoadFault {
    match name {
        "scratch" => LoadFault::AfterScratchCreate,
        "build-sync" => LoadFault::AfterBuildSync,
        "rename" => LoadFault::AfterGenerationRename,
        "current" => LoadFault::AfterCurrentSwap,
        "version" => LoadFault::AfterVersionStamp,
        _ => panic!("unknown load fault {name}"),
    }
}

/// Serialize this file's parent tests. Several tests hold a `DataDirLock`
/// (or acquire-drop-reacquire one back to back) inside the multi-threaded
/// test harness while SIBLING tests fork gate subprocesses. On Unix, a
/// forked-but-not-yet-exec'd child momentarily shares the parent's open
/// file descriptions (`O_CLOEXEC` closes only AT exec), so a concurrently
/// spawned sibling child can extend a just-dropped `flock` by that fork
/// window — under machine load the window stretches and an immediate
/// re-acquire sees `EWOULDBLOCK` (observed once in the full-workspace
/// suite: `load_refuses_held_lock` "released lock must admit the load").
/// Holding one file-scoped mutex across each test body means no sibling
/// forks while any test holds or hands over a lock, which closes the race
/// by construction instead of by retry. Child-process invocations (env
/// dispatch) each run alone in their own process — the guard is
/// uncontended there.
fn serialize_gate() -> std::sync::MutexGuard<'static, ()> {
    static GATE_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GATE_SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn kill_self() -> ! {
    #[cfg(unix)]
    // SAFETY: raising SIGKILL on the current pid takes no pointers; delivery
    // is asynchronous, so park until it lands (it models a hard crash for
    // the parent's restart matrix — the parent asserts the SIGKILL status).
    unsafe {
        libc::kill(libc::getpid(), libc::SIGKILL);
    }
    #[cfg(not(unix))]
    std::process::abort();
    #[allow(unreachable_code)]
    loop {
        std::thread::park();
    }
}

/// Subprocess: run one faulted load and die by SIGKILL (the §2.5 crash).
#[test]
fn m5_load_kill9_child() {
    let (Ok(root), Ok(input), Ok(fault)) = (
        std::env::var(CHILD_ROOT),
        std::env::var(CHILD_INPUT),
        std::env::var(CHILD_FAULT),
    ) else {
        return;
    };
    let error = load_data_dir_with_fault(
        Path::new(&input),
        LoadFormat::Native,
        Path::new(&root),
        LOAD_TENANT,
        fault_from_name(&fault),
    )
    .expect_err("child fault must interrupt the load");
    assert!(
        format!("{error:#}").contains("injected"),
        "unexpected child error chain: {error:#}"
    );
    kill_self();
}

/// Subprocess: run one complete production load (the rerun of the matrix)
/// and assert its outcome class.
#[test]
fn m5_load_complete_child() {
    let (Ok(root), Ok(input)) = (std::env::var(COMPLETE_ROOT), std::env::var(COMPLETE_INPUT))
    else {
        return;
    };
    let expect = std::env::var(COMPLETE_EXPECT).unwrap_or_else(|_| "loaded".to_owned());
    let outcome = load_data_dir(
        Path::new(&input),
        LoadFormat::Native,
        Path::new(&root),
        LOAD_TENANT,
    )
    .expect("rerun after crash must complete the load (never EEXIST)");
    match (expect.as_str(), &outcome) {
        ("loaded", LoadOutcome::Loaded(report)) if !report.resumed => {}
        ("resumed", LoadOutcome::Loaded(report)) if report.resumed => {}
        ("already", LoadOutcome::AlreadyLoaded { tenant_census }) => {
            assert!(
                tenant_census.contains(&LOAD_TENANT.raw()),
                "idempotent census lost the loaded tenant: {tenant_census:?}"
            );
        }
        (expected, other) => panic!("expected {expected} outcome, got {other:?}"),
    }
}

/// INV-M5.1/.3/.6/.23 — the §2.7 leg-(c) kill-9 sweep + §2.5 restart matrix,
/// one subprocess crash per `LoadFault` point, each followed by the matrix's
/// rerun row. RED-on-revert: stamp VERSION before CURRENT (the sweep's
/// pre-commit rows find a VERSION), make the commit callable without the
/// ledger proof (release control below), or reinstate the superseded
/// `ensure!(!final.exists())` (every post-rename rerun turns EEXIST-red).
#[test]
fn fresh_load_kill9_sweep_and_restart_matrix() {
    let _serial = serialize_gate();
    for fault in ["scratch", "build-sync", "rename", "current", "version"] {
        let fixture = tempdir().unwrap();
        let input = native_input_fixture(fixture.path());
        // The data dir is a sibling of the input: a virgin target must stay
        // virgin (the input file itself would trip the §2.3 precondition).
        let root = fixture.path().join("data");

        let status = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "m5_load_kill9_child", "--nocapture"])
            .env(CHILD_ROOT, &root)
            .env(CHILD_INPUT, &input)
            .env(CHILD_FAULT, fault)
            .status()
            .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            assert_eq!(status.signal(), Some(libc::SIGKILL), "fault={fault}");
        }

        let building = root.join(BUILDING_GENERATION);
        let final_generation = root.join(FINAL_GENERATION);
        let current = root.join("CURRENT");
        let expect_rerun = match fault {
            "scratch" | "build-sync" => {
                // Pre-commit rows: no commit coordinate exists; the orphan is
                // exactly one own-namespace `.building` dir (scratch lives
                // INSIDE it — the root is never littered).
                assert!(building.is_dir(), "fault={fault} lost the crash orphan");
                assert!(
                    !final_generation.exists() && !current.exists(),
                    "fault={fault} exposed a commit coordinate before the ledger"
                );
                assert!(
                    !building.join("VERSION").exists(),
                    "fault={fault} stamped VERSION before CURRENT"
                );
                for entry in fs::read_dir(&root).unwrap() {
                    let name = entry.unwrap().file_name();
                    let name = name.to_string_lossy().into_owned();
                    assert!(
                        [BUILDING_GENERATION, "LOCK"].contains(&name.as_str()),
                        "fault={fault} littered the data-dir root with {name:?}"
                    );
                }
                "loaded"
            }
            "rename" => {
                assert!(
                    final_generation.is_dir() && !building.exists(),
                    "fault={fault} did not publish the complete generation"
                );
                assert!(
                    !current.exists() && !final_generation.join("VERSION").exists(),
                    "fault={fault} exposed CURRENT/VERSION before their turn"
                );
                "resumed"
            }
            "current" => {
                assert_eq!(
                    fs::read_to_string(&current).unwrap(),
                    format!("{FINAL_GENERATION}\n"),
                    "fault={fault} CURRENT content"
                );
                assert!(
                    !final_generation.join("VERSION").exists(),
                    "fault={fault} VERSION must be the LAST durable act"
                );
                // INV-M5.3 visibility rollback: an unstamped fresh-load
                // generation resolves to NO committed generation.
                assert_eq!(
                    current_generation(&root).unwrap(),
                    None,
                    "fault={fault} unstamped generation became visible"
                );
                "resumed"
            }
            "version" => {
                assert!(
                    final_generation.join("VERSION").exists(),
                    "fault={fault} commit did not complete"
                );
                assert_eq!(
                    current_generation(&root).unwrap().as_deref(),
                    Some(final_generation.as_path()),
                    "fault={fault} committed generation must be visible"
                );
                "already"
            }
            _ => unreachable!(),
        };

        // §2.5 restart matrix: the rerun completes the load or no-ops.
        let status = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "m5_load_complete_child", "--nocapture"])
            .env(COMPLETE_ROOT, &root)
            .env(COMPLETE_INPUT, &input)
            .env(COMPLETE_EXPECT, expect_rerun)
            .status()
            .unwrap();
        assert!(status.success(), "fault={fault} rerun failed");

        assert_eq!(
            fs::read_to_string(root.join("CURRENT")).unwrap(),
            format!("{FINAL_GENERATION}\n"),
            "fault={fault} rerun did not commit CURRENT"
        );
        assert!(
            final_generation.join("VERSION").exists(),
            "fault={fault} rerun did not finish the VERSION-last stamp"
        );
        assert!(
            !building.exists(),
            "fault={fault} rerun left its own scratch behind"
        );
        assert!(
            !final_generation.join("scratch").exists(),
            "fault={fault} pipeline intermediates survived into the served generation"
        );
        assert_loaded_store_serves(&root);
    }
}

/// INV-M5.21 — `load_refuses_populated_data_dir`: a populated dir refuses
/// with the typed error and ZERO mutation (byte-identical tree, including
/// the case where the populated dir carries no LOCK file yet). Arm 1 is a
/// REAL durable store (the orphaned-tenant fixture class from V-1a); arm 2
/// is a foreign generation namespace. RED-on-revert: delete the virgin
/// precondition and arm 1 builds a second store over the live one.
#[test]
fn load_refuses_populated_data_dir() {
    let _serial = serialize_gate();
    // Arm 1: a real single-tenant durable store produced by production
    // bootstrap (pages.db + wal/ + VERSION + MANIFEST + LOCK).
    let populated = tempdir().unwrap();
    {
        let mode = BootstrapMode::Durable {
            data_dir: populated.path().to_path_buf(),
        };
        let (backend, guard) = bootstrap_storage_backend(&mode).expect("seed populated fixture");
        drop(backend);
        drop(guard);
    }
    let input = native_input_fixture_outside(populated.path());
    let before = tree_bytes(populated.path());
    assert!(!before.is_empty(), "populated fixture must carry a store");
    let error = load_data_dir(&input, LoadFormat::Native, populated.path(), LOAD_TENANT)
        .expect_err("populated dir must refuse");
    let refusal = error
        .downcast_ref::<LoadRefusal>()
        .expect("refusal must be the typed LoadRefusal");
    assert!(
        matches!(refusal, LoadRefusal::PopulatedDataDir { .. }),
        "unexpected refusal variant: {refusal:?}"
    );
    let message = format!("{error:#}");
    assert!(
        message.contains("arcgraph migrate") && message.contains("M5-F"),
        "refusal must name the supported alternatives: {message}"
    );
    assert_eq!(
        tree_bytes(populated.path()),
        before,
        "populated refusal must not create, modify, or remove ANY file"
    );

    // Arm 2: a dir owned by the migrate tool's namespace (no LOCK file at
    // all — the refusal must precede lockfile creation).
    let foreign = tempdir().unwrap();
    fs::create_dir(foreign.path().join("gen-v10.building")).unwrap();
    fs::write(
        foreign.path().join("gen-v10.building").join("sentinel"),
        b"migrate-owned bytes",
    )
    .unwrap();
    let input = native_input_fixture_outside(foreign.path());
    let before = tree_bytes(foreign.path());
    let error = load_data_dir(&input, LoadFormat::Native, foreign.path(), LOAD_TENANT)
        .expect_err("foreign generation namespace must refuse");
    assert!(
        error.downcast_ref::<LoadRefusal>().is_some(),
        "foreign-namespace refusal must be typed: {error:#}"
    );
    assert_eq!(
        tree_bytes(foreign.path()),
        before,
        "loader touched a foreign tool's namespace (INV-M5.22) or created LOCK before refusing"
    );
}

/// Input fixture placed OUTSIDE the target data dir (a populated dir must
/// not gain the input file either).
fn native_input_fixture_outside(_target: &Path) -> PathBuf {
    let holder = tempdir().unwrap();
    let path = native_input_fixture(holder.path());
    // Leak the tempdir so the input outlives the fixture holder.
    let (path_owned, _leaked) = (path, Box::leak(Box::new(holder)));
    path_owned
}

/// INV-M5.21 — `load_refuses_held_lock`: a concurrently-held `DataDirLock`
/// refuses before any store mutation; dropping the holder makes the same
/// invocation succeed (proving the refusal was the lock, not the dir).
#[test]
fn load_refuses_held_lock() {
    let _serial = serialize_gate();
    let fixture = tempdir().unwrap();
    let input = native_input_fixture_outside(fixture.path());
    let held = DataDirLock::acquire(fixture.path()).expect("hold the dir like a live server");
    let before = tree_bytes(fixture.path());
    let error = load_data_dir(&input, LoadFormat::Native, fixture.path(), LOAD_TENANT)
        .expect_err("held lock must refuse the load");
    assert!(
        format!("{error:#}").contains("already in use"),
        "refusal must name the #886 exclusion: {error:#}"
    );
    assert_eq!(
        tree_bytes(fixture.path()),
        before,
        "held-lock refusal must not mutate the dir"
    );
    drop(held);
    let outcome = load_data_dir(&input, LoadFormat::Native, fixture.path(), LOAD_TENANT)
        .expect("released lock must admit the load");
    assert!(matches!(outcome, LoadOutcome::Loaded(_)));
    assert_loaded_store_serves(fixture.path());
}

/// INV-M5.22 — cross-tool namespace ownership, migrate side: the migrate
/// tool's startup reaper and its explicit entry point both leave a
/// loader-owned dir untouched. (The loader side is arm 2 of
/// `load_refuses_populated_data_dir`; the full v5→v6 migration run over a
/// planted loader `.building` is asserted in
/// `m4_data_dir_migration_gate.rs::generation_commit_point_identity_kill9_sweep`.)
/// RED-on-revert: re-point the loader registry row at `gen-v10` — the
/// disjointness unit test and both plants go red.
#[test]
fn migrate_tools_leave_loader_namespace_untouched() {
    let _serial = serialize_gate();
    // A committed loader store…
    let fixture = tempdir().unwrap();
    let input = native_input_fixture_outside(fixture.path());
    load_data_dir(&input, LoadFormat::Native, fixture.path(), LOAD_TENANT).expect("seed load");
    let before = tree_bytes(fixture.path());

    // …survives the durable-bootstrap startup reaper byte-identical…
    arcgraph_cli::data_dir_migration::resume_generation_cleanup(
        fixture.path(),
        arcgraph_cli::data_dir_migration::GenerationCleanupFault::None,
    )
    .expect("startup reaper over a loader-owned dir is a no-op");
    assert_eq!(
        tree_bytes(fixture.path()),
        before,
        "migrate startup reaper touched the loader namespace"
    );

    // …and `arcgraph migrate upgrade-data-dir` refuses it by name, with
    // zero mutation.
    let error =
        upgrade_data_dir(fixture.path()).expect_err("migrate must refuse a loader-owned dir");
    assert!(
        format!("{error:#}").contains("owned by the fresh-load tool"),
        "migrate refusal must name the owning tool: {error:#}"
    );
    assert_eq!(
        tree_bytes(fixture.path()),
        before,
        "migrate refusal mutated the loader namespace"
    );
}

/// INV-M5.23 — `AlreadyLoaded` idempotence: a rerun over a committed load is
/// a byte-identical no-op with the census reported; a rerun naming a tenant
/// OUTSIDE the census is an operator error, not a silent no-op.
#[test]
fn already_loaded_rerun_is_byte_identical_noop() {
    let _serial = serialize_gate();
    let fixture = tempdir().unwrap();
    let input = native_input_fixture_outside(fixture.path());
    let outcome =
        load_data_dir(&input, LoadFormat::Native, fixture.path(), LOAD_TENANT).expect("load");
    assert!(matches!(outcome, LoadOutcome::Loaded(_)));
    let before = tree_bytes(fixture.path());

    let outcome = load_data_dir(&input, LoadFormat::Native, fixture.path(), LOAD_TENANT)
        .expect("rerun over a committed load must no-op");
    match outcome {
        LoadOutcome::AlreadyLoaded { tenant_census } => {
            assert_eq!(
                tenant_census,
                vec![TenantId::DEFAULT.raw(), LOAD_TENANT.raw()],
                "census must name DEFAULT + the loaded tenant"
            );
        }
        other => panic!("expected AlreadyLoaded, got {other:?}"),
    }
    assert_eq!(
        tree_bytes(fixture.path()),
        before,
        "AlreadyLoaded rerun must not write anything"
    );

    let error = load_data_dir(
        &input,
        LoadFormat::Native,
        fixture.path(),
        TenantId::new(88),
    )
    .expect_err("rerun naming a non-census tenant must refuse");
    assert!(
        format!("{error:#}").contains("does not contain tenant 88"),
        "census mismatch must be named: {error:#}"
    );
}

/// INV-M5.11 (fresh half) — a planted pre-existing WAL segment inside the
/// unpublished/unstamped generation is deterministically refused on resume
/// (the stale-WAL plant, verbatim intent from the leg-(b) gate).
/// RED-on-revert: carry a foreign WAL dir into the generation and the
/// resume rows commit a store whose recovery would replay foreign deltas.
#[test]
fn stale_wal_plant_refused_on_resume() {
    let _serial = serialize_gate();
    for fault in ["rename", "current"] {
        let fixture = tempdir().unwrap();
        let input = native_input_fixture(fixture.path());
        let root = fixture.path().join("data");
        let status = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "m5_load_kill9_child", "--nocapture"])
            .env(CHILD_ROOT, &root)
            .env(CHILD_INPUT, &input)
            .env(CHILD_FAULT, fault)
            .status()
            .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            assert_eq!(status.signal(), Some(libc::SIGKILL), "fault={fault}");
        }

        // Plant: a segment-0 whose bytes extend past the bare header — i.e.
        // a WAL carrying records from some other life.
        let wal = root.join(FINAL_GENERATION).join("wal");
        let mut stale = SegmentHeader {
            format_version: BUNDLE_FORMAT_V10,
        }
        .encode()
        .to_vec();
        stale.extend_from_slice(b"stale delta bytes from a prior generation");
        fs::write(wal.join(segment_filename(0)), &stale).unwrap();

        let error = load_data_dir(&input, LoadFormat::Native, &root, LOAD_TENANT)
            .expect_err("planted stale WAL must refuse the resume");
        assert!(
            format!("{error:#}").contains("contains records or a torn header"),
            "fault={fault}: unexpected stale-WAL refusal: {error:#}"
        );
        assert!(
            !root.join(FINAL_GENERATION).join("VERSION").exists(),
            "fault={fault}: a tampered generation must never finish its commit"
        );
    }
}

/// Input fixture for the concurrent-loader race: the standard 5 records plus
/// 2000 padding nodes so the build window is wide enough for two real loader
/// subprocesses to collide in. External ids are HEX-ENCODED BYTES and
/// materialization requires the decoded bytes to be UTF-8 — so hex-encode an
/// ASCII id (`n<i>`).
fn race_input_fixture(dir: &Path) -> PathBuf {
    let path = dir.join("input.jsonl");
    let record = |kind: &str, ext: &str, label: u32, extra: &str| {
        format!(
            "{{\"kind\":\"{kind}\",\"external_id\":\"{ext}\",\"label_or_type\":{label},\
             \"float_bits\":\"3ff0000000000000\",\"opaque\":\"beef\"{extra}}}"
        )
    };
    let mut body: Vec<String> = vec![
        record("node", "61", 7, ""), // "a"
        record("node", "62", 7, ""), // "b"
        record("node", "63", 9, ""), // "c"
        record(
            "relationship",
            "7231",
            3,
            ",\"source_id\":\"61\",\"target_id\":\"62\"",
        ),
        record(
            "relationship",
            "7232",
            3,
            ",\"source_id\":\"62\",\"target_id\":\"63\"",
        ),
    ];
    for i in 0..2000u32 {
        let ascii = format!("n{i:08}");
        let hex: String = ascii.bytes().map(|b| format!("{b:02x}")).collect();
        body.push(record("node", &hex, 5, ""));
    }
    fs::write(&path, body.join("\n")).expect("write race input fixture");
    path
}

/// Subprocess: run one PRODUCTION load and report the outcome class on
/// stdout (env-gated no-op when invoked as an ordinary test).
#[test]
fn m5_race_child() {
    let (Ok(root), Ok(input)) = (std::env::var(RACE_ROOT), std::env::var(RACE_INPUT)) else {
        return;
    };
    if let Ok(delay) = std::env::var(RACE_DELAY_US) {
        let delay: u64 = delay.parse().expect("delay micros");
        std::thread::sleep(std::time::Duration::from_micros(delay));
    }
    match load_data_dir(
        Path::new(&input),
        LoadFormat::Native,
        Path::new(&root),
        LOAD_TENANT,
    ) {
        Ok(LoadOutcome::Loaded(report)) => {
            println!("RACE_OUTCOME=loaded resumed={}", report.resumed);
        }
        Ok(LoadOutcome::AlreadyLoaded { tenant_census }) => {
            println!("RACE_OUTCOME=already census={tenant_census:?}");
        }
        Err(error) => {
            let chain = format!("{error:#}");
            if chain.contains("already in use") {
                println!("RACE_OUTCOME=refused-lock");
            } else if chain.contains("is populated") {
                println!("RACE_OUTCOME=refused-populated chain={chain}");
            } else {
                println!("RACE_OUTCOME=error chain={chain}");
            }
        }
    }
}

fn spawn_race_child(root: &Path, input: &Path, delay_us: u64) -> std::process::Child {
    Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "m5_race_child", "--nocapture"])
        .env(RACE_ROOT, root)
        .env(RACE_INPUT, input)
        .env(RACE_DELAY_US, delay_us.to_string())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn race child")
}

fn race_outcome_of(child: std::process::Child) -> String {
    let output = child.wait_with_output().expect("child wait");
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(
        output.status.success(),
        "race child crashed (status {:?})\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status
    );
    stdout
        .lines()
        .find(|line| line.starts_with("RACE_OUTCOME="))
        .unwrap_or_else(|| panic!("child printed no outcome\nstdout:\n{stdout}\nstderr:\n{stderr}"))
        .to_owned()
}

/// §2.2 "DataDirLock acquired first and HELD THROUGH build+commit" +
/// INV-M5.23 never-EEXIST, exercised by a REAL race: N rounds of two
/// simultaneous production loader subprocesses on one virgin dir. Per round:
///   1. Exactly one process builds+commits (`Loaded`); the loser lands in the
///      typed "already in use" lock refusal or `AlreadyLoaded` — never
///      `EEXIST`, the "classification bug" backstop, a panic, or any other
///      error class.
///   2. Final state: CURRENT -> gen-load-v6, VERSION stamped, and the store
///      cold-opens through PRODUCTION bootstrap serving the loaded tenant.
///
/// This is the hold-DURATION property `load_refuses_held_lock` cannot see
/// (it only proves acquire-refuses-when-held). RED-on-revert (MUT-G): insert
/// `drop(_lock)` right after `DataDirLock::acquire` in
/// `load_data_dir_with_fault` — both loaders then reach
/// `fs::create_dir(gen-load-v6.building)` and the loser exits the benign
/// class (EEXIST / classification-bug / swept-mid-build corruption, the #886
/// class), reddening this gate.
#[test]
fn concurrent_loaders_exactly_one_wins_and_store_serves() {
    let _serial = serialize_gate();
    let rounds: u64 = std::env::var("ARCGRAPH_M5_RACE_ROUNDS")
        .ok()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(12);
    let mut seen_loser_classes = std::collections::BTreeSet::new();
    for round in 0..rounds {
        let fixture = tempdir().unwrap();
        let input = race_input_fixture(fixture.path());
        let root = fixture.path().join("data");

        // Sample different interleavings: B starts 0..~5.4ms behind A.
        let delay_b = (round % 7) * 900;
        let a = spawn_race_child(&root, &input, 0);
        let b = spawn_race_child(&root, &input, delay_b);
        let oa = race_outcome_of(a);
        let ob = race_outcome_of(b);

        let loaded = [&oa, &ob]
            .iter()
            .filter(|o| o.starts_with("RACE_OUTCOME=loaded"))
            .count();
        let benign_loser =
            |o: &str| o.starts_with("RACE_OUTCOME=already") || o == "RACE_OUTCOME=refused-lock";
        assert_eq!(
            loaded, 1,
            "round {round}: exactly one loader must build+commit; got A={oa} B={ob}"
        );
        let loser = if oa.starts_with("RACE_OUTCOME=loaded") {
            &ob
        } else {
            &oa
        };
        assert!(
            benign_loser(loser),
            "round {round}: loser outcome must be AlreadyLoaded or the typed lock refusal; \
             got A={oa} B={ob}"
        );
        seen_loser_classes.insert(loser.split_whitespace().next().unwrap_or(loser).to_owned());

        assert_eq!(
            fs::read_to_string(root.join("CURRENT")).expect("CURRENT after race"),
            format!("{FINAL_GENERATION}\n"),
            "round {round}: race left a wrong CURRENT"
        );
        assert!(
            root.join(FINAL_GENERATION).join("VERSION").exists(),
            "round {round}: race left the generation unstamped"
        );
        assert!(
            !root.join(BUILDING_GENERATION).exists(),
            "round {round}: race left scratch behind"
        );
        assert_loaded_store_serves(&root);
    }
    eprintln!("loser classes observed across {rounds} rounds: {seen_loser_classes:?}");
}

/// INV-M5.6 release lane — the fsync-bypass negative control, verbatim from
/// the leg-(b) gate: a false complete-generation ledger proof handed to the
/// SHARED publication object must refuse publication in a release build
/// (page-cache state is irrelevant because the unproven build never becomes
/// a commit coordinate). Run with:
/// `cargo test -p arcgraph-cli --release --features fault-injection --test m5_load_attach_gate`
#[cfg(all(feature = "fault-injection", not(debug_assertions)))]
#[test]
fn load_ledger_proof_consumed_in_release() {
    let _serial = serialize_gate();
    let fixture = tempdir().unwrap();
    let input = native_input_fixture(fixture.path());
    let root = fixture.path().join("data");
    let status = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "load_ledger_proof_production_child",
            "--nocapture",
        ])
        .env(LEDGER_PROOF_ROOT, &root)
        .env(LEDGER_PROOF_INPUT, &input)
        .env("ARCGRAPH_M5_LOAD_MISSING_LEDGER", "1")
        .status()
        .unwrap();
    assert!(status.success(), "production false-proof child failed");
    assert!(
        !root.join(FINAL_GENERATION).exists() && !root.join("CURRENT").exists(),
        "a false durability proof crossed the publication boundary in release"
    );
    // The refused build reruns to completion once the proof is real.
    let outcome = load_data_dir(&input, LoadFormat::Native, &root, LOAD_TENANT)
        .expect("clean rerun after refused proof");
    assert!(matches!(outcome, LoadOutcome::Loaded(_)));
    assert_loaded_store_serves(&root);
}

#[cfg(all(feature = "fault-injection", not(debug_assertions)))]
#[test]
fn load_ledger_proof_production_child() {
    let (Ok(root), Ok(input)) = (
        std::env::var(LEDGER_PROOF_ROOT),
        std::env::var(LEDGER_PROOF_INPUT),
    ) else {
        return;
    };
    let error = load_data_dir(
        Path::new(&input),
        LoadFormat::Native,
        Path::new(&root),
        LOAD_TENANT,
    )
    .expect_err("release publication must refuse a false durability proof");
    assert!(
        format!("{error:#}").contains("complete durability ledger proof is absent"),
        "unexpected false-proof refusal: {error:#}"
    );
}

/// INV-M5.13/M5.6 — the index-files-only fsync-bypass variant: skipping the
/// index/vector pass fsync must refuse the commit (the proof carries
/// `synced = false` into the shared publication object).
#[cfg(feature = "fault-injection")]
#[test]
fn load_index_fsync_skip_refuses_commit() {
    let _serial = serialize_gate();
    let fixture = tempdir().unwrap();
    let input = native_input_fixture(fixture.path());
    let root = fixture.path().join("data");
    let status = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "load_index_fsync_skip_child", "--nocapture"])
        .env(INDEX_SKIP_ROOT, &root)
        .env(INDEX_SKIP_INPUT, &input)
        .env("ARCGRAPH_M5_SKIP_INDEX_FSYNC", "1")
        .status()
        .unwrap();
    assert!(status.success(), "index-fsync-skip child failed");
    assert!(
        !root.join(FINAL_GENERATION).exists() && !root.join("CURRENT").exists(),
        "an unsynced index pass crossed the publication boundary"
    );
}

#[cfg(feature = "fault-injection")]
#[test]
fn load_index_fsync_skip_child() {
    let (Ok(root), Ok(input)) = (
        std::env::var(INDEX_SKIP_ROOT),
        std::env::var(INDEX_SKIP_INPUT),
    ) else {
        return;
    };
    let error = load_data_dir(
        Path::new(&input),
        LoadFormat::Native,
        Path::new(&root),
        LOAD_TENANT,
    )
    .expect_err("skipped index fsync must refuse the commit");
    assert!(
        format!("{error:#}").contains("index/vector pass fsync proof is absent"),
        "unexpected index-skip refusal: {error:#}"
    );
}
