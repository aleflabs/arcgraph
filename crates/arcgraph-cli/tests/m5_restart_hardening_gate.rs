//! M5-D3 restart-hardening gate — adopted permanent controls (#1518
//! skeptic review, M5-D3 gate-fix charter FIX 1 / FIX 4). Originated as
//! skeptic probes during the M5-D3 (PR #1518) 4-skeptic gate review
//! (pinned at 7de39a4e); promoted to a permanent gate per the charter's
//! "adopt the saved probes into the gate as permanent RED-on-revert
//! controls" directive.
//!
//! Attacks beyond the shipped `stage_restart_resumes_from_durable_manifests_
//! byte_identically` gate:
//!   B  mid-stage crash (partial s4 garbage, NO manifest)  -> reset, no EEXIST
//!   C  mid-materialize crash (planted partial store files + scratch/s7
//!      leftovers + stray scratch junk)                    -> sweep + identity
//!   D  double crash (s2 then s5) then clean run           -> identity
//!   E  crash at W=4, resume at W=2                        -> identity
//!   F  post-GC damage: truncated s4 run (F1) / truncated s3 segment (F2,
//!      s1 runs already GC'd)                              -> degrade + identity
//!   G  torn s5 manifest (half bytes)                      -> degrade + identity
//!   I  input swapped after crash: same length, same first 1 MiB, byte
//!      changed beyond the fingerprint head                -> FIX 1 (full-
//!      stream `InputFingerprint::full_crc`) makes this GREEN by
//!      construction: resume now either rebuilds from scratch (fingerprint
//!      mismatch) or, if a future change re-permits a stale resume, the
//!      byte-identity assertion below still catches any divergence from a
//!      fresh load of the swapped input.
//!
//! Run with: --features fault-injection, --test-threads=1 (env seams are
//! process-global; this binary keeps every probe serial by construction).

use std::fs;
use std::path::{Path, PathBuf};

use arcgraph_cli::m5_load::{
    LoadFormat, LoadLimits, LoadOutcome, LoadReport, M5_RSS_CAP_BYTES, load_data_dir_with_limits,
};
use arcgraph_core::TenantId;
use tempfile::tempdir;

const LOAD_TENANT: TenantId = TenantId::new(83);
const BUILDING: &str = "gen-load-v6.building";
const PINNED_TIMESTAMP: &str = "2026-07-16T00:00:00Z";

fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

struct EnvGuard {
    _lock: std::sync::MutexGuard<'static, ()>,
    keys: Vec<&'static str>,
}

fn with_env(pairs: &[(&'static str, &str)]) -> EnvGuard {
    let lock = env_lock();
    for (key, value) in pairs {
        // SAFETY: all env mutation in this binary is under `env_lock`.
        unsafe { std::env::set_var(key, value) };
    }
    EnvGuard {
        _lock: lock,
        keys: pairs.iter().map(|(key, _)| *key).collect(),
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for key in &self.keys {
            // SAFETY: still under the guard's `env_lock`.
            unsafe { std::env::remove_var(key) };
        }
    }
}

fn pin_clocks() -> EnvGuard {
    with_env(&[
        ("ARCGRAPH_M5_MANIFEST_TIMESTAMP", PINNED_TIMESTAMP),
        ("ARCGRAPH_CHECKPOINT_UNIX_MS", "1752624000000"),
    ])
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(DIGITS[usize::from(byte >> 4)] as char);
        out.push(DIGITS[usize::from(byte & 0xf)] as char);
    }
    out
}

fn node_line(external: &[u8], label: u32, float_bits: u64, opaque: &[u8]) -> String {
    format!(
        "{{\"kind\":\"node\",\"external_id\":\"{}\",\"label_or_type\":{label},\
         \"float_bits\":\"{float_bits:016x}\",\"opaque\":\"{}\"}}\n",
        hex(external),
        hex(opaque)
    )
}

fn rel_line(external: &[u8], source: &[u8], target: &[u8], type_id: u32) -> String {
    format!(
        "{{\"kind\":\"relationship\",\"external_id\":\"{}\",\"source_id\":\"{}\",\
         \"target_id\":\"{}\",\"label_or_type\":{type_id},\
         \"float_bits\":\"0000000000000000\",\"opaque\":\"\"}}\n",
        hex(external),
        hex(source),
        hex(target),
    )
}

/// Mixed fixture: prefix-family ids, gaps, a hub, an oversized (chained)
/// bag, several types — the same shape family as the shipped gate.
fn fixture(dir: &Path) -> PathBuf {
    let path = dir.join("probe.jsonl");
    let mut body = String::new();
    let mut ids: Vec<Vec<u8>> = Vec::new();
    for index in 0..400_u64 {
        let mut external = match index % 4 {
            0 => format!("p{:03}", index / 4).into_bytes(),
            1 => {
                let mut id = format!("p{:03}", index / 4).into_bytes();
                id.push(0);
                id.extend_from_slice(b"tail");
                id
            }
            2 => format!("n{index:010}").into_bytes(),
            _ => format!("gap-{index:05}").into_bytes(),
        };
        if index % 4 == 0 && index >= 200 {
            external.extend_from_slice(format!("-{index}").as_bytes());
        }
        let opaque: Vec<u8> = if index == 77 {
            vec![0xAB; 9_000] // oversized chained bag -> blob/DEC-4 path
        } else {
            (0..(index % 64)).map(|byte| byte as u8).collect()
        };
        body.push_str(&node_line(&external, (index % 7) as u32, index, &opaque));
        ids.push(external);
    }
    for index in 0..1_100_u64 {
        let external = format!("r{index:010}").into_bytes();
        let source = &ids[(index as usize * 7 + 1) % 300];
        let target = if index % 3 == 0 {
            &ids[0]
        } else {
            &ids[(index as usize * 13 + 2) % 300]
        };
        body.push_str(&rel_line(&external, source, target, (index % 5) as u32));
    }
    fs::write(&path, body).expect("write fixture");
    path
}

fn tree_bytes(root: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    fn walk(base: &Path, path: &Path, out: &mut Vec<(PathBuf, Vec<u8>)>) {
        let mut entries: Vec<_> = fs::read_dir(path)
            .expect("read dir")
            .map(|entry| entry.expect("dir entry").path())
            .collect();
        entries.sort();
        for entry in entries {
            if entry.is_dir() {
                walk(base, &entry, out);
            } else {
                let rel = entry.strip_prefix(base).expect("relative").to_path_buf();
                if rel.as_os_str() == "LOCK" {
                    continue;
                }
                out.push((rel, fs::read(&entry).expect("read file")));
            }
        }
    }
    let mut out = Vec::new();
    walk(root, root, &mut out);
    out
}

fn assert_trees_identical(reference: &[(PathBuf, Vec<u8>)], root: &Path, label: &str) {
    let tree = tree_bytes(root);
    assert_eq!(reference.len(), tree.len(), "{label}: file census diverged");
    for ((expected_path, expected_bytes), (path, bytes)) in reference.iter().zip(tree.iter()) {
        assert_eq!(expected_path, path, "{label}: tree shape diverged");
        assert!(
            expected_bytes == bytes,
            "{label}: {} diverged ({} vs {} bytes)",
            path.display(),
            expected_bytes.len(),
            bytes.len()
        );
    }
}

fn limits(workers: usize) -> LoadLimits {
    LoadLimits {
        workers: Some(workers),
        sort_memory_bytes: 64 * 1024,
        rss_cap_bytes: M5_RSS_CAP_BYTES,
        rss_sample_every_ms: 25,
        max_disk_bytes: None,
    }
}

fn load_ok(root: &Path, input: &Path, lim: LoadLimits) -> LoadReport {
    match load_data_dir_with_limits(input, LoadFormat::Native, root, LOAD_TENANT, lim)
        .expect("load must succeed")
    {
        LoadOutcome::Loaded(report) => report,
        other => panic!("expected Loaded, got {other:?}"),
    }
}

/// Run a load armed to crash right after `stage`'s manifest is durable.
fn crash_after(root: &Path, input: &Path, stage: &str, lim: LoadLimits) {
    // SAFETY: caller holds the env lock via pin_clocks.
    unsafe { std::env::set_var("ARCGRAPH_M5_CRASH_AFTER_STAGE", stage) };
    let error = load_data_dir_with_limits(input, LoadFormat::Native, root, LOAD_TENANT, lim)
        .expect_err("injected stage crash must surface");
    // SAFETY: same lock.
    unsafe { std::env::remove_var("ARCGRAPH_M5_CRASH_AFTER_STAGE") };
    assert!(
        format!("{error:#}").contains("injected crash after stage"),
        "unexpected error: {error:#}"
    );
}

fn scratch(root: &Path) -> PathBuf {
    root.join(BUILDING).join("scratch")
}

/// Reference tree for the fixture at W=4 (fresh dir, uninterrupted).
fn reference(input: &Path) -> (tempfile::TempDir, Vec<(PathBuf, Vec<u8>)>, LoadReport) {
    let root = tempdir().expect("reference dir");
    let report = load_ok(root.path(), input, limits(4));
    let tree = tree_bytes(root.path());
    (root, tree, report)
}

// ─── B: mid-stage crash (partial s4, no manifest) ────────────────────────

#[test]
fn probe_b_partial_next_stage_garbage_resets_without_eexist() {
    let _env = pin_clocks();
    let dir = tempdir().expect("fixture");
    let input = fixture(dir.path());
    let (_ref_root, ref_tree, ref_report) = reference(&input);

    let root = tempdir().expect("root");
    crash_after(root.path(), &input, "s3-phase2-segments", limits(4));

    // Simulate a mid-s4 kill: partial run + index files where s4 will
    // write its very first outputs (create_new collision bait in EVERY
    // range-worker dir — empty ranges skip file creation, so covering all
    // four keeps the collision deterministic), plus a half-written file
    // with NO index sidecar.
    for worker in 0..4 {
        let dir = scratch(root.path()).join(format!("s4/w{worker}"));
        fs::create_dir_all(&dir).expect("plant s4 dir");
        fs::write(dir.join("run-00000000.bin"), b"torn-partial-run").expect("plant");
        fs::write(dir.join("run-00000000.bin.idx"), b"torn").expect("plant");
        fs::write(dir.join("run-00000001.bin"), b"x").expect("plant");
    }

    let report = load_ok(root.path(), &input, limits(4));
    assert_eq!(
        report.resumed_stages,
        vec![
            "s1-canonical-runs".to_owned(),
            "s2-phase1-counts".to_owned(),
            "s3-phase2-segments".to_owned(),
        ],
        "must resume exactly the durable prefix"
    );
    assert_eq!(report.nodes, ref_report.nodes);
    assert_eq!(report.relationships, ref_report.relationships);
    assert_trees_identical(&ref_tree, root.path(), "probe B");
}

// ─── C: mid-materialize crash (partial stores + s7 + stray junk) ─────────

#[test]
fn probe_c_partial_materialization_swept_byte_identically() {
    let _env = pin_clocks();
    let dir = tempdir().expect("fixture");
    let input = fixture(dir.path());
    let (_ref_root, ref_tree, _) = reference(&input);

    let root = tempdir().expect("root");
    crash_after(root.path(), &input, "s6-tel-segments", limits(4));

    // Simulate a crash mid-materialize: partial store artifacts in the
    // building root (files AND dirs), the s7 refs spools (create_new
    // collision bait for build_tel_direction), a stray scratch temp, and
    // a bogus manifest name nothing should ever read.
    let building = root.path().join(BUILDING);
    fs::write(building.join("LSN_SEED"), b"junk").expect("plant");
    fs::create_dir_all(building.join("wal")).expect("plant");
    fs::write(building.join("wal").join("000001.waseg"), b"torn").expect("plant");
    fs::create_dir_all(building.join("store_records")).expect("plant");
    fs::write(
        building.join("store_records").join("extent-0"),
        vec![0xCD; 8192],
    )
    .expect("plant");
    let s7 = scratch(root.path()).join("s7");
    fs::create_dir_all(&s7).expect("plant s7");
    fs::write(s7.join("tel.out.refs"), b"stale-refs").expect("plant");
    fs::write(s7.join("tel.in.refs"), b"stale-refs").expect("plant");
    fs::write(scratch(root.path()).join("leftover.tmp"), b"junk").expect("plant");
    fs::write(
        scratch(root.path()).join("manifests").join("bogus.json"),
        b"{}",
    )
    .expect("plant");

    let report = load_ok(root.path(), &input, limits(4));
    assert_eq!(report.resumed_stages.len(), 6, "all six stages resumed");
    assert_trees_identical(&ref_tree, root.path(), "probe C");
}

// ─── D: double crash then clean ──────────────────────────────────────────

#[test]
fn probe_d_double_crash_resumes_byte_identically() {
    let _env = pin_clocks();
    let dir = tempdir().expect("fixture");
    let input = fixture(dir.path());
    let (_ref_root, ref_tree, _) = reference(&input);

    let root = tempdir().expect("root");
    crash_after(root.path(), &input, "s2-phase1-counts", limits(4));
    crash_after(root.path(), &input, "s5-rel-tel-runs", limits(4));
    let report = load_ok(root.path(), &input, limits(4));
    assert_eq!(
        report.resumed_stages,
        vec![
            "s1-canonical-runs".to_owned(),
            "s2-phase1-counts".to_owned(),
            "s3-phase2-segments".to_owned(),
            "s4-resolved-runs".to_owned(),
            "s5-rel-tel-runs".to_owned(),
        ],
        "second resume must see the extended durable prefix"
    );
    assert_trees_identical(&ref_tree, root.path(), "probe D");
}

// ─── E: resume with a different worker count ─────────────────────────────

#[test]
fn probe_e_resume_with_different_worker_count_byte_identical() {
    let _env = pin_clocks();
    let dir = tempdir().expect("fixture");
    let input = fixture(dir.path());
    let (_ref_root, ref_tree, _) = reference(&input);

    let root = tempdir().expect("root");
    crash_after(root.path(), &input, "s3-phase2-segments", limits(4));
    let report = load_ok(root.path(), &input, limits(2));
    assert_eq!(report.resumed_stages.len(), 3);
    assert_trees_identical(&ref_tree, root.path(), "probe E (W=4 crash, W=2 resume)");
}

// ─── F: post-GC damage degrades safely ───────────────────────────────────

fn files_under(dir: &Path, pattern: &str) -> Vec<PathBuf> {
    fn walk(dir: &Path, pattern: &str, out: &mut Vec<PathBuf>) {
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };
        for entry in entries {
            let path = entry.expect("entry").path();
            if path.is_dir() {
                walk(&path, pattern, out);
            } else if path.to_string_lossy().contains(pattern) {
                out.push(path);
            }
        }
    }
    let mut out = Vec::new();
    walk(dir, pattern, &mut out);
    out.sort();
    out
}

fn first_file_under(dir: &Path, pattern: &str) -> PathBuf {
    files_under(dir, pattern)
        .into_iter()
        .next()
        .unwrap_or_else(|| panic!("no {pattern} under {}", dir.display()))
}

#[test]
fn probe_f1_truncated_s4_run_degrades_to_s3_resume() {
    let _env = pin_clocks();
    let dir = tempdir().expect("fixture");
    let input = fixture(dir.path());
    let (_ref_root, ref_tree, _) = reference(&input);

    let root = tempdir().expect("root");
    crash_after(root.path(), &input, "s4-resolved-runs", limits(4));
    // s1 run FILES were GC'd when s3's manifest became durable (empty
    // per-worker dirs may remain); now damage a manifest-named s4
    // resolved run (simulated torn tail after power cut).
    assert!(
        files_under(&scratch(root.path()).join("s1"), ".bin").is_empty(),
        "precondition: s1 run files must already be GC'd (staged-GC ran after s3)"
    );
    let victim = first_file_under(&scratch(root.path()).join("s4"), ".bin");
    let bytes = fs::read(&victim).expect("read victim");
    assert!(bytes.len() > 8, "victim too small to truncate meaningfully");
    fs::write(&victim, &bytes[..bytes.len() / 2]).expect("truncate victim");

    let report = load_ok(root.path(), &input, limits(4));
    assert_eq!(
        report.resumed_stages,
        vec![
            "s1-canonical-runs".to_owned(),
            "s2-phase1-counts".to_owned(),
            "s3-phase2-segments".to_owned(),
        ],
        "invalid s4 group must degrade the resume point to s3"
    );
    assert_trees_identical(&ref_tree, root.path(), "probe F1");
}

#[test]
fn probe_f2_truncated_s3_segment_after_gc_degrades_to_full_rebuild() {
    let _env = pin_clocks();
    let dir = tempdir().expect("fixture");
    let input = fixture(dir.path());
    let (_ref_root, ref_tree, _) = reference(&input);

    let root = tempdir().expect("root");
    crash_after(root.path(), &input, "s4-resolved-runs", limits(4));
    // Damage an s3 NODE segment (last consumer = the always-re-run
    // materializer, so EVERY candidate R >= 3 must validate it; R <= 2
    // needs the s1 runs, which are GC'd). Full rebuild required.
    let victim = first_file_under(&scratch(root.path()).join("s3"), "nodes-");
    let bytes = fs::read(&victim).expect("read victim");
    fs::write(&victim, &bytes[..bytes.len() / 2]).expect("truncate victim");

    let report = load_ok(root.path(), &input, limits(4));
    assert!(
        report.resumed_stages.is_empty(),
        "with s1 GC'd and s3 damaged there is no feasible resume point, got {:?}",
        report.resumed_stages
    );
    assert_trees_identical(&ref_tree, root.path(), "probe F2");
}

// ─── G: torn manifest ────────────────────────────────────────────────────

#[test]
fn probe_g_torn_manifest_degrades_one_stage() {
    let _env = pin_clocks();
    let dir = tempdir().expect("fixture");
    let input = fixture(dir.path());
    let (_ref_root, ref_tree, _) = reference(&input);

    let root = tempdir().expect("root");
    crash_after(root.path(), &input, "s5-rel-tel-runs", limits(4));
    let manifest = scratch(root.path())
        .join("manifests")
        .join("s5-rel-tel-runs.json");
    let bytes = fs::read(&manifest).expect("read manifest");
    fs::write(&manifest, &bytes[..bytes.len() / 2]).expect("tear manifest");

    let report = load_ok(root.path(), &input, limits(4));
    assert_eq!(
        report.resumed_stages,
        vec![
            "s1-canonical-runs".to_owned(),
            "s2-phase1-counts".to_owned(),
            "s3-phase2-segments".to_owned(),
            "s4-resolved-runs".to_owned(),
        ],
        "torn s5 manifest must degrade the resume point to s4"
    );
    assert_trees_identical(&ref_tree, root.path(), "probe G");
}

// ─── I: input identity beyond the 1 MiB fingerprint head ────────────────

/// Designed-skip control: a group whose LAST consumer is already durable
/// (s3 bindings after the s4 manifest) may be damaged/absent — resume must
/// STILL proceed from s4 and stay byte-identical (partial-GC semantics).
#[test]
fn probe_f3_damaged_group_with_durable_consumer_is_skipped_soundly() {
    let _env = pin_clocks();
    let dir = tempdir().expect("fixture");
    let input = fixture(dir.path());
    let (_ref_root, ref_tree, _) = reference(&input);

    let root = tempdir().expect("root");
    crash_after(root.path(), &input, "s4-resolved-runs", limits(4));
    let victim = first_file_under(&scratch(root.path()).join("s3"), "bindings-");
    fs::write(&victim, b"garbage").expect("damage skippable group");

    let report = load_ok(root.path(), &input, limits(4));
    assert_eq!(
        report.resumed_stages.len(),
        4,
        "damage to a consumed group must not lower the resume point"
    );
    assert_trees_identical(&ref_tree, root.path(), "probe F3");
}

/// >1 MiB fixture whose LAST record can be edited without changing length.
fn big_fixture(dir: &Path, name: &str, tail_marker: u8) -> PathBuf {
    let path = dir.join(name);
    let mut body = String::new();
    let mut ids: Vec<Vec<u8>> = Vec::new();
    for index in 0..600_u64 {
        // Fat opaques -> the file sails past FINGERPRINT_HEAD_BYTES.
        let opaque = vec![(index % 251) as u8; 2_048];
        let external = format!("n{index:010}").into_bytes();
        body.push_str(&node_line(&external, (index % 7) as u32, index, &opaque));
        ids.push(external);
    }
    for index in 0..200_u64 {
        let external = format!("r{index:010}").into_bytes();
        body.push_str(&rel_line(
            &external,
            &ids[(index as usize * 7) % 600],
            &ids[(index as usize * 13 + 1) % 600],
            (index % 5) as u32,
        ));
    }
    // Final node whose opaque payload carries the marker byte — same
    // length under both markers, well beyond the 1 MiB head.
    body.push_str(&node_line(b"tail-sentinel", 0, 0, &[tail_marker; 64]));
    fs::write(&path, body).expect("write big fixture");
    path
}

#[test]
fn probe_i_input_swap_beyond_fingerprint_head_diverges_or_refuses() {
    let _env = pin_clocks();
    let dir = tempdir().expect("fixture");
    let input = big_fixture(dir.path(), "input.jsonl", 0x11);
    assert!(
        fs::metadata(&input).expect("stat").len() > 2 * 1024 * 1024,
        "fixture must exceed the 1 MiB fingerprint head"
    );
    let swapped = big_fixture(dir.path(), "swapped.jsonl", 0x22);
    assert_eq!(
        fs::metadata(&input).expect("stat").len(),
        fs::metadata(&swapped).expect("stat").len(),
        "swap must preserve length"
    );
    assert_eq!(
        &fs::read(&input).expect("read")[..1 << 20],
        &fs::read(&swapped).expect("read")[..1 << 20],
        "swap must preserve the first 1 MiB (the fingerprint head)"
    );

    let root = tempdir().expect("root");
    crash_after(root.path(), &input, "s3-phase2-segments", limits(4));

    // Operator swaps the input file AFTER the crash: same length, same
    // first 1 MiB, different tail-sentinel opaque.
    fs::rename(&swapped, &input).expect("swap input in place");

    // Reference: uninterrupted load of the swapped content into a fresh dir.
    let swapped_reference_root = tempdir().expect("ref root");

    let resumed = load_data_dir_with_limits(
        &input,
        LoadFormat::Native,
        root.path(),
        LOAD_TENANT,
        limits(4),
    );

    let reference_report = load_ok(swapped_reference_root.path(), &input, limits(4));
    let ref_tree = tree_bytes(swapped_reference_root.path());

    match resumed {
        Ok(LoadOutcome::Loaded(report)) => {
            if report.resumed_stages.is_empty() {
                // Fingerprint caught the swap -> full rebuild of new input.
                assert_trees_identical(&ref_tree, root.path(), "probe I rebuild");
            } else {
                // Resume proceeded from STALE stages: the store must still
                // equal a fresh load of the NEW input for the resume to be
                // sound. If this assert fires, the loader silently served
                // the OLD input's bytes.
                assert_eq!(reference_report.nodes, report.nodes, "probe I census");
                assert_trees_identical(
                    &ref_tree,
                    root.path(),
                    "probe I STALE-RESUME (silent divergence from the swapped input)",
                );
            }
        }
        Ok(other) => panic!("unexpected outcome {other:?}"),
        Err(error) => panic!("resume after input swap errored: {error:#}"),
    }
}
