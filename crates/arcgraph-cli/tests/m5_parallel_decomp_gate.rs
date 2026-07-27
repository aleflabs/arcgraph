//! M5-D3 gates — parallel decomposition + census-derived budgets
//! (`docs/design/M5D-REDESIGN-AMENDMENT.md` §4 + §5, §M5-D3 gate table).
//!
//! | invariant | gate | RED-on-revert |
//! |---|---|---|
//! | INV-M5.24 parallel≡serial determinism | W∈{1,2,16} byte-identical generations over the tombstone/gap/duplicate fixture family, under the deterministic adversarial scheduler | `ARCGRAPH_M5_ARRIVAL_ORDER_IDS` (id assignment by arrival order, phase 1 dropped) |
//! | INV-M5.19 parallel re-proof | cross-run planted-duplicate fixture (amendment §4.2(3) proof exercised) | `ARCGRAPH_M5_RANGE_BY_RUN` (dup ranges by RUN, not by KEY) |
//! | INV-M5.25 budgets-derive-from-census | both-rung projection gates (arithmetic-only) + derived-caps-are-live substrate proof | `ARCGRAPH_M5_FIXED_BULK_CAPS` / `ARCGRAPH_M5_ZERO_BULK_BUDGET` |
//! | INV-M5.15 parallel regime | continuous-RSS at W=16 with every sample under the cap + steady plateau | `ARCGRAPH_M5_COLLECT_ALL` (collect-all control) |
//! | HEADLINE 100M | `bounded_rss_100m_production_rung_continuous`, EXECUTED on the pinned box with the run artifact attached (an `#[ignore]` without the artifact does NOT satisfy the gate) | evidence rule |
//!
//! Determinism-oracle discipline (memory: serial gates are blind to
//! default-on concurrency defects): the byte-identical gate is a
//! DETERMINISTIC BARRIER test — `ARCGRAPH_M5_WORKER_STAGGER_MS` forces
//! reverse worker completion order, so an id-assignment defect that
//! depends on arrival order diverges deterministically, not
//! timing-hopefully.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use arcgraph_cli::m5_load::{
    LoadFormat, LoadLimits, LoadOutcome, LoadRefusal, LoadReport, M5_RSS_CAP_BYTES,
    load_data_dir_with_limits, plan_owner_budgets, project_load_disk,
};
use arcgraph_cli::m5_parallel::LoadCensus;
use arcgraph_core::TenantId;
use arcgraph_storage::owner_budget::{BulkClassCensus, OwnerSubstrateBudget};
use arcgraph_storage::{OWNER_INDEX_DISK_CAP_BYTES, OWNER_PAYLOAD_DISK_CAP_BYTES};
use tempfile::tempdir;

const LOAD_TENANT: TenantId = TenantId::new(83);
const FINAL_GENERATION: &str = "gen-load-v6";
const PINNED_TIMESTAMP: &str = "2026-07-16T00:00:00Z";
const GIB: u64 = 1024 * 1024 * 1024;

// ─────────────────────────────────────────────────────────────────────
// Environment plumbing (process-global; serialized within this binary)
// ─────────────────────────────────────────────────────────────────────

fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

struct EnvGuard {
    _lock: std::sync::MutexGuard<'static, ()>,
    keys: Vec<&'static str>,
}

/// Serialize parent tests (the M5-D2 `serialize_gate` discipline): every
/// test below either holds a `DataDirLock` through a load or forks gate
/// subprocesses (`assert_red_under`); on Unix a forked-not-yet-exec'd
/// child momentarily shares the parent's open file descriptions, so a
/// sibling fork can extend a just-dropped `flock`. One file-scoped mutex
/// across each test body closes the race by construction. `with_env(&[])`
/// is the pure-serialization form.
fn with_env(pairs: &[(&'static str, &str)]) -> EnvGuard {
    let lock = env_lock();
    for (key, value) in pairs {
        // SAFETY: all env mutation in this test binary happens under
        // `env_lock`, and no other thread reads these keys concurrently
        // outside guarded sections.
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

/// Re-run `test_name` in a subprocess with `env_var` armed and assert it
/// goes RED — the reverted-defect control (same pattern as the D2 gates).
fn assert_red_under(test_name: &str, env_var: &str) {
    let output = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test_name, "--nocapture", "--test-threads", "1"])
        .env(env_var, "1")
        .output()
        .unwrap();
    assert!(
        !output.status.success(),
        "{test_name} stayed GREEN under armed {env_var} — the gate is a no-op\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

// ─────────────────────────────────────────────────────────────────────
// Fixtures
// ─────────────────────────────────────────────────────────────────────

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

fn rel_line(
    external: &[u8],
    source: &[u8],
    target: &[u8],
    type_id: u32,
    float_bits: u64,
    opaque: &[u8],
) -> String {
    format!(
        "{{\"kind\":\"relationship\",\"external_id\":\"{}\",\"source_id\":\"{}\",\
         \"target_id\":\"{}\",\"label_or_type\":{type_id},\
         \"float_bits\":\"{float_bits:016x}\",\"opaque\":\"{}\"}}\n",
        hex(external),
        hex(source),
        hex(target),
        hex(opaque)
    )
}

/// ULP-adversarial float corpus (subset of the INV-M5.12 family).
const FLOAT_CORPUS: [u64; 8] = [
    0x0000000000000000,
    0x8000000000000000,
    0x0000000000000001,
    0x3ff0000000000000,
    0x3ff0000000000001,
    0x7ff0000000000000,
    0x7ff8000000000001,
    0xffefffffffffffff,
];

/// The INV-M5.24 fixture family: mixed-length external ids including
/// prefix pairs, id gaps (nodes with no adjacency — the "tombstone/gap"
/// shape a fresh load can carry), a dense hub, several relationship
/// types, ULP-adversarial floats, empty + oversized opaques.
fn determinism_fixture(dir: &Path) -> PathBuf {
    let path = dir.join("determinism.jsonl");
    let mut body = String::new();
    let mut node_ids: Vec<Vec<u8>> = Vec::new();
    for index in 0..900_u64 {
        let mut external = match index % 5 {
            // Prefix families: `p<k>` vs `p<k>\0tail` orderings.
            0 => format!("p{:03}", index / 5).into_bytes(),
            1 => {
                let mut id = format!("p{:03}", index / 5).into_bytes();
                id.push(0);
                id.extend_from_slice(b"tail");
                id
            }
            2 => format!("n{index:012}").into_bytes(),
            3 => vec![b'x'; 1 + (index as usize % 61)],
            _ => format!("gap-{index:06}").into_bytes(),
        };
        if matches!(index % 5, 3) {
            // Make the long-run ids unique.
            external.extend_from_slice(format!("-{index}").as_bytes());
        }
        let opaque: Vec<u8> = match index % 7 {
            0 => Vec::new(),
            6 if index % 49 == 6 => vec![(index % 251) as u8; 9_000], // oversized (chained) bag
            _ => (0..(index % 90)).map(|byte| byte as u8).collect(),
        };
        body.push_str(&node_line(
            &external,
            (index % 9) as u32,
            FLOAT_CORPUS[(index % FLOAT_CORPUS.len() as u64) as usize],
            &opaque,
        ));
        node_ids.push(external);
    }
    // Dense hub + sparse tail: node 0 participates in many edges; the
    // `gap-*` nodes participate in none.
    for index in 0..2_600_u64 {
        let external = format!("r{index:012}").into_bytes();
        let source = &node_ids[(index as usize * 7 + 1) % 540];
        let target = if index % 3 == 0 {
            &node_ids[0]
        } else {
            &node_ids[(index as usize * 13 + 2) % 540]
        };
        body.push_str(&rel_line(
            &external,
            source,
            target,
            (index % 6) as u32,
            FLOAT_CORPUS[(index % FLOAT_CORPUS.len() as u64) as usize],
            &index.to_le_bytes(),
        ));
    }
    fs::write(&path, body).expect("write determinism fixture");
    path
}

/// Byte-tree of a directory, LOCK excluded (advisory lockfile, not store
/// state).
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

fn ci_limits(workers: usize) -> LoadLimits {
    LoadLimits {
        workers: Some(workers),
        sort_memory_bytes: 64 * 1024, // force many runs at fixture scale
        rss_cap_bytes: M5_RSS_CAP_BYTES,
        rss_sample_every_ms: 25,
        max_disk_bytes: None,
    }
}

fn load_report(root: &Path, input: &Path, limits: LoadLimits) -> LoadReport {
    match load_data_dir_with_limits(input, LoadFormat::Native, root, LOAD_TENANT, limits)
        .expect("load")
    {
        LoadOutcome::Loaded(report) => report,
        other => panic!("expected a fresh load, got {other:?}"),
    }
}

// ─────────────────────────────────────────────────────────────────────
// Gate 1 — INV-M5.24 parallel≡serial byte-identical determinism
// ─────────────────────────────────────────────────────────────────────

/// THE load-bearing proof: W=1 (serial), W=2, and W=16 loads of the
/// tombstone/gap/duplicate-family fixture produce BYTE-IDENTICAL
/// generations (all stores, page-LSN stamps included; MANIFEST timestamp
/// pinned so the comparison has NO exclusion list), under the
/// deterministic adversarial scheduler (reverse worker completion order).
/// W=16 additionally runs twice — a rerun must equal itself.
///
/// RED-on-revert: `ARCGRAPH_M5_ARRIVAL_ORDER_IDS` (drop the phase-1
/// prefix-sum; bases follow modeled arrival order) → W>1 bytes diverge.
#[test]
fn inv_m5_24_parallel_serial_byte_identical_generations() {
    let _env = with_env(&[
        ("ARCGRAPH_M5_MANIFEST_TIMESTAMP", PINNED_TIMESTAMP),
        ("ARCGRAPH_CHECKPOINT_UNIX_MS", "1752624000000"),
        ("ARCGRAPH_M5_WORKER_STAGGER_MS", "2"),
    ]);
    let fixture_dir = tempdir().expect("fixture dir");
    let input = determinism_fixture(fixture_dir.path());

    type TreeAndReport = (Vec<(PathBuf, Vec<u8>)>, LoadReport);
    let mut baseline: Option<TreeAndReport> = None;
    for (label, workers) in [("W=1", 1), ("W=2", 2), ("W=16", 16), ("W=16 rerun", 16)] {
        let root = tempdir().expect("data dir");
        let report = load_report(root.path(), &input, ci_limits(workers));
        assert_eq!(report.workers, workers as u64, "{label}");
        assert!(report.records > 0 && report.nodes == 900 && report.relationships == 2_600);
        // Anti-vacuousness: the fixture must exercise props, oversized
        // (chained) bags, and BOTH TEL directions, or the byte compare
        // proves nothing about the load-bearing stores.
        assert!(
            report.prop_pages > 0
                && report.chained_bags > 0
                && report.out_tel_entries == 2_600
                && report.in_tel_entries == 2_600,
            "{label}: fixture failed to exercise the full store surface: {report:?}"
        );
        let tree = tree_bytes(root.path());
        assert!(
            tree.iter()
                .any(|(path, _)| path.starts_with(FINAL_GENERATION)),
            "{label}: committed generation present"
        );
        match &baseline {
            None => baseline = Some((tree, report)),
            Some((expected_tree, expected_report)) => {
                assert_eq!(
                    (
                        expected_report.records,
                        expected_report.nodes,
                        expected_report.relationships,
                        expected_report.prop_pages,
                        expected_report.chained_bags,
                        expected_report.out_tel_entries,
                        expected_report.in_tel_entries,
                    ),
                    (
                        report.records,
                        report.nodes,
                        report.relationships,
                        report.prop_pages,
                        report.chained_bags,
                        report.out_tel_entries,
                        report.in_tel_entries,
                    ),
                    "{label}: census diverged BEFORE byte compare"
                );
                let diverged: Vec<String> = expected_tree
                    .iter()
                    .zip(tree.iter())
                    .filter(|((_, a), (_, b))| a != b)
                    .map(|((path, _), _)| path.display().to_string())
                    .collect();
                if !diverged.is_empty() {
                    eprintln!("{label}: diverging files: {diverged:?}");
                }
                assert_eq!(
                    expected_tree.len(),
                    tree.len(),
                    "{label}: file count diverged from W=1"
                );
                for ((expected_path, expected_bytes), (path, bytes)) in
                    expected_tree.iter().zip(tree.iter())
                {
                    assert_eq!(expected_path, path, "{label}: tree shape diverged");
                    assert!(
                        expected_bytes == bytes,
                        "{label}: {} diverged from W=1 ({} vs {} bytes) — INV-M5.24 \
                         byte-identity violated",
                        path.display(),
                        expected_bytes.len(),
                        bytes.len()
                    );
                }
                assert_eq!(
                    (
                        expected_report.records,
                        expected_report.nodes,
                        expected_report.relationships,
                        expected_report.prop_pages,
                        expected_report.chained_bags,
                        expected_report.out_tel_entries,
                        expected_report.in_tel_entries,
                    ),
                    (
                        report.records,
                        report.nodes,
                        report.relationships,
                        report.prop_pages,
                        report.chained_bags,
                        report.out_tel_entries,
                        report.in_tel_entries,
                    ),
                    "{label}: census diverged"
                );
            }
        }
    }
}

/// RED control for INV-M5.24: with id assignment by (modeled) worker
/// arrival order, the byte-identical gate MUST fail.
#[test]
fn inv_m5_24_red_on_revert_arrival_order_ids() {
    let _serial = with_env(&[]);
    assert_red_under(
        "inv_m5_24_parallel_serial_byte_identical_generations",
        "ARCGRAPH_M5_ARRIVAL_ORDER_IDS",
    );
}

// ─────────────────────────────────────────────────────────────────────
// Gate 2 — INV-M5.19 parallel re-proof (cross-run planted duplicate)
// ─────────────────────────────────────────────────────────────────────

/// A duplicate whose occurrences come from DIFFERENT workers' runs: one
/// copy in the first byte-range partition, one in the last (W=4), with a
/// sort budget small enough that each worker spills multiple runs. The
/// §4.2(3) proof says range assignment by KEY makes the occurrences
/// sort-adjacent in exactly one merge worker — the load MUST refuse.
///
/// RED-on-revert: `ARCGRAPH_M5_RANGE_BY_RUN` (dup adjacency recognized
/// only within a single source run) → the cross-run duplicate is missed
/// and the load completes.
#[test]
fn inv_m5_19_cross_run_planted_duplicate_refused() {
    let _serial = with_env(&[]);
    let fixture_dir = tempdir().expect("fixture dir");
    let input = fixture_dir.path().join("dup.jsonl");
    let mut body = String::new();
    // First partition: the duplicate's first occurrence + filler.
    body.push_str(&node_line(b"dup-node", 1, 0x3ff0000000000000, b"first"));
    for index in 0..400_u64 {
        body.push_str(&node_line(
            format!("filler-a-{index:06}").as_bytes(),
            0,
            0,
            &[],
        ));
    }
    // Middle partitions: unrelated nodes (no rels touch the duplicate).
    for index in 0..800_u64 {
        body.push_str(&node_line(
            format!("filler-b-{index:06}").as_bytes(),
            0,
            0,
            &[],
        ));
    }
    // Last partition: the duplicate's second occurrence (different payload).
    for index in 0..400_u64 {
        body.push_str(&node_line(
            format!("filler-c-{index:06}").as_bytes(),
            0,
            0,
            &[],
        ));
    }
    body.push_str(&node_line(b"dup-node", 2, 0x4000000000000000, b"second"));
    fs::write(&input, body).expect("write dup fixture");

    let root = tempdir().expect("data dir");
    let error = load_data_dir_with_limits(
        &input,
        LoadFormat::Native,
        root.path(),
        LOAD_TENANT,
        ci_limits(4),
    )
    .expect_err("cross-run planted duplicate must refuse the load");
    let rendered = format!("{error:#}");
    assert!(
        rendered.contains("duplicate node external_id"),
        "expected the duplicate hard error, got: {rendered}"
    );
}

/// RED control for INV-M5.19: with dup ranges assigned by RUN instead of
/// by KEY, the cross-run duplicate straddles workers and the gate above
/// MUST fail (the load no longer refuses).
#[test]
fn inv_m5_19_red_on_revert_range_by_run() {
    let _serial = with_env(&[]);
    assert_red_under(
        "inv_m5_19_cross_run_planted_duplicate_refused",
        "ARCGRAPH_M5_RANGE_BY_RUN",
    );
}

/// INV-M5.18 (salvaged from #1504 per amendment §9): a relationship
/// naming a nonexistent endpoint is a deterministic hard error under the
/// parallel join, never a skipped relationship.
#[test]
fn inv_m5_18_missing_endpoint_is_hard_error_under_parallel_join() {
    let _serial = with_env(&[]);
    let fixture_dir = tempdir().expect("fixture dir");
    let input = fixture_dir.path().join("missing.jsonl");
    let mut body = String::new();
    for index in 0..64_u64 {
        body.push_str(&node_line(format!("n{index:04}").as_bytes(), 0, 0, &[]));
    }
    body.push_str(&rel_line(b"r-ok", b"n0001", b"n0002", 0, 0, &[]));
    body.push_str(&rel_line(b"r-bad", b"n0001", b"ghost", 0, 0, &[]));
    fs::write(&input, body).expect("write fixture");
    let root = tempdir().expect("data dir");
    let error = load_data_dir_with_limits(
        &input,
        LoadFormat::Native,
        root.path(),
        LOAD_TENANT,
        ci_limits(4),
    )
    .expect_err("missing endpoint must refuse the load");
    let rendered = format!("{error:#}");
    assert!(
        rendered.contains("missing endpoint"),
        "expected the missing-endpoint hard error, got: {rendered}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// Gate 3 — INV-M5.25 budgets-derive-from-census
// ─────────────────────────────────────────────────────────────────────

/// Both-rung projection gate (arithmetic-only, runs in CI without the
/// corpora). 100M+500M and 1B+5B censuses: the derived caps cover the
/// formula need, and the LANDED fixed constants would have refused at
/// plan time (the V-2 regression pin).
///
/// RED-on-revert: `ARCGRAPH_M5_FIXED_BULK_CAPS` reinstates the fixed
/// 3 GiB / 8 GiB on the bulk path → this gate goes red.
#[test]
fn inv_m5_25_budget_projection_both_rungs() {
    // M5-D3 FIX 4 (#1518 skeptic review) — env-lock hole: `plan_owner_budgets`
    // reads process-global env seams (`ARCGRAPH_M5_FIXED_BULK_CAPS`,
    // `ARCGRAPH_M5_ZERO_BULK_BUDGET`, `ARCGRAPH_M5_ZERO_REL_PAYLOAD_BUDGET`),
    // so this test must hold the file-wide env lock like every other test
    // in this binary that touches those seams — otherwise a concurrently
    // running seam-armed test (e.g. `inv_m5_25_rel_payload_budget_is_class_and_field_scoped`)
    // can bleed its env var into this one (or vice versa) under
    // `--test-threads` > 1.
    let _serial = with_env(&[]);
    for (label, nodes, rels) in [
        ("100M rung", 100_000_000_u64, 500_000_000_u64),
        ("1B rung", 1_000_000_000, 5_000_000_000),
    ] {
        let census = LoadCensus {
            records: nodes + rels,
            nodes,
            relationships: rels,
            node_external_id_bytes: nodes * 32,
            rel_external_id_bytes: rels * 32,
            payload_bytes: (nodes + rels) * 96,
        };
        let budgets = plan_owner_budgets(&census);
        let rel_need = OwnerSubstrateBudget::projected_need_bytes(BulkClassCensus {
            entries: rels,
            external_id_bytes: rels * 32,
        });
        // The rel-bindings class alone exceeds BOTH landed constants at
        // both rungs: fixed caps cannot govern the bulk path.
        assert!(
            rel_need > OWNER_INDEX_DISK_CAP_BYTES + OWNER_PAYLOAD_DISK_CAP_BYTES,
            "{label}: fixed constants must be insufficient (V-2 pin)"
        );
        // Derived caps must cover the exact formula need per class.
        assert!(
            budgets.rel_bindings.index_cap_bytes > OWNER_INDEX_DISK_CAP_BYTES
                && budgets.rel_bindings.payload_cap_bytes > OWNER_PAYLOAD_DISK_CAP_BYTES,
            "{label}: derived rel caps must exceed the landed constants \
             (index={}, payload={})",
            budgets.rel_bindings.index_cap_bytes,
            budgets.rel_bindings.payload_cap_bytes
        );
        assert!(
            budgets
                .rel_bindings
                .index_cap_bytes
                .saturating_add(budgets.rel_bindings.payload_cap_bytes)
                >= rel_need,
            "{label}: derived caps must cover the projected need"
        );
        let projection = project_load_disk(&census, &budgets);
        assert!(
            projection.required_bytes > projection.substrate_bytes,
            "{label}: projection must include generation + scratch traffic"
        );
    }
}

/// RED control for INV-M5.25.
#[test]
fn inv_m5_25_red_on_revert_fixed_bulk_caps() {
    let _serial = with_env(&[]);
    assert_red_under(
        "inv_m5_25_budget_projection_both_rungs",
        "ARCGRAPH_M5_FIXED_BULK_CAPS",
    );
}

/// Derived-caps-are-LIVE proof (memory: gates must exercise the arm
/// production dispatches to): collapsing the derived budget to 1 byte
/// makes a well-formed small load fail with the substrate's own
/// `DiskBudgetExceeded` — the derived value IS what the owner substrate
/// enforces on the bulk path, not a parallel constant.
#[test]
fn inv_m5_25_derived_caps_reach_the_owner_substrate() {
    let fixture_dir = tempdir().expect("fixture dir");
    let input = fixture_dir.path().join("tiny.jsonl");
    let mut body = String::new();
    for index in 0..64_u64 {
        // Oversized external ids so the payload companion (capped at 1
        // byte under the seam) must take overflow writes.
        body.push_str(&node_line(
            format!("node-{index:06}-{}", "x".repeat(400)).as_bytes(),
            0,
            0,
            &[],
        ));
    }
    fs::write(&input, body).expect("write fixture");

    // Green half: the same load under real derived budgets succeeds.
    {
        let _serial = with_env(&[]);
        let root = tempdir().expect("data dir");
        load_report(root.path(), &input, ci_limits(2));
    }
    // Armed half: derived caps collapsed to 1 byte must trip the
    // substrate's fail-closed guard mid-build.
    let _env = with_env(&[("ARCGRAPH_M5_ZERO_BULK_BUDGET", "1")]);
    let root = tempdir().expect("data dir");
    let error = load_data_dir_with_limits(
        &input,
        LoadFormat::Native,
        root.path(),
        LOAD_TENANT,
        ci_limits(2),
    )
    .expect_err("1-byte derived budget must fail closed at the substrate");
    let rendered = format!("{error:#}");
    assert!(
        rendered.contains("disk budget exceeded"),
        "expected the substrate DiskBudgetExceeded, got: {rendered}"
    );
}

/// M5-D3 FIX 4 (#1518 skeptic review) — budget cap-transposition seam.
///
/// `OwnerWriters::create` (`m4_migration.rs`) destructures
/// `(index_cap, payload_cap)` per class from `budgets.{node,rel}_bindings`
/// and passes them to `OwnerForwardIndex::create` / `OwnerPayloadStore::create`
/// respectively. Transposing `index_cap_bytes`/`payload_cap_bytes` at that
/// site is UNDETECTABLE by any gate that only checks a class's caps SUM to
/// (or both individually exceed) some floor — both individual numbers stay
/// nonzero and plausible after a swap. `ARCGRAPH_M5_ZERO_REL_PAYLOAD_BUDGET`
/// is a field+class-asymmetric probe: it zeroes ONLY
/// `budgets.rel_bindings.payload_cap_bytes`, leaving every other field
/// (rel index cap, both node caps) untouched.
///
/// Two arms prove the seam is wired to the EXACT field/class it names:
/// - a NODE-ONLY load (no relationships at all) must still SUCCEED — proof
///   the seam left node caps (and the "no rel rows" path) completely alone;
/// - an overflow-external-id-bearing REL load must trip
///   `DiskBudgetExceeded` specifically on the rel PAYLOAD companion (not
///   the rel index, not any node companion).
///
/// RED-on-revert: swap `index_cap`/`payload_cap` at the
/// `OwnerForwardIndex::create`/`OwnerPayloadStore::create` call site (or
/// otherwise stop wiring `ARCGRAPH_M5_ZERO_REL_PAYLOAD_BUDGET` to the rel
/// payload cap specifically) — the second arm then either never fails, or
/// fails for the wrong reason (e.g. the rel INDEX budget, since a 1-byte
/// index cap would ALSO fail-closed, just on the wrong companion), and the
/// error-substring assertion pins exactly which companion refused.
#[test]
fn inv_m5_25_rel_payload_budget_is_class_and_field_scoped() {
    // Arm 1: node-only load (zero relationships) must succeed even with
    // the rel-bindings payload cap collapsed to 1 byte — the seam must
    // not leak into the node classes or the "no rels" path.
    {
        let _env = with_env(&[("ARCGRAPH_M5_ZERO_REL_PAYLOAD_BUDGET", "1")]);
        let fixture_dir = tempdir().expect("fixture dir");
        let input = fixture_dir.path().join("nodes_only.jsonl");
        let mut body = String::new();
        for index in 0..64_u64 {
            body.push_str(&node_line(format!("node-{index:06}").as_bytes(), 0, 0, &[]));
        }
        fs::write(&input, body).expect("write node-only fixture");
        let root = tempdir().expect("data dir");
        let report = load_report(root.path(), &input, ci_limits(2));
        assert_eq!(report.nodes, 64);
        assert_eq!(report.relationships, 0);
    }

    // Arm 2: a load with an overflow-external-id-bearing relationship must
    // trip DiskBudgetExceeded on the rel PAYLOAD companion specifically.
    {
        let _env = with_env(&[("ARCGRAPH_M5_ZERO_REL_PAYLOAD_BUDGET", "1")]);
        let fixture_dir = tempdir().expect("fixture dir");
        let input = fixture_dir.path().join("overflow_rel.jsonl");
        let mut body = String::new();
        body.push_str(&node_line(b"n000000", 0, 0, &[]));
        body.push_str(&node_line(b"n000001", 0, 0, &[]));
        for index in 0..8_u64 {
            // Oversized rel external ids so the rel payload companion
            // (capped at 1 byte under the seam) must take overflow writes.
            body.push_str(&rel_line(
                format!("rel-{index:06}-{}", "y".repeat(400)).as_bytes(),
                b"n000000",
                b"n000001",
                0,
                0,
                &[],
            ));
        }
        fs::write(&input, body).expect("write overflow-rel fixture");
        let root = tempdir().expect("data dir");
        let error = load_data_dir_with_limits(
            &input,
            LoadFormat::Native,
            root.path(),
            LOAD_TENANT,
            ci_limits(2),
        )
        .expect_err("1-byte rel payload cap must fail closed at the rel payload companion");
        let rendered = format!("{error:#}");
        // Field-precise: MUST be the owner PAYLOAD companion's error, not
        // the owner forward INDEX's (a cap-transposition bug that swaps
        // `index_cap`/`payload_cap` for the rel class would still raise
        // SOME `DiskBudgetExceeded`, just from the wrong companion — the
        // substring below pins exactly which one).
        assert!(
            rendered.contains("owner payload disk budget exceeded"),
            "expected the rel PAYLOAD companion's DiskBudgetExceeded specifically \
             (a cap-transposition bug would trip the INDEX companion instead), got: {rendered}"
        );
        assert!(
            rendered.contains("cap=1"),
            "expected the exact 1-byte zeroed rel payload cap, got: {rendered}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// Gate 4 — INV-M5.15 parallel regime (continuous RSS at W=16)
// ─────────────────────────────────────────────────────────────────────

/// FIX 3 (M5-D3 / #1518 skeptic review, INV-M5.15 vacuous-gate finding):
/// the collect-all mutant must trip the cap by construction, on ANY box at
/// ANY load — not merely "usually, with a thin margin." At 120k/120k
/// records the collected-under-mutant bytes cleared the
/// `baseline + 128 MiB` cap by as little as 1.6% on a loaded CI runner
/// (145.9 MB vs 143.6 MB), so a modest allocator/scheduling variance could
/// leave the RED-on-revert control silently GREEN under load. `RECORD_COUNT`
/// is sized so the mutant's collected bytes clear the cap by ≥2× headroom
/// (≈560 MB of accounted `SortItem` resident bytes vs the fixed 128 MiB
/// headroom) regardless of the harness baseline.
const RSS_FIXTURE_RECORD_COUNT: u64 = 400_000;

fn rss_fixture(dir: &Path) -> PathBuf {
    let path = dir.join("rss.jsonl");
    let file = fs::File::create(&path).expect("create rss fixture");
    let mut writer = std::io::BufWriter::with_capacity(4 << 20, file);
    // 400k nodes + 400k rels with 512-byte opaques — far above
    // W×sort_budget (16 × 256 KiB = 4 MiB), so external runs are forced by
    // construction, and far above the cap headroom (with ≥2× deterministic
    // margin — see `RSS_FIXTURE_RECORD_COUNT`) so the collect-all control
    // MUST trip it regardless of load-dependent OS RSS noise.
    let opaque = vec![0xa5_u8; 512];
    for index in 0..RSS_FIXTURE_RECORD_COUNT {
        writer
            .write_all(
                node_line(
                    format!("n{index:012}").as_bytes(),
                    (index % 5) as u32,
                    index,
                    &opaque,
                )
                .as_bytes(),
            )
            .expect("write node");
    }
    for index in 0..RSS_FIXTURE_RECORD_COUNT {
        writer
            .write_all(
                rel_line(
                    format!("r{index:012}").as_bytes(),
                    format!("n{:012}", (index * 7) % RSS_FIXTURE_RECORD_COUNT).as_bytes(),
                    format!("n{:012}", (index * 13 + 1) % RSS_FIXTURE_RECORD_COUNT).as_bytes(),
                    (index % 3) as u32,
                    index,
                    &opaque,
                )
                .as_bytes(),
            )
            .expect("write rel");
    }
    writer.flush().expect("flush rss fixture");
    path
}

/// INV-M5.15 under the PARALLEL regime: W=16 with per-worker 256 KiB
/// buffers over ~560 MB of input (`RSS_FIXTURE_RECORD_COUNT`). Every
/// continuous sample must sit below `baseline + headroom`, and the second
/// half of the run must plateau — resident never tracks input size.
///
/// RED-on-revert: `ARCGRAPH_M5_COLLECT_ALL` (the sort buffers collect the
/// whole input in RAM) → the cap assertion goes red. The fixture is sized
/// (FIX 3, M5-D3 / #1518 skeptic review) so this RED fires by construction,
/// with ≥2× headroom margin, on any box at any load — see
/// `inv_m5_15_red_on_revert_collect_all` and `RSS_FIXTURE_RECORD_COUNT`.
#[test]
fn inv_m5_15_parallel_regime_continuous_rss_w16() {
    let _serial = with_env(&[]);
    let fixture_dir = tempdir().expect("fixture dir");
    let input = rss_fixture(fixture_dir.path());
    let baseline = arcgraph_cli::m5_parallel::current_rss_bytes().expect("baseline rss");
    const HEADROOM: u64 = 128 * 1024 * 1024;
    let cap = baseline + HEADROOM;
    let root = tempdir().expect("data dir");
    let limits = LoadLimits {
        workers: Some(16),
        sort_memory_bytes: 256 * 1024,
        rss_cap_bytes: cap,
        rss_sample_every_ms: 10,
        max_disk_bytes: None,
    };
    let report = load_report(root.path(), &input, limits);
    assert_eq!(report.nodes, RSS_FIXTURE_RECORD_COUNT);
    assert_eq!(report.relationships, RSS_FIXTURE_RECORD_COUNT);
    assert!(
        report.rss_samples.len() >= 20,
        "continuous sampling must observe the whole run ({} samples)",
        report.rss_samples.len()
    );
    assert!(
        report
            .rss_samples
            .iter()
            .all(|sample| sample.rss_bytes <= cap),
        "every continuous sample within cap"
    );
    // Plateau: the second half of the run stays within one 64 MiB bucket
    // spread — resident is workload-shaped, not input-shaped.
    let tail = &report.rss_samples[report.rss_samples.len() / 2..];
    let tail_min = tail.iter().map(|sample| sample.rss_bytes).min().unwrap();
    let tail_max = tail.iter().map(|sample| sample.rss_bytes).max().unwrap();
    assert!(
        tail_max - tail_min <= 64 * 1024 * 1024,
        "continuous RSS did not plateau: tail spread {} bytes",
        tail_max - tail_min
    );
    println!(
        "M5_RSS_RESULT w=16 samples={} baseline={baseline} cap={cap} tail=[{tail_min},{tail_max}] elapsed_ms={}",
        report.rss_samples.len(),
        report.elapsed_ms
    );
}

/// RED control for INV-M5.15: the collect-all revert MUST trip the cap.
#[test]
fn inv_m5_15_red_on_revert_collect_all() {
    let _serial = with_env(&[]);
    assert_red_under(
        "inv_m5_15_parallel_regime_continuous_rss_w16",
        "ARCGRAPH_M5_COLLECT_ALL",
    );
}

// ─────────────────────────────────────────────────────────────────────
// Stage-level restartability (M5-D3 scope 3)
// ─────────────────────────────────────────────────────────────────────

/// Crash (typed) right after each stage's manifest becomes durable, then
/// rerun: the rerun resumes from the durable stage (reporting exactly the
/// skipped prefix) and the final generation is BYTE-IDENTICAL to an
/// uninterrupted load — restartability can never alter content.
#[test]
fn stage_restart_resumes_from_durable_manifests_byte_identically() {
    let _env = with_env(&[
        ("ARCGRAPH_M5_MANIFEST_TIMESTAMP", PINNED_TIMESTAMP),
        ("ARCGRAPH_CHECKPOINT_UNIX_MS", "1752624000000"),
    ]);
    let fixture_dir = tempdir().expect("fixture dir");
    let input = determinism_fixture(fixture_dir.path());

    // Uninterrupted reference.
    let reference_root = tempdir().expect("reference dir");
    load_report(reference_root.path(), &input, ci_limits(4));
    let reference_tree = tree_bytes(reference_root.path());

    let stages = [
        "s1-canonical-runs",
        "s2-phase1-counts",
        "s3-phase2-segments",
        "s4-resolved-runs",
        "s5-rel-tel-runs",
        "s6-tel-segments",
    ];
    for (index, stage) in stages.iter().enumerate() {
        let root = tempdir().expect("data dir");
        // SAFETY: under the file-wide env lock held by `_env`.
        unsafe { std::env::set_var("ARCGRAPH_M5_CRASH_AFTER_STAGE", stage) };
        let error = load_data_dir_with_limits(
            &input,
            LoadFormat::Native,
            root.path(),
            LOAD_TENANT,
            ci_limits(4),
        )
        .expect_err("injected stage crash must surface");
        // SAFETY: same lock.
        unsafe { std::env::remove_var("ARCGRAPH_M5_CRASH_AFTER_STAGE") };
        assert!(
            format!("{error:#}").contains("injected crash after stage"),
            "unexpected error: {error:#}"
        );

        let report = load_report(root.path(), &input, ci_limits(4));
        assert_eq!(
            report.resumed_stages,
            stages[..=index]
                .iter()
                .map(|stage| (*stage).to_owned())
                .collect::<Vec<_>>(),
            "crash after {stage}: rerun must resume from the durable prefix"
        );
        let tree = tree_bytes(root.path());
        assert_eq!(
            reference_tree.len(),
            tree.len(),
            "crash after {stage}: file census diverged"
        );
        for ((expected_path, expected_bytes), (path, bytes)) in
            reference_tree.iter().zip(tree.iter())
        {
            assert_eq!(expected_path, path, "crash after {stage}: tree diverged");
            assert!(
                expected_bytes == bytes,
                "crash after {stage}: {} diverged from the uninterrupted load",
                path.display()
            );
        }
    }
}

/// M5-D3 FIX 4 (#1518 skeptic review) — the restart gate above only ever
/// exercises UNDAMAGED durable manifests (the fault-injected crash always
/// leaves a well-formed prefix). A TORN/TAMPERED stage manifest (the
/// `files_ok=false` degrade arm — half-written bytes from a power cut
/// mid-`fs::write` before `sync_all`+`rename` durability) must never be
/// silently adopted: `read_stage_manifest` fails to parse it, `plan_stages`
/// drops back to the last STILL-VALID prefix (or 0 if none), and the rerun
/// REBUILDS that stage rather than trusting torn bytes. The final
/// generation must still be byte-identical to an uninterrupted load.
///
/// RED-on-revert: adopting the torn manifest (e.g. a JSON-tolerant partial
/// parse, or skipping the manifest re-read after the tear) would either
/// panic decoding stage input from a manifest that never fully described
/// its outputs, or resume past torn state and diverge from the reference
/// tree — this gate's tree-identity assertion catches either.
#[test]
fn stage_restart_refuses_a_torn_manifest_and_rebuilds() {
    let _env = with_env(&[
        ("ARCGRAPH_M5_MANIFEST_TIMESTAMP", PINNED_TIMESTAMP),
        ("ARCGRAPH_CHECKPOINT_UNIX_MS", "1752624000000"),
    ]);
    let fixture_dir = tempdir().expect("fixture dir");
    let input = determinism_fixture(fixture_dir.path());

    let reference_root = tempdir().expect("reference dir");
    load_report(reference_root.path(), &input, ci_limits(4));
    let reference_tree = tree_bytes(reference_root.path());

    // Crash right after s5's manifest becomes durable (s1-s5 all durable).
    let root = tempdir().expect("data dir");
    // SAFETY: under the file-wide env lock held by `_env`.
    unsafe { std::env::set_var("ARCGRAPH_M5_CRASH_AFTER_STAGE", "s5-rel-tel-runs") };
    let error = load_data_dir_with_limits(
        &input,
        LoadFormat::Native,
        root.path(),
        LOAD_TENANT,
        ci_limits(4),
    )
    .expect_err("injected stage crash must surface");
    // SAFETY: same lock.
    unsafe { std::env::remove_var("ARCGRAPH_M5_CRASH_AFTER_STAGE") };
    assert!(
        format!("{error:#}").contains("injected crash after stage"),
        "unexpected error: {error:#}"
    );

    // Tear the s5 manifest: truncate to half its bytes (simulated torn
    // write from a power cut mid-write, before the tmp+rename+dir-fsync
    // durability barrier landed the FULL image).
    let manifest = root
        .path()
        .join("gen-load-v6.building")
        .join("scratch/manifests/s5-rel-tel-runs.json");
    let bytes = fs::read(&manifest).expect("read s5 manifest");
    assert!(
        bytes.len() > 8,
        "s5 manifest too small to truncate meaningfully"
    );
    fs::write(&manifest, &bytes[..bytes.len() / 2]).expect("tear s5 manifest");

    // Rerun: the torn s5 manifest must fail to parse, degrading the resume
    // point to s4 (the last STILL-VALID durable stage) — s5 and s6 rebuild.
    let report = load_report(root.path(), &input, ci_limits(4));
    assert_eq!(
        report.resumed_stages,
        vec![
            "s1-canonical-runs".to_owned(),
            "s2-phase1-counts".to_owned(),
            "s3-phase2-segments".to_owned(),
            "s4-resolved-runs".to_owned(),
        ],
        "a torn s5 manifest must degrade the resume point to s4, never be adopted"
    );
    let tree = tree_bytes(root.path());
    assert_eq!(
        reference_tree.len(),
        tree.len(),
        "torn-manifest rebuild: file census diverged from the uninterrupted reference"
    );
    for ((expected_path, expected_bytes), (path, bytes)) in reference_tree.iter().zip(tree.iter()) {
        assert_eq!(
            expected_path, path,
            "torn-manifest rebuild: tree shape diverged"
        );
        assert!(
            expected_bytes == bytes,
            "torn-manifest rebuild: {} diverged from the uninterrupted load — a torn \
             manifest was silently adopted instead of triggering a rebuild",
            path.display()
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// Gate 5 — HEADLINE (100M half): REAL run evidence
// ─────────────────────────────────────────────────────────────────────

fn required_env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| {
        panic!(
            "{name} must point at the rung resource (this gate runs on the pinned box, \
             never in CI)"
        )
    })
}

/// Deterministic 100M+500M corpus generator (D-3 ratified rung shape).
/// Runs on the rung box; `ARCGRAPH_M5_RUNG_NODES`/`_RELS` shrink it for
/// harness rehearsals (a downscaled run is NEVER the 100M rung).
#[test]
#[ignore = "rung harness: generates ~110 GB at the full shape; run on the pinned box"]
fn generate_100m_rung_corpus() {
    let corpus = PathBuf::from(required_env("ARCGRAPH_M5_RUNG_CORPUS"));
    let nodes: u64 = std::env::var("ARCGRAPH_M5_RUNG_NODES")
        .ok()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(100_000_000);
    let rels: u64 = std::env::var("ARCGRAPH_M5_RUNG_RELS")
        .ok()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(500_000_000);
    let file = fs::File::create(&corpus).expect("create corpus");
    let mut writer = std::io::BufWriter::with_capacity(8 << 20, file);
    let started = std::time::Instant::now();
    let external = |prefix: u8, index: u64| {
        let mut id = [0_u8; 9];
        id[0] = prefix;
        id[1..].copy_from_slice(&index.to_be_bytes());
        id
    };
    for index in 0..nodes {
        writer
            .write_all(
                node_line(
                    &external(b'n', index),
                    (index % 11) as u32,
                    FLOAT_CORPUS[(index % FLOAT_CORPUS.len() as u64) as usize],
                    &(index as u32).to_le_bytes(),
                )
                .as_bytes(),
            )
            .expect("write node");
        if index % 10_000_000 == 0 {
            eprintln!("corpus: {index}/{nodes} nodes at {:?}", started.elapsed());
        }
    }
    for index in 0..rels {
        let source = external(b'n', (index.wrapping_mul(2_654_435_761)) % nodes);
        let target = external(b'n', (index.wrapping_mul(40_503).wrapping_add(7)) % nodes);
        writer
            .write_all(
                rel_line(
                    &external(b'r', index),
                    &source,
                    &target,
                    (index % 7) as u32,
                    FLOAT_CORPUS[(index % FLOAT_CORPUS.len() as u64) as usize],
                    &(index as u32).to_le_bytes(),
                )
                .as_bytes(),
            )
            .expect("write rel");
        if index % 50_000_000 == 0 {
            eprintln!("corpus: {index}/{rels} rels at {:?}", started.elapsed());
        }
    }
    writer.flush().expect("flush corpus");
    eprintln!(
        "corpus complete: {nodes} nodes + {rels} rels in {:?}",
        started.elapsed()
    );
}

/// Measured sequential scratch bandwidth (GB/s): timed 2 GiB direct-ish
/// write + fsync into the target filesystem, recorded next to the
/// headline so a throughput miss is attributable (amendment §4.3 / D-3).
fn measure_scratch_bandwidth(dir: &Path) -> f64 {
    let probe = dir.join(".bw-probe");
    let block = vec![0x5a_u8; 8 << 20];
    let started = std::time::Instant::now();
    let mut file = fs::File::create(&probe).expect("create probe");
    for _ in 0..256 {
        file.write_all(&block).expect("probe write");
    }
    file.sync_all().expect("probe fsync");
    let elapsed = started.elapsed().as_secs_f64();
    drop(file);
    let _ = fs::remove_file(&probe);
    (256.0 * 8.0 * 1024.0 * 1024.0) / elapsed / 1e9
}

/// **HEADLINE (100M half), amendment §M5-D3 gate table.** EXECUTED on the
/// pinned Linux box (D-3 floor: 16 physical cores, ≥2.5 GB/s sustained
/// scratch) over the ratified 100M nodes + 500M rels shape. Emits the run
/// artifact (throughput, full RSS series, measured scratch bandwidth,
/// host fingerprint) BEFORE asserting, so a miss ships its evidence.
/// An `#[ignore]` without the attached artifact does NOT satisfy this
/// gate (the V-2 evidence rule).
#[test]
#[ignore = "production rung: pinned Linux box only; artifact required — see gate doc"]
fn bounded_rss_100m_production_rung_continuous() {
    let corpus = PathBuf::from(required_env("ARCGRAPH_M5_RUNG_CORPUS"));
    let data_dir = PathBuf::from(required_env("ARCGRAPH_M5_RUNG_DATA_DIR"));
    let artifact_path = PathBuf::from(required_env("ARCGRAPH_M5_RUNG_ARTIFACT"));
    fs::create_dir_all(&data_dir).expect("create rung data dir");
    let scratch_bw_gbps = measure_scratch_bandwidth(&data_dir);

    let limits = LoadLimits::production();
    let started = std::time::Instant::now();
    let report = load_report(&data_dir, &corpus, limits);
    let elapsed_s = started.elapsed().as_secs_f64();
    let nodes_per_second = report.nodes as f64 / (report.elapsed_ms as f64 / 1000.0);
    let max_rss = report
        .rss_samples
        .iter()
        .map(|sample| sample.rss_bytes)
        .max()
        .unwrap_or(0);

    // Artifact FIRST (the evidence rule): full RSS series + throughput +
    // measured scratch bandwidth + host fingerprint.
    let host = {
        let uname = Command::new("uname").arg("-a").output();
        uname
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
            .unwrap_or_else(|_| "unknown".to_owned())
    };
    let series: Vec<String> = report
        .rss_samples
        .iter()
        .map(|sample| {
            format!(
                "[{},{},\"{}\"]",
                sample.at_ms, sample.rss_bytes, sample.stage
            )
        })
        .collect();
    let artifact = format!(
        "{{\n  \"gate\": \"bounded_rss_100m_production_rung_continuous\",\n  \
         \"shape\": {{\"nodes\": {}, \"relationships\": {}}},\n  \
         \"workers\": {},\n  \"elapsed_ms\": {},\n  \"wall_s\": {elapsed_s:.1},\n  \
         \"nodes_per_second\": {nodes_per_second:.0},\n  \
         \"rss_cap_bytes\": {M5_RSS_CAP_BYTES},\n  \"max_rss_bytes\": {max_rss},\n  \
         \"scratch_bandwidth_gbps\": {scratch_bw_gbps:.3},\n  \
         \"host\": \"{host}\",\n  \"resumed_stages\": {:?},\n  \
         \"rss_series_ms_bytes_stage\": [{}]\n}}\n",
        report.nodes,
        report.relationships,
        report.workers,
        report.elapsed_ms,
        report.resumed_stages,
        series.join(",")
    );
    fs::write(&artifact_path, &artifact).expect("write rung artifact");
    eprintln!(
        "M5_RUNG_RESULT nodes={} rels={} workers={} elapsed_ms={} nodes_per_s={nodes_per_second:.0} \
         max_rss={max_rss} scratch_gbps={scratch_bw_gbps:.3} artifact={}",
        report.nodes,
        report.relationships,
        report.workers,
        report.elapsed_ms,
        artifact_path.display()
    );

    // The D-3 rung assertions (after the artifact is durable).
    assert_eq!(report.nodes, 100_000_000, "the rung is the RATIFIED shape");
    assert_eq!(report.relationships, 500_000_000);
    assert!(
        max_rss <= M5_RSS_CAP_BYTES,
        "continuous RSS exceeded the 40 GiB rung cap: {max_rss}"
    );
    assert!(
        nodes_per_second >= 250_000.0,
        "throughput below the 250K nodes/s headline: {nodes_per_second:.0} \
         (measured scratch bandwidth {scratch_bw_gbps:.3} GB/s vs the 2.5 GB/s D-3 floor)"
    );
}

// ─────────────────────────────────────────────────────────────────────
// Projection refusal end-to-end (typed, zero substrate writes after plan)
// ─────────────────────────────────────────────────────────────────────

/// The plan-time projection refuses with the typed error + table BEFORE
/// any substrate write when the operator cap cannot fit the plan.
#[test]
fn plan_time_projection_refuses_with_typed_error() {
    let _serial = with_env(&[]);
    let fixture_dir = tempdir().expect("fixture dir");
    let input = determinism_fixture(fixture_dir.path());
    let root = tempdir().expect("data dir");
    let limits = LoadLimits {
        max_disk_bytes: Some(64 * 1024), // far below the fixture's need
        ..ci_limits(4)
    };
    let error =
        load_data_dir_with_limits(&input, LoadFormat::Native, root.path(), LOAD_TENANT, limits)
            .expect_err("64 KiB operator cap must refuse at plan time");
    let refusal = error
        .downcast_ref::<LoadRefusal>()
        .unwrap_or_else(|| panic!("typed LoadRefusal expected, got: {error:#}"));
    match refusal {
        LoadRefusal::ProjectedDiskExceeded {
            required_bytes,
            available_bytes,
            table,
            ..
        } => {
            assert!(required_bytes > available_bytes);
            assert!(table.contains("TOTAL required"));
        }
        other => panic!("expected ProjectedDiskExceeded, got {other:?}"),
    }
    // Fail-fast means no committed generation.
    assert!(!root.path().join(FINAL_GENERATION).exists());
}

/// Short unique ASCII (base-36) external ids — used only by the fixture
/// below, where minimizing `external_id_bytes` matters: the plan-time
/// projection's owner-substrate term
/// (`OwnerSubstrateBudget::projected_need_bytes`) charges per external-id
/// byte, so long formatted-decimal ids (`"n00000"`-style, used by the
/// other fixtures in this file) would inflate plan-time's OWN estimate
/// enough to mask the post-s6 densified-TEL term this test exists to pin
/// (see the doc comment below — with #1519 dense pricing, TEL is no
/// longer ~87x over budget, so the plan-time/post-s6 gap is now narrow
/// and easily swamped by an oversized external-id term).
fn b36(mut n: u64) -> Vec<u8> {
    const DIGITS: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    if n == 0 {
        return vec![b'0'];
    }
    let mut out = Vec::new();
    while n > 0 {
        out.push(DIGITS[(n % 36) as usize]);
        n /= 36;
    }
    out.reverse();
    out
}

/// Post-s6 EXACT STORE_TEL projection (the 100M STOP-report regression
/// pin, #1519 BLOCK_FIX FIX 3 re-anchor): a fixture whose distinct
/// (owner, type) group count makes the STORE_TEL cost exceed the
/// operator disk cap — but whose plan-time (census lower-bound)
/// projection FITS — must be refused with the typed error AFTER s6 and
/// BEFORE the materializer writes a byte of the store. Reverting the
/// post-s6 check turns this exact class into an hour-3 `ENOSPC` (the
/// 100M rung measured a ~3 TB out-TEL trajectory pre-#1519).
///
/// #1519 densifies STORE_TEL, so the pre-#1519 9.8 MB-vs-1 MB gap this
/// test originally pinned collapsed: the densified worst-case TEL cost
/// for N distinct-type single-edge rels is only ~13.5% above plan-time's
/// OWN total projection for the same census (verified below via the
/// SAME production arithmetic both the plan-time
/// ([`arcgraph_cli::m5_load::project_load_disk`]) and post-s6
/// ([`arcgraph_cli::m5_load::project_dense_tel_bytes_for_blocks`]) paths
/// use) — still a real, meaningful gap (post-s6 sees the EXACT
/// (owner, type) group count the pass-1 census cannot), just far
/// narrower than pre-#1519. The cap is computed as the midpoint between
/// the two projected totals so the test tracks the production arithmetic
/// instead of a hand-derived magic number that would silently go stale
/// on the next constant tweak.
#[test]
fn post_s6_exact_tel_projection_refuses_before_materialization() {
    let _serial = with_env(&[]);
    let fixture_dir = tempdir().expect("fixture dir");
    let input = fixture_dir.path().join("tel-heavy.jsonl");
    let mut body = String::new();
    const N: u64 = 6_000;
    // N nodes; N rels each with a DISTINCT type from a distinct source
    // into one shared hub target → N out-groups + N in-groups (2N
    // blocks, 2N entries, the packing-density worst case). Short base-36
    // external ids (see `b36` doc comment) keep plan-time's own
    // owner-substrate estimate from swamping the narrower post-#1519
    // plan-time/post-s6 gap. The census below is accumulated per-record
    // using the EXACT `encode_canonical_record` framing (1 B kind + one
    // 4 B length prefix per variable-length field) so `payload_bytes`
    // matches production exactly instead of an approximated formula.
    let mut census = LoadCensus::default();
    // Node record framing: 1 (kind) + 4+id_len (external_id) + 4 (label)
    // + 8 (float_bits) + 4+0 (opaque, empty).
    let node_payload_len = |id_len: usize| 1 + 4 + id_len + 4 + 8 + 4;
    // Relationship record framing: 1 (kind) + 4+id_len (external_id) +
    // 4+id_len (source) + 4+id_len (target) + 4 (type_id) + 8
    // (float_bits) + 4+0 (opaque, empty).
    let rel_payload_len = |ext_len: usize, src_len: usize, tgt_len: usize| {
        1 + 4 + ext_len + 4 + src_len + 4 + tgt_len + 4 + 8 + 4
    };
    for index in 0..N {
        let id = b36(index);
        body.push_str(&node_line(&id, 0, 0, &[]));
        census.nodes += 1;
        census.records += 1;
        census.node_external_id_bytes += id.len() as u64;
        census.payload_bytes += node_payload_len(id.len()) as u64;
    }
    let hub = b36(N);
    body.push_str(&node_line(&hub, 0, 0, &[]));
    census.nodes += 1;
    census.records += 1;
    census.node_external_id_bytes += hub.len() as u64;
    census.payload_bytes += node_payload_len(hub.len()) as u64;
    for index in 0..N {
        let rel_ext = b36(N + 1 + index);
        let src = b36(index);
        body.push_str(&rel_line(
            &rel_ext,
            &src,
            &hub,
            index as u32, // distinct type per rel → one block each
            0,
            &[],
        ));
        census.relationships += 1;
        census.records += 1;
        census.rel_external_id_bytes += rel_ext.len() as u64;
        census.payload_bytes += rel_payload_len(rel_ext.len(), src.len(), hub.len()) as u64;
    }
    fs::write(&input, body).expect("write fixture");
    let root = tempdir().expect("data dir");

    // Derive the cap from the SAME production arithmetic the loader
    // itself runs, at the two decision points: plan-time
    // (`project_load_disk`, pre-s6, a census lower bound) and post-s6
    // (`project_dense_tel_bytes_for_blocks` + the same `rest_bytes` term
    // `project_tel_or_refuse` computes). A cap strictly between the two
    // totals fits the former and exceeds the latter — REFUSED at post-s6,
    // not at plan-time.
    let budgets = plan_owner_budgets(&census);
    let plan_required = project_load_disk(&census, &budgets).required_bytes;
    let dense_bytes = arcgraph_cli::m5_load::project_dense_tel_bytes_for_blocks(2 * N, 2 * N);
    // Mirrors `project_tel_or_refuse`'s `rest_bytes` exactly: payload*3/2
    // + entries * PROJECTED_GENERATION_BYTES_PER_ENTRY (352).
    let rest_bytes = (census.payload_bytes.saturating_mul(3) / 2)
        .saturating_add((census.nodes.saturating_add(census.relationships)).saturating_mul(352));
    let post_s6_required = dense_bytes + rest_bytes;
    assert!(
        post_s6_required > plan_required,
        "test precondition: post-s6 densified total ({post_s6_required} B) must \
         exceed plan-time's total ({plan_required} B) for this to distinguish \
         the two refusal points; if this now fails, the #1519 dense-vs-plan \
         gap has closed further and this fixture needs re-deriving (memory: \
         verify production regime, not just green tests)"
    );
    let cap_bytes = plan_required + (post_s6_required - plan_required) / 2;
    assert!(
        cap_bytes >= plan_required && cap_bytes < post_s6_required,
        "derived cap {cap_bytes} must sit strictly between plan-time \
         ({plan_required}) and post-s6 ({post_s6_required})"
    );

    let limits = LoadLimits {
        max_disk_bytes: Some(cap_bytes), // plan fits, exact densified TEL does not
        ..ci_limits(4)
    };
    let error =
        load_data_dir_with_limits(&input, LoadFormat::Native, root.path(), LOAD_TENANT, limits)
            .expect_err("exact densified TEL cost must refuse before materialization");
    let refusal = error
        .downcast_ref::<LoadRefusal>()
        .unwrap_or_else(|| panic!("typed LoadRefusal expected, got: {error:#}"));
    let LoadRefusal::ProjectedDiskExceeded {
        required_bytes,
        table,
        ..
    } = refusal
    else {
        panic!("expected ProjectedDiskExceeded, got {refusal:?}");
    };
    assert!(
        table.contains("STORE_TEL") && table.contains("densified"),
        "refusal must name the densified layout cost: {table}"
    );
    assert!(
        table.contains("post-s6"),
        "must be the POST-S6 exact refusal, not the plan-time one: {table}"
    );
    assert!(
        *required_bytes > cap_bytes,
        "TEL pages must still dominate and exceed the cap"
    );
    // Refused BEFORE materialization: stage manifests durable through s6,
    // but no record/TEL extents written.
    let building = root.path().join("gen-load-v6.building");
    assert!(
        building
            .join("scratch/manifests/s6-tel-segments.json")
            .is_file(),
        "s6 manifest must be durable at refusal"
    );
    let tel_store = building.join("tenants/83/m4/tel.store");
    assert!(
        !tel_store.exists() || fs::metadata(&tel_store).map(|m| m.len()).unwrap_or(0) == 0,
        "no STORE_TEL bytes may be written after the refusal"
    );
    // The exact count is recorded in the manifest (within W−1 of the
    // brute-force expectation: N out-groups + ~N in-groups).
    let manifest: serde_json::Value = serde_json::from_slice(
        &fs::read(building.join("scratch/manifests/s6-tel-segments.json")).expect("read"),
    )
    .expect("parse");
    let pages = manifest["out_pages"].as_u64().unwrap() + manifest["in_pages"].as_u64().unwrap();
    assert!(
        (2 * N..2 * N + 10).contains(&pages),
        "exact TEL page census expected ~{} ({N} out + {N} in), got {pages}",
        2 * N
    );
}

/// #1519 BLOCK_FIX FIX 3 (vacuous-gate fix, RED-on-revert): the ORIGINAL
/// 600-node fixture above — 600 distinct-type single-edge rels into one
/// hub, whose PRE-#1519 page-per-block STORE_TEL cost was ~9.8 MB — must
/// now SUCCEED at a cap set just above the DENSIFIED need (computed via
/// the same production arithmetic,
/// [`arcgraph_cli::m5_load::project_dense_tel_bytes_for_blocks`], that
/// `project_tel_or_refuse` uses). Before this fix, `project_tel_or_refuse`
/// priced STORE_TEL at `tel_pages * 8 KiB` (~87x over the densified
/// layout for this exact fixture shape) and deterministically refused
/// this load even though the densified bytes it will actually write fit
/// comfortably — defeating #1519's own 100M/1B headline (a refusal gate
/// that fires on every production-shaped load it should ADMIT is
/// vacuous in the opposite direction from a gate that never fires).
///
/// RED-on-revert: reverting `project_tel_or_refuse` to the pre-fix
/// `tel_pages * PAGE_SIZE` pricing makes this load — which fits the
/// densified layout — refuse again.
#[test]
fn post_s6_dense_tel_projection_permits_a_load_page_per_block_would_refuse() {
    let _serial = with_env(&[]);
    let fixture_dir = tempdir().expect("fixture dir");
    let input = fixture_dir.path().join("tel-heavy-permit.jsonl");
    let mut body = String::new();
    const N: u64 = 600;
    for index in 0..N {
        body.push_str(&node_line(format!("n{index:04}").as_bytes(), 0, 0, &[]));
    }
    for index in 0..N {
        body.push_str(&rel_line(
            format!("r{index:04}").as_bytes(),
            format!("n{index:04}").as_bytes(),
            b"n0000",
            index as u32, // distinct type per rel → one block each
            0,
            &[],
        ));
    }
    fs::write(&input, body).expect("write fixture");
    let root = tempdir().expect("data dir");

    // Cap set just above the DENSIFIED need (2N blocks, 2N entries — the
    // exact packing-density worst case this fixture produces), computed
    // through the SAME production arithmetic `project_tel_or_refuse` uses
    // — not a hand-derived magic number. The pre-#1519 page-per-block
    // cost for this fixture was ~9.8 MB (2N pages x 8 KiB); this cap is
    // far below that, so a reverted (page-per-block) projection refuses.
    let dense_bytes = arcgraph_cli::m5_load::project_dense_tel_bytes_for_blocks(2 * N, 2 * N);
    let old_page_per_block_bytes = (2 * N) * arcgraph_core::PAGE_SIZE as u64;
    assert!(
        dense_bytes < old_page_per_block_bytes / 10,
        "test precondition: densified need ({dense_bytes} B) must be a small \
         fraction of the page-per-block cost ({old_page_per_block_bytes} B) \
         for this fixture to distinguish the two pricings"
    );
    // Leave headroom above the exact TEL need for the records/props/owner
    // rows the same generation also writes ("rest_bytes" in
    // `project_tel_or_refuse`).
    let cap_bytes = dense_bytes + 8 * 1024 * 1024;
    assert!(
        cap_bytes < old_page_per_block_bytes,
        "test precondition: the permit cap must still be BELOW the pre-#1519 \
         page-per-block cost, or a reverted projection would not refuse \
         (RED-on-revert would be vacuous)"
    );
    let limits = LoadLimits {
        max_disk_bytes: Some(cap_bytes),
        ..ci_limits(4)
    };
    let outcome =
        load_data_dir_with_limits(&input, LoadFormat::Native, root.path(), LOAD_TENANT, limits)
            .unwrap_or_else(|error| {
                panic!(
                    "densified STORE_TEL fits the cap ({cap_bytes} B >= dense need \
                     {dense_bytes} B) and must be ADMITTED, not refused: {error:#}"
                )
            });
    let LoadOutcome::Loaded(report) = outcome else {
        panic!("expected a committed load, got {outcome:?}");
    };
    assert_eq!(report.nodes, N);
    assert_eq!(report.relationships, N);
    assert!(
        root.path().join(FINAL_GENERATION).is_dir(),
        "committed generation must exist after a successful densified load"
    );
}

/// #1519 pure arithmetic (mirrors INV-M5.25's projection-only style, no
/// load, no I/O): the DENSIFIED STORE_TEL byte projection at the 100M and
/// 1B rungs must land well inside the box budget the pre-#1519
/// page-per-block layout blew through (5.9 TB @ 100M, 16 TB @ 1B against
/// an 8 TB box, per the M5-D3 100M-rung STOP-report). `project_dense_tel_bytes`
/// assumes the degenerate worst case for packing (every block distinct-type,
/// 1 entry, maximum directory overhead per block) — the SAME worst case the
/// pre-#1519 page-per-block projection used — so this is an apples-to-apples
/// before/after comparison, not a favorable-case cherry-pick.
#[test]
fn project_dense_tel_bytes_fits_100m_and_1b_budget() {
    use arcgraph_cli::m5_load::project_dense_tel_bytes;

    const TIB: u64 = 1024 * 1024 * 1024 * 1024;
    const BOX_BUDGET_BYTES: u64 = 8 * TIB;

    for (label, relationships, old_page_per_block_tib) in [
        ("100M rung", 500_000_000_u64, 7.4506_f64),
        ("1B rung", 5_000_000_000_u64, 74.506_f64),
    ] {
        let dense_bytes = project_dense_tel_bytes(relationships);
        let old_bytes = relationships
            .saturating_mul(2)
            .saturating_mul(arcgraph_core::PAGE_SIZE as u64);
        // Sanity: the "old" figure we compare against matches the
        // pre-#1519 STOP-report's own page-per-block arithmetic (2
        // directions x relationships x one 8 KiB page per block, the
        // worst-case-density fixture this projection also assumes).
        let old_tib = old_bytes as f64 / TIB as f64;
        assert!(
            (old_tib - old_page_per_block_tib).abs() < 0.01,
            "{label}: page-per-block baseline arithmetic drifted: {old_tib} vs {old_page_per_block_tib}"
        );
        assert!(
            dense_bytes < old_bytes,
            "{label}: densified projection ({dense_bytes} B) must be strictly \
             smaller than the page-per-block baseline ({old_bytes} B)"
        );
        // The headline claim: densified STORE_TEL fits comfortably inside
        // the 8 TiB box the page-per-block layout blew through.
        assert!(
            dense_bytes < BOX_BUDGET_BYTES,
            "{label}: densified STORE_TEL projection ({dense_bytes} B) must fit \
             the box budget ({BOX_BUDGET_BYTES} B) — pre-#1519 page-per-block \
             measured 5.9 TB @ 100M / 16 TB @ 1B against this same budget"
        );
        // Density factor: packing must recover close to two orders of
        // magnitude even in the worst (all-1-entry, all-distinct-type)
        // case — the actual fixture-measured factor (`tel_disk_size_is_dense`,
        // `m5_served_store_gate.rs`) is on real mixed-degree data, not this
        // degenerate worst case, so it can differ; this pins the WORST-CASE
        // floor the arithmetic guarantees.
        let factor = old_bytes as f64 / dense_bytes as f64;
        assert!(
            factor > 50.0,
            "{label}: densified worst-case factor {factor:.1}x must exceed 50x"
        );
    }
}

// Keep the 1B rung OUT of this slice: it is M5-E, co-gated with M6 entry
// (Director decision D-2). No `throughput_1b_*` test exists here by design.
const _RUNG_1B_IS_M5E: () = ();

#[allow(dead_code)]
fn gib(bytes: u64) -> f64 {
    bytes as f64 / GIB as f64
}
