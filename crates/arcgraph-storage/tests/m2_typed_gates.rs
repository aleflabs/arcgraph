//! v2 M2 — the typed-block EXIT gates at the storage grain
//! (build-plan §2 M2 EXIT 1; ADR-230 row M2; design §M2.6):
//!
//! - **G1 headline** — batch-ingest WAL-B/node on the TYPED write
//!   path, measured exactly like the M1 gate
//!   (`m1_slotted_packing.rs`), with the single-record honesty
//!   control (still ~a full page image — M3's delta-WAL leg, not
//!   M2's) and the M1-parity assertion (typed blocks must not
//!   REGRESS the M1 batch number).
//! - **G1 payload-size leg** — the typed REPRESENTATION of the
//!   incident-shape bag is ≤ the JSON encoding of the same bag (the
//!   per-payload term M2 owns under page-image WAL), and the whole
//!   block is ≤ JSON + the 4-byte CRC-32C meta_check field (the
//!   round-3 #1452 integrity premium, pinned at exactly the field
//!   width).
//! - **RED-on-revert** — the same forced-chained lever the M1 gate
//!   arms (`ARCGRAPH_M1_FORCE_CHAINED_BAGS=1` routes EVERY staged
//!   payload through dedicated chains) blows the batch measurement
//!   past 4,000 B/node in a subprocess, proving the measurement
//!   detects a coalescing revert on the typed path too.
//!
//! # The ≤ ~250 headline number (measured honestly)
//!
//! ADR-230's M2 row carries "WAL →~250 B/node"; the number originates
//! in the 3x-doc §2.4 arithmetic for **W-B2 + W-E combined** (typed
//! blocks + the DELTA WAL — storage-architecture-v2 §2.2 cites
//! exactly that pairing). Under M2 the WAL is still page-image
//! (`wal_format: page-image`; the delta WAL is M3), so the
//! bag-payload term M2 shrinks (~60 B JSON → ~52 B typed) is a small
//! share of the measured per-node bytes, which are dominated by the
//! shared page images + record + framing shares that M3's deltas
//! remove. This gate therefore pins:
//!   (a) the MEASURED number, printed on every run (the PR body
//!       carries it),
//!   (b) no-regression vs the M1 ceiling (≤ 700, the M1 gate's line,
//!       expected ≈ M1's ~332),
//!   (c) the payload-size win M2 actually owns.
//! If (a) lands ≤ 250 the stricter line is pinned by the assertion in
//! (b) being tightened — see the gate body. The ~250 all-in line is
//! M3's GATE-5 (delta WAL), where the dominant terms die.

use std::sync::Arc;

use arcgraph_core::{LabelId, TenantId};
use arcgraph_storage::crud::{CrudStore, PropertyData, commit, create_node};
use arcgraph_storage::intern::InternTable;
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::PrimaryIndex;
use arcgraph_storage::prop_block::{PropBlockBuilder, PropValue};
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{WalConfig, WalWriter};
use tempfile::TempDir;

// ─── Harness — the M1 gate's stack, verbatim (m1_slotted_packing.rs)
//     so the two headline measurements are apples-to-apples ──────────

fn test_wal_config(dir: &std::path::Path) -> WalConfig {
    WalConfig {
        dir: dir.to_path_buf(),
        segment_size_bytes: 64 * 1024 * 1024,
        group_commit_window: std::time::Duration::from_millis(2),
        group_commit_max_batch: 8,
        metrics_sink: None,
        encryption: None,
        inflight_budget_bytes: None,
    }
}

fn build_stack(
    wal_dir: &std::path::Path,
) -> (
    WalWriter,
    Arc<TxnManager>,
    Arc<PrimaryIndex>,
    Arc<CrudStore>,
) {
    let writer = WalWriter::spawn(test_wal_config(wal_dir)).unwrap();
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

fn wal_dir_bytes(wal_dir: &std::path::Path) -> u64 {
    std::fs::read_dir(wal_dir)
        .expect("read wal dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_file())
        .map(|e| e.metadata().map(|m| m.len()).unwrap_or(0))
        .sum()
}

/// The incident-shape bag as (name, value) pairs — the SAME logical
/// bag as the M1 gate's `small_bag` JSON.
fn small_bag_pairs(i: u32) -> Vec<(String, PropValue)> {
    vec![
        ("svc".to_string(), PropValue::Str(format!("api-{i:04}"))),
        ("sev".to_string(), PropValue::Str(format!("P{}", i % 4))),
        (
            "region".to_string(),
            PropValue::Str("us-east-1".to_string()),
        ),
    ]
}

/// The M1 gate's JSON form of the same bag (the payload-size
/// comparison basis).
fn small_bag_json(i: u32) -> Vec<u8> {
    format!(
        r#"{{"svc":"api-{i:04}","sev":"P{}","region":"us-east-1"}}"#,
        i % 4
    )
    .into_bytes()
}

/// Build the typed payload for node `i` against `intern` (the storage
/// grain of what the mcp write bridge produces).
fn typed_bag(i: u32, intern: &InternTable) -> PropertyData {
    let mut b = PropBlockBuilder::new();
    for (name, value) in small_bag_pairs(i) {
        b.put(
            intern.intern(TenantId::DEFAULT, &name).unwrap().raw(),
            value,
        );
    }
    let enc = b.build().expect("encode");
    assert!(
        enc.overflow_payload().is_none(),
        "incident-shape bag must not spill"
    );
    let block = enc.into_block_bytes(None).expect("finalize");
    PropertyData::TypedBlock(arcgraph_storage::prop_block::TypedBagParts {
        block,
        overflow: None,
    })
}

/// Batch-ingest N nodes with TYPED bags in ONE transaction; measured
/// WAL bytes per node (the M1 gate's measurement, typed payloads).
fn measure_batch_wal_bytes_per_node_typed(n: u32) -> u64 {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();
    let (writer, mgr, _primary, store) = build_stack(&wal_dir);
    let intern = InternTable::new();

    let mut tx = mgr.begin(TenantId::DEFAULT);
    for i in 0..n {
        create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(1),
            &typed_bag(i, &intern),
        )
        .unwrap();
    }
    commit(tx, &store).unwrap();
    writer.shutdown().unwrap();

    wal_dir_bytes(&wal_dir) / u64::from(n)
}

// ─── G1 — the headline + honesty + parity ────────────────────────────

/// The M2 batch WAL measurement: printed on every run (the PR body
/// carries the number), pinned at the M1 gate's ceiling (typed blocks
/// must not regress the batch amortization), with the payload-size
/// win asserted separately below. See the module docs for why the
/// ~250 all-in line belongs to M3's delta WAL.
#[test]
fn m2_headline_batch_wal_bytes_per_node_measured_and_no_regression() {
    let per_node = measure_batch_wal_bytes_per_node_typed(200);
    eprintln!("m2 headline: batch WAL bytes/node (typed) = {per_node}");
    assert!(
        per_node <= 700,
        "typed batch-ingest WAL must stay within the M1 gate's ceiling \
         (measured {per_node}; M1 measured ~332; pre-M1 basis 8,454) — \
         is the once-per-bundle slotted coalescing intact on the typed path?"
    );
}

/// The payload term M2 owns under page-image WAL: the typed
/// REPRESENTATION of the incident-shape bag is no larger than its
/// JSON form, and the whole block exceeds JSON by AT MOST the 4-byte
/// metadata-checksum field (and the zero-decode read it buys is gated
/// in `m2_zero_decode` / `property_payload`).
///
/// Round-3 (#1452, Director ruling): the per-block metadata checksum
/// widened CRC-8 → CRC-32C after the 8-bit width was measured too
/// narrow (2,378 of 2,096,128 two-bit key upsets cleared every
/// guard). The CRC-32C field is 4 bytes JSON simply does not carry —
/// the ruled integrity premium — so this gate pins BOTH terms
/// honestly instead of silently relaxing: the representation win
/// (block minus the checksum field ≤ JSON) stays, and the premium is
/// pinned at EXACTLY the field width so it cannot creep.
#[test]
fn m2_typed_block_payload_no_larger_than_json() {
    /// The metadata-checksum field width (block header bytes 4..8) —
    /// the only block-size term the round-3 widen added.
    const META_CHECK_PREMIUM: usize = 4;
    let intern = InternTable::new();
    for i in [0u32, 7, 199] {
        let PropertyData::TypedBlock(parts) = typed_bag(i, &intern) else {
            panic!("typed bag must be a TypedBlock");
        };
        let json = small_bag_json(i);
        eprintln!(
            "m2 payload sizes: typed = {} B ({} B sans meta_check), json = {} B (bag {i})",
            parts.block.len(),
            parts.block.len() - META_CHECK_PREMIUM,
            json.len()
        );
        assert!(
            parts.block.len() - META_CHECK_PREMIUM <= json.len(),
            "typed representation ({} B sans the {META_CHECK_PREMIUM}-B meta_check) must \
             not exceed the JSON form ({} B) for bag {i}",
            parts.block.len() - META_CHECK_PREMIUM,
            json.len()
        );
        assert!(
            parts.block.len() <= json.len() + META_CHECK_PREMIUM,
            "typed block ({} B) must not exceed the JSON form ({} B) by more than the \
             {META_CHECK_PREMIUM}-B checksum field for bag {i} — the round-3 premium \
             must not creep",
            parts.block.len(),
            json.len()
        );
    }
}

/// ADR-230's M1 honesty line holds at M2 verbatim: a lone
/// single-record auto-commit still stages the whole (one-bag) slotted
/// page image (~≥ 8 KiB per commit). No amortization until M3's delta
/// WAL — state the number, don't improve it (the M2 charter's exact
/// words).
#[test]
fn m2_honesty_single_record_commit_still_pays_a_page_image() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();
    let (writer, mgr, _primary, store) = build_stack(&wal_dir);
    let intern = InternTable::new();

    const N: u32 = 16;
    for i in 0..N {
        let mut tx = mgr.begin(TenantId::DEFAULT);
        create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(1),
            &typed_bag(i, &intern),
        )
        .unwrap();
        commit(tx, &store).unwrap();
    }
    writer.shutdown().unwrap();

    let per_commit = wal_dir_bytes(&wal_dir) / u64::from(N);
    eprintln!("m2 honesty: single-record WAL bytes/commit (typed) = {per_commit}");
    assert!(
        per_commit >= 8192,
        "single-record commits must still pay ≥ one full 8 KiB page image (measured \
         {per_commit}) — M2 is a payload re-typing; record-level deltas are M3's leg"
    );
}

// ─── RED-on-revert (subprocess, always armed) ────────────────────────

/// Helper invoked ONLY by the RED test below in a subprocess with the
/// forced-chained lever set (every staged payload — typed blocks
/// included — routes through dedicated DEC-4 chains, reverting the
/// once-per-bundle coalescing).
#[test]
#[ignore = "subprocess helper (forced-chained RED probe) — spawned by the RED gate"]
fn helper_m2_forced_chained_batch_probe() {
    assert_eq!(
        std::env::var("ARCGRAPH_M1_FORCE_CHAINED_BAGS").as_deref(),
        Ok("1"),
        "RED probe must run under the forced-chained lever"
    );
    let per_node = measure_batch_wal_bytes_per_node_typed(200);
    eprintln!("m2 RED probe: forced-chained typed batch WAL bytes/node = {per_node}");
    assert!(
        per_node >= 4_000,
        "forced-chained typed batch must reproduce the pre-M1 WAL blowup (measured {per_node})"
    );
}

/// RED-on-revert (build-plan §2 M2 EXIT 5's measurement-sensitivity
/// leg): reverting the once-per-bundle coalescing blows the SAME
/// measurement the headline pins past 4,000 B/node — proven live on
/// every run via the subprocess with the revert lever engaged.
#[test]
fn m2_red_on_revert_forced_chaining_blows_the_headline() {
    let exe = std::env::current_exe().expect("test binary");
    let out = std::process::Command::new(exe)
        .args([
            "--exact",
            "helper_m2_forced_chained_batch_probe",
            "--ignored",
            "--nocapture",
        ])
        .env("ARCGRAPH_M1_FORCE_CHAINED_BAGS", "1")
        .output()
        .expect("spawn RED probe");
    assert!(
        out.status.success(),
        "forced-chained probe must PASS its ≥4000 B/node assertion (the RED proof).\n\
         stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}
