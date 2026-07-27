//! v2 M1 — W-B1 slotted small-blob packing: the EXIT gates
//! (ADR-230 M1 row; `docs/design/v2-build-plan.md` §2 M1 EXIT;
//! `docs/design/m1-m2-m4-m5-impl-designs.md` §M1.3/§M1.5/§M1.6;
//! tracking #1430).
//!
//! Covers, at the real WAL + CRUD stack:
//!
//! 1. **The headline metric** — batch-ingest WAL-B/node ≤ ~600 (vs the
//!    measured pre-M1 8,454; `docs/perf/beyond-neo4j-3x-strategy.md`),
//!    AND the honesty control: single-record commits still pay a full
//!    page image (~8 KiB+) per commit — M1 is packing, not the M3
//!    delta WAL (the ADR-230 M1 honesty line).
//! 2. **RED-on-revert, always-armed** — a subprocess re-runs the batch
//!    probe with `ARCGRAPH_M1_FORCE_CHAINED_BAGS=1` (the exact
//!    pre-M1 one-dedicated-chain-per-bag behavior = the coalescing
//!    revert) and asserts the SAME measurement blows past the
//!    threshold — proving the headline gate detects the revert.
//! 3. **Recovery byte-equality** on a mixed small(slotted)+large
//!    (chained) store — post-replay every bag reads byte-identical
//!    (the §0.3 oracle at the M1 format).
//! 4. **RULE-MT** — ≥ 8 concurrent writer threads (the production
//!    regime) staging bags through real commits, with the checkpoint
//!    capture + the bounded tier's drain racing them; post-recovery
//!    every bag byte-identical (build-plan §0 RULE-MT: gates on a
//!    format path MUST run the concurrent regime).
//!
//! The pack/unpack proptest (EXIT 2) lives with the codec
//! (`records.rs::tests::prop_bags`) + the store lifecycle proptest
//! (`blob.rs::tests`); the migrate-on-open + kill-9 crash gate (EXIT
//! 4) drives the production bootstrap seam in
//! `crates/arcgraph-cli/tests/m1_migrate_on_open_1430.rs`.

use std::process::Command;
use std::sync::Arc;

use arcgraph_core::{LabelId, TenantId};
use arcgraph_storage::blob::BLOB_CHUNK_BYTES;
use arcgraph_storage::crud::{
    CrudStore, PropertyData, commit, create_node, crud_allocator_seed_handle, read_node_with_store,
};
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::PrimaryIndex;
use arcgraph_storage::property::{BlobRef, OVERFLOW_SLOT_MASK};
use arcgraph_storage::records::PROP_BAG_MAX_BYTES;
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{
    AllocatorSeedHandle, BlobStoreHandle, PageStoreTarget, PrimaryPageStoreHandle,
    RecordPageStoreHandle, WalConfig, WalWriter, recover_from_wal,
};
use tempfile::TempDir;

// ─── Harness (mirrors tests/wal_replay_round_trip.rs) ───────────────

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

fn recover_stack(
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
    let primary_handle: Arc<dyn PrimaryPageStoreHandle> =
        Arc::clone(primary.page_store()) as Arc<dyn PrimaryPageStoreHandle>;
    let records_handle: Arc<dyn RecordPageStoreHandle> = Arc::clone(
        store
            .records()
            .expect("dual-write stack has a record store"),
    ) as Arc<dyn RecordPageStoreHandle>;
    let blob_handle: Arc<dyn BlobStoreHandle> =
        Arc::clone(store.blob_store()) as Arc<dyn BlobStoreHandle>;
    let allocator_seed: Arc<dyn AllocatorSeedHandle> =
        crud_allocator_seed_handle(Arc::clone(&store), Arc::clone(&alloc));
    let target = PageStoreTarget::primary_only(primary_handle)
        .with_record_store(records_handle)
        .with_blob_store(blob_handle)
        .with_allocator_seed(allocator_seed);
    recover_from_wal(wal_dir, Arc::clone(&mgr), target, None).expect("recovery");
    (writer, mgr, primary, store)
}

/// Total on-disk WAL bytes (sum of segment file sizes).
fn wal_dir_bytes(wal_dir: &std::path::Path) -> u64 {
    std::fs::read_dir(wal_dir)
        .expect("read wal dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_file())
        .map(|e| e.metadata().map(|m| m.len()).unwrap_or(0))
        .sum()
}

/// A representative incident-shape small bag (~60 B of JSON bytes —
/// the payload M1 packs; content is opaque to the packer).
fn small_bag(i: u32) -> Vec<u8> {
    format!(
        r#"{{"svc":"api-{i:04}","sev":"P{}","region":"us-east-1"}}"#,
        i % 4
    )
    .into_bytes()
}

/// Batch-ingest N nodes with small bags in ONE transaction and return
/// measured WAL bytes per node. Shared by the headline gate and the
/// forced-chained RED subprocess.
fn measure_batch_wal_bytes_per_node(n: u32) -> u64 {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();
    let (writer, mgr, _primary, store) = build_stack(&wal_dir);

    let mut tx = mgr.begin(TenantId::DEFAULT);
    for i in 0..n {
        create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(1),
            &PropertyData::Blob(small_bag(i)),
        )
        .unwrap();
    }
    commit(tx, &store).unwrap();
    writer.shutdown().unwrap();

    wal_dir_bytes(&wal_dir) / u64::from(n)
}

// ─── EXIT 1 — the headline + the honesty control ─────────────────────

/// Batch-ingest WAL-B/node ≤ ~600 (design §M1.5: ~64.5 B amortized
/// page image per 60 B bag + 64 B record + framing/index shares). The
/// 700 ceiling leaves framing variance while still pinning ≥ 12× vs
/// the 8,454 pre-M1 basis; the RED subprocess (below) proves this same
/// measurement blows past 4,000 when packing is reverted.
#[test]
fn m1_headline_batch_wal_bytes_per_node_under_600ish() {
    let per_node = measure_batch_wal_bytes_per_node(200);
    eprintln!("m1 headline: batch WAL bytes/node = {per_node}");
    assert!(
        per_node <= 700,
        "batch-ingest WAL must be ≤ ~600 B/node (measured {per_node}; pre-M1 basis 8,454 — \
         is the once-per-bundle slotted coalescing intact?)"
    );
}

/// The ADR-230 M1 honesty line: a lone single-record auto-commit still
/// stages the whole (one-bag) slotted page image → pays ≥ a full
/// 8 KiB page image per commit. No amortization until M3's delta WAL.
/// This gate FAILING on a big improvement is the signal someone built
/// record-level deltas — which belongs to M3, not M1.
#[test]
fn m1_honesty_single_record_commit_still_pays_a_page_image() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();
    let (writer, mgr, _primary, store) = build_stack(&wal_dir);

    const N: u32 = 16;
    for i in 0..N {
        let mut tx = mgr.begin(TenantId::DEFAULT);
        create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(1),
            &PropertyData::Blob(small_bag(i)),
        )
        .unwrap();
        commit(tx, &store).unwrap();
    }
    writer.shutdown().unwrap();

    let per_commit = wal_dir_bytes(&wal_dir) / u64::from(N);
    eprintln!("m1 honesty: single-record WAL bytes/commit = {per_commit}");
    assert!(
        per_commit >= 8192,
        "single-record commits must still pay ≥ one full 8 KiB page image (measured \
         {per_commit}) — M1 is packing only; record-level deltas are M3's leg"
    );
}

// ─── EXIT 5 — RED-on-revert, always-armed via subprocess ─────────────

/// Helper invoked ONLY by the RED test below, in a subprocess with
/// `ARCGRAPH_M1_FORCE_CHAINED_BAGS=1` (the coalescing revert). Ignored
/// in normal runs.
#[test]
#[ignore = "subprocess helper for m1_red_on_revert_forced_chaining_blows_the_headline"]
fn helper_forced_chained_batch_probe() {
    assert_eq!(
        std::env::var("ARCGRAPH_M1_FORCE_CHAINED_BAGS").as_deref(),
        Ok("1"),
        "helper must run under the forced-chained env (spawned by the RED test)"
    );
    let per_node = measure_batch_wal_bytes_per_node(200);
    eprintln!("m1 RED probe: forced-chained batch WAL bytes/node = {per_node}");
    // The pre-M1 basis is 8,454; anything ≥ 4,000 proves the dedicated
    // 8 KiB-page-per-bag regime is back and the ≤ ~600 headline gate
    // would fail — i.e. the gate DETECTS the revert.
    assert!(
        per_node >= 4_000,
        "forced-chained batch must reproduce the pre-M1 WAL blowup (measured {per_node})"
    );
}

/// Build-plan §2 M1 EXIT 5: revert the "stage the slotted page once
/// per bundle" coalescing → batch WAL-B/node jumps back toward 8,454 →
/// the headline assertion fails. Proven live on every run by
/// re-executing the batch probe in a subprocess with the revert lever
/// engaged and asserting the measurement blows past the threshold.
#[test]
fn m1_red_on_revert_forced_chaining_blows_the_headline() {
    let exe = std::env::current_exe().expect("test binary path");
    let out = Command::new(exe)
        .args([
            "--exact",
            "helper_forced_chained_batch_probe",
            "--ignored",
            "--nocapture",
        ])
        .env("ARCGRAPH_M1_FORCE_CHAINED_BAGS", "1")
        .output()
        .expect("spawn forced-chained probe subprocess");
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "forced-chained probe must PASS its ≥4000 B/node assertion (the RED proof).\n\
         stdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stderr.contains("forced-chained batch WAL bytes/node") || stdout.contains("m1 RED probe"),
        "probe must actually have run.\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
}

// ─── EXIT 3/6 — recovery byte-equality on a mixed store ──────────────

/// Small (slotted) + large (chained) bags in the same transactions →
/// process-restart-equivalent recovery → every bag byte-identical and
/// carried by the expected representation (slot ≥ 1 vs slot 0). The
/// M1 §0.3 consistency anchor at the real WAL.
#[test]
fn m1_recovery_byte_equality_mixed_slotted_and_chained() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();
    let (writer, mgr, primary, store) = build_stack(&wal_dir);

    let mut expected = Vec::new();
    for batch in 0..4u32 {
        let mut tx = mgr.begin(TenantId::DEFAULT);
        for i in 0..25u32 {
            let n = batch * 100 + i;
            // Mix: small bags (slotted), boundary bags, multi-chunk
            // chained bags.
            let payload = match i % 5 {
                0 => vec![0xA0 | (n & 0xF) as u8; 60],
                1 => small_bag(n),
                2 => vec![(n & 0xFF) as u8; PROP_BAG_MAX_BYTES], // max slotted
                3 => vec![(n & 0xFF) as u8; PROP_BAG_MAX_BYTES + 1], // min chained
                _ => vec![(n & 0xFF) as u8; BLOB_CHUNK_BYTES * 2 + 17], // 3-page chain
            };
            let id = create_node(
                &store,
                &mut tx,
                TenantId::DEFAULT,
                LabelId::new(3),
                &PropertyData::Blob(payload.clone()),
            )
            .unwrap();
            expected.push((id, payload));
        }
        commit(tx, &store).unwrap();
    }
    writer.shutdown().unwrap();
    drop((store, primary, mgr));

    let (writer2, mgr2, _primary2, store2) = recover_stack(&wal_dir);
    let tx2 = mgr2.begin(TenantId::DEFAULT);
    for (id, payload) in &expected {
        let rec = read_node_with_store(&store2, &tx2, *id)
            .unwrap()
            .unwrap_or_else(|| panic!("node {id:?} must be readable post-replay"));
        let bref = BlobRef::decode(rec.property_ref)
            .unwrap_or_else(|| panic!("node {id:?} must carry an overflow ref"));
        // Representation matches the size class.
        if payload.len() <= PROP_BAG_MAX_BYTES {
            assert!(
                u64::from(bref.slot_id) >= 1 && u64::from(bref.slot_id) <= OVERFLOW_SLOT_MASK,
                "small bag must be slotted post-replay (node {id:?})"
            );
        } else {
            assert_eq!(bref.slot_id, 0, "large bag must stay chained (node {id:?})");
        }
        let got = store2.blob_store().get(TenantId::DEFAULT, bref).unwrap();
        assert_eq!(
            got.as_ref(),
            payload.as_slice(),
            "bag must round-trip byte-identical across recovery (node {id:?})"
        );
    }
    writer2.shutdown().unwrap();
}

// ─── RULE-MT — the ≥8-writer concurrent regime ───────────────────────

/// Build-plan §0 RULE-MT: the format gate under the concurrent regime
/// prod runs by default. 8 writer threads race real create+commit
/// cycles (contending for the tenant's open-page pool) while a 9th
/// thread hammers the checkpoint capture (`iter_pages_resident_only` —
/// the freeze-side reader that also flips INV-DURABLE bits) and the
/// blob drain. Then a full recovery replays the interleaved bundles.
/// PASS = every acked bag reads byte-identical both live and
/// post-recovery, and no two bags alias a (page, slot).
#[test]
fn m1_rule_mt_concurrent_writers_vs_capture_then_recovery() {
    const WRITERS: u64 = 8;
    const TXNS_PER_WRITER: u64 = 12;
    const BAGS_PER_TXN: u64 = 8;

    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();
    let (writer, mgr, primary, store) = build_stack(&wal_dir);

    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    // Capture-side racer: the exact freeze-side iterator the ADR-229
    // checkpoint producer runs, interleaved with writer commits.
    let capture_thread = {
        let store = Arc::clone(&store);
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || {
            let mut captures = 0u64;
            while !stop.load(std::sync::atomic::Ordering::Acquire) {
                let (resident, _evicted) = store.blob_store().iter_pages_resident_only();
                captures += 1;
                drop(resident);
                std::thread::yield_now();
            }
            captures
        })
    };

    let mut handles = Vec::new();
    for w in 0..WRITERS {
        let mgr = Arc::clone(&mgr);
        let store = Arc::clone(&store);
        handles.push(std::thread::spawn(move || {
            let mut acked: Vec<(arcgraph_core::NodeId, Vec<u8>)> = Vec::new();
            for t in 0..TXNS_PER_WRITER {
                let mut tx = mgr.begin(TenantId::DEFAULT);
                let mut staged = Vec::new();
                for b in 0..BAGS_PER_TXN {
                    let n = (w * TXNS_PER_WRITER + t) * BAGS_PER_TXN + b;
                    let payload =
                        format!(r#"{{"w":{w},"t":{t},"b":{b},"n":{n},"pad":"xxxxxxxxxxxxxxxx"}}"#)
                            .into_bytes();
                    let id = create_node(
                        &store,
                        &mut tx,
                        TenantId::DEFAULT,
                        LabelId::new(2),
                        &PropertyData::Blob(payload.clone()),
                    )
                    .unwrap();
                    staged.push((id, payload));
                }
                commit(tx, &store).unwrap();
                acked.extend(staged);
            }
            acked
        }));
    }
    let acked: Vec<_> = handles
        .into_iter()
        .flat_map(|h| h.join().expect("writer thread"))
        .collect();
    stop.store(true, std::sync::atomic::Ordering::Release);
    let captures = capture_thread.join().expect("capture thread");
    assert!(captures > 0, "the capture racer must actually have run");
    assert_eq!(
        acked.len() as u64,
        WRITERS * TXNS_PER_WRITER * BAGS_PER_TXN,
        "every commit acked"
    );

    // Live reads: byte-identical + no (page, slot) aliasing.
    let tx = mgr.begin(TenantId::DEFAULT);
    let mut seen = std::collections::HashSet::new();
    for (id, payload) in &acked {
        let rec = read_node_with_store(&store, &tx, *id).unwrap().unwrap();
        let bref = BlobRef::decode(rec.property_ref).unwrap();
        assert!(
            bref.slot_id >= 1,
            "small bags pack slotted under concurrency"
        );
        assert!(
            seen.insert((bref.page_id, bref.slot_id)),
            "no two bags may alias one (page, slot) under concurrency"
        );
        let got = store.blob_store().get(TenantId::DEFAULT, bref).unwrap();
        assert_eq!(got.as_ref(), payload.as_slice(), "live read (node {id:?})");
    }
    drop(tx);
    writer.shutdown().unwrap();
    drop((store, primary, mgr));

    // The interleaved bundles replay to the same visible bags.
    let (writer2, mgr2, _primary2, store2) = recover_stack(&wal_dir);
    let tx2 = mgr2.begin(TenantId::DEFAULT);
    for (id, payload) in &acked {
        let rec = read_node_with_store(&store2, &tx2, *id)
            .unwrap()
            .unwrap_or_else(|| panic!("node {id:?} must survive recovery (RULE-MT)"));
        let bref = BlobRef::decode(rec.property_ref).unwrap();
        let got = store2.blob_store().get(TenantId::DEFAULT, bref).unwrap();
        assert_eq!(
            got.as_ref(),
            payload.as_slice(),
            "post-recovery byte-equality under the concurrent regime (node {id:?})"
        );
    }
    writer2.shutdown().unwrap();
}
