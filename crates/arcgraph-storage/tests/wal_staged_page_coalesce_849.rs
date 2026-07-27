//! #849 B3(a) regression — WAL staged-page coalescing is durable +
//! lossless, and collapses the per-record full-page write amplification.
//!
//! ROOT CAUSE (measured in `tests/durable_ingest_throughput_849.rs`):
//! `install_create` (`crud.rs::snapshot_record_page`) and the primary
//! index's `upsert_deferred` stage a FULL `PAGE_SIZE` page snapshot per
//! record / key mutation. A B-record commit whose records pack onto a
//! handful of slotted pages previously staged each touched page B times
//! — `B × 8 KiB` of WAL for `~8 KiB` of final state (~16.7 KiB of WAL
//! PER RECORD, constant across batch sizes 1 → 1000; the #849 finding's
//! 2.5 GiB-for-140 K-records observation). The fix
//! (`TxnManager::coalesce_staged_pages`) keeps only the LAST snapshot
//! per `(kind, page_id)` in the bundle (last-write-wins within the
//! atomic commit), so WAL volume scales with PAGES TOUCHED not RECORDS
//! WRITTEN.
//!
//! These two tests are the strong oracles:
//!
//! 1. `coalesced_multi_record_commit_replays_losslessly_849` — proves
//!    coalescing is LOSSLESS: 600 distinct nodes committed in ONE tx
//!    (heavily sharing record pages + index leaves) all read back with
//!    their exact label + inline payload after a WAL-replay restart. If
//!    the kept snapshot were ever NOT the cumulative post-image, some
//!    records would be missing / stale post-replay.
//!
//! 2. `coalesced_commit_wal_scales_with_pages_not_records_849` — the
//!    RED-on-revert amplification oracle: the WAL bytes written by a
//!    600-record single commit are `< PAGE_SIZE` PER RECORD. Reverting
//!    the coalesce (one full page per record × 2 = ~16 KiB/rec) makes
//!    this assertion FAIL.

use std::sync::Arc;

use arcgraph_core::{LabelId, NodeId, PAGE_SIZE, PageId, TenantId};
use arcgraph_storage::crud::{
    CrudStore, PropertyData, commit, create_node, crud_allocator_seed_handle, read_node_with_store,
};
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::{PrimaryIndex, PrimaryKey, RecordKind};
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{
    AllocatorSeedHandle, BlobStoreHandle, PageStoreTarget, PrimaryPageStoreHandle,
    RecordPageStoreHandle, WalConfig, WalWriter, recover_from_wal,
};
use tempfile::TempDir;

fn test_wal_config(dir: &std::path::Path) -> WalConfig {
    WalConfig {
        dir: dir.to_path_buf(),
        segment_size_bytes: 64 * 1024 * 1024,
        group_commit_window: std::time::Duration::from_millis(2),
        group_commit_max_batch: 4,
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
            .expect("CrudStore::new_with_index exposes a record store"),
    ) as Arc<dyn RecordPageStoreHandle>;
    let blob_handle: Arc<dyn BlobStoreHandle> =
        Arc::clone(store.blob_store()) as Arc<dyn BlobStoreHandle>;
    let allocator_seed: Arc<dyn AllocatorSeedHandle> =
        crud_allocator_seed_handle(Arc::clone(&store), Arc::clone(&alloc));
    let target = PageStoreTarget::primary_only(primary_handle)
        .with_record_store(records_handle)
        .with_blob_store(blob_handle)
        .with_allocator_seed(allocator_seed);
    let _report = recover_from_wal(wal_dir, Arc::clone(&mgr), target, None).unwrap();
    (writer, mgr, primary, store)
}

fn wal_bytes(dir: &std::path::Path) -> u64 {
    let mut total = 0u64;
    if let Ok(read) = std::fs::read_dir(dir) {
        for entry in read.flatten() {
            if let Ok(md) = entry.metadata() {
                if md.is_file() {
                    total += md.len();
                }
            }
        }
    }
    total
}

const N: u64 = 600;

#[test]
fn coalesced_multi_record_commit_replays_losslessly_849() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();
    let tenant = TenantId::DEFAULT;

    // Commit N distinct nodes in ONE transaction. With NODE_CAPACITY
    // records per 8 KiB page, these pack onto a handful of record pages
    // and share primary-index leaves — so each touched page is staged
    // (and, pre-fix, WAL-logged) ~hundreds of times. Each node carries a
    // unique (label, inline-pair) fingerprint so the post-replay read is
    // an exact-identity oracle.
    let (writer, mgr, primary, store) = build_stack(&wal_dir);
    let mut ids = Vec::with_capacity(N as usize);
    let mut tx = mgr.begin(tenant);
    for i in 0..N {
        let id = create_node(
            &store,
            &mut tx,
            tenant,
            LabelId::new((i as u32) + 1),
            &PropertyData::InlineU32Pair(i as u32, (i as u32).wrapping_mul(2)),
        )
        .unwrap();
        ids.push((i, id));
    }
    commit(tx, &store).unwrap();

    // Restart: shutdown + fresh stack + WAL replay.
    writer.shutdown().unwrap();
    drop(store);
    drop(primary);
    drop(mgr);
    let (writer2, mgr2, _primary2, store2) = recover_stack(&wal_dir);

    // Every one of the N records survives with its EXACT fingerprint.
    let tx2 = mgr2.begin(tenant);
    for (i, id) in &ids {
        let rec = read_node_with_store(&store2, &tx2, *id)
            .unwrap()
            .unwrap_or_else(|| {
                panic!("node {i} (id={id:?}) missing post-replay — coalesce dropped a live record")
            });
        assert_eq!(
            rec.label_id,
            (*i as u32) + 1,
            "node {i} label mismatch post-replay"
        );
        assert_eq!(
            rec.inline_u32a, *i as u32,
            "node {i} inline_a mismatch post-replay"
        );
        assert_eq!(
            rec.inline_u32b,
            (*i as u32).wrapping_mul(2),
            "node {i} inline_b mismatch post-replay"
        );
    }
    writer2.shutdown().unwrap();
}

#[test]
fn coalesced_commit_wal_scales_with_pages_not_records_849() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();
    let tenant = TenantId::DEFAULT;

    let (writer, mgr, _primary, store) = build_stack(&wal_dir);
    let before = wal_bytes(&wal_dir);

    // One commit of N empty-property nodes.
    let mut tx = mgr.begin(tenant);
    for _ in 0..N {
        create_node(
            &store,
            &mut tx,
            tenant,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .unwrap();
    }
    commit(tx, &store).unwrap();
    writer.shutdown().unwrap();

    let written = wal_bytes(&wal_dir) - before;
    let per_record = written as f64 / N as f64;

    // After coalescing, the bundle logs each touched page ONCE: N empty
    // nodes occupy only a few record pages + a few index pages, so WAL
    // per record is a few HUNDRED bytes (≪ one 8 KiB page). Pre-fix,
    // each record logged its own record page + index leaf = ~2 × 8 KiB,
    // so `per_record` was ~16 KiB and this assertion FAILED. The
    // `< PAGE_SIZE` bound is the RED-on-revert oracle: reverting the
    // coalesce re-introduces ≥ 1 full page per record.
    assert!(
        per_record < PAGE_SIZE as f64,
        "WAL/record = {per_record:.0} B (total {written} B for {N} records) — \
         expected ≪ one {PAGE_SIZE} B page; the per-record full-page staged-page \
         amplification (#849 B3(a)) has regressed"
    );
}

// ─────────────────────────────────────────────────────────────────────
// #1200: sidechannel-coalesce companion of #849's staged-page coalesce.
//
// #849 added `coalesce_staged_pages` (collapse staged PAGES last-wins on
// the single-large-commit bulk path) but did NOT coalesce the SIDECHANNEL
// root-pointer writes on the same commit. The N=600 test above stops ONE
// root-growth short (height 1→2). This companion crosses the height-2→3
// boundary (>51,765 keys → 2 grow_roots → ≥ 2 `(SYSTEM, ROOT_KEY)`
// sidechannel writes at the SAME commit_lsn), so it exercises the
// `coalesce_sidechannel_writes` fix (#1200, the natural completion of
// #849). Pre-fix: crash + WAL replay strands the durable root at the
// INTERMEDIATE root (value-blind idempotency skip drops the FINAL root),
// leaving ~78% of index keys unreachable. Post-fix: 0 misses.
//
// Heavy (60k-key single commit + crash-replay) → env-gated
// panic-by-default, mirroring `durable_ingest_throughput_849.rs`.
// ─────────────────────────────────────────────────────────────────────

const HEIGHT3_KEYS: u32 = 60_000;
const REPRO_RUN_ENV: &str = "ARC1200_REPRO";
const REPRO_SKIP_OK_ENV: &str = "ARCGRAPH_ARC1200_REPRO_SKIP_OK";

/// Panic-by-default gate (`feedback_test_env_gate_panic_by_default`).
/// Returns `true` when the body should run, `false` on a loud opt-out
/// skip; panics if neither `ARC1200_REPRO=1` nor
/// `ARCGRAPH_ARC1200_REPRO_SKIP_OK=1` is set under `--ignored`.
fn repro_should_run() -> bool {
    let run = std::env::var(REPRO_RUN_ENV)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if run {
        return true;
    }
    if std::env::var(REPRO_SKIP_OK_ENV).is_ok() {
        eprintln!(
            "grow_root_height3_crash_replay_sidechannel_coalesce_1200: SKIPPING \
             (opt-out via {REPRO_SKIP_OK_ENV}=1) — set {REPRO_RUN_ENV}=1 to run"
        );
        return false;
    }
    panic!(
        "grow_root_height3_crash_replay_sidechannel_coalesce_1200: required run-flag \
         {REPRO_RUN_ENV}=1 not set. This heavy (~tens of seconds) crash-replay repro is \
         `#[ignore]`'d; when invoked via `--ignored`, {REPRO_RUN_ENV}=1 must be set so it \
         actually runs. Set {REPRO_RUN_ENV}=1 to run, or {REPRO_SKIP_OK_ENV}=1 to opt into a \
         loud-skip (hostile/CI envs only). Soft-skipping silently after a `--ignored` bypass is \
         the W12δ HIGH-1 bug class (feedback_test_env_gate_panic_by_default)."
    );
}

#[test]
#[ignore = "heavy: 60k-key single commit + crash-replay; gate ARC1200_REPRO=1"]
fn grow_root_height3_crash_replay_sidechannel_coalesce_1200() {
    if !repro_should_run() {
        return;
    }

    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();
    let tenant = TenantId::DEFAULT;

    // ── Ingest HEIGHT3_KEYS nodes in ONE commit (≥ 2 grow_roots).
    let (writer, mgr, primary, store) = build_stack(&wal_dir);
    let mut ids: Vec<NodeId> = Vec::with_capacity(HEIGHT3_KEYS as usize);
    let mut tx = mgr.begin(tenant);
    for i in 0..HEIGHT3_KEYS {
        let id = create_node(
            &store,
            &mut tx,
            tenant,
            LabelId::new(i),
            &PropertyData::Empty,
        )
        .unwrap();
        ids.push(id);
    }
    commit(tx, &store).unwrap();
    let live_root: PageId = primary.root().unwrap();

    // ── Crash + recover.
    writer.shutdown().unwrap();
    drop(store);
    drop(primary);
    drop(mgr);
    let (writer2, mgr2, recovered_primary, _store2) = recover_stack(&wal_dir);

    // Build a fresh index over the recovered page store so `root()`
    // reads the DURABLE replayed MVCC root (the recovered primary's
    // root_cache was seeded to the bootstrap leaf BEFORE replay).
    let recovered_page_store = Arc::clone(recovered_primary.page_store());
    let post_replay_index = PrimaryIndex::with_page_store(
        Arc::clone(&mgr2),
        Arc::new(PageAllocator::new()),
        None,
        recovered_page_store,
    )
    .unwrap();

    let durable_root: PageId = post_replay_index.root().unwrap();
    assert_eq!(
        durable_root, live_root,
        "#1200: post-replay durable root ({durable_root:?}) must equal the live FINAL root \
         ({live_root:?}) — a mismatch means the FINAL `(SYSTEM, ROOT_KEY)` sidechannel write \
         was dropped by the value-blind replay idempotency skip (coalesce regressed)"
    );

    let mut misses = 0u64;
    for id in &ids {
        let key = PrimaryKey::new(tenant, RecordKind::Node, id.raw());
        if post_replay_index.lookup(key).unwrap().is_none() {
            misses += 1;
        }
    }
    assert_eq!(
        misses, 0,
        "#1200: {misses} of {HEIGHT3_KEYS} keys UNREACHABLE via the post-replay primary index \
         (durable_root={durable_root:?}). Pre-fix ≈ 78% (47,072 / 60,000) lost; the sidechannel \
         coalesce (companion of #849 staged-page coalesce) must leave 0 unreachable keys."
    );

    writer2.shutdown().unwrap();
}
