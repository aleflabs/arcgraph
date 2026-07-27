//! ADR-032 Slice 2 regression: `grow_root` during a bundle commit
//! folds the SYSTEM root-pointer update into the SAME outer user
//! CommitBundle via the v2-codec sidechannel write lane. Pre-Slice-2
//! the shape was two bundles (user + standalone SYSTEM persist) with
//! a crash window between them (#66). Post-Slice-2 the shape is one
//! v2 bundle carrying both tenants atomically — #66 is closed by
//! construction.
//!
//! Load-bearing assertions (replaces the pre-Slice-2 regression
//! `grow_root_inside_bundle_produces_separate_system_commit`):
//! - A 512-insert commit that forces a grow_root emits EXACTLY ONE
//!   user CommitBundle; no separate SYSTEM bundle is emitted for
//!   the grow_root root-pointer update.
//! - The one bundle decodes under v2 with primary_tenant = DEFAULT,
//!   carries 512 primary MVCC writes, ≥ 4 IndexPage entries, and
//!   exactly 1 SYSTEM sidechannel write for `PRIMARY_INDEX_ROOT_KEY`.
//! - The bootstrap SYSTEM CommitBundle (from `PrimaryIndex::new`)
//!   still exists but runs outside any user commit; it is the ONLY
//!   other SYSTEM bundle emitted.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use arcgraph_core::{LabelId, NodeId, PageId, TenantId};
use arcgraph_storage::crud::{
    CrudStore, PropertyData, commit, create_node, crud_allocator_seed_handle,
};
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::{
    PRIMARY_INDEX_ROOT_KEY, PrimaryIndex, PrimaryKey, RecordKind,
};
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::bundle::{
    decode_commit_bundle_for_version, decode_commit_bundle_v1, decode_commit_bundle_v2,
    decode_commit_bundle_v3, decode_commit_bundle_v4, decode_commit_bundle_v5,
    decode_commit_bundle_v6,
};
use arcgraph_storage::wal::segment::{
    CURRENT_WAL_FORMAT_VERSION, SegmentHeader, list_segments, segment_filename,
};
use arcgraph_storage::wal::{
    AllocatorSeedHandle, BlobStoreHandle, PageStoreTarget, PrimaryPageStoreHandle,
    RecordPageStoreHandle, WalConfig, WalRecord, WalRecordType, WalWriter, recover_from_wal,
};
use tempfile::TempDir;

fn test_wal_config(dir: PathBuf) -> WalConfig {
    WalConfig {
        dir,
        segment_size_bytes: 64 * 1024 * 1024,
        group_commit_window: Duration::from_millis(1),
        group_commit_max_batch: 16,
        metrics_sink: None,
        encryption: None,
        inflight_budget_bytes: None,
    }
}

/// Read every WAL record alongside its segment's format_version so
/// the test can dispatch the right codec per-record.
fn drain_segments_with_version(dir: &std::path::Path) -> Vec<(u16, WalRecord)> {
    let mut out = Vec::new();
    for seg in list_segments(dir).unwrap() {
        let bytes = std::fs::read(dir.join(segment_filename(seg))).unwrap();
        if bytes.len() < SegmentHeader::SIZE {
            continue;
        }
        let header = SegmentHeader::decode(&bytes[..SegmentHeader::SIZE]).unwrap();
        let mut cursor = SegmentHeader::SIZE;
        while cursor < bytes.len() {
            let (r, consumed) = WalRecord::decode(&bytes[cursor..]).unwrap();
            out.push((header.format_version, r));
            cursor += consumed;
        }
    }
    out
}

fn build_stack() -> (TempDir, Arc<CrudStore>, Arc<TxnManager>, WalWriter) {
    let dir = TempDir::new().unwrap();
    let writer = WalWriter::spawn(test_wal_config(dir.path().to_path_buf())).unwrap();
    let handle = writer.handle();
    let mgr = Arc::new(TxnManager::with_wal(handle.clone()));
    let alloc = Arc::new(PageAllocator::new());
    let primary = Arc::new(
        PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&alloc), Some(handle.clone())).unwrap(),
    );
    let store = Arc::new(CrudStore::new_with_index(
        Some(handle.clone()),
        Arc::clone(&primary),
        alloc,
    ));
    (dir, store, mgr, writer)
}

#[test]
fn grow_root_system_root_pointer_folded_into_outer_bundle() {
    let (dir, store, mgr, writer) = build_stack();

    // 512 keys in a single transaction forces at least one split
    // cascade + grow_root. Leaf fanout is 203 per DEC-9.
    let mut tx = mgr.begin(TenantId::DEFAULT);
    for i in 0..512u32 {
        create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(i),
            &PropertyData::Empty,
        )
        .unwrap();
    }
    commit(tx, &store).unwrap();
    writer.shutdown().unwrap();

    let records = drain_segments_with_version(dir.path());

    // Partition CommitBundle records by tenant.
    let user_bundles: Vec<&(u16, WalRecord)> = records
        .iter()
        .filter(|(_, r)| {
            r.record_type == WalRecordType::CommitBundle && r.tenant_id == TenantId::DEFAULT
        })
        .collect();
    let system_bundles: Vec<&(u16, WalRecord)> = records
        .iter()
        .filter(|(_, r)| {
            r.record_type == WalRecordType::CommitBundle && r.tenant_id == TenantId::SYSTEM
        })
        .collect();

    // ── User side: EXACTLY ONE bundle — no post-commit SYSTEM persist.
    assert_eq!(
        user_bundles.len(),
        1,
        "ADR-032 §2: exactly one user CommitBundle for the 512-node commit (grow_root folds inline)"
    );
    let (user_fv, user_rec) = user_bundles[0];
    assert_eq!(
        *user_fv, CURRENT_WAL_FORMAT_VERSION,
        "#1010 (ADR-199 amendment): post-cutover commits stamp the current WAL format"
    );

    let decoded =
        decode_commit_bundle_for_version(&user_rec.payload, *user_fv, user_rec.tenant_id).unwrap();
    assert_eq!(decoded.primary_tenant, TenantId::DEFAULT);
    assert_eq!(
        decoded.mvcc_writes.len(),
        512,
        "user bundle must carry all 512 MVCC writes"
    );

    // Split cascade lower bound unchanged from pre-Slice-2.
    assert!(
        decoded.staged_pages.len() >= 4,
        "user bundle should carry all split-affected IndexPage emits, got {}",
        decoded.staged_pages.len()
    );

    // ── The grow_root SYSTEM write rides the user bundle as a
    //    sidechannel write, not as a separate SYSTEM bundle.
    assert_eq!(
        decoded.sidechannel_writes.len(),
        1,
        "grow_root must produce exactly one sidechannel SYSTEM write"
    );
    let sc = &decoded.sidechannel_writes[0];
    assert_eq!(sc.tenant_id, TenantId::SYSTEM);
    assert_eq!(sc.key, PRIMARY_INDEX_ROOT_KEY);
    assert!(sc.value.is_some(), "sidechannel write must carry root id");
    let root_bytes = sc.value.as_ref().unwrap();
    assert_eq!(root_bytes.len(), 8);
    let new_root_id = u64::from_le_bytes(root_bytes.as_ref().try_into().unwrap());
    assert!(new_root_id != 0);

    // ── SYSTEM side: ONLY the bootstrap commit from `PrimaryIndex::new`.
    //   Pre-Slice-2 there were TWO SYSTEM bundles (bootstrap + deferred
    //   grow_root persist). Post-Slice-2 the grow_root persist rides
    //   the user bundle, so only bootstrap is left.
    assert_eq!(
        system_bundles.len(),
        1,
        "ADR-032 Slice 2: exactly one SYSTEM bundle (bootstrap only); got {} — grow_root \
         must not emit a standalone SYSTEM bundle",
        system_bundles.len()
    );

    // Bootstrap bundle decodes cleanly under its segment's version
    // (v1 if it was written by PrimaryIndex::new's standalone
    // persist_root_to_mvcc path — which still uses the standalone
    // Transaction::commit that emits v2 post-Slice-2, so bootstrap
    // is v2 too).
    let (boot_fv, boot_rec) = system_bundles[0];
    let boot_decoded = match *boot_fv {
        1 => decode_commit_bundle_v1(&boot_rec.payload, boot_rec.tenant_id).unwrap(),
        2 => decode_commit_bundle_v2(&boot_rec.payload, boot_rec.tenant_id).unwrap(),
        3 => decode_commit_bundle_v3(&boot_rec.payload, boot_rec.tenant_id).unwrap(),
        4 => decode_commit_bundle_v4(&boot_rec.payload, boot_rec.tenant_id).unwrap(),
        5 => decode_commit_bundle_v5(&boot_rec.payload, boot_rec.tenant_id).unwrap(),
        // #352 Part 2 (ADR-199): bundle format bumped to v6.
        6 => decode_commit_bundle_v6(&boot_rec.payload, boot_rec.tenant_id).unwrap(),
        // #1010 (v7) + #1221 (ADR-218, v8): route through the dispatcher.
        7 | 8 => decode_commit_bundle_for_version(&boot_rec.payload, *boot_fv, boot_rec.tenant_id)
            .unwrap(),
        other => panic!("unexpected format_version on bootstrap bundle: {other}"),
    };
    assert_eq!(
        boot_decoded.mvcc_writes.len(),
        1,
        "bootstrap SYSTEM bundle: 1 MVCC write (initial root pointer)"
    );
    assert!(boot_decoded.sidechannel_writes.is_empty());
    assert!(boot_decoded.staged_pages.is_empty());
}

#[test]
fn grow_root_updates_root_cache_to_new_root_id() {
    // Sanity: after a grow_root, lookups must route through the
    // new root. Read-after-split correctness. Unchanged from
    // pre-Slice-2 semantics.
    let (_dir, store, mgr, _writer) = build_stack();
    let mut tx = mgr.begin(TenantId::DEFAULT);
    let mut ids = Vec::with_capacity(512);
    for i in 0..512u32 {
        let id = create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(i),
            &PropertyData::Empty,
        )
        .unwrap();
        ids.push(id);
    }
    commit(tx, &store).unwrap();

    // Every key we inserted must be findable post-commit.
    let reader_tx = mgr.begin(TenantId::DEFAULT);
    for id in &ids {
        let rec = arcgraph_storage::crud::read_node(&reader_tx, *id).unwrap();
        assert!(
            rec.is_some(),
            "post-split lookup missed {id:?} — root_cache may be stale"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// #1200: grow_root SYSTEM root-pointer sidechannel-coalesce —
//        crash-replay corruption repro (the LOAD-BEARING test).
//
// A single commit crossing the height-2→3 boundary (>51,765 keys; leaf
// cap 203 / internal cap 254 — `primary_index.rs`) fires ≥ 2 grow_roots
// in ONE commit. Each grow_root pushes a `(SYSTEM, ROOT_KEY)`
// SideChannelWrite at the SAME commit_lsn. Pre-fix, on crash + WAL
// replay the value-blind idempotency skip
// (`apply_replay_mvcc_write`, "Lemma I1") applies sc[0] (the
// INTERMEDIATE root) and SKIPS sc[1] (the FINAL root) — stranding the
// durable MVCC root at the intermediate one, leaving ~78% of index keys
// unreachable via the primary index after recovery (#1200; the issue's
// empirical repro = 47,072 / 60,000 misses). With the coalesce fix the
// bundle carries ONE `(SYSTEM, ROOT_KEY)` write (the final root), so
// replay restores the FINAL root and 0 keys are unreachable.
//
// This is the crux of #1200: it proves the RELEASE replay corruption is
// fixed (not just the DEBUG assert). It is the RED-on-revert oracle —
// reverting `coalesce_sidechannel_writes` makes it FAIL with thousands
// of index misses.
// ─────────────────────────────────────────────────────────────────────

/// More than 51,765 keys forces the height-2→3 growth (2nd grow_root).
/// 60_000 matches the issue's empirical repro corpus
/// (47,072 / 60,000 ≈ 78% unreachable pre-fix).
const HEIGHT3_KEYS: u32 = 60_000;

const REPRO_RUN_ENV: &str = "ARC1200_REPRO";
const REPRO_SKIP_OK_ENV: &str = "ARCGRAPH_ARC1200_REPRO_SKIP_OK";

/// Panic-by-default gate for the heavy `#[ignore]`'d crash-replay repro
/// (`feedback_test_env_gate_panic_by_default`). The 60k-key single
/// commit + crash + full WAL replay is too slow for the default
/// gauntlet, so it is `#[ignore]`'d. When a `--ignored` runner invokes
/// it, `ARC1200_REPRO=1` MUST be set or it PANICS — never a silent
/// soft-skip (the W12δ HIGH-1 bug class).
/// `ARCGRAPH_ARC1200_REPRO_SKIP_OK=1` opts into a LOUD skip for hostile
/// / CI hosts that run `--ignored` broadly.
///
/// Returns `true` when the test body should run, `false` on a loud
/// opt-out skip (and panics otherwise).
fn repro_should_run() -> bool {
    let run = std::env::var(REPRO_RUN_ENV)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if run {
        return true;
    }
    if std::env::var(REPRO_SKIP_OK_ENV).is_ok() {
        eprintln!(
            "grow_root_height3_crash_replay_no_unreachable_keys_1200: SKIPPING \
             (opt-out via {REPRO_SKIP_OK_ENV}=1) — set {REPRO_RUN_ENV}=1 to run"
        );
        return false;
    }
    panic!(
        "#1200 heavy crash-replay repro: required run-flag {REPRO_RUN_ENV}=1 \
         not set. This heavy (~tens of seconds) repro is `#[ignore]`'d; when \
         invoked via `--ignored`, {REPRO_RUN_ENV}=1 must be set so it actually \
         runs. Set {REPRO_RUN_ENV}=1 to run, or {REPRO_SKIP_OK_ENV}=1 to opt \
         into a loud-skip (hostile/CI envs only). Soft-skipping silently after \
         a `--ignored` bypass is the W12δ HIGH-1 bug class \
         (feedback_test_env_gate_panic_by_default)."
    );
}

/// Build a stack whose primary page store + txn_mgr we can recover into.
fn build_stack_in(
    wal_dir: &std::path::Path,
) -> (
    WalWriter,
    Arc<TxnManager>,
    Arc<PrimaryIndex>,
    Arc<CrudStore>,
) {
    let writer = WalWriter::spawn(test_wal_config(wal_dir.to_path_buf())).unwrap();
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

/// Recover a fresh stack from the WAL, returning the recovered primary
/// index (its page store now carries all replayed pages), the recovered
/// txn manager, the replay-skipped-idempotent counter, and the writer.
fn recover_stack_in(
    wal_dir: &std::path::Path,
) -> (WalWriter, Arc<TxnManager>, Arc<PrimaryIndex>, u64) {
    let writer = WalWriter::spawn(test_wal_config(wal_dir.to_path_buf())).unwrap();
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
    let report = recover_from_wal(wal_dir, Arc::clone(&mgr), target, None).unwrap();
    let skipped = report.metrics.bundles_skipped_idempotent;
    (writer, mgr, primary, skipped)
}

#[test]
#[ignore = "heavy: 60k-key single commit + crash-replay; gate ARC1200_REPRO=1"]
fn grow_root_height3_crash_replay_no_unreachable_keys_1200() {
    if !repro_should_run() {
        return;
    }

    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();
    let tenant = TenantId::DEFAULT;

    // ── Phase A: ingest HEIGHT3_KEYS nodes in ONE commit (≥ 2 grow_roots).
    let (writer, mgr, primary, store) = build_stack_in(&wal_dir);
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

    // Live (pre-crash) index must see ALL keys — the live path is
    // benign last-wins (the issue's RELEASE-live-correct observation).
    let live_root = primary.root().unwrap();
    {
        let mut live_misses = 0u64;
        for id in &ids {
            let key = PrimaryKey::new(tenant, RecordKind::Node, id.raw());
            if primary.lookup(key).unwrap().is_none() {
                live_misses += 1;
            }
        }
        assert_eq!(
            live_misses, 0,
            "live (pre-crash) index lost {live_misses} keys — live path should be benign last-wins"
        );
    }

    // ── Phase B: crash (writer shutdown) + drop the runtime stack.
    writer.shutdown().unwrap();
    drop(store);
    drop(primary);
    drop(mgr);

    // ── Phase C: recover from the WAL prefix.
    let (writer2, mgr2, recovered_primary, _skipped) = recover_stack_in(&wal_dir);

    // The recovered primary's `root_cache` was seeded to the fresh
    // bootstrap leaf during `PrimaryIndex::new` (before replay), so it
    // does NOT reflect the replayed MVCC root. Build a SECOND index over
    // the SAME recovered page store + recovered txn_mgr: its root_cache
    // starts empty, so `root()` reads the DURABLE replayed root from
    // MVCC, and `lookup` descends the recovered pages. This is the true
    // post-recovery index view (the path that silently lost ~78% of keys
    // pre-fix).
    let recovered_page_store = Arc::clone(recovered_primary.page_store());
    let recovered_alloc = Arc::new(PageAllocator::new());
    let post_replay_index = PrimaryIndex::with_page_store(
        Arc::clone(&mgr2),
        recovered_alloc,
        None,
        recovered_page_store,
    )
    .unwrap();

    // The durable root after replay must be the FINAL (outermost) root,
    // NOT the intermediate one stranded by the value-blind skip.
    let durable_root: PageId = post_replay_index.root().unwrap();
    assert_eq!(
        durable_root, live_root,
        "#1200: post-replay durable root ({durable_root:?}) must equal the live FINAL root \
         ({live_root:?}); a mismatch means the value-blind idempotency skip stranded the \
         durable root at an INTERMEDIATE root (the coalesce fix has regressed)"
    );

    // Every key inserted must be reachable through the post-replay
    // primary index. Pre-fix this misses ~78% (47,072 / 60,000).
    let mut misses = 0u64;
    let mut first_miss: Option<u64> = None;
    for id in &ids {
        let key = PrimaryKey::new(tenant, RecordKind::Node, id.raw());
        if post_replay_index.lookup(key).unwrap().is_none() {
            misses += 1;
            if first_miss.is_none() {
                first_miss = Some(id.raw());
            }
        }
    }
    assert_eq!(
        misses, 0,
        "#1200: {misses} of {HEIGHT3_KEYS} keys UNREACHABLE via the post-replay primary index \
         (first miss id={first_miss:?}, durable_root={durable_root:?}). Pre-fix ≈ 78% \
         (47,072 / 60,000) are lost because the FINAL root sidechannel write is dropped by the \
         value-blind replay idempotency skip. The coalesce fix must leave 0 unreachable keys."
    );

    writer2.shutdown().unwrap();
}

#[test]
#[ignore = "heavy: 60k-key single commit on a DEBUG build; gate ARC1200_REPRO=1"]
fn grow_root_height3_single_commit_no_debug_assert_1200() {
    // The same >51,765-key single commit must complete CLEANLY on a
    // DEBUG build. Pre-fix it panicked at `apply_sidechannel_mvcc_write`
    // (the `created_lsn < commit_lsn` debug-assert) when the 2nd
    // grow_root's `(SYSTEM, ROOT_KEY)` write landed at the same
    // commit_lsn. Post-fix the duplicate is coalesced away before the
    // Phase-3 apply loop, so the assert is never reached. This test
    // asserts NO PANIC — it would abort the process pre-fix on a DEBUG
    // build.
    if !repro_should_run() {
        return;
    }

    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();
    let tenant = TenantId::DEFAULT;

    let (writer, mgr, primary, store) = build_stack_in(&wal_dir);
    let mut tx = mgr.begin(tenant);
    for i in 0..HEIGHT3_KEYS {
        create_node(
            &store,
            &mut tx,
            tenant,
            LabelId::new(i),
            &PropertyData::Empty,
        )
        .unwrap();
    }
    // Pre-fix: this commit's Phase-3 apply loop trips the debug-assert
    // when applying the 2nd `(SYSTEM, ROOT_KEY)` sidechannel write at the
    // already-used commit_lsn → process abort on a DEBUG build.
    commit(tx, &store).unwrap();

    // Sanity: the SYSTEM root pointer resolves to a non-zero final root.
    let root = primary.root().unwrap();
    assert!(root.raw() != 0, "root pointer must resolve post-grow_root");
    writer.shutdown().unwrap();
}
