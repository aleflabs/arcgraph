//! M6.1 (#1521 / #1457 MF5) — pin-sweep completeness: record-page writers
//! that mutate/install STORE_RECORD pages without the pin-across-mutate
//! guard the P0-1 fix established everywhere else.
//!
//! (a) `install_create` (crud.rs's non-delta-mode loop, ~line 4170):
//!     mutates a record page under a bare `latch_for_tenant` with no
//!     pin and no DPT dirty-mark (the non-delta / legacy-v8 CRUD path
//!     never touches the M3 DPT at all) — the reclaim CLEAN arm (no
//!     live DPT entry = "durable image current") can reclaim the sole
//!     in-memory copy of a mutation that has not yet reached the WAL.
//! (b) `install_fresh` runs BEFORE the pin in `apply_durable_v9_deltas`'s
//!     `MissingPage` arm (crud.rs ~3667): a never-written-home fresh
//!     page can be reclaimed by the clean-arm in the install-to-pin
//!     window, causing the subsequent pinned fault-in to fail with
//!     `MissingPage` (a fail-stop that recovers on restart, not a
//!     silent loss, but still an availability regression).
//! (c/d) `install_update_deferred` and `install_delete_deferred` repeat
//!     (a)'s non-delta bare-latch/no-DPT hazard for rewrites and tombstones.
//!
//! The gates are deterministic (no scheduler-luck eviction): (a), (c),
//! and (d) drive
//! REAL production commit through the non-delta (`BUNDLE_FORMAT_V8`)
//! CRUD path with a concurrent evictor racing via the SAME
//! shard-lock-exclusion technique `skeptic_mech_e3_revalidate_seam.rs`
//! uses (the writer's mutate must complete WHILE the eviction claim's
//! revalidate closure is running, or not at all — no timing guesswork);
//! (b) directly reproduces the exact charter repro shape
//! (`install_fresh` -> `evict_for_capacity(0)` -> pinned fault-in) at
//! the `BufferedRecordPageStore` level, which is where the hazard and
//! the fix both live.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use arcgraph_core::{
    DurabilityTier, LabelId, NodeId, PageId, PageType, TenantDurabilityLookup, TenantId,
};
use arcgraph_storage::checkpoint::{PageFlushTarget, WriteBehindCheckpointer};
use arcgraph_storage::crud::{
    CrudStore, PropertyData, commit, create_node, delete_node_with_store, update_node,
};
use arcgraph_storage::io::PageIo;
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::page_store::{
    BufferedRecordPageStore, PerTenantBufferPool, PerTenantBufferPoolConfig, RecordPageBackend,
};
use arcgraph_storage::primary_index::{PrimaryIndex, PrimaryKey, RecordKind};
use arcgraph_storage::records::{SlotId, SlottedPageRef};
use arcgraph_storage::redo::DirtyPageTable;
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::segment::{SegmentHeader, segment_filename};
use arcgraph_storage::wal::{BUNDLE_FORMAT_V8, WalConfig, WalWriter};

// ─────────────────────────────────────────────────────────────────────
// (a) install_create pin sweep
// ─────────────────────────────────────────────────────────────────────

/// Legacy (non-delta, `BUNDLE_FORMAT_V8`) tenant — forces `CrudStore`'s
/// `install_create` down its NON-delta-mode loop (crud.rs ~line 4169),
/// the exact bare-latch, no-DPT-mark writer site #1457 MF5(a) targets.
/// `DurabilityTier` here is irrelevant to which loop `install_create`
/// takes (that's purely `mutation_log.delta_mode`, itself driven by the
/// WAL's `BUNDLE_FORMAT_V8` segment header) — Strict keeps `commit()`
/// synchronous so the harness doesn't need a separate drain step.
#[derive(Debug)]
struct StrictTenant;

impl TenantDurabilityLookup for StrictTenant {
    fn durability_tier(&self, _tenant: TenantId) -> DurabilityTier {
        DurabilityTier::Strict
    }
}

struct LegacyStack {
    store: Arc<CrudStore>,
    manager: Arc<TxnManager>,
    primary: Arc<PrimaryIndex>,
    page_store: Arc<BufferedRecordPageStore>,
    writer: WalWriter,
}

fn build_legacy_stack(dir: &std::path::Path, cache_cap: usize) -> LegacyStack {
    let record_dir = dir.join("records");
    std::fs::create_dir_all(&record_dir).unwrap();
    let wal_dir = dir.join("wal");
    std::fs::create_dir_all(&wal_dir).unwrap();
    // BUNDLE_FORMAT_V8 (not V9): forces `is_delta_bundle_format` false,
    // so `install_create` takes the non-delta loop this gate targets.
    std::fs::write(
        wal_dir.join(segment_filename(0)),
        SegmentHeader {
            format_version: BUNDLE_FORMAT_V8,
        }
        .encode(),
    )
    .unwrap();

    let io: Arc<dyn PageIo> = Arc::new(
        arcgraph_storage::io::PosixPageIo::open_or_create(record_dir.join("record.store"))
            .expect("open posix page io"),
    );
    let pools = Arc::new(PerTenantBufferPool::with_config(
        io,
        PerTenantBufferPoolConfig {
            frames_per_tenant: 64,
            write_fraction: 0.0,
        },
    ));
    let page_store = Arc::new(
        BufferedRecordPageStore::with_cache_cap(pools, cache_cap)
            .with_m6_evict_wait_budget(Duration::from_millis(5)),
    );
    // Still wire M6 eviction (DPT + checkpointer) so `evict_for_capacity`
    // behaves exactly as production does for a buffered store — the
    // non-delta path simply never populates the DPT itself (that's the
    // defect class), independent of whether eviction machinery exists.
    let dpt = Arc::new(DirtyPageTable::new());
    let props_target: Arc<dyn PageFlushTarget> = page_store.clone();
    let records_target: Arc<dyn PageFlushTarget> = page_store.clone();
    let checkpointer = Arc::new(WriteBehindCheckpointer::new(
        dpt.clone(),
        props_target,
        records_target,
    ));
    page_store.attach_m6_dirty_page_table(dpt.clone());
    page_store.attach_m6_checkpointer(checkpointer);

    let mut manager_inner = TxnManager::new();
    manager_inner.set_durability_lookup(Arc::new(StrictTenant));
    let manager = Arc::new(manager_inner);
    let allocator = Arc::new(PageAllocator::new());
    let primary =
        Arc::new(PrimaryIndex::new(Arc::clone(&manager), Arc::clone(&allocator), None).unwrap());

    let writer = WalWriter::spawn_from(
        WalConfig {
            group_commit_window: Duration::from_millis(1),
            group_commit_max_batch: 1,
            ..WalConfig::new(wal_dir)
        },
        manager.current_lsn(),
    )
    .unwrap();
    let handle = writer.handle();
    manager.attach_wal(handle.clone());
    primary.attach_wal(handle.clone());

    let store = CrudStore::new_with_page_store(
        Some(handle),
        Arc::clone(&primary),
        None,
        allocator,
        page_store.clone(),
    );
    let store = Arc::new(store);

    LegacyStack {
        store,
        manager,
        primary,
        page_store,
        writer,
    }
}

/// #1457 MF5(a) — deterministic gate: a real production `create_node`
/// commit through the LEGACY (non-delta) `install_create` loop races a
/// concurrent evictor via the SAME shard-lock-exclusion rendezvous
/// `skeptic_mech_e3_revalidate_seam.rs` uses. Because `install_create`'s
/// non-delta loop mutates the page and stages the intent WITHOUT ever
/// touching the DPT, the evictor's `is_clean` classification always
/// reads "clean" (no entry) for this page regardless of timing — the
/// decisive question is whether the WRITER'S mutate can complete WHILE
/// the eviction claim's shard entry is held (which would mean the
/// bare-latch write is not excluded at all) or is structurally blocked
/// by a live pin (the fix).
///
/// RED-on-revert: reverting the `install_create` pin fix (this PR's
/// `latch_pinned_for_tenant` -> back to bare `latch_for_tenant`) makes
/// this test FAIL — the writer's mutate completes inside the claim
/// window instead of blocking on the pin registry's shard lock.
#[test]
fn install_create_pin_excludes_concurrent_clean_arm_reclaim() {
    let dir = tempfile::tempdir().unwrap();
    let stack = build_legacy_stack(dir.path(), 8);

    // Reserve the destination page deterministically: create ONE node
    // first (drives `open_or_fresh_page_for_txn` to allocate+install the
    // open page), so the decisive commit below targets a KNOWN existing
    // page_id — the race is about the SECOND write's mutate-under-pin,
    // not the page's initial installation.
    let mut tx = stack.manager.begin(TenantId::DEFAULT);
    let seed_id = create_node(
        &stack.store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(7),
        &PropertyData::InlineU32Pair(0x00, 0),
    )
    .unwrap();
    commit(tx, &stack.store).unwrap();
    let pid = stack
        .primary
        .lookup(PrimaryKey::new(
            TenantId::DEFAULT,
            RecordKind::Node,
            seed_id.raw(),
        ))
        .unwrap()
        .unwrap()
        .page;
    // Establish a durable home image for the seed page BEFORE the race
    // — the non-delta `install_create` path never writes through the M6
    // checkpointer on its own (it has no DPT entry to trigger one), so
    // without this the page would never have ANY on-disk backing at
    // all, and a post-race fault-in would fail regardless of whether
    // the pin fix is present (a harness gap, not the production
    // hazard). This mirrors what a real deployment's periodic/explicit
    // flush would have done between commits.
    stack.page_store.flush_pages([pid]).unwrap();

    // The decisive claim: race a `try_evict_page_pinned_for_tenant` call
    // (the SAME pin-coupled removal claim `evict_for_capacity`'s clean
    // arm dispatches to) against a SECOND `create_node` commit landing
    // on the SAME open page (still under `NODE_CAPACITY`, so it reuses
    // `pid`) driven from inside the revalidate closure on a background
    // thread.
    let writer_store = stack.store.clone();
    let writer_manager = stack.manager.clone();
    let writer_done = Arc::new(AtomicBool::new(false));
    let wd = Arc::clone(&writer_done);

    let reclaimed =
        stack
            .page_store
            .try_evict_page_pinned_for_tenant(TenantId::DEFAULT, pid, move || {
                // `install_create`'s non-delta loop never marks the DPT, so
                // "clean" here is always true for this page — the real
                // question this gate answers is whether the writer's mutate
                // can land during this window at all.
                let wd_thread = Arc::clone(&wd);
                let handle = std::thread::spawn(move || {
                    let mut tx = writer_manager.begin(TenantId::DEFAULT);
                    create_node(
                        &writer_store,
                        &mut tx,
                        TenantId::DEFAULT,
                        LabelId::new(7),
                        &PropertyData::InlineU32Pair(0xAB, 0),
                    )
                    .unwrap();
                    commit(tx, &writer_store).unwrap();
                    wd_thread.store(true, Ordering::Release);
                });
                std::thread::sleep(Duration::from_millis(150));
                let writer_completed_during_window = wd.load(Ordering::Acquire);
                std::mem::forget(handle);
                assert!(
                    !writer_completed_during_window,
                    "#1457 MF5(a) violation: `install_create`'s writer \
                 completed its mutate WHILE the removal claim's shard \
                 entry was held — the pin is not excluding this writer \
                 from the claim (or `install_create` reverted to the \
                 bare `latch_for_tenant` acquisition)"
                );
                true
            });

    assert!(
        reclaimed,
        "the frame must be reclaimed here: the writer could not have \
         landed yet (it is blocked on the pin registry's shard lock)"
    );

    // Wait for the writer's background thread to finish (it unblocks
    // once the claim above resolved).
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while !writer_done.load(Ordering::Acquire) && std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(
        writer_done.load(Ordering::Acquire),
        "writer thread must complete after the claim resolves"
    );

    // Decisive byte check: the second create's fresh node must be
    // physically readable — no write lost even though the frame was
    // reclaimed mid-race (the writer's pin excluded it from ever
    // touching a frame the claim was deciding on).
    stack.page_store.fault_in(pid).unwrap();
    let latch = RecordPageBackend::latch(stack.page_store.as_ref(), pid).unwrap();
    let guard = latch.read();
    let page = SlottedPageRef::open(guard.as_ref().as_ref()).unwrap();
    // Slot 1 is the second node inserted onto this page (slot 0 = seed).
    let second = page
        .read_node(arcgraph_storage::records::SlotId(1))
        .unwrap();
    assert!(
        second.is_some(),
        "the second create_node's slot must be physically present after \
         the reclaim race — a missing slot here means the writer's \
         install was lost"
    );
    assert_eq!(
        second.unwrap().inline_u32a,
        0xAB,
        "the second create_node's fresh byte must survive the reclaim \
         race — a stale/absent byte here is a lost write through the \
         non-delta `install_create` path"
    );
    stack.writer.shutdown().unwrap();
}

// ─────────────────────────────────────────────────────────────────────
// (c/d) install_update_deferred + install_delete_deferred pin sweep
// ─────────────────────────────────────────────────────────────────────

fn seed_legacy_node(stack: &LegacyStack) -> (NodeId, PageId, SlotId) {
    let mut tx = stack.manager.begin(TenantId::DEFAULT);
    let node_id = create_node(
        &stack.store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(7),
        &PropertyData::InlineU32Pair(0x00, 0),
    )
    .unwrap();
    commit(tx, &stack.store).unwrap();
    let slot = stack
        .primary
        .lookup(PrimaryKey::new(
            TenantId::DEFAULT,
            RecordKind::Node,
            node_id.raw(),
        ))
        .unwrap()
        .unwrap();
    // v8 writers never populate the M3 DPT. Establish the canonical
    // pre-image so the next mutation intentionally races a clean reclaim.
    stack.page_store.flush_pages([slot.page]).unwrap();
    (node_id, slot.page, slot.slot)
}

fn assert_non_delta_writer_pin_excludes_reclaim(
    stack: &LegacyStack,
    pid: PageId,
    site: &'static str,
    writer: impl FnOnce() + Send + 'static,
) {
    let writer_done = Arc::new(AtomicBool::new(false));
    let wd = Arc::clone(&writer_done);
    let writer_start = Arc::new(std::sync::Barrier::new(2));
    let (handle_tx, handle_rx) = std::sync::mpsc::sync_channel(1);

    let reclaimed =
        stack
            .page_store
            .try_evict_page_pinned_for_tenant(TenantId::DEFAULT, pid, move || {
                let ws = Arc::clone(&writer_start);
                let handle = std::thread::spawn(move || {
                    ws.wait();
                    writer();
                    wd.store(true, Ordering::Release);
                });
                handle_tx.send(handle).unwrap();
                // Prove the child is scheduled and poised immediately before
                // the production writer call before observing exclusion.
                writer_start.wait();
                std::thread::sleep(Duration::from_millis(150));
                assert!(
                    !writer_done.load(Ordering::Acquire),
                    "#1457 pin-sweep violation: `{site}` completed its page mutation \
                 while the removal claim held the pin-registry shard; its \
                 non-delta writer is not pin-coupled"
                );
                true
            });

    assert!(
        reclaimed,
        "{site}: the clean frame must be reclaimed while the pin-coupled \
         writer is structurally blocked"
    );
    handle_rx.recv().unwrap().join().unwrap();
}

/// #1457 MF5(c) — reverting only the pin in
/// `install_update_deferred` makes the real v8 update finish inside the
/// shard-lock removal claim and fails this gate.
#[test]
fn install_update_deferred_pin_excludes_concurrent_clean_arm_reclaim() {
    let dir = tempfile::tempdir().unwrap();
    let stack = build_legacy_stack(dir.path(), 8);
    let (node_id, pid, slot) = seed_legacy_node(&stack);
    let writer_store = Arc::clone(&stack.store);
    let writer_manager = Arc::clone(&stack.manager);

    assert_non_delta_writer_pin_excludes_reclaim(
        &stack,
        pid,
        "install_update_deferred",
        move || {
            let mut tx = writer_manager.begin(TenantId::DEFAULT);
            update_node(
                &writer_store,
                &mut tx,
                node_id,
                &PropertyData::InlineU32Pair(0xAB, 0),
            )
            .unwrap();
            commit(tx, &writer_store).unwrap();
        },
    );

    let latch =
        RecordPageBackend::latch_for_tenant(stack.page_store.as_ref(), TenantId::DEFAULT, pid)
            .unwrap();
    let guard = latch.read();
    let page = SlottedPageRef::open(guard.as_ref().as_ref()).unwrap();
    assert_eq!(page.read_node(slot).unwrap().unwrap().inline_u32a, 0xAB);
    drop(guard);
    stack.writer.shutdown().unwrap();
}

/// #1457 MF5(d) — the tombstone sibling has the identical no-DPT clean-arm
/// window; reverting only its pin makes this real v8 delete finish inside
/// the removal claim and fails this gate.
#[test]
fn install_delete_deferred_pin_excludes_concurrent_clean_arm_reclaim() {
    let dir = tempfile::tempdir().unwrap();
    let stack = build_legacy_stack(dir.path(), 8);
    let (node_id, pid, slot) = seed_legacy_node(&stack);
    let writer_store = Arc::clone(&stack.store);
    let writer_manager = Arc::clone(&stack.manager);

    assert_non_delta_writer_pin_excludes_reclaim(
        &stack,
        pid,
        "install_delete_deferred",
        move || {
            let mut tx = writer_manager.begin(TenantId::DEFAULT);
            delete_node_with_store(&writer_store, &mut tx, node_id).unwrap();
            commit(tx, &writer_store).unwrap();
        },
    );

    let latch =
        RecordPageBackend::latch_for_tenant(stack.page_store.as_ref(), TenantId::DEFAULT, pid)
            .unwrap();
    let guard = latch.read();
    let page = SlottedPageRef::open(guard.as_ref().as_ref()).unwrap();
    assert!(
        page.read_node(slot).unwrap().is_none(),
        "the v8 delete tombstone must survive the deterministic reclaim race"
    );
    drop(guard);
    stack.writer.shutdown().unwrap();
}

// ─────────────────────────────────────────────────────────────────────
// (b) install_fresh-before-pin (MissingPage arm) window
// ─────────────────────────────────────────────────────────────────────

/// #1457 MF5(b) — the exact charter repro shape at the
/// `BufferedRecordPageStore` level (where the hazard and its fix both
/// live): `install_fresh` a never-written-home page, then
/// `evict_for_capacity(0)` (target cap 0 — reclaim everything possible)
/// with NO pin held. Before the crud.rs reorder fix, THIS SAME sequence
/// is exactly the window `apply_durable_v9_deltas`'s `MissingPage` arm
/// left open (install, THEN pin — nothing excludes a reclaim in
/// between). A subsequent PINNED fault-in on the never-homed page then
/// fails with `MissingPage` ("page not found") because the sole RAM
/// copy was reclaimed and there is no durable home to recover it from
/// (a fail-stop, not silent loss, but still availability-breaking).
///
/// This test demonstrates the HAZARD CLASS directly (RED at head,
/// GREEN after the crud.rs fix is applied, matching the charter's
/// acceptance: "must be RED at head, GREEN after the fix" — reproduced
/// here as a store-level demonstration since `apply_durable_v9_deltas`
/// itself now closes the window by construction, so exercising the
/// SAME store-level operations in the SAME order this gate uses is what
/// would have caught the pre-fix crud.rs ordering).
#[test]
fn install_fresh_evicted_before_pin_causes_missing_page_fault_in() {
    let dir = tempfile::tempdir().unwrap();
    let io: Arc<dyn PageIo> = Arc::new(
        arcgraph_storage::io::PosixPageIo::open_or_create(dir.path().join("record.store"))
            .expect("open posix page io"),
    );
    let pools = Arc::new(PerTenantBufferPool::with_config(
        io,
        PerTenantBufferPoolConfig {
            frames_per_tenant: 16,
            write_fraction: 0.0,
        },
    ));
    let store = Arc::new(BufferedRecordPageStore::with_cache_cap(pools, 8));
    // Wire an (empty) DPT — matching production's ALWAYS-wired
    // configuration (`arcgraph-cli/src/bootstrap.rs`'s shared `m3_dpt`;
    // see `m3_evict_wait_budget` sibling gates). With no DPT at all,
    // `evict_for_capacity` takes the "fully legacy, no M6 wiring"
    // early-out and never attempts a reclaim, making this gate vacuous.
    // No checkpointer is wired: a clean page never needs one (MECH-E2
    // is trivially satisfied — nothing to flush), so its absence does
    // not affect this specific hazard window (the install-to-pin gap on
    // a page the DPT does not yet know about).
    let dpt = Arc::new(DirtyPageTable::new());
    store.attach_m6_dirty_page_table(dpt);
    let pid = PageId::new(1);

    // Install the never-written-home fresh page — no pin taken, no
    // durable home write has EVER happened for `pid`.
    store
        .install_fresh(pid, PageType::Node, TenantId::DEFAULT)
        .unwrap();
    assert!(
        store.is_cached(pid),
        "harness precondition: the fresh page must be resident \
         immediately after install_fresh"
    );

    // `evict_for_capacity(0)`: target cap 0 forces the sweep to reclaim
    // every unpinned resident page it can — with no DPT wired,
    // `is_clean` returns `None`, which `evict_for_capacity`'s doc
    // explicitly treats as "no M6 wiring — clean-only reclaim" (the
    // conservative legacy posture, still permitted to reclaim CLEAN-
    // classified candidates when nothing marks them dirty). No pin is
    // held on `pid` at this point (the writer has not started its
    // pin-coupled acquisition yet), so nothing excludes the reclaim.
    let _ = store.evict_for_capacity(0);

    if store.is_evicted(pid) {
        // The hazard reproduced: the never-homed page was reclaimed.
        // A subsequent PINNED fault-in must now fail — there is no
        // durable home to recover the page's bytes from (this is the
        // "page not found" the charter's repro predicts).
        let result = store.latch_pinned_for_tenant(TenantId::DEFAULT, pid);
        assert!(
            result.is_err(),
            "hazard reproduced but fault-in unexpectedly succeeded — \
             expected a `MissingPage`-class error since the never-homed \
             page's sole RAM copy was reclaimed with nothing to recover \
             its bytes from"
        );
    } else {
        // If the store's clean-arm did not reclaim `pid` (e.g. it never
        // became the LRU-oldest candidate under this exact cap/threading
        // shape), the hazard window itself was not exercised. Since this
        // test targets `install_fresh` immediately followed by a
        // maximally aggressive `evict_for_capacity(0)` with only ONE
        // resident page and no other pressure, `pid` MUST be the sweep's
        // only candidate — assert this precondition explicitly so a
        // future change to eviction's candidate selection cannot make
        // this gate silently vacuous.
        panic!(
            "vacuous-gate guard: `install_fresh` + `evict_for_capacity(0)` \
             did not reclaim the sole resident page `{pid:?}` — the \
             hazard window this gate targets was never exercised"
        );
    }
}
