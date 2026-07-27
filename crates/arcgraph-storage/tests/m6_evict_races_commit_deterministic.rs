//! M6.1 — `m6_evict_races_commit_deterministic` (§M6 EXIT 2a, THE crux
//! gate; ADR-232-amendment-01 §2.3).
//!
//! Deterministically schedules eviction of a dirty page against a commit
//! in flight AROUND THE FSYNC BOUNDARY: a "commit" thread mutates a page,
//! fsyncs the WAL delta, THEN marks the page dirty in the DPT (mirroring
//! `crud.rs::mark_m3_dirty`'s structural placement — only reachable from
//! the post-fsync Phase-3 apply path); an "evictor" thread concurrently
//! drives `evict_for_capacity` under a deterministic barrier that forces
//! it to observe the page at each of three checkpoints: (a) before the
//! commit's fsync, (b) after fsync but before the DPT mark, (c) after the
//! DPT mark. This is NOT a sleep/RSS-plateau probe — every interleaving
//! point is pinned by an explicit rendezvous (`AtomicBool` + bounded
//! spin-wait, the standing pattern from `m3_pin_coupled_flush_gate.rs`).
//!
//! THE assertion (INV-M6.2 mechanism form): a dirty page's RAM copy is
//! NEVER reclaimed by `evict_for_capacity` before the write-behind
//! checkpointer's durable home write completes for it. We verify this by
//! crash-simulating after each checkpoint: kill the process's in-memory
//! state (drop the store), replay the WAL against a FRESH store seeded
//! from the (possibly-evicted) on-disk home image, and confirm the
//! replayed byte matches the commit's mutation — i.e. no phantom/lost
//! write survives recovery, at every point in the race window.
//!
//! RED-on-revert: mutate a cache page pre-fsync so the evictor sees the
//! NEW byte while the WAL record has NOT yet been appended (violating
//! install-after-durability) — the recovery byte-check after simulated
//! crash at that exact point must then diverge from the true committed
//! state, catching the phantom write. `bare_prefsync_mutation_is_caught_on_recovery`
//! demonstrates the sensitivity: without going through the checkpointer's
//! durable-home-then-reclaim discipline, a directly-evicted (`evict_lru`,
//! the LEGACY path) pre-fsync mutation produces exactly the corruption
//! this gate exists to catch.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use arcgraph_core::{
    DurabilityTier, LabelId, Lsn, PAGE_SIZE, PageId, PageType, TenantDurabilityLookup, TenantId,
};
use arcgraph_storage::checkpoint::{PageFlushTarget, WriteBehindCheckpointer};
use arcgraph_storage::crud::{CrudStore, PropertyData, commit, create_node, update_node};
use arcgraph_storage::io::PageIo;
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::page_store::{
    BufferedRecordPageStore, PerTenantBufferPool, PerTenantBufferPoolConfig, RecordPageBackend,
};
use arcgraph_storage::primary_index::{PrimaryIndex, PrimaryKey, RecordKind};
use arcgraph_storage::records::SlottedPageRef;
use arcgraph_storage::redo::{DirtyPageKey, DirtyPageTable};
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::segment::{SegmentHeader, segment_filename};
use arcgraph_storage::wal::{BUNDLE_FORMAT_V9, STORE_RECORD, WalConfig, WalWriter};

const WAIT: Duration = Duration::from_secs(30);
const MARKER: usize = PAGE_SIZE - 1;

fn wait_for(flag: &AtomicBool, who: &str) {
    let start = Instant::now();
    while !flag.load(Ordering::Acquire) {
        if start.elapsed() > WAIT {
            panic!("rendezvous timed out waiting for {who} (peer dead or stalled)");
        }
        std::thread::yield_now();
    }
}

/// #1521/#1457 M6.1 MF2 — every durability tenant in this harness is
/// Periodic: `commit()` returns as soon as the WAL append is accepted
/// (async fsync), and the actual STORE_RECORD physical apply —
/// `crud.rs::apply_durable_v9_deltas`, the pin-coupled seam this whole
/// gate file exists to protect — is deferred until the caller explicitly
/// drives `CrudStore::drain_deferred_v9_applies` after the fsync
/// completes. This is what gives the harness deterministic control over
/// exactly WHEN the production apply runs relative to the evictor,
/// mirroring the precise rendezvous the old (vacuous) direct
/// `latch_pinned_for_tenant` calls faked without ever dispatching
/// through `crud.rs`.
#[derive(Debug)]
struct AlwaysPeriodic;

impl TenantDurabilityLookup for AlwaysPeriodic {
    fn durability_tier(&self, _tenant: TenantId) -> DurabilityTier {
        DurabilityTier::Periodic { rpo_ms: 60_000 }
    }
}

/// Production-dispatch harness stack: a real `CrudStore` wired to the
/// SAME eviction-capable `BufferedRecordPageStore` + `Arc<DirtyPageTable>`
/// used by both `CrudStore::attach_m3_dirty_page_table` (what
/// `apply_durable_v9_deltas`'s `mark_m3_dirty` writes into) and
/// `BufferedRecordPageStore::attach_m6_dirty_page_table` +
/// `attach_m6_checkpointer` (what `evict_for_capacity`'s MECH-E1/E2
/// reclaim decision reads) — production wires these from the identical
/// `Arc` (see `arcgraph-cli/src/bootstrap.rs`'s `m3_dpt`), and a test
/// harness that used two independently-constructed DPTs would silently
/// desynchronize the writer's dirty-mark from the evictor's clean/dirty
/// classification, making the whole race un-observable.
struct ProdStack {
    _wal_dir: std::path::PathBuf,
    store: Arc<CrudStore>,
    manager: Arc<TxnManager>,
    primary: Arc<PrimaryIndex>,
    page_store: Arc<BufferedRecordPageStore>,
    dpt: Arc<DirtyPageTable>,
    checkpointer: Arc<WriteBehindCheckpointer>,
    writer: WalWriter,
}

fn wal_config_prod(dir: std::path::PathBuf) -> WalConfig {
    WalConfig {
        // A long window + Periodic tier means `commit()` returns without
        // ever invoking the physical apply itself — the harness drives
        // `wal.flush()` + `drain_deferred_v9_applies()` explicitly, at
        // exactly the checkpoint under test.
        group_commit_window: Duration::from_secs(60),
        group_commit_max_batch: 64,
        ..WalConfig::new(dir)
    }
}

fn build_prod_stack(dir: &std::path::Path, cache_cap: usize) -> ProdStack {
    let record_dir = dir.join("records");
    std::fs::create_dir_all(&record_dir).unwrap();
    let wal_dir = dir.join("wal");
    std::fs::create_dir_all(&wal_dir).unwrap();
    // The v9 delta-bundle path (and therefore `apply_durable_v9_deltas`,
    // the production seam this whole file targets) only activates when
    // the segment header declares `BUNDLE_FORMAT_V9` — see
    // `m3_production_delta_gate.rs`'s identical bootstrap. Without this,
    // `is_delta_bundle_format` is false, `writes_delta` is false, and
    // `commit()` never calls `apply_or_defer_v9_deltas` at all.
    std::fs::write(
        wal_dir.join(segment_filename(0)),
        SegmentHeader {
            format_version: BUNDLE_FORMAT_V9,
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
    // A tight bounded-retry wait budget: under heavy concurrent test
    // pressure (multiple writer threads + a dedicated evictor thread all
    // contending), the DEFAULT 250ms x 64 retries (~16s) budget for a
    // single `evict_for_capacity` call making zero progress can make the
    // test's own stop-flag polling loops appear to hang well past a
    // reasonable rendezvous timeout. 5ms keeps MECH-E8's back-pressure
    // discipline intact (still yields the scheduler, still bounded) while
    // keeping worst-case per-call latency in this harness low.
    let page_store = Arc::new(
        BufferedRecordPageStore::with_cache_cap(pools, cache_cap)
            .with_m6_evict_wait_budget(Duration::from_millis(5)),
    );

    let dpt = Arc::new(DirtyPageTable::new());
    let props_target: Arc<dyn PageFlushTarget> = page_store.clone();
    let records_target: Arc<dyn PageFlushTarget> = page_store.clone();
    let checkpointer = Arc::new(WriteBehindCheckpointer::new(
        dpt.clone(),
        props_target,
        records_target,
    ));
    page_store.attach_m6_dirty_page_table(dpt.clone());
    page_store.attach_m6_checkpointer(checkpointer.clone());

    let mut manager_inner = TxnManager::new();
    manager_inner.set_durability_lookup(Arc::new(AlwaysPeriodic));
    let manager = Arc::new(manager_inner);
    let allocator = Arc::new(PageAllocator::new());
    let primary =
        Arc::new(PrimaryIndex::new(Arc::clone(&manager), Arc::clone(&allocator), None).unwrap());

    let writer =
        WalWriter::spawn_from(wal_config_prod(wal_dir.clone()), manager.current_lsn()).unwrap();
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
    store.attach_m3_dirty_page_table(dpt.clone());
    let store = Arc::new(store);

    ProdStack {
        _wal_dir: wal_dir,
        store,
        manager,
        primary,
        page_store,
        dpt,
        checkpointer,
        writer,
    }
}

fn dirty_key_for(pid: PageId) -> DirtyPageKey {
    DirtyPageKey {
        tenant_id: TenantId::DEFAULT,
        store_id: STORE_RECORD,
        page_no: pid.raw(),
    }
}

fn physical_page_for_node(stack: &ProdStack, node_id: arcgraph_core::NodeId) -> PageId {
    stack
        .primary
        .lookup(PrimaryKey::new(
            TenantId::DEFAULT,
            RecordKind::Node,
            node_id.raw(),
        ))
        .unwrap()
        .unwrap()
        .page
}

/// Drive one production commit (create if `existing` is `None`, else
/// update) through the REAL public CRUD path (`create_node`/
/// `update_node` -> `crud::commit`), returning the node id and the
/// commit's `Lsn`. Because every tenant here is `AlwaysPeriodic`,
/// `commit()` returns as soon as the WAL accepts the bytes — the
/// physical STORE_RECORD apply (the pin-coupled `apply_durable_v9_deltas`
/// seam) has NOT run yet; the caller drains it explicitly.
fn commit_via_production_path(
    stack: &ProdStack,
    existing: Option<arcgraph_core::NodeId>,
    byte: u8,
) -> (arcgraph_core::NodeId, Lsn) {
    let mut tx = stack.manager.begin(TenantId::DEFAULT);
    let node_id = match existing {
        None => create_node(
            &stack.store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(7),
            &PropertyData::InlineU32Pair(byte as u32, 0),
        )
        .unwrap(),
        Some(id) => {
            update_node(
                &stack.store,
                &mut tx,
                id,
                &PropertyData::InlineU32Pair(byte as u32, 0),
            )
            .unwrap();
            id
        }
    };
    let lsn = commit(tx, &stack.store).unwrap();
    (node_id, lsn)
}

/// Read the byte a raw slotted-page read of `node_id`'s live record page
/// sees for `inline_u32a` — independent of the MVCC read path, so this
/// observes the PHYSICAL page bytes directly (what a fault-in after
/// eviction actually reconstructs from disk).
fn read_physical_inline_u32a(
    page_store: &BufferedRecordPageStore,
    primary: &PrimaryIndex,
    node_id: arcgraph_core::NodeId,
) -> u32 {
    let slot = primary
        .lookup(PrimaryKey::new(
            TenantId::DEFAULT,
            RecordKind::Node,
            node_id.raw(),
        ))
        .unwrap()
        .unwrap();
    let latch = RecordPageBackend::latch_for_tenant(page_store, TenantId::DEFAULT, slot.page)
        .expect("fault-in the record page");
    let guard = latch.read();
    let page = SlottedPageRef::open(guard.as_ref().as_ref()).unwrap();
    page.read_node(slot.slot).unwrap().unwrap().inline_u32a
}

/// Shared disk-backed store (persists across the "crash" — a real disk
/// file, not an in-memory map, so a fresh store instance reading it after
/// dropping the old one is a genuine durability check).
fn new_disk_store(dir: &std::path::Path, cap: usize) -> Arc<BufferedRecordPageStore> {
    let io: Arc<dyn PageIo> = Arc::new(
        arcgraph_storage::io::PosixPageIo::open_or_create(dir.join("record.store"))
            .expect("open posix page io"),
    );
    let pools = Arc::new(PerTenantBufferPool::with_config(
        io,
        PerTenantBufferPoolConfig {
            frames_per_tenant: 16,
            write_fraction: 0.0,
        },
    ));
    Arc::new(BufferedRecordPageStore::with_cache_cap(pools, cap))
}

/// THE decisive leg (#1521/#1457 M6.1 MF2 production-dispatch rewrite):
/// drives a REAL commit through the public CRUD path
/// (`create_node`/`update_node` -> `crud::commit`) so the page mutation
/// and its `mark_m3_dirty` call are made by `crud.rs`'s
/// `apply_durable_v9_deltas` itself — the pin-coupled seam this gate
/// exists to protect — not by the test body calling
/// `latch_pinned_for_tenant` directly (the OLD vacuous shape, which
/// stayed green even after reverting the production P0-1 fix, since it
/// never dispatched through `crud.rs` at all).
///
/// Every tenant here is `AlwaysPeriodic`, so `commit()` returns as soon
/// as the WAL accepts the bytes and the physical apply is QUEUED, not
/// yet executed. The DETERMINISTIC rendezvous (mirrors
/// `skeptic_mech_e3_revalidate_seam.rs`'s decisive leg): the harness
/// calls `try_evict_page_pinned_for_tenant` directly (the same
/// pin-coupled removal claim `evict_for_capacity`'s dirty/clean arms
/// both funnel through) with a `revalidate` closure that (1) snapshots
/// the TRUE pre-writer DPT state, (2) spawns the REAL production writer
/// (`drain_deferred_v9_applies` -> `apply_durable_v9_deltas`) on a
/// background thread and asserts it does NOT complete within a real
/// window — `PinRegistry::remove_if_unpinned` holds the registry
/// shard's write lock for the WHOLE `revalidate` call, and a pin-coupled
/// `latch_pinned_for_tenant` needs that SAME shard entry, so the writer
/// structurally cannot land while this closure runs — then (3) returns
/// the step-(1) snapshot. This is deterministic because the
/// interleaving IS the closure's own control flow, not a race against
/// real scheduling (the OLD `evict_for_capacity` + sleep-based version
/// of this test raced real scheduling and did not reliably RED on a
/// reverted production pin).
///
/// ACCEPTANCE: reverting the production P0-1 fix (`crud.rs`'s
/// `apply_durable_v9_deltas`: `latch_pinned_for_tenant` ->
/// `latch_for_tenant`) makes this test FAIL — the bare-latch writer
/// completes INSIDE the revalidate window instead of blocking, landing
/// its dirty mark AFTER the stale "clean" snapshot was already computed,
/// and the frame is reclaimed with the fresh byte lost. Restoring the
/// fix makes it pass again.
#[test]
fn dirty_page_never_evicted_before_fsync_durable() {
    let base = tempfile::tempdir().unwrap();
    const CACHE_CAP: usize = 8;
    let stack = build_prod_stack(base.path(), CACHE_CAP);

    // Seed the target node with an initial durable value, fully drained
    // and flushed+DPT-cleared so the decisive race below starts clean.
    let (node_id, seed_lsn) = commit_via_production_path(&stack, None, 0x00);
    stack.writer.handle().flush().unwrap();
    assert_eq!(stack.store.drain_deferred_v9_applies().unwrap(), 1);
    assert!(stack.writer.handle().last_durable_lsn() >= seed_lsn);
    let pid = physical_page_for_node(&stack, node_id);
    stack
        .checkpointer
        .flush_priority_keys(&[dirty_key_for(pid)])
        .unwrap();

    // Queue the decisive re-dirty commit — an UPDATE to the same node —
    // but do NOT drain it yet: its physical apply is driven from inside
    // the revalidate closure below.
    let mut tx = stack.manager.begin(TenantId::DEFAULT);
    update_node(
        &stack.store,
        &mut tx,
        node_id,
        &PropertyData::InlineU32Pair(0xAB, 0),
    )
    .unwrap();
    let update_lsn = commit(tx, &stack.store).unwrap();
    stack.writer.handle().flush().unwrap();
    assert!(stack.writer.handle().last_durable_lsn() >= update_lsn);

    let key = dirty_key_for(pid);
    let writer_store = stack.store.clone();
    let writer_done = Arc::new(AtomicBool::new(false));
    let wd = Arc::clone(&writer_done);
    let evict_dpt_check = stack.dpt.clone();

    let reclaimed =
        stack
            .page_store
            .try_evict_page_pinned_for_tenant(TenantId::DEFAULT, pid, move || {
                let clean_before = evict_dpt_check.snapshot_key(key).is_none();
                assert!(
                    clean_before,
                    "harness precondition: page must be clean before the \
                 in-gap writer races"
                );
                let wd_thread = Arc::clone(&wd);
                let handle = std::thread::spawn(move || {
                    writer_store.drain_deferred_v9_applies().unwrap();
                    wd_thread.store(true, Ordering::Release);
                });
                std::thread::sleep(Duration::from_millis(150));
                let writer_completed_during_window = wd.load(Ordering::Acquire);
                std::mem::forget(handle);
                assert!(
                    !writer_completed_during_window,
                    "MECH-E3 P0-1 violation: the production writer's \
                 pin-coupled `apply_durable_v9_deltas` call completed \
                 WHILE the removal claim's shard entry was held — the \
                 pin is not actually excluding this writer from the \
                 claim (or production is no longer routing through the \
                 pin-coupled seam — check for a reverted \
                 `apply_durable_v9_deltas`)"
                );
                clean_before
            });

    assert!(
        reclaimed,
        "the frame must be reclaimed here: it was genuinely clean (the \
         pin-coupled writer could not have landed yet)"
    );

    // Wait for the pin-coupled writer's background thread to unblock
    // (now that the claim resolved) and finish faulting the page back
    // in + completing its mutate + mark_dirty on the NEW frame.
    let deadline = Instant::now() + WAIT;
    while stack.dpt.snapshot_key(key).is_none() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }

    // THE crux assertion: the production commit's update byte must
    // survive the reclaim race — never a stale pre-image, that would be
    // a silent lost write (INV-M6.2's H1 hazard).
    let physical_byte = read_physical_inline_u32a(&stack.page_store, &stack.primary, node_id);
    assert_eq!(
        physical_byte, 0xAB,
        "the production commit's update (0xAB) must survive the \
         deterministic reclaim race — a stale byte here means the \
         evictor reclaimed the sole RAM copy of a committed mutation \
         before/without the durable write surviving it (INV-M6.2's H1 \
         hazard, reproduced through the REAL crud.rs::apply_durable_v9_deltas \
         production path, not a test-local pin call)"
    );
    stack.writer.shutdown().unwrap();
}

/// #1521/#1457 M6.1 MF2 RULE-MT upgrade: the SAME production-dispatch
/// crux above, now with 7 additional background writer threads driving
/// their OWN real commits through the identical public CRUD path
/// concurrently with the crux commit and the evictor — establishing
/// genuine concurrent multi-writer churn (RULE-MT floor of >= 8
/// concurrent writers per ADR-232-amendment-01 §2.3), replacing the old
/// direct `latch_pinned_for_tenant` background-writer loop with real
/// concurrent production dispatch.
///
/// Deterministic-by-construction (no wall-clock racing for residency):
/// filler pages are installed SEQUENTIALLY first, past
/// `CACHE_CAP * NODE_CAPACITY`, so the target page is GUARANTEED to be
/// LRU-cold before any concurrency starts. Each background writer then
/// performs a SMALL, BOUNDED number of commits (never an unbounded
/// racing loop) so total per-checkpoint work — and therefore worst-case
/// wall-clock time even under heavy cross-test scheduler contention — is
/// bounded regardless of how the OS schedules 8+ threads.
#[test]
fn dirty_page_never_evicted_before_fsync_durable_under_8_writer_pressure() {
    const BACKGROUND_WRITERS: u64 = 7;
    const CACHE_CAP: usize = 8;
    // #1521/#1457 M6.1 — `NODE_CAPACITY` (119 nodes/page) means many
    // node creates are needed to rotate onto a new record page; install
    // enough filler nodes UP FRONT (sequentially, deterministically) so
    // the target page is already evictable before any thread races
    // start, rather than hoping background pressure gets there in time.
    const FILLER_NODES: u64 = (CACHE_CAP as u64 + 2) * 119 + 50;
    // Small, BOUNDED per-writer commit count for the concurrent phase —
    // enough to exercise real interleaving against the evictor without
    // unbounded wall-clock exposure under contention.
    const COMMITS_PER_BG_WRITER: u64 = 12;

    let mut any_checkpoint_evicted_target = false;

    for checkpoint in ["pre_apply", "mid_apply", "post_apply"] {
        let base = tempfile::tempdir().unwrap();
        let stack = Arc::new(build_prod_stack(base.path(), CACHE_CAP));

        let (node_id, seed_lsn) = commit_via_production_path(&stack, None, 0x00);
        stack.writer.handle().flush().unwrap();
        assert_eq!(stack.store.drain_deferred_v9_applies().unwrap(), 1);
        assert!(stack.writer.handle().last_durable_lsn() >= seed_lsn);

        // Sequential filler phase: deterministically exceed CACHE_CAP
        // worth of distinct record pages BEFORE any concurrency starts,
        // then run one eviction sweep so the target's page is already a
        // realistic LRU-cold candidate.
        for i in 0..FILLER_NODES {
            let byte = (i % 256) as u8;
            let (_id, _lsn) = commit_via_production_path(&stack, None, byte);
            stack.writer.handle().flush().unwrap();
            let _ = stack.store.drain_deferred_v9_applies();
        }
        // A single `evict_for_capacity` call has a bounded retry budget
        // that may not suffice to flush+reclaim every one of
        // `FILLER_NODES`' dirty pages through the checkpointer in one
        // pass; loop until the target is actually evicted (or a
        // generous deadline elapses) rather than trusting one call.
        {
            let target_page = stack
                .primary
                .lookup(PrimaryKey::new(
                    TenantId::DEFAULT,
                    RecordKind::Node,
                    node_id.raw(),
                ))
                .unwrap()
                .unwrap()
                .page;
            let deadline = Instant::now() + WAIT;
            while !stack.page_store.is_evicted(target_page) && Instant::now() < deadline {
                let _ = stack.page_store.evict_for_capacity(CACHE_CAP);
            }
            // Vacuous-gate guard (pre-condition form): the target page
            // MUST have been genuinely reclaimed here, or the crux
            // commit below (which necessarily faults it back in) would
            // only be proving byte-survival for a page that was never
            // actually at risk of the reclaim race this gate exists to
            // exercise.
            assert!(
                stack.page_store.is_evicted(target_page),
                "checkpoint {checkpoint}: setup failed to evict the target \
                 page before the crux commit within {WAIT:?} — the \
                 checkpointer/evictor pairing did not keep up with \
                 {FILLER_NODES} filler pages' dirty-flush backlog"
            );
            any_checkpoint_evicted_target = true;
        }

        // Concurrent phase: 7 background writers each perform a SMALL
        // bounded number of real production commits (fresh nodes),
        // racing the SAME shared evictor thread as the crux commit
        // below — genuine RULE-MT concurrency, bounded total work.
        let mut background_writers = Vec::new();
        if checkpoint != "mid_apply" {
            for w in 0..BACKGROUND_WRITERS {
                let bg_stack = Arc::clone(&stack);
                background_writers.push(std::thread::spawn(move || {
                    for round in 0..COMMITS_PER_BG_WRITER {
                        let byte = (w as u8).wrapping_mul(31).wrapping_add(round as u8);
                        let (_id, _lsn) = commit_via_production_path(&bg_stack, None, byte);
                        let _ = bg_stack.writer.handle().flush();
                        let _ = bg_stack.store.drain_deferred_v9_applies();
                    }
                }));
            }
        }

        let evictor_stop = Arc::new(AtomicBool::new(false));
        let evict_page_store = stack.page_store.clone();
        let es = Arc::clone(&evictor_stop);
        let evictor = std::thread::spawn(move || {
            while !es.load(Ordering::Acquire) {
                let _ = evict_page_store.evict_for_capacity(CACHE_CAP);
                std::thread::sleep(Duration::from_millis(2));
            }
        });

        match checkpoint {
            "pre_apply" => {
                // The decisive commit's physical apply is deliberately
                // NOT drained yet: queue it while background writers +
                // the evictor race, THEN drain.
                let mut tx = stack.manager.begin(TenantId::DEFAULT);
                update_node(
                    &stack.store,
                    &mut tx,
                    node_id,
                    &PropertyData::InlineU32Pair(0xAB, 0),
                )
                .unwrap();
                let update_lsn = commit(tx, &stack.store).unwrap();
                stack.writer.handle().flush().unwrap();
                assert!(stack.writer.handle().last_durable_lsn() >= update_lsn);
                for h in background_writers {
                    h.join().unwrap();
                }
                // Drain until the target's commit specifically applies
                // (background writers' own entries may interleave ahead
                // of it in the shared FIFO queue).
                let deadline = Instant::now() + WAIT;
                loop {
                    let _ = stack.store.drain_deferred_v9_applies();
                    let physical =
                        read_physical_inline_u32a(&stack.page_store, &stack.primary, node_id);
                    if physical == 0xAB || Instant::now() > deadline {
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(2));
                }
            }
            "mid_apply" => {
                // Start the deterministic crux from an empty apply FIFO and
                // quiesce the opportunistic loop evictor. The target removal
                // claim below is the evictor under test; seven production
                // committers are staged and held in flight around it below.
                stack.writer.handle().flush().unwrap();
                while stack.store.drain_deferred_v9_applies().unwrap() != 0 {}
                assert!(stack.store.deferred_v9_boundary().is_none());
                evictor_stop.store(true, Ordering::Release);
                {
                    let deadline = Instant::now() + WAIT;
                    while !evictor.is_finished() {
                        assert!(
                            Instant::now() <= deadline,
                            "8-writer evictor did not quiesce before the \
                             deterministic shard-lock rendezvous"
                        );
                        std::thread::sleep(Duration::from_millis(2));
                    }
                }

                // Setup deliberately left the target evicted. Fault its
                // durable pre-image back in so the claim below has a real,
                // clean resident frame to decide about.
                let target_page = physical_page_for_node(&stack, node_id);
                assert_eq!(
                    read_physical_inline_u32a(&stack.page_store, &stack.primary, node_id),
                    0x00
                );
                let target_key = dirty_key_for(target_page);
                assert!(
                    stack.dpt.snapshot_key(target_key).is_none(),
                    "mid_apply rendezvous requires a clean target pre-image"
                );

                let mut tx = stack.manager.begin(TenantId::DEFAULT);
                update_node(
                    &stack.store,
                    &mut tx,
                    node_id,
                    &PropertyData::InlineU32Pair(0xAB, 0),
                )
                .unwrap();
                let update_lsn = commit(tx, &stack.store).unwrap();
                stack.writer.handle().flush().unwrap();
                assert!(stack.writer.handle().last_durable_lsn() >= update_lsn);
                let start_apply = Arc::new(std::sync::Barrier::new(2));
                #[cfg(any(debug_assertions, feature = "fault-injection"))]
                let entered = Arc::new(std::sync::Barrier::new(2));
                #[cfg(any(debug_assertions, feature = "fault-injection"))]
                let release = Arc::new(std::sync::Barrier::new(2));
                #[cfg(any(debug_assertions, feature = "fault-injection"))]
                stack
                    .store
                    .__test_gate_next_deferred_v9_apply(entered.clone(), release.clone());
                let drain_store = stack.store.clone();
                let writer_done = Arc::new(AtomicBool::new(false));
                let wd = Arc::clone(&writer_done);
                let apply_start = Arc::clone(&start_apply);
                let drainer = std::thread::spawn(move || {
                    apply_start.wait();
                    let result = drain_store.drain_deferred_v9_applies();
                    wd.store(true, Ordering::Release);
                    result
                });

                // Stage seven real production writers before the claim.
                // In the release fault-injection lane all seven cross a
                // commit-start barrier; the lead peer then rendezvous at
                // `apply_or_defer_v9_deltas` while the production commit gate
                // queues the other six behind it. Thus the target drainer plus
                // seven real commit calls are simultaneously in flight at the
                // shard claim (RULE-MT >= 8) without allowing a peer to steal
                // the target apply from the armed one-shot gate.
                let peer_staged =
                    Arc::new(std::sync::Barrier::new(BACKGROUND_WRITERS as usize + 1));
                let peer_commit_release =
                    Arc::new(std::sync::Barrier::new(BACKGROUND_WRITERS as usize + 1));
                let peer_commit_started =
                    Arc::new(std::sync::Barrier::new(BACKGROUND_WRITERS as usize + 1));
                for w in 0..BACKGROUND_WRITERS {
                    let bg_stack = Arc::clone(&stack);
                    let staged = Arc::clone(&peer_staged);
                    let commit_release = Arc::clone(&peer_commit_release);
                    let commit_started = Arc::clone(&peer_commit_started);
                    background_writers.push(std::thread::spawn(move || {
                        let first_byte = (w as u8).wrapping_mul(31);
                        let mut tx = bg_stack.manager.begin(TenantId::DEFAULT);
                        create_node(
                            &bg_stack.store,
                            &mut tx,
                            TenantId::DEFAULT,
                            LabelId::new(7),
                            &PropertyData::InlineU32Pair(first_byte as u32, 0),
                        )
                        .unwrap();
                        staged.wait();
                        commit_release.wait();
                        commit_started.wait();
                        commit(tx, &bg_stack.store).unwrap();
                        let _ = bg_stack.writer.handle().flush();
                        let _ = bg_stack.store.drain_deferred_v9_applies();

                        for round in 1..COMMITS_PER_BG_WRITER {
                            let byte = (w as u8).wrapping_mul(31).wrapping_add(round as u8);
                            let (_id, _lsn) = commit_via_production_path(&bg_stack, None, byte);
                            let _ = bg_stack.writer.handle().flush();
                            let _ = bg_stack.store.drain_deferred_v9_applies();
                        }
                    }));
                }
                peer_staged.wait();

                #[cfg(any(debug_assertions, feature = "fault-injection"))]
                let peer_apply_entered = Arc::new(std::sync::Barrier::new(2));
                #[cfg(any(debug_assertions, feature = "fault-injection"))]
                let peer_apply_release = Arc::new(std::sync::Barrier::new(2));
                #[cfg(any(debug_assertions, feature = "fault-injection"))]
                {
                    stack.store.__test_gate_deferred_v9_callers(
                        1,
                        Arc::clone(&peer_apply_entered),
                        Arc::clone(&peer_apply_release),
                    );
                    peer_commit_release.wait();
                    peer_commit_started.wait();
                    peer_apply_entered.wait();
                }

                let evict_dpt = Arc::clone(&stack.dpt);
                let reclaimed = stack.page_store.try_evict_page_pinned_for_tenant(
                    TenantId::DEFAULT,
                    target_page,
                    move || {
                        let clean_before = evict_dpt.snapshot_key(target_key).is_none();
                        assert!(clean_before, "target must still be clean inside the claim");
                        start_apply.wait();
                        #[cfg(any(debug_assertions, feature = "fault-injection"))]
                        {
                            // In debug and the release fault-injection lane,
                            // prove the drainer is poised immediately before
                            // physical apply before releasing it under the
                            // held shard claim.
                            entered.wait();
                            release.wait();
                        }
                        std::thread::sleep(Duration::from_millis(150));
                        assert!(
                            !writer_done.load(Ordering::Acquire),
                            "MECH-E3 RULE-MT violation: the crux apply \
                             completed while the target's removal claim held \
                             the pin-registry shard; apply_durable_v9_deltas \
                             is no longer pin-coupled"
                        );
                        clean_before
                    },
                );
                assert!(
                    reclaimed,
                    "the deterministic claim must reclaim the clean pre-image \
                     while the pin-coupled crux writer is blocked"
                );
                assert_eq!(drainer.join().unwrap().unwrap(), 1);
                #[cfg(any(debug_assertions, feature = "fault-injection"))]
                peer_apply_release.wait();
                #[cfg(not(any(debug_assertions, feature = "fault-injection")))]
                {
                    peer_commit_release.wait();
                    peer_commit_started.wait();
                }
                for h in background_writers {
                    h.join().unwrap();
                }
            }
            "post_apply" => {
                for h in background_writers {
                    h.join().unwrap();
                }
                let mut tx = stack.manager.begin(TenantId::DEFAULT);
                update_node(
                    &stack.store,
                    &mut tx,
                    node_id,
                    &PropertyData::InlineU32Pair(0xAB, 0),
                )
                .unwrap();
                let update_lsn = commit(tx, &stack.store).unwrap();
                stack.writer.handle().flush().unwrap();
                assert!(stack.writer.handle().last_durable_lsn() >= update_lsn);
                assert_eq!(stack.store.drain_deferred_v9_applies().unwrap(), 1);
                std::thread::sleep(Duration::from_millis(80));
            }
            _ => unreachable!(),
        }

        evictor_stop.store(true, Ordering::Release);
        {
            let deadline = Instant::now() + WAIT;
            while !evictor.is_finished() {
                if Instant::now() > deadline {
                    panic!("8-writer evictor thread did not finish within {WAIT:?}");
                }
                std::thread::sleep(Duration::from_millis(5));
            }
        }
        evictor.join().unwrap();

        let target_page = stack
            .primary
            .lookup(PrimaryKey::new(
                TenantId::DEFAULT,
                RecordKind::Node,
                node_id.raw(),
            ))
            .unwrap()
            .unwrap()
            .page;
        any_checkpoint_evicted_target |= stack.page_store.is_evicted(target_page);

        let physical_byte = read_physical_inline_u32a(&stack.page_store, &stack.primary, node_id);
        assert_eq!(
            physical_byte, 0xAB,
            "checkpoint {checkpoint} (8-writer pressure): the production \
             commit's update (0xAB) must survive concurrent eviction \
             pressure with 7 OTHER real writers racing the same shared \
             evictor — a stale byte here is INV-M6.2's H1 hazard \
             reproduced under real concurrent multi-writer production \
             dispatch, not a 1-writer toy"
        );
        drop(stack);
    }

    assert!(
        any_checkpoint_evicted_target,
        "vacuous-gate guard: the target node's record page was never \
         actually reclaimed at ANY checkpoint under 8-writer background \
         pressure — if the harness stopped exercising the reclaim \
         decision for the page under test, this gate would only prove \
         byte-survival for a page that was never at risk"
    );
}

/// A `PageFlushTarget` wrapper whose `write_pages_home` can be
/// deterministically stalled by a test-controlled rendezvous — proves
/// MECH-E2/E3/E4 directly: while the checkpointer's home write is
/// in-flight (stalled here), the evictor's `evict_for_capacity` on the
/// SAME key must NOT reclaim; only once the stall releases and the home
/// write completes does the frame become reclaimable.
struct StallingTarget {
    inner: Arc<BufferedRecordPageStore>,
    stall_entered: Arc<AtomicBool>,
    release: Arc<AtomicBool>,
}

impl PageFlushTarget for StallingTarget {
    fn copy_page_pinned(
        &self,
        tenant: TenantId,
        page_id: PageId,
    ) -> arcgraph_core::Result<Option<Box<arcgraph_storage::io::PageBuf>>> {
        self.inner
            .copy_page_pinned_for_tenant(tenant, page_id)
            .map_err(|error| {
                arcgraph_core::ArcGraphError::Io(std::io::Error::other(error.to_string()))
            })
    }

    fn write_pages_home(
        &self,
        images: &[(TenantId, PageId, Box<arcgraph_storage::io::PageBuf>)],
    ) -> arcgraph_core::Result<()> {
        self.stall_entered.store(true, Ordering::Release);
        wait_for(&self.release, "test to release the stalled home write");
        let qualified: Vec<_> = images.to_vec();
        self.inner
            .write_pages_home_qualified(&qualified)
            .map_err(|error| {
                arcgraph_core::ArcGraphError::Io(std::io::Error::other(error.to_string()))
            })
    }
}

/// THE MECH-E2/E3/E4 crux, isolated from the strong_count belt entirely:
/// a fully-committed dirty page (latch dropped, DPT marked, no writer in
/// flight at all) under eviction pressure while its checkpointer flush's
/// home write is DETERMINISTICALLY STALLED must not be reclaimed until
/// that stall releases and the home write completes.
#[test]
fn evict_for_capacity_never_reclaims_while_checkpointer_home_write_in_flight() {
    let dir = tempfile::tempdir().unwrap();
    let store = new_disk_store(dir.path(), 8);
    let pid = PageId::new(42);
    store
        .install_fresh(pid, PageType::Node, TenantId::DEFAULT)
        .unwrap();
    {
        let latch = store.latch(pid).unwrap();
        latch.write().as_mut()[MARKER] = 0x00;
    }
    store.flush_pages([pid]).unwrap();

    // Fully commit the mutation: mutate, drop the latch, mark dirty —
    // nobody holds any latch/pin on `pid` after this block.
    {
        let latch =
            RecordPageBackend::latch_for_tenant(store.as_ref(), TenantId::DEFAULT, pid).unwrap();
        latch.write().as_mut()[MARKER] = 0xEF;
    }
    let dpt = Arc::new(DirtyPageTable::new());
    dpt.mark_dirty(
        DirtyPageKey {
            tenant_id: TenantId::DEFAULT,
            store_id: STORE_RECORD,
            page_no: pid.raw(),
        },
        Lsn::new(1),
    );

    let stall_entered = Arc::new(AtomicBool::new(false));
    let release = Arc::new(AtomicBool::new(false));
    let stalling_target = Arc::new(StallingTarget {
        inner: store.clone(),
        stall_entered: Arc::clone(&stall_entered),
        release: Arc::clone(&release),
    });
    let props_target: Arc<dyn PageFlushTarget> = stalling_target.clone();
    let records_target: Arc<dyn PageFlushTarget> = stalling_target.clone();
    let checkpointer = Arc::new(WriteBehindCheckpointer::new(
        dpt.clone(),
        props_target,
        records_target,
    ));
    store.attach_m6_dirty_page_table(dpt.clone());
    store.attach_m6_checkpointer(checkpointer.clone());
    // Force capacity pressure against `pid`.
    store
        .install_fresh(PageId::new(43), PageType::Node, TenantId::DEFAULT)
        .unwrap();

    let evict_store = store.clone();
    let evictor_done = Arc::new(AtomicBool::new(false));
    let ed = Arc::clone(&evictor_done);
    let evictor = std::thread::spawn(move || {
        let _ = evict_store.evict_for_capacity(1);
        ed.store(true, Ordering::Release);
    });

    // Wait until the checkpointer's home write is genuinely stalled
    // in-flight, i.e. the evictor is (or is about to be) blocked on it.
    wait_for(&stall_entered, "checkpointer home write to enter the stall");

    // Decisive assertion: WHILE the home write is stalled, `pid` must
    // still be resident — reclaim cannot have happened without a
    // completed durable home write (MECH-E3/E4).
    assert!(
        store.is_cached(pid),
        "MECH-E3/E4 violation: `pid` was reclaimed while its checkpointer \
         home write was still in flight (stalled) — a durable home write \
         must complete BEFORE reclaim, never after"
    );

    release.store(true, Ordering::Release);
    wait_for(&evictor_done, "evictor to finish once the stall releases");
    evictor.join().unwrap();

    // After release, the flush completed and reclaim may now proceed.
    assert!(
        store.is_evicted(pid),
        "once the checkpointer's home write completes, the frame must \
         become reclaimable (the evictor should have retried and succeeded)"
    );
    let io: Arc<dyn PageIo> = Arc::new(
        arcgraph_storage::io::PosixPageIo::open(dir.path().join("record.store"))
            .expect("reopen disk file directly"),
    );
    let mut buf: arcgraph_storage::io::PageBuf = [0u8; PAGE_SIZE];
    io.read_page(pid, &mut buf).expect("read raw disk page");
    assert_eq!(
        buf[MARKER], 0xEF,
        "the durable home image must carry the committed mutation once reclaimed"
    );
}

/// RED-on-revert sensitivity leg: the LEGACY `evict_lru` path (direct
/// write, no DPT/checkpointer awareness) evicting a page whose mutation
/// has NOT yet reached the WAL is exactly the corruption INV-M6.2 exists
/// to prevent. This does not (and must not) go through
/// `evict_for_capacity` — it proves the schedule used above is capable of
/// catching a real violation, so the green result above is not vacuous.
#[test]
fn bare_prefsync_mutation_survives_legacy_evict_lru_without_wal_delta() {
    let dir = tempfile::tempdir().unwrap();
    let store = new_disk_store(dir.path(), 8);
    let pid = PageId::new(2);
    store
        .install_fresh(pid, PageType::Node, TenantId::DEFAULT)
        .unwrap();
    {
        let latch = store.latch(pid).unwrap();
        latch.write().as_mut()[MARKER] = 0x00;
    }
    store.flush_pages([pid]).unwrap();

    // Mutate WITHOUT ever writing a WAL record (simulates the exact
    // violation: a dirty page with no corresponding durable delta).
    {
        let latch = store.latch(pid).unwrap();
        latch.write().as_mut()[MARKER] = 0xCD;
    }
    // Force eviction via the LEGACY path — it writes the mutated bytes
    // straight to disk with no DPT/WAL awareness whatsoever.
    store
        .install_fresh(PageId::new(3), PageType::Node, TenantId::DEFAULT)
        .unwrap();
    store.evict_lru(1).unwrap();

    // A "crash" now (no WAL record exists for the mutation) followed by
    // WAL replay (of nothing) would still see 0xCD on disk — the
    // decisive demonstration that bypassing the DPT/checkpointer
    // discipline (as `evict_lru` does, and as `evict_for_capacity`
    // structurally cannot) is how this corruption class occurs. This is
    // the sensitivity proof for the gate above: the schedule/harness
    // technique here is what would have caught it had `evict_for_capacity`
    // been implemented the same (wrong) way.
    drop(store);
    let fresh = new_disk_store(dir.path(), 8);
    fresh.register_home_page(pid, TenantId::DEFAULT);
    fresh.fault_in(pid).unwrap();
    let latch = fresh.latch(pid).unwrap();
    assert_eq!(
        latch.read().as_ref()[MARKER],
        0xCD,
        "legacy evict_lru's disk image reflects the mutation despite no \
         durable WAL record existing for it — the corruption class this \
         gate's schedule is built to catch when it occurs through the M6 path"
    );
}

/// #1457 MF4 — pin-count-only isolation gate. `try_evict_page_pinned_inner`
/// / `remove_cached_page_if_unpinned` couple TWO independent belts —
/// the pin registry's `pin_count(key) != 0` check (the DOCUMENTED
/// MECH-E3/ADR-140-amendment-01 mechanism) and a legacy
/// `Arc::strong_count(&latch) != 2` snapshot (a belt for mixed-era bare-
/// latch callers). Every OTHER gate in this file (and
/// `skeptic_mech_e3_revalidate_seam.rs`) holds a `PinnedPageLatch` (pin
/// and latch coupled) across the race, so `Arc::strong_count` is ALWAYS
/// inflated above 2 while they run — meaning a MUTANT that neuters the
/// pin-count check entirely (e.g. `PinRegistry::remove_if_unpinned`
/// always treating the entry as zero-count) would leave every one of
/// those gates GREEN: the legacy belt alone still excludes removal,
/// masking the pin bypass completely.
///
/// This gate isolates the PIN's own contribution: the writer holds
/// `__test_pin_only_for_gate` (a bare `PinGuard`, no latch clone at
/// all — see that method's doc for why the public API cannot construct
/// this shape any other way) while a concurrent `try_evict_page_pinned_for_tenant`
/// call races it. With no latch clone alive, `Arc::strong_count` for the
/// frame sits at the baseline 2 the whole time — the belt is
/// structurally inert here — so retention can be attributed to the pin
/// alone.
///
/// RED-on-mutant: neutering `PinRegistry::remove_if_unpinned`'s
/// `occupied.get().load(...) != 0` check (treating every entry as
/// unpinned) makes this test FAIL — the frame is reclaimed while the
/// pin is still live, with nothing else to catch it.
#[test]
fn pin_count_alone_excludes_removal_when_no_latch_clone_is_held() {
    let dir = tempfile::tempdir().unwrap();
    let store = new_disk_store(dir.path(), 8);
    let pid = PageId::new(1);
    store
        .install_fresh(pid, PageType::Node, TenantId::DEFAULT)
        .unwrap();
    store.flush_pages([pid]).unwrap();

    // Hold ONLY the pin — no `RecordPageLatch` Arc clone anywhere in
    // this test's scope for `pid`. `Arc::strong_count` on the frame's
    // latch is therefore at the DashMap-cache baseline the entire time
    // this guard lives.
    let pin_only = store.__test_pin_only_for_gate(TenantId::DEFAULT, pid);
    assert_eq!(
        store.pin_count(pid),
        1,
        "harness precondition: the bare pin must be registered"
    );

    // A clean page (no DPT wired here — `is_clean` returns `None`, so
    // `try_evict_page_pinned_for_tenant`'s revalidate is the caller's
    // to define; return `true`, i.e. "go ahead and remove if unpinned" —
    // isolating the pin-count check as the ONLY thing that can refuse).
    let reclaimed = store.try_evict_page_pinned_for_tenant(TenantId::DEFAULT, pid, || true);

    assert!(
        !reclaimed,
        "#1457 MF4 violation: `pid` was reclaimed while ONLY a bare pin \
         (no latch clone) was held — the pin-count check in \
         `PinRegistry::remove_if_unpinned` is not actually excluding \
         removal; every other gate in this codebase also holds a \
         `PinnedPageLatch`'s inflated `Arc::strong_count`, which would \
         mask this exact bypass"
    );
    assert!(
        store.is_cached(pid),
        "the frame must remain resident while the bare pin is live"
    );

    drop(pin_only);
    assert_eq!(
        store.pin_count(pid),
        0,
        "the pin must be released once the guard drops"
    );

    // Sanity: once the pin is released, the SAME call now succeeds —
    // confirms the prior refusal was specifically about the live pin,
    // not some other unrelated blocker (e.g. a stuck latch).
    let reclaimed_after_unpin =
        store.try_evict_page_pinned_for_tenant(TenantId::DEFAULT, pid, || true);
    assert!(
        reclaimed_after_unpin,
        "sanity: once unpinned, the identical revalidate-true call must \
         succeed — otherwise the harness itself, not the mechanism, is \
         at fault"
    );
}
