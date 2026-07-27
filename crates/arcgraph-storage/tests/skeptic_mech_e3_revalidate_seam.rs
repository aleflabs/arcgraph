//! M6.1 (#1521 P0-1) — `skeptic_mech_e3_revalidate_seam`: THE deterministic
//! rendezvous gate for the MECH-E3 silent-durability-loss race the
//! adjudicator's skeptic PoC found (reproduced red in ~0.04s).
//!
//! `BufferedRecordPageStore::remove_cached_page_if_unpinned` decides
//! removal inside `PinRegistry::remove_if_unpinned`'s shard-locked
//! closure: `revalidate()` (the caller-supplied `is_clean`-shaped DPT
//! re-check) runs FIRST; only if it returns `true` does the closure
//! proceed to the strong-count belt and `cache.remove`. The gap this
//! gate targets is the window strictly BETWEEN `revalidate()` returning
//! `true` and `cache.remove` actually executing — if a writer's re-dirty
//! (DPT `mark_dirty`) lands in that exact window, the removal claim has
//! ALREADY committed to reclaiming and has no way to notice; the frame
//! (the writer's fresh bytes' sole RAM copy) is reclaimed anyway, and
//! the checkpointer subsequently homes the STALE disk image while
//! clearing the (about-to-be-written) DPT entry — a committed write
//! permanently lost, no crash required.
//!
//! Because `revalidate` is a caller-supplied `FnOnce` invoked exactly
//! once at that seam, this gate uses it as the deterministic rendezvous
//! point directly — no sleep, no OS-scheduling luck, no new production
//! hook: the revalidate closure (1) snapshots the TRUE pre-writer DPT
//! state (what a real `is_clean` call would have seen at that instant),
//! (2) synchronously drives the writer's dirty-marking step to
//! completion, THEN (3) returns the step-(1) snapshot — exactly modeling
//! "revalidate observed clean; a re-dirty landed immediately after,
//! before removal executed". This is deterministic because the
//! interleaving IS the closure's own control flow, not a race against
//! it.
//!
//! Two legs:
//!
//! - `bare_latch_redirty_after_revalidate_is_silently_lost` (RED-on-revert
//!   SENSITIVITY leg — proves the harness can catch the defect): the
//!   writer's dirty-mark runs via a BARE `latch_for_tenant`-mutated frame
//!   (what PRODUCTION used before this fix) with NOTHING excluding it
//!   from the concurrent removal claim. The frame is reclaimed (revalidate
//!   returned the pre-mark "clean" snapshot) and the writer's fresh byte
//!   is gone from what a subsequent fault-in reads — a stale disk image
//!   survives. This IS the exact defect class #1521 found; it is
//!   EXPECTED for the bare API, which is why v9-era dirty-marking writers
//!   must never use it (see page_store.rs's documented MUST) —
//!   demonstrated here so the decisive leg below is not a vacuous green.
//! - `pinned_redirty_after_revalidate_still_excludes_removal` (THE
//!   decisive leg, mirrors the FIXED production shape): the writer
//!   acquires through `latch_pinned_for_tenant` (what
//!   `crud.rs::apply_durable_v9_deltas` now uses for every STORE_RECORD
//!   dirty-marking mutation) BEFORE the revalidate closure even runs —
//!   `PinRegistry::pin` takes the SAME shard lock
//!   `remove_if_unpinned`'s claim already holds for this key, so the
//!   pinned writer's `pin()` call blocks until the claim's closure
//!   finishes (it cannot "land after revalidate but before removal" at
//!   all: the shard lock excludes it structurally, not by timing). The
//!   frame is retained (removal never proceeds while pinned), and the
//!   writer's mutation lands safely afterward. RED if the P0-1 fix is
//!   reverted: production would fall back to the bare API and this exact
//!   schedule reproduces the leg above's lost-write instead.

use std::sync::Arc;
use std::time::Duration;

use arcgraph_core::{
    DurabilityTier, LabelId, Lsn, PAGE_SIZE, PageId, PageType, TenantDurabilityLookup, TenantId,
};
use arcgraph_storage::checkpoint::{PageFlushTarget, WriteBehindCheckpointer};
use arcgraph_storage::crud::{CrudStore, PropertyData, commit, create_node};
use arcgraph_storage::io::{InMemoryPageIo, PageIo};
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

const MARKER: usize = PAGE_SIZE - 1;

fn new_store(cap: usize) -> Arc<BufferedRecordPageStore> {
    let io: Arc<dyn arcgraph_storage::io::PageIo> = Arc::new(InMemoryPageIo::new());
    let pools = Arc::new(PerTenantBufferPool::with_config(
        io,
        PerTenantBufferPoolConfig {
            frames_per_tenant: 16,
            write_fraction: 0.0,
        },
    ));
    Arc::new(BufferedRecordPageStore::with_cache_cap(pools, cap))
}

fn dirty_key(pid: PageId) -> DirtyPageKey {
    DirtyPageKey {
        tenant_id: TenantId::DEFAULT,
        store_id: STORE_RECORD,
        page_no: pid.raw(),
    }
}

/// #1457 MF2 — every tenant here is Periodic: `commit()` returns as soon
/// as the WAL append is accepted; the physical STORE_RECORD apply (the
/// `apply_durable_v9_deltas` seam this gate protects) is deferred until
/// the caller explicitly drives `drain_deferred_v9_applies`. This gives
/// the decisive leg's revalidate closure deterministic control over
/// exactly when the REAL production apply runs relative to the eviction
/// claim, without any sleep/scheduler luck.
#[derive(Debug)]
struct AlwaysPeriodic;

impl TenantDurabilityLookup for AlwaysPeriodic {
    fn durability_tier(&self, _tenant: TenantId) -> DurabilityTier {
        DurabilityTier::Periodic { rpo_ms: 60_000 }
    }
}

/// Production-dispatch harness for the decisive leg: a real `CrudStore`
/// wired to the SAME eviction-capable `BufferedRecordPageStore` +
/// `Arc<DirtyPageTable>` used by both `CrudStore::attach_m3_dirty_page_table`
/// (what `apply_durable_v9_deltas`'s `mark_m3_dirty` writes into) and
/// `BufferedRecordPageStore::attach_m6_dirty_page_table` (what the
/// evictor's revalidate closure reads) — mirrors production's shared
/// `m3_dpt` wiring (`arcgraph-cli/src/bootstrap.rs`).
struct ProdStack {
    store: Arc<CrudStore>,
    manager: Arc<TxnManager>,
    primary: Arc<PrimaryIndex>,
    page_store: Arc<BufferedRecordPageStore>,
    dpt: Arc<DirtyPageTable>,
    checkpointer: Arc<WriteBehindCheckpointer>,
    writer: WalWriter,
}

fn build_prod_stack(dir: &std::path::Path, cache_cap: usize) -> ProdStack {
    let record_dir = dir.join("records");
    std::fs::create_dir_all(&record_dir).unwrap();
    let wal_dir = dir.join("wal");
    std::fs::create_dir_all(&wal_dir).unwrap();
    // The v9 delta-bundle path (and therefore `apply_durable_v9_deltas`)
    // only activates when the segment header declares `BUNDLE_FORMAT_V9`
    // — see `m3_production_delta_gate.rs`'s identical bootstrap.
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
    let page_store = Arc::new(BufferedRecordPageStore::with_cache_cap(pools, cache_cap));

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

    let writer = WalWriter::spawn_from(
        WalConfig {
            group_commit_window: Duration::from_secs(60),
            group_commit_max_batch: 64,
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
    store.attach_m3_dirty_page_table(dpt.clone());
    let store = Arc::new(store);

    ProdStack {
        store,
        manager,
        primary,
        page_store,
        dpt,
        checkpointer,
        writer,
    }
}

/// Drive one real production commit (`create_node` -> `crud::commit`)
/// and leave its physical STORE_RECORD apply QUEUED (Periodic tier, not
/// yet drained) — the caller controls exactly when
/// `drain_deferred_v9_applies` (and therefore `apply_durable_v9_deltas`)
/// runs.
fn queue_production_commit(stack: &ProdStack, byte: u8) -> arcgraph_core::NodeId {
    let mut tx = stack.manager.begin(TenantId::DEFAULT);
    let node_id = create_node(
        &stack.store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(7),
        &PropertyData::InlineU32Pair(byte as u32, 0),
    )
    .unwrap();
    let lsn = commit(tx, &stack.store).unwrap();
    stack.writer.handle().flush().unwrap();
    assert!(stack.writer.handle().last_durable_lsn() >= lsn);
    node_id
}

fn physical_page_for(stack: &ProdStack, node_id: arcgraph_core::NodeId) -> PageId {
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

/// Sensitivity leg: the writer's dirty-mark rides a BARE latch (what
/// production used pre-fix) with no pin at all. The revalidate closure
/// snapshots "clean" (true, at that instant), synchronously drives the
/// bare-latch writer's mutate-then-mark-dirty to completion, THEN
/// returns the pre-writer snapshot — deterministically modeling a
/// re-dirty landing in the revalidate-to-removal gap. Demonstrates the
/// defect class #1521 found: the frame is reclaimed and the fresh byte
/// is lost.
#[test]
fn bare_latch_redirty_after_revalidate_is_silently_lost() {
    let store = new_store(8);
    let pid = PageId::new(1);
    store
        .install_fresh(pid, PageType::Node, TenantId::DEFAULT)
        .unwrap();
    // Seed + durably home the ORIGINAL byte so a stale-image reclaim is
    // observably different from the writer's fresh mutation below.
    {
        let latch = store.latch(pid).unwrap();
        latch.write().as_mut()[MARKER] = 0x00;
    }
    store.flush_pages([pid]).unwrap();

    let dpt = Arc::new(DirtyPageTable::new());
    let key = dirty_key(pid);
    let writer_store = store.clone();
    let writer_dpt = dpt.clone();

    let reclaimed = store.try_evict_page_pinned_for_tenant(TenantId::DEFAULT, pid, move || {
        // Step 1: the TRUE state at the instant revalidate is evaluated
        // — no writer has touched the page since the durable home write
        // above, so this is genuinely clean.
        let clean_before = writer_dpt.snapshot_key(key).is_none();
        assert!(
            clean_before,
            "harness precondition: page must start clean before the \
             in-gap writer runs"
        );
        // Step 2: synchronously drive the FULL bare-latch writer cycle
        // — mutate the frame, then mark it dirty — to completion,
        // landing exactly "after revalidate observed clean, before
        // removal executes". The bare latch registers no pin, so
        // nothing here is excluded by the claim this closure runs
        // inside of.
        {
            let latch =
                RecordPageBackend::latch_for_tenant(writer_store.as_ref(), TenantId::DEFAULT, pid)
                    .expect("bare latch");
            latch.write().as_mut()[MARKER] = 0xAB;
        }
        writer_dpt.mark_dirty(key, Lsn::new(1));
        // Step 3: return the STALE pre-writer snapshot — exactly what a
        // real revalidate() call evaluated microseconds earlier would
        // have returned, before this in-gap re-dirty landed.
        clean_before
    });

    assert!(
        reclaimed,
        "sensitivity leg: expected the frame to be reclaimed (revalidate \
         returned the stale 'clean' snapshot, and the bare-latch writer \
         registered no pin to exclude the claim) — if this is false the \
         harness itself is not exercising the seam this gate exists to \
         prove"
    );
    // Decisive check: fault back in and read the byte a subsequent
    // consumer would see. The reclaim happened while the DPT (at the
    // revalidate snapshot) still reported clean, so the checkpointer
    // would home the STALE image — a fault-in reads back 0x00, not the
    // writer's 0xAB, even though `mark_dirty` for 0xAB has already run.
    store.fault_in(pid).unwrap();
    let byte = store.latch(pid).unwrap().read().as_ref()[MARKER];
    assert_eq!(
        byte, 0x00,
        "MECH-E3 defect class reproduced: the bare-latch writer's fresh \
         mutation (0xAB) is GONE after reclaim — only the stale durably-\
         homed byte (0x00) survives, exactly #1521's silent lost-write \
         (a committed write permanently lost, no crash required)"
    );
}

/// THE decisive leg (#1457 MF2 production-dispatch rewrite): the SAME
/// deterministic rendezvous shape as the sensitivity leg above — a
/// QUEUED re-dirty commit whose physical apply is driven from WITHIN the
/// revalidate closure — but the writer now goes through the REAL,
/// UNMODIFIED `crud.rs::apply_durable_v9_deltas` production path, not a
/// test-local `latch_pinned_for_tenant` call.
///
/// `PinRegistry::remove_if_unpinned` holds the registry shard's
/// DashMap-entry WRITE lock for the ENTIRE duration of the `remove`
/// closure (which is where `revalidate` runs) — see pin.rs's own
/// "Ordering contract" doc. A pin-coupled writer's `latch_pinned_for_tenant`
/// calls `PinRegistry::pin`, which needs THAT SAME shard entry — so a
/// writer racing on a background thread cannot complete its pin (and
/// therefore cannot touch the frame, mutate it, or mark it dirty) until
/// the claim's closure returns. We drive the writer on a background
/// thread from inside revalidate and assert it is STILL BLOCKED when
/// revalidate returns — the structural proof that a pin-coupled writer
/// cannot land "invisibly" inside the claim the way the bare-latch
/// writer does in the sensitivity leg. The frame is then safely
/// reclaimed (genuinely clean at that instant), and the writer's pin
/// unblocks immediately after, faults in fresh bytes, and completes its
/// mutate + mark_dirty on the NEW resident frame — no write is lost.
///
/// ACCEPTANCE (the point of this rewrite): reverting the production pin
/// (`crud.rs::apply_durable_v9_deltas`: `latch_pinned_for_tenant` ->
/// `latch_for_tenant`, the shape at
/// `git show 752884d6^:crates/arcgraph-storage/src/crud.rs`) makes THIS
/// test FAIL — the bare-latch writer is NOT excluded by the claim (no
/// pin registered), so it completes its mutate+mark INSIDE the
/// revalidate window instead of blocking, and the frame is reclaimed
/// with the writer's fresh byte silently lost — exactly the sibling
/// sensitivity leg's schedule, now reproduced through the real
/// production entry point. The OLD version of this test called
/// `latch_pinned_for_tenant` directly in the test body, so it stayed
/// green even with the production fix reverted (it never dispatched
/// through `crud.rs` at all) — exactly the vacuousness #1490/#1457 MF2
/// found.
#[test]
fn pinned_redirty_after_revalidate_still_excludes_removal() {
    let dir = tempfile::tempdir().unwrap();
    let stack = build_prod_stack(dir.path(), 8);

    let seed_id = queue_production_commit(&stack, 0x00);
    assert_eq!(stack.store.drain_deferred_v9_applies().unwrap(), 1);
    let pid = physical_page_for(&stack, seed_id);
    // `flush_priority_keys` (not `flush_pages`) both writes the seed
    // home AND clears its DPT entry — required so the decisive commit
    // below starts from a genuinely clean DPT state, matching what the
    // production checkpointer's flush pass does.
    let seed_key = dirty_key(pid);
    stack.checkpointer.flush_priority_keys(&[seed_key]).unwrap();

    // Queue the decisive re-dirty commit — an UPDATE to the same node,
    // landing on the SAME physical page — but do NOT drain it yet: its
    // physical apply is driven from inside the revalidate closure below.
    {
        let mut tx = stack.manager.begin(TenantId::DEFAULT);
        arcgraph_storage::crud::update_node(
            &stack.store,
            &mut tx,
            seed_id,
            &PropertyData::InlineU32Pair(0xAB, 0),
        )
        .unwrap();
        let lsn = commit(tx, &stack.store).unwrap();
        stack.writer.handle().flush().unwrap();
        assert!(stack.writer.handle().last_durable_lsn() >= lsn);
    }

    let key = dirty_key(pid);
    let writer_store = stack.store.clone();
    let writer_done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let wd = Arc::clone(&writer_done);
    let evict_dpt = stack.dpt.clone();

    let reclaimed =
        stack
            .page_store
            .try_evict_page_pinned_for_tenant(TenantId::DEFAULT, pid, move || {
                // Step 1: the TRUE state at the instant revalidate runs —
                // the queued update has not been drained yet, so DPT is
                // genuinely clean for `pid` at this instant.
                let clean_before = evict_dpt.snapshot_key(key).is_none();
                assert!(
                    clean_before,
                    "harness precondition: page must be clean before the \
                 in-gap writer races"
                );
                // Step 2: spawn the REAL production writer on a background
                // thread — it must call `latch_pinned_for_tenant` (via
                // `apply_durable_v9_deltas`) which needs the SAME registry
                // shard entry this claim already holds write-locked.
                let wd_thread = Arc::clone(&wd);
                let handle = std::thread::spawn(move || {
                    writer_store.drain_deferred_v9_applies().unwrap();
                    wd_thread.store(true, std::sync::atomic::Ordering::Release);
                });
                // Step 3: give the writer thread a real window to run —
                // long enough that a NON-excluded (bare-latch) writer would
                // certainly have completed, but a pin-coupled writer cannot
                // possibly finish (it is blocked on this exact shard lock).
                std::thread::sleep(Duration::from_millis(150));
                let writer_completed_during_window = wd.load(std::sync::atomic::Ordering::Acquire);
                // Detach: joined after the claim resolves (a pin-coupled
                // writer only unblocks once this closure returns).
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
                // Step 4: return the genuinely-current (still clean)
                // snapshot — safe, because the writer cannot have landed.
                clean_before
            });

    assert!(
        reclaimed,
        "the frame must be reclaimed here: it was genuinely clean (the \
         pin-coupled writer could not have landed yet), so a correct \
         implementation always removes it at this point"
    );

    // The pin-coupled writer's background thread unblocks now that the
    // claim resolved; wait for it to actually finish faulting the page
    // back in and completing its mutate + mark_dirty on the NEW frame.
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while stack.dpt.snapshot_key(key).is_none() && std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }

    let physical_byte = {
        stack.page_store.fault_in(pid).unwrap();
        let latch = RecordPageBackend::latch(stack.page_store.as_ref(), pid).unwrap();
        let guard = latch.read();
        let page = SlottedPageRef::open(guard.as_ref().as_ref()).unwrap();
        let slot = stack
            .primary
            .lookup(PrimaryKey::new(
                TenantId::DEFAULT,
                RecordKind::Node,
                seed_id.raw(),
            ))
            .unwrap()
            .unwrap();
        page.read_node(slot.slot).unwrap().unwrap().inline_u32a
    };
    assert_eq!(
        physical_byte, 0xAB,
        "the pin-coupled writer must have landed its fresh byte on the \
         (possibly re-faulted) frame after the claim resolved — no \
         write is lost even though the frame was reclaimed mid-race, \
         because the writer's pin excluded it from ever touching a \
         frame the claim was deciding on"
    );
}
