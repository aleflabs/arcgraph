//! M6.1 (#1521 P0-2) — `clean_arm_unconditional_revalidate_drops_redirtied_frame`:
//! the deterministic rendezvous gate for the sibling MECH-E3 hazard at
//! `evict_for_capacity`'s CLEAN arm (`page_store.rs`, the `Some(true) =>`
//! branch of the `is_clean(key)` match).
//!
//! `evict_dirty_via_checkpointer`'s dirty-arm revalidate closure (the
//! ORIGINAL MECH-E3 fix) re-checks `is_clean` INSIDE the same pin-coupled
//! removal claim `try_evict_page_pinned_for_tenant` takes — closing the
//! window between "checkpointer confirmed durable" and "evictor claims
//! removal" where a concurrent re-dirty could otherwise be silently
//! dropped. The CLEAN arm had the IDENTICAL hazard shape at a sibling
//! seam (the window between "classified clean" and "removal claim") but,
//! before #1521 P0-2, passed an UNCONDITIONAL `|| true` revalidate
//! closure instead of re-checking `is_clean` inside the claim — so a
//! writer's re-dirty landing in that window was invisible to the
//! removal claim exactly like the P0-1 dirty-arm/bare-latch hazard,
//! just at the clean-arm's own seam instead.
//!
//! Both legs use the skeptic gate's proven technique
//! (`skeptic_mech_e3_revalidate_seam.rs`): the revalidate closure passed
//! to `try_evict_page_pinned_for_tenant` (THE real pin-coupled removal
//! claim `evict_for_capacity`'s clean arm calls) synchronously drives a
//! writer's full re-dirty cycle to completion INSIDE the closure, then
//! returns the PRE-writer "clean" snapshot — deterministically modeling
//! "classified clean; a re-dirty landed immediately after, before
//! removal executed". This is deterministic because the interleaving IS
//! the closure's own control flow, not a race against it.
//!
//! - `unconditional_revalidate_drops_redirtied_frame` (RED-on-revert
//!   SENSITIVITY leg — demonstrates the defect class the fix closes):
//!   passes a literal `|| true` revalidate closure — the EXACT pre-#1521-
//!   P0-2 clean-arm shape — so the removal claim proceeds regardless of
//!   the in-gap re-dirty. The frame is reclaimed and the writer's fresh
//!   byte is lost.
//! - `is_clean_recheck_inside_claim_excludes_removal` (THE decisive leg,
//!   mirrors the FIXED production clean-arm shape exactly): passes the
//!   SAME `is_clean`-shaped re-check the fixed clean arm now uses
//!   (`self.is_clean(key) == Some(true)`, modeled here via
//!   `dpt.snapshot_key(key).is_none()` — the same test-visible
//!   equivalent the skeptic gate uses for the dirty arm). The re-dirty
//!   inside the closure makes THIS re-check observe live dirty state,
//!   so removal is refused and the frame is retained.

use std::sync::Arc;

use arcgraph_core::{Lsn, PAGE_SIZE, PageId, PageType, TenantId};
use arcgraph_storage::io::InMemoryPageIo;
use arcgraph_storage::page_store::{
    BufferedRecordPageStore, PerTenantBufferPool, PerTenantBufferPoolConfig, RecordPageBackend,
};
use arcgraph_storage::redo::{DirtyPageKey, DirtyPageTable};
use arcgraph_storage::wal::STORE_RECORD;

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

/// RED-on-revert sensitivity leg: the EXACT pre-#1521-P0-2 clean-arm
/// shape (`|| true`, unconditional). A re-dirty landing inside the
/// revalidate closure's execution window is invisible to it, so the
/// frame is reclaimed with the writer's fresh byte lost.
#[test]
fn unconditional_revalidate_drops_redirtied_frame() {
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

    // The classic (BROKEN, pre-P0-2) clean-arm revalidate: unconditional
    // `|| true`, ignoring whatever landed in the gap.
    let reclaimed = store.try_evict_page_pinned_for_tenant(TenantId::DEFAULT, pid, move || {
        // Drive the FULL writer cycle to completion inside the closure,
        // landing exactly "after the clean classification, before
        // removal executes" — then return `true` unconditionally,
        // exactly the pre-#1521-P0-2 clean arm's own revalidate closure.
        //
        // The in-window writer is a BARE-latch writer, deliberately and
        // necessarily: (a) a pin-coupled writer structurally CANNOT land
        // in this window at all — its `PinRegistry::pin` blocks on the
        // very shard lock this claim holds for `key` until the claim
        // resolves — so the only writer class that can re-dirty inside
        // the revalidate-to-removal gap is one invisible to the pin
        // registry; and (b) calling `latch_pinned_for_tenant` from
        // inside this closure would self-deadlock on that same shard
        // lock (the structural proof of (a)).
        {
            let latch =
                RecordPageBackend::latch_for_tenant(writer_store.as_ref(), TenantId::DEFAULT, pid)
                    .expect("bare latch");
            latch.write().as_mut()[MARKER] = 0xAB;
        }
        writer_dpt.mark_dirty(key, Lsn::new(1));
        true
    });

    assert!(
        reclaimed,
        "sensitivity leg: expected the frame to be reclaimed (the \
         unconditional `|| true` revalidate ignores the in-gap re-dirty) \
         — if this is false the harness itself is not exercising the \
         hazard this gate exists to prove"
    );
    store.fault_in(pid).unwrap();
    let byte = store.latch(pid).unwrap().read().as_ref()[MARKER];
    assert_eq!(
        byte, 0x00,
        "clean-arm MECH-E3 defect class reproduced: the writer's fresh \
         mutation (0xAB) is GONE after reclaim — only the stale durably- \
         homed byte (0x00) survives, the sibling hazard to #1521's P0-1 \
         at the clean-arm's own revalidate-to-removal seam"
    );
}

/// THE decisive leg (mirrors the FIXED production clean-arm shape
/// exactly): the revalidate closure re-checks the DPT (equivalent to
/// `is_clean(key) == Some(true)`) INSIDE the same pin-coupled claim.
/// Because the (bare-latch) re-dirty happens BEFORE this re-check runs,
/// the re-check observes live dirty state and refuses removal — the
/// frame is retained and the writer's mutation survives.
#[test]
fn is_clean_recheck_inside_claim_excludes_removal() {
    let store = new_store(8);
    let pid = PageId::new(1);
    store
        .install_fresh(pid, PageType::Node, TenantId::DEFAULT)
        .unwrap();
    {
        let latch = store.latch(pid).unwrap();
        latch.write().as_mut()[MARKER] = 0x00;
    }
    store.flush_pages([pid]).unwrap();

    let dpt = Arc::new(DirtyPageTable::new());
    let key = dirty_key(pid);
    let writer_store = store.clone();
    let writer_dpt = dpt.clone();
    let revalidate_dpt = dpt.clone();

    // The FIXED (post-#1521-P0-2) clean-arm shape: re-check `is_clean`
    // (modeled here as `dpt.snapshot_key(key).is_none()`, the same
    // test-visible equivalent used by the P0-1 skeptic gate) INSIDE the
    // same pin-coupled claim.
    let reclaimed = store.try_evict_page_pinned_for_tenant(TenantId::DEFAULT, pid, move || {
        // Same BARE-latch in-window writer as the sensitivity leg (the
        // only writer class that can land inside the claim — a
        // pin-coupled writer blocks on this claim's own shard lock).
        {
            let latch =
                RecordPageBackend::latch_for_tenant(writer_store.as_ref(), TenantId::DEFAULT, pid)
                    .expect("bare latch");
            latch.write().as_mut()[MARKER] = 0xAB;
        }
        writer_dpt.mark_dirty(key, Lsn::new(1));
        // Re-check performed AFTER the in-gap re-dirty landed — this is
        // the FIX: the claim's revalidate must observe the CURRENT DPT
        // state, not a stale pre-writer snapshot.
        revalidate_dpt.snapshot_key(key).is_none()
    });

    assert!(
        !reclaimed,
        "MECH-E3 P0-2 violation: the clean-arm revalidate must see the \
         in-gap writer's dirty mark and refuse removal — if the frame \
         was reclaimed anyway, the re-check is not actually observing \
         live writer state at the seam this gate targets"
    );
    let byte = store.latch(pid).unwrap().read().as_ref()[MARKER];
    assert_eq!(
        byte, 0xAB,
        "the frame was retained, so the live latch must read back the \
         writer's fresh byte, not a stale image"
    );
}
