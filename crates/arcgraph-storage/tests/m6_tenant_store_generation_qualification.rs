//! M6.1 — `m6_tenant_store_generation_qualification` (MECH-E5; §M6.1
//! named gates, ADR-232-amendment-01 §2.3).
//!
//! Two-tenant COLLIDING-PAGE-NUMBER gate. Two tenants each own a page
//! with the SAME `PageId` (a colliding page number is routine — page
//! numbers are per-tenant-store-relative, not global); MECH-E5 requires
//! every flush AND every refault to carry the FULL
//! `(tenant_id, store_id, page_no)` identity end-to-end so eviction
//! never cross-writes or cross-reads between tenants. This is exactly
//! the "stub-double-keyed-richer-than-production" corruption class named
//! in ADR-232-amendment-01 §2.2 MECH-E5's own text (a bare-`PageId` write
//! to a colliding home).
//!
//! Both tenants' colliding-page-number pages are driven through
//! `evict_for_capacity` under real pressure (tiny cap, RULE-MT ≥8
//! concurrent writers split across the two tenants) and each tenant's
//! byte must survive independently — a cross-tenant bleed would show up
//! as tenant A reading tenant B's committed byte (or vice versa).
//!
//! RED-on-revert: routing the DPT key (or the checkpointer's flush
//! target) through a BARE `PageId` instead of the qualified
//! `(tenant_id, store_id, page_no)` triple collapses both tenants'
//! colliding pages onto one DPT/flush identity, so one tenant's
//! eviction durably homes (and the other tenant subsequently reads)
//! the WRONG tenant's bytes.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use arcgraph_core::{Lsn, PAGE_SIZE, PageId, PageType, TenantId};
use arcgraph_storage::checkpoint::{PageFlushTarget, WriteBehindCheckpointer};
use arcgraph_storage::page_store::{
    BufferedRecordPageStore, PerTenantBufferPool, PerTenantBufferPoolConfig, RecordPageBackend,
};
use arcgraph_storage::redo::{DirtyPageKey, DirtyPageTable};
use arcgraph_storage::wal::STORE_RECORD;
use tempfile::tempdir;

const WAIT: Duration = Duration::from_secs(60);
const WRITERS_PER_TENANT: u64 = 4; // 2 tenants x 4 = 8 total, RULE-MT floor.
// NOTE (M6.1 hardening background — see `page_store.rs`'s
// `remove_cached_page_if_unpinned` and `latch_for_tenant`): this gate's
// first drafts, at larger scale, surfaced FOUR pre-existing races in
// `BufferedRecordPageStore` that predate M6.1 and were previously
// unreachable at serial/legacy eviction cadence:
// (1) `remove_cached_page_if_unpinned` removed from `cache` before
//     inserting into `evicted`, leaving a real window where a key was in
//     NEITHER map (`fault_in_for_tenant`'s two-step check could then
//     spuriously report `MissingPage`) — fixed: insert-before-remove.
// (2) `untrack()` (LRU-deque bookkeeping) ran WHILE the pin registry's
//     per-key shard lock was held, serializing on the SAME global `lru`
//     mutex every `touch()` call contends for — a priority-inversion-
//     shaped bottleneck under RULE-MT pressure (observed as multi-minute
//     stalls, not a true deadlock) — fixed: deferred outside the claim.
// (3) `fault_in_for_tenant`'s double-fault-in-same-key race let a
//     losing racer's stale disk read clobber a winner's (possibly
//     already re-mutated) frame — fixed: insert-if-vacant semantics.
// (4) `latch_for_tenant`'s "ensure resident" and "get the handle" were
//     two separate, non-atomic steps, letting a concurrent eviction
//     slip in between and surface a spurious `MissingPage` — fixed: a
//     bounded retry loop.
// (5) THE decisive MECH-E3 finding: `evict_dirty_via_checkpointer`'s
//     final reclaim used an UNCONDITIONAL `|| true` revalidate closure
//     instead of re-checking the DPT — a re-dirty landing between the
//     checkpointer's flush completing and this function's removal claim
//     could have its fresh bytes silently dropped. Fixed: revalidate via
//     `is_clean(key) == Some(true)`, re-checked under the SAME
//     pin-registry claim as the removal, mirroring the write-behind
//     checkpointer's own generation-compare-and-remove discipline.
// All five are fixed in `page_store.rs`; this gate is what caught them.
// The workload size below is chosen to stay comfortably inside the
// (much reduced, post-fix) contention regime while still genuinely
// exercising RULE-MT concurrent cross-tenant collision pressure.
const PAGES_PER_WRITER: u64 = 20;

fn new_store(dir: &std::path::Path, cap: usize) -> Arc<BufferedRecordPageStore> {
    // PRODUCTION per-tenant home-file resolution (`TenantFilePageIo`):
    // each tenant gets its OWN physical file under `<dir>/tenants/<id>/`.
    // A shared single-file `PageIo` (the `PerTenantBufferPool::with_config`
    // constructor) does NOT partition storage by tenant at the physical
    // layer at all — using it here would make a colliding page number
    // collide at the byte-offset level regardless of any RAM-tier
    // identity discipline, which is a different (and already out-of-scope
    // for M6.1) concern than MECH-E5's RAM-tier tenant qualification.
    let tenant_io: Arc<dyn arcgraph_storage::page_store::TenantPageIo> = Arc::new(
        arcgraph_storage::page_store::TenantFilePageIo::new(dir, "record.store"),
    );
    let pools = Arc::new(PerTenantBufferPool::with_tenant_io(
        tenant_io,
        PerTenantBufferPoolConfig {
            frames_per_tenant: 64,
            write_fraction: 0.0,
        },
    ));
    // A tight MECH-E8 retry budget: this workload calls `evict_for_capacity`
    // on a hot loop under real contention (8 writers over a small cap),
    // so the default 250ms-per-retry budget would compound to minutes of
    // wall-clock across thousands of calls. Gates exercising MECH-E8's
    // bound directly use `m6_eviction_drain_durable_sigkill`'s wiring;
    // this gate targets MECH-E5 and wants the bound out of its way.
    Arc::new(
        BufferedRecordPageStore::with_cache_cap(pools, cap)
            .with_m6_evict_wait_budget(Duration::from_millis(2)),
    )
}

const TENANT_A: TenantId = TenantId::new(101);
const TENANT_B: TenantId = TenantId::new(102);

/// THE decisive leg: RULE-MT (8 writers, 4 per tenant) driving
/// colliding-page-number mutations for two tenants through continuous
/// eviction pressure; each tenant's final byte must be its own, never
/// the other tenant's.
#[test]
fn colliding_page_numbers_across_tenants_never_cross_write_under_eviction_pressure() {
    let dir = tempdir().unwrap();
    let cap = 32usize; // small relative to the 320-key working set: forces continuous eviction.
    let store = new_store(dir.path(), cap);

    let dpt = Arc::new(DirtyPageTable::new());
    let props_target: Arc<dyn PageFlushTarget> = store.clone();
    let records_target: Arc<dyn PageFlushTarget> = store.clone();
    let checkpointer = Arc::new(WriteBehindCheckpointer::new(
        dpt.clone(),
        props_target,
        records_target,
    ));
    store.attach_m6_dirty_page_table(dpt.clone());
    store.attach_m6_checkpointer(checkpointer.clone());

    // Pre-allocate every colliding PageId for BOTH tenants up front
    // (install_fresh is not itself concurrency-safe to race against
    // itself for the SAME key — pre-allocation keeps the RULE-MT phase
    // focused on the mutate/evict race, matching the other M6.1 gates'
    // shape). Matches the production `PageAlloc` contract (see
    // `redo::apply_recovery_delta`'s `PageAlloc` arm): every allocation
    // gets an IMMEDIATE, tenant-qualified DPT entry — a brand-new page
    // has no durable home yet, so it is never "clean" (MECH-E1) purely
    // by DPT absence until it has been through at least one flush.
    // Shared monotone LSN source: `DirtyPageTable::mark_dirty` debug-
    // asserts non-decreasing op_lsn PER KEY, and since every `(tenant,
    // page_no)` key here is touched exactly twice (pre-alloc, then one
    // mutation), a single global counter trivially satisfies that for
    // every key without per-key bookkeeping.
    let lsn_source = Arc::new(AtomicU64::new(1));
    let next_lsn = |src: &AtomicU64| src.fetch_add(1, Ordering::AcqRel);

    for i in 0..(WRITERS_PER_TENANT * PAGES_PER_WRITER) {
        let pid = PageId::new(i);
        for tenant in [TENANT_A, TENANT_B] {
            store.install_fresh(pid, PageType::Node, tenant).unwrap();
            dpt.mark_dirty(
                DirtyPageKey {
                    tenant_id: tenant,
                    store_id: STORE_RECORD,
                    page_no: pid.raw(),
                },
                Lsn::new(next_lsn(&lsn_source)),
            );
        }
    }

    let barrier = Arc::new(std::sync::Barrier::new(
        (WRITERS_PER_TENANT * 2) as usize + 1,
    ));
    let mut writers = Vec::new();
    for (tenant, byte_seed) in [(TENANT_A, 0x10u8), (TENANT_B, 0x80u8)] {
        for w in 0..WRITERS_PER_TENANT {
            let store = Arc::clone(&store);
            let dpt = Arc::clone(&dpt);
            let barrier = Arc::clone(&barrier);
            let lsn_source = Arc::clone(&lsn_source);
            writers.push(thread::spawn(move || {
                barrier.wait();
                for i in 0..PAGES_PER_WRITER {
                    let page_no = w * PAGES_PER_WRITER + i;
                    let pid = PageId::new(page_no);
                    let byte = byte_seed.wrapping_add((i % 64) as u8);
                    {
                        let latch =
                            RecordPageBackend::latch_for_tenant(store.as_ref(), tenant, pid)
                                .unwrap();
                        latch.write().as_mut()[PAGE_SIZE - 1] = byte;
                    }
                    dpt.mark_dirty(
                        DirtyPageKey {
                            tenant_id: tenant,
                            store_id: STORE_RECORD,
                            page_no,
                        },
                        Lsn::new(lsn_source.fetch_add(1, Ordering::AcqRel)),
                    );
                    let _ = store.evict_for_capacity(cap);
                }
            }));
        }
    }
    barrier.wait();
    let deadline = Instant::now() + WAIT;
    for w in writers {
        assert!(
            Instant::now() < deadline,
            "writer join exceeded the bounded wait budget"
        );
        w.join().unwrap();
    }

    // Final drain.
    let _ = checkpointer.flush_pass(Lsn::new(lsn_source.load(Ordering::Acquire)));

    // Decisive oracle: every colliding page number's TWO tenant copies
    // (fault back in independently) carry their OWN tenant's final byte,
    // never the other tenant's. `local_i` reconstructs the PER-WRITER
    // loop index from the global page_no (matching the writer's own
    // `let page_no = w * PAGES_PER_WRITER + i;` — each writer's `i`
    // ranges over `0..PAGES_PER_WRITER`, so `page_no % PAGES_PER_WRITER`
    // recovers it).
    for page_no in 0..(WRITERS_PER_TENANT * PAGES_PER_WRITER) {
        let pid = PageId::new(page_no);
        let local_i = page_no % PAGES_PER_WRITER;
        let expected_a = 0x10u8.wrapping_add((local_i % 64) as u8);
        let expected_b = 0x80u8.wrapping_add((local_i % 64) as u8);

        store.fault_in_for_tenant(TENANT_A, pid).unwrap();
        let latch_a = store.latch_for_tenant(TENANT_A, pid).unwrap();
        let byte_a = latch_a.read().as_ref()[PAGE_SIZE - 1];
        drop(latch_a);

        store.fault_in_for_tenant(TENANT_B, pid).unwrap();
        let latch_b = store.latch_for_tenant(TENANT_B, pid).unwrap();
        let byte_b = latch_b.read().as_ref()[PAGE_SIZE - 1];
        drop(latch_b);

        assert_eq!(
            byte_a, expected_a,
            "page_no {page_no}: TENANT_A read byte {byte_a:#x}, expected {expected_a:#x} \
             (expected_b was {expected_b:#x}) — a match against expected_b here \
             would mean tenant A read tenant B's home image: MECH-E5 cross-tenant \
             bleed via a colliding page number"
        );
        assert_eq!(
            byte_b, expected_b,
            "page_no {page_no}: TENANT_B read byte {byte_b:#x}, expected {expected_b:#x} \
             (expected_a was {expected_a:#x}) — MECH-E5 cross-tenant bleed"
        );
    }
}

// RED-on-revert for this gate is a code mutation (matching the
// established M6.1 discipline): replacing `dirty_page_key`'s
// `DirtyPageKey { tenant_id: key.tenant_id, store_id: key.store_id,
// page_no: key.page_id.raw() }` with a construction that drops
// `tenant_id` (folds both tenants' DPT entries for a colliding page_no
// onto one key) collapses the two tenants' eviction handshakes onto a
// single DPT entry, so one tenant's flush_priority_keys call durably
// homes (and clears the DPT for) BOTH tenants' colliding pages using
// only ONE tenant's copy — the other tenant's page then reclaims
// against a "completed" flush that never touched its bytes,
// deterministically producing a wrong byte on refault. This is the same
// mutation class MECH-E5's doc comment names (`page_store.rs`'s
// `dirty_page_key` — "always fully qualified, never a bare PageId").
