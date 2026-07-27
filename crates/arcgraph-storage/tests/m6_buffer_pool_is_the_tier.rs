//! M6.1 — `m6_buffer_pool_is_the_tier` (ADR-232-amendment-01 §2.3 epic
//! gate; M6.1 slice table).
//!
//! Proves the buffer pool genuinely serves OUT-OF-CORE: a working set
//! whose total size exceeds the configured cache cap by a wide margin
//! (10×) is served CORRECTLY by faulting pages from disk through
//! `evict_for_capacity`'s MECH-E1..E8 mechanism, with the resident
//! cache actually staying bounded near the cap throughout (not an
//! in-RAM cheat where the "cap" is nominal but nothing is ever really
//! evicted).
//!
//! RED (per the charter): make the pool hold EVERYTHING (i.e., never
//! call `evict_for_capacity` / configure a cap larger than the working
//! set) → `cache_size` grows unbounded with the working set and the
//! "is THE tier" property goes untested (the negative control
//! `cache_grows_unbounded_without_eviction_pressure` demonstrates
//! exactly this — proving the harness can tell the difference between
//! "bounded" and "not bounded").

use std::sync::Arc;

use arcgraph_core::{Lsn, PAGE_SIZE, PageId, PageType, TenantId};
use arcgraph_storage::checkpoint::{PageFlushTarget, WriteBehindCheckpointer};
use arcgraph_storage::io::{PageIo, PosixPageIo};
use arcgraph_storage::page_store::{
    BufferedRecordPageStore, PerTenantBufferPool, PerTenantBufferPoolConfig, RecordPageBackend,
};
use arcgraph_storage::redo::{DirtyPageKey, DirtyPageTable};
use arcgraph_storage::wal::STORE_RECORD;
use tempfile::tempdir;

const CACHE_CAP: usize = 32;
const WORKING_SET: u64 = 320; // 10× the cache cap.

fn new_disk_store(dir: &std::path::Path, cap: usize) -> Arc<BufferedRecordPageStore> {
    let io: Arc<dyn PageIo> =
        Arc::new(PosixPageIo::open_or_create(dir.join("record.store")).expect("open page io"));
    let pools = Arc::new(PerTenantBufferPool::with_config(
        io,
        PerTenantBufferPoolConfig {
            frames_per_tenant: 64,
            write_fraction: 0.0,
        },
    ));
    Arc::new(BufferedRecordPageStore::with_cache_cap(pools, cap))
}

/// THE decisive leg: a 10×-larger-than-cap working set is served
/// correctly (every page's byte round-trips through eviction + fault-in)
/// while the resident cache genuinely stays bounded near the cap
/// throughout the run — proving out-of-core, not an in-RAM cheat.
#[test]
fn working_set_exceeds_cap_and_is_served_correctly_via_faulting() {
    let dir = tempdir().unwrap();
    let store = new_disk_store(dir.path(), CACHE_CAP);
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

    let mut lsn = 1u64;
    let mut max_observed_cache_size = 0usize;

    // Install + immediately mutate + mark-dirty every page in the
    // working set, driving eviction after each — the continuous
    // pressure regime.
    for i in 0..WORKING_SET {
        let pid = PageId::new(i);
        store
            .install_fresh(pid, PageType::Node, TenantId::DEFAULT)
            .unwrap();
        let byte = (i as u8).wrapping_mul(97).wrapping_add(3);
        {
            let latch = RecordPageBackend::latch_for_tenant(store.as_ref(), TenantId::DEFAULT, pid)
                .unwrap();
            latch.write().as_mut()[PAGE_SIZE - 1] = byte;
        }
        dpt.mark_dirty(
            DirtyPageKey {
                tenant_id: TenantId::DEFAULT,
                store_id: STORE_RECORD,
                page_no: pid.raw(),
            },
            Lsn::new(lsn),
        );
        lsn += 1;
        let _ = store.evict_for_capacity(CACHE_CAP);
        max_observed_cache_size = max_observed_cache_size.max(store.cache_size());
    }
    // Final drain settles any stragglers.
    let _ = checkpointer.flush_pass(Lsn::new(lsn));

    // THE out-of-core assertion: the resident cache never ballooned to
    // anywhere near the working-set size — it stayed bounded near the
    // configured cap throughout the ENTIRE run (a generous slack factor
    // absorbs transient install-before-evict overshoot, but nowhere
    // close to the 10x working set).
    assert!(
        max_observed_cache_size <= CACHE_CAP * 4,
        "cache_size peaked at {max_observed_cache_size}, expected bounded \
         near cap {CACHE_CAP} (working set was {WORKING_SET}) — the pool is \
         NOT genuinely serving out-of-core if resident size tracks the \
         working set instead of the cap"
    );

    // Correctness: every page in the 10x-larger-than-cap working set is
    // still readable with its correct byte, via fault-in from disk.
    for i in 0..WORKING_SET {
        let pid = PageId::new(i);
        let expected = (i as u8).wrapping_mul(97).wrapping_add(3);
        store.fault_in(pid).unwrap();
        let latch = store.latch(pid).unwrap();
        assert_eq!(
            latch.read().as_ref()[PAGE_SIZE - 1],
            expected,
            "page {i} lost its byte — out-of-core faulting is not correct"
        );
    }
    assert_eq!(
        store.total_pages(),
        WORKING_SET as usize,
        "total tracked page count must equal the full working set"
    );
}

/// Negative control (per the charter's RED prescription): if eviction
/// pressure is NEVER applied (the pool is configured to just "hold
/// everything"), resident cache size grows unboundedly with the working
/// set — demonstrating the harness genuinely distinguishes "bounded,
/// out-of-core" from "in-RAM cheat" rather than passing regardless.
#[test]
fn cache_grows_unbounded_without_eviction_pressure() {
    let dir = tempdir().unwrap();
    let store = new_disk_store(dir.path(), CACHE_CAP);
    // No DPT/checkpointer wired, and eviction is never driven — this is
    // the "pool holds everything" configuration the charter's RED asks
    // for as the contrast case.
    for i in 0..WORKING_SET {
        store
            .install_fresh(PageId::new(i), PageType::Node, TenantId::DEFAULT)
            .unwrap();
    }
    assert_eq!(
        store.cache_size(),
        WORKING_SET as usize,
        "with no eviction pressure applied at all, resident cache size \
         tracks the FULL working set — this is the in-RAM-cheat shape the \
         positive test above must NOT exhibit"
    );
    assert!(
        store.cache_size() > CACHE_CAP * 4,
        "sanity: the unbounded-cheat shape must clearly exceed the \
         bounded threshold the positive test enforces"
    );
}

/// Sanity: an empty/lightly-loaded pool (working set <= cap) never needs
/// eviction to be correct — a positive control confirming the mechanism
/// is not itself introducing spurious evictions when there is no pressure.
#[test]
fn working_set_within_cap_never_evicts() {
    let dir = tempdir().unwrap();
    let store = new_disk_store(dir.path(), CACHE_CAP);
    let dpt = Arc::new(DirtyPageTable::new());
    let props_target: Arc<dyn PageFlushTarget> = store.clone();
    let records_target: Arc<dyn PageFlushTarget> = store.clone();
    let checkpointer = Arc::new(WriteBehindCheckpointer::new(
        dpt.clone(),
        props_target,
        records_target,
    ));
    store.attach_m6_dirty_page_table(dpt);
    store.attach_m6_checkpointer(checkpointer);

    for i in 0..(CACHE_CAP as u64 / 2) {
        store
            .install_fresh(PageId::new(i), PageType::Node, TenantId::DEFAULT)
            .unwrap();
        let _ = store.evict_for_capacity(CACHE_CAP);
    }
    assert_eq!(
        store.evicted_count(),
        0,
        "a working set within the cap must never trigger a real eviction"
    );
}
