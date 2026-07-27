//! ADR-140-amendment-01 §Decision item 5 — the concurrent-CRUD-vs-flush
//! fault-injection gate that ships WITH the PinningLatch lift.
//!
//! Two legs, one schedule:
//!
//! - **RED sensitivity (the revert lever, always-on):** the legacy
//!   bare-`strong_count` eviction (`evict_page_bare_strongcount_for_gate`
//!   — byte-for-byte the ADR-140 §D-3 shape, with the schedule hook in
//!   the documented race window) MUST lose the concurrent write under
//!   the injected schedule. A gate whose schedule cannot surface the
//!   named regression does not enforce it (doctrine §3 strong-oracles;
//!   `feedback_verify_the_production_regime`).
//! - **GREEN under the pin discipline:** the SAME schedule against the
//!   pin-coupled surface (`latch_pinned` writer vs `copy_page_pinned` +
//!   `write_pages_home` + `try_evict_page_pinned` flusher) preserves
//!   every write: the live pin refuses removal; the caller-side
//!   revalidation (the M3 DPT `dirty_gen` shape) refuses a stale-image
//!   drop after a re-dirty.
//!
//! Plus a RULE-MT unpinned-schedule hammer: N writers × 1 flusher for a
//! fixed op budget; the oracle is exact (every increment survives — a
//! determinism-equality oracle, not `>=`).
//!
//! Every rendezvous wait is BOUNDED (30 s) with a loud panic naming the
//! dead peer (`feedback_unbounded_test_rendezvous_wait_is_suite_wide_hang`).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use arcgraph_core::{PAGE_SIZE, PageId, PageType, TenantId};
use arcgraph_storage::io::{InMemoryPageIo, PageIo};
use arcgraph_storage::page_store::{
    BufferedRecordPageStore, PerTenantBufferPool, PerTenantBufferPoolConfig, RecordPageBackend,
};

const WAIT_BUDGET: Duration = Duration::from_secs(30);

/// Bounded spin-wait on a flag; panics naming the dead peer on timeout.
fn wait_for(flag: &AtomicBool, who: &str) {
    let start = Instant::now();
    while !flag.load(Ordering::Acquire) {
        if start.elapsed() > WAIT_BUDGET {
            panic!("rendezvous timed out waiting for {who} (peer dead or stalled)");
        }
        std::thread::yield_now();
    }
}

fn new_store(cap: usize) -> Arc<BufferedRecordPageStore> {
    let io: Arc<dyn PageIo> = Arc::new(InMemoryPageIo::new());
    let pools = Arc::new(PerTenantBufferPool::with_config(
        io,
        PerTenantBufferPoolConfig {
            frames_per_tenant: 16,
            write_fraction: 0.0,
        },
    ));
    Arc::new(BufferedRecordPageStore::with_cache_cap(pools, cap))
}

/// The marker offset: last byte of the page (payload area for a fresh
/// slotted page — mirrors the page_store.rs unit-test precedent).
const MARKER: usize = PAGE_SIZE - 1;

/// RED-sensitivity leg: the legacy bare-`strong_count` eviction loses a
/// write under the injected schedule. This test PASSING proves the
/// schedule genuinely surfaces the §D-3 race — i.e. the gate's green
/// leg below is not vacuous. (This is the "revert the pin coupling →
/// the schedule must surface the lost write" RED-on-revert, kept as an
/// always-on sensitivity probe instead of a one-off manual revert.)
#[test]
fn bare_strongcount_eviction_loses_concurrent_write_under_schedule() {
    let store = new_store(4);
    let pid = PageId::new(1);
    store
        .install_fresh(pid, PageType::Node, TenantId::DEFAULT)
        .unwrap();

    let writer_go = Arc::new(AtomicBool::new(false));
    let writer_done = Arc::new(AtomicBool::new(false));

    let w_store = Arc::clone(&store);
    let w_go = Arc::clone(&writer_go);
    let w_done = Arc::clone(&writer_done);
    let writer = std::thread::spawn(move || {
        wait_for(&w_go, "evictor to open the race window");
        // Legacy bare-latch write — the v8-era CRUD shape.
        let latch = w_store.latch(pid).expect("page latched");
        latch.write().as_mut()[MARKER] = 0xA1;
        drop(latch);
        w_done.store(true, Ordering::Release);
    });

    // Evict via the LEGACY shape with the schedule hook in the §D-3
    // window: [strong_count snapshot + disk write] → HOOK → [remove].
    let evicted = store.evict_page_bare_strongcount_for_gate(pid, || {
        writer_go.store(true, Ordering::Release);
        wait_for(&writer_done, "writer inside the race window");
    });
    writer.join().expect("writer panicked");
    assert!(evicted, "bare eviction must have proceeded (window shape)");

    // Fault the page back in: the writer's byte is GONE — the §D-3
    // lost write, surfaced deterministically.
    store.fault_in(pid).expect("fault_in");
    let latch = store.latch(pid).unwrap();
    let observed = latch.read().as_ref()[MARKER];
    assert_eq!(
        observed, 0x00,
        "expected the LOST write (stale disk bytes) under the bare \
         strong_count schedule; if this fails the schedule no longer \
         surfaces the §D-3 race and the gate is vacuous"
    );
}

/// GREEN sensitivity partner for the exact §D-3 window above. The
/// evictor takes its advisory bare-strong-count snapshot, then the hook
/// lets a writer acquire the pin-coupled latch. The writer remains live
/// across the removal decision. The authoritative pin claim must refuse
/// removal. Replacing the claim with the legacy snapshot→remove shape
/// makes this schedule remove the writer's frame and the test goes RED.
#[test]
fn pin_claim_closes_the_latch_vs_remove_window() {
    let store = new_store(4);
    let pid = PageId::new(3);
    store
        .install_fresh(pid, PageType::Node, TenantId::DEFAULT)
        .unwrap();

    let image = store.copy_page_pinned(pid).unwrap().unwrap();
    store.write_pages_home(&[(pid, image)]).unwrap();

    let writer_go = Arc::new(AtomicBool::new(false));
    let writer_holding = Arc::new(AtomicBool::new(false));
    let writer_release = Arc::new(AtomicBool::new(false));

    let w_store = Arc::clone(&store);
    let w_go = Arc::clone(&writer_go);
    let w_holding = Arc::clone(&writer_holding);
    let w_release = Arc::clone(&writer_release);
    let writer = std::thread::spawn(move || {
        wait_for(&w_go, "evictor to enter the check-to-remove window");
        let pinned = w_store.latch_pinned(pid).expect("pinned writer latch");
        w_holding.store(true, Ordering::Release);
        wait_for(&w_release, "evictor to finish the removal decision");
        pinned.latch().write().as_mut()[MARKER] = 0xC3;
    });

    let evicted = store.try_evict_page_pinned_with_hook_for_gate(
        pid,
        || true,
        || {
            writer_go.store(true, Ordering::Release);
            wait_for(
                &writer_holding,
                "writer to acquire pin+latch in race window",
            );
        },
    );
    writer_release.store(true, Ordering::Release);
    writer.join().expect("writer panicked");
    assert!(
        !evicted,
        "the authoritative pin claim must refuse a writer that landed \
         after the advisory strong-count snapshot"
    );
    assert!(store.is_cached(pid));

    // Flush the writer's bytes and prove the eventual clean eviction
    // round-trip preserves them.
    let image = store.copy_page_pinned(pid).unwrap().unwrap();
    store.write_pages_home(&[(pid, image)]).unwrap();
    assert!(store.try_evict_page_pinned(pid, || true));
    store.fault_in(pid).unwrap();
    assert_eq!(store.latch(pid).unwrap().read().as_ref()[MARKER], 0xC3);
}

#[test]
fn flush_pages_copies_home_without_evicting() {
    let store = new_store(4);
    let pid = PageId::new(4);
    store
        .install_fresh(pid, PageType::Node, TenantId::DEFAULT)
        .unwrap();
    store.latch_pinned(pid).unwrap().latch().write().as_mut()[MARKER] = 0xD4;

    assert_eq!(store.flush_pages([pid]).unwrap(), 1);
    assert!(store.is_cached(pid), "flush must not evict the frame");

    assert!(store.try_evict_page_pinned(pid, || true));
    store.fault_in(pid).unwrap();
    assert_eq!(store.latch(pid).unwrap().read().as_ref()[MARKER], 0xD4);
}

/// GREEN leg: the identical schedule against the pin-coupled surface
/// preserves the write. Pin refusal + dirty-revalidation refusal both
/// exercised; eviction then succeeds on the clean retry and the write
/// survives the evict → fault_in round-trip.
#[test]
fn pinned_flush_evict_preserves_concurrent_write_under_same_schedule() {
    let store = new_store(4);
    let pid = PageId::new(2);
    store
        .install_fresh(pid, PageType::Node, TenantId::DEFAULT)
        .unwrap();

    // The M3 DPT dirty_gen shape: writers bump; the flusher snapshots
    // before the copy and revalidates at evict time.
    let dirty_gen = Arc::new(AtomicU64::new(0));

    let writer_holding = Arc::new(AtomicBool::new(false));
    let writer_release = Arc::new(AtomicBool::new(false));
    let writer_done = Arc::new(AtomicBool::new(false));

    let w_store = Arc::clone(&store);
    let w_gen = Arc::clone(&dirty_gen);
    let w_holding = Arc::clone(&writer_holding);
    let w_release = Arc::clone(&writer_release);
    let w_done = Arc::clone(&writer_done);
    let writer = std::thread::spawn(move || {
        // Acquire through the PIN-COUPLED surface (the v9-era shape).
        let pinned = w_store.latch_pinned(pid).expect("pinned latch");
        w_holding.store(true, Ordering::Release);
        wait_for(&w_release, "evictor to observe the pin refusal");
        pinned.latch().write().as_mut()[MARKER] = 0xB7;
        w_gen.fetch_add(1, Ordering::AcqRel);
        drop(pinned);
        w_done.store(true, Ordering::Release);
    });

    // Flusher: copy the (pre-write) image while the writer holds the
    // pin, write it home, then attempt the drop — MUST be refused by
    // the live pin.
    wait_for(&writer_holding, "writer to acquire the pinned latch");
    let gen_at_copy = dirty_gen.load(Ordering::Acquire);
    let image = store
        .copy_page_pinned(pid)
        .expect("copy ok")
        .expect("page exists");
    store.write_pages_home(&[(pid, image)]).expect("home write");
    let evicted_while_pinned =
        store.try_evict_page_pinned(pid, || dirty_gen.load(Ordering::Acquire) == gen_at_copy);
    assert!(
        !evicted_while_pinned,
        "eviction must be refused while a pin is live (the amendment's \
         un-removable-while-pinned contract)"
    );

    // Release the writer; it writes the marker (re-dirty) and drops.
    writer_release.store(true, Ordering::Release);
    wait_for(&writer_done, "writer to write + drop the pinned latch");
    writer.join().expect("writer panicked");

    // Retry with the STALE image's revalidation: dirty_gen moved, so
    // the drop MUST be refused (the stale-home-bytes hazard).
    let evicted_stale =
        store.try_evict_page_pinned(pid, || dirty_gen.load(Ordering::Acquire) == gen_at_copy);
    assert!(
        !evicted_stale,
        "eviction with a stale flushed image must be refused by the \
         dirty_gen revalidation (re-dirty keeps the frame)"
    );

    // Clean pass: re-copy (now contains the marker), re-write home,
    // revalidate against the fresh generation → drop succeeds.
    let gen_clean = dirty_gen.load(Ordering::Acquire);
    let image = store
        .copy_page_pinned(pid)
        .expect("copy ok")
        .expect("page exists");
    store.write_pages_home(&[(pid, image)]).expect("home write");
    let evicted_clean =
        store.try_evict_page_pinned(pid, || dirty_gen.load(Ordering::Acquire) == gen_clean);
    assert!(evicted_clean, "clean eviction must proceed once unpinned");
    assert!(store.is_evicted(pid));

    // The round-trip preserves the write: fault back in, marker present.
    store.fault_in(pid).expect("fault_in");
    let latch = store.latch(pid).unwrap();
    assert_eq!(
        latch.read().as_ref()[MARKER],
        0xB7,
        "the pin-coupled flush/evict protocol lost a write"
    );
}

/// Belt: a live pin blocks the LEGACY `evict_lru` walker too (mixed-era
/// coexistence — amendment item 4).
#[test]
fn legacy_evict_lru_skips_pinned_pages() {
    let store = new_store(1);
    for i in 0..3 {
        store
            .install_fresh(PageId::new(i), PageType::Node, TenantId::DEFAULT)
            .unwrap();
    }
    let pinned = store.latch_pinned(PageId::new(0)).expect("pin page 0");
    let evicted = store.evict_lru(0).unwrap();
    assert_eq!(evicted, 2, "pages 1+2 evict; pinned page 0 must not");
    assert!(store.is_cached(PageId::new(0)));
    drop(pinned);
    let evicted = store.evict_lru(0).unwrap();
    assert_eq!(evicted, 1, "page 0 evicts once unpinned");
}

/// RULE-MT hammer: 4 pinned writers × 1 flush/evict loop, exact-count
/// oracle. Each writer performs a fixed number of read-modify-write
/// increments on a distinct byte lane under the pinned write latch;
/// the flusher concurrently copies/home-writes/evicts (dirty_gen-
/// revalidated) as fast as it can. Oracle: every lane's final value ==
/// its writer's increment budget — an exact equality, so ANY lost
/// update (the §D-3 class) fails the gate.
#[test]
fn mt_hammer_no_lost_updates_under_concurrent_flush_evict() {
    const WRITERS: usize = 4;
    const INCREMENTS: u8 = 200;

    let store = new_store(4);
    let pid = PageId::new(9);
    store
        .install_fresh(pid, PageType::Node, TenantId::DEFAULT)
        .unwrap();

    let dirty_gen = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));

    let mut writers = Vec::new();
    for lane in 0..WRITERS {
        let store = Arc::clone(&store);
        let dirty_gen = Arc::clone(&dirty_gen);
        writers.push(std::thread::spawn(move || {
            // Lane byte: distinct offsets at the page tail (clear of
            // the slotted-page header/slot area for a fresh page).
            let off = PAGE_SIZE - 1 - lane;
            for _ in 0..INCREMENTS {
                let pinned = store.latch_pinned(pid).expect("pinned latch");
                {
                    let mut g = pinned.latch().write();
                    g.as_mut()[off] = g.as_ref()[off].wrapping_add(1);
                }
                dirty_gen.fetch_add(1, Ordering::AcqRel);
                drop(pinned);
            }
        }));
    }

    let flusher = {
        let store = Arc::clone(&store);
        let dirty_gen = Arc::clone(&dirty_gen);
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || {
            let mut evictions = 0u64;
            while !stop.load(Ordering::Acquire) {
                let generation = dirty_gen.load(Ordering::Acquire);
                if let Ok(Some(image)) = store.copy_page_pinned(pid) {
                    store.write_pages_home(&[(pid, image)]).expect("home");
                    if store.try_evict_page_pinned(pid, || {
                        dirty_gen.load(Ordering::Acquire) == generation
                    }) {
                        evictions += 1;
                        // Fault straight back in so writers keep going
                        // (they fault-in themselves too; this keeps
                        // eviction pressure high).
                        let _ = store.fault_in(pid);
                    }
                }
                std::thread::yield_now();
            }
            evictions
        })
    };

    let deadline = Instant::now() + WAIT_BUDGET;
    for w in writers {
        assert!(
            Instant::now() < deadline,
            "writer join exceeded the bounded wait budget"
        );
        w.join().expect("writer panicked");
    }
    stop.store(true, Ordering::Release);
    let evictions = flusher.join().expect("flusher panicked");

    store.fault_in(pid).expect("fault_in");
    let latch = store.latch(pid).unwrap();
    let g = latch.read();
    for lane in 0..WRITERS {
        let off = PAGE_SIZE - 1 - lane;
        assert_eq!(
            g.as_ref()[off],
            INCREMENTS,
            "lane {lane} lost updates under concurrent flush/evict \
             (evictions observed: {evictions})"
        );
    }
}
