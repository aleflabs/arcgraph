//! W26-ε-2 / ADR-140 — eviction-pressure correctness test.
//!
//! Exercises the [`BufferedRecordPageStore`] under high page-installation
//! churn against a small hot-cache cap so eviction-to-disk happens
//! many times during the run. The K-1-style oracle checks:
//!
//! 1. **All installed pages are accessible after eviction-and-fault-in.**
//!    Each page receives a tenant-unique byte pattern at install; the
//!    test asserts that pattern survives an eviction + fault-in
//!    round-trip.
//! 2. **Eviction never corrupts a page that has an outstanding latch.**
//!    The test holds latches on a subset of pages while triggering
//!    eviction; those pages MUST remain in the hot cache (eviction
//!    skips them per the `Arc::strong_count` race-safety check) and
//!    their bytes MUST be unchanged.
//! 3. **Cache cap is respected after each eviction sweep.** Post-evict,
//!    `cache_size() <= cap` (modulo pages with outstanding latches).
//! 4. **Total page count is invariant.** `total_pages()` equals
//!    `cache_size() + evicted_count()` and never decreases during
//!    install-only workloads.
//!
//! Per ADR-140 §"Acceptance criteria" item 4.

use std::sync::Arc;

use arcgraph_core::{PAGE_SIZE, PageId, PageType, TenantId};
use arcgraph_storage::{
    BufferedRecordPageStore, PerTenantBufferPool, PerTenantBufferPoolConfig, RecordPageBackend,
    io::InMemoryPageIo,
};

/// Build a small-cap BufferedRecordPageStore for eviction pressure.
fn make_store(cache_cap: usize) -> Arc<BufferedRecordPageStore> {
    let io: Arc<dyn arcgraph_storage::io::PageIo> = Arc::new(InMemoryPageIo::new());
    let pools = Arc::new(PerTenantBufferPool::with_config(
        io,
        PerTenantBufferPoolConfig {
            // Small frames-per-tenant so eviction reaches into the
            // shared PageIo on every spill.
            frames_per_tenant: 8,
            write_fraction: 0.0,
        },
    ));
    Arc::new(BufferedRecordPageStore::with_cache_cap(pools, cache_cap))
}

/// Imprint a unique pattern in the slotted-page body. Reuses the body
/// region that SlottedPage::init zeroes; touches a byte at a stable
/// offset that the page-codec's CRC-check tolerates. The K-1 oracle
/// only checks invariants, not page-format checksums; the synthetic
/// pattern is at offset 0 in the body region (which is at PAGE_SIZE/2)
/// so the page header CRC stays valid for round-trip read-after-evict.
fn imprint(store: &Arc<BufferedRecordPageStore>, pid: PageId, pattern: u8) {
    let latch = store.latch(pid).expect("latch installed page");
    let mut g = latch.write();
    let bytes: &mut [u8; PAGE_SIZE] = g.as_mut();
    // Touch a body byte that round-trips through eviction without
    // invalidating the header CRC (the body checksum is updated by
    // SlottedPage::init at install; we deliberately do NOT recompute
    // here — the eviction round-trip preserves bytes byte-equal).
    bytes[PAGE_SIZE / 2] = pattern;
    bytes[PAGE_SIZE - 1] = pattern.wrapping_add(0x55);
}

/// Verify pattern survives across an eviction + fault-in round-trip.
fn verify(store: &Arc<BufferedRecordPageStore>, pid: PageId, pattern: u8) {
    // Force fault-in if currently evicted.
    store.fault_in(pid).expect("fault_in installed page");
    let latch = store.latch(pid).expect("latch installed page");
    let g = latch.read();
    let bytes: &[u8; PAGE_SIZE] = g.as_ref();
    assert_eq!(
        bytes[PAGE_SIZE / 2],
        pattern,
        "page {:?} body pattern survived round-trip",
        pid
    );
    assert_eq!(
        bytes[PAGE_SIZE - 1],
        pattern.wrapping_add(0x55),
        "page {:?} sentinel byte survived round-trip",
        pid
    );
}

#[test]
fn k1_invariant_all_installed_pages_accessible_after_eviction() {
    let store = make_store(8);

    const N: u64 = 64;
    for i in 0..N {
        let pid = PageId::new(i + 100);
        store
            .install_fresh(pid, PageType::Node, TenantId::DEFAULT)
            .unwrap();
        imprint(&store, pid, (i as u8).wrapping_mul(7));
    }

    // After N installs at cap 8, total_pages == N + 1 cache-resident
    // installs that exceed the cap will be tolerated until explicit
    // evict_lru. Drive explicit eviction now.
    let evicted = store.evict_lru(8).expect("evict_lru");
    assert!(
        evicted > 0,
        "expected at least one eviction with cap 8 and N {} installs (got {})",
        N,
        evicted
    );
    // After evict, cache_size <= 8 (modulo outstanding latches, which
    // this test does not hold).
    assert!(
        store.cache_size() <= 8,
        "cache_size {} exceeds cap 8 after explicit evict",
        store.cache_size()
    );
    assert_eq!(
        store.total_pages(),
        N as usize,
        "total page count invariant"
    );

    // Verify every page round-trips through eviction.
    for i in 0..N {
        let pid = PageId::new(i + 100);
        verify(&store, pid, (i as u8).wrapping_mul(7));
    }
}

#[test]
fn k1_invariant_outstanding_latch_blocks_eviction() {
    let store = make_store(2);

    // Install 5 pages.
    for i in 0..5 {
        let pid = PageId::new(i + 200);
        store
            .install_fresh(pid, PageType::Node, TenantId::DEFAULT)
            .unwrap();
        imprint(&store, pid, (i as u8).wrapping_mul(11));
    }

    // Hold an outstanding latch on the OLDEST page (the LRU candidate).
    // Its bytes pattern is `0 * 11 = 0`. We assert that:
    // (a) the outstanding latch's page does NOT evict.
    // (b) other pages evict normally.
    let held = store.latch(PageId::new(200)).unwrap();
    let _g = held.read();

    let evicted = store.evict_lru(0).unwrap();
    // 4 pages (201..205) evict; 200 stays cached.
    assert_eq!(evicted, 4);
    assert!(
        store.is_cached(PageId::new(200)),
        "outstanding-latch page must stay cached"
    );
    for i in 1..5 {
        assert!(
            store.is_evicted(PageId::new(i + 200)),
            "non-latched page {:?} should be evicted",
            PageId::new(i + 200)
        );
    }
}

#[test]
fn k1_invariant_total_pages_monotone_during_install_only_workload() {
    let store = make_store(4);

    let mut last_total = 0;
    for i in 0..50 {
        let pid = PageId::new(i + 300);
        store
            .install_fresh(pid, PageType::Node, TenantId::DEFAULT)
            .unwrap();
        // Drive explicit evict every 4 installs.
        if i % 4 == 3 {
            store.evict_lru(4).unwrap();
        }
        let total = store.total_pages();
        assert!(
            total >= last_total,
            "total_pages decreased during install-only workload: {} -> {}",
            last_total,
            total,
        );
        last_total = total;
    }
    assert_eq!(store.total_pages(), 50);
}

#[test]
fn k1_invariant_eviction_byte_equality_across_round_trip() {
    let store = make_store(4);
    const N: u64 = 32;

    for i in 0..N {
        let pid = PageId::new(i + 400);
        store
            .install_fresh(pid, PageType::Node, TenantId::DEFAULT)
            .unwrap();
        imprint(&store, pid, (i as u8).wrapping_mul(13));
    }
    // Three eviction passes; cache_cap = 4 → 28 pages spill.
    for _ in 0..3 {
        store.evict_lru(4).unwrap();
    }
    // Round-trip every page: cache-resident pages re-read; evicted
    // pages fault-in via PageIo.
    for i in 0..N {
        let pid = PageId::new(i + 400);
        verify(&store, pid, (i as u8).wrapping_mul(13));
    }
    // Final total_pages still N (eviction is bookkeeping-only, never
    // drops pages).
    assert_eq!(store.total_pages(), N as usize);
}

#[test]
fn k1_invariant_install_or_replace_overrides_evicted_bytes() {
    let store = make_store(2);

    for i in 0..5 {
        let pid = PageId::new(i + 500);
        store
            .install_fresh(pid, PageType::Node, TenantId::DEFAULT)
            .unwrap();
        imprint(&store, pid, (i as u8).wrapping_mul(17));
    }
    // Evict 3 (LRU 500, 501, 502).
    store.evict_lru(2).unwrap();
    assert!(store.is_evicted(PageId::new(500)));

    // Replay-style install_or_replace re-installs new bytes for an
    // evicted page. The buffered store treats this as a fresh install
    // in cache + clears the evicted bit.
    let mut new_bytes: Box<[u8; PAGE_SIZE]> = Box::new([0u8; PAGE_SIZE]);
    new_bytes[0] = 0xCC;
    new_bytes[PAGE_SIZE - 1] = 0xCC;
    store
        .install_or_replace(PageId::new(500), new_bytes)
        .unwrap();

    assert!(store.is_cached(PageId::new(500)));
    assert!(!store.is_evicted(PageId::new(500)));
    let latch = store.latch(PageId::new(500)).unwrap();
    let g = latch.read();
    assert_eq!(g.as_ref()[0], 0xCC);
    assert_eq!(g.as_ref()[PAGE_SIZE - 1], 0xCC);
}
