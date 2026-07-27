//! W26-ε-2 / ADR-140 — page-store recovery correctness test.
//!
//! Exercises crash-equivalent recovery of [`BufferedRecordPageStore`]:
//!
//! 1. Build a `BufferedRecordPageStore` over a [`PosixPageIo`] file in
//!    a tempdir.
//! 2. Install N pages with known per-page byte patterns.
//! 3. Evict every page to disk (`evict_lru(0)` drains the cache).
//! 4. `flush_all()` the underlying file (POSIX `fdatasync`).
//! 5. Drop the store — equivalent to a process crash AFTER fsync (the
//!    crash-survives-eviction case; ADR-140 §D-4 "Recovery semantics").
//! 6. Re-open the file via a fresh `PosixPageIo` + a fresh
//!    `BufferedRecordPageStore` over the same file.
//! 7. Simulate WAL replay by invoking
//!    [`RecordPageStoreHandle::install_or_replace`] with each page id
//!    plus its known bytes (this is how the real WAL replay path
//!    populates the post-restart store; ADR-140 §D-4 + the existing
//!    `recover_from_wal` dispatch).
//! 8. Verify the post-replay store reads bytes byte-equal to the
//!    pre-crash pattern (Lemma I2 — bundle-level idempotence).
//!
//! Per ADR-140 §"Acceptance criteria" item 5.

use std::sync::Arc;

use arcgraph_core::{PAGE_SIZE, PageId, PageType, TenantId};
use arcgraph_storage::{
    BufferedRecordPageStore, PerTenantBufferPool, PerTenantBufferPoolConfig, RecordPageBackend,
    io::{PageIo, PosixPageIo},
    wal::RecordPageStoreHandle,
};
use tempfile::TempDir;

/// Build a BufferedRecordPageStore over a PosixPageIo at `path`.
fn open_store(path: &std::path::Path, cache_cap: usize) -> Arc<BufferedRecordPageStore> {
    let io: Arc<dyn PageIo> = Arc::new(PosixPageIo::open_or_create(path).expect("posix open"));
    let pools = Arc::new(PerTenantBufferPool::with_config(
        io,
        PerTenantBufferPoolConfig {
            frames_per_tenant: 8,
            write_fraction: 0.0,
        },
    ));
    Arc::new(BufferedRecordPageStore::with_cache_cap(pools, cache_cap))
}

/// Deterministic byte pattern for `pid`. The pattern is at offset
/// `PAGE_SIZE/2` (body region) — chosen so the slotted-page header
/// CRC initialized by `install_fresh` doesn't gate the byte-equality
/// check. ADR-140 §D-4 §"Page-bytes durability boundary".
fn pattern_for(pid: PageId) -> [u8; 4] {
    let raw = pid.raw();
    [
        (raw & 0xFF) as u8,
        ((raw >> 8) & 0xFF) as u8,
        ((raw >> 16) & 0xFF) as u8,
        ((raw >> 24) & 0xFF) as u8,
    ]
}

fn imprint(store: &Arc<BufferedRecordPageStore>, pid: PageId) {
    let latch = store.latch(pid).expect("latch installed page");
    let mut g = latch.write();
    let bytes: &mut [u8; PAGE_SIZE] = g.as_mut();
    let p = pattern_for(pid);
    let base = PAGE_SIZE / 2;
    bytes[base..base + 4].copy_from_slice(&p);
    bytes[PAGE_SIZE - 4..PAGE_SIZE].copy_from_slice(&p);
}

fn verify_round_trip(store: &Arc<BufferedRecordPageStore>, pid: PageId) {
    let p = pattern_for(pid);
    // Fault-in if needed.
    store.fault_in(pid).expect("fault_in installed page");
    let latch = store.latch(pid).expect("latch installed page");
    let g = latch.read();
    let bytes: &[u8; PAGE_SIZE] = g.as_ref();
    let base = PAGE_SIZE / 2;
    assert_eq!(
        &bytes[base..base + 4],
        &p,
        "page {:?} body pattern survived crash-recovery",
        pid
    );
    assert_eq!(
        &bytes[PAGE_SIZE - 4..PAGE_SIZE],
        &p,
        "page {:?} sentinel pattern survived crash-recovery",
        pid
    );
}

#[test]
fn bytes_survive_evict_then_reopen() {
    // Step 1: tempdir + PosixPageIo file.
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("pages.db");

    const N: u64 = 32;

    // Phase A: pre-crash session.
    {
        let store = open_store(&path, 8);

        // Step 2: install N pages with known patterns.
        for i in 0..N {
            let pid = PageId::new(i + 600);
            store
                .install_fresh(pid, PageType::Node, TenantId::DEFAULT)
                .unwrap();
            imprint(&store, pid);
        }
        // Step 3: evict EVERY page to disk.
        let evicted = store.evict_lru(0).unwrap();
        assert_eq!(evicted, N as usize, "every page must evict");
        assert_eq!(store.cache_size(), 0, "cache drained");
        assert_eq!(store.evicted_count(), N as usize, "all evicted to disk");

        // Step 4: fdatasync.
        store.flush_all().unwrap();

        // Step 5: drop the store (simulated crash).
    }

    // Phase B: post-crash session — re-open the SAME file.
    let store2 = open_store(&path, 8);

    // The post-crash store has NO knowledge of installed pages (its
    // page_tenants map is empty). Simulate the WAL replay that
    // reconstructs the state: for each known page id, install_or_replace
    // with bytes pulled from the buffer pool's PageIo. In real recovery
    // this loop is driven by the WAL replay executor (ADR-140 §D-4).
    let raw_io: Arc<dyn PageIo> = store2.pools().io(TenantId::DEFAULT).unwrap();
    for i in 0..N {
        let pid = PageId::new(i + 600);
        let mut bytes = [0u8; PAGE_SIZE];
        // Slow-path read from disk (no buffer-pool involvement).
        raw_io
            .read_page(pid, &mut bytes)
            .expect("read evicted page from disk");
        // Drive through the replay path.
        let boxed: Box<[u8; PAGE_SIZE]> = Box::new(bytes);
        <BufferedRecordPageStore as RecordPageStoreHandle>::install_or_replace(&store2, pid, boxed)
            .expect("install_or_replace via replay handle");
    }

    // Step 8: verify byte-equality of every page through the new store.
    for i in 0..N {
        let pid = PageId::new(i + 600);
        verify_round_trip(&store2, pid);
    }

    // Sanity: post-replay store reports all N pages.
    assert_eq!(store2.len(), N as usize);
}

#[test]
fn install_or_replace_idempotent_under_double_replay() {
    // Lemma I2: a later bundle's `install_or_replace` for the same
    // page_id is a legitimate supersession (NOT corruption). The
    // BufferedRecordPageStore must idempotently absorb double-replay.
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("pages.db");
    let store = open_store(&path, 8);

    let pid = PageId::new(700);
    let mut bytes_v1: Box<[u8; PAGE_SIZE]> = Box::new([0u8; PAGE_SIZE]);
    bytes_v1[0] = 0xAA;
    bytes_v1[PAGE_SIZE - 1] = 0xAA;

    <BufferedRecordPageStore as RecordPageStoreHandle>::install_or_replace(
        &store,
        pid,
        bytes_v1.clone(),
    )
    .unwrap();
    // First install: page is in cache, bytes v1.
    {
        let latch = store.latch(pid).unwrap();
        let g = latch.read();
        assert_eq!(g.as_ref()[0], 0xAA);
    }

    // Double-replay v1: same bytes; cache retains them.
    <BufferedRecordPageStore as RecordPageStoreHandle>::install_or_replace(&store, pid, bytes_v1)
        .unwrap();
    {
        let latch = store.latch(pid).unwrap();
        let g = latch.read();
        assert_eq!(g.as_ref()[0], 0xAA);
    }
    assert_eq!(store.len(), 1, "double-replay does not duplicate the page");

    // Now replay v2 (supersession; Lemma I2).
    let mut bytes_v2: Box<[u8; PAGE_SIZE]> = Box::new([0u8; PAGE_SIZE]);
    bytes_v2[0] = 0xBB;
    bytes_v2[PAGE_SIZE - 1] = 0xBB;
    <BufferedRecordPageStore as RecordPageStoreHandle>::install_or_replace(&store, pid, bytes_v2)
        .unwrap();
    {
        let latch = store.latch(pid).unwrap();
        let g = latch.read();
        assert_eq!(g.as_ref()[0], 0xBB, "v2 supersedes v1");
        assert_eq!(g.as_ref()[PAGE_SIZE - 1], 0xBB);
    }
    assert_eq!(store.len(), 1);
}

#[test]
fn evicted_page_survives_per_tenant_pool_flush() {
    // Verify the durability boundary: evicted pages survive an
    // explicit `flush_all` (POSIX fdatasync) — both the page bytes
    // themselves (written direct via PageIo at evict time) AND any
    // dirty buffer-pool frames left over from cache+spill operations.
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("pages.db");
    let store = open_store(&path, 4);

    // Install 8 pages; evict to disk.
    for i in 0..8u64 {
        let pid = PageId::new(i + 800);
        store
            .install_fresh(pid, PageType::Node, TenantId::DEFAULT)
            .unwrap();
        imprint(&store, pid);
    }
    store.evict_lru(0).unwrap();
    store.flush_all().unwrap();

    // Drop store; re-open; verify bytes via direct PageIo read.
    drop(store);

    let store2 = open_store(&path, 4);
    let raw_io: Arc<dyn PageIo> = store2.pools().io(TenantId::DEFAULT).unwrap();
    for i in 0..8u64 {
        let pid = PageId::new(i + 800);
        let mut bytes = [0u8; PAGE_SIZE];
        raw_io
            .read_page(pid, &mut bytes)
            .expect("read after reopen");
        let pat = pattern_for(pid);
        let base = PAGE_SIZE / 2;
        assert_eq!(
            &bytes[base..base + 4],
            &pat,
            "page {:?} bytes survived reopen",
            pid
        );
    }
}
