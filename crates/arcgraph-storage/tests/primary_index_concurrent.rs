//! Concurrency linearizability check for [`PrimaryIndex`] (M2-33 item 8).
//!
//! Philosophy §F: pessimistic lock coupling means every write serializes
//! on the tree's `write_gate`, every page mutation is under a per-page
//! write latch, and concurrent readers hand-over-hand read latches. The
//! tests below exercise that no-loom-required stress: if the protocol is
//! correct, every concurrently-inserted key is findable once all writers
//! have published.

use std::sync::Arc;
use std::thread;

use arcgraph_core::{PageId, TenantId};
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::{PageSlot, PrimaryIndex, PrimaryKey, RecordKind};
use arcgraph_storage::records::SlotId;
use arcgraph_storage::transaction::TxnManager;

fn build() -> Arc<PrimaryIndex> {
    let txn_mgr = Arc::new(TxnManager::new());
    let alloc = Arc::new(PageAllocator::new());
    Arc::new(PrimaryIndex::new(txn_mgr, alloc, None).expect("fresh index"))
}

fn key(tenant: u64, id: u64) -> PrimaryKey {
    PrimaryKey::new(TenantId::new(tenant), RecordKind::Node, id)
}

fn slot(page: u64, slot: u16) -> PageSlot {
    PageSlot::new(PageId::new(page), SlotId(slot))
}

#[test]
fn concurrent_disjoint_range_inserts_all_land() {
    // 4 threads × 100 disjoint-range inserts = 400 keys total. Every
    // key must be findable after all writers have published. Keys do
    // NOT overlap across threads, so a correct crab-locked tree has no
    // duplicate-key contention — only structural-lock contention on
    // ancestor splits.
    const THREADS: u64 = 4;
    const PER_THREAD: u64 = 100;
    let idx = build();

    let mut handles = Vec::new();
    for t in 0..THREADS {
        let idx = Arc::clone(&idx);
        let start = t * PER_THREAD;
        let end = start + PER_THREAD;
        handles.push(thread::spawn(move || {
            for i in start..end {
                idx.insert(key(1, i), slot(i, 0))
                    .unwrap_or_else(|e| panic!("thread {t} insert {i}: {e}"));
            }
        }));
    }
    for h in handles {
        h.join().expect("writer thread panicked");
    }

    // Verification pass from the main thread.
    let total = THREADS * PER_THREAD;
    for i in 0..total {
        let found = idx.lookup(key(1, i)).unwrap();
        assert_eq!(
            found,
            Some(slot(i, 0)),
            "key {i} missing or wrong after concurrent inserts"
        );
    }
    // Out-of-range key absent.
    assert_eq!(idx.lookup(key(1, total + 1)).unwrap(), None);
}

#[test]
fn concurrent_readers_never_see_partial_state() {
    // 2 inserters stream keys into overlapping ranges; 2 lookup threads
    // spin in parallel. The lookup threads must never observe a TORN
    // read: `lookup(k)` at any point in time returns either `None`
    // (key hasn't been inserted yet) or `Some(slot(k, 0))` (key was
    // inserted by one of the writers) — never any other value.
    const WRITERS: u64 = 2;
    const PER_WRITER: u64 = 200;
    const READERS: usize = 2;
    const READ_ITERS: usize = 5_000;

    let idx = build();
    let total = WRITERS * PER_WRITER;

    let mut handles = Vec::new();
    for w in 0..WRITERS {
        let idx = Arc::clone(&idx);
        // Interleave the two writers' keys by mod-2 so they stress the
        // same ancestor nodes rather than partitioning the tree.
        handles.push(thread::spawn(move || {
            for j in 0..PER_WRITER {
                let id = j * WRITERS + w;
                idx.insert(key(1, id), slot(id, 0))
                    .unwrap_or_else(|e| panic!("writer {w} insert {id}: {e}"));
            }
        }));
    }
    let stop_val = total;
    for _ in 0..READERS {
        let idx = Arc::clone(&idx);
        handles.push(thread::spawn(move || {
            for _ in 0..READ_ITERS {
                for id in 0..stop_val {
                    match idx.lookup(key(1, id)).unwrap() {
                        None => {}
                        Some(found) => {
                            assert_eq!(found, slot(id, 0), "torn read on key {id}: saw {found:?}");
                        }
                    }
                }
            }
        }));
    }
    for h in handles {
        h.join().expect("thread panicked");
    }
    // Final consistency: every key is present.
    for id in 0..total {
        assert_eq!(
            idx.lookup(key(1, id)).unwrap(),
            Some(slot(id, 0)),
            "key {id} missing after concurrent writers"
        );
    }
}
