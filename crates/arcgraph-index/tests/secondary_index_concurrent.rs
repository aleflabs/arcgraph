//! Concurrency linearizability check for [`SecondaryIndex`] (P-3 from
//! the M2-D5 alt review).
//!
//! Matches the shape of
//! `arcgraph-storage::tests::primary_index_concurrent`, with the
//! secondary-specific twist that overflow-chain appends under
//! duplicate pressure exercise the tail-cache (DEC-22) under
//! contention.
//!
//! The tree serializes all mutations through `write_gate: Mutex<()>`
//! (DEC-17); this test is primarily a no-loom stress — if the
//! protocol is correct, every concurrently-inserted NodeId ends up
//! indexed and visible to later lookups.

use std::sync::Arc;
use std::thread;

use arcgraph_core::{LabelId, NodeId, StringId, TenantId};
use arcgraph_index::{PropertyValue, SecondaryIndex, SecondaryKey};
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::transaction::TxnManager;

fn build() -> Arc<SecondaryIndex> {
    let txn_mgr = Arc::new(TxnManager::new());
    let alloc = Arc::new(PageAllocator::new());
    Arc::new(SecondaryIndex::new(txn_mgr, alloc, None).expect("fresh secondary"))
}

fn key(label: u32, v: u32) -> SecondaryKey {
    SecondaryKey::new(
        TenantId::DEFAULT,
        LabelId::new(label),
        StringId::new(1),
        PropertyValue::U32(v),
    )
}

#[test]
fn concurrent_disjoint_range_inserts_all_land() {
    // 4 threads × 200 disjoint-key inserts = 800 unique
    // `(tenant, label, property_key, value)` tuples. Every tuple
    // must be findable after all writers publish.
    const THREADS: u32 = 4;
    const PER_THREAD: u32 = 200;
    let idx = build();

    let mut handles = Vec::new();
    for t in 0..THREADS {
        let idx = Arc::clone(&idx);
        let start = t * PER_THREAD;
        let end = start + PER_THREAD;
        handles.push(thread::spawn(move || {
            for i in start..end {
                idx.insert(key(1, i), NodeId::new(u64::from(i) + 1))
                    .unwrap_or_else(|e| panic!("thread {t} insert {i}: {e}"));
            }
        }));
    }
    for h in handles {
        h.join().expect("writer thread panicked");
    }

    let total = THREADS * PER_THREAD;
    for i in 0..total {
        let hits = idx.lookup(key(1, i)).unwrap();
        assert_eq!(
            hits,
            vec![NodeId::new(u64::from(i) + 1)],
            "value {i} missing or wrong after concurrent inserts"
        );
    }
    assert!(idx.lookup(key(1, total + 1)).unwrap().is_empty());
}

#[test]
fn concurrent_duplicate_inserts_saturate_chain_without_loss() {
    // Stress the tail-cache + overflow-chain crabbing under
    // duplicate-heavy contention. Four writers each push 400
    // NodeIds into ONE bucket `(label=1, value=7)`. After the test
    // every NodeId must be present; no writer's insert can be lost.
    //
    // Distinct NodeId ranges per writer prevent the "insert doesn't
    // dedup, remove doesn't balance inline vs chain" confusion —
    // each writer uses IDs in its own slice of the NodeId space.
    const WRITERS: u64 = 4;
    const PER_WRITER: u64 = 400;
    let idx = build();
    let bucket_key = key(1, 7);

    let mut handles = Vec::new();
    for w in 0..WRITERS {
        let idx = Arc::clone(&idx);
        let start = 1 + w * PER_WRITER; // NodeId 0 is the reserved sentinel.
        let end = start + PER_WRITER;
        handles.push(thread::spawn(move || {
            for i in start..end {
                idx.insert(bucket_key, NodeId::new(i))
                    .unwrap_or_else(|e| panic!("writer {w} insert {i}: {e}"));
            }
        }));
    }
    for h in handles {
        h.join().expect("writer thread panicked");
    }

    let hits = idx.lookup(bucket_key).unwrap();
    assert_eq!(
        hits.len(),
        (WRITERS * PER_WRITER) as usize,
        "every concurrent duplicate insert must be findable"
    );
    // And every specific id is present.
    for w in 0..WRITERS {
        for i in (1 + w * PER_WRITER)..(1 + (w + 1) * PER_WRITER) {
            assert!(
                hits.contains(&NodeId::new(i)),
                "NodeId({i}) from writer {w} missing"
            );
        }
    }
}

#[test]
fn concurrent_readers_never_see_partial_state() {
    // 2 inserters stream disjoint-value keys; 2 lookup threads spin
    // in parallel. A reader's `lookup(k)` must return either an
    // empty Vec (tuple not yet inserted) or `vec![expected_node]`
    // (tuple published) — never an arbitrary / torn value.
    const WRITERS: u32 = 2;
    const PER_WRITER: u32 = 400;
    const READERS: usize = 2;
    const READ_ITERS: usize = 2_000;

    let idx = build();
    let total = WRITERS * PER_WRITER;

    let mut handles = Vec::new();
    for w in 0..WRITERS {
        let idx = Arc::clone(&idx);
        handles.push(thread::spawn(move || {
            for j in 0..PER_WRITER {
                let v = j * WRITERS + w; // interleaved
                idx.insert(key(1, v), NodeId::new(u64::from(v) + 1))
                    .unwrap_or_else(|e| panic!("writer {w} insert {v}: {e}"));
            }
        }));
    }
    for _ in 0..READERS {
        let idx = Arc::clone(&idx);
        handles.push(thread::spawn(move || {
            for _ in 0..READ_ITERS {
                for v in 0..total {
                    let hits = idx.lookup(key(1, v)).unwrap();
                    match hits.len() {
                        0 => {} // not yet inserted
                        1 => assert_eq!(
                            hits[0],
                            NodeId::new(u64::from(v) + 1),
                            "torn read on value {v}: saw {hits:?}"
                        ),
                        other => panic!("unexpected hit count {other} on value {v}: {hits:?}"),
                    }
                }
            }
        }));
    }
    for h in handles {
        h.join().expect("thread panicked");
    }
    // Final consistency.
    for v in 0..total {
        assert_eq!(
            idx.lookup(key(1, v)).unwrap(),
            vec![NodeId::new(u64::from(v) + 1)],
            "value {v} missing after concurrent writers"
        );
    }
}
