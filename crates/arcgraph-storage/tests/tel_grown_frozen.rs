//! Post-grow invariant: once `TelBlock::grown()` produces a successor
//! and the caller migrates further appends to it, scans against the
//! original block must observe a frozen snapshot.
//!
//! This is the invariant M2.c's buffer-pool page-eviction path relies
//! on: after the writer departs from a block (because the block is
//! full and a larger successor has been installed), that block is
//! effectively immutable, and reads against it — potentially on a
//! different thread, racing with writes to the successor — remain
//! consistent.
//!
//! Two cases:
//! - `grown_original_is_frozen_single_threaded`: sequential check of
//!   the pre-grow / post-grow contents on both blocks.
//! - `grown_original_is_frozen_under_concurrent_writer`: reader thread
//!   loops scanning the original while the main thread appends to the
//!   successor. Every scan of the original must be the same bounded
//!   prefix as the pre-grow snapshot.

use std::sync::Arc;
use std::thread;

use arcgraph_core::{LabelId, Lsn, NodeId, TelEntry, TenantId};
use arcgraph_storage::tel::{MIN_BLOCK_BYTES, TelBlock};

fn entry(i: u64) -> TelEntry {
    TelEntry {
        dst_id: 1000 + i,
        rel_id: i,
        created_lsn: i + 1,
        expired_lsn: u64::MAX,
    }
}

fn collect_scan(b: &TelBlock) -> Vec<TelEntry> {
    b.scan(Lsn::new(u64::MAX - 1)).collect()
}

#[test]
fn grown_original_is_frozen_single_threaded() {
    // MIN_BLOCK_BYTES holds exactly one entry; grown() doubles it.
    let original = TelBlock::new(
        NodeId::new(1),
        LabelId::new(1),
        MIN_BLOCK_BYTES,
        TenantId::DEFAULT,
    )
    .unwrap();
    assert_eq!(original.capacity_entries(), 1);
    original.append(entry(0)).unwrap();

    let new_block = original
        .grown()
        .expect("grown must succeed below MAX_BLOCK_BYTES");

    // MIN_BLOCK_BYTES (64) doubles to 128 bytes → capacity 3 entries.
    // grown() already copied entry 0, so there is headroom for 2 more.
    for i in 1..3u64 {
        new_block.append(entry(i)).unwrap();
    }

    // Original keeps exactly its pre-grow contents.
    let orig_scan = collect_scan(&original);
    assert_eq!(orig_scan, vec![entry(0)]);
    assert_eq!(original.entry_count(), 1);

    // Successor holds pre-grow followed by post-grow.
    let new_scan = collect_scan(&new_block);
    let expected: Vec<TelEntry> = (0..3u64).map(entry).collect();
    assert_eq!(new_scan, expected);
}

#[test]
fn grown_original_is_frozen_under_concurrent_writer() {
    // Slightly larger original so we have a non-trivial prefix to
    // re-scan in a tight loop on the reader thread.
    let block_size = MIN_BLOCK_BYTES + 32 * 7; // holds 8 entries.
    let original = TelBlock::new(
        NodeId::new(1),
        LabelId::new(1),
        block_size,
        TenantId::DEFAULT,
    )
    .unwrap();
    for i in 0..original.capacity_entries() as u64 {
        original.append(entry(i)).unwrap();
    }
    let pre_grow: Vec<TelEntry> = collect_scan(&original);
    assert_eq!(pre_grow.len(), 8);

    let original = Arc::new(original);
    let new_block = Arc::new(original.grown().expect("grown below cap"));

    // Reader runs a fixed scan budget so overlap with the main-thread
    // appends is guaranteed without relying on a stop-flag race.
    const READER_SCANS: u64 = 10_000;
    let reader_orig = Arc::clone(&original);
    let expected_frozen = pre_grow.clone();
    let reader = thread::spawn(move || {
        for _ in 0..READER_SCANS {
            let got = collect_scan(&reader_orig);
            assert_eq!(
                got, expected_frozen,
                "original block mutated after grown() — observed {got:?}"
            );
        }
    });

    // Main thread hammers the successor. Start at index 8 (past the
    // copied prefix) and write enough entries to fill the new block's
    // extra headroom. `new_block` doubles MIN_BLOCK_BYTES + 7*32 =
    // 288 → 576 bytes, capacity 17 entries, so 9 fresh appends fit.
    for i in 8u64..17 {
        new_block.append(entry(i)).unwrap();
    }
    reader.join().expect("reader panicked");

    // Successor's final state: full 17 entries in insertion order.
    let final_new = collect_scan(&new_block);
    let expected_new: Vec<TelEntry> = (0..17u64).map(entry).collect();
    assert_eq!(final_new, expected_new);
}
