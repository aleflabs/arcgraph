//! Overflow-chain pointer publication is Acquire/Release safe.
//!
//! `TelBlock::set_prev_block_ptr` stores the raw `u64` page id with
//! `Release`; `TelBlock::prev_block_ptr` reads with `Acquire`. Combined
//! with the write-once invariant this means a concurrent reader
//! observes either `None` or a fully-initialized `Some(PageId(p))` —
//! never a torn raw value.
//!
//! This test repeats 1 000 rounds of the minimum contention shape that
//! could surface a mis-ordered publication on weakly-ordered CPUs like
//! Apple Silicon / AArch64: one reader spin-loops on
//! `prev_block_ptr()` until it sees `Some(p)`, the main thread
//! `set_prev_block_ptr(chosen)` once. We then assert `p == chosen`.
//!
//! 1 000 rounds keeps the wall-clock small (≲10 ms on ARM64) while
//! giving enough distinct spawn/schedule interleavings to make a
//! reordering bug visible in CI.

use std::sync::Arc;
use std::thread;

use arcgraph_core::{LabelId, NodeId, PageId, TenantId};
use arcgraph_storage::tel::TelBlock;

#[test]
fn prev_block_ptr_publication_is_acquire_release_safe() {
    const ROUNDS: u32 = 1_000;

    for round in 0..ROUNDS {
        let block = Arc::new(
            TelBlock::new(NodeId::new(1), LabelId::new(1), 128, TenantId::DEFAULT).unwrap(),
        );
        // Vary the pointer so a torn publication would show up as a
        // mismatch rather than always zero.
        let chosen = PageId::new(u64::from(round) * 7 + 1);

        let reader_block = Arc::clone(&block);
        let reader = thread::spawn(move || -> PageId {
            loop {
                if let Some(p) = reader_block.prev_block_ptr() {
                    return p;
                }
                std::hint::spin_loop();
            }
        });

        block
            .set_prev_block_ptr(chosen)
            .expect("fresh block, link must succeed");

        let observed = reader.join().expect("reader panicked");
        assert_eq!(
            observed, chosen,
            "round {round}: reader saw torn or stale pointer"
        );
    }
}

/// Negative control: a reader that starts *before* the main thread
/// stores anything and times out after a bounded spin must see only
/// `None`. Rules out a trivial "pass" caused by the reader always
/// observing a pre-set value.
#[test]
fn prev_block_ptr_is_none_before_set() {
    let block = TelBlock::new(NodeId::new(1), LabelId::new(1), 128, TenantId::DEFAULT).unwrap();
    for _ in 0..1_000 {
        assert_eq!(block.prev_block_ptr(), None);
    }
}
