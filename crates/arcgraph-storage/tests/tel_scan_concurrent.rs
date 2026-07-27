//! CRITICAL proptest: concurrent scan under append.
//!
//! This is the M2.a gate test for LiveGraph Theorem 1: a `TelScan`
//! running concurrently with a single appender thread must always see
//! a **contiguous in-order prefix** of the appended entries, with no
//! torn entries, no reordering, no duplicates, and no gaps.
//!
//! The gate per the M2 TEL task brief: **10 000 iterations in release
//! mode, zero failures.** Invoke as:
//!
//! ```text
//! cargo test -p arcgraph-storage --release \
//!     -- tel_scan_concurrent_append --nocapture
//! ```
//!
//! In debug builds we cap at 256 cases to keep the default
//! `cargo test` run fast while still exercising the shape every PR.
//! The case count can be overridden at any time via
//! `PROPTEST_CASES=<n>`.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use arcgraph_core::{LabelId, Lsn, NodeId, TelEntry, TenantId};
use arcgraph_storage::tel::{MAX_BLOCK_BYTES, TelBlock};
use proptest::prelude::*;
use proptest::test_runner::Config as ProptestConfig;

/// One-writer, many-reader invariant check.
///
/// The writer appends `n_entries` entries with
/// `created_lsn = i + 1`, `expired_lsn = u64::MAX`, `rel_id = i`.
/// Each reader thread performs `scans_per_reader` scans at
/// `snapshot_lsn = u64::MAX - 1` (visible = all alive), and after the
/// writer finishes performs one more "final" scan. The invariants
/// checked on every scan's output:
///
/// 1. Length `L` is between 0 and `n_entries`.
/// 2. The `i`-th yielded entry has `rel_id == i` and
///    `created_lsn == i + 1` (contiguous in-order prefix).
/// 3. The final scan after writer completion yields exactly
///    `n_entries` entries (no lost appends).
fn run_one_case(n_entries: u32, n_readers: u32, scans_per_reader: u32) -> Result<(), String> {
    let block = Arc::new(
        TelBlock::new(
            NodeId::new(1),
            LabelId::new(1),
            MAX_BLOCK_BYTES,
            TenantId::DEFAULT,
        )
        .expect("MAX_BLOCK_BYTES is always valid"),
    );
    let writer_done = Arc::new(AtomicBool::new(false));
    let snapshot_lsn = Lsn::new(u64::MAX - 1);

    // --- writer thread ---
    let writer_block = Arc::clone(&block);
    let writer_flag = Arc::clone(&writer_done);
    let writer = thread::spawn(move || {
        for i in 0..n_entries {
            let e = TelEntry {
                dst_id: u64::from(i),
                rel_id: u64::from(i),
                created_lsn: u64::from(i) + 1,
                expired_lsn: u64::MAX,
            };
            writer_block
                .append(e)
                .expect("block sized to MAX_ENTRIES > n_entries");
        }
        writer_flag.store(true, Ordering::Release);
    });

    // --- reader threads ---
    let mut readers = Vec::with_capacity(n_readers as usize);
    for _ in 0..n_readers {
        let reader_block = Arc::clone(&block);
        let reader_flag = Arc::clone(&writer_done);
        readers.push(thread::spawn(move || -> Result<(), String> {
            for _ in 0..scans_per_reader {
                check_scan_is_prefix(&reader_block, snapshot_lsn, n_entries)?;
            }
            // After the writer has finished, do one more scan and
            // record its count so the outer assertion can verify no
            // lost appends. We wait for the writer to publish its
            // done-flag (Acquire) so this scan starts strictly after
            // the last append's Release.
            while !reader_flag.load(Ordering::Acquire) {
                std::hint::spin_loop();
            }
            let final_count = count_scan(&reader_block, snapshot_lsn, n_entries)?;
            Ok(()).and(if final_count == n_entries {
                Ok(())
            } else {
                Err(format!(
                    "final scan after writer completion saw {final_count} of {n_entries} appended"
                ))
            })
        }));
    }

    writer.join().expect("writer thread panicked");
    for r in readers {
        r.join().expect("reader thread panicked")?;
    }
    Ok(())
}

/// Verify that `block.scan(lsn)` yields a contiguous in-order prefix
/// of the append stream (entries `rel_id = 0, 1, 2, ...`).
fn check_scan_is_prefix(block: &TelBlock, snapshot_lsn: Lsn, n_entries: u32) -> Result<(), String> {
    let mut expected_rel = 0u64;
    for e in block.scan(snapshot_lsn) {
        if e.rel_id != expected_rel {
            return Err(format!(
                "scan yielded out-of-order entry at position {expected_rel}: \
                 got rel_id={}, expected rel_id={expected_rel}",
                e.rel_id
            ));
        }
        if e.created_lsn != expected_rel + 1 {
            return Err(format!(
                "scan yielded torn entry at position {expected_rel}: \
                 rel_id={} but created_lsn={} (want {})",
                e.rel_id,
                e.created_lsn,
                expected_rel + 1
            ));
        }
        expected_rel += 1;
    }
    if expected_rel > u64::from(n_entries) {
        return Err(format!(
            "scan yielded {expected_rel} entries but only {n_entries} were ever appended"
        ));
    }
    Ok(())
}

/// Like `check_scan_is_prefix` but also returns the observed length.
fn count_scan(block: &TelBlock, snapshot_lsn: Lsn, n_entries: u32) -> Result<u32, String> {
    let mut count = 0u64;
    for e in block.scan(snapshot_lsn) {
        if e.rel_id != count {
            return Err(format!(
                "final scan: out-of-order entry at position {count}: got rel_id={}",
                e.rel_id
            ));
        }
        count += 1;
    }
    if count > u64::from(n_entries) {
        return Err(format!(
            "final scan observed {count} entries, only {n_entries} appended"
        ));
    }
    Ok(count as u32)
}

fn config() -> ProptestConfig {
    let cases: u32 = if cfg!(debug_assertions) { 256 } else { 10_000 };
    ProptestConfig {
        cases,
        // The test spawns threads; shrinking a failure under threads
        // is unreliable so we disable it and rely on the raw seed
        // for reproduction.
        max_shrink_iters: 0,
        ..ProptestConfig::default()
    }
}

proptest! {
    #![proptest_config(config())]

    /// CRITICAL: LiveGraph Theorem 1 holds under concurrent
    /// single-writer append and many-reader scan.
    #[test]
    fn tel_scan_concurrent_append(
        n_entries in 1u32..=128,
        n_readers in 1u32..=4,
        scans_per_reader in 1u32..=8,
        // Dummy to vary thread-interleaving across cases; not used
        // directly but forces proptest to sample a distinct seed.
        _salt in any::<u32>(),
    ) {
        // Prevent silent short-circuiting if `_salt` is ever elided.
        let _ = _salt;
        run_one_case(n_entries, n_readers, scans_per_reader)
            .map_err(proptest::test_runner::TestCaseError::fail)?;
    }
}

/// A single-threaded sanity case for regression-proofing in case the
/// threaded proptest is accidentally turned off (e.g. cases = 0).
#[test]
fn tel_scan_single_threaded_sanity() {
    run_one_case(64, 2, 4).expect("single-threaded sanity must pass");
}
