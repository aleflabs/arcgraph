//! ROOT CAUSE A gate — the lock-free owner-forward-index unlink race.
//!
//! `OwnerForwardIndex::for_each_candidate` clones the published run set under a
//! short read lock, RELEASES the lock, and only then opens each run file. The
//! writer (`insert_chunk`) publishes the merged replacement and then `unlink`s
//! the retired runs with no refcount, deferred delete, or grace period. A reader
//! holding a stale `RunMeta` therefore races the unlink and its `File::open`
//! returns `ENOENT`.
//!
//! That `ENOENT` is NOT "absent" — the entries live on in the merged replacement
//! — but pre-fix it escaped as an error and the callers laundered it:
//!   * `intern.rs` returned the reserved `STRINGID_SENTINEL` (id 0) as though the
//!     string had been interned → durable data corruption;
//!   * the projection / label-filter probes mapped it to `None` → a silently
//!     wrong query answer.
//!
//! # Why the primary gate injects the fault instead of racing threads
//!
//! The window between the snapshot and the `open` is a few instructions wide.
//! An 8-reader / 700k-lookup thread race against a continuously-merging writer
//! did NOT reproduce it once — and it passed even with the fix reverted. A
//! RED-on-revert that fires only probabilistically is not a gate.
//!
//! The DETERMINISTIC gate lives in-crate at
//! `owner_index::tests::gate_reader_survives_run_retired_under_it`, because its
//! fault seam is `#[cfg(any(test, feature = "fault-injection"))]` and must not
//! be reachable from a production build. This file keeps the parts that need no
//! seam: the real multi-threaded churn, and the disk-cap bound.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use arcgraph_storage::{OWNER_INDEX_DISK_CAP_BYTES, OwnerForwardIndex, str_hash_56};
use tempfile::tempdir;

const CHUNK_ENTRIES: u64 = 400;

/// Keys inserted FIRST, so they are durably published in some run for the whole
/// test — every lookup must resolve, at every instant.
const RESIDENT_KEYS: [&str; 4] = [
    "m4-1492-resident-alpha",
    "m4-1492-resident-beta",
    "m4-1492-resident-gamma",
    "m4-1492-resident-delta",
];

fn key_id(index: u64) -> u64 {
    9_000_000 + index
}

fn churn_chunk(index: &OwnerForwardIndex, chunk: u64) {
    let base = chunk * CHUNK_ENTRIES;
    index
        .insert_batch((0..CHUNK_ENTRIES).map(|i| {
            let n = base + i;
            (str_hash_56(&format!("m4-1492-churn-{n}")), 1_000_000 + n)
        }))
        .expect("insert_batch");
}

fn seed_resident(index: &OwnerForwardIndex) {
    index
        .insert_batch(
            RESIDENT_KEYS
                .iter()
                .enumerate()
                .map(|(i, k)| (str_hash_56(k), key_id(i as u64))),
        )
        .expect("seed resident keys");
}

fn lookup(index: &OwnerForwardIndex, key: &str, expected: u64) -> Option<u64> {
    index
        .for_each_candidate(str_hash_56(key), |candidate| Ok(candidate == expected))
        .unwrap_or_else(|error| panic!("lookup FAILED under a retired run (key {key}): {error}"))
}

/// Realism companion: the genuine multi-threaded churn. Not the oracle (the race
/// is too narrow to hit reliably — see the module note), but it must never fail
/// or lose a key.
#[test]
fn gate_owner_index_concurrent_churn_stress() {
    let dir = tempdir().unwrap();
    let index = Arc::new(
        OwnerForwardIndex::create(dir.path(), OWNER_INDEX_DISK_CAP_BYTES).expect("create index"),
    );
    seed_resident(&index);

    let stop = Arc::new(AtomicBool::new(false));
    let lookups = Arc::new(AtomicU64::new(0));

    let writer = {
        let index = Arc::clone(&index);
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || {
            for chunk in 0..60 {
                churn_chunk(&index, chunk);
            }
            stop.store(true, Ordering::Release);
        })
    };

    let readers: Vec<_> = (0..8)
        .map(|_| {
            let index = Arc::clone(&index);
            let stop = Arc::clone(&stop);
            let lookups = Arc::clone(&lookups);
            std::thread::spawn(move || {
                while !stop.load(Ordering::Acquire) {
                    for (i, key) in RESIDENT_KEYS.iter().enumerate() {
                        let expected = key_id(i as u64);
                        assert_eq!(
                            lookup(&index, key, expected),
                            Some(expected),
                            "key {key} vanished under concurrent run retirement"
                        );
                        lookups.fetch_add(1, Ordering::Relaxed);
                    }
                }
            })
        })
        .collect();

    writer.join().expect("writer thread");
    for reader in readers {
        reader.join().expect("reader thread");
    }

    assert!(lookups.load(Ordering::Relaxed) > 0, "no lookups ran");
    for (i, key) in RESIDENT_KEYS.iter().enumerate() {
        let expected = key_id(i as u64);
        assert_eq!(
            lookup(&index, key, expected),
            Some(expected),
            "key {key} lost after churn settled"
        );
    }
}

/// DEFERRAL-SAFETY gate for #1493 (the owner forward index never reclaims dead
/// candidates). PR #1492 defers compaction. That deferral is only safe if the
/// unreclaimed growth is actually BOUNDED today — not merely "expected to be
/// fine". This gate proves the hard disk cap is enforced and that the on-disk
/// footprint never exceeds it.
///
/// The failure this rules out is the dangerous one: growth that is silently
/// unbounded, where a rebind-heavy workload fills the disk. What we get instead
/// is a LOUD `DiskBudgetExceeded` at a known ceiling.
///
/// RED-on-revert: delete the `enforce_budget` call in `write_entries_run` — the
/// index then grows past its cap and this gate fails with "UNBOUNDED".
#[test]
fn gate_owner_index_disk_cap_bounds_unreclaimed_growth() {
    const CAP: u64 = 1 << 20; // 1 MiB — small enough to reach deterministically
    const MAX_CHUNKS: u64 = 100_000;

    let dir = tempdir().unwrap();
    let index = OwnerForwardIndex::create(dir.path(), CAP).expect("create index");

    fn directory_bytes(path: &std::path::Path) -> u64 {
        std::fs::read_dir(path)
            .unwrap()
            .filter_map(Result::ok)
            .filter_map(|entry| entry.metadata().ok())
            .filter(std::fs::Metadata::is_file)
            .map(|meta| meta.len())
            .sum()
    }

    // Keep inserting DISTINCT candidates (none are ever reclaimed) until the
    // budget refuses the write.
    let mut chunk = 0_u64;
    let error = loop {
        let base = chunk * CHUNK_ENTRIES;
        let result = index.insert_batch((0..CHUNK_ENTRIES).map(|i| {
            let n = base + i;
            (str_hash_56(&format!("m4-1493-bloat-{n}")), 2_000_000 + n)
        }));
        match result {
            Ok(()) => {
                chunk += 1;
                assert!(
                    chunk < MAX_CHUNKS,
                    "UNBOUNDED: wrote {} candidates without the {CAP}-byte cap ever \
                     refusing a write — dead-candidate growth is not bounded, so \
                     deferring reclamation (#1493) is NOT safe",
                    chunk * CHUNK_ENTRIES
                );
            }
            Err(error) => break error,
        }
    };

    assert!(
        matches!(
            error,
            arcgraph_storage::OwnerIndexError::DiskBudgetExceeded { .. }
        ),
        "growth must stop with a LOUD budget error, got {error:?}"
    );
    assert!(
        chunk > 0,
        "the cap refused the very first write — gate is vacuous"
    );

    let bytes = directory_bytes(dir.path());
    assert!(
        bytes <= CAP,
        "index footprint {bytes} exceeded its own {CAP}-byte cap"
    );

    // The index is still READABLE at the cap — refusing writes must not corrupt
    // or strand the already-published candidates. (Writing is *supposed* to fail
    // here; that is the whole point of the cap.)
    let probe = index
        .for_each_candidate(str_hash_56("m4-1493-bloat-0"), |candidate| {
            Ok(candidate == 2_000_000)
        })
        .expect("a capped-out index must still serve reads");
    assert_eq!(
        probe,
        Some(2_000_000),
        "candidates published before the cap must remain readable after it"
    );
}
