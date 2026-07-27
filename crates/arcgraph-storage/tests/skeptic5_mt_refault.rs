//! SKEPTIC-5 scratch (DO NOT COMMIT) — PR #1437 adversarial probe.
//!
//! Target: the concurrent DEFAULT-ON regime the RULE-MT gate never ran.
//! The gate's `build_stack` uses the UNBOUNDED BlobStore (no spill), so
//! `maybe_drain` no-ops and `resolve_page`'s re-fault is unreachable:
//! the EVICT and GET-REFAULT writer classes were structurally absent
//! from the gate (M0.x wrong-actor lesson).
//!
//! M1 makes slotted page images EVOLVE under a fixed (tenant, page_id)
//! (pool re-checkout appends → whole-image supersession), while:
//!   - `resolve_page` re-fault = miss-check, spill READ (file IO, no
//!     lock held), then UNCONDITIONAL `pages.insert` — can clobber a
//!     newer image with the stale spill image (and marks it
//!     checkpointed=true);
//!   - `evict_one` = entry SNAPSHOT, INV-DURABLE gate on the snapshot,
//!     spill WRITE (file IO), then UNCONDITIONAL `pages.remove(&key)`
//!     — can remove a newer image installed during the write, and can
//!     overwrite the spill offset with stale bytes.
//!     Pre-M1 both races were benign: chain images are write-once, so any
//!     clobber wrote byte-identical bytes. Slotted supersession breaks that
//!     premise.
//!
//! Oracle (two clauses, both promised by the PR's own design note,
//! blob.rs "CONCURRENCY DESIGN" + the eager-publish note):
//!  1. ACKED: after `publish_txn_slotted(txn)` returns, `get(ref)` for
//!     every bag of that txn must return Ok(byte-identical), forever.
//!  2. RYOW: after `stage_bag` returns a ref, the OWNING txn's `get`
//!     of that ref must return Ok(byte-identical) before commit
//!     ("the owning txn's scans ... deref this bag BEFORE commit").
//!
//! Actors: 1 writer (single tenant, 1 bag/txn, publish each txn),
//! 1 checkpoint-capture marker (`iter_pages_resident_only` — flips
//! INV-DURABLE), N readers hammering acked refs (drive re-fault), and
//! the drain running inline on every publish (high watermark = 0).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use arcgraph_core::TenantId;
use arcgraph_storage::blob::{BlobBoundConfig, BlobSpill, BlobStore};
use arcgraph_storage::property::BlobRef;
use tempfile::tempdir;

const READERS: usize = 3;
const MAX_TXNS: u64 = 400_000;
const WALL_CAP: Duration = Duration::from_secs(75);

type AckedBlobs = Arc<Mutex<Vec<(BlobRef, Vec<u8>)>>>;

fn payload_for(i: u64) -> Vec<u8> {
    // Unique, 60-140 B (always slotted-eligible, realistic bag shape).
    let pad = "x".repeat(24 + (i % 80) as usize);
    format!(r#"{{"n":{i},"pad":"{pad}"}}"#).into_bytes()
}

#[test]
fn skeptic5_bounded_tier_refault_evict_vs_slotted_republish() {
    let tmp = tempdir().unwrap();
    let spill = Arc::new(BlobSpill::open(tmp.path()).unwrap());
    // high = 0 → every publish drains; low = 0 → drain evicts every
    // checkpoint-durable page. This is the production mechanism at
    // maximum duty cycle, not a modified mechanism.
    // Watermarks: default = max duty cycle (drain on every publish).
    // SKEPTIC5_HIGH/LOW env override for the realistic-config variant.
    let cfg = BlobBoundConfig {
        high_watermark_bytes: std::env::var("SKEPTIC5_HIGH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0),
        low_watermark_bytes: std::env::var("SKEPTIC5_LOW")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0),
    };
    let store = Arc::new(BlobStore::with_bound(spill, cfg));
    let tenant = TenantId::DEFAULT;

    let stop = Arc::new(AtomicBool::new(false));
    let acked: AckedBlobs = Arc::new(Mutex::new(Vec::new()));
    let failures: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

    // ── Checkpoint-capture marker: the ADR-229 freeze-side reader that
    //    flips INV-DURABLE bits (exactly what the RULE-MT gate ran).
    let marker = {
        let store = Arc::clone(&store);
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || {
            let mut n = 0u64;
            while !stop.load(Ordering::Acquire) {
                let (resident, _evicted) = store.iter_pages_resident_only();
                drop(resident);
                n += 1;
                std::thread::yield_now();
            }
            n
        })
    };

    // ── Readers: hammer recently-acked refs (the hot/open page) plus a
    //    periodic full sweep. Every acked ref must read Ok + identical.
    let mut readers = Vec::new();
    for r in 0..READERS {
        let store = Arc::clone(&store);
        let stop = Arc::clone(&stop);
        let acked = Arc::clone(&acked);
        let failures = Arc::clone(&failures);
        readers.push(std::thread::spawn(move || {
            let mut reads = 0u64;
            let mut sweep = 0usize;
            while !stop.load(Ordering::Acquire) {
                let snap: Vec<(BlobRef, Vec<u8>)> = {
                    let g = acked.lock().unwrap();
                    if g.is_empty() {
                        drop(g);
                        std::thread::yield_now();
                        continue;
                    }
                    // Mostly the last 96 (the active page), sometimes all.
                    sweep += 1;
                    if sweep % 64 == 0 {
                        g.clone()
                    } else {
                        let start = g.len().saturating_sub(96);
                        g[start..].to_vec()
                    }
                };
                for (bref, want) in &snap {
                    reads += 1;
                    match store.get(TenantId::DEFAULT, *bref) {
                        Ok(got) => {
                            if got.as_ref() != want.as_slice() {
                                fail_msg(
                                    &failures,
                                    &stop,
                                    format!(
                                        "reader {r}: ACKED bag WRONG BYTES at page={} slot={} (want {} B, got {} B)",
                                        bref.page_id,
                                        bref.slot_id,
                                        want.len(),
                                        got.len()
                                    ),
                                );
                            }
                        }
                        Err(e) => {
                            fail_msg(
                                &failures,
                                &stop,
                                format!(
                                    "reader {r}: ACKED bag UNREADABLE at page={} slot={}: {e}",
                                    bref.page_id, bref.slot_id
                                ),
                            );
                        }
                    }
                    if stop.load(Ordering::Acquire) {
                        break;
                    }
                }
            }
            reads
        }));
    }

    // ── Writer: real M1 lifecycle per txn — stage_bag (pool checkout /
    //    fresh page + eager publish), RYOW deref, snapshot, publish
    //    (inline drain), post-ack deref.
    let started = Instant::now();
    let mut txns_done = 0u64;
    for i in 0..MAX_TXNS {
        if stop.load(Ordering::Acquire) || started.elapsed() > WALL_CAP {
            break;
        }
        let txn_id = i + 1;
        let bytes = payload_for(i);
        let (bref, snaps) = store
            .stage_bag(tenant, txn_id, &bytes)
            .expect("stage_bag must succeed");
        assert!(snaps.is_empty(), "slotted staging defers snapshots");
        assert!(bref.slot_id >= 1, "small bag must pack slotted");

        // Clause 2 — read-your-own-writes deref BEFORE commit (the
        // documented pre-M1 chain visibility timing MERGE depends on).
        match store.get(tenant, bref) {
            Ok(got) => {
                if got.as_ref() != bytes.as_slice() {
                    fail_msg(
                        &failures,
                        &stop,
                        format!(
                            "writer RYOW WRONG BYTES txn={txn_id} page={} slot={}",
                            bref.page_id, bref.slot_id
                        ),
                    );
                    break;
                }
            }
            Err(e) => {
                fail_msg(
                    &failures,
                    &stop,
                    format!(
                        "writer RYOW UNREADABLE txn={txn_id} page={} slot={}: {e}",
                        bref.page_id, bref.slot_id
                    ),
                );
                break;
            }
        }

        // Commit: capture (unused here — no WAL in this harness), then
        // publish = ack. Inline drain runs here (high watermark 0).
        let _bundle_images = store.snapshot_txn_slotted_pages(txn_id);
        store.publish_txn_slotted(txn_id).unwrap();
        acked.lock().unwrap().push((bref, bytes.clone()));
        txns_done = txn_id;

        // Clause 1 — immediate post-ack deref by the committer itself.
        match store.get(tenant, bref) {
            Ok(got) => {
                if got.as_ref() != bytes.as_slice() {
                    fail_msg(
                        &failures,
                        &stop,
                        format!(
                            "writer post-ack WRONG BYTES txn={txn_id} page={} slot={}",
                            bref.page_id, bref.slot_id
                        ),
                    );
                    break;
                }
            }
            Err(e) => {
                fail_msg(
                    &failures,
                    &stop,
                    format!(
                        "writer post-ack UNREADABLE txn={txn_id} page={} slot={}: {e}",
                        bref.page_id, bref.slot_id
                    ),
                );
                break;
            }
        }
    }
    stop.store(true, Ordering::Release);
    let marker_runs = marker.join().unwrap();
    let mut total_reads = 0u64;
    for h in readers {
        total_reads += h.join().unwrap();
    }

    // Final full sweep: every acked bag must still read byte-identical.
    let final_acked = acked.lock().unwrap().clone();
    let mut sweep_failures = 0u64;
    for (bref, want) in &final_acked {
        match store.get(tenant, *bref) {
            Ok(got) if got.as_ref() == want.as_slice() => {}
            Ok(got) => {
                sweep_failures += 1;
                failures.lock().unwrap().push(format!(
                    "final sweep WRONG BYTES page={} slot={} want {} B got {} B",
                    bref.page_id,
                    bref.slot_id,
                    want.len(),
                    got.len()
                ));
            }
            Err(e) => {
                sweep_failures += 1;
                failures.lock().unwrap().push(format!(
                    "final sweep UNREADABLE page={} slot={}: {e}",
                    bref.page_id, bref.slot_id
                ));
            }
        }
    }

    let fails = failures.lock().unwrap();
    eprintln!(
        "skeptic5 stats: txns={txns_done} acked={} marker_captures={marker_runs} reader_gets={total_reads} evicted={} refaulted={} sweep_failures={sweep_failures}",
        final_acked.len(),
        store.evicted_count(),
        store.refault_count(),
    );
    assert!(
        fails.is_empty(),
        "acked/RYOW violations under the bounded-tier concurrent regime ({} total), first 12:\n{}",
        fails.len(),
        fails
            .iter()
            .take(12)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// Free-fn variant of the failure recorder for use inside reader closures
/// (the writer uses the capturing closure `fail`).
fn fail_msg(failures: &Arc<Mutex<Vec<String>>>, stop: &Arc<AtomicBool>, msg: String) {
    failures.lock().unwrap().push(msg);
    stop.store(true, Ordering::Release);
}

#[test]
fn guarded_refault_tolerates_aborting_owner_removing_spill_offset() {
    const READERS: usize = 6;
    const ABORTS: u64 = 10_000;

    let tmp = tempdir().unwrap();
    let spill = Arc::new(BlobSpill::open(tmp.path()).unwrap());
    let store = Arc::new(BlobStore::with_bound(
        spill,
        BlobBoundConfig {
            high_watermark_bytes: 0,
            low_watermark_bytes: 0,
        },
    ));
    let current = Arc::new(Mutex::new(None::<BlobRef>));
    let stop = Arc::new(AtomicBool::new(false));

    let mut readers = Vec::new();
    for _ in 0..READERS {
        let store = Arc::clone(&store);
        let current = Arc::clone(&current);
        let stop = Arc::clone(&stop);
        readers.push(std::thread::spawn(move || {
            let mut reads = 0u64;
            while !stop.load(Ordering::Acquire) {
                let bref = *current.lock().unwrap();
                if let Some(bref) = bref {
                    // Missing is expected once rollback wins; only a panic is
                    // a failure, and join propagation below makes it fatal.
                    let _ = store.get(TenantId::DEFAULT, bref);
                    reads += 1;
                } else {
                    std::thread::yield_now();
                }
            }
            reads
        }));
    }

    for txn_id in 1..=ABORTS {
        let bytes = payload_for(txn_id);
        let (bref, snaps) = store
            .stage_bag(TenantId::DEFAULT, txn_id, &bytes)
            .expect("stage fresh bag");
        assert!(snaps.is_empty());
        let _ = store.iter_pages_resident_only();
        store.force_drain_for_test().unwrap();

        *current.lock().unwrap() = Some(bref);
        for _ in 0..4 {
            std::thread::yield_now();
        }
        store.rollback_txn_slotted(txn_id);
        *current.lock().unwrap() = None;
    }

    stop.store(true, Ordering::Release);
    let reads: u64 = readers
        .into_iter()
        .map(|reader| {
            reader
                .join()
                .expect("reader panicked during abort/refault race")
        })
        .sum();
    assert!(reads > 0, "readers must exercise the abort/refault race");
    assert!(
        store.evicted_count() >= ABORTS,
        "every fresh page should be evicted before abort"
    );
}
