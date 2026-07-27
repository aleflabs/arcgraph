//! M6.1 — `m6_lookup_owner_paired_eviction` (§M6 EXIT 2c;
//! ADR-232-amendment-01 §2.3).
//!
//! `IdempotencyStore` is the intern/idempotency page-store-backed,
//! cache-bounded owner registry named in the M6.1 build-plan leg: a
//! bounded resident forward map (`external_id -> internal_id`) paired
//! with a resident reverse map (`internal_id -> external_id`), spilling
//! BOTH directions from one durable append (`IdempotencySpill::write_binding`)
//! on eviction (`evict_one`/`evict_one_claimed`).
//!
//! RULE-MT (≥8 concurrent writers): 8 ingest threads install bindings
//! while a dedicated drainer thread continuously marks bindings durable
//! and forces eviction (`force_drain_for_test`), WHILE a dedicated
//! lookup thread continuously round-trips `external_id_for` (reverse
//! lookup — the delete-path consumer named in ADR-232-amendment-01 §2.2)
//! against already-installed keys, all running concurrently — not
//! serially staged. The oracle at the end is exact: every installed
//! `(external_id, internal_id)` pair resolves BOTH directions
//! (`get(external_id) == internal_id` AND `external_id_for(internal_id)
//! == external_id`), whether the pair currently lives resident or spilled.
//!
//! RED-on-revert: `evict_one_claimed`'s spill write normally indexes BOTH
//! forward and reverse directions from one durable append
//! (`IdempotencySpill::write_binding`). Temporarily disabling ONLY the
//! reverse-index insert there (the ADR-232-amendment-01 §2.2 "evict one
//! side to nowhere" violation) leaves `external_id_for` unable to
//! resolve an evicted binding's external_id — exactly the delete-path
//! at-least-once identity corruption the epic names. Confirmed failing
//! under that mutation (`reverse lookup lost id 0`) and green after
//! revert. The exhaustive final-oracle loop deliberately calls
//! `external_id_for` BEFORE `get()` for each id: `get()`'s forward
//! spill re-fault re-warms the resident reverse map as a side effect
//! (see `resolve_binding`'s `insert_resident_warm_if_vacant` +
//! reverse-populate), so checking forward first would launder a broken
//! reverse spill index through the (unaffected) forward path and make
//! the gate vacuous against exactly the mutant it exists to catch.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use arcgraph_core::TenantId;
use arcgraph_storage::idempotency::{
    IDEMPOTENCY_BINDING_WEIGHT_BYTES, IdempotencyBoundConfig, IdempotencySpill, IdempotencyStore,
};
use tempfile::tempdir;

const NODE: u8 = 0;
const WRITERS: u64 = 8; // RULE-MT floor.
const PER_WRITER: u64 = 1500;
const WAIT: Duration = Duration::from_secs(60);

fn bounded_store(dir: &std::path::Path, cap_bindings: u64) -> Arc<IdempotencyStore> {
    let spill = Arc::new(IdempotencySpill::open(dir).unwrap());
    let cfg = IdempotencyBoundConfig {
        high_watermark_bytes: cap_bindings * IDEMPOTENCY_BINDING_WEIGHT_BYTES,
        low_watermark_bytes: (cap_bindings / 2).max(1) * IDEMPOTENCY_BINDING_WEIGHT_BYTES,
    };
    Arc::new(IdempotencyStore::with_bound(spill, cfg))
}

/// Mark every currently-RESIDENT binding checkpoint-durable (INV-DURABLE
/// gate) so the drainer's `force_drain_for_test` below has evict-eligible
/// entries to spill.
///
/// #1521 M6.1 P1-6 de-flake: this MUST be
/// [`IdempotencyStore::mark_resident_durable_for_test`] (O(resident-count),
/// bounded by the cap), never [`IdempotencyStore::for_each_binding`]
/// (O(resident + ALL-LIVE-SPILLED), and it takes the spill's single
/// `index` mutex — the SAME mutex `evict_one_claimed`'s `write_binding`
/// needs for every eviction). The original gate called `for_each_binding`
/// here in a tight `while !stop` drainer loop against a 12,000-binding /
/// 32-binding-cap workload: as the run progresses nearly the whole set
/// lives in spill, so every drainer iteration re-walks the ENTIRE
/// (monotonically growing) spill chain while holding the mutex every
/// eviction write also needs — an O(n^2)-total, self-inflicted
/// mutex-monopolization that starves eviction (and, transitively, the
/// 8 writer threads waiting on `install`'s inline `maybe_drain` to keep
/// the resident set bounded) for load-dependent, non-deterministic
/// stretches — reproducing exactly the "writer join exceeded the bounded
/// wait budget" (116-450s) flake. `mark_resident_durable_for_test` marks
/// only the resident set (bounded by the cap) with no spill walk and no
/// contention on the spill index mutex, matching the established
/// at-scale continuous-marker pattern in
/// `m0x_round2_gates_1404.rs::ingest_at_scale`. This is a TEST-WORKLOAD
/// fix, not a production `IdempotencyStore` change — the mechanism
/// itself (`evict_one`/`evict_one_claimed`/`drain_to_low_watermark`) has
/// no starvation defect; the gate's own marker helper was the
/// bottleneck.
fn capture_mark_durable(store: &IdempotencyStore) {
    store.mark_resident_durable_for_test();
}

fn ext_id(id: u64) -> String {
    format!("ext-{id:010}")
}

/// THE decisive leg: 8 concurrent writers + a concurrent drainer (forces
/// continuous forward+reverse eviction) + a concurrent lookup hammer
/// (reverse `external_id_for` reads), all running together — not staged.
/// Forward/reverse must stay paired for every installed id.
#[test]
fn paired_forward_reverse_survive_concurrent_eviction_and_lookup() {
    let dir = tempdir().unwrap();
    let cap = 32u64; // small: forces continuous eviction against WRITERS*PER_WRITER total.
    let store = bounded_store(dir.path(), cap);
    let total = WRITERS * PER_WRITER;

    let stop = Arc::new(AtomicBool::new(false));
    let installed_high_water = Arc::new(AtomicU64::new(0));

    // Drainer: continuously marks resident bindings durable then forces
    // an eviction drain pass — the ONLY path that spills bindings (both
    // directions) and reclaims their resident frames.
    let drainer = {
        let store = Arc::clone(&store);
        let stop = Arc::clone(&stop);
        thread::spawn(move || {
            while !stop.load(Ordering::Acquire) {
                capture_mark_durable(&store);
                store.force_drain_for_test();
                thread::yield_now();
            }
            capture_mark_durable(&store);
            store.force_drain_for_test();
        })
    };

    // Lookup hammer: continuously round-trips external_id_for (reverse
    // lookup) against ids that are DEFINITELY already installed (below
    // the current high-water mark), concurrently with eviction. Every
    // resolved lookup must be internally consistent: external_id_for(id)
    // returns a string that ALSO forward-resolves back to the same id.
    let lookup_mismatches = Arc::new(AtomicU64::new(0));
    let lookups_performed = Arc::new(AtomicU64::new(0));
    let lookup_thread = {
        let store = Arc::clone(&store);
        let stop = Arc::clone(&stop);
        let high_water = Arc::clone(&installed_high_water);
        let mismatches = Arc::clone(&lookup_mismatches);
        let performed = Arc::clone(&lookups_performed);
        thread::spawn(move || {
            let mut probe: u64 = 0;
            while !stop.load(Ordering::Acquire) {
                let ceiling = high_water.load(Ordering::Acquire);
                if ceiling == 0 {
                    thread::yield_now();
                    continue;
                }
                let id = probe % ceiling;
                probe = probe.wrapping_add(1);
                if let Some(ext) = store.external_id_for(TenantId::DEFAULT, NODE, id) {
                    performed.fetch_add(1, Ordering::Relaxed);
                    // Paired consistency: the reverse-resolved external_id
                    // must ALSO forward-resolve back to the SAME internal_id
                    // — this is the decisive "forward/reverse stay
                    // consistent" oracle, checked WHILE eviction is racing
                    // concurrently, not after the fact.
                    match store.get(TenantId::DEFAULT, NODE, &ext) {
                        Some(binding) if binding.internal_id == id => {}
                        _ => {
                            mismatches.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            }
        })
    };

    // RULE-MT: ≥8 concurrent writer threads installing bindings.
    let mut writers = Vec::new();
    for t in 0..WRITERS {
        let store = Arc::clone(&store);
        let high_water = Arc::clone(&installed_high_water);
        writers.push(thread::spawn(move || {
            for i in 0..PER_WRITER {
                let id = t * PER_WRITER + i;
                store.install(TenantId::DEFAULT, NODE, &ext_id(id), id);
                // Advance the "definitely installed" watermark conservatively:
                // only ids strictly below the current minimum contiguous
                // prefix per writer are guaranteed installed by all peers,
                // but since writer ranges are disjoint and monotone per
                // writer, `id + 1` is a safe per-writer high-water; the
                // shared watermark tracks the max seen so far (an
                // UNDER-approximation is safe — the lookup thread just
                // probes a slightly smaller set, never a not-yet-installed id
                // that could legitimately miss).
                high_water.fetch_max(t * PER_WRITER + i + 1, Ordering::AcqRel);
            }
        }));
    }

    let deadline = Instant::now() + WAIT;
    for w in writers {
        assert!(
            Instant::now() < deadline,
            "writer join exceeded the bounded wait budget"
        );
        w.join().unwrap();
    }
    stop.store(true, Ordering::Release);
    drainer.join().unwrap();
    lookup_thread.join().unwrap();

    // Final settle.
    capture_mark_durable(&store);
    store.force_drain_for_test();

    assert_eq!(
        lookup_mismatches.load(Ordering::Relaxed),
        0,
        "MECH-E5/gate-2c violation: a reverse lookup (external_id_for) \
         resolved an external_id that did NOT forward-resolve back to the \
         same internal_id while eviction was concurrently draining — \
         forward/reverse desynced under paired eviction"
    );
    assert!(
        lookups_performed.load(Ordering::Relaxed) > 0,
        "lookup thread sanity: it must have performed at least one \
         successful reverse resolution during the run"
    );

    // Exhaustive final oracle: every installed pair resolves BOTH
    // directions, whether currently resident or spilled.
    //
    // ORDER MATTERS: `external_id_for` is checked BEFORE `get()` for each
    // id. `get()` (the forward lookup)'s spill re-fault side-effect
    // re-warms the resident REVERSE map (see `resolve_binding`'s
    // insert_resident_warm_if_vacant + reverse-populate) — calling `get()`
    // first would launder a broken REVERSE SPILL INDEX by re-populating
    // the resident reverse entry via the (unaffected) FORWARD path,
    // making this gate vacuous against exactly the "evict one side to
    // nowhere" mutant it exists to catch. Checking `external_id_for`
    // first forces it to resolve from a cold resident-reverse-miss
    // (spill only) at least once per id.
    assert_eq!(store.total_len(), total as usize, "logical set incomplete");
    for id in 0..total {
        let ext = ext_id(id);
        assert_eq!(
            store.external_id_for(TenantId::DEFAULT, NODE, id),
            Some(ext.clone()),
            "reverse lookup lost id {id} — the at-least-once identity RED \
             (a dropped reverse entry lets duplicate-ingest resolve wrong)"
        );
        assert_eq!(
            store
                .get(TenantId::DEFAULT, NODE, &ext)
                .map(|b| b.internal_id),
            Some(id),
            "forward lookup lost id {id}"
        );
    }
}
