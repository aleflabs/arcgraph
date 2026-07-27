//! #1404 M0.x ROUND-2 impl-ultracode REJECT remediation — the 3 gates.
//!
//! ROOT of the round-2 REJECT: the round-1 gates drove the WRONG ACTOR
//! (concurrent `install` — guarded) at the WRONG SCALE (small-N, asserting
//! `resident_len`). Production's default regime is sustained concurrent
//! **`get()` re-fault** (unguarded pre-fix) + **at-scale spill** (the index
//! itself O(N-rels)). Every gate here drives the RIGHT actor at the RIGHT
//! scale and is MULTI-THREADED + RED-on-revert (each was run against the
//! pristine pre-fix head and observed to fail before the fix landed).
//!
//! - **FIX-1 gate:** concurrent same-key `install` racing the inline
//!   drain's `evict_one` → a binding must NEVER read back `None` right
//!   after its own successful install (the blind `forward.remove(&key)`
//!   silent data loss, skeptic 1 — reproduced 1/1/3 None-losses pristine).
//! - **FIX-2 gate:** concurrent `get()` RE-FAULT threads (NOT installers)
//!   hammering spilled keys while the producer runs the write-guarded
//!   count+stream → header must equal streamed EVERY round (the unguarded
//!   read-side warm-insert third-writer, skeptics 4+5 — reproduced
//!   header=4000 vs streamed=1396 pristine → 62% checkpoint aborts → the
//!   WAL bound defeated).
//! - **FIX-3 gate:** at-scale REL ingest (10× size differential) → the
//!   spill's OWN in-RAM index must stay BOUNDED, not grow 1:1 with N (the
//!   relocated O(N-rels) OOM, skeptic 6 — reproduced 19,950 → 199,950
//!   in-RAM index entries for 20K → 200K bindings pristine).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;

use arcgraph_core::TenantId;
use arcgraph_storage::idempotency::{
    IDEMPOTENCY_BINDING_WEIGHT_BYTES, IdempotencyBoundConfig, IdempotencySpill, IdempotencyStore,
};
use tempfile::tempdir;

const NODE: u8 = 0;

fn bounded_store(dir: &std::path::Path, cap_bindings: u64) -> Arc<IdempotencyStore> {
    let spill = Arc::new(IdempotencySpill::open(dir).unwrap());
    let cfg = IdempotencyBoundConfig {
        high_watermark_bytes: cap_bindings * IDEMPOTENCY_BINDING_WEIGHT_BYTES,
        low_watermark_bytes: (cap_bindings / 2).max(1) * IDEMPOTENCY_BINDING_WEIGHT_BYTES,
    };
    Arc::new(IdempotencyStore::with_bound(spill, cfg))
}

/// Mark all resident bindings checkpoint-durable via the PRODUCTION streaming
/// capture (the INV-DURABLE gate the producer sets under the freeze), keeping
/// freshly-installed bindings continuously evict-eligible.
fn capture_mark_durable(store: &IdempotencyStore) {
    store
        .for_each_binding::<_, std::convert::Infallible>(|_, _, _, _, _| Ok(()))
        .expect("infallible");
}

// ─────────────────────────────────────────────────────────────────────
// FIX-1 — evict-race silent data loss (skeptic 1)
// ─────────────────────────────────────────────────────────────────────

/// FIX-1: concurrent same-key installs over a SHARED hot-key set while the
/// inline drain (`maybe_drain` → `evict_one`) fires on every install and a
/// background capture keeps bindings evict-eligible. THE oracle: a `get()`
/// immediately after a thread's own successful `install` must NEVER be
/// `None` — a `None` is the blind `forward.remove(&key)` dropping a FRESH
/// binding both tiers never held (silent identity loss → duplicate on
/// re-ingest).
///
/// RED-on-revert (reproduced on pristine `evict_one`'s blind remove): the
/// exact recipe below observed persisted None-losses across runs; with the
/// compare-and-remove + spill-rollback + guarded reverse-remove the losses
/// are 0.
#[test]
fn fix1_concurrent_install_vs_evict_never_loses_a_fresh_binding() {
    let dir = tempdir().unwrap();
    // cap=4 with 8 hot keys → the resident set sits permanently over the
    // high watermark → EVERY install runs the inline drain (the racing actor).
    let store = bounded_store(dir.path(), 4);
    let hot_keys = 8u64;
    let iters_per_thread = 4000u64;
    let n_threads = 4u64;

    let stop_marker = Arc::new(AtomicBool::new(false));
    let next_id = Arc::new(AtomicU64::new(1));
    let none_losses = Arc::new(AtomicU64::new(0));

    // Background marker: continuously marks resident bindings
    // checkpoint-durable so the inline drain always has evict-eligible
    // bindings (without it, INV-DURABLE keeps everything resident and the
    // race never fires). Do not use the production streaming capture here:
    // it also enumerates the entire spill while holding the spill-index lock.
    // In release that tight loop can repeatedly reacquire the lock and starve
    // every installer indefinitely. The resident-only mark is the operation
    // this harness needs, and yielding gives the installers a progress slot.
    let marker = {
        let store = Arc::clone(&store);
        let stop = Arc::clone(&stop_marker);
        thread::spawn(move || {
            while !stop.load(Ordering::Acquire) {
                store.mark_resident_durable_for_test();
                thread::yield_now();
            }
        })
    };

    // 4 installer threads over the SAME 8 hot keys: each install races other
    // installs AND the inline evictions of the same keys.
    let mut installers = Vec::new();
    for t in 0..n_threads {
        let store = Arc::clone(&store);
        let next_id = Arc::clone(&next_id);
        let losses = Arc::clone(&none_losses);
        installers.push(thread::spawn(move || {
            for i in 0..iters_per_thread {
                let k = (t + i) % hot_keys;
                let ext = format!("hot-{k}");
                let id = next_id.fetch_add(1, Ordering::AcqRel);
                store.install(TenantId::DEFAULT, NODE, &ext, id);
                // THE oracle: the key we JUST installed must resolve. A None
                // here is the evict-race dropping the fresh binding from both
                // tiers (get() consults resident AND spill — absence in both
                // is a lost identity, never a transient).
                if store.get(TenantId::DEFAULT, NODE, &ext).is_none() {
                    losses.fetch_add(1, Ordering::AcqRel);
                }
            }
        }));
    }
    for h in installers {
        h.join().unwrap();
    }
    stop_marker.store(true, Ordering::Release);
    marker.join().unwrap();

    let losses = none_losses.load(Ordering::Acquire);
    assert_eq!(
        losses, 0,
        "FIX-1 FAIL: {losses} get()==None immediately after a successful own \
         install — the evict-race dropped a fresh binding from BOTH tiers \
         (silent identity loss → duplicate on re-ingest)",
    );

    // Stale-residual oracle (verdict §17: `remove_if` ALONE leaves a stale-id
    // residual): after the storm quiesces, serially re-install every hot key
    // with a known FINAL id, then capture+drain (evicting it to spill), then
    // resolve — the answer must be exactly the FINAL id, never a stale image
    // resurrected from a rolled-back-less spill write.
    let final_base = next_id.load(Ordering::Acquire) + 1000;
    for k in 0..hot_keys {
        store.install(TenantId::DEFAULT, NODE, &format!("hot-{k}"), final_base + k);
    }
    capture_mark_durable(&store);
    store.force_drain_for_test();
    for k in 0..hot_keys {
        assert_eq!(
            store
                .get(TenantId::DEFAULT, NODE, &format!("hot-{k}"))
                .map(|b| b.internal_id),
            Some(final_base + k),
            "FIX-1 FAIL: hot-{k} resolved a STALE id after quiesce+evict — a \
             superseded spill image was never rolled back",
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// FIX-2 — read-side re-fault warm-insert is the THIRD writer (skeptics 4+5)
// ─────────────────────────────────────────────────────────────────────

/// FIX-2: concurrent `get()` RE-FAULT threads (NOT installers — the round-1
/// fix_d gate drove installs, which take the capture READ guard; the re-fault
/// warm-insert took NO guard) hammer spilled-only keys while the producer runs
/// the EXACT ADR-229 capture sequence — `capture_guard()` → `binding_count()`
/// (the count header) → `for_each_binding()` (the stream) — across many
/// rounds. THE oracle: header == streamed EVERY round. A skew is the unguarded
/// warm-insert landing BETWEEN the resident walk and the spill walk →
/// `CheckpointError::CountSkew` → the interval checkpointer aborts (~62% of
/// captures under read load pristine) → the WAL frontier never advances → the
/// unbounded-WAL regression #1404/#1365 returns.
///
/// RED-on-revert (reproduced on pristine — no guard on the re-warm): skews on
/// the very first rounds (header=4000 vs streamed=1396 class); GREEN with the
/// read-guard on both re-warm sites.
#[test]
fn fix2_concurrent_get_refault_never_skews_the_guarded_capture() {
    let dir = tempdir().unwrap();
    let n: u64 = 4000;
    // cap=8 → after the seeding drain, essentially ALL keys are spilled-only,
    // so every reader `get()` is a re-fault (the unguarded third writer).
    let store = bounded_store(dir.path(), 8);
    for i in 0..n {
        store.install(TenantId::DEFAULT, NODE, &format!("ext-{i}"), 1_000_000 + i);
    }
    // Mark durable + drain → spilled-only working set.
    capture_mark_durable(&store);
    store.force_drain_for_test();

    let stop = Arc::new(AtomicBool::new(false));

    // 6 reader threads hammering get() across the spilled key space —
    // production's MCP foreground dedupe read (adapters.rs), which re-faults
    // and warm-inserts. NOT installers: the install path was already guarded.
    let mut readers = Vec::new();
    for r in 0..6u64 {
        let store = Arc::clone(&store);
        let stop = Arc::clone(&stop);
        readers.push(thread::spawn(move || {
            let mut j = r * 700;
            while !stop.load(Ordering::Acquire) {
                let ext = format!("ext-{}", j % n);
                assert!(
                    store.get(TenantId::DEFAULT, NODE, &ext).is_some(),
                    "re-fault lost a live binding",
                );
                j += 1;
            }
        }));
    }
    // A re-spiller keeps the flux going (re-warmed keys drain back to spill,
    // so re-faults keep firing for the whole run instead of going resident).
    let respiller = {
        let store = Arc::clone(&store);
        let stop = Arc::clone(&stop);
        thread::spawn(move || {
            while !stop.load(Ordering::Acquire) {
                store.force_drain_for_test();
            }
        })
    };

    // The producer: the exact write-guarded count+stream, many rounds.
    let rounds = 400u64;
    let mut skews = Vec::new();
    for round in 0..rounds {
        let _guard = store.capture_guard();
        let header = store.binding_count();
        let streamed = store
            .for_each_binding::<_, std::convert::Infallible>(|_, _, _, _, _| Ok(()))
            .expect("infallible");
        if header != streamed {
            skews.push((round, header, streamed));
        }
    }
    stop.store(true, Ordering::Release);
    for h in readers {
        h.join().unwrap();
    }
    respiller.join().unwrap();

    assert!(
        skews.is_empty(),
        "FIX-2 FAIL: {}/{rounds} write-guarded captures skewed header≠streamed \
         under concurrent get() re-fault (first: round {} header={} streamed={}) \
         — every skew aborts a checkpoint → the WAL frontier stalls → the \
         unbounded-WAL regression returns",
        skews.len(),
        skews[0].0,
        skews[0].1,
        skews[0].2,
    );
}

// ─────────────────────────────────────────────────────────────────────
// FIX-3 — the spill's OWN in-RAM index is O(N-rels) resident (skeptic 6)
// ─────────────────────────────────────────────────────────────────────

/// Multi-threaded at-scale ingest of `n` DISTINCT bindings through a tiny
/// resident cap, so essentially every binding churns through the spill (the
/// production 20-40M-rels regime, scaled down). Uses a deliberately SMALL
/// bucket count so the on-disk chains are genuinely long (~24 nodes at
/// n=200K) — the fixed head arrays must stay flat while the chains, not the
/// RAM, absorb the growth. Returns `(spill_index_resident_entries,
/// total_len)` after the final drain.
fn ingest_at_scale(n: u64) -> (u64, usize) {
    let dir = tempdir().unwrap();
    // Small bucket count → long chains at scale (the pre-fix DashMaps never
    // had chains; the bound is only real if lookups stay correct through
    // them). 2 × 8192 head slots ≪ 200K bindings.
    let spill = Arc::new(IdempotencySpill::open_with_buckets(dir.path(), 1 << 13).unwrap());
    let cfg = IdempotencyBoundConfig {
        high_watermark_bytes: 64 * IDEMPOTENCY_BINDING_WEIGHT_BYTES,
        low_watermark_bytes: 32 * IDEMPOTENCY_BINDING_WEIGHT_BYTES,
    };
    let store = Arc::new(IdempotencyStore::with_bound(spill, cfg));
    let n_threads = 4u64;

    // Marker thread: continuously flips resident bindings checkpoint-durable
    // (INV-DURABLE) so every installer's inline drain can evict — the
    // spill-write path runs CONCURRENT with the installs, as in production.
    let stop_marker = Arc::new(AtomicBool::new(false));
    let marker = {
        let store = Arc::clone(&store);
        let stop = Arc::clone(&stop_marker);
        thread::spawn(move || {
            while !stop.load(Ordering::Acquire) {
                store.mark_resident_durable_for_test();
                thread::yield_now();
            }
        })
    };

    let mut installers = Vec::new();
    for t in 0..n_threads {
        let store = Arc::clone(&store);
        let per_thread = n / n_threads;
        installers.push(thread::spawn(move || {
            for i in 0..per_thread {
                let ext = format!("s{n}-t{t}-i{i}");
                store.install(TenantId::DEFAULT, NODE, &ext, 1 + t * n + i);
            }
        }));
    }
    for h in installers {
        h.join().unwrap();
    }
    stop_marker.store(true, Ordering::Release);
    marker.join().unwrap();

    // Push the resident remainder to spill so the index carries ~all N.
    store.mark_resident_durable_for_test();
    store.force_drain_for_test();

    // Correctness THROUGH the chains (the bound is worthless if long chains
    // lose bindings): a sampled resolve must return the exact installed id...
    for t in 0..n_threads {
        for i in (0..n / n_threads).step_by(997) {
            let ext = format!("s{n}-t{t}-i{i}");
            assert_eq!(
                store
                    .get(TenantId::DEFAULT, NODE, &ext)
                    .map(|b| b.internal_id),
                Some(1 + t * n + i),
                "FIX-3 FAIL: {ext} lost or mis-resolved through the spill index chains",
            );
        }
    }
    // ...a guarded capture must still count EXACTLY (header == streamed with
    // shadowed copies + tombstones deduped by the chain walk)...
    {
        let _guard = store.capture_guard();
        let header = store.binding_count();
        let streamed = store
            .for_each_binding::<_, std::convert::Infallible>(|_, _, _, _, _| Ok(()))
            .expect("infallible");
        assert_eq!(
            header, streamed,
            "FIX-3 FAIL: chain-walk enumeration disagrees with binding_count at n={n}",
        );
    }
    // ...and a release must tombstone through a long chain (get → None).
    store.release(TenantId::DEFAULT, NODE, &format!("s{n}-t0-i0"));
    assert!(
        store
            .get(TenantId::DEFAULT, NODE, &format!("s{n}-t0-i0"))
            .is_none(),
        "FIX-3 FAIL: released binding still resolves (tombstone lost in chain)",
    );

    (store.spill_index_resident_entries(), store.total_len())
}

/// FIX-3: at-scale REL ingest with a 10× size differential — the spill's own
/// in-RAM index footprint must stay BOUNDED (O(buckets), flat), not grow 1:1
/// with N. Pre-fix the `offsets` + `reverse_offsets` DashMaps grew exactly
/// with the binding count (reproduced 19,950 → 199,950 in-RAM entries for
/// 20K → 200K bindings — ~4-8+ GB undrainable at the production 20-40M rels,
/// the relocated OOM the round-2 verdict REJECTed on).
///
/// RED-on-revert (reproduced on the pre-fix DashMap index via the
/// `offsets.len() + reverse_offsets.len()` shim): entries grew ~10× across
/// the differential and dwarfed the bound; with the on-disk hash-chain index
/// the resident footprint is IDENTICAL at both scales.
#[test]
fn fix3_spill_index_stays_bounded_at_scale() {
    let (idx_small, len_small) = ingest_at_scale(20_000);
    let (idx_big, len_big) = ingest_at_scale(200_000);

    // The ingest really is 10× (minus the one released probe key).
    assert!(
        len_big >= len_small * 9,
        "gate harness broken: {len_small} → {len_big} bindings is not a 10× differential",
    );
    // THE oracle: resident index entries must NOT scale with N. Post-fix both
    // sides are the constant head arrays (2 × 8192); pre-fix idx_big/idx_small
    // ≈ 10 (1:1 with N).
    assert!(
        idx_big <= idx_small.saturating_mul(2),
        "FIX-3 FAIL: spill in-RAM index grew {idx_small} → {idx_big} entries \
         across a 10× ingest — the index is O(N-bindings) resident again \
         (the relocated 20-40M-rel OOM, round-2 skeptic 6)",
    );
    // And it must be a genuine bound, small against N — not merely flat by
    // being pre-sized to N-scale.
    assert!(
        idx_big < len_big as u64 / 4,
        "FIX-3 FAIL: {idx_big} resident index entries for {len_big} bindings \
         is not a sub-linear bound",
    );
}
