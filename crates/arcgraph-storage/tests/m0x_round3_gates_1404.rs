//! #1404 M0.x ROUND-3 re-ultracode REJECT remediation — the gates.
//!
//! ROOT of the round-3 REJECT: the round-2 gates drove id-CHANGING
//! overwrites (FIX-1 gate: `next_id.fetch_add` per install) and a
//! pure-ingest regime (FIX-3 gate). Production's regimes also include
//! **same-`internal_id` re-publish** (WAL replay / idempotent re-publish —
//! `install`'s own doc names both) and **ingest-then-delete** (TTL expiry,
//! re-sync, ephemeral incident entities). Every gate here is
//! MULTI-THREADED + RED-on-revert (each was run against the pre-fix code
//! and observed to fail before the fix landed).
//!
//! - **FIX-5a gate:** continuous SAME-id overwrite of hot keys under CPU
//!   OVERSUBSCRIPTION (the races need an evictor preempted between its
//!   spill-write and its compare-and-remove — thread count ≫ cores is the
//!   amplifier) + concurrent inline drain + independent readers → a
//!   continuously-live binding must NEVER read back `None` (round-3
//!   skeptic 1; reproduced at pristine `9682b7fb` with THIS recipe: 156
//!   misses in ~18s).
//! - **FIX-5b gate:** single-writer-per-key same-id re-publish with a
//!   monotone `payload_hash` → a read must never serve a hash OLDER than
//!   the writer's last completed install (the gen-stamp leg's independent
//!   oracle: an `internal_id`-only compare-and-remove silently swaps a
//!   fresh re-publish for the evictor's stale snapshot — same id, STALE
//!   hash — which the `None`/wrong-id oracles cannot see).
//! - **FIX-4 gate:** ingest-then-`release` N ≫ cap distinct keys in
//!   small-live-set churn batches (the live set never crosses the
//!   watermark, so the round-2 resident-byte trigger stays SILENT — the
//!   leak's regime) → the evict queue must be BOUNDED, not retain one
//!   stale entry per released binding for a logically-EMPTY store
//!   (round-3 skeptic 4: 50,000 stale entries for a 0-binding store
//!   pristine — ~2-4 GB at 40M-rel ingest+delete, the OOM class #1404
//!   exists to kill).
//! - **FIX-4b gate:** SAME-key install→release churn with NO checkpoint —
//!   the regime the FIX-4 gate structurally could not enter (its DISTINCT
//!   keys leave `Gone` fronts the drain reclaims; a same-key regime parks
//!   the currently-resident, permanently-`NotDurable` key at the FRONT, so
//!   the break-on-first-`NotDurable` drain reclaimed NOTHING and the
//!   duplicate backlog grew O(cycles)) → the evict queue must stay bounded
//!   by its entry cap, independent of cycle count.

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
// FIX-5 — same-id overwrite × evict double-miss (round-3 skeptic 1)
// ─────────────────────────────────────────────────────────────────────

/// FIX-5a: SAME-`internal_id` continuous re-publish (the WAL-replay /
/// idempotent-re-publish shape — every install of `hot-{k}` carries the
/// FIXED id `1000+k`) over a shared hot-key set under permanent drain
/// pressure, with INDEPENDENT reader threads. THE oracle: `get()` of a
/// continuously-live binding must NEVER be `None` — not from the
/// overwriter's own post-install read, and not from an independent reader.
///
/// This is the regime the round-2 FIX-1 gate structurally could not enter:
/// its `next_id.fetch_add` per install made every overwrite id-CHANGING, so
/// `evict_one`'s `internal_id`-only compare-and-remove never matched a fresh
/// binding. With the id held CONSTANT the pre-fix compare cannot tell the
/// fresh re-publish from the evictor's stale snapshot: the evictor drops the
/// fresh resident binding while the overwriter's install-entry retire
/// tombstones the evictor's freshly-written spill image → both tiers absent
/// with a stable `move_epoch` → a STABLE `get()==None` for a binding that
/// was never released (round-3 skeptic 1: 35 misses / 23 still-None after 50
/// retries pristine; a `move_epoch`-bump-only probe did NOT close it — 2877
/// misses).
///
/// The interleavings need an actor PREEMPTED inside a sub-microsecond
/// window (e.g. the evictor between its spill-index insert and its
/// compare-and-remove), so the gate OVERSUBSCRIBES the CPU (48 overwriters
/// plus 16 readers): at matched thread-to-core counts the window essentially
/// never opens (observed 0 misses pristine at 7 threads; 156 misses
/// pristine at 65 threads, same total op count).
///
/// FIX-5 is a four-leg unit (all in `idempotency.rs`), and this gate is RED
/// (156 misses / ~18s at pristine `9682b7fb`) when the unit is reverted:
/// (a) `evict_one` compare-and-removes on `(internal_id, install_gen)` —
///     see FIX-5b for the leg's own oracle;
/// (b) the evict-rollback (`Gone` branch) bumps `move_epoch` before its
///     tombstone — the tombstone flips the key's spill verdict live→dead,
///     so it is a hiding mover the seqlock reader must retry over (this leg
///     alone accounted for 5-10 transient reader misses per run during
///     development, epoch pinned at 0);
/// (c) at most ONE in-flight evictor per key (`evict_inflight`) — a sibling
///     evictor's rollback tombstone would shadow the winner's LIVE image
///     (a STABLE double-miss, the 23-still-None-after-50-retries class);
/// (d) the overwriter's install-entry retire is SKIPPED for a same-id
///     re-publish, so it can never tombstone the image a concurrent evictor
///     just wrote as the binding's sole durable copy.
#[test]
fn fix5a_same_id_overwrite_vs_evict_never_double_misses() {
    let dir = tempdir().unwrap();
    // cap=1 with 4 hot keys → the resident set sits permanently over the
    // high watermark → EVERY install runs the inline drain (the racing
    // evictor).
    let store = bounded_store(dir.path(), 1);
    let hot_keys = 4u64;
    let iters_per_thread = 600u64;
    let n_overwriters = 48u64;
    let n_readers = 16u64;
    let id_base = 1000u64;

    // Establish every hot key BEFORE any thread starts: from here on each
    // binding is CONTINUOUSLY live (only ever re-published with the SAME id,
    // never released), so any observed absence is a correctness violation.
    for k in 0..hot_keys {
        store.install(TenantId::DEFAULT, NODE, &format!("hot-{k}"), id_base + k);
    }

    let stop = Arc::new(AtomicBool::new(false));
    let misses = Arc::new(AtomicU64::new(0));
    let wrong_id = Arc::new(AtomicU64::new(0));

    // Background capture thread: continuously marks resident bindings
    // checkpoint-durable so the inline drain always has evict-eligible
    // bindings (without it, INV-DURABLE keeps everything resident and the
    // race never fires).
    let marker = {
        let store = Arc::clone(&store);
        let stop = Arc::clone(&stop);
        thread::spawn(move || {
            while !stop.load(Ordering::Acquire) {
                capture_mark_durable(&store);
            }
        })
    };

    // Independent reader threads: hammer the continuously-live keys the
    // whole time. A None is the double-miss (the reader-side oracle skeptic
    // 1's reproducer used); a wrong id is a stale-image resurrection.
    let mut readers = Vec::new();
    for _ in 0..n_readers {
        let store = Arc::clone(&store);
        let stop = Arc::clone(&stop);
        let misses = Arc::clone(&misses);
        let wrong_id = Arc::clone(&wrong_id);
        readers.push(thread::spawn(move || {
            let mut k = 0u64;
            while !stop.load(Ordering::Acquire) {
                let ext = format!("hot-{k}");
                match store.get(TenantId::DEFAULT, NODE, &ext) {
                    None => {
                        misses.fetch_add(1, Ordering::AcqRel);
                    }
                    Some(b) if b.internal_id != id_base + k => {
                        wrong_id.fetch_add(1, Ordering::AcqRel);
                    }
                    Some(_) => {}
                }
                k = (k + 1) % hot_keys;
            }
        }));
    }

    // Overwriter threads over the SAME hot keys, each re-publishing the
    // key's FIXED id: every install races other same-id installs AND the
    // inline evictions of the same keys.
    let mut overwriters = Vec::new();
    for t in 0..n_overwriters {
        let store = Arc::clone(&store);
        let misses = Arc::clone(&misses);
        overwriters.push(thread::spawn(move || {
            for i in 0..iters_per_thread {
                let k = (t + i) % hot_keys;
                let ext = format!("hot-{k}");
                store.install(TenantId::DEFAULT, NODE, &ext, id_base + k);
                // Own-read oracle: the binding we JUST re-published must
                // resolve (absence in BOTH tiers is a lost identity →
                // duplicate on re-ingest, never a transient).
                if store.get(TenantId::DEFAULT, NODE, &ext).is_none() {
                    misses.fetch_add(1, Ordering::AcqRel);
                }
            }
        }));
    }
    for h in overwriters {
        h.join().unwrap();
    }
    stop.store(true, Ordering::Release);
    for h in readers {
        h.join().unwrap();
    }
    marker.join().unwrap();

    let misses = misses.load(Ordering::Acquire);
    let wrong_id = wrong_id.load(Ordering::Acquire);
    assert_eq!(
        misses, 0,
        "FIX-5 FAIL: {misses} get()==None of a continuously-live same-id \
         binding — the evict-race dropped a fresh re-publish from BOTH tiers \
         (silent identity loss → duplicate on re-ingest)",
    );
    assert_eq!(
        wrong_id, 0,
        "FIX-5 FAIL: {wrong_id} reads resolved a WRONG id for a same-id \
         binding — a stale image was resurrected",
    );

    // Quiesce oracle: after the storm every hot key must still resolve to
    // its fixed id, resident or spilled.
    capture_mark_durable(&store);
    store.force_drain_for_test();
    for k in 0..hot_keys {
        assert_eq!(
            store
                .get(TenantId::DEFAULT, NODE, &format!("hot-{k}"))
                .map(|b| b.internal_id),
            Some(id_base + k),
            "FIX-5 FAIL: hot-{k} lost after quiesce+evict",
        );
    }
}

/// FIX-5b — the gen-stamp leg's INDEPENDENT oracle: hash freshness.
///
/// With legs (b)/(c)/(d) in place, reverting ONLY the `(internal_id,
/// install_gen)` compare back to `internal_id`-only no longer produces a
/// `None` (the same-id spill image the evictor wrote stays live and serves
/// the same id) — the observable regression is a SILENT STALE SERVE: the
/// evictor removes the FRESH re-publish and the store then serves the
/// evictor's OLDER snapshot — same id, stale `payload_hash`. A stale hash
/// is an at-least-once violation upstream (the payload-dedup verdict is
/// computed from it).
///
/// Oracle: ONE writer per key re-publishes the key's fixed id with a
/// strictly-increasing `payload_hash` and publishes its last COMPLETED
/// iteration; on a correct store no read may ever observe a hash below the
/// writer's completed watermark (single writer ⟹ resident is always the
/// newest value; an evicted image equals the exact value removed — that is
/// precisely what the gen-stamp guarantees). RED-on-revert: reverting the
/// gen-stamp leg alone makes this fire (stale serves), while FIX-5a stays
/// green.
#[test]
fn fix5b_same_id_overwrite_vs_evict_never_serves_stale_hash() {
    let dir = tempdir().unwrap();
    // cap=1 with 48 keys / one writer each → permanent drain pressure +
    // CPU oversubscription (48 writers + 16 readers).
    let store = bounded_store(dir.path(), 1);
    let n_keys = 48u64;
    let iters = 600u64;
    let n_readers = 16u64;
    let id_base = 5000u64;

    // Establish every key with hash 0; from here on each binding is
    // continuously live with a single monotone writer.
    for k in 0..n_keys {
        store.install_with_payload_hash(
            TenantId::DEFAULT,
            NODE,
            &format!("mono-{k}"),
            id_base + k,
            Some(0),
        );
    }
    let completed: Arc<Vec<AtomicU64>> = Arc::new((0..n_keys).map(|_| AtomicU64::new(0)).collect());

    let stop = Arc::new(AtomicBool::new(false));
    let misses = Arc::new(AtomicU64::new(0));
    let stale = Arc::new(AtomicU64::new(0));

    let marker = {
        let store = Arc::clone(&store);
        let stop = Arc::clone(&stop);
        thread::spawn(move || {
            while !stop.load(Ordering::Acquire) {
                capture_mark_durable(&store);
            }
        })
    };

    // Readers: snapshot the writer's completed watermark BEFORE the read —
    // the served hash may never be below it (it may be above: a newer
    // install can be mid-flight).
    let mut readers = Vec::new();
    for _ in 0..n_readers {
        let store = Arc::clone(&store);
        let stop = Arc::clone(&stop);
        let misses = Arc::clone(&misses);
        let stale = Arc::clone(&stale);
        let completed = Arc::clone(&completed);
        readers.push(thread::spawn(move || {
            let mut k = 0u64;
            while !stop.load(Ordering::Acquire) {
                let ext = format!("mono-{k}");
                let watermark = completed[k as usize].load(Ordering::Acquire);
                match store.get(TenantId::DEFAULT, NODE, &ext) {
                    None => {
                        misses.fetch_add(1, Ordering::AcqRel);
                    }
                    Some(b) => {
                        if b.payload_hash.unwrap_or(0) < watermark {
                            stale.fetch_add(1, Ordering::AcqRel);
                        }
                    }
                }
                k = (k + 1) % n_keys;
            }
        }));
    }

    // One writer per key: re-publish the SAME id with hash = iteration,
    // then publish the completed watermark. The own-read must see exactly
    // the value just installed (single writer ⟹ nothing newer can exist).
    let mut writers = Vec::new();
    for k in 0..n_keys {
        let store = Arc::clone(&store);
        let misses = Arc::clone(&misses);
        let stale = Arc::clone(&stale);
        let completed = Arc::clone(&completed);
        writers.push(thread::spawn(move || {
            let ext = format!("mono-{k}");
            for i in 1..=iters {
                store.install_with_payload_hash(
                    TenantId::DEFAULT,
                    NODE,
                    &ext,
                    id_base + k,
                    Some(i),
                );
                completed[k as usize].store(i, Ordering::Release);
                match store.get(TenantId::DEFAULT, NODE, &ext) {
                    None => {
                        misses.fetch_add(1, Ordering::AcqRel);
                    }
                    Some(b) => {
                        if b.payload_hash != Some(i) {
                            stale.fetch_add(1, Ordering::AcqRel);
                        }
                    }
                }
            }
        }));
    }
    for h in writers {
        h.join().unwrap();
    }
    stop.store(true, Ordering::Release);
    for h in readers {
        h.join().unwrap();
    }
    marker.join().unwrap();

    let misses = misses.load(Ordering::Acquire);
    let stale = stale.load(Ordering::Acquire);
    assert_eq!(
        misses, 0,
        "FIX-5b FAIL: {misses} get()==None of a continuously-live binding",
    );
    assert_eq!(
        stale, 0,
        "FIX-5b FAIL: {stale} reads served a STALE payload_hash for a \
         same-id re-publish — the evict-race swapped a fresh binding for \
         the evictor's older snapshot (silent at-least-once violation: the \
         payload-dedup verdict upstream is computed from this hash)",
    );

    // Quiesce oracle: every key must resolve to its final hash.
    capture_mark_durable(&store);
    store.force_drain_for_test();
    for k in 0..n_keys {
        let b = store
            .get(TenantId::DEFAULT, NODE, &format!("mono-{k}"))
            .expect("continuously-live binding lost after quiesce");
        assert_eq!(b.internal_id, id_base + k);
        assert_eq!(
            b.payload_hash,
            Some(iters),
            "FIX-5b FAIL: mono-{k} settled on a stale hash",
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// FIX-4 — evict-queue leak on ingest-then-delete (round-3 skeptic 4)
// ─────────────────────────────────────────────────────────────────────

/// FIX-4: ingest-then-delete churn (TTL expiry / re-sync / ephemeral
/// incident entities) with a SMALL live set. `release` removes a binding
/// from both tiers but leaves its evict-FIFO entry behind (reclaimed only
/// as a cheap `Gone` pop inside a drain pass) — and pre-fix the ONLY drain
/// trigger was `resident_bytes > high_watermark`, which a churn workload
/// whose live set stays at/under the cap NEVER produces. The FIFO therefore
/// retained one `(TenantId, u8, String)` entry per released binding
/// FOREVER: 50,000 stale entries for a logically-EMPTY 0-binding store
/// (round-3 skeptic 4 pristine repro; ~2-4 GB at 40M-rel churn — the OOM
/// class #1404 exists to kill).
///
/// 8 churn threads × 6,250 distinct keys each in batches of 8 (install the
/// batch, release it — the global live set stays ≤ 64 = cap, so the
/// round-2 resident-byte trigger stays SILENT) = 50,000 total released
/// bindings, the skeptic-4 magnitude. RED-on-revert: reverting FIX-4's two
/// legs (the FIFO-cap trigger in `maybe_drain` + the FIFO-cap disjunct in
/// the `drain_to_low_watermark` loop) leaves the queue at 50,000; with the
/// fix, any install that finds the FIFO past its cap reclaims the Gone
/// backlog, so the queue stays O(cap) — independent of total ever-released.
#[test]
fn fix4_release_churn_keeps_evict_queue_bounded() {
    let dir = tempdir().unwrap();
    let cap_bindings = 64u64;
    let store = bounded_store(dir.path(), cap_bindings);
    let n_threads = 8u64;
    let per_thread = 6_250u64;
    let batch = 8u64;

    let mut churners = Vec::new();
    for t in 0..n_threads {
        let store = Arc::clone(&store);
        churners.push(thread::spawn(move || {
            let mut k = 0u64;
            while k < per_thread {
                let end = (k + batch).min(per_thread);
                for i in k..end {
                    store.install(
                        TenantId::DEFAULT,
                        NODE,
                        &format!("ephemeral-{t}-{i}"),
                        1_000_000 + t * per_thread + i,
                    );
                }
                for i in k..end {
                    store.release(TenantId::DEFAULT, NODE, &format!("ephemeral-{t}-{i}"));
                }
                k = end;
            }
        }));
    }
    for h in churners {
        h.join().unwrap();
    }

    // The store is logically EMPTY: every installed binding was released.
    assert_eq!(
        store.resident_len(),
        0,
        "released bindings must leave the resident tier",
    );
    // Queue-bearing memory must be bounded by the CAP, not by total
    // ever-released. Loose bound = 8 × cap (the fix reclaims down to
    // 2 × cap entries; concurrent churn tails add slack) — pre-fix this
    // reads 50,000.
    let qlen = store.evict_queue_len();
    assert!(
        qlen <= (8 * cap_bindings) as usize,
        "FIX-4 FAIL: {qlen} evict-queue entries retained for a \
         logically-empty store (one stale entry per released binding; \
         pre-fix = 50,000 — ~2-4 GB at 40M-rel ingest-then-delete churn)",
    );
}

// ─────────────────────────────────────────────────────────────────────
// FIX-4b — same-key churn defeats the break-on-first-NotDurable drain
// ─────────────────────────────────────────────────────────────────────

/// FIX-4b: repeated SAME-key install→release churn with NO checkpoint —
/// the drain regime the FIX-4 gate above structurally cannot enter.
///
/// FIX-4's gate uses DISTINCT keys: at drain time the FIFO front is a
/// released key's `Gone` entry (the live batch is the newest pushes, at the
/// BACK), so the drain reclaims front-to-back and the break never blocks.
/// Same-key churn inverts that: every cycle's fresh install re-pushes the
/// SAME key, the drain runs INLINE on that install (the key is resident,
/// nothing is ever checkpointed → `NotDurable`), and the pre-4b
/// break-on-first-`NotDurable` fired on the very first pop — reclaiming
/// NOTHING, every install, while the duplicate backlog behind the front
/// grew one entry per cycle (queue length = O(cycles), unbounded, for a
/// live set of ONE binding).
///
/// The fix (FIX-4b, `drain_to_low_watermark`): while the FIFO is over its
/// entry cap, a `NotDurable` pop is retained aside (restored to the front
/// after the pass) and the scan CONTINUES — reclaiming `Gone` entries and
/// dropping same-key duplicates of already-retained keys (the retained
/// sibling keeps the key represented, the `evict_inflight`-skip argument).
/// A `Gone`/duplicate entry must be reclaimable REGARDLESS of a
/// not-yet-durable entry ahead of it.
///
/// One churner (the deterministic-RED shape: the drain only ever runs
/// inline on the churner's own install, so the key is ALWAYS resident +
/// `NotDurable` when probed — concurrent same-key churners would race
/// their releases into the probe window and reclaim by luck) + 4
/// concurrent reader threads (`get` of the churned key races the
/// retain/restore window against the seqlock read path). NO
/// `capture_mark_durable` anywhere: nothing is ever durable.
///
/// RED-on-revert: reverting the FIX-4b drain change (restoring the
/// unconditional `push_front` + `break` arm) leaves the queue at ~50,000
/// entries (one per cycle); with the fix it stays at ~(entry cap + 2) —
/// the cap-axis equilibrium — independent of cycle count.
#[test]
fn fix4b_same_key_churn_no_checkpoint_keeps_evict_queue_bounded() {
    let dir = tempdir().unwrap();
    let cap_bindings = 64u64; // queue entry cap = 2 × 64 = 128
    let store = bounded_store(dir.path(), cap_bindings);
    let cycles = 50_000u64;
    let done = Arc::new(AtomicBool::new(false));

    let mut readers = Vec::new();
    for _ in 0..4 {
        let store = Arc::clone(&store);
        let done = Arc::clone(&done);
        readers.push(thread::spawn(move || {
            while !done.load(Ordering::Acquire) {
                // Races the churner's install/release AND the drain's
                // retain/restore window; the binding is transient so both
                // Some and None are legitimate — the oracle is the id when
                // present (and no panic/deadlock under the new drain path).
                if let Some(b) = store.get(TenantId::DEFAULT, NODE, "hot-churn-key") {
                    assert_eq!(
                        b.internal_id, 7_777,
                        "foreign id served for the churned key"
                    );
                }
            }
        }));
    }

    for _ in 0..cycles {
        // Fresh install (the key was just released → vacant → re-pushes its
        // FIFO entry) then immediate release: the live set is ≤ 1 binding
        // (resident-byte trigger permanently SILENT), no checkpoint ever
        // runs (permanently `NotDurable`), and the FIFO gains one duplicate
        // entry per cycle that ONLY the cap-axis drain can reclaim.
        store.install(TenantId::DEFAULT, NODE, "hot-churn-key", 7_777);
        store.release(TenantId::DEFAULT, NODE, "hot-churn-key");
    }
    done.store(true, Ordering::Release);
    for h in readers {
        h.join().unwrap();
    }

    // Logically empty: the single churned binding was released last.
    assert_eq!(
        store.resident_len(),
        0,
        "released binding must leave the resident tier",
    );
    // Queue-bearing memory must be bounded by the ENTRY CAP, not by cycle
    // count. Loose bound = 8 × cap (post-fix equilibrium ≈ entry cap + 2 =
    // 130); pre-4b this reads ~50,000 — one leaked entry per
    // install→release cycle on ONE key.
    let qlen = store.evict_queue_len();
    assert!(
        qlen <= (8 * cap_bindings) as usize,
        "FIX-4b FAIL: {qlen} evict-queue entries retained for ONE churned \
         key (duplicate backlog behind a permanently-NotDurable front; \
         pre-fix = one entry per cycle, O(cycles) unbounded)",
    );
}
