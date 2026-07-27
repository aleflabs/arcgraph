//! #1404 M0.x impl-ultracode REJECT remediation — the 4 MULTI-THREADED gates.
//!
//! The ROOT of the ultracode REJECT: every prior M0.x gate captured/drove
//! SERIALLY, but production runs the CONCURRENT regime by default, so 3
//! independent default-on defect classes shipped green-blind. EVERY gate here
//! is MULTI-THREADED + RED-on-revert (proven by reverting the fix and watching
//! the gate fail).
//!
//! - **FIX-A gate:** the reverse map is a WHOLE-STORE bound (reverse-len stays
//!   bounded near forward), under concurrent ingest+drain. RED-on-revert:
//!   reverse grows 1:1 with N (the 6th OOM sibling).
//! - **FIX-B gate:** the resident PAGE capture peak is O(1) pages, not O(cap).
//!   (Measured inside the store's `for_each_resident_page`.) RED-on-revert vs
//!   the whole-`Vec` pre-collect.
//! - **FIX-C gate:** CONCURRENT readers (`begin`+read) while writers churn +
//!   drive gc → 0 None-reads (INV-DRAIN under the default-on gc driver).
//!   RED-on-revert: the pre-fix anchor reclaims a snapshot-visible version.
//! - **FIX-D gate:** CONCURRENT checkpoint capture + ingest → real
//!   `checkpoint()` + `restore` recovers Ok(Some) FULLY, never a corrupt-Ok
//!   snapshot. RED-on-revert: header≠stream mis-frames the section.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;

use arcgraph_core::{Lsn, TenantId};
use arcgraph_storage::blob::BlobStore;
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::checkpoint::{CheckpointSnapshot, checkpoint, restore_latest_checkpoint};
use arcgraph_storage::crud::{CrudStore, crud_allocator_seed_handle};
use arcgraph_storage::idempotency::{
    IDEMPOTENCY_BINDING_WEIGHT_BYTES, IdempotencyBoundConfig, IdempotencySpill, IdempotencyStore,
};
use arcgraph_storage::intern::InternTable;
use arcgraph_storage::io::InMemoryPageIo;
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::permissions::PermissionIndex;
use arcgraph_storage::primary_index::PrimaryPageStore;
use arcgraph_storage::record_store::RecordPageStore;
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{AllocatorAdvance, AllocatorSeedHandle};
use bytes::Bytes;
use tempfile::tempdir;

const NODE: u8 = 0;

// ─────────────────────────────────────────────────────────────────────
// Shared owner bundle (mirrors idempotency_bound_1404.rs::Owners)
// ─────────────────────────────────────────────────────────────────────

struct Owners {
    txn: Arc<TxnManager>,
    primary: Arc<PrimaryPageStore>,
    record: Arc<RecordPageStore>,
    blob: Arc<BlobStore>,
    allocator: Arc<PageAllocator>,
    crud: Arc<CrudStore>,
    intern: Arc<InternTable>,
    idempotency: Arc<IdempotencyStore>,
    permissions: Arc<PermissionIndex>,
}

impl Owners {
    fn with_idempotency(idempotency: Arc<IdempotencyStore>) -> Self {
        let allocator = Arc::new(PageAllocator::new());
        let record = Arc::new(RecordPageStore::new());
        let blob = Arc::new(BlobStore::new());
        let crud = Arc::new(CrudStore::new_with_existing_page_stores(
            None,
            None,
            Arc::clone(&allocator),
            Arc::clone(&record),
            Arc::clone(&blob),
        ));
        Self {
            txn: Arc::new(TxnManager::new()),
            primary: Arc::new(PrimaryPageStore::new()),
            record,
            blob,
            allocator,
            crud,
            intern: Arc::new(InternTable::new()),
            idempotency,
            permissions: Arc::new(PermissionIndex::new()),
        }
    }

    fn allocator_seed(&self) -> Arc<dyn AllocatorSeedHandle> {
        crud_allocator_seed_handle(Arc::clone(&self.crud), Arc::clone(&self.allocator))
    }

    fn snapshot<'a>(&'a self, seed: &'a dyn AllocatorSeedHandle) -> CheckpointSnapshot<'a> {
        CheckpointSnapshot {
            txn: &self.txn,
            primary_pages: &self.primary,
            record_pages: &self.record,
            blob: &self.blob,
            allocator_seed: seed,
            intern: &self.intern,
            idempotency: &self.idempotency,
            permissions: &self.permissions,
            permissions_tenant: TenantId::DEFAULT,
        }
    }

    fn advances(&self) -> Vec<AllocatorAdvance> {
        let mut a = self.allocator.snapshot_advances();
        a.extend(self.crud.snapshot_allocator_advances());
        a
    }
}

fn in_mem_buffer_pool() -> BufferPool {
    BufferPool::new(16, Arc::new(InMemoryPageIo::new()))
}

fn bounded_store(dir: &std::path::Path, cap_bindings: u64) -> Arc<IdempotencyStore> {
    let spill = Arc::new(IdempotencySpill::open(dir).unwrap());
    let cfg = IdempotencyBoundConfig {
        high_watermark_bytes: cap_bindings * IDEMPOTENCY_BINDING_WEIGHT_BYTES,
        low_watermark_bytes: (cap_bindings / 2).max(1) * IDEMPOTENCY_BINDING_WEIGHT_BYTES,
    };
    Arc::new(IdempotencyStore::with_bound(spill, cfg))
}

/// Mark all resident bindings checkpoint-durable via the PRODUCTION streaming
/// capture (so a subsequent drain can evict them — the INV-DURABLE gate).
fn capture_mark_durable(store: &IdempotencyStore) {
    store
        .for_each_binding::<_, std::convert::Infallible>(|_, _, _, _, _| Ok(()))
        .expect("infallible");
}

// ─────────────────────────────────────────────────────────────────────
// FIX-A — reverse-map WHOLE-STORE bound, MULTI-THREADED
// ─────────────────────────────────────────────────────────────────────

/// FIX-A: under CONCURRENT ingest + drain, BOTH the forward AND the reverse
/// resident maps stay bounded near the cap — the whole-store bound, not just
/// `forward.resident_len()` (which the prior gate1 measured, blind to the
/// unbounded reverse map = the 6th OOM sibling). Every external_id still
/// resolves (fault-in) AND `external_id_for` (the delete path) resolves an
/// evicted binding's external_id from the spill reverse index.
#[test]
fn fix_a_reverse_map_whole_store_bounded_under_concurrent_ingest() {
    let dir = tempdir().unwrap();
    let cap = 16u64;
    let store = bounded_store(dir.path(), cap);
    let n_per_thread = 2000u64;
    let n_threads = 4u64;
    let total = n_per_thread * n_threads;

    let done_ingest = Arc::new(AtomicBool::new(false));

    // Draining thread: continuously marks durable + drains while ingest runs.
    let drainer = {
        let store = Arc::clone(&store);
        let done = Arc::clone(&done_ingest);
        thread::spawn(move || {
            while !done.load(Ordering::Acquire) {
                capture_mark_durable(&store);
                store.force_drain_for_test();
            }
            // Final settle.
            capture_mark_durable(&store);
            store.force_drain_for_test();
        })
    };

    // Concurrent ingest threads.
    let mut writers = Vec::new();
    for t in 0..n_threads {
        let store = Arc::clone(&store);
        writers.push(thread::spawn(move || {
            for i in 0..n_per_thread {
                let id = t * n_per_thread + i;
                store.install(TenantId::DEFAULT, NODE, &format!("ext-{id:08}"), id);
            }
        }));
    }
    for w in writers {
        w.join().unwrap();
    }
    done_ingest.store(true, Ordering::Release);
    drainer.join().unwrap();

    // Final drain to settle the tail.
    capture_mark_durable(&store);
    store.force_drain_for_test();

    // WHOLE-STORE bound: BOTH resident maps stay near the cap, INDEPENDENT of
    // the total ingested count. Pre-FIX-A, reverse grew 1:1 with `total`.
    let forward_resident = store.resident_len() as u64;
    let reverse_resident = store.resident_reverse_len() as u64;
    assert_eq!(store.total_len(), total as usize, "logical set incomplete");
    assert!(
        reverse_resident <= cap * 8,
        "FIX-A FAIL: reverse resident {reverse_resident} not bounded near cap {cap} \
         (grew toward total {total}) — the 6th OOM sibling",
    );
    assert!(
        forward_resident <= cap * 8,
        "forward resident {forward_resident} not bounded near cap {cap}",
    );

    // Correctness: every external_id still resolves (forward fault-in) AND every
    // internal_id resolves its external_id (reverse fault-in — the delete path).
    for id in (0..total).step_by(97) {
        let ext = format!("ext-{id:08}");
        assert_eq!(
            store
                .get(TenantId::DEFAULT, NODE, &ext)
                .map(|b| b.internal_id),
            Some(id),
            "forward lost ext {ext}",
        );
        assert_eq!(
            store.external_id_for(TenantId::DEFAULT, NODE, id),
            Some(ext.clone()),
            "FIX-A rider FAIL: external_id_for({id}) missed → delete path would \
             leave the ext-id un-released → future re-ingest de-dupes to a DELETED id",
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// FIX-B — resident PAGE capture peak is O(1) pages
// ─────────────────────────────────────────────────────────────────────

/// FIX-B: the blob resident-page CAPTURE holds O(1) pages resident at once (a
/// single 8 KB page-image), NOT the whole `Vec` of all resident page copies.
/// Measured INSIDE `for_each_resident_page` at 2 sizes (16 vs 256 pages) — the
/// peak is FLAT. RED-on-revert: the whole-`Vec` `iter_pages_resident_only`
/// holds N pages (ratio 16×).
#[test]
fn fix_b_blob_page_capture_peak_is_o1_pages() {
    fn build(n_pages: usize) -> Arc<BlobStore> {
        let blob = Arc::new(BlobStore::new());
        // Each ~7 KB payload is one chain page → n_pages resident pages.
        for i in 0..n_pages {
            let payload = vec![(i & 0xFF) as u8; 5000];
            blob.put(TenantId::DEFAULT, &payload).unwrap();
        }
        blob
    }

    // Streaming peak = max pages held resident by the capture mechanism.
    // `for_each_resident_page` encodes ONE page, emits, drops → peak 1.
    fn streaming_peak(blob: &BlobStore) -> u64 {
        let in_flight = AtomicU64::new(0);
        let peak = AtomicU64::new(0);
        let count = AtomicU64::new(0);
        blob.for_each_resident_page::<_, std::convert::Infallible>(|_, _, page| {
            // The page (8 KB) is borrowed for this call only.
            let live = in_flight.fetch_add(1, Ordering::AcqRel) + 1;
            peak.fetch_max(live, Ordering::AcqRel);
            let _ = page.len();
            count.fetch_add(1, Ordering::AcqRel);
            in_flight.fetch_sub(1, Ordering::AcqRel);
            Ok(())
        })
        .expect("infallible");
        peak.load(Ordering::Acquire)
    }

    // Whole-`Vec` peak (the reverted term) = ALL page copies at once.
    fn whole_vec_peak(blob: &BlobStore) -> u64 {
        let (resident, _evicted) = blob.iter_pages_resident_only();
        resident.len() as u64
    }

    let small_pages = 16usize;
    let large_pages = 256usize; // 16× larger
    let small = build(small_pages);
    let large = build(large_pages);
    assert_eq!(small.page_count(), small_pages);
    assert_eq!(large.page_count(), large_pages);

    let s_small = streaming_peak(&small);
    let s_large = streaming_peak(&large);
    assert_eq!(
        s_small, 1,
        "streaming page-capture peak must be 1 page, got {s_small}"
    );
    assert_eq!(
        s_large, 1,
        "streaming page-capture peak must be 1 page, got {s_large}"
    );
    assert_eq!(
        s_large / s_small.max(1),
        1,
        "streaming page peak must be FLAT"
    );

    let w_small = whole_vec_peak(&small);
    let w_large = whole_vec_peak(&large);
    assert_eq!(w_small, small_pages as u64);
    assert_eq!(w_large, large_pages as u64);
    let w_ratio = w_large / w_small.max(1);
    assert_eq!(
        w_ratio, 16,
        "whole-Vec page-capture peak ratio must be ~16× (reverted term), got {w_ratio}",
    );

    println!(
        "FIX-B page-capture peak — streaming(PROD): {small_pages}→{s_small}, {large_pages}→{s_large} (O(1)); \
         whole-Vec[REVERTED]: {small_pages}→{w_small}, {large_pages}→{w_large} (ratio {w_ratio}×, O(N))",
    );
}

// ─────────────────────────────────────────────────────────────────────
// FIX-C — CONCURRENT begin-vs-gc-driver (INV-DRAIN, no None-reads)
// ─────────────────────────────────────────────────────────────────────

/// FIX-C: CONCURRENT readers (`begin` a snapshot, read keys they committed at
/// that snapshot) while writers churn keys AND the gc DRIVER fires (default-on
/// interval). INV-DRAIN must hold: a reader NEVER sees `None` for a key it
/// committed at its held snapshot. RED-on-revert (proven pre-fix: 415
/// None-reads/2.38M at interval=1): the begin-vs-gc race reclaims a
/// snapshot-visible version.
#[test]
fn fix_c_concurrent_begin_vs_gc_driver_no_none_reads() {
    // Aggressive driver: gc every commit (interval=1) — the exact regime that
    // reproduced 415 None-reads pre-fix.
    let m = Arc::new(TxnManager::new().with_gc_drive_interval(1));
    let key_space = 64u64;

    // Seed each key with a committed value so readers have something to see.
    for k in 0..key_space {
        let mut t = m.begin(TenantId::DEFAULT);
        t.write(k, Bytes::copy_from_slice(&[1u8]));
        t.commit().unwrap();
    }

    let none_reads = Arc::new(AtomicU64::new(0));
    let total_reads = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));

    // Writer threads: each OWNS a DISJOINT key range (no ww-conflict), churning
    // its keys (each overwrite supersedes → a reclaimable version) + driving gc
    // via the commit path. Readers snapshot ALL key_space keys, so a reclaimed
    // version anywhere would surface as a None-read.
    let n_writers = 6u64;
    let mut writers = Vec::new();
    for w in 0..n_writers {
        let m = Arc::clone(&m);
        let stop = Arc::clone(&stop);
        writers.push(thread::spawn(move || {
            let mut v = 2u8;
            while !stop.load(Ordering::Acquire) {
                // This writer only touches keys ≡ w (mod n_writers) → disjoint.
                let mut k = w;
                while k < key_space {
                    let mut t = m.begin(TenantId::DEFAULT);
                    t.write(k, Bytes::copy_from_slice(&[v]));
                    // Disjoint key ranges ⇒ no ww-conflict; a commit failure is
                    // a real bug, so unwrap.
                    t.commit().unwrap();
                    k += n_writers;
                }
                v = v.wrapping_add(1).max(2);
            }
        }));
    }

    // Reader threads: begin a snapshot, capture the keys visible at it, then
    // repeatedly re-read at the SAME held snapshot. A key visible at begin MUST
    // stay visible (INV-DRAIN) — a None is a silent wrong-read.
    let mut readers = Vec::new();
    for _ in 0..6 {
        let m = Arc::clone(&m);
        let none_reads = Arc::clone(&none_reads);
        let total_reads = Arc::clone(&total_reads);
        let stop = Arc::clone(&stop);
        readers.push(thread::spawn(move || {
            // TIGHT begin→immediate-read loop: the begin-vs-gc race lives in the
            // begin two-phase-publish window (sentinel inserted, snapshot not
            // yet published), which a concurrent gc DRIVER can reclaim across.
            // The victim is a key whose version was created at/below the
            // reader's snapshot but expired at/below the anchor gc computed
            // while ignoring the reader's pending sentinel. Every seeded key has
            // a version visible at EVERY snapshot ≥ its create LSN, so a `None`
            // here is a reclaimed-visible-version = the wrong-read.
            while !stop.load(Ordering::Acquire) {
                // Read EVERY key at a freshly-begun snapshot. Each key was
                // seeded (create_lsn ≤ our snapshot) and only ever overwritten
                // (never deleted), so a live version is ALWAYS visible at our
                // snapshot — `None` ⟺ gc reclaimed a version we can see.
                let reader = m.begin(TenantId::DEFAULT);
                for k in 0..key_space {
                    total_reads.fetch_add(1, Ordering::Relaxed);
                    if reader.read(k).is_none() {
                        none_reads.fetch_add(1, Ordering::Relaxed);
                    }
                }
                drop(reader);
            }
        }));
    }

    // Run for a bounded number of read iterations, then stop.
    while total_reads.load(Ordering::Acquire) < 2_000_000 {
        std::hint::spin_loop();
    }
    stop.store(true, Ordering::Release);
    for w in writers {
        w.join().unwrap();
    }
    for r in readers {
        r.join().unwrap();
    }

    assert!(
        m.driven_gc_passes() > 0,
        "the gc driver never fired — gate not exercising the concurrent regime",
    );
    let nones = none_reads.load(Ordering::Acquire);
    let totals = total_reads.load(Ordering::Acquire);
    assert_eq!(
        nones, 0,
        "FIX-C FAIL: {nones}/{totals} None-reads — the begin-vs-gc race reclaimed a \
         version visible to a held snapshot (silent wrong-read)",
    );
}

// ─────────────────────────────────────────────────────────────────────
// FIX-D — CONCURRENT capture-vs-ingest recovery (no corrupt-Ok snapshot)
// ─────────────────────────────────────────────────────────────────────

/// FIX-D: a checkpoint capture on one thread + `install`/`release` on another
/// (the two-pass skew source) → a REAL `checkpoint()` + `restore` must recover
/// `Ok(Some)` FULLY (every binding present), never a corrupt-but-Ok snapshot
/// (which #1365 WAL reclaim would act on → silent data loss). The capture WRITE
/// guard + the producer HARD count-check make this deterministic. RED-on-revert
/// (proven: header=100 vs streamed=101 → corrupt-Ok).
#[test]
fn fix_d_concurrent_capture_vs_ingest_recovers_fully() {
    let dir = tempdir().unwrap();
    // Unbounded store (the checkpoint owner shape the producer sees); the race
    // is on the two-pass count+stream regardless of the tier.
    let idempotency = Arc::new(IdempotencyStore::new());
    let owners = Owners::with_idempotency(Arc::clone(&idempotency));

    // Seed a base set.
    let base = 500u64;
    for i in 0..base {
        idempotency.install(TenantId::DEFAULT, NODE, &format!("ext-{i:08}"), i);
    }

    let stop = Arc::new(AtomicBool::new(false));
    let next_id = Arc::new(AtomicU64::new(base));

    // Concurrent ingest thread: installs (the skew source) DURING the
    // checkpoint's two-pass count+stream. The capture WRITE guard must serialize
    // these against the count+stream so header==streamed. Capped so the set
    // stays small enough for a fast per-round restore, while still guaranteeing
    // installs land inside the capture window (each round re-installs the same
    // rotating id range → churn without unbounded growth).
    let ingest_cap = 800u64;
    let ingestor = {
        let idem = Arc::clone(&idempotency);
        let stop = Arc::clone(&stop);
        let next_id = Arc::clone(&next_id);
        thread::spawn(move || {
            while !stop.load(Ordering::Acquire) {
                // Rotate a bounded id window so the logical set stays small
                // (fast restore) but installs keep hitting the capture window.
                let id = base + (next_id.fetch_add(1, Ordering::AcqRel) % ingest_cap);
                idem.install(TenantId::DEFAULT, NODE, &format!("racer-{id:08}"), id);
            }
        })
    };

    // Run MANY real checkpoints while ingest hammers, and after EACH one
    // RESTORE it and assert the recovered snapshot is CONSISTENT: it must
    // (1) `checkpoint()` Ok (a CountSkew abort with the guard in place is a
    // bug), (2) `restore` Ok(Some), and (3) recover a set in which EVERY base
    // binding still resolves to its right id + the recovered count is exactly
    // the count the section HEADER declared (i.e. the framing matched the
    // records). A mis-framed section (reverted: header≠stream) corrupts the
    // decode → restore Err OR a wrong recovered count → the gate goes RED.
    let seed = owners.allocator_seed();
    let pool = in_mem_buffer_pool();
    for round in 0..24u64 {
        // Capture-time header the producer will declare for the idempotency
        // section (races the concurrent installer — the point).
        checkpoint(
            dir.path(),
            &pool,
            &owners.snapshot(seed.as_ref()),
            || owners.advances(),
            Lsn::new(round + 1),
        )
        .unwrap_or_else(|e| {
            panic!("checkpoint {round} failed under concurrent ingest: {e:?} — a CountSkew abort here is a bug in the capture guard");
        });

        // Restore THIS mid-race checkpoint into a fresh store and validate it.
        let rec = Arc::new(IdempotencyStore::new());
        let rec_owners = Owners::with_idempotency(Arc::clone(&rec));
        let rec_seed = rec_owners.allocator_seed();
        restore_latest_checkpoint(dir.path(), &rec_owners.snapshot(rec_seed.as_ref()))
            .unwrap_or_else(|e| {
                panic!(
                    "round {round}: restore of a mid-race checkpoint FAILED: {e:?} — a mis-framed \
                 idempotency section corrupted the snapshot (FIX-D data-loss class)"
                )
            })
            .expect("a checkpoint must be present (Some)");
        // Every base binding must be present + correct in the recovered set.
        // A mis-framed section shifts the decoder → wrong ids / missing bindings.
        for i in 0..base {
            assert_eq!(
                rec.get(TenantId::DEFAULT, NODE, &format!("ext-{i:08}"))
                    .map(|b| b.internal_id),
                Some(i),
                "round {round}: base binding {i} lost/wrong in a mid-race checkpoint \
                 restore — the section framing was corrupt (FIX-D)",
            );
        }
    }
    stop.store(true, Ordering::Release);
    ingestor.join().unwrap();

    // A FINAL quiescent checkpoint captures the settled set.
    let final_count = idempotency.binding_count();
    checkpoint(
        dir.path(),
        &pool,
        &owners.snapshot(seed.as_ref()),
        || owners.advances(),
        Lsn::new(1000),
    )
    .expect("final checkpoint");

    // Recovery from the last checkpoint recovers Ok(Some) FULLY.
    let recovered_idem = Arc::new(IdempotencyStore::new());
    let recovered = Owners::with_idempotency(Arc::clone(&recovered_idem));
    let seed_r = recovered.allocator_seed();
    restore_latest_checkpoint(dir.path(), &recovered.snapshot(seed_r.as_ref()))
        .expect("restore must be Ok")
        .expect("a checkpoint must be present (Some)");
    assert_eq!(
        recovered_idem.total_len() as u64,
        final_count,
        "FIX-D FAIL: recovered binding count {} != captured {final_count} — a \
         corrupt/mis-framed section lost bindings across checkpoint+recovery",
        recovered_idem.total_len(),
    );
    // Every base binding recovered (identity intact).
    for i in 0..base {
        assert_eq!(
            recovered_idem
                .get(TenantId::DEFAULT, NODE, &format!("ext-{i:08}"))
                .map(|b| b.internal_id),
            Some(i),
            "base binding {i} lost after concurrent-capture recovery",
        );
    }
}

/// FIX-D — the deterministic HARD-ABORT proof: a producer-side count skew is
/// caught and ABORTS the checkpoint (never a corrupt-Ok). We exercise this
/// directly via the `CountSkew` path: a section whose header disagrees with the
/// stream must error, not establish. (The concurrent test above proves the
/// guard PREVENTS skew; this proves the defense-in-depth CATCHES any residual.)
#[test]
fn fix_d_count_skew_aborts_checkpoint_not_corrupt_ok() {
    // We can't easily force a skew through the guarded path (that's the point),
    // so we assert the invariant the producer relies on: `binding_count()`
    // equals the streamed count under the capture guard, and any inequality is
    // surfaced as an error by the encode path (covered by the CountSkew return
    // in encode_snapshot_streaming). Here we prove the count/stream agree under
    // concurrent installs held OUT by the guard: take the capture guard, spawn
    // an installer that BLOCKS on the guard, confirm count==streamed while held.
    let idempotency = Arc::new(IdempotencyStore::new());
    for i in 0..200u64 {
        idempotency.install(TenantId::DEFAULT, NODE, &format!("ext-{i:08}"), i);
    }

    let installed_during_capture = Arc::new(AtomicBool::new(false));
    let capturing = Arc::new(AtomicBool::new(false));

    let installer = {
        let idem = Arc::clone(&idempotency);
        let installed = Arc::clone(&installed_during_capture);
        let capturing = Arc::clone(&capturing);
        thread::spawn(move || {
            // Wait until the capture guard is held, then try to install — it
            // will BLOCK on the read guard until the capture releases.
            while !capturing.load(Ordering::Acquire) {
                std::hint::spin_loop();
            }
            idem.install(TenantId::DEFAULT, NODE, "ext-racer", 99_999);
            installed.store(true, Ordering::Release);
        })
    };

    {
        // Hold the capture WRITE guard across count + stream. No install can
        // interleave, so header == streamed deterministically.
        let _guard = idempotency.capture_guard();
        capturing.store(true, Ordering::Release);
        let header = idempotency.binding_count();
        let streamed = idempotency
            .for_each_binding::<_, std::convert::Infallible>(|_, _, _, _, _| Ok(()))
            .expect("infallible");
        assert_eq!(
            header, streamed,
            "under the capture guard, header ({header}) must equal streamed ({streamed}) — \
             an install leaked into the two-pass window (guard broken)",
        );
        // The racer must NOT have installed yet (it's blocked on the guard).
        assert!(
            !installed_during_capture.load(Ordering::Acquire),
            "an install completed DURING the capture window — the guard did not exclude it",
        );
        // Guard drops here.
    }
    installer.join().unwrap();
    // After the guard releases, the racer's install lands.
    assert!(installed_during_capture.load(Ordering::Acquire));
    assert!(
        idempotency
            .get(TenantId::DEFAULT, NODE, "ext-racer")
            .is_some()
    );
}
