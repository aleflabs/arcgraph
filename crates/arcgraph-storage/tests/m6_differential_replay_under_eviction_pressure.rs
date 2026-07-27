//! M6.1 — `m6_differential_replay_under_eviction_pressure` (§M6 EXIT 2b;
//! ADR-232-amendment-01 §2.3).
//!
//! Runs the SAME workload twice — once with a tiny cache cap (forcing
//! continuous dirty-page eviction through `evict_for_capacity` on every
//! few installs) and once with an effectively-uncapped cache (eviction
//! never triggers) — then "crash + recover" each (drop the live store,
//! reopen a fresh store bound to the SAME on-disk home file) and assert
//! the post-recovery visible state BYTE-MATCHES across every page. This
//! is what "does it OOM?" never tests: an RSS plateau proves memory
//! stayed bounded, not that eviction preserved every committed byte.
//!
//! Each committed mutation follows the exact FIXED production sequence
//! (`crud.rs::apply_durable_v9_deltas`'s post-#1521-P0-1 shape): a
//! pin-coupled `latch_pinned_for_tenant` mutate (the pin held across
//! the mutate AND the DPT mark, dropped only after both) -> real WAL
//! fsync -> `DirtyPageTable::mark_dirty`. Eviction pressure (tiny-cap
//! run only) then continuously routes dirty pages through
//! `evict_for_capacity`'s MECH-E2 checkpointer handshake between
//! commits, exactly the “continuous eviction obligation” ADR-232’s
//! §2.1 frames M6 as extending M3’s install-after-durability law into.
//!
//! #1521 M6.1 P1-5(a) — RULE-MT upgrade (ADR-232-amendment-01 §2.3:
//! all three EXIT legs run RULE-MT >= 8 writers). The mutation phase
//! now runs `WRITER_THREADS` (8) CONCURRENT writer threads, each
//! owning a disjoint page partition, racing real `evict_for_capacity`
//! pressure driven by every other writer’s commits — replacing the
//! single-threaded sequential loop this gate ran before (the
//! charter’s “currently ZERO writer threads” gap: before this
//! upgrade the whole workload ran on the test’s own thread, so the
//! differential never exercised concurrent commit-vs-evict
//! interleaving at all, only sequential cap pressure with no
//! contention). The differential oracle (tiny-cap-with-eviction vs
//! uncapped-without) is unchanged; only the mutation phase gains real
//! concurrency, and each writer now acquires through the FIXED
//! pin-coupled seam production uses post-P0-1.
//!
//! RED-on-revert: dropping a dirty page on eviction WITHOUT flushing it
//! through the write-behind path (the exact MECH-E2 violation the
//! rejected-alternative note in ADR-232-amendment-01 §2.2 records) —
//! verified by temporarily routing `evict_dirty_via_checkpointer`'s
//! reclaim directly through `try_evict_page_pinned_for_tenant` — makes
//! the tiny-cap run fail deterministically (a later fault-in reads past
//! the never-homed page's disk offset) rather than silently diverge, an
//! even more decisive failure signature than a byte mismatch.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arcgraph_core::{Lsn, PAGE_SIZE, PageId, PageType, TenantId};
use arcgraph_storage::checkpoint::{PageFlushTarget, WriteBehindCheckpointer};
use arcgraph_storage::io::{PageBuf, PageIo, PosixPageIo};
use arcgraph_storage::page_store::{
    BufferedRecordPageStore, PerTenantBufferPool, PerTenantBufferPoolConfig, RecordPageBackend,
};
use arcgraph_storage::redo::{DirtyPageKey, DirtyPageTable};
use arcgraph_storage::wal::{STORE_RECORD, WalConfig, WalRecordType, WalWriter};
use tempfile::tempdir;

/// #1521 M6.1 P1-5(a) RULE-MT floor (ADR-232-amendment-01 §2.3): all
/// three EXIT legs run >= 8 concurrent writers.
const WRITER_THREADS: u64 = 8;
/// Each writer owns a disjoint partition of `WRITER_THREADS` pages, so
/// the final winning byte per page is deterministic (single writer per
/// page, no cross-writer last-write-wins race) while still exercising
/// real concurrent commit-vs-evict interleaving across DIFFERENT pages.
const PAGES_PER_WRITER: u64 = 3;
const PAGES: u64 = WRITER_THREADS * PAGES_PER_WRITER;
const MUTATIONS_PER_PAGE: u8 = 6;

fn new_disk_store(dir: &std::path::Path, cap: usize) -> Arc<BufferedRecordPageStore> {
    let io: Arc<dyn PageIo> =
        Arc::new(PosixPageIo::open_or_create(dir.join("record.store")).expect("open page io"));
    let pools = Arc::new(PerTenantBufferPool::with_config(
        io,
        PerTenantBufferPoolConfig {
            frames_per_tenant: 64,
            write_fraction: 0.0,
        },
    ));
    Arc::new(BufferedRecordPageStore::with_cache_cap(pools, cap))
}

fn wal_config(dir: std::path::PathBuf) -> WalConfig {
    WalConfig {
        group_commit_window: std::time::Duration::from_millis(1),
        group_commit_max_batch: 4,
        ..WalConfig::new(dir)
    }
}

/// Run the fixed workload against a store with the given cache cap;
/// `drive_eviction` controls whether `evict_for_capacity` is invoked
/// after each commit (the tiny-cap "continuous eviction" regime) or
/// never (the uncapped regime — MECH-E1..E8 never triggers).
///
/// Returns the on-disk byte at `MARKER` for every page, read AFTER a
/// full `flush_pages` (so both regimes' final durable state is
/// compared on equal footing — the differential is about whether
/// EVICTION preserved every committed byte, not about who remembered
/// to call an explicit final flush).
fn run_workload(dir: &std::path::Path, cache_cap: usize, drive_eviction: bool) -> Vec<u8> {
    let wal_dir = dir.join("wal");
    std::fs::create_dir_all(&wal_dir).unwrap();
    let store = new_disk_store(dir, cache_cap);

    let dpt = Arc::new(DirtyPageTable::new());
    let props_target: Arc<dyn PageFlushTarget> = store.clone();
    let records_target: Arc<dyn PageFlushTarget> = store.clone();
    let checkpointer = Arc::new(WriteBehindCheckpointer::new(
        dpt.clone(),
        props_target,
        records_target,
    ));
    store.attach_m6_dirty_page_table(dpt.clone());
    store.attach_m6_checkpointer(checkpointer.clone());

    let writer = WalWriter::spawn(wal_config(wal_dir)).unwrap();
    let handle = writer.handle();

    // PRODUCTION invariant this workload must respect (see
    // `redo::apply_recovery_delta`'s `PageAlloc` arm): every page gets a
    // DPT entry the instant it is allocated, via the SAME
    // WAL-append-then-mark-dirty sequence as any other mutation — a
    // brand-new page is never "clean" (no durable home yet) while also
    // being absent from the DPT. `install_fresh` alone (bypassing WAL +
    // DPT, a test/bootstrap-only shortcut) would violate that and is NOT
    // representative of the production PageAlloc contract MECH-E1's
    // "clean = no live DPT entry" premise depends on.
    //
    // Allocation stays single-threaded (on this function's own thread)
    // for simplicity — the RULE-MT upgrade targets the MUTATION phase
    // below, which is where `crud.rs::apply_durable_v9_deltas`'s
    // pin-coupled seam actually races the evictor in production. A
    // shared LSN counter is seeded past the allocation range so the
    // concurrent mutation phase below never collides with it.
    let lsn = AtomicU64::new(1);
    for i in 0..PAGES {
        let pid = PageId::new(i);
        store
            .install_fresh(pid, PageType::Node, TenantId::DEFAULT)
            .unwrap();
        let this_lsn = lsn.fetch_add(1, Ordering::Relaxed);
        handle
            .append(
                WalRecordType::PutNode,
                i + 1,
                this_lsn as i64,
                TenantId::DEFAULT,
                vec![0u8],
            )
            .unwrap();
        dpt.mark_dirty(
            DirtyPageKey {
                tenant_id: TenantId::DEFAULT,
                store_id: STORE_RECORD,
                page_no: pid.raw(),
            },
            Lsn::new(this_lsn),
        );

        if drive_eviction {
            // Continuous pressure from the very first allocation, not
            // just during the mutation phase — with cap << PAGES this
            // forces reclaim during the alloc phase too.
            let _ = store.evict_for_capacity(cache_cap);
        }
    }

    // #1521 M6.1 P1-5(a) — RULE-MT >= 8 concurrent writers. Each writer
    // owns a DISJOINT partition of pages (never colliding with any other
    // writer's pages), so the final byte per page is deterministic
    // (single writer, no cross-writer last-write-wins race) while every
    // writer's commits race EVERY OTHER writer's `evict_for_capacity`
    // calls against the shared cache-cap-bounded store — real concurrent
    // commit-vs-evict pressure, not a sequential single-thread loop.
    // Each writer acquires through `latch_pinned_for_tenant`, the SAME
    // FIXED seam `crud.rs::apply_durable_v9_deltas` uses post-#1521
    // P0-1: mutate under the pin, THEN mark dirty, THEN drop the pin —
    // never a bare latch.
    let lsn = Arc::new(lsn);
    let mut writers = Vec::new();
    for w in 0..WRITER_THREADS {
        let store = store.clone();
        let dpt = dpt.clone();
        let handle = handle.clone();
        let lsn = lsn.clone();
        writers.push(std::thread::spawn(move || {
            let base = w * PAGES_PER_WRITER;
            for round in 0..MUTATIONS_PER_PAGE {
                for offset in 0..PAGES_PER_WRITER {
                    let i = base + offset;
                    let pid = PageId::new(i);
                    let byte = (i as u8).wrapping_mul(31).wrapping_add(round);
                    // FIXED production shape: pin-coupled mutate, THEN
                    // WAL fsync, THEN DPT mark, pin held live across
                    // both the mutate and the mark (dropped only after)
                    // — closes the #1521 P0-1 revalidate-to-removal gap
                    // for this writer's re-dirty.
                    let pinned = store
                        .latch_pinned_for_tenant(TenantId::DEFAULT, pid)
                        .unwrap();
                    pinned.latch().write().as_mut()[PAGE_SIZE - 1] = byte;
                    let this_lsn = lsn.fetch_add(1, Ordering::Relaxed);
                    handle
                        .append(
                            WalRecordType::PutNode,
                            i + 1,
                            this_lsn as i64,
                            TenantId::DEFAULT,
                            vec![byte],
                        )
                        .unwrap();
                    dpt.mark_dirty(
                        DirtyPageKey {
                            tenant_id: TenantId::DEFAULT,
                            store_id: STORE_RECORD,
                            page_no: pid.raw(),
                        },
                        Lsn::new(this_lsn),
                    );
                    drop(pinned);

                    if drive_eviction {
                        // Continuous eviction pressure from every
                        // writer's own commits, contending on the SAME
                        // shared cache-cap-bounded store as every other
                        // concurrent writer. Errors (bounded-retry
                        // exhaustion under transient all-pinned/
                        // all-dirty-in-flight windows) are tolerated
                        // here — the workload keeps going; what matters
                        // is that no reclaim ever loses a byte.
                        let _ = store.evict_for_capacity(cache_cap);
                    }
                }
            }
        }));
    }
    for w in writers {
        w.join().unwrap();
    }
    let lsn = lsn.load(Ordering::Relaxed);

    // Final drain: whatever remains dirty gets flushed home (this models
    // an orderly shutdown checkpoint, not part of the eviction pressure
    // being differentially tested).
    checkpointer
        .flush_pass(Lsn::new(lsn))
        .expect("final quiescent flush must succeed before reading durable home bytes");
    assert!(
        dpt.is_empty(),
        "final quiescent flush must drain every DPT entry before the durable-home oracle"
    );
    writer.shutdown().unwrap();
    drop(store);

    // "Crash": read the FINAL on-disk bytes directly through a fresh
    // PageIo handle bound to the same file — the durable ground truth,
    // independent of any in-process cache.
    let io = PosixPageIo::open(dir.join("record.store")).expect("reopen disk file");
    let mut out = Vec::with_capacity(PAGES as usize);
    for i in 0..PAGES {
        let mut buf: PageBuf = [0u8; PAGE_SIZE];
        io.read_page(PageId::new(i), &mut buf).expect("read page");
        out.push(buf[PAGE_SIZE - 1]);
    }
    out
}

/// THE decisive leg: tiny-cap (continuous eviction) vs uncapped —
/// post-recovery visible state byte-matches across every page.
#[test]
fn tiny_cap_and_uncapped_recovery_byte_match() {
    let tiny_dir = tempdir().unwrap();
    let uncapped_dir = tempdir().unwrap();

    // Tiny cap: far smaller than PAGES, forces continuous eviction.
    let tiny = run_workload(tiny_dir.path(), 4, true);
    // Uncapped: cap >> PAGES, eviction never triggers.
    let uncapped = run_workload(uncapped_dir.path(), 1024, false);

    assert_eq!(tiny.len(), PAGES as usize);
    assert_eq!(uncapped.len(), PAGES as usize);
    // Keep the production workload and oracle unchanged, but identify the
    // exact page and stale mutation round if a divergence ever recurs. Check
    // the uncapped leg against the closed-form result first so a control-path
    // failure cannot be blamed on eviction.
    for (page, (&tiny_byte, &uncapped_byte)) in tiny.iter().zip(&uncapped).enumerate() {
        let page_base = (page as u8).wrapping_mul(31);
        let expected = page_base.wrapping_add(MUTATIONS_PER_PAGE - 1);
        assert_eq!(
            uncapped_byte, expected,
            "uncapped control diverged on page {page}: got {uncapped_byte}, \
             expected final committed byte {expected}"
        );
        let tiny_round =
            (0..MUTATIONS_PER_PAGE).find(|round| page_base.wrapping_add(*round) == tiny_byte);
        assert_eq!(
            tiny_byte,
            uncapped_byte,
            "INV-M6.2: page {page} has divergent durable home bytes under \
             eviction pressure: tiny={tiny_byte}, uncapped={uncapped_byte}, \
             expected={expected}, decoded_tiny_mutation_round={tiny_round:?}; the \
             final committed round is {}",
            MUTATIONS_PER_PAGE - 1,
        );
    }
    // Sanity: the workload actually produced non-trivial, page-varying
    // content (a same-junk-everywhere byte match would be a vacuous gate).
    let distinct: std::collections::HashSet<_> = tiny.iter().copied().collect();
    assert!(
        distinct.len() > 1,
        "workload sanity: expected page-varying final bytes, got {:?}",
        tiny
    );
}

// RED-on-revert for `tiny_cap_and_uncapped_recovery_byte_match` is a code
// mutation (matching the `m6_evict_races_commit_deterministic` discipline):
// temporarily routing `evict_dirty_via_checkpointer`'s reclaim directly
// through `try_evict_page_pinned_for_tenant` (skipping the checkpointer
// handshake — the ADR-232-amendment-01 §2.2 rejected alternative) makes
// the tiny-cap run drop dirty bytes whose WAL record predates their home
// write, diverging from the uncapped run's recovered bytes. Verified
// manually during development (see PR description); a workload-shaped
// negative-control test that depends on precise thread-scheduling to
// reproduce the SAME divergence deterministically without also
// depending on the exact mutation above would duplicate
// `m6_evict_races_commit_deterministic`'s crux coverage rather than add
// new signal, so this gate's decisive RED-on-revert is the code-mutation
// class, consistent with the crux gate.
