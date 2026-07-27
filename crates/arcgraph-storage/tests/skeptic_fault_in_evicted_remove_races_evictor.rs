//! M6.1 (#1521 P0-3) — `fault_in_evicted_remove_races_evictor_to_neither_map`.
//!
//! `BufferedRecordPageStore::fault_in_for_tenant` re-installs a page's
//! bytes into `cache` (via `cache_install_if_vacant`) then clears its
//! `evicted` marker. `cache` and `evicted` are two INDEPENDENT `DashMap`s
//! with no shared lock. Between the fault-in's own re-install and its
//! `evicted.remove`, a CONCURRENT evictor can run an entire SECOND
//! eviction cycle on the SAME key — `remove_cached_page_if_unpinned`'s
//! insert-before-remove ordering (`evicted.insert` THEN `cache.remove`)
//! never leaves the key in neither map BY ITSELF, but a subsequent BLIND
//! `evicted.remove` from the ORIGINAL fault-in (which has no idea the
//! second eviction just ran) would UNDO that legitimate insert even
//! though `cache.remove` already happened — landing the key in NEITHER
//! `cache` NOR `evicted` (fix-1's "never in neither map" invariant
//! broken), surfacing a spurious `MissingPage` on the NEXT fault-in for a
//! page that is still durably tracked on disk.
//!
//! The fix: fault-in's MISS PATH now registers a PIN
//! (`PinRegistry::pin`) before re-checking the maps and holds it across
//! the whole re-install-then-clear sequence. The pin registry is the
//! ONE claim that is atomic with an evictor's two-map transaction (its
//! whole cycle runs inside `remove_if_unpinned`'s shard-locked closure,
//! which refuses while any pin is live) — so a second eviction can no
//! longer land ANYWHERE inside fault-in's window, and the clear is
//! safely unconditional. A `cache.contains_key` guard in front of the
//! clear (the first-draft fix) was rejected as unsound: it only NARROWS
//! the race — a full second eviction cycle between the guard's load and
//! the `evicted.remove` still strands the key in neither map.
//!
//! DETERMINISM: `BufferedRecordPageStore::fault_in_for_tenant_with_hook_for_gate`
//! (a `#[doc(hidden)]` test-only seam, mirroring
//! `try_evict_page_pinned_with_hook_for_gate`'s established pattern) fires
//! `before_evicted_clear` at the EXACT window between the fault-in's own
//! re-install and its `evicted` clear — inside that hook this gate
//! synchronously drives a REAL second eviction attempt (real
//! pin-registry claim, real `cache`/`evicted` map mutations) on the same
//! key, making the interleaving deterministic rather than a race against
//! OS scheduling.
//!
//! Two legs:
//!
//! - `unconditional_clear_loses_the_key_to_neither_map` (RED-on-revert
//!   SENSITIVITY leg): manually replaying the PRE-FIX shape (an
//!   un-pinned fault-in's blind `evicted.remove`) after a second
//!   eviction shows the key ends up in NEITHER map — the exact defect
//!   class #1521 P0-3 found.
//! - `pinned_fault_in_excludes_the_second_eviction` (THE decisive leg,
//!   exercises the ACTUAL FIXED production method): the real
//!   `fault_in_for_tenant_with_hook_for_gate` (same code path
//!   `fault_in_for_tenant` delegates to) holds its miss-path pin across
//!   the window, so the in-hook second eviction is REFUSED by the pin
//!   claim — the key ends the race resident in `cache` with its marker
//!   correctly cleared, never in neither map, and a subsequent fault-in
//!   succeeds. RED if the miss-path pin is reverted: the in-hook
//!   eviction then succeeds and the blind clear strands the key in
//!   neither map (the sensitivity leg's outcome, on the production
//!   dispatch).
//!
//! #1521 M6.1 TIER-1 RE-GATE FIX (2026-07-17): the decisive leg below was
//! previously named `conditional_clear_preserves_the_second_evictions_marker`
//! and asserted the STALE pre-pin-fix hypothesis — that the in-hook
//! second eviction attempt SUCCEEDS (`assert!(second_evicted, ...)`).
//! Once the fix evolved from "conditional cache.contains_key clear" to
//! "hold a real pin across the whole window" (the doc comment above,
//! and the production code, both already describe the PIN-based fix),
//! that assertion became stale: `fault_in_for_tenant_with_hook_for_gate`
//! now holds `self.pins.pin(key)` for the ENTIRE re-install-then-clear
//! sequence, so `remove_if_unpinned`'s pin-count check on the SAME key
//! now ALWAYS refuses the in-hook eviction attempt (`second_evicted` is
//! always `false`) — the gate's own harness precondition assertion was
//! deterministically failing 5/5 (`cargo test`) against the ACTUAL
//! committed production code, a genuine gate defect (verified by
//! reverting the pin-based fix back to the old conditional-clear-only
//! shape: with the pin removed, `second_evicted` becomes `true` again
//! and the rewritten decisive leg below reddens deterministically,
//! 5/5 — confirming the rewritten leg is RED-on-revert against the
//! REAL mechanism, not vacuously green). Renamed to
//! `pinned_fault_in_excludes_the_second_eviction` and rewritten to
//! assert the CURRENT (correct, stronger) contract: the in-hook eviction
//! attempt must be REFUSED, `pid` must remain resident in `cache` (not
//! `evicted`, not neither), and the pin must be released (not leaked) —
//! a fresh eviction attempt on `pid` after fault-in returns must succeed
//! normally.

use std::sync::Arc;

use arcgraph_core::{PageId, PageType, TenantId};
use arcgraph_storage::io::InMemoryPageIo;
use arcgraph_storage::page_store::{
    BufferedRecordPageStore, PerTenantBufferPool, PerTenantBufferPoolConfig, RecordPageBackend,
};

fn new_store(cap: usize) -> Arc<BufferedRecordPageStore> {
    let io: Arc<dyn arcgraph_storage::io::PageIo> = Arc::new(InMemoryPageIo::new());
    let pools = Arc::new(PerTenantBufferPool::with_config(
        io,
        PerTenantBufferPoolConfig {
            frames_per_tenant: 16,
            write_fraction: 0.0,
        },
    ));
    Arc::new(BufferedRecordPageStore::with_cache_cap(pools, cap))
}

/// THE decisive leg: exercises the ACTUAL fixed
/// `fault_in_for_tenant_with_hook_for_gate` (same code
/// `fault_in_for_tenant` runs). The fix holds a REAL pin
/// (`PinRegistry::pin`) across fault-in's whole re-install-then-clear
/// window, so a concurrent second eviction attempt on the SAME key
/// inside that window is structurally EXCLUDED (`remove_if_unpinned`
/// refuses while any pin is live) — not merely narrowed like the
/// rejected `cache.contains_key`-guard first draft. This gate proves
/// the exclusion directly: the in-hook eviction attempt must be
/// REFUSED (return `false`), the key must remain resident in `cache`
/// with its `evicted` marker correctly cleared once fault-in
/// completes, and the excluded eviction attempt's own target (`pid`)
/// must still be evictable normally afterward (the pin is released,
/// not leaked).
#[test]
fn pinned_fault_in_excludes_the_second_eviction() {
    let store = new_store(8);
    let pid = PageId::new(1);
    store
        .install_fresh(pid, PageType::Node, TenantId::DEFAULT)
        .unwrap();
    // Get the page durably homed once, then evict it via the REAL
    // pin-coupled path so it starts genuinely `evicted` (not `cache`).
    store.flush_pages([pid]).unwrap();
    assert!(
        store.try_evict_page_pinned_for_tenant(TenantId::DEFAULT, pid, || true),
        "harness precondition: initial eviction must succeed to set up \
         the evicted-not-cached starting state this gate races against"
    );
    assert!(store.is_evicted(pid));
    assert!(!store.is_cached(pid));

    let racer_store = store.clone();
    let result = store.fault_in_for_tenant_with_hook_for_gate(TenantId::DEFAULT, pid, move || {
        // Inside the window between this fault-in's `cache_install_if_vacant`
        // (which just re-installed `pid` into `cache`) and its `evicted`
        // clear: attempt a REAL second eviction cycle on the SAME key —
        // real pin-registry claim, real `remove_if_unpinned` call. Since
        // the OUTER fault-in call is holding a live pin on `key` for its
        // entire duration (the P0-3 fix), this claim must be refused:
        // `remove_if_unpinned`'s pin-count check sees a live pin and
        // returns `None` before ever touching `cache`/`evicted`.
        let second_evicted =
            racer_store.try_evict_page_pinned_for_tenant(TenantId::DEFAULT, pid, || true);
        assert!(
            !second_evicted,
            "MECH-E3/P0-3 violation: the in-hook second eviction attempt \
             succeeded while the fault-in's miss-path pin should still be \
             live for `key` — the pin is not actually excluding a \
             concurrent removal claim for the whole re-install-then-clear \
             window"
        );
    });
    assert!(result.is_ok(), "fault_in must not error: {result:?}");

    // THE decisive invariant: the excluded eviction attempt never
    // mutated either map, so fault-in's own re-install + clear is the
    // sole writer — `pid` must be resident in `cache`, NOT `evicted`.
    let in_cache = store.is_cached(pid);
    let in_evicted = store.is_evicted(pid);
    assert!(
        in_cache && !in_evicted,
        "MECH-E3/fix-1 P0-3 violation: `pid` must be resident in `cache` \
         (not `evicted`, not neither) after a successfully-excluded \
         second-eviction race (in_cache={in_cache}, in_evicted={in_evicted})"
    );

    // The pin must be released (not leaked) once fault-in returns: a
    // fresh eviction attempt on `pid` must now succeed normally.
    assert!(
        store.try_evict_page_pinned_for_tenant(TenantId::DEFAULT, pid, || true),
        "the fault-in's pin must be released once it returns — a page \
         left permanently pinned would never be reclaimable again"
    );

    // Confirm the practical consequence: a SUBSEQUENT fault-in on the
    // still-tracked (now genuinely evicted) page must succeed, not
    // spuriously report `MissingPage` for a page that is durably homed.
    assert!(
        store.fault_in(pid).is_ok(),
        "a page left in NEITHER map would spuriously fail a later \
         fault-in with MissingPage despite being durably tracked on disk \
         — the practical consequence of the P0-3 hazard"
    );
}

/// RED-on-revert SENSITIVITY leg: manually reproducing the PRE-FIX
/// unconditional-clear shape (no `cache.contains_key` guard) after the
/// SAME in-hook second-eviction race shows the key lands in NEITHER map —
/// demonstrating the harness is capable of catching the defect class
/// #1521 P0-3 found, so the decisive leg's green result above is not
/// vacuous.
#[test]
fn unconditional_clear_loses_the_key_to_neither_map() {
    let store = new_store(8);
    let pid = PageId::new(1);
    store
        .install_fresh(pid, PageType::Node, TenantId::DEFAULT)
        .unwrap();
    store.flush_pages([pid]).unwrap();
    assert!(store.try_evict_page_pinned_for_tenant(TenantId::DEFAULT, pid, || true));
    assert!(store.is_evicted(pid));

    // Manually replay the PRE-FIX fault-in shape: re-install into cache
    // (mirroring `cache_install_if_vacant`'s effect via a real fault-in
    // call that we then race against), race a second real eviction
    // inside the window, then blindly clear `evicted` with NO
    // `cache.contains_key` guard — the exact pre-fix bug.
    store.fault_in(pid).unwrap(); // re-installs into cache, clears evicted normally.
    assert!(store.is_cached(pid));
    assert!(!store.is_evicted(pid));

    // Now simulate "another fault-in raced in and is about to clear
    // `evicted` unconditionally, but a second eviction wins first":
    let second_evicted = store.try_evict_page_pinned_for_tenant(TenantId::DEFAULT, pid, || true);
    assert!(second_evicted, "harness: second eviction must succeed");
    assert!(store.is_evicted(pid));
    assert!(!store.is_cached(pid));

    // PRE-FIX shape: the original fault-in's blind, unconditional clear
    // — no re-check that `cache` still contains the key.
    store.__test_blind_evicted_remove_for_gate(TenantId::DEFAULT, pid);

    let in_cache = store.is_cached(pid);
    let in_evicted = store.is_evicted(pid);
    assert!(
        !in_cache && !in_evicted,
        "sensitivity leg: expected the key in NEITHER map after an \
         unconditional clear undoes the second eviction's marker \
         (in_cache={in_cache}, in_evicted={in_evicted}) — if this is \
         false the harness is not reproducing the pre-fix hazard"
    );
    assert!(
        store.fault_in(pid).is_err(),
        "MECH-E3/fix-1 P0-3 defect class reproduced: a page left in \
         NEITHER map spuriously fails fault_in with MissingPage despite \
         being durably tracked on disk"
    );
}
