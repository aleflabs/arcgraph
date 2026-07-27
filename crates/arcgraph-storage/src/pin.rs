//! Frame pin registry — the ADR-140-amendment-01 pin discipline.
//!
//! Closes the `Arc::strong_count`-snapshot TOCTOU documented in ADR-140
//! §D-3 "Race window" for the concurrent (M3 write-behind checkpointer)
//! regime: between the legacy `evict_lru`'s `strong_count(&latch) == 2`
//! snapshot and its `cache.remove_page(pid)`, a concurrent CRUD thread
//! that cloned the latch could mutate the page after removal and have
//! its writes silently discarded on the next fault-in — a lost write.
//!
//! The fix (ADR-140-amendment-01 §Decision item 2): removal may only
//! proceed through the same pin discipline that flushing and latch
//! acquisition use, so a latch outstanding anywhere makes the frame
//! un-removable — the window is closed **by construction, not by
//! posture**:
//!
//! - [`PinRegistry::pin`] increments the page's pin count and returns a
//!   RAII [`PinGuard`] (decrement on drop; the registry entry is
//!   removed on last unpin so the registry is O(concurrently-pinned
//!   pages), never O(pages-ever-pinned) — the resident-owner-census
//!   discipline).
//! - [`PinRegistry::remove_if_unpinned`] executes frame removal ONLY IF
//!   `pins == 0`, while still holding the registry shard's write lock —
//!   mutually exclusive with any in-flight `pin` (whose increment runs
//!   while holding the same shard's entry ref). There is no instant at
//!   which a pin can land between the check and frame removal.
//!
//! Callers that must couple a page LATCH lifetime to the pin hold both
//! in one wrapper (see `page_store::PinnedPageLatch`): acquire pin →
//! acquire latch → (use) → drop latch → drop pin.
//!
//! # Budget (PD#5, ADR-140-amendment-01 §Budget)
//!
//! `pin` = one DashMap entry op + one `fetch_add` (~tens of ns);
//! unpin = one `fetch_sub` + (last-unpin only) one `remove_if`;
//! `try_remove_unpinned` = one shard-write-locked predicate check. No
//! global locks; per-shard contention only.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use dashmap::DashMap;

/// One page's pin count. `Arc`-shared between the registry entry and
/// every outstanding [`PinGuard`].
type PinCell = Arc<AtomicUsize>;

/// RAII pin on one page frame. While any guard is live the frame is
/// un-removable via [`PinRegistry::remove_if_unpinned`].
///
/// Owns a handle to the registry map so the last unpin can retire the
/// key's entry (bounded-registry discipline).
#[derive(Debug)]
pub struct PinGuard<K: std::hash::Hash + Eq + Copy + Send + Sync + 'static> {
    map: Arc<DashMap<K, PinCell>>,
    key: K,
    cell: PinCell,
}

impl<K: std::hash::Hash + Eq + Copy + Send + Sync + 'static> PinGuard<K> {
    /// Current pin count on the frame (≥ 1 while this guard lives —
    /// observability / debug-asserts).
    #[must_use]
    pub fn pin_count(&self) -> usize {
        self.cell.load(Ordering::Acquire)
    }
}

impl<K: std::hash::Hash + Eq + Copy + Send + Sync + 'static> Drop for PinGuard<K> {
    fn drop(&mut self) {
        let prev = self.cell.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(prev >= 1, "PinGuard dropped with pin count already 0");
        if prev == 1 {
            // Last unpin: retire the registry entry IF still 0. A
            // concurrent `pin` either (a) grabbed the same cell under
            // the shard lock before this remove_if takes it — then the
            // predicate sees count ≥ 1 and the entry survives, or
            // (b) runs after the removal — `entry().or_default()`
            // re-creates a fresh cell. In both cases every increment
            // lands on the cell the registry currently holds (the
            // increment happens under the live entry ref), so no pin
            // is ever tracked on an orphaned cell — the ABA that would
            // silently re-open the TOCTOU is structurally excluded.
            let _ = self
                .map
                .remove_if(&self.key, |_, c| c.load(Ordering::Acquire) == 0);
        }
    }
}

/// Pin registry keyed by page key `K` (the record store uses `PageId`;
/// the delta-page store uses composite `(tenant, store, page)` keys).
///
/// The registry is allocation bookkeeping, not a lock: pinning never
/// blocks except briefly behind a same-shard removal claim, and holders
/// of a pin do NOT exclude each other — they only exclude *removal*.
/// Latch-level exclusion stays with the page's own `RwLock`.
#[derive(Debug)]
pub struct PinRegistry<K: std::hash::Hash + Eq + Copy + Send + Sync + 'static> {
    pins: Arc<DashMap<K, PinCell>>,
}

impl<K: std::hash::Hash + Eq + Copy + Send + Sync + 'static> Default for PinRegistry<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: std::hash::Hash + Eq + Copy + Send + Sync + 'static> PinRegistry<K> {
    #[must_use]
    pub fn new() -> Self {
        Self {
            pins: Arc::new(DashMap::new()),
        }
    }

    /// Pin `key`. The increment happens while the DashMap entry ref is
    /// held (shard-locked), so it is ordered against any concurrent
    /// [`Self::remove_if_unpinned`] on the same key.
    #[must_use]
    pub fn pin(&self, key: K) -> PinGuard<K> {
        let cell = {
            let entry = self.pins.entry(key).or_default();
            let cell = Arc::clone(entry.value());
            cell.fetch_add(1, Ordering::AcqRel);
            cell
        };
        PinGuard {
            map: Arc::clone(&self.pins),
            key,
            cell,
        }
    }

    /// Current pin count for `key` (0 if not currently pinned).
    #[must_use]
    pub fn pin_count(&self, key: K) -> usize {
        self.pins
            .get(&key)
            .map_or(0, |c| c.value().load(Ordering::Acquire))
    }

    /// Run `remove` while holding an exclusive removal claim for `key`.
    /// Returns `None` if a pin is live; otherwise calls `remove` while
    /// the registry shard's WRITE lock is still held and returns its
    /// result.
    ///
    /// # Ordering contract (the TOCTOU close)
    ///
    /// The callback is the coupling point: it MUST remove the frame
    /// synchronously and MUST NOT retain a latch-granting capability for
    /// later use. A racing `pin(key)` either (a) completed first, so its
    /// count refuses this claim, or (b) blocks on this shard and resumes
    /// only after `remove` has finished, so its subsequent frame lookup
    /// faults in canonical bytes. This is the TOCTOU close.
    pub fn remove_if_unpinned<R>(&self, key: K, remove: impl FnOnce() -> R) -> Option<R> {
        use dashmap::mapref::entry::Entry;

        // Materialize a zero-count entry even for a never-pinned key.
        // Keeping the OccupiedEntry alive keeps this key's shard
        // write-locked across the external frame-map removal.
        let occupied = match self.pins.entry(key) {
            Entry::Occupied(entry) => entry,
            Entry::Vacant(entry) => entry.insert_entry(Arc::new(AtomicUsize::new(0))),
        };
        if occupied.get().load(Ordering::Acquire) != 0 {
            return None;
        }

        let result = remove();
        let cell = occupied.remove();
        debug_assert_eq!(
            cell.load(Ordering::Acquire),
            0,
            "pin count changed while exclusive removal claim was held"
        );
        Some(result)
    }

    /// Predicate-only convenience for registry unit tests and
    /// observability. Frame owners must use [`Self::remove_if_unpinned`]
    /// so the frame removal itself is inside the claim.
    #[must_use]
    pub fn try_remove_unpinned(&self, key: K) -> bool {
        self.remove_if_unpinned(key, || ()).is_some()
    }

    /// Number of keys with a live pin entry (observability; bounded by
    /// concurrently-pinned pages per the remove-on-last-unpin rule).
    #[must_use]
    pub fn len(&self) -> usize {
        self.pins.len()
    }

    /// `len() == 0`.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pins.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pin_blocks_removal_until_dropped() {
        let reg: PinRegistry<u64> = PinRegistry::new();
        let g = reg.pin(7);
        assert_eq!(reg.pin_count(7), 1);
        assert!(!reg.try_remove_unpinned(7), "pinned key must not remove");
        drop(g);
        assert_eq!(reg.pin_count(7), 0);
        assert!(reg.try_remove_unpinned(7), "unpinned key must remove");
    }

    #[test]
    fn nested_pins_all_must_drop() {
        let reg: PinRegistry<u64> = PinRegistry::new();
        let g1 = reg.pin(3);
        let g2 = reg.pin(3);
        assert_eq!(reg.pin_count(3), 2);
        drop(g1);
        assert!(!reg.try_remove_unpinned(3));
        drop(g2);
        assert!(reg.try_remove_unpinned(3));
    }

    #[test]
    fn registry_entry_retired_on_last_unpin() {
        let reg: PinRegistry<u64> = PinRegistry::new();
        for i in 0..1000u64 {
            let g = reg.pin(i);
            drop(g);
        }
        assert_eq!(
            reg.len(),
            0,
            "registry must be O(concurrently-pinned), not O(ever-pinned)"
        );
    }

    #[test]
    fn never_pinned_key_is_removable() {
        let reg: PinRegistry<u64> = PinRegistry::new();
        assert!(reg.try_remove_unpinned(99));
    }

    #[test]
    fn pin_after_successful_remove_recreates_entry() {
        let reg: PinRegistry<u64> = PinRegistry::new();
        let g = reg.pin(5);
        drop(g);
        assert!(reg.try_remove_unpinned(5));
        assert_eq!(reg.pin_count(5), 0);
        let g2 = reg.pin(5);
        assert_eq!(reg.pin_count(5), 1);
        drop(g2);
    }

    /// Concurrency smoke: hammer pin/unpin against try_remove on one
    /// key. Invariant: whenever try_remove_unpinned returns true, no
    /// pin was live at that decision instant — verified structurally
    /// by having each pinner assert its OWN cell is the registry's
    /// current cell while pinned (an orphaned-cell increment would be
    /// the ABA the Drop impl's comment rules out).
    #[test]
    fn concurrent_pin_vs_remove_never_removes_pinned() {
        use std::sync::atomic::{AtomicBool, AtomicUsize};
        let reg = Arc::new(PinRegistry::<u64>::new());
        let stop = Arc::new(AtomicBool::new(false));
        let pause = Arc::new(AtomicBool::new(false));
        let paused = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();
        for _ in 0..4 {
            let reg = Arc::clone(&reg);
            let stop = Arc::clone(&stop);
            let pause = Arc::clone(&pause);
            let paused = Arc::clone(&paused);
            handles.push(std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    if pause.load(Ordering::Acquire) {
                        paused.fetch_add(1, Ordering::AcqRel);
                        while pause.load(Ordering::Acquire) && !stop.load(Ordering::Relaxed) {
                            std::thread::yield_now();
                        }
                        paused.fetch_sub(1, Ordering::AcqRel);
                        continue;
                    }
                    let g = reg.pin(1);
                    // While pinned, the registry MUST report ≥ 1 (our
                    // guard's cell is the live entry — orphaned-cell
                    // pins would read 0 here).
                    assert!(reg.pin_count(1) >= 1, "pin landed on orphaned cell (ABA)");
                    drop(g);
                }
            }));
        }
        let remover = {
            let reg = Arc::clone(&reg);
            let stop = Arc::clone(&stop);
            let pause = Arc::clone(&pause);
            let paused = Arc::clone(&paused);
            std::thread::spawn(move || {
                let mut removed = 0u64;
                for iteration in 0..200_000 {
                    if iteration == 100_000 {
                        // Force one deterministic unpinned window so the
                        // success arm is covered even when four pinners keep
                        // the key continuously pinned under a loaded suite.
                        pause.store(true, Ordering::Release);
                        while paused.load(Ordering::Acquire) != 4 {
                            std::thread::yield_now();
                        }
                        assert_eq!(reg.pin_count(1), 0);
                        assert!(reg.try_remove_unpinned(1));
                        removed += 1;
                        pause.store(false, Ordering::Release);
                        continue;
                    }
                    if reg.try_remove_unpinned(1) {
                        removed += 1;
                    }
                    std::hint::spin_loop();
                }
                stop.store(true, Ordering::Relaxed);
                removed
            })
        };
        let removed = remover.join().expect("remover panicked");
        for h in handles {
            h.join().expect("pinner panicked");
        }
        assert_eq!(reg.pin_count(1), 0);
        assert!(removed > 0, "remover never succeeded — vacuous test");
    }
}
