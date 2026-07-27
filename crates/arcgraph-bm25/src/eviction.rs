//! Idle-eviction policy (ADR-039 amendment-01 §D-11(b) / amendment-02
//! §D-13).
//!
//! Per-tenant idle counters drive the lazy eviction of inactive
//! `IndexWriter` instances. After an idle threshold fires, the
//! writer is dropped from its handle (releasing its
//! [`crate::pool::WriterPermit`]); the next write on that tenant
//! re-creates the writer via `Index::writer(heap_bytes)`. Re-open
//! cost is bounded by Tantivy's `meta.json` parse + segment scan,
//! typically tens of milliseconds (see
//! `benches/m3b_heap_policy.rs` D-12 measurements).
//!
//! # Two-axis idle definition
//!
//! - **Commit axis** ([`IDLE_EVICTION_COMMIT_THRESHOLD`]). Counts
//!   `commit_pending` invocations on the tenant since its last
//!   `upsert_document` / `delete_document`. A tenant whose buffer
//!   has been empty across N commits is, by construction, no longer
//!   actively writing.
//! - **Wall-clock axis**
//!   ([`IDLE_EVICTION_WALL_CLOCK_THRESHOLD_SECS`]). Time since the
//!   last write. Catches the "tenant stopped writing entirely" case,
//!   which the commit axis cannot observe (no commits occur after
//!   the last write).
//!
//! Either axis firing is sufficient for eviction. Both thresholds
//! are `pub const` so a v1.1 amendment can tune them in a single
//! line without an API surface change.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::Mutex;

/// Eviction threshold on commit axis (ADR-039 amendment-01 §D-11(b)
/// / amendment-02 §D-13).
///
/// `commit_pending` calls since the last `upsert_document` /
/// `delete_document` on a tenant. After this many empty-buffer
/// commits, the writer is evicted on the next opportunistic sweep.
///
/// The initial value `100` is conservative: at the v1.0 dev hardware
/// envelope (M-series MBP, ~50 ms / commit on the 1 M-doc gate
/// bench) this represents ~5 seconds of empty commits before
/// eviction, well under the 5-minute wall-clock threshold below.
/// The two axes are deliberately overlapping so a tenant cannot stay
/// pinned by either an idle commit stream or an idle wall clock
/// alone.
pub const IDLE_EVICTION_COMMIT_THRESHOLD: u64 = 100;

/// Eviction threshold on wall-clock axis (ADR-039 amendment-01
/// §D-11(b) / amendment-02 §D-13).
///
/// Wall-clock seconds since the last `upsert_document` /
/// `delete_document`. After this duration, the writer is evicted on
/// the next opportunistic sweep regardless of commit count.
///
/// Five minutes mirrors the eviction posture suggested in the
/// amendment-01 draft. v1.0 deployments with predictable batch
/// cadence (e.g., a 30-second ingestion cycle) will see eviction
/// fire only when ingestion truly stops; bursty workloads benefit
/// from the absolute bound.
pub const IDLE_EVICTION_WALL_CLOCK_THRESHOLD_SECS: u64 = 300;

/// Per-tenant idle counters. Held inside the
/// `arcgraph_bm25::handle::TantivyIndexInner` so every write /
/// commit on the handle updates the tracker without taking an extra
/// service-level lock.
///
/// Contention shape: every write takes the `last_write_time` mutex
/// briefly (one `Instant::now()` call). The atomic commit counter
/// has no lock. The eviction sweep takes the same mutex briefly to
/// read; the cost is dominated by the DashMap scan, not the per-
/// tenant lock.
pub struct IdleTracker {
    /// Wall-clock instant of the last `upsert_document` /
    /// `delete_document`. Updated by [`Self::note_write`].
    last_write_time: Mutex<Instant>,
    /// `commit_pending` calls since the last write. Reset to 0 on
    /// every write, incremented on every commit. Read by
    /// [`Self::is_idle`] without an explicit fence — relaxed
    /// ordering is sufficient because eviction correctness depends
    /// on a coarse-grained "approximately N commits ago" judgment,
    /// not a precise happens-before edge against the writer
    /// mutation.
    commits_since_last_write: AtomicU64,
}

impl std::fmt::Debug for IdleTracker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let elapsed = self.last_write_time.lock().elapsed();
        f.debug_struct("IdleTracker")
            .field("commits_since_last_write", &self.commits_since_last_write)
            .field("last_write_age", &elapsed)
            .finish()
    }
}

impl IdleTracker {
    /// Build a fresh tracker. Initial state: zero commits since
    /// last write, last-write instant = construction time. A handle
    /// freshly constructed by `Bm25Service::handle` is therefore
    /// "just written" for eviction purposes — eviction will not
    /// fire until either threshold is crossed after construction.
    #[must_use]
    pub fn new() -> Self {
        Self {
            last_write_time: Mutex::new(Instant::now()),
            commits_since_last_write: AtomicU64::new(0),
        }
    }

    /// Build a tracker with an explicit "last write was at" instant
    /// — test-only entry point so eviction-on-wall-clock can be
    /// driven without sleeping for the full 5-minute threshold.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn with_last_write(when: Instant) -> Self {
        Self {
            last_write_time: Mutex::new(when),
            commits_since_last_write: AtomicU64::new(0),
        }
    }

    /// Mark a write (`upsert_document` / `delete_document`). Resets
    /// the commit counter and bumps the wall-clock instant.
    pub fn note_write(&self) {
        *self.last_write_time.lock() = Instant::now();
        self.commits_since_last_write.store(0, Ordering::Relaxed);
    }

    /// Mark a commit. Increments the commit counter; does NOT touch
    /// the wall-clock instant. A tenant that commits without
    /// writing eventually crosses
    /// [`IDLE_EVICTION_COMMIT_THRESHOLD`].
    pub fn note_commit(&self) {
        self.commits_since_last_write
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Whether the tracker reports the tenant idle under EITHER
    /// axis. Cheap; safe to call from a sweep loop.
    #[must_use]
    pub fn is_idle(&self) -> bool {
        if self.commits_since_last_write.load(Ordering::Relaxed) >= IDLE_EVICTION_COMMIT_THRESHOLD {
            return true;
        }
        let elapsed = self.last_write_time.lock().elapsed();
        elapsed >= Duration::from_secs(IDLE_EVICTION_WALL_CLOCK_THRESHOLD_SECS)
    }

    /// Inspect the commit counter (test / observability).
    #[cfg(test)]
    #[must_use]
    pub(crate) fn commit_count(&self) -> u64 {
        self.commits_since_last_write.load(Ordering::Relaxed)
    }

    /// Wall-clock instant of the last write. Used by the LRU
    /// fallback in [`crate::Bm25Service::evict_to_make_room`] to
    /// pick the oldest writer for eviction when no tenant is
    /// strict-idle.
    #[must_use]
    pub fn last_write_time(&self) -> Instant {
        *self.last_write_time.lock()
    }
}

impl Default for IdleTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_tracker_is_not_idle() {
        let t = IdleTracker::new();
        assert!(!t.is_idle());
        assert_eq!(t.commit_count(), 0);
    }

    #[test]
    fn note_write_resets_commit_count() {
        let t = IdleTracker::new();
        for _ in 0..10 {
            t.note_commit();
        }
        assert_eq!(t.commit_count(), 10);
        t.note_write();
        assert_eq!(t.commit_count(), 0);
    }

    #[test]
    fn commit_axis_fires_at_threshold() {
        let t = IdleTracker::new();
        for _ in 0..(IDLE_EVICTION_COMMIT_THRESHOLD - 1) {
            t.note_commit();
        }
        assert!(!t.is_idle(), "below threshold must not be idle");
        t.note_commit();
        assert!(
            t.is_idle(),
            "AT threshold ({IDLE_EVICTION_COMMIT_THRESHOLD} commits) must be idle"
        );
    }

    #[test]
    fn wall_clock_axis_fires_after_threshold() {
        // Drive the wall-clock axis without sleeping the test for
        // 5 minutes by constructing the tracker with an old
        // last-write instant via `with_last_write`.
        let old_instant = Instant::now()
            .checked_sub(Duration::from_secs(
                IDLE_EVICTION_WALL_CLOCK_THRESHOLD_SECS + 1,
            ))
            .expect("subtracting 5 minutes from `now` must not underflow");
        let t = IdleTracker::with_last_write(old_instant);
        assert!(
            t.is_idle(),
            "wall-clock axis must report idle past threshold"
        );
    }

    #[test]
    fn note_write_clears_wall_clock_idle() {
        let old_instant = Instant::now()
            .checked_sub(Duration::from_secs(
                IDLE_EVICTION_WALL_CLOCK_THRESHOLD_SECS + 1,
            ))
            .expect("sub");
        let t = IdleTracker::with_last_write(old_instant);
        assert!(t.is_idle());
        t.note_write();
        assert!(
            !t.is_idle(),
            "post-write the tracker must report not-idle even \
             though it was previously past the wall-clock threshold"
        );
    }
}
