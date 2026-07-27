//! Shared writer pool (ADR-039 amendment-01 §D-11(c) / amendment-02
//! §D-14).
//!
//! At v1.0 the per-tenant `IndexWriter` is no longer eagerly
//! long-lived: it is allocated on first write under the cap of a
//! shared pool whose capacity bounds the simultaneous-active-writer
//! count across the whole `Bm25Service`. This caps active-set RAM at
//! `WRITER_POOL_SIZE × DEFAULT_WRITER_HEAP_BYTES` regardless of the
//! tenant population.
//!
//! # Why a pool of permits, not a pool of `IndexWriter` instances
//!
//! Tantivy's `IndexWriter` is bound to a specific `Index` (i.e., a
//! specific per-tenant directory). Two tenants cannot share an
//! `IndexWriter`, so a fungible pool of writer instances is not the
//! right shape. Instead the pool hands out **permits**: a tenant
//! must hold a permit for the lifetime of its writer, and dropping
//! the writer (via lazy eviction in [`crate::eviction`]) drops the
//! permit and frees a slot for another tenant.
//!
//! # Acquire: eager idle-sweep, then block, then timeout-gated orphan break
//!
//! [`WriterPool::acquire`] takes two callbacks, mirroring the two
//! tiers of ADR-039 amendment-02 §D-14's pool admission contract
//! ("strict idle, then LRU fallback for orphan writers"):
//!
//! - `on_full` — invoked once, eagerly, when the fast path finds the
//!   pool full. Expected to perform a **data-safe** opportunistic
//!   sweep ([`crate::Bm25Service::evict_idle`], strict-idle only); if
//!   it frees a permit the re-check succeeds without blocking.
//! - `on_block_timeout` — invoked ONLY after the admission block has
//!   elapsed ([`WRITER_ACQUIRE_BLOCK_TIMEOUT`]) with no permit
//!   released. Expected to force-evict an orphan
//!   ([`crate::Bm25Service::evict_one_lru`]) to break the deadlock.
//!
//! Under request-scoped semantics an in-flight writer that commits
//! within the timeout releases its permit at the natural commit
//! cadence and wakes a blocked acquirer via `notify_one` before the
//! timeout fires, so the forced (LRU) eviction does not reach it. This
//! is the #575 fix: the pre-fix code ran the LRU fallback EAGERLY on
//! the full-pool fast-path miss, so a contending tenant synchronously
//! evicted an in-flight writer and silently dropped its buffered docs
//! (Tantivy `IndexWriter::drop` rolls back uncommitted adds, ADR-039
//! §D-6). Gating it behind the block timeout keeps the pool oblivious
//! to the eviction policy (it only knows "eager" vs "after the block
//! timed out"). v1.1 may promote eviction to a background reaper.
//!
//! # #575 envelope (NOT an interleaving-independent guarantee)
//!
//! Forced eviction is data-safe **IFF** a writer's `upsert → commit`
//! gap is `< WRITER_ACQUIRE_BLOCK_TIMEOUT` (1 s). A **slower** in-flight
//! writer — gap `> WRITER_ACQUIRE_BLOCK_TIMEOUT` (a multi-thousand-doc
//! batch per ADR-039 amendment-02 §D-12, an fsync stall, or a
//! descheduled commit thread) — cannot be distinguished from a genuine
//! orphan by timing alone, so under saturation + contention it is
//! reclaimed as if it were an orphan and its uncommitted buffer is
//! dropped: a **latent residual of #575**, gated behind 1 s instead of
//! fired eagerly. Accepted for v1.0-α per ADR-039 amendment-03; the
//! genuine close (a lifecycle-signalled true-orphan distinction so
//! eviction never targets a committed-intent writer regardless of
//! timing) is tracked to #627, to land with the M4 / kernel
//! commit-wiring.
//!
//! The pool is a per-process resource bound and sees no global state
//! beyond its own `(in_use, capacity)` counters.

use std::sync::Arc;
use std::time::Duration;

use parking_lot::{Condvar, Mutex};

/// Default capacity of the shared `WriterPool` (ADR-039
/// amendment-01 §D-11(c) / amendment-02 §D-14).
///
/// Sized to the v1.0 alpha expectation that the active set
/// (tenants currently inside an upsert / commit window) is in the
/// tens-to-low-hundreds, not thousands. Tunable per deployment via
/// [`crate::Bm25Service::with_pool_size`]; v1.1 may promote the
/// constant to an env-var lookup once load shapes from production
/// confirm or refute the initial size.
///
/// Active-set RAM ceiling at `DEFAULT_WRITER_HEAP_BYTES = 16 MiB`
/// (ADR-039 amendment-01 §D-11(a)) is `WRITER_POOL_SIZE × 16 MiB`
/// = `1024 MiB`. The remaining tenant-count overhead is segments +
/// reader state, which is bounded separately by Tantivy's segment
/// merge policy.
pub const WRITER_POOL_SIZE: usize = 64;

/// How long [`WriterPool::acquire`] blocks on the admission
/// [`Condvar`] before invoking its `on_block_timeout` callback to
/// break a potential orphan-induced deadlock (ADR-039 amendment-02
/// §D-13 orphan safety net / §D-14 pool admission contract).
///
/// Under request-scoped semantics (§D-14) an in-flight writer that
/// commits within this timeout releases its [`WriterPermit`] at the
/// natural commit cadence — sub-ms to tens of ms for typical commits
/// (§D-12 evicted-rewrite envelope) — and wakes a blocked acquirer via
/// `notify_one` before the timeout. The timeout fires ONLY when NO
/// permit is released for the whole window: the holders are not making
/// progress. That is USUALLY orphan-induced saturation (tenants that
/// `upsert_document`-ed but never commit / rollback) — but a **slow
/// in-flight writer** whose `upsert → commit` gap exceeds this timeout
/// (a multi-thousand-doc batch §D-12, an fsync stall, or a descheduled
/// commit thread) is indistinguishable from an orphan here and is
/// reclaimed as one. Only then does `acquire` run the LRU fallback to
/// force-evict the LRU writer and break the deadlock.
///
/// **Why this is the #575 fix.** The pre-fix `acquire` ran the LRU
/// fallback EAGERLY the instant the pool was found full, so a
/// contending tenant synchronously evicted an in-flight writer that
/// held a committed-intent buffer — silently dropping its docs (the
/// buffer is Tantivy's rollback granularity, ADR-039 §D-6). Gating the
/// LRU fallback behind this block timeout makes the eager full-pool
/// miss data-safe (it only runs the strict-idle sweep) and bounds the
/// forced eviction of an in-flight writer to the case where its
/// `upsert → commit` gap exceeds the timeout — the accepted #575
/// envelope (ADR-039 amendment-03; residual tracked to #627), NOT an
/// "in-flight writers are never evicted" guarantee.
///
/// **Sizing.** One second is comfortably larger than any typical
/// single-commit latency (§D-12: tens of ms even for the 1 M-doc gate
/// bench), so a fast in-flight writer does not trip it; a slower one
/// (gap > 1 s) is the accepted envelope residual, not a guarantee
/// violation. It is small enough that a genuine orphan blocking a new
/// writer under saturation is reclaimed promptly. It is a
/// saturation-pressure-driven orphan break complementary to the
/// time-driven idle sweep
/// ([`crate::IDLE_EVICTION_WALL_CLOCK_THRESHOLD_SECS`] = 5 min): the
/// idle sweep reclaims orphans regardless of pressure; this timeout
/// reclaims them faster when a new writer is actually blocked. Because
/// its value defines the data-loss boundary (ADR-039 amendment-03
/// §D-18), it is `pub const` so a v1.1 amendment can tune it in one
/// line (mirrors the `IDLE_EVICTION_*` threshold style).
pub const WRITER_ACQUIRE_BLOCK_TIMEOUT: Duration = Duration::from_secs(1);

/// Internal pool state. Held under [`WriterPool::state`].
#[derive(Debug)]
struct WriterPoolState {
    /// Number of permits currently outstanding. Invariant:
    /// `in_use <= capacity` after every state transition.
    in_use: usize,
}

/// Shared `IndexWriter` admission pool (ADR-039 amendment-01
/// §D-11(c)).
///
/// Hands out [`WriterPermit`] instances up to a fixed capacity;
/// excess `acquire` calls block on a [`Condvar`] until a previously
/// issued permit drops.
pub struct WriterPool {
    state: Mutex<WriterPoolState>,
    available: Condvar,
    capacity: usize,
}

impl std::fmt::Debug for WriterPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let in_use = self.state.lock().in_use;
        f.debug_struct("WriterPool")
            .field("capacity", &self.capacity)
            .field("in_use", &in_use)
            .finish()
    }
}

impl WriterPool {
    /// Build a pool with the given capacity. Capacity is clamped to
    /// at least `1` so a misconfigured deployment cannot deadlock by
    /// configuring zero permits.
    #[must_use]
    pub fn new(capacity: usize) -> Arc<Self> {
        let capacity = capacity.max(1);
        Arc::new(Self {
            state: Mutex::new(WriterPoolState { in_use: 0 }),
            available: Condvar::new(),
            capacity,
        })
    }

    /// Configured capacity. Constant for the lifetime of the pool.
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Permits currently outstanding. Cheap snapshot; primarily a
    /// test / observability hook.
    #[must_use]
    pub fn in_use(&self) -> usize {
        self.state.lock().in_use
    }

    /// Try to acquire a permit without blocking. Returns `None` when
    /// the pool is at capacity.
    #[must_use]
    pub fn try_acquire(self: &Arc<Self>) -> Option<WriterPermit> {
        let mut state = self.state.lock();
        if state.in_use < self.capacity {
            state.in_use += 1;
            Some(WriterPermit {
                pool: Arc::clone(self),
            })
        } else {
            None
        }
    }

    /// Acquire a permit. The fast path matches [`Self::try_acquire`].
    /// On the slow path (pool full), the two-tier ADR-039
    /// amendment-02 §D-14 admission contract:
    /// 1. Calls `on_full` once — a **data-safe** eager opportunistic
    ///    sweep (strict-idle orphan reclamation,
    ///    [`crate::Bm25Service::evict_idle`]). Re-check; if it (or a
    ///    concurrent commit) freed a permit, succeed without blocking.
    /// 2. Otherwise block on the [`Condvar`]. A `commit_pending` /
    ///    `rollback_pending` on any tenant releases a permit at the
    ///    natural commit cadence (§D-14) and wakes us via
    ///    `notify_one`.
    /// 3. If the wait elapses ([`WRITER_ACQUIRE_BLOCK_TIMEOUT`]) with
    ///    the pool STILL full — i.e. no permit was released for the
    ///    whole window, so the holders are not making progress
    ///    (orphan-induced saturation) — call `on_block_timeout` (the
    ///    §D-14 "LRU fallback for orphan writers",
    ///    [`crate::Bm25Service::evict_one_lru`]) to force-evict an
    ///    orphan and break the deadlock, then retry.
    ///
    /// **#575 envelope (NOT an interleaving-independent guarantee).**
    /// The forced (LRU) eviction in step 3 is reachable ONLY after the
    /// block timed out with no natural release. An in-flight writer that
    /// commits within the timeout releases its permit (sub-ms to tens of
    /// ms for typical commits, §D-12) and wakes a blocked acquirer via
    /// `notify_one` in step 2 — before the timeout — so it is not the
    /// victim. Forced eviction is therefore data-safe **IFF** a writer's
    /// `upsert → commit` gap is `< WRITER_ACQUIRE_BLOCK_TIMEOUT`. A
    /// slower in-flight writer (gap `> WRITER_ACQUIRE_BLOCK_TIMEOUT`) is
    /// indistinguishable from a genuine orphan by timing alone and is
    /// reclaimed as one — its committed-intent buffer dropped: the
    /// latent #575 residual, gated behind the timeout instead of fired
    /// eagerly (the pre-fix code ran the LRU fallback EAGERLY in step 1,
    /// which is how a contending tenant silently dropped an in-flight
    /// writer's docs with no timing bound at all). Accepted for v1.0-α
    /// per ADR-039 amendment-03; the genuine close (a lifecycle-signalled
    /// true-orphan distinction) is tracked to #627.
    pub fn acquire<F, G>(self: &Arc<Self>, on_full: F, on_block_timeout: G) -> WriterPermit
    where
        F: Fn() -> usize,
        G: Fn() -> usize,
    {
        // Fast path.
        {
            let mut state = self.state.lock();
            if state.in_use < self.capacity {
                state.in_use += 1;
                return WriterPermit {
                    pool: Arc::clone(self),
                };
            }
        }

        // Eager, data-safe opportunistic sweep (strict-idle orphan
        // reclamation — the §D-14 "strict idle" first tier). Run
        // WITHOUT the pool's state lock held: the sweep takes its own
        // locks (DashMap / inner-writer mutexes), and an evicted
        // writer's `WriterPermit::drop` re-locks `state` via
        // `release()` (parking_lot is not reentrant).
        let _evicted = on_full();

        // Re-check + block-loop. Under request-scoped semantics
        // permits return at the natural commit cadence; the condvar
        // wait is the right primitive. The timeout gates the orphan
        // break (step 3) so it never fires for in-flight writers.
        let mut state = self.state.lock();
        loop {
            if state.in_use < self.capacity {
                state.in_use += 1;
                return WriterPermit {
                    pool: Arc::clone(self),
                };
            }
            let timed_out = self
                .available
                .wait_for(&mut state, WRITER_ACQUIRE_BLOCK_TIMEOUT)
                .timed_out();
            if timed_out && state.in_use >= self.capacity {
                // No permit released for the whole window → orphan-
                // induced saturation. Drop the state lock before the
                // forced eviction: it drops an ActiveWriter whose
                // `WriterPermit::drop` re-locks `state` via `release()`
                // (parking_lot is not reentrant — holding it here would
                // self-deadlock). The LRU scan's `try_lock` skips a
                // writer ONLY while it actively holds the mutex (inside
                // `commit`/`upsert`); a writer parked in the
                // upsert→commit gap (mutex free) is still eligible — the
                // #575 envelope residual when its gap exceeds the
                // timeout (amendment-03 §D-18).
                drop(state);
                let _evicted = on_block_timeout();
                state = self.state.lock();
            }
        }
    }

    /// Release one permit back to the pool. Internal to
    /// [`WriterPermit`]'s `Drop`.
    fn release(&self) {
        let mut state = self.state.lock();
        debug_assert!(
            state.in_use > 0,
            "WriterPool::release without a matching acquire — \
             permit accounting underflow"
        );
        state.in_use = state.in_use.saturating_sub(1);
        // Wake one waiter; if none waiting, this is a no-op.
        self.available.notify_one();
    }
}

/// RAII guard for one permit in a [`WriterPool`]. Dropping the guard
/// releases the permit and notifies one waiter.
///
/// Held inside `arcgraph_bm25::handle::ActiveWriter` for the lifetime
/// of a tenant's `IndexWriter`.
pub struct WriterPermit {
    pool: Arc<WriterPool>,
}

impl std::fmt::Debug for WriterPermit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WriterPermit")
            .field("pool_capacity", &self.pool.capacity)
            .finish()
    }
}

impl Drop for WriterPermit {
    fn drop(&mut self) {
        self.pool.release();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_acquire_returns_some_under_capacity() {
        let pool = WriterPool::new(2);
        let p1 = pool.try_acquire();
        let p2 = pool.try_acquire();
        let p3 = pool.try_acquire();
        assert!(p1.is_some());
        assert!(p2.is_some());
        assert!(p3.is_none(), "third permit must be denied at capacity 2");
        assert_eq!(pool.in_use(), 2);
    }

    #[test]
    fn permit_drop_releases_to_pool() {
        let pool = WriterPool::new(1);
        {
            let _p = pool.try_acquire().expect("first");
            assert_eq!(pool.in_use(), 1);
        }
        // Permit dropped at end of scope — pool should be empty.
        assert_eq!(pool.in_use(), 0);
        let _p2 = pool.try_acquire().expect("post-drop reacquire");
    }

    #[test]
    fn capacity_zero_clamps_to_one() {
        // Per `new`'s clamp: configuring zero capacity must not
        // deadlock the system.
        let pool = WriterPool::new(0);
        assert_eq!(pool.capacity(), 1);
        let _p = pool.try_acquire().expect("clamped capacity admits 1");
    }

    #[test]
    fn acquire_invokes_sweeper_when_full() {
        // Drive: pool size 1, acquire one permit, then issue a
        // blocking acquire from the SAME thread with a sweeper
        // that releases the permit. The sweeper must be invoked
        // (returning > 0) and the second acquire must succeed
        // without blocking forever.
        use std::sync::atomic::{AtomicUsize, Ordering};
        let pool = WriterPool::new(1);
        let first = pool.try_acquire().expect("first permit");
        let sweep_calls = AtomicUsize::new(0);
        // Wrap `first` in an Option behind an UnsafeCell-equivalent
        // we can take from inside the closure. parking_lot::Mutex
        // is the standard idiom.
        let first_slot: parking_lot::Mutex<Option<WriterPermit>> =
            parking_lot::Mutex::new(Some(first));

        let _second = pool.acquire(
            || {
                sweep_calls.fetch_add(1, Ordering::SeqCst);
                // Drop the first permit so the pool re-checks and
                // succeeds (the EAGER on-full tier frees the permit, so
                // the block + timeout-gated orphan break is never
                // reached).
                let _dropped = first_slot.lock().take();
                1
            },
            // on_block_timeout: unreachable here — the eager sweep
            // already freed a permit.
            || 0,
        );
        assert_eq!(sweep_calls.load(Ordering::SeqCst), 1);
        assert_eq!(pool.in_use(), 1, "second permit holds; first was released");
    }

    #[test]
    fn concurrent_acquire_blocks_then_succeeds_on_release() {
        // Driver: pool size 1; thread A holds the permit and sleeps;
        // thread B blocks on acquire (sweeper returns 0); thread A
        // releases; thread B unblocks and acquires.
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::mpsc;
        use std::thread;
        use std::time::Duration;

        let pool = WriterPool::new(1);
        let p_a = pool.try_acquire().expect("A");
        let pool_b = Arc::clone(&pool);
        let (tx, rx) = mpsc::channel::<()>();
        let b_acquired = Arc::new(AtomicBool::new(false));
        let b_acquired_clone = Arc::clone(&b_acquired);

        let handle = thread::spawn(move || {
            // Both callbacks return 0 (no eviction available): B blocks
            // on the condvar and is woken by A's release via
            // `notify_one` (NOT by the timeout-gated orphan break).
            let _p_b = pool_b.acquire(|| 0, || 0);
            b_acquired_clone.store(true, Ordering::SeqCst);
            // Notify the test that we successfully acquired.
            tx.send(()).expect("send acquired");
        });

        // Give thread B a moment to reach `acquire` and block.
        thread::sleep(Duration::from_millis(50));
        assert!(
            !b_acquired.load(Ordering::SeqCst),
            "B must still be blocked while A holds the only permit"
        );

        // Release A's permit; B should unblock and complete.
        drop(p_a);
        rx.recv_timeout(Duration::from_secs(2))
            .expect("B must acquire within 2s of A releasing");
        assert!(b_acquired.load(Ordering::SeqCst));
        handle.join().expect("B thread joined cleanly");
    }
}
