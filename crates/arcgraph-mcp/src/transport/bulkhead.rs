//! AHP-1 — `spawn_blocking` bulkhead for the synchronous MCP dispatch
//! (ADR-225 §3 row AHP-1; closes #999 at the transport level; rc-track).
//!
//! # The #999 mechanism this fixes
//!
//! The MCP dispatcher is synchronous (`Dispatcher::dispatch(&self, req)
//! -> Option<Value>`, `crate::transport::mod`). The served transports
//! call it **inline on the connection's async task**. Any engine call
//! that blocks — a cold page read, a group-commit `fdatasync` wait — then
//! pins the Tokio worker it runs on for the write's full duration. With
//! `W` workers, `W` concurrent durable writes starve *every* read on the
//! server, cross-tenant (ADR-225 §1.2). That is #999 at transport level.
//!
//! # What the bulkhead does
//!
//! [`DispatchBulkhead::run`] moves the blocking dispatch onto a
//! `tokio::task::spawn_blocking` thread (OFF the reactor) behind a
//! **bounded** [`tokio::sync::Semaphore`]. The reactor is freed to serve
//! other connections' reads while a write blocks; the semaphore caps the
//! number of concurrent blocking dispatches so a burst cannot spawn an
//! unbounded number of blocking-pool threads (ADR-225 §6 AHP-1 risk row:
//! "blocking-pool exhaustion under burst").
//!
//! A per-call deadline is honoured at the bulkhead boundary via
//! [`tokio::time::timeout`] around the `JoinHandle`: a timed-out request
//! stops *awaiting* the blocking work and returns [`BulkheadOutcome::TimedOut`]
//! (closing the http.rs:1420–1426 in-code TODO). `spawn_blocking` cannot
//! be force-cancelled, so the abandoned thread runs to completion — but it
//! holds its semaphore permit until it truly finishes (the permit is moved
//! *into* the closure), so the concurrency bound stays honest even across
//! a timeout.
//!
//! # Latency budget (PD#5, ADR-225 §5.5)
//!
//! Bulkhead dispatch overhead ≤ 30 µs P50 per dispatch. In the common
//! case the semaphore is uncontended (cap = 2 × cores) so the cost is one
//! uncontended permit acquire + one `spawn_blocking` handoff; both are
//! sub-microsecond amortised. The bound only engages under saturation,
//! where queueing is the intended backpressure.

use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;

/// Default multiplier applied to the logical-core count when no explicit
/// permit cap is configured (ADR-225 §3 AHP-1 row: "cap ≈ 2 × cores").
const DEFAULT_PERMITS_PER_CORE: usize = 2;

/// Resolve the default bulkhead permit cap: `2 × available_parallelism`,
/// clamped to at least 1. Falls back to `2` when the platform cannot
/// report parallelism (a single-core assumption × the 2× multiplier).
#[must_use]
pub fn default_permits() -> usize {
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1);
    (cores * DEFAULT_PERMITS_PER_CORE).max(1)
}

/// Operator-facing configuration for the dispatch bulkhead.
///
/// Carries `#[serde(deny_unknown_fields)]` under the strict public-contract policy so a
/// misspelled key rejects at startup rather than silently degrading. At
/// v1.0-α this is constructed programmatically (the CLI wires it from a
/// `serve` flag); the serde derives forward-bind against the M5 / M6
/// server-config landing, matching [`crate::transport::bolt::BoltServerConfig`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BulkheadConfig {
    /// Maximum number of concurrent blocking dispatches. `None` (the
    /// default) resolves to [`default_permits`] (2 × logical cores) at
    /// build time; `Some(0)` is clamped up to 1 by [`DispatchBulkhead::new`]
    /// (a zero-permit semaphore would deadlock every dispatch).
    #[serde(default)]
    pub permits: Option<usize>,
}

impl BulkheadConfig {
    /// Materialise the runtime [`DispatchBulkhead`] this config describes.
    #[must_use]
    pub fn resolve(&self) -> DispatchBulkhead {
        DispatchBulkhead::new(self.permits.unwrap_or_else(default_permits))
    }
}

/// The outcome of a bulkheaded dispatch.
///
/// Generic over the blocking closure's return type so each transport can
/// carry its own dispatch result (`Option<Value>` for the JSON-RPC
/// transports; a Bolt `RunOutcome` tuple for the Bolt RUN path).
#[derive(Debug)]
pub enum BulkheadOutcome<T> {
    /// The blocking dispatch ran to completion and produced `T`.
    Completed(T),
    /// The per-call deadline elapsed before the blocking dispatch
    /// returned. The caller stops awaiting and should render a
    /// deadline/cancelled response; the abandoned blocking thread runs to
    /// completion off-reactor (best-effort — `spawn_blocking` cannot be
    /// force-cancelled) while still holding its permit.
    TimedOut,
    /// The blocking task panicked (a `JoinError`). Surfaced so the caller
    /// renders an internal-error response rather than silently dropping
    /// the request.
    Panicked,
}

/// A bounded bulkhead that runs synchronous dispatch work off the Tokio
/// reactor via `spawn_blocking`, capping concurrency with a shared
/// semaphore.
///
/// Cheaply cloneable (an `Arc<Semaphore>` inside) so one instance is
/// shared across every connection/request of a transport — clones share
/// the same permit pool, which is what enforces the global bound.
#[derive(Debug, Clone)]
pub struct DispatchBulkhead {
    sem: Arc<Semaphore>,
    permits: usize,
}

impl DispatchBulkhead {
    /// Construct a bulkhead permitting `permits` concurrent blocking
    /// dispatches. `permits` is clamped up to 1 — a zero-permit semaphore
    /// would deadlock every dispatch.
    #[must_use]
    pub fn new(permits: usize) -> Self {
        let permits = permits.max(1);
        Self {
            sem: Arc::new(Semaphore::new(permits)),
            permits,
        }
    }

    /// Construct a bulkhead with the default cap (2 × logical cores).
    #[must_use]
    pub fn with_default_cap() -> Self {
        Self::new(default_permits())
    }

    /// The configured permit cap (max concurrent blocking dispatches).
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.permits
    }

    /// Currently-available permits (capacity minus in-flight dispatches).
    /// Primarily for tests + observability.
    #[must_use]
    pub fn available_permits(&self) -> usize {
        self.sem.available_permits()
    }

    /// Run `f` on a `spawn_blocking` thread behind the bounded semaphore,
    /// honouring an optional `deadline` at the bulkhead boundary.
    ///
    /// Acquires a permit first (blocking here — the intended backpressure
    /// — when all permits are in use), then hands `f` to `spawn_blocking`.
    /// The permit is moved *into* the blocking closure so it is released
    /// only when the blocking work truly finishes, keeping the concurrency
    /// bound honest even if the caller stops awaiting on a timeout.
    ///
    /// - `Some(deadline)` → the `JoinHandle` is wrapped in
    ///   [`tokio::time::timeout`]; on elapse the caller stops awaiting and
    ///   gets [`BulkheadOutcome::TimedOut`].
    /// - `None` → awaits the `JoinHandle` to completion (no per-call
    ///   deadline; e.g. the Bolt / stdio paths, which carry none).
    pub async fn run<F, T>(&self, deadline: Option<Duration>, f: F) -> BulkheadOutcome<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        // `acquire_owned` only errors when the semaphore is closed; this
        // bulkhead never closes its semaphore, so the error arm is
        // unreachable (code-quality policy: `expect` with a proving reason).
        let permit = self
            .sem
            .clone()
            .acquire_owned()
            .await
            .expect("dispatch bulkhead semaphore is never closed");

        let handle = tokio::task::spawn_blocking(move || {
            // Hold the permit for the exact lifetime of the blocking work.
            // On a deadline timeout the caller drops the `JoinHandle` but
            // NOT this permit — it is released here, when the abandoned
            // thread actually returns, so a timed-out-but-still-running
            // dispatch keeps counting against the cap.
            let _permit = permit;
            f()
        });

        match deadline {
            Some(budget) => match tokio::time::timeout(budget, handle).await {
                Ok(Ok(value)) => BulkheadOutcome::Completed(value),
                Ok(Err(_join_err)) => BulkheadOutcome::Panicked,
                Err(_elapsed) => BulkheadOutcome::TimedOut,
            },
            None => match handle.await {
                Ok(value) => BulkheadOutcome::Completed(value),
                Err(_join_err) => BulkheadOutcome::Panicked,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::Instant;

    #[test]
    fn new_clamps_zero_permits_up_to_one() {
        let b = DispatchBulkhead::new(0);
        assert_eq!(
            b.capacity(),
            1,
            "zero permits must clamp to 1 (no deadlock)"
        );
        assert_eq!(b.available_permits(), 1);
    }

    #[test]
    fn default_permits_is_two_per_core_at_least_two() {
        let p = default_permits();
        assert!(p >= 2, "default cap is 2× cores, so ≥ 2 on any platform");
    }

    #[test]
    fn config_default_resolves_to_default_permits() {
        let cfg = BulkheadConfig::default();
        assert_eq!(cfg.resolve().capacity(), default_permits());
    }

    #[test]
    fn config_explicit_permits_wins() {
        let cfg = BulkheadConfig { permits: Some(3) };
        assert_eq!(cfg.resolve().capacity(), 3);
    }

    #[test]
    fn config_rejects_unknown_fields() {
        // deny_unknown_fields: a misspelled key rejects at parse time.
        let r: Result<BulkheadConfig, _> = serde_json::from_str(r#"{"permit":4}"#);
        assert!(r.is_err(), "unknown field `permit` must reject");
        let ok: BulkheadConfig = serde_json::from_str(r#"{"permits":4}"#).unwrap();
        assert_eq!(ok.permits, Some(4));
    }

    #[tokio::test]
    async fn completed_returns_value_and_releases_permit() {
        let b = DispatchBulkhead::new(4);
        let out = b.run(None, || 21 * 2).await;
        assert!(matches!(out, BulkheadOutcome::Completed(42)));
        // Permit released after the dispatch returns.
        assert_eq!(b.available_permits(), 4, "permit must be released post-run");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn semaphore_bounds_concurrency_extra_dispatch_queues() {
        // cap = 1: a second concurrent dispatch must QUEUE behind the
        // first (not run unboundedly). We prove B has not started while A
        // holds its permit, then that B runs once A releases.
        let b = DispatchBulkhead::new(1);
        let started = Arc::new(AtomicUsize::new(0));

        // A: enters the closure, signals "started", then blocks until we
        // release it via a std mpsc (blocking recv is fine on a
        // spawn_blocking thread).
        let (release_a_tx, release_a_rx) = std::sync::mpsc::channel::<()>();
        let (a_started_tx, a_started_rx) = tokio::sync::oneshot::channel::<()>();
        let started_a = started.clone();
        let b_a = b.clone();
        let a = tokio::spawn(async move {
            b_a.run(None, move || {
                started_a.fetch_add(1, Ordering::SeqCst);
                let _ = a_started_tx.send(());
                // Park the single permit until the test releases it.
                let _ = release_a_rx.recv();
            })
            .await
        });

        // Wait until A actually holds the permit + is inside its closure.
        a_started_rx.await.expect("A entered its closure");
        assert_eq!(b.available_permits(), 0, "A holds the only permit");

        // B: launched while A holds the permit. It must not enter its
        // closure until A releases.
        let started_b = started.clone();
        let b_b = b.clone();
        let b_task = tokio::spawn(async move {
            b_b.run(None, move || {
                started_b.fetch_add(1, Ordering::SeqCst);
                7
            })
            .await
        });

        // Give B a chance to (incorrectly) start. It must NOT — the
        // semaphore holds it at permit-acquire.
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(
            started.load(Ordering::SeqCst),
            1,
            "only A has entered its closure; B must be queued at the semaphore"
        );

        // Release A → B can now acquire the permit and run.
        release_a_tx.send(()).expect("release A");
        let _ = a.await.expect("A task joins");
        let out_b = b_task.await.expect("B task joins");
        assert!(matches!(out_b, BulkheadOutcome::Completed(7)));
        assert_eq!(
            started.load(Ordering::SeqCst),
            2,
            "B ran only after A released the permit"
        );
        assert_eq!(b.available_permits(), 1, "permit returns to the pool");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn deadline_exceeded_returns_timed_out_without_awaiting_full_work() {
        // A dispatch that sleeps far longer than its deadline must return
        // TimedOut promptly (the caller stops awaiting), NOT after the
        // full blocking duration.
        let b = DispatchBulkhead::new(4);
        let start = Instant::now();
        let out: BulkheadOutcome<u8> = b
            .run(Some(Duration::from_millis(50)), || {
                std::thread::sleep(Duration::from_secs(5));
                1
            })
            .await;
        let elapsed = start.elapsed();
        assert!(matches!(out, BulkheadOutcome::TimedOut));
        assert!(
            elapsed < Duration::from_secs(1),
            "must return at the deadline (~50ms), not the 5s work; took {elapsed:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn timed_out_dispatch_keeps_its_permit_until_the_thread_finishes() {
        // The permit for a timed-out dispatch is released only when the
        // abandoned blocking thread actually returns — not at the timeout.
        let b = DispatchBulkhead::new(1);
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let b_run = b.clone();
        let task = tokio::spawn(async move {
            b_run
                .run(Some(Duration::from_millis(50)), move || {
                    let _ = release_rx.recv(); // park until released
                    0u8
                })
                .await
        });

        // After the deadline elapses the caller has TimedOut, but the
        // blocking thread still parks holding the permit.
        tokio::time::sleep(Duration::from_millis(150)).await;
        assert_eq!(
            b.available_permits(),
            0,
            "permit still held by the abandoned (still-running) blocking thread"
        );
        let out = task.await.expect("task joins");
        assert!(matches!(out, BulkheadOutcome::TimedOut));

        // Release the parked thread → permit returns to the pool.
        release_tx.send(()).expect("release parked thread");
        // Give the blocking thread a moment to drop its permit.
        for _ in 0..50 {
            if b.available_permits() == 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(
            b.available_permits(),
            1,
            "permit released once the abandoned thread finished"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn blocking_dispatch_does_not_starve_concurrent_dispatches() {
        // The #999 property (ADR-225 §1.2): a dispatch that BLOCKS (the
        // durable-write fsync wait) must not serialize / starve concurrent
        // dispatches. With capacity ≥ 2, a parked "writer" dispatch holds
        // ONE permit off-reactor while "reader" dispatches keep completing
        // promptly — they are NOT queued behind the writer.
        let b = DispatchBulkhead::new(4);

        // Writer: enters + parks (models a durable write blocked on fsync).
        let (release_writer_tx, release_writer_rx) = std::sync::mpsc::channel::<()>();
        let (writer_in_tx, writer_in_rx) = tokio::sync::oneshot::channel::<()>();
        let b_writer = b.clone();
        let writer = tokio::spawn(async move {
            b_writer
                .run(None, move || {
                    let _ = writer_in_tx.send(());
                    let _ = release_writer_rx.recv(); // park like a blocked fsync
                })
                .await
        });
        writer_in_rx
            .await
            .expect("writer entered its blocking closure");
        assert_eq!(
            b.available_permits(),
            3,
            "writer holds ONE permit; the other 3 stay free for readers"
        );

        // Readers: while the writer is parked, N read dispatches must all
        // complete promptly (they are not blocked behind the write).
        let started = Instant::now();
        let mut reader_results = Vec::new();
        for i in 0..12u32 {
            let out = b.run(None, move || i * 2).await;
            match out {
                BulkheadOutcome::Completed(v) => reader_results.push(v),
                other => panic!("reader {i} did not complete: {other:?}"),
            }
        }
        let readers_elapsed = started.elapsed();

        // The writer is STILL parked (never released) — yet all readers
        // finished. That is the un-starvation guarantee.
        assert_eq!(reader_results.len(), 12, "all readers completed");
        assert_eq!(
            reader_results,
            (0..12u32).map(|i| i * 2).collect::<Vec<_>>()
        );
        assert!(
            readers_elapsed < Duration::from_secs(2),
            "readers completed while the write was still blocked (took {readers_elapsed:?})"
        );

        // Clean up the parked writer.
        release_writer_tx.send(()).expect("release writer");
        let _ = writer.await.expect("writer joins");
    }

    /// The #999 OFF-REACTOR property — the *discriminating* regression
    /// guard.
    ///
    /// **This test is RED under an inline-dispatch revert** — it guards the
    /// OFF-REACTOR property (#999), unlike the capacity-only variant
    /// [`blocking_dispatch_does_not_starve_concurrent_dispatches`] above.
    /// That capacity test awaits its readers *sequentially on the test-body
    /// thread* with only one parked writer, so its readers never contend for
    /// a pinned reactor worker; it therefore stays GREEN even if `run()`
    /// reverts to inline dispatch, proving only semaphore-capacity
    /// un-starvation — NOT the off-reactor property #999 is about.
    ///
    /// The discriminator (per ADR-225 §1.2 + the R1 adversarial repro):
    /// `worker_threads = 2` + **TWO** writers dispatched through the REAL
    /// bulkhead that pin **BOTH** reactor workers + a reader that is a
    /// *spawned task* (not awaited inline on the test body). The
    /// `#[tokio::test]` body runs on the `block_on` thread, so the two
    /// worker threads are exactly the reactor's pool.
    ///
    /// - **Inline revert** (`run()` calls `f()` on the caller's task): each
    ///   writer's `f()` blocks the worker it landed on → both reactor
    ///   workers pinned → the spawned reader is never polled → the
    ///   `reader_ran` assertion fails (RED).
    /// - **Bulkhead** (`run()` hands `f()` to `spawn_blocking`): the writers
    ///   block on blocking-pool threads, the reactor stays free, and the
    ///   spawned reader completes while both writers are still blocked
    ///   (GREEN).
    ///
    /// All cross-thread coordination uses std primitives (mpsc + `AtomicBool`
    /// + std sleep on the block_on thread) because under the inline pathology
    /// the tokio timer driver itself can be starved — which is exactly the
    /// failure mode this guards against.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn bulkhead_off_reactor_un_starves_reader_when_writers_pin_all_workers() {
        let b = DispatchBulkhead::new(4);
        let (w1_in_tx, w1_in_rx) = std::sync::mpsc::channel::<()>();
        let (w2_in_tx, w2_in_rx) = std::sync::mpsc::channel::<()>();
        let (rel_tx1, rel_rx1) = std::sync::mpsc::channel::<()>();
        let (rel_tx2, rel_rx2) = std::sync::mpsc::channel::<()>();

        // Two writers dispatched through the REAL bulkhead. Each parks
        // inside its blocking closure (models a durable write blocked on
        // fsync), holding a blocking-pool thread — NOT a reactor worker.
        let b1 = b.clone();
        let w1 = tokio::spawn(async move {
            b1.run(None, move || {
                w1_in_tx.send(()).unwrap();
                let _ = rel_rx1.recv(); // park like a blocked fsync
            })
            .await
        });
        let b2 = b.clone();
        let w2 = tokio::spawn(async move {
            b2.run(None, move || {
                w2_in_tx.send(()).unwrap();
                let _ = rel_rx2.recv(); // park like a blocked fsync
            })
            .await
        });

        // Wait (via std, on the block_on thread) until both writers are
        // inside their blocking sections. Under an inline revert both of the
        // two reactor workers would now be pinned.
        w1_in_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("writer 1 entered its blocking closure");
        w2_in_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("writer 2 entered its blocking closure");

        // Reader: a SPAWNED task dispatched through the SAME bulkhead while
        // both writers are blocked. This is the discriminator — it is NOT
        // awaited inline on the test body, so a free reactor worker must
        // poll it. Under inline dispatch both workers are pinned and the
        // reader is never polled → RED. With the bulkhead the writers are
        // off-reactor, so a worker is free to poll the reader → GREEN.
        let reader_ran = Arc::new(AtomicBool::new(false));
        let reader_ran_c = reader_ran.clone();
        let b3 = b.clone();
        let reader = tokio::spawn(async move {
            b3.run(None, move || {
                reader_ran_c.store(true, Ordering::SeqCst);
                42u32
            })
            .await
        });

        // Give the reader 500ms of real time (std sleep — no tokio timer,
        // which could itself be starved in the inline-revert case).
        std::thread::sleep(Duration::from_millis(500));
        assert!(
            reader_ran.load(Ordering::SeqCst),
            "OFF-REACTOR UN-STARVATION FAILED (#999): the spawned reader did \
             not complete while both writers pinned the pool — under an \
             inline-dispatch revert this is the expected RED failure"
        );
        let out = reader.await.expect("reader task joins");
        assert!(matches!(out, BulkheadOutcome::Completed(42)));

        // Clean up the parked writers.
        rel_tx1.send(()).expect("release writer 1");
        rel_tx2.send(()).expect("release writer 2");
        let _ = w1.await.expect("writer 1 joins");
        let _ = w2.await.expect("writer 2 joins");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn panicking_dispatch_returns_panicked_and_releases_permit() {
        let b = DispatchBulkhead::new(2);
        let out: BulkheadOutcome<u8> = b
            .run(None, || panic!("blowing up inside the blocking closure"))
            .await;
        assert!(matches!(out, BulkheadOutcome::Panicked));
        // The permit must still be released after a panic (it is dropped
        // as the panicking thread unwinds).
        for _ in 0..50 {
            if b.available_permits() == 2 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(b.available_permits(), 2, "permit released after a panic");
    }
}
