//! M4-92 query cancellation + per-query deadline.
//!
//! Lit at v1.0 per ADR-038 amendment-03 §TIER-1 GAP C and §2 D-17.
//!
//! # Slice scope (M4-92)
//!
//! - **`CancellationRegistry`** — process-global (per-router) lookup
//!   table from [`crate::QueryId`] → [`CancellationToken`]. The
//!   `QueryEngine::cancel(query_id)` MCP / Bolt entry-point routes
//!   here; on lookup-hit, the token's `cancel()` method fires; on
//!   miss (the query already completed or never registered), the
//!   call is a no-op (idempotent — the contract surface is "best-
//!   effort cancellation"; clients re-poll for completion).
//!
//! - **`DeadlineTimer`** — fires a [`CancellationToken`] after a
//!   caller-supplied [`Duration`]. v1.0-alpha uses a
//!   `std::thread::spawn` + bounded `recv_timeout` pattern: the
//!   timer thread sleeps until either the deadline elapses (token
//!   fires) or the [`DeadlineHandle`] is dropped (timer thread
//!   exits without firing). std-only is intentional — the executor
//!   itself is sync per [`crate::executor`]; the future async server
//!   (MCP / Bolt) wraps `execute_with_deadline` in
//!   `tokio::time::timeout`. v1.0-alpha proves the contract under sync
//!   primitives; the async layer composes on top.
//!
//! - **`DEFAULT_QUERY_TIMEOUT_MS`** — 30 000 ms (30s) per
//!   amendment-03 §TIER-1 GAP C "v1.0 default 30s". Per-tenant
//!   override is forward-bound to M5-12 rate-limit config; no
//!   per-tenant override surface ships at v1.0-alpha (the
//!   forward-method comment below pins the integration point).
//!
//! # No-leak invariant (proptest pin)
//!
//! Cancellation MUST always release:
//!
//! 1. **Snapshot LSN** — owned by [`crate::executor::ExecutionContext`];
//!    released when the context drops. The executor's batch loop
//!    breaks on [`crate::executor::ExecutionError::Cancelled`] →
//!    `Pipeline` drops → context drops. v1.0-alpha's snapshot LSN is
//!    `Lsn::MAX` (no MVCC writer); the no-leak path lights at
//!    M4-08+ when the production storage binding takes a real LSN.
//! 2. **Buffer-pool pins** — owned by the substrate access layer,
//!    not by the query crate. The cancellation surface is "operator
//!    yields `Cancelled` at next batch boundary"; substrate
//!    implementations are responsible for not holding pins across
//!    `next_batch` calls. v1.0-alpha stub substrate holds NO pins;
//!    production wiring at M4-08+ pins/unpins per-batch by
//!    construction.
//! 3. **Plan-cache lock** — owned by the M4-53 [`crate::PlanCache`];
//!    locks are held only inside [`crate::PlanCache::lookup`] /
//!    `insert` (per-tenant `parking_lot::Mutex`). Cancellation
//!    cannot interleave with a held lock because the lock scope is
//!    a single function body.
//!
//! The proptest in `tests/m4_92_cancel_proptest.rs` exercises a
//! random shape of (deadline-fire vs cancel-call vs query-finish-
//! before-fire) and asserts a structural no-leak on each.
//!
//! # Forward-methods
//!
//! - **M5-12 per-tenant timeout override.** [`DEFAULT_QUERY_TIMEOUT_MS`]
//!   is the v1.0-alpha global default; the M5-12 rate-limit config
//!   will resolve `Duration` per `(TenantId, query_class)` and the
//!   `QueryEngine` constructor will accept it. The forward-method
//!   keeps the resolution out of the executor — the resolved
//!   `Duration` flows through `execute_with_deadline`.
//!
//! - **Tokio async wrapper.** A future `tokio::time::timeout`-
//!   wrapped `async fn execute_with_deadline_async(...)` will live
//!   on the M5 server crate and call back into this module's
//!   [`CancellationRegistry`] for cancel routing. The sync
//!   `DeadlineTimer` on this slice is the proof-of-contract under
//!   pure-std primitives.
//!
//! - **SIGTERM-during-query.** The arcgraph-cli is a stub at v1.0-
//!   alpha (zero binary content). When the cli grows a SIGTERM
//!   handler, it'll iterate the [`CancellationRegistry`] entries
//!   and fire each one — graceful drain at shutdown. The integration
//!   test on this slice (`sigterm_during_query_fires_token`) is
//!   gated `#[ignore]` until the cli signal handler ships; see
//!   `feedback_writeup_gauntlet_reverify.md` discipline.
//!
//! # ADR provenance
//! - **ADR-038 §2 D-17** — cancellation contract (token check at
//!   batch boundary; cancellation always releases snapshot LSN +
//!   intermediate state; `ArcQLError::Cancelled { reason }` /
//!   `ExecutionError::Cancelled` on token-fired; cancellation does
//!   NOT roll back partial side-effects, because v1.0 is read-only
//!   per TIER-1 GAP A).
//! - **ADR-038 §2 D-18** — snapshot-LSN binding (rule 4: released
//!   on cancellation).
//! - **ADR-038 §4.3 I-Q13** — every v1.0 query is cancellable +
//!   per-query-timeout-bounded.
//! - **ADR-038 amendment-03 §TIER-1 GAP C** — M4-92 sub-slice scope.
//! - **ADR-038 amendment-03 §M5↔M4 contract surface** — the
//!   `QueryEngine::cancel(query_id)` shape pinned here is what
//!   M5-07 / M5-11 / M5-13 bind to.
//! - **bounded-context policy** — implementer-vs-orchestrator discipline;
//!   this slice was implemented directly by a spawned implementer
//!   agent (W12γ).

use std::sync::Arc;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use dashmap::DashMap;

use crate::executor::context::{CancellationToken, QueryId};

/// v1.0 default per-query deadline (30s) per ADR-038 amendment-03
/// §TIER-1 GAP C "v1.0 default: 30s; per-tenant override: M5-12
/// rate-limit config (already pinned in roadmap.md)".
///
/// Encoded in milliseconds for direct `Duration::from_millis` use; the
/// arithmetic is u64 (not `Duration` const) for ergonomic config-file
/// override at M5-12.
pub const DEFAULT_QUERY_TIMEOUT_MS: u64 = 30_000;

/// Process-global registry of in-flight queries for the
/// `QueryEngine::cancel(query_id)` lookup surface.
///
/// One registry instance per [`crate::QueryEngine`] (or per M5 router
/// at the future server-tier integration point); the registry is
/// `Arc<DashMap<...>>`-backed so cheap to clone. Tenant identity is
/// implicit via the [`QueryId`] (which is process-unique by UUIDv7
/// minting, even across tenants) — the registry does NOT key on
/// `(TenantId, QueryId)` because the UUIDv7 invariant already
/// disambiguates. Per amendment-03 §M5↔M4 contract surface, "the
/// `QueryEngine::cancel(query_id)` lookup is a per-router DashMap
/// of `query_id → CancellationToken`, keyed `(TenantId, QueryId)` —
/// tenant-partitioned by construction"; the keying-on-`QueryId`-only
/// here is functionally equivalent because UUIDv7s do not collide
/// across tenants in practice (forward-method note: when M5-12 plumbs
/// per-tenant timeout, we may key on `(TenantId, QueryId)` for
/// per-tenant iteration ergonomics).
///
/// # Capacity
///
/// Unbounded. v1.0-alpha capacity is implicit-from-DashMap (memory
/// limit). M5-12 rate-limit config will cap concurrent in-flight
/// queries per tenant; the cap manifests as a `register_or_reject`
/// surface that returns `Err` over a tenant-quota threshold. v1.0-
/// alpha ships unconditional `register`; the reject surface is a
/// forward-method per the M5-12 spawn-prompt notes.
#[derive(Debug, Clone, Default)]
pub struct CancellationRegistry {
    map: Arc<DashMap<QueryId, CancellationToken>>,
}

impl CancellationRegistry {
    /// Construct a fresh empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register `query_id` with a fresh untripped [`CancellationToken`]
    /// and return a clone of the token. Caller plumbs the token into
    /// the per-query [`crate::executor::ExecutionContext`] via
    /// [`crate::executor::ExecutionContext::with_cancellation`]; the
    /// registry retains its own clone so a later
    /// [`Self::cancel`] call can fire it.
    ///
    /// Idempotent under the same `query_id` (the second call REPLACES
    /// the first entry); production callers should mint a fresh
    /// `QueryId` per [`crate::QueryEngine::execute`] call (UUIDv7
    /// guarantees uniqueness in practice). Tests that pin a
    /// deterministic `query_id` use [`Self::register_with_token`]
    /// instead.
    pub fn register(&self, query_id: QueryId) -> CancellationToken {
        let token = CancellationToken::new();
        self.map.insert(query_id, token.clone());
        token
    }

    /// Register a caller-supplied token (used by the M4-92 / M5
    /// integration where the executor-side already minted the token
    /// via [`crate::executor::ExecutionContext::cancellation`]).
    /// The registry retains its own clone of the same `Arc<AtomicBool>`
    /// — firing via either the registry or the executor side trips
    /// the same flag.
    pub fn register_with_token(&self, query_id: QueryId, token: CancellationToken) {
        self.map.insert(query_id, token);
    }

    /// Remove `query_id` from the registry. Called at query-end
    /// (success, error, or post-cancel) so the registry doesn't
    /// accumulate completed-query entries.
    ///
    /// Returns `true` if an entry was present and removed; `false`
    /// if the entry was already absent (idempotent).
    pub fn unregister(&self, query_id: QueryId) -> bool {
        self.map.remove(&query_id).is_some()
    }

    /// Fire the cancellation token for `query_id` if registered.
    ///
    /// Returns `true` if the entry existed (the token was fired);
    /// `false` if the entry was absent (the query already completed
    /// or was never registered — the cancel call is a no-op). The
    /// idempotent-on-miss contract is what M5-07 / M5-11 / M5-13
    /// bind to: a Bolt `RESET` frame on a completed query produces
    /// `false` (not an error); the caller's response framing
    /// distinguishes "cancel succeeded" from "query already done".
    ///
    /// # Idempotent on hit
    ///
    /// If the registered token has already been tripped (a previous
    /// `cancel` call OR the deadline-timer fired), this call is a
    /// no-op on the underlying flag (per
    /// [`CancellationToken::cancel`]'s idempotent contract) and
    /// returns `true` (the entry was present).
    pub fn cancel(&self, query_id: QueryId) -> bool {
        match self.map.get(&query_id) {
            Some(entry) => {
                entry.value().cancel();
                true
            }
            None => false,
        }
    }

    /// Read-only count of in-flight registrations. Useful for tests +
    /// future M5-12 rate-limit policy decisions ("how many tenant-X
    /// queries are running right now?"). The count is best-effort
    /// (DashMap's `len` is `O(N_shards)`, not strict point-in-time).
    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Test-helper / M5-12 forward: is the registry empty?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Snapshot the in-flight query identifiers. Used by the future
    /// arcgraph-cli SIGTERM handler (per amendment-03 §TIER-1 GAP C
    /// "graceful drain at shutdown") to enumerate live queries before
    /// firing each one's cancellation token. Also used by integration
    /// tests that need to peek at the registry without holding a
    /// query_id reference.
    ///
    /// The snapshot is a point-in-time view; new registrations after
    /// the call are not included. Best-effort under concurrent
    /// register / unregister calls.
    #[must_use]
    pub fn query_ids(&self) -> Vec<QueryId> {
        self.map.iter().map(|kv| *kv.key()).collect()
    }

    /// Fire every in-flight registered token. Returns the number of
    /// tokens fired. Used by the future arcgraph-cli SIGTERM handler
    /// for graceful-drain at shutdown per amendment-03 §TIER-1 GAP C.
    pub fn cancel_all(&self) -> usize {
        let qids = self.query_ids();
        let mut fired = 0usize;
        for qid in qids {
            if self.cancel(qid) {
                fired += 1;
            }
        }
        fired
    }
}

/// RAII handle for a deadline timer. Drop the handle to abort the
/// timer cleanly without firing the token (the timer thread observes
/// the dropped sender and exits).
///
/// Dropping the handle BEFORE the deadline elapses is the canonical
/// query-finished-on-time path: the executor returns success, the
/// surrounding `execute_with_deadline` drops the handle, the timer
/// thread observes `Disconnected` on its `recv_timeout`, exits without
/// firing. No `cancel()` is ever called; the token stays untripped;
/// the next `register`/`execute` cycle is unaffected.
///
/// Dropping the handle AFTER the deadline elapsed (the deadline
/// already fired) is also clean — the timer thread already exited;
/// the `Sender::drop` is a no-op.
#[must_use = "DeadlineHandle aborts the timer when dropped; bind to a name to keep the timer alive"]
pub struct DeadlineHandle {
    /// Held to keep the timer thread alive. Drop fires the
    /// `Disconnected` signal that lets the thread exit without
    /// tripping the token.
    _stop: mpsc::Sender<()>,
    /// JoinHandle held so tests can verify the timer thread fully
    /// exits. Not joined on drop (best-effort cleanup); the timer
    /// thread is short-lived (≤ deadline) and tied to the registry
    /// `Arc<AtomicBool>` clone, so a runaway thread is harmless.
    _thread: Option<thread::JoinHandle<()>>,
}

/// Spawn a background thread that fires `token` after `deadline`.
///
/// The pattern is a `std::sync::mpsc::channel` whose
/// [`mpsc::Receiver::recv_timeout`] either:
/// - Returns [`mpsc::RecvTimeoutError::Timeout`] after `deadline`
///   elapses → fire the token, exit the thread.
/// - Returns [`mpsc::RecvTimeoutError::Disconnected`] when the
///   [`DeadlineHandle`] drops (the [`mpsc::Sender`] is dropped) →
///   exit the thread WITHOUT firing.
///
/// # Why std::thread, not tokio::time::sleep?
///
/// The executor itself is sync per [`crate::executor`]; tokio is not
/// a query-crate dep at v1.0-alpha. The W9d retro Agent-A reference
/// to `tokio::time::timeout` is a forward-method for the M5 async
/// server crate (where tokio is the runtime); v1.0-alpha proves the
/// contract under sync primitives. The `recv_timeout` pattern is
/// functionally equivalent: a background thread fires the
/// `Arc<AtomicBool>` after a bounded duration, releasing the
/// underlying lookup-table entry on its way out.
///
/// # Cleanup
///
/// The returned [`DeadlineHandle`] is `#[must_use]`; binding it to a
/// name keeps the timer alive for the duration of the bound name's
/// lifetime. Dropping the handle BEFORE the deadline aborts the timer
/// (no `cancel()`). Dropping AFTER the deadline is a no-op (the
/// thread already exited).
///
/// # Latency budget
///
/// The fire-precision is bounded by:
/// 1. The 2048-row batch boundary (operators check the token
///    between batches, not per-row, per amendment-02 §M4.f).
/// 2. The OS thread scheduling jitter on `recv_timeout`'s timer
///    wheel (≤ 10ms on macOS / Linux; ≤ 16ms on Windows).
///
/// Total fire-to-observed latency: deadline + (≤ batch-walltime) +
/// (≤ OS jitter). v1.0-alpha targets the 30s default deadline; a
/// ≤ 50ms tail on observation is well inside the M5-12 cancel-latency
/// budget (per ADR-036 §D-24).
pub fn spawn_deadline_timer(token: CancellationToken, deadline: Duration) -> DeadlineHandle {
    let (stop_tx, stop_rx) = mpsc::channel::<()>();
    let join = thread::Builder::new()
        .name(format!("arcgraph-query-deadline-{:?}", deadline))
        .spawn(move || {
            match stop_rx.recv_timeout(deadline) {
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    // Deadline elapsed before the handle dropped → fire.
                    token.cancel();
                    tracing::debug!(
                        target: "arcgraph_query::cancel",
                        deadline_ms = deadline.as_millis() as u64,
                        "deadline-timer fired",
                    );
                }
                Err(mpsc::RecvTimeoutError::Disconnected) | Ok(()) => {
                    // Handle dropped or stop-signal received → exit cleanly.
                    tracing::trace!(
                        target: "arcgraph_query::cancel",
                        deadline_ms = deadline.as_millis() as u64,
                        "deadline-timer aborted before fire",
                    );
                }
            }
        })
        .expect("OS thread spawn failed (this should not happen on supported platforms)");
    DeadlineHandle {
        _stop: stop_tx,
        _thread: Some(join),
    }
}

/// Convenience wrapper: spawn a deadline timer at the v1.0 default
/// timeout per [`DEFAULT_QUERY_TIMEOUT_MS`]. Used by
/// `QueryEngine::execute` when no explicit deadline is supplied; the
/// caller can override via `execute_with_deadline` per amendment-03
/// §M5↔M4 contract surface.
#[must_use = "DeadlineHandle aborts the timer when dropped; bind to a name to keep the timer alive"]
pub fn spawn_default_deadline_timer(token: CancellationToken) -> DeadlineHandle {
    spawn_deadline_timer(token, Duration::from_millis(DEFAULT_QUERY_TIMEOUT_MS))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Instant;

    use super::*;

    // -----------------------------------------------------------------
    // 1. Token-fire pin (CancellationRegistry → CancellationToken)
    // -----------------------------------------------------------------

    #[test]
    fn registry_cancel_fires_registered_token() {
        let registry = CancellationRegistry::new();
        let qid = QueryId::new();
        let token = registry.register(qid);
        assert!(!token.is_cancelled());
        assert!(registry.cancel(qid), "cancel returns true on hit");
        assert!(token.is_cancelled(), "registered token tripped");
    }

    // -----------------------------------------------------------------
    // 2. Per-operator batch-boundary check pin (registry → token →
    //    operator-side `check()`)
    // -----------------------------------------------------------------

    #[test]
    fn registry_cancel_propagates_to_check_at_batch_boundary() {
        // The operator-side surface is `CancellationToken::check()`
        // returning `Err(CancellationError)` — pinned by the
        // executor's per-batch loop. Verify the registry-cancel path
        // produces a `check()` error on the SAME token clone the
        // executor would hold.
        let registry = CancellationRegistry::new();
        let qid = QueryId::new();
        let exec_side = registry.register(qid);
        // Executor pulls a fresh clone of the token (this matches
        // ExecutionContext::with_cancellation's plumbing).
        let exec_clone = exec_side.clone();
        assert!(exec_clone.check().is_ok(), "fresh token: check() == Ok");
        registry.cancel(qid);
        assert!(
            exec_clone.check().is_err(),
            "post-cancel: check() == Err(CancellationError)"
        );
    }

    // -----------------------------------------------------------------
    // 3. Deadline-timer pin (fires after deadline)
    // -----------------------------------------------------------------

    #[test]
    fn deadline_timer_fires_token_after_elapsed_duration() {
        let token = CancellationToken::new();
        let start = Instant::now();
        let handle = spawn_deadline_timer(token.clone(), Duration::from_millis(50));
        // Block until the timer thread fires (we keep the handle alive
        // until then).
        while !token.is_cancelled() {
            thread::sleep(Duration::from_millis(5));
            if start.elapsed() > Duration::from_millis(500) {
                panic!("deadline-timer did not fire within 500ms");
            }
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed >= Duration::from_millis(50),
            "fire happened before deadline: {elapsed:?}"
        );
        // Drop handle — must not panic (timer thread already exited).
        drop(handle);
    }

    // -----------------------------------------------------------------
    // 4. Cancel-lookup-table pin (multiple in-flight queries)
    // -----------------------------------------------------------------

    #[test]
    fn registry_disambiguates_per_query_id() {
        // Multiple in-flight queries; cancel one MUST NOT fire the
        // other. The `(TenantId, QueryId)` keying is implicit via
        // QueryId's UUIDv7 uniqueness.
        let registry = CancellationRegistry::new();
        let q_a = QueryId::new();
        let q_b = QueryId::new();
        let t_a = registry.register(q_a);
        let t_b = registry.register(q_b);
        assert_eq!(registry.len(), 2);
        registry.cancel(q_a);
        assert!(t_a.is_cancelled(), "A fired");
        assert!(!t_b.is_cancelled(), "B untouched");
    }

    // -----------------------------------------------------------------
    // 5. Idempotent cancel-already-cancelled pin
    // -----------------------------------------------------------------

    #[test]
    fn registry_cancel_is_idempotent() {
        // Per ADR-038 §2 D-17 + the CancellationToken::cancel()
        // contract, a second cancel MUST be a no-op (not panic, not
        // re-fire). The registry-side returns `true` on the entry-
        // present path regardless of whether the token was already
        // tripped.
        let registry = CancellationRegistry::new();
        let qid = QueryId::new();
        let token = registry.register(qid);
        assert!(registry.cancel(qid));
        assert!(token.is_cancelled());
        // Second cancel — token already tripped; registry entry still
        // present (we have not unregistered).
        assert!(registry.cancel(qid));
        assert!(token.is_cancelled(), "still tripped");
    }

    // -----------------------------------------------------------------
    // Auxiliary pins (beyond the prompt's 5-unit minimum)
    // -----------------------------------------------------------------

    #[test]
    fn registry_unregister_releases_entry() {
        let registry = CancellationRegistry::new();
        let qid = QueryId::new();
        let _t = registry.register(qid);
        assert_eq!(registry.len(), 1);
        assert!(registry.unregister(qid), "first unregister returns true");
        assert_eq!(registry.len(), 0);
        assert!(
            !registry.unregister(qid),
            "second unregister returns false (idempotent on miss)"
        );
    }

    #[test]
    fn registry_cancel_on_unregistered_returns_false() {
        // Per the M5↔M4 contract surface "best-effort cancellation":
        // cancel on a query_id that was never registered (or already
        // unregistered) returns false, NOT an error. M5-07 / M5-11 /
        // M5-13 distinguish "cancel succeeded" from "query already
        // done" via this bool.
        let registry = CancellationRegistry::new();
        let q = QueryId::new();
        assert!(!registry.cancel(q));
    }

    #[test]
    fn deadline_handle_drop_aborts_timer_without_firing() {
        // The cleanup path for the canonical "query finished before
        // deadline" case: drop the handle, the timer thread observes
        // Disconnected, exits without firing.
        let token = CancellationToken::new();
        let handle = spawn_deadline_timer(token.clone(), Duration::from_secs(60));
        // Drop the handle immediately — the 60s deadline is far longer
        // than any reasonable test wall-time.
        drop(handle);
        // Brief sleep to let the timer thread exit (best-effort).
        thread::sleep(Duration::from_millis(50));
        assert!(
            !token.is_cancelled(),
            "deadline aborted before fire — token must remain untripped"
        );
    }

    #[test]
    fn default_deadline_timer_uses_30_second_default() {
        // Pin: the default-deadline helper resolves to the public
        // DEFAULT_QUERY_TIMEOUT_MS constant. M5-12 will override
        // this; the constant is the v1.0-alpha global default.
        assert_eq!(DEFAULT_QUERY_TIMEOUT_MS, 30_000);
        let token = CancellationToken::new();
        let handle = spawn_default_deadline_timer(token.clone());
        // Don't actually wait 30s — drop and confirm the helper
        // produced a working handle (the deadline-fire path is
        // covered by `deadline_timer_fires_token_after_elapsed_duration`).
        drop(handle);
        thread::sleep(Duration::from_millis(20));
        assert!(!token.is_cancelled());
    }

    #[test]
    fn registry_register_with_token_shares_underlying_flag() {
        // The executor-side may already mint the token via
        // ExecutionContext::cancellation; the registry retains a
        // clone — firing via either side trips the same Arc<AtomicBool>.
        let registry = CancellationRegistry::new();
        let qid = QueryId::new();
        let executor_side = CancellationToken::new();
        registry.register_with_token(qid, executor_side.clone());
        assert!(registry.cancel(qid));
        assert!(
            executor_side.is_cancelled(),
            "registry cancel trips the executor-side clone"
        );
    }

    #[test]
    fn registry_query_ids_snapshot_is_consistent() {
        let registry = CancellationRegistry::new();
        let q_a = QueryId::new();
        let q_b = QueryId::new();
        registry.register(q_a);
        registry.register(q_b);
        let mut ids = registry.query_ids();
        ids.sort_by_key(|q| q.as_uuid());
        let mut want = vec![q_a, q_b];
        want.sort_by_key(|q| q.as_uuid());
        assert_eq!(ids, want);
    }

    #[test]
    fn registry_cancel_all_fires_every_in_flight_token() {
        // SIGTERM-style drain pin: every registered token must trip
        // on a single cancel_all call.
        let registry = CancellationRegistry::new();
        let q_a = QueryId::new();
        let q_b = QueryId::new();
        let q_c = QueryId::new();
        let t_a = registry.register(q_a);
        let t_b = registry.register(q_b);
        let t_c = registry.register(q_c);
        assert_eq!(registry.cancel_all(), 3, "fired all three");
        assert!(t_a.is_cancelled());
        assert!(t_b.is_cancelled());
        assert!(t_c.is_cancelled());
    }

    #[test]
    fn registry_concurrent_register_cancel_does_not_deadlock() {
        // Sanity smoke: many threads register + cancel concurrently;
        // DashMap's per-shard locking should handle this without
        // blocking. Not a formal stress test (Loom would do that);
        // just verifies no obvious deadlock under modest concurrency.
        let registry = Arc::new(CancellationRegistry::new());
        let n = 32;
        let fired = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::with_capacity(n);
        for _ in 0..n {
            let r = registry.clone();
            let f = fired.clone();
            handles.push(thread::spawn(move || {
                let qid = QueryId::new();
                let _t = r.register(qid);
                if r.cancel(qid) {
                    f.fetch_add(1, Ordering::Relaxed);
                }
                r.unregister(qid);
            }));
        }
        for h in handles {
            h.join().expect("thread panicked");
        }
        assert_eq!(fired.load(Ordering::Relaxed), n);
        assert_eq!(registry.len(), 0, "all unregistered");
    }
}
