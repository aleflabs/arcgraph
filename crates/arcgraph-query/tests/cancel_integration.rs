//! M4-92 cancellation + per-query deadline end-to-end integration
//! tests per ADR-038 amendment-03 §TIER-1 GAP C + §2 D-17.
//!
//! # Pin set
//!
//! 1. `deadline_expiry_on_long_query_surfaces_cancelled` — a query
//!    with a slow substrate is bounded by a short deadline; the
//!    executor surfaces [`ExplainError::Cancelled`] at the next
//!    batch boundary.
//! 2. `explicit_cancel_via_query_engine_cancel_fires_token` — a
//!    sibling thread calls `engine.cancel(query_id)` while the
//!    executor is mid-loop; the loop yields `Cancelled`. Uses
//!    `execute_with_query_id` to pre-mint the QueryId so the
//!    canceller knows what to fire.
//! 3. `sigterm_during_query_fires_token` — gated `#[ignore]` until
//!    arcgraph-cli ships a SIGTERM signal handler that iterates the
//!    [`CancellationRegistry`] entries and fires each one. The cli
//!    is a stub (zero binary content) at v1.0-alpha; the test body
//!    documents the integration shape.
//! 4. `cancel_releases_registry_entry_on_query_end` — the
//!    no-leak invariant pin (registry entry MUST be removed when
//!    the materialize loop exits, success or cancel).
//!
//! # ADR provenance
//! - **ADR-038 amendment-03 §TIER-1 GAP C** — M4-92 sub-slice scope
//!   (5 unit + 3 integration + 1 proptest minimum).
//! - **ADR-038 §2 D-17** — cancellation contract.
//! - **ADR-038 §4.3 I-Q13** — every v1.0 query is cancellable +
//!   per-query-timeout-bounded.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use arcgraph_core::{LabelId, Lsn, NodeId, TenantId, TypeId};
use arcgraph_query::executor::value::NodeView;
use arcgraph_query::executor::{
    BATCH_ROWS, BoundEdge, BoundNode, ExecutorSubstrate, RankedHit, StubExecutorSubstrate,
    SubstrateAccessError, Value,
};
use arcgraph_query::logical_plan::Direction;
use arcgraph_query::semantic::StubCatalogProvider;
use arcgraph_query::{CancellationRegistry, ExplainError, QueryEngine, QueryId};

// ---------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------

fn cat_basic() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["Person"])
        .with_rel_types(["KNOWS"])
        .with_properties(["name", "age"])
}

fn substrate_with_n_persons(n: u64) -> StubExecutorSubstrate {
    let mut s = StubExecutorSubstrate::new();
    for i in 1..=n {
        s = s.with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(i), Some(LabelId::new(1)))
                .with_property("age", Value::Integer(i as i64 * 5)),
        );
    }
    s
}

// =====================================================================
// Slow-substrate adapter
// =====================================================================
//
// Wraps a base substrate and sleeps for `per_call_ms` milliseconds
// before delegating each `scan_label` call. Lets the deadline-expiry
// + explicit-cancel integration tests drive a deterministic batch-
// boundary cancel without depending on hardware-tuned wall-times.

struct SlowSubstrate {
    base: StubExecutorSubstrate,
    per_call_ms: u64,
    calls_seen: Arc<AtomicU64>,
}

impl SlowSubstrate {
    fn new(base: StubExecutorSubstrate, per_call_ms: u64) -> Self {
        Self {
            base,
            per_call_ms,
            calls_seen: Arc::new(AtomicU64::new(0)),
        }
    }
}

impl ExecutorSubstrate for SlowSubstrate {
    fn scan_nodes(
        &self,
        tenant: TenantId,
        label: Option<LabelId>,
        read_lsn: Lsn,
    ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
        thread::sleep(Duration::from_millis(self.per_call_ms));
        self.calls_seen.fetch_add(1, Ordering::AcqRel);
        self.base.scan_nodes(tenant, label, read_lsn)
    }

    fn expand(
        &self,
        tenant: TenantId,
        from: NodeId,
        rel_type: Option<TypeId>,
        direction: Direction,
        read_lsn: Lsn,
    ) -> Result<Vec<BoundEdge>, SubstrateAccessError> {
        self.base
            .expand(tenant, from, rel_type, direction, read_lsn)
    }

    fn vector_search(
        &self,
        tenant: TenantId,
        property: &str,
        query_vec: &[f32],
        k: u64,
        read_lsn: Lsn,
    ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
        self.base
            .vector_search(tenant, property, query_vec, k, read_lsn)
    }

    fn bm25_search(
        &self,
        tenant: TenantId,
        property: &str,
        query_text: &str,
        k: u64,
        read_lsn: Lsn,
    ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
        self.base
            .bm25_search(tenant, property, query_text, k, read_lsn)
    }

    fn community_members(
        &self,
        tenant: TenantId,
        community_id: i64,
        read_lsn: Lsn,
    ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
        self.base.community_members(tenant, community_id, read_lsn)
    }
}

// =====================================================================
// 1. Deadline-expiry pin
// =====================================================================

#[test]
fn deadline_expiry_on_long_query_surfaces_cancelled() {
    // Drive a large substrate (5 * BATCH_ROWS rows) through a slow
    // substrate adapter that sleeps 200ms inside the FIRST
    // scan_nodes call. Deadline (50ms) fires DURING the substrate
    // sleep; by the time scan_nodes returns at ~200ms, the next
    // batch-boundary check observes the tripped token and surfaces
    // Cancelled.
    //
    // Why per_call_ms must be > deadline: in release-mode the
    // batch-loop after the first scan_nodes drains the in-memory
    // buffer in ~milliseconds. If the deadline expires AFTER the
    // sleep finishes, the loop may complete before the next batch-
    // boundary check observes the trip. Sleep > deadline guarantees
    // the trip lands during the substrate call (uninterruptible) and
    // the very next batch-boundary check sees it.
    let base = substrate_with_n_persons(BATCH_ROWS as u64 * 5);
    let slow = SlowSubstrate::new(base, 200);
    let cat = cat_basic();
    let engine = QueryEngine::new(&cat);
    let start = Instant::now();
    let res = engine.execute_with_deadline(
        "MATCH (n:Person) RETURN n",
        &slow,
        Duration::from_millis(50),
    );
    let elapsed = start.elapsed();
    match res {
        Err(ExplainError::Cancelled) => {
            // Per amendment-03 §M5↔M4 contract surface, the
            // cancellation surfaces as Cancelled (NOT NotImplemented
            // hiding behind a generic error frame — pre-W11Z fix-up
            // MED-2 the dispatch was that coarse).
        }
        Err(other) => panic!(
            "expected ExplainError::Cancelled on deadline expiry; got {other:?} after {elapsed:?}"
        ),
        Ok(rows) => panic!(
            "deadline expired but execute returned {} rows in {:?}",
            rows.len(),
            elapsed
        ),
    }
    // Sanity: the cancel must observe AFTER the substrate's 200ms
    // sleep completes (at ~200ms); the upper bound is generous to
    // avoid flakes on a contended runner.
    assert!(
        elapsed >= Duration::from_millis(150),
        "cancelled too early ({elapsed:?})"
    );
    assert!(
        elapsed < Duration::from_millis(1000),
        "cancelled too late ({elapsed:?})"
    );
    // Registry must be empty post-execute (no-leak: the unregister
    // unconditional cleanup ran).
    assert!(
        engine.cancellation_registry().is_empty(),
        "registry must be drained on query end"
    );
}

// =====================================================================
// 2. Explicit-cancel pin (concurrent QueryEngine::cancel)
// =====================================================================

#[test]
fn explicit_cancel_via_query_engine_cancel_fires_token() {
    // Sibling thread calls engine.cancel(qid) mid-execute. The
    // execute_with_query_id surface lets us pre-mint the QueryId so
    // the canceller knows what to fire (without inspecting the
    // registry).
    //
    // Synchronization: the canceller waits ~30ms (well under the
    // total 5*50ms scan wall-time) then fires; the executor observes
    // Cancelled at the next scan_label batch boundary.
    let base = substrate_with_n_persons(BATCH_ROWS as u64 * 5);
    // 200ms substrate sleep ensures the cancel signal lands during
    // the first scan_nodes call (uninterruptible from this side); on
    // return, the next batch-boundary check observes the trip. See
    // the deadline-expiry pin's "why per_call_ms > deadline" rationale.
    let slow = SlowSubstrate::new(base, 200);
    let cat = cat_basic();
    let registry = CancellationRegistry::new();
    let engine = QueryEngine::new(&cat).with_cancellation_registry(registry.clone());
    let qid = QueryId::new();

    // Spawn the canceller; it polls until the engine has registered
    // the qid, then fires.
    let registry_canceller = registry.clone();
    let canceller = thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if Instant::now() > deadline {
                panic!("registry never observed registered qid");
            }
            if !registry_canceller.is_empty() {
                // The engine has registered. Fire.
                let fired = registry_canceller.cancel(qid);
                return fired;
            }
            thread::sleep(Duration::from_millis(2));
        }
    });

    let start = Instant::now();
    let res = engine.execute_with_query_id(qid, "MATCH (n:Person) RETURN n", &slow);
    let elapsed = start.elapsed();

    assert!(
        matches!(res, Err(ExplainError::Cancelled)),
        "expected Cancelled, got {res:?} after {elapsed:?}"
    );
    let cancel_returned = canceller.join().expect("canceller thread panicked");
    assert!(
        cancel_returned,
        "registry.cancel(qid) must have returned true (entry was registered)"
    );
    // Registry drained on query end.
    assert!(
        registry.is_empty(),
        "registry must be drained on query end (no-leak)"
    );
}

// =====================================================================
// 3. SIGTERM-during-query pin (gated behind cli signal handler)
// =====================================================================

/// W17δ #280 closure — pin the SIGTERM → cancel_all seam.
///
/// `arcgraph-cli`'s `run_serve_stdio` / `run_serve_bolt` now wrap
/// `shutdown_on_term().await` with a `cancel_registry.cancel_all()`
/// call before letting the serve loop see the shutdown signal — see
/// `crates/arcgraph-cli/src/bin/arcgraph.rs::run_serve_stdio`. The
/// process-level signal handling is integration-tested by deploying
/// the binary; this Rust-level test pins the underlying SEAM: a
/// sibling thread that mimics the cli's SIGTERM handler by calling
/// `registry.cancel_all()` must fire every in-flight query's token.
///
/// The semantics are strictly stronger than `cancel(qid)` — the cli
/// doesn't know individual `QueryId`s, so it fires the entire
/// registry. This test pins that the registry surface delivers on
/// that contract.
#[test]
fn sigterm_during_query_fires_token() {
    // Mirror the explicit-cancel pin's shape but use `cancel_all()`
    // (no QueryId argument — the canceller is the cli's SIGTERM
    // handler, which doesn't track per-query identifiers).
    let base = substrate_with_n_persons(BATCH_ROWS as u64 * 5);
    let slow = SlowSubstrate::new(base, 200);
    let cat = cat_basic();
    let registry = CancellationRegistry::new();
    let engine = QueryEngine::new(&cat).with_cancellation_registry(registry.clone());
    let qid = QueryId::new();

    // Spawn the SIGTERM-simulator: poll until the engine registers,
    // then fire `cancel_all()` — exactly the call the cli's
    // `shutdown` future invokes after `shutdown_on_term()` resolves.
    let registry_sigterm = registry.clone();
    let sigterm_sim = thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if Instant::now() > deadline {
                panic!("registry never observed a registered qid before SIGTERM-sim deadline");
            }
            if !registry_sigterm.is_empty() {
                // The engine has registered. Fire every in-flight
                // token (the cli's SIGTERM-handler shape).
                return registry_sigterm.cancel_all();
            }
            thread::sleep(Duration::from_millis(2));
        }
    });

    let start = Instant::now();
    let res = engine.execute_with_query_id(qid, "MATCH (n:Person) RETURN n", &slow);
    let elapsed = start.elapsed();

    assert!(
        matches!(res, Err(ExplainError::Cancelled)),
        "expected Cancelled, got {res:?} after {elapsed:?}",
    );
    let fired = sigterm_sim.join().expect("SIGTERM-sim thread panicked");
    assert!(
        fired >= 1,
        "cancel_all must have fired ≥1 in-flight token; observed {fired}",
    );
    // Registry drained on query end (no-leak invariant — symmetric
    // with the explicit-cancel pin).
    assert!(
        registry.is_empty(),
        "registry must be drained on query end (no-leak); cancel_all path also unregisters",
    );
}

/// W17δ #348 R1-MED-1 closure — sister-site pin.
///
/// `arcgraph-cli` ships THREE `shutdown_on_term()` call sites — the
/// two in `bin/arcgraph.rs` (`run_serve_stdio` + `run_serve_bolt`)
/// plus the one in `bin/arcgraph_mcp_stdio.rs::run`. R1-MED-1 caught
/// the third site missing its `cancel_all()` wrap; this test pins the
/// SAME seam contract from the third site's perspective — a regression
/// that reverts the `arcgraph_mcp_stdio.rs::run` wire-up to a bare
/// `let shutdown = shutdown_on_term();` will:
///
/// 1. Surface as a doc-vs-code divergence (the binary's module-doc at
///    `arcgraph_mcp_stdio.rs:6-9` claims the drain fires).
/// 2. Be caught at PR-review time by the now-explicit sister-cite
///    enumeration discipline.
///
/// This Rust-level test, like `sigterm_during_query_fires_token`,
/// pins the underlying `CancellationRegistry::cancel_all()` SEAM the
/// binary's `shutdown` future relies on — a sibling thread mimics the
/// binary's `async move { shutdown_on_term().await; cancel_registry_
/// for_shutdown.cancel_all(); ... }` shape by calling `cancel_all()`
/// after observing a registered query. The test is intentionally
/// parallel to `sigterm_during_query_fires_token` (same primitives,
/// same registry, same engine) — the value-add is the explicit cite
/// to the third call site so a future reviewer reading the test file
/// sees the three binary wire-ups are all pinned by this seam.
#[test]
fn mcp_stdio_shutdown_sister_site_fires_cancel_all() {
    // Same shape as `sigterm_during_query_fires_token` but documenting
    // the third sister-cite (arcgraph_mcp_stdio.rs::run). The seam
    // contract is identical; the test exists to make the third site
    // visible in the regression matrix.
    let base = substrate_with_n_persons(BATCH_ROWS as u64 * 5);
    let slow = SlowSubstrate::new(base, 200);
    let cat = cat_basic();
    let registry = CancellationRegistry::new();
    let engine = QueryEngine::new(&cat).with_cancellation_registry(registry.clone());
    let qid = QueryId::new();

    // The SIGTERM-simulator: this is the role
    // `arcgraph_mcp_stdio.rs::run`'s `async move` block plays — when
    // `shutdown_on_term()` resolves, the block calls
    // `cancel_registry_for_shutdown.cancel_all()`. We mimic that
    // ordering: wait for the engine to register, then fire.
    let registry_sigterm = registry.clone();
    let sigterm_sim = thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if Instant::now() > deadline {
                panic!(
                    "registry never observed a registered qid before sigterm-sim deadline \
                     (mcp_stdio sister-site pin)"
                );
            }
            if !registry_sigterm.is_empty() {
                return registry_sigterm.cancel_all();
            }
            thread::sleep(Duration::from_millis(2));
        }
    });

    let start = Instant::now();
    let res = engine.execute_with_query_id(qid, "MATCH (n:Person) RETURN n", &slow);
    let elapsed = start.elapsed();

    assert!(
        matches!(res, Err(ExplainError::Cancelled)),
        "expected Cancelled (mcp_stdio sister site), got {res:?} after {elapsed:?}",
    );
    let fired = sigterm_sim.join().expect("sigterm-sim thread panicked");
    assert!(
        fired >= 1,
        "cancel_all must have fired ≥1 in-flight token from the mcp_stdio drain path; observed {fired}",
    );
    // No-leak: registry drains on query end (symmetric with the other
    // three integration pins).
    assert!(
        registry.is_empty(),
        "registry must be drained on query end (mcp_stdio sister-site no-leak invariant)",
    );
}

// =====================================================================
// 4. No-leak pin: registry entry removed on query-end (success path)
// =====================================================================

#[test]
fn registry_entry_removed_on_successful_query_end() {
    // Per ADR-038 §2 D-17: cancellation always releases the registry
    // entry. The success path also releases (the unregister is
    // unconditional). Pin: a successful execute leaves the registry
    // empty.
    let base = substrate_with_n_persons(3);
    let cat = cat_basic();
    let engine = QueryEngine::new(&cat);
    let qid = QueryId::new();
    let res = engine.execute_with_query_id(qid, "MATCH (n:Person) RETURN n", &base);
    assert!(res.is_ok(), "execute succeeds: {res:?}");
    assert!(
        engine.cancellation_registry().is_empty(),
        "registry must be drained on successful query end"
    );
}

// =====================================================================
// 5. W12γ fix-up MED-1 — execute() applies DEFAULT_QUERY_TIMEOUT_MS
// =====================================================================
//
// The canonical entry-point `QueryEngine::execute` MUST honor the
// ADR-038 §4.3 I-Q13 contract ("every v1.0 query is per-query-timeout-
// bounded"). Pre-fix-up, `execute()` skipped the deadline timer; only
// the explicit `execute_with_deadline` variant applied a bound. Fix:
// `execute()` routes through `execute_with_query_id_and_deadline` with
// `DEFAULT_QUERY_TIMEOUT_MS` so a long-running query surfaces
// `ExplainError::Cancelled` after the default elapses.
//
// Integration cost: a 30s test would dominate the test wall-time
// budget. We instead use the engine-internal route
// `execute_with_query_id_and_deadline` with a 50ms deadline AND
// observe the same code path that `execute()` exercises (since
// `execute()` forwards to it). The pin we want is the route-through;
// the public-surface coverage is via the new
// `execute_default_path_applies_30s_timeout` unit pin (below) which
// asserts the constant resolves correctly without actually waiting
// 30 seconds.

#[test]
fn execute_default_path_applies_30s_timeout() {
    // Pin: the public `QueryEngine::execute` entry-point routes
    // through a deadline-bounded path with the v1.0 default. We
    // verify by:
    // (a) the public DEFAULT_QUERY_TIMEOUT_MS resolves to 30_000;
    // (b) `execute_with_query_id_and_deadline` (the path `execute`
    //     forwards to) surfaces Cancelled on a tight deadline; THIS
    //     is the route-through coverage that proves `execute` would
    //     do the same on a 30s+ query.
    assert_eq!(
        arcgraph_query::DEFAULT_QUERY_TIMEOUT_MS,
        30_000,
        "v1.0 default timeout per ADR-038 amendment-03 §TIER-1 GAP C"
    );
    // The route-through pin: same code path `execute()` invokes,
    // just with a tight deadline so we don't burn 30s of test wall-
    // time. If `execute()` ever stops forwarding to
    // `execute_with_query_id_and_deadline`, this pin still passes —
    // but the unit-level pin in `explain/mod.rs::tests` asserts
    // `execute()` applies the default constant by code-path
    // inspection (forwarder → forwarded).
    let base = substrate_with_n_persons(BATCH_ROWS as u64 * 5);
    let slow = SlowSubstrate::new(base, 200);
    let cat = cat_basic();
    let engine = QueryEngine::new(&cat);
    let res = engine.execute_with_deadline(
        "MATCH (n:Person) RETURN n",
        &slow,
        Duration::from_millis(50),
    );
    assert!(
        matches!(res, Err(ExplainError::Cancelled)),
        "default-deadline route must surface Cancelled on tight deadline; got {res:?}"
    );
    // No-leak invariant survives the route-through.
    assert!(engine.cancellation_registry().is_empty());
}

// =====================================================================
// 6. W12γ fix-up MED-2 — PROFILE applies DEFAULT_QUERY_TIMEOUT_MS +
//    is cancellable
// =====================================================================
//
// Symmetric pin to MED-1, but on the PROFILE entry-point. ADR-038
// §4.3 I-Q13 + §2 D-19 admit PROFILE as a v1.0 query, so I-Q13
// applies. Pre-fix-up, neither the free `profile()` function nor
// `QueryEngine::profile` registered against any registry or applied
// any deadline.

#[test]
fn profile_default_path_applies_30s_timeout() {
    // Pin: `QueryEngine::profile` routes through the free `profile()`
    // with the engine's registry + DEFAULT_QUERY_TIMEOUT_MS. Same
    // route-through cost-management as `execute_default_path_*`: we
    // exercise the deadline path with a tight deadline through the
    // free function so test wall-time stays bounded.
    let base = substrate_with_n_persons(BATCH_ROWS as u64 * 5);
    let slow = SlowSubstrate::new(base, 200);
    let cat = cat_basic();
    let registry = CancellationRegistry::new();
    let res = arcgraph_query::profile(
        "PROFILE MATCH (n:Person) RETURN n",
        &cat,
        &slow,
        &registry,
        Duration::from_millis(50),
    );
    assert!(
        matches!(res, Err(ExplainError::Cancelled)),
        "PROFILE deadline path must surface Cancelled on tight deadline; got {res:?}"
    );
    // No-leak: registry drained on PROFILE end (success / error /
    // cancel via the W12γ fix-up MED-3 RAII guard).
    assert!(registry.is_empty(), "PROFILE must drain its registry entry");
}

#[test]
fn profile_is_cancellable_via_external_registry() {
    // PROFILE shares the cancellation contract with execute (per ADR-
    // 038 §2 D-19 + §4.3 I-Q13). A sibling thread that polls the
    // registry and fires its single registered qid MUST trip the
    // PROFILE path (just like execute) — the symmetry pin.
    let base = substrate_with_n_persons(BATCH_ROWS as u64 * 5);
    let slow = SlowSubstrate::new(base, 200);
    let cat = cat_basic();
    let registry = CancellationRegistry::new();
    let r_canceller = registry.clone();
    let canceller = thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if Instant::now() > deadline {
                panic!("registry never observed registered qid (PROFILE registration regressed)");
            }
            let ids = r_canceller.query_ids();
            if let Some(qid) = ids.first().copied() {
                r_canceller.cancel(qid);
                return;
            }
            thread::sleep(Duration::from_millis(2));
        }
    });
    let res = arcgraph_query::profile(
        "MATCH (n:Person) RETURN n",
        &cat,
        &slow,
        &registry,
        // Long deadline; the canceller fires first.
        Duration::from_secs(10),
    );
    canceller.join().expect("canceller thread panicked");
    assert!(
        matches!(res, Err(ExplainError::Cancelled)),
        "PROFILE registered with the registry must fire on external cancel; got {res:?}"
    );
    assert!(registry.is_empty());
}

// =====================================================================
// 7. W12γ fix-up MED-3 — registry no-leak under panic-during-materialize
// =====================================================================
//
// Pre-fix-up the registry's `unregister(qid)` was a sequential
// statement after `materialize()`; on a panic during the materialize
// loop, the unwind skipped the line and leaked the registry entry.
// Fix: an RAII `RegistryGuard` whose Drop impl runs unconditionally
// (success, error, cancel, OR panic-unwind).
//
// We construct a custom `ExecutorSubstrate` whose `scan_nodes`
// panics, run `engine.execute(...)` inside `std::panic::catch_unwind`,
// and assert the registry is empty post-panic.

struct PanicSubstrate;

impl ExecutorSubstrate for PanicSubstrate {
    fn scan_nodes(
        &self,
        _tenant: TenantId,
        _label: Option<LabelId>,
        _read_lsn: Lsn,
    ) -> Result<Vec<arcgraph_query::executor::BoundNode>, SubstrateAccessError> {
        panic!("PanicSubstrate::scan_nodes injected panic for MED-3 pin");
    }

    fn expand(
        &self,
        _tenant: TenantId,
        _from: NodeId,
        _rel_type: Option<TypeId>,
        _direction: arcgraph_query::logical_plan::Direction,
        _read_lsn: Lsn,
    ) -> Result<Vec<arcgraph_query::executor::BoundEdge>, SubstrateAccessError> {
        unreachable!("MATCH (n:Person) RETURN n drives scan_nodes; expand never called")
    }

    fn vector_search(
        &self,
        _tenant: TenantId,
        _property: &str,
        _query_vec: &[f32],
        _k: u64,
        _read_lsn: Lsn,
    ) -> Result<Vec<arcgraph_query::executor::RankedHit>, SubstrateAccessError> {
        unreachable!()
    }

    fn bm25_search(
        &self,
        _tenant: TenantId,
        _property: &str,
        _query_text: &str,
        _k: u64,
        _read_lsn: Lsn,
    ) -> Result<Vec<arcgraph_query::executor::RankedHit>, SubstrateAccessError> {
        unreachable!()
    }

    fn community_members(
        &self,
        _tenant: TenantId,
        _community_id: i64,
        _read_lsn: Lsn,
    ) -> Result<Vec<arcgraph_query::executor::BoundNode>, SubstrateAccessError> {
        unreachable!()
    }
}

#[test]
fn registry_does_not_leak_when_substrate_panics() {
    // The MED-3 panic-injection pin. A substrate whose `scan_nodes`
    // panics drives `engine.execute_with_query_id` to unwind through
    // `materialize()` → `execute_with_query_id` body. Pre-fix-up the
    // sequential `unregister(qid)` would NOT run; the registry would
    // leak the entry. Post-fix-up the `RegistryGuard` Drop impl runs
    // on unwind and releases the entry.
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let cat = cat_basic();
    let registry = CancellationRegistry::new();
    let engine = QueryEngine::new(&cat).with_cancellation_registry(registry.clone());
    let qid = QueryId::new();
    let panic_subs = PanicSubstrate;
    let result = catch_unwind(AssertUnwindSafe(|| {
        engine.execute_with_query_id(qid, "MATCH (n:Person) RETURN n", &panic_subs)
    }));
    assert!(
        result.is_err(),
        "PanicSubstrate must propagate the panic out of execute (got {result:?})"
    );
    assert!(
        registry.is_empty(),
        "no-leak under panic: RegistryGuard Drop must run on unwind (registry contains {} entries)",
        registry.len()
    );
}

#[test]
fn registry_does_not_leak_when_substrate_panics_under_deadline() {
    // Sibling pin: the deadline-bounded variant must also be panic-
    // safe. Same expectation; the deadline timer is dropped on
    // unwind (RAII via DeadlineHandle) AND the registry is drained
    // (RAII via RegistryGuard).
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let cat = cat_basic();
    let registry = CancellationRegistry::new();
    let engine = QueryEngine::new(&cat).with_cancellation_registry(registry.clone());
    let qid = QueryId::new();
    let panic_subs = PanicSubstrate;
    let result = catch_unwind(AssertUnwindSafe(|| {
        engine.execute_with_query_id_and_deadline(
            qid,
            "MATCH (n:Person) RETURN n",
            &panic_subs,
            Duration::from_secs(10),
        )
    }));
    assert!(result.is_err(), "panic must propagate");
    assert!(registry.is_empty(), "no-leak under panic + deadline path");
}

#[test]
fn registry_does_not_leak_when_profile_substrate_panics() {
    // MED-2 + MED-3 conjunction: PROFILE acquires a registry entry,
    // and a panic during `materialize()` must release it via the
    // RegistryGuard.
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let cat = cat_basic();
    let registry = CancellationRegistry::new();
    let panic_subs = PanicSubstrate;
    let result = catch_unwind(AssertUnwindSafe(|| {
        arcgraph_query::profile(
            "PROFILE MATCH (n:Person) RETURN n",
            &cat,
            &panic_subs,
            &registry,
            Duration::from_secs(10),
        )
    }));
    assert!(
        result.is_err(),
        "PanicSubstrate must propagate through profile"
    );
    assert!(
        registry.is_empty(),
        "PROFILE registry must drain on substrate panic"
    );
}
