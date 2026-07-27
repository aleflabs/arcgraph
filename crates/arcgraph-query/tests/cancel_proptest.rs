//! M4-92 cancellation no-leak invariant proptest per ADR-038
//! amendment-03 §TIER-1 GAP C + §2 D-17.
//!
//! # The invariant
//!
//! Cancellation MUST always release (per ADR-038 §2 D-17 + §2 D-18
//! rule 4):
//!
//! 1. **Snapshot LSN** — owned by [`ExecutionContext`]; released
//!    when the context drops at the end of the materialize loop.
//! 2. **Buffer-pool pins** — owned by the substrate access layer;
//!    v1.0-alpha stub holds none, production wiring at M4-08+ will
//!    pin/unpin per-batch.
//! 3. **Plan-cache lock** — held only inside the cache lookup /
//!    insert; cannot interleave with cancellation.
//!
//! The proptest below randomizes a sequence of (deadline-fire vs
//! cancel-call vs query-finish-before-fire) interleavings and
//! asserts a structural no-leak on each: at the end of each scenario,
//! the engine's cancellation registry MUST be empty.
//!
//! # Why a proptest vs N unit tests
//!
//! The cancellation discipline is a state-machine over two
//! fire-sources (deadline, explicit) and two outcome paths (success,
//! cancel). Random shrink-search through interleavings is more
//! reliable than enumerating cases by hand.
//!
//! # ADR provenance
//! - **ADR-038 §2 D-17** — cancellation contract.
//! - **ADR-038 §2 D-18 rule 4** — snapshot LSN released on cancel.
//! - **ADR-038 §4.3 I-Q13** — every v1.0 query is cancellable +
//!   per-query-timeout-bounded.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::Duration;

use arcgraph_core::{LabelId, Lsn, NodeId, TenantId, TypeId};
use arcgraph_query::executor::value::NodeView;
use arcgraph_query::executor::{
    BoundEdge, BoundNode, ExecutorSubstrate, RankedHit, StubExecutorSubstrate,
    SubstrateAccessError, Value,
};
use arcgraph_query::logical_plan::Direction;
use arcgraph_query::semantic::StubCatalogProvider;
use arcgraph_query::{ExplainError, QueryEngine, QueryId};

use proptest::prelude::*;

// ---------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------

fn cat_basic() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["Person"])
        .with_rel_types(["KNOWS"])
        .with_properties(["age"])
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

/// Slow-substrate adapter: sleeps `per_call_ms` per scan_nodes
/// call. Used to bound test wall-time at a known-deterministic
/// value so deadline interleavings land predictably.
struct SlowSubstrate {
    base: StubExecutorSubstrate,
    per_call_ms: u64,
    #[allow(dead_code)]
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

// ---------------------------------------------------------------------
// Outcome enum for the proptest scenarios
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
enum Scenario {
    /// Query runs to completion (no cancel, no deadline expiry).
    /// substrate has 0 rows so the scan terminates immediately.
    SuccessFast,
    /// Deadline fires before substrate exhausts.
    DeadlineExpires,
    /// Explicit `engine.cancel(qid)` from a sibling thread.
    ExplicitCancel,
    /// Both deadline AND explicit-cancel fire concurrently — the
    /// no-leak invariant must still hold (idempotent cleanup).
    BothFire,
    /// W12γ fix-up MED-3 strengthening: substrate panics during
    /// scan_nodes; the registry MUST still drain on unwind.
    /// Pre-fix-up the sequential `unregister(qid)` call after
    /// `materialize` was skipped on panic-unwind, leaking the entry.
    /// The RAII `RegistryGuard` runs on unwind; this scenario pins
    /// the structurally vacuous oracle.
    PanicDuringScan,
}

fn scenario_strategy() -> impl Strategy<Value = Scenario> {
    prop_oneof![
        Just(Scenario::SuccessFast),
        Just(Scenario::DeadlineExpires),
        Just(Scenario::ExplicitCancel),
        Just(Scenario::BothFire),
        Just(Scenario::PanicDuringScan),
    ]
}

/// Substrate that panics on scan_nodes — used by the
/// `Scenario::PanicDuringScan` arm to exercise the RAII guard's
/// panic-unwind path (W12γ fix-up MED-3).
struct PanicSubstrate;

impl ExecutorSubstrate for PanicSubstrate {
    fn scan_nodes(
        &self,
        _tenant: TenantId,
        _label: Option<LabelId>,
        _read_lsn: Lsn,
    ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
        panic!("PanicSubstrate::scan_nodes injected panic for proptest no-leak pin");
    }

    fn expand(
        &self,
        _tenant: TenantId,
        _from: NodeId,
        _rel_type: Option<TypeId>,
        _direction: Direction,
        _read_lsn: Lsn,
    ) -> Result<Vec<BoundEdge>, SubstrateAccessError> {
        unreachable!("MATCH (n:Person) RETURN n drives scan_nodes; expand never called")
    }

    fn vector_search(
        &self,
        _tenant: TenantId,
        _property: &str,
        _query_vec: &[f32],
        _k: u64,
        _read_lsn: Lsn,
    ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
        unreachable!()
    }

    fn bm25_search(
        &self,
        _tenant: TenantId,
        _property: &str,
        _query_text: &str,
        _k: u64,
        _read_lsn: Lsn,
    ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
        unreachable!()
    }

    fn community_members(
        &self,
        _tenant: TenantId,
        _community_id: i64,
        _read_lsn: Lsn,
    ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
        unreachable!()
    }
}

// ---------------------------------------------------------------------
// The proptest
// ---------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig {
        // Each case runs an actual query — keep cases low to bound
        // wall-time. 16 * (max ~600ms per case) ≈ 10s total.
        cases: 16,
        ..ProptestConfig::default()
    })]

    #[test]
    fn cancellation_always_releases_registry_entry(scenario in scenario_strategy()) {
        let cat = cat_basic();
        let engine = QueryEngine::new(&cat);
        let qid = QueryId::new();

        // The panic scenario short-circuits — we wrap in catch_unwind
        // and skip the Result-shape match (the call panicked, there
        // is no Result to inspect). The no-leak invariant is what's
        // load-bearing on this path: pre-MED-3 fix-up, the panic
        // unwind would skip the sequential `unregister(qid)` line and
        // leak the entry. Post-fix-up, the RAII RegistryGuard's Drop
        // impl runs on unwind and releases.
        if let Scenario::PanicDuringScan = scenario {
            use std::panic::{AssertUnwindSafe, catch_unwind};
            let panic_subs = PanicSubstrate;
            let result = catch_unwind(AssertUnwindSafe(|| {
                engine.execute_with_query_id(qid, "MATCH (n:Person) RETURN n", &panic_subs)
            }));
            prop_assert!(
                result.is_err(),
                "PanicSubstrate must propagate the panic out of execute_with_query_id"
            );
            prop_assert!(
                engine.cancellation_registry().is_empty(),
                "no-leak under panic: RegistryGuard Drop must release on unwind"
            );
            return Ok(());
        }

        let res = match scenario {
            Scenario::SuccessFast => {
                // 0-row substrate; the executor exits the materialize
                // loop without ever pulling a non-empty batch.
                let s = StubExecutorSubstrate::new();
                engine.execute_with_query_id(qid, "MATCH (n:Person) RETURN n", &s)
            }
            Scenario::DeadlineExpires => {
                // 200ms scan_nodes sleep means substrate returns ~200ms
                // after the materialize loop kicks off; deadline (50ms)
                // fires DURING the sleep, the executor observes the
                // cancel at the next batch boundary post-buffer-prime.
                let base = substrate_with_n_persons(50);
                let slow = SlowSubstrate::new(base, 200);
                engine.execute_with_query_id_and_deadline(
                    qid,
                    "MATCH (n:Person) RETURN n",
                    &slow,
                    Duration::from_millis(50),
                )
            }
            Scenario::ExplicitCancel => {
                let base = substrate_with_n_persons(50);
                let slow = SlowSubstrate::new(base, 200);
                // Sibling thread: poll until the registry sees the
                // qid, then fire.
                let r = engine.cancellation_registry().clone();
                let canceller = thread::spawn(move || {
                    let deadline = std::time::Instant::now() + Duration::from_secs(2);
                    while std::time::Instant::now() < deadline {
                        if !r.is_empty() {
                            r.cancel(qid);
                            return;
                        }
                        thread::sleep(Duration::from_millis(2));
                    }
                });
                let res = engine.execute_with_query_id(qid, "MATCH (n:Person) RETURN n", &slow);
                canceller.join().expect("canceller thread panicked");
                res
            }
            Scenario::BothFire => {
                let base = substrate_with_n_persons(50);
                let slow = SlowSubstrate::new(base, 200);
                let r = engine.cancellation_registry().clone();
                let canceller = thread::spawn(move || {
                    let deadline = std::time::Instant::now() + Duration::from_secs(2);
                    while std::time::Instant::now() < deadline {
                        if !r.is_empty() {
                            r.cancel(qid);
                            return;
                        }
                        thread::sleep(Duration::from_millis(2));
                    }
                });
                // Deadline ALSO active — both fire-sources are live.
                let res = engine.execute_with_query_id_and_deadline(
                    qid,
                    "MATCH (n:Person) RETURN n",
                    &slow,
                    Duration::from_millis(50),
                );
                canceller.join().expect("canceller thread panicked");
                res
            }
            Scenario::PanicDuringScan => unreachable!("handled by early-return above"),
        };

        // Outcome shape per scenario:
        match scenario {
            Scenario::SuccessFast => {
                prop_assert!(res.is_ok(), "fast-success expected Ok, got {res:?}");
            }
            Scenario::DeadlineExpires
            | Scenario::ExplicitCancel
            | Scenario::BothFire => {
                prop_assert!(
                    matches!(res, Err(ExplainError::Cancelled)),
                    "expected Cancelled, got {res:?}"
                );
            }
            Scenario::PanicDuringScan => unreachable!("handled by early-return above"),
        }

        // THE INVARIANT — registry MUST be empty regardless of
        // scenario outcome (no-leak: snapshot LSN released, buffer-
        // pool pins released, plan-cache lock released, registry
        // entry released).
        //
        // FIXME(M4-08+): strengthen oracle to assert snapshot LSN
        // release + buffer-pool pin release once production substrate
        // lands. v1.0-alpha snapshot is Lsn::MAX (no MVCC writer) and
        // stub substrate holds NO pins, so items 1-2 above are
        // vacuously true; item 3 (plan-cache lock) cannot interleave
        // with cancellation (single-function-body lock scope). When
        // M4-08+ wires production storage, items 1-2 become
        // non-trivial — the oracle here must add explicit
        // `ctx.snapshot_lsn()` post-cancel + per-call buffer-pool-pin
        // count assertions. See review packet PR #276 LOW-1.
        prop_assert!(
            engine.cancellation_registry().is_empty(),
            "no-leak invariant: registry must be drained on query end (scenario: {scenario:?})"
        );
    }
}
