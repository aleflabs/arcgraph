//! W28 Slice S6 — query-executor storage-IO fault substrate.
//!
//! # Why this slice
//!
//! Gap analysis (PR #510 §2 rank-5; ADR-165 M3 / `TEST-SLICE-PLAN-v1.md`
//! Slice S6): the ArcQL executor was only ever tested against
//! `PanicSubstrate` (panic) and logical rejects — it was **never tested
//! against substrate IO ERRORS mid-stream**. A storage page-read / WAL
//! IO error during a `scan` / `expand` must surface as a *structured*
//! [`ExecutionError::Substrate(SubstrateAccessError::Io ..)`] — NOT a
//! panic, NOT a hang, NOT a partial-row leak.
//!
//! # What this file pins (3 oracles)
//!
//! 1. [`executor_surfaces_substrate_io_error_as_structured_error`] —
//!    a [`FaultSubstrate`] returns [`SubstrateAccessError::Io`] on the
//!    configurable **Nth** `scan_nodes` / `expand` call. Driving a MATCH
//!    query through the executor yields the **exact structured variant**
//!    within a **bounded number of polls** (watchdog → no hang) and
//!    without panicking (`catch_unwind` → no panic).
//! 2. [`executor_no_leak_on_midstream_fault`] — injecting the IO fault
//!    mid-pipeline leaves **no held buffer leak**. Mirrors BOTH existing
//!    executor no-leak oracle shapes:
//!    - `m4_61_executor_proptest::prop_no_leak_on_cancellation` — the
//!      result is the *exact* clean error, never a partial `Ok(rows)`.
//!    - `m4_64a_budget_release_on_drop` — after the operator drops, the
//!      per-tenant [`MemoryBudget`] counter returns to baseline
//!      (`current_bytes == baseline`) while `peak_bytes > 0` proves the
//!      budget was genuinely exercised pre-fault (load-bearing).
//! 3. [`executor_fault_is_deterministic`] — same fixture + same fault
//!    schedule (fault on call N) ⇒ **binary-equal** result + identical
//!    fault point (`scan_calls` / `expand_calls`) on every run, per
//!    `feedback_determinism_oracle_concurrency_tests` (binary-equal
//!    reference snapshot, strictly stronger than dedupe-consistency).
//!    The stub substrate is deterministic (no RNG); the schedule fully
//!    determines the outcome.
//!
//! # Oracle discipline (ENGINEERING_DOCTRINE §3 — load-bearing)
//!
//! Every error oracle asserts the **exact** structured variant
//! (`ExecutionError::Substrate(SubstrateAccessError::Io(..))` with the
//! exact deterministic message) via `assert_eq!` — never a bare
//! `result.is_err()`, which cannot distinguish a clean IO-error
//! surfacing from a panic-converted-to-error or a wrong variant.
//!
//! # Scope
//!
//! Test-only. [`FaultSubstrate`] is a `#[cfg(test)]`-equivalent test
//! double living in `tests/`; no production type, no production-logic
//! change. The fault surface is the executor **read path** (`scan` /
//! `expand`) per the slice scope ("page-read / WAL IO error during a
//! scan/expand"); write-op substrate methods retain their trait
//! defaults (not exercised here).

use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicU64, Ordering};

use arcgraph_core::{LabelId, Lsn, NodeId, PartitionId, RelId, TenantId, TypeId};
use arcgraph_query::error::Span;
use arcgraph_query::executor::ops::{ExpandOp, PhysicalOperator, ScanOp, SortKey, SortOp};
use arcgraph_query::executor::value::{NodeView, RelView};
use arcgraph_query::executor::{
    BATCH_ROWS, BoundEdge, BoundNode, ExecutionContext, ExecutionError, ExecutorSubstrate,
    MemoryBudget, Pipeline, RankedHit, StubExecutorSubstrate, SubstrateAccessError, Value,
    execute_with_context,
};
use arcgraph_query::logical_plan::{
    Direction, LogicalExpand, LogicalPlan, LogicalScan, SortDirection,
};
use arcgraph_query::semantic::bound_ast::{BindingId, BoundExpression, BoundPropertyRef};

// =====================================================================
// FaultSubstrate — test double wrapping the canonical StubExecutorSubstrate.
// =====================================================================

/// Wraps an inner [`ExecutorSubstrate`] and injects a
/// [`SubstrateAccessError::Io`] on the configurable **Nth** `scan_nodes`
/// / `expand` call (1-indexed). Every other call (and every other
/// method) delegates transparently to the inner substrate, so the
/// fixture data + ordering match the canonical stub exactly.
///
/// The injected error message is a deterministic function of the call
/// index, so the same schedule produces a binary-equal
/// [`ExecutionError`] every run (load-bearing for the determinism
/// oracle).
struct FaultSubstrate<S: ExecutorSubstrate> {
    inner: S,
    /// 1-indexed `scan_nodes` call on which to return `Io`; `None` = never.
    scan_fault_on: Option<u64>,
    /// 1-indexed `expand` call on which to return `Io`; `None` = never.
    expand_fault_on: Option<u64>,
    scan_calls: AtomicU64,
    expand_calls: AtomicU64,
}

impl<S: ExecutorSubstrate> FaultSubstrate<S> {
    fn new(inner: S) -> Self {
        Self {
            inner,
            scan_fault_on: None,
            expand_fault_on: None,
            scan_calls: AtomicU64::new(0),
            expand_calls: AtomicU64::new(0),
        }
    }

    /// Fault the `n`-th `scan_nodes` call (1-indexed).
    fn fault_scan_on(mut self, n: u64) -> Self {
        self.scan_fault_on = Some(n);
        self
    }

    /// Fault the `n`-th `expand` call (1-indexed).
    fn fault_expand_on(mut self, n: u64) -> Self {
        self.expand_fault_on = Some(n);
        self
    }

    /// Number of `scan_nodes` calls observed so far.
    fn scan_calls(&self) -> u64 {
        self.scan_calls.load(Ordering::SeqCst)
    }

    /// Number of `expand` calls observed so far.
    fn expand_calls(&self) -> u64 {
        self.expand_calls.load(Ordering::SeqCst)
    }
}

/// Deterministic injected-fault message for `scan_nodes` call `n`. A
/// free function (not an associated fn on the generic type) so both the
/// substrate impl and the test oracles can reconstruct the exact
/// expected message without a turbofish.
fn scan_fault_msg(n: u64) -> String {
    format!("injected scan_nodes IO fault on call {n}")
}

/// Deterministic injected-fault message for `expand` call `n`.
fn expand_fault_msg(n: u64) -> String {
    format!("injected expand IO fault on call {n}")
}

impl<S: ExecutorSubstrate> ExecutorSubstrate for FaultSubstrate<S> {
    fn scan_nodes(
        &self,
        tenant: TenantId,
        label: Option<LabelId>,
        read_lsn: Lsn,
    ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
        let n = self.scan_calls.fetch_add(1, Ordering::SeqCst) + 1;
        if self.scan_fault_on == Some(n) {
            return Err(SubstrateAccessError::Io(scan_fault_msg(n)));
        }
        self.inner.scan_nodes(tenant, label, read_lsn)
    }

    fn expand(
        &self,
        tenant: TenantId,
        from: NodeId,
        rel_type: Option<TypeId>,
        direction: Direction,
        read_lsn: Lsn,
    ) -> Result<Vec<BoundEdge>, SubstrateAccessError> {
        let n = self.expand_calls.fetch_add(1, Ordering::SeqCst) + 1;
        if self.expand_fault_on == Some(n) {
            return Err(SubstrateAccessError::Io(expand_fault_msg(n)));
        }
        self.inner
            .expand(tenant, from, rel_type, direction, read_lsn)
    }

    fn node_by_id_with_context(
        &self,
        ctx: &ExecutionContext,
        id: NodeId,
    ) -> Result<Option<BoundNode>, SubstrateAccessError> {
        self.inner.node_by_id_with_context(ctx, id)
    }

    fn vector_search(
        &self,
        tenant: TenantId,
        property: &str,
        query_vec: &[f32],
        k: u64,
        read_lsn: Lsn,
    ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
        self.inner
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
        self.inner
            .bm25_search(tenant, property, query_text, k, read_lsn)
    }

    fn community_members(
        &self,
        tenant: TenantId,
        community_id: i64,
        read_lsn: Lsn,
    ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
        self.inner.community_members(tenant, community_id, read_lsn)
    }

    fn has_vector_substrate(&self) -> bool {
        self.inner.has_vector_substrate()
    }

    fn has_bm25_substrate(&self) -> bool {
        self.inner.has_bm25_substrate()
    }

    fn has_community_substrate(&self) -> bool {
        self.inner.has_community_substrate()
    }
}

// =====================================================================
// Fixtures + plan/expr builders.
// =====================================================================

/// Generous poll watchdog. The real poll count to surface the fault is
/// 1–2 for every fixture here; a bound this large can only be exceeded
/// by a genuine infinite loop (per testing-strategy §3.5: generous, NOT
/// a tight wall-clock).
const MAX_POLLS: usize = 10_000;

/// `n` label-1 nodes (ids `1..=n`). Used for the scan-fault path.
fn labeled_nodes(n: u64) -> StubExecutorSubstrate {
    let mut s = StubExecutorSubstrate::new();
    for i in 1..=n {
        s = s.with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(i), Some(LabelId::new(1))),
        );
    }
    s
}

/// Chain of `n` label-1 nodes (ids `1..=n`, each carrying `seq = i`),
/// with a single outbound `KNOWS` (type 1) edge `i -> i+1` for
/// `i in 1..n`. Scanning yields nodes in ascending-id order, so
/// `expand` call `k` corresponds to node `k`.
fn chain(n: u64) -> StubExecutorSubstrate {
    let mut s = StubExecutorSubstrate::new();
    for i in 1..=n {
        s = s.with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(i), Some(LabelId::new(1)))
                .with_property("seq", Value::Integer(i as i64)),
        );
    }
    for i in 1..n {
        s = s.with_edge(
            TenantId::DEFAULT,
            RelView::new(
                RelId::new(i),
                NodeId::new(i),
                NodeId::new(i + 1),
                Some(TypeId::new(1)),
            ),
        );
    }
    s
}

/// `MATCH (n:L1) RETURN n` — bare scan over binding 0. The synthesized
/// `ScanOp` calls `scan_nodes` exactly once (lazy first-batch prime).
fn scan_plan() -> LogicalPlan {
    LogicalPlan::Scan(LogicalScan {
        label: Some(LabelId::new(1)),
        var: BindingId::new(0),
        read_lsn: Lsn::MAX,
        span: Span::point(1, 1),
    })
}

/// `MATCH (a)-->(b) RETURN b` — bare expand. `Pipeline::build` lowers
/// this to `Scan(a) → Expand`; the scan emits one `scan_nodes` call,
/// then `expand` is called once per scanned source node.
fn expand_plan() -> LogicalPlan {
    LogicalPlan::Expand(LogicalExpand {
        from: BindingId::new(0),
        to: BindingId::new(1),
        direction: Direction::LeftToRight,
        rel_type: None,
        length_range: None,
        rel_var: None,
        span: Span::point(1, 1),
    })
}

/// `BoundExpression` for the `a` binding (binding 0).
fn var_a() -> BoundExpression {
    BoundExpression::VariableRef {
        name: "a".into(),
        binding_id: BindingId::new(0),
        span: Span::point(1, 1),
        type_info: None,
    }
}

/// `a.seq` property access (used as a sort key).
fn prop_seq(base: BoundExpression) -> BoundExpression {
    BoundExpression::PropertyAccess {
        base: Box::new(base),
        path: vec![BoundPropertyRef {
            name: "seq".into(),
            property_id: None,
            span: Span::point(1, 1),
        }],
        span: Span::point(1, 1),
        type_info: None,
    }
}

/// Outcome of driving a pipeline under a generous poll watchdog. The
/// watchdog returns a sentinel (never panics) so the caller can
/// distinguish a *hang* (`Exceeded`) from a real *panic* (caught
/// separately by `catch_unwind`).
enum Drive {
    /// The executor surfaced an error after `usize` polls.
    Err(ExecutionError, usize),
    /// The executor reached EOS cleanly (no fault surfaced).
    Eos,
    /// The watchdog tripped — suspected hang / infinite loop.
    Exceeded,
}

/// Drive `plan` through a freshly-built pipeline, polling `next_batch`
/// until it errors, reaches EOS, or trips the `max_polls` watchdog.
fn drive_bounded<S: ExecutorSubstrate>(
    plan: &LogicalPlan,
    substrate: &S,
    ctx: &ExecutionContext,
    max_polls: usize,
) -> Drive {
    let mut op = match Pipeline::build(plan) {
        Ok(op) => op,
        Err(e) => return Drive::Err(e, 0),
    };
    let mut polls = 0usize;
    loop {
        if polls >= max_polls {
            return Drive::Exceeded;
        }
        polls += 1;
        match op.next_batch(ctx, substrate) {
            Ok(b) if b.is_empty() => return Drive::Eos,
            Ok(_) => continue,
            Err(e) => return Drive::Err(e, polls),
        }
    }
}

// =====================================================================
// Oracle 1 — structured error, no panic, no hang (bounded polls).
// =====================================================================

#[test]
fn executor_surfaces_substrate_io_error_as_structured_error() {
    // ---- Sub-case A: scan_nodes IO fault on the 1st scan call. ----
    {
        let fault = FaultSubstrate::new(labeled_nodes(8)).fault_scan_on(1);
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let plan = scan_plan();

        // `catch_unwind` makes "no panic" concrete: a panic in the
        // executor surfaces here as an explicit test failure rather than
        // an unwind. The bounded-poll watchdog inside `drive_bounded`
        // returns `Drive::Exceeded` (it does NOT panic) so a hang is not
        // misreported as a panic.
        let drive = std::panic::catch_unwind(AssertUnwindSafe(|| {
            drive_bounded(&plan, &fault, &ctx, MAX_POLLS)
        }))
        .unwrap_or_else(|_| {
            panic!(
                "executor PANICKED on a scan_nodes IO fault — it MUST surface \
                 ExecutionError::Substrate(Io), not panic"
            )
        });

        match drive {
            Drive::Err(e, polls) => {
                // Strong oracle: exact structured variant + exact payload
                // (NOT a bare `is_err()` — that cannot distinguish a clean
                // IO surfacing from a panic-converted error or wrong variant).
                assert!(
                    matches!(e, ExecutionError::Substrate(SubstrateAccessError::Io(_))),
                    "scan IO fault must surface as ExecutionError::Substrate(Io); got {e:?}"
                );
                assert_eq!(
                    e,
                    ExecutionError::Substrate(SubstrateAccessError::Io(scan_fault_msg(1))),
                    "exact structured IO error (variant + message)"
                );
                assert!(
                    (1..=MAX_POLLS).contains(&polls),
                    "error returned within a bounded number of polls (got {polls})"
                );
            }
            Drive::Eos => panic!(
                "scan IO fault was NOT surfaced — executor reached EOS cleanly \
                 (fault not injected?)"
            ),
            Drive::Exceeded => panic!(
                "executor exceeded {MAX_POLLS} polls without surfacing the IO \
                 fault — suspected hang / infinite loop"
            ),
        }
        assert_eq!(
            fault.scan_calls(),
            1,
            "exactly one scan_nodes call drove the bare scan plan to the fault"
        );
    }

    // ---- Sub-case B: expand IO fault mid-stream (3rd expand call). ----
    {
        let fault = FaultSubstrate::new(chain(5)).fault_expand_on(3);
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let plan = expand_plan();

        let drive = std::panic::catch_unwind(AssertUnwindSafe(|| {
            drive_bounded(&plan, &fault, &ctx, MAX_POLLS)
        }))
        .unwrap_or_else(|_| {
            panic!(
                "executor PANICKED on an expand IO fault — it MUST surface \
                 ExecutionError::Substrate(Io), not panic"
            )
        });

        match drive {
            Drive::Err(e, polls) => {
                assert!(
                    matches!(e, ExecutionError::Substrate(SubstrateAccessError::Io(_))),
                    "expand IO fault must surface as ExecutionError::Substrate(Io); got {e:?}"
                );
                assert_eq!(
                    e,
                    ExecutionError::Substrate(SubstrateAccessError::Io(expand_fault_msg(3))),
                    "exact structured IO error (variant + message)"
                );
                assert!(
                    (1..=MAX_POLLS).contains(&polls),
                    "error returned within a bounded number of polls (got {polls})"
                );
            }
            Drive::Eos => panic!("expand IO fault was NOT surfaced — executor reached EOS cleanly"),
            Drive::Exceeded => panic!(
                "executor exceeded {MAX_POLLS} polls without surfacing the IO \
                 fault — suspected hang / infinite loop"
            ),
        }
        // Scan ran once; expand reached exactly the 3rd call (the fault
        // point) and stopped there.
        assert_eq!(
            fault.scan_calls(),
            1,
            "one scan_nodes call seeded the expand"
        );
        assert_eq!(
            fault.expand_calls(),
            3,
            "expand stopped at the injected fault point (3rd call)"
        );
    }
}

// =====================================================================
// Oracle 2 — no held-buffer leak on a mid-pipeline fault.
// =====================================================================

#[test]
fn executor_no_leak_on_midstream_fault() {
    // A `Sort` over an `Expand` over a `Scan`. The scan buffers
    // BATCH_ROWS+1 nodes; `expand` is called once per node. The fault is
    // injected on the LAST expand call — the lone node in the SECOND
    // scan batch — so the `SortOp` has already buffered (and budget-
    // reserved) the first full batch of rows BEFORE the fault. This is
    // the genuine "mid-pipeline fault while buffers are held" condition.
    let n = BATCH_ROWS as u64 + 1;
    let fault_call = n; // expand call index of the node in scan batch 2.
    let fault = FaultSubstrate::new(chain(n)).fault_expand_on(fault_call);

    let tenant = TenantId::DEFAULT;
    // Generous per-tenant cap: the query stays well under it, so the
    // budget path is the *bookkeeping* path (reserve-then-release), not
    // the rejection path.
    let budget = MemoryBudget::with_per_tenant_cap(tenant, 1_000_000_000);
    let baseline = budget.current_bytes(tenant);
    assert_eq!(baseline, 0, "fresh budget starts at zero");

    let ctx = ExecutionContext::new(tenant, PartitionId::ZERO).with_budget(budget.clone());

    // Build the physical operator tree directly (mirroring the
    // m4_64a_budget_release_on_drop.rs construction shape).
    let scan = ScanOp::new(BindingId::new(0), Some(LabelId::new(1)), Lsn::MAX);
    let expand = ExpandOp::new(
        PhysicalOperator::Scan(scan),
        BindingId::new(0),
        None,
        BindingId::new(1),
        None,
        Direction::LeftToRight,
        None,
        Lsn::MAX,
    )
    .expect("fixed-length expand builds");
    let mut sort = SortOp::new(
        PhysicalOperator::Expand(expand),
        vec![SortKey {
            expr: prop_seq(var_a()),
            direction: SortDirection::Asc,
        }],
    );

    // Drive to the fault under the poll watchdog (no hang).
    let mut polls = 0usize;
    let result = loop {
        assert!(
            polls < MAX_POLLS,
            "no-leak drive exceeded {MAX_POLLS} polls — suspected hang"
        );
        polls += 1;
        match sort.next_batch(&ctx, &fault) {
            Ok(b) if b.is_empty() => break Ok::<(), ExecutionError>(()),
            Ok(_) => continue,
            Err(e) => break Err(e),
        }
    };

    // (m4_61 no-leak shape) The fault surfaces as the EXACT clean
    // structured error — never a partial `Ok(())`/`Ok(rows)`. No partial
    // result leaks past a mid-stream fault.
    assert_eq!(
        result,
        Err(ExecutionError::Substrate(SubstrateAccessError::Io(
            expand_fault_msg(fault_call)
        ))),
        "mid-pipeline fault surfaces the exact structured IO error, not partial rows"
    );

    // Load-bearing proof: the SortOp DID reserve budget while buffering
    // the pre-fault batch (otherwise the no-leak assertion below would be
    // trivially true on an untouched counter).
    assert!(
        budget.peak_bytes(tenant) > 0,
        "budget was exercised pre-fault (peak_bytes must be > 0); got {}",
        budget.peak_bytes(tenant)
    );

    // (m4_64a release-on-drop shape) After the operator drops, the
    // per-tenant counter MUST return to baseline — no held-buffer leak.
    drop(sort);
    let after = budget.current_bytes(tenant);
    assert_eq!(
        after,
        baseline,
        "after the faulting pipeline drops, the per-tenant counter MUST return \
         to baseline ({baseline}); observed {after} (leak = {})",
        after as i64 - baseline as i64
    );
}

// =====================================================================
// Oracle 3 — fault is deterministic (binary-equal reference snapshot).
// =====================================================================

#[test]
fn executor_fault_is_deterministic() {
    // Same fixture + same schedule (expand fault on call N) every run.
    // The stub substrate is deterministic (ascending-id scan order, no
    // RNG), so the schedule fully determines the outcome.
    const RUNS: usize = 8;
    const FAULT_N: u64 = 3;

    // Snapshot = (full executor result, scan-call point, expand-call
    // point). Binary-equality across runs pins BOTH "same error" and
    // "same point" per feedback_determinism_oracle_concurrency_tests.
    type Snapshot = (Result<Vec<Vec<Value>>, ExecutionError>, u64, u64);

    let mut snapshots: Vec<Snapshot> = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let fault = FaultSubstrate::new(chain(5)).fault_expand_on(FAULT_N);
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let result = execute_with_context(&expand_plan(), &fault, &ctx);
        snapshots.push((result, fault.scan_calls(), fault.expand_calls()));
    }

    // Baseline sanity: the reference run is the structured IO error,
    // surfaced at exactly the configured point.
    {
        let (first_result, first_scan_calls, first_expand_calls) = &snapshots[0];
        assert!(
            matches!(
                first_result,
                Err(ExecutionError::Substrate(SubstrateAccessError::Io(_)))
            ),
            "determinism baseline must be the structured IO error; got {first_result:?}"
        );
        assert_eq!(
            *first_expand_calls, FAULT_N,
            "fault fired at the configured point (expand call {FAULT_N})"
        );
        assert_eq!(
            *first_scan_calls, 1,
            "exactly one scan_nodes call seeded the expand stream"
        );
    }

    // Binary-equal determinism oracle: every run is identical to run 0.
    for (i, snap) in snapshots.iter().enumerate() {
        assert_eq!(
            snap, &snapshots[0],
            "run {i} diverged from run 0 — the executor's IO-fault behavior is \
             NOT deterministic (binary-equal reference-snapshot oracle)"
        );
    }
}
