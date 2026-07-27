//! W27-ν / ADR-163 — Jepsen-style ArcQL read-side SI integration
//! tests (ACTIVE; CI-gating).
//!
//! Drives the ArcQL write-op surface (CREATE / DELETE / batched-CREATE
//! / read-modify-write) under concurrent execution against the REAL
//! MVCC kernel and verifies the recorded history is snapshot-isolation
//! legal via the [`common::checker::ArcqlSiChecker`] oracle. One test
//! per Adya 2000 §4 / Bailis 2014 §3 anomaly class, plus a steady-state
//! proptest and end-to-end executor transit pins through
//! [`arcgraph_mcp::storage::CrudExecutorSubstrate`].
//!
//! "Read-side" = the CHECKER verifies what concurrent reads (MATCH)
//! observe is SI-legal while CREATE/DELETE writes commit concurrently.
//! This needs only node-identity visibility (label/identity), NOT
//! property persistence, so it is independent of W27-α. Property
//! round-trip verification (CREATE {p} → MATCH {p}) is the write-side
//! skeleton (`jepsen_arcql_si_write_skeleton.rs`), forward-deferred to
//! W27-α per ADR-163 §"Forward-deferred".
//!
//! Run:
//!   cargo test -p arcgraph-mcp --test jepsen_arcql_si_read --release -- --nocapture
//!
//! Determinism contract: the checker predicate is the oracle, NOT a
//! reference snapshot. See `common`'s module rustdoc.

#[path = "jepsen_arcql_common/mod.rs"]
mod common;

use std::sync::Arc;

use arcgraph_core::{LabelId, NodeId, PartitionId, TenantId};
use arcgraph_storage::crud::{self, PropertyData};
use arcgraph_storage::test_harness::jepsen::history::{OpOutcome, OperationHistory, RecordedOp};

use arcgraph_query::executor::ExecutionContext;
use arcgraph_query::executor::Pipeline;
use arcgraph_query::logical_plan::{LogicalPlan, LogicalPlanLoweringVisitor};
use arcgraph_query::semantic::{
    BindingVisitor, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};
use arcgraph_query::{Statement, parse};

use common::checker::{ArcqlSiChecker, ArcqlVerdict};
use common::{
    JepsenArcqlFixture, SCAN_SENTINEL_KEY, WorkloadConfig, WorkloadKind, is_match_op,
    live_visible_count, outcome_counts, run_workload,
};

/// Expected live node count derived from the recorded history:
/// committed creates minus committed deletes. Used for the live
/// cross-check (catches "history says one thing, store says another").
fn expected_live_from_history(ops: &[RecordedOp]) -> i64 {
    let mut creates: i64 = 0;
    let mut deletes: i64 = 0;
    for op in ops {
        if op.outcome != OpOutcome::Committed || is_match_op(op) {
            continue;
        }
        for w in &op.writes {
            if w.value.is_some() {
                creates += 1;
            } else {
                deletes += 1;
            }
        }
    }
    creates - deletes
}

/// Drain + check a populated history, returning the verdict + the
/// drained ops for further assertions.
fn drain_and_check(history: &OperationHistory) -> (ArcqlVerdict, Vec<RecordedOp>) {
    let ops = history.drain_sorted();
    let verdict = ArcqlSiChecker::new().check(&ops);
    (verdict, ops)
}

// ─────────────────────────────────────────────────────────────────────
// Steady-state SI (snapshot-read consistency)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn steady_state_preserves_si() {
    let fixture = JepsenArcqlFixture::new();
    let cfg = WorkloadConfig::default();
    let history = Arc::new(OperationHistory::new());
    run_workload(
        &fixture,
        WorkloadKind::SteadyState,
        cfg,
        Arc::clone(&history),
    );

    let (verdict, ops) = drain_and_check(&history);
    assert!(verdict.is_ok(), "steady-state SI violation: {verdict}");

    // Non-vacuous: the workload actually committed work + ran MATCH ops.
    let (committed, _aborted) = outcome_counts(&ops);
    let summary = verdict.summary().unwrap();
    let attempted = (cfg.clients as usize) * (cfg.ops_per_client as usize);
    assert!(
        committed >= attempted / 2,
        "commit rate too low: {committed} of {attempted}"
    );
    assert!(
        summary.snapshot_checks > 0,
        "no MATCH ops were checked — vacuous SI pass"
    );

    // Live cross-check: store state matches the recorded history.
    let expected_live = expected_live_from_history(&ops);
    assert_eq!(
        live_visible_count(&fixture) as i64,
        expected_live,
        "live visible count diverged from recorded history"
    );
}

// ─────────────────────────────────────────────────────────────────────
// G0 — dirty write (concurrent DELETE of shared nodes)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn g0_dirty_write_atomic() {
    let fixture = JepsenArcqlFixture::new();
    let cfg = WorkloadConfig {
        seed: 0xD117_0000_0000_0001,
        seed_nodes: 16,
        ..WorkloadConfig::default()
    };
    let history = Arc::new(OperationHistory::new());
    run_workload(
        &fixture,
        WorkloadKind::G0DirtyWrite,
        cfg,
        Arc::clone(&history),
    );

    let (verdict, ops) = drain_and_check(&history);
    assert!(
        verdict.is_ok(),
        "G0 dirty-write: a node was observed in a torn/inconsistent state: {verdict}"
    );

    // Non-vacuous: deletes must have committed (some) — proving the
    // concurrent WW-conflict path was exercised, not all-aborted.
    let committed_deletes = ops
        .iter()
        .filter(|o| o.outcome == OpOutcome::Committed && !is_match_op(o))
        .flat_map(|o| o.writes.iter())
        .filter(|w| w.value.is_none())
        .count();
    assert!(
        committed_deletes > 0,
        "no DELETE committed — workload did not exercise the WW path"
    );

    let expected_live = expected_live_from_history(&ops);
    assert_eq!(live_visible_count(&fixture) as i64, expected_live);
}

// ─────────────────────────────────────────────────────────────────────
// G1a — aborted read (injected aborts; never observed)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn g1a_aborted_read_never_observed() {
    let fixture = JepsenArcqlFixture::new();
    let cfg = WorkloadConfig {
        seed: 0x6A1A_0000_0000_0001,
        abort_one_in: 3,
        ops_per_client: 80,
        ..WorkloadConfig::default()
    };
    let history = Arc::new(OperationHistory::new());
    run_workload(
        &fixture,
        WorkloadKind::G1aAbortedRead,
        cfg,
        Arc::clone(&history),
    );

    let (verdict, _ops) = drain_and_check(&history);
    assert!(
        verdict.is_ok(),
        "G1a: a MATCH observed an aborted CREATE's node: {verdict}"
    );

    // Non-vacuous fault-injection assertion: aborts genuinely happened.
    // This is the load-bearing fault-injection regression test per
    // `feedback_load_bearing_pr_requires_fault_injection_tests.md` — a
    // green verdict with zero injected aborts would be vacuous.
    let summary = verdict.summary().unwrap();
    assert!(
        summary.aborted_writes > 0,
        "no aborts were injected — G1a test is vacuous (aborted_writes=0)"
    );
    assert!(summary.snapshot_checks > 0, "no MATCH ops checked");
}

// ─────────────────────────────────────────────────────────────────────
// G1b — intermediate read (multi-write txns; all-or-none)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn g1b_intermediate_read_all_or_none() {
    let fixture = JepsenArcqlFixture::new();
    let cfg = WorkloadConfig {
        seed: 0x61B0_0000_0000_0001,
        batch_size: 4,
        ..WorkloadConfig::default()
    };
    let history = Arc::new(OperationHistory::new());
    run_workload(
        &fixture,
        WorkloadKind::G1bIntermediate,
        cfg,
        Arc::clone(&history),
    );

    let (verdict, ops) = drain_and_check(&history);
    assert!(
        verdict.is_ok(),
        "G1b: a MATCH observed a partial subset of a multi-write tx: {verdict}"
    );

    // Non-vacuous: multi-write (batch) txns committed (writes>=2 in one op).
    let multi_write_commits = ops
        .iter()
        .filter(|o| o.outcome == OpOutcome::Committed && o.writes.len() >= 2)
        .count();
    assert!(
        multi_write_commits > 0,
        "no multi-write tx committed — G1b not exercised"
    );
    assert!(verdict.summary().unwrap().snapshot_checks > 0);
}

// ─────────────────────────────────────────────────────────────────────
// G1c — circular information flow (read-modify-write; acyclic DSG)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn g1c_no_cyclic_information_flow() {
    let fixture = JepsenArcqlFixture::new();
    let cfg = WorkloadConfig {
        seed: 0x61C0_0000_0000_0001,
        ..WorkloadConfig::default()
    };
    let history = Arc::new(OperationHistory::new());
    run_workload(
        &fixture,
        WorkloadKind::G1cReadThenWrite,
        cfg,
        Arc::clone(&history),
    );

    let (verdict, ops) = drain_and_check(&history);
    assert!(
        verdict.is_ok(),
        "G1c: ww∪wr dependency graph has a cycle (circular information flow): {verdict}"
    );

    // Non-vacuous: read-modify-write ops committed (reads AND writes).
    let rmw_commits = ops
        .iter()
        .filter(|o| {
            o.outcome == OpOutcome::Committed
                && !o.writes.is_empty()
                && o.reads.iter().any(|r| r.key != SCAN_SENTINEL_KEY)
        })
        .count();
    assert!(
        rmw_commits > 0,
        "no read-modify-write tx committed — G1c not exercised"
    );
}

// ─────────────────────────────────────────────────────────────────────
// G2-item — write skew (SI PERMITS; reported, not a violation)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn g2_write_skew_permitted_under_si() {
    let fixture = JepsenArcqlFixture::new();
    let cfg = WorkloadConfig {
        seed: 0x62A0_0000_0000_0001,
        skew_threshold: 6,
        seed_nodes: 4,
        ..WorkloadConfig::default()
    };
    let history = Arc::new(OperationHistory::new());
    run_workload(
        &fixture,
        WorkloadKind::G2WriteSkew,
        cfg,
        Arc::clone(&history),
    );

    let (verdict, _ops) = drain_and_check(&history);
    // The load-bearing assertion: write skew is NOT an SI violation
    // (Adya 2000 §4.3). The checker must report OK even though write
    // skew may have occurred — and it must STILL catch any G1c / G1a /
    // snapshot-read violation in the same history.
    assert!(
        verdict.is_ok(),
        "G2 workload surfaced a genuine SI violation (G2 itself is permitted; \
         this is G1c/G1a/snapshot-read): {verdict}"
    );
    // Witness count is informational (write skew is SI-legal); print it
    // so the baseline report can document whether the run exercised the
    // skew shape. Not asserted >0 to avoid interleaving-dependent flake.
    let witnesses = verdict.summary().unwrap().writeskew_witnesses;
    println!("G2 write-skew witnesses (SI-permitted, informational): {witnesses}");
}

// ─────────────────────────────────────────────────────────────────────
// Steady-state SI proptest (release; multiple seeds)
// ─────────────────────────────────────────────────────────────────────

mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn proptest_cases() -> u32 {
        std::env::var("PROPTEST_CASES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8)
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: proptest_cases(),
            source_file: Some(file!()),
            .. ProptestConfig::default()
        })]

        /// Steady-state SI under random seeds: each case is a fresh
        /// MVCC kernel + a CREATE/MATCH workload. The checker predicate
        /// is the oracle.
        #[test]
        fn steady_state_si_under_random_seed(seed in any::<u64>()) {
            let fixture = JepsenArcqlFixture::new();
            let cfg = WorkloadConfig { seed, clients: 4, ops_per_client: 40, ..WorkloadConfig::default() };
            let history = Arc::new(OperationHistory::new());
            run_workload(&fixture, WorkloadKind::SteadyState, cfg, Arc::clone(&history));

            let ops = history.drain_sorted();
            let verdict = ArcqlSiChecker::new().check(&ops);
            prop_assert!(verdict.is_ok(), "SI violation under seed {seed:#x}: {verdict}");

            let expected_live = expected_live_from_history(&ops);
            prop_assert_eq!(live_visible_count(&fixture) as i64, expected_live, "live diverged seed={:#x}", seed);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// End-to-end executor transit pins (literal ArcQL operators →
// CrudExecutorSubstrate → real MVCC)
// ─────────────────────────────────────────────────────────────────────

/// Walk parse → bind → type-check → cross-substrate → lower for a
/// single query against a fresh `StubCatalogProvider` (the binder's
/// catalog; the *substrate* is the real `CrudExecutorSubstrate`).
/// Mirrors `crates/arcgraph-query/tests/create_node_smoke.rs`'s helper.
fn lower(query: &str) -> LogicalPlan {
    let stmt = parse(query).expect("parse OK");
    let inner = match stmt {
        Statement::Read(_) => stmt,
        other => panic!("expected Read-shaped statement, got {other:?}"),
    };
    let cat = StubCatalogProvider::new();
    let mut bound = BindingVisitor::bind(&inner, query, &cat).expect("bind OK");
    TypeCheckVisitor::check(&mut bound, &cat).expect("type-check OK");
    CrossSubstrateValidator::validate(&bound, &cat).expect("cross-substrate OK");
    LogicalPlanLoweringVisitor::lower(&bound).expect("lower OK")
}

/// Count rows emitted by executing a plan to EOS against the substrate.
fn execute_row_count(
    plan: &LogicalPlan,
    ctx: &ExecutionContext,
    substrate: &arcgraph_mcp::storage::CrudExecutorSubstrate,
) -> usize {
    let mut op = Pipeline::build(plan).expect("pipeline build OK");
    let mut rows = 0usize;
    loop {
        let batch = op.next_batch(ctx, substrate).expect("next_batch OK");
        if batch.is_empty() {
            break;
        }
        rows += batch.row_count();
    }
    rows
}

#[test]
fn executor_create_then_match_through_crud_substrate() {
    // The literal ArcQL operator surface: CREATE then MATCH through the
    // production CrudExecutorSubstrate over the real MVCC kernel.
    let fixture = JepsenArcqlFixture::new();
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);

    // CREATE three nodes (each `CreateNodeOp` → substrate.create_node →
    // begin/crud::create_node/commit).
    for _ in 0..3 {
        let plan = lower("CREATE (n) RETURN n");
        let rows = execute_row_count(&plan, &ctx, &fixture.substrate);
        assert_eq!(rows, 1, "CREATE emits exactly one row");
    }

    // MATCH (n) RETURN n → ScanOp → substrate.scan_nodes → MVCC scan.
    let plan = lower("MATCH (n) RETURN n");
    let rows = execute_row_count(&plan, &ctx, &fixture.substrate);
    assert_eq!(rows, 3, "MATCH sees the three committed CREATEs");

    // Live cross-check via the substrate trait method directly.
    assert_eq!(live_visible_count(&fixture), 3);
}

#[test]
fn concurrent_executor_creates_preserve_visibility() {
    // N threads each drive `CREATE (n) RETURN n` through the executor
    // against a SHARED CrudExecutorSubstrate; afterward a single MATCH
    // must see exactly the number of committed creates (atomicity + no
    // lost/torn creates under concurrency — the executor-level analog
    // of the crud-tier steady-state test).
    let fixture = JepsenArcqlFixture::new();
    let clients = 4u32;
    let per_client = 10u32;

    let handles: Vec<_> = (0..clients)
        .map(|c| {
            let substrate = fixture.substrate.clone();
            std::thread::Builder::new()
                .name(format!("jepsen-arcql-exec-{c}"))
                .spawn(move || {
                    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
                    let plan = lower("CREATE (n) RETURN n");
                    let mut ok = 0u32;
                    for _ in 0..per_client {
                        // Rebuild the pipeline per op (operators are
                        // stateful / single-shot).
                        let mut op = Pipeline::build(&plan).expect("build");
                        let b = op.next_batch(&ctx, &substrate).expect("batch");
                        ok += b.row_count() as u32;
                    }
                    ok
                })
                .expect("spawn")
        })
        .collect();

    let mut total_created = 0u32;
    for h in handles {
        total_created += h.join().expect("exec worker panicked");
    }
    assert_eq!(total_created, clients * per_client);

    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let plan = lower("MATCH (n) RETURN n");
    let seen = execute_row_count(&plan, &ctx, &fixture.substrate);
    assert_eq!(
        seen,
        (clients * per_client) as usize,
        "MATCH must see exactly every committed CREATE (no lost/torn creates)"
    );
}

// ─────────────────────────────────────────────────────────────────────
// Deterministic SI-property pins (not interleaving-dependent)
//
// The randomized history-checker workloads above verify SI holds over
// *whatever* interleaving occurs; these pins additionally prove the
// individual SI properties hold under a *constructed* interleaving, so
// the suite is non-vacuous even when the scheduler happens to serialize
// the random workloads (per `feedback_review_oracle_relaxations.md` —
// a green checker on a never-concurrent history would be a weak oracle).
// ─────────────────────────────────────────────────────────────────────

/// Count nodes visible to `tx`'s snapshot (1..=high_water + read_node
/// MVCC filter), mirroring `CrudExecutorSubstrate::scan_nodes`.
fn visible_count(
    fixture: &JepsenArcqlFixture,
    tx: &arcgraph_storage::transaction::Transaction<'_>,
) -> usize {
    let hw = fixture.crud.node_high_water(fixture.tenant);
    (1..=hw)
        .filter(|&raw| matches!(crud::read_node(tx, NodeId::new(raw)), Ok(Some(_))))
        .count()
}

#[test]
fn deterministic_dirty_read_freedom_g1a() {
    // A node staged by an UNCOMMITTED transaction is never visible to a
    // concurrent reader, and remains invisible after the writer aborts
    // (G1a — no dirty/aborted read).
    let fixture = JepsenArcqlFixture::new();
    let (mgr, crud, tenant) = (&fixture.mgr, &fixture.crud, fixture.tenant);
    let label = LabelId::new(1);

    let mut tx_w = mgr.begin(tenant);
    let nid = crud::create_node(crud, &mut tx_w, tenant, label, &PropertyData::Empty)
        .expect("stage create");

    // Concurrent reader (separate snapshot) must NOT see the
    // uncommitted node.
    let tx_r = mgr.begin(tenant);
    assert!(
        matches!(crud::read_node(&tx_r, nid), Ok(None)),
        "uncommitted CREATE must not be dirty-read"
    );
    drop(tx_r);

    // Abort the writer; the node never becomes durable.
    crud.discard_pending(tx_w.id());
    crud.discard_pending_installs(tx_w.id());
    drop(tx_w);

    let tx_r2 = mgr.begin(tenant);
    assert!(
        matches!(crud::read_node(&tx_r2, nid), Ok(None)),
        "aborted CREATE must never be visible (G1a)"
    );
}

#[test]
fn deterministic_snapshot_read_stability() {
    // A reader pinned at snapshot S does NOT observe a CREATE that
    // commits AFTER S (snapshot stability — the core SI read property);
    // a fresh reader does.
    let fixture = JepsenArcqlFixture::new();
    let (mgr, crud, tenant) = (&fixture.mgr, &fixture.crud, fixture.tenant);
    let label = LabelId::new(1);

    // Long-lived reader pins snapshot S over an empty graph.
    let tx_r = mgr.begin(tenant);
    assert_eq!(
        visible_count(&fixture, &tx_r),
        0,
        "reader starts on empty graph"
    );

    // A writer commits a node AFTER the reader's snapshot.
    let mut tx_w = mgr.begin(tenant);
    let nid = crud::create_node(crud, &mut tx_w, tenant, label, &PropertyData::Empty)
        .expect("stage create");
    crud::commit(tx_w, crud).expect("commit");

    // The pinned reader still sees 0 — snapshot stability.
    assert_eq!(
        visible_count(&fixture, &tx_r),
        0,
        "reader at snapshot S must not see a CREATE committed after S"
    );
    assert!(
        matches!(crud::read_node(&tx_r, nid), Ok(None)),
        "pinned reader must not observe the post-snapshot node"
    );
    drop(tx_r);

    // A fresh reader does see it.
    let tx_r2 = mgr.begin(tenant);
    assert_eq!(
        visible_count(&fixture, &tx_r2),
        1,
        "fresh reader sees the commit"
    );
}

#[test]
fn deterministic_write_skew_is_possible_under_si_g2() {
    // Canonical write skew (Adya 2000 §4.3): two concurrent
    // transactions both read the same snapshot (node count = K), both
    // decide the invariant "count < threshold" holds, both create a
    // (disjoint) node, both commit. Under SNAPSHOT ISOLATION both
    // succeed → final count = K+2, violating the application invariant
    // a SERIALIZABLE engine would have enforced. This pin proves the
    // surface is SI (write skew is POSSIBLE) — so the checker is
    // correct to report write skew as permitted, NOT a violation.
    let fixture = JepsenArcqlFixture::new();
    let (mgr, crud, tenant) = (&fixture.mgr, &fixture.crud, fixture.tenant);
    let label = LabelId::new(1);

    // Seed K nodes (the shared pre-state both txns read).
    let k = 3usize;
    for _ in 0..k {
        let mut t = mgr.begin(tenant);
        crud::create_node(crud, &mut t, tenant, label, &PropertyData::Empty).expect("seed");
        crud::commit(t, crud).expect("seed commit");
    }

    // Two transactions begin BEFORE either commits → identical snapshot.
    let mut tx_a = mgr.begin(tenant);
    let mut tx_b = mgr.begin(tenant);

    // Both read the same count K (write-skew predicate read).
    assert_eq!(visible_count(&fixture, &tx_a), k, "tx_a reads K");
    assert_eq!(
        visible_count(&fixture, &tx_b),
        k,
        "tx_b reads K (does not see tx_a's uncommitted work)"
    );

    // Both create a disjoint node (no WW conflict).
    crud::create_node(crud, &mut tx_a, tenant, label, &PropertyData::Empty).expect("a create");
    crud::create_node(crud, &mut tx_b, tenant, label, &PropertyData::Empty).expect("b create");

    // Both commit successfully under SI (disjoint writes).
    crud::commit(tx_a, crud).expect("tx_a commits under SI");
    crud::commit(tx_b, crud).expect("tx_b commits under SI");

    // Write skew occurred: final count = K+2, even though each tx saw
    // only K. A serializable engine would have aborted one.
    assert_eq!(
        live_visible_count(&fixture),
        k + 2,
        "SI permits write skew: both concurrent CREATEs committed (K+2)"
    );
}

// ─────────────────────────────────────────────────────────────────────
// JEPSEN_SIGKILL=1 — opt-in heavy fault variant
// ─────────────────────────────────────────────────────────────────────

/// Opt-in heavy in-process fault variant, gated by `JEPSEN_SIGKILL=1`
/// (the ADR-047 founding gate name; reserved so the future subprocess
/// variant adopts it at activation). When set, this runs a genuinely
/// heavier fault workload — elevated client count + op count + abort
/// rate — and re-verifies SI via the checker oracle.
///
/// This is NOT a no-op env-gate (cf. ADR-047 bank-transfer PR #344 R1
/// F-M3): it runs strictly heavier, genuinely different work than the
/// always-on steady-state path. The TRUE SIGKILL-during-commit
/// subprocess + WAL-recovery variant is forward-deferred: the v1.0-α
/// `CrudExecutorSubstrate` fixture uses `InMemoryPageIo`, so there is
/// no cross-process state to recover. See ADR-163 §"Forward-deferred".
/// The always-on `g1a_aborted_read_never_observed` test already
/// discharges the in-process fault-injection requirement, so this gate
/// is OPTIONAL.
#[test]
fn jepsen_sigkill_gate_runs_heavy_inprocess_fault_variant() {
    if std::env::var("JEPSEN_SIGKILL").as_deref() != Ok("1") {
        eprintln!(
            "skipping heavy fault variant (set JEPSEN_SIGKILL=1 to run); \
             always-on fault injection runs in g1a_aborted_read_never_observed"
        );
        return;
    }

    // Heavy: 8 clients × 200 ops × abort_one_in=2, combining abort
    // injection (G1a) at an elevated rate.
    let fixture = JepsenArcqlFixture::new();
    let cfg = WorkloadConfig {
        seed: 0x51_6C11_0000_0001,
        clients: 8,
        ops_per_client: 200,
        abort_one_in: 2,
        ..WorkloadConfig::default()
    };
    let history = Arc::new(OperationHistory::new());
    run_workload(
        &fixture,
        WorkloadKind::G1aAbortedRead,
        cfg,
        Arc::clone(&history),
    );

    let (verdict, _ops) = drain_and_check(&history);
    assert!(
        verdict.is_ok(),
        "heavy fault variant surfaced an SI violation: {verdict}"
    );
    let summary = verdict.summary().unwrap();
    assert!(
        summary.aborted_writes > 0,
        "heavy variant injected no aborts (vacuous)"
    );
    println!("JEPSEN_SIGKILL heavy variant: {verdict}");
}

/// W27-ν R2 fix-up (ADR-165 M1 clause-e) — synthetic-bad-history
/// self-tests that drive the PUBLIC [`ArcqlSiChecker::check`] oracle
/// with hand-built anomalous histories and assert it REPORTS the
/// specific violation. Every engine test above asserts
/// `verdict.is_ok()` against the real MVCC kernel; on their own those
/// are vacuous if the checker were a no-op (always `Ok`). These prove
/// the oracle FAILS on known-bad input — so the positive engine results
/// are load-bearing. Exact-variant assertions per Tier-B R1.
mod adversarial_history_tests {
    use arcgraph_core::{Lsn, TenantId};
    use arcgraph_storage::test_harness::jepsen::history::OpBuilder;
    use bytes::Bytes;

    use crate::common::checker::{ArcqlSiChecker, ArcqlViolation};
    use crate::common::{POISONED_READ_MARKER, SCAN_SENTINEL_KEY};

    fn lsn(n: u64) -> Lsn {
        Lsn::new(n)
    }

    fn val(b: &'static [u8]) -> Option<Bytes> {
        Some(Bytes::from_static(b))
    }

    /// G1a (aborted read): a committed MATCH observes a node id burned
    /// by an ABORTED CREATE. The checker MUST report the specific
    /// `ArcqlViolation::AbortedReadObserved` witness (Adya 2000 §4 G1a).
    #[test]
    fn check_detects_g1a_aborted_read() {
        // Op 0: aborted CREATE of node 42 — burns id 42 into the
        // aborted-write tracking set.
        let mut writer = OpBuilder::new(0, 0, TenantId::DEFAULT, lsn(1));
        writer.intend_write(42, val(b"ghost"));
        let aborted = writer.into_aborted();

        // Op 1: committed MATCH that observes node 42 (the sentinel read
        // marks it a MATCH; the node-42 read is the illegal observation).
        let mut reader = OpBuilder::new(1, 5, TenantId::DEFAULT, lsn(10));
        reader.observe_read(SCAN_SENTINEL_KEY, val(b"present"));
        reader.observe_read(42, val(b"ghost"));
        let match_op = reader.into_committed(lsn(11));

        let verdict = ArcqlSiChecker::new().check(&[aborted, match_op]);
        assert!(!verdict.is_ok(), "checker must FAIL on a G1a history");
        let violations = verdict.violations().expect("violations on the bad path");
        assert!(
            violations.iter().any(|v| matches!(
                v,
                ArcqlViolation::AbortedReadObserved {
                    reader_client: 1,
                    reader_op: 5,
                    node: 42,
                }
            )),
            "expected G1a AbortedReadObserved {{client 1, op 5, node 42}}; got {violations:?}"
        );
    }

    /// G0 (dirty / torn write): a MATCH observes a torn / unreadable
    /// node record (the poisoned-read marker). The checker MUST report
    /// the specific `ArcqlViolation::TornRead` witness.
    ///
    /// (R1 named this "G0 TornNode"; the implemented variant is
    /// `TornRead` — asserted exactly here.)
    #[test]
    fn check_detects_g0_torn_read() {
        let mut op = OpBuilder::new(2, 7, TenantId::DEFAULT, lsn(5));
        op.observe_read(SCAN_SENTINEL_KEY, val(b"present"));
        op.observe_read(POISONED_READ_MARKER, None); // torn / unreadable record
        let torn = op.into_committed(lsn(6));

        let verdict = ArcqlSiChecker::new().check(&[torn]);
        assert!(
            !verdict.is_ok(),
            "checker must FAIL on a G0 torn-read history"
        );
        let violations = verdict.violations().expect("violations on the bad path");
        assert!(
            violations.iter().any(|v| matches!(
                v,
                ArcqlViolation::TornRead {
                    client_id: 2,
                    op_id: 7,
                }
            )),
            "expected G0 TornRead {{client 2, op 7}}; got {violations:?}"
        );
    }
}
