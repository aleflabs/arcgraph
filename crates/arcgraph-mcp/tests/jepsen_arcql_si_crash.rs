//! S7d-1 / ADR-182 (#555) — durable Jepsen-ArcQL fixture + recover()
//! smoke.
//!
//! This is the FOUNDATION slice of the MERGE-atomicity-under-crash
//! Jepsen extension (PE-555 §6 scope B). It proves the durable fixture
//! — a real on-disk [`arcgraph_storage::io::PosixPageIo`] over
//! `<dir>/pages.db` + a real [`arcgraph_storage::wal::WalWriter`] at
//! [`arcgraph_core::DurabilityTier::Strict`] (fsync-before-ack), wired
//! through the SAME [`arcgraph_storage::router::MultiTenantRouter`] +
//! [`arcgraph_mcp::storage::CrudExecutorSubstrate`] the in-memory
//! [`common::JepsenArcqlFixture`] uses — is REAL, not vacuous: an ArcQL
//! CREATE that commits (fsync) survives a `recover()` round-trip.
//!
//! # Why this slice exists (the un-block)
//!
//! Scope B was v1.1-blocked because the in-memory `JepsenArcqlFixture`
//! uses `InMemoryPageIo` + `CrudStore::new()` (no WAL) — a SIGKILL'd
//! subprocess loses ALL state, so a crash-atomicity checker over it
//! would be vacuous (PE-532 §1). ADR-183 (#665, merged) retired that
//! block by shipping a fully-wired durable substrate (`PosixPageIo` +
//! `WalWriter` + `recover_from_wal`) reachable at the `CrudStore`/ArcQL
//! level. The durable fixture (`common::DurableJepsenArcqlFixture`) is
//! the first consumer of that surface in this test tree; this smoke is
//! its non-vacuity proof.
//!
//! # What this file contains (S7d-1/2/3/4 — the full slice)
//!
//! - **S7d-1** (`durable_*` / `fresh_*` smokes): the durable fixture +
//!   `recover()` round-trip non-vacuity proof.
//! - **S7d-4** (`crash_adversarial_history_tests`): the in-process
//!   non-vacuity gate — synthetic-bad-history self-tests proving the
//!   recovery-reconciliation predicate DETECTS a planted torn MERGE
//!   (+ acked-commit-loss, phantom-aborted) and PASSES atomic / aborted /
//!   past-watermark recovered states. Pure in-process, default gauntlet.
//! - **S7d-2** (`sigkill_merge_workload`, THIS slice): the live
//!   SIGKILL-during-MERGE subprocess workload — forks a child that runs
//!   the crud-tier write-set a `MERGE (a)-[:R]->(b)` lowers to (node a +
//!   node b + edge in one `crud::commit`) at [`DurabilityTier::Strict`],
//!   SIGKILLs it mid-commit, recovers over the same on-disk WAL+pages, and
//!   runs the
//!   real [`common::checker::ArcqlSiChecker::reconcile_arcql_pending_with_recovery`]
//!   predicate against the REAL recovered state. Behind a panic-by-default
//!   `JEPSEN_SIGKILL=1` / `ARCGRAPH_JEPSEN_SIGKILL_SKIP_OK=1` gate.
//! - **S7d-3** (the predicate + `ArcqlViolation::{PartialMergeCommit,
//!   AckedCommitLoss, PhantomCommit}` variants) lives in
//!   `common::checker` (consumed by both S7d-2 and S7d-4).
//!
//! See PE-555 §6 for the full slice plan and ADR-182 for the design.
//!
//! # Env-gate posture (two tiers)
//!
//! The S7d-1 smokes + S7d-4 self-tests are hermetic + pure-in-process (a
//! `tempfile::TempDir`, explicit deterministic inputs, NO subprocess), and
//! real on-disk `fsync` over a tempdir is a baseline POSIX capability the
//! existing durable suites already depend on in the default gauntlet
//! (`arcgraph-storage` `durability_tier_strict` /
//! `wal_commit_bundle_crash_atomicity`; the storage crash harness
//! `write_op_chaos_smoke`). They RUN by default with NO skip-gate.
//!
//! The S7d-2 live SIGKILL test (`sigkill_merge_workload`) forks a real
//! subprocess + delivers a real signal 9, so it is `#[ignore]`'d off the
//! default gauntlet and gated **panic-by-default** behind `JEPSEN_SIGKILL=1`
//! (ADR-163 §FD-3 reserved this name) with an explicit
//! `ARCGRAPH_JEPSEN_SIGKILL_SKIP_OK=1` opt-out — mirroring the K-3
//! subprocess gate posture (`ARCGRAPH_K3_SIGKILL_REBUILD_SKIP_OK`,
//! `crates/arcgraph-storage/tests/k3_sigkill_during_rebuild.rs`) exactly,
//! per `feedback_test_env_gate_panic_by_default.md` + W25-MFI-2
//! template-sync (`SPAWN_PROMPT_PREAMBLE.md` addendum 19 +
//! `docs/testing-strategy.md` §Env-gating, synced in the SAME commit).
//!
//! Run (default gauntlet — smokes + self-tests):
//!   cargo test -p arcgraph-mcp --test jepsen_arcql_si_crash -- --nocapture
//! Run (the live fault):
//!   JEPSEN_SIGKILL=1 cargo test -p arcgraph-mcp --test jepsen_arcql_si_crash \
//!     -- --ignored --nocapture

#[path = "jepsen_arcql_common/mod.rs"]
mod common;

use arcgraph_core::{LabelId, NodeId, PartitionId, TenantId};
use arcgraph_storage::crud::{self, PropertyData};

use arcgraph_query::executor::{ExecutionContext, Pipeline};
use arcgraph_query::logical_plan::{LogicalPlan, LogicalPlanLoweringVisitor};
use arcgraph_query::semantic::{
    BindingVisitor, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};
use arcgraph_query::{Statement, parse};

use common::{DurableJepsenArcqlFixture, DurableJepsenWorkspace, live_visible_count_durable};

// ─────────────────────────────────────────────────────────────────────
// End-to-end ArcQL drive helpers (parse → bind → type-check →
// cross-substrate → lower → Pipeline::build → next_batch). Mirrors the
// read-side `jepsen_arcql_si_read.rs` `lower` / `execute_row_count`
// helpers so the durable smoke exercises the LITERAL ArcQL operator
// surface against the durable substrate, not a private side-channel.
// ─────────────────────────────────────────────────────────────────────

/// Lower an ArcQL query string to a `LogicalPlan` through the full
/// front-end. (The binder uses a `StubCatalogProvider`; the *substrate*
/// at execution time is the real durable `CrudExecutorSubstrate`.)
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

/// Execute a plan to EOS against the durable substrate; return the row
/// count emitted.
fn execute_row_count(
    plan: &LogicalPlan,
    ctx: &ExecutionContext,
    fixture: &DurableJepsenArcqlFixture,
) -> usize {
    let mut op = Pipeline::build(plan).expect("pipeline build OK");
    let mut rows = 0usize;
    loop {
        let batch = op
            .next_batch(ctx, &fixture.substrate)
            .expect("next_batch OK");
        if batch.is_empty() {
            break;
        }
        rows += batch.row_count();
    }
    rows
}

// ─────────────────────────────────────────────────────────────────────
// S7d-1 recover() smoke — the `## Active verification` (ADR-133) for the
// slice: load a real durable fixture, drive a real ArcQL CREATE that
// fsyncs, recover from the WAL, and assert the node survives.
// ─────────────────────────────────────────────────────────────────────

/// The load-bearing non-vacuity proof for the durable fixture: an ArcQL
/// CREATE that commits at `DurabilityTier::Strict` (fsync-before-ack)
/// SURVIVES a `recover()` round-trip (re-open `pages.db` +
/// `recover_from_wal` into a fresh `CrudStore`).
///
/// If this passed against a non-durable fixture (the in-memory one), the
/// recovered store would be EMPTY and the assertion would fail — so a
/// green run is evidence the fixture's durable substrate is real, which
/// is the precondition for the S7d-2/3/4 crash-atomicity checker to be
/// non-vacuous.
#[test]
fn durable_create_survives_recover_via_arcql() {
    let ws = DurableJepsenWorkspace::new();

    // ── Phase 1: build the durable stack, CREATE a node via the LITERAL
    //    ArcQL executor surface, then close the WAL (graceful in-process
    //    crash proxy: the writer drains its fsync queue on shutdown, so
    //    the acked Strict commit is on disk before recovery re-opens).
    let created_count = {
        let mut fixture = DurableJepsenArcqlFixture::build(ws.data_dir());
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);

        // CREATE (n) RETURN n → CreateNodeOp → substrate.create_node →
        // begin → crud::create_node → crud::commit (Strict ⇒ fsync).
        let create_plan = lower("CREATE (n) RETURN n");
        let rows = execute_row_count(&create_plan, &ctx, &fixture);
        assert_eq!(rows, 1, "CREATE emits exactly one row");

        // Pre-crash: MATCH sees it live in the current process.
        let match_plan = lower("MATCH (n) RETURN n");
        let live = execute_row_count(&match_plan, &ctx, &fixture);
        assert_eq!(live, 1, "MATCH sees the committed CREATE pre-crash");
        assert_eq!(
            live_visible_count_durable(&fixture),
            1,
            "crud-tier cross-check agrees pre-crash"
        );

        // Drain + drop the live WAL writer, then drop the whole stack so
        // the WAL dir is closed for re-open by `recover`.
        fixture.shutdown_wal();
        live
    };
    assert_eq!(created_count, 1);

    // ── Phase 2: recover from the SAME data dir (re-open pages.db +
    //    recover_from_wal into a fresh CrudStore) and assert the node
    //    SURVIVED — the whole point of the durable fixture.
    let recovered = DurableJepsenArcqlFixture::recover(ws.data_dir());
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);

    // MATCH through the executor against the RECOVERED substrate.
    let match_plan = lower("MATCH (n) RETURN n");
    let survived = execute_row_count(&match_plan, &ctx, &recovered);
    assert_eq!(
        survived, 1,
        "the Strict-committed CREATE MUST survive recovery (durable fixture is real, not vacuous)"
    );

    // crud-tier cross-check: the recovered store reads node id 1 live.
    assert_eq!(
        live_visible_count_durable(&recovered),
        1,
        "crud-tier cross-check agrees post-recovery"
    );
    let tx = recovered.mgr().begin(recovered.tenant);
    let node = crud::read_node(&tx, NodeId::new(1)).expect("read_node OK");
    assert!(
        node.is_some(),
        "node id 1 (the recovered CREATE) reads back live post-recovery"
    );
}

/// Companion: the same Strict-CREATE-then-recover round-trip, but for a
/// node created via the crud-tier path with an explicit label (the path
/// the S7d-2 SIGKILL workload will record into `OperationHistory`). This
/// pins that the durable fixture's `crud` handle (not just the executor
/// substrate) routes through the same durable WAL, so the follow-up
/// workload can drive `crud::create_node` directly and still be
/// recovery-checkable.
#[test]
fn durable_crud_tier_create_survives_recover() {
    let ws = DurableJepsenWorkspace::new();

    let id = {
        let mut fixture = DurableJepsenArcqlFixture::build(ws.data_dir());
        let mut tx = fixture.mgr().begin(fixture.tenant);
        let id = crud::create_node(
            fixture.crud(),
            &mut tx,
            fixture.tenant,
            LabelId::new(7),
            &PropertyData::Empty,
        )
        .expect("crud create_node");
        // Strict tier ⇒ commit fsyncs before returning the LSN.
        let _lsn = crud::commit(tx, fixture.crud()).expect("crud commit (Strict ⇒ fsync)");
        fixture.shutdown_wal();
        id
    };

    let recovered = DurableJepsenArcqlFixture::recover(ws.data_dir());
    let tx = recovered.mgr().begin(recovered.tenant);
    let rec = crud::read_node(&tx, id)
        .expect("read_node OK")
        .expect("the Strict-committed crud-tier node survives recovery");
    assert_eq!(
        rec.label_id, 7,
        "the recovered node retains its label (full record survived, not a torn fragment)"
    );
}

/// Negative-control hermeticity pin: a FRESH durable fixture over a NEW
/// workspace (no prior commits) recovers to an EMPTY store. This guards
/// against a fixture whose `recover()` spuriously resurrects state from a
/// shared/leaked dir — i.e. it proves the durable round-trip's
/// "survives" assertions above are load-bearing (a node only survives
/// BECAUSE it was committed, not because recover always reports nodes).
#[test]
fn fresh_durable_fixture_recovers_empty() {
    let ws = DurableJepsenWorkspace::new();
    // Build + immediately drop with NO writes, closing the (empty) WAL.
    {
        let mut fixture = DurableJepsenArcqlFixture::build(ws.data_dir());
        fixture.shutdown_wal();
    }
    let recovered = DurableJepsenArcqlFixture::recover(ws.data_dir());
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let plan = lower("MATCH (n) RETURN n");
    let rows = execute_row_count(&plan, &ctx, &recovered);
    assert_eq!(
        rows, 0,
        "a fresh durable fixture with no commits recovers to an empty store"
    );
    assert_eq!(live_visible_count_durable(&recovered), 0);
}

// ─────────────────────────────────────────────────────────────────────
// S7d-4 — adversarial-oracle self-tests (THE NON-VACUITY GATE)
// ─────────────────────────────────────────────────────────────────────

/// Synthetic-bad-history self-tests that prove the recovery-reconciliation
/// predicate ([`ArcqlSiChecker::reconcile_arcql_pending_with_recovery`],
/// S7d-3 / ADR-182 §2.2) actually **DETECTS a planted torn MERGE**. This
/// is the **load-bearing gate** for the whole S7d slice (PE-555 §4 / §6
/// row S7d-4; `feedback_review_oracle_relaxations.md` + the W27-ν R2 + the
/// #684 / #686 R1 precedent — every oracle on this track ships with a
/// mutation/detection proof): the SIGKILL-during-MERGE active verification
/// (S7d-2, the follow-up PR) will assert `verdict.is_ok()` against a real
/// recovered history, and on its own that assertion is **vacuous if the
/// predicate were a no-op that always returned `Ok`**. These tests run
/// PURE IN-PROCESS (hand-built `OperationHistory` + a hand-built `S_rec`
/// — NO subprocess, NO real SIGKILL) so they execute in the DEFAULT
/// gauntlet and prove the predicate's correctness on every CI run, even
/// when the OS-crash path (S7d-2) is cron-gated.
///
/// Each test plants an EXACT torn (or atomic) recovered state and asserts
/// the EXACT [`ArcqlViolation::PartialMergeCommit`] variant (or its
/// absence) per Tier-B R1 exact-variant discipline. Mirrors the read-side
/// `jepsen_arcql_si_read.rs::adversarial_history_tests` shape.
mod crash_adversarial_history_tests {
    use std::collections::HashSet;

    use arcgraph_core::{Lsn, TenantId};
    use arcgraph_storage::test_harness::jepsen::history::OpBuilder;
    use bytes::Bytes;

    use crate::common::checker::{ArcqlSiChecker, ArcqlViolation, edge_key};
    use crate::common::present_marker;

    fn lsn(n: u64) -> Lsn {
        Lsn::new(n)
    }

    /// The recovery watermark the self-tests reconcile against (the
    /// `committed_fsync_watermark` boundary, ADR-034 §Slice-B). The
    /// hand-built histories control both each op's `commit_lsn` and this
    /// watermark directly (per ADR-182 §2.2 "Watermark source"); a commit
    /// at lsn ≤ `WATERMARK` is acked-durable (bullet 1), a commit at lsn >
    /// `WATERMARK` lost the ack race (bullet 2). `committed_merge_a_r_b`
    /// commits at lsn 10, comfortably ≤ this, so it is acked-durable.
    const WATERMARK: u64 = 100;

    fn watermark() -> Lsn {
        lsn(WATERMARK)
    }

    /// Marker value for a "this graph element exists" write. Same shape as
    /// the live workload's [`present_marker`]; the predicate reasons over
    /// key PRESENCE, not value, so any `Some(_)` marker tags a creation.
    fn marker() -> Option<Bytes> {
        Some(present_marker())
    }

    /// Synthetic node keys (small ids, low half of the u64 space — exactly
    /// where `NodeId::raw()` lives) and the canonical edge key for the
    /// MERGE `(a)-[:R]->(b)` under test. `R = 9` is an arbitrary type id.
    const NODE_A: u64 = 1;
    const NODE_B: u64 = 2;
    fn edge_a_r_b() -> u64 {
        edge_key(NODE_A, /* ty = R */ 9, NODE_B)
    }

    /// Build a committed MERGE op whose creation set is exactly
    /// `{ NODE_A, NODE_B, edge_key(A,R,B) }` — the three writes a
    /// `MERGE (a)-[:R]->(b)` lowers to. `commit_lsn = 10` is ≤ [`WATERMARK`]
    /// (= 100), so this op is **acked-durable** (ADR-182 §2.2 bullet 1) and
    /// the predicate requires ALL three writes present in a fully-durable
    /// recovery.
    fn committed_merge_a_r_b() -> arcgraph_storage::test_harness::jepsen::history::RecordedOp {
        let mut b = OpBuilder::new(0, 1, TenantId::DEFAULT, lsn(1));
        b.intend_write(NODE_A, marker());
        b.intend_write(NODE_B, marker());
        b.intend_write(edge_a_r_b(), marker());
        b.into_committed(lsn(10))
    }

    /// Build a committed MERGE op whose `commit_lsn = 200` is **strictly
    /// above** [`WATERMARK`] (= 100) — it committed but the commit record was
    /// past the `committed_fsync_watermark` at the crash, so it **lost the
    /// ack race** (it was never acked durable). ADR-182 §2.2 bullet 2: a
    /// past-watermark commit is legitimately as-if-absent post-recovery
    /// (none of its writes may survive; full absence is legal, any survivor
    /// is a phantom). Its creation set is the same `{A, B, edge}`.
    fn committed_merge_past_watermark()
    -> arcgraph_storage::test_harness::jepsen::history::RecordedOp {
        let mut b = OpBuilder::new(0, 2, TenantId::DEFAULT, lsn(1));
        b.intend_write(NODE_A, marker());
        b.intend_write(NODE_B, marker());
        b.intend_write(edge_a_r_b(), marker());
        b.into_committed(lsn(200))
    }

    /// **THE inherited mandatory test** (PE-555 §4.1 / PE-532 §5): a
    /// recovered history where a MERGE persisted the node(s) but NOT the
    /// edge (a torn multi-statement commit) → the predicate MUST return
    /// `ArcqlViolation::PartialMergeCommit` naming the present node(s) +
    /// the absent edge. A no-op (always-`Ok`) predicate FAILS this test.
    #[test]
    fn check_detects_partial_merge_commit() {
        let history = vec![committed_merge_a_r_b()];
        // Torn S_rec: both endpoint nodes survived, the edge did NOT
        // (the planted half-committed MERGE).
        let recovered: HashSet<u64> = HashSet::from([NODE_A, NODE_B]);

        let verdict = ArcqlSiChecker::new().reconcile_arcql_pending_with_recovery(
            &history,
            watermark(),
            &recovered,
        );
        assert!(
            !verdict.is_ok(),
            "predicate MUST FAIL on a torn (node-present, edge-absent) recovered MERGE"
        );
        let violations = verdict.violations().expect("violations on the torn path");
        assert!(
            violations.iter().any(|v| matches!(
                v,
                ArcqlViolation::PartialMergeCommit { op: (0, 1), present, absent }
                    if present.contains(&NODE_A)
                        && present.contains(&NODE_B)
                        && absent.contains(&edge_a_r_b())
            )),
            "expected PartialMergeCommit{{op (0,1), present⊇{{A,B}}, absent∋edge}}; \
             got {violations:?}"
        );
    }

    /// **Negative control** (PE-555 §4.2 item 5; dual-vacuity guard): a
    /// fully-atomic MERGE — all three writes present in `S_rec` — MUST
    /// produce NO violation. Guards against an over-eager checker that
    /// flags *every* op (the dual vacuity hole: a constant-`Violations`
    /// stub would also "pass" the detection test above).
    #[test]
    fn check_passes_fully_committed_merge() {
        let history = vec![committed_merge_a_r_b()];
        // Atomic S_rec: node A + node B + edge ALL survived.
        let recovered: HashSet<u64> = HashSet::from([NODE_A, NODE_B, edge_a_r_b()]);

        let verdict = ArcqlSiChecker::new().reconcile_arcql_pending_with_recovery(
            &history,
            watermark(),
            &recovered,
        );
        assert!(
            verdict.is_ok(),
            "a fully-durable MERGE (all writes present) is legal; got {verdict}"
        );
    }

    /// **Negative control — past-watermark whole-op loss is legal**
    /// (PE-555 §4.2, the test-#2 "both absent because the whole op was lost"
    /// leg; ADR-182 §2.2 bullet 2): a committed MERGE whose `commit_lsn`
    /// is **strictly above the watermark** (it lost the ack race) and whose
    /// ENTIRE creation set is absent from `S_rec` MUST produce NO violation
    /// — a never-acked commit is legitimately as-if-absent. This is the
    /// dual-vacuity guard against an over-eager checker that fired on
    /// *every* fully-absent committed op (which would be wrong for a
    /// past-watermark loss).
    ///
    /// NOTE the watermark-sensitivity: the SAME "fully-absent committed
    /// MERGE" recovered state is LEGAL here (commit_lsn 200 > watermark 100)
    /// but a VIOLATION in `check_detects_acked_commit_loss` below
    /// (commit_lsn 10 ≤ watermark 100). The outcome-blind predecessor
    /// treated both as legal — the bug F-1 fixed.
    #[test]
    fn check_passes_fully_absent_past_watermark_commit() {
        let history = vec![committed_merge_past_watermark()];
        // Nothing survived — and the commit was past the watermark, so this
        // is a legitimate ack-race loss (ADR-182 §2.2 bullet 2).
        let recovered: HashSet<u64> = HashSet::new();

        let verdict = ArcqlSiChecker::new().reconcile_arcql_pending_with_recovery(
            &history,
            watermark(),
            &recovered,
        );
        assert!(
            verdict.is_ok(),
            "a fully-absent past-watermark committed MERGE (lost the ack race) is legal; \
             got {verdict}"
        );
    }

    /// **THE acked-committed-FULL-LOSS detection test** (F-1 option (a);
    /// ADR-182 §2.2 bullet 1 all-absent leg): an **acked-durable** MERGE
    /// (`Committed{commit_lsn = 10 ≤ watermark = 100}`) whose ENTIRE
    /// creation set vanished post-recovery MUST flag
    /// [`ArcqlViolation::AckedCommitLoss`] — the single most severe
    /// crash-atomicity violation (the client was told "durable", recovery
    /// says "never happened"; ADR-031 §Decision + ADR-034 §Slice-B). This
    /// is the arm the OUTCOME-BLIND predecessor was missing: it treated
    /// all-absent as always-legal, so this acked-loss was INVISIBLE. A
    /// predicate that does not branch on `op.outcome` + watermark FAILS
    /// this test (it would return `Ok`).
    #[test]
    fn check_detects_acked_commit_loss() {
        // commit_lsn 10 ≤ watermark 100 ⇒ acked-durable ⇒ MUST survive.
        let history = vec![committed_merge_a_r_b()];
        // Total loss: NOTHING survived recovery.
        let recovered: HashSet<u64> = HashSet::new();

        let verdict = ArcqlSiChecker::new().reconcile_arcql_pending_with_recovery(
            &history,
            watermark(),
            &recovered,
        );
        assert!(
            !verdict.is_ok(),
            "predicate MUST FAIL on an acked-durable MERGE (commit_lsn ≤ watermark) whose \
             entire write-set vanished post-recovery (durability loss of an acked commit)"
        );
        let violations = verdict
            .violations()
            .expect("violations on the acked-loss path");
        assert!(
            violations.iter().any(|v| matches!(
                v,
                ArcqlViolation::AckedCommitLoss { op: (0, 1), commit_lsn: 10, watermark: WATERMARK, lost }
                    if lost.contains(&NODE_A)
                        && lost.contains(&NODE_B)
                        && lost.contains(&edge_a_r_b())
            )),
            "expected AckedCommitLoss{{op (0,1), commit_lsn 10, watermark {WATERMARK}, \
             lost⊇{{A,B,edge}}}}; got {violations:?}"
        );
    }

    /// **THE phantom-committed-aborted detection test** (F-1 option (a);
    /// ADR-182 §2.2 bullet 2): an op that was **`Aborted`** (rolled back —
    /// none of its writes may survive) whose writes nonetheless APPEARED in
    /// the recovered state MUST flag [`ArcqlViolation::PhantomCommit`] —
    /// recovery resurrected state that was never durably committed (the
    /// dual of acked-commit-loss). The outcome-blind predecessor treated
    /// all-present as always-legal, so this phantom was INVISIBLE. A
    /// predicate that does not branch on `op.outcome` FAILS this test.
    #[test]
    fn check_detects_phantom_aborted_commit() {
        // An aborted MERGE — its writes MUST NOT be present post-recovery.
        let mut b = OpBuilder::new(3, 7, TenantId::DEFAULT, lsn(5));
        b.intend_write(NODE_A, marker());
        b.intend_write(NODE_B, marker());
        b.intend_write(edge_a_r_b(), marker());
        let history = vec![b.into_aborted()];
        // Phantom: the aborted op's writes appeared anyway (a recovery bug
        // that replayed a rolled-back bundle).
        let recovered: HashSet<u64> = HashSet::from([NODE_A, NODE_B, edge_a_r_b()]);

        let verdict = ArcqlSiChecker::new().reconcile_arcql_pending_with_recovery(
            &history,
            watermark(),
            &recovered,
        );
        assert!(
            !verdict.is_ok(),
            "predicate MUST FAIL on an ABORTED MERGE whose writes appeared post-recovery \
             (phantom commit of a rolled-back op)"
        );
        let violations = verdict
            .violations()
            .expect("violations on the phantom path");
        assert!(
            violations.iter().any(|v| matches!(
                v,
                ArcqlViolation::PhantomCommit { op: (3, 7), reason: "aborted", present }
                    if present.contains(&NODE_A)
                        && present.contains(&NODE_B)
                        && present.contains(&edge_a_r_b())
            )),
            "expected PhantomCommit{{op (3,7), reason \"aborted\", present⊇{{A,B,edge}}}}; \
             got {violations:?}"
        );
    }

    /// **Aborted-absent** (PE-555 §4.2 item 6): a MERGE that ABORTED
    /// pre-commit, with recovery showing NEITHER node NOR edge, MUST
    /// produce NO violation (correctly atomic-absent — the abort rolled
    /// back the whole bundle).
    #[test]
    fn check_passes_aborted_merge_absent() {
        let mut b = OpBuilder::new(1, 2, TenantId::DEFAULT, lsn(3));
        b.intend_write(NODE_A, marker());
        b.intend_write(NODE_B, marker());
        b.intend_write(edge_a_r_b(), marker());
        let aborted = b.into_aborted();
        let history = vec![aborted];
        // Aborted ⇒ rolled back ⇒ nothing in the recovered state.
        let recovered: HashSet<u64> = HashSet::new();

        let verdict = ArcqlSiChecker::new().reconcile_arcql_pending_with_recovery(
            &history,
            watermark(),
            &recovered,
        );
        assert!(
            verdict.is_ok(),
            "an aborted MERGE that left no trace is correctly atomic-absent; got {verdict}"
        );
    }

    /// **Orphan-edge** (PE-555 §4.2 item 7): the *reverse* torn case — the
    /// edge `(a,R,b)` survived but an endpoint node did NOT (an edge
    /// dangling off a non-recovered endpoint) → the predicate MUST detect
    /// `PartialMergeCommit`. Covers the other torn direction from
    /// `check_detects_partial_merge_commit`, so a predicate that only
    /// checked "node ⇒ edge" (and not the symmetric "edge ⇒ node") would
    /// FAIL here.
    #[test]
    fn check_detects_orphan_edge_without_endpoint() {
        let history = vec![committed_merge_a_r_b()];
        // Torn the other way: node A + the edge survived, node B did NOT —
        // an edge whose dst endpoint vanished in recovery.
        let recovered: HashSet<u64> = HashSet::from([NODE_A, edge_a_r_b()]);

        let verdict = ArcqlSiChecker::new().reconcile_arcql_pending_with_recovery(
            &history,
            watermark(),
            &recovered,
        );
        assert!(
            !verdict.is_ok(),
            "predicate MUST FAIL on a torn (edge-present, endpoint-absent) recovered MERGE"
        );
        let violations = verdict.violations().expect("violations on the torn path");
        assert!(
            violations.iter().any(|v| matches!(
                v,
                ArcqlViolation::PartialMergeCommit { op: (0, 1), present, absent }
                    if present.contains(&edge_a_r_b())
                        && present.contains(&NODE_A)
                        && absent.contains(&NODE_B)
            )),
            "expected PartialMergeCommit{{op (0,1), present∋edge+A, absent∋B}}; got {violations:?}"
        );
    }

    /// `edge_key` namespace pin (load-bearing for the per-key granularity
    /// claim): an edge key is ALWAYS in the high half of the u64 space
    /// (bit 63 set) — so it is disjoint from node ids (low half) — and
    /// never aliases the two reserved harness sentinels
    /// (`SCAN_SENTINEL_KEY = u64::MAX`, `POISONED_READ_MARKER = u64::MAX -
    /// 1`). If this regressed, an edge write could collide with a node
    /// write and the torn-MERGE detection above would silently degrade.
    #[test]
    fn edge_key_is_disjoint_from_node_and_sentinel_namespaces() {
        use crate::common::{POISONED_READ_MARKER, SCAN_SENTINEL_KEY};
        for &(s, t, d) in &[
            (1u64, 9u64, 2u64),
            (2, 9, 1),
            (1, 0, 1),
            (7, 7, 7),
            (0, 0, 0),
        ] {
            let k = edge_key(s, t, d);
            assert!(
                k >= (1u64 << 63),
                "edge_key({s},{t},{d}) = {k} must have bit 63 set (disjoint from node ids)"
            );
            assert_ne!(
                k, SCAN_SENTINEL_KEY,
                "edge_key must not alias the scan sentinel"
            );
            assert_ne!(
                k, POISONED_READ_MARKER,
                "edge_key must not alias the poisoned-read marker"
            );
        }
        // Distinct triples ⇒ distinct keys (direction matters: A→B ≠ B→A).
        assert_ne!(
            edge_key(NODE_A, 9, NODE_B),
            edge_key(NODE_B, 9, NODE_A),
            "edge_key must distinguish (A,R,B) from (B,R,A)"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// Subprocess router — runs in EVERY child re-exec (sorts globally first,
// `aaaa_*`) so the child dispatches the registered SIGKILL-during-MERGE
// workload BEFORE any other test body executes. Mirrors
// `k3_real_subprocess_sigkill.rs::aaaa_subprocess_dispatcher_router` +
// `k3_sigkill_during_rebuild.rs`. In the PARENT (no `K1_SUBPROCESS_WORKLOAD`
// env) `maybe_dispatch_subprocess_workload()` is a no-op, so this is inert
// in the default gauntlet; in a child it routes into the workload and
// `process::exit`s, never returning.
#[test]
fn aaaa_subprocess_dispatcher_router() {
    sigkill_merge_workload::dispatch_if_subprocess();
}

// ─────────────────────────────────────────────────────────────────────
// S7d-2 — live SIGKILL-during-MERGE subprocess workload (#555, ADR-182
// §4.3 / §6 row S7d-2). The REAL end-to-end fault: fork a child, run the
// crud-tier write-set a `MERGE (a)-[:R]->(b)` lowers to (node a + node b +
// edge in one `crud::commit`) at `DurabilityTier::Strict`, SIGKILL it
// mid-commit, recover over the SAME on-disk WAL+pages, and run the real
// `reconcile_arcql_pending_with_recovery` predicate against the REAL
// recovered state. The 8 in-process `crash_adversarial_history_tests`
// self-tests above (7 predicate self-tests + 1 namespace pin) prove the
// predicate DETECTS a torn MERGE; this slice proves the real kernel under
// real SIGKILL produces a reconcilable (never-torn) state.
// ─────────────────────────────────────────────────────────────────────

mod sigkill_merge_workload {
    use std::collections::HashSet;
    use std::path::{Path, PathBuf};
    use std::sync::Once;
    use std::time::Duration;

    use arcgraph_core::{LabelId, Lsn, NodeId, RelId, TypeId};
    use arcgraph_storage::crud::{self, PropertyData};
    use arcgraph_storage::test_harness::jepsen::history::{OpBuilder, RecordedOp};
    use arcgraph_storage::test_harness::k1::subprocess::{
        SubprocessWorkloadRegistry, WORKLOAD_CLEAN_EXIT_CODE, maybe_dispatch_subprocess_workload,
        run_with_crash_window,
    };

    use crate::common::checker::{ArcqlSiChecker, edge_key};
    use crate::common::{DurableJepsenArcqlFixture, DurableJepsenWorkspace, present_marker};

    /// Run-flag that ENABLES the live SIGKILL fault (the test is
    /// `#[ignore]`'d; this must be `=1`/`true` even under `--ignored`).
    /// ADR-163 §FD-3 reserved this exact name.
    const JEPSEN_SIGKILL_ENV: &str = "JEPSEN_SIGKILL";
    /// Explicit soft-skip opt-out (hostile/CI envs only). Anything else =
    /// PANIC when the test is invoked via `--ignored` without the run-flag.
    const SKIP_OK_ENV: &str = "ARCGRAPH_JEPSEN_SIGKILL_SKIP_OK";
    /// Registry name the child re-exec routes to.
    const WORKLOAD_NAME: &str = "jepsen_arcql_sigkill_merge_workload";

    /// Relationship type id for the `MERGE (a)-[:R]->(b)` workload. `R = 9`
    /// matches the synthetic self-tests' `edge_a_r_b()` convention so the
    /// edge-key namespace is identical across the in-process and live paths.
    const REL_TYPE_RAW: u32 = 9;
    /// Node labels for the two endpoints (distinct so a label drift is
    /// visible; the predicate reasons over key presence, not label).
    const LABEL_A: u32 = 1;
    const LABEL_B: u32 = 2;

    /// Child loop bound. The total *sleep* budget (`MAX_MERGES ×
    /// SLEEP_BETWEEN_MERGES`) is wall-clock 6 s — far longer than
    /// [`CRASH_AFTER`] — so the SIGKILL ALWAYS lands mid-loop (the child
    /// cannot finish first regardless of host speed), guaranteeing
    /// `!exited_cleanly()`. If the kill syscall ever failed, the child
    /// still self-terminates within ~6 s rather than hanging.
    const MAX_MERGES: u64 = 3000;
    /// Per-merge pause so the crash window has many commit boundaries to
    /// land between (and to bound the WAL/ledger growth rate).
    const SLEEP_BETWEEN_MERGES: Duration = Duration::from_millis(2);
    /// Parent's crash window: SIGKILL the child this long after spawn.
    /// 500 ms ≫ a single Strict fsync (~ms), so dozens–hundreds of merges
    /// commit + fsync before the kill — `survived_full ≥ 1` with margin
    /// even under CI contention.
    const CRASH_AFTER: Duration = Duration::from_millis(500);

    fn jepsen_sigkill_enabled() -> bool {
        std::env::var(JEPSEN_SIGKILL_ENV)
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    }

    /// One MERGE op persisted to the sidecar ledger so the parent can
    /// reconstruct `H_pre` after the child is SIGKILL'd (the child's
    /// in-memory `OperationHistory` dies with it — no Drop runs). Five
    /// `u64` CSV fields; the rel type is the [`REL_TYPE_RAW`] const (both
    /// child + parent are the same binary, so it need not be persisted).
    #[derive(Debug, Clone, Copy)]
    struct MergeRecord {
        op_id: u64,
        commit_lsn: u64,
        node_a: u64,
        node_b: u64,
        rel_id: u64,
    }

    fn merge_ledger_path(data_dir: &Path) -> PathBuf {
        data_dir.join("merge_history.csv")
    }

    /// Append one MERGE record to `file` and `sync_data()`. Fsync-per-append
    /// is load-bearing: a SIGKILL'd process flushes no buffered IO, so the
    /// parent would read a truncated ledger and undercount the pre-crash
    /// commits otherwise — the exact `PreCrashLedger` discipline
    /// (`subprocess.rs:426`).
    fn record_merge(file: &mut std::fs::File, rec: MergeRecord) -> std::io::Result<()> {
        use std::io::Write as _;
        let line = format!(
            "{},{},{},{},{}\n",
            rec.op_id, rec.commit_lsn, rec.node_a, rec.node_b, rec.rel_id
        );
        file.write_all(line.as_bytes())?;
        file.sync_data()
    }

    /// Read the sidecar ledger with torn-trailing-row tolerance: a SIGKILL
    /// mid-`write_all` (or before `sync_data` returns) can leave a partial
    /// LAST line, which is dropped (the durable prefix is returned). A
    /// malformed NON-last line is real corruption and panics — an
    /// fsync-per-append ledger never has mid-file corruption. Mirrors
    /// `PreCrashLedger::read_all`'s codex-B-2 contract (`subprocess.rs:613`).
    fn read_merge_ledger(path: &Path) -> Vec<MergeRecord> {
        let bytes = std::fs::read(path).expect("read merge ledger");
        let s = std::str::from_utf8(&bytes).expect("merge ledger is valid utf-8");
        let lines: Vec<&str> = s.lines().collect();
        let total = lines.len();
        let mut out = Vec::with_capacity(total);
        for (idx, line) in lines.iter().enumerate() {
            let is_last = idx + 1 == total;
            let parts: Vec<&str> = line.split(',').collect();
            let parsed = (|| -> Option<MergeRecord> {
                if parts.len() != 5 {
                    return None;
                }
                Some(MergeRecord {
                    op_id: parts[0].parse().ok()?,
                    commit_lsn: parts[1].parse().ok()?,
                    node_a: parts[2].parse().ok()?,
                    node_b: parts[3].parse().ok()?,
                    rel_id: parts[4].parse().ok()?,
                })
            })();
            match parsed {
                Some(r) => out.push(r),
                // Torn trailing row (SIGKILL mid-write) — tolerated.
                None if is_last => break,
                None => panic!(
                    "merge ledger mid-file corruption at line {}: {line:?} \
                     (an fsync-per-append ledger must never corrupt mid-file)",
                    idx + 1
                ),
            }
        }
        out
    }

    /// CHILD-side workload (registered into the K-1 subprocess registry).
    /// Builds the durable fixture over `arg` (= the parent's data dir), then
    /// runs a long stream of the crud-tier write-set a `MERGE (a)-[:R]->(b)`
    /// lowers to — each ONE transaction creating node a, node b, AND the
    /// edge, committed by a SINGLE `crud::commit` (ONE `CommitBundle`, the
    /// atomic unit ADR-031
    /// §Decision guarantees) at `DurabilityTier::Strict` (fsync-before-ack,
    /// the DEFAULT-tenant tier). Every committed merge is persisted (fsync)
    /// to the sidecar ledger so the parent can reconstruct `H_pre` after the
    /// SIGKILL. The loop's sleep budget (≫ the crash window) guarantees the
    /// SIGKILL lands mid-loop.
    fn merge_workload(arg: &str) -> i32 {
        let data_dir = PathBuf::from(arg);
        let mut fixture = DurableJepsenArcqlFixture::build(&data_dir);
        let mut ledger = match std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .read(true)
            .open(merge_ledger_path(&data_dir))
        {
            Ok(f) => f,
            Err(e) => {
                eprintln!("sigkill-merge child: cannot create merge ledger: {e}");
                return 98;
            }
        };

        let tenant = fixture.tenant;
        let ty = TypeId::new(REL_TYPE_RAW);
        for op_id in 0..MAX_MERGES {
            let mut tx = fixture.mgr().begin(tenant);
            // MERGE (a)-[:R]->(b): node a + node b + edge in ONE tx.
            let a = match crud::create_node(
                fixture.crud(),
                &mut tx,
                tenant,
                LabelId::new(LABEL_A),
                &PropertyData::Empty,
            ) {
                Ok(id) => id,
                Err(e) => {
                    eprintln!("sigkill-merge child: create_node a failed: {e:?}");
                    continue;
                }
            };
            let b = match crud::create_node(
                fixture.crud(),
                &mut tx,
                tenant,
                LabelId::new(LABEL_B),
                &PropertyData::Empty,
            ) {
                Ok(id) => id,
                Err(e) => {
                    eprintln!("sigkill-merge child: create_node b failed: {e:?}");
                    continue;
                }
            };
            let rel = match crud::create_rel(
                fixture.crud(),
                &mut tx,
                tenant,
                a,
                b,
                ty,
                &PropertyData::Empty,
            ) {
                Ok(id) => id,
                Err(e) => {
                    eprintln!("sigkill-merge child: create_rel failed: {e:?}");
                    continue;
                }
            };
            // ONE commit ⇒ ONE bundle (Strict ⇒ fsync before the LSN
            // returns). Record + fsync to the ledger AFTER the ack so an
            // entry in the parent's ledger always denotes a durable commit.
            match crud::commit(tx, fixture.crud()) {
                Ok(lsn) => {
                    if let Err(e) = record_merge(
                        &mut ledger,
                        MergeRecord {
                            op_id,
                            commit_lsn: lsn.raw(),
                            node_a: a.raw(),
                            node_b: b.raw(),
                            rel_id: rel.raw(),
                        },
                    ) {
                        eprintln!("sigkill-merge child: ledger record failed: {e}");
                    }
                }
                Err(e) => eprintln!("sigkill-merge child: commit failed: {e:?}"),
            }
            std::thread::sleep(SLEEP_BETWEEN_MERGES);
        }

        // Loop finished before SIGKILL — the crash window was too long
        // (harness-tuning signal). Graceful shutdown so the parent can
        // still recover; the parent asserts `!exited_cleanly()`.
        fixture.shutdown_wal();
        WORKLOAD_CLEAN_EXIT_CODE
    }

    fn register_workload_once() {
        static ONCE: Once = Once::new();
        ONCE.call_once(|| {
            SubprocessWorkloadRegistry::register(WORKLOAD_NAME, merge_workload);
        });
    }

    /// Register the workload + dispatch if this process is a child re-exec.
    /// Called from the top-level `aaaa_subprocess_dispatcher_router` (so it
    /// runs first in EVERY child) and as the first line of the live test.
    pub fn dispatch_if_subprocess() {
        register_workload_once();
        maybe_dispatch_subprocess_workload();
    }

    /// THE live SIGKILL-during-MERGE active verification (W26-MFI-4 /
    /// ADR-133; PE-555 §4.3). `#[ignore]`'d off the default gauntlet;
    /// panic-by-default behind `JEPSEN_SIGKILL=1` with an explicit
    /// `ARCGRAPH_JEPSEN_SIGKILL_SKIP_OK=1` opt-out (mirrors
    /// `k3_sigkill_during_rebuild.rs`).
    ///
    /// Run: `JEPSEN_SIGKILL=1 cargo test -p arcgraph-mcp --test
    /// jepsen_arcql_si_crash -- --ignored --nocapture`.
    #[test]
    #[ignore = "S7d-2 live SIGKILL-during-MERGE subprocess workload (#555, \
                ADR-182) — gated by JEPSEN_SIGKILL=1; forks a child, runs the \
                crud-tier write-set a MERGE (a)-[:R]->(b) lowers to at Strict, \
                SIGKILLs mid-commit, recovers, reconciles. Run: \
                JEPSEN_SIGKILL=1 cargo test -p arcgraph-mcp --test \
                jepsen_arcql_si_crash -- --ignored --nocapture"]
    fn live_sigkill_during_merge_reconciles() {
        dispatch_if_subprocess();

        // Panic-by-default env-gate (W12δ HIGH-1 /
        // feedback_test_env_gate_panic_by_default.md). When invoked via
        // `--ignored` without JEPSEN_SIGKILL=1 it PANICS (a missing fault
        // campaign must be LOUD) unless the operator explicitly opts into a
        // soft-skip via ARCGRAPH_JEPSEN_SIGKILL_SKIP_OK=1. EXACT posture of
        // `k3_sigkill_during_rebuild_subprocess`.
        if !jepsen_sigkill_enabled() {
            if std::env::var(SKIP_OK_ENV).is_ok() {
                eprintln!(
                    "live_sigkill_during_merge_reconciles: SKIPPING (opt-in via \
                     {SKIP_OK_ENV}=1) — set {JEPSEN_SIGKILL_ENV}=1 to run the live \
                     SIGKILL-during-MERGE subprocess workload instead"
                );
                return;
            }
            panic!(
                "live_sigkill_during_merge_reconciles: required env flag \
                 {JEPSEN_SIGKILL_ENV}=1 not set. This test is `#[ignore]`'d off the \
                 default gauntlet; when invoked via `--ignored`, {JEPSEN_SIGKILL_ENV}=1 \
                 must be set so the live SIGKILL-during-MERGE fault actually fires. \
                 Set {JEPSEN_SIGKILL_ENV}=1 to run, or {SKIP_OK_ENV}=1 to opt into a \
                 soft-skip (hostile/CI envs only). Soft-skipping silently is the W12δ \
                 HIGH-1 bug class (feedback_test_env_gate_panic_by_default.md)."
            );
        }

        let ws = DurableJepsenWorkspace::new();

        // ── Phase 1: fork the child running the MERGE workload; let it
        //    commit a stream of Strict merges; SIGKILL it mid-commit.
        let record = run_with_crash_window(WORKLOAD_NAME, ws.data_dir(), CRASH_AFTER)
            .expect("crash-window run");
        eprintln!(
            "live_sigkill_during_merge: elapsed_to_kill={:?} kill_succeeded={} \
             sigkilled={} exited_cleanly={} exit_status={:?}",
            record.elapsed_to_kill,
            record.kill_succeeded,
            record.was_sigkilled(),
            record.exited_cleanly(),
            record.exit_status,
        );

        // Non-vacuity gate 1: the child was REALLY killed by signal 9 — the
        // load-bearing "no Drop, no flush-on-shutdown" crash semantics
        // (subprocess.rs:14-17). A clean exit means the fault never fired.
        assert!(
            !record.exited_cleanly(),
            "child completed all {MAX_MERGES} merges before SIGKILL — CRASH_AFTER \
             ({CRASH_AFTER:?}) too long OR MAX_MERGES too low; the fault never fired"
        );
        assert!(
            record.kill_succeeded,
            "SIGKILL syscall failed — child gone before the crash window"
        );
        #[cfg(unix)]
        assert!(
            record.was_sigkilled(),
            "child must report a SIGKILL signal-exit on Unix; got {:?}",
            record.exit_status
        );

        // ── Phase 2: recover over the SAME on-disk WAL+pages the killed
        //    child wrote, capturing the recovery watermark (= the
        //    committed_fsync_watermark boundary, ADR-034 §Slice-B / ADR-182
        //    §2.2). Exactly the commits at or below it survived.
        let (recovered, watermark) =
            DurableJepsenArcqlFixture::recover_with_watermark(ws.data_dir());

        // ── Phase 3: reconstruct H_pre from the child's fsync'd ledger +
        //    read the REAL recovered state S_rec back through the crud tier.
        let pre = read_merge_ledger(&merge_ledger_path(ws.data_dir()));

        // Non-vacuity gate 2: the child acked ≥1 Strict merge pre-SIGKILL
        // (otherwise the predicate runs over an empty history — vacuous).
        assert!(
            !pre.is_empty(),
            "child acked 0 merges pre-SIGKILL — the recovered history is empty \
             (vacuous); CRASH_AFTER too short or the workload never started"
        );

        let tenant = recovered.tenant;
        let ty_raw = u64::from(REL_TYPE_RAW);
        let mut history: Vec<RecordedOp> = Vec::with_capacity(pre.len());
        let mut recovered_keys: HashSet<u64> = HashSet::new();
        let mut survived_full = 0usize;

        let tx = recovered.mgr().begin(tenant);
        for rec in &pre {
            let edge = edge_key(rec.node_a, ty_raw, rec.node_b);

            // H_pre op: a committed MERGE whose creation set is {a, b, edge}
            // (the three writes the bundle made), tagged with the REAL
            // commit LSN the child observed.
            let mut b = OpBuilder::new(0, rec.op_id, tenant, Lsn::ZERO);
            b.intend_write(rec.node_a, Some(present_marker()));
            b.intend_write(rec.node_b, Some(present_marker()));
            b.intend_write(edge, Some(present_marker()));
            history.push(b.into_committed(Lsn::new(rec.commit_lsn)));

            // S_rec: read each element back from the REAL recovered store.
            // A node key is its `NodeId::raw()`; an edge key is
            // `edge_key(a, R, b)` (present iff the rel record recovered) —
            // the SAME per-key identities H_pre uses, so they reconcile.
            let a_present =
                crud::read_node_with_store(&recovered.crud, &tx, NodeId::new(rec.node_a))
                    .expect("read_node_with_store a")
                    .is_some();
            let b_present =
                crud::read_node_with_store(&recovered.crud, &tx, NodeId::new(rec.node_b))
                    .expect("read_node_with_store b")
                    .is_some();
            let edge_present =
                crud::read_rel_with_store(&recovered.crud, &tx, RelId::new(rec.rel_id))
                    .expect("read_rel_with_store")
                    .is_some();
            if a_present {
                recovered_keys.insert(rec.node_a);
            }
            if b_present {
                recovered_keys.insert(rec.node_b);
            }
            if edge_present {
                recovered_keys.insert(edge);
            }
            if a_present && b_present && edge_present {
                survived_full += 1;
            }
        }

        // Non-vacuity gate 3: ≥1 acked merge SURVIVED recovery IN FULL — a
        // run where nothing survived proves nothing about bundle atomicity.
        assert!(
            survived_full >= 1,
            "0 of {} acked pre-crash merges survived recovery in full — the \
             recovered state is empty/torn; the durability + reconciliation \
             proof would be vacuous",
            pre.len(),
        );

        // Durability contract (direct, predicate-INDEPENDENT teeth): every
        // acked Strict commit's LSN is at or below the recovery watermark —
        // i.e. recovery's boundary covers every commit the child was told
        // was durable (ADR-034 §Slice-B fsync-before-ack; ADR-031
        // §Decision). This is what makes the predicate's acked-durable
        // branch (commit_lsn ≤ watermark ⇒ MUST be fully present) bite on
        // EVERY op in `pre`, rather than silently treating any as
        // legitimately-lost past-watermark commits.
        for rec in &pre {
            assert!(
                rec.commit_lsn <= watermark.raw(),
                "acked merge op {} committed at lsn {} ABOVE the recovery \
                 watermark {} — an acked Strict commit was not covered by \
                 recovery (durability violation, ADR-034 §Slice-B)",
                rec.op_id,
                rec.commit_lsn,
                watermark.raw(),
            );
        }

        // ── Phase 4: THE end-to-end proof — run the recovery-reconciliation
        //    predicate over the REAL recovered history. A correct ArcQL→MVCC
        //    →WAL→recovery kernel makes a torn (half-applied) MERGE
        //    unreachable (ADR-031 §Decision bundle-atomicity), so the verdict
        //    MUST be OK. The 8 self-tests (`crash_adversarial_history_tests`:
        //    7 predicate self-tests + 1 namespace pin) prove this predicate
        //    DETECTS a planted torn / acked-loss / phantom MERGE — so this
        //    `is_ok()` assertion is NON-vacuous.
        let verdict = ArcqlSiChecker::new().reconcile_arcql_pending_with_recovery(
            &history,
            watermark,
            &recovered_keys,
        );
        eprintln!(
            "live_sigkill_during_merge: acked_merges={} survived_full={} \
             watermark={} verdict={verdict}",
            pre.len(),
            survived_full,
            watermark.raw(),
        );
        assert!(
            verdict.is_ok(),
            "recovery-reconciliation FAILED over the real recovered MERGE history \
             (a torn / half-applied bundle survived SIGKILL+recovery — ADR-031 \
             §Decision bundle-atomicity violation): {verdict}"
        );
    }
}
