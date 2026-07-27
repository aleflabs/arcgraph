//! ADR-047 §"First test landing" — Jepsen-style bank-transfer
//! workload under MVCC snapshot isolation.
//!
//! ## What this test covers
//!
//! - 4 concurrent client threads, each running 100 transfer ops.
//! - N = 10 accounts seeded with balance 100 (total sum 1000).
//! - Each op: pick (src, dst, amount), open a snapshot-isolated
//!   transaction, read both balances, write back if src has the
//!   funds, commit. Retry on `MvccConflict` up to 8 times.
//! - Recorded history (start LSN, commit LSN, reads, writes,
//!   outcome) walked by
//!   [`SnapshotIsolationChecker::check`].
//! - Two invariants asserted: **sum-invariant** (total balance
//!   never diverges from 1000) and **per-op visibility consistency**
//!   (every recorded read is consistent with the committed prefix
//!   at the reader's `start_lsn`).
//!
//! ## proptest harness
//!
//! 5 release-mode iterations under different XorShift seeds. Each
//! iteration is a fresh `TxnManager`. A failing seed is the
//! load-bearing reproduction artifact; the printed verdict carries
//! the offending history excerpt.
//!
//! ## SIGKILL variant — deferred to v1.1
//!
//! ADR-047 §"Open questions" / §"CI integration policy" reserves the
//! `JEPSEN_SIGKILL=1` opt-in path for a future test that wires the
//! K-1 subprocess SIGKILL pipeline into the workload and verifies the
//! recovery oracle preserves SI. At v0.1.0-alpha.0+1 no such test
//! exists in this file: per R1 review F-M3 (PR #344) an
//! env-gated test that constructs a `FaultInjectionContext` but
//! never invokes it + asserts the same property as the steady-state
//! variant is a positive assertion on a no-op path. The test
//! reappears at v1.1 alongside list-append + Elle when there is a
//! real consumer.
//!
//! ## Runtime expectations
//!
//! In release on a current laptop the proptest finishes in well
//! under one second: 5 cases × 4 client threads × 400 ops, all
//! in-memory through the MVCC kernel; there is no I/O on this path.
//! `PROPTEST_CASES` is env-overrideable (`PROPTEST_CASES=10_000
//! cargo test …`) for hostile-reviewer coverage.
//!
//! Run:
//!
//!   cargo test -p arcgraph-storage --release \
//!       -- jepsen_bank_transfer_snapshot --nocapture

use std::sync::Arc;

use arcgraph_storage::test_harness::jepsen::checker::SnapshotIsolationChecker;
use arcgraph_storage::test_harness::jepsen::history::{OpOutcome, OperationHistory};
use arcgraph_storage::test_harness::jepsen::workload::{
    BankTransferConfig, account_key, decode_balance, expected_sum, run_bank_transfer,
};
use arcgraph_storage::transaction::TxnManager;
use proptest::prelude::*;

/// Default number of proptest iterations. Each case is ~tens of ms
/// in release on a current laptop (5 cases × 4 client threads × 400
/// in-memory MVCC ops); the proptest as a whole finishes well under
/// one second. Hostile-reviewer coverage: set `PROPTEST_CASES=10_000`
/// to push the iteration count without editing source (per R1 F-L6).
const PROPTEST_CASES: u32 = 5;

/// Read `PROPTEST_CASES` from the environment, falling back to the
/// default constant. Allows reviewers to tighten or loosen coverage
/// without source edits (per R1 F-L6).
fn proptest_cases() -> u32 {
    std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(PROPTEST_CASES)
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: proptest_cases(),
        // Anchor `FileFailurePersistence::SourceParallel` to *this*
        // file so failing seeds land in `proptest-regressions/` for
        // replay; without this proptest emits "failed to find lib.rs"
        // and the regression path is silently disabled (R1 F-L3).
        source_file: Some(file!()),
        .. ProptestConfig::default()
    })]

    /// **Steady-state SI invariant.** Run the bank-transfer workload
    /// to completion under random (per-seed) (src, dst, amount)
    /// selection. Verify the recorded history satisfies both
    /// (a) the sum-invariant at every committed LSN, and
    /// (b) per-op visibility consistency.
    #[test]
    fn bank_transfer_preserves_si(seed in any::<u64>()) {
        let mgr = Arc::new(TxnManager::new());
        let cfg = BankTransferConfig {
            clients: 4,
            ops_per_client: 100,
            accounts: 10,
            initial_balance: 100,
            max_transfer: 50,
            max_retries: 8,
            seed,
            tenant: arcgraph_core::TenantId::DEFAULT,
        };
        let history = Arc::new(OperationHistory::new());

        run_bank_transfer(Arc::clone(&mgr), cfg, Arc::clone(&history));

        let ops = history.drain_sorted();
        let total_ops = ops.len();
        let committed_count = ops
            .iter()
            .filter(|o| matches!(o.outcome, OpOutcome::Committed))
            .count();

        // Commit-rate floor (R1 F-M6, PR #344). A `total_ops >=
        // accounts + 1` assertion would pass a workload that aborts
        // every op (the checker's verdict is vacuous on a no-commit
        // history). At minimum every seed commit must land AND at
        // least 50 % of attempted workload ops must commit — anything
        // lower indicates an OCC-conflict regression that should
        // fail loud, not be hidden behind a green SI verdict.
        let workload_ops_attempted =
            (cfg.clients as usize) * (cfg.ops_per_client as usize);
        let min_committed =
            (cfg.accounts as usize) + workload_ops_attempted / 2;
        prop_assert!(
            committed_count >= min_committed,
            "commit rate too low: {committed_count} committed of \
             {workload_ops_attempted} attempted (+ {} seed); seed={seed:#x}",
            cfg.accounts
        );

        let checker = SnapshotIsolationChecker::for_bank_transfer(
            cfg.accounts,
            cfg.initial_balance,
        );
        let verdict = checker.check(&ops);
        prop_assert!(
            verdict.is_ok(),
            "SI violation under seed {seed:#x}: {verdict}\n  total recorded ops: {total_ops}, committed: {committed_count}"
        );

        // Cross-check: read every account at the final visible LSN
        // and confirm the sum lines up. This catches the case where
        // the recorded history says one thing but the live store
        // says another.
        let reader = mgr.begin(cfg.tenant);
        let mut live_sum: u64 = 0;
        for i in 0..cfg.accounts {
            let v = reader
                .read(account_key(i))
                .and_then(|b| decode_balance(&b))
                .unwrap_or(0);
            live_sum += v;
        }
        prop_assert_eq!(
            live_sum,
            expected_sum(&cfg),
            "live sum diverged at seed {:#x}", seed
        );
    }
}

// **JEPSEN_SIGKILL=1 variant** — deliberately omitted at
// v0.1.0-alpha.0+1. Per R1 review F-M3 (PR #344): a test that
// constructs a `FaultInjectionContext` but never wires it into the
// workload + asserts the same property as the steady-state path is
// a positive assertion on a no-op path — exactly the
// `feedback_review_oracle_relaxations.md` failure mode. The
// env-gate contract is documented in ADR-047
// §"CI integration policy"; the subprocess SIGKILL pipeline +
// recovery oracle land at v1.1 alongside list-append + Elle, at
// which point the test re-appears with a real consumer (per
// `feedback_avoid_speculative_scaffolding.md`).

/// Smaller-scale sanity test always-on. Verifies the harness shape
/// at a non-proptest scale so a `cargo test` without `--release`
/// still exercises the workload end-to-end.
#[test]
fn bank_transfer_tiny_smoke() {
    let mgr = Arc::new(TxnManager::new());
    let cfg = BankTransferConfig {
        clients: 2,
        ops_per_client: 20,
        accounts: 5,
        initial_balance: 100,
        max_transfer: 25,
        max_retries: 4,
        seed: 0xCAFE_F00D_BAAD_F00D,
        tenant: arcgraph_core::TenantId::DEFAULT,
    };
    let history = Arc::new(OperationHistory::new());
    run_bank_transfer(Arc::clone(&mgr), cfg, Arc::clone(&history));

    let ops = history.drain_sorted();
    assert!(!ops.is_empty(), "history populated");

    let checker = SnapshotIsolationChecker::for_bank_transfer(cfg.accounts, cfg.initial_balance);
    let verdict = checker.check(&ops);
    assert!(verdict.is_ok(), "tiny smoke failed SI check: {verdict}");

    // Cross-check live state vs. expected sum.
    let reader = mgr.begin(cfg.tenant);
    let mut sum: u64 = 0;
    for i in 0..cfg.accounts {
        sum += reader
            .read(account_key(i))
            .and_then(|b| decode_balance(&b))
            .unwrap_or(0);
    }
    assert_eq!(sum, expected_sum(&cfg));
}
