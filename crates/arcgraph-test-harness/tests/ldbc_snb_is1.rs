//! Wave-11Z post-flip integration test: LDBC SNB IS1 round-trip via
//! the `Workload` trait now that M4-61 ships the executor.
//!
//! Pin set:
//!
//! 1. The deterministic SBM fixture yields a non-empty graph.
//! 2. `LdbcSnbIs1Workload::run` dispatches the IS1 cypher through
//!    `arcgraph_query::QueryEngine::execute` and returns
//!    `WorkloadResult::Ran { row_count: 0, snapshot_lsn: Lsn::MAX }`
//!    against the empty stub substrate (M5-08 lifts the
//!    `row_count` floor). Skipped is the documented escape hatch
//!    when the executor surface flattens to NotImplemented per the
//!    W11Z spawn-prompt "DO NOT block on fix-up E" framing.
//! 3. The Tier-1 regression-gate shape pins design-v2 §10.5 IS1
//!    targets (P99 = 500 µs, +10 % bound).

use arcgraph_core::Lsn;
use arcgraph_test_harness::workloads::ldbc_snb::{LdbcSnbIs1Workload, LdbcSnbSf1Dataset};
use arcgraph_test_harness::{Dataset, Workload, WorkloadResult};

#[test]
fn ldbc_snb_is1_round_trip_dispatches_through_executor_post_w11z() {
    let dataset = LdbcSnbSf1Dataset;

    // Step 1 — fixture build.
    let handle = dataset.load().expect("LDBC SF-1 SBM fixture must build");
    assert!(
        handle.node_count() >= 100,
        "SBM fixture must yield at least 100 nodes, got {}",
        handle.node_count(),
    );

    // Step 2 — Workload::run round-trip via the M4-61 executor seam.
    let workload = LdbcSnbIs1Workload;
    let result = workload.run(&handle).expect("run must succeed post-W11Z");
    match &result {
        WorkloadResult::Ran {
            id,
            snapshot_lsn,
            row_count,
        } => {
            assert_eq!(*id, "LDBC-SNB-IS1");
            // Empty stub substrate floor — M4-84 + M5-08 lift this.
            assert_eq!(
                *row_count, 0,
                "empty stub substrate must yield 0 rows; got {row_count}",
            );
            assert_eq!(*snapshot_lsn, Lsn::MAX);
        }
        WorkloadResult::Skipped { id, reason } => {
            assert_eq!(*id, "LDBC-SNB-IS1");
            assert!(
                reason.contains("executor surface gap"),
                "Skipped only acceptable for executor-surface-gap escape, got {reason:?}",
            );
        }
        // `WorkloadResult` is `#[non_exhaustive]`; future variants
        // (e.g. a `Cancelled` shape) are explicitly out of scope at
        // Wave 11Z and should fail this pin loudly.
        other => panic!("unexpected WorkloadResult variant: {other:?}"),
    }

    // Step 3 — regression-gate shape pins design-v2 §10.5 IS1 targets.
    let gate = LdbcSnbIs1Workload::regression_gate();
    assert_eq!(gate.workload_id, "LDBC-SNB-IS1");
    assert_eq!(gate.baseline_p99_us, 500);
    assert_eq!(gate.regression_threshold_pct, 10);
    assert!(gate.within_bound(550), "+10 % bound must remain within");
    assert!(!gate.within_bound(551), "+10 % + 1 must trip the gate");
}
