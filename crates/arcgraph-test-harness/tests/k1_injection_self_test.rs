//! W26-γ-2 D4 — arcgraph-test-harness self-test for the
//! `Dataset` / `Workload` injection seams.
//!
//! Per ADR-134 forward-binding + W26-γ-2 D4 spec ("verify the
//! harness's K-1-injection-flavored APIs faithfully inject intended
//! faults; oracle: replay determinism").
//!
//! The arcgraph-storage `k1::injection` module (out-of-crate) carries
//! the canonical per-op rate-based fault-injection API. The harness's
//! injection seam is its `WorkloadResult` shape — a workload may
//! return `Skipped` (no-fault) OR `Ran` (no-fault) OR `Skipped` with
//! an injection-driven reason. This self-test asserts the cross-
//! cutting invariants that the harness's Workload trait surface
//! preserves:
//!
//! - A `Workload` impl that injects a `Skipped` fault must remain
//!   deterministic across repeated calls.
//! - `WorkloadResult::id()` echoes the workload's id regardless of
//!   the verdict (load-bearing for log correlation).
//! - `RegressionGate::within_bound` correctly rejects observations
//!   that exceed the +N% bound (regression-gate harness).

use std::cell::Cell;
use std::sync::Mutex;

use arcgraph_core::Lsn;
use arcgraph_test_harness::dataset::{Dataset, DatasetHandle, DatasetScale};
use arcgraph_test_harness::workload::{RegressionGate, Workload, WorkloadResult};
use arcgraph_test_harness::{HarnessError, HarnessResult};

// ────────────────────── Reproducibility under repeated invocation ──────────────────────

/// Workload that returns a deterministic Skipped on every call.
/// Exercises the replay-determinism invariant the harness asserts.
struct ReplayDeterministicWorkload {
    id: &'static str,
}

impl Workload for ReplayDeterministicWorkload {
    fn id(&self) -> &'static str {
        self.id
    }
    fn domain(&self) -> &'static str {
        "test"
    }
    fn cypher(&self) -> &'static str {
        "RETURN 1"
    }
    fn run(&self, _dataset: &DatasetHandle) -> HarnessResult<WorkloadResult> {
        Ok(WorkloadResult::Skipped {
            id: self.id,
            reason: "deterministic-skip".into(),
        })
    }
}

fn synthetic_handle() -> DatasetHandle {
    let edges: Vec<(u32, u32, f32)> = (0..3).map(|u| (u, u + 1, 1.0)).collect();
    DatasetHandle::SbmGraph(arcgraph_community::Graph::from_edges_undirected(4, &edges))
}

#[test]
fn deterministic_workload_replays_identically() {
    let w = ReplayDeterministicWorkload { id: "REPLAY-1" };
    let ds = synthetic_handle();
    let r1 = w.run(&ds).expect("run 1");
    let r2 = w.run(&ds).expect("run 2");
    let r3 = w.run(&ds).expect("run 3");
    assert_eq!(r1, r2);
    assert_eq!(r2, r3);
    assert_eq!(r1.id(), "REPLAY-1");
    assert!(!r1.is_ran());
}

// ────────────────────── Per-call fault injection ──────────────────────

/// Workload that toggles between Ran and Skipped per call — exercises
/// the K-1-injection-flavored "fire every N-th call" pattern at the
/// harness level.
struct ToggleInjectingWorkload {
    counter: Mutex<u64>,
}

impl Workload for ToggleInjectingWorkload {
    fn id(&self) -> &'static str {
        "TOGGLE-1"
    }
    fn domain(&self) -> &'static str {
        "fault-injection-probe"
    }
    fn cypher(&self) -> &'static str {
        "MATCH (n) RETURN n LIMIT 1"
    }
    fn run(&self, _dataset: &DatasetHandle) -> HarnessResult<WorkloadResult> {
        let n = {
            let mut g = self.counter.lock().expect("counter");
            *g += 1;
            *g
        };
        if n % 2 == 0 {
            Ok(WorkloadResult::Ran {
                id: "TOGGLE-1",
                snapshot_lsn: Lsn::new(n),
                row_count: n,
            })
        } else {
            Ok(WorkloadResult::Skipped {
                id: "TOGGLE-1",
                reason: format!("injected-skip-on-call-{n}"),
            })
        }
    }
}

#[test]
fn toggle_workload_injects_faults_deterministically() {
    let w = ToggleInjectingWorkload {
        counter: Mutex::new(0),
    };
    let ds = synthetic_handle();
    let calls: Vec<_> = (0..6).map(|_| w.run(&ds).expect("run")).collect();
    // Odd calls (1, 3, 5) are Skipped; even calls (2, 4, 6) are Ran.
    assert!(!calls[0].is_ran());
    assert!(calls[1].is_ran());
    assert!(!calls[2].is_ran());
    assert!(calls[3].is_ran());
    assert!(!calls[4].is_ran());
    assert!(calls[5].is_ran());
    // Replay-determinism oracle: ids are stable across all calls.
    for r in &calls {
        assert_eq!(r.id(), "TOGGLE-1");
    }
}

// ────────────────────── Error-injection variant ──────────────────────

struct ErrorInjectingWorkload {
    fire_on_call: u32,
    counter: Cell<u32>,
}

// Cell is not Sync — but we only call this single-threaded, and
// Workload is not required to be Sync at v1.0-alpha. Mark unsafe
// impl to keep the trait satisfied.
unsafe impl Sync for ErrorInjectingWorkload {}

impl Workload for ErrorInjectingWorkload {
    fn id(&self) -> &'static str {
        "ERR-1"
    }
    fn domain(&self) -> &'static str {
        "fault-injection-probe"
    }
    fn cypher(&self) -> &'static str {
        "RETURN 1"
    }
    fn run(&self, _: &DatasetHandle) -> HarnessResult<WorkloadResult> {
        let n = self.counter.get() + 1;
        self.counter.set(n);
        if n == self.fire_on_call {
            return Err(HarnessError::FixtureFailed {
                reason: format!("injected error on call {n}"),
            });
        }
        Ok(WorkloadResult::Skipped {
            id: "ERR-1",
            reason: format!("call-{n}"),
        })
    }
}

#[test]
fn error_injecting_workload_fires_at_target_call() {
    let w = ErrorInjectingWorkload {
        fire_on_call: 3,
        counter: Cell::new(0),
    };
    let ds = synthetic_handle();
    assert!(w.run(&ds).expect("call 1").id() == "ERR-1");
    assert!(w.run(&ds).expect("call 2").id() == "ERR-1");
    // Call 3 fires the injected error.
    match w.run(&ds) {
        Err(HarnessError::FixtureFailed { reason }) => {
            assert!(reason.contains("call 3"), "got: {reason}");
        }
        other => panic!("expected FixtureFailed, got {other:?}"),
    }
    // Call 4 returns Skipped normally.
    assert!(w.run(&ds).expect("call 4").id() == "ERR-1");
}

// ────────────────────── Regression gate as injection oracle ──────────────────────

#[test]
fn regression_gate_rejects_observations_past_threshold() {
    let gate = RegressionGate {
        workload_id: "FAULT-INJECTED-WORKLOAD",
        baseline_p99_us: 1000,
        regression_threshold_pct: 10,
    };
    assert!(gate.within_bound(1000), "exact baseline must pass");
    assert!(gate.within_bound(1100), "exact +10% must pass");
    assert!(!gate.within_bound(1101), "+10% + 1µs must fail");
    assert!(!gate.within_bound(2000), "obvious regression must fail");
}

#[test]
fn regression_gate_handles_huge_baseline_without_overflow() {
    let gate = RegressionGate {
        workload_id: "MAX",
        baseline_p99_us: u64::MAX / 200,
        regression_threshold_pct: 10,
    };
    // saturating_mul keeps the math from overflowing.
    assert!(gate.within_bound(0)); // any obs ≤ baseline*1.10 passes
}

// ────────────────────── Dataset trait round-trip ──────────────────────

struct CustomLoaderProbe {
    nodes: u64,
    edges: u64,
}

impl Dataset for CustomLoaderProbe {
    fn id(&self) -> &'static str {
        "custom-probe"
    }
    fn domain(&self) -> &'static str {
        "test"
    }
    fn upstream_source(&self) -> &'static str {
        "(probe)"
    }
    fn license(&self) -> &'static str {
        "Apache-2.0"
    }
    fn approximate_scale(&self) -> DatasetScale {
        DatasetScale {
            nodes: self.nodes,
            edges: self.edges,
        }
    }
    fn load(&self) -> HarnessResult<DatasetHandle> {
        let edges: Vec<(u32, u32, f32)> = (0..self.edges as u32).map(|u| (u, u + 1, 1.0)).collect();
        Ok(DatasetHandle::SbmGraph(
            arcgraph_community::Graph::from_edges_undirected((self.nodes + 1) as u32, &edges),
        ))
    }
}

#[test]
fn dataset_handle_node_count_matches_loader_scale() {
    let probe = CustomLoaderProbe {
        nodes: 10,
        edges: 5,
    };
    let handle = probe.load().expect("load");
    // node_count is the underlying graph's `n()`, which the loader
    // constructed as nodes+1.
    assert_eq!(handle.node_count(), 11);
}

#[test]
fn dataset_trait_object_dispatch_works() {
    let probe: Box<dyn Dataset> = Box::new(CustomLoaderProbe { nodes: 4, edges: 2 });
    let handle = probe.load().expect("load");
    assert_eq!(handle.node_count(), 5);
    assert_eq!(probe.license(), "Apache-2.0");
    assert_eq!(probe.approximate_scale().nodes, 4);
}

// ────────────────────── HarnessError variant exhaustiveness ──────────────────────

#[test]
fn harness_error_three_variants_round_trip_display() {
    let e1 = HarnessError::NotImplementedAtV1 {
        feature: "M5-08",
        reason: "test".into(),
    };
    let e2 = HarnessError::FixtureFailed {
        reason: "test".into(),
    };
    let e3 = HarnessError::OracleDisagreement {
        reason: "test".into(),
    };
    assert!(format!("{e1}").contains("M5-08"));
    assert!(format!("{e2}").contains("fixture"));
    assert!(format!("{e3}").contains("disagreement"));
}
