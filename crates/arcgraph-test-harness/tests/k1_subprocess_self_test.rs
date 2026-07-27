//! W26-γ-2 D4 — arcgraph-test-harness self-test for
//! subprocess-lifecycle-style workload patterns.
//!
//! Per ADR-134 forward-binding + W26-γ-2 D4 spec ("verify
//! subprocess module spawn/kill/recover lifecycle").
//!
//! The arcgraph-storage `k1::subprocess` module (out-of-crate) carries
//! the canonical SIGKILL fork harness. This file's scope is the
//! harness-level analogue: a `Workload` impl that models a subprocess
//! lifecycle via an internal state-machine. The self-test asserts:
//!
//! - State transitions are deterministic given the same input.
//! - A `Skipped` result correctly carries the "subprocess not yet
//!   spawned" semantics.
//! - A `Ran` result captures the post-recovery `snapshot_lsn`.
//! - `WorkloadResult::id()` echoes the workload's id across the
//!   subprocess lifecycle for log correlation.

use std::sync::Mutex;

use arcgraph_core::Lsn;
use arcgraph_test_harness::dataset::DatasetHandle;
use arcgraph_test_harness::workload::{Workload, WorkloadResult};
use arcgraph_test_harness::{HarnessError, HarnessResult};

/// Workload that mimics the subprocess `spawn → kill → recover`
/// lifecycle:
/// - call #1: spawn (Skipped with "subprocess-spawning")
/// - call #2: kill (Skipped with "subprocess-killed")
/// - call #3: recover (Ran with snapshot_lsn carrying the recovered state)
struct SubprocessLifecycleWorkload {
    phase: Mutex<u32>,
}

impl Workload for SubprocessLifecycleWorkload {
    fn id(&self) -> &'static str {
        "SUBPROC-1"
    }
    fn domain(&self) -> &'static str {
        "subprocess-lifecycle-probe"
    }
    fn cypher(&self) -> &'static str {
        "MATCH (n) RETURN COUNT(n)"
    }
    fn run(&self, _: &DatasetHandle) -> HarnessResult<WorkloadResult> {
        let phase = {
            let mut g = self.phase.lock().expect("phase");
            *g += 1;
            *g
        };
        Ok(match phase {
            1 => WorkloadResult::Skipped {
                id: "SUBPROC-1",
                reason: "subprocess-spawning".into(),
            },
            2 => WorkloadResult::Skipped {
                id: "SUBPROC-1",
                reason: "subprocess-killed".into(),
            },
            3 => WorkloadResult::Ran {
                id: "SUBPROC-1",
                snapshot_lsn: Lsn::new(100), // canonical pre-crash committed state
                row_count: 5,
            },
            _ => WorkloadResult::Skipped {
                id: "SUBPROC-1",
                reason: format!("post-recovery-call-{phase}"),
            },
        })
    }
}

fn synthetic_handle() -> DatasetHandle {
    DatasetHandle::SbmGraph(arcgraph_community::Graph::from_edges_undirected(4, &[]))
}

// ────────────────────── Lifecycle phases ──────────────────────

#[test]
fn subprocess_workload_traverses_spawn_kill_recover() {
    let w = SubprocessLifecycleWorkload {
        phase: Mutex::new(0),
    };
    let ds = synthetic_handle();

    let spawn = w.run(&ds).expect("spawn");
    let kill = w.run(&ds).expect("kill");
    let recover = w.run(&ds).expect("recover");

    // Phase 1: spawn — Skipped with the canonical reason cite.
    match spawn {
        WorkloadResult::Skipped { id, reason } => {
            assert_eq!(id, "SUBPROC-1");
            assert!(reason.contains("subprocess-spawning"), "got: {reason}");
        }
        other => panic!("expected Skipped(spawning), got {other:?}"),
    }

    // Phase 2: kill — Skipped with the SIGKILL cite.
    match kill {
        WorkloadResult::Skipped { id, reason } => {
            assert_eq!(id, "SUBPROC-1");
            assert!(reason.contains("subprocess-killed"), "got: {reason}");
        }
        other => panic!("expected Skipped(killed), got {other:?}"),
    }

    // Phase 3: recover — Ran with the recovered snapshot_lsn.
    match recover {
        WorkloadResult::Ran {
            id,
            snapshot_lsn,
            row_count,
        } => {
            assert_eq!(id, "SUBPROC-1");
            assert_eq!(snapshot_lsn, Lsn::new(100));
            assert_eq!(row_count, 5);
        }
        other => panic!("expected Ran(recover), got {other:?}"),
    }
}

#[test]
fn subprocess_post_recovery_calls_are_skipped() {
    let w = SubprocessLifecycleWorkload {
        phase: Mutex::new(0),
    };
    let ds = synthetic_handle();
    // Advance through spawn / kill / recover.
    for _ in 0..3 {
        let _ = w.run(&ds).expect("phase");
    }
    // Post-recovery (phase 4+) returns the structured tail-skip.
    let tail = w.run(&ds).expect("tail call");
    match tail {
        WorkloadResult::Skipped { reason, .. } => {
            assert!(
                reason.contains("post-recovery-call-4"),
                "tail-skip reason must cite phase number; got: {reason}"
            );
        }
        other => panic!("expected post-recovery Skipped, got {other:?}"),
    }
}

#[test]
fn subprocess_workload_replay_determinism() {
    // Two independent workload instances at the same phase must
    // produce identical results — load-bearing for ADR-134 D-3 (seed,
    // commit) reproducibility contract.
    let w1 = SubprocessLifecycleWorkload {
        phase: Mutex::new(0),
    };
    let w2 = SubprocessLifecycleWorkload {
        phase: Mutex::new(0),
    };
    let ds = synthetic_handle();
    for _ in 0..3 {
        let r1 = w1.run(&ds).expect("w1");
        let r2 = w2.run(&ds).expect("w2");
        assert_eq!(r1, r2);
    }
}

// ────────────────────── HarnessError surfacing ──────────────────────

struct SubprocessFailureWorkload;

impl Workload for SubprocessFailureWorkload {
    fn id(&self) -> &'static str {
        "SUBPROC-FAIL"
    }
    fn domain(&self) -> &'static str {
        "subprocess-failure-probe"
    }
    fn cypher(&self) -> &'static str {
        "RETURN 1"
    }
    fn run(&self, _: &DatasetHandle) -> HarnessResult<WorkloadResult> {
        Err(HarnessError::FixtureFailed {
            reason: "subprocess fork(2) returned -1: simulated".into(),
        })
    }
}

#[test]
fn subprocess_failure_surfaces_structured_error() {
    let w = SubprocessFailureWorkload;
    let ds = synthetic_handle();
    match w.run(&ds) {
        Err(HarnessError::FixtureFailed { reason }) => {
            assert!(reason.contains("fork(2)"), "got: {reason}");
        }
        other => panic!("expected FixtureFailed, got {other:?}"),
    }
}

// ────────────────────── Cross-workload id correlation ──────────────────────

#[test]
fn ids_are_stable_across_subprocess_phases() {
    let w = SubprocessLifecycleWorkload {
        phase: Mutex::new(0),
    };
    let ds = synthetic_handle();
    let phases: Vec<_> = (0..3).map(|_| w.run(&ds).expect("phase")).collect();
    for r in &phases {
        // Every phase's result carries the workload id — load-bearing
        // for CI log correlation across the subprocess lifecycle.
        assert_eq!(r.id(), "SUBPROC-1");
    }
}
