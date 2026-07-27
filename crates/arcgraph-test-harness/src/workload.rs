//! [`Workload`] trait + [`RegressionGate`] per ADR-042.
//!
//! A [`Workload`] names a (Cypher source, dataset target) pair from
//! the per-domain query catalog. Post-M4-61 (Wave 11Z flip) the
//! Tier-1 LDBC SNB workloads dispatch live through the executor's
//! `ExecutionContext` + `Batch` pipeline; Tier-2 workloads continue
//! to return [`WorkloadResult::Skipped`] with a forward-link to
//! M5-08 (real ingestion) until the loaders ship. Downstream tests
//! consume the `Ran { row_count, snapshot_lsn }` payload uniformly.

use arcgraph_core::Lsn;

use crate::HarnessResult;
use crate::dataset::DatasetHandle;

/// A workload (Cypher pattern over a [`Dataset`](crate::Dataset)).
///
/// The trait is intentionally minimal at Wave 11β. Two future
/// dimensions are reserved as scaffolding-time `Option`-fields on
/// [`WorkloadResult`]:
///
/// 1. `snapshot_lsn` — populated post-M4-61 once the executor's
///    `ExecutionContext` acquires an LSN per ADR-038 amendment-03
///    §TIER-1 GAP E rule 1.
/// 2. `oracle_class` — populated by the per-workload catalog when
///    an [`OracleAdapter`](crate::OracleAdapter) is wired.
pub trait Workload {
    /// Stable identifier (e.g. `"LDBC-SNB-IS1"`, `"HET-3"`).
    fn id(&self) -> &'static str;
    /// Domain bucket (PRD §2.2 column).
    fn domain(&self) -> &'static str;
    /// Cypher source for the workload, as a `&'static str` so the
    /// catalog ships zero allocation in the binary.
    fn cypher(&self) -> &'static str;
    /// Run the workload against a materialised dataset.
    ///
    /// Pre-M4-61 every impl returns `WorkloadResult::Skipped`; once
    /// M4-61 ships the dispatch routes through
    /// `arcgraph_query::QueryEngine::execute` to produce a
    /// [`WorkloadResult::Ran`].
    fn run(&self, dataset: &DatasetHandle) -> HarnessResult<WorkloadResult>;
}

/// Outcome of a single workload invocation.
///
/// Carries an `id` echoing the [`Workload::id`] for log-correlation,
/// and either a `Skipped { reason }` payload (pre-M4-61) or a
/// `Ran { snapshot_lsn, row_count }` payload (post-M4-61).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum WorkloadResult {
    /// Workload skipped because a downstream milestone has not
    /// shipped. `reason` is a human-readable cite to the gating
    /// milestone (e.g. `"M4-61 not yet shipped — pre-execution"`).
    Skipped { id: &'static str, reason: String },
    /// Workload executed end-to-end. `snapshot_lsn` is the LSN the
    /// executor pinned per ADR-038 amendment-03 §TIER-1 GAP E
    /// (acquired pre-first-batch, held query-end). `row_count` is
    /// the number of rows the executor materialised.
    Ran {
        id: &'static str,
        snapshot_lsn: Lsn,
        row_count: u64,
    },
}

impl WorkloadResult {
    /// Identifier echoing [`Workload::id`].
    pub fn id(&self) -> &'static str {
        match self {
            WorkloadResult::Skipped { id, .. } | WorkloadResult::Ran { id, .. } => id,
        }
    }

    /// Convenience for assertions that want to verify a workload
    /// genuinely ran (post-M4-61 flip).
    pub fn is_ran(&self) -> bool {
        matches!(self, WorkloadResult::Ran { .. })
    }
}

/// Criterion-style perf threshold for a [`Workload`].
///
/// Mirrors the gate shape committed in ADR-042 §"CI integration"
/// (Tier-1 LDBC SNB regression gate at ≥10%). At Wave 11β only
/// LDBC SNB instantiates a [`RegressionGate`]; Tier-2 domains run
/// without one (advisory-only per ADR-042).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegressionGate {
    /// Workload this gate applies to (e.g. `"LDBC-SNB-IS1"`).
    pub workload_id: &'static str,
    /// Baseline P99 latency in microseconds for the workload's
    /// design-v2 §10.5 target. Tier-1 gates assert observed P99 is
    /// within `regression_threshold_pct` of this baseline.
    pub baseline_p99_us: u64,
    /// Allowed regression in percent (e.g. `10` for the design-v2
    /// §10.5 +10% bound).
    pub regression_threshold_pct: u32,
}

impl RegressionGate {
    /// True if the observed P99 stays within the allowed regression
    /// bound. Used by the harness post-M4-84 once benches pipe
    /// numbers back; pre-M4-84 the only consumer is the
    /// `regression_gate_threshold` unit test.
    pub fn within_bound(&self, observed_p99_us: u64) -> bool {
        let allowed = self
            .baseline_p99_us
            .saturating_mul(u64::from(100 + self.regression_threshold_pct))
            / 100;
        observed_p99_us <= allowed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trivial workload that exercises the trait shape. Returns
    /// `Skipped` regardless of input — i.e. it pins the v1.0-alpha
    /// pre-M4-61 contract.
    struct ProbeWorkload;

    impl Workload for ProbeWorkload {
        fn id(&self) -> &'static str {
            "PROBE-1"
        }
        fn domain(&self) -> &'static str {
            "scaffold-probe"
        }
        fn cypher(&self) -> &'static str {
            "MATCH (n) RETURN n LIMIT 1"
        }
        fn run(&self, _dataset: &DatasetHandle) -> HarnessResult<WorkloadResult> {
            Ok(WorkloadResult::Skipped {
                id: "PROBE-1",
                reason: "pre-M4-61".into(),
            })
        }
    }

    #[test]
    fn workload_trait_is_object_safe() {
        let probe: &dyn Workload = &ProbeWorkload;
        assert_eq!(probe.id(), "PROBE-1");
        assert_eq!(probe.domain(), "scaffold-probe");
        assert!(probe.cypher().contains("MATCH"));
    }

    #[test]
    fn workload_result_id_round_trips() {
        let r = WorkloadResult::Skipped {
            id: "PROBE-1",
            reason: "x".into(),
        };
        assert_eq!(r.id(), "PROBE-1");
        assert!(!r.is_ran());

        let ran = WorkloadResult::Ran {
            id: "PROBE-2",
            snapshot_lsn: Lsn::ZERO,
            row_count: 7,
        };
        assert_eq!(ran.id(), "PROBE-2");
        assert!(ran.is_ran());
    }

    #[test]
    fn regression_gate_within_bound_at_baseline_and_at_threshold() {
        let gate = RegressionGate {
            workload_id: "LDBC-SNB-IS1",
            // design-v2 §10.5: IS1 P99 = 500 µs.
            baseline_p99_us: 500,
            // ADR-042 §"CI integration" Tier-1 = 10 %.
            regression_threshold_pct: 10,
        };
        assert!(gate.within_bound(500), "baseline must be within bound");
        assert!(
            gate.within_bound(550),
            "exact +10% bound must remain within"
        );
        assert!(
            !gate.within_bound(551),
            "+10% + 1 must trip the regression gate"
        );
    }

    #[test]
    fn regression_gate_handles_zero_baseline_without_panic() {
        let gate = RegressionGate {
            workload_id: "EDGE",
            baseline_p99_us: 0,
            regression_threshold_pct: 10,
        };
        // A zero-baseline gate trips on any non-zero observation —
        // structurally what we want (the loader has no number yet).
        assert!(gate.within_bound(0));
        assert!(!gate.within_bound(1));
    }
}
