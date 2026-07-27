//! [`OracleAdapter`] trait and synthetic-invariant implementation.
//!
//! The retained classes are:
//!
//! 1. [`OracleClass::SyntheticInvariant`] — asserts a structural
//!    invariant on the result set.
//! 2. [`OracleClass::ReproducibleSnapshot`] — frozen result-set
//!    hash (sentinel; brittle, used sparingly).

use crate::HarnessResult;
use crate::workload::WorkloadResult;

/// One of the retained oracle classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OracleClass {
    /// Assert a structural invariant on the result-set.
    SyntheticInvariant,
    /// Hash a frozen result-set as a regression sentinel.
    ReproducibleSnapshot,
}

/// Verdict returned by an [`OracleAdapter`].
///
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum OracleVerdict {
    /// Result agreed with the oracle.
    Pass,
    /// Result disagreed with the oracle (Tier-1 only — blocks
    /// merges).
    Fail { reason: String },
    /// Oracle could not run (e.g. M5-13 not yet shipped).
    Skipped { reason: String },
}

/// Common shape across the three oracle classes.
pub trait OracleAdapter {
    /// Which class this adapter belongs to.
    fn class(&self) -> OracleClass;
    /// Compare an observed [`WorkloadResult`] against the oracle.
    fn check(&self, observed: &WorkloadResult) -> HarnessResult<OracleVerdict>;
}

/// Synthetic-invariant oracle that wraps a closure.
///
/// The closure returns `true` iff the observed result satisfies the
/// invariant.
pub struct SyntheticInvariantOracle<F> {
    /// Human-readable invariant description.
    pub invariant: &'static str,
    /// Closure that asserts the invariant. Pre-M4-61 the harness
    /// always passes the closure a `Skipped` result so the
    /// invariant should accept that shape too — the post-M4-61
    /// flip switches it to validating real result-sets.
    pub check_fn: F,
}

impl<F> std::fmt::Debug for SyntheticInvariantOracle<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SyntheticInvariantOracle")
            .field("invariant", &self.invariant)
            .field("check_fn", &"<closure>")
            .finish()
    }
}

impl<F> OracleAdapter for SyntheticInvariantOracle<F>
where
    F: Fn(&WorkloadResult) -> bool,
{
    fn class(&self) -> OracleClass {
        OracleClass::SyntheticInvariant
    }

    fn check(&self, observed: &WorkloadResult) -> HarnessResult<OracleVerdict> {
        if (self.check_fn)(observed) {
            Ok(OracleVerdict::Pass)
        } else {
            Ok(OracleVerdict::Fail {
                reason: format!("synthetic invariant failed: {}", self.invariant),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_invariant_oracle_passes_when_predicate_holds() {
        let oracle = SyntheticInvariantOracle {
            invariant: "skipped-results-allowed",
            check_fn: |observed: &WorkloadResult| {
                matches!(observed, WorkloadResult::Skipped { .. })
            },
        };
        let observed = WorkloadResult::Skipped {
            id: "ATT-3",
            reason: "pre-M4-61".into(),
        };
        let verdict = oracle.check(&observed).expect("verdict");
        assert_eq!(verdict, OracleVerdict::Pass);
    }

    #[test]
    fn synthetic_invariant_oracle_fails_when_predicate_violated() {
        let oracle = SyntheticInvariantOracle {
            invariant: "must-have-rows",
            check_fn: |observed: &WorkloadResult| {
                matches!(
                    observed,
                    WorkloadResult::Ran { row_count, .. } if *row_count > 0
                )
            },
        };
        let observed = WorkloadResult::Skipped {
            id: "HET-1",
            reason: "pre-M4-61".into(),
        };
        match oracle.check(&observed).expect("verdict") {
            OracleVerdict::Fail { reason } => {
                assert!(reason.contains("must-have-rows"));
            }
            other => panic!("expected Fail, got {other:?}"),
        }
    }

    #[test]
    fn oracle_class_round_trips_for_synthetic_invariant() {
        let oracle = SyntheticInvariantOracle {
            invariant: "always-true",
            check_fn: |_: &WorkloadResult| true,
        };
        assert_eq!(oracle.class(), OracleClass::SyntheticInvariant);
    }
}
