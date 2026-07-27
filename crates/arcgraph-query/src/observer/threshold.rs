//! 10× threshold breach event types.
//!
//! Per ADR-038 amendment-02 §M4.g: a [`ThresholdBreach`] fires when the
//! observed cardinality of an operator (aggregated by [`OperatorKind`])
//! diverges from the planner's estimated cardinality by ≥10× either
//! direction. Replan (M4-72) reads the breach set to decide whether to
//! re-plan.

use crate::observer::row_count::OperatorKind;

/// Direction of a 10× threshold breach.
///
/// # Why exempt from `#[non_exhaustive]`
///
/// The two-variant enum exhaustively covers the two possible directions;
/// adding a third would require a fundamentally new threshold semantics
/// (e.g., absolute-delta) that would be a coordinated breaking change.
/// Under the code-quality policy exemption rule, the variant set IS the public
/// contract for downstream pattern-matching consumption (replan logic +
/// observability rendering both pattern-match exhaustively).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BreachDirection {
    /// Observed ≥ `threshold_factor` × estimated. The planner under-
    /// estimated cardinality; replan should re-evaluate selectivity
    /// upward. Most common cause: stale catalog stats post-bulk-load.
    UnderEstimate,
    /// Observed ≤ estimated / `threshold_factor`. The planner over-
    /// estimated cardinality; replan should re-evaluate selectivity
    /// downward. Most common cause: stale catalog stats post-bulk-delete.
    OverEstimate,
}

impl BreachDirection {
    /// Stable string slug for `tracing` field emission.
    #[inline]
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::UnderEstimate => "under_estimate",
            Self::OverEstimate => "over_estimate",
        }
    }
}

/// A single 10× threshold breach event for one operator kind.
///
/// Constructed by [`crate::observer::RowCountObserver::threshold_breaches`]
/// from the observer's accumulated state. The replan controller reads a
/// `Vec<ThresholdBreach>` to decide whether to invoke replan and which
/// operator kinds to re-cost.
#[derive(Debug, Clone, PartialEq)]
pub struct ThresholdBreach {
    /// The operator kind whose aggregate diverged.
    pub op_kind: OperatorKind,
    /// Sum of estimated cardinalities (from the cost tree) across every
    /// operator instance of `op_kind` in the plan.
    pub estimated_card_sum: f64,
    /// Sum of observed row counts across every operator instance of
    /// `op_kind` during execution.
    pub observed_rows_sum: u64,
    /// Ratio of observed to estimated. For UnderEstimate this is
    /// `observed / estimated` ≥ `threshold_factor`. For OverEstimate
    /// this is `estimated / observed` ≥ `threshold_factor` (i.e., the
    /// inverse direction's ratio is ≥ threshold; both numerator and
    /// denominator surface here for diagnostic symmetry — render the
    /// raw `observed_rows_sum / estimated_card_sum` for human display).
    pub ratio: f64,
    /// Direction of the breach.
    pub direction: BreachDirection,
}

impl ThresholdBreach {
    /// Construct an UnderEstimate breach (`observed ≥ factor × estimated`).
    #[inline]
    #[must_use]
    pub fn under_estimate(op_kind: OperatorKind, estimated: f64, observed: u64) -> Self {
        let ratio = if estimated > 0.0 {
            observed as f64 / estimated
        } else {
            // estimated == 0 + observed > 0: ratio is unbounded, render
            // as observed (any positive number is ≥ threshold here).
            observed as f64
        };
        Self {
            op_kind,
            estimated_card_sum: estimated,
            observed_rows_sum: observed,
            ratio,
            direction: BreachDirection::UnderEstimate,
        }
    }

    /// Construct an OverEstimate breach (`observed ≤ estimated / factor`).
    #[inline]
    #[must_use]
    pub fn over_estimate(op_kind: OperatorKind, estimated: f64, observed: u64) -> Self {
        let ratio = if observed > 0 {
            estimated / observed as f64
        } else {
            // observed == 0 + estimated > 0: ratio is unbounded.
            estimated
        };
        Self {
            op_kind,
            estimated_card_sum: estimated,
            observed_rows_sum: observed,
            ratio,
            direction: BreachDirection::OverEstimate,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn under_estimate_ratio_is_observed_over_estimated() {
        // observed=1000, estimated=10 → ratio=100.0 (under-estimate by 100×).
        let b = ThresholdBreach::under_estimate(OperatorKind::Scan, 10.0, 1000);
        assert_eq!(b.ratio, 100.0);
        assert_eq!(b.direction, BreachDirection::UnderEstimate);
        assert_eq!(b.op_kind, OperatorKind::Scan);
        assert_eq!(b.estimated_card_sum, 10.0);
        assert_eq!(b.observed_rows_sum, 1000);
    }

    #[test]
    fn under_estimate_zero_estimated_renders_observed_as_ratio() {
        // estimated=0, observed=42 → ratio=42.0 (any positive observed
        // against a zero estimated is unbounded; we surface observed).
        let b = ThresholdBreach::under_estimate(OperatorKind::Scan, 0.0, 42);
        assert_eq!(b.ratio, 42.0);
    }

    #[test]
    fn over_estimate_ratio_is_estimated_over_observed() {
        // observed=10, estimated=1000 → ratio=100.0 (over-estimate by 100×).
        let b = ThresholdBreach::over_estimate(OperatorKind::Filter, 1000.0, 10);
        assert_eq!(b.ratio, 100.0);
        assert_eq!(b.direction, BreachDirection::OverEstimate);
    }

    #[test]
    fn over_estimate_zero_observed_renders_estimated_as_ratio() {
        // observed=0, estimated=42 → ratio=42.0.
        let b = ThresholdBreach::over_estimate(OperatorKind::Expand, 42.0, 0);
        assert_eq!(b.ratio, 42.0);
    }

    #[test]
    fn breach_direction_str_slugs_are_stable() {
        // Pinned for tracing-field consumer parsing.
        assert_eq!(BreachDirection::UnderEstimate.as_str(), "under_estimate");
        assert_eq!(BreachDirection::OverEstimate.as_str(), "over_estimate");
    }
}
