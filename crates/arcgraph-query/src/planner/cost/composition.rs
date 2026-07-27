//! Compositional selectivity rules for the M4-51 cost planner.
//!
//! # Why a single helper module?
//!
//! Per PR #172 (M4-42) review Finding 4 (verbatim): "M4-05 implements
//! compositional selectivity rules (AND: product; OR: 1-(1-s1)(1-s2);
//! NOT: 1-s) at the cost-planner layer. The composition site is a
//! single helper module (NOT scattered across plan-cost computations)
//! so v1.1 sketch-aware composition (correlation-aware AND/OR) can
//! swap cleanly."
//!
//! Spreading composition across [`crate::planner::cost::operator`]'s
//! per-operator cost functions would lock the v1.0 independence
//! assumption into every cost site; pulling composition HERE keeps
//! the v1.1 swap a one-file edit.
//!
//! # v1.0 composition rules
//!
//! Under the **independence assumption** (each predicate's
//! selectivity is independent of the others):
//!
//! | Operation | Formula | Intuition |
//! |-----------|---------|-----------|
//! | `s1 AND s2` | `s1 * s2` | Both predicates pass independently. |
//! | `s1 OR s2` | `1 - (1 - s1) * (1 - s2)` | At least one passes (inclusion-exclusion). |
//! | `s1 XOR s2` | `s1 + s2 - 2 * s1 * s2` | Exactly one passes (#621). |
//! | `NOT s` | `1 - s` | Complement of the predicate. |
//! | `AND(s1, ..., sN)` | `Π sᵢ` | n-ary fold of binary AND. |
//! | `OR(s1, ..., sN)` | `1 - Π (1 - sᵢ)` | n-ary fold of binary OR. |
//!
//! # v1.1 swap site (forward-link)
//!
//! When v1.1 lights correlation-aware composition (per ADR-038
//! amendment-03 §M4-04c property-value sketches forward-note), the
//! signature stays `f64 → f64 → f64`; only the inner formula changes.
//! Sketch-aware composition reads from a per-tenant correlation
//! matrix or a t-digest of joint distributions, swapping the v1.0
//! independence-product rule for an empirically-tuned correction.
//! Because every cost-site composes through THIS module, the v1.1
//! swap is a one-file change.
//!
//! # Bounds
//!
//! Every helper clamps its output into `[0.0, 1.0]`. Selectivity is
//! a probability; the cost model assumes the bound. NaN / Inf inputs
//! are clamped to 0.0 (safe degradation) — defense-in-depth against
//! a future formula refinement that could violate the unit-interval
//! invariant. Mirrors the
//! [`crate::semantic::selectivity::SelectivityEstimator`] discipline
//! (per codex M4-42 review N1).
//!
//! # ADR provenance
//! - PR #172 (M4-42) review Finding 4 — single-helper-module
//!   composition site; v1.1 swap discipline.
//! - ADR-038 §2 D-27 — selectivity-estimator surface (the per-
//!   predicate inputs this module composes).
//! - ADR-038 amendment-03 §M4-04c — v1.1 sketch-aware composition
//!   forward-note.

/// Compose two AND-ed selectivities under the v1.0 independence
/// assumption. Returns `s1 * s2`, with inputs clamped to `[0.0, 1.0]`
/// before multiplying — defense-in-depth against a future formula
/// refinement that could feed out-of-bounds inputs.
///
/// **v1.1 swap site.** Correlation-aware composition reads from a
/// joint-distribution sketch instead of multiplying.
#[inline]
#[must_use]
pub fn compose_and(s1: f64, s2: f64) -> f64 {
    clamp_unit(clamp_unit(s1) * clamp_unit(s2))
}

/// Compose two OR-ed selectivities under the v1.0 independence
/// assumption. Returns `1 - (1 - s1) * (1 - s2)`, with inputs
/// clamped to `[0.0, 1.0]` and the output clamped likewise.
/// Inclusion-exclusion principle for two events modeled as
/// independent.
///
/// **v1.1 swap site.** Correlation-aware composition adjusts for
/// joint-distribution overlap.
#[inline]
#[must_use]
pub fn compose_or(s1: f64, s2: f64) -> f64 {
    let c1 = clamp_unit(s1);
    let c2 = clamp_unit(s2);
    clamp_unit(1.0 - (1.0 - c1) * (1.0 - c2))
}

/// Compose two XOR-ed selectivities under the v1.0 independence
/// assumption (#621). Returns `s1 + s2 - 2 * s1 * s2` — the
/// probability that EXACTLY ONE of two independent events occurs
/// (`s1·(1−s2) + (1−s1)·s2`), with inputs and output clamped to
/// `[0.0, 1.0]`.
///
/// Mirrors [`compose_or`] / [`compose_and`] exactly (same clamp
/// discipline, same `f64 → f64 → f64` signature); only the formula
/// differs. For `s1 = s2 = 0.5` this returns `0.5` (two fair
/// independent coins differ half the time); `XOR` with a certain
/// operand (`s2 = 1.0`) reduces to `1 − s1` (the complement), and
/// with an impossible operand (`s2 = 0.0`) reduces to `s1`.
///
/// **v1.1 swap site.** Correlation-aware composition adjusts for
/// joint-distribution overlap, like the AND/OR helpers.
#[inline]
#[must_use]
pub fn compose_xor(s1: f64, s2: f64) -> f64 {
    let c1 = clamp_unit(s1);
    let c2 = clamp_unit(s2);
    clamp_unit(c1 + c2 - 2.0 * c1 * c2)
}

/// Compose a NOT-ed selectivity. Returns `1 - s`, with input clamped
/// to `[0.0, 1.0]`. Standard complement; NO v1.1 correlation
/// adjustment is required (NOT is a unary operator over a single
/// predicate's selectivity).
#[inline]
#[must_use]
pub fn compose_not(s: f64) -> f64 {
    clamp_unit(1.0 - clamp_unit(s))
}

/// n-ary AND fold over a slice of selectivities. Equivalent to
/// `selectivities.iter().fold(1.0, compose_and)` modulo intermediate
/// clamping. The empty slice returns `1.0` (the identity for product;
/// "no predicates" = "all rows pass").
///
/// **Numerical stability.** For very long predicate chains (`N > 50`)
/// the product can underflow to denormal / zero in f64; v1.0 plans
/// rarely exceed `N = 5`–`10` predicates per WHERE clause so this is
/// a non-issue at v1.0 scale. The v1.1 sketch-aware swap can switch
/// to log-space accumulation if needed.
#[inline]
#[must_use]
pub fn compose_n_ary_and(selectivities: &[f64]) -> f64 {
    let product = selectivities
        .iter()
        .map(|s| clamp_unit(*s))
        .product::<f64>();
    clamp_unit(product)
}

/// n-ary OR fold over a slice of selectivities. Equivalent to
/// `1 - Π (1 - sᵢ)` clamped to `[0.0, 1.0]`. The empty slice returns
/// `0.0` (the identity for OR; "no predicates" = "no rows pass").
///
/// **Why not `selectivities.iter().fold(0.0, compose_or)`?** The
/// fold is mathematically equivalent (OR is associative for
/// independent events) but the explicit complement-product form is
/// numerically more stable for chains of small selectivities — each
/// term `(1 - sᵢ)` stays close to 1.0, so the product accumulates
/// without rapid underflow.
#[inline]
#[must_use]
pub fn compose_n_ary_or(selectivities: &[f64]) -> f64 {
    if selectivities.is_empty() {
        return 0.0;
    }
    let complement_product = selectivities
        .iter()
        .map(|s| 1.0 - clamp_unit(*s))
        .product::<f64>();
    clamp_unit(1.0 - complement_product)
}

/// Clamp a finite f64 into the closed unit interval `[0.0, 1.0]`.
///
/// Defense-in-depth: a future formula refinement (correlation-aware
/// AND/OR, sketch-aware composition) that produces NaN / Inf is a
/// bug, not a graceful-degradation path. The `debug_assert` fires in
/// dev / debug builds so the regression is caught at the test
/// boundary; the `is_finite` clamp keeps production safe
/// (`debug_assert` is a no-op in release). Mirrors
/// [`crate::semantic::selectivity::clamp_unit`] (private to that
/// module, hence duplicated here per single-responsibility — the
/// composition module owns its own clamping invariant).
#[inline]
fn clamp_unit(x: f64) -> f64 {
    debug_assert!(
        x.is_finite(),
        "compose_*: input must be finite (got {x}); a future composition \
         refinement is feeding NaN/Inf — this is a bug, not a graceful-\
         degradation path."
    );
    if !x.is_finite() {
        return 0.0;
    }
    x.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // Pinned formulas (independence-assumption v1.0 baseline).
    // -----------------------------------------------------------------

    #[test]
    fn compose_and_pins_independence_product() {
        // Independence: s1 * s2.
        assert!((compose_and(0.5, 0.5) - 0.25).abs() < 1e-12);
        assert!((compose_and(0.1, 0.2) - 0.02).abs() < 1e-12);
        // Identities: 0.0 absorbs, 1.0 is identity.
        assert_eq!(compose_and(0.0, 0.7), 0.0);
        assert_eq!(compose_and(1.0, 0.7), 0.7);
    }

    #[test]
    fn compose_or_pins_inclusion_exclusion() {
        // Inclusion-exclusion: 1 - (1 - s1)(1 - s2).
        // For s1=s2=0.5 → 1 - 0.25 = 0.75.
        assert!((compose_or(0.5, 0.5) - 0.75).abs() < 1e-12);
        // For s1=0.1, s2=0.2 → 1 - 0.9*0.8 = 1 - 0.72 = 0.28.
        assert!((compose_or(0.1, 0.2) - 0.28).abs() < 1e-12);
        // Identities: 0.0 is identity, 1.0 absorbs.
        assert_eq!(compose_or(0.0, 0.7), 0.7);
        assert_eq!(compose_or(1.0, 0.7), 1.0);
    }

    #[test]
    fn compose_xor_pins_exactly_one() {
        // Exactly-one: s1 + s2 - 2*s1*s2.
        // For s1=s2=0.5 → 0.5 + 0.5 - 0.5 = 0.5.
        assert!((compose_xor(0.5, 0.5) - 0.5).abs() < 1e-12);
        // For s1=0.1, s2=0.2 → 0.3 - 2*0.02 = 0.3 - 0.04 = 0.26.
        assert!((compose_xor(0.1, 0.2) - 0.26).abs() < 1e-12);
        // Boundary reductions: a certain operand (1.0) → complement of
        // the other; an impossible operand (0.0) → the other unchanged.
        assert_eq!(compose_xor(1.0, 0.7), compose_not(0.7));
        assert!((compose_xor(0.0, 0.7) - 0.7).abs() < 1e-12);
        // Both certain → 0 (true XOR true = false); both impossible → 0.
        assert_eq!(compose_xor(1.0, 1.0), 0.0);
        assert_eq!(compose_xor(0.0, 0.0), 0.0);
    }

    #[test]
    fn compose_not_pins_complement() {
        assert_eq!(compose_not(0.0), 1.0);
        assert_eq!(compose_not(1.0), 0.0);
        assert!((compose_not(0.3) - 0.7).abs() < 1e-12);
    }

    // -----------------------------------------------------------------
    // n-ary folds.
    // -----------------------------------------------------------------

    #[test]
    fn compose_n_ary_and_empty_is_identity() {
        // Empty AND = "no predicates" = all rows pass = 1.0.
        assert_eq!(compose_n_ary_and(&[]), 1.0);
    }

    #[test]
    fn compose_n_ary_and_three_way_matches_pairwise() {
        let three = [0.5, 0.4, 0.2];
        let n_ary = compose_n_ary_and(&three);
        let pairwise = compose_and(compose_and(0.5, 0.4), 0.2);
        assert!(
            (n_ary - pairwise).abs() < 1e-12,
            "n-ary and binary fold must agree (got n_ary={n_ary}, pairwise={pairwise})"
        );
        // Direct check: 0.5 * 0.4 * 0.2 = 0.04.
        assert!((n_ary - 0.04).abs() < 1e-12);
    }

    #[test]
    fn compose_n_ary_or_empty_is_zero() {
        // Empty OR = "no predicates" = no rows pass = 0.0.
        assert_eq!(compose_n_ary_or(&[]), 0.0);
    }

    #[test]
    fn compose_n_ary_or_three_way_matches_pairwise() {
        let three = [0.5, 0.4, 0.2];
        let n_ary = compose_n_ary_or(&three);
        // 1 - 0.5 * 0.6 * 0.8 = 1 - 0.24 = 0.76
        assert!((n_ary - 0.76).abs() < 1e-12);
    }

    // -----------------------------------------------------------------
    // Bounds + degenerate inputs.
    // -----------------------------------------------------------------

    #[test]
    fn compose_clamps_inputs_above_one_and_below_zero() {
        // A future formula refinement could feed > 1 or < 0 inputs;
        // the clamp must defend.
        assert_eq!(compose_and(1.5, 0.5), 0.5);
        assert_eq!(compose_or(0.5, 1.5), 1.0);
        assert_eq!(compose_not(-0.1), 1.0);
        assert_eq!(compose_n_ary_and(&[1.5, 0.5]), 0.5);
        assert_eq!(compose_n_ary_or(&[-0.1, 0.5]), 0.5);
    }

    #[test]
    fn compose_results_stay_in_unit_interval() {
        for s1 in [0.0_f64, 0.1, 0.5, 0.9, 1.0] {
            for s2 in [0.0_f64, 0.1, 0.5, 0.9, 1.0] {
                let and_val = compose_and(s1, s2);
                let or_val = compose_or(s1, s2);
                assert!(
                    (0.0..=1.0).contains(&and_val),
                    "compose_and({s1}, {s2}) = {and_val} out of [0, 1]"
                );
                assert!(
                    (0.0..=1.0).contains(&or_val),
                    "compose_or({s1}, {s2}) = {or_val} out of [0, 1]"
                );
            }
        }
        for s in [0.0_f64, 0.1, 0.5, 0.9, 1.0] {
            let not_val = compose_not(s);
            assert!(
                (0.0..=1.0).contains(&not_val),
                "compose_not({s}) = {not_val} out of [0, 1]"
            );
        }
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "compose_*: input must be finite")]
    fn debug_asserts_on_nan_input() {
        let _ = compose_and(f64::NAN, 0.5);
    }
}
