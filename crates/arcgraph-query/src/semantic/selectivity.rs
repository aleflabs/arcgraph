//! M4-42 (M4-04b) — Selectivity estimators per predicate class.
//!
//! # Budget
//!
//! Under the performance-budget discipline (back-of-envelope before
//! implementation). The M4-05 cost-based planner is the load-bearing
//! consumer; selectivity is on the cost-planner hot path.
//!
//! - **`estimate_*` p99 ≤ 50ns** at 1M-row tenant; called
//!   `O(plan-nodes × predicates)` per query. Each call is 1–2 atomic
//!   loads on `CatalogStats` + 1 division + 1 `clamp_unit`.
//! - **No allocations** in the hot path; estimator returns `f64`.
//! - Bench: `cargo bench -p arcgraph-query --bench selectivity_estimator`
//!   establishes the baseline; future stats-redesign regressions >2×
//!   are caught there (per codex M4-2x retro Sin #1).
//!
//! [`SelectivityEstimator`] is a thin computation surface over a
//! [`CatalogProvider`] that turns the per-tenant cardinality stats
//! M4-41 collects into per-predicate selectivity factors `f ∈ [0, 1]`.
//! Selectivity is the fraction of input rows a predicate is expected
//! to retain; the M4-05 cost-based planner consumes these factors to
//! pick join orders and filter-pushdown plans.
//!
//! # Design discipline (6-slice 3-strike pattern)
//!
//! M4-42 ships a CONCRETE STRUCT, not a `pub trait
//! SelectivityEstimator`. The 6-slice precedent (M4-21 / M4-22 /
//! M4-22b / M4-23 / M4-31 — every walker concretized; zero
//! speculative traits across the chain) extends through M4-42. The
//! single in-flight consumer is the not-yet-shipped M4-05 cost
//! planner; per `feedback_avoid_speculative_scaffolding.md`, ship the
//! abstraction when there are ≥2 real consumers, not when there is
//! one imagined consumer. If a second consumer materialises (e.g., an
//! adaptive replanner that reads selectivity at execute-time), THAT
//! slice introduces the trait — not this one.
//!
//! # v1.0 estimator family
//!
//! Five concrete estimators, one per predicate class:
//!
//! | Method | Predicate shape | v1.0 formula |
//! |--------|-----------------|--------------|
//! | [`SelectivityEstimator::estimate_eq`] | `n.<prop> = <v>` | `1 / total_node_count` (uniform-distribution) |
//! | [`SelectivityEstimator::estimate_lt`] | `n.<prop> < <v>` | constant `DEFAULT_LT_SELECTIVITY` (no histograms at v1.0) |
//! | [`SelectivityEstimator::estimate_in`] | `n.<prop> IN (v1, …, vN)` | `min(N, total) / total_node_count` |
//! | [`SelectivityEstimator::estimate_label`] | `MATCH (n:Label)` | `label_cardinality / total_node_count` |
//! | [`SelectivityEstimator::estimate_rel_type`] | `MATCH ()-[r:TYPE]-()` | `rel_type_cardinality / total_rel_count` |
//!
//! Each estimator returns a finite `f64 ∈ [0.0, 1.0]`. **Never NaN,
//! never Inf, never negative, never > 1.0.** The `estimate_*`
//! contract is a hard invariant the M4-05 planner will lean on for
//! cost monotonicity; the proptest in
//! `tests/selectivity_proptest.rs` pins it across 256 random inputs.
//!
//! # Stats=None graceful degradation (per M4-41 §D-25 contract)
//!
//! When the underlying [`CatalogProvider`] returns `None` for a
//! cardinality (the "no stats collected yet" sentinel — fresh
//! tenant, cold restart per M4-41 persistence/recovery, or a label
//! that has never been observed), the estimator returns the
//! `DEFAULT_*_SELECTIVITY` constant for that predicate class. NEVER
//! panic, NEVER divide by zero, NEVER return Inf/NaN.
//!
//! The constants are mainstream cost-planner defaults (lineage:
//! Selinger et al., System R, SIGMOD 1979) used when no histograms /
//! sketches / samples are available. They are NOT a literal SQL
//! standard — there is no standardised clause prescribing 0.1 / 0.33 /
//! 0.2 / 0.5; the values are textbook teaching defaults that real
//! engines tune downward (PostgreSQL: eq=0.005, range=0.3333,
//! range_ineq=0.005; Oracle: 1/distinct fallbacks). They are NOT
//! load-bearing — once stats are collected, the formula path takes
//! over; the constants only matter for the cold-start window. The
//! M4-71 row-count observer feedback loop (forward-link below) makes
//! them progressively less load-bearing over a tenant's lifetime.
//!
//! # Stats=Some(0) — observed-then-fully-deleted
//!
//! A cardinality of `Some(0)` is distinct from `None` per M4-41:
//! `Some(0)` means "observed by the commit pipeline at least once,
//! then every record carrying it was tombstoned". The estimators
//! return `0.0` in this case (no rows can match a predicate against
//! an empty table). This sidesteps the `1/0` Inf path and the
//! `0/0` NaN path that a naïve `cardinality / total` formula would
//! produce.
//!
//! # Range / property-value sketch deferral (v1.1)
//!
//! Per ADR-038 amendment-03 §M4-04 decomposition (M4-04c sub-slice),
//! property-value histograms (equi-depth) and t-digest sketches are
//! deferred to v1.1. M4-42's [`SelectivityEstimator::estimate_lt`]
//! returns `DEFAULT_LT_SELECTIVITY` (the Selinger 1979 / textbook `0.33`
//! range constant) at v1.0; v1.1 will refine this when sketches land.
//! The v1.1 swap is signature-preserving — `estimate_lt` will read
//! the sketch when present, fall back to the constant when absent.
//!
//! # M4-71 row-count observer forward-link
//!
//! Per ADR-038 amendment-03 §"Implicit dependency edges" item 4, the
//! M4-71 row-count observer feeds OBSERVED cardinalities back into
//! `CatalogStats`. M4-42 reads from `CatalogProvider` (which reads
//! from `CatalogStats`), so the feedback loop closes naturally:
//!
//! ```text
//!     M4-71 RowCountObserver → CatalogStats → CatalogProvider → SelectivityEstimator
//! ```
//!
//! NO inline change is needed in M4-42 when M4-71 ships; the
//! architecture is feedback-friendly by construction. The forward-
//! link is documented here so future M4-71 implementers can confirm
//! the consumer surface they feed.
//!
//! # ADR provenance
//! - ADR-038 §2 D-25 — M4-41 catalog stats schema (the read source).
//! - ADR-038 §2 D-27 — M4-42 selectivity estimators per predicate
//!   class (this module — closure paragraph in the ADR).
//! - ADR-038 amendment-03 §M4-42 row — slice scope + test artifacts.
//! - ADR-038 amendment-03 §M4-04 decomposition — M4-04c v1.1 deferral
//!   for histograms / sketches.
//! - ADR-038 amendment-03 §"Implicit dependency edges" item 4 —
//!   M4-71 row-count observer forward-link.

use arcgraph_core::{LabelId, TypeId};

use crate::semantic::bound_ast::BindingId;
use crate::semantic::catalog::CatalogProvider;

/// Default selectivity for `n.<prop> = <value>` when the catalog has
/// no `total_node_count` collected.
///
/// **M4-04d empirical anchor (2026-05-06):** the 1M LDBC SNB Person
/// back-test in `tests/m4_04d_selectivity_backtest.rs` measured a
/// pooled `eq` p50 of `1.600e-4` across 350 random predicates spanning
/// `firstName / lastName / gender / age / birthday / browser /
/// locationIP`. This constant is set to `0.0002` — within 2× of the
/// empirical p50 (ratio 1.25×) per ADR-038 amendment-07. The original
/// Selinger 1979 / textbook teaching default of `0.1` was 625× too
/// high for SNB-style workloads.
///
/// **Why not PostgreSQL's `0.005`?** PG's `eqsel` is production-tuned
/// for an "average column with ~200 distinct values"; SNB Person
/// columns range from 2 distinct (gender) to ~unique (locationIP), so
/// the SNB-fixture-empirical anchor is more directly load-bearing for
/// our cold-start. PG's value would still leave us 30× off from the
/// SNB empirical p50 (outside the 2× regression-pin band).
///
/// The M4-71 row-count observer feedback should make this constant
/// less load-bearing over time (per ADR-038 D-27 + amendment-03
/// §"Implicit dependency edges" #4); it only fires when
/// `total_node_count()` returns `None` (true cold-start).
pub const DEFAULT_EQ_SELECTIVITY: f64 = 0.0002;

/// Default selectivity for `n.<prop> < <value>` (and other range
/// predicates) when no property-value histograms are available.
///
/// **M4-04d empirical anchor (2026-05-06):** the 1M LDBC SNB Person
/// back-test measured a pooled `lt` p50 of `4.912e-1` across 150
/// random predicates spanning `age / birthday / creationDate`. The
/// current `0.33` constant is within 2× of empirical (ratio 1.49×) —
/// **KEPT** at the Selinger 1979 / PostgreSQL `ineqsel = 0.3333…`
/// default per ADR-038 amendment-07.
///
/// v1.1 (M4-04c) refines this with equi-depth histograms / t-digest;
/// the v1.1 swap is signature-preserving — `estimate_lt` will read
/// the sketch when present, fall back to the constant when absent.
pub const DEFAULT_LT_SELECTIVITY: f64 = 0.33;

/// Default selectivity for `n.<prop> IN (v1, …, vN)` when no stats
/// are collected.
///
/// **M4-04d empirical anchor (2026-05-06):** the 1M LDBC SNB Person
/// back-test measured a pooled `in` p50 of `4.981e-3` across 150
/// random IN-lists (3- and 10-element `firstName` lists, 3-element
/// `speaks` language lists). This constant is set to `0.005` — within
/// 2× of the empirical p50 (ratio 1.004×) per ADR-038 amendment-07.
/// The original `0.2` (Selinger 1979 lineage, ~2× textbook eq) was 40×
/// too high for typical IN-list workloads.
///
/// Note: this constant only fires when `total_node_count()` returns
/// `None`; once stats are present, the formula path computes
/// `min(N, total) / total_node_count` directly, which is accurate.
/// The constant's role is the cold-start fallback.
pub const DEFAULT_IN_SELECTIVITY: f64 = 0.005;

/// Default selectivity for `MATCH (n:Label)` when neither
/// `label_cardinality` nor `total_node_count` is collected.
///
/// **M4-04d empirical anchor (2026-05-06):** the 1M LDBC SNB Person
/// back-test measured a multi-label tenant p50 of `9.001e-2` across
/// `[Person 0.9, Comment 0.09, Forum 0.009, Place 0.0009]`. This
/// constant is set to `0.1` — within 2× of the empirical p50 (ratio
/// 1.11×) per ADR-038 amendment-07. The original `0.5` (a "the label
/// is widely distributed" textbook default) was 5.6× too high; in
/// realistic multi-label graphs (SNB has 6+ node types), most labels
/// are minority labels and a `0.5` cold-start biases the planner
/// toward full-table-scan plans.
///
/// Once stats are present, the formula path
/// (`label_cardinality / total_node_count`) takes over; the constant
/// is the cold-start fallback only.
pub const DEFAULT_LABEL_SELECTIVITY: f64 = 0.1;

/// Default selectivity for `MATCH ()-[r:TYPE]-()` when neither
/// `rel_type_cardinality` nor `total_rel_count` is collected.
///
/// **M4-04d empirical anchor (2026-05-06):** the 1M LDBC SNB Person
/// back-test measured a multi-rel-type tenant p50 of `2.500e-1` across
/// `[KNOWS 0.6, LIKES 0.25, IS_LOCATED_IN 0.15]`. This constant is set
/// to `0.25` — matching empirical p50 exactly (ratio 1.0×) per ADR-038
/// amendment-07. The original `0.5` was at the strict 2× boundary;
/// tuning to `0.25` puts the constant inside the 2× regression-pin
/// band and aligns with the SNB-fixture empirical observation.
///
/// rel-type distributions in graph workloads are typically less skewed
/// than label distributions (the dominant rel-type frequently captures
/// 50-70 % of all edges, vs labels which often have a 10× step-down
/// hierarchy), so the rel-type cold-start default is a touch higher
/// than [`DEFAULT_LABEL_SELECTIVITY`].
pub const DEFAULT_REL_TYPE_SELECTIVITY: f64 = 0.25;

/// Concrete selectivity estimator that wraps a [`CatalogProvider`].
///
/// Stateless; cheap to construct. The borrow lifetime `'cat` ties the
/// estimator to the catalog it reads from — callers typically build a
/// fresh [`SelectivityEstimator`] for each query plan and discard it
/// once cost evaluation completes.
///
/// See the [module docs](self) for the v1.0 estimator family and the
/// stats=None / stats=Some(0) graceful-degradation contract.
pub struct SelectivityEstimator<'cat, C: CatalogProvider + ?Sized> {
    catalog: &'cat C,
}

impl<'cat, C: CatalogProvider + ?Sized> SelectivityEstimator<'cat, C> {
    /// Construct an estimator that reads from `catalog`.
    pub fn new(catalog: &'cat C) -> Self {
        Self { catalog }
    }

    /// Estimate the selectivity of `n.<prop> = <value>`.
    ///
    /// v1.0: `1 / total_node_count` under the uniform-distribution
    /// assumption. Returns [`DEFAULT_EQ_SELECTIVITY`] when
    /// `total_node_count()` is `None`; returns `0.0` when it is
    /// `Some(0)` (observed-then-fully-deleted).
    ///
    /// `var` is the binding the predicate filters; `label` is the
    /// label scoping that binding (if any). Both parameters are
    /// reserved for v1.1 (M4-04c) per-label property-value histograms
    /// and are unused at v1.0 — the v1.0 formula is tenant-wide. The
    /// signature is forward-compatible: when histograms land, the
    /// `(var, label, prop)` triple keys into the per-label sketch
    /// without a signature change.
    pub fn estimate_eq(&self, _var: BindingId, _label: Option<LabelId>) -> f64 {
        match self.catalog.total_node_count() {
            None => DEFAULT_EQ_SELECTIVITY,
            Some(0) => 0.0,
            Some(total) => clamp_unit(1.0 / total as f64),
        }
    }

    /// Estimate the selectivity of `n.<prop> < <value>` (and other
    /// open-range predicates).
    ///
    /// v1.0: returns [`DEFAULT_LT_SELECTIVITY`] (the Selinger 1979 /
    /// textbook `33%` range constant) regardless of `var` / `label`, because no
    /// property-value histograms are collected at v1.0 — there is no
    /// principled way to refine a range estimate without sketches.
    /// `Some(0)` total returns `0.0` (empty table — no rows can match).
    ///
    /// **v1.1 forward-reference.** ADR-038 amendment-03 §M4-04
    /// decomposition (M4-04c sub-slice) deferred equi-depth histograms
    /// and t-digest sketches to v1.1. When they land, this method
    /// reads the sketch when present and falls back to the constant
    /// when absent — signature-preserving swap.
    pub fn estimate_lt(&self, _var: BindingId, _label: Option<LabelId>) -> f64 {
        match self.catalog.total_node_count() {
            Some(0) => 0.0,
            _ => DEFAULT_LT_SELECTIVITY,
        }
    }

    /// Estimate the selectivity of `n.<prop> IN (v1, …, vN)`.
    ///
    /// v1.0: `list_size / total_node_count`, clamped to `[0, 1]`
    /// (a list longer than the tenant cannot exceed full-table
    /// selectivity). Returns [`DEFAULT_IN_SELECTIVITY`] when
    /// `total_node_count()` is `None`; `0.0` when `Some(0)`. An empty
    /// list (`list_size == 0`) returns `0.0` — the predicate is
    /// trivially false. As with [`Self::estimate_eq`], `var` /
    /// `label` are reserved for v1.1 per-label sketches.
    pub fn estimate_in(&self, _var: BindingId, _label: Option<LabelId>, list_size: usize) -> f64 {
        if list_size == 0 {
            return 0.0;
        }
        match self.catalog.total_node_count() {
            None => DEFAULT_IN_SELECTIVITY,
            Some(0) => 0.0,
            Some(total) => clamp_unit(list_size as f64 / total as f64),
        }
    }

    /// Estimate the selectivity of a label filter — `MATCH (n:Label)`.
    ///
    /// v1.0: `label_cardinality(label) / total_node_count` when both
    /// are `Some`, clamped to `[0, 1]`. Returns
    /// [`DEFAULT_LABEL_SELECTIVITY`] when either is `None` (the
    /// catalog's "no stats" sentinel). Returns `0.0` when
    /// `total_node_count()` is `Some(0)`.
    pub fn estimate_label(&self, label: LabelId) -> f64 {
        let total = self.catalog.total_node_count();
        let card = self.catalog.label_cardinality(label);
        match (card, total) {
            (Some(_), Some(0)) => 0.0,
            (Some(c), Some(t)) => clamp_unit(c as f64 / t as f64),
            // Either stat is missing → fall back to the default.
            (_, None) | (None, _) => DEFAULT_LABEL_SELECTIVITY,
        }
    }

    /// Estimate the selectivity of a relationship-type filter —
    /// `MATCH ()-[r:TYPE]-()`.
    ///
    /// v1.0: `rel_type_cardinality(rel_type) / total_rel_count` when
    /// both are `Some`, clamped to `[0, 1]`. Returns
    /// [`DEFAULT_REL_TYPE_SELECTIVITY`] when either is `None`. Returns
    /// `0.0` when `total_rel_count()` is `Some(0)`.
    pub fn estimate_rel_type(&self, rel_type: TypeId) -> f64 {
        let total = self.catalog.total_rel_count();
        let card = self.catalog.rel_type_cardinality(rel_type);
        match (card, total) {
            (Some(_), Some(0)) => 0.0,
            (Some(c), Some(t)) => clamp_unit(c as f64 / t as f64),
            (_, None) | (None, _) => DEFAULT_REL_TYPE_SELECTIVITY,
        }
    }
}

/// Clamp a finite f64 into the closed unit interval `[0.0, 1.0]`.
///
/// Defends against any future formula refinement that could exceed
/// the unit interval (e.g., `list_size > total` in an IN-predicate
/// race against concurrent commits). Selectivity is a probability;
/// the planner cost model assumes the bound, so the proptest in
/// `tests/selectivity_proptest.rs` pins this invariant.
#[inline]
fn clamp_unit(x: f64) -> f64 {
    // Loud-on-dev, safe-on-prod: a future formula refinement that
    // produces NaN/Inf is a bug, not a graceful-degradation path.
    // The `debug_assert` fires in dev / debug builds so the regression
    // is caught at the test boundary; the `is_finite` clamp below
    // keeps production safe (debug_assert is a no-op in release).
    // Per codex M4-42 review N1.
    debug_assert!(
        x.is_finite(),
        "clamp_unit: input must be finite (got {x}); a future formula refinement \
         is feeding NaN/Inf — this is a bug, not a graceful-degradation path. \
         Per codex M4-42 review N1."
    );
    // Reject non-finite first — `f64::clamp` propagates NaN, and Inf
    // would round to 1.0 silently. The public API never feeds NaN/Inf
    // here at v1.0 (every divide-by-zero is matched explicitly), but a
    // future formula refinement could; 0.0 is the conservative
    // cost-model choice for an unrepresentable selectivity.
    if !x.is_finite() {
        return 0.0;
    }
    x.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::catalog::StubCatalogProvider;

    // -----------------------------------------------------------------
    // 1. Default-constant pinning (5 tests — one per constant).
    //
    // The constants are named in the public API and cited by the
    // M4-05 cost planner brief; if a future refactor changes them,
    // these pins force the change to be deliberate.
    // -----------------------------------------------------------------

    #[test]
    fn default_eq_selectivity_pinned() {
        // Tuned 2026-05-06 from 0.1 to 0.0002 per ADR-038
        // amendment-07 (M4-04d empirical back-test).
        assert_eq!(DEFAULT_EQ_SELECTIVITY, 0.0002);
    }

    #[test]
    fn default_lt_selectivity_pinned() {
        // KEPT at 0.33 per ADR-038 amendment-07 — the M4-04d empirical
        // p50 of 0.491 is within 2× of 0.33 (ratio 1.49×).
        assert_eq!(DEFAULT_LT_SELECTIVITY, 0.33);
    }

    #[test]
    fn default_in_selectivity_pinned() {
        // Tuned 2026-05-06 from 0.2 to 0.005 per ADR-038 amendment-07
        // (M4-04d empirical p50 = 4.981e-3, ratio 1.004×).
        assert_eq!(DEFAULT_IN_SELECTIVITY, 0.005);
    }

    #[test]
    fn default_label_selectivity_pinned() {
        // Tuned 2026-05-06 from 0.5 to 0.1 per ADR-038 amendment-07
        // (M4-04d empirical p50 = 9.001e-2, ratio 1.11×).
        assert_eq!(DEFAULT_LABEL_SELECTIVITY, 0.1);
    }

    #[test]
    fn default_rel_type_selectivity_pinned() {
        // Tuned 2026-05-06 from 0.5 to 0.25 per ADR-038 amendment-07
        // (M4-04d empirical p50 = 0.25, ratio 1.0×).
        assert_eq!(DEFAULT_REL_TYPE_SELECTIVITY, 0.25);
    }

    // -----------------------------------------------------------------
    // 1.b. M4-04d empirical-anchor regression pins.
    //
    // Per ADR-038 amendment-07 §Decision: every DEFAULT_*_SELECTIVITY
    // constant must stay within 2× of the empirical p50 measured by
    // `tests/m4_04d_selectivity_backtest.rs` on the 1M LDBC SNB Person
    // fixture (master seed 0x4D04D5E1EC07C0DE, 2026-05-06 anchor run).
    //
    // The anchor numbers are hard-coded here so this pin is fast (no
    // 1M-row sweep) and deterministic. If the empirical anchor shifts
    // (because the fixture is retuned, or the constants are
    // intentionally re-balanced against a different workload), update
    // BOTH this file's anchors AND amendment-07 in lockstep.
    //
    // The 2× threshold is the slice's pinning convention. Crossing it
    // is a signal that either the fixture drifted (re-run the
    // back-test, update the anchor) or the constant needs re-tuning
    // (write amendment-NN, update the constant + this anchor + the
    // pinning block above).
    // -----------------------------------------------------------------

    /// 1M LDBC SNB Person fixture (`tests/m4_04d_selectivity_backtest.rs`,
    /// master seed `0x4D04D5E1EC07C0DE`) — pooled empirical p50 anchors
    /// recorded 2026-05-06 on apple-silicon aarch64 / release.
    const M4_04D_EMPIRICAL_P50_EQ: f64 = 1.600e-4;
    const M4_04D_EMPIRICAL_P50_LT: f64 = 4.912e-1;
    const M4_04D_EMPIRICAL_P50_IN: f64 = 4.981e-3;
    const M4_04D_EMPIRICAL_P50_LABEL: f64 = 9.001e-2;
    const M4_04D_EMPIRICAL_P50_REL_TYPE: f64 = 2.500e-1;

    /// The pinning threshold per ADR-038 amendment-07: every constant
    /// must be within `2.0×` of its M4-04d empirical p50 anchor.
    const M4_04D_PIN_RATIO: f64 = 2.0;

    /// Compute the symmetric-divergence ratio
    /// `max(a/b, b/a)` for two positive f64s. Equal to `1.0` when they
    /// match exactly; grows with multiplicative drift in either
    /// direction. Both inputs MUST be > 0 (otherwise the test is
    /// meaningless — caller should assert positivity first).
    fn ratio(a: f64, b: f64) -> f64 {
        debug_assert!(a > 0.0 && b > 0.0, "ratio: both inputs must be > 0");
        (a / b).max(b / a)
    }

    #[test]
    fn default_eq_within_2x_of_m4_04d_empirical_p50() {
        let r = ratio(DEFAULT_EQ_SELECTIVITY, M4_04D_EMPIRICAL_P50_EQ);
        assert!(
            r <= M4_04D_PIN_RATIO,
            "DEFAULT_EQ_SELECTIVITY ({DEFAULT_EQ_SELECTIVITY}) drifted from \
             M4-04d empirical p50 ({M4_04D_EMPIRICAL_P50_EQ}); ratio = {r:.2}× \
             (limit {M4_04D_PIN_RATIO:.1}×). Re-run the 1M back-test \
             (`cargo test -p arcgraph-query --release \
             m4_04d_empirical_selectivity_backtest -- --ignored --nocapture`) \
             and write a new ADR-038 amendment if the constant needs re-tuning.",
        );
    }

    #[test]
    fn default_lt_within_2x_of_m4_04d_empirical_p50() {
        let r = ratio(DEFAULT_LT_SELECTIVITY, M4_04D_EMPIRICAL_P50_LT);
        assert!(
            r <= M4_04D_PIN_RATIO,
            "DEFAULT_LT_SELECTIVITY ({DEFAULT_LT_SELECTIVITY}) drifted from \
             M4-04d empirical p50 ({M4_04D_EMPIRICAL_P50_LT}); ratio = {r:.2}× \
             (limit {M4_04D_PIN_RATIO:.1}×).",
        );
    }

    #[test]
    fn default_in_within_2x_of_m4_04d_empirical_p50() {
        let r = ratio(DEFAULT_IN_SELECTIVITY, M4_04D_EMPIRICAL_P50_IN);
        assert!(
            r <= M4_04D_PIN_RATIO,
            "DEFAULT_IN_SELECTIVITY ({DEFAULT_IN_SELECTIVITY}) drifted from \
             M4-04d empirical p50 ({M4_04D_EMPIRICAL_P50_IN}); ratio = {r:.2}× \
             (limit {M4_04D_PIN_RATIO:.1}×).",
        );
    }

    #[test]
    fn default_label_within_2x_of_m4_04d_empirical_p50() {
        let r = ratio(DEFAULT_LABEL_SELECTIVITY, M4_04D_EMPIRICAL_P50_LABEL);
        assert!(
            r <= M4_04D_PIN_RATIO,
            "DEFAULT_LABEL_SELECTIVITY ({DEFAULT_LABEL_SELECTIVITY}) drifted \
             from M4-04d empirical p50 ({M4_04D_EMPIRICAL_P50_LABEL}); \
             ratio = {r:.2}× (limit {M4_04D_PIN_RATIO:.1}×).",
        );
    }

    #[test]
    fn default_rel_type_within_2x_of_m4_04d_empirical_p50() {
        let r = ratio(DEFAULT_REL_TYPE_SELECTIVITY, M4_04D_EMPIRICAL_P50_REL_TYPE);
        assert!(
            r <= M4_04D_PIN_RATIO,
            "DEFAULT_REL_TYPE_SELECTIVITY ({DEFAULT_REL_TYPE_SELECTIVITY}) \
             drifted from M4-04d empirical p50 ({M4_04D_EMPIRICAL_P50_REL_TYPE}); \
             ratio = {r:.2}× (limit {M4_04D_PIN_RATIO:.1}×).",
        );
    }

    // -----------------------------------------------------------------
    // 2. Per-estimator happy-path (5 tests — known stats → formula).
    // -----------------------------------------------------------------

    #[test]
    fn estimate_eq_uses_total_node_count_when_present() {
        let cat = StubCatalogProvider::new().with_total_node_count(1_000);
        let est = SelectivityEstimator::new(&cat);
        let s = est.estimate_eq(BindingId::new(0), None);
        // Uniform-distribution: 1/1000 = 0.001
        assert!((s - 0.001).abs() < 1e-12, "got {s}");
    }

    #[test]
    fn estimate_lt_returns_default_when_total_present() {
        // estimate_lt is a constant at v1.0 (no sketches). Stats=Some
        // does NOT change the answer; only Some(0) flips to 0.0 (the
        // empty-table corner case). Sketch-aware refinement is
        // deferred to v1.1 per amendment-03 M4-04c.
        let cat = StubCatalogProvider::new().with_total_node_count(1_000);
        let est = SelectivityEstimator::new(&cat);
        let s = est.estimate_lt(BindingId::new(0), None);
        assert_eq!(s, DEFAULT_LT_SELECTIVITY);
    }

    #[test]
    fn estimate_in_uses_list_size_over_total_when_present() {
        let cat = StubCatalogProvider::new().with_total_node_count(1_000);
        let est = SelectivityEstimator::new(&cat);
        let s = est.estimate_in(BindingId::new(0), None, 5);
        assert!((s - 0.005).abs() < 1e-12, "got {s}");
    }

    #[test]
    fn estimate_label_uses_label_card_over_total_when_present() {
        let l = LabelId::new(1);
        let cat = StubCatalogProvider::new()
            .with_label_cardinality(l, 250)
            .with_total_node_count(1_000);
        let est = SelectivityEstimator::new(&cat);
        let s = est.estimate_label(l);
        assert!((s - 0.25).abs() < 1e-12, "got {s}");
    }

    #[test]
    fn estimate_rel_type_uses_type_card_over_total_when_present() {
        let t = TypeId::new(1);
        let cat = StubCatalogProvider::new()
            .with_rel_type_cardinality(t, 100)
            .with_total_rel_count(2_000);
        let est = SelectivityEstimator::new(&cat);
        let s = est.estimate_rel_type(t);
        assert!((s - 0.05).abs() < 1e-12, "got {s}");
    }

    // -----------------------------------------------------------------
    // 3. Stats=None corner case (1 test — every estimator falls back
    //    to the documented DEFAULT_*_SELECTIVITY constant).
    // -----------------------------------------------------------------

    #[test]
    fn all_estimators_fall_back_to_defaults_when_stats_none() {
        // Empty stub: no `with_*_count()` calls. Every CatalogProvider
        // stats method returns None — the documented "no stats
        // collected" sentinel per M4-41 §D-25. Each estimator MUST
        // return its DEFAULT_*_SELECTIVITY constant.
        let cat = StubCatalogProvider::new();
        let est = SelectivityEstimator::new(&cat);

        assert_eq!(
            est.estimate_eq(BindingId::new(0), None),
            DEFAULT_EQ_SELECTIVITY
        );
        assert_eq!(
            est.estimate_lt(BindingId::new(0), None),
            DEFAULT_LT_SELECTIVITY
        );
        assert_eq!(
            est.estimate_in(BindingId::new(0), None, 5),
            DEFAULT_IN_SELECTIVITY
        );
        assert_eq!(
            est.estimate_label(LabelId::new(1)),
            DEFAULT_LABEL_SELECTIVITY
        );
        assert_eq!(
            est.estimate_rel_type(TypeId::new(1)),
            DEFAULT_REL_TYPE_SELECTIVITY
        );
    }

    // -----------------------------------------------------------------
    // 4. Stats=Some(0) corner case — observed-then-fully-deleted (5
    //    tests, one per estimator). The empty-table case must return
    //    0.0, not Inf (1/0) or NaN (0/0) or the default constant.
    // -----------------------------------------------------------------

    #[test]
    fn estimate_eq_returns_zero_when_total_is_some_zero() {
        let cat = StubCatalogProvider::new().with_total_node_count(0);
        let est = SelectivityEstimator::new(&cat);
        assert_eq!(est.estimate_eq(BindingId::new(0), None), 0.0);
    }

    #[test]
    fn estimate_lt_returns_zero_when_total_is_some_zero() {
        let cat = StubCatalogProvider::new().with_total_node_count(0);
        let est = SelectivityEstimator::new(&cat);
        assert_eq!(est.estimate_lt(BindingId::new(0), None), 0.0);
    }

    #[test]
    fn estimate_in_returns_zero_when_total_is_some_zero() {
        let cat = StubCatalogProvider::new().with_total_node_count(0);
        let est = SelectivityEstimator::new(&cat);
        assert_eq!(est.estimate_in(BindingId::new(0), None, 5), 0.0);
    }

    #[test]
    fn estimate_label_returns_zero_when_total_is_some_zero() {
        let l = LabelId::new(1);
        let cat = StubCatalogProvider::new()
            .with_label_cardinality(l, 0)
            .with_total_node_count(0);
        let est = SelectivityEstimator::new(&cat);
        assert_eq!(est.estimate_label(l), 0.0);
    }

    #[test]
    fn estimate_rel_type_returns_zero_when_total_is_some_zero() {
        let t = TypeId::new(1);
        let cat = StubCatalogProvider::new()
            .with_rel_type_cardinality(t, 0)
            .with_total_rel_count(0);
        let est = SelectivityEstimator::new(&cat);
        assert_eq!(est.estimate_rel_type(t), 0.0);
    }

    // -----------------------------------------------------------------
    // 5. Bounds + degenerate inputs.
    // -----------------------------------------------------------------

    #[test]
    fn estimate_in_clamps_when_list_exceeds_total() {
        // list_size larger than total can race against a concurrent
        // delete commit; the formula `list_size / total` would exceed
        // 1.0 — the clamp pins the invariant `f ∈ [0, 1]`.
        let cat = StubCatalogProvider::new().with_total_node_count(10);
        let est = SelectivityEstimator::new(&cat);
        let s = est.estimate_in(BindingId::new(0), None, 100);
        assert_eq!(s, 1.0);
    }

    #[test]
    fn estimate_in_returns_zero_for_empty_list() {
        // `n.prop IN ()` is trivially false; selectivity is exactly 0.
        let cat = StubCatalogProvider::new().with_total_node_count(1_000);
        let est = SelectivityEstimator::new(&cat);
        assert_eq!(est.estimate_in(BindingId::new(0), None, 0), 0.0);
    }

    #[test]
    fn estimate_label_falls_back_when_label_card_missing_but_total_present() {
        // Total is collected but THIS label has never been observed
        // (a fresh tenant / a new schema label) — `label_cardinality`
        // returns None per M4-41 §D-25 contract. Falls back to
        // DEFAULT_LABEL_SELECTIVITY rather than `0 / total = 0.0`,
        // which would over-prune the planner's join space.
        let cat = StubCatalogProvider::new().with_total_node_count(1_000);
        let est = SelectivityEstimator::new(&cat);
        assert_eq!(
            est.estimate_label(LabelId::new(1)),
            DEFAULT_LABEL_SELECTIVITY
        );
    }

    #[test]
    fn estimate_rel_type_falls_back_when_type_card_missing_but_total_present() {
        let cat = StubCatalogProvider::new().with_total_rel_count(2_000);
        let est = SelectivityEstimator::new(&cat);
        assert_eq!(
            est.estimate_rel_type(TypeId::new(1)),
            DEFAULT_REL_TYPE_SELECTIVITY
        );
    }

    // -----------------------------------------------------------------
    // 6. clamp_unit dev-mode panic on non-finite input (codex N1).
    //
    // Loud-on-dev, safe-on-prod: a future formula refinement that
    // produces NaN/Inf is a bug, not a graceful-degradation path.
    // The debug_assert fires in dev builds; the is_finite clamp keeps
    // production safe. These tests are cfg-gated to debug builds only
    // because debug_assert is a no-op in release.
    // -----------------------------------------------------------------

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "clamp_unit: input must be finite")]
    fn clamp_unit_debug_asserts_on_nan() {
        let _ = clamp_unit(f64::NAN);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "clamp_unit: input must be finite")]
    fn clamp_unit_debug_asserts_on_inf() {
        let _ = clamp_unit(f64::INFINITY);
    }
}
