//! M4-04d empirical fixture shim — `PersonTenant` cardinality builder.
//!
//! # Why a shim, not a re-export?
//!
//! The full M4-04d back-test fixture lives in
//! `tests/m4_04d_selectivity_backtest.rs` and builds a 1M-row SoA
//! property tenant for the empirical p10/p50/p90/p99 sweep. That
//! fixture's load-bearing artifact for the M4-04d → M4-51 → EXPLAIN
//! transit pin (issue #262 / W9d retro CR-A-1) is its **per-label
//! cardinality shape** — the actual property column data is irrelevant
//! to the cost walker. This module exposes JUST the cardinality shape
//! through a thin `PersonTenant` struct that builds a populated
//! [`StubCatalogProvider`] suitable for EXPLAIN integration tests.
//!
//! Cardinalities mirror the M4-04d `fixture_params` module verbatim
//! (`COMMENT_COUNT`, `FORUM_COUNT`, `PLACE_COUNT`, edge mix). When
//! the M4-04d fixture cardinalities change, this module's constants
//! MUST be updated in lockstep — `tests/m4_04d_selectivity_backtest.rs`
//! is the canonical source.
//!
//! # Phase 4.2 controlled-mutation hook
//!
//! [`PersonTenant::scale_all_label_cards`] / [`PersonTenant::scale_label`]
//! return mutated copies for the Phase 4.2 oracle non-vacuity probe
//! (per `feedback_anchor_to_consumer_transit_pinning.md` reverse-test
//! discipline). The Phase 4.2 cycle in
//! `tests/m4_91_explain_integration.rs::empirical_fixture_phase_4_2_mutation_on_default_label_selectivity`
//! demonstrates that mutating producer-side cardinalities flows
//! through to consumer-side cost output.
//!
//! # ADR provenance
//! - PR #234 — M4-04d empirical `DEFAULT_*_SELECTIVITY` constants;
//!   `tests/m4_04d_selectivity_backtest.rs` builds the 1M Person
//!   fixture cited by ADR-038 amendment-07.
//! - W9d retro Agent A §8.4 / CR-A-1 EMPIRICAL fixture transit (issue
//!   #262) — the gap this shim closes.
//! - `feedback_anchor_to_consumer_transit_pinning.md` — producer-
//!   consumer transit pinning discipline.

use arcgraph_core::LabelId;
use arcgraph_query::semantic::StubCatalogProvider;

// ---------------------------------------------------------------------
// M4-04d auxiliary tenant cardinalities (mirror of `fixture_params`).
// ---------------------------------------------------------------------

/// M4-04d `COMMENT_COUNT` — auxiliary node label.
pub const COMMENT_COUNT: u64 = 100_000;
/// M4-04d `FORUM_COUNT` — auxiliary node label.
pub const FORUM_COUNT: u64 = 10_000;
/// M4-04d `PLACE_COUNT` — auxiliary node label.
pub const PLACE_COUNT: u64 = 1_000;
/// M4-04d `TOTAL_EDGES` — total relationship count across the tenant.
pub const TOTAL_EDGES: u64 = 5_000_000;
/// M4-04d edge-mix fractions: `KNOWS 60%, LIKES 25%, IS_LOCATED_IN 15%`.
pub const KNOWS_FRAC: f64 = 0.60;
pub const LIKES_FRAC: f64 = 0.25;
pub const ILI_FRAC: f64 = 0.15;

/// Default scale factor for [`PersonTenant::seed`] — SF-0.01 (10K
/// Persons). Chosen per the W10b spawn-prompt guidance: SF-1.0 (1M)
/// is the empirical anchor cited by ADR-038 amendment-07, but for an
/// EXPLAIN integration test the absolute scale is irrelevant — the
/// cost walker reads cardinalities, not row data, and the plan-order
/// pin is invariant under uniform scaling. SF-0.01 keeps the test
/// build-time effectively zero (a few HashMap inserts).
pub const DEFAULT_PERSON_COUNT_SF_0_01: u64 = 10_000;

// ---------------------------------------------------------------------
// PersonTenant builder.
// ---------------------------------------------------------------------

/// Cardinality-shape view of the M4-04d Person tenant fixture.
///
/// Mirrors the auxiliary-label cardinalities and edge mix from
/// `tests/m4_04d_selectivity_backtest.rs::fixture_params`. The values
/// are the producer-side outputs PR #234 captured empirically; this
/// struct is the consumer-side shim that flows them into a
/// [`StubCatalogProvider`] for the cost walker.
#[derive(Debug, Clone)]
pub struct PersonTenant {
    /// Person label cardinality. Default = SF-0.01 (10K).
    pub person_count: u64,
    /// Comment label cardinality.
    pub comment_count: u64,
    /// Forum label cardinality.
    pub forum_count: u64,
    /// Place label cardinality.
    pub place_count: u64,
    /// Total relationships across all rel-types.
    pub total_edges: u64,
    /// KNOWS rel-type cardinality (= `total_edges * KNOWS_FRAC`).
    pub knows_count: u64,
    /// LIKES rel-type cardinality (= `total_edges * LIKES_FRAC`).
    pub likes_count: u64,
    /// IS_LOCATED_IN rel-type cardinality (= `total_edges * ILI_FRAC`).
    pub is_located_in_count: u64,
}

impl PersonTenant {
    /// Construct the fixture at SF-0.01 (10K Persons) — the
    /// build-time-trivial default for cargo-test --release.
    pub fn seed() -> Self {
        Self::seed_sf(0.01)
    }

    /// Construct the fixture at an arbitrary scale factor relative to
    /// the canonical SF-1.0 (1M Persons) anchor cited by ADR-038
    /// amendment-07.
    ///
    /// `sf` is multiplied against 1M to derive `person_count`. SF-1.0
    /// = 1_000_000 Persons; SF-0.1 = 100_000; SF-0.01 = 10_000. The
    /// auxiliary labels (Comment, Forum, Place) and total_edges are
    /// SF-INVARIANT per the M4-04d fixture's design (`fixture_params`
    /// module sets these as constants regardless of `person_count`).
    pub fn seed_sf(sf: f64) -> Self {
        let person_count = ((sf * 1_000_000.0).round() as u64).max(1);
        Self {
            person_count,
            comment_count: COMMENT_COUNT,
            forum_count: FORUM_COUNT,
            place_count: PLACE_COUNT,
            total_edges: TOTAL_EDGES,
            knows_count: (TOTAL_EDGES as f64 * KNOWS_FRAC) as u64,
            likes_count: (TOTAL_EDGES as f64 * LIKES_FRAC) as u64,
            is_located_in_count: (TOTAL_EDGES as f64 * ILI_FRAC) as u64,
        }
    }

    /// Total node count across all labels.
    pub fn total_nodes(&self) -> u64 {
        self.person_count + self.comment_count + self.forum_count + self.place_count
    }

    /// Build a [`StubCatalogProvider`] populated with the fixture's
    /// label / rel-type cardinalities + the M4-04d edge-mix totals.
    ///
    /// Label IDs (per [`StubCatalogProvider`]'s monotonic-from-1
    /// convention):
    /// - `Person`  = `LabelId::new(1)`
    /// - `Comment` = `LabelId::new(2)`
    /// - `Forum`   = `LabelId::new(3)`
    /// - `Place`   = `LabelId::new(4)`
    ///
    /// Rel-type IDs:
    /// - `KNOWS`         = `TypeId::new(1)`
    /// - `LIKES`         = `TypeId::new(2)`
    /// - `IS_LOCATED_IN` = `TypeId::new(3)`
    pub fn build_catalog(&self) -> StubCatalogProvider {
        use arcgraph_core::TypeId;
        StubCatalogProvider::new()
            .with_labels(["Person", "Comment", "Forum", "Place"])
            .with_rel_types(["KNOWS", "LIKES", "IS_LOCATED_IN"])
            .with_properties(["name", "city", "id"])
            .with_total_node_count(self.total_nodes())
            .with_total_rel_count(self.total_edges)
            .with_label_cardinality(LabelId::new(1), self.person_count)
            .with_label_cardinality(LabelId::new(2), self.comment_count)
            .with_label_cardinality(LabelId::new(3), self.forum_count)
            .with_label_cardinality(LabelId::new(4), self.place_count)
            .with_rel_type_cardinality(TypeId::new(1), self.knows_count)
            .with_rel_type_cardinality(TypeId::new(2), self.likes_count)
            .with_rel_type_cardinality(TypeId::new(3), self.is_located_in_count)
    }

    /// Phase 4.2 mutation: return a copy with EVERY label cardinality
    /// scaled by `factor`. Mirrors a global "scale
    /// `DEFAULT_LABEL_SELECTIVITY` by `factor`" perturbation in the
    /// fall-through path where the catalog has no per-label cards
    /// (the consumer-observable effect is identical: each leaf's
    /// estimated cardinality multiplies by `factor`).
    ///
    /// Used by the Phase 4.2 oracle-non-vacuity cycle in
    /// `tests/m4_91_explain_integration.rs`.
    pub fn scale_all_label_cards(&self, factor: u64) -> Self {
        Self {
            person_count: self.person_count.saturating_mul(factor),
            comment_count: self.comment_count.saturating_mul(factor),
            forum_count: self.forum_count.saturating_mul(factor),
            place_count: self.place_count.saturating_mul(factor),
            ..self.clone()
        }
    }
}
