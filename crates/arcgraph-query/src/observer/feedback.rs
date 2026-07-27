//! Observed-stats feedback channel to M4-04 catalog stats.
//!
//! Per ADR-038 amendment-03 §"Implicit dependency edges" item 4:
//!
//! > M4-71's `RowCountObserver` emits observed cardinalities back into
//! > the M4-04 catalog stats, closing the feedback loop (observed-stats
//! > → catalog stats → next plan). The feedback channel is per-tenant
//! > per ADR-037; landing it as part of M4-71's acceptance is required.
//!
//! # Design — data structure first, no FeedbackSink trait
//!
//! Per `feedback_avoid_speculative_scaffolding.md`, this slice ships
//! [`ObservedStatsOverrides`] as a CONCRETE STRUCT and the feedback
//! application as a free function. There is no `FeedbackSink` trait —
//! the v1.0 surface has exactly two consumers (test harness + the
//! production-side strict transit pin in `arcgraph-cli/tests`), and
//! both invoke the application function directly without trait dispatch.
//! The `apply_overrides_to_stub_catalog` shim handles the test side; the
//! production-side test calls
//! `arcgraph_storage::CatalogStats::increment_label` etc. directly per
//! the strict transit pin's "real producer surface" mandate.
//!
//! # Per-tenant boundary (per ADR-037 §D-1)
//!
//! The feedback emits one set of overrides PER QUERY EXECUTION. The
//! consumer is responsible for routing each override set into the
//! correct tenant's `CatalogStats` instance — typically by reading the
//! `ExecutionContext::tenant()` field at the call site. The override
//! struct does NOT carry tenant identity (it's "an opaque set of stats
//! deltas the producer just observed") so the same struct can be
//! applied to test fixtures + production catalogs with the same shape.
//!
//! # Apportionment semantics
//!
//! See [`crate::observer::RowCountObserver::observed_overrides`]
//! rustdoc for the apportionment math:
//! - Per-label observed → estimated-card-weighted apportionment from
//!   per-Scan plan-walk entries.
//! - Per-rel-type observed → same pattern from per-Expand entries.
//! - Total nodes / rels → reported only when no label / rel-type
//!   filtering is in effect (single sweep over all rows).
//!
//! # ADR provenance
//! - ADR-038 amendment-03 §"Implicit dependency edges" item 4.
//! - ADR-037 §D-1 — per-tenant substrate composition (the boundary the
//!   feedback honors).
//! - `feedback_avoid_speculative_scaffolding.md` — ratifies the
//!   no-trait discipline at this slice.

use std::collections::HashMap;

use arcgraph_core::{LabelId, TypeId};

use crate::semantic::{CatalogProvider, StubCatalogProvider};

/// Per-tenant observed-stats overrides extracted from a single query
/// execution.
///
/// Constructed by [`crate::observer::RowCountObserver::observed_overrides`];
/// applied by the consumer to a target `CatalogStats` instance (or
/// `StubCatalogProvider` for tests via [`apply_overrides_to_stub_catalog`]).
///
/// All fields are additive — applying overrides INCREMENTS the target's
/// counters, never resets. The consumer is responsible for monotonic
/// invariants (e.g., `commits_observed` advances via the
/// `begin_commit_observation` / `observe_commit` bracket on the
/// production side).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ObservedStatsOverrides {
    /// Per-label observed-row count. Zero values are omitted (the
    /// observer only inserts when at least one row was observed for a
    /// label).
    pub label_observed: HashMap<LabelId, u64>,
    /// Per-rel-type observed-row count.
    pub rel_type_observed: HashMap<TypeId, u64>,
    /// Tenant-wide total node count if the plan executed a label-free
    /// scan; `None` for label-filtered scans (where the observed total
    /// is a partial view).
    pub total_nodes_observed: Option<u64>,
    /// Tenant-wide total rel count if the plan executed a type-free
    /// expand; `None` for type-filtered expands.
    pub total_rels_observed: Option<u64>,
}

impl ObservedStatsOverrides {
    /// `true` if the override set contains no actionable feedback
    /// (no labels, no rel-types, no totals).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.label_observed.is_empty()
            && self.rel_type_observed.is_empty()
            && self.total_nodes_observed.is_none()
            && self.total_rels_observed.is_none()
    }

    /// Sum of per-label observed counts. Convenience for tests.
    #[must_use]
    pub fn label_observed_total(&self) -> u64 {
        self.label_observed.values().copied().sum()
    }
}

/// Apply observed overrides to a [`StubCatalogProvider`] used in tests.
///
/// Returns a new stub with the cardinalities updated and the
/// `commits_observed_count` bumped by 1 (mirroring the storage-side
/// `observe_commit` convention).
///
/// The returned stub is a brand-new clone with overrides applied —
/// `StubCatalogProvider` is immutable after construction (per the M4-21
/// fluent-builder pattern), so the consumer rebuilds via the fluent
/// setters.
///
/// # Apportionment-vs-replacement semantics
///
/// `apply_overrides_to_stub_catalog` REPLACES the per-label / per-rel-
/// type counts with the observed values (rather than ADDING them) — the
/// observer's overrides represent the OBSERVED cardinality at the
/// current query, which IS the new ground truth from the executor's
/// vantage point. The production-side `CatalogStats::increment_label`
/// path is additive (per-commit hook), so on the production side the
/// caller computes the delta to pass to `increment_label`.
#[must_use]
pub fn apply_overrides_to_stub_catalog(
    base: &StubCatalogProvider,
    overrides: &ObservedStatsOverrides,
) -> StubCatalogProvider {
    let mut next = base.clone();
    for (label, count) in &overrides.label_observed {
        next = next.with_label_cardinality(*label, *count);
    }
    for (rel_type, count) in &overrides.rel_type_observed {
        next = next.with_rel_type_cardinality(*rel_type, *count);
    }
    if let Some(total) = overrides.total_nodes_observed {
        next = next.with_total_node_count(total);
    }
    if let Some(total) = overrides.total_rels_observed {
        next = next.with_total_rel_count(total);
    }
    // Bump the commit counter by 1 — mirrors the storage-side
    // observe_commit advancement that triggers M4-53 plan-cache
    // invalidation on the next lookup.
    let prev_commits = base.snapshot().commits_observed();
    next = next.with_commits_observed_count(prev_commits + 1);
    next
}

#[cfg(test)]
mod tests {
    use super::*;
    use arcgraph_core::{LabelId, TypeId};

    #[test]
    fn empty_overrides_is_empty() {
        let o = ObservedStatsOverrides::default();
        assert!(o.is_empty());
        assert_eq!(o.label_observed_total(), 0);
    }

    #[test]
    fn label_observed_total_sums_all_labels() {
        let mut o = ObservedStatsOverrides::default();
        o.label_observed.insert(LabelId::new(1), 100);
        o.label_observed.insert(LabelId::new(2), 200);
        assert_eq!(o.label_observed_total(), 300);
        assert!(!o.is_empty());
    }

    #[test]
    fn apply_overrides_replaces_label_cardinalities_and_bumps_commit_counter() {
        let base = StubCatalogProvider::new()
            .with_label_cardinality(LabelId::new(1), 50)
            .with_commits_observed_count(7);
        let mut overrides = ObservedStatsOverrides::default();
        overrides.label_observed.insert(LabelId::new(1), 1000);
        overrides.label_observed.insert(LabelId::new(2), 250);
        let next = apply_overrides_to_stub_catalog(&base, &overrides);
        let snap = next.snapshot();
        // Label 1 was REPLACED (not added) — observed cardinality wins.
        assert_eq!(snap.label_card(LabelId::new(1)), Some(1000));
        // Label 2 newly minted.
        assert_eq!(snap.label_card(LabelId::new(2)), Some(250));
        // Commit counter advanced by 1.
        assert_eq!(snap.commits_observed(), 8);
    }

    #[test]
    fn apply_overrides_propagates_rel_type_observed() {
        let base = StubCatalogProvider::new();
        let mut overrides = ObservedStatsOverrides::default();
        overrides.rel_type_observed.insert(TypeId::new(1), 42);
        let next = apply_overrides_to_stub_catalog(&base, &overrides);
        let snap = next.snapshot();
        assert_eq!(snap.rel_type_card(TypeId::new(1)), Some(42));
    }

    #[test]
    fn apply_overrides_carries_totals_when_present() {
        let base = StubCatalogProvider::new();
        let overrides = ObservedStatsOverrides {
            total_nodes_observed: Some(10_000),
            total_rels_observed: Some(50_000),
            ..Default::default()
        };
        let next = apply_overrides_to_stub_catalog(&base, &overrides);
        let snap = next.snapshot();
        assert_eq!(snap.total_nodes(), Some(10_000));
        assert_eq!(snap.total_rels(), Some(50_000));
    }
}
