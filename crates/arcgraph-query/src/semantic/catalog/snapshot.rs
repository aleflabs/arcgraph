//! Plan-time catalog snapshot consumed by the M4-51 cost planner.
//!
//! `CatalogSnapshot` is the consumer-side mirror of the storage-side
//! [`arcgraph_storage::catalog::CatalogSnapshot`] (per PR #220 / M4-04e
//! / issue #210). It carries the cross-key-consistent point-in-time
//! view of `(total_nodes, total_rels, label_cards, rel_type_cards)`
//! the cost planner needs.
//!
//! # Why a query-side mirror?
//!
//! Bounded contexts (`docs/bounded-contexts.md` §"`arcgraph-query`")
//! permit `arcgraph-query → arcgraph-storage` as a sibling dep, but
//! the established pattern for plan-time consumer types — established
//! by [`crate::semantic::catalog::CatalogProvider`] — is to declare
//! them HERE and let production storage impls translate at composition
//! time. That keeps `arcgraph-query`'s compile graph free of storage
//! internals (BufferPool, WAL, CRUD machinery) at v1.0; the dep gets
//! lit when M4-08+ executor wiring genuinely needs CRUD / scan APIs.
//!
//! See `docs/bounded-contexts.md` lines 79–96 + the
//! `CatalogProvider` rustdoc for the consumer-defined-trait pattern
//! this type extends.
//!
//! # Cross-key consistency
//!
//! Every snapshot satisfies the cross-key invariants
//!
//! - `sum(label_cards) ≤ total_nodes` (when `total_nodes` is `Some`),
//! - `sum(rel_type_cards) ≤ total_rels` (when `total_rels` is `Some`),
//!
//! provided the producer (`CatalogProvider::snapshot()`) honors the
//! contract. The storage-side production impl achieves this through
//! the two-marker SeqLock protocol documented in
//! `arcgraph_storage::catalog::stats` module docs §"Cross-key snapshot
//! mechanism"; test impls (`StubCatalogProvider`) build the snapshot
//! deterministically from a static fluent builder.
//!
//! # `None` vs `Some(0)` semantics
//!
//! Mirrors the per-counter accessors on
//! [`crate::semantic::catalog::CatalogProvider`]:
//!
//! - `total_nodes() == None` → no commit observed; cost planner uses
//!   [`crate::semantic::selectivity::DEFAULT_LABEL_SELECTIVITY`] /
//!   friends.
//! - `total_nodes() == Some(0)` → commits observed; current count is
//!   zero. Cost planner returns `0.0` selectivity.
//! - `label_card(label) == None` → label never observed by the commit
//!   pipeline. Cost planner falls back to default-label selectivity.
//! - `label_card(label) == Some(0)` → label observed-then-fully-deleted.
//!
//! # ADR provenance
//! - ADR-038 §2 D-25 — catalog stats schema (M4-41 producer).
//! - ADR-038 amendment-03 §M4-04e (issue #210) — cross-key snapshot
//!   contract; PR #220 implements the storage-side substrate.
//! - ADR-038 amendment-02 §M4.e — M4-51 cost planner consumer.

use arcgraph_core::{LabelId, NodeId, TypeId};

/// Planner-side mirror of ADR-025 §5 max out-degree sketch entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaxOutDegreeEntry {
    pub label: LabelId,
    pub rel_type: TypeId,
    pub vertex: NodeId,
    pub degree: u64,
}

/// Plan-time snapshot of catalog stats.
///
/// Captured by [`crate::semantic::catalog::CatalogProvider::snapshot`]
/// and consumed by the M4-51 cost planner. `Clone + Send + Sync`; the
/// cost planner takes ONE snapshot per plan and reads from it across
/// every cost-function call without re-paying the snapshot cost or
/// interleaving with a concurrent commit.
///
/// # Construction
///
/// Production storage catalogs construct via [`Self::from_parts`] after
/// translating from `arcgraph_storage::catalog::CatalogSnapshot`. Tests
/// (`StubCatalogProvider`) build via [`Self::from_parts`] directly from
/// fluent-builder maps. The empty / fresh-tenant snapshot is
/// [`Self::empty`].
#[derive(Debug, Clone, Default)]
pub struct CatalogSnapshot {
    total_nodes: Option<u64>,
    total_rels: Option<u64>,
    /// Sorted by [`LabelId::raw()`] for binary-search lookup.
    /// Producers MUST sort before constructing.
    label_cards: Vec<(LabelId, u64)>,
    /// Sorted by [`TypeId::raw()`] for binary-search lookup.
    rel_type_cards: Vec<(TypeId, u64)>,
    /// Tenant-wide commit-counter at snapshot capture (mirror of
    /// `arcgraph_storage::catalog::CatalogSnapshot::commits_observed`).
    /// Defaults to `0` for empty / fresh-tenant snapshots, matching
    /// the storage-side fresh-tenant value. Consumed by the M4-53
    /// plan cache as the stats-change watermark per ADR-038
    /// amendment-03 §TIER-2-a; older `CatalogProvider` impls that
    /// don't override [`crate::semantic::CatalogProvider::snapshot`]
    /// (default-impl returns [`Self::empty`]) report `0` and the cache
    /// effectively never invalidates from stats drift on those impls,
    /// which is the correct behavior — they have no stats to drift.
    commits_observed: u64,
    /// ADR-025 §5 `max_out_degree_sketch[label, rel_type]` entries.
    max_out_degree: Vec<MaxOutDegreeEntry>,
}

impl CatalogSnapshot {
    /// Construct a snapshot from its constituent parts.
    ///
    /// `label_cards` and `rel_type_cards` MUST be pre-sorted by raw
    /// id ascending; the constructor sorts defensively to enforce the
    /// invariant that [`Self::label_card`] / [`Self::rel_type_card`]
    /// rely on for `O(log n)` lookup.
    ///
    /// `commits_observed` is the stats-change watermark consumed by
    /// the M4-53 plan cache (per ADR-038 amendment-03 §TIER-2-a). Pass
    /// `0` for fresh-tenant / no-stats snapshots; production storage
    /// catalogs translate from
    /// `arcgraph_storage::catalog::CatalogSnapshot::commits_observed()`.
    #[must_use]
    pub fn from_parts(
        total_nodes: Option<u64>,
        total_rels: Option<u64>,
        mut label_cards: Vec<(LabelId, u64)>,
        mut rel_type_cards: Vec<(TypeId, u64)>,
        commits_observed: u64,
    ) -> Self {
        label_cards.sort_unstable_by_key(|(l, _)| l.raw());
        rel_type_cards.sort_unstable_by_key(|(t, _)| t.raw());
        Self {
            total_nodes,
            total_rels,
            label_cards,
            rel_type_cards,
            commits_observed,
            max_out_degree: Vec::new(),
        }
    }

    /// Construct a snapshot including ADR-025 §5 max out-degree
    /// sketch entries.
    #[must_use]
    pub fn from_parts_with_max_out_degree(
        total_nodes: Option<u64>,
        total_rels: Option<u64>,
        label_cards: Vec<(LabelId, u64)>,
        rel_type_cards: Vec<(TypeId, u64)>,
        commits_observed: u64,
        mut max_out_degree: Vec<MaxOutDegreeEntry>,
    ) -> Self {
        let mut snapshot = Self::from_parts(
            total_nodes,
            total_rels,
            label_cards,
            rel_type_cards,
            commits_observed,
        );
        max_out_degree.sort_unstable_by(|a, b| {
            a.label
                .raw()
                .cmp(&b.label.raw())
                .then_with(|| a.rel_type.raw().cmp(&b.rel_type.raw()))
                .then_with(|| b.degree.cmp(&a.degree))
                .then_with(|| a.vertex.raw().cmp(&b.vertex.raw()))
        });
        snapshot.max_out_degree = max_out_degree;
        snapshot
    }

    /// Construct an empty / fresh-tenant snapshot. Every accessor
    /// returns the "no stats" sentinel; the cost planner falls back
    /// to `DEFAULT_*_SELECTIVITY` constants per
    /// [`crate::semantic::selectivity`].
    #[must_use]
    pub fn empty() -> Self {
        Self::default()
    }

    /// Tenant-wide total node count. `None` until the first commit
    /// has been observed; `Some(_)` thereafter.
    #[must_use]
    pub fn total_nodes(&self) -> Option<u64> {
        self.total_nodes
    }

    /// Tenant-wide total relationship count. Mirrors
    /// [`Self::total_nodes`].
    #[must_use]
    pub fn total_rels(&self) -> Option<u64> {
        self.total_rels
    }

    /// Per-label cardinality at snapshot capture. `None` if the
    /// label has never been observed (the "never observed"
    /// sentinel — distinct from `Some(0)` =
    /// "observed-then-fully-deleted"). `O(log n)` binary search.
    #[must_use]
    pub fn label_card(&self, label: LabelId) -> Option<u64> {
        self.label_cards
            .binary_search_by_key(&label.raw(), |(l, _)| l.raw())
            .ok()
            .map(|idx| self.label_cards[idx].1)
    }

    /// Per-rel-type cardinality at snapshot capture. Mirrors
    /// [`Self::label_card`] semantics.
    #[must_use]
    pub fn rel_type_card(&self, rel_type: TypeId) -> Option<u64> {
        self.rel_type_cards
            .binary_search_by_key(&rel_type.raw(), |(t, _)| t.raw())
            .ok()
            .map(|idx| self.rel_type_cards[idx].1)
    }

    /// All per-label cardinalities, sorted by [`LabelId::raw()`].
    #[must_use]
    pub fn label_cards(&self) -> &[(LabelId, u64)] {
        &self.label_cards
    }

    /// All per-rel-type cardinalities, sorted by [`TypeId::raw()`].
    #[must_use]
    pub fn rel_type_cards(&self) -> &[(TypeId, u64)] {
        &self.rel_type_cards
    }

    /// ADR-025 §5 `max_out_degree_sketch[label, rel_type]` entries.
    #[must_use]
    pub fn max_out_degree_entries(&self) -> &[MaxOutDegreeEntry] {
        &self.max_out_degree
    }

    /// `true` iff at least one commit has been observed (i.e.,
    /// `total_nodes` is `Some`).
    #[must_use]
    pub fn has_observed_any(&self) -> bool {
        self.total_nodes.is_some() || self.total_rels.is_some()
    }

    /// Number of commits observed at snapshot capture.
    ///
    /// Mirror of `arcgraph_storage::catalog::CatalogSnapshot::commits_observed`
    /// per ADR-038 amendment-03 §M4-04e + amendment-03 §TIER-2-a. Consumed
    /// by the M4-53 plan cache as the stats-change watermark — every cache
    /// entry stamps this value at insert and on lookup compares the stamped
    /// value to the live catalog's current `commits_observed`. If the live
    /// value advanced, the entry is invalidated and re-planned (lazy
    /// invalidation; cheap on hit).
    ///
    /// `0` is the fresh-tenant sentinel (no commits observed yet); a cache
    /// entry stamped at `0` survives until the first commit lands. This
    /// matches storage-side semantics post-recovery (per `stats.rs:769`
    /// rebuild semantics) where the cold-start rebuild bracket counts as
    /// `1` commit observation regardless of pre-crash count.
    #[must_use]
    pub fn commits_observed(&self) -> u64 {
        self.commits_observed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_snapshot_returns_none_everywhere() {
        let snap = CatalogSnapshot::empty();
        assert_eq!(snap.total_nodes(), None);
        assert_eq!(snap.total_rels(), None);
        assert_eq!(snap.label_card(LabelId::new(1)), None);
        assert_eq!(snap.rel_type_card(TypeId::new(1)), None);
        assert!(snap.label_cards().is_empty());
        assert!(snap.rel_type_cards().is_empty());
        assert!(!snap.has_observed_any());
    }

    #[test]
    fn from_parts_sorts_label_and_rel_type_cards() {
        // Inputs deliberately unsorted; constructor must sort by raw id
        // so binary-search lookup is correct.
        let snap = CatalogSnapshot::from_parts(
            Some(1_000),
            Some(2_000),
            vec![
                (LabelId::new(7), 70),
                (LabelId::new(2), 20),
                (LabelId::new(11), 110),
            ],
            vec![(TypeId::new(5), 50), (TypeId::new(1), 10)],
            0,
        );
        let label_raws: Vec<u32> = snap.label_cards().iter().map(|(l, _)| l.raw()).collect();
        assert_eq!(label_raws, vec![2, 7, 11]);
        let type_raws: Vec<u32> = snap.rel_type_cards().iter().map(|(t, _)| t.raw()).collect();
        assert_eq!(type_raws, vec![1, 5]);

        assert_eq!(snap.label_card(LabelId::new(7)), Some(70));
        assert_eq!(snap.label_card(LabelId::new(2)), Some(20));
        assert_eq!(snap.label_card(LabelId::new(99)), None);
        assert_eq!(snap.rel_type_card(TypeId::new(5)), Some(50));
        assert_eq!(snap.rel_type_card(TypeId::new(99)), None);
    }

    #[test]
    fn snapshot_distinguishes_none_from_some_zero() {
        // Round-trip the None / Some(0) sentinel distinction so the
        // cost planner can branch on "never observed" vs.
        // "observed-then-deleted". Mirror of
        // arcgraph_storage::catalog::stats::CatalogSnapshot's contract.
        let snap =
            CatalogSnapshot::from_parts(Some(0), None, vec![(LabelId::new(1), 0)], Vec::new(), 0);
        assert_eq!(snap.total_nodes(), Some(0));
        assert_eq!(snap.total_rels(), None);
        assert_eq!(snap.label_card(LabelId::new(1)), Some(0));
        assert_eq!(snap.label_card(LabelId::new(2)), None);
        assert!(snap.has_observed_any());
    }

    #[test]
    fn commits_observed_round_trips_through_from_parts() {
        // M4-53 stats-change watermark: round-trip the commit-counter
        // value so the M4-53 plan cache can stamp entries with it on
        // insert and compare on lookup.
        let snap = CatalogSnapshot::from_parts(Some(100), None, Vec::new(), Vec::new(), 7);
        assert_eq!(snap.commits_observed(), 7);
        // Empty / fresh-tenant snapshot reports 0 (the storage-side
        // fresh-tenant sentinel).
        assert_eq!(CatalogSnapshot::empty().commits_observed(), 0);
    }
}
