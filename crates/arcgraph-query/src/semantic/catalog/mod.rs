//! Schema catalog adapter for the binding pass.
//!
//! `CatalogProvider` is consumer-defined HERE in `arcgraph-query`
//! (NOT in `arcgraph-storage`). Three reasons:
//!
//! 1. **Cyclic-dependency avoidance.** Storage may consume query
//!    types in v1.1+ (e.g., a `BoundAst` cached alongside MVCC
//!    versions); declaring the trait in storage would invert the
//!    `query → storage` edge of `docs/bounded-contexts.md`.
//! 2. **Test ergonomics.** `StubCatalogProvider` lives next to the
//!    trait — tests stub the catalog without pulling in
//!    `arcgraph-storage`'s buffer-pool / WAL machinery.
//! 3. **Bounded-context discipline.** `arcgraph-query` depends only
//!    on `arcgraph-core` for type primitives. The production
//!    catalog (lit at the executor wiring layer in M4-08+) lives
//!    in `arcgraph-storage`; that crate provides a `CatalogProvider`
//!    impl on its tenant-catalog type at composition time.
//!
//! # M4-23 substrate-availability extension
//!
//! M4-23 adds three additive predicates — [`CatalogProvider::has_vector_index`],
//! [`CatalogProvider::has_bm25_index`],
//! [`CatalogProvider::has_community_index`] — consumed by the
//! [`crate::semantic::cross_substrate::CrossSubstrateValidator`]. The
//! flags are per-tenant (the catalog already carries a `tenant()`
//! identity) and answer "does this tenant have an attached substrate?"
//! at bind-time.
//!
//! - **Vector** — per ADR-035 D-7, vector-index presence is per-tenant
//!   (`PartitionId::ZERO`).
//! - **BM25** — per ADR-039 D-4, the per-tenant Tantivy directory.
//! - **Community** — per ADR-040 D-3, the per-tenant
//!   `(TenantId, Level, NodeId)`-keyed index.
//!
//! # M4-41 catalog stats extension
//!
//! M4-41 adds four additive cardinality methods —
//! [`CatalogProvider::label_cardinality`],
//! [`CatalogProvider::rel_type_cardinality`],
//! [`CatalogProvider::total_node_count`],
//! [`CatalogProvider::total_rel_count`] — consumed by the future
//! M4-51 cost-based planner for join-ordering selectivity. Each
//! returns `Option<u64>`: `Some(n)` once stats have been collected,
//! `None` for fresh tenants whose commit pipeline has not yet fired.
//! v1.0 ships exact counts; HyperLogLog and property-value sketches
//! are deferred to v1.1 per amendment-03 M4-04c.
//!
//! Default impls return `None` so older `CatalogProvider` impls (e.g.
//! tests that pre-date M4-41) compile unchanged. Production catalogs
//! override; [`StubCatalogProvider`] exposes a fluent
//! `.with_label_cardinality(...)` / `.with_rel_type_cardinality(...)`
//! / `.with_total_node_count(...)` / `.with_total_rel_count(...)`
//! builder.
//!
//! # ADR provenance
//! - ADR-038 §2 D-1 — openCypher binding semantics baseline.
//! - ADR-038 §2 D-23 — cross-substrate validation contract (M4-23).
//! - ADR-038 §2 D-25 — catalog stats schema + collection on commit
//!   contract (M4-41).
//! - ADR-038 amendment-03 §TIER-1 GAP E — snapshot-LSN field on
//!   `BoundQuery` populated at execute-time (M4-61); the catalog
//!   layer carries the `(tenant, partition)` identity stamped on
//!   `BoundQuery` at bind-time.
//! - ADR-038 amendment-03 §"Implicit dependency edges" item 4 — M4-71
//!   `RowCountObserver` feeds observed cardinalities back into the
//!   M4-04 catalog stats; M4-41's stats schema is the receiver of
//!   that future feedback loop.
//! - ADR-035 D-7 / ADR-039 D-4 / ADR-040 D-3 — per-tenant substrate
//!   keying (the source of truth for `has_*_index` semantics).

use arcgraph_core::{LabelId, PartitionId, PropertyId, TenantId, TypeId};

pub mod snapshot;
pub mod stub;

pub use snapshot::{CatalogSnapshot, MaxOutDegreeEntry};
pub use stub::StubCatalogProvider;

/// Schema catalog adapter consumed by the binding pass.
///
/// Implementations live outside `arcgraph-query`:
/// - **Production** — `arcgraph-storage`'s tenant catalog implements
///   this trait at executor-wiring time (M4-08+).
/// - **Tests** — [`StubCatalogProvider`] is a fluent in-memory
///   builder with deterministic ID assignment.
///
/// # Resolution semantics
///
/// - [`Self::lookup_label`] and [`Self::lookup_rel_type`] return
///   `None` for unknown names. The binding pass converts these to
///   [`crate::semantic::error::BindingError::UnknownLabel`] /
///   [`crate::semantic::error::BindingError::UnknownRelType`].
/// - [`Self::lookup_property`] returns `None` only when the catalog
///   enforces a strict schema (v1.1+). The v1.0 dynamic-schema
///   convention is to always return `Some(PropertyId)`, allocating
///   a fresh interned ID on first sight. The binding pass does NOT
///   emit an error when `lookup_property` returns `None`; M4-22 may
///   add stricter handling at the type-check layer.
/// - [`Self::tenant`] and [`Self::partition`] are stamped onto the
///   bound query for downstream MVCC + partition-aware planning.
///   v1.0 invariant: `partition()` returns [`PartitionId::ZERO`]
///   per ADR-024 amendment-02 local-only architecture.
///
/// # Naming note
///
/// The M4-21 brief calls the relationship-type ID "RelTypeId". The
/// canonical codebase name (`arcgraph_core::TypeId`) wins over the
/// brief language for codebase consistency. See
/// [`Self::lookup_rel_type`].
pub trait CatalogProvider: Send + Sync {
    /// Resolve a node label name to its interned [`LabelId`].
    /// Returns `None` for unknown names.
    fn lookup_label(&self, name: &str) -> Option<LabelId>;

    /// Resolve a relationship-type name to its interned [`TypeId`].
    ///
    /// `arcgraph-core`'s canonical type for relationship-type IDs is
    /// [`TypeId`]; the M4-21 brief calls it "RelTypeId" but we use
    /// the codebase-consistent name. Returns `None` for unknown
    /// names.
    fn lookup_rel_type(&self, name: &str) -> Option<TypeId>;

    /// Resolve a property-key name to its interned [`PropertyId`].
    /// Returns `None` only when the catalog enforces a strict
    /// schema (v1.1+); the v1.0 convention is dynamic-schema
    /// fallback (always `Some`).
    fn lookup_property(&self, name: &str) -> Option<PropertyId>;

    /// The tenant this query is scoped to. Stamped onto `BoundQuery`
    /// for MVCC + per-tenant plan-cache routing.
    fn tenant(&self) -> TenantId;

    /// The partition this query is scoped to. v1.0 invariant:
    /// always [`PartitionId::ZERO`] per ADR-024 amendment-02.
    fn partition(&self) -> PartitionId;

    /// Returns `true` when this tenant has a vector index attached.
    ///
    /// Consumed by M4-23's
    /// [`crate::semantic::cross_substrate::CrossSubstrateValidator`]
    /// when validating ArcQL clauses that require the vector substrate
    /// (`RANK BY HYBRID(VECTOR(...), …)`, the `vector_distance(...)`
    /// function, the `<expr> NEAR <expr>` predicate). v1.0 keying:
    /// per-tenant (`PartitionId::ZERO`) per ADR-035 D-7.
    ///
    /// Default impl returns `false` to keep the trait additive — old
    /// `CatalogProvider` impls compile unchanged. New impls SHOULD
    /// override.
    fn has_vector_index(&self) -> bool {
        false
    }

    /// Returns `true` when this tenant has a BM25 (text) index
    /// attached.
    ///
    /// Consumed by M4-23's
    /// [`crate::semantic::cross_substrate::CrossSubstrateValidator`]
    /// when validating ArcQL clauses that require the BM25 substrate
    /// (`RANK BY HYBRID(…, TEXT(...))`, the `text_match(...)`
    /// function, the `<expr> MATCH <expr>` predicate). v1.0 keying:
    /// per-tenant Tantivy directory per ADR-039 D-4.
    fn has_bm25_index(&self) -> bool {
        false
    }

    /// Returns `true` when this tenant has a community-detection
    /// index attached.
    ///
    /// Consumed by M4-23's
    /// [`crate::semantic::cross_substrate::CrossSubstrateValidator`]
    /// when validating ArcQL clauses that require the community
    /// substrate (`n IN COMMUNITY($cid)` predicate, the
    /// `community(...)` function family). v1.0 keying:
    /// `(TenantId, Level, NodeId)` per ADR-040 D-3.
    fn has_community_index(&self) -> bool {
        false
    }

    /// **#1366 (Phase 2) — the RC-6 planner-visible index gate.**
    /// Whether the tenant has an **Online** secondary property index on
    /// `(label, property)` that the planner may route a point lookup to.
    ///
    /// Returns `true` ONLY when a declared index on exactly this
    /// `(label, property)` pair is in the `Online` state
    /// (`IndexState::planner_visible()`). A `Building` index — whose
    /// backfill tail is incomplete — MUST return `false` here: routing a
    /// query to a Building index risks a FALSE NEGATIVE (a node written
    /// after the backfill snapshot but not yet covered would be missed).
    /// This is THE gate the whole Phase-2 correctness story rests on:
    /// the identical-results test fails the moment a Building index is
    /// used (see the Building-not-used RED-on-revert test).
    ///
    /// # Default impl
    ///
    /// Returns `false` — a `CatalogProvider` with no property-index
    /// catalog (test fixtures, pre-#1366 impls) reports "no index", so
    /// the planner keeps the full-scan path. The production catalog
    /// adapter overrides this to consult the durable
    /// `PropertyIndexCatalog` state.
    fn online_property_index(&self, _label: LabelId, _property: &str) -> bool {
        false
    }

    /// Returns the cardinality of nodes carrying `label`, or `None`
    /// when stats have not yet been collected for this label.
    ///
    /// Consumed by the future M4-51 cost-based planner for
    /// join-ordering selectivity. v1.0 ships exact counts; HyperLogLog
    /// approximations and bucketed sketches are deferred to v1.1 per
    /// ADR-038 amendment-03 M4-04c.
    ///
    /// `None` is the "no-stats" sentinel: a fresh tenant whose commit
    /// pipeline has never fired returns `None` for every label, and
    /// the cost planner falls back to its default selectivity. Once
    /// the first commit lands, the catalog reports `Some(count)` for
    /// every observed label; labels that have never been observed
    /// continue to return `None` (NOT `Some(0)`).
    ///
    /// Default impl returns `None` to keep the trait additive — older
    /// `CatalogProvider` impls (pre-M4-41) compile unchanged. New impls
    /// SHOULD override (production catalogs) or expose a fluent setter
    /// (test catalogs).
    fn label_cardinality(&self, _label: LabelId) -> Option<u64> {
        None
    }

    /// Returns the cardinality of relationships of `rel_type`, or
    /// `None` when stats have not yet been collected.
    ///
    /// Same `None`-as-no-stats convention as
    /// [`Self::label_cardinality`]; same default-None additivity
    /// posture; same M4-51 consumer.
    fn rel_type_cardinality(&self, _rel_type: TypeId) -> Option<u64> {
        None
    }

    /// Returns the total node count for this tenant (regardless of
    /// label), or `None` when stats have not been collected.
    ///
    /// Distinct from summing [`Self::label_cardinality`] across all
    /// labels: the totals are O(1) atomic reads, whereas a per-label
    /// sum is O(label-count) and races with concurrent commits. M4-51
    /// uses this for tenant-wide selectivity heuristics.
    fn total_node_count(&self) -> Option<u64> {
        None
    }

    /// Returns the total relationship count for this tenant
    /// (regardless of type), or `None` when stats have not been
    /// collected. Symmetric with [`Self::total_node_count`].
    fn total_rel_count(&self) -> Option<u64> {
        None
    }

    /// Capture a plan-time [`CatalogSnapshot`] for the M4-51 cost
    /// planner.
    ///
    /// The snapshot is the cross-key-consistent point-in-time view
    /// of `(total_nodes, total_rels, label_cards, rel_type_cards)`
    /// the cost planner needs. The M4-51 walker calls this ONCE at
    /// plan-start and reads from the resulting snapshot through every
    /// per-operator cost function — preserving cross-key consistency
    /// across the full plan walk per ADR-038 amendment-03 §M4-04e
    /// (issue #210) + PR #220.
    ///
    /// # Default impl
    ///
    /// Returns [`CatalogSnapshot::empty`]. Old `CatalogProvider` impls
    /// (pre-M4-51) compile unchanged; production catalogs override to
    /// delegate to `arcgraph_storage::catalog::CatalogStats::snapshot()`
    /// (translating the storage-side struct to this query-side mirror);
    /// [`StubCatalogProvider`] overrides to assemble from its fluent
    /// builder maps.
    ///
    /// # Cross-key consistency
    ///
    /// The default impl returns an EMPTY snapshot — it does NOT attempt
    /// to assemble a snapshot from the per-counter accessors
    /// ([`Self::total_node_count`], etc.) because those accessors do
    /// NOT preserve cross-key consistency (they use `Relaxed` ordering
    /// per `arcgraph_storage::catalog::stats` module docs). A
    /// best-effort assembly via per-counter reads would silently
    /// violate the snapshot contract; an empty snapshot lets the cost
    /// planner fall through cleanly to `DEFAULT_*_SELECTIVITY`
    /// constants.
    fn snapshot(&self) -> CatalogSnapshot {
        CatalogSnapshot::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal `CatalogProvider` impl that overrides only the
    /// pre-M4-41 required methods. Confirms the M4-41 stats methods
    /// inherit their default-`None` impls without forcing every
    /// older catalog impl to re-implement.
    struct PreM4_41Provider;

    impl CatalogProvider for PreM4_41Provider {
        fn lookup_label(&self, _name: &str) -> Option<LabelId> {
            None
        }
        fn lookup_rel_type(&self, _name: &str) -> Option<TypeId> {
            None
        }
        fn lookup_property(&self, _name: &str) -> Option<PropertyId> {
            None
        }
        fn tenant(&self) -> TenantId {
            TenantId::DEFAULT
        }
        fn partition(&self) -> PartitionId {
            PartitionId::ZERO
        }
    }

    #[test]
    fn catalog_provider_default_stats_methods_return_none() {
        // Backwards-compat pin: an impl predating M4-41 (no stats
        // overrides) still compiles, and every stats method returns
        // None — the documented "no stats collected" sentinel that
        // M4-51's cost planner translates to default selectivity.
        let p = PreM4_41Provider;
        assert_eq!(p.label_cardinality(LabelId::new(1)), None);
        assert_eq!(p.rel_type_cardinality(TypeId::new(1)), None);
        assert_eq!(p.total_node_count(), None);
        assert_eq!(p.total_rel_count(), None);
    }

    #[test]
    fn catalog_provider_default_snapshot_is_empty() {
        // M4-51 backwards-compat pin: an impl predating M4-51 (no
        // `snapshot()` override) still compiles, and the default impl
        // returns an empty snapshot. The cost planner translates this
        // to "no stats collected" → DEFAULT_*_SELECTIVITY fallbacks.
        let p = PreM4_41Provider;
        let snap = p.snapshot();
        assert_eq!(snap.total_nodes(), None);
        assert_eq!(snap.total_rels(), None);
        assert!(snap.label_cards().is_empty());
        assert!(snap.rel_type_cards().is_empty());
        assert!(!snap.has_observed_any());
    }
}
