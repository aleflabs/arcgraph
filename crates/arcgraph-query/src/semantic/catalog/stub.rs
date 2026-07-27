//! In-memory `CatalogProvider` impl — test fixture + v1.0-α
//! production seed.
//!
//! `StubCatalogProvider` is a fluent builder that pre-populates a
//! known set of labels / rel-types / properties + carries a tenant
//! and partition stamp. The integration test in
//! `tests/binding_integration.rs` and the unit tests in
//! `binding.rs::tests` use it to exercise the binding pass without
//! pulling in `arcgraph-storage`'s tenant catalog.
//!
//! # Production usage at v1.0-α (W17α M4-08+)
//!
//! Per R1 review MED-2 (PR #349) the W17α MCP / Bolt raw-query
//! adapters (`StorageRawQueryExecutor`, `StorageBoltHandler`) seed
//! one of these per call from the catalog-stats snapshot — label
//! names and rel-type names are enumerated, property names are NOT
//! (the intern table has no name-side enumeration API). The v1.0-α
//! production query path therefore relies on the binding pass's
//! dynamic-name fallback for property resolution; v1.1 swaps this
//! for a storage-backed `CatalogProvider`. Production-side
//! consumers SHOULD import [`super::super::InMemoryCatalogProvider`]
//! (a type alias for this struct) so the test/production boundary
//! is visible at the import site.
//!
//! # Determinism
//!
//! ID assignment is order-of-insertion deterministic. `with_labels`
//! / `with_rel_types` / `with_properties` assign monotonically
//! increasing u32 IDs starting at 1 (zero is reserved for the
//! [`arcgraph_core::LabelId::ZERO`] sentinel convention).
//!
//! # v1.0 dynamic-schema fallback
//!
//! `lookup_property` returns `None` for unknown property names —
//! the simplest behavior that exercises the binding pass's
//! "PropertyId is best-effort" path. Tests that need property
//! resolution populate the names up-front via `with_properties`.
//! A future v1.1 catalog impl will add interior mutability to mint
//! IDs lazily; the trait already permits this (the &self method
//! signature does not preclude `RefCell`).

use std::collections::HashMap;

use arcgraph_core::{LabelId, PartitionId, PropertyId, TenantId, TypeId};

use super::{CatalogProvider, CatalogSnapshot, MaxOutDegreeEntry};

/// In-memory `CatalogProvider` impl for tests.
///
/// # Example
/// ```
/// use arcgraph_core::LabelId;
/// use arcgraph_query::semantic::StubCatalogProvider;
/// let cat = StubCatalogProvider::new()
///     .with_labels(["Person", "Doc"])
///     .with_rel_types(["KNOWS"])
///     .with_properties(["age", "name"])
///     .with_vector_index()
///     .with_bm25_index()
///     .with_community_index()
///     .with_label_cardinality(LabelId::new(1), 1_000)
///     .with_total_node_count(2_500);
/// # let _ = cat;
/// ```
#[derive(Debug, Clone)]
pub struct StubCatalogProvider {
    labels: HashMap<String, LabelId>,
    rel_types: HashMap<String, TypeId>,
    properties: HashMap<String, PropertyId>,
    next_label_id: u32,
    next_rel_type_id: u32,
    next_property_id: u32,
    tenant: TenantId,
    partition: PartitionId,
    has_vector_index: bool,
    has_bm25_index: bool,
    has_community_index: bool,
    /// **#1366 (Phase 2).** The set of `(label, property)` pairs that
    /// have an **Online** secondary property index — the RC-6
    /// planner-visible set consumed by
    /// [`CatalogProvider::online_property_index`]. Seeded by
    /// `build_catalog_for_tenant` from the durable `PropertyIndexCatalog`
    /// (Online-only); a Building index is deliberately NOT seeded here.
    online_property_indexes: std::collections::HashSet<(LabelId, String)>,
    /// M4-41: per-label exact cardinalities. `None` returns from the
    /// trait method when a label is absent from the map (matching the
    /// documented "no stats collected" sentinel).
    label_cardinalities: HashMap<LabelId, u64>,
    /// M4-41: per-rel-type exact cardinalities; same `None` semantics.
    rel_type_cardinalities: HashMap<TypeId, u64>,
    /// M4-41: tenant-wide totals. `None` when the corresponding
    /// builder has not been called.
    total_node_count: Option<u64>,
    total_rel_count: Option<u64>,
    /// ADR-025 §5 max out-degree sketch entries.
    max_out_degree: Vec<MaxOutDegreeEntry>,
    /// M4-53: stats-change watermark mirrored on every snapshot. `0`
    /// is the fresh-tenant default; tests that exercise the M4-53
    /// plan cache's stats-change-watermark invalidation use
    /// [`Self::with_commits_observed_count`] to bump it.
    commits_observed_count: u64,
}

impl StubCatalogProvider {
    /// Construct an empty catalog. Default tenant
    /// ([`TenantId::DEFAULT`]) and partition
    /// ([`PartitionId::ZERO`]).
    pub fn new() -> Self {
        Self {
            labels: HashMap::new(),
            rel_types: HashMap::new(),
            properties: HashMap::new(),
            next_label_id: 1,
            next_rel_type_id: 1,
            next_property_id: 1,
            tenant: TenantId::DEFAULT,
            partition: PartitionId::ZERO,
            has_vector_index: false,
            has_bm25_index: false,
            has_community_index: false,
            online_property_indexes: std::collections::HashSet::new(),
            label_cardinalities: HashMap::new(),
            rel_type_cardinalities: HashMap::new(),
            total_node_count: None,
            total_rel_count: None,
            max_out_degree: Vec::new(),
            commits_observed_count: 0,
        }
    }

    /// Pre-populate label names. IDs assigned monotonically from 1.
    pub fn with_labels<I, S>(mut self, labels: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        for name in labels {
            let name = name.into();
            if !self.labels.contains_key(&name) {
                self.labels.insert(name, LabelId::new(self.next_label_id));
                self.next_label_id = self.next_label_id.saturating_add(1);
            }
        }
        self
    }

    /// Pre-populate rel-type names. IDs assigned monotonically from 1.
    pub fn with_rel_types<I, S>(mut self, rel_types: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        for name in rel_types {
            let name = name.into();
            if !self.rel_types.contains_key(&name) {
                self.rel_types
                    .insert(name, TypeId::new(self.next_rel_type_id));
                self.next_rel_type_id = self.next_rel_type_id.saturating_add(1);
            }
        }
        self
    }

    /// W23-M4-08-FINALIZE additive builder — pre-populate a label
    /// name with an EXPLICIT [`LabelId`].
    ///
    /// The W17α [`crate::semantic::InMemoryCatalogProvider`] (alias
    /// for this struct) seeds labels via [`Self::with_labels`] which
    /// assigns IDs monotonically from 1. The
    /// [`arcgraph-storage`](https://docs.rs/arcgraph-storage)
    /// intern table allocates label IDs and rel-type IDs out of the
    /// SAME per-tenant counter, while this builder's
    /// [`Self::with_labels`] / [`Self::with_rel_types`] use separate
    /// counters — so a catalog seeded from a storage snapshot where
    /// labels and rel-types were interned in interleaved order can
    /// have catalog IDs that DIVERGE from the storage IDs for the
    /// same name. The substrate's `scan_nodes` / `expand` filters
    /// by the storage ID; a planner that resolves a rel-type via
    /// the catalog's [`super::CatalogProvider::lookup_rel_type`]
    /// surface will then pass a mismatched ID to the substrate and
    /// silently return zero rows.
    ///
    /// This builder lets a production-style catalog (e.g. the
    /// `arcgraph-mcp::storage::adapters::build_catalog_for_tenant`
    /// helper) install the storage IDs verbatim so the
    /// planner ↔ executor ↔ substrate ID values stay consistent
    /// for label-anchored AND rel-type-anchored queries.
    ///
    /// Idempotent on duplicate `name`s: the FIRST call wins (subsequent
    /// calls with the same name are silently ignored, matching the
    /// `with_labels` contract). `next_label_id` advances past the
    /// supplied ID so a subsequent [`Self::with_labels`] call (or
    /// another `with_label_id` call) does not collide. This keeps
    /// the monotonic-from-1 invariant for [`Self::with_labels`]
    /// callers AND lets `with_label_id` + `with_labels` compose.
    pub fn with_label_id(mut self, name: impl Into<String>, id: LabelId) -> Self {
        use std::collections::hash_map::Entry;
        let name = name.into();
        if let Entry::Vacant(slot) = self.labels.entry(name) {
            slot.insert(id);
            self.next_label_id = self.next_label_id.max(id.raw().saturating_add(1));
        }
        self
    }

    /// W23-M4-08-FINALIZE additive builder — pre-populate a
    /// rel-type name with an EXPLICIT [`TypeId`]. Symmetric to
    /// [`Self::with_label_id`]; see that method's rustdoc for the
    /// rationale + ID-consistency contract.
    pub fn with_rel_type_id(mut self, name: impl Into<String>, id: TypeId) -> Self {
        use std::collections::hash_map::Entry;
        let name = name.into();
        if let Entry::Vacant(slot) = self.rel_types.entry(name) {
            slot.insert(id);
            self.next_rel_type_id = self.next_rel_type_id.max(id.raw().saturating_add(1));
        }
        self
    }

    /// Pre-populate property names. IDs assigned monotonically from 1.
    pub fn with_properties<I, S>(mut self, props: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        for name in props {
            let name = name.into();
            if !self.properties.contains_key(&name) {
                self.properties
                    .insert(name, PropertyId::new(self.next_property_id));
                self.next_property_id = self.next_property_id.saturating_add(1);
            }
        }
        self
    }

    /// Override the tenant ID stamped on `BoundQuery`.
    pub fn with_tenant(mut self, tenant: TenantId) -> Self {
        self.tenant = tenant;
        self
    }

    /// Override the partition ID stamped on `BoundQuery`. v1.0
    /// invariant pins this to `PartitionId::ZERO`; tests may set
    /// a non-zero value to exercise the field plumbing but
    /// production callers MUST not (per ADR-024 amendment-02).
    pub fn with_partition(mut self, partition: PartitionId) -> Self {
        self.partition = partition;
        self
    }

    /// Mark this tenant as having a vector index attached. Consumed
    /// by M4-23's
    /// [`crate::semantic::cross_substrate::CrossSubstrateValidator`].
    pub fn with_vector_index(mut self) -> Self {
        self.has_vector_index = true;
        self
    }

    /// Mark this tenant as having a BM25 (text) index attached.
    /// Consumed by M4-23's
    /// [`crate::semantic::cross_substrate::CrossSubstrateValidator`].
    pub fn with_bm25_index(mut self) -> Self {
        self.has_bm25_index = true;
        self
    }

    /// Mark this tenant as having a community-detection index
    /// attached. Consumed by M4-23's
    /// [`crate::semantic::cross_substrate::CrossSubstrateValidator`].
    pub fn with_community_index(mut self) -> Self {
        self.has_community_index = true;
        self
    }

    /// **#1366 (Phase 2).** Mark `(label, property)` as having an
    /// **Online** secondary property index (RC-6 planner-visible). The
    /// production `build_catalog_for_tenant` seeds this from the durable
    /// catalog's Online indexes; tests use it to drive the planner-
    /// selection + identical-results paths. A Building index is NOT
    /// declared here (that is the Building-not-used contract).
    #[must_use]
    pub fn with_online_property_index(
        mut self,
        label: LabelId,
        property: impl Into<String>,
    ) -> Self {
        self.online_property_indexes
            .insert((label, property.into()));
        self
    }

    /// M4-41 — pre-populate the cardinality of nodes carrying `label`.
    /// Repeated calls overwrite (last write wins); the in-memory
    /// stub does not need atomic update semantics.
    pub fn with_label_cardinality(mut self, label: LabelId, count: u64) -> Self {
        self.label_cardinalities.insert(label, count);
        self
    }

    /// M4-41 — pre-populate the cardinality of relationships of
    /// `rel_type`. Same overwrite semantics as
    /// [`Self::with_label_cardinality`].
    pub fn with_rel_type_cardinality(mut self, rel_type: TypeId, count: u64) -> Self {
        self.rel_type_cardinalities.insert(rel_type, count);
        self
    }

    /// M4-41 — pre-populate the tenant-wide total node count.
    pub fn with_total_node_count(mut self, count: u64) -> Self {
        self.total_node_count = Some(count);
        self
    }

    /// M4-41 — pre-populate the tenant-wide total relationship count.
    pub fn with_total_rel_count(mut self, count: u64) -> Self {
        self.total_rel_count = Some(count);
        self
    }

    /// ADR-025 §5 — pre-populate one max out-degree sketch entry.
    pub fn with_max_out_degree(
        mut self,
        label: LabelId,
        rel_type: TypeId,
        vertex: arcgraph_core::NodeId,
        degree: u64,
    ) -> Self {
        self.max_out_degree.push(MaxOutDegreeEntry {
            label,
            rel_type,
            vertex,
            degree,
        });
        self
    }

    /// M4-53 — pre-populate the stats-change watermark
    /// (`commits_observed_count`) reported on every snapshot.
    ///
    /// Used by M4-53 plan-cache tests to drive the stats-change
    /// invalidation path: bumping this value between two snapshots
    /// simulates a commit landing between cache lookups, which the
    /// M4-53 plan cache invalidates per ADR-038 amendment-03 §TIER-2-a.
    /// `0` is the fresh-tenant default (matches storage-side semantics).
    pub fn with_commits_observed_count(mut self, count: u64) -> Self {
        self.commits_observed_count = count;
        self
    }
}

impl Default for StubCatalogProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl CatalogProvider for StubCatalogProvider {
    fn lookup_label(&self, name: &str) -> Option<LabelId> {
        self.labels.get(name).copied()
    }

    fn lookup_rel_type(&self, name: &str) -> Option<TypeId> {
        self.rel_types.get(name).copied()
    }

    fn lookup_property(&self, name: &str) -> Option<PropertyId> {
        self.properties.get(name).copied()
    }

    fn tenant(&self) -> TenantId {
        self.tenant
    }

    fn partition(&self) -> PartitionId {
        self.partition
    }

    fn has_vector_index(&self) -> bool {
        self.has_vector_index
    }

    fn has_bm25_index(&self) -> bool {
        self.has_bm25_index
    }

    fn has_community_index(&self) -> bool {
        self.has_community_index
    }

    fn online_property_index(&self, label: LabelId, property: &str) -> bool {
        // #1366 (Phase 2): RC-6 planner-visible gate. `(label, property)`
        // is index-eligible iff it was seeded via
        // `with_online_property_index` (Online only). A borrow-friendly
        // membership test over the owned-String set.
        self.online_property_indexes
            .iter()
            .any(|(l, p)| *l == label && p == property)
    }

    fn label_cardinality(&self, label: LabelId) -> Option<u64> {
        self.label_cardinalities.get(&label).copied()
    }

    fn rel_type_cardinality(&self, rel_type: TypeId) -> Option<u64> {
        self.rel_type_cardinalities.get(&rel_type).copied()
    }

    fn total_node_count(&self) -> Option<u64> {
        self.total_node_count
    }

    fn total_rel_count(&self) -> Option<u64> {
        self.total_rel_count
    }

    fn snapshot(&self) -> CatalogSnapshot {
        // Assemble a query-side snapshot from the in-memory builder
        // maps. Cross-key consistency is trivial here because the stub
        // is single-threaded and immutable after construction; the
        // production storage impl achieves the same guarantee through
        // the two-marker SeqLock protocol per
        // `arcgraph_storage::catalog::stats` module docs.
        let label_cards = self
            .label_cardinalities
            .iter()
            .map(|(l, c)| (*l, *c))
            .collect::<Vec<_>>();
        let rel_type_cards = self
            .rel_type_cardinalities
            .iter()
            .map(|(t, c)| (*t, *c))
            .collect::<Vec<_>>();
        CatalogSnapshot::from_parts_with_max_out_degree(
            self.total_node_count,
            self.total_rel_count,
            label_cards,
            rel_type_cards,
            self.commits_observed_count,
            self.max_out_degree.clone(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_stub_resolves_nothing() {
        let cat = StubCatalogProvider::new();
        assert!(cat.lookup_label("Person").is_none());
        assert!(cat.lookup_rel_type("KNOWS").is_none());
        assert!(cat.lookup_property("age").is_none());
    }

    #[test]
    fn fluent_builder_assigns_ids_monotonically() {
        let cat = StubCatalogProvider::new()
            .with_labels(["Person", "Doc"])
            .with_rel_types(["KNOWS", "WROTE"])
            .with_properties(["age", "name"]);
        let person = cat.lookup_label("Person").expect("Person label");
        let doc = cat.lookup_label("Doc").expect("Doc label");
        assert_ne!(person, doc);
        let knows = cat.lookup_rel_type("KNOWS").expect("KNOWS rel-type");
        let wrote = cat.lookup_rel_type("WROTE").expect("WROTE rel-type");
        assert_ne!(knows, wrote);
        let age = cat.lookup_property("age").expect("age property");
        let name = cat.lookup_property("name").expect("name property");
        assert_ne!(age, name);
    }

    #[test]
    fn duplicate_inserts_are_idempotent() {
        let cat = StubCatalogProvider::new()
            .with_labels(["Person"])
            .with_labels(["Person"]);
        // Same name, same ID — no monotonic bump.
        assert_eq!(cat.lookup_label("Person"), Some(LabelId::new(1)));
    }

    #[test]
    fn defaults_to_default_tenant_and_zero_partition() {
        let cat = StubCatalogProvider::new();
        assert_eq!(cat.tenant(), TenantId::DEFAULT);
        assert_eq!(cat.partition(), PartitionId::ZERO);
    }

    #[test]
    fn with_tenant_and_partition_override() {
        let cat = StubCatalogProvider::new()
            .with_tenant(TenantId::new(42))
            .with_partition(PartitionId::new(7));
        assert_eq!(cat.tenant(), TenantId::new(42));
        assert_eq!(cat.partition(), PartitionId::new(7));
    }

    #[test]
    fn substrate_flags_default_to_false() {
        let cat = StubCatalogProvider::new();
        assert!(!cat.has_vector_index());
        assert!(!cat.has_bm25_index());
        assert!(!cat.has_community_index());
    }

    #[test]
    fn substrate_flags_set_independently_via_builder() {
        let cat = StubCatalogProvider::new().with_vector_index();
        assert!(cat.has_vector_index());
        assert!(!cat.has_bm25_index());
        assert!(!cat.has_community_index());

        let cat = StubCatalogProvider::new().with_bm25_index();
        assert!(!cat.has_vector_index());
        assert!(cat.has_bm25_index());
        assert!(!cat.has_community_index());

        let cat = StubCatalogProvider::new().with_community_index();
        assert!(!cat.has_vector_index());
        assert!(!cat.has_bm25_index());
        assert!(cat.has_community_index());
    }

    #[test]
    fn substrate_flags_compose() {
        let cat = StubCatalogProvider::new()
            .with_vector_index()
            .with_bm25_index()
            .with_community_index();
        assert!(cat.has_vector_index());
        assert!(cat.has_bm25_index());
        assert!(cat.has_community_index());
    }

    #[test]
    fn stub_snapshot_round_trips_fluent_builder_state() {
        // M4-51: StubCatalogProvider::snapshot() must surface the
        // fluent-builder cardinalities exactly. Tests of the M4-51
        // cost planner depend on this round-trip — without it, every
        // cost-model test would have to be wired through the storage
        // crate's CatalogStats producer.
        let l1 = LabelId::new(1);
        let l2 = LabelId::new(7); // out-of-order label id
        let t1 = TypeId::new(2);
        let cat = StubCatalogProvider::new()
            .with_label_cardinality(l1, 100)
            .with_label_cardinality(l2, 250)
            .with_rel_type_cardinality(t1, 500)
            .with_total_node_count(1_000)
            .with_total_rel_count(2_000);
        let snap = cat.snapshot();

        assert_eq!(snap.total_nodes(), Some(1_000));
        assert_eq!(snap.total_rels(), Some(2_000));
        assert_eq!(snap.label_card(l1), Some(100));
        assert_eq!(snap.label_card(l2), Some(250));
        assert_eq!(snap.label_card(LabelId::new(99)), None);
        assert_eq!(snap.rel_type_card(t1), Some(500));
        assert_eq!(snap.rel_type_card(TypeId::new(99)), None);
        assert!(snap.has_observed_any());

        // Sorted-by-raw-id invariant: snapshot's label_cards must be
        // sorted regardless of fluent-builder insertion order.
        let raws: Vec<u32> = snap.label_cards().iter().map(|(l, _)| l.raw()).collect();
        assert_eq!(raws, vec![1, 7]);

        // Empty stub round-trips to empty snapshot.
        let empty = StubCatalogProvider::new();
        let empty_snap = empty.snapshot();
        assert_eq!(empty_snap.total_nodes(), None);
        assert!(empty_snap.label_cards().is_empty());
    }

    #[test]
    fn stub_catalog_provider_with_stats_builders() {
        // M4-41: fluent stats builders mint cardinalities the trait
        // method then surfaces. Absent labels / rel-types stay None,
        // matching the production catalog's "no stats collected"
        // sentinel.
        let l1 = LabelId::new(1);
        let l2 = LabelId::new(2);
        let t1 = TypeId::new(1);
        let cat = StubCatalogProvider::new()
            .with_label_cardinality(l1, 100)
            .with_rel_type_cardinality(t1, 250)
            .with_total_node_count(1_000)
            .with_total_rel_count(2_000);

        assert_eq!(cat.label_cardinality(l1), Some(100));
        // l2 was never minted — stays None (NOT Some(0)).
        assert_eq!(cat.label_cardinality(l2), None);
        assert_eq!(cat.rel_type_cardinality(t1), Some(250));
        assert_eq!(cat.rel_type_cardinality(TypeId::new(99)), None);
        assert_eq!(cat.total_node_count(), Some(1_000));
        assert_eq!(cat.total_rel_count(), Some(2_000));

        // Defaults: pre-builder stub returns None for all stats —
        // documents the "fresh tenant" contract.
        let empty = StubCatalogProvider::new();
        assert_eq!(empty.label_cardinality(l1), None);
        assert_eq!(empty.rel_type_cardinality(t1), None);
        assert_eq!(empty.total_node_count(), None);
        assert_eq!(empty.total_rel_count(), None);
    }
}
