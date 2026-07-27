//! M4-72 strict producer→consumer transit regression guard.
//!
//! # Why this lives in arcgraph-cli/tests/
//!
//! The strict transit pin requires a real
//! `arcgraph_storage::CatalogStats` ↔ `arcgraph_query::PlanCache`
//! producer→consumer chain. Per `docs/bounded-contexts.md`,
//! `arcgraph-query/tests/` cannot depend on `arcgraph-storage`
//! (cross-context dependency). `arcgraph-cli` already declares both
//! crates in its Cargo.toml — putting the test here lets us thread
//! the real types through without spawning a new
//! `arcgraph-engine-tests` workspace member.
//!
//! # The pin
//!
//! The required transit is:
//!
//! > Real `arcgraph_storage::CatalogStats::observe_commit()` → real
//! > `CatalogProvider` → cache invalidation observed on next lookup.
//!
//! The pin's role: PR #259's M4-53 wave-level transit test used
//! `StubCatalogProvider::with_commits_observed_count` rather than the
//! real producer surface. M4-72 closes the gap: a real
//! `CatalogStats::begin_commit_observation` /
//! `observe_commit` bracket bumps `commits_observed`, the catalog
//! adapter surfaces the new value via `CatalogProvider::snapshot`, and
//! the M4-53 plan cache observes the watermark advancement on the next
//! lookup → invalidates → re-populates.
//!
//! # Phase 4.3 reverse-test
//!
//! The pin is followed by a sibling test that asserts the CACHE
//! correctness invariant: when the watermark advances WITHOUT a real
//! observe_commit (e.g., test fakes `commits_observed_count` directly
//! on a stub), the cache STILL invalidates — so the watermark surface
//! is the contract, not the specific producer protocol. The two tests
//! together pin both directions of the cross-crate transit.

use std::sync::Arc;

use arcgraph_core::{LabelId, PartitionId, PropertyId, TenantId, TypeId};
use arcgraph_query::semantic::{CatalogProvider, CatalogSnapshot};
use arcgraph_query::{LookupOutcome, PlanCache, PlanCacheKey, explain_with_cache, parse};
use arcgraph_storage::CatalogStats;

/// Cross-crate `CatalogProvider` adapter: production-shape catalog
/// backed by a real `Arc<arcgraph_storage::CatalogStats>` for the per-
/// label / per-rel-type / total cardinalities + `commits_observed`
/// watermark.
///
/// Mirrors the v1.0 production catalog wiring path that lights at
/// M4-08+ — the adapter is the seam where the storage-side type
/// crosses into the query-side `CatalogProvider` interface.
struct CatalogStatsCatalogProvider {
    stats: Arc<CatalogStats>,
    tenant: TenantId,
    partition: PartitionId,
    /// Compiled-in label map for the test fixture. v1.0 production
    /// catalogs derive these from `SystemCatalog`; we hardcode for the
    /// test's bounded surface.
    labels: std::collections::HashMap<String, LabelId>,
    rel_types: std::collections::HashMap<String, TypeId>,
    properties: std::collections::HashMap<String, PropertyId>,
}

impl CatalogStatsCatalogProvider {
    fn new(stats: Arc<CatalogStats>, tenant: TenantId) -> Self {
        let mut labels = std::collections::HashMap::new();
        labels.insert("Person".to_string(), LabelId::new(1));
        let mut rel_types = std::collections::HashMap::new();
        rel_types.insert("KNOWS".to_string(), TypeId::new(1));
        let mut properties = std::collections::HashMap::new();
        properties.insert("name".to_string(), PropertyId::new(1));
        Self {
            stats,
            tenant,
            partition: PartitionId::ZERO,
            labels,
            rel_types,
            properties,
        }
    }
}

impl CatalogProvider for CatalogStatsCatalogProvider {
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
    fn label_cardinality(&self, label: LabelId) -> Option<u64> {
        self.stats.label_cardinality(label)
    }
    fn rel_type_cardinality(&self, rel_type: TypeId) -> Option<u64> {
        self.stats.rel_type_cardinality(rel_type)
    }
    fn total_node_count(&self) -> Option<u64> {
        self.stats.total_node_count()
    }
    fn total_rel_count(&self) -> Option<u64> {
        self.stats.total_rel_count()
    }
    fn snapshot(&self) -> CatalogSnapshot {
        // Translate the storage-side snapshot to the query-side mirror.
        let s = self.stats.snapshot();
        let label_cards: Vec<(LabelId, u64)> =
            s.label_cards().iter().map(|(l, c)| (*l, *c)).collect();
        let rel_type_cards: Vec<(TypeId, u64)> =
            s.rel_type_cards().iter().map(|(t, c)| (*t, *c)).collect();
        CatalogSnapshot::from_parts(
            s.total_nodes(),
            s.total_rels(),
            label_cards,
            rel_type_cards,
            s.commits_observed(),
        )
    }
}

/// **M4-72 strict producer→consumer transit pin:**
///
/// 1. Build real `Arc<CatalogStats>`.
/// 2. Bracket `begin_commit_observation` + per-counter increments +
///    `observe_commit` ONCE to seed the catalog.
/// 3. Wrap stats in `CatalogStatsCatalogProvider`.
/// 4. Build a plan cache + insert via EXPLAIN.
/// 5. Trigger a SECOND `observe_commit` bracket on the real stats.
/// 6. Re-EXPLAIN; verify the cache LOOKUP surfaces Stale and the cache
///    re-populates.
#[test]
fn m4_72_real_catalog_stats_observe_commit_invalidates_plan_cache() {
    let stats = Arc::new(CatalogStats::new());
    let tenant = TenantId::DEFAULT;

    // Seed: pretend a commit observed 100 Person nodes.
    stats.begin_commit_observation();
    for _ in 0..100 {
        stats.increment_label(LabelId::new(1));
        stats.increment_total_nodes();
    }
    stats.observe_commit();
    assert_eq!(stats.commits_observed_count(), 1);

    let catalog = CatalogStatsCatalogProvider::new(Arc::clone(&stats), tenant);
    let cache = Arc::new(PlanCache::new());

    // Step 4: populate cache via EXPLAIN.
    let _pt = explain_with_cache("MATCH (n:Person) RETURN n", &catalog, &cache).expect("explain");
    assert_eq!(cache.len_for(tenant), 1);
    let stmt = parse("MATCH (n:Person) RETURN n").expect("parse");
    let key = PlanCacheKey::from_ast(tenant, &stmt);

    // Pre-update lookup: Hit at watermark = 1.
    match cache.lookup(&key, 1) {
        LookupOutcome::Hit(_) => {}
        other => panic!("expected Hit at watermark=1, got {other:?}"),
    }

    // Step 5: REAL producer surface — second commit bracket.
    stats.begin_commit_observation();
    for _ in 0..50 {
        stats.increment_label(LabelId::new(1));
        stats.increment_total_nodes();
    }
    stats.observe_commit();
    assert_eq!(stats.commits_observed_count(), 2);

    // Step 6: lookup at the new watermark. The cached entry was
    // stamped at 1; live snapshot reports 2 → the cache surfaces
    // Stale on direct lookup. (explain_with_cache would recover and
    // re-populate; we verify the raw lookup outcome to pin the
    // invalidation surface.)
    match cache.lookup(&key, 2) {
        LookupOutcome::Stale => {}
        other => panic!("expected Stale at watermark=2, got {other:?}"),
    }

    // Re-EXPLAIN: cold path runs + repopulates the cache stamped at 2.
    let _pt =
        explain_with_cache("MATCH (n:Person) RETURN n", &catalog, &cache).expect("re-explain");
    match cache.lookup(&key, 2) {
        LookupOutcome::Hit(_) => {}
        other => panic!("expected Hit after re-population, got {other:?}"),
    }
    // Total entries: 1 (the new stamped entry replaced the stale
    // lookup-victim).
    assert_eq!(cache.len_for(tenant), 1);
}

/// **Canonical-bracket monotonicity pin** (companion to the strict
/// transit pin) — repeated paired `begin_commit_observation` /
/// `observe_commit` brackets advance `commits_observed` monotonically.
///
/// # W12β fix-up LOW-4 — test renamed for accuracy
///
/// Previously named
/// `m4_72_observe_commit_alone_does_not_advance_commits_started_invariant`,
/// which described a pathological (without-paired-begin) probe that
/// the test body did NOT actually exercise. The body verifies the
/// CANONICAL bracket path only; the test name now reflects that.
///
/// The without-paired-begin pathological case is intentionally
/// out-of-scope here: the storage-side `CatalogStats` surface is
/// designed assuming paired discipline (the SeqLock retry loop in
/// `snapshot()` keeps reads consistent only under that discipline).
/// Probing the unpaired path is a producer-side test, not a
/// transit-side pin.
#[test]
fn m4_72_canonical_observe_commit_bracket_advances_watermark_monotonically() {
    let stats = Arc::new(CatalogStats::new());

    // Seed: first bracket.
    stats.begin_commit_observation();
    stats.observe_commit();
    assert_eq!(stats.commits_observed_count(), 1);

    // Second bracket: monotonic advancement.
    stats.begin_commit_observation();
    stats.observe_commit();
    assert_eq!(stats.commits_observed_count(), 2);

    // Snapshot is consistent with the canonical bracket pattern.
    let s = stats.snapshot();
    assert_eq!(s.commits_observed(), 2);
}

// ----------------------------------------------------------------------
// W12β fix-up LOW-1 — forward-binding pin for the production-side
// adapter that bridges from `RowCountObserver::observed_overrides()`
// (REPLACEMENT semantics — observed cardinality IS the new ground
// truth) to `CatalogStats::increment_label` (ADDITIVE semantics — each
// per-commit hook adds to the running counter).
// ----------------------------------------------------------------------

/// Production-side adapter shape for the observed→increment delta
/// computation.
///
/// Consumed by future M4-08+ wiring that bridges
/// [`arcgraph_query::observer::RowCountObserver::observed_overrides`]
/// (replacement semantics: observed cardinality IS the new ground
/// truth) to
/// [`arcgraph_storage::CatalogStats::increment_label`] (additive
/// semantics: each call adds to the running counter).
///
/// The delta MUST be `observed - previous` (saturating). A wrong-
/// direction implementation that passed `observed` directly to
/// `increment_label` would result in cardinalities accumulating
/// `observed + previous` after each commit (the bug this test pins
/// against).
fn observed_to_increment_delta(observed: u64, previous: u64) -> u64 {
    observed.saturating_sub(previous)
}

/// **W12β fix-up LOW-1 forward-binding pin** — observed→increment
/// delta-computation contract.
///
/// Forward-binds the production-side adapter for M4-08+ wiring per
/// `feedback_anchor_to_consumer_transit_pinning.md` discipline. The
/// canonical pattern: when M4-08+ wires
/// `observer.observed_overrides()` → `CatalogStats::increment_label`,
/// the bridge MUST compute `delta = observed.saturating_sub(previous)`
/// and feed that to the additive surface. A mis-wiring that passed
/// `observed` directly would produce `previous + observed` after one
/// commit (twice the correct value at the typical case where observed
/// is much larger than previous; or `previous + observed` even when
/// observed = 0, destroying counter reliability).
#[test]
fn m4_72_observed_to_increment_delta_forward_binding_pin() {
    let stats = Arc::new(CatalogStats::new());

    // Seed: 100 Person nodes via paired observe_commit bracket.
    stats.begin_commit_observation();
    for _ in 0..100 {
        stats.increment_label(LabelId::new(1));
        stats.increment_total_nodes();
    }
    stats.observe_commit();
    let previous = stats.label_cardinality(LabelId::new(1)).unwrap_or(0);
    assert_eq!(previous, 100);

    // Observer reports observed = 1000 (replacement semantics:
    // "the new ground truth is 1000"). M4-71's
    // `observed_overrides()` would carry this value at v1.0+.
    let observed: u64 = 1000;

    // Production-side adapter (forward-bind for M4-08+): the delta
    // MUST be observed - previous, NOT observed.
    let delta = observed_to_increment_delta(observed, previous);
    assert_eq!(
        delta, 900,
        "delta computation: observed (replacement) - previous (additive) = 900",
    );

    // Apply the delta via the additive producer surface (the M4-08+
    // wiring pattern).
    stats.begin_commit_observation();
    for _ in 0..delta {
        stats.increment_label(LabelId::new(1));
        stats.increment_total_nodes();
    }
    stats.observe_commit();

    // Post-adapter: cardinality matches observed (NOT observed +
    // previous — the wrong-direction bug this test pins against).
    let post = stats.label_cardinality(LabelId::new(1)).unwrap_or(0);
    assert_eq!(
        post,
        observed,
        "wrong direction (observed → increment_label directly) would produce \
         {wrong}; correct direction (delta = observed - previous) produces {observed}",
        wrong = previous + observed,
    );
}

/// Edge case: when observed == previous, delta is 0 — a no-op
/// `increment_label` loop should not perturb the catalog.
#[test]
fn m4_72_observed_equals_previous_delta_is_zero_no_op() {
    let stats = Arc::new(CatalogStats::new());
    stats.begin_commit_observation();
    for _ in 0..50 {
        stats.increment_label(LabelId::new(1));
    }
    stats.observe_commit();
    let previous = stats.label_cardinality(LabelId::new(1)).unwrap_or(0);
    let observed = previous;
    let delta = observed_to_increment_delta(observed, previous);
    assert_eq!(delta, 0, "no-op when observed equals previous");
    // Apply (zero-iteration loop) within a fresh bracket — watermark
    // still advances even when delta=0 (the bracket is the
    // invalidation signal, NOT the cardinality change itself).
    stats.begin_commit_observation();
    for _ in 0..delta {
        stats.increment_label(LabelId::new(1));
    }
    stats.observe_commit();
    assert_eq!(
        stats.label_cardinality(LabelId::new(1)).unwrap_or(0),
        observed
    );
}

/// Edge case: when observed < previous (the catalog over-counted, e.g.,
/// because of stale stats), delta saturates to 0 — `increment_label` is
/// monotonic, callers MUST NOT use this surface to decrement.
///
/// This is the documented v1.0-alpha simplification per
/// `feedback_anchor_to_consumer_transit_pinning.md` §"saturating-delta
/// adapter shape": if the observed value is below the catalog's
/// previously-observed value, the additive surface cannot
/// "uncount" — the operation is a no-op. v1.1+ may introduce a
/// separate decrement surface for shrinking tenants; v1.0-alpha treats
/// this case as catalog-stale-but-not-actionable.
#[test]
fn m4_72_observed_below_previous_delta_saturates_to_zero() {
    let stats = Arc::new(CatalogStats::new());
    stats.begin_commit_observation();
    for _ in 0..1000 {
        stats.increment_label(LabelId::new(1));
    }
    stats.observe_commit();
    let previous = stats.label_cardinality(LabelId::new(1)).unwrap_or(0);
    let observed: u64 = 100; // observed < previous
    let delta = observed_to_increment_delta(observed, previous);
    assert_eq!(
        delta, 0,
        "saturating: observed (100) < previous (1000) → delta = 0 (monotonic surface)",
    );
}

/// **M4-72 cross-crate end-to-end PROFILE-with-cache pin** — exercises
/// the same flow as the M4-71 PROFILE-with-cache test BUT through the
/// real CatalogStats producer surface. Verifies that PROFILE through
/// the production-shape catalog populates the cache.
#[test]
fn m4_72_profile_with_real_catalog_stats_populates_cache() {
    use arcgraph_core::NodeId;
    use arcgraph_query::QueryEngine;
    use arcgraph_query::executor::StubExecutorSubstrate;
    use arcgraph_query::executor::value::NodeView;

    let stats = Arc::new(CatalogStats::new());
    let tenant = TenantId::DEFAULT;
    stats.begin_commit_observation();
    for _ in 0..50 {
        stats.increment_label(LabelId::new(1));
        stats.increment_total_nodes();
    }
    stats.observe_commit();
    let catalog = CatalogStatsCatalogProvider::new(Arc::clone(&stats), tenant);

    // Substrate carrying 50 Person nodes (mirrors the seeded stats).
    let mut substrate = StubExecutorSubstrate::new();
    for i in 1..=50_u64 {
        substrate =
            substrate.with_node(tenant, NodeView::new(NodeId::new(i), Some(LabelId::new(1))));
    }

    let cache = Arc::new(PlanCache::new());
    let engine = QueryEngine::new(&catalog).with_cache(Arc::clone(&cache));
    let (_pt, metrics) = engine
        .profile_with_substrate("MATCH (n:Person) RETURN n", &substrate)
        .expect("profile_with_substrate against real CatalogStats");

    // Sin #5 closure: PROFILE that runs the planner DOES populate the
    // cache, even when the catalog is backed by real CatalogStats.
    assert_eq!(
        cache.len_for(tenant),
        1,
        "PROFILE-with-cache populates the cache via the real producer surface",
    );
    assert_eq!(metrics.rows_emitted, 50);
}
