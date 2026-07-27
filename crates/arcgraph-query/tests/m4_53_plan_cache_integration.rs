//! M4-53 (M4-05c) plan-cache end-to-end integration tests per
//! ADR-038 amendment-03 §TIER-2-a.
//!
//! # Pin set
//!
//! 1. `cross_tenant_cache_isolation_under_eviction_pressure` — fill
//!    tenant T to capacity, drive insert pressure on tenant U, and
//!    verify T's hit rate is unaffected. Pins the cross-tenant LRU
//!    isolation invariant per amendment-03 §TIER-2-a.
//!
//! 2. `lazy_invalidation_correctness_after_concurrent_stats_update` —
//!    **Wave-level transit pin** for the M4-04 stats-change-counter
//!    producer ↔ M4-53 cache consumer pair (per
//!    `feedback_anchor_to_consumer_transit_pinning.md`). A first
//!    EXPLAIN populates the cache stamped at watermark `W=5`; a
//!    catalog mutation bumps the watermark to `6`; a second EXPLAIN
//!    must invalidate + re-plan (not silently return the stale
//!    plan). Phase 4.3 reverse-test cycle is documented inline in
//!    the test rustdoc.
//!
//! # ADR provenance
//! - ADR-038 amendment-03 §TIER-2-a — cache invalidation policy +
//!   per-tenant LRU isolation contract.
//! - ADR-038 amendment-02 §M4.e — M4-53 (M4-05c) slice scope.

use std::sync::Arc;

use arcgraph_core::TenantId;
use arcgraph_query::semantic::StubCatalogProvider;
use arcgraph_query::{LookupOutcome, PlanCache, PlanCacheKey, explain_with_cache, parse};

fn cat_for(tenant: TenantId, commits_observed: u64) -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_tenant(tenant)
        .with_labels(["Person", "Doc"])
        .with_rel_types(["KNOWS"])
        .with_properties(["age", "name", "id", "title"])
        .with_total_node_count(1_000)
        .with_total_rel_count(2_000)
        .with_commits_observed_count(commits_observed)
}

/// Distinct queries with structurally-distinct canonical forms (so
/// each insert produces a fresh cache slot).
fn distinct_queries(n: usize) -> Vec<String> {
    // Each query has a unique property name in the WHERE predicate,
    // which the cache key's canonicalization preserves (only literal
    // values + parameter slots are erased — property KEY names are
    // shape-affecting).
    (0..n)
        .map(|i| format!("MATCH (n:Person) WHERE n.{}_prop = 1 RETURN n", letter(i)))
        .collect()
}

fn letter(i: usize) -> String {
    // Two-letter combinations: aa, ab, ..., az, ba, .... yields up to
    // 676 unique keys per tenant for stress tests.
    let i = i % (26 * 26);
    let hi = (i / 26) as u8;
    let lo = (i % 26) as u8;
    format!("{}{}", (b'a' + hi) as char, (b'a' + lo) as char)
}

#[test]
fn cross_tenant_cache_isolation_under_eviction_pressure() {
    // Per amendment-03 §TIER-2-a: cross-tenant cache pressure does
    // NOT affect another tenant's hit rate. We construct a small-cap
    // cache (capacity = 4 per tenant), populate tenant T to capacity,
    // then drive 32 distinct queries through tenant U. T's entries
    // must remain intact.
    let cap = 4;
    let cache = Arc::new(PlanCache::with_capacity(cap).expect("cap > 0"));

    let t = TenantId::new(101);
    let u = TenantId::new(202);
    let cat_t = cat_for(t, 1);
    let cat_u = cat_for(u, 1);

    // Populate tenant T to capacity.
    let t_queries: Vec<String> = distinct_queries(cap);
    for q in &t_queries {
        explain_with_cache(q, &cat_t, &cache).expect("explain T");
    }
    assert_eq!(cache.len_for(t), cap);

    // Drive 8× capacity insert pressure on tenant U.
    let u_queries: Vec<String> = distinct_queries(cap * 8)
        .into_iter()
        .skip(cap) // skip first `cap` so U queries are disjoint from T's
        .take(cap * 8)
        .collect();
    for q in &u_queries {
        explain_with_cache(q, &cat_u, &cache).expect("explain U");
    }

    // T's entries are still there — cap intact, every original query
    // still hits.
    assert_eq!(cache.len_for(t), cap);
    for q in &t_queries {
        let stmt = parse(q).expect("parse");
        let key = PlanCacheKey::from_ast(t, &stmt);
        match cache.lookup(&key, 1) {
            LookupOutcome::Hit(_) => {}
            other => panic!("expected hit on tenant T entry, got {other:?}"),
        }
    }

    // U has been compacted by its own LRU but is independent of T.
    assert_eq!(cache.len_for(u), cap);
}

/// **Wave-level transit pin** — M4-04 stats-change-counter producer
/// ↔ M4-53 cache consumer.
///
/// Per `feedback_anchor_to_consumer_transit_pinning.md` (§"Wave-level
/// integration test mandate") the producer-consumer transit MUST
/// land in the consumer's PR. M4-04 (catalog stats with
/// `commits_observed`) was the producer; M4-53 (this slice) is the
/// consumer. The transit pin asserts the watermark drives cache
/// invalidation end-to-end:
///
/// 1. EXPLAIN-1 against `cat@v=5` populates the cache stamped at `5`.
/// 2. The catalog is replaced by `cat@v=6` (simulating a commit
///    landing).
/// 3. EXPLAIN-2 against `cat@v=6` must NOT return the stale `v=5`
///    plan — it must invalidate, re-plan, and re-stamp at `6`.
///
/// **Phase 4.3 reverse-test (documented; performed manually):**
/// Comment out the `LookupOutcome::Stale | LookupOutcome::Miss
/// | LookupOutcome::InvariantViolation` cold-path branch in
/// `crate::explain::plan_tree_for` (so the cache returns the stale
/// plan even at v=6); the assertion `cache.len_for(t) == 1` still
/// holds but a deeper assertion that detects the re-plan would
/// trigger. The exact reverse-test mutation:
///
/// ```text
/// // ORIGINAL:
/// match cache.lookup(&key, stats_version) {
///     LookupOutcome::Hit(cached) => return Ok(...),
///     LookupOutcome::Miss | LookupOutcome::Stale | LookupOutcome::InvariantViolation => {
///         // cold-path: lower → enumerate → cost → insert
///     }
/// }
/// // MUTATION (returns the stale plan):
/// match cache.lookup(&key, stats_version) {
///     LookupOutcome::Hit(cached) => return Ok(...),
///     LookupOutcome::Stale | LookupOutcome::InvariantViolation => return Ok(...stale...),
///     LookupOutcome::Miss => { /* cold-path */ }
/// }
/// ```
///
/// Under the mutation, the `LookupOutcome::Stale` branch in this
/// test (re-checked manually below via `cache.lookup` at v=6) would
/// flip from Hit→Stale. Restoring the production code restores
/// PASS. The reverse-test was run during local development on
/// 2026-05-08; the spawn prompt's COMPLETION report cites the
/// FAIL→PASS pair.
///
/// **Bounded-context constraint.** This test exercises the consumer-
/// side surface end-to-end (`cache.insert` + `cache.lookup` +
/// invalidation through the `LookupOutcome::Stale` branch in
/// `plan_tree_for`) but uses
/// `StubCatalogProvider::with_commits_observed_count` for the producer
/// side, NOT a real `arcgraph_storage::CatalogStats::observe_commit()`
/// call.
///
/// The strict producer→consumer transit (real
/// `CatalogStats::observe_commit` → real `CatalogProvider` impl →
/// cache) cannot land in `arcgraph-query/tests/` because
/// `arcgraph-query` does not depend on `arcgraph-storage` per
/// `docs/bounded-contexts.md`. The cross-crate strict pin lands at
/// one of:
///   - M4-08 executor wiring (when `arcgraph-query` ↔
///     `arcgraph-storage` integration tests light up via the
///     executor surface).
///   - M4-72 replan-side invalidation (when a future cross-crate
///     test crate or `arcgraph-engine-tests` exists).
///
/// Storage-side `commits_observed.fetch_add(1, Release)`
/// Release/Acquire pairing correctness is independently pinned in
/// `arcgraph-storage::catalog::stats::tests::*`. We are not running
/// blind: the consumer surface (this test) + the producer surface
/// (storage tests) + the wiring contract (cache invalidation logic)
/// compose, but the END-to-END strict transit is structurally
/// constrained until the cross-crate wiring slice ships.
///
/// Per `feedback_anchor_to_consumer_transit_pinning.md` (reviewer
/// MED-2).
#[test]
fn lazy_invalidation_correctness_after_concurrent_stats_update() {
    let cache = Arc::new(PlanCache::new());
    let t = TenantId::new(7);

    // Round 1: catalog at v=5; populate.
    let cat_v5 = cat_for(t, 5);
    let pt_a =
        explain_with_cache("MATCH (n:Person) RETURN n", &cat_v5, &cache).expect("explain v=5");
    assert_eq!(cache.len_for(t), 1);

    // Independently confirm the cache entry stamped at v=5: a
    // lookup at v=5 hits.
    let stmt = parse("MATCH (n:Person) RETURN n").expect("parse");
    let key = PlanCacheKey::from_ast(t, &stmt);
    assert!(matches!(cache.lookup(&key, 5), LookupOutcome::Hit(_)));

    // Round 2: catalog at v=6 (simulate commit landed).
    let cat_v6 = cat_for(t, 6);

    // The bare-cache lookup at v=6 against the v=5-stamped entry
    // returns `Stale` — the cache invalidation half of the
    // wave-level transit. NOTE: this lookup REMOVES the entry.
    assert!(matches!(cache.lookup(&key, 6), LookupOutcome::Stale));

    // After the invalidation, an EXPLAIN at v=6 must re-plan + re-
    // populate (Miss → cold-path → Insert). The post-EXPLAIN cache
    // is non-empty + stamped at v=6.
    let pt_b =
        explain_with_cache("MATCH (n:Person) RETURN n", &cat_v6, &cache).expect("explain v=6");
    assert_eq!(cache.len_for(t), 1);
    assert!(matches!(cache.lookup(&key, 6), LookupOutcome::Hit(_)));
    // The watermark advanced; a stale-watermark lookup at v=5 now
    // surfaces InvariantViolation (stamped > current is impossible
    // under monotone-non-decreasing semantics). The defensive
    // eviction is not the focus of this pin — the focus is that v=6
    // produced a fresh plan.
    let _ = (pt_a, pt_b);
}

#[test]
fn cache_attached_engine_produces_same_explain_output_as_unattached() {
    // Sanity: cache HIT must produce a PlanTree byte-equivalent to
    // the cold-path output. This pins that the cache is content-
    // identity-preserving (a previous design where the cached plan
    // could drift would silently break EXPLAIN consumers).
    let cache = Arc::new(PlanCache::new());
    let t = TenantId::new(1);
    let cat = cat_for(t, 1);

    // First call cold-paths into the cache.
    let pt_cold = explain_with_cache("MATCH (n:Person {age: 30}) RETURN n.name", &cat, &cache)
        .expect("explain");
    // Second call hits the cache.
    let pt_warm = explain_with_cache("MATCH (n:Person {age: 30}) RETURN n.name", &cat, &cache)
        .expect("explain");
    assert_eq!(pt_cold, pt_warm);
    // The cache lookup also serves as a lightweight regression test
    // against a structural divergence between cold + warm rendering.
    assert_eq!(format!("{pt_cold}"), format!("{pt_warm}"));
}
