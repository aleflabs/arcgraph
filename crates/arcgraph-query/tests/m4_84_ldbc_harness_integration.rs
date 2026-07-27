//! M4-84 LDBC SNB Interactive-Short integration tests.
//!
//! Closes ADR-038 amendment-02 §M4.h LDBC SNB harness wiring per
//! design-v2 §10.5 + LDBC SNB Interactive Specification §3.5.
//!
//! # Plan-time scope at v1-alpha
//!
//! The IS1..IS7 queries lower to `LogicalJoin` nodes (every multi-
//! step path pattern produces a Scan → Expand → Scan tree joined on
//! shared bindings per `lowering.rs::lower_path_pattern`); the M4-61
//! vectorized executor does NOT support `LogicalJoin` at v1-alpha
//! (per `executor/pipeline.rs::build` which surfaces
//! `ExecutionError::NotImplemented { feature: "LogicalJoin
//! (multi-pattern equi-join)", target_version: "M4-63 / M4-64" }`).
//!
//! The full-execute LDBC harness is therefore forward-deferred to
//! M4-64; the M4-84 v1-alpha harness measures the **plan-build
//! pipeline** end-to-end (parse, bind, typecheck, cross-substrate,
//! lower, enumerate, cost, plan-tree-render) — the production-ready
//! surface lit by M4-91 EXPLAIN.
//!
//! Plan-build IS a load-bearing perf gate: per design-v2 §10.5 IS1
//! P50 = 50µs, the plan-build half dominates short-execute latencies
//! (per ADR-036 §D-25 the M4-05 plan-build budget is 5ms for 8-way
//! joins; LDBC IS queries are 1-3 hop). The full-execute number
//! lights at M6 LDBC perf milestone with the real SF-1.0/10/30
//! datasets per the LDBC SNB driver contract.
//!
//! # Test inventory (per W13γ spawn prompt)
//!
//! 1. `ldbc_harness_scaffolding` — every IS1..IS7 query parses,
//!    binds, type-checks, lowers, enumerates joins, costs, and
//!    renders a PlanTree against the SF-0.0001 stub catalog without
//!    error.
//! 2. `cross_query_plan_cache_hit_rate` — running each IS query
//!    twice in succession against a per-engine `PlanCache` produces
//!    cache hits on the second pass. Pinned by reading
//!    [`PlanCache::hit_count`] / [`PlanCache::miss_count`] (W13γ
//!    fix-up MED-2 closure — byte-equality of plan-tree renderings
//!    PASSES on cache miss too because plan-build is deterministic;
//!    the counter assertion is the load-bearing oracle).
//! 3. `multi_tenant_ldbc_isolation` — the same IS query against two
//!    different tenants produces tenant-scoped PlanTrees that do
//!    not share cache entries. Pinned by miss_count delta across
//!    tenant-A → tenant-B sequence (W13γ fix-up MED-3 closure —
//!    tenant-keyed cache enforces per-tenant LRU isolation per
//!    ADR-038 amendment-03 §TIER-2-a).

use std::sync::Arc;

use arcgraph_core::TenantId;
use arcgraph_query::{PlanCache, QueryEngine};

mod common;
use common::ldbc_fixture;

// =====================================================================
// 1. LDBC harness scaffolding — every IS1..IS7 query plans clean
// =====================================================================

#[test]
fn ldbc_harness_scaffolding() {
    let cat = ldbc_fixture::catalog_sf_0_0001();
    let engine = QueryEngine::new(&cat);
    for (name, q) in ldbc_fixture::ALL_IS_QUERIES.iter() {
        let plan_tree = engine.explain(q).unwrap_or_else(|e| {
            panic!("LDBC {name} ({q}) failed to plan: {e:?}");
        });
        // PlanTree always renders to a non-empty Display form (the
        // root node carries the query's outermost operator). The pin
        // is "no panic, no error, plan tree non-empty".
        let rendered = format!("{plan_tree}");
        assert!(
            !rendered.is_empty(),
            "LDBC {name}: PlanTree must render non-empty, got `{rendered}`"
        );
    }
}

// =====================================================================
// 2. Cross-query plan-cache hit rate (consumes W12β PlanCache surface)
// =====================================================================

#[test]
fn cross_query_plan_cache_hit_rate() {
    // Per W13γ spawn prompt: "consume W12β PlanCache + post-merge
    // surface from PR #278". The pin: running each IS query twice
    // through one shared `QueryEngine::with_cache(...)` engine
    // populates the cache on the first run and short-circuits on the
    // second.
    //
    // # W13γ fix-up MED-2 (closes review-pr-285-final.md MED-2)
    //
    // Earlier draft asserted byte-equality of plan-tree renderings on
    // both passes. Plan-build is DETERMINISTIC for the same `(tenant,
    // stmt)` pair under the same `commits_observed` snapshot — the
    // byte-equality oracle PASSES even on cache MISS. The brief
    // mandated "hit rate metric is observable" which means a counter,
    // not a renderings-equal oracle. This rewrite reads
    // `PlanCache::hit_count` / `PlanCache::miss_count` directly + pins
    // the strong invariant: first pass = N misses + 0 hits; second
    // pass = N hits + 0 new misses.
    let cat = ldbc_fixture::catalog_sf_0_0001();
    let cache = Arc::new(PlanCache::new());
    let engine = QueryEngine::new(&cat).with_cache(Arc::clone(&cache));

    // Counters start at 0.
    assert_eq!(cache.hit_count(), 0, "fresh cache: hit_count = 0");
    assert_eq!(cache.miss_count(), 0, "fresh cache: miss_count = 0");

    let n = ldbc_fixture::ALL_IS_QUERIES.len() as u64;

    // First pass — every IS query is a cache MISS at the
    // `PlanCacheKey::from_ast(tenant, stmt)` lookup site, falls
    // through to the cold path (lower → enumerate → cost), and
    // populates the cache.
    for (name, q) in ldbc_fixture::ALL_IS_QUERIES.iter() {
        let _plan_tree = engine.explain(q).unwrap_or_else(|e| {
            panic!("LDBC {name} EXPLAIN cold-path failed: {e:?}");
        });
    }
    assert_eq!(
        cache.miss_count(),
        n,
        "first pass: every IS query is a cache miss (cold path populates cache)"
    );
    assert_eq!(
        cache.hit_count(),
        0,
        "first pass: no hits possible (every key is fresh)"
    );

    // Second pass — every IS query is a cache HIT (the SF-0.0001
    // catalog's `commits_observed` watermark has not advanced between
    // passes; the M4-53 stamped watermark equals the live snapshot).
    for (name, q) in ldbc_fixture::ALL_IS_QUERIES.iter() {
        let _plan_tree = engine.explain(q).unwrap_or_else(|e| {
            panic!("LDBC {name} EXPLAIN hot-path failed: {e:?}");
        });
    }
    assert_eq!(
        cache.hit_count(),
        n,
        "second pass: every IS query short-circuits the cold path \
         (load-bearing hit-rate pin per ADR-038 amendment-03 §TIER-2-a)"
    );
    assert_eq!(
        cache.miss_count(),
        n,
        "second pass: no NEW misses — the cache invariant requires \
         every fresh-watermark lookup against a populated key to return Hit"
    );

    // Structural pin: cache is attached, accessible via Arc.
    assert!(
        engine.cache().is_some(),
        "QueryEngine::with_cache wired correctly"
    );
}

// =====================================================================
// 3. Multi-tenant LDBC isolation
// =====================================================================

#[test]
fn multi_tenant_ldbc_isolation() {
    use arcgraph_query::semantic::{CatalogProvider, StubCatalogProvider};

    let tenant_a = TenantId::DEFAULT;
    let tenant_b = TenantId::new(42);

    // Tenant A — full LDBC catalog at SF-0.0001.
    let cat_a = ldbc_fixture::catalog_sf_0_0001();
    assert_eq!(
        cat_a.tenant(),
        tenant_a,
        "fixture default tenant is DEFAULT"
    );

    // Tenant B — same LDBC schema but distinct tenant + zero-data
    // cardinalities (all label_cardinality / rel_type_cardinality
    // omitted so the stub returns 0 — M4-51 cost walker handles
    // that via the `DEFAULT_LABEL_SELECTIVITY` fallback path).
    let cat_b = StubCatalogProvider::new()
        .with_tenant(tenant_b)
        .with_labels(["Person", "Place", "Forum", "Comment"])
        .with_rel_types([
            "KNOWS",
            "IS_LOCATED_IN",
            "LIKES",
            "HAS_CREATOR",
            "HAS_MEMBER",
            "CONTAINER_OF",
            "HAS_MODERATOR",
            "REPLY_OF",
        ])
        .with_properties([
            "id",
            "firstName",
            "lastName",
            "birthday",
            "locationIp",
            "browserUsed",
            "gender",
            "creationDate",
            "name",
            "title",
            "content",
            "type",
        ]);

    let cache = Arc::new(PlanCache::new());
    let engine_a = QueryEngine::new(&cat_a).with_cache(Arc::clone(&cache));
    let engine_b = QueryEngine::new(&cat_b).with_cache(Arc::clone(&cache));

    // Each tenant runs the full IS1..IS7 suite. The cache is shared
    // across both engines (matches the M5-12 multi-engine deployment
    // pattern); per ADR-038 amendment-03 §TIER-2-a the cache enforces
    // per-tenant LRU isolation, so tenant A's entries never collide
    // with tenant B's.
    //
    // # W13γ fix-up MED-3 (closes review-pr-285-final.md MED-3)
    //
    // Earlier draft only asserted both engines produce non-empty
    // PlanTree renderings — necessary but not sufficient (a regression
    // that key-collides across tenants would still produce renderable
    // plans). The load-bearing isolation pin is "tenant-keyed cache
    // entries do not satisfy each other": after tenant A populates
    // every IS query, tenant B's lookups must MISS because the cache
    // keys carry the tenant. We assert this via the miss_count delta
    // (W13γ fix-up MED-2 counter).
    let n = ldbc_fixture::ALL_IS_QUERIES.len() as u64;

    // Tenant A first pass — populates the cache with tenant-A-keyed
    // entries.
    for (name, q) in ldbc_fixture::ALL_IS_QUERIES.iter() {
        let plan_a = engine_a
            .explain(q)
            .unwrap_or_else(|e| panic!("LDBC {name} tenant A EXPLAIN: {e:?}"));
        let r_a = format!("{plan_a}");
        assert!(
            !r_a.is_empty(),
            "LDBC {name}: tenant A must produce non-empty PlanTree"
        );
    }
    let miss_a = cache.miss_count();
    assert_eq!(
        miss_a, n,
        "tenant A's first pass: every IS query is a cache miss"
    );

    // Tenant B first pass — must ALSO miss every IS query because the
    // cache is tenant-keyed (PlanCacheKey::from_ast(tenant, stmt)).
    // If a regression collapsed the tenant-key into the canonical
    // bytestream, tenant B would HIT tenant A's cached entries here +
    // miss_count would not advance.
    for (name, q) in ldbc_fixture::ALL_IS_QUERIES.iter() {
        let plan_b = engine_b
            .explain(q)
            .unwrap_or_else(|e| panic!("LDBC {name} tenant B EXPLAIN: {e:?}"));
        let r_b = format!("{plan_b}");
        assert!(
            !r_b.is_empty(),
            "LDBC {name}: tenant B must produce non-empty PlanTree"
        );
    }
    let miss_b = cache.miss_count();
    assert_eq!(
        miss_b - miss_a,
        n,
        "tenant B's lookups must MISS — tenant-keyed cache enforces \
         per-tenant LRU isolation per ADR-038 amendment-03 §TIER-2-a; \
         tenant A's hot entries must NOT satisfy tenant B (a regression \
         that key-collides across tenants would manifest as miss_b - \
         miss_a < n)"
    );

    // Per-tenant entry-count pin: each tenant has exactly N entries.
    assert_eq!(
        cache.len_for(tenant_a),
        ldbc_fixture::ALL_IS_QUERIES.len(),
        "tenant A's cache must hold N IS-query entries"
    );
    assert_eq!(
        cache.len_for(tenant_b),
        ldbc_fixture::ALL_IS_QUERIES.len(),
        "tenant B's cache must hold N IS-query entries (disjoint from tenant A)"
    );
}
