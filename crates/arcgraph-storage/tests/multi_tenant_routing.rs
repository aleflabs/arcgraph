//! M3.a Slice H — boundary pins for the [`MultiTenantRouter`]
//! per ADR-037.
//!
//! Path A boundary discipline: every assertion targets a public
//! surface (`MultiTenantRouter::{new, route, tenants}` +
//! `TenantHandle::{tenant, partition, crud, vector}`). Tests in
//! this file MUST NOT reach into router internals.
//!
//! # Pin map
//!
//! 1. `routing_basic_dispatch_per_tenant` — DEFAULT routes
//!    successfully and the returned handle reports the right
//!    `(tenant, partition)`.
//!
//! 2. `routing_tenant_isolation_no_cross_tenant_leakage` — the
//!    I-V2 regression guard. DEFAULT and SYSTEM share the same
//!    underlying `CrudStore` Arc; per-tenant counter keying
//!    inside `CrudStore` preserves isolation under
//!    `alloc_node`.
//!
//! 3. `routing_partition_id_zero_only_at_v1` — non-zero
//!    `PartitionId` surfaces
//!    `RoutingError::PartitionNotSupported`.
//!
//! 4. `routing_unknown_tenant_returns_error` — an unknown
//!    `TenantId` (not in catalog, not SYSTEM) surfaces
//!    `RoutingError::UnknownTenant` with the raw u64.
//!
//! 5. `routing_handle_cache_consistency` — repeat calls return
//!    the SAME `Arc<TenantHandle>` (v1.0 append-only cache per
//!    ADR-037 §D-6; cache invalidation is a v1.1 lift).
//!
//! 6. `routing_concurrent_route_calls_safe` — 4 OS threads,
//!    100 calls each, all returned handles share an Arc
//!    pointer (no torn handles under contention; DashMap's
//!    per-bucket lock is the only synchronisation).
//!
//! 7. `routing_partition_signature` — compile-time API-shape pin:
//!    `route` accepts `(TenantId, PartitionId)`.
//!
//! 8. `routing_handle_lifecycle` — the cache holds a strong
//!    `Arc<TenantHandle>`; consumer drops do not invalidate
//!    the cache, and a third route returns the SAME Arc as
//!    the first two.
//!
//! 9. `routing_with_durability_tier_dispatch` — handle
//!    composes with the existing per-tenant durability tier
//!    mechanism (ADR-034). Reads via
//!    `catalog.durability_tier(handle.tenant())` reflect the
//!    tier the catalog stores for the routed tenant.
//!
//! 10. `router_community_handle_plumbed_through_when_some` —
//!     M3.d-1 #2 (ADR-040 §D-3 + ADR-037 §D-1). When a
//!     `CommunityIndexProvider` is wired via the additive
//!     `MultiTenantRouter::with_community(...)` constructor,
//!     `TenantHandle::community()` returns the per-tenant
//!     handle the provider yields; when no provider is wired
//!     (the existing 3-arg `MultiTenantRouter::new(...)` path),
//!     `community()` is `None`. Symmetric to test 8's
//!     `vector_handle_plumbed_through_when_some` in router.rs.
//!
//! 13. `router_builder_full_surface_matches_consolidating_constructor`
//!     — ADR-037 amendment-01. The builder pattern is the canonical
//!     extension point for v1.1+ optional deps. Pin: a router built
//!     via `MultiTenantRouter::builder(...)` with all three optional
//!     deps wired is plumbing-equivalent to the 5-arg consolidating
//!     constructor.
//!
//! 14. `router_all_three_deps_compose_through_builder` — retained-index
//!     composability. Pin: a router built via
//!     [`MultiTenantRouter::builder(...)`] with all three optional deps
//!     (vector + community + bm25) wired returns a [`TenantHandle`]
//!     whose three accessors (`vector()`, `community()`, `bm25()`) all
//!     surface `Some(_)`. Each
//!     accessor returns the SAME `Arc` on repeated calls (handle_cache
//!     discipline per ADR-037 §D-2).
//!
//! 15. `router_per_tenant_handle_scoping_via_provider_facade` —
//!     facade-plumbing pin. Two-tenant fixture
//!     (DEFAULT + SYSTEM) where each tenant's community provider
//!     returns a `CommunityIndexHandle` whose `tenant()` matches the
//!     caller's tenant — verifies the router passes the caller's
//!     `TenantId` unchanged to the provider:
//!     this pin does NOT verify cross-tenant data isolation (the
//!     `StubProvider` unconditionally returns a handle for whichever
//!     tenant is asked).
//!
//! Per ADR-037 §D-1 / §D-2 / §D-3 / §D-4 / §D-5 / §D-6 +
//! ADR-040 §D-3 (pin #10) + ADR-037 amendment-01 (pin #13).

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use arcgraph_core::{DurabilityTier, PartitionId, TenantId};
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::catalog::SystemCatalog;
use arcgraph_storage::crud::CrudStore;
use arcgraph_storage::io::InMemoryPageIo;
use arcgraph_storage::router::{MultiTenantRouter, RoutingError};
use arcgraph_storage::transaction::TxnManager;

// ─────────────────────────────────────────────────────────────────────
// Test harness — mirrors `tests/durability_tier_strict.rs::make_setup`
// and `crates/arcgraph-storage/src/catalog.rs::tests::make_deps`.
// All tests in this file build a fresh BufferPool / TxnManager /
// SystemCatalog / CrudStore via this helper so each test is
// fully isolated.
// ─────────────────────────────────────────────────────────────────────

/// Build a freshly-bootstrapped router with no vector store
/// wired. Every test that doesn't need a vector arena calls this.
fn make_router() -> MultiTenantRouter {
    let io = Arc::new(InMemoryPageIo::new());
    let pool = BufferPool::new(8, io);
    let mgr = TxnManager::new();
    let catalog = Arc::new(SystemCatalog::new());
    catalog.bootstrap(&pool, &mgr).expect("bootstrap");
    let crud = Arc::new(CrudStore::new());
    MultiTenantRouter::new(catalog, crud, None)
}

/// Build a router AND surface its catalog so a test can
/// exercise tier-aware composition (test 9). Returns
/// `(router, catalog)` so the caller can keep the catalog Arc
/// for tier reads.
fn make_router_with_catalog() -> (MultiTenantRouter, Arc<SystemCatalog>) {
    let io = Arc::new(InMemoryPageIo::new());
    let pool = BufferPool::new(8, io);
    let mgr = TxnManager::new();
    let catalog = Arc::new(SystemCatalog::new());
    catalog.bootstrap(&pool, &mgr).expect("bootstrap");
    let crud = Arc::new(CrudStore::new());
    let router = MultiTenantRouter::new(Arc::clone(&catalog), crud, None);
    (router, catalog)
}

// ─────────────────────────────────────────────────────────────────────
// Test 1 — basic dispatch
// ─────────────────────────────────────────────────────────────────────

#[test]
fn routing_basic_dispatch_per_tenant() {
    // ADR-037 §D-2: a successful `route(DEFAULT, ZERO)` call
    // returns an `Arc<TenantHandle>` reporting the routing key
    // it was constructed for. Note: at v1.0 the catalog only
    // surfaces `DEFAULT` from `list_tenants` — there is no
    // public registration path for additional user tenants
    // (per `crates/arcgraph-storage/tests/durability_tier_mixed.rs`
    // lines 84-95). Multi-user-tenant exercising waits for the
    // M7 lifecycle slice; the routing logic is the same
    // regardless of tenant count, and the SYSTEM-tenant
    // carve-out (test 2) gives a second tenant id to exercise
    // dispatch against today.
    let router = make_router();

    let handle = router
        .route(TenantId::DEFAULT, PartitionId::ZERO)
        .expect("DEFAULT must route");

    assert_eq!(
        handle.tenant(),
        TenantId::DEFAULT,
        "handle reports the tenant it was routed for"
    );
    assert_eq!(
        handle.partition(),
        PartitionId::ZERO,
        "handle reports the partition it was routed for"
    );

    // Cache hit: re-routing the same key returns the SAME Arc.
    let handle2 = router
        .route(TenantId::DEFAULT, PartitionId::ZERO)
        .expect("re-route");
    assert!(
        Arc::ptr_eq(&handle, &handle2),
        "second route is a cache hit (same Arc)"
    );
}

// ─────────────────────────────────────────────────────────────────────
// Test 2 — tenant isolation regression guard (I-V2)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn routing_tenant_isolation_no_cross_tenant_leakage() {
    // ADR-037 §D-1: `crud` is `Arc<CrudStore>` shared across
    // every handle the router materialises. Per-tenant
    // isolation lives inside `CrudStore`'s
    // `(TenantId, …)`-keyed maps. This test pins that the
    // shared Arc does NOT smear allocator state across
    // tenants.
    //
    // We use SYSTEM (allowed via §D-5 carve-out) and DEFAULT
    // as our two tenant ids; allocations under one tenant must
    // NOT advance the other tenant's high-water counter.
    let router = make_router();

    let h_default = router
        .route(TenantId::DEFAULT, PartitionId::ZERO)
        .expect("DEFAULT");
    let h_system = router
        .route(TenantId::SYSTEM, PartitionId::ZERO)
        .expect("SYSTEM (ADR-037 §D-5 carve-out)");

    // Distinct routing keys.
    assert_ne!(
        h_default.tenant(),
        h_system.tenant(),
        "DEFAULT and SYSTEM are different tenants"
    );

    // Both handles share the SAME underlying CrudStore Arc:
    // routing is a facade, not per-tenant store creation.
    assert!(
        Arc::ptr_eq(h_default.crud(), h_system.crud()),
        "TenantHandle::crud() returns the SAME Arc across tenants \
         (per-tenant projection lives inside CrudStore, not in the router)"
    );

    let crud = h_default.crud();

    // Pristine: both counters are zero.
    assert_eq!(crud.node_high_water(TenantId::DEFAULT), 0);
    assert_eq!(crud.node_high_water(TenantId::SYSTEM), 0);

    // Allocate a node under DEFAULT (using the handle's
    // tenant identity, mirroring how M4 / M5 consumers will
    // call into CrudStore).
    let _node = crud
        .alloc_node(h_default.tenant())
        .expect("alloc under DEFAULT");

    // Isolation: DEFAULT advanced; SYSTEM still 0.
    assert_eq!(
        crud.node_high_water(TenantId::DEFAULT),
        1,
        "DEFAULT's counter advanced to 1 after one alloc"
    );
    assert_eq!(
        crud.node_high_water(TenantId::SYSTEM),
        0,
        "SYSTEM's counter MUST NOT leak DEFAULT's advance — I-V2 regression guard"
    );

    // Symmetric direction: alloc under SYSTEM does not move
    // DEFAULT.
    let _node_sys = crud
        .alloc_node(h_system.tenant())
        .expect("alloc under SYSTEM");
    assert_eq!(crud.node_high_water(TenantId::DEFAULT), 1);
    assert_eq!(crud.node_high_water(TenantId::SYSTEM), 1);
}

// ─────────────────────────────────────────────────────────────────────
// Test 3 — non-zero PartitionId surfaces a typed error
// ─────────────────────────────────────────────────────────────────────

#[test]
fn routing_rejects_nonzero_partition_ids() {
    // The public local engine accepts only PartitionId::ZERO. Any other
    // value surfaces RoutingError::PartitionNotSupported carrying the raw
    // u32 for diagnosability.
    let router = make_router();
    let err = router
        .route(TenantId::DEFAULT, PartitionId::new(1))
        .expect_err("non-zero partition must fail");

    match err {
        RoutingError::PartitionNotSupported { partition_raw } => {
            assert_eq!(partition_raw, 1, "raw partition value carried in the error");
        }
        other => panic!("expected PartitionNotSupported, got {other:?}"),
    }

    // Pin the error message to the local-engine constraint without
    // promising a future distributed shape.
    let err2 = router
        .route(TenantId::DEFAULT, PartitionId::new(7))
        .expect_err("non-zero partition (value 7) must fail");
    let msg = format!("{err2}");
    assert!(
        msg.contains("partition_id=7"),
        "error message embeds the raw partition id: {msg}"
    );
    assert!(msg.contains("PartitionId::ZERO"), "local constraint: {msg}");
    assert!(
        !msg.contains("v1.0"),
        "must not promise a future shape: {msg}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// Test 4 — unknown TenantId surfaces a typed error
// ─────────────────────────────────────────────────────────────────────

#[test]
fn routing_unknown_tenant_returns_error() {
    // ADR-037 §D-3. A tenant not in the catalog and not SYSTEM
    // surfaces RoutingError::UnknownTenant with the raw u64.
    // ID 9999 is in the user-DDL range (100+), so it's not a
    // reserved built-in.
    let router = make_router();
    let err = router
        .route(TenantId::new(9999), PartitionId::ZERO)
        .expect_err("unknown tenant must fail");

    match err {
        RoutingError::UnknownTenant { tenant_raw } => {
            assert_eq!(tenant_raw, 9999, "raw tenant value carried in the error");
        }
        other => panic!("expected UnknownTenant, got {other:?}"),
    }

    // Reserved ID range 2..=99 also surfaces UnknownTenant at
    // v1.0 (per ADR-037 §D-5 final paragraph): the catalog
    // does not pre-register that range, so dispatch fails
    // unless a future built-in tenant earns its own carve-out.
    let err_reserved = router
        .route(TenantId::new(50), PartitionId::ZERO)
        .expect_err("reserved-range tenant must fail at v1.0");
    assert!(matches!(
        err_reserved,
        RoutingError::UnknownTenant { tenant_raw: 50 }
    ));
}

// ─────────────────────────────────────────────────────────────────────
// Test 5 — append-only cache consistency
// ─────────────────────────────────────────────────────────────────────

#[test]
fn routing_handle_cache_consistency() {
    // ADR-037 §D-6: at v1.0 the cache is APPEND-ONLY. Once a
    // (tenant, partition) entry is materialised it lives until
    // the router is dropped. This test pins the cache HIT path
    // — the same Arc is returned across repeated calls. The
    // directive's "catalog mutations invalidate the cache"
    // line is a v1.1 lift; at v1.0 there is no such mutation.
    let router = make_router();

    let h1 = router
        .route(TenantId::DEFAULT, PartitionId::ZERO)
        .expect("first");
    let h2 = router
        .route(TenantId::DEFAULT, PartitionId::ZERO)
        .expect("second");
    let h3 = router
        .route(TenantId::DEFAULT, PartitionId::ZERO)
        .expect("third");

    assert!(
        Arc::ptr_eq(&h1, &h2),
        "v1.0 cache hit: 1st and 2nd routes share the Arc"
    );
    assert!(
        Arc::ptr_eq(&h2, &h3),
        "v1.0 cache hit: 2nd and 3rd routes share the Arc"
    );

    // Distinct tenant -> distinct Arc.
    let h_sys = router
        .route(TenantId::SYSTEM, PartitionId::ZERO)
        .expect("SYSTEM");
    assert!(
        !Arc::ptr_eq(&h1, &h_sys),
        "different tenant materialises a different cache entry"
    );

    // Distinct call after SYSTEM -> still cached -> same as h1.
    let h4 = router
        .route(TenantId::DEFAULT, PartitionId::ZERO)
        .expect("fourth");
    assert!(
        Arc::ptr_eq(&h1, &h4),
        "DEFAULT's cache entry survives an interleaved SYSTEM route"
    );
}

// ─────────────────────────────────────────────────────────────────────
// Test 6 — concurrent route() calls share the cached Arc
// ─────────────────────────────────────────────────────────────────────

#[test]
fn routing_concurrent_route_calls_safe() {
    // ADR-037 §D-2 final paragraph: DashMap's
    // `entry().or_insert_with(...)` is the single
    // synchronisation primitive in `route()`. This test pins
    // that 4 OS threads, each calling `route(DEFAULT, ZERO)`
    // 100 times, all return the SAME Arc — no torn handles, no
    // duplicate materialisations.
    //
    // We use `std::thread::scope` for borrow-checked
    // concurrent access (no need for Arc<Router> in the test
    // body itself).
    let router = make_router();

    // First materialisation under the main thread so the
    // child threads always observe a CACHE HIT — the router
    // must still serve the same Arc from concurrent reads.
    let canonical = router
        .route(TenantId::DEFAULT, PartitionId::ZERO)
        .expect("seed cache");

    let threads = 4usize;
    let calls_per_thread = 100usize;
    let total = AtomicUsize::new(0);

    thread::scope(|s| {
        for _ in 0..threads {
            let canonical_ref = &canonical;
            let router_ref = &router;
            let total_ref = &total;
            s.spawn(move || {
                for _ in 0..calls_per_thread {
                    let h = router_ref
                        .route(TenantId::DEFAULT, PartitionId::ZERO)
                        .expect("route under contention");
                    assert!(
                        Arc::ptr_eq(&h, canonical_ref),
                        "concurrent route MUST yield the same Arc as the canonical seed"
                    );
                    total_ref.fetch_add(1, Ordering::Relaxed);
                }
            });
        }
    });

    assert_eq!(
        total.load(Ordering::Relaxed),
        threads * calls_per_thread,
        "every thread completed every call"
    );
}

// ─────────────────────────────────────────────────────────────────────
// Test 7 — API surface pins PartitionId
// ─────────────────────────────────────────────────────────────────────

#[test]
fn routing_partition_signature() {
    // The `route` signature accepts `(TenantId, PartitionId)`.
    // This test is an API-shape pin: it constructs a
    // PartitionId via the public `new` constructor (the same
    // public surface), routes against ZERO, and binds the return type as
    // `Result<Arc<TenantHandle>, RoutingError>`.
    //
    // If a future refactor changes the signature shape, this
    // test fails to compile — the explicit type ascription on
    // the `let` is the load-bearing assertion.
    let router = make_router();
    let p = PartitionId::new(0);
    let result: Result<Arc<arcgraph_storage::TenantHandle>, RoutingError> =
        router.route(TenantId::DEFAULT, p);

    let handle = result.expect("ZERO partition routes today");
    assert_eq!(handle.partition(), PartitionId::ZERO);
    assert_eq!(handle.tenant(), TenantId::DEFAULT);

    // Sanity: PartitionId::new(0) is byte-identical with
    // PartitionId::ZERO. v1.1 lift will accept other values
    // here without changing this test's structure.
    assert_eq!(p, PartitionId::ZERO);
}

// ─────────────────────────────────────────────────────────────────────
// Test 8 — handle lifecycle: cache holds a strong Arc
// ─────────────────────────────────────────────────────────────────────

#[test]
fn routing_handle_lifecycle() {
    // ADR-037 §D-6: the v1.0 cache holds a strong
    // `Arc<TenantHandle>`. Consumer drops do NOT invalidate
    // the cache, and a subsequent route returns the same Arc
    // as the first.
    let router = make_router();

    let h1 = router
        .route(TenantId::DEFAULT, PartitionId::ZERO)
        .expect("first");
    let h2 = router
        .route(TenantId::DEFAULT, PartitionId::ZERO)
        .expect("second");
    assert!(Arc::ptr_eq(&h1, &h2), "h1 and h2 are the same Arc");

    // Capture the raw pointer before dropping h1 so we can
    // compare against future routes without holding h1
    // itself. `Arc::as_ptr` returns a raw pointer that aliases
    // the cached entry; we don't dereference it, only compare.
    let h1_ptr = Arc::as_ptr(&h1);

    drop(h1);
    // h2 still works after h1 is dropped: the strong refcount
    // is held by both h2 AND the cache.
    assert_eq!(h2.tenant(), TenantId::DEFAULT);
    assert_eq!(h2.partition(), PartitionId::ZERO);

    drop(h2);
    // The cache still holds a strong reference; a third route
    // returns the SAME Arc (same raw pointer as h1's first
    // materialisation).
    let h3 = router
        .route(TenantId::DEFAULT, PartitionId::ZERO)
        .expect("third after drops");
    let h3_ptr = Arc::as_ptr(&h3);
    assert_eq!(
        h1_ptr, h3_ptr,
        "cache survives consumer drops at v1.0 (append-only per ADR-037 §D-6)"
    );
    assert_eq!(h3.tenant(), TenantId::DEFAULT);
}

// ─────────────────────────────────────────────────────────────────────
// Test 9 — TenantHandle composes with ADR-034 durability tiers
// ─────────────────────────────────────────────────────────────────────

#[test]
fn routing_with_durability_tier_dispatch() {
    // ADR-037 §D-5 closing paragraph: the router does NOT make
    // a tier claim — tier reads / writes happen via the
    // catalog, which is exactly what M4 / M5 consumers will
    // do. This test routes for DEFAULT and reads its tier via
    // the catalog using `handle.tenant()`, proving the handle
    // composes with the existing per-tenant durability
    // mechanism.
    let (router, catalog) = make_router_with_catalog();

    let handle = router
        .route(TenantId::DEFAULT, PartitionId::ZERO)
        .expect("DEFAULT routes");

    // Bootstrap installs DEFAULT at Strict (catalog.rs:160-162
    // + ADR-034 §I-D7).
    let tier = catalog.durability_tier(handle.tenant());
    assert_eq!(
        tier,
        DurabilityTier::Strict,
        "DEFAULT bootstraps as Strict; routing-layer composition reflects the catalog"
    );

    // SYSTEM is T1-enforced regardless of catalog state
    // (catalog.rs:206-208). Routing through SYSTEM and reading
    // the tier yields Strict — the routing facade does not
    // alter tier semantics.
    let h_sys = router
        .route(TenantId::SYSTEM, PartitionId::ZERO)
        .expect("SYSTEM routes (carve-out)");
    let tier_sys = catalog.durability_tier(h_sys.tenant());
    assert_eq!(
        tier_sys,
        DurabilityTier::Strict,
        "SYSTEM is always Strict (catalog.rs:206-208 + ADR-034 §I-D7)"
    );
}

// ─────────────────────────────────────────────────────────────────────
// Pin #10 — community handle plumbing (ADR-040 §D-3 + ADR-037 §D-1).
// ─────────────────────────────────────────────────────────────────────
//
// M3.d-1 #2: when a `CommunityIndexProvider` is wired into the
// router via the additive `MultiTenantRouter::with_community(...)`
// constructor, the `TenantHandle::community()` accessor returns
// the per-tenant handle the provider yields; when no provider is
// wired (the existing 3-arg `MultiTenantRouter::new(...)` path),
// `community()` is `None`.
//
// This is the symmetric routing pin to test 8's
// `vector_handle_plumbed_through_when_some` (in router.rs's
// in-module unit tests). The community case differs from vector
// in two ways:
//
// 1. `CommunityIndexHandle` carries `tenant_id` baked-in (ADR-040
//    §D-3), so the router cannot share a single
//    `Arc<CommunityIndexHandle>` across tenants the way it
//    shares `Arc<dyn VectorPageStoreHandle>`. The provider trait
//    is the per-tenant factory.
// 2. The `MultiTenantRouter::new(...)` 3-arg signature is
//    UNCHANGED at v1.0; the new `with_community(...)` 4-arg
//    constructor is the entry point for community-aware
//    wirings. Existing call sites in arcgraph-storage and the
//    single-partition routing conformance pin
//    are not touched by this commit.

/// Stub membership index for the test fixture. The default
/// `MembershipIndex` trait methods all `unimplemented!()`, but
/// pin #10 exercises only `TenantHandle::community()` plumbing —
/// no membership trait method is ever called.
struct StubMembershipIndex;
impl arcgraph_community::MembershipIndex for StubMembershipIndex {}

/// Stub provider that returns a fresh handle for ANY tenant the
/// router asks about. The handle's tenant_id is the requested
/// tenant — pinning per-tenant isolation discipline at the
/// provider boundary (ADR-040 §D-3).
struct StubProvider;
impl arcgraph_community::CommunityIndexProvider for StubProvider {
    fn handle_for(
        &self,
        tenant: TenantId,
        _partition: PartitionId,
    ) -> Option<Arc<arcgraph_community::CommunityIndexHandle>> {
        Some(Arc::new(
            arcgraph_community::CommunityIndexHandle::for_tenant(
                tenant,
                arcgraph_community::CommunityIndexId::new(1),
                Arc::new(StubMembershipIndex),
            ),
        ))
    }
}

#[test]
fn router_community_handle_plumbed_through_when_some() {
    // SOME path: a router built via `with_community(...)` plumbs
    // the provider's per-tenant handle through `TenantHandle::community()`.
    let io = Arc::new(InMemoryPageIo::new());
    let pool = BufferPool::new(8, io);
    let mgr = TxnManager::new();
    let catalog = Arc::new(SystemCatalog::new());
    catalog.bootstrap(&pool, &mgr).expect("bootstrap");
    let crud = Arc::new(CrudStore::new());
    let provider: Arc<dyn arcgraph_community::CommunityIndexProvider> = Arc::new(StubProvider);

    let router = MultiTenantRouter::with_community(
        catalog,
        crud,
        None, // vector
        Some(Arc::clone(&provider)),
    );

    let h = router
        .route(TenantId::DEFAULT, PartitionId::ZERO)
        .expect("DEFAULT routes");
    let community = h
        .community()
        .expect("provider returned Some — community handle plumbed through");
    assert_eq!(
        community.tenant(),
        TenantId::DEFAULT,
        "provider must scope handle to caller's tenant (ADR-040 §D-3)",
    );
    assert_eq!(community.partition(), PartitionId::ZERO);

    // NONE path: the existing 3-arg `MultiTenantRouter::new(...)`
    // delegates to `with_community(.., None)`. A router built via
    // that path returns `None` from `community()` for every
    // tenant — proving `new` is the no-provider default.
    let router_none = make_router();
    let h_none = router_none
        .route(TenantId::DEFAULT, PartitionId::ZERO)
        .expect("DEFAULT routes");
    assert!(
        h_none.community().is_none(),
        "router built via 3-arg `new(...)` has no community provider"
    );
}

// ─────────────────────────────────────────────────────────────────────
// Pin #11 — BM25 handle plumbing (ADR-039 §D-9 + ADR-037 §3.3).
// ─────────────────────────────────────────────────────────────────────
//
// M3.b: when a `Bm25IndexStoreHandle` is wired into the router via
// the additive `MultiTenantRouter::new_with_bm25(...)` constructor,
// the `TenantHandle::bm25()` accessor returns the same Arc the
// router was constructed with; when the BM25 handle is `None` (the
// existing 3-arg `MultiTenantRouter::new(...)` path), `bm25()` is
// `None`.
//
// Symmetric to pin #10 above. Mirrors the
// `bm25_handle_plumbed_through_when_some` in-module unit test in
// `router.rs`, lifted to the integration boundary so the public
// surface (visible to M4 / M5 consumers) is exercised end-to-end.

/// Local mock `Bm25IndexStoreHandle` for pins #11 / #12. Mirrors the
/// `NoopBm25Store` in `router.rs::tests` — kept local here so the
/// integration test does not reach into the storage crate's private
/// `tests` module.
struct LocalNoopBm25Store;

impl arcgraph_storage::mutation_log::Bm25IndexStoreHandle for LocalNoopBm25Store {
    fn commit_pending(
        &self,
        _tenant: TenantId,
    ) -> Result<(), arcgraph_storage::mutation_log::Bm25StoreError> {
        Ok(())
    }

    fn rollback_pending(
        &self,
        _tenant: TenantId,
    ) -> Result<(), arcgraph_storage::mutation_log::Bm25StoreError> {
        Ok(())
    }
}

#[test]
fn router_bm25_handle_plumbed_through_when_some() {
    // SOME path: a router built via `new_with_bm25(...)` plumbs the
    // same `Arc<dyn Bm25IndexStoreHandle>` through every materialised
    // `TenantHandle::bm25()`.
    let io = Arc::new(InMemoryPageIo::new());
    let pool = BufferPool::new(8, io);
    let mgr = TxnManager::new();
    let catalog = Arc::new(SystemCatalog::new());
    catalog.bootstrap(&pool, &mgr).expect("bootstrap");
    let crud = Arc::new(CrudStore::new());
    let bm25: Arc<dyn arcgraph_storage::mutation_log::Bm25IndexStoreHandle> =
        Arc::new(LocalNoopBm25Store);

    let router = MultiTenantRouter::new_with_bm25(catalog, crud, None, Some(Arc::clone(&bm25)));

    let h = router
        .route(TenantId::DEFAULT, PartitionId::ZERO)
        .expect("DEFAULT routes");
    let h_bm25 = h.bm25().expect("bm25 plumbed through");
    assert!(
        Arc::ptr_eq(h_bm25, &bm25),
        "router must clone the SAME Arc into the handle (ADR-039 §D-9)"
    );

    // NONE path: the existing 3-arg `MultiTenantRouter::new(...)`
    // delegates to `new_with_bm25(.., None)`. A router built via
    // that path returns `None` from `bm25()` for every tenant.
    let router_none = make_router();
    let h_none = router_none
        .route(TenantId::DEFAULT, PartitionId::ZERO)
        .expect("DEFAULT routes");
    assert!(
        h_none.bm25().is_none(),
        "router built via 3-arg `new(...)` has no bm25 handle"
    );
}

// ─────────────────────────────────────────────────────────────────────
// Pin #12 — community + bm25 composability (post-#148 rebase).
// ─────────────────────────────────────────────────────────────────────
//
// PIN: post-#148 rebase composability — both optional deps can
// be wired together via the consolidating
// `new_with_community_and_bm25(...)` constructor without conflict.
// Replaces the otherwise implicit guarantee that `with_community`
// and `new_with_bm25` can co-exist.
//
// Per the rebase resolution: all four constructors thin-delegate
// to `new_with_community_and_bm25`; this test exercises the full
// 5-arg surface so that a future builder-refactor PR (TODO at
// `new_with_community_and_bm25`'s rustdoc) can be validated
// against the same plumbing semantics.

#[test]
fn router_community_and_bm25_handles_both_plumbed_through() {
    let io = Arc::new(InMemoryPageIo::new());
    let pool = BufferPool::new(8, io);
    let mgr = TxnManager::new();
    let catalog = Arc::new(SystemCatalog::new());
    catalog.bootstrap(&pool, &mgr).expect("bootstrap");
    let crud = Arc::new(CrudStore::new());

    // Both optional deps wired (vector left as None for brevity —
    // the vector plumbing pin already lives in router.rs's in-module
    // tests).
    let provider: Arc<dyn arcgraph_community::CommunityIndexProvider> = Arc::new(StubProvider);
    let bm25: Arc<dyn arcgraph_storage::mutation_log::Bm25IndexStoreHandle> =
        Arc::new(LocalNoopBm25Store);

    let router = MultiTenantRouter::new_with_community_and_bm25(
        catalog,
        crud,
        None, // vector
        Some(Arc::clone(&provider)),
        Some(Arc::clone(&bm25)),
    );

    let h = router
        .route(TenantId::DEFAULT, PartitionId::ZERO)
        .expect("DEFAULT routes");

    // Community must be plumbed through.
    let community = h
        .community()
        .expect("community must be plumbed through when both deps wired");
    assert_eq!(
        community.tenant(),
        TenantId::DEFAULT,
        "provider must scope handle to caller's tenant (ADR-040 §D-3)"
    );
    assert_eq!(community.partition(), PartitionId::ZERO);

    // BM25 must be plumbed through with Arc-pointer equality.
    let h_bm25 = h
        .bm25()
        .expect("bm25 must be plumbed through when both deps wired");
    assert!(
        Arc::ptr_eq(h_bm25, &bm25),
        "router must clone the SAME bm25 Arc into the handle (ADR-039 §D-9)"
    );
}

// ─────────────────────────────────────────────────────────────────────
// Pin #13 — builder full-surface ≡ consolidating ctor (ADR-037 amendment-01).
// ─────────────────────────────────────────────────────────────────────
//
// PIN: the builder pattern is the canonical extension point for
// v1.1+ optional deps. This test pins that a router built via
// `MultiTenantRouter::builder(...)` with all three optional deps
// wired (vector, community, bm25) is plumbing-equivalent to the
// 5-arg consolidating constructor — same Arc-pointer equality on
// every materialised TenantHandle accessor.
//
// Per ADR-037 amendment-01: existing constructors are preserved as
// thin-delegates, but new optional deps MUST add a `.fn()` method
// on the builder rather than a new constructor.

#[test]
fn router_builder_full_surface_matches_consolidating_constructor() {
    // Identical fixtures across the two construction paths so any
    // delta surfaces as an Arc-pointer mismatch on the handle.
    let io_a = Arc::new(InMemoryPageIo::new());
    let pool_a = BufferPool::new(8, io_a);
    let mgr_a = TxnManager::new();
    let catalog_a = Arc::new(SystemCatalog::new());
    catalog_a.bootstrap(&pool_a, &mgr_a).expect("bootstrap A");
    let crud_a = Arc::new(CrudStore::new());

    let io_b = Arc::new(InMemoryPageIo::new());
    let pool_b = BufferPool::new(8, io_b);
    let mgr_b = TxnManager::new();
    let catalog_b = Arc::new(SystemCatalog::new());
    catalog_b.bootstrap(&pool_b, &mgr_b).expect("bootstrap B");
    let crud_b = Arc::new(CrudStore::new());

    // Same dep instances feed BOTH construction paths so Arc-pointer
    // equality is meaningful.
    let provider: Arc<dyn arcgraph_community::CommunityIndexProvider> = Arc::new(StubProvider);
    let bm25: Arc<dyn arcgraph_storage::mutation_log::Bm25IndexStoreHandle> =
        Arc::new(LocalNoopBm25Store);

    // Path A: builder.
    let router_builder = MultiTenantRouter::builder(catalog_a, crud_a)
        .community(Arc::clone(&provider))
        .bm25(Arc::clone(&bm25))
        .build();

    // Path B: 5-arg consolidating constructor.
    let router_ctor = MultiTenantRouter::new_with_community_and_bm25(
        catalog_b,
        crud_b,
        None, // vector
        Some(Arc::clone(&provider)),
        Some(Arc::clone(&bm25)),
    );

    // Both routers must surface DEFAULT in `tenants()` (catalog
    // bootstraps DEFAULT only).
    assert_eq!(router_builder.tenants(), router_ctor.tenants());

    // Route under DEFAULT on each.
    let h_builder = router_builder
        .route(TenantId::DEFAULT, PartitionId::ZERO)
        .expect("builder route");
    let h_ctor = router_ctor
        .route(TenantId::DEFAULT, PartitionId::ZERO)
        .expect("ctor route");

    // Identical (tenant, partition) on the materialised handles.
    assert_eq!(h_builder.tenant(), h_ctor.tenant());
    assert_eq!(h_builder.partition(), h_ctor.partition());

    // Builder must wire the SAME bm25 Arc instance.
    let bm25_builder = h_builder.bm25().expect("builder bm25 plumbed");
    let bm25_ctor = h_ctor.bm25().expect("ctor bm25 plumbed");
    assert!(Arc::ptr_eq(bm25_builder, &bm25), "builder bm25 == fixture");
    assert!(Arc::ptr_eq(bm25_ctor, &bm25), "ctor bm25 == fixture");

    // Builder must wire community per the provider — both paths
    // surface a Some(_) handle scoped to DEFAULT.
    let comm_builder = h_builder.community().expect("builder community plumbed");
    let comm_ctor = h_ctor.community().expect("ctor community plumbed");
    assert_eq!(comm_builder.tenant(), TenantId::DEFAULT);
    assert_eq!(comm_ctor.tenant(), TenantId::DEFAULT);

    // No vector wired on either; both surface None.
    assert!(h_builder.vector().is_none());
    assert!(h_ctor.vector().is_none());
}

// ─────────────────────────────────────────────────────────────────────
// Pin #14 — retained-index handle composability.
// ─────────────────────────────────────────────────────────────────────
//
// PIN: a router built via `MultiTenantRouter::builder(...)` with all
// three optional deps (vector + community + bm25) wired returns a
// `TenantHandle` whose three accessors all surface `Some(_)`.
//
// Each accessor returns the SAME `Arc` on repeated calls (router
// `handle_cache` discipline per ADR-037 §D-2 / §D-6).

/// Stub `VectorPageStoreHandle` used by pin #14 to confirm the vector
/// `Some(_)` arm plumbs through alongside community + bm25. Mirrors
/// the in-module `NoopVectorStore` in `router.rs::tests` — kept local
/// here so this integration test does not reach into the storage
/// crate's private `tests` module.
struct LocalNoopVectorStore;

impl arcgraph_storage::vector_store::VectorPageStoreHandle for LocalNoopVectorStore {
    fn install_or_replace(
        &self,
        _tenant: TenantId,
        _page_id: arcgraph_core::PageId,
        _bytes: &[u8],
    ) -> Result<(), arcgraph_storage::vector_store::VectorStoreError> {
        Ok(())
    }

    fn restore_page_bytes(
        &self,
        _tenant: TenantId,
        _page_id: arcgraph_core::PageId,
        _bytes: &[u8],
    ) -> Result<(), arcgraph_storage::vector_store::VectorStoreError> {
        Ok(())
    }
}

#[test]
fn router_all_three_deps_compose_through_builder() {
    // Build a router via `MultiTenantRouter::builder(...)` with all
    // three optional deps wired.
    let io = Arc::new(InMemoryPageIo::new());
    let pool = BufferPool::new(8, io);
    let mgr = TxnManager::new();
    let catalog = Arc::new(SystemCatalog::new());
    catalog.bootstrap(&pool, &mgr).expect("bootstrap");
    let crud = Arc::new(CrudStore::new());

    let vector: Arc<dyn arcgraph_storage::vector_store::VectorPageStoreHandle> =
        Arc::new(LocalNoopVectorStore);
    let provider: Arc<dyn arcgraph_community::CommunityIndexProvider> = Arc::new(StubProvider);
    let bm25: Arc<dyn arcgraph_storage::mutation_log::Bm25IndexStoreHandle> =
        Arc::new(LocalNoopBm25Store);

    let router = MultiTenantRouter::builder(catalog, crud)
        .vector(Arc::clone(&vector))
        .community(Arc::clone(&provider))
        .bm25(Arc::clone(&bm25))
        .build();

    // First route: every accessor must be `Some(_)`.
    let h = router
        .route(TenantId::DEFAULT, PartitionId::ZERO)
        .expect("DEFAULT routes");

    let h_vec = h.vector().expect("vector plumbed through");
    assert!(
        Arc::ptr_eq(h_vec, &vector),
        "vector accessor must return the SAME Arc the builder was given"
    );

    let h_community = h.community().expect("community plumbed through");
    assert_eq!(
        h_community.tenant(),
        TenantId::DEFAULT,
        "provider must scope handle to caller's tenant (ADR-040 §D-3)"
    );

    let h_bm25 = h.bm25().expect("bm25 plumbed through");
    assert!(
        Arc::ptr_eq(h_bm25, &bm25),
        "bm25 accessor must return the SAME Arc the builder was given"
    );

    // Second route for the SAME `(tenant, partition)` key: each
    // accessor must yield the SAME Arc as the first call (router
    // `handle_cache` discipline per ADR-037 §D-2 / §D-6 — append-
    // only at v1.0).
    let h2 = router
        .route(TenantId::DEFAULT, PartitionId::ZERO)
        .expect("re-route");
    assert!(
        Arc::ptr_eq(&h, &h2),
        "second route is a cache hit — same TenantHandle Arc"
    );
    let h2_vec = h2.vector().expect("vector still Some on cache hit");
    let h2_community = h2.community().expect("community still Some on cache hit");
    let h2_bm25 = h2.bm25().expect("bm25 still Some on cache hit");
    assert!(Arc::ptr_eq(h_vec, h2_vec), "vector Arc identity preserved");
    assert!(Arc::ptr_eq(h_bm25, h2_bm25), "bm25 Arc identity preserved");
    // Community handles are cloned `Arc<CommunityIndexHandle>` (the
    // provider is consulted once per cache entry); the cached entry
    // returns the same Arc it materialised on the first route.
    assert!(
        Arc::ptr_eq(h_community, h2_community),
        "community Arc identity preserved across cache hit"
    );
}

// ─────────────────────────────────────────────────────────────────────
// Pin #15 — M3.c per-tenant handle scoping at the routing facade.
// ─────────────────────────────────────────────────────────────────────
//
// PIN: a router with all three M3 substrate deps wired must produce a
// `TenantHandle` for tenant A whose `community()` is scoped to tenant A,
// and a `TenantHandle` for tenant B whose `community()` is scoped to
// tenant B — even though the underlying `Arc<dyn
// CommunityIndexProvider>` is shared and BM25 / vector handles are
// shared by construction (workspace-wide stores per ADR-037 §D-1).
// The per-tenant projection lives inside each engine, not in the
// router.
//
// This pin lives at the routing-test boundary so the
// router-passes-tenant-unchanged invariant is caught quickly.

#[test]
fn router_per_tenant_handle_scoping_via_provider_facade() {
    // NOTE (codex M3.c retro F3): this pin verifies the routing
    // FACADE plumbing — the router passes the caller's `TenantId`
    // unchanged to the provider. It does NOT verify cross-tenant
    // DATA isolation; `StubProvider` unconditionally returns a
    // handle scoped to whichever tenant is asked, so a router bug
    // that leaked tenant A's request to tenant B's data path would
    // not be caught here.
    // Two-tenant fixture: DEFAULT + SYSTEM (the only two tenants
    // available at v1.0 — see pin #1 / #2). The integration test
    // covers user-tenant pairs via the same SYSTEM-carve-out
    // pattern.
    let io = Arc::new(InMemoryPageIo::new());
    let pool = BufferPool::new(8, io);
    let mgr = TxnManager::new();
    let catalog = Arc::new(SystemCatalog::new());
    catalog.bootstrap(&pool, &mgr).expect("bootstrap");
    let crud = Arc::new(CrudStore::new());

    let vector: Arc<dyn arcgraph_storage::vector_store::VectorPageStoreHandle> =
        Arc::new(LocalNoopVectorStore);
    let provider: Arc<dyn arcgraph_community::CommunityIndexProvider> = Arc::new(StubProvider);
    let bm25: Arc<dyn arcgraph_storage::mutation_log::Bm25IndexStoreHandle> =
        Arc::new(LocalNoopBm25Store);

    let router = MultiTenantRouter::builder(catalog, crud)
        .vector(Arc::clone(&vector))
        .community(Arc::clone(&provider))
        .bm25(Arc::clone(&bm25))
        .build();

    // Route under both tenants.
    let h_a = router
        .route(TenantId::DEFAULT, PartitionId::ZERO)
        .expect("tenant A (DEFAULT) routes");
    let h_b = router
        .route(TenantId::SYSTEM, PartitionId::ZERO)
        .expect("tenant B (SYSTEM, via ADR-037 §D-5 carve-out) routes");

    // Distinct routing keys.
    assert_ne!(h_a.tenant(), h_b.tenant());

    // Community: per-tenant projection. The shared `StubProvider` is
    // queried for each tenant; the returned handle's `tenant()` must
    // match the caller's tenant — a cross-tenant return would be an
    // I-V2-equivalent invariant violation per ADR-040 §D-3.
    let community_a = h_a.community().expect("provider returns Some for tenant A");
    let community_b = h_b.community().expect("provider returns Some for tenant B");
    assert_eq!(
        community_a.tenant(),
        TenantId::DEFAULT,
        "tenant A's community handle must be scoped to tenant A"
    );
    assert_eq!(
        community_b.tenant(),
        TenantId::SYSTEM,
        "tenant B's community handle must be scoped to tenant B"
    );
    assert_ne!(
        community_a.tenant(),
        community_b.tenant(),
        "the two community handles MUST be scoped to different tenants — \
         no cross-tenant leak even though the provider Arc is shared"
    );

    // BM25 + vector: workspace-wide handles are shared by construction
    // (per ADR-037 §D-1 — `Arc<dyn …>` cloned into every TenantHandle).
    // Per-tenant projection lives inside each engine via per-tenant
    // directory layout (BM25 — `<data_dir>/bm25/<tenant_id>/…` per
    // ADR-039 §D-4) and per-tenant arena keying (vector). The router's
    // contract is that the Arc identity is preserved across tenants;
    // the engines themselves enforce per-tenant isolation.
    let bm25_a = h_a.bm25().expect("bm25 Some for A");
    let bm25_b = h_b.bm25().expect("bm25 Some for B");
    assert!(
        Arc::ptr_eq(bm25_a, bm25_b),
        "bm25 commit-side Arc is shared across tenants by construction \
         (per-tenant projection lives inside Bm25Service)"
    );
    let vec_a = h_a.vector().expect("vector Some for A");
    let vec_b = h_b.vector().expect("vector Some for B");
    assert!(
        Arc::ptr_eq(vec_a, vec_b),
        "vector page-store Arc is shared across tenants by construction \
         (per-tenant arena keying lives inside VectorPageStore)"
    );

    // CRUD: same Arc identity across tenants — the I-V2 invariant
    // (per pin #2 above) lives in CrudStore's per-tenant maps, not
    // in the router.
    assert!(
        Arc::ptr_eq(h_a.crud(), h_b.crud()),
        "CrudStore Arc shared across tenants (per-tenant projection \
         lives inside CrudStore)"
    );
}
