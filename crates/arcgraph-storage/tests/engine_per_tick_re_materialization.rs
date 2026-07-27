//! Per-tick re-materialisation integration test (ADR-040 amendment-05 §D-5).
//!
//! Pins the load-bearing migration invariant: post-bootstrap commits
//! land in the next daily-refresh tick's GVE-Leiden output. v1.0
//! pre-amendment-05 (FROZEN-GRAPH posture; PR #235) did NOT honour
//! this invariant — new vertices added after bootstrap never
//! appeared in any community at any subsequent tick until operator
//! restart.
//!
//! ## What this test pins
//!
//! 1. **3-tenant bootstrap.** `bootstrap_engine` is invoked with 3
//!    tenants (DEFAULT + 2 others) plus seed topology for each.
//! 2. **First-tick assignment.** `scheduler.tick()` runs once;
//!    every seeded vertex in every tenant has a community
//!    assignment in the membership index.
//! 3. **Post-bootstrap commits.** Each tenant gets one new node +
//!    one new edge committed AFTER the first tick.
//! 4. **Second-tick visibility.** `scheduler.tick()` runs again;
//!    the new vertices MUST have community assignments in the
//!    membership index — i.e., the canonical reset re-materialised
//!    the per-tenant Graph from CrudStore and the Leiden run
//!    placed the new vertex into a community.
//!
//! ## Why this test breaks under FROZEN-GRAPH
//!
//! Pre-amendment-05, the production hook held `HashMap<TenantId,
//! Box<Graph>>` populated at boot. The second tick re-ran
//! GVE-Leiden against the SAME frozen Graph (no new vertices), so
//! the new node never appeared in the membership index — it was
//! invisible to community queries until the operator restarted the
//! engine.
//!
//! Post-amendment-05, each tick calls `adapter.materialize(tenant)`
//! afresh; new commits between ticks are visible.
//!
//! ## Reverse-test discipline (Phase 4.3)
//!
//! Reverting the amendment-05 migration — re-introducing the
//! FROZEN-GRAPH `ProductionRefreshHookBuilder` or the borrowed-ref
//! trait shape — would surface as `lookup(post_bootstrap_node)
//! == Ok(None)` after the second tick (the assertion `is_some()`
//! would fail). This is the test that pins amendment-05's load-
//! bearing semantic.

use std::sync::Arc;
use std::time::Duration;

use arcgraph_community::{CommunityIndexId, Level, MembershipIndex, SchedulerConfig};
use arcgraph_core::{LabelId, Lsn, NodeId, PartitionId, TenantId, TypeId};
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::catalog::SystemCatalog;
use arcgraph_storage::crud::{self, CrudStore, PropertyData};
use arcgraph_storage::engine::{EngineConfig, bootstrap_engine};
use arcgraph_storage::io::InMemoryPageIo;
use arcgraph_storage::transaction::TxnManager;

/// Install a small per-tenant topology directly through the
/// CrudStore. Creates `n` nodes; chains them in a ring (0-1, 1-2,
/// …, (n-1)-0). Returns the node ids.
fn install_topology(
    crud: &Arc<CrudStore>,
    mgr: &Arc<TxnManager>,
    tenant: TenantId,
    n: u64,
) -> Vec<NodeId> {
    let label = LabelId::new(1);
    let ty = TypeId::new(1);
    let mut tx = mgr.begin(tenant);
    let mut nids = Vec::with_capacity(n as usize);
    for _ in 0..n {
        nids.push(
            crud::create_node(crud, &mut tx, tenant, label, &PropertyData::Empty)
                .expect("create_node"),
        );
    }
    for i in 0..n {
        let src = nids[i as usize];
        let dst = nids[((i + 1) % n) as usize];
        if src != dst {
            crud::create_rel(crud, &mut tx, tenant, src, dst, ty, &PropertyData::Empty)
                .expect("create_rel");
        }
    }
    crud::commit(tx, crud).expect("commit");
    nids
}

/// Add one node to `tenant` connected to `attach_to`. Returns the
/// new node's id.
fn add_node_to_existing_tenant(
    crud: &Arc<CrudStore>,
    mgr: &Arc<TxnManager>,
    tenant: TenantId,
    attach_to: NodeId,
) -> NodeId {
    let label = LabelId::new(1);
    let ty = TypeId::new(1);
    let mut tx = mgr.begin(tenant);
    let new_nid = crud::create_node(crud, &mut tx, tenant, label, &PropertyData::Empty)
        .expect("post-bootstrap create_node");
    crud::create_rel(
        crud,
        &mut tx,
        tenant,
        new_nid,
        attach_to,
        ty,
        &PropertyData::Empty,
    )
    .expect("post-bootstrap create_rel");
    crud::commit(tx, crud).expect("post-bootstrap commit");
    new_nid
}

#[test]
fn three_tenants_post_bootstrap_commits_appear_in_next_tick() {
    // ─── Fixture ────────────────────────────────────────────────
    let io = Arc::new(InMemoryPageIo::new());
    let pool = BufferPool::new(8, io);
    let mgr = Arc::new(TxnManager::new());
    let catalog = Arc::new(SystemCatalog::new());
    catalog.bootstrap(&pool, &mgr).expect("bootstrap catalog");
    let crud_store = Arc::new(CrudStore::new());

    // 3 tenants exercised. DEFAULT is bootstrapped by
    // `catalog.bootstrap`. SYSTEM is auto-included via
    // `include_system_tenant=true`. The third (`tenant_c`) is
    // registered with the hook + scheduler post-bootstrap to
    // exercise the runtime tenant-add path of amendment-05's
    // `ProductionRefreshHook::register_tenant`.
    let tenant_a = TenantId::DEFAULT;
    let tenant_b = TenantId::SYSTEM;
    let tenant_c = TenantId::new(200);

    // Each tenant gets a 4-node ring.
    let nids_a = install_topology(&crud_store, &mgr, tenant_a, 4);
    let nids_b = install_topology(&crud_store, &mgr, tenant_b, 4);
    let nids_c = install_topology(&crud_store, &mgr, tenant_c, 4);

    // ─── Bootstrap ──────────────────────────────────────────────
    //
    // Long interval so the natural cadence doesn't fire mid-test;
    // we drive `tick()` explicitly. `initial_install_lsn` raised
    // above any prior commit LSN to keep monotonic install_level.
    let mut cfg = EngineConfig::new(
        Arc::clone(&catalog),
        Arc::clone(&crud_store),
        Arc::clone(&mgr),
        CommunityIndexId::new(1),
    );
    cfg.include_system_tenant = true; // Adds tenant_b (SYSTEM).
    cfg.scheduler_config = SchedulerConfig {
        interval: Duration::from_secs(3600),
        max_tick_duration: Duration::from_secs(60),
        initial_install_lsn: Lsn::new(10_000),
    };
    let handles = bootstrap_engine(cfg).expect("bootstrap_engine");

    // Post-bootstrap, register tenant_c with both hook + scheduler.
    // This exercises the runtime tenant-add path of amendment-05.
    handles.refresh_hook.register_tenant(tenant_c);
    handles.scheduler.register(tenant_c);

    // All 3 tenants are registered with the hook + scheduler.
    let registered = handles.refresh_hook.registered_tenants();
    assert!(
        registered.contains(&tenant_a),
        "tenant_a must be registered"
    );
    assert!(
        registered.contains(&tenant_b),
        "tenant_b must be registered"
    );
    assert!(
        registered.contains(&tenant_c),
        "tenant_c must be registered"
    );

    // ─── First tick ─────────────────────────────────────────────
    handles.scheduler.tick();
    let h1 = handles.scheduler.health_check();
    assert_eq!(h1.total_ticks, 1, "first tick fired");
    assert_eq!(
        h1.total_refresh_failures, 0,
        "no failures expected for healthy tenants"
    );
    assert_eq!(
        h1.total_soft_skips, 0,
        "every registered tenant has live topology; no soft-skip"
    );

    // After tick 1, every seeded vertex has a community.
    for (tenant, nids) in [
        (tenant_a, &nids_a),
        (tenant_b, &nids_b),
        (tenant_c, &nids_c),
    ] {
        for nid in nids {
            let cid = handles
                .membership_index
                .lookup(tenant, *nid, Level::FINEST, Lsn::MAX)
                .expect("lookup ok")
                .unwrap_or_else(|| {
                    panic!("post-tick-1: tenant {tenant:?} node {nid:?} must have a community")
                });
            // Not asserting specific cid; GVE-Leiden's choice
            // depends on iteration order. Just must be Some.
            let _ = cid;
        }
    }

    // ─── Post-bootstrap commits (the load-bearing window) ───────
    //
    // Add one node to EACH tenant, attached to that tenant's
    // first ring node. Pre-amendment-05 these new vertices would
    // never appear in a community after subsequent ticks (the
    // FROZEN-GRAPH would never re-materialise from CrudStore).
    let new_a = add_node_to_existing_tenant(&crud_store, &mgr, tenant_a, nids_a[0]);
    let new_b = add_node_to_existing_tenant(&crud_store, &mgr, tenant_b, nids_b[0]);
    let new_c = add_node_to_existing_tenant(&crud_store, &mgr, tenant_c, nids_c[0]);

    // ─── Second tick ────────────────────────────────────────────
    //
    // Per amendment-05, each tick calls `adapter.materialize`
    // afresh — the new commits ARE visible, and the new vertices
    // get community assignments.
    handles.scheduler.tick();
    let h2 = handles.scheduler.health_check();
    assert_eq!(h2.total_ticks, 2, "second tick fired");
    assert_eq!(
        h2.total_refresh_failures, 0,
        "no failures expected post-second-tick"
    );
    assert_eq!(
        h2.total_soft_skips, 0,
        "second tick should not soft-skip — all tenants still healthy"
    );

    // ★ The headline amendment-05 invariant: post-bootstrap
    //   commits ARE visible after the next tick.
    for (tenant, new_nid) in [(tenant_a, new_a), (tenant_b, new_b), (tenant_c, new_c)] {
        let cid = handles
            .membership_index
            .lookup(tenant, new_nid, Level::FINEST, Lsn::MAX)
            .expect("lookup ok");
        assert!(
            cid.is_some(),
            "AMENDMENT-05 INVARIANT: post-bootstrap node {new_nid:?} for tenant {tenant:?} \
             MUST appear in a community after the next tick. Got None — \
             this means the per-tick re-mat path is broken (or reverted to FROZEN-GRAPH)."
        );
    }

    // ─── Router-side observability for in-catalog tenants ──────
    //
    // The new community assignments are observable through the
    // router's TenantHandle::community() projection — pin the
    // cross-substrate path. The router only routes tenants the
    // catalog knows about (DEFAULT, SYSTEM). tenant_c was added
    // post-bootstrap to the hook + scheduler only (exercising
    // the runtime tenant-add path); since it's not in the
    // catalog, the router refuses to route it (UnknownTenant)
    // — that's the catalog-router contract, not amendment-05's
    // concern.
    for tenant in [tenant_a, tenant_b] {
        let route_handle = handles
            .router
            .route(tenant, PartitionId::ZERO)
            .expect("route");
        let community_handle = route_handle.community().expect("community handle");
        assert_eq!(community_handle.tenant(), tenant);
        assert_eq!(community_handle.partition(), PartitionId::ZERO);
    }

    // ─── Diagnostic cache populated for each tenant ─────────────
    //
    // Per amendment-05 §D-5, ProductionRefreshHook holds a
    // `last_materialised: DashMap<TenantId, Arc<Graph>>` populated
    // on each `resolve()` call. After two ticks, all 3 tenants
    // should have a cached Arc<Graph> reflecting the second-tick
    // materialisation (which includes the post-bootstrap commits).
    for (tenant, expected_n) in [
        (tenant_a, 4 + 1 /* new node */ + 1 /* phantom 0 */),
        (tenant_b, 4 + 1 + 1),
        (tenant_c, 4 + 1 + 1),
    ] {
        let cached = handles
            .refresh_hook
            .last_materialised(tenant)
            .unwrap_or_else(|| panic!("tenant {tenant:?} must have a diagnostic cache entry"));
        assert_eq!(
            cached.n(),
            expected_n,
            "tenant {tenant:?} diagnostic cache reflects second-tick materialisation"
        );
    }

    handles.scheduler.shutdown();
}

#[test]
fn warm_up_eagerly_materialises_per_tenant() {
    // Companion test: amendment-05 §D-5 specifies that operators
    // who want eager materialisation can call `warm_up`
    // explicitly. This test pins that `warm_up` populates the
    // diagnostic cache without firing a scheduler tick.
    let io = Arc::new(InMemoryPageIo::new());
    let pool = BufferPool::new(8, io);
    let mgr = Arc::new(TxnManager::new());
    let catalog = Arc::new(SystemCatalog::new());
    catalog.bootstrap(&pool, &mgr).expect("bootstrap catalog");
    let crud_store = Arc::new(CrudStore::new());

    install_topology(&crud_store, &mgr, TenantId::DEFAULT, 3);

    let mut cfg = EngineConfig::new(
        Arc::clone(&catalog),
        Arc::clone(&crud_store),
        Arc::clone(&mgr),
        CommunityIndexId::new(1),
    );
    cfg.scheduler_config = SchedulerConfig {
        interval: Duration::from_secs(3600),
        max_tick_duration: Duration::from_secs(60),
        initial_install_lsn: Lsn::new(1_000),
    };
    let handles = bootstrap_engine(cfg).expect("bootstrap");

    // Pre-warm-up: diagnostic cache is empty (bootstrap does NOT
    // pre-materialise per amendment-05).
    assert!(
        handles
            .refresh_hook
            .last_materialised(TenantId::DEFAULT)
            .is_none(),
        "amendment-05: bootstrap_engine does NOT eagerly materialise; cache must be empty"
    );

    handles
        .refresh_hook
        .warm_up(TenantId::DEFAULT)
        .expect("warm_up");

    // Post-warm-up: diagnostic cache populated, but no scheduler
    // tick has fired (membership index is still empty).
    let cached = handles
        .refresh_hook
        .last_materialised(TenantId::DEFAULT)
        .expect("warm_up populated the cache");
    assert_eq!(cached.n(), 4 /* 3 nodes + phantom 0 */);

    let h = handles.scheduler.health_check();
    assert_eq!(
        h.total_ticks, 0,
        "warm_up does NOT fire a scheduler tick — that's the scheduler's job"
    );

    handles.scheduler.shutdown();
}
