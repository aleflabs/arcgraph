//! Engine bootstrap integration test (M3.d-3 sub-task D —
//! cross-tenant isolation + end-to-end refresh).
//!
//! Pins the F-2 SUBSTANTIVE closure: after [`bootstrap_engine`]
//! returns, the production [`ProductionRefreshHook`] is wired to
//! the [`CommunityRefreshScheduler`], the [`MultiTenantRouter`]
//! surfaces per-tenant [`CommunityIndexHandle`]s through the same
//! shared [`BTreeMembershipIndex`], and a forced scheduler tick
//! installs assignments observable through the router.
//!
//! Companion to the engine module's per-file unit tests
//! (`crates/arcgraph-storage/src/engine/{bootstrap,graph_adapter,
//! refresh_hook}.rs::tests`); this integration test exercises
//! cross-tenant boundary isolation that the unit tests can't pin
//! cleanly because they live inside the module they test.
//!
use std::sync::Arc;

use arcgraph_community::{CommunityIndexId, Level};
use arcgraph_core::{LabelId, Lsn, NodeId, PartitionId, TenantId, TypeId};
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::catalog::SystemCatalog;
use arcgraph_storage::crud::{self, CrudStore, PropertyData};
use arcgraph_storage::engine::{EngineConfig, bootstrap_engine};
use arcgraph_storage::io::InMemoryPageIo;
use arcgraph_storage::transaction::TxnManager;

/// Fixture: bootstrapped catalog + crud + txn manager.
fn fixture() -> (Arc<SystemCatalog>, Arc<CrudStore>, Arc<TxnManager>) {
    let io = Arc::new(InMemoryPageIo::new());
    let pool = BufferPool::new(8, io);
    let mgr = Arc::new(TxnManager::new());
    let catalog = Arc::new(SystemCatalog::new());
    catalog.bootstrap(&pool, &mgr).expect("bootstrap");
    (catalog, Arc::new(CrudStore::new()), mgr)
}

/// Install a fully-connected (clique) topology of `n` nodes for
/// `tenant`. The clique structure is convenient for community
/// detection oracle: a single tenant's clique forms a single
/// community at FINEST.
fn install_clique(crud: &Arc<CrudStore>, mgr: &Arc<TxnManager>, tenant: TenantId, n: u64) {
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
    for i in 0..(n as usize) {
        for j in (i + 1)..(n as usize) {
            crud::create_rel(
                crud,
                &mut tx,
                tenant,
                nids[i],
                nids[j],
                ty,
                &PropertyData::Empty,
            )
            .expect("create_rel");
        }
    }
    crud::commit(tx, crud).expect("commit");
}

/// PIN: bootstrap with two tenants (DEFAULT + SYSTEM); a forced
/// scheduler tick produces per-tenant assignments observable
/// through the router's `TenantHandle::community()`. Per-tenant
/// isolation: each tenant's nodes only resolve to communities from
/// their own install.
#[test]
fn bootstrap_cross_tenant_isolation_after_forced_tick() {
    let (catalog, crud, mgr) = fixture();
    // DEFAULT: 5-node clique.
    install_clique(&crud, &mgr, TenantId::DEFAULT, 5);
    // SYSTEM: 4-node clique. Per ADR-037 §D-5 SYSTEM is the
    // single second-tenant carve-out at v1.0.
    install_clique(&crud, &mgr, TenantId::SYSTEM, 4);

    let mut cfg = EngineConfig::new(
        Arc::clone(&catalog),
        Arc::clone(&crud),
        Arc::clone(&mgr),
        CommunityIndexId::new(1),
    );
    cfg.include_system_tenant = true;
    // Long interval so natural cadence doesn't fire; we drive
    // tick() explicitly. initial_install_lsn above any commit
    // LSN we issued.
    cfg.scheduler_config = arcgraph_community::SchedulerConfig {
        interval: std::time::Duration::from_secs(3600),
        max_tick_duration: std::time::Duration::from_secs(60),
        initial_install_lsn: Lsn::new(10_000),
    };
    let handles = bootstrap_engine(cfg).expect("bootstrap");

    // Both tenants registered with the hook + scheduler.
    assert_eq!(
        handles.refresh_hook.registered_tenants().len(),
        2,
        "DEFAULT + SYSTEM must both be registered"
    );
    assert_eq!(handles.scheduler.health_check().registered_tenants, 2);

    // Force a tick. Both tenants' membership-index installs
    // should land deterministically.
    handles.scheduler.tick();
    let h = handles.scheduler.health_check();
    assert_eq!(h.total_ticks, 1);
    assert_eq!(h.total_refresh_failures, 0, "no failures expected");
    assert_eq!(h.total_soft_skips, 0, "all registered tenants resolve");

    // Pin: every DEFAULT node has a community assignment;
    // every SYSTEM node has a community assignment.
    let route_default = handles
        .router
        .route(TenantId::DEFAULT, PartitionId::ZERO)
        .expect("route DEFAULT");
    let route_system = handles
        .router
        .route(TenantId::SYSTEM, PartitionId::ZERO)
        .expect("route SYSTEM");
    let h_default = route_default.community().expect("community DEFAULT");
    let h_system = route_system.community().expect("community SYSTEM");

    // DEFAULT NodeIds are 1..=5 (CrudStore allocates 1-indexed).
    for raw in 1..=5u64 {
        let c = h_default
            .membership(NodeId::new(raw), Level::FINEST, Lsn::MAX)
            .expect("lookup");
        assert!(
            c.is_some(),
            "DEFAULT node {raw} must have an assignment after tick"
        );
    }
    // SYSTEM NodeIds are also 1..=4 (per-tenant id-space; the
    // CrudStore's `next_node` is keyed by tenant).
    for raw in 1..=4u64 {
        let c = h_system
            .membership(NodeId::new(raw), Level::FINEST, Lsn::MAX)
            .expect("lookup");
        assert!(
            c.is_some(),
            "SYSTEM node {raw} must have an assignment after tick"
        );
    }

    // Cross-tenant isolation: looking up a SYSTEM-only node id
    // (say, NodeId 4 with `tenant = DEFAULT`) is fine — it might
    // exist in DEFAULT's space too — but the assignment for
    // (DEFAULT, 4) and (SYSTEM, 4) are independent installs from
    // independent Graphs. Verify by checking that DEFAULT's
    // node 5 (which doesn't exist in SYSTEM since SYSTEM only
    // has nodes 1..=4) yields `None` when looked up under
    // SYSTEM.
    let cross_tenant = h_system
        .membership(NodeId::new(5), Level::FINEST, Lsn::MAX)
        .expect("lookup SYSTEM/5");
    assert!(
        cross_tenant.is_none(),
        "SYSTEM has no NodeId(5) (clique was 4-node) — must return None, \
         not leak DEFAULT's NodeId(5) assignment; got {cross_tenant:?}"
    );

    handles.scheduler.shutdown();
}

/// PIN: forced-tick burst stress — the engine must remain healthy
/// across many back-to-back ticks.
#[test]
fn bootstrap_handles_forced_tick_burst() {
    let (catalog, crud, mgr) = fixture();
    install_clique(&crud, &mgr, TenantId::DEFAULT, 6);

    let mut cfg = EngineConfig::new(
        Arc::clone(&catalog),
        Arc::clone(&crud),
        Arc::clone(&mgr),
        CommunityIndexId::new(1),
    );
    cfg.scheduler_config = arcgraph_community::SchedulerConfig {
        interval: std::time::Duration::from_secs(3600),
        max_tick_duration: std::time::Duration::from_secs(60),
        initial_install_lsn: Lsn::new(10_000),
    };
    let handles = bootstrap_engine(cfg).expect("bootstrap");

    // Exercise a 20-tick burst.
    const FORCED_TICK_COUNT: u64 = 20;
    for _ in 0..FORCED_TICK_COUNT {
        handles.scheduler.tick();
    }
    let h = handles.scheduler.health_check();
    assert!(h.total_ticks >= FORCED_TICK_COUNT, "tick count");
    assert_eq!(h.total_refresh_failures, 0);
    assert_eq!(h.total_soft_skips, 0);
    assert!(h.last_tick_completed);

    // Per-node assignments after the burst — each tick is the
    // same install (deterministic Leiden on the same re-materialised
    // graph: no commits between ticks, and the CrudStore→Graph
    // adapter is byte-identical on identical substrate), so the
    // latest visible install is also the right one.
    let route = handles
        .router
        .route(TenantId::DEFAULT, PartitionId::ZERO)
        .expect("route");
    let h_default = route.community().expect("community");
    for raw in 1..=6u64 {
        let c = h_default
            .membership(NodeId::new(raw), Level::FINEST, Lsn::MAX)
            .expect("lookup");
        assert!(c.is_some(), "node {raw} after burst");
    }

    handles.scheduler.shutdown();
}

/// PIN: bootstrap is idempotent under double-shutdown — Drop +
/// explicit shutdown both cleanly join the scheduler thread.
#[test]
fn bootstrap_double_shutdown_is_idempotent() {
    let (catalog, crud, mgr) = fixture();
    install_clique(&crud, &mgr, TenantId::DEFAULT, 3);

    let cfg = EngineConfig::new(
        Arc::clone(&catalog),
        Arc::clone(&crud),
        Arc::clone(&mgr),
        CommunityIndexId::new(1),
    );
    let handles = bootstrap_engine(cfg).expect("bootstrap");
    handles.scheduler.shutdown();
    let h = handles.scheduler.health_check();
    assert!(h.shut_down);
    handles.scheduler.shutdown(); // idempotent
    // Implicit Drop on `handles` runs shutdown a third time —
    // must not panic.
}

/// PIN: empty tenant (no nodes / edges yet) materialises into a
/// degenerate Graph but the bootstrap path doesn't fail. The
/// scheduler tick produces a no-op install (community count = 0
/// at FINEST since no real vertices exist; only the phantom
/// vertex 0).
#[test]
fn bootstrap_empty_tenant_does_not_fail() {
    let (catalog, crud, mgr) = fixture();
    // No ingest — DEFAULT has zero nodes / edges.

    let mut cfg = EngineConfig::new(
        Arc::clone(&catalog),
        Arc::clone(&crud),
        Arc::clone(&mgr),
        CommunityIndexId::new(1),
    );
    cfg.scheduler_config = arcgraph_community::SchedulerConfig {
        interval: std::time::Duration::from_secs(3600),
        max_tick_duration: std::time::Duration::from_secs(60),
        initial_install_lsn: Lsn::new(10_000),
    };
    let handles = bootstrap_engine(cfg).expect("bootstrap");

    // Forced tick succeeds.
    handles.scheduler.tick();
    let h = handles.scheduler.health_check();
    assert_eq!(h.total_ticks, 1);
    assert_eq!(h.total_refresh_failures, 0, "no failures on empty tenant");
    assert_eq!(h.total_soft_skips, 0, "registered tenant resolves");

    // The router still routes; the community handle is plumbed
    // through; lookups against absent nodes return Ok(None) per
    // ADR-040 amendment-04 D-3 always-Some-at-provider posture.
    let route = handles
        .router
        .route(TenantId::DEFAULT, PartitionId::ZERO)
        .expect("route");
    let h_default = route.community().expect("community handle");
    let lookup = h_default
        .membership(NodeId::new(1), Level::FINEST, Lsn::MAX)
        .expect("lookup");
    assert!(
        lookup.is_none(),
        "no nodes ingested -> no assignments -> Ok(None)"
    );

    handles.scheduler.shutdown();
}

/// PIN: the engine adapter sizes graphs as `n = high_water + 1`
/// so vertex `0` is a phantom slot reserved for the unused
/// `NodeId::ZERO` sentinel (`CrudStore::alloc_node` allocates
/// `1`-indexed). After a forced tick the membership index MUST
/// NOT carry an entry for `NodeId::ZERO` — phantom assignments
/// would surface as `Ok(Some(_))` on a `(tenant, NodeId::ZERO)`
/// lookup and silently corrupt downstream consumers (e.g.,
/// community-aware query planning, the M3.d-2 incremental delta
/// path).
///
/// Load-bearing recurrence-prevention pin. The fix is in
/// `arcgraph_community::GveLeiden::install_into`'s
/// `n_skip_prefix` parameter, threaded through
/// `arcgraph_community::OwnedRefreshInputs::n_skip_prefix` and set
/// to `1` by `arcgraph_storage::engine::ProductionRefreshHook`
/// (closing PR #235 round-2 finding MED-3). Reverting the filter
/// MUST cause this test to fail.
#[test]
fn bootstrap_filters_phantom_node_zero_after_forced_tick() {
    let (catalog, crud, mgr) = fixture();
    install_clique(&crud, &mgr, TenantId::DEFAULT, 5);

    let mut cfg = EngineConfig::new(
        Arc::clone(&catalog),
        Arc::clone(&crud),
        Arc::clone(&mgr),
        CommunityIndexId::new(1),
    );
    cfg.scheduler_config = arcgraph_community::SchedulerConfig {
        interval: std::time::Duration::from_secs(3600),
        max_tick_duration: std::time::Duration::from_secs(60),
        initial_install_lsn: Lsn::new(10_000),
    };
    let handles = bootstrap_engine(cfg).expect("bootstrap");

    handles.scheduler.tick();
    let h = handles.scheduler.health_check();
    assert_eq!(h.total_ticks, 1);
    assert_eq!(h.total_refresh_failures, 0);
    assert_eq!(h.total_soft_skips, 0);

    let route = handles
        .router
        .route(TenantId::DEFAULT, PartitionId::ZERO)
        .expect("route");
    let h_default = route.community().expect("community handle");

    // Sanity: real nodes 1..=5 DO have assignments after the tick.
    for raw in 1..=5u64 {
        let c = h_default
            .membership(NodeId::new(raw), Level::FINEST, Lsn::MAX)
            .expect("lookup real node");
        assert!(
            c.is_some(),
            "real DEFAULT node {raw} must resolve to a community"
        );
    }

    // Load-bearing pin: NodeId::ZERO must NOT have an assignment
    // at FINEST. The phantom vertex 0 in the engine-built Graph
    // is filtered by ProductionRefreshHook returning
    // n_skip_prefix = 1. Reverting that filter MUST break this
    // assertion (the install_into call would emit a (NodeId::ZERO,
    // CommunityId) pair at FINEST and the lookup below would
    // surface Ok(Some(_))).
    let phantom_finest = h_default
        .membership(NodeId::ZERO, Level::FINEST, Lsn::MAX)
        .expect("lookup NodeId::ZERO at FINEST");
    assert!(
        phantom_finest.is_none(),
        "NodeId::ZERO must not surface in the membership index after \
         a forced tick (phantom slot from n = high_water + 1 must be \
         filtered by ProductionRefreshHook's n_skip_prefix = 1); got \
         {phantom_finest:?}"
    );

    handles.scheduler.shutdown();
}
