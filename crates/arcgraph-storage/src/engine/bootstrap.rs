//! Engine bootstrap entry point (M3.d-3 sub-task C; per ADR-040
//! amendment-05 per-tick re-mat).
//!
//! [`bootstrap_engine`] is the v1.0 production composition entry
//! point that wires:
//!
//! 1. The shared [`crate::crud::CrudStore`] +
//!    [`crate::transaction::TxnManager`] +
//!    [`crate::catalog::SystemCatalog`] (storage substrate).
//! 2. The shared
//!    [`arcgraph_community::SharedBTreeIndexProvider`] (production
//!    [`arcgraph_community::CommunityIndexProvider`] from PR #218).
//! 3. The [`super::refresh_hook::ProductionRefreshHook`] (M3.d-3
//!    sub-task B; per-tick re-mat per amendment-05). Constructed
//!    empty; tenants the catalog reports + optionally
//!    [`arcgraph_core::TenantId::SYSTEM`] (per ADR-037 §D-5) are
//!    registered with the hook. **No** per-tenant materialisation
//!    happens at bootstrap — the first scheduler tick for each
//!    tenant materialises afresh from `CrudStore`.
//! 4. The [`arcgraph_community::CommunityRefreshScheduler`]
//!    (started with the production hook + every tenant registered).
//! 5. The [`crate::router::MultiTenantRouter`] (built via
//!    `MultiTenantRouterBuilder.community(provider)` per ADR-037
//!    amendment-01).
//!
//! After [`bootstrap_engine`] returns, the engine is fully
//! composed:
//!
//! - The router can `route(tenant, partition)` to surface a
//!   [`crate::router::TenantHandle`] whose `community()` projection
//!   resolves through the shared provider into the same
//!   `BTreeMembershipIndex` the scheduler installs into.
//! - The scheduler ticks at the configured cadence (default daily
//!   per ADR-040 §D-7); each tick calls
//!   [`arcgraph_community::RefreshHook::resolve`] which materialises
//!   the tenant's Graph fresh from `CrudStore`.
//!
//! ## Boot-time validation (opt-in)
//!
//! Pre-amendment-05 bootstrap pre-materialised every tenant's Graph,
//! which surfaced materialisation errors at boot. Amendment-05
//! defers materialisation to first tick. Operators who want
//! boot-time validation should call
//! `arcgraph_storage::engine::ProductionRefreshHook::warm_up` for
//! each registered tenant after `bootstrap_engine` returns and
//! before publishing the engine handles to consumers.
//!
//! Default `bootstrap_engine` does NOT call `warm_up` — boot must
//! remain robust to a single corrupted-tenant scenario; the operator
//! opts in if they want eager validation.
//!
//! ## What this slice deliberately does NOT do
//!
//! - **MCP server / wire protocol.** That is `arcgraph-mcp` /
//!   `arcgraph-cli`'s `arcgraph serve` (M5 / M6).
//! - **Vector arena wiring.** Optional per
//!   [`crate::router::MultiTenantRouterBuilder::vector`] — the
//!   bootstrap fn accepts an optional handle but does not
//!   construct a default arena (callers wire their own per
//!   ADR-035).
//! - **BM25 wiring.** Same — optional per
//!   [`crate::router::MultiTenantRouterBuilder::bm25`].
//! - **Live ingest pipeline.** v1.0 daily refresh against fresh
//!   per-tenant graphs is the canonical reset; live ingest /
//!   incremental updates are M3.d-2 (DF Leiden) at the
//!   membership-index layer.

use std::sync::Arc;

use arcgraph_community::{
    BTreeMembershipIndex, CommunityIndexId, CommunityIndexProvider, CommunityRefreshScheduler,
    LeidenParams, RefreshHook, RefreshObserver, SchedulerConfig, SharedBTreeIndexProvider,
};
use arcgraph_core::TenantId;
use thiserror::Error;
use tracing::info;

use super::graph_adapter::CrudStoreGraphAdapter;
use super::refresh_hook::ProductionRefreshHook;
use crate::catalog::SystemCatalog;
use crate::crud::CrudStore;
use crate::mutation_log::Bm25IndexStoreHandle;
use crate::router::MultiTenantRouter;
use crate::transaction::TxnManager;
use crate::vector_store::VectorPageStoreHandle;

/// Faults surfaced by [`bootstrap_engine`].
///
/// Codec-local per `docs/codec-error-translation.md`; the
/// CLI / MCP layer translates to user-facing errors.
///
/// Post-amendment-05, `bootstrap_engine` does not eagerly
/// materialise per-tenant Graphs — `register_tenant` is
/// idempotent + infallible — so this enum currently has no
/// variants. Future v1.1 bootstrap-time validations may add
/// variants.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum EngineError {}

/// Configuration for [`bootstrap_engine`].
///
/// Wraps the bootstrap-time inputs so the function signature stays
/// terse and so future v1.1+ optional deps land as additive fields
/// without breaking callers (per ADR-037 amendment-01's pattern).
pub struct EngineConfig {
    /// Shared catalog (must already be bootstrapped — the
    /// bootstrap fn does NOT call [`SystemCatalog::bootstrap`];
    /// callers chain that into their own startup sequence so
    /// catalog-bootstrap-time errors are surfaced in their own
    /// context).
    pub catalog: Arc<SystemCatalog>,
    /// Shared CRUD store.
    pub crud: Arc<CrudStore>,
    /// Shared transaction manager.
    pub txn_manager: Arc<TxnManager>,
    /// Optional vector-arena handle; threaded into the router via
    /// [`crate::router::MultiTenantRouterBuilder::vector`] when
    /// `Some`.
    pub vector: Option<Arc<dyn VectorPageStoreHandle>>,
    /// Optional BM25 commit-side handle; threaded into the router
    /// via [`crate::router::MultiTenantRouterBuilder::bm25`] when
    /// `Some`.
    pub bm25: Option<Arc<dyn Bm25IndexStoreHandle>>,
    /// The catalog-allocated identifier for the workspace's
    /// community index. v1.0 supports a single community index per
    /// deployment per ADR-040 §D-3; future v1.1 multi-index
    /// deployments allocate distinct ids per index.
    pub community_index_id: CommunityIndexId,
    /// GVE-Leiden parameters fed into the static refresh per
    /// ADR-040 §D-7. Default is sufficient for v1.0 deployments.
    pub leiden_params: LeidenParams,
    /// Scheduler cadence + soft caps. Default = daily refresh per
    /// ADR-040 §D-7.
    pub scheduler_config: SchedulerConfig,
    /// Whether to register [`TenantId::SYSTEM`] alongside the
    /// catalog-listed tenants. Per ADR-037 §D-5 SYSTEM is not in
    /// `list_tenants` but routes through the same
    /// [`MultiTenantRouter::route`] surface; deployments with
    /// catalog-side mutations attached to SYSTEM may want refresh
    /// against SYSTEM's substrate. Default `false` since SYSTEM
    /// has no user-facing edges in v1.0.
    pub include_system_tenant: bool,
    /// Optional ADR-202 community observability seam, threaded into
    /// [`CommunityRefreshScheduler::start_with_observer`]. When
    /// `Some`, every successful per-tenant refresh notifies the
    /// observer (the production impl is
    /// `arcgraph-mcp::transport::metrics::MetricsRegistry`, which
    /// sets the `arcgraph_leiden_last_run_seconds{tenant}` freshness
    /// gauge). Default `None` per the ADR-037 amendment-01
    /// additive-optional-dep pattern (`vector` / `bm25` precedents):
    /// existing callers see no change and pay one nullable-ptr check
    /// per tenant-day.
    pub refresh_observer: Option<Arc<dyn RefreshObserver>>,
}

impl EngineConfig {
    /// Construct a config with sensible defaults: no vector / BM25
    /// wiring, default Leiden params, default scheduler cadence,
    /// SYSTEM tenant excluded.
    #[must_use]
    pub fn new(
        catalog: Arc<SystemCatalog>,
        crud: Arc<CrudStore>,
        txn_manager: Arc<TxnManager>,
        community_index_id: CommunityIndexId,
    ) -> Self {
        Self {
            catalog,
            crud,
            txn_manager,
            vector: None,
            bm25: None,
            community_index_id,
            leiden_params: LeidenParams::default(),
            scheduler_config: SchedulerConfig::default(),
            include_system_tenant: false,
            refresh_observer: None,
        }
    }
}

/// Composed engine handles returned by [`bootstrap_engine`].
///
/// The struct is intentionally read-only after construction —
/// callers hold it for the lifetime of the engine and `Arc::clone`
/// the inner pieces into other subsystems (MCP server, CLI status
/// commands, integration tests).
pub struct EngineHandles {
    /// The composed multi-tenant router. Use
    /// [`MultiTenantRouter::route`] to obtain a per-tenant handle.
    pub router: MultiTenantRouter,
    /// The community refresh scheduler. Already started; the
    /// scheduler thread is running. Drop the Arc (or call
    /// [`CommunityRefreshScheduler::shutdown`]) to stop it.
    pub scheduler: Arc<CommunityRefreshScheduler>,
    /// The shared community provider — held so callers can
    /// `provider.index()` for diagnostic / direct-install paths
    /// without re-traversing the router.
    pub community_provider: Arc<SharedBTreeIndexProvider>,
    /// The production refresh hook (per-tick re-mat per ADR-040
    /// amendment-05). Held for diagnostic inspection
    /// (e.g., `hook.registered_tenants()`,
    /// `hook.last_materialised(tenant)`, `hook.warm_up(tenant)`);
    /// the scheduler also holds an `Arc<dyn RefreshHook>` clone
    /// internally.
    pub refresh_hook: Arc<ProductionRefreshHook>,
    /// The shared `Arc<BTreeMembershipIndex>` the provider +
    /// scheduler + hook all coordinate through. Exposed for
    /// engine-level diagnostics; callers normally read through
    /// the router's `TenantHandle::community()` projection.
    pub membership_index: Arc<BTreeMembershipIndex>,
}

/// Bootstrap the engine.
///
/// See module-level rustdoc for the full composition sequence.
/// Returns the composed [`EngineHandles`] on success.
///
/// # Sequencing (post amendment-05 — per-tick re-mat posture)
///
/// 1. Construct the [`SharedBTreeIndexProvider`] — the workspace's
///    single `BTreeMembershipIndex` is wrapped here.
/// 2. Construct the [`ProductionRefreshHook`] (empty; no per-tenant
///    state at construction). Register every catalog-listed tenant
///    (and optionally SYSTEM) with the hook.
/// 3. Start the [`CommunityRefreshScheduler`] with the production
///    hook + the configured scheduler config. The scheduler
///    thread is named `arcgraph-community-refresh`.
/// 4. Register every tenant with the scheduler so daily ticks pick
///    them up. Each tick materialises the tenant's Graph fresh
///    from `CrudStore` per ADR-040 amendment-05 §D-5.
/// 5. Build the [`MultiTenantRouter`] via
///    `builder(...).community(provider)` per ADR-037
///    amendment-01.
///
/// Operators who want eager materialisation (boot-time validation)
/// can call [`ProductionRefreshHook::warm_up`] on the returned
/// hook for each registered tenant. This bootstrap fn does NOT
/// call `warm_up` — boot is robust to single-tenant materialisation
/// failures.
///
/// # Errors
///
/// Currently infallible (post amendment-05); the `Result` shape is
/// retained for forward compatibility with future v1.1 boot-time
/// validations (e.g., a SystemCatalog integrity check or an
/// `EngineConfig::eager_warmup` flag).
pub fn bootstrap_engine(config: EngineConfig) -> Result<EngineHandles, EngineError> {
    info!(
        community_index_id = config.community_index_id.raw(),
        scheduler_interval = ?config.scheduler_config.interval,
        "bootstrap_engine: starting (per-tick re-mat per ADR-040 amendment-05)"
    );

    // ─── §1. Shared community provider ──────────────────────────
    //
    // SharedBTreeIndexProvider::new constructs a fresh
    // BTreeMembershipIndex internally; we read it back via
    // `.index()` so the hook + scheduler + provider all coordinate
    // through the same Arc<BTreeMembershipIndex>. Per ADR-040
    // amendment-04 §D-3.
    let community_provider = Arc::new(SharedBTreeIndexProvider::new(config.community_index_id));
    let membership_index: Arc<BTreeMembershipIndex> = Arc::clone(community_provider.index());
    let community_provider_dyn: Arc<dyn CommunityIndexProvider> =
        Arc::clone(&community_provider) as Arc<dyn CommunityIndexProvider>;

    // ─── §2. ProductionRefreshHook (per-tick re-mat per amendment-05) ────
    //
    // No per-tenant Graph materialisation at boot. The hook holds
    // an eligible-tenants set + a diagnostic Arc<Graph> cache
    // populated on first resolve(); each scheduler tick
    // re-materialises afresh.
    let adapter =
        CrudStoreGraphAdapter::new(Arc::clone(&config.crud), Arc::clone(&config.txn_manager));
    let refresh_hook = Arc::new(ProductionRefreshHook::new(
        adapter,
        Arc::clone(&membership_index),
        config.leiden_params,
    ));
    let mut tenants_to_register: Vec<TenantId> = config
        .catalog
        .list_tenants()
        .into_iter()
        .map(|r| r.tenant_id)
        .collect();
    if config.include_system_tenant {
        tenants_to_register.push(TenantId::SYSTEM);
    }
    for tenant in &tenants_to_register {
        refresh_hook.register_tenant(*tenant);
    }
    let refresh_hook_dyn: Arc<dyn RefreshHook> = Arc::clone(&refresh_hook) as Arc<dyn RefreshHook>;

    // ─── §3. Start scheduler ────────────────────────────────────
    //
    // ADR-202: the optional refresh observer threads through here so
    // successful refreshes report community freshness (the
    // `arcgraph_leiden_last_run_seconds{tenant}` gauge when the
    // observer is `arcgraph-mcp`'s `MetricsRegistry`). `None` is the
    // default-config path and preserves the pre-ADR-202 behaviour.
    let scheduler = CommunityRefreshScheduler::start_with_observer(
        config.scheduler_config,
        refresh_hook_dyn,
        config.refresh_observer,
    );
    // ─── §4. Register every tenant with the scheduler ───────────
    //
    // The scheduler's `register` is idempotent + safe under
    // concurrent calls. We register after `start` so the dedicated
    // thread has finished initialising.
    for tenant in &tenants_to_register {
        scheduler.register(*tenant);
    }

    // ─── §5. Build MultiTenantRouter ────────────────────────────
    let mut router_builder =
        MultiTenantRouter::builder(Arc::clone(&config.catalog), Arc::clone(&config.crud))
            .community(community_provider_dyn);
    if let Some(v) = config.vector {
        router_builder = router_builder.vector(v);
    }
    if let Some(b) = config.bm25 {
        router_builder = router_builder.bm25(b);
    }
    let router = router_builder.build();

    info!(
        registered_tenants = tenants_to_register.len(),
        "bootstrap_engine: complete (no eager Graph materialisation; first tick will materialise)"
    );

    Ok(EngineHandles {
        router,
        scheduler,
        community_provider,
        refresh_hook,
        membership_index,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use arcgraph_core::{LabelId, PartitionId, TypeId};

    use crate::buffer::BufferPool;
    use crate::crud::{self, PropertyData};
    use crate::io::InMemoryPageIo;

    /// Small bootstrap fixture: returns the workspace primitives a
    /// caller would assemble before calling [`bootstrap_engine`].
    fn fixture() -> (Arc<SystemCatalog>, Arc<CrudStore>, Arc<TxnManager>) {
        let io = Arc::new(InMemoryPageIo::new());
        let pool = BufferPool::new(8, io);
        let mgr = Arc::new(TxnManager::new());
        let catalog = Arc::new(SystemCatalog::new());
        catalog.bootstrap(&pool, &mgr).expect("bootstrap catalog");
        (catalog, Arc::new(CrudStore::new()), mgr)
    }

    /// Install a small per-tenant topology directly through the
    /// CrudStore (mirrors the engine's ingest path, less the WAL).
    fn install(crud: &Arc<CrudStore>, mgr: &Arc<TxnManager>, tenant: TenantId, n: u64) {
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
        // Ring topology: 0-1, 1-2, 2-0 (or just 0-1 for n=2).
        for i in 0..n {
            let src = nids[i as usize];
            let dst = nids[((i + 1) % n) as usize];
            if src != dst {
                crud::create_rel(crud, &mut tx, tenant, src, dst, ty, &PropertyData::Empty)
                    .expect("create_rel");
            }
        }
        crud::commit(tx, crud).expect("commit");
    }

    #[test]
    fn bootstrap_smoke_with_default_tenant_only() {
        // The catalog has DEFAULT post-bootstrap; ingest a small
        // topology so the materialised Graph isn't empty.
        let (catalog, crud, mgr) = fixture();
        install(&crud, &mgr, TenantId::DEFAULT, 4);

        let cfg = EngineConfig::new(
            Arc::clone(&catalog),
            Arc::clone(&crud),
            Arc::clone(&mgr),
            CommunityIndexId::new(1),
        );
        let handles = bootstrap_engine(cfg).expect("bootstrap");

        // Router is constructed and routes DEFAULT successfully.
        let h = handles
            .router
            .route(TenantId::DEFAULT, PartitionId::ZERO)
            .expect("route DEFAULT");
        assert_eq!(h.tenant(), TenantId::DEFAULT);
        // The community handle is plumbed through (the provider was
        // wired via the builder).
        assert!(
            h.community().is_some(),
            "TenantHandle::community() must be Some after bootstrap"
        );

        // Refresh hook holds DEFAULT.
        assert_eq!(
            handles.refresh_hook.registered_tenants(),
            vec![TenantId::DEFAULT]
        );

        // Scheduler is healthy and tenant is registered.
        let h = handles.scheduler.health_check();
        assert_eq!(h.registered_tenants, 1, "DEFAULT must be registered");
        assert!(!h.shut_down);

        handles.scheduler.shutdown();
    }

    #[test]
    fn bootstrap_with_include_system_tenant_registers_system() {
        let (catalog, crud, mgr) = fixture();
        install(&crud, &mgr, TenantId::DEFAULT, 3);
        install(&crud, &mgr, TenantId::SYSTEM, 2);

        let mut cfg = EngineConfig::new(
            Arc::clone(&catalog),
            Arc::clone(&crud),
            Arc::clone(&mgr),
            CommunityIndexId::new(7),
        );
        cfg.include_system_tenant = true;
        let handles = bootstrap_engine(cfg).expect("bootstrap");

        // Both DEFAULT and SYSTEM are registered.
        let regs = handles.refresh_hook.registered_tenants();
        assert!(regs.contains(&TenantId::DEFAULT), "regs = {regs:?}");
        assert!(regs.contains(&TenantId::SYSTEM), "regs = {regs:?}");
        assert_eq!(regs.len(), 2);

        let h = handles.scheduler.health_check();
        assert_eq!(h.registered_tenants, 2);

        handles.scheduler.shutdown();
    }

    #[test]
    fn bootstrap_end_to_end_refresh_populates_membership_index() {
        // End-to-end: bootstrap, ingest topology, force a tick,
        // verify the per-tenant membership index has assignments.
        use arcgraph_community::{Level, MembershipIndex};
        use arcgraph_core::Lsn;

        let (catalog, crud, mgr) = fixture();
        // 6-node topology: two triangles (0-1, 1-2, 0-2) and
        // (3-4, 4-5, 3-5).
        let label = LabelId::new(1);
        let ty = TypeId::new(1);
        let mut tx = mgr.begin(TenantId::DEFAULT);
        let mut n: Vec<arcgraph_core::NodeId> = Vec::with_capacity(6);
        for _ in 0..6 {
            n.push(
                crud::create_node(
                    &crud,
                    &mut tx,
                    TenantId::DEFAULT,
                    label,
                    &PropertyData::Empty,
                )
                .expect("create_node"),
            );
        }
        let edges = [(0usize, 1usize), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)];
        for &(u, v) in &edges {
            crud::create_rel(
                &crud,
                &mut tx,
                TenantId::DEFAULT,
                n[u],
                n[v],
                ty,
                &PropertyData::Empty,
            )
            .expect("create_rel");
        }
        crud::commit(tx, &crud).expect("commit");

        // Bootstrap with a long interval so the natural cadence
        // doesn't fire during the test; we drive `tick()`
        // explicitly. `initial_install_lsn` set above any commit
        // LSN we issued so the scheduler's monotonicity check
        // passes.
        let mut cfg = EngineConfig::new(
            Arc::clone(&catalog),
            Arc::clone(&crud),
            Arc::clone(&mgr),
            CommunityIndexId::new(1),
        );
        cfg.scheduler_config = SchedulerConfig {
            interval: std::time::Duration::from_secs(3600),
            max_tick_duration: std::time::Duration::from_secs(60),
            initial_install_lsn: Lsn::new(1_000),
        };
        let handles = bootstrap_engine(cfg).expect("bootstrap");

        // Force one tick. Per scheduler.rs this snapshots the
        // registered set into pending and runs one synchronous
        // sweep on the calling thread.
        handles.scheduler.tick();
        let h = handles.scheduler.health_check();
        assert_eq!(h.total_ticks, 1, "one forced tick");
        assert_eq!(
            h.total_refresh_failures, 0,
            "no failures expected for healthy DEFAULT tenant"
        );
        assert_eq!(
            h.total_soft_skips, 0,
            "DEFAULT was registered with the hook; no soft skip"
        );

        // After the tick, the membership index should have
        // assignments for DEFAULT at FINEST. We probe via the
        // shared membership index (the same Arc the provider
        // serves via the router).
        let lookup_n0 = handles
            .membership_index
            .lookup(
                TenantId::DEFAULT,
                arcgraph_core::NodeId::new(1),
                Level::FINEST,
                Lsn::MAX,
            )
            .expect("lookup");
        assert!(
            lookup_n0.is_some(),
            "node 1 must have a community assignment after tick"
        );
        let lookup_n5 = handles
            .membership_index
            .lookup(
                TenantId::DEFAULT,
                arcgraph_core::NodeId::new(6),
                Level::FINEST,
                Lsn::MAX,
            )
            .expect("lookup");
        assert!(
            lookup_n5.is_some(),
            "node 6 must have a community assignment after tick"
        );

        handles.scheduler.shutdown();
    }

    /// Pins ADR-202 D-4: `EngineConfig.refresh_observer` threads
    /// through `bootstrap_engine` into the scheduler, and a real
    /// forced tick (materialise → GveLeiden → install) notifies the
    /// observer exactly once per refreshed tenant.
    #[test]
    fn bootstrap_threads_refresh_observer_into_scheduler() {
        use std::sync::Mutex as StdMutex;

        use arcgraph_community::RefreshObserver;

        #[derive(Debug)]
        struct CountingObserver {
            calls: StdMutex<Vec<TenantId>>,
        }

        impl RefreshObserver for CountingObserver {
            fn record_refresh_success(&self, tenant: TenantId) {
                self.calls
                    .lock()
                    .expect("observer lock poisoned (test bug)")
                    .push(tenant);
            }
        }

        let (catalog, crud, mgr) = fixture();
        install(&crud, &mgr, TenantId::DEFAULT, 4);

        let observer = Arc::new(CountingObserver {
            calls: StdMutex::new(Vec::new()),
        });
        let mut cfg = EngineConfig::new(
            Arc::clone(&catalog),
            Arc::clone(&crud),
            Arc::clone(&mgr),
            CommunityIndexId::new(1),
        );
        cfg.scheduler_config = SchedulerConfig {
            interval: std::time::Duration::from_secs(3600),
            max_tick_duration: std::time::Duration::from_secs(60),
            initial_install_lsn: arcgraph_core::Lsn::new(1_000),
        };
        cfg.refresh_observer = Some(Arc::clone(&observer) as Arc<dyn RefreshObserver>);
        let handles = bootstrap_engine(cfg).expect("bootstrap");

        // No refresh has run yet — observer silent.
        assert!(observer.calls.lock().expect("lock").is_empty());

        handles.scheduler.tick();

        // Exactly one successful refresh (DEFAULT), exactly one
        // observer notification.
        let h = handles.scheduler.health_check();
        assert_eq!(h.total_refresh_failures, 0);
        assert_eq!(h.total_soft_skips, 0);
        assert_eq!(
            *observer.calls.lock().expect("lock"),
            vec![TenantId::DEFAULT],
            "observer must fire once for the refreshed tenant"
        );

        handles.scheduler.shutdown();
    }

    #[test]
    fn bootstrap_router_community_handle_is_same_index_as_hook() {
        // The provider, hook, and scheduler must all coordinate
        // through the SAME `Arc<BTreeMembershipIndex>` so the
        // scheduler's install_into is observable through the
        // router's TenantHandle::community().
        let (catalog, crud, mgr) = fixture();
        install(&crud, &mgr, TenantId::DEFAULT, 3);

        let cfg = EngineConfig::new(
            Arc::clone(&catalog),
            Arc::clone(&crud),
            Arc::clone(&mgr),
            CommunityIndexId::new(1),
        );
        let handles = bootstrap_engine(cfg).expect("bootstrap");

        let route_handle = handles
            .router
            .route(TenantId::DEFAULT, PartitionId::ZERO)
            .expect("route");
        let community_handle = route_handle.community().expect("community handle");
        assert_eq!(community_handle.tenant(), TenantId::DEFAULT);
        assert_eq!(community_handle.partition(), PartitionId::ZERO);

        // Pin: the membership index Arc on `EngineHandles` must
        // be ptr-equal to the one held inside the provider.
        assert!(Arc::ptr_eq(
            &handles.membership_index,
            handles.community_provider.index()
        ));

        handles.scheduler.shutdown();
    }
}
