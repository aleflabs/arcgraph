//! Production [`arcgraph_community::RefreshHook`] impl (M3.d-3
//! sub-task B; per-tick re-mat per ADR-040 amendment-05).
//!
//! Closes the residual "0 production impls" concern from F-2
//! research §1.2 by introducing the v1.0 production caller of
//! [`arcgraph_community::CommunityRefreshScheduler`]. After this
//! module lands, the scheduler has 1 production hook
//! ([`ProductionRefreshHook`]) on top of the existing 5 test impls.
//!
//! ## Per-tick re-materialisation (ADR-040 amendment-05)
//!
//! The trait surface (post amendment-05) is:
//!
//! ```text
//! fn resolve(&self, tenant: TenantId) -> Option<OwnedRefreshInputs>
//! ```
//!
//! returning `Arc<Graph>` + `Arc<BTreeMembershipIndex>`. The `Arc`
//! shape decouples the per-tenant Graph's lifetime from `&self`,
//! which lets the production hook re-materialise a fresh `Graph`
//! from `CrudStore` per scheduler tick — the canonical reset runs
//! against the *current* substrate state, not a frozen-at-bootstrap
//! snapshot.
//!
//! ## Why amendment-05 retired the v1.0 FROZEN-GRAPH posture
//!
//! Pre-amendment-05 (PR #235), this hook held a
//! `HashMap<TenantId, Box<Graph>>` populated at engine bootstrap;
//! `resolve()` returned `&'a Box<Graph>` whose `'a` was tied to
//! `&'a self`. That posture honoured the determinism property of
//! ADR-040 §D-7's "canonical reset" but NOT the substrate-freshness
//! property — tenants with continuous ingest over multi-week
//! deployments saw their canonical reset diverge from live data
//! until the operator restarted the engine. PR #235 identified this as
//! a latent-corruption risk.
//!
//! Amendment-05 (Wave 9b Slice 4) retires the FROZEN-GRAPH posture
//! entirely. This module now materialises per-tenant Graphs from
//! `CrudStore` on each scheduler tick.
//!
//! ## Construction posture (amendment-05)
//!
//! 1. [`ProductionRefreshHook::new`] takes the shared
//!    [`CrudStoreGraphAdapter`] + the workspace's shared
//!    `Arc<BTreeMembershipIndex>` + `LeidenParams`. No per-tenant
//!    state is materialised at construction.
//! 2. [`ProductionRefreshHook::register_tenant`] adds the tenant to
//!    the eligible-tenants set. Materialisation is **deferred** to
//!    the first scheduler tick for that tenant.
//! 3. [`ProductionRefreshHook::warm_up`] is an explicit opt-in for
//!    operators who want to surface materialisation errors at boot
//!    time (rather than silently soft-skipping at the first tick);
//!    the default `bootstrap_engine` does NOT call `warm_up`.
//! 4. The hook's [`RefreshHook::resolve`] impl materialises
//!    on each call via the adapter, wraps the resulting `Graph` in
//!    `Arc::new`, and returns
//!    `OwnedRefreshInputs { graph, index, params, n_skip_prefix: 1 }`
//!    (the `n_skip_prefix = 1` is the
//!    [`super::graph_adapter::CrudStoreGraphAdapter`] convention —
//!    vertex 0 is the `NodeId::ZERO` sentinel).
//!
//! ## Materialisation failure → soft-skip
//!
//! If `adapter.materialize(tenant)` returns `Err(_)`, the hook logs
//! `tracing::error!` and returns `None` — the scheduler treats it
//! like any other soft-skip per `SchedulerHealth::total_soft_skips`.
//! Subsequent ticks retry. There is no exponential backoff at v1.0;
//! a tenant whose `CrudStore` is permanently corrupted will
//! soft-skip indefinitely. v1.1 may add a per-tenant
//! unhealthy-tenant-quarantine after N consecutive soft-skips.
//!
//! ## Diagnostic cache
//!
//! The hook holds a `DashMap<TenantId, Arc<Graph>>` keyed by
//! tenant: the `Arc` from the most recent `resolve()` call is
//! retained for diagnostic inspection (e.g., `arcgraph dump`,
//! health-check surfaces, correctness audits). The cache is NOT
//! consumed across ticks for staleness — each tick materialises
//! afresh. Memory cost: same as v1.0 FROZEN-GRAPH (one Graph per
//! tenant) but with one Arc indirection. v1.0 envelope per
//! `super::mod` rustdoc covers this. The cache is cleared on
//! `unregister_tenant`.

use std::sync::Arc;

use arcgraph_community::{
    BTreeMembershipIndex, Graph, LeidenParams, OwnedRefreshInputs, RefreshHook,
};
use arcgraph_core::TenantId;
use dashmap::DashMap;
use parking_lot::RwLock;
use thiserror::Error;
use tracing::{debug, error, info};

use super::graph_adapter::{CrudStoreGraphAdapter, GraphAdapterError};

/// Errors surfaced by [`ProductionRefreshHook::warm_up`].
///
/// Note: at runtime, `RefreshHook::resolve` does NOT surface
/// errors — materialisation failures log and return `None` (the
/// scheduler's soft-skip path). `warm_up` returns errors so
/// operators who opt into eager materialisation see boot-time
/// failures.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ProductionRefreshHookError {
    /// Graph adapter failed to materialise the tenant's Graph
    /// during eager `warm_up`.
    #[error("CrudStore→Graph adapter failed for tenant {tenant_raw}: {source}")]
    GraphAdapter {
        /// Raw u64 of the affected tenant.
        tenant_raw: u64,
        /// Underlying adapter error.
        #[source]
        source: GraphAdapterError,
    },
}

/// v1.0 production [`RefreshHook`] impl (per-tick re-mat per
/// ADR-040 amendment-05).
///
/// See module-level rustdoc for the rationale and the
/// pre-amendment-05 history (FROZEN-GRAPH retirement).
///
/// `ProductionRefreshHook` is `Send + Sync` (the trait requires
/// it) — `CrudStoreGraphAdapter` is `Clone` + `Send + Sync` (it
/// holds `Arc<CrudStore>` + `Arc<TxnManager>`); `Arc<BTreeMembership
/// Index>` is `Send + Sync`; `LeidenParams` is `Copy`; `RwLock<
/// BTreeSet<TenantId>>` and `DashMap<TenantId, Arc<Graph>>` are
/// `Send + Sync` by their crate definitions.
pub struct ProductionRefreshHook {
    adapter: CrudStoreGraphAdapter,
    membership: Arc<BTreeMembershipIndex>,
    leiden_params: LeidenParams,
    /// Set of tenants the hook is willing to resolve. Tenants
    /// outside this set return `None` (soft-skip). Mutated via
    /// [`Self::register_tenant`] / [`Self::unregister_tenant`].
    /// Read-mostly: written ~once at bootstrap; read once per
    /// `resolve()` call — `RwLock` is the right primitive.
    tenants: RwLock<std::collections::BTreeSet<TenantId>>,
    /// Diagnostic cache: per-tenant `Arc<Graph>` from the most
    /// recent `resolve()` call. Not consumed across ticks for
    /// staleness — each tick materialises afresh.
    last_materialised: DashMap<TenantId, Arc<Graph>>,
}

impl ProductionRefreshHook {
    /// Construct a new `ProductionRefreshHook` over the workspace's
    /// shared resources.
    ///
    /// `adapter` is the [`CrudStoreGraphAdapter`] that materialises
    /// per-tenant Graphs; `membership` is the same shared
    /// `Arc<BTreeMembershipIndex>` the workspace's
    /// [`arcgraph_community::SharedBTreeIndexProvider`] holds (so
    /// scheduler-driven `install_into` calls land in the same index
    /// `MultiTenantRouter::route()` reads through).
    ///
    /// No per-tenant state is materialised at construction; tenants
    /// are added via [`Self::register_tenant`] and materialisation
    /// is deferred to the first scheduler tick (or to an explicit
    /// [`Self::warm_up`] call).
    #[must_use]
    pub fn new(
        adapter: CrudStoreGraphAdapter,
        membership: Arc<BTreeMembershipIndex>,
        leiden_params: LeidenParams,
    ) -> Self {
        Self {
            adapter,
            membership,
            leiden_params,
            tenants: RwLock::new(std::collections::BTreeSet::new()),
            last_materialised: DashMap::new(),
        }
    }

    /// Register `tenant` as eligible for refresh.
    ///
    /// Idempotent. No-op if already registered. Materialisation is
    /// deferred to the first scheduler tick for `tenant`; callers
    /// who want eager materialisation should additionally invoke
    /// [`Self::warm_up`] for `tenant` after registering.
    pub fn register_tenant(&self, tenant: TenantId) {
        let inserted = self.tenants.write().insert(tenant);
        if inserted {
            debug!(
                tenant = tenant.raw(),
                "ProductionRefreshHook: registered tenant"
            );
        }
    }

    /// Unregister `tenant` so subsequent `resolve(tenant)` calls
    /// return `None` (soft-skip). Drops the diagnostic cache entry
    /// (if any).
    ///
    /// Idempotent.
    pub fn unregister_tenant(&self, tenant: TenantId) {
        let removed = self.tenants.write().remove(&tenant);
        self.last_materialised.remove(&tenant);
        if removed {
            debug!(
                tenant = tenant.raw(),
                "ProductionRefreshHook: unregistered tenant"
            );
        }
    }

    /// Eagerly materialise `tenant`'s Graph. Surfaces materialisation
    /// errors (rather than silently soft-skipping at the first
    /// tick).
    ///
    /// Use this to validate at engine startup that every registered
    /// tenant's substrate is reachable. Default `bootstrap_engine`
    /// does NOT call `warm_up`; call it explicitly if you want
    /// boot-time validation. `warm_up` materialises a fresh `Graph`
    /// regardless of registration state and unconditionally inserts
    /// it into the diagnostic cache; calling it for a non-registered
    /// tenant is therefore wasted work but harmless — the orphan
    /// cache entry is cleared by [`Self::unregister_tenant`] (whose
    /// `last_materialised.remove` runs unconditionally, so it drops
    /// orphan entries even for tenants that were never in the
    /// eligible set). To restrict `warm_up` to the registered set,
    /// iterate [`Self::registered_tenants`] yourself rather than
    /// calling defensively for arbitrary tenants.
    ///
    /// # Errors
    ///
    /// - [`ProductionRefreshHookError::GraphAdapter`] if the
    ///   underlying adapter call fails.
    pub fn warm_up(&self, tenant: TenantId) -> Result<(), ProductionRefreshHookError> {
        let (graph, snapshot_lsn) = self.adapter.materialize(tenant).map_err(|source| {
            ProductionRefreshHookError::GraphAdapter {
                tenant_raw: tenant.raw(),
                source,
            }
        })?;
        info!(
            tenant = tenant.raw(),
            snapshot_lsn = snapshot_lsn.raw(),
            n = graph.n(),
            "ProductionRefreshHook::warm_up materialised tenant"
        );
        self.last_materialised.insert(tenant, Arc::new(graph));
        Ok(())
    }

    /// Tenants currently registered with the hook. Returned in
    /// ascending `TenantId::raw()` order for deterministic test
    /// output.
    #[must_use]
    pub fn registered_tenants(&self) -> Vec<TenantId> {
        let tenants = self.tenants.read();
        tenants.iter().copied().collect()
    }

    /// Whether `tenant` is registered.
    #[must_use]
    pub fn has_tenant(&self, tenant: TenantId) -> bool {
        self.tenants.read().contains(&tenant)
    }

    /// Returns the most recently materialised `Arc<Graph>` for
    /// `tenant`, if any. Diagnostic only — the production
    /// `resolve()` path materialises afresh on each call and does
    /// NOT consume this cache for staleness avoidance.
    #[must_use]
    pub fn last_materialised(&self, tenant: TenantId) -> Option<Arc<Graph>> {
        self.last_materialised.get(&tenant).map(|r| Arc::clone(&r))
    }
}

impl RefreshHook for ProductionRefreshHook {
    fn resolve(&self, tenant: TenantId) -> Option<OwnedRefreshInputs> {
        // Soft-skip if the tenant wasn't registered with the hook.
        // BTreeSet::contains is O(log n); n is bounded by the
        // catalog tenant count (typically ≤ 100 at v1.0).
        if !self.tenants.read().contains(&tenant) {
            return None;
        }
        // Materialise fresh from CrudStore. On error, log and
        // soft-skip — the scheduler continues to the next tenant
        // per ADR-040 §D-7. The lying-LSN bug from PR #235 MED-1 /
        // issue #239 is gone post-amendment-05 because
        // `materialize` now drops the outer Tx and returns the
        // inner txn's actual snapshot.
        let (graph, snapshot_lsn) = match self.adapter.materialize(tenant) {
            Ok(pair) => pair,
            Err(e) => {
                error!(
                    tenant = tenant.raw(),
                    error = ?e,
                    "ProductionRefreshHook materialisation failed; soft-skipping"
                );
                return None;
            }
        };
        let graph = Arc::new(graph);
        debug!(
            tenant = tenant.raw(),
            snapshot_lsn = snapshot_lsn.raw(),
            n = graph.n(),
            "ProductionRefreshHook::resolve hit"
        );
        // Update diagnostic cache. Overwrites on each call;
        // last_materialised reflects the most recent tick's view.
        self.last_materialised.insert(tenant, Arc::clone(&graph));
        Some(OwnedRefreshInputs {
            graph,
            index: Arc::clone(&self.membership),
            params: self.leiden_params,
            // CrudStoreGraphAdapter sizes n = high_water + 1 so
            // vertex 0 is the NodeId::ZERO sentinel; install_into
            // skips it.
            n_skip_prefix: 1,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arcgraph_core::{LabelId, TypeId};

    use crate::buffer::BufferPool;
    use crate::catalog::SystemCatalog;
    use crate::crud::{self, CrudStore, PropertyData};
    use crate::io::InMemoryPageIo;
    use crate::transaction::TxnManager;

    fn make_store() -> (Arc<CrudStore>, Arc<TxnManager>) {
        let io = Arc::new(InMemoryPageIo::new());
        let pool = BufferPool::new(8, io);
        let mgr = TxnManager::new();
        let catalog = SystemCatalog::new();
        catalog.bootstrap(&pool, &mgr).expect("bootstrap");
        (Arc::new(CrudStore::new()), Arc::new(mgr))
    }

    fn install_minimal_topology(
        crud: &Arc<CrudStore>,
        mgr: &Arc<TxnManager>,
        tenant: TenantId,
        node_count: u64,
        edges: &[(u64, u64)],
    ) {
        let label = LabelId::new(1);
        let ty = TypeId::new(1);
        let mut tx = mgr.begin(tenant);
        let mut nids: Vec<arcgraph_core::NodeId> = Vec::with_capacity(node_count as usize);
        for _ in 0..node_count {
            nids.push(
                crud::create_node(crud, &mut tx, tenant, label, &PropertyData::Empty)
                    .expect("create_node"),
            );
        }
        for &(u, v) in edges {
            crud::create_rel(
                crud,
                &mut tx,
                tenant,
                nids[u as usize],
                nids[v as usize],
                ty,
                &PropertyData::Empty,
            )
            .expect("create_rel");
        }
        crud::commit(tx, crud).expect("commit");
    }

    #[test]
    fn register_tenant_makes_resolve_succeed() {
        let (crud, mgr) = make_store();
        install_minimal_topology(&crud, &mgr, TenantId::DEFAULT, 4, &[(0, 1), (2, 3)]);

        let adapter = CrudStoreGraphAdapter::new(Arc::clone(&crud), Arc::clone(&mgr));
        let membership = Arc::new(BTreeMembershipIndex::new());
        let hook = ProductionRefreshHook::new(adapter, membership, LeidenParams::default());
        hook.register_tenant(TenantId::DEFAULT);

        assert_eq!(hook.registered_tenants(), vec![TenantId::DEFAULT]);
        assert!(hook.has_tenant(TenantId::DEFAULT));
        assert!(!hook.has_tenant(TenantId::SYSTEM));

        // Resolve materialises afresh; should yield n = high_water + 1.
        let inputs = hook.resolve(TenantId::DEFAULT).expect("resolve");
        assert_eq!(inputs.graph.n(), 5, "n = high_water + 1 = 4 + 1");
        assert_eq!(inputs.n_skip_prefix, 1, "vertex 0 is NodeId::ZERO sentinel");

        // Diagnostic cache populated post-resolve.
        let cached = hook
            .last_materialised(TenantId::DEFAULT)
            .expect("diagnostic cache populated");
        assert_eq!(cached.n(), 5);

        // Resolving an unregistered tenant -> None.
        assert!(hook.resolve(TenantId::new(9999)).is_none());
    }

    #[test]
    fn unregister_drops_diagnostic_cache_and_returns_none() {
        let (crud, mgr) = make_store();
        install_minimal_topology(&crud, &mgr, TenantId::DEFAULT, 2, &[(0, 1)]);

        let adapter = CrudStoreGraphAdapter::new(crud, mgr);
        let membership = Arc::new(BTreeMembershipIndex::new());
        let hook = ProductionRefreshHook::new(adapter, membership, LeidenParams::default());
        hook.register_tenant(TenantId::DEFAULT);
        let _ = hook.resolve(TenantId::DEFAULT).expect("first resolve");
        assert!(hook.last_materialised(TenantId::DEFAULT).is_some());

        hook.unregister_tenant(TenantId::DEFAULT);
        assert!(!hook.has_tenant(TenantId::DEFAULT));
        assert!(hook.last_materialised(TenantId::DEFAULT).is_none());
        assert!(hook.resolve(TenantId::DEFAULT).is_none());
    }

    #[test]
    fn multiple_tenants_isolate_per_tenant_state() {
        let (crud, mgr) = make_store();
        install_minimal_topology(&crud, &mgr, TenantId::DEFAULT, 3, &[(0, 1), (1, 2), (0, 2)]);
        install_minimal_topology(&crud, &mgr, TenantId::SYSTEM, 2, &[(0, 1)]);

        let adapter = CrudStoreGraphAdapter::new(crud, mgr);
        let membership = Arc::new(BTreeMembershipIndex::new());
        let hook = ProductionRefreshHook::new(adapter, membership, LeidenParams::default());
        hook.register_tenant(TenantId::DEFAULT);
        hook.register_tenant(TenantId::SYSTEM);
        assert_eq!(
            hook.registered_tenants(),
            vec![TenantId::SYSTEM, TenantId::DEFAULT]
        );

        let i_default = hook.resolve(TenantId::DEFAULT).expect("resolve DEFAULT");
        let i_system = hook.resolve(TenantId::SYSTEM).expect("resolve SYSTEM");
        // DEFAULT triangle: n = 4, 2m = 6.
        assert_eq!(i_default.graph.n(), 4);
        assert!((i_default.graph.total_weight_2m() - 6.0).abs() < 1e-6);
        // SYSTEM single edge: n = 3, 2m = 2.
        assert_eq!(i_system.graph.n(), 3);
        assert!((i_system.graph.total_weight_2m() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn unregistered_tenant_resolves_to_none() {
        // Soft-skip path: scheduler treats `None` from the hook as
        // a soft skip and does not increment failures (per
        // `scheduler.rs::SchedulerHealth::total_soft_skips`).
        let (crud, mgr) = make_store();
        let adapter = CrudStoreGraphAdapter::new(crud, mgr);
        let membership = Arc::new(BTreeMembershipIndex::new());
        let hook = ProductionRefreshHook::new(adapter, membership, LeidenParams::default());
        // No register_tenant calls.
        assert!(hook.resolve(TenantId::DEFAULT).is_none());
        assert!(hook.resolve(TenantId::SYSTEM).is_none());
    }

    #[test]
    fn warm_up_eagerly_materialises_and_populates_cache() {
        let (crud, mgr) = make_store();
        install_minimal_topology(&crud, &mgr, TenantId::DEFAULT, 4, &[(0, 1), (2, 3)]);

        let adapter = CrudStoreGraphAdapter::new(crud, mgr);
        let membership = Arc::new(BTreeMembershipIndex::new());
        let hook = ProductionRefreshHook::new(adapter, membership, LeidenParams::default());
        hook.register_tenant(TenantId::DEFAULT);

        // Pre-warm-up the diagnostic cache is empty.
        assert!(hook.last_materialised(TenantId::DEFAULT).is_none());

        hook.warm_up(TenantId::DEFAULT).expect("warm_up");
        let cached = hook
            .last_materialised(TenantId::DEFAULT)
            .expect("post-warm-up cache populated");
        assert_eq!(cached.n(), 5);
    }

    #[test]
    fn refresh_hook_object_safety() {
        // Compile-time assertion: ProductionRefreshHook fits the
        // Arc<dyn RefreshHook> shape the scheduler expects.
        let (crud, mgr) = make_store();
        let adapter = CrudStoreGraphAdapter::new(crud, mgr);
        let membership = Arc::new(BTreeMembershipIndex::new());
        let hook = ProductionRefreshHook::new(adapter, membership, LeidenParams::default());
        let _: Arc<dyn RefreshHook> = Arc::new(hook);
    }
}
