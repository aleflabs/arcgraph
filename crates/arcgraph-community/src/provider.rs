//! Per-tenant `CommunityIndexHandle` provider trait per ADR-040
//! §D-3 + §D-8 Q1.
//!
//! `arcgraph-storage`'s `MultiTenantRouter` holds an
//! `Option<Arc<dyn CommunityIndexProvider>>`; on `route()` (per
//! ADR-037 §D-1) the router consults the provider for a tenant-
//! specific [`CommunityIndexHandle`] which is then stored on the
//! `TenantHandle` for downstream call sites.
//!
//! Why a provider trait rather than a single shared handle (like
//! `arcgraph-vector`'s `Arc<dyn VectorPageStoreHandle>`)?
//! [`CommunityIndexHandle`] carries `tenant_id` baked-in per
//! ADR-040 §D-3 (so `membership(node)` is a tenant-scoped call
//! without an explicit context argument). Cloning one
//! `Arc<CommunityIndexHandle>` into every TenantHandle would
//! assign the wrong tenant; the provider abstraction lets the
//! storage layer construct the correct per-tenant handle once and
//! cache it inside the router's `handle_cache`.
//!
//! ## Production impl: [`SharedBTreeIndexProvider`]
//!
//! v1.0 ships one production [`CommunityIndexProvider`] impl,
//! [`SharedBTreeIndexProvider`], promoted from the M3.c integration
//! test fixture per ADR-040 amendment-04 (codex F-2 closure). The
//! production posture differs from the test fixture in one detail:
//! the production provider drops the test-only `populated` set and
//! instead **always returns `Some(handle)` for every tenant**. The
//! handle's `Ok(None)` orphan-node semantics in
//! `MembershipIndex::lookup` (per ADR-041 §D-3b) cover the
//! "tenant has no community state yet" case at the lookup layer
//! rather than at the provider layer.
//!
//! The "absent" arm of the router's
//! `Option<Arc<dyn CommunityIndexProvider>>` field — the case where
//! the engine boots with no community provider wired — remains
//! covered by passing `None` to `MultiTenantRouterBuilder` (and is
//! exercised by the `tests::NoneProvider` cfg(test) impl below).

use std::sync::Arc;

use arcgraph_core::{PartitionId, TenantId};

use crate::handle::CommunityIndexHandle;
use crate::ids::CommunityIndexId;
use crate::membership_index::BTreeMembershipIndex;

/// Factory for per-tenant [`CommunityIndexHandle`]s.
///
/// Implementors MUST enforce per-tenant isolation: a query for
/// `tenant_a` MUST return a handle whose `.tenant() == tenant_a`
/// (or `None`); cross-tenant handle returns are an
/// I-V2-equivalent invariant violation per ADR-040 §D-3.
pub trait CommunityIndexProvider: Send + Sync {
    /// Return the handle for `(tenant, partition)`, or `None` if
    /// the tenant has no community index allocated.
    fn handle_for(
        &self,
        tenant: TenantId,
        partition: PartitionId,
    ) -> Option<Arc<CommunityIndexHandle>>;
}

/// v1.0 production [`CommunityIndexProvider`] backed by a single
/// shared [`BTreeMembershipIndex`].
///
/// The single `Arc<BTreeMembershipIndex>` is workspace-scoped per
/// ADR-040 §D-4 — the B-tree's keying is tenant-keyed at the high-
/// order column, so a single shared index transparently serves all
/// tenants without the per-tenant per-call materialisation cost
/// that a `DashMap<TenantId, BTreeMembershipIndex>` shape would
/// incur.
///
/// ## Production posture: always return `Some(handle)`
///
/// `handle_for` always returns `Some(handle)` for any tenant. A
/// caller looking up an as-yet-unpopulated tenant gets a handle
/// whose [`MembershipIndex::lookup`] returns `Ok(None)` for orphan
/// nodes (per ADR-041 §D-3b: a `read_lsn` strictly less than the
/// earliest install for the `(tenant, level)` is the canonical
/// "no community state yet" answer). This matches the production
/// invariant: the engine wires a single global community provider
/// at boot; per-tenant data arrives over time as scheduler ticks
/// install assignments. There is no admission gate at the provider
/// layer — admission is at the lookup layer.
///
/// The legacy "absent" router arm
/// (`Option<Arc<dyn CommunityIndexProvider>>::None`) covers the
/// orthogonal case of "the engine was bootstrapped without a
/// community provider at all"; pass `None` to
/// [`MultiTenantRouterBuilder::community`] for that posture.
///
/// [`MembershipIndex::lookup`]: crate::index::MembershipIndex::lookup
/// [`MultiTenantRouterBuilder::community`]:
///     https://docs.rs/arcgraph-storage/latest/arcgraph_storage/struct.MultiTenantRouterBuilder.html#method.community
#[derive(Clone)]
pub struct SharedBTreeIndexProvider {
    index: Arc<BTreeMembershipIndex>,
    index_id: CommunityIndexId,
}

impl SharedBTreeIndexProvider {
    /// Construct a provider with a fresh empty
    /// [`BTreeMembershipIndex`]. The caller installs per-tenant
    /// assignments via [`Self::index`] (e.g.,
    /// `GveLeiden::install_into(&result, provider.index(), tenant,
    /// install_lsn, n_skip_prefix)`).
    #[must_use]
    pub fn new(index_id: CommunityIndexId) -> Self {
        Self {
            index: Arc::new(BTreeMembershipIndex::new()),
            index_id,
        }
    }

    /// Construct a provider over a caller-supplied shared index.
    /// Used when the caller needs to install assignments into the
    /// same index from outside the provider — for example, the
    /// scheduler's [`crate::RefreshHook`] borrows the same shared
    /// index when it overwrites per-tenant levels on each tick.
    #[must_use]
    pub fn with_index(index: Arc<BTreeMembershipIndex>, index_id: CommunityIndexId) -> Self {
        Self { index, index_id }
    }

    /// The shared index. Cheap-cloneable `Arc`; callers may hold
    /// the clone for the duration of an install or expose it to
    /// the [`crate::CommunityRefreshScheduler`]'s [`crate::RefreshHook`].
    #[must_use]
    pub fn index(&self) -> &Arc<BTreeMembershipIndex> {
        &self.index
    }

    /// The catalog-allocated [`CommunityIndexId`] threaded through
    /// every materialised handle.
    #[must_use]
    pub fn index_id(&self) -> CommunityIndexId {
        self.index_id
    }
}

impl CommunityIndexProvider for SharedBTreeIndexProvider {
    fn handle_for(
        &self,
        tenant: TenantId,
        partition: PartitionId,
    ) -> Option<Arc<CommunityIndexHandle>> {
        // Local-only: we serve PartitionId::ZERO.
        // The router enforces this via `PartitionNotSupported`,
        // but the provider is defensive: a non-zero partition arrival
        // here is a router-side bug.
        debug_assert_eq!(partition, PartitionId::ZERO);
        Some(Arc::new(CommunityIndexHandle::for_tenant(
            tenant,
            self.index_id,
            // The trait object Arc clones cheaply; the underlying
            // tree is the single shared `BTreeMembershipIndex`,
            // and the per-tenant projection lives inside the index's
            // tenant-keyed B-tree.
            self.index.clone(),
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A provider that returns `None` for every tenant; mirrors the
    /// "absent" arm of the `Option<Arc<dyn CommunityIndexProvider>>`
    /// router field.
    struct NoneProvider;
    impl CommunityIndexProvider for NoneProvider {
        fn handle_for(
            &self,
            _tenant: TenantId,
            _partition: PartitionId,
        ) -> Option<Arc<CommunityIndexHandle>> {
            None
        }
    }

    #[test]
    fn provider_trait_is_object_safe() {
        // Compile-time check: the trait is object-safe so the
        // router's `Arc<dyn CommunityIndexProvider>` field
        // compiles.
        let _p: Arc<dyn CommunityIndexProvider> = Arc::new(NoneProvider);
    }

    #[test]
    fn none_provider_returns_none_for_every_tenant() {
        let p = NoneProvider;
        assert!(p.handle_for(TenantId::DEFAULT, PartitionId::ZERO).is_none());
        assert!(p.handle_for(TenantId::SYSTEM, PartitionId::ZERO).is_none());
        assert!(p.handle_for(TenantId::new(42), PartitionId::ZERO).is_none());
    }

    #[test]
    fn shared_btree_provider_is_object_safe() {
        let _p: Arc<dyn CommunityIndexProvider> =
            Arc::new(SharedBTreeIndexProvider::new(CommunityIndexId::new(1)));
    }

    #[test]
    fn shared_btree_provider_returns_some_for_any_tenant() {
        // v1.0 production posture: handle_for returns Some(handle)
        // for every tenant. The handle's MembershipIndex::lookup
        // returns Ok(None) for tenants with no installed levels.
        let provider = SharedBTreeIndexProvider::new(CommunityIndexId::new(7));
        let h_default = provider
            .handle_for(TenantId::DEFAULT, PartitionId::ZERO)
            .expect("handle_for(DEFAULT)");
        let h_system = provider
            .handle_for(TenantId::SYSTEM, PartitionId::ZERO)
            .expect("handle_for(SYSTEM)");
        let h_42 = provider
            .handle_for(TenantId::new(42), PartitionId::ZERO)
            .expect("handle_for(new(42))");

        // Per-tenant isolation invariant per ADR-040 §D-3.
        assert_eq!(h_default.tenant(), TenantId::DEFAULT);
        assert_eq!(h_system.tenant(), TenantId::SYSTEM);
        assert_eq!(h_42.tenant(), TenantId::new(42));

        // The catalog-allocated index id threads through every
        // materialised handle.
        assert_eq!(h_default.index_id(), CommunityIndexId::new(7));
        assert_eq!(h_system.index_id(), CommunityIndexId::new(7));
        assert_eq!(h_42.index_id(), CommunityIndexId::new(7));
    }

    #[test]
    fn shared_btree_provider_with_index_shares_state() {
        // Two providers constructed via with_index over the same
        // Arc<BTreeMembershipIndex> see the same handle state.
        let shared = Arc::new(BTreeMembershipIndex::new());
        let p1 =
            SharedBTreeIndexProvider::with_index(Arc::clone(&shared), CommunityIndexId::new(1));
        let p2 =
            SharedBTreeIndexProvider::with_index(Arc::clone(&shared), CommunityIndexId::new(1));
        // Same Arc — cheap Arc::ptr_eq check on the index field.
        assert!(Arc::ptr_eq(p1.index(), p2.index()));
    }

    // ─── F-2 codex round-2 deeper coverage (PR #218 fix-up) ────
    //
    // The five tests above pin object-safety + factory identity +
    // Arc-pointer-equality on `with_index`. The three tests below
    // pin the production posture's load-bearing semantics that the
    // codex round-1 review (2026-05-04) flagged as missing:
    //  - concurrent reader safety (production hot path is multi-
    //    threaded route() consumers; ADR-040 §D-4)
    //  - orphan-tenant `Ok(None)` lookup (ADR-040 amendment-04 D-3
    //    "always-return-Some at provider; admission at lookup")
    //  - lifecycle Arc-strong-count reclamation (handle drop
    //    releases the cloned `Arc<BTreeMembershipIndex>` ref)

    /// Pins: under concurrent `handle_for` calls from multiple
    /// threads, every call returns `Some(handle)` with the requested
    /// tenant identity preserved, and every returned handle holds an
    /// `Arc<BTreeMembershipIndex>` clone of the provider's shared
    /// index (verified via `Arc::strong_count` after all threads
    /// join). Per ADR-040 §D-4 the v1.0 single shared index serves
    /// every tenant; the provider does NOT cache handles (the router
    /// does, per `provider.rs` module doc), so each call constructs a
    /// fresh `Arc<CommunityIndexHandle>` — the consistency invariant
    /// is on the underlying index Arc, not on the handle Arc.
    #[test]
    fn provider_handle_for_under_concurrent_readers_returns_consistent_handle() {
        let shared = Arc::new(BTreeMembershipIndex::new());
        let provider = Arc::new(SharedBTreeIndexProvider::with_index(
            Arc::clone(&shared),
            CommunityIndexId::new(11),
        ));

        // Baseline: probe + provider's internal Arc clone = 2.
        let baseline = Arc::strong_count(&shared);
        assert_eq!(
            baseline, 2,
            "probe + provider.index() should be the only Arc refs at baseline",
        );

        let n_threads = 4;
        let calls_per_thread = 32;
        let mut joins = Vec::with_capacity(n_threads);
        for thread_idx in 0..n_threads {
            let p = Arc::clone(&provider);
            let h = std::thread::spawn(move || {
                let mut local = Vec::with_capacity(calls_per_thread);
                for i in 0..calls_per_thread {
                    let raw = (thread_idx * calls_per_thread + i + 1) as u64;
                    let tenant = TenantId::new(raw);
                    let handle = p
                        .handle_for(tenant, PartitionId::ZERO)
                        .expect("handle_for must return Some under concurrent load");
                    assert_eq!(
                        handle.tenant(),
                        tenant,
                        "tenant identity must be preserved under concurrent calls",
                    );
                    assert_eq!(handle.index_id(), CommunityIndexId::new(11));
                    local.push(handle);
                }
                local
            });
            joins.push(h);
        }

        // Collect every handle so the Arc refs stay live for the
        // strong-count assertion.
        let mut all_handles = Vec::with_capacity(n_threads * calls_per_thread);
        for h in joins {
            all_handles.extend(h.join().expect("worker thread must not panic"));
        }

        // After join: probe + provider.index() + every in-flight
        // handle's clone = 2 + (n_threads * calls_per_thread).
        let in_flight = n_threads * calls_per_thread;
        assert_eq!(
            Arc::strong_count(&shared),
            baseline + in_flight,
            "every concurrent handle must hold an Arc clone of the shared index",
        );

        // Drop every handle; count must collapse back to baseline.
        drop(all_handles);
        assert_eq!(
            Arc::strong_count(&shared),
            baseline,
            "dropping all concurrent handles must release every Arc ref",
        );
    }

    /// Pins the load-bearing v1.0 production posture per ADR-040
    /// amendment-04 D-3: `handle_for` always returns `Some(handle)`,
    /// and orphan-node / orphan-tenant lookups return `Ok(None)` at
    /// the lookup layer (per ADR-041 §D-3b). Two scenarios:
    ///
    /// (a) **Orphan node, populated tenant.** Tenant T1 has installed
    ///     assignments for nodes 1..3; lookup of node 9999 (not in
    ///     the install) returns `Ok(None)`, NOT an error or panic.
    ///
    /// (b) **Orphan tenant.** Tenant T2 has NO installs at all;
    ///     `handle_for(T2)` still returns `Some(handle)`, and a
    ///     subsequent lookup of any node returns `Ok(None)`. This
    ///     pins the load-bearing semantic of dropping the test-
    ///     fixture `populated` admission gate (per amendment-04
    ///     §3.3 Negative).
    #[test]
    fn provider_handle_for_orphan_node_lookup_returns_ok_none_not_panic() {
        use crate::ids::{CommunityId, Level};
        use arcgraph_core::{Lsn, NodeId};

        let shared = Arc::new(BTreeMembershipIndex::new());
        let provider =
            SharedBTreeIndexProvider::with_index(Arc::clone(&shared), CommunityIndexId::new(7));

        // Install one snapshot for T1: nodes 1, 2, 3 → community 0.
        let t1 = TenantId::new(101);
        shared.install_level(
            t1,
            Level::FINEST,
            Lsn::new(10),
            &[
                (NodeId::new(1), CommunityId::new(0)),
                (NodeId::new(2), CommunityId::new(0)),
                (NodeId::new(3), CommunityId::new(0)),
            ],
        );

        // Scenario (a): orphan-node lookup against populated T1.
        let h1 = provider
            .handle_for(t1, PartitionId::ZERO)
            .expect("handle_for(T1) must return Some(handle)");
        let unknown = h1
            .membership(NodeId::new(9999), Level::FINEST, Lsn::MAX)
            .expect("orphan-node lookup must not error");
        assert!(
            unknown.is_none(),
            "orphan-node lookup must return Ok(None), got {unknown:?}",
        );

        // Sanity: the populated node still resolves.
        let known = h1
            .membership(NodeId::new(2), Level::FINEST, Lsn::MAX)
            .expect("populated lookup must not error");
        assert_eq!(
            known,
            Some(CommunityId::new(0)),
            "populated node must resolve to its community",
        );

        // Scenario (b): orphan-tenant lookup against unpopulated T2.
        // The v1.0 production posture demands Some(handle) here even
        // though T2 has zero installs; the handle's lookup returns
        // Ok(None) (per ADR-041 §D-3b: read_lsn predates every
        // install for (tenant, level) is the canonical "no community
        // state yet" answer).
        let t2 = TenantId::new(202);
        let h2 = provider
            .handle_for(t2, PartitionId::ZERO)
            .expect("handle_for(T2) must return Some(handle) even without installs");
        let orphan_tenant = h2
            .membership(NodeId::new(1), Level::FINEST, Lsn::MAX)
            .expect("orphan-tenant lookup must not error");
        assert!(
            orphan_tenant.is_none(),
            "orphan-tenant lookup must return Ok(None) — the v1.0 production \
             posture per ADR-040 amendment-04 D-3 (always-Some at provider; \
             admission at lookup), got {orphan_tenant:?}",
        );
    }

    /// Pins the lifecycle invariant: `handle_for` clones the
    /// provider's `Arc<BTreeMembershipIndex>` into the returned
    /// handle's `membership: Arc<dyn MembershipIndex>` field; dropping
    /// the handle decrements the strong count; dropping the provider
    /// drops the last in-provider reference. We hold an external
    /// probe `Arc` to sample `Arc::strong_count` at each step.
    /// No leak under construct → take handle → drop handle → drop
    /// provider.
    #[test]
    fn provider_handle_lifecycle_drop_releases_resources() {
        let probe = Arc::new(BTreeMembershipIndex::new());
        let baseline = Arc::strong_count(&probe);
        assert_eq!(baseline, 1, "probe is the only Arc ref before provider");

        let provider =
            SharedBTreeIndexProvider::with_index(Arc::clone(&probe), CommunityIndexId::new(1));
        assert_eq!(
            Arc::strong_count(&probe),
            baseline + 1,
            "provider holds one Arc clone of the index",
        );

        let h1 = provider
            .handle_for(TenantId::DEFAULT, PartitionId::ZERO)
            .expect("handle_for(DEFAULT) must return Some(handle)");
        assert_eq!(
            Arc::strong_count(&probe),
            baseline + 2,
            "h1 holds an additional Arc clone via handle.membership",
        );

        let h2 = provider
            .handle_for(TenantId::SYSTEM, PartitionId::ZERO)
            .expect("handle_for(SYSTEM) must return Some(handle)");
        assert_eq!(
            Arc::strong_count(&probe),
            baseline + 3,
            "h2 holds an additional Arc clone via handle.membership",
        );

        drop(h1);
        assert_eq!(
            Arc::strong_count(&probe),
            baseline + 2,
            "dropping h1 must release its Arc ref to the index",
        );

        drop(h2);
        assert_eq!(
            Arc::strong_count(&probe),
            baseline + 1,
            "dropping h2 must release its Arc ref to the index",
        );

        drop(provider);
        assert_eq!(
            Arc::strong_count(&probe),
            baseline,
            "dropping the provider must release the last in-provider Arc ref; \
             only the external probe remains",
        );
    }
}
