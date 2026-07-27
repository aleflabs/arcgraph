//! Multi-tenant routing facade — M3.a Slice H (per ADR-037).
//!
//! [`MultiTenantRouter`] is the workspace-wide single dispatch
//! surface above the storage layer. Every M4 (ArcQL executor),
//! M5 (MCP tool), and M6 (embed) consumer routes through
//! [`MultiTenantRouter::route`] to obtain an [`Arc<TenantHandle>`]
//! that bundles the per-`(TenantId, PartitionId)` resources the
//! consumer needs (`CrudStore` for graph CRUD, optional
//! `VectorPageStoreHandle` for vector workloads, optional
//! per-tenant `CommunityIndexHandle` per ADR-040 §D-3, optional
//! `Bm25IndexStoreHandle` per ADR-039 §D-9).
//! Direct-to-`CrudStore` and direct-to-`VectorPageStoreHandle`
//! calls from above the storage layer are out of contract — the
//! router IS the seam between consumer and per-tenant projection.
//!
//! # ADR provenance
//!
//! - **ADR-037 (NEW; Slice H).** Pins the [`TenantHandle`] +
//!   [`MultiTenantRouter`] + [`RoutingError`] shapes verbatim
//!   (§D-1, §D-2, §D-3); the SYSTEM-tenant carve-out (§D-5) and
//!   v1.0 append-only cache policy (§D-6); and the variant-based,
//!   no-cost-model dispatch contract (§D-4).
//! - **ADR-011 + amendment-01.** Tenant model baseline. Every
//!   storage operation is scoped to a `TenantId`;
//!   [`SystemCatalog`] under `TenantId::SYSTEM` is the
//!   authoritative tenant list.
//! - **ADR-024 amendment-02.** Every consumer-surface signature
//!   accepts `(TenantId, PartitionId)`; the single-node engine
//!   accepts only [`PartitionId::ZERO`].
//! - **ADR-035 amendment-04.** Variant-based dispatch precedent
//!   (the F.4 selectivity dispatcher in
//!   `crates/arcgraph-vector/src/dispatcher.rs`). The same
//!   discipline applies one layer up: catalog-lookup-then-
//!   dispatch, no statistics, no histograms, no heuristics.
//! - **F.4 dispatcher pattern** (`crates/arcgraph-vector/src/
//!   dispatcher.rs`). Reference for variant-based dispatch and
//!   `thiserror`-typed enum errors.
//!
//! # Back-of-envelope (design-v2 §A.3)
//!
//! - `route()` = O(N_tenants) catalog scan + O(1) DashMap
//!   insert. v1.0 N_tenants is typically 1-3, so ~50 ns per
//!   first lookup.
//! - Lazy cache: each `(tenant, partition)` materialises once;
//!   subsequent `route()` calls are pure DashMap reads (~30 ns).
//! - Memory: O(N_tenants × N_partitions × sizeof(TenantHandle)).
//!   At v1.0 N_partitions = 1, so the cache footprint is
//!   negligible.
//! - Amortised into M4 query / M5 MCP request paths where the
//!   dominant cost is the storage operation itself (≥ μs).
//!
//! # What this module is NOT
//!
//! - **No cost model, no fairness, no load shedding.**
//! - **No tenant lifecycle.** CREATE / DROP / RENAME are out
//!   of scope at v1.0 (ADR-011 §"Neutral / Open Questions").
//!   The cache is therefore append-only (ADR-037 §D-6).
//! - **No authentication.** RBAC is upstream (ADR-011
//!   §"RBAC composition"). The router's contract is dispatch,
//!   not authorization.
//! - **No resource accounting.** Per-tenant memory / storage /
//!   IOPS limits hang on `TenantHandle` accessors at v1.1
//!   (ADR-037 §3.2); consumers do not change.

use std::fmt;
use std::sync::Arc;

use arcgraph_community::{CommunityIndexHandle, CommunityIndexProvider};
use arcgraph_core::{PartitionId, TenantId};
use dashmap::DashMap;

use crate::catalog::SystemCatalog;
use crate::crud::CrudStore;
use crate::mutation_log::Bm25IndexStoreHandle;
use crate::permissions::PermissionIndex;
use crate::vector_store::VectorPageStoreHandle;

// ─────────────────────────────────────────────────────────────────────
// RoutingError
// ─────────────────────────────────────────────────────────────────────

/// Faults surfaced by [`MultiTenantRouter::route`].
///
/// Per ADR-037 §D-3 + code-quality policy (`thiserror` for library
/// errors). The error carries raw `u64` / `u32` for
/// diagnosability without forcing consumers to import
/// `arcgraph-core` for the typed `TenantId` / `PartitionId`.
///
/// `#[non_exhaustive]` is deliberately omitted — ADR-037 §D-3
/// pins the variant set, and a future variant lands via
/// amendment with the corresponding consumer migration.
#[derive(Debug, thiserror::Error)]
pub enum RoutingError {
    /// Caller passed a [`TenantId`] that is not in
    /// [`SystemCatalog::list_tenants`] and is not the
    /// SYSTEM-tenant carve-out (ADR-037 §D-5).
    ///
    /// Reserved IDs 2..=99 (per `arcgraph-core::ids::TenantId`
    /// rustdoc) surface here at v1.0; a future built-in tenant
    /// in that range registers in the catalog like any other.
    #[error("unknown tenant: tenant_id={tenant_raw}")]
    UnknownTenant {
        /// Raw u64 from the caller's `TenantId`.
        tenant_raw: u64,
    },

    /// Caller passed a [`PartitionId`] other than
    /// [`PartitionId::ZERO`].
    ///
    /// The public engine routes only [`PartitionId::ZERO`].
    #[error(
        "partition routing not supported: partition_id={partition_raw} \
         (only PartitionId::ZERO is valid)"
    )]
    PartitionNotSupported {
        /// Raw u32 from the caller's `PartitionId`.
        partition_raw: u32,
    },
}

// ─────────────────────────────────────────────────────────────────────
// TenantHandle
// ─────────────────────────────────────────────────────────────────────

/// Per-`(TenantId, PartitionId)` ergonomic surface returned by
/// [`MultiTenantRouter::route`] (ADR-037 §D-1).
///
/// Bundles the storage-layer resources a consumer needs for a
/// single tenant + partition slot:
///
/// - [`Self::tenant`] / [`Self::partition`] — the routing key,
///   carried by value (`TenantId` + `PartitionId` are `Copy`).
/// - [`Self::crud`] — `Arc<CrudStore>` for graph CRUD. The Arc
///   is shared with all other handles in the same router; per-
///   tenant isolation lives inside [`CrudStore`]'s internal
///   `(TenantId, …)`-keyed maps.
/// - [`Self::vector`] — optional vector arena handle. `None` for
///   v1.0 deployments without vector workloads (matches
///   [`CrudStore::with_vector_store`]'s opt-in posture).
/// - [`Self::community`] — optional per-tenant community-detection
///   handle (ADR-040 §D-3 + ADR-037 §D-1). Materialised by the
///   router's [`CommunityIndexProvider`] at `route()` time;
///   `None` when the router was constructed without a provider
///   (mirrors the vector opt-in posture).
/// - [`Self::bm25`] — optional shared BM25 commit-side handle
///   (ADR-039 §D-9 + ADR-037 §3.3). `None` for v1.0 deployments
///   without text-search workloads (mirrors the vector opt-in
///   posture).
///
/// `TenantHandle` is constructed once by [`MultiTenantRouter`]
/// and is immutable thereafter — no setters, no `Default` impl,
/// no public constructor (per ADR-037 §D-1 final paragraph).
pub struct TenantHandle {
    tenant_id: TenantId,
    partition_id: PartitionId,
    crud: Arc<CrudStore>,
    vector: Option<Arc<dyn VectorPageStoreHandle>>,
    community: Option<Arc<CommunityIndexHandle>>,
    /// BM25 commit-side handle per ADR-039 §D-9 (activates ADR-037
    /// §3.3 reservation). `None` for deployments without text search.
    bm25: Option<Arc<dyn Bm25IndexStoreHandle>>,
    /// Per-tenant source-ACL permission index (ADR-212 §D-4;
    /// per-tenant resource add per ADR-037-amendment-02). ALWAYS
    /// present, never `Option` — an empty index is fail-closed
    /// (every doc UNCLASSIFIED ⇒ deny-all under principal-scoped
    /// enforcement), so no "permissions not wired" unsafe state
    /// exists.
    permissions: Arc<PermissionIndex>,
}

impl TenantHandle {
    /// Private constructor — only [`MultiTenantRouter::route`]
    /// builds handles. External construction would let consumers
    /// fabricate (tenant, partition) pairs that bypass the
    /// catalog lookup, defeating the routing seam.
    // The per-tenant resource set (crud / vector / community / bm25 /
    // permissions) is the handle's structural shape, not an ad-hoc
    // parameter list — each is an independent ADR-gated substrate
    // (ADR-037-amendment-02). #1221 (ADR-218) added `permissions`.
    #[allow(clippy::too_many_arguments)]
    fn new(
        tenant_id: TenantId,
        partition_id: PartitionId,
        crud: Arc<CrudStore>,
        vector: Option<Arc<dyn VectorPageStoreHandle>>,
        community: Option<Arc<CommunityIndexHandle>>,
        bm25: Option<Arc<dyn Bm25IndexStoreHandle>>,
        // #1221 (ADR-218): a pre-built `PermissionIndex` to adopt for this
        // tenant (the SAME `Arc` durable bootstrap wired into WAL replay,
        // so recovered ACLs are enforced by the served path). `None` ⇒
        // mint a fresh empty index (the v1.0 default; fail-closed).
        permissions: Option<Arc<PermissionIndex>>,
    ) -> Self {
        Self {
            tenant_id,
            partition_id,
            crud,
            vector,
            community,
            bm25,
            permissions: permissions.unwrap_or_else(|| Arc::new(PermissionIndex::new())),
        }
    }

    /// The tenant this handle was routed for.
    #[inline]
    #[must_use]
    pub fn tenant(&self) -> TenantId {
        self.tenant_id
    }

    /// The partition this handle was routed for. Always
    /// [`PartitionId::ZERO`] — the
    /// [`RoutingError::PartitionNotSupported`] guard in
    /// [`MultiTenantRouter::route`] enforces this for every
    /// successful route.
    #[inline]
    #[must_use]
    pub fn partition(&self) -> PartitionId {
        self.partition_id
    }

    /// Shared `CrudStore` handle. Returned as `&Arc<CrudStore>`
    /// so consumers can cheaply [`Arc::clone`] when they need
    /// to store the handle in a long-lived task; per-tenant
    /// projection happens inside `CrudStore`'s internal
    /// `(TenantId, …)`-keyed maps.
    #[inline]
    #[must_use]
    pub fn crud(&self) -> &Arc<CrudStore> {
        &self.crud
    }

    /// Optional shared vector arena handle. `None` for v1.0
    /// deployments without vector workloads (mirrors
    /// [`CrudStore::with_vector_store`]'s opt-in posture).
    #[inline]
    #[must_use]
    pub fn vector(&self) -> Option<&Arc<dyn VectorPageStoreHandle>> {
        self.vector.as_ref()
    }

    /// Optional per-tenant community-detection handle (ADR-040
    /// §D-3). `None` when the router was constructed without a
    /// [`CommunityIndexProvider`] (the
    /// [`MultiTenantRouter::new`] no-provider default), or when
    /// the wired provider returned `None` for this tenant.
    #[inline]
    #[must_use]
    pub fn community(&self) -> Option<&Arc<CommunityIndexHandle>> {
        self.community.as_ref()
    }

    /// Optional shared BM25 commit-side handle per ADR-039 §D-9
    /// (activates the ADR-037 §3.3 BM25 reservation). `None` for v1.0
    /// deployments without text-search workloads (mirrors
    /// [`CrudStore::with_bm25_store`]'s opt-in posture).
    ///
    /// The handle exposes the commit / rollback dispatch trait
    /// ([`crate::mutation_log::Bm25IndexStoreHandle`]); the search-
    /// side `Bm25IndexHandle` (in `arcgraph-bm25`) is obtained
    /// directly from `Bm25Service::handle(tenant, IndexId::DEFAULT_BM25)`
    /// at v1.0. Unification deferred to v1.1 per ADR-039 OPEN-Q-3.
    #[inline]
    #[must_use]
    pub fn bm25(&self) -> Option<&Arc<dyn Bm25IndexStoreHandle>> {
        self.bm25.as_ref()
    }

    /// Per-tenant source-ACL permission index (ADR-212 §D-4 +
    /// ADR-037-amendment-02). One instance per `(tenant, partition)`
    /// cache entry — the ingest seam writes through it and the
    /// retrieval seams resolve
    /// [`PermissionIndex::effective`](crate::permissions::PermissionIndex::effective)
    /// from it, so both sides observe the same index. Never `None`:
    /// an empty index denies everything under principal-scoped
    /// enforcement (fail-closed).
    #[inline]
    #[must_use]
    pub fn permissions(&self) -> &Arc<PermissionIndex> {
        &self.permissions
    }
}

// `Arc<dyn VectorPageStoreHandle>` / `Arc<dyn Bm25IndexStoreHandle>`
// are not `Debug` (the traits themselves do not require `Debug`), so
// the auto-derive on `TenantHandle` does not compile. Implement
// `Debug` manually with a placeholder for each trait object. The
// `community` field is `Arc<CommunityIndexHandle>` which IS
// `Debug`-able, but we render it as a placeholder for symmetry with
// `vector` to keep the Debug output stable across additive trait-
// object surfaces.
impl fmt::Debug for TenantHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TenantHandle")
            .field("tenant_id", &self.tenant_id)
            .field("partition_id", &self.partition_id)
            .field("crud", &"<Arc<CrudStore>>")
            .field(
                "vector",
                match self.vector {
                    Some(_) => &"Some(<dyn VectorPageStoreHandle>)",
                    None => &"None",
                },
            )
            .field(
                "community",
                match self.community {
                    Some(_) => &"Some(<Arc<CommunityIndexHandle>>)",
                    None => &"None",
                },
            )
            .field(
                "bm25",
                match self.bm25 {
                    Some(_) => &"Some(<dyn Bm25IndexStoreHandle>)",
                    None => &"None",
                },
            )
            .finish()
    }
}

// ─────────────────────────────────────────────────────────────────────
// MultiTenantRouter
// ─────────────────────────────────────────────────────────────────────

/// Single dispatch entry point for above-the-storage-layer
/// consumers (ADR-037 §D-2).
///
/// Holds the workspace-wide [`SystemCatalog`] / [`CrudStore`] /
/// optional [`VectorPageStoreHandle`], and a lazy
/// `(TenantId, PartitionId) → Arc<TenantHandle>` cache. The
/// routing rules (§D-4) are:
///
/// 1. Reject `partition != PartitionId::ZERO` with
///    [`RoutingError::PartitionNotSupported`]. This check
///    runs FIRST so a non-ZERO partition never causes a
///    spurious tenant-lookup miss.
/// 2. Allow `TenantId::SYSTEM` (ADR-037 §D-5) — SYSTEM is not
///    in [`SystemCatalog::list_tenants`], but catalog
///    bootstrap (ADR-011) and tier change (ADR-034 §I-D7)
///    paths route through it.
/// 3. Otherwise verify the tenant is in the catalog; absent →
///    [`RoutingError::UnknownTenant`].
/// 4. Memoise the materialised handle in `Self::handle_cache`
///    via `entry().or_insert_with(...)` — DashMap's per-key
///    bucket lock is the only synchronisation needed.
///
/// # v1.0 cache policy (append-only)
///
/// Per ADR-037 §D-6, tenants are static at v1.0 (M7 ships
/// CREATE only; DROP / RENAME are post-v1.0). The cache is
/// append-only; an entry, once materialised, lives until the
/// router is dropped. The footprint is bounded by tenant count
/// (expected dozens, not thousands).
///
/// v1.1's lifecycle slice must publish a tenant-dropped event
/// and specify whether outstanding `Arc<TenantHandle>` refs
/// "continue serving in-flight, fail new" or are cancelled
/// per-handle.
pub struct MultiTenantRouter {
    catalog: Arc<SystemCatalog>,
    crud: Arc<CrudStore>,
    vector: Option<Arc<dyn VectorPageStoreHandle>>,
    /// Optional factory for per-tenant
    /// [`arcgraph_community::CommunityIndexHandle`] (ADR-040 §D-3 +
    /// ADR-037 §D-1). When `Some`, [`Self::route`] consults the
    /// provider and stores the returned handle on
    /// [`TenantHandle::community`]. When `None`, every
    /// `TenantHandle::community()` is `None`.
    ///
    /// The community case differs from vector: a
    /// `CommunityIndexHandle` carries `tenant_id` baked-in (ADR-040
    /// §D-3), so a single shared `Arc<CommunityIndexHandle>`
    /// cannot be cloned into every `TenantHandle` the way
    /// vector's `Arc<dyn VectorPageStoreHandle>` is. The provider
    /// abstraction lets the storage layer construct the correct
    /// per-tenant handle once and cache it via the
    /// `handle_cache` DashMap.
    community_provider: Option<Arc<dyn CommunityIndexProvider>>,
    /// BM25 commit-side handle per ADR-039 §D-9. Optional like
    /// [`Self::vector`] — `None` for deployments without text search.
    /// Threaded into every materialised [`TenantHandle`].
    bm25: Option<Arc<dyn Bm25IndexStoreHandle>>,
    handle_cache: DashMap<(TenantId, PartitionId), Arc<TenantHandle>>,
    /// #1221 (ADR-218): pre-built per-tenant [`PermissionIndex`] overrides
    /// adopted at [`Self::route`] time (instead of minting a fresh empty
    /// index). Durable bootstrap creates the index BEFORE WAL recovery,
    /// wires the SAME `Arc` into the replay target (so recovered ACLs land
    /// in it) AND into this map (so the served `TenantHandle::permissions()`
    /// returns it) — closing the replay-vs-serve shared-index requirement.
    /// Empty in the v1.0 ephemeral / no-durability path (every tenant gets
    /// a fresh empty index, fail-closed).
    permissions_override: DashMap<TenantId, Arc<PermissionIndex>>,
}

// `Arc<dyn VectorPageStoreHandle>` / `Arc<dyn Bm25IndexStoreHandle>`
// are not `Debug` — same rationale as `TenantHandle`'s manual impl
// above. The `community_provider` is rendered as a placeholder for
// symmetry.
impl fmt::Debug for MultiTenantRouter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MultiTenantRouter")
            .field("catalog", &"<Arc<SystemCatalog>>")
            .field("crud", &"<Arc<CrudStore>>")
            .field(
                "vector",
                match self.vector {
                    Some(_) => &"Some(<dyn VectorPageStoreHandle>)",
                    None => &"None",
                },
            )
            .field(
                "community_provider",
                match self.community_provider {
                    Some(_) => &"Some(<dyn CommunityIndexProvider>)",
                    None => &"None",
                },
            )
            .field(
                "bm25",
                match self.bm25 {
                    Some(_) => &"Some(<dyn Bm25IndexStoreHandle>)",
                    None => &"None",
                },
            )
            .field("handle_cache_len", &self.handle_cache.len())
            .finish()
    }
}

impl MultiTenantRouter {
    /// Start a [`MultiTenantRouterBuilder`]. Equivalent to
    /// `MultiTenantRouterBuilder::new(catalog, crud)`. The canonical
    /// extension point for retained optional database substrates.
    #[must_use]
    pub fn builder(catalog: Arc<SystemCatalog>, crud: Arc<CrudStore>) -> MultiTenantRouterBuilder {
        MultiTenantRouterBuilder::new(catalog, crud)
    }

    /// Construct a router over the given catalog / CRUD /
    /// optional vector handle. The catalog SHOULD be bootstrapped
    /// before [`Self::route`] is called for any user tenant —
    /// `route()` panics inside [`SystemCatalog::list_tenants`] if
    /// the catalog is unbootstrapped (per catalog.rs:177). The
    /// SYSTEM-tenant carve-out is the documented exception:
    /// `route(TenantId::SYSTEM, _)` short-circuits the catalog
    /// lookup (ADR-037 §D-5).
    ///
    /// Thin-delegate via builder per ADR-037 amendment-01.
    ///
    /// **Signature stability.** This 3-arg constructor is preserved
    /// for backward compatibility with existing call sites (M3.a's
    /// nine integration pins in `tests/multi_tenant_routing.rs`,
    /// the single-partition routing conformance
    /// tests, the multi-tenant tier proptest, plus `arcgraph-mcp`
    /// wirings). Constructed without a [`CommunityIndexProvider`]
    /// or BM25 handle; every returned [`TenantHandle::community`]
    /// and [`TenantHandle::bm25`] is `None`. To wire community,
    /// use [`Self::with_community`]; to wire BM25, use
    /// [`Self::new_with_bm25`]; to wire both, use
    /// [`Self::new_with_community_and_bm25`]. New optional deps go
    /// on [`MultiTenantRouterBuilder`].
    #[must_use]
    pub fn new(
        catalog: Arc<SystemCatalog>,
        crud: Arc<CrudStore>,
        vector: Option<Arc<dyn VectorPageStoreHandle>>,
    ) -> Self {
        let mut b = Self::builder(catalog, crud);
        if let Some(v) = vector {
            b = b.vector(v);
        }
        b.build()
    }

    /// As [`Self::new`], with an optional community-detection
    /// provider per ADR-040 §D-3 + ADR-037 §D-1. The provider is
    /// the factory for per-tenant
    /// [`arcgraph_community::CommunityIndexHandle`] plumbed onto
    /// [`TenantHandle::community`] at [`Self::route`] time.
    ///
    /// Thin-delegate via builder per ADR-037 amendment-01.
    ///
    /// `new()` is the no-provider thin delegate; the additive
    /// `with_community` constructor preserves the v1.0 3-arg
    /// signature for existing call sites (per the M3.d-1 #2
    /// additive-only constraint) while exposing the new
    /// provider arg for community-aware engine wirings.
    #[must_use]
    pub fn with_community(
        catalog: Arc<SystemCatalog>,
        crud: Arc<CrudStore>,
        vector: Option<Arc<dyn VectorPageStoreHandle>>,
        community_provider: Option<Arc<dyn CommunityIndexProvider>>,
    ) -> Self {
        let mut b = Self::builder(catalog, crud);
        if let Some(v) = vector {
            b = b.vector(v);
        }
        if let Some(p) = community_provider {
            b = b.community(p);
        }
        b.build()
    }

    /// Construct a router with optional vector AND optional BM25
    /// commit-side handles per ADR-039 §D-9. Activates the ADR-037
    /// §3.3 BM25 reservation on every materialised
    /// [`TenantHandle`].
    ///
    /// Thin-delegate via builder per ADR-037 amendment-01.
    ///
    /// At v1.0, deployments that opt into text search call this
    /// constructor with `Some(bm25_service.into())` where
    /// `bm25_service: Arc<arcgraph_bm25::Bm25Service>`. The
    /// `Bm25Service` impls
    /// [`crate::mutation_log::Bm25IndexStoreHandle`] in
    /// `arcgraph-bm25/src/store.rs`.
    ///
    /// The 3-arg [`Self::new`] is preserved as a backward-compatible
    /// shorthand (`new_with_bm25(catalog, crud, vector, None)`).
    #[must_use]
    pub fn new_with_bm25(
        catalog: Arc<SystemCatalog>,
        crud: Arc<CrudStore>,
        vector: Option<Arc<dyn VectorPageStoreHandle>>,
        bm25: Option<Arc<dyn Bm25IndexStoreHandle>>,
    ) -> Self {
        let mut b = Self::builder(catalog, crud);
        if let Some(v) = vector {
            b = b.vector(v);
        }
        if let Some(b25) = bm25 {
            b = b.bm25(b25);
        }
        b.build()
    }

    /// Full surface — both M3.b BM25 and M3.d-1 community optional
    /// deps wired in a single constructor.
    ///
    /// Thin-delegate via builder per ADR-037 amendment-01. Preserved
    /// for backward compat with the post-#149 5-arg call sites
    /// (the integration pin
    /// `router_community_and_bm25_handles_both_plumbed_through` and
    /// any future call site that wants the consolidated shape). New
    /// optional deps (M4 query layer, v1.1+ PPR / cross-encoder)
    /// land as builder methods, not as new constructors.
    #[must_use]
    pub fn new_with_community_and_bm25(
        catalog: Arc<SystemCatalog>,
        crud: Arc<CrudStore>,
        vector: Option<Arc<dyn VectorPageStoreHandle>>,
        community_provider: Option<Arc<dyn CommunityIndexProvider>>,
        bm25: Option<Arc<dyn Bm25IndexStoreHandle>>,
    ) -> Self {
        let mut b = Self::builder(catalog, crud);
        if let Some(v) = vector {
            b = b.vector(v);
        }
        if let Some(p) = community_provider {
            b = b.community(p);
        }
        if let Some(b25) = bm25 {
            b = b.bm25(b25);
        }
        b.build()
    }

    /// Route to the [`TenantHandle`] for `(tenant, partition)`.
    ///
    /// See the type-level rustdoc for the dispatch rules.
    /// Returns the cached handle on the second and subsequent
    /// calls for the same key.
    ///
    /// # Errors
    ///
    /// - [`RoutingError::PartitionNotSupported`] when
    ///   `partition != PartitionId::ZERO`.
    /// - [`RoutingError::UnknownTenant`] when `tenant` is not
    ///   `TenantId::SYSTEM` and not in
    ///   [`SystemCatalog::list_tenants`].
    pub fn route(
        &self,
        tenant: TenantId,
        partition: PartitionId,
    ) -> Result<Arc<TenantHandle>, RoutingError> {
        // §1. Partition guard — reject non-ZERO partitions before
        //     tenant lookup.
        if partition != PartitionId::ZERO {
            return Err(RoutingError::PartitionNotSupported {
                partition_raw: partition.raw(),
            });
        }

        // §2. SYSTEM-tenant carve-out (ADR-037 §D-5). SYSTEM is
        //     not in `list_tenants`, but bootstrap / tier-change
        //     paths route through it. Skip the catalog lookup.
        // §3. Otherwise, the tenant must be in the catalog.
        if tenant != TenantId::SYSTEM {
            let tenants = self.catalog.list_tenants();
            if !tenants.iter().any(|r| r.tenant_id == tenant) {
                return Err(RoutingError::UnknownTenant {
                    tenant_raw: tenant.raw(),
                });
            }
        }

        // §4. Lazy materialisation. DashMap's
        //     `entry().or_insert_with(...)` holds a per-bucket
        //     lock for the duration of the closure — a concurrent
        //     `route()` call for a DIFFERENT key proceeds
        //     unblocked; for the SAME key, only one closure runs
        //     and the rest receive the same `Arc`.
        //
        //     The community handle is materialised inside the
        //     closure so the provider's `handle_for(tenant,
        //     partition)` call runs at most once per cache entry
        //     (ADR-040 §D-3 + ADR-037 §D-1). When no provider is
        //     wired, `community` is `None` for every tenant. The
        //     BM25 handle is a single shared `Arc` cloned per entry
        //     (ADR-039 §D-9), mirroring the vector posture.
        let entry = self
            .handle_cache
            .entry((tenant, partition))
            .or_insert_with(|| {
                let community = self
                    .community_provider
                    .as_ref()
                    .and_then(|p| p.handle_for(tenant, partition));
                // #1221 (ADR-218): adopt a pre-built PermissionIndex for
                // this tenant if durable bootstrap wired one (the SAME
                // `Arc` WAL replay populated), else mint a fresh empty
                // index (fail-closed default).
                let permissions = self
                    .permissions_override
                    .get(&tenant)
                    .map(|e| Arc::clone(e.value()));
                Arc::new(TenantHandle::new(
                    tenant,
                    partition,
                    Arc::clone(&self.crud),
                    self.vector.as_ref().map(Arc::clone),
                    community,
                    self.bm25.as_ref().map(Arc::clone),
                    permissions,
                ))
            });
        Ok(Arc::clone(entry.value()))
    }

    /// Snapshot the catalog's tenant list as `Vec<TenantId>`.
    ///
    /// Catalog passthrough — no router-side caching. ADR-037
    /// §D-2: encoding v1.0's static-list assumption into the
    /// router would lock in the wrong shape; at v1.1 the catalog
    /// owns invalidation.
    ///
    /// Note: `TenantId::SYSTEM` is NOT in the returned list
    /// (catalog.rs:175-183); SYSTEM is the catalog's own MVCC
    /// home, not a user-listable tenant.
    ///
    /// # Panics
    ///
    /// Panics if the underlying catalog has not been
    /// bootstrapped (per [`SystemCatalog::list_tenants`]). All
    /// production wirings bootstrap the catalog before
    /// constructing the router.
    #[must_use]
    pub fn tenants(&self) -> Vec<TenantId> {
        self.catalog
            .list_tenants()
            .into_iter()
            .map(|r| r.tenant_id)
            .collect()
    }

    /// The shared [`SystemCatalog`] the router routes against.
    ///
    /// Exposed (symmetric with [`Self::tenants`]) so a composition
    /// layer that already holds the router — e.g. the `arcgraph serve`
    /// binary's community-scheduler wiring (ADR-202 §D-8 / §Open
    /// questions: the serve-binary scheduler slice) — can build a
    /// scheduler over the SAME catalog the served substrate uses,
    /// without the bootstrap re-threading the handle through every
    /// return tuple. Read-only borrow; per-tenant projection happens
    /// inside the catalog's own `(TenantId, …)`-keyed maps.
    #[inline]
    #[must_use]
    pub fn catalog(&self) -> &Arc<SystemCatalog> {
        &self.catalog
    }

    /// The shared [`CrudStore`] the router's per-tenant handles
    /// project from. Returned as `&Arc<CrudStore>` so consumers can
    /// cheaply [`Arc::clone`] when they need the handle in a
    /// long-lived task (e.g. the community refresh scheduler's
    /// [`crate::engine::CrudStoreGraphAdapter`], which materialises
    /// per-tenant graphs from the SAME CRUD state the served write
    /// path commits into — ADR-202 §D-8 serve-binary scheduler slice).
    #[inline]
    #[must_use]
    pub fn crud(&self) -> &Arc<CrudStore> {
        &self.crud
    }
}

// ─────────────────────────────────────────────────────────────────────
// MultiTenantRouterBuilder
// ─────────────────────────────────────────────────────────────────────

/// Builder for [`MultiTenantRouter`]. Composes optional dependencies
/// without combinatorial constructor explosion (per ADR-037
/// amendment-01).
///
/// Required: `catalog`, `crud`.
/// Optional: `vector`, `community_provider`, `bm25` (all default to `None` at v1.0).
///
/// # Why a builder
///
/// The four existing `MultiTenantRouter` constructors (`new`,
/// `with_community`, `new_with_bm25`, `new_with_community_and_bm25`)
/// document the historical accumulation of optional deps:
/// PR #143 (3-arg `new`), PR #148 (`with_community`), PR #149
/// (`new_with_bm25` and the 5-arg consolidator). Each addition
/// forced the prior constructors to thin-delegate, and the 5-arg
/// shape was the forcing function for this builder.
///
/// Retained optional database substrates plug in through builder
/// methods without breaking existing call sites.
///
/// # Example
///
/// ```ignore
/// let router = MultiTenantRouter::builder(catalog, crud)
///     .vector(vector_handle)
///     .community(provider)
///     .bm25(bm25_handle)
///     .build();
/// ```
pub struct MultiTenantRouterBuilder {
    catalog: Arc<SystemCatalog>,
    crud: Arc<CrudStore>,
    vector: Option<Arc<dyn VectorPageStoreHandle>>,
    community_provider: Option<Arc<dyn CommunityIndexProvider>>,
    bm25: Option<Arc<dyn Bm25IndexStoreHandle>>,
    /// #1221 (ADR-218): pre-built per-tenant [`PermissionIndex`] overrides
    /// (see [`MultiTenantRouter::permissions_override`]). Empty by default.
    permissions_override: Vec<(TenantId, Arc<PermissionIndex>)>,
}

impl MultiTenantRouterBuilder {
    /// Start a new builder. `catalog` and `crud` are the only
    /// required deps; every other dep is opt-in via the
    /// `.fn()` setters.
    #[must_use]
    pub fn new(catalog: Arc<SystemCatalog>, crud: Arc<CrudStore>) -> Self {
        Self {
            catalog,
            crud,
            vector: None,
            community_provider: None,
            bm25: None,
            permissions_override: Vec::new(),
        }
    }

    /// #1221 (ADR-218): adopt a pre-built [`PermissionIndex`] for `tenant`
    /// instead of minting a fresh empty one at route time. Durable
    /// bootstrap uses this to share the WAL-replay-populated index with
    /// the served path. Repeated calls for the same tenant last-write-win.
    #[must_use]
    pub fn permissions(mut self, tenant: TenantId, index: Arc<PermissionIndex>) -> Self {
        self.permissions_override.push((tenant, index));
        self
    }

    /// Wire the optional vector arena handle (mirrors
    /// `CrudStore::with_vector_store`'s opt-in posture).
    #[must_use]
    pub fn vector(mut self, vector: Arc<dyn VectorPageStoreHandle>) -> Self {
        self.vector = Some(vector);
        self
    }

    /// Wire the optional [`CommunityIndexProvider`] per ADR-040 §D-3.
    #[must_use]
    pub fn community(mut self, provider: Arc<dyn CommunityIndexProvider>) -> Self {
        self.community_provider = Some(provider);
        self
    }

    /// Wire the optional BM25 commit-side handle per ADR-039 §D-9.
    #[must_use]
    pub fn bm25(mut self, bm25: Arc<dyn Bm25IndexStoreHandle>) -> Self {
        self.bm25 = Some(bm25);
        self
    }

    /// Materialise the [`MultiTenantRouter`]. The handle cache starts empty.
    #[must_use]
    pub fn build(self) -> MultiTenantRouter {
        MultiTenantRouter {
            catalog: self.catalog,
            crud: self.crud,
            vector: self.vector,
            community_provider: self.community_provider,
            bm25: self.bm25,
            handle_cache: DashMap::new(),
            permissions_override: self.permissions_override.into_iter().collect(),
        }
    }
}

// `Arc<dyn VectorPageStoreHandle>` / `Arc<dyn Bm25IndexStoreHandle>` /
// `Arc<dyn CommunityIndexProvider>` are not `Debug` (the traits
// themselves do not require `Debug`), so the auto-derive does not
// compile. Implement `Debug` manually with placeholders mirroring the
// `MultiTenantRouter` Debug impl above.
impl fmt::Debug for MultiTenantRouterBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MultiTenantRouterBuilder")
            .field("catalog", &"<Arc<SystemCatalog>>")
            .field("crud", &"<Arc<CrudStore>>")
            .field(
                "vector",
                match self.vector {
                    Some(_) => &"Some(<dyn VectorPageStoreHandle>)",
                    None => &"None",
                },
            )
            .field(
                "community_provider",
                match self.community_provider {
                    Some(_) => &"Some(<dyn CommunityIndexProvider>)",
                    None => &"None",
                },
            )
            .field(
                "bm25",
                match self.bm25 {
                    Some(_) => &"Some(<dyn Bm25IndexStoreHandle>)",
                    None => &"None",
                },
            )
            .finish()
    }
}

// ─────────────────────────────────────────────────────────────────────
// tests — module-local unit tests for shape; the integration
// suite lives in `tests/multi_tenant_routing.rs`.
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    use arcgraph_core::PageId;

    use crate::buffer::BufferPool;
    use crate::io::InMemoryPageIo;
    use crate::mutation_log::Bm25StoreError;
    use crate::transaction::TxnManager;
    use crate::vector_store::VectorStoreError;

    /// Mock `VectorPageStoreHandle` that no-ops both methods.
    /// Used to confirm `TenantHandle::vector()` plumbs the
    /// `Some(_)` arm through correctly.
    struct NoopVectorStore;

    impl VectorPageStoreHandle for NoopVectorStore {
        fn install_or_replace(
            &self,
            _tenant: TenantId,
            _page_id: PageId,
            _bytes: &[u8],
        ) -> std::result::Result<(), VectorStoreError> {
            Ok(())
        }

        fn restore_page_bytes(
            &self,
            _tenant: TenantId,
            _page_id: PageId,
            _bytes: &[u8],
        ) -> std::result::Result<(), VectorStoreError> {
            Ok(())
        }
    }

    /// Mock `Bm25IndexStoreHandle` that no-ops both methods. Used to
    /// confirm `TenantHandle::bm25()` plumbs the `Some(_)` arm through
    /// correctly per ADR-039 §D-9.
    struct NoopBm25Store;

    impl Bm25IndexStoreHandle for NoopBm25Store {
        fn commit_pending(&self, _tenant: TenantId) -> Result<(), Bm25StoreError> {
            Ok(())
        }

        fn rollback_pending(&self, _tenant: TenantId) -> Result<(), Bm25StoreError> {
            Ok(())
        }
    }

    fn make_router_no_vector() -> MultiTenantRouter {
        let io = Arc::new(InMemoryPageIo::new());
        let pool = BufferPool::new(8, io);
        let mgr = TxnManager::new();
        let catalog = Arc::new(SystemCatalog::new());
        catalog.bootstrap(&pool, &mgr).expect("bootstrap");
        let crud = Arc::new(CrudStore::new());
        MultiTenantRouter::new(catalog, crud, None)
    }

    #[test]
    fn route_default_returns_handle_with_correct_tenant_and_partition() {
        let router = make_router_no_vector();
        let h = router
            .route(TenantId::DEFAULT, PartitionId::ZERO)
            .expect("DEFAULT routes");
        assert_eq!(h.tenant(), TenantId::DEFAULT);
        assert_eq!(h.partition(), PartitionId::ZERO);
        // No vector wired -> None.
        assert!(h.vector().is_none());
    }

    #[test]
    fn route_system_short_circuits_catalog_lookup() {
        let router = make_router_no_vector();
        let h = router
            .route(TenantId::SYSTEM, PartitionId::ZERO)
            .expect("SYSTEM allowed (ADR-037 §D-5)");
        assert_eq!(h.tenant(), TenantId::SYSTEM);
    }

    #[test]
    fn route_unknown_tenant_errors() {
        let router = make_router_no_vector();
        let err = router
            .route(TenantId::new(9999), PartitionId::ZERO)
            .expect_err("unknown tenant must fail");
        assert!(matches!(
            err,
            RoutingError::UnknownTenant { tenant_raw: 9999 }
        ));
    }

    #[test]
    fn route_non_zero_partition_errors() {
        let router = make_router_no_vector();
        let err = router
            .route(TenantId::DEFAULT, PartitionId::new(1))
            .expect_err("non-zero partition must fail");
        assert!(matches!(
            err,
            RoutingError::PartitionNotSupported { partition_raw: 1 }
        ));
    }

    #[test]
    fn route_partition_check_runs_before_tenant_check() {
        // An UNKNOWN tenant + non-zero partition surfaces
        // PartitionNotSupported, not UnknownTenant — pinning
        // the §1-before-§3 ordering in `route()`.
        let router = make_router_no_vector();
        let err = router
            .route(TenantId::new(9999), PartitionId::new(2))
            .expect_err("non-zero partition wins over unknown tenant");
        assert!(matches!(
            err,
            RoutingError::PartitionNotSupported { partition_raw: 2 }
        ));
    }

    #[test]
    fn route_caches_handle_arc() {
        let router = make_router_no_vector();
        let h1 = router
            .route(TenantId::DEFAULT, PartitionId::ZERO)
            .expect("first route");
        let h2 = router
            .route(TenantId::DEFAULT, PartitionId::ZERO)
            .expect("second route");
        assert!(
            Arc::ptr_eq(&h1, &h2),
            "v1.0 cache hit must return the same Arc"
        );
    }

    #[test]
    fn vector_handle_plumbed_through_when_some() {
        let io = Arc::new(InMemoryPageIo::new());
        let pool = BufferPool::new(8, io);
        let mgr = TxnManager::new();
        let catalog = Arc::new(SystemCatalog::new());
        catalog.bootstrap(&pool, &mgr).expect("bootstrap");
        let crud = Arc::new(CrudStore::new());
        let vector: Arc<dyn VectorPageStoreHandle> = Arc::new(NoopVectorStore);
        let router = MultiTenantRouter::new(catalog, crud, Some(Arc::clone(&vector)));
        let h = router
            .route(TenantId::DEFAULT, PartitionId::ZERO)
            .expect("route");
        let h_vec = h.vector().expect("Some plumbed through");
        assert!(
            Arc::ptr_eq(h_vec, &vector),
            "router must clone the SAME Arc into the handle"
        );
    }

    // ─── ADR-039 §D-9: BM25 wiring ──────────────────────────────────

    #[test]
    fn route_default_returns_handle_with_no_bm25_when_unwired() {
        // The 3-arg `new` constructor is the v1.0 default; deployments
        // without text search get `None` for the bm25 handle on every
        // materialised TenantHandle.
        let router = make_router_no_vector();
        let h = router
            .route(TenantId::DEFAULT, PartitionId::ZERO)
            .expect("DEFAULT routes");
        assert!(
            h.bm25().is_none(),
            "default 3-arg new should leave bm25 unwired"
        );
    }

    #[test]
    fn bm25_handle_plumbed_through_when_some() {
        // ADR-039 §D-9: `TenantHandle::bm25()` must plumb the same Arc
        // the router was constructed with. Mirrors the vector wiring
        // pin above; constructor is the new 4-arg
        // `MultiTenantRouter::new_with_bm25`.
        let io = Arc::new(InMemoryPageIo::new());
        let pool = BufferPool::new(8, io);
        let mgr = TxnManager::new();
        let catalog = Arc::new(SystemCatalog::new());
        catalog.bootstrap(&pool, &mgr).expect("bootstrap");
        let crud = Arc::new(CrudStore::new());
        let bm25: Arc<dyn Bm25IndexStoreHandle> = Arc::new(NoopBm25Store);
        let router = MultiTenantRouter::new_with_bm25(catalog, crud, None, Some(Arc::clone(&bm25)));
        let h = router
            .route(TenantId::DEFAULT, PartitionId::ZERO)
            .expect("route");
        let h_bm25 = h.bm25().expect("Some plumbed through");
        assert!(
            Arc::ptr_eq(h_bm25, &bm25),
            "router must clone the SAME Arc into the handle"
        );
    }

    #[test]
    fn new_with_bm25_threads_both_vector_and_bm25() {
        // Both optional handles must coexist on a single TenantHandle.
        let io = Arc::new(InMemoryPageIo::new());
        let pool = BufferPool::new(8, io);
        let mgr = TxnManager::new();
        let catalog = Arc::new(SystemCatalog::new());
        catalog.bootstrap(&pool, &mgr).expect("bootstrap");
        let crud = Arc::new(CrudStore::new());
        let vector: Arc<dyn VectorPageStoreHandle> = Arc::new(NoopVectorStore);
        let bm25: Arc<dyn Bm25IndexStoreHandle> = Arc::new(NoopBm25Store);
        let router = MultiTenantRouter::new_with_bm25(
            catalog,
            crud,
            Some(Arc::clone(&vector)),
            Some(Arc::clone(&bm25)),
        );
        let h = router
            .route(TenantId::DEFAULT, PartitionId::ZERO)
            .expect("route");
        assert!(
            Arc::ptr_eq(h.vector().expect("vector"), &vector),
            "vector arc"
        );
        assert!(Arc::ptr_eq(h.bm25().expect("bm25"), &bm25), "bm25 arc");
    }

    #[test]
    fn new_three_arg_signature_preserved() {
        // ADR-039 §D-9 / §"Signature stability": the existing 3-arg
        // `new(catalog, crud, vector)` signature is preserved as a
        // backward-compat shorthand for `new_with_bm25(..., None)`.
        // Pin: a route from a 3-arg-constructed router has bm25 = None.
        let router = make_router_no_vector();
        let h = router
            .route(TenantId::DEFAULT, PartitionId::ZERO)
            .expect("route");
        assert!(h.bm25().is_none());
        assert!(h.vector().is_none());
    }

    #[test]
    fn debug_renders_bm25_field() {
        // Smoke: bm25 field surfaces in TenantHandle / MultiTenantRouter
        // Debug renderings (redacted to placeholder, like vector).
        let io = Arc::new(InMemoryPageIo::new());
        let pool = BufferPool::new(8, io);
        let mgr = TxnManager::new();
        let catalog = Arc::new(SystemCatalog::new());
        catalog.bootstrap(&pool, &mgr).expect("bootstrap");
        let crud = Arc::new(CrudStore::new());
        let bm25: Arc<dyn Bm25IndexStoreHandle> = Arc::new(NoopBm25Store);
        let router = MultiTenantRouter::new_with_bm25(catalog, crud, None, Some(Arc::clone(&bm25)));
        let dbg_router = format!("{router:?}");
        assert!(dbg_router.contains("bm25"), "{dbg_router}");
        assert!(dbg_router.contains("Bm25IndexStoreHandle"), "{dbg_router}");

        let h = router
            .route(TenantId::DEFAULT, PartitionId::ZERO)
            .expect("route");
        let dbg_h = format!("{h:?}");
        assert!(dbg_h.contains("bm25"), "{dbg_h}");
    }

    #[test]
    fn tenants_passes_through_catalog_list() {
        let router = make_router_no_vector();
        let listed = router.tenants();
        // bootstrap installs DEFAULT only; SYSTEM is NOT in the
        // list (catalog.rs:175-183).
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0], TenantId::DEFAULT);
    }

    #[test]
    fn debug_impls_compile_and_redact_trait_objects() {
        // Smoke test: Debug must compile for both types and the
        // dyn handle field must surface as a placeholder, not
        // attempt a derived `<dyn …>` Debug call.
        let router = make_router_no_vector();
        let dbg = format!("{router:?}");
        assert!(dbg.contains("MultiTenantRouter"));
        let h = router
            .route(TenantId::DEFAULT, PartitionId::ZERO)
            .expect("route");
        let dbg_h = format!("{h:?}");
        assert!(dbg_h.contains("TenantHandle"));
        assert!(dbg_h.contains("DEFAULT") || dbg_h.contains("TenantId"));
    }
}
