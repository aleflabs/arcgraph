//! System catalog (M1.5-07, ADR-011 §"Catalog layer"; extended by
//! ADR-034 §Slice A for per-tenant durability tiers).
//!
//! Holds the list of logical tenants under `TenantId::SYSTEM`.
//! v1.0 ships exactly one user tenant (`DEFAULT`); the catalog
//! provides an idempotent bootstrap, a list API, and (post-ADR-034)
//! a per-tenant `DurabilityTier` field with its own get/set surface.
//!
//! Catalog state is committed into the MVCC version store under
//! `TenantId::SYSTEM` (key 0 = tenants-table header) so that recovery
//! can replay it. The `BufferPool` parameter to `bootstrap` is accepted
//! for API compatibility and remains unused there; **the M10 stage-1
//! dedicated catalog page is pinned via [`SystemCatalog::attach_page_store`]**
//! (ADR-207): production bootstrap attaches the durable pool right after
//! `bootstrap()`, which materializes the registry into the catalog root
//! page ([`CATALOG_PAGE_ID`]), read-back-verifies it, and retains the
//! pool so tier mutations write through. The on-page registry is a
//! NON-AUTHORITATIVE materialization at stage-1 — WAL/MVCC + the
//! in-memory list remain the durability authority; registry RECOVERY
//! from the page is the ADR-183 M10 stage-2 forward-pin.
//!
//! # Durability tier model (ADR-034)
//!
//! Each tenant has an associated [`DurabilityTier`]:
//!
//! - [`DurabilityTier::Strict`] — T1, fsync-per-commit. Default.
//! - [`DurabilityTier::Periodic`] — T3, background fsync within
//!   `rpo_ms`.
//!
//! The tier is **read at commit time** by
//! [`crate::transaction::Transaction::commit_with_bundle`] via the
//! catalog lookup (ADR-034 §I-D7). A tier change via
//! [`SystemCatalog::set_durability_tier`] takes effect for commits
//! after the change's own `commit_lsn`.
//!
//! **`TenantId::SYSTEM` is T1-enforced.** Any attempt to set
//! SYSTEM to [`DurabilityTier::Periodic`] returns
//! [`DurabilityTierError::SystemTenantMustBeStrict`]. The invariant
//! is load-bearing: the tier-change commit itself is a SYSTEM-tenant
//! MVCC write, and if SYSTEM were T3 the tier change could be lost
//! on crash, leaving the catalog in an inconsistent state.
//!
//! # Back-of-envelope (design-v2 §A.3)
//!
//! - Bootstrap is one MVCC commit: O(1) work, one LSN allocation,
//!   one DashMap insert. Negligible cost vs. any I/O.
//! - Tier lookup per commit is one `parking_lot::RwLock::read` +
//!   one linear scan of the tenant list (DEFAULT + SYSTEM + ≤
//!   small-user-tenant-count). ≤ 50 ns per lookup; amortised
//!   into the commit's already-microsecond-scale Phase 2.
//! - `set_durability_tier` is one MVCC commit. Frequency expected
//!   low (operator-driven; not on the hot path).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use arcgraph_core::{
    ArcGraphError, DurabilityTier, DurabilityTierError, Lsn, PageId, Result,
    TenantDurabilityLookup, TenantId,
};
use bytes::Bytes;
use parking_lot::RwLock;
use tracing::{debug, warn};

use crate::buffer::BufferPool;
use crate::transaction::{Transaction, TxnManager};

pub mod page;
pub mod stats;
pub use page::{CatalogPageError, decode_catalog_page, encode_catalog_page};
pub use stats::{CatalogSnapshot, CatalogStats};

/// Reserved page id for the catalog root. Future catalog page allocations
/// must start above this; `PageId::ZERO` is the well-known anchor.
pub const CATALOG_PAGE_ID: PageId = PageId::ZERO;

/// MVCC key inside `TenantId::SYSTEM` for the tenants-table header.
const CATALOG_TENANTS_KEY: u64 = 0;

/// MVCC key base inside `TenantId::SYSTEM` for per-tenant durability
/// tier entries. The actual key is
/// `CATALOG_DURABILITY_TIER_KEY_BASE | tenant_id.raw()`.
///
/// The high-bit prefix keeps the durability key namespace disjoint
/// from the tenants-table header key (0) and any future catalog key
/// under `TenantId::SYSTEM`.
const CATALOG_DURABILITY_TIER_KEY_BASE: u64 = 0x8000_0000_0000_0000;

/// #1513 (M5-D1b) — MVCC key base inside `TenantId::SYSTEM` for
/// per-tenant registration entries written by
/// [`SystemCatalog::register_tenant`].
///
/// The actual key is `CATALOG_TENANT_ENTRY_KEY_BASE | tenant.raw()`. The
/// `0x2000` prefix is disjoint from:
///
/// - `0` (`CATALOG_TENANTS_KEY`, tenants-table header — the DEFAULT
///   sentinel written by [`SystemCatalog::bootstrap`]).
/// - `0x8000_0000_0000_0000` (`CATALOG_DURABILITY_TIER_KEY_BASE`).
///
/// Disjointness is pinned by `catalog_key_prefixes_are_disjoint` in
/// `tests` below: aliased keys under `TenantId::SYSTEM` would silently
/// corrupt the catalog.
pub const CATALOG_TENANT_ENTRY_KEY_BASE: u64 = 0x2000_0000_0000_0000;

/// A row in the `system.tenants` virtual table.
///
/// ADR-034 §Slice A extends this with `tier: DurabilityTier`. The
/// `Default` impl of `DurabilityTier` (= [`DurabilityTier::Strict`])
/// means pre-ADR-034 catalogs upgrade to T1 for every tenant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TenantRecord {
    /// Numeric tenant identifier.
    pub tenant_id: TenantId,
    /// Human-readable database name.
    pub name: String,
    /// LSN at which this tenant was first registered in the catalog.
    pub created_lsn: Lsn,
    /// Durability tier for this tenant (ADR-034 §Slice A). Defaults
    /// to [`DurabilityTier::Strict`] for tenants registered before
    /// ADR-034; operators change it via
    /// [`SystemCatalog::set_durability_tier`].
    pub tier: DurabilityTier,
}

/// In-process system catalog, scoped to `TenantId::SYSTEM`.
///
/// A single instance is expected per `BufferPool` / `TxnManager` pair.
/// `bootstrap` is idempotent and concurrency-safe.
///
/// Post-ADR-034: also holds per-tenant durability tiers. Tier lookups
/// are O(tenant count) linear scans under the read lock; for v1.0
/// the tenant count is small enough (typically 2–3) that this is
/// cheaper than a separate DashMap indirection. If the tenant count
/// grows post-v1.1, switch to a `DashMap<TenantId, DurabilityTier>`.
pub struct SystemCatalog {
    bootstrapped: AtomicBool,
    tenants: RwLock<Vec<TenantRecord>>,
    /// M10 stage-1 (ADR-207) — the buffer pool serving the dedicated
    /// catalog root page. `None` until [`Self::attach_page_store`]
    /// runs (legacy/unit callers never attach and pay one nullable
    /// check per registry mutation). When `Some`, tier mutations
    /// write the registry through to [`CATALOG_PAGE_ID`].
    page_pool: RwLock<Option<Arc<BufferPool>>>,
}

impl Default for SystemCatalog {
    fn default() -> Self {
        Self::new()
    }
}

impl SystemCatalog {
    /// Create an uninitialised catalog. Call [`Self::bootstrap`] before
    /// any other method.
    pub fn new() -> Self {
        Self {
            bootstrapped: AtomicBool::new(false),
            tenants: RwLock::new(Vec::new()),
            page_pool: RwLock::new(None),
        }
    }

    /// Idempotent bootstrap. On first call, commits a `TenantId::SYSTEM`
    /// transaction that installs the DEFAULT tenant sentinel into the MVCC
    /// store and records it in the in-memory tenant list. On subsequent
    /// calls, verifies the sentinel is present and returns immediately.
    ///
    /// Returns `CATALOG_PAGE_ID` — the well-known page id reserved for
    /// the catalog root. M10 stage-1 (ADR-207): production bootstrap
    /// pins it by calling [`Self::attach_page_store`] right after this
    /// method; `bootstrap` itself still leaves `pool` untouched
    /// (`_pool`) so the 142 legacy `bootstrap(&pool, &mgr)` call sites
    /// keep their semantics.
    ///
    /// ADR-034 note: the DEFAULT tenant is bootstrapped at tier
    /// [`DurabilityTier::Strict`]. Operators who want T3 for the
    /// log-ingest use case call [`Self::set_durability_tier`] post-
    /// bootstrap.
    pub fn bootstrap(&self, _pool: &BufferPool, txn_mgr: &TxnManager) -> Result<PageId> {
        if self.bootstrapped.load(Ordering::Acquire) {
            return Ok(CATALOG_PAGE_ID);
        }

        // Serialize bootstrap so exactly one thread does the work.
        let mut tenants = self.tenants.write();

        // Double-checked locking: another thread may have won.
        if self.bootstrapped.load(Ordering::Acquire) {
            return Ok(CATALOG_PAGE_ID);
        }

        // Commit a SYSTEM txn that installs the default-tenant sentinel.
        let mut txn = txn_mgr.begin(TenantId::SYSTEM);
        txn.write(CATALOG_TENANTS_KEY, encode_tenant_name("default"));
        let created_lsn = txn.commit()?;

        tenants.push(TenantRecord {
            tenant_id: TenantId::DEFAULT,
            name: "default".to_string(),
            created_lsn,
            tier: DurabilityTier::default(),
        });

        self.bootstrapped.store(true, Ordering::Release);
        Ok(CATALOG_PAGE_ID)
    }

    /// Return a snapshot of all registered tenants.
    ///
    /// Always returns at least `[{TenantId::DEFAULT, "default"}]`
    /// after a successful `bootstrap`.
    ///
    /// # Panics
    ///
    /// Panics if called before [`Self::bootstrap`].
    pub fn list_tenants(&self) -> Vec<TenantRecord> {
        assert!(
            self.bootstrapped.load(Ordering::Acquire),
            "SystemCatalog::list_tenants called before bootstrap"
        );
        self.tenants.read().clone()
    }

    /// #1513 (M5-D1b, amendment `docs/design/M5D-REDESIGN-AMENDMENT.md`
    /// §10 Risk-2 ruling) — register `tenant` into the served catalog
    /// through the SAME registration shape [`Self::bootstrap`] uses for
    /// the DEFAULT tenant, so [`crate::router::MultiTenantRouter::route`]'s
    /// `UnknownTenant` guard (which consults [`Self::list_tenants`])
    /// resolves for it exactly as for any durably-created tenant.
    ///
    /// Production consumer: the cold-open path of a loaded generation
    /// (`arcgraph-cli::bootstrap`) registers every tenant in the MANIFEST
    /// `tenant_census` via this method. Without it, a fresh-loaded
    /// store's tenants are servable on disk but not catalog-listed, and
    /// the production MCP/Bolt dispatch `route(tenant, PartitionId::ZERO)`
    /// returns `RoutingError::UnknownTenant` (#1513).
    ///
    /// **Idempotent:** if `tenant` is already registered the call is a
    /// no-op returning the existing record's `created_lsn` — re-running
    /// cold open (or resuming a crash-interrupted registration sweep)
    /// converges on the same set with no duplicates.
    ///
    /// **ADR-207 stage-1 fence:** this is NOT registry recovery from the
    /// catalog root page (that is the deferred stage-2). The caller's
    /// source of truth is external (the MANIFEST census); the in-memory
    /// list stays per-boot derived state, and the root-page mirror is
    /// refreshed via the same write-through as tier changes.
    ///
    /// The registration is recorded as a SYSTEM-tenant MVCC commit under
    /// `CATALOG_TENANT_ENTRY_KEY_BASE | tenant.raw()` (the same commit
    /// shape as `bootstrap`'s DEFAULT sentinel), so `created_lsn` is a
    /// real commit LSN.
    ///
    /// # Errors
    ///
    /// - `tenant == TenantId::SYSTEM` — SYSTEM is the catalog's own MVCC
    ///   home, never a listable tenant (same invariant as
    ///   [`Self::set_durability_tier`]).
    /// - MVCC commit failure (WAL fsync error under T1).
    ///
    /// # Panics
    ///
    /// Panics if called before [`Self::bootstrap`] (programming error,
    /// same contract as [`Self::list_tenants`]).
    pub fn register_tenant(
        &self,
        txn_mgr: &TxnManager,
        tenant: TenantId,
        name: &str,
    ) -> Result<Lsn> {
        assert!(
            self.bootstrapped.load(Ordering::Acquire),
            "SystemCatalog::register_tenant called before bootstrap"
        );
        if tenant == TenantId::SYSTEM {
            return Err(ArcGraphError::Io(std::io::Error::other(
                "SystemCatalog::register_tenant: TenantId::SYSTEM is the catalog's own MVCC \
                 home and is never a listable tenant",
            )));
        }

        // Serialize registration the same way bootstrap serializes its
        // sentinel install: hold the write lock across the check + the
        // MVCC commit + the push, so two concurrent registrations of
        // the same tenant cannot both pass the presence check.
        let created_lsn;
        {
            let mut tenants = self.tenants.write();
            if let Some(existing) = tenants.iter().find(|r| r.tenant_id == tenant) {
                return Ok(existing.created_lsn);
            }
            let mut txn = txn_mgr.begin(TenantId::SYSTEM);
            txn.write(
                CATALOG_TENANT_ENTRY_KEY_BASE | tenant.raw(),
                encode_tenant_name(name),
            );
            created_lsn = txn.commit()?;
            tenants.push(TenantRecord {
                tenant_id: tenant,
                name: name.to_string(),
                created_lsn,
                tier: DurabilityTier::default(),
            });
        }

        // M10 stage-1 (ADR-207) — mirror the grown registry onto the
        // catalog root page when a pool is attached (no-op before
        // `attach_page_store`; the attach materializes the full
        // registry itself). Same posture as `set_durability_tier` §7.
        self.write_through_page();
        Ok(created_lsn)
    }

    /// ADR-034 §Slice A — look up the durability tier for `tenant`.
    ///
    /// Returns [`DurabilityTier::Strict`] (the [`Default`] value) if
    /// the tenant is not in the catalog — this preserves the
    /// pre-ADR-034 behavior for any code path that bypasses the
    /// catalog's registration surface (e.g., ad-hoc tests with a
    /// freshly-constructed `TxnManager`).
    ///
    /// **`TenantId::SYSTEM` always returns
    /// [`DurabilityTier::Strict`]**, by construction. SYSTEM is
    /// excluded from the tenant list and its tier is never stored;
    /// the invariant is enforced here and by
    /// [`Self::set_durability_tier`] (ADR-034 §I-D7).
    ///
    /// Panics if called before [`Self::bootstrap`] — callers should
    /// have a fully-initialised catalog by the time they reach commit.
    #[must_use]
    pub fn durability_tier(&self, tenant: TenantId) -> DurabilityTier {
        // SYSTEM is T1-enforced (ADR-034 §I-D7). Short-circuit without
        // taking the read lock so the assert below doesn't fire in
        // pre-bootstrap paths (e.g., the bootstrap commit itself, which
        // runs under SYSTEM before the catalog marks itself bootstrapped).
        if tenant == TenantId::SYSTEM {
            return DurabilityTier::Strict;
        }

        assert!(
            self.bootstrapped.load(Ordering::Acquire),
            "SystemCatalog::durability_tier called before bootstrap"
        );
        let guard = self.tenants.read();
        for entry in guard.iter() {
            if entry.tenant_id == tenant {
                return entry.tier;
            }
        }
        // Tenant not in catalog — default to Strict for compatibility.
        DurabilityTier::Strict
    }

    /// ADR-034 §Slice A — set the durability tier for `tenant`.
    ///
    /// The change is recorded as a SYSTEM-tenant MVCC write under
    /// key `CATALOG_DURABILITY_TIER_KEY_BASE | tenant.raw()` inside
    /// the caller's `tx`. The caller is responsible for committing
    /// `tx`; the tier takes effect for any commit whose `commit_lsn`
    /// is strictly greater than the tier-change commit's
    /// `commit_lsn` (ADR-034 §I-D7).
    ///
    /// **`tx` MUST be a `TenantId::SYSTEM` transaction.** The MVCC
    /// key is written under SYSTEM; a non-SYSTEM `tx` would write
    /// the key into the wrong tenant's chain.
    ///
    /// **Validation:**
    /// - `tier.validate()` — returns [`DurabilityTierError::RpoTooSmall`]
    ///   or [`DurabilityTierError::RpoTooLarge`] on bad `rpo_ms`.
    /// - `tenant != TenantId::SYSTEM` — returns
    ///   [`DurabilityTierError::SystemTenantMustBeStrict`] otherwise
    ///   (ADR-034 §I-D7).
    /// - `tenant` must be in the catalog — returns
    ///   [`DurabilityTierError::TenantNotFound`] otherwise.
    ///
    /// Post-change observability: [`Self::durability_tier`] reflects
    /// the new tier as soon as `tx.commit()` returns `Ok`. If
    /// `tx.commit()` fails (WAL fsync error under T1, Z-1 (b) rollback
    /// via ADR-033), the in-memory catalog state is also rolled back
    /// — the write is only promoted to `self.tenants` by
    /// `Self::apply_tier_change_after_commit` which callers invoke
    /// from a post-commit hook.
    ///
    /// At v1.0 the apply-after-commit hook is collapsed into this
    /// method: the in-memory state is updated under the catalog's
    /// RwLock BEFORE the commit. If the commit fails, the caller
    /// MUST call [`Self::revert_tier_change`] to restore the prior
    /// value. This matches the ADR-033 Z-1 (b) convention of
    /// capturing pre-mutation state for rollback; a future refactor
    /// will integrate tier changes into the `TxnMutationLog` directly.
    ///
    /// For the v1.0 shape, most operator flows look like:
    ///
    /// ```ignore
    /// let mut tx = txn_mgr.begin(TenantId::SYSTEM);
    /// catalog.set_durability_tier(&mut tx, TenantId::DEFAULT,
    ///     DurabilityTier::Periodic { rpo_ms: 500 })?;
    /// tx.commit()?; // WAL-durable; tier now live.
    /// ```
    ///
    /// which is the pattern all v1.0 tests and operators use.
    pub fn set_durability_tier(
        &self,
        tx: &mut Transaction<'_>,
        tenant: TenantId,
        tier: DurabilityTier,
    ) -> std::result::Result<(), DurabilityTierError> {
        // §1. Tier shape validation.
        tier.validate()?;

        // §2. SYSTEM-tenant T1 enforcement (ADR-034 §I-D7).
        if tenant == TenantId::SYSTEM {
            return Err(DurabilityTierError::SystemTenantMustBeStrict);
        }

        // §3. Caller's tx must be under SYSTEM.
        debug_assert_eq!(
            tx.tenant(),
            TenantId::SYSTEM,
            "set_durability_tier requires a SYSTEM-tenant transaction",
        );

        // §4. Tenant must be in the catalog.
        {
            let guard = self.tenants.read();
            if !guard.iter().any(|r| r.tenant_id == tenant) {
                return Err(DurabilityTierError::TenantNotFound {
                    tenant_raw: tenant.raw(),
                });
            }
        }

        // §5. Buffer the MVCC write. The durable record encodes the
        //     new tier; commit-time atomicity is inherited from the
        //     CommitBundle path (ADR-031). If `tier.rpo_ms` is out of
        //     range we already returned above.
        let key = CATALOG_DURABILITY_TIER_KEY_BASE | tenant.raw();
        tx.write(key, encode_durability_tier(tier));

        // §6. In-memory update. At v1.0 we update pre-commit and the
        //     caller handles rollback via revert_tier_change if
        //     commit fails. See rustdoc above.
        {
            let mut guard = self.tenants.write();
            let mut updated = false;
            for entry in guard.iter_mut() {
                if entry.tenant_id == tenant {
                    entry.tier = tier;
                    updated = true;
                    break;
                }
            }
            if !updated {
                // Pre-commit validation held the tenant-present
                // invariant, so the only way to reach here is a
                // concurrent remove (which v1.0 does not support).
                // Defend against future regression.
                return Err(DurabilityTierError::TenantNotFound {
                    tenant_raw: tenant.raw(),
                });
            }
        }

        // §7. M10 stage-1 (ADR-207) — mirror the in-memory registry
        //     onto the catalog root page when a pool is attached. The
        //     page tracks the IN-MEMORY registry (which this method
        //     just mutated pre-commit); if the caller's commit fails,
        //     `revert_tier_change` re-mirrors the restored value.
        self.write_through_page();
        Ok(())
    }

    /// ADR-034 §Slice A — revert an in-memory tier change.
    ///
    /// Called by an operator flow whose `set_durability_tier` +
    /// `tx.commit()` sequence returned `Err`. Restores `tenant`'s
    /// in-memory tier to `previous_tier`. The MVCC write in the
    /// rolled-back `tx` was undone by the Z-1 (b) path already
    /// (ADR-033); this method just re-aligns the in-process cache.
    ///
    /// No-op if the tenant is not in the catalog; same rationale as
    /// the pre-commit `TenantNotFound` check in
    /// [`Self::set_durability_tier`] — a concurrent removal is a
    /// post-v1.0 concern.
    pub fn revert_tier_change(&self, tenant: TenantId, previous_tier: DurabilityTier) {
        {
            let mut guard = self.tenants.write();
            let mut reverted = false;
            for entry in guard.iter_mut() {
                if entry.tenant_id == tenant {
                    entry.tier = previous_tier;
                    reverted = true;
                    break;
                }
            }
            if !reverted {
                return;
            }
        }
        // M10 stage-1 (ADR-207) — re-mirror the restored registry onto
        // the catalog root page (see `set_durability_tier` §7).
        self.write_through_page();
    }

    /// M10 stage-1 (ADR-207) — attach the durable buffer pool and pin
    /// the dedicated catalog root page ([`CATALOG_PAGE_ID`]).
    ///
    /// Called by both production bootstrap flows
    /// (`arcgraph-cli::bootstrap::{build_durable, build_in_memory}`)
    /// immediately after [`Self::bootstrap`]. This is the read path
    /// that makes the design-v2 §10.2 `buffer_pool_hit_rate` producer
    /// fire on REAL page traffic (GA exit-criteria §5.2 last gap —
    /// "wired ≠ producer-fires"). Sequence (ADR-207 D-3):
    ///
    /// 1. **Read-back** — `pin_read(CATALOG_PAGE_ID)`. A decodable
    ///    page is reported as `prior_registry: Some(..)`; an
    ///    I/O error (never-written page 0 — fresh dir) or
    ///    [`CatalogPageError::BadMagic`] (zeroed page 0 — pre-M10 dir)
    ///    is the NORMAL first-attach path; any other decode error is
    ///    logged + self-healed by step 2 (the page is a
    ///    non-authoritative materialization; a corrupt page must
    ///    never brick boot — WAL/MVCC remain the authority).
    /// 2. **Materialize** — encode the CURRENT in-memory registry and
    ///    write it through the pool (`pin_write` + `flush_all`). A
    ///    never-written page 0 is first created via direct
    ///    [`crate::io::PageIo::write_page`] on `io` — the buffer
    ///    pool's read-before-write slow path errors on a
    ///    never-written offset (the `BufferedRecordPageStore::evict_lru`
    ///    precedent). `io` MUST be the same [`crate::io::PageIo`] the
    ///    pool was constructed over (both production call sites build
    ///    the pool from `io` two lines above the attach; the step-3
    ///    verify read would catch a mismatched pair at boot).
    /// 3. **Verify** — `pin_read` + decode must equal the registry
    ///    just written (the ADR-204 PROVEN round-trip-gate pattern at
    ///    page granularity). Verify failure is a hard error: a pool
    ///    that cannot round-trip its only page must fail loud at
    ///    boot, not serve.
    /// 4. **Retain** — the pool is stored so
    ///    [`Self::set_durability_tier`] / [`Self::revert_tier_change`]
    ///    write the registry through (each a `pin_write` on the
    ///    resident page).
    ///
    /// **Stage-1 scope fence (ADR-207 D-5):** the prior on-page
    /// registry is NEVER loaded into the in-memory registry —
    /// divergence is logged (`prior_diverged`) citing the ADR-183
    /// M10 stage-2 forward-pin, and the page is overwritten with the
    /// current state. No durability claim changes here.
    ///
    /// # Errors
    ///
    /// Encode failure (registry exceeds one page —
    /// [`CatalogPageError::RegistryTooLarge`]), pool/IO write
    /// failure, or step-3 verify failure. All are boot-blocking by
    /// design.
    ///
    /// # Panics
    ///
    /// Panics if called before [`Self::bootstrap`] (programming
    /// error, same contract as [`Self::list_tenants`]).
    pub fn attach_page_store(
        &self,
        pool: Arc<BufferPool>,
        io: Arc<dyn crate::io::PageIo>,
    ) -> Result<CatalogPageAttachReport> {
        assert!(
            self.bootstrapped.load(Ordering::Acquire),
            "SystemCatalog::attach_page_store called before bootstrap"
        );

        // §1. Read-back — the pinned read (fires Miss on a cold page).
        let mut prior_registry: Option<Vec<TenantRecord>> = None;
        let mut healed_corruption = false;
        let mut page_on_disk = false;
        match pool.pin_read(CATALOG_PAGE_ID) {
            Ok(guard) => {
                page_on_disk = true;
                match page::decode_catalog_page(guard.as_bytes()) {
                    Ok(records) => prior_registry = Some(records),
                    Err(CatalogPageError::BadMagic) => {
                        // Zeroed page 0 (e.g. a sparse pages.db) — the
                        // normal pre-M10-dir path, not corruption.
                        debug!(
                            target: "arcgraph_storage::catalog",
                            "ADR-207 attach: page 0 readable but not a catalog page; materializing fresh",
                        );
                    }
                    Err(e) => {
                        warn!(
                            target: "arcgraph_storage::catalog",
                            error = %e,
                            "ADR-207 attach: catalog root page undecodable; treating as absent and \
                             rewriting (non-authoritative materialization — WAL/MVCC remain the \
                             durability authority)",
                        );
                        healed_corruption = true;
                    }
                }
            }
            Err(_) => {
                // Never-written page 0 — fresh data dir. Normal first
                // boot; the metric correctly records nothing (no read
                // was served).
            }
        }

        // §2. Materialize the CURRENT in-memory registry.
        let current = self.tenants.read().clone();
        let encoded = page::encode_catalog_page(&current).map_err(|e| {
            ArcGraphError::Io(std::io::Error::other(format!(
                "ADR-207 catalog page encode failed at attach: {e}",
            )))
        })?;
        if page_on_disk {
            {
                let mut guard = pool.pin_write(CATALOG_PAGE_ID, TenantId::SYSTEM)?;
                guard.as_bytes_mut().copy_from_slice(&encoded[..]);
            }
            pool.flush_all()?;
        } else {
            // First materialization — direct write on the pool's
            // backing io, then flush. `pin_write` cannot create the
            // page: its slow path reads-before-write and
            // `PageIo::read_page` errors on a never-written offset
            // (the `BufferedRecordPageStore::evict_lru` precedent).
            io.write_page(CATALOG_PAGE_ID, &encoded)?;
            io.flush()?;
        }

        // §3. Verify round-trip (PROVEN doctrine; hard error).
        {
            let guard = pool.pin_read(CATALOG_PAGE_ID)?;
            let back = page::decode_catalog_page(guard.as_bytes()).map_err(|e| {
                ArcGraphError::Io(std::io::Error::other(format!(
                    "ADR-207 catalog page verify: decode-after-write failed: {e}",
                )))
            })?;
            if back != current {
                return Err(ArcGraphError::Io(std::io::Error::other(
                    "ADR-207 catalog page verify: read-back does not match the registry just \
                     written",
                )));
            }
        }

        // §4. Divergence log + retain the pool for write-through.
        let prior_diverged = prior_registry.as_ref().is_some_and(|p| *p != current);
        if prior_diverged {
            warn!(
                target: "arcgraph_storage::catalog",
                "ADR-207 attach: prior on-page registry diverges from the bootstrap registry — \
                 EXPECTED at stage-1 (registry state does not survive restart; ADR-183 M10 \
                 stage-2 forward-pin). In-memory/WAL state stays authoritative; page rewritten.",
            );
        }
        *self.page_pool.write() = Some(pool);
        Ok(CatalogPageAttachReport {
            prior_registry,
            prior_diverged,
            healed_corruption,
        })
    }

    /// M10 stage-1 (ADR-207) — mirror the in-memory registry onto the
    /// catalog root page, when a pool is attached.
    ///
    /// Warn-and-continue on failure (NOT a hard error): callers invoke
    /// this AFTER their in-memory + MVCC effects are already in
    /// place, the page is a non-authoritative materialization, and
    /// the next attach read-back detects + logs the divergence. Same
    /// posture as the per-tenant stats-rebuild failure capture in
    /// `build_durable` §8.
    fn write_through_page(&self) {
        let pool_guard = self.page_pool.read();
        let Some(pool) = pool_guard.as_ref() else {
            return;
        };
        let records = self.tenants.read().clone();
        let encoded = match page::encode_catalog_page(&records) {
            Ok(p) => p,
            Err(e) => {
                tracing::error!(
                    target: "arcgraph_storage::catalog",
                    error = %e,
                    "ADR-207 catalog page write-through: encode failed; on-page registry is now \
                     stale (next attach read-back will flag the divergence)",
                );
                return;
            }
        };
        let result: Result<()> = (|| {
            {
                let mut guard = pool.pin_write(CATALOG_PAGE_ID, TenantId::SYSTEM)?;
                guard.as_bytes_mut().copy_from_slice(&encoded[..]);
            }
            pool.flush_all()
        })();
        if let Err(e) = result {
            tracing::error!(
                target: "arcgraph_storage::catalog",
                error = %e,
                "ADR-207 catalog page write-through: pin/flush failed; on-page registry is now \
                 stale (next attach read-back will flag the divergence)",
            );
        }
    }
}

/// Outcome of [`SystemCatalog::attach_page_store`] (M10 stage-1,
/// ADR-207). Diagnostic + test oracle; production callers log it.
#[derive(Debug)]
pub struct CatalogPageAttachReport {
    /// The registry decoded from the catalog root page BEFORE this
    /// attach overwrote it. `None` on a fresh/pre-M10/corrupt page.
    /// Stage-1 never loads this into the in-memory registry (ADR-207
    /// D-5 — recovery semantics are M10 stage-2).
    pub prior_registry: Option<Vec<TenantRecord>>,
    /// `true` iff `prior_registry` is `Some` and differs from the
    /// registry materialized by this attach.
    pub prior_diverged: bool,
    /// `true` iff page 0 was readable but undecodable (real
    /// corruption / foreign bytes — NOT the zeroed-page case) and was
    /// self-healed by this attach's rewrite.
    pub healed_corruption: bool,
}

/// ADR-034 §Slice D — allow the MVCC kernel (`TxnManager`) to resolve
/// tier at commit time without pulling the catalog type into
/// `arcgraph-core`. Implementing the trait delegates to the existing
/// [`SystemCatalog::durability_tier`] implementation, which already
/// honours the SYSTEM-T1 invariant and defaults unknown tenants to
/// `Strict`.
impl TenantDurabilityLookup for SystemCatalog {
    #[inline]
    fn durability_tier(&self, tenant: TenantId) -> DurabilityTier {
        self.durability_tier(tenant)
    }
}

fn encode_tenant_name(name: &str) -> Bytes {
    Bytes::copy_from_slice(name.as_bytes())
}

/// Encode a [`DurabilityTier`] as a compact MVCC value payload.
///
/// Wire shape (little-endian, stable across v1.0):
///
/// - Byte 0: discriminant (0 = Strict, 1 = Periodic).
/// - Bytes 1..9 (Periodic only): `rpo_ms` as `u64` LE.
///
/// This format is INDEPENDENT of the CommitBundle payload encoding
/// (ADR-031) — it is the MVCC *value* stored in a SYSTEM-tenant
/// version chain, not a record-layer format. Recovery reads it
/// through the standard MVCC replay path; no ADR-031 amendment is
/// required.
fn encode_durability_tier(tier: DurabilityTier) -> Bytes {
    match tier {
        DurabilityTier::Strict => Bytes::copy_from_slice(&[0u8]),
        DurabilityTier::Periodic { rpo_ms } => {
            let mut buf = Vec::with_capacity(9);
            buf.push(1u8);
            buf.extend_from_slice(&rpo_ms.to_le_bytes());
            Bytes::from(buf)
        }
    }
}

/// Decode a tier value from the MVCC chain. Inverse of
/// `encode_durability_tier`.
///
/// Returns `None` on malformed bytes (unknown discriminant, wrong
/// length). Callers SHOULD treat `None` as "use [`DurabilityTier::
/// default()`]" — a malformed tier entry post-ADR-034 is a
/// forward-compat concern (a future tier variant with a new
/// discriminant will be decoded as `None` by a v1.0 binary).
#[must_use]
pub fn decode_durability_tier(bytes: &[u8]) -> Option<DurabilityTier> {
    match bytes.first() {
        Some(&0) if bytes.len() == 1 => Some(DurabilityTier::Strict),
        Some(&1) if bytes.len() == 9 => {
            let mut rpo = [0u8; 8];
            rpo.copy_from_slice(&bytes[1..9]);
            Some(DurabilityTier::Periodic {
                rpo_ms: u64::from_le_bytes(rpo),
            })
        }
        _ => None,
    }
}

// ---------- tests ------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arcgraph_core::TenantId;

    use super::*;
    use crate::buffer::BufferPool;
    use crate::io::{InMemoryPageIo, PageIo};
    use crate::transaction::TxnManager;

    fn make_deps() -> (BufferPool, TxnManager) {
        let io = Arc::new(InMemoryPageIo::new());
        let pool = BufferPool::new(8, io);
        let mgr = TxnManager::new();
        (pool, mgr)
    }

    #[test]
    fn bootstrap_returns_catalog_page_id() {
        let (pool, mgr) = make_deps();
        let cat = SystemCatalog::new();
        let page_id = cat.bootstrap(&pool, &mgr).unwrap();
        assert_eq!(page_id, CATALOG_PAGE_ID);
    }

    #[test]
    fn bootstrap_is_idempotent() {
        let (pool, mgr) = make_deps();
        let cat = SystemCatalog::new();
        let p1 = cat.bootstrap(&pool, &mgr).unwrap();
        let p2 = cat.bootstrap(&pool, &mgr).unwrap();
        assert_eq!(p1, p2);
        // Only one tenant regardless of how many times bootstrap is called.
        assert_eq!(cat.list_tenants().len(), 1);
    }

    #[test]
    fn list_tenants_returns_default_after_bootstrap() {
        let (pool, mgr) = make_deps();
        let cat = SystemCatalog::new();
        cat.bootstrap(&pool, &mgr).unwrap();
        let tenants = cat.list_tenants();
        assert_eq!(tenants.len(), 1);
        assert_eq!(tenants[0].tenant_id, TenantId::DEFAULT);
        assert_eq!(tenants[0].name, "default");
        assert!(
            tenants[0].created_lsn.raw() > 0,
            "created_lsn must be non-zero"
        );
    }

    #[test]
    fn catalog_sentinel_is_readable_via_txn_manager() {
        let (pool, mgr) = make_deps();
        let cat = SystemCatalog::new();
        let lsn = cat
            .bootstrap(&pool, &mgr)
            .map(|_| cat.list_tenants()[0].created_lsn)
            .unwrap();

        // The SYSTEM transaction that bootstrap committed is visible
        // to a subsequent SYSTEM reader at a snapshot >= created_lsn.
        let val = mgr.read_at(TenantId::SYSTEM, CATALOG_TENANTS_KEY, lsn);
        assert!(val.is_some(), "catalog sentinel not in MVCC store");
        assert_eq!(val.unwrap().as_ref(), b"default");
    }

    #[test]
    #[should_panic(expected = "before bootstrap")]
    fn list_tenants_panics_before_bootstrap() {
        let cat = SystemCatalog::new();
        let _ = cat.list_tenants();
    }

    // ─── #1513 (M5-D1b): register_tenant ───────────────────────────

    #[test]
    fn register_tenant_lists_and_is_idempotent() {
        let (pool, mgr) = make_deps();
        let cat = SystemCatalog::new();
        cat.bootstrap(&pool, &mgr).unwrap();

        let tenant = TenantId::new(91);
        let lsn1 = cat.register_tenant(&mgr, tenant, "loaded-91").unwrap();
        let tenants = cat.list_tenants();
        assert_eq!(tenants.len(), 2);
        let rec = tenants
            .iter()
            .find(|r| r.tenant_id == tenant)
            .expect("registered tenant listed");
        assert_eq!(rec.name, "loaded-91");
        assert_eq!(rec.created_lsn, lsn1);
        assert_eq!(rec.tier, DurabilityTier::default());

        // Idempotent re-register: same set, same created_lsn, no dupes.
        let lsn2 = cat.register_tenant(&mgr, tenant, "loaded-91").unwrap();
        assert_eq!(lsn2, lsn1, "re-register must return the existing LSN");
        let tenants = cat.list_tenants();
        assert_eq!(tenants.len(), 2, "re-register must not duplicate");

        // DEFAULT (installed by bootstrap) is also an idempotent no-op.
        let default_lsn = cat.list_tenants()[0].created_lsn;
        let lsn3 = cat
            .register_tenant(&mgr, TenantId::DEFAULT, "default")
            .unwrap();
        assert_eq!(lsn3, default_lsn);
        assert_eq!(cat.list_tenants().len(), 2);
    }

    #[test]
    fn register_tenant_rejects_system() {
        let (pool, mgr) = make_deps();
        let cat = SystemCatalog::new();
        cat.bootstrap(&pool, &mgr).unwrap();
        assert!(
            cat.register_tenant(&mgr, TenantId::SYSTEM, "system")
                .is_err(),
            "SYSTEM must never be a listable tenant"
        );
        assert_eq!(cat.list_tenants().len(), 1);
    }

    #[test]
    fn register_tenant_entry_is_readable_via_txn_manager() {
        // The registration commit is a real SYSTEM MVCC write under
        // the #1513 key namespace — same durability shape as the
        // bootstrap sentinel.
        let (pool, mgr) = make_deps();
        let cat = SystemCatalog::new();
        cat.bootstrap(&pool, &mgr).unwrap();
        let tenant = TenantId::new(7);
        let lsn = cat.register_tenant(&mgr, tenant, "loaded-7").unwrap();
        let key = CATALOG_TENANT_ENTRY_KEY_BASE | tenant.raw();
        let val = mgr.read_at(TenantId::SYSTEM, key, lsn);
        assert!(val.is_some(), "registration entry not in MVCC store");
        assert_eq!(val.unwrap().as_ref(), b"loaded-7");
    }

    // ─── ADR-034 §Slice A: DurabilityTier on TenantRecord ─────────

    #[test]
    fn bootstrap_installs_default_tenant_as_strict() {
        let (pool, mgr) = make_deps();
        let cat = SystemCatalog::new();
        cat.bootstrap(&pool, &mgr).unwrap();
        let tenants = cat.list_tenants();
        assert_eq!(tenants[0].tier, DurabilityTier::Strict);
        assert_eq!(
            cat.durability_tier(TenantId::DEFAULT),
            DurabilityTier::Strict
        );
    }

    #[test]
    fn durability_tier_system_is_always_strict() {
        // I-D7: SYSTEM is T1-enforced, non-configurable. Holds even
        // before bootstrap (the bootstrap commit itself runs under
        // SYSTEM and queries the tier).
        let cat = SystemCatalog::new();
        assert_eq!(
            cat.durability_tier(TenantId::SYSTEM),
            DurabilityTier::Strict
        );

        let (pool, mgr) = make_deps();
        cat.bootstrap(&pool, &mgr).unwrap();
        assert_eq!(
            cat.durability_tier(TenantId::SYSTEM),
            DurabilityTier::Strict
        );
    }

    #[test]
    fn durability_tier_unknown_tenant_is_strict() {
        let (pool, mgr) = make_deps();
        let cat = SystemCatalog::new();
        cat.bootstrap(&pool, &mgr).unwrap();
        let unknown = TenantId::new(999);
        assert_eq!(cat.durability_tier(unknown), DurabilityTier::Strict);
    }

    #[test]
    #[should_panic(expected = "before bootstrap")]
    fn durability_tier_user_tenant_panics_before_bootstrap() {
        let cat = SystemCatalog::new();
        // DEFAULT is a user tenant; pre-bootstrap lookup should panic
        // just like list_tenants. (SYSTEM is exempt per I-D7.)
        let _ = cat.durability_tier(TenantId::DEFAULT);
    }

    #[test]
    fn set_durability_tier_default_to_periodic_takes_effect() {
        let (pool, mgr) = make_deps();
        let cat = SystemCatalog::new();
        cat.bootstrap(&pool, &mgr).unwrap();
        assert_eq!(
            cat.durability_tier(TenantId::DEFAULT),
            DurabilityTier::Strict
        );

        let mut tx = mgr.begin(TenantId::SYSTEM);
        cat.set_durability_tier(
            &mut tx,
            TenantId::DEFAULT,
            DurabilityTier::Periodic { rpo_ms: 250 },
        )
        .unwrap();
        tx.commit().unwrap();

        assert_eq!(
            cat.durability_tier(TenantId::DEFAULT),
            DurabilityTier::Periodic { rpo_ms: 250 }
        );
    }

    #[test]
    fn set_durability_tier_rejects_system_tenant() {
        let (pool, mgr) = make_deps();
        let cat = SystemCatalog::new();
        cat.bootstrap(&pool, &mgr).unwrap();

        let mut tx = mgr.begin(TenantId::SYSTEM);
        let err = cat
            .set_durability_tier(
                &mut tx,
                TenantId::SYSTEM,
                DurabilityTier::Periodic { rpo_ms: 100 },
            )
            .unwrap_err();
        assert_eq!(err, DurabilityTierError::SystemTenantMustBeStrict);
        // Invariant holds: post-rejection, SYSTEM is still Strict.
        assert_eq!(
            cat.durability_tier(TenantId::SYSTEM),
            DurabilityTier::Strict
        );
    }

    #[test]
    fn set_durability_tier_rejects_bad_rpo() {
        let (pool, mgr) = make_deps();
        let cat = SystemCatalog::new();
        cat.bootstrap(&pool, &mgr).unwrap();

        let mut tx = mgr.begin(TenantId::SYSTEM);
        let err = cat
            .set_durability_tier(
                &mut tx,
                TenantId::DEFAULT,
                DurabilityTier::Periodic { rpo_ms: 5 },
            )
            .unwrap_err();
        assert!(matches!(err, DurabilityTierError::RpoTooSmall { .. }));

        let mut tx2 = mgr.begin(TenantId::SYSTEM);
        let err2 = cat
            .set_durability_tier(
                &mut tx2,
                TenantId::DEFAULT,
                DurabilityTier::Periodic { rpo_ms: 120_000 },
            )
            .unwrap_err();
        assert!(matches!(err2, DurabilityTierError::RpoTooLarge { .. }));
    }

    #[test]
    fn set_durability_tier_rejects_unknown_tenant() {
        let (pool, mgr) = make_deps();
        let cat = SystemCatalog::new();
        cat.bootstrap(&pool, &mgr).unwrap();

        let mut tx = mgr.begin(TenantId::SYSTEM);
        let err = cat
            .set_durability_tier(
                &mut tx,
                TenantId::new(999),
                DurabilityTier::Periodic { rpo_ms: 100 },
            )
            .unwrap_err();
        assert!(matches!(
            err,
            DurabilityTierError::TenantNotFound { tenant_raw: 999 }
        ));
    }

    #[test]
    fn set_durability_tier_mvcc_key_is_under_system() {
        // The set operation buffers a write at key
        // (CATALOG_DURABILITY_TIER_KEY_BASE | tenant.raw()) under the
        // SYSTEM tenant. Verify that's where the durable bytes land.
        let (pool, mgr) = make_deps();
        let cat = SystemCatalog::new();
        cat.bootstrap(&pool, &mgr).unwrap();

        let mut tx = mgr.begin(TenantId::SYSTEM);
        cat.set_durability_tier(
            &mut tx,
            TenantId::DEFAULT,
            DurabilityTier::Periodic { rpo_ms: 500 },
        )
        .unwrap();
        let lsn = tx.commit().unwrap();

        let key = CATALOG_DURABILITY_TIER_KEY_BASE | TenantId::DEFAULT.raw();
        let val = mgr.read_at(TenantId::SYSTEM, key, lsn);
        assert!(val.is_some(), "tier write is at the expected MVCC key");
        let decoded = decode_durability_tier(val.unwrap().as_ref()).unwrap();
        assert_eq!(decoded, DurabilityTier::Periodic { rpo_ms: 500 });
    }

    #[test]
    fn tier_change_affects_subsequent_reads() {
        // I-D7: the change's commit_lsn is the cut-over point. Readers
        // that materialize `durability_tier` AFTER the tier-change
        // commit see the new tier.
        let (pool, mgr) = make_deps();
        let cat = SystemCatalog::new();
        cat.bootstrap(&pool, &mgr).unwrap();

        assert_eq!(
            cat.durability_tier(TenantId::DEFAULT),
            DurabilityTier::Strict
        );

        let mut tx1 = mgr.begin(TenantId::SYSTEM);
        cat.set_durability_tier(
            &mut tx1,
            TenantId::DEFAULT,
            DurabilityTier::Periodic { rpo_ms: 100 },
        )
        .unwrap();
        tx1.commit().unwrap();
        assert_eq!(
            cat.durability_tier(TenantId::DEFAULT),
            DurabilityTier::Periodic { rpo_ms: 100 }
        );

        // Flip back to Strict.
        let mut tx2 = mgr.begin(TenantId::SYSTEM);
        cat.set_durability_tier(&mut tx2, TenantId::DEFAULT, DurabilityTier::Strict)
            .unwrap();
        tx2.commit().unwrap();
        assert_eq!(
            cat.durability_tier(TenantId::DEFAULT),
            DurabilityTier::Strict
        );
    }

    #[test]
    fn revert_tier_change_restores_previous_value() {
        let (pool, mgr) = make_deps();
        let cat = SystemCatalog::new();
        cat.bootstrap(&pool, &mgr).unwrap();

        // Tier goes Strict → Periodic successfully.
        let mut tx = mgr.begin(TenantId::SYSTEM);
        cat.set_durability_tier(
            &mut tx,
            TenantId::DEFAULT,
            DurabilityTier::Periodic { rpo_ms: 100 },
        )
        .unwrap();
        tx.commit().unwrap();
        assert!(cat.durability_tier(TenantId::DEFAULT).is_periodic());

        // Operator revert simulates the Z-1 (b) path rolling back.
        cat.revert_tier_change(TenantId::DEFAULT, DurabilityTier::Strict);
        assert_eq!(
            cat.durability_tier(TenantId::DEFAULT),
            DurabilityTier::Strict
        );
    }

    #[test]
    fn revert_tier_change_unknown_tenant_noops() {
        let cat = SystemCatalog::new();
        // No panic; no state to corrupt.
        cat.revert_tier_change(TenantId::new(42), DurabilityTier::Strict);
    }

    // ─── ADR-034 §Slice A: encoding round-trip ────────────────────

    #[test]
    fn encode_decode_strict_round_trip() {
        let bytes = encode_durability_tier(DurabilityTier::Strict);
        assert_eq!(bytes.as_ref(), &[0u8]);
        assert_eq!(decode_durability_tier(&bytes), Some(DurabilityTier::Strict));
    }

    #[test]
    fn encode_decode_periodic_round_trip() {
        for rpo in [10u64, 100, 1_000, 30_000, 60_000] {
            let t = DurabilityTier::Periodic { rpo_ms: rpo };
            let bytes = encode_durability_tier(t);
            assert_eq!(bytes.len(), 9);
            assert_eq!(bytes[0], 1);
            assert_eq!(decode_durability_tier(&bytes), Some(t));
        }
    }

    #[test]
    fn decode_rejects_malformed() {
        // Empty.
        assert!(decode_durability_tier(&[]).is_none());
        // Unknown discriminant.
        assert!(decode_durability_tier(&[99u8]).is_none());
        // Strict with extra bytes.
        assert!(decode_durability_tier(&[0u8, 0u8]).is_none());
        // Periodic with wrong length.
        assert!(decode_durability_tier(&[1u8, 0u8]).is_none());
        assert!(decode_durability_tier(&[1u8]).is_none());
    }

    #[test]
    fn decode_forward_compat_unknown_discriminant() {
        // A v1.1 T2 or T4 tier would use discriminant 2 or 3. v1.0
        // decoders return None; callers treat None as "fall back to
        // DurabilityTier::default()" per rustdoc.
        assert!(decode_durability_tier(&[2u8]).is_none());
        assert!(decode_durability_tier(&[3u8, 1, 2, 3, 4, 5, 6, 7, 8]).is_none());
    }

    #[test]
    fn catalog_key_prefixes_are_disjoint() {
        // Three catalog key namespaces under TenantId::SYSTEM:
        //   - 0                      = tenants-table header
        //   - 0x2000_0000_0000_0000  = #1513 per-tenant registration entry
        //   - 0x8000_0000_0000_0000  = ADR-034 per-tenant durability tier
        //
        // Two prefixes that aliased keys would silently corrupt the
        // catalog. Pin the pairwise disjointness here — the only test
        // that would catch a constant clobber via boring text edit.
        let bases = [
            CATALOG_TENANTS_KEY,
            CATALOG_TENANT_ENTRY_KEY_BASE,
            CATALOG_DURABILITY_TIER_KEY_BASE,
        ];
        for (i, a) in bases.iter().enumerate() {
            for b in &bases[i + 1..] {
                assert_ne!(a, b, "catalog key bases must be pairwise disjoint");
            }
        }
    }

    // ─── ADR-034 §Slice A: local-only regression guard ────

    #[test]
    fn durability_tier_has_no_partition_id_at_v1() {
        // At v1.0 DurabilityTier is partition-agnostic. If M8 adds a
        // partition field, it should be a separate variant or a
        // sibling type — this test exists to catch a premature
        // distribution commit per ADR-024 amendment 02 §posture.
        let t = DurabilityTier::Strict;
        let size = std::mem::size_of_val(&t);
        // Strict is the 1-variant no-payload case; 16B enum alignment
        // leaves room for the 8B rpo_ms on Periodic. A partition_id
        // u32 would bump alignment. If this fails, ADR-024-amendment-02's
        // local-only posture is being violated.
        assert!(
            size <= 16,
            "DurabilityTier size {size} exceeds 16B; partition_id likely added prematurely"
        );
    }

    // ─── M10 stage-1 (ADR-207): attach_page_store protocol ─────────

    use crate::metrics::{CountingMetricsSink, MetricsSink, StoragePageKind};

    /// Pool + io + counting sink over a SHARED io so a second attach
    /// can observe the first attach's on-disk page.
    fn make_sinked_pool(io: &Arc<InMemoryPageIo>) -> (Arc<BufferPool>, Arc<CountingMetricsSink>) {
        let sink = Arc::new(CountingMetricsSink::new());
        let sink_dyn: Arc<dyn MetricsSink> = sink.clone();
        let io_dyn: Arc<dyn crate::io::PageIo> = Arc::clone(io) as _;
        let pool = Arc::new(BufferPool::new(8, io_dyn).with_metrics_sink(sink_dyn));
        (pool, sink)
    }

    #[test]
    fn attach_fresh_materializes_verifies_and_fires_miss() {
        let io = Arc::new(InMemoryPageIo::new());
        let (pool, sink) = make_sinked_pool(&io);
        let mgr = TxnManager::new();
        let cat = SystemCatalog::new();
        cat.bootstrap(&pool, &mgr).unwrap();

        let report = cat
            .attach_page_store(Arc::clone(&pool), Arc::clone(&io) as _)
            .expect("attach");
        // Fresh dir: no prior page, no corruption, no divergence.
        assert!(report.prior_registry.is_none());
        assert!(!report.prior_diverged);
        assert!(!report.healed_corruption);
        // The verify read served a REAL page read through the pool —
        // the §10.2 hit-rate producer fired (count > 0 strong oracle).
        assert!(
            sink.storage_pages_count(StoragePageKind::Miss) > 0,
            "fresh attach must fire ≥1 Miss (the verify pin_read cold-loads page 0)"
        );
        // And the page on disk decodes to the live registry.
        let mut buf = [0u8; arcgraph_core::PAGE_SIZE];
        io.read_page(CATALOG_PAGE_ID, &mut buf)
            .expect("page 0 written");
        let decoded = page::decode_catalog_page(&buf).expect("decodes");
        assert_eq!(decoded, cat.list_tenants());
    }

    #[test]
    fn reattach_reports_prior_registry_and_fires_pins() {
        // First "process": bootstrap + attach writes the page.
        let io = Arc::new(InMemoryPageIo::new());
        {
            let (pool, _sink) = make_sinked_pool(&io);
            let mgr = TxnManager::new();
            let cat = SystemCatalog::new();
            cat.bootstrap(&pool, &mgr).unwrap();
            cat.attach_page_store(pool, Arc::clone(&io) as _)
                .expect("first attach");
        }
        // Second "process" over the same io: read-back finds the
        // prior registry, byte-stable (same single DEFAULT tenant —
        // created_lsn is 1 in both fresh bootstraps).
        let (pool2, sink2) = make_sinked_pool(&io);
        let mgr2 = TxnManager::new();
        let cat2 = SystemCatalog::new();
        cat2.bootstrap(&pool2, &mgr2).unwrap();
        let report = cat2
            .attach_page_store(pool2, Arc::clone(&io) as _)
            .expect("second attach");
        let prior = report
            .prior_registry
            .expect("prior page present on restart");
        assert_eq!(
            prior,
            cat2.list_tenants(),
            "stable registry ⇒ no divergence"
        );
        assert!(!report.prior_diverged);
        // Restart attach = 1 Miss (cold read-back) + ≥2 Hit
        // (materialize pin_write + verify pin_read on the resident
        // page) — the ADR-207 D-3 metric truth table.
        assert_eq!(sink2.storage_pages_count(StoragePageKind::Miss), 1);
        assert!(sink2.storage_pages_count(StoragePageKind::Hit) >= 2);
    }

    #[test]
    fn tier_mutation_writes_through_and_survives_reattach() {
        let io = Arc::new(InMemoryPageIo::new());
        let (pool, sink) = make_sinked_pool(&io);
        let mgr = TxnManager::new();
        let cat = SystemCatalog::new();
        cat.bootstrap(&pool, &mgr).unwrap();
        cat.attach_page_store(Arc::clone(&pool), Arc::clone(&io) as _)
            .expect("attach");

        let hits_before = sink.storage_pages_count(StoragePageKind::Hit);
        let mut tx = mgr.begin(TenantId::SYSTEM);
        cat.set_durability_tier(
            &mut tx,
            TenantId::DEFAULT,
            DurabilityTier::Periodic { rpo_ms: 500 },
        )
        .expect("set tier");
        tx.commit().expect("commit");
        // The write-through pinned the resident page: +≥1 Hit.
        assert!(
            sink.storage_pages_count(StoragePageKind::Hit) > hits_before,
            "tier write-through must pin the catalog page"
        );

        // A second "process" reading the page back sees Periodic —
        // and DIVERGES from its own fresh bootstrap (Strict), which is
        // exactly the ADR-183 stage-2 gap the report surfaces.
        let (pool2, _sink2) = make_sinked_pool(&io);
        let mgr2 = TxnManager::new();
        let cat2 = SystemCatalog::new();
        cat2.bootstrap(&pool2, &mgr2).unwrap();
        let report = cat2
            .attach_page_store(pool2, Arc::clone(&io) as _)
            .expect("re-attach");
        let prior = report.prior_registry.expect("prior page present");
        assert_eq!(
            prior[0].tier,
            DurabilityTier::Periodic { rpo_ms: 500 },
            "page carried the prior process's tier mutation"
        );
        assert!(
            report.prior_diverged,
            "fresh Strict bootstrap ≠ prior Periodic page"
        );
        // Stage-1 scope fence: the in-memory registry was NOT seeded
        // from the page (recovery is M10 stage-2).
        assert_eq!(
            cat2.durability_tier(TenantId::DEFAULT),
            DurabilityTier::Strict
        );
    }

    #[test]
    fn revert_tier_change_re_mirrors_the_page() {
        let io = Arc::new(InMemoryPageIo::new());
        let (pool, _sink) = make_sinked_pool(&io);
        let mgr = TxnManager::new();
        let cat = SystemCatalog::new();
        cat.bootstrap(&pool, &mgr).unwrap();
        cat.attach_page_store(Arc::clone(&pool), Arc::clone(&io) as _)
            .expect("attach");

        let mut tx = mgr.begin(TenantId::SYSTEM);
        cat.set_durability_tier(
            &mut tx,
            TenantId::DEFAULT,
            DurabilityTier::Periodic { rpo_ms: 500 },
        )
        .expect("set tier");
        // Simulate a failed commit: operator reverts.
        drop(tx);
        cat.revert_tier_change(TenantId::DEFAULT, DurabilityTier::Strict);

        // The page mirrors the restored registry.
        let mut buf = [0u8; arcgraph_core::PAGE_SIZE];
        io.read_page(CATALOG_PAGE_ID, &mut buf).expect("page 0");
        let decoded = page::decode_catalog_page(&buf).expect("decodes");
        assert_eq!(decoded[0].tier, DurabilityTier::Strict);
    }

    #[test]
    fn attach_self_heals_corrupt_page() {
        let io = Arc::new(InMemoryPageIo::new());
        // Plant a corrupt (readable, non-zero, undecodable) page 0:
        // valid magic + version but a garbage CRC region.
        let mut bad = [0u8; arcgraph_core::PAGE_SIZE];
        bad[0..8].copy_from_slice(page::CATALOG_PAGE_MAGIC);
        bad[8..10].copy_from_slice(&page::CATALOG_PAGE_VERSION.to_le_bytes());
        bad[12..16].copy_from_slice(&8u32.to_le_bytes());
        bad[16..20].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        io.write_page(CATALOG_PAGE_ID, &bad).unwrap();

        let (pool, _sink) = make_sinked_pool(&io);
        let mgr = TxnManager::new();
        let cat = SystemCatalog::new();
        cat.bootstrap(&pool, &mgr).unwrap();
        let report = cat
            .attach_page_store(pool, Arc::clone(&io) as _)
            .expect("attach must not brick on corruption");
        assert!(report.healed_corruption);
        assert!(report.prior_registry.is_none());
        // Healed: the page now decodes to the live registry.
        let mut buf = [0u8; arcgraph_core::PAGE_SIZE];
        io.read_page(CATALOG_PAGE_ID, &mut buf).expect("page 0");
        assert_eq!(
            page::decode_catalog_page(&buf).expect("healed page decodes"),
            cat.list_tenants()
        );
    }

    #[test]
    fn no_attach_means_no_write_through_no_pins() {
        // Legacy callers that never attach keep the pre-ADR-207
        // posture: tier mutations succeed and page 0 stays unwritten.
        let io = Arc::new(InMemoryPageIo::new());
        let (pool, sink) = make_sinked_pool(&io);
        let mgr = TxnManager::new();
        let cat = SystemCatalog::new();
        cat.bootstrap(&pool, &mgr).unwrap();
        let mut tx = mgr.begin(TenantId::SYSTEM);
        cat.set_durability_tier(
            &mut tx,
            TenantId::DEFAULT,
            DurabilityTier::Periodic { rpo_ms: 500 },
        )
        .expect("set tier");
        tx.commit().expect("commit");
        assert_eq!(sink.storage_pages_count(StoragePageKind::Hit), 0);
        assert_eq!(sink.storage_pages_count(StoragePageKind::Miss), 0);
        let mut buf = [0u8; arcgraph_core::PAGE_SIZE];
        assert!(
            io.read_page(CATALOG_PAGE_ID, &mut buf).is_err(),
            "page 0 never written"
        );
    }
}
