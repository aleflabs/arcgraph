//! On-disk page store wire-through (W26-ε-2 / ADR-140).
//!
//! Provides the substrate that closes the W22-DB-α-1-cap RSS-linear
//! ingest blocker by bounding the resident page working set.
//!
//! # Substrate model
//!
//! Today the slotted-record page store is held in a
//! `DashMap<PageId, Arc<RwLock<Box<PageBuf>>>>`
//! ([`crate::record_store::RecordPageStore`]); RSS scales linearly
//! with the ingested record count. The wire-through introduces:
//!
//! 1. [`RecordPageBackend`] — an adapter trait that both the legacy
//!    DashMap store AND the new buffer-pool-fronted store implement.
//!    CrudStore consumers reach pages through `&dyn RecordPageBackend`
//!    rather than a concrete type.
//! 2. [`PerTenantBufferPool`] — a per-tenant
//!    [`crate::buffer::BufferPool`] registry. Each tenant gets a
//!    fixed-size pool backed by a shared [`PageIo`]. The pool's
//!    clock-sweep eviction (M1-12 / design-v2 §3.4) is the
//!    LRU-equivalent realization per the ADR-140 D-2 rationale
//!    (Postgres / Oracle / InnoDB all use the same approximation
//!    with O(1) bookkeeping; strict-LRU regresses p99 pin latency
//!    5-10×).
//! 3. [`BufferedRecordPageStore`] — a NEW page store that combines
//!    the legacy DashMap as an in-memory hot tier with the
//!    `PerTenantBufferPool` as a spill-to-disk tier. RSS is bounded
//!    by `cache_cap × PAGE_SIZE` + `frames_per_tenant × PAGE_SIZE ×
//!    active_tenants`, NOT by total ingested page count.
//!
//! # Eviction discipline (ADR-140 §Open questions)
//!
//! v1.1-α ships an EXPLICIT eviction API ([`BufferedRecordPageStore::evict_lru`])
//! invoked by ops + tests + the perf demo. Implicit cap-driven eviction
//! at install time (so the CRUD hot path bounds RSS without operator
//! action) is forward-deferred to a v1.1 follow-up that handles the
//! eviction-vs-install race surfaced in the ADR-140 D-3 §"Race window"
//! analysis. The substrate is shipped here; the implicit-evict policy
//! lands once the race-resolution discipline is pinned.
//!
//! # WAL replay co-coordination (ADR-140 D-4)
//!
//! [`BufferedRecordPageStore`] implements
//! [`crate::wal::replay::RecordPageStoreHandle`] so the replay
//! executor's `install_or_replace` dispatch routes through it
//! transparently. Replay writes go to the hot cache; the perf demo +
//! recovery test verify byte-equality survives a `evict_lru` +
//! `fault_in` round-trip.

use std::collections::{BTreeSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arcgraph_core::{
    ArcGraphError, PAGE_SIZE, PageHeader, PageId, PageType, Result as CoreResult, TenantId,
};
use dashmap::DashMap;
use parking_lot::{Mutex, RwLock};

use crate::buffer::{BufferPool, DEFAULT_WRITE_FRACTION};
use crate::io::{PageBuf, PageIo, PosixPageIo};
use crate::metrics::MetricsSink;
use crate::mutation_log::{PageStoreKind, TxnMutationLog};
use crate::pin::{PinGuard, PinRegistry};
use crate::record_store::{RecordPageLatch, RecordPageStore, RecordStoreError};
use crate::records::SlottedPage;
use crate::wal::replay::RecordPageStoreHandle;

// ─────────────────────────────────────────────────────────────────────
// PinnedPageLatch — the ADR-140-amendment-01 latch/pin coupling
// ─────────────────────────────────────────────────────────────────────

/// A page latch whose lifetime is coupled to a frame pin
/// (ADR-140-amendment-01 §Decision item 2). While this wrapper lives,
/// [`BufferedRecordPageStore::try_evict_page_pinned`] refuses to remove
/// the frame — the `strong_count`-snapshot TOCTOU (ADR-140 §D-3 "Race
/// window") is closed by construction for every caller that acquires
/// through [`BufferedRecordPageStore::latch_pinned`].
///
/// Drop order: the latch (and any `read()`/`write()` guards derived
/// from it) must be released before or with the pin — enforced
/// structurally by field order (`latch` drops before `_pin`).
#[derive(Debug)]
pub struct PinnedPageLatch {
    /// The page latch. Callers take `latch().read()` / `latch().write()`.
    latch: RecordPageLatch,
    /// The frame pin, held for the wrapper's whole lifetime.
    _pin: PinGuard<RecordPageKey>,
}

/// Tenant-qualified physical identity for an M3 record page.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RecordPageKey {
    pub tenant_id: TenantId,
    pub store_id: u16,
    pub generation_id: GenerationId,
    pub page_id: PageId,
}

/// Stable in-process identity for one immutable data-dir generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GenerationId(u64);

impl GenerationId {
    /// Ephemeral stores have no data-dir generation.
    pub const EPHEMERAL: Self = Self(0);

    /// Derive a non-zero identity from the generation's canonical path. The
    /// key is process-local (cache/DPT lifetime), so it need not be persisted.
    #[must_use]
    pub fn for_path(path: &Path) -> Self {
        let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        canonical.hash(&mut hasher);
        let raw = hasher.finish();
        Self(if raw == 0 { 1 } else { raw })
    }
}

/// Store- and generation-qualified owner for record-page cache keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageStoreIdentity {
    pub generation_id: GenerationId,
    pub store_id: u16,
}

impl PageStoreIdentity {
    #[must_use]
    pub const fn ephemeral(store_id: u16) -> Self {
        Self {
            generation_id: GenerationId::EPHEMERAL,
            store_id,
        }
    }

    #[must_use]
    pub fn for_generation(generation: &Path, store_id: u16) -> Self {
        Self {
            generation_id: GenerationId::for_path(generation),
            store_id,
        }
    }

    #[must_use]
    pub const fn page_key(self, tenant_id: TenantId, page_id: PageId) -> RecordPageKey {
        RecordPageKey {
            tenant_id,
            store_id: self.store_id,
            generation_id: self.generation_id,
            page_id,
        }
    }
}

impl RecordPageKey {
    #[must_use]
    pub const fn new(tenant_id: TenantId, page_id: PageId) -> Self {
        PageStoreIdentity::ephemeral(crate::wal::STORE_RECORD).page_key(tenant_id, page_id)
    }
}

impl PinnedPageLatch {
    /// The underlying latch (clone-free access; guards derived from it
    /// must not outlive `self` — they borrow the `Arc` this holds).
    #[must_use]
    pub fn latch(&self) -> &RecordPageLatch {
        &self.latch
    }
}

// ─────────────────────────────────────────────────────────────────────
// AnyPinnedPageLatch — trait-object-safe pin/latch coupling
// ─────────────────────────────────────────────────────────────────────

/// Backend-erased counterpart of [`PinnedPageLatch`], returned by
/// [`RecordPageBackend::latch_pinned_for_tenant`] so every v9-era
/// writer (Phase-3 delta apply, recovery redo) can acquire through the
/// pin-coupled seam via `&dyn RecordPageBackend` — the trait object
/// `crud.rs`'s `record_backend()` hands out — without the trait
/// depending on `BufferedRecordPageStore`'s concrete `PinGuard` type.
///
/// The default trait impl (used by the legacy v8-era
/// [`crate::record_store::RecordPageStore`], which never evicts —
/// there is no `PinRegistry` to register with) wraps
/// [`RecordPageBackend::latch_for_tenant`]'s bare latch with
/// `AnyPin::Inert`: sound because that store has no removal claim
/// for a pin to ever exclude. [`BufferedRecordPageStore`] overrides
/// the trait method to delegate to its real
/// [`BufferedRecordPageStore::latch_pinned_for_tenant`], carrying an
/// actual [`PinGuard`] that [`BufferedRecordPageStore::try_evict_page_pinned_for_tenant`]'s
/// removal claim excludes on.
///
/// Drop order: `latch` before `pin` (field order), matching
/// [`PinnedPageLatch`]'s contract.
#[derive(Debug)]
pub struct AnyPinnedPageLatch {
    latch: RecordPageLatch,
    _pin: AnyPin,
}

/// The pin half of [`AnyPinnedPageLatch`] — either a real frame pin
/// (buffer-pool-backed stores) or [`AnyPin::Inert`] (the legacy
/// never-evicts store, where no pin registry exists to exclude
/// against).
#[derive(Debug)]
enum AnyPin {
    /// Held ONLY for its `Drop` side-effect (decrementing the frame's
    /// pin count on last-drop) — never read.
    Real(#[allow(dead_code)] PinGuard<RecordPageKey>),
    /// No pin registry backs this store (it never evicts a resident
    /// page), so there is nothing to exclude — safe by construction,
    /// not by omission.
    Inert,
}

impl From<PinnedPageLatch> for AnyPinnedPageLatch {
    fn from(p: PinnedPageLatch) -> Self {
        Self {
            latch: p.latch,
            _pin: AnyPin::Real(p._pin),
        }
    }
}

impl AnyPinnedPageLatch {
    /// Wrap a bare latch with no pin registry backing it (the legacy
    /// v8-era [`crate::record_store::RecordPageStore`] default path —
    /// sound because that store never evicts a resident page).
    #[must_use]
    fn inert(latch: RecordPageLatch) -> Self {
        Self {
            latch,
            _pin: AnyPin::Inert,
        }
    }

    /// The underlying latch (clone-free access; guards derived from it
    /// must not outlive `self` — they borrow the `Arc` this holds).
    #[must_use]
    pub fn latch(&self) -> &RecordPageLatch {
        &self.latch
    }
}

// ─────────────────────────────────────────────────────────────────────
// RecordPageBackend — adapter trait
// ─────────────────────────────────────────────────────────────────────

/// Adapter trait over the legacy [`RecordPageStore`] (DashMap-RAM) and
/// the new [`BufferedRecordPageStore`] (cache + spill).
///
/// CrudStore consumers reach pages through `&dyn RecordPageBackend`
/// rather than a concrete type so the wire-through can swap backends
/// without touching call sites.
///
/// Per `feedback_unification_no_consumer_logic.md` — the trait is the
/// canonical extension point; backends ship constructors + identity
/// helpers, not consumer-mode toggles.
pub trait RecordPageBackend: Send + Sync {
    /// Install a freshly-initialized slotted page under `page_id` with
    /// `page_type` (one of `PageType::Node` or `PageType::Rel`).
    /// See [`RecordPageStore::install_fresh`] for the canonical
    /// semantics.
    fn install_fresh(
        &self,
        page_id: PageId,
        page_type: PageType,
        tenant: TenantId,
    ) -> Result<(), RecordStoreError>;

    /// Transactional install — records the new page in `log.new_pages`
    /// so ADR-033 Z-1 (b) rollback can drop it on WAL fsync failure.
    /// See [`RecordPageStore::install_fresh_for_txn`].
    fn install_fresh_for_txn(
        &self,
        log: &mut TxnMutationLog,
        page_id: PageId,
        page_type: PageType,
        tenant: TenantId,
    ) -> Result<(), RecordStoreError>;

    /// Capture the page's pre-mutation bytes into `log` (idempotent
    /// per-txn) and return a write-latch for in-place mutation. See
    /// [`RecordPageStore::capture_and_write`].
    fn capture_and_write(
        &self,
        log: &mut TxnMutationLog,
        page_id: PageId,
    ) -> Result<RecordPageLatch, RecordStoreError>;

    /// Tenant-qualified capture used by the production M3 record tier.
    fn capture_and_write_for_tenant(
        &self,
        log: &mut TxnMutationLog,
        tenant: TenantId,
        page_id: PageId,
    ) -> Result<RecordPageLatch, RecordStoreError> {
        let _ = tenant;
        self.capture_and_write(log, page_id)
    }

    /// Return the page latch for read or write. Errors if the page is
    /// not mapped.
    fn latch(&self, page_id: PageId) -> Result<RecordPageLatch, RecordStoreError>;

    /// Tenant-qualified latch used by the production M3 record tier.
    fn latch_for_tenant(
        &self,
        tenant: TenantId,
        page_id: PageId,
    ) -> Result<RecordPageLatch, RecordStoreError> {
        let _ = tenant;
        self.latch(page_id)
    }

    /// Pin-coupled tenant-qualified latch — the ONLY acquisition path
    /// every v9-era writer that marks a page dirty (Phase-3 delta
    /// apply, recovery redo) MUST use (page_store.rs's documented MUST
    /// on [`BufferedRecordPageStore::latch_pinned_for_tenant`]).
    /// While the returned [`AnyPinnedPageLatch`] lives, a concurrent
    /// evictor's pin-coupled removal claim is excluded — closing the
    /// MECH-E3 revalidate-to-removal race for a re-dirty landing in
    /// that window (#1521 M6.1 P0-1).
    ///
    /// Default impl: wraps [`Self::latch_for_tenant`]'s bare latch with
    /// an inert (no-op) pin — sound ONLY for a backend with no eviction
    /// / no pin registry (the legacy v8-era
    /// [`crate::record_store::RecordPageStore`]). A backend that CAN
    /// evict a resident frame MUST override this with a real pin (see
    /// [`BufferedRecordPageStore`]'s impl).
    fn latch_pinned_for_tenant(
        &self,
        tenant: TenantId,
        page_id: PageId,
    ) -> Result<AnyPinnedPageLatch, RecordStoreError> {
        self.latch_for_tenant(tenant, page_id)
            .map(AnyPinnedPageLatch::inert)
    }

    /// Idempotent install: overwrite if present, install if not
    /// (Lemma I2 — bundle-level idempotence). Used by the WAL replay
    /// executor.
    fn install_or_replace(
        &self,
        page_id: PageId,
        page: Box<PageBuf>,
    ) -> Result<(), RecordStoreError>;

    /// Tenant-qualified idempotent install used by v9 recovery.
    fn install_or_replace_for_tenant(
        &self,
        tenant: TenantId,
        page_id: PageId,
        page: Box<PageBuf>,
    ) -> Result<(), RecordStoreError> {
        let _ = tenant;
        self.install_or_replace(page_id, page)
    }

    /// Remove a page (used by ADR-033 Z-1 (b) rollback for newly-
    /// allocated pages on a rolled-back transaction).
    fn remove_page(&self, page_id: PageId) -> Option<RecordPageLatch>;

    /// Tenant-qualified remove used by M3 rollback/retirement paths.
    fn remove_page_for_tenant(&self, tenant: TenantId, page_id: PageId) -> Option<RecordPageLatch> {
        let _ = tenant;
        self.remove_page(page_id)
    }

    /// Restore a page's bytes to a captured snapshot (ADR-033 rollback).
    fn restore_page_bytes(
        &self,
        page_id: PageId,
        pre_bytes: &PageBuf,
    ) -> Result<(), RecordStoreError>;

    /// Tenant-qualified restoration used by M3 rollback paths.
    fn restore_page_bytes_for_tenant(
        &self,
        tenant: TenantId,
        page_id: PageId,
        pre_bytes: &PageBuf,
    ) -> Result<(), RecordStoreError> {
        let _ = tenant;
        self.restore_page_bytes(page_id, pre_bytes)
    }

    /// Does the backend currently know about `page_id`? For the
    /// buffered store this returns `true` for both hot-cache and
    /// evicted-to-disk pages.
    fn contains(&self, page_id: PageId) -> bool;

    /// Tenant-qualified membership query.
    fn contains_for_tenant(&self, tenant: TenantId, page_id: PageId) -> bool {
        let _ = tenant;
        self.contains(page_id)
    }

    /// Number of pages tracked by the backend (hot cache + evicted).
    fn len(&self) -> usize;

    /// `len() == 0`.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Snapshot every tracked `(PageId, latch)`. For the buffered
    /// store this faults-in evicted pages on demand (one buffer-pool
    /// read each). Used by `CrudStore::bootstrap_primary_index`.
    fn iter_pages(&self) -> Vec<(PageId, RecordPageLatch)>;

    /// Snapshot every tenant-qualified page identity.
    fn iter_pages_qualified(&self) -> Vec<(TenantId, PageId, RecordPageLatch)> {
        self.iter_pages()
            .into_iter()
            .filter_map(|(page_id, latch)| {
                let tenant = {
                    let guard = latch.read();
                    let header: &[u8; PageHeader::SIZE] =
                        guard[..PageHeader::SIZE].try_into().ok()?;
                    PageHeader::from_bytes(header).ok()?.tenant_id
                };
                Some((TenantId::new(tenant), page_id, latch))
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────
// RecordPageBackend impl for the legacy RecordPageStore
// ─────────────────────────────────────────────────────────────────────

impl RecordPageBackend for RecordPageStore {
    fn install_fresh(
        &self,
        page_id: PageId,
        page_type: PageType,
        tenant: TenantId,
    ) -> Result<(), RecordStoreError> {
        RecordPageStore::install_fresh(self, page_id, page_type, tenant)
    }

    fn install_fresh_for_txn(
        &self,
        log: &mut TxnMutationLog,
        page_id: PageId,
        page_type: PageType,
        tenant: TenantId,
    ) -> Result<(), RecordStoreError> {
        RecordPageStore::install_fresh_for_txn(self, log, page_id, page_type, tenant)
    }

    fn capture_and_write(
        &self,
        log: &mut TxnMutationLog,
        page_id: PageId,
    ) -> Result<RecordPageLatch, RecordStoreError> {
        RecordPageStore::capture_and_write(self, log, page_id)
    }

    fn latch(&self, page_id: PageId) -> Result<RecordPageLatch, RecordStoreError> {
        RecordPageStore::latch(self, page_id)
    }

    fn install_or_replace(
        &self,
        page_id: PageId,
        page: Box<PageBuf>,
    ) -> Result<(), RecordStoreError> {
        RecordPageStore::install_or_replace(self, page_id, page)
    }

    fn remove_page(&self, page_id: PageId) -> Option<RecordPageLatch> {
        RecordPageStore::remove_page(self, page_id)
    }

    fn restore_page_bytes(
        &self,
        page_id: PageId,
        pre_bytes: &PageBuf,
    ) -> Result<(), RecordStoreError> {
        RecordPageStore::restore_page_bytes(self, page_id, pre_bytes)
    }

    fn contains(&self, page_id: PageId) -> bool {
        RecordPageStore::contains(self, page_id)
    }

    fn len(&self) -> usize {
        RecordPageStore::len(self)
    }

    fn iter_pages(&self) -> Vec<(PageId, RecordPageLatch)> {
        RecordPageStore::iter_pages(self)
    }
}

// ─────────────────────────────────────────────────────────────────────
// PerTenantBufferPool
// ─────────────────────────────────────────────────────────────────────

/// Sizing knobs for [`PerTenantBufferPool`]. Per-tenant frame count
/// drives the RSS bound (frames × `PAGE_SIZE` per tenant); the
/// write-fraction split follows design-v2 §3.4 (default 20% write).
#[derive(Debug, Clone)]
pub struct PerTenantBufferPoolConfig {
    /// Number of buffer-pool frames per tenant. Default 4 096 = 32 MiB
    /// per tenant. Tunable per deployment via
    /// [`PerTenantBufferPool::with_config`].
    pub frames_per_tenant: usize,
    /// Write-pool fraction. Default 0.20 per design-v2 §3.4.
    pub write_fraction: f64,
}

impl Default for PerTenantBufferPoolConfig {
    fn default() -> Self {
        Self {
            frames_per_tenant: 4_096,
            write_fraction: DEFAULT_WRITE_FRACTION,
        }
    }
}

/// Per-tenant [`BufferPool`] registry. Each tenant gets its own pool
/// (memory-isolated per ADR-140 D-2); all pools share the same
/// underlying [`PageIo`] (single-file substrate at v1.1-α).
pub struct PerTenantBufferPool {
    io: Arc<dyn TenantPageIo>,
    config: PerTenantBufferPoolConfig,
    pools: DashMap<TenantId, Arc<BufferPool>>,
    metrics_sink: Option<Arc<dyn MetricsSink>>,
}

/// Resolves the physical home file for one tenant. M3 record pages use a
/// separate file per tenant, so page number 1 in two tenants never aliases.
pub trait TenantPageIo: Send + Sync {
    fn io_for(&self, tenant: TenantId) -> CoreResult<Arc<dyn PageIo>>;
}

struct SharedTenantPageIo {
    io: Arc<dyn PageIo>,
}

impl TenantPageIo for SharedTenantPageIo {
    fn io_for(&self, _tenant: TenantId) -> CoreResult<Arc<dyn PageIo>> {
        Ok(Arc::clone(&self.io))
    }
}

/// Production per-tenant file resolver rooted at `<generation>/tenants`.
pub struct TenantFilePageIo {
    root: PathBuf,
    file_name: String,
    files: DashMap<TenantId, Arc<dyn PageIo>>,
}

impl TenantFilePageIo {
    #[must_use]
    pub fn new(generation: &Path, file_name: impl Into<String>) -> Self {
        Self {
            root: generation.join("tenants"),
            file_name: file_name.into(),
            files: DashMap::new(),
        }
    }

    #[must_use]
    pub fn path_for(&self, tenant: TenantId) -> PathBuf {
        self.root
            .join(tenant.raw().to_string())
            .join(&self.file_name)
    }

    /// Tenant directories already present on disk, sorted for deterministic
    /// startup scans. Non-numeric directory names are ignored.
    pub fn existing_tenants(&self) -> CoreResult<Vec<TenantId>> {
        let mut tenants = Vec::new();
        match std::fs::read_dir(&self.root) {
            Ok(entries) => {
                for entry in entries {
                    let entry = entry?;
                    if !entry.file_type()?.is_dir() {
                        continue;
                    }
                    if let Some(raw) = entry.file_name().to_str().and_then(|s| s.parse().ok())
                        && entry.path().join(&self.file_name).is_file()
                    {
                        tenants.push(TenantId::new(raw));
                    }
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        tenants.sort_unstable();
        Ok(tenants)
    }
}

impl TenantPageIo for TenantFilePageIo {
    fn io_for(&self, tenant: TenantId) -> CoreResult<Arc<dyn PageIo>> {
        if let Some(io) = self.files.get(&tenant) {
            return Ok(Arc::clone(io.value()));
        }
        let path = self.path_for(tenant);
        let parent = path.parent().expect("tenant store path has a parent");
        std::fs::create_dir_all(parent)?;
        let io: Arc<dyn PageIo> = Arc::new(PosixPageIo::open_or_create(path)?);
        use dashmap::mapref::entry::Entry;
        Ok(match self.files.entry(tenant) {
            Entry::Occupied(entry) => Arc::clone(entry.get()),
            Entry::Vacant(entry) => {
                entry.insert(Arc::clone(&io));
                io
            }
        })
    }
}

impl PerTenantBufferPool {
    /// Default-sized per-tenant registry over `io`.
    #[must_use]
    pub fn new(io: Arc<dyn PageIo>) -> Self {
        Self::with_config(io, PerTenantBufferPoolConfig::default())
    }

    /// Explicit-config constructor.
    #[must_use]
    pub fn with_config(io: Arc<dyn PageIo>, config: PerTenantBufferPoolConfig) -> Self {
        assert!(
            config.frames_per_tenant > 0,
            "frames_per_tenant must be > 0"
        );
        Self {
            io: Arc::new(SharedTenantPageIo { io }),
            config,
            pools: DashMap::new(),
            metrics_sink: None,
        }
    }

    /// Production constructor with tenant-qualified home-file resolution.
    #[must_use]
    pub fn with_tenant_io(io: Arc<dyn TenantPageIo>, config: PerTenantBufferPoolConfig) -> Self {
        assert!(
            config.frames_per_tenant > 0,
            "frames_per_tenant must be > 0"
        );
        Self {
            io,
            config,
            pools: DashMap::new(),
            metrics_sink: None,
        }
    }

    /// W16γ M6-07 / ADR-045 — attach an observability sink to all
    /// per-tenant pools (existing + future). Builder-style.
    #[must_use]
    pub fn with_metrics_sink(mut self, sink: Arc<dyn MetricsSink>) -> Self {
        self.metrics_sink = Some(sink);
        self
    }

    /// Get-or-create the pool for `tenant`. Cheap-path is one DashMap
    /// `get`; the lazy-construction path holds the per-shard write
    /// lock only while inserting the fresh pool.
    pub fn pool(&self, tenant: TenantId) -> CoreResult<Arc<BufferPool>> {
        if let Some(p) = self.pools.get(&tenant) {
            return Ok(Arc::clone(p.value()));
        }
        // Race: another thread may insert first. Use Entry to settle.
        use dashmap::mapref::entry::Entry;
        let tenant_io = self.io.io_for(tenant)?;
        Ok(match self.pools.entry(tenant) {
            Entry::Occupied(e) => Arc::clone(e.get()),
            Entry::Vacant(v) => {
                let pool = BufferPool::with_split(
                    self.config.frames_per_tenant,
                    tenant_io,
                    self.config.write_fraction,
                );
                let pool = if let Some(sink) = self.metrics_sink.as_ref() {
                    pool.with_metrics_sink(Arc::clone(sink))
                } else {
                    pool
                };
                let arc = Arc::new(pool);
                v.insert(Arc::clone(&arc));
                arc
            }
        })
    }

    /// Direct access to the underlying [`PageIo`] (shared across all
    /// per-tenant pools). Used by [`BufferedRecordPageStore`] to write
    /// evicted pages directly to disk (bypassing the buffer pool's
    /// read-before-write slow path for cold pages).
    pub fn io(&self, tenant: TenantId) -> CoreResult<Arc<dyn PageIo>> {
        self.io.io_for(tenant)
    }

    /// Shared tenant home-file resolver used by startup DWB restore.
    #[must_use]
    pub fn tenant_io(&self) -> Arc<dyn TenantPageIo> {
        Arc::clone(&self.io)
    }

    /// Sizing config (frames per tenant + write fraction).
    #[must_use]
    pub fn config(&self) -> &PerTenantBufferPoolConfig {
        &self.config
    }

    /// Number of tenants with a materialized pool.
    #[must_use]
    pub fn tenant_count(&self) -> usize {
        self.pools.len()
    }

    /// Flush every per-tenant pool's dirty frames + `fdatasync` the
    /// underlying file. Idempotent.
    pub fn flush_all(&self) -> CoreResult<()> {
        for entry in self.pools.iter() {
            entry.value().flush_all()?;
        }
        Ok(())
    }

    /// Drop a tenant-local buffer-pool mapping after canonical bytes
    /// were written directly through the shared [`PageIo`].
    pub fn invalidate_page(&self, tenant: TenantId, page_id: PageId) -> CoreResult<bool> {
        Ok(self.pool(tenant)?.invalidate_page(page_id))
    }
}

// ─────────────────────────────────────────────────────────────────────
// BufferedRecordPageStore
// ─────────────────────────────────────────────────────────────────────

/// Cache + spill record-page store (ADR-140 D-1).
///
/// Combines a hot in-memory tier ([`RecordPageStore`] DashMap) with a
/// spill-to-disk tier backed by [`PerTenantBufferPool`]. RSS is bounded
/// by `cache_cap × PAGE_SIZE` plus `frames_per_tenant × PAGE_SIZE ×
/// active_tenants` (NOT by total ingested page count).
///
/// # Eviction
///
/// v1.1-α ships explicit eviction via [`Self::evict_lru`]. The hot
/// cache grows until the operator (or a periodic task) invokes
/// `evict_lru(target_cap)`. Implicit cap-driven eviction (on every
/// install) is forward-deferred per ADR-140 §"Open questions" — the
/// race-resolution discipline lands in the v1.1 follow-up.
///
/// # Fault-in
///
/// [`Self::fault_in`] re-installs an evicted page into the hot cache
/// (one buffer-pool [`BufferPool::pin_read`] + one heap copy). The
/// existing [`RecordPageBackend::latch`] path returns
/// [`RecordStoreError::MissingPage`] for evicted pages so callers MUST
/// invoke `fault_in(pid)` before `latch(pid)` if the page may be
/// disk-resident. v1.1 follow-up wires implicit fault-in into the
/// `latch` call site.
///
/// # WAL replay co-coordination
///
/// Implements [`RecordPageStoreHandle`] so the replay executor's
/// `install_or_replace` dispatch routes through the hot cache. Replay
/// writes do NOT trigger eviction (cache grows during replay; operator
/// drives `evict_lru` after replay completes).
pub struct BufferedRecordPageStore {
    identity: PageStoreIdentity,
    pools: Arc<PerTenantBufferPool>,
    cache: DashMap<RecordPageKey, RecordPageLatch>,
    cache_cap: usize,
    /// LRU touch order. Front = oldest (eviction candidate); back =
    /// freshly touched. O(N) `position` scan is acceptable at v1.1-α
    /// cache caps (~4-128 K entries); v1.2 may swap to a doubly-linked
    /// LRU if the eviction profile justifies it.
    lru: Mutex<VecDeque<RecordPageKey>>,
    /// Page ids currently spilled to disk (NOT in hot cache).
    evicted: DashMap<RecordPageKey, ()>,
    /// ADR-140-amendment-01 pin registry: frames pinned via
    /// [`Self::latch_pinned`] / [`Self::copy_page_pinned`] are
    /// un-removable by [`Self::try_evict_page_pinned`] (and by
    /// `evict_lru`, which consults pins in addition to its legacy
    /// `strong_count` posture). O(concurrently-pinned pages).
    pins: PinRegistry<RecordPageKey>,
    /// M6.1 (MECH-E1..E8) — the M3 dirty-page table this store's pages
    /// are tracked in, when wired. `None` means the legacy v1.1-α
    /// posture: `evict_lru` remains the ungated direct-write path (no
    /// MECH-E awareness) and [`Self::evict_for_capacity`] is unavailable
    /// (returns zero evictions rather than silently doing the wrong
    /// thing). Mirrors `crud.rs`'s `attach_m3_dirty_page_table` pattern.
    m6_dpt: RwLock<Option<Arc<crate::redo::DirtyPageTable>>>,
    /// M6.1 — the write-behind checkpointer whose priority-flush
    /// handshake ([`crate::checkpoint::WriteBehindCheckpointer::flush_priority_keys`])
    /// MECH-E2 requires: the evictor never calls `write_pages_home`
    /// itself, it hands off to this checkpointer, which remains the
    /// single writer. `None` alongside `m6_dpt` `None` is the legacy
    /// posture.
    m6_checkpointer: RwLock<Option<Arc<crate::checkpoint::WriteBehindCheckpointer>>>,
    /// M6.1 MECH-E8 — bounded wait budget for the eviction driver's
    /// back-pressure loop (never spin/deadlock). Configurable so gates
    /// can exercise the bound tightly; production default is generous.
    m6_evict_wait_budget: std::time::Duration,
    /// M6.1 INV-M6.11 — resident-count fast path for
    /// [`Self::evict_for_capacity`]'s "am I even over the cap?" check.
    /// `DashMap::len()` sums every internal shard (not O(1)); calling it
    /// on EVERY commit (the production call site drives
    /// `evict_for_capacity` after every mutation) measured a ~23%
    /// regression on the hot-set-resident path in the
    /// `m6_evict_hot_path` Criterion bench — breaching the ≤10% bound.
    /// This atomic mirrors `cache`'s size, maintained at every
    /// insert/remove site (see the `record_cache_insert`/
    /// `record_cache_remove` helpers), so the hot path's cap check is
    /// one relaxed load instead of a shard-locked scan.
    m6_resident_count: AtomicUsize,
}

/// Default hot-cache cap = 4 096 pages = 32 MiB at `PAGE_SIZE = 8 KiB`.
pub const DEFAULT_CACHE_CAP_PAGES: usize = 4_096;

/// M6.1 MECH-E8 — default bounded wait per back-pressure retry inside
/// [`BufferedRecordPageStore::evict_for_capacity`]'s sweep-then-wait loop.
/// Generous for production (the priority flush should complete well
/// inside this); gates configure a much tighter bound via
/// [`BufferedRecordPageStore::with_m6_evict_wait_budget`] to exercise the
/// bound itself without a 250ms-per-iteration test-suite tax.
pub const DEFAULT_M6_EVICT_WAIT_BUDGET: std::time::Duration = std::time::Duration::from_millis(250);

/// M6.1 MECH-E8 — hard ceiling on total sweep-then-wait retries before
/// [`BufferedRecordPageStore::evict_for_capacity`] gives up and surfaces
/// an explicit resource error. Bounds the LIVENESS obligation: even a
/// pathological all-pinned pool returns control to the caller instead of
/// spinning forever.
pub const DEFAULT_M6_EVICT_MAX_RETRIES: usize = 64;

impl BufferedRecordPageStore {
    /// Construct with the default cache cap.
    #[must_use]
    pub fn new(pools: Arc<PerTenantBufferPool>) -> Self {
        Self::with_cache_cap(pools, DEFAULT_CACHE_CAP_PAGES)
    }

    /// Explicit cache-cap constructor. `cache_cap > 0`.
    #[must_use]
    pub fn with_cache_cap(pools: Arc<PerTenantBufferPool>, cache_cap: usize) -> Self {
        Self::with_cache_cap_and_identity(
            pools,
            cache_cap,
            PageStoreIdentity::ephemeral(crate::wal::STORE_RECORD),
        )
    }

    /// Explicit cache cap plus the complete physical store identity.
    #[must_use]
    pub fn with_cache_cap_and_identity(
        pools: Arc<PerTenantBufferPool>,
        cache_cap: usize,
        identity: PageStoreIdentity,
    ) -> Self {
        assert!(cache_cap > 0, "cache_cap must be > 0");
        Self {
            identity,
            pools,
            cache: DashMap::new(),
            cache_cap,
            lru: Mutex::new(VecDeque::new()),
            evicted: DashMap::new(),
            pins: PinRegistry::new(),
            m6_dpt: RwLock::new(None),
            m6_checkpointer: RwLock::new(None),
            m6_evict_wait_budget: DEFAULT_M6_EVICT_WAIT_BUDGET,
            m6_resident_count: AtomicUsize::new(0),
        }
    }

    /// Construct a default-sized cache for one production generation/store.
    #[must_use]
    pub fn with_identity(pools: Arc<PerTenantBufferPool>, identity: PageStoreIdentity) -> Self {
        Self::with_cache_cap_and_identity(pools, DEFAULT_CACHE_CAP_PAGES, identity)
    }

    #[must_use]
    pub const fn identity(&self) -> PageStoreIdentity {
        self.identity
    }

    /// Canonical key used by every cache/LRU/eviction operation.
    #[must_use]
    pub const fn page_key(&self, tenant: TenantId, page_id: PageId) -> RecordPageKey {
        self.identity.page_key(tenant, page_id)
    }

    /// Shared per-tenant pool registry.
    #[must_use]
    pub fn pools(&self) -> &Arc<PerTenantBufferPool> {
        &self.pools
    }

    /// Hot-cache cap (pages).
    #[must_use]
    pub fn cache_cap(&self) -> usize {
        self.cache_cap
    }

    /// Current hot-cache size (pages).
    #[must_use]
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Number of pages currently spilled to disk.
    #[must_use]
    pub fn evicted_count(&self) -> usize {
        self.evicted.len()
    }

    /// Total tracked pages (hot cache + evicted).
    #[must_use]
    pub fn total_pages(&self) -> usize {
        self.cache.len() + self.evicted.len()
    }

    /// Flush every per-tenant pool's dirty frames + `fdatasync` the
    /// underlying file.
    pub fn flush_all(&self) -> CoreResult<()> {
        self.pools.flush_all()
    }

    fn touch(&self, key: RecordPageKey) {
        let mut lru = self.lru.lock();
        if let Some(pos) = lru.iter().position(|candidate| *candidate == key) {
            lru.remove(pos);
        }
        lru.push_back(key);
    }

    fn untrack(&self, key: RecordPageKey) {
        let mut lru = self.lru.lock();
        if let Some(pos) = lru.iter().position(|candidate| *candidate == key) {
            lru.remove(pos);
        }
    }

    /// M6.1 INV-M6.11 — call exactly once for every `self.cache` entry
    /// that transitions vacant -> occupied. Keeps `m6_resident_count` in
    /// sync so `evict_for_capacity`'s hot-path cap check never needs
    /// `DashMap::len()`'s cross-shard scan.
    fn record_cache_insert(&self) {
        self.m6_resident_count.fetch_add(1, Ordering::Relaxed);
    }

    /// M6.1 INV-M6.11 — call exactly once for every `self.cache` entry
    /// that transitions occupied -> vacant (i.e., every successful
    /// `self.cache.remove`). See [`Self::record_cache_insert`].
    fn record_cache_remove(&self) {
        self.m6_resident_count.fetch_sub(1, Ordering::Relaxed);
    }

    /// M6.1 INV-M6.11 — O(1) resident-cache size, maintained by
    /// `record_cache_insert`/`record_cache_remove` at every mutation
    /// site. Used by `evict_for_capacity`'s hot-path check instead of
    /// `self.cache.len()` (which sums every `DashMap` shard — measured
    /// as the dominant cost of the M6.1 hot-path regression before this
    /// fix, per the `m6_evict_hot_path` Criterion bench).
    fn resident_count(&self) -> usize {
        self.m6_resident_count.load(Ordering::Relaxed)
    }

    fn cache_latch(&self, key: RecordPageKey) -> Result<RecordPageLatch, RecordStoreError> {
        self.cache
            .get(&key)
            .map(|entry| Arc::clone(entry.value()))
            .ok_or(RecordStoreError::MissingPage(key.page_id))
    }

    fn cache_install_or_replace(&self, key: RecordPageKey, page: Box<PageBuf>) {
        use dashmap::mapref::entry::Entry;
        match self.cache.entry(key) {
            Entry::Occupied(entry) => {
                let latch = Arc::clone(entry.get());
                latch.write().copy_from_slice(page.as_ref());
            }
            Entry::Vacant(entry) => {
                entry.insert(Arc::new(RwLock::new(page)));
                self.record_cache_insert();
            }
        }
    }

    /// M6.1 hardening — insert-if-VACANT variant for
    /// [`Self::fault_in_for_tenant`]. Two threads racing to fault in the
    /// SAME evicted key both read the current disk image and race to
    /// install it; `cache_install_or_replace`'s occupied-overwrite
    /// semantics let the LOSER's (older or merely-redundant) disk read
    /// clobber the WINNER's already-installed — and possibly already
    /// mutated by a third thread — frame, a lost-update under RULE-MT
    /// concurrent fault-in (distinct from, but adjacent to, the
    /// ADR-140-amendment-01 §D-3 class this module already hardens
    /// against for the evict side). Fault-in only needs "some copy of
    /// the durable image is resident"; if another racer already won,
    /// this one's read is simply discarded rather than serving as a
    /// second, redundant, clobbering writer.
    fn cache_install_if_vacant(&self, key: RecordPageKey, page: Box<PageBuf>) -> bool {
        use dashmap::mapref::entry::Entry;
        if let Entry::Vacant(entry) = self.cache.entry(key) {
            entry.insert(Arc::new(RwLock::new(page)));
            self.record_cache_insert();
            true
        } else {
            false
        }
    }

    /// Fault-in: ensure `pid` is in the hot cache. If currently evicted,
    /// load via [`BufferPool::pin_read`] (one syscall on cold cache,
    /// sub-10 μs warm OS-cache hit).
    ///
    /// Returns `Ok(())` on hit OR successful fault-in. Returns
    /// [`RecordStoreError::MissingPage`] if `pid` is not tracked at
    /// all (i.e., never installed).
    pub fn fault_in_for_tenant(
        &self,
        tenant: TenantId,
        pid: PageId,
    ) -> Result<(), RecordStoreError> {
        self.fault_in_for_tenant_with_hook_for_gate(tenant, pid, || {})
    }

    /// #1521 M6.1 P0-3 — test seam for the deterministic
    /// fault-in-vs-second-eviction rendezvous. `before_evicted_clear` runs
    /// immediately after `cache_install_if_vacant` and immediately before
    /// the `evicted` clear — exactly the window a concurrent second
    /// eviction cycle on the same key would have to land inside to
    /// reproduce the P0-3 hazard (and, post-fix, exactly the window the
    /// miss-path pin proves that cycle can no longer land inside).
    /// Production callers use [`Self::fault_in_for_tenant`].
    #[doc(hidden)]
    pub fn fault_in_for_tenant_with_hook_for_gate(
        &self,
        tenant: TenantId,
        pid: PageId,
        before_evicted_clear: impl FnOnce(),
    ) -> Result<(), RecordStoreError> {
        self.fault_in_for_tenant_inner(tenant, pid, before_evicted_clear)
            .map(|_| ())
    }

    fn fault_in_for_tenant_inner(
        &self,
        tenant: TenantId,
        pid: PageId,
        before_evicted_clear: impl FnOnce(),
    ) -> Result<bool, RecordStoreError> {
        let key = self.page_key(tenant, pid);
        // Fast path (resident hit): no map is mutated, so no claim is
        // needed. `touch` only refreshes LRU order.
        if self.cache.contains_key(&key) {
            self.touch(key);
            return Ok(false);
        }
        // #1521 M6.1 P0-3 — the miss path REGISTERS A PIN for the whole
        // re-install-then-clear sequence. `cache` and `evicted` are two
        // independent `DashMap`s with no shared lock; the ONLY claim
        // that is atomic with an evictor's two-map transaction
        // (`evicted.insert` then `cache.remove`, inside
        // `PinRegistry::remove_if_unpinned`'s shard-locked closure) is
        // the pin registry itself. A conditional `cache.contains_key`
        // guard in front of the clear (the first-draft fix) merely
        // NARROWS the race: a full second eviction cycle landing between
        // the guard's load and the `evicted.remove` still strands the
        // key in NEITHER map (fix-1's "never in neither map" invariant
        // broken → spurious `MissingPage` for a durably-tracked page).
        // Holding a pin closes it structurally: while this pin is live,
        // `remove_if_unpinned`'s claim on `key` refuses (pin count ≠ 0),
        // so no eviction can interleave between the re-install below and
        // the marker clear — the clear is safely unconditional. If a
        // claim was already in flight when we got here, the `pin()` call
        // blocks on its shard lock until that claim resolves, and the
        // re-checks below observe its outcome.
        //
        // Budget (PD#5): one DashMap entry op + `fetch_add` (~tens of
        // ns), on the MISS path only — dwarfed by the disk read this
        // path already pays. The resident fast path above is untouched.
        let _pin = self.pins.pin(key);
        // Re-check both maps under the pin: while we were acquiring it,
        // an in-flight eviction may have completed, or a concurrent
        // fault-in may have already re-installed the frame.
        if self.cache.contains_key(&key) {
            self.touch(key);
            return Ok(false);
        }
        if !self.evicted.contains_key(&key) {
            return Err(RecordStoreError::MissingPage(pid));
        }
        let pool = self.pools.pool(tenant).map_err(map_io_error)?;
        let mut bytes: Box<PageBuf> = Box::new([0u8; PAGE_SIZE]);
        {
            let guard = pool.pin_read(pid).map_err(map_io_error)?;
            bytes.copy_from_slice(guard.as_bytes());
        }
        let installed = self.cache_install_if_vacant(key, bytes);
        before_evicted_clear();
        // Unconditional by design (see the pin rationale above): with
        // the pin held since BEFORE the `evicted` re-check, no removal
        // claim can have run since, so `key` is verifiably resident at
        // this instant and the marker being cleared is verifiably this
        // fault-in's to clear. (A concurrent pinned fault-in on the same
        // key may interleave — both install-if-vacant and both clear the
        // marker; idempotent either way.)
        self.evicted.remove(&key);
        self.touch(key);
        Ok(installed)
    }

    /// Legacy default-tenant fault-in wrapper.
    pub fn fault_in(&self, pid: PageId) -> Result<(), RecordStoreError> {
        self.fault_in_for_tenant(TenantId::DEFAULT, pid)
    }

    /// Register a checkpoint-durable page that already exists in the home
    /// file without faulting its 8 KiB image into RAM. Used by M3 generation
    /// bootstrap while scanning `record.store` one page at a time.
    pub fn register_home_page(&self, page_id: PageId, tenant: TenantId) {
        let key = self.page_key(tenant, page_id);
        if !self.cache.contains_key(&key) {
            self.evicted.insert(key, ());
        }
        self.touch(key);
    }

    /// Stream every tracked page identity without materializing an O(N)
    /// page-id vector. Used during single-threaded M3 bootstrap after redo.
    pub(crate) fn for_each_tracked_page<F, E>(&self, mut visit: F) -> Result<(), E>
    where
        F: FnMut(PageId, TenantId) -> Result<(), E>,
    {
        for entry in self.cache.iter() {
            visit(entry.key().page_id, entry.key().tenant_id)?;
        }
        for entry in self.evicted.iter() {
            visit(entry.key().page_id, entry.key().tenant_id)?;
        }
        Ok(())
    }

    /// Copy the newest page bytes without faulting a checkpoint-home page
    /// into the hot cache. Redone pages are already resident; unchanged home
    /// pages stream through the bounded buffer pool one at a time.
    pub(crate) fn copy_tracked_page_for_bootstrap(
        &self,
        page_id: PageId,
        tenant: TenantId,
    ) -> Result<Box<PageBuf>, RecordStoreError> {
        let key = self.page_key(tenant, page_id);
        if let Ok(latch) = self.cache_latch(key) {
            let guard = latch.read();
            let mut bytes = Box::new([0u8; PAGE_SIZE]);
            bytes.copy_from_slice(guard.as_ref());
            return Ok(bytes);
        }
        let pool = self.pools.pool(tenant).map_err(map_io_error)?;
        let guard = pool.pin_read(page_id).map_err(map_io_error)?;
        let mut bytes = Box::new([0u8; PAGE_SIZE]);
        bytes.copy_from_slice(guard.as_bytes());
        Ok(bytes)
    }

    /// Spill LRU pages to disk until `cache.len() <= target_cap` OR no
    /// further candidates exist. Returns the count actually evicted.
    ///
    /// Skips pages with outstanding latches (`Arc::strong_count > 2`:
    /// DashMap + our temporary clone = baseline; any extra is an
    /// outstanding caller). The race-safety of the `strong_count`
    /// check is analyzed in ADR-140 §"Race window" — DashMap shard
    /// locks prevent concurrent removal during a `latch` clone, so
    /// the check truthfully reports outstanding callers.
    ///
    /// Direct `PageIo::write_page` is used (NOT `BufferPool::pin_write`)
    /// to avoid the buffer-pool's read-before-write slow path for cold
    /// pages (`PosixPageIo::read_page` would `read_exact_at` on a
    /// not-yet-written offset and error). The bytes land directly in
    /// the file, then any stale per-tenant buffer-pool mapping is
    /// invalidated so a later `pin_read` faults the canonical bytes
    /// back into a fresh buffer-pool frame.
    pub fn evict_lru(&self, target_cap: usize) -> CoreResult<usize> {
        let mut evicted = 0;
        let mut written_tenants = BTreeSet::new();
        loop {
            if self.cache.len() <= target_cap {
                break;
            }
            // Pick the oldest evictable candidate.
            let candidate = {
                let mut lru = self.lru.lock();
                let mut found = None;
                for i in 0..lru.len() {
                    let key = lru[i];
                    // ADR-140-amendment-01: a PINNED frame is
                    // un-removable regardless of strong_count — the
                    // pin-coupled surface's holders (M3 checkpointer /
                    // Phase-3 appliers) block the legacy evictor too.
                    if self.pins.pin_count(key) != 0 {
                        continue;
                    }
                    if let Ok(latch) = self.cache_latch(key) {
                        // `latch` here adds +1 to strong_count. Baseline = 2
                        // (DashMap + this temporary clone); any extra means
                        // an outstanding caller's latch — skip eviction.
                        if Arc::strong_count(&latch) == 2 {
                            found = Some((i, key, latch));
                            break;
                        }
                    }
                }
                if let Some((idx, key, latch)) = found {
                    lru.remove(idx);
                    Some((key, latch))
                } else {
                    None
                }
            };
            let Some((key, latch)) = candidate else {
                // No evictable candidates; cache may temporarily exceed cap.
                break;
            };

            // Single-file PageIo does not partition by tenant at
            // v1.1-alpha, but the tenant selects the buffer pool whose
            // cached mapping must be invalidated after the direct write.
            // Snapshot bytes (under the latch's read lock) and write
            // directly to disk via the shared PageIo.
            {
                let guard = latch.read();
                self.pools
                    .io(key.tenant_id)?
                    .write_page(key.page_id, guard.as_ref())
                    .map_err(|e| {
                        ArcGraphError::Io(std::io::Error::other(format!(
                            "BufferedRecordPageStore::evict_lru write_page({:?}) failed: {}",
                            key.page_id, e,
                        )))
                    })?;
            }
            self.pools.invalidate_page(key.tenant_id, key.page_id)?;
            written_tenants.insert(key.tenant_id);
            // Drop the copy latch, then couple the final frame removal
            // to the pin registry's exclusive claim. A pin that lands
            // after candidate selection but before this point refuses
            // removal; the page is put back on the LRU for a later pass.
            drop(latch);
            if self.remove_cached_page_if_unpinned(key, || true) {
                evicted += 1;
            } else {
                self.touch(key);
            }
        }
        // Flush the file once at the end for durability of the evicted
        // pages. The per-tenant pools' dirty frames are NOT flushed
        // here — that's `flush_all()`.
        for tenant in written_tenants {
            self.pools.io(tenant)?.flush().map_err(|e| {
                ArcGraphError::Io(std::io::Error::other(format!(
                    "BufferedRecordPageStore::evict_lru flush failed: {}",
                    e,
                )))
            })?;
        }
        Ok(evicted)
    }

    // ─────────────────────────────────────────────────────────────────
    // M6.1 — MECH-E1..E8: the buffer-pool-as-THE-tier dirty-page
    // eviction mechanism (ADR-232-amendment-01 §2, INV-M6.2 mechanism
    // form). This is the "implicit cap-driven eviction" this module's
    // header doc forward-deferred at ADR-140 ("the race-resolution
    // discipline is pinned" — pinned here).
    // ─────────────────────────────────────────────────────────────────

    /// Wire the M3 dirty-page table this store's pages are tracked in.
    /// Required (alongside [`Self::attach_m6_checkpointer`]) before
    /// [`Self::evict_for_capacity`] does anything beyond clean-page
    /// reclaim. Mirrors `CrudStore::attach_m3_dirty_page_table`.
    pub fn attach_m6_dirty_page_table(&self, dpt: Arc<crate::redo::DirtyPageTable>) {
        *self.m6_dpt.write() = Some(dpt);
    }

    /// Wire the write-behind checkpointer MECH-E2's handshake hands off
    /// to. The evictor never calls `write_pages_home` itself — see
    /// [`Self::evict_for_capacity`].
    pub fn attach_m6_checkpointer(
        &self,
        checkpointer: Arc<crate::checkpoint::WriteBehindCheckpointer>,
    ) {
        *self.m6_checkpointer.write() = Some(checkpointer);
    }

    /// M6.1 MECH-E8 tuning: override the bounded per-retry wait (default
    /// [`DEFAULT_M6_EVICT_WAIT_BUDGET`]). Builder-style so gates can
    /// configure a tight bound at construction.
    #[must_use]
    pub fn with_m6_evict_wait_budget(mut self, budget: std::time::Duration) -> Self {
        self.m6_evict_wait_budget = budget;
        self
    }

    /// This store's qualified `DirtyPageKey` for `key`, for the M3 DPT
    /// (which is keyed by `(tenant_id, store_id, page_no)` — no
    /// `generation_id`, since one `BufferedRecordPageStore` already
    /// scopes exactly one generation). MECH-E5: always fully qualified,
    /// never a bare `PageId`.
    fn dirty_page_key(&self, key: RecordPageKey) -> crate::redo::DirtyPageKey {
        crate::redo::DirtyPageKey {
            tenant_id: key.tenant_id,
            store_id: key.store_id,
            page_no: key.page_id.raw(),
        }
    }

    /// MECH-E1 — classify one candidate frame. `pinned` frames are
    /// ineligible (checked by the caller via the pin registry before
    /// this is consulted); this method distinguishes `clean` (no live
    /// DPT entry — the durable store-file image is current, reclaim
    /// immediately, no I/O) from `dirty` (live DPT entry — eligible only
    /// via the MECH-E2 handshake). Returns `None` if no DPT is wired
    /// (legacy posture: every resident page is conservatively treated as
    /// requiring the handshake path, since without a DPT we cannot prove
    /// a page is clean).
    fn is_clean(&self, key: RecordPageKey) -> Option<bool> {
        let dpt = self.m6_dpt.read();
        let dpt = dpt.as_ref()?;
        Some(dpt.snapshot_key(self.dirty_page_key(key)).is_none())
    }

    /// MECH-E1..E8 — evict resident, unpinned pages until
    /// `cache.len() <= target_cap` or no further progress is possible.
    ///
    /// Unlike the legacy [`Self::evict_lru`] (which writes bytes
    /// directly, bypassing the DPT and the WAL install-after-durability
    /// law), this method NEVER performs a page write itself (MECH-E2):
    /// - **clean candidate** (MECH-E1: no live DPT entry) — the durable
    ///   store-file image already covers it; reclaimed immediately via
    ///   the pin-coupled [`Self::try_evict_page_pinned_for_tenant`], no
    ///   I/O.
    /// - **dirty candidate** — enqueued on the write-behind
    ///   checkpointer's [`crate::checkpoint::WriteBehindCheckpointer::flush_priority_keys`]
    ///   (MECH-E2's handshake: the checkpointer remains the SOLE
    ///   `write_pages_home` caller); reclaimed only if that flush
    ///   completed AND the frame's dirty generation still matches what
    ///   was flushed (MECH-E3/E4: reclaim-after-durable-home,
    ///   durable-flush-before-prune — a re-dirty between the
    ///   checkpointer's copy and this removal RETAINS the frame, exactly
    ///   the write-behind contract's existing compare-and-remove).
    ///
    /// **MECH-E8 (liveness):** if a sweep finds no unpinned candidate at
    /// all, this method does NOT spin — it releases control back to the
    /// caller after [`DEFAULT_M6_EVICT_MAX_RETRIES`] bounded waits
    /// (each capped at `m6_evict_wait_budget`), surfacing
    /// [`ArcGraphError::BufferPoolExhausted`] rather than an unbounded
    /// stall. Progress (any single reclaim) resets the retry counter, so
    /// a pool that is merely SLOW (priority flush still catching up)
    /// keeps making forward progress indefinitely; only a pool with
    /// ZERO progress across the whole bounded window gives up.
    ///
    /// Without a wired DPT ([`Self::attach_m6_dirty_page_table`]), every
    /// resident page is treated as requiring the dirty path (safe
    /// default: we cannot prove cleanliness), and without a wired
    /// checkpointer ([`Self::attach_m6_checkpointer`]) dirty pages are
    /// never reclaimed (they simply do not become eligible) — the
    /// method degrades to "clean-only reclaim" rather than doing
    /// anything unsafe.
    pub fn evict_for_capacity(&self, target_cap: usize) -> CoreResult<usize> {
        let mut evicted = 0usize;
        let mut retries_without_progress = 0usize;

        loop {
            // INV-M6.11: `resident_count()` (one relaxed atomic load) not
            // `self.cache.len()` (a cross-shard `DashMap` scan) — this
            // check runs on every commit in production, so its cost is
            // the hot-set-resident-path floor the ≤10% bound measures.
            if self.resident_count() <= target_cap {
                break;
            }
            if retries_without_progress >= DEFAULT_M6_EVICT_MAX_RETRIES {
                return Err(ArcGraphError::BufferPoolExhausted);
            }

            // MECH-E1: sweep the LRU order for the oldest UNPINNED
            // candidate. Pinned frames (MECH-E7: Phase-3 apply pins via
            // `pin_write`-equivalent `latch_pinned`) are skipped — a
            // frame evicted before an apply reaches it is refaulted
            // then pinned then applied, never the reverse.
            let candidate = {
                let lru = self.lru.lock();
                lru.iter()
                    .find(|key| self.pins.pin_count(**key) == 0)
                    .copied()
            };

            let Some(key) = candidate else {
                // Every resident page is pinned. MECH-E8: back-pressure,
                // never deadlock — bounded wait, then re-sweep. This is
                // NOT a spin: the wait budget yields the scheduler.
                retries_without_progress += 1;
                std::thread::sleep(self.m6_evict_wait_budget);
                continue;
            };

            let reclaimed = match self.is_clean(key) {
                // No DPT wired: conservative — do not attempt reclaim of
                // an unknown-dirty page. Fall through as "no progress on
                // this candidate"; the loop's bounded-retry accounting
                // still applies so an all-conservative sweep terminates
                // rather than spinning (MECH-E8 applies to this path
                // too — a store with no DPT wired is not a liveness
                // hazard, it is a legacy no-op returning promptly below).
                None => {
                    if self.m6_dpt.read().is_none() && self.m6_checkpointer.read().is_none() {
                        // Fully legacy (no M6 wiring at all): nothing
                        // more this method can safely do. Stop cleanly
                        // rather than counting toward the bounded-retry
                        // exhaustion error (that error is reserved for
                        // "pressure with wiring", not "unwired store").
                        break;
                    }
                    false
                }
                // MECH-E1 clean: durable image already current. Reclaim
                // immediately — no I/O (MECH-E2 is trivially satisfied:
                // there is nothing to write).
                //
                // #1521 M6.1 P0-2 — the revalidate closure MUST re-check
                // `is_clean` INSIDE the same pin-coupled claim
                // `try_evict_page_pinned_for_tenant` takes, exactly like
                // `evict_dirty_via_checkpointer`'s dirty-arm claim above
                // (page_store.rs's documented "CRITICAL: the revalidate
                // closure MUST re-check the DPT is STILL clean at the
                // moment of the pin-coupled removal claim"). An
                // unconditional `|| true` here was the SAME MECH-E3
                // hazard the dirty path's own revalidate closure exists
                // to close, reopened at the sibling clean-arm site: a
                // writer can re-dirty `key` between this `is_clean`
                // classification and the removal claim below, and
                // without a re-check inside the claim the frame — now
                // the ONLY place the fresh bytes exist — gets reclaimed
                // out from under the re-dirty.
                Some(true) => {
                    self.try_evict_page_pinned_for_tenant(key.tenant_id, key.page_id, || {
                        self.is_clean(key) == Some(true)
                    })
                }
                // MECH-E1 dirty: MECH-E2 handshake via the checkpointer.
                Some(false) => self.evict_dirty_via_checkpointer(key)?,
            };

            if reclaimed {
                evicted += 1;
                retries_without_progress = 0;
            } else {
                // Candidate could not be reclaimed this pass (raced a
                // pin, a re-dirty, or no checkpointer wired for a dirty
                // page). Re-touch so the LRU order is not stuck
                // re-selecting the same un-reclaimable candidate forever
                // within one bounded-retry window.
                self.touch(key);
                retries_without_progress += 1;
            }
        }
        Ok(evicted)
    }

    /// MECH-E2's handshake for one dirty candidate: enqueue on the
    /// write-behind checkpointer's priority flush, then attempt the
    /// pin-coupled, generation-revalidated removal (MECH-E3/E4). Returns
    /// `false` (not `Err`) when the checkpointer completed the flush but
    /// a concurrent re-dirty or pin raced the removal — that is a
    /// NORMAL retained-frame outcome, not a failure; the frame simply
    /// stays resident for the caller's next sweep.
    fn evict_dirty_via_checkpointer(&self, key: RecordPageKey) -> CoreResult<bool> {
        let Some(checkpointer) = self.m6_checkpointer.read().clone() else {
            // Dirty page, no checkpointer wired: cannot safely flush.
            // Never reclaim — this is the safe degrade the doc comment
            // promises, not a silent skip of MECH-E3.
            return Ok(false);
        };
        let dirty_key = self.dirty_page_key(key);
        // MECH-E2: the ONLY call this method makes toward durability is
        // handing the key to the checkpointer. It never calls
        // `write_pages_home` / `copy_page_pinned` / any I/O primitive
        // directly — that would make this a second DWB client, exactly
        // the rejected alternative ADR-232-amendment-01 §2.2 records.
        let completed = checkpointer.flush_priority_keys(&[dirty_key])?;
        if !completed.contains(&dirty_key) {
            // Either the key was not (or no longer) dirty by the time
            // the checkpointer looked (already covered by a durable
            // home — safe to reclaim), or it WAS dirty and the flush's
            // own generation-matched removal retained it (a concurrent
            // re-dirty raced the copy). Disambiguate via a fresh DPT
            // check: if it's clean NOW, the durable home from an EARLIER
            // pass (or this one, if the key was already absent) covers
            // it and reclaim may proceed; otherwise retain.
            if self.is_clean(key) != Some(true) {
                return Ok(false);
            }
        }
        // MECH-E3/E4: reclaim only after the checkpointer's durable home
        // write completed for THIS key (or it was already clean).
        //
        // CRITICAL: the revalidate closure MUST re-check the DPT is
        // STILL clean at the moment of the pin-coupled removal claim —
        // NOT an unconditional `|| true`. There is a real window between
        // `flush_priority_keys` returning above and the removal claim
        // below during which a concurrent writer can re-dirty this exact
        // key with bytes the just-completed flush never saw; an
        // unconditional revalidate would let the frame's RAM copy (now
        // the ONLY place those fresh bytes exist) be dropped — the
        // precise MECH-E3 hazard (H1: reclaim-before-durable) the
        // checkpointer handshake exists to prevent, reopened at the
        // seam between "checkpointer confirmed durable" and "evictor
        // claims removal". Re-checking `is_clean` inside the SAME
        // pin-registry claim `try_evict_page_pinned_for_tenant` takes
        // closes it: a re-dirty that landed in the window is a live DPT
        // entry, so the closure returns `false` and the frame is
        // retained (exactly the write-behind checkpointer's own
        // generation-compare-and-remove discipline, applied at this
        // second seam).
        Ok(
            self.try_evict_page_pinned_for_tenant(key.tenant_id, key.page_id, || {
                self.is_clean(key) == Some(true)
            }),
        )
    }

    // ─────────────────────────────────────────────────────────────────
    // ADR-140-amendment-01 — the pin-coupled concurrent flush surface
    // (M3's write-behind checkpointer substrate; cx-M3-6 close)
    // ─────────────────────────────────────────────────────────────────

    /// Pin-coupled latch acquisition: fault-in if evicted, pin the
    /// frame, return the latch+pin wrapper. While the wrapper lives,
    /// [`Self::try_evict_page_pinned`] (and `evict_lru`) refuse to
    /// remove the frame — the ADR-140 §D-3 lost-write TOCTOU is closed
    /// by construction for this surface.
    ///
    /// v9-era (M3) page writers — Phase-3 delta apply, recovery redo,
    /// the checkpointer — MUST acquire through this (not bare
    /// [`RecordPageBackend::latch`]). Legacy v8-era callers keep the
    /// bare API under the serial-eviction posture (amendment item 4).
    ///
    /// # Budget (PD#5)
    ///
    /// fault-in hit-path ≈ one DashMap get; pin ≈ one entry op + one
    /// `fetch_add` (~tens of ns). No I/O unless evicted.
    pub fn latch_pinned_for_tenant(
        &self,
        tenant: TenantId,
        pid: PageId,
    ) -> Result<PinnedPageLatch, RecordStoreError> {
        // Pin FIRST, then resolve the frame: a pin taken before the
        // frame lookup can pin a key whose frame a concurrent evictor
        // is dropping — but the evictor's removability check ran
        // before our pin landed only if it also removed the frame
        // first (ordering contract), in which case our fault_in below
        // re-loads canonical bytes into a fresh frame. Either way no
        // latch is ever handed out on a frame the evictor removed
        // AFTER our pin was visible.
        let key = self.page_key(tenant, pid);
        let pin = self.pins.pin(key);
        self.fault_in_for_tenant(tenant, pid)?;
        let latch = self.cache_latch(key)?;
        self.touch(key);
        Ok(PinnedPageLatch { latch, _pin: pin })
    }

    /// Legacy default-tenant wrapper. Production M3 callers use
    /// [`Self::latch_pinned_for_tenant`].
    pub fn latch_pinned(&self, pid: PageId) -> Result<PinnedPageLatch, RecordStoreError> {
        self.latch_pinned_for_tenant(TenantId::DEFAULT, pid)
    }

    /// Copy `pid`'s bytes as a self-consistent image: pin the frame,
    /// take the page's WRITE latch for the duration of the memcpy
    /// (excluding a concurrent writer mid-mutation — the amendment's
    /// copy-under-write-latch rule), release the latch, drop the pin.
    /// The pin is held across the COPY, not any I/O (amendment item 1:
    /// commit-path writers to other pages proceed throughout; a
    /// same-page writer serializes only on this page's latch for the
    /// memcpy).
    ///
    /// Returns `None` if the page is not tracked at all.
    ///
    /// # Budget (PD#5)
    ///
    /// One 8 KiB memcpy under the write latch (~1 µs class) + pin
    /// (~tens of ns). No global freeze.
    pub fn copy_page_pinned_for_tenant(
        &self,
        tenant: TenantId,
        pid: PageId,
    ) -> Result<Option<Box<PageBuf>>, RecordStoreError> {
        if !self.contains_for_tenant(tenant, pid) {
            return Ok(None);
        }
        let pinned = self.latch_pinned_for_tenant(tenant, pid)?;
        let mut out: Box<PageBuf> = Box::new([0u8; PAGE_SIZE]);
        {
            let guard = pinned.latch().write();
            out.copy_from_slice(guard.as_ref().as_ref());
        }
        drop(pinned);
        Ok(Some(out))
    }

    /// Legacy default-tenant wrapper. Production M3 callers use
    /// [`Self::copy_page_pinned_for_tenant`].
    pub fn copy_page_pinned(&self, pid: PageId) -> Result<Option<Box<PageBuf>>, RecordStoreError> {
        self.copy_page_pinned_for_tenant(TenantId::DEFAULT, pid)
    }

    /// Home-write path (amendment item 3: "the tier owes the images +
    /// home-write path"): `pwrite` each image to its home offset via
    /// the shared [`PageIo`], then `fdatasync` once. The images are
    /// caller-copied (via [`Self::copy_page_pinned`]) so no latch or
    /// pin is held across the I/O. Flush ≠ evict: frames stay cached
    /// and serviceable.
    ///
    /// The M3 checkpointer layers the doublewrite ordering ON TOP of
    /// this (DWB batch write + fsync BEFORE these home writes —
    /// IMPL-DEC-7); this method is the step-2 home-write half.
    pub fn write_pages_home_qualified(
        &self,
        images: &[(TenantId, PageId, Box<PageBuf>)],
    ) -> CoreResult<()> {
        for (tenant, pid, bytes) in images {
            self.pools
                .io(*tenant)?
                .write_page(*pid, bytes.as_ref())
                .map_err(|e| {
                    ArcGraphError::Io(std::io::Error::other(format!(
                        "write_pages_home write_page({pid:?}) failed: {e}",
                    )))
                })?;
            // Invalidate any stale per-tenant buffer-pool mapping so a
            // later pin_read faults the canonical bytes (mirrors the
            // evict_lru direct-write discipline).
            self.pools.invalidate_page(*tenant, *pid)?;
        }
        let tenants: BTreeSet<_> = images.iter().map(|(tenant, _, _)| *tenant).collect();
        for tenant in tenants {
            self.pools.io(tenant)?.flush().map_err(|e| {
                ArcGraphError::Io(std::io::Error::other(format!(
                    "write_pages_home fsync failed for tenant {tenant:?}: {e}",
                )))
            })?;
        }
        Ok(())
    }

    /// Legacy default-tenant home-write wrapper.
    pub fn write_pages_home(&self, images: &[(PageId, Box<PageBuf>)]) -> CoreResult<()> {
        let qualified: Vec<_> = images
            .iter()
            .map(|(page_id, page)| (TenantId::DEFAULT, *page_id, page.clone()))
            .collect();
        self.write_pages_home_qualified(&qualified)
    }

    /// Flush a caller-selected page set without evicting it.
    ///
    /// Each page is pinned and copied under its page write latch; all
    /// pins/latches are released before the home I/O starts. Missing
    /// page ids are ignored, matching a DPT snapshot whose page was
    /// concurrently retired by a legacy serial caller. Returns the
    /// number of images written.
    pub fn flush_pages_qualified(
        &self,
        page_ids: impl IntoIterator<Item = (TenantId, PageId)>,
    ) -> CoreResult<usize> {
        let mut images = Vec::new();
        for (tenant, pid) in page_ids {
            match self.copy_page_pinned_for_tenant(tenant, pid) {
                Ok(Some(image)) => images.push((tenant, pid, image)),
                Ok(None) => {}
                Err(e) => {
                    return Err(ArcGraphError::Io(std::io::Error::other(format!(
                        "flush_pages copy_page_pinned({pid:?}) failed: {e}",
                    ))));
                }
            }
        }
        self.write_pages_home_qualified(&images)?;
        Ok(images.len())
    }

    /// Legacy default-tenant flush wrapper.
    pub fn flush_pages(&self, page_ids: impl IntoIterator<Item = PageId>) -> CoreResult<usize> {
        self.flush_pages_qualified(
            page_ids
                .into_iter()
                .map(|page_id| (TenantId::DEFAULT, page_id)),
        )
    }

    /// Pin-guarded eviction (amendment item 3: "evict = flush + drop
    /// under the pin discipline"). Removes `pid`'s frame from the hot
    /// cache iff (a) `revalidate()` approves — the M3 caller supplies
    /// the DPT `dirty_gen` compare so a write that landed AFTER the
    /// flush copy refuses the drop (re-dirty keeps the frame), and
    /// (b) no pin is live, decided under the pin registry's shard
    /// lock, and (c) the legacy `strong_count == 2` posture ALSO holds
    /// (bare-latch holders from the v8-era surface still block
    /// removal — belt for mixed-era callers).
    ///
    /// The caller MUST have durably written the page home (via
    /// [`Self::write_pages_home`], DWB-ordered by the checkpointer)
    /// BEFORE calling this — this method does NO I/O.
    ///
    /// Order contract (pin.rs TOCTOU close): frame removal happens
    /// immediately after the pin claim, with no latch-granting path in
    /// between; a pinner racing in after the claim finds the frame
    /// gone and faults in the (durably written) canonical bytes.
    ///
    /// Returns `true` iff the frame was removed.
    pub fn try_evict_page_pinned_for_tenant(
        &self,
        tenant: TenantId,
        pid: PageId,
        revalidate: impl FnOnce() -> bool,
    ) -> bool {
        self.try_evict_page_pinned_inner(self.page_key(tenant, pid), revalidate, || {})
    }

    /// Legacy default-tenant eviction wrapper.
    pub fn try_evict_page_pinned(&self, pid: PageId, revalidate: impl FnOnce() -> bool) -> bool {
        self.try_evict_page_pinned_for_tenant(TenantId::DEFAULT, pid, revalidate)
    }

    /// Test seam for the amendment's deterministic latch-vs-remove
    /// schedule. `before_claim` runs after the legacy strong-count
    /// snapshot and immediately before the authoritative pin claim.
    /// Production callers use [`Self::try_evict_page_pinned`].
    #[doc(hidden)]
    pub fn try_evict_page_pinned_with_hook_for_gate(
        &self,
        pid: PageId,
        revalidate: impl FnOnce() -> bool,
        before_claim: impl FnOnce(),
    ) -> bool {
        self.try_evict_page_pinned_inner(
            self.page_key(TenantId::DEFAULT, pid),
            revalidate,
            before_claim,
        )
    }

    fn try_evict_page_pinned_inner(
        &self,
        key: RecordPageKey,
        revalidate: impl FnOnce() -> bool,
        before_claim: impl FnOnce(),
    ) -> bool {
        if !self.cache.contains_key(&key) {
            return false;
        }
        // An advisory legacy-latch snapshot preserves the exact §D-3
        // race seam for the RED-on-revert gate. The authoritative
        // strong-count recheck and dirty-generation revalidation happen
        // again INSIDE the pin claim below.
        if let Ok(latch) = self.cache_latch(key) {
            if Arc::strong_count(&latch) != 2 {
                return false;
            }
            drop(latch);
        } else {
            return false;
        }
        before_claim();
        self.remove_cached_page_if_unpinned(key, revalidate)
    }

    /// Atomically couple the pin-count decision to frame-map removal.
    /// `PinRegistry` keeps the key's shard write-locked while the
    /// callback runs, so a new pinner cannot land between the decision
    /// and `cache.remove_page`.
    ///
    /// M6.1 hardening: `self.evicted.insert(key, ())` runs BEFORE
    /// `self.cache.remove(&key)` (not after). `cache` and `evicted` are
    /// two independent `DashMap`s with no shared lock between them, and
    /// [`Self::fault_in_for_tenant`] reads them as two SEPARATE,
    /// non-atomic checks (`cache.contains_key` then
    /// `evicted.contains_key`). A remove-then-insert ordering leaves a
    /// real (if narrow) window where a key is in NEITHER map; under
    /// serial/legacy eviction cadence that window was vanishingly
    /// unlikely to matter, but M6.1's continuous concurrent
    /// `evict_for_capacity` (RULE-MT, many threads racing
    /// fault-in-vs-evict on overlapping keys) hits it deterministically,
    /// surfacing a spurious `MissingPage` on the racing `fault_in`
    /// caller. Insert-before-remove closes the window: at every instant
    /// the key is in `cache`, OR in `evicted`, OR (briefly) in BOTH —
    /// never in neither. `fault_in_for_tenant`'s cache-check-first order
    /// already handles the transient both-maps window correctly (a hit
    /// there short-circuits before ever consulting `evicted`).
    fn remove_cached_page_if_unpinned(
        &self,
        key: RecordPageKey,
        revalidate: impl FnOnce() -> bool,
    ) -> bool {
        let removed = self
            .pins
            .remove_if_unpinned(key, || {
                // The DPT dirty-generation check belongs inside the
                // same claim as removal: a pinner cannot mutate between
                // this revalidation and cache.remove_page.
                if !revalidate() {
                    return false;
                }
                // Mixed-era belt. Pin-coupled callers are excluded by
                // the claim; a legacy bare latch is still refused under
                // the serial-eviction coexistence posture.
                let Ok(latch) = self.cache_latch(key) else {
                    return false;
                };
                if Arc::strong_count(&latch) != 2 {
                    return false;
                }
                drop(latch);
                self.evicted.insert(key, ());
                if self.cache.remove(&key).is_none() {
                    // Should not happen (we just verified the latch above
                    // under the same claim), but if it ever does, undo
                    // the speculative evicted-insert so a genuinely-still-
                    // cached page is never ALSO marked evicted.
                    self.evicted.remove(&key);
                    return false;
                }
                self.record_cache_remove();
                // NOTE: `self.untrack(key)` (LRU-deque bookkeeping) is
                // deliberately NOT called here, inside the pins shard's
                // write-locked claim. `untrack` takes the store-wide
                // `self.lru` mutex, which is ALSO contended by every
                // `touch()` call (one per mutation, from every thread);
                // holding the fine-grained per-key pins-shard lock while
                // also blocking on that coarse-grained global lock
                // creates a priority-inversion-shaped bottleneck under
                // RULE-MT eviction pressure (many threads touching many
                // DIFFERENT keys while one thread's removal claim sits
                // blocked on the shared `lru` mutex, observed as
                // multi-minute stalls under 8-way concurrent
                // `evict_for_capacity` pressure). Deferring `untrack` to
                // just after this closure returns (still before
                // `remove_cached_page_if_unpinned` itself returns, so
                // the overall reclaim is still complete) is safe: `lru`
                // is pure sweep-order housekeeping, not a correctness
                // structure — a transient window where an
                // already-evicted key is still enqueued in `lru` only
                // costs a future sweep one wasted candidate (rejected at
                // `try_evict_page_pinned_inner`'s `cache.contains_key`
                // check), never a wrong reclaim.
                true
            })
            .unwrap_or(false);
        if removed {
            self.untrack(key);
        }
        removed
    }

    /// TEST-ONLY (the ADR-140-amendment-01 item-5 RED-on-revert lever):
    /// reproduce the LEGACY bare-`strong_count` eviction with an
    /// injectable schedule hook BETWEEN the liveness snapshot and the
    /// frame removal — the exact §D-3 race window. The concurrent gate
    /// proves the pinned path survives the schedule that makes THIS
    /// path lose a write. Never called outside tests.
    #[doc(hidden)]
    pub fn evict_page_bare_strongcount_for_gate(
        &self,
        pid: PageId,
        between_check_and_remove: impl FnOnce(),
    ) -> bool {
        let key = self.page_key(TenantId::DEFAULT, pid);
        let Ok(latch) = self.cache_latch(key) else {
            return false;
        };
        if Arc::strong_count(&latch) != 2 {
            return false;
        }
        // Snapshot bytes + direct write (the legacy evict_lru shape).
        {
            let guard = latch.read();
            if self
                .pools
                .io(TenantId::DEFAULT)
                .and_then(|io| io.write_page(pid, guard.as_ref()))
                .is_err()
            {
                return false;
            }
        }
        if self.pools.invalidate_page(TenantId::DEFAULT, pid).is_err() {
            return false;
        }
        drop(latch);
        // ── THE §D-3 RACE WINDOW: a concurrent writer can latch +
        // mutate here; the remove below then discards its write. ──
        between_check_and_remove();
        if self.cache.remove(&key).is_some() {
            self.record_cache_remove();
        }
        self.untrack(key);
        self.evicted.insert(key, ());
        if let Ok(io) = self.pools.io(TenantId::DEFAULT) {
            let _ = io.flush();
        }
        true
    }

    /// Observability: current pin count on `pid` (0 = unpinned).
    #[must_use]
    pub fn pin_count(&self, pid: PageId) -> usize {
        self.pins.pin_count(self.page_key(TenantId::DEFAULT, pid))
    }

    /// Test/observability: is `pid` currently spilled to disk?
    #[doc(hidden)]
    #[must_use]
    pub fn is_evicted(&self, pid: PageId) -> bool {
        self.evicted
            .contains_key(&self.page_key(TenantId::DEFAULT, pid))
    }

    /// Test/observability: is `pid` currently in the hot cache?
    #[doc(hidden)]
    #[must_use]
    pub fn is_cached(&self, pid: PageId) -> bool {
        self.cache
            .contains_key(&self.page_key(TenantId::DEFAULT, pid))
    }

    /// #1521 M6.1 P0-3 sensitivity-leg seam ONLY — reproduces the
    /// PRE-FIX `fault_in_for_tenant` shape's unconditional `evicted`
    /// clear (no `cache.contains_key` re-check), so
    /// `skeptic_fault_in_evicted_remove_races_evictor.rs`'s sensitivity
    /// leg can demonstrate the defect class directly. NEVER called by
    /// [`Self::fault_in_for_tenant_with_hook_for_gate`] or any
    /// production path.
    #[doc(hidden)]
    pub fn __test_blind_evicted_remove_for_gate(&self, tenant: TenantId, pid: PageId) {
        let key = self.page_key(tenant, pid);
        self.evicted.remove(&key);
    }

    /// #1457 MF4 gate seam ONLY — a pin with NO accompanying latch
    /// clone. [`Self::latch_pinned_for_tenant`] always returns pin AND
    /// latch coupled in one [`PinnedPageLatch`] wrapper (by design —
    /// production callers must never decompose them, see the type's own
    /// "Drop order" doc), so a caller cannot use the public API to hold
    /// ONLY a pin: `Arc::strong_count(&latch)` on any frame a
    /// `PinnedPageLatch` still lives for is always inflated by that
    /// wrapper's own retained clone, which means the LEGACY
    /// `Arc::strong_count != 2` belt (page_store.rs's mixed-era-callers
    /// safety net, `try_evict_page_pinned_inner` /
    /// `remove_cached_page_if_unpinned`) coincidentally shields the
    /// frame too — so a test built entirely from the public API cannot
    /// isolate whether the PIN itself (vs. the belt) is what excludes
    /// removal. This seam returns a bare `PinGuard` with no latch at
    /// all, closing that gap: with only this guard alive,
    /// `Arc::strong_count(&latch)` for the frame is back at the
    /// baseline 2 (DashMap's own entry + one transient check-clone), so
    /// a gate using this seam is exercising the PIN-COUNT check in
    /// `PinRegistry::remove_if_unpinned` exclusively — a mutant that
    /// neuters that check (e.g. always treating the occupied entry as
    /// zero-count) is caught ONLY by a gate built this way, never by one
    /// where a `PinnedPageLatch` (or any other latch clone) is also
    /// live. NEVER called by production.
    #[doc(hidden)]
    #[must_use]
    pub fn __test_pin_only_for_gate(
        &self,
        tenant: TenantId,
        pid: PageId,
    ) -> crate::pin::PinGuard<RecordPageKey> {
        self.pins.pin(self.page_key(tenant, pid))
    }
}

fn map_io_error(e: ArcGraphError) -> RecordStoreError {
    RecordStoreError::Codec(crate::records::PageError::Format(format!(
        "BufferedRecordPageStore I/O error: {}",
        e,
    )))
}

// ─────────────────────────────────────────────────────────────────────
// RecordPageBackend impl
// ─────────────────────────────────────────────────────────────────────

impl RecordPageBackend for BufferedRecordPageStore {
    fn install_fresh(
        &self,
        page_id: PageId,
        page_type: PageType,
        tenant: TenantId,
    ) -> Result<(), RecordStoreError> {
        if !matches!(
            page_type,
            PageType::Node | PageType::Rel | PageType::PropSlotted
        ) {
            return Err(RecordStoreError::UnsupportedPageType {
                got: page_type.as_byte(),
            });
        }
        let key = self.page_key(tenant, page_id);
        if self.cache.contains_key(&key) || self.evicted.contains_key(&key) {
            return Err(RecordStoreError::DuplicatePage(page_id));
        }
        let mut page = Box::new([0u8; PAGE_SIZE]);
        SlottedPage::init(page.as_mut(), PageHeader::new(page_id, page_type, tenant))?;
        self.cache.insert(key, Arc::new(RwLock::new(page)));
        self.record_cache_insert();
        self.touch(key);
        Ok(())
    }

    fn install_fresh_for_txn(
        &self,
        log: &mut TxnMutationLog,
        page_id: PageId,
        page_type: PageType,
        tenant: TenantId,
    ) -> Result<(), RecordStoreError> {
        self.install_fresh(page_id, page_type, tenant)?;
        log.new_pages.push((PageStoreKind::Record, page_id));
        Ok(())
    }

    fn capture_and_write(
        &self,
        log: &mut TxnMutationLog,
        page_id: PageId,
    ) -> Result<RecordPageLatch, RecordStoreError> {
        // Ensure cache resident (fault-in if needed) BEFORE the
        // capture-and-write touch — the underlying RecordPageStore
        // call needs the page mapped.
        self.capture_and_write_for_tenant(log, TenantId::DEFAULT, page_id)
    }

    fn capture_and_write_for_tenant(
        &self,
        log: &mut TxnMutationLog,
        tenant: TenantId,
        page_id: PageId,
    ) -> Result<RecordPageLatch, RecordStoreError> {
        self.fault_in_for_tenant(tenant, page_id)?;
        let key = self.page_key(tenant, page_id);
        let latch = self.cache_latch(key)?;
        if !log.has_captured(PageStoreKind::Record, page_id) {
            let mut snapshot = Box::new([0u8; PAGE_SIZE]);
            snapshot.copy_from_slice(latch.read().as_ref().as_ref());
            log.page_mutations
                .push((PageStoreKind::Record, page_id, snapshot));
        }
        self.touch(key);
        Ok(latch)
    }

    fn latch(&self, page_id: PageId) -> Result<RecordPageLatch, RecordStoreError> {
        // M3 lifts implicit fault-in into the canonical backend seam: durable
        // generation pages begin disk-resident and CRUD must not confuse a
        // cold page with a missing page.
        self.latch_for_tenant(TenantId::DEFAULT, page_id)
    }

    fn latch_for_tenant(
        &self,
        tenant: TenantId,
        page_id: PageId,
    ) -> Result<RecordPageLatch, RecordStoreError> {
        // M6.1 hardening: `fault_in_for_tenant` (ensure resident) and
        // `cache_latch` (get the handle) are two SEPARATE, non-atomic
        // steps. Under RULE-MT continuous `evict_for_capacity` pressure,
        // a concurrent eviction can reclaim `page_id` in the window
        // between them — `fault_in_for_tenant` truthfully reported
        // "resident" a moment ago, but by the time `cache_latch` runs
        // the frame is gone again, surfacing a spurious `MissingPage`
        // for a page that is very much still tracked (just churned back
        // to `evicted`). A bounded retry loop closes this: each
        // iteration re-establishes residency and immediately re-checks;
        // eviction MUST make forward progress to keep re-stealing this
        // exact key (MECH-E8's own bounded-retry discipline already
        // bounds how long any one evictor spins), so this retry
        // terminates in practice within a handful of iterations. The
        // cap here (guarding against actual data loss such as
        // `remove_page`/tenant teardown racing this call) surfaces the
        // page's TRUE state — `MissingPage` — rather than looping
        // forever on a page that will never come back.
        const MAX_ATTEMPTS: usize = 64;
        let key = self.page_key(tenant, page_id);
        for _ in 0..MAX_ATTEMPTS {
            self.fault_in_for_tenant(tenant, page_id)?;
            if let Ok(latch) = self.cache_latch(key) {
                self.touch(key);
                return Ok(latch);
            }
            std::thread::yield_now();
        }
        // Final attempt without swallowing the error: if we are still
        // racing after MAX_ATTEMPTS, surface the true state rather than
        // spin unboundedly (MECH-E8's back-pressure-never-deadlock
        // lesson applied to this seam too).
        self.fault_in_for_tenant(tenant, page_id)?;
        let latch = self.cache_latch(key)?;
        self.touch(key);
        Ok(latch)
    }

    /// #1521 M6.1 P0-1 — override the trait default with the REAL
    /// pin-coupled acquisition (`Self::latch_pinned_for_tenant`, the
    /// inherent method), never the inert fallback: this backend DOES
    /// evict resident frames (`try_evict_page_pinned_for_tenant`), so
    /// every dirty-marking writer routed through this trait object
    /// must register a real pin the removal claim excludes on.
    fn latch_pinned_for_tenant(
        &self,
        tenant: TenantId,
        page_id: PageId,
    ) -> Result<AnyPinnedPageLatch, RecordStoreError> {
        Self::latch_pinned_for_tenant(self, tenant, page_id).map(AnyPinnedPageLatch::from)
    }

    fn install_or_replace(
        &self,
        page_id: PageId,
        page: Box<PageBuf>,
    ) -> Result<(), RecordStoreError> {
        self.install_or_replace_for_tenant(TenantId::DEFAULT, page_id, page)
    }

    fn install_or_replace_for_tenant(
        &self,
        tenant: TenantId,
        page_id: PageId,
        page: Box<PageBuf>,
    ) -> Result<(), RecordStoreError> {
        let key = self.page_key(tenant, page_id);
        self.evicted.remove(&key);
        self.cache_install_or_replace(key, page);
        self.touch(key);
        Ok(())
    }

    fn remove_page(&self, page_id: PageId) -> Option<RecordPageLatch> {
        self.remove_page_for_tenant(TenantId::DEFAULT, page_id)
    }

    fn remove_page_for_tenant(&self, tenant: TenantId, page_id: PageId) -> Option<RecordPageLatch> {
        let key = self.page_key(tenant, page_id);
        self.evicted.remove(&key);
        self.untrack(key);
        let removed = self.cache.remove(&key).map(|(_, latch)| latch);
        if removed.is_some() {
            self.record_cache_remove();
        }
        removed
    }

    fn restore_page_bytes(
        &self,
        page_id: PageId,
        pre_bytes: &PageBuf,
    ) -> Result<(), RecordStoreError> {
        self.restore_page_bytes_for_tenant(TenantId::DEFAULT, page_id, pre_bytes)
    }

    fn restore_page_bytes_for_tenant(
        &self,
        tenant: TenantId,
        page_id: PageId,
        pre_bytes: &PageBuf,
    ) -> Result<(), RecordStoreError> {
        self.fault_in_for_tenant(tenant, page_id)?;
        let key = self.page_key(tenant, page_id);
        self.cache_latch(key)?.write().copy_from_slice(pre_bytes);
        self.touch(key);
        Ok(())
    }

    fn contains(&self, page_id: PageId) -> bool {
        self.contains_for_tenant(TenantId::DEFAULT, page_id)
    }

    fn contains_for_tenant(&self, tenant: TenantId, page_id: PageId) -> bool {
        let key = self.page_key(tenant, page_id);
        self.cache.contains_key(&key) || self.evicted.contains_key(&key)
    }

    fn len(&self) -> usize {
        self.cache.len() + self.evicted.len()
    }

    fn iter_pages(&self) -> Vec<(PageId, RecordPageLatch)> {
        // Fault-in every evicted page first so the snapshot covers the
        // full set. Bootstrap is one-shot at startup so the cost of N
        // disk reads is acceptable.
        self.iter_pages_qualified()
            .into_iter()
            .map(|(_, page_id, latch)| (page_id, latch))
            .collect()
    }

    fn iter_pages_qualified(&self) -> Vec<(TenantId, PageId, RecordPageLatch)> {
        let evicted_keys: Vec<RecordPageKey> = self.evicted.iter().map(|e| *e.key()).collect();
        for key in evicted_keys {
            // Best-effort: log and skip on fault-in failure (matches
            // the recovery-side "warn-and-continue" posture for
            // unrecoverable pages — the K-1 oracle catches structural
            // damage at the next invariant check).
            if let Err(e) = self.fault_in_for_tenant(key.tenant_id, key.page_id) {
                tracing::warn!(
                    "BufferedRecordPageStore::iter_pages fault_in({:?}) failed: {}",
                    key.page_id,
                    e,
                );
            }
        }
        self.cache
            .iter()
            .map(|entry| {
                (
                    entry.key().tenant_id,
                    entry.key().page_id,
                    Arc::clone(entry.value()),
                )
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────
// RecordPageStoreHandle impl — WAL replay co-coordination (ADR-140 D-4)
// ─────────────────────────────────────────────────────────────────────

impl RecordPageStoreHandle for BufferedRecordPageStore {
    fn install_or_replace(&self, page_id: PageId, page: Box<[u8; PAGE_SIZE]>) -> CoreResult<()> {
        let header_bytes: &[u8; PageHeader::SIZE] = page[..PageHeader::SIZE]
            .try_into()
            .expect("page header slice has fixed length");
        // Legacy v8 replay historically treated record images as opaque bytes.
        // Preserve that compatibility for malformed test fixtures; valid page
        // images route by their stamped tenant.
        let tenant = PageHeader::from_bytes(header_bytes)
            .map(|header| TenantId::new(header.tenant_id))
            .unwrap_or(TenantId::DEFAULT);
        // Replay routes through the trait impl above. Errors translate
        // via the canonical Box<PageBuf> branch.
        <Self as RecordPageBackend>::install_or_replace_for_tenant(self, tenant, page_id, page)
            .map_err(|e| {
                ArcGraphError::Io(std::io::Error::other(format!(
                    "BufferedRecordPageStore::install_or_replace failed: {}",
                    e,
                )))
            })
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::InMemoryPageIo;
    use crate::mutation_log::TxnMutationLog;
    use arcgraph_core::PageType;

    fn new_store(cap: usize) -> (Arc<InMemoryPageIo>, Arc<BufferedRecordPageStore>) {
        let io_concrete = Arc::new(InMemoryPageIo::new());
        let io_dyn: Arc<dyn PageIo> = io_concrete.clone();
        let pools = Arc::new(PerTenantBufferPool::with_config(
            io_dyn,
            PerTenantBufferPoolConfig {
                frames_per_tenant: 16,
                write_fraction: 0.0,
            },
        ));
        let store = Arc::new(BufferedRecordPageStore::with_cache_cap(pools, cap));
        (io_concrete, store)
    }

    #[test]
    fn install_fresh_round_trip() {
        let (_io, store) = new_store(4);
        let pid = PageId::new(42);
        store
            .install_fresh(pid, PageType::Node, TenantId::DEFAULT)
            .unwrap();
        assert!(store.contains(pid));
        assert_eq!(store.cache_size(), 1);
        assert_eq!(store.evicted_count(), 0);
        let latch = store.latch(pid).unwrap();
        let g = latch.read();
        assert_eq!(g.as_ref().as_ref().len(), PAGE_SIZE);
    }

    #[test]
    fn evict_lru_spills_oldest_first() {
        let (_io, store) = new_store(2);
        for i in 0..5 {
            store
                .install_fresh(PageId::new(i), PageType::Node, TenantId::DEFAULT)
                .unwrap();
        }
        assert_eq!(store.cache_size(), 5);
        let evicted = store.evict_lru(2).unwrap();
        assert_eq!(evicted, 3);
        assert_eq!(store.cache_size(), 2);
        assert_eq!(store.evicted_count(), 3);
        // The two newest (3 and 4) should still be in cache; the three
        // oldest (0, 1, 2) should be on disk.
        assert!(store.is_cached(PageId::new(3)));
        assert!(store.is_cached(PageId::new(4)));
        assert!(store.is_evicted(PageId::new(0)));
        assert!(store.is_evicted(PageId::new(1)));
        assert!(store.is_evicted(PageId::new(2)));
        // All five still report as contained.
        for i in 0..5 {
            assert!(store.contains(PageId::new(i)), "pid {} missing", i);
        }
    }

    #[test]
    fn fault_in_brings_page_back_to_cache() {
        let (_io, store) = new_store(2);
        for i in 0..3 {
            store
                .install_fresh(PageId::new(i), PageType::Node, TenantId::DEFAULT)
                .unwrap();
        }
        // Force eviction of pid 0 + 1 by writing more pages.
        store.evict_lru(1).unwrap();
        // Now fault in pid 0 — it should leave the evicted set and
        // enter the cache.
        assert!(store.is_evicted(PageId::new(0)));
        store.fault_in(PageId::new(0)).unwrap();
        assert!(store.is_cached(PageId::new(0)));
        assert!(!store.is_evicted(PageId::new(0)));
    }

    #[test]
    fn evict_lru_invalidates_stale_pool_frame_after_direct_write() {
        let (_io, store) = new_store(1);
        let pid = PageId::new(1129);

        store
            .install_fresh(pid, PageType::Node, TenantId::DEFAULT)
            .unwrap();
        {
            let mut log = TxnMutationLog::new();
            let latch = store.capture_and_write(&mut log, pid).unwrap();
            latch.write().as_mut()[PAGE_SIZE - 1] = 0xA1;
        }

        assert_eq!(store.evict_lru(0).unwrap(), 1);
        assert!(store.is_evicted(pid));

        store.fault_in(pid).unwrap();
        {
            let latch = store.latch(pid).unwrap();
            assert_eq!(latch.read().as_ref()[PAGE_SIZE - 1], 0xA1);
        }

        {
            let mut log = TxnMutationLog::new();
            let latch = store.capture_and_write(&mut log, pid).unwrap();
            latch.write().as_mut()[PAGE_SIZE - 1] = 0xB2;
        }

        assert_eq!(store.evict_lru(0).unwrap(), 1);
        assert!(store.is_evicted(pid));

        store.fault_in(pid).unwrap();
        let latch = store.latch(pid).unwrap();
        assert_eq!(
            latch.read().as_ref()[PAGE_SIZE - 1],
            0xB2,
            "fault_in must read the second canonical mutation, not a stale pool frame"
        );
    }

    #[test]
    fn install_or_replace_through_replay_handle() {
        let (_io, store) = new_store(4);
        let pid = PageId::new(100);
        let bytes: Box<PageBuf> = Box::new([0xAB; PAGE_SIZE]);
        // Drive through the RecordPageStoreHandle trait — replay's path.
        <BufferedRecordPageStore as RecordPageStoreHandle>::install_or_replace(&store, pid, bytes)
            .unwrap();
        assert!(store.contains(pid));
        let latch = store.latch(pid).unwrap();
        let g = latch.read();
        assert_eq!(g.as_ref().as_ref()[0], 0xAB);
        assert_eq!(g.as_ref().as_ref()[PAGE_SIZE - 1], 0xAB);
    }

    // ── M6.1 MECH-E1..E8 — `evict_for_capacity` wiring tests ──────────

    fn new_store_with_m6(
        cap: usize,
    ) -> (
        Arc<InMemoryPageIo>,
        Arc<BufferedRecordPageStore>,
        Arc<crate::redo::DirtyPageTable>,
        Arc<crate::checkpoint::WriteBehindCheckpointer>,
    ) {
        let (io, store) = new_store(cap);
        let dpt = Arc::new(crate::redo::DirtyPageTable::new());
        // The store is its own flush target (STORE_RECORD) for this
        // unit test — mirrors how `crud.rs` wires the M3 checkpointer
        // against `BufferedRecordPageStore` in production.
        let props_target: Arc<dyn crate::checkpoint::PageFlushTarget> = store.clone();
        let records_target: Arc<dyn crate::checkpoint::PageFlushTarget> = store.clone();
        let checkpointer = Arc::new(crate::checkpoint::WriteBehindCheckpointer::new(
            dpt.clone(),
            props_target,
            records_target,
        ));
        store.attach_m6_dirty_page_table(dpt.clone());
        store.attach_m6_checkpointer(checkpointer.clone());
        (io, store, dpt, checkpointer)
    }

    #[test]
    fn evict_for_capacity_reclaims_clean_pages_with_no_io() {
        let (io, store, _dpt, _cp) = new_store_with_m6(2);
        for i in 0..5 {
            store
                .install_fresh(PageId::new(i), PageType::Node, TenantId::DEFAULT)
                .unwrap();
        }
        // None of these pages were ever marked dirty in the DPT, so
        // every one is MECH-E1 "clean" — reclaim needs zero writes.
        let writes_before = io.writes();
        let evicted = store.evict_for_capacity(2).unwrap();
        assert_eq!(evicted, 3);
        assert_eq!(store.cache_size(), 2);
        assert_eq!(
            io.writes(),
            writes_before,
            "clean-page reclaim must perform ZERO page writes (MECH-E2)"
        );
    }

    #[test]
    fn evict_for_capacity_flushes_dirty_pages_through_checkpointer_before_reclaim() {
        let (io, store, dpt, _cp) = new_store_with_m6(1);
        let pid = PageId::new(7);
        store
            .install_fresh(pid, PageType::Node, TenantId::DEFAULT)
            .unwrap();
        // Mutate + mark dirty in the DPT — mirrors `crud.rs::mark_m3_dirty`
        // firing only from the post-WAL-fsync Phase-3 apply path.
        {
            let pinned = store.latch_pinned(pid).unwrap();
            pinned.latch().write().as_mut()[PAGE_SIZE - 1] = 0xEE;
        }
        dpt.mark_dirty(
            crate::redo::DirtyPageKey {
                tenant_id: TenantId::DEFAULT,
                store_id: crate::wal::STORE_RECORD,
                page_no: pid.raw(),
            },
            arcgraph_core::Lsn::new(1),
        );
        assert!(
            dpt.snapshot_key(crate::redo::DirtyPageKey {
                tenant_id: TenantId::DEFAULT,
                store_id: crate::wal::STORE_RECORD,
                page_no: pid.raw(),
            })
            .is_some()
        );

        let writes_before = io.writes();
        // Install a second page to force capacity pressure against cap=1.
        store
            .install_fresh(PageId::new(8), PageType::Node, TenantId::DEFAULT)
            .unwrap();
        let evicted = store.evict_for_capacity(1).unwrap();
        assert_eq!(
            evicted, 1,
            "the dirty page must be reclaimed via the checkpointer handshake"
        );
        assert!(
            io.writes() > writes_before,
            "MECH-E2/E3: the checkpointer (not the evictor) must have durably \
             written the dirty page home before reclaim"
        );
        assert!(
            dpt.snapshot_key(crate::redo::DirtyPageKey {
                tenant_id: TenantId::DEFAULT,
                store_id: crate::wal::STORE_RECORD,
                page_no: pid.raw(),
            })
            .is_none(),
            "MECH-E4: the DPT entry must clear on durable home write"
        );
        // Round-trip: the reclaimed page's mutation survived (fault back in).
        store.fault_in(pid).unwrap();
        let latch = store.latch(pid).unwrap();
        assert_eq!(latch.read().as_ref()[PAGE_SIZE - 1], 0xEE);
    }

    #[test]
    fn evict_for_capacity_skips_pinned_pages_mech_e7() {
        let (_io, store, _dpt, _cp) = new_store_with_m6(1);
        let pid = PageId::new(9);
        store
            .install_fresh(pid, PageType::Node, TenantId::DEFAULT)
            .unwrap();
        let pinned = store.latch_pinned(pid).unwrap();
        store
            .install_fresh(PageId::new(10), PageType::Node, TenantId::DEFAULT)
            .unwrap();
        // Only 1 unpinned candidate (10) is reclaimable; 9 stays pinned.
        let evicted = store.evict_for_capacity(1).unwrap();
        assert_eq!(evicted, 1);
        assert!(store.is_cached(pid), "pinned page must never be evicted");
        drop(pinned);
    }

    #[test]
    fn evict_for_capacity_without_m6_wiring_is_a_clean_only_noop() {
        // Legacy posture: no DPT/checkpointer attached at all.
        let (_io, store) = new_store(1);
        store
            .install_fresh(PageId::new(1), PageType::Node, TenantId::DEFAULT)
            .unwrap();
        store
            .install_fresh(PageId::new(2), PageType::Node, TenantId::DEFAULT)
            .unwrap();
        // Both pages are "clean" in the sense that no DPT exists to mark
        // them dirty; with no DPT wired, `is_clean` returns `None` and
        // the fully-unwired branch stops cleanly (never spins, never
        // panics) rather than attempting reclaim.
        let evicted = store.evict_for_capacity(1).unwrap();
        assert_eq!(
            evicted, 0,
            "with no M6 wiring at all, evict_for_capacity is a safe no-op"
        );
    }

    /// M6.1 INV-M6.11 — `resident_count()` (the O(1) atomic
    /// `evict_for_capacity` consults) must stay exactly in sync with
    /// `cache_size()` (the authoritative `DashMap::len()`) across every
    /// insert/remove/evict/fault-in path. A drift here would silently
    /// break the hot-path fast-return (either evicting too eagerly or
    /// never evicting at all).
    #[test]
    fn resident_count_matches_cache_size_across_mixed_workload() {
        let (_io, store) = new_store(1024);
        let dpt = Arc::new(crate::redo::DirtyPageTable::new());
        let checkpointer = Arc::new(crate::checkpoint::WriteBehindCheckpointer::new(
            dpt.clone(),
            store.clone(),
            store.clone(),
        ));
        store.attach_m6_dirty_page_table(dpt.clone());
        store.attach_m6_checkpointer(checkpointer);

        for i in 0..50u64 {
            let pid = PageId::new(i);
            store
                .install_fresh(pid, PageType::Node, TenantId::DEFAULT)
                .unwrap();
            assert_eq!(store.resident_count(), store.cache_size());
        }

        // Drive some clean evictions (no DPT entries yet).
        let evicted = store.evict_for_capacity(20).unwrap();
        assert!(evicted > 0);
        assert_eq!(store.resident_count(), store.cache_size());

        // Fault some back in.
        for i in 0..10u64 {
            let _ = store.fault_in(PageId::new(i));
            assert_eq!(store.resident_count(), store.cache_size());
        }

        // Remove a page outright.
        RecordPageBackend::remove_page(store.as_ref(), PageId::new(30));
        assert_eq!(store.resident_count(), store.cache_size());

        // install_or_replace on both a fresh and an existing key.
        RecordPageBackend::install_or_replace(
            store.as_ref(),
            PageId::new(999),
            Box::new([0u8; PAGE_SIZE]),
        )
        .unwrap();
        assert_eq!(store.resident_count(), store.cache_size());
        RecordPageBackend::install_or_replace(
            store.as_ref(),
            PageId::new(999),
            Box::new([1u8; PAGE_SIZE]),
        )
        .unwrap();
        assert_eq!(
            store.resident_count(),
            store.cache_size(),
            "overwriting an existing key must NOT double-count the insert"
        );
    }

    #[test]
    fn outstanding_latch_blocks_eviction_of_that_page() {
        let (_io, store) = new_store(1);
        for i in 0..3 {
            store
                .install_fresh(PageId::new(i), PageType::Node, TenantId::DEFAULT)
                .unwrap();
        }
        // Hold an outstanding latch on pid 0 (the LRU candidate).
        let _held_latch = store.latch(PageId::new(0)).unwrap();
        let evicted = store.evict_lru(0).unwrap();
        // pid 0 cannot evict (held); pid 1 + 2 do.
        assert_eq!(evicted, 2);
        assert!(store.is_cached(PageId::new(0)));
        assert!(store.is_evicted(PageId::new(1)));
        assert!(store.is_evicted(PageId::new(2)));
    }

    #[test]
    fn iter_pages_faults_in_evicted_pages() {
        let (_io, store) = new_store(2);
        for i in 0..4 {
            store
                .install_fresh(PageId::new(i), PageType::Node, TenantId::DEFAULT)
                .unwrap();
        }
        store.evict_lru(1).unwrap();
        let snapshot = store.iter_pages();
        // After iter_pages, all evicted pages are faulted-in. The
        // snapshot covers the full set (4 pages).
        assert_eq!(snapshot.len(), 4);
    }

    #[test]
    fn remove_page_clears_all_tracking() {
        let (_io, store) = new_store(4);
        let pid = PageId::new(7);
        store
            .install_fresh(pid, PageType::Node, TenantId::DEFAULT)
            .unwrap();
        assert!(store.contains(pid));
        let removed = store.remove_page(pid);
        assert!(removed.is_some());
        assert!(!store.contains(pid));
        assert_eq!(store.cache_size(), 0);
        assert_eq!(store.evicted_count(), 0);
    }

    #[test]
    fn per_tenant_pool_isolation() {
        let io: Arc<dyn PageIo> = Arc::new(InMemoryPageIo::new());
        let pools = Arc::new(PerTenantBufferPool::with_config(
            io,
            PerTenantBufferPoolConfig {
                frames_per_tenant: 4,
                write_fraction: 0.0,
            },
        ));
        let t1 = TenantId::new(1);
        let t2 = TenantId::new(2);
        let p1 = pools.pool(t1).unwrap();
        let p2 = pools.pool(t2).unwrap();
        // Distinct pool instances.
        assert!(!Arc::ptr_eq(&p1, &p2));
        // Each pool has its own frame count.
        assert_eq!(p1.capacity(), 4);
        assert_eq!(p2.capacity(), 4);
        // Get-or-create: a second lookup returns the SAME instance.
        let p1b = pools.pool(t1).unwrap();
        assert!(Arc::ptr_eq(&p1, &p1b));
        assert_eq!(pools.tenant_count(), 2);
    }
}
