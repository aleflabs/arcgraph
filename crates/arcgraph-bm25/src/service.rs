//! Workspace-level BM25 service (ADR-039 §D-4 + amendment-02 §D-12).
//!
//! [`Bm25Service`] holds the per-tenant `Bm25IndexHandle` cache and
//! materialises new handles on first touch. Mirrors `BlobStore` /
//! `VectorPageStore` in being workspace-scoped (one instance per
//! deployment) with per-tenant projection inside.
//!
//! # Directory layout (ADR-039 §D-4)
//!
//! ```text
//! <data_dir>/
//!   bm25/
//!     <tenant_id>/
//!       <index_id>/      ← Tantivy directory (meta.json + segments)
//! ```
//!
//! At v1.0 every tenant has exactly one `index_id` directory:
//! [`crate::IndexId::DEFAULT_BM25`] (= `0`). Per-property indexes are
//! M7 / v1.1 scope.
//!
//! # Cache policy (append-only handles, lazy writers)
//!
//! Per ADR-037 §D-6, the *handle* cache is append-only at v1.0 — once
//! a `(tenant, index)` directory has been opened, the resulting
//! `Bm25IndexHandle` lives until the service drops. Tenant lifecycle
//! (CREATE / DROP / RENAME) is M7+ scope.
//!
//! Per ADR-039 amendment-01 §D-11 (implemented in amendment-02), the
//! per-tenant `IndexWriter` inside that handle is no longer
//! eagerly long-lived: it is allocated on first write under the cap
//! of a shared [`crate::pool::WriterPool`] (capacity =
//! [`crate::WRITER_POOL_SIZE`]) and **lazily evicted** when the
//! tenant is idle (per
//! [`crate::IDLE_EVICTION_COMMIT_THRESHOLD`] /
//! [`crate::IDLE_EVICTION_WALL_CLOCK_THRESHOLD_SECS`]). This caps
//! active-set RAM at `WRITER_POOL_SIZE × DEFAULT_WRITER_HEAP_BYTES`
//! regardless of the tenant population — see ADR-039 amendment-02
//! §D-12 for the empirical envelope.

use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock, Weak};

use arcgraph_core::{PartitionId, TenantId};
use dashmap::DashMap;
use parking_lot::Mutex;
use tantivy::directory::Directory;
use tantivy::{Index, ReloadPolicy};

use crate::error::Bm25Error;
use crate::eviction::IdleTracker;
use crate::handle::{Bm25IndexHandle, IndexId, Sweeper, TantivyIndexInner};
use crate::pool::{WRITER_POOL_SIZE, WriterPool};
use crate::segment::Bm25Schema;

/// Type alias for the pluggable Tantivy directory factory used by
/// [`Bm25Service::handle`] when materialising a per-tenant index.
///
/// Production callers obtain a `Bm25Service` via [`Bm25Service::new`]
/// or [`Bm25Service::with_pool_size`] which install the default
/// factory: `|p| Box::new(MmapDirectory::open(p)?)`. The factory hook
/// exists so test-only constructors (currently
/// [`Bm25Service::with_directory_factory`], `#[doc(hidden)]`) can
/// inject a wrapping `Directory` (e.g. `FaultInjectDirectory`) for
/// fault-injection regression tests — see issue #224 (W9a M3.b N1)
/// for the rollback-path regression pin that drove this seam.
#[doc(hidden)]
pub type Bm25DirectoryFactory =
    dyn Fn(&Path) -> Result<Box<dyn Directory>, Bm25Error> + Send + Sync;

/// Default factory: open an [`tantivy::directory::MmapDirectory`] at
/// `path` and box it as `Box<dyn Directory>`. Installed by
/// [`Bm25Service::new`] / [`Bm25Service::with_pool_size`] so the
/// production code path is unchanged from before the factory hook
/// existed.
fn default_mmap_directory_factory() -> Arc<Bm25DirectoryFactory> {
    Arc::new(|path: &Path| {
        let dir = tantivy::directory::MmapDirectory::open(path)?;
        Ok(Box::new(dir) as Box<dyn Directory>)
    })
}

/// Default heap budget for a per-tenant `IndexWriter`.
///
/// **Lowered from 64 MiB to 16 MiB** per ADR-039 amendment-01
/// §D-11(a): Tantivy's documented minimum is 15 MiB; 16 MiB is
/// minimum × 1.05 (safety margin against per-Tantivy-version floor
/// changes). Sub-default heaps degrade indexing performance non-
/// linearly per Tantivy 0.21 release notes — the trade-off is
/// accepted because RAM ceiling dominates throughput at multi-tenant
/// fan-out (1000 tenants × 64 MiB = 64 GiB; the same fleet at
/// 16 MiB = 16 GiB).
///
/// The lazy-eviction (D-11(b)) + shared-pool (D-11(c)) machinery is
/// what makes the reduced default safe at scale; the constant alone
/// without that machinery would degrade indexing without bounding
/// active-set RAM. All three D-11 sub-decisions ship together per
/// amendment-01 phasing rationale.
const DEFAULT_WRITER_HEAP_BYTES: usize = 16 * 1024 * 1024;

/// Cached per-tenant entry. Owns the `Bm25IndexHandle`'s
/// `Arc<TantivyIndexInner>` and exposes a stable `Arc` over the
/// handle so `Bm25Service::handle` can clone it cheaply on cache
/// hits.
///
/// The *idle tracker* lives inside `TantivyIndexInner` (not here)
/// so write paths can update it without re-traversing the DashMap.
struct CachedHandle {
    handle: Arc<Bm25IndexHandle>,
}

/// Workspace-level BM25 service per ADR-039 §D-4.
pub struct Bm25Service {
    data_dir: PathBuf,
    handles: DashMap<(TenantId, IndexId), CachedHandle>,
    schema: Bm25Schema,
    /// Per-tenant directory-creation lock. DashMap's per-bucket lock
    /// guards the cache entry, but the directory-create race
    /// (two threads both try to `open_or_create_in_dir` on the same
    /// path before either populates the cache) is widened by
    /// Tantivy's I/O — serialise on this Mutex when missing the cache.
    ///
    /// Mean cache-miss path is per-tenant first-touch, not hot, so
    /// the global Mutex is acceptable. v1.1 may move to a per-key
    /// guard if first-touch contention surfaces.
    create_guard: Mutex<()>,
    /// Shared admission pool sized to [`WRITER_POOL_SIZE`] (or the
    /// override passed to [`Self::with_pool_size`]) per ADR-039
    /// amendment-01 §D-11(c).
    pool: Arc<WriterPool>,
    /// Heap budget passed to every per-tenant
    /// `IndexWriter` allocation. Captured here so a future v1.1
    /// per-deployment override flows through one well-defined
    /// surface.
    heap_bytes: usize,
    /// Self-reference for the eviction-callback wiring. Set once at
    /// `new()` time after the outer `Arc` is constructed; consulted
    /// by [`Self::idle_sweeper`] / [`Self::orphan_evictor`] to capture
    /// a `Weak<Bm25Service>` without forming a strong cycle.
    me: OnceLock<Weak<Self>>,
    /// Per-tenant Tantivy directory factory. Default (production):
    /// wraps [`tantivy::directory::MmapDirectory::open`]. Tests inject
    /// a wrapping `Directory` via
    /// [`Self::with_directory_factory`] (`#[doc(hidden)]`) — the
    /// production constructors install the MmapDirectory default and
    /// behave identically to before this hook was introduced. See
    /// issue #224 (W9a M3.b N1) for the rollback regression pin that
    /// drove this seam.
    directory_factory: Arc<Bm25DirectoryFactory>,
}

impl std::fmt::Debug for Bm25Service {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bm25Service")
            .field("data_dir", &self.data_dir)
            .field("handles_len", &self.handles.len())
            .field("schema", &self.schema)
            .field("pool", &self.pool)
            .field("heap_bytes", &self.heap_bytes)
            .finish()
    }
}

impl Bm25Service {
    /// Construct a service rooted at `data_dir` with the default
    /// pool size [`WRITER_POOL_SIZE`]. The `data_dir` is not created
    /// at construction — per-tenant subdirectories are created on
    /// first [`Self::handle`] call.
    ///
    /// Returns `Arc<Self>` because the service publishes a
    /// `Weak<Self>` back-reference into per-handle sweeper closures
    /// (the eviction-on-block path; ADR-039 amendment-02 §D-13).
    /// External callers that previously wrote
    /// `Arc::new(Bm25Service::new(path))` should now write
    /// `Bm25Service::new(path)` directly.
    #[must_use]
    pub fn new(data_dir: PathBuf) -> Arc<Self> {
        Self::with_pool_size(data_dir, WRITER_POOL_SIZE)
    }

    /// Construct with an explicit pool size override. Primarily a
    /// test / bench entry-point so pool-exhaustion can be exercised
    /// at small N without holding [`WRITER_POOL_SIZE`] tenants
    /// active. Production callers should use [`Self::new`].
    #[must_use]
    pub fn with_pool_size(data_dir: PathBuf, pool_size: usize) -> Arc<Self> {
        Self::build(data_dir, pool_size, default_mmap_directory_factory())
    }

    /// **Test-only.** Construct a `Bm25Service` whose per-tenant
    /// Tantivy directory is produced by `factory` instead of the
    /// default [`tantivy::directory::MmapDirectory`]. Used by
    /// `tests/eviction_recovery.rs::permit_returned_to_pool_on_tantivy_rollback_failure`
    /// (issue #224, W9a M3.b N1) to inject a `FaultInjectDirectory`
    /// wrapper that surfaces `Err` deterministically on Tantivy's
    /// rollback I/O, making the F1 rollback regression pin
    /// load-bearing.
    ///
    /// The factory closure is called once per cache-miss `(tenant,
    /// index)` pair from inside [`Self::handle`]. Production callers
    /// MUST NOT use this entry point — the public surface is
    /// [`Self::new`] / [`Self::with_pool_size`]. `#[doc(hidden)]` so
    /// the seam does not appear in rustdoc.
    #[doc(hidden)]
    #[must_use]
    pub fn with_directory_factory(
        data_dir: PathBuf,
        pool_size: usize,
        factory: Arc<Bm25DirectoryFactory>,
    ) -> Arc<Self> {
        Self::build(data_dir, pool_size, factory)
    }

    /// Internal constructor shared by [`Self::with_pool_size`] and
    /// [`Self::with_directory_factory`]. Production callers reach the
    /// MmapDirectory default; tests reach the injected factory.
    fn build(
        data_dir: PathBuf,
        pool_size: usize,
        directory_factory: Arc<Bm25DirectoryFactory>,
    ) -> Arc<Self> {
        let arc = Arc::new(Self {
            data_dir,
            handles: DashMap::new(),
            schema: Bm25Schema::build(),
            create_guard: Mutex::new(()),
            pool: WriterPool::new(pool_size),
            heap_bytes: DEFAULT_WRITER_HEAP_BYTES,
            me: OnceLock::new(),
            directory_factory,
        });
        // Best-effort `set` — the OnceLock is freshly constructed,
        // so the `set` is guaranteed to succeed; the `let _` discards
        // the impossible-Err to keep clippy quiet.
        let _ = arc.me.set(Arc::downgrade(&arc));
        arc
    }

    /// Open or materialise the [`Bm25IndexHandle`] for
    /// `(tenant, index)`.
    ///
    /// Lazily creates `<data_dir>/bm25/<tenant>/<index>/` on first
    /// touch via `tantivy::Index::open_or_create_in_dir`, then
    /// caches the resulting handle. Subsequent calls return the
    /// cached `Arc`. Per ADR-039 amendment-01 §D-11(b)+(c), the
    /// `IndexWriter` itself is **not** allocated here — the handle's
    /// writer slot starts empty and is populated lazily on first
    /// write under the shared pool's capacity bound.
    ///
    /// # Errors
    ///
    /// - [`Bm25Error::Io`] when directory creation fails.
    /// - [`Bm25Error::Tantivy`] when index open / create fails.
    pub fn handle(
        &self,
        tenant: TenantId,
        index: IndexId,
    ) -> Result<Arc<Bm25IndexHandle>, Bm25Error> {
        // Fast path: cache hit.
        if let Some(existing) = self.handles.get(&(tenant, index)) {
            return Ok(Arc::clone(&existing.value().handle));
        }

        // Slow path: serialise creation under the global guard so
        // two concurrent first-touches do not both call
        // `open_or_create_in_dir` on the same directory.
        let _guard = self.create_guard.lock();
        // Re-check after acquiring the guard.
        if let Some(existing) = self.handles.get(&(tenant, index)) {
            return Ok(Arc::clone(&existing.value().handle));
        }

        let dir = self.tenant_index_dir(tenant, index);
        std::fs::create_dir_all(&dir)?;

        // Production: factory returns Box::new(MmapDirectory::open(&dir)?).
        // Tests may inject a wrapping Directory via
        // `Self::with_directory_factory` (issue #224 W9a M3.b N1
        // FaultInjectDirectory). The factory return is `Box<dyn
        // Directory>`; `Index::open_or_create` accepts anything
        // `Into<Box<dyn Directory>>` (tantivy 0.26.1 index.rs:343).
        let directory: Box<dyn Directory> = (self.directory_factory)(&dir)?;
        let index_obj = Index::open_or_create(directory, self.schema.schema.clone())?;
        let reader = index_obj
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?;

        // Surface lazy per-tenant writer-slot allocation to ops so
        // runaway tenant counts are visible without a custom metric.
        // Per ADR-039 amendment-02 §D-12, RAM scales as
        // `min(N_active, pool_capacity) × DEFAULT_WRITER_HEAP_BYTES`
        // — bounded above by the pool size, NOT by N_tenants.
        // The original "eager allocation" warning text from PR #183
        // is kept as a lineage breadcrumb but updated to reflect the
        // new lazy-allocation + pool semantics.
        tracing::warn!(
            tenant_id = tenant.raw(),
            index_id = index.raw(),
            writer_heap_mib = self.heap_bytes / (1024 * 1024),
            pool_capacity = self.pool.capacity(),
            "BM25 handle materialised for new tenant — writer slot is \
             lazy; cumulative active-set RAM bounded by \
             pool_capacity × writer_heap. See ADR-039 amendment-02 \
             §D-11+D-12 for the writer-heap policy."
        );

        let inner = Arc::new(TantivyIndexInner {
            index: index_obj,
            writer: parking_lot::Mutex::new(None),
            reader,
            schema: self.schema.clone(),
            pool: Arc::clone(&self.pool),
            heap_bytes: self.heap_bytes,
            idle: IdleTracker::new(),
        });

        let on_full_idle_sweep = self.idle_sweeper();
        let on_block_timeout_evict = self.orphan_evictor();
        let handle = Arc::new(Bm25IndexHandle::new(
            tenant,
            PartitionId::ZERO,
            index,
            inner,
            on_full_idle_sweep,
            on_block_timeout_evict,
        ));

        self.handles.insert(
            (tenant, index),
            CachedHandle {
                handle: Arc::clone(&handle),
            },
        );
        Ok(handle)
    }

    /// Schema shared across every tenant. Useful for tests and the
    /// M4 query planner that needs to know the field layout up
    /// front.
    #[must_use]
    pub fn schema(&self) -> &Bm25Schema {
        &self.schema
    }

    /// Whether a handle for `(tenant, index)` has already been
    /// materialised. Test helper; not used on the hot path.
    #[must_use]
    pub fn has_handle(&self, tenant: TenantId, index: IndexId) -> bool {
        self.handles.contains_key(&(tenant, index))
    }

    /// Per-tenant Tantivy directory path. Public so tests can
    /// pre-create the path layout in fixtures.
    #[must_use]
    pub fn tenant_index_dir(&self, tenant: TenantId, index: IndexId) -> PathBuf {
        let mut p = self.data_dir.clone();
        p.push("bm25");
        p.push(tenant.raw().to_string());
        p.push(index.raw().to_string());
        p
    }

    /// Iterator-friendly snapshot of all `(tenant, index)` keys
    /// currently in the cache. Used by the
    /// `Bm25IndexStoreHandle::commit_pending` / `rollback_pending`
    /// dispatch in `store.rs` and by tests.
    #[must_use]
    pub(crate) fn handle_for(
        &self,
        tenant: TenantId,
        index: IndexId,
    ) -> Option<Arc<Bm25IndexHandle>> {
        self.handles
            .get(&(tenant, index))
            .map(|r| Arc::clone(&r.value().handle))
    }

    /// Sweep all cached tenants and evict the writers of any that
    /// report idle (per
    /// [`IdleTracker::is_idle`]). Returns the number of writers
    /// evicted.
    ///
    /// Called opportunistically at the tail of every
    /// `commit_pending` so commits amortise idle-cleanup at the
    /// natural rhythm of the system. May also be invoked by ops
    /// tooling for forced cleanup.
    ///
    /// Eviction uses
    /// `TantivyIndexInner::try_evict_writer` (non-blocking
    /// `try_lock`) so the sweep is reentrant-safe when invoked
    /// from inside `ensure_writer` and so contended writers are
    /// skipped without blocking the sweep.
    pub fn evict_idle(&self) -> usize {
        let mut count = 0;
        // DashMap iter holds a per-shard read guard while iterating;
        // since `try_evict_writer` only touches the inner writer
        // mutex (not the cache shape), this iteration is safe under
        // concurrent `handle()` calls.
        for entry in self.handles.iter() {
            let cached = entry.value();
            if cached.handle.inner.idle.is_idle() && cached.handle.inner.try_evict_writer() {
                count += 1;
            }
        }
        count
    }

    /// Make room for a new writer permit (ADR-039 amendment-02
    /// §D-14 sweeper). First tries strict idle eviction via
    /// [`Self::evict_idle`]; if no tenant is idle, force-evicts the
    /// LRU writer via [`Self::evict_one_lru`]. Returns the count
    /// evicted.
    ///
    /// **This is an explicit "force make room now" operation** (ops
    /// tooling / direct callers). The automatic pool-admission path
    /// (`WriterPool::acquire`) does NOT call this: it runs the
    /// data-safe [`Self::evict_idle`] eagerly and gates the
    /// [`Self::evict_one_lru`] force-eviction behind the
    /// [`crate::pool::WRITER_ACQUIRE_BLOCK_TIMEOUT`] block (so the LRU
    /// step reclaims only orphans, never in-flight writers — #575).
    /// See `Self::idle_sweeper` / `Self::orphan_evictor`.
    pub fn evict_to_make_room(&self) -> usize {
        let strict = self.evict_idle();
        if strict > 0 {
            return strict;
        }
        self.evict_one_lru()
    }

    /// Force-evict the single least-recently-written **active** writer
    /// (ADR-039 amendment-02 §D-14 "LRU fallback for orphan writers").
    /// Returns `1` if a writer was evicted, `0` if none was eligible.
    ///
    /// **#575 contract.** This is a forced eviction that DROPS the
    /// evicted writer's in-memory buffer (Tantivy `IndexWriter::drop`
    /// rolls back uncommitted adds, ADR-039 §D-6). It is therefore
    /// data-safe ONLY for orphans — tenants that `upsert_document`-ed
    /// but will never commit / rollback, whose buffer is an abandoned
    /// write equivalent to a rollback. It MUST NOT be run on the eager
    /// admission path, where it would drop an in-flight writer's
    /// committed-intent buffer (the #575 data loss).
    /// `WriterPool::acquire` only reaches it AFTER the admission block
    /// elapses with no natural permit release
    /// ([`crate::pool::WRITER_ACQUIRE_BLOCK_TIMEOUT`]) — which a writer
    /// that commits WITHIN the timeout never triggers (it releases its
    /// permit at the natural commit cadence first, §D-14). A **slower**
    /// in-flight writer whose `upsert → commit` gap exceeds the timeout
    /// is indistinguishable from an orphan by timing alone and IS
    /// reclaimed here, dropping its buffer — the accepted #575 residual
    /// (ADR-039 amendment-03 §D-18; genuine close tracked to #627), NOT
    /// an "in-flight writers are never evicted" guarantee.
    ///
    /// **Reentrant safety.** The LRU iteration uses `Mutex::try_lock`
    /// when checking each tenant's writer slot so the sweep is
    /// reentrant-safe when invoked from inside
    /// `Bm25IndexHandle::ensure_writer` (which holds the current
    /// tenant's writer mutex). Try-lock failures skip the candidate —
    /// the current tenant's lock failure naturally excludes it from
    /// the LRU set, and any writer **actively inside** `commit` /
    /// `upsert` (mutex held) is skipped too. NOTE: `try_lock` skips a
    /// writer ONLY while it actively holds the mutex; a writer parked in
    /// the `upsert → commit` gap (mutex free, committed-intent buffer
    /// held) is NOT skipped and IS an eligible victim — that is the #575
    /// residual (ADR-039 amendment-03 §D-18), bounded to in-flight
    /// writers whose `upsert → commit` gap exceeds
    /// [`crate::pool::WRITER_ACQUIRE_BLOCK_TIMEOUT`].
    pub fn evict_one_lru(&self) -> usize {
        // Find the oldest active writer and evict. Iterate using
        // `try_lock` so the sweep is reentrant-safe when invoked from
        // inside `ensure_writer`.
        let mut oldest_key: Option<(TenantId, IndexId)> = None;
        let mut oldest_time: Option<std::time::Instant> = None;
        for entry in self.handles.iter() {
            let key = *entry.key();
            let cached = entry.value();
            // Try-lock — if locked, that tenant is in active use;
            // skipping is safe (it is a poor eviction candidate
            // anyway, and skipping the current thread's own
            // handle prevents reentrant deadlock).
            let Some(guard) = cached.handle.inner.writer.try_lock() else {
                continue;
            };
            if guard.is_some() {
                let lwt = cached.handle.inner.idle.last_write_time();
                match oldest_time {
                    None => {
                        oldest_time = Some(lwt);
                        oldest_key = Some(key);
                    }
                    Some(prev) if lwt < prev => {
                        oldest_time = Some(lwt);
                        oldest_key = Some(key);
                    }
                    _ => {}
                }
            }
            // Drop the try-lock guard at end of iteration scope.
        }
        if let Some(key) = oldest_key
            && let Some(entry) = self.handles.get(&key)
            && entry.value().handle.inner.evict_writer()
        {
            return 1;
        }
        0
    }

    /// Permits currently outstanding in the shared pool — i.e.,
    /// the count of tenants with a live `IndexWriter`.
    #[must_use]
    pub fn active_writer_count(&self) -> usize {
        self.pool.in_use()
    }

    /// Pool capacity. Constant for the lifetime of the service.
    #[must_use]
    pub fn pool_capacity(&self) -> usize {
        self.pool.capacity()
    }

    /// Build the **eager, data-safe** on-full sweep closure
    /// (`WriterPool::acquire`'s `on_full`): strict-idle orphan
    /// reclamation only ([`Self::evict_idle`]), the §D-14 "strict
    /// idle" first tier. Captures a `Weak<Bm25Service>` so the pool
    /// can invoke it without a strong reference cycle.
    ///
    /// Strict-idle eviction NEVER drops an in-flight writer's
    /// committed-intent buffer (a writer crosses the idle thresholds
    /// only after [`crate::IDLE_EVICTION_COMMIT_THRESHOLD`] empty
    /// commits or [`crate::IDLE_EVICTION_WALL_CLOCK_THRESHOLD_SECS`]
    /// of no writes), so it is safe to run eagerly the instant the
    /// pool is found full.
    fn idle_sweeper(&self) -> Sweeper {
        let weak = self.me.get().cloned().unwrap_or_else(Weak::<Self>::new);
        Arc::new(move || weak.upgrade().map(|s| s.evict_idle()).unwrap_or(0))
    }

    /// Build the **timeout-gated** forced-eviction closure
    /// (`WriterPool::acquire`'s `on_block_timeout`): force-evict the
    /// LRU writer ([`Self::evict_one_lru`]), the §D-14 "LRU fallback
    /// for orphan writers". Captures a `Weak<Bm25Service>` so the pool
    /// can invoke it without a strong reference cycle.
    ///
    /// The pool reaches this ONLY after the admission block elapses
    /// ([`crate::pool::WRITER_ACQUIRE_BLOCK_TIMEOUT`]) with no natural
    /// permit release. An in-flight writer that commits within the
    /// timeout releases its permit at the natural commit cadence and
    /// wakes the blocked acquirer first, so it is not the victim. A
    /// slower in-flight writer (`upsert → commit` gap > the timeout) is
    /// indistinguishable from an orphan by timing alone and IS reclaimed
    /// here, dropping its buffer — the accepted #575 residual (ADR-039
    /// amendment-03 §D-18; genuine close tracked to #627).
    fn orphan_evictor(&self) -> Sweeper {
        let weak = self.me.get().cloned().unwrap_or_else(Weak::<Self>::new);
        Arc::new(move || weak.upgrade().map(|s| s.evict_one_lru()).unwrap_or(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arcgraph_core::Lsn;
    use tempfile::tempdir;

    #[test]
    fn service_creates_per_tenant_directory_on_first_touch() {
        let tmp = tempdir().expect("tempdir");
        let svc = Bm25Service::new(tmp.path().to_path_buf());
        assert!(!svc.has_handle(TenantId::DEFAULT, IndexId::DEFAULT_BM25));

        let _h = svc
            .handle(TenantId::DEFAULT, IndexId::DEFAULT_BM25)
            .expect("first-touch handle creates directory");
        assert!(svc.has_handle(TenantId::DEFAULT, IndexId::DEFAULT_BM25));

        let dir = svc.tenant_index_dir(TenantId::DEFAULT, IndexId::DEFAULT_BM25);
        assert!(dir.exists(), "directory must exist after first handle");
    }

    #[test]
    fn service_handle_returns_same_arc_on_repeat() {
        let tmp = tempdir().expect("tempdir");
        let svc = Bm25Service::new(tmp.path().to_path_buf());
        let a = svc
            .handle(TenantId::DEFAULT, IndexId::DEFAULT_BM25)
            .expect("first");
        let b = svc
            .handle(TenantId::DEFAULT, IndexId::DEFAULT_BM25)
            .expect("second");
        assert!(
            Arc::ptr_eq(&a, &b),
            "second handle must be same Arc (append-only cache)"
        );
    }

    #[test]
    fn handle_writer_slot_is_empty_before_first_write() {
        let tmp = tempdir().expect("tempdir");
        let svc = Bm25Service::new(tmp.path().to_path_buf());
        let h = svc
            .handle(TenantId::DEFAULT, IndexId::DEFAULT_BM25)
            .expect("handle");
        assert!(
            !h.has_active_writer(),
            "lazy writer: slot must be empty before the first upsert/delete"
        );
        assert_eq!(svc.active_writer_count(), 0);
    }

    #[test]
    fn first_upsert_allocates_writer_and_consumes_one_permit() {
        let tmp = tempdir().expect("tempdir");
        let svc = Bm25Service::new(tmp.path().to_path_buf());
        let h = svc
            .handle(TenantId::DEFAULT, IndexId::DEFAULT_BM25)
            .expect("handle");
        h.upsert_document(arcgraph_core::NodeId::new(1), "alpha", Lsn::new(1))
            .expect("upsert");
        assert!(h.has_active_writer());
        assert_eq!(
            svc.active_writer_count(),
            1,
            "the first upsert must consume exactly one pool permit"
        );
    }

    #[test]
    fn upsert_then_search_round_trips_with_commit() {
        let tmp = tempdir().expect("tempdir");
        let svc = Bm25Service::new(tmp.path().to_path_buf());
        let h = svc
            .handle(TenantId::DEFAULT, IndexId::DEFAULT_BM25)
            .expect("handle");
        h.upsert_document(
            arcgraph_core::NodeId::new(42),
            "the quick brown fox jumps over the lazy dog",
            Lsn::new(10),
        )
        .expect("upsert");
        h.commit().expect("commit");

        let hits = h
            .search("fox", 10, Lsn::new(100))
            .expect("search after commit");
        assert!(!hits.is_empty(), "should match the upserted doc");
        assert_eq!(hits[0].0.raw(), 42, "node_id must round-trip");
    }

    #[test]
    fn search_excludes_doc_with_commit_lsn_above_read_lsn() {
        // ADR-039 §D-3 visibility filter: a doc committed at
        // commit_lsn = 100 must NOT be visible to a reader at
        // read_lsn = 50.
        let tmp = tempdir().expect("tempdir");
        let svc = Bm25Service::new(tmp.path().to_path_buf());
        let h = svc
            .handle(TenantId::DEFAULT, IndexId::DEFAULT_BM25)
            .expect("handle");
        h.upsert_document(arcgraph_core::NodeId::new(7), "future doc", Lsn::new(100))
            .expect("upsert");
        h.commit().expect("commit");

        let hits_stale = h
            .search("future", 10, Lsn::new(50))
            .expect("search at stale LSN");
        assert!(
            hits_stale.is_empty(),
            "doc at commit_lsn 100 must be invisible at read_lsn 50"
        );

        let hits_fresh = h
            .search("future", 10, Lsn::new(150))
            .expect("search at fresh LSN");
        assert_eq!(
            hits_fresh.len(),
            1,
            "doc at commit_lsn 100 must be visible at read_lsn 150"
        );
    }

    #[test]
    fn delete_document_excludes_match_after_commit() {
        let tmp = tempdir().expect("tempdir");
        let svc = Bm25Service::new(tmp.path().to_path_buf());
        let h = svc
            .handle(TenantId::DEFAULT, IndexId::DEFAULT_BM25)
            .expect("handle");
        h.upsert_document(arcgraph_core::NodeId::new(99), "ephemeral", Lsn::new(1))
            .expect("upsert");
        h.commit().expect("commit upsert");

        h.delete_document(arcgraph_core::NodeId::new(99), Lsn::new(2))
            .expect("delete");
        h.commit().expect("commit delete");

        let hits = h.search("ephemeral", 10, Lsn::new(100)).expect("search");
        assert!(hits.is_empty(), "deleted doc must not match");
    }

    #[test]
    fn filtered_search_any_routes_to_search() {
        let tmp = tempdir().expect("tempdir");
        let svc = Bm25Service::new(tmp.path().to_path_buf());
        let h = svc
            .handle(TenantId::DEFAULT, IndexId::DEFAULT_BM25)
            .expect("handle");
        h.upsert_document(arcgraph_core::NodeId::new(1), "alpha beta", Lsn::new(1))
            .expect("upsert");
        h.commit().expect("commit");

        let hits = h
            .filtered_search("alpha", 10, &crate::Filter::Any, Lsn::new(100))
            .expect("filtered_search");
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn filtered_search_tenant_returns_filter_not_supported() {
        let tmp = tempdir().expect("tempdir");
        let svc = Bm25Service::new(tmp.path().to_path_buf());
        let h = svc
            .handle(TenantId::DEFAULT, IndexId::DEFAULT_BM25)
            .expect("handle");

        let err = h
            .filtered_search(
                "anything",
                10,
                &crate::Filter::Tenant(TenantId::DEFAULT),
                Lsn::new(1),
            )
            .expect_err("Tenant filter must surface FilterNotSupported");
        assert!(matches!(err, Bm25Error::FilterNotSupported { .. }));
    }

    #[test]
    fn search_with_zero_k_returns_empty() {
        let tmp = tempdir().expect("tempdir");
        let svc = Bm25Service::new(tmp.path().to_path_buf());
        let h = svc
            .handle(TenantId::DEFAULT, IndexId::DEFAULT_BM25)
            .expect("handle");
        let hits = h.search("anything", 0, Lsn::new(1)).expect("search k=0");
        assert!(hits.is_empty());
    }

    #[test]
    fn commit_releases_writer_per_request_scoped_semantics() {
        // ADR-039 amendment-02 §D-14: commit drops the writer slot
        // and returns the pool permit. Pin so a regression that
        // re-introduces tenant-scoped writer caching surfaces
        // here.
        let tmp = tempdir().expect("tempdir");
        let svc = Bm25Service::new(tmp.path().to_path_buf());
        let h = svc
            .handle(TenantId::DEFAULT, IndexId::DEFAULT_BM25)
            .expect("handle");
        h.upsert_document(arcgraph_core::NodeId::new(1), "alpha", Lsn::new(1))
            .expect("upsert");
        assert!(h.has_active_writer(), "writer allocated by upsert");
        assert_eq!(svc.active_writer_count(), 1);

        h.commit().expect("commit");
        assert!(
            !h.has_active_writer(),
            "post-commit writer slot must be empty (request-scoped)"
        );
        assert_eq!(
            svc.active_writer_count(),
            0,
            "post-commit pool permit must be returned"
        );
    }

    #[test]
    fn post_commit_reallocate_round_trips_through_disk() {
        // Eviction-recreate cycle (commit-driven): a tenant whose
        // writer was just dropped (by commit) must be able to write
        // again, with the new write landing in a fresh
        // `IndexWriter` against the same on-disk Tantivy index.
        let tmp = tempdir().expect("tempdir");
        let svc = Bm25Service::new(tmp.path().to_path_buf());
        let h = svc
            .handle(TenantId::DEFAULT, IndexId::DEFAULT_BM25)
            .expect("handle");
        h.upsert_document(arcgraph_core::NodeId::new(1), "first", Lsn::new(1))
            .expect("first upsert");
        h.commit().expect("commit first");
        assert!(!h.has_active_writer(), "post-commit writer is None");

        // Now write again — must allocate a fresh writer.
        h.upsert_document(arcgraph_core::NodeId::new(2), "second", Lsn::new(2))
            .expect("upsert post-commit");
        assert!(h.has_active_writer(), "second upsert reallocates writer");
        h.commit().expect("commit second");

        let hits_first = h.search("first", 10, Lsn::new(100)).expect("search first");
        assert_eq!(hits_first.len(), 1, "first batch doc must remain visible");

        let hits_second = h
            .search("second", 10, Lsn::new(100))
            .expect("search second");
        assert_eq!(hits_second.len(), 1, "post-commit doc must be visible");
    }

    #[test]
    fn pool_capacity_matches_default_constant() {
        let tmp = tempdir().expect("tempdir");
        let svc = Bm25Service::new(tmp.path().to_path_buf());
        assert_eq!(svc.pool_capacity(), WRITER_POOL_SIZE);
    }

    #[test]
    fn with_pool_size_overrides_capacity() {
        let tmp = tempdir().expect("tempdir");
        let svc = Bm25Service::with_pool_size(tmp.path().to_path_buf(), 4);
        assert_eq!(svc.pool_capacity(), 4);
    }

    #[test]
    fn default_writer_heap_is_16_mib() {
        // ADR-039 amendment-01 §D-11(a): the default heap is
        // 16 MiB (Tantivy minimum 15 MiB × 1.05 safety margin).
        // Pinned so a regression that bumps it back to 64 MiB
        // surfaces here rather than at deployment time.
        assert_eq!(DEFAULT_WRITER_HEAP_BYTES, 16 * 1024 * 1024);
    }
}
