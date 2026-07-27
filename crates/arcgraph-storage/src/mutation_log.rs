//! Per-transaction mutation log for WAL-failure rollback.
//!
//! Tracks every in-memory page mutation (and blob chain allocation,
//! and root-pointer change) that a transaction performs during its
//! builder phase. On WAL fsync failure the log is drained by
//! `crate::transaction::TxnManager::rollback_wal_failure` to restore
//! the page stores to their pre-W state, preventing the Z-1 ghost-
//! page hazard.
//!
use arcgraph_core::record::PAGE_SIZE;
use arcgraph_core::{Lsn, NodeId, PageId, TenantId};
use smallvec::SmallVec;

use crate::wal::delta::DeltaIntent;

/// Raw 8 KiB page buffer. Matches `primary_index::PageBuf` and
/// `record_store::PageBuf`; re-aliased here so the mutation log is
/// agnostic to which store its entries originate from.
pub type PageBuf = [u8; PAGE_SIZE];

// ─────────────────────────────────────────────────────────────────────
// PageStoreKind
// ─────────────────────────────────────────────────────────────────────

/// Which store a `new_pages` entry refers to.
///
/// On rollback, `crate::transaction::TxnManager::rollback_wal_failure`
/// dispatches on this kind to pick the correct `DashMap::remove` on
/// the corresponding page store.
///
/// v1.0 has three in-memory page stores:
/// [`crate::primary_index::PrimaryPageStore`] (primary B-tree pages),
/// [`crate::record_store::RecordPageStore`] (slotted node and rel
/// pages), and `SecondaryPageStore` (in the `arcgraph-index` crate;
/// secondary B-tree pages). Each kind routes to a different DashMap
/// on rollback.
///
/// The `Vector` variant (M3.a Slice G.1, ADR-035 §7.5) is wired as a
/// stub — Slices G.2/G.3/G.4/G.5 populate the rollback bodies. Until
/// then, draining a `Vector`-kind entry is a `tracing::warn!` no-op
/// that mirrors `Secondary`'s pre-F-1 behaviour.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum PageStoreKind {
    /// Primary B-tree index pages.
    Primary,
    /// Slotted record pages (Node and Rel).
    Record,
    /// Secondary B-tree index pages (lives in `arcgraph-index`;
    /// rollback dispatch crosses the crate boundary via a rollback-
    /// hook trait registered by the caller — see ADR-033 §3 and the
    /// Phase 3a `rollback_wal_failure` signature).
    Secondary,
    /// Vector arena pages (HNSW graph + DiskANN segments). Populated
    /// by `VectorPageStoreHandle` (M3.a Slice G.1 stub; bodies in
    /// G.2/G.3/G.4/G.5). Distinct from `Blob`'s `blob_heads`-based
    /// rollback because vector arenas use page-bytes restoration
    /// (matches Primary / Record) rather than chain-walk removal.
    Vector,
    /// BM25 text-search staging. Symbolic at v1.0: BM25 rollback drains
    /// [`TxnMutationLog::bm25_pending`] (per-tenant) rather than
    /// `page_mutations` (per-page). Reserved for v1.1+ if BM25 segment-page
    /// restoration becomes needed (per ADR-039 §D-6). The variant exists
    /// for structural symmetry with Primary / Record / Vector so that a
    /// future replay-time BM25 segment recovery slice can populate the
    /// dispatch body without touching the enum shape.
    ///
    /// At v1.0, no production path pushes `Bm25` entries into
    /// `page_mutations` or `new_pages` — Tantivy's `IndexWriter::rollback`
    /// is the rollback granularity, dispatched via the per-tenant
    /// [`bm25_pending`](TxnMutationLog::bm25_pending) drain.
    Bm25,
}

// ─────────────────────────────────────────────────────────────────────
// IndexHandle
// ─────────────────────────────────────────────────────────────────────

/// Opaque routing token for `TxnMutationLog::root_changes`.
///
/// v1.0 has one primary index per tenant. The handle is a `u32` so
/// future per-tenant-per-index routing (when secondary indexes move
/// to their own store) can encode `(tenant_id_low_bits, index_kind)`
/// without widening the field.
///
/// The actual primary index for a transaction is threaded through
/// the builder closure in the existing code; the handle is a
/// mutation-log-local identifier that rollback uses to look up the
/// correct `root_cache` atomic via an `IndexRegistry` (Phase 3a).
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct IndexHandle(pub u32);

impl IndexHandle {
    /// The single primary index handle for v1.0 single-tenant
    /// deployments. When multi-tenant / multi-index routing lands
    /// at v1.1+, additional handles will be allocated per-index.
    pub const PRIMARY: Self = Self(0);

    /// The single secondary (B-tree) index handle for v1.0
    /// single-tenant deployments. `SecondaryIndex` in `arcgraph-index`
    /// pushes `root_changes` entries tagged with this handle; the
    /// Phase 3a rollback-hook trait for the secondary index stores
    /// the old root back into its `root_cache` atomic.
    pub const SECONDARY: Self = Self(1);
}

// ─────────────────────────────────────────────────────────────────────
// TxnMutationLog
// ─────────────────────────────────────────────────────────────────────

/// Per-transaction record of in-memory state changes that must be
/// unwound on WAL fsync failure. Populated by the builder phase via
/// [`crate::primary_index::PrimaryPageStore::capture_and_latch`] and
/// its sibling helpers (Phase 2b); consumed by
/// `crate::transaction::TxnManager::rollback_wal_failure` (Phase 3a).
///
/// All four mutation-kind smallvecs are drained in a specific order
/// by rollback — see ADR-033 §5 (root-ordering) and §6 (sequence).
/// Callers populate them freely; rollback imposes order.
///
#[derive(Debug, Default)]
pub struct TxnMutationLog {
    /// Pages mutated in place during the builder phase. Each entry
    /// is `(store_kind, page_id, pre_mutation_bytes)`. On rollback,
    /// the pre-mutation bytes are restored via write-latch on the
    /// store identified by `store_kind`.
    ///
    /// **Per-store capture isolation (Y-2 fix, 2026-04-24).** The
    /// `(kind, page_id)` compound key is load-bearing: per-store
    /// `PageId` allocators are independent, so `PageId(1)` in the
    /// primary index and `PageId(1)` in the record store are
    /// DIFFERENT pages. Pre-Y-2 the dedup key was just `page_id`,
    /// which caused a capture on record page 1 to silently no-op
    /// when primary root (= page 1 in the tenant's first index)
    /// had already been captured. That collision is the common
    /// case for small tenants because both allocators start at 1.
    ///
    /// Capture is idempotent within a transaction: the first
    /// `capture_and_latch(kind, pid)` call per `(txn, kind, pid)`
    /// tuple records the pre-W bytes; subsequent calls on the
    /// same page are no-ops (the log already has the pre-W bytes).
    /// See ADR-033 §3.
    ///
    /// Mean commit touches 3–7 pages. `SmallVec` inline-stores the
    /// first 16; spills to heap only for pathological commits. The
    /// linear-scan dedup stays linear in `N_mutations` — acceptable
    /// at N ≤ 16, reconsidered if N grows.
    pub page_mutations: SmallVec<[(PageStoreKind, PageId, Box<PageBuf>); 16]>,

    /// Pages newly installed into a page store during the builder
    /// phase. Each entry is `(store_kind, page_id)`. On rollback,
    /// each page is `DashMap::remove`'d from the appropriate store.
    ///
    /// Distinct from `page_mutations` because "was not there before"
    /// cannot be represented as pre-mutation bytes — restoration is
    /// removal, not byte-overwrite.
    pub new_pages: SmallVec<[(PageStoreKind, PageId); 4]>,

    /// Root-pointer changes per index. Each entry is
    /// `(index_handle, old_root_id)`. On rollback, each index's
    /// `root_cache` atomic is restored to its pre-grow_root root_id.
    ///
    /// Ordering matters (ADR-033 §5): rollback MUST restore root
    /// pointers BEFORE removing newly-installed pages, so an
    /// in-flight reader that captures the new root_id from
    /// `root_cache.load` does not then hit a `MissingPage` on the
    /// removed new-root page.
    pub root_changes: SmallVec<[(IndexHandle, PageId); 2]>,

    /// Blob chain heads allocated via
    /// [`crate::blob::BlobStore::register_uncommitted_chain`]. Each
    /// entry is `(tenant, head_page_id)`. On rollback,
    /// [`crate::blob::BlobStore::remove_uncommitted_chain`] walks
    /// the chain and removes each page from the store's DashMap.
    ///
    /// Chains are keyed by `(tenant, page_id)` so the tenant is part
    /// of the log entry. See ADR-033 §4 for the walk-to-remove
    /// rationale.
    pub blob_heads: SmallVec<[(TenantId, u64); 4]>,

    /// Tenants whose BM25 `tantivy::IndexWriter` has buffered docs
    /// during this txn (ADR-039 §D-5 + §D-6).
    ///
    /// On WAL fsync success, the commit pipeline calls
    /// [`Bm25IndexStoreHandle::commit_pending`] for each entry; on
    /// fsync failure, the Z-1 (b) rollback closure in `crud.rs`
    /// calls [`Bm25IndexStoreHandle::rollback_pending`] per tenant.
    ///
    /// Tantivy's `IndexWriter` buffer is the rollback granularity at
    /// v1.0 — unlike `page_mutations` (per-page) or `blob_heads`
    /// (per-chain), BM25 has no pre-W byte snapshot to restore. The
    /// `IndexWriter::rollback()` call discards the in-memory document
    /// buffer for the failing txn.
    ///
    /// Linear-scan dedup mirrors `page_mutations`'s `(kind, page_id)`
    /// dedup. Mean txn touches 1 tenant; SmallVec inline 4. See
    /// [`Self::note_bm25_tenant`] for the idempotent registration
    /// helper.
    pub bm25_pending: SmallVec<[TenantId; 4]>,

    /// M3 physical redo intents built before the commit's exact contiguous
    /// LSN range is allocated. They are process-local reservation state and
    /// never reach rollback page restoration.
    pub delta_intents: Vec<DeltaIntent>,

    /// Set by the transaction manager when the attached generation emits v9.
    /// CRUD builders use it to stage physical intents without adding work to
    /// legacy v8 commits.
    pub delta_mode: bool,
}

impl TxnMutationLog {
    /// Construct an empty mutation log.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// True iff the log has no recorded mutations.
    ///
    /// Rollback is a no-op on an empty log; `wait_for_install_turn` +
    /// `advance_install_order` around an empty rollback is still
    /// correct (Phase 3 ordering is independent of the log).
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.page_mutations.is_empty()
            && self.new_pages.is_empty()
            && self.root_changes.is_empty()
            && self.blob_heads.is_empty()
            && self.bm25_pending.is_empty()
            && self.delta_intents.is_empty()
    }

    /// Total number of mutation entries across all five kinds
    /// (page_mutations, new_pages, root_changes, blob_heads,
    /// bm25_pending).
    /// Test/observability helper; not used on the hot path.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.page_mutations.len()
            + self.new_pages.len()
            + self.root_changes.len()
            + self.blob_heads.len()
            + self.bm25_pending.len()
    }

    /// True iff `(kind, page_id)` has already been captured in this
    /// log.
    ///
    /// The capture helpers use this to dedupe repeated mutations to
    /// the same page within a single transaction: the first
    /// mutation captures the pre-W bytes; subsequent mutations layer
    /// onto the first. Rollback restores the pre-W bytes, which is
    /// exactly the state the transaction started observing.
    ///
    /// **(kind, page_id) not page_id (Y-2 fix).** Each store has its
    /// own `PageId` allocator, so the page_id domains do not share
    /// a namespace. A dedup on `page_id` alone would cause a capture
    /// of (Record, PageId(1)) to silently no-op when (Primary,
    /// PageId(1)) had already been captured — losing the record-page
    /// snapshot and leaving a ghost on WAL failure.
    #[inline]
    #[must_use]
    pub fn has_captured(&self, kind: PageStoreKind, page_id: PageId) -> bool {
        self.page_mutations
            .iter()
            .any(|(k, pid, _)| *k == kind && *pid == page_id)
    }

    /// Register `tenant` as having BM25 buffered writes in this txn
    /// (ADR-039 §D-5 + §D-6).
    ///
    /// Idempotent: duplicate calls are no-ops via linear scan.
    /// Mean txn touches 1 tenant, so the linear scan is O(1) in
    /// practice; SmallVec inline-stores 4 tenants before spilling.
    ///
    /// Producers (the M4/M5/M6 query layer + future write paths
    /// invoking `Bm25IndexHandle::upsert_document` /
    /// `delete_document`) call this helper for every BM25 mutation
    /// to ensure the rollback closure can dispatch
    /// [`Bm25IndexStoreHandle::rollback_pending`] per touched
    /// tenant. The drain order is fixed by the rollback closure:
    /// `bm25_pending` runs after `blob_heads` (ADR-039 §D-6;
    /// ordering inert because Tantivy's buffer is self-contained).
    #[inline]
    pub fn note_bm25_tenant(&mut self, tenant: TenantId) {
        if !self.bm25_pending.contains(&tenant) {
            self.bm25_pending.push(tenant);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Bm25IndexStoreHandle trait + Bm25StoreError
// ─────────────────────────────────────────────────────────────────────

/// Commit-side hook for BM25 text-search per ADR-039 §D-7.
///
/// Mirrors [`crate::vector_store::VectorPageStoreHandle`] but operates
/// on per-tenant `IndexWriter` state rather than per-page bytes —
/// Tantivy's commit / rollback semantics are the rollback granularity.
/// The trait lives in `arcgraph-storage` (matches
/// `VectorPageStoreHandle`'s home) so the commit pipeline can hold the
/// trait object without taking a dependency on Tantivy or
/// `arcgraph-bm25`. The implementation lives in
/// `arcgraph-bm25::store::Bm25Service`.
///
/// # Wiring posture (v1.0)
///
/// - **Rollback is wired.** The Z-1 (b) drain in `crud.rs` calls
///   [`Self::rollback_pending`] for every entry in
///   [`TxnMutationLog::bm25_pending`] on WAL fsync failure. This is
///   the load-bearing safety hook (ADR-033).
/// - **Commit is not wired into the kernel.** The kernel-level commit
///   path lives in `transaction.rs`, which is frozen by the parallel
///   M3.b session boundary. v1.0 ships the trait + service
///   implementation; tests exercise [`Self::commit_pending`] directly
///   via `Bm25Service`. The `commit_bm25_pending` helper on
///   [`crate::crud::CrudStore`] exists as dormant code that future
///   slices wire into the kernel commit closure.
///
/// At v1.0 this is acceptable: if `commit_pending` is not invoked by
/// the kernel, BM25 docs stay buffered in the per-tenant `IndexWriter`
/// until an explicit commit fires, with no correctness loss — only
/// visibility lag for newly-inserted docs.
pub trait Bm25IndexStoreHandle: Send + Sync {
    /// Buffer an upsert into the tenant's default BM25 index.
    ///
    /// Default impl preserves back-compat for commit/rollback-only test
    /// handles; production `Bm25Service` overrides it.
    fn upsert_document(
        &self,
        tenant: TenantId,
        node_id: NodeId,
        text: &str,
        commit_lsn: Lsn,
    ) -> Result<(), Bm25StoreError> {
        let _ = (tenant, node_id, text, commit_lsn);
        Err(Bm25StoreError::SearchUnavailable {
            message: "BM25 upsert is not implemented by this handle".into(),
        })
    }

    /// Buffer a delete into the tenant's default BM25 index.
    ///
    /// Default impl preserves back-compat for commit/rollback-only test
    /// handles; production `Bm25Service` overrides it.
    fn delete_document(
        &self,
        tenant: TenantId,
        node_id: NodeId,
        commit_lsn: Lsn,
    ) -> Result<(), Bm25StoreError> {
        let _ = (tenant, node_id, commit_lsn);
        Err(Bm25StoreError::SearchUnavailable {
            message: "BM25 delete is not implemented by this handle".into(),
        })
    }

    /// Search the tenant's default BM25 index at `read_lsn`.
    ///
    /// The trait lives in `arcgraph-storage`, so it returns only core
    /// identifiers and scores; concrete Tantivy details remain isolated
    /// in `arcgraph-bm25`.
    fn search(
        &self,
        tenant: TenantId,
        query: &str,
        k: usize,
        read_lsn: Lsn,
    ) -> Result<Vec<(NodeId, f32)>, Bm25StoreError> {
        let _ = (tenant, query, k, read_lsn);
        Err(Bm25StoreError::SearchUnavailable {
            message: "BM25 search is not implemented by this handle".into(),
        })
    }

    /// Durably commit pending BM25 writes for `tenant`. Called AFTER
    /// WAL fsync success per the commit pipeline (ADR-039 §D-5).
    ///
    /// Implementations call `IndexWriter::commit()` followed by
    /// `IndexReader::reload()` so the next `searcher()` snapshot
    /// observes the just-committed docs.
    fn commit_pending(&self, tenant: TenantId) -> Result<(), Bm25StoreError>;

    /// Discard pending BM25 writes for `tenant`. Called by the
    /// Z-1 (b) rollback closure on WAL fsync failure (ADR-039 §D-6 +
    /// ADR-033 §6).
    ///
    /// Implementations call `IndexWriter::rollback()`. The rollback
    /// closure in `crud.rs` warn-and-skips on errors (matches the
    /// Vector arm posture), so transient Tantivy errors here surface
    /// as tracing only and do not cascade into the rollback dispatch.
    fn rollback_pending(&self, tenant: TenantId) -> Result<(), Bm25StoreError>;
}

/// Failure modes for [`Bm25IndexStoreHandle`] operations (ADR-039
/// §D-7).
///
/// `Tantivy` carries `String` (not `tantivy::TantivyError`) so this
/// enum lives in `arcgraph-storage` without taking a Tantivy
/// dependency. Implementations in `arcgraph-bm25` translate
/// `tantivy::TantivyError` to a string at the boundary.
///
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Bm25StoreError {
    /// No Tantivy index has been opened for this tenant. Surfaces
    /// when the rollback drain calls [`Bm25IndexStoreHandle::rollback_pending`]
    /// for a tenant whose `Bm25IndexHandle` was never materialised
    /// (e.g., because the txn registered the tenant via
    /// [`TxnMutationLog::note_bm25_tenant`] but the upsert path then
    /// failed before opening the per-tenant directory). The rollback
    /// closure swallows-with-warn — no correctness violation since
    /// no `IndexWriter` buffer exists to roll back.
    #[error("no Tantivy index opened for tenant {tenant_raw}")]
    TenantNotFound {
        /// Raw u64 from the caller's `TenantId`.
        tenant_raw: u64,
    },

    /// A `tantivy::TantivyError` occurred during commit or rollback.
    /// Carries the rendered error message; the original error type
    /// stays in `arcgraph-bm25` (where Tantivy is a dependency).
    #[error("tantivy error: {message}")]
    Tantivy {
        /// Rendered `tantivy::TantivyError` text.
        message: String,
    },

    /// The attached handle supports commit/rollback but not read-side
    /// search. Production `Bm25Service` overrides the default trait
    /// method; test-only no-op handles can keep the default.
    #[error("bm25 search unavailable: {message}")]
    SearchUnavailable {
        /// Rendered detail.
        message: String,
    },

    /// The text query could not be parsed by the backing BM25 engine.
    #[error("bm25 query parse error: {message}")]
    QueryParse {
        /// Rendered query-parser detail.
        message: String,
    },

    /// The backing BM25 document did not contain expected stored fields.
    #[error("bm25 schema violation: {message}")]
    SchemaViolation {
        /// Rendered schema detail.
        message: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_log_is_empty() {
        let log = TxnMutationLog::new();
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
    }

    #[test]
    fn has_captured_returns_false_on_empty_log() {
        let log = TxnMutationLog::new();
        assert!(!log.has_captured(PageStoreKind::Primary, PageId::new(1)));
        assert!(!log.has_captured(PageStoreKind::Record, PageId::new(1)));
    }

    #[test]
    fn has_captured_detects_recorded_kind_and_page() {
        let mut log = TxnMutationLog::new();
        let pid = PageId::new(42);
        log.page_mutations
            .push((PageStoreKind::Primary, pid, Box::new([0u8; PAGE_SIZE])));
        assert!(log.has_captured(PageStoreKind::Primary, pid));
        assert!(!log.has_captured(PageStoreKind::Primary, PageId::new(43)));
        // Y-2: different kind, same page_id — must NOT dedup.
        assert!(
            !log.has_captured(PageStoreKind::Record, pid),
            "capture on (Primary, pid) must not collide with (Record, pid)"
        );
    }

    #[test]
    fn has_captured_distinguishes_stores_on_same_page_id() {
        // Y-2 regression: PageId(1) is the common first-page
        // allocation for both primary and record stores. A compound
        // (kind, page_id) key is load-bearing to avoid silent
        // no-ops on the second store's capture.
        let mut log = TxnMutationLog::new();
        let pid = PageId::new(1);
        log.page_mutations
            .push((PageStoreKind::Primary, pid, Box::new([0xAA; PAGE_SIZE])));
        log.page_mutations
            .push((PageStoreKind::Record, pid, Box::new([0xBB; PAGE_SIZE])));
        assert!(log.has_captured(PageStoreKind::Primary, pid));
        assert!(log.has_captured(PageStoreKind::Record, pid));
        assert_eq!(log.page_mutations.len(), 2);
    }

    #[test]
    fn len_counts_all_four_kinds() {
        let mut log = TxnMutationLog::new();
        log.page_mutations.push((
            PageStoreKind::Primary,
            PageId::new(1),
            Box::new([0u8; PAGE_SIZE]),
        ));
        log.new_pages.push((PageStoreKind::Primary, PageId::new(2)));
        log.root_changes
            .push((IndexHandle::PRIMARY, PageId::new(3)));
        log.blob_heads.push((TenantId::DEFAULT, 100));
        assert_eq!(log.len(), 4);
        assert!(!log.is_empty());
    }

    #[test]
    fn index_handle_primary_is_zero() {
        // The v1.0 single-index deployment convention: PRIMARY == 0.
        // Stable across the handle-allocation ADR (future v1.1
        // secondary-index work) because the primary handle is always
        // the lowest.
        assert_eq!(IndexHandle::PRIMARY.0, 0);
    }

    #[test]
    fn page_store_kind_equality() {
        assert_eq!(PageStoreKind::Primary, PageStoreKind::Primary);
        assert_ne!(PageStoreKind::Primary, PageStoreKind::Record);
    }

    // ─── M3.a Slice G.1: Vector kind ────────────────────────────────

    #[test]
    fn page_store_kind_vector_round_trip() {
        // PageStoreKind has no on-disk byte representation (it's an
        // ephemeral in-memory rollback router). The "round trip" we
        // pin here is identity through the same APIs that exercise
        // Primary / Record / Secondary: equality, dedup keying, and
        // capture-recording in `TxnMutationLog`.

        // Vector is distinct from every existing variant.
        assert_ne!(PageStoreKind::Vector, PageStoreKind::Primary);
        assert_ne!(PageStoreKind::Vector, PageStoreKind::Record);
        assert_ne!(PageStoreKind::Vector, PageStoreKind::Secondary);
        assert_eq!(PageStoreKind::Vector, PageStoreKind::Vector);

        // `has_captured((Vector, pid))` is independent from other
        // kinds at the same `pid` — this is the Y-2 compound-key
        // invariant the Primary/Record dedup test (above) pins,
        // re-asserted for Vector so a regression in the
        // dedup helper doesn't silently merge vector mutations into
        // a sibling store's snapshot.
        let mut log = TxnMutationLog::new();
        let pid = PageId::new(1);
        log.page_mutations
            .push((PageStoreKind::Primary, pid, Box::new([0xAA; PAGE_SIZE])));
        log.page_mutations
            .push((PageStoreKind::Vector, pid, Box::new([0xCC; PAGE_SIZE])));
        assert!(log.has_captured(PageStoreKind::Primary, pid));
        assert!(log.has_captured(PageStoreKind::Vector, pid));
        assert!(!log.has_captured(PageStoreKind::Record, pid));

        // Vector entries flow through `new_pages` the same way
        // Primary and Record do.
        log.new_pages.push((PageStoreKind::Vector, PageId::new(99)));
        assert!(
            log.new_pages
                .iter()
                .any(|(k, p)| *k == PageStoreKind::Vector && *p == PageId::new(99))
        );
    }

    // ─── ADR-039 Slice 2: Bm25 kind + bm25_pending ──────────────────

    #[test]
    fn page_store_kind_bm25_is_distinct() {
        // Symbolic at v1.0 — `Bm25` is reserved for v1.1+ segment
        // restoration. Equality / inequality with siblings is the
        // entire v1.0 contract here; the variant exists so the
        // dispatch enum is exhaustive over future BM25 work.
        assert_ne!(PageStoreKind::Bm25, PageStoreKind::Primary);
        assert_ne!(PageStoreKind::Bm25, PageStoreKind::Record);
        assert_ne!(PageStoreKind::Bm25, PageStoreKind::Secondary);
        assert_ne!(PageStoreKind::Bm25, PageStoreKind::Vector);
        assert_eq!(PageStoreKind::Bm25, PageStoreKind::Bm25);
    }

    #[test]
    fn note_bm25_tenant_records_first_call() {
        let mut log = TxnMutationLog::new();
        assert!(log.bm25_pending.is_empty());
        assert!(log.is_empty());

        log.note_bm25_tenant(TenantId::DEFAULT);
        assert_eq!(log.bm25_pending.len(), 1);
        assert!(!log.is_empty());
        assert!(log.bm25_pending.contains(&TenantId::DEFAULT));
    }

    #[test]
    fn note_bm25_tenant_dedups_repeat_calls() {
        // ADR-039 §D-6: dedup mirrors `(kind, page_id)` linear-scan
        // dedup on `page_mutations`. Repeated upserts on the same
        // tenant within one txn must register exactly once so the
        // rollback drain calls `rollback_pending(tenant)` exactly
        // once (Tantivy's `IndexWriter::rollback()` is idempotent
        // but per-tenant; redundant calls are wasted work, not a
        // correctness defect).
        let mut log = TxnMutationLog::new();
        log.note_bm25_tenant(TenantId::DEFAULT);
        log.note_bm25_tenant(TenantId::DEFAULT);
        log.note_bm25_tenant(TenantId::DEFAULT);
        assert_eq!(log.bm25_pending.len(), 1);
    }

    #[test]
    fn note_bm25_tenant_records_distinct_tenants() {
        let mut log = TxnMutationLog::new();
        let t1 = TenantId::DEFAULT;
        let t2 = TenantId::new(42);
        let t3 = TenantId::new(100);
        log.note_bm25_tenant(t1);
        log.note_bm25_tenant(t2);
        log.note_bm25_tenant(t3);
        log.note_bm25_tenant(t1); // dedup
        assert_eq!(log.bm25_pending.len(), 3);
        assert!(log.bm25_pending.contains(&t1));
        assert!(log.bm25_pending.contains(&t2));
        assert!(log.bm25_pending.contains(&t3));
    }

    #[test]
    fn len_counts_bm25_pending() {
        // Adding the 5th kind extends `len()` by exactly its count
        // — the four-kinds test (above) still passes because its
        // bm25_pending stays empty.
        let mut log = TxnMutationLog::new();
        log.note_bm25_tenant(TenantId::DEFAULT);
        log.note_bm25_tenant(TenantId::new(7));
        assert_eq!(log.len(), 2);
        assert!(!log.is_empty());
    }

    #[test]
    fn bm25_store_error_renders_tenant_id() {
        let err = Bm25StoreError::TenantNotFound { tenant_raw: 42 };
        let rendered = err.to_string();
        assert!(
            rendered.contains("42"),
            "error should render tenant id: {rendered}"
        );
        assert!(rendered.contains("Tantivy") || rendered.contains("tantivy"));
    }

    #[test]
    fn bm25_store_error_carries_tantivy_string() {
        let err = Bm25StoreError::Tantivy {
            message: "writer poisoned".into(),
        };
        let rendered = err.to_string();
        assert!(
            rendered.contains("writer poisoned"),
            "error should embed message: {rendered}"
        );
    }

    /// Trait object shape: `Arc<dyn Bm25IndexStoreHandle>` must be
    /// `Send + Sync` so it can sit on `CrudStore::bm25_store` and
    /// `TenantHandle::bm25` without per-tenant locking.
    #[test]
    fn bm25_index_store_handle_is_object_safe_send_sync() {
        fn assert_send_sync<T: Send + Sync + ?Sized>() {}
        assert_send_sync::<dyn Bm25IndexStoreHandle>();
    }
}
