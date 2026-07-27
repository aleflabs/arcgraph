//! Vector arena page store — M3.a Slice G.1 stub (ADR-035 §7.5).
//!
//! Vector arenas (HNSW graph + DiskANN segment pages) live behind the
//! [`VectorPageStoreHandle`] trait so the WAL replay executor can
//! route [`crate::wal::bundle::BundlePageKind::Vector`] entries into
//! a per-tenant arena without `arcgraph-storage` taking a dependency
//! on the `arcgraph-vector` crate. This mirrors the
//! [`crate::blob::BlobStoreHandle`] pattern PR #86 (N-2) introduced
//! for blob chains.
//!
//! # Slice scope
//!
//! Slice G.1 lands the **types and trait** only:
//!
//! - [`VectorPageStoreHandle`] — replay-side trait.
//! - [`VectorStoreError`] — error taxonomy stub; extended by later
//!   slices.
//! - [`VectorPageStore`] — placeholder struct that implements the
//!   trait with `unimplemented!()` bodies so the compiler can wire
//!   the dispatch arm in [`crate::wal::replay`] without G.2/G.3/G.4/
//!   G.5 having landed.
//!
//! Subsequent slices populate the bodies:
//!
//! - **G.2 (snapshot).** Persist arena snapshots through
//!   `install_or_replace`.
//! - **G.3 (recovery).** Reload arenas from snapshots on startup.
//! - **G.4 (CommitBundle staging).** Stage pre-mutation arena pages
//!   into the v3 `staged_pages` section at commit time.
//! - **G.5 (rollback).** Wire `restore_page_bytes` into the
//!   `TxnManager::rollback_wal_failure` Z-1 (b) drain via
//!   [`crate::mutation_log::PageStoreKind::Vector`].
//!
//! # Local-only hooks (ADR-035 §8)
//!
//! `VectorPageStoreHandle` is **tenant-keyed by construction** to
//! match `BlobStoreHandle`'s multi-tenant physical layer.

use std::sync::Arc;

use arcgraph_core::{PageId, TenantId};
use thiserror::Error;

// M3.a Phase 5 sub-modules. Each placeholder is populated by a
// dedicated parallel slice; pre-declared here to prevent parallel
// branches conflicting on this file (same pattern as F-1 fold-in
// pre-declared `pub mod hnsw;` and `pub mod diskann;` in
// `arcgraph-vector/src/lib.rs`).
//
// - `snapshot`: G.2 — ARCV format flush, atomic temp-file + rename.
// - `recovery`: G.3 — recovery path + bootstrap_from_mvcc fallback.
pub mod recovery;
pub mod snapshot;

// G.2 + G.3 public re-exports. G.2 owns flush primitives in
// `snapshot.rs`; G.3 owns recovery + bootstrap primitives in
// `recovery.rs`. Both re-exported here for short import paths.
// Per the slice scope, the G.1 types above
// (`VectorPageStoreHandle`, `VectorPageStore`, `VectorStoreError`
// core variants) are untouched.
pub use snapshot::{
    CrashPoint, SectionKind, SnapshotCatalog, SnapshotPolicy, SnapshotSection, SnapshotSpec,
    SnapshotTrigger, flush_snapshot, flush_snapshot_with_crash_point, snapshot_path,
    snapshot_temp_path,
};

pub use recovery::{
    ArenaRecoveryJob, ArenaSource, EmptyWalDeltaSource, Encoding as SnapshotEncoding,
    IndexType as SnapshotIndexType, MvccVectorSource, ParsedSnapshotName, RecoveredArena,
    SNAPSHOT_FILE_EXT, SNAPSHOT_FOOTER_SIZE, SNAPSHOT_FORMAT_VERSION, SNAPSHOT_HEADER_SIZE,
    SNAPSHOT_MAGIC, SNAPSHOT_TEMP_EXT, VectorArenaPageStore, VectorPageDelta,
    VectorRecoveryRequest, WalDeltaSource, bootstrap_from_mvcc, parse_snapshot_filename,
    recover_all_arenas, recover_arena, snapshot_filename,
};

// ─────────────────────────────────────────────────────────────────────
// Error taxonomy
// ─────────────────────────────────────────────────────────────────────

/// Failure modes for [`VectorPageStoreHandle`] operations.
///
/// Extended by Slices G.2 (snapshot), G.3 (recovery), G.4 (staging),
/// and G.5 (rollback) as their bodies land. The Slice G.1 stub
/// surface is intentionally minimal: only the structural-error
/// shapes the trait method signatures require. Body-specific
/// failures (snapshot I/O errors, recovery checksum mismatches,
/// rollback-drain ordering violations) attach in their owning
/// slices.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum VectorStoreError {
    /// No vector arena is currently registered for this tenant.
    /// Returned when `install_or_replace` or `restore_page_bytes`
    /// targets a tenant that has not yet been opened (e.g., a stale
    /// WAL bundle for a dropped tenant during replay).
    #[error("vector arena not found for tenant {0:?}")]
    ArenaNotFound(TenantId),

    /// The named page is not present in the per-tenant arena.
    /// Returned by rollback paths (G.5) when `restore_page_bytes`
    /// targets a page that was never installed; signals an
    /// upstream `TxnMutationLog` capture-discipline bug.
    #[error("page {1:?} not found in arena for tenant {0:?}")]
    PageNotFound(TenantId, PageId),

    /// A [`snapshot::SnapshotSpec`] failed validation (encoding /
    /// index_type / dim out of range; byte layout overflows
    /// usize). Slice G.2 surfaces this before any I/O so callers
    /// observe a clean failure with no on-disk side effects.
    #[error("invalid snapshot spec: {0}")]
    InvalidSnapshotSpec(String),

    /// I/O failure while flushing a snapshot (open / write / fsync
    /// / rename / dir-fsync). The atomic-write protocol guarantees
    /// a graceful artifact at every interior step (Slice G.2 docs);
    /// this variant just propagates the underlying error message.
    #[error("snapshot I/O failed: {0}")]
    SnapshotIo(String),

    /// Crash injected at a [`snapshot::CrashPoint`] for Path A
    /// boundary tests. Production callers never observe this
    /// variant — it is unreachable from [`snapshot::flush_snapshot`].
    /// The variant is in the public surface so test files in
    /// `tests/` (which only see `pub` types) can match on it.
    #[error("snapshot crash injected at {0:?}")]
    CrashInjected(snapshot::CrashPoint),
}

// ─────────────────────────────────────────────────────────────────────
// Trait
// ─────────────────────────────────────────────────────────────────────

/// Routing target for a [`crate::wal::bundle::BundlePageKind::Vector`]
/// staged page during WAL replay, plus the Z-1 (b) rollback drain
/// hook for vector arena mutations.
///
/// Mirrors [`crate::blob::BlobStoreHandle`] in carrying [`TenantId`]
/// on every method because vector arenas are physically per-tenant
/// (ADR-035 §7.5 — each tenant owns an isolated arena keyed by
/// `(tenant, page_id)`). The two storage families that land
/// page-bytes through the trait are HNSW's per-arena page store and
/// DiskANN's delta segment store; both share this surface so the
/// replay executor + Z-1 (b) drain don't need to know which family
/// owns a given page.
///
/// # Local-only keying (ADR-035 §8)
///
/// The key is `(tenant, page_id)`.
///
/// # Idempotence contract (Lemma I2 — bundle-level)
///
/// `install_or_replace` is an unconditional byte-copy overwrite,
/// matching `BlobStoreHandle` (PR #86 N-2). Replay-level idempotence
/// lives upstream in the executor's
/// `applied_high_water ≥ bundle.commit_lsn` skip — bundle-level
/// supersession is by design, not error.
pub trait VectorPageStoreHandle: Send + Sync {
    /// Install or replace a vector arena page.
    ///
    /// Per-tenant keyed (matches `BlobStoreHandle` from PR #86
    /// N-2). The replay executor calls this for every
    /// [`crate::wal::bundle::BundlePageKind::Vector`] entry in a
    /// decoded `CommitBundle`'s `staged_pages` section. Slice G.2
    /// implements snapshot persistence; Slice G.4 wires the
    /// staging side at commit time so the bytes captured here are
    /// actually populated by user-driven vector mutations.
    fn install_or_replace(
        &self,
        tenant: TenantId,
        page_id: PageId,
        bytes: &[u8],
    ) -> Result<(), VectorStoreError>;

    /// Restore pre-mutation page bytes for Z-1 (b) rollback.
    ///
    /// Called by `TxnManager::rollback_wal_failure` (via the
    /// `PageStoreKind::Vector` arm in `crud.rs` populated by
    /// Slice G.5) to undo a builder-phase vector arena mutation
    /// when the WAL fsync fails. The pre-mutation bytes were
    /// captured under the arena's write latch by the corresponding
    /// G.4 staging path — the same capture-under-latch discipline
    /// `PrimaryPageStore::capture_and_latch` enforces (ADR-033 §3).
    fn restore_page_bytes(
        &self,
        tenant: TenantId,
        page_id: PageId,
        bytes: &[u8],
    ) -> Result<(), VectorStoreError>;
}

// ─────────────────────────────────────────────────────────────────────
// VectorPageStore — Slice G.1 stub implementation
// ─────────────────────────────────────────────────────────────────────

/// Stub vector page store for Slice G.1.
///
/// Compiles, exists, and implements [`VectorPageStoreHandle`] with
/// `unimplemented!()` bodies. Slice G.2 replaces these with real
/// per-tenant arena persistence. Until then, calling
/// `install_or_replace` or `restore_page_bytes` on this type panics
/// — the replay dispatch arm in [`crate::wal::replay`] only
/// instantiates this stub in tests that exercise the dispatch
/// without applying real bytes.
///
/// # Why a stub-with-panic instead of a stub-with-success
///
/// A success-no-op stub would silently let bundles with Vector
/// entries replay against an unwired backend, leaking ghost pages
/// post-recovery. The panic forces production wirings to register a
/// real implementor (Slice G.2's concrete arena store) before any
/// Vector-bearing bundle replays — a fail-loud contract that mirrors
/// `BlobStoreHandle`'s pre-N-2 stub.
#[derive(Debug, Default)]
pub struct VectorPageStore {
    /// Reserved for Slice G.2's per-tenant arena map. Held as a
    /// zero-sized phantom now so the public type lands at G.1
    /// without committing to a concrete in-memory layout that
    /// G.2 may revise after benchmarking.
    _slice_g2_reserved: (),
}

impl VectorPageStore {
    /// Construct an empty stub store. Slice G.2 will replace the
    /// signature with one that takes a buffer pool / I/O handle.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl VectorPageStoreHandle for VectorPageStore {
    fn install_or_replace(
        &self,
        _tenant: TenantId,
        _page_id: PageId,
        _bytes: &[u8],
    ) -> Result<(), VectorStoreError> {
        unimplemented!(
            "VectorPageStore::install_or_replace — Slice G.2 (snapshot persistence) populates this body"
        )
    }

    fn restore_page_bytes(
        &self,
        _tenant: TenantId,
        _page_id: PageId,
        _bytes: &[u8],
    ) -> Result<(), VectorStoreError> {
        unimplemented!(
            "VectorPageStore::restore_page_bytes — Slice G.5 (Z-1 (b) rollback) populates this body"
        )
    }
}

// ─────────────────────────────────────────────────────────────────────
// Convenience aliases
// ─────────────────────────────────────────────────────────────────────

/// Alias for the `Arc<dyn VectorPageStoreHandle>` shape that
/// `PageStoreTarget` and other replay-time wirings hold. Mirrors
/// `BlobStoreHandle`'s typical wrapping. Not currently consumed
/// inside this crate; exists so external slices (G.2/G.3/G.4) and
/// external crates (`arcgraph-vector`) have a single canonical
/// alias to import.
pub type VectorPageStoreArc = Arc<dyn VectorPageStoreHandle>;

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Pin the trait-object shape: `Arc<dyn VectorPageStoreHandle>`
    /// must be `Send + Sync` so it can sit on `PageStoreTarget`.
    /// This is a compile-time assertion via `fn(_: T)`-style
    /// shadowing — failure would surface as a "trait not Send"
    /// build error.
    #[test]
    fn vector_page_store_handle_is_object_safe_send_sync() {
        fn assert_send_sync<T: Send + Sync + ?Sized>() {}
        assert_send_sync::<dyn VectorPageStoreHandle>();
        let store: Arc<dyn VectorPageStoreHandle> = Arc::new(VectorPageStore::new());
        // Use the value so the compiler does not warn it unused.
        let _: VectorPageStoreArc = store;
    }

    /// `VectorStoreError` carries `TenantId` + `PageId` faithfully.
    /// Slices G.2-G.5 will extend the variant set; this test pins
    /// the existing two so a future variant addition is a
    /// deliberate API change rather than a silent rename.
    #[test]
    fn vector_store_error_renders_tenant_and_page() {
        let arena = VectorStoreError::ArenaNotFound(TenantId::DEFAULT);
        let rendered = arena.to_string();
        assert!(rendered.contains("vector arena not found"), "{rendered}");

        let page = VectorStoreError::PageNotFound(TenantId::DEFAULT, PageId::new(7));
        let rendered = page.to_string();
        assert!(rendered.contains("page"), "{rendered}");
        assert!(rendered.contains("not found"), "{rendered}");
    }

    /// The stub panics with a Slice-G.2 hint, not a generic
    /// `unimplemented!()`. Engineers grepping for "Slice G.2" in
    /// the codebase find this stub and know which slice owns the
    /// real body.
    #[test]
    #[should_panic(expected = "Slice G.2")]
    fn install_or_replace_panics_with_slice_hint() {
        let store = VectorPageStore::new();
        let _ = store.install_or_replace(TenantId::DEFAULT, PageId::new(1), &[0u8; 8]);
    }

    /// Same fail-loud contract for the rollback hook. Slice G.5
    /// owns the body; the panic message names that slice so a
    /// premature wiring of the stub into the rollback path
    /// surfaces with an actionable error.
    #[test]
    #[should_panic(expected = "Slice G.5")]
    fn restore_page_bytes_panics_with_slice_hint() {
        let store = VectorPageStore::new();
        let _ = store.restore_page_bytes(TenantId::DEFAULT, PageId::new(1), &[0u8; 8]);
    }
}
