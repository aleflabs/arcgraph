//! Commit-side `Bm25IndexStoreHandle` impl on `Bm25Service`
//! (ADR-039 §D-7).
//!
//! This module is the seam where `arcgraph-bm25` plugs into
//! `arcgraph-storage`'s commit pipeline. The trait
//! [`arcgraph_storage::mutation_log::Bm25IndexStoreHandle`] lives in
//! `arcgraph-storage` (so the kernel commit closure can hold the
//! trait object without a Tantivy dep) and is implemented HERE on
//! [`crate::Bm25Service`]. The asymmetry mirrors
//! `VectorPageStoreHandle`'s split (trait in `arcgraph-storage`,
//! impl in `arcgraph-vector`).
//!
//! # Wiring posture (v1.0)
//!
//! - **Rollback** is wired into the Z-1 (b) closure in
//!   `arcgraph-storage::crud`. On WAL fsync failure, the closure
//!   drains `TxnMutationLog::bm25_pending` and dispatches
//!   [`Bm25Service::rollback_pending`] per tenant.
//! - **Commit** is dormant at the kernel level (transaction.rs is
//!   frozen by the M3.b session boundary). v1.0 tests exercise
//!   [`Bm25Service::commit_pending`] directly.

use arcgraph_core::{Lsn, NodeId, TenantId};
use arcgraph_storage::mutation_log::{Bm25IndexStoreHandle, Bm25StoreError};

use crate::error::Bm25Error;
use crate::handle::IndexId;
use crate::service::Bm25Service;

impl Bm25IndexStoreHandle for Bm25Service {
    fn upsert_document(
        &self,
        tenant: TenantId,
        node_id: NodeId,
        text: &str,
        commit_lsn: Lsn,
    ) -> Result<(), Bm25StoreError> {
        let handle = self
            .handle(tenant, IndexId::DEFAULT_BM25)
            .map_err(to_store_error)?;
        handle
            .upsert_document(node_id, text, commit_lsn)
            .map_err(to_store_error)
    }

    fn delete_document(
        &self,
        tenant: TenantId,
        node_id: NodeId,
        commit_lsn: Lsn,
    ) -> Result<(), Bm25StoreError> {
        let handle = self
            .handle(tenant, IndexId::DEFAULT_BM25)
            .map_err(to_store_error)?;
        handle
            .delete_document(node_id, commit_lsn)
            .map_err(to_store_error)
    }

    fn search(
        &self,
        tenant: TenantId,
        query: &str,
        k: usize,
        read_lsn: Lsn,
    ) -> Result<Vec<(NodeId, f32)>, Bm25StoreError> {
        let handle = self
            .handle(tenant, IndexId::DEFAULT_BM25)
            .map_err(to_store_error)?;
        handle.search(query, k, read_lsn).map_err(to_store_error)
    }

    /// Durably commit the per-tenant `IndexWriter` then reload the
    /// reader so the next `searcher()` snapshot observes the
    /// just-committed docs (ADR-039 §D-5).
    ///
    /// Per ADR-039 amendment-01 §D-11(c) (implemented in
    /// amendment-02 §D-14), the writer is **request-scoped**:
    /// commit_pending closes the active write window by committing
    /// any buffered docs and then **dropping the writer slot**,
    /// returning the [`crate::pool::WriterPermit`] to the shared
    /// pool. This is what bounds active-set RAM at
    /// `WRITER_POOL_SIZE × DEFAULT_WRITER_HEAP_BYTES` regardless
    /// of tenant fan-out: any tenant whose commit is complete
    /// holds zero pool resources until its next write begins.
    ///
    /// Cost: the next `upsert_document` re-allocates an
    /// `IndexWriter` against the (already-on-disk) Tantivy index
    /// — bounded by `meta.json` parse + segment scan, typically
    /// tens of ms (see ADR-039 amendment-02 §D-12 evicted-rewrite
    /// bench). v1.1 with high-throughput single-tenant workloads
    /// may add a per-handle `commit_keep_writer` toggle if
    /// production data shows this overhead is load-bearing; v1.0
    /// prioritizes pool bound + forward progress.
    ///
    /// If the writer slot is `None` on entry (tenant never wrote,
    /// or was evicted between WAL fsync and this call), the
    /// commit is a no-op (there is nothing buffered to commit).
    /// The reader is reloaded regardless so any prior committed
    /// segments remain observable.
    ///
    /// Returns [`Bm25StoreError::TenantNotFound`] when no
    /// `Bm25IndexHandle` has been materialised for `tenant`. The
    /// rollback closure in `arcgraph-storage::crud` warn-and-skips
    /// on this error (matches the Vector arm posture); this is
    /// expected when a txn registered the tenant via
    /// `note_bm25_tenant` but the upsert path failed before opening
    /// the per-tenant directory.
    fn commit_pending(&self, tenant: TenantId) -> Result<(), Bm25StoreError> {
        let handle = self.handle_for(tenant, IndexId::DEFAULT_BM25).ok_or(
            Bm25StoreError::TenantNotFound {
                tenant_raw: tenant.raw(),
            },
        )?;
        // Take ownership of the ActiveWriter under the slot lock,
        // THEN run `commit` after the take. Whether the Tantivy
        // commit succeeds or returns `Err`, the taken `ActiveWriter`
        // (and its `WriterPermit`) drops at the end of this scope
        // unconditionally — the permit returns to the pool on the
        // error path the same as on success. Codex PR #221 F1
        // regression pin: a sustained Tantivy I/O error stream MUST
        // NOT exhaust the pool by leaking permits in pinned slots.
        let commit_result = {
            let mut guard = handle.inner.writer.lock();
            // `take()` drops the slot to `None` even if the
            // subsequent commit propagates `Err`.
            match guard.take() {
                Some(mut active) => {
                    active
                        .writer
                        .commit()
                        .map(|_| ())
                        .map_err(|e| Bm25StoreError::Tantivy {
                            message: e.to_string(),
                        })
                }
                None => Ok(()),
            }
            // `active` (and its `_permit`) drops here unconditionally
            // when the match arm exits — the pool permit is returned
            // before the error is propagated.
        };
        commit_result?;
        handle
            .inner
            .reader
            .reload()
            .map_err(|e| Bm25StoreError::Tantivy {
                message: e.to_string(),
            })?;
        // Bump the idle tracker on the commit axis. With
        // request-scoped semantics this is mostly observational —
        // the writer is already dropped — but the count is still
        // useful for the wall-clock-axis fallback that catches
        // orphaned write-without-commit cases.
        handle.inner.idle.note_commit();
        Ok(())
    }

    /// Discard the per-tenant `IndexWriter` buffer (ADR-039 §D-6).
    ///
    /// Tantivy's `IndexWriter::rollback` discards every
    /// `add_document` / `delete_term` since the last `commit`. This
    /// is the rollback granularity at v1.0 — there is no per-page
    /// pre-W byte snapshot for BM25.
    ///
    /// Per amendment-02 §D-14 (request-scoped semantics), the
    /// writer slot is **dropped** after rollback so the caller's
    /// pool permit is returned. If the slot is already `None` on
    /// entry (post-eviction or never-written), the rollback is a
    /// no-op.
    ///
    /// Returns [`Bm25StoreError::TenantNotFound`] under the same
    /// conditions as [`Self::commit_pending`].
    fn rollback_pending(&self, tenant: TenantId) -> Result<(), Bm25StoreError> {
        let handle = self.handle_for(tenant, IndexId::DEFAULT_BM25).ok_or(
            Bm25StoreError::TenantNotFound {
                tenant_raw: tenant.raw(),
            },
        )?;
        // Same Pattern-A early-take as `commit_pending`: take the
        // `ActiveWriter` first so its `WriterPermit` drops on the
        // error path as well as the success path. Codex PR #221 F1.
        // `active` (and its `_permit`) drops at the match arm's end,
        // releasing the pool permit before any `Err` from `rollback()`
        // is propagated.
        let mut guard = handle.inner.writer.lock();
        match guard.take() {
            Some(mut active) => {
                active
                    .writer
                    .rollback()
                    .map(|_| ())
                    .map_err(|e| Bm25StoreError::Tantivy {
                        message: e.to_string(),
                    })
            }
            None => Ok(()),
        }
    }
}

fn to_store_error(err: Bm25Error) -> Bm25StoreError {
    match err {
        Bm25Error::Io { message } | Bm25Error::Tantivy { message } => {
            Bm25StoreError::Tantivy { message }
        }
        Bm25Error::QueryParse { message } => Bm25StoreError::QueryParse { message },
        Bm25Error::SchemaViolation { detail } => {
            Bm25StoreError::SchemaViolation { message: detail }
        }
        Bm25Error::FilterNotSupported { variant } => {
            Bm25StoreError::SearchUnavailable { message: variant }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arcgraph_core::{Lsn, NodeId, PartitionId, TenantId};
    use arcgraph_storage::mutation_log::Bm25IndexStoreHandle;
    use tempfile::tempdir;

    use crate::Bm25Service;
    use crate::handle::IndexId;

    /// Pin: the impl satisfies `Send + Sync`, matching the trait
    /// requirement so an `Arc<Bm25Service>` can sit on
    /// `CrudStore::bm25_store` and `TenantHandle::bm25`.
    #[test]
    fn bm25_service_is_send_sync_via_trait_object() {
        fn assert_send_sync<T: Send + Sync + ?Sized>() {}
        assert_send_sync::<dyn Bm25IndexStoreHandle>();
    }

    /// Commit-side dispatch round-trip: open handle, upsert, dispatch
    /// `commit_pending` via the trait object, observe doc visible.
    #[test]
    fn commit_pending_via_trait_object_publishes_doc() {
        let tmp = tempdir().expect("tempdir");
        let svc = Bm25Service::new(tmp.path().to_path_buf());
        let h = svc
            .handle(TenantId::DEFAULT, IndexId::DEFAULT_BM25)
            .expect("handle");
        h.upsert_document(NodeId::new(1), "the cat sat on the mat", Lsn::new(5))
            .expect("upsert");
        // Commit via the trait object on the SAME service Arc.
        let trait_obj: Arc<dyn Bm25IndexStoreHandle> = Arc::clone(&svc) as _;
        trait_obj
            .commit_pending(TenantId::DEFAULT)
            .expect("commit_pending");

        let hits = h
            .search("cat", 10, Lsn::new(100))
            .expect("search after commit_pending");
        assert_eq!(hits.len(), 1, "doc must be visible after commit_pending");
        assert_eq!(hits[0].0.raw(), 1);
    }

    /// Rollback-side dispatch: open handle, upsert (uncommitted),
    /// dispatch `rollback_pending` via the trait object, observe
    /// doc absent. Matches the Z-1 (b) closure call shape from
    /// `arcgraph-storage::crud`.
    #[test]
    fn rollback_pending_via_trait_object_discards_uncommitted_doc() {
        let tmp = tempdir().expect("tempdir");
        let svc = Bm25Service::new(tmp.path().to_path_buf());
        let h = svc
            .handle(TenantId::DEFAULT, IndexId::DEFAULT_BM25)
            .expect("handle");
        h.upsert_document(NodeId::new(2), "buffered text", Lsn::new(10))
            .expect("upsert");
        // Pre-commit search would still not see the doc because the
        // reader is on a snapshot; so we don't assert visibility
        // before rollback. Just dispatch rollback through the trait.
        let trait_obj: Arc<dyn Bm25IndexStoreHandle> = Arc::clone(&svc) as _;
        trait_obj
            .rollback_pending(TenantId::DEFAULT)
            .expect("rollback_pending");

        // After rollback there is no segment to commit; commit +
        // reload via the public handle should yield zero hits.
        h.commit().expect("commit empty post-rollback");
        let hits = h
            .search("buffered", 10, Lsn::new(100))
            .expect("search post-rollback");
        assert!(
            hits.is_empty(),
            "rollback_pending must discard uncommitted upsert"
        );
    }

    /// `commit_pending` for an unopened tenant surfaces
    /// `TenantNotFound`. The rollback closure in
    /// `arcgraph-storage::crud` warns-and-skips on this — pin the
    /// surface so the warn path stays the same shape across v1.0
    /// patches.
    #[test]
    fn commit_pending_unknown_tenant_returns_tenant_not_found() {
        let tmp = tempdir().expect("tempdir");
        let svc = Bm25Service::new(tmp.path().to_path_buf());
        let trait_obj: &dyn Bm25IndexStoreHandle = svc.as_ref();
        let err = trait_obj
            .commit_pending(TenantId::new(9999))
            .expect_err("unopened tenant must error");
        let msg = err.to_string();
        assert!(msg.contains("9999"), "{msg}");
    }

    #[test]
    fn rollback_pending_unknown_tenant_returns_tenant_not_found() {
        let tmp = tempdir().expect("tempdir");
        let svc = Bm25Service::new(tmp.path().to_path_buf());
        let trait_obj: &dyn Bm25IndexStoreHandle = svc.as_ref();
        let err = trait_obj
            .rollback_pending(TenantId::new(7777))
            .expect_err("unopened tenant must error");
        let msg = err.to_string();
        assert!(msg.contains("7777"), "{msg}");
    }

    /// Search-side handle obtained from the service is bound to
    /// `PartitionId::ZERO` per ADR-039 §D-4.
    #[test]
    fn handle_partition_is_zero_at_v1() {
        let tmp = tempdir().expect("tempdir");
        let svc = Bm25Service::new(tmp.path().to_path_buf());
        let h = svc
            .handle(TenantId::DEFAULT, IndexId::DEFAULT_BM25)
            .expect("handle");
        assert_eq!(h.partition(), PartitionId::ZERO);
        assert_eq!(h.index(), IndexId::DEFAULT_BM25);
    }

    /// `commit_pending` against a tenant with no buffered writes
    /// (post-commit or never-written) is a no-op. Pinned because
    /// the rollback / reconciliation paths in
    /// `arcgraph-storage::crud` rely on the noop-on-empty contract.
    #[test]
    fn commit_pending_after_first_commit_is_noop() {
        let tmp = tempdir().expect("tempdir");
        let svc = Bm25Service::new(tmp.path().to_path_buf());
        let h = svc
            .handle(TenantId::DEFAULT, IndexId::DEFAULT_BM25)
            .expect("handle");
        h.upsert_document(NodeId::new(1), "alpha", Lsn::new(1))
            .expect("upsert");
        let trait_obj: Arc<dyn Bm25IndexStoreHandle> = Arc::clone(&svc) as _;
        trait_obj
            .commit_pending(TenantId::DEFAULT)
            .expect("first commit_pending");
        assert!(
            !h.has_active_writer(),
            "post-commit writer slot is None per request-scoped"
        );

        // Second commit_pending without any new writes — must not
        // error and must leave the cache shape intact.
        trait_obj
            .commit_pending(TenantId::DEFAULT)
            .expect("commit_pending without buffered writes is a noop");
    }
}
