//! Path-A boundary tests for ADR-039 §D-6 + ADR-033 §6: Z-1 (b)
//! rollback discipline through the `Bm25IndexStoreHandle` trait
//! object.
//!
//! All commit / rollback dispatch in this file goes through
//! `Arc<dyn Bm25IndexStoreHandle>` so the trait object's surface is
//! exercised end-to-end (matching the call shape from
//! `arcgraph-storage::crud`'s rollback closure).
//!
//! PINS:
//! - `rollback_pending_discards_uncommitted_doc` — buffered upsert is
//!   discarded by `rollback_pending`; subsequent search is empty.
//! - `commit_pending_publishes_doc` — symmetric reference: same flow,
//!   `commit_pending` instead of rollback, doc IS visible.
//! - `rollback_then_upsert_then_commit_publishes_only_second_doc` —
//!   after a rollback, the writer is in a clean state; a fresh upsert
//!   commits cleanly with no leakage from the discarded buffer.
//! - `rollback_pending_for_unknown_tenant_returns_tenant_not_found` —
//!   rollback for a never-opened tenant surfaces
//!   `Bm25StoreError::TenantNotFound { tenant_raw }`.
//! - `commit_pending_for_unknown_tenant_returns_tenant_not_found` —
//!   symmetric for the commit arm.
//! - `trait_object_dispatch_works` — `Arc<dyn Bm25IndexStoreHandle>`
//!   is constructible from `Arc<Bm25Service>` and its commit/rollback
//!   methods are reachable through the trait object (object-safety
//!   pin).
//!
//! Failure of any pin is a *contract* break, not a test bug.

use std::sync::Arc;

use arcgraph_bm25::{Bm25Service, IndexId};
use arcgraph_core::{Lsn, NodeId, TenantId};
use arcgraph_storage::mutation_log::{Bm25IndexStoreHandle, Bm25StoreError};
use tempfile::TempDir;

const TENANT: TenantId = TenantId::DEFAULT;

fn fresh_service() -> (TempDir, Arc<Bm25Service>) {
    let tmp = TempDir::new().expect("tempdir");
    let svc = Bm25Service::new(tmp.path().to_path_buf());
    (tmp, svc)
}

// PIN: ADR-039 §D-6 + ADR-033 §6 — `rollback_pending(tenant)` discards
// buffered upserts. After the trait-object rollback dispatch, a search
// (post-implicit-commit by the next commit_pending fixed-up below)
// MUST NOT find the rolled-back doc.
#[test]
fn rollback_pending_discards_uncommitted_doc() {
    let (_tmp, svc) = fresh_service();
    let h = svc.handle(TENANT, IndexId::DEFAULT_BM25).expect("handle");

    // 1. Buffer an upsert.
    h.upsert_document(NodeId::new(1), "rollback_me", Lsn::new(5))
        .expect("buffered upsert");

    // 2. Dispatch rollback through the trait object (matches the
    //    `arcgraph-storage::crud` closure's call shape).
    let trait_obj: Arc<dyn Bm25IndexStoreHandle> = svc.clone();
    trait_obj
        .rollback_pending(TENANT)
        .expect("PIN: rollback_pending must succeed for an opened tenant");

    // 3. After rollback the writer's buffer is empty. A subsequent
    //    `commit_pending` produces an empty segment; the reader must
    //    not surface the rolled-back doc.
    trait_obj
        .commit_pending(TENANT)
        .expect("commit_pending after rollback must succeed (empty buffer)");
    let hits = h
        .search("rollback_me", 10, Lsn::new(100))
        .expect("search post-rollback");
    assert!(
        hits.is_empty(),
        "PIN: ADR-039 §D-6 — `rollback_pending` MUST discard the \
         buffered upsert; reader saw {} hits: {hits:?}",
        hits.len()
    );
}

// PIN: ADR-039 §D-5 — symmetric reference: `commit_pending` publishes
// the buffered doc.
#[test]
fn commit_pending_publishes_doc() {
    let (_tmp, svc) = fresh_service();
    let h = svc.handle(TENANT, IndexId::DEFAULT_BM25).expect("handle");
    h.upsert_document(NodeId::new(1), "publish_me", Lsn::new(5))
        .expect("buffered upsert");

    let trait_obj: Arc<dyn Bm25IndexStoreHandle> = svc.clone();
    trait_obj
        .commit_pending(TENANT)
        .expect("commit_pending must succeed for an opened tenant");

    let hits = h
        .search("publish_me", 10, Lsn::new(100))
        .expect("search post-commit");
    assert_eq!(
        hits.len(),
        1,
        "PIN: ADR-039 §D-5 — `commit_pending` must publish the \
         buffered upsert; got {} hits",
        hits.len()
    );
    assert_eq!(
        hits[0].0,
        NodeId::new(1),
        "PIN: ADR-039 §D-5 — node_id must round-trip through commit"
    );
}

// PIN: ADR-039 §D-6 — after a rollback, the writer is in a clean state.
// A fresh upsert + commit publishes ONLY the second doc; the
// rolled-back first doc is NOT visible.
#[test]
fn rollback_then_upsert_then_commit_publishes_only_second_doc() {
    let (_tmp, svc) = fresh_service();
    let h = svc.handle(TENANT, IndexId::DEFAULT_BM25).expect("handle");
    let trait_obj: Arc<dyn Bm25IndexStoreHandle> = svc.clone();

    // 1. Buffer first doc.
    h.upsert_document(NodeId::new(1), "first_doc_unique", Lsn::new(1))
        .expect("buffered upsert 1");

    // 2. Rollback discards it.
    trait_obj
        .rollback_pending(TENANT)
        .expect("rollback first doc");

    // 3. Buffer second doc.
    h.upsert_document(NodeId::new(2), "second_doc_unique", Lsn::new(2))
        .expect("buffered upsert 2");

    // 4. Commit publishes the second doc only.
    trait_obj.commit_pending(TENANT).expect("commit second doc");

    let hits_first = h
        .search("first_doc_unique", 10, Lsn::new(100))
        .expect("search first");
    assert!(
        hits_first.is_empty(),
        "PIN: ADR-039 §D-6 — rolled-back first doc MUST NOT be visible \
         after subsequent commit (got {} hits)",
        hits_first.len()
    );

    let hits_second = h
        .search("second_doc_unique", 10, Lsn::new(100))
        .expect("search second");
    assert_eq!(
        hits_second.len(),
        1,
        "PIN: ADR-039 §D-6 — second (post-rollback) doc MUST be \
         visible after commit (got {} hits)",
        hits_second.len()
    );
    assert_eq!(
        hits_second[0].0,
        NodeId::new(2),
        "PIN: ADR-039 §D-6 — only the second doc's node_id round-trips"
    );
}

// PIN: ADR-039 §D-7 + Bm25StoreError taxonomy — `rollback_pending` for
// a never-opened tenant surfaces `Bm25StoreError::TenantNotFound`
// carrying the raw u64. The Z-1 (b) closure in
// `arcgraph-storage::crud` warns-and-skips on this; pinning the
// surface keeps that posture stable across patches.
#[test]
fn rollback_pending_for_unknown_tenant_returns_tenant_not_found() {
    let (_tmp, svc) = fresh_service();
    let trait_obj: Arc<dyn Bm25IndexStoreHandle> = svc.clone();

    // A tenant id that has never been touched (no `handle(...)` call).
    let unknown = TenantId::new(9999);
    let err = trait_obj
        .rollback_pending(unknown)
        .expect_err("PIN: rollback_pending for unopened tenant must error");

    match err {
        Bm25StoreError::TenantNotFound { tenant_raw } => {
            assert_eq!(
                tenant_raw, 9999,
                "PIN: ADR-039 §D-7 — TenantNotFound must carry the raw \
                 u64 from the caller's TenantId (got {tenant_raw})"
            );
        }
        other => panic!("PIN: ADR-039 §D-7 — expected TenantNotFound, got {other:?}"),
    }
}

// PIN: ADR-039 §D-7 — symmetric for the commit arm.
#[test]
fn commit_pending_for_unknown_tenant_returns_tenant_not_found() {
    let (_tmp, svc) = fresh_service();
    let trait_obj: Arc<dyn Bm25IndexStoreHandle> = svc.clone();

    let unknown = TenantId::new(7777);
    let err = trait_obj
        .commit_pending(unknown)
        .expect_err("PIN: commit_pending for unopened tenant must error");

    match err {
        Bm25StoreError::TenantNotFound { tenant_raw } => {
            assert_eq!(
                tenant_raw, 7777,
                "PIN: ADR-039 §D-7 — TenantNotFound carries raw u64 \
                 (got {tenant_raw})"
            );
        }
        other => panic!("PIN: ADR-039 §D-7 — expected TenantNotFound, got {other:?}"),
    }
}

// PIN: ADR-039 §D-7 — the trait shape must be object-safe AND
// constructible from `Arc<Bm25Service>`. This is a compile-time +
// runtime pin: if the trait grew a generic method (or a non-Self
// receiver type) without an amendment, this construction would fail
// to compile.
#[test]
fn trait_object_dispatch_works() {
    let (_tmp, svc) = fresh_service();
    // The explicit type ascription is the load-bearing assertion —
    // it pins that `Bm25Service: Bm25IndexStoreHandle` AND that the
    // trait object is constructible from `Arc<Bm25Service>`.
    let trait_obj: Arc<dyn Bm25IndexStoreHandle> = svc.clone();

    // Open the tenant so dispatch returns Ok rather than
    // TenantNotFound — proves the trait object is reachable on the
    // commit path (not just on the error path).
    let _h = svc.handle(TENANT, IndexId::DEFAULT_BM25).expect("handle");
    trait_obj
        .commit_pending(TENANT)
        .expect("PIN: trait-object commit_pending dispatch reachable");
    trait_obj
        .rollback_pending(TENANT)
        .expect("PIN: trait-object rollback_pending dispatch reachable");
}
