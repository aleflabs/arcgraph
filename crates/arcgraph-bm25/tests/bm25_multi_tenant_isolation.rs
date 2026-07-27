//! Per-tenant Tantivy directory-isolation boundary tests.
//!
//! Tantivy segments live under `<data_dir>/bm25/<tenant_id>/<index_id>/`
//! and each tenant has its own `Index` / `IndexWriter` / `IndexReader`.
//! There is NO shared term dictionary, NO cross-tenant cache.
//!
//! PINS:
//! - `tenant_a_doc_not_visible_from_tenant_b_handle` — A's docs do not
//!   leak into B's search.
//! - `tenant_b_doc_not_visible_from_tenant_a_handle` — symmetric (no
//!   leakage in either direction).
//! - `per_tenant_handle_caches_independently` — `handle(A,...)` shares
//!   an `Arc` with itself but NOT with `handle(B,...)` (ADR-037 §D-6
//!   cache discipline at the BM25 layer).
//! - `per_tenant_directories_exist_on_disk` — both per-tenant
//!   directories materialise on first-touch under `<data_dir>/bm25/`.
//!
//! Failure of any pin is a *contract* break, not a test bug.

use std::sync::Arc;

use arcgraph_bm25::{Bm25Service, IndexId};
use arcgraph_core::{Lsn, NodeId, TenantId};
use arcgraph_storage::mutation_log::Bm25IndexStoreHandle;
use tempfile::TempDir;

/// Two distinct tenant ids in the user-DDL range so there's no
/// reserved-tenant interference.
const TENANT_A: TenantId = TenantId::new(101);
const TENANT_B: TenantId = TenantId::new(202);

fn fresh_service() -> (TempDir, Arc<Bm25Service>) {
    let tmp = TempDir::new().expect("tempdir");
    let svc = Bm25Service::new(tmp.path().to_path_buf());
    (tmp, svc)
}

// PIN: ADR-039 §D-4 + Q3 — Tenant A's doc MUST NOT be visible from
// Tenant B's handle. Per-tenant directories under `<data_dir>/bm25/`
// are the sole isolation primitive at v1.0.
#[test]
fn tenant_a_doc_not_visible_from_tenant_b_handle() {
    let (_tmp, svc) = fresh_service();

    // Tenant A: upsert "alpha sentinel" + commit_pending(A).
    let h_a = svc
        .handle(TENANT_A, IndexId::DEFAULT_BM25)
        .expect("handle A");
    h_a.upsert_document(NodeId::new(1), "alpha sentinel", Lsn::new(1))
        .expect("upsert A");
    let trait_obj: Arc<dyn Bm25IndexStoreHandle> = svc.clone();
    trait_obj.commit_pending(TENANT_A).expect("commit A");

    // Tenant B: open handle, search "sentinel" — must be empty.
    let h_b = svc
        .handle(TENANT_B, IndexId::DEFAULT_BM25)
        .expect("handle B");
    let hits = h_b
        .search("sentinel", 10, Lsn::new(100))
        .expect("search on B");
    assert!(
        hits.is_empty(),
        "PIN: ADR-039 §D-4 — Tenant A's doc MUST NOT be visible from \
         Tenant B's handle (got {} hits: {hits:?})",
        hits.len()
    );

    // Sanity: A's own search DOES see the doc (proves the upsert
    // landed; the empty B result is real isolation, not a test bug).
    let hits_a = h_a
        .search("sentinel", 10, Lsn::new(100))
        .expect("search on A");
    assert_eq!(
        hits_a.len(),
        1,
        "PIN: sanity — Tenant A's own search must see Tenant A's doc"
    );
    assert_eq!(hits_a[0].0, NodeId::new(1));
}

// PIN: ADR-039 §D-4 + Q3 — symmetric. Tenant B's doc must NOT leak
// into Tenant A's search.
#[test]
fn tenant_b_doc_not_visible_from_tenant_a_handle() {
    let (_tmp, svc) = fresh_service();

    let h_b = svc
        .handle(TENANT_B, IndexId::DEFAULT_BM25)
        .expect("handle B");
    h_b.upsert_document(NodeId::new(2), "beta marker", Lsn::new(1))
        .expect("upsert B");
    let trait_obj: Arc<dyn Bm25IndexStoreHandle> = svc.clone();
    trait_obj.commit_pending(TENANT_B).expect("commit B");

    let h_a = svc
        .handle(TENANT_A, IndexId::DEFAULT_BM25)
        .expect("handle A");
    let hits = h_a
        .search("marker", 10, Lsn::new(100))
        .expect("search on A");
    assert!(
        hits.is_empty(),
        "PIN: ADR-039 §D-4 — Tenant B's doc MUST NOT be visible from \
         Tenant A's handle (got {} hits: {hits:?})",
        hits.len()
    );

    // Sanity: B's own search sees the doc.
    let hits_b = h_b
        .search("marker", 10, Lsn::new(100))
        .expect("search on B");
    assert_eq!(
        hits_b.len(),
        1,
        "PIN: sanity — Tenant B's own search must see Tenant B's doc"
    );
}

// PIN: ADR-037 §D-6 — append-only handle cache at the BM25 layer.
// `handle(A, _)` repeatedly returns the SAME `Arc`; `handle(B, _)`
// returns a DIFFERENT `Arc` (each tenant has its own underlying
// Tantivy index).
#[test]
fn per_tenant_handle_caches_independently() {
    let (_tmp, svc) = fresh_service();
    let a1 = svc
        .handle(TENANT_A, IndexId::DEFAULT_BM25)
        .expect("handle A first");
    let a2 = svc
        .handle(TENANT_A, IndexId::DEFAULT_BM25)
        .expect("handle A second");
    let b1 = svc
        .handle(TENANT_B, IndexId::DEFAULT_BM25)
        .expect("handle B first");

    assert!(
        Arc::ptr_eq(&a1, &a2),
        "PIN: ADR-037 §D-6 — repeated handle(A, _) calls must return \
         the SAME Arc (append-only cache hit)"
    );
    assert!(
        !Arc::ptr_eq(&a1, &b1),
        "PIN: ADR-037 §D-6 — handle(A, _) and handle(B, _) MUST be \
         different Arcs (per-tenant cache discipline)"
    );

    // Re-open B; same Arc as b1.
    let b2 = svc
        .handle(TENANT_B, IndexId::DEFAULT_BM25)
        .expect("handle B second");
    assert!(
        Arc::ptr_eq(&b1, &b2),
        "PIN: ADR-037 §D-6 — repeated handle(B, _) calls share Arc"
    );
}

// PIN: ADR-039 §D-4 — first-touch lazily creates
// `<data_dir>/bm25/<tenant>/<index>/`. Both tenant directories must
// exist on disk after their first `handle(...)` call.
#[test]
fn per_tenant_directories_exist_on_disk() {
    let (tmp, svc) = fresh_service();
    let _h_a = svc
        .handle(TENANT_A, IndexId::DEFAULT_BM25)
        .expect("first-touch A");
    let _h_b = svc
        .handle(TENANT_B, IndexId::DEFAULT_BM25)
        .expect("first-touch B");

    let dir_a = svc.tenant_index_dir(TENANT_A, IndexId::DEFAULT_BM25);
    let dir_b = svc.tenant_index_dir(TENANT_B, IndexId::DEFAULT_BM25);

    assert!(
        dir_a.is_dir(),
        "PIN: ADR-039 §D-4 — Tenant A's directory must exist as a \
         directory at {dir_a:?}"
    );
    assert!(
        dir_b.is_dir(),
        "PIN: ADR-039 §D-4 — Tenant B's directory must exist as a \
         directory at {dir_b:?}"
    );

    // Pin the path layout: both directories must live under
    // `<data_dir>/bm25/`. Compose the path manually to assert the
    // structural shape independent of `tenant_index_dir`'s internals.
    let bm25_root = tmp.path().join("bm25");
    let manual_a = bm25_root
        .join(TENANT_A.raw().to_string())
        .join(IndexId::DEFAULT_BM25.raw().to_string());
    let manual_b = bm25_root
        .join(TENANT_B.raw().to_string())
        .join(IndexId::DEFAULT_BM25.raw().to_string());
    assert_eq!(
        dir_a, manual_a,
        "PIN: ADR-039 §D-4 — Tenant A directory layout is \
         <data_dir>/bm25/<tenant_id>/<index_id>/"
    );
    assert_eq!(
        dir_b, manual_b,
        "PIN: ADR-039 §D-4 — Tenant B directory layout is \
         <data_dir>/bm25/<tenant_id>/<index_id>/"
    );

    // And the bm25 root itself is a directory created by the
    // `create_dir_all` in `Bm25Service::handle`.
    assert!(
        bm25_root.is_dir(),
        "PIN: ADR-039 §D-4 — `<data_dir>/bm25/` exists after first \
         tenant first-touch"
    );

    // Enumerate the bm25 directory: at least the two tenant-id
    // subdirectories must be present.
    let entries: Vec<String> = std::fs::read_dir(&bm25_root)
        .expect("read_dir bm25 root")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    assert!(
        entries.contains(&TENANT_A.raw().to_string()),
        "PIN: ADR-039 §D-4 — Tenant A's subdirectory must be \
         enumerable under bm25 root, got {entries:?}"
    );
    assert!(
        entries.contains(&TENANT_B.raw().to_string()),
        "PIN: ADR-039 §D-4 — Tenant B's subdirectory must be \
         enumerable under bm25 root, got {entries:?}"
    );
}
