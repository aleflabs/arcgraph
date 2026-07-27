//! Boundary tests for the local BM25 handle: `Bm25IndexHandle` is keyed by
//! `(TenantId, PartitionId, IndexId)`; v1.0 enforces
//! `partition_id == PartitionId::ZERO` and
//! `index_id == IndexId::DEFAULT_BM25`.
//!
//! PINS:
//! - `handle_partition_id_is_zero_at_v1` — every handle returned by
//!   `Bm25Service::handle(...)` reports `partition() == ZERO`.
//! - `handle_index_id_is_default_bm25_at_v1` — every handle reports
//!   `index() == DEFAULT_BM25`.
//! - `handle_tenant_id_round_trips` — handle reports back the tenant
//!   it was constructed for.
//! - `schema_field_count_pinned` — v1.0 schema has exactly 4 fields
//!   (`node_id`, `commit_lsn`, `expired_lsn`, `body`). v1.1 additions
//!   surface here as a deliberate test update.
//! - `tantivy_index_directory_layout_pinned` —
//!   `<data_dir>/bm25/<tenant_id>/<index_id>/` is the structural path.
//!
//! Failure of any pin is a *contract* break, not a test bug.

use std::sync::Arc;

use arcgraph_bm25::{Bm25Service, IndexId};
use arcgraph_core::{PartitionId, TenantId};
use tempfile::TempDir;

fn fresh_service() -> (TempDir, Arc<Bm25Service>) {
    let tmp = TempDir::new().expect("tempdir");
    let svc = Bm25Service::new(tmp.path().to_path_buf());
    (tmp, svc)
}

// PIN: every public BM25 handle is bound to `PartitionId::ZERO`.
#[test]
fn handle_partition_id_is_zero_at_v1() {
    let (_tmp, svc) = fresh_service();
    // Try multiple tenants — partition stays ZERO regardless.
    for raw in [TenantId::DEFAULT.raw(), 101, 202, 303] {
        let h = svc
            .handle(TenantId::new(raw), IndexId::DEFAULT_BM25)
            .expect("handle");
        assert_eq!(
            h.partition(),
            PartitionId::ZERO,
            "PIN: ADR-039 §D-4 + Q2 — handle for tenant {raw} must be \
             bound to PartitionId::ZERO at v1.0 (got {:?})",
            h.partition()
        );
    }
}

// PIN: ADR-039 §D-4 — every v1.0 BM25 handle is bound to
// `IndexId::DEFAULT_BM25`. Per-property indexes are M7 / v1.1 scope;
// this pin is load-bearing for that future lift.
#[test]
fn handle_index_id_is_default_bm25_at_v1() {
    let (_tmp, svc) = fresh_service();
    for raw in [TenantId::DEFAULT.raw(), 101, 202] {
        let h = svc
            .handle(TenantId::new(raw), IndexId::DEFAULT_BM25)
            .expect("handle");
        assert_eq!(
            h.index(),
            IndexId::DEFAULT_BM25,
            "PIN: ADR-039 §D-4 — handle for tenant {raw} must be bound \
             to IndexId::DEFAULT_BM25 at v1.0 (got {:?})",
            h.index()
        );
    }

    // Pin: DEFAULT_BM25 is structurally `IndexId(0)`. v1.1
    // renumbering would break the on-disk directory layout.
    assert_eq!(
        IndexId::DEFAULT_BM25.raw(),
        0,
        "PIN: ADR-039 §D-4 — IndexId::DEFAULT_BM25 must be numeric 0 \
         at v1.0 (got {})",
        IndexId::DEFAULT_BM25.raw()
    );
    assert_eq!(
        IndexId::DEFAULT_BM25,
        IndexId::ZERO,
        "PIN: ADR-039 §D-4 — DEFAULT_BM25 == ZERO is the v1.0 alias"
    );
}

// PIN: ADR-039 §D-8 — the tenant id passed to `handle(tenant, ...)`
// is round-tripped via `handle.tenant()`. This is the tenant-side
// half of the `(TenantId, PartitionId, IndexId)` key shape.
#[test]
fn handle_tenant_id_round_trips() {
    let (_tmp, svc) = fresh_service();
    let tid = TenantId::new(42);
    let h = svc.handle(tid, IndexId::DEFAULT_BM25).expect("handle");
    assert_eq!(
        h.tenant(),
        tid,
        "PIN: ADR-039 §D-8 — handle.tenant() must equal the tenant id \
         it was constructed for (got {:?})",
        h.tenant()
    );

    // And distinct tenants produce distinct reported tenant ids.
    let other = TenantId::new(43);
    let h_other = svc
        .handle(other, IndexId::DEFAULT_BM25)
        .expect("other handle");
    assert_ne!(
        h.tenant(),
        h_other.tenant(),
        "PIN: ADR-039 §D-8 — distinct tenant inputs yield distinct \
         handle.tenant() outputs"
    );
    assert_eq!(h_other.tenant(), other);
}

// PIN: ADR-039 §D-2 — the v1.0 schema has exactly 4 fields:
// `node_id`, `commit_lsn`, `expired_lsn`, `body`. v1.1 additions
// (e.g., per-property field) surface here as a deliberate test
// update + ADR amendment.
#[test]
fn schema_field_count_pinned() {
    let (_tmp, svc) = fresh_service();
    let schema = svc.schema();
    let count = schema.schema.fields().count();
    assert_eq!(
        count, 4,
        "PIN: ADR-039 §D-2 — v1.0 schema MUST have exactly 4 fields \
         (node_id, commit_lsn, expired_lsn, body); got {count}"
    );

    // Pin field-name set so a v1.1 lift surfaces here even if the
    // count happens to be preserved by a swap.
    let names: Vec<String> = schema
        .schema
        .fields()
        .map(|(_, entry)| entry.name().to_string())
        .collect();
    for expected in ["node_id", "commit_lsn", "expired_lsn", "body"] {
        assert!(
            names.iter().any(|n| n == expected),
            "PIN: ADR-039 §D-2 — schema must contain field '{expected}', \
             got {names:?}"
        );
    }
}

// PIN: ADR-039 §D-4 — `<data_dir>/bm25/<tenant_id>/<index_id>/` is
// the structural directory layout. We verify by inspecting the path
// components of `tenant_index_dir` and by checking the on-disk
// existence after first-touch.
#[test]
fn tantivy_index_directory_layout_pinned() {
    let (tmp, svc) = fresh_service();
    let tid = TenantId::new(101);
    let _h = svc
        .handle(tid, IndexId::DEFAULT_BM25)
        .expect("first-touch handle");

    let dir = svc.tenant_index_dir(tid, IndexId::DEFAULT_BM25);

    // Pin: the directory exists on disk after first-touch.
    assert!(
        dir.is_dir(),
        "PIN: ADR-039 §D-4 — tenant index directory must exist on disk \
         after first-touch: {dir:?}"
    );

    // Pin: the path layout components are
    // [..., "bm25", "<tenant_raw>", "<index_raw>"]. We extract the
    // last 3 components and check them in order — robust to whatever
    // tempfile prepends.
    let components: Vec<String> = dir
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    assert!(
        components.len() >= 3,
        "PIN: ADR-039 §D-4 — directory path must have ≥ 3 components, \
         got {components:?}"
    );
    let last3: &[String] = &components[components.len() - 3..];
    assert_eq!(
        last3[0], "bm25",
        "PIN: ADR-039 §D-4 — path layout is `<data_dir>/bm25/...`; \
         got '{}' at position -3 (full: {components:?})",
        last3[0]
    );
    assert_eq!(
        last3[1],
        tid.raw().to_string(),
        "PIN: ADR-039 §D-4 — second-to-last component must be the \
         raw tenant id; got '{}' (full: {components:?})",
        last3[1]
    );
    assert_eq!(
        last3[2],
        IndexId::DEFAULT_BM25.raw().to_string(),
        "PIN: ADR-039 §D-4 — last component must be the raw index id; \
         got '{}' (full: {components:?})",
        last3[2]
    );

    // Sanity: the bm25 root is under the test's tempdir (no escape).
    assert!(
        dir.starts_with(tmp.path()),
        "PIN: directory must live under the test tempdir; got {dir:?} \
         not under {:?}",
        tmp.path()
    );
}
