//! Path-A boundary tests for ADR-039 §D-3: MVCC visibility windowing
//! via the `commit_lsn ≤ read_lsn AND read_lsn < expired_lsn` filter.
//!
//! At v1.0 every doc carries `expired_lsn = u64::MAX` (structural pin
//! per ADR-039 §D-2); the second clause is therefore trivially true,
//! and the test surface focuses on the `commit_lsn ≤ read_lsn` half.
//!
//! PINS:
//! - `reader_at_older_lsn_excludes_post_lsn_doc` — a doc with
//!   `commit_lsn = 10` MUST NOT be visible at `read_lsn = 5`.
//! - `reader_at_newer_lsn_includes_doc` — same doc, visible at
//!   `read_lsn = 15`.
//! - `reader_at_exact_commit_lsn_includes_doc` — boundary case:
//!   `commit_lsn ≤ read_lsn` is INCLUSIVE on the commit_lsn side.
//! - `expired_lsn_is_structurally_max_at_v1` — pins that v1.0-produced
//!   docs are visible at `read_lsn = u64::MAX`, equivalent to
//!   `read_lsn = high`. (We can't inspect Tantivy doc bytes from an
//!   external test; the MAX-snapshot equivalence is the boundary
//!   surface that pins the structural invariant.)
//!
//! Failure of any pin is a *contract* break, not a test bug.

use std::sync::Arc;

use arcgraph_bm25::{Bm25Service, IndexId};
use arcgraph_core::{Lsn, NodeId, TenantId};
use arcgraph_storage::mutation_log::Bm25IndexStoreHandle;
use tempfile::TempDir;

fn fresh_service() -> (TempDir, Arc<Bm25Service>) {
    let tmp = TempDir::new().expect("tempdir");
    let svc = Bm25Service::new(tmp.path().to_path_buf());
    (tmp, svc)
}

/// Insert one doc at `commit_lsn` and dispatch `commit_pending` so the
/// reader sees the segment.
fn upsert_and_commit(svc: &Arc<Bm25Service>, node_id: NodeId, body: &str, commit_lsn: Lsn) {
    let h = svc
        .handle(TenantId::DEFAULT, IndexId::DEFAULT_BM25)
        .expect("handle");
    h.upsert_document(node_id, body, commit_lsn)
        .expect("upsert");
    let trait_obj: Arc<dyn Bm25IndexStoreHandle> = svc.clone();
    trait_obj
        .commit_pending(TenantId::DEFAULT)
        .expect("commit_pending");
}

// PIN: ADR-039 §D-3 — `commit_lsn ≤ read_lsn` is the lower-bound
// clause. A doc at `commit_lsn = 10` MUST NOT surface to a reader at
// `read_lsn = 5`.
#[test]
fn reader_at_older_lsn_excludes_post_lsn_doc() {
    let (_tmp, svc) = fresh_service();
    upsert_and_commit(&svc, NodeId::new(1), "future body", Lsn::new(10));

    let h = svc
        .handle(TenantId::DEFAULT, IndexId::DEFAULT_BM25)
        .expect("handle");
    let hits = h
        .search("future", 10, Lsn::new(5))
        .expect("search at stale LSN");
    assert!(
        hits.is_empty(),
        "PIN: ADR-039 §D-3 — doc with commit_lsn=10 MUST NOT be \
         visible at read_lsn=5 (got {} hits: {hits:?})",
        hits.len()
    );
}

// PIN: ADR-039 §D-3 — a reader at `read_lsn > commit_lsn` MUST see the
// doc. v1.0 `expired_lsn = MAX` makes the upper-bound clause trivially
// true.
#[test]
fn reader_at_newer_lsn_includes_doc() {
    let (_tmp, svc) = fresh_service();
    upsert_and_commit(&svc, NodeId::new(1), "future body", Lsn::new(10));

    let h = svc
        .handle(TenantId::DEFAULT, IndexId::DEFAULT_BM25)
        .expect("handle");
    let hits = h
        .search("future", 10, Lsn::new(15))
        .expect("search at fresh LSN");
    assert_eq!(
        hits.len(),
        1,
        "PIN: ADR-039 §D-3 — doc with commit_lsn=10 MUST be visible at \
         read_lsn=15 (got {} hits)",
        hits.len()
    );
    assert_eq!(
        hits[0].0,
        NodeId::new(1),
        "PIN: ADR-039 §D-3 — round-tripped node_id must be 1"
    );
}

// PIN: ADR-039 §D-3 — `commit_lsn ≤ read_lsn` is INCLUSIVE on the
// commit_lsn side. A reader at `read_lsn == commit_lsn` MUST see the
// doc. (Implementation: `RangeQuery::new(Bound::Included(0),
// Bound::Included(read_lsn))` per `crates/arcgraph-bm25/src/mvcc.rs`.)
#[test]
fn reader_at_exact_commit_lsn_includes_doc() {
    let (_tmp, svc) = fresh_service();
    upsert_and_commit(&svc, NodeId::new(1), "boundary body", Lsn::new(10));

    let h = svc
        .handle(TenantId::DEFAULT, IndexId::DEFAULT_BM25)
        .expect("handle");
    let hits = h
        .search("boundary", 10, Lsn::new(10))
        .expect("search at exact commit_lsn");
    assert_eq!(
        hits.len(),
        1,
        "PIN: ADR-039 §D-3 — boundary `read_lsn == commit_lsn` is \
         INCLUSIVE; doc must be visible (got {} hits)",
        hits.len()
    );
    assert_eq!(
        hits[0].0,
        NodeId::new(1),
        "PIN: ADR-039 §D-3 — node_id must round-trip at the inclusive \
         boundary"
    );
}

// PIN: ADR-039 §D-2 + §D-3 — every v1.0-produced doc carries
// `expired_lsn = u64::MAX`. We cannot inspect Tantivy doc bytes from
// an external integration test, so we pin the structural invariant
// via its observable consequence: a search at `read_lsn = u64::MAX`
// returns the SAME doc set as a search at any `read_lsn > commit_lsn`.
// If a future write path stamped `expired_lsn < MAX`, the MAX-LSN
// reader would lose visibility (because `expired_lsn > read_lsn` would
// be false at `read_lsn = u64::MAX - 1` already, and at MAX itself
// the saturating-add upper clause makes the test stable).
#[test]
fn expired_lsn_is_structurally_max_at_v1() {
    let (_tmp, svc) = fresh_service();
    upsert_and_commit(&svc, NodeId::new(1), "structural body", Lsn::new(10));

    let h = svc
        .handle(TenantId::DEFAULT, IndexId::DEFAULT_BM25)
        .expect("handle");

    // Reference: a "high" read snapshot well above commit_lsn.
    let hits_high = h
        .search("structural", 10, Lsn::new(1_000_000))
        .expect("search at high LSN");
    assert_eq!(
        hits_high.len(),
        1,
        "PIN: sanity — high LSN reader must see the doc"
    );

    // Pin: same result at a near-MAX snapshot. If `expired_lsn` were
    // ever stamped below `u64::MAX`, this read would lose visibility
    // because the v1.0 visibility filter requires
    // `expired_lsn > read_lsn`. The saturating_add in
    // `build_visibility_filter` makes the MAX-MAX boundary stable.
    let hits_max = h
        .search("structural", 10, Lsn::new(u64::MAX - 1))
        .expect("search at near-MAX LSN");
    assert_eq!(
        hits_max.len(),
        hits_high.len(),
        "PIN: ADR-039 §D-2 — v1.0 docs have expired_lsn = u64::MAX; a \
         near-MAX reader must see the SAME doc set as a 'high' reader \
         (got near-MAX={} high={})",
        hits_max.len(),
        hits_high.len(),
    );
    assert_eq!(
        hits_max[0].0, hits_high[0].0,
        "PIN: ADR-039 §D-2 — node_id must round-trip identically at \
         both LSN snapshots"
    );
}
