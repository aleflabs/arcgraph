//! Path-A boundary tests for ADR-039 §D-8 + ADR-035 amendment-04 §D-3:
//! `Filter::Any` is the unfiltered identity; `Filter::Tenant(_)` surfaces
//! `Bm25Error::FilterNotSupported` per the v1.0 escalation contract.
//!
//! PINS:
//! - `filter_any_matches_search_result_set` — pins
//!   `filtered_search(_, _, &Filter::Any, _) == search(_, _, _)` for
//!   both result identity AND ordering on a deterministic fixture.
//! - `filter_tenant_returns_filter_not_supported` — pins that any
//!   `Filter::Tenant(_)` variant surfaces `FilterNotSupported` with the
//!   variant string carrying `"Tenant"` so log readers can grep for
//!   the failure shape.
//! - `filter_not_supported_does_not_advance_writer_state` — pins that
//!   a `Filter::Tenant` rejection does NOT poison the per-tenant
//!   writer; subsequent upsert + commit_pending + search continues to
//!   work correctly.
//!
//! Failure of any pin is a *contract* break, not a test bug.

use std::sync::Arc;

use arcgraph_bm25::{Bm25Error, Bm25Service, Filter, IndexId};
use arcgraph_core::{Lsn, NodeId, TenantId};
use arcgraph_storage::mutation_log::Bm25IndexStoreHandle;
use tempfile::TempDir;

fn fresh_service() -> (TempDir, Arc<Bm25Service>) {
    let tmp = TempDir::new().expect("tempdir");
    let svc = Bm25Service::new(tmp.path().to_path_buf());
    (tmp, svc)
}

/// Seed a 3-doc fixture under DEFAULT and commit through the trait
/// object so the reader observes the docs.
fn seed_fixture(svc: &Arc<Bm25Service>) {
    let h = svc
        .handle(TenantId::DEFAULT, IndexId::DEFAULT_BM25)
        .expect("handle");
    h.upsert_document(NodeId::new(10), "alpha quick brown fox", Lsn::new(1))
        .expect("upsert 10");
    h.upsert_document(NodeId::new(20), "alpha lazy dog", Lsn::new(2))
        .expect("upsert 20");
    h.upsert_document(NodeId::new(30), "alpha hops over fences", Lsn::new(3))
        .expect("upsert 30");
    let trait_obj: Arc<dyn Bm25IndexStoreHandle> = svc.clone();
    trait_obj
        .commit_pending(TenantId::DEFAULT)
        .expect("commit_pending");
}

// PIN: ADR-039 §D-8 — `Filter::Any` is the F.4 dispatcher's identity:
// `filtered_search(q, k, &Filter::Any, lsn) == search(q, k, lsn)` for
// both the result set AND the ordering.
#[test]
fn filter_any_matches_search_result_set() {
    let (_tmp, svc) = fresh_service();
    seed_fixture(&svc);
    let h = svc
        .handle(TenantId::DEFAULT, IndexId::DEFAULT_BM25)
        .expect("handle");

    let plain = h.search("alpha", 10, Lsn::new(100)).expect("plain search");
    let filtered = h
        .filtered_search("alpha", 10, &Filter::Any, Lsn::new(100))
        .expect("filtered_search with Filter::Any");

    assert_eq!(
        plain.len(),
        filtered.len(),
        "PIN: ADR-039 §D-8 — Filter::Any must produce the same number \
         of hits as plain search (got plain={} filtered={})",
        plain.len(),
        filtered.len(),
    );
    assert!(
        !plain.is_empty(),
        "PIN: sanity — fixture must yield ≥ 1 hit for 'alpha'"
    );

    // Identity on (node_id, score) ordering. Both calls share the same
    // searcher snapshot, so scores are bit-identical.
    let plain_ids: Vec<NodeId> = plain.iter().map(|(n, _)| *n).collect();
    let filtered_ids: Vec<NodeId> = filtered.iter().map(|(n, _)| *n).collect();
    assert_eq!(
        plain_ids, filtered_ids,
        "PIN: ADR-039 §D-8 — Filter::Any must preserve search rank \
         order (got plain={plain_ids:?} filtered={filtered_ids:?})"
    );

    for (i, ((p_id, p_score), (f_id, f_score))) in plain.iter().zip(filtered.iter()).enumerate() {
        assert_eq!(
            p_id, f_id,
            "PIN: ADR-039 §D-8 — node_id at rank {i} must match across \
             search variants"
        );
        assert!(
            (p_score - f_score).abs() < f32::EPSILON,
            "PIN: ADR-039 §D-8 — score at rank {i} must match (got \
             plain={p_score} filtered={f_score})"
        );
    }
}

// PIN: ADR-035 amendment-04 §D-3 — `Filter::Tenant(_)` is NOT supported
// by BM25 at v1.0 (no tenant FAST field in the schema; tenant isolation
// is enforced by per-tenant directory layout instead). Any
// `Filter::Tenant(_)` call MUST surface
// `Bm25Error::FilterNotSupported { variant }` with `variant`
// containing the substring "Tenant" (the production code uses
// `format!("{:?}", filter)` so the rendered variant always names
// itself).
#[test]
fn filter_tenant_returns_filter_not_supported() {
    let (_tmp, svc) = fresh_service();
    let h = svc
        .handle(TenantId::DEFAULT, IndexId::DEFAULT_BM25)
        .expect("handle");

    let err = h
        .filtered_search(
            "anything",
            10,
            &Filter::Tenant(TenantId::DEFAULT),
            Lsn::new(100),
        )
        .expect_err(
            "PIN: ADR-035 amendment-04 §D-3 — Filter::Tenant must surface FilterNotSupported",
        );

    match err {
        Bm25Error::FilterNotSupported { variant } => {
            assert!(
                variant.contains("Tenant"),
                "PIN: ADR-039 §D-8 — FilterNotSupported.variant must \
                 carry the rendered variant name 'Tenant', got {variant:?}"
            );
        }
        other => panic!(
            "PIN: ADR-035 amendment-04 §D-3 — expected FilterNotSupported, \
             got {other:?}"
        ),
    }

    // Symmetric: a non-DEFAULT tenant id MUST also fail. This pins
    // that the rejection is on the variant shape, not on the inner
    // tenant id value.
    let err2 = h
        .filtered_search(
            "anything",
            10,
            &Filter::Tenant(TenantId::new(42)),
            Lsn::new(100),
        )
        .expect_err("Filter::Tenant(42) must also surface FilterNotSupported");
    assert!(
        matches!(err2, Bm25Error::FilterNotSupported { .. }),
        "PIN: ADR-035 amendment-04 §D-3 — every Filter::Tenant(_) variant \
         shape rejects identically, got {err2:?}"
    );
}

// PIN: ADR-039 §D-8 — a search-side `Filter::Tenant` rejection MUST NOT
// advance writer state. Pin: after the rejected `filtered_search`, the
// per-tenant `IndexWriter` continues to function correctly — a fresh
// upsert + `commit_pending` + search round-trips the doc.
#[test]
fn filter_not_supported_does_not_advance_writer_state() {
    let (_tmp, svc) = fresh_service();
    let h = svc
        .handle(TenantId::DEFAULT, IndexId::DEFAULT_BM25)
        .expect("handle");

    // 1. Upsert a doc but DO NOT commit yet.
    h.upsert_document(NodeId::new(7), "pre-rejection body", Lsn::new(1))
        .expect("upsert before rejection");

    // 2. Trigger a Filter::Tenant rejection. This must NOT touch the
    //    per-tenant writer's buffer (the rejection happens inside the
    //    search-side dispatch, before any writer work).
    let _ = h
        .filtered_search(
            "anything",
            10,
            &Filter::Tenant(TenantId::DEFAULT),
            Lsn::new(100),
        )
        .expect_err("rejection (expected)");

    // 3. Commit + search. The pre-rejection upsert must still be
    //    visible — proves the writer's buffer was not poisoned.
    let trait_obj: Arc<dyn Bm25IndexStoreHandle> = svc.clone();
    trait_obj
        .commit_pending(TenantId::DEFAULT)
        .expect("commit_pending after Filter::Tenant rejection");

    let hits = h
        .search("pre-rejection", 10, Lsn::new(100))
        .expect("search after the rejection-then-commit cycle");
    assert_eq!(
        hits.len(),
        1,
        "PIN: ADR-039 §D-8 — Filter::Tenant rejection MUST NOT \
         advance/poison the writer; the pre-rejection upsert must \
         still be visible after commit, got {} hits",
        hits.len()
    );
    assert_eq!(
        hits[0].0,
        NodeId::new(7),
        "PIN: ADR-039 §D-8 — the round-tripped node_id must be the \
         pre-rejection one"
    );
}
