//! Path-A boundary tests for ADR-039 §D-8 + design-v2 §M3.b exit:
//! top-K BM25 ranking on a small, deterministic fixture.
//!
//! PINS:
//! - `top_k_returns_matching_doc_at_rank_0` — pins that a unique-keyword
//!   match surfaces at index 0 with score > 0.
//! - `top_k_excludes_non_matching_docs` — pins that docs without the
//!   queried term are absent from the result set.
//! - `top_k_respects_k_limit` — pins that `search(_, k=2, _)` returns
//!   at most 2 hits even with > 2 matching docs.
//! - `empty_query_returns_no_results_or_parse_error` — pins the v1.0
//!   actual behaviour for empty queries (Tantivy parses `""` into a
//!   trivial query that matches nothing).
//! - `search_before_commit_pending_returns_empty` — pins that a freshly
//!   buffered upsert is invisible until `commit_pending(tenant)` fires;
//!   the reader is on `ReloadPolicy::Manual` and snapshots only durably-
//!   committed segments.
//!
//! Failure of any pin is a *contract* break, not a test bug.

use std::sync::Arc;

use arcgraph_bm25::{Bm25Error, Bm25Service, IndexId};
use arcgraph_core::{Lsn, NodeId, TenantId};
use arcgraph_storage::mutation_log::Bm25IndexStoreHandle;
use tempfile::TempDir;

/// Fresh service rooted at a unique tempdir. The TempDir guard is
/// returned so its `Drop` runs after the test body.
fn fresh_service() -> (TempDir, Arc<Bm25Service>) {
    let tmp = TempDir::new().expect("tempdir");
    let svc = Bm25Service::new(tmp.path().to_path_buf());
    (tmp, svc)
}

/// Insert 5 docs into DEFAULT tenant, then dispatch
/// `commit_pending(DEFAULT)` via the trait object so the reader
/// observes the freshly committed segment. Returns the service Arc and
/// a typed handle pinned to `DEFAULT / DEFAULT_BM25`.
fn seed_five_doc_fixture() -> (TempDir, Arc<Bm25Service>) {
    let (tmp, svc) = fresh_service();
    let h = svc
        .handle(TenantId::DEFAULT, IndexId::DEFAULT_BM25)
        .expect("handle");
    // 5 docs, one with a unique keyword "needle".
    h.upsert_document(NodeId::new(1), "alpha beta gamma", Lsn::new(1))
        .expect("upsert 1");
    h.upsert_document(NodeId::new(2), "alpha delta epsilon", Lsn::new(2))
        .expect("upsert 2");
    h.upsert_document(NodeId::new(3), "alpha zeta eta", Lsn::new(3))
        .expect("upsert 3");
    h.upsert_document(NodeId::new(4), "alpha theta iota", Lsn::new(4))
        .expect("upsert 4");
    h.upsert_document(NodeId::new(5), "needle in the haystack", Lsn::new(5))
        .expect("upsert 5 (unique-keyword)");

    let trait_obj: Arc<dyn Bm25IndexStoreHandle> = svc.clone();
    trait_obj
        .commit_pending(TenantId::DEFAULT)
        .expect("commit_pending DEFAULT");
    (tmp, svc)
}

// PIN: ADR-039 §D-8 — Top-K BM25 search returns the matching doc at
// rank 0 with score > 0 when the query is a unique keyword in the
// fixture.
#[test]
fn top_k_returns_matching_doc_at_rank_0() {
    let (_tmp, svc) = seed_five_doc_fixture();
    let h = svc
        .handle(TenantId::DEFAULT, IndexId::DEFAULT_BM25)
        .expect("handle");
    let hits = h
        .search("needle", 10, Lsn::new(100))
        .expect("PIN: ADR-039 §D-8 — search must succeed on the unique-keyword fixture");
    assert!(
        !hits.is_empty(),
        "PIN: ADR-039 §D-8 — top-K must return ≥ 1 hit for a query \
         with a unique-keyword match in the fixture"
    );
    assert_eq!(
        hits[0].0,
        NodeId::new(5),
        "PIN: ADR-039 §D-8 — the unique-keyword doc (node_id=5) must \
         be at rank 0"
    );
    // O-J (W28-S3): reference-score lower bound. Was `> 0.0`, which ANY
    // positive float — even a degenerate ~1e-30 — would pass. A
    // unique-keyword match in this 5-doc corpus has IDF ≈ ln(4) ≈ 1.386
    // and a TF-saturation factor ≈ 0.9, so the BM25 score is ≈ 1.26
    // (empirically 1.2577 with the pinned tokenizer); a floor of 1.0
    // rejects degenerate/near-zero scores with ample margin.
    assert!(
        hits[0].1 > 1.0,
        "PIN: ADR-039 §D-8 — BM25 score for the unique-keyword match must \
         clear the reference floor 1.0 (got {})",
        hits[0].1
    );

    // O-J (W28-S3): two-doc relative-ordering — a doc matching MORE query
    // terms must outrank one matching fewer. For query "alpha beta",
    // node 1 ("alpha beta gamma") matches BOTH terms ("beta" is unique →
    // high IDF), while nodes 2-4 ("alpha …") match only the low-IDF
    // "alpha". So node 1 must rank 0 with a strictly higher score than
    // any alpha-only doc. (The prior `> 0.0` oracle asserted nothing
    // about relative ranking — a searcher that scored all docs equally,
    // or inverted the order, would have passed.)
    let ranked = h
        .search("alpha beta", 10, Lsn::new(100))
        .expect("PIN: ADR-039 §D-8 — multi-term search must succeed");
    assert_eq!(
        ranked[0].0,
        NodeId::new(1),
        "PIN: ADR-039 §D-8 — the doc matching both query terms (node 1) \
         must rank 0, got {ranked:?}"
    );
    for (node, score) in ranked.iter().skip(1) {
        assert!(
            *score < ranked[0].1,
            "PIN: ADR-039 §D-8 — alpha-only doc {} (score {score}) must \
             rank strictly below the two-term match node 1 (score {})",
            node.raw(),
            ranked[0].1
        );
    }
}

// PIN: ADR-039 §D-8 — Top-K excludes docs that do not contain the
// queried term. Tantivy's default tokenizer + boolean OR semantics
// must NOT pull in `alpha`-only docs when the query is `needle`.
#[test]
fn top_k_excludes_non_matching_docs() {
    let (_tmp, svc) = seed_five_doc_fixture();
    let h = svc
        .handle(TenantId::DEFAULT, IndexId::DEFAULT_BM25)
        .expect("handle");
    let hits = h
        .search("needle", 10, Lsn::new(100))
        .expect("search must succeed");
    let ids: Vec<u64> = hits.iter().map(|(n, _)| n.raw()).collect();
    // node 5 contains "needle"; nodes 1-4 do not.
    assert!(
        ids.contains(&5),
        "PIN: ADR-039 §D-8 — node 5 (the only match) must be in the \
         result set, got {ids:?}"
    );
    for excluded in [1u64, 2, 3, 4] {
        assert!(
            !ids.contains(&excluded),
            "PIN: ADR-039 §D-8 — node {excluded} does not contain \
             'needle' and MUST NOT appear in the result set: {ids:?}"
        );
    }
}

// PIN: ADR-039 §D-8 — `k` is an upper bound on the result-set size.
// Five docs all contain `alpha`; `search(_, k=2, _)` must clip to 2.
#[test]
fn top_k_respects_k_limit() {
    let (_tmp, svc) = seed_five_doc_fixture();
    let h = svc
        .handle(TenantId::DEFAULT, IndexId::DEFAULT_BM25)
        .expect("handle");
    // 4 of 5 docs contain "alpha" (only the needle doc does not).
    let hits = h
        .search("alpha", 2, Lsn::new(100))
        .expect("search must succeed");
    assert!(
        hits.len() <= 2,
        "PIN: ADR-039 §D-8 — search(_, k=2, _) must clip to ≤ 2 hits, \
         got {} hits: {hits:?}",
        hits.len()
    );
    // Sanity: k=10 returns more than k=2 (proves the limit is the
    // active constraint, not the corpus size).
    let hits_unbounded = h
        .search("alpha", 10, Lsn::new(100))
        .expect("search must succeed");
    assert!(
        hits_unbounded.len() > hits.len(),
        "PIN: ADR-039 §D-8 — k=2 must be a strict cap when ≥ 3 docs \
         match, got k=2 → {} vs k=10 → {}",
        hits.len(),
        hits_unbounded.len()
    );
}

// PIN: ADR-039 §D-8 — empty query is well-defined: at v1.0 the
// Tantivy default `QueryParser` parses an empty string into a
// match-nothing query, yielding `Ok(vec![])`. The test pins the
// CURRENT behaviour for stability so a future tokenizer / parser
// change surfaces here.
#[test]
fn empty_query_returns_no_results_or_parse_error() {
    let (_tmp, svc) = seed_five_doc_fixture();
    let h = svc
        .handle(TenantId::DEFAULT, IndexId::DEFAULT_BM25)
        .expect("handle");
    let result = h.search("", 10, Lsn::new(100));
    match result {
        Ok(hits) => assert!(
            hits.is_empty(),
            "PIN: ADR-039 §D-8 — empty query, if accepted, returns an \
             empty Vec (got {} hits: {hits:?})",
            hits.len()
        ),
        Err(Bm25Error::QueryParse { .. }) => {
            // Acceptable alternate behaviour — parser may reject "".
        }
        Err(other) => panic!(
            "PIN: ADR-039 §D-8 — empty query must return Ok(empty) or \
             QueryParse, got {other:?}"
        ),
    }
}

// PIN: ADR-039 §D-5 — buffered upserts are NOT visible until
// `commit_pending(tenant)` fires. The reader is on
// `ReloadPolicy::Manual` and snapshots only the last committed segment;
// a buffered doc has no segment yet.
#[test]
fn search_before_commit_pending_returns_empty() {
    let (_tmp, svc) = fresh_service();
    let h = svc
        .handle(TenantId::DEFAULT, IndexId::DEFAULT_BM25)
        .expect("handle");
    h.upsert_document(NodeId::new(42), "uncommitted body", Lsn::new(1))
        .expect("upsert (buffered)");
    // Deliberately DO NOT call `commit_pending`. The reader must see
    // an empty index.
    let hits = h
        .search("uncommitted", 10, Lsn::new(100))
        .expect("search must succeed even before any commit");
    assert!(
        hits.is_empty(),
        "PIN: ADR-039 §D-5 — buffered (uncommitted) upsert MUST NOT be \
         visible to readers until `commit_pending(tenant)` fires; got \
         {} hits: {hits:?}",
        hits.len()
    );
}
