//! #1220 DEMO-KILLER regression pins: free-text user questions
//! containing Tantivy query-DSL metacharacters must NOT crash search.
//!
//! Before the fix, `Bm25IndexHandle::search` handed the raw user
//! string to Tantivy's `QueryParser::parse_query`, which interprets
//! `:` (field-prefix), `[ ]` (range), `^` (boost), and the bare
//! keywords `AND OR NOT TO` as query syntax. A VC typing
//! `"What is the status: open or closed?"` got a 502 with a leaked
//! `Field does not exist: status` parse error.
//!
//! The fix (Option A — term builder) tokenizes the user query with the
//! `body` field's analyzer, per-word, and OR-combines a `TermQuery`
//! (single-token word) or `PhraseQuery` (multi-token word, mirroring
//! the default parser's single-word→phrase grouping) over `body`, so
//! user text is searched as a bag of words and metacharacters become
//! ordinary tokens (or are stripped).
//!
//! PINS:
//! - `metachar_question_returns_hits_not_parse_error` — the acceptance
//!   bar: a question with `: [ ] ^ AND OR NOT` returns RESULTS, not a
//!   `QueryParse` error. RED-on-revert: the old `parse_query` path
//!   errored on this.
//! - `ordinary_question_still_relevant` — relevance is not regressed:
//!   a normal multi-word question still ranks the matching doc at rank
//!   0 with the OR-of-terms BM25 shape.
//! - `metachar_query_respects_acl_visibility` — the MVCC
//!   visibility/permission Must-clause is still applied even with a
//!   metacharacter query: a doc that is not yet committed at the read
//!   snapshot is filtered out.
//! - `assorted_metacharacters_never_parse_error` — a battery of
//!   individual metacharacters / keyword-only queries each return
//!   `Ok` (match-something or match-nothing), never `QueryParse`.

use std::sync::Arc;

use arcgraph_bm25::{Bm25Error, Bm25Service, IndexId};
use arcgraph_core::{Lsn, NodeId, TenantId};
use arcgraph_storage::mutation_log::Bm25IndexStoreHandle;
use tempfile::TempDir;

fn fresh_service() -> (TempDir, Arc<Bm25Service>) {
    let tmp = TempDir::new().expect("tempdir");
    let svc = Bm25Service::new(tmp.path().to_path_buf());
    (tmp, svc)
}

/// Seed a small incident-flavoured fixture and commit it so the reader
/// observes the segment.
fn seed_incident_fixture() -> (TempDir, Arc<Bm25Service>) {
    let (tmp, svc) = fresh_service();
    let h = svc
        .handle(TenantId::DEFAULT, IndexId::DEFAULT_BM25)
        .expect("handle");
    h.upsert_document(
        NodeId::new(1),
        "the incident status is open pending review by the on-call team",
        Lsn::new(1),
    )
    .expect("upsert 1");
    h.upsert_document(
        NodeId::new(2),
        "this ticket is closed after the deploy rollback completed",
        Lsn::new(2),
    )
    .expect("upsert 2");
    h.upsert_document(
        NodeId::new(3),
        "alpha beta gamma unrelated content with no incident words",
        Lsn::new(3),
    )
    .expect("upsert 3");

    let trait_obj: Arc<dyn Bm25IndexStoreHandle> = svc.clone();
    trait_obj
        .commit_pending(TenantId::DEFAULT)
        .expect("commit_pending DEFAULT");
    (tmp, svc)
}

// PIN (#1220 DEMO-KILLER): a natural-language question containing
// Tantivy metacharacters (`:`, `[`, `]`, `^`) and the bare keywords
// `AND OR NOT` must return RESULTS, not a QueryParse error.
//
// RED-on-revert: the pre-fix `QueryParser::parse_query(query)` path
// returns `Err(Bm25Error::QueryParse { ... "Field does not exist:
// status" ... })` for this input — reverting the term-builder fails
// this assertion.
#[test]
fn metachar_question_returns_hits_not_parse_error() {
    let (_tmp, svc) = seed_incident_fixture();
    let h = svc
        .handle(TenantId::DEFAULT, IndexId::DEFAULT_BM25)
        .expect("handle");

    // The exact VC-typed shape from the repro plus extra metacharacters.
    let q = "What is the status: open or closed [URGENT] AND review ^2 NOT done TO end";
    let result = h.search(q, 10, Lsn::new(100));

    let hits = match result {
        Ok(hits) => hits,
        Err(Bm25Error::QueryParse { message }) => panic!(
            "PIN(#1220): a free-text question with metacharacters must NOT \
             surface a QueryParse error — got: {message}"
        ),
        Err(other) => {
            panic!("PIN(#1220): metachar question must return Ok with hits, got {other:?}")
        }
    };

    assert!(
        !hits.is_empty(),
        "PIN(#1220): the metachar question shares terms (status/open/closed/review) \
         with the fixture and MUST return ≥1 hit, got {hits:?}"
    );
    // The two incident docs (1, 2) share more terms than the unrelated
    // doc (3); at minimum one of them must surface.
    let ids: Vec<u64> = hits.iter().map(|(n, _)| n.raw()).collect();
    assert!(
        ids.contains(&1) || ids.contains(&2),
        "PIN(#1220): an incident-relevant doc (1 or 2) must be in the \
         result set for the status question, got {ids:?}"
    );
}

// PIN (#1220): relevance is not regressed by the term-builder. A normal
// multi-word question still ranks the most-relevant doc at rank 0.
#[test]
fn ordinary_question_still_relevant() {
    let (_tmp, svc) = seed_incident_fixture();
    let h = svc
        .handle(TenantId::DEFAULT, IndexId::DEFAULT_BM25)
        .expect("handle");

    // "open pending review" matches doc 1 on three distinct terms; doc 2
    // matches none of these and doc 3 matches none.
    let hits = h
        .search("which incident is open pending review", 10, Lsn::new(100))
        .expect("ordinary question must succeed");
    assert!(
        !hits.is_empty(),
        "PIN(#1220): ordinary question must return relevant hits"
    );
    assert_eq!(
        hits[0].0,
        NodeId::new(1),
        "PIN(#1220): doc 1 matches the most query terms and must rank 0, got {hits:?}"
    );
    assert!(
        hits[0].1 > 0.0,
        "PIN(#1220): the top hit must carry a positive BM25 score, got {hits:?}"
    );
}

// PIN (#1220 ACL): the MVCC visibility/permission Must-clause is still
// composed and enforced even with a metacharacter query. A doc that is
// NOT committed at the read snapshot must be filtered out — confirming
// the term-builder change did NOT weaken the visibility filtering.
#[test]
fn metachar_query_respects_acl_visibility() {
    let (_tmp, svc) = seed_incident_fixture();
    let h = svc
        .handle(TenantId::DEFAULT, IndexId::DEFAULT_BM25)
        .expect("handle");

    // The fixture docs are committed at LSNs 1..=3. A read snapshot of
    // LSN 0 is BEFORE every doc's commit_lsn, so the visibility filter
    // (commit_lsn <= read_lsn) must exclude ALL of them — even with a
    // metacharacter query that matches their body terms.
    let q = "status: open closed [URGENT] AND review";
    let hits_at_zero = h
        .search(q, 10, Lsn::new(0))
        .expect("PIN(#1220 ACL): metachar search at read_lsn=0 must succeed (no parse error)");
    assert!(
        hits_at_zero.is_empty(),
        "PIN(#1220 ACL): the visibility Must-clause MUST filter out docs \
         not yet committed at read_lsn=0, even for a metachar query — got {hits_at_zero:?}"
    );

    // Sanity: at a snapshot AFTER the commits, the same query returns
    // hits — proving the filter is the active constraint (not a
    // match-nothing user query) and the visibility seam is intact.
    let hits_after = h
        .search(q, 10, Lsn::new(100))
        .expect("metachar search after commits must succeed");
    assert!(
        !hits_after.is_empty(),
        "PIN(#1220 ACL): the same query returns hits once docs are visible \
         (read_lsn=100), proving the visibility filter — not the user query — \
         gated the read_lsn=0 result; got {hits_after:?}"
    );
}

// PIN (#1220): a battery of individual metacharacters and keyword-only
// inputs each return Ok (never a QueryParse error). Some match nothing
// (Ok(vec![])) which is the correct, non-crashing behaviour.
#[test]
fn assorted_metacharacters_never_parse_error() {
    let (_tmp, svc) = seed_incident_fixture();
    let h = svc
        .handle(TenantId::DEFAULT, IndexId::DEFAULT_BM25)
        .expect("handle");

    let probes = [
        "status:",
        "[URGENT]",
        "a^2",
        "AND",
        "OR",
        "NOT",
        "TO",
        "AND OR NOT TO",
        "{range}",
        "wild*card?",
        "fuzzy~2",
        "+must -mustnot",
        "(grouped)",
        "back\\slash /slash",
        "\"unbalanced phrase",
        "incident AND status",
    ];
    for probe in probes {
        match h.search(probe, 10, Lsn::new(100)) {
            Ok(_) => {}
            Err(Bm25Error::QueryParse { message }) => panic!(
                "PIN(#1220): probe {probe:?} must NOT crash with a parse error, got: {message}"
            ),
            Err(other) => panic!("PIN(#1220): probe {probe:?} must return Ok, got {other:?}"),
        }
    }
}
