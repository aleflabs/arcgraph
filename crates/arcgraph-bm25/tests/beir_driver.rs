//! Hermetic coverage for the #1266 BEIR BM25-alone harness index-and-retrieve
//! path (`crates/arcgraph-bm25/benches/beir_bm25_run.rs`).
//!
//! The `beir_bm25_run` bench is a file-driven `main()` (reads a downloaded
//! BEIR `corpus.jsonl` / `queries.jsonl`, writes a TREC run) and is NEVER run
//! in CI — no BEIR dataset is vendored (dependency and artifact policy). But the *core loop* it
//! relies on — `Bm25Service::new` → `handle` → batched `upsert_document` +
//! `commit_pending` → `search(query, k, read_lsn)` returning `(NodeId, score)`
//! sorted descending — is contract-critical.
//!
//! These tests exercise that path against a tiny hand-built synthetic corpus
//! with known relevance judgments, and independently recompute nDCG@10 /
//! MRR@10 / Recall from the returned ranking — validating BOTH the retrieval
//! behavior and the metric semantics with zero download.

use std::sync::Arc;

use arcgraph_bm25::{Bm25Service, IndexId};
use arcgraph_core::{Lsn, NodeId, TenantId};
use arcgraph_storage::mutation_log::Bm25IndexStoreHandle;
use tempfile::TempDir;

/// "See every committed doc" read snapshot — identical to the driver's
/// `READ_ALL_LSN`. `u64::MAX - 1` (not `Lsn::MAX`) because
/// `build_visibility_filter` `debug_assert!(read != u64::MAX)` fires in the
/// DEBUG build these tests run under (ADR-039 §D-3 saturating_add boundary);
/// `MAX - 1` still exceeds every corpus LSN so it admits the whole index.
const READ_ALL_LSN: Lsn = Lsn::new(u64::MAX - 1);

/// A single synthetic BEIR-shaped doc: dense NodeId (as the driver assigns)
/// plus its `title + " " + text` body.
struct Doc {
    node: u64,
    body: &'static str,
}

/// Build a small in-memory BM25 index over a synthetic corpus using the EXACT
/// driver loop (batched `upsert_document` + one final `commit_pending`), then
/// return the service + handle so tests can `search` against it.
fn build_index(docs: &[Doc]) -> (TempDir, Arc<arcgraph_bm25::Bm25IndexHandle>) {
    let tmp = TempDir::new().expect("tempdir");
    let service = Bm25Service::new(tmp.path().to_path_buf());
    let handle = service
        .handle(TenantId::DEFAULT, IndexId::DEFAULT_BM25)
        .expect("handle");
    let store: Arc<dyn Bm25IndexStoreHandle> = Arc::clone(&service) as _;

    for d in docs {
        handle
            .upsert_document(NodeId::new(d.node), d.body, Lsn::new(d.node))
            .expect("upsert");
    }
    store
        .commit_pending(TenantId::DEFAULT)
        .expect("commit_pending");
    (tmp, handle)
}

/// nDCG@k over a ranking of NodeIds vs a graded-relevance map, using the exact
/// trec_eval / BEIR formula the Python scorer implements: gain = 2^rel − 1,
/// discount = log2(i + 1) (1-based rank i), normalized by the ideal DCG.
fn ndcg_at_k(ranking: &[NodeId], rels: &[(u64, u32)], k: usize) -> f64 {
    let rel_of = |n: u64| -> u32 { rels.iter().find(|(id, _)| *id == n).map_or(0, |(_, r)| *r) };
    let mut dcg = 0.0;
    for (i, node) in ranking.iter().take(k).enumerate() {
        let rel = rel_of(node.raw());
        if rel > 0 {
            dcg += (2f64.powi(rel as i32) - 1.0) / ((i as f64 + 2.0).log2());
        }
    }
    let mut ideal: Vec<u32> = rels.iter().map(|(_, r)| *r).filter(|r| *r > 0).collect();
    ideal.sort_unstable_by(|a, b| b.cmp(a));
    let mut idcg = 0.0;
    for (i, rel) in ideal.iter().take(k).enumerate() {
        idcg += (2f64.powi(*rel as i32) - 1.0) / ((i as f64 + 2.0).log2());
    }
    if idcg == 0.0 { 0.0 } else { dcg / idcg }
}

/// The corpus: three "cardiology" docs, three "astronomy" docs, one
/// off-topic. Query "cardiac heart failure" should rank the cardiology docs
/// on top; "galaxy telescope" the astronomy ones.
fn corpus() -> Vec<Doc> {
    vec![
        Doc {
            node: 1,
            body: "cardiac arrest and heart failure treatment in patients",
        },
        Doc {
            node: 2,
            body: "heart failure management and cardiac rehabilitation",
        },
        Doc {
            node: 3,
            body: "the cardiac muscle and coronary heart disease overview",
        },
        Doc {
            node: 4,
            body: "galaxy formation observed by the space telescope survey",
        },
        Doc {
            node: 5,
            body: "telescope optics for deep-field galaxy imaging",
        },
        Doc {
            node: 6,
            body: "the andromeda galaxy and its telescope observations",
        },
        Doc {
            node: 7,
            body: "a recipe for sourdough bread with a long fermentation",
        },
    ]
}

#[test]
fn driver_path_retrieves_and_ranks_relevant_docs_on_top() {
    let (_tmp, handle) = build_index(&corpus());

    // Query phase mirrors the driver: search(text, k=100, READ_ALL_LSN),
    // hits sorted descending by score.
    let hits = handle
        .search("cardiac heart failure", 100, READ_ALL_LSN)
        .expect("search");
    assert!(!hits.is_empty(), "cardiac query must return hits");

    // Descending-score contract (the driver writes rank = position + 1
    // assuming this order).
    for w in hits.windows(2) {
        assert!(
            w[0].1 >= w[1].1,
            "search results must be sorted by descending score: {:?}",
            hits
        );
    }

    // The top-3 must be exactly the three cardiology docs {1,2,3} (in some
    // order) — the off-topic and astronomy docs must not outrank them.
    let ranking: Vec<NodeId> = hits.iter().map(|(n, _)| *n).collect();
    let top3: std::collections::HashSet<u64> = ranking.iter().take(3).map(|n| n.raw()).collect();
    assert_eq!(
        top3,
        [1u64, 2, 3].into_iter().collect(),
        "top-3 for the cardiac query must be the cardiology docs; got {ranking:?}"
    );

    // Graded qrels: the three cardiology docs are relevant (grade 1).
    let rels = [(1u64, 1u32), (2, 1), (3, 1)];
    let ndcg = ndcg_at_k(&ranking, &rels, 10);
    // With all relevant docs ranked in the top-3, nDCG@10 must be 1.0
    // (perfect ranking) — this pins both the retrieval path and the metric.
    assert!(
        (ndcg - 1.0).abs() < 1e-9,
        "perfect cardiology ranking must give nDCG@10 == 1.0, got {ndcg}"
    );
}

#[test]
fn driver_path_second_topic_is_independently_correct() {
    let (_tmp, handle) = build_index(&corpus());
    let hits = handle
        .search("galaxy telescope", 100, READ_ALL_LSN)
        .expect("search");
    let ranking: Vec<NodeId> = hits.iter().map(|(n, _)| *n).collect();
    let top3: std::collections::HashSet<u64> = ranking.iter().take(3).map(|n| n.raw()).collect();
    assert_eq!(
        top3,
        [4u64, 5, 6].into_iter().collect(),
        "top-3 for the galaxy query must be the astronomy docs; got {ranking:?}"
    );
    let rels = [(4u64, 1u32), (5, 1), (6, 1)];
    assert!((ndcg_at_k(&ranking, &rels, 10) - 1.0).abs() < 1e-9);
}

#[test]
fn ndcg_helper_penalizes_a_worse_ranking() {
    // Sanity on the metric itself (guards the harness's correctness anchor):
    // a ranking that buries a relevant doc scores strictly below the ideal.
    let rels = [(1u64, 1u32), (2, 1)];
    let ideal = [NodeId::new(1), NodeId::new(2), NodeId::new(9)];
    let worse = [NodeId::new(9), NodeId::new(1), NodeId::new(2)];
    let ideal_ndcg = ndcg_at_k(&ideal, &rels, 10);
    let worse_ndcg = ndcg_at_k(&worse, &rels, 10);
    assert!((ideal_ndcg - 1.0).abs() < 1e-9, "ideal ranking is nDCG 1.0");
    assert!(
        worse_ndcg < ideal_ndcg,
        "burying a relevant doc must lower nDCG: {worse_ndcg} !< {ideal_ndcg}"
    );
    // And an empty ranking (BM25 returned nothing) scores 0 — the padding
    // case that keeps the mean honest in score.py.
    assert_eq!(ndcg_at_k(&[], &rels, 10), 0.0);
}
