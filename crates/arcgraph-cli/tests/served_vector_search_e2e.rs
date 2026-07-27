//! #765 PART-1 — served vector substrate end-to-end proof.
//!
//! The honesty gate for #765: `graph.ingest` vectors (as the `embedding`
//! node property) → `graph.search` returns RANKED ROWS (not `IndexUnavailable`,
//! not a stub). Exercises the REAL production path:
//!
//! ```text
//! StorageIngestProvider.ingest (real ingest + commit)
//!   → StorageHybridSearcher.search (real body, #765)
//!   → HnswVectorSearchProvider.vector_search (SubstrateSearchProvider)
//!   → HnswGraph::search (arcgraph-vector KNN)
//!   → ranked SearchHits
//! ```
//!
//! Also the ADR-133 Index-class active verification (recall ≥ 0.90 vs an
//! exhaustive brute-force L2 oracle — a STRONG oracle per doctrine §3) and the
//! dimension-mismatch structured-error contract (#765 honesty gate: never a
//! silent wrong result).

use std::collections::BTreeMap;
use std::sync::Arc;

use arcgraph_bm25::Bm25Service;
use arcgraph_cli::bootstrap::{BootstrapMode, bootstrap_storage_backend};
use arcgraph_cli::vector_search::HnswVectorSearchProvider;
use arcgraph_core::{Lsn, NodeId, PartitionId, TenantId};
use arcgraph_mcp::storage::{
    BoltHeldTxn, CrudExecutorSubstrate, StorageHybridSearcher, StorageIngestProvider,
    SubstrateSearchProvider,
};
use arcgraph_mcp::tools::ResponseFormat;
use arcgraph_mcp::tools::ingest::{IngestBatch, IngestProvider, IngestRecordOutcome, NodeIngest};
use arcgraph_mcp::tools::search::{MAX_SEARCH_EF, SearchRequest, search_tool};
use arcgraph_mcp::{CODE_INDEX_UNAVAILABLE, MCPError, SessionScope};
use arcgraph_query::CancellationToken;
use arcgraph_query::executor::ExecutionContext;
use arcgraph_query::executor::substrate::{ExecutorSubstrate, SetNodeMutation};
use arcgraph_query::executor::value::Value;
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::catalog::SystemCatalog;
use arcgraph_storage::crud::CrudStore;
use arcgraph_storage::io::{InMemoryPageIo, PageIo};
use arcgraph_storage::mutation_log::Bm25IndexStoreHandle;
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::PrimaryIndex;
use arcgraph_storage::router::MultiTenantRouter;
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::vector_store::VectorPageStoreHandle;
use tempfile::TempDir;

struct NoopVectorStore;

impl VectorPageStoreHandle for NoopVectorStore {
    fn install_or_replace(
        &self,
        _tenant: TenantId,
        _page_id: arcgraph_core::PageId,
        _bytes: &[u8],
    ) -> Result<(), arcgraph_storage::vector_store::VectorStoreError> {
        Ok(())
    }

    fn restore_page_bytes(
        &self,
        _tenant: TenantId,
        _page_id: arcgraph_core::PageId,
        _bytes: &[u8],
    ) -> Result<(), arcgraph_storage::vector_store::VectorStoreError> {
        Ok(())
    }
}

struct Bm25Stack {
    _tmp: TempDir,
    backend: arcgraph_mcp::storage::StorageBackend,
    _bm25: Arc<Bm25Service>,
}

fn bm25_vector_backend() -> Bm25Stack {
    let tmp = TempDir::new().expect("tempdir");
    let io: Arc<dyn PageIo> = Arc::new(InMemoryPageIo::new());
    let buffer_pool = BufferPool::new(64, io);
    let txn_manager = Arc::new(TxnManager::new());
    let catalog = Arc::new(SystemCatalog::new());
    catalog
        .bootstrap(&buffer_pool, &txn_manager)
        .expect("catalog bootstrap");
    let allocator = Arc::new(PageAllocator::new());
    let primary = Arc::new(
        PrimaryIndex::new(Arc::clone(&txn_manager), Arc::clone(&allocator), None)
            .expect("primary index"),
    );
    let bm25 = Bm25Service::new(tmp.path().join("bm25"));
    let bm25_store: Arc<dyn Bm25IndexStoreHandle> = bm25.clone();
    let crud = Arc::new(
        CrudStore::new_with_index(None, primary, allocator)
            .with_bm25_store(Arc::clone(&bm25_store)),
    );
    let vector_store: Arc<dyn VectorPageStoreHandle> = Arc::new(NoopVectorStore);
    let router = Arc::new(MultiTenantRouter::new_with_bm25(
        Arc::clone(&catalog),
        crud,
        Some(vector_store),
        Some(bm25_store),
    ));
    Bm25Stack {
        _tmp: tmp,
        backend: arcgraph_mcp::storage::StorageBackend::new(
            router,
            txn_manager,
            Arc::new(arcgraph_storage::InternTable::new()),
        ),
        _bm25: bm25,
    }
}

/// Deterministic LCG → f32 in `[0, 1)`. No `rand` dep, no clock — reproducible
/// across runs (proptest-determinism discipline).
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 40) as f32) / ((1u64 << 24) as f32)
}

/// Build an `embedding` node property from an f32 slice.
fn embedding_props(v: &[f32]) -> BTreeMap<String, serde_json::Value> {
    let mut m = BTreeMap::new();
    m.insert(
        "embedding".to_string(),
        serde_json::Value::Array(
            v.iter()
                .map(|f| {
                    serde_json::Number::from_f64(f64::from(*f))
                        .map(serde_json::Value::Number)
                        .expect("finite f32")
                })
                .collect(),
        ),
    );
    m
}

fn text_embedding_props(text: &str, v: &[f32]) -> BTreeMap<String, serde_json::Value> {
    let mut props = embedding_props(v);
    props.insert(
        "text".to_string(),
        serde_json::Value::String(text.to_string()),
    );
    props
}

/// Ingest `(external_id, label, vector)` rows and return the external→internal
/// id map (drawn from the real commit's `Inserted` outcomes).
fn ingest_vectors(
    ingest: &StorageIngestProvider,
    tenant: TenantId,
    rows: &[(String, &str, Vec<f32>)],
) -> BTreeMap<String, u64> {
    let nodes = rows
        .iter()
        .map(|(ext, label, vec)| NodeIngest {
            external_id: Some(ext.clone()),
            label: (*label).to_string(),
            properties: embedding_props(vec),
        })
        .collect();
    let summary = ingest
        .ingest(
            tenant,
            IngestBatch {
                nodes,
                relationships: vec![],
                acl_grants: vec![],
            },
        )
        .expect("ingest vectors");
    assert_eq!(summary.failed_count, 0, "ingest must have 0 failures");
    let mut map = BTreeMap::new();
    for rec in &summary.records {
        if let IngestRecordOutcome::Inserted {
            internal_id,
            external_id,
        } = rec
        {
            map.insert(external_id.clone().unwrap_or_default(), *internal_id);
        }
    }
    map
}

fn ingest_text_vectors(
    ingest: &StorageIngestProvider,
    tenant: TenantId,
    rows: &[(String, &str, &'static str, Vec<f32>)],
) -> BTreeMap<String, u64> {
    let nodes = rows
        .iter()
        .map(|(ext, label, text, vec)| NodeIngest {
            external_id: Some(ext.clone()),
            label: (*label).to_string(),
            properties: text_embedding_props(text, vec),
        })
        .collect();
    let summary = ingest
        .ingest(
            tenant,
            IngestBatch {
                nodes,
                relationships: vec![],
                acl_grants: vec![],
            },
        )
        .expect("ingest text vectors");
    assert_eq!(summary.failed_count, 0, "ingest must have 0 failures");
    let mut map = BTreeMap::new();
    for rec in &summary.records {
        if let IngestRecordOutcome::Inserted {
            internal_id,
            external_id,
        } = rec
        {
            map.insert(external_id.clone().unwrap_or_default(), *internal_id);
        }
    }
    map
}

#[test]
fn graph_search_returns_ranked_rows_end_to_end() {
    // ── The #765 honesty gate: ingest embeddings → graph.search → ranked rows.
    let (backend, _guard) =
        bootstrap_storage_backend(&BootstrapMode::InMemory).expect("bootstrap in-memory");
    let tenant = TenantId::DEFAULT;

    let ingest = StorageIngestProvider::new(backend.clone());
    // 5 docs in a 2-D plane. Query near the origin → doc-1 is the unambiguous
    // nearest; doc-4/doc-5 are far.
    let rows = vec![
        ("doc-1".to_string(), "Doc", vec![0.0_f32, 0.0]),
        ("doc-2".to_string(), "Doc", vec![1.0, 0.0]),
        ("doc-3".to_string(), "Doc", vec![0.0, 1.0]),
        ("doc-4".to_string(), "Doc", vec![10.0, 10.0]),
        ("doc-5".to_string(), "Doc", vec![5.0, 5.0]),
    ];
    let ids = ingest_vectors(&ingest, tenant, &rows);

    let provider = Arc::new(HnswVectorSearchProvider::new(backend.clone()));
    let vector_hits = provider
        .vector_search(tenant, "embedding", &[0.1_f32, 0.1], 3, Lsn::MAX)
        .expect("served provider vector_search must hydrate ranked NodeViews");
    assert_eq!(
        vector_hits[0].node.id.raw(),
        ids["doc-1"],
        "provider vector-only rank-1 must be doc-1 before graph.search adaptation"
    );
    assert!(
        vector_hits[0].node.properties.contains_key("embedding"),
        "served vector-only hits must carry hydrated stored properties; props={:?}",
        vector_hits[0].node.properties
    );

    // Bind the served provider into the production graph.search adapter.
    let provider: Arc<dyn SubstrateSearchProvider> = provider;
    let searcher = StorageHybridSearcher::new(backend.clone()).with_search_provider(provider);

    let req = SearchRequest {
        tenant_id: tenant.raw(),
        query: String::new(), // vector-only (no BM25 operand)
        query_vec: Some(vec![0.1_f32, 0.1]),
        k: Some(3),
        label_filter: None,
        ef_search: None,
        format: Some(ResponseFormat::Json),
        principal: None,
    };
    let token = CancellationToken::new();
    let resp = search_tool(&searcher, tenant, SessionScope::Power, &token, req)
        .expect("graph.search must return Ok");

    assert_eq!(resp["format"], "json");
    let body: serde_json::Value =
        serde_json::from_str(resp["body"].as_str().expect("body string")).expect("parse body");

    // ── Honesty-gate paste: the ACTUAL ranked rows graph.search returned.
    eprintln!("#765 graph.search ranked rows (query=[0.1,0.1], k=3):");
    eprintln!("  k = {}", body["k"]);
    for (i, hit) in body["hits"]
        .as_array()
        .expect("hits array")
        .iter()
        .enumerate()
    {
        eprintln!(
            "  rank {}: node_id={} label={} score={}",
            i + 1,
            hit["node_id"],
            hit["label"],
            hit["score"],
        );
    }

    let hits = body["hits"].as_array().expect("hits array");
    assert!(!hits.is_empty(), "graph.search must return rows, not empty");
    assert!(hits.len() <= 3, "must honor k=3");
    // The nearest node to [0.1,0.1] is doc-1 (origin).
    assert_eq!(
        hits[0]["node_id"].as_u64().expect("node_id u64"),
        ids["doc-1"],
        "rank-1 must be doc-1 (the nearest embedding)",
    );
    assert_eq!(hits[0]["label"].as_str(), Some("Doc"), "label resolved");
    // Scores are in (0, 1] and strictly descending (closest first).
    let scores: Vec<f64> = hits
        .iter()
        .map(|h| h["score"].as_f64().expect("score f64"))
        .collect();
    for w in scores.windows(2) {
        assert!(w[0] >= w[1], "scores must be descending: {scores:?}");
    }
    assert!(
        scores.iter().all(|s| *s > 0.0 && *s <= 1.0),
        "scores in (0,1]: {scores:?}",
    );
}

#[test]
fn served_provider_bm25_search_returns_tantivy_hits_with_hydrated_props_765() {
    let stack = bm25_vector_backend();
    let tenant = TenantId::DEFAULT;
    let ingest = StorageIngestProvider::new(stack.backend.clone());
    let rows = vec![
        (
            "alpha-doc".to_string(),
            "Doc",
            "alpha alpha alpha unique",
            vec![10.0_f32, 10.0],
        ),
        (
            "beta-doc".to_string(),
            "Doc",
            "beta beta beta other",
            vec![0.0_f32, 0.0],
        ),
    ];
    let ids = ingest_text_vectors(&ingest, tenant, &rows);

    let provider = HnswVectorSearchProvider::new(stack.backend.clone());
    let hits = provider
        .bm25_search(tenant, "text", "alpha", 5, Lsn::new(100))
        .expect("served provider bm25_search must query real Tantivy");

    assert!(
        !hits.is_empty(),
        "BM25 provider search must return the alpha hit, not the old stub error"
    );
    assert_eq!(
        hits[0].node.id.raw(),
        ids["alpha-doc"],
        "unique alpha doc must rank first; hits={hits:?}",
    );
    assert_eq!(hits[0].node.label_name.as_deref(), Some("Doc"));
    assert_eq!(
        hits[0].node.properties.get("text"),
        Some(&Value::String("alpha alpha alpha unique".to_string())),
        "provider BM25 hits must carry hydrated stored properties"
    );
}

#[test]
fn graph_search_text_only_indexes_ingested_text_without_manual_bm25_seed_985() {
    let stack = bm25_vector_backend();
    let tenant = TenantId::DEFAULT;
    let ingest = StorageIngestProvider::new(stack.backend.clone());
    let rows = vec![
        (
            "alpha-doc".to_string(),
            "Doc",
            "alpha apple",
            vec![10.0_f32, 10.0],
        ),
        (
            "beta-doc".to_string(),
            "Doc",
            "beta banana",
            vec![0.0_f32, 0.0],
        ),
    ];
    let ids = ingest_text_vectors(&ingest, tenant, &rows);

    let provider: Arc<dyn SubstrateSearchProvider> =
        Arc::new(HnswVectorSearchProvider::new(stack.backend.clone()));
    let searcher = StorageHybridSearcher::new(stack.backend.clone()).with_search_provider(provider);
    let token = CancellationToken::new();

    let resp = search_tool(
        &searcher,
        tenant,
        SessionScope::Power,
        &token,
        SearchRequest {
            tenant_id: tenant.raw(),
            query: "alpha".to_string(),
            query_vec: None,
            k: Some(1),
            label_filter: None,
            ef_search: None,
            format: Some(ResponseFormat::Json),
            principal: None,
        },
    )
    .expect("graph.search text-only Ok");
    let body: serde_json::Value = serde_json::from_str(resp["body"].as_str().expect("body string"))
        .expect("parse graph.search body");
    let hits = body["hits"].as_array().expect("hits array");
    assert_eq!(
        hits.first()
            .and_then(|h| h["node_id"].as_u64())
            .expect("rank-1 node_id"),
        ids["alpha-doc"],
        "ingested text property must be automatically populated into served BM25; hits={hits:?}",
    );
}

#[test]
fn graph_search_bm25_update_and_delete_maintenance_on_ingested_text_985() {
    let stack = bm25_vector_backend();
    let tenant = TenantId::DEFAULT;
    let ingest = StorageIngestProvider::new(stack.backend.clone());
    let rows = vec![
        (
            "mutable".to_string(),
            "Doc",
            "alpha apple",
            vec![0.0_f32, 0.0],
        ),
        (
            "other".to_string(),
            "Doc",
            "gamma grape",
            vec![1.0_f32, 1.0],
        ),
    ];
    let ids = ingest_text_vectors(&ingest, tenant, &rows);
    let mutable_id = NodeId::new(ids["mutable"]);

    let provider = Arc::new(HnswVectorSearchProvider::new(stack.backend.clone()));
    let searcher = StorageHybridSearcher::new(stack.backend.clone())
        .with_search_provider(Arc::clone(&provider) as Arc<dyn SubstrateSearchProvider>);
    let sub = CrudExecutorSubstrate::new(
        Arc::clone(stack.backend.router()),
        Arc::clone(stack.backend.txn_manager()),
        Arc::clone(stack.backend.intern_table()),
    )
    .with_search_provider(Arc::clone(&provider) as Arc<dyn SubstrateSearchProvider>);
    let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);
    let token = CancellationToken::new();

    let search_ids = |query: &str| -> Vec<u64> {
        let resp = search_tool(
            &searcher,
            tenant,
            SessionScope::Power,
            &token,
            SearchRequest {
                tenant_id: tenant.raw(),
                query: query.to_string(),
                query_vec: None,
                k: Some(5),
                label_filter: None,
                ef_search: None,
                format: Some(ResponseFormat::Json),
                principal: None,
            },
        )
        .expect("graph.search text-only Ok");
        let body: serde_json::Value =
            serde_json::from_str(resp["body"].as_str().expect("body string"))
                .expect("parse graph.search body");
        body["hits"]
            .as_array()
            .expect("hits array")
            .iter()
            .map(|h| h["node_id"].as_u64().expect("node_id u64"))
            .collect()
    };

    assert_eq!(
        search_ids("alpha").first().copied(),
        Some(mutable_id.raw()),
        "baseline: ingested alpha text must be searchable",
    );

    sub.set_node(
        tenant,
        mutable_id,
        &SetNodeMutation::PropertyAssign {
            name: "text".to_string(),
            value: Value::String("beta berry".to_string()),
        },
        &ctx,
    )
    .expect("SET mutable.text through production substrate");
    assert!(
        !search_ids("alpha").contains(&mutable_id.raw()),
        "old text must be removed from served BM25 after SET",
    );
    assert_eq!(
        search_ids("beta").first().copied(),
        Some(mutable_id.raw()),
        "new text must be indexed into served BM25 after SET",
    );

    sub.delete_node(tenant, mutable_id, true, &ctx)
        .expect("DETACH DELETE mutable through production substrate");
    assert!(
        !search_ids("beta").contains(&mutable_id.raw()),
        "deleted node must be removed from served BM25",
    );
}

#[test]
fn graph_search_bm25_set_text_to_non_string_removes_stale_doc_987() {
    let stack = bm25_vector_backend();
    let tenant = TenantId::DEFAULT;
    let ingest = StorageIngestProvider::new(stack.backend.clone());
    let rows = vec![(
        "mutable".to_string(),
        "Doc",
        "alpha apple",
        vec![0.0_f32, 0.0],
    )];
    let ids = ingest_text_vectors(&ingest, tenant, &rows);
    let mutable_id = NodeId::new(ids["mutable"]);

    let provider = Arc::new(HnswVectorSearchProvider::new(stack.backend.clone()));
    let searcher = StorageHybridSearcher::new(stack.backend.clone())
        .with_search_provider(Arc::clone(&provider) as Arc<dyn SubstrateSearchProvider>);
    let sub = CrudExecutorSubstrate::new(
        Arc::clone(stack.backend.router()),
        Arc::clone(stack.backend.txn_manager()),
        Arc::clone(stack.backend.intern_table()),
    )
    .with_search_provider(Arc::clone(&provider) as Arc<dyn SubstrateSearchProvider>);
    let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);
    let token = CancellationToken::new();

    let search_ids = |query: &str| -> Vec<u64> {
        let resp = search_tool(
            &searcher,
            tenant,
            SessionScope::Power,
            &token,
            SearchRequest {
                tenant_id: tenant.raw(),
                query: query.to_string(),
                query_vec: None,
                k: Some(5),
                label_filter: None,
                ef_search: None,
                format: Some(ResponseFormat::Json),
                principal: None,
            },
        )
        .expect("graph.search text-only Ok");
        let body: serde_json::Value =
            serde_json::from_str(resp["body"].as_str().expect("body string"))
                .expect("parse graph.search body");
        body["hits"]
            .as_array()
            .expect("hits array")
            .iter()
            .map(|h| h["node_id"].as_u64().expect("node_id u64"))
            .collect()
    };

    assert!(
        search_ids("alpha").contains(&mutable_id.raw()),
        "baseline: ingested alpha text must be searchable",
    );

    sub.set_node(
        tenant,
        mutable_id,
        &SetNodeMutation::PropertyAssign {
            name: "text".to_string(),
            value: Value::Integer(42),
        },
        &ctx,
    )
    .expect("SET mutable.text to non-string through production substrate");

    assert!(
        !search_ids("alpha").contains(&mutable_id.raw()),
        "clearing the last string text property must remove the stale BM25 doc",
    );
}

#[test]
fn graph_search_hybrid_rrf_promotes_text_strong_vector_weak_hit_765() {
    let stack = bm25_vector_backend();
    let tenant = TenantId::DEFAULT;
    let ingest = StorageIngestProvider::new(stack.backend.clone());
    let rows = vec![
        (
            "vector-near".to_string(),
            "Doc",
            "omega omega omega",
            vec![0.0_f32, 0.0],
        ),
        (
            "text-strong".to_string(),
            "Doc",
            "alpha alpha alpha alpha alpha",
            vec![10.0_f32, 10.0],
        ),
    ];
    let ids = ingest_text_vectors(&ingest, tenant, &rows);

    let provider: Arc<dyn SubstrateSearchProvider> =
        Arc::new(HnswVectorSearchProvider::new(stack.backend.clone()));
    let searcher = StorageHybridSearcher::new(stack.backend.clone()).with_search_provider(provider);
    let token = CancellationToken::new();

    let search_ids = |query: &str, query_vec: Vec<f32>| -> Vec<u64> {
        let resp = search_tool(
            &searcher,
            tenant,
            SessionScope::Power,
            &token,
            SearchRequest {
                tenant_id: tenant.raw(),
                query: query.to_string(),
                query_vec: Some(query_vec),
                k: Some(2),
                label_filter: None,
                ef_search: Some(MAX_SEARCH_EF),
                format: Some(ResponseFormat::Json),
                principal: None,
            },
        )
        .expect("graph.search Ok");
        let body: serde_json::Value =
            serde_json::from_str(resp["body"].as_str().expect("body string"))
                .expect("parse graph.search body");
        body["hits"]
            .as_array()
            .expect("hits array")
            .iter()
            .map(|h| h["node_id"].as_u64().expect("node_id u64"))
            .collect()
    };

    let vector_only = search_ids("", vec![0.0_f32, 0.0]);
    assert_eq!(
        vector_only.first().copied(),
        Some(ids["vector-near"]),
        "vector-only order must rank the nearest embedding first; hits={vector_only:?}"
    );

    let hybrid = search_ids("alpha", vec![0.0_f32, 0.0]);
    assert_eq!(
        hybrid.first().copied(),
        Some(ids["text-strong"]),
        "RRF hybrid must promote the text-strong/vector-weak node above vector-only rank; \
         vector_only={vector_only:?}, hybrid={hybrid:?}"
    );
}

#[test]
fn graph_search_hybrid_degrades_to_vector_when_bm25_unavailable_765() {
    let (backend, _guard) =
        bootstrap_storage_backend(&BootstrapMode::InMemory).expect("bootstrap in-memory");
    let tenant = TenantId::DEFAULT;
    let ingest = StorageIngestProvider::new(backend.clone());
    let rows = vec![
        ("near".to_string(), "Doc", vec![0.0_f32, 0.0]),
        ("far".to_string(), "Doc", vec![9.0_f32, 9.0]),
    ];
    let ids = ingest_vectors(&ingest, tenant, &rows);
    let provider: Arc<dyn SubstrateSearchProvider> =
        Arc::new(HnswVectorSearchProvider::new(backend.clone()));
    let searcher = StorageHybridSearcher::new(backend.clone()).with_search_provider(provider);
    let token = CancellationToken::new();

    let resp = search_tool(
        &searcher,
        tenant,
        SessionScope::Power,
        &token,
        SearchRequest {
            tenant_id: tenant.raw(),
            query: "alpha should degrade because bm25 is absent".to_string(),
            query_vec: Some(vec![0.0_f32, 0.0]),
            k: Some(1),
            label_filter: None,
            ef_search: Some(MAX_SEARCH_EF),
            format: Some(ResponseFormat::Json),
            principal: None,
        },
    )
    .expect("hybrid request must degrade to vector-only when bm25 is unavailable");
    let body: serde_json::Value = serde_json::from_str(resp["body"].as_str().expect("body string"))
        .expect("parse graph.search body");
    let hits = body["hits"].as_array().expect("hits array");
    assert_eq!(hits.len(), 1);
    assert_eq!(
        hits[0]["node_id"].as_u64(),
        Some(ids["near"]),
        "BM25-unavailable hybrid request must preserve vector-only result"
    );
}

#[test]
fn graph_search_excludes_detach_deleted_served_hnsw_node_909() {
    let (backend, _guard) =
        bootstrap_storage_backend(&BootstrapMode::InMemory).expect("bootstrap in-memory");
    let tenant = TenantId::DEFAULT;
    let ingest = StorageIngestProvider::new(backend.clone());

    let q = vec![0.25_f32, 0.75, 0.5, 1.0];
    let rows = vec![
        ("mut".to_string(), "Doc", q.clone()),
        ("near".to_string(), "Doc", vec![0.30_f32, 0.70, 0.45, 0.95]),
        ("mid".to_string(), "Doc", vec![1.0_f32, 1.0, 1.0, 1.0]),
        ("far".to_string(), "Doc", vec![9.0_f32, 9.0, 9.0, 9.0]),
    ];
    let ids = ingest_vectors(&ingest, tenant, &rows);
    let mut_id = NodeId::new(ids["mut"]);

    let provider = Arc::new(HnswVectorSearchProvider::new(backend.clone()));
    let searcher = StorageHybridSearcher::new(backend.clone())
        .with_search_provider(Arc::clone(&provider) as Arc<dyn SubstrateSearchProvider>);
    let sub = CrudExecutorSubstrate::new(
        Arc::clone(backend.router()),
        Arc::clone(backend.txn_manager()),
        Arc::clone(backend.intern_table()),
    )
    .with_search_provider(Arc::clone(&provider) as Arc<dyn SubstrateSearchProvider>);
    let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);
    let token = CancellationToken::new();

    let search_ids = |k: u32| -> Vec<u64> {
        let resp = search_tool(
            &searcher,
            tenant,
            SessionScope::Power,
            &token,
            SearchRequest {
                tenant_id: tenant.raw(),
                query: String::new(),
                query_vec: Some(q.clone()),
                k: Some(k),
                label_filter: None,
                ef_search: Some(MAX_SEARCH_EF),
                format: Some(ResponseFormat::Json),
                principal: None,
            },
        )
        .expect("graph.search Ok");
        let body: serde_json::Value =
            serde_json::from_str(resp["body"].as_str().expect("body string"))
                .expect("parse graph.search body");
        body["hits"]
            .as_array()
            .expect("hits array")
            .iter()
            .map(|h| h["node_id"].as_u64().expect("node_id u64"))
            .collect()
    };

    let baseline = search_ids(4);
    assert_eq!(
        baseline.first().copied(),
        Some(mut_id.raw()),
        "baseline: MUT is exactly at Q and must be rank-1 before delete; hits={baseline:?}",
    );

    sub.delete_node(tenant, mut_id, true, &ctx)
        .expect("DETACH DELETE MUT through production substrate");
    let visible_after_delete = sub
        .scan_nodes(tenant, None, Lsn::MAX)
        .expect("scan after delete")
        .into_iter()
        .filter(|n| n.node.id == mut_id)
        .count();
    assert_eq!(
        visible_after_delete, 0,
        "store oracle: deleted MUT must not be visible after DETACH DELETE",
    );

    let post_delete = search_ids(4);
    assert!(
        !post_delete.contains(&mut_id.raw()),
        "post-delete graph.search must not return tombstoned MUT; hits={post_delete:?}, mut={}",
        mut_id.raw(),
    );

    let recreated = ingest_vectors(
        &ingest,
        tenant,
        &[("mut-recreated".to_string(), "Doc", q.clone())],
    );
    let recreated_id = recreated["mut-recreated"];
    assert_ne!(
        recreated_id,
        mut_id.raw(),
        "normal create path allocates a fresh NodeId for the logical recreate",
    );
    let post_recreate = search_ids(4);
    assert_eq!(
        post_recreate.first().copied(),
        Some(recreated_id),
        "delete-then-recreate logical MUT at Q must be findable again; hits={post_recreate:?}",
    );
    assert!(
        !post_recreate.contains(&mut_id.raw()),
        "old tombstoned MUT id must remain excluded after recreate; hits={post_recreate:?}",
    );
}

#[test]
fn graph_search_reindexes_served_hnsw_node_after_embedding_update_909() {
    let (backend, _guard) =
        bootstrap_storage_backend(&BootstrapMode::InMemory).expect("bootstrap in-memory");
    let tenant = TenantId::DEFAULT;
    let ingest = StorageIngestProvider::new(backend.clone());

    let q = vec![0.0_f32, 0.0, 0.0, 0.0];
    let far = vec![50.0_f32, 50.0, 50.0, 50.0];
    let rows = vec![
        ("upd".to_string(), "Doc", q.clone()),
        ("near".to_string(), "Doc", vec![0.2_f32, 0.2, 0.2, 0.2]),
        ("mid".to_string(), "Doc", vec![3.0_f32, 3.0, 3.0, 3.0]),
        (
            "far-anchor".to_string(),
            "Doc",
            vec![75.0_f32, 75.0, 75.0, 75.0],
        ),
    ];
    let ids = ingest_vectors(&ingest, tenant, &rows);
    let upd_id = NodeId::new(ids["upd"]);
    let near_id = NodeId::new(ids["near"]);

    let provider = Arc::new(HnswVectorSearchProvider::new(backend.clone()));
    let searcher = StorageHybridSearcher::new(backend.clone())
        .with_search_provider(Arc::clone(&provider) as Arc<dyn SubstrateSearchProvider>);
    let sub = CrudExecutorSubstrate::new(
        Arc::clone(backend.router()),
        Arc::clone(backend.txn_manager()),
        Arc::clone(backend.intern_table()),
    )
    .with_search_provider(Arc::clone(&provider) as Arc<dyn SubstrateSearchProvider>);
    let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);
    let token = CancellationToken::new();

    let search_ids = |query_vec: Vec<f32>, k: u32| -> Vec<u64> {
        let resp = search_tool(
            &searcher,
            tenant,
            SessionScope::Power,
            &token,
            SearchRequest {
                tenant_id: tenant.raw(),
                query: String::new(),
                query_vec: Some(query_vec),
                k: Some(k),
                label_filter: None,
                ef_search: Some(MAX_SEARCH_EF),
                format: Some(ResponseFormat::Json),
                principal: None,
            },
        )
        .expect("graph.search Ok");
        let body: serde_json::Value =
            serde_json::from_str(resp["body"].as_str().expect("body string"))
                .expect("parse graph.search body");
        body["hits"]
            .as_array()
            .expect("hits array")
            .iter()
            .map(|h| h["node_id"].as_u64().expect("node_id u64"))
            .collect()
    };

    let baseline_q = search_ids(q.clone(), 4);
    assert_eq!(
        baseline_q.first().copied(),
        Some(upd_id.raw()),
        "baseline: UPD is exactly at Q and must be rank-1 before update; hits={baseline_q:?}",
    );
    sub.set_node(
        tenant,
        upd_id,
        &SetNodeMutation::PropertyAssign {
            name: "embedding".to_string(),
            value: Value::List(far.iter().map(|v| Value::Float(f64::from(*v))).collect()),
        },
        &ctx,
    )
    .expect("SET upd.embedding = FAR through production substrate");

    let stored = sub
        .scan_nodes(tenant, None, Lsn::MAX)
        .expect("scan after embedding update")
        .into_iter()
        .find(|n| n.node.id == upd_id)
        .expect("updated node remains visible");
    assert_eq!(
        stored.node.properties.get("embedding"),
        Some(&Value::List(
            far.iter().map(|v| Value::Float(f64::from(*v))).collect()
        )),
        "store oracle: embedding property must change before search re-index is credited",
    );

    let after_q = search_ids(q.clone(), 4);
    assert_ne!(
        after_q.first().copied(),
        Some(upd_id.raw()),
        "stale-vector half: UPD must no longer rank at old Q after embedding update; hits={after_q:?}",
    );
    let after_far = search_ids(far.clone(), 4);
    assert_eq!(
        after_far.first().copied(),
        Some(upd_id.raw()),
        "re-index half: UPD must be top-1 at its new FAR embedding; hits={after_far:?}",
    );
    let post_embedding_update_near = search_ids(vec![0.2_f32, 0.2, 0.2, 0.2], 4);
    let before_non_embedding_set_inserts = provider.metrics().vectors_inserted;

    sub.set_node(
        tenant,
        near_id,
        &SetNodeMutation::PropertyAssign {
            name: "name".to_string(),
            value: Value::String("x".to_string()),
        },
        &ctx,
    )
    .expect("SET near.name = 'x' through production substrate");
    let after_non_embedding = search_ids(vec![0.2_f32, 0.2, 0.2, 0.2], 4);
    assert_eq!(
        after_non_embedding, post_embedding_update_near,
        "non-embedding SET must not trigger vector re-index or perturb ranking",
    );
    assert_eq!(
        provider.metrics().vectors_inserted,
        before_non_embedding_set_inserts,
        "non-embedding SET must not queue mark_vector_node_updated / re-index",
    );
}

#[test]
fn explicit_held_txn_vector_hooks_fire_after_commit_not_before() {
    // #963: in explicit BEGIN...COMMIT mode, vector maintenance hooks must not
    // fire while writes are only staged. A search between SET/DELETE and COMMIT
    // observes committed storage, so pre-commit hooks can permanently apply the
    // wrong maintenance action.
    let (backend, _guard) =
        bootstrap_storage_backend(&BootstrapMode::InMemory).expect("bootstrap in-memory");
    let tenant = TenantId::DEFAULT;
    let ingest = StorageIngestProvider::new(backend.clone());
    let q = vec![0.0_f32, 0.0, 0.0, 0.0];
    let far = vec![80.0_f32, 80.0, 80.0, 80.0];
    let rows = vec![
        ("upd".to_string(), "Doc", q.clone()),
        (
            "far-anchor".to_string(),
            "Doc",
            vec![100.0_f32, 100.0, 100.0, 100.0],
        ),
        ("victim".to_string(), "Doc", vec![0.2_f32, 0.2, 0.2, 0.2]),
    ];
    let ids = ingest_vectors(&ingest, tenant, &rows);
    let upd_id = NodeId::new(ids["upd"]);
    let victim_id = NodeId::new(ids["victim"]);

    let provider = Arc::new(HnswVectorSearchProvider::new(backend.clone()));
    let searcher = StorageHybridSearcher::new(backend.clone())
        .with_search_provider(Arc::clone(&provider) as Arc<dyn SubstrateSearchProvider>);
    let sub = CrudExecutorSubstrate::new(
        Arc::clone(backend.router()),
        Arc::clone(backend.txn_manager()),
        Arc::clone(backend.intern_table()),
    )
    .with_search_provider(Arc::clone(&provider) as Arc<dyn SubstrateSearchProvider>);
    let token = CancellationToken::new();

    let search_ids = |query_vec: Vec<f32>, k: u32| -> Vec<u64> {
        let resp = search_tool(
            &searcher,
            tenant,
            SessionScope::Power,
            &token,
            SearchRequest {
                tenant_id: tenant.raw(),
                query: String::new(),
                query_vec: Some(query_vec),
                k: Some(k),
                label_filter: None,
                ef_search: Some(MAX_SEARCH_EF),
                format: Some(ResponseFormat::Json),
                principal: None,
            },
        )
        .expect("graph.search Ok");
        let body: serde_json::Value =
            serde_json::from_str(resp["body"].as_str().expect("body string"))
                .expect("parse graph.search body");
        body["hits"]
            .as_array()
            .expect("hits array")
            .iter()
            .map(|h| h["node_id"].as_u64().expect("node_id u64"))
            .collect()
    };

    assert_eq!(
        search_ids(q.clone(), 3).first().copied(),
        Some(upd_id.raw()),
        "baseline: updated node starts at Q",
    );

    let update_ctx = ExecutionContext::new(tenant, PartitionId::ZERO).with_held_txn(Box::new(
        BoltHeldTxn::new(backend.txn_manager().begin_owned(tenant)),
    ));
    sub.set_node(
        tenant,
        upd_id,
        &SetNodeMutation::PropertyAssign {
            name: "embedding".to_string(),
            value: Value::List(far.iter().map(|v| Value::Float(f64::from(*v))).collect()),
        },
        &update_ctx,
    )
    .expect("stage SET upd.embedding = FAR in explicit held tx");

    assert_eq!(
        search_ids(q.clone(), 3).first().copied(),
        Some(upd_id.raw()),
        "search between staged SET and COMMIT must not drain a reindex hook",
    );
    let held = update_ctx
        .take_held_txn()
        .expect("held update tx remains installed");
    sub.commit_bolt_held_handle(held)
        .expect("commit explicit update tx");
    assert_eq!(
        search_ids(far.clone(), 3).first().copied(),
        Some(upd_id.raw()),
        "after COMMIT, queued update hook re-indexes the committed FAR embedding",
    );

    let delete_ctx = ExecutionContext::new(tenant, PartitionId::ZERO).with_held_txn(Box::new(
        BoltHeldTxn::new(backend.txn_manager().begin_owned(tenant)),
    ));
    sub.delete_node(tenant, victim_id, false, &delete_ctx)
        .expect("stage DELETE victim in explicit held tx");
    let held = delete_ctx
        .take_held_txn()
        .expect("held delete tx remains installed");
    drop(held);
    assert!(
        search_ids(vec![0.2_f32, 0.2, 0.2, 0.2], 3).contains(&victim_id.raw()),
        "ROLLBACK/drop after staged DELETE must not leave a served-HNSW tombstone",
    );
}

#[test]
fn graph_search_reflects_new_ingests_after_high_water_advance() {
    // ── The #765 PART-1 derived-index LIFECYCLE claim (D6 + PR Risk #1):
    // the per-tenant HNSW is invalidated + rebuilt when the node high-water
    // mark advances on a fresh ingest, so `graph.search` reflects newly
    // ingested vectors — it does NOT stale-serve the index built on the first
    // search. This is the load-bearing claim the rest of the wiring rests on;
    // a strong oracle here (the new, NEARER node_ids must appear AND outrank
    // the older far ones) fails on either a missed rebuild (off-by-one in the
    // `built_high_water == high_water` key) or a stale-serve.
    let (backend, _guard) =
        bootstrap_storage_backend(&BootstrapMode::InMemory).expect("bootstrap in-memory");
    let tenant = TenantId::DEFAULT;
    let ingest = StorageIngestProvider::new(backend.clone());

    // Round 1: ingest 3 vectors, ALL far from the query [0,0].
    let far_rows = vec![
        ("far-1".to_string(), "Doc", vec![10.0_f32, 10.0]),
        ("far-2".to_string(), "Doc", vec![11.0, 11.0]),
        ("far-3".to_string(), "Doc", vec![12.0, 12.0]),
    ];
    let far_ids = ingest_vectors(&ingest, tenant, &far_rows);

    // Bind the served provider into the production graph.search adapter. The
    // SAME provider instance is reused across both searches — so the second
    // search hits the cache path and MUST observe the high-water advance to
    // pass (a stale-serve would re-return the round-1 index).
    let provider: Arc<dyn SubstrateSearchProvider> =
        Arc::new(HnswVectorSearchProvider::new(backend.clone()));
    let searcher = StorageHybridSearcher::new(backend.clone()).with_search_provider(provider);

    let query = vec![0.0_f32, 0.0];
    let make_req = || SearchRequest {
        tenant_id: tenant.raw(),
        query: String::new(),
        query_vec: Some(query.clone()),
        k: Some(3),
        label_filter: None,
        ef_search: None,
        format: Some(ResponseFormat::Json),
        principal: None,
    };
    let token = CancellationToken::new();

    // First search → builds + caches the HNSW over the 3 far vectors.
    let resp1 = search_tool(&searcher, tenant, SessionScope::Power, &token, make_req())
        .expect("graph.search round-1 Ok");
    let body1: serde_json::Value =
        serde_json::from_str(resp1["body"].as_str().expect("body string")).expect("parse body1");
    let hits1: Vec<u64> = body1["hits"]
        .as_array()
        .expect("hits1 array")
        .iter()
        .map(|h| h["node_id"].as_u64().expect("node_id u64"))
        .collect();
    eprintln!("#765 invalidation round-1 ranked node_ids (only far vectors ingested): {hits1:?}");
    // Round-1 results are exactly the far vectors — the nearest is far-1 ([10,10]).
    assert_eq!(
        hits1.first().copied(),
        Some(far_ids["far-1"]),
        "round-1 rank-1 must be far-1 (nearest of the far set)",
    );
    let far_id_set: std::collections::HashSet<u64> = far_ids.values().copied().collect();
    assert!(
        hits1.iter().all(|id| far_id_set.contains(id)),
        "round-1 hits must all be far vectors (no near vectors ingested yet): {hits1:?}",
    );

    // Round 2: ingest 2 MORE distinct vectors, both NEARER the query than any
    // far vector. This advances the node high-water mark.
    let near_rows = vec![
        ("near-1".to_string(), "Doc", vec![0.1_f32, 0.1]),
        ("near-2".to_string(), "Doc", vec![0.2_f32, 0.2]),
    ];
    let near_ids = ingest_vectors(&ingest, tenant, &near_rows);
    // The near node_ids must be genuinely new (distinct from the far ones).
    assert!(
        near_ids.values().all(|id| !far_id_set.contains(id)),
        "near vectors must have fresh node_ids distinct from the far set",
    );

    // Second search → the high-water mark advanced, so the derived HNSW MUST be
    // rebuilt to include the near vectors. A stale-serve would return the
    // round-1 index (far vectors only) and fail every assertion below.
    let resp2 = search_tool(&searcher, tenant, SessionScope::Power, &token, make_req())
        .expect("graph.search round-2 Ok");
    let body2: serde_json::Value =
        serde_json::from_str(resp2["body"].as_str().expect("body string")).expect("parse body2");
    let hits2: Vec<u64> = body2["hits"]
        .as_array()
        .expect("hits2 array")
        .iter()
        .map(|h| h["node_id"].as_u64().expect("node_id u64"))
        .collect();
    eprintln!("#765 invalidation round-2 ranked node_ids (2 NEARER vectors added): {hits2:?}");

    // The new NEARER vectors are now reflected (invalidation fired) AND outrank
    // the older far ones: with k=3, the top-2 must be the two near vectors
    // (closest-first), and the new node_ids must appear in the results.
    let hit2_set: std::collections::HashSet<u64> = hits2.iter().copied().collect();
    assert!(
        hit2_set.contains(&near_ids["near-1"]) && hit2_set.contains(&near_ids["near-2"]),
        "round-2 must reflect BOTH newly ingested near vectors (invalidation fired, \
         not a stale-serve): hits2={hits2:?}, near-1={}, near-2={}",
        near_ids["near-1"],
        near_ids["near-2"],
    );
    assert_eq!(
        hits2.first().copied(),
        Some(near_ids["near-1"]),
        "round-2 rank-1 must be near-1 ([0.1,0.1], the new nearest) — proves the rebuilt \
         index ranks the fresh vectors, not the stale far set",
    );
    assert_eq!(
        hits2.get(1).copied(),
        Some(near_ids["near-2"]),
        "round-2 rank-2 must be near-2 ([0.2,0.2], the new second-nearest)",
    );
}

#[test]
fn served_hnsw_recall_at_10_meets_floor_vs_brute_force() {
    // ── ADR-133 Index-class active verification: HNSW recall@10 vs an
    // exhaustive brute-force L2 oracle (a STRONG oracle — exact, not sampled).
    // Sized for a single-crate DEBUG run (so it never contends with the live
    // 10M DiskANN build); the full 10K-insert / 1K-query release bench is a
    // follow-on bench point.
    const N: usize = 500;
    const DIM: usize = 8;
    const QUERIES: usize = 50;
    const K: usize = 10;
    const RECALL_FLOOR: f64 = 0.90;

    let (backend, _guard) =
        bootstrap_storage_backend(&BootstrapMode::InMemory).expect("bootstrap in-memory");
    let tenant = TenantId::DEFAULT;
    let ingest = StorageIngestProvider::new(backend.clone());

    // Deterministic corpus.
    let mut seed = 0x5765_7635u64; // "Wv6 5" — fixed seed.
    let mut rows: Vec<(String, &str, Vec<f32>)> = Vec::with_capacity(N);
    for i in 0..N {
        let vec: Vec<f32> = (0..DIM).map(|_| lcg(&mut seed)).collect();
        rows.push((format!("v-{i}"), "Vec", vec));
    }
    let ids = ingest_vectors(&ingest, tenant, &rows);
    // Ground-truth (internal_id, vector) pairs for the brute-force oracle.
    let truth: Vec<(u64, Vec<f32>)> = rows
        .iter()
        .map(|(ext, _, v)| (ids[ext], v.clone()))
        .collect();

    let provider: Arc<dyn SubstrateSearchProvider> =
        Arc::new(HnswVectorSearchProvider::new(backend.clone()));

    let l2_sq =
        |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum() };

    let mut total_recall = 0.0_f64;
    for _ in 0..QUERIES {
        let q: Vec<f32> = (0..DIM).map(|_| lcg(&mut seed)).collect();

        // Exact top-K oracle.
        let mut exact: Vec<(u64, f32)> = truth.iter().map(|(id, v)| (*id, l2_sq(&q, v))).collect();
        exact.sort_by(|a, b| a.1.total_cmp(&b.1));
        let exact_ids: std::collections::HashSet<u64> =
            exact.iter().take(K).map(|(id, _)| *id).collect();

        // Served HNSW top-K.
        let hits = provider
            .vector_search(tenant, "embedding", &q, K as u64, Lsn::MAX)
            .expect("vector_search");
        assert_eq!(
            hits.len(),
            K,
            "served HNSW must return K hits for N>K corpus"
        );
        let hit_ids: std::collections::HashSet<u64> =
            hits.iter().map(|h| h.node.id.raw()).collect();

        let overlap = exact_ids.intersection(&hit_ids).count();
        total_recall += overlap as f64 / K as f64;
    }
    let mean_recall = total_recall / QUERIES as f64;
    eprintln!(
        "#765 served HNSW recall@{K} over N={N} dim={DIM} queries={QUERIES}: {mean_recall:.4} (floor {RECALL_FLOOR})"
    );
    assert!(
        mean_recall >= RECALL_FLOOR,
        "served HNSW recall@{K} {mean_recall:.4} below floor {RECALL_FLOOR}",
    );
}

#[test]
fn vector_search_dimension_mismatch_is_structured_error_not_silent() {
    // ── #765 honesty gate: a query whose dimension differs from the index
    // dimension MUST surface a structured error, never a silent wrong/empty
    // result (per feedback_review_oracle_relaxations).
    let (backend, _guard) =
        bootstrap_storage_backend(&BootstrapMode::InMemory).expect("bootstrap in-memory");
    let tenant = TenantId::DEFAULT;
    let ingest = StorageIngestProvider::new(backend.clone());
    let rows = vec![
        ("a".to_string(), "Vec", vec![1.0_f32, 0.0, 0.0]),
        ("b".to_string(), "Vec", vec![0.0_f32, 1.0, 0.0]),
    ];
    let _ = ingest_vectors(&ingest, tenant, &rows);

    let provider = HnswVectorSearchProvider::new(backend.clone());
    // Index dim is 3; query with dim 4.
    let r = provider.vector_search(tenant, "embedding", &[1.0, 0.0, 0.0, 0.0], 5, Lsn::MAX);
    let err = r.expect_err("dimension mismatch must be an error, not Ok");
    let msg = format!("{err}");
    assert!(
        msg.contains("dimension"),
        "error must name the dimension mismatch; got: {msg}",
    );
}

#[test]
fn graph_search_empty_index_returns_no_hits_not_error() {
    // A tenant with no ingested vectors yet → graph.search returns an empty
    // ranked list (honest), never IndexUnavailable from the body and never a
    // spurious error.
    let (backend, _guard) =
        bootstrap_storage_backend(&BootstrapMode::InMemory).expect("bootstrap in-memory");
    let tenant = TenantId::DEFAULT;
    let provider: Arc<dyn SubstrateSearchProvider> =
        Arc::new(HnswVectorSearchProvider::new(backend.clone()));
    let searcher = StorageHybridSearcher::new(backend.clone()).with_search_provider(provider);
    let req = SearchRequest {
        tenant_id: tenant.raw(),
        query: String::new(),
        query_vec: Some(vec![0.1_f32, 0.2, 0.3]),
        k: Some(5),
        label_filter: None,
        ef_search: None,
        format: Some(ResponseFormat::Json),
        principal: None,
    };
    let token = CancellationToken::new();
    let resp = search_tool(&searcher, tenant, SessionScope::Power, &token, req)
        .expect("graph.search Ok on empty index");
    let body: serde_json::Value =
        serde_json::from_str(resp["body"].as_str().unwrap()).expect("parse body");
    assert_eq!(
        body["hits"].as_array().expect("hits array").len(),
        0,
        "empty index → no hits (honest empty, not an error)",
    );
}

#[test]
fn graph_search_vector_query_without_bound_provider_is_index_unavailable() {
    // ── #765 PART-1 defensive branch (adapters.rs `StorageHybridSearcher::search`
    // provider-unbound path): the vector substrate is ATTACHED at bootstrap
    // (`TenantHandle::vector().is_some()`, so the tools/search.rs availability
    // gate passes) and a `query_vec` IS supplied — but no `SubstrateSearchProvider`
    // was bound via `with_search_provider`. The body must surface a structured
    // -32004 (`IndexUnavailable`), never a panic or a silent-empty result.
    //
    // This is a DISTINCT path from the m4_08 / search.rs availability-gate
    // -32004 (vector substrate MISSING, slug == "vector"): here the gate passes
    // and we reach the provider-not-attached branch the #765 wiring added.
    // Production bootstrap always binds the provider, so this is defense in depth.
    let (backend, _guard) =
        bootstrap_storage_backend(&BootstrapMode::InMemory).expect("bootstrap in-memory");
    let tenant = TenantId::DEFAULT;

    // No `.with_search_provider(...)` — provider stays None.
    let searcher = StorageHybridSearcher::new(backend.clone());
    let req = SearchRequest {
        tenant_id: tenant.raw(),
        query: String::new(),
        query_vec: Some(vec![0.1_f32, 0.2, 0.3]),
        k: Some(5),
        label_filter: None,
        ef_search: None,
        format: Some(ResponseFormat::Json),
        principal: None,
    };
    let token = CancellationToken::new();
    let err = search_tool(&searcher, tenant, SessionScope::Power, &token, req)
        .expect_err("vector query with no bound provider must be an error, not Ok");
    assert_eq!(
        err.code(),
        CODE_INDEX_UNAVAILABLE,
        "provider-unbound vector search must map to -32004; got code {}",
        err.code(),
    );
    match err {
        MCPError::IndexUnavailable(msg) => assert!(
            msg.contains("provider not attached"),
            "must name the unbound provider (distinct from the availability-gate \
             \"vector\" slug); got: {msg}",
        ),
        other => panic!("expected IndexUnavailable, got {other:?}"),
    }
}

/// Ingest a single node with an arbitrary property bag (used by the #787 perf
/// harness to add a lone vector / non-vector node between searches).
fn ingest_one(
    ingest: &StorageIngestProvider,
    tenant: TenantId,
    ext: &str,
    label: &str,
    props: BTreeMap<String, serde_json::Value>,
) {
    let summary = ingest
        .ingest(
            tenant,
            IngestBatch {
                nodes: vec![NodeIngest {
                    external_id: Some(ext.to_string()),
                    label: label.to_string(),
                    properties: props,
                }],
                relationships: vec![],
                acl_grants: vec![],
            },
        )
        .expect("ingest one");
    assert_eq!(summary.failed_count, 0, "ingest_one must have 0 failures");
}

#[test]
fn perf_787_read_after_write_is_incremental_not_full_rebuild() {
    // ── #787: the read-after-write cliff. The per-tenant HNSW must NOT be
    // fully O(N)-rebuilt on the first vector query after an ingest. A vector
    // query that follows a +1-vector-node ingest should cost ~warm + a single
    // incremental insert; a query after a +1 NON-vector-node ingest must reuse
    // the index entirely (vector-aware invalidation).
    //
    // NB: the wall-clock prints below are for the PR writeup (HONEST before/after
    // numbers); the load-bearing assertions are the deterministic insert/scan
    // counters (added post-fix), NOT the timings — wall-clock thresholds flake
    // (see project_703_misdiagnosis_flaky_threshold_class).
    use std::time::Instant;

    const N: usize = 2000;
    const DIM: usize = 16;

    let (backend, _guard) =
        bootstrap_storage_backend(&BootstrapMode::InMemory).expect("bootstrap in-memory");
    let tenant = TenantId::DEFAULT;
    let ingest = StorageIngestProvider::new(backend.clone());

    let mut seed = 0x787_0787u64;
    let mut rows: Vec<(String, &str, Vec<f32>)> = Vec::with_capacity(N);
    for i in 0..N {
        let vec: Vec<f32> = (0..DIM).map(|_| lcg(&mut seed)).collect();
        rows.push((format!("v-{i}"), "Vec", vec));
    }
    let t_ingest = Instant::now();
    let _ids = ingest_vectors(&ingest, tenant, &rows);
    let ingest_ms = t_ingest.elapsed().as_secs_f64() * 1e3;

    let provider = HnswVectorSearchProvider::new(backend.clone());
    let q: Vec<f32> = (0..DIM).map(|_| lcg(&mut seed)).collect();

    // Cold first query → lazy build over all N (unavoidable; the build cost).
    let t_cold = Instant::now();
    let cold = provider
        .vector_search(tenant, "embedding", &q, 10, Lsn::MAX)
        .expect("cold search");
    let cold_ms = t_cold.elapsed().as_secs_f64() * 1e3;
    assert_eq!(cold.len(), 10, "cold search returns k hits");
    let m_cold = provider.metrics();

    // Warm query (no intervening write) → fast path: no scan, no insert.
    let t_warm = Instant::now();
    let _ = provider
        .vector_search(tenant, "embedding", &q, 10, Lsn::MAX)
        .expect("warm search");
    let warm_ms = t_warm.elapsed().as_secs_f64() * 1e3;
    let m_warm = provider.metrics();

    // Ingest +1 VECTOR node, then query → THE read-after-write measurement.
    let near: Vec<f32> = (0..DIM).map(|_| 0.01_f32).collect();
    ingest_one(&ingest, tenant, "extra-vec", "Vec", embedding_props(&near));
    let t_raw = Instant::now();
    let after_vec = provider
        .vector_search(tenant, "embedding", &q, 10, Lsn::MAX)
        .expect("search after +1 vector");
    let after_vec_ms = t_raw.elapsed().as_secs_f64() * 1e3;
    assert_eq!(after_vec.len(), 10, "search after +1 vector returns k hits");
    let m_vec = provider.metrics();

    // Ingest +1 NON-VECTOR node (no embedding), then query → must reuse index.
    ingest_one(&ingest, tenant, "memo-1", "Memo", BTreeMap::new());
    let t_nonvec = Instant::now();
    let after_nonvec = provider
        .vector_search(tenant, "embedding", &q, 10, Lsn::MAX)
        .expect("search after +1 non-vector");
    let after_nonvec_ms = t_nonvec.elapsed().as_secs_f64() * 1e3;
    assert_eq!(
        after_nonvec.len(),
        10,
        "search after +1 non-vector returns k hits"
    );
    let m_nonvec = provider.metrics();

    eprintln!(
        "#787 perf (N={N}, dim={DIM}): ingest={ingest_ms:.1}ms  cold={cold_ms:.1}ms  \
         warm={warm_ms:.3}ms  after_+1_vector={after_vec_ms:.1}ms  \
         after_+1_nonvector={after_nonvec_ms:.1}ms",
    );
    eprintln!(
        "#787 oracle: cold(ins={},scan={})  warm(ins={},scan={})  +1vec(ins={},scan={})  \
         +1nonvec(ins={},scan={})",
        m_cold.vectors_inserted,
        m_cold.nodes_scanned,
        m_warm.vectors_inserted,
        m_warm.nodes_scanned,
        m_vec.vectors_inserted,
        m_vec.nodes_scanned,
        m_nonvec.vectors_inserted,
        m_nonvec.nodes_scanned,
    );

    // ── STRONG ORACLE (deterministic; the load-bearing assertions — NOT the
    // wall-clock timings, which only feed the writeup). The cold build inserts
    // exactly N + scans exactly N. Each subsequent read-after-write must do
    // O(delta) work, never O(N): a regression to a full rebuild bumps these by
    // N and fails here.
    assert_eq!(
        m_cold.vectors_inserted, N as u64,
        "cold build inserts exactly N"
    );
    assert_eq!(m_cold.nodes_scanned, N as u64, "cold build scans exactly N");
    assert_eq!(
        m_warm, m_cold,
        "warm query must not scan or insert (fast path)"
    );
    assert_eq!(
        m_vec.vectors_inserted - m_cold.vectors_inserted,
        1,
        "search after +1 vector node inserts exactly 1 (incremental), not a rebuild of N",
    );
    assert_eq!(
        m_vec.nodes_scanned - m_cold.nodes_scanned,
        1,
        "search after +1 vector node delta-scans exactly 1 node (O(delta)), not O(N)",
    );
    assert_eq!(
        m_nonvec.vectors_inserted, m_vec.vectors_inserted,
        "a non-vector ingest must NOT insert into / rebuild the vector index (vector-aware)",
    );
    assert_eq!(
        m_nonvec.nodes_scanned - m_vec.nodes_scanned,
        1,
        "search after +1 non-vector node delta-scans exactly 1 node, then reuses the index",
    );
}

#[test]
fn ingest_rejects_mixed_embedding_dimensions_non_silently() {
    // ── #786: mixed embedding dims in ONE ingest batch must NOT be silently
    // accepted-then-dropped. The first embedding establishes the batch dim; a
    // differing-dim node is rejected with failed_count + a CLEAR per-record
    // reason (the original bug returned inserted=3, failed=0 — silent).
    use arcgraph_mcp::tools::ingest::IngestError;

    let (backend, _guard) =
        bootstrap_storage_backend(&BootstrapMode::InMemory).expect("bootstrap in-memory");
    let tenant = TenantId::DEFAULT;
    let ingest = StorageIngestProvider::new(backend.clone());

    // The issue's exact deterministic repro: 4-dim node FIRST, then two 3-dim.
    let batch = IngestBatch {
        nodes: vec![
            NodeIngest {
                external_id: Some("a".into()),
                label: "Account".into(),
                properties: embedding_props(&[1.0, 1.0, 0.0, 0.0]), // 4-dim
            },
            NodeIngest {
                external_id: Some("b".into()),
                label: "Bug".into(),
                properties: embedding_props(&[1.0, 0.0, 0.0]), // 3-dim
            },
            NodeIngest {
                external_id: Some("c".into()),
                label: "Bug".into(),
                properties: embedding_props(&[0.9, 0.1, 0.0]), // 3-dim
            },
        ],
        relationships: vec![],
        acl_grants: vec![],
    };
    let summary = ingest.ingest(tenant, batch).expect("ingest");

    // NON-SILENT: the two 3-dim nodes are rejected (NOT inserted=3, failed=0).
    assert_eq!(
        summary.inserted_count, 1,
        "only the 4-dim node (a) is inserted"
    );
    assert_eq!(
        summary.failed_count, 2,
        "the two 3-dim nodes are rejected, not silently dropped (the #786 symptom)",
    );

    // Each rejection names the dimension mismatch with a clear, actionable reason.
    let mut failures = 0;
    for rec in &summary.records {
        if let IngestRecordOutcome::Failed { external_id, error } = rec {
            failures += 1;
            let ext = external_id.clone().unwrap_or_default();
            assert!(ext == "b" || ext == "c", "only b/c fail; got `{ext}`");
            match error {
                IngestError::Invalid { detail } => {
                    assert!(
                        detail.contains("dimension")
                            && detail.contains('3')
                            && detail.contains('4'),
                        "reason must name the dim mismatch (3 vs established 4); got: {detail}",
                    );
                    eprintln!("#786 ingest rejection ({ext}): {detail}");
                }
                other => panic!("expected Invalid dim-mismatch, got {other:?}"),
            }
        }
    }
    assert_eq!(failures, 2, "exactly two rejections");
}

#[test]
fn vector_search_wrong_dim_query_maps_to_invalid_params_not_execution_eval() {
    // ── #786: a wrong-dimension query_vec through the FULL graph.search MCP
    // path must surface a CLEAR -32602 invalid-params error naming the dims,
    // NOT the cryptic -32006 "execution eval" the first cut returned.
    use arcgraph_mcp::CODE_INVALID_PARAMS;

    let (backend, _guard) =
        bootstrap_storage_backend(&BootstrapMode::InMemory).expect("bootstrap in-memory");
    let tenant = TenantId::DEFAULT;
    let ingest = StorageIngestProvider::new(backend.clone());
    // Establish a 3-dim index.
    let rows = vec![
        ("a".to_string(), "Vec", vec![1.0_f32, 0.0, 0.0]),
        ("b".to_string(), "Vec", vec![0.0_f32, 1.0, 0.0]),
    ];
    let _ = ingest_vectors(&ingest, tenant, &rows);

    let provider: Arc<dyn SubstrateSearchProvider> =
        Arc::new(HnswVectorSearchProvider::new(backend.clone()));
    let searcher = StorageHybridSearcher::new(backend.clone()).with_search_provider(provider);

    // Query with a 2-dim vector (≠ index dim 3) — the issue's wrong-dim query.
    let req = SearchRequest {
        tenant_id: tenant.raw(),
        query: String::new(),
        query_vec: Some(vec![1.0_f32, 0.0]),
        k: Some(5),
        label_filter: None,
        ef_search: None,
        format: Some(ResponseFormat::Json),
        principal: None,
    };
    let token = CancellationToken::new();
    let err = search_tool(&searcher, tenant, SessionScope::Power, &token, req)
        .expect_err("wrong-dim query must be an error, not Ok");

    assert_eq!(
        err.code(),
        CODE_INVALID_PARAMS,
        "wrong-dim query_vec must map to -32602 invalid params, not -32006; got {}",
        err.code(),
    );
    match err {
        MCPError::InvalidParams(msg) => {
            assert!(
                msg.contains("dimension 2") && msg.contains("dimension 3"),
                "message must name the dim mismatch (query 2 vs index 3); got: {msg}",
            );
            eprintln!("#786 wrong-dim query error: -32602 {msg}");
        }
        other => panic!("expected InvalidParams, got {other:?}"),
    }
}

#[test]
fn graph_search_filtered_knn_does_not_collapse_under_selective_filter_815() {
    // ── #815 end-to-end: 500 vectors on the line [i,0,0]; every 10th is
    // :Rare (10 % selectivity); query [250,0,0]. POST-filtering the
    // unfiltered top-10 returns just ONE :Rare (the recall collapse); the
    // filter-during-search path (graph.search label_filter) returns the
    // TRUE 10 nearest :Rare. STRONG oracle: the exact internal_id set, by
    // brute force over only the :Rare-labelled vectors.
    let (backend, _guard) =
        bootstrap_storage_backend(&BootstrapMode::InMemory).expect("bootstrap in-memory");
    let tenant = TenantId::DEFAULT;
    let ingest = StorageIngestProvider::new(backend.clone());

    const N: u32 = 500;
    let rows: Vec<(String, &str, Vec<f32>)> = (0..N)
        .map(|i| {
            let label = if i % 10 == 0 { "Rare" } else { "Common" };
            (format!("v-{i}"), label, vec![i as f32, 0.0, 0.0])
        })
        .collect();
    let ids = ingest_vectors(&ingest, tenant, &rows);

    // Brute-force truth over ONLY the :Rare vectors: the 10 multiples of
    // 10 nearest 250 = {250,240,260,230,270,220,280,210,290,200}.
    let mut rare_i: Vec<u32> = (0..N).filter(|i| i % 10 == 0).collect();
    rare_i.sort_by_key(|i| (i64::from(*i) - 250).abs());
    let truth: std::collections::HashSet<u64> = rare_i
        .iter()
        .take(10)
        .map(|i| ids[&format!("v-{i}")])
        .collect();
    assert_eq!(truth.len(), 10);
    let rare_ids: std::collections::HashSet<u64> = (0..N)
        .filter(|i| i % 10 == 0)
        .map(|i| ids[&format!("v-{i}")])
        .collect();

    let provider: Arc<dyn SubstrateSearchProvider> =
        Arc::new(HnswVectorSearchProvider::new(backend.clone()));
    let searcher =
        StorageHybridSearcher::new(backend.clone()).with_search_provider(Arc::clone(&provider));
    let token = CancellationToken::new();
    let q = vec![250.0_f32, 0.0, 0.0];

    // ── BEFORE (post-filter): unfiltered top-10 → keep only :Rare → 1/10.
    let unfiltered = provider
        .vector_search(tenant, "embedding", &q, 10, Lsn::MAX)
        .expect("unfiltered search");
    let before: Vec<u64> = unfiltered
        .iter()
        .map(|h| h.node.id.raw())
        .filter(|id| rare_ids.contains(id))
        .collect();
    eprintln!(
        "#815 BEFORE (post-filter over unfiltered top-10): {} of 10 Rare = {before:?}",
        before.len()
    );
    assert_eq!(
        before.len(),
        1,
        "post-filter over the unfiltered top-10 collapses to 1 Rare (the #815 bug)"
    );

    // ── AFTER (filter-during-search via graph.search label_filter): 10/10.
    let req = SearchRequest {
        tenant_id: tenant.raw(),
        query: String::new(),
        query_vec: Some(q.clone()),
        k: Some(10),
        label_filter: Some(vec!["Rare".into()]),
        ef_search: None,
        format: Some(ResponseFormat::Json),
        principal: None,
    };
    let resp =
        search_tool(&searcher, tenant, SessionScope::Power, &token, req).expect("graph.search Ok");
    let body: serde_json::Value =
        serde_json::from_str(resp["body"].as_str().expect("body string")).expect("parse body");
    let hits = body["hits"].as_array().expect("hits array");
    let after: std::collections::HashSet<u64> = hits
        .iter()
        .map(|h| h["node_id"].as_u64().expect("node_id u64"))
        .collect();
    eprintln!(
        "#815 AFTER (filter-during-search, k=10, label_filter=[Rare]): {} of 10 Rare",
        after.len()
    );
    assert_eq!(
        after.len(),
        10,
        "filtered KNN must return k=10 Rare hits, not collapse to ~1"
    );
    assert_eq!(
        after, truth,
        "filter-during-search must return the TRUE 10 nearest Rare"
    );
    for h in hits {
        assert_eq!(
            h["label"].as_str(),
            Some("Rare"),
            "no non-Rare leakage into the filtered result"
        );
    }

    // ── The issue's "need k≈1/selectivity": post-filter only recovers the
    //    true 10 by inflating k to ~100.
    let wide = provider
        .vector_search(tenant, "embedding", &q, 100, Lsn::MAX)
        .expect("wide search");
    let wide_rare: std::collections::HashSet<u64> = wide
        .iter()
        .map(|h| h.node.id.raw())
        .filter(|id| rare_ids.contains(id))
        .collect();
    assert!(
        truth.is_subset(&wide_rare),
        "post-filter needs k≈100 (1/selectivity) to recover what filtered-KNN gets at k=10"
    );
}

#[test]
fn graph_search_filtered_knn_holds_recall_at_one_percent_selectivity_815() {
    // ── #815 more-selective case (~1 %): 2000 vectors on the line [i,0,0];
    // every 100th is :Rare (20 of 2000 = 1 %); query [1000,0,0], k=10. The
    // filter-during-search path must STILL return k with recall@k ≥ 0.95
    // vs an exact brute-force oracle over only the :Rare vectors — a
    // selective filter must NOT collapse recall.
    let (backend, _guard) =
        bootstrap_storage_backend(&BootstrapMode::InMemory).expect("bootstrap in-memory");
    let tenant = TenantId::DEFAULT;
    let ingest = StorageIngestProvider::new(backend.clone());

    const N: u32 = 2000;
    const K: usize = 10;
    let rows: Vec<(String, &str, Vec<f32>)> = (0..N)
        .map(|i| {
            let label = if i % 100 == 0 { "Rare" } else { "Common" };
            (format!("v-{i}"), label, vec![i as f32, 0.0, 0.0])
        })
        .collect();
    let ids = ingest_vectors(&ingest, tenant, &rows);

    // Exact truth over the 20 :Rare (multiples of 100) nearest 1000.
    let mut rare_i: Vec<u32> = (0..N).filter(|i| i % 100 == 0).collect();
    assert!(rare_i.len() > K, "need > K Rare for a real recall@K");
    rare_i.sort_by_key(|i| (i64::from(*i) - 1000).abs());
    let truth: std::collections::HashSet<u64> = rare_i
        .iter()
        .take(K)
        .map(|i| ids[&format!("v-{i}")])
        .collect();

    let provider: Arc<dyn SubstrateSearchProvider> =
        Arc::new(HnswVectorSearchProvider::new(backend.clone()));
    let rare_label = backend.intern_table().intern_label(tenant, "Rare").unwrap();
    let q = vec![1000.0_f32, 0.0, 0.0];

    let hits = provider
        .vector_search_filtered(
            tenant,
            "embedding",
            &q,
            K as u64,
            Some(&[rare_label]),
            None,
            Lsn::MAX,
        )
        .expect("filtered search");
    assert_eq!(hits.len(), K, "selective filter must still return k hits");
    let got: std::collections::HashSet<u64> = hits.iter().map(|h| h.node.id.raw()).collect();
    let recall = truth.intersection(&got).count() as f64 / K as f64;
    eprintln!("#815 filtered recall@{K} @1%-selectivity (N={N}): {recall:.4}");
    assert!(
        recall >= 0.95,
        "selective (1 %) filter must hold recall@{K} ≥ 0.95, got {recall:.4}"
    );
}

#[test]
fn graph_search_ef_search_knob_accepted_and_monotone_816a() {
    // ── #816a: graph.search {ef_search:N} is ACCEPTED (no -32602 unknown
    // field) and the recall-vs-ef curve is client-controllable: higher
    // ef_search → recall ≥ lower (monotone). Omitted ef_search == the
    // engine default (back-compat, exact). Absurd values reject gracefully.
    const N: usize = 1200;
    const DIM: usize = 96;
    const QUERIES: usize = 20;
    const K: usize = 10;
    let (backend, _guard) =
        bootstrap_storage_backend(&BootstrapMode::InMemory).expect("bootstrap in-memory");
    let tenant = TenantId::DEFAULT;
    let ingest = StorageIngestProvider::new(backend.clone());

    let mut seed = 0x8160_0A16u64;
    let rows: Vec<(String, &str, Vec<f32>)> = (0..N)
        .map(|i| {
            (
                format!("v-{i}"),
                "Vec",
                (0..DIM).map(|_| lcg(&mut seed)).collect::<Vec<f32>>(),
            )
        })
        .collect();
    let ids = ingest_vectors(&ingest, tenant, &rows);
    let truth_vecs: Vec<(u64, Vec<f32>)> =
        rows.iter().map(|(e, _, v)| (ids[e], v.clone())).collect();

    let provider: Arc<dyn SubstrateSearchProvider> =
        Arc::new(HnswVectorSearchProvider::new(backend.clone()));
    let l2 =
        |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum() };

    // Mean recall@K at a given ef over the SAME deterministic query set
    // (reset the query seed each call so the comparison across ef is fair).
    let mean_recall = |ef: Option<usize>| -> f64 {
        let mut s = 0x0FF1_CE16u64;
        let mut total = 0.0_f64;
        for _ in 0..QUERIES {
            let q: Vec<f32> = (0..DIM).map(|_| lcg(&mut s)).collect();
            let mut exact: Vec<(u64, f32)> =
                truth_vecs.iter().map(|(id, v)| (*id, l2(&q, v))).collect();
            exact.sort_by(|a, b| a.1.total_cmp(&b.1));
            let truth: std::collections::HashSet<u64> =
                exact.iter().take(K).map(|(id, _)| *id).collect();
            let hits = provider
                .vector_search_filtered(tenant, "embedding", &q, K as u64, None, ef, Lsn::MAX)
                .expect("vector_search_filtered");
            let got: std::collections::HashSet<u64> =
                hits.iter().map(|h| h.node.id.raw()).collect();
            total += truth.intersection(&got).count() as f64 / K as f64;
        }
        total / QUERIES as f64
    };

    let efs = [10usize, 50, 256];
    let recalls: Vec<f64> = efs.iter().map(|&e| mean_recall(Some(e))).collect();
    eprintln!("#816a recall-vs-ef_search curve (ef → recall@{K}):");
    for (e, r) in efs.iter().zip(&recalls) {
        eprintln!("  ef_search={e:>4} → recall {r:.4}");
    }
    // Monotone non-decreasing in ef (deterministic corpus + queries).
    for w in recalls.windows(2) {
        assert!(
            w[1] >= w[0] - 1e-9,
            "recall must be non-decreasing in ef_search: {recalls:?}"
        );
    }
    // The knob materially bites: ef=256 strictly beats ef=1 (the entire
    // ann-benchmarks recall-vs-QPS axis the issue asks for).
    assert!(
        recalls[recalls.len() - 1] > recalls[0] + 0.05,
        "higher ef_search must materially raise recall: {recalls:?}"
    );
    assert!(
        recalls[recalls.len() - 1] >= 0.90,
        "high ef_search should recover strong recall: {recalls:?}"
    );

    // ── Back-compat: omitted ef_search == the engine default (ef internally
    //    0 → HnswParams::ef_search = 128). EXACT equality (same graph, same
    //    beam → identical results).
    let default_recall = mean_recall(None);
    let ef128_recall = mean_recall(Some(128));
    assert!(
        (default_recall - ef128_recall).abs() < 1e-9,
        "omitted ef_search must equal the engine default (128): {default_recall} vs {ef128_recall}"
    );

    // ── Acceptance + graceful validation through the MCP graph.search tool.
    let searcher =
        StorageHybridSearcher::new(backend.clone()).with_search_provider(Arc::clone(&provider));
    let token = CancellationToken::new();
    let mut s = 7u64;
    let q0: Vec<f32> = (0..DIM).map(|_| lcg(&mut s)).collect();
    let mk = |ef: Option<u32>| SearchRequest {
        tenant_id: tenant.raw(),
        query: String::new(),
        query_vec: Some(q0.clone()),
        k: Some(K as u32),
        label_filter: None,
        ef_search: ef,
        format: Some(ResponseFormat::Json),
        principal: None,
    };
    // ef_search:256 ACCEPTED — the pre-#816a `-32602 unknown field` is gone.
    let ok = search_tool(
        &searcher,
        tenant,
        SessionScope::Power,
        &token,
        mk(Some(256)),
    )
    .expect("graph.search {ef_search:256} must be accepted (no -32602)");
    assert!(
        ok["body"].as_str().is_some(),
        "ef_search:256 returns a body"
    );
    eprintln!("#816a graph.search {{ef_search:256}} accepted (no -32602 unknown field)");
    // ef_search:0 rejects gracefully (InvalidParams, not a panic).
    let e0 = search_tool(&searcher, tenant, SessionScope::Power, &token, mk(Some(0)))
        .expect_err("ef_search:0 must reject");
    assert!(
        matches!(e0, MCPError::InvalidParams(_)),
        "ef_search:0 → InvalidParams (graceful), got {e0:?}"
    );
    // ef_search above the cap rejects gracefully.
    let ehi = search_tool(
        &searcher,
        tenant,
        SessionScope::Power,
        &token,
        mk(Some(MAX_SEARCH_EF + 1)),
    )
    .expect_err("ef_search>cap must reject");
    assert!(
        matches!(ehi, MCPError::InvalidParams(_)),
        "ef_search>{MAX_SEARCH_EF} → InvalidParams (graceful), got {ehi:?}"
    );
}
