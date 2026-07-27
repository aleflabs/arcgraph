//! #903 — `CrudExecutorSubstrate::bm25_search` stored-property + label
//! hydration over the real BM25 engine.
//!
//! There is no `db.index.fulltext.queryNodes` proc wired at v1.0, and the
//! production CLI `HnswVectorSearchProvider` intentionally returns
//! `IndexUnavailable` for BM25. This test therefore drives the layer that does
//! reach `CrudExecutorSubstrate::bm25_search`: a test-only
//! `SubstrateSearchProvider` adapter over the real Tantivy-backed
//! `arcgraph_bm25::Bm25Service`.
//!
//! The fixture ingests real nodes into the store through
//! `StorageIngestProvider`, explicitly publishes the same documents into the
//! real BM25 service per the existing v1.0 BM25 commit posture, and then
//! compares raw-provider hits against substrate-hydrated hits.

use std::collections::BTreeMap;
use std::sync::Arc;

use arcgraph_bm25::{Bm25Service, IndexId};
use arcgraph_core::{LabelId, Lsn, NodeId, TenantId};
use arcgraph_mcp::storage::{
    CrudExecutorSubstrate, StorageBackend, StorageIngestProvider, SubstrateSearchProvider,
};
use arcgraph_mcp::tools::ingest::{IngestBatch, IngestProvider, IngestRecordOutcome, NodeIngest};
use arcgraph_query::executor::substrate::{ExecutorSubstrate, RankedHit, SubstrateAccessError};
use arcgraph_query::executor::value::{NodeView, Value};
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

#[derive(Clone, Debug)]
struct RealBm25Provider {
    service: Arc<Bm25Service>,
    labels: Arc<BTreeMap<NodeId, Option<LabelId>>>,
}

impl SubstrateSearchProvider for RealBm25Provider {
    fn vector_search(
        &self,
        _tenant: TenantId,
        _property: &str,
        _query_vec: &[f32],
        _k: u64,
        _read_lsn: Lsn,
    ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
        Err(SubstrateAccessError::IndexUnavailable(
            "vector not used by #903 bm25 hydration e2e".into(),
        ))
    }

    fn bm25_search(
        &self,
        tenant: TenantId,
        _property: &str,
        query_text: &str,
        k: u64,
        read_lsn: Lsn,
    ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
        let handle = self
            .service
            .handle(tenant, IndexId::DEFAULT_BM25)
            .map_err(|e| SubstrateAccessError::Io(format!("bm25 handle: {e}")))?;
        let hits = handle
            .search(query_text, k as usize, read_lsn)
            .map_err(|e| SubstrateAccessError::Io(format!("bm25 search: {e}")))?;
        Ok(hits
            .into_iter()
            .map(|(node_id, score)| RankedHit {
                node: NodeView::new(node_id, self.labels.get(&node_id).copied().flatten()),
                score: f64::from(score),
            })
            .collect())
    }
}

struct TestStack {
    _tmp: TempDir,
    backend: StorageBackend,
    bm25: Arc<Bm25Service>,
}

fn bm25_backend() -> TestStack {
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
    TestStack {
        _tmp: tmp,
        backend: StorageBackend::new(
            router,
            txn_manager,
            Arc::new(arcgraph_storage::InternTable::new()),
        ),
        bm25,
    }
}

struct Doc {
    external_id: &'static str,
    label: &'static str,
    title: &'static str,
    text: &'static str,
    rank_hint: i64,
}

fn doc_props(d: &Doc) -> BTreeMap<String, serde_json::Value> {
    let mut m = BTreeMap::new();
    m.insert(
        "id".to_string(),
        serde_json::Value::String(d.external_id.to_string()),
    );
    m.insert(
        "title".to_string(),
        serde_json::Value::String(d.title.to_string()),
    );
    m.insert(
        "text".to_string(),
        serde_json::Value::String(d.text.to_string()),
    );
    m.insert(
        "rank_hint".to_string(),
        serde_json::Value::Number(serde_json::Number::from(d.rank_hint)),
    );
    m
}

fn corpus() -> Vec<Doc> {
    vec![
        Doc {
            external_id: "doc-1",
            label: "Doc",
            title: "Needle",
            text: "needle needle arcgraph hydration",
            rank_hint: 1,
        },
        Doc {
            external_id: "doc-2",
            label: "Doc",
            title: "Single Needle",
            text: "needle arcgraph search",
            rank_hint: 2,
        },
        Doc {
            external_id: "doc-3",
            label: "Doc",
            title: "Hay",
            text: "haystack storage vector",
            rank_hint: 3,
        },
    ]
}

fn ingest_docs(
    ingest: &StorageIngestProvider,
    tenant: TenantId,
    docs: &[Doc],
) -> BTreeMap<String, u64> {
    let nodes = docs
        .iter()
        .map(|d| NodeIngest {
            external_id: Some(d.external_id.to_string()),
            label: d.label.to_string(),
            properties: doc_props(d),
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
        .expect("ingest docs");
    assert_eq!(summary.failed_count, 0, "ingest must have 0 failures");
    let mut out = BTreeMap::new();
    for rec in &summary.records {
        if let IngestRecordOutcome::Inserted {
            internal_id,
            external_id,
        } = rec
        {
            out.insert(external_id.clone().unwrap_or_default(), *internal_id);
        }
    }
    out
}

fn seed_bm25(bm25: &Arc<Bm25Service>, tenant: TenantId, docs: &[Doc], ids: &BTreeMap<String, u64>) {
    let handle = bm25
        .handle(tenant, IndexId::DEFAULT_BM25)
        .expect("bm25 handle");
    for (offset, doc) in docs.iter().enumerate() {
        handle
            .upsert_document(
                NodeId::new(ids[doc.external_id]),
                doc.text,
                Lsn::new((offset + 1) as u64),
            )
            .expect("bm25 upsert");
    }
    let trait_obj: Arc<dyn Bm25IndexStoreHandle> = bm25.clone();
    trait_obj.commit_pending(tenant).expect("bm25 commit");
}

fn substrate_with_provider(
    backend: &StorageBackend,
    provider: Arc<dyn SubstrateSearchProvider>,
) -> CrudExecutorSubstrate {
    CrudExecutorSubstrate::new(
        Arc::clone(backend.router()),
        Arc::clone(backend.txn_manager()),
        Arc::clone(backend.intern_table()),
    )
    .with_search_provider(provider)
}

fn doc_by_id<'a>(internal_id: u64, ids: &BTreeMap<String, u64>, docs: &'a [Doc]) -> &'a Doc {
    let external_id = ids
        .iter()
        .find(|(_, id)| **id == internal_id)
        .map(|(external_id, _)| external_id.as_str())
        .unwrap_or_else(|| panic!("internal id {internal_id} not found"));
    docs.iter()
        .find(|d| d.external_id == external_id)
        .unwrap_or_else(|| panic!("external id {external_id} not in corpus"))
}

#[test]
fn substrate_bm25_search_hydrates_stored_props_and_preserves_rank_903() {
    let stack = bm25_backend();
    let tenant = TenantId::DEFAULT;
    let ingest = StorageIngestProvider::new(stack.backend.clone());
    let docs = corpus();
    let ids = ingest_docs(&ingest, tenant, &docs);
    seed_bm25(&stack.bm25, tenant, &docs, &ids);

    let labels = ids
        .values()
        .map(|id| (NodeId::new(*id), Some(LabelId::new(1))))
        .collect();
    let provider = Arc::new(RealBm25Provider {
        service: Arc::clone(&stack.bm25),
        labels: Arc::new(labels),
    });

    let baseline = SubstrateSearchProvider::bm25_search(
        provider.as_ref(),
        tenant,
        "text",
        "needle",
        2,
        Lsn::new(100),
    )
    .expect("raw bm25 provider search");
    assert_eq!(baseline.len(), 2, "two needle docs, k=2 -> two hits");
    for hit in &baseline {
        assert!(
            hit.node.properties.is_empty(),
            "raw BM25 provider emits empty property bags (bug precondition)"
        );
        assert!(
            hit.node.label_name.is_none(),
            "raw BM25 provider emits unresolved label names (bug precondition)"
        );
    }
    assert!(
        baseline[0].score >= baseline[1].score,
        "raw BM25 scores must be descending: {:?}",
        baseline.iter().map(|h| h.score).collect::<Vec<_>>()
    );

    let sub = substrate_with_provider(
        &stack.backend,
        Arc::clone(&provider) as Arc<dyn SubstrateSearchProvider>,
    );
    let hydrated = ExecutorSubstrate::bm25_search(&sub, tenant, "text", "needle", 2, Lsn::new(100))
        .expect("hydrated substrate bm25 search");

    assert_eq!(
        hydrated.len(),
        baseline.len(),
        "hydration must not change the hit count"
    );
    for (i, (raw, hyd)) in baseline.iter().zip(&hydrated).enumerate() {
        assert_eq!(
            hyd.node.id, raw.node.id,
            "hit[{i}] id + rank order must be identical pre/post hydration",
        );
        assert_eq!(
            hyd.score.to_bits(),
            raw.score.to_bits(),
            "hit[{i}] score must be bit-identical pre/post hydration",
        );
        assert_eq!(
            hyd.node.label, raw.node.label,
            "hit[{i}] label id must be unchanged",
        );

        let doc = doc_by_id(raw.node.id.raw(), &ids, &docs);
        assert_eq!(
            hyd.node.properties.get("id"),
            Some(&Value::String(doc.external_id.to_string())),
            "hit[{i}] external id property must hydrate",
        );
        assert_eq!(
            hyd.node.properties.get("title"),
            Some(&Value::String(doc.title.to_string())),
            "hit[{i}] title property must hydrate",
        );
        assert_eq!(
            hyd.node.properties.get("text"),
            Some(&Value::String(doc.text.to_string())),
            "hit[{i}] text property must hydrate",
        );
        assert_eq!(
            hyd.node.properties.get("rank_hint"),
            Some(&Value::Integer(doc.rank_hint)),
            "hit[{i}] numeric property must hydrate",
        );
        assert_eq!(
            hyd.node.label_name.as_deref(),
            Some("Doc"),
            "hit[{i}] label name must reverse-resolve to Doc",
        );
    }
}
