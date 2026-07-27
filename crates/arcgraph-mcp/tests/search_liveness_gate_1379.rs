//! #1379 (MUST-CON-04) — belt-and-suspenders liveness gate on the served
//! `graph.search` retrieval path.
//!
//! # What this pins
//!
//! `StorageHybridSearcher::search_filtered` runs a `read_node`-liveness
//! gate over its ranked candidates: a candidate whose MVCC record is no
//! longer LIVE (tombstoned by a committed delete) is DROPPED before the
//! response is assembled. This is defense-in-depth for the exact #1379
//! leak class — even if the ACL revoke is somehow missed (a stale index,
//! a routing miss on the revoke path), a DELETED node is not returned by
//! search.
//!
//! # The scenario (real backend, delete WITHOUT revoke)
//!
//! Ingest two real nodes into a real `CrudStore` (so their MVCC records
//! exist), then DELETE one via `crud::delete_node_with_store` — the OLD
//! path that does NOT revoke the ACL. A canned `SubstrateSearchProvider`
//! still returns BOTH ids as candidates (modeling a lazily-tombstoned
//! BM25 / vector index that hasn't scrubbed the deleted id). The liveness
//! gate must drop the deleted node and keep the live one.
//!
//! RED-on-revert: delete the `retain_live_hits` call in
//! `StorageHybridSearcher::search_filtered` and this test observes the
//! deleted node id in the results ⇒ FAILS.

use std::sync::Arc;

use arcgraph_core::{LabelId, Lsn, NodeId, PartitionId, TenantId};
use arcgraph_mcp::SessionScope;
use arcgraph_mcp::storage::substrate::SubstrateSearchProvider;
use arcgraph_mcp::storage::{StorageBackend, StorageHybridSearcher};
use arcgraph_mcp::tools::ResponseFormat;
use arcgraph_mcp::tools::search::{SearchRequest, search_tool};
use arcgraph_query::CancellationToken;
use arcgraph_query::executor::substrate::{RankedHit, SubstrateAccessError};
use arcgraph_query::executor::value::NodeView;
use arcgraph_storage::InternTable;
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::catalog::SystemCatalog;
use arcgraph_storage::crud::{self, CrudStore, PropertyData};
use arcgraph_storage::io::InMemoryPageIo;
use arcgraph_storage::mutation_log::{Bm25IndexStoreHandle, Bm25StoreError};
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::PrimaryIndex;
use arcgraph_storage::router::MultiTenantRouter;
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::vector_store::{VectorPageStoreHandle, VectorStoreError};

const TENANT: TenantId = TenantId::DEFAULT;

#[derive(Debug)]
struct NoopVectorStore;
impl VectorPageStoreHandle for NoopVectorStore {
    fn install_or_replace(
        &self,
        _tenant: TenantId,
        _page_id: arcgraph_core::PageId,
        _bytes: &[u8],
    ) -> Result<(), VectorStoreError> {
        Ok(())
    }
    fn restore_page_bytes(
        &self,
        _tenant: TenantId,
        _page_id: arcgraph_core::PageId,
        _bytes: &[u8],
    ) -> Result<(), VectorStoreError> {
        Ok(())
    }
}

#[derive(Debug)]
struct NoopBm25Store;
impl Bm25IndexStoreHandle for NoopBm25Store {
    fn commit_pending(&self, _tenant: TenantId) -> Result<(), Bm25StoreError> {
        Ok(())
    }
    fn rollback_pending(&self, _tenant: TenantId) -> Result<(), Bm25StoreError> {
        Ok(())
    }
}

/// A `SubstrateSearchProvider` that returns a FIXED list of candidate ids
/// regardless of the query — models a stale / lazily-tombstoned index
/// that still surfaces a deleted node's id. BM25 leg returns the same
/// (the served path fuses whichever legs are present).
#[derive(Debug)]
struct CannedProvider {
    ids: Vec<u64>,
}

impl CannedProvider {
    fn ranked(&self, k: u64) -> Vec<RankedHit> {
        self.ids
            .iter()
            .take(usize::try_from(k).unwrap_or(usize::MAX))
            .enumerate()
            .map(|(i, id)| RankedHit {
                node: NodeView::new(NodeId::new(*id), Some(LabelId::new(1))).with_label_name("Doc"),
                // Descending scores, deterministic.
                score: 1.0 - (i as f64) * 0.01,
            })
            .collect()
    }
}

impl SubstrateSearchProvider for CannedProvider {
    fn vector_search(
        &self,
        _tenant: TenantId,
        _property: &str,
        _query_vec: &[f32],
        k: u64,
        _read_lsn: Lsn,
    ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
        Ok(self.ranked(k))
    }

    fn bm25_search(
        &self,
        _tenant: TenantId,
        _property: &str,
        _query_text: &str,
        k: u64,
        _read_lsn: Lsn,
    ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
        Ok(self.ranked(k))
    }
}

fn fresh_backend() -> StorageBackend {
    let io = Arc::new(InMemoryPageIo::new());
    let pool = BufferPool::new(64, io);
    let mgr = Arc::new(TxnManager::new());
    let catalog = Arc::new(SystemCatalog::new());
    catalog.bootstrap(&pool, &mgr).expect("catalog bootstrap");
    let allocator = Arc::new(PageAllocator::new());
    let primary = Arc::new(
        PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&allocator), None).expect("PrimaryIndex"),
    );
    let crud = Arc::new(CrudStore::new_with_index(None, primary, allocator));
    let router = Arc::new(MultiTenantRouter::new_with_bm25(
        catalog,
        crud,
        Some(Arc::new(NoopVectorStore)),
        Some(Arc::new(NoopBm25Store)),
    ));
    let intern = Arc::new(InternTable::new());
    StorageBackend::new(router, mgr, intern)
}

/// Seed a real node in the backend's `CrudStore` (routes through the
/// tenant handle exactly like the production write path) and return its
/// internal id.
fn seed_node(backend: &StorageBackend, label: u32) -> NodeId {
    let handle = backend
        .router()
        .route(TENANT, PartitionId::ZERO)
        .expect("route tenant");
    let crud = handle.crud();
    let mut tx = backend.txn_manager().begin(TENANT);
    let id = crud::create_node(
        crud,
        &mut tx,
        TENANT,
        LabelId::new(label),
        &PropertyData::InlineU32Pair(1, 2),
    )
    .expect("create_node");
    crud::commit(tx, crud).expect("commit seed");
    id
}

/// Tombstone a node WITHOUT revoking its ACL — the pre-#1379 delete path,
/// used to isolate the liveness belt from the ACL revoke.
fn delete_node_no_revoke(backend: &StorageBackend, id: NodeId) {
    let handle = backend
        .router()
        .route(TENANT, PartitionId::ZERO)
        .expect("route tenant");
    let crud = handle.crud();
    let mut tx = backend.txn_manager().begin(TENANT);
    crud::delete_node_with_store(crud, &mut tx, id).expect("delete_node_with_store");
    crud::commit(tx, crud).expect("commit delete");
}

fn search_ids(searcher: &StorageHybridSearcher) -> Vec<u64> {
    let token = CancellationToken::new();
    let req = SearchRequest {
        tenant_id: TENANT.raw(),
        query: "anything".to_string(),
        query_vec: Some(vec![1.0_f32, 0.0, 0.0, 0.0]),
        k: Some(10),
        label_filter: None,
        ef_search: None,
        format: Some(ResponseFormat::Json),
        principal: None,
    };
    let resp = search_tool(searcher, TENANT, SessionScope::Power, &token, req).expect("search ok");
    let body: serde_json::Value =
        serde_json::from_str(resp["body"].as_str().expect("body string")).expect("parse body");
    body["hits"]
        .as_array()
        .expect("hits array")
        .iter()
        .map(|h| h["node_id"].as_u64().expect("node_id"))
        .collect()
}

#[test]
fn search_filtered_drops_deleted_node_via_liveness_gate() {
    let backend = fresh_backend();

    // Two real nodes; the provider will surface BOTH as candidates.
    let live = seed_node(&backend, 1);
    let doomed = seed_node(&backend, 1);

    let provider: Arc<dyn SubstrateSearchProvider> = Arc::new(CannedProvider {
        ids: vec![live.raw(), doomed.raw()],
    });
    let searcher = StorageHybridSearcher::new(backend.clone()).with_search_provider(provider);

    // Pre-delete: BOTH candidates come back (the provider is unfiltered).
    let pre = search_ids(&searcher);
    assert!(
        pre.contains(&live.raw()) && pre.contains(&doomed.raw()),
        "pre-delete: both live nodes must be returned; got {pre:?}"
    );

    // Delete `doomed` WITHOUT revoking its ACL (isolates the belt).
    delete_node_no_revoke(&backend, doomed);

    // Post-delete: the liveness gate must DROP the tombstoned node even
    // though the (stale) provider still surfaces it. The live node stays.
    let post = search_ids(&searcher);
    assert!(
        post.contains(&live.raw()),
        "#1379 belt: the LIVE node must still be returned; got {post:?}"
    );
    assert!(
        !post.contains(&doomed.raw()),
        "#1379 belt: the DELETED node must NOT be returned by search_filtered \
         (read_node-liveness gate), even though the provider still surfaces it \
         and the ACL was not revoked; got {post:?}"
    );
}
