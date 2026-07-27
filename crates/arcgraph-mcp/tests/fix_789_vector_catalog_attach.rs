//! #789 — fix: vector/BM25 catalog attachment for served concurrent RANK BY KNN.
//!
//! Before the fix, `build_catalog_for_tenant` never set `.with_vector_index(true)`,
//! causing `CrossSubstrateValidator` to reject ArcQL `RANK BY HYBRID(VECTOR(...), ...)`
//! queries with "cross-substrate error: substrate vector not attached" (-32005).
//!
//! After the fix, when a tenant has nodes, the catalog marks vector and BM25
//! substrates as attached, unblocking served concurrent KNN via ArcQL while still
//! correctly rejecting queries from tenants with zero nodes.

use std::sync::Arc;

use arcgraph_core::TenantId;
use arcgraph_mcp::storage::adapters::{StorageBackend, build_catalog_for_tenant};
use arcgraph_query::semantic::CatalogProvider;
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::catalog::SystemCatalog;
use arcgraph_storage::crud::{CrudStore, PropertyData};
use arcgraph_storage::intern::InternTable;
use arcgraph_storage::io::InMemoryPageIo;
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::PrimaryIndex;
use arcgraph_storage::router::MultiTenantRouter;
use arcgraph_storage::transaction::TxnManager;

#[test]
fn fresh_tenant_with_zero_nodes_correctly_rejects_vector() {
    // Setup: minimal backend with no nodes.
    let io = Arc::new(InMemoryPageIo::new());
    let pool = BufferPool::new(8, io);
    let mgr = Arc::new(TxnManager::new());
    let catalog = Arc::new(SystemCatalog::new());
    catalog.bootstrap(&pool, &mgr).expect("bootstrap");
    let allocator = Arc::new(PageAllocator::new());
    let primary = Arc::new(
        PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&allocator), None).expect("primary"),
    );
    let crud = Arc::new(CrudStore::new_with_index(None, primary, allocator));
    let router = Arc::new(MultiTenantRouter::new(catalog, Arc::clone(&crud), None));
    let intern = Arc::new(InternTable::new());
    let backend = StorageBackend::new(router, Arc::clone(&mgr), intern);
    let tenant = TenantId::DEFAULT;

    // Assert: fresh tenant (zero commits, zero nodes) → no vector substrate.
    let cat = build_catalog_for_tenant(tenant, &backend);
    assert!(
        !cat.has_vector_index(),
        "fresh tenant with 0 nodes must NOT have vector substrate"
    );
    assert!(
        !cat.has_bm25_index(),
        "fresh tenant with 0 nodes must NOT have bm25 substrate"
    );
}

#[test]
fn tenant_with_nodes_attaches_vector_and_bm25() {
    // Setup: minimal backend that we will populate with a node.
    let io = Arc::new(InMemoryPageIo::new());
    let pool = BufferPool::new(8, io);
    let mgr = Arc::new(TxnManager::new());
    let catalog = Arc::new(SystemCatalog::new());
    catalog.bootstrap(&pool, &mgr).expect("bootstrap");
    let allocator = Arc::new(PageAllocator::new());
    let primary = Arc::new(
        PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&allocator), None).expect("primary"),
    );
    let crud = Arc::new(CrudStore::new_with_index(None, primary, allocator));
    let router = Arc::new(MultiTenantRouter::new(catalog, Arc::clone(&crud), None));
    let intern = Arc::new(InternTable::new());
    let backend = StorageBackend::new(router, Arc::clone(&mgr), intern);
    let tenant = TenantId::DEFAULT;

    // Create a node so the tenant has total_node_count > 0.
    let label = backend
        .intern_table()
        .intern_label(tenant, "TestNode")
        .unwrap();
    let mut tx = backend.txn_manager().begin(tenant);
    let _ =
        arcgraph_storage::crud::create_node(&crud, &mut tx, tenant, label, &PropertyData::Empty)
            .expect("create node");
    arcgraph_storage::crud::commit(tx, &crud).expect("commit");

    // Assert: tenant with ≥1 node → vector and BM25 substrates attached.
    let cat = build_catalog_for_tenant(tenant, &backend);
    assert!(
        cat.has_vector_index(),
        "tenant with ≥1 node must have vector substrate attached"
    );
    assert!(
        cat.has_bm25_index(),
        "tenant with ≥1 node must have bm25 substrate attached"
    );
}

#[test]
fn rank_by_vector_query_binds_when_tenant_has_nodes() {
    // Full integration: create nodes, build catalog, validate an ArcQL query
    // that uses RANK BY HYBRID(VECTOR(...), ...) does NOT fail bind-time
    // cross-substrate validation.

    let io = Arc::new(InMemoryPageIo::new());
    let pool = BufferPool::new(8, io);
    let mgr = Arc::new(TxnManager::new());
    let catalog_sys = Arc::new(SystemCatalog::new());
    catalog_sys.bootstrap(&pool, &mgr).expect("bootstrap");
    let allocator = Arc::new(PageAllocator::new());
    let primary = Arc::new(
        PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&allocator), None).expect("primary"),
    );
    let crud = Arc::new(CrudStore::new_with_index(None, primary, allocator));
    let router = Arc::new(MultiTenantRouter::new(catalog_sys, Arc::clone(&crud), None));
    let intern = Arc::new(InternTable::new());
    let backend = StorageBackend::new(router, Arc::clone(&mgr), intern);
    let tenant = TenantId::DEFAULT;

    // Populate: create a node. (We don't need actual embeddings — just need
    // total_node_count > 0 to trigger substrate attachment in the catalog.)
    let label = backend.intern_table().intern_label(tenant, "Doc").unwrap();
    let mut tx = backend.txn_manager().begin(tenant);
    let _node =
        arcgraph_storage::crud::create_node(&crud, &mut tx, tenant, label, &PropertyData::Empty)
            .expect("create doc node");
    arcgraph_storage::crud::commit(tx, &crud).expect("commit");

    // Validate: build the catalog and parse+bind an ArcQL query.
    let cat = build_catalog_for_tenant(tenant, &backend);
    assert!(cat.has_vector_index(), "catalog must have vector");
    assert!(cat.has_bm25_index(), "catalog must have bm25");

    // Attempt to bind a RANK BY HYBRID query. The binding + cross-substrate
    // validation should succeed (no "substrate vector not attached" error).
    // If the fix is NOT applied, this would fail with CrossSubstrateError.
    let query_str = "MATCH (d:Doc) RANK BY HYBRID(VECTOR(d.embedding, $q, K=10), TEXT(d.text, $t, K=10)) WITH FUSION = RRF(k=10) RETURN d";
    let stmt = arcgraph_query::parse(query_str).expect("parse");
    let mut bound =
        arcgraph_query::semantic::BindingVisitor::bind(&stmt, query_str, &cat).expect("bind");
    arcgraph_query::semantic::TypeCheckVisitor::check(&mut bound, &cat).expect("type-check");

    // Cross-substrate validation should NOT fail now that has_vector_index=true.
    let validation_result =
        arcgraph_query::semantic::CrossSubstrateValidator::validate(&bound, &cat);
    assert!(
        validation_result.is_ok(),
        "RANK BY HYBRID query should validate when tenant has nodes; got: {:?}",
        validation_result
    );
}
