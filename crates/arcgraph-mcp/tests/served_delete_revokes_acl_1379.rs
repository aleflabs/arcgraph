//! #1379 (MUST-CON-04) — SERVED-path oracle: the production
//! `CrudExecutorSubstrate::delete_node` (ArcQL DELETE) revokes the deleted
//! node's doc-ACL through `TenantHandle::permissions()`, so the node stops
//! leaking. This exercises the ACTUAL substrate wiring changed by the fix
//! (`crates/arcgraph-mcp/src/storage/substrate.rs::delete_node`), reaching
//! the SHARED per-tenant `PermissionIndex` off the router exactly as the
//! ingest write-through (`apply_live_acl_grants`) does.
//!
//! # Scenario
//!
//! A `MultiTenantRouter` is built with a SHARED `PermissionIndex` wired in
//! (mirrors the durable bootstrap's `.permissions(DEFAULT, …)`). A node is
//! seeded in the router's `CrudStore` + granted a doc-ACL visible to
//! `alice`. Then the SERVED `delete_node` runs (through the
//! `ExecutorSubstrate` trait, the same path ArcQL `DELETE` takes). The
//! shared index must show the node revoked (`is_visible == false`).
//!
//! RED-on-revert: delete the `revoke_doc` block in `substrate.rs`'s
//! `delete_node` and this test observes `is_visible == true` ⇒ FAILS.

use std::collections::BTreeSet;
use std::sync::Arc;

use arcgraph_core::{LabelId, NodeId, PartitionId, TenantId};
use arcgraph_mcp::storage::substrate::CrudExecutorSubstrate;
use arcgraph_query::ExecutorSubstrate;
use arcgraph_query::executor::ExecutionContext;
use arcgraph_storage::InternTable;
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::catalog::SystemCatalog;
use arcgraph_storage::crud::{self, CrudStore, PropertyData};
use arcgraph_storage::io::InMemoryPageIo;
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::permissions::PermissionIndex;
use arcgraph_storage::primary_index::PrimaryIndex;
use arcgraph_storage::router::MultiTenantRouter;
use arcgraph_storage::transaction::TxnManager;

const TENANT: TenantId = TenantId::DEFAULT;

fn grants(items: &[&str]) -> BTreeSet<String> {
    items.iter().map(|s| (*s).to_owned()).collect()
}

/// Build a served substrate whose router carries a SHARED
/// `PermissionIndex` (the durable-bootstrap wiring shape). Returns the
/// substrate, the shared `CrudStore`, the txn manager, and the shared
/// index so the test can seed + assert against the SAME index the served
/// `delete_node` reaches.
fn fixture() -> (
    CrudExecutorSubstrate,
    Arc<CrudStore>,
    Arc<TxnManager>,
    Arc<PermissionIndex>,
) {
    let io = Arc::new(InMemoryPageIo::new());
    let pool = BufferPool::new(8, io);
    let mgr = Arc::new(TxnManager::new());
    let catalog = Arc::new(SystemCatalog::new());
    catalog.bootstrap(&pool, &mgr).expect("bootstrap catalog");
    let allocator = Arc::new(PageAllocator::new());
    let primary = Arc::new(
        PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&allocator), None).expect("primary"),
    );
    let crud = Arc::new(CrudStore::new_with_index(None, primary, allocator));
    let permissions = Arc::new(PermissionIndex::new());
    let router = Arc::new(
        MultiTenantRouter::builder(Arc::clone(&catalog), Arc::clone(&crud))
            .permissions(TENANT, Arc::clone(&permissions))
            .build(),
    );
    let intern = Arc::new(InternTable::new());
    let sub = CrudExecutorSubstrate::new(Arc::clone(&router), Arc::clone(&mgr), intern);
    (sub, crud, mgr, permissions)
}

fn seed_node(crud: &Arc<CrudStore>, mgr: &Arc<TxnManager>) -> NodeId {
    let mut tx = mgr.begin(TENANT);
    let id = crud::create_node(
        crud,
        &mut tx,
        TENANT,
        LabelId::new(1),
        &PropertyData::InlineU32Pair(1, 2),
    )
    .expect("create_node");
    crud::commit(tx, crud).expect("commit seed");
    id
}

#[test]
fn served_delete_node_revokes_shared_acl_index() {
    let (sub, crud, mgr, permissions) = fixture();

    // Seed a node + grant its doc-ACL to alice (via the SHARED index the
    // served substrate will reach off the router).
    let id = seed_node(&crud, &mgr);
    permissions.apply_doc_acl(id, grants(&["alice"]));
    assert!(
        permissions.effective("alice").is_visible(id),
        "pre-delete: alice must see the granted node"
    );

    // SERVED DELETE through the production ExecutorSubstrate (auto-commit,
    // no held txn) — the same path ArcQL `DELETE n` takes.
    let ctx = ExecutionContext::new(TENANT, PartitionId::ZERO);
    sub.delete_node(TENANT, id, false, &ctx)
        .expect("served delete_node");

    // #1379: the served delete must have revoked the doc-ACL on the SHARED
    // index — alice no longer sees the deleted node.
    assert!(
        !permissions.effective("alice").is_visible(id),
        "#1379: after the SERVED delete, alice must NOT see the node \
         (substrate.rs delete_node must call permissions().revoke_doc)"
    );
}
