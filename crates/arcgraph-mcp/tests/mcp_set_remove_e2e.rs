//! ADR-150 W26-θ Phase 4 — end-to-end SET / REMOVE (property + label)
//! through the production `CrudExecutorSubstrate`.
//!
//! Mirrors `mcp_delete_e2e.rs`. Property mutations route through
//! `arcgraph_storage::crud::update_node` / `update_rel` per ADR-150
//! §D-7 (PropertyData::Empty v1.0-α posture inherited from ADR-147
//! §"Forward-deferred"). Label mutations surface
//! `IndexUnavailable("...forward-pinned to v1.1...")` per ADR-150
//! §D-9 (the storage `update_node` primitive preserves `label_id`
//! immutably per `crud.rs:3754` "PR #170 reviewer Finding 4").

use std::sync::Arc;

use arcgraph_core::{PartitionId, TenantId};
use arcgraph_mcp::storage::substrate::CrudExecutorSubstrate;
use arcgraph_query::executor::error::ExecutionError;
use arcgraph_query::executor::substrate::SubstrateAccessError;
use arcgraph_query::executor::{ExecutionContext, Pipeline};
use arcgraph_query::logical_plan::LogicalPlanLoweringVisitor;
use arcgraph_query::parse;
use arcgraph_query::semantic::{
    BindingVisitor, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};
use arcgraph_storage::InternTable;
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::catalog::SystemCatalog;
use arcgraph_storage::crud::CrudStore;
use arcgraph_storage::io::InMemoryPageIo;
use arcgraph_storage::router::MultiTenantRouter;
use arcgraph_storage::transaction::TxnManager;

fn fixture() -> CrudExecutorSubstrate {
    let io = Arc::new(InMemoryPageIo::new());
    let pool = BufferPool::new(8, io);
    let mgr = Arc::new(TxnManager::new());
    let catalog = Arc::new(SystemCatalog::new());
    catalog.bootstrap(&pool, &mgr).expect("bootstrap catalog");
    let crud = Arc::new(CrudStore::new());
    let router = Arc::new(MultiTenantRouter::new(catalog, Arc::clone(&crud), None));
    let intern = Arc::new(InternTable::new());
    CrudExecutorSubstrate::new(router, mgr, intern)
}

/// Walk Parse → Bind → TypeCheck → CrossSubstrate → Lower → execute
/// against the production substrate. Returns the number of rows
/// emitted OR the propagated error.
fn execute(query: &str, sub: &CrudExecutorSubstrate) -> Result<usize, ExecutionError> {
    let stmt = parse(query).expect("parse OK");
    let cat = StubCatalogProvider::new();
    let mut bound = BindingVisitor::bind(&stmt, query, &cat).expect("bind OK");
    TypeCheckVisitor::check(&mut bound, &cat).expect("type-check OK");
    CrossSubstrateValidator::validate(&bound, &cat).expect("cross-substrate OK");
    let plan = LogicalPlanLoweringVisitor::lower(&bound).expect("lower OK");
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let mut op = Pipeline::build(&plan).expect("pipeline build OK");
    let mut total = 0usize;
    loop {
        let b = op.next_batch(&ctx, sub)?;
        if b.is_empty() {
            break;
        }
        total += b.row_count();
    }
    Ok(total)
}

#[test]
fn mcp_e2e_create_then_set_property_commits_via_update_node() {
    // CREATE node, then SET property — both via per-tenant
    // Transaction at the storage layer. Property values are IGNORED
    // at v1.0-α (PropertyData::Empty); the call shape exercises the
    // executor + storage wire-through.
    let sub = fixture();
    let _ = execute("CREATE (n:User) SET n.name = \"Alice\"", &sub)
        .expect("CREATE → SET property succeeds via update_node");
}

#[test]
fn mcp_e2e_create_then_set_property_replace_commits() {
    let sub = fixture();
    let _ = execute("CREATE (n:User) SET n = {name: \"Bob\", age: 30}", &sub)
        .expect("CREATE → SET property-replace succeeds");
}

#[test]
fn mcp_e2e_create_then_set_property_merge_commits() {
    let sub = fixture();
    let _ = execute("CREATE (n:User) SET n += {age: 25}", &sub)
        .expect("CREATE → SET property-merge succeeds");
}

#[test]
fn mcp_e2e_create_then_remove_property_commits_via_update_node() {
    let sub = fixture();
    let _ = execute("CREATE (n:User) REMOVE n.age", &sub)
        .expect("CREATE → REMOVE property succeeds via update_node");
}

#[test]
fn mcp_e2e_create_then_set_label_surfaces_index_unavailable() {
    // ADR-150 §D-9 forward-pin: label mutation requires schema-
    // migration support; v1.0-α surfaces IndexUnavailable.
    let sub = fixture();
    let r = execute("CREATE (n:User) SET n:VIP", &sub);
    match r {
        Err(ExecutionError::Substrate(SubstrateAccessError::IndexUnavailable(msg))) => {
            assert!(
                msg.contains("forward-pinned to v1.1") && msg.contains("ADR-150"),
                "expected forward-pin message; got `{msg}`"
            );
        }
        other => panic!("expected IndexUnavailable, got {other:?}"),
    }
}

#[test]
fn mcp_e2e_create_then_remove_label_surfaces_index_unavailable() {
    let sub = fixture();
    let r = execute("CREATE (n:User) REMOVE n:VIP", &sub);
    match r {
        Err(ExecutionError::Substrate(SubstrateAccessError::IndexUnavailable(msg))) => {
            assert!(
                msg.contains("forward-pinned to v1.1") && msg.contains("ADR-150"),
                "expected forward-pin message; got `{msg}`"
            );
        }
        other => panic!("expected IndexUnavailable, got {other:?}"),
    }
}
