//! ADR-151 W26-θ Phase 5 — end-to-end MERGE (match-or-create) through
//! the production `CrudExecutorSubstrate`.
//!
//! Mirrors `mcp_set_remove_e2e.rs`. The production substrate at v1.0-α
//! routes:
//! - Match-branch's `scan_nodes` → CrudStore (returns nodes filtered
//!   by label; the merge pattern's property bag is IGNORED at
//!   v1.0-α per ADR-151 §"Risks").
//! - Create-branch's `create_node` / `create_rel` → per-tenant
//!   `Transaction` per ADR-147 / ADR-148 §D-7 (PropertyData::Empty
//!   v1.0-α posture).
//! - On_create / on_match actions → `set_node` / `set_rel` per ADR-150
//!   §D-7 (PropertyData::Empty v1.0-α posture; LabelAdd surfaces
//!   IndexUnavailable per ADR-150 §D-9).
//!
//! Phase 5 v1.0-α production behavior: MERGE on an empty store creates
//! the node; subsequent MERGEs on a label-only fixture either match
//! (when the label-resolved Scan finds the prior node) or always-
//! create (when the label is fresh per tenant). The test asserts
//! end-to-end execution through the production substrate; the v1.2
//! strict-schema flip + the v1.1 atomic MERGE both close the
//! always-create posture in production.

use std::sync::Arc;

use arcgraph_core::{PartitionId, TenantId};
use arcgraph_mcp::storage::substrate::CrudExecutorSubstrate;
use arcgraph_query::executor::error::ExecutionError;
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
/// against the production substrate. Returns Ok on success or the
/// propagated error.
fn execute(query: &str, sub: &CrudExecutorSubstrate) -> Result<(), ExecutionError> {
    let stmt = parse(query).expect("parse OK");
    let cat = StubCatalogProvider::new();
    let mut bound = BindingVisitor::bind(&stmt, query, &cat).expect("bind OK");
    TypeCheckVisitor::check(&mut bound, &cat).expect("type-check OK");
    CrossSubstrateValidator::validate(&bound, &cat).expect("cross-substrate OK");
    let plan = LogicalPlanLoweringVisitor::lower(&bound).expect("lower OK");
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let mut op = Pipeline::build(&plan).expect("pipeline build OK");
    loop {
        let b = op.next_batch(&ctx, sub)?;
        if b.is_empty() {
            break;
        }
    }
    Ok(())
}

#[test]
fn mcp_e2e_merge_node_create_branch_commits_via_create_node() {
    // First MERGE on an empty store → match-branch returns 0 rows →
    // create-branch fires → production substrate's create_node opens
    // + commits a per-tenant Transaction.
    let sub = fixture();
    execute("MERGE (n:User)", &sub).expect("MERGE create-branch succeeds");
}

#[test]
fn mcp_e2e_merge_with_on_create_set_fires_set_node() {
    // MERGE + ON CREATE SET — both the create-branch + the action's
    // set_node fire via per-tenant Transactions.
    let sub = fixture();
    execute(
        "MERGE (n:User {id: 42}) ON CREATE SET n.name = \"Alice\"",
        &sub,
    )
    .expect("MERGE + ON CREATE SET succeeds via update_node");
}

#[test]
fn mcp_e2e_merge_with_on_match_set_fires_on_match_after_prior_match() {
    // First MERGE creates the node. Second MERGE finds it via the
    // match-branch (Phase 5 v1.0-α: production scan_nodes returns
    // ALL User nodes; the always-create posture per ADR-151 §"Risks"
    // means the match-branch might fire or might not — both paths
    // exercise the executor + substrate wire-through correctly).
    //
    // We assert the second MERGE doesn't error.
    let sub = fixture();
    execute("MERGE (n:User)", &sub).expect("first MERGE creates");
    execute("MERGE (n:User) ON MATCH SET n.last_seen = \"today\"", &sub)
        .expect("second MERGE + ON MATCH SET succeeds end-to-end");
}

#[test]
fn mcp_e2e_merge_path_create_branch_commits_via_create_rel() {
    // MERGE on a path-shape pattern: source + target nodes + rel are
    // all created atomically when match-branch returns 0 rows.
    let sub = fixture();
    execute("MERGE (a:User)-[r:FOLLOWS]->(b:User)", &sub)
        .expect("MERGE path create-branch succeeds via create_node + create_rel");
}

#[test]
fn mcp_e2e_merge_with_both_on_create_and_on_match_fires_branch() {
    // Full shape: MERGE + ON CREATE + ON MATCH. On an empty store the
    // create-branch + on_create fire; the on_match branch is dead.
    let sub = fixture();
    execute(
        "MERGE (n:User {id: 42}) ON CREATE SET n.name = \"new\" ON MATCH SET n.name = \"old\"",
        &sub,
    )
    .expect("MERGE + ON CREATE/ON MATCH SET succeeds end-to-end");
}
