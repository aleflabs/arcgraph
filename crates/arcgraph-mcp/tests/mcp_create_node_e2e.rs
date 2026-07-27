//! ADR-147 W26-θ Phase 1 — end-to-end CREATE node through the
//! production `CrudExecutorSubstrate` + `QueryEngine`.
//!
//! Wires the same path the MCP `graph.raw_query` tool eventually
//! takes: parse → bind → typecheck → cross-substrate validate →
//! lower → execute against `CrudExecutorSubstrate`. The substrate's
//! `create_node` opens a per-tenant `Transaction` and commits per
//! ADR-031 + ADR-033; the resulting node-id is read back via the same
//! substrate's `scan_nodes`.
//!
//! This integration test is the load-bearing v1.0-α pin for the
//! query-side write-path closure — it proves the ArcQL CREATE op
//! durably installs through the actual storage substrate, NOT just
//! the in-memory stub.

use std::sync::Arc;

use arcgraph_core::{LabelId, Lsn, PartitionId, TenantId};
use arcgraph_mcp::storage::substrate::CrudExecutorSubstrate;
use arcgraph_query::ExecutorSubstrate;
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

/// Build a production-shaped substrate: catalog bootstrapped,
/// CrudStore + TxnManager + intern table all wired through a fresh
/// MultiTenantRouter. Mirrors the
/// `crates/arcgraph-mcp/src/storage/substrate.rs::tests::fixture` shape.
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

/// Run the query-side pipeline (parse → bind → typecheck → cross-sub
/// → lower → execute) against the supplied substrate. Returns the
/// number of rows emitted by the executor.
fn execute(query: &str, sub: &CrudExecutorSubstrate) -> usize {
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
        let b = op.next_batch(&ctx, sub).expect("next_batch OK");
        if b.is_empty() {
            break;
        }
        total += b.row_count();
    }
    total
}

#[test]
fn mcp_e2e_create_node_persists_through_real_substrate() {
    // The full query-side pipeline + production substrate. After
    // CREATE, scan_nodes (which the MCP `graph.search` and Bolt
    // RUN/PULL paths also consume) returns the freshly-created node.
    let sub = fixture();
    let rows = execute("CREATE (n:User) RETURN n", &sub);
    assert_eq!(rows, 1, "exactly one row from CREATE (n:User) RETURN n");

    // Sanity: a follow-up MATCH (n:User) sees the new node.
    // We can't run MATCH through the executor yet (the executor's
    // MATCH path requires the User label to exist via the catalog;
    // the production catalog is dynamic-schema and adds labels at
    // first write — but the StubCatalogProvider does NOT). Instead,
    // we assert directly via the substrate's scan_nodes.
    let nodes = sub
        .scan_nodes(TenantId::DEFAULT, None, Lsn::MAX)
        .expect("scan_nodes OK");
    assert_eq!(nodes.len(), 1, "scan_nodes after CREATE finds 1 node");
}

#[test]
fn mcp_e2e_create_node_round_trips_by_label() {
    let sub = fixture();
    // CREATE three nodes with two distinct labels.
    let _ = execute("CREATE (a:User) RETURN a", &sub);
    let _ = execute("CREATE (b:User) RETURN b", &sub);
    let _ = execute("CREATE (c:Article) RETURN c", &sub);
    // All three nodes are observable via unfiltered scan.
    let all = sub
        .scan_nodes(TenantId::DEFAULT, None, Lsn::MAX)
        .expect("scan all");
    assert_eq!(all.len(), 3, "three CREATEs → three nodes");
    // User-label scan returns the two User nodes; Article scan the
    // remaining one. The intern_table dispenses stable LabelIds
    // across calls.
    let user_label = sub
        .intern_table()
        .intern_label(TenantId::DEFAULT, "User")
        .unwrap();
    let article_label = sub
        .intern_table()
        .intern_label(TenantId::DEFAULT, "Article")
        .unwrap();
    let users = sub
        .scan_nodes(TenantId::DEFAULT, Some(user_label), Lsn::MAX)
        .expect("scan User");
    let articles = sub
        .scan_nodes(TenantId::DEFAULT, Some(article_label), Lsn::MAX)
        .expect("scan Article");
    assert_eq!(users.len(), 2, "User-label scan: 2 nodes");
    assert_eq!(articles.len(), 1, "Article-label scan: 1 node");
}

#[test]
fn mcp_e2e_create_anonymous_node_still_persists() {
    let sub = fixture();
    let rows = execute("CREATE (:User)", &sub);
    assert_eq!(rows, 1, "anonymous CREATE emits 1 row (1 node created)");
    let all = sub
        .scan_nodes(TenantId::DEFAULT, None, Lsn::MAX)
        .expect("scan");
    assert_eq!(all.len(), 1);
}

#[test]
fn mcp_e2e_create_node_with_literal_properties_persists() {
    // ADR-147 §D-4: Phase 1 admits literal property bags. The
    // storage write stores `PropertyData::Empty` (per the v1.2
    // strict-schema forward-pin); the executor + substrate still
    // process the query end-to-end without error.
    let sub = fixture();
    let rows = execute(
        r#"CREATE (n:User {id: 42, name: "alice", flag: TRUE}) RETURN n"#,
        &sub,
    );
    assert_eq!(rows, 1);
    let all = sub
        .scan_nodes(TenantId::DEFAULT, None, Lsn::MAX)
        .expect("scan");
    assert_eq!(all.len(), 1);
}

#[test]
fn mcp_e2e_create_node_label_zero_used_for_anonymous() {
    // Label-less CREATE stores `LabelId::new(0)` per ADR-147 §D-7.
    let sub = fixture();
    let _ = execute("CREATE (n) RETURN n", &sub);
    let all = sub
        .scan_nodes(TenantId::DEFAULT, None, Lsn::MAX)
        .expect("scan");
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].node.label, None);
    // Filtering by label 0 surfaces no rows (the existing
    // CrudExecutorSubstrate::scan_nodes treats label_id=0 as label-
    // less, and filtering with Some(0) is structurally unsupported
    // — every label id is non-zero per the InternTable convention).
    // We assert the negative space — a User label scan finds zero.
    let user_label = sub
        .intern_table()
        .intern_label(TenantId::DEFAULT, "User")
        .unwrap();
    let users = sub
        .scan_nodes(TenantId::DEFAULT, Some(user_label), Lsn::MAX)
        .expect("scan User");
    assert!(users.is_empty(), "label-less node is not a User");
    // Defensive: LabelId::new(0) sentinel — drop ref so the compiler
    // doesn't complain about unused. The sentinel is only used at
    // storage layer; the trait surface exposes Option<LabelId>.
    let _ = LabelId::new(0);
}
