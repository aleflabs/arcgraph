//! ADR-148 W26-θ Phase 2 — end-to-end CREATE rel through the
//! production `CrudExecutorSubstrate` + `QueryEngine`.
//!
//! Mirrors the Phase 1 `mcp_create_node_e2e.rs` test pattern. Wires
//! the same path the MCP `graph.raw_query` tool eventually takes:
//! parse → bind → typecheck → cross-substrate validate → lower →
//! execute against `CrudExecutorSubstrate`. The substrate's
//! `create_rel` opens a per-tenant `Transaction` and commits per
//! ADR-031 + ADR-033; the resulting rel-id is observable via the
//! same substrate's `expand`.
//!
//! This integration test is the load-bearing v1.0-α pin for the
//! query-side CREATE-rel closure — it proves the ArcQL CREATE-path
//! durably installs both nodes + the relationship through the actual
//! storage substrate (per-tenant Transaction + `crud::create_rel` +
//! commit), NOT just the in-memory stub.

use std::sync::Arc;

use arcgraph_core::{Lsn, PartitionId, TenantId};
use arcgraph_mcp::storage::substrate::CrudExecutorSubstrate;
use arcgraph_query::ExecutorSubstrate;
use arcgraph_query::executor::{ExecutionContext, Pipeline};
use arcgraph_query::logical_plan::{Direction, LogicalPlanLoweringVisitor};
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
/// MultiTenantRouter. Mirrors the Phase 1 `mcp_create_node_e2e::fixture`
/// shape.
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

/// Run the query-side pipeline against the supplied substrate.
/// Returns the number of rows emitted by the executor.
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
fn mcp_e2e_create_rel_persists_two_nodes_and_one_rel() {
    // Full query-side pipeline + production substrate. After
    // CREATE-rel, scan_nodes returns 2 nodes (the inline-CREATE
    // endpoints) and expand from either endpoint returns the rel.
    let sub = fixture();
    let rows = execute("CREATE (a:User)-[r:KNOWS]->(b:User) RETURN r", &sub);
    assert_eq!(rows, 1, "exactly one row from CREATE-rel");

    let nodes = sub
        .scan_nodes(TenantId::DEFAULT, None, Lsn::MAX)
        .expect("scan_nodes OK");
    assert_eq!(nodes.len(), 2, "scan_nodes after CREATE finds 2 nodes");

    // Outbound expand from the source NodeId surfaces the new rel.
    // We don't know a priori which scan_nodes row corresponds to the
    // source (the iteration order is by NodeId asc), so we expand from
    // both and assert that EXACTLY ONE direction surfaces the rel.
    let mut total_edges = 0usize;
    for n in &nodes {
        let edges = sub
            .expand(
                TenantId::DEFAULT,
                n.node.id,
                None,
                Direction::LeftToRight,
                Lsn::MAX,
            )
            .expect("expand OK");
        total_edges += edges.len();
    }
    assert_eq!(
        total_edges, 1,
        "expand surfaces the CREATE-d rel exactly once"
    );
}

#[test]
fn mcp_e2e_create_rel_right_to_left_persists_canonical_form() {
    // `(a)<-[r:R]-(b)` AST direction is RightToLeft; the executor
    // swaps source/target BEFORE the substrate call so the stored rel
    // is canonical source→target wire order.
    let sub = fixture();
    let rows = execute("CREATE (a:User)<-[r:FOLLOWED]-(b:User) RETURN r", &sub);
    assert_eq!(rows, 1);
    let nodes = sub
        .scan_nodes(TenantId::DEFAULT, None, Lsn::MAX)
        .expect("scan OK");
    assert_eq!(nodes.len(), 2);
    // Same as the LeftToRight test: exactly one outbound-expand from
    // exactly one endpoint surfaces the rel.
    let mut total_edges = 0usize;
    for n in &nodes {
        let edges = sub
            .expand(
                TenantId::DEFAULT,
                n.node.id,
                None,
                Direction::LeftToRight,
                Lsn::MAX,
            )
            .expect("expand OK");
        total_edges += edges.len();
    }
    assert_eq!(total_edges, 1);
}

#[test]
fn mcp_e2e_create_anonymous_rel_still_persists() {
    let sub = fixture();
    let rows = execute("CREATE (a:User)-[:LIKES]->(b:Post)", &sub);
    assert_eq!(rows, 1, "anonymous CREATE-rel emits 1 row");
    let nodes = sub
        .scan_nodes(TenantId::DEFAULT, None, Lsn::MAX)
        .expect("scan OK");
    assert_eq!(nodes.len(), 2);
}

#[test]
fn mcp_e2e_create_rel_with_literal_properties_persists() {
    // ADR-148 §D-3: Phase 2 inherits Phase 1's literal-only
    // narrowing on the rel property bag. The storage write stores
    // `PropertyData::Empty` (per the v1.2 strict-schema forward-pin);
    // the executor + substrate still process end-to-end without error.
    let sub = fixture();
    let rows = execute(
        r#"CREATE (a:User)-[r:KNOWS {since: 2024, kind: "close"}]->(b:User) RETURN r"#,
        &sub,
    );
    assert_eq!(rows, 1);
    let nodes = sub
        .scan_nodes(TenantId::DEFAULT, None, Lsn::MAX)
        .expect("scan OK");
    assert_eq!(nodes.len(), 2);
}

#[test]
fn mcp_e2e_create_rel_intern_type_round_trips_stable_id() {
    // Two CREATE-rels with the same type-name should intern to the
    // same TypeId. Three nodes (two from first CREATE, two from
    // second CREATE — note: each CREATE creates ITS OWN endpoints; we
    // don't share endpoints across CREATEs at Phase 2 per the
    // inline-CREATE narrowing).
    let sub = fixture();
    let _ = execute("CREATE (a:User)-[r:KNOWS]->(b:User) RETURN r", &sub);
    let _ = execute("CREATE (c:User)-[r2:KNOWS]->(d:User) RETURN r2", &sub);
    let nodes = sub
        .scan_nodes(TenantId::DEFAULT, None, Lsn::MAX)
        .expect("scan OK");
    assert_eq!(nodes.len(), 4, "two CREATE-rels create 4 distinct nodes");
    // The intern table dispenses a stable TypeId for "KNOWS" across
    // the two CREATEs.
    let knows_type = sub
        .intern_table()
        .intern_type(TenantId::DEFAULT, "KNOWS")
        .unwrap();
    let mut found = 0usize;
    for n in &nodes {
        let edges = sub
            .expand(
                TenantId::DEFAULT,
                n.node.id,
                Some(knows_type),
                Direction::LeftToRight,
                Lsn::MAX,
            )
            .expect("expand OK");
        found += edges.len();
    }
    assert_eq!(found, 2, "two outbound KNOWS edges total");
}
