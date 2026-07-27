//! ADR-149 W26-θ Phase 3 — end-to-end DELETE (node + rel) through the
//! production `CrudExecutorSubstrate` + `QueryEngine`.
//!
//! Mirrors the Phase 1 + Phase 2 `mcp_create_*_e2e.rs` test patterns.
//! Wires the same path the MCP `graph.raw_query` tool eventually takes:
//! parse → bind → typecheck → cross-substrate validate → lower →
//! execute against `CrudExecutorSubstrate`. The substrate's
//! `delete_node` / `delete_rel` opens a per-tenant `Transaction` and
//! commits per ADR-031 + ADR-033; the resulting MVCC tombstone is
//! observable via the same substrate's `scan_nodes` / `expand` post-
//! commit.
//!
//! This is the load-bearing v1.0-α pin for the Phase 3 closure — it
//! proves the ArcQL DELETE clause durably tombstones nodes / rels
//! through the actual storage substrate (per-tenant Transaction +
//! `crud::delete_node_with_store` / `crud::delete_rel_with_store` +
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
/// MultiTenantRouter. Mirrors the Phase 1 / Phase 2 fixtures.
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
fn mcp_e2e_create_then_delete_node_round_trip_tombstones() {
    // CREATE → MATCH → DELETE → MATCH (empty) round-trip via the
    // production substrate. The DELETE clause tombstones the
    // CREATE-introduced node via the per-tenant Transaction +
    // delete_node_with_store + commit path.
    let sub = fixture();
    // CREATE: one node persists.
    let rows = execute("CREATE (n:User) RETURN n", &sub);
    assert_eq!(rows, 1, "CREATE emits 1 row");
    let pre_nodes = sub
        .scan_nodes(TenantId::DEFAULT, None, Lsn::MAX)
        .expect("scan OK");
    assert_eq!(pre_nodes.len(), 1, "scan finds the CREATE'd node");

    // DELETE: the CREATE-DELETE combo tombstones the just-created
    // node. The Delete operator runs at the end of the pipeline; its
    // output schema is empty.
    let _rows = execute("CREATE (n:User) DELETE n", &sub);

    // Post-state: 2 nodes existed (one from each CREATE), 1 was
    // tombstoned. `scan_nodes` returns 1 visible node (the one from
    // the first CREATE that was NOT followed by DELETE; the second
    // CREATE/DELETE pair leaves no residue).
    let post_nodes = sub
        .scan_nodes(TenantId::DEFAULT, None, Lsn::MAX)
        .expect("scan OK");
    assert_eq!(
        post_nodes.len(),
        1,
        "1 node remains (first CREATE persists; second CREATE+DELETE = no residue): {post_nodes:?}"
    );
}

#[test]
fn mcp_e2e_create_then_delete_rel_round_trip_tombstones_rel() {
    // CREATE rel → DELETE rel: both endpoints survive; the rel is
    // tombstoned.
    let sub = fixture();
    let _rows = execute("CREATE (a:User)-[r:KNOWS]->(b:User) DELETE r", &sub);

    let nodes = sub
        .scan_nodes(TenantId::DEFAULT, None, Lsn::MAX)
        .expect("scan OK");
    assert_eq!(nodes.len(), 2, "both endpoints survive: {nodes:?}");

    // The rel is tombstoned. Per `crud::delete_rel` rustdoc / issue
    // #22, scan_out still yields the TEL entry but the read_rel
    // (MVCC) tombstone-filter inside `scan_out` filters it. The
    // production substrate's `expand` reads the rel via
    // `read_rel(...)` which respects MVCC visibility → 0 visible
    // edges post-tombstone.
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
        total_edges, 0,
        "rel was MVCC-tombstoned post-DELETE: total_edges={total_edges}"
    );
}

#[test]
fn mcp_e2e_detach_delete_node_via_production_substrate_cascade() {
    // CREATE (a:User)-[r:KNOWS]->(b:User) installs the fixture
    // through the production substrate. Then call
    // `sub.delete_node(tenant, a_id, true, &arcgraph_query::executor::ExecutionContext::new(TenantId::DEFAULT, arcgraph_core::PartitionId::ZERO))` DIRECTLY to exercise the
    // production substrate's DETACH cascade path (which walks
    // `scan_out` + `scan_in` + calls `delete_rel_with_store` per
    // attached rel + `delete_node_with_store` + commit, all within
    // ONE per-tenant Transaction per ADR-149 §D-7).
    //
    // Note: we don't go through the ArcQL `DETACH DELETE` clause for
    // the source-of-rel binding because Phase 3 narrowing per
    // ADR-148 §D-9 + ADR-149 §D-9: CreateRel's output schema is just
    // `[rel_var]`, not the source/target bindings — so a CREATE→
    // DELETE-source-of-rel composition within a single statement is
    // not in the Phase 3 surface. The MATCH→DETACH-DELETE shape
    // works (the MATCH carries the binding through to the DELETE)
    // but the StubCatalogProvider doesn't know the production
    // substrate's labels, so we exercise the substrate cascade
    // directly here.
    let sub = fixture();
    let _ = execute("CREATE (a:User)-[r:KNOWS]->(b:User)", &sub);

    let pre_nodes = sub
        .scan_nodes(TenantId::DEFAULT, None, Lsn::MAX)
        .expect("scan OK");
    assert_eq!(pre_nodes.len(), 2, "fixture: 2 nodes installed");
    let a_id = pre_nodes
        .iter()
        .map(|n| n.node.id)
        .min()
        .expect("min node id");

    // Direct substrate cascade.
    sub.delete_node(
        TenantId::DEFAULT,
        a_id,
        true,
        &arcgraph_query::executor::ExecutionContext::new(
            TenantId::DEFAULT,
            arcgraph_core::PartitionId::ZERO,
        ),
    )
    .expect("DETACH DELETE OK");

    let post_nodes = sub
        .scan_nodes(TenantId::DEFAULT, None, Lsn::MAX)
        .expect("scan OK");
    assert_eq!(
        post_nodes.len(),
        1,
        "1 node remains (b) after production DETACH DELETE: {post_nodes:?}"
    );
    // No rels visible to or from b.
    let edges = sub
        .expand(
            TenantId::DEFAULT,
            post_nodes[0].node.id,
            None,
            Direction::Undirected,
            Lsn::MAX,
        )
        .expect("expand OK");
    assert_eq!(
        edges.len(),
        0,
        "no rels visible after production DETACH DELETE: {edges:?}"
    );
}

#[test]
fn mcp_e2e_bare_delete_node_with_attached_rel_surfaces_substrate_error() {
    // CREATE installs (a)-[r:KNOWS]->(b) via the production
    // substrate. Then call sub.delete_node(tenant, a_id, detach=false, &arcgraph_query::executor::ExecutionContext::new(TenantId::DEFAULT, arcgraph_core::PartitionId::ZERO))
    // — the production substrate scans the attached rels via
    // `scan_out` + `scan_in`, finds the attached rel, and surfaces
    // `SubstrateAccessError::Io("relationships attached; use DETACH
    // DELETE")` without partial side effect.
    //
    // We exercise the substrate path directly (not through the ArcQL
    // pipeline) for the same Phase 3 schema-narrowing reason as the
    // DETACH cascade test above. The Stub-level pin is in
    // `arcgraph-query/tests/delete_without_detach_fails.rs`.
    let sub = fixture();
    let _ = execute("CREATE (a:User)-[r:KNOWS]->(b:User)", &sub);

    let pre = sub
        .scan_nodes(TenantId::DEFAULT, None, Lsn::MAX)
        .expect("scan OK");
    assert_eq!(pre.len(), 2);
    let a_id = pre.iter().map(|n| n.node.id).min().expect("min node id");

    // Bare DELETE rejects.
    let result = sub.delete_node(
        TenantId::DEFAULT,
        a_id,
        false,
        &arcgraph_query::executor::ExecutionContext::new(
            TenantId::DEFAULT,
            arcgraph_core::PartitionId::ZERO,
        ),
    );
    assert!(
        matches!(
            result,
            Err(arcgraph_query::executor::substrate::SubstrateAccessError::Io(_))
        ),
        "expected Io error for bare DELETE over attached node, got {result:?}"
    );
    if let Err(arcgraph_query::executor::substrate::SubstrateAccessError::Io(msg)) = &result {
        assert!(
            msg.contains("relationships attached") || msg.contains("DETACH DELETE"),
            "Io message references attached rels: {msg}"
        );
    }

    // No partial side effect — both nodes + the rel are still
    // visible.
    let post_nodes = sub
        .scan_nodes(TenantId::DEFAULT, None, Lsn::MAX)
        .expect("scan OK");
    assert_eq!(
        post_nodes.len(),
        2,
        "no partial side effect from bare DELETE rejection: {post_nodes:?}"
    );
}

#[test]
fn mcp_e2e_anonymous_create_then_delete_isolated_node_persists_zero() {
    // Anonymous CREATE → MATCH-by-label DELETE not in this test
    // (Phase 3 narrowing — MATCH-by-property forward-pinned to Phase 5).
    // We exercise CREATE-then-DELETE with a labeled bind.
    let sub = fixture();
    // CREATE-then-DELETE for a single isolated node.
    let _ = execute("CREATE (n:User) DELETE n", &sub);
    let nodes = sub
        .scan_nodes(TenantId::DEFAULT, None, Lsn::MAX)
        .expect("scan OK");
    assert_eq!(
        nodes.len(),
        0,
        "no residue after CREATE-then-DELETE: {nodes:?}"
    );
}
