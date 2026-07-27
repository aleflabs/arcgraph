use std::sync::Arc;

use arcgraph_core::{LabelId, Lsn, NodeId, TenantId, TypeId};
use arcgraph_mcp::storage::CrudExecutorSubstrate;
use arcgraph_query::executor::{ExecutorSubstrate, SubstrateAccessError};
use arcgraph_query::logical_plan::Direction;
use arcgraph_storage::InternTable;
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::catalog::SystemCatalog;
use arcgraph_storage::crud::{CrudStore, PropertyData, commit, create_node, create_rel};
use arcgraph_storage::io::InMemoryPageIo;
use arcgraph_storage::router::MultiTenantRouter;
use arcgraph_storage::transaction::TxnManager;

fn graph_fixture() -> (
    CrudExecutorSubstrate,
    Arc<TxnManager>,
    NodeId,
    NodeId,
    TypeId,
    Lsn,
) {
    let io = Arc::new(InMemoryPageIo::new());
    let pool = BufferPool::new(8, io);
    let manager = Arc::new(TxnManager::new());
    let catalog = Arc::new(SystemCatalog::new());
    catalog
        .bootstrap(&pool, &manager)
        .expect("bootstrap catalog");
    let crud = Arc::new(CrudStore::new());
    let router = Arc::new(MultiTenantRouter::new(catalog, Arc::clone(&crud), None));
    let substrate =
        CrudExecutorSubstrate::new(router, Arc::clone(&manager), Arc::new(InternTable::new()));

    let stale = manager.current_lsn();
    let label = LabelId::new(1);
    let rel_type = TypeId::new(1);
    let mut tx = manager.begin(TenantId::DEFAULT);
    let from = create_node(
        &crud,
        &mut tx,
        TenantId::DEFAULT,
        label,
        &PropertyData::Empty,
    )
    .expect("create source node");
    let to = create_node(
        &crud,
        &mut tx,
        TenantId::DEFAULT,
        label,
        &PropertyData::Empty,
    )
    .expect("create destination node");
    create_rel(
        &crud,
        &mut tx,
        TenantId::DEFAULT,
        from,
        to,
        rel_type,
        &PropertyData::Empty,
    )
    .expect("create relationship");
    commit(tx, &crud).expect("commit graph");

    (substrate, manager, from, to, rel_type, stale)
}

fn expect_snapshot_error<T>(
    result: Result<T, SubstrateAccessError>,
    requested: Lsn,
    available: Lsn,
) {
    match result {
        Err(error) => assert_eq!(
            error,
            SubstrateAccessError::SnapshotUnavailable {
                requested,
                available,
            }
        ),
        Ok(_) => panic!(
            "snapshot request {requested:?} must not succeed at available snapshot {available:?}"
        ),
    }
}

#[test]
fn unavailable_stale_and_future_lsns_are_typed_errors_for_scan_and_expand() {
    let (substrate, manager, from, _to, rel_type, stale) = graph_fixture();
    let available = manager.current_lsn();
    assert_ne!(stale, available, "fixture must advance the visible LSN");
    let future = Lsn::new(
        available
            .raw()
            .checked_add(1)
            .expect("fixture LSN leaves future headroom"),
    );

    for requested in [stale, future] {
        expect_snapshot_error(
            substrate.scan_nodes(TenantId::DEFAULT, None, requested),
            requested,
            available,
        );
        expect_snapshot_error(
            substrate.expand(
                TenantId::DEFAULT,
                from,
                Some(rel_type),
                Direction::LeftToRight,
                requested,
            ),
            requested,
            available,
        );
    }
}

#[test]
fn exact_available_lsn_is_honoured_by_scan_and_expand() {
    let (substrate, manager, from, to, rel_type, _stale) = graph_fixture();
    let available = manager.current_lsn();

    let nodes = substrate
        .scan_nodes(TenantId::DEFAULT, None, available)
        .expect("exact available scan snapshot");
    assert_eq!(nodes.len(), 2);

    let edges = substrate
        .expand(
            TenantId::DEFAULT,
            from,
            Some(rel_type),
            Direction::LeftToRight,
            available,
        )
        .expect("exact available expand snapshot");
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].dst.id, to);
}
