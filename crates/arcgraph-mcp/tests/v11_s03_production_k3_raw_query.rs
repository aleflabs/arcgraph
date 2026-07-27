//! V11-S-03 M3: production k=3 variable-length traversal through the
//! storage-backed raw-query path.

use std::sync::Arc;

use arcgraph_core::{Lsn, TenantId};
use arcgraph_mcp::storage::{CrudExecutorSubstrate, StorageBackend, StorageRawQueryExecutor};
use arcgraph_mcp::tools::raw_query::RawQueryExecutor;
use arcgraph_query::logical_plan::Direction;
use arcgraph_query::{CancellationToken, ExecutorSubstrate};
use arcgraph_storage::InternTable;
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::catalog::SystemCatalog;
use arcgraph_storage::crud::{CrudStore, PropertyData, commit, create_node, create_rel};
use arcgraph_storage::io::InMemoryPageIo;
use arcgraph_storage::router::MultiTenantRouter;
use arcgraph_storage::transaction::TxnManager;

fn fixture() -> (
    StorageRawQueryExecutor,
    CrudExecutorSubstrate,
    Arc<TxnManager>,
    Arc<CrudStore>,
    Arc<InternTable>,
) {
    let io = Arc::new(InMemoryPageIo::new());
    let pool = BufferPool::new(8, io);
    let mgr = Arc::new(TxnManager::new());
    let catalog = Arc::new(SystemCatalog::new());
    catalog.bootstrap(&pool, &mgr).expect("bootstrap catalog");
    let crud = Arc::new(CrudStore::new());
    let router = Arc::new(MultiTenantRouter::new(catalog, Arc::clone(&crud), None));
    let intern = Arc::new(InternTable::new());
    let backend = StorageBackend::new(Arc::clone(&router), Arc::clone(&mgr), Arc::clone(&intern));
    let substrate = CrudExecutorSubstrate::new(router, Arc::clone(&mgr), Arc::clone(&intern));
    (
        StorageRawQueryExecutor::new(backend),
        substrate,
        mgr,
        crud,
        intern,
    )
}

#[test]
fn m3_production_raw_query_k3_and_anchor_release() {
    let (raw, _substrate, mgr, crud, intern) = fixture();
    let tenant = TenantId::DEFAULT;
    let start_label = intern.intern_label(tenant, "Start").unwrap();
    let n_label = intern.intern_label(tenant, "N").unwrap();
    let rel_type = intern.intern_type(tenant, "R").unwrap();

    const FANOUT: usize = 5;
    let mut tx = mgr.begin(tenant);
    let root =
        create_node(&crud, &mut tx, tenant, start_label, &PropertyData::Empty).expect("root");
    let mut l1 = Vec::new();
    let mut l2 = Vec::new();
    let mut l3 = Vec::new();
    for _ in 0..FANOUT {
        l1.push(create_node(&crud, &mut tx, tenant, n_label, &PropertyData::Empty).unwrap());
        l2.push(create_node(&crud, &mut tx, tenant, n_label, &PropertyData::Empty).unwrap());
        l3.push(create_node(&crud, &mut tx, tenant, n_label, &PropertyData::Empty).unwrap());
    }
    for dst in &l1 {
        create_rel(
            &crud,
            &mut tx,
            tenant,
            root,
            *dst,
            rel_type,
            &PropertyData::Empty,
        )
        .unwrap();
    }
    for src in &l1 {
        for dst in &l2 {
            create_rel(
                &crud,
                &mut tx,
                tenant,
                *src,
                *dst,
                rel_type,
                &PropertyData::Empty,
            )
            .unwrap();
        }
    }
    for src in &l2 {
        for dst in &l3 {
            create_rel(
                &crud,
                &mut tx,
                tenant,
                *src,
                *dst,
                rel_type,
                &PropertyData::Empty,
            )
            .unwrap();
        }
    }
    commit(tx, &crud).expect("commit fixture");

    let before = mgr.oldest_active_snapshot();
    let rows = raw
        .execute(
            tenant,
            "MATCH (a:Start)-[:R*3..3]->(b) RETURN b",
            1_000,
            &CancellationToken::new(),
        )
        .expect("raw_query k=3");
    assert_eq!(rows.row_count, FANOUT * FANOUT * FANOUT);
    assert!(!rows.truncated);

    let after = mgr.oldest_active_snapshot();
    assert_eq!(
        after,
        mgr.current_lsn(),
        "raw_query must release its snapshot anchor after materialization"
    );
    assert!(
        after >= before,
        "anchor-release sanity: before={before:?}, after={after:?}"
    );
}

#[test]
fn tombstoned_far_end_edge_is_dropped_by_eager_and_cursor_expand() {
    let (_raw, substrate, mgr, crud, intern) = fixture();
    let tenant = TenantId::DEFAULT;
    let label = intern.intern_label(tenant, "N").unwrap();
    let rel_type = intern.intern_type(tenant, "R").unwrap();

    let mut tx = mgr.begin(tenant);
    let src = create_node(&crud, &mut tx, tenant, label, &PropertyData::Empty).unwrap();
    let dst = create_node(&crud, &mut tx, tenant, label, &PropertyData::Empty).unwrap();
    create_rel(
        &crud,
        &mut tx,
        tenant,
        src,
        dst,
        rel_type,
        &PropertyData::Empty,
    )
    .unwrap();
    commit(tx, &crud).unwrap();

    let mut del = mgr.begin(tenant);
    arcgraph_storage::crud::delete_node(&mut del, dst).expect("non-cascade node tombstone");
    commit(del, &crud).unwrap();

    let eager = substrate
        .expand(
            tenant,
            src,
            Some(rel_type),
            Direction::LeftToRight,
            Lsn::MAX,
        )
        .expect("eager expand");
    assert!(
        eager.is_empty(),
        "eager expand drops edge whose far-end node is tombstoned"
    );

    let mut cursor = substrate
        .expand_cursor(
            tenant,
            src,
            Some(rel_type),
            Direction::LeftToRight,
            Lsn::MAX,
        )
        .expect("cursor expand");
    assert!(
        cursor.next().is_none(),
        "cursor expand shares materialize_bound_edge tombstone posture"
    );
}
