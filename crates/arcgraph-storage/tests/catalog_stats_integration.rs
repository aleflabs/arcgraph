//! M4-41 (M4-04a) catalog-stats commit-pipeline integration tests
//! per ADR-038 §2 D-25.
//!
//! Exercises the post-commit hook in `crud::commit` that walks the
//! drained `installs` vector and updates the per-tenant
//! [`arcgraph_storage::CatalogStats`] counters. The unit tests in
//! `arcgraph-storage::catalog::stats::tests` cover the counter
//! semantics in isolation; these integration tests pin the
//! commit-pipeline wiring (the live MVCC + dual-write commit flow
//! actually moves the counters).

use std::sync::Arc;

use arcgraph_core::{LabelId, NodeId, TenantId, TypeId};
use arcgraph_storage::crud::{
    CrudStore, PropertyData, commit, create_node, create_rel, delete_node_with_store,
};
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::PrimaryIndex;
use arcgraph_storage::transaction::TxnManager;

fn build_store() -> (Arc<TxnManager>, CrudStore) {
    let txn_mgr = Arc::new(TxnManager::new());
    let alloc = Arc::new(PageAllocator::new());
    let primary =
        Arc::new(PrimaryIndex::new(Arc::clone(&txn_mgr), Arc::clone(&alloc), None).unwrap());
    let store = CrudStore::new_with_index(None, Arc::clone(&primary), Arc::clone(&alloc));
    (txn_mgr, store)
}

#[test]
fn commit_node_creation_increments_label_cardinality() {
    // M4-41 D-25: after a commit that creates two nodes carrying the
    // same label, the per-label cardinality MUST equal the count of
    // committed creations and the tenant-wide total MUST match. Pre-
    // commit there are no stats (None) — the boundary contract pins
    // "fresh tenant returns None" until the first commit lands.
    let (mgr, store) = build_store();
    let person = LabelId::new(7);
    let doc = LabelId::new(8);

    // Pre-commit: a fresh tenant has no stats yet.
    assert!(
        store.catalog_stats(TenantId::DEFAULT).is_none(),
        "no commits yet, no stats instance materialized"
    );

    let mut tx = mgr.begin(TenantId::DEFAULT);
    let _n1 = create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        person,
        &PropertyData::Empty,
    )
    .unwrap();
    let _n2 = create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        person,
        &PropertyData::Empty,
    )
    .unwrap();
    let _n3 = create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        doc,
        &PropertyData::InlineU32Pair(1, 2),
    )
    .unwrap();
    commit(tx, &store).unwrap();

    let stats = store
        .catalog_stats(TenantId::DEFAULT)
        .expect("first commit must materialise stats");
    assert_eq!(stats.label_cardinality(person), Some(2));
    assert_eq!(stats.label_cardinality(doc), Some(1));
    // A label that was never committed stays None — NOT Some(0).
    assert_eq!(stats.label_cardinality(LabelId::new(99)), None);
    // Tenant-wide total surfaces post-first-commit.
    assert_eq!(stats.total_node_count(), Some(3));
    // Rel total observed but unmoved — Some(0), not None, because
    // the first commit triggered observe_commit() once.
    assert_eq!(stats.total_rel_count(), Some(0));

    // Cross-tenant isolation: a different tenant's stats are
    // independent and remain None until that tenant commits.
    assert!(
        store.catalog_stats(TenantId::new(42)).is_none(),
        "tenant DEFAULT's commit must not leak into tenant 42"
    );
}

#[test]
fn commit_node_deletion_decrements_label_cardinality() {
    // M4-41 D-25: a commit that tombstones a node decrements both
    // the per-label cardinality and the tenant-wide total. The
    // delete path captures the prior label via the
    // `delete_node_with_store` API so the post-commit hook knows
    // which counter to move.
    let (mgr, store) = build_store();
    let person = LabelId::new(7);

    // Create two persons in tx 1.
    let mut tx1 = mgr.begin(TenantId::DEFAULT);
    let n1 = create_node(
        &store,
        &mut tx1,
        TenantId::DEFAULT,
        person,
        &PropertyData::Empty,
    )
    .unwrap();
    let _n2 = create_node(
        &store,
        &mut tx1,
        TenantId::DEFAULT,
        person,
        &PropertyData::Empty,
    )
    .unwrap();
    commit(tx1, &store).unwrap();
    {
        let stats = store.catalog_stats(TenantId::DEFAULT).unwrap();
        assert_eq!(stats.label_cardinality(person), Some(2));
        assert_eq!(stats.total_node_count(), Some(2));
    }

    // Delete n1 in tx 2.
    let mut tx2 = mgr.begin(TenantId::DEFAULT);
    delete_node_with_store(&store, &mut tx2, n1).unwrap();
    commit(tx2, &store).unwrap();

    let stats = store.catalog_stats(TenantId::DEFAULT).unwrap();
    assert_eq!(stats.label_cardinality(person), Some(1));
    assert_eq!(stats.total_node_count(), Some(1));
}

#[test]
fn commit_relationship_creation_increments_rel_type_cardinality() {
    // M4-41 D-25: the rel-side surface mirrors the node side —
    // creating relationships of a given type increments the
    // per-type cardinality and the tenant-wide rel total. The node
    // total stays at zero (no node creation in this test).
    let (mgr, store) = build_store();
    let knows = TypeId::new(1);
    let wrote = TypeId::new(2);

    let mut tx = mgr.begin(TenantId::DEFAULT);
    // Create a few rels. Endpoints are arbitrary node ids — the
    // primary-index round-trip test in
    // `crud_dual_write_records_cycle` already pins that the
    // endpoints don't need real node records for create_rel /
    // commit to land. NodeId values must be non-zero (NodeId::ZERO
    // is the sentinel) per `arcgraph-core::ids`.
    let src = NodeId::new(1);
    let dst = NodeId::new(2);
    for _ in 0..3 {
        create_rel(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            src,
            dst,
            knows,
            &PropertyData::Empty,
        )
        .unwrap();
    }
    create_rel(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        src,
        dst,
        wrote,
        &PropertyData::Empty,
    )
    .unwrap();
    commit(tx, &store).unwrap();

    let stats = store.catalog_stats(TenantId::DEFAULT).unwrap();
    assert_eq!(stats.rel_type_cardinality(knows), Some(3));
    assert_eq!(stats.rel_type_cardinality(wrote), Some(1));
    assert_eq!(stats.rel_type_cardinality(TypeId::new(99)), None);
    // Tenant-wide totals: rels moved, nodes did not (this commit
    // never created a node).
    assert_eq!(stats.total_rel_count(), Some(4));
    assert_eq!(stats.total_node_count(), Some(0));
}
