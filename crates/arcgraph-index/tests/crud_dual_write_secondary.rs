//! M2-34 CUTOVER wiring tests: exercise the full `crud::commit` drain
//! with a `SecondaryIndex` attached and observe the published
//! property→NodeId entries.
//!
//! These tests live in `arcgraph-index/tests/` so they can directly
//! reference both `arcgraph_storage::crud` (via `new_with_indices`)
//! and the concrete `SecondaryIndex` — the storage crate's own unit
//! tests cannot import arcgraph-index because the dependency graph is
//! `index → storage`.

use std::sync::Arc;

use arcgraph_core::{LabelId, StringId, TenantId, TypeId};
use arcgraph_index::{PropertyValue, SecondaryIndex, SecondaryKey};
use arcgraph_storage::crud::{
    CrudStore, INLINE_U32A_PROPERTY_KEY, INLINE_U32B_PROPERTY_KEY, PropertyData, commit,
    create_node, create_rel, delete_node_with_store, update_node,
};
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::PrimaryIndex;
use arcgraph_storage::secondary_handle::SecondaryIndexHandle;
use arcgraph_storage::transaction::TxnManager;

fn build_store() -> (Arc<TxnManager>, CrudStore, Arc<SecondaryIndex>) {
    let txn_mgr = Arc::new(TxnManager::new());
    let alloc = Arc::new(PageAllocator::new());
    let primary =
        Arc::new(PrimaryIndex::new(Arc::clone(&txn_mgr), Arc::clone(&alloc), None).unwrap());
    let secondary =
        Arc::new(SecondaryIndex::new(Arc::clone(&txn_mgr), Arc::clone(&alloc), None).unwrap());
    let handle: Arc<dyn SecondaryIndexHandle> = Arc::clone(&secondary) as _;
    let store =
        CrudStore::new_with_indices(None, Arc::clone(&primary), Some(handle), Arc::clone(&alloc));
    (txn_mgr, store, secondary)
}

fn key(label: u32, pk: StringId, v: u32) -> SecondaryKey {
    SecondaryKey::new(
        TenantId::DEFAULT,
        LabelId::new(label),
        pk,
        PropertyValue::U32(v),
    )
}

#[test]
fn dual_write_create_emits_secondary_entry_per_property() {
    let (mgr, store, secondary) = build_store();
    let mut tx = mgr.begin(TenantId::DEFAULT);
    let id = create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(42),
        &PropertyData::InlineU32Pair(7, 8),
    )
    .unwrap();
    commit(tx, &store).unwrap();

    // Two entries: one for inline_u32a = 7, one for inline_u32b = 8.
    let hits_a = secondary
        .lookup(key(42, INLINE_U32A_PROPERTY_KEY, 7))
        .unwrap();
    let hits_b = secondary
        .lookup(key(42, INLINE_U32B_PROPERTY_KEY, 8))
        .unwrap();
    assert_eq!(hits_a, vec![id]);
    assert_eq!(hits_b, vec![id]);
}

#[test]
fn dual_write_update_diffs_property_set() {
    // RC-1 (#1366): under insert-only commit-path maintenance the NEW
    // value is inserted synchronously, but the OLD value's removal is
    // DEFERRED past the snapshot horizon (it lingers as a read-safe
    // ghost). This test pins the RC-1 contract: the ghost is present
    // immediately after the update, then reclaimed once a later commit
    // advances the horizon past the update's commit LSN.
    let (mgr, store, secondary) = build_store();
    // Initial create: (11, 22)
    let mut tx = mgr.begin(TenantId::DEFAULT);
    let id = create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(1),
        &PropertyData::InlineU32Pair(11, 22),
    )
    .unwrap();
    commit(tx, &store).unwrap();

    // Update: (11, 99) — inline_u32a unchanged, inline_u32b changes.
    let mut tx = mgr.begin(TenantId::DEFAULT);
    update_node(&store, &mut tx, id, &PropertyData::InlineU32Pair(11, 99)).unwrap();
    commit(tx, &store).unwrap();

    // inline_u32a value 11 is still indexed (unchanged, not churned).
    assert_eq!(
        secondary
            .lookup(key(1, INLINE_U32A_PROPERTY_KEY, 11))
            .unwrap(),
        vec![id]
    );
    // inline_u32b at new value 99 is indexed synchronously.
    assert_eq!(
        secondary
            .lookup(key(1, INLINE_U32B_PROPERTY_KEY, 99))
            .unwrap(),
        vec![id]
    );
    // RC-1: inline_u32b at OLD value 22 is a GHOST — still present in
    // the B-tree immediately after the update, because the update's own
    // snapshot pinned the horizon so its removal is not yet ready. This
    // is read-safe: a reader hydrates n, sees inline_u32b=99, and the
    // mandatory verify step drops the =22 candidate.
    assert_eq!(
        secondary
            .lookup(key(1, INLINE_U32B_PROPERTY_KEY, 22))
            .unwrap(),
        vec![id],
        "RC-1: the old value 22 is a deferred-removal ghost, not yet reclaimed"
    );
    assert_eq!(
        store.deferred_removal_queue_len(),
        1,
        "the old-value removal is queued awaiting the snapshot horizon"
    );

    // Advance the horizon: a later commit (with no live reader pinning
    // the old snapshot) reads a horizon past the update's commit LSN and
    // drains the pending removal.
    let mut tx = mgr.begin(TenantId::DEFAULT);
    let _other = create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(1),
        &PropertyData::InlineU32Pair(1, 2),
    )
    .unwrap();
    commit(tx, &store).unwrap();

    // Now the ghost is reclaimed.
    assert!(
        secondary
            .lookup(key(1, INLINE_U32B_PROPERTY_KEY, 22))
            .unwrap()
            .is_empty(),
        "RC-1: old inline_u32b = 22 reclaimed once the horizon passed it",
    );
    assert_eq!(
        store.deferred_removal_queue_len(),
        0,
        "the deferred-removal queue drained once the horizon cleared it"
    );
}

#[test]
fn dual_write_delete_removes_all_secondary_entries_for_node() {
    let (mgr, store, secondary) = build_store();
    let mut tx = mgr.begin(TenantId::DEFAULT);
    let id = create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(5),
        &PropertyData::InlineU32Pair(100, 200),
    )
    .unwrap();
    commit(tx, &store).unwrap();
    assert_eq!(
        secondary
            .lookup(key(5, INLINE_U32A_PROPERTY_KEY, 100))
            .unwrap(),
        vec![id]
    );

    let mut tx = mgr.begin(TenantId::DEFAULT);
    delete_node_with_store(&store, &mut tx, id).unwrap();
    commit(tx, &store).unwrap();

    // RC-1 (#1366): the delete's removals are DEFERRED past the snapshot
    // horizon (read-safe ghosts). Immediately after the delete commit
    // they are still present — a pre-delete snapshot reader would still
    // see the entry (and the MVCC verify step drops it on the tombstone,
    // never a false negative from a missing entry).
    assert_eq!(
        secondary
            .lookup(key(5, INLINE_U32A_PROPERTY_KEY, 100))
            .unwrap(),
        vec![id],
        "RC-1: delete removals are deferred ghosts, not synchronous"
    );
    assert_eq!(store.deferred_removal_queue_len(), 2);

    // Advance the horizon with a later commit to drain the pending
    // removals.
    let mut tx = mgr.begin(TenantId::DEFAULT);
    let _other = create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(5),
        &PropertyData::InlineU32Pair(1, 2),
    )
    .unwrap();
    commit(tx, &store).unwrap();

    assert!(
        secondary
            .lookup(key(5, INLINE_U32A_PROPERTY_KEY, 100))
            .unwrap()
            .is_empty(),
        "RC-1: inline_u32a entry reclaimed once the horizon passed the delete LSN"
    );
    assert!(
        secondary
            .lookup(key(5, INLINE_U32B_PROPERTY_KEY, 200))
            .unwrap()
            .is_empty(),
        "RC-1: inline_u32b entry reclaimed once the horizon passed the delete LSN"
    );
    assert_eq!(store.deferred_removal_queue_len(), 0);
}

#[test]
fn dual_write_rel_does_not_touch_secondary() {
    let (mgr, store, secondary) = build_store();
    // Create two nodes to anchor the rel.
    let mut tx = mgr.begin(TenantId::DEFAULT);
    let src = create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(1),
        &PropertyData::Empty,
    )
    .unwrap();
    let dst = create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(1),
        &PropertyData::Empty,
    )
    .unwrap();
    let _rel = create_rel(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        src,
        dst,
        TypeId::new(77),
        &PropertyData::InlineU32Pair(500, 600),
    )
    .unwrap();
    commit(tx, &store).unwrap();

    // Rel properties (500, 600) must NOT show up in the secondary —
    // rels are not indexed in M2-34.
    assert!(
        secondary
            .lookup(key(77, INLINE_U32A_PROPERTY_KEY, 500))
            .unwrap()
            .is_empty(),
        "rel property values must not leak into the node secondary"
    );
    // Nodes with Empty props still leave the (inline_u32a=0,
    // inline_u32b=0) positional entries — that's the DEC-3 "Empty
    // vs U32Pair(0,0) are indistinguishable on disk" consequence.
    // Verify that the src/dst node ids ARE in the (0, 0) bucket as a
    // positive confirmation the drain did run for nodes.
    let zero_hits = secondary
        .lookup(key(1, INLINE_U32A_PROPERTY_KEY, 0))
        .unwrap();
    assert!(zero_hits.contains(&src));
    assert!(zero_hits.contains(&dst));
}

#[test]
fn secondary_lookup_post_filters_via_read_node() {
    // Demonstrates the ADR-023 contract: a secondary lookup yields
    // candidate NodeIds; callers verify snapshot visibility by calling
    // read_node. After a delete, the secondary *is* cleaned up
    // synchronously (via the drain), so a post-delete reader sees no
    // entries for the deleted node — but even if a stale id leaked
    // through (e.g. pre-drain window), `read_node` would return None
    // at the deleter's snapshot.
    let (mgr, store, secondary) = build_store();
    let mut tx = mgr.begin(TenantId::DEFAULT);
    let id = create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(10),
        &PropertyData::InlineU32Pair(300, 400),
    )
    .unwrap();
    commit(tx, &store).unwrap();
    assert_eq!(
        secondary
            .lookup(key(10, INLINE_U32A_PROPERTY_KEY, 300))
            .unwrap(),
        vec![id]
    );

    // Delete the node.
    let mut tx = mgr.begin(TenantId::DEFAULT);
    delete_node_with_store(&store, &mut tx, id).unwrap();
    commit(tx, &store).unwrap();

    // RC-1 (#1366): post-delete the secondary entry LINGERS as a
    // deferred-removal ghost (removal deferred past the snapshot
    // horizon). This is the ADR-023 read-accelerator contract in its
    // load-bearing form: the index MAY return a stale candidate; the
    // MANDATORY verify step drops it. The ghost is present here...
    let hits = secondary
        .lookup(key(10, INLINE_U32A_PROPERTY_KEY, 300))
        .unwrap();
    assert_eq!(
        hits,
        vec![id],
        "RC-1: deleted node's entry lingers as a deferred-removal ghost"
    );

    // ...and the MVCC visibility gate filters it: crud::read_node at the
    // post-delete snapshot returns None, so the ghost never reaches a
    // query row. This is exactly why a lingering ghost is read-safe
    // while a MISSING entry (the pre-RC-1 eager-removal hazard) is not.
    let reader = mgr.begin(TenantId::DEFAULT);
    let rec = arcgraph_storage::crud::read_node(&reader, id).unwrap();
    assert!(rec.is_none(), "MVCC visibility gate filters deleted nodes");
}
