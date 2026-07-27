//! End-to-end dual-write ↔ slotted-page round-trip property test
//! (closes #28).
//!
//! Exercises the first real interaction between `records::SlottedPage`
//! (the 1007-line M2-20 codec) and a `CrudStore` commit flow wired for
//! dual-write:
//!
//! ```text
//! create_node → tx.commit → drain install → SlottedPage::insert_node
//!   → primary_index.upsert → (new txn) → primary_index.lookup
//!   → SlottedPageRef::read_node → assert every field round-trips
//! ```
//!
//! Run with `cargo test --release --test crud_dual_write_records_cycle`
//! for the full 5 000-case sweep; a debug run uses a smaller case count
//! to stay responsive in CI.
//!
//! Coverage:
//! - Create / lookup / field round-trip for `NodeRecord`.
//! - Create / lookup / field round-trip for `RelRecord`.
//! - In-place update rewrites the same slot, preserves topology
//!   fields, changes only property fields.
//! - Delete tombstones the slot AND the primary-index entry;
//!   post-delete lookups return `None`.

use std::sync::Arc;

use arcgraph_core::{LabelId, NodeId, RelId, TenantId, TypeId};
use arcgraph_storage::crud::{
    CrudStore, PropertyData, commit, create_node, create_rel, delete_node_with_store,
    delete_rel_with_store, read_node_with_store, read_rel_with_store, update_node, update_rel,
};
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::{PrimaryIndex, PrimaryKey, RecordKind};
use arcgraph_storage::records::SlottedPageRef;
use arcgraph_storage::transaction::TxnManager;
use proptest::prelude::*;

fn build_store() -> (Arc<TxnManager>, CrudStore, Arc<PrimaryIndex>) {
    let txn_mgr = Arc::new(TxnManager::new());
    let alloc = Arc::new(PageAllocator::new());
    let primary =
        Arc::new(PrimaryIndex::new(Arc::clone(&txn_mgr), Arc::clone(&alloc), None).unwrap());
    let store = CrudStore::new_with_index(None, Arc::clone(&primary), Arc::clone(&alloc));
    (txn_mgr, store, primary)
}

fn cycle_node(label: u32, prop_a: u32, prop_b: u32) -> Result<(), TestCaseError> {
    let (mgr, store, primary) = build_store();
    let mut tx = mgr.begin(TenantId::DEFAULT);
    let id = create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(label),
        &PropertyData::InlineU32Pair(prop_a, prop_b),
    )
    .unwrap();
    let commit_lsn = commit(tx, &store).unwrap();

    let key = PrimaryKey::new(TenantId::DEFAULT, RecordKind::Node, id.raw());
    let slot = primary
        .lookup(key)
        .unwrap()
        .expect("post-commit lookup must hit");
    // Direct `records::SlottedPage` read (not through the fast-path
    // helper) — this is the explicit "SlottedPage exercised from
    // commit flow" cycle that #28 calls out.
    let records = store.records().unwrap();
    let latch = records.latch(slot.page).unwrap();
    let g = latch.read();
    let page = SlottedPageRef::open(g.as_ref().as_ref()).unwrap();
    let rec = page.read_node(slot.slot).unwrap().expect("live slot");
    prop_assert_eq!(rec.id, id.raw());
    prop_assert_eq!(rec.label_id, label);
    prop_assert_eq!(rec.inline_u32a, prop_a);
    prop_assert_eq!(rec.inline_u32b, prop_b);
    // `created_lsn` was fixed up from Lsn::ZERO at install time.
    // NodeRecord doesn't carry `expired_lsn` (MVCC owns visibility);
    // RelRecord does — asserted in `cycle_rel`.
    prop_assert_eq!(rec.created_lsn, commit_lsn.raw());
    Ok(())
}

fn cycle_rel(
    src_raw: u64,
    dst_raw: u64,
    ty: u32,
    prop_a: u32,
    prop_b: u32,
) -> Result<(), TestCaseError> {
    let (mgr, store, primary) = build_store();
    // Rel endpoints only carry node-id bits; the test uses arbitrary
    // values since we don't also need node records to exist.
    let src = NodeId::new((src_raw & 0x7FFF_FFFF_FFFF_FFFF).max(1));
    let dst = NodeId::new((dst_raw & 0x7FFF_FFFF_FFFF_FFFF).max(1));
    let mut tx = mgr.begin(TenantId::DEFAULT);
    let id = create_rel(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        src,
        dst,
        TypeId::new(ty),
        &PropertyData::InlineU32Pair(prop_a, prop_b),
    )
    .unwrap();
    let commit_lsn = commit(tx, &store).unwrap();

    let key = PrimaryKey::new(TenantId::DEFAULT, RecordKind::Rel, id.raw());
    let slot = primary
        .lookup(key)
        .unwrap()
        .expect("post-commit lookup must hit");
    let records = store.records().unwrap();
    let latch = records.latch(slot.page).unwrap();
    let g = latch.read();
    let page = SlottedPageRef::open(g.as_ref().as_ref()).unwrap();
    let rec = page.read_rel(slot.slot).unwrap().expect("live slot");
    prop_assert_eq!(rec.id, id.raw());
    prop_assert_eq!(rec.type_id, ty);
    prop_assert_eq!(rec.src_id, src.raw());
    prop_assert_eq!(rec.dst_id, dst.raw());
    prop_assert_eq!(rec.inline_u32a, prop_a);
    prop_assert_eq!(rec.inline_u32b, prop_b);
    prop_assert_eq!(rec.created_lsn, commit_lsn.raw());
    prop_assert_eq!(rec.expired_lsn, u64::MAX);
    Ok(())
}

fn cycle_update(label: u32, before_a: u32, after_a: u32) -> Result<(), TestCaseError> {
    let (mgr, store, primary) = build_store();
    let mut tx = mgr.begin(TenantId::DEFAULT);
    let id = create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(label),
        &PropertyData::InlineU32Pair(before_a, before_a ^ 0xAA),
    )
    .unwrap();
    commit(tx, &store).unwrap();
    let key = PrimaryKey::new(TenantId::DEFAULT, RecordKind::Node, id.raw());
    let slot_before = primary.lookup(key).unwrap().unwrap();

    let mut tx = mgr.begin(TenantId::DEFAULT);
    update_node(
        &store,
        &mut tx,
        id,
        &PropertyData::InlineU32Pair(after_a, after_a ^ 0x55),
    )
    .unwrap();
    commit(tx, &store).unwrap();

    // Update rewrites in place: the primary-index slot is unchanged.
    let slot_after = primary.lookup(key).unwrap().unwrap();
    prop_assert_eq!(slot_before, slot_after);

    // The page now carries the new property bytes.
    let records = store.records().unwrap();
    let latch = records.latch(slot_after.page).unwrap();
    let g = latch.read();
    let page = SlottedPageRef::open(g.as_ref().as_ref()).unwrap();
    let rec = page.read_node(slot_after.slot).unwrap().unwrap();
    prop_assert_eq!(rec.label_id, label);
    prop_assert_eq!(rec.inline_u32a, after_a);
    prop_assert_eq!(rec.inline_u32b, after_a ^ 0x55);
    Ok(())
}

fn cycle_delete(label: u32) -> Result<(), TestCaseError> {
    let (mgr, store, primary) = build_store();
    let mut tx = mgr.begin(TenantId::DEFAULT);
    let id = create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(label),
        &PropertyData::Empty,
    )
    .unwrap();
    commit(tx, &store).unwrap();

    let mut tx = mgr.begin(TenantId::DEFAULT);
    delete_node_with_store(&store, &mut tx, id).unwrap();
    commit(tx, &store).unwrap();

    let key = PrimaryKey::new(TenantId::DEFAULT, RecordKind::Node, id.raw());
    prop_assert_eq!(primary.lookup(key).unwrap(), None);
    // Fast-path read returns None (index miss → MVCC sees tombstone → None).
    let reader = mgr.begin(TenantId::DEFAULT);
    prop_assert!(read_node_with_store(&store, &reader, id).unwrap().is_none());
    Ok(())
}

fn cycle_rel_update_and_delete(ty: u32, before_a: u32, after_a: u32) -> Result<(), TestCaseError> {
    // Single combined test to cover rel update AND rel delete in one
    // proptest case so the shared fixture amortizes.
    let (mgr, store, primary) = build_store();
    let src = NodeId::new(1);
    let dst = NodeId::new(2);
    let mut tx = mgr.begin(TenantId::DEFAULT);
    let id = create_rel(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        src,
        dst,
        TypeId::new(ty),
        &PropertyData::InlineU32Pair(before_a, 0),
    )
    .unwrap();
    commit(tx, &store).unwrap();

    let mut tx = mgr.begin(TenantId::DEFAULT);
    update_rel(
        &store,
        &mut tx,
        id,
        &PropertyData::InlineU32Pair(after_a, 1),
    )
    .unwrap();
    commit(tx, &store).unwrap();

    let reader = mgr.begin(TenantId::DEFAULT);
    let rec = read_rel_with_store(&store, &reader, id).unwrap().unwrap();
    prop_assert_eq!(rec.inline_u32a, after_a);
    prop_assert_eq!(rec.inline_u32b, 1);
    prop_assert_eq!(rec.type_id, ty);
    prop_assert_eq!(rec.src_id, src.raw());
    prop_assert_eq!(rec.dst_id, dst.raw());
    drop(reader);

    let mut tx = mgr.begin(TenantId::DEFAULT);
    delete_rel_with_store(&store, &mut tx, id).unwrap();
    commit(tx, &store).unwrap();

    let key = PrimaryKey::new(TenantId::DEFAULT, RecordKind::Rel, id.raw());
    prop_assert_eq!(primary.lookup(key).unwrap(), None);
    let reader = mgr.begin(TenantId::DEFAULT);
    prop_assert!(read_rel_with_store(&store, &reader, id).unwrap().is_none());
    let _ = id;
    // Silence "unused" on RelId helper import.
    let _type_hint: Option<RelId> = None;
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 5_000,
        .. ProptestConfig::default()
    })]

    #[test]
    fn node_create_cycle_roundtrips(
        label in any::<u32>(),
        a in any::<u32>(),
        b in any::<u32>(),
    ) {
        cycle_node(label, a, b)?;
    }

    #[test]
    fn rel_create_cycle_roundtrips(
        ty in any::<u32>(),
        src_raw in any::<u64>(),
        dst_raw in any::<u64>(),
        a in any::<u32>(),
        b in any::<u32>(),
    ) {
        cycle_rel(src_raw, dst_raw, ty, a, b)?;
    }

    #[test]
    fn node_update_cycle_rewrites_in_place(
        label in any::<u32>(),
        before in any::<u32>(),
        after in any::<u32>(),
    ) {
        cycle_update(label, before, after)?;
    }

    #[test]
    fn node_delete_cycle_tombstones(label in any::<u32>()) {
        cycle_delete(label)?;
    }

    #[test]
    fn rel_update_then_delete_cycle(
        ty in any::<u32>(),
        before in any::<u32>(),
        after in any::<u32>(),
    ) {
        cycle_rel_update_and_delete(ty, before, after)?;
    }
}
