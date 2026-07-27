//! M3 §2.1 physical-delta apply gate for the two page-LSN stores.

use arcgraph_core::record::{NodeRecord, PAGE_SIZE, PageHeader, PageType, RelRecord};
use arcgraph_core::{ArcGraphError, LabelId, Lsn, NodeId, PageId, RelId, TenantId, TypeId};
use arcgraph_storage::apply_physical_delta;
use arcgraph_storage::records::{SlotId, SlottedPage, SlottedPageRef};
use arcgraph_storage::wal::{DeltaOp, DeltaOpKind, STORE_BLOB_OVERFLOW, STORE_PROPS, STORE_RECORD};
use bytes::Bytes;

fn fresh_page(page_no: u64, page_type: PageType, tenant: TenantId) -> [u8; PAGE_SIZE] {
    let mut bytes = [0u8; PAGE_SIZE];
    SlottedPage::init(
        &mut bytes,
        PageHeader::new(PageId::new(page_no), page_type, tenant),
    )
    .unwrap();
    bytes
}

fn put_record(page_no: u64, slot: u16, op_lsn: u64, payload: impl Into<Bytes>) -> DeltaOp {
    DeltaOp::new(
        DeltaOpKind::PutRecord,
        STORE_RECORD,
        TenantId::DEFAULT,
        page_no,
        slot,
        Lsn::new(op_lsn),
        payload,
    )
    .unwrap()
}

fn stamp_oracle_lsn(bytes: &mut [u8; PAGE_SIZE], lsn: Lsn) {
    bytes[16..24].copy_from_slice(&lsn.raw().to_le_bytes());
}

#[test]
fn put_record_and_node_tombstone_match_direct_page_image_oracle_byte_for_byte() {
    let commit_lsn = Lsn::new(13);
    let mut delta_page = fresh_page(7, PageType::Node, TenantId::DEFAULT);
    let mut oracle_page = delta_page;

    let mut first = NodeRecord::new(NodeId::new(1), LabelId::new(2), commit_lsn);
    let second = NodeRecord::new(NodeId::new(2), LabelId::new(2), commit_lsn);
    let ops = [
        put_record(7, 0, 10, Bytes::copy_from_slice(&first.to_bytes())),
        put_record(7, 1, 11, Bytes::copy_from_slice(&second.to_bytes())),
        {
            first.inline_u32a = 99;
            put_record(7, 0, 12, Bytes::copy_from_slice(&first.to_bytes()))
        },
        DeltaOp::new(
            DeltaOpKind::TombstoneRecord,
            STORE_RECORD,
            TenantId::DEFAULT,
            7,
            1,
            Lsn::new(13),
            Bytes::new(),
        )
        .unwrap(),
    ];

    for op in &ops {
        assert!(apply_physical_delta(&mut delta_page, op, commit_lsn).unwrap());
    }

    {
        let mut oracle = SlottedPage::open(&mut oracle_page).unwrap();
        oracle
            .put_node_at(
                SlotId(0),
                &NodeRecord::new(NodeId::new(1), LabelId::new(2), commit_lsn),
            )
            .unwrap();
    }
    stamp_oracle_lsn(&mut oracle_page, Lsn::new(10));
    {
        let mut oracle = SlottedPage::open(&mut oracle_page).unwrap();
        oracle.put_node_at(SlotId(1), &second).unwrap();
    }
    stamp_oracle_lsn(&mut oracle_page, Lsn::new(11));
    {
        let mut oracle = SlottedPage::open(&mut oracle_page).unwrap();
        oracle.put_node_at(SlotId(0), &first).unwrap();
    }
    stamp_oracle_lsn(&mut oracle_page, Lsn::new(12));
    {
        let mut oracle = SlottedPage::open(&mut oracle_page).unwrap();
        oracle.tombstone(SlotId(1)).unwrap();
    }
    stamp_oracle_lsn(&mut oracle_page, Lsn::new(13));

    assert_eq!(delta_page, oracle_page);
    let once = delta_page;
    for op in &ops {
        assert!(!apply_physical_delta(&mut delta_page, op, commit_lsn).unwrap());
    }
    assert_eq!(delta_page, once, "replay twice must be byte-identical");
}

#[test]
fn same_props_page_applies_one_hundred_exact_slot_sub_lsns() {
    let mut page = fresh_page(8, PageType::PropSlotted, TenantId::new(2));
    for slot in 0..100u16 {
        let payload = Bytes::from(vec![slot as u8; 32]);
        let op = DeltaOp::new(
            DeltaOpKind::PutPropBlock,
            STORE_PROPS,
            TenantId::new(2),
            8,
            slot,
            Lsn::new(100 + u64::from(slot)),
            payload.clone(),
        )
        .unwrap();
        assert!(apply_physical_delta(&mut page, &op, Lsn::new(199)).unwrap());
    }

    let opened = SlottedPageRef::open(&page).unwrap();
    assert_eq!(opened.slot_count(), 100);
    assert_eq!(opened.header().lsn, 199);
    for slot in 0..100u16 {
        assert_eq!(
            opened.read_bag(SlotId(slot)).unwrap().unwrap(),
            &[slot as u8; 32]
        );
    }
}

#[test]
fn relationship_tombstone_uses_commit_clock_and_op_clock_only_for_page_lsn() {
    let mut page = fresh_page(9, PageType::Rel, TenantId::DEFAULT);
    let rel = RelRecord::new(
        RelId::new(4),
        TypeId::new(3),
        NodeId::new(1),
        NodeId::new(2),
        Lsn::new(30),
    );
    let put = put_record(9, 0, 40, Bytes::copy_from_slice(&rel.to_bytes()));
    let tombstone = DeltaOp::new(
        DeltaOpKind::TombstoneRecord,
        STORE_RECORD,
        TenantId::DEFAULT,
        9,
        0,
        Lsn::new(41),
        Bytes::new(),
    )
    .unwrap();
    assert!(apply_physical_delta(&mut page, &put, Lsn::new(42)).unwrap());
    assert!(apply_physical_delta(&mut page, &tombstone, Lsn::new(42)).unwrap());

    let opened = SlottedPageRef::open(&page).unwrap();
    let recovered = opened.read_rel(SlotId(0)).unwrap().unwrap();
    assert_eq!(recovered.expired_lsn, 42);
    assert_eq!(opened.header().lsn, 41);
}

#[test]
fn older_conflicting_delta_is_skipped_by_page_lsn() {
    let mut page = fresh_page(10, PageType::Node, TenantId::DEFAULT);
    let current = NodeRecord::new(NodeId::new(10), LabelId::new(1), Lsn::new(60));
    let newer = put_record(10, 0, 60, Bytes::copy_from_slice(&current.to_bytes()));
    assert!(apply_physical_delta(&mut page, &newer, Lsn::new(60)).unwrap());
    let once = page;

    let stale_payload = NodeRecord::new(NodeId::new(999), LabelId::new(9), Lsn::new(59));
    let stale = put_record(10, 0, 59, Bytes::copy_from_slice(&stale_payload.to_bytes()));
    assert!(!apply_physical_delta(&mut page, &stale, Lsn::new(60)).unwrap());
    assert_eq!(
        page, once,
        "stale conflicting bytes must not overwrite the page"
    );
}

#[test]
fn wrong_target_fails_and_reserved_slot_gap_is_an_explicit_tombstone() {
    let mut page = fresh_page(11, PageType::Node, TenantId::DEFAULT);
    let before = page;
    let record = NodeRecord::new(NodeId::new(1), LabelId::new(1), Lsn::new(1));
    let wrong_page = put_record(12, 0, 1, Bytes::copy_from_slice(&record.to_bytes()));
    let error = apply_physical_delta(&mut page, &wrong_page, Lsn::new(1)).unwrap_err();
    assert!(matches!(error, ArcGraphError::WalCorruption { .. }));
    assert_eq!(page, before);

    let gap = put_record(11, 1, 1, Bytes::copy_from_slice(&record.to_bytes()));
    assert!(apply_physical_delta(&mut page, &gap, Lsn::new(1)).unwrap());
    let opened = SlottedPageRef::open(&page).unwrap();
    assert_eq!(opened.slot_count(), 2);
    assert_eq!(opened.read_node(SlotId(0)).unwrap(), None);
    assert_eq!(opened.read_node(SlotId(1)).unwrap(), Some(record));
    assert_eq!(opened.header().lsn, 1);
    let once = page;
    assert!(!apply_physical_delta(&mut page, &gap, Lsn::new(1)).unwrap());
    assert_eq!(
        page, once,
        "redo of the sparse durable slot is byte-idempotent"
    );
}

#[test]
fn blob_overflow_is_not_a_physical_delta_apply_store_at_m3() {
    assert_eq!(STORE_BLOB_OVERFLOW, 5);
    let page = fresh_page(12, PageType::PropSlotted, TenantId::DEFAULT);
    let before = page;
    let valid = DeltaOp::new(
        DeltaOpKind::PutPropBlock,
        STORE_PROPS,
        TenantId::DEFAULT,
        12,
        0,
        Lsn::new(1),
        Bytes::from_static(b"typed"),
    )
    .unwrap();
    let mut wire = Vec::new();
    valid.encode_into(&mut wire).unwrap();
    wire[1..3].copy_from_slice(&STORE_BLOB_OVERFLOW.to_le_bytes());
    assert!(DeltaOp::decode_prefix(&wire).is_err());
    assert_eq!(page, before);
}
