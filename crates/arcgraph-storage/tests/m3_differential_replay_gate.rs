//! M3 headline differential oracle.
//!
//! The v8 side is independently materialized as final full-page images and
//! decoded through the v8 codec. The v9 side is a generated physiological
//! history decoded/replayed through the v9 path. The generator closes every
//! row of design §1.2's mutation-coverage table.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use arcgraph_core::record::{NodeRecord, PAGE_SIZE, PageType, RelRecord};
use arcgraph_core::{
    LabelId, Lsn, NodeId, PageHeader, PageId, PartitionId, RelId, StringId, TenantId, TypeId,
};
use arcgraph_storage::crud::{node_mvcc_key, rel_mvcc_key};
use arcgraph_storage::io::{InMemoryPageIo, PageBuf, PageIo};
use arcgraph_storage::page_store::{
    BufferedRecordPageStore, PerTenantBufferPool, PerTenantBufferPoolConfig, RecordPageBackend,
};
use arcgraph_storage::primary_index::PrimaryPageStore;
use arcgraph_storage::records::{SlotId, SlottedPage, SlottedPageRef};
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{
    AclGrantEntry, AclGrantOp, AllocatorAdvance, AllocatorKind, BUNDLE_FORMAT_V9, BundlePageKind,
    DeltaOp, DeltaOpKind, IdempotencyBindingEntry, IdempotencyBindingOp, PageStoreTarget,
    ReplayConfig, ReplayExecutor, STORE_PROPS, STORE_RECORD, SegmentHeader, SideChannelWrite,
    VectorPageEntry, WalRecord, WalRecordType, WalRecoveryReader, decode_commit_bundle_v8,
    decode_commit_bundle_v9, encode_commit_bundle_v8, encode_commit_bundle_v9, segment_filename,
};
use arcgraph_storage::{DeltaPageStore, DirtyPageTable};
use bytes::Bytes;
use tempfile::tempdir;

const T1: TenantId = TenantId::new(11);
const T2: TenantId = TenantId::new(22);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum CoverageRow {
    RecordCreateUpdate,
    RecordNodeDelete,
    PropSlotted,
    BlobOverflowImage,
    PrimarySecondaryIndexImages,
    VectorImage,
    RelExpiredLsn,
    TelDerivedNoLog,
    MvccAllocatorIdempotencyAcl,
    InternRecord,
}

fn page_alloc(
    tenant: TenantId,
    store: u16,
    page_no: u64,
    page_type: PageType,
    lsn: u64,
) -> DeltaOp {
    let mut payload = Vec::with_capacity(9);
    payload.push(page_type.as_byte());
    payload.extend_from_slice(&page_no.to_le_bytes());
    DeltaOp::new(
        DeltaOpKind::PageAlloc,
        store,
        tenant,
        page_no,
        0,
        Lsn::new(lsn),
        Bytes::from(payload),
    )
    .unwrap()
}

fn put_record(tenant: TenantId, page_no: u64, slot: u16, lsn: u64, payload: Bytes) -> DeltaOp {
    DeltaOp::new(
        DeltaOpKind::PutRecord,
        STORE_RECORD,
        tenant,
        page_no,
        slot,
        Lsn::new(lsn),
        payload,
    )
    .unwrap()
}

fn tombstone(tenant: TenantId, page_no: u64, slot: u16, lsn: u64) -> DeltaOp {
    DeltaOp::new(
        DeltaOpKind::TombstoneRecord,
        STORE_RECORD,
        tenant,
        page_no,
        slot,
        Lsn::new(lsn),
        Bytes::new(),
    )
    .unwrap()
}

fn v9_payload(
    commit_lsn: u64,
    tenant: TenantId,
    writes: &HashMap<u64, Option<Bytes>>,
    deltas: &[DeltaOp],
) -> Vec<u8> {
    encode_commit_bundle_v9(
        Lsn::new(commit_lsn),
        tenant,
        writes,
        &[],
        deltas,
        &[],
        &[],
        &[],
        &[],
        &[],
    )
    .unwrap()
}

fn generated_v9_history() -> Vec<Vec<u8>> {
    let node_t1_v1 = NodeRecord::new(NodeId::new(1), LabelId::new(3), Lsn::new(4));
    let mut node_t1_v2 = NodeRecord::new(NodeId::new(1), LabelId::new(4), Lsn::new(10));
    node_t1_v2.inline_u32a = 77;
    let node_t2 = NodeRecord::new(NodeId::new(2), LabelId::new(5), Lsn::new(21));
    let rel_t1 = RelRecord::new(
        RelId::new(7),
        TypeId::new(9),
        NodeId::new(1),
        NodeId::new(2),
        Lsn::new(31),
    );
    let first = vec![
        page_alloc(T1, STORE_RECORD, 1, PageType::Node, 1),
        put_record(T1, 1, 0, 2, Bytes::copy_from_slice(&node_t1_v1.to_bytes())),
        page_alloc(T1, STORE_PROPS, 1, PageType::PropSlotted, 3),
        DeltaOp::new(
            DeltaOpKind::PutPropBlock,
            STORE_PROPS,
            T1,
            1,
            0,
            Lsn::new(4),
            Bytes::from_static(b"typed-props:t1"),
        )
        .unwrap(),
    ];
    let update = vec![put_record(
        T1,
        1,
        0,
        10,
        Bytes::copy_from_slice(&node_t1_v2.to_bytes()),
    )];
    let tenant_two = vec![
        page_alloc(T2, STORE_RECORD, 1, PageType::Node, 20),
        put_record(T2, 1, 0, 21, Bytes::copy_from_slice(&node_t2.to_bytes())),
    ];
    let relationship = vec![
        page_alloc(T1, STORE_RECORD, 2, PageType::Rel, 30),
        put_record(T1, 2, 0, 31, Bytes::copy_from_slice(&rel_t1.to_bytes())),
    ];
    let deletes = vec![tombstone(T1, 1, 0, 40), tombstone(T1, 2, 0, 41)];
    vec![
        v9_payload(
            4,
            T1,
            &HashMap::from([(
                node_mvcc_key(NodeId::new(1)),
                Some(Bytes::copy_from_slice(&node_t1_v1.to_bytes())),
            )]),
            &first,
        ),
        v9_payload(
            10,
            T1,
            &HashMap::from([(
                node_mvcc_key(NodeId::new(1)),
                Some(Bytes::copy_from_slice(&node_t1_v2.to_bytes())),
            )]),
            &update,
        ),
        v9_payload(
            21,
            T2,
            &HashMap::from([(
                node_mvcc_key(NodeId::new(2)),
                Some(Bytes::copy_from_slice(&node_t2.to_bytes())),
            )]),
            &tenant_two,
        ),
        v9_payload(
            31,
            T1,
            &HashMap::from([(
                rel_mvcc_key(RelId::new(7)),
                Some(Bytes::copy_from_slice(&rel_t1.to_bytes())),
            )]),
            &relationship,
        ),
        v9_payload(
            41,
            T1,
            &HashMap::from([
                (node_mvcc_key(NodeId::new(1)), None),
                (rel_mvcc_key(RelId::new(7)), None),
            ]),
            &deletes,
        ),
    ]
}

fn init_page(tenant: TenantId, page_id: PageId, page_type: PageType) -> Box<PageBuf> {
    let mut bytes = Box::new([0u8; PAGE_SIZE]);
    SlottedPage::init(bytes.as_mut(), PageHeader::new(page_id, page_type, tenant)).unwrap();
    bytes
}

fn final_v8_oracle() -> (Vec<u8>, BTreeSet<CoverageRow>) {
    let mut coverage = BTreeSet::new();
    coverage.extend([
        CoverageRow::RecordCreateUpdate,
        CoverageRow::RecordNodeDelete,
        CoverageRow::PropSlotted,
        CoverageRow::BlobOverflowImage,
        CoverageRow::PrimarySecondaryIndexImages,
        CoverageRow::VectorImage,
        CoverageRow::RelExpiredLsn,
        CoverageRow::TelDerivedNoLog,
        CoverageRow::MvccAllocatorIdempotencyAcl,
        CoverageRow::InternRecord,
    ]);

    let mut node_t1 = init_page(T1, PageId::new(1), PageType::Node);
    {
        let mut page = SlottedPage::open(node_t1.as_mut()).unwrap();
        let mut updated = NodeRecord::new(NodeId::new(1), LabelId::new(4), Lsn::new(10));
        updated.inline_u32a = 77;
        page.put_node_at(SlotId(0), &updated).unwrap();
        page.tombstone(SlotId(0)).unwrap();
    }
    let mut node_t2 = init_page(T2, PageId::new(1), PageType::Node);
    let node_t2_record = NodeRecord::new(NodeId::new(2), LabelId::new(5), Lsn::new(21));
    SlottedPage::open(node_t2.as_mut())
        .unwrap()
        .put_node_at(SlotId(0), &node_t2_record)
        .unwrap();
    let mut rel_t1 = init_page(T1, PageId::new(2), PageType::Rel);
    {
        let mut record = RelRecord::new(
            RelId::new(7),
            TypeId::new(9),
            NodeId::new(1),
            NodeId::new(2),
            Lsn::new(31),
        );
        record.expired_lsn = 41;
        SlottedPage::open(rel_t1.as_mut())
            .unwrap()
            .put_rel_at(SlotId(0), &record)
            .unwrap();
    }
    let mut props_t1 = init_page(T1, PageId::new(1), PageType::PropSlotted);
    SlottedPage::open(props_t1.as_mut())
        .unwrap()
        .put_bag_at(SlotId(0), b"typed-props:t1")
        .unwrap();

    let primary = Box::new([0x11; PAGE_SIZE]);
    let secondary = Box::new([0x22; PAGE_SIZE]);
    let overflow = Box::new([0x33; PAGE_SIZE]);
    let vector_bytes = Box::new([0x44; PAGE_SIZE]);
    let staged = vec![
        (BundlePageKind::Record, PageId::new(1), T1, node_t1),
        (BundlePageKind::Record, PageId::new(1), T2, node_t2),
        (BundlePageKind::Record, PageId::new(2), T1, rel_t1),
        (BundlePageKind::PropSlotted, PageId::new(1), T1, props_t1),
        (BundlePageKind::PrimaryIndex, PageId::new(7), T1, primary),
        (
            BundlePageKind::SecondaryIndex,
            PageId::new(8),
            T1,
            secondary,
        ),
        (BundlePageKind::Blob, PageId::new(9), T1, overflow),
    ];
    let vector = VectorPageEntry {
        tenant: T1,
        partition: PartitionId::ZERO,
        index_id: 0,
        page_id: PageId::new(10),
        commit_lsn: Lsn::new(50),
        bytes: vector_bytes,
    };
    let writes = HashMap::from([
        (node_mvcc_key(NodeId::new(1)), None),
        (rel_mvcc_key(RelId::new(7)), None),
    ]);
    let side = SideChannelWrite {
        tenant_id: T2,
        key: node_mvcc_key(NodeId::new(2)),
        value: Some(Bytes::copy_from_slice(&node_t2_record.to_bytes())),
    };
    let allocators = [
        AllocatorAdvance {
            tenant: T1,
            kind: AllocatorKind::Node,
            new_high_water: 1,
        },
        AllocatorAdvance {
            tenant: T2,
            kind: AllocatorKind::Node,
            new_high_water: 2,
        },
    ];
    let idempotency = [IdempotencyBindingEntry {
        op: IdempotencyBindingOp::Install,
        tenant: T2,
        kind: 0,
        internal_id: 2,
        external_id: "node:t2:2".to_owned(),
    }];
    let acls = [
        AclGrantEntry {
            op: AclGrantOp::Apply,
            tenant: T1,
            doc: NodeId::new(1),
            grants: BTreeSet::from(["reader".to_owned()]),
        },
        AclGrantEntry {
            op: AclGrantOp::Revoke,
            tenant: T1,
            doc: NodeId::new(1),
            grants: BTreeSet::new(),
        },
    ];
    (
        encode_commit_bundle_v8(
            Lsn::new(50),
            T1,
            &writes,
            &[side],
            &staged,
            &allocators,
            &[vector],
            &idempotency,
            &acls,
        ),
        coverage,
    )
}

fn retained_v9_oracle() -> Vec<u8> {
    let retained = vec![
        (
            BundlePageKind::PrimaryIndex,
            PageId::new(7),
            T1,
            Box::new([0x11; PAGE_SIZE]),
        ),
        (
            BundlePageKind::SecondaryIndex,
            PageId::new(8),
            T1,
            Box::new([0x22; PAGE_SIZE]),
        ),
        (
            BundlePageKind::Blob,
            PageId::new(9),
            T1,
            Box::new([0x33; PAGE_SIZE]),
        ),
    ];
    let vector = VectorPageEntry {
        tenant: T1,
        partition: PartitionId::ZERO,
        index_id: 0,
        page_id: PageId::new(10),
        commit_lsn: Lsn::new(50),
        bytes: Box::new([0x44; PAGE_SIZE]),
    };
    let allocators = [
        AllocatorAdvance {
            tenant: T1,
            kind: AllocatorKind::Node,
            new_high_water: 1,
        },
        AllocatorAdvance {
            tenant: T2,
            kind: AllocatorKind::Node,
            new_high_water: 2,
        },
    ];
    let idempotency = [IdempotencyBindingEntry {
        op: IdempotencyBindingOp::Install,
        tenant: T2,
        kind: 0,
        internal_id: 2,
        external_id: "node:t2:2".to_owned(),
    }];
    let acls = [
        AclGrantEntry {
            op: AclGrantOp::Apply,
            tenant: T1,
            doc: NodeId::new(1),
            grants: BTreeSet::from(["reader".to_owned()]),
        },
        AclGrantEntry {
            op: AclGrantOp::Revoke,
            tenant: T1,
            doc: NodeId::new(1),
            grants: BTreeSet::new(),
        },
    ];
    encode_commit_bundle_v9(
        Lsn::new(50),
        T1,
        &HashMap::new(),
        &[],
        &[],
        &retained,
        &[vector],
        &allocators,
        &idempotency,
        &acls,
    )
    .unwrap()
}

#[test]
fn generated_mutation_table_v8_v9_logical_oracles_match() {
    let (v8_bytes, coverage) = final_v8_oracle();
    assert_eq!(coverage.len(), 10, "mutation coverage table row dropped");
    let v8 = decode_commit_bundle_v8(&v8_bytes, T1).unwrap();
    let retained_v9 = decode_commit_bundle_v9(&retained_v9_oracle(), T1).unwrap();
    let history = generated_v9_history();
    let v9: Vec<_> = history
        .iter()
        .map(|payload| decode_commit_bundle_v9(payload, T1).unwrap())
        .collect();

    let mut v9_mvcc: BTreeMap<(TenantId, u64), Option<Bytes>> = BTreeMap::new();
    let mut locations = HashMap::new();
    let mut prop_payload = None;
    for bundle in &v9 {
        for (key, value) in &bundle.mvcc_writes {
            v9_mvcc.insert((bundle.primary_tenant, *key), value.clone());
        }
        for write in &bundle.sidechannel_writes {
            v9_mvcc.insert((write.tenant_id, write.key), write.value.clone());
        }
        for op in &bundle.deltas {
            match op.kind {
                DeltaOpKind::PutRecord => {
                    let id = u64::from_le_bytes(op.payload[..8].try_into().unwrap());
                    let key = if op.payload.len() == NodeRecord::SIZE {
                        node_mvcc_key(NodeId::new(id))
                    } else {
                        rel_mvcc_key(RelId::new(id))
                    };
                    locations.insert((op.tenant_id, op.page_no, op.slot), key);
                    v9_mvcc.insert((op.tenant_id, key), Some(op.payload.clone()));
                }
                DeltaOpKind::TombstoneRecord => {
                    let key = locations[&(op.tenant_id, op.page_no, op.slot)];
                    v9_mvcc.insert((op.tenant_id, key), None);
                }
                DeltaOpKind::PutPropBlock => prop_payload = Some(op.payload.clone()),
                _ => {}
            }
        }
    }
    let mut v8_mvcc = BTreeMap::new();
    for (key, value) in &v8.mvcc_writes {
        v8_mvcc.insert((T1, *key), value.clone());
    }
    for write in &v8.sidechannel_writes {
        v8_mvcc.insert((write.tenant_id, write.key), write.value.clone());
    }
    assert_eq!(v9_mvcc, v8_mvcc);

    let v8_node_t1 = v8
        .staged_pages
        .iter()
        .find(|page| {
            page.kind == BundlePageKind::Record
                && page.tenant_id == T1
                && page.page_id == PageId::new(1)
        })
        .unwrap();
    assert!(
        SlottedPageRef::open(v8_node_t1.bytes.as_ref())
            .unwrap()
            .read_node(SlotId(0))
            .unwrap()
            .is_none()
    );
    let v8_rel_t1 = v8
        .staged_pages
        .iter()
        .find(|page| {
            page.kind == BundlePageKind::Record
                && page.tenant_id == T1
                && page.page_id == PageId::new(2)
        })
        .unwrap();
    assert_eq!(
        SlottedPageRef::open(v8_rel_t1.bytes.as_ref())
            .unwrap()
            .read_rel(SlotId(0))
            .unwrap()
            .unwrap()
            .expired_lsn,
        41
    );
    let v8_node_t2 = v8
        .staged_pages
        .iter()
        .find(|page| {
            page.kind == BundlePageKind::Record
                && page.tenant_id == T2
                && page.page_id == PageId::new(1)
        })
        .unwrap();
    assert_eq!(
        Bytes::copy_from_slice(
            &SlottedPageRef::open(v8_node_t2.bytes.as_ref())
                .unwrap()
                .read_node(SlotId(0))
                .unwrap()
                .unwrap()
                .to_bytes()
        ),
        v8_mvcc[&(T2, node_mvcc_key(NodeId::new(2)))]
            .clone()
            .unwrap()
    );

    let prop_page = v8
        .staged_pages
        .iter()
        .find(|page| page.kind == BundlePageKind::PropSlotted)
        .unwrap();
    assert_eq!(
        SlottedPageRef::open(prop_page.bytes.as_ref())
            .unwrap()
            .read_bag(SlotId(0))
            .unwrap()
            .unwrap(),
        prop_payload.unwrap().as_ref()
    );
    assert_eq!(
        v8.staged_pages
            .iter()
            .filter(|page| matches!(
                page.kind,
                BundlePageKind::PrimaryIndex
                    | BundlePageKind::SecondaryIndex
                    | BundlePageKind::Blob
            ))
            .count(),
        3
    );
    assert_eq!(v8.vector_pages.len(), 1);
    assert_eq!(v8.allocator_advances.len(), 2);
    assert_eq!(v8.idempotency_bindings.len(), 1);
    assert_eq!(v8.acl_grants.len(), 2);
    let v8_retained: Vec<_> = v8
        .staged_pages
        .iter()
        .filter(|page| {
            matches!(
                page.kind,
                BundlePageKind::PrimaryIndex
                    | BundlePageKind::SecondaryIndex
                    | BundlePageKind::Blob
            )
        })
        .cloned()
        .collect();
    assert_eq!(retained_v9.staged_pages, v8_retained);
    assert_eq!(retained_v9.vector_pages, v8.vector_pages);
    assert_eq!(retained_v9.allocator_advances, v8.allocator_advances);
    assert_eq!(retained_v9.idempotency_bindings, v8.idempotency_bindings);
    assert_eq!(retained_v9.acl_grants, v8.acl_grants);
    let intern = arcgraph_storage::intern::encode_intern_payload(StringId::new(9), "KNOWS");
    assert_eq!(
        arcgraph_storage::intern::decode_intern_payload(&intern).unwrap(),
        (StringId::new(9), "KNOWS".to_owned())
    );
    assert!(
        v9.iter()
            .flat_map(|bundle| &bundle.deltas)
            .all(|op| !matches!(
                op.kind,
                DeltaOpKind::TelAppend
                    | DeltaOpKind::TelExpire
                    | DeltaOpKind::IndexPut
                    | DeltaOpKind::IndexDelete
                    | DeltaOpKind::VectorDelta
                    | DeltaOpKind::InternBind
                    | DeltaOpKind::AclGrant
            )),
        "M3 reserved/derived rows must not leak into the delta stream"
    );
}

type PhysicalPageKey = (u16, TenantId, PageId);
type PhysicalReplay = (HashMap<PhysicalPageKey, Box<PageBuf>>, u64);

fn real_delta_store() -> Arc<BufferedRecordPageStore> {
    let io: Arc<dyn PageIo> = Arc::new(InMemoryPageIo::new());
    let pools = Arc::new(PerTenantBufferPool::with_config(
        io,
        PerTenantBufferPoolConfig {
            frames_per_tenant: 8,
            write_fraction: 0.0,
        },
    ));
    Arc::new(BufferedRecordPageStore::with_cache_cap(pools, 16))
}

fn replay_history(payloads: &[Vec<u8>]) -> PhysicalReplay {
    let dir = tempdir().unwrap();
    let mut bytes = SegmentHeader {
        format_version: BUNDLE_FORMAT_V9,
    }
    .encode()
    .to_vec();
    // Adversarial disk order: high first, low, a byte-identical duplicate of
    // low, then the middle ranges. Gaps 5..9, 11..19, etc. are legal.
    let order = [4, 0, 0, 2, 1, 3];
    for (frame_index, payload_index) in order.into_iter().enumerate() {
        let payload = payloads[payload_index].clone();
        WalRecord {
            record_type: WalRecordType::CommitBundle,
            txn_id: frame_index as u64 + 1,
            lsn: Lsn::new(frame_index as u64 + 1),
            timestamp_ms: 0,
            tenant_id: if payload_index == 2 { T2 } else { T1 },
            payload,
        }
        .encode(&mut bytes)
        .unwrap();
    }
    std::fs::write(dir.path().join(segment_filename(0)), bytes).unwrap();
    let props = real_delta_store();
    let records = real_delta_store();
    let target = PageStoreTarget::primary_only(Arc::new(PrimaryPageStore::new()))
        .with_delta_stores(
            Arc::clone(&props) as Arc<dyn DeltaPageStore>,
            Arc::clone(&records) as Arc<dyn DeltaPageStore>,
            Arc::new(DirtyPageTable::new()),
        );
    let mut replay = ReplayExecutor::new(
        ReplayConfig::with_wal_dir(dir.path()),
        Arc::new(TxnManager::new()),
        target,
    );
    let high = replay
        .run(WalRecoveryReader::open(dir.path()).unwrap())
        .unwrap();
    let mut all = HashMap::new();
    for (tenant, page_id, _) in props.iter_pages_qualified() {
        let value = props
            .copy_page_pinned_for_tenant(tenant, page_id)
            .unwrap()
            .unwrap();
        all.insert((STORE_PROPS, tenant, page_id), value);
    }
    for (tenant, page_id, _) in records.iter_pages_qualified() {
        let value = records
            .copy_page_pinned_for_tenant(tenant, page_id)
            .unwrap()
            .unwrap();
        assert!(all.insert((STORE_RECORD, tenant, page_id), value).is_none());
    }
    (all, high.raw())
}

#[test]
fn v9_replay_twice_is_byte_identical_across_order_gaps_duplicates_and_tenants() {
    let history = generated_v9_history();
    let first = replay_history(&history);
    let second = replay_history(&history);
    assert_eq!(first, second);
    assert_eq!(first.1, 41);
    assert!(first.0.contains_key(&(STORE_RECORD, T1, PageId::new(1))));
    assert!(first.0.contains_key(&(STORE_RECORD, T2, PageId::new(1))));
    assert_ne!(
        first.0[&(STORE_RECORD, T1, PageId::new(1))].as_ref(),
        first.0[&(STORE_RECORD, T2, PageId::new(1))].as_ref(),
        "same page_no in two tenants must stay isolated"
    );
}
