//! M3 DeltaOp/v9 CommitBundle wire and scope-fence gates.

use std::collections::{BTreeSet, HashMap};

use arcgraph_core::record::{NodeRecord, PAGE_SIZE, PageType};
use arcgraph_core::{ArcGraphError, LabelId, Lsn, NodeId, PageId, PartitionId, TenantId};
use arcgraph_storage::owner_row::{OwnerRow, OwnerRowClass};
use arcgraph_storage::wal::{
    AclGrantEntry, AclGrantOp, AllocatorAdvance, AllocatorKind, BUNDLE_FORMAT_V9,
    BUNDLE_FORMAT_V10, BundlePageKind, DeltaOp, DeltaOpKind, IdempotencyBindingEntry,
    IdempotencyBindingOp, STORE_BLOB_OVERFLOW, STORE_PROPS, STORE_RECORD, SideChannelWrite,
    VectorPageEntry, decode_commit_bundle_for_version, decode_commit_bundle_v8,
    decode_commit_bundle_v9, encode_commit_bundle_v8, encode_commit_bundle_v9,
    encode_commit_bundle_v10,
};
use bytes::Bytes;

const STILL_RESERVED_DELTA_KINDS: [DeltaOpKind; 6] = [
    DeltaOpKind::TelAppend,
    DeltaOpKind::TelExpire,
    DeltaOpKind::IndexPut,
    DeltaOpKind::IndexDelete,
    DeltaOpKind::VectorDelta,
    DeltaOpKind::TelGrow,
];

fn assert_reserved_for_declared_format(
    error: ArcGraphError,
    kind: DeltaOpKind,
    format_version: u16,
    milestone: &str,
) {
    assert!(matches!(error, ArcGraphError::WalCorruption { .. }));
    let expected = format!(
        "DeltaOp kind {kind:?} is reserved in declared WAL bundle format v{format_version} ({milestone})"
    );
    assert!(
        error.to_string().contains(&expected),
        "expected version-scoped rejection {expected:?}, got {error}"
    );
}

fn owner_delta(class: OwnerRowClass, id: u64, op_lsn: Lsn) -> DeltaOp {
    let row = OwnerRow::new(class, id, format!("wire-owner-{class:?}-{id}").into_bytes()).unwrap();
    let address = class.address(id).unwrap();
    DeltaOp::new_for_format(
        BUNDLE_FORMAT_V10,
        class.delta_kind(),
        class.store_id(),
        TenantId::DEFAULT,
        address.page_no,
        address.slot.raw(),
        op_lsn,
        Bytes::copy_from_slice(&row.encode()),
    )
    .unwrap()
}

fn page_alloc_payload(page_type: PageType, generation: u64) -> Bytes {
    let mut payload = Vec::with_capacity(9);
    payload.push(page_type.as_byte());
    payload.extend_from_slice(&generation.to_le_bytes());
    Bytes::from(payload)
}

fn full_delta_set() -> Vec<DeltaOp> {
    let commit_lsn = Lsn::new(207);
    let node = NodeRecord::new(NodeId::new(11), LabelId::new(3), commit_lsn);
    vec![
        DeltaOp::new(
            DeltaOpKind::PageAlloc,
            STORE_RECORD,
            TenantId::DEFAULT,
            7,
            0,
            Lsn::new(200),
            page_alloc_payload(PageType::Node, 1),
        )
        .unwrap(),
        DeltaOp::new(
            DeltaOpKind::PutRecord,
            STORE_RECORD,
            TenantId::DEFAULT,
            7,
            0,
            Lsn::new(201),
            Bytes::copy_from_slice(&node.to_bytes()),
        )
        .unwrap(),
        DeltaOp::new(
            DeltaOpKind::TombstoneRecord,
            STORE_RECORD,
            TenantId::DEFAULT,
            7,
            0,
            Lsn::new(202),
            Bytes::new(),
        )
        .unwrap(),
        DeltaOp::new(
            DeltaOpKind::PageAlloc,
            STORE_PROPS,
            TenantId::new(2),
            8,
            0,
            Lsn::new(203),
            page_alloc_payload(PageType::PropSlotted, 4),
        )
        .unwrap(),
        DeltaOp::new(
            DeltaOpKind::PutPropBlock,
            STORE_PROPS,
            TenantId::new(2),
            8,
            0,
            Lsn::new(204),
            Bytes::from_static(b"typed-block"),
        )
        .unwrap(),
        DeltaOp::new(
            DeltaOpKind::AllocAdvance,
            STORE_RECORD,
            TenantId::DEFAULT,
            0,
            0,
            Lsn::new(205),
            Bytes::from_static(&[1, 20, 0, 0, 0, 0, 0, 0, 0]),
        )
        .unwrap(),
    ]
}

#[test]
fn delta_kind_bytes_are_wire_stable_and_legality_is_version_scoped() {
    let kinds = [
        DeltaOpKind::PutRecord,
        DeltaOpKind::TombstoneRecord,
        DeltaOpKind::PutPropBlock,
        DeltaOpKind::TelAppend,
        DeltaOpKind::TelExpire,
        DeltaOpKind::IndexPut,
        DeltaOpKind::IndexDelete,
        DeltaOpKind::AllocAdvance,
        DeltaOpKind::InternBind,
        DeltaOpKind::AclGrant,
        DeltaOpKind::PageAlloc,
        DeltaOpKind::ExtentAlloc,
        DeltaOpKind::VectorDelta,
        DeltaOpKind::TelGrow,
    ];
    for (byte, kind) in kinds.into_iter().enumerate() {
        assert_eq!(kind.as_byte(), byte as u8);
        assert_eq!(
            DeltaOpKind::from_byte(byte as u8, Lsn::new(1)).unwrap(),
            kind
        );
    }
    assert!(DeltaOpKind::from_byte(14, Lsn::new(1)).is_err());

    for owner in [
        owner_delta(OwnerRowClass::InternedString, 1, Lsn::new(1)),
        owner_delta(OwnerRowClass::Grant, 1, Lsn::new(1)),
    ] {
        owner.validate_for_format(BUNDLE_FORMAT_V10).unwrap();
        let error = owner.validate_for_format(BUNDLE_FORMAT_V9).unwrap_err();
        assert_reserved_for_declared_format(error, owner.kind, BUNDLE_FORMAT_V9, "M3");
    }

    for reserved in STILL_RESERVED_DELTA_KINDS {
        for (format_version, milestone) in [(BUNDLE_FORMAT_V9, "M3"), (BUNDLE_FORMAT_V10, "M4")] {
            let error = DeltaOp::new_for_format(
                format_version,
                reserved,
                STORE_RECORD,
                TenantId::DEFAULT,
                0,
                0,
                Lsn::new(1),
                Bytes::from_static(b"still-reserved"),
            )
            .unwrap_err();
            assert_reserved_for_declared_format(error, reserved, format_version, milestone);
        }
    }
}

#[test]
fn owner_kinds_roundtrip_at_m4_but_same_wire_is_rejected_when_declared_m3() {
    let deltas = vec![
        owner_delta(OwnerRowClass::InternedString, 7, Lsn::new(1)),
        owner_delta(OwnerRowClass::Grant, 11, Lsn::new(2)),
    ];
    let wire = encode_commit_bundle_v10(
        Lsn::new(2),
        TenantId::DEFAULT,
        &HashMap::new(),
        &[],
        &deltas,
        &[],
        &[],
        &[],
        &[],
        &[],
    )
    .unwrap();
    let decoded =
        decode_commit_bundle_for_version(&wire, BUNDLE_FORMAT_V10, TenantId::DEFAULT).unwrap();
    assert_eq!(decoded.deltas, deltas);

    let error =
        decode_commit_bundle_for_version(&wire, BUNDLE_FORMAT_V9, TenantId::DEFAULT).unwrap_err();
    assert_reserved_for_declared_format(error, DeltaOpKind::InternBind, BUNDLE_FORMAT_V9, "M3");
}

fn hostile_wire_with_first_kind(format_version: u16, kind: DeltaOpKind) -> ArcGraphError {
    let deltas = full_delta_set();
    let mut wire = match format_version {
        BUNDLE_FORMAT_V9 => encode_commit_bundle_v9(
            deltas.last().unwrap().op_lsn,
            TenantId::DEFAULT,
            &HashMap::new(),
            &[],
            &deltas,
            &[],
            &[],
            &[],
            &[],
            &[],
        ),
        BUNDLE_FORMAT_V10 => encode_commit_bundle_v10(
            deltas.last().unwrap().op_lsn,
            TenantId::DEFAULT,
            &HashMap::new(),
            &[],
            &deltas,
            &[],
            &[],
            &[],
            &[],
            &[],
        ),
        other => panic!("unsupported hostile-wire fixture format {other}"),
    }
    .unwrap();
    // commit(8) + n_mvcc(4) + n_deltas(4): overwrite the first valid
    // DeltaOp kind while leaving its otherwise-valid fixed prefix intact.
    wire[16] = kind.as_byte();
    decode_commit_bundle_for_version(&wire, format_version, TenantId::DEFAULT).unwrap_err()
}

#[test]
fn declared_decoders_reject_every_kind_reserved_in_their_version() {
    for reserved in [DeltaOpKind::InternBind, DeltaOpKind::AclGrant]
        .into_iter()
        .chain(STILL_RESERVED_DELTA_KINDS)
    {
        let error = hostile_wire_with_first_kind(BUNDLE_FORMAT_V9, reserved);
        assert_reserved_for_declared_format(error, reserved, BUNDLE_FORMAT_V9, "M3");
    }
    for reserved in STILL_RESERVED_DELTA_KINDS {
        let error = hostile_wire_with_first_kind(BUNDLE_FORMAT_V10, reserved);
        assert_reserved_for_declared_format(error, reserved, BUNDLE_FORMAT_V10, "M4");
    }
}

#[test]
fn legacy_constructor_remains_m3_scoped() {
    for reserved in [DeltaOpKind::InternBind, DeltaOpKind::AclGrant] {
        let error = DeltaOp::new(
            reserved,
            STORE_RECORD,
            TenantId::DEFAULT,
            0,
            0,
            Lsn::new(1),
            Bytes::from_static(b"layout-must-not-be-invented"),
        )
        .unwrap_err();
        assert_reserved_for_declared_format(error, reserved, BUNDLE_FORMAT_V9, "M3");
    }
}

#[test]
fn blob_overflow_store_byte_is_pinned_and_put_prop_block_is_reserved_at_m3() {
    assert_eq!(STORE_BLOB_OVERFLOW, 5);

    let valid = DeltaOp::new(
        DeltaOpKind::PutPropBlock,
        STORE_PROPS,
        TenantId::DEFAULT,
        1,
        0,
        Lsn::new(1),
        Bytes::from_static(b"typed-block"),
    )
    .unwrap();
    let mut wire = Vec::new();
    valid.encode_into(&mut wire).unwrap();
    wire[1..3].copy_from_slice(&STORE_BLOB_OVERFLOW.to_le_bytes());

    let error = DeltaOp::decode_prefix(&wire).unwrap_err();
    assert!(matches!(error, ArcGraphError::WalCorruption { .. }));
    assert!(error.to_string().contains("blob.overflow"));
    assert!(
        DeltaOp::new(
            DeltaOpKind::PageAlloc,
            STORE_BLOB_OVERFLOW,
            TenantId::DEFAULT,
            1,
            0,
            Lsn::new(1),
            page_alloc_payload(PageType::PropSlotted, 1),
        )
        .is_err(),
        "store 5 is page-image-only at M3, including allocation lifecycle"
    );
}

#[test]
fn delta_fixed_prefix_offsets_include_tenant_and_full_sub_lsn() {
    let op = full_delta_set().remove(0);
    let mut wire = Vec::new();
    op.encode_into(&mut wire).unwrap();
    assert_eq!(DeltaOp::FIXED_PREFIX_LEN, 34);
    assert_eq!(wire[0], DeltaOpKind::PageAlloc.as_byte());
    assert_eq!(&wire[1..3], &STORE_RECORD.to_le_bytes());
    assert_eq!(wire[3], 0);
    assert_eq!(&wire[4..12], &TenantId::DEFAULT.raw().to_le_bytes());
    assert_eq!(&wire[12..20], &7u64.to_le_bytes());
    assert_eq!(&wire[20..22], &0u16.to_le_bytes());
    assert_eq!(&wire[22..30], &200u64.to_le_bytes());
    assert_eq!(&wire[30..34], &9u32.to_le_bytes());
    let (decoded, consumed) = DeltaOp::decode_prefix(&wire).unwrap();
    assert_eq!(consumed, wire.len());
    assert_eq!(decoded, op);
}

#[test]
fn v9_roundtrip_carries_deltas_multi_tenant_mvcc_and_retained_images() {
    let deltas = full_delta_set();
    let commit_lsn = deltas.last().unwrap().op_lsn;
    let mut primary = HashMap::new();
    primary.insert(7, Some(Bytes::from_static(b"primary")));
    let sidechannel = vec![SideChannelWrite {
        tenant_id: TenantId::SYSTEM,
        key: 9,
        value: Some(Bytes::from_static(b"system")),
    }];
    let retained = vec![
        (
            BundlePageKind::SecondaryIndex,
            PageId::new(44),
            TenantId::DEFAULT,
            Box::new([0x5Au8; PAGE_SIZE]),
        ),
        (
            BundlePageKind::Blob,
            PageId::new(45),
            TenantId::DEFAULT,
            Box::new([0x5Bu8; PAGE_SIZE]),
        ),
    ];
    let vectors = vec![VectorPageEntry {
        tenant: TenantId::DEFAULT,
        partition: PartitionId::ZERO,
        index_id: 3,
        page_id: PageId::new(5),
        commit_lsn,
        bytes: Box::new([0x6Bu8; PAGE_SIZE]),
    }];
    let allocator_advances = vec![
        AllocatorAdvance {
            tenant: TenantId::SYSTEM,
            kind: AllocatorKind::Rel,
            new_high_water: 81,
        },
        AllocatorAdvance {
            tenant: TenantId::DEFAULT,
            kind: AllocatorKind::Node,
            new_high_water: 41,
        },
    ];
    let idempotency_bindings = vec![
        IdempotencyBindingEntry {
            op: IdempotencyBindingOp::Release,
            tenant: TenantId::DEFAULT,
            kind: 0,
            internal_id: 0,
            external_id: "node:released".to_owned(),
        },
        IdempotencyBindingEntry {
            op: IdempotencyBindingOp::Install,
            tenant: TenantId::SYSTEM,
            kind: 1,
            internal_id: 91,
            external_id: "rel:installed".to_owned(),
        },
    ];
    let acl_grants = vec![
        AclGrantEntry {
            op: AclGrantOp::Apply,
            tenant: TenantId::DEFAULT,
            doc: NodeId::new(12),
            grants: BTreeSet::from(["principal:a".to_owned(), "principal:b".to_owned()]),
        },
        AclGrantEntry {
            op: AclGrantOp::Revoke,
            tenant: TenantId::DEFAULT,
            doc: NodeId::new(12),
            grants: BTreeSet::new(),
        },
    ];

    let wire = encode_commit_bundle_v9(
        commit_lsn,
        TenantId::DEFAULT,
        &primary,
        &sidechannel,
        &deltas,
        &retained,
        &vectors,
        &allocator_advances,
        &idempotency_bindings,
        &acl_grants,
    )
    .unwrap();
    let decoded =
        decode_commit_bundle_for_version(&wire, BUNDLE_FORMAT_V9, TenantId::DEFAULT).unwrap();
    let v8_wire = encode_commit_bundle_v8(
        commit_lsn,
        TenantId::DEFAULT,
        &primary,
        &sidechannel,
        &retained,
        &allocator_advances,
        &vectors,
        &idempotency_bindings,
        &acl_grants,
    );
    let v8 = decode_commit_bundle_v8(&v8_wire, TenantId::DEFAULT).unwrap();
    assert_eq!(decoded.commit_lsn, commit_lsn);
    assert_eq!(decoded.redo_range().base(), Lsn::new(200));
    assert_eq!(decoded.deltas, deltas);
    assert_eq!(decoded.mvcc_writes, primary);
    assert_eq!(decoded.sidechannel_writes, sidechannel);
    assert_eq!(decoded.staged_pages.len(), 2);
    assert_eq!(decoded.staged_pages[0].kind, BundlePageKind::SecondaryIndex);
    assert_eq!(decoded.staged_pages[1].kind, BundlePageKind::Blob);
    assert_eq!(decoded.vector_pages, vectors);
    let mut expected_advances = allocator_advances;
    expected_advances.sort_by_key(|entry| (entry.tenant.raw(), entry.kind.as_byte()));
    assert_eq!(decoded.allocator_advances, expected_advances);
    assert_eq!(decoded.allocator_advances, v8.allocator_advances);
    let mut expected_bindings = idempotency_bindings;
    expected_bindings.sort_by(|a, b| {
        a.tenant
            .raw()
            .cmp(&b.tenant.raw())
            .then(a.kind.cmp(&b.kind))
            .then_with(|| a.external_id.cmp(&b.external_id))
            .then((a.op as u8).cmp(&(b.op as u8)))
    });
    assert_eq!(decoded.idempotency_bindings, expected_bindings);
    assert_eq!(decoded.idempotency_bindings, v8.idempotency_bindings);
    assert_eq!(
        decoded.acl_grants, acl_grants,
        "ACL operations must preserve v8 append order for last-writer-wins"
    );
    assert_eq!(decoded.acl_grants, v8.acl_grants);
}

#[test]
fn v9_rejects_page_alloc_after_first_use() {
    let node = NodeRecord::new(NodeId::new(1), LabelId::new(1), Lsn::new(11));
    let deltas = vec![
        DeltaOp::new(
            DeltaOpKind::PutRecord,
            STORE_RECORD,
            TenantId::DEFAULT,
            1,
            0,
            Lsn::new(10),
            Bytes::copy_from_slice(&node.to_bytes()),
        )
        .unwrap(),
        DeltaOp::new(
            DeltaOpKind::PageAlloc,
            STORE_RECORD,
            TenantId::DEFAULT,
            1,
            0,
            Lsn::new(11),
            page_alloc_payload(PageType::Node, 1),
        )
        .unwrap(),
    ];
    assert!(
        encode_commit_bundle_v9(
            Lsn::new(11),
            TenantId::DEFAULT,
            &HashMap::new(),
            &[],
            &deltas,
            &[],
            &[],
            &[],
            &[],
            &[],
        )
        .is_err()
    );
}

#[test]
fn v9_rejects_record_page_images_and_bad_sub_lsn_sequence() {
    let deltas = full_delta_set();
    let record_image = vec![(
        BundlePageKind::Record,
        PageId::new(1),
        TenantId::DEFAULT,
        Box::new([0u8; PAGE_SIZE]),
    )];
    assert!(
        encode_commit_bundle_v9(
            deltas.last().unwrap().op_lsn,
            TenantId::DEFAULT,
            &HashMap::new(),
            &[],
            &deltas,
            &record_image,
            &[],
            &[],
            &[],
            &[],
        )
        .is_err()
    );

    let mut wire = encode_commit_bundle_v9(
        deltas.last().unwrap().op_lsn,
        TenantId::DEFAULT,
        &HashMap::new(),
        &[],
        &deltas,
        &[],
        &[],
        &[],
        &[],
        &[],
    )
    .unwrap();
    // commit(8) + n_mvcc(4) + n_deltas(4) + op_lsn offset(22).
    wire[38..46].copy_from_slice(&999u64.to_le_bytes());
    assert!(decode_commit_bundle_v9(&wire, TenantId::DEFAULT).is_err());
}

#[test]
fn every_trailing_partial_v9_bundle_is_rejected() {
    let delta = DeltaOp::new(
        DeltaOpKind::PageAlloc,
        STORE_RECORD,
        TenantId::DEFAULT,
        3,
        0,
        Lsn::new(1),
        page_alloc_payload(PageType::Node, 1),
    )
    .unwrap();
    let wire = encode_commit_bundle_v9(
        Lsn::new(1),
        TenantId::DEFAULT,
        &HashMap::new(),
        &[],
        &[delta],
        &[],
        &[],
        &[],
        &[],
        &[],
    )
    .unwrap();
    for cut in 0..wire.len() {
        assert!(
            decode_commit_bundle_v9(&wire[..cut], TenantId::DEFAULT).is_err(),
            "truncation at byte {cut} unexpectedly decoded"
        );
    }
    assert!(decode_commit_bundle_v9(&wire, TenantId::DEFAULT).is_ok());
}

#[test]
fn put_record_is_single_copy_and_divergent_mvcc_bytes_reject() {
    let commit_lsn = Lsn::new(9);
    let record = NodeRecord::new(NodeId::new(77), LabelId::new(4), commit_lsn);
    let record_bytes = Bytes::copy_from_slice(&record.to_bytes());
    let put = DeltaOp::new(
        DeltaOpKind::PutRecord,
        STORE_RECORD,
        TenantId::DEFAULT,
        1,
        0,
        commit_lsn,
        record_bytes.clone(),
    )
    .unwrap();
    let writes = HashMap::from([(
        arcgraph_storage::crud::node_mvcc_key(NodeId::new(77)),
        Some(record_bytes),
    )]);
    let wire = encode_commit_bundle_v9(
        commit_lsn,
        TenantId::DEFAULT,
        &writes,
        &[],
        std::slice::from_ref(&put),
        &[],
        &[],
        &[],
        &[],
        &[],
    )
    .unwrap();
    let decoded = decode_commit_bundle_v9(&wire, TenantId::DEFAULT).unwrap();
    assert!(decoded.mvcc_writes.is_empty());
    assert_eq!(decoded.deltas[0].payload, put.payload);

    let mut divergent = record.to_bytes();
    divergent[8] ^= 1;
    let divergent_writes = HashMap::from([(
        arcgraph_storage::crud::node_mvcc_key(NodeId::new(77)),
        Some(Bytes::copy_from_slice(&divergent)),
    )]);
    let error = encode_commit_bundle_v9(
        commit_lsn,
        TenantId::DEFAULT,
        &divergent_writes,
        &[],
        &[put],
        &[],
        &[],
        &[],
        &[],
        &[],
    )
    .unwrap_err();
    assert!(error.to_string().contains("diverges from its MVCC version"));
}
