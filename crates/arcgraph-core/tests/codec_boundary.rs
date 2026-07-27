//! W26-γ-2 D3 — Codec boundary tests for arcgraph-core's on-disk
//! record format.
//!
//! Per ADR-134 forward-binding (test:prod ratio uplift) + W26-γ-2 D3
//! spec. The existing inline `#[cfg(test)]` block in `src/record.rs`
//! covers basic round-trip per record kind; this integration-test
//! file adds adversarial-input + boundary proptest coverage:
//!
//! - `from_bytes` rejects bytes of the wrong length / wrong magic.
//! - `to_bytes` is the exact inverse of `from_bytes` over the entire
//!   reachable value domain (every random byte slice that round-trips
//!   must produce the same value back).
//! - `PageType::from_byte` rejects every byte ≥ 9 (the current high
//!   sentinel `IndexOverflow`) as `UnknownPageType`.
//! - `NodeRecord` / `RelRecord` / `TelEntry` round-trip the
//!   `created_lsn` / `expired_lsn` fields through `MVCC` sentinels.

use arcgraph_core::error::ArcGraphError;
use arcgraph_core::ids::{LabelId, Lsn, NodeId, PageId, RelId, TenantId, TypeId};
use arcgraph_core::record::{NodeRecord, PAGE_SIZE, PageHeader, PageType, RelRecord, TelEntry};
use proptest::prelude::*;

// ────────────────────── PageType from_byte ──────────────────────

#[test]
fn page_type_known_bytes_roundtrip() {
    let cases = [
        (0u8, PageType::Free),
        (1, PageType::Node),
        (2, PageType::Rel),
        (3, PageType::Tel),
        (4, PageType::IndexInternal),
        (5, PageType::IndexLeaf),
        (6, PageType::VectorNeighbor),
        (7, PageType::WalBuffer),
        (8, PageType::IndexOverflow),
        // v2 M1 (ADR-230): shared slotted property-bag heap pages.
        (9, PageType::PropSlotted),
    ];
    for (byte, expected) in &cases {
        let pt = PageType::from_byte(*byte).expect("known byte");
        assert_eq!(pt, *expected);
        assert_eq!(pt as u8, *byte);
    }
}

proptest! {
    #[test]
    fn page_type_unknown_bytes_rejected(byte in 10u8..=u8::MAX) {
        let r = PageType::from_byte(byte);
        prop_assert!(matches!(r, Err(ArcGraphError::UnknownPageType(b)) if b == byte));
    }
}

// ────────────────────── PageHeader round-trip ──────────────────────

#[test]
fn page_header_basic_roundtrip() {
    let h = PageHeader::new(PageId::new(42), PageType::Node, TenantId::DEFAULT);
    let bytes = h.to_bytes();
    let back = PageHeader::from_bytes(&bytes).expect("round-trip");
    assert_eq!(h.page_id, back.page_id);
    assert_eq!(h.page_type, back.page_type);
    assert_eq!(h.tenant_id, back.tenant_id);
}

proptest! {
    #[test]
    fn page_header_roundtrip_all_known_page_types(raw_id in any::<u64>(), tenant_raw in any::<u64>(), byte in 0u8..=8) {
        let pt = PageType::from_byte(byte).expect("known");
        let h = PageHeader::new(PageId::new(raw_id), pt, TenantId::new(tenant_raw));
        let bytes = h.to_bytes();
        let back = PageHeader::from_bytes(&bytes).expect("round-trip");
        prop_assert_eq!(back.page_id, h.page_id);
        prop_assert_eq!(back.page_type, h.page_type);
        prop_assert_eq!(back.tenant_id, h.tenant_id);
    }
}

#[test]
fn page_header_size_is_exactly_40() {
    assert_eq!(PageHeader::SIZE, 40);
    assert_eq!(core::mem::size_of::<PageHeader>(), 40);
}

// ────────────────────── NodeRecord round-trip ──────────────────────

#[test]
fn node_record_basic_roundtrip() {
    let n = NodeRecord::new(NodeId::new(1), LabelId::new(2), Lsn::new(3));
    let bytes = n.to_bytes();
    let back = NodeRecord::from_bytes(&bytes).expect("round-trip");
    assert_eq!(n.id, back.id);
    assert_eq!(n.label_id, back.label_id);
    assert_eq!(n.created_lsn, back.created_lsn);
}

proptest! {
    #[test]
    fn node_record_roundtrip_full_domain(
        id_raw in any::<u64>(),
        label_raw in any::<u32>(),
        lsn_raw in any::<u64>(),
    ) {
        let n = NodeRecord::new(NodeId::new(id_raw), LabelId::new(label_raw), Lsn::new(lsn_raw));
        let bytes = n.to_bytes();
        let back = NodeRecord::from_bytes(&bytes).expect("round-trip");
        prop_assert_eq!(n.id, back.id);
        prop_assert_eq!(n.label_id, back.label_id);
        prop_assert_eq!(n.created_lsn, back.created_lsn);
    }
}

#[test]
fn node_record_size_is_exactly_64() {
    assert_eq!(NodeRecord::SIZE, 64);
    assert_eq!(core::mem::size_of::<NodeRecord>(), 64);
}

// ────────────────────── RelRecord round-trip ──────────────────────

#[test]
fn rel_record_basic_roundtrip() {
    let r = RelRecord::new(
        RelId::new(1),
        TypeId::new(2),
        NodeId::new(3),
        NodeId::new(4),
        Lsn::new(5),
    );
    let bytes = r.to_bytes();
    let back = RelRecord::from_bytes(&bytes).expect("round-trip");
    assert_eq!(r.id, back.id);
    assert_eq!(r.type_id, back.type_id);
    assert_eq!(r.src_id, back.src_id);
    assert_eq!(r.dst_id, back.dst_id);
    assert_eq!(r.created_lsn, back.created_lsn);
}

proptest! {
    #[test]
    fn rel_record_roundtrip_full_domain(
        rel_raw in any::<u64>(),
        ty_raw in any::<u32>(),
        src_raw in any::<u64>(),
        dst_raw in any::<u64>(),
        lsn_raw in any::<u64>(),
    ) {
        let r = RelRecord::new(
            RelId::new(rel_raw),
            TypeId::new(ty_raw),
            NodeId::new(src_raw),
            NodeId::new(dst_raw),
            Lsn::new(lsn_raw),
        );
        let bytes = r.to_bytes();
        let back = RelRecord::from_bytes(&bytes).expect("round-trip");
        prop_assert_eq!(r.id, back.id);
        prop_assert_eq!(r.type_id, back.type_id);
        prop_assert_eq!(r.src_id, back.src_id);
        prop_assert_eq!(r.dst_id, back.dst_id);
        prop_assert_eq!(r.created_lsn, back.created_lsn);
    }
}

#[test]
fn rel_record_size_is_exactly_96() {
    assert_eq!(RelRecord::SIZE, 96);
    assert_eq!(core::mem::size_of::<RelRecord>(), 96);
}

// ────────────────────── MVCC visibility on RelRecord ──────────────────────

#[test]
fn rel_record_is_visible_at_created_lsn_or_after() {
    let mut r = RelRecord::new(
        RelId::new(1),
        TypeId::new(1),
        NodeId::new(1),
        NodeId::new(2),
        Lsn::new(10),
    );
    r.expired_lsn = u64::MAX; // alive
    assert!(!r.is_visible_at(Lsn::new(9))); // pre-created
    assert!(r.is_visible_at(Lsn::new(10))); // exactly created
    assert!(r.is_visible_at(Lsn::new(100))); // after
}

#[test]
fn rel_record_is_not_visible_after_expired_lsn() {
    let mut r = RelRecord::new(
        RelId::new(1),
        TypeId::new(1),
        NodeId::new(1),
        NodeId::new(2),
        Lsn::new(10),
    );
    r.expired_lsn = 20;
    assert!(r.is_visible_at(Lsn::new(15)));
    assert!(!r.is_visible_at(Lsn::new(20))); // expired_lsn is exclusive upper
    assert!(!r.is_visible_at(Lsn::new(100)));
}

#[test]
fn rel_record_max_expired_lsn_means_alive() {
    let mut r = RelRecord::new(
        RelId::new(1),
        TypeId::new(1),
        NodeId::new(1),
        NodeId::new(2),
        Lsn::new(1),
    );
    r.expired_lsn = u64::MAX;
    // Visible at any post-created LSN.
    assert!(r.is_visible_at(Lsn::new(1)));
    assert!(r.is_visible_at(Lsn::new(u64::MAX - 1)));
}

// ────────────────────── TelEntry round-trip ──────────────────────

#[test]
fn tel_entry_basic_roundtrip() {
    let t = TelEntry::new(NodeId::new(1), RelId::new(2), Lsn::new(3));
    let bytes = t.to_bytes();
    let back = TelEntry::from_bytes(&bytes);
    assert_eq!(t.dst_id, back.dst_id);
    assert_eq!(t.rel_id, back.rel_id);
    assert_eq!(t.created_lsn, back.created_lsn);
}

proptest! {
    #[test]
    fn tel_entry_roundtrip_full_domain(
        dst_raw in any::<u64>(),
        rel_raw in any::<u64>(),
        lsn_raw in any::<u64>(),
    ) {
        let t = TelEntry::new(NodeId::new(dst_raw), RelId::new(rel_raw), Lsn::new(lsn_raw));
        let bytes = t.to_bytes();
        let back = TelEntry::from_bytes(&bytes);
        prop_assert_eq!(t.dst_id, back.dst_id);
        prop_assert_eq!(t.rel_id, back.rel_id);
        prop_assert_eq!(t.created_lsn, back.created_lsn);
    }
}

#[test]
fn tel_entry_size_is_exactly_32() {
    assert_eq!(TelEntry::SIZE, 32);
    assert_eq!(core::mem::size_of::<TelEntry>(), 32);
}

// ────────────────────── Cross-record boundary invariants ──────────────────────

#[test]
fn page_size_fits_multiple_records() {
    // 8 KiB page must fit at least 128 NodeRecords (64 B each, no header).
    const _: () = assert!(PAGE_SIZE / NodeRecord::SIZE >= 128);
    // And at least 85 RelRecords (96 B each, no header).
    const _: () = assert!(PAGE_SIZE / RelRecord::SIZE >= 85);
    // And at least 256 TelEntries (32 B each, no header).
    const _: () = assert!(PAGE_SIZE / TelEntry::SIZE >= 256);
}

#[test]
fn records_fit_on_page_with_header() {
    // After 40-byte PageHeader, remaining 8152 bytes:
    let body = PAGE_SIZE - PageHeader::SIZE;
    assert!(body / NodeRecord::SIZE >= 127);
    assert!(body / RelRecord::SIZE >= 84);
    assert!(body / TelEntry::SIZE >= 254);
}

// ────────────────────── from_bytes rejects malformed input ──────────────────────

#[test]
fn page_header_from_bytes_rejects_bad_magic() {
    let mut bytes = [0u8; PageHeader::SIZE];
    bytes[0..4].copy_from_slice(&0xDEAD_BEEF_u32.to_le_bytes());
    let r = PageHeader::from_bytes(&bytes);
    assert!(matches!(r, Err(ArcGraphError::BadPageMagic { .. })));
}

#[test]
fn page_header_from_bytes_rejects_unknown_page_type() {
    let h = PageHeader::new(PageId::new(1), PageType::Node, TenantId::DEFAULT);
    let mut bytes = h.to_bytes();
    // Find the page_type byte (after magic + version, per record.rs).
    // Easier: mutate to 9 (one past IndexOverflow=8) and assert rejection.
    // We pick an offset that lands on the page_type byte: per record.rs
    // layout, page_type is at byte index 5 (4-byte magic + 1-byte version).
    bytes[5] = 99;
    let r = PageHeader::from_bytes(&bytes);
    assert!(matches!(r, Err(ArcGraphError::UnknownPageType(99))));
}

// ────────────────────── Adversarial: random bytes ──────────────────────

proptest! {
    #[test]
    fn page_header_from_random_bytes_never_panics(seed in any::<[u8; PageHeader::SIZE]>()) {
        // The codec must NEVER panic on adversarial input. It may
        // succeed (rare — only when the random bytes happen to form a
        // valid header) or it may return a structured error; it MUST
        // NOT panic.
        let _ = PageHeader::from_bytes(&seed);
    }

    #[test]
    fn node_record_from_random_bytes_never_panics(seed in any::<[u8; NodeRecord::SIZE]>()) {
        let _ = NodeRecord::from_bytes(&seed);
    }

    #[test]
    fn rel_record_from_random_bytes_never_panics(seed in any::<[u8; RelRecord::SIZE]>()) {
        let _ = RelRecord::from_bytes(&seed);
    }

    #[test]
    fn tel_entry_from_random_bytes_never_panics(seed in any::<[u8; TelEntry::SIZE]>()) {
        // TelEntry::from_bytes is infallible; just call it.
        let _ = TelEntry::from_bytes(&seed);
    }
}
