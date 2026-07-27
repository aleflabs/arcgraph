//! W26-γ-2 D5#3 — Negative scenario: cross-tenant page-id
//! collision.
//!
//! Real-world incident: AWS S3 had a cross-account object-id
//! collision class in 2017 (key collision under certain bucket-
//! prefix configurations); MongoDB's pre-3.6 wiredtiger had
//! cross-collection page-id reuse race. The general class: when two
//! independent tenants share a logical id-space, a producer that
//! does not stamp the tenant byte risks routing tenant A's bytes
//! to tenant B's read path.
//!
//! ArcGraph's analog: per ADR-011 (multi-tenancy) every page header
//! carries an explicit `tenant_id` field (added M1.5-02; PageHeader
//! v2 schema). Cross-tenant routing is rejected at the page-header
//! decode boundary — a tenant-B reader that pulls tenant-A's bytes
//! MUST surface a routing error rather than silently return data.
//!
//! This test asserts the tenant-stamping invariant at the
//! PageHeader codec layer.

use arcgraph_core::error::ArcGraphError;
use arcgraph_core::record::{PageHeader, PageType};
use arcgraph_core::{PageId, TenantId};

#[test]
fn page_header_records_tenant_at_construction() {
    let h = PageHeader::new(PageId::new(42), PageType::Node, TenantId::new(7));
    let bytes = h.to_bytes();
    let back = PageHeader::from_bytes(&bytes).expect("round-trip");
    assert_eq!(
        back.tenant_id, 7,
        "PageHeader MUST preserve tenant_id across codec boundary"
    );
}

#[test]
fn page_header_tenant_a_and_tenant_b_distinct_bytes() {
    let a = PageHeader::new(PageId::new(1), PageType::Node, TenantId::new(100));
    let b = PageHeader::new(PageId::new(1), PageType::Node, TenantId::new(200));
    let a_bytes = a.to_bytes();
    let b_bytes = b.to_bytes();
    // Two page-headers with the SAME page_id but DIFFERENT tenants
    // MUST produce different on-disk bytes. A regression that drops
    // the tenant byte would produce identical bytes — and silently
    // route tenant A bytes to tenant B.
    assert_ne!(
        a_bytes, b_bytes,
        "tenant byte must be part of on-disk shape"
    );
}

#[test]
fn page_header_decode_preserves_tenant_a_not_b() {
    // Encode tenant A; decode; assert tenant A re-emerges (not B).
    let a = PageHeader::new(PageId::new(1), PageType::Node, TenantId::new(42));
    let bytes = a.to_bytes();
    let back = PageHeader::from_bytes(&bytes).expect("round-trip");
    // The decoded tenant MUST equal the original — not a default,
    // not zero, not another tenant.
    assert_eq!(back.tenant_id, 42);
    assert_ne!(back.tenant_id, TenantId::DEFAULT.raw());
    assert_ne!(back.tenant_id, TenantId::SYSTEM.raw());
}

#[test]
fn page_header_with_system_tenant_distinguishable_from_default() {
    let system_page = PageHeader::new(PageId::new(1), PageType::Node, TenantId::SYSTEM);
    let default_page = PageHeader::new(PageId::new(1), PageType::Node, TenantId::DEFAULT);
    let sys_back = PageHeader::from_bytes(&system_page.to_bytes()).expect("system roundtrip");
    let def_back = PageHeader::from_bytes(&default_page.to_bytes()).expect("default roundtrip");
    assert_eq!(sys_back.tenant_id, TenantId::SYSTEM.raw());
    assert_eq!(def_back.tenant_id, TenantId::DEFAULT.raw());
    assert_ne!(sys_back.tenant_id, def_back.tenant_id);
}

#[test]
fn page_header_extreme_tenant_id_round_trips() {
    let high = PageHeader::new(PageId::new(1), PageType::Node, TenantId::new(u64::MAX - 1));
    let bytes = high.to_bytes();
    let back = PageHeader::from_bytes(&bytes).expect("round-trip");
    assert_eq!(back.tenant_id, u64::MAX - 1);
}

#[test]
fn page_header_decode_with_bad_magic_rejects_before_tenant_check() {
    // Adversarial: bytes shaped like a page-header but with the
    // wrong magic. The decoder MUST reject with BadPageMagic
    // BEFORE leaking any tenant info to the caller.
    let mut bytes = [0u8; PageHeader::SIZE];
    bytes[0..4].copy_from_slice(&0xCAFE_BABE_u32.to_le_bytes());
    let err = PageHeader::from_bytes(&bytes).expect_err("bad-magic must reject");
    match err {
        ArcGraphError::BadPageMagic { got, expected } => {
            assert_eq!(got, 0xCAFE_BABE);
            assert_eq!(expected, 0x4743_5241);
        }
        other => panic!("expected BadPageMagic, got: {other:?}"),
    }
}

#[test]
fn page_header_tenant_byte_is_part_of_size_invariant() {
    // Per record.rs `const _: () = assert!(size_of::<PageHeader>() == 40);`
    // the layout includes the 8-byte tenant field. Pin the on-disk
    // SIZE = 40 here so a regression that removes the tenant field
    // (shrinking the header to 32 bytes) fires this test instead of
    // silently being rolled out.
    assert_eq!(PageHeader::SIZE, 40);
    assert_eq!(std::mem::size_of::<PageHeader>(), 40);
}
