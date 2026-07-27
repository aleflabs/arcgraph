//! ADR-032 Slice 1: CommitBundle v2 codec — multi-tenant
//! per-entry-tenant-id wire format, segment-header format_version
//! dispatch, and mixed v1+v2 segment read.
//!
//! Slice 1 only lands the codec. The commit authority still emits
//! v1 payloads until Slice 2's cutover (`commit_with_bundle_writes`
//! flips to `encode_commit_bundle_v2` in the same commit that flips
//! `SegmentHeader::current()` to `format_version=2`). So this test
//! file manually constructs segment files on disk to exercise the
//! v2 codec end-to-end through the record framing — neither path is
//! reachable via the runtime commit flow in Slice 1.

use std::time::Duration;

use arcgraph_core::{Lsn, TenantId};
use arcgraph_storage::wal::{
    BUNDLE_FORMAT_V1, BUNDLE_FORMAT_V2, SUPPORTED_WAL_FORMAT_VERSIONS, SegmentHeader,
    SideChannelWrite, WAL_SEGMENT_MAGIC, WalConfig, WalRecord, WalRecordType, WalRecoveryReader,
    WalWriter, decode_commit_bundle_for_version, decode_commit_bundle_v1, decode_commit_bundle_v2,
    encode_commit_bundle, encode_commit_bundle_v2, segment_filename,
};
use bytes::Bytes;
use std::collections::HashMap;
use tempfile::tempdir;

/// Manually stamp a WAL segment header with an explicit
/// `format_version`. Needed because `SegmentHeader::current()`
/// returns the binary's current stamp (Slice 1: v1); producing v2
/// segments in Slice 1 tests requires explicit header bytes.
fn stamp_header(version: u16) -> [u8; SegmentHeader::SIZE] {
    let mut out = [0u8; SegmentHeader::SIZE];
    out[0..4].copy_from_slice(&WAL_SEGMENT_MAGIC);
    out[4..6].copy_from_slice(&version.to_le_bytes());
    // Reserved bytes 6..8 stay zero.
    out
}

/// Build a fully-framed WAL record (CRC + header + payload bytes)
/// the way `SegmentWriter::append` would deliver it to disk.
fn frame_record(
    record_type: WalRecordType,
    txn_id: u64,
    lsn: Lsn,
    tenant_id: TenantId,
    payload: Vec<u8>,
) -> Vec<u8> {
    let rec = WalRecord {
        record_type,
        txn_id,
        lsn,
        timestamp_ms: 0,
        tenant_id,
        payload,
    };
    rec.encode_to_vec().expect("framed record")
}

fn mk_writes(entries: &[(u64, &[u8])]) -> HashMap<u64, Option<Bytes>> {
    entries
        .iter()
        .map(|(k, v)| (*k, Some(Bytes::copy_from_slice(v))))
        .collect()
}

// ─── Slice 1 test 1 — v2 codec roundtrip (3 tenants × 2 writes) ──

#[test]
fn wal_commit_bundle_v2_codec_roundtrip() {
    // ADR-032 §2: a v2 bundle carries per-entry tenant on every
    // MVCC write so one bundle can cover multiple tenants. This
    // test builds the canonical "mixed tenants + IndexPage entries"
    // shape, encodes, decodes, and asserts full roundtrip equality
    // (including tenant preservation on each write).
    let primary_tenant = TenantId::DEFAULT;
    let primary = mk_writes(&[(10u64, &b"primary-10"[..]), (20u64, &b"primary-20"[..])]);
    let custom_a = TenantId::new(42);
    let custom_b = TenantId::new(99);
    let sidechannel = vec![
        SideChannelWrite {
            tenant_id: TenantId::SYSTEM,
            key: 100,
            value: Some(Bytes::from_static(b"sys-100")),
        },
        SideChannelWrite {
            tenant_id: TenantId::SYSTEM,
            key: 200,
            value: None, // tombstone
        },
        SideChannelWrite {
            tenant_id: custom_a,
            key: 300,
            value: Some(Bytes::from_static(b"a-300")),
        },
        SideChannelWrite {
            tenant_id: custom_a,
            key: 400,
            value: None,
        },
        SideChannelWrite {
            tenant_id: custom_b,
            key: 500,
            value: Some(Bytes::from_static(b"b-500")),
        },
        SideChannelWrite {
            tenant_id: custom_b,
            key: 600,
            value: Some(Bytes::from_static(b"b-600")),
        },
    ];
    // 4 IndexPage snapshots. Content doesn't matter — we roundtrip
    // byte-identity only.
    let staged = (0..4u64)
        .map(|i| arcgraph_storage::wal::StagedEmit {
            kind: arcgraph_storage::wal::BundlePageKind::PrimaryIndex,
            page_id: arcgraph_core::PageId::new(1000 + i),
            bytes: {
                let mut b: Box<[u8; arcgraph_core::PAGE_SIZE]> =
                    Box::new([0u8; arcgraph_core::PAGE_SIZE]);
                for byte in b.iter_mut() {
                    *byte = (i as u8).wrapping_mul(0x33);
                }
                b
            },
        })
        .collect::<Vec<_>>();
    let staged_tenant = TenantId::SYSTEM;

    let payload = encode_commit_bundle_v2(
        Lsn::new(7777),
        primary_tenant,
        &primary,
        &sidechannel,
        &staged,
        staged_tenant,
    );

    let decoded = decode_commit_bundle_v2(&payload, primary_tenant).unwrap();

    // ── commit_lsn + primary tenant stamped on decoded bundle.
    assert_eq!(decoded.commit_lsn, Lsn::new(7777));
    assert_eq!(decoded.primary_tenant, primary_tenant);

    // ── Primary partition: all primary_tenant writes round-trip.
    assert_eq!(decoded.mvcc_writes.len(), 2);
    assert_eq!(
        decoded.mvcc_writes.get(&10u64).unwrap().as_deref(),
        Some(&b"primary-10"[..])
    );
    assert_eq!(
        decoded.mvcc_writes.get(&20u64).unwrap().as_deref(),
        Some(&b"primary-20"[..])
    );

    // ── Sidechannel partition: 6 non-primary writes, sorted by
    //    (tenant.raw(), key) ascending — SYSTEM=0, DEFAULT=1,
    //    custom_a=42, custom_b=99.
    assert_eq!(decoded.sidechannel_writes.len(), 6);
    let expect_order: &[(TenantId, u64, Option<&[u8]>)] = &[
        (TenantId::SYSTEM, 100, Some(&b"sys-100"[..])),
        (TenantId::SYSTEM, 200, None),
        (custom_a, 300, Some(&b"a-300"[..])),
        (custom_a, 400, None),
        (custom_b, 500, Some(&b"b-500"[..])),
        (custom_b, 600, Some(&b"b-600"[..])),
    ];
    for (i, sc) in decoded.sidechannel_writes.iter().enumerate() {
        let (exp_tenant, exp_key, exp_value) = &expect_order[i];
        assert_eq!(sc.tenant_id, *exp_tenant, "entry {i} tenant");
        assert_eq!(sc.key, *exp_key, "entry {i} key");
        assert_eq!(sc.value.as_deref(), *exp_value, "entry {i} value");
    }

    // ── IndexPage entries roundtrip unchanged.
    assert_eq!(decoded.staged_pages.len(), 4);
    for (i, p) in decoded.staged_pages.iter().enumerate() {
        assert_eq!(p.page_id.raw(), 1000 + i as u64);
        assert_eq!(p.tenant_id, staged_tenant);
        let fill = (i as u8).wrapping_mul(0x33);
        assert!(p.bytes.iter().all(|&b| b == fill));
    }
}

// ─── Slice 1 test 2 — v1 decoder synthesizes tenant from caller ──

#[test]
fn wal_commit_bundle_v1_decoder_synthesizes_tenant() {
    // A v1 bundle on the wire carries NO per-entry tenant. When the
    // replay executor processes a v1 segment it stamps the WAL
    // record header's tenant onto the decoded bundle (the "implicit
    // single-tenant context" of ADR-032 §2). This test takes a
    // manually-constructed v1 payload and decodes it under three
    // different caller-supplied tenants, asserting the v1 decoder
    // stamps each one correctly.
    let writes = mk_writes(&[(1u64, &b"alpha"[..]), (2u64, &b"beta"[..])]);
    let v1_payload = encode_commit_bundle(Lsn::new(42), &writes, &[], TenantId::DEFAULT);

    for tenant in [TenantId::DEFAULT, TenantId::SYSTEM, TenantId::new(8675)] {
        let decoded = decode_commit_bundle_v1(&v1_payload, tenant).unwrap();
        assert_eq!(
            decoded.primary_tenant, tenant,
            "v1 decoder must stamp caller-supplied tenant (record header) onto bundle"
        );
        // v1 never has sidechannel writes.
        assert!(decoded.sidechannel_writes.is_empty());
        // Primary partition carries all v1 writes under the stamped tenant.
        assert_eq!(decoded.mvcc_writes.len(), 2);
        assert_eq!(
            decoded.mvcc_writes.get(&1u64).unwrap().as_deref(),
            Some(&b"alpha"[..])
        );
        assert_eq!(
            decoded.mvcc_writes.get(&2u64).unwrap().as_deref(),
            Some(&b"beta"[..])
        );
    }
}

// ─── Slice 1 test 3 — mixed v1 + v2 segments read clean ──────────

#[test]
fn wal_segment_mixed_v1_v2_read() {
    // Build a WAL dir that contains one v1 segment followed by one
    // v2 segment, each with one CommitBundle record. The reader
    // accepts both (SUPPORTED_WAL_FORMAT_VERSIONS = [1, 2] in
    // Slice 1); per-record decode uses the segment's format_version
    // via decode_commit_bundle_for_version.
    let dir = tempdir().unwrap();

    // ── Segment 0: v1 header + one v1 CommitBundle record.
    let v1_writes = mk_writes(&[(0xABu64, &b"v1-user-write"[..])]);
    let v1_payload = encode_commit_bundle(Lsn::new(1), &v1_writes, &[], TenantId::DEFAULT);
    let v1_record = frame_record(
        WalRecordType::CommitBundle,
        /* txn_id */ 1,
        /* lsn */ Lsn::new(1),
        TenantId::DEFAULT,
        v1_payload,
    );
    let mut seg0 = Vec::new();
    seg0.extend_from_slice(&stamp_header(BUNDLE_FORMAT_V1));
    seg0.extend_from_slice(&v1_record);
    std::fs::write(dir.path().join(segment_filename(0)), &seg0).unwrap();

    // ── Segment 1: v2 header + one v2 CommitBundle record
    //    carrying both a primary write and a SYSTEM sidechannel.
    let v2_primary = mk_writes(&[(0xCDu64, &b"v2-user-write"[..])]);
    let v2_side = vec![SideChannelWrite {
        tenant_id: TenantId::SYSTEM,
        key: 0xEE,
        value: Some(Bytes::from_static(b"v2-system-write")),
    }];
    let v2_payload = encode_commit_bundle_v2(
        Lsn::new(2),
        TenantId::DEFAULT,
        &v2_primary,
        &v2_side,
        &[],
        TenantId::DEFAULT,
    );
    let v2_record = frame_record(
        WalRecordType::CommitBundle,
        /* txn_id */ 2,
        /* lsn */ Lsn::new(2),
        TenantId::DEFAULT,
        v2_payload,
    );
    let mut seg1 = Vec::new();
    seg1.extend_from_slice(&stamp_header(BUNDLE_FORMAT_V2));
    seg1.extend_from_slice(&v2_record);
    std::fs::write(dir.path().join(segment_filename(1)), &seg1).unwrap();

    // ── Reader iterates cleanly across both segments.
    //    SUPPORTED_WAL_FORMAT_VERSIONS must now include both.
    assert!(SUPPORTED_WAL_FORMAT_VERSIONS.contains(&BUNDLE_FORMAT_V1));
    assert!(SUPPORTED_WAL_FORMAT_VERSIONS.contains(&BUNDLE_FORMAT_V2));

    let reader = WalRecoveryReader::open(dir.path()).unwrap();
    let records: Vec<WalRecord> = reader.collect_all().unwrap();
    assert_eq!(records.len(), 2, "expected 1 v1 record + 1 v2 record");

    // Record 0 is from segment 0 (v1). Decode with v1 → primary
    // partition = {0xAB → "v1-user-write"}; sidechannel empty.
    let r0 = &records[0];
    assert_eq!(r0.record_type, WalRecordType::CommitBundle);
    let b0 = decode_commit_bundle_for_version(&r0.payload, BUNDLE_FORMAT_V1, r0.tenant_id).unwrap();
    assert_eq!(b0.commit_lsn, Lsn::new(1));
    assert_eq!(b0.primary_tenant, TenantId::DEFAULT);
    assert_eq!(b0.mvcc_writes.len(), 1);
    assert_eq!(
        b0.mvcc_writes.get(&0xABu64).unwrap().as_deref(),
        Some(&b"v1-user-write"[..])
    );
    assert!(b0.sidechannel_writes.is_empty());

    // Record 1 is from segment 1 (v2). Decode with v2 → primary
    // partition = {0xCD → "v2-user-write"}; sidechannel has SYSTEM
    // 0xEE → "v2-system-write".
    let r1 = &records[1];
    assert_eq!(r1.record_type, WalRecordType::CommitBundle);
    let b1 = decode_commit_bundle_for_version(&r1.payload, BUNDLE_FORMAT_V2, r1.tenant_id).unwrap();
    assert_eq!(b1.commit_lsn, Lsn::new(2));
    assert_eq!(b1.primary_tenant, TenantId::DEFAULT);
    assert_eq!(b1.mvcc_writes.len(), 1);
    assert_eq!(
        b1.mvcc_writes.get(&0xCDu64).unwrap().as_deref(),
        Some(&b"v2-user-write"[..])
    );
    assert_eq!(b1.sidechannel_writes.len(), 1);
    assert_eq!(b1.sidechannel_writes[0].tenant_id, TenantId::SYSTEM);
    assert_eq!(b1.sidechannel_writes[0].key, 0xEE);
    assert_eq!(
        b1.sidechannel_writes[0].value.as_deref(),
        Some(&b"v2-system-write"[..])
    );
}

// ─── Slice 2 test — runtime WalWriter segments stamp v2 ──────────

#[test]
fn wal_segment_format_version_2_stamped_on_new_segments() {
    // Issue #129 P0 fix cutover: a fresh WalWriter stamps v4 on
    // every new segment. Commit path emits v4 bundles
    // simultaneously, so v4-stamped segments always carry
    // v4-shaped CommitBundles (extends v3 with allocator_advances
    // tail). Test name retained for git-history / blame
    // continuity.
    let dir = tempdir().unwrap();
    let config = WalConfig {
        dir: dir.path().to_path_buf(),
        segment_size_bytes: 16 * 1024 * 1024,
        group_commit_window: Duration::from_millis(1),
        group_commit_max_batch: 1,
        metrics_sink: None,
        encryption: None,

        inflight_budget_bytes: None,
    };
    let writer = WalWriter::spawn(config).unwrap();
    writer
        .handle()
        .append(WalRecordType::PutNode, 1, 0, TenantId::DEFAULT, vec![0xAA])
        .unwrap();
    writer.shutdown().unwrap();

    let path = dir.path().join(segment_filename(0));
    let bytes = std::fs::read(&path).unwrap();
    let header = SegmentHeader::decode(&bytes[..SegmentHeader::SIZE]).unwrap();
    assert_eq!(
        header.format_version,
        arcgraph_storage::wal::CURRENT_WAL_FORMAT_VERSION,
        "writer stamps the CURRENT format version on new segments \
         (#352 Part 2 / ADR-199 bumped it to v6)"
    );
}
