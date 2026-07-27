//! ADR-032 Slice 3d E5 gate — `recover_from_wal` integration
//!
//! Replaces the pre-Slice-3 "uniform empty-post divergence apply-
//! on-replay unwired" E5 smoke with a real post-replay state
//! assertion: a CommitBundle durable on disk is applied via
//! `recover_from_wal`, and the reconstituted TxnManager +
//! PrimaryPageStore see every pre-crash MVCC write + index page.
//!
//! #66 is closed end-to-end after this gate: produce (ADR-031 +
//! ADR-032 Slice 1 + Slice 2) + consume (Slice 3 / this gate).

use std::collections::HashMap;
use std::sync::Arc;

use arcgraph_core::{Lsn, PAGE_SIZE, PageId, TenantId};
use arcgraph_storage::primary_index::PrimaryPageStore;
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{
    BundlePageKind, PageStoreTarget, PrimaryPageStoreHandle, SideChannelWrite, StagedEmit,
    WalConfig, WalRecordType, WalWriter, encode_commit_bundle_v8, recover_from_wal,
};
use bytes::Bytes;
use tempfile::tempdir;

fn write_bundle(
    dir: &std::path::Path,
    commit_lsn: Lsn,
    tenant: TenantId,
    mvcc: &HashMap<u64, Option<Bytes>>,
    staged: &[StagedEmit],
) {
    let cfg = WalConfig {
        dir: dir.to_path_buf(),
        segment_size_bytes: 64 * 1024 * 1024,
        group_commit_window: std::time::Duration::from_millis(2),
        group_commit_max_batch: 4,
        metrics_sink: None,
        encryption: None,

        inflight_budget_bytes: None,
    };
    let writer = WalWriter::spawn(cfg).unwrap();
    let handle = writer.handle();
    let staged_v4: Vec<(
        BundlePageKind,
        arcgraph_core::PageId,
        TenantId,
        Box<[u8; PAGE_SIZE]>,
    )> = staged
        .iter()
        .map(|e| (e.kind, e.page_id, tenant, e.bytes.clone()))
        .collect();
    let payload = encode_commit_bundle_v8(
        commit_lsn,
        tenant,
        mvcc,
        &[] as &[SideChannelWrite],
        &staged_v4,
        &[],
        &[],
        &[], // #352 Part 2: no idempotency bindings in this fixture
        &[], // #1221: no acl_grants in this fixture
    );
    handle
        .append(WalRecordType::CommitBundle, 1, 0, tenant, payload)
        .unwrap();
    writer.shutdown().unwrap();
}

#[test]
fn e5_gate_single_commit_replays_mvcc_and_pages() {
    let dir = tempdir().unwrap();
    let mut mvcc = HashMap::new();
    mvcc.insert(7u64, Some(Bytes::from_static(b"seven")));
    mvcc.insert(11u64, Some(Bytes::from_static(b"eleven")));
    let mut page_bytes: Box<[u8; PAGE_SIZE]> = Box::new([0u8; PAGE_SIZE]);
    for (i, b) in page_bytes.iter_mut().enumerate() {
        *b = (i & 0xFF) as u8;
    }
    let staged = vec![StagedEmit {
        kind: arcgraph_storage::wal::BundlePageKind::PrimaryIndex,
        page_id: PageId::new(100),
        bytes: page_bytes,
    }];
    write_bundle(dir.path(), Lsn::new(1), TenantId::DEFAULT, &mvcc, &staged);

    let txn_mgr = Arc::new(TxnManager::new());
    let primary_store = Arc::new(PrimaryPageStore::new());
    let primary: Arc<dyn PrimaryPageStoreHandle> =
        Arc::clone(&primary_store) as Arc<dyn PrimaryPageStoreHandle>;
    let target = PageStoreTarget::primary_only(primary);
    let report = recover_from_wal(dir.path(), Arc::clone(&txn_mgr), target, None).unwrap();

    assert_eq!(report.applied_commit_lsn, Lsn::new(1));
    assert_eq!(report.metrics.bundles_applied, 1);
    assert_eq!(report.metrics.mvcc_versions_installed, 2);
    assert_eq!(report.metrics.index_pages_applied, 1);
    assert!(report.last_wal_lsn > Lsn::ZERO);
    assert!(report.torn_tail.is_none());

    // Post-replay TxnManager sees both writes at snapshot=1.
    assert_eq!(
        txn_mgr
            .read_at(TenantId::DEFAULT, 7, Lsn::new(1))
            .as_deref(),
        Some(&b"seven"[..])
    );
    assert_eq!(
        txn_mgr
            .read_at(TenantId::DEFAULT, 11, Lsn::new(1))
            .as_deref(),
        Some(&b"eleven"[..])
    );
    // Page 100 is in the primary store WITH the exact staged ramp bytes
    // (byte `j` == `(j & 0xFF)`). O-F (W28-S3): was latchability-only
    // (`.is_ok()`), which proved the slot existed but not that replay
    // installed the staged byte pattern.
    let latch = primary_store
        .latch(PageId::new(100))
        .expect("page 100 must be installed after replay");
    let g = latch.read();
    assert!(
        g.as_ref()
            .as_ref()
            .iter()
            .enumerate()
            .all(|(j, &b)| b == (j & 0xFF) as u8),
        "replayed page 100 must equal the staged ramp pattern (byte j == j & 0xFF)"
    );
    // Counter advanced to 1 so next allocate() returns 2.
    assert_eq!(txn_mgr.current_lsn(), Lsn::new(1));
}

#[test]
fn e5_gate_empty_wal_yields_pristine_state() {
    let dir = tempdir().unwrap();
    let txn_mgr = Arc::new(TxnManager::new());
    let primary: Arc<dyn PrimaryPageStoreHandle> = Arc::new(PrimaryPageStore::new());
    let target = PageStoreTarget::primary_only(primary);
    let report = recover_from_wal(dir.path(), Arc::clone(&txn_mgr), target, None).unwrap();

    assert_eq!(report.applied_commit_lsn, Lsn::ZERO);
    assert_eq!(report.metrics.bundles_applied, 0);
    assert_eq!(report.metrics.records_total, 0);
    assert!(report.torn_tail.is_none());
    assert_eq!(txn_mgr.current_lsn(), Lsn::ZERO);
}
