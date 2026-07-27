//! ADR-031 regression: every MVCC commit emits exactly one
//! `CommitBundle` WAL record.
//!
//! Pre-ADR-031 shape: each commit produced `1 IndexPage` (from the
//! primary-index staged-emit drain) + `1 CommitMarker` (from
//! `Transaction::commit_writes`) = **2 WAL records** and **2
//! sequential group-commit fires** per commit. On M2-E2 that
//! capped throughput at ~684 TPS even after the ADR-030 gate
//! relaxation.
//!
//! Post-ADR-031: `crud::commit` calls
//! `tx.commit_with_bundle(builder)`; the builder collects staged
//! `IndexPage` snapshots from `primary.upsert_deferred` and the MVCC
//! kernel's Phase 2 fires one `wal.append(CommitBundle)` record with
//! both the MVCC write-set AND the IndexPage entries in a single
//! atomic payload. `records/commit` drops from 2.02 → 1.00 on the
//! E2 workload.
//!
//! Load-bearing assertions:
//! - every commit produces exactly one `CommitBundle` WAL record
//!   (records/commit ≈ 1.00);
//! - the bundle payload decodes cleanly and carries both MVCC
//!   writes + IndexPage entries for the primary-index mutation;
//! - the legacy `Commit = 2` and standalone `IndexPage = 11` record
//!   types are NEVER emitted on the commit path post-fix (they stay
//!   in the codec for pre-fix WAL compat only).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use arcgraph_core::{LabelId, TenantId};
use arcgraph_storage::crud::{CrudStore, PropertyData, commit, create_node};
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::PrimaryIndex;
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::bundle::decode_commit_bundle_v8;
use arcgraph_storage::wal::segment::{SegmentHeader, list_segments, segment_filename};
use arcgraph_storage::wal::{WalConfig, WalRecord, WalRecordType, WalWriter};
use tempfile::TempDir;

fn test_wal_config(dir: PathBuf) -> WalConfig {
    WalConfig {
        dir,
        segment_size_bytes: 64 * 1024 * 1024,
        group_commit_window: Duration::from_millis(1),
        group_commit_max_batch: 16,
        metrics_sink: None,
        encryption: None,
        inflight_budget_bytes: None,
    }
}

fn drain_segments(dir: &std::path::Path) -> Vec<WalRecord> {
    let mut out = Vec::new();
    for seg in list_segments(dir).unwrap() {
        let bytes = std::fs::read(dir.join(segment_filename(seg))).unwrap();
        // Skip the fixed segment header (magic + format_version + reserved)
        // landed by ADR-031's companion fix #39 (PR #68).
        if bytes.len() < SegmentHeader::SIZE {
            continue;
        }
        SegmentHeader::decode(&bytes[..SegmentHeader::SIZE]).unwrap();
        let mut cursor = SegmentHeader::SIZE;
        while cursor < bytes.len() {
            let (r, consumed) = WalRecord::decode(&bytes[cursor..]).unwrap();
            out.push(r);
            cursor += consumed;
        }
    }
    out
}

/// Build a CrudStore wired with a WAL writer, a TxnManager that
/// logs every commit, and a primary index that emits IndexPage
/// records via the WAL. Returns `(TempDir, CrudStore, TxnManager,
/// WalWriter)` — the caller owns the WalWriter and calls
/// `.shutdown()` at end-of-test to flush the final batch.
fn build_stack() -> (TempDir, CrudStore, Arc<TxnManager>, WalWriter) {
    let dir = TempDir::new().unwrap();
    let writer = WalWriter::spawn(test_wal_config(dir.path().to_path_buf())).unwrap();
    let handle = writer.handle();
    let mgr = Arc::new(TxnManager::with_wal(handle.clone()));
    let alloc = Arc::new(PageAllocator::new());
    let primary = Arc::new(
        PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&alloc), Some(handle.clone())).unwrap(),
    );
    let store = CrudStore::new_with_index(Some(handle.clone()), Arc::clone(&primary), alloc);
    (dir, store, mgr, writer)
}

#[test]
fn single_create_node_emits_exactly_one_commit_bundle() {
    let (dir, store, mgr, writer) = build_stack();

    let mut tx = mgr.begin(TenantId::DEFAULT);
    let id = create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(7),
        &PropertyData::InlineU32Pair(3, 4),
    )
    .unwrap();
    commit(tx, &store).unwrap();
    writer.shutdown().unwrap();

    // Count WAL records by type. The PrimaryIndex::new bootstrap
    // emits 1 legacy IndexPage (for the fresh root leaf) + 1
    // CommitBundle (for the SYSTEM root-pointer MVCC commit). The
    // user commit contributes exactly 1 additional CommitBundle.
    let records = drain_segments(dir.path());
    let n_bundle = records
        .iter()
        .filter(|r| r.record_type == WalRecordType::CommitBundle)
        .count();
    let n_legacy_commit = records
        .iter()
        .filter(|r| r.record_type == WalRecordType::Commit)
        .count();
    let n_legacy_indexpage = records
        .iter()
        .filter(|r| r.record_type == WalRecordType::IndexPage)
        .count();

    assert_eq!(
        n_legacy_commit, 0,
        "ADR-031: legacy Commit = 2 records MUST NOT be emitted (found {n_legacy_commit}): {records:?}"
    );
    // Exactly two CommitBundles expected: bootstrap SYSTEM-tenant
    // root write + user commit. IndexPage legacy records come from
    // the bootstrap fresh-root emit (1) — acceptable because
    // PrimaryIndex::new is pre-bundle (called outside any txn).
    assert_eq!(
        n_bundle, 2,
        "expected 2 CommitBundle records (bootstrap + user); got {n_bundle}"
    );
    assert!(
        n_legacy_indexpage <= 1,
        "bootstrap is the only emitter of legacy IndexPage records; got {n_legacy_indexpage}"
    );

    // Find the user commit bundle (tenant = DEFAULT, not SYSTEM).
    let user_bundle = records
        .iter()
        .find(|r| r.record_type == WalRecordType::CommitBundle && r.tenant_id == TenantId::DEFAULT)
        .expect("user-tenant CommitBundle must be present");
    let decoded = decode_commit_bundle_v8(&user_bundle.payload, user_bundle.tenant_id).unwrap();
    assert_eq!(decoded.mvcc_writes.len(), 1, "one node MVCC write");
    assert!(
        !decoded.staged_pages.is_empty(),
        "user bundle must carry the primary-index IndexPage emit"
    );
    let _ = id;
}

#[test]
fn eight_commits_emit_exactly_eight_user_bundles() {
    // ADR-031 records/commit target: records/commit == 1.00 on a
    // no-split insert workload. Eight sequential create_node calls
    // land 8 user-tenant CommitBundles + the bootstrap pair (1
    // SYSTEM CommitBundle + 1 legacy IndexPage from PrimaryIndex::
    // new). No other WAL records should appear.
    let (dir, store, mgr, writer) = build_stack();
    for i in 0..8u32 {
        let mut tx = mgr.begin(TenantId::DEFAULT);
        create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(i),
            &PropertyData::Empty,
        )
        .unwrap();
        commit(tx, &store).unwrap();
    }
    writer.shutdown().unwrap();

    let records = drain_segments(dir.path());
    let user_bundles: Vec<&WalRecord> = records
        .iter()
        .filter(|r| {
            r.record_type == WalRecordType::CommitBundle && r.tenant_id == TenantId::DEFAULT
        })
        .collect();
    assert_eq!(
        user_bundles.len(),
        8,
        "expected 8 user CommitBundles, got {} (records: {:?})",
        user_bundles.len(),
        records
    );

    // Every user bundle carries exactly one MVCC write + at least
    // one IndexPage. records/commit for the user workload == 1.0
    // exactly (each commit is a single CommitBundle record).
    for (i, b) in user_bundles.iter().enumerate() {
        let decoded = decode_commit_bundle_v8(&b.payload, b.tenant_id).unwrap();
        assert_eq!(decoded.mvcc_writes.len(), 1, "bundle {i}: 1 MVCC write");
        assert!(
            !decoded.staged_pages.is_empty(),
            "bundle {i}: must carry IndexPage entries"
        );
    }
}

#[test]
fn raw_txn_without_store_emits_zero_index_pages() {
    // A `Transaction::commit()` call that bypasses `crud::commit`
    // (no builder) emits a CommitBundle with zero IndexPage
    // entries — byte-compatible with the legacy `Commit = 2`
    // payload plus a trailing `n_index_pages = 0` field.
    let dir = TempDir::new().unwrap();
    let writer = WalWriter::spawn(test_wal_config(dir.path().to_path_buf())).unwrap();
    let mgr = TxnManager::with_wal(writer.handle());

    let mut tx = mgr.begin(TenantId::DEFAULT);
    tx.write(42, bytes::Bytes::from_static(b"hello"));
    let commit_lsn = tx.commit().unwrap();
    writer.shutdown().unwrap();

    let records = drain_segments(dir.path());
    assert_eq!(records.len(), 1, "one CommitBundle per commit");
    let r = &records[0];
    assert_eq!(r.record_type, WalRecordType::CommitBundle);
    let decoded = decode_commit_bundle_v8(&r.payload, r.tenant_id).unwrap();
    assert_eq!(decoded.commit_lsn, commit_lsn);
    assert_eq!(decoded.mvcc_writes.len(), 1);
    assert!(
        decoded.staged_pages.is_empty(),
        "no-builder commit must produce an empty IndexPage section"
    );
}

#[test]
fn records_per_commit_approaches_one_over_many_commits() {
    // E2-shape probe without the bench harness: 32 sequential
    // `create_node` commits should yield records/commit ≈ 1.00 on
    // the user tenant (1 CommitBundle per commit). The bootstrap
    // legacy records (≤ 2) don't count toward the ratio because
    // they fire before the measurement window.
    let (dir, store, mgr, writer) = build_stack();
    let target_commits = 32u32;
    for i in 0..target_commits {
        let mut tx = mgr.begin(TenantId::DEFAULT);
        create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(i),
            &PropertyData::Empty,
        )
        .unwrap();
        commit(tx, &store).unwrap();
    }
    writer.shutdown().unwrap();

    let records = drain_segments(dir.path());
    let user_bundles = records
        .iter()
        .filter(|r| {
            r.record_type == WalRecordType::CommitBundle && r.tenant_id == TenantId::DEFAULT
        })
        .count();
    let ratio = user_bundles as f64 / target_commits as f64;
    assert_eq!(
        user_bundles, target_commits as usize,
        "user tenant: one CommitBundle per commit (got {user_bundles}, expected {target_commits})"
    );
    assert!(
        (ratio - 1.0).abs() < 0.01,
        "records/commit ratio {ratio} must be 1.00 on no-split inserts"
    );
}
