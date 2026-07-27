//! ADR-032 §9 replay regression suite.
//!
//! Each `M*` test in this file maps 1:1 onto a row of the
//! §9.2 crash-replay matrix. Row M13 lands partially (gap
//! tolerance); the full M13 pathological-slow-builder proptest
//! is deferred to the follow-up Slice 5 Jepsen-torture session.
//!
//! Together with the in-tree `wal::replay::tests` unit tests,
//! this file is the ADR-032 testing obligation §9.1 + §9.2 + a
//! §9.3 property check. §9.4 Jepsen torture + §9.5 bench impact
//! are out of this slice's scope.
//!
//! The suite hits the public `recover_from_wal` entry point
//! (Slice 3d) rather than the internal executor, so each test
//! doubles as an E5-gate end-to-end smoke.

use std::collections::HashMap;
use std::sync::Arc;

use arcgraph_core::{ArcGraphError, Lsn, PAGE_SIZE, PageId, TenantId};
use arcgraph_storage::primary_index::PrimaryPageStore;
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{
    BundlePageKind, PageStoreTarget, PrimaryPageStoreHandle, ReplayConfig, ReplayExecutor,
    SideChannelWrite, StagedEmit, WalConfig, WalRecordType, WalRecoveryReader, WalWriter,
    encode_commit_bundle_v4, encode_commit_bundle_v8, recover_from_wal,
};
use bytes::Bytes;
use proptest::prelude::*;
use tempfile::tempdir;

// ─── Test helpers ───────────────────────────────────────────────

fn mk_page(fill: u8) -> Box<[u8; PAGE_SIZE]> {
    Box::new([fill; PAGE_SIZE])
}

fn wal_cfg(dir: &std::path::Path) -> WalConfig {
    WalConfig {
        dir: dir.to_path_buf(),
        segment_size_bytes: 64 * 1024 * 1024,
        group_commit_window: std::time::Duration::from_millis(2),
        group_commit_max_batch: 16,
        metrics_sink: None,
        encryption: None,
        inflight_budget_bytes: None,
    }
}

/// Write a v6 CommitBundle record with the given `commit_lsn` +
/// MVCC writes + side-channel writes + staged_pages entries.
///
/// **#352 Part 2 (ADR-199) update:** the WAL writer stamps v6 segment
/// headers, so this helper encodes v6 payloads (extends v5 with a
/// trailing `idempotency_bindings` section). Empty `allocator_advances`,
/// `vector_pages`, and `idempotency_bindings` keep existing regression
/// tests (which exercise replay behavior independent of issue #129 /
/// Slice G.4 / #352 staging) shape-compatible.
fn write_bundle(
    dir: &std::path::Path,
    commit_lsn: Lsn,
    tenant: TenantId,
    mvcc: &HashMap<u64, Option<Bytes>>,
    sidechannel: &[SideChannelWrite],
    staged: &[StagedEmit],
) {
    let writer = WalWriter::spawn(wal_cfg(dir)).unwrap();
    let handle = writer.handle();
    let staged_v4: Vec<(BundlePageKind, PageId, TenantId, Box<[u8; PAGE_SIZE]>)> = staged
        .iter()
        .map(|e| (e.kind, e.page_id, tenant, e.bytes.clone()))
        .collect();
    let payload = encode_commit_bundle_v8(
        commit_lsn,
        tenant,
        mvcc,
        sidechannel,
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

/// Write N bundles where commit_lsn `i` has an MVCC write at key `i`
/// with value `b"v{i}"` and one IndexPage entry at page_id 100+i.
fn write_n_bundles(dir: &std::path::Path, n: u64) {
    for i in 1..=n {
        let mut mvcc = HashMap::new();
        mvcc.insert(i, Some(Bytes::from(format!("v{i}"))));
        let staged = vec![StagedEmit {
            kind: arcgraph_storage::wal::BundlePageKind::PrimaryIndex,
            page_id: PageId::new(100 + i),
            bytes: mk_page((i % 256) as u8),
        }];
        write_bundle(dir, Lsn::new(i), TenantId::DEFAULT, &mvcc, &[], &staged);
    }
}

fn fresh_target() -> (Arc<TxnManager>, Arc<PrimaryPageStore>, PageStoreTarget) {
    let txn_mgr = Arc::new(TxnManager::new());
    let primary_store = Arc::new(PrimaryPageStore::new());
    let primary: Arc<dyn PrimaryPageStoreHandle> =
        Arc::clone(&primary_store) as Arc<dyn PrimaryPageStoreHandle>;
    let target = PageStoreTarget::primary_only(primary);
    (txn_mgr, primary_store, target)
}

// ─── M1 — empty WAL ─────────────────────────────────────────────

#[test]
fn m1_replay_empty_wal_is_noop() {
    let dir = tempdir().unwrap();
    let (txn_mgr, primary_store, target) = fresh_target();
    let report = recover_from_wal(dir.path(), Arc::clone(&txn_mgr), target, None).unwrap();
    assert_eq!(report.applied_commit_lsn, Lsn::ZERO);
    assert_eq!(report.metrics.bundles_applied, 0);
    assert_eq!(txn_mgr.current_lsn(), Lsn::ZERO);
    assert!(primary_store.is_empty());
}

// ─── M2 — single bundle applies cleanly ─────────────────────────

#[test]
fn m2_replay_single_bundle_applies_cleanly() {
    let dir = tempdir().unwrap();
    let mut mvcc = HashMap::new();
    mvcc.insert(42u64, Some(Bytes::from_static(b"the answer")));
    let staged = vec![StagedEmit {
        kind: arcgraph_storage::wal::BundlePageKind::PrimaryIndex,
        page_id: PageId::new(1000),
        bytes: mk_page(0x7E),
    }];
    write_bundle(
        dir.path(),
        Lsn::new(1),
        TenantId::DEFAULT,
        &mvcc,
        &[],
        &staged,
    );

    let (txn_mgr, primary_store, target) = fresh_target();
    let report = recover_from_wal(dir.path(), Arc::clone(&txn_mgr), target, None).unwrap();
    assert_eq!(report.applied_commit_lsn, Lsn::new(1));
    assert_eq!(report.metrics.bundles_applied, 1);
    assert_eq!(
        txn_mgr
            .read_at(TenantId::DEFAULT, 42, Lsn::new(1))
            .as_deref(),
        Some(&b"the answer"[..])
    );
    // O-F (W28-S3): read the staged page back and assert its bytes equal
    // the 0x7E fill — was latchability-only (`.is_ok()`), which proved the
    // page slot existed but NOT that replay installed the staged bytes (a
    // replay that latched an all-zero / wrong page would have passed).
    let latch = primary_store
        .latch(PageId::new(1000))
        .expect("staged page 1000 must be installed after replay");
    let g = latch.read();
    assert!(
        g.as_ref().as_ref().iter().all(|&b| b == 0x7E),
        "replayed staged page 1000 must equal the 0x7E fill pattern"
    );
}

// ─── M3 — 100 bundles, all applied in order ─────────────────────

#[test]
fn m3_replay_multiple_bundles_in_order() {
    let dir = tempdir().unwrap();
    write_n_bundles(dir.path(), 100);

    let (txn_mgr, primary_store, target) = fresh_target();
    let report = recover_from_wal(dir.path(), Arc::clone(&txn_mgr), target, None).unwrap();
    assert_eq!(report.applied_commit_lsn, Lsn::new(100));
    assert_eq!(report.metrics.bundles_applied, 100);
    assert_eq!(report.metrics.mvcc_versions_installed, 100);
    assert_eq!(report.metrics.index_pages_applied, 100);

    // All 100 MVCC writes visible.
    for i in 1u64..=100 {
        let expected = format!("v{i}");
        let got = txn_mgr.read_at(TenantId::DEFAULT, i, Lsn::new(100));
        assert_eq!(got.as_deref(), Some(expected.as_bytes()), "key {i}");
        // All 100 index pages installed with the EXACT staged fill byte
        // `(i % 256)`. O-F (W28-S3): was latchability-only (`.is_ok()`),
        // blind to a replay that installed the page slot but with wrong
        // / zeroed bytes.
        let latch = primary_store
            .latch(PageId::new(100 + i))
            .unwrap_or_else(|e| panic!("page {} must be installed: {e:?}", 100 + i));
        let g = latch.read();
        let fill = (i % 256) as u8;
        assert!(
            g.as_ref().as_ref().iter().all(|&b| b == fill),
            "replayed page {} must equal staged fill {fill:#04x}",
            100 + i
        );
    }
    assert_eq!(txn_mgr.current_lsn(), Lsn::new(100));
}

// ─── M4 — out-of-order arrival sorts via buffer ─────────────────

#[test]
fn m4_replay_out_of_order_arrival_sorts_via_buffer() {
    // Craft a WAL with commit_lsns in the order [5, 3, 7, 1] —
    // the buffer must sort them and apply in ascending order.
    let dir = tempdir().unwrap();
    for lsn in [5u64, 3, 7, 1] {
        let mut mvcc = HashMap::new();
        mvcc.insert(lsn, Some(Bytes::from(format!("v{lsn}"))));
        write_bundle(
            dir.path(),
            Lsn::new(lsn),
            TenantId::DEFAULT,
            &mvcc,
            &[],
            &[],
        );
    }

    let (txn_mgr, _primary_store, target) = fresh_target();
    let report = recover_from_wal(dir.path(), Arc::clone(&txn_mgr), target, None).unwrap();
    assert_eq!(report.applied_commit_lsn, Lsn::new(7));
    assert_eq!(report.metrics.bundles_applied, 4);

    // All 4 writes visible at snapshot=7.
    for lsn in [1u64, 3, 5, 7] {
        let expected = format!("v{lsn}");
        let got = txn_mgr.read_at(TenantId::DEFAULT, lsn, Lsn::new(7));
        assert_eq!(got.as_deref(), Some(expected.as_bytes()));
    }
}

// ─── M5 — torn tail halts gracefully ────────────────────────────

#[test]
fn m5_replay_torn_tail_halts_gracefully() {
    let dir = tempdir().unwrap();
    write_n_bundles(dir.path(), 5);
    // Truncate the last segment to produce a torn tail.
    let segs = arcgraph_storage::wal::list_segments(dir.path()).unwrap();
    let last_seg = *segs.last().unwrap();
    let path = dir
        .path()
        .join(arcgraph_storage::wal::segment_filename(last_seg));
    let len = std::fs::metadata(&path).unwrap().len();
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(len.saturating_sub(30))
        .unwrap();

    let (_, _, target) = fresh_target();
    // Torn tail ⇒ replay completes cleanly (non-fatal).
    let (txn_mgr, _ps, target_fresh) = fresh_target();
    let _ = target;
    let report = recover_from_wal(dir.path(), Arc::clone(&txn_mgr), target_fresh, None).unwrap();
    // Some prefix applied; tail torn.
    assert!(report.metrics.bundles_applied < 5);
    assert!(report.torn_tail.is_some());
}

// ─── M6 — format mismatch halts with error ──────────────────────

#[test]
fn m6_replay_format_mismatch_halts_with_error() {
    // Hand-craft a segment file with a bogus format_version.
    let dir = tempdir().unwrap();
    let seg_path = dir.path().join(arcgraph_storage::wal::segment_filename(0));
    let mut buf = Vec::new();
    buf.extend_from_slice(&arcgraph_storage::wal::WAL_SEGMENT_MAGIC);
    buf.extend_from_slice(&99u16.to_le_bytes()); // unsupported
    buf.extend_from_slice(&0u16.to_le_bytes());
    std::fs::write(&seg_path, &buf).unwrap();

    let (txn_mgr, _, target) = fresh_target();
    let err = recover_from_wal(dir.path(), txn_mgr, target, None).unwrap_err();
    match err {
        ArcGraphError::WalFormatMismatch { found_version, .. } => {
            assert_eq!(found_version, 99);
        }
        other => panic!("expected WalFormatMismatch, got {other:?}"),
    }
}

// ─── M7 — double replay is idempotent ───────────────────────────

#[test]
fn m7_replay_idempotent_on_double_run() {
    let dir = tempdir().unwrap();
    write_n_bundles(dir.path(), 5);

    let (txn_mgr, _ps, target) = fresh_target();
    let report1 = recover_from_wal(dir.path(), Arc::clone(&txn_mgr), target, None).unwrap();
    assert_eq!(report1.applied_commit_lsn, Lsn::new(5));
    assert_eq!(report1.metrics.bundles_applied, 5);

    // Second run on same TxnManager → 0 new bundles applied.
    let primary2: Arc<dyn PrimaryPageStoreHandle> = Arc::new(PrimaryPageStore::new());
    let target2 = PageStoreTarget::primary_only(primary2);
    let report2 = recover_from_wal(dir.path(), Arc::clone(&txn_mgr), target2, None).unwrap();
    assert_eq!(report2.applied_commit_lsn, Lsn::new(5));
    assert_eq!(report2.metrics.bundles_applied, 0);
    assert!(report2.metrics.bundles_skipped_idempotent >= 5);
}

// ─── M8 — SYSTEM sidechannel write folded into user bundle ──────

#[test]
fn m8_replay_with_grow_root_system_sidechannel() {
    // ADR-032 §2 F1: a user commit can carry a SYSTEM-tenant
    // root-pointer write alongside its user MVCC writes. Replay
    // must apply both to their respective tenant chains.
    let dir = tempdir().unwrap();
    let mut mvcc = HashMap::new();
    mvcc.insert(99u64, Some(Bytes::from_static(b"user-value")));
    let sidechannel = vec![SideChannelWrite {
        tenant_id: TenantId::SYSTEM,
        key: 1, // PRIMARY_INDEX_ROOT_KEY
        value: Some(Bytes::copy_from_slice(&7u64.to_le_bytes())),
    }];
    write_bundle(
        dir.path(),
        Lsn::new(1),
        TenantId::DEFAULT,
        &mvcc,
        &sidechannel,
        &[],
    );

    let (txn_mgr, _ps, target) = fresh_target();
    let report = recover_from_wal(dir.path(), Arc::clone(&txn_mgr), target, None).unwrap();
    assert_eq!(report.applied_commit_lsn, Lsn::new(1));
    assert_eq!(report.metrics.mvcc_versions_installed, 2);

    // User tenant sees the user value.
    assert_eq!(
        txn_mgr
            .read_at(TenantId::DEFAULT, 99, Lsn::new(1))
            .as_deref(),
        Some(&b"user-value"[..])
    );
    // SYSTEM tenant sees the root-pointer value.
    let root = txn_mgr.read_at(TenantId::SYSTEM, 1, Lsn::new(1));
    let root_bytes: [u8; 8] = root.as_deref().unwrap().try_into().unwrap();
    assert_eq!(u64::from_le_bytes(root_bytes), 7);
}

// ─── M9 — buffer overflow spills to disk ────────────────────────

#[test]
fn m9_replay_buffer_overflow_spills_to_disk() {
    let dir = tempdir().unwrap();
    write_n_bundles(dir.path(), 30);
    let spill_dir = tempdir().unwrap();
    let (txn_mgr, primary_store, target) = fresh_target();
    let cfg = ReplayConfig {
        max_buffer_bundles: 4,
        max_buffer_bytes: usize::MAX,
        spill_enabled: true,
        spill_dir: spill_dir.path().to_path_buf(),
    };
    let report = recover_from_wal(dir.path(), Arc::clone(&txn_mgr), target, Some(cfg)).unwrap();
    assert_eq!(report.applied_commit_lsn, Lsn::new(30));
    assert_eq!(report.metrics.bundles_applied, 30);
    assert!(report.metrics.spill_files_created > 0);
    assert!(report.metrics.bundles_spilled > 0);

    for i in 1u64..=30 {
        let expected = format!("v{i}");
        let got = txn_mgr.read_at(TenantId::DEFAULT, i, Lsn::new(30));
        assert_eq!(got.as_deref(), Some(expected.as_bytes()));
        // O-F (W28-S3): assert the replayed page bytes equal the staged
        // fill `(i % 256)`, not just that the page is latchable — the
        // spill→reload path must preserve the staged page body.
        let latch = primary_store
            .latch(PageId::new(100 + i))
            .unwrap_or_else(|e| panic!("page {} must be installed: {e:?}", 100 + i));
        let g = latch.read();
        let fill = (i % 256) as u8;
        assert!(
            g.as_ref().as_ref().iter().all(|&b| b == fill),
            "replayed page {} must equal staged fill {fill:#04x}",
            100 + i
        );
    }
}

// ─── M10 — spill file recovery on crash during replay ───────────

#[test]
fn m10_replay_spill_file_recovery_on_crash_during_replay() {
    // Simulated scenario:
    //   1. Executor mid-replay spills N bundles; crashes before
    //      final_drain.
    //   2. Next replay opens the WAL + finds the spill files on
    //      disk → loads them + merges with the WAL re-read.
    //   3. Post-replay state matches the single-pass baseline.
    //
    // We don't actually crash here; we execute a first partial
    // replay that ends while spill files still exist on disk
    // (by leaving the spill dir intact), then re-run replay
    // with the same spill dir and assert metrics reflect the
    // reloaded spill files.
    let wal_dir = tempdir().unwrap();
    write_n_bundles(wal_dir.path(), 12);
    let spill_dir = tempdir().unwrap();

    // Manually drive the executor to spill some bundles then
    // halt before final_drain.
    let (txn_mgr, _ps, target) = fresh_target();
    let cfg = ReplayConfig {
        max_buffer_bundles: 3,
        max_buffer_bytes: usize::MAX,
        spill_enabled: true,
        spill_dir: spill_dir.path().to_path_buf(),
    };
    // We can't truly "crash" a Rust test, so instead we run the
    // full replay, verify that (a) spill files existed during
    // replay (gauge ≥ 1 file created), and (b) post-completion
    // the spill dir is cleared.
    let reader = WalRecoveryReader::open(wal_dir.path()).unwrap();
    let mut exec = ReplayExecutor::new(cfg.clone(), Arc::clone(&txn_mgr), target);
    let applied = exec.run(reader).unwrap();
    assert_eq!(applied, Lsn::new(12));
    let snap = exec.metrics().snapshot();
    assert!(
        snap.spill_files_created > 0,
        "expected spill file created during overflow path"
    );

    // After completion, the spill dir was discarded (§5 "deleted
    // on successful replay completion"); this is the §5 contract.
    assert!(
        !spill_dir.path().exists()
            || std::fs::read_dir(spill_dir.path())
                .map(|it| it.count() == 0)
                .unwrap_or(true),
        "spill dir should be empty/removed after successful replay"
    );

    // Synthesize M10's "spill persists across a crashed first
    // replay" scenario: manually write a spill file containing
    // bundles 13 and 14, then run a fresh executor whose WAL
    // ALSO contains bundles 13+14. The executor should pick
    // them up from spill without double-applying from WAL (Lemma
    // I1 de-dup).
    //
    // Note: without a real crash harness, this subtest is a
    // best-effort approximation. The full crash-during-replay
    // scenario lands in the Slice 5 Jepsen-torture follow-up.
    let spill2 = tempdir().unwrap();
    let bundles = vec![
        arcgraph_storage::wal::decode_commit_bundle_v4(
            &encode_commit_bundle_v4(
                Lsn::new(13),
                TenantId::DEFAULT,
                &HashMap::from([(13u64, Some(Bytes::from_static(b"v13-spill")))]),
                &[],
                &[],
                &[],
            ),
            TenantId::DEFAULT,
        )
        .unwrap(),
    ];
    arcgraph_storage::wal::spill::write_spill_batch(spill2.path(), &bundles).unwrap();
    assert!(arcgraph_storage::wal::spill::count_spill_files(spill2.path()).unwrap() >= 1);

    // Append bundle 13 to a NEW wal dir + run replay from it
    // with the pre-populated spill_dir.
    let wal_dir2 = tempdir().unwrap();
    let mut mvcc13 = HashMap::new();
    mvcc13.insert(13u64, Some(Bytes::from_static(b"v13-wal")));
    write_bundle(
        wal_dir2.path(),
        Lsn::new(13),
        TenantId::DEFAULT,
        &mvcc13,
        &[],
        &[],
    );
    let (txn_mgr2, _ps2, target2) = fresh_target();
    let cfg2 = ReplayConfig {
        max_buffer_bundles: 8192,
        max_buffer_bytes: usize::MAX,
        spill_enabled: true,
        spill_dir: spill2.path().to_path_buf(),
    };
    let r = recover_from_wal(wal_dir2.path(), Arc::clone(&txn_mgr2), target2, Some(cfg2)).unwrap();
    assert_eq!(r.applied_commit_lsn, Lsn::new(13));
    assert_eq!(
        r.metrics.bundles_applied, 1,
        "dedup: one logical commit_lsn"
    );
    // Either WAL value or spill value landed — both are at the
    // same commit_lsn so Lemma I1 picks one deterministically.
    // The executor drains the in-memory buffer first, then spill,
    // so the first-seen version wins the chain.last slot; the
    // second is skipped as idempotent. Either value is
    // acceptable per §9 M10.
    let got = txn_mgr2.read_at(TenantId::DEFAULT, 13, Lsn::new(13));
    let got_bytes = got.as_deref().unwrap();
    assert!(
        got_bytes == b"v13-wal" || got_bytes == b"v13-spill",
        "unexpected value: {:?}",
        got_bytes
    );
}

// ─── M11 — orphan page invokes bootstrap ────────────────────────

#[test]
fn m11_replay_orphan_page_invokes_bootstrap_from_mvcc() {
    use arcgraph_storage::primary_index::encode_index_page_payload;
    let dir = tempdir().unwrap();
    // Legacy IndexPage record — orphan.
    let writer = WalWriter::spawn(wal_cfg(dir.path())).unwrap();
    let h = writer.handle();
    let page_bytes: [u8; PAGE_SIZE] = [0xCC; PAGE_SIZE];
    let payload = encode_index_page_payload(PageId::new(777), TenantId::DEFAULT, &page_bytes);
    h.append(WalRecordType::IndexPage, 1, 0, TenantId::DEFAULT, payload)
        .unwrap();
    writer.shutdown().unwrap();

    let invoked = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let flag = Arc::clone(&invoked);
    let txn_mgr = Arc::new(TxnManager::new());
    let primary: Arc<dyn PrimaryPageStoreHandle> = Arc::new(PrimaryPageStore::new());
    let target = PageStoreTarget::primary_only(primary).with_bootstrap(move |_| {
        flag.store(true, std::sync::atomic::Ordering::Release);
        Ok(())
    });
    let report = recover_from_wal(dir.path(), txn_mgr, target, None).unwrap();
    assert_eq!(report.metrics.orphan_pages_detected, 1);
    assert_eq!(report.metrics.bootstrap_from_mvcc_invoked, 1);
    assert!(invoked.load(std::sync::atomic::Ordering::Acquire));
}

// ─── M12 — orphan + bootstrap failure halts with error ──────────

#[test]
fn m12_replay_orphan_double_failure_halts_with_error() {
    use arcgraph_storage::primary_index::encode_index_page_payload;
    let dir = tempdir().unwrap();
    let writer = WalWriter::spawn(wal_cfg(dir.path())).unwrap();
    let h = writer.handle();
    let page_bytes: [u8; PAGE_SIZE] = [0xDD; PAGE_SIZE];
    let payload = encode_index_page_payload(PageId::new(888), TenantId::DEFAULT, &page_bytes);
    h.append(WalRecordType::IndexPage, 1, 0, TenantId::DEFAULT, payload)
        .unwrap();
    writer.shutdown().unwrap();

    let txn_mgr = Arc::new(TxnManager::new());
    let primary: Arc<dyn PrimaryPageStoreHandle> = Arc::new(PrimaryPageStore::new());
    let target = PageStoreTarget::primary_only(primary).with_bootstrap(|_| {
        Err(ArcGraphError::WalCorruption {
            lsn: Lsn::ZERO,
            reason: "simulated bootstrap failure".to_owned(),
        })
    });
    let err = recover_from_wal(dir.path(), txn_mgr, target, None).unwrap_err();
    match err {
        ArcGraphError::UnrecoverableOrphans {
            orphan_count,
            reason,
        } => {
            assert!(orphan_count >= 1);
            assert!(
                reason.contains("simulated bootstrap failure"),
                "got: {reason}"
            );
        }
        other => panic!("expected UnrecoverableOrphans, got {other:?}"),
    }
}

// ─── M13 partial — gap tolerance preserves chain semantics ──────

#[test]
fn m13_partial_replay_gap_tolerance_preserves_si() {
    // Synthesize commit_lsn sequence {1, 2, 4} — commit 3 was
    // torn-dropped on crash. Readers must see v1 at snapshot=1,
    // v2 at snapshot=2, same v2 at snapshot=3 (no new value), and
    // v4 at snapshot=4. §R7 chain semantics.
    let dir = tempdir().unwrap();
    for lsn in [1u64, 2, 4] {
        let mut mvcc = HashMap::new();
        mvcc.insert(0u64, Some(Bytes::from(format!("v{lsn}"))));
        write_bundle(
            dir.path(),
            Lsn::new(lsn),
            TenantId::DEFAULT,
            &mvcc,
            &[],
            &[],
        );
    }

    let (txn_mgr, _ps, target) = fresh_target();
    let report = recover_from_wal(dir.path(), Arc::clone(&txn_mgr), target, None).unwrap();
    assert_eq!(report.applied_commit_lsn, Lsn::new(4));
    assert_eq!(
        txn_mgr
            .read_at(TenantId::DEFAULT, 0, Lsn::new(1))
            .as_deref(),
        Some(&b"v1"[..])
    );
    assert_eq!(
        txn_mgr
            .read_at(TenantId::DEFAULT, 0, Lsn::new(2))
            .as_deref(),
        Some(&b"v2"[..])
    );
    // Gap at 3: v2 is still live (not yet expired by v3).
    assert_eq!(
        txn_mgr
            .read_at(TenantId::DEFAULT, 0, Lsn::new(3))
            .as_deref(),
        Some(&b"v2"[..])
    );
    assert_eq!(
        txn_mgr
            .read_at(TenantId::DEFAULT, 0, Lsn::new(4))
            .as_deref(),
        Some(&b"v4"[..])
    );
}

// ─── Property test — commit_lsn sort invariant ──────────────────

// ADR-032 §9.3 `crash_replay_idempotence` + commit_lsn ordering.
// Generate a random WAL with random commit_lsns + values; assert
// post-replay that readers see the highest-commit_lsn value per
// key at snapshot = max_commit_lsn. Torture-mode (10K cases) is
// deferred to the Slice 5 Jepsen-style follow-up session.
proptest! {
    #![proptest_config(ProptestConfig {
        cases: 64,
        .. ProptestConfig::default()
    })]

    #[test]
    fn replay_commit_lsn_sort_invariant(
        keys in prop::collection::vec(1u64..=20u64, 1..=30),
        lsn_offset in 0u64..100,
    ) {
        let dir = tempdir().unwrap();
        // Assign a unique commit_lsn to each op; shuffle them so
        // WAL order ≠ commit_lsn order.
        let mut ops: Vec<(Lsn, u64, Option<Bytes>)> = keys
            .iter()
            .enumerate()
            .map(|(i, k)| {
                let lsn = Lsn::new(1 + (i as u64) + lsn_offset);
                (lsn, *k, Some(Bytes::from(format!("{}:{}", *k, lsn.raw()))))
            })
            .collect();
        // Rotate to deterministically scramble order.
        let rot = ops.len() / 3;
        if ops.len() > 2 {
            ops.rotate_right(rot);
        }
        for (lsn, k, v) in &ops {
            let mut mvcc = HashMap::new();
            mvcc.insert(*k, v.clone());
            write_bundle(dir.path(), *lsn, TenantId::DEFAULT, &mvcc, &[], &[]);
        }

        let (txn_mgr, _ps, target) = fresh_target();
        let report = recover_from_wal(dir.path(), Arc::clone(&txn_mgr), target, None).unwrap();

        // applied_commit_lsn == max observed.
        let max_lsn = ops.iter().map(|(l, _, _)| l.raw()).max().unwrap();
        prop_assert_eq!(report.applied_commit_lsn.raw(), max_lsn);

        // For every key, readers see the highest-commit_lsn value
        // at snapshot = max_lsn.
        let mut expected: HashMap<u64, (u64, Bytes)> = HashMap::new();
        for (lsn, k, v) in &ops {
            let slot = expected.entry(*k).or_insert((0, Bytes::new()));
            if lsn.raw() > slot.0 {
                *slot = (lsn.raw(), v.clone().unwrap_or_default());
            }
        }
        for (k, (_, v)) in &expected {
            let got = txn_mgr.read_at(TenantId::DEFAULT, *k, Lsn::new(max_lsn));
            prop_assert_eq!(got.as_deref(), Some(v.as_ref()));
        }
    }
}
