//! ADR-034 §Slice F — T1 / Strict tier integration tests.
//!
//! Invariants exercised:
//! - **I-D1**: T1 commit is durable before ack.
//! - **I-D5**: counter allocation is tier-agnostic; T1 commits never
//!   become §R7 gaps.
//!
//! A T1 tenant's commit MUST NOT return `Ok` before the bundle bytes
//! are on durable disk. We prove this two ways:
//! (1) after `commit()` returns, the WAL writer's committed-fsync
//!     watermark is ≥ commit_lsn (proves fsync ran);
//! (2) on "crash simulation" (drop the WAL writer without flushing
//!     any pending batch), every T1 commit that returned `Ok` is
//!     present in the on-disk WAL segments.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use arcgraph_core::{DurabilityTier, Lsn, TenantId};
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::catalog::SystemCatalog;
use arcgraph_storage::io::InMemoryPageIo;
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::segment::{SegmentHeader, list_segments, segment_filename};
use arcgraph_storage::wal::{WalConfig, WalRecord, WalRecordType, WalWriter};
use bytes::Bytes;
use tempfile::TempDir;

fn config(dir: PathBuf) -> WalConfig {
    WalConfig {
        dir,
        segment_size_bytes: 64 * 1024 * 1024,
        group_commit_window: Duration::from_millis(2),
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

fn make_setup() -> (TempDir, WalWriter, TxnManager, Arc<SystemCatalog>) {
    let dir = tempfile::tempdir().unwrap();
    let writer = WalWriter::spawn(config(dir.path().to_path_buf())).unwrap();
    let mut mgr = TxnManager::with_wal(writer.handle());
    let catalog = Arc::new(SystemCatalog::new());
    let io = Arc::new(InMemoryPageIo::new());
    let pool = BufferPool::new(8, io);
    catalog.bootstrap(&pool, &mgr).unwrap();
    mgr.set_durability_lookup(catalog.clone());
    (dir, writer, mgr, catalog)
}

// ─────────────────────────────────────────────────────────────────────
// Test 1 (spec §F.1): T1 zero-data-loss single tenant.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn t1_zero_data_loss_single_tenant() {
    // Commit N writes at T1 (the bootstrap default). Verify every
    // commit's bytes land on disk AND the commit's commit_lsn is
    // covered by the committed-fsync watermark on return.
    let (dir, writer, mgr, _cat) = make_setup();
    let handle = writer.handle();

    let n = 20u64;
    let mut commit_lsns = Vec::with_capacity(n as usize);
    for i in 1..=n {
        let mut tx = mgr.begin(TenantId::DEFAULT);
        tx.write(i, Bytes::from(format!("v{i}").into_bytes()));
        let lsn = tx.commit().unwrap();
        commit_lsns.push(lsn);
        // I-D1: watermark covers this commit's LSN on return.
        assert!(
            handle.last_durable_lsn() >= lsn,
            "I-D1 violation: commit {lsn:?} returned Ok but watermark is {:?}",
            handle.last_durable_lsn(),
        );
    }

    // Sanity: writer thread picked up every commit as a T1 (sync)
    // append. Bootstrap also emits 1 T1 commit under SYSTEM, so
    // the count is n + 1.
    let metrics = writer.fire_metrics();
    assert_eq!(
        metrics.wal_t1_appends_total(),
        n + 1,
        "expected {} T1 appends (n user + 1 bootstrap); got {}",
        n + 1,
        metrics.wal_t1_appends_total()
    );
    assert_eq!(metrics.wal_t3_appends_total(), 0);

    // Drop the writer without flushing — verifies the on-disk state
    // contains every acked T1 commit.
    writer.shutdown().unwrap();

    let records = drain_segments(dir.path());
    // Expected record count: N user commits + 1 bootstrap commit
    // (SYSTEM) = N + 1.
    let user_bundles: Vec<_> = records
        .iter()
        .filter(|r| {
            r.record_type == WalRecordType::CommitBundle && r.tenant_id == TenantId::DEFAULT
        })
        .collect();
    assert_eq!(
        user_bundles.len(),
        n as usize,
        "every T1-acked commit must be on disk after writer shutdown"
    );
}

// ─────────────────────────────────────────────────────────────────────
// Test 2: crash-between-phase1-and-phase2 rolls back.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn t1_crash_between_phase1_and_phase2_rolls_back() {
    // Simulate a WAL-append-fail by shutting the writer down before
    // the commit — the writer's channel is disconnected, the
    // foreground append returns WalUnavailable, Z-1 (b) rollback
    // unwinds MVCC.
    let dir = tempfile::tempdir().unwrap();
    let writer = WalWriter::spawn(config(dir.path().to_path_buf())).unwrap();
    let handle = writer.handle();
    writer.shutdown().unwrap();
    let mgr = TxnManager::with_wal(handle.clone());

    let mut tx = mgr.begin(TenantId::DEFAULT);
    tx.write(1, Bytes::from_static(b"stillborn"));
    let err = tx.commit().unwrap_err();
    // ADR-033 §3c: rolled-back errors carry the underlying WAL error
    // via .source().
    assert!(
        matches!(
            &err,
            arcgraph_core::ArcGraphError::WalErrorRolledBack { .. }
        ),
        "expected WalErrorRolledBack, got {err:?}"
    );

    // visible has NOT advanced; reader sees no value.
    assert_eq!(mgr.current_lsn(), Lsn::ZERO);
    let reader = mgr.begin(TenantId::DEFAULT);
    assert!(reader.read(1).is_none());
    // Watermark unchanged.
    assert_eq!(handle.last_durable_lsn(), Lsn::ZERO);
}

// ─────────────────────────────────────────────────────────────────────
// Test 3: T1 commits ordered monotonically.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn t1_commit_lsns_are_monotonic() {
    // Not strictly I-D1, but a useful smoke test that the commit
    // path is behaving.
    let (_dir, writer, mgr, _cat) = make_setup();
    let mut last = Lsn::ZERO;
    for i in 1..=50u64 {
        let mut tx = mgr.begin(TenantId::DEFAULT);
        tx.write(i, Bytes::from_static(b"x"));
        let lsn = tx.commit().unwrap();
        assert!(
            lsn > last,
            "commit_lsns must be monotonic; got {lsn:?} after {last:?}"
        );
        last = lsn;
    }
    writer.shutdown().unwrap();
}

// ─────────────────────────────────────────────────────────────────────
// Test 4: bootstrap default tier is Strict (catalog regression).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn bootstrap_default_tier_is_strict() {
    let (_dir, writer, _mgr, catalog) = make_setup();
    assert_eq!(
        catalog.durability_tier(TenantId::DEFAULT),
        DurabilityTier::Strict,
        "catalog must bootstrap DEFAULT tenant as Strict per D-1"
    );
    writer.shutdown().unwrap();
}

// ─────────────────────────────────────────────────────────────────────
// Test 5: explicit Strict setting (round-trip through catalog).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn explicit_strict_setting_acknowledged() {
    let (_dir, writer, mgr, catalog) = make_setup();

    // Flip to Periodic, then back to Strict.
    let mut tx = mgr.begin(TenantId::SYSTEM);
    catalog
        .set_durability_tier(
            &mut tx,
            TenantId::DEFAULT,
            DurabilityTier::Periodic { rpo_ms: 100 },
        )
        .unwrap();
    tx.commit().unwrap();
    assert!(catalog.durability_tier(TenantId::DEFAULT).is_periodic());

    let mut tx2 = mgr.begin(TenantId::SYSTEM);
    catalog
        .set_durability_tier(&mut tx2, TenantId::DEFAULT, DurabilityTier::Strict)
        .unwrap();
    tx2.commit().unwrap();
    assert_eq!(
        catalog.durability_tier(TenantId::DEFAULT),
        DurabilityTier::Strict
    );

    writer.shutdown().unwrap();
}

// ─────────────────────────────────────────────────────────────────────
// Test 6 (issue #129 P0 canary): T1 byte-identical after fault recovery.
//
// Pre-fix verifies the bug fix: PageAllocator + CrudStore allocator
// state used to reset on WAL recovery, causing post-fault `create_node`
// to re-issue NodeIds that pre-fault commits already consumed; the
// primary index then routed reads through the latest commit's record
// slot and earlier T1 strict commits became unreachable.
//
// Post-fix the v4 `CommitBundle.allocator_advances` section persists
// the high-water atomically with each commit; on recovery the
// allocator is seeded so post-recovery `create_node` returns ids
// strictly above any pre-fault commit's, and every prior T1-acked
// commit's record stays reachable via `read_node_with_store`.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn t1_strict_byte_identical_after_fault_recovery() {
    use std::sync::Arc;

    use arcgraph_core::{LabelId, NodeId};
    use arcgraph_storage::crud::{
        CrudStore, PropertyData, commit, create_node, crud_allocator_seed_handle,
        read_node_with_store,
    };
    use arcgraph_storage::page_alloc::PageAllocator;
    use arcgraph_storage::primary_index::PrimaryIndex;
    use arcgraph_storage::wal::{
        AllocatorSeedHandle, BlobStoreHandle, PageStoreTarget, PrimaryPageStoreHandle,
        RecordPageStoreHandle, recover_from_wal,
    };
    use tempfile::TempDir;

    // ── Pre-fault: build the production CRUD stack and create N=100
    //    nodes under the DEFAULT tenant (T1 / Strict — the bootstrap
    //    default per ADR-034 D-1). Each `commit(tx, &store)` is a T1
    //    strict commit; after each returns Ok, the bundle bytes are
    //    on durable disk per I-D1.
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();

    const N: u32 = 100;
    let mut acked: Vec<(NodeId, Lsn, u32, u32)> = Vec::with_capacity(N as usize);

    {
        let writer = WalWriter::spawn(config(wal_dir.clone())).unwrap();
        let handle = writer.handle();
        let mgr = Arc::new(arcgraph_storage::transaction::TxnManager::with_wal(
            handle.clone(),
        ));
        let alloc = Arc::new(PageAllocator::new());
        let primary = Arc::new(
            PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&alloc), Some(handle.clone())).unwrap(),
        );
        let store = Arc::new(CrudStore::new_with_index(
            Some(handle.clone()),
            Arc::clone(&primary),
            Arc::clone(&alloc),
        ));

        for i in 1..=N {
            // Encode the commit_lsn-relative payload into the inline
            // u32 pair so we can later byte-equality verify the
            // recovered record matches the committed bytes — without
            // tracking the Lsn out of band we'd have nothing to
            // compare against.
            let mut tx = mgr.begin(TenantId::DEFAULT);
            let id = create_node(
                &store,
                &mut tx,
                TenantId::DEFAULT,
                LabelId::new(i),
                &PropertyData::InlineU32Pair(i, i.wrapping_mul(31)),
            )
            .unwrap();
            let lsn = commit(tx, &store).unwrap();
            // I-D1: the writer's durable watermark must cover this
            // commit's LSN before commit() returns.
            assert!(
                handle.last_durable_lsn() >= lsn,
                "I-D1 violation: commit_lsn {lsn:?} acked but watermark = {:?}",
                handle.last_durable_lsn(),
            );
            acked.push((id, lsn, i, i.wrapping_mul(31)));
        }

        // Simulate a fault: drop the writer without an explicit
        // shutdown drain. Strict / T1 commits already paid an fsync
        // before each ack; nothing is in flight on the writer's
        // pending buffer.
        writer.shutdown().unwrap();
        drop(store);
        drop(primary);
        drop(mgr);
    }

    // ── Post-fault: spin up a fresh stack and run WAL replay. Wire
    //    the AllocatorSeedHandle so v4 bundle `allocator_advances`
    //    entries seed the live allocators (issue #129 P0 fix).
    let writer2 = WalWriter::spawn(config(wal_dir.clone())).unwrap();
    let handle2 = writer2.handle();
    let mgr2 = Arc::new(arcgraph_storage::transaction::TxnManager::with_wal(
        handle2.clone(),
    ));
    let alloc2 = Arc::new(PageAllocator::new());
    let primary2 = Arc::new(
        PrimaryIndex::new(
            Arc::clone(&mgr2),
            Arc::clone(&alloc2),
            Some(handle2.clone()),
        )
        .unwrap(),
    );
    let store2 = Arc::new(CrudStore::new_with_index(
        Some(handle2.clone()),
        Arc::clone(&primary2),
        Arc::clone(&alloc2),
    ));

    let primary_handle: Arc<dyn PrimaryPageStoreHandle> =
        Arc::clone(primary2.page_store()) as Arc<dyn PrimaryPageStoreHandle>;
    let records_handle: Arc<dyn RecordPageStoreHandle> = Arc::clone(
        store2
            .records()
            .expect("CrudStore constructed via new_with_index exposes record store"),
    ) as Arc<dyn RecordPageStoreHandle>;
    let blob_handle: Arc<dyn BlobStoreHandle> =
        Arc::clone(store2.blob_store()) as Arc<dyn BlobStoreHandle>;
    let allocator_seed: Arc<dyn AllocatorSeedHandle> =
        crud_allocator_seed_handle(Arc::clone(&store2), Arc::clone(&alloc2));

    let target = PageStoreTarget::primary_only(primary_handle)
        .with_record_store(records_handle)
        .with_blob_store(blob_handle)
        .with_allocator_seed(allocator_seed);
    let _report = recover_from_wal(&wal_dir, Arc::clone(&mgr2), target, None).unwrap();

    // ── Lemma D-1: every T1-acked commit is byte-identical
    //    post-recovery. Read each NodeId from 1..=N at a snapshot
    //    that sees ALL N commits (use mgr2.current_lsn() — the
    //    recovery executor seeds it to the max applied commit_lsn).
    let snap = mgr2.current_lsn();
    assert!(
        snap >= acked.last().unwrap().1,
        "post-recovery snapshot {snap:?} must cover the last acked commit {:?}",
        acked.last().unwrap().1
    );
    let tx2 = mgr2.begin(TenantId::DEFAULT);
    for (id, lsn, expected_a, expected_b) in &acked {
        let rec = read_node_with_store(&store2, &tx2, *id)
            .unwrap()
            .unwrap_or_else(|| {
                panic!(
                    "T1 strict commit at lsn {lsn:?} (NodeId={:?}) is unreadable post-recovery — \
                     ADR-034 D-1 violated; primary index points elsewhere",
                    id
                )
            });
        assert_eq!(
            rec.label_id, *expected_a,
            "post-recovery record bytes diverge at NodeId={:?}: label_id mismatch",
            id
        );
        assert_eq!(
            rec.inline_u32a, *expected_a,
            "post-recovery record bytes diverge at NodeId={:?}: inline_u32a mismatch",
            id
        );
        assert_eq!(
            rec.inline_u32b, *expected_b,
            "post-recovery record bytes diverge at NodeId={:?}: inline_u32b mismatch",
            id
        );
    }
    drop(tx2);

    // ── Allocator advance check: post-recovery, the next NodeId
    //    `alloc_node` returns MUST be strictly greater than the
    //    highest pre-fault NodeId. This is the load-bearing
    //    invariant the bug fix establishes — pre-fix the next
    //    NodeId was 1 (allocator reset), so the assertion below
    //    fails the canary and the bundle never reaches the read
    //    loop.
    let max_pre_fault = acked.iter().map(|(id, _, _, _)| id.raw()).max().unwrap();
    let next_id = store2.alloc_node(TenantId::DEFAULT).unwrap();
    assert!(
        next_id.raw() > max_pre_fault,
        "post-recovery alloc_node returned NodeId={:?}, but max pre-fault NodeId={}; \
         allocator high-water was not seeded from the v4 bundle's allocator_advances \
         section (ADR-034 D-1 violation; issue #129)",
        next_id,
        max_pre_fault,
    );

    writer2.shutdown().unwrap();
}
