//! SVC-1 P2 / #1365 / ADR-229 §Segment reclamation — WAL segment reclamation
//! + bounded-WAL integration tests.
//!
//! P1 (#1371) bounded restart-recovery TIME (anchored replay). P2 bounds WAL
//! SIZE: it DELETES segments whose committed effects are already captured in
//! the checkpoint's full-state snapshot. The #1365 rc-blocker is the concrete
//! 10M failure — a 167 GB WAL that filled the disk and could not restart.
//!
//! THE data-loss risk (ADR-229 §Consequences): reclaiming a segment whose
//! `commit_lsn > checkpoint_lsn` silently loses that commit on the next
//! restart (the anchored replay skips at/below the frontier, and the segment
//! is gone). Every test here is RED-on-revert against the exact guard it
//! protects:
//!
//! - `reclamation_bounds_wal_and_preserves_all_committed_data` (the headline
//!   #1365 oracle): checkpoint during ingest → segments below the frontier
//!   DELETED (WAL bounded) → anchored restart recovers ALL committed data.
//!   RED-on-revert = never-reclaim → WAL unbounded; reclaim-above-frontier →
//!   data loss.
//! - `reclamation_never_deletes_at_or_above_frontier_no_dataloss` (boundary):
//!   a segment holding a commit ABOVE the frontier is kept, and the anchored
//!   restart recovers it. RED-on-revert = weaken the `> frontier` guard →
//!   the above-frontier commit is LOST.
//! - `crash_mid_reclamation_recovers_no_dataloss` (crash-injection): a crash
//!   between deleting segment K and segment K+1 (partial reclamation) leaves a
//!   VALID checkpoint + a contiguous WAL suffix → anchored restart recovers
//!   every committed value.
//! - `bounded_recovery_after_reclamation` (the availability fix): after
//!   reclamation the anchored replay reads only the post-frontier tail, and
//!   segment count is bounded — the property that makes a 10M-scale restart
//!   fast.

use std::collections::HashMap;
use std::sync::Arc;

use arcgraph_core::{Lsn, PAGE_SIZE, PageId, TenantId};
use arcgraph_storage::BlobStoreHandle;
use arcgraph_storage::blob::BlobStore;
use arcgraph_storage::crud::{CrudStore, crud_allocator_seed_handle};
use arcgraph_storage::idempotency::IdempotencyStore;
use arcgraph_storage::intern::InternTable;
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::permissions::PermissionIndex;
use arcgraph_storage::primary_index::PrimaryPageStore;
use arcgraph_storage::record_store::RecordPageStore;
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{
    AllocatorAdvance, AllocatorSeedHandle, BundlePageKind, PageStoreTarget, PrimaryPageStoreHandle,
    RecordPageStoreHandle, StopReason, WalConfig, WalRecordType, WalWriter,
    encode_commit_bundle_v8, list_segments, reclaim_segments_below, recover_from_wal_encrypted,
    recover_from_wal_encrypted_anchored, segment_count, segment_filename,
};
use bytes::Bytes;
use tempfile::tempdir;

// ─── owners bundle (mirrors the durable-bootstrap replay target) ──

struct Owners {
    txn: Arc<TxnManager>,
    primary: Arc<PrimaryPageStore>,
    record: Arc<RecordPageStore>,
    blob: Arc<BlobStore>,
    allocator: Arc<PageAllocator>,
    crud: Arc<CrudStore>,
    intern: Arc<InternTable>,
    idempotency: Arc<IdempotencyStore>,
    permissions: Arc<PermissionIndex>,
}

impl Owners {
    fn fresh() -> Self {
        let allocator = Arc::new(PageAllocator::new());
        let record = Arc::new(RecordPageStore::new());
        let blob = Arc::new(BlobStore::new());
        let crud = Arc::new(CrudStore::new_with_existing_page_stores(
            None,
            None,
            Arc::clone(&allocator),
            Arc::clone(&record),
            Arc::clone(&blob),
        ));
        Self {
            txn: Arc::new(TxnManager::new()),
            primary: Arc::new(PrimaryPageStore::new()),
            record,
            blob,
            allocator,
            crud,
            intern: Arc::new(InternTable::new()),
            idempotency: Arc::new(IdempotencyStore::new()),
            permissions: Arc::new(PermissionIndex::new()),
        }
    }

    fn allocator_seed(&self) -> Arc<dyn AllocatorSeedHandle> {
        crud_allocator_seed_handle(Arc::clone(&self.crud), Arc::clone(&self.allocator))
    }

    fn target(&self) -> PageStoreTarget {
        let primary: Arc<dyn PrimaryPageStoreHandle> =
            Arc::clone(&self.primary) as Arc<dyn PrimaryPageStoreHandle>;
        let record: Arc<dyn RecordPageStoreHandle> =
            Arc::clone(&self.record) as Arc<dyn RecordPageStoreHandle>;
        let blob: Arc<dyn BlobStoreHandle> = Arc::clone(&self.blob) as Arc<dyn BlobStoreHandle>;
        PageStoreTarget::primary_only(primary)
            .with_record_store(record)
            .with_blob_store(blob)
            .with_allocator_seed(self.allocator_seed())
            .with_intern_table(Arc::clone(&self.intern))
            .with_idempotency_store(Arc::clone(&self.idempotency))
            .with_permission_index(Arc::clone(&self.permissions))
    }

    fn snapshot<'a>(
        &'a self,
        seed: &'a dyn AllocatorSeedHandle,
    ) -> arcgraph_storage::checkpoint::CheckpointSnapshot<'a> {
        arcgraph_storage::checkpoint::CheckpointSnapshot {
            txn: &self.txn,
            primary_pages: &self.primary,
            record_pages: &self.record,
            blob: &self.blob,
            allocator_seed: seed,
            intern: &self.intern,
            idempotency: &self.idempotency,
            permissions: &self.permissions,
            permissions_tenant: TenantId::DEFAULT,
        }
    }

    fn advances(&self) -> Vec<AllocatorAdvance> {
        let mut a = self.allocator.snapshot_advances();
        a.extend(self.crud.snapshot_allocator_advances());
        a
    }

    /// Establish a REAL checkpoint at `frontier` via the producer
    /// (`checkpoint::checkpoint`): freeze + full-state snapshot + sidecar.
    /// Uses an in-memory buffer pool (catalog flush is a no-op here).
    fn establish_checkpoint(&self, data_dir: &std::path::Path, frontier: Lsn) {
        let seed = self.allocator_seed();
        let advances = self.advances();
        let pool =
            arcgraph_storage::buffer::BufferPool::new(16, Arc::new(InMemoryPageIoShim::new()));
        arcgraph_storage::checkpoint::checkpoint(
            data_dir,
            &pool,
            &self.snapshot(seed.as_ref()),
            || advances.clone(),
            frontier,
        )
        .unwrap();
    }
}

// A tiny in-memory PageIo shim so the producer's catalog flush is a no-op.
use arcgraph_storage::io::InMemoryPageIo as InMemoryPageIoShim;

// ─── WAL fixture helpers ──────────────────────────────────────────

/// WAL config with a tiny segment so each bundle (carrying a full page)
/// lands in its own segment — lets us exercise multi-segment reclamation
/// deterministically.
fn wal_cfg_tiny(dir: &std::path::Path) -> WalConfig {
    WalConfig {
        dir: dir.to_path_buf(),
        segment_size_bytes: 64,
        group_commit_window: std::time::Duration::from_millis(2),
        group_commit_max_batch: 1,
        metrics_sink: None,
        encryption: None,
        inflight_budget_bytes: None,
    }
}

fn mk_page(fill: u8) -> Box<[u8; PAGE_SIZE]> {
    Box::new([fill; PAGE_SIZE])
}

/// Append one v8 CommitBundle at `commit_lsn` carrying an MVCC write at key
/// `commit_lsn` (value `b"v{lsn}"`) + one primary page image.
fn append_bundle(handle: &arcgraph_storage::wal::WalHandle, commit_lsn: u64) {
    let mut mvcc: HashMap<u64, Option<Bytes>> = HashMap::new();
    mvcc.insert(commit_lsn, Some(Bytes::from(format!("v{commit_lsn}"))));
    let staged: Vec<(BundlePageKind, PageId, TenantId, Box<[u8; PAGE_SIZE]>)> = vec![(
        BundlePageKind::PrimaryIndex,
        PageId::new(1000 + commit_lsn),
        TenantId::DEFAULT,
        mk_page((commit_lsn % 256) as u8),
    )];
    let payload = encode_commit_bundle_v8(
        Lsn::new(commit_lsn),
        TenantId::DEFAULT,
        &mvcc,
        &[],
        &staged,
        &[],
        &[],
        &[],
        &[],
    );
    handle
        .append(
            WalRecordType::CommitBundle,
            1,
            0,
            TenantId::DEFAULT,
            payload,
        )
        .unwrap();
}

/// Write bundles commit_lsn 1..=n into `dir`, one per segment (tiny segment
/// size). Returns nothing; the WAL is on disk after shutdown.
fn write_bundles_tiny_segments(dir: &std::path::Path, n: u64) {
    let writer = WalWriter::spawn(wal_cfg_tiny(dir)).unwrap();
    let handle = writer.handle();
    for lsn in 1..=n {
        append_bundle(&handle, lsn);
    }
    writer.shutdown().unwrap();
}

fn recover_full(dir: &std::path::Path, owners: &Owners) {
    recover_from_wal_encrypted(dir, Arc::clone(&owners.txn), owners.target(), None, None).unwrap();
}

fn read(owners: &Owners, key: u64) -> Option<Bytes> {
    let snap = owners.txn.current_lsn();
    owners.txn.read_at(TenantId::DEFAULT, key, snap)
}

// ─────────────────────────────────────────────────────────────────
// (a) THE #1365 headline oracle — reclamation bounds the WAL and
//     recovers EVERY committed value.
// ─────────────────────────────────────────────────────────────────

/// Checkpoint during ingest → WAL segments below the frontier DELETED (WAL
/// bounded) → an anchored restart recovers ALL N+M committed values.
///
/// RED-on-revert (bound): comment out `reclaim_segments_below` → segment
/// count stays at N+M+ (WAL unbounded) → the `segments_after < segments_before`
/// assert FAILS.
///
/// RED-on-revert (no data loss): if reclamation deleted a segment above the
/// frontier, the post-restart read of a key > frontier would return `None`
/// → the recover-everything assert FAILS.
#[test]
fn reclamation_bounds_wal_and_preserves_all_committed_data() {
    const N: u64 = 20; // committed below the checkpoint frontier
    const M: u64 = 5; // committed after the checkpoint (must survive in WAL)
    let dir = tempdir().unwrap();
    let wal = dir.path();

    // Ingest N commits (one segment each), then populate p1 + establish a
    // real checkpoint at frontier N.
    write_bundles_tiny_segments(wal, N);
    let p1 = Owners::fresh();
    recover_full(wal, &p1);
    for i in 1..=N {
        assert_eq!(
            read(&p1, i),
            Some(Bytes::from(format!("v{i}"))),
            "pre N {i}"
        );
    }
    let frontier = Lsn::new(N);
    p1.establish_checkpoint(wal, frontier);

    // M more commits AFTER the checkpoint (these live only in the WAL tail).
    let writer = WalWriter::spawn(wal_cfg_tiny(wal)).unwrap();
    let handle = writer.handle();
    for i in (N + 1)..=(N + M) {
        append_bundle(&handle, i);
    }
    writer.shutdown().unwrap();

    let segments_before = segment_count(wal).unwrap();

    // Reclaim segments fully below the frontier (THE P2 step).
    let report = reclaim_segments_below(wal, frontier).unwrap();
    assert!(
        !report.deleted_segments.is_empty(),
        "some segments below frontier {N} must be reclaimed: {report:?}",
    );

    // THE BOUND: the WAL shrank — segments below the frontier are gone.
    let segments_after = segment_count(wal).unwrap();
    assert!(
        segments_after < segments_before,
        "WAL must shrink after reclamation: before={segments_before} after={segments_after}",
    );

    // No data loss: an anchored restart recovers ALL N+M committed values.
    // The below-frontier N come from the checkpoint snapshot; the M
    // post-frontier come from the surviving WAL tail.
    let p2 = Owners::fresh();
    let restore = {
        let seed2 = p2.allocator_seed();
        arcgraph_storage::checkpoint::restore_latest_checkpoint(wal, &p2.snapshot(seed2.as_ref()))
            .unwrap()
            .expect("a full-state checkpoint must be found")
    };
    assert_eq!(restore.checkpoint_lsn, frontier);
    recover_from_wal_encrypted_anchored(
        wal,
        Arc::clone(&p2.txn),
        p2.target(),
        None,
        None,
        restore.checkpoint_lsn,
    )
    .unwrap();
    for i in 1..=(N + M) {
        assert_eq!(
            read(&p2, i),
            Some(Bytes::from(format!("v{i}"))),
            "post-reclamation restart: committed key {i} must survive (data loss if None)",
        );
    }
}

// ─────────────────────────────────────────────────────────────────
// (b/c boundary) — never delete a segment at/above the frontier, and the
//     anchored restart recovers the above-frontier commits.
// ─────────────────────────────────────────────────────────────────

/// A segment holding a commit ABOVE the checkpoint frontier must NEVER be
/// reclaimed, and an anchored restart must recover that commit. This is the
/// end-to-end (through-recovery) version of the unit boundary test.
///
/// CRITICAL test design: the checkpoint snapshot must capture ONLY the
/// ≤ FRONTIER state, so the above-frontier commits live ONLY in the WAL.
/// We therefore write 1..=FRONTIER, checkpoint (snapshot = exactly those),
/// THEN append FRONTIER+1..=TOTAL to the WAL tail. If reclamation wrongly
/// deletes an above-frontier segment, those commits are gone from BOTH the
/// snapshot AND the WAL → the anchored restart returns `None` for them.
///
/// RED-on-revert: weaken the reclaim `max > frontier` guard (so it deletes
/// at/above the frontier) → keys FRONTIER+1..=TOTAL are LOST on the anchored
/// restart → the final read assert FAILS with `None`. (Verified: with the
/// guard forced false, this test panics "above-frontier commit 7 must
/// survive".)
#[test]
fn reclamation_never_deletes_at_or_above_frontier_no_dataloss() {
    const FRONTIER: u64 = 6; // checkpoint at 6
    const TOTAL: u64 = 12; // commits 7..=12 are appended AFTER the checkpoint
    let dir = tempdir().unwrap();
    let wal = dir.path();

    // Phase 1: write ONLY 1..=FRONTIER, populate p_front, checkpoint at
    // FRONTIER — the snapshot captures exactly keys 1..=FRONTIER (nothing
    // above), so an above-frontier commit will exist ONLY in the WAL.
    write_bundles_tiny_segments(wal, FRONTIER);
    let p_front = Owners::fresh();
    recover_full(wal, &p_front);
    let frontier = Lsn::new(FRONTIER);
    p_front.establish_checkpoint(wal, frontier);

    // Phase 2: append FRONTIER+1..=TOTAL to the WAL tail (these are NOT in
    // the snapshot — the ONLY durable copy is their WAL segments).
    let writer = WalWriter::spawn(wal_cfg_tiny(wal)).unwrap();
    let handle = writer.handle();
    for i in (FRONTIER + 1)..=TOTAL {
        append_bundle(&handle, i);
    }
    writer.shutdown().unwrap();

    // Reclaim below the frontier. The above-frontier segments MUST be kept.
    let report = reclaim_segments_below(wal, frontier).unwrap();

    // Prove it end-to-end: an anchored restart recovers ALL commits,
    // including 7..=12 which live ONLY in the (kept) above-frontier segments.
    let p2 = Owners::fresh();
    let restore = {
        let seed2 = p2.allocator_seed();
        arcgraph_storage::checkpoint::restore_latest_checkpoint(wal, &p2.snapshot(seed2.as_ref()))
            .unwrap()
            .expect("checkpoint found")
    };
    assert_eq!(restore.checkpoint_lsn, frontier);
    recover_from_wal_encrypted_anchored(
        wal,
        Arc::clone(&p2.txn),
        p2.target(),
        None,
        None,
        restore.checkpoint_lsn,
    )
    .unwrap();
    for i in (FRONTIER + 1)..=TOTAL {
        assert_eq!(
            read(&p2, i),
            Some(Bytes::from(format!("v{i}"))),
            "above-frontier commit {i} must survive reclamation (report {report:?})",
        );
    }
    // And the below-frontier commits survive too (from the snapshot).
    for i in 1..=FRONTIER {
        assert_eq!(
            read(&p2, i),
            Some(Bytes::from(format!("v{i}"))),
            "below {i}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────
// (b) crash-injection — a crash mid-reclamation loses no committed data.
// ─────────────────────────────────────────────────────────────────

/// Simulate a crash BETWEEN deleting two segments (partial reclamation): the
/// checkpoint is already durable, and the WAL suffix (from the first
/// surviving segment up) is contiguous. An anchored restart recovers EVERY
/// committed value.
///
/// We reproduce the partial-delete by reclaiming, then manually deleting one
/// MORE below-frontier segment "by hand" is not needed — instead we model
/// the crash as: reclamation ran and deleted a prefix, then the process died
/// before it could delete the rest. Recovery over the remaining segments +
/// the durable checkpoint must lose nothing.
#[test]
fn crash_mid_reclamation_recovers_no_dataloss() {
    const N: u64 = 16;
    const M: u64 = 4;
    let dir = tempdir().unwrap();
    let wal = dir.path();

    write_bundles_tiny_segments(wal, N);
    let p1 = Owners::fresh();
    recover_full(wal, &p1);
    let frontier = Lsn::new(N);
    p1.establish_checkpoint(wal, frontier);

    // M post-checkpoint commits in the WAL tail.
    let writer = WalWriter::spawn(wal_cfg_tiny(wal)).unwrap();
    let handle = writer.handle();
    for i in (N + 1)..=(N + M) {
        append_bundle(&handle, i);
    }
    writer.shutdown().unwrap();

    // "Crash mid-reclamation": manually delete only the FIRST reclaimable
    // segment (segment 0), leaving the rest of the below-frontier prefix
    // still on disk — exactly the state after a crash between the first
    // unlink and the second.
    let segs = list_segments(wal).unwrap();
    let low = segs[0];
    std::fs::remove_file(wal.join(segment_filename(low))).unwrap();
    // (No dir-fsync — modeling the crash. The checkpoint is already durable.)

    // Anchored restart: the checkpoint restores below-frontier state; the
    // WAL suffix (segments 1..) replays the rest. NO committed value lost.
    let p2 = Owners::fresh();
    let restore = {
        let seed2 = p2.allocator_seed();
        arcgraph_storage::checkpoint::restore_latest_checkpoint(wal, &p2.snapshot(seed2.as_ref()))
            .unwrap()
            .expect("checkpoint durable across the crash")
    };
    assert_eq!(restore.checkpoint_lsn, frontier);
    recover_from_wal_encrypted_anchored(
        wal,
        Arc::clone(&p2.txn),
        p2.target(),
        None,
        None,
        restore.checkpoint_lsn,
    )
    .unwrap();
    for i in 1..=(N + M) {
        assert_eq!(
            read(&p2, i),
            Some(Bytes::from(format!("v{i}"))),
            "crash-mid-reclamation: committed key {i} must survive",
        );
    }
}

// ─────────────────────────────────────────────────────────────────
// (d) bounded recovery — after reclamation the anchored replay reads only
//     the post-frontier tail, and the WAL is bounded.
// ─────────────────────────────────────────────────────────────────

/// After reclamation, the segment count is bounded (≈ post-frontier tail plus
/// the segment straddling the frontier) AND the anchored replay applies ONLY
/// the post-frontier bundles. This is the availability property:
/// restart-recovery is O(WAL-since-checkpoint), and the disk footprint is
/// bounded regardless of total history.
#[test]
fn bounded_recovery_after_reclamation() {
    const N: u64 = 40; // large below-frontier history
    const M: u64 = 3; // small post-frontier tail
    let dir = tempdir().unwrap();
    let wal = dir.path();

    write_bundles_tiny_segments(wal, N);
    let p1 = Owners::fresh();
    recover_full(wal, &p1);
    let frontier = Lsn::new(N);
    p1.establish_checkpoint(wal, frontier);

    let writer = WalWriter::spawn(wal_cfg_tiny(wal)).unwrap();
    let handle = writer.handle();
    for i in (N + 1)..=(N + M) {
        append_bundle(&handle, i);
    }
    writer.shutdown().unwrap();

    let before = segment_count(wal).unwrap();
    let report = reclaim_segments_below(wal, frontier).unwrap();
    let after = segment_count(wal).unwrap();

    // The reclaimed prefix is the vast majority of the N below-frontier
    // segments; the remaining count is bounded by the tail (M) plus the
    // straddling/active segment — NOT O(N).
    assert!(
        after < before,
        "reclamation must shrink the WAL: before {before} after {after}",
    );
    assert!(
        after as u64 <= M + 3,
        "post-reclamation WAL must be bounded by the post-frontier tail (~{M}+active), got {after} \
         segments (report {report:?})",
    );
    // The reason it stopped is either reaching the active segment or the
    // first above-frontier segment — never a data-loss stop.
    assert!(
        matches!(
            report.stop_reason,
            StopReason::ReachedActiveSegment | StopReason::AboveFrontier
        ),
        "must stop safely, got {:?}",
        report.stop_reason,
    );

    // And the anchored recovery replays only the M post-frontier bundles.
    let p2 = Owners::fresh();
    let restore = {
        let seed2 = p2.allocator_seed();
        arcgraph_storage::checkpoint::restore_latest_checkpoint(wal, &p2.snapshot(seed2.as_ref()))
            .unwrap()
            .expect("checkpoint found")
    };
    let rep = recover_from_wal_encrypted_anchored(
        wal,
        Arc::clone(&p2.txn),
        p2.target(),
        None,
        None,
        restore.checkpoint_lsn,
    )
    .unwrap();
    assert_eq!(
        rep.metrics.bundles_applied, M,
        "anchored replay must apply ONLY the {M} post-frontier bundles, got {}",
        rep.metrics.bundles_applied,
    );
    for i in 1..=(N + M) {
        assert_eq!(read(&p2, i), Some(Bytes::from(format!("v{i}"))), "key {i}");
    }
}
