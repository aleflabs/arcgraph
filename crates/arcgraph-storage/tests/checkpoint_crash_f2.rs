//! F-2 crash injection DURING a checkpoint snapshot write — clean recovery,
//! no torn/acked-lost data (rc-readiness gap, #1425 review; ADR-229
//! crash-atomicity).
//!
//! ## What this suite verifies (per the F-2 charter)
//!
//! Crash injection at four points during the checkpoint snapshot write:
//!
//! 1. **Mid-body** — partial temp (body sections truncated), no rename,
//!    no sidecar updated.
//! 2. **Mid-supplement** — body complete, supplement section truncated,
//!    no rename, no sidecar updated.
//! 3. **Pre-rename** — temp fully written (body + supplement + footer/CRC),
//!    NOT renamed to `CHECKPOINT.snap`, no sidecar updated.
//! 4. **Post-publish, pre-WAL-reclaim** — complete checkpoint (snapshot
//!    + sidecar both durable), crash before WAL segment reclamation.
//!
//! For each crash point the suite asserts:
//!
//! - **Recovery completes cleanly**: falls back to WAL / prior checkpoint per
//!   ADR-229 atomicity; orphaned/partial temp artifacts are ignored.
//! - **No acked-lost**: every ACKED commit (appended to WAL + `Ok` returned)
//!   is readable post-recovery. Verified deterministically — every commit
//!   carries a unique key and value, the post-recovery read checks all of them.
//! - **No torn/corrupt state**: no partial/inconsistent records are served;
//!   a corrupt snapshot falls back to from-zero WAL replay, owners remain
//!   pristine until WAL replay populates them correctly.
//!
//! A fifth scenario covers **post-snapshot-rename, pre-sidecar** (the
//! crash window between ADR-229 steps 2 and 3) — the snapshot file is
//! overwritten with NEW content but the sidecar still points at the OLD
//! frontier. The resulting LSN mismatch triggers from-zero WAL replay,
//! preserving all committed data.
//!
//! ## Crash injection approach (no production-code hooks)
//!
//! All injection is via **filesystem manipulation** — we write/truncate/
//! delete files in the data dir directly, mirroring the approach in
//! `wal_checkpoint_849.rs` (`m0_5_crash_mid_stream_partial_temp_ignored_*`).
//! No production source is modified. Test-harness-only, per the K-1
//! `mod.rs §"Hooks vs production"` discipline.
//!
//! ## RED-on-revert evidence
//!
//! Each test includes an inline RED-on-revert assertion or comment naming
//! the specific guard that prevents the unsafe path and what mutation would
//! make the test fail. See [`red_on_revert_truncated_snapshot_rejected`] for
//! the dedicated guard-verification test.
//!
//! Run:
//! ```text
//! cargo test -p arcgraph-storage --test checkpoint_crash_f2 -- --nocapture
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use arcgraph_core::{Lsn, PAGE_SIZE, PageId, TenantId};
use arcgraph_storage::blob::BlobStore;
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::checkpoint::{
    CHECKPOINT_SNAPSHOT_FILE, CheckpointSidecar, CheckpointSnapshot, checkpoint,
    read_latest_sidecar, restore_latest_checkpoint, write_sidecar_atomic, write_snapshot_atomic,
};
use arcgraph_storage::crud::{CrudStore, crud_allocator_seed_handle};
use arcgraph_storage::idempotency::IdempotencyStore;
use arcgraph_storage::intern::InternTable;
use arcgraph_storage::io::InMemoryPageIo;
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::permissions::PermissionIndex;
use arcgraph_storage::primary_index::PrimaryPageStore;
use arcgraph_storage::record_store::RecordPageStore;
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{
    AllocatorAdvance, AllocatorSeedHandle, BlobStoreHandle as WalBlobStoreHandle, BundlePageKind,
    PageStoreTarget, PrimaryPageStoreHandle, RecordPageStoreHandle, WalConfig, WalRecordType,
    WalWriter, encode_commit_bundle_v8, recover_from_wal_encrypted,
    recover_from_wal_encrypted_anchored,
};
use bytes::Bytes;
use tempfile::tempdir;

// ──────────────────────────────────────────────────────────────────────────────
// Shared owner bundle (mirrors wal_checkpoint_849.rs / wal_reclamation_p2_1365.rs)
// ──────────────────────────────────────────────────────────────────────────────

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
        let blob: Arc<dyn WalBlobStoreHandle> =
            Arc::clone(&self.blob) as Arc<dyn WalBlobStoreHandle>;
        PageStoreTarget::primary_only(primary)
            .with_record_store(record)
            .with_blob_store(blob)
            .with_allocator_seed(self.allocator_seed())
            .with_intern_table(Arc::clone(&self.intern))
            .with_idempotency_store(Arc::clone(&self.idempotency))
            .with_permission_index(Arc::clone(&self.permissions))
    }

    fn snapshot<'a>(&'a self, seed: &'a dyn AllocatorSeedHandle) -> CheckpointSnapshot<'a> {
        CheckpointSnapshot {
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

    fn establish_checkpoint(&self, data_dir: &std::path::Path, frontier: Lsn) {
        let seed = self.allocator_seed();
        let advances = self.advances();
        let pool = in_mem_buffer_pool();
        checkpoint(
            data_dir,
            &pool,
            &self.snapshot(seed.as_ref()),
            || advances.clone(),
            frontier,
        )
        .unwrap_or_else(|e| panic!("establish_checkpoint at {}: {e:?}", frontier.raw()));
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// WAL + recovery helpers (mirrors wal_checkpoint_849.rs)
// ──────────────────────────────────────────────────────────────────────────────

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

fn in_mem_buffer_pool() -> BufferPool {
    BufferPool::new(16, Arc::new(InMemoryPageIo::new()))
}

/// Append one CommitBundle at `commit_lsn` carrying an MVCC write at key
/// `commit_lsn` (value `b"v{lsn}"`) + one primary page image. Identical to
/// the wal_checkpoint_849.rs helper — the same commit shape is used so the
/// checkpoint snapshot captures MVCC rows AND page images.
fn write_bundle(wal_dir: &std::path::Path, commit_lsn: u64) {
    let writer = WalWriter::spawn(wal_cfg(wal_dir)).unwrap();
    let handle = writer.handle();
    let mut mvcc: HashMap<u64, Option<Bytes>> = HashMap::new();
    mvcc.insert(commit_lsn, Some(Bytes::from(format!("v{commit_lsn}"))));
    let page: Box<[u8; PAGE_SIZE]> = Box::new([(commit_lsn % 256) as u8; PAGE_SIZE]);
    let staged = vec![(
        BundlePageKind::PrimaryIndex,
        PageId::new(1000 + commit_lsn),
        TenantId::DEFAULT,
        page,
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
    writer.shutdown().unwrap();
}

fn write_bundles(wal_dir: &std::path::Path, lsns: impl IntoIterator<Item = u64>) {
    for lsn in lsns {
        write_bundle(wal_dir, lsn);
    }
}

/// Recover `owners` from the full WAL (from-zero replay).
fn recover_full(
    wal_dir: &std::path::Path,
    owners: &Owners,
) -> arcgraph_storage::wal::RecoveryReport {
    recover_from_wal_encrypted(
        wal_dir,
        Arc::clone(&owners.txn),
        owners.target(),
        None,
        None,
    )
    .unwrap()
}

/// Read the MVCC value at `key` visible at the current committed watermark.
fn read(owners: &Owners, key: u64) -> Option<Bytes> {
    let snap = owners.txn.current_lsn();
    owners.txn.read_at(TenantId::DEFAULT, key, snap)
}

/// Assert every key 1..=n is readable with the expected `v{i}` value.
fn assert_all_readable(owners: &Owners, n: u64, context: &str) {
    for i in 1..=n {
        assert_eq!(
            read(owners, i),
            Some(Bytes::from(format!("v{i}"))),
            "F-2 acked-lost: key {i} missing after recovery — {context}",
        );
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Crash-injection helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Build the bytes a REAL checkpoint snapshot would produce at `frontier`
/// from `owners` (via `write_snapshot_atomic` to a temp location, then read
/// back). This gives us a valid byte-stream that we can truncate at specific
/// percentages to simulate a crash at any point in the streaming write.
fn snapshot_bytes_for(owners: &Owners, frontier: Lsn, scratch_dir: &std::path::Path) -> Vec<u8> {
    // Use a separate scratch dir so we don't disturb the test's data dir.
    let seed = owners.allocator_seed();
    let advances = owners.advances();
    write_snapshot_atomic(
        scratch_dir,
        &owners.snapshot(seed.as_ref()),
        frontier,
        &advances,
    )
    .unwrap();
    std::fs::read(scratch_dir.join(CHECKPOINT_SNAPSHOT_FILE)).unwrap()
}

/// Place an orphaned partial temp file (at `pct`% of `full_bytes`) in
/// `data_dir` WITHOUT renaming it and WITHOUT updating the sidecar.
/// Simulates the "crash mid-stream" scenario: the process died between
/// opening the temp and the `rename` call (or between any two write calls).
///
/// The temp file name follows the same unique-temp convention as the
/// production `StreamingSnapshotWrite::open` path (same prefix, unique
/// suffix), so it exercises the exact artifact class that recovery must ignore.
fn place_orphaned_partial_temp(data_dir: &std::path::Path, full_bytes: &[u8], pct: usize) {
    let partial_len = full_bytes.len() * pct / 100;
    let tmp_name = format!(
        "{CHECKPOINT_SNAPSHOT_FILE}.tmp.{}.99999",
        std::process::id()
    );
    std::fs::write(data_dir.join(&tmp_name), &full_bytes[..partial_len]).unwrap();
}

// ──────────────────────────────────────────────────────────────────────────────
// Test 1: Crash mid-body, NO prior checkpoint
//
// Scenario: A fresh store commits N records. A checkpoint begins writing its
// snapshot body, crashes at 50% through the body (partial temp, no rename, no
// sidecar). Recovery has NO sidecar to anchor on → from-zero WAL replay. Every
// acked commit must be readable post-recovery.
//
// RED-on-revert: if `restore_latest_checkpoint` were to scan for and accept
// orphaned temp files as if they were the established snapshot, a truncated body
// would fail the CRC check inside `decode_and_restore`. Verifying it returns
// `None` (not `Some`) proves the "no stale-temp" contract is upheld.
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn crash_mid_body_no_prior_checkpoint_from_zero_wal_replay() {
    const N: u64 = 12;

    let dir = tempdir().unwrap();
    let wal_dir = dir.path();

    // Commit N records to the WAL.
    write_bundles(wal_dir, 1..=N);

    // Populate owners from the full WAL (gives us state to snapshot from).
    let p1 = Owners::fresh();
    recover_full(wal_dir, &p1);
    assert_all_readable(&p1, N, "pre-crash baseline");

    // Capture what the real checkpoint snapshot bytes would look like.
    // (Use a separate scratch dir so the test data dir has no snapshot yet.)
    let scratch = tempdir().unwrap();
    let snap_bytes = snapshot_bytes_for(&p1, Lsn::new(N), scratch.path());

    // Simulate crash mid-body (50%): write a partial orphaned temp, no rename.
    place_orphaned_partial_temp(wal_dir, &snap_bytes, 50);

    // RED-on-revert check (inline): there must be NO sidecar (no established
    // checkpoint). If the orphaned temp were ever promoted to the snapshot slot,
    // it would be the only "candidate" — but it's not in CHECKPOINT.snap.
    assert!(
        read_latest_sidecar(wal_dir).unwrap().is_none(),
        "no sidecar must exist after a crash before the sidecar write",
    );

    // Recovery: no sidecar → from-zero WAL replay, owners starting pristine.
    let p2 = Owners::fresh();
    let seed2 = p2.allocator_seed();
    let restore = restore_latest_checkpoint(wal_dir, &p2.snapshot(seed2.as_ref())).unwrap();
    // RED-on-revert: if restore returned Some here (found a checkpoint from the
    // partial temp), the anchored replay would skip committed WAL records →
    // acked-lost. Asserting None proves the partial-temp invariant holds.
    assert!(
        restore.is_none(),
        "crash-mid-body (no prior): from-zero path expected (None), got {:?}",
        restore.map(|r| r.checkpoint_lsn),
    );

    let report =
        recover_from_wal_encrypted(wal_dir, Arc::clone(&p2.txn), p2.target(), None, None).unwrap();
    assert_eq!(
        report.metrics.bundles_applied, N,
        "from-zero replay must apply all N bundles",
    );
    assert_all_readable(&p2, N, "crash-mid-body, no prior checkpoint");
}

// ──────────────────────────────────────────────────────────────────────────────
// Test 2: Crash mid-body WITH a prior established checkpoint
//
// A prior checkpoint is established at N1. M more commits follow. A second
// checkpoint begins and crashes mid-body (50% of new snapshot written to temp,
// no rename, no sidecar update). Recovery must:
//   - find the prior sidecar (still valid, still points at N1 snapshot)
//   - restore the prior checkpoint (frontier = N1)
//   - replay only the M post-checkpoint WAL records
//   - serve all N1+M acked commits
//
// RED-on-revert: if the crash left a partial temp AND recovery somehow tried to
// use it (bypassing the "only read CHECKPOINT.snap" contract), it would be
// CRC-corrupt → Err(Corrupt). Asserting `restore.checkpoint_lsn == N1` proves
// the prior checkpoint was used, not the orphaned temp.
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn crash_mid_body_with_prior_checkpoint_recovers_prior_plus_tail() {
    const N1: u64 = 10;
    const M: u64 = 5;

    let dir = tempdir().unwrap();
    let wal_dir = dir.path();

    // Establish prior checkpoint at N1.
    write_bundles(wal_dir, 1..=N1);
    let p1 = Owners::fresh();
    recover_full(wal_dir, &p1);
    p1.establish_checkpoint(wal_dir, Lsn::new(N1));

    // M more commits after the prior checkpoint.
    write_bundles(wal_dir, (N1 + 1)..=(N1 + M));
    let p1b = Owners::fresh();
    recover_full(wal_dir, &p1b);

    // Capture what the second checkpoint (at N1+M) would produce.
    let scratch = tempdir().unwrap();
    let snap_bytes = snapshot_bytes_for(&p1b, Lsn::new(N1 + M), scratch.path());

    // Simulate crash at 50% of the second snapshot body.
    place_orphaned_partial_temp(wal_dir, &snap_bytes, 50);

    // Sidecar must still point at N1 (the prior).
    assert_eq!(
        read_latest_sidecar(wal_dir)
            .unwrap()
            .unwrap()
            .checkpoint_lsn,
        Lsn::new(N1),
        "sidecar unchanged after mid-body crash",
    );

    // Recovery: prior checkpoint restored at N1, then M records replayed.
    let p2 = Owners::fresh();
    let seed2 = p2.allocator_seed();
    let restore = restore_latest_checkpoint(wal_dir, &p2.snapshot(seed2.as_ref()))
        .unwrap()
        .expect("prior checkpoint must be found");
    assert_eq!(
        restore.checkpoint_lsn,
        Lsn::new(N1),
        "recovery anchored at prior frontier, not the orphaned partial temp",
    );

    let report = recover_from_wal_encrypted_anchored(
        wal_dir,
        Arc::clone(&p2.txn),
        p2.target(),
        None,
        None,
        restore.checkpoint_lsn,
    )
    .unwrap();
    assert_eq!(
        report.metrics.bundles_applied, M,
        "only the M post-checkpoint records replayed (anchored at N1)",
    );
    assert_all_readable(&p2, N1 + M, "crash-mid-body, with prior checkpoint");
}

// ──────────────────────────────────────────────────────────────────────────────
// Test 3: Crash mid-supplement WITH a prior checkpoint
//
// The supplement section (EvictedSupplement, carrying post-guard evicted page
// images) begins AFTER the main body sections complete. Simulated by truncating
// the snapshot bytes at 90% (body sections done, supplement truncated).
// Recovery behavior is identical to Test 2 (prior checkpoint used, tail replayed).
//
// RED-on-revert: same as Test 2. The supplement-truncated temp (still orphaned,
// not renamed) is invisible to `restore_latest_checkpoint`; the prior sidecar
// and CHECKPOINT.snap remain the established checkpoint.
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn crash_mid_supplement_with_prior_checkpoint_recovers_prior_plus_tail() {
    const N1: u64 = 8;
    const M: u64 = 4;

    let dir = tempdir().unwrap();
    let wal_dir = dir.path();

    write_bundles(wal_dir, 1..=N1);
    let p1 = Owners::fresh();
    recover_full(wal_dir, &p1);
    p1.establish_checkpoint(wal_dir, Lsn::new(N1));

    write_bundles(wal_dir, (N1 + 1)..=(N1 + M));
    let p1b = Owners::fresh();
    recover_full(wal_dir, &p1b);

    let scratch = tempdir().unwrap();
    let snap_bytes = snapshot_bytes_for(&p1b, Lsn::new(N1 + M), scratch.path());

    // 90% covers the main body sections; the supplement and footer are truncated.
    place_orphaned_partial_temp(wal_dir, &snap_bytes, 90);

    assert_eq!(
        read_latest_sidecar(wal_dir)
            .unwrap()
            .unwrap()
            .checkpoint_lsn,
        Lsn::new(N1),
        "sidecar unchanged after mid-supplement crash",
    );

    let p2 = Owners::fresh();
    let seed2 = p2.allocator_seed();
    let restore = restore_latest_checkpoint(wal_dir, &p2.snapshot(seed2.as_ref()))
        .unwrap()
        .expect("prior checkpoint found");
    assert_eq!(restore.checkpoint_lsn, Lsn::new(N1));

    let report = recover_from_wal_encrypted_anchored(
        wal_dir,
        Arc::clone(&p2.txn),
        p2.target(),
        None,
        None,
        restore.checkpoint_lsn,
    )
    .unwrap();
    assert_eq!(report.metrics.bundles_applied, M);
    assert_all_readable(&p2, N1 + M, "crash-mid-supplement, with prior checkpoint");
}

// ──────────────────────────────────────────────────────────────────────────────
// Test 4: Crash pre-rename — temp fully written, NOT renamed, sidecar unchanged
//
// The streaming snapshot is 100% complete (all body sections + evicted
// supplement + footer CRC) and written to the crash-atomic temp file, but the
// `rename(tmp → CHECKPOINT.snap)` never executed (crash between write+fsync and
// the rename). Recovery: CHECKPOINT.snap is the prior snapshot; sidecar says
// N1; anchored replay of M records.
//
// RED-on-revert: a fully-written temp sitting beside the prior CHECKPOINT.snap
// does not interfere. `restore_latest_checkpoint` reads CHECKPOINT.snap (not
// CHECKPOINT.snap.tmp.*) — verified by asserting frontier == N1, not N1+M.
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn crash_pre_rename_complete_temp_ignored_recovers_prior_plus_tail() {
    const N1: u64 = 9;
    const M: u64 = 6;

    let dir = tempdir().unwrap();
    let wal_dir = dir.path();

    write_bundles(wal_dir, 1..=N1);
    let p1 = Owners::fresh();
    recover_full(wal_dir, &p1);
    p1.establish_checkpoint(wal_dir, Lsn::new(N1));

    write_bundles(wal_dir, (N1 + 1)..=(N1 + M));
    let p1b = Owners::fresh();
    recover_full(wal_dir, &p1b);

    let scratch = tempdir().unwrap();
    let snap_bytes = snapshot_bytes_for(&p1b, Lsn::new(N1 + M), scratch.path());

    // 100% written to orphaned temp — the pre-rename crash point.
    // This is the most subtle case: the temp is valid (passes CRC), but it
    // is NOT at the canonical CHECKPOINT.snap path → recovery ignores it.
    place_orphaned_partial_temp(wal_dir, &snap_bytes, 100);

    assert_eq!(
        read_latest_sidecar(wal_dir)
            .unwrap()
            .unwrap()
            .checkpoint_lsn,
        Lsn::new(N1),
        "sidecar unchanged after pre-rename crash",
    );

    let p2 = Owners::fresh();
    let seed2 = p2.allocator_seed();
    let restore = restore_latest_checkpoint(wal_dir, &p2.snapshot(seed2.as_ref()))
        .unwrap()
        .expect("prior checkpoint found");
    // RED-on-revert: if this were N1+M (the temp's frontier), recovery would
    // anchor at N1+M and skip the M post-checkpoint WAL records → acked-lost.
    assert_eq!(
        restore.checkpoint_lsn,
        Lsn::new(N1),
        "pre-rename crash: prior frontier used, not the un-renamed temp's frontier",
    );

    let report = recover_from_wal_encrypted_anchored(
        wal_dir,
        Arc::clone(&p2.txn),
        p2.target(),
        None,
        None,
        restore.checkpoint_lsn,
    )
    .unwrap();
    assert_eq!(report.metrics.bundles_applied, M);
    assert_all_readable(&p2, N1 + M, "crash-pre-rename, complete temp ignored");
}

// ──────────────────────────────────────────────────────────────────────────────
// Test 5: Crash post-snapshot-rename, pre-sidecar (with prior checkpoint)
//
// ADR-229 §Decision step 2 (rename) completes (CHECKPOINT.snap now carries the
// NEW snapshot at N1+M), but step 3 (sidecar write) never ran. The sidecar
// still says N1. `decode_and_restore` detects the LSN mismatch between the
// sidecar frontier (N1) and the snapshot header (N1+M) → owners stay pristine
// → from-zero WAL replay recovers everything.
//
// RED-on-revert: if the LSN-mismatch check in `decode_and_restore` were
// removed, the N1+M snapshot would be applied with the sidecar's N1 frontier.
// Recovery would then replay WAL records above N1 on top of already-restored
// N1+M state, potentially duplicating effects on non-idempotent owners.
// Asserting `restore == None` proves the guard triggers correctly.
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn crash_post_snapshot_rename_pre_sidecar_lsn_mismatch_fallback_to_wal() {
    const N1: u64 = 7;
    const M: u64 = 5;

    let dir = tempdir().unwrap();
    let wal_dir = dir.path();

    // Prior checkpoint at N1.
    write_bundles(wal_dir, 1..=N1);
    let p1 = Owners::fresh();
    recover_full(wal_dir, &p1);
    p1.establish_checkpoint(wal_dir, Lsn::new(N1));

    // M more commits.
    write_bundles(wal_dir, (N1 + 1)..=(N1 + M));
    let p1b = Owners::fresh();
    recover_full(wal_dir, &p1b);

    // Step 2 (snapshot rename) runs: CHECKPOINT.snap is overwritten with the
    // NEW snapshot (frontier = N1+M) — but step 3 (sidecar) never runs.
    // Simulate by writing the new snapshot directly to the canonical path.
    let seed_b = p1b.allocator_seed();
    let advances_b = p1b.advances();
    write_snapshot_atomic(
        wal_dir,
        &p1b.snapshot(seed_b.as_ref()),
        Lsn::new(N1 + M),
        &advances_b,
    )
    .unwrap();
    // Sidecar is intentionally NOT updated (simulating the crash between steps
    // 2 and 3). Verify it still says N1.
    assert_eq!(
        read_latest_sidecar(wal_dir)
            .unwrap()
            .unwrap()
            .checkpoint_lsn,
        Lsn::new(N1),
        "sidecar still at N1 after snapshot-only rename",
    );

    // Recovery: sidecar says N1, CHECKPOINT.snap says N1+M → LSN mismatch
    // → restore_latest_checkpoint returns None (owners left pristine).
    let p2 = Owners::fresh();
    let seed2 = p2.allocator_seed();
    let restore = restore_latest_checkpoint(wal_dir, &p2.snapshot(seed2.as_ref())).unwrap();
    assert!(
        restore.is_none(),
        "post-snapshot-rename, pre-sidecar: LSN mismatch must trigger from-zero \
         (got Some at {:?})",
        restore.map(|r| r.checkpoint_lsn),
    );
    // Owners must be pristine (no partial state from the mismatched snapshot).
    assert_eq!(
        p2.txn.current_lsn(),
        Lsn::ZERO,
        "owners must be pristine after LSN-mismatch fallback",
    );

    // From-zero WAL replay recovers ALL N1+M committed records.
    let report =
        recover_from_wal_encrypted(wal_dir, Arc::clone(&p2.txn), p2.target(), None, None).unwrap();
    assert_eq!(
        report.metrics.bundles_applied,
        N1 + M,
        "from-zero replay applies all N1+M bundles",
    );
    assert_all_readable(&p2, N1 + M, "crash-post-snapshot-rename, pre-sidecar");
}

// ──────────────────────────────────────────────────────────────────────────────
// Test 6: Crash post-publish (complete checkpoint), pre-WAL-reclaim
//
// Both ADR-229 steps 2 and 3 completed successfully — the checkpoint is fully
// established (CHECKPOINT.snap at N, sidecar at N). A "crash" then occurs
// before the background WAL segment reclamation task removes segments ≤ N.
// Recovery: checkpoint at N restored, WAL tail (M records) replayed, all N+M
// commits readable. WAL reclamation is an optimization for WAL size-bounding
// (#1365, ADR-229 §Segment reclamation) — correctness does not depend on it.
//
// RED-on-revert: if recovery failed to anchor at N (e.g. if `full_state_snapshot`
// flag were not set on the sidecar), it would replay the entire WAL from zero.
// This is functionally correct but the suite asserts EXACTLY M bundles were
// applied (proving the anchor held), so any regress in the anchor flag would
// surface here as M > expected.
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn crash_post_publish_pre_wal_reclaim_anchored_recovery_no_acked_lost() {
    const N: u64 = 15;
    const M: u64 = 7;

    let dir = tempdir().unwrap();
    let wal_dir = dir.path();

    // Commit N records, then establish a complete checkpoint at N.
    write_bundles(wal_dir, 1..=N);
    let p1 = Owners::fresh();
    recover_full(wal_dir, &p1);
    p1.establish_checkpoint(wal_dir, Lsn::new(N));

    // M more commits after the checkpoint (the post-checkpoint WAL tail).
    write_bundles(wal_dir, (N + 1)..=(N + M));

    // "Crash" here: no WAL reclamation, no segment deletion. The WAL contains
    // all N+M records across its segments. Recovery must use the checkpoint.
    let p2 = Owners::fresh();
    let seed2 = p2.allocator_seed();
    let restore = restore_latest_checkpoint(wal_dir, &p2.snapshot(seed2.as_ref()))
        .unwrap()
        .expect("complete checkpoint must be found");
    assert_eq!(
        restore.checkpoint_lsn,
        Lsn::new(N),
        "checkpoint at N must be the established frontier",
    );
    assert_eq!(
        restore.counts.mvcc_records, N,
        "snapshot captured exactly N MVCC records",
    );

    let report = recover_from_wal_encrypted_anchored(
        wal_dir,
        Arc::clone(&p2.txn),
        p2.target(),
        None,
        None,
        restore.checkpoint_lsn,
    )
    .unwrap();
    // RED-on-revert: this asserts EXACTLY M — if the anchor were broken,
    // from-zero replay would apply N+M bundles and this assertion would fail.
    assert_eq!(
        report.metrics.bundles_applied, M,
        "anchored replay applies only the M post-checkpoint tail (WAL reclaim not required)",
    );
    assert_all_readable(&p2, N + M, "crash-post-publish, pre-WAL-reclaim");
}

// ──────────────────────────────────────────────────────────────────────────────
// Test 7 (RED-on-revert verification): truncated / corrupt snapshot rejected
//
// Demonstrate that the CRC guard in `decode_and_restore` is the load-bearing
// property that makes crash points A/B/C safe. We simulate what WOULD happen
// if a partial (crash-truncated) snapshot body ended up at the canonical
// CHECKPOINT.snap path with a matching sidecar pointing at it — the scenario
// that the both-or-neither atomicity contract prevents in production.
//
// Expected: `restore_latest_checkpoint` returns `None` (corrupt, owners
// pristine, from-zero WAL replay). From-zero replay then recovers all data.
//
// Mutation that would make this test RED: remove the CRC check in
// `checkpoint::snapshot::decode_and_restore`. Without it, a truncated
// body would be accepted as-is (partial state restored → WAL anchored at
// the corrupt snapshot LSN → acked-lost on subsequent reads).
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn red_on_revert_truncated_snapshot_rejected_from_zero_recovers_all() {
    const N: u64 = 10;

    let dir = tempdir().unwrap();
    let wal_dir = dir.path();

    write_bundles(wal_dir, 1..=N);
    let p1 = Owners::fresh();
    recover_full(wal_dir, &p1);
    // Write a VALID snapshot (correct CRC, all sections) to CHECKPOINT.snap.
    let seed1 = p1.allocator_seed();
    let advances1 = p1.advances();
    write_snapshot_atomic(
        wal_dir,
        &p1.snapshot(seed1.as_ref()),
        Lsn::new(N),
        &advances1,
    )
    .unwrap();
    // Write the matching sidecar.
    write_sidecar_atomic(
        wal_dir,
        &CheckpointSidecar::full_state(Lsn::new(N), Lsn::new(N), 0),
    )
    .unwrap();

    // Now TRUNCATE CHECKPOINT.snap to 50% — simulating what would happen if
    // the crash-mid-body partial content were somehow promoted to the canonical
    // path (the scenario atomicity prevents). This is the adversarial mutation.
    let snap_path = wal_dir.join(CHECKPOINT_SNAPSHOT_FILE);
    let mut snap_bytes = std::fs::read(&snap_path).unwrap();
    snap_bytes.truncate(snap_bytes.len() / 2);
    std::fs::write(&snap_path, &snap_bytes).unwrap();

    // Recovery must detect the truncation via CRC mismatch and fall back to
    // from-zero WAL replay (owners pristine).
    let p2 = Owners::fresh();
    let seed2 = p2.allocator_seed();
    let restore = restore_latest_checkpoint(wal_dir, &p2.snapshot(seed2.as_ref())).unwrap();
    // RED-on-revert: if this asserts Some(...), the CRC guard was bypassed →
    // partial state is in owners → WAL anchored at N with partial records →
    // acked-lost. The test going RED here is the signal to investigate.
    assert!(
        restore.is_none(),
        "truncated snapshot (CRC corrupt) must be rejected (got Some at {:?})",
        restore.map(|r| r.checkpoint_lsn),
    );
    assert_eq!(
        p2.txn.current_lsn(),
        Lsn::ZERO,
        "owners pristine after CRC-corrupt snapshot (no partial state installed)",
    );

    // From-zero WAL replay recovers ALL N committed records.
    let report =
        recover_from_wal_encrypted(wal_dir, Arc::clone(&p2.txn), p2.target(), None, None).unwrap();
    assert_eq!(report.metrics.bundles_applied, N);
    assert_all_readable(
        &p2,
        N,
        "red-on-revert: CRC-corrupt snapshot → from-zero safe",
    );
}

// ──────────────────────────────────────────────────────────────────────────────
// Test 8: multi-point sweep — 3 crash fractions × prior-checkpoint variant
//
// Parameterized sweep over multiple truncation percentages (25%, 50%, 75%) and
// the supplement-complete (90%) point, with a prior checkpoint. Verifies the
// acked-lost invariant holds at every crash fraction without requiring per-point
// tests. This is the compact regression oracle: if any of the checkpoint
// subsystem changes broke the partial-temp-ignore path at a specific write
// fraction, this test catches it.
// ──────────────────────────────────────────────────────────────────────────────

#[test]
fn crash_sweep_multiple_fractions_with_prior_checkpoint_no_acked_lost() {
    const N1: u64 = 8;
    const M: u64 = 4;

    for pct in [25usize, 50, 75, 90, 100] {
        let dir = tempdir().unwrap();
        let wal_dir = dir.path();

        write_bundles(wal_dir, 1..=N1);
        let p1 = Owners::fresh();
        recover_full(wal_dir, &p1);
        p1.establish_checkpoint(wal_dir, Lsn::new(N1));

        write_bundles(wal_dir, (N1 + 1)..=(N1 + M));
        let p1b = Owners::fresh();
        recover_full(wal_dir, &p1b);

        let scratch = tempdir().unwrap();
        let snap_bytes = snapshot_bytes_for(&p1b, Lsn::new(N1 + M), scratch.path());

        place_orphaned_partial_temp(wal_dir, &snap_bytes, pct);

        assert_eq!(
            read_latest_sidecar(wal_dir)
                .unwrap()
                .unwrap()
                .checkpoint_lsn,
            Lsn::new(N1),
            "pct={pct}: sidecar unchanged",
        );

        let p2 = Owners::fresh();
        let seed2 = p2.allocator_seed();
        let restore = restore_latest_checkpoint(wal_dir, &p2.snapshot(seed2.as_ref()))
            .unwrap()
            .expect("prior checkpoint must be found");
        assert_eq!(
            restore.checkpoint_lsn,
            Lsn::new(N1),
            "pct={pct}: recovery anchored at prior frontier N1",
        );

        let report = recover_from_wal_encrypted_anchored(
            wal_dir,
            Arc::clone(&p2.txn),
            p2.target(),
            None,
            None,
            restore.checkpoint_lsn,
        )
        .unwrap();
        assert_eq!(
            report.metrics.bundles_applied, M,
            "pct={pct}: only M post-checkpoint records replayed",
        );
        assert_all_readable(&p2, N1 + M, &format!("crash-sweep pct={pct}"));
    }
}
