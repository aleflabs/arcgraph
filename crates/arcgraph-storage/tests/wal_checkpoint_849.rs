//! SVC-1 / #849 / ADR-229 — WAL checkpoint producer + checkpoint-anchored
//! recovery integration tests.
//!
//! The Director-named rc-oracle lives here: the BOUNDED-RECOVERY test
//! proves that after a checkpoint, restart-recovery replays ONLY the
//! post-checkpoint WAL tail (bounded), not the entire history — the
//! property that makes a 167 GB / 10M-scale WAL restartable in bounded
//! time (the #849 rc-blocker, CZ-proven).
//!
//! Post-ULTRACODE (verdict 1371) this suite ALSO carries the three
//! blocking-bug regression oracles, each RED-on-revert:
//! - BLOCK-1 (CORRUPTION): concurrent alloc in the capture window → the
//!   next-allocated id must be STRICTLY ABOVE every restored id (Node,
//!   Rel, Page). RED-on-revert = drain allocator BEFORE the frontier /
//!   without the commit-freeze → id reuse.
//! - BLOCK-2 (CORRUPTION): a page image captured while a commit's WAL
//!   fsync has not returned → phantom durable record. RED-on-revert =
//!   capture without the commit-freeze → phantom survives.
//! - BLOCK-3 (DATALOSS): an LSN-mismatched snapshot must leave owners
//!   PRISTINE (from-zero replay recovers everything). RED-on-revert =
//!   restore-then-check → committed records ≤ the untrusted LSN lost.

use std::collections::HashMap;
use std::sync::Arc;

use arcgraph_core::{Lsn, PAGE_SIZE, PageId, TenantId};
use arcgraph_storage::BlobStoreHandle;
use arcgraph_storage::blob::BlobStore;
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::checkpoint::{
    CheckpointSidecar, CheckpointSnapshot, read_latest_sidecar, restore_latest_checkpoint,
    write_sidecar_atomic, write_snapshot_atomic,
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
    AllocatorAdvance, AllocatorKind, AllocatorSeedHandle, BundlePageKind, PageStoreTarget,
    PrimaryPageStoreHandle, RecordPageStoreHandle, SideChannelWrite, StagedEmit, WalConfig,
    WalRecordType, WalWriter, encode_commit_bundle_v8, recover_from_wal_encrypted,
    recover_from_wal_encrypted_anchored,
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

    /// Build a `CheckpointSnapshot` view over these owners. `seed` must
    /// outlive the returned view (borrowed).
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

    /// Collect the full allocator-advance set (page-kind + Node/Rel).
    fn advances(&self) -> Vec<AllocatorAdvance> {
        let mut a = self.allocator.snapshot_advances();
        a.extend(self.crud.snapshot_allocator_advances());
        a
    }

    /// Write a full-state checkpoint at `frontier` via the codec entry
    /// points (encode current owner state + advances, then sidecar). Uses
    /// the SAME `write_snapshot_atomic` / `write_sidecar_atomic` the
    /// producer uses; a serial test has no concurrency so no freeze is
    /// needed here.
    fn write_checkpoint(&self, dir: &std::path::Path, frontier: Lsn) {
        let seed = self.allocator_seed();
        let advances = self.advances();
        write_snapshot_atomic(dir, &self.snapshot(seed.as_ref()), frontier, &advances).unwrap();
        write_sidecar_atomic(dir, &CheckpointSidecar::full_state(frontier, frontier, 0)).unwrap();
    }
}

fn in_mem_buffer_pool() -> BufferPool {
    BufferPool::new(16, Arc::new(InMemoryPageIo::new()))
}

// ─── WAL fixture helpers ──────────────────────────────────────────

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

fn mk_page(fill: u8) -> Box<[u8; PAGE_SIZE]> {
    Box::new([fill; PAGE_SIZE])
}

/// Append one v8 CommitBundle at `commit_lsn` carrying an MVCC write at
/// key `commit_lsn` (value `b"v{lsn}"`) + one primary page image.
fn write_bundle(dir: &std::path::Path, commit_lsn: u64) {
    let writer = WalWriter::spawn(wal_cfg(dir)).unwrap();
    let handle = writer.handle();
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
    writer.shutdown().unwrap();
}

/// Append bundles for commit_lsn in `lsns`.
fn write_bundles(dir: &std::path::Path, lsns: impl IntoIterator<Item = u64>) {
    for lsn in lsns {
        write_bundle(dir, lsn);
    }
}

/// Recover `owners` from the full WAL (from zero) — the initial "populate
/// the served state" pass that a first boot would run.
fn recover_full(dir: &std::path::Path, owners: &Owners) -> arcgraph_storage::wal::RecoveryReport {
    recover_from_wal_encrypted(dir, Arc::clone(&owners.txn), owners.target(), None, None).unwrap()
}

/// Read the MVCC value at `key` visible at the current committed
/// watermark.
fn read(owners: &Owners, key: u64) -> Option<Bytes> {
    let snap = owners.txn.current_lsn();
    owners.txn.read_at(TenantId::DEFAULT, key, snap)
}

// ─────────────────────────────────────────────────────────────────
// THE rc-oracle — BOUNDED-RECOVERY, BOTH DIRECTIONS
// ─────────────────────────────────────────────────────────────────

/// Direction 1 (the bound): N commits → checkpoint → M more commits →
/// restart → recovery replays ONLY the M post-checkpoint records, and
/// the recovered state contains ALL N+M committed values.
#[test]
fn bounded_recovery_replays_only_post_checkpoint_tail() {
    const N: u64 = 20;
    const M: u64 = 5;
    let dir = tempdir().unwrap();
    let wal = dir.path();

    write_bundles(wal, 1..=N);
    let p1 = Owners::fresh();
    recover_full(wal, &p1);
    for i in 1..=N {
        assert_eq!(
            read(&p1, i),
            Some(Bytes::from(format!("v{i}"))),
            "N key {i}"
        );
    }

    let checkpoint_lsn = Lsn::new(N);
    p1.write_checkpoint(dir.path(), checkpoint_lsn);

    write_bundles(wal, (N + 1)..=(N + M));

    let p2 = Owners::fresh();
    let restore = {
        let seed2 = p2.allocator_seed();
        restore_latest_checkpoint(dir.path(), &p2.snapshot(seed2.as_ref()))
            .unwrap()
            .expect("a full-state checkpoint must be found")
    };
    assert_eq!(restore.checkpoint_lsn, checkpoint_lsn);
    assert_eq!(restore.counts.mvcc_records, N);

    let report = recover_from_wal_encrypted_anchored(
        wal,
        Arc::clone(&p2.txn),
        p2.target(),
        None,
        None,
        restore.checkpoint_lsn,
    )
    .unwrap();

    // (a) THE BOUND: exactly M bundles applied, the N below the frontier skipped.
    assert_eq!(
        report.metrics.bundles_applied, M,
        "checkpoint-anchored recovery must replay ONLY the {M} post-checkpoint records, \
         got {} (the #849 rc-blocker bound)",
        report.metrics.bundles_applied,
    );
    assert!(report.metrics.bundles_skipped_idempotent >= N);

    // (b) recovered state == full pre-restart committed state: ALL N+M visible.
    for i in 1..=(N + M) {
        assert_eq!(
            read(&p2, i),
            Some(Bytes::from(format!("v{i}"))),
            "post-restart key {i}"
        );
    }
}

/// Direction 2 (RED-on-revert): DISABLE checkpoint anchoring (floor =
/// Lsn::ZERO) → recovery replays ALL N+M records, NOT just M.
#[test]
fn revert_disables_anchoring_replays_everything() {
    const N: u64 = 20;
    const M: u64 = 5;
    let dir = tempdir().unwrap();
    let wal = dir.path();

    write_bundles(wal, 1..=N);
    let p1 = Owners::fresh();
    recover_full(wal, &p1);
    p1.write_checkpoint(dir.path(), Lsn::new(N));
    write_bundles(wal, (N + 1)..=(N + M));

    let p2 = Owners::fresh();
    let report = recover_from_wal_encrypted_anchored(
        wal,
        Arc::clone(&p2.txn),
        p2.target(),
        None,
        None,
        Lsn::ZERO, // ← anchoring disabled
    )
    .unwrap();

    assert_eq!(
        report.metrics.bundles_applied,
        N + M,
        "with anchoring DISABLED (floor=0), recovery replays the ENTIRE history",
    );
    assert!(report.metrics.bundles_applied > M);
}

// ─────────────────────────────────────────────────────────────────
// BLOCK-3 (DATALOSS) — LSN-mismatch fallback leaves owners PRISTINE
// ─────────────────────────────────────────────────────────────────

/// Post-fix (BLOCK-3): a sidecar@99 pointing at a snapshot whose header
/// carries LSN 20 (capturing only key 5) MUST be rejected WITHOUT
/// touching live owners or the TxnManager watermark. Recovery then does a
/// genuine from-zero replay and recovers ALL WAL-committed keys 5..=20.
///
/// This is the reproduced ULTRACODE test inverted to GREEN. RED-on-revert
/// = if `decode_and_restore` mutates owners before the LSN cross-check
/// (the pre-fix behaviour), `current_lsn()` is polluted to 20, the
/// anchored replay skips ≤20, and keys 6..=20 are LOST.
#[test]
fn block3_lsn_mismatch_leaves_owners_pristine_no_dataloss() {
    let dir = tempdir().unwrap();
    let wal = dir.path();

    // WAL holds committed keys 5..=20.
    write_bundles(wal, 5..=20);

    // Build an UNTRUSTED snapshot @ LSN 20 that captures ONLY key 5 (the
    // torn/divergent-producer shape), but stamp the SIDECAR @ 99 — the
    // mismatch the fix must catch.
    let src = Owners::fresh();
    src.txn.apply_replay_mvcc_write(
        Lsn::new(20),
        TenantId::DEFAULT,
        5,
        Some(Bytes::from_static(b"v5")),
    );
    src.txn.seed_after_replay(Lsn::new(20));
    let seed = src.allocator_seed();
    let advances = src.advances();
    // Snapshot header LSN = 20 (capturing only key 5).
    write_snapshot_atomic(
        dir.path(),
        &src.snapshot(seed.as_ref()),
        Lsn::new(20),
        &advances,
    )
    .unwrap();
    // Sidecar frontier = 99 (≠ 20). The producer would NEVER write this
    // pair, but a divergent interval-vs-shutdown race / torn establish can.
    write_sidecar_atomic(
        dir.path(),
        &CheckpointSidecar::full_state(Lsn::new(99), Lsn::new(99), 0),
    )
    .unwrap();

    // RESTART: restore MUST reject the mismatch and leave owners pristine.
    let p = Owners::fresh();
    let seed_r = p.allocator_seed();
    let restore = restore_latest_checkpoint(dir.path(), &p.snapshot(seed_r.as_ref())).unwrap();
    assert!(
        restore.is_none(),
        "an LSN-mismatched snapshot MUST fall back to None (owners pristine)",
    );
    // The owners' TxnManager watermark MUST be pristine (Lsn::ZERO) — NOT
    // polluted to the untrusted snapshot LSN 20. This is the exact
    // pollution that caused the reproduced data loss.
    assert_eq!(
        p.txn.current_lsn(),
        Lsn::ZERO,
        "BLOCK-3: the untrusted snapshot must NOT have seeded the TxnManager watermark",
    );
    // key 5 must NOT have been restored from the untrusted snapshot.
    assert_eq!(
        read(&p, 5),
        None,
        "no partial restore of the untrusted snapshot"
    );

    // Genuine from-zero replay (frontier = ZERO) recovers ALL WAL keys.
    let report = recover_from_wal_encrypted_anchored(
        wal,
        Arc::clone(&p.txn),
        p.target(),
        None,
        None,
        Lsn::ZERO,
    )
    .unwrap();
    assert_eq!(
        report.metrics.bundles_applied, 16,
        "from-zero replays all 16 (keys 5..=20)"
    );
    for i in 5..=20 {
        assert_eq!(
            read(&p, i),
            Some(Bytes::from(format!("v{i}"))),
            "BLOCK-3: committed key {i} must survive (was silently LOST pre-fix)",
        );
    }
}

/// RED-on-revert companion for BLOCK-3: this test drives the OLD
/// (buggy) contract explicitly — restore-into-live-owners THEN check the
/// mismatch — and asserts the data-loss it produces. It documents the
/// exact failure the fix prevents; if someone reverts the
/// fail-before-touch ordering, `block3_lsn_mismatch_leaves_owners_pristine`
/// above FAILS (owners polluted). We keep this as an executable proof of
/// the loss mechanism (it manually reproduces the pollution).
#[test]
fn block3_revert_pollution_mechanism_loses_data() {
    let dir = tempdir().unwrap();
    let wal = dir.path();
    write_bundles(wal, 5..=20);

    // Manually reproduce the PRE-FIX pollution: seed the TxnManager to the
    // untrusted snapshot LSN (what the old decode_and_restore did before
    // the caller's mismatch check), then anchor "from-zero".
    let p = Owners::fresh();
    p.txn.apply_replay_mvcc_write(
        Lsn::new(20),
        TenantId::DEFAULT,
        5,
        Some(Bytes::from_static(b"v5")),
    );
    p.txn.seed_after_replay(Lsn::new(20)); // ← the pollution
    assert_eq!(p.txn.current_lsn(), Lsn::new(20));

    // "from-zero" anchored replay — but baseline = current_lsn() = 20.
    let report = recover_from_wal_encrypted_anchored(
        wal,
        Arc::clone(&p.txn),
        p.target(),
        None,
        None,
        Lsn::ZERO,
    )
    .unwrap();
    // The polluted baseline silently anchors at 20 → keys ≤20 skipped.
    assert_eq!(
        report.metrics.bundles_applied, 0,
        "polluted baseline skips ALL bundles ≤ untrusted LSN (the loss mechanism)",
    );
    for i in 6..=20 {
        assert_eq!(
            read(&p, i),
            None,
            "key {i} LOST under the pre-fix pollution (proof)"
        );
    }
}

// ─────────────────────────────────────────────────────────────────
// BLOCK-1 (CORRUPTION) — allocator captured consistently ⇒ no id reuse
// ─────────────────────────────────────────────────────────────────

/// After restore, the next-allocated id (Node, Rel, and every Page
/// domain) MUST be STRICTLY ABOVE every id the snapshot restored — an
/// allocator high-water captured before the frontier (or under-counted)
/// would let `alloc_*` re-issue a restored id, aliasing a committed row
/// (ADR-034 D-1 / #129 class). NB-3 fold-in.
///
/// This exercises the encode/restore allocator path directly: a source
/// with committed ids up to a high-water, snapshotted, restored into fresh
/// owners; the restored allocator seed must place the next id above.
/// RED-on-revert: drop the allocator section from the snapshot (or seed a
/// stale high-water) → `alloc_node` returns a restored id.
#[test]
fn block1_next_id_strictly_above_restored_ids_all_domains() {
    let dir = tempdir().unwrap();

    // Source: allocate Node/Rel/Page ids so their high-waters are set,
    // and install MVCC rows at those ids (the "restored" set).
    let src = Owners::fresh();
    let node_hw = 100u64;
    let rel_hw = 50u64;
    // Seed CRUD Node/Rel allocators + PageNode allocator to known highs.
    src.crud.apply_allocator_advance(AllocatorAdvance {
        tenant: TenantId::DEFAULT,
        kind: AllocatorKind::Node,
        new_high_water: node_hw,
    });
    src.crud.apply_allocator_advance(AllocatorAdvance {
        tenant: TenantId::DEFAULT,
        kind: AllocatorKind::Rel,
        new_high_water: rel_hw,
    });
    src.allocator
        .seed_from_advance(TenantId::DEFAULT, arcgraph_core::PageType::Node, 200);
    // A committed node row at the node high-water id.
    src.txn.apply_replay_mvcc_write(
        Lsn::new(7),
        TenantId::DEFAULT,
        node_hw,
        Some(Bytes::from_static(b"node")),
    );
    src.txn.seed_after_replay(Lsn::new(7));

    src.write_checkpoint(dir.path(), Lsn::new(7));

    // Restore into fresh owners.
    let dst = Owners::fresh();
    let seed_r = dst.allocator_seed();
    let restore = restore_latest_checkpoint(dir.path(), &dst.snapshot(seed_r.as_ref()))
        .unwrap()
        .expect("checkpoint found");
    assert_eq!(restore.checkpoint_lsn, Lsn::new(7));

    // Next-allocated id must be STRICTLY ABOVE the restored high-water in
    // EVERY domain — else a restored/committed id is aliased.
    let next_node = dst.crud.alloc_node(TenantId::DEFAULT).unwrap();
    assert!(
        next_node.raw() > node_hw,
        "BLOCK-1: next alloc_node {} must be > restored node high-water {node_hw} (id reuse!)",
        next_node.raw(),
    );
    let next_rel = dst.crud.alloc_rel(TenantId::DEFAULT).unwrap();
    assert!(
        next_rel.raw() > rel_hw,
        "BLOCK-1: next alloc_rel {} must be > restored rel high-water {rel_hw}",
        next_rel.raw(),
    );
    let next_page = dst
        .allocator
        .alloc(TenantId::DEFAULT, arcgraph_core::PageType::Node);
    assert!(
        next_page.raw() > 200,
        "BLOCK-1: next PageNode alloc {} must be > restored page high-water 200",
        next_page.raw(),
    );
}

// ─────────────────────────────────────────────────────────────────
// BLOCK-2 (CORRUPTION) — the commit-freeze makes capture consistent
// ─────────────────────────────────────────────────────────────────

/// The producer holds `TxnManager::checkpoint_freeze` (the commit/checkpoint
/// WRITE guard) across the whole capture. This test proves the guard is
/// mutually exclusive with the commit read-guard: a background thread that
/// holds the commit read side blocks the checkpoint freeze until it
/// releases (and vice-versa), so a commit can never be mid-page-write
/// while the checkpoint captures page images — the BLOCK-2 phantom-record
/// precondition is structurally impossible.
///
/// We assert the exclusion directly (a spawned holder of the read guard
/// delays the freeze) rather than racing a real crash, which is
/// non-deterministic; the exclusion is the invariant that makes the
/// phantom impossible.
#[test]
fn block2_checkpoint_freeze_excludes_commit_readers() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    let txn = Arc::new(TxnManager::new());
    let commit_side_active = Arc::new(AtomicBool::new(false));
    let freeze_acquired_while_commit_active = Arc::new(AtomicBool::new(false));

    // Thread A: hold the COMMIT read guard for a bounded window.
    let txn_a = Arc::clone(&txn);
    let active_a = Arc::clone(&commit_side_active);
    let a = std::thread::spawn(move || {
        let _read = txn_a.__test_commit_read_guard();
        active_a.store(true, Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(150));
        active_a.store(false, Ordering::SeqCst);
        // guard drops here
    });

    // Wait until A holds the read guard.
    while !commit_side_active.load(Ordering::SeqCst) {
        std::thread::sleep(Duration::from_millis(1));
    }

    // Thread B: try to acquire the checkpoint freeze. It MUST block until
    // A releases — so when B finally gets it, A's commit side is inactive.
    let txn_b = Arc::clone(&txn);
    let active_b = Arc::clone(&commit_side_active);
    let observed_b = Arc::clone(&freeze_acquired_while_commit_active);
    let b = std::thread::spawn(move || {
        let _freeze = txn_b.checkpoint_freeze();
        // If the freeze excludes the read guard, the commit side is NOT
        // active at the instant we acquire.
        if active_b.load(Ordering::SeqCst) {
            observed_b.store(true, Ordering::SeqCst);
        }
    });

    a.join().unwrap();
    b.join().unwrap();

    assert!(
        !freeze_acquired_while_commit_active.load(Ordering::SeqCst),
        "BLOCK-2: checkpoint_freeze acquired the WRITE guard while a commit READ guard was held \
         — the exclusion that prevents phantom-page capture is broken",
    );
}

// ─────────────────────────────────────────────────────────────────
// Crash-atomicity — a crash between snapshot + sidecar falls back
// ─────────────────────────────────────────────────────────────────

/// Crash BETWEEN the full-state snapshot durable and the sidecar durable:
/// recovery falls back to the PREVIOUS checkpoint (here: none → from-zero)
/// and loses NO committed data. Simulated by writing the snapshot but NOT
/// the sidecar (the sidecar rename is the establishing step).
#[test]
fn crash_between_snapshot_and_sidecar_falls_back_no_data_loss() {
    const N: u64 = 12;
    let dir = tempdir().unwrap();
    let wal = dir.path();
    write_bundles(wal, 1..=N);
    let p1 = Owners::fresh();
    recover_full(wal, &p1);

    // Write the snapshot but CRASH before the sidecar.
    let seed = p1.allocator_seed();
    let advances = p1.advances();
    write_snapshot_atomic(
        dir.path(),
        &p1.snapshot(seed.as_ref()),
        Lsn::new(N),
        &advances,
    )
    .unwrap();
    assert!(read_latest_sidecar(dir.path()).unwrap().is_none());

    let p2 = Owners::fresh();
    let seed2 = p2.allocator_seed();
    let restore = restore_latest_checkpoint(dir.path(), &p2.snapshot(seed2.as_ref())).unwrap();
    assert!(
        restore.is_none(),
        "a half-checkpoint must NOT be treated as valid"
    );

    let report = recover_from_wal_encrypted_anchored(
        wal,
        Arc::clone(&p2.txn),
        p2.target(),
        None,
        None,
        Lsn::ZERO,
    )
    .unwrap();
    assert_eq!(report.metrics.bundles_applied, N);
    for i in 1..=N {
        assert_eq!(read(&p2, i), Some(Bytes::from(format!("v{i}"))));
    }
}

// ─────────────────────────────────────────────────────────────────
// #1404 M0.5 — STREAMED checkpoint crash-mid-stream (partial temp
// ignored) + recovery byte-equality via the prior established checkpoint.
// ─────────────────────────────────────────────────────────────────

/// #1404 M0.5 — a crash MID-STREAM (at 25% / 50% / 75% of the streamed
/// snapshot) leaves a PARTIAL, un-renamed temp file with NO new sidecar.
/// Recovery MUST ignore the partial temp and fall back to the PRIOR
/// established checkpoint + WAL replay → the recovered graph is byte-equal
/// to the pre-crash state.
///
/// Faithful to the streaming write: `StreamingSnapshotWrite` writes into a
/// process-unique `CHECKPOINT.snap.tmp.<pid>.<seq>` and only renames it to
/// `CHECKPOINT.snap` at `finalize_atomic`. A crash before that rename (and
/// before the sidecar, the ESTABLISH point) leaves the partial temp
/// orphaned — recovery reads `CHECKPOINT.snap` (the prior checkpoint) and
/// the prior sidecar, never the partial temp. We simulate the crash by
/// producing the prior checkpoint via the real streaming producer, then
/// writing a truncated partial temp at each fraction WITHOUT renaming /
/// re-siding, then asserting recovery uses the prior frontier + all data.
///
/// RED-on-revert: if a partial temp were ever consumed by recovery (e.g.
/// if the establish point moved off the sidecar rename), the truncated
/// snapshot would CRC-fail or LSN-mismatch and corrupt/lose data.
#[test]
fn m0_5_crash_mid_stream_partial_temp_ignored_recovers_prior_checkpoint() {
    use arcgraph_storage::checkpoint::checkpoint;

    for pct in [25u64, 50, 75] {
        const N1: u64 = 15; // prior established checkpoint frontier
        const M: u64 = 4; // post-prior-checkpoint WAL tail
        let dir = tempdir().unwrap();
        let wal = dir.path();

        // Populate + establish a PRIOR checkpoint at N1 via the REAL
        // streaming producer (the establish point = its sidecar).
        write_bundles(wal, 1..=N1);
        let p1 = Owners::fresh();
        recover_full(wal, &p1);
        let seed1 = p1.allocator_seed();
        let pool = in_mem_buffer_pool();
        let advances1 = p1.advances();
        let prior = checkpoint(
            dir.path(),
            &pool,
            &p1.snapshot(seed1.as_ref()),
            || advances1.clone(),
            Lsn::new(N1),
        )
        .expect("prior streamed checkpoint");
        assert_eq!(prior.checkpoint_lsn, Lsn::new(N1));

        // More WAL after the prior checkpoint.
        write_bundles(wal, (N1 + 1)..=(N1 + M));

        // Capture the established snapshot bytes so we can build a faithful
        // PARTIAL temp (a prefix of a real streamed snapshot at a higher
        // frontier would look like — a truncated stream).
        let established = std::fs::read(
            dir.path()
                .join(arcgraph_storage::checkpoint::CHECKPOINT_SNAPSHOT_FILE),
        )
        .unwrap();

        // Simulate the crash: a SECOND checkpoint began streaming a fresh
        // snapshot into a unique temp but crashed at `pct`% — leaving a
        // partial temp, NO rename to CHECKPOINT.snap, NO new sidecar. We
        // write that partial temp with the exact unique-temp shape the
        // producer would have used.
        let partial_len = (established.len() as u64 * pct / 100) as usize;
        let partial = &established[..partial_len];
        let partial_tmp = dir.path().join(format!(
            "{}.tmp.{}.999999",
            arcgraph_storage::checkpoint::CHECKPOINT_SNAPSHOT_FILE,
            std::process::id(),
        ));
        std::fs::write(&partial_tmp, partial).unwrap();

        // The prior checkpoint (sidecar + CHECKPOINT.snap) is STILL the only
        // established one — the partial temp has no sidecar pointing at it.
        assert_eq!(
            read_latest_sidecar(dir.path())
                .unwrap()
                .unwrap()
                .checkpoint_lsn,
            Lsn::new(N1),
            "the partial temp must NOT have moved the established frontier ({pct}%)",
        );

        // RECOVERY: restore MUST use the PRIOR checkpoint (N1), never the
        // partial temp, then replay the M-record tail.
        let p2 = Owners::fresh();
        let seed2 = p2.allocator_seed();
        let restore = restore_latest_checkpoint(dir.path(), &p2.snapshot(seed2.as_ref()))
            .unwrap()
            .expect("prior checkpoint must be found ({pct}%)");
        assert_eq!(
            restore.checkpoint_lsn,
            Lsn::new(N1),
            "recovery anchored at the PRIOR frontier, not the partial temp ({pct}%)",
        );

        let report = recover_from_wal_encrypted_anchored(
            wal,
            Arc::clone(&p2.txn),
            p2.target(),
            None,
            None,
            restore.checkpoint_lsn,
        )
        .unwrap();
        assert_eq!(
            report.metrics.bundles_applied, M,
            "only the M post-prior-checkpoint records replay ({pct}%)",
        );
        // Byte-equal to pre-crash: ALL N1+M committed values are visible.
        for i in 1..=(N1 + M) {
            assert_eq!(
                read(&p2, i),
                Some(Bytes::from(format!("v{i}"))),
                "post-crash-mid-stream key {i} ({pct}%)",
            );
        }
    }
}

/// #1404 M0.5 companion — a clean STREAMED checkpoint via the producer,
/// then a crash (drop) + recovery reads the streamed snapshot: recovered
/// graph byte-identical. This is the recovery-byte-equality gate over the
/// streaming write path (the pages/records were streamed, not whole-Vec).
#[test]
fn m0_5_clean_streamed_checkpoint_recovers_byte_equal() {
    use arcgraph_storage::checkpoint::checkpoint;

    const N: u64 = 22;
    const M: u64 = 6;
    let dir = tempdir().unwrap();
    let wal = dir.path();
    write_bundles(wal, 1..=N);
    let p1 = Owners::fresh();
    recover_full(wal, &p1);

    // Real streaming producer establishes the checkpoint.
    let seed = p1.allocator_seed();
    let pool = in_mem_buffer_pool();
    let advances = p1.advances();
    let report = checkpoint(
        dir.path(),
        &pool,
        &p1.snapshot(seed.as_ref()),
        || advances.clone(),
        Lsn::new(N),
    )
    .expect("streamed checkpoint");
    assert_eq!(report.checkpoint_lsn, Lsn::new(N));

    write_bundles(wal, (N + 1)..=(N + M));

    // "Crash" = fresh owners; recovery reads the STREAMED snapshot.
    let p2 = Owners::fresh();
    let seed2 = p2.allocator_seed();
    let restore = restore_latest_checkpoint(dir.path(), &p2.snapshot(seed2.as_ref()))
        .unwrap()
        .expect("streamed checkpoint found");
    assert_eq!(restore.checkpoint_lsn, Lsn::new(N));
    assert_eq!(restore.counts.mvcc_records, N);

    let rep = recover_from_wal_encrypted_anchored(
        wal,
        Arc::clone(&p2.txn),
        p2.target(),
        None,
        None,
        restore.checkpoint_lsn,
    )
    .unwrap();
    assert_eq!(rep.metrics.bundles_applied, M);
    for i in 1..=(N + M) {
        assert_eq!(
            read(&p2, i),
            Some(Bytes::from(format!("v{i}"))),
            "streamed-checkpoint recovery key {i}",
        );
    }
}

/// A sidecar that references a MISSING snapshot also falls back.
#[test]
fn sidecar_without_snapshot_falls_back() {
    let dir = tempdir().unwrap();
    write_sidecar_atomic(
        dir.path(),
        &CheckpointSidecar::full_state(Lsn::new(99), Lsn::new(99), 0),
    )
    .unwrap();
    let owners = Owners::fresh();
    let seed = owners.allocator_seed();
    let restore = restore_latest_checkpoint(dir.path(), &owners.snapshot(seed.as_ref())).unwrap();
    assert!(restore.is_none());
}

// ─────────────────────────────────────────────────────────────────
// No-checkpoint back-compat + round-trip + snapshot-body-corrupt (NB-1c)
// ─────────────────────────────────────────────────────────────────

#[test]
fn no_checkpoint_replays_from_zero() {
    const N: u64 = 8;
    let dir = tempdir().unwrap();
    let wal = dir.path();
    write_bundles(wal, 1..=N);

    let owners = Owners::fresh();
    let seed = owners.allocator_seed();
    let restore = restore_latest_checkpoint(dir.path(), &owners.snapshot(seed.as_ref())).unwrap();
    assert!(restore.is_none());

    let report = recover_from_wal_encrypted_anchored(
        wal,
        Arc::clone(&owners.txn),
        owners.target(),
        None,
        None,
        Lsn::ZERO,
    )
    .unwrap();
    assert_eq!(report.metrics.bundles_applied, N);
    for i in 1..=N {
        assert_eq!(read(&owners, i), Some(Bytes::from(format!("v{i}"))));
    }
}

#[test]
fn checkpoint_roundtrip_latest_valid_used() {
    let dir = tempdir().unwrap();
    let wal = dir.path();
    write_bundles(wal, 1..=10);
    let owners = Owners::fresh();
    recover_full(wal, &owners);

    owners.write_checkpoint(dir.path(), Lsn::new(10));
    assert_eq!(
        read_latest_sidecar(dir.path())
            .unwrap()
            .unwrap()
            .checkpoint_lsn,
        Lsn::new(10),
    );

    write_bundles(wal, 11..=15);
    let owners2 = Owners::fresh();
    recover_full(wal, &owners2);
    owners2.write_checkpoint(dir.path(), Lsn::new(15));

    assert_eq!(
        read_latest_sidecar(dir.path())
            .unwrap()
            .unwrap()
            .checkpoint_lsn,
        Lsn::new(15),
    );
}

/// NB-1c (now a genuinely-safe non-mutating path post-BLOCK-3): a
/// corrupt snapshot body (flipped byte → CRC fails) falls back to
/// from-zero WITHOUT touching owners.
#[test]
fn snapshot_body_corrupt_falls_back_owners_pristine() {
    let dir = tempdir().unwrap();
    let wal = dir.path();
    write_bundles(wal, 1..=6);
    let p1 = Owners::fresh();
    recover_full(wal, &p1);
    p1.write_checkpoint(dir.path(), Lsn::new(6));

    // Flip a byte deep in the snapshot body (past the header) → CRC fails.
    let snap_path = dir
        .path()
        .join(arcgraph_storage::checkpoint::CHECKPOINT_SNAPSHOT_FILE);
    let mut bytes = std::fs::read(&snap_path).unwrap();
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xFF;
    std::fs::write(&snap_path, &bytes).unwrap();

    let p2 = Owners::fresh();
    let seed2 = p2.allocator_seed();
    let restore = restore_latest_checkpoint(dir.path(), &p2.snapshot(seed2.as_ref())).unwrap();
    assert!(restore.is_none(), "CRC-corrupt snapshot → from-zero");
    assert_eq!(
        p2.txn.current_lsn(),
        Lsn::ZERO,
        "owners pristine after CRC fail"
    );

    let report = recover_from_wal_encrypted_anchored(
        wal,
        Arc::clone(&p2.txn),
        p2.target(),
        None,
        None,
        Lsn::ZERO,
    )
    .unwrap();
    assert_eq!(report.metrics.bundles_applied, 6);
    for i in 1..=6 {
        assert_eq!(read(&p2, i), Some(Bytes::from(format!("v{i}"))));
    }
}

/// A full-state snapshot round-trips every owner into fresh owners.
#[test]
fn full_state_snapshot_roundtrips_every_owner() {
    let dir = tempdir().unwrap();

    let src = Owners::fresh();
    src.txn.apply_replay_mvcc_write(
        Lsn::new(5),
        TenantId::DEFAULT,
        42,
        Some(Bytes::from_static(b"row")),
    );
    src.txn.seed_after_replay(Lsn::new(5));
    src.primary
        .install_or_replace(PageId::new(7), mk_page(0xAB))
        .unwrap();
    src.record
        .install_or_replace(PageId::new(9), mk_page(0xCD))
        .unwrap();
    let _blob_ref = src.blob.put(TenantId::DEFAULT, b"hello blob").unwrap();
    let sid = arcgraph_core::StringId::new(3);
    src.intern.intern_install(TenantId::DEFAULT, sid, "Account");
    src.idempotency
        .install(TenantId::DEFAULT, 1, "ext-99", 12345);
    let mut grants = std::collections::BTreeSet::new();
    grants.insert("alice".to_owned());
    src.permissions
        .apply_doc_acl_replayed(arcgraph_core::NodeId::new(77), grants.clone());

    src.write_checkpoint(dir.path(), Lsn::new(5));

    let dst = Owners::fresh();
    let seed_r = dst.allocator_seed();
    let restore = restore_latest_checkpoint(dir.path(), &dst.snapshot(seed_r.as_ref()))
        .unwrap()
        .expect("checkpoint found");
    assert_eq!(restore.checkpoint_lsn, Lsn::new(5));

    assert_eq!(
        dst.txn.read_at(TenantId::DEFAULT, 42, Lsn::new(5)),
        Some(Bytes::from_static(b"row")),
    );
    assert!(
        dst.primary
            .iter_pages()
            .iter()
            .any(|(p, _)| *p == PageId::new(7))
    );
    assert!(
        dst.record
            .iter_pages()
            .iter()
            .any(|(p, _)| *p == PageId::new(9))
    );
    assert!(!dst.blob.iter_pages().unwrap().is_empty());
    assert_eq!(
        dst.intern
            .try_resolve(TenantId::DEFAULT, sid)
            .unwrap()
            .as_deref()
            .map(String::as_str),
        Some("Account"),
    );
    assert_eq!(
        dst.idempotency
            .get(TenantId::DEFAULT, 1, "ext-99")
            .map(|b| b.internal_id),
        Some(12345),
    );
    assert!(
        dst.permissions
            .effective("alice")
            .is_visible(arcgraph_core::NodeId::new(77))
    );
    assert_eq!(restore.counts.mvcc_records, 1);
    assert_eq!(restore.counts.permission_docs, 1);
}

/// End-to-end producer path: the real `checkpoint::checkpoint` (with the
/// commit-freeze + buffer-pool flush) establishes a checkpoint, and an
/// anchored restart replays only the post-frontier tail.
#[test]
fn producer_checkpoint_then_anchored_restart() {
    const N: u64 = 10;
    const M: u64 = 3;
    let dir = tempdir().unwrap();
    let wal = dir.path();
    write_bundles(wal, 1..=N);
    let p1 = Owners::fresh();
    recover_full(wal, &p1);

    // Real producer path (freeze + flush + snapshot + sidecar).
    let seed = p1.allocator_seed();
    let pool = in_mem_buffer_pool();
    let advances = p1.advances();
    let report = arcgraph_storage::checkpoint::checkpoint(
        dir.path(),
        &pool,
        &p1.snapshot(seed.as_ref()),
        || advances.clone(),
        Lsn::new(N),
    )
    .unwrap();
    assert_eq!(report.checkpoint_lsn, Lsn::new(N));

    write_bundles(wal, (N + 1)..=(N + M));

    let p2 = Owners::fresh();
    let seed2 = p2.allocator_seed();
    let restore = restore_latest_checkpoint(dir.path(), &p2.snapshot(seed2.as_ref()))
        .unwrap()
        .expect("checkpoint found");
    let rep = recover_from_wal_encrypted_anchored(
        wal,
        Arc::clone(&p2.txn),
        p2.target(),
        None,
        None,
        restore.checkpoint_lsn,
    )
    .unwrap();
    assert_eq!(rep.metrics.bundles_applied, M);
    for i in 1..=(N + M) {
        assert_eq!(read(&p2, i), Some(Bytes::from(format!("v{i}"))));
    }
}

// ─────────────────────────────────────────────────────────────────
// REQ-1 (ULTRACODE re-verify) — the SECOND commit path
// (commit_index_pages_atomic) takes the checkpoint read guard, closing
// the BLOCK-1 index-page id-reuse + BLOCK-2 torn-index residuals.
// ─────────────────────────────────────────────────────────────────

/// Helper: build a StagedEmit carrying an IndexPage image at `page_id`.
fn index_staged(page_id: u64, fill: u8) -> StagedEmit {
    StagedEmit {
        kind: BundlePageKind::PrimaryIndex,
        page_id: PageId::new(page_id),
        bytes: mk_page(fill),
    }
}

/// REQ-1(a+b): the standalone index commit path (`commit_index_pages_atomic`
/// — the grow_root / SYSTEM root-pointer path) MUST take the checkpoint
/// READ guard for its full span, so a concurrent checkpoint that holds the
/// WRITE freeze BLOCKS the index commit until it releases. Without the
/// guard (the pre-REQ-1 state) the index commit runs WHILE the checkpoint
/// captures → an IndexPage can be captured whose SYSTEM/IndexLeaf
/// allocator high-water is NOT captured → B-tree page aliasing on restart.
///
/// We assert the exclusion directly against the REAL commit path (not the
/// synthetic proxy): while a thread holds `checkpoint_freeze`, a
/// `commit_index_pages_atomic` call does NOT complete; it completes only
/// after the freeze drops.
///
/// RED-on-revert: remove `let _checkpoint_read = self.checkpoint_lock.read();`
/// from `commit_index_pages_atomic` → the index commit completes WHILE the
/// freeze is held → `committed_while_frozen` is observed true → FAIL.
#[test]
fn req1_index_commit_path_blocks_under_checkpoint_freeze() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    let txn = Arc::new(TxnManager::new());
    let freeze_held = Arc::new(AtomicBool::new(false));
    let committed_while_frozen = Arc::new(AtomicBool::new(false));

    // Thread A: hold the checkpoint WRITE freeze for a window.
    let txn_a = Arc::clone(&txn);
    let held_a = Arc::clone(&freeze_held);
    let a = std::thread::spawn(move || {
        let _freeze = txn_a.checkpoint_freeze();
        held_a.store(true, Ordering::SeqCst);
        std::thread::sleep(Duration::from_millis(200));
        held_a.store(false, Ordering::SeqCst);
        // freeze drops here
    });

    while !freeze_held.load(Ordering::SeqCst) {
        std::thread::sleep(Duration::from_millis(1));
    }

    // Thread B: a REAL standalone index commit (no WAL → in-memory only).
    // With the read guard it MUST block until A's freeze drops.
    let txn_b = Arc::clone(&txn);
    let held_b = Arc::clone(&freeze_held);
    let observed_b = Arc::clone(&committed_while_frozen);
    let b = std::thread::spawn(move || {
        let staged = [index_staged(500, 0x11)];
        let sc = [SideChannelWrite {
            tenant_id: TenantId::SYSTEM,
            key: 1, // PRIMARY_INDEX_ROOT_KEY
            value: Some(Bytes::from_static(b"root")),
        }];
        txn_b
            .commit_index_pages_atomic(None, &staged, &sc)
            .expect("index commit");
        // If the commit returned while A's freeze was still held, the
        // second path did NOT take the read guard (the REQ-1 bug).
        if held_b.load(Ordering::SeqCst) {
            observed_b.store(true, Ordering::SeqCst);
        }
    });

    a.join().unwrap();
    b.join().unwrap();

    assert!(
        !committed_while_frozen.load(Ordering::SeqCst),
        "REQ-1: commit_index_pages_atomic COMPLETED while a checkpoint freeze was held — the \
         second commit path does not take the checkpoint read guard (index-page id-reuse / \
         torn-index-capture window is OPEN)",
    );
}

/// REQ-1(a): BLOCK-1 coverage for the SYSTEM / IndexLeaf page domain
/// (the original block1 test only covered Node/Rel/PageType::Node). After a
/// checkpoint captures a SYSTEM IndexLeaf allocator high-water, restore must
/// place the next IndexLeaf page id STRICTLY ABOVE every restored index page.
#[test]
fn req1_next_index_leaf_page_id_above_restored() {
    let dir = tempdir().unwrap();

    let src = Owners::fresh();
    // Seed the SYSTEM IndexLeaf allocator high-water + install an index
    // page (record-store page domain stands in for the index page image).
    src.allocator
        .seed_from_advance(TenantId::SYSTEM, arcgraph_core::PageType::IndexLeaf, 300);
    src.txn.seed_after_replay(Lsn::new(3));
    src.write_checkpoint(dir.path(), Lsn::new(3));

    let dst = Owners::fresh();
    let seed_r = dst.allocator_seed();
    let restore = restore_latest_checkpoint(dir.path(), &dst.snapshot(seed_r.as_ref()))
        .unwrap()
        .expect("checkpoint found");
    assert_eq!(restore.checkpoint_lsn, Lsn::new(3));

    let next_leaf = dst
        .allocator
        .alloc(TenantId::SYSTEM, arcgraph_core::PageType::IndexLeaf);
    assert!(
        next_leaf.raw() > 300,
        "REQ-1: next SYSTEM IndexLeaf page id {} must be > restored high-water 300 (index-page \
         aliasing!)",
        next_leaf.raw(),
    );
}

/// REQ-1(c): a REAL concurrent-writer test on the index commit path racing
/// a real `checkpoint()`. A writer loop fires `commit_index_pages_atomic`
/// (durable, with WAL) while a checkpoint runs; after restart, (a) all
/// committed index roots survive and (b) no phantom/torn index page. The
/// commit-freeze read/write exclusion makes a mid-stage capture impossible.
#[test]
fn req1_concurrent_index_commit_vs_checkpoint_no_phantom() {
    use std::sync::atomic::{AtomicBool, Ordering};

    let dir = tempdir().unwrap();
    let wal_dir = dir.path().join("wal");
    std::fs::create_dir_all(&wal_dir).unwrap();

    // A durable TxnManager + WAL so commits fsync.
    let writer = WalWriter::spawn(wal_cfg(&wal_dir)).unwrap();
    let handle = writer.handle();
    let owners = Owners::fresh();
    owners.txn.attach_wal(handle.clone());

    let stop = Arc::new(AtomicBool::new(false));

    // Writer thread: repeatedly commit a SYSTEM root-pointer + index page
    // via the standalone index path (now guarded by checkpoint_lock.read).
    let txn_w = Arc::clone(&owners.txn);
    let handle_w = handle.clone();
    let stop_w = Arc::clone(&stop);
    let w = std::thread::spawn(move || {
        let mut n = 0u64;
        while !stop_w.load(Ordering::SeqCst) {
            n += 1;
            let staged = [index_staged(600 + n, (n % 250) as u8)];
            let sc = [SideChannelWrite {
                tenant_id: TenantId::SYSTEM,
                key: 1,
                value: Some(Bytes::from(format!("root{n}"))),
            }];
            let _ = txn_w.commit_index_pages_atomic(Some(&handle_w), &staged, &sc);
        }
        n
    });

    // Main: run several checkpoints concurrently with the writer.
    let seed = owners.allocator_seed();
    let pool = in_mem_buffer_pool();
    for _ in 0..5 {
        let advances = owners.advances();
        let _ = arcgraph_storage::checkpoint::checkpoint(
            dir.path(),
            &pool,
            &owners.snapshot(seed.as_ref()),
            || advances.clone(),
            owners.txn.current_lsn(),
        )
        .expect("checkpoint");
        std::thread::sleep(std::time::Duration::from_millis(5));
    }

    stop.store(true, Ordering::SeqCst);
    let committed = w.join().unwrap();
    writer.shutdown().unwrap();
    assert!(
        committed > 0,
        "writer must have committed at least one index op"
    );

    // The last established checkpoint's frontier must be consistent (no
    // panic / no torn capture). Re-open + restore proves the snapshot is
    // well-formed and the SYSTEM root is present at the frontier.
    let p2 = Owners::fresh();
    let seed2 = p2.allocator_seed();
    let restore = restore_latest_checkpoint(dir.path(), &p2.snapshot(seed2.as_ref()))
        .unwrap()
        .expect("a checkpoint must be established");
    // The restored SYSTEM root pointer (key 1) is visible at the frontier
    // (a phantom / torn capture would corrupt or omit it).
    let root = p2.txn.read_at(TenantId::SYSTEM, 1, restore.checkpoint_lsn);
    assert!(
        root.is_some(),
        "REQ-1(c): the SYSTEM root pointer must be present + consistent in the checkpoint \
         (a torn/phantom index capture would omit or corrupt it)",
    );
}

// ─────────────────────────────────────────────────────────────────
// REQ-2 (ULTRACODE re-verify) — the checkpoint freeze does NO disk
// fault-in (resident-only capture); the availability regression is closed.
// ─────────────────────────────────────────────────────────────────

/// REQ-2: the resident-only iterators return the pages WITHOUT faulting,
/// and report ZERO evicted ids for the wired pure-DashMap stores — so the
/// encode-under-freeze touches no disk. This is the assertion the verdict
/// asked for: the write-guard hold-time does not include disk fault-in.
#[test]
fn req2_resident_only_capture_reports_no_evicted_no_fault() {
    let owners = Owners::fresh();
    // Install some pages in each store.
    owners
        .primary
        .install_or_replace(PageId::new(1), mk_page(0x01))
        .unwrap();
    owners
        .primary
        .install_or_replace(PageId::new(2), mk_page(0x02))
        .unwrap();
    owners
        .record
        .install_or_replace(PageId::new(3), mk_page(0x03))
        .unwrap();
    let _ = owners.blob.put(TenantId::DEFAULT, b"blob bytes").unwrap();

    // The resident-only iterators return ALL pages as resident + ZERO
    // evicted (no disk read) — the property that keeps the freeze fault-free.
    let (primary_res, primary_evicted) = owners.primary.iter_pages_resident_only();
    assert_eq!(primary_res.len(), 2, "all primary pages resident");
    assert!(
        primary_evicted.is_empty(),
        "REQ-2: primary must report ZERO evicted (no fault)"
    );

    let (record_res, record_evicted) = owners.record.iter_pages_resident_only();
    assert_eq!(record_res.len(), 1, "all record pages resident");
    assert!(
        record_evicted.is_empty(),
        "REQ-2: record must report ZERO evicted (no fault)"
    );

    let (blob_res, blob_evicted) = owners.blob.iter_pages_resident_only();
    assert!(!blob_res.is_empty(), "blob pages resident");
    assert!(
        blob_evicted.is_empty(),
        "REQ-2: blob must report ZERO evicted (no fault)"
    );
}

/// REQ-2 end-to-end: a real `checkpoint()` over stores with pages produces
/// a valid snapshot whose evicted-supplement is empty (nothing faulted
/// under the guard), and an anchored restart round-trips every page. Proves
/// the resident-only refactor did not break the format or lose pages.
#[test]
fn req2_producer_resident_only_roundtrips_pages() {
    let dir = tempdir().unwrap();
    let owners = Owners::fresh();
    owners
        .primary
        .install_or_replace(PageId::new(11), mk_page(0xAA))
        .unwrap();
    owners
        .record
        .install_or_replace(PageId::new(12), mk_page(0xBB))
        .unwrap();
    let _ = owners.blob.put(TenantId::DEFAULT, b"blobby").unwrap();
    owners.txn.apply_replay_mvcc_write(
        Lsn::new(4),
        TenantId::DEFAULT,
        7,
        Some(Bytes::from_static(b"r")),
    );
    owners.txn.seed_after_replay(Lsn::new(4));

    let seed = owners.allocator_seed();
    let pool = in_mem_buffer_pool();
    let advances = owners.advances();
    let report = arcgraph_storage::checkpoint::checkpoint(
        dir.path(),
        &pool,
        &owners.snapshot(seed.as_ref()),
        || advances.clone(),
        Lsn::new(4),
    )
    .expect("checkpoint");
    assert_eq!(report.checkpoint_lsn, Lsn::new(4));

    let dst = Owners::fresh();
    let seed_r = dst.allocator_seed();
    let restore = restore_latest_checkpoint(dir.path(), &dst.snapshot(seed_r.as_ref()))
        .unwrap()
        .expect("checkpoint found");
    assert_eq!(restore.checkpoint_lsn, Lsn::new(4));
    assert!(
        dst.primary
            .iter_pages()
            .iter()
            .any(|(p, _)| *p == PageId::new(11))
    );
    assert!(
        dst.record
            .iter_pages()
            .iter()
            .any(|(p, _)| *p == PageId::new(12))
    );
    assert!(!dst.blob.iter_pages().unwrap().is_empty());
    assert_eq!(
        dst.txn.read_at(TenantId::DEFAULT, 7, Lsn::new(4)),
        Some(Bytes::from_static(b"r"))
    );
}
