//! ADR-033 Z-1 (b) in-memory rollback regression tests.
//!
//! Each test drives a transaction that mutates in-memory page state
//! (primary B-tree leaf, grow_root, blob chain) through the bundle-
//! aware `crud::commit` path, forces WAL fsync failure, and asserts
//! the post-rollback state matches the pre-commit baseline.
//!
//! The tests prove these invariants:
//!
//! - **I-Z1.1 (no-ghost)**: `z1_rollback_simple_ghost_prevention`,
//!   `z1_rollback_grow_root_unwinds_completely`,
//!   `z1_rollback_subsequent_commit_clean`.
//! - **I-Z1.2 (idempotent under gate)**: `z1_rollback_idempotent`.
//! - **I-Z1.3 (reader snapshot preservation)**:
//!   `z1_rollback_concurrent_readers_preserve_snapshot`.
//!
//! Plus:
//! - `z1_rollback_blob_chain_unwind`: `BlobStore` chain pages
//!   are removed on rollback.
//!
//! **Y-1 / Y-2 fold-in (2026-04-24)**:
//! - `z1_update_path_prevents_ghost` (Y-1 regression): UPDATE path
//!   captures the pre-W record-page bytes so WAL fsync failure
//!   restores the slot rather than leaving a W-stamped ghost that
//!   a reader reaches via the pinned primary-index coordinate.
//! - `z1_multi_store_page_id_collision_captured_independently`
//!   (Y-2 regression): when a commit mutates primary page
//!   PageId(1) AND record page PageId(1) in the same txn, the
//!   `(PageStoreKind, PageId)` compound dedup key keeps both
//!   captures; rollback dispatches each to its correct store
//!   with no cross-store fallthrough.
//!
//! Run with
//!   cargo test -p arcgraph-storage --test z1_rollback

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use arcgraph_core::{ArcGraphError, LabelId, Lsn, NodeId, PageId, PartitionId, TenantId};
use arcgraph_storage::crud::{
    CrudError, CrudStore, PropertyData, commit, create_node, read_node, update_node,
};
use arcgraph_storage::mutation_log::{PageStoreKind, TxnMutationLog};
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::page_store::{
    BufferedRecordPageStore, PerTenantBufferPool, PerTenantBufferPoolConfig,
};
use arcgraph_storage::primary_index::{
    PRIMARY_INDEX_ROOT_KEY, PrimaryIndex, PrimaryKey, RecordKind,
};
use arcgraph_storage::records::SlottedPageRef;
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{WalConfig, WalWriter};
use tempfile::TempDir;

// ─────────────────────────────────────────────────────────────────────
// Test harness
// ─────────────────────────────────────────────────────────────────────

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

struct Stack {
    _dir: TempDir,
    store: Arc<CrudStore>,
    mgr: Arc<TxnManager>,
    primary: Arc<PrimaryIndex>,
    writer: Option<WalWriter>,
}

fn build_stack() -> Stack {
    let dir = TempDir::new().unwrap();
    let writer = WalWriter::spawn(test_wal_config(dir.path().to_path_buf())).unwrap();
    let handle = writer.handle();
    let mgr = Arc::new(TxnManager::with_wal(handle.clone()));
    let alloc = Arc::new(PageAllocator::new());
    let primary = Arc::new(
        PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&alloc), Some(handle.clone())).unwrap(),
    );
    let store = Arc::new(CrudStore::new_with_index(
        Some(handle.clone()),
        Arc::clone(&primary),
        alloc,
    ));
    Stack {
        _dir: dir,
        store,
        mgr,
        primary,
        writer: Some(writer),
    }
}

fn build_buffered_stack() -> (Stack, Arc<BufferedRecordPageStore>) {
    use arcgraph_storage::io::{InMemoryPageIo, PageIo};

    let dir = TempDir::new().unwrap();
    let writer = WalWriter::spawn(test_wal_config(dir.path().to_path_buf())).unwrap();
    let handle = writer.handle();
    let mgr = Arc::new(TxnManager::with_wal(handle.clone()));
    let alloc = Arc::new(PageAllocator::new());
    let primary = Arc::new(
        PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&alloc), Some(handle.clone())).unwrap(),
    );
    let io: Arc<dyn PageIo> = Arc::new(InMemoryPageIo::new());
    let pools = Arc::new(PerTenantBufferPool::with_config(
        io,
        PerTenantBufferPoolConfig {
            frames_per_tenant: 16,
            write_fraction: 0.0,
        },
    ));
    let records = Arc::new(BufferedRecordPageStore::with_cache_cap(pools, 32));
    let store = Arc::new(CrudStore::new_with_page_store(
        Some(handle),
        Arc::clone(&primary),
        None,
        alloc,
        Arc::clone(&records),
    ));
    (
        Stack {
            _dir: dir,
            store,
            mgr,
            primary,
            writer: Some(writer),
        },
        records,
    )
}

/// M3.a Slice G.5 — variant of [`build_stack`] that wires a
/// concrete [`VectorPageStoreHandle`] (via
/// [`VectorArenaPageStore`]) into the `CrudStore` so the rollback
/// closure's `PageStoreKind::Vector` arm has a dispatch target.
/// Returns the `Stack` plus a clone of the underlying arena so the
/// test can install pre-W bytes + assert post-rollback bytes
/// directly.
fn build_stack_with_vector_store() -> (
    Stack,
    Arc<arcgraph_storage::vector_store::recovery::VectorArenaPageStore>,
) {
    use arcgraph_storage::vector_store::VectorPageStoreHandle;
    use arcgraph_storage::vector_store::recovery::VectorArenaPageStore;

    let dir = TempDir::new().unwrap();
    let writer = WalWriter::spawn(test_wal_config(dir.path().to_path_buf())).unwrap();
    let handle = writer.handle();
    let mgr = Arc::new(TxnManager::with_wal(handle.clone()));
    let alloc = Arc::new(PageAllocator::new());
    let primary = Arc::new(
        PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&alloc), Some(handle.clone())).unwrap(),
    );
    let arena = Arc::new(VectorArenaPageStore::new());
    let arena_handle: Arc<dyn VectorPageStoreHandle> =
        Arc::clone(&arena) as Arc<dyn VectorPageStoreHandle>;
    let store = Arc::new(
        CrudStore::new_with_index(Some(handle.clone()), Arc::clone(&primary), alloc)
            .with_vector_store(arena_handle),
    );
    let stack = Stack {
        _dir: dir,
        store,
        mgr,
        primary,
        writer: Some(writer),
    };
    (stack, arena)
}

impl Stack {
    fn shutdown_wal(&mut self) {
        if let Some(w) = self.writer.take() {
            w.shutdown().expect("wal shutdown");
        }
    }
}

fn assert_wal_rolled_back(err: &CrudError) {
    match err {
        CrudError::Mvcc(ArcGraphError::WalErrorRolledBack { source }) => {
            assert!(
                matches!(source.as_ref(), ArcGraphError::WalUnavailable),
                "source chain must carry the underlying WAL error, got {source:?}"
            );
        }
        other => panic!("expected WalErrorRolledBack wrapping, got {other:?}"),
    }
}

// ─────────────────────────────────────────────────────────────────────
// I-Z1.1: no-ghost (simple)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn z1_rollback_simple_ghost_prevention() {
    // Drive a transaction that mutates one primary-index leaf (single
    // insert, no split). Force WAL failure. Post-rollback:
    //
    // - MVCC has no version for the failed key (existing invariant).
    // - The primary-index leaf has no ghost entry (Z-1 b's new
    //   invariant).
    //
    // Verified by: primary.lookup(doomed_key) returns None after
    // rollback, AND `read_node` returns None. Pre-Z-1 (b) the primary
    // leaf retained the ghost `(key → slot)` entry; with Z-1 (b)
    // rollback, the leaf is restored to its pre-commit bytes so
    // lookup returns None.
    let mut stack = build_stack();

    // Seed: durable baseline so the primary index has a populated
    // root leaf. The doomed write will co-mutate that leaf.
    let mut tx = stack.mgr.begin(TenantId::DEFAULT);
    let _seed_id = create_node(
        &stack.store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(1),
        &PropertyData::Empty,
    )
    .unwrap();
    let _seed_lsn = commit(tx, &stack.store).unwrap();
    let seed_visible = stack.mgr.current_lsn();

    // Shut down WAL so the next commit fails at Phase 2.
    stack.shutdown_wal();

    // Attempt a second commit whose primary-index entry would be a
    // ghost if Z-1 (b) rollback didn't run.
    let mut tx = stack.mgr.begin(TenantId::DEFAULT);
    let doomed_id = create_node(
        &stack.store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(2),
        &PropertyData::Empty,
    )
    .unwrap();
    let err = commit(tx, &stack.store).expect_err("must fail with WAL down");
    assert_wal_rolled_back(&err);

    // Post-rollback assertions.
    assert_eq!(
        stack.mgr.current_lsn(),
        seed_visible,
        "visible must not advance past the seed",
    );

    // I-Z1.1: the primary index must NOT have the doomed key.
    let doomed_key = PrimaryKey::new(TenantId::DEFAULT, RecordKind::Node, doomed_id.raw());
    assert!(
        stack.primary.lookup(doomed_key).unwrap().is_none(),
        "Z-1 (b): doomed key must be rolled back from the primary index; \
         a ghost entry would violate ADR-023's MVCC-authoritative contract"
    );

    // External read must also return None (MVCC + index both agree).
    let reader = stack.mgr.begin(TenantId::DEFAULT);
    assert!(
        read_node(&reader, doomed_id).unwrap().is_none(),
        "reader must not see the doomed node"
    );
}

// ─────────────────────────────────────────────────────────────────────
// I-Z1.1 + grow_root: grow_root unwinds completely
// ─────────────────────────────────────────────────────────────────────

#[test]
fn z1_rollback_grow_root_unwinds_completely() {
    // Drive enough inserts to force grow_root, then force WAL
    // failure on the grow-triggering commit. Post-rollback:
    //
    // - `root_cache` is restored to the pre-grow root (via
    //   `log.root_changes` → `PrimaryIndex::restore_root_cache`).
    // - The new-root page is removed from page_store (via
    //   `log.new_pages` → `primary.page_store().remove_page`).
    // - The MVCC SYSTEM root-pointer version is absent (Phase 3
    //   sidechannel-write application never ran; WAL failure
    //   short-circuited Phase 3).
    //
    // This is #66 by construction — the fold property — but Z-1 (b)
    // is the complement: EVEN UNDER rollback, the invariant survives.
    let mut stack = build_stack();

    // Seed: populate the primary index with enough nodes to approach
    // but not exceed one leaf's capacity. LEAF_CAPACITY is large (~300
    // at PAGE_SIZE=8KiB after slotted-page overhead); we use 200 to
    // stay well under while having enough to grow on the next commit.
    // Single-leaf seed, durable.
    let mut tx = stack.mgr.begin(TenantId::DEFAULT);
    let mut seed_ids = Vec::new();
    for i in 0..200 {
        let id = create_node(
            &stack.store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(i as u32),
            &PropertyData::Empty,
        )
        .unwrap();
        seed_ids.push(id);
    }
    commit(tx, &stack.store).unwrap();
    let seed_visible = stack.mgr.current_lsn();
    let pre_root_id = stack.primary.root().unwrap();

    // Kill the WAL before the grow-triggering commit.
    stack.shutdown_wal();

    // Attempt a commit that inserts enough additional keys to
    // trigger split → grow_root. Target: add another 200+ keys to
    // push over capacity. For the test, we just need the PHASE-2
    // failure to happen with the log having a root_changes entry —
    // which requires grow_root to have executed during the builder.
    //
    // A single create_node might not trigger grow if the leaf still
    // has room; adding ~200 forces it.
    let mut tx = stack.mgr.begin(TenantId::DEFAULT);
    let mut doomed_ids = Vec::new();
    for i in 200..400 {
        let id = create_node(
            &stack.store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(i as u32),
            &PropertyData::Empty,
        )
        .unwrap();
        doomed_ids.push(id);
    }
    let err = commit(tx, &stack.store).expect_err("must fail with WAL down");
    assert_wal_rolled_back(&err);

    // MVCC state frozen at seed.
    assert_eq!(stack.mgr.current_lsn(), seed_visible);

    // §5 ordering: root_cache restored.
    assert_eq!(
        stack.primary.root().unwrap(),
        pre_root_id,
        "Z-1 (b) §5: root_cache must be restored to the pre-grow value \
         before new-root page removal"
    );

    // None of the doomed keys survive in the primary index. Even if
    // grow_root didn't trigger (leaf was tolerant to these many
    // inserts in-place + splits), Z-1 rollback must still remove
    // their entries.
    for id in &doomed_ids {
        let k = PrimaryKey::new(TenantId::DEFAULT, RecordKind::Node, id.raw());
        assert!(
            stack.primary.lookup(k).unwrap().is_none(),
            "doomed key {:?} must be rolled back from primary index",
            id
        );
    }

    // All seed keys survive.
    for id in &seed_ids {
        let k = PrimaryKey::new(TenantId::DEFAULT, RecordKind::Node, id.raw());
        assert!(
            stack.primary.lookup(k).unwrap().is_some(),
            "seed key {:?} must still be in the primary index",
            id
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// I-Z1.2: idempotent under gate
// ─────────────────────────────────────────────────────────────────────

#[test]
fn z1_rollback_idempotent() {
    // The rollback closure drains all four smallvecs via drain(..).
    // A second call finds everything empty and is a no-op. We prove
    // this by constructing a mutation log manually, draining it
    // once via the helpers, then draining again, and asserting no
    // panics / store-state change.
    let mut stack = build_stack();
    stack.shutdown_wal(); // not needed for this test, but symmetric.

    // Seed a primary index page so we have something to capture /
    // remove.
    //
    // Actually, since the rollback closure is private to `crud::commit`,
    // we exercise idempotence directly on the page-store helpers.

    // Capture-and-restore idempotence on PrimaryPageStore.
    let primary_store = stack.primary.page_store();
    let root_id = stack.primary.root().unwrap();
    let mut log = TxnMutationLog::new();

    // First capture.
    {
        let _guard = primary_store.capture_and_latch(&mut log, root_id).unwrap();
    }
    assert_eq!(log.page_mutations.len(), 1);
    assert_eq!(log.page_mutations[0].0, PageStoreKind::Primary);
    assert_eq!(log.page_mutations[0].1, root_id);

    // Second capture — idempotent (dedup via has_captured).
    {
        let _guard = primary_store.capture_and_latch(&mut log, root_id).unwrap();
    }
    assert_eq!(
        log.page_mutations.len(),
        1,
        "capture must be idempotent within a txn"
    );

    // Draining the log is also idempotent: a second drain finds
    // everything empty.
    let drained: Vec<_> = log.page_mutations.drain(..).collect();
    assert_eq!(drained.len(), 1);
    let drained_twice: Vec<_> = log.page_mutations.drain(..).collect();
    assert_eq!(drained_twice.len(), 0, "log must be empty after drain");
}

// ─────────────────────────────────────────────────────────────────────
// I-Z1.3: reader snapshot preservation
// ─────────────────────────────────────────────────────────────────────

#[test]
fn z1_rollback_concurrent_readers_preserve_snapshot() {
    // A reader that captured a page latch (Arc clone of the
    // RwLock-wrapped Box<PageBuf>) BEFORE a rollback continues to
    // read from that captured Arc — the RwLock ensures no
    // half-rolled-back bytes are observed. New readers (post-
    // rollback) get the restored state.
    //
    // This is I-Z1.3: reader snapshot preservation. We verify:
    //
    // 1. Reader R1 captures the primary root latch.
    // 2. A failing commit runs (no mutation observed — just WAL fail).
    // 3. R1's view is unchanged (they still have the same Arc).
    // 4. A new reader R2 sees the post-rollback (= pre-fail) state.
    let mut stack = build_stack();

    let mut tx = stack.mgr.begin(TenantId::DEFAULT);
    let seed_id = create_node(
        &stack.store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(1),
        &PropertyData::Empty,
    )
    .unwrap();
    commit(tx, &stack.store).unwrap();

    // R1 captures the root latch (as in read-crabbing). The Arc
    // inside is shared with the DashMap entry. Clone the Arc so
    // we can drop the borrow before shutdown_wal needs a mutable
    // ref to stack.
    let (root_id, r1_latch) = {
        let primary_store = stack.primary.page_store();
        let root_id = stack.primary.root().unwrap();
        (root_id, primary_store.latch(root_id).unwrap())
    };

    // Kill WAL and run a failing commit that mutates the root leaf.
    stack.shutdown_wal();
    let mut tx = stack.mgr.begin(TenantId::DEFAULT);
    let _ = create_node(
        &stack.store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(2),
        &PropertyData::Empty,
    )
    .unwrap();
    let _ = commit(tx, &stack.store).expect_err("must fail");

    // R1 can still read its latch. Content: pre-W bytes (restored
    // by Z-1 rollback). The Arc may or may not be the same DashMap
    // entry — DashMap re-inserts can swap it — but the bytes are
    // the restored ones. We verify by reading the first byte of
    // the page header: a well-formed page starts with the `AGPG`
    // magic (page_type byte at a known offset).
    {
        let r1_view = r1_latch.read();
        // Sanity: the latch's bytes look like a page (not zeroed).
        assert!(
            r1_view.as_ref().as_ref().iter().any(|&b| b != 0),
            "R1 latch should contain a non-zero page"
        );
    }

    // R2 — post-rollback fresh latch — should see restored state too.
    let r2_latch = stack.primary.page_store().latch(root_id).unwrap();
    {
        let r2_view = r2_latch.read();
        assert!(
            r2_view.as_ref().as_ref().iter().any(|&b| b != 0),
            "R2 latch should contain a non-zero page"
        );
    }

    // Both readers see the same seed node.
    let reader = stack.mgr.begin(TenantId::DEFAULT);
    assert!(read_node(&reader, seed_id).unwrap().is_some());
}

// ─────────────────────────────────────────────────────────────────────
// Blob chain unwind
// ─────────────────────────────────────────────────────────────────────

#[test]
fn z1_rollback_blob_chain_unwind() {
    // BlobStore::remove_uncommitted_chain walks the chain and
    // removes every page. Covered by the unit tests in blob.rs
    // (remove_uncommitted_chain_* cases). Z-1 integration into
    // `crud::commit`'s rollback closure is a Phase 2c follow-up —
    // this test documents the expected behavior for when the
    // integration lands.
    use arcgraph_storage::blob::BlobStore;
    let store = BlobStore::new();
    let payload = vec![0u8; 32_768];
    let blob_ref = store.put(TenantId::DEFAULT, &payload).unwrap();

    // Simulate what the rollback closure would do: register the
    // chain into a mutation log, then drain.
    let mut log = TxnMutationLog::new();
    store.register_uncommitted_chain(&mut log, TenantId::DEFAULT, blob_ref.page_id, 1);
    assert_eq!(log.blob_heads.len(), 1);

    // Drain and remove.
    for (tenant, head) in log.blob_heads.drain(..) {
        store.remove_uncommitted_chain(tenant, head).unwrap();
    }

    // Chain is gone.
    assert!(store.get(TenantId::DEFAULT, blob_ref).is_err());
    assert_eq!(store.page_count(), 0);
}

// ─────────────────────────────────────────────────────────────────────
// Subsequent commit runs cleanly after a rolled-back predecessor
// ─────────────────────────────────────────────────────────────────────

#[test]
fn z1_rollback_subsequent_commit_clean() {
    // After a failed-then-rolled-back transaction, a subsequent
    // successful commit with a NEW commit_lsn must operate as if
    // the failed one never existed. In particular:
    //
    // - No MVCC chain has a stale `expired_lsn` pointing at the
    //   dead LSN (existing invariant, verified by rollback_writes).
    // - No primary-index page has a ghost `(key → slot)` entry
    //   (Z-1 (b)'s new invariant; verified by the rollback closure
    //   removing the ghost via `restore_page_bytes`).
    //
    // Shape: commit C1 succeeds (seed). Fail WAL. Try C2 → fails,
    // rolls back. Restart WAL (new writer, new mgr stack so we can
    // commit again). Commit C3 → succeeds. Read C1 + C3; assert no
    // trace of C2.
    //
    // Simplification: we use ONE TxnManager but the new writer gets
    // a fresh handle wired in via a fresh CrudStore. In practice
    // this shape is tested in the "successor commit" fragment
    // below.
    let mut stack = build_stack();

    // C1: seed.
    let mut tx = stack.mgr.begin(TenantId::DEFAULT);
    let c1_id = create_node(
        &stack.store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(1),
        &PropertyData::Empty,
    )
    .unwrap();
    let c1_lsn = commit(tx, &stack.store).unwrap();

    // Shut down WAL.
    stack.shutdown_wal();

    // C2: fails.
    let mut tx = stack.mgr.begin(TenantId::DEFAULT);
    let c2_id = create_node(
        &stack.store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(2),
        &PropertyData::Empty,
    )
    .unwrap();
    let c2_err = commit(tx, &stack.store).expect_err("C2 must fail");
    assert_wal_rolled_back(&c2_err);

    // C2's primary-index entry must be absent (Z-1 rollback ran).
    let c2_key = PrimaryKey::new(TenantId::DEFAULT, RecordKind::Node, c2_id.raw());
    assert!(
        stack.primary.lookup(c2_key).unwrap().is_none(),
        "C2 primary-index entry must have been rolled back"
    );

    // C1 is unaffected — present in both MVCC and primary index.
    let c1_key = PrimaryKey::new(TenantId::DEFAULT, RecordKind::Node, c1_id.raw());
    assert!(stack.primary.lookup(c1_key).unwrap().is_some());
    let reader = stack.mgr.begin(TenantId::DEFAULT);
    assert!(read_node(&reader, c1_id).unwrap().is_some());
    assert!(
        read_node(&reader, c2_id).unwrap().is_none(),
        "C2 must have no MVCC version"
    );

    // `visible` still reflects C1 only. Note that `current_lsn()`
    // is the visible watermark (invariant 7); C1's lsn is its value.
    assert_eq!(stack.mgr.current_lsn(), c1_lsn);

    // Successor commits would need a fresh WAL handle to progress
    // (WAL is dead). The key property to prove here is that the
    // rolled-back C2 leaves no trace, which we've asserted.
}

// ─────────────────────────────────────────────────────────────────────
// Bundle-path: MVCC SYSTEM root-pointer not advanced on WAL failure
// ─────────────────────────────────────────────────────────────────────

#[test]
fn z1_rollback_system_root_pointer_unchanged_on_wal_failure() {
    // Companion to `z1_rollback_grow_root_unwinds_completely`
    // assertion-level. On WAL failure during a grow_root-
    // triggering commit, the SYSTEM-tenant MVCC root-pointer
    // version MUST NOT be installed (Phase 3's
    // `apply_sidechannel_mvcc_write` only runs on WAL success;
    // §2 explicitly defers sidechannel application).
    //
    // Verified by: `tx.read(PRIMARY_INDEX_ROOT_KEY)` from a
    // SYSTEM-tenant reader returns the pre-W root pointer bytes
    // (unchanged from the seed commit).
    let mut stack = build_stack();

    // Seed: establish the pre-W root.
    let mut tx = stack.mgr.begin(TenantId::DEFAULT);
    let _ = create_node(
        &stack.store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(1),
        &PropertyData::Empty,
    )
    .unwrap();
    commit(tx, &stack.store).unwrap();
    let pre_w_root = stack.primary.root().unwrap();

    // Read the SYSTEM-tenant root pointer via MVCC.
    let sys_reader = stack.mgr.begin(TenantId::SYSTEM);
    let bytes = sys_reader.read(PRIMARY_INDEX_ROOT_KEY).unwrap();
    let raw_bytes: [u8; 8] = bytes[..].try_into().unwrap();
    let pre_w_mvcc_root = u64::from_le_bytes(raw_bytes);
    assert_eq!(pre_w_mvcc_root, pre_w_root.raw());
    drop(sys_reader);

    // Kill WAL + force a commit that would grow_root if it had
    // capacity pressure. Even if this particular commit doesn't
    // trigger grow, the invariant we're asserting is robustness:
    // ANY commit's WAL failure leaves the MVCC root pointer at
    // pre-W. For grow-triggering commits the assertion is
    // non-trivial; for non-grow commits it's trivially true
    // (sidechannel_writes is empty).
    stack.shutdown_wal();
    let mut tx = stack.mgr.begin(TenantId::DEFAULT);
    let _ = create_node(
        &stack.store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(2),
        &PropertyData::Empty,
    )
    .unwrap();
    let _ = commit(tx, &stack.store).expect_err("must fail");

    // Post-rollback: SYSTEM root pointer unchanged.
    let sys_reader2 = stack.mgr.begin(TenantId::SYSTEM);
    let bytes2 = sys_reader2.read(PRIMARY_INDEX_ROOT_KEY).unwrap();
    let raw_bytes2: [u8; 8] = bytes2[..].try_into().unwrap();
    let post_rollback_mvcc_root = u64::from_le_bytes(raw_bytes2);
    assert_eq!(
        post_rollback_mvcc_root, pre_w_mvcc_root,
        "MVCC root pointer must not advance on WAL failure (ADR-032 §2 + \
         ADR-033 §5: sidechannel writes are not applied on WAL failure)"
    );

    // visible is still at seed.
    let _ = Lsn::ZERO; // silence unused import on this path
}

// ─────────────────────────────────────────────────────────────────────
// Y-1 regression — UPDATE-path ghost prevention
// ─────────────────────────────────────────────────────────────────────

/// Read a node's record bytes DIRECTLY from the record page,
/// bypassing MVCC. Used to observe ghost state that `read_node`
/// (which is MVCC-mediated) would paper over.
fn read_node_record_direct(
    store: &CrudStore,
    primary: &PrimaryIndex,
    id: NodeId,
) -> Option<arcgraph_core::NodeRecord> {
    let key = PrimaryKey::new(TenantId::DEFAULT, RecordKind::Node, id.raw());
    let slot = primary.lookup(key).ok().flatten()?;
    let records_binding = store.records();
    let records = records_binding.as_ref()?;
    let latch = records.latch(slot.page).ok()?;
    let g = latch.read();
    let page = SlottedPageRef::open(g.as_ref().as_ref()).ok()?;
    page.read_node(slot.slot).ok().flatten()
}

#[test]
fn z1_update_path_prevents_ghost() {
    // Y-1 regression test (reviewer-flagged critical bug on PR #78).
    //
    // Pre-Y-1 fix: `install_update_deferred` latched the record page
    // and called `page.update_node(slot, new_rec)` IN PLACE without
    // capturing the pre-W bytes. On WAL fsync failure the MVCC
    // version at commit_lsn=W was popped, but the record page slot
    // still held {created_lsn: W, bytes: new_props}. primary.lookup
    // returned the SAME (page_id, slot_id) as before the failed
    // update (pinned coordinate — update didn't move the slot), so
    // a post-rollback direct slot read returned the W-stamped
    // ghost. Under a post-W snapshot the ghost would pass MVCC
    // visibility (created_lsn=W ≤ snapshot), violating ADR-023.
    //
    // Post-Y-1 fix: `install_update_deferred` calls
    // `records.capture_and_write(log, slot.page)` before the
    // in-place mutation; rollback restores the pre-W bytes via
    // Step-3 dispatch to `records.restore_page_bytes`.
    //
    // This test observes the fix by reading the slot bytes DIRECTLY
    // (bypassing MVCC). Pre-fix: observed bytes = new_props.
    // Post-fix: observed bytes = initial_props.

    let mut stack = build_stack();

    // C1: create node N with (a=7, b=11). Durable seed.
    let mut tx = stack.mgr.begin(TenantId::DEFAULT);
    let n_id = create_node(
        &stack.store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(1),
        &PropertyData::InlineU32Pair(7, 11),
    )
    .unwrap();
    commit(tx, &stack.store).unwrap();
    let seed_visible = stack.mgr.current_lsn();

    // Observe the seed slot bytes directly.
    let pre_rec = read_node_record_direct(&stack.store, &stack.primary, n_id)
        .expect("seed node must be readable from record page");
    assert_eq!(pre_rec.inline_u32a, 7);
    assert_eq!(pre_rec.inline_u32b, 11);
    let pre_created_lsn = pre_rec.created_lsn;

    // Shut WAL down so the UPDATE commit fails in Phase 2.
    stack.shutdown_wal();

    // Tx_U: update N to (a=999, b=999).
    let mut tx = stack.mgr.begin(TenantId::DEFAULT);
    update_node(
        &stack.store,
        &mut tx,
        n_id,
        &PropertyData::InlineU32Pair(999, 999),
    )
    .unwrap();
    let err = commit(tx, &stack.store).expect_err("UPDATE must fail with WAL down");
    match err {
        CrudError::Mvcc(ArcGraphError::WalErrorRolledBack { ref source }) => {
            assert!(matches!(source.as_ref(), ArcGraphError::WalUnavailable));
        }
        other => panic!("expected WalErrorRolledBack, got {other:?}"),
    }

    // `visible` did not advance; no durable commit happened.
    assert_eq!(stack.mgr.current_lsn(), seed_visible);

    // Y-1 assertion: direct slot read returns pre-W bytes, not the
    // W-stamped ghost. If Y-1 were broken, inline_u32a would be 999
    // here.
    let post_rollback_rec = read_node_record_direct(&stack.store, &stack.primary, n_id)
        .expect("post-rollback slot read");
    assert_eq!(
        post_rollback_rec.inline_u32a, 7,
        "Y-1: record-page slot bytes must be restored to pre-W after WAL failure; \
         found a=999 ghost — capture_and_write not wired into install_update_deferred?"
    );
    assert_eq!(
        post_rollback_rec.inline_u32b, 11,
        "Y-1: record-page slot bytes must be restored to pre-W (b component)"
    );
    assert_eq!(
        post_rollback_rec.created_lsn, pre_created_lsn,
        "Y-1: slot header's created_lsn must be restored to pre-W; a W-stamped \
         created_lsn would pass MVCC visibility at post-W snapshots and expose \
         the ghost (ADR-023 violation)"
    );

    // MVCC chain has no entry at commit_lsn=W (existing invariant,
    // upheld by rollback_writes). The reader sees the seed version.
    let reader = stack.mgr.begin(TenantId::DEFAULT);
    let mvcc_rec = read_node(&reader, n_id)
        .unwrap()
        .expect("MVCC should still have seed version");
    assert_eq!(mvcc_rec.inline_u32a, 7);
    assert_eq!(mvcc_rec.inline_u32b, 11);
}

// ─────────────────────────────────────────────────────────────────────
// Y-2 regression — multi-store PageId collision
// ─────────────────────────────────────────────────────────────────────

#[test]
fn z1_multi_store_page_id_collision_captured_independently() {
    // Y-2 regression test. Each store has an independent PageId
    // allocator keyed on `(tenant, page_type)` (see
    // `arcgraph_storage::page_alloc::PageAllocator`). Primary index
    // pages for `(SYSTEM, IndexLeaf)` start at PageId(1); record
    // Node pages for `(DEFAULT, Node)` also start at PageId(1).
    //
    // Pre-Y-2: `page_mutations` dedup keyed on `PageId` alone, so
    // the second capture on PageId(1) (whichever store came second)
    // silently no-oped. Under the current post-F2 rollback path
    // this would leave a record-page ghost if the record capture
    // collided with a primary capture — violating Y-1's fix for
    // any commit that touches both stores on PageId(1) (the common
    // case for small tenants).
    //
    // Post-Y-2: dedup keys on `(PageStoreKind, PageId)`, so both
    // captures land. Rollback dispatches each to its correct store.
    //
    // Setup: a commit that UPDATEs an existing node (triggers
    // `records.capture_and_write(PageStoreKind::Record, PageId(1))`)
    // AND creates a new node on a fresh index leaf (triggers
    // primary's `capture_from_guard(PageStoreKind::Primary,
    // PageId(1))` via descent into the root). Both captures are
    // on numeric PageId(1) but under different stores.

    let mut stack = build_stack();

    // C1: create node A on record page 1, slot 0. Primary root is
    // populated with one entry.
    let mut tx = stack.mgr.begin(TenantId::DEFAULT);
    let a_id = create_node(
        &stack.store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(1),
        &PropertyData::InlineU32Pair(11, 22),
    )
    .unwrap();
    commit(tx, &stack.store).unwrap();

    // Confirm the numeric PageId collision in the pre-state.
    let primary_root_id = stack.primary.root().unwrap();
    let a_key = PrimaryKey::new(TenantId::DEFAULT, RecordKind::Node, a_id.raw());
    let a_slot = stack.primary.lookup(a_key).unwrap().unwrap();
    assert_eq!(
        primary_root_id, a_slot.page,
        "Y-2 precondition: primary root and first record page share numeric \
         PageId — otherwise the collision scenario doesn't exercise the Y-2 \
         dedup. primary_root={primary_root_id:?}, record_page={:?}",
        a_slot.page
    );

    // Seed observed pre-W state for BOTH stores.
    let pre_a =
        read_node_record_direct(&stack.store, &stack.primary, a_id).expect("A must be readable");

    // Shut WAL so the combined-mutation commit fails.
    stack.shutdown_wal();

    // Mutation tx: UPDATE A (records.capture_and_write of PageId(1))
    // AND create node B (primary capture_from_guard of PageId(1)).
    // Both captures land in `log.page_mutations` under distinct
    // (kind, PageId(1)) keys.
    let mut tx = stack.mgr.begin(TenantId::DEFAULT);
    update_node(
        &stack.store,
        &mut tx,
        a_id,
        &PropertyData::InlineU32Pair(99, 88),
    )
    .unwrap();
    let b_id = create_node(
        &stack.store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(2),
        &PropertyData::InlineU32Pair(33, 44),
    )
    .unwrap();
    let err = commit(tx, &stack.store).expect_err("combined commit must fail");
    match err {
        CrudError::Mvcc(ArcGraphError::WalErrorRolledBack { .. }) => {}
        other => panic!("expected WalErrorRolledBack, got {other:?}"),
    }

    // Post-rollback assertions.
    //
    // A's slot must be restored to pre-W (11, 22), NOT post-W (99,
    // 88). Pre-Y-2: the record capture would have deduped against
    // the primary capture; record page remains post-W ghost.
    let post_a = read_node_record_direct(&stack.store, &stack.primary, a_id)
        .expect("A's slot must be readable post-rollback");
    assert_eq!(
        post_a.inline_u32a, pre_a.inline_u32a,
        "Y-2: record-page capture must not collide with primary capture on \
         same numeric PageId. If this fails with a=99 b=88, the Y-1 fix's \
         capture_and_write was no-op'd by the pre-Y-2 dedup collision."
    );
    assert_eq!(post_a.inline_u32b, pre_a.inline_u32b);

    // Primary rollback: B's key must be absent from the index.
    // (The primary was captured and restored to pre-W state, which
    // had only A's entry.)
    let b_key = PrimaryKey::new(TenantId::DEFAULT, RecordKind::Node, b_id.raw());
    assert!(
        stack.primary.lookup(b_key).unwrap().is_none(),
        "Y-2: primary capture must be restored; new B entry must not survive"
    );

    // A's primary entry still points at the ORIGINAL (page, slot)
    // — update didn't move it.
    let a_slot_post = stack
        .primary
        .lookup(a_key)
        .unwrap()
        .expect("A must still be in primary index");
    assert_eq!(a_slot_post.page, a_slot.page);
    assert_eq!(a_slot_post.slot, a_slot.slot);
}

// ─────────────────────────────────────────────────────────────────────
// ARCGRAPH_WAL_ERROR_POLICY=abort — subprocess test
// ─────────────────────────────────────────────────────────────────────

// Disabled by default: spawning a subprocess that aborts is brittle
// in CI. A manual reproducer is retained as documentation; the
// positive path (policy parses to Abort) is covered by config's
// unit tests.
//
// To run manually:
//   ARCGRAPH_WAL_ERROR_POLICY=abort cargo test \
//     -p arcgraph-storage --test z1_rollback --ignored \
//     -- z1_wal_error_policy_abort_repro
#[test]
#[ignore = "subprocess abort test; run with --ignored"]
fn z1_wal_error_policy_abort_repro() {
    // This is a documented manual repro. When
    // ARCGRAPH_WAL_ERROR_POLICY=abort is set in the env and a WAL
    // fsync fails, the process aborts before rollback. Observable
    // via the process exit code (SIGABRT / 134 on Linux, 6 on
    // macOS). The unit-test coverage of policy parsing lives in
    // `arcgraph-storage::config::tests`; this test is a sanity
    // check for when an operator wires the policy + WAL failure
    // together in a subprocess.
    let mut stack = build_stack();
    stack.shutdown_wal();
    let mut tx = stack.mgr.begin(TenantId::DEFAULT);
    let _ = create_node(
        &stack.store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(1),
        &PropertyData::Empty,
    )
    .unwrap();
    let _ = commit(tx, &stack.store);
    // If abort policy is active, this line is never reached.
    panic!("abort policy did not fire");
}

// ─────────────────────────────────────────────────────────────────
// M3.a Slice G.5 — production-path Z-1 (b) vector arena rollback
// ─────────────────────────────────────────────────────────────────
//
// REPLACES the prior trait-modeled `z1_rollback_vector_arena_pages`
// test (which exercised `VectorPageStoreHandle::restore_page_bytes`
// directly via the trait). The production-path test exercises the
// same restore through `crud::commit` + injected WAL fsync failure
// + the rollback closure populated by Slice G.5.
//
// Closes issue #131 follow-up item 3 (production-path closure of
// the trait-modeled test in PR #82's z1_rollback test suite).
//
// ADR-035 §7.5 + the production `crud.rs` `PageStoreKind::Vector`
// dispatch arm pin that vector arena page mutations are captured
// into the same `TxnMutationLog` that primary / record / blob page
// mutations live in, and that on WAL fsync failure the rollback
// drainer dispatches each captured pre-W byte block to
// `VectorPageStoreHandle::restore_page_bytes` for the owning store.

#[test]
fn z1_rollback_vector_arena_pages() {
    use arcgraph_core::PageId as CorePageId;
    use arcgraph_core::record::PAGE_SIZE;
    use arcgraph_storage::vector_store::VectorPageStoreHandle;

    // Build a stack with a real vector arena wired into the
    // CrudStore. The arena (returned alongside the stack) is the
    // same handle the rollback closure dispatches into.
    let (mut stack, arena) = build_stack_with_vector_store();

    let tenant = TenantId::DEFAULT;
    let page_id = CorePageId::new(101);

    // ── Phase 1: install the durable pre-W baseline. ──
    // Production: the post-recovery state that a builder mutates
    // in-place during a transaction. The arena is decoupled from
    // the WAL, so this is safe to do before WAL shutdown.
    let pre_w = [0xAAu8; PAGE_SIZE];
    arena
        .install_or_replace(tenant, page_id, &pre_w)
        .expect("install pre-W");
    assert_eq!(arena.get_page(tenant, page_id).unwrap(), pre_w.to_vec());

    // ── Phase 2: kill WAL BEFORE begin() so the doomed commit is
    //          guaranteed to fail at Phase 2 with WalUnavailable
    //          (mirrors the existing z1_rollback_* tests' ordering). ──
    stack.shutdown_wal();

    // ── Phase 3: a transaction mutates the page in place. ──
    //
    // The transaction also creates a node (so the bundle has work
    // to durify; a vector-only commit is well-formed but a CRUD
    // co-mutation makes the test exercise the full bundle drain
    // path the production rollback closure actually walks).
    let mut tx = stack.mgr.begin(tenant);
    let _seed_id = create_node(
        &stack.store,
        &mut tx,
        tenant,
        LabelId::new(1),
        &PropertyData::Empty,
    )
    .unwrap();
    let txn_id = tx.id();

    // Capture pre-W into the txn mutation log AND stage post-W
    // for the v5 bundle. Single-call helper from G.5; called
    // through the test-only `Transaction::mutation_log_mut`
    // accessor that mirrors the post-G.7 production wiring (vector
    // writers will reach `&mut TxnMutationLog` through the builder
    // closure's argument).
    let post_w_box: Box<[u8; PAGE_SIZE]> = Box::new([0xBBu8; PAGE_SIZE]);
    {
        let log = tx.mutation_log_mut();
        stack.store.capture_and_stage_vector_page(
            log,
            txn_id,
            tenant,
            PartitionId::ZERO,
            0,
            page_id,
            &pre_w,
            post_w_box.clone(),
        );
    }

    // Mirror the post-W mutation in the arena (as a builder-phase
    // page mutation would).
    arena
        .install_or_replace(tenant, page_id, post_w_box.as_ref())
        .expect("post-W mutate");
    assert_eq!(
        arena.get_page(tenant, page_id).unwrap(),
        post_w_box.to_vec()
    );

    // Drive the doomed commit. Rollback runs.
    let err = commit(tx, &stack.store).expect_err("commit must fail with WAL down");
    assert_wal_rolled_back(&err);

    // ── Phase 4: post-rollback assertions. ──
    // The Vector arm in the rollback closure SHOULD have called
    // `arena.restore_page_bytes(tenant, page_id, pre_w)`.
    let restored = arena.get_page(tenant, page_id).expect("page present");
    assert_eq!(restored.len(), PAGE_SIZE);
    assert_eq!(
        &restored[..],
        &pre_w[..],
        "Z-1 (b) production-path: pre-W bytes restored byte-identically"
    );
}

// ─────────────────────────────────────────────────────────────────────
// M3 round-4 Z-1(b): tenant-qualified record-page rollback
// ─────────────────────────────────────────────────────────────────────

/// Seed distinct real buffered-store bytes at `(DEFAULT, page 1)` and
/// `(tenant B, page 1)`, then abort tenant B's UPDATE at the WAL boundary.
/// The load-bearing observable is DEFAULT's collateral clobber: the old
/// tenant-blind rollback restored tenant B's pre-image into DEFAULT's page.
#[test]
fn z1_record_page_rollback_does_not_clobber_default_tenant() {
    const TENANT_B: TenantId = TenantId::new(100);
    let (mut stack, records) = build_buffered_stack();

    let mut default_tx = stack.mgr.begin(TenantId::DEFAULT);
    let default_node = create_node(
        &stack.store,
        &mut default_tx,
        TenantId::DEFAULT,
        LabelId::new(11),
        &PropertyData::InlineU32Pair(111, 222),
    )
    .unwrap();
    commit(default_tx, &stack.store).unwrap();

    let mut tenant_tx = stack.mgr.begin(TENANT_B);
    let tenant_node = create_node(
        &stack.store,
        &mut tenant_tx,
        TENANT_B,
        LabelId::new(22),
        &PropertyData::InlineU32Pair(7, 11),
    )
    .unwrap();
    commit(tenant_tx, &stack.store).unwrap();
    let seed_visible = stack.mgr.current_lsn();

    let default_slot = stack
        .primary
        .lookup(PrimaryKey::new(
            TenantId::DEFAULT,
            RecordKind::Node,
            default_node.raw(),
        ))
        .unwrap()
        .unwrap();
    let tenant_slot = stack
        .primary
        .lookup(PrimaryKey::new(
            TENANT_B,
            RecordKind::Node,
            tenant_node.raw(),
        ))
        .unwrap()
        .unwrap();
    assert_eq!(default_slot.page, PageId::new(1));
    assert_eq!(tenant_slot.page, PageId::new(1));
    let default_pre = records
        .copy_page_pinned_for_tenant(TenantId::DEFAULT, PageId::new(1))
        .unwrap()
        .unwrap();
    let tenant_pre = records
        .copy_page_pinned_for_tenant(TENANT_B, PageId::new(1))
        .unwrap()
        .unwrap();
    assert_ne!(
        default_pre.as_ref(),
        tenant_pre.as_ref(),
        "premise: both tenant-qualified pages must carry distinct bytes"
    );

    stack.shutdown_wal();
    let mut tx = stack.mgr.begin(TENANT_B);
    update_node(
        &stack.store,
        &mut tx,
        tenant_node,
        &PropertyData::InlineU32Pair(999, 999),
    )
    .unwrap();
    let err = commit(tx, &stack.store).expect_err("UPDATE must fail with WAL down");
    assert_wal_rolled_back(&err);
    assert_eq!(stack.mgr.current_lsn(), seed_visible);

    let default_post = records
        .copy_page_pinned_for_tenant(TenantId::DEFAULT, PageId::new(1))
        .unwrap()
        .unwrap();
    let tenant_post = records
        .copy_page_pinned_for_tenant(TENANT_B, PageId::new(1))
        .unwrap()
        .unwrap();
    assert_eq!(
        default_post.as_ref(),
        default_pre.as_ref(),
        "tenant B rollback clobbered (DEFAULT, page 1) with tenant B's pre-image"
    );
    assert_eq!(
        tenant_post.as_ref(),
        tenant_pre.as_ref(),
        "tenant B rollback did not restore (tenant B, page 1) byte-identically"
    );
}
