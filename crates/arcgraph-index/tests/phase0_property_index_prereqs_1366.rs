//! Property-Index **Phase 0** — the load-bearing MVCC / crash-consistency
//! prerequisites (rc-blocker #1366). Storage-backed, RED-on-revert
//! sensitivity-verified per the #1378 lesson (a gate that cannot catch
//! its named regression does not enforce it).
//!
//! These tests live in `arcgraph-index/tests/` so they can wire the
//! concrete `SecondaryIndex` into a `CrudStore` via `new_with_indices`
//! — the storage crate cannot import `arcgraph-index` (the dependency
//! graph is `index → storage`).
//!
//! Coverage (each with its RED-on-revert companion):
//! - **(a) RC-1 false-negative:** a reader on a snapshot predating a
//!   writer's `a → b` update MUST still find the node via `email = a`.
//!   RED-on-revert = the eager-removal path observably MISSES it.
//! - **(b) aborted-txn rollback drains secondary pages (Z-1 F-1):** an
//!   aborted insert leaves no leaked/inconsistent secondary pages; the
//!   pre-abort structure is restored. RED-on-revert = dropping the drain
//!   leaves the aborted page mapped.
//! - **(c) Building-state node is indexed (RC-2):** a node written while
//!   the index is `Building` is found once it goes `Online`. RED-on-
//!   revert = an `Online`-only maintenance gate leaves it absent.
//! - deferred-removal-applies-only-after-horizon (RC-1 timing) and
//!   idempotent overlap are exercised via the shared harness.
//!
//! (Test (d) — read-lookup does not grow the InternTable — lives next
//! to `InternTable` in `arcgraph-storage::intern` because it needs no
//! index wiring.)

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use arcgraph_core::{LabelId, NodeId, StringId, TenantId};
use arcgraph_index::{PropertyValue, SecondaryIndex, SecondaryKey};
use arcgraph_storage::crud::{
    CrudError, CrudStore, INLINE_U32A_PROPERTY_KEY, PropertyData, commit, create_node, read_node,
    update_node,
};
use arcgraph_storage::mutation_log::TxnMutationLog;
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::PrimaryIndex;
use arcgraph_storage::secondary_handle::{IndexState, SecondaryIndexHandle};
use arcgraph_storage::transaction::{Transaction, TxnManager};
use arcgraph_storage::wal::{WalConfig, WalWriter};
use tempfile::TempDir;

// ─────────────────────────────────────────────────────────────────────
// Harness
// ─────────────────────────────────────────────────────────────────────

fn build_store() -> (Arc<TxnManager>, Arc<CrudStore>, Arc<SecondaryIndex>) {
    let txn_mgr = Arc::new(TxnManager::new());
    let alloc = Arc::new(PageAllocator::new());
    let primary =
        Arc::new(PrimaryIndex::new(Arc::clone(&txn_mgr), Arc::clone(&alloc), None).unwrap());
    let secondary =
        Arc::new(SecondaryIndex::new(Arc::clone(&txn_mgr), Arc::clone(&alloc), None).unwrap());
    let handle: Arc<dyn SecondaryIndexHandle> = Arc::clone(&secondary) as _;
    let store = Arc::new(CrudStore::new_with_indices(
        None,
        Arc::clone(&primary),
        Some(handle),
        Arc::clone(&alloc),
    ));
    (txn_mgr, store, secondary)
}

/// The secondary index keys `inline_u32a` under a reserved property key.
/// We model "email = value" as `inline_u32a = value` (Phase 0 has no
/// named-property catalog; the RC-1/RC-2/rollback invariants are
/// value-shape-agnostic).
fn email_key(label: u32, value: u32) -> SecondaryKey {
    SecondaryKey::new(
        TenantId::DEFAULT,
        LabelId::new(label),
        INLINE_U32A_PROPERTY_KEY,
        PropertyValue::U32(value),
    )
}

/// The load-bearing property-index lookup under the ADR-023 contract:
/// the B-tree yields candidate NodeIds; each is hydrated through the
/// reader's snapshot and the property is re-checked. Returns the set of
/// nodes that are actually visible at `reader`'s snapshot AND still
/// carry `email = value`. This is precisely what a `PropertyIndexScan`
/// executor would do, so a false negative here is a silent wrong result
/// in a real query.
fn property_lookup_verified(
    secondary: &SecondaryIndex,
    reader: &Transaction<'_>,
    label: u32,
    value: u32,
) -> Vec<NodeId> {
    let candidates = secondary.lookup(email_key(label, value)).unwrap();
    candidates
        .into_iter()
        .filter(|&id| {
            // MVCC verify: hydrate through the reader's snapshot and
            // re-check the property equals the looked-up value.
            match read_node(reader, id) {
                Ok(Some(rec)) => rec.inline_u32a == value,
                _ => false,
            }
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────
// (a) RC-1 false-negative — THE load-bearing gate
// ─────────────────────────────────────────────────────────────────────

/// **Test (a) — GREEN.** A reader on a snapshot predating writer W's
/// `email a → b` update MUST still find node `n` when it looks up
/// `email = a`. Under RC-1 insert-only maintenance the `(email=a) → n`
/// entry is a deferred-removal ghost (kept live until the horizon
/// passes), so the pre-commit reader hydrates `n`, sees `email = a` at
/// its own snapshot, and the verify step keeps it: exactly one found.
#[test]
fn rc1_old_snapshot_reader_still_finds_pre_update_value() {
    let (mgr, store, secondary) = build_store();

    // Seed: n with email = a (= 100).
    let mut tx = mgr.begin(TenantId::DEFAULT);
    let n = create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(7),
        &PropertyData::InlineU32Pair(100, 0),
    )
    .unwrap();
    commit(tx, &store).unwrap();

    // A reader R begins on a snapshot BEFORE W's update — it must see
    // email = a for n forever, and any index lookup it drives for
    // email = a must find n.
    let reader = mgr.begin(TenantId::DEFAULT);

    // Writer W updates email a → b (100 → 200) and commits.
    let mut tx = mgr.begin(TenantId::DEFAULT);
    update_node(&store, &mut tx, n, &PropertyData::InlineU32Pair(200, 0)).unwrap();
    commit(tx, &store).unwrap();

    // THE assertion: R's index lookup for email = a still finds n
    // (exactly one). A missing entry would be a silent wrong result
    // that candidate-then-verify cannot recover.
    let found = property_lookup_verified(secondary.as_ref(), &reader, 7, 100);
    assert_eq!(
        found,
        vec![n],
        "RC-1: a pre-update snapshot reader must still find n via email=a \
         (the deferred-removal ghost keeps the entry live)",
    );

    // And a fresh reader (snapshot after W) correctly sees email = b:
    // email = a yields nothing (verify-filtered ghost), email = b finds n.
    let fresh = mgr.begin(TenantId::DEFAULT);
    assert!(
        property_lookup_verified(secondary.as_ref(), &fresh, 7, 100).is_empty(),
        "a post-update reader must NOT see email=a (verify filters the ghost)",
    );
    assert_eq!(
        property_lookup_verified(secondary.as_ref(), &fresh, 7, 200),
        vec![n],
        "a post-update reader sees email=b",
    );
}

/// **Test (a) — RED-on-revert.** This reproduces the pre-RC-1 eager
/// removal by synchronously zeroing the `(email=a) → n` slot at update
/// time (exactly what `remove_property_deferred` did on the old-value
/// side of the commit drain). It proves the guard above is sensitive:
/// under the reverted behavior the pre-update snapshot reader OBSERVABLY
/// MISSES n. The two tests cannot both pass under the same code path.
#[test]
fn rc1_eager_removal_makes_old_snapshot_reader_miss_red_on_revert() {
    let (mgr, store, secondary) = build_store();

    let mut tx = mgr.begin(TenantId::DEFAULT);
    let n = create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(7),
        &PropertyData::InlineU32Pair(100, 0),
    )
    .unwrap();
    commit(tx, &store).unwrap();

    let reader = mgr.begin(TenantId::DEFAULT);

    // Writer updates a → b.
    let mut tx = mgr.begin(TenantId::DEFAULT);
    update_node(&store, &mut tx, n, &PropertyData::InlineU32Pair(200, 0)).unwrap();
    commit(tx, &store).unwrap();

    // REVERT: eagerly zero the old-value slot NOW (the pre-RC-1 commit
    // path did this at commit-builder time — before the horizon passed).
    // This is the exact regression RC-1 removed.
    let removed = secondary
        .remove(email_key(7, 100), n)
        .expect("eager remove");
    assert!(removed, "the eager revert must find and zero the old slot");

    // Under the reverted (eager) behavior the pre-update snapshot reader
    // now MISSES n on email = a — a silent false negative.
    let found = property_lookup_verified(secondary.as_ref(), &reader, 7, 100);
    assert!(
        found.is_empty(),
        "RED-on-revert: with eager removal the old-snapshot reader MISSES n \
         (this is the false-negative RC-1 prevents; the GREEN test above \
         asserts the opposite under the shipped deferred path)",
    );
}

/// RC-1 timing: a deferred removal applies ONLY once the snapshot
/// horizon reaches the removing commit's LSN. While a reader that
/// predates the update is still live, the removal must NOT apply.
#[test]
fn rc1_deferred_removal_waits_for_snapshot_horizon() {
    let (mgr, store, secondary) = build_store();

    let mut tx = mgr.begin(TenantId::DEFAULT);
    let n = create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(9),
        &PropertyData::InlineU32Pair(100, 0),
    )
    .unwrap();
    commit(tx, &store).unwrap();

    // Long-running reader pins the horizon below the coming update.
    let pinning_reader = mgr.begin(TenantId::DEFAULT);

    let mut tx = mgr.begin(TenantId::DEFAULT);
    update_node(&store, &mut tx, n, &PropertyData::InlineU32Pair(200, 0)).unwrap();
    commit(tx, &store).unwrap();
    assert_eq!(store.deferred_removal_queue_len(), 1);

    // Even after several later commits, while `pinning_reader` is live
    // the removal cannot apply (its snapshot could still observe email=a).
    for v in 300..305u32 {
        let mut tx = mgr.begin(TenantId::DEFAULT);
        let _ = create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(9),
            &PropertyData::InlineU32Pair(v, 0),
        )
        .unwrap();
        commit(tx, &store).unwrap();
    }
    assert_eq!(
        store.deferred_removal_queue_len(),
        1,
        "the removal must stay queued while a pre-update reader is live",
    );
    assert_eq!(
        secondary.lookup(email_key(9, 100)).unwrap(),
        vec![n],
        "the ghost is still present while pinned",
    );

    // Release the pinning reader, then one more commit advances the
    // horizon past the update — the removal now applies.
    pinning_reader.abort();
    let mut tx = mgr.begin(TenantId::DEFAULT);
    let _ = create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(9),
        &PropertyData::InlineU32Pair(999, 0),
    )
    .unwrap();
    commit(tx, &store).unwrap();
    assert_eq!(store.deferred_removal_queue_len(), 0);
    assert!(
        secondary.lookup(email_key(9, 100)).unwrap().is_empty(),
        "once the horizon cleared the update LSN, the ghost is reclaimed",
    );
}

// ─────────────────────────────────────────────────────────────────────
// (c) RC-2 Building-state write-follows-declare
// ─────────────────────────────────────────────────────────────────────

/// **Test (c) — GREEN.** A node written while the index is `Building`
/// (write-follows-declare) is found once the index goes `Online`.
/// `maintenance_active()` is TRUE in `Building`, so the commit drain
/// indexes the node even before the `Online` flip.
#[test]
fn rc2_building_state_node_is_indexed() {
    let (mgr, store, secondary) = build_store();

    // Enter Building (Phase-1 CREATE INDEX would do this before backfill).
    secondary.set_index_state(IndexState::Building);
    assert_eq!(secondary.index_state(), IndexState::Building);
    assert!(
        secondary.maintenance_active(),
        "RC-2: maintenance MUST be active in Building (write-follows-declare)",
    );

    // Write a node while Building.
    let mut tx = mgr.begin(TenantId::DEFAULT);
    let n = create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(4),
        &PropertyData::InlineU32Pair(555, 0),
    )
    .unwrap();
    commit(tx, &store).unwrap();

    // Flip to Online (Phase-1: rides the same bundle as the final
    // backfill watermark).
    secondary.set_index_state(IndexState::Online);
    assert!(secondary.index_state().planner_visible());

    // The Building-written node is present now that the index is Online.
    let reader = mgr.begin(TenantId::DEFAULT);
    let found = property_lookup_verified(secondary.as_ref(), &reader, 4, 555);
    assert_eq!(
        found,
        vec![n],
        "RC-2: the Building-written node must be found once Online",
    );
}

/// **Test (c) — RED-on-revert.** Reverts the write-follows-declare rule
/// to an `Online`-only maintenance gate: with maintenance skipped while
/// `Building`, the Building-written node is ABSENT once the index goes
/// `Online` — a permanent false negative. This proves the GREEN test's
/// sensitivity. We revert via a wrapper handle whose `maintenance_active`
/// returns `planner_visible()` (Online-only), which the commit drain
/// consults through the exact same trait method.
#[test]
fn rc2_online_only_gate_drops_building_node_red_on_revert() {
    use arcgraph_core::PageId;
    use arcgraph_storage::StagedEmit;
    use arcgraph_storage::mutation_log::{PageBuf, TxnMutationLog};
    use arcgraph_storage::secondary_handle::{SecondaryIndexHandleError, SecondaryIndexValue};

    /// A wrapper that reverts RC-2's rule: it delegates every method to
    /// the real `SecondaryIndex` EXCEPT `maintenance_active`, which it
    /// gates on `Online` only (the pre-RC-2 hazard). All maintenance
    /// calls delegate, so the ONLY behavioral difference is the gate.
    #[derive(Debug)]
    struct OnlineOnlyGate(Arc<SecondaryIndex>);

    impl SecondaryIndexHandle for OnlineOnlyGate {
        fn insert_property(
            &self,
            t: TenantId,
            l: LabelId,
            pk: StringId,
            v: SecondaryIndexValue,
            n: NodeId,
        ) -> Result<(), SecondaryIndexHandleError> {
            self.0.insert_property(t, l, pk, v, n)
        }
        fn remove_property(
            &self,
            t: TenantId,
            l: LabelId,
            pk: StringId,
            v: SecondaryIndexValue,
            n: NodeId,
        ) -> Result<bool, SecondaryIndexHandleError> {
            self.0.remove_property(t, l, pk, v, n)
        }
        fn insert_property_deferred(
            &self,
            t: TenantId,
            l: LabelId,
            pk: StringId,
            v: SecondaryIndexValue,
            n: NodeId,
            log: &mut TxnMutationLog,
        ) -> Result<Vec<StagedEmit>, SecondaryIndexHandleError> {
            self.0.insert_property_deferred(t, l, pk, v, n, log)
        }
        fn remove_property_deferred(
            &self,
            t: TenantId,
            l: LabelId,
            pk: StringId,
            v: SecondaryIndexValue,
            n: NodeId,
            log: &mut TxnMutationLog,
        ) -> Result<Vec<StagedEmit>, SecondaryIndexHandleError> {
            self.0.remove_property_deferred(t, l, pk, v, n, log)
        }
        fn persist_pending_root_update(&self) -> Result<(), SecondaryIndexHandleError> {
            // Call the trait method (returns SecondaryIndexHandleError),
            // not the inherent method (returns SecondaryIndexError).
            SecondaryIndexHandle::persist_pending_root_update(self.0.as_ref())
        }
        fn rollback_remove_page(&self, p: PageId) {
            self.0.rollback_remove_page(p)
        }
        fn rollback_restore_page(
            &self,
            p: PageId,
            b: &PageBuf,
        ) -> Result<(), SecondaryIndexHandleError> {
            self.0.rollback_restore_page(p, b)
        }
        fn rollback_restore_root(&self, r: PageId) {
            self.0.rollback_restore_root(r)
        }
        fn index_state(&self) -> IndexState {
            self.0.index_state()
        }
        fn set_index_state(&self, s: IndexState) {
            self.0.set_index_state(s)
        }
        // THE REVERT: gate maintenance on Online only (drops the RULE
        // that maintenance also applies in Building).
        fn maintenance_active(&self) -> bool {
            self.0.index_state().planner_visible()
        }
    }

    let txn_mgr = Arc::new(TxnManager::new());
    let alloc = Arc::new(PageAllocator::new());
    let primary =
        Arc::new(PrimaryIndex::new(Arc::clone(&txn_mgr), Arc::clone(&alloc), None).unwrap());
    let secondary =
        Arc::new(SecondaryIndex::new(Arc::clone(&txn_mgr), Arc::clone(&alloc), None).unwrap());
    let reverted: Arc<dyn SecondaryIndexHandle> = Arc::new(OnlineOnlyGate(Arc::clone(&secondary)));
    let store = Arc::new(CrudStore::new_with_indices(
        None,
        Arc::clone(&primary),
        Some(reverted),
        Arc::clone(&alloc),
    ));

    // Building: under the reverted Online-only gate, maintenance is
    // SKIPPED while Building.
    secondary.set_index_state(IndexState::Building);
    let mut tx = txn_mgr.begin(TenantId::DEFAULT);
    let n = create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(4),
        &PropertyData::InlineU32Pair(555, 0),
    )
    .unwrap();
    commit(tx, &store).unwrap();

    // Flip Online — but the Building-written node was never indexed.
    secondary.set_index_state(IndexState::Online);
    let reader = txn_mgr.begin(TenantId::DEFAULT);
    let found = property_lookup_verified(secondary.as_ref(), &reader, 4, 555);
    assert!(
        found.is_empty(),
        "RED-on-revert: an Online-only maintenance gate drops the \
         Building-written node n (permanent false negative). The GREEN \
         test above asserts the opposite under the shipped RC-2 rule.",
    );
    // Sanity: n really exists in MVCC — the miss is index-only.
    assert!(read_node(&reader, n).unwrap().is_some());
}

// ─────────────────────────────────────────────────────────────────────
// (b) Aborted-txn rollback drains secondary pages (Z-1 F-1)
// ─────────────────────────────────────────────────────────────────────

fn wal_config(dir: PathBuf) -> WalConfig {
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

/// A WAL-backed dual-write stack (primary + secondary) whose WAL can be
/// shut down to force the next commit's Phase-2 fsync to fail, driving
/// the Z-1 (b) rollback closure.
struct WalStack {
    _dir: TempDir,
    store: Arc<CrudStore>,
    mgr: Arc<TxnManager>,
    secondary: Arc<SecondaryIndex>,
    writer: Option<WalWriter>,
}

fn build_wal_stack() -> WalStack {
    let dir = TempDir::new().unwrap();
    let writer = WalWriter::spawn(wal_config(dir.path().to_path_buf())).unwrap();
    let handle = writer.handle();
    let mgr = Arc::new(TxnManager::with_wal(handle.clone()));
    let alloc = Arc::new(PageAllocator::new());
    let primary = Arc::new(
        PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&alloc), Some(handle.clone())).unwrap(),
    );
    let secondary = Arc::new(
        SecondaryIndex::new(Arc::clone(&mgr), Arc::clone(&alloc), Some(handle.clone())).unwrap(),
    );
    let sec_handle: Arc<dyn SecondaryIndexHandle> = Arc::clone(&secondary) as _;
    let store = Arc::new(CrudStore::new_with_indices(
        Some(handle.clone()),
        Arc::clone(&primary),
        Some(sec_handle),
        alloc,
    ));
    WalStack {
        _dir: dir,
        store,
        mgr,
        secondary,
        writer: Some(writer),
    }
}

/// **Test (b) — GREEN, integrated.** An aborted node insert (WAL fsync
/// failure) leaves the secondary index structurally consistent: no
/// leaked pages, and no entry for the aborted node's value that would be
/// missing for a still-visible node. The seed node's entry survives; the
/// aborted node's entry does not linger as a live candidate.
#[test]
fn rc_rollback_aborted_insert_leaves_secondary_consistent() {
    let mut stack = build_wal_stack();

    // Seed a durable node so the secondary root leaf is populated.
    let mut tx = stack.mgr.begin(TenantId::DEFAULT);
    let seed = create_node(
        &stack.store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(3),
        &PropertyData::InlineU32Pair(10, 0),
    )
    .unwrap();
    commit(tx, &stack.store).unwrap();

    let pages_before = stack.secondary.page_store().len();
    let seed_hits_before = stack.secondary.lookup(email_key(3, 10)).unwrap();
    assert_eq!(seed_hits_before, vec![seed]);

    // Shut down the WAL so the next commit fails at Phase 2.
    if let Some(w) = stack.writer.take() {
        w.shutdown().expect("wal shutdown");
    }

    // Doomed insert: a node with a fresh value (20). Its commit fails →
    // the Z-1 (b) rollback closure runs, draining the secondary page it
    // captured / installed.
    let mut tx = stack.mgr.begin(TenantId::DEFAULT);
    let _doomed = create_node(
        &stack.store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(3),
        &PropertyData::InlineU32Pair(20, 0),
    )
    .unwrap();
    let err = commit(tx, &stack.store).expect_err("commit must fail with WAL down");
    assert!(
        matches!(
            err,
            CrudError::Mvcc(arcgraph_core::ArcGraphError::WalErrorRolledBack { .. })
        ),
        "expected WalErrorRolledBack, got {err:?}",
    );

    // Structural consistency post-rollback:
    // 1. No leaked pages — the store's page set is exactly the pre-abort
    //    set (the doomed insert mutated the root leaf in place; the
    //    rollback restored its pre-W bytes, and any split page is
    //    removed).
    assert_eq!(
        stack.secondary.page_store().len(),
        pages_before,
        "Z-1 F-1: aborted secondary pages must be drained (no leak)",
    );
    // 2. The aborted node's value is NOT a live entry (the leaf was
    //    restored to pre-abort bytes).
    assert!(
        stack.secondary.lookup(email_key(3, 20)).unwrap().is_empty(),
        "Z-1 F-1: the aborted insert's entry must be rolled back",
    );
    // 3. The seed node's entry is untouched — rollback did not corrupt
    //    the surviving structure.
    assert_eq!(
        stack.secondary.lookup(email_key(3, 10)).unwrap(),
        vec![seed],
        "the seed entry must survive the rollback intact",
    );
}

/// **Test (b) — GREEN, split path.** When the aborted insert triggers a
/// leaf split (a fresh page is allocated + installed), rollback removes
/// that fresh page. Exercised at the primitive level: `insert_deferred`
/// captures the fresh page into the mutation log, and the published
/// `rollback_remove_page` drain removes it — restoring the page count.
#[test]
fn rc_rollback_drains_fresh_split_page() {
    let (_mgr, _store, secondary) = build_store();

    // Fill the root leaf to capacity so the next distinct-key insert
    // splits (allocates a fresh page). LEAF_CAPACITY is 127; a distinct
    // value per key.
    let mut log = TxnMutationLog::new();
    for v in 0..127u32 {
        let _ = secondary
            .insert_deferred(email_key(1, v), NodeId::new(u64::from(v) + 1), &mut log)
            .unwrap();
    }
    // Drain the fill's captures — the fill is "committed" for the test's
    // purposes (we only want to roll back the split-causing insert).
    let pages_after_fill = secondary.page_store().len();

    // The split-causing insert: a fresh mutation log captures the new
    // split page + the in-place leaf edits.
    let mut split_log = TxnMutationLog::new();
    let _ = secondary
        .insert_deferred(email_key(1, 500), NodeId::new(9999), &mut split_log)
        .unwrap();
    assert!(
        secondary.page_store().len() > pages_after_fill,
        "the split-causing insert must allocate a fresh page",
    );
    assert!(
        !split_log.new_pages.is_empty(),
        "the split must have recorded a fresh secondary page in the log",
    );

    // GREEN: drain the log via the published rollback dispatch (exactly
    // what the crud.rs Z-1 rollback closure does for Secondary entries).
    let handle: &dyn SecondaryIndexHandle = &*secondary;
    for (_kind, page_id) in split_log.new_pages.drain(..) {
        handle.rollback_remove_page(page_id);
    }
    for (_kind, page_id, pre_bytes) in split_log.page_mutations.drain(..) {
        handle
            .rollback_restore_page(page_id, pre_bytes.as_ref())
            .unwrap();
    }
    for (_handle, old_root) in split_log.root_changes.drain(..) {
        handle.rollback_restore_root(old_root);
    }

    assert_eq!(
        secondary.page_store().len(),
        pages_after_fill,
        "Z-1 F-1: the fresh split page must be removed on rollback",
    );
}

/// **Test (b) — RED-on-revert.** Reproduces the pre-F-1 warn-skip: when
/// the Secondary rollback drain is DROPPED (the arms did nothing), the
/// aborted insert's fresh page STAYS mapped in the secondary page store
/// — a structural leak / inconsistency. This proves the drain tests
/// above are sensitive.
#[test]
fn rc_rollback_drop_drain_leaks_page_red_on_revert() {
    let (_mgr, _store, secondary) = build_store();

    let mut log = TxnMutationLog::new();
    for v in 0..127u32 {
        let _ = secondary
            .insert_deferred(email_key(1, v), NodeId::new(u64::from(v) + 1), &mut log)
            .unwrap();
    }
    let pages_after_fill = secondary.page_store().len();

    let mut split_log = TxnMutationLog::new();
    let _ = secondary
        .insert_deferred(email_key(1, 500), NodeId::new(9999), &mut split_log)
        .unwrap();
    let pages_with_split = secondary.page_store().len();
    assert!(pages_with_split > pages_after_fill);
    assert!(!split_log.new_pages.is_empty());

    // REVERT: the pre-F-1 behavior — the Secondary arms warn-and-skip, so
    // NOTHING is drained. We simply drop `split_log` without dispatching.
    drop(split_log);

    // Under the reverted (no-drain) behavior the fresh split page STAYS
    // mapped — a leak the F-1 fix closes.
    assert_eq!(
        secondary.page_store().len(),
        pages_with_split,
        "RED-on-revert: without the secondary drain the aborted split \
         page leaks (stays mapped). The GREEN drain tests above remove it.",
    );
}

// ─────────────────────────────────────────────────────────────────────
// (b) crud-closure-level rollback: split + grow-root THROUGH commit()
//
// The two tests above prove the drain at the primitive level (calling
// `rollback_remove_page` / `rollback_restore_root` directly, or driving
// a single-entry-leaf abort that never splits). Neither drives a
// SPLITTING or ROOT-GROWING secondary insert THROUGH the real
// `crud::commit` rollback CLOSURE. The two tests below close that
// coverage gap (#1398 ultracode required-fix): they abort a `create_node`
// whose secondary insert SPLITS the root leaf / GROWS the root, then
// assert the crud-closure's Secondary `new_pages` / `root_changes` arms
// (crud.rs Step 1/Step 2 of the Z-1 (b) rollback closure) actually ran.
//
// SEED MATH. `create_node(InlineU32Pair(a, b))` inserts TWO distinct
// secondary keys — `(label, INLINE_U32A_PROPERTY_KEY, a)` and
// `(label, INLINE_U32B_PROPERTY_KEY, b)` — because the property_key
// discriminator (A=1, B=2) differs. Keys sort tenant→label→property_key
// →value (secondary_btree.rs §"Keys sort"), so per node we add 2 fresh
// leaf entries when both `a` and `b` are distinct across the seed. The
// root leaf holds LEAF_CAPACITY = 127 entries.
// ─────────────────────────────────────────────────────────────────────

/// **Test (a) — GREEN, crud-closure leaf split.** An aborted
/// `create_node` whose secondary insert SPLITS a leaf (allocating a
/// fresh page recorded in the mutation log's `new_pages`) leaves NO
/// leaked page after `commit()` fails: the crud-closure Secondary
/// `new_pages` arm (`crud.rs` Step 2 → `rollback_remove_page`) drains
/// the fresh split page. The seed grows the root ONCE (a committed
/// 2-level tree), so the doomed insert splits a leaf into the
/// already-internal root — exercising `new_pages` WITHOUT a
/// `root_changes` grow-root, isolating the split-page drain arm.
///
/// RED-on-revert: disable the crud-closure Secondary `new_pages` arm
/// (the `PageStoreKind::Secondary => secondary.rollback_remove_page(..)`
/// body in `crud.rs` Step 2) → the fresh split page leaks and this test
/// FAILS with `len() > pages_before`. The existing 8 tests stay GREEN
/// under that disable (the coverage gap this test closes).
#[test]
fn rc_rollback_aborted_insert_forcing_leaf_split_through_commit() {
    let mut stack = build_wal_stack();
    let label = LabelId::new(11);

    // Seed 95 nodes (i = 0..94) ⇒ 190 distinct keys. Empirically (the
    // ascending-key fill), this leaves a 2-level tree of exactly 3 pages
    // — one internal root + two leaves — and the NEXT ascending node
    // (i = 95, its B-key) splits a leaf that is at LEAF_CAPACITY,
    // promoting one separator into the (2-child, fanout-255) internal
    // root WITHOUT growing it. So the doomed insert below fires the
    // crud-closure Secondary `new_pages` arm and NOT `root_changes`,
    // isolating the split-page drain. (See the `zz_probe`-derived
    // transition table in the PR: node 63 grows the root 1→3 pages;
    // node 95 is a pure leaf split 3→4 pages, ≈160 separators clear of
    // INTERNAL_CAPACITY = 254.)
    for i in 0..95u32 {
        let mut tx = stack.mgr.begin(TenantId::DEFAULT);
        create_node(
            &stack.store,
            &mut tx,
            TenantId::DEFAULT,
            label,
            // Distinct A (property_key 1) and distinct B (property_key 2)
            // per node; B sorts after every A regardless of value.
            &PropertyData::InlineU32Pair(i, 1_000_000 + i),
        )
        .unwrap();
        commit(tx, &stack.store).unwrap();
    }

    // Deterministic: the ascending fill leaves exactly a 3-page 2-level
    // tree (internal root + 2 leaves), so the doomed leaf split promotes
    // into the existing root (no grow-root).
    let pages_before = stack.secondary.page_store().len();
    assert_eq!(
        pages_before, 3,
        "seed must leave a 2-level tree (internal root + 2 leaves = 3 \
         pages) so the doomed insert splits a leaf INTO the root without \
         growing it; got {pages_before} page(s)",
    );
    // A seed value is present pre-abort.
    let seed_hits_before = stack.secondary.lookup(email_key(11, 0)).unwrap();
    assert_eq!(seed_hits_before.len(), 1, "seed A-value 0 must be indexed");

    // Shut down the WAL so the next commit fails at Phase 2.
    if let Some(w) = stack.writer.take() {
        w.shutdown().expect("wal shutdown");
    }

    // Doomed insert: node 95 — the next ascending node. Its B-key
    // (1_000_095) lands in the rightmost leaf which is at LEAF_CAPACITY,
    // so the leaf splits (fresh page → mutation-log `new_pages`) and the
    // separator promotes into the non-full internal root (no grow-root).
    // Then the commit's Phase-2 fsync fails, driving the crud rollback
    // closure.
    let mut tx = stack.mgr.begin(TenantId::DEFAULT);
    let _doomed = create_node(
        &stack.store,
        &mut tx,
        TenantId::DEFAULT,
        label,
        &PropertyData::InlineU32Pair(95, 1_000_095),
    )
    .unwrap();
    let err = commit(tx, &stack.store).expect_err("commit must fail with WAL down");
    assert!(
        matches!(
            err,
            CrudError::Mvcc(arcgraph_core::ArcGraphError::WalErrorRolledBack { .. })
        ),
        "expected WalErrorRolledBack, got {err:?}",
    );

    // Structural consistency post-rollback:
    // 1. NO leaked pages — the crud-closure `new_pages` arm removed the
    //    fresh split page; `page_mutations` restored the split leaf +
    //    parent bytes. The page set is exactly the pre-abort set.
    assert_eq!(
        stack.secondary.page_store().len(),
        pages_before,
        "Z-1 F-1 (crud closure): the aborted leaf-split page must be \
         drained by the Secondary new_pages arm (no leak)",
    );
    // 2. The aborted node's values are NOT live entries.
    assert!(
        stack
            .secondary
            .lookup(email_key(11, 95))
            .unwrap()
            .is_empty(),
        "the aborted insert's A entry must be rolled back",
    );
    assert!(
        stack
            .secondary
            .lookup(email_key(11, 1_000_095))
            .unwrap()
            .is_empty(),
        "the aborted insert's B entry must be rolled back",
    );
    // 3. A seed entry survives intact — rollback did not corrupt the
    //    surviving structure (also proves the root is still readable).
    assert_eq!(
        stack.secondary.lookup(email_key(11, 0)).unwrap().len(),
        1,
        "the seed entry must survive the rollback intact",
    );
}

/// **Test (b) — GREEN, crud-closure grow-root.** An aborted
/// `create_node` whose secondary insert SPLITS the root leaf AND GROWS
/// THE ROOT (a new internal root installed, the old root pushed onto the
/// mutation log's `root_changes`) leaves the index structurally
/// consistent after `commit()` fails: the crud-closure Secondary
/// `root_changes` arm (`crud.rs` Step 1 → `rollback_restore_root`)
/// restores `root_cache` to the old root and clears the pending
/// grow-root stash, while the `new_pages` arm removes the aborted new
/// root + fresh right-leaf pages.
///
/// The seed fills the initial single root LEAF to 126 entries (63 nodes
/// × 2 distinct keys), so the doomed node's A-key fills it to 127 and
/// its B-key overflows → root-leaf split → `grow_root` (the leaf IS the
/// root, so the split has no parent to absorb it). `grow_root` pushes
/// `(SECONDARY, old_root)` onto `log.root_changes`.
///
/// RED-on-revert: disable the crud-closure Secondary `root_changes` arm
/// (the `IndexHandle::SECONDARY => secondary.rollback_restore_root(..)`
/// body in `crud.rs` Step 1). Then `root_cache` is left pointing at the
/// aborted NEW root — which the still-live `new_pages` arm has REMOVED
/// from the page store — so a post-abort `lookup` (which resolves
/// `root()` from the stale cache and latches the removed page) FAILS
/// with `MissingPage`. Under the shipped arm the lookup succeeds. The
/// existing 8 tests stay GREEN under that disable (the coverage gap this
/// test closes).
#[test]
fn rc_rollback_aborted_insert_forcing_grow_root_through_commit() {
    let mut stack = build_wal_stack();
    let label = LabelId::new(12);

    // Seed 63 nodes ⇒ 126 distinct keys, leaving the single root LEAF at
    // 126 entries (< LEAF_CAPACITY 127, so the seed never splits — the
    // tree stays a single leaf / single page).
    for i in 0..63u32 {
        let mut tx = stack.mgr.begin(TenantId::DEFAULT);
        create_node(
            &stack.store,
            &mut tx,
            TenantId::DEFAULT,
            label,
            &PropertyData::InlineU32Pair(i, 1_000_000 + i),
        )
        .unwrap();
        commit(tx, &stack.store).unwrap();
    }

    // Still a single leaf ⇒ exactly one secondary page. The doomed
    // insert below will split it and grow the root.
    let pages_before = stack.secondary.page_store().len();
    assert_eq!(
        pages_before, 1,
        "seed must leave a single root leaf (one page) so the doomed \
         insert grows the root; got {pages_before} page(s)",
    );
    let seed_hits_before = stack.secondary.lookup(email_key(12, 0)).unwrap();
    assert_eq!(seed_hits_before.len(), 1, "seed A-value 0 must be indexed");

    // Shut down the WAL so the next commit fails at Phase 2.
    if let Some(w) = stack.writer.take() {
        w.shutdown().expect("wal shutdown");
    }

    // Doomed insert: node 64's A-key fills the root leaf to 127, its
    // B-key overflows → root-leaf split with no parent → `grow_root`
    // installs a new internal root and pushes the old root onto
    // `log.root_changes`. Then Phase-2 fsync fails → crud rollback
    // closure runs (Step 1 restores the root; Step 2 removes the fresh
    // pages).
    let mut tx = stack.mgr.begin(TenantId::DEFAULT);
    let _doomed = create_node(
        &stack.store,
        &mut tx,
        TenantId::DEFAULT,
        label,
        &PropertyData::InlineU32Pair(500_000, 2_000_000),
    )
    .unwrap();
    let err = commit(tx, &stack.store).expect_err("commit must fail with WAL down");
    assert!(
        matches!(
            err,
            CrudError::Mvcc(arcgraph_core::ArcGraphError::WalErrorRolledBack { .. })
        ),
        "expected WalErrorRolledBack, got {err:?}",
    );

    // Structural consistency post-rollback:
    // 1. The root was restored to the pre-grow-root single leaf — a
    //    lookup resolves `root()` from the restored cache and finds a
    //    MAPPED page. (Under a disabled `root_changes` arm this errors
    //    with MissingPage because the cache still names the removed new
    //    root — the RED-on-revert signal for this arm.)
    let seed_hits = stack
        .secondary
        .lookup(email_key(12, 0))
        .expect("post-abort lookup must resolve a mapped root (root_changes arm)");
    assert_eq!(
        seed_hits.len(),
        1,
        "the seed entry must survive the grow-root rollback intact",
    );
    // 2. NO leaked pages — back to the single pre-abort leaf (the
    //    `new_pages` arm removed both the fresh right leaf and the fresh
    //    internal root; `page_mutations` restored the root leaf bytes).
    assert_eq!(
        stack.secondary.page_store().len(),
        pages_before,
        "Z-1 F-1 (crud closure): the aborted grow-root pages must be \
         drained; the store is back to the single pre-abort leaf",
    );
    // 3. The aborted node's value is NOT a live entry.
    assert!(
        stack
            .secondary
            .lookup(email_key(12, 500_000))
            .unwrap()
            .is_empty(),
        "the aborted insert's A entry must be rolled back",
    );
}
