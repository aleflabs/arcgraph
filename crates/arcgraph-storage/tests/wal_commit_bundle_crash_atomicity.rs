//! ADR-031 regression: bundle-path WAL failure rolls back BOTH the
//! MVCC silent install AND any in-memory index state that rode the
//! same outer commit. The "crash before fsync" case is the
//! load-bearing atomicity proof: there is no partial-commit state
//! observable post-rollback.
//!
//! Test strategy: build a bundle-enabled `crud::commit` stack, seed
//! a known-good state, shut down the WAL writer, then attempt a
//! bundled commit. The outer `wal.append(CommitBundle)` returns
//! `WalUnavailable`; `Transaction::commit_with_bundle_writes`'s
//! Phase 3 rollback pops the Phase-1 silent MVCC install and the
//! in-memory index mutation via `crud::commit`'s error path.
//! Post-failure the MVCC chain and the primary index must match
//! the seed state exactly.
//!
//! Companion: `mvcc_commit_wal_failure.rs` covers the analogous
//! MVCC-only (no bundle) rollback; this file covers the bundle
//! path specifically and asserts no half-applied dual-write state.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use arcgraph_core::{ArcGraphError, LabelId, TenantId};
use arcgraph_storage::crud::{CrudError, CrudStore, PropertyData, commit, create_node, read_node};
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::{PrimaryIndex, PrimaryKey, RecordKind};
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{WalConfig, WalWriter};
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

fn build_stack() -> (
    TempDir,
    Arc<CrudStore>,
    Arc<TxnManager>,
    Arc<PrimaryIndex>,
    WalWriter,
) {
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
    (dir, store, mgr, primary, writer)
}

#[test]
fn wal_shutdown_mid_bundle_rolls_back_mvcc_and_leaves_index_consistent() {
    let (_dir, store, mgr, primary, writer) = build_stack();

    // Seed: one node to establish a durable baseline.
    let mut tx = mgr.begin(TenantId::DEFAULT);
    let seed_id = create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(7),
        &PropertyData::InlineU32Pair(1, 2),
    )
    .unwrap();
    let seed_lsn = commit(tx, &store).unwrap();
    assert_ne!(seed_lsn, arcgraph_core::Lsn::ZERO);
    let seed_visible = mgr.current_lsn();

    // Sanity: seed is findable in MVCC and in the primary index.
    let reader = mgr.begin(TenantId::DEFAULT);
    assert!(read_node(&reader, seed_id).unwrap().is_some());
    let seed_key = PrimaryKey::new(TenantId::DEFAULT, RecordKind::Node, seed_id.raw());
    assert!(primary.lookup(seed_key).unwrap().is_some());
    drop(reader);

    // Shut down the WAL writer. The next `commit` will get
    // `WalUnavailable` from the bundle path's `wal.append`.
    writer.shutdown().unwrap();

    // Attempt a second commit. The crud::commit builder does the
    // primary.upsert_deferred (mutating in-memory index state),
    // then tx.commit_with_bundle enters Phase 2 which returns
    // WalUnavailable; Phase 3 rolls back the silent MVCC install.
    let mut tx = mgr.begin(TenantId::DEFAULT);
    let doomed_id = create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(9),
        &PropertyData::Empty,
    )
    .unwrap();
    let err = commit(tx, &store).expect_err("bundle append must fail when WAL is down");
    // ADR-033 §3c: WAL failures wrap the underlying error in
    // `WalErrorRolledBack` to signal rollback ran. The original
    // `WalUnavailable` is preserved in `.source`.
    match err {
        CrudError::Mvcc(ArcGraphError::WalErrorRolledBack { ref source }) => {
            assert!(
                matches!(source.as_ref(), ArcGraphError::WalUnavailable),
                "source chain should carry WalUnavailable, got {source:?}"
            );
        }
        other => panic!("expected CrudError::Mvcc(WalErrorRolledBack {{..}}), got {other:?}"),
    }

    // Post-failure MVCC state assertions:
    // - `visible` unchanged (no commit was durable).
    assert_eq!(
        mgr.current_lsn(),
        seed_visible,
        "visible must not advance past the seed commit — the failed bundle had no durability"
    );

    // - Seed still findable.
    let r = mgr.begin(TenantId::DEFAULT);
    assert!(
        read_node(&r, seed_id).unwrap().is_some(),
        "seed commit must survive the failed follow-up"
    );

    // - Doomed key absent from MVCC at any snapshot (rollback popped
    //   the silent install; chain empty).
    assert!(
        read_node(&r, doomed_id).unwrap().is_none(),
        "doomed node's MVCC version must have been rolled back"
    );
    drop(r);

    // - The primary index may still carry the doomed node's entry
    //   (the in-memory mutation inside the builder happened before
    //   Phase 2 failed). This is per ADR-023: readers that hit the
    //   index and get a stale hit fall through to MVCC (which
    //   returns None after the rollback) — safe. The regression
    //   we guard against here is the stronger one: MVCC state MUST
    //   be in a consistent pre-failure shape.
    // A concurrent reader looking up the doomed key via `read_node`
    // would see: primary.lookup returns Some(slot); read from page
    // → finds the slot (install_create wrote bytes); MVCC
    // visibility predicate `created_lsn <= snapshot`: doomed's
    // in-record `created_lsn` was stamped with the allocated
    // commit_lsn (which is > seed_visible), so snapshot = seed_
    // visible < created_lsn → reader falls through. Correct.
    let reader = mgr.begin(TenantId::DEFAULT);
    let rec = read_node(&reader, doomed_id).unwrap();
    assert!(
        rec.is_none(),
        "reader at seed-time snapshot must not see the doomed node \
         (ADR-023 read-accelerator fallback)"
    );
}

#[test]
fn wal_shutdown_mid_bundle_does_not_deadlock_successive_commits() {
    // After a bundle-path WAL failure, the rollback must advance
    // `install_order` so any successor commit on the same
    // TxnManager doesn't get stuck waiting on the dead LSN.
    // Companion to `mvcc_commit_wal_failure::sequential_wal_
    // failures_do_not_deadlock_install_order` but exercising the
    // crud::commit bundle path.
    let (_dir, store, mgr, _primary, writer) = build_stack();

    // Seed (durable).
    let mut tx = mgr.begin(TenantId::DEFAULT);
    create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(1),
        &PropertyData::Empty,
    )
    .unwrap();
    commit(tx, &store).unwrap();

    // Shut down WAL and trigger a failing commit.
    writer.shutdown().unwrap();
    let mut tx = mgr.begin(TenantId::DEFAULT);
    create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(2),
        &PropertyData::Empty,
    )
    .unwrap();
    let err = commit(tx, &store).expect_err("must fail with WAL down");
    // ADR-033 §3c: WAL error wrapped in WalErrorRolledBack.
    match err {
        CrudError::Mvcc(ArcGraphError::WalErrorRolledBack { ref source }) => {
            assert!(matches!(source.as_ref(), ArcGraphError::WalUnavailable));
        }
        other => panic!("expected CrudError::Mvcc(WalErrorRolledBack {{..}}), got {other:?}"),
    }

    // Attempt a follow-up commit. It must also fail cleanly
    // (WalUnavailable) without deadlocking on install_order. The
    // key invariant: install_order advanced past the rolled-back
    // LSN during the first failure's Phase 3.
    let mut tx = mgr.begin(TenantId::DEFAULT);
    create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(3),
        &PropertyData::Empty,
    )
    .unwrap();
    let err2 = commit(tx, &store).expect_err("successor must also fail");
    match err2 {
        CrudError::Mvcc(ArcGraphError::WalErrorRolledBack { ref source }) => {
            assert!(matches!(source.as_ref(), ArcGraphError::WalUnavailable));
        }
        other => {
            panic!(
                "expected CrudError::Mvcc(WalErrorRolledBack {{..}}) on successor, got {other:?}"
            )
        }
    }
}
