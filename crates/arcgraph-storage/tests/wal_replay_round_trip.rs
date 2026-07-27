//! PR #79 review fold-in — Y-5 round-trip integration tests, plus
//! N-2 (issue #81) blob round-trip tests.
//!
//! These tests drive the REAL commit flow:
//!   create_node / update_node / delete_node via the CRUD API →
//!   commit → shutdown → fresh stack → recover_from_wal →
//!   read_node_with_store round-trip.
//!
//! The review on the first cut of PR #79 identified two critical
//! correctness bugs:
//!
//! - **X-1** (`install_or_replace` byte-equality over-strictness):
//!   `PrimaryIndex::new()` emits one legacy `IndexPage = 11`
//!   record; later commits emit `CommitBundle = 12` entries for
//!   the SAME `page_id` with post-commit bytes. Byte-equality
//!   check on re-install halts replay with a false
//!   "Lemma I2 violation" corruption error.
//!
//! - **X-2** (record pages absent from the bundle): the CRUD
//!   install paths mutate `RecordPageStore` in-memory but never
//!   stage post-mutation page bytes into the bundle. On replay
//!   `RecordPageStore` starts empty; `read_node_with_store`
//!   → `records.latch(slot.page)?` → `MissingPage` error.
//!
//! N-2 (issue #81) closes the parallel gap for BlobStore: pre-N-2
//! `PropertyData::Blob` workloads hit
//! `BlobError::MissingHead { tenant, head }` post-replay because
//! the v3 bundle reserved `BundlePageKind::Blob` but the executor
//! never routed it, AND the commit builder never staged blob chain
//! page bytes. After N-2 lands the four `..._blob_..._post_replay`
//! tests below pass.
//!
//! Before the X-1 + X-2 + N-2 fixes these tests FAIL with exactly
//! the reviewer's repro signature. After the fixes they pass.

use std::sync::Arc;

use arcgraph_core::{LabelId, TenantId, TypeId};
use arcgraph_index::{PropertyValue, SecondaryIndex, SecondaryKey};
use arcgraph_storage::crud::{
    CrudStore, INLINE_U32A_PROPERTY_KEY, PropertyData, commit, create_node, create_rel,
    crud_allocator_seed_handle, delete_node_with_store, read_node_with_store, read_rel_with_store,
    update_node,
};
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::{PrimaryIndex, PrimaryKey, RecordKind};
use arcgraph_storage::records::SlottedPageRef;
use arcgraph_storage::secondary_handle::SecondaryIndexHandle;
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{
    AllocatorSeedHandle, BlobStoreHandle, PageStoreTarget, PrimaryPageStoreHandle,
    RecordPageStoreHandle, SecondaryPageStoreHandle, WalConfig, WalWriter, recover_from_wal,
};
use tempfile::TempDir;

// ─── Helpers ────────────────────────────────────────────────────

fn test_wal_config(dir: &std::path::Path) -> WalConfig {
    WalConfig {
        dir: dir.to_path_buf(),
        segment_size_bytes: 64 * 1024 * 1024,
        group_commit_window: std::time::Duration::from_millis(2),
        group_commit_max_batch: 4,
        metrics_sink: None,
        encryption: None,
        inflight_budget_bytes: None,
    }
}

/// Build a full CRUD stack (WalWriter + TxnManager + PrimaryIndex
/// + CrudStore). Mirrors the production wiring pattern.
///
/// Issue #129 P0 fix: returns `Arc<CrudStore>` (was `CrudStore`)
/// so the seed-handle wiring in [`recover_stack`] can clone an Arc
/// for replay-time allocator seeding without forcing the test
/// callers to track ownership.
fn build_stack(
    wal_dir: &std::path::Path,
) -> (
    WalWriter,
    Arc<TxnManager>,
    Arc<PrimaryIndex>,
    Arc<CrudStore>,
) {
    let writer = WalWriter::spawn(test_wal_config(wal_dir)).unwrap();
    let handle = writer.handle();
    let mgr = Arc::new(TxnManager::with_wal(handle.clone()));
    let alloc = Arc::new(PageAllocator::new());
    let primary = Arc::new(
        PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&alloc), Some(handle.clone())).unwrap(),
    );
    let store = Arc::new(CrudStore::new_with_index(
        Some(handle.clone()),
        Arc::clone(&primary),
        Arc::clone(&alloc),
    ));
    (writer, mgr, primary, store)
}

/// Recover a fresh stack from `wal_dir`. The returned stack is
/// equivalent to a process-restart post-crash.
///
/// Wires the replay target to include BOTH the primary page store
/// and the record page store (X-2 fix). Before the X-2 fix the
/// record-store leg was absent from the bundle, so replay would
/// route nothing to the record store even if the target
/// registered it. After X-2 lands the v3 bundle's staged_pages
/// dispatches by PageStoreKind into the right store.
fn recover_stack(
    wal_dir: &std::path::Path,
) -> (
    WalWriter,
    Arc<TxnManager>,
    Arc<PrimaryIndex>,
    Arc<CrudStore>,
) {
    let writer = WalWriter::spawn(test_wal_config(wal_dir)).unwrap();
    let handle = writer.handle();
    let mgr = Arc::new(TxnManager::with_wal(handle.clone()));
    let alloc = Arc::new(PageAllocator::new());
    let primary = Arc::new(
        PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&alloc), Some(handle.clone())).unwrap(),
    );
    let store = Arc::new(CrudStore::new_with_index(
        Some(handle.clone()),
        Arc::clone(&primary),
        Arc::clone(&alloc),
    ));
    let primary_handle: Arc<dyn PrimaryPageStoreHandle> =
        Arc::clone(primary.page_store()) as Arc<dyn PrimaryPageStoreHandle>;
    let records_handle: Arc<dyn RecordPageStoreHandle> = Arc::clone(
        store
            .records()
            .expect("CrudStore constructed via new_with_index exposes record store"),
    ) as Arc<dyn RecordPageStoreHandle>;
    // N-2 (issue #81): wire the BlobStoreHandle too so replay can
    // reconstruct chain pages into the CrudStore's fresh BlobStore.
    // Pre-N-2 a PropertyData::Blob workload hit MissingHead post-
    // replay because the bundle's Blob-kind entries had nowhere to
    // land.
    let blob_handle: Arc<dyn BlobStoreHandle> =
        Arc::clone(store.blob_store()) as Arc<dyn BlobStoreHandle>;
    // Issue #129 P0 fix: wire the AllocatorSeedHandle so v4
    // bundle `allocator_advances` entries seed live counters in
    // commit_lsn order. Without this, post-recovery `alloc_node`
    // re-issues NodeIds that pre-fault commits consumed (ADR-034
    // D-1 violation).
    let allocator_seed: Arc<dyn AllocatorSeedHandle> =
        crud_allocator_seed_handle(Arc::clone(&store), Arc::clone(&alloc));
    let target = PageStoreTarget::primary_only(primary_handle)
        .with_record_store(records_handle)
        .with_blob_store(blob_handle)
        .with_allocator_seed(allocator_seed);
    let _report = recover_from_wal(wal_dir, Arc::clone(&mgr), target, None).unwrap();
    (writer, mgr, primary, store)
}

// ─── Y-5 round-trip tests ───────────────────────────────────────

#[test]
fn round_trip_crud_commit_shutdown_recover_read() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();

    // Pre-crash: real commit path.
    let (writer, mgr, primary, store) = build_stack(&wal_dir);
    let mut tx = mgr.begin(TenantId::DEFAULT);
    let id = create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(42),
        &PropertyData::InlineU32Pair(42, 99),
    )
    .unwrap();
    commit(tx, &store).unwrap();

    // Shutdown.
    writer.shutdown().unwrap();
    drop(store);
    drop(primary);
    drop(mgr);

    // Post-crash: fresh stack + recovery.
    let (writer2, mgr2, _primary2, store2) = recover_stack(&wal_dir);
    let tx2 = mgr2.begin(TenantId::DEFAULT);
    let rec = read_node_with_store(&store2, &tx2, id)
        .unwrap()
        .expect("node should be readable post-replay");
    assert_eq!(rec.label_id, 42);
    assert_eq!(rec.inline_u32a, 42);
    assert_eq!(rec.inline_u32b, 99);
    writer2.shutdown().unwrap();
}

#[test]
fn round_trip_update_node_post_replay() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();

    let (writer, mgr, primary, store) = build_stack(&wal_dir);
    // Commit 1: create.
    let mut tx = mgr.begin(TenantId::DEFAULT);
    let id = create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(7),
        &PropertyData::InlineU32Pair(1, 2),
    )
    .unwrap();
    commit(tx, &store).unwrap();
    // Commit 2: update.
    let mut tx = mgr.begin(TenantId::DEFAULT);
    update_node(&store, &mut tx, id, &PropertyData::InlineU32Pair(100, 200)).unwrap();
    commit(tx, &store).unwrap();

    writer.shutdown().unwrap();
    drop(store);
    drop(primary);
    drop(mgr);

    let (writer2, mgr2, _primary2, store2) = recover_stack(&wal_dir);
    let tx2 = mgr2.begin(TenantId::DEFAULT);
    let rec = read_node_with_store(&store2, &tx2, id)
        .unwrap()
        .expect("node should be readable post-replay");
    assert_eq!(rec.inline_u32a, 100);
    assert_eq!(rec.inline_u32b, 200);
    writer2.shutdown().unwrap();
}

#[test]
fn round_trip_delete_node_post_replay() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();

    let (writer, mgr, primary, store) = build_stack(&wal_dir);
    let mut tx = mgr.begin(TenantId::DEFAULT);
    let id = create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(1),
        &PropertyData::Empty,
    )
    .unwrap();
    commit(tx, &store).unwrap();
    let mut tx = mgr.begin(TenantId::DEFAULT);
    delete_node_with_store(&store, &mut tx, id).unwrap();
    commit(tx, &store).unwrap();

    writer.shutdown().unwrap();
    drop(store);
    drop(primary);
    drop(mgr);

    let (writer2, mgr2, _primary2, store2) = recover_stack(&wal_dir);
    let tx2 = mgr2.begin(TenantId::DEFAULT);
    let rec = read_node_with_store(&store2, &tx2, id).unwrap();
    assert!(
        rec.is_none(),
        "deleted node should NOT be readable post-replay"
    );
    writer2.shutdown().unwrap();
}

#[test]
fn round_trip_multi_commit_post_replay() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();

    let (writer, mgr, primary, store) = build_stack(&wal_dir);
    let mut ids = Vec::new();
    for i in 0u32..10 {
        let mut tx = mgr.begin(TenantId::DEFAULT);
        let id = create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(i),
            &PropertyData::InlineU32Pair(i * 10, i * 100),
        )
        .unwrap();
        commit(tx, &store).unwrap();
        ids.push((id, i));
    }
    writer.shutdown().unwrap();
    drop(store);
    drop(primary);
    drop(mgr);

    let (writer2, mgr2, _primary2, store2) = recover_stack(&wal_dir);
    let tx2 = mgr2.begin(TenantId::DEFAULT);
    for (id, i) in &ids {
        let rec = read_node_with_store(&store2, &tx2, *id)
            .unwrap()
            .unwrap_or_else(|| panic!("node {:?} not readable post-replay", id));
        assert_eq!(rec.inline_u32a, i * 10);
        assert_eq!(rec.inline_u32b, i * 100);
    }
    writer2.shutdown().unwrap();
}

// ─── N-2 (issue #81) blob round-trip tests ──────────────────────
//
// Pre-this-PR these all failed with
// `BlobError::MissingHead { tenant, head }` because the post-replay
// BlobStore was empty — the v3 bundle's `BundlePageKind::Blob`
// entries had nowhere to land. Slice N1 + N2 + N3 close the gap.

use arcgraph_storage::BLOB_CHUNK_BYTES;
use arcgraph_storage::property::{PropertyReadout, decode_node as decode_property_node};

/// Build a deterministic pseudo-random payload of `len` bytes with
/// a `seed`-derived marker so assertion failures point at the
/// offending blob.
fn blob_payload(seed: u8, len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| seed.wrapping_add((i % 251) as u8))
        .collect()
}

#[test]
fn round_trip_node_with_blob_property_post_replay() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();

    let (writer, mgr, primary, store) = build_stack(&wal_dir);
    // Size `> BLOB_CHUNK_BYTES / 2` and `<= BLOB_CHUNK_BYTES` so
    // the chain has exactly one page — covers the "single-page
    // head" branch of decode.
    let payload = blob_payload(0xA5, 4 * 1024);
    let mut tx = mgr.begin(TenantId::DEFAULT);
    let id = create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(9),
        &PropertyData::Blob(payload.clone()),
    )
    .unwrap();
    commit(tx, &store).unwrap();

    writer.shutdown().unwrap();
    drop(store);
    drop(primary);
    drop(mgr);

    let (writer2, mgr2, _primary2, store2) = recover_stack(&wal_dir);
    let tx2 = mgr2.begin(TenantId::DEFAULT);
    let rec = read_node_with_store(&store2, &tx2, id)
        .unwrap()
        .expect("node should be readable post-replay");
    assert_eq!(rec.label_id, 9);
    // Decode the overflow property and dereference through the
    // post-replay BlobStore — the same path a real workload hits.
    match decode_property_node(&rec) {
        PropertyReadout::Overflow(blob_ref) => {
            let round = store2
                .blob_store()
                .get(TenantId::DEFAULT, blob_ref)
                .expect("post-replay blob deref must succeed");
            assert_eq!(round.as_ref(), payload.as_slice());
        }
        other => panic!("expected overflow readout, got {other:?}"),
    }
    writer2.shutdown().unwrap();
}

/// HIGH #810 — a single batched ingest (N property-bearing nodes in ONE
/// `crud::commit`) survives a restart with every blob intact.
///
/// #810 removed the per-record synchronous `PutBlob` WAL fsync; the N
/// blob chains now ride the single `CommitBundle` as
/// `BundlePageKind::Blob` staged pages. This test is the durability
/// counterpart to the throughput assertion in
/// `tests/ingest_batch_commit_fsync.rs`: it proves that folding N
/// per-record fsyncs into one commit fsync does NOT cost durability —
/// after a process-restart-equivalent recover, all N nodes AND their
/// blob payloads dereference correctly from the bundle-reconstructed
/// `BlobStore`. (The pre-#810 path would also survive, but via the
/// redundant standalone PutBlob records; this pins recovery on the
/// in-bundle Blob pages alone.)
#[test]
fn round_trip_batched_blob_ingest_all_survive_post_replay() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();

    const N: u32 = 64;

    let (writer, mgr, primary, store) = build_stack(&wal_dir);

    // ONE transaction, ONE commit — the #810 batched-ingest shape.
    let mut tx = mgr.begin(TenantId::DEFAULT);
    let mut ids: Vec<(arcgraph_core::NodeId, Vec<u8>)> = Vec::with_capacity(N as usize);
    for i in 0..N {
        // Distinct single-page payload per node so a swapped/lost blob
        // is detectable (not just "some blob is present").
        let payload = blob_payload(i as u8, 4 * 1024);
        let id = create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(9),
            &PropertyData::Blob(payload.clone()),
        )
        .unwrap();
        ids.push((id, payload));
    }
    commit(tx, &store).unwrap();

    writer.shutdown().unwrap();
    drop(store);
    drop(primary);
    drop(mgr);

    // Process-restart-equivalent recovery from the WAL dir alone.
    let (writer2, mgr2, _primary2, store2) = recover_stack(&wal_dir);
    let tx2 = mgr2.begin(TenantId::DEFAULT);
    for (id, expected) in &ids {
        let rec = read_node_with_store(&store2, &tx2, *id)
            .unwrap()
            .unwrap_or_else(|| panic!("node {id:?} must be readable post-replay (#810)"));
        assert_eq!(rec.label_id, 9);
        match decode_property_node(&rec) {
            PropertyReadout::Overflow(blob_ref) => {
                let round = store2
                    .blob_store()
                    .get(TenantId::DEFAULT, blob_ref)
                    .unwrap_or_else(|e| {
                        panic!("post-replay blob deref for node {id:?} must succeed (#810): {e}")
                    });
                assert_eq!(
                    round.as_ref(),
                    expected.as_slice(),
                    "node {id:?} blob payload corrupted/lost across restart (#810)"
                );
            }
            other => panic!("node {id:?}: expected overflow readout, got {other:?}"),
        }
    }
    writer2.shutdown().unwrap();
}

#[test]
fn round_trip_node_with_large_blob_multi_page_post_replay() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();

    let (writer, mgr, primary, store) = build_stack(&wal_dir);
    // 4 × BLOB_CHUNK_BYTES + overhang → 5 chain pages; exercises
    // both head (carries total_len) + multiple tails (total_len = 0)
    // on decode, AND validates next_page pointers thread correctly
    // when rebuilt page-by-page via install_or_replace.
    let payload = blob_payload(0x5A, BLOB_CHUNK_BYTES * 4 + 100);
    let expected_chunks = payload.len().div_ceil(BLOB_CHUNK_BYTES);
    assert!(expected_chunks >= 5);

    let mut tx = mgr.begin(TenantId::DEFAULT);
    let id = create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(42),
        &PropertyData::Blob(payload.clone()),
    )
    .unwrap();
    commit(tx, &store).unwrap();

    writer.shutdown().unwrap();
    drop(store);
    drop(primary);
    drop(mgr);

    let (writer2, mgr2, _primary2, store2) = recover_stack(&wal_dir);
    let tx2 = mgr2.begin(TenantId::DEFAULT);
    let rec = read_node_with_store(&store2, &tx2, id)
        .unwrap()
        .expect("multi-page blob node should be readable post-replay");
    match decode_property_node(&rec) {
        PropertyReadout::Overflow(blob_ref) => {
            let round = store2
                .blob_store()
                .get(TenantId::DEFAULT, blob_ref)
                .expect("post-replay multi-page blob deref must succeed");
            assert_eq!(round.len(), payload.len());
            assert_eq!(round.as_ref(), payload.as_slice());
        }
        other => panic!("expected overflow readout, got {other:?}"),
    }
    writer2.shutdown().unwrap();
}

#[test]
fn round_trip_update_blob_property_post_replay() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();

    let (writer, mgr, primary, store) = build_stack(&wal_dir);
    let payload_a = blob_payload(0x11, 6_000);
    let payload_b = blob_payload(0xEE, 12_000);

    // Commit 1: create node with blob A.
    let mut tx = mgr.begin(TenantId::DEFAULT);
    let id = create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(1),
        &PropertyData::Blob(payload_a.clone()),
    )
    .unwrap();
    commit(tx, &store).unwrap();

    // Commit 2: update node, replacing blob A with blob B. The
    // old chain's pages stay in BlobStore (no GC at v1.0 per
    // ADR-020); the MVCC chain's latest version's
    // NodeRecord.property_ref now points at B's fresh chain.
    let mut tx = mgr.begin(TenantId::DEFAULT);
    update_node(&store, &mut tx, id, &PropertyData::Blob(payload_b.clone())).unwrap();
    commit(tx, &store).unwrap();

    writer.shutdown().unwrap();
    drop(store);
    drop(primary);
    drop(mgr);

    let (writer2, mgr2, _primary2, store2) = recover_stack(&wal_dir);
    let tx2 = mgr2.begin(TenantId::DEFAULT);
    let rec = read_node_with_store(&store2, &tx2, id)
        .unwrap()
        .expect("updated node should be readable post-replay");
    match decode_property_node(&rec) {
        PropertyReadout::Overflow(blob_ref) => {
            let round = store2
                .blob_store()
                .get(TenantId::DEFAULT, blob_ref)
                .expect("post-replay blob deref (B) must succeed");
            // The latest version's blob_ref points at B's chain;
            // bytes must equal B (not A).
            assert_ne!(round.as_ref(), payload_a.as_slice());
            assert_eq!(round.as_ref(), payload_b.as_slice());
        }
        other => panic!("expected overflow readout, got {other:?}"),
    }
    writer2.shutdown().unwrap();
}

#[test]
fn round_trip_delete_node_with_blob_property_post_replay() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();

    let (writer, mgr, primary, store) = build_stack(&wal_dir);
    let payload = blob_payload(0xC3, 10_000);

    // Commit 1: create node with blob A.
    let mut tx = mgr.begin(TenantId::DEFAULT);
    let id = create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(4),
        &PropertyData::Blob(payload.clone()),
    )
    .unwrap();
    commit(tx, &store).unwrap();
    // Snapshot the head page id for the post-replay assertion.
    let pre_crash_blob_ref = {
        let tx_read = mgr.begin(TenantId::DEFAULT);
        let rec = read_node_with_store(&store, &tx_read, id).unwrap().unwrap();
        match decode_property_node(&rec) {
            PropertyReadout::Overflow(b) => b,
            other => panic!("expected overflow pre-crash, got {other:?}"),
        }
    };

    // Commit 2: delete the node. ADR-020 says no blob GC at v1.0,
    // so the chain pages persist in BlobStore even though the
    // MVCC chain now has a tombstone.
    let mut tx = mgr.begin(TenantId::DEFAULT);
    delete_node_with_store(&store, &mut tx, id).unwrap();
    commit(tx, &store).unwrap();

    writer.shutdown().unwrap();
    drop(store);
    drop(primary);
    drop(mgr);

    let (writer2, mgr2, _primary2, store2) = recover_stack(&wal_dir);
    let tx2 = mgr2.begin(TenantId::DEFAULT);
    let rec = read_node_with_store(&store2, &tx2, id).unwrap();
    assert!(
        rec.is_none(),
        "deleted node should NOT be readable post-replay"
    );
    // The orphaned chain pages must still be in BlobStore —
    // `get` against the pre-crash head resolves to the original
    // payload. Validates that blob replay is independent of MVCC
    // tombstoning (blob GC is out of v1.0 scope per ADR-020).
    let round = store2
        .blob_store()
        .get(TenantId::DEFAULT, pre_crash_blob_ref)
        .expect("orphaned blob chain should still be reconstructable");
    assert_eq!(round.as_ref(), payload.as_slice());
    writer2.shutdown().unwrap();
}

#[test]
fn round_trip_rel_post_replay() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();

    let (writer, mgr, primary, store) = build_stack(&wal_dir);
    let mut tx = mgr.begin(TenantId::DEFAULT);
    let src = create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(1),
        &PropertyData::Empty,
    )
    .unwrap();
    let dst = create_node(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(2),
        &PropertyData::Empty,
    )
    .unwrap();
    let rel_id = create_rel(
        &store,
        &mut tx,
        TenantId::DEFAULT,
        src,
        dst,
        TypeId::new(5),
        &PropertyData::InlineU32Pair(77, 88),
    )
    .unwrap();
    commit(tx, &store).unwrap();

    writer.shutdown().unwrap();
    drop(store);
    drop(primary);
    drop(mgr);

    let (writer2, mgr2, _primary2, store2) = recover_stack(&wal_dir);
    let tx2 = mgr2.begin(TenantId::DEFAULT);
    let n_src = read_node_with_store(&store2, &tx2, src).unwrap().unwrap();
    let n_dst = read_node_with_store(&store2, &tx2, dst).unwrap().unwrap();
    assert_eq!(n_src.label_id, 1);
    assert_eq!(n_dst.label_id, 2);
    let rel = read_rel_with_store(&store2, &tx2, rel_id)
        .unwrap()
        .expect("rel should be readable post-replay");
    assert_eq!(rel.src_id, src.raw());
    assert_eq!(rel.dst_id, dst.raw());
    assert_eq!(rel.type_id, 5);
    assert_eq!(rel.inline_u32a, 77);
    assert_eq!(rel.inline_u32b, 88);
    writer2.shutdown().unwrap();
}

/// #811 acceptance (1) — first edge from cold dual-writes cleanly with
/// ZERO store divergence, DURABLE leg (survives WAL replay).
///
/// The sibling `round_trip_rel_post_replay` above asserts only that the
/// rel is *readable* post-replay — and `read_rel_with_store` falls back
/// to MVCC on a primary-index miss, so it stays GREEN even when the
/// dual-write record store / primary index never received the rel. That
/// makes it green-both-ways for #811: it cannot catch the silent
/// divergence.
///
/// This test asserts the dual-write target DIRECTLY, post-replay: the
/// rel must be in the reconstructed primary index AND the reconstructed
/// record store, on a page DISTINCT from the node page, with both node
/// pages surviving. Pre-#811 the rel record page collided with the node
/// record page (`PageId(1)` from each independent counter) so the rel
/// was never staged into the bundle; post-replay the primary lookup
/// returns `None` and the record store has no rel page → RED. Post-fix
/// the unified record-page domain (one flat keyspace) keeps the ids
/// distinct end-to-end through commit → WAL fsync → replay.
#[test]
fn round_trip_rel_dual_write_in_record_store_post_replay() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();
    let tenant = TenantId::DEFAULT;

    let (writer, mgr, primary, store) = build_stack(&wal_dir);
    let mut tx = mgr.begin(tenant);
    let src = create_node(
        &store,
        &mut tx,
        tenant,
        LabelId::new(1),
        &PropertyData::Empty,
    )
    .unwrap();
    let dst = create_node(
        &store,
        &mut tx,
        tenant,
        LabelId::new(2),
        &PropertyData::Empty,
    )
    .unwrap();
    let rel_id = create_rel(
        &store,
        &mut tx,
        tenant,
        src,
        dst,
        TypeId::new(5),
        &PropertyData::InlineU32Pair(77, 88),
    )
    .unwrap();
    commit(tx, &store).unwrap();

    // Pre-crash localization: the LIVE dual-write already published the
    // rel into the primary index (pre-#811: absent — the collision made
    // install_create fail before it could publish). Asserting here
    // distinguishes a live-path regression from a replay-path one.
    let rel_key = PrimaryKey::new(tenant, RecordKind::Rel, rel_id.raw());
    assert!(
        primary.lookup(rel_key).unwrap().is_some(),
        "#811: the rel must be dual-written into the primary index pre-crash (pre-fix: absent)"
    );

    writer.shutdown().unwrap();
    drop(store);
    drop(primary);
    drop(mgr);

    // Post-crash: fresh stack + recovery, then assert the dual-write
    // target (primary index + record store) was reconstructed WITH the
    // rel — not just MVCC-readable.
    let (writer2, _mgr2, primary2, store2) = recover_stack(&wal_dir);
    let records2 = store2.records().expect("dual-write configured");

    let rel_slot = primary2.lookup(rel_key).unwrap().expect(
        "#811: post-replay the rel must be in the reconstructed primary index \
         (pre-fix: None — it was never dual-written/staged)",
    );
    assert!(
        records2.contains(rel_slot.page),
        "#811: post-replay the record store must map the rel page"
    );
    {
        let latch = records2.latch(rel_slot.page).unwrap();
        let g = latch.read();
        let page = SlottedPageRef::open(g.as_ref().as_ref()).unwrap();
        let rec = page
            .read_rel(rel_slot.slot)
            .unwrap()
            .expect("rel slot live post-replay");
        assert_eq!(rec.id, rel_id.raw());
        assert_eq!(rec.src_id, src.raw());
        assert_eq!(rec.dst_id, dst.raw());
        assert_eq!(rec.type_id, 5);
        assert_eq!(rec.inline_u32a, 77);
        assert_eq!(rec.inline_u32b, 88);
    }

    // The rel page is DISTINCT from the node page, and the node page
    // survived replay (no cross-kind install_or_replace overwrite).
    let node_key = PrimaryKey::new(tenant, RecordKind::Node, src.raw());
    let node_slot = primary2
        .lookup(node_key)
        .unwrap()
        .expect("node indexed post-replay");
    assert_ne!(
        rel_slot.page, node_slot.page,
        "#811: post-replay the rel and node record pages must occupy distinct ids"
    );
    assert!(
        records2.contains(node_slot.page),
        "#811: the node record page must survive replay alongside the rel page"
    );

    writer2.shutdown().unwrap();
}

// ─── N-2 Slice N5: secondary index replay verification ─────────────
//
// Reviewer's "unverified" note from PR #79: "Secondary index replay
// path: I did NOT verify. Similar issues may exist." Closes that.
// Pre-PR-#79 X-2 Step C the secondary write path already staged into
// the commit bundle via `insert_property_deferred`; this test pins
// the invariant and catches any future drift.

/// Build a full CRUD stack WITH a secondary index wired in.
fn build_stack_with_secondary(
    wal_dir: &std::path::Path,
) -> (
    WalWriter,
    Arc<TxnManager>,
    Arc<PrimaryIndex>,
    Arc<SecondaryIndex>,
    CrudStore,
) {
    let writer = WalWriter::spawn(test_wal_config(wal_dir)).unwrap();
    let handle = writer.handle();
    let mgr = Arc::new(TxnManager::with_wal(handle.clone()));
    let alloc = Arc::new(PageAllocator::new());
    let primary = Arc::new(
        PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&alloc), Some(handle.clone())).unwrap(),
    );
    let secondary = Arc::new(
        SecondaryIndex::new(Arc::clone(&mgr), Arc::clone(&alloc), Some(handle.clone())).unwrap(),
    );
    let secondary_as_handle: Arc<dyn SecondaryIndexHandle> = Arc::clone(&secondary) as _;
    let store = CrudStore::new_with_indices(
        Some(handle.clone()),
        Arc::clone(&primary),
        Some(secondary_as_handle),
        Arc::clone(&alloc),
    );
    (writer, mgr, primary, secondary, store)
}

/// Recover a CRUD stack WITH a secondary index (plus blob + record).
///
/// Wires all four page stores into the replay target so every
/// `BundlePageKind::{PrimaryIndex,SecondaryIndex,Record,Blob}` entry
/// lands on its owning store. Closes the reviewer's "unverified"
/// note by exercising the SecondaryIndex leg end-to-end.
fn recover_stack_with_secondary(
    wal_dir: &std::path::Path,
) -> (
    WalWriter,
    Arc<TxnManager>,
    Arc<PrimaryIndex>,
    Arc<SecondaryIndex>,
    CrudStore,
) {
    let writer = WalWriter::spawn(test_wal_config(wal_dir)).unwrap();
    let handle = writer.handle();
    let mgr = Arc::new(TxnManager::with_wal(handle.clone()));
    let alloc = Arc::new(PageAllocator::new());
    let primary = Arc::new(
        PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&alloc), Some(handle.clone())).unwrap(),
    );
    let secondary = Arc::new(
        SecondaryIndex::new(Arc::clone(&mgr), Arc::clone(&alloc), Some(handle.clone())).unwrap(),
    );
    let secondary_as_handle: Arc<dyn SecondaryIndexHandle> = Arc::clone(&secondary) as _;
    let store = CrudStore::new_with_indices(
        Some(handle.clone()),
        Arc::clone(&primary),
        Some(secondary_as_handle),
        Arc::clone(&alloc),
    );
    let primary_handle: Arc<dyn PrimaryPageStoreHandle> =
        Arc::clone(primary.page_store()) as Arc<dyn PrimaryPageStoreHandle>;
    let secondary_pages_handle: Arc<dyn SecondaryPageStoreHandle> =
        Arc::clone(secondary.page_store()) as Arc<dyn SecondaryPageStoreHandle>;
    let records_handle: Arc<dyn RecordPageStoreHandle> = Arc::clone(
        store
            .records()
            .expect("CrudStore constructed via new_with_indices exposes record store"),
    ) as Arc<dyn RecordPageStoreHandle>;
    let blob_handle: Arc<dyn BlobStoreHandle> =
        Arc::clone(store.blob_store()) as Arc<dyn BlobStoreHandle>;
    let target = PageStoreTarget::new(primary_handle, secondary_pages_handle)
        .with_record_store(records_handle)
        .with_blob_store(blob_handle);
    let _report = recover_from_wal(wal_dir, Arc::clone(&mgr), target, None).unwrap();
    (writer, mgr, primary, secondary, store)
}

#[test]
fn round_trip_secondary_indexed_property_post_replay() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();

    let label = LabelId::new(11);
    let tenant = TenantId::DEFAULT;
    // node_properties() emits `(INLINE_U32A_PROPERTY_KEY, U32(rec.inline_u32a))`
    // and the U32B sibling as the two positional properties the
    // secondary index reverse-maps. Use a distinctive value so the
    // lookup is unambiguous.
    let indexed_value = 0x1234_5678u32;

    // Pre-crash: create a node whose inline_u32a hits the secondary.
    let (writer, mgr, primary, secondary, store) = build_stack_with_secondary(&wal_dir);
    let mut tx = mgr.begin(tenant);
    let id = create_node(
        &store,
        &mut tx,
        tenant,
        label,
        &PropertyData::InlineU32Pair(indexed_value, 0),
    )
    .unwrap();
    commit(tx, &store).unwrap();

    // Sanity: the pre-crash secondary lookup resolves immediately.
    let pre_key = SecondaryKey::new(
        tenant,
        label,
        INLINE_U32A_PROPERTY_KEY,
        PropertyValue::U32(indexed_value),
    );
    let pre_hits = secondary
        .lookup(pre_key)
        .expect("secondary lookup pre-crash");
    assert!(
        pre_hits.contains(&id),
        "pre-crash secondary lookup must contain node id {id:?}, got {pre_hits:?}",
    );

    writer.shutdown().unwrap();
    drop(store);
    drop(primary);
    drop(secondary);
    drop(mgr);

    // Post-crash: fresh stack + recovery that wires the secondary.
    let (writer2, mgr2, _primary2, secondary2, store2) = recover_stack_with_secondary(&wal_dir);

    // Read-side: node is readable via primary (proves record pages
    // reconstructed per PR #79 X-2).
    let tx2 = mgr2.begin(tenant);
    let rec = read_node_with_store(&store2, &tx2, id)
        .unwrap()
        .expect("node should be readable post-replay");
    assert_eq!(rec.label_id, label.raw());
    assert_eq!(rec.inline_u32a, indexed_value);

    // Reverse-index side: secondary lookup on (tenant, label,
    // INLINE_U32A_PROPERTY_KEY, U32(indexed_value)) resolves to the
    // node id. Proves the SecondaryIndex B-tree pages were
    // reconstructed from the v3 bundle's
    // `BundlePageKind::SecondaryIndex` entries.
    let post_key = SecondaryKey::new(
        tenant,
        label,
        INLINE_U32A_PROPERTY_KEY,
        PropertyValue::U32(indexed_value),
    );
    let post_hits = secondary2
        .lookup(post_key)
        .expect("secondary lookup post-crash must succeed");
    assert!(
        post_hits.contains(&id),
        "post-replay secondary lookup must contain node id {id:?}, got {post_hits:?}",
    );
    writer2.shutdown().unwrap();
}

// ─────────────────────────────────────────────────────────────────
// M3.a Phase 5.5 — vector workload extension
// ─────────────────────────────────────────────────────────────────
//
// Per Path A directive 2026-04-26 + Phase 5.5 spec §2.1: extend the
// existing CRUD round-trip with a vector workload. Verifies that
// adding a vector arena commit alongside the existing primary /
// record / blob commits does not disrupt the established round-trip
// flow, AND that the vector arena's snapshot+recovery primitive
// delivers the same byte-identical recovery the CRUD path enjoys.
//
// Production CRUD wiring of `VectorPageStoreHandle` into `CrudStore`
// lands in Slice G.5; until then the vector half exercises the same
// `recover_arena` chain that production WAL recovery will dispatch
// through. The test models the full pre-crash → snapshot → shutdown
// → post-crash → recover sequence at the recovery-API level.

/// M3.a Slice G.4 — production-path vector workload round-trip.
///
/// Replaces the prior hand-built `VecWalDelta` + `EmptyMvcc`
/// simulation with the production v5 commit-bundle path:
///
///   1. Stage vector arena pages via
///      [`CrudStore::stage_vector_page`] BEFORE the regular CRUD
///      `commit()` call. The CRUD commit drains the staged emits
///      into the v5 `CommitBundle`'s `vector_pages` section.
///   2. Crash + drop the in-memory state.
///   3. Wire a recording `VectorPageStoreHandle` into the fresh
///      stack's `PageStoreTarget`.
///   4. `recover_from_wal` decodes each v5 bundle, applies
///      `staged_pages` → `vector_pages` → `allocator_advances`
///      (Lemma I3) into the wired stores.
///   5. Assert every staged page lands in the recording handle
///      byte-identically AND in commit_lsn order (monotonic
///      ascending).
///
/// Pinned by issue #131 follow-up item 3 (production-path
/// simulation gap) + ADR-031 amendment-02 + ADR-035 §4.5/§4.6.
#[test]
fn wal_replay_vector_workload_round_trip() {
    use std::sync::Mutex as StdMutex;

    use arcgraph_core::{PageId, PartitionId};
    use arcgraph_storage::vector_store::{VectorPageStoreHandle, VectorStoreError};
    use arcgraph_storage::wal::{
        AllocatorSeedHandle, BlobStoreHandle, PageStoreTarget, PrimaryPageStoreHandle,
        RecordPageStoreHandle, WalWriter, recover_from_wal,
    };

    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();

    // ── Pre-crash: stage vector pages + run CRUD commit ──
    //
    // The prior version of this test fed a hand-built `VecWalDelta`
    // simulator into `recover_arena`. The simulator never exercised
    // the production v5 commit-bundle path: the bytes never landed
    // on disk inside a real `CommitBundle` record, so a regression
    // in `encode_commit_bundle_v5` / replay's `vector_pages` apply
    // leg would not have been caught here.
    //
    // Slice G.4 closes the gap by routing through
    // `CrudStore::stage_vector_page` → `commit()` →
    // `encode_commit_bundle_v5` → WAL fsync → `recover_from_wal` →
    // `VectorPageStoreHandle::install_or_replace`.
    let (writer, mgr, primary, store) = build_stack(&wal_dir);

    let tenant = TenantId::DEFAULT;

    // First commit: regular CRUD work + 2 vector pages.
    let mut tx = mgr.begin(tenant);
    let crud_id = create_node(
        &store,
        &mut tx,
        tenant,
        LabelId::new(73),
        &PropertyData::InlineU32Pair(2026, 426),
    )
    .unwrap();
    let txn1_id = tx.id();
    // Stage two vector arena pages on this txn. The bytes will ride
    // the same v5 `CommitBundle` fsync as the CRUD writes.
    let v_page1_bytes = mk_filled_page(0xA5);
    let v_page2_bytes = mk_filled_page(0x5A);
    store.stage_vector_page(
        txn1_id,
        tenant,
        PartitionId::ZERO,
        0, // index_id always 0 at v1.0
        PageId::new(1),
        v_page1_bytes.clone(),
    );
    store.stage_vector_page(
        txn1_id,
        tenant,
        PartitionId::ZERO,
        0,
        PageId::new(2),
        v_page2_bytes.clone(),
    );
    let commit1_lsn = commit(tx, &store).unwrap();

    // Second commit: 1 more vector page, no CRUD work. Pin that
    // a "vector-only" bundle is well-formed under the v5 codec
    // even when the primary write set is empty.
    //
    // Note: a no-op MVCC commit is a fast-path early-return in
    // `commit_with_bundle_writes` (see transaction.rs), so we
    // include a minimal CRUD write to drive the bundle path.
    let mut tx2 = mgr.begin(tenant);
    let crud_id2 = create_node(
        &store,
        &mut tx2,
        tenant,
        LabelId::new(74),
        &PropertyData::InlineU32Pair(2027, 428),
    )
    .unwrap();
    let txn2_id = tx2.id();
    let v_page3_bytes = mk_filled_page(0xC3);
    store.stage_vector_page(
        txn2_id,
        tenant,
        PartitionId::ZERO,
        0,
        PageId::new(3),
        v_page3_bytes.clone(),
    );
    let commit2_lsn = commit(tx2, &store).unwrap();
    assert!(
        commit2_lsn > commit1_lsn,
        "second commit_lsn ({commit2_lsn:?}) must exceed first ({commit1_lsn:?})"
    );

    // Shutdown.
    writer.shutdown().unwrap();
    drop(store);
    drop(primary);
    drop(mgr);

    // ── Post-crash: build fresh stack with a recording vector store
    //    handle, recover from WAL, assert byte-identity + apply
    //    order. ──
    let recorder = Arc::new(RecordingVectorStore::default());
    let recorder_handle: Arc<dyn VectorPageStoreHandle> =
        Arc::clone(&recorder) as Arc<dyn VectorPageStoreHandle>;

    let writer2 = WalWriter::spawn(test_wal_config(&wal_dir)).unwrap();
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
        .with_vector_store(recorder_handle)
        .with_allocator_seed(allocator_seed);
    let report = recover_from_wal(&wal_dir, Arc::clone(&mgr2), target, None).unwrap();

    // CRUD half MUST still round-trip — the v5 commit path keeps the
    // existing v4 prefix shape intact.
    let tx_read = mgr2.begin(tenant);
    let rec1 = read_node_with_store(&store2, &tx_read, crud_id)
        .unwrap()
        .expect("CRUD node from commit 1 must round-trip post-replay");
    assert_eq!(rec1.label_id, 73);
    assert_eq!(rec1.inline_u32a, 2026);
    assert_eq!(rec1.inline_u32b, 426);
    let rec2 = read_node_with_store(&store2, &tx_read, crud_id2)
        .unwrap()
        .expect("CRUD node from commit 2 must round-trip post-replay");
    assert_eq!(rec2.label_id, 74);

    // Vector half: every staged page MUST land in the recorder, in
    // commit_lsn ascending order.
    let calls = recorder.calls.lock().unwrap();
    assert_eq!(
        calls.len(),
        3,
        "expected exactly 3 vector_pages installs (2 from commit1 + 1 from commit2); \
         got {}",
        calls.len()
    );

    // Find each install by page_id; assert tenant + bytes byte-identity.
    for (expected_pid, expected_bytes) in [
        (1u64, &v_page1_bytes),
        (2u64, &v_page2_bytes),
        (3u64, &v_page3_bytes),
    ] {
        let found = calls
            .iter()
            .find(|(_t, pid, _b)| pid.raw() == expected_pid)
            .unwrap_or_else(|| panic!("vector page {expected_pid} not installed during replay"));
        assert_eq!(
            found.0, tenant,
            "vector page {expected_pid} routed to wrong tenant"
        );
        assert_eq!(
            found.2.as_slice(),
            expected_bytes.as_ref(),
            "vector page {expected_pid} bytes drifted (cross-replay corruption)"
        );
    }

    // Strengthened oracle: install order MUST be commit_lsn
    // monotonic. The WAL replay executor sorts by commit_lsn before
    // applying, so the first 2 calls (from commit1) precede the 3rd
    // (from commit2). Within commit1 the order is the encoder's
    // sort by (tenant, partition, index_id, page_id, commit_lsn) —
    // page 1 before page 2.
    assert_eq!(
        calls[0].1.raw(),
        1,
        "first install must be commit1's first page (page_id=1); got {:?}",
        calls[0].1
    );
    assert_eq!(
        calls[1].1.raw(),
        2,
        "second install must be commit1's second page (page_id=2); got {:?}",
        calls[1].1
    );
    assert_eq!(
        calls[2].1.raw(),
        3,
        "third install must be commit2's only page (page_id=3); got {:?}",
        calls[2].1
    );

    // Strengthened oracle: the recovery report MUST show bundles
    // applied (proves the test really went through the v5 codec
    // path, not a Vec-backed mock that bypassed the bundle).
    assert!(
        report.applied_commit_lsn >= commit2_lsn,
        "post-replay applied_commit_lsn ({:?}) must cover both commits ({:?})",
        report.applied_commit_lsn,
        commit2_lsn
    );

    writer2.shutdown().unwrap();

    // ── Local helpers ──
    fn mk_filled_page(fill: u8) -> Box<[u8; arcgraph_core::PAGE_SIZE]> {
        Box::new([fill; arcgraph_core::PAGE_SIZE])
    }

    /// Recording mock: stores every install_or_replace call so the
    /// test can assert byte-identity + apply order. Mirrors the
    /// `RecordingVectorStore` pattern used in `wal_bundle_v5.rs`.
    #[derive(Default)]
    struct RecordingVectorStore {
        calls: StdMutex<Vec<(TenantId, PageId, Vec<u8>)>>,
    }

    impl VectorPageStoreHandle for RecordingVectorStore {
        fn install_or_replace(
            &self,
            tenant: TenantId,
            page_id: PageId,
            bytes: &[u8],
        ) -> std::result::Result<(), VectorStoreError> {
            self.calls
                .lock()
                .unwrap()
                .push((tenant, page_id, bytes.to_vec()));
            Ok(())
        }
        fn restore_page_bytes(
            &self,
            _tenant: TenantId,
            _page_id: PageId,
            _bytes: &[u8],
        ) -> std::result::Result<(), VectorStoreError> {
            Ok(())
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// M3.a Slice G.5 — rollback + replay full-loop pin
// ─────────────────────────────────────────────────────────────────
//
// Pin the production loop end-to-end:
//
//   1. Build a stack with a real `VectorArenaPageStore` wired into
//      the CrudStore via `with_vector_store`.
//   2. Run TWO vector commits — the first commits successfully via
//      `capture_and_stage_vector_page`; the second is doomed (WAL
//      shutdown) and rolls back via the Slice G.5 dispatch arm.
//   3. Crash + recover from WAL into a fresh stack with a
//      recording vector handle.
//   4. Assert ONLY the first commit's vector page bytes appear in
//      the recovered arena (the doomed second commit MUST NOT
//      leave a ghost in the WAL — the bundle never fsynced — and
//      MUST be rolled back in-memory by the Z-1 (b) closure).
//
// Closes the Slice G.5 wirebench: capture + stage + commit + WAL
// fail + rollback + replay.

#[test]
fn wal_replay_after_z1_rollback_vector_workload() {
    use arcgraph_core::record::PAGE_SIZE;
    use arcgraph_core::{PageId, PartitionId};
    use arcgraph_storage::vector_store::recovery::VectorArenaPageStore;
    use arcgraph_storage::vector_store::{VectorPageStoreHandle, VectorStoreError};
    use arcgraph_storage::wal::{
        AllocatorSeedHandle, BlobStoreHandle, PageStoreTarget, PrimaryPageStoreHandle,
        RecordPageStoreHandle, WalWriter, recover_from_wal,
    };

    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();

    // ── Build stack with vector store wired ──
    let writer = WalWriter::spawn(test_wal_config(&wal_dir)).unwrap();
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
        CrudStore::new_with_index(
            Some(handle.clone()),
            Arc::clone(&primary),
            Arc::clone(&alloc),
        )
        .with_vector_store(arena_handle),
    );

    let tenant = TenantId::DEFAULT;
    let success_page = PageId::new(401);
    let doomed_page = PageId::new(402);
    let success_pre_w = [0xA1u8; PAGE_SIZE];
    let success_post_w: Box<[u8; PAGE_SIZE]> = Box::new([0xB1u8; PAGE_SIZE]);
    let doomed_pre_w = [0xA2u8; PAGE_SIZE];
    let doomed_post_w: Box<[u8; PAGE_SIZE]> = Box::new([0xB2u8; PAGE_SIZE]);

    arena
        .install_or_replace(tenant, success_page, &success_pre_w)
        .expect("install pre-W (success)");
    arena
        .install_or_replace(tenant, doomed_page, &doomed_pre_w)
        .expect("install pre-W (doomed)");

    // ── Commit 1: succeeds. Vector page bytes ride the v5 bundle. ──
    let mut tx1 = mgr.begin(tenant);
    let _seed1 = create_node(
        &store,
        &mut tx1,
        tenant,
        LabelId::new(1),
        &PropertyData::Empty,
    )
    .unwrap();
    let txn1_id = tx1.id();
    {
        let log = tx1.mutation_log_mut();
        store.capture_and_stage_vector_page(
            log,
            txn1_id,
            tenant,
            PartitionId::ZERO,
            0,
            success_page,
            &success_pre_w,
            success_post_w.clone(),
        );
    }
    arena
        .install_or_replace(tenant, success_page, success_post_w.as_ref())
        .expect("post-W (success)");
    let _commit_lsn = commit(tx1, &store).expect("commit 1 must succeed");

    // ── Commit 2: doomed. WAL is killed; rollback runs. ──
    writer
        .shutdown()
        .expect("shutdown writer (force WAL fsync failure)");
    let mut tx2 = mgr.begin(tenant);
    let _seed2 = create_node(
        &store,
        &mut tx2,
        tenant,
        LabelId::new(2),
        &PropertyData::Empty,
    )
    .unwrap();
    let txn2_id = tx2.id();
    {
        let log = tx2.mutation_log_mut();
        store.capture_and_stage_vector_page(
            log,
            txn2_id,
            tenant,
            PartitionId::ZERO,
            0,
            doomed_page,
            &doomed_pre_w,
            doomed_post_w.clone(),
        );
    }
    arena
        .install_or_replace(tenant, doomed_page, doomed_post_w.as_ref())
        .expect("post-W (doomed)");
    let err = commit(tx2, &store).expect_err("commit 2 must fail");
    match err {
        arcgraph_storage::crud::CrudError::Mvcc(
            arcgraph_core::ArcGraphError::WalErrorRolledBack { .. },
        ) => {}
        other => panic!("expected WalErrorRolledBack on doomed commit, got {other:?}"),
    }

    // After rollback the doomed page in the arena is back to its
    // pre-W bytes; the success page still holds its post-W bytes
    // (the success commit's mutation is durable in-memory and
    // persisted via the v5 bundle).
    assert_eq!(
        arena.get_page(tenant, success_page).unwrap().as_slice(),
        success_post_w.as_ref(),
        "success commit's post-W bytes survive rollback of unrelated txn"
    );
    assert_eq!(
        arena.get_page(tenant, doomed_page).unwrap().as_slice(),
        &doomed_pre_w,
        "doomed commit's vector page rolled back to pre-W bytes"
    );

    // ── Crash + recover into a fresh stack with a recording handle. ──
    drop(store);
    drop(primary);
    drop(mgr);
    drop(arena);

    let recorder = Arc::new(RecordingVectorStore::default());
    let recorder_handle: Arc<dyn VectorPageStoreHandle> =
        Arc::clone(&recorder) as Arc<dyn VectorPageStoreHandle>;
    let writer2 = WalWriter::spawn(test_wal_config(&wal_dir)).unwrap();
    let handle2 = writer2.handle();
    let mgr2 = Arc::new(TxnManager::with_wal(handle2.clone()));
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
        .with_vector_store(recorder_handle)
        .with_allocator_seed(allocator_seed);
    let _report = recover_from_wal(&wal_dir, Arc::clone(&mgr2), target, None).unwrap();

    // Assert the recorder saw the SUCCESS commit's vector page —
    // and ONLY that one. The doomed commit never reached the WAL
    // (its bundle never fsynced), so replay sees no ghost.
    let calls = recorder.calls.lock().unwrap();
    let success_hits: Vec<_> = calls
        .iter()
        .filter(|(_, pid, _)| *pid == success_page)
        .collect();
    let doomed_hits: Vec<_> = calls
        .iter()
        .filter(|(_, pid, _)| *pid == doomed_page)
        .collect();
    assert_eq!(
        success_hits.len(),
        1,
        "success commit's vector page must replay exactly once; got {}",
        success_hits.len()
    );
    assert_eq!(
        success_hits[0].2.as_slice(),
        success_post_w.as_ref(),
        "success commit's post-W bytes round-trip byte-identically through replay"
    );
    assert!(
        doomed_hits.is_empty(),
        "doomed commit's vector page must NOT appear in WAL replay (rollback prevented WAL fsync); \
         got {} entries",
        doomed_hits.len()
    );

    writer2.shutdown().unwrap();

    // ── Local helpers ──
    /// Recording mock — stores every install_or_replace call so we
    /// can assert byte-identity + presence/absence of specific page
    /// IDs.
    #[derive(Default)]
    struct RecordingVectorStore {
        calls: std::sync::Mutex<Vec<(TenantId, PageId, Vec<u8>)>>,
    }

    impl VectorPageStoreHandle for RecordingVectorStore {
        fn install_or_replace(
            &self,
            tenant: TenantId,
            page_id: PageId,
            bytes: &[u8],
        ) -> std::result::Result<(), VectorStoreError> {
            self.calls
                .lock()
                .unwrap()
                .push((tenant, page_id, bytes.to_vec()));
            Ok(())
        }
        fn restore_page_bytes(
            &self,
            _tenant: TenantId,
            _page_id: PageId,
            _bytes: &[u8],
        ) -> std::result::Result<(), VectorStoreError> {
            Ok(())
        }
    }
}
