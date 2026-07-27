//! #352 Part 2 (ADR-199) — storage-level pins that WAL replay
//! reconstructs the [`IdempotencyStore`] `external_id → internal_id`
//! map from the v6+ `CommitBundle` `idempotency_bindings` section, AND
//! that the binding is **atomic with its commit** (the crash-before-
//! commit "both-or-neither" property).
//!
//! This isolates the replay arm added by #352 Part 2: `crud::commit`
//! folds each staged binding into the bundle, and `recover_from_wal`
//! — when wired with [`PageStoreTarget::with_idempotency_store`] —
//! installs them into the served store via [`IdempotencyStore::install`].
//! The end-to-end `graph.ingest` restart oracle lives in
//! `crates/arcgraph-cli/tests/durable_idempotency_restart_352.rs`; this
//! file pins the storage-crate seam in isolation (no MCP / crud stack).
//!
//! # Why the fold is atomicity-correct (the crash-before-commit test)
//!
//! Because the binding rides INSIDE the same CRC-protected `CommitBundle`
//! WAL record as the commit's MVCC writes, there is no window in which
//! the binding is durable but the commit is not (the failure mode of a
//! standalone pre-commit record — see ADR-199 §Revision 2026-06-07). A
//! crash that tears the bundle drops the WHOLE record: NEITHER the MVCC
//! write NOR the binding is recovered. `torn_bundle_drops_binding_and_commit_atomically`
//! proves exactly this with a truncated-tail fault injection.

use std::collections::HashMap;
use std::sync::Arc;

use arcgraph_core::{Lsn, TenantId};
use arcgraph_storage::IdempotencyStore;
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::PrimaryIndex;
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{
    IdempotencyBindingEntry, IdempotencyBindingOp, PageStoreTarget, PrimaryPageStoreHandle,
    WalConfig, WalRecordType, WalWriter, encode_commit_bundle_v8, list_segments, recover_from_wal,
    segment_filename,
};
use bytes::Bytes;
use tempfile::TempDir;

const KIND_NODE: u8 = 0;

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

/// Encode a minimal current-format `CommitBundle` carrying one MVCC write (the
/// "node") AND one idempotency binding for the SAME commit, then append
/// it to the WAL at `wal_dir` as a single `CommitBundle` record.
fn write_bundle_with_node_and_binding(
    wal_dir: &std::path::Path,
    tenant: TenantId,
    commit_lsn: Lsn,
    mvcc_key: u64,
    external_id: &str,
    internal_id: u64,
) {
    let mut mvcc: HashMap<u64, Option<Bytes>> = HashMap::new();
    mvcc.insert(mvcc_key, Some(Bytes::from_static(b"node-record-payload")));
    let bindings = vec![IdempotencyBindingEntry {
        op: IdempotencyBindingOp::Install,
        tenant,
        kind: KIND_NODE,
        internal_id,
        external_id: external_id.to_owned(),
    }];
    let payload = encode_commit_bundle_v8(
        commit_lsn,
        tenant,
        &mvcc,
        &[], // sidechannel
        &[], // staged_pages
        &[], // allocator_advances
        &[], // vector_pages
        &bindings,
        &[], // #1221: no acl_grants in this idempotency fixture
    );
    let writer = WalWriter::spawn(test_wal_config(wal_dir)).unwrap();
    let handle = writer.handle();
    handle
        .append(WalRecordType::CommitBundle, 0, 0, tenant, payload)
        .expect("append CommitBundle");
    writer.shutdown().unwrap();
}

fn write_v7_bundle_with_bindings(
    wal_dir: &std::path::Path,
    tenant: TenantId,
    commit_lsn: Lsn,
    bindings: &[IdempotencyBindingEntry],
) {
    let payload = encode_commit_bundle_v8(
        commit_lsn,
        tenant,
        &HashMap::new(),
        &[],
        &[],
        &[],
        &[],
        bindings,
        &[], // #1221: no acl_grants in this idempotency fixture
    );
    let writer = WalWriter::spawn(test_wal_config(wal_dir)).unwrap();
    let handle = writer.handle();
    handle
        .append(WalRecordType::CommitBundle, 0, 0, tenant, payload)
        .expect("append CommitBundle");
    writer.shutdown().unwrap();
}

/// Recover the WAL at `wal_dir` into a throwaway primary store, wiring
/// `store` as the idempotency store iff `Some`. Returns the recovery
/// report so callers can read `applied_commit_lsn` +
/// `idempotency_bindings_recovered`.
fn recover_into(
    wal_dir: &std::path::Path,
    store: Option<Arc<IdempotencyStore>>,
) -> arcgraph_storage::wal::RecoveryReport {
    let mgr = Arc::new(TxnManager::new());
    let alloc = Arc::new(PageAllocator::new());
    let primary =
        Arc::new(PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&alloc), None).expect("primary"));
    let primary_handle: Arc<dyn PrimaryPageStoreHandle> =
        Arc::clone(primary.page_store()) as Arc<dyn PrimaryPageStoreHandle>;
    let mut target = PageStoreTarget::primary_only(primary_handle);
    if let Some(s) = store {
        target = target.with_idempotency_store(s);
    }
    recover_from_wal(wal_dir, mgr, target, None).expect("recover_from_wal")
}

// ─────────────────────────────────────────────────────────────────────
// Clean recovery — the binding (and its commit) are recovered together.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn idempotency_binding_recovered_from_bundle_replay() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();
    let tenant = TenantId::DEFAULT;

    write_bundle_with_node_and_binding(&wal_dir, tenant, Lsn::new(5), 1, "alice", 42);

    let store = Arc::new(IdempotencyStore::new());
    let report = recover_into(&wal_dir, Some(Arc::clone(&store)));

    // The binding resolves to its original internal id post-replay.
    let binding = store
        .get(tenant, KIND_NODE, "alice")
        .expect("binding MUST recover");
    assert_eq!(
        binding.internal_id, 42,
        "binding MUST recover to its original internal id",
    );
    assert_eq!(
        binding.payload_hash, None,
        "v6 replay has no payload hash; cross-restart conflict detection is approach B",
    );
    assert_eq!(
        report.metrics.idempotency_bindings_recovered, 1,
        "exactly one idempotency binding replayed",
    );
    // The owning commit was applied (its commit_lsn is the high-water).
    assert_eq!(
        report.applied_commit_lsn,
        Lsn::new(5),
        "the bundle carrying the binding was applied",
    );
}

#[test]
fn idempotency_release_replayed_from_v7_bundle() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();
    let tenant = TenantId::DEFAULT;
    let bindings = vec![
        IdempotencyBindingEntry {
            op: IdempotencyBindingOp::Install,
            tenant,
            kind: KIND_NODE,
            internal_id: 99,
            external_id: "x".to_owned(),
        },
        IdempotencyBindingEntry {
            op: IdempotencyBindingOp::Release,
            tenant,
            kind: KIND_NODE,
            internal_id: 0,
            external_id: "x".to_owned(),
        },
    ];
    write_v7_bundle_with_bindings(&wal_dir, tenant, Lsn::new(6), &bindings);

    let store = Arc::new(IdempotencyStore::new());
    let report = recover_into(&wal_dir, Some(Arc::clone(&store)));

    assert_eq!(store.get(tenant, KIND_NODE, "x"), None);
    assert_eq!(store.total_len(), 0);
    assert_eq!(report.metrics.idempotency_bindings_recovered, 2);
}

// ─────────────────────────────────────────────────────────────────────
// Crash-before-commit — a TORN bundle drops BOTH the commit AND the
// binding (no torn state). This is the property the v6 fold guarantees
// that a standalone pre-commit record could NOT.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn torn_bundle_drops_binding_and_commit_atomically() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();
    let tenant = TenantId::DEFAULT;

    // Write the v6 bundle (node MVCC write + binding), then simulate a
    // crash mid-fsync by truncating the segment tail so the bundle record
    // is torn (its CRC/payload tail is incomplete).
    write_bundle_with_node_and_binding(&wal_dir, tenant, Lsn::new(5), 1, "bob", 99);

    let segs = list_segments(&wal_dir).unwrap();
    let last = *segs.last().unwrap();
    let path = wal_dir.join(segment_filename(last));
    let len = std::fs::metadata(&path).unwrap().len();
    // Knock 8 bytes off the tail — enough to truncate the single bundle
    // record so it decodes as a torn tail (dropped on recovery).
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(len.saturating_sub(8))
        .unwrap();

    let store = Arc::new(IdempotencyStore::new());
    let report = recover_into(&wal_dir, Some(Arc::clone(&store)));

    // NEITHER the binding NOR the commit survived the torn write —
    // both-or-neither, no torn state.
    assert!(
        store.is_empty(),
        "torn bundle MUST NOT install the binding (got {} entries)",
        store.total_len(),
    );
    assert_eq!(
        report.metrics.idempotency_bindings_recovered, 0,
        "no binding recovered from a torn bundle",
    );
    // The torn bundle's commit (commit_lsn = 5) was NOT applied: the
    // post-replay high-water stayed BELOW 5 (a fresh TxnManager baselines
    // above ZERO, so the oracle is "did not reach the bundle's lsn", not
    // "== 0"). Binding + commit share fate — both absent.
    assert!(
        report.applied_commit_lsn.raw() < 5,
        "the torn bundle's commit (lsn 5) MUST NOT be applied; got applied_commit_lsn={:?}",
        report.applied_commit_lsn,
    );
    // The torn tail is a clean, expected condition (not a hard error).
    assert!(
        report.torn_tail.is_some(),
        "the truncated bundle is reported as a torn tail, not a corruption halt",
    );
}

// ─────────────────────────────────────────────────────────────────────
// No store wired — the apply arm is a no-op (pre-fix posture preserved).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn idempotency_replay_without_store_is_noop() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();
    let tenant = TenantId::DEFAULT;

    write_bundle_with_node_and_binding(&wal_dir, tenant, Lsn::new(5), 1, "carol", 7);

    // Recover WITHOUT wiring an idempotency store: the bundle still
    // applies (commit recovered) but the binding apply arm is skipped.
    let report = recover_into(&wal_dir, None);
    assert_eq!(
        report.metrics.idempotency_bindings_recovered, 0,
        "no store wired ⇒ idempotency apply arm is a no-op",
    );
    assert_eq!(
        report.applied_commit_lsn,
        Lsn::new(5),
        "the commit itself still recovers without an idempotency store",
    );
}
