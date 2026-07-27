//! K-1c+d integration-test shared helpers.
//!
//! Each `tests/k1c_encoding_mismatch_*_replay.rs` integration test
//! pulls this in via `mod k1c_common;`. The helpers consolidate the
//! K-1 stack construction + clean recovery + state-snapshot
//! boilerplate so each test file focuses on the encoding-mismatch
//! injection + oracle assertion (Phase 4.3 reverse-test loop).
//!
//! Per Cargo's integration-test idiom (every `tests/foo.rs` is its
//! own crate root), shared modules live under `tests/k1c_common/mod.rs`
//! and each test file does `mod k1c_common;` to compile in the
//! helpers locally. The duplication is per-binary linker-cost only;
//! the source is single-sourced here.
//!
//! ## What this module provides
//!
//! - [`K1cStack`] — RAII WAL stack mirroring `tests/k1_smoke_30s.rs`'s
//!   `K1Stack`, but with a `recover()` helper that runs the M4-41
//!   cold-start MVCC stats rebuild post-recovery (so the oracle's
//!   strict-mode comparison is non-vacuous).
//! - [`run_workload`] — driver that commits N tenants × M nodes per
//!   tenant under T1 strict durability, with deterministic bytes per
//!   `(tenant, node)` so the K-1 oracle's `latest_t1` + `any_history`
//!   maps are easy to construct from the workload spec.
//! - [`build_pre_crash_state`] — `CommittedState` builder from the
//!   workload spec.
//! - [`build_recovered_state`] — `RecoveredState` builder from a
//!   recovered stack.
//! - [`pick_target_key`] — picks a deterministic `(TenantId, NodeId)`
//!   from the workload spec for tampering.
//! - [`older_t1_bytes_for`] — for I-3 chain-layout drift, returns a
//!   second-historical T1 bytes triple distinct from the latest at
//!   the target. The workload guarantees ≥ 2 historical T1 commits
//!   per target key.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use arcgraph_core::{LabelId, NodeId, TenantId};
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::catalog::SystemCatalog;
use arcgraph_storage::crud::{
    CrudStore, PropertyData, commit, create_node, crud_allocator_seed_handle, read_node_with_store,
    update_node,
};
use arcgraph_storage::io::InMemoryPageIo;
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::PrimaryIndex;
use arcgraph_storage::test_harness::k1::oracle::{
    CommittedBytes, CommittedState, CommittedStatsRebuild, RecoveredState, snapshot_catalog_stats,
};
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{
    AllocatorSeedHandle, BackgroundFsyncFailAction, BackgroundFsyncScheduler, BlobStoreHandle,
    PageStoreTarget, PrimaryPageStoreHandle, RecordPageStoreHandle, WalConfig, WalWriter,
    recover_from_wal,
};
use tempfile::TempDir;

// ─────────────────────────────────────────────────────────────────
// WAL stack construction (mirrors tests/k1_smoke_30s.rs)
// ─────────────────────────────────────────────────────────────────

fn test_wal_config(dir: PathBuf) -> WalConfig {
    WalConfig {
        dir,
        segment_size_bytes: 64 * 1024 * 1024,
        group_commit_window: Duration::from_millis(2),
        group_commit_max_batch: 32,
        metrics_sink: None,
        encryption: None,
        inflight_budget_bytes: None,
    }
}

pub struct K1cStack {
    pub writer: Option<WalWriter>,
    pub scheduler: Option<Arc<BackgroundFsyncScheduler>>,
    pub mgr: Arc<TxnManager>,
    pub primary: Arc<PrimaryIndex>,
    pub store: Arc<CrudStore>,
    pub catalog: Arc<SystemCatalog>,
}

impl K1cStack {
    pub fn build(dir: &Path) -> Self {
        let writer = WalWriter::spawn(test_wal_config(dir.to_path_buf())).unwrap();
        let scheduler = BackgroundFsyncScheduler::start(
            writer.handle(),
            BackgroundFsyncFailAction::RollbackAndContinue,
        );
        let handle = writer.handle();
        let mut mgr_inner = TxnManager::with_wal(handle.clone());
        let catalog = Arc::new(SystemCatalog::new());
        let io = Arc::new(InMemoryPageIo::new());
        let pool = BufferPool::new(8, io);
        catalog.bootstrap(&pool, &mgr_inner).unwrap();
        mgr_inner.set_durability_lookup(catalog.clone());
        let mgr = Arc::new(mgr_inner);

        let alloc = Arc::new(PageAllocator::new());
        let primary = Arc::new(
            PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&alloc), Some(handle.clone())).unwrap(),
        );
        let store = Arc::new(CrudStore::new_with_index(
            Some(handle.clone()),
            Arc::clone(&primary),
            Arc::clone(&alloc),
        ));
        Self {
            writer: Some(writer),
            scheduler: Some(scheduler),
            mgr,
            primary,
            store,
            catalog,
        }
    }

    pub fn recover(dir: &Path) -> Self {
        let writer = WalWriter::spawn(test_wal_config(dir.to_path_buf())).unwrap();
        let scheduler = BackgroundFsyncScheduler::start(
            writer.handle(),
            BackgroundFsyncFailAction::RollbackAndContinue,
        );
        let handle = writer.handle();
        let mut mgr_inner = TxnManager::with_wal(handle.clone());
        let catalog = Arc::new(SystemCatalog::new());
        let io = Arc::new(InMemoryPageIo::new());
        let pool = BufferPool::new(8, io);
        catalog.bootstrap(&pool, &mgr_inner).unwrap();
        mgr_inner.set_durability_lookup(catalog.clone());
        let mgr = Arc::new(mgr_inner);

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
        let records_handle: Arc<dyn RecordPageStoreHandle> =
            Arc::clone(store.records().expect("records")) as Arc<dyn RecordPageStoreHandle>;
        let blob_handle: Arc<dyn BlobStoreHandle> =
            Arc::clone(store.blob_store()) as Arc<dyn BlobStoreHandle>;
        let allocator_seed: Arc<dyn AllocatorSeedHandle> =
            crud_allocator_seed_handle(Arc::clone(&store), Arc::clone(&alloc));
        let target = PageStoreTarget::primary_only(primary_handle)
            .with_record_store(records_handle)
            .with_blob_store(blob_handle)
            .with_allocator_seed(allocator_seed);
        let report = recover_from_wal(dir, Arc::clone(&mgr), target, None).unwrap();

        // M4-41 cold-start MVCC stats rebuild (per ADR-038
        // amendment-06 §D-25.1) — synchronous at recovery time so the
        // strict-mode oracle is non-vacuous.
        let _rebuild = arcgraph_storage::recovery::rebuild_all_tenant_stats(
            report.applied_commit_lsn,
            &mgr,
            &store,
        );

        Self {
            writer: Some(writer),
            scheduler: Some(scheduler),
            mgr,
            primary,
            store,
            catalog,
        }
    }

    pub fn shutdown(mut self) {
        if let Some(s) = self.scheduler.take() {
            let _ = s.shutdown();
        }
        if let Some(w) = self.writer.take() {
            let _ = w.shutdown();
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// Workload spec — N tenants × M overwrites
// ─────────────────────────────────────────────────────────────────

/// One commit op in the deterministic workload. Every commit is T1
/// strict (DEFAULT tier under v1.0 catalog defaults). The same
/// `(tenant, node)` may be overwritten multiple times so the
/// `any_history` set has ≥ 2 entries per key (load-bearing for I-3
/// MVCC chain layout drift testing).
#[derive(Debug, Clone, Copy)]
pub struct WorkloadCommit {
    pub tenant: TenantId,
    pub label: u32,
    pub a: u32,
    pub b: u32,
    /// Whether this commit creates a NEW node (allocates fresh
    /// NodeId) or overwrites an existing node. The first commit per
    /// `(tenant, label)` group is `Create`; subsequent commits are
    /// `Overwrite { node_idx }` (overwrites the Nth node within the
    /// group).
    pub op: WorkloadOp,
}

#[derive(Debug, Clone, Copy)]
pub enum WorkloadOp {
    /// Create a new node with the given inline u32 pair.
    Create,
    /// Overwrite node-index `idx` within the same `(tenant, label)`
    /// group. Bytes change to (a, b).
    Overwrite { idx: usize },
}

/// Plan a deterministic 2-tenant × 3-node × 2-overwrites workload.
/// Returns 6 `WorkloadCommit` entries: 3 creates + 3 overwrites,
/// per tenant. Total 12 commits.
pub fn plan_workload(tenant_a: TenantId, tenant_b: TenantId) -> Vec<WorkloadCommit> {
    let mut plan = Vec::with_capacity(12);
    for (tenant_idx, tenant) in [tenant_a, tenant_b].into_iter().enumerate() {
        let label_base = (tenant_idx as u32 + 1) * 100_000;
        for node_idx in 0..3 {
            let label = label_base + node_idx as u32;
            // Create
            plan.push(WorkloadCommit {
                tenant,
                label,
                a: 1_000 + node_idx as u32,
                b: 2_000 + node_idx as u32,
                op: WorkloadOp::Create,
            });
            // Overwrite (later T1 supersedes earlier T1 — gives I-3
            // chain-layout drift its earlier-vs-latest pair)
            plan.push(WorkloadCommit {
                tenant,
                label,
                a: 9_000 + node_idx as u32,
                b: 8_000 + node_idx as u32,
                op: WorkloadOp::Overwrite { idx: node_idx },
            });
        }
    }
    plan
}

/// Run the workload against the stack. Returns the per-`(tenant,
/// label)` allocated `NodeId`s so callers can pre-compute the
/// expected `(tenant, NodeId)` keys.
pub fn run_workload(stack: &K1cStack, plan: &[WorkloadCommit]) -> HashMap<(TenantId, u32), NodeId> {
    let mut allocated: HashMap<(TenantId, u32), NodeId> = HashMap::new();
    for cmt in plan {
        let mut tx = stack.mgr.begin(cmt.tenant);
        match cmt.op {
            WorkloadOp::Create => {
                let id = create_node(
                    &stack.store,
                    &mut tx,
                    cmt.tenant,
                    LabelId::new(cmt.label),
                    &PropertyData::InlineU32Pair(cmt.a, cmt.b),
                )
                .expect("create_node");
                commit(tx, &stack.store).expect("commit create");
                allocated.insert((cmt.tenant, cmt.label), id);
            }
            WorkloadOp::Overwrite { idx: _ } => {
                let id = *allocated
                    .get(&(cmt.tenant, cmt.label))
                    .expect("overwrite must follow a create with same (tenant,label)");
                update_node(
                    &stack.store,
                    &mut tx,
                    id,
                    &PropertyData::InlineU32Pair(cmt.a, cmt.b),
                )
                .expect("update_node");
                commit(tx, &stack.store).expect("commit overwrite");
            }
        }
    }
    allocated
}

// ─────────────────────────────────────────────────────────────────
// Pre-crash + recovered state builders
// ─────────────────────────────────────────────────────────────────

/// Build the K-1 oracle's `CommittedState` from the workload plan +
/// allocated NodeIds. Mirrors `tests/k1_smoke_30s.rs::build_committed_state`
/// for the rebuild semantics (commits_observed=1 per tenant per
/// recovery cycle per ADR-038 amendment-06 §D-25.1 step 2).
pub fn build_pre_crash_state(
    plan: &[WorkloadCommit],
    allocated: &HashMap<(TenantId, u32), NodeId>,
) -> CommittedState {
    let mut s = CommittedState::default();
    let mut stats: HashMap<TenantId, CommittedStatsRebuild> = HashMap::new();
    for cmt in plan {
        let id = *allocated
            .get(&(cmt.tenant, cmt.label))
            .expect("workload allocated NodeId for every commit");
        let key = (cmt.tenant, id);
        let bytes: CommittedBytes = (cmt.label, cmt.a, cmt.b);
        s.any_history.entry(key).or_default().insert(bytes);
        // Every commit is T1 strict under v1.0 catalog defaults, so
        // the LATEST T1 at this key is the LAST commit recorded for
        // it. Iteration order is the plan's order, so the latest
        // entry overwrites earlier ones.
        s.latest_t1.insert(key, bytes);
        if matches!(cmt.op, WorkloadOp::Create) {
            let st = stats.entry(cmt.tenant).or_default();
            *st.label_counts.entry(LabelId::new(cmt.label)).or_insert(0) += 1;
            st.total_nodes += 1;
        }
    }
    // Coalesced commits_observed per amendment-06 §D-25.1 step 2.
    for st in stats.values_mut() {
        st.commits_observed = 1;
    }
    s.total_commits = plan.len() as u64;
    s.stats_by_tenant = stats;
    s
}

/// Build the K-1 oracle's `RecoveredState` from the recovered stack,
/// reading every `(tenant, NodeId)` key from the pre-crash state via
/// `read_node_with_store` + per-tenant `catalog_stats`.
pub fn build_recovered_state(
    stack: &K1cStack,
    pre: &CommittedState,
    labels: &[LabelId],
) -> RecoveredState {
    let mut rec = RecoveredState::default();
    let mut tenants_seen: HashSet<TenantId> = HashSet::new();
    for (tenant, id) in pre.any_history.keys() {
        tenants_seen.insert(*tenant);
        let tx = stack.mgr.begin(*tenant);
        match read_node_with_store(&stack.store, &tx, *id) {
            Ok(Some(rec_node)) => {
                rec.bytes_by_key.insert(
                    (*tenant, *id),
                    (
                        rec_node.label_id,
                        rec_node.inline_u32a,
                        rec_node.inline_u32b,
                    ),
                );
            }
            Ok(None) => {}
            Err(e) => panic!("k1c read_node_with_store error: {e:?}"),
        }
    }
    for tenant in tenants_seen {
        if let Some(stats) = stack.store.catalog_stats(tenant) {
            let snap = snapshot_catalog_stats(&stats, labels, &[]);
            rec.stats_by_tenant.insert(tenant, snap);
        }
    }
    rec
}

/// Build a per-tenant pre-crash + recovered map split by tenant. Used
/// by I-5 cross-tenant tests that drive `verify_cross_tenant_invariants`.
pub fn split_by_tenant(
    pre: CommittedState,
    rec: RecoveredState,
) -> (
    HashMap<TenantId, CommittedState>,
    HashMap<TenantId, RecoveredState>,
) {
    let mut pre_by: HashMap<TenantId, CommittedState> = HashMap::new();
    let mut rec_by: HashMap<TenantId, RecoveredState> = HashMap::new();
    for ((tenant, node), history) in pre.any_history.iter() {
        let s = pre_by.entry(*tenant).or_default();
        s.any_history.insert((*tenant, *node), history.clone());
        if let Some(latest) = pre.latest_t1.get(&(*tenant, *node)) {
            s.latest_t1.insert((*tenant, *node), *latest);
        }
        s.total_commits += 1;
    }
    for (tenant, stats) in pre.stats_by_tenant.iter() {
        let s = pre_by.entry(*tenant).or_default();
        s.stats_by_tenant.insert(*tenant, stats.clone());
    }
    for ((tenant, node), bytes) in rec.bytes_by_key.iter() {
        let r = rec_by.entry(*tenant).or_default();
        r.bytes_by_key.insert((*tenant, *node), *bytes);
    }
    for (tenant, stats) in rec.stats_by_tenant.iter() {
        let r = rec_by.entry(*tenant).or_default();
        r.stats_by_tenant.insert(*tenant, stats.clone());
    }
    (pre_by, rec_by)
}

/// Pick a deterministic target key from the pre-crash state — the
/// `(tenant, NodeId)` whose raw `(tenant_raw, node_id_raw)` pair sorts
/// first ascendingly. Stable across runs; load-bearing for tests
/// whose reverse-test cycle requires the SAME target each run.
pub fn pick_target_key(pre: &CommittedState) -> (TenantId, NodeId) {
    let mut keys: Vec<(TenantId, NodeId)> = pre.any_history.keys().copied().collect();
    keys.sort_by_key(|(t, n)| (t.raw(), n.raw()));
    keys.into_iter()
        .next()
        .expect("pick_target_key: pre-crash state is empty")
}

/// Return the FIRST historical T1 bytes triple at `target` that is
/// distinct from the latest T1. Used by I-3 chain-layout drift tests.
pub fn older_t1_bytes_for(pre: &CommittedState, target: (TenantId, NodeId)) -> CommittedBytes {
    let latest = *pre
        .latest_t1
        .get(&target)
        .expect("older_t1_bytes_for: target has no latest T1");
    let history = pre
        .any_history
        .get(&target)
        .expect("older_t1_bytes_for: target has no any_history");
    let mut older: Vec<CommittedBytes> = history.iter().copied().filter(|b| *b != latest).collect();
    older.sort();
    *older
        .first()
        .expect("older_t1_bytes_for: target has no historical T1 distinct from latest — workload must overwrite ≥1×")
}

/// Build a `TempDir` + `wal/` subdir for a K-1c test.
pub fn fresh_workdir() -> (TempDir, PathBuf) {
    let workspace = TempDir::new().unwrap();
    let wal_dir = workspace.path().join("wal");
    std::fs::create_dir_all(&wal_dir).unwrap();
    (workspace, wal_dir)
}

/// Sorted, deduplicated label list across the workload — used by
/// `build_recovered_state` for the per-tenant `snapshot_catalog_stats`
/// projection.
pub fn workload_labels(plan: &[WorkloadCommit]) -> Vec<LabelId> {
    let mut labels: HashSet<LabelId> = HashSet::new();
    for cmt in plan {
        labels.insert(LabelId::new(cmt.label));
    }
    let mut out: Vec<LabelId> = labels.into_iter().collect();
    out.sort_by_key(|l| l.raw());
    out
}

/// Sanity-check the workload landed cleanly with no encoding-mismatch
/// — every test runs this first to guarantee that any subsequent
/// oracle violation comes from the injected corruption, not a
/// pre-existing recovery bug.
pub fn assert_clean_recovery(pre: &CommittedState, rec: &RecoveredState) {
    use arcgraph_storage::test_harness::k1::oracle::{
        OracleConfig, verify_post_recovery_invariants,
    };
    let report = verify_post_recovery_invariants(pre, rec, &OracleConfig::default()).expect(
        "k1c clean-recovery sanity: oracle must pass on the un-tampered post-recovery state",
    );
    assert!(report.unique_keys > 0);
    assert_eq!(report.t1_keys, report.unique_keys);
    assert_eq!(report.t1_satisfied, report.t1_keys);
}
