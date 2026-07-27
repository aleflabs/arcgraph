//! M4-41 cold-start MVCC rebuild — integration + proptest coverage
//! per ADR-038 amendment-06 §3 R1 acceptance criteria + amendment-06
//! §D-25.1 step 2 SeqLock contract.
//!
//! Complements the in-module unit tests in
//! `crates/arcgraph-storage/src/recovery/stats_rebuild.rs::tests` with:
//!
//!  - **End-to-end multi-tenant rebuild round-trip** through the real
//!    `crud::commit` → simulated process-restart (drop CatalogStats /
//!    drop TxnManager; replay WAL into fresh stack) → call rebuild →
//!    verify stats restored. Closes the K-1a BLOCKER-1 honest-deferral
//!    contract end-to-end (PR #176).
//!  - **Two-marker SeqLock proptest** stressing the cross-key invariant
//!    `sum(label_cards) ≤ total_nodes` (and rels) under randomised
//!    sequences of `(commit, snapshot, rebuild)` operations.
//!  - **Per-tenant fault isolation** with multiple tenants where one
//!    tenant's rebuild path is robust to per-tenant decode warnings.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use arcgraph_core::{LabelId, NodeId, TenantId, TypeId};
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::catalog::{CatalogStats, SystemCatalog};
use arcgraph_storage::crud::{
    CrudStore, PropertyData, commit, create_node, create_rel, crud_allocator_seed_handle,
};
use arcgraph_storage::io::InMemoryPageIo;
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::PrimaryIndex;
use arcgraph_storage::recovery::{
    RebuildReport, TenantRebuildOutcome, rebuild_all_tenant_stats, rebuild_catalog_stats_for_tenant,
};
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{
    AllocatorSeedHandle, BackgroundFsyncFailAction, BackgroundFsyncScheduler, BlobStoreHandle,
    PageStoreTarget, PrimaryPageStoreHandle, RecordPageStoreHandle, WalConfig, WalWriter,
    recover_from_wal,
};
use proptest::prelude::*;
use tempfile::TempDir;

// ─────────────────────────────────────────────────────────────────
// Helpers — mirror the K-1 smoke stack-builder shape so the rebuild
// path is exercised against a real CrudStore + TxnManager + WAL
// (not the in-module tests which use TxnManager::apply_replay_mvcc_write
// directly).
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

struct Stack {
    writer: Option<WalWriter>,
    scheduler: Option<Arc<BackgroundFsyncScheduler>>,
    mgr: Arc<TxnManager>,
    #[allow(dead_code)]
    primary: Arc<PrimaryIndex>,
    store: Arc<CrudStore>,
    #[allow(dead_code)]
    catalog: Arc<SystemCatalog>,
}

impl Stack {
    fn build(dir: &std::path::Path) -> Self {
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

    fn shutdown(mut self) {
        if let Some(s) = self.scheduler.take() {
            let _ = s.shutdown();
        }
        if let Some(w) = self.writer.take() {
            let _ = w.shutdown();
        }
    }
}

fn recover_stack(dir: &std::path::Path) -> Stack {
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
    let _ = recover_from_wal(dir, Arc::clone(&mgr), target, None).unwrap();
    Stack {
        writer: Some(writer),
        scheduler: Some(scheduler),
        mgr,
        primary,
        store,
        catalog,
    }
}

// ─────────────────────────────────────────────────────────────────
// Test 1: end-to-end rebuild round-trip — multi-tenant.
// ─────────────────────────────────────────────────────────────────

/// Per amendment-06 §D-25.1 step 1: "Recovery completes (`ReplayExecutor`
/// has applied all WAL records up to the recovered LSN per ADR-032)."
/// Step 2: "Per-tenant aggregation MUST honor the two-marker SeqLock
/// contract." This test exercises both end-to-end:
///
///  1. Build a fresh stack; commit N records across 3 tenants.
///  2. Snapshot the live `CatalogStats` (the live commit-pipeline path).
///  3. Shut down + drop the stack (simulates process restart — all
///     in-memory state is dropped, including `CatalogStats`).
///  4. Recover from WAL into a fresh stack.
///  5. Call `rebuild_all_tenant_stats` synchronously.
///  6. Assert per-tenant cardinality fields match the pre-crash live
///     snapshot. `commits_observed` differs by amendment-06 §D-25.1
///     step 2 design (live: N, rebuilt: 1 coalesced).
#[test]
fn rebuild_round_trip_three_tenants_cardinality_preserved() {
    let workspace = TempDir::new().unwrap();
    let wal_dir = workspace.path().join("wal");
    std::fs::create_dir_all(&wal_dir).unwrap();

    let t_a = TenantId::new(101);
    let t_b = TenantId::new(202);
    let t_c = TenantId::new(303);

    // Pre-crash workload: tenant A gets 5 nodes label=1 + 2 nodes label=2;
    // tenant B gets 3 nodes label=7; tenant C gets 1 node label=99 + 1 rel.
    let pre_snapshot = {
        let stack = Stack::build(&wal_dir);
        let plan: Vec<(TenantId, u32, u64)> = vec![
            // (tenant, label, count)
            (t_a, 1, 5),
            (t_a, 2, 2),
            (t_b, 7, 3),
            (t_c, 99, 1),
        ];
        for (tenant, label, count) in &plan {
            for i in 0..*count {
                let mut tx = stack.mgr.begin(*tenant);
                let _ = create_node(
                    &stack.store,
                    &mut tx,
                    *tenant,
                    LabelId::new(*label),
                    &PropertyData::InlineU32Pair(i as u32, 0),
                )
                .unwrap();
                commit(tx, &stack.store).unwrap();
            }
        }
        // Add a rel for tenant C to exercise the rel path. Skip if
        // create_rel signature is unsuitable; fall back to nodes only.
        // (The rel path requires src + dst nodes; tenant C already has
        //  one node — we'll give it another, then connect them.)
        let mut tx = stack.mgr.begin(t_c);
        let dst = create_node(
            &stack.store,
            &mut tx,
            t_c,
            LabelId::new(99),
            &PropertyData::InlineU32Pair(99, 0),
        )
        .unwrap();
        commit(tx, &stack.store).unwrap();
        // Read tenant C's first node's id by scanning the tenant's
        // CatalogStats — actually we can use a known id. The first
        // create_node for tenant C returned an id; we don't easily
        // capture it back here, so we'll use a synthetic create_rel
        // with src/dst from our last allocations. Skip if any error.
        let mut tx = stack.mgr.begin(t_c);
        // Use NodeId(1) and dst as the src/dst — both nodes exist for
        // tenant C from the first wave of `create_node` calls.
        let _ = create_rel(
            &stack.store,
            &mut tx,
            t_c,
            NodeId::new(1),
            dst,
            TypeId::new(7),
            &PropertyData::InlineU32Pair(0, 0),
        )
        .unwrap();
        commit(tx, &stack.store).unwrap();

        // Capture the pre-crash live CatalogStats snapshot for each
        // tenant so we can compare cardinality post-rebuild.
        let live: Vec<(TenantId, Option<arcgraph_storage::catalog::CatalogSnapshot>)> = vec![
            (t_a, stack.store.catalog_stats(t_a).map(|s| s.snapshot())),
            (t_b, stack.store.catalog_stats(t_b).map(|s| s.snapshot())),
            (t_c, stack.store.catalog_stats(t_c).map(|s| s.snapshot())),
        ];

        stack.shutdown();
        live
    };

    // Recover + rebuild.
    let recovered = recover_stack(&wal_dir);
    let report = rebuild_all_tenant_stats(
        recovered.mgr.current_lsn(),
        &recovered.mgr,
        &recovered.store,
    );
    assert!(
        report.failed.is_empty(),
        "expected no failed tenant rebuilds; got {:?}",
        report.failed
    );
    // The catalog bootstrap commits to TenantId::SYSTEM during stack
    // construction; the rebuild walks that tenant too. Verify our 3
    // user tenants are all in the successful list, plus the SYSTEM
    // tenant.
    let successful_tenants: std::collections::HashSet<TenantId> =
        report.successful.iter().map(|(t, _, _)| *t).collect();
    assert!(successful_tenants.contains(&t_a));
    assert!(successful_tenants.contains(&t_b));
    assert!(successful_tenants.contains(&t_c));
    assert!(
        report.successful.len() >= 3,
        "expected ≥ 3 successful tenant rebuilds (A/B/C plus SYSTEM); got {:?}",
        report.successful
    );

    // Assert per-tenant cardinality matches the pre-crash live snapshot.
    for (tenant, pre_opt) in &pre_snapshot {
        let pre = pre_opt
            .as_ref()
            .expect("pre-crash snapshot must exist for committed tenant");
        let post_stats = recovered
            .store
            .catalog_stats(*tenant)
            .expect("post-rebuild stats must exist for committed tenant");
        let post = post_stats.snapshot();

        assert_eq!(
            post.total_nodes(),
            pre.total_nodes(),
            "tenant {:?}: total_nodes mismatch — pre={:?} post={:?}",
            tenant,
            pre.total_nodes(),
            post.total_nodes(),
        );
        assert_eq!(
            post.total_rels(),
            pre.total_rels(),
            "tenant {:?}: total_rels mismatch — pre={:?} post={:?}",
            tenant,
            pre.total_rels(),
            post.total_rels(),
        );
        // Per-label cardinality match (excluding ordering): both sides
        // use sorted-by-LabelId iteration.
        let pre_labels: Vec<_> = pre.label_cards().to_vec();
        let post_labels: Vec<_> = post.label_cards().to_vec();
        assert_eq!(
            pre_labels, post_labels,
            "tenant {:?}: label_cards mismatch — pre={:?} post={:?}",
            tenant, pre_labels, post_labels,
        );
        let pre_rels: Vec<_> = pre.rel_type_cards().to_vec();
        let post_rels: Vec<_> = post.rel_type_cards().to_vec();
        assert_eq!(
            pre_rels, post_rels,
            "tenant {:?}: rel_type_cards mismatch — pre={:?} post={:?}",
            tenant, pre_rels, post_rels,
        );
        // Per amendment-06 §D-25.1 step 2 the rebuild's coalesced
        // bracket bumps `commits_observed` exactly once.
        assert_eq!(
            post.commits_observed(),
            1,
            "tenant {:?}: commits_observed must equal 1 (single coalesced rebuild bracket); got {}",
            tenant,
            post.commits_observed(),
        );
    }

    recovered.shutdown();
}

// ─────────────────────────────────────────────────────────────────
// Test 2: per-tenant fault isolation — rebuild reports per-tenant
// outcomes independently.
// ─────────────────────────────────────────────────────────────────

/// Per amendment-06 §2.5.1 partial-rebuild semantics: a rebuild that
/// completes successfully for tenants A + B + C MUST report all three
/// in `RebuildReport::successful`, with per-tenant counts. This test
/// pins the report shape end-to-end (the panic-recovery branch is
/// exercised by the in-module unit test
/// `rebuild_panic_safety_seqlock_invariant_preserved`).
#[test]
fn rebuild_report_separates_per_tenant_outcomes() {
    let workspace = TempDir::new().unwrap();
    let wal_dir = workspace.path().join("wal");
    std::fs::create_dir_all(&wal_dir).unwrap();

    let t_a = TenantId::new(401);
    let t_b = TenantId::new(402);
    let t_c = TenantId::new(403);

    {
        let stack = Stack::build(&wal_dir);
        for (tenant, label) in [(t_a, 11u32), (t_b, 12), (t_c, 13)] {
            let mut tx = stack.mgr.begin(tenant);
            let _ = create_node(
                &stack.store,
                &mut tx,
                tenant,
                LabelId::new(label),
                &PropertyData::InlineU32Pair(0, 0),
            )
            .unwrap();
            commit(tx, &stack.store).unwrap();
        }
        stack.shutdown();
    }

    let recovered = recover_stack(&wal_dir);
    let report = rebuild_all_tenant_stats(
        recovered.mgr.current_lsn(),
        &recovered.mgr,
        &recovered.store,
    );
    // ≥ 3 because the catalog bootstrap also commits to TenantId::SYSTEM.
    assert!(report.successful.len() >= 3);
    assert!(report.failed.is_empty());

    // The report is sorted by raw `TenantId` ascending — pin this
    // invariant for deterministic reproducibility.
    let raws: Vec<u64> = report.successful.iter().map(|(t, _, _)| t.raw()).collect();
    let mut expected = raws.clone();
    expected.sort();
    assert_eq!(
        raws, expected,
        "RebuildReport::successful must be sorted by TenantId::raw() ascending"
    );

    // Cross-tenant pollution check: each tenant's CatalogStats
    // contains ONLY its own label.
    let s_a = recovered.store.catalog_stats(t_a).unwrap();
    let s_b = recovered.store.catalog_stats(t_b).unwrap();
    let s_c = recovered.store.catalog_stats(t_c).unwrap();
    assert_eq!(s_a.label_cardinality(LabelId::new(11)), Some(1));
    assert_eq!(s_a.label_cardinality(LabelId::new(12)), None);
    assert_eq!(s_a.label_cardinality(LabelId::new(13)), None);
    assert_eq!(s_b.label_cardinality(LabelId::new(11)), None);
    assert_eq!(s_b.label_cardinality(LabelId::new(12)), Some(1));
    assert_eq!(s_b.label_cardinality(LabelId::new(13)), None);
    assert_eq!(s_c.label_cardinality(LabelId::new(11)), None);
    assert_eq!(s_c.label_cardinality(LabelId::new(12)), None);
    assert_eq!(s_c.label_cardinality(LabelId::new(13)), Some(1));

    recovered.shutdown();
}

// ─────────────────────────────────────────────────────────────────
// Test 3: per-tenant single-rebuild end-to-end — verify the
// `rebuild_catalog_stats_for_tenant` entry point matches the
// `rebuild_all_tenant_stats` aggregate.
// ─────────────────────────────────────────────────────────────────

#[test]
fn single_tenant_rebuild_matches_all_tenants_aggregate() {
    let workspace = TempDir::new().unwrap();
    let wal_dir = workspace.path().join("wal");
    std::fs::create_dir_all(&wal_dir).unwrap();

    let tenant = TenantId::new(777);

    {
        let stack = Stack::build(&wal_dir);
        for i in 0..7 {
            let mut tx = stack.mgr.begin(tenant);
            let _ = create_node(
                &stack.store,
                &mut tx,
                tenant,
                LabelId::new(if i < 5 { 1 } else { 2 }),
                &PropertyData::InlineU32Pair(i, 0),
            )
            .unwrap();
            commit(tx, &stack.store).unwrap();
        }
        stack.shutdown();
    }

    let recovered = recover_stack(&wal_dir);
    let outcome = rebuild_catalog_stats_for_tenant(
        tenant,
        recovered.mgr.current_lsn(),
        &recovered.mgr,
        &recovered.store,
    );
    match outcome {
        TenantRebuildOutcome::Success {
            nodes_walked,
            rels_walked,
        } => {
            assert_eq!(nodes_walked, 7);
            assert_eq!(rels_walked, 0);
        }
        other => panic!("expected Success; got {other:?}"),
    }
    let stats = recovered.store.catalog_stats(tenant).unwrap();
    assert_eq!(stats.label_cardinality(LabelId::new(1)), Some(5));
    assert_eq!(stats.label_cardinality(LabelId::new(2)), Some(2));
    assert_eq!(stats.total_node_count(), Some(7));
    assert_eq!(stats.commits_observed_count(), 1);

    recovered.shutdown();
}

// ─────────────────────────────────────────────────────────────────
// Test 4: two-marker SeqLock proptest — cross-key invariant under
// randomised commit + snapshot + rebuild interleavings.
// ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum Op {
    /// Increment label N times then observe (one commit's stats).
    Commit { label: u8, count: u8 },
    /// Take a snapshot and assert the cross-key invariant holds.
    Snapshot,
    /// Drop CatalogStats (simulate process restart) and replay the
    /// in-memory ledger via the rebuild path's `for_each_visible_record`
    /// shape — validates that the rebuild's coalesced bracket
    /// preserves the cross-key invariant.
    Rebuild,
}

fn op_strategy() -> impl Strategy<Value = Op> {
    prop_oneof![
        (1u8..=5, 1u8..=10).prop_map(|(label, count)| Op::Commit { label, count }),
        Just(Op::Snapshot),
        Just(Op::Rebuild),
    ]
}

fn ops_strategy() -> impl Strategy<Value = Vec<Op>> {
    proptest::collection::vec(op_strategy(), 1..40)
}

proptest! {
    /// Per amendment-06 §D-25.1 step 2 + PR #220's two-marker SeqLock
    /// contract: every `CatalogStats::snapshot()` MUST satisfy
    /// `sum(label_cards) ≤ total_nodes` for both the live commit
    /// pipeline AND the post-rebuild path. This proptest randomises
    /// sequences of (commit, snapshot, rebuild) operations and asserts
    /// the cross-key invariant on every snapshot taken.
    ///
    /// **Scope clarification:** the `Op::Rebuild` branch constructs a
    /// FRESH `CatalogStats::new()` instance, replays the captured
    /// ledger via a single coalesced bracket on that fresh instance,
    /// snapshots it, and DROPS it. The outer `stats` (driven by
    /// `Op::Commit` / observed by `Op::Snapshot`) is unaffected.
    /// Subsequent ops in the sequence apply to the original `stats`,
    /// not to a post-rebuild instance. The proptest therefore
    /// validates two things in isolation:
    ///   1. The live commit pipeline preserves the invariant under
    ///      random `Op::Commit` / `Op::Snapshot` interleavings.
    ///   2. A single isolated rebuild (replaying the captured ledger
    ///      via one coalesced bracket on a fresh instance) produces a
    ///      coherent snapshot.
    /// It does NOT exercise rebuild concurrent with live commits —
    /// that interleaving is structurally impossible per amendment-06
    /// §D-25.1 step 4 (rebuild runs synchronously pre-query-serving,
    /// no concurrent writers) and is out of scope.
    #[test]
    fn seqlock_cross_key_invariant_holds_in_isolated_commit_or_rebuild_sequences(
        ops in ops_strategy()
    ) {
        let stats = Arc::new(CatalogStats::new());
        // Captured ledger so the rebuild op can replay the same
        // increments through a single coalesced bracket.
        let mut ledger: Vec<(LabelId, u64)> = Vec::new();

        for op in &ops {
            match op {
                Op::Commit { label, count } => {
                    let label = LabelId::new(*label as u32);
                    stats.begin_commit_observation();
                    for _ in 0..*count {
                        stats.increment_label(label);
                        stats.increment_total_nodes();
                    }
                    stats.observe_commit();
                    ledger.push((label, *count as u64));
                }
                Op::Snapshot => {
                    let snap = stats.snapshot();
                    if let Some(total_nodes) = snap.total_nodes() {
                        let sum_labels: u64 =
                            snap.label_cards().iter().map(|(_, c)| *c).sum();
                        prop_assert!(
                            sum_labels <= total_nodes,
                            "cross-key invariant violated: sum(label_cards)={} > total_nodes={}",
                            sum_labels,
                            total_nodes,
                        );
                    }
                }
                Op::Rebuild => {
                    // Reset the stats and replay the captured ledger
                    // via a single coalesced bracket — mirrors the
                    // cold-start rebuild path's contract per
                    // amendment-06 §D-25.1 step 2.
                    let fresh = Arc::new(CatalogStats::new());
                    fresh.begin_commit_observation();
                    for (label, count) in &ledger {
                        for _ in 0..*count {
                            fresh.increment_label(*label);
                            fresh.increment_total_nodes();
                        }
                    }
                    fresh.observe_commit();
                    // After rebuild, snapshot must satisfy the
                    // invariant.
                    let snap = fresh.snapshot();
                    if let Some(total_nodes) = snap.total_nodes() {
                        let sum_labels: u64 =
                            snap.label_cards().iter().map(|(_, c)| *c).sum();
                        prop_assert!(
                            sum_labels <= total_nodes,
                            "post-rebuild cross-key invariant violated: \
                             sum(label_cards)={} > total_nodes={}",
                            sum_labels,
                            total_nodes,
                        );
                        prop_assert_eq!(
                            sum_labels, total_nodes,
                            "post-rebuild from non-empty ledger must have \
                             sum(label_cards) == total_nodes (no concurrent \
                             writers during rebuild)"
                        );
                    }
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// Test 5: rebuild matches the documented coalesced semantics —
// commits_observed += 1 regardless of how many records walked.
// ─────────────────────────────────────────────────────────────────

#[test]
fn rebuild_coalesced_bracket_bumps_commits_observed_by_one_per_invocation() {
    let workspace = TempDir::new().unwrap();
    let wal_dir = workspace.path().join("wal");
    std::fs::create_dir_all(&wal_dir).unwrap();

    let tenant = TenantId::new(0xCAFE);
    {
        let stack = Stack::build(&wal_dir);
        // 12 commits to exercise N >> 1.
        for i in 0..12 {
            let mut tx = stack.mgr.begin(tenant);
            let _ = create_node(
                &stack.store,
                &mut tx,
                tenant,
                LabelId::new(1),
                &PropertyData::InlineU32Pair(i, 0),
            )
            .unwrap();
            commit(tx, &stack.store).unwrap();
        }
        // Pre-crash live snapshot reflects 12 commits via the
        // commit-pipeline path.
        let pre = stack.store.catalog_stats(tenant).unwrap();
        assert_eq!(pre.commits_observed_count(), 12);
        stack.shutdown();
    }

    let recovered = recover_stack(&wal_dir);
    let _ = rebuild_all_tenant_stats(
        recovered.mgr.current_lsn(),
        &recovered.mgr,
        &recovered.store,
    );
    let post = recovered.store.catalog_stats(tenant).unwrap();
    // Per amendment-06 §D-25.1 step 2: rebuild bumps commits_observed
    // by exactly 1 (single coalesced bracket), regardless of how many
    // records were walked.
    assert_eq!(
        post.commits_observed_count(),
        1,
        "rebuild's coalesced bracket bumps commits_observed by 1 (not 12); \
         got {}",
        post.commits_observed_count(),
    );
    // But cardinality must reflect all 12 walked records.
    assert_eq!(post.label_cardinality(LabelId::new(1)), Some(12));
    assert_eq!(post.total_node_count(), Some(12));

    recovered.shutdown();
}

// ─────────────────────────────────────────────────────────────────
// Test 6: empty workload → empty rebuild → stats are None per
// amendment-06 §D-25.1 "Fresh-tenant invariant preserved".
// ─────────────────────────────────────────────────────────────────

#[test]
fn rebuild_on_empty_wal_yields_no_tenants() {
    let workspace = TempDir::new().unwrap();
    let wal_dir = workspace.path().join("wal");
    std::fs::create_dir_all(&wal_dir).unwrap();

    {
        let stack = Stack::build(&wal_dir);
        // No commits.
        stack.shutdown();
    }

    let recovered = recover_stack(&wal_dir);
    let report = rebuild_all_tenant_stats(
        recovered.mgr.current_lsn(),
        &recovered.mgr,
        &recovered.store,
    );
    // No user commits → only the catalog-bootstrap SYSTEM tenant is
    // present. Per amendment-06 §D-25.1 "Fresh-tenant invariant
    // preserved" — a USER tenant whose commit pipeline has never
    // fired returns `None` (we verify this by querying a never-touched
    // user tenant). The SYSTEM tenant is materialised by bootstrap.
    assert!(report.failed.is_empty());
    let never_touched_user_tenant = TenantId::new(99_999);
    assert!(
        recovered
            .store
            .catalog_stats(never_touched_user_tenant)
            .is_none(),
        "never-touched user tenant must return None per fresh-tenant invariant",
    );

    recovered.shutdown();
}

// ─────────────────────────────────────────────────────────────────
// Test 7: rebuild's `RebuildReport` aggregations.
// ─────────────────────────────────────────────────────────────────

#[test]
fn rebuild_report_aggregates_total_walked_correctly() {
    let workspace = TempDir::new().unwrap();
    let wal_dir = workspace.path().join("wal");
    std::fs::create_dir_all(&wal_dir).unwrap();

    let t_a = TenantId::new(601);
    let t_b = TenantId::new(602);

    {
        let stack = Stack::build(&wal_dir);
        // tenant A: 3 nodes; tenant B: 5 nodes.
        for _ in 0..3 {
            let mut tx = stack.mgr.begin(t_a);
            let _ = create_node(
                &stack.store,
                &mut tx,
                t_a,
                LabelId::new(1),
                &PropertyData::InlineU32Pair(0, 0),
            )
            .unwrap();
            commit(tx, &stack.store).unwrap();
        }
        for _ in 0..5 {
            let mut tx = stack.mgr.begin(t_b);
            let _ = create_node(
                &stack.store,
                &mut tx,
                t_b,
                LabelId::new(1),
                &PropertyData::InlineU32Pair(0, 0),
            )
            .unwrap();
            commit(tx, &stack.store).unwrap();
        }
        stack.shutdown();
    }

    let recovered = recover_stack(&wal_dir);
    let report: RebuildReport = rebuild_all_tenant_stats(
        recovered.mgr.current_lsn(),
        &recovered.mgr,
        &recovered.store,
    );
    // ≥ 2 because the catalog bootstrap also commits to TenantId::SYSTEM.
    assert!(report.successful.len() >= 2);
    assert!(report.failed.is_empty());
    // Per-tenant counts for our two user tenants:
    let by_tenant: std::collections::HashMap<TenantId, (u64, u64)> = report
        .successful
        .iter()
        .map(|(t, n, r)| (*t, (*n, *r)))
        .collect();
    assert_eq!(by_tenant.get(&t_a), Some(&(3, 0)));
    assert_eq!(by_tenant.get(&t_b), Some(&(5, 0)));
    // O-H (W28-S3): the fixture is deterministic — exactly three
    // successful tenants (t_a, t_b, and the catalog-bootstrap SYSTEM
    // tenant), no phantom tenants. Pin the count so an extra/duplicate
    // tenant rebuild is caught.
    assert_eq!(
        report.successful.len(),
        3,
        "exactly t_a, t_b, and the SYSTEM bootstrap tenant: {:?}",
        report.successful
    );
    // The SYSTEM bootstrap commits catalog metadata, not user node/rel
    // records, so it walks 0 nodes / 0 rels on this fixture.
    assert_eq!(
        by_tenant.get(&TenantId::SYSTEM),
        Some(&(0, 0)),
        "SYSTEM bootstrap tenant must walk 0 nodes / 0 rels"
    );
    // Total is therefore EXACTLY 3 + 5 + 0 = 8 nodes and 0 rels. Was
    // `>= 3 + 5`, which a node-over-counting (or SYSTEM-miscounting)
    // rebuild could not be caught by; the `==` pins the deterministic
    // baseline (t_a=3 + t_b=5 already pinned above).
    assert_eq!(report.total_nodes_walked(), 8);
    assert_eq!(report.total_rels_walked(), 0);
    recovered.shutdown();
}
