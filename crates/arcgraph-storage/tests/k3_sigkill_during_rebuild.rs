//! K-3 SIGKILL-during-rebuild seam (issue #256; closes the K-1c/K-1d
//! /K-2/K-3 fault-coverage matrix).
//!
//! ## What this test verifies
//!
//! The cold-start MVCC stats rebuild bracket
//! ([`arcgraph_storage::recovery::rebuild_all_tenant_stats`] +
//! [`arcgraph_storage::recovery::rebuild_catalog_stats_for_tenant`])
//! ships per ADR-038 amendment-06 §D-25.1 step 2 with the **4-invariant
//! SeqLock primitive** (per `feedback_seqlock_panic_safety_primitive.md`):
//!
//! 1. `begin_commit_observation()` OUTSIDE `catch_unwind`.
//! 2. walk INSIDE `AssertUnwindSafe`.
//! 3. `observe_commit()` UNCONDITIONALLY OUTSIDE.
//! 4. panic SWALLOWED for per-tenant isolation.
//!
//! In-process panic safety is exercised by the existing M4-41 unit
//! tests. **SIGKILL is a different beast**: process is dead; no Drop;
//! OS reaps. The recovery path's behavior on next boot must:
//!
//! - Detect a half-bracketed SeqLock state (`commits_started >
//!   commits_observed`) — IMPOSSIBLE in this codebase post-rebuild
//!   because the SIGKILL kills the in-memory `CatalogStats` along with
//!   the process; on next boot the `CatalogStats` is freshly
//!   constructed (counters at zero) and the rebuild brackets re-fire.
//! - The `commits_observed` counter is monotone non-decreasing per
//!   tenant across recovery cycles (per amendment-06 §D-25.1 invariant).
//! - The post-recovery snapshot is fully reverted (no partial
//!   cardinality state from the SIGKILL'd run).
//!
//! Per the issue #256 spec there are three fault windows:
//!
//! 1. **SIGKILL mid-rebuild** — between `begin_commit_observation` and
//!    `observe_commit`. No Drop; SeqLock invariant left half-bracketed
//!    in memory but the in-memory state dies with the process.
//! 2. **SIGKILL between recover_from_wal returning and
//!    rebuild_all_tenant_stats starting** — process crashes after WAL
//!    replay but before stats rebuild begins. Next boot must re-recover
//!    + re-rebuild with no skipped tenants.
//! 3. **SIGKILL after rebuild but before strict-oracle assertion fires**
//!    — process crashes mid-recovery-validation. The post-rebuild state
//!    is on-disk durable (WAL); the in-memory CatalogStats is gone but
//!    next-boot rebuild reproduces it.
//!
//! ## Test layout
//!
//! - **3 unit tests** model each window in-process by replacing SIGKILL
//!   with "drop the K3Stack mid-bracket" — the in-process equivalent
//!   of a process exit (no Drop on the `CatalogStats`'s atomics; the
//!   stack's owning `Arc` chain is released; next-boot reconstruction
//!   matches the SIGKILL recovery contract).
//! - **1 subprocess-fork integration test** (`#[ignore]`'d by default;
//!   gated by `K3_SIGKILL_REBUILD=1`) — the load-bearing seam: forks
//!   a child running the rebuild workload, SIGKILLs it during the
//!   rebuild bracket, then re-opens the WAL in the parent + verifies
//!   rebuild equivalence (post-SIGKILL rebuild == no-SIGKILL rebuild
//!   for the same WAL state).
//!
//! ## Why `#[ignore]` for the subprocess test
//!
//! The subprocess fork-and-SIGKILL path adds ~2 s wall + spawns a
//! re-exec'd test binary. CI runs the unit tests (which exercise the
//! same contract via in-process drop) on every push; the subprocess
//! variant is operator-grade, gated by `K3_SIGKILL_REBUILD=1`.
//!
//! ## Run
//!
//! ```ignore
//! # Unit tests (always run):
//! cargo test -p arcgraph-storage --release --test k3_sigkill_during_rebuild
//!
//! # Full subprocess SIGKILL integration:
//! K3_SIGKILL_REBUILD=1 cargo test -p arcgraph-storage --release \
//!     --test k3_sigkill_during_rebuild -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::sync::{Arc, Once};
use std::time::Duration;

use arcgraph_core::{LabelId, NodeId, TenantId};
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::catalog::SystemCatalog;
use arcgraph_storage::crud::{
    CrudStore, PropertyData, commit, create_node, crud_allocator_seed_handle,
};
use arcgraph_storage::io::InMemoryPageIo;
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::PrimaryIndex;
use arcgraph_storage::recovery::{
    TenantRebuildOutcome, rebuild_all_tenant_stats, rebuild_catalog_stats_for_tenant,
};
use arcgraph_storage::test_harness::k1::subprocess::{
    SubprocessWorkloadRegistry, WORKLOAD_CLEAN_EXIT_CODE, maybe_dispatch_subprocess_workload,
    run_with_crash_window,
};
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{
    AllocatorSeedHandle, BackgroundFsyncFailAction, BackgroundFsyncScheduler, BlobStoreHandle,
    PageStoreTarget, PrimaryPageStoreHandle, RecordPageStoreHandle, WalConfig, WalWriter,
    recover_from_wal,
};
use tempfile::TempDir;

const SIGKILL_REBUILD_ENV: &str = "K3_SIGKILL_REBUILD";
const WORKLOAD_NAME: &str = "k3_sigkill_during_rebuild_workload";

fn sigkill_subprocess_enabled() -> bool {
    std::env::var(SIGKILL_REBUILD_ENV)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

// ─────────────────────────────────────────────────────────────────
// K3RebuildStack — local helper (mirrors K-1/K-2 stack helpers)
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

struct K3RebuildStack {
    writer: Option<WalWriter>,
    scheduler: Option<Arc<BackgroundFsyncScheduler>>,
    mgr: Arc<TxnManager>,
    #[allow(dead_code)]
    primary: Arc<PrimaryIndex>,
    store: Arc<CrudStore>,
    #[allow(dead_code)]
    catalog: Arc<SystemCatalog>,
}

impl K3RebuildStack {
    fn build(dir: &Path) -> Self {
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

/// Recover WAL but DO NOT call `rebuild_all_tenant_stats`. Used by the
/// "window 2" unit test to model the seam between recovery and rebuild.
fn recover_stack_no_rebuild(dir: &Path) -> K3RebuildStack {
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
    K3RebuildStack {
        writer: Some(writer),
        scheduler: Some(scheduler),
        mgr,
        primary,
        store,
        catalog,
    }
}

/// Recover WAL AND call `rebuild_all_tenant_stats` (the canonical
/// post-recovery shape used by K-1/K-2 stacks).
///
/// Per amendment-06 §2.5.1 a panicking tenant rebuild surfaces in
/// `report.failed`; we log it here so a SIGKILL-bracket regression
/// that breaks per-tenant rebuild surfaces in stderr instead of being
/// discarded with the report.
fn recover_stack_with_rebuild(dir: &Path) -> K3RebuildStack {
    let stack = recover_stack_no_rebuild(dir);
    let rebuild_report =
        rebuild_all_tenant_stats(stack.mgr.current_lsn(), &stack.mgr, &stack.store);
    if !rebuild_report.failed.is_empty() {
        tracing::error!(
            target: "arcgraph_storage::recovery",
            failed = ?rebuild_report.failed,
            "rebuild_all_tenant_stats reported per-tenant failures during K-3 SIGKILL-rebuild recover"
        );
    }
    stack
}

// ─────────────────────────────────────────────────────────────────
// Workload helper
// ─────────────────────────────────────────────────────────────────

fn commit_n_nodes(stack: &K3RebuildStack, tenant: TenantId, label: u32, n: u32) -> Vec<NodeId> {
    let mut ids = Vec::with_capacity(n as usize);
    for i in 0..n {
        let mut tx = stack.mgr.begin(tenant);
        let id = create_node(
            &stack.store,
            &mut tx,
            tenant,
            LabelId::new(label),
            &PropertyData::InlineU32Pair(i, i.wrapping_mul(7)),
        )
        .expect("create_node");
        commit(tx, &stack.store).expect("commit");
        ids.push(id);
    }
    ids
}

// ─────────────────────────────────────────────────────────────────
// Unit test 1 — Window 1: half-bracketed mid-rebuild via panic
// ─────────────────────────────────────────────────────────────────

/// Issue #256 window 1: SIGKILL mid-rebuild — between
/// `begin_commit_observation` and `observe_commit`. SIGKILL is
/// modeled in-process by:
///
/// 1. Build a stack; commit N nodes; shutdown.
/// 2. Recover (no rebuild yet).
/// 3. Manually call `begin_commit_observation()` for one tenant
///    (modeling the rebuild bracket's first marker).
/// 4. **Drop the stack mid-bracket** — the equivalent of SIGKILL
///    from the recovery contract's POV: the in-memory `CatalogStats`
///    `commits_started` counter is gone with the stack; on-disk
///    state is unaffected.
/// 5. Recover + rebuild from scratch; assert the rebuilt
///    `CatalogStats` matches what a no-fault rebuild would produce.
///
/// The contract: the next-boot rebuild MUST reproduce the canonical
/// post-rebuild state regardless of any half-bracketed in-memory
/// state from the SIGKILL'd run.
#[test]
fn window1_sigkill_mid_rebuild_next_boot_converges() {
    let workspace = TempDir::new().expect("tmpdir");
    let wal_dir = workspace.path().join("wal");
    std::fs::create_dir_all(&wal_dir).expect("mkdir");

    let tenant = TenantId::new(101);
    const NODE_COUNT: u32 = 7;

    // (a) Pre-crash: commit 7 nodes; shutdown cleanly.
    {
        let stack = K3RebuildStack::build(&wal_dir);
        commit_n_nodes(&stack, tenant, 11, NODE_COUNT);
        stack.shutdown();
    }

    // (b) Recovery cycle 1 (the "SIGKILL'd" run): recover, fire the
    //     begin marker, then DROP the stack mid-bracket (modeling
    //     SIGKILL). The on-disk WAL is untouched; the in-memory
    //     half-bracketed state dies with the dropped stack.
    {
        let stack = recover_stack_no_rebuild(&wal_dir);
        let stats = stack.store.init_catalog_stats(tenant);
        stats.begin_commit_observation();
        // SIGKILL surrogate: drop the entire stack without observing.
        // No `observe_commit()` runs. The in-process equivalent of
        // a process kill mid-bracket.
        stack.shutdown();
    }

    // (c) Recovery cycle 2 (the "next boot" run): recover + rebuild
    //     from scratch; assert the rebuilt state matches what a
    //     no-fault rebuild would produce.
    let recovered = recover_stack_with_rebuild(&wal_dir);

    let post = recovered
        .store
        .catalog_stats(tenant)
        .expect("post-rebuild stats must exist for committed tenant");
    let snap = post.snapshot();
    assert_eq!(
        snap.total_nodes(),
        Some(NODE_COUNT as u64),
        "window 1: post-rebuild total_nodes must equal pre-crash count; \
         got {:?}",
        snap.total_nodes()
    );
    assert_eq!(
        snap.label_cards()
            .iter()
            .find(|(l, _)| l.raw() == 11)
            .map(|(_, c)| *c),
        Some(NODE_COUNT as u64),
        "window 1: post-rebuild label_card for label=11 must equal pre-crash count"
    );
    // commits_observed is the rebuild's coalesced single-bracket per
    // amendment-06 §D-25.1 step 2. The "SIGKILL'd" prior run's
    // half-bracketed state is gone (in-memory; died with the dropped
    // stack); the next-boot rebuild produces a clean single-bracket.
    assert_eq!(
        snap.commits_observed(),
        1,
        "window 1: post-rebuild commits_observed must equal 1 \
         (single coalesced rebuild bracket); got {}",
        snap.commits_observed()
    );

    recovered.shutdown();
}

// ─────────────────────────────────────────────────────────────────
// Unit test 2 — Window 2: SIGKILL between recover and rebuild
// ─────────────────────────────────────────────────────────────────

/// Issue #256 window 2: SIGKILL between `recover_from_wal` returning
/// and `rebuild_all_tenant_stats` starting. Modeled in-process by:
///
/// 1. Build a stack; commit N nodes; shutdown.
/// 2. Recovery cycle 1: recover (NO rebuild). Drop the stack — the
///    "SIGKILL" point. CatalogStats was never populated; no
///    half-bracket state.
/// 3. Recovery cycle 2: recover + rebuild from scratch; assert
///    rebuilt state matches reference.
///
/// The contract: the next-boot rebuild does not depend on the prior
/// run reaching the rebuild step.
#[test]
fn window2_sigkill_between_recover_and_rebuild_next_boot_converges() {
    let workspace = TempDir::new().expect("tmpdir");
    let wal_dir = workspace.path().join("wal");
    std::fs::create_dir_all(&wal_dir).expect("mkdir");

    let tenant = TenantId::new(202);
    const NODE_COUNT: u32 = 13;

    {
        let stack = K3RebuildStack::build(&wal_dir);
        commit_n_nodes(&stack, tenant, 22, NODE_COUNT);
        stack.shutdown();
    }

    // SIGKILL surrogate cycle 1: recover (no rebuild) then drop.
    {
        let stack = recover_stack_no_rebuild(&wal_dir);
        // Sanity: rebuild has not run, so CatalogStats for the tenant
        // either does not exist OR exists with commits_observed == 0.
        // (The stack init may pre-create some entries via catalog
        // bootstrap; the tenant-of-interest is the commit_n_nodes
        // tenant which has not been rebuilt yet.)
        if let Some(s) = stack.store.catalog_stats(tenant) {
            assert_eq!(
                s.commits_observed_count(),
                0,
                "window 2: pre-rebuild commits_observed for the \
                 commit-only tenant must be 0; got {}",
                s.commits_observed_count()
            );
        }
        stack.shutdown();
    }

    // Next boot: recover + rebuild.
    let recovered = recover_stack_with_rebuild(&wal_dir);
    let post = recovered
        .store
        .catalog_stats(tenant)
        .expect("post-rebuild stats must exist for committed tenant");
    let snap = post.snapshot();
    assert_eq!(
        snap.total_nodes(),
        Some(NODE_COUNT as u64),
        "window 2: post-rebuild total_nodes must equal pre-crash count"
    );
    assert_eq!(
        snap.commits_observed(),
        1,
        "window 2: post-rebuild commits_observed must equal 1 \
         (single coalesced rebuild bracket)"
    );
    recovered.shutdown();
}

// ─────────────────────────────────────────────────────────────────
// Unit test 3 — Window 3: SIGKILL after rebuild but before strict-oracle
// ─────────────────────────────────────────────────────────────────

/// Issue #256 window 3: SIGKILL after rebuild but before the strict-
/// oracle assertion fires. Modeled in-process by:
///
/// 1. Build a stack; commit N nodes; shutdown.
/// 2. Recovery cycle 1: recover + rebuild successfully. Drop the
///    stack — the "SIGKILL" point. CatalogStats is fully populated
///    in memory but dies with the dropped stack.
/// 3. Recovery cycle 2: recover + rebuild from scratch; assert
///    rebuilt state matches reference (proving rebuild is
///    idempotent across consecutive cycles).
///
/// The contract: the rebuild's `commits_observed = 1` does NOT bump
/// to 2 across two consecutive cycles — each cycle is an independent
/// fresh-tenant rebuild, single coalesced observation.
#[test]
fn window3_sigkill_after_rebuild_next_boot_idempotent() {
    let workspace = TempDir::new().expect("tmpdir");
    let wal_dir = workspace.path().join("wal");
    std::fs::create_dir_all(&wal_dir).expect("mkdir");

    let tenant = TenantId::new(303);
    const NODE_COUNT: u32 = 5;

    {
        let stack = K3RebuildStack::build(&wal_dir);
        commit_n_nodes(&stack, tenant, 33, NODE_COUNT);
        stack.shutdown();
    }

    // SIGKILL surrogate cycle 1: recover + rebuild successfully, then
    // drop. The rebuilt state is in memory only; the WAL on disk is
    // unchanged.
    let cycle1_snapshot = {
        let stack = recover_stack_with_rebuild(&wal_dir);
        let s = stack
            .store
            .catalog_stats(tenant)
            .expect("post-rebuild stats must exist");
        let snap = s.snapshot();
        stack.shutdown();
        snap
    };
    assert_eq!(
        cycle1_snapshot.total_nodes(),
        Some(NODE_COUNT as u64),
        "window 3 cycle 1: rebuild must populate total_nodes"
    );
    assert_eq!(
        cycle1_snapshot.commits_observed(),
        1,
        "window 3 cycle 1: rebuild must produce single coalesced bracket"
    );

    // Next boot: recover + rebuild AGAIN. The result must be byte-equal
    // to cycle 1 (rebuild is deterministic + idempotent — per
    // amendment-06 §D-25.1 step 2 — running it twice on the same WAL
    // produces the same CatalogStats with commits_observed == 1 each
    // time, NOT commits_observed == 2).
    let recovered = recover_stack_with_rebuild(&wal_dir);
    let post = recovered
        .store
        .catalog_stats(tenant)
        .expect("post-rebuild stats must exist (cycle 2)");
    let snap2 = post.snapshot();
    assert_eq!(
        snap2.total_nodes(),
        cycle1_snapshot.total_nodes(),
        "window 3: cycle 2 total_nodes must equal cycle 1"
    );
    assert_eq!(
        snap2.commits_observed(),
        1,
        "window 3: cycle 2 commits_observed must be 1 — rebuild is \
         per-recovery (one bracket per cold-start), NOT cumulative \
         across recovery cycles; got {}",
        snap2.commits_observed()
    );
    let pre_labels: Vec<_> = cycle1_snapshot.label_cards().to_vec();
    let post_labels: Vec<_> = snap2.label_cards().to_vec();
    assert_eq!(
        pre_labels, post_labels,
        "window 3: cycle 2 label_cards must equal cycle 1 (idempotent rebuild)"
    );
    recovered.shutdown();
}

// ─────────────────────────────────────────────────────────────────
// Unit test 4 — per-tenant rebuild monotone non-decreasing across cycles
// ─────────────────────────────────────────────────────────────────

/// Per amendment-06 §D-25.1 invariant referenced by issue #256:
/// "`commits_observed` is monotone non-decreasing OR snapshot is fully
/// reverted (never partial)".
///
/// At the per-tenant rebuild level, the rebuild produces
/// `commits_observed = 1` per cold-start. Across N consecutive
/// independent cold-starts (each modeling a fault recovery), the
/// counter snaps back to 1 each time (NOT N) because the in-memory
/// state is fresh on every recovery.
///
/// "Monotone non-decreasing" applies at the LIVE-stack level
/// (commits_observed grows as commits arrive); at the post-rebuild
/// level the invariant is "rebuild is idempotent + the counter is
/// always 1 post-rebuild".
///
/// The "fully reverted" alternative applies to the per-tenant
/// `PartialFailure` outcome — the test exercises the success path
/// here; the partial-failure path is exercised by the existing
/// in-module unit test
/// (`stats_rebuild::tests::rebuild_panic_safety_seqlock_invariant_preserved`).
#[test]
fn rebuild_commits_observed_resets_to_one_per_recovery_cycle() {
    let workspace = TempDir::new().expect("tmpdir");
    let wal_dir = workspace.path().join("wal");
    std::fs::create_dir_all(&wal_dir).expect("mkdir");

    let tenant = TenantId::new(404);
    const NODE_COUNT: u32 = 3;

    {
        let stack = K3RebuildStack::build(&wal_dir);
        commit_n_nodes(&stack, tenant, 44, NODE_COUNT);
        stack.shutdown();
    }

    for cycle in 1..=4 {
        let recovered = recover_stack_with_rebuild(&wal_dir);
        let post = recovered
            .store
            .catalog_stats(tenant)
            .expect("post-rebuild stats must exist on every cycle");
        let snap = post.snapshot();
        assert_eq!(
            snap.commits_observed(),
            1,
            "cycle {}: commits_observed must be 1 (per-recovery cold-start); \
             cumulative count would mean rebuild bumped a stale counter, \
             violating the SeqLock invariant",
            cycle
        );
        assert_eq!(
            snap.total_nodes(),
            Some(NODE_COUNT as u64),
            "cycle {}: total_nodes must equal pre-crash count",
            cycle
        );
        recovered.shutdown();
    }
}

// ─────────────────────────────────────────────────────────────────
// Unit test 5 — partial-failure path: per-tenant fault isolation
// ─────────────────────────────────────────────────────────────────

/// Per amendment-06 §2.5.1 partial-rebuild semantics: a rebuild that
/// panics for one tenant marks that tenant `recovery_failed` but
/// does NOT block other tenants' rebuilds.
///
/// SIGKILL is a process-level fault (not per-tenant), so the
/// "partial failure" of an in-process panic is a STRICTER contract
/// than SIGKILL would test (SIGKILL kills all tenants' rebuilds at
/// once). This test pins the per-tenant fault isolation contract
/// via the K-1 oracle's existing per-tenant snapshot mechanism.
///
/// Test shape: 3 tenants, all commit nodes; all 3 rebuild
/// successfully. The negative path (one tenant panicking via
/// `decode_node_bytes` failure) is exercised by the existing
/// in-module unit test; this test pins the success path's
/// per-tenant report shape under the K-3 SIGKILL framing.
#[test]
fn three_tenant_rebuild_per_tenant_independent() {
    let workspace = TempDir::new().expect("tmpdir");
    let wal_dir = workspace.path().join("wal");
    std::fs::create_dir_all(&wal_dir).expect("mkdir");

    let tenants = [TenantId::new(501), TenantId::new(502), TenantId::new(503)];
    {
        let stack = K3RebuildStack::build(&wal_dir);
        for (i, t) in tenants.iter().enumerate() {
            commit_n_nodes(&stack, *t, 50 + i as u32, 3);
        }
        stack.shutdown();
    }

    let recovered = recover_stack_no_rebuild(&wal_dir);
    let report = rebuild_all_tenant_stats(
        recovered.mgr.current_lsn(),
        &recovered.mgr,
        &recovered.store,
    );

    // All 3 tenants must succeed (catalog bootstrap also commits to
    // SYSTEM, so successful.len() may be ≥ 4 — we assert ≥ 3 for the
    // user tenants).
    assert!(
        report.successful.len() >= 3,
        "expected ≥ 3 successful tenants; got {:?}",
        report.successful
    );
    assert!(
        report.failed.is_empty(),
        "expected zero failed tenants; got {:?}",
        report.failed
    );

    for t in &tenants {
        let stats = recovered
            .store
            .catalog_stats(*t)
            .expect("post-rebuild stats per tenant");
        let snap = stats.snapshot();
        assert_eq!(
            snap.total_nodes(),
            Some(3),
            "tenant {:?}: total_nodes must be 3",
            t.raw()
        );
        assert_eq!(
            snap.commits_observed(),
            1,
            "tenant {:?}: rebuild bracket fired exactly once",
            t.raw()
        );
    }
    recovered.shutdown();
}

// ─────────────────────────────────────────────────────────────────
// Per-tenant rebuild — direct API exercise
// ─────────────────────────────────────────────────────────────────

/// Pin the per-tenant rebuild API directly. The test exercises the
/// function `rebuild_catalog_stats_for_tenant` (not `_all_tenant_`)
/// so a regression that changes only the per-tenant path's contract
/// surfaces here — even if the parallel driver's behavior is
/// unchanged.
#[test]
fn per_tenant_rebuild_api_returns_success_with_walked_counts() {
    let workspace = TempDir::new().expect("tmpdir");
    let wal_dir = workspace.path().join("wal");
    std::fs::create_dir_all(&wal_dir).expect("mkdir");

    let tenant = TenantId::new(606);
    const NODE_COUNT: u32 = 9;

    {
        let stack = K3RebuildStack::build(&wal_dir);
        commit_n_nodes(&stack, tenant, 66, NODE_COUNT);
        stack.shutdown();
    }

    let recovered = recover_stack_no_rebuild(&wal_dir);
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
            assert_eq!(
                nodes_walked, NODE_COUNT as u64,
                "per-tenant rebuild: nodes_walked must equal pre-crash count"
            );
            assert_eq!(
                rels_walked, 0,
                "per-tenant rebuild: rels_walked must be 0 (no rels committed)"
            );
        }
        TenantRebuildOutcome::PartialFailure { panic_message } => {
            panic!("per-tenant rebuild unexpectedly failed: {panic_message}")
        }
    }
    recovered.shutdown();
}

// ─────────────────────────────────────────────────────────────────
// Subprocess SIGKILL integration test (gated; the load-bearing pin)
// ─────────────────────────────────────────────────────────────────

mod subprocess_pin {
    use super::*;

    /// Subprocess workload: open the WAL stack, commit a
    /// few nodes for each of N tenants, sleep briefly so the parent's
    /// SIGKILL window can land mid-rebuild, then INVOKE the rebuild
    /// repeatedly in a loop. The parent SIGKILLs at a window tuned
    /// to land DURING the rebuild loop; the next-boot recovery in the
    /// parent must still pick up the WAL state cleanly.
    ///
    /// Returns [`WORKLOAD_CLEAN_EXIT_CODE`] only if the loop completes
    /// without SIGKILL — a harness-tuning signal (window too long /
    /// MAX iterations too small).
    pub fn rebuild_loop_workload(arg: &str) -> i32 {
        const MAX_ITERATIONS: u32 = 200;
        const SLEEP_PER_ITER: Duration = Duration::from_millis(20);

        let workspace = PathBuf::from(arg);
        let wal_dir = workspace.join("wal");
        if let Err(e) = std::fs::create_dir_all(&wal_dir) {
            eprintln!("k3 sigkill child: cannot mkdir {wal_dir:?}: {e}");
            return 99;
        }

        // Phase 1 — commit a few rows per tenant under the live stack.
        // Then drop. The WAL now holds a multi-tenant pre-crash state
        // the rebuild loop will exercise.
        {
            let stack = K3RebuildStack::build(&wal_dir);
            for (i, tenant_raw) in [701u64, 702, 703, 704, 705].iter().enumerate() {
                let t = TenantId::new(*tenant_raw);
                let label = 70 + i as u32;
                for j in 0..6u32 {
                    let mut tx = stack.mgr.begin(t);
                    let _ = create_node(
                        &stack.store,
                        &mut tx,
                        t,
                        LabelId::new(label),
                        &PropertyData::InlineU32Pair(j, j.wrapping_mul(11)),
                    )
                    .expect("create_node");
                    commit(tx, &stack.store).expect("commit");
                }
            }
            stack.shutdown();
        }

        // Phase 2 — rebuild loop. Each iteration: open the stack,
        // recover, rebuild, drop. The parent SIGKILLs sometime during
        // this loop; it can land between any of the steps.
        for _ in 0..MAX_ITERATIONS {
            let stack = recover_stack_with_rebuild(&wal_dir);
            std::thread::sleep(SLEEP_PER_ITER);
            stack.shutdown();
        }

        // Loop completed without SIGKILL — harness-tuning signal.
        WORKLOAD_CLEAN_EXIT_CODE
    }

    pub fn register_workload_once() {
        static ONCE: Once = Once::new();
        ONCE.call_once(|| {
            SubprocessWorkloadRegistry::register(WORKLOAD_NAME, rebuild_loop_workload);
        });
    }

    pub fn dispatch_if_subprocess() {
        register_workload_once();
        maybe_dispatch_subprocess_workload();
    }

    /// Subprocess router — routes the child re-exec'd test binary to
    /// the workload BEFORE any other test body runs. Same convention
    /// as `k1_subprocess_smoke::aaaa_subprocess_dispatcher_router`
    /// (alphabetically-first test name so the harness hits this
    /// before the gated `#[ignore]` test).
    #[test]
    fn aaaa_subprocess_dispatcher_router() {
        dispatch_if_subprocess();
    }

    #[test]
    #[ignore = "K-3 SIGKILL-during-rebuild subprocess pin — gated by \
                K3_SIGKILL_REBUILD=1; ~5 s wall. Closes #256."]
    fn k3_sigkill_during_rebuild_subprocess() {
        dispatch_if_subprocess();

        // V-1 (W28-S3): panic-by-default env-gate (was a silent
        // soft-skip — the W12δ HIGH-1 bug class per
        // `feedback_test_env_gate_panic_by_default.md`). This test is
        // `#[ignore]`'d off the default gauntlet; when invoked via
        // `--ignored` without K3_SIGKILL_REBUILD=1 it must PANIC (so a
        // missing campaign is loud) unless the operator explicitly opts
        // into a soft-skip via ARCGRAPH_K3_SIGKILL_REBUILD_SKIP_OK=1.
        if !sigkill_subprocess_enabled() {
            let skip_ok = std::env::var("ARCGRAPH_K3_SIGKILL_REBUILD_SKIP_OK").is_ok();
            if skip_ok {
                eprintln!(
                    "k3_sigkill_during_rebuild_subprocess: SKIPPING (opt-in via \
                     ARCGRAPH_K3_SIGKILL_REBUILD_SKIP_OK=1) — set \
                     {SIGKILL_REBUILD_ENV}=1 to run the SIGKILL-during-rebuild \
                     subprocess campaign instead"
                );
                return;
            }
            panic!(
                "k3_sigkill_during_rebuild_subprocess: required env flag \
                 {SIGKILL_REBUILD_ENV}=1 not set. This test is `#[ignore]`'d off \
                 the default gauntlet; when invoked via `--ignored`, \
                 {SIGKILL_REBUILD_ENV}=1 must be set so the SIGKILL-during-rebuild \
                 subprocess campaign actually runs. Set {SIGKILL_REBUILD_ENV}=1 to \
                 run, or ARCGRAPH_K3_SIGKILL_REBUILD_SKIP_OK=1 to opt into a \
                 soft-skip (hostile/CI envs only). Soft-skipping silently is the \
                 W12δ HIGH-1 bug class (feedback_test_env_gate_panic_by_default.md)."
            );
        }

        let workspace = TempDir::new().expect("workspace tmpdir");
        // Pre-create WAL dir so the child + parent agree on the layout.
        std::fs::create_dir_all(workspace.path().join("wal")).expect("mkdir wal");

        // Crash window tuned to land mid-rebuild loop. Phase 1 commits
        // (~30 commits) take ~50 ms; Phase 2 rebuilds at ~25 ms each.
        // 600 ms gives Phase 1 + ~20 rebuild iterations — plenty of
        // chance for SIGKILL to land mid-rebuild.
        let crash_after = Duration::from_millis(600);
        let started = std::time::Instant::now();
        let record = run_with_crash_window(WORKLOAD_NAME, workspace.path(), crash_after)
            .expect("crash window");
        eprintln!(
            "k3_sigkill_during_rebuild_subprocess: spawn→reap elapsed={:?} \
             elapsed_to_kill={:?} kill_succeeded={} sigkilled={} \
             exited_cleanly={} exit_status={:?}",
            started.elapsed(),
            record.elapsed_to_kill,
            record.kill_succeeded,
            record.was_sigkilled(),
            record.exited_cleanly(),
            record.exit_status,
        );

        assert!(
            !record.exited_cleanly(),
            "k3 sigkill subprocess: workload completed before SIGKILL — \
             window {crash_after:?} too long OR MAX_ITERATIONS too low"
        );
        assert!(
            record.kill_succeeded,
            "k3 sigkill subprocess: SIGKILL syscall failed — child gone \
             before crash window"
        );
        #[cfg(unix)]
        assert!(
            record.was_sigkilled(),
            "k3 sigkill subprocess: child must report SIGKILL signal-exit \
             on Unix; got {:?}",
            record.exit_status
        );

        // Parent: re-open the WAL the child wrote into. Recovery +
        // rebuild MUST complete cleanly even though the child was
        // SIGKILL'd mid-rebuild.
        let wal_dir = workspace.path().join("wal");
        let recovered = recover_stack_with_rebuild(&wal_dir);

        // Verify each pre-crash tenant's rebuild produced the expected
        // post-recovery state. The child committed 6 nodes per tenant
        // for 5 tenants under labels 70..75.
        for (i, tenant_raw) in [701u64, 702, 703, 704, 705].iter().enumerate() {
            let t = TenantId::new(*tenant_raw);
            let label = 70 + i as u32;
            let stats = recovered
                .store
                .catalog_stats(t)
                .unwrap_or_else(|| panic!("post-recovery stats missing for tenant {tenant_raw}"));
            let snap = stats.snapshot();
            assert_eq!(
                snap.total_nodes(),
                Some(6),
                "tenant {tenant_raw}: post-SIGKILL rebuild total_nodes must be 6"
            );
            assert_eq!(
                snap.label_cards()
                    .iter()
                    .find(|(l, _)| l.raw() == label)
                    .map(|(_, c)| *c),
                Some(6),
                "tenant {tenant_raw}: post-SIGKILL rebuild label_card for \
                 label={label} must be 6"
            );
            assert_eq!(
                snap.commits_observed(),
                1,
                "tenant {tenant_raw}: post-SIGKILL commits_observed must be 1 \
                 (single coalesced rebuild bracket)"
            );
        }

        recovered.shutdown();
    }
}
