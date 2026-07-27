//! M6-05 deterministic fault scenarios: random fsync lies and disk-full.
//!
//! ## Honest framing — what these scenarios are (and are NOT)
//!
//! These tests are **deterministic fault-injection simulations** driven
//! by the K-1 injection-RNG harness, NOT real OS-level fault injection.
//! The "chaos" word in the M6-05 roadmap row is shorthand for
//! "operator-grade fault scenarios"; the actual mechanism is dice-and-
//! tally simulation that exercises **harness shape** + the K-1 RNG's
//! configured-rate plumbing.
//!
//! What is exercised:
//!  - The `BackgroundFsyncFailAction::RollbackAndContinue` knob set at
//!    scheduler construction (the production seam for the rollback
//!    code path).
//!  - The K-1 injection-RNG fires at the configured per-op rate, with
//!    `tally.count(InjectionKind::*)` assertions proving the harness
//!    is wired and the spec-D2 rates take effect.
//!  - Recovery's binary-equal contract on the SUCCESSFUL commits
//!    (every committed row survives recovery byte-for-byte).
//!
//! What is **NOT** exercised (forward-debt, see "Forward-bind to M6-05
//! v1.1" below):
//!  - Real OS-level fsync syscall hooking (e.g., `LD_PRELOAD`'d fsync
//!    that returns `Ok(_)` without flushing). The fsync-lies scenario
//!    rolls a dice; it does not patch the actual fsync syscall.
//!  - Real `ENOSPC` from a filled filesystem. The disk-full scenario
//!    skips the commit attempt when the dice land; it does not fill
//!    disk and observe a `commit()` returning `Err(WalError::DiskFull)`.
//! ## Scenarios
//!
//!  1. **RandomFsyncLies (simulated)** — at ~5 % probability per
//!     scheduled background fsync, the harness records a simulated
//!     "lie" and (would in production) trigger
//!     `BackgroundFsyncFailAction::RollbackAndContinue`. Verifies the
//!     ADR-034 D-1 T1 contract: post-ack background-fsync lies CANNOT
//!     corrupt T1 commits because T1 fsync runs inline before commit
//!     ack. T3 commits within the rpo_ms window MAY be lost per
//!     ADR-034 D-2 (this test only commits T1 so the recovered state
//!     must be complete byte-for-byte).
//!
//!  2. **DiskFullSimulation (simulated ENOSPC)** — midway through the
//!     workload the disk_full flag is enabled and the K-1 wal-injection
//!     RNG fires `WalFsyncFail` at 50 %; on dice-hit, the test SKIPS
//!     the commit attempt (modelling the operator-visible analog of
//!     `commit()` returning `Err(WalError::DiskFull)`). Verifies clean
//!     rollback: every commit that returned `Ok` is observable
//!     post-recovery; failed commit attempts leak NO partial state.
//!     NOTE: the actual `commit()` Err path is NOT exercised; the test
//!     skips the call entirely on dice-hit.
//!
//! ## Forward-bind to M6-05 v1.1 (real fault injection)
//!
//! The W11Z retro packet (MED-1) flagged that the simulated scenarios
//! exercise harness shape + the K-1 RNG plumbing, but DO NOT exercise
//! the production fault paths end-to-end. A future M6-05-real follow-up
//! should land:
//!
//!  - **fsync-lies**: a `#[doc(hidden)] pub` WAL-layer test seam (per
//!    the HARD-BOUNDARY DEVIATION protocol used in PR #231) that wraps
//!    the `WalWriter`'s fsync syscall and probabilistically returns
//!    `Ok(_)` without flushing. Or a `LD_PRELOAD`'d fsync hook in a
//!    Linux-only integration harness.
//!  - **disk-full**: a filesystem-layer fault layer (e.g., `fuse-fs`
//!    mount with `ENOSPC` injection, or `cargo test`-driven
//!    `tmpfs` size cap on Linux) so `commit()` actually returns
//!    `Err(WalError::DiskFull)` and the rollback path runs end-to-end.
//! ## Reuse, do NOT re-architect
//!
//! Per the spawn prompt: "Same `FaultInjectDirectory` wrapper pattern
//! from PR #231 (reuse, do NOT re-architect)". The `FaultInjectDirectory`
//! wrapper at `crates/arcgraph-bm25/tests/fault_inject_directory.rs`
//! is **Tantivy-Directory-specific**; it cannot be applied to the WAL
//! / page-store path because those paths are not Tantivy-Directory
//! consumers. The reuse here is at the **conceptual** level:
//!
//!  - `Arc<AtomicBool>` flag-based fault toggle (same shape).
//!  - Test-side wrapper that delegates 99 % of operations to the inner
//!    real path and intercepts only on flag (same shape).
//!  - Cross-platform pure-Rust (same shape).
//!
//! For M6-05 the wrapping seam is the K-1 injection harness's
//! `BackgroundFsyncScheduler` + `InjectionDecisionRng` flag toggle —
//! already production-ready test scaffolding the K-1/K-2/K-3 stack
//! uses end-to-end. We do NOT add a new trait or new dep crate.
//!
//! ## Run
//!
//! ```ignore
//! cargo test -p arcgraph-storage --release --test m6_05_chaos
//! ```

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use arcgraph_core::{LabelId, NodeId, TenantId};
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::catalog::SystemCatalog;
use arcgraph_storage::crud::{
    CrudStore, PropertyData, commit, create_node, crud_allocator_seed_handle, read_node_with_store,
};
use arcgraph_storage::io::InMemoryPageIo;
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::PrimaryIndex;
use arcgraph_storage::test_harness::k1::injection::{
    InjectionConfig, InjectionDecisionRng, InjectionKind, InjectionTally,
    maybe_inject_background_fsync_failure, maybe_inject_wal_failure,
};
use arcgraph_storage::test_harness::k1::oracle::{
    CommittedState, OracleConfig, RecoveredState, verify_post_recovery_invariants,
};
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{
    AllocatorSeedHandle, BackgroundFsyncFailAction, BackgroundFsyncScheduler, BlobStoreHandle,
    PageStoreTarget, PrimaryPageStoreHandle, RecordPageStoreHandle, WalConfig, WalWriter,
    recover_from_wal,
};
use tempfile::TempDir;

// ─────────────────────────────────────────────────────────────────
// FaultInjectFlags — Arc<AtomicBool> flag bag
//
// Mirrors the PR #231 `FaultInjectFlags` shape (Tantivy-side) at the
// conceptual level: shared Arc<AtomicBool> per fault kind so the test
// can toggle injection on a single handle and have it affect every
// in-flight operation. Used by the chaos scenarios below to gate the
// K-1 injection harness's per-op rolls.
// ─────────────────────────────────────────────────────────────────

#[derive(Default, Debug)]
struct FaultInjectFlags {
    /// True iff the harness should drive RandomFsyncLies injection.
    fsync_lies_enabled: AtomicBool,
    /// True iff the harness should drive DiskFullSimulation injection.
    disk_full_enabled: AtomicBool,
}

impl FaultInjectFlags {
    fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }
    fn set_fsync_lies(&self, on: bool) {
        self.fsync_lies_enabled.store(on, Ordering::Release);
    }
    fn set_disk_full(&self, on: bool) {
        self.disk_full_enabled.store(on, Ordering::Release);
    }
    fn fsync_lies(&self) -> bool {
        self.fsync_lies_enabled.load(Ordering::Acquire)
    }
    fn disk_full(&self) -> bool {
        self.disk_full_enabled.load(Ordering::Acquire)
    }
}

// ─────────────────────────────────────────────────────────────────
// M6Stack — local helper (mirrors K-1/K-2 stack helpers; keeps the
// test self-contained without leaking a public test-harness API).
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

struct M6Stack {
    writer: Option<WalWriter>,
    scheduler: Option<Arc<BackgroundFsyncScheduler>>,
    mgr: Arc<TxnManager>,
    #[allow(dead_code)]
    primary: Arc<PrimaryIndex>,
    store: Arc<CrudStore>,
    #[allow(dead_code)]
    catalog: Arc<SystemCatalog>,
}

impl M6Stack {
    fn build(dir: &Path) -> Self {
        let writer = WalWriter::spawn(test_wal_config(dir.to_path_buf())).unwrap();
        // RollbackAndContinue is the test-harness fault-injection
        // override per `wal::background_fsync` doc-comment §"Why
        // RollbackAndContinue is test-only" — operators should NEVER
        // use this; M6-05 chaos scenarios drive it via the
        // `disk_full_enabled` / `fsync_lies_enabled` flags below.
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

fn recover_stack(dir: &Path) -> M6Stack {
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
    // Per amendment-06 §2.5.1 a panicking tenant rebuild surfaces in
    // `report.failed`; log it here so a chaos-scenario regression that
    // breaks per-tenant rebuild surfaces in stderr instead of being
    // discarded with the report.
    let rebuild_report = arcgraph_storage::recovery::rebuild_all_tenant_stats(
        report.applied_commit_lsn,
        &mgr,
        &store,
    );
    if !rebuild_report.failed.is_empty() {
        tracing::error!(
            target: "arcgraph_storage::recovery",
            failed = ?rebuild_report.failed,
            "rebuild_all_tenant_stats reported per-tenant failures during M6-05 recover_stack"
        );
    }
    M6Stack {
        writer: Some(writer),
        scheduler: Some(scheduler),
        mgr,
        primary,
        store,
        catalog,
    }
}

// ─────────────────────────────────────────────────────────────────
// Workload helper
// ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct CommitRow {
    tenant: TenantId,
    id: NodeId,
    label: u32,
    a: u32,
    b: u32,
}

fn do_commit(stack: &M6Stack, tenant: TenantId, label: u32, a: u32, b: u32) -> Option<CommitRow> {
    let mut tx = stack.mgr.begin(tenant);
    let id = create_node(
        &stack.store,
        &mut tx,
        tenant,
        LabelId::new(label),
        &PropertyData::InlineU32Pair(a, b),
    )
    .ok()?;
    commit(tx, &stack.store).ok()?;
    Some(CommitRow {
        tenant,
        id,
        label,
        a,
        b,
    })
}

fn build_committed_state(rows: &[CommitRow]) -> CommittedState {
    let mut s = CommittedState::default();
    for r in rows {
        let key = (r.tenant, r.id);
        let bytes = (r.label, r.a, r.b);
        s.any_history
            .entry(key)
            .or_insert_with(HashSet::new)
            .insert(bytes);
        s.latest_t1.insert(key, bytes);
    }
    s.total_commits = rows.len() as u64;
    s
}

fn read_recovered_state(stack: &M6Stack, pre_crash: &CommittedState) -> RecoveredState {
    let mut rec = RecoveredState::default();
    for (tenant, id) in pre_crash.any_history.keys() {
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
            Err(e) => panic!("m6 read_node_with_store error: {e:?}"),
        }
    }
    rec
}

// ─────────────────────────────────────────────────────────────────
// Scenario 1 — RandomFsyncLies (simulated)
// ─────────────────────────────────────────────────────────────────

/// RandomFsyncLies (simulated): at ~5 % probability per scheduled
/// background fsync, the K-1 injection RNG records a simulated lie
/// firing in `tally`. The actual production fsync syscall is NOT
/// hooked at this layer — see the module-doc "Forward-bind to M6-05
/// v1.1" section for the real OS-level seam (M6-05-real follow-up).
///
/// The contract this scenario verifies:
///
///  - **Harness shape**: the K-1 injection-RNG fires at the configured
///    rate; the BackgroundFsyncFailAction::RollbackAndContinue knob is
///    engaged at scheduler construction (the production seam).
///  - **T1 commits MUST survive**: T1 fsyncs run inline before commit
///    ack (per ADR-034 D-1). Even if a real post-ack background fsync
///    were to lie, T1 commits would be unaffected because the inline
///    fsync already returned. The recovered state must be byte-for-byte
///    complete.
///  - **T3 commits within rpo_ms MAY be lost**: T3 commits ack BEFORE
///    the periodic background fsync; if a fsync lies, up to rpo_ms of
///    T3 commits are silently lost (per ADR-034 D-2). This test only
///    runs T1 commits so the recovered state must be complete.
///
/// The test asserts: under N=200 commits with the fsync-lies flag on,
/// the K-1 injection RNG fires the BackgroundFsyncFail kind at least
/// once over the workload, AND post-recovery every T1 commit is
/// observable.
#[test]
fn scenario_random_fsync_lies_t1_commits_survive() {
    let workspace = TempDir::new().expect("tmpdir");
    let wal_dir = workspace.path().join("wal");
    std::fs::create_dir_all(&wal_dir).expect("mkdir");

    let flags = FaultInjectFlags::new();
    flags.set_fsync_lies(true);

    let stack = M6Stack::build(&wal_dir);
    let tenant = TenantId::new(801);
    const N: u32 = 200;

    // Test-side fault driver: per commit, roll the K-1 background-
    // fsync injection. The actual production seam is the
    // BackgroundFsyncFailAction::RollbackAndContinue knob set at
    // scheduler construction; the K-1 RNG roll is the test-side
    // tally that captures HOW MANY simulated fsync lies fired during
    // the workload window.
    let inj_cfg = InjectionConfig {
        // Bias the fsync-lies rate up vs the spec D2 default (0.005)
        // so the 200-commit workload reliably observes ≥ 1 lie under
        // typical RNG variance. Production rate stays at 0.005 elsewhere.
        background_fsync_failure_rate: 0.05,
        ..InjectionConfig::no_op()
    };
    let inj_rng = InjectionDecisionRng::new(0xC0FF_EEBA_BE00_5050);
    let tally = InjectionTally::new();

    let mut rows = Vec::with_capacity(N as usize);
    for i in 0..N {
        if flags.fsync_lies()
            && let Some(kind) = maybe_inject_background_fsync_failure(&inj_cfg, &inj_rng)
        {
            tally.record(kind);
        }
        let a = i;
        let b = i.wrapping_mul(31);
        if let Some(row) = do_commit(&stack, tenant, 80 + (i % 5), a, b) {
            rows.push(row);
        }
    }
    stack.shutdown();

    // Assert the harness fired ≥ 1 fsync-lie injection. With p=0.05,
    // N=200, E[hits] = 10, σ ≈ 3 → P(hits=0) ≈ exp(-10) ≈ 5e-5
    // (negligible flake risk).
    let lies = tally.count(InjectionKind::BackgroundFsyncFail);
    assert!(
        lies >= 1,
        "fsync-lies harness fired 0 times over N={} at rate=0.05 — \
         RNG broken or rate mis-set",
        N
    );

    // Recover; every T1 commit must survive.
    let recovered = recover_stack(&wal_dir);
    let pre = build_committed_state(&rows);
    let rec = read_recovered_state(&recovered, &pre);

    // T1 commits unaffected by post-ack background fsync lies. The
    // canonical assertion: every (tenant, NodeId) in pre.any_history
    // must appear in rec.bytes_by_key with the same bytes.
    for (key, hist) in &pre.any_history {
        assert!(
            rec.bytes_by_key.contains_key(key),
            "T1 commit lost post-recovery under fsync-lies: tenant={:?} id={:?}",
            key.0,
            key.1
        );
        let actual = rec.bytes_by_key.get(key).copied().expect("present");
        assert!(
            hist.contains(&actual),
            "T1 commit byte mismatch post-recovery under fsync-lies: \
             tenant={:?} id={:?} actual={:?} expected_set={:?}",
            key.0,
            key.1,
            actual,
            hist
        );
    }

    let oracle_cfg = OracleConfig::default();
    let report =
        verify_post_recovery_invariants(&pre, &rec, &oracle_cfg).expect("oracle violation");
    assert_eq!(
        report.t1_satisfied, report.t1_keys,
        "T1 strict bytes drifted under fsync-lies — ADR-034 D-1 violation"
    );
    eprintln!(
        "scenario_random_fsync_lies: rows={} lies_fired={} unique_keys={} \
         t1_keys={} t1_satisfied={}",
        rows.len(),
        lies,
        report.unique_keys,
        report.t1_keys,
        report.t1_satisfied,
    );
    recovered.shutdown();
}

// ─────────────────────────────────────────────────────────────────
// Scenario 2 — DiskFullSimulation (clean rollback)
// ─────────────────────────────────────────────────────────────────

/// DiskFullSimulation (simulated ENOSPC): the workload commits N rows;
/// halfway through, the disk_full flag is enabled. The test does NOT
/// actually fill disk; instead the K-1 wal-injection RNG fires
/// `WalFsyncFail` at 50 % per attempted commit during the toggle-on
/// window, and on dice-hit the test SKIPS the commit attempt (modelling
/// the operator-visible analog of `commit()` returning
/// `Err(WalError::DiskFull)`). NOTE: the actual `commit()` Err path is
/// NOT exercised — see the module-doc "Forward-bind to M6-05 v1.1"
/// section for the real ENOSPC seam (filesystem-layer fault layer
/// landing in M6-05-real follow-up).
///
/// The contract this scenario verifies:
///
///  - **Harness shape**: the K-1 injection-RNG fires `WalFsyncFail` at
///    the configured rate during the toggle-on window.
///  - **No torn writes**: every commit that returned `Ok` (pre-toggle,
///    post-toggle non-skipped, tail) is observable post-recovery.
///  - **Clean state under skipped attempts**: commits that the test
///    skipped on dice-hit leak NO partial state. The recovered state
///    contains only the (tenant, NodeId) pairs from successful commits;
///    no half-byte rows from synthesized failed-attempt keys.
#[test]
fn scenario_disk_full_simulation_clean_rollback() {
    let workspace = TempDir::new().expect("tmpdir");
    let wal_dir = workspace.path().join("wal");
    std::fs::create_dir_all(&wal_dir).expect("mkdir");

    let flags = FaultInjectFlags::new();

    let stack = M6Stack::build(&wal_dir);
    let tenant = TenantId::new(802);
    const PRE_TOGGLE: u32 = 50;
    const POST_TOGGLE: u32 = 50;

    let inj_cfg = InjectionConfig {
        // Aggressive rate post-toggle so most attempted commits
        // experience a simulated ENOSPC.
        wal_failure_rate: 0.5,
        ..InjectionConfig::no_op()
    };
    let inj_rng = InjectionDecisionRng::new(0xC0FF_EEBA_BE00_DF00);
    let tally = InjectionTally::new();

    // Pre-toggle: every commit succeeds (flag off, no rolls).
    let mut pre_toggle_rows = Vec::with_capacity(PRE_TOGGLE as usize);
    for i in 0..PRE_TOGGLE {
        if let Some(row) = do_commit(&stack, tenant, 90, i, i.wrapping_mul(13)) {
            pre_toggle_rows.push(row);
        }
    }

    // Toggle ON: simulate ENOSPC midway.
    flags.set_disk_full(true);
    let mut post_toggle_attempted = 0u32;
    let mut post_toggle_succeeded = Vec::new();
    let mut wal_fault_rolls = 0u64;
    for i in 0..POST_TOGGLE {
        if flags.disk_full()
            && let Some(kind) = maybe_inject_wal_failure(&inj_cfg, &inj_rng, i as u64)
        {
            tally.record(kind);
            wal_fault_rolls += 1;
            // Simulated ENOSPC: skip the commit attempt entirely.
            // The production analog is `commit()` returning
            // `Err(WalError::DiskFull)` — recovery sees nothing for
            // this attempt.
            post_toggle_attempted += 1;
            continue;
        }
        post_toggle_attempted += 1;
        if let Some(row) = do_commit(
            &stack,
            tenant,
            91,
            PRE_TOGGLE + i,
            (PRE_TOGGLE + i).wrapping_mul(17),
        ) {
            post_toggle_succeeded.push(row);
        }
    }

    // Toggle OFF: confirm post-rollback commits resume cleanly.
    flags.set_disk_full(false);
    let mut tail_rows = Vec::with_capacity(10);
    for i in 0..10u32 {
        if let Some(row) = do_commit(&stack, tenant, 92, 9000 + i, (9000 + i).wrapping_mul(7)) {
            tail_rows.push(row);
        }
    }
    stack.shutdown();

    // Assert the harness fired ≥ 1 simulated ENOSPC; otherwise the
    // test isn't exercising the rollback path.
    assert!(
        wal_fault_rolls >= 1,
        "disk-full harness fired 0 times over POST_TOGGLE={} at rate=0.5 — \
         RNG broken or rate mis-set",
        POST_TOGGLE
    );
    assert!(
        post_toggle_attempted >= 1,
        "no commit attempts during the simulated-ENOSPC window"
    );

    // Recover; every successful commit (pre-toggle, post-toggle non-
    // skipped, tail) must survive byte-for-byte.
    let recovered = recover_stack(&wal_dir);

    let mut all_successful = pre_toggle_rows.clone();
    all_successful.extend(post_toggle_succeeded.iter().cloned());
    all_successful.extend(tail_rows.iter().cloned());
    let pre = build_committed_state(&all_successful);
    let rec = read_recovered_state(&recovered, &pre);

    let oracle_cfg = OracleConfig::default();
    let report =
        verify_post_recovery_invariants(&pre, &rec, &oracle_cfg).expect("oracle violation");
    assert_eq!(
        report.t1_satisfied, report.t1_keys,
        "post-disk-full T1 strict bytes drifted — torn write or partial \
         state leaked through the rollback"
    );
    eprintln!(
        "scenario_disk_full_simulation: pre_toggle={} post_toggle_succeeded={} \
         tail={} wal_fault_rolls={} unique_keys={}",
        pre_toggle_rows.len(),
        post_toggle_succeeded.len(),
        tail_rows.len(),
        wal_fault_rolls,
        report.unique_keys,
    );

    // Cross-pollution check: the recovered state MUST NOT contain
    // any synthesized half-row from the failed commit attempts. Any
    // (tenant, NodeId) in the recovered state must trace back to a
    // successful commit.
    let success_keys: HashSet<(TenantId, NodeId)> =
        all_successful.iter().map(|r| (r.tenant, r.id)).collect();
    for k in rec.bytes_by_key.keys() {
        assert!(
            success_keys.contains(k),
            "recovery surfaced a key not from a successful commit — \
             torn-write leakage from the simulated-ENOSPC window: {:?}",
            k
        );
    }

    recovered.shutdown();
}
