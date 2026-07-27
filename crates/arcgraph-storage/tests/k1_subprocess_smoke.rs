//! K-1a subprocess SIGKILL WAL workload smoke (gated `K1_SUBPROCESS_SMOKE=1`).
//!
//! Per codex M3 retro Finding HIGH-1: the K-1 30 s smoke openly admits
//! it uses "in-thread WAL teardown / restart cycle as the recovery
//! proxy" — exactly the gap the SIGKILL subprocess harness was built
//! to close. The two pre-existing subprocess workloads (`infinite_loop_
//! workload` + `clean_exit_workload`) just `thread::sleep` — no WAL
//! writes, no recovery state. ~565 LOC of subprocess.rs scaffolding
//! shipped in PR #176 with NOTHING exercising the harness end-to-end.
//!
//! This test closes the gap: the child opens a real WAL stack, commits
//! N rows under [`DurabilityTier::Strict`] to a real WAL, records each
//! to [`PreCrashLedger`], gets SIGKILL'd by the parent's crash-window
//! timer, and is restarted by the parent which feeds the
//! `PreCrashLedger::read_all(...)` output to
//! [`verify_post_recovery_invariants`] against a freshly-recovered
//! WAL stack.
//!
//! ## Why gated `K1_SUBPROCESS_SMOKE=1`
//!
//! Wall time is ~5 s per run (workload commits ~50 rows over 2 s →
//! SIGKILL at the 2 s crash window → recovery + oracle ~1 s). The
//! `cargo test` default invocation is single-threaded per test file
//! and the cost adds up across CI runs; gating keeps it opt-in.
//! A future K-1c row in the roadmap wires a 1-hour cron campaign
//! that drives this end-to-end at scale.
//!
//! ## Run
//!
//! ```ignore
//! K1_SUBPROCESS_SMOKE=1 cargo test -p arcgraph-storage --release \
//!   --test k1_subprocess_smoke -- --ignored --nocapture
//! ```
//!
//! ## What this test does NOT cover
//!
//! - Multi-tenant SIGKILL workloads (single tenant `TenantId::DEFAULT`
//!   in K-1a; multi-tenant lands in K-1b).
//! - Snapshot install crash points (K-1c+d).
//! - 1-hour campaign + commodity NVMe + artifact archival (K-1c).
//! - Encoding-mismatch I-V coverage (K-1d).
//!
//! ## Stats consistency
//!
//! The oracle's stats-consistency check is set non-fatal here for the
//! same reason as `k1_smoke_30s` — WAL recovery does not rebuild
//! per-tenant `CatalogStats` (the increment hooks live in
//! `crud::commit`, not in `ReplayExecutor`). M4-41 follow-up flips
//! this to fatal. See `test_harness::k1::oracle` invariant 5.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Once};
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
use arcgraph_storage::test_harness::k1::oracle::{
    CommittedState, CommittedStatsRebuild, OracleConfig, RecoveredState, snapshot_catalog_stats,
    verify_post_recovery_invariants,
};
use arcgraph_storage::test_harness::k1::subprocess::{
    LedgerRecord, PreCrashLedger, SubprocessWorkloadRegistry, WORKLOAD_CLEAN_EXIT_CODE,
    maybe_dispatch_subprocess_workload, run_with_crash_window,
};
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{
    AllocatorSeedHandle, BackgroundFsyncFailAction, BackgroundFsyncScheduler, BlobStoreHandle,
    PageStoreTarget, PrimaryPageStoreHandle, RecordPageStoreHandle, WalConfig, WalWriter,
    recover_from_wal,
};
use std::collections::{HashMap, HashSet};
use tempfile::TempDir;

const SUBPROCESS_SMOKE_ENV: &str = "K1_SUBPROCESS_SMOKE";
const WORKLOAD_NAME: &str = "k1_wal_commit_workload";

fn subprocess_smoke_enabled() -> bool {
    std::env::var(SUBPROCESS_SMOKE_ENV)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

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

struct K1Stack {
    writer: Option<WalWriter>,
    scheduler: Option<Arc<BackgroundFsyncScheduler>>,
    mgr: Arc<TxnManager>,
    #[allow(dead_code)]
    primary: Arc<PrimaryIndex>,
    store: Arc<CrudStore>,
    catalog: Arc<SystemCatalog>,
}

impl K1Stack {
    fn build(wal_dir: &Path) -> Self {
        let writer = WalWriter::spawn(test_wal_config(wal_dir.to_path_buf())).unwrap();
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

fn recover_stack(wal_dir: &Path) -> K1Stack {
    let writer = WalWriter::spawn(test_wal_config(wal_dir.to_path_buf())).unwrap();
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
    let report = recover_from_wal(wal_dir, Arc::clone(&mgr), target, None).unwrap();

    // M4-41 cold-start MVCC stats rebuild (per ADR-038 amendment-06
    // §D-25.1). The subprocess variant exercises the full
    // process-restart boundary — no shared memory survives across the
    // SIGKILL boundary, so the rebuild path's "from MVCC at recovered
    // LSN" reconstruction is the strictest test of the contract. See
    // `tests/k1_smoke_30s.rs::recover_stack` for the full rationale.
    let rebuild_report = arcgraph_storage::recovery::rebuild_all_tenant_stats(
        report.applied_commit_lsn,
        &mgr,
        &store,
    );
    if !rebuild_report.failed.is_empty() {
        eprintln!(
            "k1_subprocess_smoke M4-41 rebuild: {} tenant(s) marked recovery_failed: {:?}",
            rebuild_report.failed.len(),
            rebuild_report.failed,
        );
    }

    K1Stack {
        writer: Some(writer),
        scheduler: Some(scheduler),
        mgr,
        primary,
        store,
        catalog,
    }
}

// ─────────────────────────────────────────────────────────────────
// Workspace layout: arg = TempDir; <arg>/wal + <arg>/ledger.csv
// ─────────────────────────────────────────────────────────────────

fn wal_dir_for(workspace: &Path) -> PathBuf {
    workspace.join("wal")
}

fn ledger_path_for(workspace: &Path) -> PathBuf {
    workspace.join("ledger.csv")
}

// ─────────────────────────────────────────────────────────────────
// CHILD-SIDE: real WAL workload
// ─────────────────────────────────────────────────────────────────

/// Real WAL workload: commits N rows under [`DurabilityTier::Strict`]
/// to a real WAL + records each to [`PreCrashLedger`]. Sleeps briefly
/// between commits so the parent's crash-window timer can fire mid-loop
/// (the typical case — the test wants SIGKILL during commit pressure,
/// not at idle).
///
/// Returns [`WORKLOAD_CLEAN_EXIT_CODE`] if the loop runs to completion
/// (which happens only if the parent's crash window is longer than
/// `MAX_COMMITS * SLEEP_PER_COMMIT`; the test tunes the window so this
/// is a harness-regression signal, not a normal outcome).
fn wal_commit_workload(arg: &str) -> i32 {
    const MAX_COMMITS: u32 = 200;
    const SLEEP_BETWEEN_COMMITS: Duration = Duration::from_millis(10);

    let workspace = PathBuf::from(arg);
    let wal_dir = wal_dir_for(&workspace);
    let ledger_path = ledger_path_for(&workspace);
    if let Err(e) = std::fs::create_dir_all(&wal_dir) {
        eprintln!("k1_subprocess child: cannot mkdir {wal_dir:?}: {e}");
        return 99;
    }

    let stack = K1Stack::build(&wal_dir);
    let ledger = match PreCrashLedger::create(&ledger_path) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("k1_subprocess child: cannot create ledger {ledger_path:?}: {e}");
            return 98;
        }
    };

    let tenant = TenantId::DEFAULT;
    let mut user_label: u32 = 100_000;
    for _i in 0..MAX_COMMITS {
        user_label = user_label.wrapping_add(1);
        let a: u32 = user_label.wrapping_mul(7);
        let b: u32 = user_label.wrapping_mul(13);

        // One transaction → one node create → one commit. T1 Strict
        // (default tenant tier) so the WAL fsyncs on every commit.
        let mut tx = stack.mgr.begin(tenant);
        let id = match create_node(
            &stack.store,
            &mut tx,
            tenant,
            LabelId::new(user_label),
            &PropertyData::InlineU32Pair(a, b),
        ) {
            Ok(id) => id,
            Err(e) => {
                eprintln!("k1_subprocess child: create_node failed: {e:?}");
                continue;
            }
        };
        match commit(tx, &stack.store) {
            Ok(_) => {
                // Tier 1 = Strict per oracle::tier_to_u8.
                if let Err(e) = ledger.record(tenant.raw(), id.raw(), user_label, a, b, 1) {
                    eprintln!("k1_subprocess child: ledger.record failed: {e}");
                }
            }
            Err(e) => {
                eprintln!("k1_subprocess child: commit failed: {e:?}");
            }
        }
        std::thread::sleep(SLEEP_BETWEEN_COMMITS);
    }

    // Workload completed without SIGKILL — drop the stack cleanly so
    // a graceful test run still leaves the WAL in a recoverable
    // state. The parent should treat WORKLOAD_CLEAN_EXIT_CODE as a
    // harness-tuning signal (window too long).
    stack.shutdown();
    WORKLOAD_CLEAN_EXIT_CODE
}

// ─────────────────────────────────────────────────────────────────
// Test-side dispatcher (must run at the top of every #[test] in
// this file so the child subprocess routes to its workload).
// ─────────────────────────────────────────────────────────────────

fn register_workloads_once() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        SubprocessWorkloadRegistry::register(WORKLOAD_NAME, wal_commit_workload);
    });
}

fn dispatch_if_subprocess() {
    register_workloads_once();
    maybe_dispatch_subprocess_workload();
}

// ─────────────────────────────────────────────────────────────────
// PARENT-SIDE: assemble pre-crash + recovered states; run oracle
// ─────────────────────────────────────────────────────────────────

fn build_committed_state_from_ledger(rows: &[LedgerRecord]) -> CommittedState {
    let mut s = CommittedState::default();
    let mut stats: HashMap<TenantId, CommittedStatsRebuild> = HashMap::new();
    for rec in rows {
        let tenant = TenantId::new(rec.tenant_raw);
        let id = NodeId::new(rec.node_id_raw);
        let key = (tenant, id);
        let bytes = (rec.label, rec.a, rec.b);
        s.any_history
            .entry(key)
            .or_insert_with(HashSet::new)
            .insert(bytes);
        // tier == 1 ⇒ Strict per oracle::u8_to_tier.
        if rec.tier == 1 {
            s.latest_t1.insert(key, bytes);
        }
        let st = stats.entry(tenant).or_default();
        *st.label_counts.entry(LabelId::new(rec.label)).or_insert(0) += 1;
        st.total_nodes += 1;
        // commits_observed is set per-tenant below to match the M4-41
        // cold-start rebuild semantics (single coalesced observation
        // per tenant per amendment-06 §D-25.1 step 2).
    }
    for st in stats.values_mut() {
        st.commits_observed = 1;
    }
    s.total_commits = rows.len() as u64;
    s.stats_by_tenant = stats;
    s
}

fn build_recovered_state(
    stack: &K1Stack,
    committed: &CommittedState,
    labels: &[LabelId],
) -> RecoveredState {
    let mut rec = RecoveredState::default();
    let mut tenants_seen: HashSet<TenantId> = HashSet::new();
    for (tenant, id) in committed.any_history.keys() {
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
            Err(e) => panic!("k1_subprocess_smoke read_node_with_store error: {e:?}"),
        }
    }
    for tenant in tenants_seen {
        if let Some(stats) = stack.store.catalog_stats(tenant) {
            let snap = snapshot_catalog_stats(&stats, labels, &[]);
            rec.stats_by_tenant.insert(tenant, snap);
        }
    }
    let _ = stack.catalog;
    rec
}

// ─────────────────────────────────────────────────────────────────
// The smoke test
// ─────────────────────────────────────────────────────────────────

/// Subprocess router: this test exists ONLY so the child re-exec'd
/// test binary hits `dispatch_if_subprocess()` BEFORE the harness
/// decides which tests to run. The child is invoked via
/// `Command::new(current_exe)` in `build_workload_command`, which
/// inherits the parent's stdio but NOT `--ignored`. The
/// `k1_subprocess_smoke_wal_workload` test is `#[ignore]`'d so the
/// child harness would skip it and exit code 0, leaving the child
/// process never executing the workload.
///
/// Putting the dispatcher in a non-ignored test (which the child DOES
/// run) ensures the workload always dispatches: the dispatcher returns
/// in the parent (env var absent → no-op) but exits the process in the
/// child (env var set → run workload → process::exit).
#[test]
fn aaaa_subprocess_dispatcher_router() {
    dispatch_if_subprocess();
}

#[test]
#[ignore = "K-1a subprocess smoke; gated by K1_SUBPROCESS_SMOKE=1; ~5 s wall \
            (panics if neither K1_SUBPROCESS_SMOKE=1 nor \
            ARCGRAPH_SUBPROCESS_SMOKE_SKIP_OK=1 is set; see \
            feedback_test_env_gate_panic_by_default.md)"]
fn k1_subprocess_smoke_wal_workload() {
    // Top of test: child subprocesses re-exec into this binary, hit
    // the dispatcher, run wal_commit_workload, exit. Parent then
    // proceeds past this line.
    dispatch_if_subprocess();

    // Panic-by-default per `feedback_test_env_gate_panic_by_default.md`
    // (W12 retro INDEPENDENT REVIEW L1-MED-2[c] sibling soft-skip
    // sweep). Soft-skipping when K1_SUBPROCESS_SMOKE != 1 even after
    // an operator BYPASSED the `#[ignore]` gate via `--ignored` was
    // the W12δ HIGH-1 bug class — the test reported pass without
    // running the SIGKILL workload.
    //
    // Two opt-outs (specific to the test surface, NOT a generic
    // SKIP_OK so accidental opt-outs don't cascade):
    //
    //   * `K1_SUBPROCESS_SMOKE=1` — operator wants the smoke to run.
    //   * `ARCGRAPH_SUBPROCESS_SMOKE_SKIP_OK=1` — hostile-env opt-out
    //     (build-system testing, CI host where subprocess fork
    //     isn't available). Emits a clear "skipped (opt-in)" message
    //     rather than soft-skipping green.
    //
    // Absence of both → PANIC with a message naming the env-flag
    // escape hatches.
    let smoke_run = subprocess_smoke_enabled();
    let skip_ok = std::env::var("ARCGRAPH_SUBPROCESS_SMOKE_SKIP_OK").is_ok();
    if !smoke_run {
        if skip_ok {
            eprintln!(
                "k1_subprocess_smoke_wal_workload: SKIPPING (opt-in via \
                 ARCGRAPH_SUBPROCESS_SMOKE_SKIP_OK=1) — set \
                 {SUBPROCESS_SMOKE_ENV}=1 to run the smoke instead"
            );
            return;
        }
        panic!(
            "k1_subprocess_smoke_wal_workload: required env flag \
             {SUBPROCESS_SMOKE_ENV}=1 not set. This test is \
             `#[ignore]`'d to keep it off the default gauntlet; when \
             invoked via `--ignored`, {SUBPROCESS_SMOKE_ENV}=1 must \
             be set so the SIGKILL subprocess workload actually runs. \
             Set {SUBPROCESS_SMOKE_ENV}=1 to run, or \
             ARCGRAPH_SUBPROCESS_SMOKE_SKIP_OK=1 to opt into a \
             soft-skip (hostile envs only). Soft-skipping silently \
             after `--ignored` bypass is the W12δ HIGH-1 bug class \
             (`feedback_test_env_gate_panic_by_default.md`)."
        );
    }

    // ── Parent: spawn child + SIGKILL after crash window ──
    let workspace = TempDir::new().unwrap();
    // Pre-create WAL dir so child + parent see the same path layout.
    std::fs::create_dir_all(wal_dir_for(workspace.path())).unwrap();

    let crash_after = Duration::from_millis(800);
    let started = std::time::Instant::now();
    let record =
        run_with_crash_window(WORKLOAD_NAME, workspace.path(), crash_after).expect("crash window");
    let elapsed = started.elapsed();

    eprintln!(
        "k1_subprocess_smoke: spawn→reap elapsed={:?} elapsed_to_kill={:?} \
         kill_succeeded={} sigkilled={} exited_cleanly={} exit_status={:?}",
        elapsed,
        record.elapsed_to_kill,
        record.kill_succeeded,
        record.was_sigkilled(),
        record.exited_cleanly(),
        record.exit_status,
    );

    // (1) Workload must NOT have completed before SIGKILL — that
    // would mean the workload ran to MAX_COMMITS=200 in 800ms which
    // is a harness regression (or extreme machine speed). If this
    // fires, lower MAX_COMMITS or shorten crash_after.
    assert!(
        !record.exited_cleanly(),
        "workload completed before SIGKILL — crash window {crash_after:?} too long \
         OR MAX_COMMITS too low; harness needs tuning"
    );

    // (2) The kill must have landed (not failed because the process
    // had already exited). On Unix, the exit status must reflect a
    // signal.
    assert!(
        record.kill_succeeded,
        "SIGKILL syscall failed — child process gone before crash window"
    );
    #[cfg(unix)]
    assert!(
        record.was_sigkilled(),
        "child must report SIGKILL signal-exit on Unix; got {:?}",
        record.exit_status
    );

    // ── Parent: read pre-crash ledger ──
    let ledger_path = ledger_path_for(workspace.path());
    let rows = PreCrashLedger::read_all(&ledger_path).expect("read pre-crash ledger");
    eprintln!(
        "k1_subprocess_smoke: pre-crash ledger has {} committed rows post-SIGKILL",
        rows.len()
    );
    // (3) The workload must have committed SOMETHING before SIGKILL.
    // If the ledger is empty the harness is broken (workload didn't
    // start, or SIGKILL fired before the first commit).
    assert!(
        !rows.is_empty(),
        "pre-crash ledger is empty — workload didn't commit before SIGKILL"
    );

    // ── Parent: recover the WAL stack against a fresh in-process
    // stack rooted at the SAME wal_dir the child wrote into. The
    // pre-existing stack's in-memory pages are gone (PROCESS DIED);
    // recovery rebuilds everything from WAL replay alone — exactly
    // the contract ADR-031 §R3 (commit-bundle atomicity) +
    // ADR-034 D-1 (T1 strict durability) rest on. ──
    let recovered = recover_stack(&wal_dir_for(workspace.path()));

    // ── Build oracle inputs ──
    let pre = build_committed_state_from_ledger(&rows);
    let mut all_labels: Vec<LabelId> = pre
        .stats_by_tenant
        .values()
        .flat_map(|s| s.label_counts.keys().copied())
        .collect();
    all_labels.sort_by_key(|l| l.raw());
    all_labels.dedup();
    let rec = build_recovered_state(&recovered, &pre, &all_labels);

    // ── Run the oracle ──
    //
    // M4-41 cold-start MVCC stats rebuild (per ADR-038 amendment-06
    // §3 R1 acceptance criteria items 2 + 3): the subprocess variant
    // is the strictest oracle of the K-1a three because it exercises
    // the full process-restart boundary (no shared in-memory state
    // survives across the SIGKILL boundary). The cold-start rebuild
    // path is the only mechanism that can repopulate per-tenant
    // CatalogStats post-restart; the strict-mode oracle below
    // verifies that mechanism's correctness end-to-end.
    let oracle_cfg = OracleConfig::default();
    assert!(
        oracle_cfg.stats_inconsistency_fatal,
        "M4-41 R1 acceptance criterion (per ADR-038 amendment-06 §3 item 3): \
         this K-1 subprocess smoke MUST run with stats_inconsistency_fatal=true; \
         the cold-start rebuild path makes the strict oracle non-vacuous AT THE \
         strictest cross-process restart boundary."
    );
    let report = verify_post_recovery_invariants(&pre, &rec, &oracle_cfg)
        .expect("k1_subprocess_smoke oracle violation");

    eprintln!(
        "k1_subprocess_smoke: unique_keys={} t1_keys={} t1_satisfied={} \
         historical_match={} t3_rpo_lost={} stats_consistent={:?}",
        report.unique_keys,
        report.t1_keys,
        report.t1_satisfied,
        report.historical_match,
        report.t3_rpo_lost,
        report.stats_consistent_by_tenant,
    );

    // ── Hard contract checks ──

    // (4) Some commits were observed pre-crash AND every committed
    // row is observable post-recovery (T1 Strict durability per
    // ADR-034 D-1). Missing T1 commits would have been caught
    // by `T1Missing` above; here we double-check the report's
    // satisfied counts match.
    assert!(
        report.unique_keys > 0,
        "0 unique keys in pre-crash ledger — harness regression"
    );
    assert_eq!(
        report.t1_keys, report.unique_keys,
        "every workload commit was T1 Strict; t1_keys must equal unique_keys"
    );
    assert_eq!(
        report.t1_satisfied, report.t1_keys,
        "T1 strict bytes drifted post-recovery — ADR-034 D-1 violation"
    );

    // ── Cleanup ──
    recovered.shutdown();
}
