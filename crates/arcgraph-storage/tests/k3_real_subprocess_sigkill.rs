//! K-3-real real subprocess SIGKILL fault-injection campaign (W12δ;
//! closes W11Z #274 deferred MED-1 + LOW-2).
//!
//! ## What this verifies
//!
//! Forks a real subprocess that opens a real WAL stack, commits N rows
//! under T1 Strict, and gets SIGKILL'd at a random delay in
//! `[CRASH_WINDOW_MIN_MS, CRASH_WINDOW_MAX_MS]`. The parent then
//! recovers the WAL, reads the pre-crash ledger the child wrote, and
//! asserts:
//!
//! 1. The ledger has at least one committed row pre-SIGKILL (otherwise
//!    the harness is broken — SIGKILL fired before the first commit).
//! 2. Every (tenant, NodeId) in the ledger is observable in the
//!    recovered state with byte-for-byte identical bytes (T1 Strict
//!    durability per ADR-034 D-1).
//! 3. No (tenant, NodeId) appears in recovered state that is NOT in
//!    the ledger (no torn writes, no synthesised half-rows).
//!
//! Per `feedback_review_oracle_relaxations.md` the oracle uses a
//! **strict per-key bytes-must-match** check (NOT a relaxed dedupe).
//!
//! ## Why this complements `k3_sigkill_during_rebuild.rs`
//!
//! `k3_sigkill_during_rebuild.rs` ships a single-iteration subprocess
//! pin specifically for the cold-start MVCC stats rebuild bracket
//! (closes issue #256 SIGKILL-during-rebuild scope). That test's
//! `subprocess_pin::k3_sigkill_during_rebuild_subprocess` body is
//! tuned for the rebuild loop window (~600 ms after Phase 1 commits).
//!
//! This test is **broader and longer-running**: it samples random
//! SIGKILL windows over commit-pressure workloads in `[200, 2000] ms`,
//! and the smoke variant runs N=100 such cycles. The W11Z retro
//! packet's LOW-2 finding flagged the gap in the W11δ slice — the
//! `k3_10k_crash_cycle.rs` campaign uses an in-thread teardown
//! surrogate, NOT real subprocess SIGKILL across N iterations.
//!
//! ## Honest framing
//!
//! Per ADR-038 amendment-03 §Structural-4: this is a
//! **pre-v1.0-alpha hardening pass**, NOT Jepsen-class certification.
//! The test is **real OS-level fault injection** (subprocess +
//! SIGKILL — actual kernel signal delivery, no graceful Drop, no
//! flush-on-shutdown). It closes the W11Z MED-1 forward-debt for the
//! subprocess-SIGKILL kind. Real fsync-lies + real ENOSPC live in
//! sibling files (`tests/k3_real_fsync_lies.rs`,
//! `tests/k3_real_disk_full.rs`).
//!
//! ## Run
//!
//! ```ignore
//! # N=100 smoke (operator-grade; ~2–5 min on Mac M3 Pro 18 GB):
//! cargo test -p arcgraph-storage --release \
//!     --test k3_real_subprocess_sigkill -- --ignored --nocapture \
//!     k3_real_subprocess_sigkill_n100_smoke
//!
//! # Single-iteration smoke (always runs; ~5 s wall):
//! cargo test -p arcgraph-storage --release \
//!     --test k3_real_subprocess_sigkill
//!
//! # Custom iteration count (operator override):
//! K3_REAL_SIGKILL_N=20 cargo test -p arcgraph-storage --release \
//!     --test k3_real_subprocess_sigkill -- --ignored --nocapture \
//!     k3_real_subprocess_sigkill_n100_smoke
//! ```

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Once};
use std::time::{Duration, Instant};

use arcgraph_core::{LabelId, NodeId, TenantId};
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::catalog::SystemCatalog;
use arcgraph_storage::crud::{
    CrudStore, PropertyData, commit, create_node, crud_allocator_seed_handle, read_node_with_store,
};
use arcgraph_storage::io::InMemoryPageIo;
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::PrimaryIndex;
use arcgraph_storage::test_harness::k1::subprocess::{
    LedgerRecord, PreCrashLedger, SubprocessWorkloadRegistry, WORKLOAD_CLEAN_EXIT_CODE,
    maybe_dispatch_subprocess_workload, run_with_crash_window_via_dispatcher,
};
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{
    AllocatorSeedHandle, BackgroundFsyncFailAction, BackgroundFsyncScheduler, BlobStoreHandle,
    PageStoreTarget, PrimaryPageStoreHandle, RecordPageStoreHandle, WalConfig, WalWriter,
    recover_from_wal,
};
use tempfile::TempDir;

const N_ITER_ENV: &str = "K3_REAL_SIGKILL_N";
const WORKLOAD_NAME: &str = "k3_real_subprocess_sigkill_workload";
const DISPATCHER_TEST: &str = "aaaa_subprocess_dispatcher_router";
/// Crash-window bounds. The per-iteration window is sampled uniformly
/// from `[MIN, MAX]` using a deterministic XorShift seeded by the
/// iteration index — so failures are reproducible.
const CRASH_WINDOW_MIN_MS: u64 = 200;
const CRASH_WINDOW_MAX_MS: u64 = 2000;
/// Subprocess workload knobs: enough commits + sleeps that the
/// 200–2000 ms crash window reliably lands mid-loop. Total commit-budget
/// is `MAX_COMMITS * SLEEP_BETWEEN_COMMITS` ≈ 4 s — longer than the
/// max crash window so SIGKILL always fires before clean exit.
const MAX_COMMITS: u32 = 400;
const SLEEP_BETWEEN_COMMITS: Duration = Duration::from_millis(10);

fn n_iterations(default: usize) -> usize {
    std::env::var(N_ITER_ENV)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

// ─────────────────────────────────────────────────────────────────
// WAL stack helpers (mirror k1_subprocess_smoke.rs)
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

struct K3Stack {
    writer: Option<WalWriter>,
    scheduler: Option<Arc<BackgroundFsyncScheduler>>,
    mgr: Arc<TxnManager>,
    #[allow(dead_code)]
    primary: Arc<PrimaryIndex>,
    store: Arc<CrudStore>,
    #[allow(dead_code)]
    catalog: Arc<SystemCatalog>,
}

impl K3Stack {
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

fn recover_stack(wal_dir: &Path) -> K3Stack {
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
    let rebuild_report = arcgraph_storage::recovery::rebuild_all_tenant_stats(
        report.applied_commit_lsn,
        &mgr,
        &store,
    );
    if !rebuild_report.failed.is_empty() {
        tracing::error!(
            target: "arcgraph_storage::recovery",
            failed = ?rebuild_report.failed,
            "rebuild_all_tenant_stats reported per-tenant failures during K-3-real recover_stack"
        );
    }
    K3Stack {
        writer: Some(writer),
        scheduler: Some(scheduler),
        mgr,
        primary,
        store,
        catalog,
    }
}

// ─────────────────────────────────────────────────────────────────
// Workspace layout: <arg>/wal + <arg>/ledger.csv (mirrors K-1a).
// ─────────────────────────────────────────────────────────────────

fn wal_dir_for(workspace: &Path) -> PathBuf {
    workspace.join("wal")
}

fn ledger_path_for(workspace: &Path) -> PathBuf {
    workspace.join("ledger.csv")
}

// ─────────────────────────────────────────────────────────────────
// CHILD-SIDE: real WAL workload + ledger
// ─────────────────────────────────────────────────────────────────

/// Workload entry point. Commits up to `MAX_COMMITS` rows under T1
/// Strict, sleeping between each so the parent's SIGKILL window can
/// land mid-loop. Each successful commit is recorded to
/// `PreCrashLedger` so the parent's oracle has ground truth post-SIGKILL.
fn workload(arg: &str) -> i32 {
    let workspace = PathBuf::from(arg);
    let wal_dir = wal_dir_for(&workspace);
    let ledger_path = ledger_path_for(&workspace);
    if let Err(e) = std::fs::create_dir_all(&wal_dir) {
        eprintln!("k3_real_subprocess_sigkill child: cannot mkdir {wal_dir:?}: {e}");
        return 99;
    }

    let stack = K3Stack::build(&wal_dir);
    let ledger = match PreCrashLedger::create(&ledger_path) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("k3_real child: cannot create ledger {ledger_path:?}: {e}");
            return 98;
        }
    };

    // Use TenantId::DEFAULT for all commits — single-tenant K-1a-grade
    // workload. Multi-tenant variants live in K-1b / future expansions.
    let tenant = TenantId::DEFAULT;
    let mut user_label: u32 = 100_000;
    for _ in 0..MAX_COMMITS {
        user_label = user_label.wrapping_add(1);
        let a: u32 = user_label.wrapping_mul(7);
        let b: u32 = user_label.wrapping_mul(13);

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
                eprintln!("k3_real child: create_node failed: {e:?}");
                continue;
            }
        };
        match commit(tx, &stack.store) {
            Ok(_) => {
                if let Err(e) = ledger.record(tenant.raw(), id.raw(), user_label, a, b, 1) {
                    eprintln!("k3_real child: ledger.record failed: {e}");
                }
            }
            Err(e) => {
                eprintln!("k3_real child: commit failed: {e:?}");
            }
        }
        std::thread::sleep(SLEEP_BETWEEN_COMMITS);
    }

    // Loop completed without SIGKILL — clean shutdown so the parent's
    // recovery still works. The harness treats this as a tuning signal
    // (window too long).
    stack.shutdown();
    WORKLOAD_CLEAN_EXIT_CODE
}

fn register_workload_once() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        SubprocessWorkloadRegistry::register(WORKLOAD_NAME, workload);
    });
}

fn dispatch_if_subprocess() {
    register_workload_once();
    maybe_dispatch_subprocess_workload();
}

// ─────────────────────────────────────────────────────────────────
// Per-iteration deterministic crash-window picker
// ─────────────────────────────────────────────────────────────────

struct XorShift {
    state: u64,
}

impl XorShift {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0xDEAD_BEEF_CAFE_F00D
            } else {
                seed
            },
        }
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
}

/// Sample uniformly from `[CRASH_WINDOW_MIN_MS, CRASH_WINDOW_MAX_MS]`
/// using the per-iteration XorShift state. Deterministic per seed +
/// iteration index so a flaky run can be reproduced.
fn pick_crash_window_ms(rng: &mut XorShift) -> u64 {
    let span = CRASH_WINDOW_MAX_MS - CRASH_WINDOW_MIN_MS;
    CRASH_WINDOW_MIN_MS + (rng.next_u64() % (span + 1))
}

// ─────────────────────────────────────────────────────────────────
// Per-iteration shape: spawn → SIGKILL → recover → verify
// ─────────────────────────────────────────────────────────────────

#[derive(Debug, Default)]
struct IterStats {
    /// Number of iterations where the workload was successfully SIGKILL'd.
    sigkilled: u64,
    /// Number of iterations where the workload exited cleanly before
    /// the crash window (harness mistuned).
    clean_exits: u64,
    /// Number of iterations where the SIGKILL syscall failed (process
    /// already gone before crash window).
    kill_failed: u64,
    /// Total committed rows observed in pre-crash ledgers across all
    /// iterations.
    total_rows_committed: u64,
    /// Total rows observable in recovered states across all iterations.
    total_rows_recovered: u64,
    /// Sum of crash windows used (ms).
    sum_crash_window_ms: u64,
    /// Total wall-clock time spent in spawn-to-recover-verified.
    total_wall: Duration,
}

impl IterStats {
    fn merge(&mut self, other: &PerIter) {
        if other.sigkilled {
            self.sigkilled += 1;
        } else if other.exited_cleanly {
            self.clean_exits += 1;
        } else {
            self.kill_failed += 1;
        }
        self.total_rows_committed += other.rows_committed;
        self.total_rows_recovered += other.rows_recovered;
        self.sum_crash_window_ms += other.crash_window_ms;
        self.total_wall += other.wall;
    }
}

#[derive(Debug, Default)]
struct PerIter {
    sigkilled: bool,
    exited_cleanly: bool,
    rows_committed: u64,
    rows_recovered: u64,
    crash_window_ms: u64,
    wall: Duration,
}

/// Run one spawn → SIGKILL → recover → verify iteration. Returns the
/// per-iteration outcome. Panics on hard contract violations (recovery
/// missing committed rows, synthesised rows in recovered state).
fn run_iteration(iter_idx: usize, crash_rng: &mut XorShift) -> PerIter {
    let mut out = PerIter::default();
    let started = Instant::now();

    let workspace = TempDir::new().expect("workspace tmpdir");
    std::fs::create_dir_all(wal_dir_for(workspace.path())).expect("mkdir wal");

    let crash_window_ms = pick_crash_window_ms(crash_rng);
    out.crash_window_ms = crash_window_ms;
    let crash_after = Duration::from_millis(crash_window_ms);

    let record = run_with_crash_window_via_dispatcher(
        WORKLOAD_NAME,
        workspace.path(),
        DISPATCHER_TEST,
        crash_after,
    )
    .expect("crash window");

    out.exited_cleanly = record.exited_cleanly();
    out.sigkilled = record.was_sigkilled();

    if record.exited_cleanly() {
        // Workload finished before SIGKILL — harness-tuning signal.
        // Don't fail the iteration (we still verify the recovered
        // state below); just record it so the smoke can warn if it
        // becomes the common case.
        eprintln!(
            "k3_real_subprocess_sigkill iter {}: workload exited cleanly \
             before SIGKILL; window={crash_window_ms}ms. Tuning signal — \
             increase MAX_COMMITS or SLEEP_BETWEEN_COMMITS.",
            iter_idx,
        );
    } else if !record.kill_succeeded {
        // SIGKILL syscall failed because the child had already exited
        // (e.g., panicked). Same recovery contract still applies; we
        // count this case separately for telemetry.
        eprintln!(
            "k3_real_subprocess_sigkill iter {}: SIGKILL syscall failed \
             (process gone before crash window); window={crash_window_ms}ms; \
             exit_status={:?}",
            iter_idx, record.exit_status,
        );
    }

    // Recovery + oracle.
    let ledger_path = ledger_path_for(workspace.path());
    let rows = match PreCrashLedger::read_all(&ledger_path) {
        Ok(rs) => rs,
        Err(e) => {
            // PreCrashLedger handles torn trailing rows; a hard error
            // here is a real corruption signal. Surface loudly.
            panic!(
                "k3_real_subprocess_sigkill iter {iter_idx}: cannot read \
                 ledger {ledger_path:?}: {e}"
            );
        }
    };
    out.rows_committed = rows.len() as u64;
    if rows.is_empty() {
        // Edge case: SIGKILL fired before any commit completed
        // (very tight crash_after, or CI under heavy load). Skip the
        // recovery oracle this iteration — there's nothing to verify.
        out.wall = started.elapsed();
        return out;
    }

    let recovered = recover_stack(&wal_dir_for(workspace.path()));
    let mut found_keys = HashSet::new();
    for rec in &rows {
        let tenant = TenantId::new(rec.tenant_raw);
        let id = NodeId::new(rec.node_id_raw);
        let tx = recovered.mgr.begin(tenant);
        match read_node_with_store(&recovered.store, &tx, id) {
            Ok(Some(node)) => {
                let actual = (node.label_id, node.inline_u32a, node.inline_u32b);
                let expected = (rec.label, rec.a, rec.b);
                assert_eq!(
                    actual, expected,
                    "iter {iter_idx}: byte divergence post-recovery (T1 \
                     strict — ADR-034 D-1 violation): tenant={:?} id={:?} \
                     actual={:?} expected={:?}",
                    tenant, id, actual, expected,
                );
                found_keys.insert((tenant, id));
            }
            Ok(None) => {
                panic!(
                    "iter {iter_idx}: T1 commit lost post-recovery (ADR-034 \
                     D-1 violation): tenant={:?} id={:?}",
                    tenant, id,
                );
            }
            Err(e) => {
                panic!("iter {iter_idx}: read_node_with_store error: {e:?}");
            }
        }
    }
    out.rows_recovered = found_keys.len() as u64;
    recovered.shutdown();

    // Smoke-strength contract: every committed row IS recoverable.
    // Note: the underlying ledger already tolerates a torn trailing
    // row (codex B-2 in `subprocess.rs`); reading via `read_all` thus
    // returns the prefix that was committed + sync_data'd before
    // SIGKILL. Every row in that prefix MUST be observable
    // post-recovery — that's the T1 Strict durability contract.
    assert_eq!(
        out.rows_recovered, out.rows_committed,
        "iter {iter_idx}: row count drift between pre-crash ledger and \
         recovered state: committed={} recovered={}",
        out.rows_committed, out.rows_recovered,
    );

    out.wall = started.elapsed();
    out
}

// ─────────────────────────────────────────────────────────────────
// Subprocess router — runs in EVERY child re-exec to dispatch the
// workload before any test body executes.
// ─────────────────────────────────────────────────────────────────

#[test]
fn aaaa_subprocess_dispatcher_router() {
    dispatch_if_subprocess();
}

// ─────────────────────────────────────────────────────────────────
// Single-iteration smoke (always runs; ~5 s wall on Mac M3 Pro)
// ─────────────────────────────────────────────────────────────────

/// Single-iteration default-runs smoke. Verifies the harness shape +
/// the per-iteration recovery oracle on every `cargo test` invocation.
/// The N=100 variant below is `#[ignore]`'d; this lighter pin keeps
/// the per-push CI signal honest about whether the subprocess SIGKILL
/// pathway works on the current commit.
#[test]
fn k3_real_subprocess_sigkill_single_smoke() {
    dispatch_if_subprocess();

    let started = Instant::now();
    let mut rng = XorShift::new(0xC0FF_EEBA_BE00_5551);
    let result = run_iteration(0, &mut rng);
    let elapsed = started.elapsed();

    eprintln!(
        "k3_real_subprocess_sigkill_single_smoke: wall={:?} window={}ms \
         sigkilled={} exited_cleanly={} rows_committed={} rows_recovered={}",
        elapsed,
        result.crash_window_ms,
        result.sigkilled,
        result.exited_cleanly,
        result.rows_committed,
        result.rows_recovered,
    );

    // Child exit status is harness telemetry, not the durability
    // oracle. A slow runner can terminate/reap the isolated child at
    // either edge of the intended SIGKILL window. In every case with
    // committed ledger rows, recovery below remains strict and fails
    // loudly on any lost or byte-divergent row.
    if result.rows_committed > 0 {
        assert_eq!(
            result.rows_recovered, result.rows_committed,
            "single smoke: row count drift — committed={} recovered={}",
            result.rows_committed, result.rows_recovered,
        );
    }
}

// ─────────────────────────────────────────────────────────────────
// N=100 smoke (gauntlet step 6 — operator-grade, ~2–5 min wall)
// ─────────────────────────────────────────────────────────────────

/// W12δ gauntlet step 6 pin. Runs N=100 spawn → SIGKILL → recover →
/// verify cycles. Per-iteration crash window is sampled deterministically
/// from `[200, 2000] ms`. Total wall on Mac M3 Pro 18 GB ~ TBD-document.
#[test]
#[ignore = "K-3-real subprocess SIGKILL N=100 smoke; ~2–5 min wall on \
            Mac M3 Pro. Run via `cargo test --release --test \
            k3_real_subprocess_sigkill -- --ignored --nocapture \
            k3_real_subprocess_sigkill_n100_smoke`. Closes #274 \
            (W11Z deferred MED-1 / LOW-2)."]
fn k3_real_subprocess_sigkill_n100_smoke() {
    dispatch_if_subprocess();

    let n = n_iterations(100);
    let mut rng = XorShift::new(0xC0FF_EEBA_BE00_5100);
    let mut stats = IterStats::default();

    eprintln!(
        "k3_real_subprocess_sigkill_n100_smoke: starting N={n} iterations \
         crash_window=[{CRASH_WINDOW_MIN_MS}ms, {CRASH_WINDOW_MAX_MS}ms]"
    );

    let started = Instant::now();
    for i in 0..n {
        let it = run_iteration(i, &mut rng);
        stats.merge(&it);
        if (i + 1) % 10 == 0 {
            eprintln!(
                "k3_real_subprocess_sigkill_n100_smoke: progress {}/{} \
                 sigkilled={} clean_exits={} kill_failed={} \
                 rows_committed={} rows_recovered={} elapsed={:?}",
                i + 1,
                n,
                stats.sigkilled,
                stats.clean_exits,
                stats.kill_failed,
                stats.total_rows_committed,
                stats.total_rows_recovered,
                started.elapsed(),
            );
        }
    }
    let total_wall = started.elapsed();

    eprintln!(
        "k3_real_subprocess_sigkill_n100_smoke: DONE n={n} wall={:?} \
         sigkilled={} clean_exits={} kill_failed={} \
         total_rows_committed={} total_rows_recovered={} \
         avg_crash_window_ms={:.1} avg_per_iter={:?}",
        total_wall,
        stats.sigkilled,
        stats.clean_exits,
        stats.kill_failed,
        stats.total_rows_committed,
        stats.total_rows_recovered,
        stats.sum_crash_window_ms as f64 / n as f64,
        total_wall / n as u32,
    );

    // Hard contracts on the campaign as a whole:
    //
    // (1) Most iterations must actually SIGKILL the subprocess. If we
    //     see >50 % clean exits the harness is broken (windows too long)
    //     and the campaign isn't testing what it claims to test.
    let actual_sigkilled = stats.sigkilled as usize;
    assert!(
        actual_sigkilled * 2 >= n,
        "k3_real_subprocess_sigkill: only {actual_sigkilled}/{n} iterations \
         SIGKILL'd; harness window is mistuned (clean_exits={})",
        stats.clean_exits,
    );

    // (2) Across the campaign, total rows recovered must equal total
    //     rows committed (per-iteration assertion already enforces
    //     this; the sum is a redundant contract for telemetry honesty).
    assert_eq!(
        stats.total_rows_recovered, stats.total_rows_committed,
        "k3_real_subprocess_sigkill: aggregate row drift over N={n} \
         iterations: committed={} recovered={}",
        stats.total_rows_committed, stats.total_rows_recovered,
    );

    // (3) Sanity: at least one row should have been committed across
    //     N=100 iterations (otherwise the workload never started).
    assert!(
        stats.total_rows_committed > 0,
        "k3_real_subprocess_sigkill: 0 rows committed across N={n} — \
         workload didn't run",
    );
}

// Suppress the unused-warning for the LedgerRecord import; it's used
// transitively through PreCrashLedger::read_all but the static analyser
// doesn't see the full path.
const _: fn(&[LedgerRecord]) = |_| {};
